//! Record and replay of model calls, so an eval run is reproducible offline and a
//! prompt change can be compared on the same replies.
//!
//! A Tape stores each call's raw, pre-decode [`Reply`] under a [`key`] of everything
//! that shapes it, so replay re-runs decoding and the retry (which has its own key),
//! and the Meter reports real per-role tokens, cost and truncations offline. Errors
//! are never recorded, so a transient 5xx is never pinned.

use std::collections::BTreeMap;
use std::fs;
use std::future::Future;
use std::io;
use std::path::Path;
use std::sync::{Mutex, PoisonError};

use blake2::Blake2b;
use blake2::digest::consts::U16;

use super::{ModelError, Reply, Request};
use crate::config::ModelRoute;
use crate::identity::hash_hex;

/// Recorded replies by [`key`]. A live tape replays hits and calls the model only
/// on a miss; a replay-only tape fails a miss with [`ModelError::Unrecorded`], as a
/// live call would fail.
#[derive(Debug)]
pub struct Tape {
    live: bool,
    calls: Mutex<BTreeMap<String, Reply>>,
}

impl Tape {
    /// The tape at `path`. A missing file is an empty tape when `live`, else an
    /// error: replaying nothing would fail every call.
    pub fn load(path: &Path, live: bool) -> io::Result<Self> {
        let calls = match fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes)?,
            Err(e) if live && e.kind() == io::ErrorKind::NotFound => BTreeMap::new(),
            Err(e) => return Err(e),
        };
        Ok(Self {
            live,
            calls: Mutex::new(calls),
        })
    }

    /// Writes the tape to `path` through a temporary file, so an interrupted save
    /// leaves the previous tape intact.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let json = serde_json::to_vec_pretty(&*self.calls())?;
        let tmp = path.with_extension("tmp");
        fs::write(&tmp, json)?;
        fs::rename(&tmp, path)
    }

    /// The recording for `key`, else `live`'s reply, recorded when this tape is live.
    pub(crate) async fn play(
        &self,
        key: String,
        live: impl Future<Output = Result<Reply, ModelError>>,
    ) -> Result<Reply, ModelError> {
        if let Some(reply) = self.calls().get(&key) {
            return Ok(reply.clone());
        }
        if !self.live {
            return Err(ModelError::Unrecorded);
        }
        let reply = live.await?;
        self.calls().insert(key, reply.clone());
        Ok(reply)
    }

    fn calls(&self) -> std::sync::MutexGuard<'_, BTreeMap<String, Reply>> {
        self.calls.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Blake2b-128 over everything that shapes a reply: `provider/model`, system, user,
/// schema name and JSON, and `max_tokens`, each NUL-terminated. The model and budget
/// are in it, so another model misses loudly, and a raised budget re-records instead
/// of replaying stale truncations. Schema JSON prints with sorted
/// keys because serde_json's `preserve_order` is off (only tree-sitter's
/// build-dependency enables it, and resolver v2+ keeps that separate); a pinned-key
/// test catches a flip.
pub fn key(route: &ModelRoute, req: &Request<'_>) -> String {
    let (schema_name, schema_json) = req.schema.map_or((String::new(), String::new()), |s| {
        (s.name.to_owned(), s.json.to_string())
    });
    let parts = [
        format!("{}/{}", route.provider, route.model),
        req.system.to_owned(),
        req.user.to_owned(),
        schema_name,
        schema_json,
        req.max_tokens.to_string(),
    ]
    .map(|p| p + "\0");
    hash_hex::<Blake2b<U16>>(parts.iter().map(String::as_bytes))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use serde_json::json;

    use super::*;
    use crate::model::tests::{Script, Small, reply, route, router, small_schema, tally};
    use crate::model::{Meter, Schema, Stop};
    use crate::state::ModelRole;

    /// A per-test path in the temp dir, removed first so a stale run can't leak in.
    fn temp(name: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("diffagent-tape-{}-{name}.json", std::process::id()));
        let _ = fs::remove_file(&path);
        path
    }

    fn req<'a>(user: &'a str, schema: Option<&'a Schema>, max_tokens: u32) -> Request<'a> {
        Request {
            system: "sys",
            user,
            shared: 0,
            schema,
            max_tokens,
        }
    }

    #[test]
    fn key_covers_what_shapes_a_reply() {
        let schema = small_schema();
        let other_schema = Schema {
            name: "other",
            ..schema.clone()
        };
        let mut other_json = schema.clone();
        other_json.json["required"] = json!(["n"]);
        let base = key(&route("haiku-5-5"), &req("u", Some(&schema), 100));
        let changed = [
            ("model", key(&route("glm-5"), &req("u", Some(&schema), 100))),
            (
                "user",
                key(&route("haiku-5-5"), &req("v", Some(&schema), 100)),
            ),
            (
                "schema name",
                key(&route("haiku-5-5"), &req("u", Some(&other_schema), 100)),
            ),
            (
                "schema json",
                key(&route("haiku-5-5"), &req("u", Some(&other_json), 100)),
            ),
            ("no schema", key(&route("haiku-5-5"), &req("u", None, 100))),
            (
                "max_tokens",
                key(&route("haiku-5-5"), &req("u", Some(&schema), 200)),
            ),
        ];
        for (name, k) in changed {
            assert_ne!(k, base, "{name}");
        }
        let mut provider = route("haiku-5-5");
        provider.provider = "openrouter".to_owned();
        assert_ne!(
            key(&provider, &req("u", Some(&schema), 100)),
            base,
            "provider"
        );
        let system = Request {
            system: "other",
            ..req("u", Some(&schema), 100)
        };
        assert_ne!(key(&route("haiku-5-5"), &system), base, "system");
    }

    /// Pinned bytes: a change re-records every tape, so it must be deliberate. Also
    /// catches serde_json's `preserve_order` turning on, which reorders schema keys.
    /// Value from Python's `hashlib.blake2b(digest_size=16)` over the NUL-terminated
    /// parts, schema dumped with `sort_keys=True`.
    #[test]
    fn pinned_key() {
        let schema = Schema {
            name: "s",
            json: json!({"type": "object", "additionalProperties": false,
                         "required": [], "properties": {}}),
        };
        assert_eq!(
            key(&route("haiku-5-5"), &req("user", Some(&schema), 4000)),
            "603a417054ec86b65b020cc1394c839d"
        );
    }

    #[tokio::test]
    async fn record_save_load_replay() {
        let path = temp("record");
        let schema = small_schema();
        let valid = r#"{"n":5,"items":[],"note":null}"#;
        let invalid = r#"{"n":"five","items":[],"note":null}"#;
        let model = Script::new(&[
            (invalid, Stop::End),
            (valid, Stop::End),
            ("prose", Stop::MaxTokens),
        ]);
        let live = Arc::new(Tape::load(&path, true).unwrap());
        let r = router(model.clone()).with_tape(live.clone());
        let m = Meter::default();
        let first = req("review", Some(&schema), 100);
        let got: Small = r.json(ModelRole::Review, &first, &m).await.unwrap();
        assert_eq!(got.n, 5);
        let prose = r
            .text(ModelRole::Review, &req("p", None, 10), &m)
            .await
            .unwrap();
        assert_eq!(prose, reply("prose", Stop::MaxTokens));
        // A live tape replays its hits: the same calls again reach no model.
        let got: Small = r.json(ModelRole::Review, &first, &m).await.unwrap();
        assert_eq!((got.n, model.calls()), (5, 3));
        live.save(&path).unwrap();

        // Replay-only: the retry pair and the prose come back from the file, metered
        // with their recorded tokens, and the model is never called.
        let idle = Script::new(&[]);
        let tape = Arc::new(Tape::load(&path, false).unwrap());
        let r = router(idle.clone()).with_tape(tape);
        let m = Meter::default();
        let got: Small = r.json(ModelRole::Review, &first, &m).await.unwrap();
        assert_eq!(got.n, 5);
        let replayed = r
            .text(ModelRole::Review, &req("p", None, 10), &m)
            .await
            .unwrap();
        assert_eq!(replayed, prose);
        let t = tally(&r, &m);
        assert_eq!((t.calls, t.retried, t.truncated), (3, 1, 1));
        assert_eq!(t.tokens.input, 3000);
        assert_eq!(idle.calls(), 0);

        let miss = r.text(ModelRole::Review, &req("new", None, 10), &m).await;
        assert_eq!(miss, Err(ModelError::Unrecorded));
        let raised = r.text(ModelRole::Review, &req("p", None, 20), &m).await;
        assert_eq!(
            raised,
            Err(ModelError::Unrecorded),
            "a raised budget re-records"
        );
        let _ = fs::remove_file(&path);
    }

    #[tokio::test]
    async fn errors_are_never_recorded() {
        let path = temp("errors");
        let tape = Arc::new(Tape::load(&path, true).unwrap());
        let r = router(Script::new(&[])).with_tape(tape.clone());
        let got = r
            .text(ModelRole::Review, &req("u", None, 10), &Meter::default())
            .await;
        assert!(matches!(got, Err(ModelError::Request(_))), "{got:?}");
        tape.save(&path).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "{}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn missing_file() {
        let path = temp("missing");
        assert_eq!(
            Tape::load(&path, false).unwrap_err().kind(),
            io::ErrorKind::NotFound
        );
        assert!(Tape::load(&path, true).is_ok(), "live: starts empty");
    }
}
