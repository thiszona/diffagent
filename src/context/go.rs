//! Go context through the `diffagent-goctx` subprocess (`goctx/`). One run
//! per Review type-checks the reviewed Go files' packages and answers, per file,
//! with the definitions its visible lines use and the callers of what it changed.
//! The type-checker's memory lands in that process, never the worker's; this
//! client bounds how many run at once, for how long, and how much they print.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

use super::{ContextPack, Snippet};
use crate::diff::{DiffIndex, FileChange, FileStatus};
use crate::proc::{self, Cmd, ProcError};

/// The most files one run sends (goctx rejects more); the rest, in path order,
/// get `error: "go file cap"`.
const MAX_FILES: usize = 32;

/// What goctx may print: snippets for 32 files are far less.
const STDOUT_CAP: usize = 4 << 20;

/// How long past its deadline goctx gets: it stops loading at the deadline and
/// still answers with what it has.
const GRACE: Duration = Duration::from_secs(15);

/// The only caller variables goctx gets: where Go keeps its caches and temp
/// files. No token, so with `GOPROXY=off` it never reaches the network.
const PASSED: [&str; 4] = ["HOME", "GOCACHE", "GOMODCACHE", "GOTMPDIR"];

/// Runs `diffagent-goctx`. One per process: its semaphore bounds the type-checks
/// running at once, each a Review's memory spike, and [`GoCtx::with_modcache`]'s
/// per-Review clients share it.
pub struct GoCtx {
    bin: OsString,
    env: Vec<(String, OsString)>,
    permits: Arc<Semaphore>,
    grace: Duration,
}

/// One run's outcome.
#[derive(Debug, Default)]
pub struct GoPacks {
    /// One per reviewed Go file with a patch: its snippets, or why it has none.
    pub packs: BTreeMap<String, ContextPack>,
    pub report: GoReport,
}

/// What a run says beyond its packs, for the log: a
/// [`GoCtxError::BadRequest`] at error level, any other failure as a warning.
#[derive(Debug, Default)]
pub struct GoReport {
    /// goctx's toolchain, e.g. `go1.27.3`; empty when it didn't answer.
    pub go: String,
    /// goctx's load, type and scan errors, the root shown as `.`.
    pub errors: Vec<String>,
    /// Why the run failed as a whole; every file sent has this in its `error`.
    pub failure: Option<GoCtxError>,
    /// Wall time, the wait for a load slot included; zero when nothing ran.
    pub took: Duration,
}

/// Why a run gave nothing. A [`GoCtxError::BadRequest`] is a bug in this client,
/// worth an error log; the rest are warnings.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GoCtxError {
    /// Exit 2.
    #[error("diffagent-goctx rejected the request: {0}")]
    BadRequest(String),
    /// Exit 3: its heap passed the hard cap above `GOMEMLIMIT`, so it stopped itself
    /// before the cgroup's OOM killer would.
    #[error("diffagent-goctx: {0}")]
    Memory(String),
    /// The deadline passed waiting for a load slot.
    #[error("diffagent-goctx: no load slot before the deadline")]
    Busy,
    /// It couldn't start, timed out, failed or printed too much.
    #[error(transparent)]
    Proc(ProcError),
    /// What it printed wasn't the protocol's JSON.
    #[error("diffagent-goctx: unreadable output: {0}")]
    Output(String),
}

#[derive(Serialize)]
struct Request<'a> {
    root: &'a Path,
    deadline_ms: i64,
    files: Vec<FileRequest<'a>>,
}

#[derive(Serialize)]
struct FileRequest<'a> {
    path: &'a str,
    /// The new-file lines the model is shown.
    visible: Vec<i64>,
    changed: Vec<i64>,
}

#[derive(Deserialize)]
struct Response {
    go: String,
    errors: Vec<String>,
    files: Vec<FileResult>,
}

#[derive(Deserialize)]
struct FileResult {
    path: String,
    skipped: String,
    snippets: Vec<Snippet>,
}

impl GoCtx {
    /// `bin` is `GOCTX_PATH`, `memory_mb` its `GOMEMLIMIT`, `permits`
    /// `MAX_CONCURRENT_LOADS`. Of `env`, only `HOME`, `GOCACHE`, `GOMODCACHE` and
    /// `GOTMPDIR` are kept (the worker's, or the eval's fetch cache). The rest of
    /// goctx's environment is fixed: `GOTOOLCHAIN=local` fails loudly instead of
    /// fetching a toolchain whose export data x/tools can't read, `GOENV=off`
    /// keeps the worker's `go env -w` file out, and `-trimpath` keeps the
    /// checkout's directory out of the build cache's keys: without it every
    /// Review, each in a new directory, recompiled the repository's packages
    /// (128-152 s for a large Go service, past the 120 s deadline).
    pub fn new(
        bin: impl Into<OsString>,
        memory_mb: u32,
        env: Vec<(String, OsString)>,
        permits: usize,
    ) -> Self {
        let mut env: Vec<_> = env
            .into_iter()
            .filter(|(k, _)| PASSED.contains(&k.as_str()))
            .collect();
        let fixed = [
            ("GOTOOLCHAIN", "local"),
            ("GOENV", "off"),
            ("GOFLAGS", "-trimpath"),
            ("GOWORK", "off"),
            ("GOPROXY", "off"),
            ("CGO_ENABLED", "0"),
        ];
        env.extend(fixed.map(|(k, v)| (k.to_owned(), v.into())));
        env.push(("GOMEMLIMIT".to_owned(), format!("{memory_mb}MiB").into()));
        Self {
            bin: bin.into(),
            env,
            permits: Arc::new(Semaphore::new(permits.clamp(1, Semaphore::MAX_PERMITS))),
            grace: GRACE,
        }
    }

    /// This client for one Review whose modules live in `modcache`: the same load
    /// slots, with `GOMODCACHE` set to it for goctx and in [`GoCtx::env`].
    pub fn with_modcache(&self, modcache: &Path) -> Self {
        let mut env = self.env.clone();
        env.retain(|(k, _)| k != "GOMODCACHE");
        env.push(("GOMODCACHE".to_owned(), modcache.into()));
        Self {
            bin: self.bin.clone(),
            env,
            permits: Arc::clone(&self.permits),
            grace: self.grace,
        }
    }

    /// goctx's environment, which `workspace::go_mod_download` starts from.
    pub fn env(&self) -> &[(String, OsString)] {
        &self.env
    }

    /// Context for the reviewed `.go` files of `files` with a patch (others are
    /// ignored): one goctx run over the first 32 in path order, behind the load
    /// semaphore, within `deadline` (the wait included) plus 15 s. Files past the
    /// cap, and every file when the root has no `go.mod` or the run fails, get a
    /// pack holding only an `error`; a file goctx skipped has its reason there.
    pub async fn packs(&self, root: &Path, files: &[&FileChange], deadline: Duration) -> GoPacks {
        let started = Instant::now();
        let mut files: Vec<&FileChange> = files.iter().copied().filter(|f| is_go(f)).collect();
        files.sort_by(|a, b| a.path.cmp(&b.path));
        files.dedup_by(|a, b| a.path == b.path);
        let (sent, capped) = files.split_at(files.len().min(MAX_FILES));
        let mut out = GoPacks::default();
        for f in capped {
            out.packs.insert(f.path.clone(), failed("go file cap"));
        }
        if sent.is_empty() {
            return out;
        }
        if !root.join("go.mod").is_file() {
            for f in sent {
                out.packs
                    .insert(f.path.clone(), failed("no go.mod at the root"));
            }
            return out;
        }
        let result = self.run(root, sent, deadline, started).await;
        out.report.took = started.elapsed();
        match result {
            Ok(resp) => {
                let mut results: BTreeMap<String, FileResult> = resp
                    .files
                    .into_iter()
                    .map(|r| (r.path.clone(), r))
                    .collect();
                for f in sent {
                    let pack = results.remove(&f.path).map_or_else(
                        || failed("diffagent-goctx: no result for the file"),
                        |r| ContextPack {
                            snippets: r.snippets,
                            related: Vec::new(),
                            error: Some(r.skipped).filter(|s| !s.is_empty()),
                        },
                    );
                    out.packs.insert(f.path.clone(), pack);
                }
                out.report.go = resp.go;
                out.report.errors = resp.errors;
            }
            Err(e) => {
                // goctx's stderr (a panic, say) may name the root.
                let error = e.to_string().replace(&*root.to_string_lossy(), ".");
                for f in sent {
                    out.packs.insert(f.path.clone(), failed(&error));
                }
                out.report.failure = Some(e);
            }
        }
        out
    }

    async fn run(
        &self,
        root: &Path,
        files: &[&FileChange],
        deadline: Duration,
        started: Instant,
    ) -> Result<Response, GoCtxError> {
        let Ok(Ok(_permit)) = tokio::time::timeout(deadline, self.permits.acquire()).await else {
            return Err(GoCtxError::Busy);
        };
        let left = deadline.saturating_sub(started.elapsed());
        let deadline_ms = i64::try_from(left.as_millis()).unwrap_or(i64::MAX);
        if deadline_ms < 1 {
            return Err(GoCtxError::Busy);
        }
        let root = std::path::absolute(root).map_err(|e| {
            GoCtxError::Proc(ProcError::Spawn {
                program: self.bin.to_string_lossy().into_owned(),
                reason: e.to_string(),
            })
        })?;
        let files = files.iter().map(|f| {
            let diff = DiffIndex::parse(&f.patch);
            FileRequest {
                path: &f.path,
                visible: diff.lines().map(|(n, _)| n).collect(),
                changed: diff.changed().into_iter().collect(),
            }
        });
        let request = Request {
            root: &root,
            deadline_ms,
            files: files.collect(),
        };
        let stdin =
            serde_json::to_vec(&request).map_err(|e| GoCtxError::BadRequest(e.to_string()))?;
        let cmd = Cmd {
            program: self.bin.clone(),
            args: Vec::new(),
            cwd: root.clone(),
            env: self.env.clone(),
            stdin,
            timeout: left.saturating_add(self.grace),
            stdout_cap: STDOUT_CAP,
        };
        let out = proc::run(cmd, &[]).await.map_err(|e| match e {
            ProcError::Exit {
                code: 2, stderr, ..
            } => GoCtxError::BadRequest(stderr),
            ProcError::Exit {
                code: 3, stderr, ..
            } => GoCtxError::Memory(stderr),
            e => GoCtxError::Proc(e),
        })?;
        serde_json::from_slice(&out).map_err(|e| GoCtxError::Output(e.to_string()))
    }
}

/// A reviewed Go file goctx can use: `go` itself only reads lowercase `.go`, and
/// goctx rejects any other path.
fn is_go(f: &FileChange) -> bool {
    f.path.ends_with(".go") && f.status != FileStatus::Removed && !f.patch.is_empty()
}

fn failed(error: &str) -> ContextPack {
    ContextPack {
        error: Some(error.to_owned()),
        ..ContextPack::default()
    }
}

#[cfg(test)]
pub(super) mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::context::SnippetKind;
    use crate::scan::tests::executable;

    /// A fresh root with a go.mod, plus the directory a fake writes to.
    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("diffagent-goctx-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("root")).unwrap();
        std::fs::write(dir.join("root/go.mod"), "module m\n").unwrap();
        dir
    }

    /// A `diffagent-goctx` stand-in in `dir` that runs `script`.
    pub(in crate::context) fn fake(dir: &Path, script: &str) -> PathBuf {
        let bin = dir.join("diffagent-goctx");
        executable(&bin, &format!("#!/bin/sh\n{script}\n"));
        bin
    }

    pub(in crate::context) fn change(path: &str, status: FileStatus, patch: &str) -> FileChange {
        FileChange {
            path: path.to_owned(),
            status,
            patch: patch.to_owned(),
            previous: None,
        }
    }

    fn go_file(path: &str) -> FileChange {
        change(path, FileStatus::Modified, "@@ -1,2 +1,3 @@\n a\n+b\n c")
    }

    fn errors(packs: &GoPacks) -> Vec<(&str, Option<&str>)> {
        packs
            .packs
            .iter()
            .map(|(p, pack)| (p.as_str(), pack.error.as_deref()))
            .collect()
    }

    const EMPTY: &str = r#"{"go":"go1.27.3","errors":[],"files":[]}"#;

    #[tokio::test]
    async fn canned_answer_request_and_env() {
        let dir = scratch("canned");
        let answer = r#"{"go":"go1.27.3","errors":["./b/b.go:1:1: oops"],"files":[
            {"path":"a.go","skipped":"","snippets":[{"kind":"definition","symbol":"T",
             "path":"t/t.go","start_line":3,"end_line":5,"text":"type T int"}]},
            {"path":"b/b.go","skipped":"build-constrained","snippets":[]}]}"#;
        let d = dir.display();
        let bin = fake(
            &dir,
            &format!("cat > {d}/request.json; env > {d}/env; printf '%s' '{answer}'"),
        );
        let env = [
            ("HOME", "/home/w"),
            ("GOCACHE", "/cache/go"),
            ("GITHUB_TOKEN", "ghs_secret"),
            ("GOFLAGS", "-mod=mod"),
        ];
        let env = env.map(|(k, v)| (k.to_owned(), v.into())).to_vec();
        let go = GoCtx::new(&bin, 800, env, 1);
        let files = [
            go_file("b/b.go"),
            go_file("a.go"),
            change("removed.go", FileStatus::Removed, "@@ -1 +0,0 @@\n-a"),
            change("nopatch.go", FileStatus::Modified, ""),
            change("web.ts", FileStatus::Added, "@@ -0,0 +1 @@\n+a"),
            change("UPPER.GO", FileStatus::Added, "@@ -0,0 +1 @@\n+a"),
        ];
        let files: Vec<&FileChange> = files.iter().collect();
        let root = dir.join("root");
        let got = go.packs(&root, &files, Duration::from_secs(60)).await;

        assert_eq!(got.report.failure, None);
        assert_eq!(got.report.go, "go1.27.3");
        assert_eq!(got.report.errors, ["./b/b.go:1:1: oops"]);
        assert!(got.report.took > Duration::ZERO);
        assert_eq!(
            errors(&got),
            [("a.go", None), ("b/b.go", Some("build-constrained"))]
        );
        let a = &got.packs["a.go"].snippets;
        assert_eq!(a.len(), 1);
        assert_eq!(
            (a[0].kind, a[0].path.as_str()),
            (SnippetKind::Definition, "t/t.go")
        );

        let request: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("request.json")).unwrap()).unwrap();
        assert_eq!(request["root"], root.to_str().unwrap());
        let ms = request["deadline_ms"].as_i64().unwrap();
        assert!(ms > 59_000 && ms <= 60_000, "{ms}");
        assert_eq!(
            request["files"],
            serde_json::json!([
                {"path": "a.go", "visible": [1, 2, 3], "changed": [2]},
                {"path": "b/b.go", "visible": [1, 2, 3], "changed": [2]},
            ])
        );

        let env = std::fs::read_to_string(dir.join("env")).unwrap();
        let mut vars: Vec<&str> = env
            .lines()
            // What the shell running the fake adds itself.
            .filter(|l| {
                !["PWD=", "OLDPWD=", "SHLVL=", "_="]
                    .iter()
                    .any(|p| l.starts_with(p))
            })
            .filter(|l| !l.starts_with("PATH="))
            .collect();
        vars.sort_unstable();
        assert_eq!(
            vars,
            [
                "CGO_ENABLED=0",
                "GOCACHE=/cache/go",
                "GOENV=off",
                "GOFLAGS=-trimpath",
                "GOMEMLIMIT=800MiB",
                "GOPROXY=off",
                "GOTOOLCHAIN=local",
                "GOWORK=off",
                "HOME=/home/w",
            ],
            "{env}"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn a_hang_dies_at_the_deadline_plus_grace() {
        let dir = scratch("hang");
        let mut go = GoCtx::new(fake(&dir, "sleep 60"), 800, Vec::new(), 1);
        go.grace = Duration::from_millis(300);
        let file = go_file("a.go");
        let started = Instant::now();
        let got = go
            .packs(&dir.join("root"), &[&file], Duration::from_millis(200))
            .await;
        let took = started.elapsed();
        assert!(
            took >= Duration::from_millis(450) && took < Duration::from_secs(5),
            "{took:?}"
        );
        let Some(GoCtxError::Proc(ProcError::Timeout { timeout, .. })) = &got.report.failure else {
            panic!("{:?}", got.report.failure);
        };
        assert!(*timeout > Duration::from_millis(300) && *timeout <= Duration::from_millis(500));
        let error = got.report.failure.as_ref().unwrap().to_string();
        assert_eq!(errors(&got), [("a.go", Some(error.as_str()))]);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn failures_set_error() {
        let dir = scratch("failures");
        type Check = fn(&GoCtxError) -> bool;
        let cases: [(&str, Check); 5] = [
            ("echo 'bad request: files: 33' >&2; exit 2", |e| {
                *e == GoCtxError::BadRequest("bad request: files: 33".to_owned())
            }),
            (
                "echo 'heap 1300 MiB passed its hard cap' >&2; exit 3",
                |e| *e == GoCtxError::Memory("heap 1300 MiB passed its hard cap".to_owned()),
            ),
            ("exit 1", |e| {
                matches!(e, GoCtxError::Proc(ProcError::Exit { code: 1, .. }))
            }),
            ("printf 'not json'", |e| matches!(e, GoCtxError::Output(_))),
            ("head -c 4194305 /dev/zero", |e| {
                matches!(
                    e,
                    GoCtxError::Proc(ProcError::TooLarge {
                        cap: STDOUT_CAP,
                        ..
                    })
                )
            }),
        ];
        let file = go_file("a.go");
        for (script, want) in cases {
            let go = GoCtx::new(fake(&dir, script), 800, Vec::new(), 1);
            let got = go
                .packs(&dir.join("root"), &[&file], Duration::from_secs(10))
                .await;
            let failure = got.report.failure.as_ref().unwrap();
            assert!(want(failure), "{script}: {failure:?}");
            let error = failure.to_string();
            assert_eq!(errors(&got), [("a.go", Some(error.as_str()))], "{script}");
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn a_second_run_waits_for_the_slot_and_times_out() {
        let dir = scratch("semaphore");
        let go = GoCtx::new(
            fake(&dir, &format!("sleep 1; printf '%s' '{EMPTY}'")),
            800,
            Vec::new(),
            1,
        );
        let (a, b) = (go_file("a.go"), go_file("b.go"));
        let (a, b) = ([&a], [&b]);
        let root = dir.join("root");
        let first = go.packs(&root, &a, Duration::from_secs(10));
        let second = async {
            // Let the first take the one slot.
            tokio::time::sleep(Duration::from_millis(100)).await;
            go.packs(&root, &b, Duration::from_millis(300)).await
        };
        let (first, second) = tokio::join!(first, second);
        assert_eq!(first.report.failure, None);
        assert_eq!(second.report.failure, Some(GoCtxError::Busy));
        assert_eq!(
            errors(&second),
            [(
                "b.go",
                Some("diffagent-goctx: no load slot before the deadline")
            )]
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// A Review's client has its own module cache in place of the shared one, and waits
    /// for the same load slot as every other.
    #[tokio::test]
    async fn a_review_client_has_its_modcache_and_shares_the_slots() {
        let dir = scratch("view");
        let bin = fake(&dir, &format!("printf '%s' '{EMPTY}'"));
        let env = [("GOMODCACHE", "/tmp/gomodcache"), ("GOCACHE", "/cache/go")];
        let env = env.map(|(k, v)| (k.to_owned(), v.into())).to_vec();
        let go = GoCtx::new(bin, 800, env, 1);
        let review = go.with_modcache(Path::new("/ws/gomodcache"));
        let get = |go: &GoCtx, key: &str| -> Vec<OsString> {
            let vars = go.env().iter().filter(|(k, _)| k == key);
            vars.map(|(_, v)| v.clone()).collect()
        };
        assert_eq!(get(&review, "GOMODCACHE"), ["/ws/gomodcache"]);
        assert_eq!(get(&go, "GOMODCACHE"), ["/tmp/gomodcache"]);
        assert_eq!(get(&review, "GOCACHE"), ["/cache/go"]);

        let held = go.permits.acquire().await.unwrap();
        let file = go_file("a.go");
        let root = dir.join("root");
        let busy = review
            .packs(&root, &[&file], Duration::from_millis(200))
            .await;
        assert_eq!(busy.report.failure, Some(GoCtxError::Busy));
        drop(held);
        let ran = review.packs(&root, &[&file], Duration::from_secs(10)).await;
        assert_eq!(ran.report.failure, None);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn thirty_three_files_send_thirty_two() {
        let dir = scratch("cap");
        let d = dir.display();
        let bin = fake(
            &dir,
            &format!("cat > {d}/request.json; printf '%s' '{EMPTY}'"),
        );
        let go = GoCtx::new(bin, 800, Vec::new(), 1);
        let files: Vec<FileChange> = (0..33).map(|i| go_file(&format!("f{i:02}.go"))).collect();
        let files: Vec<&FileChange> = files.iter().rev().collect();
        let got = go
            .packs(&dir.join("root"), &files, Duration::from_secs(10))
            .await;

        let request: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("request.json")).unwrap()).unwrap();
        let sent: Vec<&str> = request["files"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f["path"].as_str().unwrap())
            .collect();
        let want: Vec<String> = (0..32).map(|i| format!("f{i:02}.go")).collect();
        assert_eq!(sent, want);
        assert_eq!(got.packs.len(), 33);
        assert_eq!(got.packs["f32.go"].error.as_deref(), Some("go file cap"));
        assert_eq!(
            got.packs["f00.go"].error.as_deref(),
            Some("diffagent-goctx: no result for the file")
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn no_root_go_mod_runs_nothing() {
        let dir = scratch("nomod");
        std::fs::remove_file(dir.join("root/go.mod")).unwrap();
        let d = dir.display();
        let go = GoCtx::new(fake(&dir, &format!("touch {d}/ran")), 800, Vec::new(), 1);
        let file = go_file("a.go");
        let got = go
            .packs(&dir.join("root"), &[&file], Duration::from_secs(10))
            .await;
        assert_eq!(errors(&got), [("a.go", Some("no go.mod at the root"))]);
        assert!(!dir.join("ran").exists());
        assert!(
            go.packs(&dir.join("root"), &[], Duration::from_secs(10))
                .await
                .packs
                .is_empty()
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
