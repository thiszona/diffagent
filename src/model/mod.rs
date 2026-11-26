//! Model calls: one interface that returns JSON matching a schema
//! (not a forced tool call, which some providers now reject) over three wires,
//! routed per [`ModelRole`] by `config::Models`, bounded by one deadline, metered per
//! role, and recordable on a [`Tape`](tape::Tape) for offline replay.

pub mod anthropic;
pub mod bedrock;
pub mod http;
pub mod openai;
pub mod tape;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::{Config, ModelRoute, Wire};
use crate::state::ModelRole;
use anthropic::AnthropicModel;
use bedrock::BedrockModel;
use openai::OpenAiModel;
use tape::Tape;

/// A named JSON schema in the portable style (see [`portable`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Schema {
    pub name: &'static str,
    pub json: Value,
}

/// One call: a system prompt, one user message, and an output budget. Temperature
/// is never sent: nothing configures it, and Anthropic's `2026-10-21` API rejects it.
#[derive(Debug, Clone, Copy)]
pub struct Request<'a> {
    pub system: &'a str,
    pub user: &'a str,
    /// Leading bytes of `user` that other requests send too (a PR's conventions, notes
    /// and intent, for each of its files): the Anthropic and Bedrock wires mark them for
    /// their prompt cache, the OpenAI-compatible one ignores it. 0 for none. Not in the
    /// tape key: the bytes sent are the same.
    pub shared: usize,
    /// The JSON the reply must match; `None` for prose.
    pub schema: Option<&'a Schema>,
    pub max_tokens: u32,
}

impl Request<'_> {
    /// `user` as its shared head and the rest, when `shared` cuts it into two
    /// non-empty parts on a char boundary: what a caching wire sends as two blocks.
    pub fn split_user(&self) -> Option<(&str, &str)> {
        let parts = self.user.split_at_checked(self.shared);
        parts.filter(|(head, tail)| !head.is_empty() && !tail.is_empty())
    }
}

/// A model's raw reply, before any validation. This is what a [`Tape`] records, so a
/// replay re-runs decoding and the retry exactly as the live call did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Reply {
    pub text: String,
    pub stop: Stop,
    pub tokens: Tokens,
}

/// Why the model stopped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stop {
    End,
    MaxTokens,
    /// Refusal, content filter, guardrail, context overflow: the provider's reason.
    Other(String),
}

/// Token counts, normalized across wires: `input` excludes cached tokens.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tokens {
    pub input: i64,
    pub output: i64,
    pub cache_read: i64,
    pub cache_write: i64,
}

impl Tokens {
    /// Saturating: the counts come unchecked from the provider's response.
    pub fn add(&mut self, t: Tokens) {
        self.input = self.input.saturating_add(t.input);
        self.output = self.output.saturating_add(t.output);
        self.cache_read = self.cache_read.saturating_add(t.cache_read);
        self.cache_write = self.cache_write.saturating_add(t.cache_write);
    }
}

/// One wire to one model.
#[async_trait]
pub trait Model: Send + Sync {
    async fn complete(&self, req: &Request<'_>) -> Result<Reply, ModelError>;
}

/// Why a model call gave no usable answer. Keys travel only in headers, so no
/// message here can contain one.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ModelError {
    /// The provider answered with an error status; `message` is cut to 300 chars.
    #[error("provider returned {status}: {message}")]
    Http { status: u16, message: String },
    /// The request never got an answer: connection, TLS, SDK.
    #[error("request failed: {0}")]
    Request(String),
    #[error("model call timed out")]
    Timeout,
    #[error("reply cut off at max_tokens")]
    Truncated,
    #[error("model stopped: {0}")]
    Stopped(String),
    #[error("invalid reply: {0}")]
    Invalid(String),
    /// Replay found no recording for this call.
    #[error("no recording for this call")]
    Unrecorded,
    #[error("no model configured for the {0:?} role")]
    NoRoute(ModelRole),
}

/// Every model call, by role. Each call keeps its safe default in its caller.
///
/// | Call | Role |
/// |---|---|
/// | per-file review | review |
/// | walkthrough, diagram | summary |
/// | verifier (optional: [`Router::has`]) | verify |
/// | reply intent; every reply in a Thread | classify |
/// | Q&A answer, approval gate, concession check | review |
/// | lesson distillation | summary |
///
/// Every call runs under `per_file_timeout`, the one bound on every wire, so no call is
/// unbounded.
#[derive(Clone)] // Arc fields: router.clone().with_tape(t) per fixture is cheap.
pub struct Router {
    routes: HashMap<ModelRole, Route>,
    timeout: Duration,
    tape: Option<Arc<Tape>>,
}

#[derive(Clone)]
struct Route {
    route: ModelRoute,
    model: Arc<dyn Model>,
    /// Dollars per 1M tokens, (input, output).
    price: (f64, f64),
}

impl Router {
    /// The router `cfg.models` describes: one model per role, verify only when set.
    /// Bedrock routes share one SDK client and the HTTP wires one [`http::client`],
    /// each built only when a route needs it.
    pub async fn from_config(cfg: &Config) -> Result<Self, ModelError> {
        let m = &cfg.models;
        let mut sdk: Option<aws_sdk_bedrockruntime::Client> = None;
        let mut web = None;
        let mut roles: Vec<(ModelRole, ModelRoute, Arc<dyn Model>)> = Vec::new();
        for (role, route) in [
            (ModelRole::Review, Some(&m.review)),
            (ModelRole::Summary, Some(&m.summary)),
            (ModelRole::Verify, m.verify.as_ref()),
            (ModelRole::Classify, Some(&m.classify)),
        ] {
            let Some(route) = route else { continue };
            let model: Arc<dyn Model> = match &route.wire {
                Wire::BedrockConverse => {
                    let client = match &sdk {
                        Some(c) => c.clone(),
                        None => sdk
                            .insert(bedrock::client(cfg.bedrock_region.as_deref()).await)
                            .clone(),
                    };
                    Arc::new(BedrockModel::new(client, &route.model))
                }
                Wire::AnthropicMessages { base_url, api_key } => Arc::new(AnthropicModel::new(
                    shared(&mut web)?,
                    base_url,
                    api_key.clone(),
                    &route.model,
                )),
                Wire::OpenAiChat { base_url, api_key } => Arc::new(OpenAiModel::new(
                    shared(&mut web)?,
                    base_url,
                    api_key.clone(),
                    &route.provider,
                    &route.model,
                )),
            };
            roles.push((role, route.clone(), model));
        }
        Ok(Self::new(cfg, roles))
    }

    /// A router over `roles`, with the deadline and prices from `cfg`. A role left
    /// out has no route: [`Router::has`] is false and calls fail with `NoRoute`.
    pub fn new(cfg: &Config, roles: Vec<(ModelRole, ModelRoute, Arc<dyn Model>)>) -> Self {
        let routes = roles
            .into_iter()
            .map(|(role, route, model)| {
                let price = prices(cfg, role);
                (
                    role,
                    Route {
                        route,
                        model,
                        price,
                    },
                )
            })
            .collect();
        Self {
            routes,
            timeout: cfg.per_file_timeout,
            tape: None,
        }
    }

    /// Records or replays every call on `tape`.
    pub fn with_tape(mut self, tape: Arc<Tape>) -> Self {
        self.tape = Some(tape);
        self
    }

    /// Whether `role` has a model (verify is optional).
    pub fn has(&self, role: ModelRole) -> bool {
        self.routes.contains_key(&role)
    }

    /// One call, its raw reply metered on `m`. A truncated reply is returned as is:
    /// the caller decides whether cut-off prose is usable.
    pub async fn text(
        &self,
        role: ModelRole,
        req: &Request<'_>,
        m: &Meter,
    ) -> Result<Reply, ModelError> {
        let r = self.routes.get(&role).ok_or(ModelError::NoRoute(role))?;
        let live = async {
            tokio::time::timeout(self.timeout, r.model.complete(req))
                .await
                .map_err(|_| ModelError::Timeout)?
        };
        let reply = match &self.tape {
            Some(t) => t.play(tape::key(&r.route, req), live).await?,
            None => live.await?,
        };
        m.update(role, |t| {
            t.calls += 1;
            t.tokens.add(reply.tokens);
            if reply.stop == Stop::MaxTokens {
                t.truncated += 1;
            }
        });
        Ok(reply)
    }

    /// One call decoded as `T` against `req.schema`. A reply cut off at `max_tokens`
    /// is `Truncated` and any other early stop `Stopped`, neither retried. A reply
    /// that does not decode gets one retry, with the error appended to the user
    /// message; a second failure is `Invalid`.
    pub async fn json<T: DeserializeOwned>(
        &self,
        role: ModelRole,
        req: &Request<'_>,
        m: &Meter,
    ) -> Result<T, ModelError> {
        let schema = req
            .schema
            .ok_or_else(|| ModelError::Invalid("no schema".to_owned()))?;
        let reply = self.text(role, req, m).await?;
        let problem = match accept(&reply, schema) {
            Err(ModelError::Invalid(problem)) => problem,
            other => return other,
        };
        m.update(role, |t| t.retried += 1);
        let user = format!(
            "{}\n\nYour previous reply could not be used: {problem}. Reply again with only \
             the JSON object matching the schema.",
            req.user
        );
        let retry = Request {
            user: &user,
            ..*req
        };
        accept(&self.text(role, &retry, m).await?, schema)
    }

    /// Per-role usage and cost so far, in role order; roles never called are left out.
    pub fn usage(&self, m: &Meter) -> Vec<RoleUsage> {
        let tallies = m.0.lock().unwrap_or_else(PoisonError::into_inner);
        [
            ModelRole::Review,
            ModelRole::Summary,
            ModelRole::Verify,
            ModelRole::Classify,
        ]
        .into_iter()
        .filter_map(|role| {
            let tally = *tallies.get(&role)?;
            let r = self.routes.get(&role)?;
            Some(RoleUsage {
                role,
                provider: r.route.provider.clone(),
                model: r.route.model.clone(),
                tally,
                cost_usd: cost(tally.tokens, r.price),
            })
        })
        .collect()
    }
}

/// The process's one HTTP client, built on first use.
fn shared(web: &mut Option<reqwest::Client>) -> Result<reqwest::Client, ModelError> {
    if let Some(c) = web {
        return Ok(c.clone());
    }
    Ok(web.insert(http::client()?).clone())
}

/// `reply` decoded as `T`, if it ended normally.
fn accept<T: DeserializeOwned>(reply: &Reply, schema: &Schema) -> Result<T, ModelError> {
    match &reply.stop {
        Stop::MaxTokens => Err(ModelError::Truncated),
        Stop::Other(why) => Err(ModelError::Stopped(why.clone())),
        Stop::End => decode(&reply.text, &schema.json).map_err(ModelError::Invalid),
    }
}

/// Dollars per 1M tokens for `role`, (input, output). Summary and classify have
/// their own prices, each falling back to the review price when 0.
// ponytail: verify priced as review; add its prices when it gets its own model.
fn prices(cfg: &Config, role: ModelRole) -> (f64, f64) {
    let review = (cfg.model_price_in_per_m, cfg.model_price_out_per_m);
    let own = match role {
        ModelRole::Summary => (cfg.summary_price_in_per_m, cfg.summary_price_out_per_m),
        ModelRole::Classify => (cfg.classify_price_in_per_m, cfg.classify_price_out_per_m),
        ModelRole::Verify => (cfg.verify_price_in_per_m, cfg.verify_price_out_per_m),
        ModelRole::Review => return review,
    };
    let or_review = |p: f64, r: f64| if p == 0.0 { r } else { p };
    (or_review(own.0, review.0), or_review(own.1, review.1))
}

/// Cache tokens as input tokens at full price: a write costs 1.25×, a read 0.1×.
///
/// ponytail: Anthropic's 5-minute rates for Sonnet and Haiku, the cache the Anthropic
/// and Bedrock wires write; Opus 5.5 (0.05×) and Fable 5.1 (0.025×) reads, Bedrock's
/// own rates, and an OpenAI-compatible provider's are priced the same. Add a cache-read
/// price if a role moves to one of them.
fn cached_input(t: Tokens) -> f64 {
    1.25 * t.cache_write as f64 + 0.1 * t.cache_read as f64
}

/// Dollars for `t` at (input, output) dollars per million tokens, cache tokens
/// included: the usage rows and `diffagent-eval`'s `--max-usd` both price by it.
pub fn cost(t: Tokens, (price_in, price_out): (f64, f64)) -> f64 {
    t.input as f64 / 1e6 * price_in
        + t.output as f64 / 1e6 * price_out
        + cached_input(t) / 1e6 * price_in
}

/// Per-role tallies for one job. Every reply is counted: truncated, invalid and
/// retried ones too.
#[derive(Debug, Default)]
pub struct Meter(Mutex<HashMap<ModelRole, Tally>>);

impl Meter {
    fn update(&self, role: ModelRole, f: impl FnOnce(&mut Tally)) {
        let mut tallies = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        f(tallies.entry(role).or_default());
    }
}

/// What one role spent in one job.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Tally {
    pub tokens: Tokens,
    /// Replies received, retries included.
    pub calls: u32,
    /// Retries after a reply that did not decode.
    pub retried: u32,
    /// Replies cut off at `max_tokens`.
    pub truncated: u32,
}

/// One role's usage with its route and estimated cost.
#[derive(Debug, Clone, PartialEq)]
pub struct RoleUsage {
    pub role: ModelRole,
    pub provider: String,
    pub model: String,
    pub tally: Tally,
    pub cost_usd: f64,
}

/// Decodes a reply's text as `T`, forgiving what models get wrong without changing
/// meaning: (1) the object is taken from a ```json fence or surrounding prose (the
/// first complete value from the first `{`, so prose after it may hold a `}`); (2)
/// parsed; (3) coerced by `schema`: a numeric string or
/// integral float becomes an integer where one is wanted, and a string where an
/// array or object is wanted is parsed; (4) deserialized, ignoring unknown keys. A
/// missing nullable field is `None`; only a missing non-null field fails. Enum,
/// range and safe-default checks belong to the callers.
pub fn decode<T: DeserializeOwned>(text: &str, schema: &Value) -> Result<T, String> {
    let start = text.find('{').ok_or("no JSON object in the reply")?;
    let mut value: Value = serde_json::Deserializer::from_str(&text[start..])
        .into_iter()
        .next()
        .ok_or("no JSON object in the reply")?
        .map_err(|e| format!("not JSON: {e}"))?;
    coerce(&mut value, schema)?;
    serde_json::from_value(value).map_err(|e| e.to_string())
}

/// Coerces `v` toward `schema` in place, walking only `properties` and `items`.
fn coerce(v: &mut Value, schema: &Value) -> Result<(), String> {
    if let Value::String(s) = v {
        if wants(schema, "array") || wants(schema, "object") {
            *v = serde_json::from_str(s)
                .map_err(|e| format!("string-encoded array or object is not JSON: {e}"))?;
        } else if wants(schema, "integer") {
            let s = s.trim();
            if let Some(n) = s.parse().ok().or_else(|| integral(s.parse().ok())) {
                *v = Value::from(n);
            }
        }
    } else if let Value::Number(n) = v
        && n.is_f64()
        && wants(schema, "integer")
        && let Some(i) = integral(n.as_f64())
    {
        *v = Value::from(i);
    }
    match v {
        Value::Object(map) => {
            let props = schema.get("properties").and_then(Value::as_object);
            for (key, sub) in props.into_iter().flatten() {
                if let Some(field) = map.get_mut(key) {
                    coerce(field, sub).map_err(|e| format!("{key}: {e}"))?;
                }
            }
        }
        Value::Array(items) => {
            if let Some(sub) = schema.get("items") {
                for (i, item) in items.iter_mut().enumerate() {
                    coerce(item, sub).map_err(|e| format!("[{i}]: {e}"))?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// `f` as an i64 when it is a whole number in range.
fn integral(f: Option<f64>) -> Option<i64> {
    // i64::MAX as f64 rounds up to 2^63, so the upper bound is exclusive.
    f.filter(|f| f.fract() == 0.0 && *f >= i64::MIN as f64 && *f < i64::MAX as f64)
        .map(|f| f as i64)
}

/// Whether `schema`'s type is `ty`, alone or nullable.
fn wants(schema: &Value, ty: &str) -> bool {
    match schema.get("type") {
        Some(Value::String(t)) => t == ty,
        Some(Value::Array(ts)) => ts.iter().any(|t| t == ty),
        _ => false,
    }
}

/// The most nullable type arrays a schema may hold: the Anthropic and Bedrock cap.
const MAX_NULLABLE: usize = 16;

/// Keywords the portable schema style forbids: limits are checked in code, and
/// composition is not portable across providers.
const FORBIDDEN: [&str; 23] = [
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "minContains",
    "maxContains",
    "minProperties",
    "maxProperties",
    "pattern",
    "patternProperties",
    "$ref",
    "$defs",
    "definitions",
    "anyOf",
    "oneOf",
    "allOf",
    "not",
    "if",
];

/// Checks the portable schema style: the root is an object; every object
/// has `additionalProperties: false` and all its properties `required`; no limit,
/// `pattern`, reference or composition keywords; every node typed, nullable only as
/// `"type": [T, "null"]`, at most 16 times.
pub fn portable(schema: &Value) -> Result<(), String> {
    if !wants(schema, "object") {
        return Err("the root must be an object".to_owned());
    }
    let mut nullable = 0;
    check(schema, "$", &mut nullable)?;
    if nullable > MAX_NULLABLE {
        return Err(format!(
            "{nullable} nullable types, more than the {MAX_NULLABLE} providers accept"
        ));
    }
    Ok(())
}

fn check(schema: &Value, path: &str, nullable: &mut usize) -> Result<(), String> {
    if let Some(k) = FORBIDDEN.iter().find(|k| schema.get(**k).is_some()) {
        return Err(format!("{path}: {k} is not portable"));
    }
    match schema.get("type") {
        Some(Value::String(_)) => {}
        Some(Value::Array(ts)) if ts.len() == 2 && ts.iter().any(|t| t == "null") => {
            *nullable += 1;
        }
        _ => return Err(format!(r#"{path}: type must be T or [T, "null"]"#)),
    }
    if wants(schema, "object") {
        if schema.get("additionalProperties") != Some(&Value::Bool(false)) {
            return Err(format!("{path}: additionalProperties must be false"));
        }
        let required: Vec<&str> = schema
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        let props = schema.get("properties").and_then(Value::as_object);
        for (key, sub) in props.into_iter().flatten() {
            if !required.contains(&key.as_str()) {
                return Err(format!("{path}.{key}: every property must be required"));
            }
            check(sub, &format!("{path}.{key}"), nullable)?;
        }
    }
    if let Some(items) = schema.get("items") {
        check(items, &format!("{path}[]"), nullable)?;
    }
    Ok(())
}

#[cfg(test)]
pub(super) mod tests {
    use std::collections::VecDeque;

    use serde_json::json;

    use super::*;
    use crate::config::Wire;

    /// A model that answers from a script and records each request's user message.
    #[derive(Default)]
    pub(crate) struct Script {
        replies: Mutex<VecDeque<Reply>>,
        pub(crate) users: Mutex<Vec<String>>,
    }

    impl Script {
        pub(crate) fn new(replies: &[(&str, Stop)]) -> Arc<Self> {
            let replies = replies
                .iter()
                .map(|(text, stop)| reply(text, stop.clone()))
                .collect();
            Arc::new(Self {
                replies: Mutex::new(replies),
                users: Mutex::default(),
            })
        }

        pub(crate) fn calls(&self) -> usize {
            self.users.lock().unwrap().len()
        }
    }

    #[async_trait]
    impl Model for Script {
        async fn complete(&self, req: &Request<'_>) -> Result<Reply, ModelError> {
            self.users.lock().unwrap().push(req.user.to_owned());
            let next = self.replies.lock().unwrap().pop_front();
            next.ok_or_else(|| ModelError::Request("script exhausted".to_owned()))
        }
    }

    struct Hang;

    #[async_trait]
    impl Model for Hang {
        async fn complete(&self, _: &Request<'_>) -> Result<Reply, ModelError> {
            std::future::pending().await
        }
    }

    pub(crate) fn reply(text: &str, stop: Stop) -> Reply {
        Reply {
            text: text.to_owned(),
            stop,
            tokens: Tokens {
                input: 1000,
                output: 100,
                cache_read: 10,
                cache_write: 1,
            },
        }
    }

    pub(crate) fn route(model: &str) -> ModelRoute {
        ModelRoute {
            provider: "bedrock".to_owned(),
            model: model.to_owned(),
            wire: Wire::BedrockConverse,
        }
    }

    pub(crate) fn config(vars: &[(&str, &str)]) -> Config {
        let vars: HashMap<String, String> = vars
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        Config::from_lookup(|k| vars.get(k).cloned()).unwrap()
    }

    /// A router with only the review role, on `model`.
    pub(crate) fn router(model: Arc<dyn Model>) -> Router {
        let cfg = config(&[
            ("MODEL_PRICE_IN_PER_M", "3"),
            ("MODEL_PRICE_OUT_PER_M", "15"),
        ]);
        Router::new(&cfg, vec![(ModelRole::Review, route("haiku-5-5"), model)])
    }

    pub(crate) fn tally(router: &Router, m: &Meter) -> Tally {
        router.usage(m).first().map(|u| u.tally).unwrap_or_default()
    }

    /// The wire tests' request: `schema`, or none.
    pub(crate) fn req(schema: Option<&Schema>) -> Request<'_> {
        Request {
            system: "sys",
            user: "review this",
            shared: 0,
            schema,
            max_tokens: 4000,
        }
    }

    /// A small portable schema exercising every coercion.
    pub(crate) fn small_schema() -> Schema {
        Schema {
            name: "small",
            json: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["n", "items", "note"],
                "properties": {
                    "n": {"type": "integer"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": false,
                            "required": ["k"],
                            "properties": {"k": {"type": "integer"}}
                        }
                    },
                    "note": {"type": ["string", "null"]}
                }
            }),
        }
    }

    #[derive(Debug, PartialEq, Deserialize)]
    pub(crate) struct Small {
        pub(crate) n: i64,
        items: Vec<Item>,
        note: Option<String>,
    }

    #[derive(Debug, PartialEq, Deserialize)]
    struct Item {
        k: i64,
    }

    fn small(n: i64, ks: &[i64], note: Option<&str>) -> Small {
        Small {
            n,
            items: ks.iter().map(|k| Item { k: *k }).collect(),
            note: note.map(str::to_owned),
        }
    }

    #[test]
    fn decode_table() {
        let schema = small_schema().json;
        let cases: [(&str, &str, Option<Small>); 16] = [
            (
                "trailing prose with a brace",
                "{\"n\":1,\"items\":[],\"note\":null}\nUse `map[string]int{}` here.",
                Some(small(1, &[], None)),
            ),
            (
                "fence, then prose with a brace",
                "```json\n{\"n\":2,\"items\":[],\"note\":\"}\"}\n```\nDone }",
                Some(small(2, &[], Some("}"))),
            ),
            (
                "plain",
                r#"{"n":1,"items":[{"k":2}],"note":"x"}"#,
                Some(small(1, &[2], Some("x"))),
            ),
            (
                "fence",
                "```json\n{\"n\":1,\"items\":[],\"note\":null}\n```",
                Some(small(1, &[], None)),
            ),
            (
                "prose around",
                "Here is the review:\n{\"n\":1,\"items\":[],\"note\":null}\nHope it helps.",
                Some(small(1, &[], None)),
            ),
            (
                "string and float integers",
                r#"{"n":"7","items":[{"k":2.0},{"k":" 3 "}],"note":null}"#,
                Some(small(7, &[2, 3], None)),
            ),
            (
                "non-integral float",
                r#"{"n":1.5,"items":[],"note":null}"#,
                None,
            ),
            (
                "non-numeric string",
                r#"{"n":"seven","items":[],"note":null}"#,
                None,
            ),
            (
                "string-encoded array",
                r#"{"n":1,"items":"[{\"k\":\"4\"}]","note":null}"#,
                Some(small(1, &[4], None)),
            ),
            (
                "cut-off string-encoded array",
                r#"{"n":1,"items":"[{\"k\":4},{\"k","note":null}"#,
                None,
            ),
            ("EOF", r#"{"n":1,"items":[{"k":2}"#, None),
            (
                "missing nullable",
                r#"{"n":1,"items":[]}"#,
                Some(small(1, &[], None)),
            ),
            (
                "unknown keys ignored",
                r#"{"n":1,"items":[{"k":1,"suggestion ":"x"}],"note":null,"extra":true}"#,
                Some(small(1, &[1], None)),
            ),
            ("missing non-null", r#"{"items":[],"note":null}"#, None),
            ("empty reply", "", None),
            ("no object", "I could not review this file.", None),
        ];
        for (name, text, want) in cases {
            let got = decode::<Small>(text, &schema);
            match want {
                Some(want) => assert_eq!(got, Ok(want), "{name}"),
                None => assert!(got.is_err(), "{name}: {got:?}"),
            }
        }
    }

    #[test]
    fn decode_error_names_the_field() {
        let err = decode::<Small>(r#"{"n":1,"items":"[{","note":null}"#, &small_schema().json)
            .unwrap_err();
        assert!(err.starts_with("items: string-encoded"), "{err}");
    }

    /// The per-file findings shape, in portable style.
    fn findings_schema() -> Value {
        let s = |t: &str| json!({"type": t});
        json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["findings"],
            "properties": {
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["start_line", "end_line", "severity", "title", "body",
                                     "code_snippet", "suggestion"],
                        "properties": {
                            "start_line": s("integer"),
                            "end_line": s("integer"),
                            "severity": s("string"),
                            "title": s("string"),
                            "body": s("string"),
                            "code_snippet": s("string"),
                            "suggestion": {"type": ["string", "null"]}
                        }
                    }
                }
            }
        })
    }

    #[derive(Deserialize)]
    struct Findings {
        findings: Vec<Found>,
    }

    #[derive(Deserialize)]
    #[allow(
        dead_code,
        reason = "decoded to prove the shape; only some fields are read"
    )]
    struct Found {
        start_line: i64,
        end_line: i64,
        severity: String,
        title: String,
        body: String,
        code_snippet: String,
        suggestion: Option<String>,
    }

    /// Per-file replies the way recorded ones came back: line numbers as strings,
    /// invented keys, `suggestion` never named right, and sometimes a string-encoded
    /// findings array that was cut off.
    #[test]
    fn decode_recorded_replies() {
        let schema = findings_schema();
        assert_eq!(portable(&schema), Ok(()));
        let cases = [
            (
                "string line numbers",
                r#"{"findings":[{"start_line":"12","end_line":"14","severity":"major",
                    "title":"t","body":"b","code_snippet":"total := sum(items)"}]}"#,
                Ok(vec![(12, 14)]),
            ),
            (
                "invented keys and a misnamed suggestion",
                r#"{"findings":[{"start_line":"3","end_line":"3","severity":"minor",
                    "title":"t","body":"b","code_snippet":"c","confidence":"high",
                    "category":"bug","suggestion ":"x := 1","suggested_fix":"x := 1"}],
                    "summary":"one issue"}"#,
                Ok(vec![(3, 3)]),
            ),
            ("no findings", r#"{"findings":[]}"#, Ok(Vec::new())),
            (
                "string-encoded array",
                r#"{"findings":"[{\"start_line\":\"5\",\"end_line\":\"6\",\"severity\":\"major\",\"title\":\"t\",\"body\":\"b\",\"code_snippet\":\"c\"}]"}"#,
                Ok(vec![(5, 6)]),
            ),
            (
                "cut-off string-encoded array",
                r#"{"findings":"[{\"start_line\":\"5\",\"end_line\":\"5\",\"severity\":\"major\",\"title\":\"t\",\"body\":\"the loop never"}"#,
                Err("findings: string-encoded"),
            ),
        ];
        for (name, text, want) in cases {
            match (decode::<Findings>(text, &schema), want) {
                (Ok(got), Ok(want)) => {
                    let lines: Vec<(i64, i64)> = got
                        .findings
                        .iter()
                        .map(|f| (f.start_line, f.end_line))
                        .collect();
                    assert_eq!(lines, want, "{name}");
                    assert!(
                        got.findings.iter().all(|f| f.suggestion.is_none()),
                        "{name}"
                    );
                }
                (Err(e), Err(prefix)) => assert!(e.starts_with(prefix), "{name}: {e}"),
                (got, _) => panic!("{name}: {:?}", got.map(|f| f.findings.len())),
            }
        }
    }

    #[tokio::test(start_paused = true)]
    async fn json_calls() {
        let schema = small_schema();
        let valid = r#"{"n":1,"items":[],"note":null}"#;
        let coerced = r#"{"n":"1","items":"[]","note":null}"#;
        let invalid = r#"{"n":"one","items":[],"note":null}"#;
        let cut = r#"{"n":1,"items":"[{\"k\":1},{\"k","note":null}"#;
        let end = || Stop::End;
        // (name, script, want, calls, retried, truncated)
        let cases = [
            ("coercion only", vec![(coerced, end())], Ok(1), 1, 0, 0),
            ("valid", vec![(valid, end())], Ok(1), 1, 0, 0),
            (
                "invalid then valid",
                vec![(invalid, end()), (valid, end())],
                Ok(1),
                2,
                1,
                0,
            ),
            (
                "cut-off string-encoded array then valid",
                vec![(cut, end()), (valid, end())],
                Ok(1),
                2,
                1,
                0,
            ),
            (
                "invalid twice",
                vec![(invalid, end()), (cut, end())],
                Err("invalid"),
                2,
                1,
                0,
            ),
            (
                "truncated",
                vec![(cut, Stop::MaxTokens)],
                Err("truncated"),
                1,
                0,
                1,
            ),
            (
                "stopped",
                vec![(valid, Stop::Other("guardrail".to_owned()))],
                Err("stopped"),
                1,
                0,
                0,
            ),
        ];
        for (name, script, want, calls, retried, truncated) in cases {
            let model = Script::new(&script);
            let router = router(model.clone());
            let m = Meter::default();
            let req = Request {
                system: "sys",
                user: "review this",
                shared: 0,
                schema: Some(&schema),
                max_tokens: 100,
            };
            let got = router.json::<Small>(ModelRole::Review, &req, &m).await;
            let got = match got {
                Ok(s) => Ok(s.n),
                Err(ModelError::Invalid(_)) => Err("invalid"),
                Err(ModelError::Truncated) => Err("truncated"),
                Err(ModelError::Stopped(why)) if why == "guardrail" => Err("stopped"),
                Err(e) => panic!("{name}: {e}"),
            };
            assert_eq!(got, want, "{name}");
            let t = tally(&router, &m);
            assert_eq!(
                (t.calls, t.retried, t.truncated, model.calls()),
                (calls, retried, truncated, calls as usize),
                "{name}"
            );
            assert_eq!(
                t.tokens.output,
                100 * i64::from(calls),
                "{name}: every reply metered"
            );
            let users = model.users.lock().unwrap();
            if let Some(retry) = users.get(1) {
                assert!(
                    retry.starts_with("review this\n\nYour previous reply could not be used: "),
                    "{name}: {retry}"
                );
            }
        }
    }

    #[tokio::test(start_paused = true)]
    async fn json_failures() {
        let schema = small_schema();
        let m = Meter::default();
        let req = Request {
            system: "sys",
            user: "u",
            shared: 0,
            schema: Some(&schema),
            max_tokens: 100,
        };

        let model = Script::new(&[]);
        let r = router(model.clone());
        let no_schema = Request {
            schema: None,
            ..req
        };
        let got = r.json::<Small>(ModelRole::Review, &no_schema, &m).await;
        assert_eq!(got, Err(ModelError::Invalid("no schema".to_owned())));
        assert_eq!(model.calls(), 0);

        let got = router(Arc::new(Hang))
            .json::<Small>(ModelRole::Review, &req, &m)
            .await;
        assert_eq!(got, Err(ModelError::Timeout));

        assert!(!r.has(ModelRole::Verify) && r.has(ModelRole::Review));
        let got = r.json::<Small>(ModelRole::Verify, &req, &m).await;
        assert_eq!(got, Err(ModelError::NoRoute(ModelRole::Verify)));
        assert!(r.usage(&m).is_empty(), "failed calls are not replies");
    }

    #[tokio::test]
    async fn usage_and_cost() {
        let model = Script::new(&[("prose", Stop::End), ("cut", Stop::MaxTokens)]);
        let r = router(model);
        let m = Meter::default();
        let req = Request {
            system: "s",
            user: "u",
            shared: 0,
            schema: None,
            max_tokens: 10,
        };
        assert_eq!(
            r.text(ModelRole::Review, &req, &m).await.unwrap().text,
            "prose"
        );
        let cut = r.text(ModelRole::Review, &req, &m).await.unwrap();
        assert_eq!(
            cut.stop,
            Stop::MaxTokens,
            "text returns a truncated reply as is"
        );
        let usage = r.usage(&m);
        let want = RoleUsage {
            role: ModelRole::Review,
            provider: "bedrock".to_owned(),
            model: "haiku-5-5".to_owned(),
            tally: Tally {
                tokens: Tokens {
                    input: 2000,
                    output: 200,
                    cache_read: 20,
                    cache_write: 2,
                },
                calls: 2,
                retried: 0,
                truncated: 1,
            },
            // 2000 in at $3/M + 200 out at $15/M, then 2 cache writes at 1.25× and 20
            // reads at 0.1× the input price.
            cost_usd: 0.006 + 0.003 + (2.0 * 1.25 + 20.0 * 0.1) * 3.0 / 1e6,
        };
        assert_eq!(usage.len(), 1);
        assert!((usage[0].cost_usd - want.cost_usd).abs() < 1e-12);
        assert_eq!(
            RoleUsage {
                cost_usd: want.cost_usd,
                ..usage[0].clone()
            },
            want
        );
    }

    /// HTTP wires only: a Bedrock route would load the AWS config from the machine.
    #[tokio::test]
    async fn from_config_routes_each_role() {
        let cfg = config(&[
            ("REVIEW_MODEL", "openai/gpt-5"),
            ("OPENAI_API_KEY", "k1"),
            ("SUMMARY_MODEL", "anthropic/claude-haiku-4-5"),
            ("ANTHROPIC_API_KEY", "k2"),
        ]);
        let r = Router::from_config(&cfg).await.unwrap();
        let review = Some(&cfg.models.review);
        let cases = [
            (ModelRole::Review, review),
            (ModelRole::Summary, Some(&cfg.models.summary)),
            (ModelRole::Verify, None),
            (ModelRole::Classify, review),
        ];
        for (role, want) in cases {
            assert_eq!(r.routes.get(&role).map(|r| &r.route), want, "{role:?}");
        }
    }

    #[test]
    fn role_prices() {
        let review = [
            ("MODEL_PRICE_IN_PER_M", "3"),
            ("MODEL_PRICE_OUT_PER_M", "15"),
        ];
        let cases = [
            (vec![], ModelRole::Summary, (3.0, 15.0)),
            (
                vec![("SUMMARY_MODEL_PRICE_IN_PER_M", "0.06")],
                ModelRole::Summary,
                (0.06, 15.0),
            ),
            (
                vec![
                    ("SUMMARY_MODEL_PRICE_IN_PER_M", "0.06"),
                    ("SUMMARY_MODEL_PRICE_OUT_PER_M", "0.24"),
                ],
                ModelRole::Summary,
                (0.06, 0.24),
            ),
            (
                vec![("SUMMARY_MODEL_PRICE_OUT_PER_M", "0.24")],
                ModelRole::Review,
                (3.0, 15.0),
            ),
            (
                vec![("SUMMARY_MODEL_PRICE_IN_PER_M", "0.06")],
                ModelRole::Verify,
                (3.0, 15.0),
            ),
            (
                vec![("SUMMARY_MODEL_PRICE_IN_PER_M", "0.06")],
                ModelRole::Classify,
                (3.0, 15.0),
            ),
            (
                vec![
                    ("CLASSIFY_MODEL_PRICE_IN_PER_M", "0.1"),
                    ("CLASSIFY_MODEL_PRICE_OUT_PER_M", "0.5"),
                ],
                ModelRole::Classify,
                (0.1, 0.5),
            ),
            (
                vec![("CLASSIFY_MODEL_PRICE_OUT_PER_M", "0.5")],
                ModelRole::Classify,
                (3.0, 0.5),
            ),
            (
                vec![("CLASSIFY_MODEL_PRICE_IN_PER_M", "0.1")],
                ModelRole::Summary,
                (3.0, 15.0),
            ),
            (
                vec![
                    ("VERIFY_MODEL_PRICE_IN_PER_M", "0.1"),
                    ("VERIFY_MODEL_PRICE_OUT_PER_M", "0.5"),
                ],
                ModelRole::Verify,
                (0.1, 0.5),
            ),
            (
                vec![("VERIFY_MODEL_PRICE_IN_PER_M", "0.1")],
                ModelRole::Review,
                (3.0, 15.0),
            ),
        ];
        for (extra, role, want) in cases {
            let mut vars = review.to_vec();
            vars.extend(extra);
            assert_eq!(prices(&config(&vars), role), want, "{role:?} {vars:?}");
        }
    }

    /// Usage numbers come unchecked from the provider; summing them never panics.
    #[test]
    fn tokens_saturate() {
        let max = Tokens {
            input: i64::MAX,
            output: i64::MAX,
            cache_read: i64::MAX,
            cache_write: i64::MAX,
        };
        let mut t = max;
        t.add(max);
        assert_eq!(t, max);
    }

    #[test]
    fn portable_rules() {
        let obj = |props: Value| {
            let required: Vec<&String> = props.as_object().unwrap().keys().collect();
            json!({"type": "object", "additionalProperties": false,
                   "required": required, "properties": props})
        };
        let with = |key: &str, value: Value| {
            let mut s = json!({"type": "string"});
            s[key] = value;
            obj(json!({ "a": s }))
        };
        let many_nullable: serde_json::Map<String, Value> = (0..17)
            .map(|i| (format!("f{i}"), json!({"type": ["string", "null"]})))
            .collect();
        let sixteen_nullable: serde_json::Map<String, Value> =
            many_nullable.clone().into_iter().take(16).collect();
        let cases = [
            (
                "passing",
                obj(json!({
                    "n": {"type": "integer"},
                    "tags": {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}},
                    "inner": obj(json!({"x": {"type": ["boolean", "null"]}})),
                    "list": {"type": ["array", "null"],
                             "items": obj(json!({"y": {"type": "number"}}))}
                })),
                None,
            ),
            ("16 nullable", obj(Value::Object(sixteen_nullable)), None),
            (
                "root not an object",
                json!({"type": "array", "items": {"type": "string"}}),
                Some("root must be an object"),
            ),
            (
                "additionalProperties missing",
                json!({"type": "object", "required": ["a"],
                       "properties": {"a": {"type": "string"}}}),
                Some("$: additionalProperties"),
            ),
            (
                "additionalProperties in items",
                obj(json!({"l": {"type": "array", "items": {"type": "object",
                    "required": [], "properties": {}}}})),
                Some("$.l[]: additionalProperties"),
            ),
            (
                "property not required",
                json!({"type": "object", "additionalProperties": false, "required": ["a"],
                       "properties": {"a": {"type": "string"}, "b": {"type": "string"}}}),
                Some("$.b: every property must be required"),
            ),
            ("minimum", with("minimum", json!(0)), Some("minimum")),
            ("maximum", with("maximum", json!(9)), Some("maximum")),
            ("maxLength", with("maxLength", json!(9)), Some("maxLength")),
            ("maxItems", with("maxItems", json!(9)), Some("maxItems")),
            (
                "minContains",
                with("minContains", json!(1)),
                Some("minContains"),
            ),
            (
                "maxContains",
                with("maxContains", json!(9)),
                Some("maxContains"),
            ),
            (
                "patternProperties",
                with("patternProperties", json!({"^a": {"type": "string"}})),
                Some("patternProperties"),
            ),
            ("pattern", with("pattern", json!("^a")), Some("pattern")),
            ("$ref", with("$ref", json!("#/$defs/x")), Some("$ref")),
            ("$defs", with("$defs", json!({})), Some("$defs")),
            ("anyOf", with("anyOf", json!([])), Some("anyOf")),
            ("oneOf", with("oneOf", json!([])), Some("oneOf")),
            ("allOf", with("allOf", json!([])), Some("allOf")),
            ("not", with("not", json!({})), Some("not")),
            ("if", with("if", json!({})), Some("if")),
            (
                "type union",
                obj(json!({"a": {"type": ["string", "integer"]}})),
                Some("type must be"),
            ),
            (
                "three-way nullable",
                obj(json!({"a": {"type": ["string", "integer", "null"]}})),
                Some("type must be"),
            ),
            (
                "untyped",
                obj(json!({"a": {"enum": ["x"]}})),
                Some("type must be"),
            ),
            (
                "17 nullable",
                obj(Value::Object(many_nullable)),
                Some("17 nullable"),
            ),
        ];
        for (name, schema, want) in cases {
            match (portable(&schema), want) {
                (Ok(()), None) => {}
                (Err(e), Some(want)) => assert!(e.contains(want), "{name}: {e}"),
                (got, _) => panic!("{name}: {got:?}"),
            }
        }
    }
}
