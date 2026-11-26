//! The Anthropic Messages wire: `POST {base_url}/v1/messages` on API version
//! `2026-10-21`, with the schema in `output_config.format` (the provider's own
//! schema-constrained output). Only text blocks are the answer:
//! `thinking`, `redacted_thinking` and `fallback` blocks are dropped.

use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{Model, ModelError, Reply, Request, Stop, Tokens, http};
use crate::config::Secret;

/// The API version every request sends. It rejects `temperature`, `top_p` and
/// `top_k` outright, so none is ever sent.
const VERSION: &str = "2026-10-21";

/// The models we send `"fallbacks": "default"` for: a safety-classifier decline in a
/// category with a recommended fallback is re-run server-side on that category's
/// model (on Sonnet 5.5: `cyber` and `frontier_llm`, on Sonnet 5); any other decline
/// stays a refusal. Haiku has no server-side fallback.
/// https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback
const FALLBACK_MODELS: [&str; 4] = [
    "claude-sonnet-5-5",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-fable-5-1",
];

/// The beta that gates the `"default"` form; any other date rejects it with a 400.
const FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";

fn fallback(model: &str) -> bool {
    FALLBACK_MODELS.contains(&model)
}

/// One model over the Messages API. Debug is safe to log: the key is a [`Secret`].
#[derive(Debug)]
pub struct AnthropicModel {
    http: Client,
    url: String,
    api_key: Secret,
    model: String,
}

impl AnthropicModel {
    /// `http` is the process's one client ([`http::client`]); `base_url` and
    /// `api_key` come from the route's `Wire::AnthropicMessages`.
    pub fn new(http: Client, base_url: &str, api_key: Secret, model: &str) -> Self {
        Self {
            http,
            url: format!("{}/v1/messages", base_url.trim_end_matches('/')),
            api_key,
            model: model.to_owned(),
        }
    }

    fn headers(&self) -> Vec<(&str, &str)> {
        let mut headers = vec![
            ("x-api-key", self.api_key.expose()),
            ("anthropic-version", VERSION),
        ];
        if fallback(&self.model) {
            headers.push(("anthropic-beta", FALLBACK_BETA));
        }
        headers
    }
}

#[async_trait]
impl Model for AnthropicModel {
    async fn complete(&self, req: &Request<'_>) -> Result<Reply, ModelError> {
        let body = body(&self.model, req).to_string().into_bytes();
        let key = self.api_key.expose();
        reply(&http::post(&self.http, &self.url, &self.headers(), body, key).await?)
    }
}

/// The request: one system prompt, one user message, the output budget, the schema
/// when there is one, and server-side fallback where the model has it. Nothing
/// else: no sampling keys, no thinking config.
///
/// The prompt cache (5-minute TTL) gets two breakpoints: the system prompt, which
/// every request of its role and language sends, and the `shared` head of the user
/// message, its own text block. The bytes are the same either way; a prefix under the
/// model's minimum is just not cached.
fn body(model: &str, req: &Request<'_>) -> Value {
    let cached =
        |text: &str| json!({"type": "text", "text": text, "cache_control": {"type": "ephemeral"}});
    let content = match req.split_user() {
        Some((head, tail)) => json!([cached(head), {"type": "text", "text": tail}]),
        None => json!(req.user),
    };
    let mut body = json!({
        "model": model,
        "max_tokens": req.max_tokens,
        "system": [cached(req.system)],
        "messages": [{"role": "user", "content": content}],
    });
    if let Some(schema) = req.schema {
        body["output_config"] = json!({"format": {"type": "json_schema", "schema": schema.json}});
    }
    if fallback(model) {
        body["fallbacks"] = json!("default");
    }
    body
}

/// A 2xx response body as a [`Reply`]; error statuses never get here, since
/// [`http::post`] turns them into `Http` errors.
fn reply(body: &[u8]) -> Result<Reply, ModelError> {
    let msg: Message = serde_json::from_slice(body)
        .map_err(|e| ModelError::Invalid(format!("not a Messages response: {e}")))?;
    let text = msg
        .content
        .into_iter()
        .filter(|b| b.kind == "text")
        .map(|b| b.text)
        .collect();
    // After a fallback, top-level usage covers only the attempt that answered;
    // `iterations` has one entry per attempt, the declined ones included.
    // ponytail: every attempt is priced at the role's rate, though a fallback hop
    // bills at its own model's and a decline before output in an unbilled category
    // ("cyber", "general_harms") is free; split by entry model if refusals add up.
    let tokens = match msg.usage.iterations.as_deref() {
        Some(attempts) if !attempts.is_empty() => {
            attempts.iter().fold(Tokens::default(), |mut sum, a| {
                sum.add(a.tokens());
                sum
            })
        }
        _ => msg.usage.tokens(),
    };
    Ok(Reply {
        text,
        stop: stop(&msg.stop_reason),
        tokens,
    })
}

/// `refusal`, `pause_turn`, `model_context_window_exceeded` and anything newer are
/// `Other`; so is `tool_use`, which can't happen since no tools are sent.
fn stop(reason: &str) -> Stop {
    match reason {
        "end_turn" | "stop_sequence" => Stop::End,
        "max_tokens" => Stop::MaxTokens,
        other => Stop::Other(other.to_owned()),
    }
}

/// The parts of a Messages response the Router uses; the rest is ignored.
#[derive(Deserialize)]
struct Message {
    content: Vec<Block>,
    stop_reason: String,
    usage: Usage,
}

#[derive(Deserialize)]
struct Block {
    #[serde(rename = "type")]
    kind: String,
    /// Only text blocks have it.
    #[serde(default)]
    text: String,
}

/// Top-level usage, or one attempt's entry in `iterations`.
#[derive(Deserialize)]
struct Usage {
    input_tokens: i64,
    output_tokens: i64,
    cache_read_input_tokens: Option<i64>,
    cache_creation_input_tokens: Option<i64>,
    iterations: Option<Vec<Usage>>,
}

impl Usage {
    /// Anthropic's `input_tokens` already excludes cache reads and writes.
    fn tokens(&self) -> Tokens {
        Tokens {
            input: self.input_tokens,
            output: self.output_tokens,
            cache_read: self.cache_read_input_tokens.unwrap_or(0),
            cache_write: self.cache_creation_input_tokens.unwrap_or(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tests::{req, small_schema};

    #[test]
    fn request_body() {
        let schema = small_schema();
        // Shape from https://platform.claude.com/docs/en/build-with-claude/prompt-caching
        let cache = json!({"type": "ephemeral"});
        let plain = json!({
            "model": "claude-haiku-5-5",
            "max_tokens": 4000,
            "system": [{"type": "text", "text": "sys", "cache_control": cache}],
            "messages": [{"role": "user", "content": "review this"}],
        });
        // The head every file of a PR sends, its own block; a head that is all of the
        // message, or runs past it, is not split.
        let mut split = plain.clone();
        split["messages"][0]["content"] = json!([
            {"type": "text", "text": "review ", "cache_control": cache},
            {"type": "text", "text": "this"},
        ]);
        let shared = |n| Request {
            shared: n,
            ..req(None)
        };
        let mut structured = plain.clone();
        // Shape from https://platform.claude.com/docs/en/build-with-claude/structured-outputs
        structured["output_config"] =
            json!({"format": {"type": "json_schema", "schema": schema.json}});
        // Shape from https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback
        let mut fallback = structured.clone();
        fallback["model"] = json!("claude-sonnet-5-5");
        fallback["fallbacks"] = json!("default");
        let cases = [
            ("prose", "claude-haiku-5-5", req(None), plain.clone()),
            ("schema", "claude-haiku-5-5", req(Some(&schema)), structured),
            (
                "fallback",
                "claude-sonnet-5-5",
                req(Some(&schema)),
                fallback,
            ),
            ("shared", "claude-haiku-5-5", shared(7), split),
            (
                "shared whole",
                "claude-haiku-5-5",
                shared(11),
                plain.clone(),
            ),
            (
                "shared past the end",
                "claude-haiku-5-5",
                shared(99),
                plain.clone(),
            ),
        ];
        for (name, model, req, want) in cases {
            let got = body(model, &req);
            for key in ["temperature", "top_p", "top_k"] {
                assert!(got.get(key).is_none(), "{name}: {key} sent");
            }
            assert_eq!(got, want, "{name}");
        }
    }

    /// The beta header goes with `"fallbacks": "default"` and only with it: Haiku
    /// has no server-side fallback, and the header pairs with no other model here.
    #[test]
    fn fallback_header() {
        let cases = [
            ("claude-sonnet-5-5", true),
            ("claude-opus-5-5", true),
            ("claude-opus-5", true),
            ("claude-fable-5-1", true),
            ("claude-haiku-5-5", false),
            ("claude-sonnet-5", false),
            ("claude-sonnet-5-5-20261001", false),
        ];
        for (model, want) in cases {
            let m = AnthropicModel::new(
                http::client().unwrap(),
                "https://api.anthropic.com",
                Secret::new("k"),
                model,
            );
            let beta = m.headers().contains(&("anthropic-beta", FALLBACK_BETA));
            assert_eq!(beta, want, "{model}: header");
            let fallbacks = body(model, &req(None)).get("fallbacks").is_some();
            assert_eq!(fallbacks, want, "{model}: body");
        }
    }

    /// A response envelope as in
    /// https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons,
    /// with the given content, stop reason and usage.
    fn message(content: Value, stop_reason: &str, usage: Value) -> Vec<u8> {
        json!({
            "id": "msg_01234",
            "type": "message",
            "role": "assistant",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": null,
            "stop_details": null,
            "usage": usage,
        })
        .to_string()
        .into_bytes()
    }

    /// Every stop reason in
    /// https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons.
    #[test]
    fn stops() {
        let other = |s: &str| Stop::Other(s.to_owned());
        let cases = [
            ("end_turn", Stop::End),
            ("stop_sequence", Stop::End),
            ("max_tokens", Stop::MaxTokens),
            ("refusal", other("refusal")),
            ("pause_turn", other("pause_turn")),
            (
                "model_context_window_exceeded",
                other("model_context_window_exceeded"),
            ),
            ("tool_use", other("tool_use")),
        ];
        for (reason, want) in cases {
            assert_eq!(stop(reason), want, "{reason}");
        }
        let cut = reply(&message(
            json!([]),
            "max_tokens",
            json!({"input_tokens": 1, "output_tokens": 2}),
        ));
        assert_eq!(cut.map(|r| r.stop), Ok(Stop::MaxTokens));
        // The whole fallback chain declined: still a refusal.
        let declined = reply(&message(
            json!([{"type": "fallback", "from": {"model": "claude-sonnet-5-5"},
                    "to": {"model": "claude-sonnet-5"},
                    "trigger": {"type": "refusal", "category": "cyber"}}]),
            "refusal",
            json!({"input_tokens": 1, "output_tokens": 0}),
        ));
        assert_eq!(declined.map(|r| r.stop), Ok(other("refusal")));
    }

    #[test]
    fn replies() {
        // From https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons
        let answer = json!([{"type": "text", "text": "Here's the answer to your question..."}]);
        let usage = json!({"input_tokens": 100, "output_tokens": 50});
        let tokens = Tokens {
            input: 100,
            output: 50,
            ..Tokens::default()
        };
        // From https://platform.claude.com/docs/en/build-with-claude/thinking
        let thinking = json!([
            {"type": "thinking", "thinking": "Let me break this down...",
             "signature": "WaUjzkypQ2mUEVM36O2Txu...."},
            {"type": "text", "text": "Based on my analysis..."}
        ]);
        let split = json!([
            {"type": "redacted_thinking", "data": "EmwKAhgB..."},
            {"type": "text", "text": "{\"n\":1,"},
            {"type": "text", "text": "\"items\":[]}"}
        ]);
        // From https://platform.claude.com/docs/en/build-with-claude/prompt-caching, where
        // total input = cache_read + cache_creation + input_tokens: input is uncached.
        let cached = json!({
            "input_tokens": 2048,
            "cache_read_input_tokens": 1800,
            "cache_creation_input_tokens": 248,
            "output_tokens": 503,
            "cache_creation": {"ephemeral_5m_input_tokens": 148, "ephemeral_1h_input_tokens": 100}
        });
        let cached_tokens = Tokens {
            input: 2048,
            output: 503,
            cache_read: 1800,
            cache_write: 248,
        };
        // After https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback
        // (cache counts added): a decline before output, then the fallback's answer.
        // Top-level usage is the answering attempt only, so the attempts are summed.
        let fell_back = json!([
            {"type": "fallback", "from": {"model": "claude-fable-5"},
             "to": {"model": "claude-opus-4-8"},
             "trigger": {"type": "refusal", "category": "cyber"}},
            {"type": "text", "text": "Hi! How can I help you today?"}
        ]);
        let attempts = json!({
            "input_tokens": 412,
            "output_tokens": 264,
            "cache_read_input_tokens": 10,
            "cache_creation_input_tokens": 20,
            "iterations": [
                {"type": "message", "model": "claude-fable-5", "input_tokens": 535,
                 "output_tokens": 0, "cache_read_input_tokens": 0,
                 "cache_creation_input_tokens": 0},
                {"type": "fallback_message", "model": "claude-opus-4-8", "input_tokens": 412,
                 "output_tokens": 264, "cache_read_input_tokens": 10,
                 "cache_creation_input_tokens": 20}
            ]
        });
        let summed = Tokens {
            input: 947,
            output: 264,
            cache_read: 10,
            cache_write: 20,
        };
        let cases = [
            (
                "text",
                answer.clone(),
                usage.clone(),
                "Here's the answer to your question...",
                tokens,
            ),
            (
                "thinking dropped",
                thinking,
                usage.clone(),
                "Based on my analysis...",
                tokens,
            ),
            (
                "text blocks joined",
                split,
                usage,
                r#"{"n":1,"items":[]}"#,
                tokens,
            ),
            (
                "cache tokens",
                answer.clone(),
                cached,
                "Here's the answer to your question...",
                cached_tokens,
            ),
            (
                "fallback dropped, attempts summed",
                fell_back,
                attempts,
                "Hi! How can I help you today?",
                summed,
            ),
            (
                "null iterations",
                answer,
                json!({"input_tokens": 100, "output_tokens": 50, "iterations": null}),
                "Here's the answer to your question...",
                tokens,
            ),
        ];
        for (name, content, usage, text, tokens) in cases {
            let want = Reply {
                text: text.to_owned(),
                stop: Stop::End,
                tokens,
            };
            assert_eq!(
                reply(&message(content, "end_turn", usage)),
                Ok(want),
                "{name}"
            );
        }
        for bad in ["", "{}", r#"{"content":[],"stop_reason":"end_turn"}"#] {
            assert!(
                matches!(reply(bad.as_bytes()), Err(ModelError::Invalid(_))),
                "{bad:?}"
            );
        }
    }

    /// An error status never reaches [`reply`]: `http::post` maps it with
    /// `http::error`, which reads Anthropic's error shape.
    #[test]
    fn error_body() {
        // From https://platform.claude.com/docs/en/api/errors
        let body = r#"{
            "type": "error",
            "error": {
                "type": "not_found_error",
                "message": "The requested resource could not be found."
            },
            "request_id": "req_011CSHoEeqs5C35K2UUqR7Fy"
        }"#;
        assert_eq!(
            http::error(404, body.as_bytes(), ""),
            ModelError::Http {
                status: 404,
                message: "The requested resource could not be found.".to_owned()
            }
        );
    }

    #[test]
    fn key_never_shown() {
        let key = "sk-ant-api03-test-only-key";
        let m = AnthropicModel::new(
            http::client().unwrap(),
            "https://api.anthropic.com/",
            Secret::new(key),
            "claude-opus-5-5",
        );
        assert_eq!(m.url, "https://api.anthropic.com/v1/messages");
        let debug = format!("{m:?}");
        assert!(!debug.contains(key), "{debug}");
    }
}
