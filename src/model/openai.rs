//! The OpenAI Chat Completions wire: `POST {base_url}/chat/completions`, for OpenAI
//! and every OpenAI-compatible provider (zai, deepseek, openrouter, and any
//! `LLM_PROVIDER_<NAME>_BASE_URL`). Only OpenAI gets the provider's own schema
//! output; the others get JSON mode with the schema in the system message, and
//! `decode` plus the Router's one retry check the rest.

use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{Model, ModelError, Reply, Request, Stop, Tokens, http};
use crate::config::Secret;

/// One model on one OpenAI-compatible endpoint. Debug is safe to log: the key is a
/// [`Secret`] and travels only in the `Authorization` header.
#[derive(Debug)]
pub struct OpenAiModel {
    http: Client,
    url: String,
    api_key: Secret,
    provider: String,
    model: String,
}

impl OpenAiModel {
    /// `http` is the process's one client ([`http::client`]); `base_url` and
    /// `api_key` come from the route's `Wire::OpenAiChat`, `provider` and `model`
    /// from the route itself.
    pub fn new(http: Client, base_url: &str, api_key: Secret, provider: &str, model: &str) -> Self {
        Self {
            http,
            url: format!("{}/chat/completions", base_url.trim_end_matches('/')),
            api_key,
            provider: provider.to_owned(),
            model: model.to_owned(),
        }
    }
}

#[async_trait]
impl Model for OpenAiModel {
    async fn complete(&self, req: &Request<'_>) -> Result<Reply, ModelError> {
        let auth = format!("Bearer {}", self.api_key.expose());
        let headers = [("authorization", auth.as_str())];
        let body = body(&self.provider, &self.model, req)
            .to_string()
            .into_bytes();
        reply(&http::post(&self.http, &self.url, &headers, body, self.api_key.expose()).await?)
    }
}

/// The request: one system and one user message, the output budget, and the schema
/// when there is one. No sampling keys. OpenAI gets strict `json_schema` output and
/// `max_completion_tokens`, since its reasoning models reject `max_tokens` (strict
/// schemas on those models are unverified). Every other provider gets `json_object`
/// mode, which promises only some JSON object, so the schema is appended to the
/// system message; that text also carries the word "JSON", which JSON mode requires.
fn body(provider: &str, model: &str, req: &Request<'_>) -> Value {
    let native = provider == "openai";
    let (system, format) = match req.schema {
        None => (req.system.to_owned(), None),
        Some(s) if native => (
            req.system.to_owned(),
            Some(json!({
                "type": "json_schema",
                "json_schema": {"name": s.name, "strict": true, "schema": s.json},
            })),
        ),
        Some(s) => (
            format!(
                "{}\n\nReply with only a JSON object matching this JSON schema:\n{}",
                req.system, s.json
            ),
            Some(json!({"type": "json_object"})),
        ),
    };
    let mut body = json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": req.user},
        ],
    });
    let budget = if native {
        "max_completion_tokens"
    } else {
        "max_tokens"
    };
    body[budget] = req.max_tokens.into();
    if let Some(format) = format {
        body["response_format"] = format;
    }
    body
}

/// A 2xx response body as a [`Reply`]: the first choice's `content`, with any
/// `reasoning_content` dropped. Error statuses never get here, since
/// [`http::post`] turns them into `Http` errors.
fn reply(body: &[u8]) -> Result<Reply, ModelError> {
    let completion: Completion = serde_json::from_slice(body)
        .map_err(|e| ModelError::Invalid(format!("not a chat completion: {e}")))?;
    let choice = completion
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| ModelError::Invalid("a chat completion with no choices".to_owned()))?;
    Ok(Reply {
        stop: stop(&choice),
        text: choice.message.content.unwrap_or_default(),
        tokens: tokens(&completion.usage.unwrap_or_default()),
    })
}

/// `length` is `MaxTokens`; `stop` and `tool_calls` are `End`. A refusal (which
/// comes with `stop`), `content_filter`, Z.ai's `sensitive` and anything else are
/// `Other`.
fn stop(choice: &Choice) -> Stop {
    if choice
        .message
        .refusal
        .as_deref()
        .is_some_and(|r| !r.is_empty())
    {
        return Stop::Other("refusal".to_owned());
    }
    match choice.finish_reason.as_deref() {
        Some("stop" | "tool_calls") => Stop::End,
        Some("length") => Stop::MaxTokens,
        Some(other) => Stop::Other(other.to_owned()),
        None => Stop::Other("no finish_reason".to_owned()),
    }
}

/// Usage with `input` uncached, as on every wire: `prompt_tokens` is taken to count
/// cache reads and writes too, so both come off it.
// ponytail: inferred from the Responses docs; Chat Completions doesn't state it.
fn tokens(u: &Usage) -> Tokens {
    let details = u.prompt_tokens_details.as_ref();
    let cache_read = details.and_then(|d| d.cached_tokens).unwrap_or(0);
    let cache_write = details.and_then(|d| d.cache_write_tokens).unwrap_or(0);
    Tokens {
        input: u
            .prompt_tokens
            .unwrap_or(0)
            .saturating_sub(cache_read)
            .saturating_sub(cache_write)
            .max(0),
        output: u.completion_tokens.unwrap_or(0),
        cache_read,
        cache_write,
    }
}

/// The parts of a chat completion the Router uses; the rest is ignored. Usage
/// fields are optional, since compatible providers differ in what they send.
#[derive(Deserialize)]
struct Completion {
    choices: Vec<Choice>,
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct Choice {
    #[serde(default)]
    message: Message,
    finish_reason: Option<String>,
}

#[derive(Default, Deserialize)]
struct Message {
    content: Option<String>,
    refusal: Option<String>,
}

#[derive(Default, Deserialize)]
struct Usage {
    prompt_tokens: Option<i64>,
    completion_tokens: Option<i64>,
    prompt_tokens_details: Option<PromptDetails>,
}

#[derive(Deserialize)]
struct PromptDetails {
    cached_tokens: Option<i64>,
    cache_write_tokens: Option<i64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tests::{req, small_schema};

    #[test]
    fn request_bodies() {
        let schema = small_schema();
        let messages = |system: &str| {
            json!([
                {"role": "system", "content": system},
                {"role": "user", "content": "review this"},
            ])
        };
        let in_prompt = format!(
            "sys\n\nReply with only a JSON object matching this JSON schema:\n{}",
            schema.json
        );
        // response_format shapes from
        // https://developers.openai.com/api/docs/guides/structured-outputs and
        // https://api-docs.deepseek.com/guides/json_mode
        let openai_schema = json!({
            "model": "m",
            "messages": messages("sys"),
            "max_completion_tokens": 4000,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "small", "strict": true, "schema": schema.json},
            },
        });
        let openai_prose = json!({
            "model": "m",
            "messages": messages("sys"),
            "max_completion_tokens": 4000,
        });
        let compat_schema = json!({
            "model": "m",
            "messages": messages(&in_prompt),
            "max_tokens": 4000,
            "response_format": {"type": "json_object"},
        });
        let compat_prose = json!({"model": "m", "messages": messages("sys"), "max_tokens": 4000});
        let mut cases = vec![
            ("openai", req(Some(&schema)), openai_schema),
            ("openai", req(None), openai_prose),
        ];
        for provider in ["zai", "deepseek", "openrouter", "my-llm"] {
            cases.push((provider, req(Some(&schema)), compat_schema.clone()));
            cases.push((provider, req(None), compat_prose.clone()));
        }
        for (provider, req, want) in cases {
            let name = format!("{provider}, schema: {}", req.schema.is_some());
            let got = body(provider, "m", &req);
            for key in ["temperature", "top_p", "top_k"] {
                assert!(got.get(key).is_none(), "{name}: {key} sent");
            }
            assert_eq!(got, want, "{name}");
        }
        assert!(in_prompt.contains("JSON"), "JSON mode needs the word");
    }

    /// A completion envelope as in the DeepSeek example below, with the given
    /// message and finish reason.
    fn completion(message: Value, finish_reason: Value) -> Vec<u8> {
        json!({
            "id": "930c60df-bf64-41c9-a88e-3ec75f81e00e",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 16, "completion_tokens": 10, "total_tokens": 26},
        })
        .to_string()
        .into_bytes()
    }

    /// Finish reasons from openai-python's `Choice.finish_reason`
    /// (https://github.com/openai/openai-python/blob/main/src/openai/types/chat/chat_completion.py),
    /// https://docs.z.ai/api-reference/llm/chat-completion and
    /// https://api-docs.deepseek.com/api/create-chat-completion.
    #[test]
    fn stops() {
        let other = |s: &str| Stop::Other(s.to_owned());
        let text = json!({"role": "assistant", "content": "x"});
        // Text from https://developers.openai.com/api/docs/guides/structured-outputs#refusals,
        // field from openai-python's `ChatCompletionMessage.refusal`.
        let refusal = json!({"role": "assistant",
                             "refusal": "I'm sorry, I cannot assist with that request."});
        let cases = [
            (text.clone(), json!("stop"), Stop::End),
            (text.clone(), json!("tool_calls"), Stop::End),
            (text.clone(), json!("length"), Stop::MaxTokens),
            (
                text.clone(),
                json!("content_filter"),
                other("content_filter"),
            ),
            (text.clone(), json!("sensitive"), other("sensitive")),
            (
                text.clone(),
                json!("model_context_window_exceeded"),
                other("model_context_window_exceeded"),
            ),
            (
                text.clone(),
                json!("insufficient_system_resource"),
                other("insufficient_system_resource"),
            ),
            (text.clone(), json!("function_call"), other("function_call")),
            (text, Value::Null, other("no finish_reason")),
            (refusal, json!("stop"), other("refusal")),
        ];
        for (message, finish_reason, want) in cases {
            let name = format!("{message} {finish_reason}");
            let got = reply(&completion(message, finish_reason)).map(|r| r.stop);
            assert_eq!(got, Ok(want), "{name}");
        }
    }

    #[test]
    fn replies() {
        // Verbatim from https://api-docs.deepseek.com/api/create-chat-completion
        let deepseek = r#"{
          "id": "930c60df-bf64-41c9-a88e-3ec75f81e00e",
          "choices": [
            {
              "finish_reason": "stop",
              "index": 0,
              "message": {
                "content": "Hello! How can I help you today?",
                "role": "assistant"
              },
              "logprobs": null
            }
          ],
          "created": 1705651092,
          "model": "deepseek-flash",
          "object": "chat.completion",
          "system_fingerprint": "fp_7a09fdf9c2",
          "usage": {
            "completion_tokens": 10,
            "prompt_tokens": 16,
            "total_tokens": 26,
            "prompt_tokens_details": {
              "cached_tokens": 0
            },
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 16
          }
        }"#;
        let want = Reply {
            text: "Hello! How can I help you today?".to_owned(),
            stop: Stop::End,
            tokens: Tokens {
                input: 16,
                output: 10,
                ..Tokens::default()
            },
        };
        assert_eq!(reply(deepseek.as_bytes()), Ok(want));

        // DeepSeek's and Z.ai's thinking models put reasoning in `reasoning_content`.
        let thinking = json!({"role": "assistant", "reasoning_content": "Let me think...",
                              "content": "{\"n\":1}"});
        let got = reply(&completion(thinking, json!("stop"))).map(|r| r.text);
        assert_eq!(got, Ok(r#"{"n":1}"#.to_owned()));

        let refusal = json!({"role": "assistant", "content": null, "refusal": "No."});
        let got = reply(&completion(refusal, json!("stop"))).map(|r| r.text);
        assert_eq!(got, Ok(String::new()), "null content is empty text");

        for bad in ["", "{}", "not json", r#"{"choices":[]}"#] {
            assert!(
                matches!(reply(bad.as_bytes()), Err(ModelError::Invalid(_))),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn usage_normalized() {
        let cases = [
            (
                // Numbers from https://developers.openai.com/api/docs/guides/prompt-caching:
                // "1,000 input tokens are ordinary input: 12000 - 4000 - 7000".
                // Field names from openai-python's `CompletionUsage`.
                json!({"prompt_tokens": 12000, "completion_tokens": 300,
                       "prompt_tokens_details": {"cached_tokens": 4000,
                                                 "cache_write_tokens": 7000}}),
                Tokens {
                    input: 1000,
                    output: 300,
                    cache_read: 4000,
                    cache_write: 7000,
                },
            ),
            (
                // From https://openrouter.ai/docs/use-cases/usage-accounting
                json!({"completion_tokens": 2,
                       "completion_tokens_details": {"reasoning_tokens": 0},
                       "cost": 0.95, "cost_details": {"upstream_inference_cost": 19},
                       "prompt_tokens": 194,
                       "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 100,
                                                 "audio_tokens": 0},
                       "total_tokens": 196}),
                Tokens {
                    input: 94,
                    output: 2,
                    cache_read: 0,
                    cache_write: 100,
                },
            ),
            (
                // Z.ai sends only `cached_tokens`.
                json!({"prompt_tokens": 100, "completion_tokens": 5,
                       "prompt_tokens_details": {"cached_tokens": 60}}),
                Tokens {
                    input: 40,
                    output: 5,
                    cache_read: 60,
                    cache_write: 0,
                },
            ),
            (
                json!({"prompt_tokens": 10, "completion_tokens": 5,
                       "prompt_tokens_details": null}),
                Tokens {
                    input: 10,
                    output: 5,
                    ..Tokens::default()
                },
            ),
            (
                json!({"prompt_tokens": 10, "completion_tokens": 5,
                       "prompt_tokens_details": {"cached_tokens": 30}}),
                Tokens {
                    input: 0,
                    output: 5,
                    cache_read: 30,
                    cache_write: 0,
                },
            ),
            (Value::Null, Tokens::default()),
        ];
        for (usage, want) in cases {
            let body = json!({
                "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
                "usage": usage,
            });
            let got = reply(body.to_string().as_bytes()).map(|r| r.tokens);
            assert_eq!(got, Ok(want), "{usage}");
        }
    }

    /// An error status never reaches [`reply`]: `http::post` maps it with
    /// `http::error`, which reads the OpenAI error shape.
    #[test]
    fn error_body() {
        // Message and fields from https://developers.openai.com/api/docs/guides/error-codes
        let body = r#"{
            "error": {
                "message": "Invalid `service_tier` argument: The requested service tier is not allowed for this project.",
                "type": "invalid_request_error",
                "param": "service_tier",
                "code": null
            }
        }"#;
        let want = "Invalid `service_tier` argument: The requested service tier is not \
                    allowed for this project.";
        assert_eq!(
            http::error(400, body.as_bytes(), ""),
            ModelError::Http {
                status: 400,
                message: want.to_owned()
            }
        );
    }

    #[test]
    fn key_never_shown() {
        let key = "sk-proj-test-only-key";
        let m = OpenAiModel::new(
            http::client().unwrap(),
            "https://api.openai.com/v1/",
            Secret::new(key),
            "openai",
            "gpt-5",
        );
        assert_eq!(m.url, "https://api.openai.com/v1/chat/completions");
        let debug = format!("{m:?}");
        assert!(!debug.contains(key), "{debug}");
        let err = http::error(
            401,
            br#"{"error":{"message":"Incorrect API key provided"}}"#,
            key,
        );
        assert!(!format!("{err} {err:?}").contains(key));
    }
}
