//! The Bedrock Converse wire, through the AWS SDK: credentials come from the IAM chain
//! or `AWS_BEARER_TOKEN_BEDROCK` (both read by the SDK), so no key passes through here.
//! Retries are the SDK's standard retryer (3 attempts); the Router's deadline bounds
//! the whole call.

use async_trait::async_trait;
use aws_config::{BehaviorVersion, Region};
use aws_sdk_bedrockruntime::Client;
use aws_sdk_bedrockruntime::error::{
    BuildError, DisplayErrorContext, ProvideErrorMetadata, SdkError,
};
use aws_sdk_bedrockruntime::operation::converse::builders::ConverseFluentBuilder;
use aws_sdk_bedrockruntime::operation::converse::{ConverseError, ConverseOutput};
use aws_sdk_bedrockruntime::types::{
    CachePointBlock, CachePointType, ContentBlock, ConversationRole, InferenceConfiguration,
    JsonSchemaDefinition, Message, OutputConfig, OutputFormat, OutputFormatStructure,
    OutputFormatType, StopReason, SystemContentBlock, TokenUsage,
};

use super::{Model, ModelError, Reply, Request, Stop, Tokens, http};

/// The one SDK client every Bedrock route shares, built once per process. Without
/// `region` the SDK's own region chain applies.
pub async fn client(region: Option<&str>) -> Client {
    let mut loader = aws_config::defaults(BehaviorVersion::latest());
    if let Some(region) = region {
        loader = loader.region(Region::new(region.to_owned()));
    }
    Client::new(&loader.load().await)
}

/// One Bedrock model (a model id or inference profile) over the shared SDK client.
#[derive(Debug, Clone)]
pub struct BedrockModel {
    client: Client,
    model_id: String,
}

impl BedrockModel {
    pub fn new(client: Client, model_id: impl Into<String>) -> Self {
        Self {
            client,
            model_id: model_id.into(),
        }
    }
}

#[async_trait]
impl Model for BedrockModel {
    async fn complete(&self, req: &Request<'_>) -> Result<Reply, ModelError> {
        let out = request(self.client.converse().model_id(&self.model_id), req)?
            .send()
            .await
            .map_err(error)?;
        Ok(reply(&out))
    }
}

/// `req` as a Converse call: one system text block, one user message, `max_tokens`
/// only (temperature, top_p and top_k are never sent; nothing configures them), and
/// the schema as Converse's native JSON-schema output.
///
/// The prompt cache (5-minute default TTL) gets the Anthropic wire's two breakpoints:
/// a cache point after the system prompt, and one after the user message's shared
/// head, which then is its own text block.
fn request(
    b: ConverseFluentBuilder,
    req: &Request<'_>,
) -> Result<ConverseFluentBuilder, ModelError> {
    let point = CachePointBlock::builder()
        .r#type(CachePointType::Default)
        .build()
        .map_err(built)?;
    let user = Message::builder().role(ConversationRole::User);
    let user = match req.split_user() {
        Some((head, tail)) => user
            .content(ContentBlock::Text(head.to_owned()))
            .content(ContentBlock::CachePoint(point.clone()))
            .content(ContentBlock::Text(tail.to_owned())),
        None => user.content(ContentBlock::Text(req.user.to_owned())),
    };
    let user = user.build().map_err(built)?;
    let max_tokens = i32::try_from(req.max_tokens).unwrap_or(i32::MAX);
    let b = b
        .system(SystemContentBlock::Text(req.system.to_owned()))
        .system(SystemContentBlock::CachePoint(point))
        .messages(user)
        .inference_config(
            InferenceConfiguration::builder()
                .max_tokens(max_tokens)
                .build(),
        );
    let Some(schema) = req.schema else {
        return Ok(b);
    };
    let definition = JsonSchemaDefinition::builder()
        .name(schema.name)
        .schema(schema.json.to_string())
        .build()
        .map_err(built)?;
    let format = OutputFormat::builder()
        .r#type(OutputFormatType::JsonSchema)
        .structure(OutputFormatStructure::JsonSchema(definition))
        .build()
        .map_err(built)?;
    Ok(b.output_config(OutputConfig::builder().text_format(format).build()))
}

/// A required SDK field left unset: a bug here, never the model's doing.
fn built(e: BuildError) -> ModelError {
    ModelError::Request(format!("building the Converse request: {e}"))
}

/// The reply's text blocks joined, reasoning dropped. A reply with no message is
/// empty text, so it is still metered and a JSON call still gets its retry.
fn reply(out: &ConverseOutput) -> Reply {
    let text = out
        .output()
        .and_then(|o| o.as_message().ok())
        .map(|m| {
            m.content()
                .iter()
                .filter_map(|b| b.as_text().ok())
                .map(String::as_str)
                .collect()
        })
        .unwrap_or_default();
    Reply {
        text,
        stop: stop(out.stop_reason()),
        tokens: tokens(out.usage()),
    }
}

fn stop(reason: &StopReason) -> Stop {
    match reason {
        StopReason::EndTurn | StopReason::StopSequence => Stop::End,
        StopReason::MaxTokens => Stop::MaxTokens,
        // Output that broke the schema: decoding fails, so the call gets its retry.
        StopReason::MalformedModelOutput | StopReason::MalformedToolUse => Stop::End,
        other => Stop::Other(other.as_str().to_owned()),
    }
}

/// Converse's `inputTokens` already excludes cache reads and writes: "total input
/// tokens = inputTokens + cacheReadInputTokens + cacheWriteInputTokens"
/// (<https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html>).
fn tokens(usage: Option<&TokenUsage>) -> Tokens {
    usage.map_or_else(Tokens::default, |u| Tokens {
        input: u.input_tokens().into(),
        output: u.output_tokens().into(),
        cache_read: u.cache_read_input_tokens().unwrap_or(0).into(),
        cache_write: u.cache_write_input_tokens().unwrap_or(0).into(),
    })
}

/// An error Bedrock answered (throttling, validation, access) is `Http` with the
/// body's `message`, like the other wires, after its code (`ValidationException`),
/// which Bedrock sends outside the body; anything else (connection, timeout,
/// credentials) is `Request` with the SDK's full error chain.
fn error(e: SdkError<ConverseError>) -> ModelError {
    let (Some(raw), Some(service)) = (e.raw_response(), e.as_service_error()) else {
        return ModelError::Request(DisplayErrorContext(&e).to_string());
    };
    let body = raw.body().bytes().unwrap_or_default();
    match (http::error(raw.status().as_u16(), body, ""), service.code()) {
        (ModelError::Http { status, message }, Some(code)) => ModelError::Http {
            status,
            message: format!("{code}: {message}"),
        },
        (other, _) => other,
    }
}

#[cfg(test)]
mod tests {
    use aws_sdk_bedrockruntime::config::Token;
    use aws_sdk_bedrockruntime::config::http::HttpResponse;
    use aws_sdk_bedrockruntime::error::ErrorMetadata;
    use aws_sdk_bedrockruntime::types::error::{ThrottlingException, ValidationException};
    use aws_sdk_bedrockruntime::types::{ReasoningContentBlock, ReasoningTextBlock};
    use serde_json::Value;

    use super::*;
    use crate::model::tests::small_schema;

    /// A fake Bedrock API key, in the shape the console issues.
    const KEY: &str = "ABSKQmVkcm9ja0FQSUtleS1mYWtlLXRlc3Qta2V5";

    /// A client that would authenticate with `KEY`; no test ever sends with it.
    fn offline() -> Client {
        let conf = aws_sdk_bedrockruntime::Config::builder()
            .behavior_version(BehaviorVersion::latest())
            .region(Region::new("us-east-1"))
            .bearer_token(Token::new(KEY, None))
            .build();
        Client::from_conf(conf)
    }

    /// A given region wins over whatever the machine's chain would pick. Unset is not
    /// tested: the chain ends at instance metadata, a network call.
    #[tokio::test]
    async fn client_keeps_a_given_region() {
        let c = client(Some("eu-west-1")).await;
        assert_eq!(c.config().region(), Some(&Region::new("eu-west-1")));
    }

    #[test]
    fn request_shape() {
        let schema = small_schema();
        let point = CachePointBlock::builder()
            .r#type(CachePointType::Default)
            .build()
            .unwrap();
        let text = |s: &str| ContentBlock::Text(s.to_owned());
        // (schema, max_tokens, max_tokens sent, shared, the user message's blocks)
        let cases = [
            (
                Some(&schema),
                4000,
                4000,
                0,
                vec![text("diff --git a/x.go b/x.go")],
            ),
            (
                None,
                u32::MAX,
                i32::MAX,
                5,
                vec![
                    text("diff "),
                    ContentBlock::CachePoint(point.clone()),
                    text("--git a/x.go b/x.go"),
                ],
            ),
        ];
        for (schema, max_tokens, sent, shared, blocks) in cases {
            let req = Request {
                system: "You review diffs.",
                user: "diff --git a/x.go b/x.go",
                shared,
                schema,
                max_tokens,
            };
            let b = offline()
                .converse()
                .model_id("global.anthropic.claude-haiku-5-5");
            let input = request(b, &req).unwrap();
            let input = input.as_input();
            assert_eq!(
                input.get_system(),
                &Some(vec![
                    SystemContentBlock::Text("You review diffs.".to_owned()),
                    SystemContentBlock::CachePoint(point.clone()),
                ])
            );
            let user = Message::builder()
                .role(ConversationRole::User)
                .set_content(Some(blocks))
                .build()
                .unwrap();
            assert_eq!(input.get_messages(), &Some(vec![user]));
            // Exactly max_tokens: no temperature, top_p or stop sequences. top_k could
            // only travel in additionalModelRequestFields, and a forced tool in toolConfig.
            assert_eq!(
                input.get_inference_config(),
                &Some(InferenceConfiguration::builder().max_tokens(sent).build())
            );
            assert_eq!(input.get_additional_model_request_fields(), &None);
            assert_eq!(input.get_tool_config(), &None);
            let Some(schema) = schema else {
                assert_eq!(input.get_output_config(), &None);
                continue;
            };
            let format = input
                .get_output_config()
                .as_ref()
                .and_then(OutputConfig::text_format)
                .unwrap();
            assert_eq!(format.r#type(), &OutputFormatType::JsonSchema);
            let definition = format.structure().unwrap().as_json_schema().unwrap();
            assert_eq!(definition.name(), Some("small"));
            let sent: Value = serde_json::from_str(definition.schema()).unwrap();
            assert_eq!(sent, schema.json);
        }
    }

    fn output(
        content: Vec<ContentBlock>,
        reason: StopReason,
        usage: Option<TokenUsage>,
    ) -> ConverseOutput {
        let mut message = Message::builder().role(ConversationRole::Assistant);
        for block in content {
            message = message.content(block);
        }
        let mut out = ConverseOutput::builder()
            .output(aws_sdk_bedrockruntime::types::ConverseOutput::Message(
                message.build().unwrap(),
            ))
            .stop_reason(reason);
        if let Some(usage) = usage {
            out = out.usage(usage);
        }
        out.build().unwrap()
    }

    fn usage(input: i32, output: i32, cache: Option<(i32, i32)>) -> TokenUsage {
        let mut u = TokenUsage::builder()
            .input_tokens(input)
            .output_tokens(output)
            .total_tokens(input + output + cache.map_or(0, |(r, w)| r + w));
        if let Some((read, write)) = cache {
            u = u
                .cache_read_input_tokens(read)
                .cache_write_input_tokens(write);
        }
        u.build().unwrap()
    }

    fn text(s: &str) -> ContentBlock {
        ContentBlock::Text(s.to_owned())
    }

    fn counted(input: i64, output: i64, cache_read: i64, cache_write: i64) -> Tokens {
        Tokens {
            input,
            output,
            cache_read,
            cache_write,
        }
    }

    /// The first two cases are the example responses in the Converse API reference
    /// (<https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html>);
    /// the cached one follows the usage fields of the prompt-caching guide.
    #[test]
    fn replies() {
        let reasoning = ContentBlock::ReasoningContent(ReasoningContentBlock::ReasoningText(
            ReasoningTextBlock::builder()
                .text("The user wants JSON.")
                .build()
                .unwrap(),
        ));
        let cases = [
            (
                "docs: end_turn",
                output(
                    vec![text("<text generated by the model>")],
                    StopReason::EndTurn,
                    Some(usage(30, 628, None)),
                ),
                "<text generated by the model>",
                counted(30, 628, 0, 0),
            ),
            (
                "docs: stop_sequence",
                output(
                    vec![text("<text generated by the model>")],
                    StopReason::StopSequence,
                    Some(usage(51, 442, None)),
                ),
                "<text generated by the model>",
                counted(51, 442, 0, 0),
            ),
            (
                "cached input stays out of input",
                output(
                    vec![text("{}")],
                    StopReason::EndTurn,
                    Some(usage(7, 225, Some((2870, 0)))),
                ),
                "{}",
                counted(7, 225, 2870, 0),
            ),
            (
                "reasoning dropped, text blocks joined",
                output(
                    vec![reasoning, text(r#"{"n":"#), text("1}")],
                    StopReason::EndTurn,
                    Some(usage(10, 20, Some((0, 5)))),
                ),
                r#"{"n":1}"#,
                counted(10, 20, 0, 5),
            ),
            (
                "no usage",
                output(vec![text("x")], StopReason::EndTurn, None),
                "x",
                Tokens::default(),
            ),
            (
                "no message",
                ConverseOutput::builder()
                    .stop_reason(StopReason::EndTurn)
                    .build()
                    .unwrap(),
                "",
                Tokens::default(),
            ),
        ];
        for (name, out, want_text, want_tokens) in cases {
            let got = reply(&out);
            assert_eq!(
                got,
                Reply {
                    text: want_text.to_owned(),
                    stop: Stop::End,
                    tokens: want_tokens,
                },
                "{name}"
            );
        }
    }

    /// Every `stopReason` the Converse API reference lists, plus one it may add.
    #[test]
    fn stops() {
        let other = |s: &str| Stop::Other(s.to_owned());
        let cases = [
            (StopReason::EndTurn, Stop::End),
            (StopReason::StopSequence, Stop::End),
            (StopReason::MaxTokens, Stop::MaxTokens),
            (StopReason::MalformedModelOutput, Stop::End),
            (StopReason::MalformedToolUse, Stop::End),
            (StopReason::ToolUse, other("tool_use")),
            (
                StopReason::GuardrailIntervened,
                other("guardrail_intervened"),
            ),
            (StopReason::ContentFiltered, other("content_filtered")),
            (
                StopReason::ModelContextWindowExceeded,
                other("model_context_window_exceeded"),
            ),
            (
                StopReason::from("brand_new_reason"),
                other("brand_new_reason"),
            ),
        ];
        for (reason, want) in cases {
            let got = reply(&output(vec![text("{}")], reason.clone(), None)).stop;
            assert_eq!(got, want, "{reason:?}");
        }
    }

    fn service(err: ConverseError, status: u16, body: &str) -> SdkError<ConverseError> {
        let raw = HttpResponse::new(status.try_into().unwrap(), body.to_owned().into());
        SdkError::service_error(err, raw)
    }

    /// Statuses from the Converse API reference's Errors section; Bedrock's error
    /// bodies are `{"message": ...}`, the code travels apart. A code the SDK did not
    /// parse (the throttling case) leaves the message as is.
    #[test]
    fn errors() {
        let validation = || {
            ConverseError::ValidationException(
                ValidationException::builder()
                    .message(
                        "The input fails to satisfy the constraints specified by Amazon Bedrock.",
                    )
                    .meta(ErrorMetadata::builder().code("ValidationException").build())
                    .build(),
            )
        };
        let throttling = ConverseError::ThrottlingException(
            ThrottlingException::builder()
                .message("Too many requests, please wait before trying again.")
                .build(),
        );
        let long = format!(r#"{{"message":"{}"}}"#, "x".repeat(400));
        let cases = [
            (
                service(
                    validation(),
                    400,
                    r#"{"message":"The input fails to satisfy the constraints specified by Amazon Bedrock."}"#,
                ),
                ModelError::Http {
                    status: 400,
                    message: "ValidationException: The input fails to satisfy the constraints \
                              specified by Amazon Bedrock."
                        .to_owned(),
                },
            ),
            (
                service(
                    throttling,
                    429,
                    r#"{"message":"Too many requests, please wait before trying again."}"#,
                ),
                ModelError::Http {
                    status: 429,
                    message: "Too many requests, please wait before trying again.".to_owned(),
                },
            ),
            (
                service(validation(), 400, &long),
                ModelError::Http {
                    status: 400,
                    message: format!("ValidationException: {}", "x".repeat(300)),
                },
            ),
        ];
        for (e, want) in cases {
            assert_eq!(error(e), want);
        }
        let got = error(SdkError::timeout_error("connect timed out"));
        match &got {
            ModelError::Request(m) => assert!(m.contains("connect timed out"), "{m}"),
            other => panic!("{other:?}"),
        }
        for shown in [format!("{got}"), format!("{got:?}")] {
            assert!(!shown.contains(KEY), "{shown}");
        }
    }

    #[test]
    fn key_never_printed() {
        let model = BedrockModel::new(offline(), "global.anthropic.claude-haiku-5-5");
        let shown = format!("{model:?}");
        assert!(
            shown.contains("global.anthropic.claude-haiku-5-5"),
            "{shown}"
        );
        assert!(!shown.contains(KEY), "{shown}");
    }
}
