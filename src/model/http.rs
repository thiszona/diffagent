//! The HTTP plumbing the two reqwest wires (OpenAI Chat Completions, Anthropic
//! Messages) share: one client, one retrying POST, one error shape.

use std::error::Error as _;
use std::time::Duration;

use reqwest::Client;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue, RETRY_AFTER};

use super::ModelError;

/// The longest provider error message an `Http` error keeps.
const MAX_MESSAGE: usize = 300;

/// The one client every reqwest wire shares. A 10 s connect timeout and no read
/// timeout: the Router's deadline bounds each call, and a new schema's first call
/// can spend 90–180 s compiling it before the first byte. Redirects are never
/// followed: reqwest strips only `Authorization` on a cross-host hop, so one would
/// carry Anthropic's `x-api-key` and the prompt to another host.
pub fn client() -> Result<Client, ModelError> {
    Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| ModelError::Request(chain(&e)))
}

/// POSTs a JSON `body` and returns the 2xx response body. Transient failures are
/// retried per [`retry_delay`]; any other status, or the last transient one, is
/// [`ModelError::Http`] via [`error`], with `key` redacted. Every header value is
/// marked sensitive, so the API key never shows in a `Debug` of the request.
pub async fn post(
    client: &Client,
    url: &str,
    headers: &[(&str, &str)],
    body: Vec<u8>,
    key: &str,
) -> Result<Vec<u8>, ModelError> {
    let mut map = HeaderMap::new();
    map.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    for (name, value) in headers {
        // Never echo the value: it is usually the key.
        let invalid = || ModelError::Request(format!("header {name}: invalid"));
        let name = HeaderName::from_bytes(name.as_bytes()).map_err(|_| invalid())?;
        let mut value = HeaderValue::from_str(value).map_err(|_| invalid())?;
        value.set_sensitive(true);
        map.insert(name, value);
    }
    let mut attempt = 0;
    loop {
        let sent = client
            .post(url)
            .headers(map.clone())
            .body(body.clone())
            .send()
            .await;
        let (status, retry_after, failure) = match sent {
            Ok(resp) => {
                let status = resp.status().as_u16();
                let retry_after = resp
                    .headers()
                    .get(RETRY_AFTER)
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_owned);
                match resp.bytes().await {
                    Ok(bytes) if (200..300).contains(&status) => return Ok(bytes.to_vec()),
                    Ok(bytes) => (Some(status), retry_after, error(status, &bytes, key)),
                    // The body broke off mid-read: a transport error, whatever the status.
                    Err(e) => (None, None, ModelError::Request(chain(&e))),
                }
            }
            Err(e) => (None, None, ModelError::Request(chain(&e))),
        };
        match retry_delay(status, retry_after.as_deref(), attempt) {
            Some(wait) => tokio::time::sleep(wait).await,
            None => return Err(failure),
        }
        attempt += 1;
    }
}

/// How long to wait before retrying a failed attempt, or `None` to give up.
/// `status` is `None` for a transport error; `retries` is how many retries were
/// already made. 408, 429, 500, 502, 503, 504, 529 and transport errors are retried
/// at most twice, after `Retry-After` seconds (capped at 30 s) when the provider
/// sent them, else 2 s then 8 s.
// ponytail: a spend-cap 429 is retried too (~10 s wasted).
pub fn retry_delay(
    status: Option<u16>,
    retry_after: Option<&str>,
    retries: u32,
) -> Option<Duration> {
    let transient = status.is_none_or(|s| matches!(s, 408 | 429 | 500 | 502 | 503 | 504 | 529));
    let fallback = match retries {
        0 => 2,
        1 => 8,
        _ => return None,
    };
    if !transient {
        return None;
    }
    let secs = retry_after
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map_or(fallback, |s| s.min(30));
    Some(Duration::from_secs(secs))
}

/// A non-2xx response as [`ModelError::Http`]. Both OpenAI-compatible and Anthropic
/// error bodies carry `error.message`; anything else keeps the raw body. Either way
/// the message is cut to 300 chars, since a provider can echo a whole prompt back,
/// after the API `key` (empty for Bedrock, which has none) is replaced by `***`: a
/// provider can echo it too, and a cut through it would leave a prefix behind.
pub fn error(status: u16, body: &[u8], key: &str) -> ModelError {
    let parsed: Option<serde_json::Value> = serde_json::from_slice(body).ok();
    let message = parsed
        .as_ref()
        .and_then(|v| v.pointer("/error/message").or_else(|| v.get("message")))
        .and_then(serde_json::Value::as_str)
        .map_or_else(|| String::from_utf8_lossy(body).into_owned(), str::to_owned);
    let message = if key.is_empty() {
        message
    } else {
        message.replace(key, "***")
    };
    ModelError::Http {
        status,
        message: cut(&message, MAX_MESSAGE),
    }
}

/// `s` cut to at most `max` chars.
pub(crate) fn cut(s: &str, max: usize) -> String {
    s.char_indices()
        .nth(max)
        .map_or(s, |(i, _)| &s[..i])
        .to_owned()
}

/// An error with its sources, since reqwest's own message is only the outer layer
/// ("error sending request") and the cause (DNS, TLS, reset) is in the chain.
pub(crate) fn chain(e: &reqwest::Error) -> String {
    let mut out = e.to_string();
    let mut source = e.source();
    while let Some(s) = source {
        out.push_str(": ");
        out.push_str(&s.to_string());
        source = s.source();
    }
    out
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    use super::*;

    #[test]
    fn retry_delays() {
        let secs = |s| Some(Duration::from_secs(s));
        let cases = [
            // (status, Retry-After, retries made, want)
            (None, None, 0, secs(2)),
            (None, None, 1, secs(8)),
            (None, None, 2, None),
            (Some(429), None, 0, secs(2)),
            (Some(429), Some("5"), 0, secs(5)),
            (Some(429), Some(" 7 "), 1, secs(7)),
            (Some(429), Some("120"), 0, secs(30)),
            (Some(429), Some("0"), 0, secs(0)),
            (Some(429), Some("Wed, 21 Oct 2026 07:28:00 GMT"), 0, secs(2)),
            (Some(429), Some("5"), 2, None),
            (Some(408), None, 0, secs(2)),
            (Some(500), None, 1, secs(8)),
            (Some(502), None, 0, secs(2)),
            (Some(503), None, 0, secs(2)),
            (Some(504), None, 0, secs(2)),
            (Some(529), None, 0, secs(2)),
            (Some(400), None, 0, None),
            (Some(401), Some("5"), 0, None),
            (Some(403), None, 0, None),
            (Some(404), None, 0, None),
            (Some(501), None, 0, None),
        ];
        for (status, retry_after, retries, want) in cases {
            assert_eq!(
                retry_delay(status, retry_after, retries),
                want,
                "{status:?} {retry_after:?} after {retries} retries"
            );
        }
    }

    /// A 3xx is an error, never followed: reqwest strips only `Authorization` on a
    /// cross-host hop, so a redirect would carry `x-api-key` and the prompt along.
    #[tokio::test]
    async fn redirects_not_followed() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let mut s = stream.unwrap();
                let mut req = Vec::new();
                let mut buf = [0; 1024];
                while !req.ends_with(b"\r\n\r\n{}") {
                    let n = s.read(&mut buf).unwrap();
                    if n == 0 {
                        break;
                    }
                    req.extend_from_slice(&buf[..n]);
                }
                let resp = if req.starts_with(b"POST /moved ") {
                    "HTTP/1.1 200 OK\r\nconnection: close\r\ncontent-length: 2\r\n\r\n{}".to_owned()
                } else {
                    format!(
                        "HTTP/1.1 308 Permanent Redirect\r\nlocation: http://{addr}/moved\r\n\
                         connection: close\r\ncontent-length: 0\r\n\r\n"
                    )
                };
                s.write_all(resp.as_bytes()).unwrap();
            }
        });
        let url = format!("http://{addr}/v1/messages");
        let got = post(
            &client().unwrap(),
            &url,
            &[("x-api-key", "k")],
            b"{}".to_vec(),
            "k",
        )
        .await;
        assert_eq!(
            got,
            Err(ModelError::Http {
                status: 308,
                message: String::new()
            })
        );
    }

    #[test]
    fn error_bodies() {
        let long = "é".repeat(400);
        let cases = [
            (
                r#"{"error":{"message":"Invalid schema","type":"invalid_request_error"}}"#
                    .to_owned(),
                "Invalid schema".to_owned(),
            ),
            (
                r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#
                    .to_owned(),
                "Overloaded".to_owned(),
            ),
            (r#"{"message":"Bad key"}"#.to_owned(), "Bad key".to_owned()),
            ("upstream timeout".to_owned(), "upstream timeout".to_owned()),
            (long.clone(), "é".repeat(300)),
        ];
        for (body, want) in cases {
            assert_eq!(
                error(400, body.as_bytes(), ""),
                ModelError::Http {
                    status: 400,
                    message: want
                },
                "{body}"
            );
        }
    }

    /// A provider can echo the key back; it is redacted before the 300-char cut, so a
    /// key the cut crosses leaves no prefix behind either.
    #[test]
    fn the_key_is_redacted_before_the_cut() {
        let key = "keykeykeykeykeykeykey";
        let echoed = format!("{}{key} is not a valid key", "x".repeat(MAX_MESSAGE - 10));
        let body = serde_json::json!({"error": {"message": echoed}}).to_string();
        let ModelError::Http { message, .. } = error(401, body.as_bytes(), key) else {
            panic!("not Http");
        };
        assert!(!message.contains(&key[..10]), "{message}");
        assert_eq!(
            message,
            format!("{}*** is not", "x".repeat(MAX_MESSAGE - 10))
        );
        // Bedrock has no key: nothing is replaced.
        assert_eq!(
            error(400, b"key", ""),
            ModelError::Http {
                status: 400,
                message: "key".to_owned()
            }
        );
    }
}
