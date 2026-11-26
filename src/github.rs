//! The GitHub App client: the App JWT, cached installation tokens, and
//! one `send` that holds every retry rule. It is built on [`http::client`], which
//! follows no redirects, so a token never travels to another host.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue, USER_AGENT};
use reqwest::{Client, Method};
use ring::rand::SystemRandom;
use ring::signature::{RSA_PKCS1_SHA256, RsaKeyPair};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;
use tokio::time::Instant;

use crate::config::Secret;
use crate::model::http::{self, chain, cut};

#[cfg(test)]
pub(crate) mod fake;

/// GitHub's REST and GraphQL root.
pub const API: &str = "https://api.github.com";

const API_VERSION: &str = "2022-11-28";

/// Bounds one send, from connect to the last body byte.
const TIMEOUT: Duration = Duration::from_secs(30);

/// Sends per request: the first, then at most two more.
const MAX_SENDS: u32 = 3;

/// The longest rate-limit wait a request sits out; a longer one is the job's to retry.
const MAX_RATE_WAIT: Duration = Duration::from_secs(60);

/// A token is replaced this long before it expires, so none dies mid-request.
const TOKEN_MARGIN: Duration = Duration::from_secs(5 * 60);

/// GitHub's tokens live an hour; the cap keeps a bogus `expires_at` off the clock's edge.
const TOKEN_LIFE: Duration = Duration::from_secs(60 * 60);

const PER_PAGE: usize = 100;

/// A reply read whole: every call but those that name a cap.
const UNCAPPED: usize = usize::MAX;

/// 3,000 items, GitHub's own cap on a PR's files.
const MAX_PAGES: usize = 30;

/// The longest error text kept: a body can echo the whole request back.
const MAX_MESSAGE: usize = 300;

/// Why a GitHub call failed. Texts keep the status and at most 300 chars of the body,
/// with the token or JWT replaced, so every error can be logged.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GhError {
    #[error("github: not found")]
    NotFound,
    /// A 422: GitHub refused the content, such as an anchor off the diff.
    #[error("github: unprocessable: {0}")]
    Unprocessable(String),
    /// A create got a 5xx, or no answer after it was sent: it may exist or not.
    #[error("github: create unconfirmed")]
    Ambiguous,
    /// Refused until this time, longer than one request waits.
    #[error("github: rate limited until {}", unix(.0))]
    RateLimited(SystemTime),
    /// A reply longer than its call allows, read no further.
    #[error("github: reply over the size cap")]
    TooLarge,
    #[error("github: {0}")]
    Http(String),
}

/// The App. Its JWT signs `GET /app` and the token mints; every other call goes
/// through a [`Gh`]. Clones share the key, the HTTP client and the token cache.
#[derive(Clone)]
pub struct Github {
    api_base: String,
    app_id: i64,
    key: Arc<RsaKeyPair>,
    http: Client,
    tokens: Arc<Mutex<HashMap<i64, Cached>>>,
}

/// An installation token and when to stop using it.
struct Cached {
    token: Secret,
    until: Instant,
}

/// Whether a request may be sent again after a transport error or a 5xx.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Resend {
    /// Reads, edits, reactions, GraphQL and token mints: a second copy changes nothing.
    Safe,
    /// A create: a second copy would post twice.
    Never,
}

impl fmt::Debug for Github {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Github")
            .field("api_base", &self.api_base)
            .field("app_id", &self.app_id)
            .finish_non_exhaustive()
    }
}

impl Github {
    /// The App at `api_base` ([`API`], or a fake in tests), signing with `pem`, the
    /// PKCS#1 PEM key GitHub issues. Errors never hold key bytes.
    pub fn new(api_base: &str, app_id: i64, pem: &Secret) -> Result<Self, GhError> {
        let body: String = pem
            .expose()
            .lines()
            .filter(|l| !l.trim_start().starts_with("-----"))
            .flat_map(str::split_whitespace)
            .collect();
        let der = STANDARD
            .decode(body)
            .map_err(|_| GhError::Http("app key: not base64".to_owned()))?;
        let key = RsaKeyPair::from_der(&der).map_err(|e| GhError::Http(format!("app key: {e}")))?;
        let http = http::client().map_err(|e| GhError::Http(e.to_string()))?;
        Ok(Self {
            api_base: api_base.trim_end_matches('/').to_owned(),
            app_id,
            key: Arc::new(key),
            http,
            tokens: Arc::default(),
        })
    }

    /// The login our comments carry: `<slug>[bot]`, from `GET /app`.
    pub async fn bot_login(&self) -> Result<String, GhError> {
        #[derive(Deserialize)]
        struct App {
            slug: String,
        }
        let app: App = decode(&self.app_call(Method::GET, "/app").await?)?;
        Ok(format!("{}[bot]", app.slug))
    }

    /// A client for one installation, made per job, with the cached token (minted
    /// when there is none, or it is within 5 min of expiry).
    pub async fn installation(&self, id: i64) -> Result<Gh, GhError> {
        Ok(Gh {
            token: self.token(id).await?,
            github: self.clone(),
            installation: id,
        })
    }

    /// A request signed with a fresh App JWT; a 401 here is a bad key, never retried.
    async fn app_call(&self, method: Method, path: &str) -> Result<Vec<u8>, GhError> {
        let jwt = self.jwt(SystemTime::now())?;
        let (status, body) = self
            .send(&jwt, method, path, None, Resend::Safe, UNCAPPED)
            .await?;
        check(status, body, &jwt)
    }

    async fn token(&self, id: i64) -> Result<Secret, GhError> {
        let cached = self
            .cache()
            .get(&id)
            .filter(|c| Instant::now() < c.until)
            .map(|c| c.token.clone());
        match cached {
            Some(token) => Ok(token),
            None => self.mint(id).await,
        }
    }

    /// A new installation token, cached until 5 min before it expires. Two jobs that
    /// race here each mint one; both work.
    async fn mint(&self, id: i64) -> Result<Secret, GhError> {
        #[derive(Deserialize)]
        struct Minted {
            token: String,
            expires_at: String,
        }
        let path = format!("/app/installations/{id}/access_tokens");
        let minted: Minted = decode(&self.app_call(Method::POST, &path).await?)?;
        let expires = OffsetDateTime::parse(&minted.expires_at, &Rfc3339)
            .map_err(|_| GhError::Http("token expires_at: unreadable".to_owned()))?;
        let left = SystemTime::from(expires)
            .duration_since(SystemTime::now())
            .unwrap_or_default()
            .min(TOKEN_LIFE);
        let token = Secret::new(minted.token);
        self.cache().insert(
            id,
            Cached {
                token: token.clone(),
                until: Instant::now() + left.saturating_sub(TOKEN_MARGIN),
            },
        );
        Ok(token)
    }

    fn cache(&self) -> MutexGuard<'_, HashMap<i64, Cached>> {
        self.tokens.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The App JWT: RS256 over `iat` 60 s back (clock drift), `exp` 9 min on (GitHub
    /// allows 10) and `iss` the App id, as a string.
    fn jwt(&self, now: SystemTime) -> Result<String, GhError> {
        let secs = unix(&now);
        let claims = serde_json::json!({
            "iat": secs.saturating_sub(60),
            "exp": secs + 540,
            "iss": self.app_id.to_string(),
        });
        let input = format!(
            "{}.{}",
            URL_SAFE_NO_PAD.encode(r#"{"alg":"RS256","typ":"JWT"}"#),
            URL_SAFE_NO_PAD.encode(claims.to_string())
        );
        let mut sig = vec![0u8; self.key.public().modulus_len()];
        self.key
            .sign(
                &RSA_PKCS1_SHA256,
                &SystemRandom::new(),
                input.as_bytes(),
                &mut sig,
            )
            .map_err(|_| GhError::Http("app jwt: signing failed".to_owned()))?;
        Ok(format!("{input}.{}", URL_SAFE_NO_PAD.encode(sig)))
    }

    /// Sends one request under every retry rule and returns the status and body it
    /// ends with, for [`check`] to judge. At most three sends in all:
    /// - each carries the API version and a 30 s limit;
    /// - a [`Resend::Safe`] request goes again after a transport error or a 5xx,
    ///   1 s then 4 s later; a create does not, and a 5xx or a lost answer after
    ///   sending is [`GhError::Ambiguous`];
    /// - a rate limit refused the request, so any request, a create too, waits it
    ///   out up to 60 s and goes again; a longer wait is [`GhError::RateLimited`];
    /// - a reply body over `limit` bytes is [`GhError::TooLarge`], read no further.
    async fn send(
        &self,
        auth: &str,
        method: Method,
        path: &str,
        body: Option<&[u8]>,
        resend: Resend,
        limit: usize,
    ) -> Result<(u16, Vec<u8>), GhError> {
        let mut bearer = HeaderValue::from_str(&format!("Bearer {auth}"))
            .map_err(|_| GhError::Http("authorization: invalid".to_owned()))?;
        bearer.set_sensitive(true);
        let url = format!("{}{path}", self.api_base);
        let mut sends = 0;
        loop {
            sends += 1;
            let last = sends == MAX_SENDS;
            let mut req = self
                .http
                .request(method.clone(), url.as_str())
                .timeout(TIMEOUT)
                .header(ACCEPT, "application/vnd.github+json")
                .header("x-github-api-version", API_VERSION)
                .header(USER_AGENT, "diffagent")
                .header(AUTHORIZATION, bearer.clone());
            if let Some(body) = body {
                req = req
                    .header(CONTENT_TYPE, "application/json")
                    .body(body.to_vec());
            }
            let answer = async {
                let mut resp = req.send().await?;
                let status = resp.status().as_u16();
                let limited = rate_limited(status, resp.headers(), SystemTime::now());
                let mut bytes = Vec::new();
                while let Some(chunk) = resp.chunk().await? {
                    if chunk.len() > limit - bytes.len() {
                        return Ok((status, limited, None));
                    }
                    bytes.extend_from_slice(&chunk);
                }
                Ok::<_, reqwest::Error>((status, limited, Some(bytes)))
            }
            .await;
            let (status, limited, bytes) = match answer {
                Ok(answer) => answer,
                // Refused at connect, it was never sent; past that it may have landed.
                Err(e) if resend == Resend::Never => {
                    return Err(if e.is_connect() {
                        GhError::Http(clean(&chain(&e), auth))
                    } else {
                        GhError::Ambiguous
                    });
                }
                Err(e) if last => return Err(GhError::Http(clean(&chain(&e), auth))),
                Err(e) => {
                    tracing::warn!(path, attempt = sends, err = %clean(&chain(&e), auth), "github resend");
                    tokio::time::sleep(backoff(sends)).await;
                    continue;
                }
            };
            let Some(bytes) = bytes else {
                return Err(GhError::TooLarge);
            };
            if let Some(until) = limited {
                let wait = until.duration_since(SystemTime::now()).unwrap_or_default();
                if last || wait > MAX_RATE_WAIT {
                    return Err(GhError::RateLimited(until));
                }
                tracing::warn!(
                    path,
                    attempt = sends,
                    wait_secs = wait.as_secs(),
                    "github rate limited"
                );
                tokio::time::sleep(wait).await;
                continue;
            }
            if status >= 500 {
                if resend == Resend::Never {
                    return Err(GhError::Ambiguous);
                }
                if !last {
                    tracing::warn!(path, attempt = sends, status, "github resend");
                    tokio::time::sleep(backoff(sends)).await;
                    continue;
                }
            }
            return Ok((status, bytes));
        }
    }
}

/// One installation's client, made per job.
#[derive(Debug)]
pub struct Gh {
    /// The installation token when the job started; git fetches with it. Requests
    /// read the cache instead, so a job that outlives this token still gets answers.
    pub token: Secret,
    github: Github,
    installation: i64,
}

impl Gh {
    pub async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T, GhError> {
        self.get_within(path, UNCAPPED).await
    }

    /// [`Gh::get`] of a reply at most `limit` bytes long; a longer one is
    /// [`GhError::TooLarge`].
    pub async fn get_within<T: DeserializeOwned>(
        &self,
        path: &str,
        limit: usize,
    ) -> Result<T, GhError> {
        decode(
            &self
                .call(Method::GET, path, None, Resend::Safe, limit)
                .await?,
        )
    }

    /// Every item of a list endpoint, 100 a page, until a short page or 30 pages.
    pub async fn pages<T: DeserializeOwned>(&self, path: &str) -> Result<Vec<T>, GhError> {
        Ok(self.listing(path).await?.0)
    }

    /// [`Gh::pages`], and whether the listing may be cut: its 30th page was full, so
    /// items past it are missing.
    pub async fn listing<T: DeserializeOwned>(
        &self,
        path: &str,
    ) -> Result<(Vec<T>, bool), GhError> {
        let sep = if path.contains('?') { '&' } else { '?' };
        let mut out = Vec::new();
        for page in 1..=MAX_PAGES {
            let items: Vec<T> = self
                .get(&format!("{path}{sep}per_page={PER_PAGE}&page={page}"))
                .await?;
            let short = items.len() < PER_PAGE;
            out.extend(items);
            if short {
                return Ok((out, false));
            }
        }
        tracing::warn!(
            path,
            pages = MAX_PAGES,
            "github listing cut at the page cap"
        );
        Ok((out, true))
    }

    /// POSTs a create. It is sent once: a 5xx or a lost answer is
    /// [`GhError::Ambiguous`], and the caller checks before sending it again.
    pub async fn create<T: DeserializeOwned>(
        &self,
        path: &str,
        body: &impl Serialize,
    ) -> Result<T, GhError> {
        let body = encode(body)?;
        decode(
            &self
                .call(Method::POST, path, Some(&body), Resend::Never, UNCAPPED)
                .await?,
        )
    }

    /// POSTs a reaction to `path`. GitHub keeps one per user and content (200 when it
    /// exists), so it is sent again like an edit.
    pub async fn react(&self, path: &str, content: &str) -> Result<(), GhError> {
        let body = encode(&serde_json::json!({"content": content}))?;
        self.call(Method::POST, path, Some(&body), Resend::Safe, UNCAPPED)
            .await
            .map(drop)
    }

    /// POSTs a check run. Each one is computed from the store when it is sent, so a
    /// duplicate is the same run or a newer one: it is sent again like an edit.
    pub async fn check_run(&self, path: &str, body: &serde_json::Value) -> Result<(), GhError> {
        let body = encode(body)?;
        self.call(Method::POST, path, Some(&body), Resend::Safe, UNCAPPED)
            .await
            .map(drop)
    }

    /// PATCHes an edit, which is safe to send twice.
    pub async fn edit<T: DeserializeOwned>(
        &self,
        path: &str,
        body: &impl Serialize,
    ) -> Result<T, GhError> {
        let body = encode(body)?;
        decode(
            &self
                .call(Method::PATCH, path, Some(&body), Resend::Safe, UNCAPPED)
                .await?,
        )
    }

    /// A GraphQL query or idempotent mutation, returning its `data`. A 200 with
    /// `errors` is an error.
    pub async fn graphql<T: DeserializeOwned>(
        &self,
        query: &str,
        variables: &serde_json::Value,
    ) -> Result<T, GhError> {
        #[derive(Deserialize)]
        struct Reply {
            data: Option<serde_json::Value>,
            #[serde(default)]
            errors: Vec<Problem>,
        }
        #[derive(Deserialize)]
        struct Problem {
            #[serde(default)]
            message: String,
        }
        let body = encode(&serde_json::json!({"query": query, "variables": variables}))?;
        let reply: Reply = decode(
            &self
                .call(
                    Method::POST,
                    "/graphql",
                    Some(&body),
                    Resend::Safe,
                    UNCAPPED,
                )
                .await?,
        )?;
        if !reply.errors.is_empty() {
            let messages: Vec<&str> = reply.errors.iter().map(|e| e.message.as_str()).collect();
            return Err(GhError::Http(format!(
                "graphql: {}",
                cut(&messages.join("; "), MAX_MESSAGE)
            )));
        }
        serde_json::from_value(reply.data.unwrap_or_default()).map_err(|e| {
            GhError::Http(format!(
                "graphql decode: {}",
                cut(&e.to_string(), MAX_MESSAGE)
            ))
        })
    }

    /// One request with the installation token. A 401 means it was revoked or
    /// expired early: one fresh token, then that answer stands.
    async fn call(
        &self,
        method: Method,
        path: &str,
        body: Option<&[u8]>,
        resend: Resend,
        limit: usize,
    ) -> Result<Vec<u8>, GhError> {
        let token = self.github.token(self.installation).await?;
        let (status, bytes) = self
            .github
            .send(token.expose(), method.clone(), path, body, resend, limit)
            .await?;
        if status != 401 {
            return check(status, bytes, token.expose());
        }
        let token = self.github.mint(self.installation).await?;
        let (status, bytes) = self
            .github
            .send(token.expose(), method, path, body, resend, limit)
            .await?;
        check(status, bytes, token.expose())
    }
}

/// The body of a 2xx, or the error its status means.
fn check(status: u16, body: Vec<u8>, secret: &str) -> Result<Vec<u8>, GhError> {
    match status {
        200..=299 => Ok(body),
        404 => Err(GhError::NotFound),
        422 => Err(GhError::Unprocessable(clean(
            &String::from_utf8_lossy(&body),
            secret,
        ))),
        _ => Err(GhError::Http(format!(
            "{status}: {}",
            clean(&String::from_utf8_lossy(&body), secret)
        ))),
    }
}

/// When a refused request may go again, or `None` when it was not refused for a rate
/// limit. A 429 is one; a 403 only with `retry-after` or `x-ratelimit-remaining: 0`,
/// since a plain 403 is a permission. GitHub asks for a minute when it names no time.
fn rate_limited(status: u16, headers: &HeaderMap, now: SystemTime) -> Option<SystemTime> {
    let header = |name: &'static str| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(str::trim)
    };
    let retry_after = header("retry-after");
    let exhausted = header("x-ratelimit-remaining") == Some("0");
    let refused = status == 429 || (status == 403 && (retry_after.is_some() || exhausted));
    if !refused {
        return None;
    }
    let after = retry_after
        .and_then(|v| v.parse().ok())
        .and_then(|secs| now.checked_add(Duration::from_secs(secs)));
    let reset = header("x-ratelimit-reset")
        .filter(|_| exhausted)
        .and_then(|v| v.parse().ok())
        .and_then(|secs| UNIX_EPOCH.checked_add(Duration::from_secs(secs)));
    Some(after.or(reset).unwrap_or(now + Duration::from_secs(60)))
}

/// The wait before resend `sends + 1` of a safe request: 1 s, then 4 s.
fn backoff(sends: u32) -> Duration {
    Duration::from_secs(4u64.pow(sends - 1))
}

fn unix(t: &SystemTime) -> u64 {
    t.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn encode(body: &impl Serialize) -> Result<Vec<u8>, GhError> {
    serde_json::to_vec(body).map_err(|e| GhError::Http(format!("encode: {e}")))
}

fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, GhError> {
    serde_json::from_slice(bytes)
        .map_err(|e| GhError::Http(format!("decode: {}", cut(&e.to_string(), MAX_MESSAGE))))
}

/// `text` fit for an error: `secret` replaced first, then cut to 300 chars, so no
/// prefix of it survives either.
fn clean(text: &str, secret: &str) -> String {
    let text = if secret.is_empty() {
        text.to_owned()
    } else {
        text.replace(secret, "<redacted>")
    };
    cut(text.trim(), MAX_MESSAGE)
}

#[cfg(test)]
mod tests {
    use ring::signature::{RSA_PKCS1_2048_8192_SHA256, UnparsedPublicKey};
    use serde_json::{Value, json};

    use super::fake::{APP_ID, Fake, FaultKind, app_key};
    use super::*;

    const PR: &str = "/repos/o/r/pulls/1";

    /// A fake with PR 1 and a client for installation 7 (one token minted).
    async fn setup() -> (Fake, Gh) {
        let fake = Fake::start().await;
        fake.world().pulls.insert(1, json!({"number": 1}));
        let gh = fake.github().installation(7).await.unwrap();
        (fake, gh)
    }

    /// Neither a minted token nor a JWT shows in `e`, however it is printed.
    fn assert_no_secret(e: &GhError) {
        for shown in [e.to_string(), format!("{e:?}")] {
            assert!(
                !shown.contains("ghs_fake") && !shown.contains("eyJhbGci"),
                "{shown}"
            );
        }
    }

    #[test]
    fn the_jwt_is_rs256_for_the_app() {
        let github = Github::new("http://127.0.0.1:9", APP_ID, &app_key()).unwrap();
        let now = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let jwt = github.jwt(now).unwrap();
        let (input, sig) = jwt.rsplit_once('.').unwrap();
        let sig = URL_SAFE_NO_PAD.decode(sig).unwrap();
        let public =
            UnparsedPublicKey::new(&RSA_PKCS1_2048_8192_SHA256, github.key.public().as_ref());
        public.verify(input.as_bytes(), &sig).unwrap();
        assert!(public.verify(b"tampered", &sig).is_err());

        let part = |s: &str| -> Value {
            serde_json::from_slice(&URL_SAFE_NO_PAD.decode(s).unwrap()).unwrap()
        };
        let (header, claims) = input.split_once('.').unwrap();
        assert_eq!(part(header), json!({"alg": "RS256", "typ": "JWT"}));
        assert_eq!(
            part(claims),
            json!({"iat": 1_699_999_940, "exp": 1_700_000_540, "iss": "4242"})
        );

        for bad in ["", "not base64 !", "QUJD"] {
            let err = Github::new("http://x", APP_ID, &Secret::new(bad)).unwrap_err();
            assert!(
                matches!(&err, GhError::Http(m) if m.starts_with("app key")),
                "{bad:?}: {err:?}"
            );
        }
    }

    #[tokio::test]
    async fn bot_login_is_the_app_slug() {
        let fake = Fake::start().await;
        let login = fake.github().bot_login().await.unwrap();
        assert_eq!(login, "diffagent[bot]");
    }

    /// The fake's tokens live an hour, so the cache holds one for 55 min. Time moves
    /// only while paused, with no request in flight.
    #[tokio::test]
    async fn tokens_are_cached_until_five_minutes_before_expiry() {
        let fake = Fake::start().await;
        let github = fake.github();
        let first = github.installation(7).await.unwrap();
        let again = github.installation(7).await.unwrap();
        assert_eq!(again.token, first.token);
        assert_eq!(fake.world().minted, 1);
        github.installation(8).await.unwrap();
        assert_eq!(fake.world().minted, 2, "each installation has its own");

        tokio::time::pause();
        tokio::time::advance(Duration::from_secs(55 * 60 - 3)).await;
        tokio::time::resume();
        github.installation(7).await.unwrap();
        assert_eq!(fake.world().minted, 2, "3 s before expiry - 5 min: cached");

        tokio::time::pause();
        tokio::time::advance(Duration::from_secs(4)).await;
        tokio::time::resume();
        let fresh = github.installation(7).await.unwrap();
        assert_eq!(fake.world().minted, 3);
        assert_ne!(fresh.token, first.token);
    }

    #[tokio::test]
    async fn a_401_refetches_the_token_once() {
        let (fake, gh) = setup().await;
        let route = format!("GET {PR}");
        fake.world().fault(&route, 1, FaultKind::Before(401));
        let pr: Value = gh.get(PR).await.unwrap();
        assert_eq!(pr["number"], 1);
        assert_eq!(fake.world().minted, 2);

        fake.world().fault(&route, 2, FaultKind::Before(401));
        let err = gh.get::<Value>(PR).await.unwrap_err();
        assert!(err.to_string().starts_with("github: 401: "), "{err}");
        assert_eq!(fake.world().minted, 3, "one refetch, not a loop");
        assert_eq!(fake.sent(&route), 4);
        assert_no_secret(&err);
    }

    #[tokio::test]
    async fn reads_retry_a_5xx_twice_then_fail() {
        let (fake, gh) = setup().await;
        let route = format!("GET {PR}");
        fake.world().fault(&route, 2, FaultKind::Before(502));
        let started = Instant::now();
        let pr: Value = gh.get(PR).await.unwrap();
        assert_eq!(pr["number"], 1);
        assert_eq!(fake.sent(&route), 3);
        assert!(started.elapsed() >= Duration::from_secs(5), "1 s, then 4 s");

        fake.world().fault(&route, 3, FaultKind::Before(502));
        let err = gh.get::<Value>(PR).await.unwrap_err();
        assert!(err.to_string().starts_with("github: 502: "), "{err}");
        assert_eq!(fake.sent(&route), 6);
        assert_no_secret(&err);
    }

    #[tokio::test]
    async fn a_create_applied_then_failed_is_sent_once_and_ambiguous() {
        let (fake, gh) = setup().await;
        let route = "POST /repos/o/r/issues/1/comments";
        fake.world().fault(route, 1, FaultKind::After(502));
        let err = gh
            .create::<Value>("/repos/o/r/issues/1/comments", &json!({"body": "hi"}))
            .await
            .unwrap_err();
        assert_eq!(err, GhError::Ambiguous);
        assert_eq!(fake.sent(route), 1);
        assert_eq!(fake.world().issue_comments.len(), 1, "it was applied");
    }

    #[tokio::test]
    async fn rate_limits_are_waited_out_or_returned() {
        let (fake, gh) = setup().await;
        // A refused create is sent again after `retry-after`.
        let route = "POST /repos/o/r/issues/1/comments";
        let limited = FaultKind::Limited {
            status: 403,
            retry_after: Some(1),
            reset: None,
        };
        fake.world().fault(route, 1, limited);
        let started = Instant::now();
        let _: Value = gh
            .create("/repos/o/r/issues/1/comments", &json!({"body": "hi"}))
            .await
            .unwrap();
        assert!(started.elapsed() >= Duration::from_secs(1));
        assert_eq!(fake.sent(route), 2);
        assert_eq!(fake.world().issue_comments.len(), 1);

        // A reset an hour away is the job's to wait for.
        let reset = unix(&SystemTime::now()) + 3600;
        let limited = FaultKind::Limited {
            status: 403,
            retry_after: None,
            reset: Some(reset),
        };
        let get = format!("GET {PR}");
        fake.world().fault(&get, 1, limited);
        let err = gh.get::<Value>(PR).await.unwrap_err();
        assert_eq!(
            err,
            GhError::RateLimited(UNIX_EPOCH + Duration::from_secs(reset))
        );
        assert_eq!(fake.sent(&get), 1);
    }

    #[tokio::test]
    async fn listings_take_100_a_page_up_to_30_pages() {
        let (fake, gh) = setup().await;
        let route = "GET /repos/o/r/pulls/1/files";
        // (files, listed, pages read, maybe cut): a full 30th page may hide more.
        let cases = [
            (0, 0, 1, false),
            (150, 150, 2, false),
            (200, 200, 3, false),
            (3000, 3000, 30, true),
            (3100, 3000, 30, true),
        ];
        for (files, want, pages, cut) in cases {
            fake.world().files.insert(
                1,
                (0..files)
                    .map(|i| json!({"filename": format!("f{i}.go")}))
                    .collect(),
            );
            let before = fake.sent(route);
            let (got, maybe_cut): (Vec<Value>, bool) =
                gh.listing("/repos/o/r/pulls/1/files").await.unwrap();
            assert_eq!(got.len(), want, "{files} files");
            assert_eq!(maybe_cut, cut, "{files} files");
            assert_eq!(fake.sent(route) - before, pages, "{files} files");
        }
    }

    /// A capped read stops at its cap: a longer reply is `TooLarge`, and one that fits
    /// is read whole.
    #[tokio::test]
    async fn a_capped_get_reads_no_further() {
        let (fake, gh) = setup().await;
        let key = ("a.go".to_owned(), "s".to_owned());
        fake.world().contents.insert(key, "x".repeat(10_000));
        let path = "/repos/o/r/contents/a.go?ref=s";
        let over = gh.get_within::<Value>(path, 1_000).await;
        assert_eq!(over.unwrap_err(), GhError::TooLarge);
        let fits: Value = gh.get_within(path, 1 << 20).await.unwrap();
        assert_eq!(fits["encoding"], "base64");
        assert_eq!(gh.get::<Value>(path).await.unwrap(), fits);
    }

    #[tokio::test]
    async fn graphql_errors_in_a_200_are_errors() {
        let (_fake, gh) = setup().await;
        let resolve = "mutation($id: ID!) { resolveReviewThread(input: {threadId: $id}) \
                       { thread { isResolved } } }";
        let err = gh
            .graphql::<Value>(resolve, &json!({"id": "PRRT_9"}))
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .starts_with("github: graphql: Could not resolve to a node"),
            "{err}"
        );
    }

    /// Every fault body echoes the request's `Authorization`, so each error below
    /// would carry the token or JWT unless it is replaced.
    #[tokio::test]
    async fn errors_never_hold_a_token() {
        let fake = Fake::start().await;
        let github = fake.github();
        fake.world().fault(
            "POST /app/installations/*/access_tokens",
            1,
            FaultKind::Before(401),
        );
        let err = github.installation(7).await.unwrap_err();
        assert!(
            matches!(&err, GhError::Http(m) if m.starts_with("401: ") && m.contains("<redacted>")),
            "{err:?}"
        );
        assert_no_secret(&err);

        fake.world().pulls.insert(1, json!({"number": 1}));
        let gh = github.installation(7).await.unwrap();
        let cases = [
            (400, "github: 400: "),
            (403, "github: 403: "),
            (404, "github: not found"),
            (422, "github: unprocessable: "),
        ];
        for (status, want) in cases {
            fake.world()
                .fault(&format!("GET {PR}"), 1, FaultKind::Before(status));
            let err = gh.get::<Value>(PR).await.unwrap_err();
            assert!(err.to_string().starts_with(want), "{status}: {err}");
            assert_no_secret(&err);
        }
    }

    /// The routes the worker drives, through `Gh`: a review whose comments show up one
    /// listing late, comment edits, Threads listed and resolved, the status comment,
    /// a file at a ref, a compare and a permission.
    #[tokio::test]
    async fn the_fake_serves_a_review_round_trip() {
        let (fake, gh) = setup().await;
        {
            let mut w = fake.world();
            w.compares
                .insert("a...b".to_owned(), json!({"status": "ahead", "files": []}));
            w.contents.insert(
                ("CLAUDE.md".to_owned(), "a".to_owned()),
                "rules\n".to_owned(),
            );
            w.permissions.insert("alice".to_owned(), "write".to_owned());
            w.fault("POST /repos/o/r/pulls/1/reviews", 1, FaultKind::Hide(1));
        }
        let review: Value = gh
            .create(
                "/repos/o/r/pulls/1/reviews",
                &json!({"commit_id": "b", "event": "COMMENT", "body": "", "comments": [
                    {"path": "a.go", "line": 3, "side": "RIGHT", "body": "one"},
                    {"path": "a.go", "line": 9, "side": "RIGHT", "body": "two"},
                ]}),
            )
            .await
            .unwrap();
        let by_review = format!("/repos/o/r/pulls/1/reviews/{}/comments", review["id"]);
        let late: Vec<Value> = gh.pages(&by_review).await.unwrap();
        assert!(late.is_empty(), "hidden from the first listing");
        let comments: Vec<Value> = gh.pages(&by_review).await.unwrap();
        assert_eq!(comments.len(), 2);
        assert_eq!(comments[0]["user"]["login"], "diffagent[bot]");
        let first = &comments[0]["id"];
        let edited: Value = gh
            .edit(
                &format!("/repos/o/r/pulls/comments/{first}"),
                &json!({"body": "closed"}),
            )
            .await
            .unwrap();
        assert_eq!(edited["body"], "closed");
        let all: Vec<Value> = gh.pages("/repos/o/r/pulls/1/comments").await.unwrap();
        assert_eq!(all.len(), 2);

        let listed: Value = gh
            .graphql(
                "query { repository { pullRequest { reviewThreads { nodes { id } } } } }",
                &json!({}),
            )
            .await
            .unwrap();
        let thread = &listed["repository"]["pullRequest"]["reviewThreads"]["nodes"][0];
        assert_eq!(&thread["comments"]["nodes"][0]["databaseId"], first);
        let resolve = "mutation($id: ID!) { resolveReviewThread(input: {threadId: $id}) \
                       { thread { isResolved } } }";
        let done: Value = gh
            .graphql(resolve, &json!({"id": thread["id"]}))
            .await
            .unwrap();
        assert_eq!(done["resolveReviewThread"]["thread"]["isResolved"], true);
        assert!(fake.world().threads[0].resolved);

        let status: Value = gh
            .create(
                "/repos/o/r/issues/1/comments",
                &json!({"body": "Reviewing"}),
            )
            .await
            .unwrap();
        let _: Value = gh
            .edit(
                &format!("/repos/o/r/issues/comments/{}", status["id"]),
                &json!({"body": "Done"}),
            )
            .await
            .unwrap();
        let notes: Vec<Value> = gh.pages("/repos/o/r/issues/1/comments").await.unwrap();
        assert_eq!(notes[0]["body"], "Done");

        let file: Value = gh.get("/repos/o/r/contents/CLAUDE.md?ref=a").await.unwrap();
        let content: String = file["content"]
            .as_str()
            .unwrap()
            .split_whitespace()
            .collect();
        assert_eq!(STANDARD.decode(content).unwrap(), b"rules\n");
        let compare: Value = gh.get("/repos/o/r/compare/a...b").await.unwrap();
        assert_eq!(compare["status"], "ahead");
        let perm: Value = gh
            .get("/repos/o/r/collaborators/alice/permission")
            .await
            .unwrap();
        assert_eq!(perm["permission"], "write");
        let none = gh
            .get::<Value>("/repos/o/r/collaborators/bob/permission")
            .await;
        assert_eq!(none.unwrap_err(), GhError::NotFound);
        assert_eq!(fake.world().reviews.len(), 1);
    }

    #[test]
    fn rate_limit_rules() {
        let now = UNIX_EPOCH + Duration::from_secs(1_000_000);
        let at = |secs| Some(now + Duration::from_secs(secs));
        let cases = [
            (429, vec![], at(60)),
            (429, vec![("retry-after", "7")], at(7)),
            (403, vec![("retry-after", "2")], at(2)),
            (
                403,
                vec![
                    ("x-ratelimit-remaining", "0"),
                    ("x-ratelimit-reset", "1000600"),
                ],
                at(600),
            ),
            (403, vec![("x-ratelimit-remaining", "0")], at(60)),
            (
                403,
                vec![
                    ("x-ratelimit-remaining", "12"),
                    ("x-ratelimit-reset", "1000600"),
                ],
                None,
            ),
            (403, vec![], None),
            (502, vec![("retry-after", "1")], None),
        ];
        for (status, headers, want) in cases {
            let mut map = HeaderMap::new();
            for (name, value) in &headers {
                map.insert(
                    reqwest::header::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                    HeaderValue::from_str(value).unwrap(),
                );
            }
            assert_eq!(
                rate_limited(status, &map, now),
                want,
                "{status} {headers:?}"
            );
        }
    }
}
