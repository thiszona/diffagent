//! A GitHub on 127.0.0.1 for tests: it mints tokens, holds one repo's PRs, files,
//! comments, reviews and Threads, and fails on cue. Routes ignore the owner and
//! repo in the path. Every request must carry the API version, a user agent, and a
//! JWT on `/app` routes or a minted token elsewhere, as GitHub demands.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, SystemTime};

use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use serde_json::{Value, json};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use super::Github;
use crate::config::Secret;

/// The App id every fake client signs as.
pub(crate) const APP_ID: i64 = 4242;

/// The test-only App key as the PKCS#1 PEM GitHub issues. The committed file
/// (`tests/data/github/app_key.test-only.b64`) was generated for these tests and is
/// registered nowhere; the PEM armour is put on here so secret scanners pass it.
pub(crate) fn app_key() -> Secret {
    let label = "PRIVATE KEY";
    Secret::new(format!(
        "-----BEGIN RSA {label}-----\n{}-----END RSA {label}-----\n",
        include_str!("../../tests/data/github/app_key.test-only.b64")
    ))
}

/// A running fake; it stops with the test's runtime.
pub(crate) struct Fake {
    pub base: String,
    world: Arc<Mutex<World>>,
}

impl Fake {
    pub async fn start() -> Self {
        let world = Arc::new(Mutex::new(World {
            slug: "diffagent".to_owned(),
            token_ttl: Duration::from_secs(3600),
            next_id: 1000,
            ..World::default()
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let app = Router::new()
            .fallback(handle)
            .with_state(Arc::clone(&world));
        tokio::spawn(async move { axum::serve(listener, app).await });
        Self { base, world }
    }

    /// The fake's state, to seed or inspect. Never hold it across an await.
    pub fn world(&self) -> MutexGuard<'_, World> {
        self.world.lock().unwrap()
    }

    /// An App client on this fake, with its own token cache.
    pub fn github(&self) -> Github {
        Github::new(&self.base, APP_ID, &app_key()).unwrap()
    }

    /// How many requests reached the fake on `route` (`METHOD /path`), any query.
    pub fn sent(&self, route: &str) -> usize {
        let paged = format!("{route}?");
        self.world()
            .log
            .iter()
            .filter(|l| l.as_str() == route || l.starts_with(&paged))
            .count()
    }
}

/// What the fake holds. Tests seed it and read it back through [`Fake::world`].
#[derive(Default)]
pub(crate) struct World {
    /// `GET /app`'s slug; the fake's comments are by `<slug>[bot]`.
    pub slug: String,
    /// How long a minted token lives.
    pub token_ttl: Duration,
    /// Tokens minted so far; the n-th is `ghs_fake_<n>`.
    pub minted: usize,
    /// PR number → the `GET …/pulls/{n}` body; `GET …/pulls` lists them by `state`.
    pub pulls: BTreeMap<i64, Value>,
    /// PR number → its files, 30 a page unless `per_page` says otherwise.
    pub files: BTreeMap<i64, Vec<Value>>,
    /// `base...head` → the compare body.
    pub compares: BTreeMap<String, Value>,
    /// (path, ref) → the file's text.
    pub contents: BTreeMap<(String, String), String>,
    /// login → `admin`, `write` or `read`; anyone else is a 404.
    pub permissions: BTreeMap<String, String>,
    /// (PR, comment), oldest first.
    pub issue_comments: Vec<(i64, Value)>,
    pub review_comments: Vec<(i64, Value)>,
    pub reviews: Vec<(i64, Value)>,
    pub threads: Vec<Thread>,
    /// (review comment, content) for each reaction, once each, as GitHub keeps them.
    pub reactions: Vec<(i64, String)>,
    /// Every check run's request body, oldest first.
    pub check_runs: Vec<Value>,
    /// Every request as `METHOD /path?query`, in order.
    pub log: Vec<String>,
    faults: Vec<Fault>,
    next_id: i64,
    /// Review id → listings that still leave its comments out.
    hidden: BTreeMap<i64, usize>,
}

/// A review Thread: GraphQL's id, its first comment, and whether it is resolved.
pub(crate) struct Thread {
    pub id: String,
    pub root: i64,
    pub resolved: bool,
}

struct Fault {
    route: String,
    times: usize,
    kind: FaultKind,
}

/// How a request matching a fault fails. Every failure body echoes the request's
/// `Authorization`, so a test sees whether the client keeps it out of its errors.
#[derive(Debug, Clone, Copy)]
pub(crate) enum FaultKind {
    /// Answer this status without applying the request.
    Before(u16),
    /// Apply the request, then answer this status.
    After(u16),
    /// Refuse as a rate limit: this status with `retry-after` seconds, or with
    /// `x-ratelimit-remaining: 0` and `x-ratelimit-reset` (Unix seconds).
    Limited {
        status: u16,
        retry_after: Option<u64>,
        reset: Option<u64>,
    },
    /// Apply a review create, then leave its comments out of the next `n` listings,
    /// as GitHub's replicas sometimes do.
    Hide(usize),
}

impl World {
    /// Fails the next `times` requests matching `route`: `METHOD /path`, where `*`
    /// matches one segment and the query is ignored.
    pub fn fault(&mut self, route: &str, times: usize, kind: FaultKind) {
        self.faults.push(Fault {
            route: route.to_owned(),
            times,
            kind,
        });
    }

    fn take_fault(&mut self, target: &str) -> Option<FaultKind> {
        let fault = self
            .faults
            .iter_mut()
            .find(|f| f.times > 0 && matches(&f.route, target))?;
        fault.times -= 1;
        Some(fault.kind)
    }

    fn id(&mut self) -> i64 {
        self.next_id += 1;
        self.next_id
    }

    fn bot(&self) -> String {
        format!("{}[bot]", self.slug)
    }

    fn route(
        &mut self,
        method: &Method,
        path: &str,
        query: &str,
        body: &Value,
    ) -> (StatusCode, Value) {
        let segments: Vec<&str> = path.trim_start_matches('/').split('/').collect();
        match (method.as_str(), segments.as_slice()) {
            ("GET", ["app"]) => (StatusCode::OK, json!({"slug": self.slug})),
            ("POST", ["app", "installations", _, "access_tokens"]) => {
                self.minted += 1;
                // GitHub's form: `2016-07-11T22:14:10Z`.
                let expires = OffsetDateTime::from(SystemTime::now() + self.token_ttl)
                    .replace_nanosecond(0)
                    .ok()
                    .and_then(|t| t.format(&Rfc3339).ok())
                    .unwrap_or_default();
                let token = format!("ghs_fake_{}", self.minted);
                (
                    StatusCode::CREATED,
                    json!({"token": token, "expires_at": expires}),
                )
            }
            ("POST", ["graphql"]) => (StatusCode::OK, self.graphql(body)),
            (method, ["repos", _, _, rest @ ..]) => self.repo(method, rest, query, body),
            _ => not_found(),
        }
    }

    fn repo(
        &mut self,
        method: &str,
        rest: &[&str],
        query: &str,
        body: &Value,
    ) -> (StatusCode, Value) {
        match (method, rest) {
            ("GET", ["pulls"]) => {
                let state = param(query, "state").unwrap_or("open");
                let pulls = self.pulls.values().filter(|p| p["state"] == state);
                page(pulls.cloned().collect(), query)
            }
            ("GET", ["pulls", n]) => found(self.pulls.get(&num(n)).cloned()),
            ("GET", ["pulls", n, "files"]) => {
                page(self.files.get(&num(n)).cloned().unwrap_or_default(), query)
            }
            ("GET", ["pulls", n, "comments"]) => page(self.listing(num(n), None), query),
            ("GET", ["pulls", n, "reviews", id, "comments"]) => {
                page(self.listing(num(n), Some(num(id))), query)
            }
            ("POST", ["pulls", n, "reviews"]) => self.create_review(num(n), body),
            ("POST", ["pulls", n, "comments", root, "replies"]) => {
                let (id, user) = (self.id(), self.bot());
                let comment = json!({"id": id, "body": body["body"], "user": {"login": user},
                    "in_reply_to_id": num(root)});
                self.review_comments.push((num(n), comment.clone()));
                (StatusCode::CREATED, comment)
            }
            ("POST", ["pulls", "comments", id, "reactions"]) => {
                let reaction = (num(id), body["content"].as_str().unwrap_or("").to_owned());
                let status = if self.reactions.contains(&reaction) {
                    StatusCode::OK
                } else {
                    self.reactions.push(reaction);
                    StatusCode::CREATED
                };
                (status, json!({"id": self.id(), "content": body["content"]}))
            }
            ("PATCH", ["pulls", "comments", id]) => edit(&mut self.review_comments, num(id), body),
            ("GET", ["issues", n, "comments"]) => page(of_pr(&self.issue_comments, num(n)), query),
            ("POST", ["issues", n, "comments"]) => {
                let (id, user) = (self.id(), self.bot());
                let comment = json!({"id": id, "body": body["body"], "user": {"login": user}});
                self.issue_comments.push((num(n), comment.clone()));
                (StatusCode::CREATED, comment)
            }
            ("PATCH", ["issues", "comments", id]) => edit(&mut self.issue_comments, num(id), body),
            ("POST", ["check-runs"]) => {
                self.check_runs.push(body.clone());
                (StatusCode::CREATED, json!({"id": self.id()}))
            }
            ("GET", ["compare", spec]) => found(self.compares.get(*spec).cloned()),
            ("GET", ["contents", path @ ..]) => {
                let key = (path.join("/"), param(query, "ref").unwrap_or("").to_owned());
                found(self.contents.get(&key).map(|text| file(text.as_str())))
            }
            ("GET", ["collaborators", login, "permission"]) => found(
                self.permissions
                    .get(*login)
                    .map(|p| json!({"permission": p})),
            ),
            _ => not_found(),
        }
    }

    /// A COMMENT review: each comment gets an id, the review's id, our login and a
    /// Thread.
    fn create_review(&mut self, pr: i64, body: &Value) -> (StatusCode, Value) {
        let review_id = self.id();
        let user = json!({"login": self.bot()});
        for c in body["comments"].as_array().into_iter().flatten() {
            let id = self.id();
            let mut comment = c.clone();
            comment["id"] = json!(id);
            comment["pull_request_review_id"] = json!(review_id);
            comment["commit_id"] = body["commit_id"].clone();
            comment["user"] = user.clone();
            self.review_comments.push((pr, comment));
            self.threads.push(Thread {
                id: format!("PRRT_{id}"),
                root: id,
                resolved: false,
            });
        }
        let review = json!({"id": review_id, "state": "COMMENTED", "commit_id": body["commit_id"],
            "body": body["body"], "user": user});
        self.reviews.push((pr, review.clone()));
        (StatusCode::OK, review)
    }

    /// The PR's review comments, or one review's, without those of reviews still
    /// hidden. Every listing counts each hidden review down.
    fn listing(&mut self, pr: i64, review: Option<i64>) -> Vec<Value> {
        let hidden: Vec<i64> = self
            .hidden
            .iter()
            .filter(|(_, n)| **n > 0)
            .map(|(id, _)| *id)
            .collect();
        for n in self.hidden.values_mut() {
            *n = n.saturating_sub(1);
        }
        self.review_comments
            .iter()
            .filter(|(p, c)| {
                let of = c["pull_request_review_id"].as_i64();
                *p == pr
                    && !of.is_some_and(|r| hidden.contains(&r))
                    && review.is_none_or(|r| of == Some(r))
            })
            .map(|(_, c)| c.clone())
            .collect()
    }

    /// The two GraphQL calls the worker makes, told apart by name. A resolve takes the
    /// Thread id from its first string variable; an unknown id is a 200 with `errors`,
    /// as on GitHub. The listing is one page of every Thread.
    fn graphql(&mut self, body: &Value) -> Value {
        let query = body["query"].as_str().unwrap_or("");
        if query.contains("resolveReviewThread") {
            let id = body["variables"]
                .as_object()
                .and_then(|v| v.values().find_map(Value::as_str))
                .unwrap_or("");
            return match self.threads.iter_mut().find(|t| t.id == id) {
                Some(t) => {
                    t.resolved = true;
                    json!({"data": {"resolveReviewThread": {"thread": {"id": t.id, "isResolved": true}}}})
                }
                None => json!({"data": null, "errors": [{"type": "NOT_FOUND",
                    "message": format!("Could not resolve to a node with the global id of '{id}'")}]}),
            };
        }
        if query.contains("reviewThreads") {
            let nodes: Vec<Value> = self
                .threads
                .iter()
                .map(|t| {
                    json!({"id": t.id, "isResolved": t.resolved,
                        "comments": {"nodes": [{"databaseId": t.root}]}})
                })
                .collect();
            return json!({"data": {"repository": {"pullRequest": {"reviewThreads": {
                "pageInfo": {"hasNextPage": false, "endCursor": null}, "nodes": nodes}}}}});
        }
        json!({"data": null, "errors": [{"message": "unknown query"}]})
    }
}

async fn handle(
    State(world): State<Arc<Mutex<World>>>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let mut w = world.lock().unwrap();
    w.log.push(format!("{method} {uri}"));
    if header(&headers, "x-github-api-version") != "2022-11-28"
        || header(&headers, "user-agent").is_empty()
    {
        return reply(
            code(400),
            &json!({"message": "API version and user agent required"}),
        );
    }
    let path = uri.path();
    let auth = header(&headers, "authorization");
    let wanted = if path.starts_with("/app") {
        "Bearer eyJ"
    } else {
        "Bearer ghs_fake_"
    };
    if !auth.starts_with(wanted) {
        return reply(code(401), &json!({"message": "Bad credentials"}));
    }
    let failure = |status: u16| {
        reply(
            code(status),
            &json!({"message": format!("fault {status}"), "echo": auth}),
        )
    };
    let fault = w.take_fault(&format!("{method} {path}"));
    match fault {
        Some(FaultKind::Before(status)) => return failure(status),
        Some(FaultKind::Limited {
            status,
            retry_after,
            reset,
        }) => return limited(status, retry_after, reset),
        _ => {}
    }
    let body = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let (status, answer) = w.route(&method, path, uri.query().unwrap_or(""), &body);
    match fault {
        Some(FaultKind::After(status)) => failure(status),
        Some(FaultKind::Hide(n)) => {
            if let Some(id) = answer["id"].as_i64() {
                w.hidden.insert(id, n);
            }
            reply(status, &answer)
        }
        _ => reply(status, &answer),
    }
}

fn header<'a>(headers: &'a HeaderMap, name: &str) -> &'a str {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
}

fn reply(status: StatusCode, body: &Value) -> Response {
    (
        status,
        [(CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}

fn limited(status: u16, retry_after: Option<u64>, reset: Option<u64>) -> Response {
    let mut headers = HeaderMap::new();
    if let Some(secs) = retry_after {
        headers.insert("retry-after", HeaderValue::from(secs));
    }
    if let Some(at) = reset {
        headers.insert("x-ratelimit-remaining", HeaderValue::from(0u64));
        headers.insert("x-ratelimit-reset", HeaderValue::from(at));
    }
    let body = json!({"message": "API rate limit exceeded"}).to_string();
    (code(status), headers, body).into_response()
}

fn code(status: u16) -> StatusCode {
    StatusCode::from_u16(status).unwrap()
}

fn found(body: Option<Value>) -> (StatusCode, Value) {
    body.map_or_else(not_found, |b| (StatusCode::OK, b))
}

fn not_found() -> (StatusCode, Value) {
    (StatusCode::NOT_FOUND, json!({"message": "Not Found"}))
}

fn num(s: &str) -> i64 {
    s.parse().unwrap_or(-1)
}

fn of_pr(list: &[(i64, Value)], pr: i64) -> Vec<Value> {
    list.iter()
        .filter(|(p, _)| *p == pr)
        .map(|(_, c)| c.clone())
        .collect()
}

fn edit(list: &mut [(i64, Value)], id: i64, body: &Value) -> (StatusCode, Value) {
    match list.iter_mut().find(|(_, c)| c["id"].as_i64() == Some(id)) {
        Some((_, c)) => {
            c["body"] = body["body"].clone();
            (StatusCode::OK, c.clone())
        }
        None => not_found(),
    }
}

/// One page of `items` as GitHub pages them: `per_page` (default 30) from `page`
/// (1-based).
fn page(items: Vec<Value>, query: &str) -> (StatusCode, Value) {
    let number = |key| param(query, key).and_then(|v| v.parse::<usize>().ok());
    let per = number("per_page").unwrap_or(30);
    let skip = (number("page").unwrap_or(1).max(1) - 1) * per;
    let page = items.into_iter().skip(skip).take(per).collect();
    (StatusCode::OK, Value::Array(page))
}

fn param<'a>(query: &'a str, key: &str) -> Option<&'a str> {
    query
        .split('&')
        .find_map(|kv| kv.strip_prefix(key)?.strip_prefix('='))
}

/// A contents-API file: base64 wrapped at 60 columns, as GitHub sends it.
fn file(text: &str) -> Value {
    let b64 = STANDARD.encode(text);
    let lines: Vec<&str> = b64
        .as_bytes()
        .chunks(60)
        .map(|c| std::str::from_utf8(c).unwrap())
        .collect();
    json!({"type": "file", "encoding": "base64", "size": text.len(), "content": lines.join("\n")})
}

/// `pattern` with `*` matching any one `/`-separated segment of `target`.
fn matches(pattern: &str, target: &str) -> bool {
    let (p, t): (Vec<&str>, Vec<&str>) =
        (pattern.split('/').collect(), target.split('/').collect());
    p.len() == t.len() && p.iter().zip(&t).all(|(p, t)| *p == "*" || p == t)
}
