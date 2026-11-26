//! Jira issue context for the PR's intent: the issue the branch, title or body
//! names, its parent and epic, sub-tasks and links, distilled into a short blob the
//! reviewer reads. A sub-task's parent is its story, whose acceptance criteria the
//! sub-task works toward, so that story's description is kept too. Read-only, HTTP
//! Basic with a service account's email and API token.

use std::fmt::Write as _;
use std::time::Duration;

use regex::Regex;
use reqwest::Client;
use reqwest::header::ACCEPT;
use serde::Deserialize;

use crate::config::{Config, Secret};
use crate::model::http;

/// Bounds the whole lookup: every key tried and the epic.
const TIMEOUT: Duration = Duration::from_secs(10);

/// A Jira key (`PROJ-123`): ASCII word boundaries and digits.
const KEY: &str = r"(?-u:\b)[A-Z][A-Z0-9]+-[0-9]+(?-u:\b)";

const MAX_KEYS: usize = 3;

const FIELDS: &str = "summary,description,status,issuetype,parent,subtasks,issuelinks";

/// How much of an issue's description is kept, in bytes: the whole of a story, whose
/// goal, scope, out of scope and scenarios give its criteria their meaning (a long
/// one runs near 10 KB), as the Ticket fit and the per-file review read it.
const MAX_DESCRIPTION: usize = 12_000;

/// Why no context came back. Messages hold keys and statuses, never the token.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum JiraError {
    #[error("no issue key")]
    NoKey,
    /// A key that is not ASCII letters, digits and `-`, never put in a URL.
    #[error("jira: not an issue key")]
    BadKey,
    /// A key outside `JIRA_PROJECTS`, never sent to Jira.
    #[error("jira {0}: not in JIRA_PROJECTS")]
    Project(String),
    #[error("jira {key}: status {status}")]
    Status { key: String, status: u16 },
    #[error("jira request: {0}")]
    Request(String),
    #[error("jira decode: {0}")]
    Decode(String),
    #[error("jira timed out after {0:?}")]
    Timeout(Duration),
}

/// The distinct Jira keys in `branch`, then `title`, then `body`, at most three.
/// More than one is kept because the pattern also matches `UTF-8` and `SHA-256`;
/// [`Jira::context`] uses the first that exists.
pub fn keys(branch: &str, title: &str, body: &str) -> Vec<String> {
    let Ok(re) = Regex::new(KEY) else {
        return Vec::new();
    };
    let mut out: Vec<String> = Vec::new();
    for m in [branch, title, body]
        .into_iter()
        .flat_map(|s| re.find_iter(s))
    {
        if out.len() == MAX_KEYS {
            break;
        }
        if !out.iter().any(|k| k == m.as_str()) {
            out.push(m.as_str().to_owned());
        }
    }
    out
}

/// The issue a PR names: its key, its context as [`render`] writes it, and the key of
/// the story whose description that context holds, when the issue is a sub-task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ticket {
    pub key: String,
    pub text: String,
    pub story: Option<String>,
}

/// A Jira Cloud site (REST v2, plain-text descriptions).
#[derive(Debug)]
pub struct Jira {
    base: String,
    email: String,
    token: Secret,
    /// `JIRA_PROJECTS`; empty allows any project.
    projects: Vec<String>,
    http: Client,
    timeout: Duration,
}

impl Jira {
    /// The configured site, or `None` unless the base URL, email and API token are
    /// all set.
    pub fn from_config(cfg: &Config) -> Option<Self> {
        if cfg.jira_base_url.is_empty()
            || cfg.jira_email.is_empty()
            || cfg.jira_api_token.is_empty()
        {
            return None;
        }
        Some(Self {
            base: cfg.jira_base_url.trim_end_matches('/').to_owned(),
            email: cfg.jira_email.clone(),
            token: cfg.jira_api_token.clone(),
            projects: cfg.jira_projects.clone(),
            http: http::client().ok()?,
            timeout: TIMEOUT,
        })
    }

    /// The rendered context of the first of `keys` Jira answers 200 for, with its
    /// parent's epic when that fetch succeeds. One timeout bounds it all;
    /// otherwise the last key's error comes back.
    pub async fn context(&self, keys: &[String]) -> Result<Ticket, JiraError> {
        tokio::time::timeout(self.timeout, self.lookup(keys))
            .await
            .map_err(|_| JiraError::Timeout(self.timeout))?
    }

    async fn lookup(&self, keys: &[String]) -> Result<Ticket, JiraError> {
        let mut last = JiraError::NoKey;
        for key in keys {
            match self.issue(key).await {
                Ok(issue) => {
                    let parent = match &issue.fields.parent {
                        Some(p) => self.issue(&p.key).await.ok(),
                        None => None,
                    };
                    return Ok(Ticket {
                        text: render(&issue, parent.as_ref()),
                        story: story(&issue, parent.as_ref()).map(|p| p.key.clone()),
                        key: issue.key,
                    });
                }
                Err(e) => last = e,
            }
        }
        Err(last)
    }

    async fn issue(&self, key: &str) -> Result<Issue, JiraError> {
        // A parent's key comes from Jira's answer: in the path, `..` or `?` would fetch
        // something else.
        if key.is_empty() || !key.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-') {
            return Err(JiraError::BadKey);
        }
        // A PR's author picks its keys, and the service account may read projects they
        // cannot; a parent is held to the same projects.
        let project = key.rsplit_once('-').map_or(key, |(p, _)| p);
        if !self.projects.is_empty() && !self.projects.iter().any(|p| p == project) {
            return Err(JiraError::Project(key.to_owned()));
        }
        let url = format!("{}/rest/api/2/issue/{key}?fields={FIELDS}", self.base);
        let resp = self
            .http
            .get(url)
            .basic_auth(&self.email, Some(self.token.expose()))
            .header(ACCEPT, "application/json")
            .send()
            .await
            .map_err(request)?;
        let status = resp.status().as_u16();
        if status != 200 {
            return Err(JiraError::Status {
                key: key.to_owned(),
                status,
            });
        }
        let body = resp.bytes().await.map_err(request)?;
        serde_json::from_slice(&body).map_err(|e| JiraError::Decode(e.to_string()))
    }
}

/// A reqwest error with its causes (the outer message alone is "error sending
/// request"). It names the URL, which holds no credential.
fn request(e: reqwest::Error) -> JiraError {
    JiraError::Request(http::chain(&e))
}

/// The fields of an issue the context uses.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Issue {
    #[serde(default)]
    key: String,
    #[serde(default)]
    fields: Fields,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct Fields {
    #[serde(default)]
    summary: String,
    description: Option<String>,
    status: Option<Named>,
    issuetype: Option<Named>,
    parent: Option<IssueRef>,
    #[serde(default)]
    subtasks: Vec<IssueRef>,
    #[serde(default)]
    issuelinks: Vec<Link>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct Named {
    #[serde(default)]
    name: String,
    /// An issue type's flag; a status has none.
    #[serde(default)]
    subtask: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct IssueRef {
    #[serde(default)]
    key: String,
    #[serde(default)]
    fields: RefFields,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct RefFields {
    #[serde(default)]
    summary: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct Link {
    #[serde(rename = "inwardIssue")]
    inward: Option<IssueRef>,
    #[serde(rename = "outwardIssue")]
    outward: Option<IssueRef>,
}

/// `parent` (the fetched parent issue) when `issue` is a sub-task: the story the
/// sub-task is part of.
fn story<'a>(issue: &Issue, parent: Option<&'a Issue>) -> Option<&'a Issue> {
    let subtask = issue.fields.issuetype.as_ref().is_some_and(|t| t.subtask);
    parent.filter(|_| subtask)
}

/// `Jira KEY [type, status]: summary`, then the description cut at 12,000 bytes.
fn heading(out: &mut String, label: &str, issue: &Issue) {
    let f = &issue.fields;
    let name = |n: &Option<Named>| n.as_ref().map_or("", |n| n.name.as_str()).to_owned();
    let _ = writeln!(
        out,
        "{label} {} [{}, {}]: {}",
        issue.key,
        name(&f.issuetype),
        name(&f.status),
        f.summary
    );
    let description = f.description.as_deref().unwrap_or("").trim();
    if !description.is_empty() {
        let cut = description.floor_char_boundary(MAX_DESCRIPTION);
        out.push_str(&description[..cut]);
        if cut < description.len() {
            out.push_str("\n… (truncated)");
        }
        out.push('\n');
    }
}

/// The intent blob for `issue`: type, status, summary, description (cut at 12,000
/// bytes), parent (with its own description when it is a sub-task's story), the
/// epic from `parent` (the fetched parent issue), sub-tasks and linked issues.
pub fn render(issue: &Issue, parent: Option<&Issue>) -> String {
    let f = &issue.fields;
    let mut out = String::new();
    heading(&mut out, "Jira", issue);
    if let Some(story) = story(issue, parent) {
        heading(&mut out, "Parent", story);
    } else if let Some(p) = &f.parent {
        let _ = writeln!(out, "Parent {}: {}", p.key, p.fields.summary);
    }
    if f.parent.is_some()
        && let Some(epic) = parent.and_then(|p| p.fields.parent.as_ref())
    {
        let _ = writeln!(out, "Epic {}: {}", epic.key, epic.fields.summary);
    }
    if !f.subtasks.is_empty() {
        out.push_str("Sub-tasks:\n");
        for s in &f.subtasks {
            let _ = writeln!(out, "- {}: {}", s.key, s.fields.summary);
        }
    }
    for l in &f.issuelinks {
        if let Some(r) = l.inward.as_ref().or(l.outward.as_ref()) {
            let _ = writeln!(out, "Related {}: {}", r.key, r.fields.summary);
        }
    }
    out.trim().to_owned()
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::time::Instant;

    use super::*;

    const TOKEN: &str = "jira-token-s3cret";

    /// base64("bot@acme.io:jira-token-s3cret")
    const AUTH: &str = "Basic Ym90QGFjbWUuaW86amlyYS10b2tlbi1zM2NyZXQ=";

    const STORY: &str = r#"{
        "key":"PROJ-123",
        "fields":{
            "summary":"Add login screen",
            "description":"User can sign in with email + password.",
            "status":{"name":"In Progress"},
            "issuetype":{"name":"Story"},
            "parent":{"key":"PROJ-100","fields":{"summary":"Auth epic slice"}},
            "subtasks":[{"key":"PROJ-124","fields":{"summary":"Wire API"}}],
            "issuelinks":[
                {"outwardIssue":{"key":"PROJ-130","fields":{"summary":"Password reset"}}},
                {"inwardIssue":{"key":"PROJ-131","fields":{"summary":"Blocked by SSO"}}},
                {}
            ]
        }}"#;

    const PARENT: &str = r#"{"key":"PROJ-100","fields":{"summary":"Auth epic slice",
        "parent":{"key":"PROJ-1","fields":{"summary":"Q3 Authentication"}}}}"#;

    const RENDERED: &str = "Jira PROJ-123 [Story, In Progress]: Add login screen\n\
        User can sign in with email + password.\n\
        Parent PROJ-100: Auth epic slice\n\
        Epic PROJ-1: Q3 Authentication\n\
        Sub-tasks:\n\
        - PROJ-124: Wire API\n\
        Related PROJ-130: Password reset\n\
        Related PROJ-131: Blocked by SSO";

    fn issue(json: &str) -> Issue {
        serde_json::from_str(json).unwrap()
    }

    fn reply(status: &str, body: &str) -> String {
        format!(
            "HTTP/1.1 {status}\r\ncontent-type: application/json\r\nconnection: close\r\n\
             content-length: {}\r\n\r\n{body}",
            body.len()
        )
    }

    fn read_head(s: &mut TcpStream) -> String {
        let mut head = Vec::new();
        let mut buf = [0; 1024];
        while !head.ends_with(b"\r\n\r\n") {
            match s.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => head.extend_from_slice(&buf[..n]),
            }
        }
        String::from_utf8_lossy(&head).into_owned()
    }

    /// A Jira on a local port: 401 without the right Basic auth, else `respond`'s
    /// answer to the request path (with its query) after `delay`.
    fn serve(delay: Duration, respond: fn(&str, &str) -> String) -> Jira {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let addr = base.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let mut s = stream.unwrap();
                let head = read_head(&mut s);
                let path = head.split(' ').nth(1).unwrap_or("").to_owned();
                let authed = head.lines().any(|l| {
                    l.to_ascii_lowercase().starts_with("authorization:") && l[14..].trim() == AUTH
                });
                std::thread::sleep(delay);
                let resp = if authed {
                    respond(&addr, &path)
                } else {
                    reply("401 Unauthorized", "")
                };
                let _ = s.write_all(resp.as_bytes());
            }
        });
        client_at(&base, Duration::from_secs(5))
    }

    fn client_at(base: &str, timeout: Duration) -> Jira {
        Jira {
            base: base.to_owned(),
            email: "bot@acme.io".to_owned(),
            token: Secret::new(TOKEN),
            projects: Vec::new(),
            http: http::client().unwrap(),
            timeout,
        }
    }

    fn issue_path(key: &str) -> String {
        format!("/rest/api/2/issue/{key}?fields={FIELDS}")
    }

    fn routes(addr: &str, path: &str) -> String {
        let route = |key: &str| path == issue_path(key);
        if route("PROJ-123") {
            reply("200 OK", STORY)
        } else if route("PROJ-100") {
            reply("200 OK", PARENT)
        } else if route("PROJ-200") {
            reply(
                "200 OK",
                r#"{"key":"PROJ-200","fields":{"summary":"Orphan",
                "parent":{"key":"PROJ-201","fields":{"summary":"Gone"}}}}"#,
            )
        } else if route("PROJ-300") {
            reply(
                "200 OK",
                r#"{"key":"PROJ-300","fields":{"summary":"Child",
                "parent":{"key":"x/../PROJ-100","fields":{"summary":"Sneaky"}}}}"#,
            )
        } else if route("MOVED-1") {
            format!(
                "HTTP/1.1 302 Found\r\nlocation: {addr}{}\r\nconnection: close\r\n\
                 content-length: 0\r\n\r\n",
                issue_path("PROJ-123")
            )
        } else if route("BAD-1") {
            reply("200 OK", "<html>")
        } else {
            reply("404 Not Found", "")
        }
    }

    fn assert_no_token(e: &JiraError) {
        assert!(!e.to_string().contains(TOKEN), "{e}");
        assert!(!format!("{e:?}").contains(TOKEN), "{e:?}");
    }

    #[test]
    fn keys_table() {
        let cases: [(&str, &str, &str, &[&str]); 10] = [
            ("feature/PROJ-123-add-login", "", "", &["PROJ-123"]),
            ("no key here", "[ABC-9] title", "body", &["ABC-9"]),
            ("", "", "Closes DEV-4567 in the body", &["DEV-4567"]),
            ("lowercase-12 not a key", "also none", "", &[]),
            ("main", "", "", &[]),
            ("", "UTF-8 fix PROJ-12", "", &["UTF-8", "PROJ-12"]),
            (
                "b/PROJ-1",
                "PROJ-2 and PROJ-1",
                "DEV-3",
                &["PROJ-1", "PROJ-2", "DEV-3"],
            ),
            ("A1-1 B1-2", "C1-3", "D1-4", &["A1-1", "B1-2", "C1-3"]),
            ("éPROJ-7 X-1", "PROJ-77x", "", &["PROJ-7"]),
            ("P-1", "PROJ-1٣", "", &["PROJ-1"]),
        ];
        for (branch, title, body, want) in cases {
            assert_eq!(
                keys(branch, title, body),
                want,
                "{branch:?} {title:?} {body:?}"
            );
        }
    }

    /// A sub-task's fetched parent is its story: its heading and description are kept,
    /// and the epic above it stays one line. A story's parent (the epic) is one line.
    #[test]
    fn a_subtask_keeps_its_storys_description() {
        let task = r#"{"key":"PROJ-7","fields":{"summary":"[FE] Dialog",
            "description":"Show the dialog.","status":{"name":"Done"},
            "issuetype":{"name":"Sub-task","subtask":true},
            "parent":{"key":"PROJ-6","fields":{"summary":"Confirm a changed total"}}}}"#;
        let story = r#"{"key":"PROJ-6","fields":{"summary":"Confirm a changed total",
            "description":"Acceptance Criteria\n- The buttons cannot be pressed twice",
            "status":{"name":"In Progress"},"issuetype":{"name":"Story"},
            "parent":{"key":"PROJ-1","fields":{"summary":"Checkout"}}}}"#;
        let want = "Jira PROJ-7 [Sub-task, Done]: [FE] Dialog\n\
            Show the dialog.\n\
            Parent PROJ-6 [Story, In Progress]: Confirm a changed total\n\
            Acceptance Criteria\n\
            - The buttons cannot be pressed twice\n\
            Epic PROJ-1: Checkout";
        assert_eq!(render(&issue(task), Some(&issue(story))), want);
        assert_eq!(story_key(task, Some(story)).as_deref(), Some("PROJ-6"));

        // A story names its epic in one line.
        assert_eq!(story_key(STORY, Some(PARENT)), None);
    }

    fn story_key(task: &str, parent: Option<&str>) -> Option<String> {
        let parent = parent.map(issue);
        story(&issue(task), parent.as_ref()).map(|p| p.key.clone())
    }

    #[test]
    fn render_cases() {
        assert_eq!(render(&issue(STORY), Some(&issue(PARENT))), RENDERED);

        // The parent fetch failed: no epic line.
        let without_epic = RENDERED.replace("Epic PROJ-1: Q3 Authentication\n", "");
        assert_eq!(render(&issue(STORY), None), without_epic);

        let bare = r#"{"key":"X-1","fields":{"summary":"S","description":null,"status":null}}"#;
        assert_eq!(render(&issue(bare), None), "Jira X-1 [, ]: S");

        // 11,999 ASCII bytes, then a 2-byte char straddling the 12,000-byte cut.
        let long = format!("{}é tail", "a".repeat(11_999));
        let json = serde_json::json!({"key": "X-2", "fields": {"summary": "S", "description": format!("  {long}  ")}});
        let got = render(&issue(&json.to_string()), None);
        let want = format!("Jira X-2 [, ]: S\n{}\n… (truncated)", "a".repeat(11_999));
        assert_eq!(got, want);

        let exact = "é".repeat(6000);
        let json =
            serde_json::json!({"key": "X-3", "fields": {"summary": "S", "description": exact}});
        let got = render(&issue(&json.to_string()), None);
        assert_eq!(got, format!("Jira X-3 [, ]: S\n{exact}"));
    }

    #[tokio::test]
    async fn context_uses_the_first_key_jira_has() {
        let jira = serve(Duration::ZERO, routes);
        let found = keys("feature/x", "UTF-8 fix PROJ-123", "");
        assert_eq!(found, ["UTF-8", "PROJ-123"]);
        let got = jira.context(&found).await.unwrap();
        assert_eq!(
            (got.key.as_str(), got.text.as_str()),
            ("PROJ-123", RENDERED)
        );
        assert_eq!(got.story, None, "a story's parent is its epic");

        // The epic's fetch fails (404): the rest still renders.
        let got = jira.context(&["PROJ-200".to_owned()]).await.unwrap();
        assert_eq!(
            got.text,
            "Jira PROJ-200 [, ]: Orphan\nParent PROJ-201: Gone"
        );

        // A parent key that is no Jira key never reaches a URL path, where `..` would
        // fetch another issue.
        let got = jira.context(&["PROJ-300".to_owned()]).await.unwrap();
        assert_eq!(
            got.text,
            "Jira PROJ-300 [, ]: Child\nParent x/../PROJ-100: Sneaky"
        );

        let cases = [
            (vec![], JiraError::NoKey),
            (
                vec!["NOPE-1".to_owned(), "NOPE-2".to_owned()],
                JiraError::Status {
                    key: "NOPE-2".to_owned(),
                    status: 404,
                },
            ),
            // A redirect is an answer, never followed: it would carry the token along.
            (
                vec!["MOVED-1".to_owned()],
                JiraError::Status {
                    key: "MOVED-1".to_owned(),
                    status: 302,
                },
            ),
        ];
        for (keys, want) in cases {
            let err = jira.context(&keys).await.unwrap_err();
            assert_eq!(err, want, "{keys:?}");
            assert_no_token(&err);
        }
        let err = jira.context(&["BAD-1".to_owned()]).await.unwrap_err();
        assert!(matches!(err, JiraError::Decode(_)), "{err:?}");
    }

    #[tokio::test]
    async fn wrong_credentials_and_no_server_never_show_the_token() {
        let jira = serve(Duration::ZERO, routes);
        let wrong = Jira {
            token: Secret::new("not-the-token"),
            ..jira
        };
        let err = wrong.context(&["PROJ-123".to_owned()]).await.unwrap_err();
        assert_eq!(
            err,
            JiraError::Status {
                key: "PROJ-123".to_owned(),
                status: 401
            }
        );

        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let gone = client_at(&format!("http://127.0.0.1:{port}"), Duration::from_secs(5));
        let err = gone.context(&["PROJ-123".to_owned()]).await.unwrap_err();
        assert!(matches!(err, JiraError::Request(_)), "{err:?}");
        assert_no_token(&err);
    }

    /// With `JIRA_PROJECTS` set, a key in another project is refused before any request:
    /// this client points at a closed port, where a request would fail as `Request`.
    #[tokio::test]
    async fn a_key_outside_the_projects_is_never_sent() {
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let gone = Jira {
            projects: vec!["PROJ".to_owned()],
            ..client_at(&format!("http://127.0.0.1:{port}"), Duration::from_secs(5))
        };
        for key in ["SEC-9", "PROJX-1", "UTF-8"] {
            let got = gone.context(&[key.to_owned()]).await;
            assert_eq!(got, Err(JiraError::Project(key.to_owned())), "{key}");
        }
        let sent = gone.context(&["PROJ-1".to_owned()]).await;
        assert!(matches!(sent, Err(JiraError::Request(_))), "{sent:?}");

        // The first key in the projects that Jira has is used, its parent with it.
        let jira = Jira {
            projects: vec!["PROJ".to_owned()],
            ..serve(Duration::ZERO, routes)
        };
        let keys = ["SEC-9", "PROJ-123"].map(str::to_owned);
        assert_eq!(jira.context(&keys).await.unwrap().text, RENDERED);
    }

    /// Each answer takes 300 ms, so three keys fit a per-request timeout of 500 ms
    /// but not the one 500 ms bound around the whole lookup.
    #[tokio::test]
    async fn a_slow_jira_hits_the_one_timeout() {
        let slow = serve(Duration::from_millis(300), routes);
        let slow = Jira {
            timeout: Duration::from_millis(500),
            ..slow
        };
        let keys = ["NOPE-1", "NOPE-2", "PROJ-123"].map(str::to_owned);
        let started = Instant::now();
        let err = slow.context(&keys).await.unwrap_err();
        assert_eq!(err, JiraError::Timeout(Duration::from_millis(500)));
        assert!(started.elapsed() < Duration::from_secs(3));
        assert_no_token(&err);
    }

    #[test]
    fn from_config_needs_all_three() {
        let base = |url: &str, email: &str, token: &str| {
            let vars = [
                ("JIRA_BASE_URL", url),
                ("JIRA_EMAIL", email),
                ("JIRA_API_TOKEN", token),
            ];
            Config::from_lookup(|k| {
                vars.iter()
                    .find(|(name, _)| *name == k)
                    .map(|(_, v)| (*v).to_owned())
            })
            .unwrap()
        };
        let jira = Jira::from_config(&base("https://acme.atlassian.net/", "bot@acme.io", "t"));
        assert_eq!(
            jira.map(|j| j.base).as_deref(),
            Some("https://acme.atlassian.net")
        );
        assert!(Jira::from_config(&base("", "bot@acme.io", "t")).is_none());
        assert!(Jira::from_config(&base("https://a", "", "t")).is_none());
        assert!(Jira::from_config(&base("https://a", "bot@acme.io", "")).is_none());
    }
}
