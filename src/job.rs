//! What a webhook delivery asks for: the [`Job`] it queues, read from the payload alone.
//! Pure, with no GitHub or model call, so the webhook answers well inside GitHub's 10 s.

use std::sync::{LazyLock, OnceLock};
use std::time::{Duration, SystemTime};

use regex::Regex;
use serde::de::IgnoredAny;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

/// The mention before [`set_mention`] runs: in dev mode, tests and diffagent-eval.
const DEFAULT_MENTION: &str = "@diffagent";

static MENTION: OnceLock<String> = OnceLock::new();

/// How a comment addresses the bot: `@<the App's slug>`, so any operator's App answers
/// to its own handle.
/// ponytail: one global, as a process runs one App (one `GITHUB_APP_ID`); carry it per
/// job if a process ever serves several Apps.
pub fn mention() -> &'static str {
    MENTION.get().map_or(DEFAULT_MENTION, String::as_str)
}

/// Sets [`mention`] from the App's `bot_login` (`<slug>[bot]`), once at startup, before
/// any webhook or job is handled: the patterns built from it compile on first use. A
/// second call changes nothing.
pub fn set_mention(bot_login: &str) {
    MENTION.get_or_init(|| mention_of(bot_login));
}

/// `@<slug>` from a bot login, `<slug>[bot]`.
fn mention_of(bot_login: &str) -> String {
    format!("@{}", bot_login.strip_suffix("[bot]").unwrap_or(bot_login))
}

/// A push's Review waits this long, so a burst of pushes (a rebase, fixup spam) collapses:
/// by the time the first job runs, the head has moved and it stops.
const SYNC_DELAY: Duration = Duration::from_secs(30);

/// One unit of work, queued as JSON in `jobs.payload`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Job {
    pub installation_id: i64,
    /// GitHub's repository id, from the payload: the store and identity are keyed by it.
    pub repo_id: i64,
    pub owner: String,
    pub repo: String,
    pub pr: i64,
    /// Who triggered it: the commenter, or the PR event's sender.
    pub actor: String,
    pub kind: Kind,
    /// When the comment that queued it was posted, on GitHub's clock: a command acts on
    /// what was there then. `None` for a PR event, and in a payload without it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commented_at: Option<SystemTime>,
    /// The actor pushed the PR's head: the sender of an `opened`, `reopened` or
    /// `synchronize` event. Nobody who pushed to a PR is a Maintainer on it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub pushed: bool,
}

/// What a job does. Variants are append-only, so a payload queued before a deploy still
/// decodes after it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// `command` when the `review` command asked for it, which also unpauses the PR.
    Review {
        command: bool,
    },
    Pause,
    Resolve,
    /// The `lessons` command: list the repo's active Lessons.
    Lessons,
    /// The `forget` command. `id` is the Lesson it names (`forget N`); `root` is the
    /// Thread it was replied in, whose Lesson it forgets when it names none, and
    /// `reply` the reply's own id, which the note posted in the Thread is marked with.
    Forget {
        id: Option<i64>,
        root: Option<i64>,
        #[serde(default)]
        reply: Option<i64>,
    },
    /// A reply in a Thread that is not a `forget`: `root` is the Thread's first comment,
    /// `reply` the reply's own id, and `head_sha` the PR head when it was posted, which
    /// its usage rows name.
    Reply {
        root: i64,
        reply: i64,
        body: String,
        head_sha: String,
    },
    /// The `learn` command: confirms the Lesson proposed in the Thread rooted at `root`,
    /// with `reply` the command's own id, which the note posted in the Thread is marked
    /// with. Both are `None` for a top-level `learn`, which has no Thread.
    Learn {
        root: Option<i64>,
        reply: Option<i64>,
    },
    /// The `close` command: suppresses the Finding of the Thread rooted at `root`, with
    /// no Lesson, and `reply` is the command's own id, which the note posted in the
    /// Thread is marked with. Both are `None` for a top-level `close`, which has no
    /// Thread.
    Close {
        root: Option<i64>,
        reply: Option<i64>,
    },
}

impl Job {
    /// The slot the job runs in: one job per PR, and one per Thread, runs at a time. A
    /// reply writes no PR state, so it runs in its Thread's slot and never waits behind
    /// a Review; `forget` reopens a Finding a Review may be reading, so it takes the PR's.
    pub fn slot(&self) -> String {
        match &self.kind {
            Kind::Reply { root, .. } => format!("thread:{}:{root}", self.repo_id),
            _ => pr_slot(self.repo_id, self.pr),
        }
    }
}

/// The slot of PR `pr`'s jobs, which every kind but a Thread reply runs in.
pub fn pr_slot(repo_id: i64, pr: i64) -> String {
    format!("pr:{repo_id}:{pr}")
}

/// The job a delivery of `event` (`X-GitHub-Event`) asks for, and how long it waits
/// before it is due. `None` when there is nothing to do; an error when the payload of an
/// event we handle does not decode.
pub fn from_event(event: &str, body: &[u8]) -> Result<Option<(Job, Duration)>, serde_json::Error> {
    Ok(match event {
        "pull_request" => pull_request(serde_json::from_slice(body)?),
        "issue_comment" => issue_comment(serde_json::from_slice(body)?),
        "pull_request_review_comment" => review_comment(serde_json::from_slice(body)?),
        _ => None,
    })
}

/// A PR opened, reopened, marked ready or pushed to. A draft waits until it is marked
/// ready. Senders are not filtered, so dependabot's PRs are reviewed.
fn pull_request(e: PullRequestEvent) -> Option<(Job, Duration)> {
    let delay = match e.action.as_str() {
        "opened" | "reopened" | "ready_for_review" => Duration::ZERO,
        "synchronize" => SYNC_DELAY,
        _ => return None,
    };
    if e.pull_request.draft && e.action != "ready_for_review" {
        return None;
    }
    let kind = Kind::Review { command: false };
    let job = Job {
        pushed: e.action != "ready_for_review",
        ..job(e.repository, e.installation, e.sender, e.number, kind)
    };
    Some((job, delay))
}

/// A command in a new comment on a PR. A Bot's comment is dropped: without the mention
/// filter the bot could answer itself.
fn issue_comment(e: IssueCommentEvent) -> Option<(Job, Duration)> {
    if e.action != "created" || e.issue.pull_request.is_none() || e.sender.kind == "Bot" {
        return None;
    }
    let kind = command(&e.comment.body)?;
    let job = Job {
        commented_at: e.comment.created_at.as_deref().and_then(instant),
        ..job(e.repository, e.installation, e.sender, e.issue.number, kind)
    };
    Some((job, Duration::ZERO))
}

/// An RFC 3339 time, as GitHub writes them; `None` when it is not one.
fn instant(text: &str) -> Option<SystemTime> {
    OffsetDateTime::parse(text, &Rfc3339)
        .ok()
        .map(SystemTime::from)
}

/// A reply in a Thread: a new review comment with `in_reply_to_id`, which GitHub sets to
/// the Thread's first comment. `forget`, as a command or the bare word, forgets the
/// Thread's Lesson in the PR slot, `learn` confirms the Lesson proposed there and
/// `close` closes the Thread's Finding, the same way; anything else is a reply, mention
/// or not, and is
/// classified, since few replies mention the bot. Other commands are PR-wide, so a
/// Thread does not run them: a `resolve` meant for one Finding would close every one. A
/// Bot's reply is dropped, so the bot never answers itself.
fn review_comment(e: ReviewCommentEvent) -> Option<(Job, Duration)> {
    let root = e.comment.in_reply_to_id?;
    if e.action != "created" || e.sender.kind == "Bot" {
        return None;
    }
    let bare = |word: &str| e.comment.body.trim().eq_ignore_ascii_case(word);
    let kind = match command(&e.comment.body) {
        Some(Kind::Forget { id, .. }) => Kind::Forget {
            id,
            root: Some(root),
            reply: Some(e.comment.id),
        },
        None if bare("forget") => Kind::Forget {
            id: None,
            root: Some(root),
            reply: Some(e.comment.id),
        },
        Some(Kind::Learn { .. }) => Kind::Learn {
            root: Some(root),
            reply: Some(e.comment.id),
        },
        None if bare("learn") => Kind::Learn {
            root: Some(root),
            reply: Some(e.comment.id),
        },
        Some(Kind::Close { .. }) => Kind::Close {
            root: Some(root),
            reply: Some(e.comment.id),
        },
        None if bare("close") => Kind::Close {
            root: Some(root),
            reply: Some(e.comment.id),
        },
        _ => Kind::Reply {
            root,
            reply: e.comment.id,
            body: e.comment.body,
            head_sha: e.pull_request.head.sha,
        },
    };
    let pr = e.pull_request.number;
    Some((
        job(e.repository, e.installation, e.sender, pr, kind),
        Duration::ZERO,
    ))
}

/// [`command_pattern`] for our [`mention`]. `None` only if it failed to compile, which a
/// test rules out.
static COMMAND: LazyLock<Option<Regex>> = LazyLock::new(|| command_pattern(mention()));

/// The command pattern: `mention`, then a command word, at the start of any line,
/// case-insensitive. Words after the command are ignored, except the Lesson id after
/// `forget` (`forget 42`, or `forget #42`). `approve` waits for the merge gate.
fn command_pattern(mention: &str) -> Option<Regex> {
    Regex::new(&format!(
        r"(?im)^\s*{}\s+(review|resolve|pause|lessons|forget|learn|close)\b(?:[ \t]+#?([0-9]+)\b)?",
        regex::escape(mention)
    ))
    .ok()
}

fn command(body: &str) -> Option<Kind> {
    parse_command(COMMAND.as_ref()?, body)
}

fn parse_command(pattern: &Regex, body: &str) -> Option<Kind> {
    let caps = pattern.captures(body)?;
    match caps.get(1)?.as_str().to_ascii_lowercase().as_str() {
        "review" => Some(Kind::Review { command: true }),
        "pause" => Some(Kind::Pause),
        "resolve" => Some(Kind::Resolve),
        "lessons" => Some(Kind::Lessons),
        "forget" => Some(Kind::Forget {
            id: caps.get(2).and_then(|n| n.as_str().parse().ok()),
            root: None,
            reply: None,
        }),
        "learn" => Some(Kind::Learn {
            root: None,
            reply: None,
        }),
        "close" => Some(Kind::Close {
            root: None,
            reply: None,
        }),
        _ => None,
    }
}

fn job(
    repository: Repository,
    installation: Installation,
    sender: Sender,
    pr: i64,
    kind: Kind,
) -> Job {
    Job {
        installation_id: installation.id,
        repo_id: repository.id,
        owner: repository.owner.login,
        repo: repository.name,
        pr,
        actor: sender.login,
        kind,
        commented_at: None,
        pushed: false,
    }
}

// The payloads, with only the fields used.

#[derive(Deserialize)]
struct PullRequestEvent {
    action: String,
    number: i64,
    pull_request: PullRequest,
    repository: Repository,
    installation: Installation,
    sender: Sender,
}

#[derive(Deserialize)]
struct PullRequest {
    #[serde(default)]
    draft: bool,
}

#[derive(Deserialize)]
struct IssueCommentEvent {
    action: String,
    issue: Issue,
    comment: Comment,
    repository: Repository,
    installation: Installation,
    sender: Sender,
}

#[derive(Deserialize)]
struct Issue {
    number: i64,
    /// Present only when the issue is a PR.
    pull_request: Option<IgnoredAny>,
}

#[derive(Deserialize)]
struct Comment {
    body: String,
    #[serde(default)]
    created_at: Option<String>,
}

#[derive(Deserialize)]
struct ReviewCommentEvent {
    action: String,
    comment: ReviewComment,
    pull_request: ReviewedPull,
    repository: Repository,
    installation: Installation,
    sender: Sender,
}

#[derive(Deserialize)]
struct ReviewComment {
    id: i64,
    body: String,
    /// The Thread's first comment; absent on a comment that starts a Thread.
    in_reply_to_id: Option<i64>,
}

#[derive(Deserialize)]
struct ReviewedPull {
    number: i64,
    head: Head,
}

#[derive(Deserialize)]
struct Head {
    sha: String,
}

#[derive(Deserialize)]
struct Repository {
    id: i64,
    name: String,
    owner: Owner,
}

#[derive(Deserialize)]
struct Owner {
    login: String,
}

#[derive(Deserialize)]
struct Sender {
    login: String,
    /// `User`, `Bot` or `Organization`.
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Deserialize)]
struct Installation {
    id: i64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    const DATA: &str = "tests/data/webhook";

    /// The fixture `name`, with each JSON pointer in `edits` set to its value.
    fn payload(name: &str, edits: &[(&str, Value)]) -> Vec<u8> {
        let raw = std::fs::read(format!("{DATA}/{name}.json")).expect("fixture");
        let mut v: Value = serde_json::from_slice(&raw).expect("fixture is JSON");
        for (pointer, value) in edits {
            *v.pointer_mut(pointer).expect("pointer exists") = value.clone();
        }
        serde_json::to_vec(&v).expect("encode")
    }

    fn review(pr: i64, actor: &str, command: bool) -> Job {
        Job {
            installation_id: 61_234_567,
            repo_id: 812_345_678,
            owner: "acme".into(),
            repo: "shop".into(),
            pr,
            actor: actor.into(),
            kind: Kind::Review { command },
            commented_at: None,
            pushed: false,
        }
    }

    /// When the fixture's comment was posted: 2026-11-20T09:30:00Z.
    fn commented() -> Option<SystemTime> {
        Some(SystemTime::UNIX_EPOCH + Duration::from_secs(1_795_167_000))
    }

    /// The sender of an event that sets the head is its pusher; marking a PR ready
    /// pushes nothing.
    #[test]
    fn pull_request_actions() {
        let secs = Duration::from_secs;
        // (action, draft, the delay of the job queued, its sender pushed)
        let cases: [(&str, bool, Option<Duration>, bool); 9] = [
            ("opened", false, Some(secs(0)), true),
            ("reopened", false, Some(secs(0)), true),
            ("ready_for_review", false, Some(secs(0)), false),
            ("synchronize", false, Some(secs(30)), true),
            ("closed", false, None, false),
            ("edited", false, None, false),
            ("opened", true, None, false),
            ("synchronize", true, None, false),
            ("ready_for_review", true, Some(secs(0)), false),
        ];
        for (action, draft, delay, pushed) in cases {
            let body = payload(
                "pull_request",
                &[
                    ("/action", json!(action)),
                    ("/pull_request/draft", json!(draft)),
                ],
            );
            let got = from_event("pull_request", &body).expect("decodes");
            let job = Job {
                pushed,
                ..review(7, "alice", false)
            };
            let want = delay.map(|delay| (job, delay));
            assert_eq!(got, want, "{action} draft={draft}");
        }
    }

    #[test]
    fn ids_come_from_the_payload() {
        let body = payload(
            "pull_request",
            &[
                ("/repository/id", json!(42)),
                ("/installation/id", json!(9)),
                ("/number", json!(8)),
            ],
        );
        let (job, _) = from_event("pull_request", &body).unwrap().unwrap();
        assert_eq!((job.repo_id, job.installation_id, job.pr), (42, 9, 8));
        assert_eq!(job.slot(), "pr:42:8");
    }

    #[test]
    fn commands() {
        let forget = |id| {
            Some(Kind::Forget {
                id,
                root: None,
                reply: None,
            })
        };
        let learn = Some(Kind::Learn {
            root: None,
            reply: None,
        });
        let close = Some(Kind::Close {
            root: None,
            reply: None,
        });
        let cases: [(&str, Option<Kind>); 31] = [
            ("@diffagent review", Some(Kind::Review { command: true })),
            ("@diffagent resolve", Some(Kind::Resolve)),
            ("@diffagent pause", Some(Kind::Pause)),
            ("@DIFFAGENT Review", Some(Kind::Review { command: true })),
            (
                "please\n@diffagent review",
                Some(Kind::Review { command: true }),
            ),
            ("   @diffagent pause", Some(Kind::Pause)),
            (
                "@diffagent review the whole thing",
                Some(Kind::Review { command: true }),
            ),
            ("@diffagent  pause\r\nthanks", Some(Kind::Pause)),
            ("@diffagent reviewed", None),
            ("@diffagent approve because", None),
            ("@diffagent lessons", Some(Kind::Lessons)),
            ("@diffagent lessons please", Some(Kind::Lessons)),
            ("@diffagent forget 42", forget(Some(42))),
            ("@diffagent FORGET #7 it was wrong", forget(Some(7))),
            ("@diffagent forget 42.", forget(Some(42))),
            ("@diffagent forget", forget(None)),
            ("@diffagent forget abc", forget(None)),
            ("@diffagent forget 42abc", forget(None)),
            // The id must be on the command's line.
            ("@diffagent forget\n42", forget(None)),
            ("@diffagent forget 99999999999999999999", forget(None)),
            ("@diffagent forgetful", None),
            ("@diffagent learn", learn.clone()),
            ("@DiffAgent LEARN this one", learn),
            ("@diffagent learned", None),
            ("@diffagent close", close.clone()),
            ("@diffagent Close it, this is handled", close),
            ("@diffagent closed", None),
            ("@diffagent summarize", None),
            ("ping @diffagent review", None),
            ("looks good to me", None),
            ("", None),
        ];
        for (body, want) in cases {
            let edits = [("/comment/body", json!(body))];
            let got = from_event("issue_comment", &payload("issue_comment", &edits)).unwrap();
            assert_eq!(got.map(|(j, _)| j.kind), want, "{body:?}");
        }
    }

    /// Another operator's App answers to its own slug, and only to it.
    #[test]
    fn commands_follow_the_app_slug() {
        let acme = command_pattern(&mention_of("acme-review[bot]")).unwrap();
        let cases = [
            ("@acme-review review", Some(Kind::Review { command: true })),
            ("@diffagent review", None),
            ("@acme-reviewer review", None),
        ];
        for (body, want) in cases {
            assert_eq!(parse_command(&acme, body), want, "{body:?}");
        }
    }

    #[test]
    fn issue_comments() {
        let commenter = Job {
            commented_at: commented(),
            ..review(9, "bob", true)
        };
        type Edits<'a> = &'a [(&'a str, Value)];
        let cases: [(&str, Edits, bool); 5] = [
            ("a command", &[], true),
            ("edited", &[("/action", json!("edited"))], false),
            ("deleted", &[("/action", json!("deleted"))], false),
            (
                "on an issue",
                &[("/issue/pull_request", Value::Null)],
                false,
            ),
            ("by a Bot", &[("/sender/type", json!("Bot"))], false),
        ];
        for (name, edits, queued) in cases {
            let got = from_event("issue_comment", &payload("issue_comment", edits)).unwrap();
            let want = queued.then(|| (commenter.clone(), Duration::ZERO));
            assert_eq!(got, want, "{name}");
        }
    }

    /// A reply in a Thread, rooted at comment 2612345600 on PR 9.
    #[test]
    fn review_comments() {
        const ROOT: i64 = 2_612_345_600;
        let reply = |body: &str| Kind::Reply {
            root: ROOT,
            reply: 2_612_345_678,
            body: body.into(),
            head_sha: "4b2d6f8a0c1e3a5b7d9f1e3c5a7b9d1f3e5a7c9b".into(),
        };
        let forget = |id| Kind::Forget {
            id,
            root: Some(ROOT),
            reply: Some(2_612_345_678),
        };
        let learn = Kind::Learn {
            root: Some(ROOT),
            reply: Some(2_612_345_678),
        };
        let close = Kind::Close {
            root: Some(ROOT),
            reply: Some(2_612_345_678),
        };
        let thread = "thread:812345678:2612345600";
        let pr = "pr:812345678:9";
        let fixture =
            "This is intended: the retry is idempotent because the callback carries its own key.";
        let body = |b: &str| ("/comment/body", json!(b));
        // (name, edits, the kind and slot queued).
        type Case<'a> = (&'a str, &'a [(&'a str, Value)], Option<(Kind, &'a str)>);
        let cases: [Case; 20] = [
            (
                "a reply, no mention needed",
                &[],
                Some((reply(fixture), thread)),
            ),
            (
                "a mention",
                &[body("@diffagent kenapa seperti ini?")],
                Some((reply("@diffagent kenapa seperti ini?"), thread)),
            ),
            (
                "not a reply: it starts a Thread",
                &[("/comment/in_reply_to_id", Value::Null)],
                None,
            ),
            ("edited", &[("/action", json!("edited"))], None),
            ("deleted", &[("/action", json!("deleted"))], None),
            ("by a Bot", &[("/sender/type", json!("Bot"))], None),
            (
                "forget this Thread's Lesson",
                &[body("@diffagent forget")],
                Some((forget(None), pr)),
            ),
            (
                "forget a named Lesson",
                &[body("@diffagent forget 42")],
                Some((forget(Some(42)), pr)),
            ),
            (
                "the bare word",
                &[body(" Forget\n")],
                Some((forget(None), pr)),
            ),
            (
                "learn the Lesson proposed here",
                &[body("@diffagent learn")],
                Some((learn.clone(), pr)),
            ),
            ("the bare learn", &[body("Learn\n")], Some((learn, pr))),
            (
                "close this Thread's Finding",
                &[body("@diffagent close")],
                Some((close.clone(), pr)),
            ),
            ("the bare close", &[body(" close ")], Some((close, pr))),
            (
                "close in a sentence is a reply",
                &[body("close enough, but why?")],
                Some((reply("close enough, but why?"), thread)),
            ),
            (
                "learn in a sentence is a reply",
                &[body("learn from this, it is fine")],
                Some((reply("learn from this, it is fine"), thread)),
            ),
            (
                "forget in a sentence is a reply",
                &[body("forget it, this is fine")],
                Some((reply("forget it, this is fine"), thread)),
            ),
            (
                "resolve is PR-wide, so a reply",
                &[body("@diffagent resolve, this is handled upstream")],
                Some((
                    reply("@diffagent resolve, this is handled upstream"),
                    thread,
                )),
            ),
            (
                "pause is a reply",
                &[body("@diffagent pause")],
                Some((reply("@diffagent pause"), thread)),
            ),
            (
                "lessons is a reply",
                &[body("@diffagent lessons")],
                Some((reply("@diffagent lessons"), thread)),
            ),
            (
                "review is a reply",
                &[body("@diffagent review why is this flagged")],
                Some((reply("@diffagent review why is this flagged"), thread)),
            ),
        ];
        for (name, edits, want) in cases {
            let body = payload("pull_request_review_comment", edits);
            let got = from_event("pull_request_review_comment", &body).unwrap();
            let got = got.map(|(j, delay)| {
                assert_eq!(delay, Duration::ZERO, "{name}");
                assert_eq!((j.repo_id, j.pr, j.actor.as_str()), (812_345_678, 9, "bob"));
                let slot = j.slot();
                (j.kind, slot)
            });
            let want = want.map(|(k, slot)| (k, slot.to_owned()));
            assert_eq!(got, want, "{name}");
        }
    }

    #[test]
    fn bot_pull_requests_are_reviewed() {
        let edits = [
            ("/sender/login", json!("dependabot[bot]")),
            ("/sender/type", json!("Bot")),
        ];
        let got = from_event("pull_request", &payload("pull_request", &edits)).unwrap();
        assert_eq!(
            got.map(|(j, _)| j.actor).as_deref(),
            Some("dependabot[bot]")
        );
    }

    #[test]
    fn other_events_and_bad_payloads() {
        let unused = ["ping", "push", "installation", "pull_request_review"];
        for event in unused {
            assert_eq!(from_event(event, b"not json").unwrap(), None, "{event}");
        }
        for event in [
            "pull_request",
            "issue_comment",
            "pull_request_review_comment",
        ] {
            assert!(from_event(event, b"not json").is_err(), "{event}");
        }
        let no_repo_id = payload("pull_request", &[("/repository/id", Value::Null)]);
        assert!(from_event("pull_request", &no_repo_id).is_err());
    }

    #[test]
    fn job_round_trips_as_json() {
        let kinds = [
            Kind::Review { command: true },
            Kind::Pause,
            Kind::Resolve,
            Kind::Lessons,
            Kind::Forget {
                id: Some(42),
                root: None,
                reply: None,
            },
            Kind::Reply {
                root: 1,
                reply: 2,
                body: "why?".into(),
                head_sha: "abc".into(),
            },
            Kind::Learn {
                root: Some(1),
                reply: Some(2),
            },
            Kind::Close {
                root: Some(1),
                reply: Some(2),
            },
            Kind::Close {
                root: None,
                reply: None,
            },
        ];
        for (kind, commented_at) in kinds
            .into_iter()
            .flat_map(|k| [(k.clone(), None), (k, commented())])
        {
            let job = Job {
                kind,
                commented_at,
                ..review(7, "alice", false)
            };
            let text = serde_json::to_string(&job).unwrap();
            assert_eq!(serde_json::from_str::<Job>(&text).unwrap(), job, "{text}");
        }
        // The stored form: queued payloads must keep decoding across deploys.
        let text = serde_json::to_string(&review(7, "alice", true)).unwrap();
        assert!(
            text.contains(r#""kind":{"review":{"command":true}}"#),
            "{text}"
        );
    }

    /// The queued wire form of every `Kind`, as literals: a payload queued before a
    /// deploy must decode after it, so a change here is a change to the frozen contract.
    /// Each encodes back to its literal, except `forget` without `reply`, the form
    /// queued before `reply` was added.
    #[test]
    fn queued_payloads_keep_their_wire_form() {
        let head = r#"{"installation_id":61234567,"repo_id":812345678,"owner":"acme","repo":"shop","pr":7,"actor":"alice","kind":"#;
        let forget = |id, root, reply| Kind::Forget { id, root, reply };
        let cases = [
            (
                r#"{"review":{"command":true}}"#,
                Kind::Review { command: true },
                true,
            ),
            (
                r#"{"review":{"command":false}}"#,
                Kind::Review { command: false },
                true,
            ),
            (r#""pause""#, Kind::Pause, true),
            (r#""resolve""#, Kind::Resolve, true),
            (r#""lessons""#, Kind::Lessons, true),
            (
                r#"{"forget":{"id":42,"root":null,"reply":null}}"#,
                forget(Some(42), None, None),
                true,
            ),
            (
                r#"{"forget":{"id":null,"root":500,"reply":501}}"#,
                forget(None, Some(500), Some(501)),
                true,
            ),
            (
                r#"{"forget":{"id":null,"root":500}}"#,
                forget(None, Some(500), None),
                false,
            ),
            (
                r#"{"reply":{"root":500,"reply":501,"body":"why?","head_sha":"abc"}}"#,
                Kind::Reply {
                    root: 500,
                    reply: 501,
                    body: "why?".into(),
                    head_sha: "abc".into(),
                },
                true,
            ),
            (
                r#"{"learn":{"root":500,"reply":501}}"#,
                Kind::Learn {
                    root: Some(500),
                    reply: Some(501),
                },
                true,
            ),
            (
                r#"{"learn":{"root":null,"reply":null}}"#,
                Kind::Learn {
                    root: None,
                    reply: None,
                },
                true,
            ),
            (
                r#"{"close":{"root":500,"reply":501}}"#,
                Kind::Close {
                    root: Some(500),
                    reply: Some(501),
                },
                true,
            ),
            (
                r#"{"close":{"root":null,"reply":null}}"#,
                Kind::Close {
                    root: None,
                    reply: None,
                },
                true,
            ),
        ];
        for (kind, want, round_trips) in cases {
            let wire = format!("{head}{kind}}}");
            let job = Job {
                kind: want,
                ..review(7, "alice", false)
            };
            assert_eq!(serde_json::from_str::<Job>(&wire).unwrap(), job, "{wire}");
            if round_trips {
                assert_eq!(serde_json::to_string(&job).unwrap(), wire);
            }
        }
        // A command's comment time follows its kind; a payload without one has none.
        let wire = format!(
            r#"{head}"resolve","commented_at":{{"secs_since_epoch":1795167000,"nanos_since_epoch":0}}}}"#
        );
        let job = Job {
            kind: Kind::Resolve,
            commented_at: commented(),
            ..review(7, "alice", false)
        };
        assert_eq!(serde_json::from_str::<Job>(&wire).unwrap(), job, "{wire}");
        assert_eq!(serde_json::to_string(&job).unwrap(), wire);
        // A push's sender is marked; a payload without the mark pushed nothing.
        let wire = format!(r#"{head}{{"review":{{"command":false}}}},"pushed":true}}"#);
        let job = Job {
            pushed: true,
            ..review(7, "alice", false)
        };
        assert_eq!(serde_json::from_str::<Job>(&wire).unwrap(), job, "{wire}");
        assert_eq!(serde_json::to_string(&job).unwrap(), wire);
    }
}
