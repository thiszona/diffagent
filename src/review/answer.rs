//! The two calls a reply in a Thread can make after C6: the answer (C5) and the Lesson's
//! rule (C7), with what decides and feeds C5: [`addressed`] and the file [`window`].
//! The model only writes text. Whether a concession closes a Finding, and whether a rule
//! is stored, the worker decides. `diffagent-eval thread` calls the same functions, so it
//! measures what the worker sends.

use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Deserializer};
use tracing::warn;

use super::classify::Asked;
use super::prompt::{
    ANSWER_MAX_TOKENS, ANSWER_SYSTEM, DISTILL_MAX_TOKENS, DISTILL_SYSTEM, answer_schema,
    answer_user, distill_schema, distill_user,
};
use crate::diff::normalize_code;
use crate::finding::Category;
use crate::job::mention;
use crate::model::{Meter, ModelError, Request, Router};
use crate::state::ModelRole;

/// The longest rule stored, in chars, its ellipsis included: rules go into every
/// later prompt under their scope, so they stay short.
pub const MAX_RULE_CHARS: usize = 280;

/// One comment of a Thread, as C5 sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Turn {
    /// The author's login.
    pub author: String,
    /// Posted by the bot, judged by its author alone: a marker in a body proves nothing.
    pub ours: bool,
    pub body: String,
}

/// Where the Finding is and the code an answer is grounded in.
#[derive(Debug, Clone, Copy)]
pub struct Code<'a> {
    pub start_line: i64,
    pub end_line: i64,
    /// The commit the Finding was last seen at, which the window is read at.
    pub sha: &'a str,
    /// The Thread root's `diff_hunk`.
    pub hunk: &'a str,
    /// The file around the Finding, as (its first line's number, the lines); `None`
    /// when it could not be read.
    pub window: Option<(i64, &'a str)>,
}

/// C5's reply. `concedes` is the model's claim: it closes nothing on its own.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Answer {
    /// Trimmed and never blank.
    #[serde(deserialize_with = "non_blank")]
    pub answer: String,
    pub concedes: bool,
}

/// A blank answer fails to decode, so the router retries it once and then gives
/// `Invalid`: nothing is posted empty.
fn non_blank<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    let s = String::deserialize(d)?;
    match s.trim() {
        "" => Err(serde::de::Error::custom("the answer is empty")),
        t => Ok(t.to_owned()),
    }
}

/// C5, on the review role: answers `reply` about the Finding in `asked`, grounded in
/// `code`, after the Thread's earlier `thread` turns, oldest first. Any error is the
/// caller's `TryAgain`.
pub async fn answer(
    router: &Router,
    meter: &Meter,
    asked: &Asked<'_>,
    code: &Code<'_>,
    thread: &[Turn],
    reply: &str,
) -> Result<Answer, ModelError> {
    let schema = answer_schema();
    let user = answer_user(asked, code, thread, reply);
    let req = Request {
        system: ANSWER_SYSTEM,
        user: &user,
        shared: 0,
        schema: Some(&schema),
        max_tokens: ANSWER_MAX_TOKENS,
    };
    router.json(ModelRole::Review, &req, meter).await
}

/// C7's reply.
#[derive(Deserialize)]
struct Distilled {
    rule: String,
}

/// C7, on the summary role: the Lesson's rule from the gated `reason`, capped by
/// [`cap_rule`]. It never fails: a blank reply or any error gives [`fallback_rule`],
/// so the suppress never depends on this call.
pub async fn distill(
    router: &Router,
    meter: &Meter,
    asked: &Asked<'_>,
    category: Category,
    scope: &str,
    reason: &str,
) -> String {
    let schema = distill_schema();
    let user = distill_user(asked, category, scope, reason);
    let req = Request {
        system: DISTILL_SYSTEM,
        user: &user,
        shared: 0,
        schema: Some(&schema),
        max_tokens: DISTILL_MAX_TOKENS,
    };
    match router
        .json::<Distilled>(ModelRole::Summary, &req, meter)
        .await
    {
        Ok(d) if !d.rule.trim().is_empty() => cap_rule(&d.rule),
        Ok(_) => fallback_rule(asked.title, reason),
        Err(e) => {
            warn!(path = %asked.path, err = %e, "distill failed; the Lesson gets the fallback rule");
            fallback_rule(asked.title, reason)
        }
    }
}

/// Our [`mention`] as a whole token, any case.
static MENTIONED: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(&format!(r"(?i){}\b", regex::escape(mention()))).ok());

/// The Lesson rule without a model call: `Do not raise "<title>"-type findings like
/// this one: <reason>.`, the mention stripped, capped by [`cap_rule`].
pub fn fallback_rule(title: &str, reason: &str) -> String {
    let reason = match MENTIONED.as_ref() {
        Some(re) => re.replace_all(reason, ""),
        None => reason.into(),
    };
    let reason = normalize_code(&reason);
    let mut rule = format!(
        "Do not raise \"{}\"-type findings like this one",
        normalize_code(title)
    );
    if !reason.is_empty() {
        rule.push_str(": ");
        rule.push_str(&reason);
    }
    if !rule.ends_with(['.', '!', '?']) {
        rule.push('.');
    }
    cap_rule(&rule)
}

/// `rule` on one line, at most [`MAX_RULE_CHARS`] chars: a longer rule is cut on a
/// char, never inside one, and ends with `…`.
pub fn cap_rule(rule: &str) -> String {
    let rule = normalize_code(rule);
    if rule.chars().count() <= MAX_RULE_CHARS {
        return rule;
    }
    let mut cut: String = rule.chars().take(MAX_RULE_CHARS - 1).collect();
    cut.push('…');
    cut
}

/// Lines of the file shown on each side of the Finding.
const WINDOW_LINES: i64 = 60;

/// The file window's budget, in bytes.
pub const WINDOW_BYTES: usize = 24_000;

/// Lines `start - 60` to `end + 60` of `content` (1-based, clamped to the file), as
/// (its first line's number, the lines), cut to `cap` bytes: whole lines are dropped
/// from the side farther from the Finding first, and only a single line still over
/// `cap` is cut, on a char boundary. `None` for an empty file.
pub fn window(content: &str, start: i64, end: i64, cap: usize) -> Option<(i64, String)> {
    let lines: Vec<&str> = content.lines().collect();
    let n = i64::try_from(lines.len()).ok().filter(|n| *n > 0)?;
    let start = start.clamp(1, n);
    let end = end.clamp(start, n);
    let (mut lo, mut hi) = ((start - WINDOW_LINES).max(1), (end + WINDOW_LINES).min(n));
    let len = |i: i64| lines[usize::try_from(i - 1).unwrap_or(0)].len();
    // Each line but the last also costs its line break.
    let mut size: usize = (lo..=hi).map(|i| len(i) + 1).sum::<usize>() - 1;
    while size > cap && lo < hi {
        if start - lo >= hi - end {
            size -= len(lo) + 1;
            lo += 1;
        } else {
            size -= len(hi) + 1;
            hi -= 1;
        }
    }
    let from = usize::try_from(lo - 1).unwrap_or(0);
    let to = usize::try_from(hi).unwrap_or(0);
    let mut text = lines[from..to].join("\n");
    text.truncate(text.floor_char_boundary(cap));
    Some((lo, text))
}

/// A first word that makes a reply a question without a `?`.
const INTERROGATIVES: [&str; 26] = [
    "apa",
    "apakah",
    "kenapa",
    "mengapa",
    "kok",
    "gimana",
    "bagaimana",
    "emang",
    "kapan",
    "siapa",
    "mana",
    "dimana",
    "berapa",
    "jelasin",
    "jelaskan",
    "why",
    "what",
    "how",
    "when",
    "which",
    "who",
    "where",
    "is",
    "are",
    "can",
    "does",
];

/// The same for the first two words.
const PHRASES: [&str; 5] = [
    "di mana",
    "tolong jelas",
    "minta tolong",
    "bantu jelas",
    "coba jelas",
];

/// An `@login` mention: not inside a word, so an email address is none.
static MENTIONS: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"(?:^|[^\w@])@([A-Za-z0-9-]+)").ok());

/// Whether a reply is meant for us, so C5 answers it: it mentions us, or it is shaped
/// as a question and mentions no one else.
pub fn addressed(reply: &str) -> bool {
    let Some(re) = MENTIONS.as_ref() else {
        return false;
    };
    let ours = mention().trim_start_matches('@');
    let (mut us, mut others) = (false, false);
    for login in re.captures_iter(reply).filter_map(|c| c.get(1)) {
        if login.as_str().eq_ignore_ascii_case(ours) {
            us = true;
        } else {
            others = true;
        }
    }
    us || (!others && question_shape(reply))
}

/// With our mention removed, the reply holds a `?`, or opens
/// with an interrogative word or two.
fn question_shape(reply: &str) -> bool {
    let text = match MENTIONED.as_ref() {
        Some(re) => re.replace_all(reply, ""),
        None => reply.into(),
    };
    if text.contains('?') {
        return true;
    }
    let words: Vec<String> = text
        .split_whitespace()
        .take(2)
        .map(str::to_lowercase)
        .collect();
    let Some(first) = words.first() else {
        return false;
    };
    let first = first.trim_matches(['.', ',', '!', ':', ';', '"', '\'']);
    INTERROGATIVES.contains(&first)
        || words
            .get(1)
            .is_some_and(|second| PHRASES.contains(&format!("{first} {second}").as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finding::Severity;
    use crate::model::Stop;
    use crate::model::tests::Script;
    use crate::review::tests::router;

    const ASKED: Asked<'static> = Asked {
        severity: Severity::Major,
        path: "pkg/cart/total.go",
        title: "Dropped error",
        body: "When ParsePrice fails, the total is zero.",
        snippet: "",
    };

    const CODE: Code<'static> = Code {
        start_line: 12,
        end_line: 12,
        sha: "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
        hunk: "@@ -10,2 +10,3 @@\n a\n+b, _ := ParsePrice(x)\n c",
        window: Some((10, "a\nb, _ := ParsePrice(x)\nc")),
    };

    #[test]
    fn mention_patterns_compile() {
        assert!(MENTIONED.is_some() && MENTIONS.is_some());
    }

    #[test]
    fn addressed_replies() {
        let cases = [
            ("@diffagent kenapa seperti ini?", true),
            ("@Diffagent this is intended", true),
            ("why is this a bug", true),
            ("Kenapa begini", true),
            ("di mana key-nya dicek", true),
            ("tolong jelasin", false),
            ("Tolong jelas dong", true),
            ("is this right", true),
            ("this is fine?", true),
            ("this is fine", false),
            ("ok, noted", false),
            ("", false),
            ("@carol why did you change this?", false),
            ("@carol @diffagent why?", true),
            ("mail me at dev@example.com, why?", true),
            ("@diffagenter why?", false),
            ("\"Why\" not", true),
        ];
        for (reply, want) in cases {
            assert_eq!(addressed(reply), want, "{reply:?}");
        }
    }

    #[test]
    fn windows() {
        let file: String = (1..=200).map(|n| format!("line {n}\n")).collect();
        let lines = |a: i64, b: i64| {
            (a..=b)
                .map(|n| format!("line {n}"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let wide = "€".repeat(10_000);
        // (name, content, start, end, cap, want).
        type Case<'a> = (&'a str, &'a str, i64, i64, usize, Option<(i64, String)>);
        let cases: Vec<Case> = vec![
            (
                "±60 lines",
                &file,
                100,
                101,
                WINDOW_BYTES,
                Some((40, lines(40, 161))),
            ),
            (
                "clamped at the top",
                &file,
                2,
                2,
                WINDOW_BYTES,
                Some((1, lines(1, 62))),
            ),
            (
                "clamped at the end",
                &file,
                199,
                250,
                WINDOW_BYTES,
                Some((139, lines(139, 200))),
            ),
            (
                "a start past the end",
                &file,
                500,
                500,
                WINDOW_BYTES,
                Some((140, lines(140, 200))),
            ),
            (
                "cut to whole lines",
                &file,
                100,
                100,
                40,
                Some((99, lines(99, 102))),
            ),
            (
                "one line over the cap",
                "aé€😀b\n",
                1,
                1,
                8,
                Some((1, "aé€".to_owned())),
            ),
            (
                "one wide line over the budget",
                &wide,
                1,
                1,
                WINDOW_BYTES,
                Some((1, "€".repeat(WINDOW_BYTES / 3))),
            ),
            (
                "crlf",
                "a\r\nb\r\n",
                1,
                1,
                100,
                Some((1, "a\nb".to_owned())),
            ),
            ("empty", "", 1, 1, 100, None),
        ];
        for (name, content, start, end, cap, want) in cases {
            let got = window(content, start, end, cap);
            assert_eq!(got, want, "{name}");
            if let Some((_, text)) = got {
                assert!(text.len() <= cap, "{name}");
            }
        }
    }

    #[test]
    fn cap_rule_table() {
        let at = "x".repeat(MAX_RULE_CHARS);
        let over = "x".repeat(MAX_RULE_CHARS + 1);
        let wide = "€".repeat(MAX_RULE_CHARS + 5);
        let cases = [
            ("short", "Keep it.".to_owned(), "Keep it.".to_owned()),
            (
                "whitespace",
                "  a\n\tb  c \r\n".to_owned(),
                "a b c".to_owned(),
            ),
            ("at the cap", at.clone(), at),
            (
                "over the cap",
                over,
                format!("{}…", "x".repeat(MAX_RULE_CHARS - 1)),
            ),
            (
                "multibyte",
                wide,
                format!("{}…", "€".repeat(MAX_RULE_CHARS - 1)),
            ),
            ("empty", String::new(), String::new()),
        ];
        for (name, input, want) in cases {
            let got = cap_rule(&input);
            assert_eq!(got, want, "{name}");
            assert!(got.chars().count() <= MAX_RULE_CHARS, "{name}");
        }
    }

    #[test]
    fn fallback_rule_table() {
        let long = "sengaja ".repeat(60);
        let cases = [
            (
                "the plain shape",
                "Dropped error",
                "the error is logged upstream",
                "Do not raise \"Dropped error\"-type findings like this one: the error is \
                 logged upstream.",
            ),
            (
                "no double stop",
                "Dropped error",
                "It is logged upstream.",
                "Do not raise \"Dropped error\"-type findings like this one: It is logged \
                 upstream.",
            ),
            (
                "mention stripped, any case",
                " Dropped\nerror ",
                "@Diffagent  ini\nsengaja @diffagent",
                "Do not raise \"Dropped error\"-type findings like this one: ini sengaja.",
            ),
            (
                "no reason",
                "Dropped error",
                " \n",
                "Do not raise \"Dropped error\"-type findings like this one.",
            ),
        ];
        for (name, title, reason, want) in cases {
            assert_eq!(fallback_rule(title, reason), want, "{name}");
        }
        let got = fallback_rule("Dropped error", &long);
        assert_eq!(got.chars().count(), MAX_RULE_CHARS);
        assert!(got.ends_with('…'));
    }

    #[tokio::test]
    async fn answer_decodes_and_retries_a_blank_one() {
        let ok = r#"{"answer": " It stands: `total.go:12` drops the error. ", "concedes": false}"#;
        let blank = r#"{"answer": " ", "concedes": true}"#;
        let conceded = r#"{"answer": "You are right.", "concedes": true}"#;
        let want = |answer: &str, concedes| {
            Ok(Answer {
                answer: answer.to_owned(),
                concedes,
            })
        };
        let cases = [
            (
                vec![(ok, Stop::End)],
                want("It stands: `total.go:12` drops the error.", false),
                1,
            ),
            (
                vec![(blank, Stop::End), (conceded, Stop::End)],
                want("You are right.", true),
                2,
            ),
            (
                vec![(blank, Stop::End), (blank, Stop::End)],
                Err("invalid"),
                2,
            ),
            (vec![(ok, Stop::MaxTokens)], Err("truncated"), 1),
            (vec![], Err("request"), 1),
        ];
        for (script, want, calls) in cases {
            let model = Script::new(&script);
            let router = router(ModelRole::Review, model.clone());
            let turns = [Turn {
                author: "alice".to_owned(),
                ours: false,
                body: "earlier".to_owned(),
            }];
            let got = answer(&router, &Meter::default(), &ASKED, &CODE, &turns, "kenapa?")
                .await
                .map_err(|e| match e {
                    ModelError::Invalid(_) => "invalid",
                    ModelError::Truncated => "truncated",
                    ModelError::Request(_) => "request",
                    other => panic!("{other:?}"),
                });
            assert_eq!(got, want, "{script:?}");
            assert_eq!(model.calls(), calls, "{script:?}");
            let users = model.users.lock().unwrap();
            assert!(users[0].ends_with("grounded in the code shown."));
            if calls == 2 {
                assert!(users[1].contains("the answer is empty"), "{}", users[1]);
            }
        }
    }

    /// C5 runs on the review role only: with no review route it is `NoRoute`.
    #[tokio::test]
    async fn answer_is_a_review_call() {
        let router = router(ModelRole::Summary, Script::new(&[]));
        let got = answer(&router, &Meter::default(), &ASKED, &CODE, &[], "x").await;
        assert_eq!(got, Err(ModelError::NoRoute(ModelRole::Review)));
        assert_eq!(ANSWER_MAX_TOKENS, 4_000);
    }

    #[tokio::test]
    async fn distill_decodes_or_falls_back() {
        let reason = "ParsePrice never fails on a stored price";
        let fallback = fallback_rule(ASKED.title, reason);
        let long = format!(r#"{{"rule": "{}"}}"#, "y".repeat(400));
        let cases = [
            (
                ModelRole::Summary,
                vec![(
                    r#"{"rule": " The error from ParsePrice is ignored because it never fails on a stored price (error-handling). "}"#,
                    Stop::End,
                )],
                "The error from ParsePrice is ignored because it never fails on a stored price \
                 (error-handling).",
            ),
            (
                ModelRole::Summary,
                vec![(r#"{"rule": " \n"}"#, Stop::End)],
                fallback.as_str(),
            ),
            (
                ModelRole::Summary,
                vec![(r#"{"rule": "x"#, Stop::MaxTokens)],
                fallback.as_str(),
            ),
            (ModelRole::Summary, vec![], fallback.as_str()),
            (
                ModelRole::Review,
                vec![(r#"{"rule": "x"}"#, Stop::End)],
                fallback.as_str(),
            ),
        ];
        for (role, script, want) in cases {
            let router = router(role, Script::new(&script));
            let got = distill(
                &router,
                &Meter::default(),
                &ASKED,
                Category::ErrorHandling,
                "pkg/cart",
                reason,
            )
            .await;
            assert_eq!(got, want, "{role:?} {script:?}");
        }
        let router = router(ModelRole::Summary, Script::new(&[(&long, Stop::End)]));
        let got = distill(
            &router,
            &Meter::default(),
            &ASKED,
            Category::ErrorHandling,
            "",
            reason,
        )
        .await;
        assert_eq!(got.chars().count(), MAX_RULE_CHARS);
        assert_eq!(DISTILL_MAX_TOKENS, 1_000);
    }
}
