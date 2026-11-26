//! The review pipeline's model calls: which files get reviewed, the per-file review
//! (C1) fanned out with bounded concurrency, the cap on low-severity findings, and
//! the walkthrough with its diagram (C2+C3), the second look at a critical Finding
//! (C4), and the Ticket fit (C8). Everything after the reply (identity,
//! plan, render, scan) is reused from earlier phases. Only [`ModelRole`] is shared
//! with the state layer: no store types and no worker code.

pub mod answer;
pub mod classify;
pub mod prompt;

use std::cmp::Reverse;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use regex::Regex;
use serde::Deserialize;
use tokio::sync::Semaphore;
use tokio::task::{Id, JoinError, JoinSet};
use tokio::time::{Instant, timeout_at};
use tracing::warn;

use crate::context::ContextPack;
use crate::diff::{FileChange, FileStatus, normalize_code};
use crate::finding::{Category, Finding, Severity};
use crate::jira::Ticket;
use crate::lang::Lang;
use crate::model::{Meter, ModelError, Request, Router};
use crate::render::{BulletGroup, CohortRow, Fit, Requirement, TicketFit, Walkthrough};
use crate::state::ModelRole;
use prompt::{
    REVIEW_MAX_TOKENS, SUMMARY_MAX_TOKENS, SUMMARY_SYSTEM, TICKET_SYSTEM, VERIFY_MAX_TOKENS,
    VERIFY_SYSTEM, findings_schema, line_counts, review_system, review_user, summary_schema,
    summary_user, ticket_schema, verify_schema, verify_user,
};

/// What the PR tells every file's review. The worker fills it: lessons ranked by
/// `learn::rank`, intent from the title, body and Jira, conventions at the base SHA.
#[derive(Debug, Clone, Copy, Default)]
pub struct PrContext<'a> {
    pub conventions: &'a str,
    pub intent: &'a str,
    /// (scope, rule); an empty scope is repo-wide.
    pub lessons: &'a [(String, String)],
}

/// The per-file reviews of one Review.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FileReview {
    pub findings: Vec<Finding>,
    /// Paths whose call failed: a Review with failures is never an all-clear.
    pub failed: Vec<String>,
}

/// Deny suffixes: generated code, lockfiles, minified assets.
const DENY_SUFFIXES: [&str; 18] = [
    ".pb.go",
    "_gen.go",
    ".gen.go",
    "_templ.go",
    ".min.js",
    ".min.css",
    "go.sum",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "composer.lock",
    ".pb.dart",
    ".g.dart",
    ".freezed.dart",
    "_pb2.py",
    "_pb2_grpc.py",
    ".designer.cs",
    ".g.cs",
];

/// Generated and vendored tree markers, matched in `/path/`: whole segments, so
/// `VendorList.tsx` is kept, and `.pb.` inside a name for protobuf output that
/// [`DENY_SUFFIXES`] misses (`a.pb.gw.go`, `a.pb.ts`).
const DENY_DIR_SEGMENTS: [&str; 9] = [
    "/vendor/",
    "/node_modules/",
    "/.venv/",
    "/venv/",
    "/site-packages/",
    "/ent/",
    "/generated/",
    "/__generated__/",
    ".pb.",
];

/// Whether a changed file is worth a model call: not removed, a non-blank patch, a
/// language we parse, not a test file, not generated or vendored.
pub fn reviewable(f: &FileChange) -> bool {
    // ponytail: no enry; Lang stands in for its vendored/config/binary lists.
    let lower = f.path.to_lowercase();
    let segments = format!("/{lower}/");
    // Go's marker is `Code generated … DO NOT EDIT.`: a bare "DO NOT EDIT" is prose.
    let generated = f.patch.lines().take(5).any(|l| {
        l.contains("@generated")
            || l.split_once("Code generated")
                .is_some_and(|(_, rest)| rest.contains("DO NOT EDIT"))
    });
    f.status != FileStatus::Removed
        && !f.patch.trim().is_empty()
        && Lang::for_path(&f.path).is_some()
        && !is_test_file(&f.path)
        && !DENY_SUFFIXES.iter().any(|s| lower.ends_with(s))
        && !DENY_DIR_SEGMENTS.iter().any(|d| segments.contains(d))
        && !generated
}

/// Whether a path is a test file: unit, spec and story files draw a flood of
/// hallucinated nitpicks for little review value. JVM, .NET, Swift and PHPUnit tests
/// are named by their class (`FooTest.java`), so that suffix is matched with its
/// case: `Latest.java` is not a test.
fn is_test_file(path: &str) -> bool {
    let lower = path.to_lowercase();
    let rooted = format!("/{lower}");
    let base = lower.rsplit('/').next().unwrap_or(&lower);
    let file = path.rsplit('/').next().unwrap_or(path);
    let class = file.rsplit_once('.').map_or(file, |(stem, _)| stem);
    let by_class = matches!(
        Lang::for_path(path),
        Some(Lang::Java | Lang::Kotlin | Lang::CSharp | Lang::Swift | Lang::Php | Lang::Scala)
    );
    rooted.contains("/__tests__/")
        || rooted.contains("/__mocks__/")
        || [
            "_test.go",
            "_test.py",
            "_test.exs",
            "_spec.rb",
            "_test.rb",
            "_test.cc",
            "_test.cpp",
            "_unittest.cc",
        ]
        .iter()
        .any(|s| base.ends_with(s))
        || (base.starts_with("test_") && base.ends_with(".py"))
        || base == "conftest.py"
        || (by_class
            && ["Test", "Tests", "Spec", "Suite"]
                .iter()
                .any(|s| class.ends_with(s)))
        || [".test.", ".spec.", ".stories."]
            .iter()
            .any(|s| base.contains(s))
}

/// C1's reply as the schema shapes it. `Severity` and `Category` decode directly:
/// the schema grammar enforces the enums, and a bad value from another wire is
/// `Invalid` and retried rather than defaulted, since a default would hide a wire bug.
#[derive(Deserialize)]
struct Raw {
    findings: Vec<RawFinding>,
}

#[derive(Deserialize)]
struct RawFinding {
    body: String,
    category: Category,
    end_line: i64,
    severity: Severity,
    snippet: String,
    start_line: i64,
    suggestion: Option<String>,
    title: String,
}

impl RawFinding {
    /// The finding on `path`, or `None` without a title or body.
    fn into_finding(self, path: &str) -> Option<Finding> {
        if self.title.trim().is_empty() || self.body.trim().is_empty() {
            return None;
        }
        Some(Finding {
            path: path.to_owned(),
            start_line: self.start_line,
            end_line: self.end_line.max(self.start_line),
            severity: self.severity,
            category: self.category,
            title: self.title,
            body: self.body,
            snippet: self.snippet,
            suggestion: self.suggestion.unwrap_or_default(),
        })
    }
}

/// The longest the other files wait for the first one. Its prompt cache entry is
/// readable once its reply starts, seconds after it is sent, long before the reply
/// ends; a slow first file must not hold the rest.
const WARM_WAIT: Duration = if cfg!(test) {
    Duration::from_secs(2)
} else {
    Duration::from_secs(5)
};

/// Reviews each file in one review-role call, at most `concurrency` in flight. The
/// smallest file of the largest group sharing a system prompt goes first and alone,
/// so the rest of its group read the prompt cache it writes. Findings come back in
/// input order. Any error fails that file only; a file without a pack gets a
/// diff-only review. A file still waiting for a slot or running at `deadline` is
/// cancelled and failed, so a slow Review still posts what finished inside its job's
/// timeout.
pub async fn review_files(
    router: &Router,
    meter: &Arc<Meter>,
    files: &[FileChange],
    packs: &BTreeMap<String, ContextPack>,
    pr: &PrContext<'_>,
    concurrency: usize,
    deadline: Instant,
) -> FileReview {
    let schema = Arc::new(findings_schema());
    let gate = Arc::new(Semaphore::new(concurrency.max(1)));
    let empty = ContextPack::default();
    let mut tasks = JoinSet::new();
    // A panicked task returns no index, so its task id names its file.
    let mut index = HashMap::new();
    let mut results: Vec<Option<Result<Vec<Finding>, String>>> = vec![None; files.len()];
    // A prompt cache entry is readable only once its reply has started, so files sent
    // together would each write the shared prefix; and it covers the system prompt, so
    // only files with the same one read it.
    // ponytail: only the largest group is warmed; another writes its prefix once per
    // file in its first wave.
    let systems: Vec<String> = files
        .iter()
        .map(|f| Lang::for_path(&f.path).map_or_else(|| prompt::CORE.to_owned(), review_system))
        .collect();
    let mut group: HashMap<&str, usize> = HashMap::new();
    for s in &systems {
        *group.entry(s).or_default() += 1;
    }
    let first = (0..files.len())
        .min_by_key(|&i| (Reverse(group[systems[i].as_str()]), files[i].patch.len()));
    let rest = (0..files.len()).filter(|i| Some(*i) != first);
    for (n, i) in first.into_iter().chain(rest).enumerate() {
        if n == 1 {
            // At most WARM_WAIT, and never past the deadline: then the rest, in order.
            let until = deadline.min(Instant::now() + WARM_WAIT);
            if let Ok(Some(joined)) = timeout_at(until, tasks.join_next()).await {
                record(&mut results, &index, joined);
            }
        }
        let f = &files[i];
        let system = systems[i].clone();
        let (user, shared) = review_user(f, packs.get(&f.path).unwrap_or(&empty), pr);
        // A head no other file with this system prompt sends is never read back.
        let shared = if group[system.as_str()] > 1 {
            shared
        } else {
            0
        };
        let (router, meter, schema, gate) =
            (router.clone(), meter.clone(), schema.clone(), gate.clone());
        let path = f.path.clone();
        let task = tasks.spawn(async move {
            // The semaphore is never closed, so the permit is always granted.
            let _permit = gate.acquire_owned().await.ok();
            let req = Request {
                system: &system,
                user: &user,
                shared,
                schema: Some(&schema),
                max_tokens: REVIEW_MAX_TOKENS,
            };
            let raw = router.json::<Raw>(ModelRole::Review, &req, &meter).await;
            let found = raw.map(|r| {
                let found = r.findings.into_iter();
                found
                    .filter_map(|f| f.into_finding(&path))
                    .collect::<Vec<_>>()
            });
            (i, found)
        });
        index.insert(task.id(), i);
    }
    let joined = async {
        while let Some(joined) = tasks.join_next().await {
            record(&mut results, &index, joined);
        }
    };
    // At the deadline the unfinished files keep `None`; the set aborts them on drop.
    let _ = timeout_at(deadline, joined).await;
    let mut review = FileReview::default();
    // Each failed file logs its reason: `failed_files` alone can't tell an IAM denial
    // from a throttle or a timeout.
    for (f, result) in files.iter().zip(results) {
        let path = f.path.as_str();
        match result {
            Some(Ok(found)) => {
                review.findings.extend(found);
                continue;
            }
            Some(Err(e)) => warn!(path, err = %e, "file review failed"),
            // Every finished or panicked task left a result: only the deadline leaves none.
            None => warn!(path, "file review cut at the review deadline"),
        }
        review.failed.push(f.path.clone());
    }
    review
}

/// One file's finished task into `results`.
fn record(
    results: &mut [Option<Result<Vec<Finding>, String>>],
    index: &HashMap<Id, usize>,
    joined: Result<(usize, Result<Vec<Finding>, ModelError>), JoinError>,
) {
    match joined {
        Ok((i, found)) => results[i] = Some(found.map_err(|e| e.to_string())),
        // A panic fails its file with its message: "task 7 panicked with message …".
        Err(e) => results[index[&e.id()]] = Some(Err(e.to_string())),
    }
}

/// Minor and nitpick findings open per PR.
const LOW_SEVERITY_SLOTS: usize = 3;

/// Bounds the noise of one Review's new Findings, after reconcile (which must see
/// every Finding, or a capped one would resolve and then be posted again): exact
/// duplicates on (path, category, lines, normalized snippet) go first; every
/// critical and major finding stays; at most three minor or nitpick findings stay,
/// less `open_minors` (the minor or nitpick Findings reconcile kept open), minor
/// first, then in path order. Returns the kept indices into `findings`, ascending.
/// The lines are in the key because a snippet can be empty or shared (`}`):
/// without them two findings that identity keeps apart would merge here, a major
/// one among them. Equal lines give identity an equal anchor and flagged code, so
/// it collapses whatever this does.
pub fn cap(findings: &[&Finding], open_minors: usize) -> Vec<usize> {
    let mut seen = HashSet::new();
    let unique: Vec<usize> = findings
        .iter()
        .enumerate()
        .filter(|(_, f)| {
            let code = normalize_code(&f.snippet);
            seen.insert((f.path.as_str(), f.category, f.start_line, f.end_line, code))
        })
        .map(|(i, _)| i)
        .collect();
    let mut low: Vec<usize> = unique
        .iter()
        .copied()
        .filter(|&i| findings[i].severity <= Severity::Minor)
        .collect();
    low.sort_by(|&a, &b| {
        let (a, b) = (findings[a], findings[b]);
        b.severity
            .cmp(&a.severity)
            .then_with(|| a.path.cmp(&b.path))
    });
    let slots = LOW_SEVERITY_SLOTS.saturating_sub(open_minors);
    let dropped: HashSet<usize> = low.into_iter().skip(slots).collect();
    unique
        .into_iter()
        .filter(|i| !dropped.contains(i))
        .collect()
}

/// C4's reply: whether the second reviewer holds a Finding critical, and why.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Verdict {
    pub analysis: String,
    pub critical: bool,
}

/// One verify-role call on a critical Finding of `file`, which sees what C1 saw of
/// the file (`pack`) but no team notes: a note never lowers a severity. The worker
/// decides what a `false` changes.
pub async fn verify(
    router: &Router,
    meter: &Meter,
    finding: &Finding,
    file: &FileChange,
    pack: &ContextPack,
) -> Result<Verdict, ModelError> {
    let schema = verify_schema();
    let user = verify_user(finding, file, pack);
    let req = Request {
        system: VERIFY_SYSTEM,
        user: &user,
        shared: 0,
        schema: Some(&schema),
        max_tokens: VERIFY_MAX_TOKENS,
    };
    router.json(ModelRole::Verify, &req, meter).await
}

/// C2+C3's reply as the schema shapes it.
#[derive(Deserialize)]
struct RawWalkthrough {
    bullets: Vec<RawBullet>,
    changes: Vec<RawChange>,
    diagram: Option<String>,
    effort_minutes: i64,
    effort_score: i64,
    summary: String,
}

#[derive(Deserialize)]
struct RawBullet {
    heading: String,
    items: Vec<String>,
}

#[derive(Deserialize)]
struct RawChange {
    files: Vec<String>,
    summary: String,
}

/// The walkthrough and its diagram in one summary-role call. It never sees lessons. An
/// empty `changes` is `Invalid`; the worker falls back to [`basic_walkthrough`].
pub async fn walkthrough(
    router: &Router,
    meter: &Meter,
    files: &[FileChange],
    intent: &str,
) -> Result<Walkthrough, ModelError> {
    Ok(walkthrough_with_dropped(router, meter, files, intent)
        .await?
        .0)
}

/// [`walkthrough`], plus the model's diagram when it was not null or blank but would
/// not render and so was dropped: prod renders without it, the eval fails on it.
pub async fn walkthrough_with_dropped(
    router: &Router,
    meter: &Meter,
    files: &[FileChange],
    intent: &str,
) -> Result<(Walkthrough, Option<String>), ModelError> {
    let schema = summary_schema();
    let user = summary_user(files, intent);
    let req = Request {
        system: SUMMARY_SYSTEM,
        user: &user,
        shared: 0,
        schema: Some(&schema),
        max_tokens: SUMMARY_MAX_TOKENS,
    };
    let raw: RawWalkthrough = router.json(ModelRole::Summary, &req, meter).await?;
    let changes: Vec<CohortRow> = raw
        .changes
        .into_iter()
        .filter(|c| !c.files.is_empty())
        .map(|c| CohortRow {
            cohort: top_dir(&c.files[0]).to_owned(),
            files: c.files,
            summary: c.summary.trim().to_owned(),
        })
        .collect();
    if changes.is_empty() {
        return Err(ModelError::Invalid(
            "the walkthrough has no changes".to_owned(),
        ));
    }
    let bullets = raw
        .bullets
        .into_iter()
        .filter(|b| !b.heading.trim().is_empty() && !b.items.is_empty())
        .map(|b| BulletGroup {
            heading: b.heading.trim().to_owned(),
            items: b.items,
        })
        .collect();
    let proposed = raw.diagram.unwrap_or_default();
    let kept = diagram(&proposed);
    let dropped = (kept.is_empty() && !proposed.trim().is_empty()).then_some(proposed);
    let w = Walkthrough {
        bullets,
        summary: raw.summary.trim().to_owned(),
        changes,
        effort_score: u32::try_from(raw.effort_score.clamp(1, 5)).unwrap_or(1),
        effort_minutes: u32::try_from(raw.effort_minutes.max(0)).unwrap_or(u32::MAX),
        diagram: kept,
        ..stats(files)
    };
    Ok((w, dropped))
}

/// C8's reply as the schema shapes it.
#[derive(Deserialize)]
struct RawFit {
    context: String,
    requirements: Vec<RawRequirement>,
}

#[derive(Deserialize)]
struct RawRequirement {
    note: String,
    requirement: String,
    source: Source,
    status: Fit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Source {
    Issue,
    Story,
}

/// Requirements a Ticket fit shows from the issue, and from its story.
const MAX_REQUIREMENTS: usize = 10;

/// The PR's files checked against its Jira issue in one summary-role call, over the
/// walkthrough's layout with the issue as the intent: the first ten requirements
/// with text from the issue, and from its story when the context holds one. It never
/// sees lessons, and it decides nothing.
pub async fn ticket_fit(
    router: &Router,
    meter: &Meter,
    files: &[FileChange],
    ticket: &Ticket,
) -> Result<TicketFit, ModelError> {
    let schema = ticket_schema();
    let user = summary_user(files, &ticket.text);
    let req = Request {
        system: TICKET_SYSTEM,
        user: &user,
        shared: 0,
        schema: Some(&schema),
        max_tokens: SUMMARY_MAX_TOKENS,
    };
    let raw: RawFit = router.json(ModelRole::Summary, &req, meter).await?;
    let sources = [Source::Issue, Source::Story];
    let requirements = sources
        .into_iter()
        .filter(|s| *s == Source::Issue || ticket.story.is_some())
        .flat_map(|source| {
            raw.requirements
                .iter()
                .filter(move |r| r.source == source && !r.requirement.trim().is_empty())
                .take(MAX_REQUIREMENTS)
        })
        .map(|r| Requirement {
            text: r.requirement.trim().to_owned(),
            fit: r.status,
            note: r.note.trim().to_owned(),
            from_story: r.source == Source::Story,
        })
        .collect();
    Ok(TicketFit {
        key: ticket.key.clone(),
        story: ticket.story.clone(),
        context: raw.context.trim().to_owned(),
        requirements,
    })
}

/// The fallback text: "Review of N changed file(s)." and the files grouped by top
/// directory, with the same counts as [`walkthrough`].
pub fn basic_walkthrough(files: &[FileChange]) -> Walkthrough {
    let mut by_dir: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for f in files {
        by_dir
            .entry(top_dir(&f.path))
            .or_default()
            .push(f.path.clone());
    }
    let changes = by_dir
        .into_iter()
        .map(|(dir, mut paths)| {
            paths.sort();
            CohortRow {
                cohort: dir.to_owned(),
                files: paths,
                summary: String::new(),
            }
        })
        .collect();
    Walkthrough {
        summary: format!("Review of {} changed file(s).", files.len()),
        changes,
        ..stats(files)
    }
}

/// The reviewed-file counts, from the patches.
fn stats(files: &[FileChange]) -> Walkthrough {
    let (additions, deletions) = files.iter().fold((0, 0), |(a, d), f| {
        let (fa, fd) = line_counts(&f.patch);
        (a + fa, d + fd)
    });
    Walkthrough {
        files_changed: files.len(),
        additions,
        deletions,
        ..Walkthrough::default()
    }
}

/// The cohort label: the first path segment, or `(root)`.
fn top_dir(path: &str) -> &str {
    path.split_once('/').map_or("(root)", |(dir, _)| dir)
}

/// The diagram to render: the extracted Mermaid if it will render, else nothing.
fn diagram(text: &str) -> String {
    let d = extract_mermaid(text);
    if valid_sequence_diagram(&d) {
        d
    } else {
        String::new()
    }
}

/// The contents of a ```mermaid block, else everything from
/// the first standalone `sequenceDiagram` line up to a closing fence, else the
/// trimmed text for the validator to reject.
fn extract_mermaid(text: &str) -> String {
    let t = text.trim();
    if let Some(i) = t.find("```mermaid") {
        let rest = &t[i + "```mermaid".len()..];
        let rest = rest.find("```").map_or(rest, |j| &rest[..j]);
        return rest.trim().to_owned();
    }
    let lines: Vec<&str> = t.split('\n').collect();
    let Some(start) = lines.iter().position(|l| l.trim() == "sequenceDiagram") else {
        return t.to_owned();
    };
    let body = &lines[start..];
    let end = body
        .iter()
        .position(|l| l.trim() == "```")
        .unwrap_or(body.len());
    body[..end].join("\n").trim().to_owned()
}

/// The line shapes the validator accepts, compiled once. `None` only if a pattern
/// failed to compile, which a test rules out; every diagram is then dropped.
static MERMAID: LazyLock<Option<[Regex; 4]>> = LazyLock::new(|| {
    Some([
        // Caller<arrow>[+/-]Callee: text
        Regex::new(r"^[A-Za-z0-9_]+\s*(?:--?>>?|--?[x\)])\s*[+-]?[A-Za-z0-9_]+\s*:.+$").ok()?,
        Regex::new(r"^(?:participant|actor)\s+[A-Za-z0-9_]+(?:\s+as\s+.+)?$").ok()?,
        Regex::new(
            r"(?i)^note\s+(?:left of|right of|over)\s+[A-Za-z0-9_]+(?:\s*,\s*[A-Za-z0-9_]+)?\s*:.+$",
        )
        .ok()?,
        Regex::new(r"^(?:activate|deactivate)\s+[A-Za-z0-9_]+$").ok()?,
    ])
});

/// GitHub shows "Unable to render" for broken Mermaid,
/// so a diagram is kept only if it starts with `sequenceDiagram`, has a message,
/// balances every block with `end`, and has no line the validator does not know.
pub fn valid_sequence_diagram(d: &str) -> bool {
    let Some([msg, participant, note, activate]) = MERMAID.as_ref() else {
        return false;
    };
    let mut lines = d.split('\n');
    if lines.next().map(str::trim) != Some("sequenceDiagram") {
        return false;
    }
    let (mut depth, mut messages) = (0_i32, 0);
    for raw in lines {
        let line = raw.trim();
        if line.is_empty() || line.starts_with("%%") {
            continue;
        }
        let kw = line
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .to_lowercase();
        match kw.as_str() {
            "end" => {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
            }
            "loop" | "alt" | "opt" | "par" | "critical" | "break" | "rect" | "box" => depth += 1,
            "else" | "and" | "option" if depth == 0 => return false,
            "else" | "and" | "option" | "autonumber" | "title" => {}
            _ if activate.is_match(line) || participant.is_match(line) || note.is_match(line) => {}
            _ if msg.is_match(line) => messages += 1,
            _ => return false,
        }
    }
    depth == 0 && messages > 0
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use async_trait::async_trait;
    use serde_json::{Value, json};

    use super::prompt::tests::{PATCH, change};
    use super::*;
    use crate::config::Config;
    use crate::model::tests::{Script, reply, route};
    use crate::model::{Model, Reply, Stop};

    #[test]
    fn reviewable_rules() {
        let removed = FileChange {
            status: FileStatus::Removed,
            ..change("a.go", PATCH)
        };
        let cases = [
            ("kept", change("pkg/a.go", PATCH), true),
            ("ts", change("web/App.tsx", PATCH), true),
            ("removed", removed, false),
            ("blank patch", change("a.go", " \n"), false),
            ("no Lang", change("README.md", PATCH), false),
            ("python", change("app.py", PATCH), true),
            ("rust", change("src/lib.rs", PATCH), true),
            ("python test", change("tests/test_app.py", PATCH), false),
            ("conftest", change("conftest.py", PATCH), false),
            ("protobuf python", change("api/a_pb2.py", PATCH), false),
            ("venv", change(".venv/lib/a.py", PATCH), false),
            ("java test", change("src/FooTest.java", PATCH), false),
            ("kotlin spec", change("src/FooSpec.kt", PATCH), false),
            ("c# tests", change("src/FooTests.cs", PATCH), false),
            (
                "Latest is not a test",
                change("src/Latest.java", PATCH),
                true,
            ),
            (
                "lower-case test name kept",
                change("src/latest.php", PATCH),
                true,
            ),
            ("ruby test", change("test/a_test.rb", PATCH), false),
            ("c++ test", change("src/a_test.cc", PATCH), false),
            ("go test", change("pkg/a_test.go", PATCH), false),
            ("ts test", change("web/a.test.ts", PATCH), false),
            ("spec", change("web/a.spec.tsx", PATCH), false),
            ("stories", change("web/Button.stories.tsx", PATCH), false),
            ("__tests__", change("web/__tests__/a.ts", PATCH), false),
            ("__mocks__", change("__mocks__/a.js", PATCH), false),
            ("protobuf", change("api/a.pb.go", PATCH), false),
            ("gen suffix", change("db/models_gen.go", PATCH), false),
            ("templ", change("ui/page_templ.go", PATCH), false),
            ("minified", change("static/app.min.js", PATCH), false),
            ("upper-case suffix", change("API/A.PB.GO", PATCH), false),
            ("vendor", change("vendor/x/a.go", PATCH), false),
            ("vendor, leading", change("vendor/a.go", PATCH), false),
            ("nested vendor", change("pkg/vendor/a.go", PATCH), false),
            (
                "vendor in a name kept",
                change("src/components/VendorList.tsx", PATCH),
                true,
            ),
            ("vendors dir kept", change("src/vendors/a.ts", PATCH), true),
            ("node_modules", change("node_modules/x/a.js", PATCH), false),
            (
                "node_modules in a name kept",
                change("web/node_modules_doc.ts", PATCH),
                true,
            ),
            (
                "generated in a name kept",
                change("src/generatedTypes.ts", PATCH),
                true,
            ),
            ("protobuf gateway", change("api/a.pb.gw.go", PATCH), false),
            ("ts protobuf", change("web/gen/a.pb.ts", PATCH), false),
            ("pb in a name kept", change("src/pbkdf.ts", PATCH), true),
            ("ent", change("internal/ent/user.go", PATCH), false),
            (
                "ent-like dir kept",
                change("internal/entity/user.go", PATCH),
                true,
            ),
            (
                "generated dir",
                change("src/generated/api.ts", PATCH),
                false,
            ),
            (
                "__generated__",
                change("src/__generated__/q.ts", PATCH),
                false,
            ),
            (
                "generated header",
                change(
                    "a.go",
                    "@@ -0,0 +1,2 @@\n+// Code generated by x. DO NOT EDIT.\n+package a",
                ),
                false,
            ),
            (
                "@generated header",
                change("a.ts", "@@ -0,0 +1,2 @@\n+/* @generated */\n+export {}"),
                false,
            ),
            (
                "bare DO NOT EDIT kept",
                change(
                    "a.go",
                    "@@ -0,0 +1,2 @@\n+// DO NOT EDIT without a migration.\n+package a",
                ),
                true,
            ),
            (
                "Code generated without DO NOT EDIT kept",
                change(
                    "a.go",
                    "@@ -0,0 +1,2 @@\n+// Code generated once.\n+package a",
                ),
                true,
            ),
            (
                "header after line 5",
                change("a.go", "@@ -1,6 +1,6 @@\n a\n b\n c\n d\n+// DO NOT EDIT"),
                true,
            ),
        ];
        for (name, f, want) in cases {
            assert_eq!(reviewable(&f), want, "{name}");
        }
    }

    fn finding(path: &str, severity: Severity, category: Category, snippet: &str) -> Finding {
        Finding {
            path: path.to_owned(),
            start_line: 1,
            end_line: 1,
            severity,
            category,
            title: format!("{severity:?} {snippet}"),
            body: "b".to_owned(),
            snippet: snippet.to_owned(),
            suggestion: String::new(),
        }
    }

    fn at(f: Finding, start_line: i64, end_line: i64) -> Finding {
        Finding {
            start_line,
            end_line,
            ..f
        }
    }

    #[test]
    fn cap_table() {
        use Category::{Logic, Maintainability, Validation};
        use Severity::{Critical, Major, Minor, Nitpick};
        let f = finding;
        let cases = [
            ("empty", 0, vec![], vec![]),
            (
                "all serious kept",
                0,
                vec![
                    f("a", Major, Logic, "1"),
                    f("a", Critical, Logic, "2"),
                    f("b", Major, Logic, "3"),
                ],
                vec!["Major 1", "Critical 2", "Major 3"],
            ),
            (
                "three low slots, minor first then path, input order out",
                0,
                vec![
                    f("z", Nitpick, Maintainability, "n1"),
                    f("b", Minor, Logic, "m1"),
                    f("a", Nitpick, Maintainability, "n2"),
                    f("c", Major, Logic, "M"),
                    f("a", Minor, Logic, "m2"),
                    f("y", Minor, Logic, "m3"),
                ],
                vec!["Minor m1", "Major M", "Minor m2", "Minor m3"],
            ),
            (
                "nitpicks by path",
                0,
                vec![
                    f("d", Nitpick, Maintainability, "4"),
                    f("c", Nitpick, Maintainability, "3"),
                    f("b", Nitpick, Maintainability, "2"),
                    f("a", Nitpick, Maintainability, "1"),
                ],
                vec!["Nitpick 3", "Nitpick 2", "Nitpick 1"],
            ),
            (
                "duplicates collapse before the low slots are counted",
                0,
                vec![
                    f("a", Minor, Logic, "x  :=  1"),
                    f("a", Minor, Logic, "x := 1"),
                    f("a", Minor, Logic, "\tx := 1"),
                    f("b", Minor, Logic, "y"),
                    f("c", Minor, Logic, "z"),
                ],
                vec!["Minor x  :=  1", "Minor y", "Minor z"],
            ),
            (
                "duplicates differ by path or category",
                0,
                vec![
                    f("a", Major, Logic, "x"),
                    f("a", Major, Logic, "x"),
                    f("b", Major, Logic, "x"),
                    f("a", Major, Validation, "x"),
                ],
                vec!["Major x", "Major x", "Major x"],
            ),
            (
                "an empty or shared snippet on other lines is no duplicate",
                0,
                vec![
                    f("a", Major, Logic, ""),
                    at(f("a", Major, Logic, ""), 45, 45),
                    at(f("a", Major, Logic, "}"), 20, 20),
                    at(f("a", Major, Logic, "}"), 20, 22),
                    at(f("a", Major, Logic, " } "), 20, 20),
                ],
                vec!["Major ", "Major ", "Major }", "Major }"],
            ),
            (
                "open minors take low slots",
                1,
                vec![
                    f("a", Minor, Logic, "m1"),
                    f("b", Nitpick, Maintainability, "n1"),
                    f("c", Major, Logic, "M"),
                    f("d", Minor, Logic, "m2"),
                ],
                vec!["Minor m1", "Major M", "Minor m2"],
            ),
            (
                "a fourth minor is not posted",
                3,
                vec![f("a", Minor, Logic, "m1"), f("b", Critical, Logic, "C")],
                vec!["Critical C"],
            ),
            (
                "more open minors than slots",
                5,
                vec![
                    f("a", Nitpick, Maintainability, "n1"),
                    f("b", Major, Logic, "M"),
                ],
                vec!["Major M"],
            ),
        ];
        for (name, open_minors, input, want) in cases {
            let refs: Vec<&Finding> = input.iter().collect();
            let kept = cap(&refs, open_minors);
            assert!(kept.is_sorted(), "{name}: input order");
            let titles: Vec<&str> = kept.iter().map(|&i| input[i].title.as_str()).collect();
            assert_eq!(titles, want, "{name}");
        }
    }

    type Replies = Vec<(String, Stop)>;

    /// A model that answers by the reviewed path (from the user message), after
    /// that path's delay, and tracks how many calls are in flight and each one's
    /// (path, shared bytes). A reply of `panic` panics.
    #[derive(Default)]
    struct ByPath {
        replies: Mutex<HashMap<String, Replies>>,
        delays: HashMap<String, u64>,
        in_flight: AtomicUsize,
        peak: AtomicUsize,
        calls: AtomicUsize,
        sent: Mutex<Vec<(String, usize)>>,
    }

    #[async_trait]
    impl Model for ByPath {
        async fn complete(&self, req: &crate::model::Request<'_>) -> Result<Reply, ModelError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let path = req
                .user
                .split("<review_diff path=\"")
                .nth(1)
                .and_then(|r| r.split('"').next())
                .unwrap_or_default()
                .to_owned();
            self.sent.lock().unwrap().push((path.clone(), req.shared));
            let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            let delay = self.delays.get(&path).copied().unwrap_or(1);
            tokio::time::sleep(Duration::from_secs(delay)).await;
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            let next = {
                let mut replies = self.replies.lock().unwrap();
                let queue = replies.entry(path).or_default();
                (!queue.is_empty()).then(|| queue.remove(0))
            };
            match next {
                Some((text, _)) if text == "panic" => panic!("planted"),
                Some((text, stop)) => Ok(reply(&text, stop)),
                None => std::future::pending().await,
            }
        }
    }

    pub(super) fn router(role: ModelRole, model: Arc<dyn Model>) -> Router {
        let cfg = Config::from_lookup(|_| None).unwrap();
        Router::new(&cfg, vec![(role, route("haiku-5-5"), model)])
    }

    fn item(title: &str, category: &str, start: i64, end: i64, suggestion: Value) -> Value {
        json!({"body": "When x is empty, it panics.", "category": category, "end_line": end,
               "severity": "major", "snippet": "b", "start_line": start,
               "suggestion": suggestion, "title": title})
    }

    fn findings(items: &[Value]) -> String {
        json!({ "findings": items }).to_string()
    }

    #[tokio::test(start_paused = true)]
    async fn review_files_fails_files_alone_and_keeps_order() {
        let ok = |t: &str| findings(&[item(t, "logic", 2, 2, Value::Null)]);
        let cut = r#"{"findings": [{"body": "#;
        let bad = findings(&[item("x", "bogus", 2, 2, Value::Null)]);
        let files: Vec<FileChange> = ["a.go", "b.ts", "c.go", "d.go", "e.go", "f.js"]
            .iter()
            .map(|p| change(p, PATCH))
            .collect();
        // (path, its replies in call order, its delay in seconds)
        let script: [(&str, Replies, u64); 6] = [
            ("a.go", vec![(ok("first"), Stop::End)], 5),
            ("b.ts", vec![(cut.to_owned(), Stop::MaxTokens)], 1),
            ("c.go", vec![(bad.clone(), Stop::End), (bad, Stop::End)], 2),
            ("d.go", vec![], 1), // never answers: the router's timeout
            ("e.go", vec![(findings(&[]), Stop::End)], 3),
            ("f.js", vec![(ok("last"), Stop::End)], 1),
        ];
        let model = Arc::new(ByPath {
            replies: Mutex::new(
                script
                    .iter()
                    .map(|(p, r, _)| ((*p).to_owned(), r.clone()))
                    .collect(),
            ),
            delays: script
                .iter()
                .map(|(p, _, d)| ((*p).to_owned(), *d))
                .collect(),
            ..ByPath::default()
        });
        let router = router(ModelRole::Review, model.clone());
        let meter = Arc::new(Meter::default());
        let pr = PrContext::default();
        let got = review_files(&router, &meter, &files, &BTreeMap::new(), &pr, 2, later()).await;
        let titles: Vec<(&str, &str)> = got
            .findings
            .iter()
            .map(|f| (f.path.as_str(), f.title.as_str()))
            .collect();
        assert_eq!(titles, [("a.go", "first"), ("f.js", "last")]);
        assert_eq!(got.failed, ["b.ts", "c.go", "d.go"]);
        assert_eq!(
            model.peak.load(Ordering::SeqCst),
            2,
            "never more than concurrency"
        );
        assert_eq!(
            model.calls.load(Ordering::SeqCst),
            7,
            "c.go was retried once"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn review_files_concurrency_bound() {
        for concurrency in [0, 1, 3, 10] {
            let files: Vec<FileChange> =
                (0..6).map(|i| change(&format!("f{i}.go"), PATCH)).collect();
            let replies = files
                .iter()
                .map(|f| (f.path.clone(), vec![(findings(&[]), Stop::End)]))
                .collect();
            let model = Arc::new(ByPath {
                replies: Mutex::new(replies),
                ..ByPath::default()
            });
            let router = router(ModelRole::Review, model.clone());
            let pr = PrContext::default();
            let got = review_files(
                &router,
                &Arc::default(),
                &files,
                &BTreeMap::new(),
                &pr,
                concurrency,
                later(),
            )
            .await;
            assert_eq!(got, FileReview::default(), "{concurrency}");
            // The first file answers (1 s) before WARM_WAIT, alone; then the other 5.
            let want = concurrency.clamp(1, 5);
            assert_eq!(model.peak.load(Ordering::SeqCst), want, "{concurrency}");
        }
    }

    /// A deadline no test reaches.
    fn later() -> Instant {
        Instant::now() + Duration::from_secs(3600)
    }

    /// The smallest file of the largest system-prompt group goes first and alone, so the
    /// rest of its group read the prompt cache it wrote; the rest start at WARM_WAIT if
    /// it is slow. A file marks the PR's shared head unless no other file with its
    /// system prompt sends it.
    #[tokio::test(start_paused = true)]
    async fn review_files_warms_the_cache_with_the_smallest_file() {
        let big = format!("{PATCH}\n+more");
        let pr = PrContext {
            intent: "Fix the lookup.",
            ..PrContext::default()
        };
        let go3 = vec![("a.go", big.as_str()), ("b.go", PATCH), ("c.go", &big)];
        // (name, files, the first sent and its seconds, peak in flight, files marking
        // a shared head); the others take 3 s, WARM_WAIT is 2 s.
        let cases = [
            (
                "one language",
                go3.clone(),
                ("b.go", 1),
                2,
                vec!["a.go", "b.go", "c.go"],
            ),
            (
                "the largest group first",
                vec![("a.ts", PATCH), ("b.go", &big), ("c.go", &big)],
                ("b.go", 1),
                2,
                vec!["b.go", "c.go"],
            ),
            (
                "a slow first file",
                go3,
                ("b.go", 5),
                3,
                vec!["a.go", "b.go", "c.go"],
            ),
            ("one file", vec![("a.go", PATCH)], ("a.go", 1), 1, vec![]),
        ];
        for (name, files, (first, secs), peak, shared) in cases {
            let files: Vec<FileChange> = files.iter().map(|(p, d)| change(p, d)).collect();
            let replies = files
                .iter()
                .map(|f| (f.path.clone(), vec![(findings(&[]), Stop::End)]))
                .collect();
            let delay = |f: &FileChange| if f.path == first { secs } else { 3 };
            let delays = files.iter().map(|f| (f.path.clone(), delay(f))).collect();
            let model = Arc::new(ByPath {
                replies: Mutex::new(replies),
                delays,
                ..ByPath::default()
            });
            let router = router(ModelRole::Review, model.clone());
            let got = review_files(
                &router,
                &Arc::default(),
                &files,
                &BTreeMap::new(),
                &pr,
                3,
                later(),
            )
            .await;
            assert_eq!(got, FileReview::default(), "{name}");
            let sent = model.sent.lock().unwrap().clone();
            assert_eq!(sent[0].0, first, "{name}");
            assert_eq!(model.peak.load(Ordering::SeqCst), peak, "{name}");
            let marks = |p: &String| shared.contains(&p.as_str());
            assert!(
                sent.iter().all(|(p, n)| (*n > 0) == marks(p)),
                "{name}: {sent:?}"
            );
        }
    }

    /// One slot, 5 s per file, a 7 s deadline: a.go finishes, b.go is cut while
    /// running and c.go never starts; both fail, and c.go makes no call.
    #[tokio::test(start_paused = true)]
    async fn unstarted_files_fail_at_the_deadline() {
        let files: Vec<FileChange> = ["a.go", "b.go", "c.go"]
            .iter()
            .map(|p| change(p, PATCH))
            .collect();
        let ok = findings(&[item("first", "logic", 2, 2, Value::Null)]);
        let model = Arc::new(ByPath {
            replies: Mutex::new(
                files
                    .iter()
                    .map(|f| (f.path.clone(), vec![(ok.clone(), Stop::End)]))
                    .collect(),
            ),
            delays: files.iter().map(|f| (f.path.clone(), 5)).collect(),
            ..ByPath::default()
        });
        let router = router(ModelRole::Review, model.clone());
        let pr = PrContext::default();
        let deadline = Instant::now() + Duration::from_secs(7);
        let got = review_files(
            &router,
            &Arc::default(),
            &files,
            &BTreeMap::new(),
            &pr,
            1,
            deadline,
        )
        .await;
        let paths: Vec<&str> = got.findings.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(paths, ["a.go"]);
        assert_eq!(got.failed, ["b.go", "c.go"]);
        assert_eq!(model.calls.load(Ordering::SeqCst), 2, "c.go never called");
    }

    /// The log lines written while it is the thread's subscriber; a current-thread
    /// test runtime runs the spawned reviews on this thread too.
    #[derive(Clone, Default)]
    struct Logs(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for Logs {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// a.go panics and b.go is still running at the deadline: a.go logs a failure with
    /// the panic, not a deadline cut, whose remedy is a longer STRUCTURAL_TIMEOUT.
    #[tokio::test(start_paused = true)]
    async fn a_panic_is_no_deadline_cut() {
        let files: Vec<FileChange> = ["a.go", "b.go", "c.go"]
            .iter()
            .map(|p| change(p, PATCH))
            .collect();
        let model = Arc::new(ByPath {
            replies: Mutex::new(HashMap::from([
                ("a.go".to_owned(), vec![("panic".to_owned(), Stop::End)]),
                ("c.go".to_owned(), vec![(findings(&[]), Stop::End)]),
            ])),
            ..ByPath::default()
        });
        let router = router(ModelRole::Review, model);
        let pr = PrContext::default();
        // A second live dispatcher: with ours alone, a callsite a parallel test hits
        // first caches that thread's "never" (tracing-core's `has_just_one`).
        let _second = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        let logs = Logs::default();
        let writer = logs.clone();
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(move || writer.clone())
            .finish();
        let _logs = tracing::subscriber::set_default(subscriber);
        let deadline = Instant::now() + Duration::from_secs(5);
        let got = review_files(
            &router,
            &Arc::default(),
            &files,
            &BTreeMap::new(),
            &pr,
            3,
            deadline,
        )
        .await;
        assert_eq!(got.failed, ["a.go", "b.go"]);
        let text = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
        let lines: Vec<Value> = text
            .lines()
            .map(|l| serde_json::from_str::<Value>(l).unwrap()["fields"].take())
            .collect();
        let said: Vec<(&str, &str)> = lines
            .iter()
            .map(|f| (f["path"].as_str().unwrap(), f["message"].as_str().unwrap()))
            .collect();
        assert_eq!(
            said,
            [
                ("a.go", "file review failed"),
                ("b.go", "file review cut at the review deadline"),
            ]
        );
        let err = lines[0]["err"].as_str().unwrap();
        assert!(err.contains("panicked with message \"planted\""), "{err}");
    }

    #[tokio::test]
    async fn decode_rules() {
        let items = [
            item("kept", "nil-safety", 5, 3, json!("x := 1")),
            item("", "logic", 2, 2, Value::Null),
            item(" ", "logic", 2, 2, Value::Null),
            json!({"body": " ", "category": "logic", "end_line": 2, "severity": "minor",
                   "snippet": "b", "start_line": 2, "suggestion": null, "title": "t"}),
            item("null suggestion", "error-handling", 2, 4, Value::Null),
        ];
        let model = Script::new(&[(&findings(&items), Stop::End)]);
        let router = router(ModelRole::Review, model);
        let pr = PrContext::default();
        let files = [change("pkg/a.go", PATCH)];
        let got = review_files(
            &router,
            &Arc::default(),
            &files,
            &BTreeMap::new(),
            &pr,
            1,
            later(),
        )
        .await;
        // (title, category, start, end, suggestion), all on the reviewed path.
        let want = [
            ("kept", Category::NilSafety, 5, 5, "x := 1"),
            ("null suggestion", Category::ErrorHandling, 2, 4, ""),
        ]
        .map(
            |(title, category, start_line, end_line, suggestion)| Finding {
                path: "pkg/a.go".to_owned(),
                start_line,
                end_line,
                severity: Severity::Major,
                category,
                title: title.to_owned(),
                body: "When x is empty, it panics.".to_owned(),
                snippet: "b".to_owned(),
                suggestion: suggestion.to_owned(),
            },
        );
        assert_eq!(got.findings, want);
        assert!(got.failed.is_empty());
    }

    /// An enum value outside the schema is `Invalid`, retried, and the retry counts.
    #[tokio::test]
    async fn unknown_category_is_retried() {
        let bad = findings(&[item("x", "style", 2, 2, Value::Null)]);
        let good = findings(&[item("x", "logic", 2, 2, Value::Null)]);
        let model = Script::new(&[(&bad, Stop::End), (&good, Stop::End)]);
        let router = router(ModelRole::Review, model.clone());
        let meter = Arc::new(Meter::default());
        let pr = PrContext::default();
        let files = [change("a.go", PATCH)];
        let got = review_files(&router, &meter, &files, &BTreeMap::new(), &pr, 1, later()).await;
        assert_eq!(got.findings.len(), 1);
        assert_eq!(model.calls(), 2);
        let usage = router.usage(&meter);
        assert_eq!(usage[0].tally.retried, 1);
        let users = model.users.lock().unwrap();
        assert!(users[1].contains("unknown variant `style`"), "{}", users[1]);
    }

    fn walk_reply(changes: Value, score: i64, diagram: Value) -> String {
        json!({
            "bullets": [
                {"heading": "Bug Fixes", "items": ["Fix `lookup`."]},
                {"heading": " ", "items": ["dropped"]},
                {"heading": "Chores", "items": []},
            ],
            "changes": changes,
            "diagram": diagram,
            "effort_minutes": -5,
            "effort_score": score,
            "summary": " Fixes the lookup. ",
        })
        .to_string()
    }

    #[tokio::test]
    async fn walkthrough_maps_the_reply() {
        let files = [
            change("pkg/a.go", PATCH),
            change("main.go", "@@ -1,2 +1,1 @@\n-x\n-y\n+z"),
        ];
        let changes = json!([
            {"files": ["pkg/a.go"], "summary": " Fixed. "},
            {"files": [], "summary": "dropped"},
            {"files": ["main.go"], "summary": "Wired."},
        ]);
        let mermaid = "sequenceDiagram\n    A->>B: call";
        // (effort score, clamped, diagram, kept diagram, dropped diagram)
        let cases = [
            (9, 5, json!(mermaid), mermaid, None),
            (0, 1, Value::Null, "", None),
            (2, 2, json!(" "), "", None),
            (3, 3, json!("NONE"), "", Some("NONE")),
            (4, 4, json!("graph TD; A-->B"), "", Some("graph TD; A-->B")),
        ];
        for (score, want_score, diagram, want_diagram, want_dropped) in cases {
            let text = walk_reply(changes.clone(), score, diagram);
            let model = Script::new(&[(&text, Stop::End), (&text, Stop::End)]);
            let router = router(ModelRole::Summary, model.clone());
            let (got, dropped) =
                walkthrough_with_dropped(&router, &Meter::default(), &files, "Fix lookup.")
                    .await
                    .unwrap();
            assert_eq!(dropped.as_deref(), want_dropped, "{score}");
            let plain = walkthrough(&router, &Meter::default(), &files, "Fix lookup.").await;
            assert_eq!(plain.as_ref(), Ok(&got), "{score}");
            let want = Walkthrough {
                bullets: vec![BulletGroup {
                    heading: "Bug Fixes".to_owned(),
                    items: vec!["Fix `lookup`.".to_owned()],
                }],
                summary: "Fixes the lookup.".to_owned(),
                changes: vec![
                    CohortRow {
                        cohort: "pkg".to_owned(),
                        files: vec!["pkg/a.go".to_owned()],
                        summary: "Fixed.".to_owned(),
                    },
                    CohortRow {
                        cohort: "(root)".to_owned(),
                        files: vec!["main.go".to_owned()],
                        summary: "Wired.".to_owned(),
                    },
                ],
                effort_score: want_score,
                effort_minutes: 0,
                diagram: want_diagram.to_owned(),
                files_changed: 2,
                additions: 2,
                deletions: 2,
                ..Walkthrough::default()
            };
            assert_eq!(got, want, "{score}");
            assert!(
                model.users.lock().unwrap()[0].contains("<pr_intent>\nFix lookup.\n</pr_intent>")
            );
        }
    }

    #[tokio::test]
    async fn walkthrough_errors() {
        let files = [change("a.go", PATCH)];
        let no_changes = walk_reply(json!([{"files": [], "summary": "x"}]), 2, Value::Null);
        let cases: [(Vec<(&str, Stop)>, &str); 3] = [
            (vec![(&no_changes, Stop::End)], "invalid"),
            (vec![(&no_changes[..20], Stop::MaxTokens)], "truncated"),
            (vec![], "request"),
        ];
        for (script, want) in cases {
            let router = router(ModelRole::Summary, Script::new(&script));
            let got = walkthrough(&router, &Meter::default(), &files, "").await;
            let kind = match got {
                Err(ModelError::Invalid(_)) => "invalid",
                Err(ModelError::Truncated) => "truncated",
                Err(ModelError::Request(_)) => "request",
                other => panic!("{want}: {other:?}"),
            };
            assert_eq!(kind, want);
        }
    }

    /// Blank requirements go, the rest keep their order up to ten, and the issue
    /// is the prompt's intent.
    #[tokio::test]
    async fn ticket_fit_maps_the_reply() {
        let row = |source: &str, requirement: &str, status: &str, note: &str| json!({"note": note, "requirement": requirement, "source": source, "status": status});
        let mut requirements = vec![
            row("issue", " Lookup works. ", "done", " pkg/a.go: lookup "),
            row("issue", " ", "missing", "x"),
            row("story", "Story rule", "partial", ""),
        ];
        for n in 0..12 {
            requirements.push(row("issue", &format!("R{n}"), "missing", ""));
            requirements.push(row("story", &format!("S{n}"), "missing", ""));
        }
        let context = " The story adds lookup; this task is its UI. ";
        let text = json!({ "context": context, "requirements": requirements }).to_string();
        let ticket = Ticket {
            key: "PROJ-1".to_owned(),
            text: "Jira PROJ-1 [Sub-task, To Do]: Lookup".to_owned(),
            story: Some("PROJ-0".to_owned()),
        };
        let files = [change("pkg/a.go", PATCH)];
        let fit = async |ticket: &Ticket| {
            let model = Script::new(&[(&text, Stop::End)]);
            let router = router(ModelRole::Summary, model.clone());
            let meter = Meter::default();
            let got = ticket_fit(&router, &meter, &files, ticket).await.unwrap();
            (got, model.users.lock().unwrap()[0].clone())
        };

        let (got, user) = fit(&ticket).await;
        assert_eq!(
            (got.key.as_str(), got.story.as_deref()),
            ("PROJ-1", Some("PROJ-0"))
        );
        let first = Requirement {
            text: "Lookup works.".to_owned(),
            fit: Fit::Done,
            note: "pkg/a.go: lookup".to_owned(),
            from_story: false,
        };
        assert_eq!(got.context, "The story adds lookup; this task is its UI.");
        assert_eq!(got.requirements[0], first);
        let (issue, story): (Vec<_>, Vec<_>) = got.requirements.iter().partition(|r| !r.from_story);
        assert_eq!(
            (issue.len(), story.len()),
            (MAX_REQUIREMENTS, MAX_REQUIREMENTS)
        );
        assert_eq!(
            (issue[9].text.as_str(), story[0].text.as_str()),
            ("R8", "Story rule")
        );
        assert!(user.contains("<pr_intent>\nJira PROJ-1 [Sub-task, To Do]: Lookup\n</pr_intent>"));

        // Without a story in the context, story rows are the model's invention.
        let (got, _) = fit(&Ticket {
            story: None,
            ..ticket.clone()
        })
        .await;
        assert_eq!(got.story, None);
        assert!(got.requirements.iter().all(|r| !r.from_story));
    }

    #[test]
    fn basic_walkthrough_text() {
        let files = [
            change("web/b.ts", PATCH),
            change("main.go", "@@ -1 +1 @@\n-x\n+y"),
            change("web/a.ts", PATCH),
        ];
        let got = basic_walkthrough(&files);
        let row = |cohort: &str, files: &[&str]| CohortRow {
            cohort: cohort.to_owned(),
            files: files.iter().map(|f| (*f).to_owned()).collect(),
            summary: String::new(),
        };
        let want = Walkthrough {
            summary: "Review of 3 changed file(s).".to_owned(),
            changes: vec![
                row("(root)", &["main.go"]),
                row("web", &["web/a.ts", "web/b.ts"]),
            ],
            files_changed: 3,
            additions: 3,
            deletions: 1,
            ..Walkthrough::default()
        };
        assert_eq!(got, want);
    }

    #[test]
    fn mermaid_patterns_compile() {
        assert!(MERMAID.is_some());
    }

    #[test]
    fn mermaid_validation() {
        let cases = [
            ("sequenceDiagram\n  A->>B: hi", true),
            (
                "sequenceDiagram\n  participant A as The API\n  actor U\n  U->>A: go\n  A-->>U: ok",
                true,
            ),
            (
                "sequenceDiagram\n  Note over A,B: x\n  A-xB: lost\n  A-)B: async",
                true,
            ),
            ("sequenceDiagram\n  loop each\n    A->>B: x\n  end", true),
            (
                "sequenceDiagram\n  alt ok\n    A->>B: x\n  else bad\n    A->>C: y\n  end",
                true,
            ),
            (
                "sequenceDiagram\n  %% comment\n\n  autonumber\n  activate A\n  A->>+B: x",
                true,
            ),
            ("sequenceDiagram\n  loop each\n    A->>B: x", false),
            ("sequenceDiagram\n  A->>B: x\n  end", false),
            ("sequenceDiagram\n  else\n  A->>B: x", false),
            ("sequenceDiagram\n  participant A", false),
            ("sequenceDiagram\n  A->>B x", false),
            ("sequenceDiagram\n  A->>B: x\n  A->>B: cut mid", true),
            ("graph TD\n  A-->B", false),
            ("NONE", false),
            ("", false),
        ];
        for (d, want) in cases {
            assert_eq!(valid_sequence_diagram(d), want, "{d}");
        }
    }

    #[test]
    fn mermaid_extraction() {
        let cases = [
            (
                "```mermaid\nsequenceDiagram\n  A->>B: x\n```\nprose",
                "sequenceDiagram\n  A->>B: x",
            ),
            (
                "Here is the flow:\nsequenceDiagram\n  A->>B: x\n```",
                "sequenceDiagram\n  A->>B: x",
            ),
            (
                "  sequenceDiagram\n  A->>B: x  ",
                "sequenceDiagram\n  A->>B: x",
            ),
            ("NONE", "NONE"),
        ];
        for (text, want) in cases {
            assert_eq!(extract_mermaid(text), want, "{text}");
        }
        assert_eq!(diagram("NONE"), "");
        assert_eq!(
            diagram("sequenceDiagram\n  A->>B: x"),
            "sequenceDiagram\n  A->>B: x"
        );
    }

    /// Diagram replies the way recorded ones came back, through extract and validate:
    /// NONE gives no diagram, a complete one renders, and one cut off at the token
    /// limit is dropped unless the cut fell inside a message's text, which still
    /// renders. A cut summary reply is `Truncated` as a whole (`walkthrough_errors`),
    /// so none reaches `diagram` in a Review.
    #[test]
    fn recorded_style_diagrams() {
        const WHOLE: &str = "sequenceDiagram\n  participant C as Cart\n  C->>P: Charge(total)\n  alt paid\n    P-->>C: receipt\n  else declined\n    P-->>C: error\n  end";
        let fenced = format!("Here is the flow:\n```mermaid\n{WHOLE}\n```\nThat is all.");
        let cases = [
            ("NONE", "NONE", ""),
            ("NONE with whitespace", "\n NONE \n", ""),
            ("prose only", "The change only renames a field.", ""),
            ("complete, fenced in prose", fenced.as_str(), WHOLE),
            ("complete, unfenced", WHOLE, WHOLE),
            (
                "cut inside a block",
                "```mermaid\nsequenceDiagram\n  C->>P: Charge(total)\n  alt paid\n    P-->>C: receipt",
                "",
            ),
            (
                "cut inside an arrow",
                "```mermaid\nsequenceDiagram\n  C->>P: Charge(total)\n  P-->>",
                "",
            ),
            (
                "cut inside a message's text",
                "```mermaid\nsequenceDiagram\n  C->>P: Charge(total)\n  P-->>C: rece",
                "sequenceDiagram\n  C->>P: Charge(total)\n  P-->>C: rece",
            ),
        ];
        for (name, text, want) in cases {
            assert_eq!(diagram(text), want, "{name}");
        }
    }
}
