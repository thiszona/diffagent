//! Comment rendering: the GitHub markdown the bot posts, and the hidden markers it
//! reads back. Pure string building, so every section is testable without GitHub.
//!
//! Markers are versioned (`diffagent:v1`), so a comment with an unversioned marker
//! or another version is never read as ours.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;

use crate::diff::normalize_code;
use crate::finding::{Finding, Severity};
use crate::job::mention;

/// The bot's name without an App to take a slug from: dev mode and diffagent-eval.
const DEFAULT_BOT_NAME: &str = "diffagent";

/// The status comment's heading for the bot `name` (`Deps::bot_name`):
/// `### {name} review`, or `### {name}` when the name already ends in "review", so it
/// never reads "review review".
fn title(name: &str) -> String {
    if name.to_ascii_lowercase().ends_with("review") {
        format!("### {name}")
    } else {
        format!("### {name} review")
    }
}

/// The placeholder posted the moment a Review starts; it is edited into the
/// Walkthrough when the Review completes, so the PR shows one comment that morphs.
pub fn reviewing_banner(name: &str) -> String {
    format!(
        "{}\n\n\
         > [!NOTE]\n\
         > **Review in progress.** This comment becomes the review report when the review \
         completes.",
        title(name)
    )
}

/// Replaces [`reviewing_banner`] when the Review could not complete. The job is
/// retried, so the next attempt (or the next push) runs it again.
pub fn failed_banner(name: &str) -> String {
    format!(
        "{}\n\n\
         > [!WARNING]\n\
         > **Review interrupted.** It could not complete and will retry.",
        title(name)
    )
}

/// Whether `body` is exactly a status comment's banner under the bot `name`: what a
/// Review adopts as its status comment. A Walkthrough never is one.
pub fn is_banner(name: &str, body: &str) -> bool {
    body == reviewing_banner(name) || body == failed_banner(name)
}

/// GitHub refuses a comment longer than 65,536 characters with a 422, which ends a
/// Review Dead. Counted in bytes, never fewer than characters.
pub const MAX_COMMENT: usize = 65_536;

const CUT_NOTE: &str = "\n\n… _(cut here: GitHub limits a comment to 65,536 characters)_\n";

/// `body`, cut on a char boundary under [`MAX_COMMENT`] with a note when it is longer.
///
/// ponytail: keeps the head, so a cut Walkthrough loses its tail (commands, agent
/// brief) and may leave a fence or `<details>` open; budget each part if that is ever
/// seen.
pub fn fit_comment(mut body: String) -> String {
    if body.len() > MAX_COMMENT {
        body.truncate(body.floor_char_boundary(MAX_COMMENT - CUT_NOTE.len()));
        body.push_str(CUT_NOTE);
    }
    body
}

/// What the Walkthrough comment shows: the summary model's output plus the
/// deterministic facts of the Review.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Walkthrough {
    /// Categorized summary points ("Bug Fixes", "New Features", ...).
    pub bullets: Vec<BulletGroup>,
    /// Prose summary.
    pub summary: String,
    /// The Files table: one row per group of files.
    pub changes: Vec<CohortRow>,
    /// Estimated review effort, 1 (trivial) to 5 (very heavy); 0 = no estimate.
    pub effort_score: u32,
    /// Estimated review minutes; 0 = no estimate.
    pub effort_minutes: u32,
    /// A Mermaid sequence diagram, if the summary model drew one.
    pub diagram: String,
    /// Reviewed-file stats. They differ from GitHub's own PR stats because tests,
    /// lockfiles and generated files are filtered out before review.
    pub files_changed: usize,
    pub additions: usize,
    pub deletions: usize,
    /// Earlier Findings this push resolved.
    pub addressed: Vec<String>,
    /// Files whose model call failed: a Review with failures is incomplete, never
    /// an all-clear.
    pub failed_files: usize,
    /// Findings from earlier pushes still open: no all-clear while any remain.
    pub prior_open: usize,
    /// The PR's files checked against its Jira issue, when it names one.
    pub ticket: Option<TicketFit>,
    /// The head commit the Review read; empty when unknown.
    pub head: String,
    /// `owner/repo #N`, for the heading; empty when unknown.
    pub pr: String,
    /// The bot's name, for the heading; empty is [`DEFAULT_BOT_NAME`].
    pub bot_name: String,
    /// The merge gate runs, so an open critical Finding blocks the merge.
    pub merge_gate: bool,
    /// The Open Findings that fail the merge gate once this Review's rows are written;
    /// 0 with the gate off.
    pub blockers: usize,
}

/// The Ticket fit: the issue's requirements, and its story's when it is a sub-task,
/// and whether the PR meets each. Advisory: it never touches the merge gate.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TicketFit {
    pub key: String,
    /// The story's key, when its criteria were checked too.
    pub story: Option<String>,
    /// What the story asks for and which part of it the issue is, in a sentence or two.
    pub context: String,
    pub requirements: Vec<Requirement>,
}

/// One requirement of the issue or its story, in one line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Requirement {
    pub text: String,
    pub fit: Fit,
    /// Where the PR meets it, or what is missing.
    pub note: String,
    /// One of the story's acceptance criteria, not the issue's own.
    pub from_story: bool,
}

/// How much of a requirement the PR's diffs show.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Fit {
    Done,
    Partial,
    Missing,
}

/// One categorized block of summary points.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BulletGroup {
    pub heading: String,
    pub items: Vec<String>,
}

/// One row of the Files table: a group of files and what changed there.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CohortRow {
    pub cohort: String,
    pub files: Vec<String>,
    pub summary: String,
}

/// The severity's label, shown in bold.
pub fn severity_label(s: Severity) -> &'static str {
    match s {
        Severity::Nitpick => "NIT",
        Severity::Minor => "MINOR",
        Severity::Major => "MAJOR",
        Severity::Critical => "CRITICAL",
    }
}

/// `s` on one line with its `|` escaped, so it stays inside one table cell.
pub(crate) fn table_cell(s: &str) -> String {
    normalize_code(s).replace('|', "\\|")
}

/// `s` on one line as inline code. A backtick would end the span early, so it is
/// dropped.
fn code(s: &str) -> String {
    format!("`{}`", normalize_code(s).replace('`', ""))
}

/// The finding's header: its severity, category, the function it is in and whether it
/// blocks the merge, then its title as a heading. Critical and major findings use
/// GitHub alert callouts so real defects visually dominate the PR. The title is
/// collapsed to one line: a title spanning lines could otherwise open a
/// ```suggestion fence that [`sanitize_body`] never sees. The heading follows the
/// tag line with no blank line, so the header stays one block, which the Thread's
/// prose extraction skips.
fn severity_header(f: &Finding, symbol: Option<&str>, blocks: bool) -> String {
    let mut tags = vec![
        format!("**{}**", severity_label(f.severity)),
        f.category.as_str().to_owned(),
    ];
    tags.extend(symbol.map(code));
    if blocks {
        tags.push("blocks merge".to_owned());
    }
    let (line, title) = (tags.join(" · "), normalize_code(&f.title));
    match f.severity {
        Severity::Critical => format!("> [!CAUTION]\n> {line}\n#### {title}\n"),
        Severity::Major => format!("> [!WARNING]\n> {line}\n#### {title}\n"),
        Severity::Minor | Severity::Nitpick => format!("{line}\n#### {title}\n"),
    }
}

/// The Walkthrough's heading and its line of facts: the head, the reviewed files and
/// lines, the review load, and how many of the `located` Findings' locations matched
/// the diff (`verified`).
pub fn render_heading(w: &Walkthrough, verified: usize, located: usize) -> String {
    let name = if w.bot_name.is_empty() {
        DEFAULT_BOT_NAME
    } else {
        &w.bot_name
    };
    let mut b = title(name);
    if !w.pr.is_empty() {
        b.push_str(&format!(" · {}", w.pr));
    }
    b.push('\n');
    let mut facts = Vec::new();
    if !w.head.is_empty() {
        facts.push(format!("`{}`", w.head.get(..7).unwrap_or(&w.head)));
    }
    if w.files_changed > 0 {
        let noun = if w.files_changed == 1 {
            "file"
        } else {
            "files"
        };
        facts.push(format!("{} {noun}", w.files_changed));
        facts.push(format!("+{} −{}", w.additions, w.deletions));
    }
    if w.effort_score > 0 {
        let mut load = format!("review load **{}**", effort_label(w.effort_score));
        if w.effort_minutes > 0 {
            load.push_str(&format!(", about {} min", w.effort_minutes));
        }
        facts.push(load);
    }
    if located > 0 {
        facts.push(format!(
            "{verified} of {located} locations checked against the diff"
        ));
    }
    if !facts.is_empty() {
        b.push_str(&format!("\n{}\n", facts.join(" · ")));
    }
    b
}

/// The Walkthrough's sections: the summary, `posted` (this push's inline Findings
/// with the function each is in) as a table, the Findings fixed since, the key
/// changes, the Files table, the Ticket fit and the Mermaid diagram.
pub fn render_walkthrough(w: &Walkthrough, posted: &[(&Finding, Option<&str>)]) -> String {
    let mut b = String::new();
    let summary = w.summary.trim();
    if !summary.is_empty() {
        b.push_str(&format!("\n#### Summary\n\n{summary}\n"));
    } else if w.bullets.is_empty() {
        b.push_str("\n#### Summary\n\n_No summary available._\n");
    }

    if !posted.is_empty() {
        b.push_str(
            "\n#### Findings on this push\n\n| Level | Finding | Location |\n|---|---|---|\n",
        );
        for (f, symbol) in posted {
            let mut at = code(&format!("{}:{}", f.path, f.start_line));
            if let Some(s) = symbol {
                at.push_str(&format!(" · {}", code(s)));
            }
            b.push_str(&format!(
                "| **{}** | {} | {} |\n",
                severity_label(f.severity),
                table_cell(&f.title),
                table_cell(&at)
            ));
        }
    }

    if !w.addressed.is_empty() {
        b.push_str("\n#### Fixed since the last review\n\n");
        for a in &w.addressed {
            b.push_str(&format!("- ✓ {}\n", a.trim()));
        }
    }

    if !w.bullets.is_empty() {
        b.push_str("\n#### Key changes\n");
        for g in &w.bullets {
            b.push_str(&format!("\n**{}**\n", g.heading));
            for item in &g.items {
                b.push_str(&format!("- {}\n", item.trim()));
            }
        }
    }

    if !w.changes.is_empty() {
        b.push_str("\n#### Files\n\n| Area | Files | What changed |\n|---|---|---|\n");
        for row in &w.changes {
            let files: Vec<String> = row.files.iter().map(|p| table_cell(&code(p))).collect();
            b.push_str(&format!(
                "| {} | {} | {} |\n",
                table_cell(&row.cohort),
                files.join("<br>"),
                table_cell(&row.summary)
            ));
        }
    }

    if let Some(t) = w.ticket.as_ref().filter(|t| !t.requirements.is_empty()) {
        b.push_str(&format!("\n#### Ticket {}\n", normalize_code(&t.key)));
        match normalize_code(&t.context) {
            c if c.is_empty() => {}
            c => b.push_str(&format!("\n> {c}\n")),
        }
        let story = t
            .story
            .as_deref()
            .map(|k| format!("**Story {}**", normalize_code(k)));
        for (from_story, group) in [(false, None), (true, story)] {
            let rows: Vec<&Requirement> = t
                .requirements
                .iter()
                .filter(|r| r.from_story == from_story)
                .collect();
            if rows.is_empty() {
                continue;
            }
            b.push('\n');
            if let Some(g) = group {
                b.push_str(&format!("{g}\n\n"));
            }
            b.push_str("| Status | Requirement | Note |\n|---|---|---|\n");
            for r in rows {
                let status = match r.fit {
                    Fit::Done => "✓ done",
                    Fit::Partial => "~ partial",
                    Fit::Missing => "✗ missing",
                };
                b.push_str(&format!(
                    "| {status} | {} | {} |\n",
                    table_cell(&r.text),
                    table_cell(&r.note)
                ));
            }
        }
        b.push_str(
            "\n<sub>Checked by the model against this PR alone: work in another PR shows as \
             missing. The ticket never blocks a merge.</sub>\n",
        );
    }

    let diagram = w.diagram.trim();
    if !diagram.is_empty() {
        // GitHub renders ```mermaid natively.
        let fence = code_fence(diagram);
        b.push_str(&format!(
            "\n#### Flow\n\n{fence}mermaid\n{diagram}\n{fence}\n"
        ));
    }
    b
}

/// The review load a 1-5 effort score names.
fn effort_label(score: u32) -> &'static str {
    match score {
        0 | 1 => "trivial",
        2 => "light",
        3 => "medium",
        4 => "heavy",
        _ => "very heavy",
    }
}

/// The collapsed command list appended to every Walkthrough, each with who may send
/// it: `review` the PR's author or a Maintainer, the rest Maintainers only.
pub fn render_commands() -> String {
    let m = mention();
    format!(
        "\n<details>\n<summary>Commands</summary>\n\n\
         | Comment | What it does | Who |\n\
         |---|---|---|\n\
         | `{m} review` | Reviews the current head again, and resumes a paused PR | Author or Maintainer |\n\
         | `{m} pause` | Stops automatic reviews on this PR until the next `review` | Maintainer |\n\
         | `{m} resolve` | Closes every open finding | Maintainer |\n\
         | `{m} lessons` | Lists this repository's Lessons | Maintainer |\n\
         | `{m} forget <id>` | Deletes that Lesson and reopens the finding it suppressed | Maintainer |\n\
         | `{m} learn` | In a finding's thread, stores the Lesson proposed there and suppresses the finding | Maintainer |\n\
         | `{m} close` | In a finding's thread, closes that finding with no Lesson | Maintainer |\n\
         \n</details>\n"
    )
}

/// One paste-able brief for coding agents covering every Finding, numbered, split into
/// the anchored ones (`inline`, each with the function it is in) and those not on a
/// changed line. Empty when there are none.
pub fn render_consolidated_agent_prompt(
    inline: &[(&Finding, Option<&str>)],
    demoted: &[&Finding],
) -> String {
    let total = inline.len() + demoted.len();
    if total == 0 {
        return String::new();
    }
    let mut body = String::from(
        "Check each item against the code as it is now; change only what is still wrong.\n",
    );
    let anchored = inline
        .iter()
        .map(|(f, s)| agent_prompt(&anchored_loc(f, *s), f));
    let elsewhere = demoted.iter().map(|f| agent_prompt(&around(f), f));
    let mut n = 0;
    for (heading, items) in [
        ("On changed lines", anchored.collect::<Vec<_>>()),
        ("Elsewhere", elsewhere.collect()),
    ] {
        if !items.is_empty() {
            body.push_str(&format!("\n{heading}:\n"));
            for item in items {
                n += 1;
                body.push_str(&format!("{n}. {item}\n"));
            }
        }
    }
    let body = body.trim_end_matches('\n');
    let fence = code_fence(body);
    let noun = if total == 1 { "finding" } else { "findings" };
    format!(
        "\n<details>\n<summary>Brief for coding agents ({total} {noun})</summary>\n\n\
         {fence}\n{body}\n{fence}\n\n</details>\n"
    )
}

/// One inline comment body: the header, body, the one-click suggestion (when
/// `committable`) or else a proposed-fix diff against the anchored span, the agent
/// brief, a line on how the Finding closes, and the hidden Fingerprint marker.
///
/// `symbol` is the function the anchored lines are in, and `blocks` says the Finding
/// fails the merge gate. `old_lines` is the anchored span's current content, which
/// turns the proposed fix into a real red/green diff. `verified` (the location matched
/// the diff) only changes the fix's label; `committable` (verified, whole span in the
/// diff, and allowed by config) posts the ```suggestion block instead. They are
/// separate so advisory mode can turn off one-click without claiming the location is
/// unverified.
pub fn render_comment(
    f: &Finding,
    fingerprint: &str,
    symbol: Option<&str>,
    blocks: bool,
    verified: bool,
    committable: bool,
    old_lines: &[&str],
) -> String {
    let mut b = severity_header(f, symbol, blocks);
    b.push_str(&format!("\n{}\n", sanitize_body(&f.body)));

    // Defensive: the plan already clears no-op suggestions, but one reaching here
    // must never render a self-cancelling diff or a block proposing unchanged code.
    let mut fix = String::new();
    let sug = f.suggestion.trim_end_matches('\n');
    if !sug.trim().is_empty() && !is_no_op_suggestion(&f.suggestion, old_lines) {
        fix = if committable {
            let fence = code_fence(sug);
            format!(
                "\n<details open>\n<summary>Suggested change</summary>\n\n\
                 {fence}suggestion\n{sug}\n{fence}\n\n</details>\n"
            )
        } else {
            render_proposed_fix(sug, old_lines, verified)
        };
    }

    let prompt = agent_prompt(&anchored_loc(f, symbol), f);
    let fence = code_fence(&prompt);
    let mut tail = format!(
        "\n<details>\n<summary>Brief for a coding agent</summary>\n\n{fence}\n{prompt}\n{fence}\n\n</details>\n"
    );
    tail.push_str(&format!("\n<sub>{}</sub>\n", closes(f.severity)));
    // Lets a retried job recognise a comment a crashed attempt already posted.
    tail.push_str(&format!("\n{}\n", finding_marker(fingerprint)));

    // A wide span (a whole new file) with a suggestion can pass GitHub's comment limit,
    // and one comment it refuses fails the whole review post: the fix goes first.
    // ponytail: a body or prompt that alone passes the limit is still refused.
    if b.len() + fix.len() + tail.len() > MAX_COMMENT {
        fix = FIX_LEFT_OUT.to_owned();
    }
    b + &fix + &tail
}

/// How a Finding of severity `s` closes: a reply never closes a critical one.
fn closes(s: Severity) -> String {
    match s {
        Severity::Critical => format!(
            "A critical finding closes only when a push fixes it or a Maintainer comments \
             `{} resolve`.",
            mention()
        ),
        _ => "Disagree? Reply with your reason. A Maintainer's reason becomes a Lesson for \
              this repository."
            .to_owned(),
    }
}

/// Stands in for a proposed fix too long for a GitHub comment.
const FIX_LEFT_OUT: &str =
    "\n_The proposed fix is left out: it does not fit in a GitHub comment._\n";

/// The suggestion as a ```diff against the span's current content; additions only
/// when the old side is unknown.
fn render_proposed_fix(sug: &str, old_lines: &[&str], verified: bool) -> String {
    let removed = old_lines.iter().map(|l| format!("-{l}"));
    let added = sug.split('\n').map(|l| format!("+{l}"));
    let diff = removed.chain(added).collect::<Vec<_>>().join("\n");
    let fence = code_fence(&diff);
    let summary = if verified {
        "Proposed fix (review before applying)"
    } else {
        "Proposed fix (location not matched in the diff; check it before applying)"
    };
    format!(
        "\n<details>\n<summary>{summary}</summary>\n\n{fence}diff\n{diff}\n{fence}\n\n</details>\n"
    )
}

/// Whether the suggestion only repeats the span's current content: a "fix" that
/// changes nothing. A trailing newline and per-line trailing whitespace are ignored,
/// so a cosmetic echo still counts. An unknown old side (empty) can never prove a
/// no-op, and a blank suggestion is not one: there is nothing to suppress.
pub(crate) fn is_no_op_suggestion(suggestion: &str, old_lines: &[&str]) -> bool {
    let trim = |l: &str| l.trim_end_matches([' ', '\t']).to_owned();
    !old_lines.is_empty()
        && !suggestion.trim().is_empty()
        && suggestion
            .trim_end_matches('\n')
            .split('\n')
            .map(trim)
            .eq(old_lines.iter().map(|l| trim(l)))
}

/// Makes model prose safe to post: GitHub treats any ```suggestion fence inside an
/// inline review comment as a one-click change, so a body that smuggles one would
/// bypass the verified/committable gate. Such fences become plain code blocks. Lines
/// end at `\r\n`, `\r` or `\n`, as in CommonMark, and come out ending in `\n`. A fence
/// the prose leaves open is closed: the next fence line posted after it would close it
/// instead, and the code that line was meant to open would be read as markdown.
pub(crate) fn sanitize_body(body: &str) -> String {
    let mut lines: Vec<String> = body
        .trim()
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .split('\n')
        .map(neutralize)
        .collect();
    let close = unclosed(&lines);
    lines.extend(close);
    lines.join("\n")
}

/// The line that closes the fence `lines` leave open, if any: the opener's run after
/// its lead with list markers blanked, so it closes inside the same blockquote or list
/// item. A closer is a run of the opener's character at least as long with nothing
/// after it, at the same blockquote depth; a backtick run with a backtick after it
/// opens nothing.
fn unclosed(lines: &[String]) -> Option<String> {
    let depth = |lead: &str| lead.matches('>').count();
    let mut open: Option<(&str, &str)> = None;
    for line in lines {
        let Some((lead, run, info)) = fence(line) else {
            continue;
        };
        open = match open {
            None if run.starts_with('`') && info.contains('`') => None,
            None => Some((lead, run)),
            Some((l, r))
                if depth(l) == depth(lead)
                    && run.starts_with(&r[..1])
                    && run.len() >= r.len()
                    && info.trim().is_empty() =>
            {
                None
            }
            keep => keep,
        };
    }
    let (lead, run) = open?;
    let lead: String = lead
        .chars()
        .map(|c| {
            if c == '>' || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect();
    Some(format!("{lead}{run}"))
}

/// `line`, its fence's info string made inert when GitHub could read it as
/// `suggestion`: every leading `suggestion` word goes, in any case, and an info string
/// holding `&` goes whole, since an entity there is decoded and can spell the word.
fn neutralize(line: &str) -> String {
    let Some((lead, run, info)) = fence(line) else {
        return line.to_owned();
    };
    if info.contains('&') {
        return format!("{lead}{run}");
    }
    let mut info = info.to_owned();
    while let Some(at) = suggestion_at(&info) {
        info.replace_range(at..at + "suggestion".len(), "");
    }
    format!("{lead}{run}{info}")
}

/// Where the first word of `info` starts when it is `suggestion`, in any case.
fn suggestion_at(info: &str) -> Option<usize> {
    let word = info.trim_start();
    let first = word.split_whitespace().next()?;
    first
        .eq_ignore_ascii_case("suggestion")
        .then_some(info.len() - word.len())
}

/// A fence line as (lead, run, info): the run of three or more backticks or tildes,
/// what precedes it, and the rest. A fence opens inside blockquotes and list items too,
/// so the lead is indentation and any run of `>` and list markers.
fn fence(line: &str) -> Option<(&str, &str, &str)> {
    let mut rest = line.trim_start();
    loop {
        let marker = if rest.starts_with('>') {
            1
        } else {
            list_marker(rest)
        };
        if marker == 0 {
            break;
        }
        rest = rest[marker..].trim_start();
    }
    let c = rest.chars().next().filter(|c| matches!(c, '`' | '~'))?;
    let info = rest.trim_start_matches(c);
    let run = &rest[..rest.len() - info.len()];
    (run.len() >= 3).then_some((&line[..line.len() - rest.len()], run, info))
}

/// The length of the list marker opening `s`: `-`, `+` or `*`, or one to nine digits
/// then `.` or `)`, followed by whitespace; 0 when there is none.
fn list_marker(s: &str) -> usize {
    let digits = s.bytes().take_while(u8::is_ascii_digit).count();
    let len = match s.as_bytes().get(digits) {
        Some(b'-' | b'+' | b'*') if digits == 0 => 1,
        Some(b'.' | b')') if (1..=9).contains(&digits) => digits + 1,
        _ => return 0,
    };
    if s[len..].starts_with(char::is_whitespace) {
        len
    } else {
        0
    }
}

/// A backtick fence one longer than the longest backtick run in `s` (at least three),
/// so content holding its own fences cannot close the block early.
pub(crate) fn code_fence(s: &str) -> String {
    let longest = s.split(|c| c != '`').map(str::len).max().unwrap_or(0);
    "`".repeat((longest + 1).max(3))
}

/// An anchored Finding's location for its brief: `path:start-end`, then the function
/// it is in.
fn anchored_loc(f: &Finding, symbol: Option<&str>) -> String {
    let mut loc = if f.start_line > 0 && f.end_line > f.start_line {
        format!("{}:{}-{}", f.path, f.start_line, f.end_line)
    } else if f.start_line > 0 {
        format!("{}:{}", f.path, f.start_line)
    } else {
        f.path.clone()
    };
    if let Some(s) = symbol {
        loc.push_str(&format!(" ({s})"));
    }
    loc
}

/// The location of a Finding not on a changed line: its line failed verification, so
/// it is given as approximate.
fn around(f: &Finding) -> String {
    if f.start_line > 0 {
        format!("{}, around line {}", f.path, f.start_line)
    } else {
        f.path.clone()
    }
}

/// A deterministic fix brief for a coding agent, built from the finding's structured
/// fields and always naming its location `loc`. The title is collapsed to one line and
/// the whole brief goes through [`sanitize_body`], as the prose does: the brief is
/// posted as code, but a block the prose leaves open, an HTML `<pre>` say, can end on
/// the brief's own lines and show the rest of it as markdown.
fn agent_prompt(loc: &str, f: &Finding) -> String {
    let title = normalize_code(&f.title);
    let mut prompt = if loc.is_empty() {
        format!("{title}.")
    } else {
        format!("{loc} — {title}.")
    };
    let body = f.body.trim();
    if !body.is_empty() {
        prompt.push(' ');
        prompt.push_str(body);
    }
    sanitize_body(&prompt)
}

/// Opens every hidden marker of ours. The reply sanitizer and the Finding-prose
/// extraction read it too: a copy that drifts from it stops neutralizing a forged
/// marker. A marker in any other format, an older version included, is not read.
pub const MARKER_OPEN: &str = "<!-- diffagent:v1 ";
const MARKER_CLOSE: &str = " -->";

/// The hidden marker carrying a Finding's Fingerprint, appended to its comment so a
/// retried job and the reply handler can recover which Finding a comment is.
pub fn finding_marker(fingerprint: &str) -> String {
    format!("{MARKER_OPEN}finding={fingerprint}{MARKER_CLOSE}")
}

/// The Fingerprint in a comment body's marker: hex, or `<hex>:<hex>` for a salted
/// co-located Finding. Unversioned markers are not recognized.
pub fn parse_finding_marker(body: &str) -> Option<&str> {
    let hex = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_hexdigit());
    parse_marker(body, "finding", |p| {
        let ok = match p.split_once(':') {
            Some((base, salt)) => hex(base) && hex(salt),
            None => hex(p),
        };
        ok.then_some(p)
    })
}

/// The hidden marker ending every reply we post in a Thread: the id of the reply it
/// answers. It is the reply's completion record, so a rerun that finds it posts nothing.
pub fn turn_marker(reply: i64) -> String {
    format!("{MARKER_OPEN}turn={reply}{MARKER_CLOSE}")
}

/// The reply id in a comment body's turn marker. Content-only: a human who pastes the
/// marker still parses here, so trust must come from the comment's author.
pub fn parse_turn_marker(body: &str) -> Option<i64> {
    parse_marker(body, "turn", |p| {
        // `parse` alone would accept a leading '+'.
        p.bytes()
            .all(|b| b.is_ascii_digit())
            .then(|| p.parse().ok())
            .flatten()
    })
}

/// The hidden marker of a Lesson a Thread reply proposes, which `learn` confirms. The
/// payload is base64, whose alphabet has no `-` or `>`, so no rule ends the comment
/// early. It goes right before the reply's turn marker.
pub fn proposal_marker(payload: &str) -> String {
    let encoded = STANDARD.encode(payload);
    format!("{MARKER_OPEN}proposal={encoded}{MARKER_CLOSE}")
}

/// The payload of the proposal marker right before the body's last marker, its turn
/// marker. Content-only, as [`parse_turn_marker`] is: trust comes from the comment's
/// author.
pub fn parse_proposal_marker(body: &str) -> Option<String> {
    let (before, _) = body.rsplit_once(MARKER_OPEN)?;
    parse_marker(before, "proposal", |p| {
        String::from_utf8(STANDARD.decode(p).ok()?).ok()
    })
}

/// The payload of the body's last marker when its key is `key` and `parse` accepts it.
/// Only the last is ours: the model's title, body and agent prompt come before it, so
/// a marker forged there (any key) is never read.
fn parse_marker<'a, T>(
    body: &'a str,
    key: &str,
    parse: impl Fn(&'a str) -> Option<T>,
) -> Option<T> {
    let (payload, _) = last_marker(body)?
        .strip_prefix(key)?
        .strip_prefix('=')?
        .split_once(MARKER_CLOSE)?;
    parse(payload)
}

/// What follows the body's last [`MARKER_OPEN`]: our marker, then what we appended.
fn last_marker(body: &str) -> Option<&str> {
    body.rsplit_once(MARKER_OPEN).map(|(_, rest)| rest)
}

/// Leads the closure note appended to a Finding's comment when a later commit fixes
/// it.
const FIXED: &str = "**Fixed.**";

/// The closure appended to a Finding's comment when a later commit fixes it.
pub fn addressed_note(commit_sha: &str) -> String {
    let short = commit_sha.get(..7).unwrap_or(commit_sha);
    format!("\n\n{FIXED} No longer found at commit `{short}`.")
}

/// The closure appended when a Maintainer resolves a Finding by command, with no
/// fixing commit to cite.
pub const RESOLVED_NOTE: &str = "\n\n**Resolved** by a Maintainer.";

/// Whether a comment body carries a closure note, i.e. belongs to a resolved Finding. Only text after our marker counts: one the model
/// wrote into the body is not a note. The dedup ignores markers on comments carrying
/// one, so a same-Fingerprint regression (a revert) is posted again instead of
/// swallowed as a duplicate.
pub fn is_addressed_closure(body: &str) -> bool {
    let leads = [FIXED, RESOLVED_NOTE.trim_start()];
    last_marker(body)
        .and_then(|m| m.split_once(MARKER_CLOSE))
        .is_some_and(|(_, after)| leads.iter().any(|lead| after.contains(lead)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finding::Category;

    const FP: &str = "0a1b2c3d4e5f60718293a4b5c6d7e8f9";

    fn finding(severity: Severity, title: &str, body: &str, suggestion: &str) -> Finding {
        Finding {
            path: String::new(),
            start_line: 0,
            end_line: 0,
            severity,
            category: Category::Logic,
            title: title.to_owned(),
            body: body.to_owned(),
            snippet: String::new(),
            suggestion: suggestion.to_owned(),
        }
    }

    /// `render_comment` with no symbol, blocking nothing.
    fn comment(f: &Finding, verified: bool, committable: bool, old: &[&str]) -> String {
        render_comment(f, FP, None, false, verified, committable, old)
    }

    fn assert_contains(out: &str, wants: &[&str]) {
        for want in wants {
            assert!(out.contains(want), "missing {want:?}\n---\n{out}");
        }
    }

    /// A body over GitHub's limit is cut on a char boundary, with the note, to fit.
    #[test]
    fn fit_comment_cuts_only_what_github_refuses() {
        // (name, body, cut)
        let cases = [
            ("short", "ok".to_owned(), false),
            ("at the limit", "a".repeat(MAX_COMMENT), false),
            ("one over", "a".repeat(MAX_COMMENT + 1), true),
            ("multibyte", "é".repeat(MAX_COMMENT), true),
        ];
        for (name, body, cut) in cases {
            let got = fit_comment(body.clone());
            assert!(got.len() <= MAX_COMMENT, "{name}: {}", got.len());
            assert_eq!(got.ends_with(CUT_NOTE), cut, "{name}");
            if !cut {
                assert_eq!(got, body, "{name}");
            }
        }
    }

    /// Our banners are adopted; a Walkthrough or an edited banner is not.
    #[test]
    fn banners_are_recognized() {
        let (reviewing, failed) = (reviewing_banner("diffagent"), failed_banner("diffagent"));
        let cases = [
            (reviewing.as_str(), true),
            (failed.as_str(), true),
            (
                "> [!NOTE]\n> **diffagent** is on it — analyzing the diff + cross-file context.",
                false,
            ),
            ("### diffagent review\n\n#### Summary\n\ns\n", false),
            ("", false),
        ];
        for (body, want) in cases {
            assert_eq!(is_banner("diffagent", body), want, "{body}");
        }
    }

    /// `BOT_NAME` heads the banners and the Walkthrough, without "review review"; only
    /// the configured name's banners are ours, and an "is on it" note never is one.
    #[test]
    fn banners_and_heading_carry_the_bot_name() {
        // (name, heading)
        let cases = [
            ("diffagent", "### diffagent review"),
            ("acme", "### acme review"),
            ("acme-review", "### acme-review"),
            ("Acme_REVIEW", "### Acme_REVIEW"),
            ("review", "### review"),
            ("reviewer", "### reviewer review"),
        ];
        for (name, heading) in cases {
            let reviewing = reviewing_banner(name);
            assert_eq!(
                reviewing,
                format!(
                    "{heading}\n\n> [!NOTE]\n> **Review in progress.** This comment becomes the \
                     review report when the review completes."
                ),
                "{name}"
            );
            let failed = failed_banner(name);
            assert_eq!(
                failed,
                format!(
                    "{heading}\n\n> [!WARNING]\n> **Review interrupted.** It could not complete \
                     and will retry."
                ),
                "{name}"
            );
            assert!(
                is_banner(name, &reviewing) && is_banner(name, &failed),
                "{name}"
            );
            let other = format!("{name}2");
            assert!(!is_banner(&other, &reviewing), "{name}");
            assert!(!is_banner(&other, &failed), "{name}");
            let on_it = format!(
                "> [!NOTE]\n> **{name}** is on it — analyzing the diff + cross-file context."
            );
            assert!(!is_banner(name, &on_it), "{name}");

            let w = Walkthrough {
                bot_name: name.to_owned(),
                pr: "acme/shop #42".to_owned(),
                ..Walkthrough::default()
            };
            assert_eq!(
                render_heading(&w, 0, 0),
                format!("{heading} · acme/shop #42\n"),
                "{name}"
            );
        }
    }

    /// A proposed fix over a whole new file would pass GitHub's comment limit: it is
    /// left out, and the comment keeps its prose, brief and marker.
    #[test]
    fn a_fix_too_long_for_a_comment_is_left_out() {
        let line = "x := computeTheValueForThisLine(input, options) // a long line";
        let old: Vec<&str> = vec![line; MAX_COMMENT / line.len() + 1];
        let f = finding(Severity::Major, "Wrong value", "It is wrong.", "x := 1");
        let out = comment(&f, true, false, &old);
        assert!(out.len() <= MAX_COMMENT, "{}", out.len());
        assert!(out.contains(FIX_LEFT_OUT) && !out.contains("Proposed fix"));
        assert!(out.contains("It is wrong.") && out.contains("Brief for a coding agent"));
        assert!(out.ends_with(&format!("\n{}\n", finding_marker(FP))));
    }

    #[test]
    fn committable_suggestion() {
        let mut f = finding(
            Severity::Major,
            "Missing nil check",
            "notificationRepo is not guarded.",
            "if notificationRepo == nil {\n\tpanic(\"notificationRepo is nil\")\n}",
        );
        (f.path, f.start_line, f.end_line) = ("app/x.go".to_owned(), 40, 42);
        let out = render_comment(
            &f,
            FP,
            Some("Wire"),
            false,
            true,
            true,
            &["old line a", "old line b"],
        );
        assert_contains(
            &out,
            &[
                "> [!WARNING]\n> **MAJOR** · logic · `Wire`\n#### Missing nil check\n\n",
                "notificationRepo is not guarded.",
                "<details open>\n<summary>Suggested change</summary>",
                "```suggestion\nif notificationRepo == nil {",
                "<summary>Brief for a coding agent</summary>",
                "app/x.go:40-42 (Wire) — Missing nil check. notificationRepo is not guarded.",
                "<sub>Disagree? Reply with your reason.",
            ],
        );
        // The one-click block replaces the advisory diff.
        assert!(
            !out.contains("Proposed fix") && !out.contains("-old line a"),
            "{out}"
        );
    }

    #[test]
    fn suggestion_render_modes() {
        let f = finding(Severity::Major, "t", "b", "x := 1");
        // (name, verified, committable, old, one-click, unmatched warning)
        let cases = [
            (
                "uncommittable is an illustration with a warning",
                false,
                false,
                &[][..],
                false,
                true,
            ),
            (
                "advisory mode never claims a verified location is unverified",
                true,
                false,
                &["x := 0"][..],
                false,
                false,
            ),
            (
                "committable posts the one-click block",
                true,
                true,
                &["x := 0"][..],
                true,
                false,
            ),
        ];
        for (name, verified, committable, old, one_click, warning) in cases {
            let out = comment(&f, verified, committable, old);
            assert_eq!(out.contains("```suggestion"), one_click, "{name}:\n{out}");
            assert_eq!(out.contains("Proposed fix"), !one_click, "{name}:\n{out}");
            assert!(out.contains("+x := 1") || one_click, "{name}:\n{out}");
            assert_eq!(
                out.contains("location not matched in the diff"),
                warning,
                "{name}:\n{out}"
            );
        }
    }

    #[test]
    fn no_op_suggestion_never_renders() {
        let f = finding(Severity::Major, "t", "b", "x := 0  \n");
        let out = comment(&f, true, true, &["x := 0"]);
        assert!(
            !out.contains("Proposed fix") && !out.contains("```suggestion"),
            "{out}"
        );
    }

    #[test]
    fn severity_styles() {
        // (severity, symbol, blocks, header, closing line)
        let cases = [
            (
                Severity::Critical,
                Some("Cart.Total"),
                true,
                "> [!CAUTION]\n> **CRITICAL** · logic · `Cart.Total` · blocks merge\n#### T\n",
                "A critical finding closes only when a push fixes it or a Maintainer \
                 comments `@diffagent resolve`.",
            ),
            (
                Severity::Critical,
                None,
                false,
                "> [!CAUTION]\n> **CRITICAL** · logic\n#### T\n",
                "closes only when a push fixes it",
            ),
            (
                Severity::Major,
                Some("Login"),
                false,
                "> [!WARNING]\n> **MAJOR** · logic · `Login`\n#### T\n",
                "Disagree? Reply with your reason. A Maintainer's reason becomes a Lesson \
                 for this repository.",
            ),
            (
                Severity::Minor,
                None,
                false,
                "**MINOR** · logic\n#### T\n",
                "Disagree?",
            ),
            (
                Severity::Nitpick,
                Some("a`b"),
                false,
                "**NIT** · logic · `ab`\n#### T\n",
                "Disagree?",
            ),
        ];
        for (sev, symbol, blocks, header, closing) in cases {
            let f = finding(sev, "T", "B", "");
            let out = render_comment(&f, FP, symbol, blocks, false, false, &[]);
            assert!(out.starts_with(header), "{sev:?}:\n{out}");
            assert!(out.contains("<sub>"), "{sev:?}:\n{out}");
            assert!(out.contains(closing), "{sev:?}:\n{out}");
            assert_eq!(
                out.contains("> [!"),
                sev >= Severity::Major,
                "{sev:?}:\n{out}"
            );
        }
    }

    #[test]
    fn multi_line_title_cannot_smuggle_a_suggestion() {
        let f = finding(Severity::Minor, "T\n```suggestion\nrm -rf /\n```", "B", "");
        let out = comment(&f, false, false, &[]);
        let header: Vec<&str> = out.lines().take(2).collect();
        assert_eq!(
            header,
            ["**MINOR** · logic", "#### T ```suggestion rm -rf / ```"]
        );
        // No fence opens before the details blocks, and the brief one-lines the title
        // too.
        let (body_section, _) = out.split_once("<details>").unwrap_or_default();
        assert!(
            body_section
                .lines()
                .all(|l| !l.trim_start().starts_with("```")),
            "{body_section}"
        );
    }

    #[test]
    fn agent_prompt_is_file_anchored() {
        // (path, start, end, symbol, title, body, brief)
        let cases = [
            (
                "a.go",
                5,
                5,
                None,
                "Style",
                "minor.",
                "a.go:5 — Style. minor.",
            ),
            (
                "app/h.go",
                10,
                12,
                Some("Handler.Serve"),
                "Nil deref",
                "guard x before use.",
                "app/h.go:10-12 (Handler.Serve) — Nil deref. guard x before use.",
            ),
            ("b.go", 0, 0, None, "Add test", "", "b.go — Add test."),
            ("", 0, 0, None, "T", "", "T."),
        ];
        for (path, start, end, symbol, title, body, want) in cases {
            let mut f = finding(Severity::Nitpick, title, body, "");
            (f.path, f.start_line, f.end_line) = (path.to_owned(), start, end);
            assert_eq!(agent_prompt(&anchored_loc(&f, symbol), &f), want);
            let out = render_comment(&f, FP, symbol, false, true, true, &[]);
            assert!(
                out.contains("Brief for a coding agent") && out.contains(want),
                "{out}"
            );
            assert!(
                !out.contains("```suggestion"),
                "no suggestion, no block:\n{out}"
            );
        }
    }

    #[test]
    fn code_fence_grows_past_backtick_runs() {
        assert_eq!(code_fence("plain code"), "```");
        assert_eq!(code_fence("doc := \"```mermaid\""), "````");
        assert_eq!(code_fence("a ````` b"), "``````");
    }

    #[test]
    fn heading_states_the_reviews_facts() {
        let w = Walkthrough {
            pr: "acme/shop #42".to_owned(),
            head: "a1b2c3d4e5f6".to_owned(),
            files_changed: 4,
            additions: 120,
            deletions: 8,
            effort_score: 3,
            effort_minutes: 20,
            ..Walkthrough::default()
        };
        assert_eq!(
            render_heading(&w, 3, 4),
            "### diffagent review · acme/shop #42\n\n`a1b2c3d` · 4 files · +120 −8 · review load \
             **medium**, about 20 min · 3 of 4 locations checked against the diff\n"
        );
        // (score, minutes, load)
        let loads = [
            (1, 0, "review load **trivial**"),
            (2, 5, "review load **light**, about 5 min"),
            (4, 0, "review load **heavy**"),
            (5, 90, "review load **very heavy**, about 90 min"),
        ];
        for (effort_score, effort_minutes, want) in loads {
            let w = Walkthrough {
                effort_score,
                effort_minutes,
                ..Walkthrough::default()
            };
            assert_eq!(
                render_heading(&w, 0, 0),
                format!("### diffagent review\n\n{want}\n")
            );
        }
        assert_eq!(
            render_heading(&Walkthrough::default(), 0, 0),
            "### diffagent review\n"
        );
        let one = Walkthrough {
            files_changed: 1,
            ..Walkthrough::default()
        };
        assert!(render_heading(&one, 0, 0).contains("1 file · +0 −0"));
    }

    #[test]
    fn walkthrough() {
        let w = Walkthrough {
            bullets: vec![BulletGroup {
                heading: "Bug Fixes".to_owned(),
                items: vec!["fixes nil deref".to_owned()],
            }],
            summary: "Adds admin force-cancel.".to_owned(),
            changes: vec![
                CohortRow {
                    cohort: "app".to_owned(),
                    files: vec!["app/a.go".to_owned(), "app/b.go".to_owned()],
                    summary: "force-cancel handler".to_owned(),
                },
                CohortRow {
                    cohort: "domain".to_owned(),
                    files: vec!["domain/o.go".to_owned()],
                    summary: "order state | split\nover lines".to_owned(),
                },
            ],
            addressed: vec!["Nil pointer when the cart is empty".to_owned()],
            ..Walkthrough::default()
        };
        let mut f = finding(Severity::Critical, "Token | logged\nin plain text", "b", "");
        (f.path, f.start_line) = ("api/auth.go".to_owned(), 42);
        let minor = Finding {
            severity: Severity::Minor,
            title: "Case-sensitive match".to_owned(),
            start_line: 9,
            ..f.clone()
        };
        let out = render_walkthrough(&w, &[(&f, Some("Login")), (&minor, None)]);
        assert_contains(
            &out,
            &[
                "\n#### Summary\n\nAdds admin force-cancel.\n",
                "\n#### Findings on this push\n\n| Level | Finding | Location |\n|---|---|---|\n\
                 | **CRITICAL** | Token \\| logged in plain text | `api/auth.go:42` · `Login` |\n\
                 | **MINOR** | Case-sensitive match | `api/auth.go:9` |\n",
                "\n#### Fixed since the last review\n\n- ✓ Nil pointer when the cart is empty\n",
                "\n#### Key changes\n\n**Bug Fixes**\n- fixes nil deref\n",
                "\n#### Files\n\n| Area | Files | What changed |\n|---|---|---|\n\
                 | app | `app/a.go`<br>`app/b.go` | force-cancel handler |\n\
                 | domain | `domain/o.go` | order state \\| split over lines |\n",
            ],
        );
        assert!(!out.contains("#### Flow"), "no diagram set:\n{out}");
        let sections: Vec<&str> = out.lines().filter(|l| l.starts_with("#### ")).collect();
        assert_eq!(
            sections,
            [
                "#### Summary",
                "#### Findings on this push",
                "#### Fixed since the last review",
                "#### Key changes",
                "#### Files"
            ]
        );

        let empty = render_walkthrough(&Walkthrough::default(), &[]);
        assert_eq!(empty, "\n#### Summary\n\n_No summary available._\n");
        let bullets_only = Walkthrough {
            summary: String::new(),
            ..w
        };
        assert!(!render_walkthrough(&bullets_only, &[]).contains("#### Summary"));
    }

    #[test]
    fn walkthrough_diagram() {
        let w = Walkthrough {
            summary: "s".to_owned(),
            diagram: "sequenceDiagram\n  A->>B: call\n  B-->>A: ok".to_owned(),
            ..Walkthrough::default()
        };
        assert_contains(
            &render_walkthrough(&w, &[]),
            &[
                "\n#### Flow\n\n",
                "```mermaid",
                "sequenceDiagram",
                "A->>B: call",
            ],
        );
        // A diagram holding a fence cannot close its block early.
        let w = Walkthrough {
            diagram: "sequenceDiagram\n  A->>B: ```done```".to_owned(),
            ..w
        };
        assert!(
            render_walkthrough(&w, &[])
                .ends_with("\n````mermaid\nsequenceDiagram\n  A->>B: ```done```\n````\n"),
            "{}",
            render_walkthrough(&w, &[])
        );
    }

    /// Each requirement is one table row, whatever newlines or pipes the model put in
    /// it; an empty Ticket fit shows nothing.
    #[test]
    fn walkthrough_ticket_fit() {
        let requirement = |text: &str, fit, note: &str, from_story| Requirement {
            text: text.to_owned(),
            fit,
            note: note.to_owned(),
            from_story,
        };
        let fit = TicketFit {
            key: "PROJ-124".to_owned(),
            story: Some("PROJ-123".to_owned()),
            context:
                "Customers confirm or cancel\nan order whose total changed; this task is the web dialog."
                    .to_owned(),
            requirements: vec![
                requirement("Show the dialog", Fit::Done, "form.tsx: ConfirmDialog", false),
                requirement("Buttons cannot be pressed twice", Fit::Missing, "", true),
                requirement("Both buttons\ndisable", Fit::Partial, "", false),
                requirement(
                    "Fire analytics",
                    Fit::Missing,
                    "no `confirm_*` | x\n\n- event",
                    false,
                ),
            ],
        };
        let w = Walkthrough {
            summary: "s".to_owned(),
            ticket: Some(fit.clone()),
            ..Walkthrough::default()
        };
        let want = "\n#### Ticket PROJ-124\n\n\
                    > Customers confirm or cancel an order whose total changed; this task is the web dialog.\n\n\
                    | Status | Requirement | Note |\n|---|---|---|\n\
                    | ✓ done | Show the dialog | form.tsx: ConfirmDialog |\n\
                    | ~ partial | Both buttons disable |  |\n\
                    | ✗ missing | Fire analytics | no `confirm_*` \\| x - event |\n\
                    \n**Story PROJ-123**\n\n\
                    | Status | Requirement | Note |\n|---|---|---|\n\
                    | ✗ missing | Buttons cannot be pressed twice |  |\n\
                    \n<sub>Checked by the model against this PR alone: work in another PR shows as \
                    missing. The ticket never blocks a merge.</sub>\n";
        let out = render_walkthrough(&w, &[]);
        assert!(out.contains(want), "{out}");

        let none = Walkthrough {
            ticket: Some(TicketFit {
                requirements: vec![],
                ..fit
            }),
            ..w
        };
        assert!(!render_walkthrough(&none, &[]).contains("#### Ticket"));
    }

    #[test]
    fn commands_name_who_may_send_each() {
        let out = render_commands();
        assert_contains(
            &out,
            &[
                "<summary>Commands</summary>",
                "| Comment | What it does | Who |",
                "| `@diffagent review` | Reviews the current head again, and resumes a paused PR \
                 | Author or Maintainer |",
                "| `@diffagent pause` |",
                "| `@diffagent resolve` | Closes every open finding | Maintainer |",
                "| `@diffagent lessons` |",
                "| `@diffagent forget <id>` |",
                "| `@diffagent learn` | In a finding's thread, stores the Lesson proposed there \
                 and suppresses the finding | Maintainer |",
                "| `@diffagent close` | In a finding's thread, closes that finding with no \
                 Lesson | Maintainer |",
            ],
        );
    }

    #[test]
    fn consolidated_agent_prompt() {
        let mut inline = finding(Severity::Major, "Nil deref", "guard x.", "");
        (inline.path, inline.start_line, inline.end_line) = ("a.go".to_owned(), 10, 12);
        let mut demoted = finding(Severity::Minor, "Add test", "cover the new path.", "");
        (demoted.path, demoted.start_line) = ("b.go".to_owned(), 30);
        let out = render_consolidated_agent_prompt(&[(&inline, Some("Run"))], &[&demoted]);
        assert_contains(
            &out,
            &[
                "<summary>Brief for coding agents (2 findings)</summary>",
                "```\nCheck each item against the code as it is now; change only what is still \
                 wrong.\n\nOn changed lines:\n1. a.go:10-12 (Run) — Nil deref. guard x.\n\n\
                 Elsewhere:\n2. b.go, around line 30 — Add test. cover the new path.\n```",
            ],
        );
        assert_eq!(render_consolidated_agent_prompt(&[], &[]), "");
        assert!(
            render_consolidated_agent_prompt(&[], &[&demoted]).contains("(1 finding)"),
            "singular"
        );

        // A body holding a fence cannot close the consolidated block early.
        let fenced = finding(Severity::Minor, "T", "see ```go\nx\n```", "");
        assert!(render_consolidated_agent_prompt(&[(&fenced, None)], &[]).contains("````\nCheck"));
    }

    #[test]
    fn sanitize_body_neutralizes_smuggled_suggestion() {
        let got = sanitize_body("Fix this.\n```suggestion\nrm -rf / (committed in one click)\n```");
        assert!(!got.contains("```suggestion"), "{got}");
        assert!(
            got.contains("rm -rf"),
            "content survives as a plain block:\n{got}"
        );
        assert_eq!(
            sanitize_body("uses `suggestion` inline"),
            "uses `suggestion` inline"
        );
        assert!(!sanitize_body("x\n  ``` suggestion\ny").contains("suggestion"));
    }

    /// Any fence GitHub opens as a suggestion: backticks or tildes, three or more, and
    /// an info string whose first word is `suggestion`.
    #[test]
    fn sanitize_body_neutralizes_every_suggestion_fence() {
        let cases = [
            ("x\n~~~suggestion\ny\n~~~", "x\n~~~\ny\n~~~"),
            ("x\n~~~~ suggestion\ny\n~~~~", "x\n~~~~ \ny\n~~~~"),
            ("x\n`````suggestion\ny\n`````", "x\n`````\ny\n`````"),
            ("x\n```suggestion title\ny\n```", "x\n``` title\ny\n```"),
            ("x\n  ```Suggestion\ny\n```", "x\n  ```\ny\n```"),
            // A lone CR ends a line, as does CRLF.
            ("x\r```suggestion\rEVIL\r```", "x\n```\nEVIL\n```"),
            ("x\r\n```suggestion\r\ny\r\n```", "x\n```\ny\n```"),
            // Inside a blockquote or a list item.
            ("> ```suggestion\n> EVIL\n> ```", "> ```\n> EVIL\n> ```"),
            (">```suggestion\n>EVIL\n>```", ">```\n>EVIL\n>```"),
            ("- ```suggestion\n  EVIL\n  ```", "- ```\n  EVIL\n  ```"),
            (
                "x\n2) ~~~ suggestion\n   y\n   ~~~",
                "x\n2) ~~~ \n   y\n   ~~~",
            ),
            (
                "> 1. * ```suggestion\n>      y\n>      ```",
                "> 1. * ```\n>      y\n>      ```",
            ),
            // An entity in the info string is decoded, so it can spell the word.
            ("```&#115;uggestion\nEVIL\n```", "```\nEVIL\n```"),
            ("x\n~~~ go &amp; more\ny\n~~~", "x\n~~~\ny\n~~~"),
            // Dropping the first word cannot leave another in its place.
            ("x\n```suggestion Suggestion\ny\n```", "x\n``` \ny\n```"),
            // Not a suggestion fence: left as written.
            ("x\n```suggestions\ny\n```", "x\n```suggestions\ny\n```"),
            ("x\n``suggestion\ny", "x\n``suggestion\ny"),
            (
                "x\n```rust\n// a suggestion\n```",
                "x\n```rust\n// a suggestion\n```",
            ),
            ("say ```suggestion``` inline", "say ```suggestion``` inline"),
            ("-```suggestion\ny", "-```suggestion\ny"),
            ("1.```suggestion\ny", "1.```suggestion\ny"),
        ];
        for (body, want) in cases {
            assert_eq!(sanitize_body(body), want, "{body:?}");
        }
    }

    #[test]
    fn backtick_body_cannot_break_agent_prompt_fence() {
        let f = finding(
            Severity::Minor,
            "T",
            "evil ``` body\n```suggestion\nx\n```",
            "",
        );
        let out = comment(&f, false, false, &[]);
        // Both copies of the body lose the suggestion, and the brief's fence outgrows
        // the body's runs.
        assert!(out.contains("````\nT."), "{out}");
        assert!(!out.contains("```suggestion"), "{out}");
    }

    /// The line that opens a top-level fence whose info string's first word is
    /// `suggestion`, walking `md` as CommonMark does for top-level fences and `<pre>`
    /// blocks: what GitHub would offer as a one-click change.
    fn exposed_suggestion(md: &str) -> Option<&str> {
        let (mut fence, mut pre) = (None::<(char, usize)>, false);
        for line in md.split('\n') {
            let lower = line.to_ascii_lowercase();
            if pre {
                pre = !lower.contains("</pre>");
                continue;
            }
            let t = line.trim_start_matches(' ');
            if line.len() - t.len() > 3 {
                continue;
            }
            let c = t.chars().next().filter(|c| matches!(c, '`' | '~'));
            let run = c.map_or(0, |c| t.len() - t.trim_start_matches(c).len());
            let info = c.map_or("", |c| t.trim_start_matches(c).trim());
            match (fence, c) {
                (Some((open, n)), Some(c)) if c == open && run >= n && info.is_empty() => {
                    fence = None;
                }
                (Some(_), _) => {}
                (None, Some(c)) if run >= 3 && !(c == '`' && info.contains('`')) => {
                    let word = info.split_whitespace().next().unwrap_or("");
                    if word.eq_ignore_ascii_case("suggestion") {
                        return Some(line);
                    }
                    fence = Some((c, run));
                }
                (None, _) => {
                    let after = lower.trim_start_matches(' ').strip_prefix("<pre");
                    if after.is_some_and(|a| a.is_empty() || a.starts_with([' ', '\t', '>'])) {
                        pre = !lower.contains("</pre>");
                    }
                }
            }
        }
        None
    }

    /// No copy of a Finding shows a ```suggestion block. The prose can leave a fence
    /// or a `<pre>` block open, which the brief's own lines then close, so the brief
    /// must hold none either; so must the consolidated brief, after whatever the
    /// Walkthrough leaves open before it.
    #[test]
    fn no_copy_exposes_a_suggestion() {
        // (title, body)
        let cases = [
            ("T", "x\n```suggestion\nEVIL\n```\n````"),
            ("T\n```suggestion\nEVIL\n```\nY", "x\n````"),
            ("T", "a</pre>\n```suggestion\nEVIL\n```\n<pre>"),
            ("T", "evil ``` body\n```suggestion\nx\n```"),
        ];
        for (title, body) in cases {
            let mut f = finding(Severity::Minor, title, body, "");
            (f.path, f.start_line, f.end_line) = ("a.go".to_owned(), 5, 5);
            let brief = render_consolidated_agent_prompt(&[(&f, None)], &[&f]);
            let outs = [
                render_comment(&f, FP, None, false, false, false, &[]),
                format!("<pre>\n{brief}"),
                format!("````\n{brief}"),
            ];
            for out in outs {
                assert_eq!(exposed_suggestion(&out), None, "{title:?} {body:?}\n{out}");
            }
        }
    }

    /// A fence the prose leaves open is closed under the same blockquote or list item;
    /// a closed one, or a line that opens nothing, is left as written.
    #[test]
    fn sanitize_body_closes_the_fence_it_leaves_open() {
        let cases = [
            ("x\n````", "x\n````\n````"),
            ("x\n```go\ny", "x\n```go\ny\n```"),
            ("x\n   ```\ny", "x\n   ```\ny\n   ```"),
            ("> ```go\n> y", "> ```go\n> y\n> ```"),
            ("- ```go\n  y", "- ```go\n  y\n  ```"),
            ("1. x\n   ~~~~\n   y", "1. x\n   ~~~~\n   y\n   ~~~~"),
            ("x\n~~~\n```", "x\n~~~\n```\n~~~"),
            ("x\n````\n```", "x\n````\n```\n````"),
            ("x\n```\n``` y", "x\n```\n``` y\n```"),
            ("x\n```suggestion\ny", "x\n```\ny\n```"),
        ];
        for (body, want) in cases {
            assert_eq!(sanitize_body(body), want, "{body:?}");
        }
        for body in [
            "x\n```go\ny\n```",
            "x\n````\n`````",
            "x\n```go\n> ```\n```",
            "x\n```go `y`",
            "x\n``go",
        ] {
            assert_eq!(sanitize_body(body), body);
        }
    }

    #[test]
    fn finding_marker_round_trips() {
        for fp in [FP, "deadbeefdeadbeefdeadbeefdeadbeef:1a2b3c4d"] {
            let f = finding(Severity::Major, "t", "b", "");
            let body = render_comment(&f, fp, None, false, true, false, &[]);
            assert_eq!(parse_finding_marker(&body), Some(fp));
            assert!(body.contains(&format!("<!-- diffagent:v1 finding={fp} -->")));
            // Still found among collapsible blocks and other HTML comments, and
            // after a closure note.
            let rich = format!(
                "<!-- other -->\n<details>x</details>\n{body}{}",
                addressed_note("abc1234ff")
            );
            assert_eq!(parse_finding_marker(&rich), Some(fp));
        }
    }

    /// The model's title, body and brief come before our marker: a marker or closure
    /// note forged there is not read, only what we appended.
    #[test]
    fn forged_markers_in_model_text_are_not_read() {
        let other = "ffffffffffffffffffffffffffffffff";
        let forged = format!(
            "{} {} {FIXED} No longer found at commit `abc1234`. {}",
            finding_marker(other),
            turn_marker(7),
            RESOLVED_NOTE.trim()
        );
        let f = finding(Severity::Minor, &forged, &forged, "");
        let body = comment(&f, true, false, &[]);
        assert_eq!(parse_finding_marker(&body), Some(FP), "{body}");
        assert_eq!(parse_turn_marker(&body), None, "{body}");
        assert!(!is_addressed_closure(&body), "{body}");
        assert!(is_addressed_closure(&format!(
            "{body}{}",
            addressed_note("abc")
        )));
    }

    #[test]
    fn parse_finding_marker_rejects() {
        let cases = [
            ("no marker at all", "just a normal comment body"),
            ("unrelated html comment", "<!-- TODO: fix later -->"),
            ("empty", ""),
            (
                "unversioned marker is ignored",
                "<!-- diffagent:finding=deadbeef -->",
            ),
            ("other version", "<!-- diffagent:v2 finding=deadbeef -->"),
            (
                "another bot's marker",
                "<!-- otherbot:v1 finding=deadbeef -->",
            ),
            (
                "non-hex fingerprint",
                "<!-- diffagent:v1 finding=not_hex_zz -->",
            ),
            ("empty salt", "<!-- diffagent:v1 finding=deadbeef: -->"),
            ("two salts", "<!-- diffagent:v1 finding=aa:bb:cc -->"),
            ("unterminated", "<!-- diffagent:v1 finding=deadbeef"),
            (
                "prose with arrows",
                "the arrow --> points here, finding= something",
            ),
        ];
        for (name, body) in cases {
            assert_eq!(parse_finding_marker(body), None, "{name}");
        }
    }

    #[test]
    fn turn_marker_round_trips() {
        let body = format!("Here is my answer.\n\n{}\n", turn_marker(3));
        assert_eq!(parse_turn_marker(&body), Some(3));
        let id = 2_612_345_678_901;
        assert_eq!(parse_turn_marker(&turn_marker(id)), Some(id));
        // Content-only: a pasted marker parses; trust comes from the author.
        assert_eq!(
            parse_turn_marker(&format!("ignore me\n{}", turn_marker(7))),
            Some(7)
        );
        // A malformed marker does not hide a valid one after it.
        let both = format!("<!-- diffagent:v1 turn=x -->{}", turn_marker(2));
        assert_eq!(parse_turn_marker(&both), Some(2));
    }

    /// A proposal's payload comes back whole, whatever it holds, from right before the
    /// turn marker, and only from there.
    #[test]
    fn proposal_marker_round_trips() {
        let payload = r#"{"rule":"a --> b <!-- diffagent:v1 turn=1 --> é\n"}"#;
        let marker = proposal_marker(payload);
        let encoded = &marker[MARKER_OPEN.len()..marker.len() - MARKER_CLOSE.len()];
        assert!(!encoded.contains(['-', '>']), "{marker}");
        let body = format!("The note.\n\n{marker}\n{}", turn_marker(9));
        assert_eq!(parse_proposal_marker(&body).as_deref(), Some(payload));
        assert_eq!(parse_turn_marker(&body), Some(9));
        let cases = [
            ("no turn marker after it", marker.clone()),
            (
                "after the turn marker",
                format!("{}\n{marker}", turn_marker(9)),
            ),
            (
                "not base64",
                format!("{MARKER_OPEN}proposal=*{MARKER_CLOSE}{}", turn_marker(9)),
            ),
            ("no proposal", format!("An answer.\n\n{}", turn_marker(9))),
        ];
        for (name, body) in cases {
            assert_eq!(parse_proposal_marker(&body), None, "{name}");
        }
    }

    #[test]
    fn parse_turn_marker_rejects() {
        let cases = [
            "just a normal comment body",
            "the arrow --> points here, turn= maybe but not a real marker -->",
            "<!-- diffagent:v1 turn= -->",
            "<!-- diffagent:v1 turn=abc -->",
            "<!-- diffagent:v1 turn=+3 -->",
            "<!-- diffagent:v1 turn=-1 -->",
            "<!-- diffagent:v1 turn=99999999999999999999999 -->",
            "<!-- diffagent:turn=3 -->",
            "<!-- otherbot:v1 turn=3 -->",
            "<!-- diff:turn=3 -->",
            "<!-- diffagent:v1 finding=deadbeef -->",
        ];
        for body in cases {
            assert_eq!(parse_turn_marker(body), None, "{body}");
        }
    }

    /// Both closure notes mark a resolved Finding's comment.
    #[test]
    fn addressed_closure() {
        let note = addressed_note("abc1234def567890");
        assert_eq!(note, "\n\n**Fixed.** No longer found at commit `abc1234`.");
        assert_eq!(
            addressed_note("abc"),
            "\n\n**Fixed.** No longer found at commit `abc`."
        );

        let fresh = format!("## bug\n\nbody\n{}", finding_marker(FP));
        for closure in [note.as_str(), RESOLVED_NOTE] {
            let resolved = format!("{fresh}{closure}");
            assert!(is_addressed_closure(&resolved), "{closure}");
            // Detection comes from the closure, not from the marker going away.
            assert_eq!(parse_finding_marker(&resolved), Some(FP));
        }
        assert!(!is_addressed_closure(RESOLVED_NOTE), "no marker of ours");
        assert!(!is_addressed_closure(&fresh));
    }
}
