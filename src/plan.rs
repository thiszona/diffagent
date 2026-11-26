//! The post plan: every Finding of a Review resolved against its file's diff into
//! inline comments, plus the Walkthrough text that carries everything that could not
//! be anchored. Pure, so the whole review output is testable without GitHub.

use std::collections::{HashMap, HashSet};

use serde::Serialize;
use tree_sitter::Tree;

use crate::anchor::{Anchor, Outcome, resolve_anchor};
use crate::diff::DiffIndex;
use crate::finding::{Finding, Severity};
use crate::identity::enclosing_symbol;
use crate::job::mention;
use crate::lang::Lang;
use crate::render::{
    Walkthrough, fit_comment, is_no_op_suggestion, render_commands, render_comment,
    render_consolidated_agent_prompt, render_heading, render_walkthrough, severity_label,
};

/// One inline comment ready to post.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Comment {
    /// Where on the finding's file (`finding.path`) the comment goes.
    pub anchor: Anchor,
    pub body: String,
    /// The finding, with its lines moved to the anchor so every rendered location
    /// points at the verified position rather than the model's claim.
    pub finding: Finding,
    /// Links the posted comment back to the persisted Finding.
    pub fingerprint: String,
    /// The named function the anchored lines start in, when the file's source parsed.
    pub symbol: Option<String>,
}

/// How every finding's anchor resolved. Demoted and snapped rates are the live
/// wrong-line signal; the verified rate scores a model's snippet discipline.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct OutcomeStats {
    pub inline: usize,
    pub relocated: usize,
    pub snapped: usize,
    /// No valid anchor; folded into the Walkthrough.
    pub demoted: usize,
    /// Anchors whose snippet matched the diff content.
    pub verified: usize,
    /// Comments carrying a one-click suggestion.
    pub committable: usize,
}

impl OutcomeStats {
    /// The number of findings the stats cover.
    pub fn findings(&self) -> usize {
        self.inline + self.relocated + self.snapped + self.demoted
    }
}

/// A Review's output, ready for the GitHub adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    /// The Walkthrough comment, including every Finding that could not be anchored.
    pub walkthrough: String,
    /// Inline comments, ordered by path then line.
    pub comments: Vec<Comment>,
    pub stats: OutcomeStats,
}

/// Resolves every finding against its file's diff and assembles the plan.
///
/// `findings` pairs each Finding with its Fingerprint (as `identity` produces them);
/// `patches` maps a path to its unified diff, and `sources` to its new-file text,
/// which names the function each comment is in (a path missing from it names none).
/// `allow_committable` gates the one-click suggestion: when false every suggestion
/// renders as an advisory diff only. `advisory` holds the Fingerprints whose
/// suggestion `suggest::breaks_syntax` rejected: they keep the advisory diff and lose
/// the one-click block.
pub fn build_plan(
    walkthrough: &Walkthrough,
    findings: &[(String, &Finding)],
    patches: &HashMap<String, String>,
    sources: &HashMap<String, String>,
    allow_committable: bool,
    advisory: &HashSet<String>,
) -> Plan {
    let indexes: HashMap<&str, DiffIndex> = patches
        .iter()
        .map(|(path, patch)| (path.as_str(), DiffIndex::parse(patch)))
        .collect();

    let mut trees = HashMap::new();
    let mut comments = Vec::new();
    let mut demoted = Vec::new();
    let mut stats = OutcomeStats::default();
    for (fingerprint, f) in findings {
        // A file without a patch has no diff to anchor on: demoted like any other.
        let resolved = indexes
            .get(f.path.as_str())
            .and_then(|idx| resolve_anchor(f, idx).map(|r| (idx, r)));
        let Some((idx, (outcome, anchor))) = resolved else {
            stats.demoted += 1;
            demoted.push(*f);
            continue;
        };
        match outcome {
            Outcome::Inline => stats.inline += 1,
            Outcome::Relocated => stats.relocated += 1,
            Outcome::Snapped => stats.snapped += 1,
        }
        if anchor.verified {
            stats.verified += 1;
        }
        let mut finding = (*f).clone();
        finding.start_line = anchor.start_line.unwrap_or(anchor.line);
        finding.end_line = anchor.line;

        let old = span_content(idx, &anchor);
        let mut committable =
            anchor.committable && allow_committable && !advisory.contains(fingerprint);
        // A replacement identical to the current code is not a fix: it would render
        // a self-cancelling diff, or a one-click block proposing what the Maintainer
        // already wrote. Checked here, in advisory mode too, where the suggestion
        // syntax check is skipped.
        if is_no_op_suggestion(&finding.suggestion, &old) {
            finding.suggestion.clear();
            committable = false;
        }
        if committable {
            stats.committable += 1;
        }
        let symbol = symbol_at(&mut trees, sources, &finding.path, finding.start_line);
        // Only a critical Finding fails the merge gate.
        let blocks = walkthrough.merge_gate && finding.severity == Severity::Critical;
        let body = render_comment(
            &finding,
            fingerprint,
            symbol.as_deref(),
            blocks,
            anchor.verified,
            committable,
            &old,
        );
        comments.push(Comment {
            anchor,
            body,
            finding,
            fingerprint: fingerprint.clone(),
            symbol,
        });
    }

    // Before sorting: the consolidated brief lists findings in review order.
    let inline: Vec<(&Finding, Option<&str>)> = comments.iter().map(posted).collect();
    let prompt = render_consolidated_agent_prompt(&inline, &demoted);
    comments
        .sort_by(|a, b| (&a.finding.path, a.anchor.line).cmp(&(&b.finding.path, b.anchor.line)));
    // The table puts the worst first, each severity by location.
    let mut table: Vec<(&Finding, Option<&str>)> = comments.iter().map(posted).collect();
    table.sort_by_key(|(f, _)| std::cmp::Reverse(f.severity));

    let walkthrough = fit_comment(
        [
            render_heading(walkthrough, stats.verified, stats.findings()),
            render_status_note(walkthrough, findings.len()),
            render_walkthrough(walkthrough, &table),
            render_demoted(&demoted),
            render_commands(),
            prompt,
        ]
        .concat(),
    );
    Plan {
        walkthrough,
        comments,
        stats,
    }
}

/// A comment's Finding with the function it is in, as the Walkthrough lists it.
fn posted(c: &Comment) -> (&Finding, Option<&str>) {
    (&c.finding, c.symbol.as_deref())
}

/// The named function enclosing `line` of `path`'s source, each file parsed once into
/// `trees`; `None` without a source, a grammar, or an enclosing function.
fn symbol_at<'s>(
    trees: &mut HashMap<&'s str, Option<(Lang, Tree)>>,
    sources: &'s HashMap<String, String>,
    path: &str,
    line: i64,
) -> Option<String> {
    let (path, source) = sources.get_key_value(path)?;
    let parsed = trees.entry(path.as_str()).or_insert_with(|| {
        let lang = Lang::for_path(path)?;
        Some((lang, lang.parse(source)?))
    });
    let (lang, tree) = parsed.as_ref()?;
    enclosing_symbol(*lang, tree, source, line)
}

/// The anchored span's current content: the old side of a proposed-fix diff. Empty
/// when any line of the span is outside the diff: no trustworthy old side.
fn span_content<'a>(idx: &'a DiffIndex, a: &Anchor) -> Vec<&'a str> {
    (a.start_line.unwrap_or(a.line)..=a.line)
        .map(|n| idx.content(n))
        .collect::<Option<_>>()
        .unwrap_or_default()
}

/// The un-anchorable findings, collapsed under the Walkthrough so they are surfaced,
/// not lost. Their line numbers failed verification, so locations are labelled
/// approximate rather than stated as fact.
fn render_demoted(demoted: &[&Finding]) -> String {
    if demoted.is_empty() {
        return String::new();
    }
    let mut b = format!(
        "\n<details>\n<summary>Not on a changed line ({})</summary>\n\n",
        demoted.len()
    );
    for f in demoted {
        let loc = if f.start_line > 0 {
            format!("`{}`, around line {}", f.path, f.start_line)
        } else {
            format!("`{}`", f.path)
        };
        b.push_str(&format!(
            "- **{}** {loc} — {}. {}\n",
            severity_label(f.severity),
            f.title,
            f.body.trim()
        ));
    }
    b.push_str("\n</details>\n");
    b
}

/// The merge gate when it is blocked, then the Review's state: an explicit all-clear
/// when the Review found nothing, without which a clean PR's comment never says so. No
/// findings is NOT an all-clear when files failed to review or earlier findings are
/// still open: "good to merge" there would be a lie.
fn render_status_note(w: &Walkthrough, findings: usize) -> String {
    let mut b = String::new();
    if w.blockers > 0 {
        let noun = if w.blockers == 1 {
            "finding"
        } else {
            "findings"
        };
        b.push_str(&format!(
            "\n> [!CAUTION]\n> **Merge gate: blocked by {} critical {noun}.** Push a fix, or a \
             Maintainer who neither opened nor pushed to the PR comments `{} resolve`.\n",
            w.blockers,
            mention()
        ));
    }
    if w.failed_files > 0 {
        b.push_str(&format!(
            "\n> [!WARNING]\n> **Review incomplete.** {} file(s) could not be reviewed.\n",
            w.failed_files
        ));
    } else if findings == 0 && w.prior_open > 0 {
        b.push_str(&format!(
            "\n> [!NOTE]\n> **No new findings in this push.** {} earlier finding(s) are still \
             open.\n",
            w.prior_open
        ));
    } else if findings == 0 {
        b.push_str(
            "\n> [!TIP]\n> **No blocking findings.** Nothing in this push needs a change before \
             merge.\n",
        );
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finding::{Category, Severity};

    /// New-file lines 10-16 are addressable; line 12 is "\tresult := fetchRemoteValue(ctx)"
    /// and line 13 is "\tcache.Store(key, result)".
    const VERIFY_PATCH: &str = "@@ -10,4 +10,7 @@ func Foo() {\n \ttotal := compute(alpha)\n \tbeta := total + 1\n+\tresult := fetchRemoteValue(ctx)\n+\tcache.Store(key, result)\n+\tnotifyListeners(result)\n \treturn total\n }";

    fn finding(
        start: i64,
        end: i64,
        severity: Severity,
        title: &str,
        snippet: &str,
        suggestion: &str,
    ) -> Finding {
        Finding {
            path: "foo.go".to_owned(),
            start_line: start,
            end_line: end,
            severity,
            category: Category::Logic,
            title: title.to_owned(),
            body: "b".to_owned(),
            snippet: snippet.to_owned(),
            suggestion: suggestion.to_owned(),
        }
    }

    fn plan(w: &Walkthrough, findings: &[Finding], allow_committable: bool) -> Plan {
        let patches = HashMap::from([("foo.go".to_owned(), VERIFY_PATCH.to_owned())]);
        let input: Vec<(String, &Finding)> = findings
            .iter()
            .enumerate()
            .map(|(i, f)| (format!("{i:032x}"), f))
            .collect();
        build_plan(
            w,
            &input,
            &patches,
            &HashMap::new(),
            allow_committable,
            &HashSet::new(),
        )
    }

    fn assert_contains(out: &str, wants: &[&str]) {
        for want in wants {
            assert!(out.contains(want), "missing {want:?}\n---\n{out}");
        }
    }

    #[test]
    fn inline_snapped_and_demoted() {
        let w = Walkthrough {
            summary: "Summary text".to_owned(),
            ..Walkthrough::default()
        };
        let findings = [
            finding(
                12,
                13,
                Severity::Major,
                "inline",
                "result := fetchRemoteValue(ctx)",
                "fixed := fetchRemoteValue(ctx)\nguard(fixed)",
            ),
            finding(9, 9, Severity::Minor, "snapped", "", ""),
            finding(99, 99, Severity::Critical, "demoted-far", "", ""),
            finding(0, 0, Severity::Nitpick, "demoted-filelevel", "", ""),
        ];
        let p = plan(&w, &findings, true);
        assert_eq!(p.comments.len(), 2, "inline + snapped");

        // Only the verified finding carries a one-click suggestion.
        for c in &p.comments {
            assert_eq!(
                c.body.contains("```suggestion"),
                c.anchor.committable,
                "{}",
                c.body
            );
        }
        assert_eq!(
            p.comments.iter().filter(|c| c.anchor.committable).count(),
            1
        );
        // Sorted by line; each carries its own Fingerprint and marker.
        assert_eq!(p.comments[0].finding.title, "snapped");
        assert_eq!(p.comments[1].fingerprint, format!("{:032x}", 0));
        assert!(p.comments[1].body.contains(&format!("finding={:032x}", 0)));

        // Demoted findings are folded into the Walkthrough with approximate
        // locations: never presented as verified fact, never dropped.
        assert_contains(
            &p.walkthrough,
            &[
                "Not on a changed line (2)",
                "demoted-far",
                "demoted-filelevel",
                "around line 99",
            ],
        );
        assert!(!p.walkthrough.contains("foo.go:99"));
    }

    #[test]
    fn file_without_patch_is_demoted() {
        let mut f = finding(
            12,
            12,
            Severity::Major,
            "t",
            "result := fetchRemoteValue(ctx)",
            "",
        );
        f.path = "other.go".to_owned();
        let p = plan(&Walkthrough::default(), &[f], true);
        assert!(p.comments.is_empty());
        assert_eq!(p.stats.demoted, 1);
    }

    #[test]
    fn relocated_aligns_agent_prompt() {
        // Claims line 10 but quotes the code on line 13.
        let f = finding(
            10,
            10,
            Severity::Major,
            "wrong line claim",
            "cache.Store(key, result)",
            "",
        );
        let p = plan(&Walkthrough::default(), &[f], true);
        assert_eq!(p.comments.len(), 1);
        let c = &p.comments[0];
        assert!(c.anchor.line == 13 && c.anchor.verified, "{:?}", c.anchor);
        assert_eq!((c.finding.start_line, c.finding.end_line), (13, 13));
        assert!(
            c.body.contains("foo.go:13 — wrong line claim."),
            "{}",
            c.body
        );
        assert!(
            p.walkthrough.contains("1. foo.go:13 — wrong line claim."),
            "{}",
            p.walkthrough
        );
    }

    #[test]
    fn status_note() {
        let one = [finding(12, 12, Severity::Major, "t", "", "")];
        // (name, failed files, prior open, has a finding, wanted, unwanted)
        let cases = [
            (
                "clean review says so",
                0,
                0,
                false,
                Some("**No blocking findings.**"),
                "Review incomplete",
            ),
            (
                "findings suppress the all-clear",
                0,
                0,
                true,
                None,
                "No blocking findings",
            ),
            (
                "failed files are never an all-clear",
                2,
                0,
                false,
                Some("> [!WARNING]\n> **Review incomplete.** 2 file(s) could not be reviewed."),
                "No blocking findings",
            ),
            (
                "prior open findings suppress the all-clear",
                0,
                3,
                false,
                Some(
                    "> [!NOTE]\n> **No new findings in this push.** 3 earlier finding(s) are still open.",
                ),
                "No blocking findings",
            ),
        ];
        for (name, failed_files, prior_open, has_finding, want, unwanted) in cases {
            let w = Walkthrough {
                failed_files,
                prior_open,
                ..Walkthrough::default()
            };
            let findings: &[Finding] = if has_finding { &one } else { &[] };
            let out = plan(&w, findings, true).walkthrough;
            if let Some(want) = want {
                assert!(out.contains(want), "{name}:\n{out}");
            }
            assert!(!out.contains(unwanted), "{name}:\n{out}");
        }
    }

    #[test]
    fn no_findings() {
        let w = Walkthrough {
            summary: "s".to_owned(),
            ..Walkthrough::default()
        };
        let p = plan(&w, &[], true);
        assert!(p.comments.is_empty());
        assert!(!p.walkthrough.contains("Not on a changed line"));
        assert!(
            !p.walkthrough.contains("checked against the diff"),
            "no verification count"
        );
        assert!(!p.walkthrough.contains("Brief for coding agents"));
        // The command list is always appended.
        assert_contains(
            &p.walkthrough,
            &["<summary>Commands</summary>", "@diffagent review"],
        );
    }

    /// A Walkthrough over GitHub's comment limit is cut to fit, never refused.
    #[test]
    fn an_oversized_walkthrough_fits_a_comment() {
        let w = Walkthrough {
            summary: "s".repeat(crate::render::MAX_COMMENT),
            ..Walkthrough::default()
        };
        let p = plan(&w, &[], true);
        assert!(p.walkthrough.len() <= crate::render::MAX_COMMENT);
        assert!(p.walkthrough.contains("GitHub limits a comment"));
    }

    #[test]
    fn outcome_stats() {
        let findings = [
            // verified inline with suggestion: inline + verified + committable
            finding(
                12,
                12,
                Severity::Major,
                "a",
                "result := fetchRemoteValue(ctx)",
                "x",
            ),
            // snippet on the wrong line: relocated + verified
            finding(10, 10, Severity::Major, "c", "cache.Store(key, result)", ""),
            // no snippet, just outside: snapped
            finding(9, 9, Severity::Minor, "d", "", ""),
            // nowhere: demoted
            finding(99, 99, Severity::Nitpick, "e", "", ""),
        ];
        let p = plan(&Walkthrough::default(), &findings, true);
        let want = OutcomeStats {
            inline: 1,
            relocated: 1,
            snapped: 1,
            demoted: 1,
            verified: 2,
            committable: 1,
        };
        assert_eq!(p.stats, want);
        assert!(
            p.walkthrough
                .contains("2 of 4 locations checked against the diff"),
            "{}",
            p.walkthrough
        );
    }

    #[test]
    fn proposed_fix_diff() {
        let f = finding(
            12,
            12,
            Severity::Major,
            "a",
            "result := fetchRemoteValue(ctx)",
            "result, err := fetchRemoteValue(ctx)",
        );
        let advisory = plan(&Walkthrough::default(), std::slice::from_ref(&f), false);
        assert_eq!(advisory.comments.len(), 1);
        assert_contains(
            &advisory.comments[0].body,
            &[
                "<summary>Proposed fix (review before applying)</summary>",
                "-\tresult := fetchRemoteValue(ctx)", // real old content from the diff
                "+result, err := fetchRemoteValue(ctx)",
            ],
        );
        let p = plan(&Walkthrough::default(), &[f], true);
        assert_contains(
            &p.comments[0].body,
            &[
                "<details open>\n<summary>Suggested change</summary>",
                "```suggestion\nresult, err := fetchRemoteValue(ctx)",
            ],
        );
        assert!(!p.comments[0].body.contains("Proposed fix"));
        assert_eq!(p.comments[0].finding.title, "a");
    }

    #[test]
    fn no_op_suggestion_suppressed() {
        // (name, start, end, suggestion, is a no-op)
        let cases = [
            (
                "single-line identical",
                12,
                12,
                "\tresult := fetchRemoteValue(ctx)",
                true,
            ),
            (
                "identical modulo trailing whitespace",
                12,
                12,
                "\tresult := fetchRemoteValue(ctx)   ",
                true,
            ),
            (
                "multi-line span identical",
                12,
                13,
                "\tresult := fetchRemoteValue(ctx)\n\tcache.Store(key, result)",
                true,
            ),
            (
                "genuine change still renders",
                12,
                12,
                "\tresult, err := fetchRemoteValue(ctx)",
                false,
            ),
        ];
        for (name, start, end, suggestion, no_op) in cases {
            let f = finding(
                start,
                end,
                Severity::Major,
                "t",
                "result := fetchRemoteValue(ctx)",
                suggestion,
            );
            let p = plan(&Walkthrough::default(), &[f], true);
            assert_eq!(p.comments.len(), 1, "{name}");
            let c = &p.comments[0];
            assert_eq!(
                c.body.contains("Suggested change"),
                !no_op,
                "{name}:\n{}",
                c.body
            );
            assert_eq!(
                c.body.contains("```suggestion"),
                !no_op,
                "{name}:\n{}",
                c.body
            );
            assert_eq!(p.stats.committable, usize::from(!no_op), "{name}");
            assert_eq!(
                c.finding.suggestion.is_empty(),
                no_op,
                "{name}: cleared when a no-op"
            );
        }
    }

    #[test]
    fn advisory_mode_no_committable() {
        let f = finding(
            12,
            12,
            Severity::Major,
            "a",
            "result := fetchRemoteValue(ctx)",
            "result, err := fetchRemoteValue(ctx)",
        );
        let p = plan(&Walkthrough::default(), &[f], false);
        assert_eq!(p.comments.len(), 1);
        let body = &p.comments[0].body;
        assert!(body.contains("Proposed fix"), "{body}");
        assert!(
            !body.contains("```suggestion") && !body.contains("Suggested change"),
            "{body}"
        );
        assert_eq!(p.stats.committable, 0);
        // Committability is intrinsic to the anchor; advisory mode gates rendering.
        assert!(p.comments[0].anchor.committable);
    }

    /// A Fingerprint whose suggestion breaks the syntax posts the advisory diff
    /// only; its neighbour keeps the one-click block.
    #[test]
    fn advisory_fingerprint_posts_without_one_click() {
        let fix = |title: &str, snippet: &str, suggestion: &str| {
            finding(12, 12, Severity::Major, title, snippet, suggestion)
        };
        let findings = [
            fix(
                "broken",
                "result := fetchRemoteValue(ctx)",
                "result := fetch((",
            ),
            Finding {
                start_line: 13,
                end_line: 13,
                ..fix(
                    "fine",
                    "cache.Store(key, result)",
                    "cache.Store(key, result, ttl)",
                )
            },
        ];
        let patches = HashMap::from([("foo.go".to_owned(), VERIFY_PATCH.to_owned())]);
        let input = [
            ("aa".to_owned(), &findings[0]),
            ("bb".to_owned(), &findings[1]),
        ];
        let advisory = HashSet::from(["aa".to_owned()]);
        let p = build_plan(
            &Walkthrough::default(),
            &input,
            &patches,
            &HashMap::new(),
            true,
            &advisory,
        );
        let one_click: Vec<(&str, bool)> = p
            .comments
            .iter()
            .map(|c| (c.fingerprint.as_str(), c.body.contains("```suggestion")))
            .collect();
        assert_eq!(one_click, [("aa", false), ("bb", true)]);
        let broken = &p.comments[0].body;
        assert!(
            broken.contains("<summary>Proposed fix (review before applying)</summary>"),
            "{broken}"
        );
        assert!(!broken.contains("Suggested change"), "{broken}");
        assert_eq!(p.stats.committable, 1);
        assert!(
            p.comments[0].anchor.committable,
            "the anchor still could be"
        );
    }

    /// Lines 1-16 of `foo.go`, with lines 3-16 inside `Foo`: the patch's new side.
    fn foo_source() -> String {
        let body = "\tx := 1\n".repeat(12);
        format!("package p\n\nfunc Foo() {{\n{body}}}\n")
    }

    /// A comment names the function its anchored lines are in, in its header, its brief
    /// and the Walkthrough's table; a file with no source, or no function there, names none.
    #[test]
    fn comments_name_their_function() {
        let f = finding(
            12,
            12,
            Severity::Major,
            "a",
            "result := fetchRemoteValue(ctx)",
            "",
        );
        let patches = HashMap::from([("foo.go".to_owned(), VERIFY_PATCH.to_owned())]);
        let input = [("aa".to_owned(), &f)];
        // (name, sources, symbol)
        let cases = [
            (
                "parsed",
                HashMap::from([("foo.go".to_owned(), foo_source())]),
                Some("Foo"),
            ),
            ("no source", HashMap::new(), None),
            (
                "no function there",
                HashMap::from([("foo.go".to_owned(), "\u{0}".repeat(3))]),
                None,
            ),
        ];
        for (name, sources, want) in cases {
            let p = build_plan(
                &Walkthrough::default(),
                &input,
                &patches,
                &sources,
                true,
                &HashSet::new(),
            );
            let c = &p.comments[0];
            assert_eq!(c.symbol.as_deref(), want, "{name}");
            let Some(sym) = want else {
                assert!(c.body.contains("foo.go:12 — a."), "{name}: {}", c.body);
                continue;
            };
            assert_contains(
                &c.body,
                &[
                    &format!("> **MAJOR** · logic · `{sym}`\n"),
                    &format!("foo.go:12 ({sym}) — a."),
                ],
            );
            assert_contains(
                &p.walkthrough,
                &[
                    &format!("| **MAJOR** | a | `foo.go:12` · `{sym}` |"),
                    &format!("1. foo.go:12 ({sym}) — a."),
                ],
            );
        }
    }

    /// Only a critical Finding says it blocks, and only with the merge gate on; the
    /// gate's alert shows only when the worker counted blockers.
    #[test]
    fn the_merge_gate_is_claimed_only_when_known() {
        let findings = [
            finding(
                12,
                12,
                Severity::Critical,
                "c",
                "result := fetchRemoteValue(ctx)",
                "",
            ),
            finding(13, 13, Severity::Major, "m", "cache.Store(key, result)", ""),
        ];
        // (name, merge gate, blockers, critical blocks, alert)
        let cases = [
            ("gate off", false, 0, false, None),
            ("gate on, none counted", true, 0, true, None),
            (
                "gate on, one blocker",
                true,
                1,
                true,
                Some("> [!CAUTION]\n> **Merge gate: blocked by 1 critical finding.** Push a fix"),
            ),
            (
                "gate on, two blockers",
                true,
                2,
                true,
                Some("**Merge gate: blocked by 2 critical findings.**"),
            ),
        ];
        for (name, merge_gate, blockers, blocks, alert) in cases {
            let w = Walkthrough {
                merge_gate,
                blockers,
                ..Walkthrough::default()
            };
            let p = plan(&w, &findings, true);
            let body = |title: &str| {
                let c = p.comments.iter().find(|c| c.finding.title == title);
                c.map(|c| c.body.as_str()).unwrap_or_default()
            };
            assert_eq!(body("c").contains("· blocks merge"), blocks, "{name}");
            assert!(!body("m").contains("blocks merge"), "{name}");
            assert_eq!(
                p.walkthrough.contains("Merge gate"),
                alert.is_some(),
                "{name}"
            );
            if let Some(alert) = alert {
                assert!(p.walkthrough.contains(alert), "{name}:\n{}", p.walkthrough);
            }
        }
    }

    /// The Walkthrough reads top to bottom: heading and facts, state, sections, then
    /// the collapsed blocks.
    #[test]
    fn walkthrough_order() {
        let w = Walkthrough {
            summary: "s".to_owned(),
            pr: "acme/shop #42".to_owned(),
            head: "a1b2c3d4".to_owned(),
            failed_files: 1,
            ..Walkthrough::default()
        };
        let findings = [
            finding(
                12,
                12,
                Severity::Minor,
                "i",
                "result := fetchRemoteValue(ctx)",
                "",
            ),
            finding(99, 99, Severity::Major, "d", "", ""),
        ];
        let out = plan(&w, &findings, true).walkthrough;
        assert!(
            out.starts_with("### diffagent review · acme/shop #42\n\n`a1b2c3d` · "),
            "{out}"
        );
        let order = [
            "1 of 2 locations checked against the diff",
            "> [!WARNING]",
            "#### Summary",
            "#### Findings on this push",
            "<summary>Not on a changed line (1)</summary>",
            "<summary>Commands</summary>",
            "<summary>Brief for coding agents (2 findings)</summary>",
        ];
        let at: Vec<usize> = order
            .iter()
            .map(|s| out.find(s).unwrap_or(usize::MAX))
            .collect();
        assert!(at.windows(2).all(|w| w[0] < w[1]), "{at:?}\n{out}");
        assert!(
            out.contains("- **MAJOR** `foo.go`, around line 99 — d. b\n"),
            "{out}"
        );
    }
}
