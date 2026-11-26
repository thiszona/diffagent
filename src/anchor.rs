//! Anchoring: where on the diff a Finding's inline comment goes, if anywhere.
//!
//! The model's line numbers are untrusted, so the finding's quoted snippet is checked
//! against the diff before a comment is placed: a comment on the wrong line is worse
//! than one folded into the Walkthrough. The tests below pin each rule
//! [`resolve_anchor`] lists, and the edge cases around them: ties, the added-line
//! preference, the committable span cap and out-of-range line numbers.

use crate::diff::{DiffIndex, MIN_SNIPPET_LEN, normalize_code};
use crate::finding::Finding;

/// How far to look for a nearby diff line when a snippet-less finding points just
/// outside the hunks (commonly the function header above them).
const SNAP_WINDOW: i64 = 3;

/// How far a snippet match may sit from the claimed start line and still be trusted;
/// further away means the snippet is generic boilerplate or the model is lost.
const RELOCATE_WINDOW: i64 = 20;

/// The widest span a one-click suggestion may replace: beyond this a wrong suggestion
/// does too much damage in one click.
const MAX_COMMITTABLE_SPAN: i64 = 20;

/// How a finding's anchor was resolved. A finding with no valid anchor is demoted to
/// the Walkthrough, which [`resolve_anchor`] reports as `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// On the claimed span.
    Inline,
    /// Moved to the line the snippet actually matches.
    Relocated,
    /// The nearest diff line to a snippet-less claim (unverified).
    Snapped,
}

/// A validated position for an inline comment on the finding's file, on the NEW side
/// of the diff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Anchor {
    /// The last line of the span (GitHub's `line`).
    pub line: i64,
    /// The first line of a multi-line span; `None` for a single line.
    pub start_line: Option<i64>,
    /// The finding's snippet matches the anchored content.
    pub verified: bool,
    /// A one-click suggestion can safely replace exactly this span.
    pub committable: bool,
}

/// Maps a finding's claimed line range onto the diff, using its snippet to check that
/// the lines really hold the code it is about:
///
/// - span in the diff and the snippet matches inside it: verified inline anchor
///   (committable when the finding carries a suggestion);
/// - the snippet matches a different line: verified single-line anchor there;
/// - no usable snippet, span in the diff: unverified inline anchor;
/// - no usable snippet, span just outside the diff: the nearest diff line;
/// - otherwise `None`: demote to the Walkthrough, never drop.
///
/// The asymmetry is deliberate: a finding WITH a snippet that matches nowhere in the
/// diff is treated as fabricated and demoted, because posting it on lines that merely
/// exist in the diff is exactly how comments land on the wrong code.
pub fn resolve_anchor(f: &Finding, idx: &DiffIndex) -> Option<(Outcome, Anchor)> {
    if f.start_line <= 0 {
        return None;
    }
    let (start, end) = (f.start_line, f.end_line.max(f.start_line));
    // Short-circuits at the first gap, so a hallucinated huge span costs no more
    // than the diff itself.
    let all_in = (start..=end).all(|n| idx.has(n));

    // Verification needs a snippet long enough to be unambiguous; "}" or
    // "return err" would "verify" half the file, so those take the membership-only
    // path instead.
    let snip = first_snippet_line(&f.snippet);
    if normalize_code(snip).len() >= MIN_SNIPPET_LEN {
        resolve_verified(f, snip, idx, start, end, all_in)
    } else {
        resolve_legacy(idx, start, end, all_in)
    }
}

/// The first non-blank line of the snippet: matching is per line, and models
/// sometimes quote the whole span despite being asked for one line.
fn first_snippet_line(snippet: &str) -> &str {
    snippet
        .split('\n')
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or_default()
}

/// Anchors a finding that quotes its flagged code, verifying or relocating it by
/// content.
fn resolve_verified(
    f: &Finding,
    snip: &str,
    idx: &DiffIndex,
    start: i64,
    end: i64,
    all_in: bool,
) -> Option<(Outcome, Anchor)> {
    match idx.match_in_span(start, end, snip) {
        Some(_) if all_in => Some((
            Outcome::Inline,
            Anchor {
                line: end,
                start_line: (end > start).then_some(start),
                verified: true,
                committable: !f.suggestion.trim().is_empty() && end - start <= MAX_COMMITTABLE_SPAN,
            },
        )),
        // The key line is real and in the diff, but the span leaks outside the
        // hunks: anchor on the verified line alone.
        Some(n) => Some(relocated(n)),
        None => relocate_by_snippet(snip, idx, start),
    }
}

/// The snippet matches nothing in the claimed span, so the line numbers are wrong:
/// find where the quoted code actually lives, within [`RELOCATE_WINDOW`] of the
/// claim and only when one match is strictly closest.
fn relocate_by_snippet(snip: &str, idx: &DiffIndex, start: i64) -> Option<(Outcome, Anchor)> {
    let mut matches = idx.find_snippet(snip);
    // Findings are about changed code: when both added and context lines match,
    // the added lines win.
    if matches.iter().any(|n| idx.is_added(*n)) {
        matches.retain(|n| idx.is_added(*n));
    }
    let dist = |n: i64| (n - start).abs();
    // Stable, so equally distant matches keep ascending order.
    matches.sort_by_key(|n| dist(*n));
    match matches.as_slice() {
        [best, rest @ ..]
            if dist(*best) <= RELOCATE_WINDOW
                && rest.first().is_none_or(|r| dist(*r) > dist(*best)) =>
        {
            Some(relocated(*best))
        }
        // No match (fabricated, or quoting unchanged code outside the diff), too
        // far, or a tie too ambiguous to pick from: never post it inline.
        _ => None,
    }
}

/// Anchors a finding without a usable snippet (a noncompliant model response):
/// membership checks only, never verified, never committable.
fn resolve_legacy(
    idx: &DiffIndex,
    start: i64,
    end: i64,
    all_in: bool,
) -> Option<(Outcome, Anchor)> {
    if all_in {
        let anchor = Anchor {
            line: end,
            start_line: (end > start).then_some(start),
            verified: false,
            committable: false,
        };
        return Some((Outcome::Inline, anchor));
    }
    // Prefer snapping near the START of the span, where the finding points.
    [start, end]
        .into_iter()
        .find_map(|n| idx.nearest(n, SNAP_WINDOW))
        .map(|near| {
            let anchor = Anchor {
                line: near,
                start_line: None,
                verified: false,
                committable: false,
            };
            (Outcome::Snapped, anchor)
        })
}

/// A verified single-line anchor at line `n`. The claimed span is discarded, so a
/// multi-line suggestion can no longer apply: never committable.
fn relocated(n: i64) -> (Outcome, Anchor) {
    let anchor = Anchor {
        line: n,
        start_line: None,
        verified: true,
        committable: false,
    };
    (Outcome::Relocated, anchor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finding::{Category, Severity};

    /// New-file lines 10-16 are addressable, with content long enough to relocate by.
    const VERIFY_PATCH: &str = "@@ -10,4 +10,7 @@ func Foo() {\n \ttotal := compute(alpha)\n \tbeta := total + 1\n+\tresult := fetchRemoteValue(ctx)\n+\tcache.Store(key, result)\n+\tnotifyListeners(result)\n \treturn total\n }";

    fn finding(start: i64, end: i64, snippet: &str, suggestion: &str) -> Finding {
        Finding {
            path: "f.go".to_owned(),
            start_line: start,
            end_line: end,
            severity: Severity::Major,
            category: Category::Logic,
            title: String::new(),
            body: String::new(),
            snippet: snippet.to_owned(),
            suggestion: suggestion.to_owned(),
        }
    }

    fn anchor(line: i64, start_line: Option<i64>, verified: bool, committable: bool) -> Anchor {
        Anchor {
            line,
            start_line,
            verified,
            committable,
        }
    }

    #[test]
    fn verified() {
        let idx = DiffIndex::parse(VERIFY_PATCH);
        let cases = [
            (
                "span in diff with matching snippet and suggestion is committable",
                finding(12, 13, "result := fetchRemoteValue(ctx)", "x"),
                Some((Outcome::Inline, anchor(13, Some(12), true, true))),
            ),
            (
                "snippet matches a different line: relocated there",
                finding(10, 10, "cache.Store(key, result)", ""),
                Some((Outcome::Relocated, anchor(13, None, true, false))),
            ),
            (
                "span leaks past the hunk but the snippet line is inside: relocated to it",
                finding(14, 18, "notifyListeners(result)", "x"),
                Some((Outcome::Relocated, anchor(14, None, true, false))),
            ),
            (
                "snippet matches nothing in the diff: demoted, not posted on wrong lines",
                finding(12, 12, "completelyFabricatedCall(foo)", ""),
                None,
            ),
            (
                "whitespace differences do not break verification",
                finding(13, 13, "cache.Store(key,  result)  ", ""),
                Some((Outcome::Inline, anchor(13, None, true, false))),
            ),
            (
                "a multi-line snippet is matched by its first non-blank line",
                finding(
                    12,
                    12,
                    "\n  result := fetchRemoteValue(ctx)\n  cache.Store(key, result)",
                    "",
                ),
                Some((Outcome::Inline, anchor(12, None, true, false))),
            ),
        ];
        for (name, f, want) in cases {
            assert_eq!(resolve_anchor(&f, &idx), want, "{name}");
        }
    }

    #[test]
    fn ambiguous_snippet() {
        // The same statement appears on lines 12 and 14.
        let idx = DiffIndex::parse(
            "@@ -10,3 +10,6 @@ func Foo() {\n \ttotal := compute(alpha)\n \tbeta := total + 1\n+\tcache.Store(key, result)\n+\tother := transform(beta)\n+\tcache.Store(key, result)\n \treturn total\n}",
        );
        let cases = [
            (
                "closest match wins",
                20,
                Some((Outcome::Relocated, anchor(14, None, true, false))),
            ),
            ("equidistant matches are too ambiguous: demoted", 13, None),
        ];
        for (name, start, want) in cases {
            let f = finding(start, start, "cache.Store(key, result)", "");
            assert_eq!(resolve_anchor(&f, &idx), want, "{name}");
        }
    }

    #[test]
    fn no_snippet_legacy() {
        let idx = DiffIndex::parse(VERIFY_PATCH);
        let cases = [
            (
                "span in diff anchors inline but is never committable",
                finding(12, 13, "", "x"),
                Some((Outcome::Inline, anchor(13, Some(12), false, false))),
            ),
            (
                "just outside snaps near the START of the span",
                finding(9, 25, "", "x"),
                Some((Outcome::Snapped, anchor(10, None, false, false))),
            ),
            ("far outside demoted", finding(99, 99, "", ""), None),
            ("file-level demoted", finding(0, 0, "", ""), None),
            (
                "short snippets are too generic to verify: membership only",
                finding(12, 12, "}", "x"),
                Some((Outcome::Inline, anchor(12, None, false, false))),
            ),
            (
                "an end line near i64::MAX still snaps by its start",
                finding(9, i64::MAX, "", ""),
                Some((Outcome::Snapped, anchor(10, None, false, false))),
            ),
            (
                "a start line near i64::MAX is demoted, not an overflow",
                finding(i64::MAX, i64::MAX, "", ""),
                None,
            ),
        ];
        for (name, f, want) in cases {
            assert_eq!(resolve_anchor(&f, &idx), want, "{name}");
        }
    }

    #[test]
    fn committable_span_cap() {
        // A verified span wider than MAX_COMMITTABLE_SPAN anchors but is not
        // committable.
        let patch = format!(
            "@@ -1,0 +1,30 @@\n{}",
            "+\tstatementNumber := value\n".repeat(30)
        );
        let idx = DiffIndex::parse(&patch);
        let f = finding(1, 30, "statementNumber := value", "x");
        assert_eq!(
            resolve_anchor(&f, &idx),
            Some((Outcome::Inline, anchor(30, Some(1), true, false)))
        );
    }

    #[test]
    fn huge_end_line_is_bounded() {
        let idx = DiffIndex::parse(VERIFY_PATCH);
        let f = finding(1, 2_000_000_000, "thisCodeExistsNowhereInTheDiff()", "");
        assert_eq!(resolve_anchor(&f, &idx), None);
    }

    #[test]
    fn far_unique_match_demotes() {
        // A unique match 100 lines from the claim: generic snippet or a lost model.
        let idx = DiffIndex::parse("@@ -1,1 +100,2 @@\n ctx line here\n+cache.Store(key, result)");
        let f = finding(1, 1, "cache.Store(key, result)", "");
        assert_eq!(resolve_anchor(&f, &idx), None);
    }

    #[test]
    fn ambiguous_prefers_added_line() {
        // The same statement is unchanged context (line 10) and added (line 12).
        let idx = DiffIndex::parse(
            "@@ -10,3 +10,4 @@\n \tcache.Store(key, result)\n \tbeta := total + 1\n+\tcache.Store(key, result)\n \treturn total",
        );
        let f = finding(11, 11, "cache.Store(key, result)", "");
        assert_eq!(
            resolve_anchor(&f, &idx),
            Some((Outcome::Relocated, anchor(12, None, true, false)))
        );
    }
}
