//! Unified-diff parsing: the one place that maps a GitHub `patch` to NEW-file line
//! numbers. A blank context line still counts, or every line after it is off by one.
//! Everything that needs diff lines goes through [`DiffIndex`], so that rule lives in
//! one place.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// Snippets shorter than this (normalized) are too generic ("}", "return err") to
/// relocate by, or to match as a substring of a longer line.
pub const MIN_SNIPPET_LEN: usize = 12;

/// One addressable NEW-file line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Line {
    /// The line's text with the diff marker stripped.
    pub content: String,
    /// Whether this diff added the line (as opposed to unchanged context).
    pub added: bool,
    norm: String,
}

/// One file of a pull request, in the shape GitHub's files API and the eval
/// fixtures' `files` use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct FileChange {
    pub path: String,
    pub status: FileStatus,
    /// Empty when GitHub sent none (binary or very large files).
    #[serde(default)]
    pub patch: String,
    /// The path before a rename (GitHub's `previous_filename`): an incremental
    /// Review counts it as reviewed, so the old path's open Findings resolve.
    #[serde(default)]
    pub previous: Option<String>,
}

/// GitHub's file status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FileStatus {
    Added,
    Removed,
    Modified,
    Renamed,
    Copied,
    Changed,
    Unchanged,
}

/// The NEW-file lines a comment can be posted on (added and context lines), with
/// their content, so a finding's quoted snippet can be checked against the line it
/// claims to be about.
#[derive(Debug, Clone, Default)]
pub struct DiffIndex {
    right: BTreeMap<i64, Line>,
    /// Where each run of removed lines sat; see [`DiffIndex::changed`].
    removed: BTreeSet<i64>,
}

impl DiffIndex {
    /// Builds the index from one file's unified-diff hunks.
    pub fn parse(patch: &str) -> Self {
        let mut idx = Self::default();
        let mut next: Option<i64> = None;
        // A removed run seen and not yet placed on a new-file line.
        let mut gap = false;
        for line in patch.lines() {
            if let Some(start) = hunk_new_start(line) {
                idx.close_gap(&mut gap, next);
                next = Some(start);
                continue;
            }
            let Some(n) = next.as_mut() else { continue };
            let (content, added) = match line.as_bytes().first() {
                // GitHub strips the single leading space from a blank context line,
                // but it is still a real new-file line.
                None => ("", false),
                Some(b'+') => (&line[1..], true),
                Some(b' ') => (&line[1..], false),
                // Removed lines don't exist in the new file; only where they sat does.
                Some(b'-') => {
                    gap = true;
                    continue;
                }
                // "\ No newline at end of file" is not a line either.
                Some(b'\\') => continue,
                // Be lenient: an unknown leading character is treated as context.
                Some(_) => (line, false),
            };
            if std::mem::take(&mut gap) && *n > 0 {
                idx.removed.insert(*n);
            }
            idx.add(*n, content, added);
            *n += 1;
        }
        idx.close_gap(&mut gap, next);
        idx
    }

    /// Places a removed run that ended its hunk on the line before it: nothing
    /// follows it in the hunk, so it was at the end of the file.
    fn close_gap(&mut self, gap: &mut bool, next: Option<i64>) {
        if std::mem::take(gap)
            && let Some(n) = next.filter(|n| *n > 1)
        {
            self.removed.insert(n - 1);
        }
    }

    fn add(&mut self, n: i64, content: &str, added: bool) {
        if n > 0 {
            let line = Line {
                content: content.to_owned(),
                added,
                norm: normalize_code(content),
            };
            self.right.insert(n, line);
        }
    }

    /// The addressable lines in ascending order.
    pub fn lines(&self) -> impl Iterator<Item = (i64, &Line)> {
        self.right.iter().map(|(n, l)| (*n, l))
    }

    /// Whether new-file line `n` is addressable.
    pub fn has(&self, n: i64) -> bool {
        self.right.contains_key(&n)
    }

    /// The content of new-file line `n`, if it is addressable.
    pub fn content(&self, n: i64) -> Option<&str> {
        self.right.get(&n).map(|l| l.content.as_str())
    }

    /// Whether new-file line `n` was added by this diff.
    pub fn is_added(&self, n: i64) -> bool {
        self.right.get(&n).is_some_and(|l| l.added)
    }

    /// The new-file lines this diff changed: every added line, plus where each run
    /// of removed lines sat (the line now in its place, or the line before it at
    /// the end of a file). `parse` keeps no removed lines, so without this a
    /// declaration changed only by a deletion would never count as changed.
    pub fn changed(&self) -> BTreeSet<i64> {
        self.right
            .iter()
            .filter(|(_, l)| l.added)
            .map(|(n, _)| *n)
            .chain(self.removed.iter().copied())
            .collect()
    }

    /// The content of the addressable lines in `start..=end`, in order. Lines of the
    /// range that are not in the diff are skipped.
    pub fn new_side(&self, start: i64, end: i64) -> Vec<&str> {
        if start > end {
            return Vec::new();
        }
        self.right
            .range(start..=end)
            .map(|(_, l)| l.content.as_str())
            .collect()
    }

    /// The closest addressable line to `n` within `window`: the line itself, then
    /// closer lines, earlier before later at equal distance. `n` comes from the model,
    /// so it may be absurd; the arithmetic saturates rather than overflowing.
    pub fn nearest(&self, n: i64, window: i64) -> Option<i64> {
        (0..=window).find_map(|w| {
            [n.saturating_sub(w), n.saturating_add(w)]
                .into_iter()
                .find(|c| self.has(*c))
        })
    }

    /// The first line in `start..=end` matching the snippet. Walks the index, not the
    /// range, so a hallucinated huge range costs no more than the diff itself.
    pub fn match_in_span(&self, start: i64, end: i64, snippet: &str) -> Option<i64> {
        if start > end {
            return None;
        }
        let want = normalize_code(snippet);
        self.right
            .range(start..=end)
            .find(|(_, l)| code_match(&l.norm, &want))
            .map(|(n, _)| *n)
    }

    /// Every line matching the snippet, ascending. Snippets shorter than
    /// [`MIN_SNIPPET_LEN`] match nothing: too ambiguous to relocate by.
    pub fn find_snippet(&self, snippet: &str) -> Vec<i64> {
        let want = normalize_code(snippet);
        if want.len() < MIN_SNIPPET_LEN {
            return Vec::new();
        }
        self.right
            .iter()
            .filter(|(_, l)| code_match(&l.norm, &want))
            .map(|(n, _)| *n)
            .collect()
    }
}

/// Collapses whitespace runs to single spaces and trims, so indentation and
/// tab/space differences never break a match.
pub fn normalize_code(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Whether normalized snippet `want` matches normalized line `got`: exactly, or as a
/// substring when the snippet is long enough to be unambiguous (the model quoted
/// part of a long line). Never the reverse: a short line inside a long snippet would
/// let generic fragments "verify" unrelated lines.
fn code_match(got: &str, want: &str) -> bool {
    !got.is_empty()
        && !want.is_empty()
        && (got == want || (want.len() >= MIN_SNIPPET_LEN && got.contains(want)))
}

/// The NEW-file start line of a hunk header `@@ -a[,b] +c[,d] @@`.
pub fn hunk_new_start(line: &str) -> Option<i64> {
    let rest = line.strip_prefix("@@ -")?;
    let (old, rest) = rest.split_once(" +")?;
    let (new, _) = rest.split_once(" @@")?;
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let count_ok = |r: Option<(&str, &str)>, whole: &str| match r {
        Some((start, count)) => digits(start) && digits(count),
        None => digits(whole),
    };
    if !count_ok(old.split_once(','), old) || !count_ok(new.split_once(','), new) {
        return None;
    }
    new.split(',').next()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    // New-file lines 10-16 are addressable (context + added).
    const SAMPLE: &str = "@@ -10,4 +10,7 @@ func Foo() {\n \ta := 1\n \tb := 2\n+\tc := 3\n+\td := 4\n+\te := 5\n \treturn a\n }";

    #[test]
    fn addressable_lines() {
        let d = DiffIndex::parse(SAMPLE);
        for n in 10..=16 {
            assert!(d.has(n), "line {n} should be addressable");
        }
        for n in [9, 17, 0, 100] {
            assert!(!d.has(n), "line {n} should not be addressable");
        }
        assert_eq!(d.content(12), Some("\tc := 3"));
        assert_eq!(d.content(10), Some("\ta := 1"));
        assert_eq!(d.content(9), None);
        assert!(d.is_added(12) && !d.is_added(10));
    }

    #[test]
    fn blank_context_line_advances() {
        // GitHub sends a blank context line as "", which is still line 2.
        let d = DiffIndex::parse("@@ -1,4 +1,5 @@\n line one text\n\n+added line text\n line four");
        assert!(d.has(2), "the blank line itself is addressable");
        assert_eq!(d.content(3), Some("added line text"));
        assert_eq!(d.content(4), Some("line four"));
        assert_eq!(d.new_side(2, 3), vec!["", "added line text"]);
    }

    #[test]
    fn hunk_headers() {
        let cases = [
            ("@@ -10,4 +10,7 @@ func Foo() {", Some(10)),
            ("@@ -5 +5 @@", Some(5)),
            ("@@ -1,0 +1 @@", Some(1)),
            ("@@ -a,4 +10,7 @@", None),
            ("@@@ -1,2 -1,2 +1,3 @@@", None),
            (" @@ -1 +1 @@", None),
        ];
        for (header, want) in cases {
            assert_eq!(hunk_new_start(header), want, "{header}");
        }
    }

    #[test]
    fn nearest() {
        let d = DiffIndex::parse(SAMPLE);
        assert_eq!(d.nearest(12, 3), Some(12));
        assert_eq!(d.nearest(9, 3), Some(10), "snap up");
        assert_eq!(d.nearest(18, 3), Some(16), "snap down");
        assert_eq!(d.nearest(100, 3), None);
    }

    #[test]
    fn find_snippet() {
        let d = DiffIndex::parse(
            "@@ -1,2 +1,5 @@\n first context line here\n+result := fetchRemoteValue(ctx)\n+short\n+result := fetchRemoteValue(ctx)\n last context line here",
        );
        assert_eq!(
            d.find_snippet("result := fetchRemoteValue(ctx)"),
            vec![2, 4]
        );
        assert!(
            d.find_snippet("short").is_empty(),
            "too short to relocate by"
        );
        assert!(d.find_snippet("nothingLikeThisExistsAnywhere()").is_empty());
        assert_eq!(
            d.match_in_span(1, 1_000_000_000, "result := fetchRemoteValue(ctx)"),
            Some(2)
        );
    }

    #[test]
    fn changed_lines() {
        let cases = [
            ("added lines", "@@ -1,2 +1,3 @@\n a\n+b\n c", vec![2]),
            (
                "deletion-only hunk: the line now in its place",
                "@@ -1,4 +1,3 @@\n a\n-b\n c\n d",
                vec![2],
            ),
            (
                "a removed run counts once",
                "@@ -1,5 +1,3 @@\n a\n-b\n-c\n d\n e",
                vec![2],
            ),
            ("replaced line", "@@ -1,3 +1,3 @@\n a\n-b\n+B\n c", vec![2]),
            (
                "end of file: the line before",
                "@@ -8,4 +8,3 @@\n a\n b\n c\n-d",
                vec![10],
            ),
            (
                "end of file without a newline",
                "@@ -1,2 +1,1 @@\n a\n-b\n\\ No newline at end of file",
                vec![1],
            ),
            (
                "end of a hunk before the next",
                "@@ -1,3 +1,2 @@\n a\n b\n-c\n@@ -20,2 +19,3 @@\n x\n+y\n z",
                vec![2, 20],
            ),
            (
                "after a blank context line",
                "@@ -1,4 +1,3 @@\n x\n\n-y\n z",
                vec![3],
            ),
            ("only context", "@@ -1,2 +1,2 @@\n a\n b", vec![]),
        ];
        for (name, patch, want) in cases {
            let got: Vec<i64> = DiffIndex::parse(patch).changed().into_iter().collect();
            assert_eq!(got, want, "{name}");
        }
    }

    #[test]
    fn file_change_shape() {
        let got: Vec<FileChange> = serde_json::from_str(
            r#"[{"Path": "a.png", "Status": "removed"},
                {"Path": "b.go", "Status": "renamed", "Patch": "@@ -1 +1 @@\n-x\n+y", "Additions": 1}]"#,
        )
        .unwrap();
        assert_eq!(got[0].status, FileStatus::Removed);
        assert_eq!(got[0].patch, "", "a missing patch is empty");
        assert_eq!(got[1].status, FileStatus::Renamed);
        let statuses: Vec<FileStatus> = serde_json::from_str(
            r#"["added","removed","modified","renamed","copied","changed","unchanged"]"#,
        )
        .unwrap();
        assert_eq!(statuses.len(), 7);
    }

    #[test]
    fn previous_survives_serde() {
        let got: Vec<FileChange> = serde_json::from_str(
            r#"[{"Path": "b.go", "Status": "renamed", "Previous": "a.go"},
                {"Path": "c.go", "Status": "modified"}]"#,
        )
        .unwrap();
        assert_eq!(got[0].previous.as_deref(), Some("a.go"));
        assert_eq!(got[1].previous, None, "a missing previous is None");
        let back: Vec<FileChange> =
            serde_json::from_str(&serde_json::to_string(&got).unwrap()).unwrap();
        assert_eq!(back, got);
    }
}
