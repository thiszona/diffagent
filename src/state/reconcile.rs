//! Reconcile: a new Review's Findings against the PR's open ones, by Fingerprint, so a
//! push re-posts only what is new and closes what was fixed. Pure; the caller persists.

use std::collections::HashSet;

use super::StoredFinding;
use crate::finding::Finding;

/// One Review's Findings bucketed against the PR's open ones.
#[derive(Debug, Default, PartialEq)]
pub struct Reconciliation<'a> {
    /// Not open on the PR: post them.
    pub new: Vec<(&'a str, &'a Finding)>,
    /// Open and found again: their comments stay as they are.
    pub reaffirmed: Vec<(&'a str, &'a Finding)>,
    /// Open, not found again, on a file this push did not review: absence proves
    /// nothing, so they stay open.
    pub carried: Vec<&'a StoredFinding>,
    /// Open and gone from a file this push reviewed: fixed.
    pub resolved: Vec<&'a StoredFinding>,
}

/// Buckets `fresh` (a Review's Findings with their Fingerprints, from
/// `identity::fingerprint_findings`) against `open` (the PR's open Findings).
/// `reviewed` says whether this push reviewed a path; a full Review passes `|_| true`.
/// A rename is reviewed under both its old and new path, so a Finding stored under the
/// old one can resolve.
///
/// `fresh` keeps its order and a repeated Fingerprint counts once. A suppressed Finding is
/// not open, so a re-detected one lands in `new`: the caller drops those it does not
/// reopen.
pub fn reconcile<'a>(
    fresh: &'a [(String, &'a Finding)],
    open: &'a [StoredFinding],
    reviewed: impl Fn(&str) -> bool,
) -> Reconciliation<'a> {
    let open_fps: HashSet<&str> = open.iter().map(|f| f.fingerprint.as_str()).collect();
    let mut seen = HashSet::new();
    let mut r = Reconciliation::default();
    for (fp, f) in fresh {
        if !seen.insert(fp.as_str()) {
            continue;
        }
        let bucket = if open_fps.contains(fp.as_str()) {
            &mut r.reaffirmed
        } else {
            &mut r.new
        };
        bucket.push((fp, *f));
    }
    for o in open {
        if seen.contains(o.fingerprint.as_str()) {
            continue;
        }
        if reviewed(&o.path) {
            r.resolved.push(o);
        } else {
            r.carried.push(o);
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finding::{Category, Severity};
    use crate::state::FindingStatus;

    fn fresh_finding() -> Finding {
        Finding {
            path: "a.go".into(),
            start_line: 1,
            end_line: 1,
            severity: Severity::Major,
            category: Category::Logic,
            title: "t".into(),
            body: String::new(),
            snippet: String::new(),
            suggestion: String::new(),
        }
    }

    fn open_finding(fp: &str, path: &str) -> StoredFinding {
        StoredFinding {
            repo_id: 1,
            pr: 1,
            fingerprint: fp.into(),
            path: path.into(),
            start_line: 1,
            end_line: 1,
            severity: Severity::Major,
            category: Category::Logic,
            title: fp.into(),
            comment_id: None,
            status: FindingStatus::Open,
            first_sha: "s1".into(),
            last_sha: "s1".into(),
        }
    }

    struct Case {
        name: &'static str,
        fresh: &'static [&'static str],
        /// (Fingerprint, path).
        open: &'static [(&'static str, &'static str)],
        /// `None` is a full Review.
        reviewed: Option<&'static [&'static str]>,
        /// Fingerprints per bucket: new, reaffirmed, carried, resolved.
        want: [&'static [&'static str]; 4],
    }

    #[test]
    fn buckets() {
        let cases = [
            Case {
                name: "first pass: all new",
                fresh: &["a", "b"],
                open: &[],
                reviewed: None,
                want: [&["a", "b"], &[], &[], &[]],
            },
            Case {
                name: "mixed",
                fresh: &["a", "c"],
                open: &[("a", "x.go"), ("b", "x.go")],
                reviewed: None,
                want: [&["c"], &["a"], &[], &["b"]],
            },
            Case {
                name: "nothing fresh: all resolved",
                fresh: &[],
                open: &[("a", "x.go"), ("b", "y.go")],
                reviewed: None,
                want: [&[], &[], &[], &["a", "b"]],
            },
            Case {
                name: "a repeated fingerprint counts once",
                fresh: &["a", "a"],
                open: &[],
                reviewed: None,
                want: [&["a"], &[], &[], &[]],
            },
            Case {
                name: "fresh order is kept",
                fresh: &["c", "a", "b"],
                open: &[("a", "x.go")],
                reviewed: None,
                want: [&["c", "b"], &["a"], &[], &[]],
            },
            Case {
                name: "unreviewed file: carried, not resolved",
                fresh: &[],
                open: &[("a", "touched.go"), ("b", "untouched.go")],
                reviewed: Some(&["touched.go"]),
                want: [&[], &[], &["b"], &["a"]],
            },
            Case {
                name: "found again on an unreviewed file: reaffirmed",
                fresh: &["b"],
                open: &[("b", "untouched.go")],
                reviewed: Some(&["touched.go"]),
                want: [&[], &["b"], &[], &[]],
            },
            Case {
                name: "rename in scope under its old name: resolved",
                fresh: &[],
                open: &[("x", "old_b.go")],
                reviewed: Some(&["new_b.go", "old_b.go"]),
                want: [&[], &[], &[], &["x"]],
            },
        ];
        let f = fresh_finding();
        for c in cases {
            let fresh: Vec<(String, &Finding)> =
                c.fresh.iter().map(|fp| ((*fp).to_owned(), &f)).collect();
            let open: Vec<StoredFinding> = c
                .open
                .iter()
                .map(|(fp, path)| open_finding(fp, path))
                .collect();
            let r = reconcile(&fresh, &open, |path| {
                c.reviewed.is_none_or(|paths| paths.contains(&path))
            });
            let got = [
                fresh_fps(&r.new),
                fresh_fps(&r.reaffirmed),
                stored_fps(&r.carried),
                stored_fps(&r.resolved),
            ];
            assert_eq!(got, c.want, "{}", c.name);
        }
    }

    fn fresh_fps<'a>(bucket: &[(&'a str, &Finding)]) -> Vec<&'a str> {
        bucket.iter().map(|(fp, _)| *fp).collect()
    }

    fn stored_fps<'a>(bucket: &[&'a StoredFinding]) -> Vec<&'a str> {
        bucket.iter().map(|o| o.fingerprint.as_str()).collect()
    }
}
