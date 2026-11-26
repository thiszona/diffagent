//! The gitleaks scanner and how its hits become findings.
//!
//! Every file a PR controls that could tell gitleaks to look away is ignored: the
//! tree's `.gitleaks.toml` and `.gitleaksignore` are deleted first, gitleaks runs
//! from outside the tree so its default `--gitleaks-ignore-path .` sees no PR file,
//! and `--ignore-gitleaks-allow` drops inline `gitleaks:allow` comments. Secret
//! values never leave gitleaks: it redacts its report, and only the rule, the
//! description and the location are read from it.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs;
use std::io::ErrorKind;
use std::path::Path;
use std::time::Duration;

use serde::Deserialize;

use crate::context::ContextPack;
use crate::diff::{DiffIndex, hunk_new_start};
use crate::finding::{Category, Finding, Severity};
use crate::proc::{self, Cmd, ProcError};
use crate::workspace;

/// Bounds one gitleaks run, so a hung scanner can't hold the Review.
const TIMEOUT: Duration = Duration::from_secs(90);

/// The report, written to the checkout's directory, outside the tree.
const REPORT: &str = "gitleaks.json";

const REPORT_CAP: usize = 16 << 20;

/// One leak gitleaks found, at tree-relative lines `start..=end`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hit {
    pub rule: String,
    pub description: String,
    pub path: String,
    pub start: i64,
    pub end: i64,
}

/// Runs `gitleaks dir` over the tree at `root` from `work` (the checkout's
/// directory, outside the tree), with `config` when it is not blank, and returns
/// its hits. It deletes the tree's `.gitleaks.toml` and `.gitleaksignore`.
pub async fn gitleaks(
    bin: &str,
    root: &Path,
    work: &Path,
    config: &str,
) -> Result<Vec<Hit>, ProcError> {
    run(bin, root, work, config, TIMEOUT).await
}

async fn run(
    bin: &str,
    root: &Path,
    work: &Path,
    config: &str,
    timeout: Duration,
) -> Result<Vec<Hit>, ProcError> {
    let fail = |reason: String| ProcError::Spawn {
        program: bin.to_owned(),
        reason,
    };
    for name in [".gitleaks.toml", ".gitleaksignore"] {
        // Whatever it is: when `.gitleaks.toml` exists at all, even as a
        // directory, gitleaks searches the tree for a `.gitleaks.*` config.
        let path = root.join(name);
        let removed = match fs::symlink_metadata(&path) {
            Ok(meta) if meta.is_dir() => fs::remove_dir_all(&path),
            Ok(_) => fs::remove_file(&path),
            Err(e) => Err(e),
        };
        match removed {
            Err(e) if e.kind() != ErrorKind::NotFound => {
                return Err(fail(format!("removing {name}: {e}")));
            }
            _ => {}
        }
    }
    let mut args: Vec<OsString> = vec!["dir".into(), root.into()];
    args.extend(
        [
            "--no-banner",
            "--redact",
            "--ignore-gitleaks-allow",
            "--report-format",
            "json",
            "--report-path",
        ]
        .map(OsString::from),
    );
    args.push(work.join(REPORT).into());
    args.extend(["--exit-code", "0"].map(OsString::from));
    let config = config.trim();
    if !config.is_empty() {
        args.extend(["--config", config].map(OsString::from));
    }
    let cmd = Cmd {
        program: bin.into(),
        args,
        cwd: work.to_owned(),
        env: Vec::new(),
        stdin: Vec::new(),
        timeout,
        stdout_cap: 1 << 20,
    };
    proc::run(cmd, &[]).await?;
    let report = workspace::read(work, REPORT, REPORT_CAP)
        .ok_or_else(|| fail("report missing, over 16 MiB or not UTF-8".to_owned()))?;
    // serde's message can quote a value from the report: only where it broke is kept.
    parse_report(report.as_bytes(), root).map_err(|e| {
        fail(format!(
            "report unreadable at line {}, column {}",
            e.line(),
            e.column()
        ))
    })
}

/// The fields read from one gitleaks v8 report entry; `Match` and `Secret` are
/// never read.
#[derive(Deserialize)]
struct Entry {
    #[serde(rename = "RuleID", default)]
    rule: String,
    #[serde(rename = "Description", default)]
    description: String,
    #[serde(rename = "File", default)]
    file: String,
    #[serde(rename = "StartLine", default)]
    start: i64,
    #[serde(rename = "EndLine", default)]
    end: i64,
}

/// The hits in a gitleaks JSON report of a scan of `root`. Paths outside the tree
/// and lines below 1 are dropped. No hits is `[]`, or `null` as Go encodes a nil
/// slice; anything else that is not a list of entries (empty, cut off, another
/// shape) is an error, so a broken scan is never read as a clean one.
pub fn parse_report(report: &[u8], root: &Path) -> Result<Vec<Hit>, serde_json::Error> {
    let entries: Option<Vec<Entry>> = serde_json::from_slice(report)?;
    Ok(entries
        .unwrap_or_default()
        .into_iter()
        .filter(|e| e.start > 0)
        .filter_map(|e| {
            Some(Hit {
                path: rel_path(root, &e.file)?,
                end: e.end.max(e.start),
                start: e.start,
                rule: e.rule,
                description: e.description,
            })
        })
        .collect())
}

/// `file` relative to `root` with `/` separators, or `None` when it escapes the
/// root or is the root. gitleaks reports `<root>/<rel>` for an
/// absolute root, so the match is lexical.
fn rel_path(root: &Path, file: &str) -> Option<String> {
    let file = Path::new(file.trim());
    let rel = if file.is_absolute() {
        file.strip_prefix(root).ok()?
    } else {
        file
    };
    crate::context::ts::join("", rel.to_str()?).filter(|p| !p.is_empty())
}

/// The hits touching a line this PR added, as Critical security findings.
/// Scanners see the whole tree; without this every PR would carry the repo's
/// old leaks. The body names the rule and never the secret; `snippet` stays empty.
pub fn findings(hits: &[Hit], diffs: &BTreeMap<String, DiffIndex>) -> Vec<Finding> {
    hits.iter()
        .filter(|h| {
            diffs.get(&h.path).is_some_and(|d| {
                d.lines()
                    .any(|(n, l)| l.added && (h.start..=h.end).contains(&n))
            })
        })
        .map(|h| Finding {
            path: h.path.clone(),
            start_line: h.start,
            end_line: h.end,
            severity: Severity::Critical,
            category: Category::Security,
            title: title(h),
            body: body(h),
            snippet: String::new(),
            suggestion: String::new(),
        })
        .collect()
}

/// Whether a stored Finding is titled and graded as [`findings`] makes them. The model
/// writes titles too, so this is no proof of origin: such a row's absence resolves it
/// only where gitleaks ran over its file's patch and the model was sent the file.
pub fn from_gitleaks(severity: Severity, category: Category, title: &str) -> bool {
    severity == Severity::Critical
        && category == Category::Security
        && title.starts_with("Hardcoded secret")
}

fn title(h: &Hit) -> String {
    match h.rule.trim() {
        "" => "Hardcoded secret detected".to_owned(),
        rule => format!("Hardcoded secret: {rule}"),
    }
}

fn body(h: &Hit) -> String {
    let rule = match h.rule.trim() {
        "" => "generic-secret",
        rule => rule,
    };
    let mut out = format!("**Potential hardcoded secret** detected by `gitleaks` (rule `{rule}`).");
    let description = h.description.trim();
    if !description.is_empty() {
        out.push(' ');
        out.push_str(description);
    }
    out.push_str(
        "\n\nThis line appears to contain a credential. Remove it from source, **rotate the \
         secret** (assume it is compromised), and load it from a secret manager or environment \
         variable instead. The value is intentionally not echoed here.",
    );
    out
}

/// What a masked line of context shows instead of its text.
pub(crate) const REDACTED: &str = "[redacted by gitleaks]";

/// Masks the cross-file context inside a gitleaks hit, so a secret elsewhere in
/// the tree never reaches a prompt: each such line of a pack snippet becomes
/// [`REDACTED`], and so does each such added or context line of a related diff,
/// keeping its diff marker. Removed lines are not in the scanned tree and stay.
pub fn mask(packs: &mut BTreeMap<String, ContextPack>, hits: &[Hit]) {
    let secret = |path: &str, n: i64| {
        hits.iter()
            .any(|h| h.path == path && (h.start..=h.end).contains(&n))
    };
    for pack in packs.values_mut() {
        for s in &mut pack.snippets {
            let lines = s.text.split('\n').zip(s.start_line..);
            let masked = lines.map(|(line, n)| if secret(&s.path, n) { REDACTED } else { line });
            s.text = masked.collect::<Vec<_>>().join("\n");
        }
        for r in &mut pack.related {
            r.patch = mask_patch(&r.patch, |n| secret(&r.path, n));
        }
    }
}

/// The PR's own `patch` of `path` masked as [`mask`] masks a related diff: the
/// per-file and verify prompts show it. Unchanged where no hit falls.
pub fn mask_diff(path: &str, patch: &str, hits: &[Hit]) -> String {
    mask_patch(patch, |n| {
        hits.iter()
            .any(|h| h.path == path && (h.start..=h.end).contains(&n))
    })
}

/// `patch` with each new-file line `secret` flags replaced by [`REDACTED`] behind
/// its marker, numbered as [`DiffIndex::parse`] numbers them; every other byte kept.
fn mask_patch(patch: &str, secret: impl Fn(i64) -> bool) -> String {
    let mut out = String::with_capacity(patch.len());
    let mut next: Option<i64> = None;
    for piece in patch.split_inclusive('\n') {
        let line = piece.strip_suffix('\n').unwrap_or(piece);
        if let Some(start) = hunk_new_start(line) {
            next = Some(start);
        } else if let Some(n) = next.as_mut() {
            // A blank or unknown leading character is context, as in `parse`.
            let marker = match line.as_bytes().first() {
                Some(b'-' | b'\\') => None,
                Some(b'+') => Some('+'),
                _ => Some(' '),
            };
            if let Some(marker) = marker {
                let hit = secret(*n);
                *n += 1;
                if hit {
                    out.push(marker);
                    out.push_str(REDACTED);
                    out.push_str(&piece[line.len()..]);
                    continue;
                }
            }
        }
        out.push_str(piece);
    }
    out
}

/// The model's findings without those overlapping a tool finding in the same
/// file, then the tool findings: deterministic ground truth wins the overlap.
pub fn merge(model: Vec<Finding>, tool: Vec<Finding>) -> Vec<Finding> {
    let mut out: Vec<Finding> = model
        .into_iter()
        .filter(|f| {
            !tool.iter().any(|t| {
                t.path == f.path && overlaps(f.start_line, f.end_line, t.start_line, t.end_line)
            })
        })
        .collect();
    out.extend(tool);
    out
}

/// Whether `[a1, a2]` and `[b1, b2]` intersect; a file-level finding (line 0)
/// overlaps nothing, so it is never merged away.
fn overlaps(a1: i64, a2: i64, b1: i64, b2: i64) -> bool {
    a1 > 0 && b1 > 0 && a1 <= b2.max(b1) && b1 <= a2.max(a1)
}

#[cfg(test)]
pub(crate) mod tests {
    use std::path::PathBuf;
    use std::process::Command;

    use super::*;

    const SECRET: &str = "AKIAIOSFODNN7EXAMPLE_super_secret_dont_leak";

    /// A fresh directory for one test.
    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("diffagent-scan-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("src")).unwrap();
        dir
    }

    /// Writes an executable `sh` script through a child process, so no thread of
    /// this test binary ever holds it open for writing: a fork racing an exec of
    /// it would fail with "text file busy".
    pub(crate) fn executable(path: &Path, script: &str) {
        let status = Command::new("sh")
            .args([
                "-c",
                "printf '%s' \"$2\" > \"$1\" && chmod 755 \"$1\"",
                "sh",
            ])
            .arg(path)
            .arg(script)
            .status()
            .unwrap();
        assert!(status.success());
    }

    fn hit(path: &str, start: i64, end: i64) -> Hit {
        Hit {
            rule: "aws-access-token".to_owned(),
            description: "AWS Access Token".to_owned(),
            path: path.to_owned(),
            start,
            end,
        }
    }

    fn finding(path: &str, start: i64, end: i64, title: &str) -> Finding {
        Finding {
            path: path.to_owned(),
            start_line: start,
            end_line: end,
            severity: Severity::Major,
            category: Category::Logic,
            title: title.to_owned(),
            body: String::new(),
            snippet: String::new(),
            suggestion: String::new(),
        }
    }

    #[test]
    fn parse_report_maps_and_never_reads_the_secret() {
        let root = Path::new("/tmp/ws");
        let report = format!(
            r#"[{{"RuleID":"aws-access-token","Description":"AWS Access Token","File":"creds.go",
                "StartLine":3,"EndLine":3,"Match":"{SECRET}","Secret":"{SECRET}"}},
               {{"RuleID":"x","File":"/tmp/ws/a/b.go","StartLine":7,"EndLine":2}}]"#
        );
        let got = parse_report(report.as_bytes(), root).unwrap();
        let want = vec![
            hit("creds.go", 3, 3),
            Hit {
                rule: "x".to_owned(),
                description: String::new(),
                path: "a/b.go".to_owned(),
                start: 7,
                end: 7,
            },
        ];
        assert_eq!(got, want);
        assert!(!format!("{got:?}").contains(SECRET));
    }

    /// No hits is `[]` (or Go's `null`); a report that is not one is an error, so the
    /// scan counts as not run instead of clean.
    #[test]
    fn parse_report_empty_malformed_and_bad_entries() {
        let root = Path::new("/tmp/ws");
        for input in ["[]", " [ ]\n", "null"] {
            assert_eq!(
                parse_report(input.as_bytes(), root).unwrap(),
                [],
                "{input:?}"
            );
        }
        let broken = [
            "",
            "  ",
            "not json",
            r#"{"oops":1}"#,
            "[1]",
            r#"[{"RuleID":"x","File":"a.go","StartLine":2}"#,
        ];
        for input in broken {
            assert!(parse_report(input.as_bytes(), root).is_err(), "{input:?}");
        }
        let report = r#"[
            {"RuleID":"x","File":"","StartLine":3},
            {"RuleID":"y","File":"creds.go","StartLine":0},
            {"RuleID":"z","File":"../escape.go","StartLine":1}
        ]"#;
        assert_eq!(parse_report(report.as_bytes(), root).unwrap(), []);
    }

    #[test]
    fn rel_paths() {
        let root = Path::new("/tmp/ws");
        let cases = [
            ("creds.go", Some("creds.go")),
            ("/tmp/ws/a/b.go", Some("a/b.go")),
            ("./creds.go", Some("creds.go")),
            (" a/./c/../b.go ", Some("a/b.go")),
            ("/etc/passwd", None),
            ("/tmp/wsx/a.go", None),
            ("/tmp/ws/../x.go", None),
            ("a/../../x.go", None),
            ("a/..", None),
            ("", None),
            ("/tmp/ws", None),
        ];
        for (file, want) in cases {
            assert_eq!(rel_path(root, file).as_deref(), want, "{file}");
        }
    }

    #[test]
    fn findings_keep_only_added_line_hits() {
        // New-file lines 10 and 12 added, 11 context.
        let diffs = BTreeMap::from([(
            "foo.go".to_owned(),
            DiffIndex::parse("@@ -10,1 +10,3 @@ func F() {\n+a\n b\n+c"),
        )]);
        let hits = [
            hit("foo.go", 10, 10),
            hit("foo.go", 11, 11),
            hit("foo.go", 11, 12),
            hit("foo.go", 99, 99),
            hit("other.go", 10, 10),
        ];
        let got = findings(&hits, &diffs);
        let spans: Vec<(i64, i64)> = got.iter().map(|f| (f.start_line, f.end_line)).collect();
        assert_eq!(spans, [(10, 10), (11, 12)]);
        let f = &got[0];
        assert_eq!(
            (f.severity, f.category, f.title.as_str()),
            (
                Severity::Critical,
                Category::Security,
                "Hardcoded secret: aws-access-token"
            )
        );
        assert_eq!(
            f.body,
            "**Potential hardcoded secret** detected by `gitleaks` (rule `aws-access-token`). \
             AWS Access Token\n\nThis line appears to contain a credential. Remove it from \
             source, **rotate the secret** (assume it is compromised), and load it from a \
             secret manager or environment variable instead. The value is intentionally not \
             echoed here."
        );
        assert_eq!((f.snippet.as_str(), f.suggestion.as_str()), ("", ""));

        let bare = Hit {
            rule: " ".to_owned(),
            description: String::new(),
            ..hit("foo.go", 12, 12)
        };
        let f = &findings(&[bare], &diffs)[0];
        assert_eq!(f.title, "Hardcoded secret detected");
        for f in got.iter().chain([f]) {
            assert!(
                from_gitleaks(f.severity, f.category, &f.title),
                "{}",
                f.title
            );
        }
        let model = &got[0];
        assert!(!from_gitleaks(
            Severity::Major,
            model.category,
            &model.title
        ));
        assert!(
            f.body.starts_with(
                "**Potential hardcoded secret** detected by `gitleaks` (rule `generic-secret`).\n\n"
            ),
            "{}",
            f.body
        );
    }

    /// End to end from a report holding the secret: no field of the finding has it.
    #[test]
    fn findings_never_echo_the_secret() {
        let report = format!(
            r#"[{{"RuleID":"aws-access-token","Description":"AWS","File":"creds.go",
                "StartLine":1,"EndLine":1,"Match":"{SECRET}","Secret":"{SECRET}"}}]"#
        );
        let hits = parse_report(report.as_bytes(), Path::new("/w")).unwrap();
        let patch = format!("@@ -0,0 +1,1 @@\n+const k = \"{SECRET}\"");
        let diffs = BTreeMap::from([("creds.go".to_owned(), DiffIndex::parse(&patch))]);
        let got = findings(&hits, &diffs);
        assert_eq!(got.len(), 1);
        assert!(!format!("{got:?}").contains(SECRET));
    }

    #[test]
    fn overlap_table() {
        let cases = [
            (5, 5, 4, 6, true),
            (1, 3, 3, 9, true),
            (1, 2, 3, 4, false),
            (5, 0, 5, 0, true),
            (6, 0, 4, 5, false),
            (0, 0, 1, 2, false),
            (1, 2, 0, 0, false),
            (-1, 3, 1, 2, false),
        ];
        for (a1, a2, b1, b2, want) in cases {
            assert_eq!(overlaps(a1, a2, b1, b2), want, "{a1},{a2} vs {b1},{b2}");
        }
    }

    #[test]
    fn merge_lets_the_tool_win_an_overlap() {
        let model = vec![
            finding("a.go", 5, 5, "llm secret-ish"),
            finding("a.go", 50, 50, "llm logic bug"),
            finding("b.go", 5, 5, "other file"),
            finding("a.go", 0, 0, "file level"),
        ];
        let tool = vec![finding("a.go", 4, 6, "gitleaks")];
        let titles: Vec<String> = merge(model.clone(), tool)
            .into_iter()
            .map(|f| f.title)
            .collect();
        assert_eq!(
            titles,
            ["llm logic bug", "other file", "file level", "gitleaks"]
        );
        assert_eq!(merge(model.clone(), Vec::new()), model);
    }

    /// Only lines inside a hit on the same path are masked: a snippet's by its
    /// line number, a related diff's added or context line by its new-file number
    /// (the removed line takes none). Another path's context is left alone.
    #[test]
    fn mask_redacts_context_inside_a_hit() {
        use crate::context::{RelatedDiff, Relation, Snippet, SnippetKind};
        let snippet = |path: &str| Snippet {
            kind: SnippetKind::Definition,
            symbol: "key".to_owned(),
            path: path.to_owned(),
            start_line: 10,
            end_line: 12,
            text: format!("const (\n\tkey = \"{SECRET}\"\n)"),
        };
        // New-file lines 4, 5 (added) and 6.
        let patch = format!("@@ -4,3 +4,3 @@\n a := 1\n-old := 0\n+key := \"{SECRET}\"\n b := 2\n");
        let related = |path: &str| RelatedDiff {
            path: path.to_owned(),
            relation: Relation::Imports,
            patch: patch.clone(),
        };
        let pack = ContextPack {
            snippets: vec![snippet("cfg/keys.go"), snippet("other.go")],
            related: vec![related("cfg/keys.go"), related("other.go")],
            error: None,
        };
        let packs = BTreeMap::from([("a.go".to_owned(), pack)]);

        let mut unmasked = packs.clone();
        mask(&mut unmasked, &[]);
        assert_eq!(unmasked, packs, "no hit, no change");

        let mut masked = packs.clone();
        mask(
            &mut masked,
            &[hit("cfg/keys.go", 11, 11), hit("cfg/keys.go", 5, 5)],
        );
        let got = &masked["a.go"];
        assert_eq!(got.snippets[0].text, "const (\n[redacted by gitleaks]\n)");
        assert_eq!(
            got.related[0].patch,
            "@@ -4,3 +4,3 @@\n a := 1\n-old := 0\n+[redacted by gitleaks]\n b := 2\n"
        );
        assert_eq!(got.snippets[1], packs["a.go"].snippets[1]);
        assert_eq!(got.related[1], packs["a.go"].related[1]);
    }

    /// A fake gitleaks records how it was run and writes a report naming a file
    /// under the root it was given.
    #[tokio::test]
    async fn gitleaks_runs_outside_the_tree_with_pr_config_gone() {
        let work = scratch("run");
        let root = work.join("src");
        fs::write(root.join("creds.go"), "x").unwrap();
        // A directory: gitleaks would then load the PR's `.gitleaks.json`.
        fs::create_dir_all(root.join(".gitleaks.toml/sub")).unwrap();
        fs::write(root.join(".gitleaks.toml/sub/.keep"), "").unwrap();
        fs::write(root.join(".gitleaksignore"), "creds.go:aws:1").unwrap();
        let log = scratch("run-log");
        let bin = log.join("gitleaks");
        executable(
            &bin,
            &format!(
                r#"#!/bin/sh
log={log}
root=$2
printf '%s\n' "$@" > "$log/args"
pwd -P > "$log/cwd"
ls -A "$root" > "$log/tree"
while [ $# -gt 0 ]; do
  if [ "$1" = --report-path ]; then report=$2; fi
  shift
done
printf '[{{"RuleID":"r","File":"%s/creds.go","StartLine":2,"EndLine":2,"Secret":"REDACTED"}}]' "$root" > "$report"
"#,
                log = log.display()
            ),
        );

        let hits = gitleaks(bin.to_str().unwrap(), &root, &work, " /etc/gl.toml ")
            .await
            .unwrap();

        let report = work.join("gitleaks.json");
        let want_args = [
            "dir",
            root.to_str().unwrap(),
            "--no-banner",
            "--redact",
            "--ignore-gitleaks-allow",
            "--report-format",
            "json",
            "--report-path",
            report.to_str().unwrap(),
            "--exit-code",
            "0",
            "--config",
            "/etc/gl.toml",
        ];
        let args = fs::read_to_string(log.join("args")).unwrap();
        assert_eq!(args.lines().collect::<Vec<_>>(), want_args);
        let cwd = PathBuf::from(fs::read_to_string(log.join("cwd")).unwrap().trim());
        assert_eq!(cwd, work.canonicalize().unwrap());
        assert!(!cwd.starts_with(root.canonicalize().unwrap()));
        assert_eq!(fs::read_to_string(log.join("tree")).unwrap(), "creds.go\n");
        assert_eq!(
            hits,
            [Hit {
                rule: "r".to_owned(),
                description: String::new(),
                path: "creds.go".to_owned(),
                start: 2,
                end: 2,
            }]
        );
        fs::remove_dir_all(work).unwrap();
        fs::remove_dir_all(log).unwrap();
    }

    #[tokio::test]
    async fn gitleaks_hang_and_missing_report() {
        let work = scratch("hang");
        let root = work.join("src");
        let bin = work.join("gitleaks");
        executable(&bin, "#!/bin/sh\nsleep 60\n");
        let err = run(
            bin.to_str().unwrap(),
            &root,
            &work,
            "",
            Duration::from_millis(300),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ProcError::Timeout { .. }), "{err:?}");

        executable(&bin, "#!/bin/sh\nexit 0\n");
        let err = gitleaks(bin.to_str().unwrap(), &root, &work, "")
            .await
            .unwrap_err();
        assert!(matches!(err, ProcError::Spawn { .. }), "{err:?}");

        // An empty or cut-off report is no clean scan.
        for report in ["", "[{"] {
            fs::write(work.join(REPORT), report).unwrap();
            let err = gitleaks(bin.to_str().unwrap(), &root, &work, "")
                .await
                .unwrap_err();
            assert!(
                matches!(err, ProcError::Spawn { .. }),
                "{report:?}: {err:?}"
            );
        }
        fs::remove_dir_all(work).unwrap();
    }
}
