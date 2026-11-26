//! Suggestion checking: splice a finding's proposed replacement into the NEW file
//! over its flagged lines and re-parse. A suggestion that no longer parses would
//! break the build if applied with one click, so it is shown as an advisory diff
//! instead (the explanation still posts).
//!
//! Checking is best-effort and scoped to the languages in [`Lang`]; anything else
//! passes. The worst failure is downgrading a VALID suggestion, so this only blames
//! the suggestion when it is confident: the finding's snippet sits in the cited lines
//! (the same precondition GitHub uses for a committable suggestion) and the original
//! file parses on its own. Any doubt leaves the suggestion one-click.

use crate::diff::normalize_code;
use crate::finding::Finding;
use crate::lang::Lang;

/// Files larger than this aren't checked: a giant generated file (protobuf stubs,
/// mocks) isn't worth parsing twice per suggestion on a memory-tight node.
pub const MAX_CHECK_BYTES: usize = 1 << 20;

/// Whether applying `f`'s suggestion over lines `start_line..=end_line` of `source`
/// (the full NEW file at `f.path`) turns a parsing file into one with a syntax error
/// (`ERROR` or `MISSING` node). `true` means: post it as advisory, not one-click.
pub fn breaks_syntax(source: &str, f: &Finding) -> bool {
    // Render commits the suggestion without trailing newlines; check the same bytes.
    let suggestion = f.suggestion.trim_end_matches('\n');
    if suggestion.trim().is_empty() || f.start_line <= 0 || source.len() > MAX_CHECK_BYTES {
        return false;
    }
    let Some(lang) = Lang::for_path(&f.path) else {
        return false;
    };
    let lines: Vec<&str> = source.lines().collect();
    // The model's line numbers may be off (anchoring relocates them later); splicing
    // at wrong coordinates could break valid code and blame the suggestion.
    if !snippet_aligned(&lines, f.start_line, f.end_line, &f.snippet) {
        return false;
    }
    let Some(spliced) = splice(&lines, f.start_line, f.end_line, suggestion) else {
        return false;
    };
    // If the original doesn't parse (build tags, partial syntax, not really that
    // language), a parse error after splicing proves nothing.
    // ponytail: re-parses the original per suggestion; cache per path if a Review
    // ever carries many suggestions in one large file.
    parses(lang, source) == Some(true) && parses(lang, &spliced) == Some(false)
}

/// `None` when the grammar can't load: unknown, never blamed on the suggestion.
fn parses(lang: Lang, source: &str) -> Option<bool> {
    lang.parse(source).map(|t| !t.root_node().has_error())
}

/// `lines` with `start..=end` (1-based) replaced by `replacement`, or `None` when the
/// span is outside the file.
fn splice(lines: &[&str], start: i64, end: i64, replacement: &str) -> Option<String> {
    let first = usize::try_from(start.checked_sub(1)?).ok()?;
    let last = usize::try_from(end.max(start)).ok()?;
    let mut out = lines.get(..first)?.to_vec();
    out.extend(replacement.split('\n'));
    out.extend_from_slice(lines.get(last..)?);
    Some(out.join("\n"))
}

/// Whether the snippet's first non-blank line appears in lines `start..=end`,
/// whitespace-collapsed. Containment either way tolerates the model quoting part of
/// the line, or a little more than it.
fn snippet_aligned(lines: &[&str], start: i64, end: i64, snippet: &str) -> bool {
    let first_line = snippet.lines().map(str::trim).find(|l| !l.is_empty());
    let want = normalize_code(first_line.unwrap_or_default());
    let (Ok(first), Ok(last)) = (
        usize::try_from(start.saturating_sub(1)),
        usize::try_from(end.max(start)),
    ) else {
        return false;
    };
    !want.is_empty()
        && lines
            .iter()
            .take(last)
            .skip(first)
            .map(|l| normalize_code(l))
            .any(|got| !got.is_empty() && (got.contains(&want) || want.contains(&got)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finding::{Category, Severity};

    // Lines: 1 package, 2 blank, 3 import, 4 blank, 5 func main() {, 6 x := 1,
    // 7 fmt.Println(x), 8 }.
    const GO: &str =
        "package main\n\nimport \"fmt\"\n\nfunc main() {\n\tx := 1\n\tfmt.Println(x)\n}\n";
    const TS: &str = "export function add(a: number, b: number): number {\n\tconst sum = a + b;\n\treturn sum;\n}\n";
    const TSX: &str = "export const C = () => {\n\tconst x = 1;\n\treturn <div>{x}</div>;\n};\n";
    const JS: &str = "function add(a, b) {\n\treturn a + b;\n}\n";
    const PY: &str = "def f():\n    return 1\n";

    fn finding(path: &str, line: i64, snippet: &str, suggestion: &str) -> Finding {
        Finding {
            path: path.to_owned(),
            start_line: line,
            end_line: line,
            severity: Severity::Minor,
            category: Category::Logic,
            title: String::new(),
            body: String::new(),
            snippet: snippet.to_owned(),
            suggestion: suggestion.to_owned(),
        }
    }

    #[test]
    fn breaks_syntax_cases() {
        let big = format!("{GO}\n// {}\n", "x".repeat(MAX_CHECK_BYTES));
        // (name, path, source, line, snippet, suggestion, breaks)
        let cases = [
            ("go valid", "main.go", GO, 6, "x := 1", "\tx := 2", false),
            (
                "go broken",
                "main.go",
                GO,
                6,
                "x := 1",
                "\tx := := )(",
                true,
            ),
            (
                "misaligned snippet",
                "main.go",
                GO,
                7,
                "x := 1",
                "broken )((",
                false,
            ),
            ("no snippet", "main.go", GO, 6, "", "broken )((", false),
            ("last line valid", "main.go", GO, 8, "}", "}", false),
            ("last line broken", "main.go", GO, 8, "}", "x := 1", true),
            (
                "trailing newline",
                "main.go",
                GO,
                6,
                "x := 1",
                "\tx := 2\n",
                false,
            ),
            (
                "multi-line valid",
                "main.go",
                GO,
                6,
                "x := 1",
                "\tx := 1\n\ty := 2\n\t_ = y",
                false,
            ),
            ("out of bounds", "main.go", GO, 999, "x := 1", "x", false),
            ("empty suggestion", "main.go", GO, 6, "x := 1", "", false),
            ("file-level finding", "main.go", GO, 0, "x := 1", "x", false),
            (
                "oversize file",
                "main.go",
                &big,
                6,
                "x := 1",
                "broken )((",
                false,
            ),
            (
                "unparseable original",
                "broken.go",
                "package main\nfunc (((\n",
                2,
                "func (((",
                "\tx := := )(",
                false,
            ),
            // go/parser rejects this ("expression in defer must be function call");
            // tree-sitter's grammar accepts any expression, so it stays one-click.
            (
                "go grammar is looser",
                "main.go",
                GO,
                6,
                "x := 1",
                "\tdefer x",
                false,
            ),
            (
                "ts valid",
                "m.ts",
                TS,
                2,
                "const sum = a + b;",
                "\tconst sum = a + b + 0;",
                false,
            ),
            (
                "ts broken",
                "m.ts",
                TS,
                2,
                "const sum = a + b;",
                "\tconst sum = a + (",
                true,
            ),
            (
                "ts uppercase ext",
                "M.TS",
                TS,
                2,
                "const sum = a + b;",
                "\tconst sum = a + (",
                true,
            ),
            (
                "tsx broken",
                "C.tsx",
                TSX,
                2,
                "const x = 1;",
                "\tconst x = ;",
                true,
            ),
            (
                "js valid",
                "m.js",
                JS,
                2,
                "return a + b;",
                "\treturn a + b + 0;",
                false,
            ),
            (
                "js broken",
                "m.js",
                JS,
                2,
                "return a + b;",
                "\treturn a + (",
                true,
            ),
            (
                "python broken",
                "app.py",
                PY,
                2,
                "return 1",
                "    return (",
                true,
            ),
            (
                "unsupported language",
                "app.lua",
                PY,
                2,
                "return 1",
                "    return (",
                false,
            ),
        ];
        for (name, path, src, line, snippet, suggestion, want) in cases {
            let f = finding(path, line, snippet, suggestion);
            assert_eq!(breaks_syntax(src, &f), want, "{name}");
        }
    }

    #[test]
    fn splice_spans() {
        let lines = ["a", "b", "c"];
        let cases = [
            (2, 2, Some("a\nX\nY\nc")),
            (1, 3, Some("X\nY")),
            (3, 1, Some("a\nb\nX\nY")), // end before start: just the start line
            (3, 3, Some("a\nb\nX\nY")),
            (4, 4, None),
            (2, 4, None),
            (0, 1, None),
        ];
        for (start, end, want) in cases {
            assert_eq!(
                splice(&lines, start, end, "X\nY").as_deref(),
                want,
                "{start}..={end}"
            );
        }
    }
}
