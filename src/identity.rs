//! Finding identity: the Fingerprint that says "this is the same problem as last
//! push", so a finding is not re-posted as new, its old thread not wrongly marked
//! addressed, and a wontfix not silently lapsed.
//!
//! Three rules keep an identity stable:
//! - Every [`Lang`] anchors to the innermost named function enclosing the line. A
//!   hash of the flagged lines alone would mint a new identity whenever the flagged
//!   line is edited.
//! - The content-anchor lines come from [`DiffIndex`], which counts blank context
//!   lines, so the hash covers the right lines.
//! - Co-located findings are told apart by the model-picked [`Category`], not by the
//!   model's title, so a reworded title keeps its identity. Two findings still sharing
//!   symbol and category are told apart by their flagged line's code, never by prose.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use blake2::digest::consts::{U8, U16};
use blake2::{Blake2b, Digest};
use tree_sitter::{Node, Tree};

use crate::diff::{DiffIndex, normalize_code};
use crate::finding::{Category, Finding};
use crate::lang::Lang;

/// Gives every Finding of one Review its Fingerprint, in input order.
///
/// `patches` and `sources` are keyed by path: the file's unified diff, and its full
/// new-file text when a checkout is available. A path missing from `sources` (or an
/// unparsed language) gets content anchors. Exact duplicates (same base Fingerprint
/// and same flagged-line code) collapse to the first one. When several findings still
/// share a base, each gets `base:<hash of its flagged-line code>`, so the result does
/// not depend on the order the model listed them in.
pub fn fingerprint_findings<'a>(
    repo: &str,
    findings: &'a [Finding],
    patches: &HashMap<String, String>,
    sources: &HashMap<String, String>,
) -> Vec<(String, &'a Finding)> {
    let mut files: HashMap<&str, ChangedFile<'_>> = HashMap::new();
    let mut seen = HashSet::new();
    let mut entries = Vec::new();
    for f in findings {
        let file = files.entry(f.path.as_str()).or_insert_with(|| {
            ChangedFile::new(&f.path, patches.get(&f.path), sources.get(&f.path))
        });
        let base = fingerprint(repo, &f.path, &file.anchor(f), f.category);
        let code = file.flagged_code(f);
        if seen.insert((base.clone(), code.clone())) {
            entries.push((base, code, f));
        }
    }

    let mut per_base: HashMap<String, usize> = HashMap::new();
    for (base, _, _) in &entries {
        *per_base.entry(base.clone()).or_default() += 1;
    }
    entries
        .into_iter()
        .map(|(base, code, f)| {
            if per_base.get(&base).is_some_and(|n| *n > 1) {
                let salt = hash_hex::<Blake2b<U8>>([code.as_bytes()]);
                (format!("{base}:{salt}"), f)
            } else {
                (base, f)
            }
        })
        .collect()
}

/// What identity needs from one changed file, built once per path.
struct ChangedFile<'s> {
    diff: DiffIndex,
    parsed: Option<(Lang, Tree, &'s str)>,
}

impl<'s> ChangedFile<'s> {
    fn new(path: &str, patch: Option<&String>, source: Option<&'s String>) -> Self {
        let parsed = source.and_then(|src| {
            let lang = Lang::for_path(path)?;
            Some((lang, lang.parse(src)?, src.as_str()))
        });
        Self {
            diff: patch.map(|p| DiffIndex::parse(p)).unwrap_or_default(),
            parsed,
        }
    }

    /// `sym:<name>` of the enclosing named function, else a hash of the flagged
    /// lines' code.
    fn anchor(&self, f: &Finding) -> String {
        let symbol = self
            .parsed
            .as_ref()
            .and_then(|(lang, tree, src)| enclosing_symbol(*lang, tree, src, f.start_line));
        match symbol {
            Some(name) => format!("sym:{name}"),
            None => content_anchor(
                &self
                    .diff
                    .new_side(f.start_line, f.end_line.max(f.start_line)),
            ),
        }
    }

    /// The flagged line's code, whitespace-collapsed: the diff's line when it is
    /// addressable, else the model's quoted snippet.
    fn flagged_code(&self, f: &Finding) -> String {
        normalize_code(self.diff.content(f.start_line).unwrap_or(&f.snippet))
    }
}

/// The innermost named function or method whose lines include `line` (1-based, NEW
/// file): Go `Func` / `Recv.Method`, TS/JS `func` / `Class.method` / `name` of
/// `const name = () => …`, and in the other languages `func` / `Type.method`.
/// Anonymous functions are skipped, so a line inside a callback belongs to the named
/// function around it. `None` for top-level code.
pub fn enclosing_symbol(lang: Lang, tree: &Tree, source: &str, line: i64) -> Option<String> {
    let row = usize::try_from(line.checked_sub(1)?).ok()?;
    // Several siblings can share the row (`export` then `const …`), so visit every
    // node on it and keep the smallest named one: the innermost.
    let mut best: Option<(usize, String)> = None;
    let mut stack = vec![tree.root_node()];
    while let Some(node) = stack.pop() {
        if row < node.start_position().row || node.end_position().row < row {
            continue;
        }
        if let Some(name) = symbol_name(lang, node, source) {
            let len = node.byte_range().len();
            if best.as_ref().is_none_or(|(l, _)| len < *l) {
                best = Some((len, name));
            }
        }
        let mut cursor = node.walk();
        stack.extend(node.named_children(&mut cursor));
    }
    best.map(|(_, name)| name)
}

/// The symbol name `node` declares, if it is a named function or method. Kinds are
/// read per language, since grammars reuse names (`method_declaration` is a Go method
/// with a receiver and a Java method without one).
fn symbol_name(lang: Lang, node: Node<'_>, source: &str) -> Option<String> {
    match lang {
        Lang::Go | Lang::TypeScript | Lang::Tsx | Lang::JavaScript => go_ts_name(node, source),
        _ => declared_name(lang, node, source),
    }
}

/// The node kinds that declare a named function or method in `lang`, and the kinds
/// whose name qualifies one declared inside them (`Class.method`).
fn declaration_kinds(lang: Lang) -> (&'static [&'static str], &'static [&'static str]) {
    match lang {
        Lang::Go | Lang::TypeScript | Lang::Tsx | Lang::JavaScript => (&[], &[]),
        Lang::Python => (&["function_definition"], &["class_definition"]),
        Lang::Java => (
            &[
                "method_declaration",
                "constructor_declaration",
                "compact_constructor_declaration",
            ],
            &[
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
                "record_declaration",
            ],
        ),
        Lang::Kotlin => (
            &["function_declaration"],
            &["class_declaration", "object_declaration"],
        ),
        Lang::CSharp => (
            &[
                "method_declaration",
                "constructor_declaration",
                "destructor_declaration",
                "local_function_statement",
            ],
            &[
                "class_declaration",
                "struct_declaration",
                "interface_declaration",
                "record_declaration",
            ],
        ),
        Lang::Php => (
            &["function_definition", "method_declaration"],
            &[
                "class_declaration",
                "interface_declaration",
                "trait_declaration",
                "enum_declaration",
            ],
        ),
        Lang::Ruby => (&["method", "singleton_method"], &["class", "module"]),
        Lang::Rust => (
            &["function_item", "function_signature_item"],
            &["impl_item", "trait_item"],
        ),
        Lang::Swift => (
            &[
                "function_declaration",
                "init_declaration",
                "protocol_function_declaration",
            ],
            &["class_declaration", "protocol_declaration"],
        ),
        Lang::C => (&["function_definition"], &[]),
        Lang::Cpp => (
            &["function_definition"],
            &["class_specifier", "struct_specifier"],
        ),
        Lang::Scala => (
            &["function_definition", "function_declaration"],
            &["class_definition", "object_definition", "trait_definition"],
        ),
    }
}

/// [`symbol_name`] outside Go and TS/JS: the declared name, qualified by the nearest
/// enclosing type. A Rust `impl` is named by its type, generics dropped, so
/// `impl<T> Stack<T>` gives `Stack.push`. C and C++ name a function in its
/// declarator (`int *f(void)`, `A::f`).
fn declared_name(lang: Lang, node: Node<'_>, source: &str) -> Option<String> {
    let (functions, owners) = declaration_kinds(lang);
    if !functions.contains(&node.kind()) {
        return None;
    }
    let text = |n: Node<'_>| n.utf8_text(source.as_bytes()).ok().map(str::to_owned);
    let name = match node.child_by_field_name("name") {
        Some(n) => text(n)?,
        None => {
            let mut d = node.child_by_field_name("declarator")?;
            while !matches!(
                d.kind(),
                "identifier"
                    | "field_identifier"
                    | "qualified_identifier"
                    | "destructor_name"
                    | "operator_name"
            ) {
                d = d
                    .child_by_field_name("declarator")
                    .or_else(|| d.named_child(0))?;
            }
            text(d)?
        }
    };
    let owner = std::iter::successors(node.parent(), Node::parent)
        .find(|n| owners.contains(&n.kind()))
        .and_then(|o| {
            o.child_by_field_name("name")
                .or_else(|| o.child_by_field_name("type"))
        })
        .and_then(text);
    Some(match owner {
        Some(o) => format!("{}.{name}", o.split(['<', '[']).next().unwrap_or(&o).trim()),
        None => name,
    })
}

/// [`symbol_name`] for Go and TS/JS.
fn go_ts_name(node: Node<'_>, source: &str) -> Option<String> {
    let text = |n: Node<'_>| n.utf8_text(source.as_bytes()).ok();
    let field = |n: Node<'_>, f: &str| n.child_by_field_name(f).and_then(text);
    match node.kind() {
        // Go functions and TS/JS function declarations.
        "function_declaration" | "generator_function_declaration" => {
            field(node, "name").map(str::to_owned)
        }
        // Go method: the receiver's type without pointer or type parameters, so
        // `(s *Stack[T]) Push` is `Stack.Push`.
        "method_declaration" => {
            let recv = node.child_by_field_name("receiver")?.named_child(0)?;
            let ty = field(recv, "type")?.trim_start_matches('*');
            let ty = ty.split('[').next()?.trim();
            Some(format!("{ty}.{}", field(node, "name")?))
        }
        // TS/JS method: qualified by its class (or the variable holding an object
        // literal) when that has a name.
        "method_definition" => {
            let method = field(node, "name")?;
            let owner = node.parent().and_then(|body| body.parent());
            Some(match owner.and_then(|o| field(o, "name")) {
                Some(owner) => format!("{owner}.{method}"),
                None => method.to_owned(),
            })
        }
        // `const name = () => …` / `const name = function …`.
        "variable_declarator" => {
            let value = node.child_by_field_name("value")?.kind();
            let is_fn = matches!(
                value,
                "arrow_function" | "function_expression" | "generator_function"
            );
            if is_fn {
                field(node, "name").map(str::to_owned)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Hash of the flagged lines' code, each whitespace-collapsed so reindenting or
/// moving them doesn't change it.
fn content_anchor(lines: &[&str]) -> String {
    let collapsed: Vec<String> = lines.iter().map(|l| normalize_code(l) + "\n").collect();
    hash_hex::<Blake2b<U8>>(collapsed.iter().map(String::as_bytes))
}

/// Blake2b-128 over (repo, path, anchor, category), each trimmed, lowercased and
/// NUL-terminated so concatenations can't alias ("ab"+"c" vs "a"+"bc"). Title, body
/// and severity are deliberately absent: rewording or re-grading a finding must not
/// make it new.
fn fingerprint(repo: &str, path: &str, anchor: &str, category: Category) -> String {
    let parts = [repo, path, anchor, category.as_str()].map(|p| p.trim().to_lowercase() + "\0");
    hash_hex::<Blake2b<U16>>(parts.iter().map(String::as_bytes))
}

pub(crate) fn hash_hex<'p, D: Digest>(parts: impl IntoIterator<Item = &'p [u8]>) -> String {
    let mut h = D::new();
    for p in parts {
        h.update(p);
    }
    h.finalize().iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finding::Severity;

    fn symbol(path: &str, src: &str, line: i64) -> Option<String> {
        let lang = Lang::for_path(path).expect("a parsed language");
        let tree = lang.parse(src).expect("parses");
        enclosing_symbol(lang, &tree, src, line)
    }

    const GO: &str = "package p

var top = 1

func Foo() {
\tx := 1
\tgo func() {
\t\tuse(x)
\t}()
}

func (s *Server) Handle() {
\ts.n++
}

func (s *Stack[T]) Push(v T) {
\ts.items = append(s.items, v)
}

func (m Map[K, V]) Get(k K) V {
\treturn m.m[k]
}
";

    const TS: &str = "import { x } from './x';

export class Svc {
  fetch(id: string) {
    return id;
  }
}

export const load = async (id: string) => {
  return id;
};

function outer() {
  items.forEach((x) => {
    use(x);
  });
  const inner = function () {
    work();
  };
}

console.log(load);
";

    const TSX: &str = "export const Card = () => {
  return <div onClick={() => go()}>hi</div>;
};
";

    const JS: &str = "class A {
  m() {
    return 1;
  }
}
const api = {
  get() {
    return 2;
  },
};
function* gen() {
  yield 1;
}
";

    #[test]
    fn enclosing_symbols() {
        let cases = [
            ("a.go", GO, 3, None),
            ("a.go", GO, 5, Some("Foo")),
            ("a.go", GO, 6, Some("Foo")),
            ("a.go", GO, 8, Some("Foo")), // inside an anonymous goroutine
            ("a.go", GO, 10, Some("Foo")),
            ("a.go", GO, 11, None),
            ("a.go", GO, 13, Some("Server.Handle")),
            ("a.go", GO, 17, Some("Stack.Push")),
            ("a.go", GO, 21, Some("Map.Get")),
            ("a.go", GO, 0, None),
            ("a.go", GO, 999, None),
            ("a.ts", TS, 1, None),
            ("a.ts", TS, 3, None), // class line, not a function
            ("a.ts", TS, 5, Some("Svc.fetch")),
            ("a.ts", TS, 9, Some("load")),
            ("a.ts", TS, 10, Some("load")),
            ("a.ts", TS, 15, Some("outer")), // inside an anonymous callback
            ("a.ts", TS, 17, Some("inner")),
            ("a.ts", TS, 18, Some("inner")),
            ("a.ts", TS, 20, Some("outer")),
            ("a.ts", TS, 22, None),
            ("C.tsx", TSX, 2, Some("Card")),
            ("a.js", JS, 3, Some("A.m")),
            ("a.js", JS, 8, Some("api.get")),
            ("a.js", JS, 12, Some("gen")),
        ];
        for (path, src, line, want) in cases {
            assert_eq!(symbol(path, src, line).as_deref(), want, "{path}:{line}");
        }
    }

    #[test]
    fn enclosing_symbols_in_other_languages() {
        let cases = [
            (
                "a.py",
                "import os\nclass A:\n    def m(self):\n        pass\ndef f():\n    pass\n",
                6,
                Some("f"),
            ),
            (
                "a.py",
                "import os\nclass A:\n    def m(self):\n        pass\n",
                4,
                Some("A.m"),
            ),
            ("a.py", "import os\n", 1, None),
            (
                "A.java",
                "package p;\nclass A {\n  int m() { return 1; }\n  A() {}\n}\n",
                3,
                Some("A.m"),
            ),
            (
                "A.java",
                "package p;\nclass A {\n  int m() { return 1; }\n  A() {}\n}\n",
                4,
                Some("A.A"),
            ),
            (
                "A.java",
                "package p;\nclass A {\n  int m() { return 1; }\n}\n",
                1,
                None,
            ),
            (
                "a.kt",
                "package p\nclass A {\n  fun m() = 1\n}\nfun f() = 2\n",
                3,
                Some("A.m"),
            ),
            (
                "a.kt",
                "package p\nclass A {\n  fun m() = 1\n}\nfun f() = 2\n",
                5,
                Some("f"),
            ),
            (
                "A.cs",
                "namespace N;\nclass A {\n  int M() { return 1; }\n}\n",
                3,
                Some("A.M"),
            ),
            (
                "a.php",
                "<?php\nclass A {\n  function m() { return 1; }\n}\nfunction f() {}\n",
                3,
                Some("A.m"),
            ),
            (
                "a.php",
                "<?php\nclass A {\n  function m() { return 1; }\n}\nfunction f() {}\n",
                5,
                Some("f"),
            ),
            (
                "a.rb",
                "require 'x'\nclass A\n  def m\n    1\n  end\nend\n",
                4,
                Some("A.m"),
            ),
            (
                "a.rb",
                "require 'x'\nmodule M\n  def self.s\n    1\n  end\nend\n",
                4,
                Some("M.s"),
            ),
            (
                "a.rs",
                "use x;\nimpl<T> Stack<T> {\n    fn push(&mut self) {}\n}\nfn f() {}\n",
                3,
                Some("Stack.push"),
            ),
            (
                "a.rs",
                "use x;\nimpl Display for A {\n    fn fmt(&self) {}\n}\nfn f() {}\n",
                3,
                Some("A.fmt"),
            ),
            (
                "a.rs",
                "use x;\nimpl<T> Stack<T> {\n    fn push(&mut self) {}\n}\nfn f() {}\n",
                5,
                Some("f"),
            ),
            ("a.rs", "use x;\n", 1, None),
            (
                "A.swift",
                "import X\nclass A {\n  func m() -> Int { return 1 }\n}\nfunc f() {}\n",
                3,
                Some("A.m"),
            ),
            (
                "A.swift",
                "import X\nclass A {\n  func m() -> Int { return 1 }\n}\nfunc f() {}\n",
                5,
                Some("f"),
            ),
            (
                "a.c",
                "#include <x.h>\nstatic int *g(void);\nint *f(int x) {\n  return 0;\n}\n",
                4,
                Some("f"),
            ),
            (
                "a.c",
                "#include <x.h>\nint f(int x) {\n  return x;\n}\n",
                1,
                None,
            ),
            (
                "a.cpp",
                "#include <x>\nclass A {\n  int m() { return 1; }\n};\nint A::n() {\n  return 2;\n}\n",
                3,
                Some("A.m"),
            ),
            (
                "a.cpp",
                "#include <x>\nclass A {\n  int m() { return 1; }\n};\nint A::n() {\n  return 2;\n}\n",
                6,
                Some("A::n"),
            ),
            (
                "A.scala",
                "package p\nobject A {\n  def m(x: Int): Int = x\n}\n",
                3,
                Some("A.m"),
            ),
            (
                "A.scala",
                "package p\nclass B[T] {\n  def m(x: Int): Int = x\n}\n",
                3,
                Some("B.m"),
            ),
        ];
        for (path, src, line, want) in cases {
            assert_eq!(symbol(path, src, line).as_deref(), want, "{path}:{line}");
        }
    }

    fn finding(path: &str, line: i64, category: Category, title: &str) -> Finding {
        Finding {
            path: path.to_owned(),
            start_line: line,
            end_line: line,
            severity: Severity::Major,
            category,
            title: title.to_owned(),
            body: String::new(),
            snippet: String::new(),
            suggestion: String::new(),
        }
    }

    fn map(path: &str, text: &str) -> HashMap<String, String> {
        HashMap::from([(path.to_owned(), text.to_owned())])
    }

    /// The Fingerprint of the single finding `f` against one file.
    fn fp(f: &Finding, patch: &str, source: Option<&str>) -> String {
        let sources = source.map(|s| map(&f.path, s)).unwrap_or_default();
        let out = fingerprint_findings(
            "org/repo",
            std::slice::from_ref(f),
            &map(&f.path, patch),
            &sources,
        );
        assert_eq!(out.len(), 1);
        out.into_iter().next().expect("one finding").0
    }

    const TS_V1: &str = "export function save(u: User) {\n  db.write(u);\n  return u.id;\n}\n";
    const TS_V1_PATCH: &str =
        "@@ -1,3 +1,4 @@\n export function save(u: User) {\n+  db.write(u);\n   return u.id;\n }";
    // Same function, a line added above the flagged one and the flagged line edited.
    const TS_V2: &str = "export function save(u: User) {\n  validate(u);\n  db.write(u, opts);\n  return u.id;\n}\n";
    const TS_V2_PATCH: &str = "@@ -1,3 +1,5 @@\n export function save(u: User) {\n+  validate(u);\n+  db.write(u, opts);\n   return u.id;\n }";

    #[test]
    fn stable_identity() {
        let base = finding("a.ts", 2, Category::ErrorHandling, "write error ignored");
        let reworded = Finding {
            title: "Unchecked failure from db.write".into(),
            body: "different prose".into(),
            ..base.clone()
        };
        let regraded = Finding {
            severity: Severity::Critical,
            ..base.clone()
        };
        let moved = Finding {
            start_line: 3,
            end_line: 3,
            ..base.clone()
        };
        let want = fp(&base, TS_V1_PATCH, Some(TS_V1));
        let cases = [
            ("reworded title", &reworded, TS_V1_PATCH, Some(TS_V1)),
            ("severity change", &regraded, TS_V1_PATCH, Some(TS_V1)),
            ("function edited", &moved, TS_V2_PATCH, Some(TS_V2)),
        ];
        for (name, f, patch, src) in cases {
            assert_eq!(fp(f, patch, src), want, "{name}");
        }
        let other = Finding {
            category: Category::Logic,
            ..base.clone()
        };
        assert_ne!(
            fp(&other, TS_V1_PATCH, Some(TS_V1)),
            want,
            "category is identity"
        );
        assert_ne!(
            fp(&base, TS_V1_PATCH, None),
            want,
            "no source: content anchor"
        );
    }

    #[test]
    fn content_anchor_reads_true_lines() {
        // Line 2 is a blank context line; the flagged line 3 is the added one.
        let patch =
            "@@ -1,3 +1,4 @@\n const a = 1;\n\n+const token = process.env.TOKEN;\n const b = 2;";
        let src = "const a = 1;\n\nconst token = process.env.TOKEN;\nconst b = 2;\n";
        let f = finding("cfg.ts", 3, Category::Security, "secret");
        let want = |line: &str| {
            fingerprint(
                "org/repo",
                "cfg.ts",
                &content_anchor(&[line]),
                Category::Security,
            )
        };
        // Top-level code has no symbol, with or without the source.
        assert_eq!(
            fp(&f, patch, Some(src)),
            want("const token = process.env.TOKEN;")
        );
        assert_eq!(
            fp(&f, patch, None),
            want("  const   token = process.env.TOKEN;")
        );
        assert_ne!(fp(&f, patch, None), want("const b = 2;"), "off-by-one");
    }

    #[test]
    fn colocated_findings() {
        let src = "func Run() {\n\ta := load()\n\tb := load()\n\tuse(a, b)\n}\n";
        let patch =
            "@@ -0,0 +1,5 @@\n+func Run() {\n+\ta := load()\n+\tb := load()\n+\tuse(a, b)\n+}";
        let a = finding("r.go", 2, Category::ErrorHandling, "a ignores error");
        let b = finding("r.go", 3, Category::ErrorHandling, "b ignores error");
        let a_again = finding(
            "r.go",
            2,
            Category::ErrorHandling,
            "reworded duplicate of a",
        );
        let c = finding("r.go", 4, Category::Logic, "wrong order");
        let run = |fs: &[Finding]| -> HashMap<String, String> {
            fingerprint_findings("org/repo", fs, &map("r.go", patch), &map("r.go", src))
                .into_iter()
                .map(|(fp, f)| (f.title.clone(), fp))
                .collect()
        };
        let fwd = run(&[a.clone(), b.clone(), a_again.clone(), c.clone()]);
        let rev = run(&[c, a_again, b, a]);
        assert_eq!(fwd.len(), 3, "exact duplicate collapses: {fwd:?}");
        assert_eq!(rev.len(), 3, "exact duplicate collapses: {rev:?}");
        let (fa, fb) = (&fwd["a ignores error"], &fwd["b ignores error"]);
        assert_ne!(fa, fb);
        assert_eq!(fb, &rev["b ignores error"], "order-independent");
        assert_eq!(
            fa, &rev["reworded duplicate of a"],
            "the duplicate has a's identity"
        );
        assert_eq!(fa.len(), 32 + 1 + 16, "salted: {fa}");
        assert_eq!(fwd["wrong order"].len(), 32, "a unique base stays bare");
    }

    /// Pinned bytes: changing the hash input re-posts every open finding, so it must
    /// be deliberate. Values from Python's `hashlib.blake2b(digest_size=…)`.
    #[test]
    fn pinned_hashes() {
        assert_eq!(content_anchor(&["\tx :=  1"]), "bf81218190fc1510");
        assert_eq!(
            fingerprint(" Org/Repo", "a.go", "sym:Foo", Category::NilSafety),
            "1bf6d08d32145f356373a0b951b23383"
        );
    }
}
