//! Source languages the reviewer parses with tree-sitter. A file is reviewed only in
//! one of them. Identity (enclosing symbol) and suggestion checking both start here,
//! so a file is never parsed with the wrong grammar in one place and the right one in
//! another. Context packs cover Go, TypeScript and JavaScript only; a file in another
//! language is reviewed from its diff alone.

use std::path::Path;

use tree_sitter::{Parser, Tree};

/// A language with a tree-sitter grammar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    Go,
    TypeScript,
    Tsx,
    JavaScript,
    Python,
    Java,
    Kotlin,
    CSharp,
    Php,
    Ruby,
    Rust,
    Swift,
    C,
    Cpp,
    Scala,
}

impl Lang {
    /// The language of `path` by extension, case-insensitively (macOS and Windows
    /// checkouts keep `.TS`), or `None` for files we don't parse. `.jsx` uses the
    /// JavaScript grammar, which already understands JSX. A `.h` header may be C or
    /// C++: the C++ grammar extends C's, so it parses both.
    pub fn for_path(path: &str) -> Option<Self> {
        let ext = Path::new(path).extension()?.to_str()?.to_ascii_lowercase();
        match ext.as_str() {
            "go" => Some(Self::Go),
            "tsx" => Some(Self::Tsx),
            "ts" | "mts" | "cts" => Some(Self::TypeScript),
            "js" | "jsx" | "mjs" | "cjs" => Some(Self::JavaScript),
            "py" => Some(Self::Python),
            "java" => Some(Self::Java),
            "kt" | "kts" => Some(Self::Kotlin),
            "cs" => Some(Self::CSharp),
            "php" => Some(Self::Php),
            "rb" | "rake" => Some(Self::Ruby),
            "rs" => Some(Self::Rust),
            "swift" => Some(Self::Swift),
            "c" => Some(Self::C),
            "cc" | "cpp" | "cxx" | "h" | "hh" | "hpp" | "hxx" => Some(Self::Cpp),
            "scala" | "sc" => Some(Self::Scala),
            _ => None,
        }
    }

    /// Parses `source`. Syntax errors don't fail the parse: they show up as
    /// `ERROR`/`MISSING` nodes (`has_error()`). `None` means the grammar could not be
    /// loaded, which a caller treats the same as an unsupported language.
    pub fn parse(self, source: &str) -> Option<Tree> {
        let grammar: tree_sitter::Language = match self {
            Self::Go => tree_sitter_go::LANGUAGE.into(),
            Self::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Self::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
            Self::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            Self::Python => tree_sitter_python::LANGUAGE.into(),
            Self::Java => tree_sitter_java::LANGUAGE.into(),
            Self::Kotlin => tree_sitter_kotlin_ng::LANGUAGE.into(),
            Self::CSharp => tree_sitter_c_sharp::LANGUAGE.into(),
            Self::Php => tree_sitter_php::LANGUAGE_PHP.into(),
            Self::Ruby => tree_sitter_ruby::LANGUAGE.into(),
            Self::Rust => tree_sitter_rust::LANGUAGE.into(),
            Self::Swift => tree_sitter_swift::LANGUAGE.into(),
            Self::C => tree_sitter_c::LANGUAGE.into(),
            Self::Cpp => tree_sitter_cpp::LANGUAGE.into(),
            Self::Scala => tree_sitter_scala::LANGUAGE.into(),
        };
        let mut parser = Parser::new();
        parser.set_language(&grammar).ok()?;
        parser.parse(source, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn for_path() {
        let cases = [
            ("main.go", Some(Lang::Go)),
            ("pkg/Server.GO", Some(Lang::Go)),
            ("app/C.tsx", Some(Lang::Tsx)),
            ("a.ts", Some(Lang::TypeScript)),
            ("types.d.ts", Some(Lang::TypeScript)),
            ("M.TS", Some(Lang::TypeScript)),
            ("a.mts", Some(Lang::TypeScript)),
            ("a.cts", Some(Lang::TypeScript)),
            ("a.js", Some(Lang::JavaScript)),
            ("a.jsx", Some(Lang::JavaScript)),
            ("a.mjs", Some(Lang::JavaScript)),
            ("a.cjs", Some(Lang::JavaScript)),
            ("app.py", Some(Lang::Python)),
            ("A.java", Some(Lang::Java)),
            ("a.kt", Some(Lang::Kotlin)),
            ("build.gradle.kts", Some(Lang::Kotlin)),
            ("A.cs", Some(Lang::CSharp)),
            ("index.php", Some(Lang::Php)),
            ("a.rb", Some(Lang::Ruby)),
            ("tasks.rake", Some(Lang::Ruby)),
            ("lib.rs", Some(Lang::Rust)),
            ("A.swift", Some(Lang::Swift)),
            ("a.c", Some(Lang::C)),
            ("a.h", Some(Lang::Cpp)),
            ("a.cpp", Some(Lang::Cpp)),
            ("a.hpp", Some(Lang::Cpp)),
            ("A.scala", Some(Lang::Scala)),
            ("stub.pyi", None),
            ("Makefile", None),
            ("dir.ts/README", None),
            ("go", None),
        ];
        for (path, want) in cases {
            assert_eq!(Lang::for_path(path), want, "{path}");
        }
    }

    #[test]
    fn parse() {
        let cases = [
            (Lang::Go, "package p\n\nfunc F() int { return 1 }\n", false),
            (Lang::Go, "package p\nfunc (((\n", true),
            (Lang::TypeScript, "const x: number = 1;\n", false),
            (Lang::TypeScript, "const x: number = (;\n", true),
            (Lang::Tsx, "const C = () => <div>{1}</div>;\n", false),
            (Lang::Tsx, "const C = () => <div>{1</div>;\n", true),
            (Lang::JavaScript, "const C = () => <div />;\n", false),
            (Lang::JavaScript, "function f( {\n", true),
            (Lang::Python, "def f(x):\n    return x\n", false),
            (Lang::Python, "def f(x:\n", true),
            (Lang::Java, "class A { int f() { return 1; } }\n", false),
            (Lang::Java, "class A { int f( { }\n", true),
            (Lang::Kotlin, "fun f(x: Int): Int = x\n", false),
            (Lang::Kotlin, "fun f(x: Int = \n", true),
            (Lang::CSharp, "class A { int F() { return 1; } }\n", false),
            (Lang::CSharp, "class A { int F( { }\n", true),
            (Lang::Php, "<?php\nfunction f($x) { return $x; }\n", false),
            (Lang::Php, "<?php\nfunction f($x { }\n", true),
            (Lang::Ruby, "def f(x)\n  x\nend\n", false),
            (Lang::Ruby, "def f(x\n", true),
            (Lang::Rust, "fn f(x: u8) -> u8 { x }\n", false),
            (Lang::Rust, "fn f(x: u8 -> u8 { x }\n", true),
            (Lang::Swift, "func f(x: Int) -> Int { return x }\n", false),
            (Lang::Swift, "func f(x: Int -> Int { return x }\n", true),
            (Lang::C, "int f(int x) { return x; }\n", false),
            (Lang::C, "int f(int x { return x; }\n", true),
            (
                Lang::Cpp,
                "class A { public: int f() { return 1; } };\n",
                false,
            ),
            (Lang::Cpp, "class A { int f( { };\n", true),
            (Lang::Scala, "object A { def f(x: Int): Int = x }\n", false),
            (Lang::Scala, "object A { def f(x: Int: Int = x }\n", true),
        ];
        for (lang, src, has_error) in cases {
            let tree = lang.parse(src).expect("grammar loads");
            assert_eq!(tree.root_node().has_error(), has_error, "{lang:?}: {src}");
        }
    }
}
