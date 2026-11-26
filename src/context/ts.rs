//! TS/JS context: tsconfig aliases, the resolver, export tables, definitions and
//! callers, in Rust with tree-sitter (only Go goes through a subprocess).
//!
//! [`build`] runs once per Review in `spawn_blocking`. It only reads the checkout,
//! through [`workspace::read`]'s rules, and checks the deadline between files. Its
//! output is byte-deterministic: repo-relative paths, `BTreeMap`s, sorted walks.
//!
//! Resolution follows TypeScript where real projects need it. `paths` takes the
//! wildcard with the longest prefix and never falls back to a shorter one: the
//! expo-app tree lists `@/*` first, and next-app's `@/hooks/*` must beat `@/*`.
//! Substitutions are relative to `baseUrl`, else to the file declaring `paths`
//! (next-app has no `baseUrl`), and a bare specifier tries `baseUrl` (js-app's
//! `"src"`).
//! Following relative imports only would lose every aliased import.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use regex::Regex;
use serde::Deserialize;
use tree_sitter::Node;

use super::{ContextPack, Snippet, SnippetKind};
use crate::diff::{DiffIndex, FileChange, FileStatus};
use crate::lang::Lang;
use crate::workspace;

/// A TS/JS file read for context; the caller scan skips bigger files.
const SOURCE_CAP: usize = 256 * 1024;
/// A tsconfig or jsconfig file.
const CONFIG_CAP: usize = 64 * 1024;
/// `extends` levels followed below the root config.
const MAX_EXTENDS: usize = 5;
/// Config paths tried in all, the root's included: bounds a wide `extends` array
/// as [`MAX_EXTENDS`] bounds a deep one.
const MAX_CONFIG_PATHS: usize = 16;
/// Substitutions tried per `paths` key (the repos list one).
const MAX_SUBSTITUTIONS: usize = 8;
/// Re-exports one lookup follows (the repos need at most 3).
const MAX_HOPS: usize = 4;
/// Files one lookup may open, `export *` fan-out included.
const MAX_LOOKUP_FILES: usize = 32;
/// A snippet's lines, its JSDoc included.
const SNIPPET_LINES: i64 = 40;
/// Changed exports of one file whose callers are looked for.
const MAX_NAMES: usize = 8;
/// The caller scan reads at most this many files...
const SCAN_FILES: usize = 10_000;
/// ...and this many bytes.
const SCAN_BYTES: u64 = 64 << 20;
/// Candidates parsed per name, confirmed or not.
const PARSED_PER_NAME: usize = 50;
const FILES_PER_NAME: usize = 6;
const SITES_PER_FILE: usize = 2;
const CALLERS_PER_FILE: usize = 8;
/// A caller site shows its use line from 2 before to 15 after.
const SITE_BEFORE: i64 = 2;
const SITE_AFTER: i64 = 15;

/// Directories the caller scan never enters: dependencies, output, native
/// projects and tests.
const SKIP_DIRS: [&str; 9] = [
    "node_modules",
    ".git",
    ".next",
    "dist",
    "build",
    "coverage",
    "ios",
    "android",
    "__tests__",
];
/// Extensions an extensionless specifier is tried with, then `index` with each.
const PROBE_EXTS: [&str; 5] = [".ts", ".tsx", ".d.ts", ".js", ".jsx"];
/// Specifiers ending in these name data or assets, never code.
const ASSET_EXTS: [&str; 26] = [
    "json", "css", "scss", "sass", "less", "svg", "png", "jpg", "jpeg", "gif", "webp", "avif",
    "ico", "bmp", "mp3", "mp4", "wav", "webm", "ttf", "otf", "woff", "woff2", "eot", "txt", "md",
    "html",
];
const DEADLINE: &str = "deadline exceeded";

/// One Review's TS/JS context, for `context::build`.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Output {
    /// A pack per reviewed TS/JS file that is not removed and has a patch:
    /// snippets in rank order (definitions, callers, then second-tier
    /// definitions), `error` when partial. `related` is left for `context::build`
    /// to fill from `imports`.
    pub packs: BTreeMap<String, ContextPack>,
    /// Every PR TS/JS file in the tree, reviewed or not, mapped to the repo files
    /// its import and re-export specifiers resolve to: related diffs go both ways.
    pub imports: BTreeMap<String, BTreeSet<String>>,
    /// The reviewed files' code specifiers, `(resolved, all)`: relative ones, those
    /// matching a `paths` key, and bare ones whose first segment is under
    /// `baseUrl`; data and assets excluded. `FixtureContext::ts_imports`.
    pub specifiers: (u32, u32),
}

/// Builds the TS/JS side of one Review's context from the checkout at `root`.
///
/// `pr` is every file of the pull request and `review` the paths that get a pack;
/// non-TS/JS paths in either are ignored. Blocking: `context::build` calls it in
/// `spawn_blocking` with `Instant::now() + STRUCTURAL_TIMEOUT`. The deadline is
/// checked between files; a file reached after it gets `error:
/// "deadline exceeded"`, and the imports of files not reached are left out.
pub fn build(root: &Path, pr: &[FileChange], review: &[&str], deadline: Instant) -> Output {
    let mut repo = Repo::new(root);
    let mut live: Vec<&FileChange> = pr
        .iter()
        .filter(|f| f.status != FileStatus::Removed && ts_lang(&f.path).is_some())
        .collect();
    live.sort_by(|a, b| a.path.cmp(&b.path));
    live.dedup_by(|a, b| a.path == b.path);
    let reviewed: Vec<&FileChange> = live
        .iter()
        .copied()
        .filter(|f| !f.patch.is_empty() && review.contains(&f.path.as_str()))
        .collect();

    // What each reviewed file changed that others may call; one scan for all.
    let mut names = BTreeMap::new();
    for f in &reviewed {
        if Instant::now() >= deadline {
            break;
        }
        let changed = DiffIndex::parse(&f.patch).changed();
        names.insert(f.path.as_str(), repo.changed_names(&f.path, &changed));
    }
    let words: BTreeSet<&str> = names.values().flatten().map(|n| n.word.as_str()).collect();
    let mentions = if words.is_empty() {
        Mentions::default()
    } else {
        repo.mentions(&words, deadline)
    };

    let mut out = Output::default();
    for f in &reviewed {
        let pack = match names.get(f.path.as_str()) {
            Some(names) if Instant::now() < deadline => repo.pack(f, names, &mentions),
            _ => ContextPack {
                error: Some(DEADLINE.to_owned()),
                ..ContextPack::default()
            },
        };
        out.packs.insert(f.path.clone(), pack);
    }
    for f in &live {
        if Instant::now() >= deadline {
            break;
        }
        let (files, resolved, code) = repo.imports(&f.path);
        if reviewed.iter().any(|r| r.path == f.path) {
            out.specifiers.0 = out.specifiers.0.saturating_add(resolved);
            out.specifiers.1 = out.specifiers.1.saturating_add(code);
        }
        out.imports.insert(f.path.clone(), files);
    }
    out
}

/// The tree-sitter grammar of a TS/JS path; `None` for every other language.
fn ts_lang(path: &str) -> Option<Lang> {
    Lang::for_path(path).filter(|l| matches!(l, Lang::TypeScript | Lang::Tsx | Lang::JavaScript))
}

// ---- Config ----------------------------------------------------------------

/// The two compiler options resolution needs, after `extends`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Config {
    /// `baseUrl`, repo-relative (`""` is the root).
    base_url: Option<String>,
    /// `paths` in declaration order: ties go to the first.
    paths: Vec<(String, Vec<String>)>,
    /// The directory of the file that declared `paths`, which substitutions are
    /// relative to when there is no `baseUrl`.
    paths_dir: String,
}

#[derive(Deserialize)]
struct RawConfig {
    #[serde(default)]
    extends: Option<Extends>,
    #[serde(default, rename = "compilerOptions")]
    options: RawOptions,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Extends {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize, Default)]
struct RawOptions {
    #[serde(default, rename = "baseUrl")]
    base_url: Option<String>,
    #[serde(default)]
    paths: Option<Paths>,
}

/// `paths` with its keys in file order, which `serde_json::Map` would sort.
struct Paths(Vec<(String, Vec<String>)>);

impl<'de> Deserialize<'de> for Paths {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct Ordered;
        impl<'de> serde::de::Visitor<'de> for Ordered {
            type Value = Paths;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a map of path patterns to substitutions")
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<Paths, A::Error> {
                let mut entries = Vec::new();
                while let Some(entry) = map.next_entry()? {
                    entries.push(entry);
                }
                Ok(Paths(entries))
            }
        }
        d.deserialize_map(Ordered)
    }
}

/// The root's `tsconfig.json`, else `jsconfig.json`, with why it could not be
/// used. Strict JSON: a parse failure leaves relative imports only.
fn load_config(root: &Path) -> (Config, Option<String>) {
    for name in ["tsconfig.json", "jsconfig.json"] {
        let Some(text) = workspace::read(root, name, CONFIG_CAP) else {
            continue;
        };
        let mut cfg = Config::default();
        let mut seen = BTreeSet::from([name.to_owned()]);
        return match apply(root, name, &text, 0, &mut seen, &mut cfg) {
            Ok(()) => (cfg, None),
            Err(e) => (Config::default(), Some(format!("{name}: {e}"))),
        };
    }
    (Config::default(), None)
}

/// Applies config file `path` onto `cfg`: what it extends first, in order (a
/// later base wins), then its own options key by key; `paths` is one key, so a
/// child's replaces a base's whole. Only relative `extends` inside the root are
/// followed, each path tried once, [`MAX_EXTENDS`] levels deep and
/// [`MAX_CONFIG_PATHS`] in all; a base that can't be read or parsed is skipped
/// (expo-app extends `expo/tsconfig.base`, absent in a checkout and without
/// `paths`).
fn apply(
    root: &Path,
    path: &str,
    text: &str,
    depth: usize,
    seen: &mut BTreeSet<String>,
    cfg: &mut Config,
) -> Result<(), serde_json::Error> {
    let raw: RawConfig = serde_json::from_str(text)?;
    let dir = super::dir(path);
    let bases = match raw.extends {
        None => Vec::new(),
        Some(Extends::One(s)) => vec![s],
        Some(Extends::Many(v)) => v,
    };
    for spec in bases.iter().filter(|s| depth < MAX_EXTENDS && relative(s)) {
        let Some(base) = join(dir, spec) else {
            continue;
        };
        if seen.len() >= MAX_CONFIG_PATHS {
            break;
        }
        // Marked before reading, so a repeated entry costs no read.
        if !seen.insert(base.clone()) {
            continue;
        }
        // TypeScript tries the path as written, then with `.json`.
        let found = [base.clone(), format!("{base}.json")]
            .into_iter()
            .find_map(|p| Some((workspace::read(root, &p, CONFIG_CAP)?, p)));
        if let Some((text, found)) = found
            && (found == base || seen.insert(found.clone()))
        {
            let _skipped_if_broken = apply(root, &found, &text, depth + 1, seen, cfg);
        }
    }
    if let Some(base_url) = raw.options.base_url.and_then(|b| join(dir, &b)) {
        cfg.base_url = Some(base_url);
    }
    if let Some(paths) = raw.options.paths {
        cfg.paths = paths.0;
        dir.clone_into(&mut cfg.paths_dir);
    }
    Ok(())
}

impl Config {
    /// The `paths` entry `spec` matches and what its `*` captured: an exact key
    /// first, else the wildcard with the longest prefix, the first on a tie.
    fn matching<'s>(&self, spec: &'s str) -> Option<(&[String], &'s str)> {
        if let Some((_, subs)) = self.paths.iter().find(|(k, _)| k == spec) {
            return Some((subs, ""));
        }
        let mut best: Option<(usize, &[String], &'s str)> = None;
        for (key, subs) in &self.paths {
            let Some((prefix, suffix)) = key.split_once('*') else {
                continue;
            };
            let fits = !suffix.contains('*')
                && spec.len() >= prefix.len() + suffix.len()
                && spec.starts_with(prefix)
                && spec.ends_with(suffix);
            if fits && best.is_none_or(|(len, ..)| prefix.len() > len) {
                let star = &spec[prefix.len()..spec.len() - suffix.len()];
                best = Some((prefix.len(), subs, star));
            }
        }
        best.map(|(_, subs, star)| (subs, star))
    }

    /// What `paths` substitutions are relative to.
    fn paths_base(&self) -> &str {
        self.base_url.as_deref().unwrap_or(&self.paths_dir)
    }
}

// ---- Paths and reading -----------------------------------------------------

/// TypeScript's relative specifier: `.`, `..`, or starting `./` or `../`.
fn relative(spec: &str) -> bool {
    matches!(spec, "." | "..") || spec.starts_with("./") || spec.starts_with("../")
}

/// Whether `spec` names data or an asset (`./en.json`, `./logo.png`).
fn is_asset(spec: &str) -> bool {
    let last = spec.rsplit('/').next().unwrap_or(spec);
    last.rsplit_once('.')
        .is_some_and(|(_, ext)| ASSET_EXTS.contains(&ext.to_ascii_lowercase().as_str()))
}

/// `dir/rel` as a clean repo-relative path, or `None` when it climbs out of the
/// root or `rel` is absolute.
pub(crate) fn join(dir: &str, rel: &str) -> Option<String> {
    if rel.starts_with('/') {
        return None;
    }
    let mut parts: Vec<&str> = dir.split('/').filter(|s| !s.is_empty()).collect();
    for seg in rel.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop()?;
            }
            s => parts.push(s),
        }
    }
    Some(parts.join("/"))
}

/// Lines `start..=end` (1-based) of `src`.
fn lines(src: &str, start: i64, end: i64) -> String {
    let skip = usize::try_from(start.saturating_sub(1)).unwrap_or(0);
    let take = usize::try_from(end - start + 1).unwrap_or(0);
    src.lines()
        .skip(skip)
        .take(take)
        .collect::<Vec<_>>()
        .join("\n")
}

// ---- Syntax ------------------------------------------------------------------

/// What a definition is; the variants are in rank order, first ranked first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Kind {
    Type,
    Function,
    Value,
}

/// A declaration's lines, the JSDoc above it included; not yet clipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Decl {
    kind: Kind,
    start: i64,
    end: i64,
}

fn text<'s>(node: Node<'_>, src: &'s str) -> &'s str {
    node.utf8_text(src.as_bytes()).unwrap_or("")
}

fn unquote(s: &str) -> &str {
    s.trim_matches(|c| matches!(c, '"' | '\'' | '`'))
}

/// 1-based line of a tree-sitter row.
fn line_of(row: usize) -> i64 {
    i64::try_from(row).map_or(i64::MAX, |r| r + 1)
}

fn line(node: Node<'_>) -> i64 {
    line_of(node.start_position().row)
}

/// Visits `node` and its named descendants in source order, without recursion
/// (a PR controls the nesting depth); `visit` says whether to enter a node.
fn walk<'t>(node: Node<'t>, mut visit: impl FnMut(Node<'t>) -> bool) {
    let mut stack = vec![node];
    while let Some(n) = stack.pop() {
        if visit(n) {
            let mut cursor = n.walk();
            let children: Vec<Node<'t>> = n.named_children(&mut cursor).collect();
            stack.extend(children.into_iter().rev());
        }
    }
}

/// The identifier-like nodes under `node` in source order, outside import
/// statements and export lists: there they are bindings, not uses.
fn uses(node: Node<'_>) -> Vec<Node<'_>> {
    let mut out = Vec::new();
    walk(node, |n| match n.kind() {
        "import_statement" | "export_clause" => false,
        "identifier" | "type_identifier" | "shorthand_property_identifier" => {
            out.push(n);
            false
        }
        _ => true,
    });
    out
}

/// For `ns` in `ns.x` (or the type `ns.T`), the member's name.
fn member(node: Node<'_>, src: &str) -> Option<String> {
    let parent = node.parent()?;
    let (object, property) = match parent.kind() {
        "member_expression" => ("object", "property"),
        "nested_type_identifier" => ("module", "name"),
        _ => return None,
    };
    if parent.child_by_field_name(object)? != node {
        return None;
    }
    Some(text(parent.child_by_field_name(property)?, src).to_owned())
}

/// The names a declaration binds, each with its kind and, for a variable, the
/// value it is bound to. Destructuring binds every name in the pattern.
fn declared(node: Node<'_>) -> Vec<(Node<'_>, Kind, Option<Node<'_>>)> {
    let kind = match node.kind() {
        "interface_declaration" | "type_alias_declaration" | "enum_declaration" => Kind::Type,
        "function_declaration"
        | "generator_function_declaration"
        | "function_signature"
        | "class_declaration"
        | "abstract_class_declaration" => Kind::Function,
        "lexical_declaration" | "variable_declaration" => {
            let mut cursor = node.walk();
            let declarators: Vec<Node<'_>> = node
                .named_children(&mut cursor)
                .filter(|d| d.kind() == "variable_declarator")
                .collect();
            return declarators
                .into_iter()
                .flat_map(|d| {
                    let value = d.child_by_field_name("value");
                    let names = d.child_by_field_name("name").map(pattern_names);
                    let names = names.unwrap_or_default();
                    names.into_iter().map(move |n| (n, Kind::Value, value))
                })
                .collect();
        }
        _ => return Vec::new(),
    };
    node.child_by_field_name("name")
        .map(|n| vec![(n, kind, None)])
        .unwrap_or_default()
}

/// The identifiers a declarator's name binds: itself, or every name of a
/// destructuring pattern (`const { a, b: c, ...d } = x`), defaults excluded.
fn pattern_names(pattern: Node<'_>) -> Vec<Node<'_>> {
    let mut out = Vec::new();
    let mut stack = vec![pattern];
    while let Some(n) = stack.pop() {
        match n.kind() {
            "identifier" | "shorthand_property_identifier_pattern" => out.push(n),
            "assignment_pattern" | "object_assignment_pattern" => {
                stack.extend(n.child_by_field_name("left"));
            }
            "pair_pattern" => stack.extend(n.child_by_field_name("value")),
            "object_pattern" | "array_pattern" | "rest_pattern" => {
                let mut cursor = n.walk();
                let children: Vec<Node<'_>> = n.named_children(&mut cursor).collect();
                stack.extend(children.into_iter().rev());
            }
            _ => {}
        }
    }
    out
}

/// The statement a declaration belongs to: `export` and `declare` included.
fn statement(node: Node<'_>) -> Node<'_> {
    let mut stmt = node;
    while let Some(p) = stmt
        .parent()
        .filter(|p| matches!(p.kind(), "export_statement" | "ambient_declaration"))
    {
        stmt = p;
    }
    stmt
}

/// The lines of declaration statement `stmt`, from the JSDoc right above it.
fn decl(stmt: Node<'_>, kind: Kind, src: &str) -> Decl {
    let doc = stmt.prev_named_sibling().filter(|c| {
        c.kind() == "comment"
            && text(*c, src).starts_with("/**")
            && c.end_position().row + 1 >= stmt.start_position().row
    });
    Decl {
        kind,
        start: line(doc.unwrap_or(stmt)),
        end: line_of(stmt.end_position().row),
    }
}

/// A function, arrow function or class value is a function definition; the rest
/// are values.
fn value_kind(node: Node<'_>) -> Kind {
    match node.kind() {
        "function_expression" | "function" | "arrow_function" | "generator_function" | "class" => {
            Kind::Function
        }
        _ => Kind::Value,
    }
}

// ---- Modules: imports and the export table ---------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
enum Imported {
    Default,
    Named(String),
    Namespace,
}

#[derive(Debug)]
struct Import {
    spec: String,
    /// Local name and what it imports.
    bindings: Vec<(String, Imported)>,
}

/// Where an exported name comes from.
#[derive(Debug)]
enum Export {
    /// A local name: a top-level declaration, or an import re-exported by an
    /// export list.
    Local(String),
    /// `export { name as … } from 'spec'`.
    From { spec: String, name: String },
    /// An anonymous `export default …`.
    Own(Decl),
}

/// One file's imports, export table and top-level declarations, built once per
/// Review.
#[derive(Debug)]
struct Module {
    src: String,
    imports: Vec<Import>,
    /// Every import and re-export specifier, in order.
    specs: Vec<String>,
    exports: BTreeMap<String, Export>,
    /// `export * from` specifiers, in order.
    stars: Vec<String>,
    /// Top-level declared names, destructuring included (next-app's
    /// `create-navigation.ts` exports `useRouter` that way).
    decls: BTreeMap<String, Decl>,
}

impl Module {
    fn load(root: &Path, path: &str) -> Option<Module> {
        let lang = ts_lang(path)?;
        let src = workspace::read(root, path, SOURCE_CAP)?;
        let tree = lang.parse(&src)?;
        let mut m = Module {
            src: String::new(),
            imports: Vec::new(),
            specs: Vec::new(),
            exports: BTreeMap::new(),
            stars: Vec::new(),
            decls: BTreeMap::new(),
        };
        let top = tree.root_node();
        let mut cursor = top.walk();
        for node in top.named_children(&mut cursor) {
            match node.kind() {
                "import_statement" => m.import(node, &src),
                "export_statement" => m.export(node, &src),
                _ => {
                    m.declare(node, node, &src);
                }
            }
        }
        m.src = src;
        Some(m)
    }

    fn import(&mut self, node: Node<'_>, src: &str) {
        let Some(source) = node.child_by_field_name("source") else {
            return;
        };
        let spec = unquote(text(source, src)).to_owned();
        let mut bindings = Vec::new();
        let mut cursor = node.walk();
        let clauses: Vec<Node<'_>> = node
            .named_children(&mut cursor)
            .filter(|c| c.kind() == "import_clause")
            .collect();
        for clause in clauses {
            let mut cursor = clause.walk();
            for part in clause.named_children(&mut cursor) {
                match part.kind() {
                    "identifier" => bindings.push((text(part, src).to_owned(), Imported::Default)),
                    "namespace_import" => {
                        if let Some(id) = part.named_child(0) {
                            bindings.push((text(id, src).to_owned(), Imported::Namespace));
                        }
                    }
                    "named_imports" => {
                        let mut cursor = part.walk();
                        for s in part.named_children(&mut cursor) {
                            let Some(name) = s.child_by_field_name("name") else {
                                continue;
                            };
                            let local = s.child_by_field_name("alias").unwrap_or(name);
                            let name = unquote(text(name, src)).to_owned();
                            bindings.push((text(local, src).to_owned(), Imported::Named(name)));
                        }
                    }
                    _ => {}
                }
            }
        }
        self.specs.push(spec.clone());
        self.imports.push(Import { spec, bindings });
    }

    fn export(&mut self, node: Node<'_>, src: &str) {
        let source = node
            .child_by_field_name("source")
            .map(|s| unquote(text(s, src)).to_owned());
        if let Some(spec) = &source {
            self.specs.push(spec.clone());
        }
        let mut cursor = node.walk();
        let default = node.children(&mut cursor).any(|c| c.kind() == "default");
        if let Some(d) = node.child_by_field_name("declaration") {
            let names = self.declare(d, node, src);
            if default {
                let export = match names.first() {
                    Some(name) => Export::Local(name.clone()),
                    None => Export::Own(decl(node, value_kind(d), src)),
                };
                self.exports.entry("default".to_owned()).or_insert(export);
            } else {
                for name in names {
                    self.exports
                        .entry(name.clone())
                        .or_insert(Export::Local(name));
                }
            }
            return;
        }
        if default {
            if let Some(v) = node.child_by_field_name("value") {
                let export = match v.kind() {
                    "identifier" => Export::Local(text(v, src).to_owned()),
                    _ => Export::Own(decl(node, value_kind(v), src)),
                };
                self.exports.entry("default".to_owned()).or_insert(export);
            }
            return;
        }
        let mut cursor = node.walk();
        let children: Vec<Node<'_>> = node.named_children(&mut cursor).collect();
        if children.iter().any(|c| c.kind() == "namespace_export") {
            return; // `export * as ns from`: a namespace, not followed
        }
        let Some(clause) = children.iter().find(|c| c.kind() == "export_clause") else {
            if let Some(spec) = source {
                self.stars.push(spec);
            }
            return;
        };
        let mut cursor = clause.walk();
        for s in clause.named_children(&mut cursor) {
            let Some(name) = s.child_by_field_name("name") else {
                continue;
            };
            let name = unquote(text(name, src)).to_owned();
            let alias = s
                .child_by_field_name("alias")
                .map_or_else(|| name.clone(), |a| unquote(text(a, src)).to_owned());
            let export = match &source {
                Some(spec) => Export::From {
                    spec: spec.clone(),
                    name,
                },
                None => Export::Local(name),
            };
            self.exports.entry(alias).or_insert(export);
        }
    }

    /// Records the top-level names `node` declares, as statement `stmt`.
    fn declare(&mut self, node: Node<'_>, stmt: Node<'_>, src: &str) -> Vec<String> {
        let node = match node.kind() {
            "ambient_declaration" => match node.named_child(0) {
                Some(inner) => inner,
                None => return Vec::new(),
            },
            _ => node,
        };
        let mut names = Vec::new();
        for (name, kind, _) in declared(node) {
            let name = text(name, src).to_owned();
            self.decls
                .entry(name.clone())
                .or_insert_with(|| decl(stmt, kind, src));
            names.push(name);
        }
        names
    }

    /// The import binding local name `local`, with its specifier.
    fn binding(&self, local: &str) -> Option<(&str, &Imported)> {
        self.imports.iter().find_map(|i| {
            let (_, imported) = i.bindings.iter().find(|(l, _)| l == local)?;
            Some((i.spec.as_str(), imported))
        })
    }
}

/// A declaration a snippet can show, and the name it was found by.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Def {
    path: String,
    symbol: String,
    decl: Decl,
}

/// A declaration of the reviewed file at any depth, scoped by the block that
/// holds it.
struct Local<'t> {
    name: String,
    name_line: i64,
    decl: Decl,
    scope: Range<usize>,
    value: Option<Node<'t>>,
}

/// Every declaration in `top`'s tree, nested ones included (a PR that calls a
/// nested same-file function had the model call it undefined).
fn locals<'t>(top: Node<'t>, src: &str) -> Vec<Local<'t>> {
    let mut out = Vec::new();
    walk(top, |node| {
        for (name, kind, value) in declared(node) {
            let stmt = statement(node);
            out.push(Local {
                name: text(name, src).to_owned(),
                name_line: line(name),
                decl: decl(stmt, kind, src),
                scope: stmt.parent().unwrap_or(top).byte_range(),
                value,
            });
        }
        true
    });
    out
}

/// The declaration of `name` in scope at byte `at`: the innermost one.
fn innermost<'a, 't>(locals: &'a [Local<'t>], name: &str, at: usize) -> Option<&'a Local<'t>> {
    locals
        .iter()
        .filter(|l| l.name == name && l.scope.contains(&at))
        .min_by_key(|l| l.scope.len())
}

/// Definitions kept once per declaration, at their first use.
#[derive(Default)]
struct Found(BTreeMap<(String, i64), (Use, Def)>);

/// Where a definition was first used: added lines rank before context lines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Use {
    context: bool,
    line: i64,
    column: usize,
}

impl Found {
    fn add(&mut self, def: Def, at: Use) {
        let key = (def.path.clone(), def.decl.start);
        match self.0.get_mut(&key) {
            Some(kept) if at < kept.0 => *kept = (at, def),
            Some(_) => {}
            None => {
                self.0.insert(key, (at, def));
            }
        }
    }

    /// Rank order: by kind, then first use, then (path, line).
    fn sorted(self) -> Vec<Def> {
        let mut all: Vec<(Use, Def)> = self.0.into_values().collect();
        all.sort_by(|(ua, a), (ub, b)| {
            (a.decl.kind, ua, &a.path, a.decl.start).cmp(&(b.decl.kind, ub, &b.path, b.decl.start))
        });
        all.into_iter().map(|(_, d)| d).collect()
    }
}

/// An export of a changed file whose declaration holds a changed line.
#[derive(Debug, Clone)]
struct Name {
    exported: String,
    /// What callers' text must contain: the exported name, or a default
    /// export's local name.
    word: String,
    decl: Decl,
}

/// Which files of the tree mention each name, and why the scan stopped early.
#[derive(Debug, Default)]
struct Mentions {
    files: BTreeMap<String, Vec<String>>,
    error: Option<String>,
}

// ---- The per-Review resolver -------------------------------------------------

/// One Review's resolver: the config, and caches by (directory, specifier) and
/// by file.
struct Repo {
    root: PathBuf,
    config: Config,
    config_error: Option<String>,
    resolved: HashMap<(String, String), Option<String>>,
    modules: HashMap<String, Option<Rc<Module>>>,
}

impl Repo {
    fn new(root: &Path) -> Repo {
        let (config, config_error) = load_config(root);
        Repo {
            root: root.to_owned(),
            config,
            config_error,
            resolved: HashMap::new(),
            modules: HashMap::new(),
        }
    }

    fn module(&mut self, path: &str) -> Option<Rc<Module>> {
        if let Some(m) = self.modules.get(path) {
            return m.clone();
        }
        let m = Module::load(&self.root, path).map(Rc::new);
        self.modules.insert(path.to_owned(), m.clone());
        m
    }

    /// The repo file `spec`, imported from file `from`, resolves to: relative;
    /// `paths` (exact key, else the longest wildcard prefix, its substitutions in
    /// order, no shorter fallback); `baseUrl`; else a package, `None`.
    fn resolve(&mut self, from: &str, spec: &str) -> Option<String> {
        let key = (super::dir(from).to_owned(), spec.to_owned());
        if let Some(hit) = self.resolved.get(&key) {
            return hit.clone();
        }
        let found = self.resolve_uncached(&key.0, spec);
        self.resolved.insert(key, found.clone());
        found
    }

    fn resolve_uncached(&self, dir: &str, spec: &str) -> Option<String> {
        if is_asset(spec) {
            return None;
        }
        if relative(spec) {
            return self.probe(&join(dir, spec)?);
        }
        if let Some((subs, star)) = self.config.matching(spec) {
            let base = self.config.paths_base();
            let hit = subs
                .iter()
                .take(MAX_SUBSTITUTIONS)
                .find_map(|s| self.probe(&join(base, &s.replacen('*', star, 1))?));
            if hit.is_some() {
                return hit;
            }
        }
        let base = self.config.base_url.as_deref()?;
        self.probe(&join(base, spec)?)
    }

    /// The file `path` names: itself when it has a TS/JS extension, then with
    /// each of [`PROBE_EXTS`], then `index` with each, inside the root.
    fn probe(&self, path: &str) -> Option<String> {
        let explicit = ts_lang(path).map(|_| path.to_owned());
        let files = PROBE_EXTS.iter().map(|e| format!("{path}{e}"));
        let index = PROBE_EXTS.iter().map(|e| match path {
            "" => format!("index{e}"),
            _ => format!("{path}/index{e}"),
        });
        explicit.into_iter().chain(files).chain(index).find(|c| {
            workspace::contained(&self.root, c)
                .and_then(|p| fs::metadata(p).ok())
                .is_some_and(|m| m.is_file())
        })
    }

    /// Whether `spec` names code in the repo, resolvable or not: what the
    /// resolution rate counts.
    fn is_code(&self, spec: &str) -> bool {
        if is_asset(spec) {
            return false;
        }
        if relative(spec) || self.config.matching(spec).is_some() {
            return true;
        }
        let Some(base) = &self.config.base_url else {
            return false;
        };
        let first = spec.split('/').next().unwrap_or(spec);
        join(base, first).is_some_and(|p| {
            workspace::contained(&self.root, &p).is_some() || self.probe(&p).is_some()
        })
    }

    /// The repo files `path` imports or re-exports from, and its code
    /// specifiers: (resolved, all).
    fn imports(&mut self, path: &str) -> (BTreeSet<String>, u32, u32) {
        let mut files = BTreeSet::new();
        let (mut resolved, mut code) = (0u32, 0u32);
        let Some(m) = self.module(path) else {
            return (files, 0, 0);
        };
        for spec in &m.specs {
            let found = self.resolve(path, spec);
            if self.is_code(spec) {
                code = code.saturating_add(1);
                resolved = resolved.saturating_add(u32::from(found.is_some()));
            }
            files.extend(found);
        }
        (files, resolved, code)
    }

    /// Where `name`, exported by `path`, is declared, following export lists,
    /// `export … from` and `export *` at most [`MAX_HOPS`] re-exports and
    /// [`MAX_LOOKUP_FILES`] files deep. `"default"` is the default export.
    fn lookup(&mut self, path: &str, name: &str) -> Option<Def> {
        let mut seen = BTreeSet::new();
        let mut files = BTreeSet::new();
        self.lookup_in(path, name, 0, &mut seen, &mut files)
    }

    fn lookup_in(
        &mut self,
        path: &str,
        name: &str,
        hops: usize,
        seen: &mut BTreeSet<(String, String)>,
        files: &mut BTreeSet<String>,
    ) -> Option<Def> {
        if hops > MAX_HOPS || !seen.insert((path.to_owned(), name.to_owned())) {
            return None;
        }
        files.insert(path.to_owned());
        if files.len() > MAX_LOOKUP_FILES {
            return None;
        }
        let m = self.module(path)?;
        let found = |symbol: &str, decl: Decl| Def {
            path: path.to_owned(),
            symbol: symbol.to_owned(),
            decl,
        };
        let (spec, next) = match m.exports.get(name) {
            Some(Export::Own(decl)) => return Some(found(name, *decl)),
            Some(Export::Local(local)) => {
                if let Some(decl) = m.decls.get(local) {
                    return Some(found(local, *decl));
                }
                match m.binding(local)? {
                    (spec, Imported::Named(n)) => (spec, n.as_str()),
                    (spec, Imported::Default) => (spec, "default"),
                    (_, Imported::Namespace) => return None,
                }
            }
            Some(Export::From { spec, name }) => (spec.as_str(), name.as_str()),
            None if name == "default" => return None,
            None => {
                for spec in &m.stars {
                    let Some(target) = self.resolve(path, spec) else {
                        continue;
                    };
                    if let Some(def) = self.lookup_in(&target, name, hops + 1, seen, files) {
                        return Some(def);
                    }
                }
                return None;
            }
        };
        let target = self.resolve(path, spec)?;
        self.lookup_in(&target, next, hops + 1, seen, files)
    }

    /// The declaration use `node` of file `path` refers to, when it is off the
    /// diff's visible lines: the innermost same-file declaration in scope, else
    /// an import followed to its declaration (default by `"default"`, `ns.x` by
    /// `x`). A same-file one comes back as a [`Local`] too, for tier 2.
    fn refers<'a, 't>(
        &mut self,
        path: &str,
        m: &Module,
        locals: &'a [Local<'t>],
        node: Node<'t>,
        diff: &DiffIndex,
    ) -> Option<(Def, Option<&'a Local<'t>>)> {
        let name = text(node, &m.src);
        if let Some(local) = innermost(locals, name, node.start_byte()) {
            if diff.has(local.name_line) {
                return None;
            }
            let def = Def {
                path: path.to_owned(),
                symbol: name.to_owned(),
                decl: local.decl,
            };
            return Some((def, Some(local)));
        }
        let (spec, imported) = m.binding(name)?;
        let export = match imported {
            Imported::Named(n) => n.clone(),
            Imported::Default => "default".to_owned(),
            Imported::Namespace => member(node, &m.src)?,
        };
        let target = self.resolve(path, spec)?;
        self.lookup(&target, &export).map(|d| (d, None))
    }

    /// Tier 1: what the identifiers, type identifiers, shorthand properties and
    /// `ns.x` on the diff's visible lines refer to. Tier 2: for a tier-1
    /// same-file variable, what its value's identifiers refer to, one hop
    /// (`const session = useRegisterSession()` reaches `useRegisterSession`).
    fn definitions(&mut self, path: &str, m: &Module, diff: &DiffIndex) -> (Vec<Def>, Vec<Def>) {
        let Some(tree) = ts_lang(path).and_then(|l| l.parse(&m.src)) else {
            return (Vec::new(), Vec::new());
        };
        let top = tree.root_node();
        let locals = locals(top, &m.src);
        let (mut tier1, mut tier2) = (Found::default(), Found::default());
        let mut followed = BTreeSet::new();
        for node in uses(top) {
            let line = line(node);
            if !diff.has(line) {
                continue;
            }
            let Some((def, local)) = self.refers(path, m, &locals, node, diff) else {
                continue;
            };
            let context = !diff.is_added(line);
            let column = node.start_position().column;
            tier1.add(
                def,
                Use {
                    context,
                    line,
                    column,
                },
            );
            let Some(value) = local.and_then(|l| l.value) else {
                continue;
            };
            if !followed.insert(value.id()) {
                continue;
            }
            for rhs in uses(value) {
                if let Some((def, _)) = self.refers(path, m, &locals, rhs, diff) {
                    let at = Use {
                        context,
                        line: line_of(rhs.start_position().row),
                        column: rhs.start_position().column,
                    };
                    tier2.add(def, at);
                }
            }
        }
        (tier1.sorted(), tier2.sorted())
    }

    fn snippet(&mut self, def: &Def) -> Option<Snippet> {
        let m = self.module(&def.path)?;
        let end = def.decl.end.min(def.decl.start + SNIPPET_LINES - 1);
        Some(Snippet {
            kind: SnippetKind::Definition,
            symbol: def.symbol.clone(),
            path: def.path.clone(),
            start_line: def.decl.start,
            end_line: end,
            text: lines(&m.src, def.decl.start, end),
        })
    }

    /// What file `path` exports whose declaration holds a `changed` line, in
    /// declaration order, at most [`MAX_NAMES`].
    fn changed_names(&mut self, path: &str, changed: &BTreeSet<i64>) -> Vec<Name> {
        let Some(m) = self.module(path) else {
            return Vec::new();
        };
        let mut names: Vec<Name> = m
            .exports
            .iter()
            .filter_map(|(exported, export)| {
                let Export::Local(local) = export else {
                    return None;
                };
                let decl = *m.decls.get(local)?;
                changed.range(decl.start..=decl.end).next()?;
                let word = if exported == "default" {
                    local
                } else {
                    exported
                };
                Some(Name {
                    exported: exported.clone(),
                    word: word.clone(),
                    decl,
                })
            })
            .collect();
        names.sort_by(|a, b| (a.decl.start, &a.exported).cmp(&(b.decl.start, &b.exported)));
        names.dedup_by_key(|n| n.decl.start);
        names.truncate(MAX_NAMES);
        names
    }

    /// One pass over the tree's TS/JS files for `words`: each whole name (word
    /// characters and `$`, so `data$` and `$store` count) looked up in the set, so no
    /// pattern grows with the names. Sorted, no symlinks, no dependency, output or
    /// test files, none over 256 KiB, at most [`SCAN_FILES`] files and [`SCAN_BYTES`].
    fn mentions(&self, words: &BTreeSet<&str>, deadline: Instant) -> Mentions {
        let mut out = Mentions::default();
        let Ok(re) = Regex::new(r"[\w$]+") else {
            out.error = Some("caller scan pattern rejected".to_owned());
            return out;
        };
        let (mut read, mut bytes) = (0usize, 0u64);
        let mut dirs = vec![String::new()];
        while let Some(dir) = dirs.pop() {
            let Ok(entries) = fs::read_dir(self.root.join(&dir)) else {
                continue;
            };
            let mut entries: Vec<(String, fs::FileType, u64)> = entries
                .flatten()
                .filter_map(|e| {
                    let name = e.file_name().into_string().ok()?;
                    Some((name, e.file_type().ok()?, e.metadata().ok()?.len()))
                })
                .collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            let rel = |name: &str| match dir.as_str() {
                "" => name.to_owned(),
                d => format!("{d}/{name}"),
            };
            let mut subdirs = Vec::new();
            for (name, kind, len) in entries {
                if kind.is_dir() && !SKIP_DIRS.contains(&name.as_str()) {
                    subdirs.push(rel(&name));
                }
                let test = name.contains(".test.") || name.contains(".spec.");
                if !kind.is_file() || test || ts_lang(&name).is_none() || len > SOURCE_CAP as u64 {
                    continue;
                }
                if Instant::now() >= deadline {
                    out.error = Some(DEADLINE.to_owned());
                    return out;
                }
                if read == SCAN_FILES || bytes + len > SCAN_BYTES {
                    out.error = Some("caller scan cap reached".to_owned());
                    return out;
                }
                read += 1;
                bytes += len;
                let path = rel(&name);
                let Some(src) = workspace::read(&self.root, &path, SOURCE_CAP) else {
                    continue;
                };
                let found: BTreeSet<&str> = re
                    .find_iter(&src)
                    .map(|m| m.as_str())
                    .filter(|w| words.contains(w))
                    .collect();
                for word in found {
                    out.files
                        .entry(word.to_owned())
                        .or_default()
                        .push(path.clone());
                }
            }
            dirs.extend(subdirs.into_iter().rev());
        }
        out
    }

    /// Caller snippets for changed file `path`: per name, candidates that share
    /// more of its directory first, then by path; at most [`PARSED_PER_NAME`]
    /// parsed, [`FILES_PER_NAME`] confirmed, [`CALLERS_PER_FILE`] in all.
    fn callers(&mut self, path: &str, names: &[Name], mentions: &Mentions) -> Vec<Snippet> {
        let mut out = Vec::new();
        for name in names {
            let Some(files) = mentions.files.get(&name.word) else {
                continue;
            };
            let mut candidates: Vec<&String> = files.iter().filter(|f| *f != path).collect();
            candidates.sort_by_key(|f| (Reverse(shared_dirs(f, path)), f.as_str()));
            let mut confirmed = 0;
            for file in candidates.into_iter().take(PARSED_PER_NAME) {
                if confirmed == FILES_PER_NAME || out.len() == CALLERS_PER_FILE {
                    break;
                }
                let sites = self.sites(file, path, name);
                if !sites.is_empty() {
                    confirmed += 1;
                    let room = CALLERS_PER_FILE - out.len();
                    out.extend(sites.into_iter().take(room));
                }
            }
        }
        out
    }

    /// Where `file` uses `name` of `target`: it must import a binding of it
    /// (named, aliased, default or namespace) that the lookup resolves to the
    /// changed declaration, barrels included, and use it outside import and
    /// export statements. At most [`SITES_PER_FILE`], each the use line −2 to
    /// +15 clipped to its top-level declaration; a use inside an earlier site is
    /// skipped.
    fn sites(&mut self, file: &str, target: &str, name: &Name) -> Vec<Snippet> {
        let Some(m) = self.module(file) else {
            return Vec::new();
        };
        let mut bindings: Vec<(&str, Option<&str>)> = Vec::new();
        for import in &m.imports {
            let Some(to) = self.resolve(file, &import.spec) else {
                continue;
            };
            for (local, imported) in &import.bindings {
                let (export, member) = match imported {
                    Imported::Named(n) => (n.as_str(), None),
                    Imported::Default => ("default", None),
                    Imported::Namespace => (name.exported.as_str(), Some(name.exported.as_str())),
                };
                let hit = self.lookup(&to, export);
                if hit.is_some_and(|d| d.path == target && d.decl == name.decl) {
                    bindings.push((local, member));
                }
            }
        }
        let Some(tree) = ts_lang(file).and_then(|l| l.parse(&m.src)) else {
            return Vec::new();
        };
        if bindings.is_empty() {
            return Vec::new();
        }
        let top = tree.root_node();
        let mut out: Vec<Snippet> = Vec::new();
        for node in uses(top) {
            let used = bindings.iter().any(|(local, member_name)| {
                text(node, &m.src) == *local
                    && member_name.is_none_or(|n| member(node, &m.src).as_deref() == Some(n))
            });
            let line = line(node);
            let inside = out
                .iter()
                .any(|s| (s.start_line..=s.end_line).contains(&line));
            if !used || inside {
                continue;
            }
            let decl = top_level(node);
            let start = (line - SITE_BEFORE).max(line_of(decl.start_position().row));
            let end = (line + SITE_AFTER).min(line_of(decl.end_position().row));
            out.push(Snippet {
                kind: SnippetKind::Caller,
                symbol: name.word.clone(),
                path: file.to_owned(),
                start_line: start,
                end_line: end,
                text: lines(&m.src, start, end),
            });
            if out.len() == SITES_PER_FILE {
                break;
            }
        }
        out
    }

    /// The pack of reviewed file `f`: definitions, callers, then tier 2, each
    /// (path, start) once.
    fn pack(&mut self, f: &FileChange, names: &[Name], mentions: &Mentions) -> ContextPack {
        let mut snippets = Vec::new();
        let unreadable = match self.module(&f.path) {
            None => Some("not readable in the checkout"),
            Some(m) => {
                let (tier1, tier2) = self.definitions(&f.path, &m, &DiffIndex::parse(&f.patch));
                let callers = self.callers(&f.path, names, mentions);
                let first: Vec<Snippet> = tier1.iter().filter_map(|d| self.snippet(d)).collect();
                let last: Vec<Snippet> = tier2.iter().filter_map(|d| self.snippet(d)).collect();
                let mut seen = BTreeSet::new();
                for s in first.into_iter().chain(callers).chain(last) {
                    if seen.insert((s.path.clone(), s.start_line)) {
                        snippets.push(s);
                    }
                }
                None
            }
        };
        let mut errors: Vec<&str> = self.config_error.iter().map(String::as_str).collect();
        if !names.is_empty() {
            errors.extend(mentions.error.as_deref());
        }
        errors.extend(unreadable);
        ContextPack {
            snippets,
            related: Vec::new(),
            error: (!errors.is_empty()).then(|| errors.join("; ")),
        }
    }
}

/// How many leading directories two repo-relative paths share.
fn shared_dirs(a: &str, b: &str) -> usize {
    let (a, b) = (super::dir(a), super::dir(b));
    a.split('/')
        .zip(b.split('/'))
        .take_while(|(x, y)| x == y && !x.is_empty())
        .count()
}

/// The top-level statement holding `node`.
fn top_level(node: Node<'_>) -> Node<'_> {
    let mut n = node;
    while let Some(p) = n.parent() {
        if p.parent().is_none() {
            return n;
        }
        n = p;
    }
    n
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::context::go::tests::change;

    const DATA: &str = "tests/data/context/ts";

    fn root(repo: &str) -> PathBuf {
        Path::new(DATA).join(repo)
    }

    fn later() -> Instant {
        Instant::now() + Duration::from_secs(60)
    }

    fn keys(cfg: &Config) -> Vec<&str> {
        cfg.paths.iter().map(|(k, _)| k.as_str()).collect()
    }

    #[test]
    fn configs() {
        let (web, err) = load_config(&root("next-app"));
        assert_eq!(err, None);
        assert_eq!(web.base_url, None, "next-app has no baseUrl");
        assert_eq!(keys(&web).len(), 14);
        assert_eq!(
            keys(&web)[..4],
            ["@/components/*", "@/lib/*", "@/hooks/*", "@/api"]
        );
        assert_eq!(keys(&web).last(), Some(&"@/*"));
        assert_eq!(web.paths_dir, "");

        let (expo, err) = load_config(&root("expo-app"));
        assert_eq!(err, None, "the missing expo/tsconfig.base is skipped");
        assert_eq!(expo.base_url.as_deref(), Some(""));
        assert_eq!(keys(&expo)[..2], ["@/*", "@/features/*"]);

        let (js, err) = load_config(&root("js-app"));
        assert_eq!(err, None);
        assert_eq!(js.base_url.as_deref(), Some("src"), "jsconfig.json");
        assert!(js.paths.is_empty());

        // Bases in order, a loop back to the root, an escape and a package: the
        // later base's paths win whole, baseUrl stays from the earlier one.
        let (chain, err) = load_config(&root("chain"));
        assert_eq!(err, None);
        assert_eq!(chain.base_url.as_deref(), Some("lib"));
        assert_eq!(keys(&chain), ["#x/*", "#up/*"]);
        assert_eq!(chain.paths_dir, "cfg");

        // Six levels: the sixth's baseUrl never applies, so paths stay relative
        // to the fifth's directory.
        let (deep, err) = load_config(&root("deep"));
        assert_eq!(err, None);
        assert_eq!(deep.base_url, None);
        assert_eq!(keys(&deep), ["#five/*"]);
        assert_eq!(deep.paths_dir, "cfg");
    }

    /// A wide `extends` array or `paths` entry costs bounded reads: 15 bases
    /// after the root, repeats free; 8 substitutions.
    #[test]
    fn config_reads_are_bounded() {
        let dir = std::env::temp_dir().join(format!("diffagent-ts-{}-wide", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("b15/s9")).unwrap();
        let bases: Vec<String> = (1..=20)
            .flat_map(|n| [format!("./c{n}.json"), "./c1.json".to_owned()])
            .collect();
        let subs: Vec<String> = (1..=9).map(|n| format!("./s{n}/*")).collect();
        let root = serde_json::json!({
            "extends": bases,
            "compilerOptions": { "paths": { "#s/*": subs } },
        });
        fs::write(dir.join("tsconfig.json"), root.to_string()).unwrap();
        for n in 1..=20 {
            let base = format!(r#"{{"compilerOptions": {{"baseUrl": "b{n}"}}}}"#);
            fs::write(dir.join(format!("c{n}.json")), base).unwrap();
        }
        // Substitutions are relative to baseUrl, the last base's.
        fs::write(dir.join("b15/s9/x.ts"), "export const x = 1;\n").unwrap();

        let mut repo = Repo::new(&dir);
        assert_eq!(repo.config_error, None);
        assert_eq!(repo.config.base_url.as_deref(), Some("b15"));
        assert_eq!(repo.resolve("a.ts", "#s/x"), None, "s9 is the ninth");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn broken_config_falls_back_to_relative_imports() {
        let mut cfg = Config::default();
        let err = apply(
            Path::new(DATA),
            "x.json",
            "{ \"compilerOptions\": { // no\n } }",
            0,
            &mut BTreeSet::new(),
            &mut cfg,
        );
        assert!(err.is_err(), "strict JSON");
        assert_eq!(cfg, Config::default());
    }

    #[test]
    fn resolve() {
        let from = "src/features/x/y.ts";
        let cases = [
            ("next-app", from, "@/api", Some("src/shared/api/index.ts")),
            (
                "next-app",
                from,
                "@/api/error",
                Some("src/shared/api/error.ts"),
            ),
            (
                "next-app",
                from,
                "@/hooks/use-payment-countdown",
                Some("src/shared/hooks/use-payment-countdown.ts"),
            ),
            (
                "next-app",
                from,
                "@/features/auth/session",
                Some("src/features/auth/session.ts"),
            ),
            ("next-app", from, "@/i18n", Some("src/shared/i18n/index.ts")),
            (
                "next-app",
                from,
                "@/types/global",
                Some("src/types/global.d.ts"),
            ),
            ("next-app", from, "@/lib", None),
            ("next-app", from, "react", None),
            (
                "next-app",
                "src/features/auth/hooks/a.ts",
                "./use-register-session",
                Some("src/features/auth/hooks/use-register-session.ts"),
            ),
            (
                "next-app",
                "src/features/auth/hooks/a.ts",
                "../session.ts",
                Some("src/features/auth/session.ts"),
            ),
            ("next-app", "src/a.ts", "../../outside.json", None),
            (
                "next-app",
                "src/a.ts",
                "../../../next-app/src/shared/api/error",
                None,
            ),
            ("next-app", "src/a.ts", "./shared/api/messages.json", None),
            (
                "expo-app",
                from,
                "@/components/Button",
                Some("src/shared/components/Button.tsx"),
            ),
            ("expo-app", from, "@/components/Missing", None),
            (
                "expo-app",
                from,
                "@/features/orders/utils/time-rules",
                Some("src/features/orders/utils/time-rules.ts"),
            ),
            (
                "expo-app",
                from,
                "src/components/Button",
                Some("src/components/Button.tsx"),
            ),
            (
                "js-app",
                "src/pages/Home.js",
                "components/Header",
                Some("src/components/Header.js"),
            ),
            (
                "js-app",
                "src/pages/Home.js",
                "utils",
                Some("src/utils/index.js"),
            ),
            ("chain", "a.ts", "#x/a", Some("lib/x/a.ts")),
            ("chain", "a.ts", "#old/o", None),
            ("chain", "a.ts", "#up/chain/lib/x/a", None),
            ("chain", "a.ts", "#out/chain/lib/x/a", None),
            ("deep", "a.ts", "#five/x", Some("cfg/five/x.ts")),
        ];
        for (repo, from, spec, want) in cases {
            let mut r = Repo::new(&root(repo));
            assert_eq!(r.resolve(from, spec).as_deref(), want, "{repo}: {spec}");
            assert_eq!(
                r.resolve(from, spec).as_deref(),
                want,
                "{repo}: {spec}, cached"
            );
        }
    }

    #[test]
    fn lookup_follows_reexports() {
        let mut r = Repo::new(&root("next-app"));
        let def = |path: &str, symbol: &str, kind: Kind, start: i64, end: i64| Def {
            path: path.to_owned(),
            symbol: symbol.to_owned(),
            decl: Decl { kind, start, end },
        };
        let cases = [
            // Three hops to a destructured export.
            (
                "src/shared/i18n/index.ts",
                "useRouter",
                Some(def(
                    "src/shared/i18n/navigation/create-navigation.ts",
                    "useRouter",
                    Kind::Value,
                    7,
                    7,
                )),
            ),
            (
                "src/shared/i18n/index.ts",
                "locales",
                Some(def(
                    "src/shared/i18n/config.ts",
                    "locales",
                    Kind::Value,
                    1,
                    1,
                )),
            ),
            ("src/shared/i18n/index.ts", "nothing", None),
            (
                "src/shared/api/index.ts",
                "ApiError",
                Some(def("src/shared/api/error.ts", "ApiError", Kind::Type, 1, 5)),
            ),
            (
                "src/features/auth/hooks/format-otp.ts",
                "default",
                Some(def(
                    "src/features/auth/hooks/format-otp.ts",
                    "formatOtp",
                    Kind::Function,
                    1,
                    48,
                )),
            ),
            (
                "src/shared/lib/search-form-utils.ts",
                "computeEarliestStart",
                Some(def(
                    "src/shared/lib/time.ts",
                    "computeEarliestStart",
                    Kind::Function,
                    14,
                    25,
                )),
            ),
            (
                "src/shared/chain/c1.ts",
                "far",
                Some(def("src/shared/chain/c5.ts", "far", Kind::Value, 1, 1)),
            ),
            ("src/shared/chain/c0.ts", "far", None), // five hops
        ];
        for (path, name, want) in cases {
            assert_eq!(r.lookup(path, name), want, "{path}: {name}");
        }
    }

    /// (kind, symbol, path, start, end) of each snippet.
    fn brief(pack: &ContextPack) -> Vec<(SnippetKind, &str, &str, i64, i64)> {
        pack.snippets
            .iter()
            .map(|s| {
                (
                    s.kind,
                    s.symbol.as_str(),
                    s.path.as_str(),
                    s.start_line,
                    s.end_line,
                )
            })
            .collect()
    }

    const FORM: &str = "src/features/auth/hooks/use-verify-otp-form.ts";
    const FORM_PATCH: &str = "@@ -18,9 +18,10 @@ export function useVerifyOtpForm() {

   const isSessionExpired = useIsExpired(session?.expired_at);
   const canSubmit = otp.length >= limit && !isSessionExpired;

   function handleError(err: unknown) {
+    if (!isApiError(err)) store.clearAuthSession();
     router.replace('/register');
   }

   return { otp: formatOtp(otp), setOtp, setSession, canSubmit, handleError };";

    const TIME: &str = "src/shared/lib/time.ts";
    const TIME_PATCH: &str = "@@ -18,7 +18,7 @@
 export function computeEarliestStart(
   now: Date,
-  advanceHours = 4,
+  advanceMinutes = 4 * 60,
   tz?: string
 ): { h: number; m: number } | undefined {
   const { h, m } = getServerHM(now, tz);
@@ -26,5 +26,5 @@

 /** Formats hours and minutes as HH:MM. */
 export function formatHM(h: number, m: number): string {
-  return `${h}:${m}`;
+  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
 }";

    fn web_build() -> Output {
        let pr = [
            change(FORM, FileStatus::Modified, FORM_PATCH),
            change(TIME, FileStatus::Modified, TIME_PATCH),
            change(
                "src/shared/lib/lead-time.ts",
                FileStatus::Modified,
                "@@ -1 +1 @@\n-x\n+y",
            ),
            change("README.md", FileStatus::Modified, "@@ -1 +1 @@\n-x\n+y"),
        ];
        build(&root("next-app"), &pr, &[FORM, TIME, "README.md"], later())
    }

    #[test]
    fn definitions_in_order() {
        use SnippetKind::Definition as D;
        let out = web_build();
        let pack = &out.packs[FORM];
        assert_eq!(pack.error, None);
        assert_eq!(
            brief(pack),
            [
                // Functions: added-line uses first, then context lines.
                (D, "isApiError", "src/shared/api/error.ts", 7, 10),
                (D, "clearAuthSession", "src/features/auth/session.ts", 1, 4),
                (
                    D,
                    "useIsExpired",
                    "src/shared/hooks/use-payment-countdown.ts",
                    18,
                    30
                ),
                (
                    D,
                    "formatOtp",
                    "src/features/auth/hooks/format-otp.ts",
                    1,
                    40
                ),
                // Same-file values declared off the hunk, nested ones included.
                (D, "session", FORM, 15, 15),
                (D, "otp", FORM, 16, 16),
                (D, "limit", FORM, 17, 17),
                (D, "router", FORM, 14, 14),
                // Tier 2, after the (absent) callers: what those values come from.
                (
                    D,
                    "useRegisterSession",
                    "src/features/auth/hooks/use-register-session.ts",
                    7,
                    23
                ),
                (
                    D,
                    "useRouter",
                    "src/shared/i18n/navigation/create-navigation.ts",
                    7,
                    7
                ),
                (D, "OTP_LENGTH", FORM, 11, 11),
            ]
        );
        let clipped = &pack.snippets[3];
        assert_eq!(clipped.text.lines().count(), 40, "the 40-line clip");
        assert!(
            clipped.text.starts_with("/**\n * Groups an OTP"),
            "with its JSDoc"
        );
        assert_eq!(
            pack.snippets[4].text,
            "  const { session, setSession } = useRegisterSession();"
        );
        assert_eq!(out.specifiers, (6, 6), "react is a package, not code");
    }

    #[test]
    fn callers_capped_in_candidate_order() {
        use SnippetKind::{Caller as C, Definition as D};
        let out = web_build();
        let pack = &out.packs[TIME];
        assert_eq!(pack.error, None);
        let cest = "computeEarliestStart";
        assert_eq!(
            brief(pack),
            [
                (D, "getServerHM", TIME, 1, 5),
                // Same directory first. notes.ts only mentions the name and
                // search-form-utils.ts only re-exports it; tests never count.
                (C, cest, "src/shared/lib/lead-time.ts", 6, 9),
                // Two sites per file; the use on line 5 sits inside the first.
                (C, cest, "src/shared/lib/search-slot-form-utils.ts", 3, 7),
                (C, cest, "src/shared/lib/search-slot-form-utils.ts", 9, 12),
                // Through the `../lib` barrel, then a namespace import.
                (C, cest, "src/shared/schemas/quick-search.ts", 3, 5),
                (C, cest, "src/shared/schemas/slot-search.ts", 3, 5),
                // Aliased; six files per name, so c.ts is dropped.
                (C, cest, "src/features/scheduling/a.ts", 3, 3),
                (C, cest, "src/features/scheduling/b.ts", 3, 3),
                // Eight per changed file: clock.ts is dropped.
                (C, "formatHM", "src/shared/lib/display.ts", 3, 3),
            ]
        );
        assert_eq!(
            pack.snippets[2].text,
            "export function earliestToday(now: Date) {\n  const first = computeEarliestStart(now, 4);\n  const again = computeEarliestStart(now, 4);\n  return first ?? again;\n}"
        );
    }

    /// `$` is part of a JS name: `data$` and `$store` are found, and a longer name that
    /// holds one is not it.
    #[test]
    fn caller_scan_finds_dollar_names() {
        let dir = std::env::temp_dir().join(format!("diffagent-ts-{}-dollar", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("a.ts"),
            "export const data$ = 1;\nexport const $store = 2;\n",
        )
        .unwrap();
        fs::write(
            dir.join("b.ts"),
            "import { data$ } from './a';\nconst x = data$ + data$x + $$store;\n",
        )
        .unwrap();
        fs::write(dir.join("c.ts"), "const y = $store.get(plain);\n").unwrap();
        let words = BTreeSet::from(["data$", "$store", "plain"]);
        let got = Repo::new(&dir).mentions(&words, later());
        assert_eq!(got.error, None);
        let want = BTreeMap::from([
            (
                "$store".to_owned(),
                vec!["a.ts".to_owned(), "c.ts".to_owned()],
            ),
            (
                "data$".to_owned(),
                vec!["a.ts".to_owned(), "b.ts".to_owned()],
            ),
            ("plain".to_owned(), vec!["c.ts".to_owned()]),
        ]);
        assert_eq!(got.files, want);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn import_graph_and_packs() {
        let out = web_build();
        assert_eq!(
            out.packs.keys().collect::<Vec<_>>(),
            [FORM, TIME],
            "no pack for lead-time.ts (not reviewed) or README.md"
        );
        let want: BTreeSet<String> = [
            "src/features/auth/hooks/format-otp.ts",
            "src/features/auth/hooks/use-register-session.ts",
            "src/features/auth/session.ts",
            "src/shared/api/index.ts",
            "src/shared/hooks/use-payment-countdown.ts",
            "src/shared/i18n/index.ts",
        ]
        .map(str::to_owned)
        .into();
        assert_eq!(out.imports[FORM], want);
        assert_eq!(
            out.imports["src/shared/lib/lead-time.ts"],
            BTreeSet::from([TIME.to_owned()])
        );
        assert!(out.imports[TIME].is_empty());
        assert_eq!(out, web_build(), "deterministic");
        let json = serde_json::to_string(&out.packs).unwrap();
        let abs = root("next-app").canonicalize().unwrap();
        assert!(!json.contains(abs.to_str().unwrap()), "no root path");
    }

    #[test]
    fn specifiers_counted_by_base_url() {
        let pr = [change(
            "src/pages/Home.js",
            FileStatus::Modified,
            "@@ -1 +1 @@\n-x\n+import React from 'react';",
        )];
        let out = build(&root("js-app"), &pr, &["src/pages/Home.js"], later());
        // components/Header and utils resolve, components/Gone doesn't; react is a
        // package and the png and css are assets.
        assert_eq!(out.specifiers, (2, 3));
        let want = ["src/components/Header.js", "src/utils/index.js"].map(str::to_owned);
        assert_eq!(out.imports["src/pages/Home.js"], want.into());
        assert_eq!(out.packs["src/pages/Home.js"].error, None);
    }

    #[test]
    fn expired_deadline_sets_error() {
        let pr = [change(FORM, FileStatus::Modified, FORM_PATCH)];
        let out = build(&root("next-app"), &pr, &[FORM], Instant::now());
        let pack = &out.packs[FORM];
        assert!(pack.snippets.is_empty());
        assert_eq!(pack.error.as_deref(), Some("deadline exceeded"));
        assert!(out.imports.is_empty());
    }

    #[test]
    fn unreadable_file_sets_error() {
        let pr = [change(
            "src/gone.ts",
            FileStatus::Modified,
            "@@ -1 +1 @@\n-x\n+y",
        )];
        let out = build(&root("next-app"), &pr, &["src/gone.ts"], later());
        assert_eq!(
            out.packs["src/gone.ts"].error.as_deref(),
            Some("not readable in the checkout")
        );
    }

    #[test]
    fn helpers() {
        assert_eq!(join("a/b", "../c/./d"), Some("a/c/d".to_owned()));
        assert_eq!(join("a", "../.."), None);
        assert_eq!(join("a", "/etc"), None);
        assert_eq!(
            shared_dirs("src/shared/lib/a.ts", "src/shared/lib/time.ts"),
            3
        );
        assert_eq!(
            shared_dirs("src/features/a.ts", "src/shared/lib/time.ts"),
            1
        );
        assert_eq!(shared_dirs("a.ts", "b.ts"), 0);
        assert!(is_asset("./en.JSON") && is_asset("assets/logo.png"));
        assert!(!is_asset("./order.types") && !is_asset("@/lib/time"));
        assert!(relative(".") && relative("../x") && !relative(".x") && !relative("x"));
    }
}
