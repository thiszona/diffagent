//! The quality scorer: how many labeled problems a review run raised. Pure: it reads
//! only the label files and a run's findings, both already parsed, and returns the
//! scorecard that `diffagent-eval score` prints, or per-label
//! [`outcomes`] that [`paired`] compares between versions. Also pure, [`coverage`]:
//! how much of the labels' cross-file evidence the committed context packs reach,
//! with no model call.

use std::cmp::Ordering;
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use crate::context::{ContextPack, Relation, SnippetKind};
use crate::diff::{DiffIndex, FileStatus};
use crate::finding::Severity;
use crate::lang::Lang;

/// How far, in lines, a finding may sit from a labeled range and still count as
/// raising it. Models anchor a few lines off the line a human would pick.
pub const WINDOW: i64 = 3;

/// Labeled fixtures by name: `labels.history.json` and `labels.planted.json`,
/// combined with [`merge`].
pub type Labels = BTreeMap<String, LabeledFixture>;

/// One fixture's labels; its repo and commits are ignored.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct LabeledFixture {
    pub labels: Vec<Label>,
}

/// A known problem, or known false alarm, at a range of new-file lines.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Label {
    pub path: String,
    pub start_line: i64,
    pub end_line: i64,
    pub severity: Severity,
    pub label: LabelKind,
    /// Only planted bugs carry one; missing means visible in the diff.
    #[serde(default)]
    pub scope: Option<Scope>,
    /// Lines elsewhere that prove the problem, which this file's context should
    /// reach; only [`coverage`] reads them.
    #[serde(default)]
    pub evidence: Vec<Evidence>,
}

/// A line, in the tree at the fixture's commit, that proves a label.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Evidence {
    pub path: String,
    pub line: i64,
}

/// What a label says about a finding raised at its range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LabelKind {
    /// A maintainer explained why this finding was a false alarm.
    Rejected,
    /// A later push fixed what this finding reported.
    Accepted,
    /// A bug planted on purpose.
    Planted,
}

/// Where the evidence for a planted bug lives, when not in the diff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Scope {
    /// Provable only from another file, such as the callee's definition.
    CrossFile,
}

/// One fixture's review output, as a run's `findings.json` or a baseline stores it.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct FixtureFindings {
    pub fixture: String,
    pub failed_files: usize,
    /// The files whose review failed. A baseline may hold only the count, so when this
    /// is absent any failure counts as losing every label the run missed.
    #[serde(default)]
    pub failed_paths: Option<Vec<String>>,
    /// `null` when the run found nothing.
    #[serde(default)]
    pub findings: Option<Vec<ScoredFinding>>,
}

/// The fields of a finding the scorer reads; the rest are ignored.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ScoredFinding {
    pub path: String,
    pub start_line: i64,
    pub end_line: i64,
    pub severity: Severity,
}

/// The scorer's label sets, in row order; [`coverage`] uses the same rows.
const ROW_NAMES: [&str; 4] = [
    "**rejected** (known false alarms; lower is better)",
    "**accepted** (known real issues; higher is better)",
    "planted, visible in the diff",
    "planted, provable only from another file",
];

/// One label set's counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Row {
    pub labels: usize,
    /// Labels with at least one finding in range.
    pub raised: usize,
    /// Labels not raised whose file's review failed.
    pub lost: usize,
    /// Raised labels where some finding in range has the label's severity.
    pub severity_matched: usize,
}

/// A run's score against the labels.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Scorecard {
    pub rejected: Row,
    pub accepted: Row,
    pub planted_diff: Row,
    pub planted_cross_file: Row,
    /// Labeled fixtures scored; unlabeled ones are left out of every total.
    pub fixtures: usize,
    pub findings: usize,
    pub failed_files: usize,
    /// Findings in range of no label: unlabeled, so neither right nor wrong.
    pub unmatched: usize,
}

impl Scorecard {
    /// Accepted over all raised history labels, as (numerator, denominator): of the
    /// findings with a known verdict, how many were real.
    pub fn precision(&self) -> (usize, usize) {
        let acc = self.accepted.raised;
        (acc, acc + self.rejected.raised)
    }

    /// Accepted labels raised, as (numerator, denominator).
    pub fn recall(&self) -> (usize, usize) {
        (self.accepted.raised, self.accepted.labels)
    }

    fn row_mut(&mut self, label: &Label) -> &mut Row {
        match (label.label, label.scope) {
            (LabelKind::Rejected, _) => &mut self.rejected,
            (LabelKind::Accepted, _) => &mut self.accepted,
            (LabelKind::Planted, None) => &mut self.planted_diff,
            (LabelKind::Planted, Some(Scope::CrossFile)) => &mut self.planted_cross_file,
        }
    }

    fn rows(&self) -> [(&'static str, Row); 4] {
        let [rejected, accepted, diff, cross_file] = ROW_NAMES;
        [
            (rejected, self.rejected),
            (accepted, self.accepted),
            (diff, self.planted_diff),
            (cross_file, self.planted_cross_file),
        ]
    }
}

/// Why a run cannot be scored.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ScoreError {
    #[error("fixture {0} is labeled in more than one label file")]
    Duplicate(String),
    #[error("labeled fixture {0} is missing from the findings")]
    Missing(String),
}

/// Combines label files. A fixture in two files is an error, so one file never
/// silently replaces another's labels.
pub fn merge(files: impl IntoIterator<Item = Labels>) -> Result<Labels, ScoreError> {
    let mut all = Labels::new();
    for (name, fixture) in files.into_iter().flatten() {
        match all.entry(name) {
            Entry::Occupied(e) => return Err(ScoreError::Duplicate(e.key().clone())),
            Entry::Vacant(e) => {
                e.insert(fixture);
            }
        }
    }
    Ok(all)
}

/// Scores a run: every labeled fixture must be in `runs`; fixtures without labels
/// (poisoned_lesson, intent) are skipped.
pub fn score(labels: &Labels, runs: &[FixtureFindings]) -> Result<Scorecard, ScoreError> {
    score_within(labels, runs, WINDOW)
}

fn score_within(
    labels: &Labels,
    runs: &[FixtureFindings],
    window: i64,
) -> Result<Scorecard, ScoreError> {
    let by_name: HashMap<&str, &FixtureFindings> =
        runs.iter().map(|r| (r.fixture.as_str(), r)).collect();
    let mut card = Scorecard::default();
    for (name, fixture) in labels {
        let run = by_name
            .get(name.as_str())
            .ok_or_else(|| ScoreError::Missing(name.clone()))?;
        let findings = run.findings.as_deref().unwrap_or_default();
        card.fixtures += 1;
        card.findings += findings.len();
        card.failed_files += run.failed_files;
        card.unmatched += findings
            .iter()
            .filter(|f| !fixture.labels.iter().any(|l| matches(f, l, window)))
            .count();
        for label in &fixture.labels {
            let mut hits = findings.iter().filter(|f| matches(f, label, window));
            let row = card.row_mut(label);
            row.labels += 1;
            match hits.next() {
                Some(first) => {
                    row.raised += 1;
                    let same = |f: &ScoredFinding| f.severity == label.severity;
                    row.severity_matched += usize::from(same(first) || hits.any(same));
                }
                None => row.lost += usize::from(lost(run, &label.path)),
            }
        }
    }
    Ok(card)
}

/// Same file, and the closed line ranges overlap once the label's is widened by
/// `window`. A start line of 0 (unknown) means the finding or label is one line.
pub fn matches(f: &ScoredFinding, l: &Label, window: i64) -> bool {
    let (f_start, f_end) = span(f.start_line, f.end_line);
    let (l_start, l_end) = span(l.start_line, l.end_line);
    f.path == l.path
        && f_start <= l_end.saturating_add(window)
        && f_end >= l_start.saturating_sub(window)
}

fn span(start: i64, end: i64) -> (i64, i64) {
    (if start == 0 { end } else { start }, end)
}

fn lost(run: &FixtureFindings, path: &str) -> bool {
    match &run.failed_paths {
        Some(paths) => paths.iter().any(|p| p == path),
        None => run.failed_files > 0,
    }
}

/// Per label, in [`Labels`] order (fixture, then label), whether a run got it right:
/// raised for an accepted or planted label, not raised for a rejected one. A label
/// lost to a failed file is never right, nor is one whose fixture is missing from
/// `runs`, so a run cannot gain by failing. Two runs scored over the same `labels`
/// line up label by label, which is what [`paired`] compares.
pub fn outcomes(labels: &Labels, runs: &[FixtureFindings]) -> Vec<bool> {
    let by_name: HashMap<&str, &FixtureFindings> =
        runs.iter().map(|r| (r.fixture.as_str(), r)).collect();
    let mut out = Vec::new();
    for (name, fixture) in labels {
        let run = by_name.get(name.as_str()).copied();
        out.extend(
            fixture
                .labels
                .iter()
                .map(|l| run.is_some_and(|r| right(r, l))),
        );
    }
    out
}

fn right(run: &FixtureFindings, label: &Label) -> bool {
    let findings = run.findings.as_deref().unwrap_or_default();
    let raised = findings.iter().any(|f| matches(f, label, WINDOW));
    match label.label {
        LabelKind::Rejected => !raised && !lost(run, &label.path),
        LabelKind::Accepted | LabelKind::Planted => raised,
    }
}

/// How a candidate's runs compare with a base's, label by label.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Paired {
    /// Labels the candidate gets right in a larger share of its runs than the base.
    pub gained: usize,
    /// Labels it gets right in a smaller share.
    pub lost: usize,
    /// Labels on which the sides disagree, `gained + lost`: between two single runs
    /// of one version, the noise band.
    pub d: usize,
    /// One-sided: the chance, were the versions no different, that the candidate gets
    /// at least as many outcomes right as it did.
    pub p: f64,
}

/// An exact stratified permutation test of `cand` against `base`, each a list of
/// runs' [`outcomes`] over the same labels. Were the versions no different, a
/// label's runs from both sides would be exchangeable: given how many of them got it
/// right, the candidate's count is hypergeometric. `p` convolves those per-label
/// distributions into the chance of the candidate's total being at least what it
/// was. A label every run gets right, or every run wrong, adds nothing. With one run
/// per side this is the exact sign test over the disagreeing labels.
///
/// # Panics
///
/// If the runs do not all have the same length: they were not scored over the same
/// labels, so no label lines up and any result would be wrong.
pub fn paired(base: &[Vec<bool>], cand: &[Vec<bool>]) -> Paired {
    let runs = || base.iter().chain(cand);
    let labels = runs().map(Vec::len).max().unwrap_or_default();
    assert!(
        runs().all(|r| r.len() == labels),
        "runs scored over different labels"
    );
    let (m, n) = (base.len(), cand.len());
    let (mut gained, mut lost, mut observed) = (0, 0, 0);
    // dist[s]: the chance the candidate gets s outcomes right, over the labels so far.
    let mut dist = vec![1.0];
    for i in 0..labels {
        let a = base.iter().filter(|r| r[i]).count();
        let b = cand.iter().filter(|r| r[i]).count();
        // The shares b/n and a/m, compared without dividing by a side with no runs.
        match (b * m).cmp(&(a * n)) {
            Ordering::Greater => gained += 1,
            Ordering::Less => lost += 1,
            Ordering::Equal => {}
        }
        observed += b;
        dist = convolve(&dist, &hypergeometric(m + n, a + b, n));
    }
    // observed <= dist.len() - 1: each label's b is within its distribution's support.
    let p = dist[observed..].iter().sum::<f64>().min(1.0);
    Paired {
        gained,
        lost,
        d: gained + lost,
        p,
    }
}

/// The chance that `draws` runs picked from `total`, `good` of which got a label
/// right, include exactly j right ones, for j in `0..=min(draws, good)`.
fn hypergeometric(total: usize, good: usize, draws: usize) -> Vec<f64> {
    let all = choose(total, draws);
    (0..=draws.min(good))
        .map(|j| choose(good, j) * choose(total - good, draws - j) / all)
        .collect()
}

/// n choose k; exact for the few runs a comparison has, since each step's product
/// is a whole number.
fn choose(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
}

/// The distribution of the sum of two independent counts; neither is empty.
fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            out[i + j] += x * y;
        }
    }
    out
}

/// The scorecard table and totals. With `against`, each count column
/// is followed by the same count from `against`, and its totals get their own line.
pub fn render(card: &Scorecard, against: Option<&Scorecard>) -> String {
    let header = [
        "Raised",
        "Lost to a failed file",
        "Severity matched (of raised)",
    ]
    .map(str::to_owned);
    let their_header = against.map(|_| header.clone().map(|h| format!("{h} (against)")));
    let mut out = format!(
        "| Label set | Labels |{}\n|{}\n",
        cells(&header, their_header.as_ref()),
        "---|".repeat(if against.is_some() { 8 } else { 5 }),
    );
    let theirs = against.map(Scorecard::rows);
    for (i, (name, row)) in card.rows().into_iter().enumerate() {
        let their_counts = theirs.map(|t| counts(t[i].1));
        out.push_str(&format!(
            "| {name} | {} |{}\n",
            row.labels,
            cells(&counts(row), their_counts.as_ref()),
        ));
    }
    out.push_str(&format!("\n{}", totals(card)));
    if let Some(a) = against {
        out.push_str(&format!("Against: {}", totals(a)));
    }
    out
}

fn counts(r: Row) -> [String; 3] {
    [
        format!("**{}**", r.raised),
        r.lost.to_string(),
        r.severity_matched.to_string(),
    ]
}

/// `" a | b | c |"`, or with `theirs` each cell followed by its twin.
fn cells(ours: &[String; 3], theirs: Option<&[String; 3]>) -> String {
    let mut out = String::new();
    for (i, cell) in ours.iter().enumerate() {
        out.push_str(&format!(" {cell} |"));
        if let Some(t) = theirs {
            out.push_str(&format!(" {} |", t[i]));
        }
    }
    out
}

fn totals(card: &Scorecard) -> String {
    let (p, pd) = card.precision();
    let (r, rd) = card.recall();
    format!(
        "{} fixtures, {} findings, {} failed files, {} findings matching no label; \
         precision {p}/{pd} ({}), accepted recall {r}/{rd} ({}).\n",
        card.fixtures,
        card.findings,
        card.failed_files,
        card.unmatched,
        percent(p, pd),
        percent(r, rd),
    )
}

fn percent(n: usize, d: usize) -> String {
    match d {
        0 => "n/a".to_owned(),
        _ => format!("{}%", (n * 100 + d / 2) / d),
    }
}

/// One fixture's context as `diffagent-eval context` writes it to
/// `eval/synthetic/context/<fixture>.json`. Committed, so coverage and recordings
/// replay it offline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixtureContext {
    pub repo: String,
    pub sha: String,
    /// Every PR file with its status.
    pub files: Vec<(String, FileStatus)>,
    pub conventions: String,
    /// By reviewed file.
    pub packs: BTreeMap<String, ContextPack>,
    /// TS/JS code specifiers resolved, of all seen.
    pub ts_imports: (u32, u32),
}

/// Where in a pack an evidence line was found: which list, and the index in it
/// (from 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Found {
    Snippet { kind: SnippetKind, index: usize },
    Related { relation: Relation, index: usize },
}

/// Whether a label file's context reaches an evidence line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reach {
    /// In the pack fitted to the budget, at that place.
    Hit(Found),
    /// Only in the unfitted pack: the budget cut it.
    Cut(Found),
    Missed,
}

/// One evidence line and how far the context got.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceReach {
    pub fixture: String,
    pub path: String,
    pub line: i64,
    pub reach: Reach,
}

/// One label set's evidence counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CoverageRow {
    pub evidence: usize,
    pub hit: usize,
    pub cut: usize,
    pub missed: usize,
}

/// One pack's size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackStat {
    pub fixture: String,
    pub path: String,
    pub snippets: usize,
    pub related: usize,
    /// Unfitted.
    pub tokens: u32,
    pub error: Option<String>,
}

/// What [`coverage`] measured.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Coverage {
    pub budget: u32,
    /// Rows as the scorer's.
    pub rejected: CoverageRow,
    pub accepted: CoverageRow,
    pub planted_diff: CoverageRow,
    pub planted_cross_file: CoverageRow,
    /// By fixture, then label, then evidence order.
    pub evidence: Vec<EvidenceReach>,
    /// By fixture, then path.
    pub packs: Vec<PackStat>,
    /// TS/JS code specifiers resolved, of all seen, over every fixture.
    pub ts_imports: (usize, usize),
    /// Go packs without an error, and with one (goctx skipped or failed the file).
    pub go_loaded: usize,
    pub go_skipped: usize,
}

impl Coverage {
    fn row_mut(&mut self, label: &Label) -> &mut CoverageRow {
        match (label.label, label.scope) {
            (LabelKind::Rejected, _) => &mut self.rejected,
            (LabelKind::Accepted, _) => &mut self.accepted,
            (LabelKind::Planted, None) => &mut self.planted_diff,
            (LabelKind::Planted, Some(Scope::CrossFile)) => &mut self.planted_cross_file,
        }
    }

    fn rows(&self) -> [(&'static str, CoverageRow); 4] {
        let [rejected, accepted, diff, cross_file] = ROW_NAMES;
        [
            (rejected, self.rejected),
            (accepted, self.accepted),
            (diff, self.planted_diff),
            (cross_file, self.planted_cross_file),
        ]
    }
}

/// Measures, with no model call, whether each label's evidence is in its file's
/// pack: a hit when the pack fitted to `budget` tokens has a snippet of that path
/// covering the line, or a related diff of that path whose new side has it; a cut
/// when only the unfitted pack does. A fixture without context misses everything.
pub fn coverage(
    labels: &Labels,
    contexts: &BTreeMap<String, FixtureContext>,
    budget: u32,
) -> Coverage {
    let mut cov = Coverage {
        budget,
        ..Coverage::default()
    };
    for (name, fixture) in labels {
        for label in &fixture.labels {
            let pack = contexts.get(name).and_then(|c| c.packs.get(&label.path));
            let fitted = pack.map(|p| p.fit(budget));
            for ev in &label.evidence {
                let reach = match (fitted.as_ref().and_then(|f| find(f, ev)), pack) {
                    (Some(found), _) => Reach::Hit(found),
                    (None, Some(p)) => find(p, ev).map_or(Reach::Missed, Reach::Cut),
                    (None, None) => Reach::Missed,
                };
                let row = cov.row_mut(label);
                row.evidence += 1;
                match reach {
                    Reach::Hit(_) => row.hit += 1,
                    Reach::Cut(_) => row.cut += 1,
                    Reach::Missed => row.missed += 1,
                }
                cov.evidence.push(EvidenceReach {
                    fixture: name.clone(),
                    path: ev.path.clone(),
                    line: ev.line,
                    reach,
                });
            }
        }
    }
    for (name, ctx) in contexts {
        cov.ts_imports.0 += ctx.ts_imports.0 as usize;
        cov.ts_imports.1 += ctx.ts_imports.1 as usize;
        for (path, pack) in &ctx.packs {
            if Lang::for_path(path) == Some(Lang::Go) {
                match pack.error {
                    None => cov.go_loaded += 1,
                    Some(_) => cov.go_skipped += 1,
                }
            }
            cov.packs.push(PackStat {
                fixture: name.clone(),
                path: path.clone(),
                snippets: pack.snippets.len(),
                related: pack.related.len(),
                tokens: pack.tokens(),
                error: pack.error.clone(),
            });
        }
    }
    cov
}

/// The first snippet of `ev`'s path whose lines cover it, else the first related
/// diff of that path whose new side has it.
fn find(pack: &ContextPack, ev: &Evidence) -> Option<Found> {
    let snippet = pack
        .snippets
        .iter()
        .enumerate()
        .find(|(_, s)| s.path == ev.path && (s.start_line..=s.end_line).contains(&ev.line));
    if let Some((index, s)) = snippet {
        return Some(Found::Snippet {
            kind: s.kind,
            index,
        });
    }
    pack.related
        .iter()
        .enumerate()
        .find(|(_, r)| r.path == ev.path && DiffIndex::parse(&r.patch).has(ev.line))
        .map(|(index, r)| Found::Related {
            relation: r.relation,
            index,
        })
}

/// The coverage report: each evidence line, the rows' totals, every pack's size,
/// then the TS resolution rate and the Go files loaded and skipped.
pub fn render_coverage(cov: &Coverage) -> String {
    let mut out = format!(
        "Evidence at a budget of {} tokens:\n\n\
         | Fixture | Evidence | Reached | Found as |\n|---|---|---|---|\n",
        cov.budget
    );
    for e in &cov.evidence {
        let (reached, found) = match e.reach {
            Reach::Hit(f) => ("**hit**", found_as(f)),
            Reach::Cut(f) => ("cut", found_as(f)),
            Reach::Missed => ("missed", String::new()),
        };
        out.push_str(&format!(
            "| {} | {}:{} | {reached} | {found} |\n",
            e.fixture, e.path, e.line
        ));
    }
    out.push_str("\n| Label set | Evidence lines | Hit | Cut | Missed |\n|---|---|---|---|---|\n");
    for (name, r) in cov.rows() {
        out.push_str(&format!(
            "| {name} | {} | **{}** | {} | {} |\n",
            r.evidence, r.hit, r.cut, r.missed
        ));
    }
    out.push_str(
        "\n| Fixture | Pack | Snippets | Related | Tokens | Error |\n|---|---|---|---|---|---|\n",
    );
    for p in &cov.packs {
        let error = p.error.as_deref().map(cell).unwrap_or_default();
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {error} |\n",
            p.fixture, p.path, p.snippets, p.related, p.tokens
        ));
    }
    let (resolved, specifiers) = cov.ts_imports;
    out.push_str(&format!(
        "\nTS resolution {resolved}/{specifiers} code specifiers ({}); \
         Go files {} loaded, {} skipped.\n",
        percent(resolved, specifiers),
        cov.go_loaded,
        cov.go_skipped,
    ));
    out
}

fn found_as(found: Found) -> String {
    match found {
        Found::Snippet { kind, index } => {
            let kind = match kind {
                SnippetKind::Definition => "definition",
                SnippetKind::Caller => "caller",
            };
            format!("{kind}, snippet {}", index + 1)
        }
        Found::Related { relation, index } => {
            let relation = match relation {
                Relation::Imports => "imports",
                Relation::ImportedBy => "imported by",
                Relation::SamePackage => "same package",
            };
            format!("related diff {} ({relation})", index + 1)
        }
    }
}

/// `s` safe inside a Markdown table cell.
fn cell(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', " ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{RelatedDiff, Snippet};

    /// Two fixtures with one label of each fate: a rejected label raised, one not
    /// raised and one lost to the failed `c.go`; accepted labels raised in range, 3
    /// lines off, and lost; a planted label raised and a cross-file one missed.
    fn sample() -> (Labels, Vec<FixtureFindings>) {
        let mut cross_file = label(LabelKind::Planted, "y.go", 3, 3);
        cross_file.scope = Some(Scope::CrossFile);
        let mut planted = label(LabelKind::Planted, "x.ts", 7, 7);
        planted.severity = Severity::Critical;
        let labels = Labels::from([
            (
                "history".to_owned(),
                LabeledFixture {
                    labels: vec![
                        label(LabelKind::Rejected, "a.go", 10, 10),
                        label(LabelKind::Rejected, "a.go", 40, 40),
                        label(LabelKind::Rejected, "c.go", 9, 9),
                        label(LabelKind::Accepted, "a.go", 20, 20),
                        label(LabelKind::Accepted, "b.go", 5, 5),
                        label(LabelKind::Accepted, "c.go", 1, 1),
                    ],
                },
            ),
            (
                "planted".to_owned(),
                LabeledFixture {
                    labels: vec![planted, cross_file],
                },
            ),
        ]);
        let runs = vec![
            FixtureFindings {
                fixture: "history".to_owned(),
                failed_files: 1,
                failed_paths: Some(vec!["c.go".to_owned()]),
                findings: Some(vec![
                    finding("a.go", 10, 10, Severity::Minor),
                    finding("a.go", 23, 23, Severity::Major),
                    finding("b.go", 5, 5, Severity::Major),
                    finding("a.go", 100, 100, Severity::Major),
                ]),
            },
            FixtureFindings {
                fixture: "planted".to_owned(),
                failed_files: 0,
                failed_paths: None,
                findings: Some(vec![finding("x.ts", 7, 7, Severity::Critical)]),
            },
        ];
        (labels, runs)
    }

    fn row(labels: usize, raised: usize, lost: usize, severity_matched: usize) -> Row {
        Row {
            labels,
            raised,
            lost,
            severity_matched,
        }
    }

    /// The accepted label 3 lines from its finding is raised only within 3 lines.
    #[test]
    fn a_two_line_window_raises_fewer_accepted() {
        let (labels, runs) = sample();
        let cases = [(WINDOW, row(3, 2, 1, 2), 1), (2, row(3, 1, 1, 1), 2)];
        for (window, accepted, unmatched) in cases {
            let card = score_within(&labels, &runs, window).unwrap();
            assert_eq!(card.accepted, accepted, "window {window}");
            assert_eq!(card.unmatched, unmatched, "window {window}");
            assert_eq!(card.rejected, row(3, 1, 1, 0), "window {window}");
        }
    }

    /// The plain scorecard with its precision and recall totals, then the same run
    /// at a 2-line window rendered against it.
    #[test]
    fn render_against_puts_both_runs_side_by_side() {
        let (labels, runs) = sample();
        let wide = score(&labels, &runs).unwrap();
        assert_eq!((wide.precision(), wide.recall()), ((2, 3), (2, 3)));
        let want = [
            "| Label set | Labels | Raised | Lost to a failed file | Severity matched (of raised) |",
            "|---|---|---|---|---|",
            "| **rejected** (known false alarms; lower is better) | 3 | **1** | 1 | 0 |",
            "| **accepted** (known real issues; higher is better) | 3 | **2** | 1 | 2 |",
            "| planted, visible in the diff | 1 | **1** | 0 | 1 |",
            "| planted, provable only from another file | 1 | **0** | 0 | 0 |",
            "",
            "2 fixtures, 5 findings, 1 failed files, 1 findings matching no label; \
             precision 2/3 (67%), accepted recall 2/3 (67%).",
        ];
        assert_eq!(render(&wide, None).lines().collect::<Vec<_>>(), want);

        let narrow = score_within(&labels, &runs, 2).unwrap();
        assert_eq!((narrow.precision(), narrow.recall()), ((1, 2), (1, 3)));
        let want = [
            "| Label set | Labels | Raised | Raised (against) | Lost to a failed file | \
             Lost to a failed file (against) | Severity matched (of raised) | \
             Severity matched (of raised) (against) |",
            "|---|---|---|---|---|---|---|---|",
            "| **rejected** (known false alarms; lower is better) | 3 | **1** | **1** | 1 | 1 | 0 | 0 |",
            "| **accepted** (known real issues; higher is better) | 3 | **1** | **2** | 1 | 1 | 1 | 2 |",
            "| planted, visible in the diff | 1 | **1** | **1** | 0 | 0 | 1 | 1 |",
            "| planted, provable only from another file | 1 | **0** | **0** | 0 | 0 | 0 | 0 |",
            "",
            "2 fixtures, 5 findings, 1 failed files, 2 findings matching no label; \
             precision 1/2 (50%), accepted recall 1/3 (33%).",
            "Against: 2 fixtures, 5 findings, 1 failed files, 1 findings matching no label; \
             precision 2/3 (67%), accepted recall 2/3 (67%).",
        ];
        assert_eq!(
            render(&narrow, Some(&wide)).lines().collect::<Vec<_>>(),
            want
        );
    }

    fn label(kind: LabelKind, path: &str, start: i64, end: i64) -> Label {
        Label {
            path: path.to_owned(),
            start_line: start,
            end_line: end,
            severity: Severity::Major,
            label: kind,
            scope: None,
            evidence: Vec::new(),
        }
    }

    fn finding(path: &str, start: i64, end: i64, severity: Severity) -> ScoredFinding {
        ScoredFinding {
            path: path.to_owned(),
            start_line: start,
            end_line: end,
            severity,
        }
    }

    #[test]
    fn match_window() {
        let l = label(LabelKind::Accepted, "a.go", 10, 12);
        let cases = [
            ("3 below the end", "a.go", 15, 15, true),
            ("4 below the end", "a.go", 16, 16, false),
            ("3 above the start", "a.go", 7, 7, true),
            ("4 above the start", "a.go", 6, 6, false),
            ("spans the label", "a.go", 1, 30, true),
            ("other file", "b.go", 11, 11, false),
            ("start 0 uses the end line", "a.go", 0, 14, true),
            ("start 0 is not line 0", "a.go", 0, 20, false),
        ];
        for (name, path, start, end, want) in cases {
            let f = finding(path, start, end, Severity::Major);
            assert_eq!(matches(&f, &l, WINDOW), want, "{name}");
        }
        let zero = label(LabelKind::Accepted, "a.go", 0, 12);
        let f = finding("a.go", 5, 8, Severity::Major);
        assert!(
            !matches(&f, &zero, WINDOW),
            "label start 0 uses its end line"
        );
    }

    fn one(labels: Vec<Label>) -> Labels {
        Labels::from([("fx".to_owned(), LabeledFixture { labels })])
    }

    #[test]
    fn lost_to_a_failed_file() {
        let labels = one(vec![
            label(LabelKind::Accepted, "a.go", 10, 10),
            label(LabelKind::Accepted, "b.go", 10, 10),
            label(LabelKind::Accepted, "c.go", 10, 10),
        ]);
        let raised_c = Some(vec![finding("c.go", 10, 10, Severity::Minor)]);
        let cases = [
            ("failed_paths names a.go", Some(vec!["a.go".to_owned()]), 1),
            (
                "failed_paths names c.go, which was raised",
                Some(vec!["c.go".to_owned()]),
                0,
            ),
            ("count fallback loses every missed label", None, 2),
        ];
        for (name, failed_paths, want) in cases {
            let run = FixtureFindings {
                fixture: "fx".to_owned(),
                failed_files: 1,
                failed_paths,
                findings: raised_c.clone(),
            };
            let card = score(&labels, &[run]).unwrap();
            assert_eq!(card.accepted.lost, want, "{name}");
            assert_eq!(card.accepted.raised, 1, "{name}");
        }
    }

    #[test]
    fn counts_severity_unmatched_and_skips_unlabeled_fixtures() {
        let mut planted = label(LabelKind::Planted, "a.go", 10, 10);
        planted.scope = Some(Scope::CrossFile);
        let labels = one(vec![planted, label(LabelKind::Rejected, "a.go", 50, 50)]);
        let runs: Vec<FixtureFindings> = serde_json::from_str(
            r#"[
              {"fixture": "fx", "failed_files": 0, "inline": null, "findings": [
                {"path": "a.go", "start_line": 9, "end_line": 9, "severity": "minor", "title": "t"},
                {"path": "a.go", "start_line": 11, "end_line": 11, "severity": "major"},
                {"path": "a.go", "start_line": 49, "end_line": 49, "severity": "nitpick"},
                {"path": "a.go", "start_line": 30, "end_line": 30, "severity": "major"}
              ]},
              {"fixture": "intent", "failed_files": 3, "findings": [
                {"path": "x.go", "start_line": 1, "end_line": 1, "severity": "major"}
              ]}
            ]"#,
        )
        .unwrap();
        let card = score(&labels, &runs).unwrap();
        assert_eq!(card.planted_cross_file, row(1, 1, 0, 1));
        assert_eq!(card.rejected, row(1, 1, 0, 0));
        assert_eq!(
            (
                card.fixtures,
                card.findings,
                card.failed_files,
                card.unmatched
            ),
            (1, 4, 0, 1)
        );
    }

    #[test]
    fn null_findings_score_as_none() {
        let labels = one(vec![label(LabelKind::Accepted, "a.go", 10, 10)]);
        let runs: Vec<FixtureFindings> = serde_json::from_str(
            r#"[{"fixture": "fx", "failed_files": 0, "findings": null, "inline": null}]"#,
        )
        .unwrap();
        let card = score(&labels, &runs).unwrap();
        assert_eq!(card.accepted, row(1, 0, 0, 0));
        assert_eq!(card.findings, 0);
    }

    #[test]
    fn evidence_is_optional() {
        let labels: Vec<Label> = serde_json::from_str(
            r#"[
              {"path": "a.go", "start_line": 1, "end_line": 2, "severity": "major",
               "label": "planted"},
              {"path": "a.go", "start_line": 1, "end_line": 2, "severity": "major",
               "label": "planted", "scope": "cross-file",
               "evidence": [{"path": "b.go", "line": 7}]}
            ]"#,
        )
        .unwrap();
        assert!(labels[0].evidence.is_empty());
        assert_eq!(
            labels[1].evidence,
            vec![Evidence {
                path: "b.go".to_owned(),
                line: 7
            }]
        );
    }

    fn evidence(path: &str, line: i64) -> Evidence {
        Evidence {
            path: path.to_owned(),
            line,
        }
    }

    fn snippet(kind: SnippetKind, path: &str, start: i64, end: i64, bytes: usize) -> Snippet {
        Snippet {
            kind,
            symbol: "F".to_owned(),
            path: path.to_owned(),
            start_line: start,
            end_line: end,
            text: "x".repeat(bytes),
        }
    }

    fn context(packs: BTreeMap<String, ContextPack>) -> FixtureContext {
        FixtureContext {
            repo: "acme/r".to_owned(),
            sha: "0".repeat(40),
            files: vec![("a.go".to_owned(), FileStatus::Modified)],
            conventions: String::new(),
            packs,
            ts_imports: (0, 0),
        }
    }

    #[test]
    fn coverage_outcomes() {
        let mut planted = label(LabelKind::Planted, "a.go", 10, 10);
        planted.scope = Some(Scope::CrossFile);
        planted.evidence = vec![
            evidence("x.go", 15),
            evidence("y.go", 5),
            evidence("v.go", 3),
            evidence("z.go", 2),
            evidence("z.go", 3),
            evidence("x.go", 21),
            evidence("w.go", 1),
        ];
        let mut no_context = label(LabelKind::Planted, "b.go", 1, 1);
        no_context.evidence = vec![evidence("x.go", 15)];
        let labels = Labels::from([
            (
                "fx".to_owned(),
                LabeledFixture {
                    labels: vec![planted],
                },
            ),
            (
                "gone".to_owned(),
                LabeledFixture {
                    labels: vec![no_context],
                },
            ),
        ]);
        let pack = ContextPack {
            snippets: vec![
                snippet(SnippetKind::Definition, "x.go", 10, 20, 40), // 10 tokens
                snippet(SnippetKind::Caller, "y.go", 1, 30, 400),     // 100
                snippet(SnippetKind::Caller, "v.go", 1, 5, 40),       // 10
            ],
            related: vec![RelatedDiff {
                path: "z.go".to_owned(),
                relation: Relation::ImportedBy,
                patch: "@@ -1,2 +1,3 @@\n a\n+b\n c".to_owned(), // 6
            }],
            error: None,
        };
        let contexts = BTreeMap::from([(
            "fx".to_owned(),
            context(BTreeMap::from([("a.go".to_owned(), pack)])),
        )]);
        let cov = coverage(&labels, &contexts, 50);

        let snip = |kind, index| Found::Snippet { kind, index };
        let got: Vec<(&str, &str, i64, Reach)> = cov
            .evidence
            .iter()
            .map(|e| (e.fixture.as_str(), e.path.as_str(), e.line, e.reach))
            .collect();
        let want = vec![
            (
                "fx",
                "x.go",
                15,
                Reach::Hit(snip(SnippetKind::Definition, 0)),
            ),
            ("fx", "y.go", 5, Reach::Cut(snip(SnippetKind::Caller, 1))),
            // The cut snippet no longer counts: v.go is second in the fitted pack.
            ("fx", "v.go", 3, Reach::Hit(snip(SnippetKind::Caller, 1))),
            (
                "fx",
                "z.go",
                2,
                Reach::Hit(Found::Related {
                    relation: Relation::ImportedBy,
                    index: 0,
                }),
            ),
            (
                "fx",
                "z.go",
                3,
                Reach::Hit(Found::Related {
                    relation: Relation::ImportedBy,
                    index: 0,
                }),
            ),
            ("fx", "x.go", 21, Reach::Missed),
            ("fx", "w.go", 1, Reach::Missed),
            ("gone", "x.go", 15, Reach::Missed),
        ];
        assert_eq!(got, want);
        let row = |evidence, hit, cut, missed| CoverageRow {
            evidence,
            hit,
            cut,
            missed,
        };
        assert_eq!(cov.planted_cross_file, row(7, 4, 1, 2));
        assert_eq!(cov.planted_diff, row(1, 0, 0, 1));
        assert_eq!(cov.rejected, CoverageRow::default());
        assert_eq!((cov.go_loaded, cov.go_skipped, cov.packs.len()), (1, 0, 1));
        assert_eq!(cov.packs[0].tokens, 126);

        // A related diff whose new side lacks the line is no hit.
        let mut narrow = labels.clone();
        narrow.get_mut("fx").unwrap().labels[0].evidence = vec![evidence("z.go", 9)];
        let cov = coverage(&narrow, &contexts, 50);
        assert_eq!(cov.evidence[0].reach, Reach::Missed);
    }

    #[test]
    fn render_coverage_report() {
        let mut planted = label(LabelKind::Planted, "a.go", 10, 10);
        planted.evidence = vec![evidence("x.go", 15), evidence("q.go", 1)];
        let labels = one(vec![planted]);
        let pack = ContextPack {
            snippets: vec![snippet(SnippetKind::Caller, "x.go", 10, 20, 8)],
            related: Vec::new(),
            error: Some("load: a | b\nc".to_owned()),
        };
        let mut ctx = context(BTreeMap::from([("a.go".to_owned(), pack)]));
        ctx.ts_imports = (99, 100);
        let ctx: FixtureContext =
            serde_json::from_str(&serde_json::to_string(&ctx).unwrap()).unwrap();
        let cov = coverage(&labels, &BTreeMap::from([("fx".to_owned(), ctx)]), 6000);
        let want = [
            "Evidence at a budget of 6000 tokens:",
            "",
            "| Fixture | Evidence | Reached | Found as |",
            "|---|---|---|---|",
            "| fx | x.go:15 | **hit** | caller, snippet 1 |",
            "| fx | q.go:1 | missed |  |",
            "",
            "| Label set | Evidence lines | Hit | Cut | Missed |",
            "|---|---|---|---|---|",
            "| **rejected** (known false alarms; lower is better) | 0 | **0** | 0 | 0 |",
            "| **accepted** (known real issues; higher is better) | 0 | **0** | 0 | 0 |",
            "| planted, visible in the diff | 2 | **1** | 0 | 1 |",
            "| planted, provable only from another file | 0 | **0** | 0 | 0 |",
            "",
            "| Fixture | Pack | Snippets | Related | Tokens | Error |",
            "|---|---|---|---|---|---|",
            "| fx | a.go | 1 | 0 | 2 | load: a \\| b c |",
            "",
            "TS resolution 99/100 code specifiers (99%); Go files 0 loaded, 1 skipped.",
        ];
        assert_eq!(render_coverage(&cov).lines().collect::<Vec<_>>(), want);
    }

    /// Right: a rejected label not raised, an accepted or planted one raised. A label
    /// lost to the failed `c.go` is wrong either way.
    #[test]
    fn outcomes_per_label() {
        let (labels, runs) = sample();
        let want = [
            ("rejected, raised", false),
            ("rejected, not raised", true),
            ("rejected, lost", false),
            ("accepted, 3 lines off", true),
            ("accepted, raised", true),
            ("accepted, lost", false),
            ("planted, raised", true),
            ("planted cross-file, missed", false),
        ];
        let got = outcomes(&labels, &runs);
        assert_eq!(got.len(), want.len());
        for (got, (name, want)) in got.into_iter().zip(want) {
            assert_eq!(got, want, "{name}");
        }
        let none = outcomes(&labels, &[]);
        assert_eq!(none.len(), want.len());
        assert!(
            none.iter().all(|&g| !g),
            "a missing fixture gets nothing right"
        );
    }

    /// One run per side, from (base, candidate) outcomes per label.
    fn one_each(pairs: &[(bool, bool)]) -> (Vec<Vec<bool>>, Vec<Vec<bool>>) {
        let (base, cand) = pairs.iter().copied().unzip();
        (vec![base], vec![cand])
    }

    /// `n` copies of (base, candidate).
    fn times(n: usize, pair: (bool, bool)) -> Vec<(bool, bool)> {
        vec![pair; n]
    }

    #[test]
    fn paired_with_one_run_each_is_the_sign_test() {
        let (gain, lose) = ((false, true), (true, false));
        let agree = [(true, true), (false, false), (true, true)];
        let cases = [
            (
                "5-0",
                [times(5, gain), times(0, lose)].concat(),
                (5, 0),
                "0.031",
            ),
            (
                "7-1",
                [times(7, gain), times(1, lose)].concat(),
                (7, 1),
                "0.035",
            ),
            (
                "9-1",
                [times(9, gain), times(1, lose)].concat(),
                (9, 1),
                "0.011",
            ),
            ("no disagreements", Vec::new(), (0, 0), "1.000"),
            ("0-3", times(3, lose), (0, 3), "1.000"),
        ];
        for (name, pairs, (gained, lost), p) in cases {
            let (base, cand) = one_each(&[pairs, agree.to_vec()].concat());
            let got = paired(&base, &cand);
            assert_eq!(
                (got.gained, got.lost, got.d),
                (gained, lost, gained + lost),
                "{name}"
            );
            assert_eq!(format!("{:.3}", got.p), p, "{name}");
        }
    }

    #[test]
    fn paired_with_more_runs_on_one_side() {
        let run = vec![true, false, true, false];
        let same = paired(&[run.clone(), run.clone()], &[run]);
        assert_eq!((same.gained, same.lost, same.d, same.p), (0, 0, 0, 1.0));

        // Base wrong twice, candidate right once: under no difference, the one right
        // run of three is the candidate's with chance 1/3, not the sign test's 1/2.
        let got = paired(&[vec![false], vec![false]], &[vec![true]]);
        assert_eq!((got.gained, got.lost, got.d), (1, 0, 1));
        assert!((got.p - 1.0 / 3.0).abs() < 1e-12, "{}", got.p);

        // Shares are compared: right in 1 of 2 base runs and in the candidate's one
        // is a gain, which two right runs of three make likely (2/3).
        let half = paired(&[vec![true], vec![false]], &[vec![true]]);
        assert_eq!((half.gained, half.lost), (1, 0));
        assert!((half.p - 2.0 / 3.0).abs() < 1e-12, "{}", half.p);
    }

    #[test]
    fn errors() {
        let labels = one(vec![label(LabelKind::Accepted, "a.go", 10, 10)]);
        assert_eq!(
            score(&labels, &[]),
            Err(ScoreError::Missing("fx".to_owned()))
        );
        assert_eq!(
            merge([labels.clone(), labels]),
            Err(ScoreError::Duplicate("fx".to_owned()))
        );
    }
}
