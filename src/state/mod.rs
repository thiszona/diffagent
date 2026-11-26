//! Durable state across pushes and replies: which Findings each PR has open, resolved or
//! suppressed, per-PR bookkeeping, the Lessons Maintainers taught, and what each Review
//! cost. One [`Store`] trait, two implementations: [`PgStore`] in production and
//! [`MemoryStore`] for dev mode and tests. One contract suite runs against both, so they
//! cannot drift apart.
//!
//! In the schema, repos are keyed by GitHub's numeric id, so a rename keeps their
//! state; Findings carry their [`Category`], suppression is a [`FindingStatus`] rather
//! than a flag, and "none" is `None`, never 0 or "".

#[cfg(test)]
pub(crate) mod contract;
mod memory;
mod postgres;
mod reconcile;

use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::finding::{Category, Severity};
use crate::learn;
use crate::plan::OutcomeStats;

pub use memory::MemoryStore;
pub use postgres::PgStore;
pub use reconcile::{Reconciliation, reconcile};

/// A repository the App is installed on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Repo {
    /// GitHub's numeric repository id: it survives renames and transfers, so a repo's
    /// Findings and Lessons do too.
    pub id: i64,
    pub installation_id: i64,
    /// `owner/name` as last seen; display only.
    pub full_name: String,
}

/// Per-PR bookkeeping. The default is a PR the store has never seen.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PrState {
    /// The status comment to keep editing instead of posting another.
    pub status_comment_id: Option<i64>,
    /// Muted by a `pause` command until the next `review` command.
    pub paused: bool,
    pub last_reviewed_sha: Option<String>,
    /// When a completed Review last saved `last_reviewed_sha`: a `review` command posted
    /// before it is spent.
    pub last_reviewed_at: Option<SystemTime>,
    /// The head a Maintainer approved; a new push changes the head and re-arms the gate.
    pub approved_sha: Option<String>,
    /// The cached Walkthrough, opaque to the store, so a push that touches no new files
    /// skips the whole-PR model calls.
    pub walkthrough: Option<String>,
}

/// Where a Finding is in its lifecycle. Stored as its serde name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FindingStatus {
    /// In the latest Review of its file.
    Open,
    /// Gone from a later Review, or closed by a `resolve` command.
    Resolved,
    /// A Maintainer rejected it: not posted again on this PR unless it turns critical.
    Suppressed,
}

/// A Finding as the store tracks it across pushes. Body, snippet and suggestion are not
/// kept: the Thread has them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredFinding {
    pub repo_id: i64,
    pub pr: i64,
    /// From `identity::fingerprint_findings`; the identity within the PR.
    pub fingerprint: String,
    pub path: String,
    pub start_line: i64,
    pub end_line: i64,
    pub severity: Severity,
    pub category: Category,
    pub title: String,
    /// The Thread root, once its comment is posted.
    pub comment_id: Option<i64>,
    pub status: FindingStatus,
    /// The head where it was first seen; later upserts keep it.
    pub first_sha: String,
    /// The head where it was last seen or carried.
    pub last_sha: String,
}

/// A lifecycle transition in a Finding's append-only history. Stored as its serde name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    /// First raised on the PR.
    New,
    /// Raised again while open.
    Reaffirm,
    /// Gone from a reviewed file, or closed by a `resolve` command.
    Resolve,
    /// Still open on a file this push did not review.
    Carry,
    Suppress,
    /// A `forget` undid a suppression.
    Unsuppress,
    /// A Maintainer said they will fix it; it stays open until a push resolves it.
    Acknowledge,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FindingEvent {
    pub fingerprint: String,
    pub kind: EventKind,
    /// The head commit; `None` for an event caused by a reply, which has none.
    pub sha: Option<String>,
}

/// What kind of correction taught a Lesson. Stored as its serde name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LessonKind {
    /// Not recorded.
    Unknown,
    /// The Finding was wrong.
    FalsePositive,
    /// The Finding was right, but the team does it this way on purpose.
    Convention,
}

/// A Lesson to record.
#[derive(Debug, Clone, PartialEq)]
pub struct NewLesson {
    pub repo_id: i64,
    pub rule: String,
    pub kind: LessonKind,
    /// The directory subtree it applies to; empty is global. Re-teaching the rule keeps
    /// the first scope, so a rule is never narrowed behind a Maintainer's back.
    pub scope: String,
    /// Explicit: the store has no default.
    pub confidence: f64,
    pub author: String,
    pub pr: Option<i64>,
    /// The root of the Thread it was learned in: one Lesson per Thread, so
    /// `forget` in the Thread and a rerun of the reply find it, and the Thread is its
    /// provenance.
    pub source_comment_id: Option<i64>,
}

/// A stored Lesson: what ranking and decay read, plus provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredLesson {
    /// The fields [`learn::rank`] and [`learn::decay`] read; `created_at` is always set.
    pub lesson: learn::Lesson,
    pub repo_id: i64,
    pub kind: LessonKind,
    pub author: String,
    pub pr: Option<i64>,
    pub source_comment_id: Option<i64>,
    /// The last decay run's value. Measured only: ranking still reads the live
    /// confidence.
    pub decayed_confidence: Option<f64>,
}

/// The model role a usage row is for, as in `config::Models`. Stored as its serde name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelRole {
    Review,
    Summary,
    Verify,
    Classify,
}

/// What one model role cost in one Review.
#[derive(Debug, Clone, PartialEq)]
pub struct Usage {
    pub repo_id: i64,
    pub pr: i64,
    pub head_sha: String,
    pub role: ModelRole,
    pub provider: String,
    pub model: String,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cache_read_tokens: i64,
    pub cache_write_tokens: i64,
    pub cost_usd: f64,
    /// The Review's anchoring outcome, on the review role's row; default on the others.
    pub anchors: OutcomeStats,
    pub failed_files: i64,
    /// Reviewable files; `kept_files - reviewed_files` is what an incremental push saved.
    pub kept_files: i64,
    pub reviewed_files: i64,
}

/// A webhook delivery to queue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewJob {
    /// `X-GitHub-Delivery`: the dedupe key.
    pub delivery: String,
    /// `pr:<repo_id>:<n>` or `thread:<repo_id>:<root>`; one job per slot runs at a time.
    pub slot: String,
    /// The `job::Job`, as JSON.
    pub payload: String,
    /// How long after the insert it becomes due. The store's clock adds it, so the due
    /// time and the claim compare on one clock.
    pub delay: Duration,
}

/// A job taken by [`Store::claim`], leased to one worker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Claimed {
    pub id: i64,
    pub delivery: String,
    /// 1 on the first claim; a lapsed lease or a retry claims it again with one more.
    pub attempt: i32,
    pub payload: String,
}

/// How an attempt ended, for [`Store::finish`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobEnd {
    Done,
    /// Queued again, due `after` from now on the store's clock, the one the claim
    /// compares against: a due time read off the pod's clock was off by its skew.
    Retry {
        after: Duration,
        error: String,
    },
    /// Never claimed again; pruned like a done job.
    Dead {
        error: String,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// A write named a repo that [`Store::upsert_repo`] never recorded.
    #[error("repo {0} is not recorded")]
    UnknownRepo(i64),
    #[error("database: {0}")]
    Database(#[from] sqlx::Error),
    #[error("migrating the schema: {0}")]
    Migrate(#[from] sqlx::migrate::MigrateError),
    /// A stored enum this build does not know, such as a severity a later version added.
    #[error("{column}: unknown value {value:?}")]
    UnknownValue { column: &'static str, value: String },
}

/// Persistence for the reviewer. Every write that names a repo fails with
/// [`StoreError::UnknownRepo`] until [`Store::upsert_repo`] has recorded it; ids from
/// another repo never match, so one repo can never touch another's rows.
#[async_trait]
pub trait Store: Send + Sync {
    /// Records a repo, or refreshes its installation and name after a rename or transfer.
    async fn upsert_repo(&self, repo: &Repo) -> Result<(), StoreError>;

    /// The PR's state; the default for a PR never saved.
    async fn pr_state(&self, repo_id: i64, pr: i64) -> Result<PrState, StoreError>;
    /// Replaces the PR's whole state: read, modify, then save, or fields are lost.
    async fn save_pr_state(&self, repo_id: i64, pr: i64, state: &PrState)
    -> Result<(), StoreError>;

    /// The PR's Findings in `status`, ordered by path, start line, then Fingerprint.
    async fn findings(
        &self,
        repo_id: i64,
        pr: i64,
        status: FindingStatus,
    ) -> Result<Vec<StoredFinding>, StoreError>;
    /// The PR's Findings in `status` first stored at or before `at`, in
    /// [`Store::findings`]' order: those a command posted at `at` could see. A later
    /// upsert keeps a Finding's first time.
    async fn findings_before(
        &self,
        repo_id: i64,
        pr: i64,
        status: FindingStatus,
        at: SystemTime,
    ) -> Result<Vec<StoredFinding>, StoreError>;
    /// Inserts a Finding, or updates the one with its Fingerprint: location, severity,
    /// title and last head always; the comment only when `f.comment_id` is set. The first
    /// head is kept, and a suppressed Finding stays suppressed unless `f` is critical: a
    /// Maintainer never dismisses a critical, so a suppression made on a lower grade does
    /// not cover one. Otherwise only [`Store::unsuppress_finding`] reopens it.
    async fn upsert_finding(&self, f: &StoredFinding) -> Result<(), StoreError>;
    /// Marks these open Findings resolved. Other statuses and unknown Fingerprints are
    /// left alone, so a suppressed Finding never turns resolved.
    async fn resolve_findings(
        &self,
        repo_id: i64,
        pr: i64,
        fingerprints: &[String],
    ) -> Result<(), StoreError>;
    /// The Finding whose Thread root is `comment_id`; the smallest Fingerprint if several
    /// share it.
    async fn finding_by_comment(
        &self,
        repo_id: i64,
        pr: i64,
        comment_id: i64,
    ) -> Result<Option<StoredFinding>, StoreError>;
    async fn finding_by_fingerprint(
        &self,
        repo_id: i64,
        pr: i64,
        fingerprint: &str,
    ) -> Result<Option<StoredFinding>, StoreError>;
    /// Appends a transition to a Finding's history. A Fingerprint with no stored Finding
    /// records nothing, so upsert the Finding first.
    async fn append_finding_event(
        &self,
        repo_id: i64,
        pr: i64,
        fingerprint: &str,
        kind: EventKind,
        sha: Option<&str>,
    ) -> Result<(), StoreError>;
    /// The PR's Finding history, oldest first.
    async fn finding_events(&self, repo_id: i64, pr: i64) -> Result<Vec<FindingEvent>, StoreError>;

    /// Suppresses a Finding whatever its status, so it is not posted again on this PR
    /// unless it turns critical.
    async fn suppress_finding(
        &self,
        repo_id: i64,
        pr: i64,
        fingerprint: &str,
    ) -> Result<(), StoreError>;
    /// Reopens a suppressed Finding (a `forget`, or a Review that finds it critical);
    /// any other status is left alone.
    async fn unsuppress_finding(
        &self,
        repo_id: i64,
        pr: i64,
        fingerprint: &str,
    ) -> Result<(), StoreError>;

    /// Records that `login` pushed to the PR; a repeat changes nothing.
    async fn add_pusher(&self, repo_id: i64, pr: i64, login: &str) -> Result<(), StoreError>;
    /// Everyone recorded as having pushed to the PR, in byte order.
    async fn pushers(&self, repo_id: i64, pr: i64) -> Result<Vec<String>, StoreError>;

    /// Records a Lesson and returns its id. Re-teaching a rule the repo already has
    /// revives that row instead: it becomes newest (its decay clock restarts) with the new
    /// confidence, author and active status; the new PR, source comment and kind replace
    /// the old only when known; the scope is kept.
    async fn add_lesson(&self, l: &NewLesson) -> Result<i64, StoreError>;
    /// The repo's active Lessons, newest first; `None` is all of them. The candidates
    /// for [`learn::rank`], which filters by scope.
    async fn lessons(
        &self,
        repo_id: i64,
        limit: Option<usize>,
    ) -> Result<Vec<StoredLesson>, StoreError>;
    /// The Lesson learned in the Thread rooted at `comment_id`, if any; the newest if
    /// several were.
    async fn lesson_by_source_comment(
        &self,
        repo_id: i64,
        comment_id: i64,
    ) -> Result<Option<StoredLesson>, StoreError>;
    /// Records that these Lessons were injected into a review prompt.
    async fn mark_lessons_triggered(&self, repo_id: i64, ids: &[i64]) -> Result<(), StoreError>;
    /// Deletes a Lesson (a `forget`); whether it existed.
    async fn delete_lesson(&self, repo_id: i64, id: i64) -> Result<bool, StoreError>;
    /// Stores `(id, value)` pairs as decayed confidence, all or none. The live confidence
    /// is never touched.
    async fn set_decayed_confidence(
        &self,
        repo_id: i64,
        updates: &[(i64, f64)],
    ) -> Result<(), StoreError>;
    /// The repos with at least one Lesson, ascending: the decay run's work list.
    async fn repos_with_lessons(&self) -> Result<Vec<i64>, StoreError>;

    async fn record_usage(&self, u: &Usage) -> Result<(), StoreError>;
    /// The PR's usage rows, oldest first.
    async fn usage(&self, repo_id: i64, pr: i64) -> Result<Vec<Usage>, StoreError>;

    /// Queues a delivery; false when it was already queued. Nothing is recorded before
    /// this insert, so a delivery that got a 500 can be redelivered.
    async fn enqueue(&self, j: &NewJob) -> Result<bool, StoreError>;
    /// Takes the next due job whose slot has nothing running, Thread slots first, then
    /// the oldest due time, and leases it for `lease`; `threads_only` takes only Thread
    /// slots, for the worker's Thread lane. Running jobs whose lease lapsed are queued
    /// again first, so a job whose pod died runs again. `None` when nothing is due, or
    /// when a concurrent claim took the slot first.
    async fn claim(
        &self,
        lease: Duration,
        threads_only: bool,
    ) -> Result<Option<Claimed>, StoreError>;
    /// The payloads of the jobs queued in `slot` and not running, oldest due first.
    async fn queued_in(&self, slot: &str) -> Result<Vec<String>, StoreError>;
    /// Ends attempt `attempt` of job `id`. A job no longer running under that attempt
    /// (its lease lapsed and it was claimed again) is left alone.
    async fn finish(&self, id: i64, attempt: i32, end: JobEnd) -> Result<(), StoreError>;
    /// Deletes done and dead jobs last changed at least `older_than` ago; how many.
    /// Their deliveries can then be queued again.
    async fn prune_jobs(&self, older_than: Duration) -> Result<u64, StoreError>;
}

/// Thread slots are claimed first: a reply is a person waiting, a Review is not.
const THREAD_SLOT: &str = "thread:";
