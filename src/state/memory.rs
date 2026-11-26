//! The in-memory store: dev mode (no `DATABASE_URL`) and fast tests elsewhere in the crate.
//! It lives as long as the process. Every rule `PgStore` gets from SQL is spelled out here,
//! and the contract suite holds the two to the same behavior.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, SystemTime};

use async_trait::async_trait;

use super::{
    Claimed, EventKind, FindingEvent, FindingStatus, JobEnd, LessonKind, NewJob, NewLesson,
    PrState, Repo, Store, StoreError, StoredFinding, StoredLesson, THREAD_SLOT, Usage,
};
use crate::finding::Severity;
use crate::learn;

#[derive(Debug, Default)]
pub struct MemoryStore(Mutex<Data>);

#[derive(Debug, Default)]
struct Data {
    repos: HashMap<i64, Repo>,
    prs: HashMap<(i64, i64), PrState>,
    /// Per (repo, PR), by Fingerprint. Sorted, so a lookup by comment returns the smallest
    /// Fingerprint, as Postgres's `ORDER BY` does.
    findings: HashMap<(i64, i64), BTreeMap<String, StoredFinding>>,
    /// When each Finding was first stored, per (repo, PR), by Fingerprint.
    born: HashMap<(i64, i64), HashMap<String, SystemTime>>,
    events: HashMap<(i64, i64), Vec<FindingEvent>>,
    pushers: HashMap<(i64, i64), BTreeSet<String>>,
    /// Every repo's Lessons, oldest first; a revived Lesson moves to the end.
    lessons: Vec<StoredLesson>,
    last_lesson_id: i64,
    usage: Vec<Usage>,
    /// The job queue, by id, so ties on the due time go to the older job as in Postgres.
    jobs: BTreeMap<i64, JobRow>,
    last_job_id: i64,
}

/// A `jobs` row; the error text is for operators reading Postgres, so it is not kept.
#[derive(Debug)]
struct JobRow {
    delivery: String,
    slot: String,
    payload: String,
    state: JobState,
    run_at: SystemTime,
    attempts: i32,
    updated_at: SystemTime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JobState {
    Queued,
    Running {
        lease_until: SystemTime,
    },
    /// Done or dead: never claimed again, and pruned alike.
    Ended,
}

impl MemoryStore {
    fn data(&self) -> MutexGuard<'_, Data> {
        // No method can panic halfway through an update, so a poisoned lock still guards
        // consistent data.
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Data {
    /// The foreign key Postgres enforces on every `repo_id`.
    fn require_repo(&self, repo_id: i64) -> Result<(), StoreError> {
        if self.repos.contains_key(&repo_id) {
            Ok(())
        } else {
            Err(StoreError::UnknownRepo(repo_id))
        }
    }

    /// Moves the named Findings to `to`, if their status is `from` (any when `None`).
    fn transition(
        &mut self,
        (repo_id, pr): (i64, i64),
        fingerprints: &[String],
        from: Option<FindingStatus>,
        to: FindingStatus,
    ) {
        let Some(findings) = self.findings.get_mut(&(repo_id, pr)) else {
            return;
        };
        for fp in fingerprints {
            if let Some(f) = findings.get_mut(fp)
                && from.is_none_or(|s| s == f.status)
            {
                f.status = to;
            }
        }
    }

    /// The repo's Lessons whose id is in `ids`.
    fn lessons_mut<'a>(
        &'a mut self,
        repo_id: i64,
        ids: &'a [i64],
    ) -> impl Iterator<Item = &'a mut StoredLesson> {
        self.lessons
            .iter_mut()
            .filter(move |s| s.repo_id == repo_id && ids.contains(&s.lesson.id))
    }
}

#[async_trait]
impl Store for MemoryStore {
    async fn upsert_repo(&self, repo: &Repo) -> Result<(), StoreError> {
        self.data().repos.insert(repo.id, repo.clone());
        Ok(())
    }

    async fn pr_state(&self, repo_id: i64, pr: i64) -> Result<PrState, StoreError> {
        Ok(self
            .data()
            .prs
            .get(&(repo_id, pr))
            .cloned()
            .unwrap_or_default())
    }

    async fn save_pr_state(
        &self,
        repo_id: i64,
        pr: i64,
        state: &PrState,
    ) -> Result<(), StoreError> {
        let mut d = self.data();
        d.require_repo(repo_id)?;
        d.prs.insert((repo_id, pr), state.clone());
        Ok(())
    }

    async fn findings(
        &self,
        repo_id: i64,
        pr: i64,
        status: FindingStatus,
    ) -> Result<Vec<StoredFinding>, StoreError> {
        let d = self.data();
        let mut out: Vec<StoredFinding> = d
            .findings
            .get(&(repo_id, pr))
            .into_iter()
            .flat_map(BTreeMap::values)
            .filter(|f| f.status == status)
            .cloned()
            .collect();
        out.sort_by(|a, b| {
            (&a.path, a.start_line, &a.fingerprint).cmp(&(&b.path, b.start_line, &b.fingerprint))
        });
        Ok(out)
    }

    async fn findings_before(
        &self,
        repo_id: i64,
        pr: i64,
        status: FindingStatus,
        at: SystemTime,
    ) -> Result<Vec<StoredFinding>, StoreError> {
        let mut out = self.findings(repo_id, pr, status).await?;
        let d = self.data();
        let born = d.born.get(&(repo_id, pr));
        out.retain(|f| {
            born.and_then(|b| b.get(&f.fingerprint))
                .is_some_and(|t| *t <= at)
        });
        Ok(out)
    }

    async fn upsert_finding(&self, f: &StoredFinding) -> Result<(), StoreError> {
        let mut d = self.data();
        d.require_repo(f.repo_id)?;
        d.born
            .entry((f.repo_id, f.pr))
            .or_default()
            .entry(f.fingerprint.clone())
            .or_insert_with(SystemTime::now);
        let findings = d.findings.entry((f.repo_id, f.pr)).or_default();
        let mut next = f.clone();
        if let Some(old) = findings.get(&f.fingerprint) {
            next.first_sha.clone_from(&old.first_sha);
            next.comment_id = f.comment_id.or(old.comment_id);
            if old.status == FindingStatus::Suppressed && f.severity != Severity::Critical {
                next.status = FindingStatus::Suppressed;
            }
        }
        findings.insert(f.fingerprint.clone(), next);
        Ok(())
    }

    async fn resolve_findings(
        &self,
        repo_id: i64,
        pr: i64,
        fingerprints: &[String],
    ) -> Result<(), StoreError> {
        self.data().transition(
            (repo_id, pr),
            fingerprints,
            Some(FindingStatus::Open),
            FindingStatus::Resolved,
        );
        Ok(())
    }

    async fn finding_by_comment(
        &self,
        repo_id: i64,
        pr: i64,
        comment_id: i64,
    ) -> Result<Option<StoredFinding>, StoreError> {
        let d = self.data();
        Ok(d.findings
            .get(&(repo_id, pr))
            .and_then(|m| m.values().find(|f| f.comment_id == Some(comment_id)))
            .cloned())
    }

    async fn finding_by_fingerprint(
        &self,
        repo_id: i64,
        pr: i64,
        fingerprint: &str,
    ) -> Result<Option<StoredFinding>, StoreError> {
        let d = self.data();
        Ok(d.findings
            .get(&(repo_id, pr))
            .and_then(|m| m.get(fingerprint))
            .cloned())
    }

    async fn append_finding_event(
        &self,
        repo_id: i64,
        pr: i64,
        fingerprint: &str,
        kind: EventKind,
        sha: Option<&str>,
    ) -> Result<(), StoreError> {
        let mut d = self.data();
        let known = d
            .findings
            .get(&(repo_id, pr))
            .is_some_and(|m| m.contains_key(fingerprint));
        if known {
            d.events
                .entry((repo_id, pr))
                .or_default()
                .push(FindingEvent {
                    fingerprint: fingerprint.to_owned(),
                    kind,
                    sha: sha.map(str::to_owned),
                });
        }
        Ok(())
    }

    async fn finding_events(&self, repo_id: i64, pr: i64) -> Result<Vec<FindingEvent>, StoreError> {
        Ok(self
            .data()
            .events
            .get(&(repo_id, pr))
            .cloned()
            .unwrap_or_default())
    }

    async fn suppress_finding(
        &self,
        repo_id: i64,
        pr: i64,
        fingerprint: &str,
    ) -> Result<(), StoreError> {
        self.data().transition(
            (repo_id, pr),
            &[fingerprint.to_owned()],
            None,
            FindingStatus::Suppressed,
        );
        Ok(())
    }

    async fn unsuppress_finding(
        &self,
        repo_id: i64,
        pr: i64,
        fingerprint: &str,
    ) -> Result<(), StoreError> {
        self.data().transition(
            (repo_id, pr),
            &[fingerprint.to_owned()],
            Some(FindingStatus::Suppressed),
            FindingStatus::Open,
        );
        Ok(())
    }

    async fn add_pusher(&self, repo_id: i64, pr: i64, login: &str) -> Result<(), StoreError> {
        let mut d = self.data();
        d.require_repo(repo_id)?;
        let logins = d.pushers.entry((repo_id, pr)).or_default();
        logins.insert(login.to_owned());
        Ok(())
    }

    async fn pushers(&self, repo_id: i64, pr: i64) -> Result<Vec<String>, StoreError> {
        let d = self.data();
        let logins = d.pushers.get(&(repo_id, pr));
        Ok(logins.into_iter().flatten().cloned().collect())
    }

    async fn add_lesson(&self, l: &NewLesson) -> Result<i64, StoreError> {
        let mut d = self.data();
        d.require_repo(l.repo_id)?;
        let now = SystemTime::now();
        let same_rule = d
            .lessons
            .iter()
            .position(|s| s.repo_id == l.repo_id && s.lesson.rule == l.rule);
        let lesson = match same_rule {
            Some(i) => {
                let mut s = d.lessons.remove(i);
                s.lesson.created_at = Some(now);
                s.lesson.confidence = l.confidence;
                s.author.clone_from(&l.author);
                s.pr = l.pr.or(s.pr);
                s.source_comment_id = l.source_comment_id.or(s.source_comment_id);
                if l.kind != LessonKind::Unknown {
                    s.kind = l.kind;
                }
                s
            }
            None => {
                d.last_lesson_id += 1;
                StoredLesson {
                    lesson: learn::Lesson {
                        id: d.last_lesson_id,
                        rule: l.rule.clone(),
                        scope: l.scope.clone(),
                        created_at: Some(now),
                        confidence: l.confidence,
                        ..learn::Lesson::default()
                    },
                    repo_id: l.repo_id,
                    kind: l.kind,
                    author: l.author.clone(),
                    pr: l.pr,
                    source_comment_id: l.source_comment_id,
                    decayed_confidence: None,
                }
            }
        };
        let id = lesson.lesson.id;
        d.lessons.push(lesson);
        Ok(id)
    }

    async fn lessons(
        &self,
        repo_id: i64,
        limit: Option<usize>,
    ) -> Result<Vec<StoredLesson>, StoreError> {
        let d = self.data();
        Ok(d.lessons
            .iter()
            .rev()
            .filter(|s| s.repo_id == repo_id)
            .take(limit.unwrap_or(usize::MAX))
            .cloned()
            .collect())
    }

    async fn lesson_by_source_comment(
        &self,
        repo_id: i64,
        comment_id: i64,
    ) -> Result<Option<StoredLesson>, StoreError> {
        let d = self.data();
        Ok(d.lessons
            .iter()
            .filter(|s| s.repo_id == repo_id && s.source_comment_id == Some(comment_id))
            .max_by_key(|s| s.lesson.id)
            .cloned())
    }

    async fn mark_lessons_triggered(&self, repo_id: i64, ids: &[i64]) -> Result<(), StoreError> {
        let now = SystemTime::now();
        for s in self.data().lessons_mut(repo_id, ids) {
            s.lesson.trigger_count += 1;
            s.lesson.last_triggered_at = Some(now);
        }
        Ok(())
    }

    async fn delete_lesson(&self, repo_id: i64, id: i64) -> Result<bool, StoreError> {
        let mut d = self.data();
        let before = d.lessons.len();
        d.lessons
            .retain(|s| !(s.repo_id == repo_id && s.lesson.id == id));
        Ok(d.lessons.len() < before)
    }

    async fn set_decayed_confidence(
        &self,
        repo_id: i64,
        updates: &[(i64, f64)],
    ) -> Result<(), StoreError> {
        let mut d = self.data();
        for &(id, value) in updates {
            for s in d.lessons_mut(repo_id, &[id]) {
                s.decayed_confidence = Some(value);
            }
        }
        Ok(())
    }

    async fn repos_with_lessons(&self) -> Result<Vec<i64>, StoreError> {
        let d = self.data();
        let repos: BTreeSet<i64> = d.lessons.iter().map(|s| s.repo_id).collect();
        Ok(repos.into_iter().collect())
    }

    async fn record_usage(&self, u: &Usage) -> Result<(), StoreError> {
        let mut d = self.data();
        d.require_repo(u.repo_id)?;
        d.usage.push(u.clone());
        Ok(())
    }

    async fn usage(&self, repo_id: i64, pr: i64) -> Result<Vec<Usage>, StoreError> {
        let d = self.data();
        Ok(d.usage
            .iter()
            .filter(|u| u.repo_id == repo_id && u.pr == pr)
            .cloned()
            .collect())
    }

    async fn enqueue(&self, j: &NewJob) -> Result<bool, StoreError> {
        let mut d = self.data();
        if d.jobs.values().any(|row| row.delivery == j.delivery) {
            return Ok(false);
        }
        let now = SystemTime::now();
        d.last_job_id += 1;
        let id = d.last_job_id;
        d.jobs.insert(
            id,
            JobRow {
                delivery: j.delivery.clone(),
                slot: j.slot.clone(),
                payload: j.payload.clone(),
                state: JobState::Queued,
                run_at: now + j.delay,
                attempts: 0,
                updated_at: now,
            },
        );
        Ok(true)
    }

    async fn claim(
        &self,
        lease: Duration,
        threads_only: bool,
    ) -> Result<Option<Claimed>, StoreError> {
        let now = SystemTime::now();
        let mut d = self.data();
        for row in d.jobs.values_mut() {
            if matches!(row.state, JobState::Running { lease_until } if lease_until <= now) {
                row.state = JobState::Queued;
                row.updated_at = now;
            }
        }
        let running: HashSet<&str> = d
            .jobs
            .values()
            .filter(|row| matches!(row.state, JobState::Running { .. }))
            .map(|row| row.slot.as_str())
            .collect();
        let next = d
            .jobs
            .iter()
            .filter(|(_, row)| {
                row.state == JobState::Queued
                    && row.run_at <= now
                    && !running.contains(row.slot.as_str())
                    && (!threads_only || row.slot.starts_with(THREAD_SLOT))
            })
            .min_by_key(|&(&id, row)| (!row.slot.starts_with(THREAD_SLOT), row.run_at, id))
            .map(|(&id, _)| id);
        let Some((id, row)) = next.and_then(|id| d.jobs.get_mut(&id).map(|row| (id, row))) else {
            return Ok(None);
        };
        row.state = JobState::Running {
            lease_until: now + lease,
        };
        row.attempts += 1;
        row.updated_at = now;
        Ok(Some(Claimed {
            id,
            delivery: row.delivery.clone(),
            attempt: row.attempts,
            payload: row.payload.clone(),
        }))
    }

    async fn queued_in(&self, slot: &str) -> Result<Vec<String>, StoreError> {
        let d = self.data();
        let mut queued: Vec<(&i64, &JobRow)> = d
            .jobs
            .iter()
            .filter(|(_, row)| row.state == JobState::Queued && row.slot == slot)
            .collect();
        queued.sort_by_key(|&(&id, row)| (row.run_at, id));
        Ok(queued
            .into_iter()
            .map(|(_, row)| row.payload.clone())
            .collect())
    }

    async fn finish(&self, id: i64, attempt: i32, end: JobEnd) -> Result<(), StoreError> {
        let mut d = self.data();
        let Some(row) = d
            .jobs
            .get_mut(&id)
            .filter(|row| row.attempts == attempt && matches!(row.state, JobState::Running { .. }))
        else {
            return Ok(());
        };
        match end {
            JobEnd::Retry { after, .. } => {
                row.state = JobState::Queued;
                row.run_at = SystemTime::now() + after;
            }
            JobEnd::Done | JobEnd::Dead { .. } => row.state = JobState::Ended,
        }
        row.updated_at = SystemTime::now();
        Ok(())
    }

    async fn prune_jobs(&self, older_than: Duration) -> Result<u64, StoreError> {
        let Some(cutoff) = SystemTime::now().checked_sub(older_than) else {
            return Ok(0);
        };
        let mut d = self.data();
        let before = d.jobs.len();
        d.jobs
            .retain(|_, row| !(row.state == JobState::Ended && row.updated_at <= cutoff));
        Ok((before - d.jobs.len()) as u64)
    }
}
