//! The Postgres store, for production. Queries are checked at run time rather than by
//! sqlx's `query!` macros: every one runs in the contract suite against a real Postgres,
//! which catches what the macros would, without a database or `.sqlx` files at build time.
//!
//! Reads use `SELECT *` and decode by column name, so one decoder per row type serves
//! every query and column order never matters.

use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use serde::Serialize;
use serde::de::DeserializeOwned;
use sqlx::Row;
use sqlx::postgres::{PgPool, PgPoolOptions, PgRow};
use time::OffsetDateTime;

use super::{
    Claimed, EventKind, FindingEvent, FindingStatus, JobEnd, NewJob, NewLesson, PrState, Repo,
    Store, StoreError, StoredFinding, StoredLesson, THREAD_SLOT, Usage,
};
use crate::learn;
use crate::plan::OutcomeStats;

#[derive(Debug, Clone)]
pub struct PgStore {
    pool: PgPool,
}

impl PgStore {
    /// Connects and applies any pending migration. sqlx holds an advisory lock while
    /// migrating, so pods starting together apply each migration once.
    ///
    /// Each pod opens up to 5 connections, not sqlx's 10: a pod needs 3 to 5 (two
    /// lanes, two claim loops, the webhook's inserts), so the database's connection
    /// limit needs 5 per pod.
    ///
    /// A migration this binary does not know is ignored, so a rollback to an older
    /// image starts on the newer schema instead of failing with `VersionMissing`. That
    /// holds only while migrations stay additive: an older binary must still work on
    /// every table and column it knows.
    pub async fn connect(url: &str) -> Result<Self, StoreError> {
        let pool = PgPoolOptions::new().max_connections(5).connect(url).await?;
        let mut migrator = sqlx::migrate!();
        migrator.set_ignore_missing(true);
        migrator.run(&pool).await?;
        Ok(Self { pool })
    }

    /// Moves the named Findings to `to`, if their status is `from` (any when `None`).
    async fn transition(
        &self,
        (repo_id, pr): (i64, i64),
        fingerprints: &[String],
        from: Option<FindingStatus>,
        to: FindingStatus,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "UPDATE findings SET status = $4, updated_at = now()
              WHERE repo_id = $1 AND pr = $2 AND fingerprint = ANY($3)
                AND ($5::text IS NULL OR status = $5)",
        )
        .bind(repo_id)
        .bind(pr)
        .bind(fingerprints)
        .bind(encode("status", to)?)
        .bind(from.map(|s| encode("status", s)).transpose()?)
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}

#[async_trait]
impl Store for PgStore {
    async fn upsert_repo(&self, repo: &Repo) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO repos (id, installation_id, full_name) VALUES ($1, $2, $3)
             ON CONFLICT (id) DO UPDATE SET
               installation_id = EXCLUDED.installation_id,
               full_name = EXCLUDED.full_name",
        )
        .bind(repo.id)
        .bind(repo.installation_id)
        .bind(&repo.full_name)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn pr_state(&self, repo_id: i64, pr: i64) -> Result<PrState, StoreError> {
        let row = sqlx::query("SELECT * FROM pull_requests WHERE repo_id = $1 AND number = $2")
            .bind(repo_id)
            .bind(pr)
            .fetch_optional(&self.pool)
            .await?;
        let Some(row) = row else {
            return Ok(PrState::default());
        };
        let reviewed_at: Option<OffsetDateTime> = row.try_get("last_reviewed_at")?;
        Ok(PrState {
            status_comment_id: row.try_get("status_comment_id")?,
            paused: row.try_get("paused")?,
            last_reviewed_sha: row.try_get("last_reviewed_sha")?,
            last_reviewed_at: reviewed_at.map(SystemTime::from),
            approved_sha: row.try_get("approved_sha")?,
            walkthrough: row.try_get("walkthrough")?,
        })
    }

    async fn save_pr_state(
        &self,
        repo_id: i64,
        pr: i64,
        state: &PrState,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO pull_requests
               (repo_id, number, status_comment_id, paused, last_reviewed_sha, approved_sha,
                walkthrough, last_reviewed_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
             ON CONFLICT (repo_id, number) DO UPDATE SET
               status_comment_id = EXCLUDED.status_comment_id,
               paused = EXCLUDED.paused,
               last_reviewed_sha = EXCLUDED.last_reviewed_sha,
               approved_sha = EXCLUDED.approved_sha,
               walkthrough = EXCLUDED.walkthrough,
               last_reviewed_at = EXCLUDED.last_reviewed_at,
               updated_at = now()",
        )
        .bind(repo_id)
        .bind(pr)
        .bind(state.status_comment_id)
        .bind(state.paused)
        .bind(&state.last_reviewed_sha)
        .bind(&state.approved_sha)
        .bind(&state.walkthrough)
        .bind(state.last_reviewed_at.map(OffsetDateTime::from))
        .execute(&self.pool)
        .await
        .map_err(on_repo(repo_id))?;
        Ok(())
    }

    async fn findings(
        &self,
        repo_id: i64,
        pr: i64,
        status: FindingStatus,
    ) -> Result<Vec<StoredFinding>, StoreError> {
        // COLLATE "C" orders bytewise, as MemoryStore does, whatever the server's locale.
        sqlx::query(
            r#"SELECT * FROM findings WHERE repo_id = $1 AND pr = $2 AND status = $3
               ORDER BY path COLLATE "C", start_line, fingerprint COLLATE "C""#,
        )
        .bind(repo_id)
        .bind(pr)
        .bind(encode("status", status)?)
        .fetch_all(&self.pool)
        .await?
        .iter()
        .map(finding)
        .collect()
    }

    async fn findings_before(
        &self,
        repo_id: i64,
        pr: i64,
        status: FindingStatus,
        at: SystemTime,
    ) -> Result<Vec<StoredFinding>, StoreError> {
        sqlx::query(
            r#"SELECT * FROM findings
                WHERE repo_id = $1 AND pr = $2 AND status = $3 AND created_at <= $4
                ORDER BY path COLLATE "C", start_line, fingerprint COLLATE "C""#,
        )
        .bind(repo_id)
        .bind(pr)
        .bind(encode("status", status)?)
        .bind(OffsetDateTime::from(at))
        .fetch_all(&self.pool)
        .await?
        .iter()
        .map(finding)
        .collect()
    }

    async fn upsert_finding(&self, f: &StoredFinding) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO findings
               (repo_id, pr, fingerprint, path, start_line, end_line, severity, category, title,
                comment_id, status, first_sha, last_sha)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
             ON CONFLICT (repo_id, pr, fingerprint) DO UPDATE SET
               path = EXCLUDED.path,
               start_line = EXCLUDED.start_line,
               end_line = EXCLUDED.end_line,
               severity = EXCLUDED.severity,
               category = EXCLUDED.category,
               title = EXCLUDED.title,
               comment_id = COALESCE(EXCLUDED.comment_id, findings.comment_id),
               status = CASE WHEN findings.status = 'suppressed'
                                  AND EXCLUDED.severity <> 'critical' THEN 'suppressed'
                             ELSE EXCLUDED.status END,
               last_sha = EXCLUDED.last_sha,
               updated_at = now()",
        )
        .bind(f.repo_id)
        .bind(f.pr)
        .bind(&f.fingerprint)
        .bind(&f.path)
        .bind(f.start_line)
        .bind(f.end_line)
        .bind(encode("severity", f.severity)?)
        .bind(encode("category", f.category)?)
        .bind(&f.title)
        .bind(f.comment_id)
        .bind(encode("status", f.status)?)
        .bind(&f.first_sha)
        .bind(&f.last_sha)
        .execute(&self.pool)
        .await
        .map_err(on_repo(f.repo_id))?;
        Ok(())
    }

    async fn resolve_findings(
        &self,
        repo_id: i64,
        pr: i64,
        fingerprints: &[String],
    ) -> Result<(), StoreError> {
        self.transition(
            (repo_id, pr),
            fingerprints,
            Some(FindingStatus::Open),
            FindingStatus::Resolved,
        )
        .await
    }

    async fn finding_by_comment(
        &self,
        repo_id: i64,
        pr: i64,
        comment_id: i64,
    ) -> Result<Option<StoredFinding>, StoreError> {
        sqlx::query(
            r#"SELECT * FROM findings WHERE repo_id = $1 AND pr = $2 AND comment_id = $3
               ORDER BY fingerprint COLLATE "C" LIMIT 1"#,
        )
        .bind(repo_id)
        .bind(pr)
        .bind(comment_id)
        .fetch_optional(&self.pool)
        .await?
        .as_ref()
        .map(finding)
        .transpose()
    }

    async fn finding_by_fingerprint(
        &self,
        repo_id: i64,
        pr: i64,
        fingerprint: &str,
    ) -> Result<Option<StoredFinding>, StoreError> {
        sqlx::query("SELECT * FROM findings WHERE repo_id = $1 AND pr = $2 AND fingerprint = $3")
            .bind(repo_id)
            .bind(pr)
            .bind(fingerprint)
            .fetch_optional(&self.pool)
            .await?
            .as_ref()
            .map(finding)
            .transpose()
    }

    async fn append_finding_event(
        &self,
        repo_id: i64,
        pr: i64,
        fingerprint: &str,
        kind: EventKind,
        sha: Option<&str>,
    ) -> Result<(), StoreError> {
        // INSERT ... SELECT: an unknown Fingerprint selects no row, so records nothing.
        sqlx::query(
            "INSERT INTO finding_events (finding_id, kind, sha)
             SELECT id, $4, $5 FROM findings
              WHERE repo_id = $1 AND pr = $2 AND fingerprint = $3",
        )
        .bind(repo_id)
        .bind(pr)
        .bind(fingerprint)
        .bind(encode("kind", kind)?)
        .bind(sha)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn finding_events(&self, repo_id: i64, pr: i64) -> Result<Vec<FindingEvent>, StoreError> {
        sqlx::query(
            "SELECT f.fingerprint, e.kind, e.sha
               FROM finding_events e JOIN findings f ON f.id = e.finding_id
              WHERE f.repo_id = $1 AND f.pr = $2
              ORDER BY e.id",
        )
        .bind(repo_id)
        .bind(pr)
        .fetch_all(&self.pool)
        .await?
        .iter()
        .map(|row| {
            Ok(FindingEvent {
                fingerprint: row.try_get("fingerprint")?,
                kind: decode(row, "kind")?,
                sha: row.try_get("sha")?,
            })
        })
        .collect()
    }

    async fn suppress_finding(
        &self,
        repo_id: i64,
        pr: i64,
        fingerprint: &str,
    ) -> Result<(), StoreError> {
        self.transition(
            (repo_id, pr),
            &[fingerprint.to_owned()],
            None,
            FindingStatus::Suppressed,
        )
        .await
    }

    async fn unsuppress_finding(
        &self,
        repo_id: i64,
        pr: i64,
        fingerprint: &str,
    ) -> Result<(), StoreError> {
        self.transition(
            (repo_id, pr),
            &[fingerprint.to_owned()],
            Some(FindingStatus::Suppressed),
            FindingStatus::Open,
        )
        .await
    }

    async fn add_pusher(&self, repo_id: i64, pr: i64, login: &str) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO pushers (repo_id, pr, login) VALUES ($1, $2, $3)
             ON CONFLICT DO NOTHING",
        )
        .bind(repo_id)
        .bind(pr)
        .bind(login)
        .execute(&self.pool)
        .await
        .map_err(on_repo(repo_id))?;
        Ok(())
    }

    async fn pushers(&self, repo_id: i64, pr: i64) -> Result<Vec<String>, StoreError> {
        Ok(sqlx::query_scalar(
            r#"SELECT login FROM pushers WHERE repo_id = $1 AND pr = $2
                ORDER BY login COLLATE "C""#,
        )
        .bind(repo_id)
        .bind(pr)
        .fetch_all(&self.pool)
        .await?)
    }

    async fn add_lesson(&self, l: &NewLesson) -> Result<i64, StoreError> {
        let id = sqlx::query_scalar(
            "INSERT INTO lessons
               (repo_id, rule, kind, scope, confidence, author, pr, source_comment_id)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
             ON CONFLICT (repo_id, rule) DO UPDATE SET
               kind = CASE WHEN EXCLUDED.kind = 'unknown' THEN lessons.kind
                           ELSE EXCLUDED.kind END,
               confidence = EXCLUDED.confidence,
               status = 'active',
               author = EXCLUDED.author,
               pr = COALESCE(EXCLUDED.pr, lessons.pr),
               source_comment_id = COALESCE(EXCLUDED.source_comment_id, lessons.source_comment_id),
               created_at = now(),
               updated_at = now()
             RETURNING id",
        )
        .bind(l.repo_id)
        .bind(&l.rule)
        .bind(encode("kind", l.kind)?)
        .bind(&l.scope)
        .bind(l.confidence)
        .bind(&l.author)
        .bind(l.pr)
        .bind(l.source_comment_id)
        .fetch_one(&self.pool)
        .await
        .map_err(on_repo(l.repo_id))?;
        Ok(id)
    }

    async fn lessons(
        &self,
        repo_id: i64,
        limit: Option<usize>,
    ) -> Result<Vec<StoredLesson>, StoreError> {
        // LIMIT NULL is no limit. The id breaks ties between Lessons taught in one
        // transaction, which share now().
        sqlx::query(
            "SELECT * FROM lessons WHERE repo_id = $1 AND status = 'active'
              ORDER BY created_at DESC, id DESC LIMIT $2",
        )
        .bind(repo_id)
        .bind(limit.map(|n| i64::try_from(n).unwrap_or(i64::MAX)))
        .fetch_all(&self.pool)
        .await?
        .iter()
        .map(lesson)
        .collect()
    }

    async fn lesson_by_source_comment(
        &self,
        repo_id: i64,
        comment_id: i64,
    ) -> Result<Option<StoredLesson>, StoreError> {
        sqlx::query(
            "SELECT * FROM lessons WHERE repo_id = $1 AND source_comment_id = $2
              ORDER BY id DESC LIMIT 1",
        )
        .bind(repo_id)
        .bind(comment_id)
        .fetch_optional(&self.pool)
        .await?
        .as_ref()
        .map(lesson)
        .transpose()
    }

    async fn mark_lessons_triggered(&self, repo_id: i64, ids: &[i64]) -> Result<(), StoreError> {
        sqlx::query(
            "UPDATE lessons SET trigger_count = trigger_count + 1, last_triggered_at = now()
              WHERE repo_id = $1 AND id = ANY($2)",
        )
        .bind(repo_id)
        .bind(ids)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn delete_lesson(&self, repo_id: i64, id: i64) -> Result<bool, StoreError> {
        let done = sqlx::query("DELETE FROM lessons WHERE repo_id = $1 AND id = $2")
            .bind(repo_id)
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(done.rows_affected() > 0)
    }

    async fn set_decayed_confidence(
        &self,
        repo_id: i64,
        updates: &[(i64, f64)],
    ) -> Result<(), StoreError> {
        // One statement, so the batch is atomic without an explicit transaction.
        let (ids, values): (Vec<i64>, Vec<f64>) = updates.iter().copied().unzip();
        sqlx::query(
            "UPDATE lessons SET decayed_confidence = u.value, updated_at = now()
               FROM UNNEST($2::bigint[], $3::float8[]) AS u (id, value)
              WHERE lessons.repo_id = $1 AND lessons.id = u.id",
        )
        .bind(repo_id)
        .bind(ids)
        .bind(values)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn repos_with_lessons(&self) -> Result<Vec<i64>, StoreError> {
        Ok(
            sqlx::query_scalar("SELECT DISTINCT repo_id FROM lessons ORDER BY repo_id")
                .fetch_all(&self.pool)
                .await?,
        )
    }

    async fn record_usage(&self, u: &Usage) -> Result<(), StoreError> {
        let a = u.anchors;
        sqlx::query(
            "INSERT INTO usage
               (repo_id, pr, head_sha, role, provider, model, input_tokens, output_tokens,
                cache_read_tokens, cache_write_tokens, est_cost_usd, inline, relocated, snapped,
                demoted, verified, committable, failed_files, kept_files, reviewed_files)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17,
                     $18, $19, $20)",
        )
        .bind(u.repo_id)
        .bind(u.pr)
        .bind(&u.head_sha)
        .bind(encode("role", u.role)?)
        .bind(&u.provider)
        .bind(&u.model)
        .bind(u.input_tokens)
        .bind(u.output_tokens)
        .bind(u.cache_read_tokens)
        .bind(u.cache_write_tokens)
        .bind(u.cost_usd)
        .bind(int::<_, i64>(a.inline))
        .bind(int::<_, i64>(a.relocated))
        .bind(int::<_, i64>(a.snapped))
        .bind(int::<_, i64>(a.demoted))
        .bind(int::<_, i64>(a.verified))
        .bind(int::<_, i64>(a.committable))
        .bind(u.failed_files)
        .bind(u.kept_files)
        .bind(u.reviewed_files)
        .execute(&self.pool)
        .await
        .map_err(on_repo(u.repo_id))?;
        Ok(())
    }

    async fn usage(&self, repo_id: i64, pr: i64) -> Result<Vec<Usage>, StoreError> {
        sqlx::query("SELECT * FROM usage WHERE repo_id = $1 AND pr = $2 ORDER BY id")
            .bind(repo_id)
            .bind(pr)
            .fetch_all(&self.pool)
            .await?
            .iter()
            .map(usage)
            .collect()
    }

    async fn enqueue(&self, j: &NewJob) -> Result<bool, StoreError> {
        let done = sqlx::query(
            "INSERT INTO jobs (delivery, slot, payload, run_at) VALUES ($1, $2, $3, now() + $4)
             ON CONFLICT (delivery) DO NOTHING",
        )
        .bind(&j.delivery)
        .bind(&j.slot)
        .bind(&j.payload)
        .bind(j.delay)
        .execute(&self.pool)
        .await?;
        Ok(done.rows_affected() == 1)
    }

    async fn claim(
        &self,
        lease: Duration,
        threads_only: bool,
    ) -> Result<Option<Claimed>, StoreError> {
        let mut tx = self.pool.begin().await?;
        // SKIP LOCKED: a lapsed job another claim is requeueing is left to it, so two
        // claims never wait on each other here.
        sqlx::query(
            "UPDATE jobs SET state = 'queued', lease_until = NULL, updated_at = now()
              WHERE id IN (SELECT id FROM jobs WHERE state = 'running' AND lease_until <= now()
                           FOR UPDATE SKIP LOCKED)",
        )
        .execute(&mut *tx)
        .await?;
        // A concurrent claim can pick another job of the same slot, since neither sees
        // the other's uncommitted row; jobs_running then fails the later one.
        let taken = sqlx::query(
            "UPDATE jobs SET state = 'running', attempts = attempts + 1,
                    lease_until = now() + $2, updated_at = now()
              WHERE id = (SELECT id FROM jobs j
                           WHERE state = 'queued' AND run_at <= now()
                             AND (NOT $3 OR starts_with(slot, $1))
                             AND NOT EXISTS (SELECT 1 FROM jobs r
                                              WHERE r.slot = j.slot AND r.state = 'running')
                           ORDER BY starts_with(slot, $1) DESC, run_at, id
                           LIMIT 1 FOR UPDATE SKIP LOCKED)
              RETURNING id, delivery, attempts, payload",
        )
        .bind(THREAD_SLOT)
        .bind(lease)
        .bind(threads_only)
        .fetch_optional(&mut *tx)
        .await;
        let row = match taken {
            Ok(row) => row,
            // Dropping the transaction rolls it back.
            Err(e)
                if e.as_database_error()
                    .is_some_and(|db| db.is_unique_violation()) =>
            {
                return Ok(None);
            }
            Err(e) => return Err(e.into()),
        };
        tx.commit().await?;
        let Some(row) = row else {
            return Ok(None);
        };
        Ok(Some(Claimed {
            id: row.try_get("id")?,
            delivery: row.try_get("delivery")?,
            attempt: row.try_get("attempts")?,
            payload: row.try_get("payload")?,
        }))
    }

    async fn queued_in(&self, slot: &str) -> Result<Vec<String>, StoreError> {
        Ok(sqlx::query_scalar(
            "SELECT payload FROM jobs WHERE slot = $1 AND state = 'queued' ORDER BY run_at, id",
        )
        .bind(slot)
        .fetch_all(&self.pool)
        .await?)
    }

    async fn finish(&self, id: i64, attempt: i32, end: JobEnd) -> Result<(), StoreError> {
        let (state, after, error) = match end {
            JobEnd::Done => ("done", None, None),
            JobEnd::Retry { after, error } => ("queued", Some(after), Some(error)),
            JobEnd::Dead { error } => ("dead", None, Some(error)),
        };
        sqlx::query(
            "UPDATE jobs SET state = $3, run_at = COALESCE(now() + $4, run_at), error = $5,
                    lease_until = NULL, updated_at = now()
              WHERE id = $1 AND attempts = $2 AND state = 'running'",
        )
        .bind(id)
        .bind(attempt)
        .bind(state)
        .bind(after)
        .bind(error)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn prune_jobs(&self, older_than: Duration) -> Result<u64, StoreError> {
        let done = sqlx::query(
            "DELETE FROM jobs WHERE state IN ('done', 'dead') AND updated_at <= now() - $1",
        )
        .bind(older_than)
        .execute(&self.pool)
        .await?;
        Ok(done.rows_affected())
    }
}

fn finding(row: &PgRow) -> Result<StoredFinding, StoreError> {
    Ok(StoredFinding {
        repo_id: row.try_get("repo_id")?,
        pr: row.try_get("pr")?,
        fingerprint: row.try_get("fingerprint")?,
        path: row.try_get("path")?,
        start_line: row.try_get("start_line")?,
        end_line: row.try_get("end_line")?,
        severity: decode(row, "severity")?,
        category: decode(row, "category")?,
        title: row.try_get("title")?,
        comment_id: row.try_get("comment_id")?,
        status: decode(row, "status")?,
        first_sha: row.try_get("first_sha")?,
        last_sha: row.try_get("last_sha")?,
    })
}

fn lesson(row: &PgRow) -> Result<StoredLesson, StoreError> {
    let time = |column| -> Result<Option<SystemTime>, StoreError> {
        let t: Option<OffsetDateTime> = row.try_get(column)?;
        Ok(t.map(SystemTime::from))
    };
    Ok(StoredLesson {
        lesson: learn::Lesson {
            id: row.try_get("id")?,
            rule: row.try_get("rule")?,
            scope: row.try_get("scope")?,
            created_at: time("created_at")?,
            confidence: row.try_get("confidence")?,
            trigger_count: int(row.try_get::<i64, _>("trigger_count")?),
            contradicted_count: int(row.try_get::<i64, _>("contradicted_count")?),
            last_triggered_at: time("last_triggered_at")?,
        },
        repo_id: row.try_get("repo_id")?,
        kind: decode(row, "kind")?,
        author: row.try_get("author")?,
        pr: row.try_get("pr")?,
        source_comment_id: row.try_get("source_comment_id")?,
        decayed_confidence: row.try_get("decayed_confidence")?,
    })
}

fn usage(row: &PgRow) -> Result<Usage, StoreError> {
    let count = |column| -> Result<usize, StoreError> { Ok(int(row.try_get::<i64, _>(column)?)) };
    Ok(Usage {
        repo_id: row.try_get("repo_id")?,
        pr: row.try_get("pr")?,
        head_sha: row.try_get("head_sha")?,
        role: decode(row, "role")?,
        provider: row.try_get("provider")?,
        model: row.try_get("model")?,
        input_tokens: row.try_get("input_tokens")?,
        output_tokens: row.try_get("output_tokens")?,
        cache_read_tokens: row.try_get("cache_read_tokens")?,
        cache_write_tokens: row.try_get("cache_write_tokens")?,
        cost_usd: row.try_get("est_cost_usd")?,
        anchors: OutcomeStats {
            inline: count("inline")?,
            relocated: count("relocated")?,
            snapped: count("snapped")?,
            demoted: count("demoted")?,
            verified: count("verified")?,
            committable: count("committable")?,
        },
        failed_files: row.try_get("failed_files")?,
        kept_files: row.try_get("kept_files")?,
        reviewed_files: row.try_get("reviewed_files")?,
    })
}

/// An enum's stored form: its serde name, the same one the model's schema uses.
fn encode(
    column: &'static str,
    value: impl Serialize + std::fmt::Debug,
) -> Result<String, StoreError> {
    match serde_json::to_value(&value) {
        Ok(serde_json::Value::String(name)) => Ok(name),
        _ => Err(StoreError::UnknownValue {
            column,
            value: format!("{value:?}"),
        }),
    }
}

/// Reads an enum stored by [`encode`].
fn decode<T: DeserializeOwned>(row: &PgRow, column: &'static str) -> Result<T, StoreError> {
    let name: String = row.try_get(column)?;
    serde_json::from_value(serde_json::Value::String(name.clone())).map_err(|_| {
        StoreError::UnknownValue {
            column,
            value: name,
        }
    })
}

/// Converts between Rust's unsigned counts and Postgres's BIGINT. Counts are never
/// negative nor near `i64::MAX`, so an out-of-range value reads as 0 instead of failing.
fn int<A, B: TryFrom<A> + Default>(n: A) -> B {
    B::try_from(n).unwrap_or_default()
}

/// Maps a foreign-key violation, which on these writes only `repo_id` can cause, to
/// [`StoreError::UnknownRepo`].
fn on_repo(repo_id: i64) -> impl FnOnce(sqlx::Error) -> StoreError {
    move |e| match e.as_database_error() {
        Some(db) if db.is_foreign_key_violation() => StoreError::UnknownRepo(repo_id),
        _ => StoreError::Database(e),
    }
}
