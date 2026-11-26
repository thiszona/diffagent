//! The worker: claims queued jobs and runs each under its kind's timeout.
//!
//! Every pod runs two claim loops: the main lane, `WORKER_CONCURRENCY` jobs of any kind,
//! and the Thread lane, one reply at a time. A job's lease outlasts its timeout, so a job
//! whose pod died is claimed again once the lease lapses, and a job that keeps failing
//! ends Dead with an error line, so no reaper is needed.

pub mod notes;
mod post;
mod review;
mod thread;

use std::future::Future;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, SystemTime};

use tokio::sync::{OwnedSemaphorePermit, Semaphore, watch};
use tokio::time::{Instant, sleep, timeout};
use tracing::{error, info, warn};

use crate::config::Config;
use crate::context::GoCtx;
use crate::github::{Gh, GhError, Github};
use crate::jira::Jira;
use crate::job::{Job, Kind};
use crate::model::{Meter, RoleUsage, Router};
use crate::plan::OutcomeStats;
use crate::state::{Claimed, JobEnd, ModelRole, Store, StoreError, Usage};

/// How long a claimed job is ours: above the longest kind's timeout (8 min) plus its
/// finish, so a live job's lease never lapses.
const LEASE: Duration = Duration::from_mins(9);

/// The longest a shutdown waits for running jobs; the pod's grace is 600 s. A job still
/// running when the pod dies is claimed again when its lease lapses.
const DRAIN: Duration = Duration::from_mins(9);

/// How often an idle loop asks for work.
const IDLE_POLL: Duration = Duration::from_secs(1);

/// Attempts per job. A third failure, or a fourth claim (each lease lapsed), is Dead.
const MAX_ATTEMPTS: i32 = 3;

/// A failed attempt runs again this long × its attempt number later.
const RETRY_STEP: Duration = Duration::from_secs(30);

/// Done and dead rows are kept this long: GitHub's manual redelivery reaches 3 days back.
const KEEP_JOBS: Duration = Duration::from_hours(3 * 24);

const PRUNE_EVERY: Duration = Duration::from_hours(1);

/// What every job shares, built once at startup.
pub struct Deps {
    pub store: Arc<dyn Store>,
    pub github: Github,
    pub router: Router,
    pub goctx: GoCtx,
    pub jira: Option<Jira>,
    pub cfg: Config,
    /// `<slug>[bot]`: only comments by this login are ours.
    pub bot_login: String,
    /// Where checkouts fetch from: `workspace::GITHUB`, or a `file://` URL in tests.
    pub remote: String,
}

impl Deps {
    /// The job's installation client. A 401 or 403 minting its token (a bad App key, a
    /// suspended installation) ends the job Dead. The same status on any other call
    /// stays a GitHub error, retried: a secondary rate limit can be a 403 with no header.
    pub async fn installation(&self, id: i64) -> Result<Gh, JobError> {
        self.github.installation(id).await.map_err(|e| match &e {
            GhError::Http(text) if matches!(status(text), Some(401 | 403)) => {
                JobError::Fatal(e.to_string())
            }
            _ => JobError::Github(e),
        })
    }
}

/// What an attempt reports for its usage rows. It lives outside the attempt's timed
/// future, so a failed or timed-out attempt still records what its model calls cost,
/// and the worker writes every row, once.
#[derive(Debug, Default)]
pub struct Report {
    /// The head SHA, set once read: every usage row names it.
    pub head: OnceLock<String>,
    /// A Review's counts, set once known.
    pub counts: OnceLock<Counts>,
    /// Set once a Review found a Finding that blocks the merge gate, before storing it.
    pub blocked: OnceLock<()>,
}

/// A Review's anchors and file counts, for the review role's usage row.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counts {
    pub anchors: OutcomeStats,
    pub failed: i64,
    pub kept: i64,
    pub reviewed: i64,
}

/// Why an attempt failed, which decides whether it runs again.
#[derive(Debug, thiserror::Error)]
pub enum JobError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error(transparent)]
    Github(#[from] GhError),
    /// Ran past its kind's limit, so its future was dropped.
    #[error("timed out after {0:?}")]
    Timeout(Duration),
    /// Running it again cannot help.
    #[error("{0}")]
    Fatal(String),
}

/// Claims and runs jobs until `stop` turns true, then waits up to 9 min for the running
/// ones.
pub async fn run(deps: Arc<Deps>, stop: watch::Receiver<bool>) {
    serve(deps, stop, dispatch).await;
}

/// Runs one job.
async fn dispatch(
    deps: Arc<Deps>,
    job: Job,
    meter: Arc<Meter>,
    report: Arc<Report>,
) -> Result<(), JobError> {
    match job.kind {
        Kind::Review { .. } => review::review(&deps, &job, &meter, &report, review::GITLEAKS).await,
        Kind::Pause => review::pause(&deps, &job).await,
        Kind::Resolve => review::resolve(&deps, &job).await,
        Kind::Lessons => thread::lessons(&deps, &job).await,
        Kind::Forget { .. } => thread::forget(&deps, &job).await,
        Kind::Reply { .. } => thread::reply(&deps, &job, &meter, &report, review::GITLEAKS).await,
        Kind::Learn { .. } | Kind::Close { .. } => thread::command(&deps, &job).await,
    }
}

/// Each kind's limit.
fn kind_limit(kind: &Kind) -> Duration {
    match kind {
        Kind::Review { .. } => Duration::from_secs(8 * 60),
        Kind::Reply { .. } => Duration::from_secs(3 * 60),
        Kind::Pause
        | Kind::Resolve
        | Kind::Lessons
        | Kind::Forget { .. }
        | Kind::Learn { .. }
        | Kind::Close { .. } => Duration::from_secs(2 * 60),
    }
}

/// The two claim loops, with the job runner as a parameter so tests can run their own.
/// The Thread lane has its own permit: a reply is a person waiting, and with shared
/// permits two 8-minute Reviews would hold every answer (or one, at
/// `WORKER_CONCURRENCY=1`). The main lane takes Thread jobs too, first, when it is free.
async fn serve<H, F>(deps: Arc<Deps>, stop: watch::Receiver<bool>, handle: H)
where
    H: Fn(Arc<Deps>, Job, Arc<Meter>, Arc<Report>) -> F + Clone + Send + 'static,
    F: Future<Output = Result<(), JobError>> + Send + 'static,
{
    let size = u32::try_from(deps.cfg.worker_concurrency)
        .unwrap_or(u32::MAX)
        .max(1);
    tokio::join!(
        lane(Arc::clone(&deps), stop.clone(), handle.clone(), size, false),
        lane(deps, stop, handle, 1, true),
    );
}

/// One claim loop with `size` permits; `threads_only` claims only Thread slots. The main
/// lane also prunes old jobs. On a stop it claims nothing more and waits up to 9 min for
/// its running jobs.
async fn lane<H, F>(
    deps: Arc<Deps>,
    mut stop: watch::Receiver<bool>,
    handle: H,
    size: u32,
    threads_only: bool,
) where
    H: Fn(Arc<Deps>, Job, Arc<Meter>, Arc<Report>) -> F + Clone + Send + 'static,
    F: Future<Output = Result<(), JobError>> + Send + 'static,
{
    let permits = Arc::new(Semaphore::new(size as usize));
    let mut pruned: Option<Instant> = None;
    loop {
        let permit = tokio::select! {
            biased;
            () = stopped(&mut stop) => break,
            p = Arc::clone(&permits).acquire_owned() => match p {
                Ok(p) => p,
                Err(_) => break,
            },
        };
        if !threads_only && pruned.is_none_or(|t| t.elapsed() >= PRUNE_EVERY) {
            prune(&*deps.store).await;
            pruned = Some(Instant::now());
        }
        match deps.store.claim(LEASE, threads_only).await {
            Ok(Some(c)) => {
                tokio::spawn(attempt(Arc::clone(&deps), c, handle.clone(), permit));
            }
            Ok(None) => idle(&mut stop).await,
            Err(e) => {
                warn!(err = %e, "claiming a job failed");
                idle(&mut stop).await;
            }
        }
    }
    if timeout(DRAIN, permits.acquire_many(size)).await.is_err() {
        warn!("jobs still running at shutdown; each runs again when its lease lapses");
    }
}

/// Resolves once `stop` is true, or its sender is gone.
async fn stopped(stop: &mut watch::Receiver<bool>) {
    let _ = stop.wait_for(|&s| s).await;
}

/// Waits one poll, or less when a stop comes.
async fn idle(stop: &mut watch::Receiver<bool>) {
    tokio::select! {
        () = sleep(IDLE_POLL) => {}
        () = stopped(stop) => {}
    }
}

async fn prune(store: &dyn Store) {
    match store.prune_jobs(KEEP_JOBS).await {
        Ok(0) => {}
        Ok(n) => info!(pruned = n, "old jobs pruned"),
        Err(e) => warn!(err = %e, "pruning jobs failed"),
    }
}

/// Runs one claimed job and records how it ended. The permit is held until then.
async fn attempt<H, F>(deps: Arc<Deps>, c: Claimed, handle: H, _permit: OwnedSemaphorePermit)
where
    H: Fn(Arc<Deps>, Job, Arc<Meter>, Arc<Report>) -> F,
    F: Future<Output = Result<(), JobError>>,
{
    let end = match serde_json::from_str::<Job>(&c.payload) {
        Ok(job) => run_job(&deps, &c, job, handle).await,
        Err(e) => {
            let error = format!("undecodable payload: {e}");
            error!(job = c.id, delivery = %c.delivery, attempts = c.attempt, err = %error, "job dead");
            JobEnd::Dead { error }
        }
    };
    if let Err(e) = deps.store.finish(c.id, c.attempt, end).await {
        warn!(job = c.id, attempt = c.attempt, err = %e, "finishing a job failed; it runs again when its lease lapses");
    }
}

/// One attempt under its kind's limit. The meter and the head live outside the timed
/// future, so a failed or timed-out attempt still records what its model calls cost.
async fn run_job<H, F>(deps: &Arc<Deps>, c: &Claimed, job: Job, handle: H) -> JobEnd
where
    H: Fn(Arc<Deps>, Job, Arc<Meter>, Arc<Report>) -> F,
    F: Future<Output = Result<(), JobError>>,
{
    let repo = format!("{}/{}", job.owner, job.repo);
    if c.attempt > MAX_ATTEMPTS {
        let error = format!("no attempt of {MAX_ATTEMPTS} finished: each lease lapsed");
        error!(job = c.id, repo = %repo, pr = job.pr, attempts = c.attempt - 1, err = %error, "job dead");
        review::dead(deps, &job, &Report::default()).await;
        return JobEnd::Dead { error };
    }
    let meter = Arc::new(Meter::default());
    let report = Arc::new(Report::default());
    let limit = kind_limit(&job.kind);
    let run = handle(
        Arc::clone(deps),
        job.clone(),
        Arc::clone(&meter),
        Arc::clone(&report),
    );
    let result = timeout(limit, run)
        .await
        .unwrap_or(Err(JobError::Timeout(limit)));
    record_usage(deps, &job, &repo, &meter, &report).await;
    let end = ending(&result, c.attempt, SystemTime::now());
    match &end {
        JobEnd::Done => {
            info!(job = c.id, repo = %repo, pr = job.pr, attempt = c.attempt, "job done")
        }
        JobEnd::Retry { error, .. } => {
            warn!(job = c.id, repo = %repo, pr = job.pr, attempt = c.attempt, err = %error, "job failed; it runs again");
        }
        JobEnd::Dead { error } => {
            error!(job = c.id, repo = %repo, pr = job.pr, attempts = c.attempt, err = %error, "job dead");
            review::dead(deps, &job, &report).await;
        }
    }
    end
}

/// How an attempt ends. What retrying cannot fix (the job's own `Fatal`, which includes
/// a 401 or 403 on the installation; a 404; a 422) and a third failure are Dead.
/// Anything else (the store, a 5xx, any other 401 or 403, a timeout, an unconfirmed
/// create) runs again 30 s × attempt later, or when a rate limit resets: a delay the
/// store adds to its own clock. `now` is only for the reset, an absolute time.
fn ending(result: &Result<(), JobError>, attempt: i32, now: SystemTime) -> JobEnd {
    let Err(e) = result else {
        return JobEnd::Done;
    };
    let error = e.to_string();
    let hopeless = matches!(
        e,
        JobError::Fatal(_) | JobError::Github(GhError::NotFound | GhError::Unprocessable(_))
    );
    if hopeless || attempt >= MAX_ATTEMPTS {
        return JobEnd::Dead { error };
    }
    let after = match e {
        JobError::Github(GhError::RateLimited(reset)) => {
            reset.duration_since(now).unwrap_or_default()
        }
        _ => RETRY_STEP * attempt.unsigned_abs(),
    };
    JobEnd::Retry { after, error }
}

/// The status a GitHub `Http` error starts with (`"401: …"`); none for a transport error.
fn status(text: &str) -> Option<u16> {
    text.split_once(':')?.0.parse().ok()
}

/// Writes the attempt's usage rows, whether it finished, failed or timed out: one per
/// role it called, with the Review's counts when it reported them.
async fn record_usage(deps: &Deps, job: &Job, repo: &str, meter: &Meter, report: &Report) {
    let used = deps.router.usage(meter);
    if used.is_empty() {
        return;
    }
    let Some(head) = report.head.get() else {
        warn!(
            repo,
            pr = job.pr,
            "model calls before the head was read; usage not recorded"
        );
        return;
    };
    for u in usage_rows(&used, job, head, report.counts.get()) {
        if let Err(e) = deps.store.record_usage(&u).await {
            warn!(repo, pr = job.pr, err = %e, "recording usage failed");
        }
    }
}

/// One usage row per role called. The counts describe the Review, so they go on the
/// review role's row only, and summing rows never double counts them.
fn usage_rows(used: &[RoleUsage], job: &Job, head: &str, counts: Option<&Counts>) -> Vec<Usage> {
    used.iter()
        .map(|u| {
            let c = counts
                .filter(|_| u.role == ModelRole::Review)
                .copied()
                .unwrap_or_default();
            Usage {
                repo_id: job.repo_id,
                pr: job.pr,
                head_sha: head.to_owned(),
                role: u.role,
                provider: u.provider.clone(),
                model: u.model.clone(),
                input_tokens: u.tally.tokens.input,
                output_tokens: u.tally.tokens.output,
                cache_read_tokens: u.tally.tokens.cache_read,
                cache_write_tokens: u.tally.tokens.cache_write,
                cost_usd: u.cost_usd,
                anchors: c.anchors,
                failed_files: c.failed,
                kept_files: c.kept,
                reviewed_files: c.reviewed,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tokio::sync::{Notify, mpsc};

    use super::*;
    use crate::github::fake;
    use crate::model::tests::{Script, route};
    use crate::model::{Model, Request, Stop};
    use crate::state::{MemoryStore, NewJob, Repo};

    type Outcome = Result<(), JobError>;

    /// No GitHub listens here: a job that calls it fails to connect.
    const NOWHERE: &str = "http://127.0.0.1:9";

    fn deps(concurrency: usize, api: &str) -> Arc<Deps> {
        let n = concurrency.to_string();
        let cfg = Config::from_lookup(|k| (k == "WORKER_CONCURRENCY").then(|| n.clone())).unwrap();
        let model: Arc<dyn Model> = Script::new(&[("fine", Stop::End)]);
        Arc::new(Deps {
            store: Arc::new(MemoryStore::default()),
            github: Github::new(api, fake::APP_ID, &fake::app_key()).unwrap(),
            router: Router::new(&cfg, vec![(ModelRole::Review, route("m"), model)]),
            goctx: GoCtx::new("diffagent-goctx", 800, Vec::new(), 1),
            jira: None,
            cfg,
            bot_login: "diffagent[bot]".to_owned(),
            remote: "file:///nowhere".to_owned(),
        })
    }

    fn job(pr: i64, kind: Kind) -> Job {
        Job {
            installation_id: 1,
            repo_id: 7,
            owner: "acme".to_owned(),
            repo: "api".to_owned(),
            pr,
            actor: "dev".to_owned(),
            kind,
            commented_at: None,
            pushed: false,
        }
    }

    async fn enqueue(store: &dyn Store, delivery: &str, j: &Job) {
        let new = NewJob {
            delivery: delivery.to_owned(),
            slot: j.slot(),
            payload: serde_json::to_string(j).unwrap(),
            delay: Duration::ZERO,
        };
        assert!(store.enqueue(&new).await.unwrap());
    }

    #[tokio::test]
    async fn a_job_runs_and_is_done() {
        let d = deps(2, NOWHERE);
        let j = job(1, Kind::Pause);
        enqueue(&*d.store, "d1", &j).await;
        let (tx, mut rx) = mpsc::unbounded_channel();
        let (stop, stopped) = watch::channel(false);
        let worker = tokio::spawn(serve(Arc::clone(&d), stopped, move |_, job, _, _| {
            let tx = tx.clone();
            async move {
                let _ = tx.send(job);
                Outcome::Ok(())
            }
        }));
        assert_eq!(rx.recv().await, Some(j));
        stop.send_replace(true);
        worker.await.unwrap();
        assert_eq!(
            d.store.claim(LEASE, false).await.unwrap(),
            None,
            "never runs again"
        );
        assert_eq!(
            d.store.prune_jobs(Duration::ZERO).await.unwrap(),
            1,
            "ended"
        );
    }

    /// A finished attempt's usage is written by the worker too, with the Review's counts,
    /// so every row has one writer; a timed-out one runs again and keeps its usage.
    #[tokio::test(start_paused = true)]
    async fn an_attempt_records_its_usage_finished_or_timed_out() {
        let counts = Counts {
            failed: 1,
            kept: 9,
            reviewed: 4,
            ..Counts::default()
        };
        for hang in [true, false] {
            let d = deps(1, NOWHERE);
            let repo = Repo {
                id: 7,
                installation_id: 1,
                full_name: "acme/api".to_owned(),
            };
            d.store.upsert_repo(&repo).await.unwrap();
            enqueue(&*d.store, "d1", &job(1, Kind::Review { command: false })).await;
            let (tx, mut rx) = mpsc::unbounded_channel();
            let (stop, stopped) = watch::channel(false);
            let worker = tokio::spawn(serve(
                Arc::clone(&d),
                stopped,
                move |deps, _, meter, report| {
                    let tx = tx.clone();
                    async move {
                        report.head.set("abc123".to_owned()).unwrap();
                        let req = Request {
                            system: "s",
                            user: "u",
                            shared: 0,
                            schema: None,
                            max_tokens: 10,
                        };
                        deps.router
                            .text(ModelRole::Review, &req, &meter)
                            .await
                            .unwrap();
                        let _ = tx.send(());
                        if hang {
                            std::future::pending::<()>().await;
                        } else {
                            report.counts.set(counts).unwrap();
                        }
                        Outcome::Ok(())
                    }
                },
            ));
            rx.recv().await.unwrap();
            stop.send_replace(true);
            // A hung Review's drain waits out its 8 min limit (on the paused clock).
            worker.await.unwrap();
            let rows = d.store.usage(7, 1).await.unwrap();
            assert_eq!(rows.len(), 1, "hang {hang}: {rows:?}");
            let r = &rows[0];
            assert_eq!((r.role, r.head_sha.as_str()), (ModelRole::Review, "abc123"));
            assert_eq!((r.input_tokens, r.output_tokens), (1000, 100));
            if hang {
                let claimed = d.store.claim(LEASE, false).await.unwrap();
                assert_eq!(claimed, None, "due in 30 s");
                let pruned = d.store.prune_jobs(Duration::ZERO).await.unwrap();
                assert_eq!(pruned, 0, "queued");
            } else {
                assert_eq!((r.failed_files, r.kept_files, r.reviewed_files), (1, 9, 4));
            }
        }
    }

    #[tokio::test(start_paused = true)]
    async fn dead_jobs_never_run() {
        let d = deps(1, NOWHERE);
        // Three attempts whose leases lapsed: the fourth claim is past the limit.
        enqueue(&*d.store, "lapsed", &job(1, Kind::Pause)).await;
        for attempt in 1..=MAX_ATTEMPTS {
            let c = d.store.claim(Duration::ZERO, false).await.unwrap().unwrap();
            assert_eq!(c.attempt, attempt);
        }
        let junk = NewJob {
            delivery: "junk".to_owned(),
            slot: "pr:7:2".to_owned(),
            payload: "{".to_owned(),
            delay: Duration::ZERO,
        };
        assert!(d.store.enqueue(&junk).await.unwrap());
        let ran = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&ran);
        let (stop, stopped) = watch::channel(false);
        let worker = tokio::spawn(serve(Arc::clone(&d), stopped, move |_, _, _, _| {
            counted.fetch_add(1, Ordering::SeqCst);
            async { Outcome::Ok(()) }
        }));
        sleep(Duration::from_secs(5)).await;
        stop.send_replace(true);
        worker.await.unwrap();
        assert_eq!(ran.load(Ordering::SeqCst), 0);
        assert_eq!(
            d.store.prune_jobs(Duration::ZERO).await.unwrap(),
            2,
            "both ended"
        );
    }

    #[tokio::test]
    async fn a_stop_claims_nothing_more_and_waits_for_running_jobs() {
        let d = deps(1, NOWHERE);
        enqueue(&*d.store, "first", &job(1, Kind::Pause)).await;
        let (tx, mut rx) = mpsc::unbounded_channel();
        let release = Arc::new(Notify::new());
        let held = Arc::clone(&release);
        let (stop, stopped) = watch::channel(false);
        let worker = tokio::spawn(serve(Arc::clone(&d), stopped, move |_, job, _, _| {
            let (tx, held) = (tx.clone(), Arc::clone(&held));
            async move {
                let _ = tx.send(job.pr);
                held.notified().await;
                Outcome::Ok(())
            }
        }));
        assert_eq!(rx.recv().await, Some(1));
        enqueue(&*d.store, "second", &job(2, Kind::Pause)).await;
        stop.send_replace(true);
        release.notify_one();
        worker.await.unwrap();
        assert!(rx.try_recv().is_err(), "the second job never ran");
        let second = d.store.claim(LEASE, false).await.unwrap().unwrap();
        assert_eq!((second.delivery.as_str(), second.attempt), ("second", 1));
        assert_eq!(
            d.store.prune_jobs(Duration::ZERO).await.unwrap(),
            1,
            "first done"
        );
    }

    /// A reply runs while a Review holds the only main permit: the Thread lane has its own.
    #[tokio::test]
    async fn a_thread_job_runs_while_a_review_holds_the_only_permit() {
        let d = deps(1, NOWHERE);
        enqueue(&*d.store, "push", &job(1, Kind::Review { command: false })).await;
        let (tx, mut rx) = mpsc::unbounded_channel();
        let release = Arc::new(Notify::new());
        let held = Arc::clone(&release);
        let (stop, stopped) = watch::channel(false);
        let worker = tokio::spawn(serve(Arc::clone(&d), stopped, move |_, job, _, _| {
            let (tx, held) = (tx.clone(), Arc::clone(&held));
            async move {
                let review = matches!(job.kind, Kind::Review { .. });
                let _ = tx.send(job);
                if review {
                    held.notified().await;
                }
                Outcome::Ok(())
            }
        }));
        assert!(matches!(
            rx.recv().await,
            Some(Job {
                kind: Kind::Review { .. },
                ..
            })
        ));
        let reply = job(
            1,
            Kind::Reply {
                root: 99,
                reply: 100,
                body: "why?".to_owned(),
                head_sha: "abc".to_owned(),
            },
        );
        assert_eq!(reply.slot(), "thread:7:99");
        enqueue(&*d.store, "reply", &reply).await;
        let answered = timeout(Duration::from_secs(5), rx.recv()).await;
        assert_eq!(answered, Ok(Some(reply)), "answered while the Review runs");
        release.notify_one();
        stop.send_replace(true);
        worker.await.unwrap();
        assert_eq!(
            d.store.prune_jobs(Duration::ZERO).await.unwrap(),
            2,
            "both done"
        );
    }

    /// A 401 or 403 on the installation token ends the job Dead, and no other 401 or
    /// 403 does.
    #[tokio::test]
    async fn only_the_installation_token_ends_a_job_on_401_or_403() {
        let gh = fake::Fake::start().await;
        let d = deps(1, &gh.base);
        let now = SystemTime::UNIX_EPOCH;
        let mint = "POST /app/installations/*/access_tokens";
        for status in [401, 403] {
            gh.world().fault(mint, 1, fake::FaultKind::Before(status));
            let got = d.installation(1).await.map(|_| ());
            assert!(
                matches!(ending(&got, 1, now), JobEnd::Dead { .. }),
                "{status}: {got:?}"
            );
        }
        // A secondary rate limit may come as a bare 403 on any call: it runs again.
        let client = d.installation(1).await.unwrap();
        gh.world().pulls.insert(1, serde_json::json!({"number": 1}));
        let pr = "/repos/o/r/pulls/1";
        gh.world()
            .fault(&format!("GET {pr}"), 1, fake::FaultKind::Before(403));
        let got = client
            .get::<serde_json::Value>(pr)
            .await
            .map(|_| ())
            .map_err(JobError::from);
        assert!(
            matches!(ending(&got, 1, now), JobEnd::Retry { .. }),
            "{got:?}"
        );
    }

    #[test]
    fn endings() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000);
        let reset = now + Duration::from_secs(3600);
        let secs = Duration::from_secs;
        let gh = |e: GhError| Err(JobError::Github(e));
        let http = |s: &str| gh(GhError::Http(s.to_owned()));
        // (result, attempt, Some(retry after) or None for Dead; Done is the first row).
        let cases: Vec<(Outcome, i32, Option<Duration>)> = vec![
            (
                Err(JobError::Store(StoreError::UnknownRepo(7))),
                1,
                Some(secs(30)),
            ),
            (http("502: bad gateway"), 2, Some(secs(60))),
            (http("error sending request"), 1, Some(secs(30))),
            (
                Err(JobError::Timeout(Duration::from_secs(480))),
                1,
                Some(secs(30)),
            ),
            (gh(GhError::Ambiguous), 1, Some(secs(30))),
            (gh(GhError::RateLimited(reset)), 2, Some(secs(3600))),
            (gh(GhError::RateLimited(now)), 2, Some(secs(0))),
            (http("502: bad gateway"), MAX_ATTEMPTS, None),
            (gh(GhError::NotFound), 1, None),
            (gh(GhError::Unprocessable("line".to_owned())), 1, None),
            (http("401: bad credentials"), 1, Some(secs(30))),
            (
                http("403: You have exceeded a secondary rate limit"),
                1,
                Some(secs(30)),
            ),
            (Err(JobError::Fatal("not built yet".to_owned())), 1, None),
        ];
        assert_eq!(ending(&Ok(()), 1, now), JobEnd::Done);
        for (result, attempt, want) in &cases {
            let got = ending(result, *attempt, now);
            let error = result.as_ref().unwrap_err().to_string();
            let want = match want {
                Some(after) => JobEnd::Retry {
                    after: *after,
                    error,
                },
                None => JobEnd::Dead { error },
            };
            assert_eq!(got, want, "{result:?} on attempt {attempt}");
        }
    }

    #[test]
    fn anchors_and_counts_go_on_the_review_row() {
        let tally = crate::model::Tally::default();
        let used = [ModelRole::Review, ModelRole::Summary].map(|role| RoleUsage {
            role,
            provider: "bedrock".to_owned(),
            model: "m".to_owned(),
            tally,
            cost_usd: 0.5,
        });
        let counts = Counts {
            anchors: OutcomeStats {
                inline: 2,
                ..OutcomeStats::default()
            },
            failed: 1,
            kept: 9,
            reviewed: 4,
        };
        let rows = usage_rows(&used, &job(3, Kind::Pause), "h", Some(&counts));
        let shape: Vec<_> = rows
            .iter()
            .map(|u| {
                (
                    u.role,
                    u.anchors.inline,
                    u.failed_files,
                    u.kept_files,
                    u.reviewed_files,
                )
            })
            .collect();
        assert_eq!(
            shape,
            [
                (ModelRole::Review, 2, 1, 9, 4),
                (ModelRole::Summary, 0, 0, 0, 0)
            ]
        );
        assert!(
            rows.iter()
                .all(|u| u.repo_id == 7 && u.pr == 3 && u.head_sha == "h")
        );
    }
}
