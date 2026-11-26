//! The Store contract: every case runs against [`MemoryStore`] and against [`PgStore`] on
//! a real Postgres, so dev mode and tests cannot drift from production. The cases include
//! the schema's rename-safe repos, the suppressed status, and the repo foreign key.
//!
//! The Postgres is embedded (`postgresql_embedded`), so `cargo test` needs no Docker. Its
//! binaries download once into `~/.theseus`, after one GitHub API call (CI passes
//! `GITHUB_TOKEN` against rate limits). Each test gets its own database on a shared server.

use std::sync::{Arc, Mutex};

use super::*;
use crate::finding::{Category, Severity};

/// Runs each case as `memory_store::<case>` and `pg_store::<case>`.
macro_rules! contract {
    ($($case:ident),+ $(,)?) => {
        mod memory_store {
            $(
                #[tokio::test]
                async fn $case() {
                    super::$case(&super::MemoryStore::default()).await;
                }
            )+
        }

        mod pg_store {
            $(
                #[tokio::test]
                async fn $case() {
                    let db = super::embedded::Db::new(stringify!($case)).await;
                    super::$case(&db.store).await;
                }
            )+
        }
    };
}

contract!(
    repo_rename_keeps_history,
    writes_need_a_recorded_repo,
    upsert_open_resolve,
    findings_before,
    findings_in_location_order,
    reaffirm_keeps_first_sha_and_comment,
    reaffirm_keeps_suppressed,
    suppressed_status,
    unsuppress_finding,
    resolved_findings_accumulate,
    finding_by_fingerprint,
    finding_by_comment,
    finding_events,
    pr_state_round_trip,
    pushers_per_pr,
    same_rule_is_one_lesson,
    reteach_revives,
    lesson_round_trip,
    lessons_newest_first,
    lesson_by_source_comment,
    mark_lessons_triggered,
    delete_lesson,
    set_decayed_confidence,
    repos_with_lessons,
    usage_round_trip,
    jobs_dedupe_and_delay,
    jobs_one_running_per_slot,
    jobs_thread_lane,
    jobs_lapsed_lease_reclaimed,
    jobs_stale_finish,
    jobs_retry,
    jobs_prune,
    jobs_queued_in_a_slot,
);

/// One shared Postgres server, and a fresh database per test.
pub(crate) mod embedded {
    use std::sync::{Arc, Weak};

    use postgresql_embedded::{PostgreSQL, Settings, VersionReq};
    use tokio::sync::Mutex;

    use super::{LEASE, PgStore, Store, job};

    /// An exact release: installing then fetches one release by tag instead of listing
    /// them all, and a cached install needs no network at all.
    const VERSION: &str = "=16.15.0";

    /// The running server, while any test holds it. A static server would never be
    /// dropped, and `pg_ctl` detaches it from this process, so it would outlive
    /// `cargo test`. Instead the last test holding it stops it on drop, and a test that
    /// starts later boots a fresh one.
    static SERVER: Mutex<Weak<PostgreSQL>> = Mutex::const_new(Weak::new());

    async fn server() -> Arc<PostgreSQL> {
        let mut slot = SERVER.lock().await;
        if let Some(server) = slot.upgrade() {
            return server;
        }
        let settings = Settings {
            version: VersionReq::parse(VERSION).expect("valid version"),
            ..Settings::default()
        };
        let mut pg = PostgreSQL::new(settings);
        pg.setup().await.expect("install postgres");
        pg.start().await.expect("start postgres");
        let pg = Arc::new(pg);
        *slot = Arc::downgrade(&pg);
        pg
    }

    pub struct Db {
        // Declared first so the pool closes before the server can stop.
        pub store: PgStore,
        _server: Arc<PostgreSQL>,
    }

    impl Db {
        pub async fn new(name: &str) -> Self {
            let server = server().await;
            server.create_database(name).await.expect("create database");
            let store = PgStore::connect(&server.settings().url(name))
                .await
                .expect("connect");
            Self {
                store,
                _server: server,
            }
        }
    }

    /// Every pod start connects to an already-migrated database.
    #[tokio::test]
    async fn reconnect_applies_nothing_twice() {
        let db = Db::new("reconnect_applies_nothing_twice").await;
        let url = db._server.settings().url("reconnect_applies_nothing_twice");
        PgStore::connect(&url).await.expect("reconnect");
    }

    /// A rollback: an older image connects to a database a newer release migrated.
    #[tokio::test]
    async fn older_image_starts_on_newer_schema() {
        let db = Db::new("older_image_starts_on_newer_schema").await;
        let url = db
            ._server
            .settings()
            .url("older_image_starts_on_newer_schema");
        let pool = sqlx::PgPool::connect(&url).await.expect("pool");
        sqlx::query(
            "INSERT INTO _sqlx_migrations (version, description, success, checksum, execution_time)
             VALUES (9999, 'from a newer release', true, '\\x00', 0)",
        )
        .execute(&pool)
        .await
        .expect("record a newer migration");
        PgStore::connect(&url).await.expect("reconnect");
    }

    /// Two pods claiming at once: one job per slot runs, whoever wins the race, and
    /// jobs in different slots both run.
    #[tokio::test]
    async fn concurrent_claims() {
        let db = Db::new("concurrent_claims").await;
        let s = &db.store;
        for round in 0..10 {
            let slot = format!("pr:1:{round}");
            for d in ["a", "b"] {
                s.enqueue(&job(&format!("{round}{d}"), &slot))
                    .await
                    .unwrap();
            }
            let (x, y) = tokio::join!(s.claim(LEASE, false), s.claim(LEASE, false));
            let won = [x.unwrap(), y.unwrap()].into_iter().flatten().count();
            assert_eq!(won, 1, "round {round}");
        }

        for d in ["x", "y"] {
            s.enqueue(&job(d, &format!("pr:2:{d}"))).await.unwrap();
        }
        let (x, y) = tokio::join!(s.claim(LEASE, false), s.claim(LEASE, false));
        let (x, y) = (x.unwrap().expect("x"), y.unwrap().expect("y"));
        assert_ne!(x.id, y.id);
    }
}

/// A store that fails a method on cue: `skip` calls pass, then `times` fail.
pub(crate) struct Faulty {
    pub(crate) inner: Arc<dyn Store>,
    pub(crate) faults: Mutex<Vec<(&'static str, usize, usize)>>,
}

impl Faulty {
    fn check(&self, method: &str) -> Result<(), StoreError> {
        let mut faults = self.faults.lock().unwrap();
        let Some(f) = faults.iter_mut().find(|f| f.0 == method && f.2 > 0) else {
            return Ok(());
        };
        if f.1 > 0 {
            f.1 -= 1;
            return Ok(());
        }
        f.2 -= 1;
        Err(StoreError::Database(sqlx::Error::PoolTimedOut))
    }
}

macro_rules! faulty {
    ($($name:ident($($arg:ident: $ty:ty),*) -> $ret:ty;)+) => {
        #[async_trait]
        impl Store for Faulty {
            $(async fn $name(&self, $($arg: $ty),*) -> Result<$ret, StoreError> {
                self.check(stringify!($name))?;
                self.inner.$name($($arg),*).await
            })+
        }
    };
}

faulty! {
    upsert_repo(repo: &Repo) -> ();
    pr_state(repo_id: i64, pr: i64) -> PrState;
    save_pr_state(repo_id: i64, pr: i64, state: &PrState) -> ();
    findings(repo_id: i64, pr: i64, status: FindingStatus) -> Vec<StoredFinding>;
    findings_before(repo_id: i64, pr: i64, status: FindingStatus, at: SystemTime)
        -> Vec<StoredFinding>;
    upsert_finding(f: &StoredFinding) -> ();
    resolve_findings(repo_id: i64, pr: i64, fingerprints: &[String]) -> ();
    finding_by_comment(repo_id: i64, pr: i64, comment_id: i64) -> Option<StoredFinding>;
    finding_by_fingerprint(repo_id: i64, pr: i64, fingerprint: &str) -> Option<StoredFinding>;
    append_finding_event(repo_id: i64, pr: i64, fingerprint: &str, kind: EventKind,
        sha: Option<&str>) -> ();
    finding_events(repo_id: i64, pr: i64) -> Vec<FindingEvent>;
    suppress_finding(repo_id: i64, pr: i64, fingerprint: &str) -> ();
    unsuppress_finding(repo_id: i64, pr: i64, fingerprint: &str) -> ();
    add_pusher(repo_id: i64, pr: i64, login: &str) -> ();
    pushers(repo_id: i64, pr: i64) -> Vec<String>;
    add_lesson(l: &NewLesson) -> i64;
    lessons(repo_id: i64, limit: Option<usize>) -> Vec<StoredLesson>;
    lesson_by_source_comment(repo_id: i64, comment_id: i64) -> Option<StoredLesson>;
    mark_lessons_triggered(repo_id: i64, ids: &[i64]) -> ();
    delete_lesson(repo_id: i64, id: i64) -> bool;
    set_decayed_confidence(repo_id: i64, updates: &[(i64, f64)]) -> ();
    repos_with_lessons() -> Vec<i64>;
    record_usage(u: &Usage) -> ();
    usage(repo_id: i64, pr: i64) -> Vec<Usage>;
    enqueue(j: &NewJob) -> bool;
    claim(lease: Duration, threads_only: bool) -> Option<Claimed>;
    queued_in(slot: &str) -> Vec<String>;
    finish(id: i64, attempt: i32, end: JobEnd) -> ();
    prune_jobs(older_than: Duration) -> u64;
}

const PR: i64 = 7;

async fn repo(s: &dyn Store, id: i64) -> i64 {
    s.upsert_repo(&Repo {
        id,
        installation_id: 1,
        full_name: format!("o/r{id}"),
    })
    .await
    .unwrap();
    id
}

fn finding(repo_id: i64, fp: &str) -> StoredFinding {
    StoredFinding {
        repo_id,
        pr: PR,
        fingerprint: fp.into(),
        path: "a.go".into(),
        start_line: 1,
        end_line: 1,
        severity: Severity::Major,
        category: Category::Logic,
        title: format!("title {fp}"),
        comment_id: None,
        status: FindingStatus::Open,
        first_sha: "sha1".into(),
        last_sha: "sha1".into(),
    }
}

fn lesson(repo_id: i64, rule: &str) -> NewLesson {
    NewLesson {
        repo_id,
        rule: rule.into(),
        kind: LessonKind::FalsePositive,
        scope: String::new(),
        confidence: 0.6,
        author: "alice".into(),
        pr: Some(PR),
        source_comment_id: None,
    }
}

fn usage(repo_id: i64, role: ModelRole) -> Usage {
    Usage {
        repo_id,
        pr: PR,
        head_sha: "sha1".into(),
        role,
        provider: "bedrock".into(),
        model: "m".into(),
        input_tokens: 1000,
        output_tokens: 200,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        cost_usd: 0.25,
        anchors: OutcomeStats::default(),
        failed_files: 0,
        kept_files: 0,
        reviewed_files: 0,
    }
}

async fn fps(s: &dyn Store, repo_id: i64, pr: i64, status: FindingStatus) -> Vec<String> {
    let found = s.findings(repo_id, pr, status).await.unwrap();
    found.into_iter().map(|f| f.fingerprint).collect()
}

async fn rules(s: &dyn Store, repo_id: i64, limit: Option<usize>) -> Vec<String> {
    let found = s.lessons(repo_id, limit).await.unwrap();
    found.into_iter().map(|l| l.lesson.rule).collect()
}

/// The repo's only Lesson with this rule.
async fn by_rule(s: &dyn Store, repo_id: i64, rule: &str) -> StoredLesson {
    let found = s.lessons(repo_id, None).await.unwrap();
    let mut matching = found.into_iter().filter(|l| l.lesson.rule == rule);
    let l = matching.next().expect("lesson exists");
    assert!(matching.next().is_none(), "one lesson per rule");
    l
}

async fn repo_rename_keeps_history(s: &dyn Store) {
    let old = |id| Repo {
        id,
        installation_id: 1,
        full_name: "o/old".into(),
    };
    s.upsert_repo(&old(42)).await.unwrap();
    s.upsert_finding(&finding(42, "fp")).await.unwrap();
    s.add_lesson(&lesson(42, "rule")).await.unwrap();

    // Renamed and transferred: same GitHub id. Another repo then takes the old name.
    let renamed = Repo {
        id: 42,
        installation_id: 2,
        full_name: "o2/new".into(),
    };
    s.upsert_repo(&renamed).await.unwrap();
    s.upsert_repo(&old(43)).await.unwrap();

    assert_eq!(fps(s, 42, PR, FindingStatus::Open).await, ["fp"]);
    assert_eq!(rules(s, 42, None).await, ["rule"]);
    assert!(fps(s, 43, PR, FindingStatus::Open).await.is_empty());
    assert!(rules(s, 43, None).await.is_empty());
}

async fn writes_need_a_recorded_repo(s: &dyn Store) {
    let unknown = |r: Result<(), StoreError>| matches!(r, Err(StoreError::UnknownRepo(9)));
    assert!(unknown(s.save_pr_state(9, PR, &PrState::default()).await));
    assert!(unknown(s.upsert_finding(&finding(9, "fp")).await));
    assert!(unknown(s.add_lesson(&lesson(9, "rule")).await.map(drop)));
    assert!(unknown(s.record_usage(&usage(9, ModelRole::Review)).await));
    assert!(unknown(s.add_pusher(9, PR, "bob").await));

    assert!(fps(s, 9, PR, FindingStatus::Open).await.is_empty());
    assert!(rules(s, 9, None).await.is_empty());
    assert!(s.usage(9, PR).await.unwrap().is_empty());
    assert!(s.pushers(9, PR).await.unwrap().is_empty());
}

async fn upsert_open_resolve(s: &dyn Store) {
    let r = repo(s, 1).await;
    s.upsert_finding(&finding(r, "fp1")).await.unwrap();
    s.upsert_finding(&StoredFinding {
        path: "b.go".into(),
        ..finding(r, "fp2")
    })
    .await
    .unwrap();
    assert_eq!(fps(s, r, PR, FindingStatus::Open).await, ["fp1", "fp2"]);

    s.resolve_findings(r, PR, &["fp1".into()]).await.unwrap();
    assert_eq!(fps(s, r, PR, FindingStatus::Open).await, ["fp2"]);
    assert_eq!(fps(s, r, PR, FindingStatus::Resolved).await, ["fp1"]);

    // Unknown Fingerprints and other PRs are ignored.
    s.resolve_findings(r, PR, &["ghost".into()]).await.unwrap();
    s.resolve_findings(r, PR + 1, &["fp2".into()])
        .await
        .unwrap();
    s.resolve_findings(r, PR, &[]).await.unwrap();
    assert_eq!(fps(s, r, PR, FindingStatus::Open).await, ["fp2"]);
}

/// A Finding counts from its first store; writing it again keeps that time.
async fn findings_before(s: &dyn Store) {
    let r = repo(s, 1).await;
    let gap = || tokio::time::sleep(Duration::from_millis(20));
    s.upsert_finding(&finding(r, "old")).await.unwrap();
    gap().await;
    let at = SystemTime::now();
    gap().await;
    s.upsert_finding(&finding(r, "new")).await.unwrap();
    s.upsert_finding(&finding(r, "old")).await.unwrap();
    let elsewhere = StoredFinding {
        pr: PR + 1,
        ..finding(r, "elsewhere")
    };
    s.upsert_finding(&elsewhere).await.unwrap();
    let before = async |status, at| {
        let found = s.findings_before(r, PR, status, at).await.unwrap();
        found.into_iter().map(|f| f.fingerprint).collect::<Vec<_>>()
    };
    let open = FindingStatus::Open;
    assert_eq!(before(open, at).await, ["old"]);
    assert_eq!(before(open, SystemTime::now()).await, ["new", "old"]);
    assert!(before(open, at - Duration::from_secs(60)).await.is_empty());

    s.resolve_findings(r, PR, &["old".into()]).await.unwrap();
    assert!(before(open, at).await.is_empty());
    assert_eq!(before(FindingStatus::Resolved, at).await, ["old"]);
}

async fn findings_in_location_order(s: &dyn Store) {
    let r = repo(s, 1).await;
    // Bytewise path order: "B.go" sorts before "a.go" whatever the server's locale.
    let at = |fp: &str, path: &str, line| StoredFinding {
        path: path.into(),
        start_line: line,
        end_line: line,
        ..finding(r, fp)
    };
    for f in [
        at("z", "b.go", 5),
        at("y", "a.go", 10),
        at("x", "a.go", 2),
        at("w", "a.go", 2),
        at("v", "B.go", 1),
    ] {
        s.upsert_finding(&f).await.unwrap();
    }
    assert_eq!(
        fps(s, r, PR, FindingStatus::Open).await,
        ["v", "w", "x", "y", "z"]
    );
}

async fn reaffirm_keeps_first_sha_and_comment(s: &dyn Store) {
    let r = repo(s, 1).await;
    s.upsert_finding(&StoredFinding {
        comment_id: Some(555),
        ..finding(r, "fp")
    })
    .await
    .unwrap();

    // The next push: moved, reworded, regraded, no comment id this round.
    let again = StoredFinding {
        start_line: 9,
        end_line: 10,
        severity: Severity::Minor,
        title: "reworded".into(),
        first_sha: "sha2".into(),
        last_sha: "sha2".into(),
        ..finding(r, "fp")
    };
    s.upsert_finding(&again).await.unwrap();
    let want = StoredFinding {
        first_sha: "sha1".into(),
        comment_id: Some(555),
        ..again.clone()
    };
    let got = s.finding_by_fingerprint(r, PR, "fp").await.unwrap();
    assert_eq!(got.as_ref(), Some(&want));
    assert_eq!(
        fps(s, r, PR, FindingStatus::Open).await,
        ["fp"],
        "no duplicate"
    );

    // A new comment id replaces the old one.
    s.upsert_finding(&StoredFinding {
        comment_id: Some(777),
        ..again
    })
    .await
    .unwrap();
    let got = s.finding_by_fingerprint(r, PR, "fp").await.unwrap();
    assert_eq!(got.and_then(|f| f.comment_id), Some(777));
}

async fn reaffirm_keeps_suppressed(s: &dyn Store) {
    let r = repo(s, 1).await;
    // A routine reaffirm (status open) must not reopen it; a critical one does, since a
    // suppression never covers a critical.
    let cases = [
        ("minor", Severity::Minor, FindingStatus::Suppressed),
        ("major", Severity::Major, FindingStatus::Suppressed),
        ("critical", Severity::Critical, FindingStatus::Open),
    ];
    for (fp, severity, want) in cases {
        s.upsert_finding(&finding(r, fp)).await.unwrap();
        s.suppress_finding(r, PR, fp).await.unwrap();
        let again = StoredFinding {
            severity,
            ..finding(r, fp)
        };
        s.upsert_finding(&again).await.unwrap();
        let got = s.finding_by_fingerprint(r, PR, fp).await.unwrap();
        assert_eq!(got.map(|f| f.status), Some(want), "{fp}");
    }
}

async fn suppressed_status(s: &dyn Store) {
    let r = repo(s, 1).await;
    s.upsert_finding(&finding(r, "a")).await.unwrap();
    s.upsert_finding(&finding(r, "b")).await.unwrap();
    s.resolve_findings(r, PR, &["b".into()]).await.unwrap();

    // Open and resolved Findings both suppress.
    s.suppress_finding(r, PR, "a").await.unwrap();
    s.suppress_finding(r, PR, "b").await.unwrap();
    s.suppress_finding(r, PR, "ghost").await.unwrap();
    // Resolving never touches a suppressed Finding.
    s.resolve_findings(r, PR, &["a".into(), "b".into()])
        .await
        .unwrap();

    assert_eq!(fps(s, r, PR, FindingStatus::Suppressed).await, ["a", "b"]);
    assert!(fps(s, r, PR, FindingStatus::Open).await.is_empty());
    assert!(fps(s, r, PR, FindingStatus::Resolved).await.is_empty());
    let a = s.finding_by_fingerprint(r, PR, "a").await.unwrap();
    assert_eq!(a.map(|f| f.status), Some(FindingStatus::Suppressed));
}

async fn unsuppress_finding(s: &dyn Store) {
    let r = repo(s, 1).await;
    for fp in ["sup", "open", "res"] {
        s.upsert_finding(&finding(r, fp)).await.unwrap();
    }
    s.suppress_finding(r, PR, "sup").await.unwrap();
    s.resolve_findings(r, PR, &["res".into()]).await.unwrap();

    for fp in ["sup", "open", "res", "ghost"] {
        s.unsuppress_finding(r, PR, fp).await.unwrap();
    }
    assert_eq!(fps(s, r, PR, FindingStatus::Open).await, ["open", "sup"]);
    assert_eq!(fps(s, r, PR, FindingStatus::Resolved).await, ["res"]);
    assert!(fps(s, r, PR, FindingStatus::Suppressed).await.is_empty());
}

async fn resolved_findings_accumulate(s: &dyn Store) {
    let r = repo(s, 1).await;
    for fp in ["a", "b"] {
        s.upsert_finding(&finding(r, fp)).await.unwrap();
    }
    assert!(fps(s, r, PR, FindingStatus::Resolved).await.is_empty());

    s.resolve_findings(r, PR, &["a".into()]).await.unwrap();
    assert_eq!(fps(s, r, PR, FindingStatus::Resolved).await, ["a"]);
    s.resolve_findings(r, PR, &["b".into()]).await.unwrap();
    assert_eq!(fps(s, r, PR, FindingStatus::Resolved).await, ["a", "b"]);

    // Found again: open, and out of the resolved set.
    s.upsert_finding(&finding(r, "a")).await.unwrap();
    assert_eq!(fps(s, r, PR, FindingStatus::Resolved).await, ["b"]);
    assert_eq!(fps(s, r, PR, FindingStatus::Open).await, ["a"]);
}

async fn finding_by_fingerprint(s: &dyn Store) {
    let r = repo(s, 1).await;
    let other = repo(s, 2).await;
    let f = StoredFinding {
        path: "foo.go".into(),
        ..finding(r, "fp")
    };
    s.upsert_finding(&f).await.unwrap();

    let got = s.finding_by_fingerprint(r, PR, "fp").await.unwrap();
    assert_eq!(got, Some(f));
    for (repo_id, pr, fp) in [(r, PR, "nope"), (r, PR + 1, "fp"), (other, PR, "fp")] {
        let got = s.finding_by_fingerprint(repo_id, pr, fp).await.unwrap();
        assert_eq!(got, None, "{repo_id}/{pr}/{fp}");
    }
}

async fn finding_by_comment(s: &dyn Store) {
    let r = repo(s, 1).await;
    for (fp, comment) in [
        ("fp-b", Some(5)),
        ("fp-a", Some(5)),
        ("fp-c", Some(6)),
        ("fp-d", None),
    ] {
        s.upsert_finding(&StoredFinding {
            comment_id: comment,
            ..finding(r, fp)
        })
        .await
        .unwrap();
    }
    let cases = [
        (PR, 5, Some("fp-a")), // two share it: the smallest Fingerprint
        (PR, 6, Some("fp-c")),
        (PR, 7, None),
        (PR + 1, 5, None),
    ];
    for (pr, comment, want) in cases {
        let got = s.finding_by_comment(r, pr, comment).await.unwrap();
        assert_eq!(
            got.map(|f| f.fingerprint).as_deref(),
            want,
            "{pr}/{comment}"
        );
    }
}

async fn finding_events(s: &dyn Store) {
    let r = repo(s, 1).await;
    // No Finding yet: nothing recorded.
    s.append_finding_event(r, PR, "fp", EventKind::New, Some("sha1"))
        .await
        .unwrap();
    assert!(s.finding_events(r, PR).await.unwrap().is_empty());

    s.upsert_finding(&finding(r, "fp")).await.unwrap();
    let history = [
        (EventKind::New, Some("sha1")),
        (EventKind::Reaffirm, Some("sha2")),
        (EventKind::Suppress, None), // from a reply: no head commit
        (EventKind::Unsuppress, None),
        (EventKind::Acknowledge, None),
        (EventKind::Carry, Some("sha3")),
        (EventKind::Resolve, Some("sha4")),
    ];
    for (kind, sha) in history {
        s.append_finding_event(r, PR, "fp", kind, sha)
            .await
            .unwrap();
    }
    let want: Vec<FindingEvent> = history
        .iter()
        .map(|&(kind, sha)| FindingEvent {
            fingerprint: "fp".into(),
            kind,
            sha: sha.map(str::to_owned),
        })
        .collect();
    assert_eq!(s.finding_events(r, PR).await.unwrap(), want);
    assert!(s.finding_events(r, PR + 1).await.unwrap().is_empty());
}

async fn pr_state_round_trip(s: &dyn Store) {
    let r = repo(s, 1).await;
    assert_eq!(s.pr_state(r, PR).await.unwrap(), PrState::default());

    let full = PrState {
        status_comment_id: Some(999),
        paused: true,
        last_reviewed_sha: Some("sha9".into()),
        // Whole seconds: Postgres keeps microseconds.
        last_reviewed_at: Some(SystemTime::UNIX_EPOCH + Duration::from_secs(1_795_167_000)),
        approved_sha: Some("sha8".into()),
        walkthrough: Some(r#"{"prose":"x"}"#.into()),
    };
    s.save_pr_state(r, PR, &full).await.unwrap();
    assert_eq!(s.pr_state(r, PR).await.unwrap(), full);
    assert_eq!(s.pr_state(r, PR + 1).await.unwrap(), PrState::default());

    // A save replaces the whole row.
    s.save_pr_state(r, PR, &PrState::default()).await.unwrap();
    assert_eq!(s.pr_state(r, PR).await.unwrap(), PrState::default());
}

async fn pushers_per_pr(s: &dyn Store) {
    let r = repo(s, 1).await;
    let other = repo(s, 2).await;
    for login in ["bob", "alice", "bob"] {
        s.add_pusher(r, PR, login).await.unwrap();
    }
    s.add_pusher(r, PR + 1, "carol").await.unwrap();
    s.add_pusher(other, PR, "dave").await.unwrap();
    assert_eq!(s.pushers(r, PR).await.unwrap(), ["alice", "bob"]);
    assert_eq!(s.pushers(r, PR + 1).await.unwrap(), ["carol"]);
    assert!(s.pushers(r, PR + 2).await.unwrap().is_empty());
}

async fn same_rule_is_one_lesson(s: &dyn Store) {
    let r = repo(s, 1).await;
    let first = s.add_lesson(&lesson(r, "same rule")).await.unwrap();
    for _ in 0..4 {
        let id = s.add_lesson(&lesson(r, "same rule")).await.unwrap();
        assert_eq!(id, first);
    }
    s.add_lesson(&lesson(r, "other rule")).await.unwrap();
    assert_eq!(rules(s, r, None).await, ["other rule", "same rule"]);
}

async fn reteach_revives(s: &dyn Store) {
    let r = repo(s, 1).await;
    let first = NewLesson {
        scope: "internal/x".into(),
        confidence: 0.3,
        source_comment_id: Some(100),
        pr: Some(1),
        ..lesson(r, "same rule")
    };
    let id = s.add_lesson(&first).await.unwrap();
    let taught = by_rule(s, r, "same rule").await.lesson.created_at;
    s.add_lesson(&lesson(r, "newer rule")).await.unwrap();
    s.mark_lessons_triggered(r, &[id]).await.unwrap();

    let reteach = NewLesson {
        kind: LessonKind::Convention,
        scope: "cmd".into(),
        confidence: 0.6,
        author: "bob".into(),
        pr: Some(2),
        source_comment_id: Some(200),
        ..first.clone()
    };
    assert_eq!(s.add_lesson(&reteach).await.unwrap(), id);
    assert_eq!(
        rules(s, r, None).await,
        ["same rule", "newer rule"],
        "revived is newest"
    );
    let got = by_rule(s, r, "same rule").await;
    assert!(got.lesson.created_at > taught, "the decay clock restarts");
    assert_eq!(got.lesson.scope, "internal/x", "the first scope is kept");
    assert_eq!(got.lesson.confidence, 0.6);
    assert_eq!(got.author, "bob");
    assert_eq!(got.kind, LessonKind::Convention);
    assert_eq!(got.pr, Some(2));
    assert_eq!(got.source_comment_id, Some(200));
    assert_eq!(
        (got.lesson.trigger_count, got.lesson.contradicted_count),
        (1, 0),
        "telemetry survives"
    );

    // A re-teach that does not know the kind, PR or comment keeps the known ones.
    let vague = NewLesson {
        kind: LessonKind::Unknown,
        author: "carol".into(),
        pr: None,
        source_comment_id: None,
        ..first
    };
    s.add_lesson(&vague).await.unwrap();
    let got = by_rule(s, r, "same rule").await;
    assert_eq!(got.author, "carol");
    assert_eq!(got.kind, LessonKind::Convention);
    assert_eq!(got.pr, Some(2));
    assert_eq!(got.source_comment_id, Some(200));
}

async fn lesson_round_trip(s: &dyn Store) {
    let r = repo(s, 1).await;
    let new = NewLesson {
        repo_id: r,
        rule: "no naked returns".into(),
        kind: LessonKind::FalsePositive,
        scope: "internal/state".into(),
        confidence: 0.42,
        author: "maintainer".into(),
        pr: Some(7),
        source_comment_id: Some(987_654),
    };
    let id = s.add_lesson(&new).await.unwrap();
    let got = by_rule(s, r, "no naked returns").await;
    assert!(got.lesson.created_at.is_some());
    let want = StoredLesson {
        lesson: learn::Lesson {
            id,
            rule: new.rule,
            scope: new.scope,
            created_at: got.lesson.created_at,
            confidence: 0.42,
            trigger_count: 0,
            contradicted_count: 0,
            last_triggered_at: None,
        },
        repo_id: r,
        kind: LessonKind::FalsePositive,
        author: new.author,
        pr: Some(7),
        source_comment_id: Some(987_654),
        decayed_confidence: None,
    };
    assert_eq!(got, want);
}

async fn lessons_newest_first(s: &dyn Store) {
    let r = repo(s, 1).await;
    let other = repo(s, 2).await;
    for rule in ["a", "b", "c"] {
        s.add_lesson(&lesson(r, rule)).await.unwrap();
    }
    s.add_lesson(&lesson(other, "x")).await.unwrap();

    assert_eq!(rules(s, r, None).await, ["c", "b", "a"]);
    assert_eq!(rules(s, r, Some(2)).await, ["c", "b"]);
    assert!(rules(s, r, Some(0)).await.is_empty());
    assert_eq!(rules(s, other, None).await, ["x"]);
}

async fn lesson_by_source_comment(s: &dyn Store) {
    let r = repo(s, 1).await;
    let other = repo(s, 2).await;
    s.add_lesson(&NewLesson {
        source_comment_id: Some(4242),
        ..lesson(r, "from a reply")
    })
    .await
    .unwrap();
    s.add_lesson(&lesson(r, "no provenance")).await.unwrap();

    let got = s.lesson_by_source_comment(r, 4242).await.unwrap();
    assert_eq!(got.map(|l| l.lesson.rule).as_deref(), Some("from a reply"));
    for (repo_id, comment) in [(r, 7777), (r, 0), (other, 4242)] {
        let got = s.lesson_by_source_comment(repo_id, comment).await.unwrap();
        assert_eq!(got, None, "{repo_id}/{comment}");
    }
}

async fn mark_lessons_triggered(s: &dyn Store) {
    let r = repo(s, 1).await;
    let other = repo(s, 2).await;
    let one = s.add_lesson(&lesson(r, "one")).await.unwrap();
    s.add_lesson(&lesson(r, "two")).await.unwrap();
    let elsewhere = s.add_lesson(&lesson(other, "elsewhere")).await.unwrap();
    let count = |l: StoredLesson| (l.lesson.trigger_count, l.lesson.last_triggered_at.is_some());

    s.mark_lessons_triggered(r, &[]).await.unwrap();
    s.mark_lessons_triggered(r, &[one]).await.unwrap();
    assert_eq!(count(by_rule(s, r, "one").await), (1, true));
    assert_eq!(count(by_rule(s, r, "two").await), (0, false));

    // Accumulates; a repeated id counts once; unknown and other repos' ids are ignored.
    s.mark_lessons_triggered(r, &[one, one, 999_999, elsewhere])
        .await
        .unwrap();
    assert_eq!(count(by_rule(s, r, "one").await), (2, true));
    assert_eq!(count(by_rule(s, other, "elsewhere").await), (0, false));
}

async fn delete_lesson(s: &dyn Store) {
    let r = repo(s, 1).await;
    let other = repo(s, 2).await;
    let id = s.add_lesson(&lesson(r, "rule")).await.unwrap();
    let elsewhere = s.add_lesson(&lesson(other, "elsewhere")).await.unwrap();

    assert!(
        !s.delete_lesson(r, elsewhere).await.unwrap(),
        "another repo's id"
    );
    assert_eq!(rules(s, other, None).await, ["elsewhere"]);
    assert!(s.delete_lesson(r, id).await.unwrap());
    assert!(rules(s, r, None).await.is_empty());
    assert!(!s.delete_lesson(r, id).await.unwrap(), "already gone");
}

async fn set_decayed_confidence(s: &dyn Store) {
    let r = repo(s, 1).await;
    let other = repo(s, 2).await;
    let id = s.add_lesson(&lesson(r, "rule")).await.unwrap();
    let elsewhere = s.add_lesson(&lesson(other, "elsewhere")).await.unwrap();

    s.set_decayed_confidence(r, &[]).await.unwrap();
    s.set_decayed_confidence(r, &[(id, 0.42), (999_999, 0.99), (elsewhere, 0.1)])
        .await
        .unwrap();
    let got = by_rule(s, r, "rule").await;
    assert_eq!(got.decayed_confidence, Some(0.42));
    assert_eq!(
        got.lesson.confidence, 0.6,
        "the live confidence is untouched"
    );
    let untouched = by_rule(s, other, "elsewhere").await;
    assert_eq!(untouched.decayed_confidence, None);
}

async fn repos_with_lessons(s: &dyn Store) {
    for id in [3, 1, 2] {
        repo(s, id).await;
    }
    let gone = s.add_lesson(&lesson(3, "rule")).await.unwrap();
    s.add_lesson(&lesson(1, "rule")).await.unwrap();
    assert_eq!(s.repos_with_lessons().await.unwrap(), [1, 3]);

    s.delete_lesson(3, gone).await.unwrap();
    assert_eq!(s.repos_with_lessons().await.unwrap(), [1]);
}

async fn usage_round_trip(s: &dyn Store) {
    let r = repo(s, 1).await;
    let review = Usage {
        cache_read_tokens: 300,
        cache_write_tokens: 40,
        cost_usd: 0.0123,
        anchors: OutcomeStats {
            inline: 3,
            relocated: 1,
            snapped: 1,
            demoted: 2,
            verified: 4,
            committable: 1,
        },
        failed_files: 1,
        kept_files: 12,
        reviewed_files: 5,
        ..usage(r, ModelRole::Review)
    };
    let summary = Usage {
        provider: "anthropic".into(),
        model: "claude-haiku-4-5".into(),
        ..usage(r, ModelRole::Summary)
    };
    let elsewhere = Usage {
        pr: PR + 1,
        ..usage(r, ModelRole::Verify)
    };
    for u in [&review, &summary, &elsewhere] {
        s.record_usage(u).await.unwrap();
    }
    assert_eq!(s.usage(r, PR).await.unwrap(), [review, summary]);
    assert_eq!(s.usage(r, PR + 1).await.unwrap(), [elsewhere]);
}

/// The worker's lease; long enough that no test sees it lapse.
const LEASE: Duration = Duration::from_secs(9 * 60);

fn job(delivery: &str, slot: &str) -> NewJob {
    NewJob {
        delivery: delivery.into(),
        slot: slot.into(),
        payload: format!(r#"{{"delivery":"{delivery}"}}"#),
        delay: Duration::ZERO,
    }
}

async fn claim(s: &dyn Store) -> Option<Claimed> {
    s.claim(LEASE, false).await.unwrap()
}

/// Claims until nothing is due; the deliveries claimed, in order.
async fn claim_all(s: &dyn Store) -> Vec<Claimed> {
    let mut out = Vec::new();
    while let Some(c) = claim(s).await {
        out.push(c);
    }
    out
}

fn deliveries(claimed: &[Claimed]) -> Vec<&str> {
    claimed.iter().map(|c| c.delivery.as_str()).collect()
}

async fn jobs_dedupe_and_delay(s: &dyn Store) {
    assert!(s.enqueue(&job("d1", "pr:1:1")).await.unwrap());
    assert!(
        !s.enqueue(&job("d1", "pr:1:2")).await.unwrap(),
        "a redelivery"
    );
    let push = NewJob {
        delay: Duration::from_secs(30),
        ..job("d2", "pr:1:3")
    };
    assert!(s.enqueue(&push).await.unwrap());

    let want = Claimed {
        id: 0,
        delivery: "d1".into(),
        attempt: 1,
        payload: r#"{"delivery":"d1"}"#.into(),
    };
    let got = claim(s).await.expect("d1 is due");
    assert_eq!(Claimed { id: 0, ..got }, want);
    assert_eq!(claim(s).await, None, "d2 waits 30 s");
}

async fn jobs_one_running_per_slot(s: &dyn Store) {
    for (d, slot) in [
        ("a", "pr:1:7"),
        ("b", "pr:1:7"),
        ("c", "pr:1:8"),
        ("t", "thread:1:99"),
    ] {
        s.enqueue(&job(d, slot)).await.unwrap();
    }
    // The Thread slot goes first though queued last, and runs beside PR 7's job; b waits.
    let running = claim_all(s).await;
    assert_eq!(deliveries(&running), ["t", "a", "c"]);

    let a = &running[1];
    s.finish(a.id, a.attempt, JobEnd::Done).await.unwrap();
    assert_eq!(deliveries(&claim_all(s).await), ["b"]);
}

/// The worker's Thread lane takes only Thread slots, so a reply never waits behind the
/// Reviews holding every other permit.
async fn jobs_thread_lane(s: &dyn Store) {
    for (d, slot) in [
        ("a", "pr:1:7"),
        ("t", "thread:1:99"),
        ("u", "thread:1:99"),
        ("v", "thread:1:98"),
    ] {
        s.enqueue(&job(d, slot)).await.unwrap();
    }
    let lane = async || s.claim(LEASE, true).await.unwrap();
    let t = lane().await.expect("a Thread job");
    let v = lane().await.expect("the other Thread");
    assert_eq!(deliveries(&[t.clone(), v]), ["t", "v"]);
    assert_eq!(lane().await, None, "u waits for t; a is not a Thread job");

    s.finish(t.id, t.attempt, JobEnd::Done).await.unwrap();
    assert_eq!(deliveries(&claim_all(s).await), ["u", "a"]);
}

async fn jobs_lapsed_lease_reclaimed(s: &dyn Store) {
    s.enqueue(&job("d", "pr:1:7")).await.unwrap();
    // A zero lease has lapsed by the next claim, as if each pod died mid-job.
    let mut attempts = Vec::new();
    while attempts.len() < 4 {
        let c = s
            .claim(Duration::ZERO, false)
            .await
            .unwrap()
            .expect("lease lapsed");
        assert_eq!(c.delivery, "d");
        attempts.push((c.id, c.attempt));
    }
    let id = attempts[0].0;
    assert_eq!(attempts, [(id, 1), (id, 2), (id, 3), (id, 4)]);

    // The worker finishes the fourth claim Dead without running it.
    let dead = JobEnd::Dead {
        error: "3 attempts".into(),
    };
    s.finish(id, 4, dead).await.unwrap();
    assert_eq!(s.claim(Duration::ZERO, false).await.unwrap(), None);
}

async fn jobs_stale_finish(s: &dyn Store) {
    s.enqueue(&job("d", "pr:1:7")).await.unwrap();
    let old = s.claim(Duration::ZERO, false).await.unwrap().expect("due");
    let new = claim(s).await.expect("lease lapsed");
    assert_eq!((new.id, new.attempt), (old.id, 2));

    // The first attempt's pod comes back and reports: nothing changes.
    s.finish(old.id, old.attempt, JobEnd::Done).await.unwrap();
    let retry = JobEnd::Retry {
        after: Duration::ZERO,
        error: "502".into(),
    };
    s.finish(old.id, old.attempt, retry).await.unwrap();
    assert_eq!(claim(s).await, None, "still running under attempt 2");
    assert_eq!(s.prune_jobs(Duration::ZERO).await.unwrap(), 0, "not done");

    // Attempt 2 finishes; a repeated report of it changes nothing either.
    s.finish(new.id, new.attempt, JobEnd::Done).await.unwrap();
    let retry = JobEnd::Retry {
        after: Duration::ZERO,
        error: "502".into(),
    };
    s.finish(new.id, new.attempt, retry).await.unwrap();
    assert_eq!(claim(s).await, None, "done stays done");
    assert_eq!(s.prune_jobs(Duration::ZERO).await.unwrap(), 1);
}

/// A retry is due `after` from now on the store's clock, the one the claim reads.
async fn jobs_retry(s: &dyn Store) {
    s.enqueue(&job("soon", "pr:1:1")).await.unwrap();
    s.enqueue(&job("later", "pr:1:2")).await.unwrap();
    for c in claim_all(s).await {
        let after = if c.delivery == "later" {
            Duration::from_secs(3600)
        } else {
            Duration::ZERO
        };
        let retry = JobEnd::Retry {
            after,
            error: "502".into(),
        };
        s.finish(c.id, c.attempt, retry).await.unwrap();
    }
    let again = claim_all(s).await;
    assert_eq!(deliveries(&again), ["soon"], "later waits an hour");
    assert_eq!(again[0].attempt, 2);
}

async fn jobs_prune(s: &dyn Store) {
    for (d, slot) in [
        ("done", "pr:1:1"),
        ("dead", "pr:1:2"),
        ("running", "pr:1:3"),
    ] {
        s.enqueue(&job(d, slot)).await.unwrap();
    }
    let queued = NewJob {
        delay: Duration::from_secs(3600),
        ..job("queued", "pr:1:4")
    };
    s.enqueue(&queued).await.unwrap();
    let claimed = claim_all(s).await;
    assert_eq!(deliveries(&claimed), ["done", "dead", "running"]);
    s.finish(claimed[0].id, 1, JobEnd::Done).await.unwrap();
    let dead = JobEnd::Dead { error: "x".into() };
    s.finish(claimed[1].id, 1, dead).await.unwrap();

    let three_days = Duration::from_secs(3 * 24 * 3600);
    assert_eq!(
        s.prune_jobs(three_days).await.unwrap(),
        0,
        "all under 3 days"
    );
    assert_eq!(s.prune_jobs(Duration::ZERO).await.unwrap(), 2);

    // A pruned delivery can be queued again; the running and queued ones are kept.
    assert!(s.enqueue(&job("done", "pr:1:1")).await.unwrap());
    assert!(!s.enqueue(&job("running", "pr:1:3")).await.unwrap());
    assert!(!s.enqueue(&job("queued", "pr:1:4")).await.unwrap());
}

/// Only queued jobs of the slot, running and ended ones aside, oldest due first.
async fn jobs_queued_in_a_slot(s: &dyn Store) {
    let later = NewJob {
        delay: Duration::from_secs(3600),
        ..job("later", "pr:1:7")
    };
    for j in [
        job("running", "pr:1:7"),
        later,
        job("soon", "pr:1:7"),
        job("elsewhere", "pr:1:8"),
    ] {
        s.enqueue(&j).await.unwrap();
    }
    let running = claim(s).await.expect("due");
    assert_eq!(running.delivery, "running");
    let payloads = |names: &[&str]| -> Vec<String> {
        names
            .iter()
            .map(|d| format!(r#"{{"delivery":"{d}"}}"#))
            .collect()
    };
    assert_eq!(
        s.queued_in("pr:1:7").await.unwrap(),
        payloads(&["soon", "later"])
    );

    s.finish(running.id, running.attempt, JobEnd::Done)
        .await
        .unwrap();
    let soon = claim(s).await.expect("due");
    s.finish(soon.id, soon.attempt, JobEnd::Dead { error: "x".into() })
        .await
        .unwrap();
    assert_eq!(s.queued_in("pr:1:7").await.unwrap(), payloads(&["later"]));
    assert!(s.queued_in("pr:1:9").await.unwrap().is_empty());
}
