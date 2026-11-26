//! Lesson ranking and decay: which maintainer-taught rules go into a review prompt,
//! in what order, and how much each one is still trusted. Pure: every function takes
//! its clock as a parameter, reads only its arguments, and persists nothing. The
//! `LEARN_*` env overrides are applied by `config`.

use std::f64::consts::LN_2;
use std::time::{Duration, SystemTime};

/// A maintainer-taught rule, with only the fields ranking and decay read.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Lesson {
    pub id: i64,
    /// The one-line guidance injected into the prompt.
    pub rule: String,
    /// The directory subtree the rule applies to (forward-slash, no leading or
    /// trailing slash). Empty means global: eligible on every review.
    pub scope: String,
    /// When it was taught. `None` (unknown) counts as fresh, so a bad timestamp
    /// never demotes a rule.
    pub created_at: Option<SystemTime>,
    /// How much the rule is trusted, in `[0, 1]`.
    pub confidence: f64,
    /// Times it was injected into a review prompt.
    pub trigger_count: u64,
    /// Times a later maintainer reply opposed it after it was injected.
    pub contradicted_count: u64,
    /// When it was last injected; `None` until the first time.
    pub last_triggered_at: Option<SystemTime>,
}

/// The ranking and decay knobs. [`Config::default`] holds the production values.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Most lessons injected into one prompt; 0 means no cap.
    pub cap_n: usize,
    /// Lessons fetched before scope filtering, so a relevant scoped rule is not
    /// starved by newer unrelated ones.
    pub candidate_limit: usize,
    /// Decay's recency half-life, and also the age past which a global lesson is
    /// demoted below the directory-scoped ones. Zero turns recency decay off.
    pub half_life: Duration,
    /// Decay never returns less, so an aged or contradicted rule still injects
    /// instead of silently vanishing.
    pub confidence_floor: f64,
    /// z-score of the Wilson interval (1.96 is 95%).
    pub wilson_z: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            cap_n: 20,
            candidate_limit: 100,
            half_life: Duration::from_secs(365 * 24 * 3600),
            confidence_floor: 0.1,
            wilson_z: 1.96,
        }
    }
}

/// Chooses the lessons to inject for a PR touching `changed_dirs`: fresh globals
/// (current repo-wide policy), then scoped lessons whose directory the PR touches,
/// then stale globals; each tier ordered by confidence, and the result cut only at
/// [`Config::cap_n`]. Confidence orders but never filters: under the cap every
/// eligible lesson is returned.
///
/// `candidates` must arrive newest-first; within a tier, equal confidence keeps that
/// order. Empty `changed_dirs` means "no file info": every scoped lesson is eligible,
/// so a suppressing rule is never dropped for lack of data.
pub fn rank(
    candidates: Vec<Lesson>,
    changed_dirs: &[&str],
    now: SystemTime,
    cfg: &Config,
) -> Vec<Lesson> {
    // Already directories: canonicalize, never climb to a parent.
    let dirs: Vec<String> = changed_dirs.iter().map(|d| canonical_dir(d)).collect();
    let (mut fresh, mut scoped, mut stale) = (Vec::new(), Vec::new(), Vec::new());
    for l in candidates {
        if l.scope.is_empty() {
            if is_stale(&l, now, cfg.half_life) {
                stale.push(l);
            } else {
                fresh.push(l);
            }
        } else if dirs.is_empty() || dirs.iter().any(|d| scope_matches(&l.scope, d)) {
            scoped.push(l);
        }
    }
    let mut out = Vec::with_capacity(fresh.len() + scoped.len() + stale.len());
    for mut tier in [fresh, scoped, stale] {
        // Stable sort, so equal confidence keeps the newest-first order. total_cmp
        // because a non-total order (NaN) can make Rust's sort panic.
        tier.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
        out.extend(tier);
    }
    if cfg.cap_n > 0 {
        out.truncate(cfg.cap_n);
    }
    out
}

/// A lesson's confidence recomputed from its signal: the Wilson lower bound of
/// triggers vs contradictions (the stored confidence when there is no signal yet),
/// halved every [`Config::half_life`] since it last triggered, clamped to
/// `[confidence_floor, 1]`. The caller stores it as a shadow value beside the live
/// confidence; nothing ranks by it yet.
pub fn decay(l: &Lesson, now: SystemTime, cfg: &Config) -> f64 {
    let pos = l.trigger_count as f64;
    let n = pos + l.contradicted_count as f64;
    let mut score = if n > 0.0 {
        wilson_lower_bound(pos, n, cfg.wilson_z)
    } else {
        l.confidence
    };
    // A future timestamp skips attenuation rather than boosting the score.
    if !cfg.half_life.is_zero()
        && let Some(age) = l.last_triggered_at.and_then(|t| now.duration_since(t).ok())
    {
        score *= (-LN_2 * age.as_secs_f64() / cfg.half_life.as_secs_f64()).exp();
    }
    // The floor is constant, so a stored confidence already below it is raised to
    // it: all the weakest lessons rank alike, but none vanishes.
    score.max(cfg.confidence_floor).min(1.0)
}

/// The lower bound of the Wilson score interval for `pos` successes out of `n > 0`:
/// penalizes tiny samples, so a lesson earns trust instead of scoring 100% off a
/// single trigger.
fn wilson_lower_bound(pos: f64, n: f64, z: f64) -> f64 {
    let phat = pos / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let centre = phat + z2 / (2.0 * n);
    let margin = z * ((phat * (1.0 - phat) + z2 / (4.0 * n)) / n).sqrt();
    (centre - margin) / denom
}

/// Whether a global lesson is older than `ttl`, by its creation time: trigger
/// telemetry says when it was used, not whether the policy is current. An unknown
/// or future creation time is fresh.
fn is_stale(l: &Lesson, now: SystemTime, ttl: Duration) -> bool {
    l.created_at
        .and_then(|t| now.duration_since(t).ok())
        .is_some_and(|age| age > ttl)
}

/// Normalizes a directory to forward slashes with no leading or trailing slash;
/// one that escapes the repo becomes the root ("").
fn canonical_dir(d: &str) -> String {
    let d = d.replace('\\', "/");
    let d = d.trim_matches('/');
    if d == "." || d == ".." || d.starts_with("../") {
        String::new()
    } else {
        d.to_owned()
    }
}

/// Whether a lesson scoped to `scope` applies to a PR touching directory `dir`: the
/// same directory, a subdirectory, or a parent, always on slash boundaries so
/// `internal/state` never matches `internal/stateful`.
fn scope_matches(scope: &str, dir: &str) -> bool {
    dir == scope || is_under(dir, scope) || is_under(scope, dir)
}

fn is_under(path: &str, dir: &str) -> bool {
    path.strip_prefix(dir)
        .is_some_and(|rest| rest.starts_with('/'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::UNIX_EPOCH;

    const HOUR: Duration = Duration::from_secs(3600);

    /// The fixed clock every test uses: 2026-06-23T12:00:00Z.
    fn now() -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(1_782_216_000)
    }

    fn global(id: i64, created: SystemTime, confidence: f64) -> Lesson {
        Lesson {
            id,
            created_at: Some(created),
            confidence,
            ..Lesson::default()
        }
    }

    fn scoped(id: i64, scope: &str, created: SystemTime) -> Lesson {
        Lesson {
            id,
            scope: scope.to_owned(),
            created_at: Some(created),
            ..Lesson::default()
        }
    }

    fn ids(ls: &[Lesson]) -> Vec<i64> {
        ls.iter().map(|l| l.id).collect()
    }

    fn signal(confidence: f64, triggers: u64, contradictions: u64) -> Lesson {
        Lesson {
            confidence,
            trigger_count: triggers,
            contradicted_count: contradictions,
            ..Lesson::default()
        }
    }

    #[test]
    fn never_shrinks_under_cap() {
        let cfg = Config::default();
        let stale = now() - 2 * cfg.half_life;
        let cands = vec![
            global(1, now(), 0.0),
            scoped(2, "internal/state", now()),
            global(3, stale, 0.1),
        ];
        let out = rank(cands, &["internal/state"], now(), &cfg);
        let mut got = ids(&out);
        got.sort_unstable();
        assert_eq!(
            got,
            vec![1, 2, 3],
            "every eligible lesson survives under the cap"
        );
    }

    #[test]
    fn caps_at_n_dropping_stale_globals_first() {
        let cfg = Config {
            cap_n: 3,
            ..Config::default()
        };
        let stale = now() - 2 * cfg.half_life;
        let cands = vec![
            global(1, now(), 0.0),
            global(2, now() - HOUR, 0.0),
            scoped(3, "internal/state", now()),
            global(4, stale, 0.0),
            global(5, stale - HOUR, 0.0),
        ];
        let out = rank(cands, &["internal/state"], now(), &cfg);
        assert_eq!(ids(&out), vec![1, 2, 3]);
    }

    #[test]
    fn tier_order() {
        let cfg = Config::default();
        let stale = now() - 2 * cfg.half_life;
        let cands = vec![
            global(10, now(), 0.0),
            global(11, now() - HOUR, 0.0),
            scoped(20, "internal/state", now()),
            global(30, stale, 0.0),
        ];
        let out = rank(cands, &["internal/state"], now(), &cfg);
        assert_eq!(ids(&out), vec![10, 11, 20, 30], "fresh, scoped, stale");
    }

    #[test]
    fn scope_gating() {
        let cfg = Config::default();
        let cands = || {
            vec![
                global(1, now(), 0.0),
                scoped(2, "internal/state", now()),
                scoped(3, "internal/review", now()),
            ]
        };
        let out = rank(cands(), &["internal/state"], now(), &cfg);
        assert_eq!(
            ids(&out),
            vec![1, 2],
            "untouched scope excluded, global kept"
        );

        let out = rank(cands(), &[], now(), &cfg);
        assert_eq!(ids(&out), vec![1, 2, 3], "no changed dirs: all eligible");
    }

    #[test]
    fn confidence_outranks_recency() {
        let cands = vec![global(1, now(), 0.3), global(2, now() - HOUR, 0.6)];
        let out = rank(cands, &[], now(), &Config::default());
        assert_eq!(ids(&out), vec![2, 1]);
    }

    #[test]
    fn mixed_confidence_changes_survival_under_cap() {
        let cfg = Config {
            cap_n: 1,
            ..Config::default()
        };
        let cands = vec![global(1, now(), 0.3), global(2, now() - HOUR, 0.6)];
        let out = rank(cands, &[], now(), &cfg);
        assert_eq!(
            ids(&out),
            vec![2],
            "the higher-confidence older row survives"
        );
    }

    #[test]
    fn low_confidence_ranks_below_taught() {
        let cands = vec![
            global(1, now(), 0.3),
            global(2, now() - 2 * HOUR, 0.3),
            global(3, now() - HOUR, 0.6),
            global(4, now() - 3 * HOUR, 0.6),
        ];
        let out = rank(cands, &[], now(), &Config::default());
        assert_eq!(
            ids(&out),
            vec![3, 4, 1, 2],
            "taught first, recency breaks ties"
        );
    }

    #[test]
    fn scope_matches_on_slash_boundaries() {
        let cases = [
            ("internal/state", "internal/state", true),
            ("internal/state", "internal/state/sub", true),
            ("internal/state/sub", "internal/state", true),
            ("internal/state", "internal/stateful", false),
            ("internal/state", "internal/review", false),
            ("internal/state", "", false),
            ("cmd", "cmd/diffagent", true),
        ];
        for (scope, dir, want) in cases {
            assert_eq!(scope_matches(scope, dir), want, "{scope:?} vs {dir:?}");
        }
    }

    #[test]
    fn canonical_dir_never_climbs() {
        let cases = [
            ("internal/state", "internal/state"),
            ("/internal/state/", "internal/state"),
            ("internal\\state", "internal/state"),
            (".", ""),
            ("..", ""),
            ("../outside", ""),
            ("", ""),
        ];
        for (input, want) in cases {
            assert_eq!(canonical_dir(input), want, "{input:?}");
        }
    }

    #[test]
    fn decay_no_signal_returns_stored() {
        let got = decay(&signal(0.6, 0, 0), now(), &Config::default());
        assert!((got - 0.6).abs() <= 1e-9, "got {got}");
    }

    #[test]
    fn decay_floor_clamp() {
        let cfg = Config::default();
        let mut contradicted = signal(0.3, 1, 50);
        contradicted.last_triggered_at = Some(now() - 10 * cfg.half_life);
        let got = decay(&contradicted, now(), &cfg);
        assert!(got >= cfg.confidence_floor - 1e-12, "below floor: {got}");

        let mut perfect = signal(1.0, 1000, 0);
        perfect.last_triggered_at = Some(now());
        let got = decay(&perfect, now(), &cfg);
        assert!(got <= 1.0 + 1e-12, "above 1.0: {got}");
    }

    #[test]
    fn decay_deterministic() {
        let cfg = Config::default();
        let mut l = signal(0.5, 8, 2);
        l.last_triggered_at = Some(now() - 30 * 24 * HOUR);
        let first = decay(&l, now(), &cfg);
        for _ in 0..5 {
            assert!((decay(&l, now(), &cfg) - first).abs() <= 1e-12);
        }
    }

    #[test]
    fn decay_half_life() {
        let cfg = Config::default();
        let mut just = signal(0.6, 20, 0);
        just.last_triggered_at = Some(now());
        let mut aged = just.clone();
        aged.last_triggered_at = Some(now() - cfg.half_life);

        let score_just = decay(&just, now(), &cfg);
        let score_aged = decay(&aged, now(), &cfg);
        assert!(score_just > 0.0);
        let ratio = score_aged / score_just;
        assert!((ratio - 0.5).abs() <= 0.01, "ratio {ratio}, want ~0.5");
    }

    #[test]
    fn decay_unknown_or_future_timestamp_skips_recency() {
        let cfg = Config::default();
        let pure_wilson = wilson_lower_bound(5.0, 5.0, cfg.wilson_z);
        // An unknown time is None.
        for last in [None, Some(now() + 24 * HOUR)] {
            let mut l = signal(0.7, 5, 0);
            l.last_triggered_at = last;
            let got = decay(&l, now(), &cfg);
            assert!(
                !got.is_nan() && got >= cfg.confidence_floor,
                "{last:?}: {got}"
            );
            assert_eq!(got, pure_wilson, "{last:?}: no attenuation");
        }
    }

    #[test]
    fn decay_stored_below_floor_clamped_up() {
        let cfg = Config::default();
        let got = decay(&signal(0.05, 0, 0), now(), &cfg);
        assert!((got - cfg.confidence_floor).abs() <= 1e-9, "got {got}");
    }

    #[test]
    fn decay_zero_half_life_no_nan() {
        let mut l = signal(0.6, 10, 2);
        l.last_triggered_at = Some(now() - 100 * 24 * HOUR);

        let cfg = Config {
            half_life: Duration::ZERO,
            wilson_z: 1.96,
            confidence_floor: 0.1,
            ..Config::default()
        };
        let got = decay(&l, now(), &cfg);
        assert!(got.is_finite() && got >= cfg.confidence_floor, "got {got}");

        let zero = Config {
            cap_n: 0,
            candidate_limit: 0,
            half_life: Duration::ZERO,
            confidence_floor: 0.0,
            wilson_z: 0.0,
        };
        assert!(decay(&l, now(), &zero).is_finite());
    }

    #[test]
    fn wilson_lower_bound_table() {
        let z = 1.96;
        let cases = [
            ("1/1 tiny sample penalised", 1.0, 1.0, 0.0, 0.30),
            ("1000/1000 large sample high", 1000.0, 1000.0, 0.99, 1.0),
            ("0/10 all failures", 0.0, 10.0, 0.0, 0.0001),
            ("8/10 mixed", 8.0, 10.0, 0.40, 0.80),
            ("80/100 mixed larger n", 80.0, 100.0, 0.70, 0.88),
        ];
        for (name, pos, n, min, max) in cases {
            let got = wilson_lower_bound(pos, n, z);
            assert!((0.0..=1.0).contains(&got), "{name}: out of [0,1]: {got}");
            assert!(got >= min - 1e-9 && got <= max + 1e-9, "{name}: {got}");
        }
        assert!(
            wilson_lower_bound(80.0, 100.0, z) > wilson_lower_bound(8.0, 10.0, z),
            "more evidence at the same ratio raises the bound"
        );
    }

    #[test]
    fn default_knobs() {
        let cfg = Config::default();
        assert_eq!(cfg.cap_n, 20);
        assert_eq!(cfg.candidate_limit, 100);
        assert_eq!(cfg.half_life, 8760 * HOUR);
        assert_eq!(cfg.confidence_floor, 0.1);
        assert_eq!(cfg.wilson_z, 1.96);
    }
}
