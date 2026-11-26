//! Runtime configuration from the environment: the service's settings with their
//! defaults, plus per-role model routing. `BEDROCK_REGION` has no default: unset, the
//! AWS SDK's own region chain picks the region.
//! Parsing goes through a lookup function, so tests never touch the process env.

use std::fmt;
use std::time::Duration;

use ipnet::IpNet;
use tracing::warn;

use crate::learn;

/// The review model when nothing else is configured: Haiku 4.5 via a global
/// cross-region inference profile, so Bedrock may serve it from any region.
const DEFAULT_REVIEW_MODEL: &str = "global.anthropic.claude-haiku-4-5-20251001-v1:0";

/// A value that must never reach logs. Debug prints only whether it is set, and
/// there is no Display. Equality is not constant-time: never use it to check a
/// credential.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct Secret(String);

impl Secret {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// The raw value, for the one place that sends it.
    pub fn expose(&self) -> &str {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(if self.0.is_empty() {
            "Secret(<unset>)"
        } else {
            "Secret(<redacted>)"
        })
    }
}

/// How a provider is called.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Wire {
    /// Bedrock Converse through the AWS SDK: region from `BEDROCK_REGION`, else the
    /// SDK's region chain; credentials from IAM or `AWS_BEARER_TOKEN_BEDROCK`, so no
    /// key here.
    BedrockConverse,
    AnthropicMessages {
        base_url: String,
        api_key: Secret,
    },
    OpenAiChat {
        base_url: String,
        api_key: Secret,
    },
}

/// The model one role calls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRoute {
    pub provider: String,
    /// The provider's model id, verbatim (it may contain '/' or ':').
    pub model: String,
    pub wire: Wire,
}

impl ModelRoute {
    fn bedrock(model: impl Into<String>) -> Self {
        Self {
            provider: "bedrock".to_owned(),
            model: model.into(),
            wire: Wire::BedrockConverse,
        }
    }
}

/// The model behind each role, from `<ROLE>_MODEL=provider/model`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Models {
    /// Per-file bug finding. Default: Bedrock `BEDROCK_MODEL_ID`.
    pub review: ModelRoute,
    /// Walkthrough and diagram. Default: Bedrock `BEDROCK_SUMMARY_MODEL_ID`, else the
    /// review model.
    pub summary: ModelRoute,
    /// The second look at a critical Finding before it blocks (C4). Default: Bedrock
    /// `BEDROCK_VERIFIER_MODEL_ID`, else off.
    pub verify: Option<ModelRoute>,
    /// The intent of a maintainer's reply to a finding. Default: the review model.
    pub classify: ModelRoute,
}

/// A setting that cannot be used. Messages name the variable, never its value.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ConfigError {
    #[error("{var}: not a valid integer")]
    NotAnInteger { var: &'static str },
    #[error("{var}: expected <provider>/<model>")]
    MalformedModel { var: &'static str },
    #[error(
        "{var}: unknown provider (built in: bedrock, anthropic, openai, zai, deepseek, \
         openrouter; others need LLM_PROVIDER_<NAME>_BASE_URL)"
    )]
    UnknownProvider { var: &'static str },
    #[error("{var}: the provider needs {key_var}")]
    MissingApiKey { var: &'static str, key_var: String },
    #[error("missing required config: {}", vars.join(", "))]
    Missing { vars: Vec<&'static str> },
    #[error("{var}: entry {entry} is not a CIDR range (a.b.c.d/n or x:y::/n)")]
    NotACidr { var: &'static str, entry: usize },
    #[error(
        "BOT_NAME: expected 1 to 39 letters, digits, '.', '_' or '-', starting with a letter \
         or digit"
    )]
    BadBotName,
}

/// All runtime settings. Debug is safe to log: every secret is a [`Secret`].
#[derive(Debug, Clone)]
pub struct Config {
    pub port: String,
    pub webhook_secret: Secret,
    /// The peers the listener accepts a connection from; empty accepts any.
    pub webhook_allowed_ips: Vec<IpNet>,
    /// 0 (unset) means dev mode: the webhook queues into memory and no worker runs.
    pub github_app_id: i64,
    /// PEM.
    pub github_private_key: Secret,
    /// Postgres: state and the job queue. Required with App credentials; empty is dev
    /// mode's memory, lost on restart.
    pub database_url: Secret,
    /// Empty means no ticket intent.
    pub jira_base_url: String,
    pub jira_email: String,
    pub jira_api_token: Secret,
    /// Jira project keys, upper-cased; when set, an issue in any other project is never
    /// looked up. Empty looks up any project the account can read.
    pub jira_projects: Vec<String>,
    /// Unset leaves the region to the AWS SDK's own chain (`AWS_REGION`, the profile,
    /// instance metadata).
    pub bedrock_region: Option<String>,
    pub models: Models,
    /// Dollars per 1M tokens, for the cost estimate.
    pub model_price_in_per_m: f64,
    pub model_price_out_per_m: f64,
    pub summary_price_in_per_m: f64,
    pub summary_price_out_per_m: f64,
    pub classify_price_in_per_m: f64,
    pub classify_price_out_per_m: f64,
    pub verify_price_in_per_m: f64,
    pub verify_price_out_per_m: f64,
    /// Review jobs per pod; memory-bound, not CPU-bound.
    pub worker_concurrency: usize,
    /// Model calls in flight per PR.
    pub per_file_concurrency: usize,
    /// Bound on one file's model call, so one slow file can't eat the job budget.
    pub per_file_timeout: Duration,
    /// `diffagent-goctx` processes per pod: each type-checks a Review's Go packages, the
    /// real memory spike.
    pub max_concurrent_loads: usize,
    /// Bound on one Review's context build; always below `per_file_timeout`.
    pub structural_timeout: Duration,
    /// The `diffagent-goctx` binary; a bare name is looked up on `PATH`.
    pub goctx_path: String,
    /// goctx's soft memory limit (`GOMEMLIMIT`), in MiB.
    pub goctx_memory_mb: u32,
    /// `owner/repo` entries; empty reviews every installed repo.
    pub repo_allowlist: Vec<String>,
    pub enable_gitleaks: bool,
    /// Empty uses gitleaks' defaults.
    pub gitleaks_config_path: String,
    /// Respond to replies on findings. On unless set to "false".
    pub enable_feedback: bool,
    /// One-click suggestion blocks; off renders advisory diffs only.
    pub committable_suggestions: bool,
    /// Posts the merge gate's check run, under the bot's name. Off until the App has
    /// Checks: write, since every post would fail.
    pub merge_gate: bool,
    /// `BOT_NAME`: the bot's name in place of the App's slug, for the merge gate's
    /// check run, which a ruleset requires by name, and the status comment's heading.
    /// Never the `@` handle, which stays the slug. `None` when unset.
    pub bot_name: Option<String>,
    pub learn: learn::Config,
}

impl Config {
    /// Reads the process environment.
    pub fn from_env() -> Result<Self, ConfigError> {
        Self::from_lookup(|key| std::env::var(key).ok())
    }

    /// Reads settings through `lookup`. An empty value counts as unset, and an
    /// unparseable number or duration, or a NaN or negative price, falls back to its
    /// default with a warning naming the variable, except `BOT_NAME`, which must be
    /// valid when set. Tracing is set up before this runs, so the warning is logged
    /// here.
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Result<Self, ConfigError> {
        let env = Env(&lookup);
        let per_file_timeout = env.duration("PER_FILE_TIMEOUT", Duration::from_secs(4 * 60));
        let mut structural_timeout =
            env.duration("STRUCTURAL_TIMEOUT", Duration::from_secs(2 * 60));
        // A structural timeout must leave budget for the model call after it, or it
        // would fail the whole file instead of degrading to a diff-only review.
        if structural_timeout >= per_file_timeout {
            structural_timeout = per_file_timeout / 2;
        }
        let github_app_id = match env.get("GITHUB_APP_ID") {
            Some(v) => v.parse().map_err(|_| ConfigError::NotAnInteger {
                var: "GITHUB_APP_ID",
            })?,
            None => 0,
        };
        Ok(Self {
            port: env.string("PORT", "8080"),
            webhook_secret: env.secret("GITHUB_WEBHOOK_SECRET"),
            webhook_allowed_ips: cidrs(&env, "WEBHOOK_ALLOWED_IPS")?,
            github_app_id,
            github_private_key: env.secret("GITHUB_APP_PRIVATE_KEY"),
            database_url: env.secret("DATABASE_URL"),
            jira_base_url: env.string("JIRA_BASE_URL", ""),
            jira_email: env.string("JIRA_EMAIL", ""),
            jira_api_token: env.secret("JIRA_API_TOKEN"),
            jira_projects: split_list(&env.string("JIRA_PROJECTS", ""))
                .iter()
                .map(|p| p.to_ascii_uppercase())
                .collect(),
            bedrock_region: env.get("BEDROCK_REGION"),
            models: models(&env)?,
            model_price_in_per_m: env.price("MODEL_PRICE_IN_PER_M"),
            model_price_out_per_m: env.price("MODEL_PRICE_OUT_PER_M"),
            summary_price_in_per_m: env.price("SUMMARY_MODEL_PRICE_IN_PER_M"),
            summary_price_out_per_m: env.price("SUMMARY_MODEL_PRICE_OUT_PER_M"),
            classify_price_in_per_m: env.price("CLASSIFY_MODEL_PRICE_IN_PER_M"),
            classify_price_out_per_m: env.price("CLASSIFY_MODEL_PRICE_OUT_PER_M"),
            verify_price_in_per_m: env.price("VERIFY_MODEL_PRICE_IN_PER_M"),
            verify_price_out_per_m: env.price("VERIFY_MODEL_PRICE_OUT_PER_M"),
            worker_concurrency: env.parsed("WORKER_CONCURRENCY", 8),
            per_file_concurrency: env.parsed("PER_FILE_CONCURRENCY", 4),
            per_file_timeout,
            max_concurrent_loads: env.parsed("MAX_CONCURRENT_LOADS", 1),
            structural_timeout,
            goctx_path: env.string("GOCTX_PATH", "diffagent-goctx"),
            goctx_memory_mb: env.parsed("GOCTX_MEMORY_MB", 800),
            repo_allowlist: split_list(&env.string("REPO_ALLOWLIST", "")),
            enable_gitleaks: env.flag("ENABLE_GITLEAKS"),
            gitleaks_config_path: env.string("GITLEAKS_CONFIG_PATH", ""),
            enable_feedback: env.get("ENABLE_FEEDBACK").as_deref() != Some("false"),
            committable_suggestions: env.flag("COMMITTABLE_SUGGESTIONS"),
            merge_gate: env.flag("ENABLE_MERGE_GATE"),
            bot_name: bot_name(lookup("BOT_NAME"))?,
            learn: learn_config(&env),
        })
    }

    /// Whether this is a GitHub App, which runs the worker. Without an App id it is
    /// dev mode.
    pub fn prod(&self) -> bool {
        self.github_app_id != 0
    }

    /// Fails fast at startup when required settings are missing. An App also needs its
    /// key and Postgres: each replica on its own memory would lose and double jobs.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let mut vars = Vec::new();
        if self.webhook_secret.is_empty() {
            vars.push("GITHUB_WEBHOOK_SECRET");
        }
        if self.prod() {
            if self.github_private_key.is_empty() {
                vars.push("GITHUB_APP_PRIVATE_KEY");
            }
            if self.database_url.is_empty() {
                vars.push("DATABASE_URL");
            }
        }
        if vars.is_empty() {
            Ok(())
        } else {
            Err(ConfigError::Missing { vars })
        }
    }
}

/// Resolves every role. Without any `<ROLE>_MODEL` every role is a Bedrock model from
/// the `BEDROCK_*_MODEL_ID` settings.
fn models(env: &Env) -> Result<Models, ConfigError> {
    let review = role(env, "REVIEW_MODEL")?.unwrap_or_else(|| {
        ModelRoute::bedrock(env.string("BEDROCK_MODEL_ID", DEFAULT_REVIEW_MODEL))
    });
    let summary_fallback = env.get("BEDROCK_SUMMARY_MODEL_ID").map(ModelRoute::bedrock);
    let verify_fallback = env
        .get("BEDROCK_VERIFIER_MODEL_ID")
        .map(ModelRoute::bedrock);
    Ok(Models {
        summary: role(env, "SUMMARY_MODEL")?
            .or(summary_fallback)
            .unwrap_or_else(|| review.clone()),
        verify: role(env, "VERIFY_MODEL")?.or(verify_fallback),
        classify: role(env, "CLASSIFY_MODEL")?.unwrap_or_else(|| review.clone()),
        review,
    })
}

/// `var`'s route when it is set.
fn role(env: &Env, var: &'static str) -> Result<Option<ModelRoute>, ConfigError> {
    env.get(var).map(|v| parse_route(env, var, &v)).transpose()
}

/// Parses `provider/model`, splitting on the first '/' because model ids contain
/// '/' themselves (`openrouter/deepseek/deepseek-chat`). Every provider but Bedrock
/// needs `<PROVIDER>_API_KEY`; a keyless local server can take any dummy key.
/// `LLM_PROVIDER_<NAME>_BASE_URL` adds an OpenAI-compatible provider, or points a
/// built-in one at another endpoint.
fn parse_route(env: &Env, var: &'static str, value: &str) -> Result<ModelRoute, ConfigError> {
    let (provider, model) = value
        .split_once('/')
        .filter(|(p, m)| !p.is_empty() && !m.is_empty())
        .ok_or(ConfigError::MalformedModel { var })?;
    if provider == "bedrock" {
        return Ok(ModelRoute::bedrock(model));
    }
    let (anthropic, default_url) = match provider {
        "anthropic" => (true, Some("https://api.anthropic.com")),
        "openai" => (false, Some("https://api.openai.com/v1")),
        "zai" => (false, Some("https://api.z.ai/api/paas/v4")),
        "deepseek" => (false, Some("https://api.deepseek.com")),
        "openrouter" => (false, Some("https://openrouter.ai/api/v1")),
        _ => (false, None),
    };
    let name = env_name(provider);
    let base_url = env
        .get(&format!("LLM_PROVIDER_{name}_BASE_URL"))
        .or(default_url.map(str::to_owned))
        .ok_or(ConfigError::UnknownProvider { var })?;
    let key_var = format!("{name}_API_KEY");
    let api_key = Secret(
        env.get(&key_var)
            .ok_or(ConfigError::MissingApiKey { var, key_var })?,
    );
    let wire = if anthropic {
        Wire::AnthropicMessages { base_url, api_key }
    } else {
        Wire::OpenAiChat { base_url, api_key }
    };
    Ok(ModelRoute {
        provider: provider.to_owned(),
        model: model.to_owned(),
        wire,
    })
}

/// A provider name as it appears in env var names: `my-llm` → `MY_LLM`.
fn env_name(provider: &str) -> String {
    provider
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
}

/// The `LEARN_*` overrides of the ranking knobs.
fn learn_config(env: &Env) -> learn::Config {
    let d = learn::Config::default();
    learn::Config {
        cap_n: env.parsed("LEARN_CAP_N", d.cap_n),
        candidate_limit: env.parsed("LEARN_CANDIDATE_LIMIT", d.candidate_limit),
        half_life: env.duration("LEARN_HALF_LIFE", d.half_life),
        ..d
    }
}

/// Env access: empty is unset, and a bad value falls back to the default rather than
/// failing startup, with a warning that names the variable and never shows the value:
/// it could be a secret pasted into the wrong one.
struct Env<'a>(&'a dyn Fn(&str) -> Option<String>);

impl Env<'_> {
    fn get(&self, key: &str) -> Option<String> {
        (self.0)(key).filter(|v| !v.is_empty())
    }

    fn string(&self, key: &str, default: &str) -> String {
        self.get(key).unwrap_or_else(|| default.to_owned())
    }

    fn secret(&self, key: &str) -> Secret {
        Secret(self.get(key).unwrap_or_default())
    }

    /// Exactly "true"; anything else is false.
    fn flag(&self, key: &str) -> bool {
        self.get(key).as_deref() == Some("true")
    }

    fn parsed<T: std::str::FromStr>(&self, key: &str, default: T) -> T {
        self.checked(key, default, |v| v.parse().ok())
    }

    /// Dollars per 1M tokens: a finite number, zero or more.
    fn price(&self, key: &str) -> f64 {
        self.checked(key, 0.0, |v| {
            v.parse().ok().filter(|p: &f64| p.is_finite() && *p >= 0.0)
        })
    }

    /// A positive Go duration ("4m", "90s", "8760h"); anything else is `default`.
    fn duration(&self, key: &str, default: Duration) -> Duration {
        self.checked(key, default, |v| {
            parse_go_duration(v).filter(|d| !d.is_zero())
        })
    }

    /// `key` read by `read`; `default` when unset, or with a warning when `read`
    /// refuses the value.
    fn checked<T>(&self, key: &str, default: T, read: impl Fn(&str) -> Option<T>) -> T {
        let Some(value) = self.get(key) else {
            return default;
        };
        read(&value).unwrap_or_else(|| {
            warn!(var = key, "config value not valid; using the default");
            default
        })
    }
}

/// Parses Go's duration syntax: one or more decimal numbers each with a unit (ns, us,
/// µs, ms, s, m, h), like "1h30m" or "1.5s". There is no "d" unit: 365 days is
/// "8760h". A sign is rejected, since every caller needs a positive duration anyway.
fn parse_go_duration(s: &str) -> Option<Duration> {
    if s.is_empty() {
        return None;
    }
    let mut rest = s;
    let mut secs = 0.0;
    while !rest.is_empty() {
        let unit_at = rest
            .find(|c: char| !(c.is_ascii_digit() || c == '.'))
            .unwrap_or(rest.len());
        let (number, tail) = rest.split_at(unit_at);
        let value: f64 = number.parse().ok()?;
        let unit_end = tail
            .find(|c: char| c.is_ascii_digit() || c == '.')
            .unwrap_or(tail.len());
        let (unit, tail) = tail.split_at(unit_end);
        let scale = match unit {
            "ns" => 1e-9,
            "us" | "µs" | "μs" => 1e-6,
            "ms" => 1e-3,
            "s" => 1.0,
            "m" => 60.0,
            "h" => 3600.0,
            _ => return None,
        };
        secs += value * scale;
        rest = tail;
    }
    Duration::try_from_secs_f64(secs).ok()
}

/// Splits a comma-separated value into trimmed, non-empty items.
/// `var`'s comma-separated CIDR ranges, IPv4 or IPv6. A bad entry fails startup: a
/// range dropped with a warning would turn away the deliveries it was meant to let in.
fn cidrs(env: &Env, var: &'static str) -> Result<Vec<IpNet>, ConfigError> {
    let entries = split_list(&env.string(var, ""));
    let parse = |(i, e): (usize, &String)| {
        e.parse()
            .map_err(|_| ConfigError::NotACidr { var, entry: i + 1 })
    };
    entries.iter().enumerate().map(parse).collect()
}

/// `BOT_NAME` when set and not empty: 1 to 39 of `[A-Za-z0-9._-]`, starting with a
/// letter or digit.
/// It names the required check run and shows in Markdown, so a bad one stops startup
/// rather than falling back: a fallback would post a check run no ruleset requires.
fn bot_name(value: Option<String>) -> Result<Option<String>, ConfigError> {
    let Some(name) = value.filter(|v| !v.is_empty()) else {
        return Ok(None);
    };
    let first = name
        .bytes()
        .next()
        .is_some_and(|b| b.is_ascii_alphanumeric());
    let rest = name
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b".-_".contains(&b));
    if first && rest && name.len() <= 39 {
        Ok(Some(name))
    } else {
        Err(ConfigError::BadBotName)
    }
}

fn split_list(v: &str) -> Vec<String> {
    v.split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(str::to_owned)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn load(vars: &[(&str, &str)]) -> Result<Config, ConfigError> {
        let env: HashMap<String, String> = vars
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        Config::from_lookup(|k| env.get(k).cloned())
    }

    fn route(provider: &str, model: &str, wire: Wire) -> ModelRoute {
        ModelRoute {
            provider: provider.to_owned(),
            model: model.to_owned(),
            wire,
        }
    }

    fn openai_chat(base_url: &str, key: &str) -> Wire {
        Wire::OpenAiChat {
            base_url: base_url.to_owned(),
            api_key: Secret::new(key),
        }
    }

    #[test]
    fn validate() {
        let c = load(&[]).unwrap();
        assert_eq!(
            c.validate(),
            Err(ConfigError::Missing {
                vars: vec!["GITHUB_WEBHOOK_SECRET"]
            })
        );
        let c = load(&[("GITHUB_WEBHOOK_SECRET", "s")]).unwrap();
        assert_eq!(c.validate(), Ok(()));
    }

    #[test]
    fn validate_app_needs_key_and_database() {
        let app = [("GITHUB_WEBHOOK_SECRET", "s"), ("GITHUB_APP_ID", "1")];
        let key = ("GITHUB_APP_PRIVATE_KEY", "pem");
        let db = ("DATABASE_URL", "postgres://h/db");
        type Vars<'a> = &'a [(&'a str, &'a str)];
        let cases: [(Vars, &[&str]); 4] = [
            (&[], &["GITHUB_APP_PRIVATE_KEY", "DATABASE_URL"]),
            (&[key], &["DATABASE_URL"]),
            (&[db], &["GITHUB_APP_PRIVATE_KEY"]),
            (&[key, db], &[]),
        ];
        for (extra, missing) in cases {
            let c = load(&[&app[..], extra].concat()).unwrap();
            assert!(c.prod());
            let want = if missing.is_empty() {
                Ok(())
            } else {
                Err(ConfigError::Missing {
                    vars: missing.to_vec(),
                })
            };
            assert_eq!(c.validate(), want, "{extra:?}");
        }

        // Dev mode: no App, so neither is needed, and an unknown setting means nothing.
        let c = load(&[("GITHUB_WEBHOOK_SECRET", "s"), ("UNKNOWN_SETTING", "x")]).unwrap();
        assert!(!c.prod());
        assert_eq!(c.validate(), Ok(()));
    }

    #[test]
    fn defaults_and_app_id() {
        let c = load(&[("GITHUB_WEBHOOK_SECRET", "s")]).unwrap();
        assert_eq!(c.port, "8080");
        assert_eq!(c.github_app_id, 0);
        assert_eq!(c.worker_concurrency, 8);
        assert_eq!(c.per_file_concurrency, 4);
        assert_eq!(c.max_concurrent_loads, 1);
        assert_eq!(c.goctx_path, "diffagent-goctx");
        assert_eq!(c.goctx_memory_mb, 800);
        assert!(c.enable_feedback, "feedback is on by default");
        assert!(!c.enable_gitleaks && !c.committable_suggestions);
        assert!(!c.merge_gate, "the merge gate is off by default");
        assert!(!c.prod());
        assert_eq!(c.learn, learn::Config::default());
        let c = load(&[("GITHUB_APP_ID", "12345")]).unwrap();
        assert_eq!(c.github_app_id, 12345);
        assert!(c.prod());
    }

    #[test]
    fn bedrock_region_has_no_default() {
        type Vars<'a> = &'a [(&'a str, &'a str)];
        let cases: [(Vars, Option<&str>); 3] = [
            (&[], None),
            (&[("BEDROCK_REGION", "")], None),
            (&[("BEDROCK_REGION", "eu-west-1")], Some("eu-west-1")),
        ];
        for (vars, want) in cases {
            let c = load(vars).unwrap();
            assert_eq!(c.bedrock_region.as_deref(), want, "{vars:?}");
        }
    }

    #[test]
    fn bad_app_id() {
        let err = load(&[("GITHUB_APP_ID", "not-a-number")]).unwrap_err();
        assert_eq!(
            err,
            ConfigError::NotAnInteger {
                var: "GITHUB_APP_ID"
            }
        );
        assert!(
            !err.to_string().contains("not-a-number"),
            "never echo the value"
        );
    }

    #[test]
    fn repo_allowlist() {
        let c = load(&[("REPO_ALLOWLIST", " acme/a , acme/b ,")]).unwrap();
        assert_eq!(c.repo_allowlist, vec!["acme/a", "acme/b"]);
        assert!(load(&[]).unwrap().repo_allowlist.is_empty());
    }

    #[test]
    fn webhook_allowed_ips() {
        const VAR: &str = "WEBHOOK_ALLOWED_IPS";
        let bad = |entry| Err(ConfigError::NotACidr { var: VAR, entry });
        // (value, the ranges, or the entry refused)
        let cases: [(&str, Result<&[&str], ConfigError>); 10] = [
            ("", Ok(&[])),
            (
                " 192.30.252.0/22 , 2a0a:a440::/29,",
                Ok(&["192.30.252.0/22", "2a0a:a440::/29"]),
            ),
            ("0.0.0.0/0,::/0", Ok(&["0.0.0.0/0", "::/0"])),
            ("10.1.2.3/32, ::1/128", Ok(&["10.1.2.3/32", "::1/128"])),
            ("10.0.0.0/8,192.30.252.1", bad(2)),
            ("10.0.0.0/33", bad(1)),
            ("::/129", bad(1)),
            ("10.0.0/8", bad(1)),
            ("github", bad(1)),
            ("10.0.0.0/8,/8", bad(2)),
        ];
        for (value, want) in cases {
            let got = load(&[(VAR, value)]).map(|c| c.webhook_allowed_ips);
            let want = want.map(|nets| {
                nets.iter()
                    .map(|n| n.parse::<IpNet>().unwrap())
                    .collect::<Vec<_>>()
            });
            assert_eq!(got, want, "{value:?}");
        }
    }

    #[test]
    fn jira_projects() {
        let c = load(&[("JIRA_PROJECTS", " PROJ, ops ,,")]).unwrap();
        assert_eq!(c.jira_projects, ["PROJ", "OPS"]);
        assert!(load(&[]).unwrap().jira_projects.is_empty());
    }

    /// `BOT_NAME` unset or empty leaves the App's slug; a bad one stops startup.
    #[test]
    fn bot_name_unset_valid_and_bad_values() {
        let long = "a".repeat(40);
        let cases = [
            (None, Ok(None)),
            (Some("acme-review"), Ok(Some("acme-review"))),
            (Some("Acme.Bot_2"), Ok(Some("Acme.Bot_2"))),
            (Some("7"), Ok(Some("7"))),
            (Some(&long[1..]), Ok(Some(&long[1..]))),
            (Some(""), Ok(None)),
            (Some(&long), Err(ConfigError::BadBotName)),
            (Some("acme bot"), Err(ConfigError::BadBotName)),
            (Some("acme#1"), Err(ConfigError::BadBotName)),
            (Some("acme\nreview"), Err(ConfigError::BadBotName)),
            (Some("-acme"), Err(ConfigError::BadBotName)),
            (Some("akmé"), Err(ConfigError::BadBotName)),
        ];
        for (value, want) in cases {
            let vars: Vec<(&str, &str)> = value.map(|v| ("BOT_NAME", v)).into_iter().collect();
            let got = load(&vars).map(|c| c.bot_name);
            assert_eq!(got, want.map(|n| n.map(str::to_owned)), "{value:?}");
        }
        let err = ConfigError::BadBotName.to_string();
        assert!(err.starts_with("BOT_NAME: "), "{err}");
    }

    #[test]
    fn flags_need_exactly_true() {
        let c = load(&[
            ("ENABLE_GITLEAKS", "true"),
            ("COMMITTABLE_SUGGESTIONS", "1"),
            ("ENABLE_FEEDBACK", "false"),
            ("ENABLE_MERGE_GATE", "TRUE"),
        ])
        .unwrap();
        assert!(c.enable_gitleaks);
        assert!(!c.committable_suggestions);
        assert!(!c.merge_gate);
        assert!(!c.enable_feedback);
    }

    #[test]
    fn summary_model_config() {
        let c = load(&[]).unwrap();
        assert_eq!(
            c.models.summary, c.models.review,
            "unset: shares the review model"
        );
        assert_eq!(
            (c.summary_price_in_per_m, c.summary_price_out_per_m),
            (0.0, 0.0)
        );

        let c = load(&[
            ("BEDROCK_SUMMARY_MODEL_ID", "amazon.nova-lite-v1:0"),
            ("SUMMARY_MODEL_PRICE_IN_PER_M", "0.06"),
            ("SUMMARY_MODEL_PRICE_OUT_PER_M", "0.24"),
        ])
        .unwrap();
        assert_eq!(
            c.models.summary,
            ModelRoute::bedrock("amazon.nova-lite-v1:0")
        );
        assert_eq!(
            (c.summary_price_in_per_m, c.summary_price_out_per_m),
            (0.06, 0.24)
        );
    }

    /// A JSON log sink for one test.
    #[derive(Clone, Default)]
    struct Logs(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for Logs {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// A bad number, a NaN or negative price, or a bad duration falls back to its
    /// default with a warning that names the variable and never shows the value,
    /// which could be a secret pasted into the wrong variable.
    #[test]
    fn bad_values_fall_back_with_a_warning() {
        let pasted = "ghp_pastedintothewrongvariable";
        let vars = [
            ("MODEL_PRICE_IN_PER_M", "NaN"),
            ("MODEL_PRICE_OUT_PER_M", "-1"),
            ("VERIFY_MODEL_PRICE_IN_PER_M", "inf"),
            ("SUMMARY_MODEL_PRICE_IN_PER_M", "0.06"),
            ("WORKER_CONCURRENCY", pasted),
            ("PER_FILE_TIMEOUT", "30"),
        ];
        // A second live dispatcher: with ours alone, a callsite a parallel test hits
        // first caches that thread's "never" (tracing-core's `has_just_one`).
        let _second = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        let logs = Logs::default();
        let writer = logs.clone();
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(move || writer.clone())
            .finish();
        let c = tracing::subscriber::with_default(subscriber, || load(&vars).unwrap());
        assert_eq!(c.model_price_in_per_m, 0.0);
        assert_eq!(c.model_price_out_per_m, 0.0);
        assert_eq!(c.verify_price_in_per_m, 0.0);
        assert_eq!(c.summary_price_in_per_m, 0.06);
        assert_eq!(c.worker_concurrency, 8);
        assert_eq!(c.per_file_timeout, Duration::from_secs(240));
        let text = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
        let mut warned: Vec<String> = text
            .lines()
            .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
            .filter(|l| l["level"] == "WARN")
            .map(|l| l["fields"]["var"].as_str().unwrap().to_owned())
            .collect();
        warned.sort();
        assert_eq!(
            warned,
            [
                "MODEL_PRICE_IN_PER_M",
                "MODEL_PRICE_OUT_PER_M",
                "PER_FILE_TIMEOUT",
                "VERIFY_MODEL_PRICE_IN_PER_M",
                "WORKER_CONCURRENCY",
            ]
        );
        assert!(!text.contains("pasted"), "{text}");
    }

    #[test]
    fn timeouts() {
        let secs = Duration::from_secs;
        // A bare number has no unit, so it falls back to the default rather than 0.
        for (value, want) in [("", secs(240)), ("90s", secs(90)), ("30", secs(240))] {
            let c = load(&[("PER_FILE_TIMEOUT", value)]).unwrap();
            assert_eq!(c.per_file_timeout, want, "PER_FILE_TIMEOUT={value:?}");
        }
        for (value, want) in [("", secs(120)), ("30s", secs(30)), ("30", secs(120))] {
            let c = load(&[("STRUCTURAL_TIMEOUT", value)]).unwrap();
            assert_eq!(c.structural_timeout, want, "STRUCTURAL_TIMEOUT={value:?}");
        }
    }

    #[test]
    fn goctx_settings() {
        let c = load(&[
            ("GOCTX_PATH", "/usr/local/bin/diffagent-goctx"),
            ("GOCTX_MEMORY_MB", "1200"),
            ("MAX_CONCURRENT_LOADS", "2"),
        ])
        .unwrap();
        assert_eq!(c.goctx_path, "/usr/local/bin/diffagent-goctx");
        assert_eq!(c.goctx_memory_mb, 1200);
        assert_eq!(c.max_concurrent_loads, 2);
        let c = load(&[("GOCTX_MEMORY_MB", "800MiB")]).unwrap();
        assert_eq!(c.goctx_memory_mb, 800, "a bad value falls back");
    }

    #[test]
    fn structural_timeout_clamped_below_per_file() {
        let c = load(&[("STRUCTURAL_TIMEOUT", "5m"), ("PER_FILE_TIMEOUT", "4m")]).unwrap();
        assert_eq!(c.structural_timeout, Duration::from_secs(2 * 60));
    }

    #[test]
    fn go_durations() {
        let cases = [
            ("4m", Some(240.0)),
            ("90s", Some(90.0)),
            ("8760h", Some(8760.0 * 3600.0)),
            ("1h30m", Some(5400.0)),
            ("1.5s", Some(1.5)),
            ("300ms", Some(0.3)),
            ("10µs", Some(0.00001)),
            ("0", None),
            ("30", None),
            ("365d", None),
            ("-1h", None),
            ("", None),
            (".s", None),
            ("h", None),
        ];
        for (input, want) in cases {
            let got = parse_go_duration(input).map(|d| d.as_secs_f64());
            match (got, want) {
                (Some(g), Some(w)) => assert!((g - w).abs() < 1e-9, "{input:?}: {g}"),
                _ => assert_eq!(got, want, "{input:?}"),
            }
        }
    }

    #[test]
    fn learn_knobs() {
        let c = load(&[
            ("LEARN_CAP_N", "5"),
            ("LEARN_CANDIDATE_LIMIT", "50"),
            ("LEARN_HALF_LIFE", "24h"),
        ])
        .unwrap();
        let want = learn::Config {
            cap_n: 5,
            candidate_limit: 50,
            half_life: Duration::from_secs(24 * 3600),
            ..learn::Config::default()
        };
        assert_eq!(c.learn, want);

        let c = load(&[("LEARN_HALF_LIFE", "365d"), ("LEARN_CAP_N", "x")]).unwrap();
        assert_eq!(c.learn, learn::Config::default(), "bad values fall back");
    }

    #[test]
    fn routing_defaults() {
        let c = load(&[]).unwrap();
        let haiku = ModelRoute::bedrock(DEFAULT_REVIEW_MODEL);
        assert_eq!(c.models.review, haiku);
        assert_eq!(c.models.summary, haiku);
        assert_eq!(c.models.verify, None);
        assert_eq!(c.models.classify, haiku);
    }

    #[test]
    fn routing_bedrock_model_ids() {
        let c = load(&[
            ("BEDROCK_MODEL_ID", "model-a"),
            ("BEDROCK_VERIFIER_MODEL_ID", "model-v"),
        ])
        .unwrap();
        assert_eq!(c.models.review, ModelRoute::bedrock("model-a"));
        assert_eq!(c.models.summary, ModelRoute::bedrock("model-a"));
        assert_eq!(c.models.verify, Some(ModelRoute::bedrock("model-v")));
        assert_eq!(c.models.classify, ModelRoute::bedrock("model-a"));
    }

    #[test]
    fn routing_role_overrides() {
        let c = load(&[
            ("REVIEW_MODEL", "anthropic/claude-sonnet-4-5"),
            ("ANTHROPIC_API_KEY", "ak"),
            ("CLASSIFY_MODEL", "zai/glm-4.6"),
            ("ZAI_API_KEY", "zk"),
            ("VERIFY_MODEL", "bedrock/amazon.nova-pro-v1:0"),
            ("BEDROCK_MODEL_ID", "ignored-when-review-model-set"),
        ])
        .unwrap();
        let review = route(
            "anthropic",
            "claude-sonnet-4-5",
            Wire::AnthropicMessages {
                base_url: "https://api.anthropic.com".to_owned(),
                api_key: Secret::new("ak"),
            },
        );
        assert_eq!(c.models.review, review);
        assert_eq!(
            c.models.summary, review,
            "summary falls back to the review model"
        );
        assert_eq!(
            c.models.classify,
            route(
                "zai",
                "glm-4.6",
                openai_chat("https://api.z.ai/api/paas/v4", "zk")
            )
        );
        assert_eq!(
            c.models.verify,
            Some(ModelRoute::bedrock("amazon.nova-pro-v1:0"))
        );
    }

    #[test]
    fn routing_builtin_base_urls() {
        let cases = [
            ("openai", "OPENAI_API_KEY", "https://api.openai.com/v1"),
            ("deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com"),
            (
                "openrouter",
                "OPENROUTER_API_KEY",
                "https://openrouter.ai/api/v1",
            ),
        ];
        for (provider, key_var, url) in cases {
            let value = format!("{provider}/m");
            let c = load(&[("REVIEW_MODEL", &value), (key_var, "k")]).unwrap();
            assert_eq!(c.models.review, route(provider, "m", openai_chat(url, "k")));
        }
    }

    #[test]
    fn routing_splits_on_first_slash() {
        let c = load(&[
            ("REVIEW_MODEL", "openrouter/deepseek/deepseek-chat:free"),
            ("OPENROUTER_API_KEY", "k"),
            ("SUMMARY_MODEL", "bedrock/us.amazon.nova-lite-v1:0"),
        ])
        .unwrap();
        assert_eq!(c.models.review.provider, "openrouter");
        assert_eq!(c.models.review.model, "deepseek/deepseek-chat:free");
        assert_eq!(
            c.models.summary,
            ModelRoute::bedrock("us.amazon.nova-lite-v1:0")
        );
    }

    #[test]
    fn routing_custom_provider() {
        let c = load(&[
            ("REVIEW_MODEL", "my-llm/qwen3-coder"),
            ("LLM_PROVIDER_MY_LLM_BASE_URL", "http://llm.internal/v1"),
            ("MY_LLM_API_KEY", "k"),
        ])
        .unwrap();
        assert_eq!(
            c.models.review,
            route(
                "my-llm",
                "qwen3-coder",
                openai_chat("http://llm.internal/v1", "k")
            )
        );

        // A built-in keeps its wire format but can point elsewhere.
        let c = load(&[
            ("REVIEW_MODEL", "anthropic/claude-haiku-4-5"),
            ("LLM_PROVIDER_ANTHROPIC_BASE_URL", "https://proxy.internal"),
            ("ANTHROPIC_API_KEY", "k"),
        ])
        .unwrap();
        assert!(matches!(
            c.models.review.wire,
            Wire::AnthropicMessages { ref base_url, .. } if base_url == "https://proxy.internal"
        ));
    }

    #[test]
    fn routing_errors() {
        let cases: [(&[(&str, &str)], ConfigError); 6] = [
            (
                &[("REVIEW_MODEL", "gpt-5")],
                ConfigError::MalformedModel {
                    var: "REVIEW_MODEL",
                },
            ),
            (
                &[("SUMMARY_MODEL", "/gpt-5")],
                ConfigError::MalformedModel {
                    var: "SUMMARY_MODEL",
                },
            ),
            (
                &[("VERIFY_MODEL", "openai/")],
                ConfigError::MalformedModel {
                    var: "VERIFY_MODEL",
                },
            ),
            (
                &[("CLASSIFY_MODEL", "mystery/m1")],
                ConfigError::UnknownProvider {
                    var: "CLASSIFY_MODEL",
                },
            ),
            (
                &[("REVIEW_MODEL", "openai/gpt-5")],
                ConfigError::MissingApiKey {
                    var: "REVIEW_MODEL",
                    key_var: "OPENAI_API_KEY".to_owned(),
                },
            ),
            (
                &[
                    ("REVIEW_MODEL", "my-llm/m"),
                    ("LLM_PROVIDER_MY_LLM_BASE_URL", "http://x"),
                ],
                ConfigError::MissingApiKey {
                    var: "REVIEW_MODEL",
                    key_var: "MY_LLM_API_KEY".to_owned(),
                },
            ),
        ];
        for (vars, want) in cases {
            let err = load(vars).unwrap_err();
            assert_eq!(err, want);
            let msg = err.to_string();
            for (_, value) in vars {
                assert!(!msg.contains(value), "{msg:?} echoes {value:?}");
            }
        }
    }

    #[test]
    fn debug_redacts_secrets() {
        let secrets = [
            ("GITHUB_WEBHOOK_SECRET", "hook-s3cret"),
            ("GITHUB_APP_PRIVATE_KEY", "-----BEGIN PEM-----"),
            ("DATABASE_URL", "postgres://u:db-s3cret@h/db"),
            ("JIRA_API_TOKEN", "jira-s3cret"),
            ("OPENAI_API_KEY", "sk-openai-s3cret"),
        ];
        let mut vars = secrets.to_vec();
        vars.push(("REVIEW_MODEL", "openai/gpt-5"));
        let c = load(&vars).unwrap();
        let debug = format!("{c:?}");
        for (var, value) in secrets {
            assert!(!debug.contains(value), "{var} leaked into Debug");
        }
        assert!(debug.contains("Secret(<redacted>)"));
        assert_eq!(c.models.review.provider, "openai");
        assert_eq!(format!("{:?}", Secret::default()), "Secret(<unset>)");
    }
}
