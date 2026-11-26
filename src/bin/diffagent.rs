//! The diffagent service: GitHub's webhooks in and, with App credentials, the worker
//! that runs the jobs they queue. Without an App id it is dev mode: the webhook queues
//! into memory and no worker runs.

use std::error::Error;
use std::process::ExitCode;
use std::sync::Arc;

use diffagent::config::Config;
use diffagent::context::GoCtx;
use diffagent::github::{self, Github};
use diffagent::jira::Jira;
use diffagent::model::Router;
use diffagent::state::{MemoryStore, PgStore, Store};
use diffagent::worker::{self, Deps};
use diffagent::{job, webhook, workspace};
use tokio::net::TcpListener;
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::watch;
use tracing::{error, info, warn};

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt().json().flatten_event(true).init();
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            error!(err = %e, "fatal");
            ExitCode::FAILURE
        }
    }
}

/// Serves until SIGTERM (or Ctrl-C), then stops claiming, drains HTTP and waits for the
/// running jobs (the worker caps that at 9 min).
async fn run() -> Result<(), Box<dyn Error>> {
    let cfg = Config::from_env()?;
    let (app, deps) = start(&cfg).await?;
    let mut term = signal(SignalKind::terminate())?;
    let listener = TcpListener::bind(format!("0.0.0.0:{}", cfg.port)).await?;
    info!(port = %cfg.port, "ingest listening");
    let (stop, stopped) = watch::channel(false);
    let worker = deps.map(|d| tokio::spawn(worker::run(d, stopped)));
    let shutdown = async move {
        tokio::select! {
            _ = term.recv() => {}
            _ = tokio::signal::ctrl_c() => {}
        }
        info!("shutting down");
        stop.send_replace(true);
    };
    webhook::serve(listener, app, cfg.webhook_allowed_ips.clone(), shutdown).await;
    if let Some(w) = worker {
        w.await?;
    }
    Ok(())
}

/// Everything before the listener opens; any failure exits non-zero. App credentials
/// need their key and Postgres, so a pod that can post never runs on memory.
async fn start(cfg: &Config) -> Result<(axum::Router, Option<Arc<Deps>>), Box<dyn Error>> {
    cfg.validate()?;
    let store: Arc<dyn Store> = if cfg.database_url.is_empty() {
        Arc::new(MemoryStore::default())
    } else {
        Arc::new(PgStore::connect(cfg.database_url.expose()).await?)
    };
    let app = webhook::router(Arc::clone(&store), cfg);
    if !cfg.prod() {
        warn!("dev mode: jobs queue in memory and no worker runs; set GITHUB_APP_ID for the App");
        return Ok((app, None));
    }
    let github = Github::new(github::API, cfg.github_app_id, &cfg.github_private_key)?;
    let router = Router::from_config(cfg).await?;
    let env = std::env::vars_os()
        .filter_map(|(k, v)| Some((k.into_string().ok()?, v)))
        .collect();
    let goctx = GoCtx::new(
        &cfg.goctx_path,
        cfg.goctx_memory_mb,
        env,
        cfg.max_concurrent_loads,
    );
    let swept = workspace::sweep(&std::env::temp_dir());
    let bot_login = github.bot_login().await?;
    // Before the listener and the worker, so no command is parsed under another mention.
    job::set_mention(&bot_login);
    info!(bot = %bot_login, swept, workers = cfg.worker_concurrency, "prod mode");
    let deps = Deps {
        store,
        github,
        router,
        goctx,
        jira: Jira::from_config(cfg),
        cfg: cfg.clone(),
        bot_login,
        remote: workspace::GITHUB.to_owned(),
    };
    Ok((app, Some(Arc::new(deps))))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(vars: &[(&str, &str)]) -> Config {
        Config::from_lookup(|k| {
            vars.iter()
                .find(|(n, _)| *n == k)
                .map(|(_, v)| (*v).to_owned())
        })
        .unwrap()
    }

    #[tokio::test]
    async fn dev_mode_answers_healthz_and_runs_no_worker() {
        let (app, deps) = start(&config(&[("GITHUB_WEBHOOK_SECRET", "s")]))
            .await
            .unwrap();
        assert!(deps.is_none());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(webhook::serve(
            listener,
            app,
            Vec::new(),
            std::future::pending(),
        ));
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let res = client
            .get(format!("http://{addr}/healthz"))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);
        assert_eq!(res.text().await.unwrap(), "ok");
    }

    /// Refused before any connection: the database URL points nowhere reachable.
    #[tokio::test]
    async fn an_app_needs_its_key_and_postgres() {
        let app = [("GITHUB_WEBHOOK_SECRET", "s"), ("GITHUB_APP_ID", "1")];
        let key = ("GITHUB_APP_PRIVATE_KEY", "pem");
        let db = ("DATABASE_URL", "postgres://127.0.0.1:9/none");
        for (extra, missing) in [
            (key, "DATABASE_URL"),
            (db, "GITHUB_APP_PRIVATE_KEY"),
            (
                ("GITHUB_APP_ID", "1"),
                "GITHUB_APP_PRIVATE_KEY, DATABASE_URL",
            ),
        ] {
            let err = start(&config(&[&app[..], &[extra][..]].concat()))
                .await
                .err()
                .unwrap();
            assert_eq!(
                err.to_string(),
                format!("missing required config: {missing}")
            );
        }
    }
}
