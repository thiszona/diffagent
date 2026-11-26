//! The one runner for git, go, goctx and gitleaks. A child gets only `PATH` and
//! the variables its [`Cmd`] names, never the pod's environment, with its database
//! URL and app key. It leads its own process group, and a timeout, or the call's
//! future being dropped, kills the whole group, so no `compile` or
//! `git-remote-https` outlives the call or holds one of its pipes.
//! Errors name the program, the exit code and a redacted stderr excerpt, never args
//! or env.

use std::ffi::OsString;
use std::io;
use std::os::unix::process::ExitStatusExt;
use std::path::PathBuf;
use std::process::{ExitStatus, Stdio};
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::{ChildStdin, Command};

use crate::config::Secret;

/// The longest stderr excerpt an error keeps.
const MAX_STDERR: usize = 300;

/// How much stderr is kept; the rest is read and dropped, so a chatty child never
/// blocks on a full pipe.
const STDERR_CAP: u64 = 64 * 1024;

/// One command to run. No `Debug`: args and env hold paths and settings an error
/// must not show (and a `Command`'s own `Debug` prints its env).
pub(crate) struct Cmd {
    pub program: OsString,
    pub args: Vec<OsString>,
    pub cwd: PathBuf,
    /// Set after `PATH`, the one variable the worker passes on, so it can replace it.
    pub env: Vec<(String, OsString)>,
    /// Written to the child's stdin, which is then closed; empty gives no stdin.
    pub stdin: Vec<u8>,
    /// Bounds the whole run: spawn to exit, all output read.
    pub timeout: Duration,
    /// More stdout than this many bytes is [`ProcError::TooLarge`].
    pub stdout_cap: usize,
}

/// Why a run failed. `pub` (in a crate-private module) so the public functions
/// built on [`run`] can return it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProcError {
    /// The child could not be started, or reading its pipes failed.
    #[error("{program}: {reason}")]
    Spawn { program: String, reason: String },
    #[error("{program}: timed out after {timeout:?}")]
    Timeout { program: String, timeout: Duration },
    #[error("{program}: more than {cap} bytes of output")]
    TooLarge { program: String, cap: usize },
    /// A non-zero exit; a child killed by a signal reports 128 + the signal, as
    /// shells do. `stderr` is trimmed, cut to 300 chars, every secret as `***`.
    #[error("{program}: exit {code}: {stderr}")]
    Exit {
        program: String,
        code: i32,
        stderr: String,
    },
}

/// What stops the concurrent pipe work early.
enum Failure {
    TooLarge,
    Io(io::Error),
}

/// Runs `cmd` and returns its stdout. Stdin is written while stdout and stderr are
/// read and the exit awaited, all inside one timeout, so a child that fills a pipe
/// while we write never deadlocks. On a timeout or oversized output the child's
/// whole process group is killed. Every non-empty value in `secrets` shows as `***`
/// in the error's stderr.
pub(crate) async fn run(cmd: Cmd, secrets: &[&Secret]) -> Result<Vec<u8>, ProcError> {
    let program = cmd.program.to_string_lossy().into_owned();
    let mut command = Command::new(&cmd.program);
    command.args(&cmd.args).current_dir(&cmd.cwd).env_clear();
    if let Some(path) = std::env::var_os("PATH") {
        command.env("PATH", path);
    }
    command
        .envs(cmd.env.iter().map(|(k, v)| (k, v)))
        .stdin(if cmd.stdin.is_empty() {
            Stdio::null()
        } else {
            Stdio::piped()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .kill_on_drop(true);
    let mut child = command.spawn().map_err(|e| ProcError::Spawn {
        program: program.clone(),
        reason: e.to_string(),
    })?;
    // Declared after `child`, so it drops first: an unreaped leader keeps its
    // pid, the group's id, from being reused before the kill.
    let mut group = Group(child.id());
    let (stdin, stdout, stderr) = (child.stdin.take(), child.stdout.take(), child.stderr.take());
    let work = async {
        tokio::try_join!(
            feed(stdin, &cmd.stdin),
            read_capped(stdout, cmd.stdout_cap),
            read_stderr(stderr),
            async { child.wait().await.map_err(Failure::Io) },
        )
    };
    // Every way out but an exit kills the group as `group` drops: a timeout,
    // oversized output, a pipe error, or this future dropped mid-run.
    match tokio::time::timeout(cmd.timeout, work).await {
        Ok(Ok(((), out, err, status))) => {
            // The leader is reaped, so its pid may be reused: never kill now.
            group.0 = None;
            if status.success() {
                return Ok(out);
            }
            Err(ProcError::Exit {
                program,
                code: exit_code(status),
                stderr: excerpt(&err, secrets),
            })
        }
        Ok(Err(Failure::TooLarge)) => Err(ProcError::TooLarge {
            program,
            cap: cmd.stdout_cap,
        }),
        Ok(Err(Failure::Io(e))) => Err(ProcError::Spawn {
            program,
            reason: e.to_string(),
        }),
        Err(_) => Err(ProcError::Timeout {
            program,
            timeout: cmd.timeout,
        }),
    }
}

/// The process group a running child leads, killed when this drops unless
/// cleared first.
struct Group(Option<u32>);

impl Drop for Group {
    /// Kills the group with the `kill` utility: std has no `killpg`, and `libc`
    /// stays out. Blocking, for the few milliseconds `kill` takes, since a drop
    /// can't await; best effort, as the group may be gone.
    fn drop(&mut self) {
        let Some(pid) = self.0 else { return };
        let mut kill = std::process::Command::new("kill");
        kill.args(["-KILL", "--", &format!("-{pid}")])
            .env_clear()
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        if let Some(path) = std::env::var_os("PATH") {
            kill.env("PATH", path);
        }
        let _ = kill.status();
    }
}

/// Writes `data` and closes stdin. A child that exits without reading it all
/// breaks the pipe; its exit status says what happened, so that error is dropped.
async fn feed(stdin: Option<ChildStdin>, data: &[u8]) -> Result<(), Failure> {
    if let Some(mut pipe) = stdin {
        let _ = pipe.write_all(data).await;
    }
    Ok(())
}

/// Reads at most `cap + 1` bytes: one past the cap is enough to know it's too much.
async fn read_capped(pipe: Option<impl AsyncRead + Unpin>, cap: usize) -> Result<Vec<u8>, Failure> {
    let mut buf = Vec::new();
    if let Some(pipe) = pipe {
        let limit = u64::try_from(cap).unwrap_or(u64::MAX).saturating_add(1);
        pipe.take(limit)
            .read_to_end(&mut buf)
            .await
            .map_err(Failure::Io)?;
    }
    if buf.len() > cap {
        return Err(Failure::TooLarge);
    }
    Ok(buf)
}

/// Keeps the first [`STDERR_CAP`] bytes and drains the rest.
async fn read_stderr(pipe: Option<impl AsyncRead + Unpin>) -> Result<Vec<u8>, Failure> {
    let mut buf = Vec::new();
    if let Some(mut pipe) = pipe {
        (&mut pipe)
            .take(STDERR_CAP)
            .read_to_end(&mut buf)
            .await
            .map_err(Failure::Io)?;
        tokio::io::copy(&mut pipe, &mut tokio::io::sink())
            .await
            .map_err(Failure::Io)?;
    }
    Ok(buf)
}

fn exit_code(status: ExitStatus) -> i32 {
    status
        .code()
        .or_else(|| status.signal().map(|s| 128 + s))
        .unwrap_or(-1)
}

/// Stderr for an error: secrets replaced before the cut, so no part of one
/// survives it, then trimmed and cut to [`MAX_STDERR`] chars.
fn excerpt(stderr: &[u8], secrets: &[&Secret]) -> String {
    let mut text = String::from_utf8_lossy(stderr).into_owned();
    for secret in secrets.iter().filter(|s| !s.is_empty()) {
        text = text.replace(secret.expose(), "***");
    }
    crate::model::http::cut(text.trim(), MAX_STDERR)
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::time::Instant;

    use super::*;

    fn sh(script: &str) -> Cmd {
        Cmd {
            program: "sh".into(),
            args: vec!["-c".into(), script.into()],
            cwd: std::env::temp_dir(),
            env: Vec::new(),
            stdin: Vec::new(),
            timeout: Duration::from_secs(10),
            stdout_cap: 1 << 20,
        }
    }

    /// A fresh directory for one test, under the system temp dir.
    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("diffagent-proc-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn alive(pid: &str) -> bool {
        std::process::Command::new("kill")
            .args(["-0", pid])
            .stderr(Stdio::null())
            .status()
            .unwrap()
            .success()
    }

    #[tokio::test]
    async fn only_path_and_cmd_env_reach_the_child() {
        let mut cmd = sh("env");
        cmd.program = "env".into();
        cmd.args.clear();
        cmd.env = vec![("DIFFAGENT_X".to_owned(), "1".into())];
        let out = String::from_utf8(run(cmd, &[]).await.unwrap()).unwrap();
        let mut keys: Vec<&str> = out
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|kv| kv.0)
            .collect();
        keys.sort_unstable();
        assert_eq!(keys, ["DIFFAGENT_X", "PATH"], "{out}");

        let mut cmd = sh("echo \"$PATH\"");
        cmd.env = vec![("PATH".to_owned(), "/bin:/usr/bin".into())];
        assert_eq!(run(cmd, &[]).await.unwrap(), b"/bin:/usr/bin\n");
    }

    #[tokio::test]
    async fn a_hang_times_out_and_its_whole_group_dies() {
        let dir = scratch("hang");
        let pidfile = dir.join("pid");
        let mut cmd = sh("sleep 60 & echo $! > \"$PIDFILE\"; wait");
        cmd.env = vec![("PIDFILE".to_owned(), pidfile.clone().into())];
        cmd.timeout = Duration::from_millis(500);
        let started = Instant::now();
        let err = run(cmd, &[]).await.unwrap_err();
        assert!(matches!(err, ProcError::Timeout { .. }), "{err:?}");
        assert!(started.elapsed() < Duration::from_secs(5));

        let grandchild = std::fs::read_to_string(&pidfile).unwrap();
        let grandchild = grandchild.trim();
        // Killed at once; reaped by init shortly after, which `kill -0` waits for.
        let mut tries = 0;
        while alive(grandchild) && tries < 50 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            tries += 1;
        }
        assert!(
            !alive(grandchild),
            "the backgrounded sleep outlived the timeout"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// A Review cancelled mid-run drops the future: the group dies with it.
    #[tokio::test]
    async fn a_dropped_run_kills_its_whole_group() {
        let dir = scratch("drop");
        let pidfile = dir.join("pid");
        let mut cmd = sh("sleep 60 & echo $! > \"$PIDFILE\"; wait");
        cmd.env = vec![("PIDFILE".to_owned(), pidfile.clone().into())];
        cmd.timeout = Duration::from_secs(60);
        let cancelled = tokio::time::timeout(Duration::from_millis(500), run(cmd, &[])).await;
        assert!(cancelled.is_err(), "still running when dropped");

        let grandchild = std::fs::read_to_string(&pidfile).unwrap();
        let grandchild = grandchild.trim();
        let mut tries = 0;
        while alive(grandchild) && tries < 50 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            tries += 1;
        }
        assert!(
            !alive(grandchild),
            "the backgrounded sleep outlived the drop"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn stdout_cap() {
        let mut cmd = sh("");
        cmd.program = "yes".into();
        cmd.args.clear();
        cmd.stdout_cap = 1000;
        let err = run(cmd, &[]).await.unwrap_err();
        assert_eq!(
            err,
            ProcError::TooLarge {
                program: "yes".to_owned(),
                cap: 1000
            }
        );

        let mut cmd = sh("printf abcd");
        cmd.stdout_cap = 4;
        assert_eq!(
            run(cmd, &[]).await.unwrap(),
            b"abcd",
            "exactly the cap fits"
        );
    }

    #[tokio::test]
    async fn a_mebibyte_through_stdin_does_not_deadlock() {
        let data: Vec<u8> = (0..1 << 20).map(|i: u32| (i % 251) as u8).collect();
        let mut cmd = sh("cat");
        cmd.stdin = data.clone();
        cmd.stdout_cap = 2 << 20;
        assert_eq!(run(cmd, &[]).await.unwrap(), data);
    }

    #[tokio::test]
    async fn exit_errors() {
        let token = Secret::new("ghs_0123456789abcdef");
        let unused = Secret::new("");
        let mut cmd = sh(
            "echo \"fatal: could not read from https://x-access-token:$T@github.com/o/r\" >&2; exit 128",
        );
        cmd.env = vec![("T".to_owned(), token.expose().into())];
        let err = run(cmd, &[&unused, &token]).await.unwrap_err();
        assert_eq!(
            err,
            ProcError::Exit {
                program: "sh".to_owned(),
                code: 128,
                stderr: "fatal: could not read from https://x-access-token:***@github.com/o/r"
                    .to_owned(),
            }
        );
        assert!(!err.to_string().contains(token.expose()));

        let cases = [
            ("exit 7", 7, String::new()),
            ("kill -9 $$", 137, String::new()),
            ("printf '%0400d' 0 >&2; exit 1", 1, "0".repeat(300)),
            ("printf ' é \\n' >&2; exit 2", 2, "é".to_owned()),
        ];
        for (script, code, stderr) in cases {
            let err = run(sh(script), &[]).await.unwrap_err();
            let want = ProcError::Exit {
                program: "sh".to_owned(),
                code,
                stderr,
            };
            assert_eq!(err, want, "{script}");
        }
    }

    #[tokio::test]
    async fn spawn_errors_name_only_the_program() {
        let mut cmd = sh("");
        cmd.program = "/nonexistent/diffagent-proc-test".into();
        cmd.args = vec!["--token=abc".into()];
        let err = run(cmd, &[]).await.unwrap_err();
        assert!(matches!(err, ProcError::Spawn { .. }), "{err:?}");
        assert!(!err.to_string().contains("abc"), "{err}");

        let mut cmd = sh("true");
        cmd.cwd = Path::new("/nonexistent/diffagent-proc-dir").to_owned();
        assert!(matches!(run(cmd, &[]).await, Err(ProcError::Spawn { .. })));
    }
}
