//! The checkout one Review shares, and the only way Rust reads a tree.
//!
//! A checkout is a shallow fetch of one commit into `<base>/diffagent-ws-<pid>-<n>/src`.
//! Git gets the installation token from a credential helper that reads it from the
//! environment, so it never reaches argv, a URL, `.git/config` or git's output (an
//! insteadOf URL in git's config would be echoed in its errors).
//! Symlinks are checked out as plain files: a committed link to `/proc/<pid>/environ`
//! or `/dev/zero` must never reach `go list`, gitleaks or a prompt.

use std::ffi::OsString;
use std::fs::{self, DirBuilder, File};
use std::io::{self, Read};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use crate::config::Secret;
use crate::proc::{self, Cmd, ProcError};

/// Where production checkouts fetch from; tests pass a `file://` or local URL.
pub const GITHUB: &str = "https://github.com";

/// Bounds each git step, and `go mod download`.
const STEP_TIMEOUT: Duration = Duration::from_secs(120);

/// Git and go print little on stdout; this only bounds a misbehaving child.
const STDOUT_CAP: usize = 1 << 20;

/// The most bytes a checkout writes. Every concurrent Review's checkout and the Go build
/// cache share one `/tmp`, and git stores a repeated file once, so a fetch of a few KiB
/// can name gigabytes. 1 GiB holds the trees of large real repositories.
const MAX_TREE_BYTES: u64 = 1 << 30;

/// The most files a checkout writes, for the same reason and for the volume's inodes.
const MAX_TREE_ENTRIES: usize = 200_000;

/// What `git ls-tree` may print: one size line per entry, at most 12 bytes for an entry
/// within [`MAX_TREE_BYTES`], so a longer listing is past a budget too.
const LISTING_CAP: usize = MAX_TREE_ENTRIES * 12;

/// The credential helper git runs for the remote: the token comes from the
/// environment git passes it, never from argv or config on disk.
const HELPER: &str =
    r#"!f(){ echo username=x-access-token; echo "password=$DIFFAGENT_GIT_TOKEN"; }; f"#;

/// Why a checkout failed. Messages never hold the token: git's stderr is redacted
/// by [`proc::run`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WorkspaceError {
    /// A SHA, owner or repo that could smuggle an option or a path; checked before
    /// anything is created or spawned.
    #[error("invalid {0}")]
    Invalid(&'static str),
    #[error("workspace dir: {0}")]
    Dir(String),
    #[error("git {step}: {source}")]
    Git {
        step: &'static str,
        source: ProcError,
    },
    /// The head's tree is past [`MAX_TREE_BYTES`] or [`MAX_TREE_ENTRIES`]; none of it
    /// was written.
    #[error("tree of {entries} files and {bytes} bytes is past the checkout budget")]
    TooLarge { entries: usize, bytes: u64 },
}

/// One checked-out commit. The tree is at [`Checkout::root`]; [`Checkout::dir`]
/// holds it and scratch files (gitleaks' report, the module cache) outside it.
/// Dropping it removes the whole directory.
#[derive(Debug)]
pub struct Checkout {
    dir: PathBuf,
}

impl Checkout {
    /// The private directory holding the tree, for files that must stay outside it.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The checked-out tree.
    pub fn root(&self) -> PathBuf {
        self.dir.join("src")
    }

    /// This Review's Go module cache, outside the tree. A PR chooses its modules, so a
    /// cache shared across Reviews and installations would grow with every PR; this one
    /// goes with the checkout.
    pub fn modcache(&self) -> PathBuf {
        self.dir.join("gomodcache")
    }
}

impl Drop for Checkout {
    /// On a runtime the tree is removed on a blocking thread, since a large checkout
    /// takes a while to delete and would stall the async workers; without one, at once.
    /// One lost to a shutdown is left to [`sweep`].
    fn drop(&mut self) {
        let dir = std::mem::take(&mut self.dir);
        match tokio::runtime::Handle::try_current() {
            Ok(rt) => drop(rt.spawn_blocking(move || remove(&dir))),
            Err(_) => {
                let _ = remove(&dir);
            }
        }
    }
}

/// Removes `dir` and everything in it. Go writes its module cache read-only, and an
/// entry of a directory without write permission cannot be removed, so after a first
/// failure every directory under `dir` is made the owner's to write and it is tried again.
fn remove(dir: &Path) -> io::Result<()> {
    if fs::remove_dir_all(dir).is_ok() {
        return Ok(());
    }
    writable(dir);
    fs::remove_dir_all(dir)
}

/// Makes `dir` and every directory under it 0700; `file_type` doesn't follow
/// symlinks, so a link is never followed out.
fn writable(dir: &Path) {
    let _ = fs::set_permissions(dir, fs::Permissions::from_mode(0o700));
    for entry in fs::read_dir(dir).into_iter().flatten().flatten() {
        if entry.file_type().is_ok_and(|t| t.is_dir()) {
            writable(&entry.path());
        }
    }
}

/// Shallow-fetches `owner/repo` at `sha` from `remote` into a new directory under
/// `base`, with symlinks checked out as plain files. The tree is sized before it is
/// written, and one past a budget is refused. A failed step removes everything it made.
pub async fn checkout(
    base: &Path,
    remote: &str,
    owner: &str,
    repo: &str,
    sha: &str,
    token: &Secret,
) -> Result<Checkout, WorkspaceError> {
    if !is_sha(sha) {
        return Err(WorkspaceError::Invalid("sha"));
    }
    if !is_name(owner) {
        return Err(WorkspaceError::Invalid("owner"));
    }
    if !is_name(repo) {
        return Err(WorkspaceError::Invalid("repo"));
    }
    let co = scratch(base)?;
    let root = co.root();
    let url = format!("{remote}/{owner}/{repo}.git");
    let git = async |step, args: &[&str], stdout_cap| {
        let cmd = Cmd {
            program: "git".into(),
            args: args.iter().map(OsString::from).collect(),
            cwd: root.clone(),
            env: git_env(remote, token),
            stdin: Vec::new(),
            timeout: STEP_TIMEOUT,
            stdout_cap,
        };
        proc::run(cmd, &[token])
            .await
            .map_err(|source| WorkspaceError::Git { step, source })
    };
    git("init", &["init", "-q"], STDOUT_CAP).await?;
    git("remote", &["remote", "add", "origin", &url], STDOUT_CAP).await?;
    let fetch = ["fetch", "-q", "--depth", "1", "--no-tags", "origin", sha];
    git("fetch", &fetch, STDOUT_CAP).await?;
    let sizes = [
        "ls-tree",
        "-r",
        "--full-tree",
        "--format=%(objectsize)",
        "FETCH_HEAD",
    ];
    tree_fits(&git("ls-tree", &sizes, LISTING_CAP).await?)?;
    let write = ["-c", "core.symlinks=false", "checkout", "-q", "FETCH_HEAD"];
    git("checkout", &write, STDOUT_CAP).await?;
    Ok(co)
}

/// Refuses a tree past a budget, from `git ls-tree`'s size line per entry (`-` for a
/// submodule, which checks out as an empty directory).
fn tree_fits(listing: &[u8]) -> Result<(), WorkspaceError> {
    let (mut entries, mut bytes) = (0, 0u64);
    for size in listing.split(|&b| b == b'\n').filter(|l| !l.is_empty()) {
        entries += 1;
        let size = std::str::from_utf8(size).ok().and_then(|s| s.parse().ok());
        bytes = bytes.saturating_add(size.unwrap_or(0));
    }
    if entries > MAX_TREE_ENTRIES || bytes > MAX_TREE_BYTES {
        return Err(WorkspaceError::TooLarge { entries, bytes });
    }
    Ok(())
}

/// An empty tree in a new private directory under `base`, removed on drop and swept
/// like a checkout: a checkout fetches into it, and a Thread's answer scans its file
/// window in one.
pub fn scratch(base: &Path) -> Result<Checkout, WorkspaceError> {
    let co = Checkout {
        dir: new_dir(base).map_err(|e| dir_error(base, &e))?,
    };
    fs::create_dir(co.root()).map_err(|e| dir_error(&co.root(), &e))?;
    Ok(co)
}

/// A directory that could not be made, named with its path.
fn dir_error(path: &Path, e: &io::Error) -> WorkspaceError {
    WorkspaceError::Dir(format!("{}: {e}", path.display()))
}

/// Best effort `go mod download` in `root`, so the type-checker finds private
/// modules: the one Go step with network and the token. `go_env` is goctx's
/// environment; `GOPROXY=off` is dropped, git's credential setup and
/// `GOPRIVATE=github.com/<owner>` added. Nothing runs without a root `go.mod`.
pub async fn go_mod_download(
    root: &Path,
    go_env: &[(String, OsString)],
    owner: &str,
    token: &Secret,
) -> Result<(), ProcError> {
    if !root.join("go.mod").is_file() {
        return Ok(());
    }
    let cmd = Cmd {
        program: "go".into(),
        args: vec!["mod".into(), "download".into()],
        cwd: root.to_owned(),
        env: mod_download_env(go_env, owner, token),
        stdin: Vec::new(),
        timeout: STEP_TIMEOUT,
        stdout_cap: STDOUT_CAP,
    };
    proc::run(cmd, &[token]).await.map(drop)
}

fn mod_download_env(
    go_env: &[(String, OsString)],
    owner: &str,
    token: &Secret,
) -> Vec<(String, OsString)> {
    let mut env: Vec<_> = go_env
        .iter()
        .filter(|(k, _)| k != "GOPROXY")
        .cloned()
        .collect();
    env.extend(git_env(GITHUB, token));
    env.push(("GOPRIVATE".to_owned(), format!("github.com/{owner}").into()));
    env
}

/// Git with no system or global config, no prompt, and the token only in the
/// variable the credential helper for `remote` reads.
fn git_env(remote: &str, token: &Secret) -> Vec<(String, OsString)> {
    [
        ("GIT_TERMINAL_PROMPT", "0"),
        ("GIT_CONFIG_NOSYSTEM", "1"),
        ("GIT_CONFIG_GLOBAL", "/dev/null"),
        ("GIT_CONFIG_COUNT", "1"),
        ("GIT_CONFIG_KEY_0", &format!("credential.{remote}.helper")),
        ("GIT_CONFIG_VALUE_0", HELPER),
        ("DIFFAGENT_GIT_TOKEN", token.expose()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v.into()))
    .collect()
}

fn is_sha(s: &str) -> bool {
    s.len() == 40 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// A GitHub owner or repo name that is safe in a URL path and as an argument.
fn is_name(s: &str) -> bool {
    !s.is_empty()
        && s != "."
        && s != ".."
        && !s.starts_with('-')
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-'))
}

/// Creates `base/diffagent-ws-<pid>-<n>` (0700) with the first free `n`: a container's
/// worker is PID 1 after every restart, so an earlier run's directory can exist.
fn new_dir(base: &Path) -> io::Result<PathBuf> {
    let pid = std::process::id();
    let mut n = 0u32;
    loop {
        let dir = base.join(format!("diffagent-ws-{pid}-{n}"));
        match DirBuilder::new().mode(0o700).create(&dir) {
            Ok(()) => return Ok(dir),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => n += 1,
            Err(e) => return Err(e),
        }
    }
}

/// Removes every `diffagent-ws-*` directory in `base` and returns how many: the
/// emptyDir outlives an OOM-killed worker, whose checkouts nothing else removes.
pub fn sweep(base: &Path) -> usize {
    let Ok(entries) = fs::read_dir(base) else {
        return 0;
    };
    entries
        .flatten()
        .filter(|e| e.file_name().as_bytes().starts_with(b"diffagent-ws-"))
        // `file_type` doesn't follow symlinks, so a link is never followed out.
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .filter(|e| remove(&e.path()).is_ok())
        .count()
}

/// The canonical path of `rel` when it exists under the canonical root. `rel` must be
/// non-empty and relative, without `..` or NUL; symlinks are resolved before the check.
pub fn contained(root: &Path, rel: impl AsRef<Path>) -> Option<PathBuf> {
    let rel = rel.as_ref();
    let plain = rel
        .components()
        .all(|c| matches!(c, Component::Normal(_) | Component::CurDir));
    if rel.as_os_str().is_empty() || !plain || rel.as_os_str().as_bytes().contains(&0) {
        return None;
    }
    let root = root.canonicalize().ok()?;
    let path = root.join(rel).canonicalize().ok()?;
    path.starts_with(&root).then_some(path)
}

/// Reads the regular file `rel` under `root`, if it is at most `cap` bytes of
/// UTF-8. `rel` must be relative, without `..` or NUL, and its canonical path must
/// stay under the canonical root (a non-canonical root breaks under macOS's `/var`
/// symlink). Anything else is `None`.
pub fn read(root: &Path, rel: impl AsRef<Path>, cap: usize) -> Option<String> {
    let path = contained(root, rel)?;
    // Checked before opening: opening a FIFO would block.
    let cap = u64::try_from(cap).unwrap_or(u64::MAX);
    let meta = fs::metadata(&path).ok()?;
    if !meta.is_file() || meta.len() > cap {
        return None;
    }
    let mut buf = Vec::new();
    File::open(&path)
        .ok()?
        .take(cap)
        .read_to_end(&mut buf)
        .ok()?;
    String::from_utf8(buf).ok()
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::net::TcpListener;
    use std::os::unix::fs::{PermissionsExt, symlink};
    use std::process::Command;

    use super::*;

    const SHA: &str = "0123456789abcdef0123456789abcdef01234567";

    /// A fresh directory for one test.
    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("diffagent-wstest-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Runs git in `dir` with no user or system config.
    fn git(dir: &Path, args: &[&str]) -> String {
        git_in(dir, args, b"")
    }

    /// [`git`] with `input` on its stdin.
    fn git_in(dir: &Path, args: &[&str], input: &[u8]) -> String {
        let mut child = Command::new("git")
            .args(args)
            .current_dir(dir)
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap())
            .envs([
                ("GIT_CONFIG_NOSYSTEM", "1"),
                ("GIT_CONFIG_GLOBAL", "/dev/null"),
                ("GIT_AUTHOR_NAME", "t"),
                ("GIT_AUTHOR_EMAIL", "t@example.com"),
                ("GIT_COMMITTER_NAME", "t"),
                ("GIT_COMMITTER_EMAIL", "t@example.com"),
            ])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(input).unwrap();
        let out = child.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap().trim().to_owned()
    }

    #[tokio::test]
    async fn checkout_turns_symlinks_into_files_and_skips_a_taken_dir() {
        let tmp = scratch("checkout");
        let repo = tmp.join("remote/o/r.git");
        fs::create_dir_all(&repo).unwrap();
        git(&repo, &["init", "-q"]);
        fs::write(repo.join("a.txt"), "hello\n").unwrap();
        symlink("/etc/passwd", repo.join("passwd")).unwrap();
        git(&repo, &["add", "-A"]);
        git(&repo, &["commit", "-q", "-m", "x"]);
        let sha = git(&repo, &["rev-parse", "HEAD"]);

        let base = tmp.join("base");
        let taken = base.join(format!("diffagent-ws-{}-0", std::process::id()));
        fs::create_dir_all(&taken).unwrap();
        let remote = format!("file://{}", tmp.join("remote").display());
        let token = Secret::new("ghs_checkout_token");
        let co = checkout(&base, &remote, "o", "r", &sha, &token)
            .await
            .unwrap();

        assert_eq!(
            co.dir(),
            base.join(format!("diffagent-ws-{}-1", std::process::id()))
        );
        let mode = fs::metadata(co.dir()).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o700);
        let link = fs::symlink_metadata(co.root().join("passwd")).unwrap();
        assert!(link.is_file(), "the symlink was checked out as a link");
        assert_eq!(
            read(&co.root(), "passwd", 1 << 10).as_deref(),
            Some("/etc/passwd")
        );
        assert_eq!(
            read(&co.root(), "a.txt", 1 << 10).as_deref(),
            Some("hello\n")
        );
        let config = fs::read_to_string(co.root().join(".git/config")).unwrap();
        assert!(!config.contains(token.expose()), "{config}");

        let dir = co.dir().to_owned();
        drop(co);
        assert!(gone(&dir).await, "drop left the checkout behind");
        assert!(taken.exists(), "drop removed another run's dir");
        fs::remove_dir_all(tmp).unwrap();
    }

    /// A remote `o/r` under `tmp` whose one commit names one `size`-byte blob of zeros
    /// from `entries` paths; its SHA. Git stores the blob once, so the remote and the
    /// fetch stay small whatever the tree adds up to.
    fn repeated_blob(tmp: &Path, entries: usize, size: usize) -> String {
        let repo = tmp.join("remote/o/r.git");
        fs::create_dir_all(&repo).unwrap();
        git(&repo, &["init", "-q"]);
        fs::write(repo.join("zero"), vec![0; size]).unwrap();
        let blob = git(&repo, &["hash-object", "-w", "zero"]);
        fs::remove_file(repo.join("zero")).unwrap();
        let listing: String = (0..entries)
            .map(|i| format!("100644 blob {blob}\tf{i}\n"))
            .collect();
        let tree = git_in(&repo, &["mktree"], listing.as_bytes());
        let sha = git(&repo, &["commit-tree", &tree, "-m", "x"]);
        git(&repo, &["update-ref", "HEAD", &sha]);
        sha
    }

    /// A tree past its byte budget is refused before anything of it is written, and
    /// the checkout's directory is removed: 1025 paths name one 1 MiB blob, a fetch of
    /// a few KiB. [`tree_fits_the_budgets`] covers the entry budget.
    #[tokio::test]
    async fn a_tree_past_its_budget_is_refused_unwritten() {
        let tmp = scratch("budget");
        let sha = repeated_blob(&tmp, 1025, 1 << 20);
        let base = tmp.join("base");
        fs::create_dir_all(&base).unwrap();
        let remote = format!("file://{}", tmp.join("remote").display());
        let err = checkout(&base, &remote, "o", "r", &sha, &Secret::new("t"))
            .await
            .unwrap_err();
        let want = WorkspaceError::TooLarge {
            entries: 1025,
            bytes: 1025 << 20,
        };
        assert_eq!(err, want);
        let dir = base.join(format!("diffagent-ws-{}-0", std::process::id()));
        assert!(gone(&dir).await, "the refused checkout left files");
        fs::remove_dir_all(tmp).unwrap();
    }

    #[test]
    fn tree_fits_the_budgets() {
        let lines = |n: usize, line: &str| line.repeat(n);
        let full = (MAX_TREE_BYTES / 2).to_string();
        let cases = [
            (String::new(), None),
            ("12\n-\n0\n".to_owned(), None),
            (format!("{full}\n{full}\n"), None),
            (
                format!("{full}\n{full}\n1\n"),
                Some((3, MAX_TREE_BYTES + 1)),
            ),
            (lines(MAX_TREE_ENTRIES, "0\n"), None),
            (
                lines(MAX_TREE_ENTRIES + 1, "-\n"),
                Some((MAX_TREE_ENTRIES + 1, 0)),
            ),
        ];
        for (listing, refused) in cases {
            let want = refused.map(|(entries, bytes)| WorkspaceError::TooLarge { entries, bytes });
            assert_eq!(
                tree_fits(listing.as_bytes()).err(),
                want,
                "{:.40?}",
                listing
            );
        }
    }

    /// Whether `dir` is removed within 10 s: a drop on the runtime removes it on a
    /// blocking thread.
    async fn gone(dir: &Path) -> bool {
        for _ in 0..1000 {
            if !dir.exists() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        false
    }

    /// A directory that cannot be made is named in the error.
    #[test]
    fn a_dir_error_names_its_path() {
        let err = super::scratch(Path::new("/nonexistent/diffagent-base")).unwrap_err();
        assert!(
            err.to_string().contains("/nonexistent/diffagent-base: "),
            "{err}"
        );
    }

    /// Off a runtime the drop removes the tree before it returns; on one, soon after.
    #[test]
    fn drop_removes_the_tree_with_or_without_a_runtime() {
        let base = scratch("drop");
        let co = super::scratch(&base).unwrap();
        fs::write(co.root().join("a"), "x").unwrap();
        let dir = co.dir().to_owned();
        drop(co);
        assert!(!dir.exists(), "no runtime: removed at once");

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        let co = super::scratch(&base).unwrap();
        let dir = co.dir().to_owned();
        rt.block_on(async move {
            drop(co);
            assert!(
                gone(&dir).await,
                "on a runtime: removed by a blocking thread"
            );
        });
        fs::remove_dir_all(base).unwrap();
    }

    #[tokio::test]
    async fn bad_input_fails_before_anything_is_made() {
        let base = scratch("invalid");
        let cases = [
            ("0123456789ABCDEF0123456789abcdef01234567", "o", "r", "sha"),
            (&SHA[..39], "o", "r", "sha"),
            ("0123456789abcdef0123456789abcdef012345678", "o", "r", "sha"),
            ("--upload-pack=x00000000000000000000000000", "o", "r", "sha"),
            (SHA, "", "r", "owner"),
            (SHA, ".", "r", "owner"),
            (SHA, "..", "r", "owner"),
            (SHA, "-c", "r", "owner"),
            (SHA, "o/x", "r", "owner"),
            (SHA, "o x", "r", "owner"),
            (SHA, "o", "..", "repo"),
            (SHA, "o", "-r", "repo"),
            (SHA, "o", "r?x", "repo"),
        ];
        let token = Secret::new("t");
        for (sha, owner, repo, field) in cases {
            let got = checkout(&base, "file:///nonexistent", owner, repo, sha, &token).await;
            assert_eq!(
                got.unwrap_err(),
                WorkspaceError::Invalid(field),
                "{sha} {owner} {repo}"
            );
        }
        assert_eq!(fs::read_dir(&base).unwrap().count(), 0);
        fs::remove_dir_all(base).unwrap();
    }

    /// A server that demands credentials: git gets the token from the helper and
    /// sends it, the server refuses it, and the failed fetch's error never holds it.
    #[tokio::test]
    async fn the_helper_sends_the_token_and_errors_never_show_it() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let mut s = stream.unwrap();
                let mut head = Vec::new();
                let mut buf = [0; 1024];
                while !head.ends_with(b"\r\n\r\n") {
                    match s.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => head.extend_from_slice(&buf[..n]),
                    }
                }
                let _ = tx.send(String::from_utf8_lossy(&head).into_owned());
                let _ = s.write_all(
                    b"HTTP/1.1 401 Unauthorized\r\nwww-authenticate: Basic realm=\"x\"\r\n\
                      content-length: 0\r\nconnection: close\r\n\r\n",
                );
            }
        });
        let base = scratch("auth");
        // Distinctive but low-entropy, so secret scanners (ours included) do not flag the test.
        let token = Secret::new("tokentokentokentoken");
        let err = checkout(&base, &format!("http://{addr}"), "o", "r", SHA, &token)
            .await
            .unwrap_err();
        assert!(
            matches!(
                err,
                WorkspaceError::Git {
                    step: "fetch",
                    source: ProcError::Exit { .. }
                }
            ),
            "{err:?}"
        );
        assert!(!err.to_string().contains(token.expose()), "{err}");
        assert!(!format!("{err:?}").contains(token.expose()), "{err:?}");
        // base64("x-access-token:tokentokentokentoken")
        let auth = "Basic eC1hY2Nlc3MtdG9rZW46dG9rZW50b2tlbnRva2VudG9rZW4=";
        let heads: Vec<String> = rx.try_iter().collect();
        assert!(
            heads.iter().any(|h| h.contains(auth)),
            "the helper's credentials never arrived: {heads:?}"
        );
        let left = fs::read_dir(&base)
            .unwrap()
            .next()
            .map(|e| e.unwrap().path());
        assert!(
            gone(&left.unwrap_or_default()).await,
            "a failed checkout left files"
        );
        assert_eq!(fs::read_dir(&base).unwrap().count(), 0);
        fs::remove_dir_all(base).unwrap();
    }

    /// A stored (uncompressed) zip of `files`, as a module proxy serves a module.
    fn zip(files: &[(&str, &str)]) -> Vec<u8> {
        fn crc32(data: &[u8]) -> u32 {
            let step = |c: u32| {
                if c & 1 == 1 {
                    (c >> 1) ^ 0xEDB8_8320
                } else {
                    c >> 1
                }
            };
            !data.iter().fold(!0, |crc, &b| {
                (0..8).fold(crc ^ u32::from(b), |c, _| step(c))
            })
        }
        let (mut out, mut central) = (Vec::new(), Vec::new());
        for (name, data) in files {
            let sizes = [crc32(data.as_bytes()), data.len() as u32, data.len() as u32];
            let sizes: Vec<u8> = sizes.iter().flat_map(|v| v.to_le_bytes()).collect();
            let name_len = (name.len() as u16).to_le_bytes();
            let offset = (out.len() as u32).to_le_bytes();
            // Version 20, no flags, stored, a zero time and date.
            out.extend(0x0403_4b50u32.to_le_bytes());
            out.extend([20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            out.extend(&sizes);
            out.extend(name_len);
            out.extend([0, 0]);
            out.extend(name.as_bytes());
            out.extend(data.as_bytes());
            central.extend(0x0201_4b50u32.to_le_bytes());
            central.extend([20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            central.extend(&sizes);
            central.extend(name_len);
            central.extend([0; 12]);
            central.extend(offset);
            central.extend(name.as_bytes());
        }
        let count = (files.len() as u16).to_le_bytes();
        let (size, start) = (central.len() as u32, out.len() as u32);
        out.extend(central);
        out.extend(0x0605_4b50u32.to_le_bytes());
        out.extend([0, 0, 0, 0]);
        out.extend(count);
        out.extend(count);
        out.extend(size.to_le_bytes());
        out.extend(start.to_le_bytes());
        out.extend([0, 0]);
        out
    }

    /// `go mod download` fills the checkout's own module cache, read-only as Go writes
    /// it, and dropping the checkout removes all of it. The module comes from a proxy on
    /// disk, named in a `GOENV` file since `GOPROXY` itself is dropped.
    #[tokio::test]
    async fn a_checkout_holding_downloaded_modules_is_removed() {
        let tmp = scratch("modcache");
        let at = tmp.join("proxy/example.com/m/@v");
        fs::create_dir_all(&at).unwrap();
        let module = "module example.com/m\n";
        fs::write(at.join("list"), "v1.0.0\n").unwrap();
        fs::write(at.join("v1.0.0.info"), r#"{"Version":"v1.0.0"}"#).unwrap();
        fs::write(at.join("v1.0.0.mod"), module).unwrap();
        let files = [
            ("example.com/m@v1.0.0/go.mod", module),
            ("example.com/m@v1.0.0/m.go", "package m\n"),
        ];
        fs::write(at.join("v1.0.0.zip"), zip(&files)).unwrap();
        let goenv = tmp.join("goenv");
        let proxy = tmp.join("proxy");
        fs::write(
            &goenv,
            format!("GOPROXY=file://{}\nGOSUMDB=off\n", proxy.display()),
        )
        .unwrap();

        let base = tmp.join("base");
        fs::create_dir(&base).unwrap();
        let co = super::scratch(&base).unwrap();
        let app = "module example.com/app\n\ngo 1.21\n\nrequire example.com/m v1.0.0\n";
        fs::write(co.root().join("go.mod"), app).unwrap();
        let go_env = [
            ("GOMODCACHE", co.modcache().into_os_string()),
            ("GOENV", goenv.into_os_string()),
            ("GOTOOLCHAIN", "local".into()),
            ("HOME", tmp.clone().into_os_string()),
        ];
        let go_env: Vec<(String, OsString)> = go_env.map(|(k, v)| (k.to_owned(), v)).into();
        go_mod_download(&co.root(), &go_env, "acme", &Secret::new("t"))
            .await
            .unwrap();
        let module = co.modcache().join("example.com/m@v1.0.0");
        assert!(
            module.join("m.go").is_file(),
            "nothing in the checkout's cache"
        );
        let mode = fs::metadata(&module).unwrap().permissions().mode();
        assert_eq!(mode & 0o200, 0, "go left the module writable");

        let dir = co.dir().to_owned();
        drop(co);
        assert!(gone(&dir).await, "drop left the module cache behind");
        fs::remove_dir_all(tmp).unwrap();
    }

    #[test]
    fn mod_download_env_opens_the_proxy_and_adds_git() {
        let go_env: Vec<(String, OsString)> = vec![
            ("GOTOOLCHAIN".to_owned(), "local".into()),
            ("GOPROXY".to_owned(), "off".into()),
            ("GOFLAGS".to_owned(), "".into()),
        ];
        let env = mod_download_env(&go_env, "acme", &Secret::new("tok"));
        let get = |k: &str| env.iter().find(|(key, _)| key == k).map(|(_, v)| v.clone());
        assert_eq!(get("GOPROXY"), None);
        assert_eq!(get("GOTOOLCHAIN"), Some("local".into()));
        assert_eq!(get("GOFLAGS"), Some("".into()));
        assert_eq!(get("GOPRIVATE"), Some("github.com/acme".into()));
        assert_eq!(
            get("GIT_CONFIG_KEY_0"),
            Some("credential.https://github.com.helper".into())
        );
        assert_eq!(get("DIFFAGENT_GIT_TOKEN"), Some("tok".into()));
    }

    #[test]
    fn read_stays_in_the_tree() {
        let tmp = scratch("read");
        let root = tmp.join("root");
        fs::create_dir_all(root.join("sub")).unwrap();
        fs::write(root.join("a.txt"), "hello").unwrap();
        fs::write(root.join("sub/b.txt"), "b").unwrap();
        fs::write(root.join("ten"), "0123456789").unwrap();
        fs::write(root.join("big"), "01234567890").unwrap();
        fs::write(root.join("bin"), [0xff, 0xfe]).unwrap();
        fs::write(tmp.join("secret"), "outside").unwrap();
        symlink(tmp.join("secret"), root.join("out")).unwrap();
        symlink("a.txt", root.join("in")).unwrap();
        let status = Command::new("mkfifo")
            .arg(root.join("fifo"))
            .status()
            .unwrap();
        assert!(status.success());
        symlink(&root, tmp.join("link")).unwrap();

        let abs = root.join("a.txt");
        let cases: [(&Path, &str, Option<&str>); 17] = [
            (&root, "a.txt", Some("hello")),
            (&root, "./sub/b.txt", Some("b")),
            (&root, "in", Some("hello")),
            (&root, "ten", Some("0123456789")),
            (&tmp.join("link"), "a.txt", Some("hello")),
            (&root, "big", None),
            (&root, "../secret", None),
            (&root, "sub/../a.txt", None),
            (&root, abs.to_str().unwrap(), None),
            (&root, "out", None),
            (&root, "fifo", None),
            (&root, "bin", None),
            (&root, "sub", None),
            (&root, "", None),
            (&root, "missing", None),
            (&root, "a\0b", None),
            (&tmp.join("nope"), "a.txt", None),
        ];
        for (root, rel, want) in cases {
            assert_eq!(read(root, rel, 10).as_deref(), want, "{rel:?}");
        }
        fs::remove_dir_all(tmp).unwrap();
    }

    #[test]
    fn sweep_removes_only_workspace_dirs() {
        let base = scratch("sweep");
        fs::create_dir_all(base.join("diffagent-ws-1-0/src/.git")).unwrap();
        fs::write(base.join("diffagent-ws-1-0/src/a"), "x").unwrap();
        // Read-only, as Go writes its module cache.
        let read_only = fs::Permissions::from_mode(0o555);
        fs::set_permissions(base.join("diffagent-ws-1-0/src"), read_only).unwrap();
        fs::create_dir(base.join("diffagent-ws-7-3")).unwrap();
        fs::create_dir(base.join("keep")).unwrap();
        fs::write(base.join("keep/a"), "x").unwrap();
        fs::write(base.join("diffagent-ws-file"), "x").unwrap();
        symlink(base.join("keep"), base.join("diffagent-ws-link")).unwrap();

        assert_eq!(sweep(&base), 2);
        let mut left: Vec<String> = fs::read_dir(&base)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        left.sort();
        assert_eq!(left, ["diffagent-ws-file", "diffagent-ws-link", "keep"]);
        assert!(base.join("keep/a").exists());
        assert_eq!(sweep(&base.join("missing")), 0);
        fs::remove_dir_all(base).unwrap();
    }
}
