//! The webhook front door: verify, parse, filter, enqueue. It makes no GitHub or model
//! call, so every delivery is answered well inside GitHub's 10 s; the worker does the rest.
//! [`serve`] is the listener: every phase of a connection before the handler has a
//! deadline, and the connections open at once have a cap, since anyone who can reach the
//! port can open one and the worker shares the process. With `WEBHOOK_ALLOWED_IPS` set,
//! a peer outside it is closed before it takes one of those connections.

use std::future::Future;
use std::io;
use std::net::IpAddr;
use std::pin::pin;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, FromRequest, Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::routing::{get, post};
use hmac::{Hmac, KeyInit, Mac};
use hyper_util::rt::{TokioExecutor, TokioIo, TokioTimer};
use hyper_util::server::conn::auto::Builder;
use hyper_util::service::TowerToHyperService;
use ipnet::IpNet;
use sha2::Sha256;
use tokio::net::TcpListener;
use tokio::sync::{Semaphore, watch};
use tracing::{error, info, warn};

use crate::config::{Config, Secret};
use crate::job;
use crate::state::{NewJob, Store};

/// Caps a delivery's size to bound memory and reject abuse; a larger body gets 413.
const MAX_BODY: usize = 5 << 20;

/// Bounds reading a request's head, from a connection's open or the end of its last
/// response: a peer that sends nothing, or a partial head, is cut off instead of
/// holding a socket, a task and a head buffer of up to about 400 KiB.
const HEAD_TIMEOUT: Duration = Duration::from_secs(10);

/// The connections open at once. One past it is accepted and closed at once, so a
/// stalled peer cannot run the process out of file descriptors or memory. GitHub opens
/// a few at a time; 256 leaves most of a 1024 descriptor limit to the worker's git,
/// go and API calls.
const MAX_CONNECTIONS: usize = 256;

/// The most a delivery's body may take to arrive. The signature can only be checked on
/// the whole body: without a deadline, a client trickling bytes holds a task, a growing
/// buffer and a permit, unauthenticated.
const READ_TIMEOUT: Duration = Duration::from_secs(10);

/// The least a body gets, however small. The permit a body holds is shared with
/// GitHub's deliveries, so a stalled unsigned one must give it back soon; GitHub sends
/// a small delivery in milliseconds.
const READ_FLOOR: Duration = Duration::from_secs(2);

/// How long a body of `len` declared bytes may take: [`READ_FLOOR`] plus 1 µs a byte
/// (a sender keeping up 1 MB/s), at most [`READ_TIMEOUT`].
fn read_deadline(len: u64) -> Duration {
    (READ_FLOOR + Duration::from_micros(len)).min(READ_TIMEOUT)
}

/// The deliveries handled at once. Each buffers up to [`MAX_BODY`] before its signature
/// can be checked, so this bounds unauthenticated memory at 64 × 5 MiB = 320 MiB; a
/// delivery past it gets 503 unread. GitHub never redelivers on its own, so a 503 is lost
/// until someone redelivers it (by hand or through the API). Real deliveries are
/// small and answered in milliseconds, far under 64 at once.
const MAX_IN_FLIGHT: usize = 64;

/// Bounds the insert, so a slow database still answers inside GitHub's 10 s. The 500
/// records nothing, so the delivery can be redelivered.
const ENQUEUE_TIMEOUT: Duration = Duration::from_secs(3);

/// What every request shares.
struct Hook {
    store: Arc<dyn Store>,
    secret: Secret,
    /// `owner/repo` entries; empty allows every repo.
    allowlist: Vec<String>,
    /// `ENABLE_FEEDBACK`: off drops Thread replies; commands still work.
    feedback: bool,
    /// One permit per delivery handled, taken before its body is read.
    in_flight: Arc<Semaphore>,
}

impl Hook {
    /// Whether `repo` (`owner/repo`) may be reviewed. GitHub names are case-insensitive.
    fn allowed(&self, repo: &str) -> bool {
        self.allowlist.is_empty() || self.allowlist.iter().any(|r| r.eq_ignore_ascii_case(repo))
    }
}

/// Serves `app` on `listener` until `shutdown` resolves, then stops accepting, lets each
/// open connection finish the request it is on, and returns once all have closed.
/// Connections are HTTP/1, each head read within [`HEAD_TIMEOUT`], and at most
/// [`MAX_CONNECTIONS`] are open at once. A peer outside `allowed`, when it names any
/// range, is closed before it takes a slot or is read.
pub async fn serve(
    listener: TcpListener,
    app: Router,
    allowed: Vec<IpNet>,
    shutdown: impl Future<Output = ()>,
) {
    let open = Arc::new(Semaphore::new(MAX_CONNECTIONS));
    serve_with(listener, app, allowed, shutdown, open).await;
}

/// [`serve`] with `open` holding one permit per connection open.
async fn serve_with(
    listener: TcpListener,
    app: Router,
    allowed: Vec<IpNet>,
    shutdown: impl Future<Output = ()>,
    open: Arc<Semaphore>,
) {
    // HTTP/1 only, so no protocol sniff (which has no deadline) runs before the head timer.
    let mut builder = Builder::new(TokioExecutor::new()).http1_only();
    builder
        .http1()
        .timer(TokioTimer::new())
        .header_read_timeout(HEAD_TIMEOUT);
    let builder = Arc::new(builder);
    let (stop, stopping) = watch::channel(false);
    let mut shutdown = pin!(shutdown);
    loop {
        let (stream, peer) = tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok((stream, peer)) => (stream, peer.ip()),
                Err(e) => {
                    accept_failed(e).await;
                    continue;
                }
            },
            () = &mut shutdown => break,
        };
        if !admits(&allowed, peer) {
            warn!(peer = %peer, "peer outside WEBHOOK_ALLOWED_IPS; closing its connection");
            continue;
        }
        let Ok(permit) = Arc::clone(&open).try_acquire_owned() else {
            warn!("too many connections open; closing a new one");
            continue;
        };
        let (builder, app, mut stopping) = (Arc::clone(&builder), app.clone(), stopping.clone());
        tokio::spawn(async move {
            let _permit = permit;
            let service = TowerToHyperService::new(app);
            let mut conn = pin!(builder.serve_connection(TokioIo::new(stream), service));
            tokio::select! {
                _ = conn.as_mut() => return,
                _ = stopping.wait_for(|&stop| stop) => conn.as_mut().graceful_shutdown(),
            }
            let _ = conn.await;
        });
    }
    drop(listener);
    // Each connection's task holds a receiver until it ends.
    drop(stopping);
    stop.send_replace(true);
    stop.closed().await;
}

/// Whether `peer` may connect: an empty `allowed` admits every peer. An IPv4-mapped IPv6
/// peer (`::ffff:a.b.c.d`) also matches as its IPv4 address, since a dual-stack socket
/// reports an IPv4 client that way.
fn admits(allowed: &[IpNet], peer: IpAddr) -> bool {
    let v4 = peer.to_canonical();
    allowed.is_empty() || allowed.iter().any(|n| n.contains(&peer) || n.contains(&v4))
}

/// A failed accept: one a peer reset first is skipped; anything else (out of file
/// descriptors, say) is logged and retried after 1 s, since retrying at once would spin.
async fn accept_failed(e: io::Error) {
    use io::ErrorKind::{ConnectionAborted, ConnectionRefused, ConnectionReset};
    if matches!(
        e.kind(),
        ConnectionAborted | ConnectionRefused | ConnectionReset
    ) {
        return;
    }
    error!(err = %e, "accept failed");
    tokio::time::sleep(Duration::from_secs(1)).await;
}

/// `POST /webhook`, and `GET /healthz`, which answers "ok" and checks nothing.
pub fn router(store: Arc<dyn Store>, cfg: &Config) -> Router {
    app(store, cfg, Arc::new(Semaphore::new(MAX_IN_FLIGHT)))
}

/// [`router`] with `in_flight` bounding the deliveries handled at once.
fn app(store: Arc<dyn Store>, cfg: &Config, in_flight: Arc<Semaphore>) -> Router {
    let hook = Hook {
        store,
        secret: cfg.webhook_secret.clone(),
        allowlist: cfg.repo_allowlist.clone(),
        feedback: cfg.enable_feedback,
        in_flight,
    };
    Router::new()
        .route("/webhook", post(webhook))
        .route("/healthz", get(|| async { "ok" }))
        .layer(DefaultBodyLimit::max(MAX_BODY))
        .with_state(Arc::new(hook))
}

/// One delivery, in order: 411 no `Content-Length` and 413 one over [`MAX_BODY`] (both
/// unread, before a permit is taken), 503 too many at once (unread), 408 not sent within
/// [`read_deadline`], 401 not signed, 400 unreadable, 204 nothing to do or not allowed,
/// then 202 queued, 200 already queued, or 500 when the insert failed or timed out, so it
/// can be redelivered.
async fn webhook(State(hook): State<Arc<Hook>>, headers: HeaderMap, req: Request) -> StatusCode {
    let event = header(&headers, "x-github-event").unwrap_or_default();
    let delivery = header(&headers, "x-github-delivery").unwrap_or_default();
    // GitHub always sends a length; the body's deadline is sized from it.
    let Some(len) = header(&headers, "content-length").and_then(|v| v.parse::<u64>().ok()) else {
        warn!(event, delivery, "webhook without a content length");
        return StatusCode::LENGTH_REQUIRED;
    };
    if len > MAX_BODY as u64 {
        warn!(event, delivery, len, "webhook body over the limit");
        return StatusCode::PAYLOAD_TOO_LARGE;
    }
    let Ok(_slot) = hook.in_flight.try_acquire() else {
        warn!(event, delivery, "too many deliveries in flight");
        return StatusCode::SERVICE_UNAVAILABLE;
    };
    let deadline = read_deadline(len);
    let body = match tokio::time::timeout(deadline, Bytes::from_request(req, &())).await {
        Ok(Ok(body)) => body,
        Ok(Err(rejected)) => return rejected.status(),
        Err(_) => {
            warn!(event, delivery, "webhook body not sent in time");
            return StatusCode::REQUEST_TIMEOUT;
        }
    };
    if !verified(&hook.secret, &headers, &body) {
        warn!(event, delivery, "webhook signature rejected");
        return StatusCode::UNAUTHORIZED;
    }
    // The delivery id is the dedupe key; GitHub always sends one.
    if delivery.is_empty() {
        warn!(event, "webhook without a delivery id");
        return StatusCode::BAD_REQUEST;
    }
    let (job, delay) = match job::from_event(event, &body) {
        Ok(Some(found)) => found,
        Ok(None) => {
            info!(event, delivery, "webhook not actionable; acked");
            return StatusCode::NO_CONTENT;
        }
        Err(e) => {
            warn!(event, delivery, err = %e, "webhook parse failed");
            return StatusCode::BAD_REQUEST;
        }
    };
    let repo = format!("{}/{}", job.owner, job.repo);
    if !hook.allowed(&repo) {
        info!(repo = %repo, delivery, "repo not in allowlist; skipping");
        return StatusCode::NO_CONTENT;
    }
    if !hook.feedback && matches!(job.kind, job::Kind::Reply { .. }) {
        info!(repo = %repo, delivery, pr = job.pr, "feedback disabled; reply skipped");
        return StatusCode::NO_CONTENT;
    }
    let payload = match serde_json::to_string(&job) {
        Ok(payload) => payload,
        Err(e) => {
            error!(delivery, err = %e, "encode job failed");
            return StatusCode::INTERNAL_SERVER_ERROR;
        }
    };
    let new = NewJob {
        delivery: delivery.to_owned(),
        slot: job.slot(),
        payload,
        delay,
    };
    match tokio::time::timeout(ENQUEUE_TIMEOUT, hook.store.enqueue(&new)).await {
        Ok(Ok(true)) => {
            // A reply's body is a person's text, kept out of the log.
            let kind = match &job.kind {
                job::Kind::Reply { .. } => "reply".to_owned(),
                k => format!("{k:?}"),
            };
            info!(event, delivery, repo = %repo, pr = job.pr, slot = %new.slot, kind, "job enqueued");
            StatusCode::ACCEPTED
        }
        Ok(Ok(false)) => {
            info!(delivery, repo = %repo, pr = job.pr, "delivery already queued");
            StatusCode::OK
        }
        Ok(Err(e)) => {
            error!(delivery, err = %e, "enqueue failed");
            StatusCode::INTERNAL_SERVER_ERROR
        }
        Err(_) => {
            error!(delivery, "enqueue timed out");
            StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}

fn header<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name)?.to_str().ok()
}

/// Whether `X-Hub-Signature-256` is `sha256=<hex>` of the raw body under the secret;
/// `verify_slice` compares in constant time. The SHA-1 `X-Hub-Signature` is never read,
/// and an empty secret verifies nothing.
fn verified(secret: &Secret, headers: &HeaderMap, body: &[u8]) -> bool {
    let Some(tag) = header(headers, "x-hub-signature-256")
        .and_then(|v| v.strip_prefix("sha256="))
        .and_then(unhex)
    else {
        return false;
    };
    if secret.is_empty() {
        return false;
    }
    let Ok(mut mac) = Hmac::<Sha256>::new_from_slice(secret.expose().as_bytes()) else {
        return false;
    };
    mac.update(body);
    mac.verify_slice(&tag).is_ok()
}

/// Decodes hex in either case; `None` for anything else, an odd length included.
fn unhex(s: &str) -> Option<Vec<u8>> {
    let (pairs, odd) = s.as_bytes().as_chunks::<2>();
    if !odd.is_empty() {
        return None;
    }
    let digit = |c: u8| char::from(c).to_digit(16);
    pairs
        .iter()
        .map(|&[hi, lo]| u8::try_from(digit(hi)? * 16 + digit(lo)?).ok())
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use std::fmt::Write as _;
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::sync::Mutex;

    use tokio::net::TcpListener;
    use tokio::sync::oneshot::Receiver;

    use super::*;
    use crate::state::contract::Faulty;
    use crate::state::{JobEnd, MemoryStore};

    const SECRET: &str = "It's a Secret to Everybody";

    fn config(allowlist: &str) -> Config {
        Config::from_lookup(|key| match key {
            "GITHUB_WEBHOOK_SECRET" => Some(SECRET.to_owned()),
            "REPO_ALLOWLIST" => Some(allowlist.to_owned()),
            _ => None,
        })
        .expect("config")
    }

    /// An actionable delivery: `pull_request` opened on acme/shop.
    fn opened() -> Vec<u8> {
        std::fs::read("tests/data/webhook/pull_request.json").expect("fixture")
    }

    pub(crate) fn sign(secret: &str, body: &[u8]) -> String {
        let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).expect("key");
        mac.update(body);
        mac.finalize()
            .into_bytes()
            .iter()
            .fold("sha256=".to_owned(), |mut s, b| {
                let _ = write!(s, "{b:02x}");
                s
            })
    }

    /// Serves `app` on a free 127.0.0.1 port through [`super::serve`]; its base URL.
    pub(crate) async fn serve(app: Router) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(super::serve(
            listener,
            app,
            Vec::new(),
            std::future::pending(),
        ));
        format!("http://{addr}")
    }

    /// Connects to `addr`, sends `sent` and reads until the server closes: what it
    /// answered, or an error once `wait` passes with the connection still open.
    fn until_closed(addr: &str, sent: &[u8], wait: Duration) -> std::io::Result<Vec<u8>> {
        let mut conn = TcpStream::connect(addr)?;
        conn.set_read_timeout(Some(wait))?;
        conn.write_all(sent)?;
        let mut answer = Vec::new();
        conn.read_to_end(&mut answer)?;
        Ok(answer)
    }

    /// A delivery's head declaring 100 bytes, then one byte of its body.
    const STALLED: &[u8] = b"POST /webhook HTTP/1.1\r\nhost: x\r\ncontent-length: 100\r\n\r\n{";

    /// Sends `sent` to `addr` from a plain thread; the answer's first 12 bytes
    /// (`HTTP/1.1 NNN`), or an error once `wait` passes without them.
    fn status(
        addr: &str,
        sent: impl AsRef<[u8]> + Send + 'static,
        wait: Duration,
    ) -> Receiver<std::io::Result<[u8; 12]>> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let addr = addr.trim_start_matches("http://").to_owned();
        std::thread::spawn(move || {
            let read = || {
                let mut conn = TcpStream::connect(&addr)?;
                conn.set_read_timeout(Some(wait))?;
                conn.write_all(sent.as_ref())?;
                let mut status = [0; 12];
                conn.read_exact(&mut status)?;
                Ok(status)
            };
            tx.send(read())
        });
        rx
    }

    /// Waits up to 5 s for `slots` to have `free` permits.
    async fn wait_for(slots: &Semaphore, free: usize) {
        for _ in 0..500 {
            if slots.available_permits() == free {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("{} slots free, not {free}", slots.available_permits());
    }

    fn client() -> reqwest::Client {
        reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client")
    }

    /// POSTs one delivery with `signature` as its one signature header; the status.
    pub(crate) async fn deliver(
        base: &str,
        delivery: &str,
        event: &str,
        body: Vec<u8>,
        signature: Option<(&str, String)>,
    ) -> u16 {
        let mut req = client()
            .post(format!("{base}/webhook"))
            .header("x-github-delivery", delivery)
            .header("x-github-event", event);
        if let Some((name, value)) = signature {
            req = req.header(name, value);
        }
        req.body(body).send().await.expect("send").status().as_u16()
    }

    /// A correctly signed `pull_request` delivery.
    async fn post_signed(base: &str, delivery: &str, body: Vec<u8>) -> u16 {
        let signature = Some(("x-hub-signature-256", sign(SECRET, &body)));
        deliver(base, delivery, "pull_request", body, signature).await
    }

    /// How many jobs `store` holds: each is claimed and finished until none is left.
    async fn drain(store: &dyn Store) -> usize {
        let mut n = 0;
        while let Some(c) = store
            .claim(Duration::from_secs(60), false)
            .await
            .expect("claim")
        {
            store
                .finish(c.id, c.attempt, JobEnd::Done)
                .await
                .expect("finish");
            n += 1;
        }
        n
    }

    #[tokio::test]
    async fn responses() {
        let store = Arc::new(MemoryStore::default());
        let base = serve(router(store.clone(), &config(""))).await;
        let pr = opened();
        let ping = br#"{"zen":"Keep it logically awesome.","hook_id":1}"#.to_vec();
        let sha256 = |secret: &str, body: &[u8]| Some(("x-hub-signature-256", sign(secret, body)));
        let cases = [
            (
                "good signature",
                "pull_request",
                pr.clone(),
                sha256(SECRET, &pr),
                202,
            ),
            (
                "wrong secret",
                "pull_request",
                pr.clone(),
                sha256("guess", &pr),
                401,
            ),
            (
                "another body's",
                "pull_request",
                pr.clone(),
                sha256(SECRET, b"{}"),
                401,
            ),
            ("missing", "pull_request", pr.clone(), None, 401),
            (
                "bare hex",
                "pull_request",
                pr.clone(),
                Some((
                    "x-hub-signature-256",
                    sign(SECRET, &pr).replace("sha256=", ""),
                )),
                401,
            ),
            // The SHA-1 header alone: never read, so its value does not matter.
            (
                "SHA-1 only",
                "pull_request",
                pr.clone(),
                Some(("x-hub-signature", format!("sha1={}", "ab".repeat(20)))),
                401,
            ),
            (
                "unparseable",
                "pull_request",
                b"{".to_vec(),
                sha256(SECRET, b"{"),
                400,
            ),
            (
                "not actionable",
                "ping",
                ping.clone(),
                sha256(SECRET, &ping),
                204,
            ),
        ];
        for (i, (name, event, body, signature, want)) in cases.into_iter().enumerate() {
            let got = deliver(&base, &format!("d-{i}"), event, body, signature).await;
            assert_eq!(got, want, "{name}");
        }
        assert_eq!(drain(&*store).await, 1, "only the good delivery is queued");

        let health = client()
            .get(format!("{base}/healthz"))
            .send()
            .await
            .expect("healthz");
        assert_eq!(health.status().as_u16(), 200);
        assert_eq!(health.text().await.expect("body"), "ok");
    }

    #[tokio::test]
    async fn body_limit() {
        let base = serve(router(Arc::new(MemoryStore::default()), &config(""))).await;
        // JSON allows trailing whitespace, so a padded payload is still a valid delivery.
        let cases = [(MAX_BODY, 202), (MAX_BODY + 1, 413)];
        for (i, (len, want)) in cases.into_iter().enumerate() {
            let mut body = opened();
            body.resize(len, b' ');
            let got = post_signed(&base, &format!("d-{i}"), body).await;
            assert_eq!(got, want, "{len} bytes");
        }
    }

    /// A body that stops coming gets 408 at its read deadline, `READ_FLOOR` for the 100
    /// bytes `STALLED` declares, so the test runs on real time in about 2 s. A paused
    /// clock jumps to the next timer whenever the runtime has nothing ready, and the head
    /// timer is armed from the accept: had the client thread not written its head yet,
    /// the jump would close the connection with no answer. The client is a plain thread
    /// on a blocking socket with a real-time guard.
    #[tokio::test]
    async fn a_stalled_body_gets_408() {
        let base = serve(router(Arc::new(MemoryStore::default()), &config(""))).await;
        let start = std::time::Instant::now();
        let rx = status(&base, STALLED, Duration::from_secs(60));
        let status = rx.await.expect("client").expect("an answer within 60 s");
        assert_eq!(&status, b"HTTP/1.1 408");
        let took = start.elapsed();
        assert!(
            took >= READ_FLOOR,
            "cut off after {took:?}, before the deadline"
        );
        assert!(took < Duration::from_secs(60));
    }

    /// A connection that sends nothing, or a head with no blank line, is closed at the
    /// head deadline, on the paused clock: the head timer is the only one armed, so a
    /// jump of the clock lands on it whether or not the client has written yet. The
    /// client is a plain thread (`spawn_blocking` would stop the clock).
    #[tokio::test(start_paused = true)]
    async fn a_stalled_head_is_cut_off() {
        let base = serve(router(Arc::new(MemoryStore::default()), &config(""))).await;
        let addr = base.trim_start_matches("http://").to_owned();
        for sent in [&b""[..], b"POST /webhook HTTP/1.1\r\nhost: x\r\n"] {
            let start = tokio::time::Instant::now();
            let (tx, rx) = tokio::sync::oneshot::channel();
            let addr = addr.clone();
            std::thread::spawn(move || tx.send(until_closed(&addr, sent, Duration::from_secs(30))));
            let got = rx.await.expect("client");
            assert!(got.is_ok(), "{sent:?} still open: {got:?}");
            assert!(start.elapsed() >= HEAD_TIMEOUT, "{sent:?} closed early");
        }
    }

    /// Past the cap a connection is accepted and closed before anything is read, and
    /// one that closes frees its slot.
    #[tokio::test]
    async fn a_connection_past_the_cap_is_closed_at_once() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr").to_string();
        let open = Arc::new(Semaphore::new(2));
        let app = router(Arc::new(MemoryStore::default()), &config(""));
        let served = serve_with(
            listener,
            app,
            Vec::new(),
            std::future::pending(),
            open.clone(),
        );
        tokio::spawn(served);
        let held: Vec<TcpStream> = (0..2)
            .map(|_| TcpStream::connect(&addr).expect("connect"))
            .collect();
        wait_for(&open, 0).await;

        let (tx, rx) = tokio::sync::oneshot::channel();
        let third = addr.clone();
        std::thread::spawn(move || tx.send(until_closed(&third, b"", Duration::from_secs(5))));
        let got = rx.await.expect("client");
        assert_eq!(got.expect("closed at once"), b"");

        drop(held);
        wait_for(&open, 2).await;
        let health = client()
            .get(format!("http://{addr}/healthz"))
            .send()
            .await
            .expect("healthz");
        assert_eq!(health.status().as_u16(), 200);
    }

    /// An empty list admits every peer; a mapped IPv6 peer matches IPv4 ranges, and
    /// `::/0` too.
    #[test]
    fn admits_table() {
        let hooks = ["192.30.252.0/22", "2a0a:a440::/29"];
        // (ranges, peer, admitted)
        let cases: [(&[&str], &str, bool); 17] = [
            (&[], "203.0.113.9", true),
            (&[], "::1", true),
            (&hooks, "192.30.252.1", true),
            (&hooks, "192.30.255.255", true),
            (&hooks, "192.31.0.1", false),
            (&hooks, "2a0a:a447:ffff::1", true),
            (&hooks, "2a0a:a448::1", false),
            (&hooks, "::ffff:192.30.252.7", true),
            (&hooks, "::ffff:10.0.0.1", false),
            (&["0.0.0.0/0"], "8.8.8.8", true),
            (&["0.0.0.0/0"], "2001:db8::1", false),
            (&["::/0"], "2001:db8::1", true),
            (&["::/0"], "::ffff:8.8.8.8", true),
            (&["10.1.2.3/32"], "10.1.2.3", true),
            (&["10.1.2.3/32"], "10.1.2.4", false),
            (&["2001:db8::1/128"], "2001:db8::1", true),
            (&["2001:db8::1/128"], "2001:db8::2", false),
        ];
        for (ranges, peer, want) in cases {
            let nets: Vec<IpNet> = ranges.iter().map(|n| n.parse().unwrap()).collect();
            let ip: IpAddr = peer.parse().unwrap();
            assert_eq!(admits(&nets, ip), want, "{peer} in {ranges:?}");
        }
    }

    /// A peer outside the ranges is closed before anything is read, and holds no slot
    /// while its socket stays open; a peer inside them is served.
    #[tokio::test]
    async fn a_peer_outside_the_allowed_ranges_is_closed_at_once() {
        for (ranges, admitted) in [("10.0.0.0/8,::1/128", false), ("127.0.0.0/8", true)] {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
            let addr = listener.local_addr().expect("addr").to_string();
            let open = Arc::new(Semaphore::new(2));
            let app = router(Arc::new(MemoryStore::default()), &config(""));
            let allowed = ranges.split(',').map(|n| n.parse().unwrap()).collect();
            let served = serve_with(listener, app, allowed, std::future::pending(), open.clone());
            tokio::spawn(served);
            let (tx, rx) = tokio::sync::oneshot::channel();
            let peer = addr.clone();
            std::thread::spawn(move || {
                let mut conn = TcpStream::connect(&peer).expect("connect");
                conn.set_read_timeout(Some(Duration::from_secs(5)))
                    .expect("timeout");
                let mut first = [0; 1];
                let read = conn.read(&mut first);
                // Held open until the slots are counted.
                tx.send((read.map(|n| n == 0), conn)).ok();
            });
            if admitted {
                wait_for(&open, 1).await;
                let health = client()
                    .get(format!("http://{addr}/healthz"))
                    .send()
                    .await
                    .expect("healthz");
                assert_eq!(health.status().as_u16(), 200);
                continue;
            }
            let (closed, held) = rx.await.expect("client");
            assert!(
                closed.expect("closed, not timed out"),
                "{ranges}: closed at once"
            );
            assert_eq!(open.available_permits(), 2, "{ranges}: no slot held");
            drop(held);
        }
    }

    /// Shutdown stops accepting, closes the kept-alive connection and returns.
    #[tokio::test]
    async fn shutdown_closes_connections_and_returns() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let app = router(Arc::new(MemoryStore::default()), &config(""));
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let shutdown = async {
            let _ = stopped.await;
        };
        let served = tokio::spawn(super::serve(listener, app, Vec::new(), shutdown));
        // The client keeps its connection alive after the answer.
        let client = client();
        let health = format!("http://{addr}/healthz");
        let got = client.get(&health).send().await.expect("healthz");
        assert_eq!(got.status().as_u16(), 200);
        stop.send(()).expect("serving");
        let served = tokio::time::timeout(Duration::from_secs(5), served).await;
        served.expect("serve returned").expect("serve task");
        assert!(client.get(&health).send().await.is_err(), "still accepting");
    }

    /// With every slot taken, a delivery gets 503 at once, its body never read: a body
    /// still coming would otherwise wait out the 10 s read deadline. A freed slot takes
    /// the next delivery.
    #[tokio::test]
    async fn a_delivery_past_the_limit_gets_503_unread() {
        let store = Arc::new(MemoryStore::default());
        let in_flight = Arc::new(Semaphore::new(1));
        let base = serve(app(store.clone(), &config(""), in_flight.clone())).await;
        let taken = in_flight.try_acquire().expect("a free slot");
        let rx = status(&base, STALLED, Duration::from_secs(5));
        let status = rx.await.expect("client").expect("an answer within 5 s");
        assert_eq!(&status, b"HTTP/1.1 503");
        assert_eq!(post_signed(&base, "d-1", opened()).await, 503);
        drop(taken);
        assert_eq!(post_signed(&base, "d-1", opened()).await, 202);
        assert_eq!(drain(&*store).await, 1);
    }

    /// The length is checked before a permit is taken: with none free, a request with
    /// no length gets 411 and one declaring too much 413, never 503.
    #[tokio::test]
    async fn the_length_is_checked_before_a_permit() {
        let store = Arc::new(MemoryStore::default());
        let base = serve(app(store, &config(""), Arc::new(Semaphore::new(0)))).await;
        let too_large = format!("content-length: {}\r\n", MAX_BODY + 1);
        let cases: [(&str, &[u8; 12]); 4] = [
            ("", b"HTTP/1.1 411"),
            ("transfer-encoding: chunked\r\n", b"HTTP/1.1 411"),
            (&too_large, b"HTTP/1.1 413"),
            ("content-length: 100\r\n", b"HTTP/1.1 503"),
        ];
        for (length, want) in cases {
            let head = format!("POST /webhook HTTP/1.1\r\nhost: x\r\n{length}\r\n");
            let rx = status(&base, head, Duration::from_secs(5));
            let got = rx.await.expect("client").expect("an answer within 5 s");
            assert_eq!(&got, want, "{length:?}");
        }
    }

    /// Unsigned bodies that stall hold every permit only until the short deadline their
    /// small declared length gets: a signed delivery meanwhile gets 503, and once they
    /// are cut off with 408, well before [`READ_TIMEOUT`], it is queued.
    #[tokio::test]
    async fn stalled_unsigned_bodies_give_their_permits_back_soon() {
        let store = Arc::new(MemoryStore::default());
        let in_flight = Arc::new(Semaphore::new(4));
        let base = serve(app(store.clone(), &config(""), in_flight.clone())).await;
        let start = std::time::Instant::now();
        let stalled: Vec<_> = (0..4)
            .map(|_| status(&base, STALLED, Duration::from_secs(15)))
            .collect();
        wait_for(&in_flight, 0).await;
        assert_eq!(post_signed(&base, "d-real", opened()).await, 503);
        assert_eq!(drain(&*store).await, 0);
        for rx in stalled {
            let got = rx.await.expect("client").expect("an answer within 15 s");
            assert_eq!(&got, b"HTTP/1.1 408");
        }
        let took = start.elapsed();
        assert!(took >= READ_FLOOR && took < READ_TIMEOUT / 2, "{took:?}");
        assert_eq!(post_signed(&base, "d-real", opened()).await, 202);
        assert_eq!(drain(&*store).await, 1);
    }

    #[test]
    fn read_deadline_scales_with_the_length() {
        let cases = [
            (0, Duration::from_secs(2)),
            (100, Duration::from_micros(2_000_100)),
            (MAX_BODY as u64, Duration::from_micros(7_242_880)),
            (u64::MAX, READ_TIMEOUT),
        ];
        for (len, want) in cases {
            assert_eq!(read_deadline(len), want, "{len}");
        }
    }

    #[tokio::test]
    async fn redelivery_is_200_with_one_row() {
        let store = Arc::new(MemoryStore::default());
        let base = serve(router(store.clone(), &config(""))).await;
        assert_eq!(post_signed(&base, "d-1", opened()).await, 202);
        assert_eq!(post_signed(&base, "d-1", opened()).await, 200);
        assert_eq!(drain(&*store).await, 1);
    }

    /// The first insert fails, as when Postgres is unreachable.
    #[tokio::test]
    async fn store_error_then_redelivery() {
        let faulty = Arc::new(Faulty {
            inner: Arc::new(MemoryStore::default()),
            faults: Mutex::new(vec![("enqueue", 0, 1)]),
        });
        let base = serve(router(faulty.clone(), &config(""))).await;
        assert_eq!(post_signed(&base, "d-1", opened()).await, 500);
        assert_eq!(
            post_signed(&base, "d-1", opened()).await,
            202,
            "the 500 recorded nothing"
        );
        assert_eq!(drain(&*faulty.inner).await, 1);
    }

    /// `ENABLE_FEEDBACK=false` drops Thread replies, but a command replied in a Thread
    /// still runs.
    #[tokio::test]
    async fn feedback_off_skips_replies() {
        let raw =
            std::fs::read("tests/data/webhook/pull_request_review_comment.json").expect("fixture");
        let mut forget: serde_json::Value = serde_json::from_slice(&raw).expect("JSON");
        forget["comment"]["body"] = "@diffagent forget".into();
        let forget = serde_json::to_vec(&forget).expect("encode");
        let cases = [
            ("true", raw.clone(), 202),
            ("false", raw, 204),
            ("false", forget, 202),
        ];
        for (i, (feedback, body, want)) in cases.into_iter().enumerate() {
            let cfg = Config::from_lookup(|key| match key {
                "GITHUB_WEBHOOK_SECRET" => Some(SECRET.to_owned()),
                "ENABLE_FEEDBACK" => Some(feedback.to_owned()),
                _ => None,
            })
            .expect("config");
            let store = Arc::new(MemoryStore::default());
            let base = serve(router(store.clone(), &cfg)).await;
            let signature = Some(("x-hub-signature-256", sign(SECRET, &body)));
            let event = "pull_request_review_comment";
            let got = deliver(&base, "d-1", event, body, signature).await;
            assert_eq!(got, want, "case {i}");
            assert_eq!(drain(&*store).await, usize::from(want == 202), "case {i}");
        }
    }

    #[tokio::test]
    async fn allowlist() {
        let cases = [
            ("", 202),
            ("acme/other", 204),
            ("acme/other, Acme/Shop", 202),
        ];
        for (allow, want) in cases {
            let store = Arc::new(MemoryStore::default());
            let base = serve(router(store.clone(), &config(allow))).await;
            assert_eq!(post_signed(&base, "d-1", opened()).await, want, "{allow:?}");
            assert_eq!(drain(&*store).await, usize::from(want == 202), "{allow:?}");
        }
    }
}
