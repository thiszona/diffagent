# Setup

## Install a GitHub App

Create a GitHub App in your account or organization settings, under Developer settings ›
GitHub Apps › New GitHub App. Its name sets its slug, the handle the bot answers to.

Make the webhook active, with the URL `https://<your-host>/webhook` and a random secret.
The same secret is `GITHUB_WEBHOOK_SECRET`.

Give it these repository permissions:

| Permission | Access | Why |
|---|---|---|
| Pull requests | Read and write | Read the PR and its files; post the GitHub review and its inline comments |
| Contents | Read and write | Clone the head; resolving review threads needs write |
| Issues | Read and write | The status comment, the Walkthrough and the notes on the PR |
| Checks | Read and write | Only with the [Merge gate](merge-gate.md): the `diffagent` check run |
| Metadata | Read | Required by GitHub |

Subscribe to the events Pull request, Issue comment and Pull request review comment
(`pull_request`, `issue_comment`, `pull_request_review_comment`).

After creating it, note the App ID (`GITHUB_APP_ID`) and generate a private key: the
downloaded PEM's text is `GITHUB_APP_PRIVATE_KEY`. Install the App on the repositories to
review; `REPO_ALLOWLIST` narrows them further.

## Run

The [quick start](../README.md#quick-start) has the `docker run` and `cargo run` commands.

Without `GITHUB_APP_ID` the service runs in dev mode: deliveries are verified and queued in
memory (in Postgres if `DATABASE_URL` is set), and no worker runs. With it,
`GITHUB_APP_PRIVATE_KEY` and `DATABASE_URL` are required and the worker runs. Either way
it listens on `PORT` with the [endpoints](#image-and-endpoints) below.

Outside the image, the worker runs `git`, and `gitleaks` when enabled, from `PATH`. Go
context needs the `go` toolchain and `diffagent-goctx` on `PATH` or at `GOCTX_PATH`. Build
the helper with:

```sh
go build -C goctx -o ../target/diffagent-goctx
```

## Deploy

### Image and endpoints

The image is `ghcr.io/thiszona/diffagent:<tag>`, for linux/amd64 and linux/arm64,
published for every `v*` tag; `v0.1.0` is the first. It runs `/usr/local/bin/diffagent`
as the non-root user and group 65532 and listens on port 8080:

- `POST /webhook` takes GitHub's deliveries. Before reading a body it answers 411 to a
  request without a `Content-Length` (GitHub always sends one) and 413 to one declaring
  more than 5 MiB. It then checks the `X-Hub-Signature-256` and queues each delivery
  once by its delivery id. It reads at most 64 deliveries at once and answers 503 past
  that.
- The signature can only be checked on the whole body, so a body that stalls holds one
  of those 64 until its deadline: 2 s plus 1 s per MB declared, at most 10 s. A sender
  without the secret that keeps 64 stalled bodies going makes GitHub's deliveries get
  503 meanwhile. Allow `/webhook` only from GitHub's `hooks` IP ranges (listed by
  `GET https://api.github.com/meta`) at your ingress, or put a proxy in front that
  buffers each request body whole before forwarding it.
- GitHub does not redeliver a failed delivery (a 5xx, or no answer within 10 s) on its
  own. Redeliver it from the App's Advanced settings or through the REST API, within 3
  days; GitHub documents a [scheduled workflow](https://docs.github.com/en/webhooks/using-webhooks/automatically-redelivering-failed-deliveries-for-a-github-app-webhook)
  that does it. A lost `pull_request` delivery leaves that push unreviewed, and with the
  [Merge gate](merge-gate.md) on, without its check run: `@<your-app-slug> review` fixes
  both.
- The listener speaks HTTP/1.1 only. It closes a connection whose request head has not
  arrived within 10 s of its opening or of its last answer, and keeps at most 256
  connections open: one past that is closed as soon as it is accepted.
- A sender without the secret can also keep those 256 connections taken. When the port is
  reached directly, as with `docker run -p 8080:8080`, set `WEBHOOK_ALLOWED_IPS` to
  GitHub's `hooks` ranges from `GET https://api.github.com/meta`, comma-separated: a
  connection from any other address is closed before it takes one of the 256 or is read,
  and logged with its address. GitHub can change those ranges, so check them again from
  time to time and on any delivery that fails to connect. Behind a proxy or load
  balancer the peer is the proxy, so leave the setting unset there and allow GitHub's
  ranges at the ingress, as above.
- `GET /healthz` answers `200 ok`. The listener opens only once startup succeeded:
  settings valid, Postgres reached and migrated, and with an App, GitHub's `GET /app`
  answered. A failed start exits non-zero instead.

The image bundles git, the Go toolchain (1.27.3, which `diffagent-goctx` needs), gitleaks
8.30.1 and `diffagent-goctx`, all on `PATH`.

### Writable /tmp

`HOME`, `TMPDIR`, `GOCACHE`, `GOMODCACHE` and `GOTMPDIR` all point under `/tmp`, and each
Review's checkout is a `diffagent-ws-*` directory there, holding that Review's own Go
module cache beside its tree (in place of `GOMODCACHE`). With a read-only root filesystem,
mount a writable volume at `/tmp`. Checkouts left by a killed worker are removed at the
next start.

A checkout is sized before it is written: a PR head whose tree holds more than 1 GiB or
200,000 files is not checked out, and that Review reads the diff only (logged as
`checkout failed`, `past the checkout budget`). At most, `/tmp` holds
`WORKER_CONCURRENCY` checkouts at that budget, plus the Go build cache.

### Network

The service connects to GitHub (the API and git over HTTPS), your model providers, your
Jira site if set, and, for a repo with a root `go.mod`, the Go module proxy
(`go mod download`). Modules under the repo's owner are fetched from GitHub with the App's
token.

Modules are fetched per Review: each Review of a Go change downloads its modules into its
own checkout's cache, removed with it. A PR chooses its modules, so a cache shared by every
Review and installation would grow with each PR and nothing would trim it; the cost is a
fresh download, within the 2-minute step limit, on every Review of a Go repo. The build
cache (`GOCACHE`) stays shared.

### TLS

No private or third-party CA is baked in.

- Postgres trusts the webpki roots, plus the file named by `sslrootcert` when `sslmode` is
  `verify-ca` or `verify-full`. For a private CA, mount it and name it in the URL:
  `DATABASE_URL='postgres://user:pass@host:5432/db?sslmode=verify-full&sslrootcert=/etc/diffagent/db-ca.pem'`.
  libpq's `PGSSLMODE` and `PGSSLROOTCERT` apply when the URL leaves them out.
- HTTPS (GitHub, model providers, Jira) and git trust the image's system CA store.

### Postgres

One database holds the state and the job queue. Migrations run at startup under an
advisory lock, so replicas may start together. Each pod opens up to 5 connections.
Replicas share the queue (`SKIP LOCKED`), so you can run more than one.

### Shutdown

On SIGTERM a pod stops claiming jobs and waits up to 9 minutes for the running ones. Give
it that long (`docker stop -t 540`, `terminationGracePeriodSeconds: 540`). A job cut short
runs again once its lease runs out.

### Memory

Reviews are memory-bound, not CPU-bound. Per pod:

- `WORKER_CONCURRENCY` (default 8) jobs run at once, plus one Thread reply on its own lane.
  Each holds its PR's files, diffs and model replies. Lower it to fit a smaller memory
  limit.
- Go type-checking runs outside the worker, in `diffagent-goctx`: at most
  `MAX_CONCURRENT_LOADS` (default 1) at once, each with a soft limit of `GOCTX_MEMORY_MB`
  (default 800 MiB, Go's `GOMEMLIMIT`). The limit is soft: the helper stops itself when
  its own memory passes 1.5 times it (1200 MiB by default), and that Review goes on
  without Go context, logged as `goctx failed` with `passed its hard cap`. If a repo hits
  that often, raise `GOCTX_MEMORY_MB`. Neither limit covers the `go` processes it starts,
  so size the pod for the worker plus 1.5 × `GOCTX_MEMORY_MB` per load, with headroom.
- `diffagent-goctx` and its children are the first a cgroup OOM kills, so a big type-check
  never takes the worker down. That Review goes on without Go context.
- A checkout, with its module cache, on a memory-backed `/tmp` counts against the limit too.
