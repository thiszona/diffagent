# Contributing

## Build and test

The toolchain is pinned in `rust-toolchain.toml`. The Store's tests start an embedded
Postgres, so `cargo test` needs no Docker (its first run downloads Postgres). It runs `git`
and `go` from `PATH`.

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo deny check                      # licenses, advisories, sources (deny.toml)
cd goctx && test -z "$(gofmt -l .)" && go vet ./... && go test ./...
```

CI runs the same in `.github/workflows/rust.yml`; `.github/workflows/coverage.yml`
measures line coverage on `main` for the README badge. To run the service with Go context
outside the image, see [Run](docs/setup.md#run). `docker build .` builds the image for the
host's architecture.

## Layout

```text
src/bin/diffagent.rs      the service: webhook, worker, graceful SIGTERM
src/bin/diffagent-eval.rs the review-quality eval (eval/README.md)
src/webhook.rs, job.rs    verify, parse and queue a delivery
src/worker/               claim jobs; a Review, Thread replies and commands
src/review/               model calls: per-file review, Walkthrough, classify, answer
src/model/                Bedrock Converse, Anthropic Messages, OpenAI-compatible; tapes
src/context/              cross-file context: TS/JS with tree-sitter, Go through goctx
src/state/                Store trait, PgStore (Postgres) and MemoryStore
src/diff.rs, anchor.rs, plan.rs, render.rs, identity.rs, suggest.rs, scan.rs, learn.rs
goctx/                    diffagent-goctx, the Go helper for type-checked Go context
migrations/               Postgres schema, embedded and run at startup
eval/                     the eval's synthetic corpus, fixtures and fetch script
tests/data/               webhook payloads, TS/JS context trees, the test App key
Dockerfile                the service image
```

[How it works](docs/how-it-works.md) shows how the pieces fit together.

## Conventions

- Every file opens with a `//!` doc saying what it owns and why. Item docs explain intent
  and invariants.
- Use the terms in [GLOSSARY.md](GLOSSARY.md), capitalized as there.
- Fixtures and docs use neutral names such as `acme/shop` and `PROJ-123`.
- Modules are split by domain under `src/`. Pure logic (diff, anchor, plan, render,
  identity, learn, reconcile, eval, prompts) does no IO, so it is tested without GitHub
  or a model.
- Each module has one `thiserror` error enum; `Box<dyn Error>` is for binaries only. Error
  messages never carry secrets.
- No `unwrap`, `expect`, `print!` or `dbg!` outside tests, and no `unsafe`. The lints in
  `Cargo.toml` enforce this.
- Secrets are `config::Secret`. Child processes go through `proc`, and files in a checkout
  are read through `workspace::read`.
- Log with `tracing` macros and structured fields: `info!(repo = %r, pr = n, "message")`.
- Tests sit beside the code in `#[cfg(test)] mod tests`, table-driven. Use the fakes, not
  mocks: `github::fake`, `MemoryStore`, scripted models and tapes. Config parsing takes a
  lookup function, so tests never touch the process environment.
- Add a trait only at a seam with more than one implementation.
- `job::Job` is queued as JSON, so `Job` and `Kind` are append-only: a job queued before a
  deploy must still decode after it.
- Every effect must be safe to repeat, since delivery is at least once.
- Prompts are byte-deterministic, since the tape key covers them.

## Eval

`diffagent-eval` scores reviews against labeled problems. See
[eval/README.md](eval/README.md).

## Pull requests

Work on a branch and open a pull request against `main`. Commit messages follow
Conventional Commits, as in `git log`: `feat(review): ...`, `fix(config): ...`, `docs: ...`,
`test: ...`, `ci: ...`, `chore: ...`. A pull request merges once CI is green.

## Releases

Pushing a `v*` tag runs `.github/workflows/release.yml`. It builds amd64 and arm64
natively and publishes one multi-arch image to `ghcr.io/<owner>/<repo>:<tag>` with
GitHub's own token, so a fork publishes to its own namespace.

## Security

Report security issues privately through GitHub's private vulnerability reporting, not in
a public issue.
