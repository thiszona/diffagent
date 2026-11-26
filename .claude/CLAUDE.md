## Project

**diffagent**

diffagent is a self-hosted GitHub App that reviews pull requests with LLM-backed
analysis and posts anchored inline comments (a review bot written in Rust,
Apache-2.0, run by each operator under their own App). Its review threads are
**conversational**: a Maintainer can reply `@<app-slug> kenapa seperti ini?` and get
a useful, analysis-grounded answer in-thread. Every Maintainer correction
feeds a **confidence-scored, self-correcting per-repo learning loop** that compounds
review quality over time without adding noise.

**Core Value:** When a Maintainer questions a Finding, the bot gives a useful,
analysis-grounded answer in that Thread — and every correction durably makes
future Reviews smarter, not noisier.

### Constraints

- **Tech stack**: Rust. The only Go is `goctx/`, since Rust has no equivalent of `go/packages`. The pipeline is webhook → queue → worker → review → post → state
- **Frozen contract**: `job::Job` is queued as JSON, so `Job` and `Kind` are append-only — add variants/fields, never repurpose one (a payload queued before a deploy must decode after it)
- **Cost**: LLM cost is a live concern — file-scoped re-analysis + reuse prompt-cache / structural-context where possible
- **Memory**: a large PR must not run the pod out of memory — re-analysis is single-file, not full-repo
- **Trust**: learning must be maintainer-gated, auditable (provenance), and reversible (`forget`)

## Technology Stack

- Rust 1.100.0 (`rust-toolchain.toml`), edition 2024, one crate (`Cargo.toml`): library in `src/lib.rs`, binaries `diffagent` (the service) and `diffagent-eval` (the eval)
- tokio, axum (webhook), reqwest with rustls (GitHub, Jira, Anthropic/OpenAI wires), sqlx (Postgres, migrations embedded from `migrations/` and run at startup), aws-sdk-bedrockruntime (Converse), tree-sitter with grammars for every reviewed language (`src/lang.rs`: Go, TS/JS, Python, Java, Kotlin, C#, PHP, Ruby, Rust, Swift, C, C++, Scala), serde/serde_json, thiserror, tracing (JSON logs), ring (App JWT), hmac/sha2 (webhook signature), blake2, regex
- Tests: `postgresql_embedded` boots a real Postgres for the Store contract suite (no Docker)
- `goctx/`: `diffagent-goctx`, a separate Go module (go/packages) for type-checked Go context. The only Go in the repo
- Runtime: Postgres holds state and the job queue; the worker runs `git`, `go` + `diffagent-goctx`, and `gitleaks` from `PATH`
- Models: Bedrock by default; Anthropic and OpenAI-compatible providers per role (`REVIEW_MODEL=provider/model`)
- Settings: environment only, parsed in `src/config.rs`; the variable table is in `docs/configuration.md`
- Checks: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`, `cargo deny check`; in `goctx/`: `gofmt -l .` prints nothing, `go vet ./...`, `go test ./...` (CI: `.github/workflows/rust.yml`)
- Fixtures and docs use neutral names (`acme/shop`, `PROJ-123`)
- Release: a `v*` tag runs `.github/workflows/release.yml`, which builds the root `Dockerfile` natively for amd64 and arm64 and publishes `ghcr.io/<repo>:<tag>`; operators deploy it themselves (`docs/setup.md`)

## Conventions

- Every file opens with a `//!` doc saying what it owns and why; item docs explain intent and invariants
- Use the `GLOSSARY.md` terms, capitalized as there: Review, Finding, Thread, Lesson, Maintainer, Fingerprint, Walkthrough
- Modules by domain under `src/`; pure logic (diff, anchor, plan, render, identity, learn, reconcile, eval, prompts) stays free of IO so it is testable without GitHub or a model
- Errors: one `thiserror` enum per module (`ConfigError`, `GhError`, `ModelError`, `StoreError`, `JobError`, ...); `Box<dyn Error>` only in binaries. Messages never carry secrets
- No `unwrap`/`expect`/`print!`/`dbg!` outside tests (clippy lints in `Cargo.toml`, test allowance in `clippy.toml`); `unsafe` is forbidden. `diffagent-eval` writes through an `impl Write`
- Secrets are `config::Secret` (redacted `Debug`, no `Display`); child processes go through `proc` (env cleared to `PATH` plus what the `Cmd` names, process-group kill on timeout); files in a checkout are read through `workspace::read`
- Logging: `tracing` macros with structured fields, `info!(repo = %r, pr = n, "message")`
- Config parsing takes a lookup function, so tests never touch the process env
- Tests sit in `#[cfg(test)] mod tests` beside the code, table-driven over arrays of cases; fakes instead of mocks: `github::fake` (a local GitHub), `MemoryStore`, scripted models and tapes. The Store contract suite runs on both `MemoryStore` and `PgStore`
- Traits only at seams with more than one implementation: `Store`, `Model`, the webhook's `Queue`

## Architecture

```text
POST /webhook -> webhook.rs (HMAC, 5 MiB cap, allowlist) -> job::from_event -> jobs table
worker/ claims (SKIP LOCKED, leases, per-kind timeouts, retry then Dead)
  Review: workspace checkout -> context (TS/JS tree-sitter, Go via diffagent-goctx) + gitleaks
          -> review/ model calls (per-file C1, walkthrough C2+C3) -> identity
          -> second look at new criticals (C4) -> reconcile
          -> plan (anchor) -> render -> worker/post.rs -> state
  Thread reply: classify (C6) -> gate in code -> answer (C5) / Lesson rule (C7) -> state
  Commands: review, pause, resolve, lessons, forget
```

| Module | Owns |
|---|---|
| `webhook`, `job` | Front door; `Job` and its append-only `Kind` |
| `worker/` | Claim loops (main lane + one-at-a-time Thread lane), Review, Thread replies, commands, notes, posting |
| `review/` | Prompts, per-file review, walkthrough, reply classification and gate, answers |
| `model/` | `Router` per role over Bedrock, Anthropic and OpenAI-compatible wires; schema-JSON replies; `Tape` record/replay; usage `Meter` |
| `context/` | `ContextPack` per file: TS/JS in Rust, Go through the goctx subprocess |
| `state/` | `Store` trait, `PgStore`, `MemoryStore`, contract suite, reconcile |
| `diff`, `anchor`, `plan`, `render`, `identity`, `suggest`, `scan`, `learn`, `lang` | Diff lines, anchoring, post plan, comment markdown and `diffagent:v1` markers, Fingerprints, suggestion re-parse, gitleaks, Lesson ranking, tree-sitter languages |
| `github`, `workspace`, `proc`, `jira`, `config` | App client with retries, checkouts, subprocesses, ticket intent, settings |

Invariants:

- Delivery is at least once: every effect is safe to repeat (comments checked by marker, completion records written last)
- Model output is untrusted: anchors are checked against the diff, the model proposes a reply's class and code decides, a critical Finding is never suppressed by a reply
- Prompts are byte-deterministic, since the tape key covers them
- One Review's Go type-check runs in `diffagent-goctx`, bounded in count, memory and time, never in the worker

## Agent skills

### Issue tracker

GitHub Issues on `thiszona/diffagent`, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: the glossary is the root `GLOSSARY.md`. See `docs/agents/domain.md`.
