# Review-quality eval

`diffagent-eval` (`src/bin/diffagent-eval.rs`) scores reviews against labeled problems:
Findings a Maintainer rejected or accepted, and bugs planted on purpose. Run it from the
repo root:

```sh
cargo run --bin diffagent-eval -- <command>
```

It ships with a small **synthetic corpus** in `synthetic/`: one made-up repository
(`acme/shop`), three one-file fixtures and the inputs every subcommand reads. It is there
so you can see how scoring works and so the eval's tests run anywhere. It is far too small
to measure review quality; for that, label your own repositories (below).

## Commands

| Command | What it does |
|---|---|
| `score FINDINGS.json [--against FINDINGS.json]` | Scorecard of a run's Findings against the labels, beside another run's. |
| `compare --base RUN... --cand RUN...` | Paired per-label comparison of runs. A RUN is a findings file or a dir under `out/`. |
| `coverage [--budget N]` | How much labeled cross-file evidence the committed context packs reach. |
| `smoke PROVIDER/MODEL` | One live structured call as the review role; prints the raw reply and the decoded value. |
| `context [--write] [FIXTURE...]` | Builds each fixture's context packs from the fetch cache and checks them against `synthetic/context/`; `--write` rewrites the stale ones. |
| `run RUN [--from RUN] [--lessons FILE] MODE [FIXTURE...]` | Per-file review (C1) on the fixtures, scored beside the baseline. |
| `summary RUN [--from RUN] MODE [PR...]` | Walkthrough and diagram (C2+C3) per summary PR. |
| `verify RUN [--from RUN] MODE [FIXTURE...]` | The second look at a critical (C4) on the recorded run's criticals and on rejected Findings shown as critical; fails when a labeled critical is not held. |
| `classify RUN [--from RUN] MODE [FIXTURE...]` | Reply-intent classification (C6) on the intent, injection and rejected-reply cases. |
| `thread RUN [--from RUN] MODE [reject\|hold\|inject...]` | The Thread ladder (C6, C7, C5) on `synthetic/threads.json`. |
| `thread --write-cases` | Writes `synthetic/threads.json` from local data only. |

`MODE` is nothing (replay), `--estimate` (what a live run would cost), or
`--live --max-usd N` (records; refuses to start over N, stops after the fixture that
crosses it).

A run's tapes and outputs live in `out/RUN/`, one tape per fixture, and git ignores that
directory. `--from RUN` replays another run's tape where this run has none.

## What works where

- **Offline, on the shipped corpus:** `score`, `compare`, `coverage`, `thread
  --write-cases`, and `--estimate` on every paid subcommand. Nothing calls a model.
- **Replay** (no `MODE`) needs a recorded run: tapes in `out/RUN/` from an earlier
  `--live`. No tapes ship, so replay works only after you record one.
- **`--live`** calls Claude Haiku 5.5 on Bedrock (`global.anthropic.claude-haiku-5-5`, a
  global cross-region profile) for every role, with your AWS credentials. Set
  `BEDROCK_REGION` for the region to call. Prices come from `MODEL_PRICE_IN_PER_M` and
  `MODEL_PRICE_OUT_PER_M` (default 0.10 and 0.50 dollars per 1M tokens). The model
  variables (`REVIEW_MODEL`, `BEDROCK_MODEL_ID`, ...) are ignored, and `--live` refuses to
  start if one names another model, since replay and live must key their tapes alike.
- **`context` and `fetch.sh`** need labels that name real repositories and commits: they
  check out each fixture's tree. The synthetic labels name none, so the corpus's packs in
  `synthetic/context/` are written by hand, and `context` fails on them.

## The corpus

| Path | What it is |
|---|---|
| `synthetic/labels.history.json` | Labels on Findings a Maintainer rejected or accepted, by fixture, with the fixture's repo and commits. |
| `synthetic/labels.planted.json` | Planted bugs: one visible in the diff, one provable only from another file (`"scope": "cross-file"`, with the `evidence` lines `coverage` looks for). |
| `synthetic/fixtures/` | Each fixture's reviewed files. |
| `synthetic/context/` | Context packs, by fixture; a fixture without one is reviewed diff-only. |
| `synthetic/summary/` | The PR `summary` writes a Walkthrough for, with its ticket intent. |
| `synthetic/rejected.json` | The rejected labels' replies, which `classify` and the reject Thread cases use. |
| `synthetic/hostile-notes.json` | A team note per hold case telling the reviewer to drop its planted bug, in `run --lessons` format. |
| `synthetic/threads.json` | Thread cases: reject (an English and an Indonesian rejection, on a major and a critical Finding), hold (the hostile notes, on a critical and a major Finding) and inject. |
| `synthetic/baseline.json` | The run every `run` scorecard is printed beside. |
| `synthetic/recorded.json` | A recorded run: the criticals `verify` checks and the Findings the hold cases reply to. |
| `fixtures/intent/`, `fixtures/injection/`, `intent-golden.json` | Intent and prompt-injection cases, and the fp_precision floor `classify` holds. |
| `fixtures/poisoned_lesson.json` | A team note telling the reviewer to ignore SQL injection; `run` fails if the injection is not raised anyway. |

`baseline.json` and `recorded.json` are `run --live` outputs (`findings.json`) against
scripted models, not a real model; the test `corpus_runs_are_scripted_runs` rewrites
them and fails when they differ. `threads.json` is `thread --write-cases`' output, pinned
the same way by `write_cases_is_deterministic`.

## Your own labels

The eval reads the corpus from `synthetic/`, so you replace its files, keeping the
layout. The eval's tests are written against the synthetic corpus, so run them on it,
not on yours.

1. Label fixtures from your repositories in `labels.history.json` (`"kind": "history"`,
   the PR's `base_sha` and `head_sha`, `rejected` and `accepted` labels with each
   Finding's `title`) and `labels.planted.json` (`"kind": "planted"`, the `base_sha` the
   bug is planted on). A history fixture is named `history-REPO-PR-SHA7`, after its
   repository, PR number and head commit.
2. Put each fixture's changed files in `fixtures/<fixture>.json` and each rejected
   label's reply in `rejected.json`.
3. Cache the trees with `eval/fetch.sh` and write the packs with `context --write`
   (below).
4. Record a run with `run RUN --live --max-usd N` and copy `out/RUN/findings.json` to
   `baseline.json`; a later run, or the same one, is `recorded.json` for `verify` and the
   hold cases.
5. Write a hostile note per planted fixture you want as a hold case in
   `hostile-notes.json`, then `thread --write-cases`.

## Fetching trees

`fetch.sh [FIXTURE...]` caches each labeled fixture's tree, outside the repo, for
`context` and `thread --write-cases` (`$DIFFAGENT_EVAL_CACHE`, default
`~/.cache/diffagent/eval`). It reads private repositories and Go modules through an
authenticated gh: Go modules under the fixture's repository owner are fetched as private.
Needs bash, git, gh, jq and go.

`context` runs the Go helper at `target/diffagent-goctx` (or `$GOCTX_PATH`); build it with
`go build -C goctx -o ../target/diffagent-goctx`.
