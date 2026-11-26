# Configuration

diffagent reads its settings from the environment (`src/config.rs`). An empty value counts
as unset; a number or duration that does not parse, or a NaN or
negative price, falls back to its default with a warning naming the variable (never its
value). Durations use Go's syntax (`90s`, `4m`, `8760h`).

| Variable | Default | Meaning |
|---|---|---|
| `GITHUB_WEBHOOK_SECRET` | required | Webhook HMAC secret (`X-Hub-Signature-256`). |
| `GITHUB_APP_ID` | unset: dev mode | The App's id. |
| `GITHUB_APP_PRIVATE_KEY` | required with an App id | The App's private key, the PEM's text. |
| `DATABASE_URL` | required with an App id | Postgres: state and the job queue. Unset in dev mode: memory, lost on restart. |
| `PORT` | `8080` | Listen port. |
| `WEBHOOK_ALLOWED_IPS` | every peer | Comma-separated CIDR ranges, IPv4 or IPv6 (`192.30.252.0/22,2a0a:a440::/29`); a connection from outside them is closed unread. An IPv4-mapped IPv6 peer matches IPv4 ranges. A bad entry stops startup. Set it only when the port is reached directly ([Deploy](setup.md#image-and-endpoints)). |
| `REPO_ALLOWLIST` | every installed repo | Comma-separated `owner/repo`. |
| `ENABLE_FEEDBACK` | on | `false` ignores replies to Findings. |
| `ENABLE_GITLEAKS` | off | `true` scans the PR head for secrets with gitleaks. |
| `GITLEAKS_CONFIG_PATH` | gitleaks' defaults | A gitleaks config file. |
| `COMMITTABLE_SUGGESTIONS` | off | `true` posts one-click suggestion blocks; off posts advisory diffs. |
| `ENABLE_MERGE_GATE` | off | `true` posts the [Merge gate](merge-gate.md)'s check run, under the bot's name (`BOT_NAME`); needs Checks: write. |
| `BOT_NAME` | the App's slug; `diffagent` without an App | The bot's name: the Merge gate's check run, which a ruleset requires by that name, and the status comment's and Walkthrough's heading (`### <name> review`, or `### <name>` when the name already ends in `review`). 1 to 39 of `A-Z a-z 0-9 . _ -`, starting with a letter or digit; anything else stops startup. The `@` handle stays the App's slug. |
| `BEDROCK_REGION` | unset: the AWS SDK's region chain | Bedrock's region. Unset, the SDK picks it (`AWS_REGION`, the profile, instance metadata). Credentials: any AWS SDK source, or `AWS_BEARER_TOKEN_BEDROCK`. |
| `BEDROCK_MODEL_ID` | `global.anthropic.claude-haiku-4-5-20251001-v1:0` | Review model on Bedrock: Claude Haiku 4.5 through a global cross-region profile. |
| `BEDROCK_SUMMARY_MODEL_ID` | the review model | Walkthrough model on Bedrock. |
| `BEDROCK_VERIFIER_MODEL_ID` | off | Second-look model on Bedrock. |
| `REVIEW_MODEL` | Bedrock, `BEDROCK_MODEL_ID` | `provider/model` for the per-file review and Thread answers; overrides `BEDROCK_MODEL_ID`. |
| `SUMMARY_MODEL` | `BEDROCK_SUMMARY_MODEL_ID`, else the review model | `provider/model` for the Walkthrough, its diagram and Ticket fit, and Lesson rules. |
| `CLASSIFY_MODEL` | the review model | `provider/model` for classifying a Maintainer's reply. |
| `VERIFY_MODEL` | `BEDROCK_VERIFIER_MODEL_ID`, else off | `provider/model` for the [second look](merge-gate.md#second-look-at-a-new-critical) at a new critical Finding. |
| `<NAME>_API_KEY` | required for its provider | The key for a provider other than Bedrock: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `ZAI_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, or a custom provider's. A keyless local server takes any value. |
| `LLM_PROVIDER_<NAME>_BASE_URL` | the built-in endpoint | Points a provider at another endpoint; for any other name, adds an OpenAI-compatible provider. |
| `MODEL_PRICE_IN_PER_M`, `MODEL_PRICE_OUT_PER_M` | `0` | Review model's dollars per 1M tokens in and out, for the recorded usage cost. |
| `SUMMARY_MODEL_PRICE_IN_PER_M`, `SUMMARY_MODEL_PRICE_OUT_PER_M` | the review price | The summary model's. |
| `CLASSIFY_MODEL_PRICE_IN_PER_M`, `CLASSIFY_MODEL_PRICE_OUT_PER_M` | the review price | The classify model's. |
| `VERIFY_MODEL_PRICE_IN_PER_M`, `VERIFY_MODEL_PRICE_OUT_PER_M` | the review price | The verify model's. |
| `WORKER_CONCURRENCY` | `8` | Jobs per pod ([memory](setup.md#memory)). |
| `PER_FILE_CONCURRENCY` | `4` | Model calls in flight per PR. |
| `PER_FILE_TIMEOUT` | `4m` | Bound on one model call. |
| `STRUCTURAL_TIMEOUT` | `2m` | Bound on a Review's context build; at or above `PER_FILE_TIMEOUT` it becomes half of it. |
| `MAX_CONCURRENT_LOADS` | `1` | `diffagent-goctx` runs per pod. |
| `GOCTX_PATH` | `diffagent-goctx` | The Go context helper; a bare name is looked up on `PATH`. |
| `GOCTX_MEMORY_MB` | `800` | The helper's `GOMEMLIMIT`, in MiB. |
| `JIRA_BASE_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN` | off | [Jira](jira.md); on only when all three are set. |
| `JIRA_PROJECTS` | every project | Comma-separated Jira project keys (`PROJ,OPS`, upper-cased); when set, an issue in any other project is never looked up. |
| `LEARN_CAP_N` | `20` | Most Lessons in one prompt; `0` means no cap. |
| `LEARN_CANDIDATE_LIMIT` | `100` | Lessons fetched before filtering by the changed directories. |
| `LEARN_HALF_LIFE` | `8760h` | A Lesson's recency half-life, and the age past which a repo-wide Lesson ranks below directory-scoped ones. |

## Model providers

The built-in providers are `bedrock` (no key), `anthropic` (`https://api.anthropic.com`),
`openai` (`https://api.openai.com/v1`), `zai` (`https://api.z.ai/api/paas/v4`),
`deepseek` (`https://api.deepseek.com`) and `openrouter` (`https://openrouter.ai/api/v1`).

A model id may contain `/`; only the first one splits provider from model
(`openrouter/deepseek/deepseek-chat`). A provider's name in a variable is upper-cased,
with any other character as `_` (`my-llm` reads `MY_LLM_API_KEY`).

## Data residency

The default review model, which the summary and classify roles share unless set, is a
Bedrock global cross-region inference profile. Bedrock may serve it from any supported AWS
commercial Region, not only yours, so diffs, file excerpts and Thread history may be
processed outside your region. diffagent has no residency guard, since residency is each
operator's own requirement. If you need it, set every role's route to a regional model id
or to a provider you trust.

## Eval

The eval tool (`diffagent-eval`, [`eval/README.md`](../eval/README.md)) has its own
settings. Its `smoke` subcommand reads the service's settings above, with `REVIEW_MODEL`
from its argument. Every other subcommand takes only these from the environment, and
`--live` refuses to start while a model variable names another model:

| Variable | Default | Meaning |
|---|---|---|
| `BEDROCK_REGION` | unset: the AWS SDK's region chain | The region `--live` calls. |
| `MODEL_PRICE_IN_PER_M`, `MODEL_PRICE_OUT_PER_M` | `0.10`, `0.50` | Dollars per 1M tokens for `--estimate` and `--max-usd`. |
| `GOCTX_PATH` | `target/diffagent-goctx` | The helper `context` runs. |
| `DIFFAGENT_EVAL_CACHE` | `~/.cache/diffagent/eval` | The fetch cache of fixture trees. |
