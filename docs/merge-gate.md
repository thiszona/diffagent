# Merge gate

The Merge gate is your App's check run on your repositories, which your rulesets can
require. With `ENABLE_MERGE_GATE=true` the bot posts a check run named `diffagent` on the
PR's head.

The run fails while a critical Finding is open, in a Thread or not. An open major is a
warning: it makes the run `neutral`, which passes, and so does a Review that could not
finish some files. Only critical Findings block: a wrong block stops a merge, so only the
surest grade may. `BLOCKING` in `src/worker/review.rs` makes majors block too.

The run's summary lists each blocker (under "Blocking") and each warning (under "Should be
fixed"), up to 50 each, with a link to its Thread when it has one, then how they clear. The
Walkthrough also says when the gate is blocked. Made required, the check run stops a merge
past a critical Finding.

A check run belongs to a commit, not a PR, so a pass also counts every other open PR at
the same head. Their criticals fail it, their majors make it `neutral`, and it waits for
their Reviews there.

## Second look at a new critical

With `VERIFY_MODEL` set, a critical the model reports that is not already stored as
critical gets a second look first (C4: the same rubric and context, without team notes).
If the second check does not hold it critical, it is posted as a major with its reason, so
it warns instead of blocking. A failed check keeps it critical. gitleaks' Findings are
never checked.

## Clearing a Finding

A blocking Finding leaves only by:

- a push that fixes it: the model, shown the file's whole patch, no longer reports it in a
  file the push changed. After a force-push, or in a file the last Review failed on, that
  takes a later push that changes the file again. A file the model is not sent (generated,
  no patch from GitHub, or past the 3,000 files GitHub lists for a PR) or sees only in part
  (a patch over 30,000 bytes is cut) keeps its Findings open;
- `@<your-app-slug> resolve` from a Maintainer. It closes what was open when the comment
  was posted: a Finding a Review stores after it (a push still queued or running) stays
  open.

A secret Finding (critical security, titled "Hardcoded secret") leaves by a push only where
gitleaks ran (`ENABLE_GITLEAKS=true`) over its file's patch and the model was sent the
file: the model can write that title too, so neither alone proves the secret is gone. One
in a config or test file, which the model is not sent, leaves by `resolve`, after the
secret is rotated.

A warning also leaves by a Maintainer's command in its Thread: `learn`, confirming the
Lesson a false positive or won't fix proposed, or `close`. A reply alone closes nothing,
since the models that read it also read the Finding, whose text the PR's author shaped.
Neither command closes a critical Finding, and a suppression never covers one: a
suppressed Finding a later Review finds critical (after the second look) is open again
and blocks.

Resolving the conversation in GitHub changes nothing, and neither does Re-run: the bot
drops `check_run.rerequested`.

## Paused and failed Reviews

A Review that is paused or still waiting posts nothing, so GitHub keeps waiting. A push, a
`review` command or redelivering the webhook from the App's Advanced tab moves it on.

A Review that ends Dead posts `failure` while a critical is open or its last attempt found
one, else `neutral` ("Review failed"), so its PR is not blocked unexplained.

## Turning it on

1. In your App's settings, set the repository permission Checks to Read and write, then
   accept it on each installation.
2. Set `ENABLE_MERGE_GATE=true` and redeploy. Every reviewed PR then shows the check, not
   yet required.
3. In a sandbox repo, once a run has passed, add a ruleset requiring `diffagent` with its
   source pinned to your App. Check that:
   - a known critical blocks the merge;
   - a Maintainer's `resolve` passes the same head;
   - a `review` command there that finds it again blocks again;
   - a lone major shows `neutral` and does not block;
   - a commit status named `diffagent` from anyone else does not satisfy the rule;
   - the author's own `resolve` changes nothing, and neither does that of a writer who
     pushed to the PR.
4. On your other repos' default branches, require it the same way, preferably as rulesets,
   with the source pinned to your App, never "any source". Not on a merge-queue branch.
   Leave "Require conversation resolution before merging" off. Keep a small bypass list:
   it is the break-glass for a Dead Review, a repeating 422 or a bot outage.
5. Open PRs get a run on their next push, a `review` command, or a close and reopen (free:
   it posts from the store). Require the check once they drain.
6. **Remove the required check before you turn the flag off, take a repo off
   `REPO_ALLOWLIST`, or suspend or uninstall the App.** Otherwise every PR there waits
   forever.
