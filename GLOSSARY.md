# diffagent

A GitHub App that reviews pull requests, answers questions in its review threads, and
learns per-repo rules from Maintainer corrections.

## Reviews

**Review**:
One run of the reviewer over a pull request at one head commit, producing Findings, a
Walkthrough and a Verdict.
_Avoid_: review run, analysis, scan

**GitHub review**:
The batch of inline comments the bot posts to GitHub as a single pull-request review.
_Avoid_: review (unqualified)

**`review` command**:
An @mention asking the bot to run a new Review.
_Avoid_: re-review, trigger

**Finding**:
One problem the bot reports at a location in a changed file, with a severity.
_Avoid_: issue, comment, bug

**Fingerprint**:
A Finding's identity across pushes, derived from where it is in the code, never from its
wording or line number.
_Avoid_: finding ID, hash

**Walkthrough**:
The pull-request-level summary of what changed, posted with a Review.
_Avoid_: summary, overview

**Ticket fit**:
The part of a Walkthrough that lists the requirements of the Jira issue a pull request names, and for a sub-task those of its story it works toward, each marked done, partial or missing in that pull request. Advisory: it never blocks a merge.
_Avoid_: acceptance check, ticket compliance

**Verdict**:
The bot's overall decision on a Review: approve, request changes, or comment.
_Avoid_: approval, gate result

**Merge gate**:
The `diffagent` check run, failing while a critical Finding is open. An open major is
a warning that leaves it `neutral`, which passes. Only critical Findings block a merge: a
wrong block stops one, so only the surest grade may.
_Avoid_: review gate, Verdict

## Conversation and learning

**Thread**:
The GitHub review-comment conversation rooted at a Finding's comment.
_Avoid_: discussion, conversation

**Maintainer**:
A person with write access to the repository, except the PR's author and anyone who
pushed to it, on that PR; only Maintainers' corrections can become Lessons.
_Avoid_: reviewer, owner, admin

**Lesson**:
A per-repo rule learned from a Maintainer's correction, carrying a confidence and its
provenance. The bot proposes it in the Thread, and stores it only once a Maintainer
confirms it with `learn`.
_Avoid_: learning, memory, rule
