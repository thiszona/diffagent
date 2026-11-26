# Jira

diffagent works with any Jira Cloud site. Set all three variables to turn it on:

- `JIRA_BASE_URL`: your site, such as `https://your-site.atlassian.net`;
- `JIRA_EMAIL` and `JIRA_API_TOKEN`: an account that can read the issues, and its API
  token (read-only, HTTP Basic).

A PR names an issue by its key (`PROJ-123`) in its branch, title or body; the first key
Jira knows is used. The issue, its parent, epic, sub-tasks and links become the Review's
ticket intent, which the per-file review reads.

The PR's author picks those keys, and the lookup uses the service account, so it is bound:

- Only a PR whose branch is in the repository itself names an issue. A PR from a fork,
  which anyone may open on a public repository, gets no Jira lookup.
- With `JIRA_PROJECTS` set (`PROJ,OPS`), a key in any other project is never sent to
  Jira, an issue's parent included.

The Ticket fit below is posted on the PR for anyone who can read it, so give the account
read access only to projects whose issues the repositories' PR authors may see, or name
those projects in `JIRA_PROJECTS`.

The Walkthrough then also has a Ticket fit: the issue's requirements, each marked done,
partial or missing in this PR, from one more summary-model call. For a sub-task, the
acceptance criteria of its story that the sub-task works toward are listed too, under the
story's key. A Ticket fit never blocks a merge.

A Jira error, or no answer within 10 s, leaves the Review without ticket intent.
