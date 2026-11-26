//! A Review's inline comments on GitHub: listing ours, posting the plan's new ones as
//! one review, and closing resolved ones.
//!
//! Every create is checked first by its finding marker, so a rerun never posts a
//! Finding twice: [`Poster::existing`] lists ours, [`Poster::post_review`] posts only
//! what that listing lacks, and a create GitHub never confirmed is listed for before
//! the still-missing comments are sent, once.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use serde::Deserialize;
use serde_json::{Value, json};
use tracing::{info, warn};

use super::JobError;
use super::review::Created;
use super::thread::Listed;
use crate::github::{Gh, GhError};
use crate::job::Job;
use crate::plan::{Comment, Plan};
use crate::render::{is_addressed_closure, parse_finding_marker};
use crate::state::StoredFinding;

/// Listings, at most, for comments GitHub created but does not list yet.
const TRIES: usize = 3;

/// The wait between those listings. Tests only count listings, so theirs is short.
const GAP: Duration = if cfg!(test) {
    Duration::from_millis(10)
} else {
    Duration::from_secs(2)
};

/// Thread pages per listing: 3,000 Threads, REST's page cap.
const THREAD_PAGES: usize = 30;

/// A PR's Threads, each with its first comment's id. The first comment of a Thread is
/// its root, which is the comment a Finding links to.
const THREADS: &str = "query($owner: String!, $repo: String!, $pr: Int!, $cursor: String) {
  repository(owner: $owner, name: $repo) { pullRequest(number: $pr) {
    reviewThreads(first: 100, after: $cursor) {
      pageInfo { hasNextPage endCursor }
      nodes { id isResolved comments(first: 1) { nodes { databaseId } } } } } } }";

const RESOLVE: &str =
    "mutation($id: ID!) { resolveReviewThread(input: {threadId: $id}) { thread { id } } }";

/// Our finding comments on a PR, as one listing found them.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct Existing {
    /// The open ones: Fingerprint → (comment id, body).
    pub open: BTreeMap<String, (i64, String)>,
    /// The ids of those with a closure note. A Finding linked to one of these has no
    /// open comment: a critical or major one is posted again, like a regression.
    pub closed: BTreeSet<i64>,
}

/// One PR's review comments and Threads, for one job.
pub(crate) struct Poster<'a> {
    gh: &'a Gh,
    owner: &'a str,
    repo: &'a str,
    pr: i64,
    /// `<slug>[bot]`: only its comments are ours.
    bot: &'a str,
}

impl<'a> Poster<'a> {
    pub(crate) fn new(gh: &'a Gh, job: &'a Job, bot: &'a str) -> Self {
        Self {
            gh,
            owner: &job.owner,
            repo: &job.repo,
            pr: job.pr,
            bot,
        }
    }

    /// Our finding comments: by `bot`, with our finding marker. A human's pasted marker
    /// is not ours. Those with a closure note are only listed as closed: their marker
    /// must not hide a regression of its Finding. Of two open with one Fingerprint, the
    /// oldest wins.
    pub(crate) async fn existing(&self) -> Result<Existing, GhError> {
        let listed: Vec<Listed> = self
            .gh
            .pages(&self.path(&format!("pulls/{}/comments", self.pr)))
            .await?;
        let mut ours = Existing::default();
        for c in listed {
            let by_us = c.user.as_ref().is_some_and(|u| u.login == self.bot);
            let Some(fp) = parse_finding_marker(&c.body).filter(|_| by_us) else {
                continue;
            };
            if is_addressed_closure(&c.body) {
                ours.closed.insert(c.id);
            } else {
                let fp = fp.to_owned();
                ours.open.entry(fp).or_insert((c.id, c.body));
            }
        }
        Ok(ours)
    }

    /// Posts the plan's comments that `existing` lacks as one COMMENT review at `head`,
    /// and returns every plan comment's id by Fingerprint: listed or new. Nothing new
    /// means no POST.
    ///
    /// The new ids are read back by review id. A create GitHub never confirmed is
    /// listed for, and only what is still missing is sent again, once. A 422 refuses
    /// the anchors: Dead with the anchors in its error, unless the head moved, when
    /// this returns `None`: the Review is stale, the job ends Done, and the new head
    /// has its own job.
    pub(crate) async fn post_review(
        &self,
        head: &str,
        plan: &Plan,
        existing: &Existing,
    ) -> Result<Option<BTreeMap<String, i64>>, JobError> {
        let mut ids: BTreeMap<String, i64> = plan
            .comments
            .iter()
            .filter_map(|c| Some((c.fingerprint.clone(), existing.open.get(&c.fingerprint)?.0)))
            .collect();
        let mut todo: Vec<&Comment> = plan
            .comments
            .iter()
            .filter(|c| !ids.contains_key(&c.fingerprint))
            .collect();
        if todo.is_empty() {
            return Ok(Some(ids));
        }
        let created = match self.create(head, &todo).await {
            Err(GhError::Ambiguous) => {
                warn!(repo = %self.full(), pr = self.pr, comments = todo.len(), "review create unconfirmed; listing before sending again");
                ids.extend(self.find(&todo, None).await?);
                todo.retain(|c| !ids.contains_key(&c.fingerprint));
                if todo.is_empty() {
                    return Ok(Some(ids));
                }
                self.create(head, &todo).await
            }
            other => other,
        };
        let review = match created {
            Ok(id) => id,
            Err(GhError::Unprocessable(why)) => return self.refused(head, &todo, &why).await,
            Err(e) => return Err(e.into()),
        };
        let found = self.find(&todo, Some(review)).await?;
        if found.len() < todo.len() {
            // The comments exist, so the rerun's listing links them without a post.
            return Err(GhError::Http(format!(
                "review {review}: {} of {} comments not listed after {TRIES} tries",
                todo.len() - found.len(),
                todo.len()
            ))
            .into());
        }
        ids.extend(found);
        Ok(Some(ids))
    }

    /// Closes resolved Findings: appends `note` (`addressed_note(head)`, or
    /// `RESOLVED_NOTE` for the `resolve` command) to each body `existing` listed, then
    /// resolves their Threads from one listing. `existing` holds no comment with a
    /// closure note, so a rerun skips the notes an earlier attempt added but still
    /// resolves a Thread it left open, found by the stored comment id.
    ///
    /// The note and the Thread are cosmetic, as in a Thread reply, so a failure is
    /// only logged: failing on one would re-run the whole paid Review, and GitHub
    /// refuses `resolveReviewThread` to an App without `contents: write`, which would
    /// leave every such Review Dead with no check run. A comment left without its note
    /// reads as open, so a later Review links it again instead of posting another.
    pub(crate) async fn close(&self, findings: &[StoredFinding], note: &str, existing: &Existing) {
        let mut roots = Vec::new();
        for f in findings {
            match existing.open.get(&f.fingerprint) {
                Some((id, body)) => {
                    if let Err(e) = self.append(*id, &format!("{body}{note}")).await {
                        warn!(repo = %self.full(), pr = self.pr, comment = id, err = %e,
                            "closure note failed");
                    }
                    roots.push(*id);
                }
                None => roots.extend(f.comment_id),
            }
        }
        if let Err(e) = self.resolve(&roots).await {
            warn!(repo = %self.full(), pr = self.pr, err = %e, "resolving threads failed");
        }
    }

    /// Resolves the Threads rooted at `roots` from one listing; a resolved Thread is
    /// left as it is.
    pub(crate) async fn resolve(&self, roots: &[i64]) -> Result<(), GhError> {
        if roots.is_empty() {
            return Ok(());
        }
        let threads = self.threads().await?;
        for root in roots {
            if let Some((thread, false)) = threads.get(root) {
                let _: Value = self.gh.graphql(RESOLVE, &json!({"id": thread})).await?;
            }
        }
        Ok(())
    }

    /// Creates a COMMENT review of `todo` at `head`; its id.
    async fn create(&self, head: &str, todo: &[&Comment]) -> Result<i64, GhError> {
        let path = self.path(&format!("pulls/{}/reviews", self.pr));
        let review: Created = self.gh.create(&path, &review_body(head, todo)).await?;
        Ok(review.id)
    }

    /// Lists until every comment of `todo` shows, [`TRIES`] times [`GAP`] apart: GitHub
    /// can take a moment to list what it created. `review` lists that review's
    /// comments; `None` lists all of ours. Fingerprint → id of those found.
    async fn find(
        &self,
        todo: &[&Comment],
        review: Option<i64>,
    ) -> Result<BTreeMap<String, i64>, GhError> {
        let want: BTreeSet<&str> = todo.iter().map(|c| c.fingerprint.as_str()).collect();
        let mut found = BTreeMap::new();
        for i in 0..TRIES {
            if i > 0 {
                tokio::time::sleep(GAP).await;
            }
            let listed: Vec<(String, i64)> = match review {
                Some(id) => {
                    let path = self.path(&format!("pulls/{}/reviews/{id}/comments", self.pr));
                    let listed: Vec<Listed> = self.gh.pages(&path).await?;
                    listed
                        .into_iter()
                        .filter_map(|c| Some((parse_finding_marker(&c.body)?.to_owned(), c.id)))
                        .collect()
                }
                None => self
                    .existing()
                    .await?
                    .open
                    .into_iter()
                    .map(|(fp, (id, _))| (fp, id))
                    .collect(),
            };
            found.extend(
                listed
                    .into_iter()
                    .filter(|(fp, _)| want.contains(fp.as_str())),
            );
            if found.len() == want.len() {
                break;
            }
        }
        Ok(found)
    }

    /// A 422 on the create: stale when the live head moved, else Dead with the anchors.
    async fn refused(
        &self,
        head: &str,
        todo: &[&Comment],
        why: &str,
    ) -> Result<Option<BTreeMap<String, i64>>, JobError> {
        #[derive(Deserialize)]
        struct Pull {
            head: Head,
        }
        #[derive(Deserialize)]
        struct Head {
            sha: String,
        }
        let live: Pull = self
            .gh
            .get(&self.path(&format!("pulls/{}", self.pr)))
            .await?;
        if live.head.sha != head {
            info!(repo = %self.full(), pr = self.pr, head, live = %live.head.sha, "review refused after a push; the new head has its own job");
            return Ok(None);
        }
        let anchors: Vec<String> = todo.iter().map(|c| anchor(c)).collect();
        Err(GhError::Unprocessable(format!("{why}; anchors: {}", anchors.join(", "))).into())
    }

    /// Replaces a review comment's body. One deleted since the listing has nothing
    /// left to close.
    async fn append(&self, id: i64, body: &str) -> Result<(), GhError> {
        let path = self.path(&format!("pulls/comments/{id}"));
        match self.gh.edit::<Value>(&path, &json!({"body": body})).await {
            Ok(_) | Err(GhError::NotFound) => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Every Thread of the PR: root comment id → (Thread id, resolved).
    async fn threads(&self) -> Result<BTreeMap<i64, (String, bool)>, GhError> {
        let mut out = BTreeMap::new();
        let mut cursor = Value::Null;
        for _ in 0..THREAD_PAGES {
            let vars =
                json!({"owner": self.owner, "repo": self.repo, "pr": self.pr, "cursor": cursor});
            let data: Value = self.gh.graphql(THREADS, &vars).await?;
            let page = &data["repository"]["pullRequest"]["reviewThreads"];
            let Some(nodes) = page["nodes"].as_array() else {
                return Err(GhError::Http("graphql: no reviewThreads".to_owned()));
            };
            for t in nodes {
                let root = t["comments"]["nodes"][0]["databaseId"].as_i64();
                if let (Some(id), Some(root)) = (t["id"].as_str(), root) {
                    out.insert(root, (id.to_owned(), t["isResolved"] == true));
                }
            }
            if page["pageInfo"]["hasNextPage"] != true {
                return Ok(out);
            }
            cursor = page["pageInfo"]["endCursor"].clone();
        }
        warn!(repo = %self.full(), pr = self.pr, pages = THREAD_PAGES, "review threads cut at the page cap");
        Ok(out)
    }

    fn path(&self, rest: &str) -> String {
        format!("/repos/{}/{}/{rest}", self.owner, self.repo)
    }

    fn full(&self) -> String {
        format!("{}/{}", self.owner, self.repo)
    }
}

/// The create-review body: a COMMENT review pinned to `head`, so GitHub reads every
/// line against the diff that was reviewed, not a newer push. A span adds its first
/// line; a single line sends neither start field.
fn review_body(head: &str, todo: &[&Comment]) -> Value {
    let comments: Vec<Value> = todo
        .iter()
        .map(|c| {
            let mut draft = json!({"path": c.finding.path, "body": c.body,
                "line": c.anchor.line, "side": "RIGHT"});
            if let Some(start) = c.anchor.start_line {
                draft["start_line"] = json!(start);
                draft["start_side"] = json!("RIGHT");
            }
            draft
        })
        .collect();
    json!({"commit_id": head, "event": "COMMENT", "comments": comments})
}

/// `path:line`, or `path:start-line` for a span.
fn anchor(c: &Comment) -> String {
    match c.anchor.start_line {
        Some(start) => format!("{}:{start}-{}", c.finding.path, c.anchor.line),
        None => format!("{}:{}", c.finding.path, c.anchor.line),
    }
}

#[cfg(test)]
mod tests {
    use std::time::SystemTime;

    use super::*;
    use crate::anchor::Anchor;
    use crate::finding::{Category, Finding, Severity};
    use crate::github::fake::{Fake, FaultKind};
    use crate::job::Kind;
    use crate::plan::OutcomeStats;
    use crate::render::{RESOLVED_NOTE, addressed_note, finding_marker};
    use crate::state::{FindingStatus, JobEnd};

    const BOT: &str = "diffagent[bot]";
    const HEAD: &str = "c0ffee0123456789";
    const POST: &str = "POST /repos/o/r/pulls/1/reviews";
    const LIST: &str = "GET /repos/o/r/pulls/1/comments";
    const A: &str = "0a";
    const B: &str = "0b";
    const C: &str = "0c";

    /// A fake with PR 1 at [`HEAD`], and a client for it.
    async fn setup() -> (Fake, Gh) {
        let fake = Fake::start().await;
        fake.world()
            .pulls
            .insert(1, json!({"number": 1, "head": {"sha": HEAD}}));
        let gh = fake.github().installation(7).await.unwrap();
        (fake, gh)
    }

    fn job() -> Job {
        Job {
            installation_id: 7,
            repo_id: 9,
            owner: "o".to_owned(),
            repo: "r".to_owned(),
            pr: 1,
            actor: "dev".to_owned(),
            kind: Kind::Review { command: false },
            commented_at: None,
            pushed: false,
        }
    }

    fn body(fp: &str) -> String {
        format!("**MINOR** · logic\n#### t\n\nb\n\n{}\n", finding_marker(fp))
    }

    fn comment(fp: &str, path: &str, line: i64, start_line: Option<i64>) -> Comment {
        Comment {
            anchor: Anchor {
                line,
                start_line,
                verified: true,
                committable: false,
            },
            body: body(fp),
            finding: Finding {
                path: path.to_owned(),
                start_line: start_line.unwrap_or(line),
                end_line: line,
                severity: Severity::Minor,
                category: Category::Logic,
                title: "t".to_owned(),
                body: "b".to_owned(),
                snippet: String::new(),
                suggestion: String::new(),
            },
            fingerprint: fp.to_owned(),
            symbol: None,
        }
    }

    fn plan(fps: &[&str]) -> Plan {
        Plan {
            walkthrough: String::new(),
            comments: fps
                .iter()
                .enumerate()
                .map(|(i, fp)| comment(fp, "a.go", 3 + i64::try_from(i).unwrap(), None))
                .collect(),
            stats: OutcomeStats::default(),
        }
    }

    fn stored(fp: &str, comment_id: Option<i64>) -> StoredFinding {
        StoredFinding {
            repo_id: 9,
            pr: 1,
            fingerprint: fp.to_owned(),
            path: "a.go".to_owned(),
            start_line: 3,
            end_line: 3,
            severity: Severity::Minor,
            category: Category::Logic,
            title: "t".to_owned(),
            comment_id,
            status: FindingStatus::Open,
            first_sha: HEAD.to_owned(),
            last_sha: HEAD.to_owned(),
        }
    }

    /// The review comments the fake holds, by Fingerprint.
    fn held(fake: &Fake) -> Vec<(String, Value)> {
        fake.world()
            .review_comments
            .iter()
            .map(|(_, c)| {
                let fp = parse_finding_marker(c["body"].as_str().unwrap()).unwrap();
                (fp.to_owned(), c.clone())
            })
            .collect()
    }

    fn ids(pairs: &[(&str, i64)]) -> Option<BTreeMap<String, i64>> {
        Some(
            pairs
                .iter()
                .map(|(fp, id)| ((*fp).to_owned(), *id))
                .collect(),
        )
    }

    #[test]
    fn the_create_review_body() {
        let one = comment(A, "a.go", 3, None);
        let span = comment(B, "b.go", 9, Some(7));
        assert_eq!(
            review_body(HEAD, &[&one, &span]),
            json!({"commit_id": HEAD, "event": "COMMENT", "comments": [
                {"path": "a.go", "body": body(A), "line": 3, "side": "RIGHT"},
                {"path": "b.go", "body": body(B), "line": 9, "side": "RIGHT",
                 "start_line": 7, "start_side": "RIGHT"},
            ]})
        );
    }

    /// The body above reaches the fake's create-review route as one review at the head;
    /// a second run with the listing posts nothing.
    #[tokio::test]
    async fn one_review_at_the_head_then_nothing_new_posts_nothing() {
        let (fake, gh) = setup().await;
        let job = job();
        let p = Poster::new(&gh, &job, BOT);
        let mut plan = plan(&[A]);
        plan.comments.push(comment(B, "b.go", 9, Some(7)));

        let got = p
            .post_review(HEAD, &plan, &Existing::default())
            .await
            .unwrap();
        let comments = held(&fake);
        let id = |i: usize| comments[i].1["id"].as_i64().unwrap();
        assert_eq!(got, ids(&[(A, id(0)), (B, id(1))]));
        let (a, b) = (&comments[0].1, &comments[1].1);
        assert_eq!(
            (&a["path"], &a["line"], &a["side"]),
            (&json!("a.go"), &json!(3), &json!("RIGHT"))
        );
        assert!(
            a.get("start_line").is_none() && a.get("start_side").is_none(),
            "{a}"
        );
        assert_eq!(
            (&b["start_line"], &b["start_side"]),
            (&json!(7), &json!("RIGHT"))
        );
        assert_eq!(fake.world().reviews[0].1["commit_id"], HEAD);

        let listed = p.existing().await.unwrap();
        assert_eq!(p.post_review(HEAD, &plan, &listed).await.unwrap(), got);
        assert_eq!(fake.sent(POST), 1, "nothing new: no POST");
    }

    /// GitHub can list a new review's comments late: the read-back lists again, 3
    /// times in all. Still hidden after that, the attempt fails, and the rerun's listing
    /// links them without posting again.
    #[tokio::test]
    async fn the_read_back_waits_out_hidden_comments() {
        for hidden in [0, 1, 2, 3] {
            let (fake, gh) = setup().await;
            let job = job();
            let p = Poster::new(&gh, &job, BOT);
            let plan = plan(&[A, B]);
            fake.world().fault(POST, 1, FaultKind::Hide(hidden));
            let got = p.post_review(HEAD, &plan, &Existing::default()).await;
            let reads = fake
                .world()
                .log
                .iter()
                .filter(|l| l.contains("/reviews/") && l.starts_with("GET "))
                .count();
            assert_eq!(reads, (hidden + 1).min(TRIES), "{hidden} hidden");
            let comments = held(&fake);
            let want = ids(&[
                (A, comments[0].1["id"].as_i64().unwrap()),
                (B, comments[1].1["id"].as_i64().unwrap()),
            ]);
            if hidden < TRIES {
                assert_eq!(got.unwrap(), want, "{hidden} hidden");
                continue;
            }
            let err = got.unwrap_err();
            assert!(
                matches!(
                    super::super::ending(&Err(err), 1, SystemTime::UNIX_EPOCH),
                    JobEnd::Retry { .. }
                ),
                "runs again"
            );
            let listed = p.existing().await.unwrap();
            assert_eq!(p.post_review(HEAD, &plan, &listed).await.unwrap(), want);
            assert_eq!(fake.sent(POST), 1, "posted once");
        }
    }

    /// An unconfirmed create is listed for, 3 times, and only what is still missing is
    /// sent again, once.
    #[tokio::test]
    async fn an_unconfirmed_create_posts_only_the_missing_once() {
        // Applied, then a 502: the listing finds both, so nothing is sent again.
        let (fake, gh) = setup().await;
        let job = job();
        let p = Poster::new(&gh, &job, BOT);
        fake.world().fault(POST, 1, FaultKind::After(502));
        let got = p
            .post_review(HEAD, &plan(&[A, B]), &Existing::default())
            .await
            .unwrap();
        let comments = held(&fake);
        assert_eq!(comments.len(), 2);
        assert_eq!(got.unwrap().len(), 2);
        assert_eq!((fake.sent(POST), fake.sent(LIST)), (1, 1));

        // Not applied, but A was posted by an earlier attempt the caller's listing
        // missed: three listings find A, and B alone is posted.
        let (fake, gh) = setup().await;
        let p = Poster::new(&gh, &job, BOT);
        fake.world().review_comments.push((
            1,
            json!({"id": 900, "body": body(A), "user": {"login": BOT}}),
        ));
        fake.world().fault(POST, 1, FaultKind::Before(502));
        let got = p
            .post_review(HEAD, &plan(&[A, B]), &Existing::default())
            .await
            .unwrap();
        let comments = held(&fake);
        let fps: Vec<&str> = comments.iter().map(|(fp, _)| fp.as_str()).collect();
        assert_eq!(fps, [A, B]);
        assert_eq!(
            got,
            ids(&[(A, 900), (B, comments[1].1["id"].as_i64().unwrap())])
        );
        assert_eq!((fake.sent(POST), fake.sent(LIST)), (2, TRIES));

        // Unconfirmed twice: the second is not sent again.
        let (fake, gh) = setup().await;
        let p = Poster::new(&gh, &job, BOT);
        fake.world().fault(POST, 2, FaultKind::Before(502));
        let err = p
            .post_review(HEAD, &plan(&[A]), &Existing::default())
            .await
            .unwrap_err();
        assert!(
            matches!(err, JobError::Github(GhError::Ambiguous)),
            "{err:?}"
        );
        assert_eq!(fake.sent(POST), 2);
        assert!(fake.world().review_comments.is_empty());
    }

    /// A 422 at the reviewed head is Dead, its error naming the anchors; after a push
    /// it is stale, and the job is Done.
    #[tokio::test]
    async fn a_422_is_dead_unless_the_head_moved() {
        let (fake, gh) = setup().await;
        let job = job();
        let p = Poster::new(&gh, &job, BOT);
        let mut plan = plan(&[A]);
        plan.comments.push(comment(B, "b.go", 9, Some(7)));
        fake.world().fault(POST, 1, FaultKind::Before(422));
        let err = p
            .post_review(HEAD, &plan, &Existing::default())
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("anchors: a.go:3, b.go:7-9"),
            "{err}"
        );
        assert!(matches!(
            super::super::ending(&Err(err), 1, SystemTime::UNIX_EPOCH),
            JobEnd::Dead { .. }
        ));

        fake.world()
            .pulls
            .insert(1, json!({"number": 1, "head": {"sha": "f00d"}}));
        fake.world().fault(POST, 1, FaultKind::Before(422));
        let got = p
            .post_review(HEAD, &plan, &Existing::default())
            .await
            .unwrap();
        assert_eq!(got, None, "stale: Done");
        assert!(fake.world().review_comments.is_empty());
    }

    /// Each closed comment gets one note and its Thread resolved, from one Thread
    /// listing. A failed listing is only logged; a later close adds no second note and
    /// resolves the Threads by their stored ids.
    #[tokio::test]
    async fn close_notes_once_and_lists_threads_once() {
        let (fake, gh) = setup().await;
        let job = job();
        let p = Poster::new(&gh, &job, BOT);
        let posted = p
            .post_review(HEAD, &plan(&[A, B, C]), &Existing::default())
            .await
            .unwrap()
            .unwrap();
        let resolved = [stored(A, Some(posted[A])), stored(B, None)];
        let note = addressed_note("abcdef0123");
        let graphql = "POST /graphql";

        let listed = p.existing().await.unwrap();
        p.close(&resolved, &note, &listed).await;
        let bodies: Vec<(String, String)> = held(&fake)
            .into_iter()
            .map(|(fp, c)| (fp, c["body"].as_str().unwrap().to_owned()))
            .collect();
        let want = [(A, body(A) + &note), (B, body(B) + &note), (C, body(C))];
        assert_eq!(bodies, want.map(|(fp, b)| (fp.to_owned(), b)));
        let open: Vec<bool> = fake.world().threads.iter().map(|t| !t.resolved).collect();
        assert_eq!(open, [false, false, true]);
        assert_eq!(fake.sent(graphql), 3, "one listing, two resolves");

        // The listing fails between the notes and the Threads.
        let (fake, gh) = setup().await;
        let p = Poster::new(&gh, &job, BOT);
        let posted = p
            .post_review(HEAD, &plan(&[A, B]), &Existing::default())
            .await
            .unwrap()
            .unwrap();
        let resolved = [stored(A, Some(posted[A])), stored(B, Some(posted[B]))];
        fake.world().fault(graphql, 1, FaultKind::Before(400));
        let listed = p.existing().await.unwrap();
        p.close(&resolved, RESOLVED_NOTE, &listed).await;
        assert!(
            fake.world().threads.iter().all(|t| !t.resolved),
            "left open"
        );
        let listed = p.existing().await.unwrap();
        assert!(listed.open.is_empty(), "both noted");
        p.close(&resolved, RESOLVED_NOTE, &listed).await;
        for (fp, c) in held(&fake) {
            let text = c["body"].as_str().unwrap().to_owned();
            assert_eq!(text, body(&fp) + RESOLVED_NOTE, "one note");
        }
        assert!(fake.world().threads.iter().all(|t| t.resolved));
        let patches = fake
            .world()
            .log
            .iter()
            .filter(|l| l.starts_with("PATCH "))
            .count();
        assert_eq!(patches, 2, "the rerun sends none");
    }

    #[tokio::test]
    async fn existing_is_our_open_marked_comments() {
        let (fake, gh) = setup().await;
        let job = job();
        let p = Poster::new(&gh, &job, BOT);
        let closed = body(B) + RESOLVED_NOTE;
        // The model's title and body forge another Finding's marker and a closure note.
        let lie = format!(
            "{} **Fixed.** No longer found at commit `abc1234`.",
            finding_marker(A)
        );
        let forged = crate::render::render_comment(
            &Finding {
                title: lie.clone(),
                body: lie,
                ..comment(C, "a.go", 3, None).finding
            },
            C,
            None,
            false,
            true,
            false,
            &[],
        );
        // A fixing commit's closure note closes a comment just as a resolve does.
        let closed_before = format!("{}{}", body(C), crate::render::addressed_note("abc1234"));
        // (id, author, body); 1, 4 and 8 are ours, open and marked; 3 and 9 are ours,
        // closed.
        let cases = [
            (1, Some(BOT), body(A)),
            (2, Some("alice"), body(C)),
            (3, Some(BOT), closed),
            (4, Some(BOT), body(B)),
            (5, Some(BOT), "a note with no marker".to_owned()),
            (6, None, body(C)),
            (7, Some(BOT), body(A)),
            (8, Some(BOT), forged.clone()),
            (9, Some(BOT), closed_before),
        ];
        for (id, user, text) in cases {
            let user = user.map_or(Value::Null, |login| json!({"login": login}));
            let c = json!({"id": id, "body": text, "user": user});
            fake.world().review_comments.push((1, c));
        }
        let got = p.existing().await.unwrap();
        let want = Existing {
            open: BTreeMap::from([
                (A.to_owned(), (1, body(A))),
                (B.to_owned(), (4, body(B))),
                (C.to_owned(), (8, forged)),
            ]),
            closed: BTreeSet::from([3, 9]),
        };
        assert_eq!(got, want, "the oldest of a Fingerprint wins");
    }
}
