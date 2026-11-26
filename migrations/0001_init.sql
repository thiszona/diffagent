-- diffagent's database: repos, per-PR state, Findings and their history, Lessons, usage.
-- Applied by sqlx::migrate! on connect; never edit an applied migration, add a new one.
--
-- "None" is NULL, never a sentinel 0 or ''.

-- Fail fast instead of queueing behind a live lock in production; LOCAL scopes these to
-- this migration's transaction.
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

-- Keyed by GitHub's numeric repository id, not the name, so a rename or transfer keeps
-- every Finding and Lesson. full_name is display only, refreshed on each sighting.
CREATE TABLE repos (
    id              BIGINT PRIMARY KEY,
    installation_id BIGINT NOT NULL,
    full_name       TEXT   NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Per-PR state, saved as a whole row (callers read, modify, write).
CREATE TABLE pull_requests (
    repo_id           BIGINT  NOT NULL REFERENCES repos (id) ON DELETE CASCADE,
    number            BIGINT  NOT NULL,
    status_comment_id BIGINT,                         -- the one status comment to keep editing
    paused            BOOLEAN NOT NULL DEFAULT false,
    last_reviewed_sha TEXT,
    approved_sha      TEXT,                           -- a Maintainer's approve, until the next push
    walkthrough       TEXT,                           -- cached Walkthrough, opaque to the store
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (repo_id, number)
);

-- A Finding's identity is (repo_id, pr, fingerprint). Suppression is a status, so a
-- suppressed Finding can never also be open.
CREATE TABLE findings (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    repo_id     BIGINT NOT NULL REFERENCES repos (id) ON DELETE CASCADE,
    pr          BIGINT NOT NULL,
    fingerprint TEXT   NOT NULL,
    path        TEXT   NOT NULL,
    start_line  BIGINT NOT NULL,
    end_line    BIGINT NOT NULL,
    severity    TEXT   NOT NULL,                      -- crate::finding::Severity
    category    TEXT   NOT NULL,                      -- crate::finding::Category
    title       TEXT   NOT NULL,
    comment_id  BIGINT,                               -- the Thread root, once posted
    status      TEXT   NOT NULL CHECK (status IN ('open', 'resolved', 'suppressed')),
    first_sha   TEXT   NOT NULL,
    last_sha    TEXT   NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (repo_id, pr, fingerprint)
);
CREATE INDEX findings_open_idx ON findings (repo_id, pr) WHERE status = 'open';

-- Append-only lifecycle history, one row per transition.
CREATE TABLE finding_events (
    id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    finding_id BIGINT NOT NULL REFERENCES findings (id) ON DELETE CASCADE,
    kind       TEXT   NOT NULL CHECK (kind IN (
        'new', 'reaffirm', 'resolve', 'carry', 'suppress', 'unsuppress', 'acknowledge'
    )),
    sha        TEXT,                                  -- NULL for a reply, which has no head commit
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX finding_events_finding_idx ON finding_events (finding_id);

-- Maintainer-taught rules. Identity is the rule text per repo: re-teaching revives the
-- row. source_comment_id is provenance and the redelivery guard, not identity.
CREATE TABLE lessons (
    id                   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    repo_id              BIGINT NOT NULL REFERENCES repos (id) ON DELETE CASCADE,
    rule                 TEXT   NOT NULL,
    kind                 TEXT   NOT NULL CHECK (kind IN ('unknown', 'false_positive', 'convention')),
    scope                TEXT   NOT NULL,             -- directory subtree; '' is global
    confidence           DOUBLE PRECISION NOT NULL,
    decayed_confidence   DOUBLE PRECISION,            -- measure-only shadow; NULL until a decay run
    status               TEXT   NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'retired')),
    source_comment_id    BIGINT,
    author               TEXT   NOT NULL,
    pr                   BIGINT,
    trigger_count        BIGINT NOT NULL DEFAULT 0,
    last_triggered_at    TIMESTAMPTZ,
    contradicted_count   BIGINT NOT NULL DEFAULT 0,
    last_contradicted_at TIMESTAMPTZ,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (repo_id, rule)
);
-- Serves the newest-first candidate fetch for ranking.
CREATE INDEX lessons_active_idx ON lessons (repo_id, created_at DESC) WHERE status = 'active';

-- Token use and cost per Review and model role. The anchoring outcome stats describe the
-- Review, so only the review role's row carries them (zero elsewhere).
CREATE TABLE usage (
    id                 BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    repo_id            BIGINT NOT NULL REFERENCES repos (id) ON DELETE CASCADE,
    pr                 BIGINT NOT NULL,
    head_sha           TEXT   NOT NULL,
    role               TEXT   NOT NULL,
    provider           TEXT   NOT NULL,
    model              TEXT   NOT NULL,
    input_tokens       BIGINT NOT NULL,
    output_tokens      BIGINT NOT NULL,
    cache_read_tokens  BIGINT NOT NULL,
    cache_write_tokens BIGINT NOT NULL,
    est_cost_usd       DOUBLE PRECISION NOT NULL,
    inline             BIGINT NOT NULL,
    relocated          BIGINT NOT NULL,
    snapped            BIGINT NOT NULL,
    demoted            BIGINT NOT NULL,
    verified           BIGINT NOT NULL,
    committable        BIGINT NOT NULL,
    failed_files       BIGINT NOT NULL,
    kept_files         BIGINT NOT NULL,
    reviewed_files     BIGINT NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX usage_repo_idx ON usage (repo_id, created_at);
CREATE INDEX usage_model_idx ON usage (provider, model, created_at);
