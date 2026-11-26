-- Who pushed to each PR: the sender of its opened, reopened and synchronize events.
-- Nobody who pushed code to a PR is a Maintainer on it, so they cannot dismiss what their
-- own push brought in. Never edit an applied migration, add a new one.

-- As in 0001: fail fast instead of queueing behind a live lock.
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

CREATE TABLE pushers (
    repo_id    BIGINT NOT NULL REFERENCES repos (id) ON DELETE CASCADE,
    pr         BIGINT NOT NULL,
    login      TEXT   NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (repo_id, pr, login)
);
