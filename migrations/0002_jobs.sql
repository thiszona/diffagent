-- The job queue. Never edit an applied migration, add a new one.
--
-- The UNIQUE delivery is the webhook's dedupe: the insert is the only record, so a
-- delivery that got a 500 can be redelivered. A slot is 'pr:<repo_id>:<n>' or
-- 'thread:<repo_id>:<root>', and the partial unique index lets one job per slot run.

-- As in 0001: fail fast instead of queueing behind a live lock.
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

CREATE TABLE jobs (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    delivery    TEXT NOT NULL UNIQUE,                 -- X-GitHub-Delivery
    slot        TEXT NOT NULL,
    payload     TEXT NOT NULL,                        -- crate::job::Job as JSON
    state       TEXT NOT NULL DEFAULT 'queued' CHECK (state IN ('queued', 'running', 'done', 'dead')),
    run_at      TIMESTAMPTZ NOT NULL,
    lease_until TIMESTAMPTZ,
    -- At most 4: the fourth claim ends the job Dead.
    -- squawk-ignore prefer-bigint-over-int
    attempts    INT NOT NULL DEFAULT 0,
    error       TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX jobs_running ON jobs (slot) WHERE state = 'running';
