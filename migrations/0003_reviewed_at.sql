-- When last_reviewed_sha was last saved by a Review that completed, on the pod's clock:
-- a `review` command posted before it is spent. NULL until then. Never edit an applied
-- migration, add a new one.

-- As in 0001: fail fast instead of queueing behind a live lock.
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

ALTER TABLE pull_requests ADD COLUMN last_reviewed_at TIMESTAMPTZ;
