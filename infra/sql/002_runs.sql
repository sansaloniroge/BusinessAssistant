-- =========================
-- runs (for eval/judge & observability)
-- =========================

CREATE TABLE IF NOT EXISTS runs (
  tenant_id        TEXT        NOT NULL,
  run_id           UUID        NOT NULL,
  user_id          UUID        NOT NULL,
  conversation_id  UUID        NOT NULL,

  question         TEXT        NOT NULL,
  answer           TEXT        NOT NULL,
  model            TEXT        NOT NULL,

  usage            JSONB       NULL,
  retrieved_doc_ids JSONB      NOT NULL DEFAULT '[]'::jsonb,
  retrieval_debug  JSONB       NOT NULL DEFAULT '{}'::jsonb,

  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, run_id)
);

CREATE INDEX IF NOT EXISTS ix_runs_tenant_created_at
  ON runs (tenant_id, created_at DESC);

CREATE INDEX IF NOT EXISTS ix_runs_conversation
  ON runs (tenant_id, conversation_id);

ALTER TABLE runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE runs FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS runs_tenant_isolation ON runs;
CREATE POLICY runs_tenant_isolation
  ON runs
  USING (tenant_id = current_setting('app.tenant_id', true));

