-- =========================
-- eval (LLM-as-judge) connected to runs
-- =========================

CREATE TABLE IF NOT EXISTS eval_cases (
  tenant_id        TEXT        NOT NULL,
  eval_case_id     UUID        NOT NULL,

  question         TEXT        NOT NULL,
  expected_doc_ids JSONB       NOT NULL DEFAULT '[]'::jsonb,
  notes            TEXT        NULL,

  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, eval_case_id)
);

CREATE INDEX IF NOT EXISTS ix_eval_cases_tenant_created_at
  ON eval_cases (tenant_id, created_at DESC);

ALTER TABLE eval_cases ENABLE ROW LEVEL SECURITY;
ALTER TABLE eval_cases FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS eval_cases_tenant_isolation ON eval_cases;
CREATE POLICY eval_cases_tenant_isolation
  ON eval_cases
  USING (tenant_id = current_setting('app.tenant_id', true));


CREATE TABLE IF NOT EXISTS eval_runs (
  tenant_id        TEXT        NOT NULL,
  eval_run_id      UUID        NOT NULL,
  model            TEXT        NOT NULL,
  mode             TEXT        NOT NULL,
  max_cases        INT         NOT NULL,

  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, eval_run_id)
);

CREATE INDEX IF NOT EXISTS ix_eval_runs_tenant_created_at
  ON eval_runs (tenant_id, created_at DESC);

ALTER TABLE eval_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE eval_runs FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS eval_runs_tenant_isolation ON eval_runs;
CREATE POLICY eval_runs_tenant_isolation
  ON eval_runs
  USING (tenant_id = current_setting('app.tenant_id', true));


CREATE TABLE IF NOT EXISTS eval_results (
  tenant_id        TEXT        NOT NULL,
  eval_result_id   UUID        NOT NULL,

  eval_run_id      UUID        NULL,
  eval_case_id     UUID        NULL,
  run_id           UUID        NOT NULL,

  overall          INT         NOT NULL CHECK (overall BETWEEN 0 AND 5),
  faithfulness     INT         NOT NULL CHECK (faithfulness BETWEEN 0 AND 5),
  relevance        INT         NOT NULL CHECK (relevance BETWEEN 0 AND 5),
  citation_quality INT         NOT NULL CHECK (citation_quality BETWEEN 0 AND 5),
  refusal_correctness INT      NOT NULL CHECK (refusal_correctness BETWEEN 0 AND 5),
  rationale        TEXT        NOT NULL,

  judge_model      TEXT        NOT NULL,
  judge_usage      JSONB       NULL,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, eval_result_id)
);

CREATE INDEX IF NOT EXISTS ix_eval_results_by_run
  ON eval_results (tenant_id, run_id);

CREATE INDEX IF NOT EXISTS ix_eval_results_by_eval_run
  ON eval_results (tenant_id, eval_run_id);

ALTER TABLE eval_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE eval_results FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS eval_results_tenant_isolation ON eval_results;
CREATE POLICY eval_results_tenant_isolation
  ON eval_results
  USING (tenant_id = current_setting('app.tenant_id', true));
