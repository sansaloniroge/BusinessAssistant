"""runs table + RLS

Revision ID: 0002
Revises: 0001
Create Date: 2026-03-09
"""

from __future__ import annotations

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
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
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS runs")

