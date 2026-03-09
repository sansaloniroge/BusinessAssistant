"""conversations/messages + RLS

Revision ID: 0004
Revises: 0003
Create Date: 2026-03-09
"""

from __future__ import annotations

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE EXTENSION IF NOT EXISTS pgcrypto;

        CREATE TABLE IF NOT EXISTS conversations (
          tenant_id        TEXT        NOT NULL,
          conversation_id  UUID        NOT NULL,
          created_by       UUID        NOT NULL,
          created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
          PRIMARY KEY (tenant_id, conversation_id)
        );

        CREATE INDEX IF NOT EXISTS ix_conversations_tenant_created_at
          ON conversations (tenant_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS messages (
          tenant_id        TEXT        NOT NULL,
          message_id       UUID        NOT NULL,
          conversation_id  UUID        NOT NULL,
          role             TEXT        NOT NULL,
          content          TEXT        NOT NULL,
          created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
          PRIMARY KEY (tenant_id, message_id),
          FOREIGN KEY (tenant_id, conversation_id)
            REFERENCES conversations(tenant_id, conversation_id)
            ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS ix_messages_conversation_created_at
          ON messages (tenant_id, conversation_id, created_at);

        ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
        ALTER TABLE conversations FORCE ROW LEVEL SECURITY;

        ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
        ALTER TABLE messages FORCE ROW LEVEL SECURITY;

        DROP POLICY IF EXISTS conversations_tenant_isolation ON conversations;
        CREATE POLICY conversations_tenant_isolation
          ON conversations
          USING (tenant_id = current_setting('app.tenant_id', true));

        DROP POLICY IF EXISTS messages_tenant_isolation ON messages;
        CREATE POLICY messages_tenant_isolation
          ON messages
          USING (tenant_id = current_setting('app.tenant_id', true));
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS messages;
        DROP TABLE IF EXISTS conversations;
        """
    )

