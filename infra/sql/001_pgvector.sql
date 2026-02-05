-- Enable extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Helper: keep updated_at in sync
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =========================
-- documents
-- =========================
CREATE TABLE IF NOT EXISTS documents (
  tenant_id      TEXT        NOT NULL,
  doc_id         UUID        NOT NULL,

  title          TEXT        NOT NULL,
  status         TEXT        NOT NULL DEFAULT 'pending',
  source_type    TEXT        NOT NULL DEFAULT 'upload',
  access_level   TEXT        NOT NULL DEFAULT 'public',

  blob_uri       TEXT        NULL,
  checksum       TEXT        NULL,

  department     TEXT        NULL,
  doc_type       TEXT        NULL,
  tags           TEXT[]      NOT NULL DEFAULT ARRAY[]::text[],
  language       TEXT        NULL,
  doc_date       TIMESTAMPTZ NULL,

  metadata       JSONB       NOT NULL DEFAULT '{}'::jsonb,

  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, doc_id)
);

DROP TRIGGER IF EXISTS trg_documents_updated_at ON documents;
CREATE TRIGGER trg_documents_updated_at
BEFORE UPDATE ON documents
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE INDEX IF NOT EXISTS ix_documents_tenant_status
  ON documents (tenant_id, status);

CREATE INDEX IF NOT EXISTS ix_documents_tenant_department
  ON documents (tenant_id, department);

CREATE INDEX IF NOT EXISTS ix_documents_tenant_doc_type
  ON documents (tenant_id, doc_type);

CREATE INDEX IF NOT EXISTS ix_documents_tags_gin
  ON documents USING GIN (tags);

CREATE INDEX IF NOT EXISTS ix_documents_metadata_gin
  ON documents USING GIN (metadata);


-- =========================
-- document_chunks
-- =========================
-- NOTE: if this table already exists from an earlier migration, you may need a
-- dedicated ALTER migration instead of CREATE. For V1 we keep it self-contained.
CREATE TABLE IF NOT EXISTS document_chunks (
  tenant_id       TEXT         NOT NULL,
  chunk_id        TEXT         NOT NULL,
  doc_id          UUID         NOT NULL,

  title           TEXT         NOT NULL,
  content         TEXT         NOT NULL,

  -- IMPORTANT: set dimension to your embedding model (example: 1536)
  embedding       VECTOR(1536) NOT NULL,

  -- Denormalized filter columns (fast path)
  department      TEXT         NULL,
  doc_type        TEXT         NULL,
  tags            TEXT[]       NOT NULL DEFAULT ARRAY[]::text[],
  doc_date        TIMESTAMPTZ  NULL,

  -- Extra metadata (section/page/source refs, etc.)
  metadata        JSONB        NOT NULL DEFAULT '{}'::jsonb,

  embedding_model TEXT         NOT NULL,
  chunker_version TEXT         NOT NULL,

  created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, chunk_id),
  FOREIGN KEY (tenant_id, doc_id) REFERENCES documents(tenant_id, doc_id) ON DELETE CASCADE
);

-- Useful indexes
CREATE INDEX IF NOT EXISTS ix_chunks_tenant_doc
  ON document_chunks (tenant_id, doc_id);

CREATE INDEX IF NOT EXISTS ix_chunks_tenant_department
  ON document_chunks (tenant_id, department);

CREATE INDEX IF NOT EXISTS ix_chunks_tenant_doc_type
  ON document_chunks (tenant_id, doc_type);

CREATE INDEX IF NOT EXISTS ix_chunks_tags_gin
  ON document_chunks USING GIN (tags);

CREATE INDEX IF NOT EXISTS ix_chunks_doc_date
  ON document_chunks (tenant_id, doc_date);

-- JSONB filter index (kept for ad-hoc metadata filters)
CREATE INDEX IF NOT EXISTS ix_chunks_metadata_gin
  ON document_chunks USING GIN (metadata);

-- Vector index
-- HNSW (recommended when available)
CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw
  ON document_chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);


-- =========================
-- RLS (Row-Level Security)
-- =========================
-- Convention: set `app.tenant_id` per-connection:
--   SET app.tenant_id = '<tenant>';
-- Then RLS ensures strict isolation.

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;

-- Force RLS so table owners don't accidentally bypass it.
ALTER TABLE documents FORCE ROW LEVEL SECURITY;
ALTER TABLE document_chunks FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS documents_tenant_isolation ON documents;
CREATE POLICY documents_tenant_isolation
  ON documents
  USING (tenant_id = current_setting('app.tenant_id', true));

DROP POLICY IF EXISTS chunks_tenant_isolation ON document_chunks;
CREATE POLICY chunks_tenant_isolation
  ON document_chunks
  USING (tenant_id = current_setting('app.tenant_id', true));
