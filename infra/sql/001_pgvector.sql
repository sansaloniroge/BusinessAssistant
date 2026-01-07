-- Enable extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Core table: one row per chunk
CREATE TABLE IF NOT EXISTS document_chunks (
  tenant_id      TEXT        NOT NULL,
  chunk_id       TEXT        NOT NULL,
  doc_id         UUID        NOT NULL,
  title          TEXT        NOT NULL,
  content        TEXT        NOT NULL,

  -- IMPORTANT: set dimension to your embedding model (example: 1536)
  embedding      VECTOR(1536) NOT NULL,

  -- Business + technical metadata (filters, ACL, dates, tags, etc.)
  metadata       JSONB       NOT NULL DEFAULT '{}'::jsonb,

  -- Traceability
  embedding_model  TEXT      NOT NULL,
  chunker_version  TEXT      NOT NULL,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),

  PRIMARY KEY (tenant_id, chunk_id)
);

-- Useful indexes
CREATE INDEX IF NOT EXISTS ix_chunks_tenant_doc
  ON document_chunks (tenant_id, doc_id);

-- JSONB filter index (helps metadata filters)
CREATE INDEX IF NOT EXISTS ix_chunks_metadata_gin
  ON document_chunks USING GIN (metadata);

-- Vector index (choose ONE based on your pgvector version and workload)
-- HNSW (recommended when available)
-- CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw
--   ON document_chunks USING hnsw (embedding vector_cosine_ops)
--   WITH (m = 16, ef_construction = 64);

-- IVF_FLAT (older but common)
-- CREATE INDEX IF NOT EXISTS ix_chunks_embedding_ivfflat
--   ON document_chunks USING ivfflat (embedding vector_cosine_ops)
--   WITH (lists = 100);
