## Architecture

The system is designed as a multi-tenant RAG platform with async ingestion and strict retrieval guarantees.

See full architecture diagrams here:
👉 [Architecture documentation](docs/architecture.md)

## Vector Store – Score and filter contract

- Search score: `score = 1 - (embedding <=> query)` using cosine distance in pgvector.
  - Interpretation: higher is better; 1.0 indicates very close, lower values indicate less similarity.
  - Retrieval calculates `evidence_strength` as the average of the scores of the selected chunks.

- Supported MetaFilters operators:
  - `$eq`: equality (columns: department, doc_type; or JSONB metadata key→value).
  - `$in`: value in list (JSONB metadata: (metadata->>‘key’) = ANY(array)).
  - `$contains_any`: intersection in arrays
    - Columns: `tags && ARRAY[...]`.
    - JSONB metadata arrays: `(metadata->‘key’) ?| ARRAY[...]`.
  - `__gte` / `__lte` in date fields (suffix convention):
    - `doc_date__gte` / `doc_date__lte` on column `doc_date`.
    - For date metadata: `(metadata->>‘key’)::timestamptz >= / <= ...`.

- Embedding dimension: fixed at `VECTOR(1536)` and validated in the adapter.
  - Query and chunk embeddings must be 1536 bytes long; otherwise, an early error occurs.

- Multi-tenant: isolation by `tenant_id` (TEXT) via RLS and WHERE clauses in the adapter.
