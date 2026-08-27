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

- Multi-tenant: isolation by `tenant_id` (TEXT) via RLS *and* explicit `WHERE tenant_id = $X` clauses in the adapter — both matter. The app connects as `app_runtime`, a dedicated non-superuser/non-bypassrls role (`alembic/versions/0005_app_runtime_role.py`); migrations run as a separate admin role. This isn't cosmetic: a Postgres superuser ignores RLS unconditionally regardless of `ENABLE`/`FORCE ROW LEVEL SECURITY`, so connecting as one (as this project did until this was fixed) makes RLS policies decorative — real isolation would depend entirely on every query remembering its `WHERE tenant_id` filter, with no second layer of defense. `tests/test_migrations_and_rls.py` verifies isolation is actually enforced by inserting under one tenant and confirming a second tenant reads zero rows, on both `runs`/`conversations`/`messages` and `documents`/`document_chunks` (the RAG content itself).

## Evaluation

Quantitative evaluation via LLM-as-judge (`EvalJudgeService`), run against the **real** stack: real embeddings (`OpenAIEmbeddingService`), real retrieval (`PgvectorVectorStore`), real generation (`OpenAILLMClient`), scored 0–5 by `gpt-4.1-mini` on `faithfulness`, `relevance`, `citation_quality`, `refusal_correctness`, `overall`.

**Dataset**: 8 cases (`apps/api/evals/fixtures.py`) against the 5 synthetic documents ingested by `scripts/ingest_documents.py` — 5 "grounded" questions (one per document, evidence should exist) and 3 "out-of-domain" questions (no ingested document covers them; a correct answer is a refusal). This is a small, hand-written smoke set, not a statistically representative benchmark — treat these numbers as a snapshot of one run, not a stable SLA.

**Latest run** (`tenant_test`, mode `strict`, n=8):

| Metric | Value |
|---|---|
| Overall (avg) | 4.62 / 5 |
| Faithfulness (avg) | 4.88 / 5 |
| Relevance (avg) | 5.00 / 5 |
| Citation quality (avg) | 3.62 / 5 |
| Refusal correctness (avg) | 3.75 / 5 |
| Latency p50 / p95 (chat call) | 1.8s / 3.4s |
| Correctly refused (out-of-domain, 3 cases) | 3 / 3 |
| False rejections (grounded, 5 cases) | 2 / 5 |

**Known limitations of this evaluation, stated explicitly rather than glossed over:**
- **n=8 is a smoke test, not a benchmark.** No statistical significance; useful to catch regressions, not to claim a quality percentage.
- **2 of 5 grounded questions were incorrectly refused** in strict mode: retrieval found the right document (top cosine score 0.6+), but the LLM's answer didn't include a `[C1]`-style citation tag that `CitationService.validate_strict` requires, so it was rejected as ungrounded. This is a real, reproducible gap in strict-mode citation formatting, not a retrieval failure — a good candidate for future work (e.g. few-shot examples in the system prompt for citation formatting).
- **`refusal_correctness` scoring is inconsistent for non-refusal answers.** The judge prompt only defines this dimension for the "did refuse" case; when the assistant answers normally, the judge sometimes scores it 5 ("not applicable, default pass") and sometimes 0 — an artifact of prompt ambiguity, not of assistant behavior. Treat this specific dimension's average with caution until the judge prompt is tightened.
- **Cost per query is not implemented.** `LLMUsage.cost_estimate_usd` always resolves to `0.0` — no per-model pricing table exists in the code yet, so this run reports token counts (≈7.8k total tokens across 8 chat calls) but not a real dollar cost.
- **Citations passed to the judge are a proxy, not the full picture.** The `runs` table doesn't persist full `Citation` objects (chunk_id/title/snippet), only `retrieval_debug.used_chunk_ids`; the judge sees chunk IDs, not the cited text itself.

**How to reproduce**: ingest sample data (`python -m scripts.ingest_documents`), run the API locally with real providers (`DEV_DUMMY_LLM=false DEV_DUMMY_EMBEDDINGS=false uvicorn apps.api.asgi:app`), then `python -m scripts.eval_runner --tenant-id tenant_test --user-id <uuid>`. Raw results land in `.eval_artifacts/eval_run_<id>.json` (gitignored).
