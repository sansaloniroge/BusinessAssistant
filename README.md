# BusinessAssistant

An internal "ask your company's docs" assistant: a multi-tenant RAG API that answers questions strictly from ingested company documents, always with citations, and refuses to answer when it doesn't have enough evidence.

## Problem

Internal knowledge (HR policies, IT/security rules, finance processes, runbooks) is scattered across documents that employees either can't find or don't read. A generic chatbot would happily hallucinate an answer instead of saying "I don't know" — which is worse than no answer at all in a compliance-sensitive company context. This project is a narrow, honest answer to that: retrieval-grounded chat, per-tenant isolated, that would rather refuse than make something up.

## Architecture

👉 [Architecture documentation](docs/architecture.md) has the full C4 diagrams. Read `docs/architecture.md`'s "Implementation status" section first — the diagrams describe a target architecture (async worker, queue, object storage, reranker) that's broader than what's actually implemented today; see [Known limitations](#known-limitations) below for the honest gap.

What's real today: `FastAPI` API (`chat` / `eval` / `health` routers) → retrieval over `pgvector` → OpenAI for embeddings and generation → citation-checked answer, with every run logged to Postgres.

## Stack

- **API**: FastAPI + uvicorn, async throughout (asyncpg)
- **Vector store**: Postgres + pgvector (cosine distance, HNSW index), with Postgres RLS for tenant isolation
- **LLM / embeddings**: OpenAI (`gpt-4.1-mini` generation and judge, `text-embedding-3-small` embeddings), behind small adapter classes so swapping providers doesn't touch business logic
- **Migrations**: Alembic
- **Observability**: OpenTelemetry (traces + metrics) via `apps/api/services/observability.py`
- **Rate limiting**: Redis-backed, fails open if Redis is unreachable (see limitations — no Redis service in `docker-compose.yml` yet)
- **Tests**: pytest (unit tests mock the DB connection; `@pytest.mark.integration` tests hit a real Postgres)

## How to run it

```bash
cp .env.example .env
# edit .env: set OPENAI_API_KEY

docker compose up -d postgres
set -a && source .env && set +a

python -m alembic upgrade head          # schema + RLS policies + app_runtime/rls_test roles
python -m scripts.ingest_documents      # embeds & uploads the 5 sample docs in scripts/sample_docs/

DEV_DUMMY_LLM=false DEV_DUMMY_EMBEDDINGS=false \
  uvicorn apps.api.asgi:app --reload    # real OpenAI calls; omit the two DEV_DUMMY_* flags to use free dummy stubs instead
```

Then, in DEV mode (`APP_ENV` unset or `dev`/`local`), auth accepts plain headers instead of a signed JWT:

```bash
curl -X POST http://localhost:8000/chat \
  -H "X-Tenant-Id: tenant_test" -H "X-User-Id: 11111111-1111-1111-1111-111111111111" \
  -H "Content-Type: application/json" \
  -d '{"message": "Is MFA required for corporate email?", "mode": "strict", "top_k": 12}'
```

Run the eval suite: `python -m scripts.eval_runner --tenant-id tenant_test --user-id <uuid>` (see [Evaluation](#evaluation)). Run tests: `pytest` (unit only) or `pytest -m integration` (needs the real DB from the steps above).

Verified end-to-end from a completely clean `docker compose down -v` + fresh volume as of this README.

## Key decisions

- **RAG, not fine-tuning.** The knowledge base changes constantly (policies get updated) and per-tenant isolation is a hard requirement — fine-tuning a model per tenant, or retraining on every doc change, doesn't fit either constraint. RAG also gives citations for free, which fine-tuned recall doesn't.
- **Strict mode refuses instead of guessing.** `CitationService.validate_strict` rejects answers that don't cite retrieved chunks in the `[C1]`-style format the system prompt requires. This is deliberately conservative: the [Evaluation](#evaluation) run shows it currently over-refuses (2/5 grounded questions incorrectly rejected because the LLM's citation formatting wasn't consistent) — a false "I don't know" is judged a smaller failure here than a fabricated policy answer.
- **Naive paragraph chunking, not semantic chunking.** `scripts/ingest_documents.py` splits by paragraph up to ~800 chars with a small overlap. It doesn't understand document structure. This is intentionally the cheapest thing that works for a handful of documents, not a claim that it's optimal — see limitations.
- **`text-embedding-3-small` / `gpt-4.1-mini`.** Both are OpenAI's cheaper tier: this is a portfolio project evaluated on dozens of queries, not a cost-at-scale decision. The adapter boundary (`OpenAIEmbeddingService`, `OpenAILLMClient`) exists specifically so swapping to a bigger model, or a different provider, is a one-class change.
- **RLS *and* explicit `WHERE tenant_id` filters, not either/or.** The app connects as `app_runtime`, a dedicated non-superuser/non-bypassrls Postgres role (`alembic/versions/0005_app_runtime_role.py`) — a superuser ignores RLS unconditionally, which is what this project did until that migration, making RLS decorative. `tests/test_migrations_and_rls.py` proves isolation by actually inserting under one tenant and confirming a second tenant reads zero rows, not just checking a policy exists.

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

Raw results land in `.eval_artifacts/eval_run_<id>.json` (gitignored) — re-run with `python -m scripts.eval_runner --tenant-id tenant_test --user-id <uuid>`.

## Known limitations

Stated explicitly rather than glossed over — a small, honestly-scoped project is worth more than one that oversells itself:

- **The async ingestion pipeline is design, not code.** `apps/api/services/ports.py` defines `Extractor`/`Loader`/`Normalizer`/`Chunker`/`DeadLetterQueue` protocols, and `IngestionService` has a real pending→processing→ready/failed state machine — but nothing wires them together, and there's no worker or queue. The only working ingestion path is the synchronous `scripts/ingest_documents.py` script. `docs/architecture.md`'s worker/queue/object-storage diagram is the target, not the current state.
- **n=8 eval set is a smoke test, not a benchmark.** No statistical significance; useful to catch regressions, not to claim a quality percentage.
- **Strict mode over-refuses on citation formatting**, not just on missing evidence (2/5 grounded questions in the eval run above) — a real, reproducible gap, good candidate for few-shot examples in the system prompt.
- **`refusal_correctness` judge scoring is inconsistent for non-refusal answers** — a prompt ambiguity in `EvalJudgeService`, not assistant behavior. Treat that one metric with caution.
- **Cost per query isn't implemented.** `LLMUsage.cost_estimate_usd` always resolves to `0.0` — no per-model pricing table exists yet.
- **Citations passed to the judge are a proxy** (`retrieval_debug.used_chunk_ids`), not the full cited text — the `runs` table doesn't persist full `Citation` objects.
- **Rate limiting fails open by default and there's no Redis service in `docker-compose.yml`** — `check_rate_limit` silently allows all requests if Redis is unreachable, which is exactly the state a fresh `docker compose up` leaves you in unless you run Redis separately.
- **No deployed demo yet** — verified via a clean `docker compose up` + migrations + ingestion + eval run instead (see [How to run it](#how-to-run-it)).

## What's next

- Wire the ingestion ports into an actual worker + queue, replacing the synchronous script.
- Fix strict-mode citation formatting (the biggest gap the eval run surfaced).
- Add a Redis service to `docker-compose.yml` so rate limiting is exercised by default instead of silently failing open.
- Implement real per-model cost tracking.
- Grow the eval dataset past a smoke-test size once there's more real content to test against.
