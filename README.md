# BusinessAssistant

An internal "ask your company's docs" assistant: a multi-tenant RAG API that answers questions strictly from ingested company documents, always with citations, and refuses to answer when it doesn't have enough evidence.

## Problem

Internal knowledge (HR policies, IT/security rules, finance processes, runbooks) is scattered across documents that employees either can't find or don't read. A generic chatbot would happily hallucinate an answer instead of saying "I don't know" — which is worse than no answer at all in a compliance-sensitive company context. This project is a narrow, honest answer to that: retrieval-grounded chat, per-tenant isolated, that would rather refuse than make something up.

## Demo

![A question the ingested docs answer, cited; a question they don't cover, refused](docs/demo.gif)

Two real calls against the running API (no mocking): a question the sample docs actually cover comes back cited (`[C1]` → source doc), a question nothing covers gets refused instead of a guess. Full setup below reproduces this from scratch; `docs/demo.tape` is the [VHS](https://github.com/charmbracelet/vhs) script that generated the GIF (`vhs docs/demo.tape`, API already running).

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
- **RLS *and* explicit `WHERE tenant_id` filters, not either/or** — see below, this one has its own story.

## A security bug I found and fixed: RLS was silently decorative

Multi-tenant isolation is easy to claim and easy to get subtly wrong, so instead of assuming the Postgres RLS policies already in place were doing their job, I tried to actually break isolation. They weren't.

**What I assumed:** every tenant-scoped table (`documents`, `document_chunks`, `runs`, `conversations`, `messages`, the eval tables) had `ENABLE`/`FORCE ROW LEVEL SECURITY` and a `USING (tenant_id = current_setting('app.tenant_id', true))` policy — which was true, and had been since early on. I assumed that meant isolation was enforced.

**What I found:** the role the app actually connected as (`app`, from `POSTGRES_USER` in docker-compose) is a Postgres **superuser**. A superuser ignores row-level security unconditionally — `FORCE ROW LEVEL SECURITY` has no effect on it, and neither does the `SET row_security = on` the code already had, on a mistaken assumption about what that setting controls (it only matters for non-superuser roles with the separate `BYPASSRLS` attribute). I confirmed this directly against `pg_roles` rather than trusting a comment in the code that claimed otherwise.

**Why it had gone unnoticed:** every test that exercised RLS mocked the database connection, so none of them ever hit real Postgres enforcement. And every real query also had an explicit `WHERE tenant_id = $X` filter, so results were always correct in practice — RLS was a second layer of defense that had quietly stopped doing anything, with no visible symptom.

**The fix:** a dedicated, non-superuser, non-bypassrls role (`app_runtime`) for the app's actual runtime connections, created idempotently by `alembic/versions/0005_app_runtime_role.py` with only the DML grants it needs; migrations keep running as the original admin role. Fixing this also surfaced a second bug hiding behind the first: `set_config('app.tenant_id', $1, true)`'s `true` is `SET LOCAL` semantics, which don't persist without an explicit transaction wrapping the statement that depends on them — `app.tenant_id` was silently resetting between the `set_config` call and the query after it, invisible while the connecting role bypassed RLS anyway, and immediately fatal (blocked inserts) the moment RLS started actually being enforced.

**Proof, not a claim:** `tests/test_migrations_and_rls.py` inserts a row under one tenant and asserts a second tenant reads zero rows, for `runs`, `conversations`/`messages`, and `documents`/`document_chunks`. I also verified by hand, connected directly as `app_runtime`: zero rows with no tenant set, zero rows for a fabricated tenant, and a cross-tenant `INSERT` rejected outright with `InsufficientPrivilegeError`.

Full details in [PR #23](https://github.com/sansaloniroge/BusinessAssistant/pull/23).

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
- **No deployed/hosted demo** — a public `/chat` backed by real OpenAI calls and dev-mode header auth is a real cost/abuse surface for a portfolio project. Verified instead via a clean `docker compose up` + migrations + ingestion + eval run (see [How to run it](#how-to-run-it)) and the GIF above, which is a real recorded run, not a mockup.

## What's next

- Wire the ingestion ports into an actual worker + queue, replacing the synchronous script.
- Fix strict-mode citation formatting (the biggest gap the eval run surfaced).
- Add a Redis service to `docker-compose.yml` so rate limiting is exercised by default instead of silently failing open.
- Implement real per-model cost tracking.
- Grow the eval dataset past a smoke-test size once there's more real content to test against.

## License

[MIT](LICENSE)
