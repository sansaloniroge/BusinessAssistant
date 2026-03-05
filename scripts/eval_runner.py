from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import asyncpg
import httpx

from apps.api.evals.fixtures import fixtures_for_tenant
from apps.api.evals.repo import PostgresEvalCasesRepo
from apps.api.evals.runs_repo import PostgresEvalRunsRepo


@dataclass(slots=True, frozen=True)
class RunnerConfig:
    base_url: str
    tenant_id: str
    user_id: UUID
    role: str = "user"
    scopes: str = ""

    # Auth real llegará en el paso 9; por ahora se soportan headers.
    # Si se pasa jwt, se envía como Authorization: Bearer <jwt>.
    jwt: str | None = None

    max_cases: int = 50
    timeout_s: float = 60.0

    out_dir: str = "./.eval_artifacts"

    database_url: str | None = None
    persist_eval_cases: bool = False

    eval_model: str = "gpt-4.1-mini"
    eval_mode: str = "strict"


def _headers(cfg: RunnerConfig) -> dict[str, str]:
    h = {
        "X-Tenant-Id": cfg.tenant_id,
        "X-User-Id": str(cfg.user_id),
        "X-Role": cfg.role,
        "X-Scopes": cfg.scopes,
        "Content-Type": "application/json",
    }
    if cfg.jwt:
        h["Authorization"] = f"Bearer {cfg.jwt}"
    return h


async def _run_case(
    client: httpx.AsyncClient,
    cfg: RunnerConfig,
    *,
    eval_run_id: str,
    case_idx: int,
    case: dict[str, Any],
) -> dict[str, Any]:
    req = {
        "message": case["question"],
        "mode": case.get("mode", cfg.eval_mode),
        "top_k": 12,
        "use_rerank": False,
        "filters": {
            # Metadata opcional para trazabilidad; hoy retrieval la ignora si no está mapeada.
            "eval_run_id": eval_run_id,
            "eval_case_id": case.get("eval_case_id"),
        },
        "conversation_id": case.get("conversation_id"),
    }

    t0 = time.perf_counter()
    r = await client.post(f"{cfg.base_url}/chat", headers=_headers(cfg), json=req)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    r.raise_for_status()
    data = r.json()

    run_id = data.get("run_id")
    if not run_id:
        raise RuntimeError("Chat response missing run_id")

    # Dispara judge para persistir eval_result conectado al run.
    eval_case_id = case.get("eval_case_id")
    judge_url = f"{cfg.base_url}/eval/runs/{run_id}/judge"

    params = {"eval_run_id": eval_run_id}
    if eval_case_id:
        params["eval_case_id"] = eval_case_id

    j = await client.post(
        judge_url,
        params=params,
        headers=_headers(cfg),
        json={},
    )
    j.raise_for_status()
    judge = j.json()

    return {
        "case_index": case_idx,
        "eval_run_id": eval_run_id,
        "eval_case_id": eval_case_id,
        "case": case,
        "request": req,
        "response": data,
        "run_id": run_id,
        "judge": judge,
        "latency_ms": latency_ms,
        "usage": data.get("usage"),
    }


async def _persist_dataset_if_enabled(cfg: RunnerConfig, *, eval_run_id: str, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not cfg.persist_eval_cases:
        return cases
    if not cfg.database_url:
        raise SystemExit("--persist-eval-cases requiere --database-url (o env DATABASE_URL)")

    pool = await asyncpg.create_pool(dsn=cfg.database_url, min_size=1, max_size=3)
    try:
        # 1) eval_runs (metadata de la ejecución)
        await PostgresEvalRunsRepo(pool).insert_eval_run(
            tenant_id=cfg.tenant_id,
            eval_run_id=UUID(eval_run_id),
            model=cfg.eval_model,
            mode=cfg.eval_mode,
            max_cases=cfg.max_cases,
        )

        # 2) eval_cases (dataset)
        repo = PostgresEvalCasesRepo(pool)
        out: list[dict[str, Any]] = []
        for c in cases:
            eval_case_id = uuid4()

            from packages.shared.schemas.eval_dataset import EvalCaseFixture

            fixture = EvalCaseFixture.model_validate(c)
            await repo.upsert_case(case_id=eval_case_id, case=fixture)

            cc = dict(c)
            cc["eval_case_id"] = str(eval_case_id)
            out.append(cc)

        return out
    finally:
        await pool.close()


async def main() -> int:
    ap = argparse.ArgumentParser(description="Eval runner (dataset → runs reales → judge → persistencia)")
    ap.add_argument("--base-url", default=os.getenv("API_BASE_URL", "http://localhost:8000"))
    ap.add_argument("--tenant-id", default=os.getenv("EVAL_TENANT_ID"))
    ap.add_argument("--user-id", default=os.getenv("EVAL_USER_ID"))
    ap.add_argument("--jwt", default=os.getenv("EVAL_JWT"))
    ap.add_argument("--max-cases", type=int, default=int(os.getenv("EVAL_MAX_CASES", "50")))
    ap.add_argument("--out-dir", default=os.getenv("EVAL_OUT_DIR", "./.eval_artifacts"))

    ap.add_argument("--persist-eval-cases", action="store_true", default=False)
    ap.add_argument("--database-url", default=os.getenv("DATABASE_URL"))

    ap.add_argument("--eval-model", default=os.getenv("EVAL_MODEL", "gpt-4.1-mini"))
    ap.add_argument("--eval-mode", default=os.getenv("EVAL_MODE", "strict"))

    args = ap.parse_args()

    if not args.tenant_id:
        raise SystemExit("Missing --tenant-id (or env EVAL_TENANT_ID)")
    if not args.user_id:
        raise SystemExit("Missing --user-id (or env EVAL_USER_ID)")

    cfg = RunnerConfig(
        base_url=str(args.base_url).rstrip("/"),
        tenant_id=str(args.tenant_id),
        user_id=UUID(str(args.user_id)),
        jwt=str(args.jwt) if args.jwt else None,
        max_cases=int(args.max_cases),
        out_dir=str(args.out_dir),
        database_url=str(args.database_url) if args.database_url else None,
        persist_eval_cases=bool(args.persist_eval_cases),
        eval_model=str(args.eval_model),
        eval_mode=str(args.eval_mode),
    )

    os.makedirs(cfg.out_dir, exist_ok=True)

    # Dataset por tenant
    fixtures = fixtures_for_tenant(tenant_id=cfg.tenant_id, user_id=cfg.user_id)
    cases = [c.model_dump(mode="json") for c in fixtures][: cfg.max_cases]

    eval_run_id = str(uuid4())
    cases = await _persist_dataset_if_enabled(cfg, eval_run_id=eval_run_id, cases=cases)

    out_path = os.path.join(cfg.out_dir, f"eval_run_{eval_run_id}.json")

    results: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=cfg.timeout_s) as client:
        for i, case in enumerate(cases):
            results.append(await _run_case(client, cfg, eval_run_id=eval_run_id, case_idx=i, case=case))

    payload = {
        "eval_run_id": eval_run_id,
        "tenant_id": cfg.tenant_id,
        "base_url": cfg.base_url,
        "created_at": time.time(),
        "n": len(results),
        "eval_model": cfg.eval_model,
        "eval_mode": cfg.eval_mode,
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(out_path)
    return 0


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(main()))
