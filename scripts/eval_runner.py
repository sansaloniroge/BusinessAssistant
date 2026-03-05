from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import httpx

from apps.api.evals.fixtures import fixtures_for_tenant


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


async def _run_case(client: httpx.AsyncClient, cfg: RunnerConfig, case_idx: int, case: dict[str, Any]) -> dict[str, Any]:
    req = {
        "message": case["question"],
        "mode": case.get("mode", "strict"),
        "top_k": 12,
        "use_rerank": False,
        "filters": None,
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
    j = await client.post(
        f"{cfg.base_url}/eval/runs/{run_id}/judge",
        headers=_headers(cfg),
        json={},
    )
    j.raise_for_status()
    judge = j.json()

    return {
        "case_index": case_idx,
        "case": case,
        "request": req,
        "response": data,
        "run_id": run_id,
        "judge": judge,
        "latency_ms": latency_ms,
        "usage": data.get("usage"),
    }


async def main() -> int:
    ap = argparse.ArgumentParser(description="Eval runner (dataset → runs reales → judge → persistencia)")
    ap.add_argument("--base-url", default=os.getenv("API_BASE_URL", "http://localhost:8000"))
    ap.add_argument("--tenant-id", default=os.getenv("EVAL_TENANT_ID"))
    ap.add_argument("--user-id", default=os.getenv("EVAL_USER_ID"))
    ap.add_argument("--jwt", default=os.getenv("EVAL_JWT"))
    ap.add_argument("--max-cases", type=int, default=int(os.getenv("EVAL_MAX_CASES", "50")))
    ap.add_argument("--out-dir", default=os.getenv("EVAL_OUT_DIR", "./.eval_artifacts"))

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
    )

    os.makedirs(cfg.out_dir, exist_ok=True)

    # Dataset por tenant
    fixtures = fixtures_for_tenant(tenant_id=cfg.tenant_id, user_id=cfg.user_id)
    cases = [c.model_dump(mode="json") for c in fixtures][: cfg.max_cases]

    run_id = str(uuid4())
    out_path = os.path.join(cfg.out_dir, f"eval_run_{run_id}.json")

    results: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=cfg.timeout_s) as client:
        for i, case in enumerate(cases):
            results.append(await _run_case(client, cfg, i, case))

    payload = {
        "eval_run_id": run_id,
        "tenant_id": cfg.tenant_id,
        "base_url": cfg.base_url,
        "created_at": time.time(),
        "n": len(results),
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(out_path)
    return 0


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(main()))

