from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

METRICS = ["faithfulness", "relevance", "citation_quality", "refusal_correctness"]


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def score_eval_run(run_payload: dict[str, Any]) -> dict[str, Any]:
    results = run_payload.get("results", [])
    if not results:
        raise SystemExit("eval run JSON has no results — nothing to score")

    overall_scores = [float(r["judge"]["overall"]) for r in results]
    avg_overall = _avg(overall_scores)

    per_metric = {m: _avg([float(r["judge"].get(m, 0)) for r in results]) for m in METRICS}

    return {
        "n": len(results),
        "avg_overall": avg_overall,
        "score_pct": avg_overall / 5.0 * 100.0,
        "per_metric_avg": per_metric,
        "tenant_id": run_payload.get("tenant_id"),
        "eval_mode": run_payload.get("eval_mode"),
        "eval_model": run_payload.get("eval_model"),
    }


def _commit_sha(explicit: str | None) -> str:
    if explicit:
        return explicit
    env_sha = os.getenv("GITHUB_SHA")
    if env_sha:
        return env_sha
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description="Score an eval run and gate it against eval_baseline.json")
    ap.add_argument("eval_run_json", help="Path to a eval_run_<id>.json produced by scripts/eval_runner.py")
    ap.add_argument("--baseline", default="eval_baseline.json")
    ap.add_argument("--tolerance-pct", type=float, default=float(os.getenv("EVAL_TOLERANCE_PCT", "2.0")))
    ap.add_argument("--out", default="eval_result.json")
    ap.add_argument("--commit-sha", default=None)
    args = ap.parse_args()

    run_payload = json.loads(Path(args.eval_run_json).read_text(encoding="utf-8"))
    scored = score_eval_run(run_payload)

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        raise SystemExit(f"Missing baseline file: {baseline_path}. Run scripts/update_eval_baseline.py first.")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_score_pct = float(baseline["score_pct"])

    threshold_pct = baseline_score_pct - args.tolerance_pct
    passed = scored["score_pct"] >= threshold_pct

    result = {
        **scored,
        "commit_sha": _commit_sha(args.commit_sha),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline_score_pct": baseline_score_pct,
        "tolerance_pct": args.tolerance_pct,
        "threshold_pct": threshold_pct,
        "delta_pct": scored["score_pct"] - baseline_score_pct,
        "passed": passed,
    }

    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    status = "PASS" if passed else "FAIL"
    print(
        f"[{status}] score={scored['score_pct']:.2f}% (avg_overall={scored['avg_overall']:.2f}/5, n={scored['n']}) "
        f"baseline={baseline_score_pct:.2f}% threshold={threshold_pct:.2f}% delta={result['delta_pct']:+.2f}pp"
    )

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
