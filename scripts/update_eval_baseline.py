from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from scripts.eval_gate_check import score_eval_run


def _commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Overwrite eval_baseline.json with the score from a real eval run, "
        "after manual confirmation. Use this when a change intentionally improves the system."
    )
    ap.add_argument("eval_run_json", help="Path to a eval_run_<id>.json produced by scripts/eval_runner.py")
    ap.add_argument("--baseline", default="eval_baseline.json")
    ap.add_argument("--yes", action="store_true", help="Skip the interactive confirmation prompt")
    args = ap.parse_args()

    run_payload = json.loads(Path(args.eval_run_json).read_text(encoding="utf-8"))
    scored = score_eval_run(run_payload)

    baseline_path = Path(args.baseline)
    old_score_pct = None
    if baseline_path.exists():
        old_score_pct = json.loads(baseline_path.read_text(encoding="utf-8")).get("score_pct")

    print(f"New score: {scored['score_pct']:.2f}% (avg_overall={scored['avg_overall']:.2f}/5, n={scored['n']})")
    if old_score_pct is not None:
        print(f"Current baseline: {old_score_pct:.2f}%")

    if not args.yes:
        answer = input(f"Overwrite {baseline_path} with the new score? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted, baseline not changed.")
            return 1

    new_baseline = {
        "score_pct": scored["score_pct"],
        "avg_overall": scored["avg_overall"],
        "n": scored["n"],
        "tenant_id": scored["tenant_id"],
        "eval_mode": scored["eval_mode"],
        "eval_model": scored["eval_model"],
        "per_metric_avg": scored["per_metric_avg"],
        "commit_sha": _commit_sha(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    baseline_path.write_text(json.dumps(new_baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {baseline_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
