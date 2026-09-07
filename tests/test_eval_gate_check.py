import json

import pytest

from scripts.eval_gate_check import score_eval_run


def _run_payload(overall_scores: list[int]) -> dict:
    return {
        "tenant_id": "tenant_test",
        "eval_mode": "strict",
        "eval_model": "gpt-4.1-mini",
        "results": [
            {
                "judge": {
                    "overall": score,
                    "faithfulness": 5,
                    "relevance": 5,
                    "citation_quality": 4,
                    "refusal_correctness": 5,
                }
            }
            for score in overall_scores
        ],
    }


def test_score_eval_run_computes_average_percentage():
    scored = score_eval_run(_run_payload([5, 4, 5, 5]))

    assert scored["n"] == 4
    assert scored["avg_overall"] == pytest.approx(4.75)
    assert scored["score_pct"] == pytest.approx(95.0)
    assert scored["tenant_id"] == "tenant_test"


def test_score_eval_run_raises_on_empty_results():
    with pytest.raises(SystemExit):
        score_eval_run({"results": []})


def test_gate_check_cli_passes_within_tolerance(tmp_path):
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps(_run_payload([5, 5, 5, 5])), encoding="utf-8")

    baseline_path = tmp_path / "eval_baseline.json"
    baseline_path.write_text(json.dumps({"score_pct": 99.0}), encoding="utf-8")

    out_path = tmp_path / "eval_result.json"

    from scripts.eval_gate_check import main
    import sys

    argv = sys.argv
    sys.argv = [
        "eval_gate_check",
        str(run_path),
        "--baseline",
        str(baseline_path),
        "--tolerance-pct",
        "2.0",
        "--out",
        str(out_path),
    ]
    try:
        exit_code = main()
    finally:
        sys.argv = argv

    assert exit_code == 0
    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert result["passed"] is True
    assert result["score_pct"] == pytest.approx(100.0)


def test_gate_check_cli_fails_below_tolerance(tmp_path):
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps(_run_payload([2, 2, 2, 2])), encoding="utf-8")

    baseline_path = tmp_path / "eval_baseline.json"
    baseline_path.write_text(json.dumps({"score_pct": 99.0}), encoding="utf-8")

    out_path = tmp_path / "eval_result.json"

    from scripts.eval_gate_check import main
    import sys

    argv = sys.argv
    sys.argv = [
        "eval_gate_check",
        str(run_path),
        "--baseline",
        str(baseline_path),
        "--tolerance-pct",
        "2.0",
        "--out",
        str(out_path),
    ]
    try:
        exit_code = main()
    finally:
        sys.argv = argv

    assert exit_code == 1
    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert result["passed"] is False
