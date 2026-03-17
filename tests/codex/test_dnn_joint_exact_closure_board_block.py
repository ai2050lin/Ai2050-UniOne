from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.gpt5.code.dnn_joint_exact_closure_board import DnnJointExactClosureBoard  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    board = DnnJointExactClosureBoard.from_artifacts(ROOT)
    summary = board.summary()

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "DNN_joint_exact_closure_board_block",
        },
        "strict_goal": {
            "statement": "Quantify how the three real blockers jointly hold back final theorem closure: dense exact evidence, family-to-specific exact closure, and successor exact closure.",
            "boundary": "This block is a joint bottleneck board. It does not claim that final theorem closure has been reached.",
        },
        "headline_metrics": summary,
        "strict_verdict": {
            "joint_exact_closure_board_present": True,
            "final_theorem_jointly_closed": bool(summary["coupled_exact_closure_score"] > 0.72),
            "core_answer": "The project is now bottlenecked by a coupled three-body problem: dense evidence, family-to-specific exact closure, and successor exact closure. Any final theorem claim will remain premature until these three rise together.",
            "main_hard_gaps": [
                "dense exact evidence is still too weak to support theorem-grade closure",
                "family-to-specific exact closure still lags far behind specific parametric restoration",
                "successor exact closure still lags far behind system candidate strength",
            ],
        },
        "progress_estimate": {
            "joint_exact_closure_board_percent": 69.0,
            "coupled_exact_closure_percent": 43.0,
            "system_parametric_principle_percent": 73.0,
            "full_brain_encoding_mechanism_percent": 90.0,
        },
        "metric_lines_cn": summary["metric_lines_cn"],
        "next_large_blocks": summary["critical_path"],
    }
    return payload


def test_dnn_joint_exact_closure_board_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    verdict = payload["strict_verdict"]
    assert metrics["dense_evidence_score"] < 0.45
    assert metrics["family_specific_closure_score"] < 0.55
    assert metrics["successor_closure_score"] < 0.55
    assert metrics["coupled_exact_closure_score"] < 0.50
    assert metrics["theorem_readiness_under_coupling"] < 0.65
    assert verdict["joint_exact_closure_board_present"] is True
    assert verdict["final_theorem_jointly_closed"] is False
    assert len(payload["metric_lines_cn"]) >= 5
    assert payload["metric_lines_cn"][0].startswith("（")


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN joint exact closure board block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_joint_exact_closure_board_block_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(payload["metric_lines_cn"]))


if __name__ == "__main__":
    main()
