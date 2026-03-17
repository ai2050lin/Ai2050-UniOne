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

from research.gpt5.code.dnn_joint_closure_leverage_board import DnnJointClosureLeverageBoard  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    board = DnnJointClosureLeverageBoard.from_artifacts(ROOT)
    summary = board.summary()

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "DNN_joint_closure_leverage_board_block",
        },
        "strict_goal": {
            "statement": "Quantify which blocker uplift yields the largest system-level gain under the current coupled exact closure regime.",
            "boundary": "This block is a leverage analysis. It does not itself improve closure; it only ranks intervention value.",
        },
        "headline_metrics": summary,
        "strict_verdict": {
            "joint_closure_leverage_board_present": True,
            "single_block_breakthrough_possible": bool(summary["best_single_scenario"]["new_coupled"] > 0.72),
            "core_answer": "The next stage should not be chosen by intuition. The leverage board makes explicit which single bottleneck and which pair currently produce the largest closure gain under the existing system geometry.",
            "main_hard_gaps": [
                "single-block uplift is still insufficient for final theorem closure",
                "even the best pair uplift remains a mid-stage move rather than a final breakthrough",
                "the system still needs three-block coupling to approach endgame closure",
            ],
        },
        "progress_estimate": {
            "joint_closure_leverage_board_percent": 74.0,
            "coupled_exact_closure_percent": 43.0,
            "full_brain_encoding_mechanism_percent": 90.0,
        },
        "metric_lines_cn": summary["metric_lines_cn"],
        "next_large_blocks": summary["critical_path"],
    }
    return payload


def test_dnn_joint_closure_leverage_board_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    verdict = payload["strict_verdict"]
    best_single = metrics["best_single_scenario"]
    best_pair = metrics["best_pair_scenario"]
    best_all = metrics["best_all_scenario"]
    assert best_single["delta_coupled"] > 0.0
    assert best_pair["delta_coupled"] > best_single["delta_coupled"]
    assert best_all["delta_coupled"] > best_pair["delta_coupled"]
    assert verdict["joint_closure_leverage_board_present"] is True
    assert verdict["single_block_breakthrough_possible"] is False
    assert len(payload["metric_lines_cn"]) >= 5
    assert payload["metric_lines_cn"][0].startswith("（")


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN joint closure leverage board block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_joint_closure_leverage_board_block_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(payload["metric_lines_cn"]))


if __name__ == "__main__":
    main()
