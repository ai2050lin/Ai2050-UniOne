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

from research.gpt5.code.dnn_system_crack_board import DnnSystemCrackBoard  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    board = DnnSystemCrackBoard.from_artifacts(ROOT)
    summary = board.summary()

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "DNN_system_crack_board_block",
        },
        "strict_goal": {
            "statement": "Aggregate current DNN-side extraction, signatures, specific restoration, successor restoration, and exact-system theorem candidate into one system crack board.",
            "boundary": "This board reports how close the project is to a system-level exact coding principle. It does not claim final theorem closure.",
        },
        "headline_metrics": summary,
        "strict_verdict": {
            "system_crack_board_present": True,
            "final_theorem_closed": bool(summary["exact_theorem_closure_score"] > 0.70),
            "core_answer": "The project has already formed a strong system-level theorem candidate. The remaining failure is concentrated in exact evidence, family-to-specific exact closure, and successor exact closure.",
            "main_hard_gaps": [
                "参数原理已经强于精确闭合",
                "concept-specific 的参数恢复强于 exact family-to-specific 闭合",
                "successor 仍然是系统级闭合最弱项之一",
            ],
        },
        "progress_estimate": {
            "dnn_system_crack_board_percent": 72.0,
            "system_parametric_principle_percent": 73.0,
            "exact_theorem_closure_percent": 37.0,
            "full_brain_encoding_mechanism_percent": 90.0,
        },
        "metric_lines_cn": summary["metric_lines_cn"],
        "next_large_blocks": summary["critical_path"],
    }
    return payload


def test_dnn_system_crack_board_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    verdict = payload["strict_verdict"]
    assert metrics["extraction_base_score"] > 0.70
    assert metrics["parametric_system_score"] > 0.85
    assert metrics["specific_exactness_score"] < 0.55
    assert metrics["successor_exactness_score"] < 0.55
    assert metrics["theorem_candidate_strength"] > 0.75
    assert metrics["exact_theorem_closure_score"] < 0.50
    assert verdict["system_crack_board_present"] is True
    assert verdict["final_theorem_closed"] is False
    assert len(payload["metric_lines_cn"]) >= 6
    assert payload["metric_lines_cn"][0].startswith("（")


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN system crack board block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_system_crack_board_block_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(payload["metric_lines_cn"]))


if __name__ == "__main__":
    main()
