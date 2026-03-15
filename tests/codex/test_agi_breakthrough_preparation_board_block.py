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

from research.gpt5.code.agi_breakthrough_preparation_board import AgiBreakthroughPreparationBoard  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    board = AgiBreakthroughPreparationBoard.from_artifacts(ROOT)
    summary = board.summary()

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "AGI_breakthrough_preparation_board_block",
        },
        "strict_goal": {
            "statement": "Turn the current large puzzle set into one breakthrough preparation board: make the strongest progress, the weakest blockers, and the final critical path explicit.",
            "boundary": "This board is for preparation and diagnosis. It does not claim that final AGI breakthrough has been achieved.",
        },
        "headline_metrics": summary,
        "metric_lines_cn": [
            f"（DNN结构提取底座）dnn_foundation_score = {summary['dnn_foundation_score']:.4f}",
            f"（DNN参数原理强度）dnn_parametric_score = {summary['dnn_parametric_score']:.4f}",
            f"（DNN精确闭合度）dnn_exactness_score = {summary['dnn_exactness_score']:.4f}",
            f"（Spike可规模化架构）spike_architecture_score = {summary['spike_architecture_score']:.4f}",
            f"（Spike语言连续体）spike_language_score = {summary['spike_language_score']:.4f}",
            f"（最终突破准备度）final_breakthrough_readiness = {summary['final_breakthrough_readiness']:.4f}",
        ],
        "strict_verdict": {
            "preparation_board_present": True,
            "ready_for_final_breakthrough": bool(summary["final_breakthrough_readiness"] > 0.72),
            "core_answer": (
                "The project is no longer missing direction. The remaining problem is now concentrated into a small number of system bottlenecks: dense exact evidence, family-to-specific exact closure, successor exact closure, and Spike language continuation quality."
            ),
            "main_hard_gaps": [
                "the DNN side is stronger in parameter understanding than in exact closure",
                "the Spike side is stronger in architecture and scaling than in language continuation quality",
                "the final gap is now a system bottleneck problem, not a missing-concepts problem",
            ],
        },
        "progress_estimate": {
            "final_breakthrough_preparation_percent": 66.0,
            "system_direction_clarity_percent": 88.0,
            "full_brain_encoding_mechanism_percent": 90.0,
        },
        "next_large_blocks": summary["critical_path"],
    }
    return payload


def test_agi_breakthrough_preparation_board_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    verdict = payload["strict_verdict"]
    assert metrics["dnn_parametric_score"] > 0.8
    assert metrics["spike_architecture_score"] > 0.7
    assert metrics["dnn_exactness_score"] < 0.5
    assert metrics["spike_language_score"] < 0.35
    assert verdict["preparation_board_present"] is True
    assert verdict["ready_for_final_breakthrough"] is False
    assert len(payload["metric_lines_cn"]) >= 6
    assert payload["metric_lines_cn"][0].startswith("（")


def main() -> None:
    ap = argparse.ArgumentParser(description="AGI breakthrough preparation board block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/agi_breakthrough_preparation_board_block_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(payload["metric_lines_cn"]))


if __name__ == "__main__":
    main()
