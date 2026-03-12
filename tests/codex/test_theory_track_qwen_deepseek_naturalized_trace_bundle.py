#!/usr/bin/env python
"""
把 Qwen / DeepSeek 现有结构工件和 long-chain theorem frontier 接成统一自然化长链 trace 主包。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: List[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def main() -> None:
    t0 = time.time()
    headroom = load_json(ROOT / "tests" / "codex_temp" / "qwen_deepseek_mining_headroom_assessment_20260312.json")
    routes = load_json(ROOT / "tests" / "codex_temp" / "qwen_deepseek_improvement_routes_20260312.json")
    next_block = load_json(ROOT / "tests" / "codex_temp" / "qwen_deepseek_next_priority_block_20260312.json")
    successor = load_json(ROOT / "tests" / "codex_temp" / "theory_track_successor_strengthened_priority34_pass_fail_20260312.json")

    qwen = headroom["models"]["qwen3_4b"]
    deepseek = headroom["models"]["deepseek_7b"]
    frontier = successor["frontier_state"]
    inventory_constraints = successor["inventory_constraints"]

    cross_model_axes = {
        "family_patch": 1.0,
        "protocol_calling": float(mean([
            qwen["coverage_terms"]["protocol_calling"],
            deepseek["coverage_terms"]["protocol_calling"],
        ])),
        "relation_chain": float(mean([
            qwen["coverage_terms"]["recovery_proxy"],
            deepseek["coverage_terms"]["recovery_proxy"],
        ])),
        "online_tool_chain": float(mean([
            qwen["coverage_terms"]["hard_interface_success"],
            deepseek["coverage_terms"]["hard_interface_success"],
        ])),
        "stage_structure": clamp01(float(inventory_constraints["temporal_cross_to_within_ratio"]) - 1.0 + 0.75),
        "successor_coherence": clamp01(1.0 - float(inventory_constraints["chain_successor_to_cross_stage_ratio"])),
        "orientation_stability": float(mean([
            qwen["coverage_terms"]["orientation_stability"],
            deepseek["coverage_terms"]["orientation_stability"],
        ])),
    }

    missing_axes = [
        {"axis": k, "gap": float(1.0 - v)}
        for k, v in sorted(cross_model_axes.items(), key=lambda item: item[1])
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "core_constraint": "qwen_deepseek_naturalized_trace_bundle",
            "runtime_sec": float(time.time() - t0),
        },
        "headline_metrics": {
            "cross_model_mean_completion": float(mean([
                qwen["mining_completion_score"],
                deepseek["mining_completion_score"],
            ])),
            "cross_model_mean_headroom": float(mean([
                qwen["remaining_headroom"],
                deepseek["remaining_headroom"],
            ])),
            "strict_theorem_core_size": int(len(frontier["strengthened_theorems"])),
            "queued_theorem_count": int(len(frontier["queued_theorems"])),
        },
        "naturalized_trace_axes": cross_model_axes,
        "missing_axes": missing_axes,
        "frontier_state": frontier,
        "inventory_constraints": inventory_constraints,
        "priority_block": next_block["priority_block"],
        "all_data_excavated": False,
        "project_readout": {
            "summary": (
                "当前数据还远没有挖完。已经挖到的主要是 family patch、部分 protocol、"
                "relation/tool 在线链、阶段头和 long-chain theorem frontier，但自然长链内部 trace、"
                "更强 successor coherence、以及脑侧同构投影仍明显不足。"
            ),
            "next_question": (
                "下一步最值钱的不是再补零散工件，而是把 concept / relation / context / tool / "
                "temporal / successor 统一到同一批自然长链 trace 中。"
            ),
        },
    }

    out_path = ROOT / "tests" / "codex_temp" / "qwen_deepseek_naturalized_trace_bundle_20260312.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["missing_axes"][:5], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
