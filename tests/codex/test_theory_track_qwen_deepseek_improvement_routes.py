#!/usr/bin/env python
"""
基于 headroom 评估，生成 Qwen / DeepSeek 下一阶段的大任务块路线图。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    t0 = time.time()
    assessment = load_json(
        ROOT / "tests" / "codex_temp" / "qwen_deepseek_mining_headroom_assessment_20260312.json"
    )
    qwen = assessment["models"]["qwen3_4b"]
    deepseek = assessment["models"]["deepseek_7b"]

    def top_gap(model: Dict[str, Any], name: str) -> float:
        for row in model["gap_ranking"]:
            if row["gap"] == name:
                return float(row["score"])
        return 0.0

    routes: Dict[str, List[Dict[str, Any]]] = {
        "qwen3_4b": [
            {
                "block": "protocol_basis_task_bridge_closure",
                "priority": 1,
                "why": "Qwen 当前最大缺口是 protocol_calling、shared_basis、task-side bridge。",
                "targets": [
                    "protocol_calling_gap",
                    "recovery_proxy_gap",
                ],
            },
            {
                "block": "orientation_consistency_and_real_readout_alignment",
                "priority": 2,
                "why": "Qwen 实际 orientation gap 不大，但仍足以限制更强 readout bridge。",
                "targets": [
                    "orientation_gap_penalty",
                    "joint_proxy_gap",
                ],
            },
            {
                "block": "long_chain_reasoning_trace_extraction",
                "priority": 3,
                "why": "Qwen 现有在线成功率高，适合进一步抽自然长链 reasoning traces。",
                "targets": [
                    "stage_head_gap",
                    "online_recovery_gap",
                ],
            },
        ],
        "deepseek_7b": [
            {
                "block": "relation_tool_online_chain_hardening",
                "priority": 1,
                "why": "DeepSeek 当前最大缺口集中在 hard interface、online recovery、joint proxy stability。",
                "targets": [
                    "hard_interface_gap",
                    "online_recovery_gap",
                    "joint_proxy_gap",
                ],
            },
            {
                "block": "orientation_rotation_and_stage_head_stabilization",
                "priority": 2,
                "why": "DeepSeek 的 orientation gap 和 stage head gap 仍明显偏大。",
                "targets": [
                    "orientation_gap_penalty",
                    "stage_head_gap",
                ],
            },
            {
                "block": "same_protocol_internal_trace_upgrade",
                "priority": 3,
                "why": "DeepSeek 还需要更高密度、同协议、长链内部工件来降低代理偏差。",
                "targets": [
                    "protocol_calling_gap",
                    "recovery_proxy_gap",
                ],
            },
        ],
    }

    cross_model_block = {
        "block": "naturalized_qwen_deepseek_long_chain_inventory_bundle",
        "priority": 0,
        "why": "两模型共同最缺的不是更多零散分数，而是真实长链 reasoning / relation / tool traces。",
        "targets": [
            "theorem_pruning",
            "P3/P4_intervention_pruning",
            "ICSPB_transport_law_closure",
        ],
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "core_constraint": "qwen_deepseek_improvement_routes",
            "runtime_sec": float(time.time() - t0),
        },
        "cross_model_priority_block": cross_model_block,
        "routes": routes,
        "headline_metrics": {
            "qwen_top_gap": float(top_gap(qwen, qwen["gap_ranking"][0]["gap"])),
            "deepseek_top_gap": float(top_gap(deepseek, deepseek["gap_ranking"][0]["gap"])),
            "qwen_remaining_headroom": float(qwen["remaining_headroom"]),
            "deepseek_remaining_headroom": float(deepseek["remaining_headroom"]),
        },
    }

    out_path = ROOT / "tests" / "codex_temp" / "qwen_deepseek_improvement_routes_20260312.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["cross_model_priority_block"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
