#!/usr/bin/env python
"""
给出下一阶段 Qwen / DeepSeek 自然化长链 trace 主计划。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    t0 = time.time()
    bundle = load_json(ROOT / "tests" / "codex_temp" / "qwen_deepseek_naturalized_trace_bundle_20260312.json")
    analysis_space = load_json(ROOT / "tests" / "codex_temp" / "qwen_deepseek_systematic_analysis_space_20260312.json")

    priority_block = [
        {
            "priority": 1,
            "block": "cross_model_real_long_chain_trace_capture",
            "why": "当前最大公共缺口是自然长链 trace 本身还不够。",
            "targets": ["protocol_calling", "online_tool_chain", "stage_structure", "successor_coherence"],
        },
        {
            "priority": 2,
            "block": "deepseek_relation_tool_hardening_with_same_protocol_trace",
            "why": "DeepSeek 剩余空间更大，且当前硬伤集中在 relation/tool 在线链。",
            "targets": ["hard_interface_gap", "online_recovery_gap", "orientation_gap_penalty"],
        },
        {
            "priority": 3,
            "block": "qwen_protocol_basis_task_bridge_deepening",
            "why": "Qwen 当前主要欠挖 protocol / basis / task-bridge 深层联系。",
            "targets": ["protocol_calling_gap", "recovery_proxy_gap"],
        },
        {
            "priority": 4,
            "block": "icspb_theorem_pruning_with_long_chain_invariants",
            "why": "把自然长链 trace 直接反压到 theorem、A(I)、M_feas(I)、P3/P4 intervention。",
            "targets": ["strict_theorem_core", "queued_theorems", "intervention_pruning"],
        },
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "core_constraint": "qwen_deepseek_naturalized_trace_master_plan",
            "runtime_sec": float(time.time() - t0),
        },
        "headline_metrics": {
            "cross_model_mean_headroom": float(bundle["headline_metrics"]["cross_model_mean_headroom"]),
            "strict_theorem_core_size": int(bundle["headline_metrics"]["strict_theorem_core_size"]),
            "analysis_method_count": int(analysis_space["headline_metrics"]["analysis_method_count"]),
        },
        "priority_block": priority_block,
    }

    out_path = ROOT / "tests" / "codex_temp" / "qwen_deepseek_naturalized_trace_master_plan_20260312.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["priority_block"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
