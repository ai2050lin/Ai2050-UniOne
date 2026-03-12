#!/usr/bin/env python
"""
生成 Qwen / DeepSeek 下一阶段统一优先级任务块。
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
    headroom = load_json(ROOT / "tests" / "codex_temp" / "qwen_deepseek_mining_headroom_assessment_20260312.json")
    routes = load_json(ROOT / "tests" / "codex_temp" / "qwen_deepseek_improvement_routes_20260312.json")

    qwen = headroom["models"]["qwen3_4b"]
    deepseek = headroom["models"]["deepseek_7b"]

    priority_block: List[Dict[str, Any]] = [
        {
            "priority": 1,
            "block": "cross_model_naturalized_long_chain_trace_bundle",
            "why": routes["cross_model_priority_block"]["why"],
            "expected_value": "为 ICSPB theorem、A(I)、M_feas(I) 和 P3/P4 intervention 提供统一长链统计约束。",
        },
        {
            "priority": 2,
            "block": "qwen_protocol_basis_task_bridge_bundle",
            "why": routes["routes"]["qwen3_4b"][0]["why"],
            "expected_value": "补强 Qwen 的 protocol / basis / task-bridge，减少其相对 DeepSeek 的结构侧欠挖。",
        },
        {
            "priority": 3,
            "block": "deepseek_relation_tool_online_chain_bundle",
            "why": routes["routes"]["deepseek_7b"][0]["why"],
            "expected_value": "直打 DeepSeek relation/tool 在线链脆弱区，是当前 DeepSeek 侧最大提升空间。",
        },
        {
            "priority": 4,
            "block": "same_protocol_internal_trace_upgrade_bundle",
            "why": routes["routes"]["deepseek_7b"][2]["why"],
            "expected_value": "降低 DeepSeek 代理偏差，让 Qwen / DeepSeek 同协议内部 trace 更可比较。",
        },
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "core_constraint": "qwen_deepseek_next_priority_block",
            "runtime_sec": float(time.time() - t0),
        },
        "headline_metrics": {
            "qwen_remaining_headroom": float(qwen["remaining_headroom"]),
            "deepseek_remaining_headroom": float(deepseek["remaining_headroom"]),
            "deepseek_minus_qwen_headroom": float(deepseek["remaining_headroom"] - qwen["remaining_headroom"]),
        },
        "priority_block": priority_block,
    }

    out_path = ROOT / "tests" / "codex_temp" / "qwen_deepseek_next_priority_block_20260312.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["priority_block"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
