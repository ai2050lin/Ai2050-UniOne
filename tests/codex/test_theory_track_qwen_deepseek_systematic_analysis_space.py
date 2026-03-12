#!/usr/bin/env python
"""
生成基于现有数据的系统分析空间图：还能用哪些分析方式更系统化地理解语言编码机制和数学原理。
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
    bundle = load_json(ROOT / "tests" / "codex_temp" / "qwen_deepseek_naturalized_trace_bundle_20260312.json")

    axes = bundle["naturalized_trace_axes"]
    missing = {row["axis"]: float(row["gap"]) for row in bundle["missing_axes"]}

    analysis_space: List[Dict[str, Any]] = [
        {
            "method": "family_patch_statistics",
            "target": "对象基底、family patch、cross-family 边界",
            "strength": float(axes["family_patch"]),
            "next_value": "继续确认 object atlas 是否是主骨架，而不是局部假象。",
        },
        {
            "method": "protocol_calling_analysis",
            "target": "concept -> protocol field / field boundary / task-bridge",
            "strength": float(axes["protocol_calling"]),
            "next_value": "直接回答语言概念如何从 object patch 进入可用协议层。",
        },
        {
            "method": "relation_tool_online_chain_analysis",
            "target": "relation / tool 在线链、hard interface、rollback-recovery",
            "strength": float(axes["online_tool_chain"]),
            "next_value": "回答推理和工具使用中的 relation 链是如何在编码里被组织起来的。",
        },
        {
            "method": "stage_successor_trace_analysis",
            "target": "temporal stage、successor coherence、long-chain transport",
            "strength": float(min(axes["stage_structure"], max(0.0, 1.0 - missing["successor_coherence"]))),
            "next_value": "回答推理过程本身如何被编码，而不只是概念和关系如何被编码。",
        },
        {
            "method": "orientation_gap_analysis",
            "target": "concept-led vs relation-led、orientation rotation、readout mismatch",
            "strength": float(axes["orientation_stability"]),
            "next_value": "解释为什么不同模型会在同一编码主干上分化出不同取向。",
        },
        {
            "method": "theorem_pruning_and_survival",
            "target": "ICSPB theorem pruning / survival / intervention binding",
            "strength": float(bundle["headline_metrics"]["strict_theorem_core_size"] / 6.0),
            "next_value": "把统计规律压成可失败的数学理论，而不是停在描述层。",
        },
    ]

    analysis_space.sort(key=lambda row: row["strength"], reverse=True)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "core_constraint": "qwen_deepseek_systematic_analysis_space",
            "runtime_sec": float(time.time() - t0),
        },
        "analysis_space": analysis_space,
        "headline_metrics": {
            "analysis_method_count": len(analysis_space),
            "top_method": analysis_space[0]["method"],
            "weakest_method": analysis_space[-1]["method"],
        },
        "project_readout": {
            "summary": (
                "更系统化理解语言背后的编码机制，不能只做结构图，也不能只做在线 benchmark。"
                "最有效的是把对象 patch、protocol 调用、relation/tool 在线链、"
                "stage/successor 长链 trace、orientation gap、theorem survival 统一到同一张分析空间图里。"
            ),
            "next_question": (
                "后续不是问‘再看什么数据’，而是问‘哪些分析方法能把已有数据压成更强的编码与数学不变量’。"
            ),
        },
    }

    out_path = ROOT / "tests" / "codex_temp" / "qwen_deepseek_systematic_analysis_space_20260312.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["analysis_space"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
