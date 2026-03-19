from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]


def build_spiking_bridge() -> Dict[str, object]:
    mappings: List[Dict[str, object]] = [
        {
            "current_term": "Atlas_static",
            "spiking_view": "长时间尺度上的稳定放电簇与可重复微回路",
            "meaning": "静态家族片区可看成长期稳定吸引域，而不是一次性瞬时激活。",
        },
        {
            "current_term": "Offset_static",
            "spiking_view": "同一吸引域内的相位偏移与细粒度时序差",
            "meaning": "概念偏移可对应同族脉冲簇内部的细结构差分。",
        },
        {
            "current_term": "Frontier_dynamic",
            "spiking_view": "高放电密度前沿与阈值附近的优先生长路径",
            "meaning": "密度前沿可重写成高强度脉冲传播最先占优的边界。",
        },
        {
            "current_term": "Subfield_dynamic",
            "spiking_view": "功能性微回路团簇",
            "meaning": "内部子场可看成承担不同功能的脉冲回路族，而不是单纯抽象标签。",
        },
        {
            "current_term": "Window_closure",
            "spiking_view": "时间窗内的同步收束",
            "meaning": "词元窗口闭包可映射成多个时间窗中的同步化与抑制竞争。",
        },
        {
            "current_term": "Closure_boundary",
            "spiking_view": "吸引域锁定边界与失败逸出边界",
            "meaning": "闭包成功或失败可对应脉冲动力系统是否进入稳定吸引域。",
        },
    ]

    optimizations = [
        "把总量变量改写成时间窗积分变量，而不是单步平均量",
        "把内部子场改写成兴奋-抑制微回路的有效连接模式",
        "把密度前沿改写成阈值附近的优先传播边界",
        "把闭包量改写成吸引域进入概率与稳定保持时间",
        "把控制轴改写成对放电节律、相位同步和竞争抑制的调制项",
    ]

    return {
        "record_type": "stage56_spiking_math_bridge_summary",
        "main_judgment": (
            "如果从脉冲神经网络角度优化当前数学体系，关键不是把现有变量全部推翻，"
            "而是把它们重写成时间窗、同步、竞争、吸引域和兴奋-抑制回路上的有效变量。"
        ),
        "mappings": mappings,
        "optimizations": optimizations,
        "proto_spiking_equation": (
            "U_spike = Attractor_static + Phase_offset + Propagation_frontier + "
            "Circuit_subfield + Synchrony_window + Basin_boundary"
        ),
        "answer_to_user_question": (
            "从脉冲神经网络视角，当前体系最值得优化的不是名称，而是变量底座："
            "把空间平均变量改成时序变量，把抽象子场改成微回路变量，把闭包改成吸引域进入与保持。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 脉冲神经网络桥接摘要",
        "",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        f"- proto_spiking_equation: {summary.get('proto_spiking_equation', '')}",
        "",
        "## Mappings",
    ]
    for row in list(summary.get("mappings", [])):
        row = dict(row)
        lines.append(
            f"- {row.get('current_term', '')}: {row.get('spiking_view', '')} / {row.get('meaning', '')}"
        )
    lines.extend(["", "## Optimizations"])
    for row in list(summary.get("optimizations", [])):
        lines.append(f"- {row}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bridge the current math system to a spiking-neural-network view")
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_spiking_math_bridge_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_spiking_bridge()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "record_type": summary["record_type"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
