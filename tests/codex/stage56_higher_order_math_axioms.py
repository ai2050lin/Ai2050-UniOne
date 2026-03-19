from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]


def build_axioms() -> Dict[str, object]:
    axioms: List[Dict[str, object]] = [
        {
            "name": "局部图册公理",
            "statement": "任一概念必须先落在某个局部家族图册中，然后才能定义家族内偏移。",
            "current_support": "family patch（家族片区） + concept offset（概念偏移）",
        },
        {
            "name": "高质量前沿公理",
            "statement": "真正有解释力的不是全网参与集合，而是高质量前沿的压缩、分离与汇合方式。",
            "current_support": "密度前沿（density frontier，密度前沿）",
        },
        {
            "name": "功能子场公理",
            "statement": "轴标签本身不执行功能，真正执行功能的是轴内部的子场或微机制。",
            "current_support": "logic_prototype / logic_fragile_bridge / syntax_constraint_conflict",
        },
        {
            "name": "时间窗收束公理",
            "statement": "主要收束不发生在最后一个词元，而发生在句尾前若干窗口中的竞争与筛选。",
            "current_support": "词元窗口（token window，词元窗口）",
        },
        {
            "name": "闭包边界公理",
            "statement": "语言输出是否成立，取决于系统是否跨过闭包边界进入稳定联合表示。",
            "current_support": "union_joint_adv / union_synergy_joint / strict_positive_synergy",
        },
        {
            "name": "分层统一公理",
            "statement": "静态本体层、动态生成层和控制层必须被组织进同一变量系统。",
            "current_support": "统一主方程与控制轴并场",
        },
    ]

    return {
        "record_type": "stage56_higher_order_math_axioms_summary",
        "main_judgment": (
            "如果当前成果要进一步上升为更一般的数学体系，最合理的下一步不是先找一个现成学科替代，"
            "而是先把当前稳定出现的结构压成公理组。"
        ),
        "axioms": axioms,
        "proto_theorem_direction": (
            "若一个系统同时满足局部图册、公理化前沿、功能子场、时间窗收束和闭包边界，"
            "则该系统应允许从局部身份到全局输出的分层可组合表示。"
        ),
        "math_possibility_answer": (
            "当前成果已经开始支持更高阶数学体系的可能性，但更现实的路径是先形成公理组，再讨论它与群论、拓扑学、纤维束或动力系统的严格对应。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 更高阶数学体系公理草案",
        "",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        f"- proto_theorem_direction: {summary.get('proto_theorem_direction', '')}",
        "",
        "## Axioms",
    ]
    for row in list(summary.get("axioms", [])):
        row = dict(row)
        lines.append(
            f"- {row.get('name', '')}: {row.get('statement', '')} / {row.get('current_support', '')}"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Draft a first axiom set for a more general math system behind language")
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_higher_order_math_axioms_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_axioms()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "record_type": summary["record_type"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
