from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]


def build_axiom_explanations() -> Dict[str, object]:
    axioms: List[Dict[str, object]] = [
        {
            "name": "局部图册公理",
            "core_principle": (
                "任一概念不能直接在全局单一空间里被完整定义，而要先落在某个局部家族区域中，"
                "再在该局部区域内部定义它自己的身份。"
            ),
            "why_it_appears": (
                "因为当前结果不断显示：概念之间更像先形成家族性聚类，再在同族内部出现稳定偏移，"
                "这比“所有概念都在一个均匀全局空间里平等分布”更符合数据。"
            ),
            "current_evidence": [
                "family patch（家族片区）",
                "concept offset（概念偏移）",
                "局部线性切片只占少数，大多数结构更像路径束与局部图册",
            ],
            "math_meaning": (
                "它更接近图册、局部坐标和局部截面的思想，而不是单一全局坐标系。"
            ),
        },
        {
            "name": "高质量前沿公理",
            "core_principle": (
                "真正决定解释力的不是所有参与神经元的并集，而是高质量、高密度、最有判别力的前沿如何压缩、分离和汇合。"
            ),
            "why_it_appears": (
                "因为在放开有效神经元数量后，全网几乎都可参与，"
                "所以“是否参与”失去区分力，只剩高质量前沿的形状还有解释力。"
            ),
            "current_evidence": [
                "broad_support_base（广支撑底座）",
                "long_separation_frontier（长期分离前沿）",
                "pair_compaction_middle_mean / pair_coverage_middle_mean（成对中段压缩均值 / 覆盖均值）",
            ],
            "math_meaning": (
                "它要求把变量从“神经元集合”升级成“密度场与前沿几何”。"
            ),
        },
        {
            "name": "功能子场公理",
            "core_principle": (
                "真正执行功能的不是 style / logic / syntax（风格 / 逻辑 / 句法）这些大轴标签本身，"
                "而是它们内部的功能性细分机制。"
            ),
            "why_it_appears": (
                "因为同一个控制轴里会同时包含正项、负项、浅桥接、骨架化、筛选等不同机制，"
                "如果不拆子场，很多方向会互相抵消。"
            ),
            "current_evidence": [
                "logic_prototype（逻辑原型）",
                "logic_fragile_bridge（逻辑脆弱桥接）",
                "syntax_constraint_conflict（句法约束型冲突）",
            ],
            "math_meaning": (
                "它要求把轴级变量分解成更细的场、子空间或微机制通道。"
            ),
        },
        {
            "name": "时间窗收束公理",
            "core_principle": (
                "系统的主要收束不发生在最后一个词元本身，而发生在句尾前若干词元窗口中的连续竞争、筛选与同步。"
            ),
            "why_it_appears": (
                "因为现有结果反复表明：最后一个词元更多像读出点，"
                "真正的骨架预收束与句法筛选发生在更早的尾部窗口。"
            ),
            "current_evidence": [
                "tail_pos_-9..-8（倒数第9到第8词元）",
                "tail_pos_-8..-5（倒数第8到第5词元）",
                "tail_pos_-6..-3（倒数第6到第3词元）",
            ],
            "math_meaning": (
                "它要求主方程显式包含时间窗变量，而不能只用静态层均值。"
            ),
        },
        {
            "name": "闭包边界公理",
            "core_principle": (
                "语言输出是否成立，不是某些局部特征是否出现，而是系统是否跨过闭包边界，进入稳定联合表示。"
            ),
            "why_it_appears": (
                "因为有联合路由并不等于真正形成闭包；当前严格正协同仍然是少数成功态。"
            ),
            "current_evidence": [
                "union_joint_adv（联合优势）",
                "union_synergy_joint（联合协同）",
                "strict_positive_synergy（严格正协同）",
            ],
            "math_meaning": (
                "它更像边界、相变和吸引域进入问题，而不是普通连续平均量问题。"
            ),
        },
        {
            "name": "分层统一公理",
            "core_principle": (
                "静态本体层、动态生成层和控制层不能各自独立描述，必须被组织进同一套变量系统。"
            ),
            "why_it_appears": (
                "因为如果只看静态层，解释不了生成；只看动态层，又解释不了概念本体；"
                "只看控制层，会把语言现象误当成本体。"
            ),
            "current_evidence": [
                "统一主方程",
                "控制轴并场",
                "静态项实证估计",
            ],
            "math_meaning": (
                "它要求最终理论既能描述局部身份，也能描述时间演化和闭包边界。"
            ),
        },
    ]

    return {
        "record_type": "stage56_axiom_explainer_summary",
        "axioms": axioms,
        "main_judgment": (
            "这六条公理不是为了换一种说法，而是为了把当前反复出现的稳定结构压成更少、"
            "更接近高阶数学体系的核心约束。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 公理解释摘要",
        "",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Axioms",
    ]
    for row in list(summary.get("axioms", [])):
        row = dict(row)
        evidence = "；".join(row.get("current_evidence", []))
        lines.extend(
            [
                f"- {row.get('name', '')}",
                f"  core_principle: {row.get('core_principle', '')}",
                f"  why_it_appears: {row.get('why_it_appears', '')}",
                f"  current_evidence: {evidence}",
                f"  math_meaning: {row.get('math_meaning', '')}",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Explain the six higher-order math axioms in the current research line")
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_axiom_explainer_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_axiom_explanations()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "record_type": summary["record_type"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
