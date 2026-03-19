from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]


def build_outline() -> Dict[str, object]:
    principles: List[Dict[str, object]] = [
        {
            "name": "单一微观机制不等于简单宏观理论",
            "claim": (
                "兴奋神经元、抑制神经元、脉冲这类低层机制可以很单一，"
                "但在大规模重复、层级堆叠、反馈回路和时间展开后，"
                "会长出新的中观和宏观有效变量。"
            ),
            "implication": "当前理论变复杂，不是因为底层规则复杂，而是因为我们仍在寻找正确的粗粒化变量。",
        },
        {
            "name": "静态本体层",
            "claim": "family patch（家族片区） + concept offset（概念偏移） 更像局部图册和局部坐标偏移。",
            "implication": "这一层适合成为一般数学体系里的静态结构层。",
        },
        {
            "name": "动态生成层",
            "claim": "密度前沿 + 内部子场 + 词元窗口 + 闭包量 更像分层动力系统中的状态场、子通道和收束量。",
            "implication": "这一层适合成为一般数学体系里的演化与闭包层。",
        },
        {
            "name": "一般数学体系候选",
            "claim": (
                "如果当前成果继续收敛，它有机会变成一个比单一现有学科更一般的数学体系："
                "局部图册 + 分层场 + 受控动力系统 + 拓扑闭包。"
            ),
            "implication": "这不是简单替换成某一门现成数学，而是建立新的统一变量体系。",
        },
    ]

    axioms = [
        "局部身份由家族片区与概念偏移给出",
        "高质量表示由密度前沿而不是非零集合决定",
        "真正起作用的是轴内部子场，而不是轴标签本身",
        "生成机制在句尾前窗口完成主要收束",
        "闭包量决定表示是否真正稳定成立",
    ]

    return {
        "record_type": "stage56_general_math_system_outline",
        "main_answer_1": (
            "当前研究支持：大脑可以由相对单一的低层神经机制运行，"
            "但在系统层面长出更复杂的有效结构；因此复杂理论与简单底层机制并不矛盾。"
        ),
        "main_answer_2": (
            "当前研究也支持：这些成果有机会继续被整理成更一般的数学体系，"
            "前提是把静态本体层与动态生成层压到同一套可计算变量和可判伪方程中。"
        ),
        "principles": principles,
        "proto_axioms": axioms,
        "candidate_unified_form": (
            "U = Atlas_static + Field_dynamic + Control_evolution + Closure_boundary"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 一般数学体系整理摘要",
        "",
        f"- main_answer_1: {summary.get('main_answer_1', '')}",
        f"- main_answer_2: {summary.get('main_answer_2', '')}",
        "",
        "## Principles",
    ]
    for row in list(summary.get("principles", [])):
        row = dict(row)
        lines.append(f"- {row.get('name', '')}: {row.get('claim', '')} / {row.get('implication', '')}")
    lines.extend(["", "## Proto Axioms"])
    for row in list(summary.get("proto_axioms", [])):
        lines.append(f"- {row}")
    lines.extend(["", "## Unified Form", f"- {summary.get('candidate_unified_form', '')}"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Outline how current results could become a more general math system")
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_general_math_system_outline_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_outline()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "record_type": summary["record_type"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
