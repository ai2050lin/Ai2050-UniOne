from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]


def build_constraints() -> Dict[str, object]:
    constraints: List[Dict[str, object]] = [
        {
            "axiom": "局部图册公理",
            "equation_constraint": "Atlas_static(term) > 0 and Offset_static(term) > 0",
            "meaning": "概念身份必须先有局部家族支撑，再有家族内偏移。",
        },
        {
            "axiom": "高质量前沿公理",
            "equation_constraint": "Frontier_dynamic = f(compaction, coverage, separation)",
            "meaning": "前沿项必须由压缩、覆盖、分离三类变量共同定义，而不是单一总量定义。",
        },
        {
            "axiom": "功能子场公理",
            "equation_constraint": (
                "Subfield_dynamic = logic_prototype - logic_fragile_bridge + syntax_constraint_conflict"
            ),
            "meaning": "子场项必须显式分解正负机制，不能把大轴平均成一个量。",
        },
        {
            "axiom": "时间窗收束公理",
            "equation_constraint": "Window_closure = g(tail_pos_-9..-3)",
            "meaning": "窗口项必须显式依赖句尾前多个时间窗，而不是最后一个词元。",
        },
        {
            "axiom": "闭包边界公理",
            "equation_constraint": (
                "Closure_boundary = h(union_joint_adv, union_synergy_joint, strict_positive_synergy)"
            ),
            "meaning": "闭包边界必须同时参考联合优势、联合协同和严格正协同，而不是单指标判断。",
        },
        {
            "axiom": "分层统一公理",
            "equation_constraint": (
                "U_fit_plus = Atlas_static + Offset_static + Frontier_dynamic + "
                "Subfield_dynamic + Window_closure + Closure_boundary + "
                "Style_control + Logic_control + Syntax_control"
            ),
            "meaning": "最终主式必须把静态层、动态层和控制层显式并到同一方程中。",
        },
    ]

    return {
        "record_type": "stage56_axiom_to_equation_summary",
        "constraints": constraints,
        "proto_equation_system": {
            "master_equation": (
                "U_fit_plus(term, ctx) = Atlas_static + Offset_static + Frontier_dynamic + "
                "Subfield_dynamic + Window_closure + Closure_boundary + "
                "Style_control + Logic_control + Syntax_control"
            ),
            "closure_condition": (
                "SuccessfulClosure iff Closure_boundary > 0 and union_synergy_joint > 0 and strict_positive_synergy = 1"
            ),
            "window_condition": "Window_closure depends on tail_pos_-9..-3 rather than final_token_only",
        },
        "main_judgment": (
            "当前六条公理已经可以进入第一版方程约束层。"
            "这说明项目第一次开始从解释框架推进到约束型理论框架。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 公理到方程摘要",
        "",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Constraints",
    ]
    for row in list(summary.get("constraints", [])):
        row = dict(row)
        lines.append(
            f"- {row.get('axiom', '')}: {row.get('equation_constraint', '')} / {row.get('meaning', '')}"
        )
    lines.extend(["", "## Proto Equation System"])
    proto = dict(summary.get("proto_equation_system", {}))
    for key, value in proto.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compress the six axioms into a first equation-constraint system")
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_axiom_to_equation_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_constraints()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "record_type": summary["record_type"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
