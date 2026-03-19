from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from stage56_fullsample_regression_runner import read_json

ROOT = Path(__file__).resolve().parents[2]


def build_summary(
    generalized_summary: Dict[str, object],
    load_summary: Dict[str, object],
) -> Dict[str, object]:
    load_signs = dict(load_summary.get("sign_matrix", {}))
    generalized_formulas = dict(generalized_summary.get("generalized_formulas", {}))

    refined_formulas = {
        "general_state_observation": (
            "y_general(pair) ~= a * G(pair) + d * D(pair) + p * gd(pair) - l * L_base(pair) + eps(pair)"
        ),
        "strict_state_observation": (
            "y_strict(pair) ~= y_general(pair) + s * L_select(pair) + eta(pair)"
        ),
        "base_load_operator": "L_base(pair) = (gs(pair) + sd(pair)) / 2",
        "strict_select_operator": "L_select(pair) = (sd(pair) - gs(pair)) / 2",
        "operator_form_judgment": (
            "当前更一般化的系统公式已经可以从“层级状态 + 通道向量 + 目标条件负载算子”"
            "继续压成“基础负载算子 + 严格选择算子”的双算子结构。"
        ),
    }

    return {
        "record_type": "stage56_generalized_formula_refinement_summary",
        "generalized_formulas": generalized_formulas,
        "load_operator_signs": load_signs,
        "refined_formulas": refined_formulas,
        "main_judgment": (
            "系统级一般化公式现在可以进一步缩短成："
            "一般目标由 G、D、gd 和基础负载算子共同决定，"
            "严格目标则在一般目标之上额外叠加严格选择算子。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 一般化公式精炼摘要",
        "",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Refined Formulas",
    ]
    for key, value in dict(summary.get("refined_formulas", {})).items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Refine the generalized system formula with load operators")
    ap.add_argument(
        "--generalized-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_system_generalized_formula_20260320" / "summary.json"),
    )
    ap.add_argument(
        "--load-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_load_operator_closure_20260320" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_generalized_formula_refinement_20260320"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    generalized_summary = read_json(Path(args.generalized_summary_json))
    load_summary = read_json(Path(args.load_summary_json))
    summary = build_summary(generalized_summary, load_summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
