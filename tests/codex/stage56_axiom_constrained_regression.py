from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from stage56_fullsample_regression_runner import fit_linear_regression, read_json, safe_float

ROOT = Path(__file__).resolve().parents[2]


FEATURE_NAMES = [
    "atlas_axiom_feature",
    "offset_axiom_feature",
    "frontier_axiom_feature",
    "subfield_axiom_feature",
    "window_axiom_feature",
    "control_axiom_feature",
]


def project_nonnegative(value: float) -> float:
    return value if value > 0.0 else 0.0


def build_axiom_rows(design_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in design_rows:
        atlas = safe_float(row.get("atlas_static_proxy"))
        offset = safe_float(row.get("offset_static_proxy"))
        frontier = safe_float(row.get("frontier_dynamic_proxy"))
        logic_prototype = safe_float(row.get("logic_prototype_proxy"))
        logic_fragile_bridge = safe_float(row.get("logic_fragile_bridge_proxy"))
        syntax_constraint_conflict = safe_float(row.get("syntax_constraint_conflict_proxy"))
        window_hidden = safe_float(row.get("window_hidden_proxy"))
        window_mlp = safe_float(row.get("window_mlp_proxy"))
        style_control = safe_float(row.get("style_control_proxy"))
        logic_control = safe_float(row.get("logic_control_proxy"))
        syntax_control = safe_float(row.get("syntax_control_proxy"))

        out.append(
            {
                **dict(row),
                "atlas_axiom_feature": project_nonnegative(atlas),
                "offset_axiom_feature": project_nonnegative(offset),
                "frontier_axiom_feature": project_nonnegative(frontier),
                "subfield_axiom_feature": project_nonnegative(logic_prototype)
                + project_nonnegative(syntax_constraint_conflict)
                + project_nonnegative(-logic_fragile_bridge),
                "window_axiom_feature": project_nonnegative((window_hidden + window_mlp) / 2.0),
                "control_axiom_feature": logic_control + project_nonnegative(syntax_control) - project_nonnegative(style_control),
            }
        )
    return out


def fit_axiom_constrained(rows: Sequence[Dict[str, object]], target_name: str) -> Dict[str, object]:
    fit = fit_linear_regression(rows, FEATURE_NAMES, target_name)
    weights = {key: safe_float(value) for key, value in dict(fit.get("weights", {})).items()}
    constrained = dict(weights)
    for feature_name in ("atlas_axiom_feature", "offset_axiom_feature", "frontier_axiom_feature", "subfield_axiom_feature", "window_axiom_feature"):
        if safe_float(constrained.get(feature_name)) < 0.0:
            constrained[feature_name] = 0.0
    return {
        "target_name": target_name,
        "base_weights": weights,
        "axiom_constrained_weights": constrained,
        "row_count": len(rows),
    }


def build_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    axiom_rows = build_axiom_rows(rows)
    fits = [
        fit_axiom_constrained(axiom_rows, "union_joint_adv"),
        fit_axiom_constrained(axiom_rows, "union_synergy_joint"),
        fit_axiom_constrained(axiom_rows, "strict_positive_synergy"),
    ]
    return {
        "record_type": "stage56_axiom_constrained_regression_summary",
        "row_count": len(axiom_rows),
        "feature_names": list(FEATURE_NAMES),
        "fits": fits,
        "main_judgment": (
            "公理约束已经从符号先验推进到特征重写和非负主项约束，"
            "主方程现在更接近按公理形状进入拟合，而不是拟合后再解释。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 公理约束回归摘要",
        "",
        f"- row_count: {summary.get('row_count', 0)}",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Fits",
    ]
    for fit in list(summary.get("fits", [])):
        fit = dict(fit)
        lines.append(f"- target: {fit.get('target_name', '')}")
        for key, value in dict(fit.get("axiom_constrained_weights", {})).items():
            lines.append(f"  {key}: {safe_float(value):+.6f}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run axiom-shaped constrained regression on sample design rows")
    ap.add_argument(
        "--design-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_fullsample_regression_runner_20260319" / "design_rows.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_axiom_constrained_regression_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(read_json(Path(args.design_rows_json)).get("rows", []))
    summary = build_summary(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "row_count": summary["row_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
