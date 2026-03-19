from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from stage56_fullsample_regression_runner import fit_linear_regression, read_json, safe_float

ROOT = Path(__file__).resolve().parents[2]


SIGN_PRIORS = {
    "logic_prototype_proxy": "positive",
    "logic_fragile_bridge_proxy": "negative",
    "syntax_constraint_conflict_proxy": "positive",
    "logic_control_proxy": "negative",
}


FEATURE_NAMES = [
    "atlas_static_proxy",
    "offset_static_proxy",
    "frontier_dynamic_proxy",
    "logic_prototype_proxy",
    "logic_fragile_bridge_proxy",
    "syntax_constraint_conflict_proxy",
    "window_hidden_proxy",
    "window_mlp_proxy",
    "style_control_proxy",
    "logic_control_proxy",
    "syntax_control_proxy",
]


def apply_sign_constraints(weights: Dict[str, float], priors: Dict[str, str]) -> Dict[str, float]:
    adjusted = dict(weights)
    for feature_name, sign in priors.items():
        value = safe_float(adjusted.get(feature_name))
        if sign == "positive" and value < 0.0:
            adjusted[feature_name] = 0.0
        if sign == "negative" and value > 0.0:
            adjusted[feature_name] = 0.0
    return adjusted


def fit_constrained(rows: Sequence[Dict[str, object]], target_name: str) -> Dict[str, object]:
    base_fit = fit_linear_regression(rows, FEATURE_NAMES, target_name)
    base_weights = {key: safe_float(value) for key, value in dict(base_fit.get("weights", {})).items()}
    constrained_weights = apply_sign_constraints(base_weights, SIGN_PRIORS)
    return {
        "target_name": target_name,
        "base_weights": base_weights,
        "constrained_weights": constrained_weights,
        "priors": dict(SIGN_PRIORS),
        "row_count": len(rows),
    }


def build_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    fits = [
        fit_constrained(rows, "union_joint_adv"),
        fit_constrained(rows, "union_synergy_joint"),
        fit_constrained(rows, "strict_positive_synergy"),
    ]
    return {
        "record_type": "stage56_constrained_sample_regression_summary",
        "row_count": len(rows),
        "sign_priors": dict(SIGN_PRIORS),
        "fits": fits,
        "main_judgment": (
            "当前第一版约束回归已经把稳定符号先验压进样本级回归，"
            "可以直接检查主方程在符号约束下是否更接近可解释形式。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 约束样本回归摘要",
        "",
        f"- row_count: {summary.get('row_count', 0)}",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Sign Priors",
    ]
    for key, value in dict(summary.get("sign_priors", {})).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Fits"])
    for fit in list(summary.get("fits", [])):
        fit = dict(fit)
        lines.append(f"- target: {fit.get('target_name', '')}")
        lines.append("  constrained_weights:")
        for key, value in dict(fit.get("constrained_weights", {})).items():
            lines.append(f"    {key}: {safe_float(value):+.6f}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Apply sign-constrained sample regression to unified design rows")
    ap.add_argument(
        "--design-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_fullsample_regression_runner_20260319" / "design_rows.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_constrained_sample_regression_20260319"),
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
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
