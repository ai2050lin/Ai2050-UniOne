from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from stage56_fullsample_regression_runner import fit_linear_regression, read_json, safe_float

ROOT = Path(__file__).resolve().parents[2]


KEY_FEATURES = [
    "atlas_static_proxy",
    "offset_static_proxy",
    "frontier_dynamic_proxy",
    "logic_prototype_proxy",
    "logic_fragile_bridge_proxy",
    "syntax_constraint_conflict_proxy",
    "style_control_proxy",
    "logic_control_proxy",
    "syntax_control_proxy",
]


def sign_of(value: float) -> str:
    if value > 1e-12:
        return "positive"
    if value < -1e-12:
        return "negative"
    return "neutral"


def fit_rows(rows: Sequence[Dict[str, object]], target_name: str) -> Dict[str, float]:
    fit = fit_linear_regression(rows, KEY_FEATURES, target_name)
    return {feature: safe_float(dict(fit.get("weights", {})).get(feature)) for feature in KEY_FEATURES}


def build_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    target_names = ["union_joint_adv", "union_synergy_joint", "strict_positive_synergy"]
    fits = {target_name: fit_rows(rows, target_name) for target_name in target_names}
    sign_matrix = {
        feature: {target_name: sign_of(safe_float(fits[target_name].get(feature))) for target_name in target_names}
        for feature in KEY_FEATURES
    }
    stable_features = []
    for feature, targets in sign_matrix.items():
        signs = {sign for sign in targets.values() if sign != "neutral"}
        if len(signs) == 1 and signs:
            stable_features.append({"feature": feature, "sign": next(iter(signs))})
    return {
        "record_type": "stage56_sign_stability_runner_summary",
        "row_count": len(rows),
        "sign_matrix": sign_matrix,
        "stable_features": stable_features,
        "main_judgment": (
            "当前样本级符号稳定分析已经能直接检查控制轴和子场项在三个目标上的符号是否一致。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 样本级符号稳定摘要",
        "",
        f"- row_count: {summary.get('row_count', 0)}",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Stable Features",
    ]
    for row in list(summary.get("stable_features", [])):
        row = dict(row)
        lines.append(f"- {row.get('feature', '')}: {row.get('sign', '')}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Check sign stability of control-axis and subfield terms on sample-level regression")
    ap.add_argument(
        "--design-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_fullsample_regression_runner_20260319" / "design_rows.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_sign_stability_runner_20260319"),
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
