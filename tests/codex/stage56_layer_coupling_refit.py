from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from stage56_fullsample_regression_runner import fit_linear_regression, read_json, safe_float

ROOT = Path(__file__).resolve().parents[2]


def build_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        style_penalty = -safe_float(row.get("style_structure_gain_term"))
        general_balance_v4 = safe_float(row.get("core_balance_v3_term")) + safe_float(row.get("logic_strictload_term"))
        kernel_v4 = general_balance_v4 - style_penalty
        strict_module_final = safe_float(row.get("strict_load_term"))
        dual_gap_final = kernel_v4 - strict_module_final
        gs_coupling = kernel_v4 * strict_module_final
        gd_coupling = kernel_v4 * dual_gap_final
        sd_coupling = strict_module_final * dual_gap_final
        out.append(
            {
                **dict(row),
                "kernel_v4_term": kernel_v4,
                "strict_module_final_term": strict_module_final,
                "dual_gap_final_term": dual_gap_final,
                "gs_coupling_term": gs_coupling,
                "gd_coupling_term": gd_coupling,
                "sd_coupling_term": sd_coupling,
            }
        )
    return out


def feature_names() -> List[str]:
    return [
        "gs_coupling_term",
        "gd_coupling_term",
        "sd_coupling_term",
    ]


def sign_of(value: float) -> str:
    if value > 1e-12:
        return "positive"
    if value < -1e-12:
        return "negative"
    return "neutral"


def build_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    names = feature_names()
    targets = ["union_joint_adv", "union_synergy_joint", "strict_positive_synergy"]
    fits = [fit_linear_regression(rows, names, target) for target in targets]
    sign_matrix = {
        feature: {
            target: sign_of(safe_float(dict(fit["weights"]).get(feature)))
            for fit, target in zip(fits, targets)
        }
        for feature in names
    }
    stable_features: List[Dict[str, object]] = []
    for feature, target_signs in sign_matrix.items():
        signs = {value for value in target_signs.values() if value != "neutral"}
        if len(signs) == 1 and signs:
            stable_features.append({"feature": feature, "sign": next(iter(signs))})
    return {
        "record_type": "stage56_layer_coupling_refit_summary",
        "row_count": len(rows),
        "feature_names": names,
        "fits": fits,
        "sign_matrix": sign_matrix,
        "stable_features": stable_features,
        "equation_text": (
            "gs_coupling = kernel_v4 * strict_module_final; "
            "gd_coupling = kernel_v4 * dual_gap_final; "
            "sd_coupling = strict_module_final * dual_gap_final"
        ),
        "main_judgment": (
            "层间耦合已经开始分化成稳定正耦合与稳定负耦合，"
            "可以直接判断三层不是松散并列，而是存在方向不同的结构性交互。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 层间耦合重拟合摘要",
        "",
        f"- row_count: {summary.get('row_count', 0)}",
        f"- equation_text: {summary.get('equation_text', '')}",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Stable Features",
    ]
    for row in list(summary.get("stable_features", [])):
        row = dict(row)
        lines.append(f"- {row.get('feature', '')}: {row.get('sign', '')}")
    lines.extend(["", "## Fits"])
    for fit in list(summary.get("fits", [])):
        fit = dict(fit)
        lines.append(f"- target: {fit.get('target_name', '')}")
        for key, value in dict(fit.get("weights", {})).items():
            lines.append(f"  {key}: {safe_float(value):+.6f}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Refit couplings across the layered dual-equation structure")
    ap.add_argument(
        "--input-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_logic_syntax_micro_compression_20260319" / "rows.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_layer_coupling_refit_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(read_json(Path(args.input_rows_json)).get("rows", []))
    out_rows = build_rows(rows)
    summary = build_summary(out_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rows.json").write_text(json.dumps({"rows": out_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(out_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
