from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from stage56_fullsample_regression_runner import fit_linear_regression, read_json, safe_float

ROOT = Path(__file__).resolve().parents[2]


def pair_key(row: Dict[str, object]) -> Tuple[str, str, str, str]:
    return (
        str(row.get("model_id", "")),
        str(row.get("category", "")),
        str(row.get("prototype_term", "")),
        str(row.get("instance_term", "")),
    )


def join_rows(
    closed_rows: Sequence[Dict[str, object]],
    control_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    control_map = {pair_key(row): dict(row) for row in control_rows}
    out: List[Dict[str, object]] = []
    for row in closed_rows:
        key = pair_key(row)
        control_row = control_map.get(key)
        if control_row is None:
            continue
        out.append(
            {
                **dict(row),
                "logic_structure_gain_term": safe_float(control_row.get("logic_compaction_mid"))
                - safe_float(control_row.get("logic_delta_l2")),
                "syntax_structure_gain_term": safe_float(control_row.get("syntax_coverage_mid"))
                - safe_float(control_row.get("syntax_delta_l2")),
                "style_structure_gain_term": safe_float(control_row.get("style_delta_mean_abs"))
                - safe_float(control_row.get("style_coverage_mid")),
            }
        )
    return out


def feature_names() -> List[str]:
    return [
        "closed_form_balance_v2_term",
        "alignment_load_v2_term",
        "logic_structure_gain_term",
        "syntax_structure_gain_term",
        "style_structure_gain_term",
    ]


def sign_of(value: float) -> str:
    if value > 1e-12:
        return "positive"
    if value < -1e-12:
        return "negative"
    return "neutral"


def build_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    names = feature_names()
    fits = [
        fit_linear_regression(rows, names, "union_joint_adv"),
        fit_linear_regression(rows, names, "union_synergy_joint"),
        fit_linear_regression(rows, names, "strict_positive_synergy"),
    ]
    sign_matrix = {
        feature: {
            dict(fit).get("target_name", ""): sign_of(safe_float(dict(dict(fit).get("weights", {})).get(feature)))
            for fit in fits
        }
        for feature in names
    }
    stable_features: List[Dict[str, object]] = []
    for feature, targets in sign_matrix.items():
        signs = {sign for sign in targets.values() if sign != "neutral"}
        if len(signs) == 1 and signs:
            stable_features.append({"feature": feature, "sign": next(iter(signs))})
    return {
        "record_type": "stage56_control_axis_closed_form_integration_summary",
        "row_count": len(rows),
        "feature_names": names,
        "fits": fits,
        "sign_matrix": sign_matrix,
        "stable_features": stable_features,
        "equation_text": (
            "U_closed_ctrl(pair) = closed_form_balance_v2 + alignment_load_v2 + logic_structure_gain + syntax_structure_gain + style_structure_gain"
        ),
        "main_judgment": (
            "闭式核已开始吸收逻辑、句法、风格的稳定控制子通道，"
            "可以直接判断控制轴是否更适合作为闭式核的微修正，而不是独立粗总项。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 控制轴并入闭式核摘要",
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
    ap = argparse.ArgumentParser(description="Integrate stable control subchannels into the closed-form kernel")
    ap.add_argument(
        "--closed-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_closed_form_kernel_refit_20260319" / "rows.json"),
    )
    ap.add_argument(
        "--control-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_control_axis_decomposition_20260319" / "rows.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_control_axis_closed_form_integration_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    closed_rows = list(read_json(Path(args.closed_rows_json)).get("rows", []))
    control_rows = list(read_json(Path(args.control_rows_json)).get("rows", []))
    rows = join_rows(closed_rows, control_rows)
    summary = build_summary(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rows.json").write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
