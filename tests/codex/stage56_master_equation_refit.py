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
    design_rows: Sequence[Dict[str, object]],
    static_rows: Sequence[Dict[str, object]],
    window_rows: Sequence[Dict[str, object]],
    style_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    static_map = {pair_key(row): dict(row) for row in static_rows}
    window_map = {pair_key(row): dict(row) for row in window_rows}
    style_map = {pair_key(row): dict(row) for row in style_rows}

    out: List[Dict[str, object]] = []
    for row in design_rows:
        key = pair_key(row)
        static_row = static_map.get(key)
        window_row = window_map.get(key)
        style_row = style_map.get(key)
        if static_row is None or window_row is None or style_row is None:
            continue
        out.append(
            {
                "model_id": key[0],
                "category": key[1],
                "prototype_term": key[2],
                "instance_term": key[3],
                "identity_margin_term": safe_float(static_row.get("identity_margin_direct")),
                "frontier_term": safe_float(row.get("frontier_dynamic_proxy")),
                "logic_prototype_term": safe_float(row.get("logic_prototype_proxy")),
                "logic_fragile_bridge_term": safe_float(row.get("logic_fragile_bridge_proxy")),
                "syntax_constraint_conflict_term": safe_float(row.get("syntax_constraint_conflict_proxy")),
                "window_dominance_term": safe_float(window_row.get("generated_dominance_mean")),
                "style_alignment_term": safe_float(style_row.get("style_alignment")),
                "style_midfield_term": safe_float(style_row.get("style_midfield")),
                "logic_control_term": safe_float(row.get("logic_control_proxy")),
                "union_joint_adv": safe_float(row.get("union_joint_adv")),
                "union_synergy_joint": safe_float(row.get("union_synergy_joint")),
                "strict_positive_synergy": safe_float(row.get("strict_positive_synergy")),
            }
        )
    return out


def feature_names() -> List[str]:
    return [
        "identity_margin_term",
        "frontier_term",
        "logic_prototype_term",
        "logic_fragile_bridge_term",
        "syntax_constraint_conflict_term",
        "window_dominance_term",
        "style_alignment_term",
        "style_midfield_term",
        "logic_control_term",
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
        "record_type": "stage56_master_equation_refit_summary",
        "row_count": len(rows),
        "feature_names": names,
        "fits": fits,
        "sign_matrix": sign_matrix,
        "stable_features": stable_features,
        "equation_text": (
            "U_refit(pair) = a1 * identity_margin + a2 * frontier + a3 * logic_prototype "
            "+ a4 * logic_fragile_bridge + a5 * syntax_constraint_conflict "
            "+ a6 * window_dominance + a7 * style_alignment + a8 * style_midfield + a9 * logic_control"
        ),
        "main_judgment": (
            "主方程重拟合已经把静态边距、窗口主导性、稳定子场和细化后的 style 子通道并到同一条样本级方程里，"
            "可以直接检查新变量是否比旧粗代理更稳定。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 主方程重拟合摘要",
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
    ap = argparse.ArgumentParser(description="Refit the master equation with identity margin and window dominance")
    ap.add_argument(
        "--design-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_fullsample_regression_runner_20260319" / "design_rows.json"),
    )
    ap.add_argument(
        "--static-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_static_direct_measure_20260319" / "rows.json"),
    )
    ap.add_argument(
        "--window-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_window_term_strengthening_20260319" / "rows.json"),
    )
    ap.add_argument(
        "--style-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_style_axis_refinement_20260319" / "rows.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_master_equation_refit_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    design_rows = list(read_json(Path(args.design_rows_json)).get("rows", []))
    static_rows = list(read_json(Path(args.static_rows_json)).get("rows", []))
    window_rows = list(read_json(Path(args.window_rows_json)).get("rows", []))
    style_rows = list(read_json(Path(args.style_rows_json)).get("rows", []))
    rows = join_rows(design_rows, static_rows, window_rows, style_rows)
    summary = build_summary(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rows.json").write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
