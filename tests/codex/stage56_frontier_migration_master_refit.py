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
    refit_rows: Sequence[Dict[str, object]],
    frontier_rows: Sequence[Dict[str, object]],
    window_gate_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    frontier_map = {pair_key(row): dict(row) for row in frontier_rows}
    gate_map = {pair_key(row): dict(row) for row in window_gate_rows}
    out: List[Dict[str, object]] = []
    for row in refit_rows:
        key = pair_key(row)
        frontier_row = frontier_map.get(key)
        gate_row = gate_map.get(key)
        if frontier_row is None or gate_row is None:
            continue
        out.append(
            {
                "model_id": key[0],
                "category": key[1],
                "prototype_term": key[2],
                "instance_term": key[3],
                "identity_margin_term": safe_float(row.get("identity_margin_term")),
                "syntax_constraint_conflict_term": safe_float(row.get("syntax_constraint_conflict_term")),
                "logic_fragile_bridge_term": safe_float(row.get("logic_fragile_bridge_term")),
                "style_alignment_term": safe_float(row.get("style_alignment_term")),
                "frontier_positive_migration_term": safe_float(frontier_row.get("frontier_compaction_late_shift"))
                + safe_float(frontier_row.get("frontier_balance_term")),
                "frontier_negative_base_term": safe_float(frontier_row.get("frontier_compaction_term"))
                + safe_float(frontier_row.get("frontier_coverage_term"))
                + safe_float(frontier_row.get("frontier_separation_term")),
                "window_gate_positive_term": safe_float(gate_row.get("window_positive_core_term"))
                + safe_float(gate_row.get("window_syntax_term")),
                "window_gate_negative_term": safe_float(gate_row.get("window_negative_core_term"))
                + safe_float(gate_row.get("window_fragile_term")),
                "union_joint_adv": safe_float(row.get("union_joint_adv")),
                "union_synergy_joint": safe_float(row.get("union_synergy_joint")),
                "strict_positive_synergy": safe_float(row.get("strict_positive_synergy")),
            }
        )
    return out


def feature_names() -> List[str]:
    return [
        "identity_margin_term",
        "syntax_constraint_conflict_term",
        "logic_fragile_bridge_term",
        "style_alignment_term",
        "frontier_positive_migration_term",
        "frontier_negative_base_term",
        "window_gate_positive_term",
        "window_gate_negative_term",
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
        "record_type": "stage56_frontier_migration_master_refit_summary",
        "row_count": len(rows),
        "feature_names": names,
        "fits": fits,
        "sign_matrix": sign_matrix,
        "stable_features": stable_features,
        "equation_text": (
            "U_gate_refit(pair) = b1 * identity_margin + b2 * syntax_constraint_conflict "
            "+ b3 * logic_fragile_bridge + b4 * style_alignment "
            "+ b5 * frontier_positive_migration + b6 * frontier_negative_base "
            "+ b7 * window_gate_positive + b8 * window_gate_negative"
        ),
        "main_judgment": (
            "主方程已把粗前沿项和粗窗口项替换成迁移型前沿与条件门型窗口子项，"
            "可以直接判断主方程是否进一步变短、变稳。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 前沿迁移并场重拟合摘要",
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
    ap = argparse.ArgumentParser(description="Refit master equation with frontier migration and window gate terms")
    ap.add_argument(
        "--refit-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_master_equation_refit_20260319" / "rows.json"),
    )
    ap.add_argument(
        "--frontier-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_frontier_heterogeneity_split_20260319" / "rows.json"),
    )
    ap.add_argument(
        "--window-gate-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_window_condition_gate_closure_20260319" / "rows.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_frontier_migration_master_refit_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    refit_rows = list(read_json(Path(args.refit_rows_json)).get("rows", []))
    frontier_rows = list(read_json(Path(args.frontier_rows_json)).get("rows", []))
    window_gate_rows = list(read_json(Path(args.window_gate_rows_json)).get("rows", []))
    rows = join_rows(refit_rows, frontier_rows, window_gate_rows)
    summary = build_summary(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rows.json").write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
