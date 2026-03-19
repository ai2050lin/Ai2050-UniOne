from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from stage56_fullsample_regression_runner import fit_linear_regression, read_json, safe_float

ROOT = Path(__file__).resolve().parents[2]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def pair_key(row: Dict[str, object]) -> Tuple[str, str, str, str]:
    return (
        str(row.get("model_id", "")),
        str(row.get("category", "")),
        str(row.get("prototype_term", "")),
        str(row.get("instance_term", "")),
    )


def join_rows(
    deep_rows: Sequence[Dict[str, object]],
    style_rows: Sequence[Dict[str, object]],
    frontier_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    style_map = {pair_key(row): dict(row) for row in style_rows}
    frontier_map = {pair_key(row): dict(row) for row in frontier_rows}

    out: List[Dict[str, object]] = []
    for row in deep_rows:
        key = pair_key(row)
        style_row = style_map.get(key)
        frontier_row = frontier_map.get(key)
        if style_row is None or frontier_row is None:
            continue

        window = safe_float(row.get("window_dominance_term"))
        style_positive_core = mean(
            [
                safe_float(style_row.get("style_delta_mean_abs")),
                safe_float(style_row.get("style_role_align_compaction")),
                safe_float(style_row.get("style_alignment")),
                safe_float(style_row.get("style_reorder_pressure")),
                safe_float(style_row.get("style_gap")),
            ]
        )
        style_negative_core = mean(
            [
                safe_float(style_row.get("style_compaction_mid")),
                safe_float(style_row.get("style_coverage_mid")),
                safe_float(style_row.get("style_delta_l2")),
                safe_float(style_row.get("style_role_align_coverage")),
                safe_float(style_row.get("style_midfield")),
            ]
        )
        frontier_positive_migration = mean(
            [
                safe_float(frontier_row.get("frontier_compaction_late_shift")),
                safe_float(frontier_row.get("frontier_balance_term")),
            ]
        )
        frontier_negative_base = mean(
            [
                safe_float(frontier_row.get("frontier_compaction_term")),
                safe_float(frontier_row.get("frontier_coverage_term")),
                safe_float(frontier_row.get("frontier_separation_term")),
            ]
        )

        out.append(
            {
                **dict(row),
                "window_style_positive_term": window * style_positive_core,
                "window_style_negative_term": window * style_negative_core,
                "window_frontier_positive_term": window * frontier_positive_migration,
                "window_frontier_negative_term": window * frontier_negative_base,
            }
        )
    return out


def feature_names() -> List[str]:
    return [
        "window_style_positive_term",
        "window_style_negative_term",
        "window_frontier_positive_term",
        "window_frontier_negative_term",
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
        "record_type": "stage56_window_condition_gate_closure_summary",
        "row_count": len(rows),
        "feature_names": names,
        "fits": fits,
        "sign_matrix": sign_matrix,
        "stable_features": stable_features,
        "main_judgment": (
            "窗口条件门已被继续拆成风格正核/负核和前沿迁移/基础两类子门，"
            "可以直接判断窗口层到底在放大正迁移，还是在放大旧的负核基础。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 窗口条件门收口摘要",
        "",
        f"- row_count: {summary.get('row_count', 0)}",
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
    ap = argparse.ArgumentParser(description="Close unresolved window terms by splitting style and frontier gates")
    ap.add_argument(
        "--deep-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_window_dominance_deep_split_20260319" / "rows.json"),
    )
    ap.add_argument(
        "--style-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_style_axis_refinement_20260319" / "rows.json"),
    )
    ap.add_argument(
        "--frontier-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_frontier_heterogeneity_split_20260319" / "rows.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_window_condition_gate_closure_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    deep_rows = list(read_json(Path(args.deep_rows_json)).get("rows", []))
    style_rows = list(read_json(Path(args.style_rows_json)).get("rows", []))
    frontier_rows = list(read_json(Path(args.frontier_rows_json)).get("rows", []))
    rows = join_rows(deep_rows, style_rows, frontier_rows)
    summary = build_summary(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rows.json").write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
