#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from cross_model_language_shared import PROJECT_ROOT, build_all_model_bundles, clamp01


OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage141_cross_model_layer_isomorphism_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE141_CROSS_MODEL_LAYER_ISOMORPHISM_REPORT.md"


def load_cached_summary(output_dir: Path) -> Dict[str, object] | None:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    return None


def normalize_layer(layer_index: float, layer_count: int) -> float:
    if layer_count <= 1:
        return 0.0
    return float(layer_index) / float(layer_count - 1)


def weighted_affine_fit(xs: Sequence[float], ys: Sequence[float], ws: Sequence[float]) -> Dict[str, float]:
    w_sum = sum(ws)
    if w_sum <= 1e-12:
        return {"slope": 1.0, "intercept": 0.0, "mae": 0.0}
    x_mean = sum(w * x for x, w in zip(xs, ws)) / w_sum
    y_mean = sum(w * y for y, w in zip(ys, ws)) / w_sum
    cov = sum(w * (x - x_mean) * (y - y_mean) for x, y, w in zip(xs, ys, ws))
    var = sum(w * (x - x_mean) ** 2 for x, w in zip(xs, ws))
    slope = cov / var if var > 1e-12 else 0.0
    intercept = y_mean - slope * x_mean
    mae = sum(w * abs((slope * x + intercept) - y) for x, y, w in zip(xs, ys, ws)) / w_sum
    return {"slope": slope, "intercept": intercept, "mae": mae}


def spacing_ratio(early: float, route: float, late: float) -> float:
    denom = late - early
    if abs(denom) <= 1e-12:
        return 0.0
    return (route - early) / denom


def extract_model_rows() -> List[Dict[str, object]]:
    bundles = build_all_model_bundles()
    gpt2 = bundles["gpt2"]
    gpt2_stage123 = gpt2["stage123"]
    gpt2_stage124 = gpt2["stage124"]
    gpt2_stage130 = gpt2["stage130"]

    rows = [
        {
            "model_key": "gpt2",
            "display_name": "GPT-2",
            "layer_count": int(gpt2_stage123["layer_count"]),
            "early_layer": int(gpt2_stage130["dominant_general_layer_index"]),
            "route_layer": int(gpt2_stage123["dominant_layer_index"]),
            "late_layer": int(gpt2_stage124["dominant_general_layer_index"]),
            "early_support": float(gpt2_stage130.get("syntax_stability_rate", 1.0)),
            "route_support": float(gpt2_stage123["route_shift_layer_localization_score"]),
            "late_support": float(gpt2_stage124["noun_neuron_basic_probe_score"]),
        }
    ]

    for model_key in ("qwen3", "deepseek7b"):
        bundle = bundles[model_key]
        mapped = bundle["transfer"]["mapped_layers"]
        route_summary = bundle["summary"]["route_summary"]
        noun_basic_summary = bundle["summary"]["noun_basic_summary"]
        rows.append(
            {
                "model_key": model_key,
                "display_name": bundle["display_name"],
                "layer_count": int(mapped["layer_count"]),
                "early_layer": int(mapped["early_layer"]),
                "route_layer": int(route_summary["dominant_layer_index"]),
                "late_layer": int(noun_basic_summary["dominant_general_layer_index"]),
                "early_support": float(bundle["transfer"]["qwen_core_metrics"]["syntax_stability_rate"]),
                "route_support": float(route_summary["route_shift_layer_localization_score"]),
                "late_support": float(noun_basic_summary["wordclass_neuron_basic_probe_score"]),
            }
        )

    for row in rows:
        row["early_norm"] = normalize_layer(float(row["early_layer"]), int(row["layer_count"]))
        row["route_norm"] = normalize_layer(float(row["route_layer"]), int(row["layer_count"]))
        row["late_norm"] = normalize_layer(float(row["late_layer"]), int(row["layer_count"]))
        row["spacing_ratio"] = spacing_ratio(float(row["early_norm"]), float(row["route_norm"]), float(row["late_norm"]))
        row["order_is_valid"] = bool(row["early_layer"] <= row["route_layer"] <= row["late_layer"])
    return rows


def compare_to_gpt2(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    gpt2_row = next(row for row in rows if row["model_key"] == "gpt2")
    source_xs = [float(gpt2_row["early_layer"]), float(gpt2_row["route_layer"]), float(gpt2_row["late_layer"])]
    out: List[Dict[str, object]] = []
    for row in rows:
        if row["model_key"] == "gpt2":
            continue
        target_ys = [float(row["early_layer"]), float(row["route_layer"]), float(row["late_layer"])]
        weights = [max(0.05, float(row["early_support"])), max(0.05, float(row["route_support"])), max(0.05, float(row["late_support"]))]
        affine = weighted_affine_fit(source_xs, target_ys, weights)
        norm_gap = (
            abs(float(row["early_norm"]) - float(gpt2_row["early_norm"]))
            + abs(float(row["route_norm"]) - float(gpt2_row["route_norm"]))
            + abs(float(row["late_norm"]) - float(gpt2_row["late_norm"]))
        ) / 3.0
        ratio_gap = abs(float(row["spacing_ratio"]) - float(gpt2_row["spacing_ratio"]))
        order_score = 1.0 if row["order_is_valid"] else 0.0
        isomorphism_score = (
            0.45 * clamp01(1.0 - norm_gap)
            + 0.25 * clamp01(1.0 - affine["mae"] / max(1.0, float(row["layer_count"] - 1)))
            + 0.15 * clamp01(1.0 - ratio_gap)
            + 0.15 * order_score
        )
        out.append(
            {
                "model_key": row["model_key"],
                "display_name": row["display_name"],
                "affine_slope": affine["slope"],
                "affine_intercept": affine["intercept"],
                "affine_mae": affine["mae"],
                "normalized_gap_mean": norm_gap,
                "spacing_ratio_gap": ratio_gap,
                "order_is_valid": order_score == 1.0,
                "layer_isomorphism_score": isomorphism_score,
            }
        )
    return out


def build_summary(rows: Sequence[Dict[str, object]], pair_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    mean_score = sum(float(row["layer_isomorphism_score"]) for row in pair_rows) / len(pair_rows)
    best_pair = max(pair_rows, key=lambda row: float(row["layer_isomorphism_score"]))
    weakest_pair = min(pair_rows, key=lambda row: float(row["layer_isomorphism_score"]))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage141_cross_model_layer_isomorphism",
        "title": "跨模型层同构块",
        "status_short": "cross_model_layer_isomorphism_ready",
        "model_count": len(rows),
        "pair_count": len(pair_rows),
        "mean_layer_isomorphism_score": mean_score,
        "best_aligned_model": best_pair["display_name"],
        "best_aligned_score": best_pair["layer_isomorphism_score"],
        "weakest_aligned_model": weakest_pair["display_name"],
        "weakest_aligned_score": weakest_pair["layer_isomorphism_score"],
        "model_rows": list(rows),
        "pair_rows": list(pair_rows),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage141: 跨模型层同构块",
        "",
        "## 核心结果",
        f"- 模型数量: {summary['model_count']}",
        f"- 配对数量: {summary['pair_count']}",
        f"- 平均层同构分数: {summary['mean_layer_isomorphism_score']:.4f}",
        f"- 最佳对齐模型: {summary['best_aligned_model']} ({summary['best_aligned_score']:.4f})",
        f"- 最弱对齐模型: {summary['weakest_aligned_model']} ({summary['weakest_aligned_score']:.4f})",
        "",
        "## 模型层位坐标",
    ]
    for row in summary["model_rows"]:
        lines.append(
            "- "
            f"{row['display_name']}: "
            f"layers={row['layer_count']}, "
            f"early={row['early_layer']} ({row['early_norm']:.4f}), "
            f"route={row['route_layer']} ({row['route_norm']:.4f}), "
            f"late={row['late_layer']} ({row['late_norm']:.4f}), "
            f"support=({row['early_support']:.4f}, {row['route_support']:.4f}, {row['late_support']:.4f})"
        )
    lines.extend(["", "## 相对 GPT-2 的同构结果"])
    for row in summary["pair_rows"]:
        lines.append(
            "- "
            f"{row['display_name']}: "
            f"score={row['layer_isomorphism_score']:.4f}, "
            f"slope={row['affine_slope']:.4f}, "
            f"intercept={row['affine_intercept']:.4f}, "
            f"mae={row['affine_mae']:.4f}, "
            f"norm_gap={row['normalized_gap_mean']:.4f}, "
            f"ratio_gap={row['spacing_ratio_gap']:.4f}, "
            f"order_valid={row['order_is_valid']}"
        )
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> Dict[str, object]:
    if not force:
        cached = load_cached_summary(output_dir)
        if cached is not None:
            return cached
    rows = extract_model_rows()
    pair_rows = compare_to_gpt2(rows)
    summary = build_summary(rows, pair_rows)
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="跨模型层同构块")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重算")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
