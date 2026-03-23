#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage123: route shift（选路偏移）layer localization（层定位）分析。

目标：
1. 基于 Stage122 的逐层轨迹，定位副词触发的动词位选路偏移主要集中在哪些层。
2. 区分早层 / 中层 / 后层的贡献，避免只看全层平均值。
3. 为后续从词类层桥接到隐状态层的数学分析提供层级锚点。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from stage122_adverb_context_route_shift_probe import OUTPUT_DIR as STAGE122_OUTPUT_DIR
from stage122_adverb_context_route_shift_probe import run_analysis as run_stage122_analysis


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage123_route_shift_layer_localization_20260323"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def ensure_stage122_outputs() -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    summary_path = STAGE122_OUTPUT_DIR / "summary.json"
    trace_path = STAGE122_OUTPUT_DIR / "layer_trace_rows.json"
    if not summary_path.exists() or not trace_path.exists():
        run_stage122_analysis(output_dir=STAGE122_OUTPUT_DIR)
    return load_json(summary_path), load_json(trace_path)


def layer_band_name(layer_idx: int, layer_count: int) -> str:
    band_size = max(1, layer_count // 3)
    if layer_idx < band_size:
        return "early"
    if layer_idx < min(layer_count, band_size * 2):
        return "middle"
    return "late"


def summarize_layers(trace_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    layer_count = len(trace_rows[0]["verb_route_advantage_by_layer"])
    layer_rows = []
    peak_hist = [0 for _ in range(layer_count)]

    for row in trace_rows:
        peak_hist[int(row["peak_layer_index"])] += 1

    for layer_idx in range(layer_count):
        route_values = [float(row["verb_route_advantage_by_layer"][layer_idx]) for row in trace_rows]
        last_values = [float(row["last_route_advantage_by_layer"][layer_idx]) for row in trace_rows]
        attention_values = [float(row["modifier_attention_advantage_by_layer"][layer_idx]) for row in trace_rows]
        route_mean = sum(route_values) / len(route_values)
        last_mean = sum(last_values) / len(last_values)
        attention_mean = sum(attention_values) / len(attention_values)
        positive_rate = sum(1 for value in route_values if value > 0.0) / len(route_values)
        peak_hit_rate = peak_hist[layer_idx] / len(trace_rows)
        layer_rows.append(
            {
                "layer_index": layer_idx,
                "layer_band": layer_band_name(layer_idx, layer_count),
                "verb_route_advantage_mean": float(route_mean),
                "last_route_advantage_mean": float(last_mean),
                "modifier_attention_advantage_mean": float(attention_mean),
                "positive_route_rate": float(positive_rate),
                "peak_hit_rate": float(peak_hit_rate),
            }
        )

    band_rows = []
    for band_name in ["early", "middle", "late"]:
        band_layers = [row for row in layer_rows if row["layer_band"] == band_name]
        band_rows.append(
            {
                "band_name": band_name,
                "layer_count": len(band_layers),
                "verb_route_advantage_mean": float(
                    sum(row["verb_route_advantage_mean"] for row in band_layers) / max(1, len(band_layers))
                ),
                "modifier_attention_advantage_mean": float(
                    sum(row["modifier_attention_advantage_mean"] for row in band_layers) / max(1, len(band_layers))
                ),
                "peak_hit_rate_mean": float(
                    sum(row["peak_hit_rate"] for row in band_layers) / max(1, len(band_layers))
                ),
            }
        )

    dominant_layer = max(layer_rows, key=lambda row: row["verb_route_advantage_mean"])
    dominant_peak_layer = max(layer_rows, key=lambda row: row["peak_hit_rate"])
    onset_candidates = [
        row
        for row in layer_rows
        if row["verb_route_advantage_mean"] > 0.002 and row["positive_route_rate"] >= 0.58
    ]
    earliest_stable_layer = onset_candidates[0]["layer_index"] if onset_candidates else dominant_layer["layer_index"]

    sorted_bands = sorted(band_rows, key=lambda row: row["verb_route_advantage_mean"], reverse=True)
    best_band = sorted_bands[0]
    second_band = sorted_bands[1]
    band_separation = best_band["verb_route_advantage_mean"] - second_band["verb_route_advantage_mean"]

    localization_score = (
        0.30 * clamp01(dominant_layer["verb_route_advantage_mean"] / 0.01)
        + 0.25 * clamp01(dominant_peak_layer["peak_hit_rate"] / 0.35)
        + 0.25 * clamp01(best_band["verb_route_advantage_mean"] / 0.006)
        + 0.20 * clamp01(band_separation / 0.0015)
    )

    top_layers = sorted(layer_rows, key=lambda row: row["verb_route_advantage_mean"], reverse=True)[:5]

    return {
        "layer_count": layer_count,
        "layer_rows": layer_rows,
        "band_rows": band_rows,
        "dominant_layer_index": int(dominant_layer["layer_index"]),
        "dominant_layer_band": dominant_layer["layer_band"],
        "dominant_layer_route_advantage_mean": float(dominant_layer["verb_route_advantage_mean"]),
        "dominant_layer_peak_hit_rate": float(dominant_peak_layer["peak_hit_rate"]),
        "dominant_peak_layer_index": int(dominant_peak_layer["layer_index"]),
        "earliest_stable_layer_index": int(earliest_stable_layer),
        "best_band_name": best_band["band_name"],
        "best_band_route_advantage_mean": float(best_band["verb_route_advantage_mean"]),
        "band_separation_margin": float(band_separation),
        "route_shift_layer_localization_score": float(localization_score),
        "top_layers": top_layers,
    }


def build_summary(
    stage122_summary: Dict[str, object],
    layer_summary: Dict[str, object],
) -> Dict[str, object]:
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage123_route_shift_layer_localization",
        "title": "Route Shift 层定位分析",
        "status_short": "gpt2_route_shift_layer_localized",
        "model_name": "gpt2",
        "source_stage": "stage122_adverb_context_route_shift_probe",
        "source_case_count": stage122_summary["case_count"],
        "source_dynamic_score": stage122_summary["adverb_context_route_shift_score"],
        **layer_summary,
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage123: Route Shift 层定位分析",
        "",
        "## 核心结果",
        f"- 来源样本对数量: {summary['source_case_count']}",
        f"- 来源动态分数: {summary['source_dynamic_score']:.4f}",
        f"- 主导层: L{summary['dominant_layer_index']}",
        f"- 主导层带: {summary['dominant_layer_band']}",
        f"- 主导峰值层: L{summary['dominant_peak_layer_index']}",
        f"- 最早稳定层: L{summary['earliest_stable_layer_index']}",
        f"- 最强层带: {summary['best_band_name']}",
        f"- 层定位分数: {summary['route_shift_layer_localization_score']:.4f}",
        "",
        "## 逐层前五名",
    ]

    for row in summary["top_layers"]:
        lines.append(
            "- "
            f"L{row['layer_index']} ({row['layer_band']}): "
            f"route_adv={row['verb_route_advantage_mean']:.6f}, "
            f"peak_hit={row['peak_hit_rate']:.4f}, "
            f"attn_adv={row['modifier_attention_advantage_mean']:.6f}"
        )

    lines.extend(["", "## 层带汇总"])
    for row in summary["band_rows"]:
        lines.append(
            "- "
            f"{row['band_name']}: "
            f"route_adv={row['verb_route_advantage_mean']:.6f}, "
            f"attn_adv={row['modifier_attention_advantage_mean']:.6f}, "
            f"peak_hit_mean={row['peak_hit_rate_mean']:.4f}"
        )

    lines.extend(
        [
            "",
            "## 理论提示",
            "- 如果偏移集中在中后层，说明副词更像在上下文整合后改变选路，而不是只在最早词法层起作用。",
            "- 如果最早稳定层明显早于主导峰值层，说明副词效应可能先萌发，再在更深层完成放大。",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "STAGE123_ROUTE_SHIFT_LAYER_LOCALIZATION_REPORT.md"
    layers_path = output_dir / "layer_rows.json"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    report_path.write_text(build_report(summary), encoding="utf-8-sig")
    layers_path.write_text(json.dumps(summary["layer_rows"], ensure_ascii=False, indent=2), encoding="utf-8-sig")
    return {"summary": summary_path, "report": report_path, "layer_rows": layers_path}


def run_analysis(*, output_dir: Path = OUTPUT_DIR) -> Dict[str, object]:
    stage122_summary, trace_rows = ensure_stage122_outputs()
    layer_summary = summarize_layers(trace_rows)
    summary = build_summary(stage122_summary, layer_summary)
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Route Shift 层定位分析")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Stage123 输出目录")
    args = parser.parse_args()

    summary = run_analysis(output_dir=Path(args.output_dir))
    print(
        json.dumps(
            {
                "status_short": summary["status_short"],
                "output_dir": str(Path(args.output_dir)),
                "dominant_layer_index": summary["dominant_layer_index"],
                "route_shift_layer_localization_score": summary["route_shift_layer_localization_score"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
