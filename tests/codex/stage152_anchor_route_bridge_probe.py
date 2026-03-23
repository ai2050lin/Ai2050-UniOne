#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage152_anchor_route_bridge_probe_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE152_ANCHOR_ROUTE_BRIDGE_PROBE_REPORT.md"

STAGE123_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage123_route_shift_layer_localization_20260323" / "summary.json"
STAGE130_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage130_multisyntax_noun_context_probe_20260323" / "summary.json"
STAGE139_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage139_qwen3_language_validation_suite_20260323" / "summary.json"
STAGE140_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage140_deepseek_language_validation_suite_20260323" / "summary.json"


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def bridge_formula(anchor_score: float, route_score: float, localization_score: float) -> float:
    return clamp01(math.sqrt(max(anchor_score, 0.0) * max(route_score, 0.0)) * (0.5 + 0.5 * localization_score))


def build_model_rows() -> List[Dict[str, object]]:
    stage123 = load_json(STAGE123_SUMMARY_PATH)
    stage130 = load_json(STAGE130_SUMMARY_PATH)
    stage139 = load_json(STAGE139_SUMMARY_PATH)
    stage140 = load_json(STAGE140_SUMMARY_PATH)

    rows = [
        {
            "model_name": "GPT-2",
            "anchor_score": float(stage130["syntax_stability_rate"]),
            "route_score": float(stage123["source_dynamic_score"]),
            "route_localization_score": float(stage123["route_shift_layer_localization_score"]),
        },
        {
            "model_name": "Qwen3-4B",
            "anchor_score": float(stage139["transfer_summary"]["qwen_core_metrics"]["syntax_stability_rate"]),
            "route_score": float(stage139["transfer_summary"]["qwen_core_metrics"]["adverb_context_route_shift_score"]),
            "route_localization_score": float(stage139["transfer_summary"]["qwen_core_metrics"]["route_shift_layer_localization_score"]),
        },
        {
            "model_name": "DeepSeek-R1-Distill-Qwen-7B",
            "anchor_score": float(stage140["transfer_summary"]["qwen_core_metrics"]["syntax_stability_rate"]),
            "route_score": float(stage140["transfer_summary"]["qwen_core_metrics"]["adverb_context_route_shift_score"]),
            "route_localization_score": float(stage140["transfer_summary"]["qwen_core_metrics"]["route_shift_layer_localization_score"]),
        },
    ]
    for row in rows:
        row["anchor_route_bridge_score"] = bridge_formula(
            float(row["anchor_score"]),
            float(row["route_score"]),
            float(row["route_localization_score"]),
        )
    return rows


def build_summary(model_rows: List[Dict[str, object]]) -> Dict[str, object]:
    strongest = max(model_rows, key=lambda row: float(row["anchor_route_bridge_score"]))
    weakest = min(model_rows, key=lambda row: float(row["anchor_route_bridge_score"]))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage152_anchor_route_bridge_probe",
        "title": "早层定锚到门控路由桥接探针",
        "status_short": "anchor_route_bridge_ready",
        "model_count": len(model_rows),
        "mean_anchor_route_bridge_score": mean(float(row["anchor_route_bridge_score"]) for row in model_rows),
        "strongest_model_name": str(strongest["model_name"]),
        "weakest_model_name": str(weakest["model_name"]),
        "model_rows": model_rows,
    }


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage152: 早层定锚到门控路由桥接探针",
        "",
        "## 核心结果",
        f"- 模型数: {summary['model_count']}",
        f"- 平均桥接分数: {summary['mean_anchor_route_bridge_score']:.4f}",
        f"- 最强模型: {summary['strongest_model_name']}",
        f"- 最弱模型: {summary['weakest_model_name']}",
        "",
        "## 模型明细",
    ]
    for row in summary["model_rows"]:
        lines.append(
            f"- {row['model_name']}: anchor={row['anchor_score']:.4f}; "
            f"route={row['route_score']:.4f}; "
            f"localization={row['route_localization_score']:.4f}; "
            f"bridge={row['anchor_route_bridge_score']:.4f}"
        )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> Dict[str, object]:
    if not force and SUMMARY_PATH.exists():
        return load_json(SUMMARY_PATH)
    model_rows = build_model_rows()
    summary = build_summary(model_rows)
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="早层定锚到门控路由桥接探针")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
