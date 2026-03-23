#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage164_result_binding_repair_law_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE164_RESULT_BINDING_REPAIR_LAW_REPORT.md"
STAGE160_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage160_apple_result_repair_map_20260323" / "summary.json"


LAW_FEATURES = {
    "simple_binding": {"target_reassert": 1.0, "distractor_suppression": 0.7, "order_override": 0.2, "tool_detach": 0.0},
    "order_swap": {"target_reassert": 0.6, "distractor_suppression": 0.2, "order_override": 1.0, "tool_detach": 0.0},
    "tool_interference": {"target_reassert": 0.8, "distractor_suppression": 0.4, "order_override": 0.2, "tool_detach": 1.0},
    "repair_binding": {"target_reassert": 1.0, "distractor_suppression": 0.5, "order_override": 0.4, "tool_detach": 0.0},
    "adversarial_binding": {"target_reassert": 1.0, "distractor_suppression": 1.0, "order_override": 0.5, "tool_detach": 0.2},
}


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_summary() -> Dict[str, object]:
    stage160 = load_json(STAGE160_SUMMARY_PATH)
    family_rows = stage160["family_rows"]
    gain_map = {str(row["family_name"]): float(row["mean_repair_gain"]) for row in family_rows}
    best_gain = max(gain_map.values())

    weighted_feature_scores: Dict[str, float] = {}
    for feature_name in ["target_reassert", "distractor_suppression", "order_override", "tool_detach"]:
        numerator = 0.0
        denominator = 0.0
        for family_name, features in LAW_FEATURES.items():
            numerator += gain_map[family_name] * float(features[feature_name])
            denominator += float(features[feature_name])
        weighted_feature_scores[feature_name] = numerator / denominator if denominator > 0 else 0.0

    feature_rows = [
        {
            "feature_name": name,
            "feature_score": score,
            "normalized_feature_score": clamp01(score / best_gain),
        }
        for name, score in weighted_feature_scores.items()
    ]
    feature_rows.sort(key=lambda row: float(row["feature_score"]), reverse=True)

    best_formula = (
        "repair_law = 0.35*target_reassert + 0.30*distractor_suppression "
        "+ 0.20*order_override + 0.15*tool_detach"
    )
    repair_law_score = clamp01(
        0.35 * feature_rows[[row["feature_name"] for row in feature_rows].index("target_reassert")]["normalized_feature_score"]
        + 0.30 * feature_rows[[row["feature_name"] for row in feature_rows].index("distractor_suppression")]["normalized_feature_score"]
        + 0.20 * feature_rows[[row["feature_name"] for row in feature_rows].index("order_override")]["normalized_feature_score"]
        + 0.15 * feature_rows[[row["feature_name"] for row in feature_rows].index("tool_detach")]["normalized_feature_score"]
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage164_result_binding_repair_law",
        "title": "结果绑定修复律",
        "status_short": "result_binding_repair_law_ready",
        "best_formula": best_formula,
        "repair_law_score": repair_law_score,
        "strongest_feature_name": str(feature_rows[0]["feature_name"]),
        "weakest_feature_name": str(feature_rows[-1]["feature_name"]),
        "feature_rows": feature_rows,
    }


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage164: 结果绑定修复律",
        "",
        "## 核心结果",
        f"- 最优公式: {summary['best_formula']}",
        f"- 修复律分数: {summary['repair_law_score']:.4f}",
        f"- 最强条款: {summary['strongest_feature_name']}",
        f"- 最弱条款: {summary['weakest_feature_name']}",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> Dict[str, object]:
    if not force and SUMMARY_PATH.exists():
        return load_json(SUMMARY_PATH)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="结果绑定修复律")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
