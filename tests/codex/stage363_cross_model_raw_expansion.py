#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage263_qwen_deepseek_complete_behavior_suite import run_analysis as run_stage263
from stage264_qwen_deepseek_complete_structural_aggregate import run_analysis as run_stage264
from stage265_qwen_deepseek_complete_final_review import run_analysis as run_stage265
from stage266_qwen_deepseek_parameter_hook_compare import run_analysis as run_stage266
from stage267_qwen_deepseek_same_class_competition_compare import run_analysis as run_stage267
from stage271_cross_model_natural_source_fidelity_compression import run_analysis as run_stage271
from stage303_shared_base_bias_cross_model_joint_review import run_analysis as run_stage303
from stage324_first_principles_cross_model_reinforced_review import run_analysis as run_stage324
from stage328_cross_model_common_core_compression import run_analysis as run_stage328


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage363_cross_model_raw_expansion_20260325"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s263 = run_stage263(force=False)
    s264 = run_stage264(force=False)
    s265 = run_stage265(force=False)
    s266 = run_stage266(force=False)
    s267 = run_stage267(force=False)
    s271 = run_stage271(force=False)
    s303 = run_stage303(force=False)
    s324 = run_stage324(force=False)
    s328 = run_stage328(force=False)

    cross_model_rows = []

    for row in s263["model_rows"]:
        cross_model_rows.append(
            {
                "source_stage": "stage263",
                "model_name": row["display_name"],
                "metric_name": "direct_behavior_score",
                "strength": float(row["direct_score"]),
            }
        )
        for category_row in row["category_rows"]:
            cross_model_rows.append(
                {
                    "source_stage": "stage263",
                    "model_name": row["display_name"],
                    "metric_name": f"category::{category_row['category']}",
                    "strength": float(category_row["score"]),
                }
            )

    for row in s264["model_rows"]:
        cross_model_rows.append(
            {
                "source_stage": "stage264",
                "model_name": row["display_name"],
                "metric_name": "historical_structure_score",
                "strength": float(row["historical_structure_score"]),
            }
        )
        cross_model_rows.append(
            {
                "source_stage": "stage264",
                "model_name": row["display_name"],
                "metric_name": "complete_score",
                "strength": float(row["complete_score"]),
            }
        )
        for part_name, value in row["structure_parts"].items():
            cross_model_rows.append(
                {
                    "source_stage": "stage264",
                    "model_name": row["display_name"],
                    "metric_name": f"structure::{part_name}",
                    "strength": float(value),
                }
            )

    for row in s265["model_rows"]:
        cross_model_rows.append(
            {
                "source_stage": "stage265",
                "model_name": row["display_name"],
                "metric_name": "complete_final_score",
                "strength": float(row["complete_score"]),
            }
        )
    cross_model_rows.append(
        {
            "source_stage": "stage265",
            "model_name": "Qwen_vs_DeepSeek",
            "metric_name": "score_gap",
            "strength": float(s265["score_gap"]),
        }
    )

    for model_row in s266["model_rows"]:
        cross_model_rows.append(
            {
                "source_stage": "stage266",
                "model_name": model_row["display_name"],
                "metric_name": "parameter_hook_score",
                "strength": float(model_row["parameter_hook_score"]),
            }
        )
        for contrast_row in model_row["contrast_rows"]:
            cross_model_rows.append(
                {
                    "source_stage": "stage266",
                    "model_name": model_row["display_name"],
                    "metric_name": f"contrast::{contrast_row['contrast_name']}",
                    "strength": float(contrast_row["contrast_score"]),
                }
            )
            for layer_row in contrast_row["layer_rows"]:
                cross_model_rows.append(
                    {
                        "source_stage": "stage266",
                        "model_name": model_row["display_name"],
                        "metric_name": f"{contrast_row['contrast_name']}::{layer_row['layer_name']}",
                        "strength": float(layer_row["mean_abs_delta"]),
                    }
                )

    for model_row in s267["model_rows"]:
        cross_model_rows.append(
            {
                "source_stage": "stage267",
                "model_name": model_row["display_name"],
                "metric_name": "same_class_score",
                "strength": float(model_row["same_class_score"]),
            }
        )
        for family_row in model_row["family_rows"]:
            cross_model_rows.append(
                {
                    "source_stage": "stage267",
                    "model_name": model_row["display_name"],
                    "metric_name": f"family::{family_row['family']}::margin_mean",
                    "strength": float(family_row["margin_mean"]),
                }
            )
            cross_model_rows.append(
                {
                    "source_stage": "stage267",
                    "model_name": model_row["display_name"],
                    "metric_name": f"family::{family_row['family']}::positive_rate",
                    "strength": float(family_row["positive_rate"]),
                }
            )

    for row in s271["model_rows"]:
        cross_model_rows.append(
            {
                "source_stage": "stage271",
                "model_name": row["display_name"],
                "metric_name": "natural_fidelity_score",
                "strength": float(row["natural_fidelity_score"]),
            }
        )
        cross_model_rows.append(
            {
                "source_stage": "stage271",
                "model_name": row["display_name"],
                "metric_name": "repair_fidelity_score",
                "strength": float(row["repair_fidelity_score"]),
            }
        )
        cross_model_rows.append(
            {
                "source_stage": "stage271",
                "model_name": row["display_name"],
                "metric_name": "repair_gain_score",
                "strength": float(row["repair_gain_score"]),
            }
        )
        for probe_row in row["probe_rows"]:
            cross_model_rows.append(
                {
                    "source_stage": "stage271",
                    "model_name": row["display_name"],
                    "metric_name": f"family::{probe_row['family']}::natural_margin",
                    "strength": float(probe_row["natural_margin"]),
                }
            )
            cross_model_rows.append(
                {
                    "source_stage": "stage271",
                    "model_name": row["display_name"],
                    "metric_name": f"family::{probe_row['family']}::repair_margin",
                    "strength": float(probe_row["repair_margin"]),
                }
            )

    for row in s303["model_rows"]:
        cross_model_rows.append(
            {
                "source_stage": "stage303",
                "model_name": row["display_name"],
                "metric_name": "shared_base_bias_joint_score",
                "strength": float(row["parameter_bridge_score"]),
            }
        )

    cross_model_rows.append(
        {
            "source_stage": "stage324",
            "model_name": "cross_model",
            "metric_name": "cross_model_score",
            "strength": float(s324["cross_model_score"]),
        }
    )

    for row in s328["common_rows"]:
        cross_model_rows.append(
            {
                "source_stage": "stage328",
                "model_name": row["core_name"],
                "metric_name": "common_core_strength",
                "strength": float(row["strength"]),
            }
        )

    expansion_score = sum(float(row["strength"]) for row in cross_model_rows) / max(1, len(cross_model_rows))

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage363_cross_model_raw_expansion",
        "title": "跨模型原始数据扩张图",
        "status_short": "cross_model_raw_expansion_ready",
        "expansion_score": float(expansion_score),
        "cross_model_rows": cross_model_rows,
        "top_gap_name": "当前阶段已将跨模型原始行扩展到行为层、结构层、共同主核层和竞争层。",
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="跨模型原始数据扩张图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
