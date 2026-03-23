#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage149_rolling_expansion_scheduler_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE149_ROLLING_EXPANSION_SCHEDULER_REPORT.md"

STAGE141_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage141_cross_model_layer_isomorphism_20260323" / "summary.json"
STAGE143_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage143_triple_model_joint_variable_inversion_20260323" / "summary.json"
STAGE148_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage148_variable_identifiability_dataset_pack_20260323" / "summary.json"
STAGE139_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage139_qwen3_language_validation_suite_20260323" / "summary.json"
STAGE140_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage140_deepseek_language_validation_suite_20260323" / "summary.json"

VARIABLE_ORDER = ["a", "r", "f", "g", "q", "b"]


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def roundup_to_16(value: float) -> int:
    return int(math.ceil(value / 16.0) * 16)


def summarize_model_metrics(summary: Dict[str, object]) -> Dict[str, float]:
    transfer = summary["transfer_summary"]
    field = summary["field_summary"]
    return {
        "syntax_stability_rate": float(transfer["qwen_core_metrics"]["syntax_stability_rate"]),
        "anaphora_ellipsis_score": float(summary["anaphora_summary"]["anaphora_ellipsis_score"]),
        "complex_discourse_score": float(summary["discourse_summary"]["complex_discourse_noun_propagation_score"]),
        "route_shift_score": float(transfer["qwen_core_metrics"]["adverb_context_route_shift_score"]),
        "joint_score": float(summary["joint_summary"]["noun_verb_joint_propagation_score"]),
        "result_score": float(summary["result_summary"]["noun_verb_result_chain_score"]),
        "q_weight": float(field["best_weights"]["q"]),
        "b_weight": float(field["best_weights"]["b"]),
        "g_weight": float(field["best_weights"]["g"]),
    }


def build_support_scores(
    qwen_metrics: Dict[str, float],
    deepseek_metrics: Dict[str, float],
) -> Dict[str, float]:
    return {
        "a": mean([qwen_metrics["syntax_stability_rate"], deepseek_metrics["syntax_stability_rate"]]),
        "r": mean([qwen_metrics["anaphora_ellipsis_score"], deepseek_metrics["anaphora_ellipsis_score"]]),
        "f": mean([qwen_metrics["complex_discourse_score"], deepseek_metrics["complex_discourse_score"]]),
        "g": mean(
            [
                qwen_metrics["route_shift_score"],
                deepseek_metrics["route_shift_score"],
                qwen_metrics["joint_score"],
                deepseek_metrics["joint_score"],
                qwen_metrics["result_score"],
                deepseek_metrics["result_score"],
                qwen_metrics["g_weight"],
                deepseek_metrics["g_weight"],
            ]
        ),
        "q": mean([qwen_metrics["q_weight"], deepseek_metrics["q_weight"]]),
        "b": mean([qwen_metrics["b_weight"], deepseek_metrics["b_weight"]]),
    }


def build_proxy_strengths(triple_summary: Dict[str, object]) -> Dict[str, float]:
    proxy_values = {name: float(triple_summary[f"{name}_proxy_mean"]) for name in VARIABLE_ORDER}
    strongest = max(proxy_values.values()) if proxy_values else 1.0
    return {name: clamp01(proxy_values[name] / strongest) for name in VARIABLE_ORDER}


def weakest_model_for_variable(variable_name: str, qwen_metrics: Dict[str, float], deepseek_metrics: Dict[str, float]) -> str:
    metric_name = {
        "a": "syntax_stability_rate",
        "r": "anaphora_ellipsis_score",
        "f": "complex_discourse_score",
        "g": "joint_score",
        "q": "q_weight",
        "b": "b_weight",
    }[variable_name]
    return "Qwen3-4B" if qwen_metrics[metric_name] <= deepseek_metrics[metric_name] else "DeepSeek-R1-Distill-Qwen-7B"


def recommended_difficulties(priority_score: float) -> List[str]:
    if priority_score >= 0.75:
        return ["hard", "adversarial"]
    if priority_score >= 0.55:
        return ["medium", "hard", "adversarial"]
    return ["medium", "hard"]


def build_schedule_rows(
    bundle_summary: Dict[str, object],
    triple_summary: Dict[str, object],
    qwen_summary: Dict[str, object],
    deepseek_summary: Dict[str, object],
    layer_summary: Dict[str, object],
) -> List[Dict[str, object]]:
    bundle_rows = {str(row["variable_name"]): row for row in bundle_summary["variable_rows"]}
    qwen_metrics = summarize_model_metrics(qwen_summary)
    deepseek_metrics = summarize_model_metrics(deepseek_summary)
    support_scores = build_support_scores(qwen_metrics, deepseek_metrics)
    proxy_strengths = build_proxy_strengths(triple_summary)
    layer_penalty = 1.0 - clamp01(float(layer_summary["mean_layer_isomorphism_score"]))
    rows: List[Dict[str, object]] = []

    for variable_name in VARIABLE_ORDER:
        bundle = bundle_rows[variable_name]
        identifiability_score = float(bundle["variable_identifiability_score"])
        transfer_support = support_scores[variable_name]
        proxy_strength = proxy_strengths[variable_name]
        layer_bonus_penalty = layer_penalty if variable_name in {"a", "g", "b"} else 0.0
        priority_score = clamp01(
            0.45 * (1.0 - transfer_support)
            + 0.35 * (1.0 - proxy_strength)
            + 0.10 * (1.0 - identifiability_score)
            + 0.10 * layer_bonus_penalty
        )
        multiplier = 3.0 if priority_score >= 0.75 else 2.5 if priority_score >= 0.60 else 2.0 if priority_score >= 0.45 else 1.5
        current_case_count = int(bundle["case_count"])
        target_case_count = roundup_to_16(current_case_count * multiplier)
        additional_case_count = max(16, target_case_count - current_case_count)
        contrast_types = [value for value in bundle["contrast_types"] if value != "primary"]
        rows.append(
            {
                "variable_name": variable_name,
                "priority_score": priority_score,
                "transfer_support_score": transfer_support,
                "proxy_strength_score": proxy_strength,
                "identifiability_score": identifiability_score,
                "weakest_model_name": weakest_model_for_variable(variable_name, qwen_metrics, deepseek_metrics),
                "recommended_family_names": list(bundle["family_names"]),
                "recommended_contrast_types": contrast_types,
                "recommended_difficulties": recommended_difficulties(priority_score),
                "current_case_count": current_case_count,
                "recommended_additional_case_count": additional_case_count,
                "supporting_variables": list(bundle["supporting_variables"]),
                "hypothesis": bundle["hypothesis"],
            }
        )
    rows.sort(key=lambda row: (-float(row["priority_score"]), VARIABLE_ORDER.index(str(row["variable_name"]))))
    return rows


def build_summary(
    schedule_rows: List[Dict[str, object]],
    layer_summary: Dict[str, object],
    triple_summary: Dict[str, object],
) -> Dict[str, object]:
    total_additional_case_count = sum(int(row["recommended_additional_case_count"]) for row in schedule_rows)
    highest_priority_variable = str(schedule_rows[0]["variable_name"])
    lowest_priority_variable = str(schedule_rows[-1]["variable_name"])
    mean_priority_score = mean([float(row["priority_score"]) for row in schedule_rows])
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage149_rolling_expansion_scheduler",
        "title": "滚动扩量调度块",
        "status_short": "rolling_expansion_scheduler_ready",
        "variable_count": len(schedule_rows),
        "mean_priority_score": mean_priority_score,
        "highest_priority_variable": highest_priority_variable,
        "lowest_priority_variable": lowest_priority_variable,
        "total_additional_case_count": total_additional_case_count,
        "mean_layer_isomorphism_score": float(layer_summary["mean_layer_isomorphism_score"]),
        "triple_joint_inversion_score": float(triple_summary["joint_inversion_score"]),
        "schedule_rows": schedule_rows,
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage149: 滚动扩量调度块",
        "",
        "## 核心结果",
        f"- 变量数: {summary['variable_count']}",
        f"- 平均优先级分数: {summary['mean_priority_score']:.4f}",
        f"- 最高优先变量: {summary['highest_priority_variable']}",
        f"- 最低优先变量: {summary['lowest_priority_variable']}",
        f"- 建议新增样本总数: {summary['total_additional_case_count']}",
        f"- 跨模型层同构分数: {summary['mean_layer_isomorphism_score']:.4f}",
        f"- 三模型联合反演分数: {summary['triple_joint_inversion_score']:.4f}",
        "",
        "## 调度表",
    ]
    for row in summary["schedule_rows"]:
        lines.append(
            "- "
            f"{row['variable_name']}: "
            f"priority={row['priority_score']:.4f}; "
            f"weakest_model={row['weakest_model_name']}; "
            f"families={','.join(row['recommended_family_names'])}; "
            f"contrasts={','.join(row['recommended_contrast_types'])}; "
            f"difficulties={','.join(row['recommended_difficulties'])}; "
            f"add_cases={row['recommended_additional_case_count']}"
        )
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> Dict[str, object]:
    if not force and SUMMARY_PATH.exists():
        return load_json(SUMMARY_PATH)
    layer_summary = load_json(STAGE141_SUMMARY_PATH)
    triple_summary = load_json(STAGE143_SUMMARY_PATH)
    bundle_summary = load_json(STAGE148_SUMMARY_PATH)
    qwen_summary = load_json(STAGE139_SUMMARY_PATH)
    deepseek_summary = load_json(STAGE140_SUMMARY_PATH)
    schedule_rows = build_schedule_rows(bundle_summary, triple_summary, qwen_summary, deepseek_summary, layer_summary)
    summary = build_summary(schedule_rows, layer_summary, triple_summary)
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="滚动扩量调度块")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重算")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
