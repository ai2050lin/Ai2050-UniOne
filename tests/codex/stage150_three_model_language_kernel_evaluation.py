#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage150_three_model_language_kernel_evaluation_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE150_THREE_MODEL_LANGUAGE_KERNEL_EVALUATION_REPORT.md"

STAGE121_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage121_adverb_gate_bridge_probe_20260323" / "summary.json"
STAGE123_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage123_route_shift_layer_localization_20260323" / "summary.json"
STAGE130_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage130_multisyntax_noun_context_probe_20260323" / "summary.json"
STAGE133_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage133_complex_discourse_noun_propagation_20260323" / "summary.json"
STAGE134_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage134_noun_verb_joint_propagation_20260323" / "summary.json"
STAGE136_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage136_anaphora_ellipsis_propagation_20260323" / "summary.json"
STAGE137_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage137_noun_verb_result_chain_20260323" / "summary.json"
STAGE138_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage138_conditional_gating_field_reconstruction_20260323" / "summary.json"
STAGE139_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage139_qwen3_language_validation_suite_20260323" / "summary.json"
STAGE140_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage140_deepseek_language_validation_suite_20260323" / "summary.json"
STAGE141_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage141_cross_model_layer_isomorphism_20260323" / "summary.json"
STAGE143_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage143_triple_model_joint_variable_inversion_20260323" / "summary.json"


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def phenomenon_verdict(scores: List[float]) -> str:
    min_score = min(scores)
    mean_score = mean(scores)
    if min_score >= 0.60 and mean_score >= 0.65:
        return "stable_core"
    if mean_score >= 0.50:
        return "partial_core"
    return "weak_core"


def build_model_rows() -> List[Dict[str, object]]:
    stage121 = load_json(STAGE121_SUMMARY_PATH)
    stage123 = load_json(STAGE123_SUMMARY_PATH)
    stage130 = load_json(STAGE130_SUMMARY_PATH)
    stage133 = load_json(STAGE133_SUMMARY_PATH)
    stage134 = load_json(STAGE134_SUMMARY_PATH)
    stage136 = load_json(STAGE136_SUMMARY_PATH)
    stage137 = load_json(STAGE137_SUMMARY_PATH)
    stage138 = load_json(STAGE138_SUMMARY_PATH)
    stage139 = load_json(STAGE139_SUMMARY_PATH)
    stage140 = load_json(STAGE140_SUMMARY_PATH)

    gpt2_row = {
        "model_key": "gpt2",
        "display_name": "GPT-2",
        "test_mode": "历史真实测试快照复用",
        "bridge_score": float(stage121["adverb_gate_bridge_score"]),
        "route_shift_dynamic_score": float(stage123["source_dynamic_score"]),
        "route_localization_score": float(stage123["route_shift_layer_localization_score"]),
        "syntax_anchor_score": float(stage130["syntax_stability_rate"]),
        "discourse_score": float(stage133["complex_discourse_noun_propagation_score"]),
        "joint_score": float(stage134["noun_verb_joint_propagation_score"]),
        "anaphora_score": float(stage136["anaphora_ellipsis_propagation_score"]),
        "result_chain_score": float(stage137["noun_verb_result_chain_score"]),
        "conditional_field_score": float(stage138["conditional_gating_field_score"]),
        "conditional_field_formula": str(stage138["best_formula"]),
        "transfer_verdict": "reference_anchor",
        "theory_check_pass_rate": 1.0,
    }

    qwen_row = {
        "model_key": "qwen3",
        "display_name": "Qwen3-4B",
        "test_mode": "本轮真实实跑",
        "bridge_score": float(stage139["field_summary"]["stage121_adverb_gate_bridge_score"]),
        "route_shift_dynamic_score": float(stage139["transfer_summary"]["qwen_core_metrics"]["adverb_context_route_shift_score"]),
        "route_localization_score": float(stage139["field_summary"]["stage123_route_shift_layer_localization_score"]),
        "syntax_anchor_score": float(stage139["transfer_summary"]["qwen_core_metrics"]["syntax_stability_rate"]),
        "discourse_score": float(stage139["discourse_summary"]["complex_discourse_noun_propagation_score"]),
        "joint_score": float(stage139["joint_summary"]["noun_verb_joint_propagation_score"]),
        "anaphora_score": float(stage139["anaphora_summary"]["anaphora_ellipsis_score"]),
        "result_chain_score": float(stage139["result_summary"]["noun_verb_result_chain_score"]),
        "conditional_field_score": float(stage139["field_summary"]["conditional_gating_field_score"]),
        "conditional_field_formula": str(stage139["field_summary"]["best_formula"]),
        "transfer_verdict": str(stage139["transfer_summary"]["transfer_verdict"]),
        "theory_check_pass_rate": float(stage139["transfer_summary"]["theory_check_pass_rate"]),
    }

    deepseek_row = {
        "model_key": "deepseek7b",
        "display_name": "DeepSeek-R1-Distill-Qwen-7B",
        "test_mode": "本轮真实实跑",
        "bridge_score": float(stage140["field_summary"]["stage121_adverb_gate_bridge_score"]),
        "route_shift_dynamic_score": float(stage140["transfer_summary"]["qwen_core_metrics"]["adverb_context_route_shift_score"]),
        "route_localization_score": float(stage140["field_summary"]["stage123_route_shift_layer_localization_score"]),
        "syntax_anchor_score": float(stage140["transfer_summary"]["qwen_core_metrics"]["syntax_stability_rate"]),
        "discourse_score": float(stage140["discourse_summary"]["complex_discourse_noun_propagation_score"]),
        "joint_score": float(stage140["joint_summary"]["noun_verb_joint_propagation_score"]),
        "anaphora_score": float(stage140["anaphora_summary"]["anaphora_ellipsis_score"]),
        "result_chain_score": float(stage140["result_summary"]["noun_verb_result_chain_score"]),
        "conditional_field_score": float(stage140["field_summary"]["conditional_gating_field_score"]),
        "conditional_field_formula": str(stage140["field_summary"]["best_formula"]),
        "transfer_verdict": str(stage140["transfer_summary"]["transfer_verdict"]),
        "theory_check_pass_rate": float(stage140["transfer_summary"]["theory_check_pass_rate"]),
    }
    return [gpt2_row, qwen_row, deepseek_row]


def build_phenomenon_rows(model_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    phenomenon_specs = [
        ("adverb_route_shift", "route_shift_dynamic_score"),
        ("conditional_field_fit", "conditional_field_score"),
        ("discourse_remention", "discourse_score"),
        ("syntax_anchor_band", "syntax_anchor_score"),
        ("anaphora_repair", "anaphora_score"),
        ("noun_verb_joint_chain", "joint_score"),
        ("noun_verb_result_chain", "result_chain_score"),
    ]
    rows = []
    for phenomenon_name, metric_name in phenomenon_specs:
        model_scores = {row["display_name"]: float(row[metric_name]) for row in model_rows}
        score_values = list(model_scores.values())
        rows.append(
            {
                "phenomenon_name": phenomenon_name,
                "metric_name": metric_name,
                "mean_score": mean(score_values),
                "min_score": min(score_values),
                "max_score": max(score_values),
                "verdict": phenomenon_verdict(score_values),
                "model_scores": model_scores,
            }
        )
    return rows


def build_summary(model_rows: List[Dict[str, object]], phenomenon_rows: List[Dict[str, object]]) -> Dict[str, object]:
    layer_summary = load_json(STAGE141_SUMMARY_PATH)
    triple_summary = load_json(STAGE143_SUMMARY_PATH)
    stable_core_count = sum(1 for row in phenomenon_rows if row["verdict"] == "stable_core")
    partial_core_count = sum(1 for row in phenomenon_rows if row["verdict"] == "partial_core")
    weak_core_count = sum(1 for row in phenomenon_rows if row["verdict"] == "weak_core")
    overall_kernel_score = clamp01(
        0.35 * mean(float(row["theory_check_pass_rate"]) for row in model_rows)
        + 0.20 * float(layer_summary["mean_layer_isomorphism_score"])
        + 0.20 * float(triple_summary["joint_inversion_score"])
        + 0.25 * mean(float(row["mean_score"]) for row in phenomenon_rows)
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage150_three_model_language_kernel_evaluation",
        "title": "三模型语言主核评估块",
        "status_short": "three_model_language_kernel_ready",
        "model_count": len(model_rows),
        "phenomenon_count": len(phenomenon_rows),
        "stable_core_count": stable_core_count,
        "partial_core_count": partial_core_count,
        "weak_core_count": weak_core_count,
        "overall_kernel_score": overall_kernel_score,
        "layer_isomorphism_score": float(layer_summary["mean_layer_isomorphism_score"]),
        "joint_inversion_score": float(triple_summary["joint_inversion_score"]),
        "joint_best_formula": str(triple_summary["best_formula"]),
        "weakest_proxy_name": str(triple_summary["weakest_proxy_name"]),
        "strongest_proxy_name": str(triple_summary["strongest_proxy_name"]),
        "model_rows": model_rows,
        "phenomenon_rows": phenomenon_rows,
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage150: 三模型语言主核评估块",
        "",
        "## 核心结果",
        f"- 模型数: {summary['model_count']}",
        f"- 现象数: {summary['phenomenon_count']}",
        f"- 稳定主核数: {summary['stable_core_count']}",
        f"- 过渡主核数: {summary['partial_core_count']}",
        f"- 弱主核数: {summary['weak_core_count']}",
        f"- 总体主核分数: {summary['overall_kernel_score']:.4f}",
        f"- 层同构分数: {summary['layer_isomorphism_score']:.4f}",
        f"- 三模型联合反演式: {summary['joint_best_formula']}",
        "",
        "## 模型汇总",
    ]
    for row in summary["model_rows"]:
        lines.append(
            "- "
            f"{row['display_name']}: "
            f"test_mode={row['test_mode']}; "
            f"transfer={row['transfer_verdict']}; "
            f"pass_rate={row['theory_check_pass_rate']:.4f}; "
            f"field={row['conditional_field_formula']}"
        )
    lines.append("")
    lines.append("## 现象汇总")
    for row in summary["phenomenon_rows"]:
        score_text = ", ".join(f"{name}={value:.4f}" for name, value in row["model_scores"].items())
        lines.append(
            "- "
            f"{row['phenomenon_name']}: "
            f"verdict={row['verdict']}; "
            f"mean={row['mean_score']:.4f}; "
            f"min={row['min_score']:.4f}; "
            f"scores={score_text}"
        )
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> Dict[str, object]:
    if not force and SUMMARY_PATH.exists():
        return load_json(SUMMARY_PATH)
    model_rows = build_model_rows()
    phenomenon_rows = build_phenomenon_rows(model_rows)
    summary = build_summary(model_rows, phenomenon_rows)
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="三模型语言主核评估块")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重算")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
