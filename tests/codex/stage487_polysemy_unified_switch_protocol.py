#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / f"stage487_polysemy_unified_switch_protocol_{time.strftime('%Y%m%d')}"
)

STAGE433_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage433_polysemous_noun_family_generalization_20260402" / "summary.json"
STAGE447_QWEN3_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage447_polysemy_family_switch_protocol_20260403" / "qwen3" / "summary.json"
STAGE447_DEEPSEEK_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage447_polysemy_family_switch_protocol_20260403" / "deepseek7b_cpu" / "summary.json"
STAGE448_QWEN3_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage448_apple_switch_layer_scan_and_neuron_counts_20260403" / "qwen3_cpu" / "summary.json"
STAGE448_DEEPSEEK_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage448_apple_switch_layer_scan_and_neuron_counts_20260403" / "deepseek7b_cpu" / "summary.json"
STAGE479_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage479_apple_switch_mixed_circuit_search_20260403" / "summary.json"
STAGE480_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage480_apple_switch_exact_core_scan_20260403" / "summary.json"
STAGE481_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage481_apple_switch_pair_order_analysis_20260403" / "summary.json"
STAGE482_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage482_apple_switch_direction_tracking_20260403" / "summary.json"
STAGE483_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage483_apple_switch_residual_basis_20260403" / "summary.json"
STAGE484_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage484_apple_switch_signed_residual_basis_20260403" / "summary.json"


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def find_model_result(summary: Dict[str, object], model_key: str) -> Dict[str, object]:
    model_results = summary.get("model_results")
    if isinstance(model_results, list):
        for row in model_results:
            if row.get("model_key") == model_key:
                return row
    model_map = summary.get("models")
    if isinstance(model_map, dict) and model_key in model_map:
        return model_map[model_key]
    raise KeyError(f"Missing model_key={model_key}")


def find_noun_result(model_row: Dict[str, object], noun_id: str) -> Dict[str, object]:
    for row in model_row["noun_results"]:
        if row["noun_id"] == noun_id:
            return row
    raise KeyError(f"Missing noun_id={noun_id}")


def unit_map(model_row: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    return {unit["unit_id"]: unit for unit in model_row["units"]}


def build_model_summary(
    model_key: str,
    stage433_model: Dict[str, object],
    stage447_model: Dict[str, object],
    stage448_model: Dict[str, object],
    stage479_model: Dict[str, object],
    stage480_model: Dict[str, object],
    stage481_model: Dict[str, object],
    stage482_model: Dict[str, object],
    stage483_model: Dict[str, object],
    stage484_model: Dict[str, object],
) -> Dict[str, object]:
    noun_rows: List[Dict[str, object]] = []
    for noun_row in stage447_model["noun_results"]:
        noun_id = noun_row["noun_id"]
        stage433_noun = find_noun_result(stage433_model, noun_id)
        best_balance = stage433_noun["best_balance_layer"]
        noun_rows.append(
            {
                "noun_id": noun_id,
                "best_switch_layer": int(noun_row["best_switch_layer"]),
                "best_balance_layer": int(best_balance["layer_index"]),
                "sense_active_jaccard": float(noun_row["sense_active_jaccard"]),
                "ordinary_control_mean_active_jaccard": float(noun_row["ordinary_control_mean_active_jaccard"]),
                "ordinary_vs_polysemy_gap": float(noun_row["ordinary_vs_polysemy_gap"]),
                "shared_core_similarity": float(best_balance["shared_core_similarity"]),
                "structured_delta_ratio": float(best_balance["delta_structured_ratio"]),
                "readout_accuracy": float(best_balance["sense_readout_accuracy"]),
            }
        )

    stage480_candidates = stage480_model["candidates"]
    stage482_units = unit_map(stage482_model)
    stage483_units = unit_map(stage483_model)
    stage484_units = unit_map(stage484_model)
    stage448_result = stage448_model

    apple_extension = {
        "global_counts": stage448_result["global_counts"],
        "best_sensitive_layer": stage448_result["best_sensitive_layer"],
        "best_shared_layer": stage448_result["best_shared_layer"],
        "mixed_circuit_final_subset_ids": [row["candidate_id"] for row in stage479_model["final_subset"]],
        "exact_core_candidate_ids": [row["candidate_id"] for row in stage480_candidates],
        "exact_core_shapley_order": [row["candidate_id"] for row in sorted(stage480_candidates, key=lambda item: float(item["shapley_utility"]), reverse=True)],
        "order_roles": stage481_model["roles"],
        "best_order": stage481_model["utility_focus"]["best_order"]["order"],
        "direction_tracking_units": {
            unit_id: {
                "peak_effect_layer": int(stage482_units[unit_id]["tracking"]["peak_effect_layer"]),
                "late_mean_relative_drop": float(stage482_units[unit_id]["tracking"]["late_mean_relative_drop"]),
            }
            for unit_id in stage482_units
        },
        "residual_alignment_units": {
            unit_id: {
                "peak_contrast_alignment_layer": int(stage483_units[unit_id]["tracking"]["peak_contrast_alignment_layer"]),
                "peak_contrast_switch_coupling": float(stage483_units[unit_id]["tracking"]["peak_contrast_switch_coupling"]),
                "peak_pc1_explained_variance_ratio": float(stage483_units[unit_id]["tracking"]["peak_pc1_explained_variance_ratio"]),
            }
            for unit_id in stage483_units
        },
        "signed_tracking_units": {
            unit_id: {
                "forward_peak_layer": int(stage484_units[unit_id]["tracking"]["forward_peak_layer"]),
                "reverse_peak_layer": int(stage484_units[unit_id]["tracking"]["reverse_peak_layer"]),
                "late_mean_signed_contrast_switch_coupling": float(stage484_units[unit_id]["tracking"]["late_mean_signed_contrast_switch_coupling"]),
            }
            for unit_id in stage484_units
        },
    }

    return {
        "model_key": model_key,
        "model_name": stage447_model["model_name"],
        "noun_count": len(noun_rows),
        "mean_polysemy_active_jaccard": mean([float(row["sense_active_jaccard"]) for row in noun_rows]),
        "mean_ordinary_active_jaccard": mean([float(row["ordinary_control_mean_active_jaccard"]) for row in noun_rows]),
        "mean_gap": mean([float(row["ordinary_vs_polysemy_gap"]) for row in noun_rows]),
        "mean_shared_core_similarity": mean([float(row["shared_core_similarity"]) for row in noun_rows]),
        "large_gap_nouns": [row["noun_id"] for row in noun_rows if float(row["ordinary_vs_polysemy_gap"]) >= 0.10],
        "noun_rows": noun_rows,
        "apple_deep_extension": apple_extension,
    }


def build_summary() -> Dict[str, object]:
    stage433 = load_json(STAGE433_SUMMARY_PATH)
    stage447_qwen3 = load_json(STAGE447_QWEN3_PATH)
    stage447_deepseek = load_json(STAGE447_DEEPSEEK_PATH)
    stage448_qwen3 = load_json(STAGE448_QWEN3_PATH)
    stage448_deepseek = load_json(STAGE448_DEEPSEEK_PATH)
    stage479 = load_json(STAGE479_SUMMARY_PATH)
    stage480 = load_json(STAGE480_SUMMARY_PATH)
    stage481 = load_json(STAGE481_SUMMARY_PATH)
    stage482 = load_json(STAGE482_SUMMARY_PATH)
    stage483 = load_json(STAGE483_SUMMARY_PATH)
    stage484 = load_json(STAGE484_SUMMARY_PATH)

    qwen3_summary = build_model_summary(
        "qwen3",
        find_model_result(stage433, "qwen3"),
        find_model_result(stage447_qwen3, "qwen3"),
        find_model_result(stage448_qwen3, "qwen3"),
        stage479["models"]["qwen3"],
        stage480["models"]["qwen3"],
        stage481["models"]["qwen3"],
        stage482["models"]["qwen3"],
        stage483["models"]["qwen3"],
        stage484["models"]["qwen3"],
    )
    deepseek_summary = build_model_summary(
        "deepseek7b",
        find_model_result(stage433, "deepseek7b"),
        find_model_result(stage447_deepseek, "deepseek7b"),
        find_model_result(stage448_deepseek, "deepseek7b"),
        stage479["models"]["deepseek7b"],
        stage480["models"]["deepseek7b"],
        stage481["models"]["deepseek7b"],
        stage482["models"]["deepseek7b"],
        stage483["models"]["deepseek7b"],
        stage484["models"]["deepseek7b"],
    )

    qwen_nouns = {row["noun_id"]: row for row in qwen3_summary["noun_rows"]}
    deepseek_nouns = {row["noun_id"]: row for row in deepseek_summary["noun_rows"]}
    shared_large_gap_nouns = sorted(
        noun_id
        for noun_id in qwen_nouns
        if noun_id in deepseek_nouns
        and float(qwen_nouns[noun_id]["ordinary_vs_polysemy_gap"]) >= 0.10
        and float(deepseek_nouns[noun_id]["ordinary_vs_polysemy_gap"]) >= 0.10
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage487_polysemy_unified_switch_protocol",
        "title": "多义词统一切换机制正式协议",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": {"qwen3": qwen3_summary, "deepseek7b": deepseek_summary},
        "cross_model_summary": {
            "shared_large_gap_nouns": shared_large_gap_nouns,
            "shared_large_gap_support_rate": mean([1.0 if noun_id in shared_large_gap_nouns else 0.0 for noun_id in sorted(set(qwen_nouns.keys()) & set(deepseek_nouns.keys()))]),
            "mean_gap_qwen3": float(qwen3_summary["mean_gap"]),
            "mean_gap_deepseek7b": float(deepseek_summary["mean_gap"]),
            "core_answer": "当前最稳的统一结论是：多义词不是普通上下文扰动的放大版，而更像共享底座之上的低重合切换。四个多义词家族在两个模型里都出现了稳定的“普通上下文高重合、多义切换低重合”结构；苹果则额外提供了最深的机制证据，显示切换可以进一步压成头骨架或神经元锚点。",
        },
        "sources": {
            "stage433": str(STAGE433_SUMMARY_PATH),
            "stage447_qwen3": str(STAGE447_QWEN3_PATH),
            "stage447_deepseek7b": str(STAGE447_DEEPSEEK_PATH),
            "stage448_qwen3": str(STAGE448_QWEN3_PATH),
            "stage448_deepseek7b": str(STAGE448_DEEPSEEK_PATH),
            "stage479": str(STAGE479_SUMMARY_PATH),
            "stage480": str(STAGE480_SUMMARY_PATH),
            "stage481": str(STAGE481_SUMMARY_PATH),
            "stage482": str(STAGE482_SUMMARY_PATH),
            "stage483": str(STAGE483_SUMMARY_PATH),
            "stage484": str(STAGE484_SUMMARY_PATH),
        },
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "## 跨模型统一结论",
        f"- shared_large_gap_nouns = {', '.join(summary['cross_model_summary']['shared_large_gap_nouns'])}",
        f"- shared_large_gap_support_rate = {summary['cross_model_summary']['shared_large_gap_support_rate']:.4f}",
        f"- mean_gap_qwen3 = {summary['cross_model_summary']['mean_gap_qwen3']:.4f}",
        f"- mean_gap_deepseek7b = {summary['cross_model_summary']['mean_gap_deepseek7b']:.4f}",
        f"- core_answer = {summary['cross_model_summary']['core_answer']}",
        "",
    ]
    for model_key in ["qwen3", "deepseek7b"]:
        row = summary["models"][model_key]
        apple = row["apple_deep_extension"]
        lines.extend(
            [
                f"## 模型 {model_key}",
                f"- mean_polysemy_active_jaccard = {row['mean_polysemy_active_jaccard']:.4f}",
                f"- mean_ordinary_active_jaccard = {row['mean_ordinary_active_jaccard']:.4f}",
                f"- mean_gap = {row['mean_gap']:.4f}",
                f"- mean_shared_core_similarity = {row['mean_shared_core_similarity']:.4f}",
                f"- large_gap_nouns = {', '.join(row['large_gap_nouns'])}",
            ]
        )
        for noun_row in row["noun_rows"]:
            lines.append(
                f"- {noun_row['noun_id']}: switch=L{noun_row['best_switch_layer']}, balance=L{noun_row['best_balance_layer']}, "
                f"poly_jaccard={noun_row['sense_active_jaccard']:.4f}, ordinary_jaccard={noun_row['ordinary_control_mean_active_jaccard']:.4f}, "
                f"gap={noun_row['ordinary_vs_polysemy_gap']:.4f}, shared_core={noun_row['shared_core_similarity']:.4f}"
            )
        lines.extend(
            [
                f"- apple_best_sensitive_layer = L{apple['best_sensitive_layer']['layer_index']}",
                f"- apple_mixed_circuit_final_subset_ids = {', '.join(apple['mixed_circuit_final_subset_ids'])}",
                f"- apple_exact_core_shapley_order = {', '.join(apple['exact_core_shapley_order'])}",
                f"- apple_best_order = {' -> '.join(apple['best_order'])}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多义词统一切换机制正式协议")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary()
    write_outputs(summary, Path(args.output_dir))
    print(json.dumps({"status_short": "stage487_ready", "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
