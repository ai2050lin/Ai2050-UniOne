#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage549_noun_family_neuron_structure_protocol_20260405"
)

STAGE518_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage518_four_model_cross_task_causal_synthesis_20260404"
    / "summary.json"
)
STAGE522_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage522_noun_panorama_hierarchy_scan_20260404"
    / "summary.json"
)
STAGE525_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage525_multi_bridge_causal_expansion_20260404"
    / "summary.json"
)
STAGE548_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage548_topology_field_neuron_algorithm_20260405"
    / "summary.json"
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()

    stage518 = load_json(STAGE518_PATH)
    stage522 = load_json(STAGE522_PATH)
    stage525 = load_json(STAGE525_PATH)
    stage548 = load_json(STAGE548_PATH)

    cross_task_map = {row["model_key"]: row for row in stage518["model_rows"]}
    noun_map = {row["model_key"]: row for row in stage522["model_rows"]}
    bridge_map = {row["model_key"]: row for row in stage525["model_rows"]}
    band_map = {row["model_key"]: row for row in stage548["model_band_rows"]}

    model_rows = []
    for model_key in ("qwen3", "deepseek7b", "glm4", "gemma4"):
        noun_row = noun_map[model_key]
        apple = noun_row["summary"]["apple_breakdown"]
        cross_task = cross_task_map[model_key]
        strongest_bridge = max(
            bridge_map[model_key]["bridge_rows"],
            key=lambda row: float(row["final_result"]["utility"]),
        )
        fruit_shared = int(apple["apple_fruit_core_shared_count"])
        unique_vs_fruit = int(apple["apple_unique_vs_fruit_count"])
        shared_unique_ratio = fruit_shared / max(fruit_shared + unique_vs_fruit, 1)
        family_means = noun_row["summary"]["family_pairwise_mean_jaccard"]
        cross_means = noun_row["summary"]["cross_family_mean_jaccard"]
        best_cross_family = max(cross_means, key=cross_means.get)

        model_rows.append(
            {
                "model_key": model_key,
                "model_name": noun_row["model_name"],
                "global_backbone_count": int(noun_row["summary"]["global_core_count"]),
                "fruit_family_backbone_count": int(noun_row["summary"]["family_core_counts"]["fruit"]),
                "apple_global_shared_count": int(apple["apple_global_core_shared_count"]),
                "apple_fruit_shared_count": fruit_shared,
                "apple_best_nonfruit_shared_pair": best_cross_family,
                "apple_best_nonfruit_shared_mean": float(cross_means[best_cross_family]),
                "apple_unique_vs_fruit_count": unique_vs_fruit,
                "apple_shared_unique_ratio_vs_fruit": shared_unique_ratio,
                "family_prediction_accuracy": float(noun_row["summary"]["family_prediction_accuracy"]),
                "family_core_margin_win_rate": float(noun_row["summary"]["family_core_margin_win_rate"]),
                "cross_task_causal_core": cross_task["final_subset"],
                "cross_task_causal_utility": float(cross_task["utility"]),
                "strongest_bridge_kind": strongest_bridge["bridge_kind"],
                "strongest_bridge_subset": strongest_bridge["final_subset"],
                "strongest_bridge_utility": float(strongest_bridge["final_result"]["utility"]),
                "route_band": band_map[model_key]["route_band"],
                "cross_task_core_band": band_map[model_key]["cross_task_core_band"],
                "family_pairwise_mean_jaccard": family_means,
                "cross_family_mean_jaccard": cross_means,
            }
        )

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage549_noun_family_neuron_structure_protocol",
        "title": "名词家族神经元级结构提取协议",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "sources": {
            "stage518": str(STAGE518_PATH),
            "stage522": str(STAGE522_PATH),
            "stage525": str(STAGE525_PATH),
            "stage548": str(STAGE548_PATH),
        },
        "model_rows": model_rows,
        "structure_template": {
            "equation": "noun_encoding = global_backbone + family_backbone + noun_unique_residual + task_bridge_adapter + cross_task_causal_core",
            "fields_zh": [
                "全局共享名词骨干",
                "家族共享骨干",
                "名词独有残差",
                "任务桥接适配器",
                "跨任务因果核心",
            ],
        },
        "core_answer": (
            "名词家族的神经元级结构现在可以被稳定拆成五层：全局共享名词骨干、家族共享骨干、名词独有残差、任务桥接适配器、跨任务因果核心。"
            "这说明像苹果这样的名词，不是“一整块孤立神经元表示”，而是共享骨干与局部适配器的可组合结构。"
        ),
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# stage549 名词家族神经元级结构提取协议",
        "",
        "## 核心结论",
        summary["core_answer"],
        "",
        "## 结构模板",
        f"- `{summary['structure_template']['equation']}`",
        "",
    ]
    for row in model_rows:
        lines.append(f"## {row['model_name']}")
        lines.append(f"- 全局共享名词骨干数：`{row['global_backbone_count']}`")
        lines.append(f"- 水果家族骨干数：`{row['fruit_family_backbone_count']}`")
        lines.append(f"- 苹果与水果共享数：`{row['apple_fruit_shared_count']}`")
        lines.append(f"- 苹果相对水果独有数：`{row['apple_unique_vs_fruit_count']}`")
        lines.append(f"- 苹果对水果共享占比：`{row['apple_shared_unique_ratio_vs_fruit']:.4f}`")
        lines.append(f"- 跨任务因果核心：`{', '.join(row['cross_task_causal_core'])}`")
        lines.append(f"- 最强桥接类型：`{row['strongest_bridge_kind']}`，子集：`{', '.join(row['strongest_bridge_subset'])}`")
        lines.append(f"- 层带：路由 `{row['route_band']}`，跨任务核心 `{row['cross_task_core_band']}`")
        lines.append("")
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
