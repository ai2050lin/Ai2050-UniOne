#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Dict, List

from multimodel_language_shared import (
    MODEL_SPECS,
    ablate_layer_component,
    candidate_score_map,
    discover_layers,
    evenly_spaced_layers,
    free_model,
    load_model_bundle,
    restore_layer_component,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage503_concept_hierarchy_causal_path_triple_model_20260404"
)
MODEL_KEYS = ["qwen3", "deepseek7b", "glm4"]

HIERARCHY_TASKS: List[Dict[str, object]] = [
    {
        "task_id": "food_parent",
        "name": "食物父类归属",
        "text": "苹果属于",
        "target": "水果",
        "candidates": ["水果", "蔬菜", "主食", "肉类"],
    },
    {
        "task_id": "vegetable_parent",
        "name": "蔬菜父类归属",
        "text": "白菜属于",
        "target": "蔬菜",
        "candidates": ["蔬菜", "水果", "主食", "肉类"],
    },
    {
        "task_id": "animal_parent",
        "name": "动物父类归属",
        "text": "麻雀属于",
        "target": "鸟类",
        "candidates": ["鸟类", "鱼类", "昆虫", "哺乳动物"],
    },
    {
        "task_id": "time_parent",
        "name": "时间父类归属",
        "text": "去年属于",
        "target": "过去",
        "candidates": ["过去", "未来", "现在", "空间"],
    },
    {
        "task_id": "color_parent",
        "name": "颜色父类归属",
        "text": "深蓝属于",
        "target": "蓝色",
        "candidates": ["蓝色", "绿色", "红色", "黄色"],
    },
    {
        "task_id": "spatial_parent",
        "name": "空间父类归属",
        "text": "里和外都属于",
        "target": "位置",
        "candidates": ["位置", "方向", "时间", "颜色"],
    },
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_task(model, tokenizer, task: dict) -> dict:
    sample_layers = evenly_spaced_layers(model, count=5)
    text = str(task["text"])
    target = str(task["target"])
    candidates = list(task["candidates"])
    baseline_scores = candidate_score_map(model, tokenizer, text, candidates)
    baseline_target = float(baseline_scores[target])
    baseline_pred = max(baseline_scores, key=baseline_scores.get)
    layer_rows = []

    for layer_idx in sample_layers:
        for component in ("attn", "mlp"):
            layer, original = ablate_layer_component(model, layer_idx, component)
            try:
                ablated_scores = candidate_score_map(model, tokenizer, text, candidates)
            finally:
                restore_layer_component(layer, component, original)
            ablated_target = float(ablated_scores[target])
            target_drop = baseline_target - ablated_target
            control_shift = statistics.mean(
                abs(float(ablated_scores[c]) - float(baseline_scores[c]))
                for c in candidates
                if c != target
            )
            supported = target_drop > max(0.01, control_shift * 2.0)
            layer_rows.append(
                {
                    "layer": layer_idx,
                    "component": component,
                    "baseline_target_score": round(baseline_target, 6),
                    "ablated_target_score": round(ablated_target, 6),
                    "target_drop": round(target_drop, 6),
                    "control_abs_shift": round(control_shift, 6),
                    "specific_support": supported,
                }
            )

    attn_peak = max(
        (row for row in layer_rows if row["component"] == "attn"),
        key=lambda x: x["target_drop"],
    )
    mlp_peak = max(
        (row for row in layer_rows if row["component"] == "mlp"),
        key=lambda x: x["target_drop"],
    )
    return {
        "task_id": task["task_id"],
        "task_name": task["name"],
        "baseline_scores": {k: round(v, 6) for k, v in baseline_scores.items()},
        "baseline_pred": baseline_pred,
        "layer_rows": layer_rows,
        "attn_peak": attn_peak,
        "mlp_peak": mlp_peak,
        "best_component": "attn" if attn_peak["target_drop"] > mlp_peak["target_drop"] else "mlp",
    }


def run_model(model_key: str) -> dict:
    model, tokenizer = load_model_bundle(model_key, prefer_cuda=False)
    try:
        task_rows = [run_task(model, tokenizer, task) for task in HIERARCHY_TASKS]
        support_rows = [
            row
            for task in task_rows
            for row in task["layer_rows"]
            if row["specific_support"]
        ]
        return {
            "model_label": MODEL_SPECS[model_key]["label"],
            "layer_count": len(discover_layers(model)),
            "task_rows": task_rows,
            "aggregate": {
                "specific_support_count": len(support_rows),
                "task_count": len(HIERARCHY_TASKS),
                "best_component_counts": {
                    "attn": sum(1 for task in task_rows if task["best_component"] == "attn"),
                    "mlp": sum(1 for task in task_rows if task["best_component"] == "mlp"),
                },
                "mean_best_target_drop": round(
                    statistics.mean(max(task["attn_peak"]["target_drop"], task["mlp_peak"]["target_drop"]) for task in task_rows),
                    6,
                ),
            },
        }
    finally:
        free_model(model)


def build_report(summary: dict) -> str:
    lines = ["# stage503 概念层次因果路径三模型测试", ""]
    for model_key, row in summary["models"].items():
        agg = row["aggregate"]
        lines.append(f"## {model_key}")
        lines.append("")
        lines.append(f"- 特异因果支持层数: `{agg['specific_support_count']}`")
        lines.append(f"- 最优组件统计: `{json.dumps(agg['best_component_counts'], ensure_ascii=False)}`")
        lines.append(f"- 平均最强下降: `{agg['mean_best_target_drop']}`")
        lines.append("")
    lines.append("## 综合结论")
    lines.append("")
    lines.append(f"- {summary['core_answer']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    start = time.time()
    models = {model_key: run_model(model_key) for model_key in MODEL_KEYS}
    summary = {
        "stage": "stage503_concept_hierarchy_causal_path_triple_model",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
        "core_answer": (
            "如果概念层次关系在层级消融下表现出明显目标特异下降，"
            "那么概念家族并不只是几何现象，而是已经进入了可打中的因果路径。"
        ),
        "elapsed_seconds": round(time.time() - start, 1),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
