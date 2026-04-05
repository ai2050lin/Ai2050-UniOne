#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
from pathlib import Path

from multimodel_language_shared import MODEL_SPECS, discover_layers, free_model, load_model_bundle
from stage501_long_distance_cross_token_routing_triple_model import LONG_DISTANCE_TASKS, run_task


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage508_long_distance_cross_token_routing_quad_model_20260404"
)
MODEL_KEYS = ["qwen3", "deepseek7b", "glm4", "gemma4"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_model(model_key: str) -> dict:
    model, tokenizer = load_model_bundle(model_key, prefer_cuda=False)
    try:
        results = {}
        for task in LONG_DISTANCE_TASKS:
            results[task["task_id"]] = run_task(model, tokenizer, task)
        route_count = sum(1 for row in results.values() if row["dominant_mechanism"] == "route_heads")
        write_count = sum(1 for row in results.values() if row["dominant_mechanism"] == "write_neurons")
        mixed_count = sum(1 for row in results.values() if row["dominant_mechanism"] == "mixed")
        return {
            "model_label": MODEL_SPECS[model_key]["label"],
            "layer_count": len(discover_layers(model)),
            "used_cuda": False,
            "results": results,
            "aggregate": {
                "route_heads_dominant_count": route_count,
                "write_neurons_dominant_count": write_count,
                "mixed_count": mixed_count,
                "total_tasks": len(LONG_DISTANCE_TASKS),
                "route_heads_ratio": round(route_count / len(LONG_DISTANCE_TASKS), 4),
                "avg_attn_to_mlp_ratio": round(
                    sum(row["attn_to_mlp_ratio"] for row in results.values()) / len(results),
                    4,
                ),
            },
        }
    finally:
        free_model(model)


def build_report(summary: dict) -> str:
    lines = ["# stage508 长距离跨词元路由四模型内部机制协议", ""]
    lines.append("- 本轮在 stage501 的基础上，将 Gemma4 纳入同一层内协议。")
    lines.append("")
    for model_key, model_row in summary["models"].items():
        agg = model_row["aggregate"]
        lines.append(f"## {model_key}")
        lines.append("")
        lines.append(f"- 模型：`{model_row['model_label']}`")
        lines.append(f"- 层数：`{model_row['layer_count']}`")
        lines.append(f"- route_heads 主导：`{agg['route_heads_dominant_count']}/{agg['total_tasks']}`")
        lines.append(f"- write_neurons 主导：`{agg['write_neurons_dominant_count']}/{agg['total_tasks']}`")
        lines.append(f"- mixed：`{agg['mixed_count']}/{agg['total_tasks']}`")
        lines.append(f"- 平均 attn/mlp 比：`{agg['avg_attn_to_mlp_ratio']}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    summary = {
        "stage": "stage508_long_distance_cross_token_routing_quad_model",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {},
    }
    for model_key in MODEL_KEYS:
        summary["models"][model_key] = run_model(model_key)
    summary["elapsed_seconds"] = round(time.time() - started, 3)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
