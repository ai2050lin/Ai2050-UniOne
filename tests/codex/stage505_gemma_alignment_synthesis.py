#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE501_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage501_long_distance_cross_token_routing_triple_model_20260404"
    / "summary.json"
)
STAGE504_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage504_quad_model_external_control_suite_20260404"
    / "summary.json"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage505_gemma_alignment_synthesis_20260404"
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def vector_from_category_summary(cat_summary: dict) -> list[float]:
    keys = ["polysemy", "pattern_completion", "long_route", "concept_hierarchy", "attribute_binding"]
    return [float(cat_summary[key]["accuracy"]) for key in keys]


def euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def build_report(summary: dict) -> str:
    lines = ["# stage505 Gemma4 行为谱系定位综合摘要", ""]
    lines.append(f"- Gemma4 行为最近邻：`{summary['gemma_behavior_nearest_model']}`")
    lines.append(f"- Gemma4 到最近邻距离：`{summary['gemma_behavior_nearest_distance']}`")
    lines.append(f"- Gemma4 是否已有层内机制协议：`{summary['gemma_has_internal_mechanistic_access']}`")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append("- Gemma4 当前已经进入统一外部行为协议，但尚未进入层内挂钩协议。")
    lines.append("- 在行为向量上，Gemma4 当前更接近 DeepSeek7B，而不是 Qwen3 或 GLM4。")
    lines.append("- 这只是行为近邻，不等于机制同构。")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    stage501 = load_json(STAGE501_PATH)
    stage504 = load_json(STAGE504_PATH)

    category_vectors = {
        model_key: vector_from_category_summary(model_row["summary"]["category_summary"])
        for model_key, model_row in stage504["models"].items()
    }
    gemma_vec = category_vectors["gemma4"]
    distances = {}
    for model_key, vec in category_vectors.items():
        if model_key == "gemma4":
            continue
        distances[model_key] = round(euclidean(gemma_vec, vec), 4)

    nearest_model = min(distances, key=distances.get)
    summary = {
        "stage": "stage505_gemma_alignment_synthesis",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gemma_behavior_vector": gemma_vec,
        "distance_to_existing_models": distances,
        "gemma_behavior_nearest_model": nearest_model,
        "gemma_behavior_nearest_distance": distances[nearest_model],
        "internal_routing_reference": {
            model_key: stage501["models"][model_key]["aggregate"]
            for model_key in ("qwen3", "deepseek7b", "glm4")
        },
        "gemma_has_internal_mechanistic_access": False,
        "core_answer": (
            "Gemma4 当前已进入统一外部行为图谱，行为上更接近 DeepSeek7B，"
            "但由于其当前是 GGUF/Ollama 形态，尚未进入层内机制协议，"
            "因此还不能把这种近邻关系直接上升为机制同构。"
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
