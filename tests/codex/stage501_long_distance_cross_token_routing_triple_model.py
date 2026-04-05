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
    / "stage501_long_distance_cross_token_routing_triple_model_20260404"
)
MODEL_KEYS = ["qwen3", "deepseek7b", "glm4"]
MAX_PROMPTS_PER_TASK = 1

LONG_DISTANCE_TASKS: List[Dict[str, object]] = [
    {
        "task_id": "long_pronoun",
        "name": "长距离责任归属",
        "prompts": [
            {
                "text": "张三昨天在部门例会上批评了李四，因为李四连续三周没有提交周报。晚上经理单独找人谈话时，最终承认错误的是",
                "target": "李四",
                "candidates": ["李四", "张三", "经理", "周报"],
            },
            {
                "text": "王老师提醒小刘尽快补交作业，因为小刘上周生病耽误了课程。后来班主任再次问起这件事时，主动说明情况的是",
                "target": "小刘",
                "candidates": ["小刘", "王老师", "班主任", "作业"],
            },
        ],
    },
    {
        "task_id": "long_reference_chain",
        "name": "多跳指代链",
        "prompts": [
            {
                "text": "小猫追着老鼠跑进仓库，老鼠钻进纸箱，纸箱又被工人搬到门口。整个过程里最先开始追逐的是",
                "target": "小猫",
                "candidates": ["小猫", "老鼠", "纸箱", "工人"],
            },
            {
                "text": "老师把作业交给班长，班长又转给学习委员，学习委员最后发给同学。最早拿着作业的人是",
                "target": "老师",
                "candidates": ["老师", "班长", "学习委员", "同学"],
            },
        ],
    },
    {
        "task_id": "long_quantity",
        "name": "长距离数量链",
        "prompts": [
            {
                "text": "仓库里先搬进三箱苹果，又搬进两箱橙子，随后卖出一箱苹果。现在仓库里一共还剩",
                "target": "四箱",
                "candidates": ["四箱", "五箱", "三箱", "两箱"],
            },
            {
                "text": "书架上原来有十本书，后来借走三本，又新买了两本。现在书架上一共有",
                "target": "九本",
                "candidates": ["九本", "八本", "七本", "十本"],
            },
        ],
    },
    {
        "task_id": "long_logic_chain",
        "name": "长距离逻辑链",
        "prompts": [
            {
                "text": "如果设备过热就会报警；只要报警，工程师就会停机检查。现在设备已经过热，所以工程师接下来会",
                "target": "停机检查",
                "candidates": ["停机检查", "继续运行", "离开现场", "更换设备"],
            },
            {
                "text": "如果合同到期就需要续签；只要没有续签，系统就会自动停用账号。现在合同已经到期，而且还没续签，所以账号会被",
                "target": "停用",
                "candidates": ["停用", "保留", "转移", "删除"],
            },
        ],
    },
    {
        "task_id": "long_attribute",
        "name": "长距离属性保持",
        "prompts": [
            {
                "text": "这件外套昨天被染成深蓝色，今天放在灯光下看起来也没有变浅。所以这件外套的颜色仍然是",
                "target": "深蓝色",
                "candidates": ["深蓝色", "浅蓝色", "红色", "白色"],
            },
            {
                "text": "这杯果汁刚做出来很甜，中途没有加水也没有放酸味剂，所以它现在喝起来应该还是",
                "target": "很甜",
                "candidates": ["很甜", "很酸", "很苦", "很淡"],
            },
        ],
    },
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def classify_mechanism(avg_attn: float, avg_mlp: float) -> str:
    if avg_attn > avg_mlp * 1.25:
        return "route_heads"
    if avg_mlp > avg_attn * 1.25:
        return "write_neurons"
    return "mixed"


def run_task(model, tokenizer, task: dict) -> dict:
    sample_layers = evenly_spaced_layers(model, count=5)
    prompt_rows = []

    for prompt_row in list(task["prompts"])[:MAX_PROMPTS_PER_TASK]:
        text = str(prompt_row["text"])
        target = str(prompt_row["target"])
        candidates = list(prompt_row["candidates"])
        baseline_scores = candidate_score_map(model, tokenizer, text, candidates)
        baseline_target = float(baseline_scores[target])
        baseline_pred = max(baseline_scores, key=baseline_scores.get)

        layer_results = []
        for layer_idx in sample_layers:
            for component in ("attn", "mlp"):
                layer, original = ablate_layer_component(model, layer_idx, component)
                try:
                    ablated_scores = candidate_score_map(model, tokenizer, text, candidates)
                finally:
                    restore_layer_component(layer, component, original)
                ablated_target = float(ablated_scores[target])
                impact = baseline_target - ablated_target
                layer_results.append(
                    {
                        "layer": layer_idx,
                        "component": component,
                        "baseline_target_score": round(baseline_target, 6),
                        "ablated_target_score": round(ablated_target, 6),
                        "target_drop": round(impact, 6),
                        "baseline_pred": baseline_pred,
                        "ablated_pred": max(ablated_scores, key=ablated_scores.get),
                    }
                )

        prompt_rows.append(
            {
                "text": text,
                "target": target,
                "candidates": candidates,
                "baseline_scores": {k: round(v, 6) for k, v in baseline_scores.items()},
                "baseline_pred": baseline_pred,
                "layer_results": layer_results,
            }
        )

    layer_attn_impacts = {}
    layer_mlp_impacts = {}
    for layer_idx in sample_layers:
        attn_vals = [
            row["target_drop"]
            for prompt in prompt_rows
            for row in prompt["layer_results"]
            if row["layer"] == layer_idx and row["component"] == "attn"
        ]
        mlp_vals = [
            row["target_drop"]
            for prompt in prompt_rows
            for row in prompt["layer_results"]
            if row["layer"] == layer_idx and row["component"] == "mlp"
        ]
        layer_attn_impacts[str(layer_idx)] = round(statistics.mean(attn_vals), 6) if attn_vals else 0.0
        layer_mlp_impacts[str(layer_idx)] = round(statistics.mean(mlp_vals), 6) if mlp_vals else 0.0

    avg_attn = statistics.mean(layer_attn_impacts.values()) if layer_attn_impacts else 0.0
    avg_mlp = statistics.mean(layer_mlp_impacts.values()) if layer_mlp_impacts else 0.0
    dominant = classify_mechanism(avg_attn, avg_mlp)
    peak_attn_layer = int(max(layer_attn_impacts, key=layer_attn_impacts.get))
    peak_mlp_layer = int(max(layer_mlp_impacts, key=layer_mlp_impacts.get))

    return {
        "task_id": task["task_id"],
        "task_name": task["name"],
        "prompt_count": len(prompt_rows),
        "layer_attn_impacts": layer_attn_impacts,
        "layer_mlp_impacts": layer_mlp_impacts,
        "avg_attn_impact": round(avg_attn, 6),
        "avg_mlp_impact": round(avg_mlp, 6),
        "attn_to_mlp_ratio": round(avg_attn / max(avg_mlp, 1e-9), 4),
        "dominant_mechanism": dominant,
        "peak_attn_layer": peak_attn_layer,
        "peak_mlp_layer": peak_mlp_layer,
        "prompt_results": prompt_rows,
    }


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
            "results": results,
            "aggregate": {
                "route_heads_dominant_count": route_count,
                "write_neurons_dominant_count": write_count,
                "mixed_count": mixed_count,
                "total_tasks": len(LONG_DISTANCE_TASKS),
                "route_heads_ratio": round(route_count / len(LONG_DISTANCE_TASKS), 4),
                "avg_attn_to_mlp_ratio": round(
                    statistics.mean(row["attn_to_mlp_ratio"] for row in results.values()),
                    4,
                ),
            },
        }
    finally:
        free_model(model)


def build_report(summary: dict) -> str:
    lines = ["# stage501 长距离跨词元路由三模型测试", ""]
    for model_key, model_row in summary["models"].items():
        agg = model_row["aggregate"]
        lines.append(f"## {model_key}")
        lines.append("")
        lines.append(f"- route_heads 主导数: `{agg['route_heads_dominant_count']}/{agg['total_tasks']}`")
        lines.append(f"- write_neurons 主导数: `{agg['write_neurons_dominant_count']}/{agg['total_tasks']}`")
        lines.append(f"- mixed 主导数: `{agg['mixed_count']}/{agg['total_tasks']}`")
        lines.append(f"- 平均 attn/mlp 比: `{agg['avg_attn_to_mlp_ratio']}`")
        lines.append("")
        for task_id, row in model_row["results"].items():
            lines.append(f"### {task_id}")
            lines.append("")
            lines.append(f"- 主导机制: `{row['dominant_mechanism']}`")
            lines.append(f"- 平均 attn 影响: `{row['avg_attn_impact']}`")
            lines.append(f"- 平均 mlp 影响: `{row['avg_mlp_impact']}`")
            lines.append(f"- 峰值 attn 层: `L{row['peak_attn_layer']}`")
            lines.append(f"- 峰值 mlp 层: `L{row['peak_mlp_layer']}`")
            lines.append("")
    lines.append("## 综合结论")
    lines.append("")
    lines.append(f"- {summary['core_answer']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    start = time.time()
    models = {}
    for model_key in MODEL_KEYS:
        models[model_key] = run_model(model_key)

    core_answer = (
        "在更长距离的跨词元任务中，如果 route_heads 仍然没有稳定单独抬头，"
        "就说明语言搬运机制更可能是头与神经元共同组成的混合回路，而不是简单的“注意力单独接管”。"
    )
    summary = {
        "stage": "stage501_long_distance_cross_token_routing_triple_model",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
        "core_answer": core_answer,
        "elapsed_seconds": round(time.time() - start, 1),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
