#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import torch

from multimodel_language_shared import MODEL_SPECS, free_model, load_model_bundle
from stage433_polysemous_noun_family_generalization import POLYSEMOUS_CASES, analyze_single_noun as analyze_balance_single_noun
from stage447_polysemy_family_switch_protocol import analyze_single_noun as analyze_switch_single_noun


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage511_glm4_polysemy_switch_protocol_20260404"
)
MODEL_KEY = "glm4"


class _OutFeatureShim:
    def __init__(self, out_features: int):
        self.out_features = int(out_features)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def patch_glm4_mlp_compat(model) -> None:
    for layer in model.model.layers:
        mlp = layer.mlp
        if not hasattr(mlp, "gate_proj") and hasattr(mlp, "down_proj"):
            mlp.gate_proj = _OutFeatureShim(mlp.down_proj.in_features)


def build_stage433_like_summary(model, tokenizer) -> Dict[str, object]:
    noun_results = [analyze_balance_single_noun(model, tokenizer, noun_spec) for noun_spec in POLYSEMOUS_CASES]
    return {
        "model_results": [
            {
                "model_key": MODEL_KEY,
                "noun_results": noun_results,
            }
        ]
    }


def analyze_model(*, prefer_cuda: bool) -> Dict[str, object]:
    model, tokenizer = load_model_bundle(MODEL_KEY, prefer_cuda=prefer_cuda)
    try:
        patch_glm4_mlp_compat(model)
        stage433_summary = build_stage433_like_summary(model, tokenizer)
        noun_results = [
            analyze_switch_single_noun(model, tokenizer, stage433_summary, MODEL_KEY, noun_spec)
            for noun_spec in POLYSEMOUS_CASES
        ]
        aggregate = {
            "noun_count": len(noun_results),
            "polysemy_split_support_count": sum(int(row["supports_polysemy_split"]) for row in noun_results),
            "switch_causality_support_count": sum(int(row["supports_switch_causality"]) for row in noun_results),
            "mean_polysemy_jaccard": sum(float(row["sense_active_jaccard"]) for row in noun_results) / max(1, len(noun_results)),
            "mean_ordinary_jaccard": sum(float(row["ordinary_control_mean_active_jaccard"]) for row in noun_results)
            / max(1, len(noun_results)),
            "mean_switch_prob_drop": sum(
                float(row["switch_axis_ablation"]["switch_axis_delta"]["mean_correct_prob_drop"]) for row in noun_results
            )
            / max(1, len(noun_results)),
            "mean_control_prob_drop": sum(
                float(row["switch_axis_ablation"]["control_axis_delta"]["mean_correct_prob_drop"]) for row in noun_results
            )
            / max(1, len(noun_results)),
            "best_switch_layers": {row["noun_id"]: int(row["best_switch_layer"]) for row in noun_results},
        }
        return {
            "model_key": MODEL_KEY,
            "model_name": MODEL_SPECS[MODEL_KEY]["label"],
            "used_cuda": bool(prefer_cuda and torch.cuda.is_available()),
            "noun_results": noun_results,
            "aggregate": aggregate,
        }
    finally:
        free_model(model)


def build_cross_summary(model_result: Dict[str, object]) -> Dict[str, object]:
    aggregate = model_result["aggregate"]
    noun_count = int(aggregate["noun_count"])
    polysemy_split_rate = float(aggregate["polysemy_split_support_count"]) / max(1, noun_count)
    switch_causality_rate = float(aggregate["switch_causality_support_count"]) / max(1, noun_count)
    mean_polysemy = float(aggregate["mean_polysemy_jaccard"])
    mean_ordinary = float(aggregate["mean_ordinary_jaccard"])
    gap = mean_ordinary - mean_polysemy
    if polysemy_split_rate >= 0.5 and switch_causality_rate >= 0.5:
        core_answer = "GLM4 在当前多义词协议里已经表现出较稳定的低重合切换结构。"
    elif gap < 0 and switch_causality_rate < 0.5:
        core_answer = "GLM4 在当前多义词协议里更像带明显默认义偏置的弱切换系统。"
    else:
        core_answer = "GLM4 在当前多义词协议里出现了部分切换迹象，但还没有形成稳定的低重合切换结构。"
    return {
        "total_noun_cases": noun_count,
        "polysemy_split_support_rate": polysemy_split_rate,
        "switch_causality_support_rate": switch_causality_rate,
        "mean_polysemy_active_jaccard": mean_polysemy,
        "mean_ordinary_active_jaccard": mean_ordinary,
        "mean_ordinary_vs_polysemy_gap": gap,
        "mean_switch_axis_prob_drop": float(aggregate["mean_switch_prob_drop"]),
        "mean_control_axis_prob_drop": float(aggregate["mean_control_prob_drop"]),
        "core_answer": core_answer,
    }


def build_report(summary: Dict[str, object]) -> str:
    model_row = summary["model_results"][0]
    agg = model_row["aggregate"]
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "## 核心结论",
        summary["cross_model_summary"]["core_answer"],
        "",
        f"- 模型：`{model_row['model_name']}`",
        f"- 多义词数量：`{agg['noun_count']}`",
        f"- 低重合切换支持数：`{agg['polysemy_split_support_count']}`",
        f"- 切换轴因果支持数：`{agg['switch_causality_support_count']}`",
        f"- 平均多义词重合：`{agg['mean_polysemy_jaccard']:.4f}`",
        f"- 平均普通上下文重合：`{agg['mean_ordinary_jaccard']:.4f}`",
        f"- 平均切换轴概率下降：`{agg['mean_switch_prob_drop']:.4f}`",
        f"- 平均控制轴概率下降：`{agg['mean_control_prob_drop']:.4f}`",
        "",
    ]
    for noun_row in model_row["noun_results"]:
        switch_delta = noun_row["switch_axis_ablation"]["switch_axis_delta"]
        control_delta = noun_row["switch_axis_ablation"]["control_axis_delta"]
        lines.extend(
            [
                f"## {noun_row['noun_id']}",
                f"- 最佳切换层：`L{noun_row['best_switch_layer']}`",
                f"- 多义词重合：`{noun_row['sense_active_jaccard']:.4f}`",
                f"- 普通上下文重合：`{noun_row['ordinary_control_mean_active_jaccard']:.4f}`",
                f"- 差值：`{noun_row['ordinary_vs_polysemy_gap']:.4f}`",
                f"- 切换轴概率下降：`{switch_delta['mean_correct_prob_drop']:.4f}`",
                f"- 控制轴概率下降：`{control_delta['mean_correct_prob_drop']:.4f}`",
                f"- 是否支持低重合切换：`{noun_row['supports_polysemy_split']}`",
                f"- 是否支持切换轴因果：`{noun_row['supports_switch_causality']}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def write_outputs(summary: Dict[str, object]) -> None:
    ensure_dir(OUTPUT_DIR)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "REPORT.md").write_text(build_report(summary), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GLM4 多义词切换内部机制协议")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()
    prefer_cuda = (not args.cpu) and torch.cuda.is_available()
    model_result = analyze_model(prefer_cuda=prefer_cuda)
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage511_glm4_polysemy_switch_protocol",
        "title": "GLM4 多义词切换内部机制协议",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - start_time, 3),
        "used_cuda": bool(model_result["used_cuda"]),
        "model_results": [model_result],
        "cross_model_summary": build_cross_summary(model_result),
    }
    write_outputs(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
