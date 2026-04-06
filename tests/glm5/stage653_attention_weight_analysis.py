#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage653: 全层注意力权重分析

目标：直接读取attention权重，分析A/B版本间各层attention pattern的差异，
找到"能力关键层"的attention特征。

方法：
1. 对每个case，分别在prompt_a和prompt_b下运行model(output_attentions=True)
2. 计算各层attention pattern的cos差异：cos(attn_a, attn_b)
3. 找到attention差异最大的层（这些层在"关注不同位置"）
4. 对比四个模型的attention差异层分布

预注册判伪条件：
INV-230: "attention差异最大的层与因果消融最伤层一致"
如果attention差异峰值层与最伤层不一致（偏差>3层），则INV-230被推翻。
"""

from __future__ import annotations

import sys
import json
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    ablate_layer_component,
    discover_layers,
    free_model,
    load_model_bundle,
    restore_layer_component,
    score_candidate_avg_logprob,
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")


@dataclass(frozen=True)
class TestCase:
    name: str
    prompt_a: str
    positive_a: str
    negative_a: str
    prompt_b: str
    positive_b: str
    negative_b: str


CASES: List[TestCase] = [
    TestCase("syllogism",
             "All mammals are animals. All cats are mammals. Cats are",
             " animals", " reptiles",
             "All birds are animals. All sparrows are birds. Sparrows are",
             " animals", " insects"),
    TestCase("relation_capital",
             "Paris is the capital of France. The capital of France is",
             " Paris", " Berlin",
             "Berlin is the capital of Germany. The capital of Germany is",
             " Berlin", " Paris"),
    TestCase("arithmetic",
             "If x = 7 and y = 3, then x + y =",
             " 10", " 11",
             "If x = 15 and y = 8, then x + y =",
             " 23", " 22"),
    TestCase("syntax_sv",
             "The key to the cabinet", " is", " are",
             "The keys to the cabinet", " are", " is"),
]


def get_attention_weights(model, tokenizer, text: str) -> Optional[List[torch.Tensor]]:
    """获取所有层的attention权重"""
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    try:
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=attention_mask,
                          output_attentions=True)
        if output.attentions is not None:
            # 将attentions移到CPU
            return [a.detach().cpu() for a in output.attentions]
        return None
    except Exception as e:
        print(f"    attention extraction error: {e}")
        return None


def flatten_attn(attn_weights: torch.Tensor) -> torch.Tensor:
    """将attention权重展平为向量，用于计算cos差异"""
    # attn_weights: [batch, num_heads, seq_len, seq_len]
    return attn_weights[0].flatten()  # [num_heads * seq_len * seq_len]


def attn_cos_diff(attn_a: List[torch.Tensor], attn_b: List[torch.Tensor]) -> List[float]:
    """计算每层attention pattern的cos差异"""
    diffs = []
    for la, lb in zip(attn_a, attn_b):
        fa = flatten_attn(la)
        fb = flatten_attn(lb)
        # 如果序列长度不同，只比较公共最小长度部分
        if fa.shape != fb.shape:
            min_len = min(fa.shape[0], fb.shape[0])
            fa = fa[:min_len]
            fb = fb[:min_len]
        cos = torch.dot(fa, fb) / (fa.norm() * fb.norm() + 1e-10)
        diffs.append(1.0 - cos.item())  # 转为"差异"而非"相似度"
    return diffs


def find_most_harmful_layer(model, tokenizer, case: TestCase) -> Dict:
    """找到MLP消融伤害最大的层"""
    baseline_a = score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.positive_a) - \
                 score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.negative_a)
    baseline_b = score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.positive_b) - \
                 score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.negative_b)
    baseline = (baseline_a + baseline_b) / 2.0

    num_layers = len(discover_layers(model))
    best_drop = 0
    best_layer = 0

    # 只测试偶数层以节省时间
    test_layers = list(range(0, num_layers, max(1, num_layers // 15)))

    for layer_idx in test_layers:
        layer_obj, orig = ablate_layer_component(model, layer_idx, "mlp")
        try:
            abl_a = score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.positive_a) - \
                    score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.negative_a)
            abl_b = score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.positive_b) - \
                    score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.negative_b)
            abl_margin = (abl_a + abl_b) / 2.0
        finally:
            restore_layer_component(layer_obj, "mlp", orig)

        drop = baseline - abl_margin
        if drop > best_drop:
            best_drop = drop
            best_layer = layer_idx

    return {"layer": best_layer, "damage": best_drop}


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python stage653_attention_weight_analysis.py [qwen3|deepseek7b|glm4|gemma4]")
        sys.exit(1)
    model_key = sys.argv[1]
    run_dir = OUTPUT_DIR / f"stage653_attention_{model_key}_{TIMESTAMP}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = None
    try:
        print(f"[Stage653] Loading {model_key}...")
        model, tokenizer = load_model_bundle(model_key, prefer_cuda=True)
        num_layers = len(discover_layers(model))
        print(f"[Stage653] layers={num_layers}")

        records = []
        for case in CASES:
            print(f"\n[Stage653] {case.name}")

            # 获取A/B版本的attention
            attn_a = get_attention_weights(model, tokenizer, case.prompt_a)
            attn_b = get_attention_weights(model, tokenizer, case.prompt_b)

            if attn_a is None or attn_b is None:
                print(f"  SKIP: attention not available")
                continue

            actual_layers = min(len(attn_a), len(attn_b), num_layers)
            print(f"  attention layers: {actual_layers}")

            # 计算每层差异
            diffs = attn_cos_diff(attn_a[:actual_layers], attn_b[:actual_layers])
            max_diff = max(diffs)
            max_diff_layer = diffs.index(max_diff)
            mean_diff = statistics.mean(diffs)

            print(f"  max_diff={max_diff:.4f} at L{max_diff_layer}, mean_diff={mean_diff:.4f}")

            # 找最伤层（粗略搜索）
            harm_info = find_most_harmful_layer(model, tokenizer, case)
            layer_offset = abs(harm_info["layer"] - max_diff_layer)

            print(f"  most_harmful_layer: L{harm_info['layer']}, offset={layer_offset}")

            records.append({
                "case": case.name,
                "max_attn_diff": round(max_diff, 4),
                "max_diff_layer": max_diff_layer,
                "mean_attn_diff": round(mean_diff, 4),
                "most_harmful_layer": harm_info["layer"],
                "layer_offset": layer_offset,
                "per_layer_diffs": [round(d, 4) for d in diffs],
            })

        # 汇总
        if records:
            print(f"\n{'='*50}")
            print("Summary:")
            avg_offset = statistics.mean([r["layer_offset"] for r in records])
            avg_max_diff = statistics.mean([r["max_attn_diff"] for r in records])
            consistent = sum(1 for r in records if r["layer_offset"] <= 3)

            print(f"  avg_layer_offset: {avg_offset:.1f}")
            print(f"  avg_max_attn_diff: {avg_max_diff:.4f}")
            print(f"  consistent (offset<=3): {consistent}/{len(records)}")

            # INV-230 check
            inv230 = "SURVIVED" if consistent >= len(records) * 0.75 else "FALSIFIED"
            print(f"  INV-230 (attn_peak=harm_layer): {inv230}")

            # Top-3 attention差异最大的层（跨case汇总）
            all_max_layers = [r["max_diff_layer"] for r in records]
            print(f"  peak layers: {all_max_layers}")

            payload = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": model_key,
                "num_layers": num_layers,
                "avg_layer_offset": round(avg_offset, 1),
                "avg_max_attn_diff": round(avg_max_diff, 4),
                "consistent_count": consistent,
                "total_count": len(records),
                "inv230_result": inv230,
                "records": records,
            }
            (run_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved: {run_dir / 'summary.json'}")
    finally:
        if model is not None:
            free_model(model)


if __name__ == "__main__":
    main()
