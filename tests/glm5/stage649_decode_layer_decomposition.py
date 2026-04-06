#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage649: 解码层逆向工程——最后N层对方向恢复的贡献分解

目标：量化Transformer最后N层中每一层对方向恢复的贡献
方法：
1. 在L0提取语法/推理的delta方向
2. 在"最伤层"消融后，逐层恢复（从倒数第一层开始逐步恢复）
3. 测试恢复到第k层时margin恢复了多少
4. 找到"关键解码层"——哪一层贡献最大

核心问题：P3发现方向在5层后cos≈0，但Gemma4恢复率275%。
如果因果效力在解码层，那么最后几层应该对恢复有巨大贡献。

预注册判伪条件：
INV-217_extra: "最后5层贡献>50%的方向恢复力"
如果四模型中最后5层的累积恢复力不超过总恢复力的50%，则被推翻。
"""

from __future__ import annotations

import sys
import json
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    ablate_layer_component,
    discover_layers,
    evenly_spaced_layers,
    free_model,
    load_model_bundle,
    restore_layer_component,
    score_candidate_avg_logprob,
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@dataclass(frozen=True)
class TestCase:
    name: str
    capability: str
    prompt_a: str
    positive_a: str
    negative_a: str
    prompt_b: str
    positive_b: str
    negative_b: str


CASES: List[TestCase] = [
    TestCase("syntax_sv", "syntax",
             "The key to the cabinet", " is", " are",
             "The keys to the cabinet", " are", " is"),
    TestCase("syllogism", "reasoning",
             "All mammals are animals. All cats are mammals. Cats are",
             " animals", " reptiles",
             "All birds are animals. All sparrows are birds. Sparrows are",
             " animals", " insects"),
    TestCase("relation_capital", "relation",
             "Paris is the capital of France. The capital of France is",
             " Paris", " Berlin",
             "Berlin is the capital of Germany. The capital of Germany is",
             " Berlin", " Paris"),
]


def case_margin(model, tokenizer, case: TestCase) -> float:
    ma = score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.positive_a) - \
         score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.negative_a)
    mb = score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.positive_b) - \
         score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.negative_b)
    return float((ma + mb) / 2.0)


def extract_layer_last_token(model, tokenizer, prompt: str, layer_idx: int) -> torch.Tensor:
    layers = discover_layers(model)
    device = get_model_device(model)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    captured = {}

    def hook_fn(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["value"] = hidden[0, -1, :].detach().float().cpu()

    handle = layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    return captured["value"]


def find_most_harmful(model, tokenizer, case: TestCase, layer_indices: List[int]) -> Dict:
    baseline = case_margin(model, tokenizer, case)
    best_drop = 0
    best_layer = 0
    best_comp = "attn"
    for li in layer_indices:
        for comp in ("attn", "mlp"):
            layer, orig = ablate_layer_component(model, li, comp)
            try:
                abl = case_margin(model, tokenizer, case)
            finally:
                restore_layer_component(layer, comp, orig)
            drop = baseline - abl
            if drop > best_drop:
                best_drop = drop
                best_layer = li
                best_comp = comp
    return {"layer": best_layer, "component": best_comp, "damage": best_drop, "baseline": baseline}


def progressive_restore_test(model, tokenizer, case: TestCase, harm_layer: int,
                              harm_comp: str, num_layers: int, direction: torch.Tensor,
                              test_last_n: int = 10) -> Dict:
    """
    在最伤层消融后，从倒数第1层开始逐步恢复，
    测试恢复到第k层时的margin恢复量。
    """
    device = get_model_device(model)
    direction = direction.to(device)

    # 先测全消融baseline
    harm_layer_obj, harm_orig = ablate_layer_component(model, harm_layer, harm_comp)

    def run_with_restore_at_layers(restore_layers: List[int]) -> float:
        """在指定的层注入direction，其余层保持消融"""
        ablated_layers = [(harm_layer, harm_comp, harm_orig)]
        hooks = []

        for rl in restore_layers:
            if rl == harm_layer:
                continue
            # 只注入direction，不消融
            layer_obj = discover_layers(model)[rl]

            def make_inject_hook(lidx):
                def inject_hook(module, inputs, output):
                    if isinstance(output, tuple):
                        hidden = output[0].clone()
                        hidden[0, -1, :] = hidden[0, -1, :] + direction
                        return (hidden,) + output[1:]
                    hidden = output.clone()
                    hidden[0, -1, :] = hidden[0, -1, :] + direction
                    return hidden
                return inject_hook

            h = layer_obj.register_forward_hook(make_inject_hook(rl))
            hooks.append(h)

        try:
            ma = score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.positive_a) - \
                 score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.negative_a)
            mb = score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.positive_b) - \
                 score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.negative_b)
            return float((ma + mb) / 2.0)
        finally:
            for h in hooks:
                h.remove()

    # 全消融baseline
    fully_ablated_margin = case_margin(model, tokenizer, case)

    # 逐步从最后N层恢复
    results = {}
    for n in range(1, test_last_n + 1):
        # 恢复倒数n层（即最后n层都注入direction）
        restore_start = num_layers - n
        restore_layers = list(range(max(restore_start, 0), num_layers))
        # 排除harm_layer
        restore_layers = [l for l in restore_layers if l != harm_layer]
        if not restore_layers:
            results[n] = fully_ablated_margin
            continue
        margin = run_with_restore_at_layers(restore_layers)
        results[n] = margin

    restore_layer_component(harm_layer_obj, harm_comp, harm_orig)

    # 计算恢复百分比
    damage = results[0] - fully_ablated_margin if 0 in results else None
    # baseline在find_most_harmful中已知

    return {
        "fully_ablated": fully_ablated_margin,
        "per_n_layers": {str(n): round(v, 4) for n, v in results.items()},
        "last_5_recovery": round(results.get(5, 0) - fully_ablated_margin, 4),
        "last_10_recovery": round(results.get(10, 0) - fully_ablated_margin, 4),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python stage649_decode_layer_decomposition.py [qwen3|deepseek7b|glm4|gemma4]")
        sys.exit(1)
    model_key = sys.argv[1]
    run_dir = OUTPUT_DIR / f"stage649_decode_{model_key}_{TIMESTAMP}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = None
    try:
        print(f"[Stage649] Loading {model_key}...")
        model, tokenizer = load_model_bundle(model_key, prefer_cuda=True)
        num_layers = len(discover_layers(model))
        layer_indices = evenly_spaced_layers(model, count=7)
        print(f"[Stage649] layers={num_layers}, sample={layer_indices}")

        records = []
        for i, case in enumerate(CASES):
            print(f"\n[Stage649] ({i+1}/{len(CASES)}) {case.name}")
            harm = find_most_harmful(model, tokenizer, case, layer_indices)
            print(f"  harm: L{harm['layer']}/{harm['component']}, damage={harm['damage']:.4f}")

            if harm["damage"] < 0.01:
                print("  damage too small, skip")
                continue

            # 提取方向
            delta = extract_layer_last_token(model, tokenizer, case.prompt_a, harm["layer"]) - \
                    extract_layer_last_token(model, tokenizer, case.prompt_b, harm["layer"])

            # 逐步恢复测试
            test_n = min(10, num_layers - 1)
            restore_result = progressive_restore_test(
                model, tokenizer, case, harm["layer"], harm["component"],
                num_layers, delta, test_last_n=test_n
            )

            # 分析最后5层贡献比例
            total_recovery = restore_result["last_10_recovery"]
            last5_share = restore_result["last_5_recovery"] / max(abs(total_recovery), 1e-8) * 100

            print(f"  fully_ablated: {restore_result['fully_ablated']:.4f}")
            print(f"  last5_recovery: {restore_result['last_5_recovery']:.4f}")
            print(f"  last10_recovery: {restore_result['last_10_recovery']:.4f}")
            print(f"  last5_share: {last5_share:.1f}%")

            records.append({
                "case_name": case.name,
                "capability": case.capability,
                "harm_layer": harm["layer"],
                "harm_component": harm["component"],
                "damage": harm["damage"],
                "baseline_margin": harm["baseline"],
                "restore": restore_result,
                "last5_share_pct": round(last5_share, 1),
            })

        # Summary
        avg_last5_share = statistics.mean([r["last5_share_pct"] for r in records]) if records else 0
        print(f"\n{'='*60}")
        print(f"[Stage649] Summary: {model_key}")
        print(f"  avg last5_share: {avg_last5_share:.1f}%")
        inv_result = "SURVIVED" if avg_last5_share > 50 else "FALSIFIED"
        print(f"  INV-217_extra (last5>50%): {inv_result}")

        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_key,
            "num_layers": num_layers,
            "avg_last5_share_pct": round(avg_last5_share, 1),
            "inv_result": inv_result,
            "records": records,
        }
        (run_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {run_dir / 'summary.json'}")
    finally:
        if model is not None:
            free_model(model)


if __name__ == "__main__":
    main()
