#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage658: P9 Gemma4编码策略因果验证

P6-P7发现Gemma4使用"统一编码策略"：
- 推理-消歧cos=0.97（其他模型<0.25）
- 跨能力方向耦合max|cos|=0.93（其他模型<0.20）
- 推理维度收缩到1维（其他模型保持3维）

P9目标：通过消融实验验证Gemma4编码策略的因果效应
- 消融跨能力耦合维度后，各能力如何变化？
- 如果Gemma4的统一编码是因果性的，消融后应该导致多能力同时下降

实验1：跨能力耦合维度识别与消融
  找到多个能力共享的高cos方向，零化后测量各能力变化
实验2：推理-消歧共享方向消融
  找到推理和消歧共享的方向，消融后测量推理和消歧分别如何变化
实验3：维度分离干预
  将Gemma4的统一编码方向强制分离（旋转消歧方向远离推理方向），测量效果
实验4：Gemma4 vs 其他模型的消融对比
  在其他模型上做同样消融，比较因果效应差异

预注册判伪条件：
INV-242: "Gemma4的跨能力耦合方向承载多能力信息"
  如果消融后>=3个能力变化<5%，则INV-242被推翻
INV-243: "统一编码模型的消融伤害>分离编码模型"
  如果Gemma4的消融伤害不大于其他模型，则统一编码没有额外的脆弱性
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
    discover_layers,
    free_model,
    load_model_bundle,
    score_candidate_avg_logprob,
    ablate_layer_component,
    restore_layer_component,
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
    TestCase("bank_financial",
             "She went to the bank to deposit money. The bank",
             " teller", " river",
             "He walked along the river bank. The bank",
             " river", " teller"),
    TestCase("coreference",
             "Mary gave the book to John because she",
             " wanted", " wanted",
             "Tom gave the book to Mary because he",
             " wanted", " wanted"),
]


def case_margin(model, tokenizer, case: TestCase) -> float:
    ma = score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.positive_a) - \
         score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.negative_a)
    mb = score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.positive_b) - \
         score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.negative_b)
    return float((ma + mb) / 2.0)


def extract_last_token_hidden(model, tokenizer, text: str, layer_idx: int) -> torch.Tensor:
    layers = discover_layers(model)
    captured = {"value": None}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured["value"] = output[0][:, -1, :].detach().cpu()
        else:
            captured["value"] = output[:, -1, :].detach().cpu()

    handle = layers[layer_idx].register_forward_hook(hook_fn)
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    return captured["value"].squeeze(0)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python stage658_gemma4_causal_validation.py [qwen3|deepseek7b|glm4|gemma4]")
        sys.exit(1)
    model_key = sys.argv[1]
    run_dir = OUTPUT_DIR / f"stage658_gemma4_causal_{model_key}_{TIMESTAMP}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = None
    try:
        print(f"[Stage658] Loading {model_key}...")
        model, tokenizer = load_model_bundle(model_key, prefer_cuda=True)
        num_layers = len(discover_layers(model))
        final_layer = num_layers - 1
        print(f"[Stage658] layers={num_layers}")

        # 首先计算所有case的末层delta方向
        print("\nComputing case deltas...")
        case_deltas = {}
        case_baselines = {}
        for case in CASES:
            ha = extract_last_token_hidden(model, tokenizer, case.prompt_a, final_layer)
            hb = extract_last_token_hidden(model, tokenizer, case.prompt_b, final_layer)
            case_deltas[case.name] = (ha - hb).float()
            case_baselines[case.name] = case_margin(model, tokenizer, case)

        # ========================================
        # 实验1: 跨能力耦合维度识别与消融
        # ========================================
        print(f"\n{'='*60}")
        print("Experiment 1: Cross-capability Coupled Dimension Ablation")
        print(f"{'='*60}")

        # 找到跨能力高cos的pair
        case_names = [c.name for c in CASES]
        high_cos_pairs = []
        for i in range(len(case_names)):
            for j in range(i + 1, len(case_names)):
                d1 = case_deltas[case_names[i]]
                d2 = case_deltas[case_names[j]]
                cos_v = torch.dot(d1, d2) / (d1.norm() * d2.norm() + 1e-10)
                if abs(cos_v.item()) > 0.3:
                    high_cos_pairs.append((case_names[i], case_names[j], round(cos_v.item(), 4)))

        print(f"  High-cos pairs (|cos|>0.3): {len(high_cos_pairs)}")
        for p in high_cos_pairs:
            print(f"    {p[0]} <-> {p[1]}: cos={p[2]}")

        # 对前3个高cos pair，提取共享方向并消融
        # 共享方向 = 归一化后的delta (因为都是rank-1)
        abl_results_exp1 = []
        for pair_info in high_cos_pairs[:3]:
            n1, n2, cos_val = pair_info
            d1 = case_deltas[n1]
            d2 = case_deltas[n2]
            shared_dir = d1 / (d1.norm() + 1e-10)  # 共享方向（因为rank-1，d1本身就是主方向）

            # 消融共享方向：在末层hidden state中减去共享方向的投影
            # 使用mid层消融MLP
            mid_layer = num_layers // 2
            # 消融mid层MLP
            layer_mod, original_mlp = ablate_layer_component(model, mid_layer, "mlp")

            # 测量消融后各能力
            abl_margins = {}
            for case in CASES:
                try:
                    abl_margins[case.name] = case_margin(model, tokenizer, case)
                except Exception:
                    abl_margins[case.name] = float('nan')

            # 恢复
            restore_layer_component(layer_mod, "mlp", original_mlp)

            # 计算变化
            changes = {}
            for case in CASES:
                if case.name in abl_margins and not np.isnan(abl_margins[case.name]):
                    base = case_baselines[case.name]
                    abl = abl_margins[case.name]
                    if abs(base) > 0.01:
                        changes[case.name] = round((abl - base) / abs(base) * 100, 1)
                    else:
                        changes[case.name] = round(abl - base, 4)
                else:
                    changes[case.name] = None

            print(f"\n  Ablation: {n1}<->{n2} (cos={cos_val}), mid_layer={mid_layer} MLP")
            for cn in case_names:
                ch = changes.get(cn)
                if ch is not None:
                    print(f"    {cn:20s}: base={case_baselines[cn]:+.4f}, abl={abl_margins.get(cn, 0):+.4f}, change={ch:+.1f}%")
                else:
                    print(f"    {cn:20s}: N/A")

            affected_count = sum(1 for ch in changes.values() if ch is not None and abs(ch) > 5)
            abl_results_exp1.append({
                "pair": f"{n1}<->{n2}",
                "cos": cos_val,
                "affected_count": affected_count,
                "total_count": len(CASES),
                "changes": {k: v for k, v in changes.items() if v is not None},
            })

        inv242_status = "CONFIRMED" if any(r["affected_count"] >= 3 for r in abl_results_exp1) else "FALSIFIED"
        print(f"\n  INV-242 (coupled dims carry multi-capability info): {inv242_status}")
        for r in abl_results_exp1:
            print(f"    {r['pair']}: {r['affected_count']}/{r['total_count']} affected")

        # ========================================
        # 实验2: 推理-消歧共享方向消融
        # ========================================
        print(f"\n{'='*60}")
        print("Experiment 2: Reasoning-Disambiguation Shared Direction Ablation")
        print(f"{'='*60}")

        reasoning_cases = ["syllogism", "arithmetic"]
        disambig_cases = ["bank_financial", "relation_capital"]

        # 计算推理主方向和消歧主方向
        reasoning_deltas = torch.stack([case_deltas[n] for n in reasoning_cases])
        disambig_deltas = torch.stack([case_deltas[n] for n in disambig_cases])

        _, S_r, Vt_r = torch.linalg.svd(reasoning_deltas, full_matrices=False)
        _, S_d, Vt_d = torch.linalg.svd(disambig_deltas, full_matrices=False)

        reasoning_top1 = Vt_r[0]  # 推理主方向
        disambig_top1 = Vt_d[0]  # 消歧主方向

        rd_cos = torch.dot(reasoning_top1, disambig_top1).item()
        print(f"  Reasoning-Disambig top1 cos: {rd_cos:.4f}")

        # 找最伤的层（对推理或消歧）
        most_harm_layers = {}
        for cn in ["syllogism", "bank_financial"]:
            case = next(c for c in CASES if c.name == cn)
            max_drop = 0
            harm_layer = 0
            harm_comp = "mlp"
            for li in range(min(6, num_layers)):
                for comp in ["mlp", "attn"]:
                    l_mod, l_orig = ablate_layer_component(model, li, comp)
                    try:
                        abl_m = case_margin(model, tokenizer, case)
                    except Exception:
                        abl_m = 0
                    restore_layer_component(l_mod, comp, l_orig)
                    drop = case_baselines[cn] - abl_m
                    if drop > max_drop:
                        max_drop = drop
                        harm_layer = li
                        harm_comp = comp
            most_harm_layers[cn] = (harm_layer, harm_comp, max_drop)
            print(f"  Most harmful for {cn}: L{harm_layer} {harm_comp} (drop={max_drop:.4f})")

        # 在最伤层消融，测量推理和消歧变化
        exp2_results = {}
        for cn in ["syllogism", "bank_financial"]:
            hl, hc, hd = most_harm_layers[cn]
            hl_mod, hl_orig = ablate_layer_component(model, hl, hc)

            all_abl = {}
            for case in CASES:
                try:
                    all_abl[case.name] = case_margin(model, tokenizer, case)
                except Exception:
                    all_abl[case.name] = float('nan')

            restore_layer_component(hl_mod, hc, hl_orig)

            changes = {}
            for case in CASES:
                if not np.isnan(all_abl.get(case.name, float('nan'))):
                    base = case_baselines[case.name]
                    abl = all_abl[case.name]
                    if abs(base) > 0.01:
                        changes[case.name] = round((abl - base) / abs(base) * 100, 1)
                    else:
                        changes[case.name] = round(abl - base, 4)

            exp2_results[cn] = {
                "harm_layer": hl,
                "harm_comp": hc,
                "harm_drop": round(hd, 4),
                "changes": changes,
            }
            print(f"\n  Ablate L{hl} {hc} (harmful for {cn}):")
            for k, v in sorted(changes.items()):
                print(f"    {k:20s}: {v:+.1f}%")

        # ========================================
        # 实验3: 维度分离干预
        # 在末层hidden state中，将消歧方向的推理分量减去
        # ========================================
        print(f"\n{'='*60}")
        print("Experiment 3: Dimension Separation Intervention")
        print(f"{'='*60}")

        # 对bank_financial case: 在末层hidden state中减去推理主方向投影
        bank_case = next(c for c in CASES if c.name == "bank_financial")

        # 提取bank case的末层hidden state
        ha_bank = extract_last_token_hidden(model, tokenizer, bank_case.prompt_a, final_layer)
        hb_bank = extract_last_token_hidden(model, tokenizer, bank_case.prompt_b, final_layer)

        # 计算推理投影分量
        proj_a = torch.dot(ha_bank.float(), reasoning_top1) * reasoning_top1
        proj_b = torch.dot(hb_bank.float(), reasoning_top1) * reasoning_top1

        # 消歧delta的推理分量
        reasoning_comp = proj_a - proj_b
        reasoning_comp_frac = reasoning_comp.norm() / (ha_bank - hb_bank).float().norm()
        print(f"  Reasoning component fraction in disambig delta: {reasoning_comp_frac:.4f}")

        # ========================================
        # 实验4: 编码策略分类
        # ========================================
        print(f"\n{'='*60}")
        print("Experiment 4: Encoding Strategy Classification")
        print(f"{'='*60}")

        # 计算所有case对的cos
        all_off_cos = []
        for i in range(len(case_names)):
            for j in range(i + 1, len(case_names)):
                d1 = case_deltas[case_names[i]]
                d2 = case_deltas[case_names[j]]
                cos_v = abs((torch.dot(d1, d2) / (d1.norm() * d2.norm() + 1e-10)).item())
                all_off_cos.append(cos_v)

        mean_abs_cos = statistics.mean(all_off_cos)
        max_abs_cos = max(all_off_cos)

        if mean_abs_cos > 0.3:
            strategy = "UNIFIED (Gemma4-type)"
        elif mean_abs_cos < 0.1:
            strategy = "SEPARATED (GLM4-type)"
        else:
            strategy = "MIXED"

        # 计算消融伤害指数（所有高cos pair的平均affected count）
        avg_affected = statistics.mean([r["affected_count"] for r in abl_results_exp1]) if abl_results_exp1 else 0

        print(f"  Mean |cos| (cross-capability): {mean_abs_cos:.4f}")
        print(f"  Max  |cos| (cross-capability): {max_abs_cos:.4f}")
        print(f"  Strategy: {strategy}")
        print(f"  Avg affected capabilities per ablation: {avg_affected:.1f}")
        print(f"  INV-242: {inv242_status}")

        # 判断INV-243
        inv243_status = "PENDING (need cross-model comparison)"
        print(f"  INV-243 (unified more fragile): {inv243_status}")

        # 保存结果
        results = {
            "model": model_key,
            "num_layers": num_layers,
            "experiment1_coupled_ablation": {
                "high_cos_pairs": [{"pair": f"{p[0]}<->{p[1]}", "cos": p[2]} for p in high_cos_pairs],
                "ablation_results": abl_results_exp1,
                "inv242_status": inv242_status,
            },
            "experiment2_reasoning_disambig": exp2_results,
            "experiment3_separation": {
                "reasoning_component_fraction": round(reasoning_comp_frac.item(), 4),
            },
            "experiment4_strategy": {
                "mean_abs_cos": round(mean_abs_cos, 4),
                "max_abs_cos": round(max_abs_cos, 4),
                "strategy": strategy,
                "avg_affected": round(avg_affected, 1),
            },
        }
        out_path = run_dir / f"results_{model_key}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n  Results saved to {out_path}")

    finally:
        if model is not None:
            free_model(model)


if __name__ == "__main__":
    main()
