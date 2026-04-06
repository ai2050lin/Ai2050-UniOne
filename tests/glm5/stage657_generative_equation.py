#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage657: P8 生成方程原型

P0-P7积累了大量关于编码策略的知识：
- INV-208: delta统一rank-1
- INV-229: 前层MLP存在抑制(28-45%)
- INV-195: 跨能力方向近似正交(3/4模型)
- 统一方程: logit_margin = cos(d, delta_u) * ||d|| * ||delta_u||
- 残差流方程: h_l = h_0 + sum(delta_k), 误差=0

P8目标：构建两类生成方程
(a) GLM4型分离编码方程: 能力方向近似正交
(b) Gemma4型统一编码方程: 能力方向高度耦合

实验1：方向预测方程 — 给定句子A和B，用前几层delta预测末层delta方向
实验2：幅度预测方程 — 用逐层delta范数累积预测末层delta幅度
实验3：logit margin预测 — 组合方向和幅度预测最终消歧效果
实验4：跨模型方程精度比较

预注册判伪条件：
INV-240: "前层delta方向可预测末层delta方向(cos>0.5)"
  如果cos<0.3, 则前层delta方向不含末层方向信息
INV-241: "累积范数可预测末层delta范数(R^2>0.5)"
  如果R^2<0.3, 则幅度预测不可行
"""

from __future__ import annotations

import sys
import json
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    discover_layers,
    free_model,
    load_model_bundle,
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
        print("Usage: python stage657_generative_equation.py [qwen3|deepseek7b|glm4|gemma4]")
        sys.exit(1)
    model_key = sys.argv[1]
    run_dir = OUTPUT_DIR / f"stage657_generative_{model_key}_{TIMESTAMP}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = None
    try:
        print(f"[Stage657] Loading {model_key}...")
        model, tokenizer = load_model_bundle(model_key, prefer_cuda=True)
        num_layers = len(discover_layers(model))
        final_layer = num_layers - 1
        print(f"[Stage657] layers={num_layers}")

        # 采样层: L0, L2, L4, L_mid, L_last
        sample_layers = sorted(set([
            0,
            min(2, final_layer),
            min(4, final_layer),
            num_layers // 2,
            final_layer,
        ]))

        # ========================================
        # 实验1: 方向预测方程
        # 用前层delta方向预测末层delta方向
        # ========================================
        print(f"\n{'='*60}")
        print("Experiment 1: Direction Prediction Equation")
        print(f"{'='*60}")

        dir_prediction_results = []

        for case in CASES:
            # 提取末层真实delta方向
            ha_last = extract_last_token_hidden(model, tokenizer, case.prompt_a, final_layer)
            hb_last = extract_last_token_hidden(model, tokenizer, case.prompt_b, final_layer)
            delta_last = (ha_last - hb_last).float()
            true_dir = delta_last / (delta_last.norm() + 1e-10)

            layer_cos = {}
            for li in sample_layers[:-1]:  # 不含末层
                ha_i = extract_last_token_hidden(model, tokenizer, case.prompt_a, li)
                hb_i = extract_last_token_hidden(model, tokenizer, case.prompt_b, li)
                delta_i = (ha_i - hb_i).float()
                early_dir = delta_i / (delta_i.norm() + 1e-10)
                cos_val = torch.dot(true_dir, early_dir).item()
                layer_cos[li] = round(cos_val, 4)

            # 用L0方向预测末层方向
            l0_cos = layer_cos.get(0, 0)
            # 用线性组合(所有前层delta之和)预测
            combined_delta = torch.zeros_like(delta_last)
            for li in sample_layers[:-1]:
                ha_i = extract_last_token_hidden(model, tokenizer, case.prompt_a, li)
                hb_i = extract_last_token_hidden(model, tokenizer, case.prompt_b, li)
                combined_delta += (ha_i - hb_i).float()
            if combined_delta.norm() > 1e-6:
                combined_dir = combined_delta / combined_delta.norm()
                combined_cos = torch.dot(true_dir, combined_dir).item()
            else:
                combined_cos = 0.0

            # 用残差流累积预测(完整逐层)
            ha_0 = extract_last_token_hidden(model, tokenizer, case.prompt_a, 0)
            hb_0 = extract_last_token_hidden(model, tokenizer, case.prompt_b, 0)
            residual_delta = torch.zeros_like(delta_last)
            for li in range(num_layers):
                ha_i = extract_last_token_hidden(model, tokenizer, case.prompt_a, li)
                hb_i = extract_last_token_hidden(model, tokenizer, case.prompt_b, li)
                # delta_l = h_l - h_{l-1} (近似: 用前层)
                if li > 0:
                    delta_k = (ha_i - ha_prev).float() - (hb_i - hb_prev).float()
                    residual_delta += delta_k
                ha_prev = ha_i
                hb_prev = hb_i

            if residual_delta.norm() > 1e-6:
                residual_dir = residual_delta / residual_delta.norm()
                residual_cos = torch.dot(true_dir, residual_dir).item()
            else:
                residual_cos = 0.0

            dir_prediction_results.append({
                "case": case.name,
                "l0_cos": round(l0_cos, 4),
                "combined_cos": round(combined_cos, 4),
                "residual_cos": round(residual_cos, 4),
                "layer_cos": layer_cos,
            })
            print(f"  {case.name:20s}: L0={l0_cos:+.4f}, combined={combined_cos:+.4f}, residual={residual_cos:+.4f}")

        avg_l0_cos = statistics.mean([r["l0_cos"] for r in dir_prediction_results])
        avg_combined_cos = statistics.mean([r["combined_cos"] for r in dir_prediction_results])
        avg_residual_cos = statistics.mean([r["residual_cos"] for r in dir_prediction_results])

        inv240_status = "CONFIRMED" if avg_residual_cos > 0.5 else ("PARTIAL" if avg_residual_cos > 0.3 else "FALSIFIED")
        print(f"\n  Mean L0 cos: {avg_l0_cos:.4f}")
        print(f"  Mean combined cos: {avg_combined_cos:.4f}")
        print(f"  Mean residual cos: {avg_residual_cos:.4f}")
        print(f"  INV-240 (direction prediction): {inv240_status}")

        # ========================================
        # 实验2: 幅度预测方程
        # 用逐层delta范数累积预测末层delta范数
        # ========================================
        print(f"\n{'='*60}")
        print("Experiment 2: Magnitude Prediction Equation")
        print(f"{'='*60}")

        mag_prediction_results = []

        for case in CASES:
            ha_last = extract_last_token_hidden(model, tokenizer, case.prompt_a, final_layer)
            hb_last = extract_last_token_hidden(model, tokenizer, case.prompt_b, final_layer)
            true_norm = (ha_last - hb_last).float().norm().item()

            # 方法1: 逐层delta范数绝对值累加
            abs_sum = 0.0
            for li in range(num_layers):
                if li == 0:
                    ha_prev = extract_last_token_hidden(model, tokenizer, case.prompt_a, 0)
                    hb_prev = extract_last_token_hidden(model, tokenizer, case.prompt_b, 0)
                    continue
                ha_i = extract_last_token_hidden(model, tokenizer, case.prompt_a, li)
                hb_i = extract_last_token_hidden(model, tokenizer, case.prompt_b, li)
                dk = (ha_i - ha_prev).float() - (hb_i - hb_prev).float()
                abs_sum += dk.norm().item()
                ha_prev = ha_i
                hb_prev = hb_i

            # 方法2: 逐层delta范数的L2累加(考虑方向)
            l2_sum_sq = 0.0
            ha_0 = extract_last_token_hidden(model, tokenizer, case.prompt_a, 0)
            hb_0 = extract_last_token_hidden(model, tokenizer, case.prompt_b, 0)
            for li in range(1, num_layers):
                ha_i = extract_last_token_hidden(model, tokenizer, case.prompt_a, li)
                hb_i = extract_last_token_hidden(model, tokenizer, case.prompt_b, li)
                dk = (ha_i - ha_prev).float() - (hb_i - hb_prev).float()
                l2_sum_sq += dk.norm().item() ** 2
                ha_prev = ha_i
                hb_prev = hb_i
            l2_norm = (l2_sum_sq ** 0.5)

            # 方法3: 直接计算 d_norm 增长
            # d_l = h_l^A - h_l^B, 直接用残差流差
            ha_prev = ha_0
            hb_prev = hb_0
            d_direct = (ha_0 - hb_0).float()
            for li in range(1, num_layers):
                ha_i = extract_last_token_hidden(model, tokenizer, case.prompt_a, li)
                hb_i = extract_last_token_hidden(model, tokenizer, case.prompt_b, li)
                d_direct = (ha_i - hb_i).float()
                ha_prev = ha_i
                hb_prev = hb_i

            ratio_abs = abs_sum / true_norm if true_norm > 1e-6 else float('inf')
            ratio_l2 = l2_norm / true_norm if true_norm > 1e-6 else float('inf')
            ratio_direct = d_direct.norm().item() / true_norm if true_norm > 1e-6 else 1.0

            mag_prediction_results.append({
                "case": case.name,
                "true_norm": round(true_norm, 2),
                "abs_sum": round(abs_sum, 2),
                "l2_norm": round(l2_norm, 2),
                "ratio_abs": round(ratio_abs, 2),
                "ratio_l2": round(ratio_l2, 2),
                "ratio_direct": round(ratio_direct, 2),
            })
            print(f"  {case.name:20s}: ||d||={true_norm:.2f}, abs_sum={abs_sum:.2f}(x{ratio_abs:.1f}), "
                  f"l2={l2_norm:.2f}(x{ratio_l2:.1f}), direct={ratio_direct:.2f}")

        # 计算R^2: abs_sum vs true_norm
        true_norms = [r["true_norm"] for r in mag_prediction_results]
        abs_sums = [r["abs_sum"] for r in mag_prediction_results]
        mean_true = statistics.mean(true_norms)
        ss_tot = sum((t - mean_true) ** 2 for t in true_norms)
        ss_res = sum((a - t) ** 2 for a, t in zip(abs_sums, true_norms))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        inv241_status = "CONFIRMED" if r2 > 0.5 else ("PARTIAL" if r2 > 0.3 else "FALSIFIED")
        print(f"\n  R^2 (abs_sum vs true_norm): {r2:.4f}")
        print(f"  INV-241 (magnitude prediction): {inv241_status}")

        # ========================================
        # 实验3: logit margin预测
        # 组合方向和幅度预测最终消歧效果
        # ========================================
        print(f"\n{'='*60}")
        print("Experiment 3: Logit Margin Prediction")
        print(f"{'='*60}")

        logit_results = []

        for case in CASES:
            # 真实margin
            true_margin = case_margin(model, tokenizer, case)

            # 末层delta方向和幅度
            ha_last = extract_last_token_hidden(model, tokenizer, case.prompt_a, final_layer)
            hb_last = extract_last_token_hidden(model, tokenizer, case.prompt_b, final_layer)
            delta = (ha_last - hb_last).float()
            d_norm = delta.norm().item()
            d_dir = delta / (d_norm + 1e-10)

            # unembed方向: 用unembed矩阵(而非input embedding)
            try:
                # 获取unembed矩阵
                if hasattr(model, 'lm_head'):
                    unembed = model.lm_head
                elif hasattr(model, 'output'):
                    unembed = model.output
                else:
                    # 尝试从config获取
                    unembed = None

                if unembed is not None and hasattr(unembed, 'weight'):
                    pos_ids = tokenizer(case.positive_a.strip(), add_special_tokens=False)["input_ids"]
                    neg_ids = tokenizer(case.negative_a.strip(), add_special_tokens=False)["input_ids"]
                    # 使用unembed矩阵的行
                    W = unembed.weight.detach().cpu().float()
                    pos_emb = W[pos_ids].mean(dim=0)
                    neg_emb = W[neg_ids].mean(dim=0)
                    delta_u = pos_emb - neg_emb
                    delta_u_norm = delta_u.norm().item()
                    delta_u_dir = delta_u / (delta_u_norm + 1e-10)
                    cos_alignment = torch.dot(d_dir, delta_u_dir).item()
                    predicted_margin = cos_alignment * d_norm * delta_u_norm
                else:
                    cos_alignment = 0
                    predicted_margin = 0
                    delta_u_norm = 0
            except Exception as e:
                cos_alignment = 0
                predicted_margin = 0
                delta_u_norm = 0

            logit_results.append({
                "case": case.name,
                "true_margin": round(true_margin, 4),
                "d_norm": round(d_norm, 2),
                "delta_u_norm": round(delta_u_norm, 4),
                "cos_alignment": round(cos_alignment, 6),
                "predicted_margin": round(predicted_margin, 4),
                "prediction_ratio": round(predicted_margin / abs(true_margin), 4) if abs(true_margin) > 0.01 else float('inf'),
            })
            print(f"  {case.name:20s}: true={true_margin:+.4f}, pred={predicted_margin:+.4f}, "
                  f"d_norm={d_norm:.2f}, cos={cos_alignment:.6f}")

        # 计算R^2
        true_margins = [r["true_margin"] for r in logit_results]
        pred_margins = [r["predicted_margin"] for r in logit_results]
        mean_tm = statistics.mean(true_margins)
        ss_tot_m = sum((t - mean_tm) ** 2 for t in true_margins)
        ss_res_m = sum((p - t) ** 2 for p, t in zip(pred_margins, true_margins))
        r2_margin = 1 - ss_res_m / ss_tot_m if ss_tot_m > 0 else 0

        # Pearson correlation
        if len(true_margins) > 1:
            std_t = statistics.pstdev(true_margins) + 1e-10
            std_p = statistics.pstdev(pred_margins) + 1e-10
            mean_p = statistics.mean(pred_margins)
            pearson = sum((t - mean_tm) * (p - mean_p) for t, p in zip(true_margins, pred_margins)) / (len(true_margins) * std_t * std_p)
        else:
            pearson = 0

        print(f"\n  R^2 (logit margin prediction): {r2_margin:.4f}")
        print(f"  Pearson correlation: {pearson:.4f}")
        print(f"  Mean cos alignment: {statistics.mean([r['cos_alignment'] for r in logit_results]):.6f}")

        # ========================================
        # 实验4: 跨模型方程精度比较
        # GLM4型(分离) vs Gemma4型(统一)
        # ========================================
        print(f"\n{'='*60}")
        print("Experiment 4: Encoding Strategy Classification")
        print(f"{'='*60}")

        # 计算跨case方向余弦矩阵
        case_deltas = {}
        for case in CASES:
            ha = extract_last_token_hidden(model, tokenizer, case.prompt_a, final_layer)
            hb = extract_last_token_hidden(model, tokenizer, case.prompt_b, final_layer)
            case_deltas[case.name] = (ha - hb).float()

        cos_matrix = {}
        case_names = [c.name for c in CASES]
        off_diag_cos = []
        for i, n1 in enumerate(case_names):
            for j, n2 in enumerate(case_names):
                if i >= j:
                    continue
                d1 = case_deltas[n1]
                d2 = case_deltas[n2]
                cos_v = torch.dot(d1, d2) / (d1.norm() * d2.norm() + 1e-10).item()
                cos_val = round(cos_v.item(), 4)
                cos_matrix[f"{n1}-{n2}"] = cos_val
                off_diag_cos.append(abs(cos_val))

        mean_abs_cos = statistics.mean(off_diag_cos) if off_diag_cos else 0
        max_abs_cos = max(off_diag_cos) if off_diag_cos else 0

        # 编码策略分类
        if mean_abs_cos > 0.3:
            strategy = "UNIFIED (Gemma4-type)"
        elif mean_abs_cos < 0.1:
            strategy = "SEPARATED (GLM4-type)"
        else:
            strategy = "MIXED"

        print(f"  Mean |cos| (cross-capability): {mean_abs_cos:.4f}")
        print(f"  Max  |cos| (cross-capability): {max_abs_cos:.4f}")
        print(f"  Encoding strategy: {strategy}")
        print(f"  Direction prediction cos (residual): {avg_residual_cos:.4f}")
        print(f"  Magnitude prediction R^2: {r2:.4f}")
        print(f"  Logit margin R^2: {r2_margin:.4f}")

        # 保存结果
        results = {
            "model": model_key,
            "num_layers": num_layers,
            "experiment1_direction": {
                "avg_l0_cos": round(avg_l0_cos, 4),
                "avg_combined_cos": round(avg_combined_cos, 4),
                "avg_residual_cos": round(avg_residual_cos, 4),
                "inv240_status": inv240_status,
                "per_case": dir_prediction_results,
            },
            "experiment2_magnitude": {
                "r2_abs_sum": round(r2, 4),
                "inv241_status": inv241_status,
                "per_case": mag_prediction_results,
            },
            "experiment3_logit": {
                "r2_margin": round(r2_margin, 4),
                "pearson": round(pearson, 4),
                "per_case": logit_results,
            },
            "experiment4_strategy": {
                "mean_abs_cos": round(mean_abs_cos, 4),
                "max_abs_cos": round(max_abs_cos, 4),
                "strategy": strategy,
                "cos_matrix": cos_matrix,
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
