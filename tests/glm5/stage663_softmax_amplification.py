#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage663: P16 Softmax非线性放大效应

P15发现（预期）：cos_alignment仅3-5%，但模型仍然能做出正确选择
P16目标：分析softmax是否将微小的logit差非线性放大到可区分的概率

核心思路：
1. 计算raw logit分布 logit = W_u @ h_L
2. 分析logit间距分布（top-1与correct之间的gap）
3. 计算softmax后的概率比
4. 量化非线性放大倍数：prob_ratio / exp(logit_gap)
5. 分析"温度"效应——不同温度下放大倍数如何变化
6. 测试：消歧方向对齐度(cos=3%)经过softmax后是否变得显著

额外实验：
7. 零化消歧方向后，softmax概率分布的变化
8. logit差 vs margin(score函数)的定量关系
"""

from __future__ import annotations

import sys
import io
import json

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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
    prompt_b: str
    positive_a: str
    negative_a: str


CASES = [
    TestCase("bank_financial",
             "The bank approved the loan with favorable terms.",
             "The river bank was covered with wild flowers.",
             "financial", "river"),
    TestCase("syllogism",
             "All men are mortal. Socrates is a man. Therefore,",
             "All men are mortal. Socrates is a cat. Therefore,",
             "Socrates is mortal", "Socrates is immortal"),
    TestCase("arithmetic",
             "If x = 7 and y = 3, then x + y =",
             "If x = 7 and y = 3, then x - y =",
             "10", "4"),
    TestCase("grammar_subj",
             "The cats are sleeping on the sofa quietly.",
             "The cat is sleeping on the sofa quietly.",
             "are", "is"),
    TestCase("coreference",
             "John told Mary that he would help her with the project.",
             "Mary told John that he would help her with the project.",
             "John", "Mary"),
    TestCase("style_formal",
             "Dear Professor Smith, I would like to inquire about",
             "Hey what's up Smith, wanna know about",
             "regarding", "bout"),
]


def extract_last_hidden(model, tokenizer, prompt: str) -> torch.Tensor:
    """提取最后一层hidden state"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    hidden_cache = {}
    layers = discover_layers(model)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hs = output[0][:, -1, :].detach().cpu().float().squeeze(0)
        else:
            hs = output[:, -1, :].detach().cpu().float().squeeze(0)
        hidden_cache["last"] = hs

    h = layers[len(layers) - 1].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        h.remove()
    return hidden_cache["last"]


def get_unembed_matrix(model) -> torch.Tensor:
    if hasattr(model, "lm_head"):
        return model.lm_head.weight.detach().cpu().float()
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens.weight.detach().cpu().float()
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte.weight.detach().cpu().float()
    raise RuntimeError("未找到unembedding矩阵(lm_head)")


def token_id_from_text(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if ids else tokenizer.encode(" ", add_special_tokens=False)[0]


def softmax_with_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return torch.softmax(logits / temperature, dim=-1)


def entropy(probs: torch.Tensor) -> float:
    log_probs = torch.log(probs + 1e-10)
    return -(probs * log_probs).sum().item()


def run_experiment(model_key):
    print(f"\n{'='*70}")
    print(f"Stage663: P16 Softmax放大效应 - {model_key}")
    print(f"{'='*70}")

    model, tokenizer = load_model_bundle(model_key)
    W_u = get_unembed_matrix(model)
    vocab_size = W_u.shape[0]

    # ========== 实验1: 基础softmax放大分析 ==========
    print("\n[Experiment 1] 基础Softmax放大分析...")

    amplification_results = []

    for case in CASES:
        h_a = extract_last_hidden(model, tokenizer, case.prompt_a)
        logits = W_u @ h_a

        tid_correct = token_id_from_text(tokenizer, case.positive_a)
        tid_incorrect = token_id_from_text(tokenizer, case.negative_a)

        logit_correct = logits[tid_correct].item()
        logit_incorrect = logits[tid_incorrect].item()
        logit_gap = logit_correct - logit_incorrect

        probs = torch.softmax(logits, dim=-1)
        prob_correct = probs[tid_correct].item()
        prob_incorrect = probs[tid_incorrect].item()
        prob_ratio = prob_correct / (prob_incorrect + 1e-10)

        # 理论放大倍数（softmax的梯度效应）
        # softmax下 prob_c/prob_i ≈ exp(logit_c - logit_i)
        # 放大倍数 = prob_ratio / exp(logit_gap)  应该≈1
        theoretical_ratio = np.exp(logit_gap)
        amplification_factor = prob_ratio / (theoretical_ratio + 1e-10)

        # logit分布的熵（信息量指标）
        ent = entropy(probs)

        # top-1 token
        top1_idx = logits.argmax().item()
        top1_logit = logits[top1_idx].item()
        top1_prob = probs[top1_idx].item()
        top1_token = repr(tokenizer.decode([top1_idx]))

        # correct token与top-1的关系
        gap_to_top1 = logit_correct - top1_logit

        # score函数margin
        score_margin = (score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.positive_a) -
                       score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.negative_a))

        result = {
            "case": case.name,
            "logit_correct": round(logit_correct, 4),
            "logit_incorrect": round(logit_incorrect, 4),
            "logit_gap": round(logit_gap, 4),
            "prob_correct": round(prob_correct, 8),
            "prob_incorrect": round(prob_incorrect, 8),
            "prob_ratio": round(prob_ratio, 4),
            "amplification_factor": round(amplification_factor, 4),
            "entropy": round(ent, 4),
            "top1_token": top1_token.strip(),
            "top1_logit": round(top1_logit, 4),
            "top1_prob": round(top1_prob, 6),
            "gap_correct_vs_top1": round(gap_to_top1, 4),
            "score_margin": round(score_margin, 4),
        }
        amplification_results.append(result)

        print(f"  {case.name}: logit_gap={logit_gap:.2f}, prob_c={prob_correct:.6f}, "
              f"prob_i={prob_incorrect:.6f}, ratio={prob_ratio:.1f}x, amp={amplification_factor:.2f}x")
        print(f"    top1='{top1_token.strip()}'(prob={top1_prob:.4f}), gap_to_top1={gap_to_top1:.2f}")
        print(f"    entropy={ent:.2f}, score_margin={score_margin:.4f}")

    # ========== 实验2: 温度扫描 ==========
    print("\n[Experiment 2] 温度扫描...")

    temperature_results = {}
    test_case = CASES[0]  # bank_financial
    h_a = extract_last_hidden(model, tokenizer, test_case.prompt_a)
    logits = W_u @ h_a

    tid_correct = token_id_from_text(tokenizer, test_case.positive_a)
    tid_incorrect = token_id_from_text(tokenizer, test_case.negative_a)

    temperatures = [0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 5.0, 10.0]

    for temp in temperatures:
        probs = softmax_with_temperature(logits, temp)
        p_c = probs[tid_correct].item()
        p_i = probs[tid_incorrect].item()
        ratio = p_c / (p_i + 1e-10)
        ent = entropy(probs)

        # 放大倍数（相对于线性缩放）
        logit_gap = (logits[tid_correct] - logits[tid_incorrect]).item()
        scaled_gap = logit_gap / temp
        theoretical_ratio = np.exp(scaled_gap)
        amp = ratio / (theoretical_ratio + 1e-10)

        temperature_results[str(temp)] = {
            "prob_correct": round(p_c, 8),
            "prob_incorrect": round(p_i, 8),
            "prob_ratio": round(ratio, 4),
            "entropy": round(ent, 4),
            "amplification": round(amp, 4),
        }
        print(f"  T={temp:.1f}: prob_c={p_c:.6f}, prob_i={p_i:.6f}, "
              f"ratio={ratio:.2f}x, entropy={ent:.2f}, amp={amp:.2f}x")

    # ========== 实验3: 消歧方向零化实验 ==========
    print("\n[Experiment 3] 消歧方向零化对logit分布的影响...")

    zeroing_results = []
    for case in CASES[:4]:
        h_a = extract_last_hidden(model, tokenizer, case.prompt_a)
        h_b = extract_last_hidden(model, tokenizer, case.prompt_b)
        d_L = h_a - h_b

        logits_orig = W_u @ h_a
        probs_orig = torch.softmax(logits_orig, dim=-1)

        # 零化消歧方向：从h_a中移除d_L在h_a上的投影
        d_proj = torch.dot(h_a, d_L) / (d_L.norm() ** 2 + 1e-10) * d_L
        h_a_zeroed = h_a - d_proj

        logits_zeroed = W_u @ h_a_zeroed
        probs_zeroed = torch.softmax(logits_zeroed, dim=-1)

        tid_correct = token_id_from_text(tokenizer, case.positive_a)
        tid_incorrect = token_id_from_text(tokenizer, case.negative_a)

        # 原始 vs 零化后
        orig_gap = logits_orig[tid_correct].item() - logits_orig[tid_incorrect].item()
        zeroed_gap = logits_zeroed[tid_correct].item() - logits_zeroed[tid_incorrect].item()
        gap_change = (zeroed_gap - orig_gap) / (abs(orig_gap) + 1e-10) * 100

        orig_prob_c = probs_orig[tid_correct].item()
        zeroed_prob_c = probs_zeroed[tid_correct].item()
        prob_c_change = (zeroed_prob_c - orig_prob_c) / (orig_prob_c + 1e-10) * 100

        # 排名变化
        orig_rank = (logits_orig.argsort(descending=True) == tid_correct).nonzero()
        zeroed_rank = (logits_zeroed.argsort(descending=True) == tid_correct).nonzero()
        orig_rank = orig_rank.item() if len(orig_rank) > 0 else -1
        zeroed_rank = zeroed_rank.item() if len(zeroed_rank) > 0 else -1

        # 概率分布变化（KL散度）
        kl_div = (probs_orig * (torch.log(probs_orig + 1e-10) - torch.log(probs_zeroed + 1e-10))).sum().item()

        d_proj_norm = d_proj.norm().item()
        d_l_norm = d_L.norm().item()

        result = {
            "case": case.name,
            "d_proj_norm": round(d_proj_norm, 4),
            "d_l_norm": round(d_l_norm, 4),
            "orig_gap": round(orig_gap, 4),
            "zeroed_gap": round(zeroed_gap, 4),
            "gap_change_pct": round(gap_change, 2),
            "orig_prob_correct": round(orig_prob_c, 8),
            "zeroed_prob_correct": round(zeroed_prob_c, 8),
            "prob_c_change_pct": round(prob_c_change, 2),
            "orig_rank": orig_rank,
            "zeroed_rank": zeroed_rank,
            "kl_divergence": round(kl_div, 6),
        }
        zeroing_results.append(result)

        print(f"  {case.name}: gap {orig_gap:.2f}->{zeroed_gap:.2f}({gap_change:+.1f}%), "
              f"prob_c {orig_prob_c:.6f}->{zeroed_prob_c:.6f}({prob_c_change:+.1f}%), "
              f"rank {orig_rank}->{zeroed_rank}, KL={kl_div:.6f}")

    # ========== 实验4: logit_gap vs score_margin的定量关系 ==========
    print("\n[Experiment 4] logit_gap vs score_margin定量关系...")

    correlation_data = []
    for case in CASES:
        h_a = extract_last_hidden(model, tokenizer, case.prompt_a)
        logits = W_u @ h_a

        tid_correct = token_id_from_text(tokenizer, case.positive_a)
        tid_incorrect = token_id_from_text(tokenizer, case.negative_a)

        logit_gap = (logits[tid_correct] - logits[tid_incorrect]).item()

        score_margin = (score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.positive_a) -
                       score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.negative_a))

        correlation_data.append({"case": case.name, "logit_gap": logit_gap, "score_margin": score_margin})
        print(f"  {case.name}: logit_gap={logit_gap:.4f}, score_margin={score_margin:.4f}")

    # 线性回归
    x = np.array([d["logit_gap"] for d in correlation_data])
    y = np.array([d["score_margin"] for d in correlation_data])
    mx, my = x.mean(), y.mean()
    ss_xx = ((x - mx) ** 2).sum()
    ss_yy = ((y - my) ** 2).sum()
    ss_xy = ((x - mx) * (y - my)).sum()
    pearson = ss_xy / np.sqrt(ss_xx * ss_yy) if ss_xx > 0 and ss_yy > 0 else 0
    r2 = pearson ** 2
    slope = ss_xy / ss_xx if ss_xx > 0 else 0
    intercept = my - slope * mx

    print(f"\n  logit_gap vs score_margin: Pearson={pearson:.4f}, R^2={r2:.4f}")
    print(f"  Linear: score_margin = {slope:.4f} * logit_gap + {intercept:.4f}")

    # ========== 实验5: 多token对的cos对齐 vs logit_gap的关系 ==========
    print("\n[Experiment 5] cos_alignment vs logit_gap关系...")

    test_case = CASES[0]
    h_a = extract_last_hidden(model, tokenizer, test_case.prompt_a)
    h_b = extract_last_hidden(model, tokenizer, test_case.prompt_b)
    d_L = h_a - h_b

    tid_correct = token_id_from_text(tokenizer, test_case.positive_a)
    tid_incorrect = token_id_from_text(tokenizer, test_case.negative_a)

    # 采样100个token对
    np.random.seed(42)
    n_samples = 100
    pair_data = []

    # 确保包含正确pair
    for t1, t2, label in [(tid_correct, tid_incorrect, "correct_pair")]:
        u_diff = W_u[t1] - W_u[t2]
        cos_val = float(torch.dot(d_L, u_diff) / (d_L.norm() * u_diff.norm() + 1e-10))
        logit_gap = float(torch.dot(h_a, W_u[t1])) - float(torch.dot(h_a, W_u[t2]))
        pair_data.append({"cos": cos_val, "logit_gap": logit_gap, "label": label})

    # 随机采样
    rand_indices = np.random.choice(vocab_size, size=(n_samples, 2), replace=False)
    for i in range(n_samples):
        t1, t2 = rand_indices[i]
        u_diff = W_u[t1] - W_u[t2]
        cos_val = float(torch.dot(d_L, u_diff) / (d_L.norm() * u_diff.norm() + 1e-10))
        logit_gap = float(torch.dot(h_a, W_u[t1])) - float(torch.dot(h_a, W_u[t2]))
        pair_data.append({"cos": cos_val, "logit_gap": logit_gap, "label": "random"})

    cos_arr = np.array([d["cos"] for d in pair_data])
    gap_arr = np.array([d["logit_gap"] for d in pair_data])
    mx_c, my_g = cos_arr.mean(), gap_arr.mean()
    ss_cc = ((cos_arr - mx_c) ** 2).sum()
    ss_gg = ((gap_arr - my_g) ** 2).sum()
    ss_cg = ((cos_arr - mx_c) * (gap_arr - my_g)).sum()
    pearson_cg = ss_cg / np.sqrt(ss_cc * ss_gg) if ss_cc > 0 and ss_gg > 0 else 0

    print(f"  cos_alignment vs logit_gap: Pearson={pearson_cg:.4f}")
    correct_entry = pair_data[0]
    print(f"  正确pair: cos={correct_entry['cos']:.4f}, logit_gap={correct_entry['logit_gap']:.2f}")
    print(f"  随机pair: cos mean={np.mean(cos_arr[1:]):.4f}, std={np.std(cos_arr[1:]):.4f}")

    # ========== 保存 ==========
    results = {
        "model": model_key,
        "experiment1_amplification": {
            "per_case": amplification_results,
            "mean_amplification": round(statistics.mean([r["amplification_factor"] for r in amplification_results]), 4),
        },
        "experiment2_temperature": temperature_results,
        "experiment3_zeroing": {
            "per_case": zeroing_results,
            "mean_gap_change": round(statistics.mean([r["gap_change_pct"] for r in zeroing_results]), 2),
        },
        "experiment4_correlation": {
            "per_case": correlation_data,
            "pearson": round(pearson, 4),
            "r2": round(r2, 4),
            "slope": round(slope, 4),
            "intercept": round(intercept, 4),
        },
        "experiment5_cos_vs_gap": {
            "pearson": round(pearson_cg, 4),
            "correct_pair": {"cos": round(correct_entry["cos"], 6), "logit_gap": round(correct_entry["logit_gap"], 4)},
            "random_mean_cos": round(np.mean(cos_arr[1:]), 6),
            "random_std_cos": round(np.std(cos_arr[1:]), 6),
        },
    }

    out_dir = OUTPUT_DIR / f"stage663_softmax_amplification_{model_key}_{TIMESTAMP}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"results_{model_key}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_dir}")
    free_model(model)
    return results


if __name__ == "__main__":
    model_key = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_experiment(model_key)
