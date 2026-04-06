#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage662: P15 Logit精确分解 — 从cos=3%到可预测margin

P14发现：cos_alignment仅3-5%，margin低估100倍
P15目标：不使用score_candidate_avg_logprob黑箱，直接从hidden state × unembed矩阵
        计算每个token的logit，分解margin的来源。

核心思路：
1. 提取最后一层hidden state h_L（A版和B版）
2. 计算logit = W_u @ h_L 对所有vocab token
3. 分解margin = logit(correct) - logit(incorrect) 为：
   - 消歧分量: cos(d_L, u_diff) × ||d_L|| × ||u_diff||
   - 共享基底分量: cos(h_shared, u_diff) × ||h_shared|| × ||u_diff||
   其中 d_L = h_A - h_B, h_shared = (h_A + h_B)/2, u_diff = u_correct - u_incorrect

预注册判伪：
INV-280: "消歧分量对margin的贡献 > 10%"
INV-281: "margin = shared_component + disamb_component（加法分解误差<20%）"
INV-282: "消歧分量与score_candidate_avg_logprob得到的margin相关(Pearson>0.7)"
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
]


def extract_last_hidden(model, tokenizer, prompt: str) -> torch.Tensor:
    """提取最后一层transformer block的hidden state（最后一个token位置）"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    hidden_cache = {}
    layers = discover_layers(model)
    last_layer = len(layers) - 1

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hs = output[0][:, -1, :].detach().cpu().float().squeeze(0)
        else:
            hs = output[:, -1, :].detach().cpu().float().squeeze(0)
        hidden_cache["last"] = hs

    h = layers[last_layer].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        h.remove()
    return hidden_cache["last"]


def get_unembed_matrix(model) -> torch.Tensor:
    """获取unembedding矩阵 W_u"""
    # 标准HF模型结构
    if hasattr(model, "lm_head"):
        return model.lm_head.weight.detach().cpu().float()
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        # 部分模型weight-tie（共享embedding和unembedding）
        return model.model.embed_tokens.weight.detach().cpu().float()
    # GLM4可能有不同结构
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte.weight.detach().cpu().float()
    raise RuntimeError("未找到unembedding矩阵(lm_head)")


def compute_logits_from_hidden(h: torch.Tensor, W_u: torch.Tensor) -> torch.Tensor:
    """直接计算 logit = W_u @ h"""
    return W_u @ h


def token_id_from_text(tokenizer, text: str) -> int:
    """获取文本的token id（取第一个token）"""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return tokenizer.encode(" ", add_special_tokens=False)[0]
    return ids[0]


def run_experiment(model_key):
    print(f"\n{'='*70}")
    print(f"Stage662: P15 Logit精确分解 - {model_key}")
    print(f"{'='*70}")

    model, tokenizer = load_model_bundle(model_key)
    W_u = get_unembed_matrix(model)
    vocab_size = W_u.shape[0]
    hidden_dim = W_u.shape[1]
    print(f"  Unembed matrix: vocab_size={vocab_size}, hidden_dim={hidden_dim}")
    print(f"  W_u dtype: {W_u.dtype}")

    # ========== 实验1: Logit精确分解 ==========
    print("\n[Experiment 1: P15] Logit精确分解 (INV-280/281/282)...")

    decomposition_results = []

    for case in CASES:
        print(f"\n  --- {case.name} ---")
        h_a = extract_last_hidden(model, tokenizer, case.prompt_a)
        h_b = extract_last_hidden(model, tokenizer, case.prompt_b)

        # 消歧差异向量
        d_L = h_a - h_b
        # 共享基底（A版和B版的平均）
        h_shared = (h_a + h_b) / 2.0

        # unembed方向
        tid_correct = token_id_from_text(tokenizer, case.positive_a)
        tid_incorrect = token_id_from_text(tokenizer, case.negative_a)
        u_correct = W_u[tid_correct]
        u_incorrect = W_u[tid_incorrect]
        u_diff = u_correct - u_incorrect

        # 从hidden state直接计算logit
        logit_a = compute_logits_from_hidden(h_a, W_u)
        logit_b = compute_logits_from_hidden(h_b, W_u)

        raw_margin_a = logit_a[tid_correct].item() - logit_a[tid_incorrect].item()
        raw_margin_b = logit_b[tid_correct].item() - logit_b[tid_incorrect].item()

        # 用score函数验证
        score_margin_a = score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.positive_a) - \
                         score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.negative_a)
        score_margin_b = (score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.positive_a) -
                         score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.negative_a))

        # 分解margin
        d_norm = d_L.norm().item()
        h_shared_norm = h_shared.norm().item()
        u_diff_norm = u_diff.norm().item()

        # 消歧分量: cos(d_L, u_diff) × ||d_L|| × ||u_diff||
        cos_disamb = (torch.dot(d_L, u_diff) / (d_L.norm() * u_diff.norm() + 1e-10)).item()
        disamb_component = cos_disamb * d_norm * u_diff_norm

        # 共享基底分量: cos(h_shared, u_diff) × ||h_shared|| × ||u_diff||
        cos_shared = (torch.dot(h_shared, u_diff) / (h_shared.norm() * u_diff.norm() + 1e-10)).item()
        shared_component = cos_shared * h_shared_norm * u_diff_norm

        # 实际margin vs 分解预测
        raw_margin = raw_margin_a  # A版prompt的正确margin
        predicted_margin = disamb_component + shared_component
        pred_error = abs(raw_margin - predicted_margin) / (abs(raw_margin) + 1e-10) * 100

        # 消歧分量贡献比
        total_abs = abs(disamb_component) + abs(shared_component) + 1e-10
        disamb_pct = abs(disamb_component) / total_abs * 100
        shared_pct = abs(shared_component) / total_abs * 100

        # h_a对u_diff的对齐（A版的完整margin分解）
        cos_a_u_diff = (torch.dot(h_a, u_diff) / (h_a.norm() * u_diff.norm() + 1e-10)).item()
        a_component = cos_a_u_diff * h_a.norm().item() * u_diff_norm

        # 验证: h_a @ u_diff = d_L @ u_diff + h_shared @ u_diff
        # 因为 h_a = h_shared + d_L/2
        verify_sum = (disamb_component / 2) + shared_component
        verify_direct = torch.dot(h_a, u_diff).item()

        result = {
            "case": case.name,
            "d_norm": round(d_norm, 4),
            "h_shared_norm": round(h_shared_norm, 4),
            "u_diff_norm": round(u_diff_norm, 4),
            "cos_disamb": round(cos_disamb, 6),
            "cos_shared": round(cos_shared, 6),
            "disamb_component": round(disamb_component, 4),
            "shared_component": round(shared_component, 4),
            "disamb_pct": round(disamb_pct, 2),
            "shared_pct": round(shared_pct, 2),
            "raw_margin_a": round(raw_margin_a, 4),
            "raw_margin_b": round(raw_margin_b, 4),
            "score_margin_a": round(score_margin_a, 4),
            "predicted_margin": round(predicted_margin, 4),
            "pred_error_pct": round(pred_error, 2),
            "verify_sum": round(verify_sum, 4),
            "verify_direct": round(verify_direct, 4),
        }
        decomposition_results.append(result)

        print(f"    d_norm={d_norm:.2f}, cos_disamb={cos_disamb:.4f}, cos_shared={cos_shared:.4f}")
        print(f"    disamb_comp={disamb_component:.2f}({disamb_pct:.1f}%), shared_comp={shared_component:.2f}({shared_pct:.1f}%)")
        print(f"    raw_margin_A={raw_margin_a:.2f}, predicted={predicted_margin:.2f}, error={pred_error:.1f}%")
        print(f"    score_margin_A={score_margin_a:.4f}")

    # 判伪INV-280
    mean_disamb_pct = statistics.mean([r["disamb_pct"] for r in decomposition_results])
    inv280 = "CONFIRMED" if mean_disamb_pct > 10 else "FALSIFIED"
    print(f"\n  INV-280 (消歧分量>10%): mean_disamb_pct={mean_disamb_pct:.1f}%, {inv280}")

    # 判伪INV-281
    mean_pred_error = statistics.mean([r["pred_error_pct"] for r in decomposition_results])
    inv281 = "CONFIRMED" if mean_pred_error < 20 else "FALSIFIED"
    print(f"  INV-281 (分解误差<20%): mean_error={mean_pred_error:.1f}%, {inv281}")

    # ========== 实验2: 完整logit分布分析 ==========
    print("\n[Experiment 2] 完整logit分布分析...")

    logit_dist_results = {}
    for case in CASES[:3]:  # 取3个样本
        h_a = extract_last_hidden(model, tokenizer, case.prompt_a)
        logits = compute_logits_from_hidden(h_a, W_u)

        tid_correct = token_id_from_text(tokenizer, case.positive_a)
        tid_incorrect = token_id_from_text(tokenizer, case.negative_a)

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        top10_logits = sorted_logits[:10].tolist()
        top10_ids = sorted_indices[:10].tolist()
        top10_tokens = [tokenizer.decode([t]) for t in top10_ids]

        # 正确和错误token在logit分布中的位置
        rank_correct = (sorted_indices == tid_correct).nonzero()
        rank_incorrect = (sorted_indices == tid_incorrect).nonzero()
        rank_correct = rank_correct.item() if len(rank_correct) > 0 else -1
        rank_incorrect = rank_incorrect.item() if len(rank_incorrect) > 0 else -1

        # top-1和correct之间的logit差
        logit_correct = logits[tid_correct].item()
        logit_incorrect = logits[tid_incorrect].item()
        logit_gap = logit_correct - logit_incorrect

        # softmax后的概率
        probs = torch.softmax(logits, dim=-1)
        prob_correct = probs[tid_correct].item()
        prob_incorrect = probs[tid_incorrect].item()
        prob_ratio = prob_correct / (prob_incorrect + 1e-10)

        # softmax放大效应
        amplification = prob_ratio / (np.exp(logit_gap) + 1e-10)  # 应该≈1

        dist_info = {
            "case": case.name,
            "logit_correct": round(logit_correct, 4),
            "logit_incorrect": round(logit_incorrect, 4),
            "logit_gap": round(logit_gap, 4),
            "rank_correct": rank_correct,
            "rank_incorrect": rank_incorrect,
            "prob_correct": round(prob_correct, 6),
            "prob_incorrect": round(prob_incorrect, 6),
            "prob_ratio": round(prob_ratio, 4),
            "top5_tokens": [repr(t.strip()) + f"({l:.2f})" for t, l in zip(top10_tokens[:5], top10_logits[:5])],
        }
        logit_dist_results[case.name] = dist_info

        print(f"  {case.name}: logit_gap={logit_gap:.2f}, rank_correct={rank_correct}, "
              f"prob_correct={prob_correct:.6f}, prob_incorrect={prob_incorrect:.6f}, "
              f"prob_ratio={prob_ratio:.1f}x")
        print(f"    Top-5: {dist_info['top5_tokens']}")

    # ========== 实验3: cos_alignment的全面测量 ==========
    print("\n[Experiment 3] cos_alignment全面测量...")
    alignment_results = []

    for case in CASES:
        h_a = extract_last_hidden(model, tokenizer, case.prompt_a)
        h_b = extract_last_hidden(model, tokenizer, case.prompt_b)
        d_L = h_a - h_b

        tid_correct = token_id_from_text(tokenizer, case.positive_a)
        tid_incorrect = token_id_from_text(tokenizer, case.negative_a)
        u_diff = W_u[tid_correct] - W_u[tid_incorrect]

        cos_d_u = (torch.dot(d_L, u_diff) / (d_L.norm() * u_diff.norm() + 1e-10)).item()

        # d_L与所有unembed方向的最大cos
        W_u_normed = W_u / (W_u.norm(dim=1, keepdim=True) + 1e-10)
        d_normed = d_L / (d_L.norm() + 1e-10)
        all_cos = W_u_normed @ d_normed
        max_cos_idx = all_cos.argmax().item()
        max_cos = all_cos[max_cos_idx].item()
        max_cos_token = tokenizer.decode([max_cos_idx])

        # top-5 closest tokens
        top5_cos_vals, top5_cos_ids = all_cos.topk(5)
        top5_closest = [repr(tokenizer.decode([t]).strip()) + f"({c:.4f})" for t, c in zip(top5_cos_ids.tolist(), top5_cos_vals.tolist())]

        alignment_results.append({
            "case": case.name,
            "cos_d_u_diff": round(cos_d_u, 6),
            "max_cos_overall": round(max_cos, 6),
            "max_cos_token": max_cos_token.strip(),
            "top5_closest": top5_closest,
        })
        print(f"  {case.name}: cos(d,u_diff)={cos_d_u:.4f}, max_cos={max_cos:.4f}({repr(max_cos_token.strip())})")
        print(f"    Top-5 closest: {top5_closest}")

    mean_cos_align = statistics.mean([r["cos_d_u_diff"] for r in alignment_results])
    mean_max_cos = statistics.mean([r["max_cos_overall"] for r in alignment_results])
    print(f"  Mean cos(d,u_diff): {mean_cos_align:.4f}")
    print(f"  Mean max_cos(all u): {mean_max_cos:.4f}")

    # ========== 实验4: 消歧方向 vs 多token对的cos分布 ==========
    print("\n[Experiment 4] 消歧方向与随机token对的cos分布...")

    test_case = CASES[0]  # bank_financial
    h_a = extract_last_hidden(model, tokenizer, test_case.prompt_a)
    h_b = extract_last_hidden(model, tokenizer, test_case.prompt_b)
    d_L = h_a - h_b
    d_normed = d_L / (d_L.norm() + 1e-10)

    # 采样1000个随机token对
    np.random.seed(42)
    n_samples = 1000
    random_cos_values = []
    random_gap_values = []

    W_u_normed = W_u / (W_u.norm(dim=1, keepdim=True) + 1e-10)
    random_indices = np.random.choice(vocab_size, size=(n_samples, 2), replace=False)

    for i in range(n_samples):
        t1, t2 = random_indices[i]
        u_diff_r = W_u_normed[t1] - W_u_normed[t2]
        u_diff_r_norm = u_diff_r / (u_diff_r.norm() + 1e-10)
        cos_r = torch.dot(d_normed, u_diff_r_norm).item()
        random_cos_values.append(cos_r)
        # logit gap
        gap = torch.dot(h_a, W_u[t1]).item() - torch.dot(h_a, W_u[t2]).item()
        random_gap_values.append(gap)

    # 正确token对的cos在随机分布中的百分位
    tid_correct = token_id_from_text(tokenizer, test_case.positive_a)
    tid_incorrect = token_id_from_text(tokenizer, test_case.negative_a)
    u_diff_correct = W_u[tid_correct] - W_u[tid_incorrect]
    cos_correct = torch.dot(d_normed, u_diff_correct / (u_diff_correct.norm() + 1e-10)).item()

    percentile = np.mean(np.array(random_cos_values) < cos_correct) * 100

    print(f"  正确token对cos={cos_correct:.4f}, 在随机分布中的百分位={percentile:.1f}%")
    print(f"  随机cos分布: mean={statistics.mean(random_cos_values):.4f}, "
          f"std={statistics.stdev(random_cos_values):.4f}")
    print(f"  随机cos max={max(random_cos_values):.4f}, min={min(random_cos_values):.4f}")
    print(f"  随机logit_gap: mean={statistics.mean(random_gap_values):.2f}, std={statistics.stdev(random_gap_values):.2f}")

    # ========== 实验5: h_A的完整margin分解 ==========
    print("\n[Experiment 5] h_A完整margin的三分量分解...")

    triple_decomp_results = []
    for case in CASES:
        h_a = extract_last_hidden(model, tokenizer, case.prompt_a)
        tid_correct = token_id_from_text(tokenizer, case.positive_a)
        tid_incorrect = token_id_from_text(tokenizer, case.negative_a)

        # 完整margin = h_a @ (u_correct - u_incorrect)
        u_c = W_u[tid_correct]
        u_i = W_u[tid_incorrect]
        u_diff = u_c - u_i

        actual_margin = torch.dot(h_a, u_diff).item()

        # 将h_a分解为：embedding基底(第一层hidden) + 网络增量
        # 简化版：用h的norm和方向来分析
        h_norm = h_a.norm().item()
        cos_h_u_diff = (torch.dot(h_a, u_diff) / (h_a.norm() * u_diff.norm() + 1e-10)).item()

        # 用PCA分析u_diff方向在vocab空间中的特殊性
        # 计算 u_diff 的稀疏度（有多少维显著非零）
        u_diff_abs = u_diff.abs()
        u_diff_sorted = torch.sort(u_diff_abs, descending=True).values
        total_energy = (u_diff ** 2).sum().item()
        cum_energy = torch.cumsum(u_diff_sorted ** 2, dim=0) / total_energy
        r90_u_diff = (cum_energy >= 0.9).nonzero()
        r90_u_diff = r90_u_diff[0].item() + 1 if len(r90_u_diff) > 0 else hidden_dim

        triple_decomp_results.append({
            "case": case.name,
            "actual_margin": round(actual_margin, 4),
            "h_norm": round(h_norm, 2),
            "cos_h_u_diff": round(cos_h_u_diff, 6),
            "u_diff_r90": r90_u_diff,
        })
        print(f"  {case.name}: margin={actual_margin:.2f}, h_norm={h_norm:.1f}, "
              f"cos(h,u_diff)={cos_h_u_diff:.4f}, u_diff_r90={r90_u_diff}")

    # ========== 保存 ==========
    results = {
        "model": model_key,
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "experiment1_decomposition": {
            "per_case": decomposition_results,
            "inv280_status": inv280,
            "mean_disamb_pct": round(mean_disamb_pct, 2),
            "inv281_status": inv281,
            "mean_pred_error": round(mean_pred_error, 2),
        },
        "experiment2_logit_dist": logit_dist_results,
        "experiment3_alignment": {
            "per_case": alignment_results,
            "mean_cos_d_u_diff": round(mean_cos_align, 6),
            "mean_max_cos": round(mean_max_cos, 6),
        },
        "experiment4_random_distribution": {
            "case": test_case.name,
            "cos_correct": round(cos_correct, 6),
            "percentile": round(percentile, 2),
            "random_cos_mean": round(statistics.mean(random_cos_values), 6),
            "random_cos_std": round(statistics.stdev(random_cos_values), 6),
            "random_cos_max": round(max(random_cos_values), 6),
        },
        "experiment5_triple_decomp": triple_decomp_results,
    }

    out_dir = OUTPUT_DIR / f"stage662_logit_decomposition_{model_key}_{TIMESTAMP}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"results_{model_key}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_dir}")
    free_model(model)
    return results


if __name__ == "__main__":
    model_key = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_experiment(model_key)
