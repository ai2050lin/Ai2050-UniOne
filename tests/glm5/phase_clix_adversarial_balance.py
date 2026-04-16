#!/usr/bin/env python3
"""
Phase CLIX: 对抗平衡的数学结构 — "为什么要互相抵消?"
======================================================

核心目标: 理解CLVIII发现的关键现象——G和A大规模对抗平衡

CLVIII关键发现:
  Qwen3 apple: ΣG=-31.25, ΣA=+35.74, final=4.39 (8倍抵消!)
  DS7B apple:  ΣG=-280.5, ΣA=+289.8, final=9.30 (30倍抵消!)
  → G和A符号相反,绝对值远大于final,只留下微小残差

实验设计:
  P695: 对抗平衡的逐层动力学
    - 每层G和A的符号模式: 是固定"G负A正"还是逐层交替?
    - G和A的符号协方差: 如果G_l>0, A_l是否也>0?
    - 累积抵消分析: 哪些层的抵消最严重?

  P696: 对抗平衡的鲁棒性理论
    - 如果G或A被噪声扰动, final_logit如何变化?
    - 对抗平衡是否提供了"误差缓冲"? (鲁棒性优势)
    - 理论推导: 对抗平衡下的logit方差

  P697: 对抗平衡的词汇特异性
    - 不同词的G/A符号模式是否不同?
    - 水果/动物/动词/功能词的对抗平衡模式对比
    - G/A比例是否与词频/词性相关?

实验模型: qwen3 -> deepseek7b -> glm4 (串行, 避免OOM)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import json
import time
import argparse
import gc
from datetime import datetime
from pathlib import Path
from scipy import stats
from model_utils import load_model, get_model_info, get_layer_weights, release_model, MODEL_CONFIGS

import functools
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
print = functools.partial(print, flush=True)

OUT_DIR = Path("d:/develop/TransformerLens-main/results/phase_clix")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path("d:/develop/TransformerLens-main/tests/glm5_temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ============ 测试词汇 ============
TEST_WORDS = {
    "apple":    {"cat": "fruit", "pos": "noun",    "freq": "high"},
    "banana":   {"cat": "fruit", "pos": "noun",    "freq": "high"},
    "orange":   {"cat": "fruit", "pos": "noun",    "freq": "high"},
    "cat":      {"cat": "animal", "pos": "noun",   "freq": "high"},
    "dog":      {"cat": "animal", "pos": "noun",   "freq": "high"},
    "car":      {"cat": "vehicle", "pos": "noun",  "freq": "high"},
    "run":      {"cat": "action", "pos": "verb",   "freq": "high"},
    "red":      {"cat": "color", "pos": "adjective", "freq": "high"},
    "the":      {"cat": "function", "pos": "determiner", "freq": "very_high"},
}


def safe_sigmoid(x):
    """安全 sigmoid, 避免溢出"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def get_layers(model):
    """获取模型的 transformer 层"""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    return []


def compute_layer_GA(model, tokenizer, word, W_U, model_info, device):
    """
    计算指定词在所有层的G和A项的logit贡献
    
    返回:
      all_G_logit: [n_layers] — 每层G项对word logit的贡献
      all_A_logit: [n_layers] — 每层A项对word logit的贡献
      all_delta_logit: [n_layers] — 每层总贡献
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers = get_layers(model)
    
    word_ids = tokenizer.encode(word, add_special_tokens=False)
    if not word_ids:
        return None, None, None
    word_id = word_ids[0]
    W_U_word = W_U[word_id]
    
    text = f"The word is {word}."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    all_G_logit = np.zeros(n_layers)
    all_A_logit = np.zeros(n_layers)
    all_delta_logit = np.zeros(n_layers)
    
    for li in range(n_layers):
        h_before = outputs.hidden_states[li][0, -1, :].float().detach().cpu().numpy()
        h_after = outputs.hidden_states[li + 1][0, -1, :].float().detach().cpu().numpy()
        delta = h_after - h_before
        
        # 精确计算G
        lw = get_layer_weights(layers[li], d_model, mlp_type)
        W_gate = lw.W_gate
        W_up = lw.W_up
        W_down = lw.W_down
        
        if W_gate is not None and W_up is not None and W_down is not None:
            gate_val = safe_sigmoid(W_gate @ h_before)
            up_val = W_up @ h_before
            G_exact = W_down @ (gate_val * up_val)
            A_approx = delta - G_exact
            
            all_G_logit[li] = float(W_U_word @ G_exact)
            all_A_logit[li] = float(W_U_word @ A_approx)
        else:
            all_G_logit[li] = 0.0
            all_A_logit[li] = 0.0
        
        all_delta_logit[li] = float(W_U_word @ delta)
        
        del lw, W_gate, W_up, W_down
    
    del outputs
    gc.collect()
    
    return all_G_logit, all_A_logit, all_delta_logit


# ================================================================
# P695: 对抗平衡的逐层动力学
# ================================================================
def p695_adversarial_dynamics(model, tokenizer, model_info, W_embed, W_U, device):
    """
    P695: 对抗平衡的逐层动力学

    核心问题: 每层G和A的符号模式是什么?
    
    分析:
    1. 每层G和A的符号: G_sign(l), A_sign(l)
    2. 符号一致性: G和A是否总是符号相反?
    3. 符号交替模式: 是否存在"G正A负"→"G负A正"的交替?
    4. 累积抵消: cumsum(G+A)的轨迹是否单调?
    5. 抵消层 vs 增强层的统计
    """
    print("\n" + "="*70)
    print("P695: 对抗平衡的逐层动力学")
    print("="*70)
    
    n_layers = model_info.n_layers
    results = {}
    
    for word in TEST_WORDS:
        t0 = time.time()
        all_G, all_A, all_delta = compute_layer_GA(model, tokenizer, word, W_U, model_info, device)
        if all_G is None:
            continue
        
        # 1) 符号分析
        G_signs = np.sign(all_G)  # +1, 0, -1
        A_signs = np.sign(all_A)
        
        # G和A同号的层 vs 异号的层
        same_sign = np.sum(G_signs * A_signs > 0)   # 同号(增强)
        opposite_sign = np.sum(G_signs * A_signs < 0) # 异号(对抗)
        zero_layers = np.sum((G_signs == 0) | (A_signs == 0))
        
        # 2) 符号协方差: corr(G_sign, A_sign)
        nonzero_mask = (G_signs != 0) & (A_signs != 0)
        if np.sum(nonzero_mask) > 5:
            sign_corr = float(np.corrcoef(G_signs[nonzero_mask], A_signs[nonzero_mask])[0, 1])
        else:
            sign_corr = 0.0
        
        # 3) 累积轨迹
        cumG = np.cumsum(all_G)
        cumA = np.cumsum(all_A)
        cumDelta = np.cumsum(all_delta)
        
        # 4) 抵消层 vs 增强层
        # 抵消: |G+A| < max(|G|, |A|) — 即G和A部分抵消
        is_cancel = np.abs(all_delta) < np.maximum(np.abs(all_G), np.abs(all_A))
        n_cancel = int(np.sum(is_cancel))
        n_enhance = int(np.sum(~is_cancel))
        
        # 5) 抵消比: |G+A| / (|G|+|A|) — 0=完全抵消, 1=完全同向
        cancel_ratio = np.abs(all_delta) / np.maximum(np.abs(all_G) + np.abs(all_A), 1e-10)
        mean_cancel_ratio = float(np.mean(cancel_ratio))
        
        # 6) 符号交替频率
        G_sign_changes = np.sum(np.diff(G_signs) != 0)
        A_sign_changes = np.sum(np.diff(A_signs) != 0)
        
        # 7) 末端层模式
        last3_G = all_G[-3:]
        last3_A = all_A[-3:]
        last3_delta = all_delta[-3:]
        
        results[word] = {
            "word_info": TEST_WORDS[word],
            "n_layers": n_layers,
            "same_sign_layers": int(same_sign),
            "opposite_sign_layers": int(opposite_sign),
            "zero_layers": int(zero_layers),
            "sign_correlation": sign_corr,
            "cancel_layers": n_cancel,
            "enhance_layers": n_enhance,
            "cancel_enhance_ratio": float(n_cancel / max(n_enhance, 1)),
            "mean_cancel_ratio": mean_cancel_ratio,
            "G_sign_changes": int(G_sign_changes),
            "A_sign_changes": int(A_sign_changes),
            # 累积数据(关键层)
            "cumG_final": float(cumG[-1]),
            "cumA_final": float(cumA[-1]),
            "cumDelta_final": float(cumDelta[-1]),
            # 末端层详情
            "last3_G": [float(v) for v in last3_G],
            "last3_A": [float(v) for v in last3_A],
            "last3_delta": [float(v) for v in last3_delta],
            # 每层数据(精简: 每4层取一个+末端3层)
            "G_logit_sampled": {f"L{li}": float(all_G[li]) for li in list(range(0, n_layers, 4)) + list(range(max(0, n_layers-3), n_layers)) if li < n_layers},
            "A_logit_sampled": {f"L{li}": float(all_A[li]) for li in list(range(0, n_layers, 4)) + list(range(max(0, n_layers-3), n_layers)) if li < n_layers},
            "delta_logit_sampled": {f"L{li}": float(all_delta[li]) for li in list(range(0, n_layers, 4)) + list(range(max(0, n_layers-3), n_layers)) if li < n_layers},
        }
        
        elapsed = time.time() - t0
        print(f"  {word}: same_sign={same_sign}, opposite={opposite_sign}, "
              f"sign_corr={sign_corr:.3f}, cancel/enhance={n_cancel}/{n_enhance}, "
              f"mean_cancel_ratio={mean_cancel_ratio:.3f}, "
              f"cumG={cumG[-1]:.2f}, cumA={cumA[-1]:.2f}, "
              f"elapsed={elapsed:.1f}s")
    
    # 跨词比较
    if len(results) >= 3:
        # 按词性分组
        by_pos = {}
        for word, wr in results.items():
            pos = wr["word_info"]["pos"]
            if pos not in by_pos:
                by_pos[pos] = []
            by_pos[pos].append(wr["mean_cancel_ratio"])
        
        pos_cancel = {pos: float(np.mean(vals)) for pos, vals in by_pos.items()}
        
        # 按语义类别分组
        by_cat = {}
        for word, wr in results.items():
            cat = wr["word_info"]["cat"]
            if cat not in by_cat:
                by_cat[cat] = []
            by_cat[cat].append(wr["sign_correlation"])
        
        cat_sign = {cat: float(np.mean(vals)) for cat, vals in by_cat.items()}
        
        results["_cross_comparison"] = {
            "pos_cancel_ratio": pos_cancel,
            "cat_sign_correlation": cat_sign,
        }
    
    return results


# ================================================================
# P696: 对抗平衡的鲁棒性理论
# ================================================================
def p696_robustness_theory(model, tokenizer, model_info, W_embed, W_U, device):
    """
    P696: 对抗平衡的鲁棒性理论

    核心问题: 对抗平衡是否提供了"误差缓冲"?

    如果G和A大规模抵消,那么:
    - G的噪声会被A的噪声部分抵消? (鲁棒性优势)
    - 还是G和A的噪声叠加放大? (脆弱性劣势)

    实验:
    1. 对G项添加高斯噪声, 测量final_logit的变化
    2. 对A项添加高斯噪声, 测量final_logit的变化
    3. 对G+A同时添加噪声(但方向相反), 测量final_logit的变化
    4. 理论推导: 对抗平衡下的logit方差

    关键: 如果Var(final) < Var(G) + Var(A), 则对抗平衡提供鲁棒性
    """
    print("\n" + "="*70)
    print("P696: 对抗平衡的鲁棒性理论")
    print("="*70)
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers = get_layers(model)
    
    results = {}
    
    for word in ["apple", "banana", "cat", "run", "the"]:
        word_ids = tokenizer.encode(word, add_special_tokens=False)
        if not word_ids:
            continue
        word_id = word_ids[0]
        W_U_word = W_U[word_id]
        
        print(f"\n  --- {word} ---")
        t0 = time.time()
        
        # 前向传播获取基线
        text = f"The word is {word}."
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # 收集各层G和A
        all_G = []
        all_A = []
        all_h = []
        
        for li in range(n_layers):
            h_before = outputs.hidden_states[li][0, -1, :].float().detach().cpu().numpy()
            h_after = outputs.hidden_states[li + 1][0, -1, :].float().detach().cpu().numpy()
            delta = h_after - h_before
            
            lw = get_layer_weights(layers[li], d_model, mlp_type)
            if lw.W_gate is not None and lw.W_up is not None and lw.W_down is not None:
                gate_val = safe_sigmoid(lw.W_gate @ h_before)
                up_val = lw.W_up @ h_before
                G_exact = lw.W_down @ (gate_val * up_val)
                A_approx = delta - G_exact
            else:
                G_exact = np.zeros_like(delta)
                A_approx = delta.copy()
            
            all_G.append(G_exact)
            all_A.append(A_approx)
            all_h.append(h_before)
            
            del lw
        
        # 基线logit
        h_final = outputs.hidden_states[-1][0, -1, :].float().detach().cpu().numpy()
        baseline_logit = float(W_U_word @ h_final)
        
        # 噪声实验
        noise_levels = [0.01, 0.05, 0.1, 0.5]  # 相对于|G|或|A|的噪声比例
        n_trials = 50
        rng = np.random.RandomState(42)
        
        noise_results = {}
        
        for noise_level in noise_levels:
            logit_G_noise = []
            logit_A_noise = []
            logit_both_noise = []
            logit_anti_noise = []  # 反向噪声: G加正噪声, A加负噪声
            
            for trial in range(n_trials):
                # 构造带噪声的h_final
                # h_final_noisy = h_0 + Σ(G_l + noise_G_l + A_l + noise_A_l)
                # 简化: h_final_noisy = h_final + Σ(noise_G_l + noise_A_l)
                
                # G噪声: 每层G加噪声
                total_G_noise_logit = 0.0
                total_A_noise_logit = 0.0
                
                for li in range(n_layers):
                    G_norm = np.linalg.norm(all_G[li])
                    A_norm = np.linalg.norm(all_A[li])
                    
                    # G噪声: 与G同维的高斯噪声
                    noise_G = rng.randn(d_model) * noise_level * G_norm / max(np.sqrt(d_model), 1.0)
                    noise_A = rng.randn(d_model) * noise_level * A_norm / max(np.sqrt(d_model), 1.0)
                    
                    total_G_noise_logit += float(W_U_word @ noise_G)
                    total_A_noise_logit += float(W_U_word @ noise_A)
                
                # 只扰动G
                logit_G_noise.append(baseline_logit + total_G_noise_logit)
                # 只扰动A
                logit_A_noise.append(baseline_logit + total_A_noise_logit)
                # 同时扰动G和A
                logit_both_noise.append(baseline_logit + total_G_noise_logit + total_A_noise_logit)
                # 反向扰动: G加正噪声, A加负噪声(模拟G和A的噪声方向相反)
                logit_anti_noise.append(baseline_logit + total_G_noise_logit - total_A_noise_logit)
            
            noise_results[f"noise_{noise_level}"] = {
                "G_noise_std": float(np.std(logit_G_noise) / max(abs(baseline_logit), 1e-10)),
                "A_noise_std": float(np.std(logit_A_noise) / max(abs(baseline_logit), 1e-10)),
                "both_noise_std": float(np.std(logit_both_noise) / max(abs(baseline_logit), 1e-10)),
                "anti_noise_std": float(np.std(logit_anti_noise) / max(abs(baseline_logit), 1e-10)),
                "G_noise_abs_std": float(np.std(logit_G_noise)),
                "A_noise_abs_std": float(np.std(logit_A_noise)),
                "both_noise_abs_std": float(np.std(logit_both_noise)),
                "anti_noise_abs_std": float(np.std(logit_anti_noise)),
                # 关键: both_noise是否>G_noise+A_noise? (是否叠加放大)
                "amplification_ratio": float(np.std(logit_both_noise)) / max(
                    float(np.std(logit_G_noise)) + float(np.std(logit_A_noise)), 1e-10
                ),
            }
        
        # 理论推导: 对抗平衡下的logit方差
        # 如果 G_l 和 A_l 的logit贡献高度负相关:
        #   Var(ΣG + ΣA) = Var(ΣG) + Var(ΣA) + 2Cov(ΣG, ΣA)
        # 如果Cov(ΣG, ΣA) < 0 (对抗), 则Var(ΣG+ΣA) < Var(ΣG) + Var(ΣA)
        # → 对抗平衡提供方差缩减!
        
        G_logit_per_layer = np.array([float(W_U_word @ G) for G in all_G])
        A_logit_per_layer = np.array([float(W_U_word @ A) for A in all_A])
        
        var_G = float(np.var(G_logit_per_layer))
        var_A = float(np.var(A_logit_per_layer))
        cov_GA = float(np.cov(G_logit_per_layer, A_logit_per_layer)[0, 1])
        var_sum = float(np.var(G_logit_per_layer + A_logit_per_layer))
        
        # 方差缩减比
        variance_reduction = (var_G + var_A + 2*cov_GA) / max(var_G + var_A, 1e-10)
        
        results[word] = {
            "baseline_logit": baseline_logit,
            "noise_experiments": noise_results,
            "variance_analysis": {
                "var_G_per_layer": var_G,
                "var_A_per_layer": var_A,
                "cov_GA_per_layer": cov_GA,
                "var_sum_per_layer": var_sum,
                "variance_reduction_ratio": variance_reduction,
                "is_variance_reduced": variance_reduction < 1.0,
            },
        }
        
        elapsed = time.time() - t0
        var_red = "YES" if variance_reduction < 1.0 else "NO"
        print(f"  {word}: baseline={baseline_logit:.4f}, "
              f"var_reduction={variance_reduction:.4f} ({var_red}), "
              f"cov_GA={cov_GA:.2f}, "
              f"noise_0.1_both_std={noise_results['noise_0.1']['both_noise_abs_std']:.4f}, "
              f"elapsed={elapsed:.1f}s")
        
        del outputs, all_G, all_A
        gc.collect()
    
    return results


# ================================================================
# P697: 对抗平衡的词汇特异性
# ================================================================
def p697_lexical_specificity(model, tokenizer, model_info, W_embed, W_U, device):
    """
    P697: 对抗平衡的词汇特异性

    核心问题: 不同词的G/A符号模式是否不同?

    分析:
    1. 每个词的G/A比例(逐层)
    2. 水果/动物/动词/功能词的对抗平衡模式对比
    3. G/A比例与词频/词性的相关性
    4. 词语义的"签名": G_logit_profile 和 A_logit_profile 是否是词的指纹?
    
    关键: 如果G/A模式是词特异的, 则对抗平衡是编码的一部分
          如果G/A模式是通用的, 则对抗平衡是架构的固有性质
    """
    print("\n" + "="*70)
    print("P697: 对抗平衡的词汇特异性")
    print("="*70)
    
    n_layers = model_info.n_layers
    results = {}
    
    # 收集所有词的G和A profile
    all_word_G = {}
    all_word_A = {}
    all_word_delta = {}
    
    for word in TEST_WORDS:
        t0 = time.time()
        G_logit, A_logit, delta_logit = compute_layer_GA(model, tokenizer, word, W_U, model_info, device)
        if G_logit is None:
            continue
        
        all_word_G[word] = G_logit
        all_word_A[word] = A_logit
        all_word_delta[word] = delta_logit
        
        # 计算G/A比例
        GA_ratio_per_layer = np.abs(G_logit) / np.maximum(np.abs(A_logit), 1e-10)
        mean_GA_ratio = float(np.mean(GA_ratio_per_layer))
        
        # G和A的累积贡献
        cumG = np.cumsum(G_logit)
        cumA = np.cumsum(A_logit)
        
        # G和A的贡献集中度: 哪些层贡献了主要G/A
        G_top3_layers = np.argsort(np.abs(G_logit))[::-1][:3]
        A_top3_layers = np.argsort(np.abs(A_logit))[::-1][:3]
        
        # 词语的G/A "指纹": 逐层G/(G+A)的比例
        GA_frac = G_logit / np.maximum(np.abs(G_logit) + np.abs(A_logit), 1e-10)
        
        # G主导的层 vs A主导的层
        G_dominant = int(np.sum(np.abs(G_logit) > np.abs(A_logit)))
        A_dominant = int(np.sum(np.abs(A_logit) > np.abs(G_logit)))
        
        results[word] = {
            "word_info": TEST_WORDS[word],
            "mean_GA_ratio": mean_GA_ratio,
            "cumG_final": float(cumG[-1]),
            "cumA_final": float(cumA[-1]),
            "G_dominant_layers": G_dominant,
            "A_dominant_layers": A_dominant,
            "G_top3_layers": [int(l) for l in G_top3_layers],
            "A_top3_layers": [int(l) for l in A_top3_layers],
            # G/A指纹(采样层)
            "GA_frac_sampled": {f"L{li}": float(GA_frac[li]) 
                               for li in list(range(0, n_layers, 4)) + list(range(max(0, n_layers-3), n_layers)) 
                               if li < n_layers},
        }
        
        elapsed = time.time() - t0
        print(f"  {word}: mean_GA_ratio={mean_GA_ratio:.3f}, "
              f"G_dominant={G_dominant}, A_dominant={A_dominant}, "
              f"cumG={cumG[-1]:.2f}, cumA={cumA[-1]:.2f}, "
              f"elapsed={elapsed:.1f}s")
    
    # 跨词G/A指纹比较
    if len(all_word_G) >= 3:
        words_list = list(all_word_G.keys())
        n_words = len(words_list)
        
        # G profile的词间余弦相似度
        G_matrix = np.array([all_word_G[w] for w in words_list])  # [n_words, n_layers]
        A_matrix = np.array([all_word_A[w] for w in words_list])
        
        G_cos_matrix = np.zeros((n_words, n_words))
        A_cos_matrix = np.zeros((n_words, n_words))
        
        for i in range(n_words):
            for j in range(n_words):
                gi_norm = np.linalg.norm(G_matrix[i])
                gj_norm = np.linalg.norm(G_matrix[j])
                if gi_norm > 1e-10 and gj_norm > 1e-10:
                    G_cos_matrix[i, j] = float(np.dot(G_matrix[i], G_matrix[j]) / (gi_norm * gj_norm))
                
                ai_norm = np.linalg.norm(A_matrix[i])
                aj_norm = np.linalg.norm(A_matrix[j])
                if ai_norm > 1e-10 and aj_norm > 1e-10:
                    A_cos_matrix[i, j] = float(np.dot(A_matrix[i], A_matrix[j]) / (ai_norm * aj_norm))
        
        # 按语义类别分组计算组内/组间相似度
        fruit_words = [w for w in words_list if TEST_WORDS.get(w, {}).get("cat") == "fruit"]
        animal_words = [w for w in words_list if TEST_WORDS.get(w, {}).get("cat") == "animal"]
        
        fruit_idx = [words_list.index(w) for w in fruit_words]
        animal_idx = [words_list.index(w) for w in animal_words]
        
        # G profile: 水果组内 vs 水果-动物组间
        G_within_fruit = []
        G_between = []
        if len(fruit_idx) >= 2 and len(animal_idx) >= 1:
            for i in range(len(fruit_idx)):
                for j in range(i+1, len(fruit_idx)):
                    G_within_fruit.append(G_cos_matrix[fruit_idx[i], fruit_idx[j]])
            for fi in fruit_idx:
                for ai in animal_idx:
                    G_between.append(G_cos_matrix[fi, ai])
        
        A_within_fruit = []
        A_between = []
        if len(fruit_idx) >= 2 and len(animal_idx) >= 1:
            for i in range(len(fruit_idx)):
                for j in range(i+1, len(fruit_idx)):
                    A_within_fruit.append(A_cos_matrix[fruit_idx[i], fruit_idx[j]])
            for fi in fruit_idx:
                for ai in animal_idx:
                    A_between.append(A_cos_matrix[fi, ai])
        
        # G/A指纹的可区分性: 用PCA降维后,不同词是否分开?
        from sklearn.decomposition import PCA
        
        # 用G和A拼接作为指纹
        GA_fingerprint = np.concatenate([G_matrix, A_matrix], axis=1)  # [n_words, 2*n_layers]
        if n_words > 2:
            pca = PCA(n_components=min(n_words-1, 5))
            pca.fit(GA_fingerprint)
            pca_explained = pca.explained_variance_ratio_.tolist()
        else:
            pca_explained = [1.0]
        
        results["_cross_comparison"] = {
            "G_cosine_matrix": {
                "words": words_list,
                "matrix": [[float(G_cos_matrix[i, j]) for j in range(n_words)] for i in range(n_words)]
            },
            "A_cosine_matrix": {
                "words": words_list,
                "matrix": [[float(A_cos_matrix[i, j]) for j in range(n_words)] for i in range(n_words)]
            },
            "G_within_fruit_mean": float(np.mean(G_within_fruit)) if G_within_fruit else 0,
            "G_between_fruit_animal_mean": float(np.mean(G_between)) if G_between else 0,
            "A_within_fruit_mean": float(np.mean(A_within_fruit)) if A_within_fruit else 0,
            "A_between_fruit_animal_mean": float(np.mean(A_between)) if A_between else 0,
            "G_fingerprint_separation": float(np.mean(G_between) - np.mean(G_within_fruit)) if G_between and G_within_fruit else 0,
            "A_fingerprint_separation": float(np.mean(A_between) - np.mean(A_within_fruit)) if A_between and A_within_fruit else 0,
            "GA_fingerprint_PCA": pca_explained,
            "G_is_word_specific": float(np.mean(G_between) - np.mean(G_within_fruit)) > 0 if G_between and G_within_fruit else None,
            "A_is_word_specific": float(np.mean(A_between) - np.mean(A_within_fruit)) > 0 if A_between and A_within_fruit else None,
        }
    
    return results


# ================================================================
# 主函数
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()

    model_name = args.model
    print("="*70)
    print(f"Phase CLIX: 对抗平衡的数学结构 — {model_name}")
    print("="*70)

    # 加载模型
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)

    print(f"Model: {model_info.model_class}")
    print(f"Layers: {model_info.n_layers}, d_model: {model_info.d_model}, "
          f"d_mlp: {model_info.intermediate_size}, vocab: {model_info.vocab_size}")

    # 获取 W_embed 和 W_U
    W_embed = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    lm_head = model.get_output_embeddings()
    if lm_head is not None:
        W_U = lm_head.weight.detach().float().cpu().numpy()
    else:
        W_U = W_embed.copy()

    print(f"W_embed: {W_embed.shape}, W_U: {W_U.shape}")

    all_results = {
        "model": model_name,
        "model_info": {
            "class": model_info.model_class,
            "n_layers": model_info.n_layers,
            "d_model": model_info.d_model,
            "d_mlp": model_info.intermediate_size,
            "vocab_size": model_info.vocab_size,
            "mlp_type": model_info.mlp_type,
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # P695: 对抗平衡的逐层动力学
    all_results["P695"] = p695_adversarial_dynamics(model, tokenizer, model_info, W_embed, W_U, device)
    gc.collect()

    # P696: 对抗平衡的鲁棒性理论
    all_results["P696"] = p696_robustness_theory(model, tokenizer, model_info, W_embed, W_U, device)
    gc.collect()

    # P697: 对抗平衡的词汇特异性
    all_results["P697"] = p697_lexical_specificity(model, tokenizer, model_info, W_embed, W_U, device)
    gc.collect()

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUT_DIR / f"phase_clix_{model_name}_{timestamp}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {out_file}")

    # 打印关键发现摘要
    print("\n" + "="*70)
    print("关键发现摘要")
    print("="*70)
    print(f"\n模型: {model_name}")

    # P695 摘要
    if "P695" in all_results:
        print("\n--- P695: 逐层动力学 ---")
        for word in ["apple", "banana", "cat", "the"]:
            if word in all_results["P695"]:
                wr = all_results["P695"][word]
                print(f"  {word}: same_sign={wr.get('same_sign_layers',0)}, "
                      f"opposite={wr.get('opposite_sign_layers',0)}, "
                      f"sign_corr={wr.get('sign_correlation',0):.3f}, "
                      f"cancel/enhance={wr.get('cancel_layers',0)}/{wr.get('enhance_layers',0)}, "
                      f"mean_cancel_ratio={wr.get('mean_cancel_ratio',0):.3f}")

    # P696 摘要
    if "P696" in all_results:
        print("\n--- P696: 鲁棒性理论 ---")
        for word in ["apple", "banana", "cat", "the"]:
            if word in all_results["P696"]:
                wr = all_results["P696"][word]
                va = wr.get("variance_analysis", {})
                var_red = "YES" if va.get("is_variance_reduced", False) else "NO"
                print(f"  {word}: var_reduction={va.get('variance_reduction_ratio',0):.4f} ({var_red}), "
                      f"cov_GA={va.get('cov_GA_per_layer',0):.2f}")

    # P697 摘要
    if "P697" in all_results:
        print("\n--- P697: 词汇特异性 ---")
        for word in ["apple", "banana", "cat", "run", "the"]:
            if word in all_results["P697"]:
                wr = all_results["P697"][word]
                print(f"  {word}: mean_GA_ratio={wr.get('mean_GA_ratio',0):.3f}, "
                      f"G_dominant={wr.get('G_dominant_layers',0)}, A_dominant={wr.get('A_dominant_layers',0)}")
        
        if "_cross_comparison" in all_results["P697"]:
            cc = all_results["P697"]["_cross_comparison"]
            print(f"  G_fingerprint_separation={cc.get('G_fingerprint_separation',0):.3f}, "
                  f"A_fingerprint_separation={cc.get('A_fingerprint_separation',0):.3f}, "
                  f"G_is_word_specific={cc.get('G_is_word_specific')}, "
                  f"A_is_word_specific={cc.get('A_is_word_specific')}")

    # 释放模型
    release_model(model)
    print("[model_utils] GPU memory released")


if __name__ == "__main__":
    main()
