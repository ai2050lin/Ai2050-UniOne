#!/usr/bin/env python3
"""
Phase CLXIV: 词频与编码特征的关系 — 验证"紫牛效应"
====================================================

核心假说: 人脑对罕见事物(紫牛)记忆深刻, 深度网络是否也有类似机制?
  - 低频词(罕见) -> 更大的A_logit贡献(需要更多"激活")
  - 高频词(常见) -> G项抑制更有效(训练样本多, 权重优化更好)
  - 注意力头可能充当"频率通道"——某些头专门检测罕见模式

实验设计:
  P713: 100词的词频统计与编码特征的相关性
    - 使用模型内部token频率(从训练语料的近似)衡量词频
    - 计算词频与G_logit, A_logit, GA_cos, diff_retention的相关性
    - 使用Spearman和Pearson相关系数

  P714: 低频词vs高频词的编码差异(验证"紫牛效应")
    - 将词按频率分为三组: 高频/中频/低频
    - 比较三组的编码特征差异
    - 核心验证: 低频词的|A_logit|是否显著高于高频词

  P715: 注意力头对低频/高频词的选择性(稀有信号检测头)
    - 分析每个注意力头对不同频率词的G_logit贡献
    - 寻找专门对低频词敏感的"稀有信号检测头"
    - 与CLVIII的头角色分类交叉验证

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
from model_utils import (
    load_model, get_layers, get_model_info,
    get_W_U, release_model, get_sample_layers
)

# ===== 词表定义 (与CLXIII相同) =====
WORD_CATEGORIES = {
    "fruit": ["apple", "banana", "orange", "grape", "mango", "peach", "cherry", "lemon", "pear", "melon"],
    "animal": ["cat", "dog", "bird", "fish", "bear", "lion", "tiger", "wolf", "fox", "horse"],
    "tool": ["knife", "hammer", "saw", "drill", "nail", "screw", "wrench", "pliers", "axe", "blade"],
    "body": ["hand", "foot", "head", "eye", "ear", "nose", "mouth", "arm", "leg", "neck"],
    "abstract": ["love", "hate", "fear", "hope", "truth", "justice", "beauty", "wisdom", "peace", "power"],
    "color": ["red", "blue", "green", "yellow", "black", "white", "pink", "brown", "gray", "gold"],
    "nature": ["sun", "moon", "star", "rain", "snow", "wind", "fire", "water", "rock", "tree"],
    "food": ["bread", "rice", "cake", "soup", "meat", "milk", "cheese", "egg", "salt", "wine"],
    "transport": ["car", "bus", "train", "ship", "boat", "plane", "bike", "truck", "taxi", "rail"],
    "building": ["house", "door", "wall", "roof", "room", "hall", "gate", "path", "road", "bridge"],
}

ALL_WORDS = []
WORD_TO_CATEGORY = {}
for cat, words in WORD_CATEGORIES.items():
    for w in words:
        ALL_WORDS.append(w)
        WORD_TO_CATEGORY[w] = cat

# ===== 近似词频表 (基于英语语料库的大致排名) =====
# 来源: 综合多个英语词频表 (BNC, SUBTLEX, Google Ngram)
# 值为大致的频率排名 (1=最常见), 越小越常见
APPROX_FREQ_RANK = {
    # body (极高频)
    "hand": 200, "head": 250, "eye": 300, "foot": 400, "arm": 450,
    "mouth": 500, "ear": 600, "nose": 800, "leg": 550, "neck": 900,
    # color (高频)
    "black": 300, "white": 350, "red": 400, "blue": 450, "green": 500,
    "yellow": 700, "brown": 800, "pink": 900, "gray": 1000, "gold": 1100,
    # nature (高频)
    "water": 250, "fire": 300, "tree": 350, "sun": 400, "wind": 450,
    "rain": 500, "snow": 600, "star": 500, "moon": 600, "rock": 700,
    # food (中高频)
    "milk": 600, "meat": 650, "bread": 700, "egg": 750, "rice": 800,
    "salt": 850, "cake": 900, "soup": 950, "wine": 1000, "cheese": 1100,
    # animal (中频)
    "dog": 500, "bird": 550, "fish": 500, "horse": 600, "cat": 550,
    "bear": 700, "lion": 800, "tiger": 900, "wolf": 1000, "fox": 1100,
    # transport (中频)
    "car": 400, "bus": 600, "train": 650, "ship": 700, "boat": 750,
    "plane": 700, "bike": 800, "truck": 850, "taxi": 900, "rail": 1200,
    # building (中频)
    "house": 300, "door": 400, "wall": 450, "room": 350, "road": 450,
    "bridge": 600, "roof": 700, "hall": 650, "gate": 700, "path": 750,
    # fruit (中低频)
    "apple": 800, "orange": 700, "lemon": 1000, "cherry": 1100, "grape": 1200,
    "peach": 1300, "mango": 1800, "melon": 1600, "banana": 900, "pear": 1400,
    # tool (低频)
    "knife": 900, "hammer": 1200, "nail": 1000, "saw": 1300, "drill": 1500,
    "screw": 1600, "wrench": 2000, "pliers": 2500, "axe": 1800, "blade": 1700,
    # abstract (中高频, 但概念抽象)
    "love": 300, "power": 350, "hope": 400, "peace": 500, "truth": 550,
    "fear": 500, "justice": 800, "wisdom": 1000, "beauty": 700, "hate": 800,
}


# ===== 通用工具函数 =====

def get_n_kv_heads(sa, model):
    if hasattr(sa, 'num_key_value_heads'):
        return sa.num_key_value_heads
    elif hasattr(sa, 'config') and hasattr(sa.config, 'num_key_value_heads'):
        return sa.config.num_key_value_heads
    elif hasattr(model, 'config') and hasattr(model.config, 'num_key_value_heads'):
        return model.config.num_key_value_heads
    else:
        return get_n_heads(sa, model)


def get_n_heads(sa, model):
    if hasattr(sa, 'num_heads'):
        return sa.num_heads
    elif hasattr(sa, 'config') and hasattr(sa.config, 'num_attention_heads'):
        return sa.config.num_attention_heads
    elif hasattr(model, 'config') and hasattr(model.config, 'num_attention_heads'):
        return model.config.num_attention_heads
    else:
        return 32


def get_head_dim(sa, n_q):
    return sa.o_proj.weight.shape[1] // n_q


def rms_norm(h, eps=1e-6):
    d = h.shape[-1]
    norm = np.sqrt(np.mean(h ** 2) + eps)
    return h * np.sqrt(d) / norm


def compute_A_contrib(sa, model, h_normed):
    n_kv = get_n_kv_heads(sa, model)
    n_q = get_n_heads(sa, model)
    head_dim = get_head_dim(sa, n_q)
    
    W_o = sa.o_proj.weight.detach().float().cpu().numpy()
    W_v = sa.v_proj.weight.detach().float().cpu().numpy()
    V_all = W_v @ h_normed
    
    n_groups = n_q // n_kv
    if n_groups > 1:
        V_expanded = np.zeros(n_q * head_dim)
        for kv_h in range(n_kv):
            for g in range(n_groups):
                q_h = kv_h * n_groups + g
                V_expanded[q_h * head_dim: (q_h + 1) * head_dim] = V_all[kv_h * head_dim: (kv_h + 1) * head_dim]
        return W_o @ V_expanded
    else:
        return W_o @ V_all


def compute_G_contrib(mlp, h_normed, mlp_type):
    W_down = mlp.down_proj.weight.detach().float().cpu().numpy()
    if mlp_type == "split_gate_up":
        W_gate = mlp.gate_proj.weight.detach().float().cpu().numpy()
        W_up = mlp.up_proj.weight.detach().float().cpu().numpy()
        gate_logits = np.clip(W_gate @ h_normed, -500, 500)
        gate_acts = 1.0 / (1.0 + np.exp(-gate_logits))
        up_vals = W_up @ h_normed
        return W_down @ (gate_acts * up_vals)
    else:
        W_gate_up = mlp.gate_up_proj.weight.detach().float().cpu().numpy()
        half = W_gate_up.shape[0] // 2
        W_gate = W_gate_up[:half]
        W_up = W_gate_up[half:]
        gate_logits = np.clip(W_gate @ h_normed, -500, 500)
        gate_acts = 1.0 / (1.0 + np.exp(-gate_logits))
        up_vals = W_up @ h_normed
        return W_down @ (gate_acts * up_vals)


def compute_head_level_contributions(sa, model, h_normed, W_U_word, n_q, head_dim):
    """
    计算每个注意力头对目标词logit的贡献
    返回: head_logits[i] = W_U_word @ (W_o[:, i*head_dim:(i+1)*head_dim] @ V[i*head_dim:(i+1)*head_dim])
    """
    n_kv = get_n_kv_heads(sa, model)
    W_o = sa.o_proj.weight.detach().float().cpu().numpy()  # [d_model, n_q*head_dim]
    W_v = sa.v_proj.weight.detach().float().cpu().numpy()  # [n_kv*head_dim, d_model]
    
    V_all = W_v @ h_normed  # [n_kv*head_dim]
    
    # GQA扩展
    n_groups = n_q // n_kv
    if n_groups > 1:
        V_expanded = np.zeros(n_q * head_dim)
        for kv_h in range(n_kv):
            for g in range(n_groups):
                q_h = kv_h * n_groups + g
                V_expanded[q_h * head_dim: (q_h + 1) * head_dim] = V_all[kv_h * head_dim: (kv_h + 1) * head_dim]
    else:
        V_expanded = V_all
    
    # 每个头的贡献
    head_logits = np.zeros(n_q)
    for h_idx in range(n_q):
        v_head = V_expanded[h_idx * head_dim: (h_idx + 1) * head_dim]
        w_o_head = W_o[:, h_idx * head_dim: (h_idx + 1) * head_dim]
        head_logits[h_idx] = float(W_U_word @ (w_o_head @ v_head))
    
    return head_logits


def spearman_corr(x, y):
    """Spearman rank correlation"""
    from scipy.stats import spearmanr
    r, p = spearmanr(x, y)
    return r, p


def pearson_corr(x, y):
    """Pearson correlation"""
    from scipy.stats import pearsonr
    r, p = pearsonr(x, y)
    return r, p


# ============================================================
# P713: 词频与编码特征的相关性
# ============================================================
def p713_frequency_correlation(model, tokenizer, device, model_info):
    """
    计算100个词的词频与编码特征的相关性
    """
    print("\n" + "="*60)
    print("P713: 词频与编码特征的相关性")
    print("="*60)
    
    W_U = get_W_U(model)
    layers = get_layers(model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # 采样层
    sample_layers = get_sample_layers(n_layers, 12)
    for li in range(max(0, n_layers - 3), n_layers):
        if li not in sample_layers:
            sample_layers.append(li)
    sample_layers = sorted(set(sample_layers))
    
    print(f"  采样层: {sample_layers}")
    
    word_data = []
    skipped = []
    
    for wi, word in enumerate(ALL_WORDS):
        # 检查token数量
        word_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(word_ids) != 1:
            skipped.append((word, len(word_ids)))
            continue
        
        word_id = word_ids[0]
        W_U_word = W_U[word_id]
        freq_rank = APPROX_FREQ_RANK.get(word, 1500)  # 默认中低频
        
        prompt = f"The {word} is"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # 提取每层的编码特征
        all_G_logit = []
        all_A_logit = []
        all_GA_cos = []
        all_diff_ret = []
        all_adversarial = []
        
        # 末端层特征
        end_G_logit = None
        end_A_logit = None
        end_GA_cos = None
        end_diff_ret = None
        
        for li in sample_layers:
            h_before = outputs.hidden_states[li][0, -1, :].float().detach().cpu().numpy()
            h_after = outputs.hidden_states[li + 1][0, -1, :].float().detach().cpu().numpy()
            
            try:
                layer = layers[li]
                sa = layer.self_attn
                mlp = layer.mlp
                
                h_normed = rms_norm(h_before)
                A_contrib = compute_A_contrib(sa, model, h_normed)
                G_contrib = compute_G_contrib(mlp, h_normed, model_info.mlp_type)
            except Exception as e:
                continue
            
            A_norm = np.linalg.norm(A_contrib)
            G_norm = np.linalg.norm(G_contrib)
            
            if A_norm < 1e-10 or G_norm < 1e-10:
                continue
            
            A_dir = A_contrib / A_norm
            G_dir = G_contrib / G_norm
            
            G_logit = float(W_U_word @ G_contrib)
            A_logit = float(W_U_word @ A_contrib)
            GA_cos = float(np.dot(G_dir, A_dir))
            
            # diff_retention
            h_raw = h_before + G_contrib + A_contrib
            diff_signal = G_contrib + A_contrib
            diff_norm_raw = np.linalg.norm(diff_signal)
            h_raw_norm = np.linalg.norm(h_raw)
            scale_factor = np.sqrt(d_model) / max(h_raw_norm, 1e-10)
            diff_ret = diff_norm_raw * scale_factor / max(diff_norm_raw, 1e-10)
            
            is_adv = (G_logit * A_logit < 0)
            
            all_G_logit.append(G_logit)
            all_A_logit.append(A_logit)
            all_GA_cos.append(GA_cos)
            all_diff_ret.append(float(diff_ret))
            all_adversarial.append(is_adv)
            
            # 末端层
            if li >= n_layers - 3:
                end_G_logit = G_logit
                end_A_logit = A_logit
                end_GA_cos = GA_cos
                end_diff_ret = float(diff_ret)
        
        if not all_G_logit:
            skipped.append((word, "no valid layers"))
            continue
        
        word_data.append({
            "word": word,
            "category": WORD_TO_CATEGORY[word],
            "freq_rank": freq_rank,
            "log_freq": np.log(freq_rank),  # 对数频率(更均匀)
            "word_id": word_id,
            # 全层统计
            "mean_G_logit": float(np.mean(all_G_logit)),
            "mean_A_logit": float(np.mean(all_A_logit)),
            "mean_GA_cos": float(np.mean(all_GA_cos)),
            "mean_diff_ret": float(np.mean(all_diff_ret)),
            "adversarial_rate": float(np.mean(all_adversarial)),
            # 末端层
            "end_G_logit": end_G_logit,
            "end_A_logit": end_A_logit,
            "end_GA_cos": end_GA_cos,
            "end_diff_ret": end_diff_ret,
            # 绝对值(衡量编码强度)
            "mean_abs_G_logit": float(np.mean(np.abs(all_G_logit))),
            "mean_abs_A_logit": float(np.mean(np.abs(all_A_logit))),
            "end_abs_G_logit": abs(end_G_logit) if end_G_logit is not None else None,
            "end_abs_A_logit": abs(end_A_logit) if end_A_logit is not None else None,
            # G-A比例
            "mean_GA_ratio": float(np.mean(np.abs(all_G_logit)) / max(np.mean(np.abs(all_A_logit)), 1e-10)),
        })
    
    print(f"  有效词: {len(word_data)}, 跳过: {len(skipped)}")
    for w, reason in skipped[:5]:
        print(f"    跳过: {w} ({reason})")
    
    # ===== 相关性分析 =====
    print("\n--- 词频与编码特征的相关性 ---")
    
    freq_ranks = np.array([wd["freq_rank"] for wd in word_data])
    log_freqs = np.array([wd["log_freq"] for wd in word_data])
    
    features = [
        ("mean_G_logit", "全层平均G_logit"),
        ("mean_A_logit", "全层平均A_logit"),
        ("mean_GA_cos", "全层平均GA_cos"),
        ("mean_diff_ret", "全层平均diff_retention"),
        ("adversarial_rate", "对抗率"),
        ("end_G_logit", "末端G_logit"),
        ("end_A_logit", "末端A_logit"),
        ("end_GA_cos", "末端GA_cos"),
        ("mean_abs_G_logit", "全层平均|G_logit|"),
        ("mean_abs_A_logit", "全层平均|A_logit|"),
        ("end_abs_G_logit", "末端|G_logit|"),
        ("end_abs_A_logit", "末端|A_logit|"),
        ("mean_GA_ratio", "G/A强度比"),
    ]
    
    correlations = {}
    print(f"\n  {'特征':<25s} {'Spearman_r':>12s} {'p_value':>12s} {'Pearson_r':>12s} {'p_value':>12s}")
    print("  " + "-"*75)
    
    for feat_name, feat_desc in features:
        vals = []
        for wd in word_data:
            v = wd.get(feat_name)
            if v is not None:
                vals.append(v)
            else:
                vals.append(np.nan)
        vals = np.array(vals)
        
        # 移除nan
        mask = ~np.isnan(vals)
        if mask.sum() < 5:
            continue
        
        sp_r, sp_p = spearman_corr(log_freqs[mask], vals[mask])
        pe_r, pe_p = pearson_corr(log_freqs[mask], vals[mask])
        
        correlations[feat_name] = {
            "spearman_r": float(sp_r), "spearman_p": float(sp_p),
            "pearson_r": float(pe_r), "pearson_p": float(pe_p),
            "n_samples": int(mask.sum()),
        }
        
        sig = "***" if sp_p < 0.001 else "**" if sp_p < 0.01 else "*" if sp_p < 0.05 else ""
        print(f"  {feat_desc:<25s} {sp_r:>12.4f} {sp_p:>12.6f} {pe_r:>12.4f} {pe_p:>12.6f} {sig}")
    
    # ===== 按频率分组统计 =====
    print("\n--- 按频率分组的编码特征 ---")
    
    # 三分位分组
    freq_tertiles = np.percentile(freq_ranks, [33, 67])
    high_freq = [wd for wd in word_data if wd["freq_rank"] <= freq_tertiles[0]]
    mid_freq = [wd for wd in word_data if freq_tertiles[0] < wd["freq_rank"] <= freq_tertiles[1]]
    low_freq = [wd for wd in word_data if wd["freq_rank"] > freq_tertiles[1]]
    
    print(f"  高频组(n={len(high_freq)}): rank <= {freq_tertiles[0]:.0f}")
    print(f"  中频组(n={len(mid_freq)}): rank {freq_tertiles[0]:.0f} - {freq_tertiles[1]:.0f}")
    print(f"  低频组(n={len(low_freq)}): rank > {freq_tertiles[1]:.0f}")
    
    group_stats = {}
    for group_name, group_data in [("high_freq", high_freq), ("mid_freq", mid_freq), ("low_freq", low_freq)]:
        if not group_data:
            continue
        
        stats = {}
        for feat_name in ["end_G_logit", "end_A_logit", "end_abs_G_logit", "end_abs_A_logit",
                          "mean_GA_cos", "mean_diff_ret", "adversarial_rate", "mean_GA_ratio"]:
            vals = [wd.get(feat_name) for wd in group_data if wd.get(feat_name) is not None]
            if vals:
                stats[feat_name] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "median": float(np.median(vals)),
                }
        
        group_stats[group_name] = stats
        
        print(f"\n  {group_name}:")
        for feat_name, feat_stats in stats.items():
            print(f"    {feat_name}: mean={feat_stats['mean']:.4f}, std={feat_stats['std']:.4f}")
    
    # ===== "紫牛效应"核心验证 =====
    print("\n--- 紫牛效应核心验证 ---")
    print("  假说: 低频词的|A_logit| > 高频词的|A_logit| (罕见词需要更多激活)")
    
    for feat_name, feat_desc in [("end_abs_A_logit", "末端|A_logit|"), 
                                  ("end_abs_G_logit", "末端|G_logit|"),
                                  ("mean_abs_A_logit", "全层|A_logit|"),
                                  ("mean_abs_G_logit", "全层|G_logit|")]:
        high_vals = [wd.get(feat_name) for wd in high_freq if wd.get(feat_name) is not None]
        low_vals = [wd.get(feat_name) for wd in low_freq if wd.get(feat_name) is not None]
        
        if high_vals and low_vals:
            high_mean = np.mean(high_vals)
            low_mean = np.mean(low_vals)
            ratio = low_mean / max(abs(high_mean), 1e-10)
            
            # t检验
            from scipy.stats import ttest_ind
            t_stat, t_p = ttest_ind(high_vals, low_vals)
            
            sig = "***" if t_p < 0.001 else "**" if t_p < 0.01 else "*" if t_p < 0.05 else "n.s."
            print(f"  {feat_desc}: 高频={high_mean:.2f}, 低频={low_mean:.2f}, "
                  f"低/高比={ratio:.3f}, t={t_stat:.3f}, p={t_p:.4f} {sig}")
    
    return {
        "n_valid_words": len(word_data),
        "n_skipped": len(skipped),
        "correlations": correlations,
        "group_stats": group_stats,
        "freq_tertiles": [float(freq_tertiles[0]), float(freq_tertiles[1])],
        "word_data": word_data,
    }


# ============================================================
# P714: 低频词vs高频词的编码差异
# ============================================================
def p714_frequency_groups(model, tokenizer, device, model_info, word_data_from_p713=None):
    """
    深入分析低频vs高频词的编码差异
    """
    print("\n" + "="*60)
    print("P714: 低频词vs高频词的编码差异(紫牛效应)")
    print("="*60)
    
    W_U = get_W_U(model)
    layers = get_layers(model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # 从P713结果中获取word_data (如果有的话)
    if word_data_from_p713:
        word_data = word_data_from_p713
    else:
        # 需要重新提取(这里简化,只分析关键特征)
        print("  警告: 没有P713数据,跳过P714")
        return {"status": "skipped"}
    
    # 分组
    freq_ranks = np.array([wd["freq_rank"] for wd in word_data])
    freq_tertiles = np.percentile(freq_ranks, [33, 67])
    
    high_freq = [wd for wd in word_data if wd["freq_rank"] <= freq_tertiles[0]]
    low_freq = [wd for wd in word_data if wd["freq_rank"] > freq_tertiles[1]]
    
    print(f"  高频组: {len(high_freq)}词")
    print(f"  低频组: {len(low_freq)}词")
    
    # ===== 逐层分析: 低频vs高频的G和A差异 =====
    print("\n--- 逐层分析 ---")
    
    # 采样层
    sample_layers = get_sample_layers(n_layers, 12)
    for li in range(max(0, n_layers - 3), n_layers):
        if li not in sample_layers:
            sample_layers.append(li)
    sample_layers = sorted(set(sample_layers))
    
    layer_diffs = []
    
    for li in sample_layers:
        # 对高频词和低频词分别计算该层的平均G_logit和A_logit
        high_G = []
        high_A = []
        low_G = []
        low_A = []
        
        for wd in high_freq:
            word = wd["word"]
            word_ids = tokenizer.encode(word, add_special_tokens=False)
            if len(word_ids) != 1:
                continue
            word_id = word_ids[0]
            W_U_word = W_U[word_id]
            
            prompt = f"The {word} is"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            h_before = outputs.hidden_states[li][0, -1, :].float().detach().cpu().numpy()
            
            try:
                layer = layers[li]
                sa = layer.self_attn
                mlp = layer.mlp
                h_normed = rms_norm(h_before)
                A_contrib = compute_A_contrib(sa, model, h_normed)
                G_contrib = compute_G_contrib(mlp, h_normed, model_info.mlp_type)
                
                high_G.append(float(W_U_word @ G_contrib))
                high_A.append(float(W_U_word @ A_contrib))
            except:
                continue
        
        for wd in low_freq:
            word = wd["word"]
            word_ids = tokenizer.encode(word, add_special_tokens=False)
            if len(word_ids) != 1:
                continue
            word_id = word_ids[0]
            W_U_word = W_U[word_id]
            
            prompt = f"The {word} is"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            h_before = outputs.hidden_states[li][0, -1, :].float().detach().cpu().numpy()
            
            try:
                layer = layers[li]
                sa = layer.self_attn
                mlp = layer.mlp
                h_normed = rms_norm(h_before)
                A_contrib = compute_A_contrib(sa, model, h_normed)
                G_contrib = compute_G_contrib(mlp, h_normed, model_info.mlp_type)
                
                low_G.append(float(W_U_word @ G_contrib))
                low_A.append(float(W_U_word @ A_contrib))
            except:
                continue
        
        if high_G and low_G:
            layer_diffs.append({
                "layer": li,
                "high_mean_G": float(np.mean(high_G)),
                "low_mean_G": float(np.mean(low_G)),
                "high_mean_A": float(np.mean(high_A)),
                "low_mean_A": float(np.mean(low_A)),
                "G_diff": float(np.mean(low_G) - np.mean(high_G)),
                "A_diff": float(np.mean(low_A) - np.mean(high_A)),
                "high_abs_G": float(np.mean(np.abs(high_G))),
                "low_abs_G": float(np.mean(np.abs(low_G))),
                "high_abs_A": float(np.mean(np.abs(high_A))),
                "low_abs_A": float(np.mean(np.abs(low_A))),
            })
            
            ld = layer_diffs[-1]
            print(f"  L{li:2d}: G_diff={ld['G_diff']:+.2f}, A_diff={ld['A_diff']:+.2f}, "
                  f"|G|: high={ld['high_abs_G']:.2f} low={ld['low_abs_G']:.2f}, "
                  f"|A|: high={ld['high_abs_A']:.2f} low={ld['low_abs_A']:.2f}")
    
    # ===== 统计检验 =====
    print("\n--- 统计检验 ---")
    
    # 综合所有层的结果
    all_G_diffs = [ld["G_diff"] for ld in layer_diffs]
    all_A_diffs = [ld["A_diff"] for ld in layer_diffs]
    all_absG_diffs = [ld["low_abs_G"] - ld["high_abs_G"] for ld in layer_diffs]
    all_absA_diffs = [ld["low_abs_A"] - ld["high_abs_A"] for ld in layer_diffs]
    
    from scipy.stats import ttest_1samp
    
    # H0: diff = 0 (低频=高频)
    for name, diffs in [("G_diff(低-高)", all_G_diffs), ("A_diff(低-高)", all_A_diffs),
                         ("|G|_diff(低-高)", all_absG_diffs), ("|A|_diff(低-高)", all_absA_diffs)]:
        if diffs:
            t_stat, t_p = ttest_1samp(diffs, 0)
            mean_diff = np.mean(diffs)
            sig = "***" if t_p < 0.001 else "**" if t_p < 0.01 else "*" if t_p < 0.05 else "n.s."
            print(f"  {name}: mean={mean_diff:+.4f}, t={t_stat:.3f}, p={t_p:.4f} {sig}")
    
    # ===== 语义类别内的频率效应 =====
    print("\n--- 语义类别内的频率效应 ---")
    
    for cat in WORD_CATEGORIES:
        cat_words = [wd for wd in word_data if wd["category"] == cat]
        if len(cat_words) < 3:
            continue
        
        cat_freqs = [wd["freq_rank"] for wd in cat_words]
        cat_median = np.median(cat_freqs)
        cat_high = [wd for wd in cat_words if wd["freq_rank"] <= cat_median]
        cat_low = [wd for wd in cat_words if wd["freq_rank"] > cat_median]
        
        if cat_high and cat_low:
            high_absA = np.mean([wd["end_abs_A_logit"] for wd in cat_high if wd.get("end_abs_A_logit") is not None])
            low_absA = np.mean([wd["end_abs_A_logit"] for wd in cat_low if wd.get("end_abs_A_logit") is not None])
            high_absG = np.mean([wd["end_abs_G_logit"] for wd in cat_high if wd.get("end_abs_G_logit") is not None])
            low_absG = np.mean([wd["end_abs_G_logit"] for wd in cat_low if wd.get("end_abs_G_logit") is not None])
            
            print(f"  {cat:12s}: |A| high={high_absA:.2f} low={low_absA:.2f} ratio={low_absA/max(high_absA,1e-10):.3f}, "
                  f"|G| high={high_absG:.2f} low={low_absG:.2f} ratio={low_absG/max(high_absG,1e-10):.3f}")
    
    return {
        "layer_diffs": layer_diffs,
        "mean_G_diff": float(np.mean(all_G_diffs)) if all_G_diffs else None,
        "mean_A_diff": float(np.mean(all_A_diffs)) if all_A_diffs else None,
        "mean_absG_diff": float(np.mean(all_absG_diffs)) if all_absG_diffs else None,
        "mean_absA_diff": float(np.mean(all_absA_diffs)) if all_absA_diffs else None,
    }


# ============================================================
# P715: 注意力头对低频/高频词的选择性
# ============================================================
def p715_head_selectivity(model, tokenizer, device, model_info):
    """
    分析注意力头对低频/高频词的选择性(稀有信号检测头)
    """
    print("\n" + "="*60)
    print("P715: 注意力头对低频/高频词的选择性")
    print("="*60)
    
    W_U = get_W_U(model)
    layers = get_layers(model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # 选择代表性词: 高频5 + 低频5
    high_words = ["hand", "head", "black", "water", "house"]  # 极高频
    low_words = ["pliers", "wrench", "mango", "melon", "axe"]  # 极低频
    test_words = high_words + low_words
    
    # 验证token
    valid_high = []
    valid_low = []
    for word in high_words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) == 1:
            valid_high.append(word)
    for word in low_words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) == 1:
            valid_low.append(word)
    
    print(f"  有效高频词: {valid_high}")
    print(f"  有效低频词: {valid_low}")
    
    # 只分析末端3层
    end_layers = list(range(max(0, n_layers - 3), n_layers))
    print(f"  分析层: {end_layers}")
    
    # 获取模型头数
    sa0 = layers[0].self_attn
    n_q = get_n_heads(sa0, model)
    n_kv = get_n_kv_heads(sa0, model)
    head_dim = get_head_dim(sa0, n_q)
    
    print(f"  n_q={n_q}, n_kv={n_kv}, head_dim={head_dim}")
    
    # 收集每个头对每个词的logit贡献
    head_contribs = {}  # {word: {layer: head_logits[n_q]}}
    
    for word in test_words:
        word_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(word_ids) != 1:
            continue
        
        word_id = word_ids[0]
        W_U_word = W_U[word_id]
        
        prompt = f"The {word} is"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        head_contribs[word] = {}
        
        for li in end_layers:
            h_before = outputs.hidden_states[li][0, -1, :].float().detach().cpu().numpy()
            
            try:
                layer = layers[li]
                sa = layer.self_attn
                h_normed = rms_norm(h_before)
                head_logits = compute_head_level_contributions(sa, model, h_normed, W_U_word, n_q, head_dim)
                head_contribs[word][li] = head_logits
            except Exception as e:
                continue
    
    # ===== 分析头的频率选择性 =====
    print("\n--- 头的频率选择性分析 ---")
    
    # 对每层,计算每个头对高频vs低频词的平均贡献
    head_selectivity_data = {}
    
    for li in end_layers:
        high_head_means = np.zeros(n_q)
        low_head_means = np.zeros(n_q)
        n_high = 0
        n_low = 0
        
        for word in valid_high:
            if word in head_contribs and li in head_contribs[word]:
                high_head_means += head_contribs[word][li]
                n_high += 1
        
        for word in valid_low:
            if word in head_contribs and li in head_contribs[word]:
                low_head_means += head_contribs[word][li]
                n_low += 1
        
        if n_high > 0:
            high_head_means /= n_high
        if n_low > 0:
            low_head_means /= n_low
        
        # 频率选择性 = 对低频词的贡献 - 对高频词的贡献
        freq_selectivity = low_head_means - high_head_means
        
        # 找出最有"稀有信号检测"特性的头
        # (对低频词贡献大, 对高频词贡献小)
        rare_detect_score = low_head_means - high_head_means  # 正=偏向低频
        common_detect_score = high_head_means - low_head_means  # 正=偏向高频
        
        head_selectivity_data[li] = {
            "high_means": high_head_means.tolist(),
            "low_means": low_head_means.tolist(),
            "freq_selectivity": freq_selectivity.tolist(),
        }
        
        # Top 5 稀有信号检测头
        rare_heads = np.argsort(rare_detect_score)[::-1][:5]
        common_heads = np.argsort(common_detect_score)[::-1][:5]
        
        print(f"\n  Layer {li}:")
        print(f"    稀有信号检测头(对低频词敏感):")
        for h in rare_heads:
            print(f"      Head {h}: low={low_head_means[h]:.3f}, high={high_head_means[h]:.3f}, "
                  f"selectivity={rare_detect_score[h]:+.3f}")
        
        print(f"    常见信号检测头(对高频词敏感):")
        for h in common_heads:
            print(f"      Head {h}: high={high_head_means[h]:.3f}, low={low_head_means[h]:.3f}, "
                  f"selectivity={common_detect_score[h]:+.3f}")
    
    # ===== 头角色的频率偏好 =====
    print("\n--- 头角色的频率偏好 ---")
    
    # 将头分为4类: 抑制头(G_logit<0), 激活头(A_logit>0), 低频偏好, 高频偏好
    # 这里简化: 统计对低频词贡献正的头比例
    for li in end_layers:
        if li not in head_selectivity_data:
            continue
        
        sel = head_selectivity_data[li]
        freq_sel = np.array(sel["freq_selectivity"])
        
        # 低频偏好头: freq_selectivity > 0
        rare_pref = np.sum(freq_sel > 0)
        common_pref = np.sum(freq_sel < 0)
        neutral = np.sum(freq_sel == 0)
        
        # 低频偏好头的贡献强度
        rare_pref_strength = np.mean(freq_sel[freq_sel > 0]) if rare_pref > 0 else 0
        common_pref_strength = np.mean(np.abs(freq_sel[freq_sel < 0])) if common_pref > 0 else 0
        
        print(f"  Layer {li}: 低频偏好头={rare_pref}/{n_q}, 高频偏好头={common_pref}/{n_q}, "
              f"低频偏好强度={rare_pref_strength:.4f}, 高频偏好强度={common_pref_strength:.4f}")
    
    # ===== 验证: 低频偏好头是否与"语义头"重合 =====
    print("\n--- 与语义头的重合分析 ---")
    
    # 语义头 = 对目标词logit贡献大的头(无论正负)
    for li in end_layers:
        if li not in head_selectivity_data:
            continue
        
        sel = head_selectivity_data[li]
        freq_sel = np.array(sel["freq_selectivity"])
        
        # 所有测试词的头贡献
        all_word_contribs = []
        for word in test_words:
            if word in head_contribs and li in head_contribs[word]:
                all_word_contribs.append(np.abs(head_contribs[word][li]))
        
        if all_word_contribs:
            mean_abs_contrib = np.mean(all_word_contribs, axis=0)  # 每个头的平均绝对贡献
            
            # Top 10 语义头(贡献最大的头)
            top_semantic = set(np.argsort(mean_abs_contrib)[::-1][:10])
            
            # 低频偏好头(top 10)
            top_rare_pref = set(np.argsort(freq_sel)[::-1][:10])
            
            # 高频偏好头(top 10)
            top_common_pref = set(np.argsort(-freq_sel)[::-1][:10])
            
            # 重合
            rare_semantic_overlap = len(top_rare_pref & top_semantic)
            common_semantic_overlap = len(top_common_pref & top_semantic)
            
            print(f"  Layer {li}: 低频偏好-语义头重合={rare_semantic_overlap}/10, "
                  f"高频偏好-语义头重合={common_semantic_overlap}/10")
    
    return {
        "head_selectivity_data": head_selectivity_data,
        "n_q": n_q,
        "n_kv": n_kv,
        "valid_high": valid_high,
        "valid_low": valid_low,
    }


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Phase CLXIV: 词频与编码特征")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["qwen3", "deepseek7b", "glm4"])
    args = parser.parse_args()
    
    model_name = args.model
    print(f"\n{'='*60}")
    print(f"Phase CLXIV: 词频与编码特征 - {model_name}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # 加载模型
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    
    print(f"\n模型信息:")
    print(f"  class: {model_info.model_class}")
    print(f"  n_layers: {model_info.n_layers}")
    print(f"  d_model: {model_info.d_model}")
    print(f"  vocab_size: {model_info.vocab_size}")
    print(f"  mlp_type: {model_info.mlp_type}")
    
    # P713
    p713_results = p713_frequency_correlation(model, tokenizer, device, model_info)
    
    # P714 (使用P713的word_data)
    p714_results = p714_frequency_groups(model, tokenizer, device, model_info, 
                                          word_data_from_p713=p713_results.get("word_data"))
    
    # P715
    p715_results = p715_head_selectivity(model, tokenizer, device, model_info)
    
    # 保存结果
    result_dir = f"d:/develop/TransformerLens-main/results/phase_clxiv"
    os.makedirs(result_dir, exist_ok=True)
    
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj
    
    results = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "p713": convert(p713_results),
        "p714": convert(p714_results),
        "p715": convert(p715_results),
    }
    
    # 移除大的word_data以节省空间(只保留统计)
    if "word_data" in results["p713"]:
        del results["p713"]["word_data"]
    
    result_file = os.path.join(result_dir, f"{model_name}_results.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {result_file}")
    
    # 释放模型
    release_model(model)
    gc.collect()
    
    print(f"\n{'='*60}")
    print(f"Phase CLXIV ({model_name}) 完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
