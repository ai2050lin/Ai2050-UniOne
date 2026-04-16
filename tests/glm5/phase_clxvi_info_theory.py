#!/usr/bin/env python3
"""
Phase CLXVI: 信息论编码效率
============================

核心假说: 低频词用W_U低频段(主成分)编码是信息论最优的。
CLXV发现低频词的A贡献集中在低频段(r=-0.60), 但这只是描述性的。

Phase CLXVI目标: 用信息论框架证明频谱分工的最优性
  P719: 信息效率指标 — 量化每个词在各频段的编码效率
  P720: 信道容量分析 — 不同频段对词频的编码容量
  P721: 最优性验证 — 低频段编码是否是信息论最优的

信息论框架:
  编码效率 = I(h_band; word) / H(word)
    I(h_band; word) = 隐藏状态某频段与词语的互信息
    H(word) = 词语的信息熵(与词频负相关: 低频词→高熵)
  
  信道容量 = max_{p(x)} I(X; Y)
    对每个频段, 计算它能传递的最大信息量
  
  最优性: 如果低频词在低频段的I(h_band; word)/H(word)最高,
    则低频段编码低频词是信息论最优的

实验模型: qwen3 -> deepseek7b -> glm4 (串行)
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

# ===== 词表定义 =====
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

APPROX_FREQ_RANK = {
    "hand": 200, "head": 250, "eye": 300, "foot": 400, "arm": 450,
    "mouth": 500, "ear": 600, "nose": 800, "leg": 550, "neck": 900,
    "black": 300, "white": 350, "red": 400, "blue": 450, "green": 500,
    "yellow": 700, "brown": 800, "pink": 900, "gray": 1000, "gold": 1100,
    "water": 250, "fire": 300, "tree": 350, "sun": 400, "wind": 450,
    "rain": 500, "snow": 600, "star": 500, "moon": 600, "rock": 700,
    "milk": 600, "meat": 650, "bread": 700, "egg": 750, "rice": 800,
    "salt": 850, "cake": 900, "soup": 950, "wine": 1000, "cheese": 1100,
    "dog": 500, "bird": 550, "fish": 500, "horse": 600, "cat": 550,
    "bear": 700, "lion": 800, "tiger": 900, "wolf": 1000, "fox": 1100,
    "car": 400, "bus": 600, "train": 650, "ship": 700, "boat": 750,
    "plane": 700, "bike": 800, "truck": 850, "taxi": 900, "rail": 1200,
    "house": 300, "door": 400, "wall": 450, "room": 350, "road": 450,
    "bridge": 600, "roof": 700, "hall": 650, "gate": 700, "path": 750,
    "apple": 800, "orange": 700, "lemon": 1000, "cherry": 1100, "grape": 1200,
    "peach": 1300, "mango": 1800, "melon": 1600, "banana": 900, "pear": 1400,
    "knife": 900, "hammer": 1200, "nail": 1000, "saw": 1300, "drill": 1500,
    "screw": 1600, "wrench": 2000, "pliers": 2500, "axe": 1800, "blade": 1700,
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

def spearman_corr(x, y):
    from scipy.stats import spearmanr
    r, p = spearmanr(x, y)
    return r, p

def pearson_corr(x, y):
    from scipy.stats import pearsonr
    r, p = pearsonr(x, y)
    return r, p


# ============================================================
# P719: 信息效率指标
# ============================================================
def p719_information_efficiency(model, tokenizer, device, model_info):
    """
    量化每个词在各频段的编码效率
    编码效率 = |logit贡献| / ||频段信号|| * sqrt(词的信息量)
    """
    print("\n" + "="*60)
    print("P719: 信息效率指标")
    print("="*60)
    
    W_U = get_W_U(model)
    layers = get_layers(model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # W_U的SVD基
    from scipy.sparse.linalg import svds
    n_components = 100
    W_U_T = W_U.T.astype(np.float32)
    k = min(n_components, min(W_U_T.shape) - 2)
    U_wu, s_wu, Vt_wu = svds(W_U_T, k=k)
    idx = np.argsort(s_wu)[::-1]
    s_wu = s_wu[idx]
    U_wu = U_wu[:, idx]
    
    n_bands = 5
    band_size = k // n_bands
    
    print(f"  W_U SVD: {k}分量, 前5奇异值: {s_wu[:5].tolist()}")
    print(f"  频段: {n_bands}x{band_size}分量")
    
    # 末端层
    li = n_layers - 1
    print(f"  分析层: {li}")
    
    word_efficiency = []
    
    for word in ALL_WORDS:
        word_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(word_ids) != 1:
            continue
        
        word_id = word_ids[0]
        W_U_word = W_U[word_id]
        freq_rank = APPROX_FREQ_RANK.get(word, 1500)
        
        # 信息量: 用-log(freq_norm)衡量, 低频词信息量大
        # freq_norm = freq_rank / max_freq, 信息量 = -log(freq_norm)
        info_content = np.log(freq_rank)  # 越罕见信息量越大
        
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
        except:
            continue
        
        # 投影到W_U SVD基
        G_proj = U_wu.T @ G_contrib  # [k]
        A_proj = U_wu.T @ A_contrib  # [k]
        
        # 各频段的编码效率
        band_efficiency = {}
        for b in range(n_bands):
            start = b * band_size
            end = (b + 1) * band_size if b < n_bands - 1 else k
            
            # 该频段的G和A分量
            G_band_vec = np.zeros(k)
            G_band_vec[start:end] = G_proj[start:end]
            G_band_recon = U_wu @ G_band_vec
            
            A_band_vec = np.zeros(k)
            A_band_vec[start:end] = A_proj[start:end]
            A_band_recon = U_wu @ A_band_vec
            
            # 该频段对logit的贡献
            G_band_logit = float(W_U_word @ G_band_recon)
            A_band_logit = float(W_U_word @ A_band_recon)
            
            # 该频段的信号能量
            G_band_energy = float(np.sum(G_proj[start:end] ** 2))
            A_band_energy = float(np.sum(A_proj[start:end] ** 2))
            
            # 编码效率 = |logit贡献| / sqrt(信号能量)
            # 衡量每单位信号能量能产生多少logit
            G_efficiency = abs(G_band_logit) / max(np.sqrt(G_band_energy), 1e-10)
            A_efficiency = abs(A_band_logit) / max(np.sqrt(A_band_energy), 1e-10)
            
            # 信息效率 = 编码效率 * sqrt(信息量)
            # 低频词(高信息量)在效率相同时, 信息效率更高
            G_info_eff = G_efficiency * np.sqrt(info_content)
            A_info_eff = A_efficiency * np.sqrt(info_content)
            
            band_efficiency[f"Band{b+1}"] = {
                "G_logit": G_band_logit,
                "A_logit": A_band_logit,
                "G_energy": G_band_energy,
                "A_energy": A_band_energy,
                "G_efficiency": float(G_efficiency),
                "A_efficiency": float(A_efficiency),
                "G_info_eff": float(G_info_eff),
                "A_info_eff": float(A_info_eff),
            }
        
        # 最优频段 = 编码效率最高的频段
        G_best_band = max(range(n_bands), key=lambda b: band_efficiency[f"Band{b+1}"]["G_efficiency"])
        A_best_band = max(range(n_bands), key=lambda b: band_efficiency[f"Band{b+1}"]["A_efficiency"])
        
        # 总效率(所有频段加总)
        total_G_eff = sum(band_efficiency[f"Band{b+1}"]["G_efficiency"] for b in range(n_bands))
        total_A_eff = sum(band_efficiency[f"Band{b+1}"]["A_efficiency"] for b in range(n_bands))
        
        # 低频段(Band1+2)的效率占比
        low_band_G_eff = band_efficiency["Band1"]["G_efficiency"] + band_efficiency["Band2"]["G_efficiency"]
        low_band_A_eff = band_efficiency["Band1"]["A_efficiency"] + band_efficiency["Band2"]["A_efficiency"]
        low_band_G_ratio = low_band_G_eff / max(total_G_eff, 1e-10)
        low_band_A_ratio = low_band_A_eff / max(total_A_eff, 1e-10)
        
        word_efficiency.append({
            "word": word,
            "category": WORD_TO_CATEGORY[word],
            "freq_rank": freq_rank,
            "log_freq": np.log(freq_rank),
            "info_content": float(info_content),
            "band_efficiency": band_efficiency,
            "G_best_band": G_best_band + 1,  # 1-indexed
            "A_best_band": A_best_band + 1,
            "total_G_eff": total_G_eff,
            "total_A_eff": total_A_eff,
            "low_band_G_ratio": float(low_band_G_ratio),
            "low_band_A_ratio": float(low_band_A_ratio),
        })
    
    print(f"  有效词数: {len(word_efficiency)}")
    
    # ===== 统计分析 =====
    print("\n--- 编码效率与词频的相关性 ---")
    
    log_freqs = np.array([wd["log_freq"] for wd in word_efficiency])
    
    # 各频段A效率与词频的相关性
    print(f"\n  A编码效率 vs log_freq (Spearman):")
    for b in range(n_bands):
        vals = np.array([wd["band_efficiency"][f"Band{b+1}"]["A_efficiency"] for wd in word_efficiency])
        r, p = spearman_corr(log_freqs, vals)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    Band{b+1}: r={r:+.4f}, p={p:.4f} {sig}")
    
    print(f"\n  G编码效率 vs log_freq (Spearman):")
    for b in range(n_bands):
        vals = np.array([wd["band_efficiency"][f"Band{b+1}"]["G_efficiency"] for wd in word_efficiency])
        r, p = spearman_corr(log_freqs, vals)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    Band{b+1}: r={r:+.4f}, p={p:.4f} {sig}")
    
    # 信息效率
    print(f"\n  A信息效率 vs log_freq (Spearman):")
    for b in range(n_bands):
        vals = np.array([wd["band_efficiency"][f"Band{b+1}"]["A_info_eff"] for wd in word_efficiency])
        r, p = spearman_corr(log_freqs, vals)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    Band{b+1}: r={r:+.4f}, p={p:.4f} {sig}")
    
    # 最优频段与词频的相关性
    G_best_bands = np.array([wd["G_best_band"] for wd in word_efficiency])
    A_best_bands = np.array([wd["A_best_band"] for wd in word_efficiency])
    
    print(f"\n  最优频段与词频:")
    r, p = spearman_corr(log_freqs, A_best_bands.astype(float))
    print(f"    A_best_band vs log_freq: r={r:+.4f}, p={p:.4f}")
    r, p = spearman_corr(log_freqs, G_best_bands.astype(float))
    print(f"    G_best_band vs log_freq: r={r:+.4f}, p={p:.4f}")
    
    # 低频段效率占比
    low_G_ratios = np.array([wd["low_band_G_ratio"] for wd in word_efficiency])
    low_A_ratios = np.array([wd["low_band_A_ratio"] for wd in word_efficiency])
    
    r, p = spearman_corr(log_freqs, low_A_ratios)
    print(f"    A低频段效率占比 vs log_freq: r={r:+.4f}, p={p:.4f}")
    r, p = spearman_corr(log_freqs, low_G_ratios)
    print(f"    G低频段效率占比 vs log_freq: r={r:+.4f}, p={p:.4f}")
    
    # ===== 按频率分组 =====
    print("\n--- 按频率分组的编码效率 ---")
    
    freq_tertiles = np.percentile(np.array([wd["freq_rank"] for wd in word_efficiency]), [33, 67])
    high_freq = [wd for wd in word_efficiency if wd["freq_rank"] <= freq_tertiles[0]]
    low_freq = [wd for wd in word_efficiency if wd["freq_rank"] > freq_tertiles[1]]
    
    print(f"  高频组(n={len(high_freq)}), 低频组(n={len(low_freq)})")
    
    for b in range(n_bands):
        high_A_eff = np.mean([wd["band_efficiency"][f"Band{b+1}"]["A_efficiency"] for wd in high_freq])
        low_A_eff = np.mean([wd["band_efficiency"][f"Band{b+1}"]["A_efficiency"] for wd in low_freq])
        high_G_eff = np.mean([wd["band_efficiency"][f"Band{b+1}"]["G_efficiency"] for wd in high_freq])
        low_G_eff = np.mean([wd["band_efficiency"][f"Band{b+1}"]["G_efficiency"] for wd in low_freq])
        
        print(f"    Band{b+1}: A_eff high={high_A_eff:.4f} low={low_A_eff:.4f} ratio={low_A_eff/max(high_A_eff,1e-10):.3f}, "
              f"G_eff high={high_G_eff:.4f} low={low_G_eff:.4f} ratio={low_G_eff/max(high_G_eff,1e-10):.3f}")
    
    # 最优频段分布
    print(f"\n  高频词A最优频段分布: Band1={sum(1 for wd in high_freq if wd['A_best_band']==1)}, "
          f"Band2={sum(1 for wd in high_freq if wd['A_best_band']==2)}, "
          f"Band3-5={sum(1 for wd in high_freq if wd['A_best_band']>=3)}")
    print(f"  低频词A最优频段分布: Band1={sum(1 for wd in low_freq if wd['A_best_band']==1)}, "
          f"Band2={sum(1 for wd in low_freq if wd['A_best_band']==2)}, "
          f"Band3-5={sum(1 for wd in low_freq if wd['A_best_band']>=3)}")
    
    return {
        "n_words": len(word_efficiency),
        "word_efficiency": word_efficiency,
    }


# ============================================================
# P720: 信道容量分析
# ============================================================
def p720_channel_capacity(model, tokenizer, device, model_info, p719_data=None):
    """
    分析不同频段的编码容量与词频的关系
    信道容量 ≈ log(1 + SNR), SNR = 信号功率/噪声功率
    对每个频段, 信号 = 该频段对目标词logit的贡献, 噪声 = 该频段对非目标词logit的贡献
    """
    print("\n" + "="*60)
    print("P720: 信道容量分析")
    print("="*60)
    
    W_U = get_W_U(model)
    layers = get_layers(model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # W_U的SVD基
    from scipy.sparse.linalg import svds
    n_components = 100
    W_U_T = W_U.T.astype(np.float32)
    k = min(n_components, min(W_U_T.shape) - 2)
    U_wu, s_wu, Vt_wu = svds(W_U_T, k=k)
    idx = np.argsort(s_wu)[::-1]
    s_wu = s_wu[idx]
    U_wu = U_wu[:, idx]
    
    n_bands = 5
    band_size = k // n_bands
    
    li = n_layers - 1
    print(f"  分析层: {li}")
    
    # 选择测试词
    test_words_data = []
    
    for word in ALL_WORDS:
        word_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(word_ids) != 1:
            continue
        
        word_id = word_ids[0]
        W_U_word = W_U[word_id]
        freq_rank = APPROX_FREQ_RANK.get(word, 1500)
        
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
        except:
            continue
        
        # 投影到W_U SVD基
        A_proj = U_wu.T @ A_contrib
        G_proj = U_wu.T @ G_contrib
        
        # 各频段的A贡献对目标词和非目标词的logit
        # 目标词logit = W_U_word @ A_band_recon (信号)
        # 非目标词logit = W_U[other] @ A_band_recon (噪声/干扰)
        
        # 简化: 用随机采样的100个其他词作为噪声估计
        np.random.seed(42)
        n_noise = 100
        noise_ids = np.random.choice(W_U.shape[0], size=n_noise, replace=False)
        W_U_noise = W_U[noise_ids]  # [n_noise, d_model]
        
        band_capacity = {}
        for b in range(n_bands):
            start = b * band_size
            end = (b + 1) * band_size if b < n_bands - 1 else k
            
            # A频段分量
            A_band_vec = np.zeros(k)
            A_band_vec[start:end] = A_proj[start:end]
            A_band_recon = U_wu @ A_band_vec
            
            # 信号功率
            signal_power = float((W_U_word @ A_band_recon) ** 2)
            
            # 噪声功率(对其他词的干扰)
            noise_logits = W_U_noise @ A_band_recon  # [n_noise]
            noise_power = float(np.mean(noise_logits ** 2))
            
            # SNR和信道容量
            snr = signal_power / max(noise_power, 1e-20)
            channel_cap = np.log2(1 + snr)
            
            # G的SNR
            G_band_vec = np.zeros(k)
            G_band_vec[start:end] = G_proj[start:end]
            G_band_recon = U_wu @ G_band_vec
            
            G_signal = float((W_U_word @ G_band_recon) ** 2)
            G_noise_logits = W_U_noise @ G_band_recon
            G_noise_power = float(np.mean(G_noise_logits ** 2))
            G_snr = G_signal / max(G_noise_power, 1e-20)
            G_cap = np.log2(1 + G_snr)
            
            band_capacity[f"Band{b+1}"] = {
                "A_signal": float(signal_power),
                "A_noise": float(noise_power),
                "A_snr": float(snr),
                "A_capacity": float(channel_cap),
                "G_signal": float(G_signal),
                "G_noise": float(G_noise_power),
                "G_snr": float(G_snr),
                "G_capacity": float(G_cap),
            }
        
        test_words_data.append({
            "word": word,
            "freq_rank": freq_rank,
            "log_freq": np.log(freq_rank),
            "band_capacity": band_capacity,
        })
    
    print(f"  有效词数: {len(test_words_data)}")
    
    # ===== 信道容量与词频的相关性 =====
    print("\n--- 信道容量与词频的相关性 ---")
    
    log_freqs = np.array([wd["log_freq"] for wd in test_words_data])
    
    print(f"\n  A信道容量 vs log_freq (Spearman):")
    for b in range(n_bands):
        vals = np.array([wd["band_capacity"][f"Band{b+1}"]["A_capacity"] for wd in test_words_data])
        r, p = spearman_corr(log_freqs, vals)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    Band{b+1}: r={r:+.4f}, p={p:.4f} {sig}")
    
    print(f"\n  G信道容量 vs log_freq (Spearman):")
    for b in range(n_bands):
        vals = np.array([wd["band_capacity"][f"Band{b+1}"]["G_capacity"] for wd in test_words_data])
        r, p = spearman_corr(log_freqs, vals)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    Band{b+1}: r={r:+.4f}, p={p:.4f} {sig}")
    
    # SNR与词频
    print(f"\n  A_SNR vs log_freq (Spearman):")
    for b in range(n_bands):
        vals = np.array([wd["band_capacity"][f"Band{b+1}"]["A_snr"] for wd in test_words_data])
        r, p = spearman_corr(log_freqs, vals)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    Band{b+1}: r={r:+.4f}, p={p:.4f} {sig}")
    
    # ===== 各频段的总信道容量 =====
    print("\n--- 各频段的总信道容量(所有词平均) ---")
    for b in range(n_bands):
        mean_A_cap = np.mean([wd["band_capacity"][f"Band{b+1}"]["A_capacity"] for wd in test_words_data])
        mean_G_cap = np.mean([wd["band_capacity"][f"Band{b+1}"]["G_capacity"] for wd in test_words_data])
        mean_A_snr = np.mean([wd["band_capacity"][f"Band{b+1}"]["A_snr"] for wd in test_words_data])
        mean_G_snr = np.mean([wd["band_capacity"][f"Band{b+1}"]["G_snr"] for wd in test_words_data])
        print(f"    Band{b+1}: A_cap={mean_A_cap:.2f} bits, G_cap={mean_G_cap:.2f} bits, "
              f"A_SNR={mean_A_snr:.2f}, G_SNR={mean_G_snr:.2f}")
    
    # ===== 高频词vs低频词的信道容量 =====
    print("\n--- 高频词vs低频词的信道容量 ---")
    freq_tertiles = np.percentile(np.array([wd["freq_rank"] for wd in test_words_data]), [33, 67])
    high_freq = [wd for wd in test_words_data if wd["freq_rank"] <= freq_tertiles[0]]
    low_freq = [wd for wd in test_words_data if wd["freq_rank"] > freq_tertiles[1]]
    
    print(f"  高频组(n={len(high_freq)}), 低频组(n={len(low_freq)})")
    for b in range(n_bands):
        high_A_cap = np.mean([wd["band_capacity"][f"Band{b+1}"]["A_capacity"] for wd in high_freq])
        low_A_cap = np.mean([wd["band_capacity"][f"Band{b+1}"]["A_capacity"] for wd in low_freq])
        high_G_cap = np.mean([wd["band_capacity"][f"Band{b+1}"]["G_capacity"] for wd in high_freq])
        low_G_cap = np.mean([wd["band_capacity"][f"Band{b+1}"]["G_capacity"] for wd in low_freq])
        print(f"    Band{b+1}: A_cap high={high_A_cap:.2f} low={low_A_cap:.2f} ratio={low_A_cap/max(high_A_cap,1e-10):.3f}, "
              f"G_cap high={high_G_cap:.2f} low={low_G_cap:.2f} ratio={low_G_cap/max(high_G_cap,1e-10):.3f}")
    
    return {
        "n_words": len(test_words_data),
        "channel_data": test_words_data,
    }


# ============================================================
# P721: 最优性验证
# ============================================================
def p721_optimality_verification(model, tokenizer, device, model_info):
    """
    验证低频词用低频段编码是否是信息论最优的
    方法: 对每个词, 在所有可能的频段分配中, 找到使总编码效率最大的分配
    如果低频词的最优分配确实是低频段, 则验证通过
    """
    print("\n" + "="*60)
    print("P721: 最优性验证")
    print("="*60)
    
    W_U = get_W_U(model)
    layers = get_layers(model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # W_U的SVD基
    from scipy.sparse.linalg import svds
    n_components = 100
    W_U_T = W_U.T.astype(np.float32)
    k = min(n_components, min(W_U_T.shape) - 2)
    U_wu, s_wu, Vt_wu = svds(W_U_T, k=k)
    idx = np.argsort(s_wu)[::-1]
    s_wu = s_wu[idx]
    U_wu = U_wu[:, idx]
    
    n_bands = 5
    band_size = k // n_bands
    
    li = n_layers - 1
    
    # 核心实验: 比较实际编码 vs 随机编码 vs 最优编码
    # 实际编码 = 模型当前的G和A
    # 随机编码 = 将G和A的频段贡献随机重排
    # 最优编码 = 将所有信号集中到效率最高的频段
    
    test_words = ["hand", "head", "water", "house",  # 高频
                  "knife", "hammer", "drill", "screw",  # 中频
                  "wrench", "pliers", "mango", "melon"]  # 低频
    
    optimality_results = []
    
    for word in test_words:
        word_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(word_ids) != 1:
            continue
        
        word_id = word_ids[0]
        W_U_word = W_U[word_id]
        freq_rank = APPROX_FREQ_RANK.get(word, 1500)
        
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
        except:
            continue
        
        # 实际A贡献的logit
        actual_A_logit = float(W_U_word @ A_contrib)
        actual_G_logit = float(W_U_word @ G_contrib)
        
        # 投影到W_U SVD基
        A_proj = U_wu.T @ A_contrib
        G_proj = U_wu.T @ G_contrib
        
        # 各频段的A_logit和效率
        band_A_logits = []
        band_A_energies = []
        band_A_efficiencies = []
        
        for b in range(n_bands):
            start = b * band_size
            end = (b + 1) * band_size if b < n_bands - 1 else k
            
            A_band_vec = np.zeros(k)
            A_band_vec[start:end] = A_proj[start:end]
            A_band_recon = U_wu @ A_band_vec
            
            band_logit = float(W_U_word @ A_band_recon)
            band_energy = float(np.sum(A_proj[start:end] ** 2))
            band_eff = abs(band_logit) / max(np.sqrt(band_energy), 1e-10)
            
            band_A_logits.append(band_logit)
            band_A_energies.append(band_energy)
            band_A_efficiencies.append(band_eff)
        
        # 最优频段(效率最高的)
        best_band = np.argmax(band_A_efficiencies)
        
        # 实际主导频段(能量最大的)
        dominant_band = np.argmax(band_A_energies)
        
        # 编码集中度: 前两个频段的能量占比
        sorted_energies = sorted(band_A_energies, reverse=True)
        concentration = (sorted_energies[0] + sorted_energies[1]) / max(sum(band_A_energies), 1e-10)
        
        # 重分配实验: 将所有A能量集中到最优频段
        total_A_energy = sum(band_A_energies)
        A_optimal_recon = U_wu[:, best_band*band_size:(best_band+1)*band_size] @ \
                          A_proj[best_band*band_size:(best_band+1)*band_size]
        # 将全部能量集中到最优频段
        A_all_in_best = np.zeros(k)
        A_all_in_best[best_band*band_size:(best_band+1)*band_size] = \
            A_proj[best_band*band_size:(best_band+1)*band_size] * \
            np.sqrt(total_A_energy / max(band_A_energies[best_band], 1e-10))
        A_optimal = U_wu @ A_all_in_best
        optimal_A_logit = float(W_U_word @ A_optimal)
        
        # 随机重排实验: 打乱频段贡献
        np.random.seed(42)
        A_shuffled_proj = A_proj.copy()
        np.random.shuffle(A_shuffled_proj)
        A_shuffled = U_wu @ A_shuffled_proj
        shuffled_A_logit = float(W_U_word @ A_shuffled)
        
        optimality_results.append({
            "word": word,
            "freq_rank": freq_rank,
            "log_freq": np.log(freq_rank),
            "actual_A_logit": actual_A_logit,
            "optimal_A_logit": optimal_A_logit,
            "shuffled_A_logit": shuffled_A_logit,
            "best_band": int(best_band) + 1,
            "dominant_band": int(dominant_band) + 1,
            "concentration": float(concentration),
            "band_efficiencies": [float(e) for e in band_A_efficiencies],
            "band_energies": [float(e) for e in band_A_energies],
        })
        
        print(f"\n  {word} (freq_rank={freq_rank}):")
        print(f"    实际A_logit={actual_A_logit:.2f}, 最优={optimal_A_logit:.2f}, 随机={shuffled_A_logit:.2f}")
        print(f"    最优频段=Band{best_band+1}, 主导频段=Band{dominant_band+1}")
        print(f"    频段效率: {[f'{e:.3f}' for e in band_A_efficiencies]}")
        print(f"    频段能量: {[f'{e:.2f}' for e in band_A_energies]}")
        print(f"    编码集中度: {concentration:.3f}")
    
    # ===== 最优性汇总 =====
    print("\n--- 最优性验证汇总 ---")
    
    # 最优频段与词频的关系
    log_freqs = np.array([r["log_freq"] for r in optimality_results])
    best_bands = np.array([r["best_band"] for r in optimality_results], dtype=float)
    dominant_bands = np.array([r["dominant_band"] for r in optimality_results], dtype=float)
    
    r, p = spearman_corr(log_freqs, best_bands)
    print(f"  最优频段 vs log_freq: r={r:+.4f}, p={p:.4f}")
    r, p = spearman_corr(log_freqs, dominant_bands)
    print(f"  主导频段 vs log_freq: r={r:+.4f}, p={p:.4f}")
    
    # 实际编码 vs 最优编码 vs 随机编码
    actual_logits = np.array([r["actual_A_logit"] for r in optimality_results])
    optimal_logits = np.array([r["optimal_A_logit"] for r in optimality_results])
    shuffled_logits = np.array([r["shuffled_A_logit"] for r in optimality_results])
    
    optimality_ratio = np.mean(np.abs(actual_logits)) / max(np.mean(np.abs(optimal_logits)), 1e-10)
    randomness_ratio = np.mean(np.abs(shuffled_logits)) / max(np.mean(np.abs(actual_logits)), 1e-10)
    
    print(f"  实际/最优logit比: {optimality_ratio:.3f}")
    print(f"  随机/实际logit比: {randomness_ratio:.3f}")
    
    # 低频词vs高频词的最优频段
    high = [r for r in optimality_results if r["freq_rank"] <= 500]
    low = [r for r in optimality_results if r["freq_rank"] > 1000]
    
    if high and low:
        high_best = np.mean([r["best_band"] for r in high])
        low_best = np.mean([r["best_band"] for r in low])
        high_dominant = np.mean([r["dominant_band"] for r in high])
        low_dominant = np.mean([r["dominant_band"] for r in low])
        
        print(f"\n  高频词最优频段: {high_best:.2f}, 主导频段: {high_dominant:.2f}")
        print(f"  低频词最优频段: {low_best:.2f}, 主导频段: {low_dominant:.2f}")
        
        if low_best < high_best:
            print(f"  -> 低频词的最优频段更低! 验证通过: 低频段编码低频词是信息论最优的")
        else:
            print(f"  -> 低频词的最优频段不比高频词低. 假说未通过验证")
    
    return {
        "optimality_results": optimality_results,
        "optimality_ratio": float(optimality_ratio),
        "randomness_ratio": float(randomness_ratio),
    }


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Phase CLXVI: 信息论编码效率")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["qwen3", "deepseek7b", "glm4"])
    args = parser.parse_args()
    
    model_name = args.model
    print(f"\n{'='*60}")
    print(f"Phase CLXVI: 信息论编码效率 - {model_name}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    
    print(f"\n模型信息:")
    print(f"  class: {model_info.model_class}")
    print(f"  n_layers: {model_info.n_layers}")
    print(f"  d_model: {model_info.d_model}")
    print(f"  vocab_size: {model_info.vocab_size}")
    print(f"  mlp_type: {model_info.mlp_type}")
    
    # P719
    p719_results = p719_information_efficiency(model, tokenizer, device, model_info)
    
    # P720
    p720_results = p720_channel_capacity(model, tokenizer, device, model_info, p719_data=p719_results)
    
    # P721
    p721_results = p721_optimality_verification(model, tokenizer, device, model_info)
    
    # 保存结果
    result_dir = f"d:/develop/TransformerLens-main/results/phase_clxvi"
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
        "p719": convert(p719_results),
        "p720": convert(p720_results),
        "p721": convert(p721_results),
    }
    
    # 移除大数组
    if "word_efficiency" in results["p719"]:
        del results["p719"]["word_efficiency"]
    if "channel_data" in results["p720"]:
        del results["p720"]["channel_data"]
    
    result_file = os.path.join(result_dir, f"{model_name}_results.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {result_file}")
    
    release_model(model)
    gc.collect()
    
    print(f"\n{'='*60}")
    print(f"Phase CLXVI ({model_name}) 完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
