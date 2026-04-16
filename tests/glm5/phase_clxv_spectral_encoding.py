#!/usr/bin/env python3
"""
Phase CLXV: 频谱编码理论
=========================

核心假说: 深度网络的编码机制具有"频谱力学"特征——不同频率的信息
使用不同的编码通道, 类似于人脑对低频(常见)和高频(罕见)信息的差异化处理。

CLXIV关键发现:
  - diff_retention与词频显著正相关(r=0.36-0.51): 低频词的差分信号保留更高
  - 低频词的|A_logit|更大: "紫牛效应"
  - 注意力头有频率偏好但不专一

Phase CLXV目标: 将"频率"从统计相关提升到机制性理解
  - 注意力权重的频谱结构: 是否存在"频率通道"?
  - G和A的频段分布: 低频信息是否集中在特定频段?
  - RMSNorm的频谱选择效率: 对不同频段的信号选择效率是否不同?

频谱分析方法:
  1. 对注意力权重矩阵W_attn做SVD分解: W_attn = U S V^T
  2. SVD的奇异值对应"频率通道"的能量
  3. 低阶分量 = "低频通道"(平滑/全局特征)
  4. 高阶分量 = "高频通道"(细节/局部特征)
  5. 分析G和A在各频段的能量分布

实验设计:
  P716: 注意力权重的频谱分析
    - 对W_o, W_v, W_down做SVD
    - 分析奇异值谱的分布特征
    - 比较高频词vs低频词的频谱差异

  P717: G和A在不同频段的分布
    - 将G_contrib和A_contrib投影到W_U的SVD基上
    - 分析低频/高频分量的能量比
    - 比较低频词vs高频词的频段分布差异

  P718: RMSNorm对不同频段信号的选择效率
    - 将h_before分解为低频/高频分量
    - 分别计算RMSNorm对低频/高频分量的缩放因子
    - 验证RMSNorm是否选择性放大高频(罕见)信息

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

# 近似词频排名
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
# P716: 注意力权重的频谱分析
# ============================================================
def p716_weight_spectrum(model, tokenizer, device, model_info):
    """
    分析注意力权重矩阵的频谱(SVD)特征
    """
    print("\n" + "="*60)
    print("P716: 注意力权重的频谱分析")
    print("="*60)
    
    layers = get_layers(model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # 选择5个代表性层: 前1/4, 中间, 后1/4, 末端2层
    rep_layers = [
        n_layers // 4,
        n_layers // 2,
        3 * n_layers // 4,
        n_layers - 2,
        n_layers - 1,
    ]
    rep_layers = sorted(set(rep_layers))
    
    print(f"  代表性层: {rep_layers}")
    
    spectrum_data = {}
    
    for li in rep_layers:
        layer = layers[li]
        sa = layer.self_attn
        mlp = layer.mlp
        
        # W_o频谱
        W_o = sa.o_proj.weight.detach().float().cpu().numpy()  # [d_model, n_q*head_dim]
        # W_v频谱
        W_v = sa.v_proj.weight.detach().float().cpu().numpy()  # [n_kv*head_dim, d_model]
        # W_down频谱
        W_down = mlp.down_proj.weight.detach().float().cpu().numpy()  # [d_model, intermediate]
        
        # SVD分解 (取前50个分量)
        n_components = 50
        
        # W_o: [d_model, d_attn] -> 只分析较小的维度
        min_dim_o = min(W_o.shape)
        k_o = min(n_components, min_dim_o - 1)
        try:
            from scipy.sparse.linalg import svds
            U_o, s_o, Vt_o = svds(W_o, k=k_o)
            # 排序
            idx = np.argsort(s_o)[::-1]
            s_o = s_o[idx]
        except:
            s_o = np.zeros(k_o)
        
        # W_v: [d_kv, d_model]
        min_dim_v = min(W_v.shape)
        k_v = min(n_components, min_dim_v - 1)
        try:
            U_v, s_v, Vt_v = svds(W_v, k=k_v)
            idx = np.argsort(s_v)[::-1]
            s_v = s_v[idx]
        except:
            s_v = np.zeros(k_v)
        
        # W_down: [d_model, intermediate]
        min_dim_d = min(W_down.shape)
        k_d = min(n_components, min_dim_d - 1)
        try:
            U_d, s_d, Vt_d = svds(W_down, k=k_d)
            idx = np.argsort(s_d)[::-1]
            s_d = s_d[idx]
        except:
            s_d = np.zeros(k_d)
        
        # 频谱特征
        def spectrum_features(s_vals, name):
            total_energy = np.sum(s_vals ** 2)
            if total_energy < 1e-20:
                return {}
            
            # 低频能量比(前10%分量的能量占比)
            n_low = max(1, len(s_vals) // 10)
            low_energy = np.sum(s_vals[:n_low] ** 2) / total_energy
            
            # 高频能量比(后50%分量)
            n_high = len(s_vals) // 2
            high_energy = np.sum(s_vals[n_high:] ** 2) / total_energy
            
            # 频谱衰减率(前10个分量的指数衰减拟合)
            if len(s_vals) >= 10:
                log_s = np.log(s_vals[:10] + 1e-20)
                x = np.arange(10)
                try:
                    decay_rate = -np.polyfit(x, log_s, 1)[0]
                except:
                    decay_rate = 0
            else:
                decay_rate = 0
            
            # 频谱熵(能量分布的均匀性)
            p = s_vals ** 2 / total_energy
            p = p[p > 0]
            spectral_entropy = -np.sum(p * np.log(p)) / np.log(len(s_vals)) if len(s_vals) > 1 else 0
            
            # 中位数奇异值
            median_s = np.median(s_vals)
            
            return {
                f"{name}_total_energy": float(total_energy),
                f"{name}_low_freq_ratio": float(low_energy),
                f"{name}_high_freq_ratio": float(high_energy),
                f"{name}_decay_rate": float(decay_rate),
                f"{name}_spectral_entropy": float(spectral_entropy),
                f"{name}_median_sv": float(median_s),
                f"{name}_max_sv": float(s_vals[0]),
                f"{name}_condition_number": float(s_vals[0] / max(s_vals[-1], 1e-20)),
            }
        
        layer_spectrum = {}
        layer_spectrum.update(spectrum_features(s_o, "W_o"))
        layer_spectrum.update(spectrum_features(s_v, "W_v"))
        layer_spectrum.update(spectrum_features(s_d, "W_down"))
        
        # W_o的奇异值前10
        layer_spectrum["W_o_top10_sv"] = s_o[:10].tolist()
        layer_spectrum["W_v_top10_sv"] = s_v[:10].tolist()
        layer_spectrum["W_down_top10_sv"] = s_d[:10].tolist()
        
        spectrum_data[li] = layer_spectrum
        
        print(f"\n  Layer {li}:")
        for key in ["W_o_low_freq_ratio", "W_v_low_freq_ratio", "W_down_low_freq_ratio",
                     "W_o_high_freq_ratio", "W_v_high_freq_ratio", "W_down_high_freq_ratio",
                     "W_o_decay_rate", "W_v_decay_rate", "W_down_decay_rate",
                     "W_o_spectral_entropy", "W_v_spectral_entropy", "W_down_spectral_entropy"]:
            if key in layer_spectrum:
                print(f"    {key}: {layer_spectrum[key]:.4f}")
    
    # ===== 层间频谱演化 =====
    print("\n--- 层间频谱演化 ---")
    
    for weight_name in ["W_o", "W_v", "W_down"]:
        print(f"\n  {weight_name}:")
        for feat in ["low_freq_ratio", "high_freq_ratio", "decay_rate", "spectral_entropy"]:
            vals = []
            for li in rep_layers:
                key = f"{weight_name}_{feat}"
                if key in spectrum_data[li]:
                    vals.append((li, spectrum_data[li][key]))
            
            if vals:
                layers_list = [v[0] for v in vals]
                feat_vals = [v[1] for v in vals]
                trend = "increasing" if feat_vals[-1] > feat_vals[0] else "decreasing"
                print(f"    {feat}: {[f'{v:.4f}' for v in feat_vals]} ({trend})")
    
    # ===== 高频词vs低频词的W_U频谱分析 =====
    print("\n--- W_U频谱与词频的关系 ---")
    
    W_U = get_W_U(model)  # [vocab_size, d_model]
    
    # 对W_U做SVD (取前100个分量)
    n_wu_components = 100
    from scipy.sparse.linalg import svds
    W_U_T = W_U.T.astype(np.float32)  # [d_model, vocab_size]
    k_wu = min(n_wu_components, min(W_U_T.shape) - 2)
    U_wu, s_wu, Vt_wu = svds(W_U_T, k=k_wu)
    idx = np.argsort(s_wu)[::-1]
    s_wu = s_wu[idx]
    U_wu = U_wu[:, idx]
    
    print(f"  W_U SVD: top 10 singular values = {s_wu[:10].tolist()}")
    print(f"  W_U SVD: low_freq_ratio(前10%) = {np.sum(s_wu[:10]**2)/np.sum(s_wu**2):.4f}")
    print(f"  W_U SVD: high_freq_ratio(后50%) = {np.sum(s_wu[50:]**2)/np.sum(s_wu**2):.4f}")
    
    # 分析高频词和低频词在W_U各频段的投影
    high_words = ["hand", "head", "black", "water", "house"]
    low_words = ["pliers", "wrench", "mango", "melon", "axe"]
    
    high_freq_projections = []
    low_freq_projections = []
    
    for word in high_words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) != 1:
            continue
        word_vec = W_U[ids[0]]  # [d_model]
        # 投影到W_U的SVD基
        proj_coeffs = U_wu.T @ word_vec  # [k_wu]
        high_freq_projections.append(proj_coeffs)
    
    for word in low_words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) != 1:
            continue
        word_vec = W_U[ids[0]]
        proj_coeffs = U_wu.T @ word_vec
        low_freq_projections.append(proj_coeffs)
    
    if high_freq_projections and low_freq_projections:
        high_proj = np.array(high_freq_projections)  # [n_high, k_wu]
        low_proj = np.array(low_freq_projections)    # [n_low, k_wu]
        
        # 各频段的平均能量
        n_bands = 5
        band_size = k_wu // n_bands
        
        print(f"\n  各频段能量(高频词 vs 低频词):")
        for b in range(n_bands):
            start = b * band_size
            end = (b + 1) * band_size if b < n_bands - 1 else k_wu
            
            high_band_energy = np.mean(np.sum(high_proj[:, start:end] ** 2, axis=1))
            low_band_energy = np.mean(np.sum(low_proj[:, start:end] ** 2, axis=1))
            
            print(f"    Band {b+1} (components {start}-{end}): "
                  f"high={high_band_energy:.4f}, low={low_band_energy:.4f}, "
                  f"low/high={low_band_energy/max(high_band_energy, 1e-10):.3f}")
    
    return {
        "spectrum_data": spectrum_data,
        "rep_layers": rep_layers,
        "W_U_spectrum": {
            "top10_sv": s_wu[:10].tolist(),
            "low_freq_ratio": float(np.sum(s_wu[:10]**2) / np.sum(s_wu**2)),
            "high_freq_ratio": float(np.sum(s_wu[50:]**2) / np.sum(s_wu**2)),
        },
    }


# ============================================================
# P717: G和A在不同频段的分布
# ============================================================
def p717_GA_frequency_bands(model, tokenizer, device, model_info):
    """
    分析G和A贡献在W_U频段上的分布
    """
    print("\n" + "="*60)
    print("P717: G和A在不同频段的分布")
    print("="*60)
    
    W_U = get_W_U(model)
    layers = get_layers(model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # W_U的SVD基(预计算)
    from scipy.sparse.linalg import svds
    n_components = 100
    W_U_T = W_U.T.astype(np.float32)
    k = min(n_components, min(W_U_T.shape) - 2)
    U_wu, s_wu, Vt_wu = svds(W_U_T, k=k)
    idx = np.argsort(s_wu)[::-1]
    s_wu = s_wu[idx]
    U_wu = U_wu[:, idx]
    
    print(f"  W_U SVD基: {U_wu.shape}, 前10奇异值: {s_wu[:5].tolist()}")
    
    # 选择测试词: 高频5 + 低频5
    high_words = ["hand", "head", "black", "water", "house"]
    low_words = ["pliers", "wrench", "mango", "melon", "axe"]
    test_words = high_words + low_words
    
    # 频段定义
    n_bands = 5
    band_size = k // n_bands
    band_names = [f"Band{i+1}({i*band_size}-{(i+1)*band_size-1})" for i in range(n_bands)]
    
    # 末端层分析
    end_layers = list(range(max(0, n_layers - 3), n_layers))
    print(f"  分析层: {end_layers}")
    
    word_band_data = {}
    
    for word in test_words:
        word_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(word_ids) != 1:
            continue
        
        word_id = word_ids[0]
        W_U_word = W_U[word_id]
        is_low_freq = word in low_words
        
        prompt = f"The {word} is"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        word_band_data[word] = {
            "is_low_freq": is_low_freq,
            "freq_rank": APPROX_FREQ_RANK.get(word, 1500),
        }
        
        for li in end_layers:
            h_before = outputs.hidden_states[li][0, -1, :].float().detach().cpu().numpy()
            
            try:
                layer = layers[li]
                sa = layer.self_attn
                mlp = layer.mlp
                h_normed = rms_norm(h_before)
                A_contrib = compute_A_contrib(sa, model, h_normed)
                G_contrib = compute_G_contrib(mlp, h_normed, model_info.mlp_type)
            except Exception as e:
                continue
            
            # 将G和A投影到W_U的SVD基上
            G_proj = U_wu.T @ G_contrib  # [k]
            A_proj = U_wu.T @ A_contrib  # [k]
            
            # 各频段能量
            G_band_energy = []
            A_band_energy = []
            for b in range(n_bands):
                start = b * band_size
                end = (b + 1) * band_size if b < n_bands - 1 else k
                
                G_band_energy.append(float(np.sum(G_proj[start:end] ** 2)))
                A_band_energy.append(float(np.sum(A_proj[start:end] ** 2)))
            
            # 总能量
            G_total = np.sum(G_contrib ** 2)
            A_total = np.sum(A_contrib ** 2)
            
            # 各频段能量比
            G_band_ratio = [e / max(G_total, 1e-20) for e in G_band_energy]
            A_band_ratio = [e / max(A_total, 1e-20) for e in A_band_energy]
            
            # 低频(前2个band)vs高频(后2个band)能量比
            G_low_high_ratio = (G_band_energy[0] + G_band_energy[1]) / max(G_band_energy[-1] + G_band_energy[-2], 1e-20)
            A_low_high_ratio = (A_band_energy[0] + A_band_energy[1]) / max(A_band_energy[-1] + A_band_energy[-2], 1e-20)
            
            word_band_data[word][f"L{li}"] = {
                "G_band_energy": G_band_energy,
                "A_band_energy": A_band_energy,
                "G_band_ratio": G_band_ratio,
                "A_band_ratio": A_band_ratio,
                "G_total": float(G_total),
                "A_total": float(A_total),
                "G_low_high_ratio": float(G_low_high_ratio),
                "A_low_high_ratio": float(A_low_high_ratio),
            }
    
    # ===== 汇总分析 =====
    print("\n--- 高频词vs低频词的频段分布 ---")
    
    for li in end_layers:
        high_G_bands = []
        high_A_bands = []
        low_G_bands = []
        low_A_bands = []
        high_G_lh = []
        high_A_lh = []
        low_G_lh = []
        low_A_lh = []
        
        for word in high_words:
            key = f"L{li}"
            if word in word_band_data and key in word_band_data[word]:
                high_G_bands.append(word_band_data[word][key]["G_band_energy"])
                high_A_bands.append(word_band_data[word][key]["A_band_energy"])
                high_G_lh.append(word_band_data[word][key]["G_low_high_ratio"])
                high_A_lh.append(word_band_data[word][key]["A_low_high_ratio"])
        
        for word in low_words:
            key = f"L{li}"
            if word in word_band_data and key in word_band_data[word]:
                low_G_bands.append(word_band_data[word][key]["G_band_energy"])
                low_A_bands.append(word_band_data[word][key]["A_band_energy"])
                low_G_lh.append(word_band_data[word][key]["G_low_high_ratio"])
                low_A_lh.append(word_band_data[word][key]["A_low_high_ratio"])
        
        if not high_G_bands or not low_G_bands:
            continue
        
        high_G_mean = np.mean(high_G_bands, axis=0)
        low_G_mean = np.mean(low_G_bands, axis=0)
        high_A_mean = np.mean(high_A_bands, axis=0)
        low_A_mean = np.mean(low_A_bands, axis=0)
        
        print(f"\n  Layer {li}:")
        print(f"    G频段能量(高频词): {[f'{v:.2f}' for v in high_G_mean]}")
        print(f"    G频段能量(低频词): {[f'{v:.2f}' for v in low_G_mean]}")
        print(f"    A频段能量(高频词): {[f'{v:.2f}' for v in high_A_mean]}")
        print(f"    A频段能量(低频词): {[f'{v:.2f}' for v in low_A_mean]}")
        
        # 低频/高频比
        G_ratio = low_G_mean / np.maximum(high_G_mean, 1e-10)
        A_ratio = low_A_mean / np.maximum(high_A_mean, 1e-10)
        print(f"    G低频/高频比: {[f'{v:.2f}' for v in G_ratio]}")
        print(f"    A低频/高频比: {[f'{v:.2f}' for v in A_ratio]}")
        
        # 低高频比
        if high_G_lh and low_G_lh:
            print(f"    G低频段/高频段比: 高频词={np.mean(high_G_lh):.3f}, 低频词={np.mean(low_G_lh):.3f}")
            print(f"    A低频段/高频段比: 高频词={np.mean(high_A_lh):.3f}, 低频词={np.mean(low_A_lh):.3f}")
    
    # ===== 跨词频的相关性分析 =====
    print("\n--- 频段分布与词频的相关性 ---")
    
    # 收集所有有效词的数据
    all_word_band_features = []
    
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
        
        # 只分析最后一层
        li = n_layers - 1
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
        G_proj = U_wu.T @ G_contrib
        A_proj = U_wu.T @ A_contrib
        
        G_total = np.sum(G_contrib ** 2)
        A_total = np.sum(A_contrib ** 2)
        
        # 各频段能量比
        G_band_ratios = []
        A_band_ratios = []
        for b in range(n_bands):
            start = b * band_size
            end = (b + 1) * band_size if b < n_bands - 1 else k
            G_band_ratios.append(np.sum(G_proj[start:end] ** 2) / max(G_total, 1e-20))
            A_band_ratios.append(np.sum(A_proj[start:end] ** 2) / max(A_total, 1e-20))
        
        # 低频段/高频段比
        G_lh = (np.sum(G_proj[:2*band_size] ** 2)) / max(np.sum(G_proj[-2*band_size:] ** 2), 1e-20)
        A_lh = (np.sum(A_proj[:2*band_size] ** 2)) / max(np.sum(A_proj[-2*band_size:] ** 2), 1e-20)
        
        all_word_band_features.append({
            "word": word,
            "freq_rank": freq_rank,
            "log_freq": np.log(freq_rank),
            "G_band_ratios": G_band_ratios,
            "A_band_ratios": A_band_ratios,
            "G_low_high_ratio": float(G_lh),
            "A_low_high_ratio": float(A_lh),
        })
    
    print(f"  有效词数: {len(all_word_band_features)}")
    
    # 相关性
    log_freqs = np.array([wd["log_freq"] for wd in all_word_band_features])
    
    print(f"\n  频段能量比与词频的相关性(Spearman):")
    for b in range(n_bands):
        G_vals = np.array([wd["G_band_ratios"][b] for wd in all_word_band_features])
        A_vals = np.array([wd["A_band_ratios"][b] for wd in all_word_band_features])
        
        Gr, Gp = spearman_corr(log_freqs, G_vals)
        Ar, Ap = spearman_corr(log_freqs, A_vals)
        
        Gsig = "***" if Gp < 0.001 else "**" if Gp < 0.01 else "*" if Gp < 0.05 else ""
        Asig = "***" if Ap < 0.001 else "**" if Ap < 0.01 else "*" if Ap < 0.05 else ""
        
        print(f"    {band_names[b]}: G r={Gr:+.4f}{Gsig}, A r={Ar:+.4f}{Asig}")
    
    # 低频段/高频段比与词频的相关性
    G_lh_vals = np.array([wd["G_low_high_ratio"] for wd in all_word_band_features])
    A_lh_vals = np.array([wd["A_low_high_ratio"] for wd in all_word_band_features])
    
    Gr_lh, Gp_lh = spearman_corr(log_freqs, G_lh_vals)
    Ar_lh, Ap_lh = spearman_corr(log_freqs, A_lh_vals)
    
    print(f"\n  低频段/高频段比与词频:")
    print(f"    G: r={Gr_lh:+.4f}, p={Gp_lh:.4f}")
    print(f"    A: r={Ar_lh:+.4f}, p={Ap_lh:.4f}")
    
    return {
        "word_band_data": {k: v for k, v in word_band_data.items() if isinstance(v, dict)},
        "n_bands": n_bands,
        "band_names": band_names,
        "band_freq_correlations": {
            "G_low_high_vs_freq": {"r": float(Gr_lh), "p": float(Gp_lh)},
            "A_low_high_vs_freq": {"r": float(Ar_lh), "p": float(Ap_lh)},
        },
    }


# ============================================================
# P718: RMSNorm对不同频段信号的选择效率
# ============================================================
def p718_rmsnorm_spectral_selectivity(model, tokenizer, device, model_info):
    """
    分析RMSNorm对不同频段信号的选择效率
    """
    print("\n" + "="*60)
    print("P718: RMSNorm对不同频段信号的选择效率")
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
    
    # 频段
    n_bands = 5
    band_size = k // n_bands
    
    # 测试词
    test_words = ["hand", "head", "black", "water", "house",  # 高频
                  "pliers", "wrench", "mango", "melon", "axe"]  # 低频
    
    # 末端层
    end_layers = list(range(max(0, n_layers - 3), n_layers))
    print(f"  分析层: {end_layers}")
    
    # 核心实验: 分解h_before为低频/高频分量, 分别计算RMSNorm的缩放因子
    rmsnorm_selectivity = {}
    
    for word in test_words:
        word_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(word_ids) != 1:
            continue
        
        word_id = word_ids[0]
        W_U_word = W_U[word_id]
        is_low_freq = word in ["pliers", "wrench", "mango", "melon", "axe"]
        
        prompt = f"The {word} is"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        rmsnorm_selectivity[word] = {"is_low_freq": is_low_freq}
        
        for li in end_layers:
            h_before = outputs.hidden_states[li][0, -1, :].float().detach().cpu().numpy()
            h_after = outputs.hidden_states[li + 1][0, -1, :].float().detach().cpu().numpy()
            
            try:
                layer = layers[li]
                sa = layer.self_attn
                mlp = layer.mlp
                h_normed = rms_norm(h_before)
                A_contrib = compute_A_contrib(sa, model, h_normed)
                G_contrib = compute_G_contrib(mlp, h_normed, model_info.mlp_type)
            except:
                continue
            
            # 将h_before投影到W_U SVD基上
            h_proj = U_wu.T @ h_before  # [k]
            
            # 各频段的h_before分量
            h_band_norms = []
            for b in range(n_bands):
                start = b * band_size
                end = (b + 1) * band_size if b < n_bands - 1 else k
                band_vec = np.zeros(k)
                band_vec[start:end] = h_proj[start:end]
                h_band = U_wu @ band_vec  # 重建到d_model空间
                h_band_norms.append(np.linalg.norm(h_band))
            
            # h_before + G + A (RMSNorm前的原始信号)
            h_raw = h_before + G_contrib + A_contrib
            h_raw_proj = U_wu.T @ h_raw
            
            # RMSNorm缩放因子
            h_before_norm = np.linalg.norm(h_before)
            h_raw_norm = np.linalg.norm(h_raw)
            
            scale_before = np.sqrt(d_model) / max(h_before_norm, 1e-10)
            scale_after = np.sqrt(d_model) / max(h_raw_norm, 1e-10)
            
            # 各频段G和A的贡献
            G_proj = U_wu.T @ G_contrib
            A_proj = U_wu.T @ A_contrib
            
            G_band_energy = []
            A_band_energy = []
            for b in range(n_bands):
                start = b * band_size
                end = (b + 1) * band_size if b < n_bands - 1 else k
                G_band_energy.append(float(np.sum(G_proj[start:end] ** 2)))
                A_band_energy.append(float(np.sum(A_proj[start:end] ** 2)))
            
            # RMSNorm后各频段的缩放
            # h_after ≈ RMSNorm(h_raw) = h_raw * scale_after
            # 所以各频段在RMSNorm后的能量 ≈ 原始能量 * scale_after^2
            # 但这不完全正确, 因为RMSNorm是全局缩放, 不是逐频段缩放
            
            # 更精确的方法: 直接分析h_after的频段分布
            h_after_proj = U_wu.T @ h_after  # [k] (实际需要减去残差)
            # h_after = h_before + A + G (粗略)
            # 但实际有RMSNorm和非线性, 所以需要更仔细的分析
            
            # 简化: 计算RMSNorm对各频段的"有效增益"
            # 增益 = (h_after中该频段的能量) / (h_raw中该频段的能量)
            h_raw_band_energy = []
            h_after_band_energy_approx = []
            for b in range(n_bands):
                start = b * band_size
                end = (b + 1) * band_size if b < n_bands - 1 else k
                h_raw_band_energy.append(float(np.sum(h_raw_proj[start:end] ** 2)))
                # 近似: h_after ≈ scale_after * h_raw
                h_after_band_energy_approx.append(float(np.sum(h_raw_proj[start:end] ** 2)) * scale_after ** 2)
            
            # 各频段的有效增益
            band_gains = []
            for b in range(n_bands):
                raw_e = h_raw_band_energy[b]
                if raw_e > 1e-20:
                    gain = scale_after ** 2  # 全局增益
                    band_gains.append(float(gain))
                else:
                    band_gains.append(0.0)
            
            # 关键指标: 各频段G+A贡献对logit的影响
            # 分频段计算G和A对目标词logit的贡献
            G_band_logit = []
            A_band_logit = []
            for b in range(n_bands):
                start = b * band_size
                end = (b + 1) * band_size if b < n_bands - 1 else k
                
                # 重建该频段的G和A
                G_band_vec = np.zeros(k)
                G_band_vec[start:end] = G_proj[start:end]
                G_band_recon = U_wu @ G_band_vec
                
                A_band_vec = np.zeros(k)
                A_band_vec[start:end] = A_proj[start:end]
                A_band_recon = U_wu @ A_band_vec
                
                G_band_logit.append(float(W_U_word @ G_band_recon))
                A_band_logit.append(float(W_U_word @ A_band_recon))
            
            rmsnorm_selectivity[word][f"L{li}"] = {
                "scale_before": float(scale_before),
                "scale_after": float(scale_after),
                "scale_ratio": float(scale_after / max(scale_before, 1e-10)),
                "h_before_norm": float(h_before_norm),
                "h_raw_norm": float(h_raw_norm),
                "G_band_energy": G_band_energy,
                "A_band_energy": A_band_energy,
                "G_band_logit": G_band_logit,
                "A_band_logit": A_band_logit,
                "h_band_norms": h_band_norms,
            }
    
    # ===== 汇总分析 =====
    print("\n--- RMSNorm频谱选择效率 ---")
    
    high_words_list = ["hand", "head", "black", "water", "house"]
    low_words_list = ["pliers", "wrench", "mango", "melon", "axe"]
    
    for li in end_layers:
        print(f"\n  Layer {li}:")
        
        # 高频词和低频词的平均频段贡献
        high_G_band_logit = []
        high_A_band_logit = []
        low_G_band_logit = []
        low_A_band_logit = []
        high_scale_ratio = []
        low_scale_ratio = []
        
        for word in high_words_list:
            key = f"L{li}"
            if word in rmsnorm_selectivity and key in rmsnorm_selectivity[word]:
                d = rmsnorm_selectivity[word][key]
                high_G_band_logit.append(d["G_band_logit"])
                high_A_band_logit.append(d["A_band_logit"])
                high_scale_ratio.append(d["scale_ratio"])
        
        for word in low_words_list:
            key = f"L{li}"
            if word in rmsnorm_selectivity and key in rmsnorm_selectivity[word]:
                d = rmsnorm_selectivity[word][key]
                low_G_band_logit.append(d["G_band_logit"])
                low_A_band_logit.append(d["A_band_logit"])
                low_scale_ratio.append(d["scale_ratio"])
        
        if not high_G_band_logit or not low_G_band_logit:
            continue
        
        high_G_mean = np.mean(high_G_band_logit, axis=0)
        low_G_mean = np.mean(low_G_band_logit, axis=0)
        high_A_mean = np.mean(high_A_band_logit, axis=0)
        low_A_mean = np.mean(low_A_band_logit, axis=0)
        
        print(f"    RMSNorm缩放比: 高频词={np.mean(high_scale_ratio):.4f}, "
              f"低频词={np.mean(low_scale_ratio):.4f}")
        
        print(f"    G各频段logit(高频): {[f'{v:.2f}' for v in high_G_mean]}")
        print(f"    G各频段logit(低频): {[f'{v:.2f}' for v in low_G_mean]}")
        print(f"    A各频段logit(高频): {[f'{v:.2f}' for v in high_A_mean]}")
        print(f"    A各频段logit(低频): {[f'{v:.2f}' for v in low_A_mean]}")
        
        # 低频/高频在各频段的比例
        G_band_ratio = low_G_mean / np.maximum(np.abs(high_G_mean), 1e-10)
        A_band_ratio = low_A_mean / np.maximum(np.abs(high_A_mean), 1e-10)
        print(f"    G低/高频段比: {[f'{v:.2f}' for v in G_band_ratio]}")
        print(f"    A低/高频段比: {[f'{v:.2f}' for v in A_band_ratio]}")
    
    # ===== 频谱选择的统一分析 =====
    print("\n--- 频谱选择统一分析 ---")
    print("  核心问题: RMSNorm是否对不同频段的信号有不同的选择效率?")
    print("  如果RMSNorm是全局缩放(对所有频段相同的scale), 则频谱选择效率相同")
    print("  但如果G+A改变了信号的频谱分布, 则RMSNorm间接实现了频谱选择")
    
    # 计算所有词在最后一层的频段特征
    all_band_features = []
    
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
        
        li = n_layers - 1
        h_before = outputs.hidden_states[li][0, -1, :].float().detach().cpu().numpy()
        h_after = outputs.hidden_states[li + 1][0, -1, :].float().detach().cpu().numpy()
        
        try:
            layer = layers[li]
            sa = layer.self_attn
            mlp = layer.mlp
            h_normed = rms_norm(h_before)
            A_contrib = compute_A_contrib(sa, model, h_normed)
            G_contrib = compute_G_contrib(mlp, h_normed, model_info.mlp_type)
        except:
            continue
        
        # 各频段对目标词logit的贡献
        G_proj = U_wu.T @ G_contrib
        A_proj = U_wu.T @ A_contrib
        
        # 低频段(Band1+2) vs 高频段(Band4+5)的logit贡献
        G_low_logit = 0
        G_high_logit = 0
        A_low_logit = 0
        A_high_logit = 0
        
        for b in range(n_bands):
            start = b * band_size
            end = (b + 1) * band_size if b < n_bands - 1 else k
            
            band_vec_G = np.zeros(k)
            band_vec_G[start:end] = G_proj[start:end]
            G_band_recon = U_wu @ band_vec_G
            
            band_vec_A = np.zeros(k)
            band_vec_A[start:end] = A_proj[start:end]
            A_band_recon = U_wu @ band_vec_A
            
            logit_G = float(W_U_word @ G_band_recon)
            logit_A = float(W_U_word @ A_band_recon)
            
            if b < 2:  # 低频段
                G_low_logit += logit_G
                A_low_logit += logit_A
            elif b >= n_bands - 2:  # 高频段
                G_high_logit += logit_G
                A_high_logit += logit_A
        
        # 差分信号在低频/高频段的能量
        diff = G_contrib + A_contrib
        diff_proj = U_wu.T @ diff
        
        diff_low_energy = 0
        diff_high_energy = 0
        for b in range(n_bands):
            start = b * band_size
            end = (b + 1) * band_size if b < n_bands - 1 else k
            if b < 2:
                diff_low_energy += float(np.sum(diff_proj[start:end] ** 2))
            elif b >= n_bands - 2:
                diff_high_energy += float(np.sum(diff_proj[start:end] ** 2))
        
        diff_lh_ratio = diff_low_energy / max(diff_high_energy, 1e-20)
        
        all_band_features.append({
            "word": word,
            "freq_rank": freq_rank,
            "log_freq": np.log(freq_rank),
            "G_low_logit": G_low_logit,
            "G_high_logit": G_high_logit,
            "A_low_logit": A_low_logit,
            "A_high_logit": A_high_logit,
            "diff_lh_ratio": diff_lh_ratio,
        })
    
    print(f"  有效词数: {len(all_band_features)}")
    
    # 相关性分析
    log_freqs = np.array([wd["log_freq"] for wd in all_band_features])
    
    for feat_name, feat_desc in [
        ("G_low_logit", "G低频段logit"),
        ("G_high_logit", "G高频段logit"),
        ("A_low_logit", "A低频段logit"),
        ("A_high_logit", "A高频段logit"),
        ("diff_lh_ratio", "差分信号低/高频段比"),
    ]:
        vals = np.array([wd[feat_name] for wd in all_band_features])
        r, p = spearman_corr(log_freqs, vals)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {feat_desc} vs log_freq: r={r:+.4f}, p={p:.4f} {sig}")
    
    return {
        "rmsnorm_selectivity": {k: v for k, v in rmsnorm_selectivity.items() if isinstance(v, dict)},
        "n_band_features": len(all_band_features),
        "band_freq_correlations": {
            feat: {"r": float(spearman_corr(log_freqs, np.array([wd[feat] for wd in all_band_features]))[0]),
                   "p": float(spearman_corr(log_freqs, np.array([wd[feat] for wd in all_band_features]))[1])}
            for feat in ["G_low_logit", "G_high_logit", "A_low_logit", "A_high_logit", "diff_lh_ratio"]
        },
    }


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Phase CLXV: 频谱编码理论")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["qwen3", "deepseek7b", "glm4"])
    args = parser.parse_args()
    
    model_name = args.model
    print(f"\n{'='*60}")
    print(f"Phase CLXV: 频谱编码理论 - {model_name}")
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
    
    # P716
    p716_results = p716_weight_spectrum(model, tokenizer, device, model_info)
    
    # P717
    p717_results = p717_GA_frequency_bands(model, tokenizer, device, model_info)
    
    # P718
    p718_results = p718_rmsnorm_spectral_selectivity(model, tokenizer, device, model_info)
    
    # 保存结果
    result_dir = f"d:/develop/TransformerLens-main/results/phase_clxv"
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
        "p716": convert(p716_results),
        "p717": convert(p717_results),
        "p718": convert(p718_results),
    }
    
    result_file = os.path.join(result_dir, f"{model_name}_results.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {result_file}")
    
    # 释放模型
    release_model(model)
    gc.collect()
    
    print(f"\n{'='*60}")
    print(f"Phase CLXV ({model_name}) 完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
