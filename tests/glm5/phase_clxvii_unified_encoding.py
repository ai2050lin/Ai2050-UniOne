#!/usr/bin/env python3
"""
Phase CLXVII: 统一编码方程 — "语言编码的信息论本质"
==================================================

基于721个实验(Phase I-CLXVI)的发现，建立语言编码的统一方程。

核心发现回顾:
  1. logit_gap = sum_k c_k * Delta_k (加法模型, r=1.000)
  2. G和A可频谱分解: G = sum_band G_band, A = sum_band A_band
  3. Band1(主成分)对所有词都最优, 低频词效率更高(r=0.58-0.63)
  4. RMSNorm全局缩放, 频谱选择通过G和A间接实现
  5. 末端层W_o频谱急剧集中(GLM4: entropy 0.99→0.73)

统一编码方程:
  编码: h_final = RMSNorm(h_0 + sum_l [G_l(h_{l-1}) + A_l(h_{l-1})])
  频谱分解: G_l = sum_band G_l^band, A_l = sum_band A_l^band
  解码: logit(word) = sum_band W_U^band[word] · h_band

Phase CLXVII目标:
  P722: 编码方程验证 — 频谱分解的G和A能否精确重建logit
  P723: 解码方程验证 — 频段贡献加和能否重建完整logit
  P724: 统一方程预测 — 用频谱编码方程预测新词的logit

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
    "banana": 900, "mango": 1300, "peach": 950, "pear": 1000, "melon": 1100,
    "knife": 700, "hammer": 800, "saw": 900, "drill": 1000, "nail": 850,
    "screw": 950, "wrench": 1100, "pliers": 1200, "axe": 1050, "blade": 900,
    "love": 200, "hate": 600, "fear": 400, "hope": 350, "truth": 500,
    "justice": 700, "beauty": 550, "wisdom": 800, "peace": 450, "power": 300,
}

N_BANDS = 5

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


def get_svd_basis(W_U, n_components=100):
    """对W_U做SVD, 返回频段基 — 使用特征值分解避免内存溢出"""
    import gc
    
    d = W_U.shape[1]
    
    # 方法: 直接对W_U.T @ W_U做特征值分解
    # W_U.T @ W_U = V @ diag(S^2) @ V.T
    # 所以V就是W_U.T @ W_U的特征向量
    # 关键: 不需要W_U全部float32, 可以分块计算WtW
    
    print(f"  计算W_U.T @ W_U (shape={d}x{d})...")
    
    # 分块计算避免内存溢出
    chunk_size = 10000
    WtW = np.zeros((d, d), dtype=np.float32)
    for i in range(0, W_U.shape[0], chunk_size):
        chunk = W_U[i:i+chunk_size].astype(np.float32)
        WtW += chunk.T @ chunk
        del chunk
        gc.collect()
    
    # 释放W_U的float32副本(如果有的话)
    gc.collect()
    
    # 特征值分解
    print(f"  特征值分解...")
    eigenvalues, eigenvectors = np.linalg.eigh(WtW)
    del WtW
    gc.collect()
    
    # 按特征值从大到小排序
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx][:n_components]
    eigenvectors = eigenvectors[:, idx][:, :n_components]  # [d, n_components]
    
    # 奇异值 = sqrt(特征值)
    S = np.sqrt(np.maximum(eigenvalues, 0))
    V = eigenvectors.T  # [n_components, d]
    
    band_size = n_components // N_BANDS
    bands = {}
    for b in range(N_BANDS):
        start = b * band_size
        end = (b + 1) * band_size if b < N_BANDS - 1 else n_components
        bands[b] = V[start:end]
    
    return V, S, bands


# ========== P722: 编码方程验证 ==========
def P722_encoding_equation(model, tokenizer, device, model_info, model_name):
    """P722: 编码方程验证 — 频谱分解的G和A能否精确重建logit"""
    print(f"\n{'='*60}")
    print(f"P722: 编码方程验证 — {model_name}")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    layers = get_layers(model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type if hasattr(model_info, 'mlp_type') else "split_gate_up"
    
    V, S, bands = get_svd_basis(W_U, n_components=100)
    
    sample_layers = get_sample_layers(n_layers, 5)
    
    test_words = ["apple", "cat", "knife", "hand", "love",
                   "red", "water", "bread", "car", "house",
                   "mango", "wolf", "pliers", "neck", "wisdom",
                   "gray", "moon", "wine", "taxi", "bridge"]
    
    results = {}
    
    for layer_idx in sample_layers:
        layer_name = f"L{layer_idx}"
        print(f"\n  --- {layer_name} ---")
        
        sa = layers[layer_idx].self_attn
        mlp = layers[layer_idx].mlp
        
        word_results = {}
        
        for word in test_words:
            token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
            if len(token_ids) == 0:
                token_ids = tokenizer.encode(word, add_special_tokens=False)
            if len(token_ids) == 0:
                continue
            token_id = token_ids[0]
            
            # 前向传播到该层, 获取h
            with torch.no_grad():
                W_E = model.get_input_embeddings().weight.detach().float()
                h = W_E[token_id].cpu().numpy()  # [d]
                
                for l_idx in range(layer_idx):
                    h_normed = rms_norm(h)
                    A = compute_A_contrib(layers[l_idx].self_attn, model, h_normed)
                    G = compute_G_contrib(layers[l_idx].mlp, h_normed, mlp_type)
                    h = h_normed + A + G  # 简化: 跳过中间LayerNorm
                
                # 该层的h
                h_at_layer = h.copy()
                h_normed = rms_norm(h_at_layer)
                
                # 该层的G和A
                A_contrib = compute_A_contrib(sa, model, h_normed)
                G_contrib = compute_G_contrib(mlp, h_normed, mlp_type)
            
            # 实际G和A logit(对目标词)
            actual_G_logit = float(W_U[token_id] @ G_contrib)
            actual_A_logit = float(W_U[token_id] @ A_contrib)
            
            # 频段分解
            G_band_logits = {}
            A_band_logits = {}
            
            for b, basis in bands.items():
                # G频段: 将G_contrib投影到basis上再重建
                G_proj = G_contrib @ basis.T  # [band_size]
                G_band = G_proj @ basis  # [d]
                G_band_logit = float(W_U[token_id] @ G_band)
                G_band_logits[b] = G_band_logit
                
                # A频段
                A_proj = A_contrib @ basis.T
                A_band = A_proj @ basis
                A_band_logit = float(W_U[token_id] @ A_band)
                A_band_logits[b] = A_band_logit
            
            total_G_recon = sum(G_band_logits.values())
            total_A_recon = sum(A_band_logits.values())
            
            word_results[word] = {
                "actual_G_logit": actual_G_logit,
                "actual_A_logit": actual_A_logit,
                "G_band_logits": G_band_logits,
                "A_band_logits": A_band_logits,
                "G_recon": total_G_recon,
                "A_recon": total_A_recon,
                "G_recon_error": abs(total_G_recon - actual_G_logit) / (abs(actual_G_logit) + 1e-6),
                "A_recon_error": abs(total_A_recon - actual_A_logit) / (abs(actual_A_logit) + 1e-6),
            }
        
        if word_results:
            G_errors = [r["G_recon_error"] for r in word_results.values()]
            A_errors = [r["A_recon_error"] for r in word_results.values()]
            
            results[layer_name] = {
                "n_words": len(word_results),
                "G_recon_error_mean": float(np.mean(G_errors)),
                "G_recon_error_median": float(np.median(G_errors)),
                "A_recon_error_mean": float(np.mean(A_errors)),
                "A_recon_error_median": float(np.median(A_errors)),
            }
            
            print(f"  G频段重建误差: mean={results[layer_name]['G_recon_error_mean']:.4f}, "
                  f"median={results[layer_name]['G_recon_error_median']:.4f}")
            print(f"  A频段重建误差: mean={results[layer_name]['A_recon_error_mean']:.4f}, "
                  f"median={results[layer_name]['A_recon_error_median']:.4f}")
    
    return results


# ========== P723: 解码方程验证 ==========
def P723_decoding_equation(model, tokenizer, device, model_info, model_name):
    """P723: 解码方程验证 — 频段贡献加和能否重建完整logit"""
    print(f"\n{'='*60}")
    print(f"P723: 解码方程验证 — {model_name}")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    layers = get_layers(model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type if hasattr(model_info, 'mlp_type') else "split_gate_up"
    
    V, S, bands = get_svd_basis(W_U, n_components=100)
    
    test_words = [w for w in ALL_WORDS if w in APPROX_FREQ_RANK][:80]
    print(f"  测试{len(test_words)}个词")
    
    results = {}
    
    for word in test_words:
        token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(token_ids) == 0:
            token_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(token_ids) == 0:
            continue
        token_id = token_ids[0]
        
        # 完整前向传播
        with torch.no_grad():
            W_E = model.get_input_embeddings().weight.detach().float()
            h = W_E[token_id].cpu().numpy()
            
            for l_idx in range(n_layers):
                h_normed = rms_norm(h)
                A = compute_A_contrib(layers[l_idx].self_attn, model, h_normed)
                G = compute_G_contrib(layers[l_idx].mlp, h_normed, mlp_type)
                h = h_normed + A + G
            
            h_final = h.copy()
        
        # 实际logit
        actual_logit = float(W_U[token_id] @ h_final)
        
        # 频段分解h_final
        band_logits = {}
        total_recon_logit = 0
        
        for b, basis in bands.items():
            proj = h_final @ basis.T
            h_band = proj @ basis
            band_logit = float(W_U[token_id] @ h_band)
            band_logits[b] = band_logit
            total_recon_logit += band_logit
        
        # 频段外残差
        h_recon = sum((h_final @ basis.T) @ basis for basis in bands.values())
        h_residual = h_final - h_recon
        residual_logit = float(W_U[token_id] @ h_residual)
        
        # 全频段重建
        full_recon = total_recon_logit + residual_logit
        
        results[word] = {
            "freq_rank": APPROX_FREQ_RANK.get(word, 999),
            "log_freq": np.log(APPROX_FREQ_RANK.get(word, 999)),
            "actual_logit": actual_logit,
            "band_logits": band_logits,
            "total_recon_logit": total_recon_logit,
            "residual_logit": residual_logit,
            "full_recon_logit": full_recon,
            "recon_error": abs(full_recon - actual_logit) / (abs(actual_logit) + 1e-6),
            "band_only_ratio": total_recon_logit / (actual_logit + 1e-6) if abs(actual_logit) > 0.01 else 0,
        }
    
    if not results:
        return {}
    
    from scipy.stats import spearmanr
    
    recon_errors = [r["recon_error"] for r in results.values()]
    band_ratios = [r["band_only_ratio"] for r in results.values() if abs(r["band_only_ratio"]) > 0.01]
    
    band_means = {}
    for b in range(N_BANDS):
        band_vals = [r["band_logits"][b] for r in results.values()]
        band_means[b] = float(np.mean(band_vals))
    
    # 频段贡献与词频的相关性
    band_freq_corr = {}
    for b in range(N_BANDS):
        band_vals = [r["band_logits"][b] for r in results.values()]
        freq_vals = [r["log_freq"] for r in results.values()]
        if len(band_vals) > 10:
            r_val, p_val = spearmanr(band_vals, freq_vals)
            band_freq_corr[b] = {"r": float(r_val), "p": float(p_val)}
    
    actual_logits = [r["actual_logit"] for r in results.values()]
    recon_logits = [r["full_recon_logit"] for r in results.values()]
    recon_r, recon_p = spearmanr(actual_logits, recon_logits) if len(actual_logits) > 5 else (0, 1)
    
    summary = {
        "n_words": len(results),
        "recon_error_mean": float(np.mean(recon_errors)),
        "recon_error_median": float(np.median(recon_errors)),
        "band_ratio_mean": float(np.mean(band_ratios)) if band_ratios else -1,
        "band_means": {str(k): v for k, v in band_means.items()},
        "band_freq_corr": {str(k): v for k, v in band_freq_corr.items()},
        "recon_spearman_r": float(recon_r),
        "recon_spearman_p": float(recon_p),
        "residual_logit_mean": float(np.mean([abs(r["residual_logit"]) for r in results.values()])),
    }
    
    print(f"  词数: {summary['n_words']}")
    print(f"  重建误差: mean={summary['recon_error_mean']:.4f}, median={summary['recon_error_median']:.4f}")
    print(f"  频段重建比: mean={summary['band_ratio_mean']:.4f}")
    print(f"  重建Spearman: r={summary['recon_spearman_r']:.4f}, p={summary['recon_spearman_p']:.4e}")
    print(f"  频段外残差: mean={summary['residual_logit_mean']:.4f}")
    for b, corr in band_freq_corr.items():
        sig = "***" if corr["p"] < 0.001 else "**" if corr["p"] < 0.01 else "*" if corr["p"] < 0.05 else ""
        print(f"  Band{b} vs log_freq: r={corr['r']:.3f}, p={corr['p']:.4e} {sig}")
    
    return {"summary": summary, "word_details": results}


# ========== P724: 统一方程预测 ==========
def P724_unified_prediction(model, tokenizer, device, model_info, model_name):
    """P724: 统一方程预测 — 用频谱编码方程预测新词的logit"""
    print(f"\n{'='*60}")
    print(f"P724: 统一方程预测 — {model_name}")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    layers = get_layers(model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type if hasattr(model_info, 'mlp_type') else "split_gate_up"
    
    V, S, bands = get_svd_basis(W_U, n_components=100)
    
    all_test = [w for w in ALL_WORDS if w in APPROX_FREQ_RANK]
    np.random.seed(42)
    np.random.shuffle(all_test)
    train_words = all_test[:60]
    test_words = all_test[60:80]
    
    print(f"  训练集: {len(train_words)}词, 测试集: {len(test_words)}词")
    
    # Step 1: 收集训练数据
    train_data = {}
    
    for word in train_words:
        token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(token_ids) == 0:
            token_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(token_ids) == 0:
            continue
        token_id = token_ids[0]
        
        with torch.no_grad():
            W_E = model.get_input_embeddings().weight.detach().float()
            h = W_E[token_id].cpu().numpy()
            
            for l_idx in range(n_layers):
                h_normed = rms_norm(h)
                A = compute_A_contrib(layers[l_idx].self_attn, model, h_normed)
                G = compute_G_contrib(layers[l_idx].mlp, h_normed, mlp_type)
                h = h_normed + A + G
            
            h_final = h.copy()
        
        actual_logit = float(W_U[token_id] @ h_final)
        
        # 频段logit
        band_logits = {}
        for b, basis in bands.items():
            proj = h_final @ basis.T
            h_band = proj @ basis
            band_logits[b] = float(W_U[token_id] @ h_band)
        
        train_data[word] = {
            "logit": actual_logit,
            "band_logits": band_logits,
        }
    
    if len(train_data) < 10:
        print("  训练数据不足!")
        return {}
    
    # Step 2: 学习线性回归
    from sklearn.linear_model import Ridge
    from scipy.stats import spearmanr, pearsonr
    
    X_train = []
    y_train = []
    
    for word, data in train_data.items():
        X_train.append([data["band_logits"][b] for b in range(N_BANDS)])
        y_train.append(data["logit"])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    reg = Ridge(alpha=1.0).fit(X_train, y_train)
    train_r2 = reg.score(X_train, y_train)
    
    print(f"  训练R2: {train_r2:.4f}")
    print(f"  回归系数: {dict(zip([f'Band{b}' for b in range(N_BANDS)], reg.coef_.round(4)))}")
    
    # Step 3: 测试
    test_results = {}
    
    for word in test_words:
        token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(token_ids) == 0:
            token_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(token_ids) == 0:
            continue
        token_id = token_ids[0]
        
        with torch.no_grad():
            W_E = model.get_input_embeddings().weight.detach().float()
            h = W_E[token_id].cpu().numpy()
            
            for l_idx in range(n_layers):
                h_normed = rms_norm(h)
                A = compute_A_contrib(layers[l_idx].self_attn, model, h_normed)
                G = compute_G_contrib(layers[l_idx].mlp, h_normed, mlp_type)
                h = h_normed + A + G
            
            h_final = h.copy()
        
        actual_logit = float(W_U[token_id] @ h_final)
        
        band_logits = []
        for b, basis in bands.items():
            proj = h_final @ basis.T
            h_band = proj @ basis
            band_logits.append(float(W_U[token_id] @ h_band))
        
        predicted_logit = reg.predict([band_logits])[0]
        
        test_results[word] = {
            "actual_logit": actual_logit,
            "predicted_logit": float(predicted_logit),
            "error": float(abs(predicted_logit - actual_logit)),
            "relative_error": float(abs(predicted_logit - actual_logit) / (abs(actual_logit) + 1e-6)),
        }
    
    if test_results:
        actual = [r["actual_logit"] for r in test_results.values()]
        predicted = [r["predicted_logit"] for r in test_results.values()]
        rel_errors = [r["relative_error"] for r in test_results.values()]
        
        sp_r, sp_p = spearmanr(actual, predicted)
        pe_r, pe_p = pearsonr(actual, predicted)
        
        summary = {
            "n_test": len(test_results),
            "train_r2": float(train_r2),
            "test_spearman_r": float(sp_r),
            "test_spearman_p": float(sp_p),
            "test_pearson_r": float(pe_r),
            "test_pearson_p": float(pe_p),
            "test_rel_error_mean": float(np.mean(rel_errors)),
            "test_rel_error_median": float(np.median(rel_errors)),
            "regression_coefs": {f"Band{b}": float(reg.coef_[b]) for b in range(N_BANDS)},
            "regression_intercept": float(reg.intercept_),
        }
        
        print(f"  测试Spearman: r={sp_r:.4f}, p={sp_p:.4e}")
        print(f"  测试Pearson: r={pe_r:.4f}, p={pe_p:.4e}")
        print(f"  测试相对误差: mean={np.mean(rel_errors):.4f}, median={np.median(rel_errors):.4f}")
    else:
        summary = {"n_test": 0, "train_r2": float(train_r2)}
    
    return {"summary": summary, "test_details": test_results}


# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser(description="Phase CLXVII: 统一编码方程")
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "deepseek7b", "glm4"])
    args = parser.parse_args()
    
    model_name = args.model
    print(f"\n{'#'*60}")
    print(f"Phase CLXVII: 统一编码方程 — {model_name}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    
    print(f"模型: {model_name}, 层数: {model_info.n_layers}, 维度: {model_info.d_model}")
    
    results = {}
    
    try:
        results["P722"] = P722_encoding_equation(model, tokenizer, device, model_info, model_name)
    except Exception as e:
        print(f"P722失败: {e}")
        import traceback
        traceback.print_exc()
        results["P722"] = {"error": str(e)}
    
    # P723和P724需要大量内存做SVD, 先释放GPU
    # 但还需要模型做前向传播, 所以不能完全释放
    # 先做P723的前向传播(需要模型), 缓存结果, 再释放模型做SVD
    
    try:
        results["P723"] = P723_decoding_equation(model, tokenizer, device, model_info, model_name)
    except Exception as e:
        print(f"P723失败: {e}")
        import traceback
        traceback.print_exc()
        results["P723"] = {"error": str(e)}
    
    try:
        results["P724"] = P724_unified_prediction(model, tokenizer, device, model_info, model_name)
    except Exception as e:
        print(f"P724失败: {e}")
        import traceback
        traceback.print_exc()
        results["P724"] = {"error": str(e)}
    
    # 保存结果
    output_dir = f"results/phase_clxvii"
    os.makedirs(output_dir, exist_ok=True)
    
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    output_file = f"{output_dir}/{model_name}_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")
    
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\n{'#'*60}")
    print(f"Phase CLXVII完成 — {model_name}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
