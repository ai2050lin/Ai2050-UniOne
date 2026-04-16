#!/usr/bin/env python3
"""
Phase CLXVIII: 编码方程的因果验证
===================================

CLXVII建立了统一编码方程: logit = sum_band alpha_band * band_logit + beta
DS7B的R2=0.96, 但这只是相关性, 不是因果性。

Phase CLXVIII目标: 通过干预实验确认因果方向
  P725: Band1干预 — 增强/减弱Band1信号, 观察logit变化
  P726: RMSNorm因果 — 修改RMSNorm缩放比, 观察diff_retention变化
  P727: 边界条件 — 多义词/组合词/长尾词的编码偏离

因果验证逻辑:
  如果增强Band1 → 低频词logit增加 → Band1是低频词编码的因果通道
  如果修改RMSNorm → diff_retention变化 → RMSNorm是紫牛效应的因果机制

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

# 多义词列表
POLYSEMOUS_WORDS = {
    "bank": {"rank": 400, "senses": ["financial institution", "river edge"]},
    "plant": {"rank": 500, "senses": ["vegetation", "factory"]},
    "rock": {"rank": 700, "senses": ["stone", "music genre"]},
    "iron": {"rank": 800, "senses": ["metal", "appliance"]},
    "bat": {"rank": 1100, "senses": ["animal", "sports equipment"]},
    "ring": {"rank": 600, "senses": ["jewelry", "sound"]},
    "letter": {"rank": 500, "senses": ["alphabet", "mail"]},
    "right": {"rank": 150, "senses": ["correct", "direction"]},
    "light": {"rank": 250, "senses": ["illumination", "weight"]},
    "match": {"rank": 700, "senses": ["game", "fire starter"]},
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
    """对W_U做特征值分解获取频段基"""
    import gc
    d = W_U.shape[1]
    chunk_size = 10000
    WtW = np.zeros((d, d), dtype=np.float32)
    for i in range(0, W_U.shape[0], chunk_size):
        chunk = W_U[i:i+chunk_size].astype(np.float32)
        WtW += chunk.T @ chunk
        del chunk
        gc.collect()
    eigenvalues, eigenvectors = np.linalg.eigh(WtW)
    del WtW
    gc.collect()
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx][:n_components]
    eigenvectors = eigenvectors[:, idx][:, :n_components]
    S = np.sqrt(np.maximum(eigenvalues, 0))
    V = eigenvectors.T  # [n_components, d]
    band_size = n_components // N_BANDS
    bands = {}
    for b in range(N_BANDS):
        start = b * band_size
        end = (b + 1) * band_size if b < N_BANDS - 1 else n_components
        bands[b] = V[start:end]
    return V, S, bands

def forward_pass(model, tokenizer, word, layers, n_layers, mlp_type, device):
    """完整前向传播, 返回h_final和各层中间结果"""
    token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(token_ids) == 0:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
    if len(token_ids) == 0:
        return None
    token_id = token_ids[0]
    
    with torch.no_grad():
        W_E = model.get_input_embeddings().weight.detach().float()
        h = W_E[token_id].cpu().numpy()
        
        layer_h = [h.copy()]  # 各层h
        
        for l_idx in range(n_layers):
            h_normed = rms_norm(h)
            A = compute_A_contrib(layers[l_idx].self_attn, model, h_normed)
            G = compute_G_contrib(layers[l_idx].mlp, h_normed, mlp_type)
            h = h_normed + A + G
            layer_h.append(h.copy())
    
    return {
        "token_id": token_id,
        "h_final": h,
        "layer_h": layer_h,
    }


# ========== P725: Band1干预实验 ==========
def P725_band_intervention(model, tokenizer, device, model_info, model_name):
    """P725: 增强Band1信号, 观察logit变化 — 确认Band1的因果作用"""
    print(f"\n{'='*60}")
    print(f"P725: Band1干预实验 — {model_name}")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    layers = get_layers(model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type if hasattr(model_info, 'mlp_type') else "split_gate_up"
    
    V, S, bands = get_svd_basis(W_U, n_components=100)
    
    # 选择高频词和低频词各10个
    sorted_words = sorted([(w, APPROX_FREQ_RANK.get(w, 999)) for w in ALL_WORDS if w in APPROX_FREQ_RANK], 
                          key=lambda x: x[1])
    high_freq_words = [w for w, r in sorted_words[:15]]  # 高频(rank<450)
    low_freq_words = [w for w, r in sorted_words[-15:]]  # 低频(rank>900)
    
    test_words = high_freq_words + low_freq_words
    print(f"  高频词({len(high_freq_words)}): {high_freq_words[:5]}...")
    print(f"  低频词({len(low_freq_words)}): {low_freq_words[:5]}...")
    
    # 干预参数
    band_scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]  # Band1缩放比
    
    results = {}
    
    for word in test_words:
        data = forward_pass(model, tokenizer, word, layers, n_layers, mlp_type, device)
        if data is None:
            continue
        
        token_id = data["token_id"]
        h_final = data["h_final"]
        
        # 原始logit
        original_logit = float(W_U[token_id] @ h_final)
        
        # 各频段原始logit
        band_logits_orig = {}
        for b, basis in bands.items():
            proj = h_final @ basis.T
            h_band = proj @ basis
            band_logits_orig[b] = float(W_U[token_id] @ h_band)
        
        # Band1干预: 缩放Band1分量
        intervention_results = {}
        for scale in band_scales:
            # 重构h_final: Band1缩放 + 其他不变
            h_modified = np.zeros_like(h_final)
            for b, basis in bands.items():
                proj = h_final @ basis.T
                if b == 0:  # Band1
                    h_modified += (scale * proj) @ basis
                else:
                    h_modified += proj @ basis
            
            # 加上频段外残差
            h_recon_orig = sum((h_final @ basis.T) @ basis for basis in bands.values())
            h_residual = h_final - h_recon_orig
            h_modified += h_residual
            
            modified_logit = float(W_U[token_id] @ h_modified)
            
            # 各频段的修改后logit
            band_logits_mod = {}
            for b, basis in bands.items():
                proj = h_modified @ basis.T
                h_band = proj @ basis
                band_logits_mod[b] = float(W_U[token_id] @ h_band)
            
            intervention_results[scale] = {
                "modified_logit": modified_logit,
                "logit_change": modified_logit - original_logit,
                "relative_change": (modified_logit - original_logit) / (abs(original_logit) + 1e-6),
                "band_logits": band_logits_mod,
            }
        
        results[word] = {
            "freq_rank": APPROX_FREQ_RANK.get(word, 999),
            "log_freq": np.log(APPROX_FREQ_RANK.get(word, 999)),
            "is_low_freq": APPROX_FREQ_RANK.get(word, 999) > 800,
            "original_logit": original_logit,
            "band_logits_orig": band_logits_orig,
            "interventions": intervention_results,
        }
    
    # 分析: Band1增强对低频词 vs 高频词的影响差异
    if results:
        from scipy.stats import spearmanr, mannwhitneyu
        
        # 对每个缩放比, 计算低频词vs高频词的logit变化
        for scale in band_scales:
            low_changes = [r["interventions"][scale]["logit_change"] 
                          for r in results.values() if r["is_low_freq"]]
            high_changes = [r["interventions"][scale]["logit_change"] 
                           for r in results.values() if not r["is_low_freq"]]
            
            low_rel = [r["interventions"][scale]["relative_change"] 
                      for r in results.values() if r["is_low_freq"]]
            high_rel = [r["interventions"][scale]["relative_change"] 
                       for r in results.values() if not r["is_low_freq"]]
            
            # logit变化与log_freq的相关性
            all_changes = [r["interventions"][scale]["logit_change"] for r in results.values()]
            all_freq = [r["log_freq"] for r in results.values()]
            r_change_freq, p_change_freq = spearmanr(all_changes, all_freq) if len(all_changes) > 5 else (0, 1)
            
            print(f"\n  Band1 scale={scale}:")
            print(f"    低频词logit变化: mean={np.mean(low_changes):.2f}, rel={np.mean(low_rel):.4f}")
            print(f"    高频词logit变化: mean={np.mean(high_changes):.2f}, rel={np.mean(high_rel):.4f}")
            if len(low_changes) > 3 and len(high_changes) > 3:
                u_stat, u_p = mannwhitneyu(low_changes, high_changes, alternative='two-sided')
                print(f"    Mann-Whitney U: stat={u_stat:.1f}, p={u_p:.4f}")
            print(f"    logit变化 vs log_freq: Spearman r={r_change_freq:.3f}, p={p_change_freq:.4e}")
    
    return results


# ========== P726: RMSNorm因果验证 ==========
def P726_rmsnorm_causality(model, tokenizer, device, model_info, model_name):
    """P726: 修改RMSNorm缩放比, 观察diff_retention变化"""
    print(f"\n{'='*60}")
    print(f"P726: RMSNorm因果验证 — {model_name}")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    layers = get_layers(model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type if hasattr(model_info, 'mlp_type') else "split_gate_up"
    
    V, S, bands = get_svd_basis(W_U, n_components=100)
    
    # 选择20个词
    test_words = [w for w in ALL_WORDS if w in APPROX_FREQ_RANK][:20]
    
    # RMSNorm缩放干预
    rms_scales = [0.5, 0.75, 1.0, 1.25, 1.5]
    
    results = {}
    
    for word in test_words:
        token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(token_ids) == 0:
            token_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(token_ids) == 0:
            continue
        token_id = token_ids[0]
        
        word_results = {}
        
        for rms_scale in rms_scales:
            # 修改后的前向传播: 在末端3层修改RMSNorm
            with torch.no_grad():
                W_E = model.get_input_embeddings().weight.detach().float()
                h = W_E[token_id].cpu().numpy()
                
                for l_idx in range(n_layers):
                    # RMSNorm
                    h_normed = rms_norm(h)
                    
                    # 在末端3层应用缩放干预
                    if l_idx >= n_layers - 3:
                        h_normed = h_normed * rms_scale
                    
                    A = compute_A_contrib(layers[l_idx].self_attn, model, h_normed)
                    G = compute_G_contrib(layers[l_idx].mlp, h_normed, mlp_type)
                    h = h_normed + A + G
                
                h_final = h
            
            # 计算logit
            logit = float(W_U[token_id] @ h_final)
            
            # 频段logit
            band_logits = {}
            for b, basis in bands.items():
                proj = h_final @ basis.T
                h_band = proj @ basis
                band_logits[b] = float(W_U[token_id] @ h_band)
            
            # diff_retention: h_final中差分信号的保留
            # 差分信号 = h_final - mean(h_final)
            h_mean = np.mean(h_final)
            h_diff = h_final - h_mean
            diff_norm = np.sqrt(np.mean(h_diff**2))
            
            # 信号在W_U方向的投影比例
            w_u_word = W_U[token_id]
            w_u_norm = np.sqrt(np.mean(w_u_word**2))
            projection = float(w_u_word @ h_final) / (w_u_norm * np.sqrt(np.mean(h_final**2)) + 1e-6)
            
            word_results[rms_scale] = {
                "logit": logit,
                "band_logits": band_logits,
                "diff_norm": float(diff_norm),
                "projection": float(projection),
            }
        
        results[word] = {
            "freq_rank": APPROX_FREQ_RANK.get(word, 999),
            "log_freq": np.log(APPROX_FREQ_RANK.get(word, 999)),
            "is_low_freq": APPROX_FREQ_RANK.get(word, 999) > 800,
            "rms_results": word_results,
        }
    
    # 分析: RMSNorm缩放对低频词vs高频词的diff_retention影响差异
    if results:
        print(f"\n  RMSNorm缩放对logit和diff_norm的影响:")
        for rms_scale in rms_scales:
            low_logits = [r["rms_results"][rms_scale]["logit"] 
                         for r in results.values() if r["is_low_freq"]]
            high_logits = [r["rms_results"][rms_scale]["logit"] 
                          for r in results.values() if not r["is_low_freq"]]
            low_diffs = [r["rms_results"][rms_scale]["diff_norm"]
                        for r in results.values() if r["is_low_freq"]]
            high_diffs = [r["rms_results"][rms_scale]["diff_norm"]
                         for r in results.values() if not r["is_low_freq"]]
            
            print(f"  rms_scale={rms_scale}:")
            print(f"    低频词 logit={np.mean(low_logits):.2f}, diff={np.mean(low_diffs):.4f}")
            print(f"    高频词 logit={np.mean(high_logits):.2f}, diff={np.mean(high_diffs):.4f}")
            print(f"    低/高 logit比={np.mean(low_logits)/(np.mean(high_logits)+1e-6):.4f}")
            print(f"    低/高 diff比={np.mean(low_diffs)/(np.mean(high_diffs)+1e-6):.4f}")
    
    return results


# ========== P727: 编码方程边界条件 ==========
def P727_boundary_conditions(model, tokenizer, device, model_info, model_name):
    """P727: 多义词/组合词/长尾词的编码偏离"""
    print(f"\n{'='*60}")
    print(f"P727: 编码方程边界条件 — {model_name}")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    layers = get_layers(model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type if hasattr(model_info, 'mlp_type') else "split_gate_up"
    
    V, S, bands = get_svd_basis(W_U, n_components=100)
    
    # 测试词: 多义词 + 普通词 + 极低频词
    test_cases = {
        "polysemous": list(POLYSEMOUS_WORDS.keys()),
        "normal": ["apple", "cat", "knife", "hand", "red", "water", "car", "house", "bread", "dog"],
        "rare": ["wrench", "pliers", "melon", "mango", "axe", "saw", "screw", "taxi", "rail", "cherry"],
    }
    
    results = {}
    
    for case_type, words in test_cases.items():
        case_results = {}
        
        for word in words:
            token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
            if len(token_ids) == 0:
                token_ids = tokenizer.encode(word, add_special_tokens=False)
            if len(token_ids) == 0:
                continue
            token_id = token_ids[0]
            
            # 前向传播
            with torch.no_grad():
                W_E = model.get_input_embeddings().weight.detach().float()
                h = W_E[token_id].cpu().numpy()
                
                for l_idx in range(n_layers):
                    h_normed = rms_norm(h)
                    A = compute_A_contrib(layers[l_idx].self_attn, model, h_normed)
                    G = compute_G_contrib(layers[l_idx].mlp, h_normed, mlp_type)
                    h = h_normed + A + G
                
                h_final = h
            
            # 完整logit
            logit = float(W_U[token_id] @ h_final)
            
            # 频段logit
            band_logits = {}
            total_band_logit = 0
            for b, basis in bands.items():
                proj = h_final @ basis.T
                h_band = proj @ basis
                bl = float(W_U[token_id] @ h_band)
                band_logits[b] = bl
                total_band_logit += bl
            
            # 残差
            h_recon = sum((h_final @ basis.T) @ basis for basis in bands.values())
            h_residual = h_final - h_recon
            residual_logit = float(W_U[token_id] @ h_residual)
            
            # Band1占比
            band1_ratio = band_logits[0] / (abs(logit) + 1e-6) if abs(logit) > 0.01 else 0
            
            # 频段能量分布
            band_energy = {}
            for b, basis in bands.items():
                proj = h_final @ basis.T
                band_energy[b] = float(np.sum(proj**2))
            
            # 频谱熵
            total_energy = sum(band_energy.values()) + 1e-10
            band_probs = [e / total_energy for e in band_energy.values()]
            spectral_entropy = -sum(p * np.log2(p + 1e-10) for p in band_probs)
            
            case_results[word] = {
                "freq_rank": APPROX_FREQ_RANK.get(word, POLYSEMOUS_WORDS.get(word, {}).get("rank", 999)),
                "logit": logit,
                "band_logits": band_logits,
                "total_band_logit": total_band_logit,
                "residual_logit": residual_logit,
                "band1_ratio": band1_ratio,
                "band_energy": band_energy,
                "spectral_entropy": spectral_entropy,
            }
        
        results[case_type] = case_results
    
    # 分析: 多义词 vs 普通词 vs 长尾词的差异
    print(f"\n  各类词的编码特征对比:")
    for case_type in ["polysemous", "normal", "rare"]:
        case_data = results.get(case_type, {})
        if not case_data:
            continue
        
        band1_ratios = [r["band1_ratio"] for r in case_data.values()]
        entropies = [r["spectral_entropy"] for r in case_data.values()]
        band1_logits = [r["band_logits"][0] for r in case_data.values()]
        residual_ratios = [abs(r["residual_logit"]) / (abs(r["logit"]) + 1e-6) for r in case_data.values()]
        
        print(f"  {case_type} ({len(case_data)}词):")
        print(f"    Band1占比: mean={np.mean(band1_ratios):.4f}")
        print(f"    频谱熵: mean={np.mean(entropies):.4f}")
        print(f"    Band1 logit: mean={np.mean(band1_logits):.2f}")
        print(f"    残差占比: mean={np.mean(residual_ratios):.4f}")
    
    # 多义词分析: 不同sense是否有不同的频段分布?
    print(f"\n  多义词的频段分布特征:")
    for word, info in POLYSEMOUS_WORDS.items():
        if word in results.get("polysemous", {}):
            r = results["polysemous"][word]
            print(f"    {word}({', '.join(info['senses'][:2])}): "
                  f"Band1={r['band_logits'][0]:.2f}, "
                  f"entropy={r['spectral_entropy']:.3f}, "
                  f"Band1_ratio={r['band1_ratio']:.4f}")
    
    return results


# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser(description="Phase CLXVIII: 编码方程因果验证")
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "deepseek7b", "glm4"])
    args = parser.parse_args()
    
    model_name = args.model
    print(f"\n{'#'*60}")
    print(f"Phase CLXVIII: 编码方程因果验证 — {model_name}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    
    print(f"模型: {model_name}, 层数: {model_info.n_layers}, 维度: {model_info.d_model}")
    
    results = {}
    
    try:
        results["P725"] = P725_band_intervention(model, tokenizer, device, model_info, model_name)
    except Exception as e:
        print(f"P725失败: {e}")
        import traceback
        traceback.print_exc()
        results["P725"] = {"error": str(e)}
    
    try:
        results["P726"] = P726_rmsnorm_causality(model, tokenizer, device, model_info, model_name)
    except Exception as e:
        print(f"P726失败: {e}")
        import traceback
        traceback.print_exc()
        results["P726"] = {"error": str(e)}
    
    try:
        results["P727"] = P727_boundary_conditions(model, tokenizer, device, model_info, model_name)
    except Exception as e:
        print(f"P727失败: {e}")
        import traceback
        traceback.print_exc()
        results["P727"] = {"error": str(e)}
    
    # 保存结果
    output_dir = f"results/phase_clxviii"
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
    print(f"Phase CLXVIII完成 — {model_name}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
