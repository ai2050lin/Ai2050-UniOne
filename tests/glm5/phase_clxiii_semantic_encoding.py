#!/usr/bin/env python3
"""
Phase CLXIII: 大规模语义编码规律分析 (100词)
=============================================

核心目标: 分析100个名词的语义/语法编码特征,通过统计找到编码机制的规律

CLXII关键发现:
  - RMSNorm = 差分信号选择器: 对抗信号保留效率是合作的5-19倍
  - 对抗平衡是全局涌现属性, 不是单层局部最优
  - 之前只用了9个词, 统计功效不足

实验设计:
  P710: 100词的编码特征提取
    - 10个语义类别, 每类10词
    - 每个词提取: G/A logit贡献, GA_cos, diff_retention, 头角色分布
    - 全层分析(采样12层+末端3层)

  P711: 编码规律的聚类分析
    - PCA降维 + K-means聚类
    - 与语言学分类对比(Rand Index / Adjusted Rand Index)
    - 发现语义类/语法类/功能类的编码模式

  P712: 跨词编码规律的统一方程
    - 分析编码特征与词频/语义特征的相关性
    - 寻找统一的编码方程参数

词表设计(10类×10词, 优先单token词):
  水果: apple, banana, orange, grape, mango, peach, cherry, lemon, pear, melon
  动物: cat, dog, bird, fish, bear, lion, tiger, wolf, fox, horse
  工具: knife, hammer, saw, drill, nail, screw, wrench, pliers, axe, blade
  身体: hand, foot, head, eye, ear, nose, mouth, arm, leg, neck
  抽象: love, hate, fear, hope, truth, justice, beauty, wisdom, peace, power
  颜色: red, blue, green, yellow, black, white, pink, brown, gray, gold
  自然: sun, moon, star, rain, snow, wind, fire, water, rock, tree
  食物: bread, rice, cake, soup, meat, milk, cheese, egg, salt, wine
  交通: car, bus, train, ship, boat, plane, bike, truck, taxi, rail
  建筑: house, door, wall, roof, room, hall, gate, path, road, bridge

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
        gate_logits = np.clip(W_gate @ h_normed, -500, 500)  # 防止overflow
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


# ============================================================
# P710: 100词编码特征提取
# ============================================================
def p710_extract_encoding_features(model, tokenizer, device, model_info):
    """
    提取100个词的编码特征
    """
    print("\n" + "="*60)
    print("P710: 100词编码特征提取")
    print("="*60)
    
    W_U = get_W_U(model)
    layers = get_layers(model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # 采样层: 均匀12层 + 末端3层
    sample_layers = get_sample_layers(n_layers, 12)
    for li in range(max(0, n_layers - 3), n_layers):
        if li not in sample_layers:
            sample_layers.append(li)
    sample_layers = sorted(set(sample_layers))
    
    print(f"  采样层: {sample_layers}")
    print(f"  词数: {len(ALL_WORDS)}")
    
    word_features = {}
    skipped = []
    
    for wi, word in enumerate(ALL_WORDS):
        # 检查token数量
        word_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(word_ids) != 1:
            skipped.append((word, len(word_ids)))
            continue
        
        word_id = word_ids[0]
        W_U_word = W_U[word_id]
        
        prompt = f"The {word} is"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # 提取每层的编码特征
        layer_data = []
        
        for li in sample_layers:
            h_before = outputs.hidden_states[li][0, -1, :].float().detach().cpu().numpy()
            h_after = outputs.hidden_states[li + 1][0, -1, :].float().detach().cpu().numpy()
            
            G_contrib, A_contrib = None, None
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
            
            # 核心特征
            G_logit = float(W_U_word @ G_contrib)
            A_logit = float(W_U_word @ A_contrib)
            GA_cos = float(np.dot(G_dir, A_dir))
            
            # 差分信号保留
            h_raw = h_before + G_contrib + A_contrib
            diff_signal = G_contrib + A_contrib
            diff_norm = np.linalg.norm(diff_signal)
            h_raw_norm = np.linalg.norm(h_raw)
            scale_factor = np.sqrt(d_model) / max(h_raw_norm, 1e-10)
            diff_retention = diff_norm * scale_factor / max(diff_norm, 1e-10)
            
            # 对抗vs合作
            is_adversarial = (G_logit * A_logit < 0)
            total_logit = G_logit + A_logit
            max_abs = max(abs(G_logit), abs(A_logit), 1e-10)
            opposition_ratio = abs(total_logit) / max_abs
            
            # 范数比
            G_A_norm_ratio = G_norm / max(A_norm, 1e-10)
            
            # delta_h分析
            delta_h = h_after - h_before
            delta_logit = float(W_U_word @ delta_h)
            
            layer_data.append({
                "layer": li,
                "G_logit": G_logit,
                "A_logit": A_logit,
                "GA_cos": GA_cos,
                "diff_retention": float(diff_retention),
                "is_adversarial": is_adversarial,
                "opposition_ratio": opposition_ratio,
                "G_norm": float(G_norm),
                "A_norm": float(A_norm),
                "G_A_norm_ratio": float(G_A_norm_ratio),
                "delta_logit": delta_logit,
                "scale_factor": float(scale_factor),
            })
        
        # 汇总该词的特征
        if layer_data:
            # 全层统计
            adversarial_rate = np.mean([ld["is_adversarial"] for ld in layer_data])
            mean_GA_cos = np.mean([ld["GA_cos"] for ld in layer_data])
            mean_diff_ret = np.mean([ld["diff_retention"] for ld in layer_data])
            mean_opp_ratio = np.mean([ld["opposition_ratio"] for ld in layer_data])
            
            # 末端层特征(最后3层)
            end_layers = [ld for ld in layer_data if ld["layer"] >= n_layers - 3]
            if end_layers:
                end_G_logit = np.mean([ld["G_logit"] for ld in end_layers])
                end_A_logit = np.mean([ld["A_logit"] for ld in end_layers])
                end_GA_cos = np.mean([ld["GA_cos"] for ld in end_layers])
                end_delta_logit = np.mean([ld["delta_logit"] for ld in end_layers])
            else:
                end_G_logit = end_A_logit = end_GA_cos = end_delta_logit = 0
            
            # G和A的贡献模式: 正向/负向/混合
            G_pos_rate = np.mean([1 if ld["G_logit"] > 0 else 0 for ld in layer_data])
            A_pos_rate = np.mean([1 if ld["A_logit"] > 0 else 0 for ld in layer_data])
            
            # 末端层G和A的贡献符号
            if end_layers:
                end_G_sign = np.sign(np.mean([ld["G_logit"] for ld in end_layers]))
                end_A_sign = np.sign(np.mean([ld["A_logit"] for ld in end_layers]))
            else:
                end_G_sign = end_A_sign = 0
            
            word_features[word] = {
                "category": WORD_TO_CATEGORY[word],
                "word_id": word_id,
                "adversarial_rate": float(adversarial_rate),
                "mean_GA_cos": float(mean_GA_cos),
                "mean_diff_retention": float(mean_diff_ret),
                "mean_opposition_ratio": float(mean_opp_ratio),
                "G_pos_rate": float(G_pos_rate),
                "A_pos_rate": float(A_pos_rate),
                "end_G_logit": float(end_G_logit),
                "end_A_logit": float(end_A_logit),
                "end_GA_cos": float(end_GA_cos),
                "end_delta_logit": float(end_delta_logit),
                "end_G_sign": float(end_G_sign),
                "end_A_sign": float(end_A_sign),
                "layer_data": layer_data,
            }
        
        del outputs
        gc.collect()
        
        if (wi + 1) % 10 == 0:
            print(f"  进度: {wi+1}/{len(ALL_WORDS)} 词完成")
    
    if skipped:
        print(f"  跳过多token词: {skipped}")
    
    print(f"  有效词数: {len(word_features)}/{len(ALL_WORDS)}")
    
    return word_features


# ============================================================
# P711: 编码规律的聚类分析
# ============================================================
def p711_clustering_analysis(word_features):
    """
    对编码特征进行聚类分析,与语言学分类对比
    """
    print("\n" + "="*60)
    print("P711: 编码规律的聚类分析")
    print("="*60)
    
    if len(word_features) < 20:
        print("  词数不足, 跳过聚类分析")
        return {}
    
    # 构建特征矩阵
    words = sorted(word_features.keys())
    feature_names = [
        "adversarial_rate", "mean_GA_cos", "mean_diff_retention",
        "mean_opposition_ratio", "G_pos_rate", "A_pos_rate",
        "end_G_logit", "end_A_logit", "end_GA_cos", "end_delta_logit",
    ]
    
    X = np.array([[word_features[w][fn] for fn in feature_names] for w in words])
    categories = [word_features[w]["category"] for w in words]
    
    # 标准化
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X - X_mean) / X_std
    
    # PCA
    from numpy.linalg import svd
    # 中心化已做
    U, S, Vt = svd(X_norm, full_matrices=False)
    
    # 前几个主成分的方差解释比
    total_var = np.sum(S ** 2)
    var_explained = S ** 2 / total_var
    
    print(f"  PCA方差解释比: PC1={var_explained[0]:.3f}, PC2={var_explained[1]:.3f}, PC3={var_explained[2]:.3f}")
    print(f"  累计: PC1-3={sum(var_explained[:3]):.3f}, PC1-5={sum(var_explained[:5]):.3f}")
    
    # K-means聚类 (k=10, 与语义类别数一致)
    n_clusters = min(10, len(words) // 3)
    
    # 简单K-means实现
    def kmeans(X, k, max_iter=100, seed=42):
        np.random.seed(seed)
        n = X.shape[0]
        # 初始化: 随机选k个点
        idx = np.random.choice(n, k, replace=False)
        centers = X[idx].copy()
        
        for _ in range(max_iter):
            # 分配
            dists = np.zeros((n, k))
            for ki in range(k):
                dists[:, ki] = np.sum((X - centers[ki]) ** 2, axis=1)
            labels = np.argmin(dists, axis=1)
            
            # 更新中心
            new_centers = np.zeros_like(centers)
            for ki in range(k):
                mask = labels == ki
                if mask.sum() > 0:
                    new_centers[ki] = X[mask].mean(axis=0)
                else:
                    new_centers[ki] = centers[ki]
            
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
        
        return labels, centers
    
    # 用PCA前5维做聚类
    X_pca = U[:, :5] * S[:5]  # 投影到前5个主成分
    labels, centers = kmeans(X_pca, n_clusters)
    
    # 与语言学分类对比: Adjusted Rand Index
    cat_to_int = {cat: i for i, cat in enumerate(sorted(set(categories)))}
    true_labels = np.array([cat_to_int[c] for c in categories])
    
    # 计算ARI
    def adjusted_rand_index(labels1, labels2):
        from itertools import combinations
        n = len(labels1)
        if n < 2:
            return 0
        
        # 构建配对
        pairs = list(combinations(range(n), 2))
        a = sum(1 for i, j in pairs if labels1[i] == labels1[j] and labels2[i] == labels2[j])
        b = sum(1 for i, j in pairs if labels1[i] != labels1[j] and labels2[i] != labels2[j])
        c = sum(1 for i, j in pairs if labels1[i] == labels1[j] and labels2[i] != labels2[j])
        d = sum(1 for i, j in pairs if labels1[i] != labels1[j] and labels2[i] == labels2[j])
        
        # Rand Index
        ri = (a + b) / len(pairs)
        
        # Expected RI
        n_pairs = len(pairs)
        row_sums = {}
        col_sums = {}
        for i in range(n):
            row_sums[labels1[i]] = row_sums.get(labels1[i], 0) + 1
            col_sums[labels2[i]] = col_sums.get(labels2[i], 0) + 1
        
        sum_comb_row = sum(v * (v - 1) // 2 for v in row_sums.values())
        sum_comb_col = sum(v * (v - 1) // 2 for v in col_sums.values())
        
        expected = (sum_comb_row * sum_comb_col) / n_pairs
        max_ri = 0.5 * (sum_comb_row + sum_comb_col)
        
        if max_ri == expected:
            return 0
        
        ari = (ri * n_pairs - expected) / (max_ri - expected)
        return ari
    
    ari = adjusted_rand_index(true_labels, labels)
    print(f"  Adjusted Rand Index (编码聚类 vs 语义类别): {ari:.4f}")
    
    # 分析每个聚类的语义组成
    print("\n  聚类 vs 语义类别交叉表:")
    cluster_cat_counts = {}
    for ki in range(n_clusters):
        mask = labels == ki
        cats_in_cluster = [categories[i] for i in range(len(words)) if mask[i]]
        cat_counts = {}
        for c in cats_in_cluster:
            cat_counts[c] = cat_counts.get(c, 0) + 1
        cluster_cat_counts[ki] = cat_counts
        
        top_cats = sorted(cat_counts.items(), key=lambda x: -x[1])[:3]
        print(f"    Cluster {ki}: {sum(mask)}词, top类别: {top_cats}")
    
    # 分析每个语义类别的编码特征均值
    print("\n  语义类别的编码特征均值:")
    cat_features = {}
    for cat in sorted(set(categories)):
        cat_words = [w for w in words if word_features[w]["category"] == cat]
        if not cat_words:
            continue
        cat_feats = {}
        for fn in feature_names:
            vals = [word_features[w][fn] for w in cat_words]
            cat_feats[fn] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }
        cat_features[cat] = cat_feats
        
        print(f"    {cat}: adv_rate={cat_feats['adversarial_rate']['mean']:.3f}, "
              f"GA_cos={cat_feats['mean_GA_cos']['mean']:.3f}, "
              f"diff_ret={cat_feats['mean_diff_retention']['mean']:.4f}, "
              f"end_G={cat_feats['end_G_logit']['mean']:.2f}, "
              f"end_A={cat_feats['end_A_logit']['mean']:.2f}")
    
    # PCA前2维的类别分布
    print("\n  PCA前2维的类别中心:")
    pc1 = U[:, 0] * S[0]
    pc2 = U[:, 1] * S[1]
    for cat in sorted(set(categories)):
        mask = np.array([c == cat for c in categories])
        print(f"    {cat}: PC1={pc1[mask].mean():.3f}, PC2={pc2[mask].mean():.3f}")
    
    return {
        "var_explained": var_explained[:10].tolist(),
        "ari": float(ari),
        "n_clusters": n_clusters,
        "cluster_labels": labels.tolist(),
        "true_labels": true_labels.tolist(),
        "cat_features": cat_features,
        "pca_pc1": pc1.tolist(),
        "pca_pc2": pc2.tolist(),
        "feature_names": feature_names,
    }


# ============================================================
# P712: 跨词编码规律的统一方程
# ============================================================
def p712_unified_equation(word_features, clustering_results):
    """
    分析编码特征与语义特征的相关性,寻找统一方程参数
    """
    print("\n" + "="*60)
    print("P712: 跨词编码规律的统一方程")
    print("="*60)
    
    if len(word_features) < 20:
        print("  词数不足, 跳过统一方程分析")
        return {}
    
    words = sorted(word_features.keys())
    
    # 1. 末端层G_logit vs A_logit的关系
    end_G = np.array([word_features[w]["end_G_logit"] for w in words])
    end_A = np.array([word_features[w]["end_A_logit"] for w in words])
    end_delta = np.array([word_features[w]["end_delta_logit"] for w in words])
    
    # G_logit + A_logit ≈ delta_logit?
    G_plus_A = end_G + end_A
    corr_GA_delta = np.corrcoef(G_plus_A, end_delta)[0, 1]
    print(f"  1. G_logit + A_logit vs delta_logit 相关系数: {corr_GA_delta:.4f}")
    
    # 2. 对抗率与GA_cos的关系
    adv_rates = np.array([word_features[w]["adversarial_rate"] for w in words])
    ga_coses = np.array([word_features[w]["mean_GA_cos"] for w in words])
    corr_adv_gacos = np.corrcoef(adv_rates, ga_coses)[0, 1]
    print(f"  2. adversarial_rate vs GA_cos 相关系数: {corr_adv_gacos:.4f}")
    
    # 3. diff_retention与opposition_ratio的关系
    diff_rets = np.array([word_features[w]["mean_diff_retention"] for w in words])
    opp_ratios = np.array([word_features[w]["mean_opposition_ratio"] for w in words])
    corr_diff_opp = np.corrcoef(diff_rets, opp_ratios)[0, 1]
    print(f"  3. diff_retention vs opposition_ratio 相关系数: {corr_diff_opp:.4f}")
    
    # 4. 语义类别间的编码差异
    print("\n  4. 语义类别间编码差异(ANOVA-like):")
    categories = sorted(set(word_features[w]["category"] for w in words))
    
    # 对每个特征,计算类间方差/类内方差
    feature_names = ["adversarial_rate", "mean_GA_cos", "mean_diff_retention",
                     "end_G_logit", "end_A_logit", "end_delta_logit"]
    
    feature_discrimination = {}
    for fn in feature_names:
        all_vals = [word_features[w][fn] for w in words]
        grand_mean = np.mean(all_vals)
        
        # 类间方差
        ss_between = 0
        ss_within = 0
        for cat in categories:
            cat_vals = [word_features[w][fn] for w in words if word_features[w]["category"] == cat]
            if len(cat_vals) < 2:
                continue
            cat_mean = np.mean(cat_vals)
            ss_between += len(cat_vals) * (cat_mean - grand_mean) ** 2
            ss_within += sum((v - cat_mean) ** 2 for v in cat_vals)
        
        f_ratio = ss_between / max(ss_within, 1e-10)
        feature_discrimination[fn] = float(f_ratio)
        print(f"    {fn}: F-ratio={f_ratio:.3f}")
    
    # 最具区分力的特征
    best_features = sorted(feature_discrimination.items(), key=lambda x: -x[1])[:3]
    print(f"\n  最具语义区分力的特征: {[f[0] for f in best_features]}")
    
    # 5. 线性回归: 末端delta_logit ≈ α*G_logit + β*A_logit
    from numpy.linalg import lstsq
    X_reg = np.column_stack([end_G, end_A, np.ones(len(end_G))])
    coeffs, residuals, _, _ = lstsq(X_reg, end_delta, rcond=None)
    r_squared = 1 - residuals[0] / np.sum((end_delta - end_delta.mean()) ** 2) if len(residuals) > 0 else 0
    
    print(f"\n  5. 线性回归: delta = α*G + β*A + c")
    print(f"     α={coeffs[0]:.4f}, β={coeffs[1]:.4f}, c={coeffs[2]:.4f}")
    print(f"     R-squared={r_squared:.4f}")
    
    # 6. 编码方程: logit(word) = f(category, adversarial_balance, diff_retention)
    # 尝试简单模型
    end_G_signs = np.array([word_features[w]["end_G_sign"] for w in words])
    end_A_signs = np.array([word_features[w]["end_A_sign"] for w in words])
    
    # G和A的末端符号模式
    sign_patterns = {}
    for w in words:
        gs = word_features[w]["end_G_sign"]
        as_ = word_features[w]["end_A_sign"]
        pattern = f"G{'+'if gs>0 else '-' if gs<0 else '0'}A{'+'if as_>0 else '-' if as_<0 else '0'}"
        if pattern not in sign_patterns:
            sign_patterns[pattern] = []
        sign_patterns[pattern].append(word_features[w]["category"])
    
    print(f"\n  6. 末端层G/A符号模式分布:")
    for pattern, cats in sorted(sign_patterns.items()):
        cat_counts = {}
        for c in cats:
            cat_counts[c] = cat_counts.get(c, 0) + 1
        print(f"    {pattern}: {len(cats)}词, {dict(sorted(cat_counts.items()))}")
    
    return {
        "corr_GA_delta": float(corr_GA_delta),
        "corr_adv_gacos": float(corr_adv_gacos),
        "corr_diff_opp": float(corr_diff_opp),
        "feature_discrimination": feature_discrimination,
        "regression_coeffs": coeffs.tolist(),
        "r_squared": float(r_squared),
        "sign_patterns": {k: v for k, v in sign_patterns.items()},
    }


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Phase CLXIII: 大规模语义编码规律分析")
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "deepseek7b", "glm4"],
                       help="模型名称")
    args = parser.parse_args()
    
    model_name = args.model
    print(f"\n{'='*60}")
    print(f"Phase CLXIII: 大规模语义编码规律分析(100词) — {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"  模型: {model_info.model_class}, {model_info.n_layers}层, d={model_info.d_model}")
    
    t0 = time.time()
    
    # P710
    p710_results = p710_extract_encoding_features(model, tokenizer, device, model_info)
    
    # P711
    p711_results = p711_clustering_analysis(p710_results)
    
    # P712
    p712_results = p712_unified_equation(p710_results, p711_results)
    
    elapsed = time.time() - t0
    print(f"\n总用时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "results/phase_clxiii"
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    output = {
        "phase": "CLXIII",
        "model": model_name,
        "timestamp": timestamp,
        "elapsed_seconds": elapsed,
        "n_words": len(p710_results),
        "model_info": {
            "class": model_info.model_class,
            "n_layers": model_info.n_layers,
            "d_model": model_info.d_model,
        },
        "p710_word_features": convert(p710_results),
        "p711_clustering": convert(p711_results),
        "p712_unified": convert(p712_results),
    }
    
    output_file = f"{output_dir}/phase_clxiii_{model_name}_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_file}")
    
    release_model(model)
    
    # 打印摘要
    print(f"\n{'='*60}")
    print("Phase CLXIII 摘要")
    print(f"{'='*60}")
    
    # 类别均值摘要
    print("\n语义类别编码特征均值:")
    categories = sorted(set(WORD_TO_CATEGORY.values()))
    for cat in categories:
        cat_words = [w for w in p710_results if p710_results[w]["category"] == cat]
        if not cat_words:
            continue
        adv = np.mean([p710_results[w]["adversarial_rate"] for w in cat_words])
        ga = np.mean([p710_results[w]["mean_GA_cos"] for w in cat_words])
        dr = np.mean([p710_results[w]["mean_diff_retention"] for w in cat_words])
        eg = np.mean([p710_results[w]["end_G_logit"] for w in cat_words])
        ea = np.mean([p710_results[w]["end_A_logit"] for w in cat_words])
        print(f"  {cat:10s}: adv={adv:.3f}, GA_cos={ga:.3f}, diff_ret={dr:.4f}, G={eg:.2f}, A={ea:.2f}")
    
    if p711_results:
        print(f"\n聚类 ARI: {p711_results.get('ari', 'N/A')}")
    if p712_results:
        print(f"回归 R-squared: {p712_results.get('r_squared', 'N/A')}")


if __name__ == "__main__":
    main()
