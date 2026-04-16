#!/usr/bin/env python3
"""
Phase CLVII: 从 Logit 空间反推 G 项的因果结构 — 破解编码机制
================================================================

核心目标: 回答"苹果的编码是什么"——一套编码同时承载语义、语法、效率

路线三核心思路:
  CLVI 证明 FFN 是超分布式计算(99% neuron 共享, Key/Value 无语义)
  → 在 neuron 空间找不到结构
  → 转向 logit 空间分析: G 项在 logit 空间的投影结构

关键约束: 编码必须同时满足
  1. 语义: apple vs banana vs cat 的 logit 指纹可区分
  2. 语法: 词性(名词/动词/形容词)在 logit 空间有规律
  3. 效率: 极少维度的 G 项承载大量区分信息

实验设计:
  P688: G 项的 logit 空间投影结构
    - 计算多个概念的 G 项, 投影到 W_U 空间
    - 分析 logit_G 的 PCA 维度, 概念指纹的独特性
  
  P689: logit 空间的概念区分度
    - apple/banana/cat/car 的 G 项在 logit 空间是否正交?
    - 语义距离 vs logit 距离的关系
  
  P690: neuron 贡献的幂律结构
    - logit_G[apple] = Σ_i contrib_i 的贡献分布
    - 是幂律(有结构) 还是指数衰减(无结构)?
  
  P691: 因果干预验证
    - 消融 top-K 贡献 neuron, 验证编码的可分解性

实验模型: qwen3 -> deepseek7b -> glm4 (串行, 避免OOM)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
import json
import time
import argparse
import gc
from datetime import datetime
from pathlib import Path
from scipy import stats
from scipy.sparse.linalg import svds
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from model_utils import load_model, get_model_info, get_layer_weights, release_model, MODEL_CONFIGS

import functools
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
print = functools.partial(print, flush=True)

OUT_DIR = Path("d:/develop/TransformerLens-main/results/phase_clvii")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path("d:/develop/TransformerLens-main/tests/glm5_temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ============ 测试词汇 ============
# 多层次语义: 同家族水果 / 跨家族动物+工具 / 语法混合(动词/形容词)
TEST_WORDS = {
    # 同家族 (水果) — 语义距离近
    "apple":    {"cat": "fruit", "pos": "noun",    "freq": "high"},
    "banana":   {"cat": "fruit", "pos": "noun",    "freq": "high"},
    "orange":   {"cat": "fruit", "pos": "noun",    "freq": "high"},
    # 跨家族 (动物) — 语义距离远
    "cat":      {"cat": "animal", "pos": "noun",   "freq": "high"},
    "dog":      {"cat": "animal", "pos": "noun",   "freq": "high"},
    # 跨家族 (工具) — 更远
    "car":      {"cat": "vehicle", "pos": "noun",  "freq": "high"},
    # 语法混合 — 测试编码是否承载语法信息
    "run":      {"cat": "action", "pos": "verb",   "freq": "high"},
    "red":      {"cat": "color", "pos": "adjective", "freq": "high"},
    "the":      {"cat": "function", "pos": "determiner", "freq": "very_high"},
}

# 语法角色词组 — 测试"苹果"的编码是否因语法角色变化
SYNTAX_TEMPLATES = [
    "I eat an {word}.",       # 宾语
    "The {word} is red.",     # 主语
    "This is a {word}.",      # 表语
    "{word} tastes sweet.",   # 主语(不同语境)
]


def extract_tensor(t):
    """从 tensor 中提取 last token 的 numpy 向量"""
    if t.dim() == 3:
        return t[0, -1, :].float().detach().cpu().numpy()
    elif t.dim() == 2:
        return t[-1, :].float().detach().cpu().numpy()
    return t.float().detach().cpu().numpy().flatten()


def get_layers(model):
    """获取模型的 transformer 层"""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    return []


def safe_sigmoid(x):
    """安全 sigmoid, 避免溢出"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def compute_G_term(model, tokenizer, word, layer_idx, device):
    """
    计算指定词在指定层的 G 项 (FFN 输出)
    
    返回:
      G: [d_model] — FFN 输出向量
      h_pre: [d_model] — FFN 输入 (resid_mid)
      gate_values: [d_mlp] — gate 值
      up_values: [d_mlp] — up 值
    """
    text = f"The word is {word}."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # hidden_states[li+1] = 第 li 层输出 (含 FFN)
    # hidden_states[li] = 第 li 层输入 (不含第 li 层)
    # G ≈ hidden_states[li+1] - hidden_states[li]  (粗近似, 含 Attention)
    # 更精确: 用权重直接计算 FFN
    
    h_pre = extract_tensor(outputs.hidden_states[layer_idx])      # 第 li 层输入
    h_post = extract_tensor(outputs.hidden_states[layer_idx + 1]) # 第 li 层输出
    
    return h_pre, h_post


# ================================================================
# P688: G 项的 logit 空间投影结构
# ================================================================
def p688_G_logit_projection(model, tokenizer, model_info, W_embed, W_U, device):
    """
    P688: G 项的 logit 空间投影结构
    
    核心问题: G 项在 logit 空间的投影有什么结构?
    
    算法:
    1. 对每个测试词, 计算每层的 G 项
    2. 将 G 投影到 W_U 空间: logit_G = W_U @ G / ||G||  (归一化余弦)
    3. 分析:
       a) logit_G 的 PCA 维度 — 多少维解释 95% 方差?
       b) 概念指纹的独特性 — 不同词的 logit_G 余弦相似度
       c) 语义/语法/效率的编码维度分配
    
    关键约束: 编码必须同时满足三种能力
       语义维度: 区分 apple vs banana vs cat
       语法维度: 编码词性(名词/动词/形容词)
       效率约束: 总维度 ≤ 25-32 (G 项内在维度)
    """
    print("\n" + "="*70)
    print("P688: G 项的 logit 空间投影结构")
    print("="*70)
    
    layers = get_layers(model)
    n_layers = len(layers)
    d_model = model_info.d_model
    d_mlp = model_info.intermediate_size
    mlp_type = model_info.mlp_type
    
    # 关键层: 前4层 + 中间 + 最后2层
    key_layers = list(range(min(4, n_layers))) + [n_layers // 2, n_layers - 2, n_layers - 1]
    key_layers = sorted(set([l for l in key_layers if l < n_layers]))
    
    # 获取各词的 token id
    word_token_ids = {}
    for word in TEST_WORDS:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids:
            word_token_ids[word] = ids[0]
    
    # 预计算 W_U 的 SVD (用于投影分析)
    n_svd = min(100, min(W_U.shape) - 2)
    svd = TruncatedSVD(n_components=n_svd)
    svd.fit(W_U)
    U_svd = svd.components_  # [n_svd, d_model]
    S_svd = svd.singular_values_
    explained_var = svd.explained_variance_ratio_
    
    print(f"  W_U SVD: top-1 explains {explained_var[0]*100:.1f}%, "
          f"top-10 explains {sum(explained_var[:10])*100:.1f}%, "
          f"top-50 explains {sum(explained_var[:50])*100:.1f}%")
    
    results = {}
    
    for li in key_layers:
        t0 = time.time()
        lw = get_layer_weights(layers[li], d_model, mlp_type)
        W_gate = lw.W_gate
        W_up = lw.W_up
        W_down = lw.W_down
        
        if W_gate is None or W_up is None or W_down is None:
            print(f"  L{li}: Missing weight matrices, skip")
            continue
        
        layer_result = {
            "layer": li,
            "word_G_norms": {},
            "word_logit_G_norms": {},
            "logit_G_cosine_matrix": {},
            "logit_G_pca_dims": {},
            "semantic_grammar_efficiency": {},
        }
        
        # 收集各词的 G 项和 logit_G
        all_G = {}  # word -> [d_model]
        all_logit_G = {}  # word -> [vocab_size] (归一化)
        
        for word in TEST_WORDS:
            h_pre, h_post = compute_G_term(model, tokenizer, word, li, device)
            G = h_post - h_pre  # 近似 G 项
            
            # 精确计算 FFN 输出
            gate_val = safe_sigmoid(W_gate @ h_pre)
            up_val = W_up @ h_pre
            post_val = gate_val * up_val
            G_exact = W_down @ post_val
            
            all_G[word] = G_exact
            G_norm = np.linalg.norm(G_exact)
            
            # 投影到 logit 空间
            logit_G = W_U @ G_exact  # [vocab_size]
            logit_G_norm = np.linalg.norm(logit_G)
            
            # 归一化 logit_G (用于余弦比较)
            if logit_G_norm > 1e-10:
                logit_G_normalized = logit_G / logit_G_norm
            else:
                logit_G_normalized = logit_G
            
            all_logit_G[word] = logit_G_normalized
            
            layer_result["word_G_norms"][word] = float(G_norm)
            layer_result["word_logit_G_norms"][word] = float(logit_G_norm)
            
            # G 在 W_U SVD 方向上的投影
            G_svd_coeffs = U_svd @ G_exact  # [n_svd]
            G_svd_energy = np.sum(G_svd_coeffs**2)
            top10_energy = np.sum(np.sort(G_svd_coeffs**2)[-10:])
            top30_energy = np.sum(np.sort(G_svd_coeffs**2)[-30:])
            
            layer_result[f"G_svd_{word}"] = {
                "top10_ratio": float(top10_energy / max(G_svd_energy, 1e-10)),
                "top30_ratio": float(top30_energy / max(G_svd_energy, 1e-10)),
                "G_svd_energy": float(G_svd_energy),
            }
        
        # ---- 概念间的余弦相似度矩阵 ----
        words_list = list(all_logit_G.keys())
        n_words = len(words_list)
        cos_matrix = np.zeros((n_words, n_words))
        for i in range(n_words):
            for j in range(n_words):
                cos_matrix[i, j] = np.dot(all_logit_G[words_list[i]], all_logit_G[words_list[j]])
        
        layer_result["logit_G_cosine_matrix"] = {
            "words": words_list,
            "matrix": [[float(cos_matrix[i, j]) for j in range(n_words)] for i in range(n_words)]
        }
        
        # ---- logit_G 的 PCA 维度 ----
        logit_G_matrix = np.array([all_logit_G[w] for w in words_list])  # [n_words, vocab_size]
        # PCA 在词间做 (词少, 维度高)
        if n_words > 2:
            pca = PCA()
            pca.fit(logit_G_matrix)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            dim_95 = int(np.searchsorted(cumvar, 0.95)) + 1
            dim_99 = int(np.searchsorted(cumvar, 0.99)) + 1
        else:
            dim_95 = dim_99 = n_words
        
        layer_result["logit_G_pca_dims"] = {
            "dim_95": dim_95,
            "dim_99": dim_99,
            "top1_var": float(pca.explained_variance_ratio_[0]) if n_words > 1 else 1.0,
        }
        
        # ---- 语义/语法/效率维度分配 ----
        # 语义区分: 同家族内 cos (apple-banana) vs 跨家族 cos (apple-cat)
        fruit_cos = [cos_matrix[words_list.index("apple"), words_list.index("banana")]]
        cross_cos = [cos_matrix[words_list.index("apple"), words_list.index("cat")],
                     cos_matrix[words_list.index("apple"), words_list.index("car")]]
        grammar_cos = [cos_matrix[words_list.index("apple"), words_list.index("run")],
                       cos_matrix[words_list.index("apple"), words_list.index("red")]]
        function_cos = [cos_matrix[words_list.index("apple"), words_list.index("the")]]
        
        layer_result["semantic_grammar_efficiency"] = {
            "same_family_cos_mean": float(np.mean(fruit_cos)),
            "cross_family_cos_mean": float(np.mean(cross_cos)),
            "cross_grammar_cos_mean": float(np.mean(grammar_cos)),
            "function_word_cos_mean": float(np.mean(function_cos)),
            "semantic_separation": float(np.mean(cross_cos) - np.mean(fruit_cos)),  # 正=好区分
        }
        
        elapsed = time.time() - t0
        print(f"  L{li}: G_norm(apple)={layer_result['word_G_norms'].get('apple', 0):.2f}, "
              f"logit_G_pca_dim95={dim_95}, "
              f"same_family_cos={np.mean(fruit_cos):.3f}, "
              f"cross_family_cos={np.mean(cross_cos):.3f}, "
              f"separation={np.mean(cross_cos)-np.mean(fruit_cos):.3f}, "
              f"elapsed={elapsed:.1f}s")
        
        results[f"L{li}"] = layer_result
        
        del all_G, all_logit_G, logit_G_matrix
        gc.collect()
    
    return results


# ================================================================
# P689: logit 空间的概念区分度
# ================================================================
def p689_logit_concept_separation(model, tokenizer, model_info, W_embed, W_U, device):
    """
    P689: logit 空间的概念区分度
    
    核心问题: 不同概念的 G 项在 logit 空间是否正交?
    
    更深入的分析:
    1. apple 的 logit_G 激活了哪些词的 logit? → "苹果的语义邻居"
    2. 这些邻居是否构成一个语义一致的集合?
    3. 语义距离 vs logit 距离的定量关系
    
    关键洞察: 如果编码同时承载语义+语法+效率, 那么:
    - 语义: apple 的 logit_G 应该激活水果相关词
    - 语法: apple 的 logit_G 应该激活名词相关位置
    - 效率: 这些激活应该集中在少数维度
    """
    print("\n" + "="*70)
    print("P689: logit 空间的概念区分度")
    print("="*70)
    
    layers = get_layers(model)
    n_layers = len(layers)
    d_model = model_info.d_model
    d_mlp = model_info.intermediate_size
    mlp_type = model_info.mlp_type
    
    # 集中分析最后2层 (最终编码层)
    analysis_layers = [max(0, n_layers - 2), n_layers - 1]
    if n_layers > 4:
        analysis_layers += [1, n_layers // 2]
    analysis_layers = sorted(set(analysis_layers))
    
    results = {}
    
    for li in analysis_layers:
        t0 = time.time()
        lw = get_layer_weights(layers[li], d_model, mlp_type)
        W_gate = lw.W_gate
        W_up = lw.W_up
        W_down = lw.W_down
        
        if W_gate is None or W_up is None or W_down is None:
            continue
        
        layer_result = {
            "layer": li,
            "word_top_logit_words": {},
            "word_self_logit": {},
            "concept_separation_ratios": {},
            "semantic_neighbor_analysis": {},
        }
        
        for word in TEST_WORDS:
            h_pre, h_post = compute_G_term(model, tokenizer, word, li, device)
            
            # 精确 FFN 输出
            gate_val = safe_sigmoid(W_gate @ h_pre)
            up_val = W_up @ h_pre
            G_exact = W_down @ (gate_val * up_val)
            
            # logit_G = G 在 logit 空间的效果
            logit_G = W_U @ G_exact  # [vocab_size]
            
            # 1) 激活了哪些词的 logit? (top-20)
            top20_idx = np.argsort(logit_G)[-20:][::-1]
            top20_words = [tokenizer.decode([int(j)]) for j in top20_idx]
            top20_vals = [float(logit_G[j]) for j in top20_idx]
            
            layer_result["word_top_logit_words"][word] = [
                {"word": w, "logit_contribution": float(v)}
                for w, v in zip(top20_words, top20_vals)
            ]
            
            # 2) 自身 logit 贡献
            word_id = tokenizer.encode(word, add_special_tokens=False)
            if word_id:
                layer_result["word_self_logit"][word] = float(logit_G[word_id[0]])
            
            # 3) 语义邻居分析 — G 项激活的词是否与源词语义相关?
            #    检查 top-100 中有多少是同家族词
            top100_idx = np.argsort(logit_G)[-100:][::-1]
            
            # 定义语义相关的词组 (手工标注, 避免分词问题)
            semantic_groups = {
                "fruit": ["apple", "banana", "orange", "fruit", "grape", "pear", "peach", "mango"],
                "animal": ["cat", "dog", "animal", "pet", "bird", "fish", "horse"],
                "vehicle": ["car", "vehicle", "truck", "bus", "bike", "train"],
                "color": ["red", "blue", "green", "color", "yellow", "white", "black"],
                "action": ["run", "walk", "move", "go", "eat", "drink"],
            }
            
            word_cat = TEST_WORDS[word]["cat"]
            same_group = semantic_groups.get(word_cat, [])
            
            # 统计 top-100 中属于同组的比例
            n_same_group = 0
            n_checked = 0
            for idx in top100_idx:
                decoded = tokenizer.decode([int(idx)]).strip().lower()
                if decoded in same_group:
                    n_same_group += 1
                n_checked += 1
            
            layer_result["semantic_neighbor_analysis"][word] = {
                "same_group_in_top100": n_same_group,
                "top1_word": top20_words[0] if top20_words else "",
                "top1_value": top20_vals[0] if top20_vals else 0,
                "top10_words": top20_words[:10],
            }
        
        # 4) 概念区分比: 组间距离 / 组内距离
        # 计算 G 项的余弦相似度
        all_G = {}
        for word in TEST_WORDS:
            h_pre, h_post = compute_G_term(model, tokenizer, word, li, device)
            gate_val = safe_sigmoid(W_gate @ h_pre)
            up_val = W_up @ h_pre
            G = W_down @ (gate_val * up_val)
            G_norm = np.linalg.norm(G)
            if G_norm > 1e-10:
                all_G[word] = G / G_norm
            else:
                all_G[word] = G
        
        # 组内 cos (水果间)
        fruits = ["apple", "banana", "orange"]
        animals = ["cat", "dog"]
        within_fruit = []
        for i in range(len(fruits)):
            for j in range(i+1, len(fruits)):
                if fruits[i] in all_G and fruits[j] in all_G:
                    within_fruit.append(float(np.dot(all_G[fruits[i]], all_G[fruits[j]])))
        
        # 组间 cos (水果-动物)
        between = []
        for f in fruits:
            for a in animals:
                if f in all_G and a in all_G:
                    between.append(float(np.dot(all_G[f], all_G[a])))
        
        separation_ratio = float(np.mean(between)) / max(float(np.mean(within_fruit)), 1e-10) if within_fruit else 0
        
        layer_result["concept_separation_ratios"] = {
            "within_fruit_cos_mean": float(np.mean(within_fruit)) if within_fruit else 0,
            "between_fruit_animal_cos_mean": float(np.mean(between)) if between else 0,
            "separation_ratio": separation_ratio,  # <1 = 好区分 (组内 > 组间)
        }
        
        elapsed = time.time() - t0
        wfc = float(np.mean(within_fruit)) if within_fruit else 0
        bfc = float(np.mean(between)) if between else 0
        print(f"  L{li}: within_fruit_cos={wfc:.3f}, "
              f"between_fruit_animal_cos={bfc:.3f}, "
              f"separation_ratio={separation_ratio:.3f}, "
              f"elapsed={elapsed:.1f}s")
        
        results[f"L{li}"] = layer_result
        
        del all_G
        gc.collect()
    
    return results


# ================================================================
# P690: neuron 贡献的幂律结构
# ================================================================
def p690_neuron_power_law(model, tokenizer, model_info, W_embed, W_U, device):
    """
    P690: neuron 贡献的幂律结构
    
    核心问题: logit_G[apple] = Σ_i contrib_i 中, 
    contrib_i 的分布是幂律(有结构)还是指数衰减(无结构)?
    
    如果幂律: 少数 neuron 贡献大部分 logit → 编码有结构可解
    如果指数: 所有 neuron 贡献等权 → 编码不可分解
    
    更深入:
    1. 计算每个 neuron 对 logit(apple) 的精确贡献
    2. 排序后画累积分布 — 是否存在"肘点"?
    3. 拟合幂律 vs 指数, 看哪个更好
    
    关键: 这是回答"编码效率"的核心实验
    如果幂律指数 < 2 → 极少数 neuron 承载大部分信息 → 编码高度压缩
    如果幂律指数 > 4 或指数衰减 → 编码是分散的
    """
    print("\n" + "="*70)
    print("P690: neuron 贡献的幂律结构")
    print("="*70)
    
    layers = get_layers(model)
    n_layers = len(layers)
    d_model = model_info.d_model
    d_mlp = model_info.intermediate_size
    mlp_type = model_info.mlp_type
    
    # 集中分析最后2层 + 黄金层(L1)
    analysis_layers = [1, n_layers // 2, max(0, n_layers - 2), n_layers - 1]
    analysis_layers = sorted(set([l for l in analysis_layers if l < n_layers]))
    
    results = {}
    
    for li in analysis_layers:
        t0 = time.time()
        lw = get_layer_weights(layers[li], d_model, mlp_type)
        W_gate = lw.W_gate
        W_up = lw.W_up
        W_down = lw.W_down
        
        if W_gate is None or W_up is None or W_down is None:
            continue
        
        layer_result = {
            "layer": li,
            "words": {},
        }
        
        for word in ["apple", "banana", "cat", "run"]:
            h_pre, h_post = compute_G_term(model, tokenizer, word, li, device)
            
            # 精确计算 gate, up, post
            gate_val = safe_sigmoid(W_gate @ h_pre)
            up_val = W_up @ h_pre
            post_val = gate_val * up_val  # [d_mlp]
            
            # 每个neuron对 logit(word) 的贡献
            word_id = tokenizer.encode(word, add_special_tokens=False)
            if not word_id:
                continue
            word_id = word_id[0]
            
            # W_U[word_id] @ W_down[:,i] * post_val[i] = neuron i 对 logit(word) 的贡献
            W_U_word = W_U[word_id]  # [d_model]
            
            # 分块计算避免内存问题
            CHUNK = 2000
            n_neurons = W_down.shape[1]
            contributions = np.zeros(n_neurons)
            
            for chunk_start in range(0, n_neurons, CHUNK):
                chunk_end = min(chunk_start + CHUNK, n_neurons)
                W_down_chunk = W_down[:, chunk_start:chunk_end]  # [d_model, chunk]
                # neuron i 对 logit(word) 的贡献 = (W_U_word · W_down[:,i]) * post_val[i]
                dot_products = W_U_word @ W_down_chunk  # [chunk]
                contributions[chunk_start:chunk_end] = dot_products * post_val[chunk_start:chunk_end]
            
            # 排序贡献 (绝对值)
            abs_contributions = np.abs(contributions)
            sorted_contrib = np.sort(abs_contributions)[::-1]
            total_contrib = np.sum(abs_contributions)
            
            if total_contrib < 1e-10:
                continue
            
            # 累积贡献
            cumsum = np.cumsum(sorted_contrib) / total_contrib
            
            # 找肘点: 累积贡献达到 50%, 90%, 99% 需要多少 neuron?
            n_50 = int(np.searchsorted(cumsum, 0.50)) + 1
            n_90 = int(np.searchsorted(cumsum, 0.90)) + 1
            n_99 = int(np.searchsorted(cumsum, 0.99)) + 1
            
            # 拟合幂律: log(rank) vs log(contribution)
            ranks = np.arange(1, len(sorted_contrib) + 1)
            mask = sorted_contrib > 0
            if np.sum(mask) > 10:
                log_ranks = np.log(ranks[mask])
                log_contribs = np.log(sorted_contrib[mask])
                
                # 线性拟合 log(contrib) = a * log(rank) + b
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    log_ranks[:min(1000, len(log_ranks))], 
                    log_contribs[:min(1000, len(log_contribs))]
                )
                power_law_exponent = -slope  # 负斜率取绝对值
                power_law_r2 = r_value**2
                
                # 也拟合指数衰减: log(contrib) = a * rank + b
                exp_ranks = ranks[mask].astype(float)
                exp_slope, exp_intercept, exp_r_value, _, _ = stats.linregress(
                    exp_ranks[:min(1000, len(exp_ranks))],
                    log_contribs[:min(1000, len(log_contribs))]
                )
                exp_r2 = exp_r_value**2
            else:
                power_law_exponent = 0
                power_law_r2 = 0
                exp_r2 = 0
            
            # 符号分析: 正贡献 vs 负贡献
            n_positive = int(np.sum(contributions > 0))
            n_negative = int(np.sum(contributions < 0))
            sum_positive = float(np.sum(contributions[contributions > 0]))
            sum_negative = float(np.sum(contributions[contributions < 0]))
            
            # Top-10 贡献 neuron 的详细信息
            top10_idx = np.argsort(abs_contributions)[-10:][::-1]
            top10_info = []
            for ni in top10_idx:
                # 计算该neuron在排序列表中的位置 (cumulative ratio)
                rank = int(np.sum(abs_contributions >= abs_contributions[ni])) - 1
                cum_ratio = float(cumsum[min(rank, len(cumsum)-1)])
                top10_info.append({
                    "neuron": int(ni),
                    "contribution": float(contributions[ni]),
                    "abs_contribution": float(abs_contributions[ni]),
                    "gate": float(gate_val[ni]),
                    "up": float(up_val[ni]),
                    "cumulative_ratio": cum_ratio,
                })
            
            word_result = {
                "total_neurons": n_neurons,
                "n_for_50pct": n_50,
                "n_for_90pct": n_90,
                "n_for_99pct": n_99,
                "ratio_50pct": float(n_50 / n_neurons),
                "ratio_90pct": float(n_90 / n_neurons),
                "power_law_exponent": float(power_law_exponent),
                "power_law_r2": float(power_law_r2),
                "exp_decay_r2": float(exp_r2),
                "better_fit": "power_law" if power_law_r2 > exp_r2 else "exponential",
                "n_positive_contrib": n_positive,
                "n_negative_contrib": n_negative,
                "sum_positive": sum_positive,
                "sum_negative": sum_negative,
                "top10_neurons": top10_info,
                "self_logit_contribution": float(np.sum(contributions)),  # Σcontrib_i = logit(word) from G
            }
            
            layer_result["words"][word] = word_result
        
        elapsed = time.time() - t0
        apple_info = layer_result["words"].get("apple", {})
        print(f"  L{li}: apple n_50={apple_info.get('n_for_50pct', '?')}, "
              f"n_90={apple_info.get('n_for_90pct', '?')}, "
              f"power_law_exp={apple_info.get('power_law_exponent', 0):.2f} "
              f"(R²={apple_info.get('power_law_r2', 0):.3f}), "
              f"better_fit={apple_info.get('better_fit', '?')}, "
              f"elapsed={elapsed:.1f}s")
        
        results[f"L{li}"] = layer_result
        
        del contributions, abs_contributions, sorted_contrib
        gc.collect()
    
    return results


# ================================================================
# P691: 因果干预验证
# ================================================================
def p691_causal_intervention(model, tokenizer, model_info, W_embed, W_U, device):
    """
    P691: 因果干预验证
    
    核心问题: 消融 top-K 贡献 neuron, logit(apple) 如何变化?
    
    实验:
    1. 找到 apple 在最后层的 top-K 贡献 neuron
    2. 消融这些 neuron (gate_i = 0), 重新计算 logit(apple)
    3. 对比随机消融 K 个 neuron 的效果
    4. 如果 top-K 消融 >> 随机消融: 编码有结构
       如果差异小: 分布式编码是真实的
    
    进阶: 
    5. 交叉消融: 用 banana 的 top-K 消融, 看 apple 的 logit 变化
       如果交叉消融也影响 apple: 共享 neuron 的权重不同是关键
       如果交叉消融不影响 apple: 存在词特异 neuron
    """
    print("\n" + "="*70)
    print("P691: 因果干预验证")
    print("="*70)
    
    layers = get_layers(model)
    n_layers = len(layers)
    d_model = model_info.d_model
    d_mlp = model_info.intermediate_size
    mlp_type = model_info.mlp_type
    
    # 只分析最后1层 (最终编码层)
    li = n_layers - 1
    lw = get_layer_weights(layers[li], d_model, mlp_type)
    W_gate = lw.W_gate
    W_up = lw.W_up
    W_down = lw.W_down
    
    if W_gate is None or W_up is None or W_down is None:
        print("  Missing weight matrices, skip")
        return {}
    
    n_neurons = W_gate.shape[0]
    
    results = {
        "layer": li,
        "n_neurons": n_neurons,
        "interventions": {},
    }
    
    for target_word in ["apple", "banana", "cat"]:
        h_pre, _ = compute_G_term(model, tokenizer, target_word, li, device)
        
        # 计算 gate, up, post
        gate_val = safe_sigmoid(W_gate @ h_pre)
        up_val = W_up @ h_pre
        post_val = gate_val * up_val
        
        # 完整 FFN 输出
        G_full = W_down @ post_val
        logit_full = W_U @ G_full
        
        # 基线 logit
        word_id = tokenizer.encode(target_word, add_special_tokens=False)
        if not word_id:
            continue
        word_id = word_id[0]
        baseline_logit = float(logit_full[word_id])
        
        # 每个 neuron 对 logit(target_word) 的贡献
        W_U_word = W_U[word_id]
        contributions = np.zeros(n_neurons)
        CHUNK = 2000
        for chunk_start in range(0, n_neurons, CHUNK):
            chunk_end = min(chunk_start + CHUNK, n_neurons)
            W_down_chunk = W_down[:, chunk_start:chunk_end]
            dot_products = W_U_word @ W_down_chunk
            contributions[chunk_start:chunk_end] = dot_products * post_val[chunk_start:chunk_end]
        
        abs_contributions = np.abs(contributions)
        top_neurons = np.argsort(abs_contributions)[::-1]
        
        # 消融实验: K = 10, 50, 100, 500
        K_values = [10, 50, 100, 500]
        intervention_result = {
            "baseline_logit": baseline_logit,
            "ablating_own_top_k": {},
            "ablating_random_k": {},
            "ablating_other_word_top_k": {},
        }
        
        for K in K_values:
            if K > n_neurons:
                continue
            
            # ---- 1) 消融自身 top-K ----
            ablate_neurons = top_neurons[:K]
            post_ablated = post_val.copy()
            post_ablated[ablate_neurons] = 0
            G_ablated = W_down @ post_ablated
            logit_ablated = W_U @ G_ablated
            logit_after = float(logit_ablated[word_id])
            delta_logit = logit_after - baseline_logit
            
            intervention_result["ablating_own_top_k"][str(K)] = {
                "logit_after": logit_after,
                "delta_logit": delta_logit,
                "delta_ratio": delta_logit / max(abs(baseline_logit), 1e-10),
            }
            
            # ---- 2) 随机消融 K 个 ----
            rng = np.random.RandomState(42)
            random_neurons = rng.choice(n_neurons, size=K, replace=False)
            post_random = post_val.copy()
            post_random[random_neurons] = 0
            G_random = W_down @ post_random
            logit_random = W_U @ G_random
            random_logit_after = float(logit_random[word_id])
            random_delta = random_logit_after - baseline_logit
            
            intervention_result["ablating_random_k"][str(K)] = {
                "logit_after": random_logit_after,
                "delta_logit": random_delta,
                "delta_ratio": random_delta / max(abs(baseline_logit), 1e-10),
            }
        
        # ---- 3) 交叉消融: 用 banana/cat 的 top-K 消融 apple ----
        if target_word == "apple":
            for other_word in ["banana", "cat"]:
                h_pre_other, _ = compute_G_term(model, tokenizer, other_word, li, device)
                gate_other = safe_sigmoid(W_gate @ h_pre_other)
                up_other = W_up @ h_pre_other
                post_other = gate_other * up_other
                
                contributions_other = np.zeros(n_neurons)
                other_id = tokenizer.encode(other_word, add_special_tokens=False)
                if not other_id:
                    continue
                other_id = other_id[0]
                W_U_other = W_U[other_id]
                for chunk_start in range(0, n_neurons, CHUNK):
                    chunk_end = min(chunk_start + CHUNK, n_neurons)
                    W_down_chunk = W_down[:, chunk_start:chunk_end]
                    dot_products = W_U_other @ W_down_chunk
                    contributions_other[chunk_start:chunk_end] = dot_products * post_other[chunk_start:chunk_end]
                
                other_top_neurons = np.argsort(np.abs(contributions_other))[::-1]
                
                for K in [10, 50, 100]:
                    if K > n_neurons:
                        continue
                    # 用 other_word 的 top-K 消融, 但测量 apple 的 logit
                    cross_ablate = other_top_neurons[:K]
                    post_cross = post_val.copy()
                    post_cross[cross_ablate] = 0
                    G_cross = W_down @ post_cross
                    logit_cross = W_U @ G_cross
                    cross_logit_after = float(logit_cross[word_id])
                    cross_delta = cross_logit_after - baseline_logit
                    
                    intervention_result["ablating_other_word_top_k"][f"{other_word}_K{K}"] = {
                        "logit_after": cross_logit_after,
                        "delta_logit": cross_delta,
                        "delta_ratio": cross_delta / max(abs(baseline_logit), 1e-10),
                    }
        
        results["interventions"][target_word] = intervention_result
        print(f"  {target_word}: baseline_logit={baseline_logit:.2f}, "
              f"ablate_top10_delta={intervention_result['ablating_own_top_k'].get('10', {}).get('delta_ratio', 0):.3f}, "
              f"random10_delta={intervention_result['ablating_random_k'].get('10', {}).get('delta_ratio', 0):.3f}")
    
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
    print(f"Phase CLVII: Logit 空间编码结构分析 — {model_name}")
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
    
    # P688: G 项的 logit 空间投影结构
    all_results["P688"] = p688_G_logit_projection(model, tokenizer, model_info, W_embed, W_U, device)
    gc.collect()
    
    # P689: logit 空间的概念区分度
    all_results["P689"] = p689_logit_concept_separation(model, tokenizer, model_info, W_embed, W_U, device)
    gc.collect()
    
    # P690: neuron 贡献的幂律结构
    all_results["P690"] = p690_neuron_power_law(model, tokenizer, model_info, W_embed, W_U, device)
    gc.collect()
    
    # P691: 因果干预验证
    all_results["P691"] = p691_causal_intervention(model, tokenizer, model_info, W_embed, W_U, device)
    gc.collect()
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUT_DIR / f"phase_clvii_{model_name}_{timestamp}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {out_file}")
    
    # 打印关键发现摘要
    print("\n" + "="*70)
    print("关键发现摘要")
    print("="*70)
    print(f"\n模型: {model_name}")
    
    # P688 摘要
    if "P688" in all_results:
        print("\n--- P688: G 项 logit 投影 ---")
        for layer_key, lr in all_results["P688"].items():
            if isinstance(lr, dict) and "semantic_grammar_efficiency" in lr:
                se = lr["semantic_grammar_efficiency"]
                print(f"  {layer_key}: same_family_cos={se.get('same_family_cos_mean', 0):.3f}, "
                      f"cross_family_cos={se.get('cross_family_cos_mean', 0):.3f}, "
                      f"separation={se.get('semantic_separation', 0):.3f}")
    
    # P689 摘要
    if "P689" in all_results:
        print("\n--- P689: 概念区分度 ---")
        for layer_key, lr in all_results["P689"].items():
            if isinstance(lr, dict) and "concept_separation_ratios" in lr:
                cs = lr["concept_separation_ratios"]
                print(f"  {layer_key}: within_fruit={cs.get('within_fruit_cos_mean', 0):.3f}, "
                      f"between_fruit_animal={cs.get('between_fruit_animal_cos_mean', 0):.3f}, "
                      f"ratio={cs.get('separation_ratio', 0):.3f}")
    
    # P690 摘要
    if "P690" in all_results:
        print("\n--- P690: 幂律结构 ---")
        for layer_key, lr in all_results["P690"].items():
            if isinstance(lr, dict) and "words" in lr:
                for word, wr in lr["words"].items():
                    print(f"  {layer_key} {word}: n_50={wr.get('n_for_50pct', '?')}, "
                          f"power_exp={wr.get('power_law_exponent', 0):.2f} "
                          f"(R²={wr.get('power_law_r2', 0):.3f}), "
                          f"fit={wr.get('better_fit', '?')}")
    
    # P691 摘要
    if "P691" in all_results:
        print("\n--- P691: 因果干预 ---")
        p691 = all_results["P691"]
        if "interventions" in p691:
            for word, iv in p691["interventions"].items():
                top10 = iv.get("ablating_own_top_k", {}).get("10", {})
                rand10 = iv.get("ablating_random_k", {}).get("10", {})
                print(f"  {word}: baseline={iv.get('baseline_logit', 0):.2f}, "
                      f"top10_delta_ratio={top10.get('delta_ratio', 0):.3f}, "
                      f"random10_delta_ratio={rand10.get('delta_ratio', 0):.3f}")
    
    # 释放模型
    release_model(model)
    print("[model_utils] GPU memory released")


if __name__ == "__main__":
    main()
