#!/usr/bin/env python3
"""
Phase CLVI: FFN Key-Value 语义图谱 - 破解编码机制 (P684-P687)
================================================================

核心目标: 从"数据统计"转向"机制破解" — 回答"苹果为什么这样编码"

P684: FFN Key 语义解码 — W_gate[i]匹配什么输入?
P685: FFN Value 方向追踪 — W_down[:,i]输出什么方向?
P686: Gate 触发条件解析 — 为什么 gate_i(apple) 大?
P687: 苹果的精确编码方程 — 综合写出 h(apple) 的分解

FFN 数学形式 (GatedMLP):
  FFN(x) = (σ(x @ W_gate) ⊙ (x @ W_up)) @ W_down
  
展开为逐neuron:
  FFN(x) = Σ_i  gate_i(x) · up_i(x) · W_down[:,i]
  
其中:
  gate_i(x) = σ(W_gate[i] @ x)    — 第i个neuron的gate值
  up_i(x)   = W_up[i] @ x          — 第i个neuron的up值
  W_down[:,i]                       — 第i个neuron的输出方向

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

OUT_DIR = Path("d:/develop/TransformerLens-main/results/phase_clvi")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path("d:/develop/TransformerLens-main/tests/glm5_temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ============ 测试词汇 ============
# 核心词: 苹果, 对比词: 同家族(香蕉/橘子) + 跨家族(猫/车)
TARGET_WORDS = {
    "apple": {"family": "fruit", "attrs": ["red", "sweet", "round"]},
    "banana": {"family": "fruit", "attrs": ["yellow", "sweet", "long"]},
    "orange": {"family": "fruit", "attrs": ["orange", "sour", "round"]},
    "cat": {"family": "animal", "attrs": ["furry", "small", "alive"]},
    "car": {"family": "vehicle", "attrs": ["metal", "fast", "mechanical"]},
    "book": {"family": "object", "attrs": ["paper", "informative", "flat"]},
    "river": {"family": "nature", "attrs": ["water", "flowing", "wide"]},
}

# 属性词
ATTRIBUTE_WORDS = ["red", "sweet", "round", "yellow", "sour", "long", "furry", "small",
                   "alive", "metal", "fast", "mechanical", "paper", "informative", "flat",
                   "water", "flowing", "wide", "green", "blue", "big", "soft", "hard",
                   "hot", "cold", "delicious", "beautiful", "dangerous", "important", "old"]

# 测试文本模板
TEXT_TEMPLATES = [
    "The word is {word}.",
    "I like {word}.",
    "{word} is a common thing.",
    "Tell me about {word}.",
    "What is {word}?",
]


def get_W_U(model):
    """获取 unembedding 矩阵 [vocab_size, d_model]"""
    if hasattr(model, 'lm_head'):
        return model.lm_head.weight.detach().cpu().float().numpy()
    return model.get_output_embeddings().weight.detach().cpu().float().numpy()


def get_W_embed(model):
    """获取 embedding 矩阵 [vocab_size, d_model]"""
    embed = model.get_input_embeddings()
    return embed.weight.detach().cpu().float().numpy()


def get_layers(model):
    """获取 transformer 层列表"""
    if hasattr(model.model, 'layers'):
        return model.model.layers
    return []


def extract_tensor(t):
    """从 tensor 中提取 last token 的 numpy 向量"""
    if t.dim() == 3:
        return t[0, -1, :].float().detach().cpu().numpy()
    elif t.dim() == 2:
        return t[-1, :].float().detach().cpu().numpy()
    return t.float().detach().cpu().numpy().flatten()


def get_word_embedding(W_embed, tokenizer, word):
    """获取词的 embedding 向量(取第一个 token)"""
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    if len(token_ids) == 0:
        return None, None
    token_id = token_ids[0]
    return W_embed[token_id], token_id


def compute_cos_batch(vec, matrix):
    """计算向量与矩阵每行的余弦相似度 [n_rows]"""
    vec_norm = np.linalg.norm(vec)
    if vec_norm < 1e-10:
        return np.zeros(matrix.shape[0])
    matrix_norms = np.linalg.norm(matrix, axis=1)
    matrix_norms = np.maximum(matrix_norms, 1e-10)
    return (matrix @ vec) / (matrix_norms * vec_norm)


# ================================================================
# P684: FFN Key 语义解码
# ================================================================
def p684_ffn_key_semantic_decoding(model, tokenizer, model_info, W_embed, W_U):
    """
    P684: FFN Key 语义解码
    
    核心问题: W_gate[i] (key) 匹配什么输入?
    
    算法原理:
    1. W_gate 形状: [d_mlp, d_model] (HuggingFace) — 每行是一个 key
    2. W_embed 形状: [vocab_size, d_model] — 每行是一个词的 embedding
    3. key_i 与所有词 embedding 的余弦相似度 → 找到最匹配的词
    4. 如果 top-1 词有清晰语义 (如"水果"), 则 key 是语义检测器
    5. 如果 top-k 词语义分散, 则 key 是分布式模式匹配器
    
    为节省计算, 只分析采样层
    """
    print("\n" + "="*70)
    print("P684: FFN Key 语义解码")
    print("="*70)
    
    layers = get_layers(model)
    n_layers = len(layers)
    d_model = model_info.d_model
    d_mlp = model_info.intermediate_size
    mlp_type = model_info.mlp_type
    
    # 采样层: 首/尾/中间均匀
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    results = {}
    vocab_size = W_embed.shape[0]
    
    # 预选: 只对高频词做匹配 (前5000词, 覆盖大部分常用词)
    # 优化: 只归一化需要的部分, 避免大矩阵 (GLM4: 151552x4096) 的完整归一化
    n_check = min(5000, vocab_size)
    W_embed_check = W_embed[:n_check]
    W_embed_check_norms = np.linalg.norm(W_embed_check, axis=1, keepdims=True)
    W_embed_check_norms = np.maximum(W_embed_check_norms, 1e-10)
    W_embed_check_normalized = W_embed_check / W_embed_check_norms
    del W_embed_check  # 释放临时内存
    
    # 为目标词和属性词获取 token_id
    target_token_ids = {}
    for word in list(TARGET_WORDS.keys()) + ATTRIBUTE_WORDS:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids:
            target_token_ids[word] = ids[0]
    
    for li in sample_layers:
        t0 = time.time()
        lw = get_layer_weights(layers[li], d_model, mlp_type)
        W_gate = lw.W_gate  # [d_mlp, d_model]
        W_up = lw.W_up      # [d_mlp, d_model]
        
        if W_gate is None:
            print(f"  L{li}: W_gate is None, skip")
            continue
        
        # 采样 FFN neurons (前100个 + 随机100个)
        n_neurons = W_gate.shape[0]
        sample_neurons = list(range(min(100, n_neurons)))
        if n_neurons > 100:
            rng = np.random.RandomState(42)
            sample_neurons += rng.choice(n_neurons, size=min(100, n_neurons-100), replace=False).tolist()
        sample_neurons = sorted(set(sample_neurons))
        
        layer_result = {
            "layer": li,
            "n_neurons_sampled": len(sample_neurons),
            "neurons_with_clear_semantic": 0,
            "neurons_target_word_key": {},
            "key_top_words_examples": [],
            "key_cos_distribution": {},
        }
        
        # 对目标词, 记录它们是哪些 key 的 top-1 匹配
        word_to_key_count = {w: 0 for w in target_token_ids}
        
        # 批量计算 key-embedding 余弦
        W_gate_sampled = W_gate[sample_neurons]  # [n_sample, d_model]
        W_gate_norms = np.linalg.norm(W_gate_sampled, axis=1, keepdims=True)
        W_gate_norms = np.maximum(W_gate_norms, 1e-10)
        W_gate_normalized = W_gate_sampled / W_gate_norms
        
        # cos_matrix: [n_sample, n_check]
        cos_matrix = W_gate_normalized @ W_embed_check_normalized.T
        
        # 每行 top-10 的词
        top_k = 10
        for idx, ni in enumerate(sample_neurons):
            top_indices = np.argsort(cos_matrix[idx])[-top_k:][::-1]
            top_cos = cos_matrix[idx][top_indices]
            top_words = [tokenizer.decode([int(j)]) for j in top_indices]
            
            # 检查 top-1 是否是目标词
            for tid in top_indices[:1]:
                for word, wid in target_token_ids.items():
                    if tid == wid:
                        word_to_key_count[word] = word_to_key_count.get(word, 0) + 1
            
            # 只记录前20个neuron的详细结果
            if idx < 20:
                layer_result["key_top_words_examples"].append({
                    "neuron": int(ni),
                    "top_words": top_words,
                    "top_cos": [float(c) for c in top_cos],
                })
        
        # 统计 key 的余弦分布
        all_max_cos = np.max(cos_matrix, axis=1)
        layer_result["key_cos_distribution"] = {
            "mean_max_cos": float(np.mean(all_max_cos)),
            "median_max_cos": float(np.median(all_max_cos)),
            "p90_max_cos": float(np.percentile(all_max_cos, 90)),
            "p10_max_cos": float(np.percentile(all_max_cos, 10)),
        }
        
        # 判断 key 是否有清晰语义: top-1 cos > 0.5 视为有清晰匹配
        n_clear = int(np.sum(all_max_cos > 0.5))
        layer_result["neurons_with_clear_semantic"] = n_clear
        
        layer_result["word_to_key_count"] = {k: int(v) for k, v in word_to_key_count.items()}
        
        elapsed = time.time() - t0
        print(f"  L{li}: sampled {len(sample_neurons)} neurons, "
              f"clear_semantic={n_clear}/{len(sample_neurons)} ({100*n_clear/len(sample_neurons):.1f}%), "
              f"mean_max_cos={np.mean(all_max_cos):.3f}, "
              f"elapsed={elapsed:.1f}s", flush=True)
        
        results[f"L{li}"] = layer_result
    
    # 释放 numpy 内存
    del W_gate_sampled, W_gate_normalized, cos_matrix
    gc.collect()
    
    return results


# ================================================================
# P685: FFN Value 方向追踪
# ================================================================
def p685_ffn_value_direction_tracking(model, tokenizer, model_info, W_embed, W_U):
    """
    P685: FFN Value 方向追踪
    
    核心问题: W_down[:,i] (value) 输出什么方向?
    
    算法原理:
    1. W_down 形状: [d_model, d_mlp] — 每列是一个 value 方向
    2. W_U 形状: [vocab_size, d_model] — 每行是一个词的 logit 方向
    3. value_i 与 W_U 各行的余弦 → value 指向哪些词的 logit
    4. 如果 value_i 主要指向 "apple" 行 → neuron_i 输出 "apple" 方向
    5. 如果 value_i 分散指向多个词 → neuron_i 是分布式输出
    
    这直接揭示: FFN 输出 = Σ gate_i · up_i · (某些词的 logit 方向)
    """
    print("\n" + "="*70)
    print("P685: FFN Value 方向追踪")
    print("="*70)
    
    layers = get_layers(model)
    n_layers = len(layers)
    d_model = model_info.d_model
    d_mlp = model_info.intermediate_size
    mlp_type = model_info.mlp_type
    
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    # 目标词 token ids
    target_token_ids = {}
    for word in list(TARGET_WORDS.keys()) + ATTRIBUTE_WORDS:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids:
            target_token_ids[word] = ids[0]
    
    # W_U 的归一化 — 分块处理避免大矩阵内存
    W_U_norms = np.linalg.norm(W_U, axis=1, keepdims=True)
    W_U_norms = np.maximum(W_U_norms, 1e-10)
    # 不预计算完整 W_U_normalized, 改为按块归一化
    
    # 对 W_U 做 SVD, 取前50个方向
    n_svd = min(50, min(W_U.shape) - 2)
    svd = TruncatedSVD(n_components=n_svd)
    svd.fit(W_U)
    U_svd = svd.components_  # [n_svd, d_model]
    S_svd = svd.singular_values_
    
    results = {}
    CHUNK_SIZE = 5000  # 分块大小, 避免内存溢出
    
    for li in sample_layers:
        t0 = time.time()
        lw = get_layer_weights(layers[li], d_model, mlp_type)
        W_down = lw.W_down  # [d_model, d_mlp]
        
        if W_down is None:
            print(f"  L{li}: W_down is None, skip")
            continue
        
        n_neurons = W_down.shape[1]
        sample_neurons = list(range(min(100, n_neurons)))
        if n_neurons > 100:
            rng = np.random.RandomState(42)
            sample_neurons += rng.choice(n_neurons, size=min(100, n_neurons-100), replace=False).tolist()
        sample_neurons = sorted(set(sample_neurons))
        n_sample = len(sample_neurons)
        
        layer_result = {
            "layer": li,
            "value_to_word_examples": [],
            "value_svd_projection": {},
            "value_concentration": {},
            "apple_value_neurons": [],
        }
        
        # 对每个采样 neuron, 计算 value 方向与 W_U 的匹配
        W_down_sampled = W_down[:, sample_neurons]  # [d_model, n_sample]
        # 归一化 W_down 的列
        W_down_norms = np.linalg.norm(W_down_sampled, axis=0, keepdims=True)
        W_down_norms = np.maximum(W_down_norms, 1e-10)
        W_down_sampled_norm = W_down_sampled / W_down_norms  # [d_model, n_sample]
        
        # 1) value → W_U 行的余弦 — 分块计算
        n_vocab = W_U.shape[0]
        all_top5_indices = np.zeros((n_sample, 5), dtype=int)
        all_top5_cos = np.full((n_sample, 5), -2.0)
        all_max_cos = np.zeros(n_sample)
        apple_cos_per_neuron = np.zeros(n_sample)
        
        for chunk_start in range(0, n_vocab, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, n_vocab)
            W_U_chunk = W_U[chunk_start:chunk_end]
            W_U_chunk_norm = W_U_chunk / W_U_norms[chunk_start:chunk_end]
            cos_chunk = W_U_chunk_norm @ W_down_sampled_norm  # [chunk, n_sample]
            
            for ni in range(n_sample):
                col = cos_chunk[:, ni]
                # 更新 max_cos
                col_max = np.max(col)
                if col_max > all_max_cos[ni]:
                    all_max_cos[ni] = col_max
                
                # 合并 top-5
                merged_cos = np.concatenate([all_top5_cos[ni], col[np.argsort(col)[-5:][::-1]]])
                merged_idx = np.concatenate([all_top5_indices[ni], np.argsort(col)[-5:][::-1] + chunk_start])
                top5_pos = np.argsort(merged_cos)[-5:][::-1]
                all_top5_cos[ni] = merged_cos[top5_pos]
                all_top5_indices[ni] = merged_idx[top5_pos]
                
                # apple 特殊追踪
                if "apple" in target_token_ids:
                    apple_id = target_token_ids["apple"]
                    if chunk_start <= apple_id < chunk_end:
                        apple_cos_per_neuron[ni] = col[apple_id - chunk_start]
            
            del W_U_chunk, W_U_chunk_norm, cos_chunk
        
        # 整理 top-5 匹配词结果
        for idx, ni in enumerate(sample_neurons):
            top_words = [tokenizer.decode([int(j)]) for j in all_top5_indices[idx]]
            if idx < 20:
                layer_result["value_to_word_examples"].append({
                    "neuron": int(ni),
                    "top_words": top_words,
                    "top_cos": [float(c) for c in all_top5_cos[idx]],
                })
        
        # 2) value 在 W_U SVD 方向上的投影
        svd_coeffs = U_svd @ W_down_sampled  # [n_svd, n_sample]
        
        neuron_energy = np.sum(svd_coeffs**2, axis=0)  # [n_sample]
        top1_energy = np.max(svd_coeffs**2, axis=0)
        top5_energy = np.sum(np.sort(svd_coeffs**2, axis=0)[-5:], axis=0)
        
        layer_result["value_svd_projection"] = {
            "mean_top1_ratio": float(np.mean(top1_energy / np.maximum(neuron_energy, 1e-10))),
            "mean_top5_ratio": float(np.mean(top5_energy / np.maximum(neuron_energy, 1e-10))),
            "mean_total_energy": float(np.mean(neuron_energy)),
        }
        
        # 3) Value 集中度: 最大 cos 值的分布
        layer_result["value_concentration"] = {
            "mean_max_cos": float(np.mean(all_max_cos)),
            "median_max_cos": float(np.median(all_max_cos)),
            "p90_max_cos": float(np.percentile(all_max_cos, 90)),
            "n_concentrated_gt05": int(np.sum(all_max_cos > 0.5)),
        }
        
        # 4) 专门找 "apple" 的 value neuron
        if "apple" in target_token_ids:
            top_apple_neurons = np.argsort(apple_cos_per_neuron)[-10:][::-1]
            layer_result["apple_value_neurons"] = [
                {
                    "neuron": int(sample_neurons[idx]),
                    "cos_to_apple": float(apple_cos_per_neuron[idx]),
                }
                for idx in top_apple_neurons
            ]
        
        elapsed = time.time() - t0
        print(f"  L{li}: value_concentration mean_max_cos={np.mean(all_max_cos):.3f}, "
              f"n>0.5={np.sum(all_max_cos>0.5)}/{len(sample_neurons)}, "
              f"SVD top1_ratio={layer_result['value_svd_projection']['mean_top1_ratio']:.3f}, "
              f"elapsed={elapsed:.1f}s", flush=True)
        
        results[f"L{li}"] = layer_result
        
        # 释放内存
        del svd_coeffs, W_down_sampled, W_down_sampled_norm
        gc.collect()
    
    return results


# ================================================================
# P686: Gate 触发条件解析
# ================================================================
def p686_gate_trigger_condition(model, tokenizer, model_info, W_embed, W_U, device):
    """
    P686: Gate 触发条件解析
    
    核心问题: 为什么 gate_i(apple_h) 大?
    
    算法原理:
    1. gate_i = σ(W_gate[i] @ h), h 是当前隐藏状态
    2. W_gate[i] 与 W_embed[j] 的余弦 → gate_i 对哪些输入词敏感
    3. 用实际文本前向传播, 收集 gate 激活值
    4. 对比 apple vs banana vs cat 的 gate 激活模式
    5. 找到 apple 特异激活的 neuron 及其触发条件
    
    关键区别: P684 分析 key 的"静态"语义, P686 分析"动态"激活
    """
    print("\n" + "="*70)
    print("P686: Gate 触发条件解析")
    print("="*70)
    
    layers = get_layers(model)
    n_layers = len(layers)
    d_model = model_info.d_model
    d_mlp = model_info.intermediate_size
    mlp_type = model_info.mlp_type
    
    sample_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    sample_layers = sorted(set([l for l in sample_layers if l < n_layers]))
    
    # 目标词
    test_words = ["apple", "banana", "cat", "car", "river"]
    
    results = {}
    
    for li in sample_layers:
        t0 = time.time()
        lw = get_layer_weights(layers[li], d_model, mlp_type)
        W_gate = lw.W_gate  # [d_mlp, d_model]
        
        if W_gate is None:
            print(f"  L{li}: W_gate is None, skip")
            continue
        
        n_neurons = W_gate.shape[0]
        layer_result = {
            "layer": li,
            "gate_patterns": {},
            "apple_specific_neurons": [],
            "gate_trigger_words": {},
        }
        
        # 用 hook 收集实际 gate 激活
        layer = layers[li]
        gate_activations = {}
        
        def make_gate_hook(word):
            def hook_fn(module, input, output):
                # output 可能是 tuple, 取第一个
                if isinstance(output, tuple):
                    gate_out = output[0]
                else:
                    gate_out = output
                # 取 last token
                gate_np = extract_tensor(gate_out)
                gate_activations[word] = gate_np
            return hook_fn
        
        # 注册 hook 到 mlp.gate_proj
        # 注意: 不同架构 hook 位置不同
        # 对于 HuggingFace 模型, 我们用 forward hook 在 mlp 层
        # 但 gate 值在 mlp 内部, 需要特殊处理
        # 替代方案: 用 W_gate 和隐藏状态直接计算
        
        # 获取每层的隐藏状态
        word_hidden_states = {}
        for word in test_words:
            text = f"The word is {word}."
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            # hidden_states[li+1] 是第 li 层输出 (hidden_states[0] 是 embedding)
            if li + 1 < len(outputs.hidden_states):
                h = extract_tensor(outputs.hidden_states[li + 1])
            else:
                h = extract_tensor(outputs.hidden_states[-1])
            word_hidden_states[word] = h
        
        # 计算 gate 值: gate_i = σ(W_gate[i] @ h)
        word_gates = {}
        for word, h in word_hidden_states.items():
            gate_pre = W_gate @ h  # [d_mlp]
            gate_pre_clipped = np.clip(gate_pre, -500, 500)
            gate_val = 1.0 / (1.0 + np.exp(-gate_pre_clipped))  # sigmoid
            word_gates[word] = gate_val
        
        # 找 apple 特异神经元
        apple_gate = word_gates["apple"]
        banana_gate = word_gates["banana"]
        cat_gate = word_gates["cat"]
        
        # apple_vs_others: apple 的 gate 减去其他词的平均 gate
        other_avg = np.mean([word_gates[w] for w in test_words if w != "apple"], axis=0)
        apple_specificity = apple_gate - other_avg
        
        # Top apple-specific neurons
        top_specific = np.argsort(apple_specificity)[-20:][::-1]
        layer_result["apple_specific_neurons"] = [
            {
                "neuron": int(ni),
                "apple_gate": float(apple_gate[ni]),
                "other_avg_gate": float(other_avg[ni]),
                "specificity": float(apple_specificity[ni]),
            }
            for ni in top_specific
        ]
        
        # Gate 激活分布统计
        for word in test_words:
            g = word_gates[word]
            layer_result["gate_patterns"][word] = {
                "mean": float(np.mean(g)),
                "std": float(np.std(g)),
                "n_active_gt05": int(np.sum(g > 0.5)),
                "n_active_gt09": int(np.sum(g > 0.9)),
                "max": float(np.max(g)),
                "sparsity": float(np.mean(g < 0.01)),  # 近似稀疏度
            }
        
        # Apple 特异 neuron 的 gate 触发词分析
        for ni in top_specific[:5]:
            # 计算 W_gate[ni] 与目标词 embedding 的余弦
            key_vec = W_gate[ni]
            key_norm = np.linalg.norm(key_vec)
            trigger_words = {}
            if key_norm > 1e-10:
                for word in test_words:
                    h = word_hidden_states[word]
                    cos = float(np.dot(key_vec, h) / (key_norm * np.linalg.norm(h)))
                    trigger_words[word] = cos
            layer_result["gate_trigger_words"][f"neuron_{ni}"] = trigger_words
        
        elapsed = time.time() - t0
        apple_sparsity = layer_result["gate_patterns"]["apple"]["sparsity"]
        n_specific = len([n for n in top_specific if apple_specificity[n] > 0.1])
        print(f"  L{li}: apple sparsity={apple_sparsity:.3f}, "
              f"n_active(>0.5)={layer_result['gate_patterns']['apple']['n_active_gt05']}, "
              f"apple-specific(>0.1)={n_specific}, "
              f"elapsed={elapsed:.1f}s")
        
        results[f"L{li}"] = layer_result
        
        del word_hidden_states, word_gates
        gc.collect()
    
    return results


# ================================================================
# P687: 苹果的精确编码方程
# ================================================================
def p687_apple_encoding_equation(model, tokenizer, model_info, W_embed, W_U, device):
    """
    P687: 苹果的精确编码方程
    
    核心目标: 写出 h(apple) 的精确分解
    
    h(apple) = h_base + Σ_{i∈active(apple)} gate_i(h) · up_i(h) · W_down[:,i]
    
    其中:
    - h_base = 该层 apple 输入的隐藏状态 (FFN 输入前)
    - active(apple) = gate_i > threshold 的 neuron 集合
    - gate_i, up_i, W_down[:,i] 各项的精确贡献
    
    验证: 重建 h(apple) 的 R²
    
    算法原理:
    1. 用实际前向传播获取 h_pre (FFN输入) 和 h_post (FFN输出)
    2. 计算 FFN_output = h_post - h_pre
    3. 展开为 neuron 贡献: contrib_i = gate_i · up_i · W_down[:,i]
    4. 验证: Σ contrib_i ≈ FFN_output (R² 接近 1)
    5. 对 apple vs banana 对比 neuron 贡献差异
    """
    print("\n" + "="*70)
    print("P687: 苹果的精确编码方程")
    print("="*70)
    
    layers = get_layers(model)
    n_layers = len(layers)
    d_model = model_info.d_model
    d_mlp = model_info.intermediate_size
    mlp_type = model_info.mlp_type
    
    # 关键层: 前3层 (L0-L2 黄金层) + 中间层 + 最后层
    key_layers = list(range(min(4, n_layers))) + [n_layers // 2, n_layers - 1]
    key_layers = sorted(set([l for l in key_layers if l < n_layers]))
    
    test_words = ["apple", "banana", "cat"]
    threshold = 0.01  # gate 阈值
    
    results = {}
    
    for li in key_layers:
        t0 = time.time()
        lw = get_layer_weights(layers[li], d_model, mlp_type)
        W_gate = lw.W_gate  # [d_mlp, d_model]
        W_up = lw.W_up      # [d_mlp, d_model]
        W_down = lw.W_down  # [d_model, d_mlp]
        
        if W_gate is None or W_up is None or W_down is None:
            print(f"  L{li}: Missing weight matrices, skip")
            continue
        
        n_neurons = W_gate.shape[0]
        layer_result = {
            "layer": li,
            "words": {},
            "cross_word_comparison": {},
        }
        
        for word in test_words:
            text = f"The word is {word}."
            inputs = tokenizer(text, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            # h_pre = FFN 输入 = layer output (注意: hidden_states[li+1] 是第 li 层输出)
            # 但我们想要的是 FFN 输入 = Attention 输出 = resid_mid
            # 由于 HuggingFace 模型不直接输出 resid_mid, 我们用近似:
            # FFN_input ≈ hidden_states[li+1] (第 li 层输出) 中 last token
            # 更精确: 用 hook 获取
            
            h_full = extract_tensor(outputs.hidden_states[li + 1])  # 第 li 层输出, last token
            h_next = extract_tensor(outputs.hidden_states[li + 2]) if li + 2 < len(outputs.hidden_states) else h_full
            
            # 近似 FFN output = h_next - h_after_attn (但需要更精确)
            # 这里用 W_gate 直接计算 gate 和 up 值
            
            gate_pre = W_gate @ h_full  # [d_mlp]
            # 安全 sigmoid, 避免溢出
            gate_pre_clipped = np.clip(gate_pre, -500, 500)
            gate_val = 1.0 / (1.0 + np.exp(-gate_pre_clipped))
            up_val = W_up @ h_full  # [d_mlp]
            post_val = gate_val * up_val  # [d_mlp]
            ffn_output = W_down @ post_val  # [d_model]
            
            # 逐 neuron 贡献
            active_mask = gate_val > threshold
            n_active = int(np.sum(active_mask))
            
            # Top contributing neurons (对 h 方向的贡献大小)
            contrib_norms = np.abs(gate_val * up_val) * np.linalg.norm(W_down, axis=0)
            top_contrib_neurons = np.argsort(contrib_norms)[-20:][::-1]
            
            word_result = {
                "n_active_neurons": n_active,
                "active_ratio": float(n_active / n_neurons),
                "ffn_output_norm": float(np.linalg.norm(ffn_output)),
                "top_contributing_neurons": [
                    {
                        "neuron": int(ni),
                        "gate": float(gate_val[ni]),
                        "up": float(up_val[ni]),
                        "post": float(post_val[ni]),
                        "contrib_norm": float(contrib_norms[ni]),
                    }
                    for ni in top_contrib_neurons[:10]
                ],
            }
            
            layer_result["words"][word] = word_result
        
        # 跨词对比: apple vs banana 的 neuron 贡献差异
        if "apple" in layer_result["words"] and "banana" in layer_result["words"]:
            # 重新计算以获取完整对比
            _ag = W_gate @ extract_tensor(
                model(**tokenizer(f"The word is apple.", return_tensors="pt").to(device),
                       output_hidden_states=True).hidden_states[li + 1]
            )
            apple_gate = 1.0 / (1.0 + np.exp(-np.clip(_ag, -500, 500)))
            _bg = W_gate @ extract_tensor(
                model(**tokenizer(f"The word is banana.", return_tensors="pt").to(device),
                       output_hidden_states=True).hidden_states[li + 1]
            )
            banana_gate = 1.0 / (1.0 + np.exp(-np.clip(_bg, -500, 500)))
            
            # Apple 特有激活 (apple 有但 banana 没有)
            apple_only = (apple_gate > threshold) & (banana_gate <= threshold)
            # 共享激活
            shared = (apple_gate > threshold) & (banana_gate > threshold)
            
            layer_result["cross_word_comparison"] = {
                "apple_only_neurons": int(np.sum(apple_only)),
                "shared_neurons": int(np.sum(shared)),
                "banana_only_neurons": int(np.sum((banana_gate > threshold) & (apple_gate <= threshold))),
            }
            
            # Apple 特有 neuron 的 value 方向
            apple_only_indices = np.where(apple_only)[0]
            if len(apple_only_indices) > 0:
                # 最多取10个
                sample_ao = apple_only_indices[:min(10, len(apple_only_indices))]
                apple_only_values = W_down[:, sample_ao]  # [d_model, n_ao]
                
                # 这些 value 方向在 W_U 上的投影
                proj_to_WU = W_U @ apple_only_values  # [vocab_size, n_ao]
                top_words_per_ao = []
                for idx in range(proj_to_WU.shape[1]):
                    top5 = np.argsort(proj_to_WU[:, idx])[-5:][::-1]
                    top5_words = [tokenizer.decode([int(j)]) for j in top5]
                    top_words_per_ao.append({
                        "neuron": int(sample_ao[idx]),
                        "top_logit_words": top5_words,
                    })
                layer_result["cross_word_comparison"]["apple_only_value_words"] = top_words_per_ao
        
        elapsed = time.time() - t0
        apple_info = layer_result["words"].get("apple", {})
        print(f"  L{li}: apple active={apple_info.get('n_active_neurons', '?')}, "
              f"ffn_norm={apple_info.get('ffn_output_norm', '?'):.2f}, "
              f"elapsed={elapsed:.1f}s")
        
        results[f"L{li}"] = layer_result
        gc.collect()
    
    return results


# ================================================================
# 主函数
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="Phase CLVI: FFN Key-Value Semantic Map")
    parser.add_argument("--model", choices=["qwen3", "deepseek7b", "glm4"], default="qwen3")
    parser.add_argument("--skip-p684", action="store_true", help="Skip P684")
    parser.add_argument("--skip-p685", action="store_true", help="Skip P685")
    parser.add_argument("--skip-p686", action="store_true", help="Skip P686")
    parser.add_argument("--skip-p687", action="store_true", help="Skip P687")
    args = parser.parse_args()
    
    model_name = args.model
    print(f"\n{'='*70}")
    print(f"Phase CLVI: FFN Key-Value 语义图谱 — {model_name}")
    print(f"{'='*70}")
    
    # 加载模型
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    
    print(f"Model: {model_info.model_class}")
    print(f"Layers: {model_info.n_layers}, d_model: {model_info.d_model}, "
          f"d_mlp: {model_info.intermediate_size}, vocab: {model_info.vocab_size}")
    
    # 获取全局矩阵
    W_embed = get_W_embed(model)  # [vocab_size, d_model]
    W_U = get_W_U(model)          # [vocab_size, d_model]
    
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
    
    # P684: Key 语义解码
    if not args.skip_p684:
        p684_result = p684_ffn_key_semantic_decoding(model, tokenizer, model_info, W_embed, W_U)
        all_results["P684"] = p684_result
    
    # P685: Value 方向追踪
    if not args.skip_p685:
        p685_result = p685_ffn_value_direction_tracking(model, tokenizer, model_info, W_embed, W_U)
        all_results["P685"] = p685_result
    
    # P686: Gate 触发条件
    if not args.skip_p686:
        p686_result = p686_gate_trigger_condition(model, tokenizer, model_info, W_embed, W_U, device)
        all_results["P686"] = p686_result
    
    # P687: 精确编码方程
    if not args.skip_p687:
        p687_result = p687_apple_encoding_equation(model, tokenizer, model_info, W_embed, W_U, device)
        all_results["P687"] = p687_result
    
    # 保存结果
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"phase_clvi_{model_name}_{ts}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")
    
    # 释放模型
    release_model(model)
    
    # 打印关键发现摘要
    print_summary(all_results)
    
    return all_results


def print_summary(results):
    """打印关键发现摘要"""
    print("\n" + "="*70)
    print("关键发现摘要")
    print("="*70)
    
    model_name = results.get("model", "?")
    print(f"\n模型: {model_name}")
    
    # P684 摘要
    if "P684" in results:
        print("\n--- P684: Key 语义解码 ---")
        for layer_key, lr in results["P684"].items():
            if isinstance(lr, dict) and "key_cos_distribution" in lr:
                cd = lr["key_cos_distribution"]
                n_clear = lr.get("neurons_with_clear_semantic", 0)
                n_sampled = lr.get("n_neurons_sampled", 1)
                print(f"  {layer_key}: mean_max_cos={cd.get('mean_max_cos', '?'):.3f}, "
                      f"clear_semantic={n_clear}/{n_sampled} ({100*n_clear/max(n_sampled,1):.1f}%)")
                # 显示一些 key 的 top 词
                if lr.get("key_top_words_examples"):
                    ex = lr["key_top_words_examples"][0]
                    print(f"    示例: neuron {ex['neuron']} top词={ex['top_words'][:5]}")
    
    # P685 摘要
    if "P685" in results:
        print("\n--- P685: Value 方向追踪 ---")
        for layer_key, lr in results["P685"].items():
            if isinstance(lr, dict) and "value_concentration" in lr:
                vc = lr["value_concentration"]
                svd = lr.get("value_svd_projection", {})
                print(f"  {layer_key}: mean_max_cos={vc.get('mean_max_cos', '?'):.3f}, "
                      f"n>0.5={vc.get('n_concentrated_gt05', '?')}, "
                      f"SVD top1_ratio={svd.get('mean_top1_ratio', '?'):.3f}")
                if lr.get("apple_value_neurons"):
                    top_avn = lr["apple_value_neurons"][0]
                    print(f"    Apple top value neuron: {top_avn['neuron']}, cos={top_avn['cos_to_apple']:.3f}")
    
    # P686 摘要
    if "P686" in results:
        print("\n--- P686: Gate 触发条件 ---")
        for layer_key, lr in results["P686"].items():
            if isinstance(lr, dict) and "gate_patterns" in lr:
                gp = lr["gate_patterns"]
                apple_gp = gp.get("apple", {})
                n_specific = len([n for n in lr.get("apple_specific_neurons", []) if n.get("specificity", 0) > 0.1])
                print(f"  {layer_key}: apple sparsity={apple_gp.get('sparsity', '?'):.3f}, "
                      f"active(>0.5)={apple_gp.get('n_active_gt05', '?')}, "
                      f"apple-specific(>0.1)={n_specific}")
    
    # P687 摘要
    if "P687" in results:
        print("\n--- P687: 苹果精确编码方程 ---")
        for layer_key, lr in results["P687"].items():
            if isinstance(lr, dict) and "words" in lr:
                apple = lr["words"].get("apple", {})
                cw = lr.get("cross_word_comparison", {})
                print(f"  {layer_key}: apple active={apple.get('n_active_neurons', '?')}, "
                      f"ffn_norm={apple.get('ffn_output_norm', '?'):.2f}, "
                      f"apple_only={cw.get('apple_only_neurons', '?')}, "
                      f"shared={cw.get('shared_neurons', '?')}")


if __name__ == "__main__":
    main()
