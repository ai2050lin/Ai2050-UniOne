#!/usr/bin/env python3
"""
Phase CLX: 注意力头的语义角色 — "A项到底做了什么?"
====================================================

核心目标: 精确分解A项为单个注意力头的贡献,分析哪些头负责语义

CLIX关键发现:
  Qwen3/DS7B: 89-93%层G和A异号, sign_corr=-0.80到-0.93
  A项携带词特异信息, 是语义区分的主要来源
  但A = Σ_head(softmax(QK^T)V)W_O^h, 无法区分哪些头贡献了语义

实验设计:
  P698: 注意力头的logit贡献分解
    - 用hook捕获每层attention的per-head输出
    - 每个头的logit贡献: logit_h = W_U[word] · head_output_h
    - 稀疏性: 少数头贡献了大部分logit?

  P699: 语义头vs语法头vs抑制头的分类
    - 语义头: 对内容词贡献正logit
    - 语法头: 对功能词(the)贡献正logit
    - 抑制头: 对大部分词贡献负logit

  P700: 跨词注意力头共享分析
    - 水果词是否共享同一批语义头?
    - 头的"语义特异性指数"

重要注意: GQA(Grouped Query Attention)
  Qwen3: 32 Q heads, 8 KV heads → 每个KV head服务4个Q head
  DeepSeek7B: 28 Q heads, 4 KV heads → 每个KV head服务7个Q head
  GLM4: 32 Q heads, 32 KV heads → 标准MHA

  在GQA中, per-head分解基于Q头: 每个Q头有独立的注意力模式
  但V是共享的, 所以我们用W_o的列来区分头的输出

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
from model_utils import load_model, get_model_info, get_layer_weights, release_model, MODEL_CONFIGS

import functools
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
print = functools.partial(print, flush=True)

OUT_DIR = Path("d:/develop/TransformerLens-main/results/phase_clx")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path("d:/develop/TransformerLens-main/tests/glm5_temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ============ 测试词汇 ============
TEST_WORDS = {
    "apple":    {"cat": "fruit", "pos": "noun",      "freq": "high"},
    "banana":   {"cat": "fruit", "pos": "noun",      "freq": "high"},
    "orange":   {"cat": "fruit", "pos": "noun",      "freq": "high"},
    "cat":      {"cat": "animal", "pos": "noun",     "freq": "high"},
    "dog":      {"cat": "animal", "pos": "noun",     "freq": "high"},
    "car":      {"cat": "vehicle", "pos": "noun",    "freq": "high"},
    "run":      {"cat": "action", "pos": "verb",     "freq": "high"},
    "red":      {"cat": "color", "pos": "adjective", "freq": "high"},
    "the":      {"cat": "function", "pos": "determiner", "freq": "very_high"},
}


def get_layers(model):
    """获取模型的 transformer 层"""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    return []


def get_n_heads(sa, model, d_model):
    """获取注意力头数(兼容不同模型架构)"""
    if hasattr(sa, 'num_heads'):
        return sa.num_heads
    elif hasattr(sa, 'config') and hasattr(sa.config, 'num_attention_heads'):
        return sa.config.num_attention_heads
    elif hasattr(model, 'config') and hasattr(model.config, 'num_attention_heads'):
        return model.config.num_attention_heads
    else:
        head_dim = sa.head_dim if hasattr(sa, 'head_dim') else 128
        return d_model // head_dim


def get_n_kv_heads(sa, model):
    """获取KV头数(兼容GQA)"""
    if hasattr(sa, 'num_key_value_heads'):
        return sa.num_key_value_heads
    elif hasattr(model, 'config') and hasattr(model.config, 'num_key_value_heads'):
        return model.config.num_key_value_heads
    else:
        return get_n_heads(sa, model, model.config.hidden_size if hasattr(model, 'config') else 2560)


def compute_per_head_logits_via_attn_output(model, tokenizer, model_info, W_U, device, word):
    """
    用hook捕获每层attention输出,然后通过W_o列分解为per-head贡献
    
    核心思路:
    - Attention输出: attn_output = concat(head_1, ..., head_n) @ W_o
    - 我们可以获取 attn_output (通过hook)
    - 但 attn_output 是拼接后的, 需要反推per-head贡献
    
    方法: 利用attention权重矩阵
    - attn_output的形状: [seq_len, n_heads * head_dim] (拼接后)
    - attn_output @ W_o → [seq_len, d_model] (输出)
    - 我们用attn_weights来获取每个头的注意力模式
    - 然后计算: per_head_logit[h] = W_U[word] @ (attn_output_h @ W_o[:, h*hd:(h+1)*hd])
    
    注意: 在GQA中, W_o仍然有n_heads*head_dim列, 所以按列切片即可
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)

    word_ids = tokenizer.encode(word, add_special_tokens=False)
    if not word_ids:
        return None
    word_id = word_ids[0]
    W_U_word = W_U[word_id]

    text = f"The word is {word}."
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # 前向传播,获取hidden_states和attentions
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # 采样层(避免OOM)
    sample_layers = sorted(set(
        list(range(0, n_layers, max(n_layers // 12, 1))) +
        list(range(max(0, n_layers - 3), n_layers))
    ))

    head_logits = {}  # {layer_idx: array[n_heads]}
    head_norms = {}   # {layer_idx: array[n_heads]}

    for li in sample_layers:
        layer = layers[li]
        sa = layer.self_attn
        n_heads = get_n_heads(sa, model, d_model)
        head_dim = d_model // n_heads

        # 获取W_o
        W_o = sa.o_proj.weight.detach().float().cpu().numpy()  # [d_model, n_heads*head_dim]
        
        # W_v在GQA中: [n_kv_heads * head_dim, d_model]
        W_v = sa.v_proj.weight.detach().float().cpu().numpy()
        
        n_kv_heads = get_n_kv_heads(sa, model)
        n_q_heads = n_heads  # Q头数
        
        # 检查维度
        kv_dim = n_kv_heads * head_dim  # V投影的输出维度
        q_dim = n_q_heads * head_dim    # attention输出的维度(W_o的输入维度)
        
        actual_v_dim = W_v.shape[0]
        actual_o_dim = W_o.shape[1]
        
        h_before = outputs.hidden_states[li][0, -1, :].float().detach().cpu().numpy()
        
        # V投影: [n_kv_heads * head_dim]
        V_all = W_v @ h_before

        # per-head logit贡献
        # 在GQA中: n_q_heads > n_kv_heads
        # 每个KV头服务 n_q_heads/n_kv_heads 个Q头
        # 但W_o是 [d_model, n_q_heads*head_dim], 所以可以按Q头切片
        
        # 方法: 基于KV头分解(更自然,因为V是按KV头组织的)
        # 每个KV头的贡献 = 该KV头的V × W_o中对应列的加权和
        
        n_groups = n_q_heads // n_kv_heads  # 每个KV头服务的Q头数
        
        if n_groups > 1:
            # GQA: 每个KV头的V被复制n_groups次, 然后分别乘W_o的不同部分
            # KV头h对应Q头: h*n_groups, h*n_groups+1, ..., (h+1)*n_groups-1
            # 该KV头对logit的总贡献 = Σ_{q=h*ng}^{(h+1)*ng-1} (W_U @ W_o[:, q*hd:(q+1)*hd]) @ V_h
            
            effective_n_heads = n_kv_heads
            per_head_logit = np.zeros(effective_n_heads)
            per_head_norm = np.zeros(effective_n_heads)
            
            for h in range(effective_n_heads):
                V_h = V_all[h * head_dim: (h + 1) * head_dim]  # [head_dim]
                
                # 该KV头对应的所有Q头的W_o切片之和
                total_logit = 0.0
                for q in range(h * n_groups, (h + 1) * n_groups):
                    W_o_q = W_o[:, q * head_dim: (q + 1) * head_dim]  # [d_model, head_dim]
                    direction_q = W_U_word @ W_o_q  # [head_dim]
                    total_logit += float(direction_q @ V_h)
                
                per_head_logit[h] = total_logit
                per_head_norm[h] = float(np.linalg.norm(V_h))
        else:
            # 标准MHA: 每个头独立
            effective_n_heads = n_q_heads
            per_head_logit = np.zeros(effective_n_heads)
            per_head_norm = np.zeros(effective_n_heads)
            
            for h in range(effective_n_heads):
                V_h = V_all[h * head_dim: (h + 1) * head_dim]
                W_o_h = W_o[:, h * head_dim: (h + 1) * head_dim]
                direction_h = W_U_word @ W_o_h
                per_head_logit[h] = float(direction_h @ V_h)
                per_head_norm[h] = float(np.linalg.norm(V_h))

        head_logits[li] = per_head_logit
        head_norms[li] = per_head_norm

    del outputs
    gc.collect()

    return head_logits, head_norms, sample_layers


# ================================================================
# P698: 注意力头的logit贡献分解
# ================================================================
def p698_head_logit_decomposition(model, tokenizer, model_info, W_embed, W_U, device):
    """
    P698: 注意力头的logit贡献分解

    核心问题: 哪些注意力头贡献了word的logit?

    分析:
    1. 每个头的logit贡献: logit_h = W_U[word] · (V_h · W_o_h.T)
    2. 正向头 vs 负向头的比例
    3. logit贡献的稀疏性: top-k头贡献了多少比例?
    4. 头的"角色": 哪些头一致地贡献正向/负向logit?
    """
    print("\n" + "="*70)
    print("P698: 注意力头的logit贡献分解")
    print("="*70)

    n_layers = model_info.n_layers
    results = {}

    for word in TEST_WORDS:
        t0 = time.time()
        ret = compute_per_head_logits_via_attn_output(model, tokenizer, model_info, W_U, device, word)
        if ret is None:
            continue
        head_logits, head_norms, sample_layers = ret

        # 聚合所有采样层的头信息
        all_head_logits = []
        all_head_ids = []
        positive_count = 0
        negative_count = 0

        for li in sorted(head_logits.keys()):
            hl = head_logits[li]
            for h in range(len(hl)):
                all_head_logits.append(hl[h])
                all_head_ids.append((li, h))
                if hl[h] > 0:
                    positive_count += 1
                else:
                    negative_count += 1

        all_head_logits = np.array(all_head_logits)
        total_heads = len(all_head_logits)

        # 稀疏性
        sorted_logits = np.sort(np.abs(all_head_logits))[::-1]
        total_abs = np.sum(sorted_logits)
        if total_abs > 0:
            top1_ratio = float(sorted_logits[0] / total_abs)
            top5_ratio = float(np.sum(sorted_logits[:5]) / total_abs)
            top10_ratio = float(np.sum(sorted_logits[:min(10, total_heads)]) / total_abs)
            top10pct_ratio = float(np.sum(sorted_logits[:max(1, total_heads // 10)]) / total_abs)
        else:
            top1_ratio = top5_ratio = top10_ratio = top10pct_ratio = 0.0

        # Top-5 正向和负向头
        top_positive_idx = np.argsort(all_head_logits)[::-1][:5]
        top_negative_idx = np.argsort(all_head_logits)[:5]

        top_positive_heads = [(int(all_head_ids[i][0]), int(all_head_ids[i][1]),
                               float(all_head_logits[i])) for i in top_positive_idx]
        top_negative_heads = [(int(all_head_ids[i][0]), int(all_head_ids[i][1]),
                               float(all_head_logits[i])) for i in top_negative_idx]

        # 每层摘要
        layer_summary = {}
        for li in sorted(head_logits.keys()):
            hl = head_logits[li]
            layer_summary[f"L{li}"] = {
                "n_positive": int(np.sum(hl > 0)),
                "n_negative": int(np.sum(hl <= 0)),
                "sum_logit": float(np.sum(hl)),
                "max_logit": float(np.max(hl)) if len(hl) > 0 else 0,
                "min_logit": float(np.min(hl)) if len(hl) > 0 else 0,
                "std_logit": float(np.std(hl)) if len(hl) > 0 else 0,
            }

        results[word] = {
            "word_info": TEST_WORDS[word],
            "total_heads": total_heads,
            "positive_heads": positive_count,
            "negative_heads": negative_count,
            "pos_neg_ratio": float(positive_count / max(negative_count, 1)),
            "sparsity": {
                "top1_ratio": top1_ratio,
                "top5_ratio": top5_ratio,
                "top10_ratio": top10_ratio,
                "top10pct_ratio": top10pct_ratio,
            },
            "top5_positive_heads": top_positive_heads,
            "top5_negative_heads": top_negative_heads,
            "layer_summary": layer_summary,
        }

        elapsed = time.time() - t0
        print(f"  {word}: total_heads={total_heads}, pos/neg={positive_count}/{negative_count}, "
              f"top10%_ratio={top10pct_ratio:.3f}, "
              f"elapsed={elapsed:.1f}s")

        gc.collect()

    return results


# ================================================================
# P699: 语义头vs语法头vs抑制头的分类
# ================================================================
def p699_head_classification(model, tokenizer, model_info, W_embed, W_U, device):
    """
    P699: 语义头vs语法头vs抑制头的分类

    核心问题: 注意力头可以分成哪些类型?

    分类标准:
    - 语义头(Semantic): 对内容词贡献正logit
    - 语法头(Syntactic): 对功能词(the)贡献正logit,对内容词贡献小
    - 抑制头(Suppressor): 对大部分词贡献负logit
    - 混合头(Mixed): 没有明确模式
    """
    print("\n" + "="*70)
    print("P699: 语义头vs语法头vs抑制头的分类")
    print("="*70)

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)

    # 采样层
    sample_layers = sorted(set(
        list(range(0, n_layers, max(n_layers // 10, 1))) +
        list(range(max(0, n_layers - 3), n_layers))
    ))

    words_list = list(TEST_WORDS.keys())
    fruit_words = [w for w in words_list if TEST_WORDS[w]["cat"] == "fruit"]
    animal_words = [w for w in words_list if TEST_WORDS[w]["cat"] == "animal"]
    function_words = [w for w in words_list if TEST_WORDS[w]["cat"] == "function"]
    content_words = [w for w in words_list if TEST_WORDS[w]["cat"] not in ["function"]]

    # 收集每个头在所有词上的logit贡献
    head_profiles = {}  # (li, h) -> {word: logit}

    for word in words_list:
        t0 = time.time()
        ret = compute_per_head_logits_via_attn_output(model, tokenizer, model_info, W_U, device, word)
        if ret is None:
            continue
        head_logits, _, _ = ret

        for li in head_logits:
            for h in range(len(head_logits[li])):
                key = (li, h)
                if key not in head_profiles:
                    head_profiles[key] = {}
                head_profiles[key][word] = float(head_logits[li][h])

        elapsed = time.time() - t0
        print(f"  收集 {word} 的头贡献... elapsed={elapsed:.1f}s")

    # 分类每个头
    results = {}
    head_types = {"semantic": [], "syntactic": [], "suppressor": [], "mixed": []}

    for key, profile in head_profiles.items():
        li, h = key

        content_logits = [profile.get(w, 0) for w in content_words]
        fruit_logits = [profile.get(w, 0) for w in fruit_words]
        function_logits = [profile.get(w, 0) for w in function_words]
        all_logits = [profile.get(w, 0) for w in words_list]

        mean_content = float(np.mean(content_logits)) if content_logits else 0
        mean_fruit = float(np.mean(fruit_logits)) if fruit_logits else 0
        mean_function = float(np.mean(function_logits)) if function_logits else 0
        mean_all = float(np.mean(all_logits))
        std_all = float(np.std(all_logits))

        # 分类
        frac_negative = np.mean([l < 0 for l in all_logits])
        if mean_all < -0.5 and frac_negative > 0.6:
            head_type = "suppressor"
        elif abs(mean_fruit - mean_function) < 0.3 and std_all < 0.5:
            head_type = "mixed"
        elif mean_fruit > mean_function + 0.3:
            head_type = "semantic"
        elif mean_function > mean_fruit + 0.3:
            head_type = "syntactic"
        else:
            head_type = "mixed"

        head_types[head_type].append(key)

        results[f"L{li}_H{h}"] = {
            "layer": li,
            "head": h,
            "type": head_type,
            "semantic_score": float(mean_fruit),
            "syntactic_score": float(mean_function),
            "suppressor_score": float(-mean_all),
            "mean_all": mean_all,
            "std_all": std_all,
            "profile": {w: float(profile.get(w, 0)) for w in words_list},
        }

    # 统计
    type_counts = {t: len(v) for t, v in head_types.items()}
    type_ratios = {t: float(len(v) / max(len(head_profiles), 1)) for t, v in head_types.items()}

    semantic_layers = [li for li, h in head_types["semantic"]]
    syntactic_layers = [li for li, h in head_types["syntactic"]]
    suppressor_layers = [li for li, h in head_types["suppressor"]]

    results["_summary"] = {
        "total_heads_analyzed": len(head_profiles),
        "type_counts": type_counts,
        "type_ratios": type_ratios,
        "semantic_layer_distribution": {
            "mean": float(np.mean(semantic_layers)) if semantic_layers else 0,
            "std": float(np.std(semantic_layers)) if semantic_layers else 0,
        },
        "syntactic_layer_distribution": {
            "mean": float(np.mean(syntactic_layers)) if syntactic_layers else 0,
            "std": float(np.std(syntactic_layers)) if syntactic_layers else 0,
        },
        "suppressor_layer_distribution": {
            "mean": float(np.mean(suppressor_layers)) if suppressor_layers else 0,
            "std": float(np.std(suppressor_layers)) if suppressor_layers else 0,
        },
    }

    print(f"\n  头分类统计:")
    print(f"    总头数: {len(head_profiles)}")
    for t, count in type_counts.items():
        print(f"    {t}: {count} ({type_ratios[t]*100:.1f}%)")
    if semantic_layers:
        print(f"    语义头层分布: mean={np.mean(semantic_layers):.1f}, std={np.std(semantic_layers):.1f}")
    if suppressor_layers:
        print(f"    抑制头层分布: mean={np.mean(suppressor_layers):.1f}, std={np.std(suppressor_layers):.1f}")

    return results


# ================================================================
# P700: 跨词注意力头共享分析
# ================================================================
def p700_cross_word_head_sharing(model, tokenizer, model_info, W_embed, W_U, device):
    """
    P700: 跨词注意力头共享分析

    核心问题:
    1. 同一头在不同词上的贡献是否一致?
    2. 水果词是否共享同一批语义头?
    3. 头的"语义特异性指数"
    """
    print("\n" + "="*70)
    print("P700: 跨词注意力头共享分析")
    print("="*70)

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)

    sample_layers = sorted(set(
        list(range(0, n_layers, max(n_layers // 10, 1))) +
        list(range(max(0, n_layers - 3), n_layers))
    ))

    words_list = list(TEST_WORDS.keys())
    fruit_words = [w for w in words_list if TEST_WORDS[w]["cat"] == "fruit"]
    animal_words = [w for w in words_list if TEST_WORDS[w]["cat"] == "animal"]

    # 收集每个头在所有词上的logit贡献
    head_profiles = {}

    for word in words_list:
        t0 = time.time()
        ret = compute_per_head_logits_via_attn_output(model, tokenizer, model_info, W_U, device, word)
        if ret is None:
            continue
        head_logits, _, _ = ret

        for li in head_logits:
            for h in range(len(head_logits[li])):
                key = (li, h)
                if key not in head_profiles:
                    head_profiles[key] = {}
                head_profiles[key][word] = float(head_logits[li][h])

        elapsed = time.time() - t0
        print(f"  收集 {word} 的头贡献... elapsed={elapsed:.1f}s")

    if len(head_profiles) < 10:
        return {"warning": "insufficient heads"}

    # 构建头矩阵
    head_ids = sorted(head_profiles.keys())
    head_matrix = np.array([[head_profiles[hid].get(w, 0) for w in words_list]
                            for hid in head_ids])  # [n_heads, n_words]

    # 1) 词间相关性矩阵
    word_corr = np.corrcoef(head_matrix.T) if head_matrix.shape[1] > 1 else np.eye(len(words_list))

    # 2) 水果共享头
    fruit_idx = [words_list.index(w) for w in fruit_words]
    animal_idx = [words_list.index(w) for w in animal_words]

    fruit_shared_heads = []
    for i, hid in enumerate(head_ids):
        fruit_logits = head_matrix[i, fruit_idx]
        if np.all(fruit_logits > 0):
            fruit_shared_heads.append({
                "head": f"L{hid[0]}_H{hid[1]}",
                "layer": hid[0],
                "head_idx": hid[1],
                "mean_fruit_logit": float(np.mean(fruit_logits)),
            })

    # 3) 语义特异性指数
    categories = {}
    for cat in set(TEST_WORDS[w]["cat"] for w in words_list):
        cat_words_idx = [words_list.index(w) for w in words_list if TEST_WORDS[w]["cat"] == cat]
        non_cat_words_idx = [words_list.index(w) for w in words_list if TEST_WORDS[w]["cat"] != cat]

        cat_specificity = []
        for i in range(len(head_ids)):
            if cat_words_idx and non_cat_words_idx:
                spec = float(np.mean(head_matrix[i, cat_words_idx]) -
                             np.mean(head_matrix[i, non_cat_words_idx]))
            else:
                spec = 0.0
            cat_specificity.append(spec)

        categories[cat] = {
            "mean_specificity": float(np.mean(cat_specificity)),
            "max_specificity": float(np.max(cat_specificity)),
            "std_specificity": float(np.std(cat_specificity)),
            "n_high_specificity_heads": int(np.sum(np.array(cat_specificity) > 0.5)),
        }

    # 4) 头的聚类
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    if head_matrix.shape[0] > 4:
        scaler = StandardScaler()
        head_matrix_scaled = scaler.fit_transform(head_matrix)

        n_clusters = min(4, head_matrix.shape[0] - 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(head_matrix_scaled)

        cluster_info = {}
        for c in range(n_clusters):
            cluster_heads = [head_ids[i] for i in range(len(labels)) if labels[i] == c]
            cluster_profiles = head_matrix[labels == c]
            cluster_info[f"cluster_{c}"] = {
                "n_heads": len(cluster_heads),
                "mean_profile": {w: float(np.mean(cluster_profiles[:, wi]))
                                 for wi, w in enumerate(words_list)},
                "layers": sorted(set([hid[0] for hid in cluster_heads])),
            }
    else:
        cluster_info = {"note": "too few heads for clustering"}

    # 5) 头间相似度
    n_heads_total = head_matrix.shape[0]
    max_sample = min(200, n_heads_total)
    sample_idx = np.random.choice(n_heads_total, max_sample, replace=False) if n_heads_total > max_sample else np.arange(n_heads_total)

    sample_profiles = head_matrix[sample_idx]
    norms = np.linalg.norm(sample_profiles, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = sample_profiles / norms
    cos_matrix = normalized @ normalized.T

    same_layer_sims = []
    cross_layer_sims = []
    for i in range(len(sample_idx)):
        for j in range(i + 1, len(sample_idx)):
            hi = head_ids[sample_idx[i]]
            hj = head_ids[sample_idx[j]]
            sim = cos_matrix[i, j]
            if hi[0] == hj[0]:
                same_layer_sims.append(sim)
            else:
                cross_layer_sims.append(sim)

    head_similarity = {
        "same_layer_mean_cos": float(np.mean(same_layer_sims)) if same_layer_sims else 0,
        "cross_layer_mean_cos": float(np.mean(cross_layer_sims)) if cross_layer_sims else 0,
        "same_layer_std_cos": float(np.std(same_layer_sims)) if same_layer_sims else 0,
        "cross_layer_std_cos": float(np.std(cross_layer_sims)) if cross_layer_sims else 0,
    }

    results = {
        "word_correlation_matrix": {
            "words": words_list,
            "matrix": [[float(word_corr[i, j]) for j in range(len(words_list))]
                        for i in range(len(words_list))],
        },
        "fruit_shared_heads": fruit_shared_heads,
        "n_fruit_shared_heads": len(fruit_shared_heads),
        "category_specificity": categories,
        "cluster_info": cluster_info,
        "head_similarity": head_similarity,
    }

    print(f"\n  跨词头共享分析摘要:")
    print(f"    水果共享头数: {len(fruit_shared_heads)}")
    for cat, info in categories.items():
        print(f"    {cat} 特异性: mean={info['mean_specificity']:.3f}, "
              f"max={info['max_specificity']:.3f}, "
              f"high_spec_heads={info['n_high_specificity_heads']}")
    print(f"    同层头间cos: {head_similarity['same_layer_mean_cos']:.3f}")
    print(f"    跨层头间cos: {head_similarity['cross_layer_mean_cos']:.3f}")

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
    print(f"Phase CLX: 注意力头的语义角色 — {model_name}")
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

    # 获取头数信息
    layers = get_layers(model)
    if layers:
        sa0 = layers[0].self_attn
        n_q = get_n_heads(sa0, model, model_info.d_model)
        n_kv = get_n_kv_heads(sa0, model)
        print(f"Attention: n_q_heads={n_q}, n_kv_heads={n_kv}, "
              f"head_dim={model_info.d_model // n_q}, GQA={'Yes' if n_q != n_kv else 'No'}")

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

    # P698: 注意力头的logit贡献分解
    all_results["P698"] = p698_head_logit_decomposition(model, tokenizer, model_info, W_embed, W_U, device)
    gc.collect()

    # P699: 语义头vs语法头vs抑制头的分类
    all_results["P699"] = p699_head_classification(model, tokenizer, model_info, W_embed, W_U, device)
    gc.collect()

    # P700: 跨词注意力头共享分析
    all_results["P700"] = p700_cross_word_head_sharing(model, tokenizer, model_info, W_embed, W_U, device)
    gc.collect()

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUT_DIR / f"phase_clx_{model_name}_{timestamp}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {out_file}")

    # 打印关键发现摘要
    print("\n" + "="*70)
    print("关键发现摘要")
    print("="*70)
    print(f"\n模型: {model_name}")

    # P698 摘要
    if "P698" in all_results:
        print("\n--- P698: 头logit分解 ---")
        for word in ["apple", "banana", "cat", "the"]:
            if word in all_results["P698"]:
                wr = all_results["P698"][word]
                sp = wr.get("sparsity", {})
                print(f"  {word}: pos/neg={wr.get('positive_heads',0)}/{wr.get('negative_heads',0)}, "
                      f"top10%_ratio={sp.get('top10pct_ratio',0):.3f}, "
                      f"top1_ratio={sp.get('top1_ratio',0):.3f}")

    # P699 摘要
    if "P699" in all_results:
        print("\n--- P699: 头分类 ---")
        summary = all_results["P699"].get("_summary", {})
        print(f"  总头数: {summary.get('total_heads_analyzed', 0)}")
        for t, count in summary.get("type_counts", {}).items():
            ratio = summary.get("type_ratios", {}).get(t, 0)
            print(f"  {t}: {count} ({ratio*100:.1f}%)")

    # P700 摘要
    if "P700" in all_results:
        print("\n--- P700: 跨词头共享 ---")
        print(f"  水果共享头数: {all_results['P700'].get('n_fruit_shared_heads', 0)}")
        hs = all_results["P700"].get("head_similarity", {})
        if "same_layer_mean_cos" in hs:
            print(f"  同层头间cos: {hs['same_layer_mean_cos']:.3f}")
            print(f"  跨层头间cos: {hs['cross_layer_mean_cos']:.3f}")

    # 释放模型
    release_model(model)
    print("[model_utils] GPU memory released")


if __name__ == "__main__":
    main()
