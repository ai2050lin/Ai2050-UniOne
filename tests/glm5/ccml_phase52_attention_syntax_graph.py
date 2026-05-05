"""
Phase 52: Attention → Syntax Graph — 注意力还原语法图
======================================================

SLT v3.1 核心假设:
  语言结构 = attention-induced graph
  h_i^{l+1} = Σ_j R_l(i,j) · V_j  →  Transformer = message passing on graph

关键纠偏(来自Phase 51审视):
  1. role ≠ 独立子空间, role = semantic子空间中的方向扰动
  2. role ≠ position, role = f(position, syntax, attention flow)
  3. role_vec(word) = r_global + ε_i (全局共享+词特异)
  4. 单token可有多角色: h_i = Σ role_{i→j} 压缩表示

Phase 52 四个实验:
  52A: 信息流矩阵 — R(i,j) = Σ_h A(i,j,h) * ||V(j,h)||
       → 不是原始attention，而是attention-weighted information flow
  52B: 语法头选择 — 用MI筛选编码语法关系的head
       → 3-6个head通常就够
  52C: 依赖树恢复 — 用R矩阵构建语法树，计算UAS/LAS
       → 对比baseline: 随机/最近邻/cosine similarity
  52D: Head→语法功能分解 — 每个head编码什么语法关系?
       → subject-head / object-head / modifier-head
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import torch
import numpy as np
import gc
import time
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from model_utils import (load_model, get_layers, get_model_info, release_model,
                          safe_decode, get_W_U)


# ============================================================
# 辅助函数
# ============================================================
def get_all_token_hidden_states(model, tokenizer, sentence, n_layers, device):
    """收集句子所有token在所有层的hidden states"""
    from model_utils import collect_layer_outputs
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        embed = model.get_input_embeddings()(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        outputs = collect_layer_outputs(model, embed, position_ids, n_layers)
    
    result = {}
    for li in range(n_layers):
        key = f"L{li}"
        if key in outputs:
            result[li] = outputs[key][0].numpy()
    
    del outputs, embed
    return result, input_ids[0].cpu().numpy()


def find_token_position(token_ids, tokenizer, target_word):
    """找到目标词在token序列中的位置"""
    decoded = [safe_decode(tokenizer, tid) for tid in token_ids]
    positions = []
    for i, d in enumerate(decoded):
        if target_word.lower() in d.lower().strip():
            positions.append(i)
    if positions:
        return positions
    for i, d in enumerate(decoded):
        if target_word.lower() in d.lower():
            positions.append(i)
    return positions


def get_attention_and_values(model, tokenizer, sentence, n_layers, device, target_layers=None):
    """
    收集指定层的attention权重和V投影
    
    返回:
      attn_weights: {layer_idx: array[n_heads, seq_len, seq_len]}
      v_projections: {layer_idx: array[n_heads, seq_len, d_head]}
      input_ids: array[seq_len]
    """
    if target_layers is None:
        target_layers = list(range(n_layers))
    
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    layers = get_layers(model)
    attn_weights = {}
    v_projections = {}
    
    def make_attn_hook(li, n_heads, d_head):
        def hook(module, input, output):
            # output通常包含(attn_output, attn_weights, past_key_value)
            if len(output) > 1 and output[1] is not None:
                attn_weights[li] = output[1].detach().float().cpu()[0].numpy()
        return hook
    
    # 更完整的方法: 用output_attentions=True
    with torch.no_grad():
        embed = model.get_input_embeddings()(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        try:
            outputs = model(inputs_embeds=embed, position_ids=position_ids, 
                           output_attentions=True)
            all_attn = outputs.attentions  # tuple of [1, n_heads, seq, seq]
            
            for li in target_layers:
                if li < len(all_attn) and all_attn[li] is not None:
                    attn_weights[li] = all_attn[li].detach().float().cpu()[0].numpy()
        except Exception as e:
            print(f"  [get_attention_and_values] Failed: {e}")
    
    # 获取V投影 — 需要手动计算
    # V_j^h = h_j @ W_v[:, h*d_head:(h+1)*d_head]
    for li in target_layers:
        if li not in attn_weights:
            continue
        
        layer = layers[li]
        attn_mod = layer.self_attn if hasattr(layer, 'self_attn') else layer.attention
        
        # 获取W_v权重
        W_v = attn_mod.v_proj.weight.detach().float().cpu().numpy()  # [d_model, d_model] or [n_heads*d_head, d_model]
        
        # 获取hidden states
        h_dict, _ = get_all_token_hidden_states(model, tokenizer, sentence, n_layers, device)
        if li not in h_dict:
            continue
        
        h = h_dict[li]  # [seq_len, d_model]
        seq_len = h.shape[0]
        
        # 计算V = h @ W_v^T
        V = h @ W_v.T  # [seq_len, d_model] or [seq_len, n_heads*d_head]
        
        n_heads = attn_weights[li].shape[0]
        d_head = V.shape[1] // n_heads
        
        # 拆分为各head的V
        v_proj = np.zeros((n_heads, seq_len, d_head))
        for h_idx in range(n_heads):
            v_proj[h_idx] = V[:, h_idx*d_head:(h_idx+1)*d_head]
        
        v_projections[li] = v_proj
        
        del h_dict
        gc.collect()
    
    del embed
    return attn_weights, v_projections, input_ids[0].cpu().numpy()


# ============================================================
# 带依赖标注的句子集
# ============================================================
# 每个句子附带gold-standard依赖关系
# (head_pos, dep_pos, relation_type)
# pos是token在序列中的位置（0-based，0通常是BOS）
# relation_type: nsubj(主语), dobj(直接宾语), det(限定词), aux(助动词), prep(介词), amod(形容词修饰), nmod(名词修饰)

DEP_ANNOTATED_SENTENCES = [
    # 格式: (sentence, [(head_pos, dep_pos, relation_type), ...], word_map)
    # word_map: {pos: word} (1-based from tokenizer, 0=BOS)
    
    # 简单及物句
    {
        "sentence": "The cat chases the mouse.",
        "deps": [("chases", "cat", "nsubj"), ("chases", "mouse", "dobj"), 
                 ("cat", "The", "det"), ("mouse", "the", "det")],
        "type": "active_transitive"
    },
    {
        "sentence": "The dog bites the bone.",
        "deps": [("bites", "dog", "nsubj"), ("bites", "bone", "dobj"),
                 ("dog", "The", "det"), ("bone", "the", "det")],
        "type": "active_transitive"
    },
    {
        "sentence": "The man reads the book.",
        "deps": [("reads", "man", "nsubj"), ("reads", "book", "dobj"),
                 ("man", "The", "det"), ("book", "the", "det")],
        "type": "active_transitive"
    },
    {
        "sentence": "The boy throws the ball.",
        "deps": [("throws", "boy", "nsubj"), ("throws", "ball", "dobj"),
                 ("boy", "The", "det"), ("ball", "the", "det")],
        "type": "active_transitive"
    },
    {
        "sentence": "The woman opens the door.",
        "deps": [("opens", "woman", "nsubj"), ("opens", "door", "dobj"),
                 ("woman", "The", "det"), ("door", "the", "det")],
        "type": "active_transitive"
    },
    {
        "sentence": "The teacher writes the letter.",
        "deps": [("writes", "teacher", "nsubj"), ("writes", "letter", "dobj"),
                 ("teacher", "The", "det"), ("letter", "the", "det")],
        "type": "active_transitive"
    },
    {
        "sentence": "The fox hunts the rabbit.",
        "deps": [("hunts", "fox", "nsubj"), ("hunts", "rabbit", "dobj"),
                 ("fox", "The", "det"), ("rabbit", "the", "det")],
        "type": "active_transitive"
    },
    {
        "sentence": "The bear eats the honey.",
        "deps": [("eats", "bear", "nsubj"), ("eats", "honey", "dobj"),
                 ("bear", "The", "det"), ("honey", "the", "det")],
        "type": "active_transitive"
    },
    {
        "sentence": "The lion attacks the deer.",
        "deps": [("attacks", "lion", "nsubj"), ("attacks", "deer", "dobj"),
                 ("lion", "The", "det"), ("deer", "the", "det")],
        "type": "active_transitive"
    },
    {
        "sentence": "The chef cooks the meal.",
        "deps": [("cooks", "chef", "nsubj"), ("cooks", "meal", "dobj"),
                 ("chef", "The", "det"), ("meal", "the", "det")],
        "type": "active_transitive"
    },
    
    # 被动句
    {
        "sentence": "The mouse is chased by the cat.",
        "deps": [("chased", "mouse", "nsubjpass"), ("chased", "is", "aux"),
                 ("chased", "cat", "agent"), ("cat", "by", "case"), 
                 ("mouse", "The", "det"), ("cat", "the", "det")],
        "type": "passive"
    },
    {
        "sentence": "The bone is bitten by the dog.",
        "deps": [("bitten", "bone", "nsubjpass"), ("bitten", "is", "aux"),
                 ("bitten", "dog", "agent"), ("dog", "by", "case"),
                 ("bone", "The", "det"), ("dog", "the", "det")],
        "type": "passive"
    },
    {
        "sentence": "The book is read by the man.",
        "deps": [("read", "book", "nsubjpass"), ("read", "is", "aux"),
                 ("read", "man", "agent"), ("man", "by", "case"),
                 ("book", "The", "det"), ("man", "the", "det")],
        "type": "passive"
    },
    {
        "sentence": "The ball is thrown by the boy.",
        "deps": [("thrown", "ball", "nsubjpass"), ("thrown", "is", "aux"),
                 ("thrown", "boy", "agent"), ("boy", "by", "case"),
                 ("ball", "The", "det"), ("boy", "the", "det")],
        "type": "passive"
    },
    {
        "sentence": "The letter is written by the teacher.",
        "deps": [("written", "letter", "nsubjpass"), ("written", "is", "aux"),
                 ("written", "teacher", "agent"), ("teacher", "by", "case"),
                 ("letter", "The", "det"), ("teacher", "the", "det")],
        "type": "passive"
    },
    
    # 不及物句
    {
        "sentence": "The cat sleeps.",
        "deps": [("sleeps", "cat", "nsubj"), ("cat", "The", "det")],
        "type": "intransitive"
    },
    {
        "sentence": "The dog runs.",
        "deps": [("runs", "dog", "nsubj"), ("dog", "The", "det")],
        "type": "intransitive"
    },
    {
        "sentence": "The bird flies.",
        "deps": [("flies", "bird", "nsubj"), ("bird", "The", "det")],
        "type": "intransitive"
    },
    {
        "sentence": "The man walks.",
        "deps": [("walks", "man", "nsubj"), ("man", "The", "det")],
        "type": "intransitive"
    },
    {
        "sentence": "The woman sings.",
        "deps": [("sings", "woman", "nsubj"), ("woman", "The", "det")],
        "type": "intransitive"
    },
    
    # 带修饰语
    {
        "sentence": "The big cat chases the small mouse.",
        "deps": [("chases", "cat", "nsubj"), ("chases", "mouse", "dobj"),
                 ("cat", "big", "amod"), ("mouse", "small", "amod"),
                 ("cat", "The", "det"), ("mouse", "the", "det")],
        "type": "modified"
    },
    {
        "sentence": "The old man reads the long book.",
        "deps": [("reads", "man", "nsubj"), ("reads", "book", "dobj"),
                 ("man", "old", "amod"), ("book", "long", "amod"),
                 ("man", "The", "det"), ("book", "the", "det")],
        "type": "modified"
    },
    {
        "sentence": "The fast dog chases the slow cat.",
        "deps": [("chases", "dog", "nsubj"), ("chases", "cat", "dobj"),
                 ("dog", "fast", "amod"), ("cat", "slow", "amod"),
                 ("dog", "The", "det"), ("cat", "the", "det")],
        "type": "modified"
    },
    
    # 介词短语
    {
        "sentence": "The cat sits on the mat.",
        "deps": [("sits", "cat", "nsubj"), ("sits", "mat", "nmod"),
                 ("mat", "on", "case"), ("cat", "The", "det"), ("mat", "the", "det")],
        "type": "prepositional"
    },
    {
        "sentence": "The dog runs in the park.",
        "deps": [("runs", "dog", "nsubj"), ("runs", "park", "nmod"),
                 ("park", "in", "case"), ("dog", "The", "det"), ("park", "the", "det")],
        "type": "prepositional"
    },
    
    # 复杂句
    {
        "sentence": "The cat that chased the dog ran.",
        "deps": [("ran", "cat", "nsubj"), ("chased", "cat", "nsubj"), 
                 ("chased", "dog", "dobj"), ("cat", "The", "det"), 
                 ("dog", "the", "det")],
        "type": "relative_clause"
    },
    {
        "sentence": "The man who read the book smiled.",
        "deps": [("smiled", "man", "nsubj"), ("read", "man", "nsubj"),
                 ("read", "book", "dobj"), ("man", "The", "det"),
                 ("book", "the", "det")],
        "type": "relative_clause"
    },
]


def resolve_dep_positions(token_ids, tokenizer, deps):
    """
    将依赖标注中的word-based位置转换为token-based位置
    
    Args:
        token_ids: tokenizer输出
        tokenizer: 分词器
        deps: [(head_word, dep_word, relation_type), ...]
    
    Returns:
        [(head_pos, dep_pos, relation_type), ...] — 位置是token index
    """
    resolved = []
    for head_word, dep_word, rel_type in deps:
        head_pos_list = find_token_position(token_ids, tokenizer, head_word)
        dep_pos_list = find_token_position(token_ids, tokenizer, dep_word)
        
        if head_pos_list and dep_pos_list:
            # 取第一个匹配的位置
            resolved.append((head_pos_list[0], dep_pos_list[0], rel_type))
    
    return resolved


# ============================================================
# 52A: 信息流矩阵
# ============================================================
def exp_52a_information_flow_matrix(model, tokenizer, info, model_name):
    """
    核心问题: 原始attention vs 信息流 R(i,j)=Σ_h A(i,j,h)*||V(j,h)||
    哪个更好编码语法关系?
    """
    print("\n" + "="*70)
    print("52A: Information Flow Matrix — 信息流矩阵")
    print("="*70)
    
    n_layers = info.n_layers
    d_model = info.d_model
    device = next(model.parameters()).device
    sample_layers = list(range(0, n_layers, max(1, n_layers//8))) + [n_layers-1]
    sample_layers = sorted(set(sample_layers))
    
    # 选几个句子做详细分析
    test_sents = DEP_ANNOTATED_SENTENCES[:10]
    
    print("\n--- Step 1: Compute R(i,j) = Σ_h A(i,j,h) * ||V(j,h)|| ---")
    
    for sent_data in test_sents[:3]:
        sentence = sent_data["sentence"]
        print(f"\n  Sentence: {sentence}")
        
        attn_weights, v_projections, token_ids = get_attention_and_values(
            model, tokenizer, sentence, n_layers, device, target_layers=sample_layers
        )
        
        # 解码tokens
        tokens = [safe_decode(tokenizer, tid) for tid in token_ids]
        print(f"  Tokens: {tokens}")
        
        # 解析依赖
        resolved_deps = resolve_dep_positions(token_ids, tokenizer, sent_data["deps"])
        
        for li in [sample_layers[len(sample_layers)//2], sample_layers[-1]]:
            if li not in attn_weights or li not in v_projections:
                continue
            
            A = attn_weights[li]  # [n_heads, seq, seq]
            V = v_projections[li]  # [n_heads, seq, d_head]
            n_heads = A.shape[0]
            seq_len = A.shape[1]
            
            # 1. 原始attention (平均)
            mean_attn = A.mean(axis=0)  # [seq, seq]
            
            # 2. 信息流 R(i,j) = Σ_h A(i,j,h) * ||V(j,h)||
            R = np.zeros((seq_len, seq_len))
            for h_idx in range(n_heads):
                v_norms = np.linalg.norm(V[h_idx], axis=1)  # [seq]
                R += A[h_idx] * v_norms[np.newaxis, :]  # A[i,j]*||V[j]||
            
            # 3. 距离正则化
            tau = 3.0
            dist_mat = np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)[None, :])
            R_dist = R * np.exp(-dist_mat / tau)
            mean_attn_dist = mean_attn * np.exp(-dist_mat / tau)
            
            # 对每种矩阵，检查: 对每个dep，head是否在top-k中?
            results = {"raw_attn": 0, "info_flow": 0, "attn_dist": 0, "info_dist": 0, "random": 0}
            total_deps = 0
            
            for head_pos, dep_pos, rel_type in resolved_deps:
                if head_pos >= seq_len or dep_pos >= seq_len:
                    continue
                
                total_deps += 1
                
                # 对dep_pos，看head_pos是否在top-3最相关位置中
                k = min(3, seq_len - 1)
                
                # Raw attention: dep→head (dep attends to head?)
                top_attn = np.argsort(mean_attn[dep_pos, 1:])[-k:] + 1  # skip BOS
                if head_pos in top_attn:
                    results["raw_attn"] += 1
                
                # Info flow: dep→head
                top_R = np.argsort(R[dep_pos, 1:])[-k:] + 1
                if head_pos in top_R:
                    results["info_flow"] += 1
                
                # Distance-regularized attention
                top_attn_d = np.argsort(mean_attn_dist[dep_pos, 1:])[-k:] + 1
                if head_pos in top_attn_d:
                    results["attn_dist"] += 1
                
                # Distance-regularized info flow
                top_R_d = np.argsort(R_dist[dep_pos, 1:])[-k:] + 1
                if head_pos in top_R_d:
                    results["info_dist"] += 1
                
                # Random baseline (top-k from 1..seq_len-1)
                # P(hit) = k/(seq_len-1) for one dep
                results["random"] += k / (seq_len - 1)
            
            if total_deps > 0:
                print(f"\n  Layer {li} — Dependency head recovery (top-{k}):")
                for method, count in results.items():
                    if method == "random":
                        print(f"    {method:12s}: {count/total_deps:.3f} (expected by chance)")
                    else:
                        print(f"    {method:12s}: {count/total_deps:.3f} ({count}/{total_deps})")
        
        del attn_weights, v_projections
        gc.collect()
    
    # --- Step 2: 批量统计 ---
    print("\n--- Step 2: Batch Statistics ---")
    print("  Averaging across all sentences and layers...")
    
    method_scores = defaultdict(list)  # method -> [acc_per_sentence_per_layer]
    
    for sent_data in test_sents:
        sentence = sent_data["sentence"]
        
        try:
            attn_weights, v_projections, token_ids = get_attention_and_values(
                model, tokenizer, sentence, n_layers, device, target_layers=sample_layers
            )
        except:
            continue
        
        resolved_deps = resolve_dep_positions(token_ids, tokenizer, sent_data["deps"])
        if len(resolved_deps) < 2:
            del attn_weights, v_projections
            gc.collect()
            continue
        
        for li in sample_layers:
            if li not in attn_weights or li not in v_projections:
                continue
            
            A = attn_weights[li]
            V = v_projections[li]
            n_heads = A.shape[0]
            seq_len = A.shape[1]
            
            mean_attn = A.mean(axis=0)
            
            R = np.zeros((seq_len, seq_len))
            for h_idx in range(n_heads):
                v_norms = np.linalg.norm(V[h_idx], axis=1)
                R += A[h_idx] * v_norms[np.newaxis, :]
            
            tau = 3.0
            dist_mat = np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)[None, :])
            R_dist = R * np.exp(-dist_mat / tau)
            mean_attn_dist = mean_attn * np.exp(-dist_mat / tau)
            
            k = min(3, seq_len - 1)
            
            for method_name, matrix in [("raw_attn", mean_attn), ("info_flow", R),
                                         ("attn_dist", mean_attn_dist), ("info_dist", R_dist)]:
                correct = 0
                total = 0
                for head_pos, dep_pos, rel_type in resolved_deps:
                    if head_pos >= seq_len or dep_pos >= seq_len:
                        continue
                    total += 1
                    top_k = np.argsort(matrix[dep_pos, 1:])[-k:] + 1
                    if head_pos in top_k:
                        correct += 1
                
                if total > 0:
                    method_scores[method_name].append(correct / total)
        
        del attn_weights, v_projections
        gc.collect()
    
    print("\n--- Summary: Head Recovery Rate (top-3) ---")
    for method in ["raw_attn", "info_flow", "attn_dist", "info_dist"]:
        if method_scores[method]:
            print(f"  {method:12s}: {np.mean(method_scores[method]):.3f} ± {np.std(method_scores[method]):.3f}")
    
    # Random baseline
    random_scores = []
    for sent_data in test_sents:
        sentence = sent_data["sentence"]
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        seq_len = inputs["input_ids"].shape[1]
        k = min(3, seq_len - 1)
        n_deps = len(sent_data["deps"])
        random_scores.append(k / (seq_len - 1))
    if random_scores:
        print(f"  {'random':12s}: {np.mean(random_scores):.3f}")
    
    print("\n" + "="*70)
    print("52A SUMMARY: Information Flow Matrix")
    print("="*70)


# ============================================================
# 52B: 语法头选择
# ============================================================
def exp_52b_syntactic_head_selection(model, tokenizer, info, model_name):
    """
    核心问题: 哪些attention head编码了语法关系?
    
    方法: 对每个head，计算其attention pattern与依赖关系的MI
    选top-k head
    """
    print("\n" + "="*70)
    print("52B: Syntactic Head Selection — 语法头选择")
    print("="*70)
    
    n_layers = info.n_layers
    d_model = info.d_model
    device = next(model.parameters()).device
    
    # 用中间层（通常语法信息最集中）
    target_layers = list(range(max(0, n_layers//4), min(n_layers, 3*n_layers//4)))
    if n_layers <= 10:
        target_layers = list(range(n_layers))
    
    test_sents = DEP_ANNOTATED_SENTENCES[:15]
    
    print("\n--- Step 1: Compute per-head syntax score ---")
    print("  For each head: what fraction of dependency edges are in top-3 attention?")
    
    head_syntax_scores = {}  # (layer, head_idx) -> score
    
    for li in target_layers:
        for h_idx_guess in range(32):  # 假设最多32个head
            scores_for_this_head = []
            
            for sent_data in test_sents:
                sentence = sent_data["sentence"]
                
                try:
                    attn_weights, _, token_ids = get_attention_and_values(
                        model, tokenizer, sentence, n_layers, device, target_layers=[li]
                    )
                except:
                    continue
                
                if li not in attn_weights:
                    continue
                
                A = attn_weights[li]  # [n_heads, seq, seq]
                
                if h_idx_guess >= A.shape[0]:
                    break
                
                resolved_deps = resolve_dep_positions(token_ids, tokenizer, sent_data["deps"])
                seq_len = A.shape[1]
                
                correct = 0
                total = 0
                k = min(3, seq_len - 1)
                
                for head_pos, dep_pos, rel_type in resolved_deps:
                    if head_pos >= seq_len or dep_pos >= seq_len:
                        continue
                    total += 1
                    top_k = np.argsort(A[h_idx_guess, dep_pos, 1:])[-k:] + 1
                    if head_pos in top_k:
                        correct += 1
                
                if total > 0:
                    scores_for_this_head.append(correct / total)
                
                del attn_weights
                gc.collect()
            
            if scores_for_this_head:
                head_syntax_scores[(li, h_idx_guess)] = np.mean(scores_for_this_head)
            
            # 如果head数不够就跳过
            if h_idx_guess >= 31:
                break
        
        # 打印每层的top heads
        layer_heads = [(k, v) for k, v in head_syntax_scores.items() if k[0] == li]
        if layer_heads:
            layer_heads.sort(key=lambda x: -x[1])
            top3 = layer_heads[:3]
            print(f"  Layer {li}: Top heads = {[(h, f'{s:.3f}') for (_, h), s in top3]}")
    
    # 全局top heads
    print("\n--- Step 2: Global Top Heads ---")
    all_heads = sorted(head_syntax_scores.items(), key=lambda x: -x[1])
    n_top = min(10, len(all_heads))
    
    print(f"  Top-{n_top} syntax-encoding heads:")
    for (li, hi), score in all_heads[:n_top]:
        print(f"    Layer {li:2d}, Head {hi:2d}: score={score:.3f}")
    
    # 随机baseline
    random_score = np.mean([3.0 / (len(s["deps"]) + 5) for s in test_sents])
    print(f"  Random baseline: ~{random_score:.3f}")
    
    # --- Step 3: 用top-k heads构建组合R矩阵 ---
    print("\n--- Step 3: Combined R from Top Heads ---")
    
    top_k_heads = [(li, hi) for (li, hi), _ in all_heads[:6]]
    
    combined_scores = []
    individual_scores = []
    
    for sent_data in test_sents[:10]:
        sentence = sent_data["sentence"]
        
        # 收集所有top head的attention
        needed_layers = sorted(set(li for li, _ in top_k_heads))
        
        try:
            attn_weights, _, token_ids = get_attention_and_values(
                model, tokenizer, sentence, n_layers, device, target_layers=needed_layers
            )
        except:
            continue
        
        resolved_deps = resolve_dep_positions(token_ids, tokenizer, sent_data["deps"])
        
        # 对每个需要评估的层
        for li in needed_layers:
            if li not in attn_weights:
                continue
            
            A = attn_weights[li]
            seq_len = A.shape[1]
            
            # Combined R from top heads at this layer
            heads_at_layer = [hi for l, hi in top_k_heads if l == li]
            if not heads_at_layer:
                continue
            
            R_combined = np.zeros((seq_len, seq_len))
            for hi in heads_at_layer:
                if hi < A.shape[0]:
                    R_combined += A[hi]
            
            # Mean attention
            R_mean = A.mean(axis=0)
            
            k = min(3, seq_len - 1)
            
            # Combined
            correct_comb = 0
            correct_mean = 0
            total = 0
            
            for head_pos, dep_pos, rel_type in resolved_deps:
                if head_pos >= seq_len or dep_pos >= seq_len:
                    continue
                total += 1
                
                top_k_comb = np.argsort(R_combined[dep_pos, 1:])[-k:] + 1
                top_k_mean = np.argsort(R_mean[dep_pos, 1:])[-k:] + 1
                
                if head_pos in top_k_comb:
                    correct_comb += 1
                if head_pos in top_k_mean:
                    correct_mean += 1
            
            if total > 0:
                combined_scores.append(correct_comb / total)
                individual_scores.append(correct_mean / total)
        
        del attn_weights
        gc.collect()
    
    if combined_scores:
        print(f"  Top-6 heads combined: {np.mean(combined_scores):.3f}")
        print(f"  Mean of all heads:    {np.mean(individual_scores):.3f}")
    
    print("\n" + "="*70)
    print("52B SUMMARY: Syntactic Head Selection")
    print("="*70)


# ============================================================
# 52C: 依赖树恢复 (UAS/LAS)
# ============================================================
def exp_52c_dependency_tree_recovery(model, tokenizer, info, model_name):
    """
    核心问题: 从attention能恢复多少语法树结构?
    
    ★ 关键修正:
    在因果(自回归)模型中, token i 只能 attend 到 j < i
    依赖关系中 head→dependent 的方向:
      - 如果 head 在 dependent 左边: dependent attend to head ✓ (因果模型自然)
      - 如果 head 在 dependent 右边: 不可能! → 需要用 A[head, dependent] 方向
    
    正确方法: 对每对 (head, dep), 检查:
      1. A[dep, head] (dep attend to head) — 当 head < dep
      2. A[head, dep] (head attend to dep) — 当 head > dep (罕见但存在)
    
    简化: 由于大部分语法关系中 head 在 dependent 前面或后面,
          我们对每个 token i, 检查它"最关注谁"和"谁最关注它"
    
    更简单的方法: 对每条依赖边 (head, dep), 
      检查在 A[dep_pos, :] 或 A[head_pos, :] 中对方是否在 top-k
    """
    print("\n" + "="*70)
    print("52C: Dependency Tree Recovery — 依赖树恢复")
    print("="*70)
    
    n_layers = info.n_layers
    d_model = info.d_model
    device = next(model.parameters()).device
    
    sample_layers = list(range(0, n_layers, max(1, n_layers//6))) + [n_layers-1]
    sample_layers = sorted(set(sample_layers))
    
    test_sents = DEP_ANNOTATED_SENTENCES[:20]
    
    print("\n--- Step 1: Attention-based dependency recovery ---")
    print("  For each (head, dep) pair, check if attention recovers the edge")
    
    # 存储各方法在各层的得分
    uas_scores = defaultdict(lambda: defaultdict(list))  # method -> layer -> [acc]
    
    for sent_data in test_sents:
        sentence = sent_data["sentence"]
        
        # Tokenize and get token positions
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        token_ids = inputs["input_ids"][0].cpu().numpy()
        tokens = [safe_decode(tokenizer, tid) for tid in token_ids]
        seq_len = len(tokens)
        
        # Resolve dependency positions
        resolved_deps = resolve_dep_positions(token_ids, tokenizer, sent_data["deps"])
        
        if len(resolved_deps) < 2:
            continue
        
        # Get hidden states for cosine baseline
        h_dict, _ = get_all_token_hidden_states(model, tokenizer, sentence, n_layers, device)
        
        # Get attention weights
        try:
            attn_out = model(input_ids=inputs["input_ids"], output_attentions=True)
            all_attn = attn_out.attentions
        except:
            del h_dict
            gc.collect()
            continue
        
        for li in sample_layers:
            if li >= len(all_attn) or all_attn[li] is None:
                continue
            if li not in h_dict:
                continue
            
            A = all_attn[li].detach().float().cpu().numpy()[0]  # [n_heads, seq, seq]
            mean_attn = A.mean(axis=0)  # [seq, seq]
            h = h_dict[li]
            
            # Cosine similarity
            h_norm = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)
            cos_mat = h_norm @ h_norm.T
            
            # 信息流 R (without V projection for simplicity — use A * norm_decay)
            # 简化: 用 mean_attn + 距离正则
            tau = 3.0
            dist_mat = np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)[None, :])
            attn_dist = mean_attn * np.exp(-dist_mat / tau)
            
            # ★★★ 核心修正: 对每条依赖 (head_pos, dep_pos)
            # 方法A: 检查 dep 是否 attend to head (dep_pos > head_pos 时才有意义)
            # 方法B: 检查 head 是否 attend to dep (head_pos > dep_pos 时才有意义)
            # 方法C: 综合方向: max(A[dep,head], A[head,dep])
            # 方法D: 只看 A[dep, head] (dep→head方向)
            
            for method_name, matrix, direction in [
                ("attn_dep2head", mean_attn, "dep→head"),      # dep attend to head
                ("attn_head2dep", mean_attn, "head→dep"),      # head attend to dep  
                ("attn_bidir", mean_attn, "bidir"),            # bidirectional
                ("attn_dist_d2h", attn_dist, "dep→head"),
                ("cosine_bidir", cos_mat, "bidir"),
            ]:
                uas_correct = 0
                total = 0
                
                for head_pos, dep_pos, rel_type in resolved_deps:
                    if head_pos >= seq_len or dep_pos >= seq_len:
                        continue
                    if head_pos == dep_pos:
                        continue
                    
                    total += 1
                    
                    if direction == "dep→head":
                        # dep attend to head: 检查 matrix[dep_pos, head_pos] 在 top-k 中
                        row = matrix[dep_pos, 1:dep_pos+1] if dep_pos > 1 else matrix[dep_pos, :1]  # 因果mask
                        if len(row) == 0:
                            continue
                        k = min(3, len(row))
                        top_k = np.argsort(row)[-k:] + 1
                        if head_pos in top_k:
                            uas_correct += 1
                    
                    elif direction == "head→dep":
                        # head attend to dep: 只有 head > dep 时才有意义
                        if head_pos <= dep_pos:
                            # head 在 dep 左边，不能 attend 到 dep
                            # 检查 dep 的右侧是否有 head (不可能在因果模型中)
                            continue  # skip this pair for this method
                        row = matrix[head_pos, 1:head_pos+1]
                        if len(row) == 0:
                            continue
                        k = min(3, len(row))
                        top_k = np.argsort(row)[-k:] + 1
                        if dep_pos in top_k:
                            uas_correct += 1
                        # 但这减少了total，不公平，所以补上
                        total  # 不减total
                    
                    elif direction == "bidir":
                        # 双向: 检查 max(A[dep,head], A[head,dep]) 
                        # 从 dep 的角度: top-k 包含 head?
                        # 从 head 的角度: top-k 包含 dep?
                        hit = False
                        
                        # dep→head (如果 dep > head)
                        if dep_pos > head_pos:
                            row = matrix[dep_pos, 1:dep_pos+1]
                            if len(row) > 0:
                                k = min(3, len(row))
                                top_k = np.argsort(row)[-k:] + 1
                                if head_pos in top_k:
                                    hit = True
                        
                        # head→dep (如果 head > dep)
                        if head_pos > dep_pos:
                            row = matrix[head_pos, 1:head_pos+1]
                            if len(row) > 0:
                                k = min(3, len(row))
                                top_k = np.argsort(row)[-k:] + 1
                                if dep_pos in top_k:
                                    hit = True
                        
                        if hit:
                            uas_correct += 1
                
                if total > 0:
                    uas_scores[method_name][li].append(uas_correct / total)
        
        del h_dict, all_attn
        gc.collect()
    
    # --- 打印结果 ---
    print("\n--- UAS by Method and Layer ---")
    
    best_overall = {}
    for method in ["attn_dep2head", "attn_bidir", "attn_dist_d2h", "cosine_bidir"]:
        if not uas_scores[method]:
            continue
        
        layer_means = {}
        for li in sorted(uas_scores[method].keys()):
            vals = uas_scores[method][li]
            if vals:
                layer_means[li] = np.mean(vals)
        
        if layer_means:
            best_layer = max(layer_means, key=layer_means.get)
            best_score = layer_means[best_layer]
            best_overall[method] = (best_score, best_layer)
            
            key_layers = sorted(layer_means.keys())[::max(1, len(layer_means)//5)]
            layer_strs = [f"L{li}={layer_means[li]:.3f}" for li in key_layers]
            print(f"  {method:18s}: best={best_score:.3f}@L{best_layer}, [{', '.join(layer_strs)}]")
    
    # Random baseline
    print(f"  {'random':18s}: ~0.33 (top-3 from ~10 positions)")
    
    # --- Step 2: Per-relation-type analysis ---
    print("\n--- Step 2: Per-Relation-Type Recovery ---")
    
    rel_scores = defaultdict(lambda: defaultdict(list))  # rel_type -> method -> [acc]
    
    for sent_data in test_sents:
        sentence = sent_data["sentence"]
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        token_ids = inputs["input_ids"][0].cpu().numpy()
        seq_len = len(token_ids)
        
        resolved_deps = resolve_dep_positions(token_ids, tokenizer, sent_data["deps"])
        
        try:
            attn_out = model(input_ids=inputs["input_ids"], output_attentions=True)
            all_attn = attn_out.attentions
        except:
            continue
        
        # 用最佳层
        best_li = max(best_overall.values(), key=lambda x: x[0])[1] if best_overall else n_layers // 2
        
        for li in [best_li]:
            if li >= len(all_attn) or all_attn[li] is None:
                continue
            
            A = all_attn[li].detach().float().cpu().numpy()[0]
            mean_attn = A.mean(axis=0)
            
            for head_pos, dep_pos, rel_type in resolved_deps:
                if head_pos >= seq_len or dep_pos >= seq_len or head_pos == dep_pos:
                    continue
                
                # dep→head method
                if dep_pos > head_pos:
                    row = mean_attn[dep_pos, 1:dep_pos+1]
                    if len(row) > 0:
                        k = min(3, len(row))
                        top_k = np.argsort(row)[-k:] + 1
                        hit = head_pos in top_k
                        rel_scores[rel_type]["attn_dep2head"].append(float(hit))
                
                # bidir
                hit = False
                if dep_pos > head_pos:
                    row = mean_attn[dep_pos, 1:dep_pos+1]
                    if len(row) > 0:
                        k = min(3, len(row))
                        top_k = np.argsort(row)[-k:] + 1
                        if head_pos in top_k:
                            hit = True
                if head_pos > dep_pos:
                    row = mean_attn[head_pos, 1:head_pos+1]
                    if len(row) > 0:
                        k = min(3, len(row))
                        top_k = np.argsort(row)[-k:] + 1
                        if dep_pos in top_k:
                            hit = True
                rel_scores[rel_type]["bidir"].append(float(hit))
        
        del all_attn
        gc.collect()
    
    for rel_type in ["nsubj", "dobj", "det", "amod"]:
        if rel_type in rel_scores:
            for method in ["attn_dep2head", "bidir"]:
                vals = rel_scores[rel_type].get(method, [])
                if vals:
                    print(f"  {rel_type:10s} ({method:16s}): {np.mean(vals):.3f} (N={len(vals)})")
    
    print("\n" + "="*70)
    print("52C SUMMARY: Dependency Tree Recovery")
    print("="*70)


# ============================================================
# 52D: Head → 语法功能分解
# ============================================================
def exp_52d_head_syntax_function(model, tokenizer, info, model_name):
    """
    核心问题: 每个attention head编码什么语法关系?
    
    方法: 对每个head，检查其attention pattern在不同语法关系上的特异性
    """
    print("\n" + "="*70)
    print("52D: Head → Syntax Function Decomposition — 语法功能分解")
    print("="*70)
    
    n_layers = info.n_layers
    d_model = info.d_model
    device = next(model.parameters()).device
    
    # 中间层（语法最集中）
    mid_layers = list(range(max(0, n_layers//4), min(n_layers, 3*n_layers//4)))
    if len(mid_layers) > 8:
        step = len(mid_layers) // 8
        mid_layers = mid_layers[::step]
    
    test_sents = DEP_ANNOTATED_SENTENCES[:15]
    
    # 关系类型
    REL_TYPES = ["nsubj", "dobj", "nsubjpass", "agent", "amod", "det", "aux", "case", "nmod"]
    
    print("\n--- Step 1: Per-head attention to specific relation types ---")
    print("  For each head: average attention from dep to head, grouped by relation type")
    
    # head_attn_by_rel[(layer, head_idx)][rel_type] = [attn_values]
    head_attn_by_rel = defaultdict(lambda: defaultdict(list))
    
    for sent_data in test_sents:
        sentence = sent_data["sentence"]
        
        try:
            attn_weights, _, token_ids = get_attention_and_values(
                model, tokenizer, sentence, n_layers, device, target_layers=mid_layers
            )
        except:
            continue
        
        resolved_deps = resolve_dep_positions(token_ids, tokenizer, sent_data["deps"])
        
        for li in mid_layers:
            if li not in attn_weights:
                continue
            
            A = attn_weights[li]  # [n_heads, seq, seq]
            n_heads = A.shape[0]
            seq_len = A.shape[1]
            
            for head_pos, dep_pos, rel_type in resolved_deps:
                if head_pos >= seq_len or dep_pos >= seq_len:
                    continue
                
                for h_idx in range(n_heads):
                    # ★ 修正: 使用双向attention
                    # 如果 dep > head: dep 可以 attend to head (A[dep, head])
                    # 如果 head > dep: head 可以 attend to dep (A[head, dep])
                    if dep_pos > head_pos:
                        attn_val = A[h_idx, dep_pos, head_pos]
                    elif head_pos > dep_pos:
                        attn_val = A[h_idx, head_pos, dep_pos]
                    else:
                        attn_val = 0
                    
                    head_attn_by_rel[(li, h_idx)][rel_type].append(attn_val)
        
        del attn_weights
        gc.collect()
    
    # 分析: 哪个head对哪种关系最特异?
    print("\n--- Step 2: Head Specialization ---")
    print("  Which head pays most attention to which relation type?")
    
    # 对每种关系类型，找最特异的head
    for rel_type in ["nsubj", "dobj", "det", "amod"]:
        head_scores = []
        
        for (li, h_idx), rel_dict in head_attn_by_rel.items():
            if rel_type not in rel_dict or len(rel_dict[rel_type]) < 3:
                continue
            
            # 该head对该关系的平均注意力
            mean_attn_to_rel = np.mean(rel_dict[rel_type])
            
            # 该head对其他关系的平均注意力
            other_attns = []
            for other_rel, vals in rel_dict.items():
                if other_rel != rel_type:
                    other_attns.extend(vals)
            
            if other_attns:
                mean_attn_other = np.mean(other_attns)
                specificity = mean_attn_to_rel / (mean_attn_other + 1e-8)
                head_scores.append((li, h_idx, mean_attn_to_rel, specificity))
        
        if head_scores:
            head_scores.sort(key=lambda x: -x[3])
            print(f"\n  Relation: {rel_type}")
            print(f"  Top-5 specialized heads:")
            for li, h_idx, mean_attn, spec in head_scores[:5]:
                print(f"    L{li:2d} H{h_idx:2d}: attn={mean_attn:.4f}, specificity={spec:.2f}")
    
    # --- Step 3: 关系类型分类器 ---
    print("\n--- Step 3: Relation Type Classification from Head Patterns ---")
    print("  Can we classify relation type from head attention patterns?")
    
    # 构建特征: 对每个(dep, head)对，特征是所有head的attn值
    X_features = []
    y_labels = []
    
    # 需要知道n_heads
    n_heads_ref = None
    
    for sent_data in test_sents:
        sentence = sent_data["sentence"]
        
        try:
            attn_weights, _, token_ids = get_attention_and_values(
                model, tokenizer, sentence, n_layers, device, target_layers=mid_layers
            )
        except:
            continue
        
        resolved_deps = resolve_dep_positions(token_ids, tokenizer, sent_data["deps"])
        
        for li in mid_layers:
            if li not in attn_weights:
                continue
            
            A = attn_weights[li]
            n_heads = A.shape[0]
            if n_heads_ref is None:
                n_heads_ref = n_heads
            seq_len = A.shape[1]
            
            for head_pos, dep_pos, rel_type in resolved_deps:
                if head_pos >= seq_len or dep_pos >= seq_len:
                    continue
                
                # 特征: 所有head从dep→head的attention值
                feat = A[:, dep_pos, head_pos]  # [n_heads]
                X_features.append(feat)
                y_labels.append(rel_type)
        
        del attn_weights
        gc.collect()
    
    if len(X_features) > 20 and len(set(y_labels)) >= 3:
        X = np.array(X_features)
        y = np.array(y_labels)
        
        # 过滤只有少量样本的类
        class_counts = defaultdict(int)
        for label in y:
            class_counts[label] += 1
        valid_classes = {c for c, n in class_counts.items() if n >= 5}
        
        mask = np.array([label in valid_classes for label in y])
        X = X[mask]
        y = y[mask]
        
        if len(set(y)) >= 2 and len(X) > 20:
            try:
                clf = LogisticRegression(max_iter=2000, C=1.0)
                scores = cross_val_score(clf, X, y, cv=min(5, len(X)//10), scoring='accuracy')
                random_baseline = max(class_counts[c] for c in valid_classes) / len(y)
                print(f"  Relation classification from head attention patterns:")
                print(f"    Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
                print(f"    Random baseline: {random_baseline:.3f}")
                print(f"    N_samples: {len(X)}, N_classes: {len(set(y))}")
            except Exception as e:
                print(f"  Classification failed: {e}")
    
    print("\n" + "="*70)
    print("52D SUMMARY: Head → Syntax Function")
    print("="*70)


# ============================================================
# 主函数
# ============================================================
EXPERIMENTS = {
    1: ("52A: Information Flow Matrix", exp_52a_information_flow_matrix),
    2: ("52B: Syntactic Head Selection", exp_52b_syntactic_head_selection),
    3: ("52C: Dependency Tree Recovery (UAS/LAS)", exp_52c_dependency_tree_recovery),
    4: ("52D: Head → Syntax Function", exp_52d_head_syntax_function),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, 
                        choices=['deepseek7b', 'glm4', 'qwen3'],
                        help='Model to test')
    parser.add_argument('--exp', type=int, required=True,
                        choices=list(EXPERIMENTS.keys()),
                        help='Experiment number (1-4)')
    args = parser.parse_args()
    
    model_name = args.model
    exp_num = args.exp
    
    print(f"\n{'='*70}")
    print(f"Phase 52: Attention → Syntax Graph — 注意力还原语法图")
    print(f"Model: {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    print(f"  Model: {info.name}, Layers: {info.n_layers}, d_model: {info.d_model}")
    
    exp_name, exp_fn = EXPERIMENTS[exp_num]
    exp_fn(model, tokenizer, info, model_name)
    
    release_model(model)
    
    print(f"\n{'='*70}")
    print(f"Phase 52 COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
