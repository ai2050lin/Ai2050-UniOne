"""
Phase CLXXXIII: 位置敏感纤维分析 — 修复Mean-Pooling的致命缺陷
==================================================

核心问题: Phase CLXXXII发现英文时态base_ratio=0.81(弱纤维)
但时态信息在动词token位置上，mean-pooling混合了所有token!

修复方案: 
M1: Last-token隐藏状态 — 最常用的decoder-only模型表示方式
M2: Verb-position隐藏状态 — 直接提取动词token位置的表示
M3: 跨层演化 — 时态方向在不同层的纤维性变化

预测:
- 动词位置的时态方向应该比mean-pooling更强(更纤维化)
- 浅层→深层: 时态从流形操作→方向操作(纤维化递增)

运行方式:
  python tests/glm5/position_sensitive_fiber.py --model qwen3 --test m1
  python tests/glm5/position_sensitive_fiber.py --model glm4 --test m2
  python tests/glm5/position_sensitive_fiber.py --model deepseek7b --test m3
  python tests/glm5/position_sensitive_fiber.py --model qwen3 --test all
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import gc
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from collections import defaultdict


# ============================================================
# 英文测试集 (增强版: 明确标注动词位置)
# ============================================================

TENSE_PAIRS = [
    # (present_sentence, past_sentence, present_verb, past_verb)
    # 动词在句子中的位置需要通过tokenization确定
    ("The cat walks every day", "The cat walked yesterday", "walk", "walked"),
    ("The dog runs fast", "The dog ran fast", "run", "ran"),
    ("The bird flies high", "The bird flew high", "fly", "flew"),
    ("The tree grows tall", "The tree grew tall", "grow", "grew"),
    ("The fish swims deep", "The fish swam deep", "swim", "swam"),
    ("She sees the mountain", "She saw the mountain", "see", "saw"),
    ("He takes the book", "He took the book", "take", "took"),
    ("The child comes home", "The child came home", "come", "came"),
    ("The woman drives slowly", "The woman drove slowly", "drive", "drove"),
    ("The river flows east", "The river flowed east", "flow", "flowed"),
    ("The star shines bright", "The star shone bright", "shine", "shone"),
    ("The cloud moves slowly", "The cloud moved slowly", "move", "moved"),
    ("The door opens wide", "The door opened wide", "open", "opened"),
    ("The phone rings loud", "The phone rang loud", "ring", "rang"),
    ("The flower blooms late", "The flower bloomed late", "bloom", "bloomed"),
    ("I think about it", "I thought about it", "think", "thought"),
    ("She writes a letter", "She wrote a letter", "write", "wrote"),
    ("The bridge stands firm", "The bridge stood firm", "stand", "stood"),
    ("The table holds weight", "The table held weight", "hold", "held"),
    ("The river flows north", "The river flowed north", "flow", "flowed"),
]

POLARITY_PAIRS = [
    ("The cat is here", "The cat is not here", "cat"),
    ("The dog is happy", "The dog is not happy", "dog"),
    ("The book was found", "The book was not found", "book"),
    ("I like the car", "I do not like the car", "car"),
    ("She knows the answer", "She does not know the answer", "answer"),
    ("The house is big", "The house is not big", "house"),
    ("The river flows north", "The river does not flow north", "river"),
    ("He can swim", "He cannot swim", "swim"),
    ("The bird will come", "The bird will not come", "bird"),
    ("The door was closed", "The door was not closed", "door"),
    ("The phone is working", "The phone is not working", "phone"),
    ("The flower has bloomed", "The flower has not bloomed", "flower"),
    ("I understand the plan", "I do not understand the plan", "plan"),
    ("She likes the movie", "She does not like the movie", "movie"),
    ("The bridge is safe", "The bridge is not safe", "bridge"),
    ("The child was playing", "The child was not playing", "child"),
    ("The star is visible", "The star is not visible", "star"),
    ("The cloud disappeared", "The cloud did not disappear", "cloud"),
    ("The key works well", "The key does not work well", "key"),
    ("The table holds weight", "The table does not hold weight", "table"),
]

SEMANTIC_PAIRS = [
    # (sentence1, sentence2) — 不同名词，同时态
    ("The cat walks every day", "The dog walks every day"),
    ("The bird flies high", "The fish flies high"),
    ("The tree grows tall", "The river grows tall"),
    ("She sees the mountain", "She sees the river"),
    ("He takes the book", "He takes the key"),
    ("The star shines bright", "The cloud shines bright"),
    ("The door opens wide", "The phone opens wide"),
    ("I think about it", "I write about it"),
    ("The bridge stands firm", "The house stands firm"),
    ("The child comes home", "The woman comes home"),
]


# ============================================================
# M1: Last-Token纤维分析
# ============================================================

def m1_last_token_fiber(model_name):
    """
    M1: 用last-token隐藏状态重新做欧氏距离分析
    
    核心假设: 
    - Mean-pooling混合了所有token，稀释了时态信息
    - Last-token是decoder-only模型的标准表示，更集中
    - 预测: last-token的时态base_ratio会更低(更纤维化)
    """
    print("\n" + "="*70)
    print("M1: Last-Token纤维分析")
    print("="*70)
    
    model, tokenizer, device, d_model, n_layers = _load_model(model_name)
    if model is None:
        return {}
    
    target_layer = n_layers // 2
    
    # === 提取方向 ===
    print("\n  [M1.1] 提取功能方向 (last-token)")
    d_tense, tense_cos = _extract_direction_last_token(
        model, tokenizer, device, TENSE_PAIRS[:15], target_layer, 'tense')
    d_polarity, pol_cos = _extract_direction_last_token(
        model, tokenizer, device, POLARITY_PAIRS[:15], target_layer, 'polarity')
    
    if d_tense is None:
        print("  ERROR: 无法提取时态方向")
        _release_model(model)
        return {}
    
    results = {}
    
    # === M1.2: 时态纤维 (last-token) ===
    print("\n  [M1.2] 时态纤维 (last-token)")
    tense_results = _compute_euclidean_fiber(
        model, tokenizer, device, TENSE_PAIRS, target_layer, 
        d_tense, d_polarity, 'tense', use_last_token=True)
    results['tense'] = tense_results
    
    # === M1.3: 极性纤维 (last-token) ===
    print("\n  [M1.3] 极性纤维 (last-token)")
    polarity_results = _compute_euclidean_fiber(
        model, tokenizer, device, POLARITY_PAIRS, target_layer,
        d_tense, d_polarity, 'polarity', use_last_token=True)
    results['polarity'] = polarity_results
    
    # === M1.4: 语义替换 (last-token) ===
    print("\n  [M1.4] 语义替换 (last-token)")
    semantic_results = _compute_semantic_euclidean(
        model, tokenizer, device, SEMANTIC_PAIRS, target_layer,
        d_tense, d_polarity, use_last_token=True)
    results['semantic'] = semantic_results
    
    # === 对比 ===
    print("\n  [M1.5] Last-Token vs Mean-Pooling对比")
    _print_comparison(results)
    
    _release_model(model)
    return results


# ============================================================
# M2: Verb-Position纤维分析
# ============================================================

def m2_verb_position_fiber(model_name):
    """
    M2: 用动词位置的token隐藏状态做纤维分析
    
    核心假设:
    - 时态信息最集中在动词token上
    - "walk"和"walked"的差异主要体现在动词token位置
    - 预测: verb-position的时态base_ratio最低(最强纤维)
    
    方法:
    - Tokenize句子
    - 找到动词token的位置(通过匹配动词字符串)
    - 提取该位置的隐藏状态
    """
    print("\n" + "="*70)
    print("M2: Verb-Position纤维分析")
    print("="*70)
    
    model, tokenizer, device, d_model, n_layers = _load_model(model_name)
    if model is None:
        return {}
    
    target_layer = n_layers // 2
    
    # === M2.1: 找到动词token位置 ===
    print("\n  [M2.1] 定位动词token")
    
    verb_positions = {}
    for present, past, verb_p, verb_pa in TENSE_PAIRS:
        pos_p = _find_verb_position(tokenizer, present, verb_p)
        pos_pa = _find_verb_position(tokenizer, past, verb_pa)
        verb_positions[present] = pos_p
        verb_positions[past] = pos_pa
        
        if len(verb_positions) <= 5:
            toks_p = [t.encode('ascii', 'replace').decode() for t in tokenizer.convert_ids_to_tokens(tokenizer.encode(present))]
            toks_pa = [t.encode('ascii', 'replace').decode() for t in tokenizer.convert_ids_to_tokens(tokenizer.encode(past))]
            print(f"    {present}: verb_pos={pos_p}, tokens={toks_p}")
            print(f"    {past}: verb_pos={pos_pa}, tokens={toks_pa}")
    
    # === M2.2: 提取动词位置的隐藏状态 ===
    print("\n  [M2.2] 提取动词位置隐藏状态")
    
    tense_h_verb = []  # (h_present_verb, h_past_verb) pairs
    for present, past, verb_p, verb_pa in TENSE_PAIRS:
        pos_p = verb_positions.get(present, -1)
        pos_pa = verb_positions.get(past, -1)
        
        if pos_p < 0 or pos_pa < 0:
            continue
        
        h_p = _get_position_hidden(model, tokenizer, device, present, target_layer, pos_p)
        h_pa = _get_position_hidden(model, tokenizer, device, past, target_layer, pos_pa)
        
        if h_p is not None and h_pa is not None:
            tense_h_verb.append((h_p, h_pa, present))
    
    print(f"    成功提取 {len(tense_h_verb)} 对动词位置隐藏状态")
    
    if not tense_h_verb:
        print("  ERROR: 无法提取动词位置隐藏状态")
        _release_model(model)
        return {}
    
    # === M2.3: 提取动词位置的方向 ===
    print("\n  [M2.3] 提取动词位置的时态方向")
    
    diffs = []
    for h_p, h_pa, _ in tense_h_verb:
        diff = h_pa - h_p
        norm = np.linalg.norm(diff)
        if norm > 1e-8:
            diffs.append(diff / norm)
    
    if diffs:
        d_tense_verb = np.mean(diffs, axis=0)
        d_tense_verb = d_tense_verb / (np.linalg.norm(d_tense_verb) + 1e-30)
        
        # 内部一致性
        cosines = [float(np.dot(d_tense_verb, d)) for d in diffs]
        avg_cos = float(np.mean(cosines))
        print(f"    动词位置时态方向: {len(diffs)}样本, 内部一致性cos={avg_cos:.4f}")
    else:
        print("  ERROR: 无法提取方向")
        _release_model(model)
        return {}
    
    # === M2.4: 欧氏距离纤维分析 ===
    print("\n  [M2.4] 动词位置的欧氏距离纤维分析")
    
    results = {
        'direction_consistency': float(avg_cos),
        'per_pair': [],
    }
    
    for h_p, h_pa, sent in tense_h_verb:
        orig_dist = float(np.linalg.norm(h_pa - h_p))
        
        proj_p = np.dot(h_p, d_tense_verb)
        proj_pa = np.dot(h_pa, d_tense_verb)
        h_p_base = h_p - proj_p * d_tense_verb
        h_pa_base = h_pa - proj_pa * d_tense_verb
        base_dist = float(np.linalg.norm(h_p_base - h_pa_base))
        
        base_ratio = base_dist / max(orig_dist, 1e-10)
        tense_energy = 1.0 - base_ratio
        
        results['per_pair'].append({
            'sentence': sent,
            'orig_dist': orig_dist,
            'base_dist': base_dist,
            'base_ratio': base_ratio,
            'tense_energy': tense_energy,
        })
        
        if len(results['per_pair']) <= 5:
            print(f"    {sent}: base_ratio={base_ratio:.6f}, tense_energy={tense_energy:.4f}")
    
    avg_base_ratio = np.mean([x['base_ratio'] for x in results['per_pair']])
    avg_tense_energy = np.mean([x['tense_energy'] for x in results['per_pair']])
    
    results['avg_base_ratio'] = float(avg_base_ratio)
    results['avg_tense_energy'] = float(avg_tense_energy)
    
    print(f"\n    动词位置: avg_base_ratio={avg_base_ratio:.6f}, "
          f"tense_energy={avg_tense_energy:.4f}")
    
    # === M2.5: 对比mean-pooling ===
    print("\n  [M2.5] 动词位置 vs Mean-Pooling对比")
    
    # Mean-pooling基线
    mp_results = _compute_euclidean_fiber(
        model, tokenizer, device, TENSE_PAIRS, target_layer,
        d_tense_verb, None, 'tense', use_last_token=False)
    
    results['mean_pooling_comparison'] = {
        'verb_position_base_ratio': float(avg_base_ratio),
        'mean_pooling_base_ratio': mp_results.get('avg_base_ratio', 'N/A'),
        'improvement': 'N/A',
    }
    
    if isinstance(mp_results.get('avg_base_ratio'), float):
        mp_br = mp_results['avg_base_ratio']
        improvement = (mp_br - avg_base_ratio) / max(mp_br, 1e-10) * 100
        results['mean_pooling_comparison']['improvement'] = float(improvement)
        print(f"    Mean-pooling base_ratio: {mp_br:.6f}")
        print(f"    Verb-position base_ratio: {avg_base_ratio:.6f}")
        print(f"    纤维性提升: {improvement:.1f}%")
    
    # === M2.6: 语义控制 ===
    print("\n  [M2.6] 语义控制 (名词位置)")
    
    # 提取名词位置（"The"后的第1个非特殊token）的隐藏状态
    noun_h_pairs = []
    for s1, s2 in SEMANTIC_PAIRS[:8]:
        h1 = _get_position_hidden(model, tokenizer, device, s1, target_layer, 1)  # 第1个token≈名词
        h2 = _get_position_hidden(model, tokenizer, device, s2, target_layer, 1)
        if h1 is not None and h2 is not None:
            orig_dist = float(np.linalg.norm(h1 - h2))
            proj1 = np.dot(h1, d_tense_verb)
            proj2 = np.dot(h2, d_tense_verb)
            h1_base = h1 - proj1 * d_tense_verb
            h2_base = h2 - proj2 * d_tense_verb
            base_dist = float(np.linalg.norm(h1_base - h2_base))
            base_ratio = base_dist / max(orig_dist, 1e-10)
            noun_h_pairs.append(base_ratio)
    
    if noun_h_pairs:
        semantic_base_ratio = float(np.mean(noun_h_pairs))
        results['semantic_control'] = {
            'semantic_base_ratio_at_noun_pos': semantic_base_ratio,
            'tense_base_ratio_at_verb_pos': float(avg_base_ratio),
            'selectivity': float(semantic_base_ratio / max(avg_base_ratio, 1e-10)),
        }
        print(f"    名词位置base_ratio: {semantic_base_ratio:.6f}")
        print(f"    动词位置base_ratio: {avg_base_ratio:.6f}")
        print(f"    选择性(名词/动词): {semantic_base_ratio/max(avg_base_ratio, 1e-10):.1f}x")
    
    _release_model(model)
    return results


# ============================================================
# M3: 跨层演化分析
# ============================================================

def m3_cross_layer_evolution(model_name):
    """
    M3: 时态方向的纤维性在不同层如何变化?
    
    核心问题:
    - 浅层: 时态信息尚未被提取，纤维性低
    - 中间层: 时态方向被"写"到特定方向上，纤维性高
    - 深层: 时态信息被转移到输出空间，纤维性可能降低
    
    预测:
    - 纤维性(1-base_ratio)应该先升后降
    - 峰值在中间偏后层
    """
    print("\n" + "="*70)
    print("M3: 跨层演化分析")
    print("="*70)
    
    model, tokenizer, device, d_model, n_layers = _load_model(model_name)
    if model is None:
        return {}
    
    # 采样层
    if n_layers <= 12:
        sample_layers = list(range(n_layers))
    else:
        sample_layers = list(range(0, n_layers, max(1, n_layers // 16)))
        if n_layers - 1 not in sample_layers:
            sample_layers.append(n_layers - 1)
    
    print(f"  采样层: {sample_layers}")
    
    results = {
        'layers': sample_layers,
        'tense_last_token': [],
        'tense_mean_pool': [],
        'polarity_last_token': [],
        'direction_consistency': [],
    }
    
    for layer_idx in sample_layers:
        print(f"\n  --- Layer {layer_idx} ---")
        
        # === 时态 (last-token) ===
        d_tense_lt, tense_cos_lt = _extract_direction_last_token(
            model, tokenizer, device, TENSE_PAIRS[:12], layer_idx, 'tense')
        
        if d_tense_lt is not None:
            tense_lt = _compute_euclidean_fiber(
                model, tokenizer, device, TENSE_PAIRS, layer_idx,
                d_tense_lt, None, 'tense', use_last_token=True)
            results['tense_last_token'].append({
                'layer': layer_idx,
                'base_ratio': tense_lt.get('avg_base_ratio', 1.0),
                'tense_energy': tense_lt.get('avg_tense_energy', 0.0),
                'direction_consistency': float(tense_cos_lt) if tense_cos_lt else 0.0,
            })
            
            # === 时态 (mean-pool) ===
            tense_mp = _compute_euclidean_fiber(
                model, tokenizer, device, TENSE_PAIRS, layer_idx,
                d_tense_lt, None, 'tense', use_last_token=False)
            results['tense_mean_pool'].append({
                'layer': layer_idx,
                'base_ratio': tense_mp.get('avg_base_ratio', 1.0),
            })
        else:
            results['tense_last_token'].append({
                'layer': layer_idx,
                'base_ratio': 1.0,
                'tense_energy': 0.0,
                'direction_consistency': 0.0,
            })
            results['tense_mean_pool'].append({
                'layer': layer_idx,
                'base_ratio': 1.0,
            })
        
        # === 极性 (last-token) ===
        d_pol_lt, pol_cos_lt = _extract_direction_last_token(
            model, tokenizer, device, POLARITY_PAIRS[:10], layer_idx, 'polarity')
        
        if d_pol_lt is not None:
            pol_lt = _compute_euclidean_fiber(
                model, tokenizer, device, POLARITY_PAIRS, layer_idx,
                d_tense_lt if d_tense_lt is not None else None, d_pol_lt, 'polarity', use_last_token=True)
            results['polarity_last_token'].append({
                'layer': layer_idx,
                'base_ratio': pol_lt.get('avg_base_ratio', 1.0),
            })
        else:
            results['polarity_last_token'].append({
                'layer': layer_idx,
                'base_ratio': 1.0,
            })
        
        # 方向一致性
        results['direction_consistency'].append({
            'layer': layer_idx,
            'tense_cos': float(tense_cos_lt) if tense_cos_lt else 0.0,
            'polarity_cos': float(pol_cos_lt) if pol_cos_lt else 0.0,
        })
    
    # === 汇总 ===
    print("\n  [M3.4] 跨层演化汇总")
    
    print("\n    Layer | Tense(LT) | Tense(MP) | Polarity(LT) | Tense_Cos | Pol_Cos")
    print("    " + "-"*65)
    for i, layer_idx in enumerate(sample_layers):
        t_lt = results['tense_last_token'][i]['base_ratio'] if i < len(results['tense_last_token']) else 1.0
        t_mp = results['tense_mean_pool'][i]['base_ratio'] if i < len(results['tense_mean_pool']) else 1.0
        p_lt = results['polarity_last_token'][i]['base_ratio'] if i < len(results['polarity_last_token']) else 1.0
        t_cos = results['direction_consistency'][i]['tense_cos'] if i < len(results['direction_consistency']) else 0.0
        p_cos = results['direction_consistency'][i]['polarity_cos'] if i < len(results['direction_consistency']) else 0.0
        
        print(f"    L{layer_idx:3d}  | {t_lt:.4f}    | {t_mp:.4f}    | {p_lt:.4f}       | {t_cos:.4f}   | {p_cos:.4f}")
    
    # 找到纤维性峰值
    tense_energies_lt = [1 - x['base_ratio'] for x in results['tense_last_token']]
    if tense_energies_lt:
        peak_idx = np.argmax(tense_energies_lt)
        peak_layer = sample_layers[peak_idx]
        peak_energy = tense_energies_lt[peak_idx]
        print(f"\n    时态纤维性峰值: Layer {peak_layer}, energy={peak_energy:.4f}")
        results['tense_peak_layer'] = peak_layer
        results['tense_peak_energy'] = float(peak_energy)
    
    _release_model(model)
    return results


# ============================================================
# 核心辅助函数
# ============================================================

def _load_model(model_name):
    """加载模型"""
    if model_name in ["glm4", "deepseek7b"]:
        model, tokenizer, device = _load_model_4bit(model_name)
    else:
        from model_utils import load_model
        model, tokenizer, device = load_model(model_name)
    
    if model is None:
        return None, None, None, 0, 0
    
    d_model = model.config.hidden_size
    n_layers = len(_get_layers(model))
    return model, tokenizer, device, d_model, n_layers


def _load_model_4bit(model_name):
    """加载4-bit量化模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    MODEL_PATHS = {
        "glm4": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "deepseek7b": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    }
    
    path = MODEL_PATHS.get(model_name)
    if not path:
        return None, None, None
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        path, quantization_config=bnb_config, device_map="auto",
        trust_remote_code=True, local_files_only=True,
    )
    model.eval()
    device = next(model.parameters()).device
    return model, tokenizer, device


def _get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(f"Cannot find layers in {type(model).__name__}")


def _release_model(model):
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("[release] GPU memory released")


def _find_verb_position(tokenizer, sentence, verb):
    """
    找到动词token在句子中的位置
    
    返回: token位置索引 (0-based, 包括BOS token)
    """
    input_ids = tokenizer.encode(sentence, add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # 搜索包含动词字符串的token
    verb_lower = verb.lower()
    for i, tok in enumerate(tokens):
        tok_clean = tok.lower().replace('▁', '').replace('Ġ', '').strip()
        if tok_clean == verb_lower or tok_clean.startswith(verb_lower):
            return i
    
    # 如果没找到精确匹配，尝试子串匹配
    for i, tok in enumerate(tokens):
        if verb_lower in tok.lower().replace('▁', '').replace('Ġ', ''):
            return i
    
    # 如果还是没找到，返回-1
    return -1


def _get_position_hidden(model, tokenizer, device, text, layer_idx, token_pos):
    """获取指定token位置的隐藏状态"""
    try:
        toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            h = model(**toks, output_hidden_states=True).hidden_states[layer_idx]
        
        seq_len = h.shape[1]
        if token_pos < 0 or token_pos >= seq_len:
            return None
        
        return h[0, token_pos].detach().cpu().float().numpy()
    except:
        return None


def _get_last_token_hidden(model, tokenizer, device, text, layer_idx):
    """获取last-token的隐藏状态"""
    try:
        toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            h = model(**toks, output_hidden_states=True).hidden_states[layer_idx]
        return h[0, -1].detach().cpu().float().numpy()
    except:
        return None


def _get_mean_pool_hidden(model, tokenizer, device, text, layer_idx):
    """获取mean-pooled隐藏状态"""
    try:
        toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            h = model(**toks, output_hidden_states=True).hidden_states[layer_idx]
        return h[0].mean(0).detach().cpu().float().numpy()
    except:
        return None


def _extract_direction_last_token(model, tokenizer, device, pairs, layer_idx, direction_type):
    """
    从句对中提取last-token功能方向
    
    Returns:
        (direction, avg_consistency_cos)
    """
    diffs = []
    
    for pair in pairs:
        if direction_type == 'tense':
            s1, s2 = pair[0], pair[1]
        else:
            s1, s2 = pair[0], pair[1]
        
        h1 = _get_last_token_hidden(model, tokenizer, device, s1, layer_idx)
        h2 = _get_last_token_hidden(model, tokenizer, device, s2, layer_idx)
        
        if h1 is None or h2 is None:
            continue
        
        diff = h2 - h1
        norm = np.linalg.norm(diff)
        if norm > 1e-8:
            diffs.append(diff / norm)
    
    if not diffs:
        return None, None
    
    d = np.mean(diffs, axis=0)
    d = d / (np.linalg.norm(d) + 1e-30)
    
    cosines = [float(np.dot(d, dd)) for dd in diffs]
    avg_cos = float(np.mean(cosines))
    
    print(f"    {direction_type}方向(last-token): {len(diffs)}样本, cos={avg_cos:.4f}")
    return d, avg_cos


def _compute_euclidean_fiber(model, tokenizer, device, pairs, layer_idx,
                              d_tense, d_polarity, fiber_type, use_last_token=True):
    """
    计算欧氏距离纤维性
    
    Args:
        fiber_type: 'tense' or 'polarity'
        use_last_token: True=last-token, False=mean-pool
    """
    if use_last_token:
        get_fn = lambda t: _get_last_token_hidden(model, tokenizer, device, t, layer_idx)
    else:
        get_fn = lambda t: _get_mean_pool_hidden(model, tokenizer, device, t, layer_idx)
    
    results = []
    
    for pair in pairs:
        if fiber_type == 'tense':
            s1, s2 = pair[0], pair[1]
        else:
            s1, s2 = pair[0], pair[1]
        
        h1 = get_fn(s1)
        h2 = get_fn(s2)
        
        if h1 is None or h2 is None:
            continue
        
        orig_dist = float(np.linalg.norm(h1 - h2))
        
        # 去掉功能方向分量
        d_func = d_tense if fiber_type == 'tense' else d_polarity
        if d_func is not None:
            proj1 = np.dot(h1, d_func)
            proj2 = np.dot(h2, d_func)
            h1_base = h1 - proj1 * d_func
            h2_base = h2 - proj2 * d_func
            
            # 如果有另一个方向，也去掉
            d_other = d_polarity if fiber_type == 'tense' else d_tense
            if d_other is not None:
                h1_base = h1_base - np.dot(h1_base, d_other) * d_other
                h2_base = h2_base - np.dot(h2_base, d_other) * d_other
            
            base_dist = float(np.linalg.norm(h1_base - h2_base))
        else:
            base_dist = orig_dist
        
        base_ratio = base_dist / max(orig_dist, 1e-10)
        func_energy = 1.0 - base_ratio
        
        results.append({
            'orig_dist': orig_dist,
            'base_dist': base_dist,
            'base_ratio': base_ratio,
            'func_energy': func_energy,
        })
    
    if not results:
        return {}
    
    avg_base_ratio = np.mean([x['base_ratio'] for x in results])
    avg_func_energy = np.mean([x['func_energy'] for x in results])
    
    print(f"    {fiber_type}({'LT' if use_last_token else 'MP'}): "
          f"base_ratio={avg_base_ratio:.6f}, energy={avg_func_energy:.4f}, "
          f"n={len(results)}")
    
    return {
        'avg_base_ratio': float(avg_base_ratio),
        'avg_func_energy': float(avg_func_energy),
        'n_samples': len(results),
        'samples': results[:10],
    }


def _compute_semantic_euclidean(model, tokenizer, device, pairs, layer_idx,
                                 d_tense, d_polarity, use_last_token=True):
    """计算语义替换的欧氏距离(对比基准)"""
    if use_last_token:
        get_fn = lambda t: _get_last_token_hidden(model, tokenizer, device, t, layer_idx)
    else:
        get_fn = lambda t: _get_mean_pool_hidden(model, tokenizer, device, t, layer_idx)
    
    results = []
    
    for s1, s2 in pairs:
        h1 = get_fn(s1)
        h2 = get_fn(s2)
        
        if h1 is None or h2 is None:
            continue
        
        orig_dist = float(np.linalg.norm(h1 - h2))
        
        # 去掉功能方向分量
        h1_base = h1.copy()
        h2_base = h2.copy()
        if d_tense is not None:
            h1_base = h1_base - np.dot(h1_base, d_tense) * d_tense
            h2_base = h2_base - np.dot(h2_base, d_tense) * d_tense
        if d_polarity is not None:
            h1_base = h1_base - np.dot(h1_base, d_polarity) * d_polarity
            h2_base = h2_base - np.dot(h2_base, d_polarity) * d_polarity
        
        base_dist = float(np.linalg.norm(h1_base - h2_base))
        base_ratio = base_dist / max(orig_dist, 1e-10)
        
        results.append({
            'pair': f"{s1[:20]}-{s2[:20]}",
            'base_ratio': base_ratio,
        })
    
    if not results:
        return {}
    
    avg_base_ratio = np.mean([x['base_ratio'] for x in results])
    
    print(f"    语义({'LT' if use_last_token else 'MP'}): "
          f"base_ratio={avg_base_ratio:.6f}, n={len(results)}")
    
    return {
        'avg_base_ratio': float(avg_base_ratio),
        'n_samples': len(results),
    }


def _print_comparison(results):
    """打印对比表格"""
    print("\n    " + "="*60)
    print("    纤维性对比 (1 - base_ratio = 功能方向解释的方差比例)")
    print("    " + "="*60)
    
    print(f"    {'维度':<15} {'base_ratio':<12} {'func_energy':<12} {'纤维判定'}")
    print("    " + "-"*55)
    
    for dim in ['tense', 'polarity', 'semantic']:
        if dim in results:
            br = results[dim].get('avg_base_ratio', 1.0)
            fe = results[dim].get('avg_func_energy', 0.0)
            
            if fe > 0.7:
                verdict = "强纤维"
            elif fe > 0.3:
                verdict = "中等纤维"
            elif fe > 0.1:
                verdict = "弱纤维"
            else:
                verdict = "无纤维"
            
            print(f"    {dim:<15} {br:.6f}     {fe:.4f}       {verdict}")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Position-Sensitive Fiber Analysis")
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--test", type=str, default="all",
                        choices=["m1", "m2", "m3", "all"],
                        help="运行哪个测试")
    args = parser.parse_args()
    
    model_name = args.model
    print(f"\n{'#'*70}")
    print(f"# Phase CLXXXIII: 位置敏感纤维分析")
    print(f"# Model: {model_name}")
    print(f"{'#'*70}")
    
    all_results = {
        'model': model_name,
        'timestamp': datetime.now().isoformat(),
    }
    
    output_dir = Path(r"d:\Ai2050\TransformerLens-Project\results\position_fiber")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # M1: Last-token纤维
    if args.test in ["m1", "all"]:
        t0 = time.time()
        m1_results = m1_last_token_fiber(model_name)
        all_results['m1_last_token'] = m1_results
        print(f"\n  M1耗时: {time.time()-t0:.1f}s")
    
    # M2: Verb-position纤维
    if args.test in ["m2", "all"]:
        t0 = time.time()
        m2_results = m2_verb_position_fiber(model_name)
        all_results['m2_verb_position'] = m2_results
        print(f"\n  M2耗时: {time.time()-t0:.1f}s")
    
    # M3: 跨层演化
    if args.test in ["m3", "all"]:
        t0 = time.time()
        m3_results = m3_cross_layer_evolution(model_name)
        all_results['m3_cross_layer'] = m3_results
        print(f"\n  M3耗时: {time.time()-t0:.1f}s")
    
    # 保存结果
    output_path = output_dir / f"{model_name}_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"所有结果已保存到 {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
