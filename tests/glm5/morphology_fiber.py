"""
Phase CLXXXIV: 形态-纤维关联验证 — 规则vs不规则动词分离 + 极性因果干预
=====================================================================

核心假说: "纤维性取决于形态规则性"
  - 规则动词(walk→walked): 形态变化可预测 → 更纤维化
  - 不规则动词(run→ran): 形态变化不可预测 → 更弱纤维化
  - 否定(not/不): 独立助词 → 最强纤维

测试:
N1: 规则vs不规则动词的纤维性(base_ratio)对比
N2: 极性方向的因果干预 — 沿d_polarity干预, logit lens验证"not"的logit变化
N3: 方向一致性分析 — 规则/不规则/否定方向的内部一致性cos

运行方式:
  python tests/glm5/morphology_fiber.py --model qwen3 --test n1
  python tests/glm5/morphology_fiber.py --model qwen3 --test n2
  python tests/glm5/morphology_fiber.py --model qwen3 --test n3
  python tests/glm5/morphology_fiber.py --model qwen3 --test all
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


# ============================================================
# 测试集: 规则动词 vs 不规则动词
# ============================================================

# 规则动词: +ed, 有明确形态规则
REGULAR_VERB_PAIRS = [
    # (present_sentence, past_sentence, verb_present, verb_past)
    ("The cat walks every day", "The cat walked yesterday", "walk", "walked"),
    ("The cloud moves slowly", "The cloud moved slowly", "move", "moved"),
    ("The door opens wide", "The door opened wide", "open", "opened"),
    ("The flower blooms late", "The flower bloomed late", "bloom", "bloomed"),
    ("The river flows east", "The river flowed east", "flow", "flowed"),
    ("She plays the piano", "She played the piano", "play", "played"),
    ("He jumps high", "He jumped high", "jump", "jumped"),
    ("The car stops here", "The car stopped here", "stop", "stopped"),
    ("They work hard", "They worked hard", "work", "worked"),
    ("The rain drops fall", "The rain drops fell", "drop", "dropped"),
    ("She smiles at me", "She smiled at me", "smile", "smiled"),
    ("The dog barks loud", "The dog barked loud", "bark", "barked"),
    ("He types fast", "He typed fast", "type", "typed"),
    ("The candle burns bright", "The candle burned bright", "burn", "burned"),
    ("The wind blows cold", "The wind blew cold", "blow", "blew"),
    ("She cooks dinner", "She cooked dinner", "cook", "cooked"),
    ("The bell rings loud", "The bell rang loud", "ring", "rang"),
    ("He paints walls", "He painted walls", "paint", "painted"),
    ("The ship sails far", "The ship sailed far", "sail", "sailed"),
    ("She cleans the house", "She cleaned the house", "clean", "cleaned"),
    # 更多规则动词
    ("The clock ticks", "The clock ticked", "tick", "ticked"),
    ("The fish swims deep", "The fish swam deep", "swim", "swam"),
    ("The phone rings", "The phone rang", "ring", "rang"),
    ("He walks to school", "He walked to school", "walk", "walked"),
    ("The snow falls down", "The snow fell down", "fall", "fell"),
]

# 不规则动词: 元音变化/完全不同的词
IRREGULAR_VERB_PAIRS = [
    # (present_sentence, past_sentence, verb_present, verb_past)
    ("The dog runs fast", "The dog ran fast", "run", "ran"),
    ("The bird flies high", "The bird flew high", "fly", "flew"),
    ("The tree grows tall", "The tree grew tall", "grow", "grew"),
    ("She sees the mountain", "She saw the mountain", "see", "saw"),
    ("He takes the book", "He took the book", "take", "took"),
    ("The child comes home", "The child came home", "come", "came"),
    ("The woman drives slowly", "The woman drove slowly", "drive", "drove"),
    ("The star shines bright", "The star shone bright", "shine", "shone"),
    ("I think about it", "I thought about it", "think", "thought"),
    ("She writes a letter", "She wrote a letter", "write", "wrote"),
    ("The bridge stands firm", "The bridge stood firm", "stand", "stood"),
    ("The table holds weight", "The table held weight", "hold", "held"),
    ("The man gives gifts", "The man gave gifts", "give", "gave"),
    ("She sings a song", "She sang a song", "sing", "sang"),
    ("The king fights well", "The king fought well", "fight", "fought"),
    ("The boy knows the way", "The boy knew the way", "know", "knew"),
    ("She speaks softly", "She spoke softly", "speak", "spoke"),
    ("The wheel breaks down", "The wheel broke down", "break", "broke"),
    ("He chooses wisely", "He chose wisely", "choose", "chose"),
    ("The water freezes", "The water froze", "freeze", "froze"),
    # 更多不规则动词
    ("She rides the horse", "She rode the horse", "ride", "rode"),
    ("The bell rings at noon", "The bell rang at noon", "ring", "rang"),
    ("He draws a picture", "He drew a picture", "draw", "drew"),
    ("The candle burns out", "The candle burned out", "burn", "burned"),
    ("She wears a dress", "She wore a dress", "wear", "wore"),
]

# 极性测试集(用于N2因果干预)
POLARITY_PAIRS = [
    # (affirmative, negative, concept)
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

# 语义控制测试集(名词替换)
SEMANTIC_PAIRS = [
    # (sentence1, sentence2, word1, word2) — 只有名词不同
    ("The cat sits here", "The dog sits here", "cat", "dog"),
    ("The king rules well", "The queen rules well", "king", "queen"),
    ("The man walks slow", "The woman walks slow", "man", "woman"),
    ("The boy runs fast", "The girl runs fast", "boy", "girl"),
    ("The apple is red", "The orange is red", "apple", "orange"),
    ("The car drives fast", "The truck drives fast", "car", "truck"),
    ("The cat is sleeping", "The dog is sleeping", "cat", "dog"),
    ("The house is large", "The building is large", "house", "building"),
    ("He reads the book", "He reads the paper", "book", "paper"),
    ("She eats the apple", "She eats the banana", "apple", "banana"),
    ("The river is deep", "The lake is deep", "river", "lake"),
    ("The star shines bright", "The moon shines bright", "star", "moon"),
    ("The cat chased mice", "The dog chased mice", "cat", "dog"),
    ("The tree fell down", "The rock fell down", "tree", "rock"),
    ("The bird sang loud", "The child sang loud", "bird", "child"),
]


# ============================================================
# 辅助函数
# ============================================================

def _load_model_full(model_name):
    """加载模型并返回完整信息"""
    if model_name in ["glm4", "deepseek7b"]:
        model, tokenizer, device = _load_model_4bit(model_name)
    else:
        from model_utils import load_model
        model, tokenizer, device = load_model(model_name)
    
    if model is None:
        return None, None, None, 0, 0, None
    
    d_model = model.config.hidden_size
    n_layers = len(_get_layers(model))
    
    W_U = None
    try:
        if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
            W_U = model.lm_head.weight.detach().cpu().float().numpy()
    except:
        pass
    
    return model, tokenizer, device, d_model, n_layers, W_U


def _load_model_4bit(model_name):
    """加载4-bit量化模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    MODEL_PATHS = {
        "glm4": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "deepseek7b": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    }
    
    path = MODEL_PATHS.get(model_name)
    if not path:
        print(f"  ERROR: Unknown model {model_name}")
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


def _get_hidden_states(model, tokenizer, device, s1, s2, layer_idx):
    """获取两个句子的mean-pooled隐藏状态"""
    try:
        toks1 = tokenizer(s1, return_tensors="pt", truncation=True, max_length=64).to(device)
        toks2 = tokenizer(s2, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            h1 = model(**toks1, output_hidden_states=True).hidden_states[layer_idx]
            h2 = model(**toks2, output_hidden_states=True).hidden_states[layer_idx]
        r1 = h1[0].mean(0).detach().cpu().float().numpy()
        r2 = h2[0].mean(0).detach().cpu().float().numpy()
        return r1, r2
    except:
        return None, None


def _get_single_hidden(model, tokenizer, device, text, layer_idx):
    """获取单个句子的mean-pooled隐藏状态"""
    try:
        toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            h = model(**toks, output_hidden_states=True).hidden_states[layer_idx]
        return h[0].mean(0).detach().cpu().float().numpy()
    except:
        return None


def _extract_functional_direction(model, tokenizer, device, pairs, layer_idx, direction_type='tense'):
    """
    从句对中提取功能方向
    
    Args:
        pairs: 句对列表
            tense: [(present, past, verb_pres, verb_past), ...]
            polarity: [(aff, neg, concept), ...]
        direction_type: 'tense' or 'polarity'
    
    Returns:
        d_direction: 归一化的方向向量 [d_model]
        cos_mean: 内部一致性
    """
    diffs = []
    
    for pair in pairs:
        if direction_type in ('tense', 'regular', 'irregular'):
            s1, s2 = pair[0], pair[1]
        else:  # polarity or semantic
            s1, s2 = pair[0], pair[1]
        
        h1, h2 = _get_hidden_states(model, tokenizer, device, s1, s2, layer_idx)
        if h1 is None or h2 is None:
            continue
        
        diff = h2 - h1
        norm = np.linalg.norm(diff)
        if norm > 1e-8:
            diffs.append(diff / norm)
    
    if not diffs:
        return None, 0.0
    
    d = np.mean(diffs, axis=0)
    d = d / (np.linalg.norm(d) + 1e-30)
    
    # 计算内部一致性
    cos_mean = 0.0
    if len(diffs) > 1:
        cosines = [float(np.dot(d, dd)) for dd in diffs]
        cos_mean = float(np.mean(cosines))
    
    return d, cos_mean


def _compute_base_ratio_euclidean(model, tokenizer, device, pairs, direction, layer_idx, 
                                   direction_type='tense'):
    """
    计算欧氏距离base_ratio
    
    base_ratio = ||h2 - h1 - proj_{d}(h2 - h1)|| / ||h2 - h1||
    base_ratio 越小 → 方向解释越多变化 → 越纤维化
    
    Returns:
        dict: {mean_base_ratio, per_sample_ratios, cos_base, cos_orig}
    """
    ratios = []
    cos_bases = []
    cos_origs = []
    
    for pair in pairs:
        s1, s2 = pair[0], pair[1]
        
        h1, h2 = _get_hidden_states(model, tokenizer, device, s1, s2, layer_idx)
        if h1 is None or h2 is None:
            continue
        
        delta = h2 - h1
        orig_dist = np.linalg.norm(delta)
        if orig_dist < 1e-8:
            continue
        
        # 沿direction投影
        proj_len = np.dot(delta, direction)
        proj_vec = proj_len * direction
        base_vec = delta - proj_vec
        base_dist = np.linalg.norm(base_vec)
        
        base_ratio = base_dist / orig_dist
        ratios.append(float(base_ratio))
        
        # 余弦相似度(对比)
        cos_orig = float(np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-30))
        h1_base = h1 + proj_vec  # 加上投影
        cos_base = float(np.dot(h1_base, h2) / (np.linalg.norm(h1_base) * np.linalg.norm(h2) + 1e-30))
        cos_bases.append(float(cos_base))
        cos_origs.append(float(cos_orig))
    
    if not ratios:
        return None
    
    return {
        'mean_base_ratio': float(np.mean(ratios)),
        'std_base_ratio': float(np.std(ratios)),
        'median_base_ratio': float(np.median(ratios)),
        'mean_cos_base': float(np.mean(cos_bases)),
        'mean_cos_orig': float(np.mean(cos_origs)),
        'per_sample_ratios': ratios,
        'n_samples': len(ratios),
        'energy_in_direction': float(1.0 - np.mean(ratios)),
    }


# ============================================================
# N1: 规则vs不规则动词的纤维性对比
# ============================================================

def n1_regular_vs_irregular(model_name):
    """
    N1: 规则动词vs不规则动词的纤维性(base_ratio)对比
    
    核心假说:
    - 规则动词: 形态变化可预测(walk→walked) → 更纤维化(lower base_ratio)
    - 不规则动词: 形态变化不可预测(run→ran) → 更弱纤维化(higher base_ratio)
    
    同时与极性(强纤维)和语义(无纤维)对比
    """
    print("\n" + "="*70)
    print("N1: 规则vs不规则动词的纤维性对比")
    print("="*70)
    
    model, tokenizer, device, d_model, n_layers, W_U = _load_model_full(model_name)
    if model is None:
        return {}
    
    target_layer = n_layers // 2
    
    results = {
        'model': model_name,
        'target_layer': target_layer,
        'd_model': d_model,
        'regular': {},
        'irregular': {},
        'all_tense': {},
        'polarity': {},
        'semantic': {},
    }
    
    # --- Step 1: 提取各类方向 ---
    print("\n  [N1.1] 提取功能方向")
    
    # 规则动词方向
    regular_pairs = REGULAR_VERB_PAIRS[:20]
    d_regular, cos_regular = _extract_functional_direction(
        model, tokenizer, device, regular_pairs, target_layer, 'regular'
    )
    print(f"    规则动词方向: cos_mean={cos_regular:.4f}, n={len(regular_pairs)}")
    
    # 不规则动词方向
    irregular_pairs = IRREGULAR_VERB_PAIRS[:20]
    d_irregular, cos_irregular = _extract_functional_direction(
        model, tokenizer, device, irregular_pairs, target_layer, 'irregular'
    )
    print(f"    不规则动词方向: cos_mean={cos_irregular:.4f}, n={len(irregular_pairs)}")
    
    # 全部时态方向
    all_tense_pairs = regular_pairs + irregular_pairs
    d_all_tense, cos_all = _extract_functional_direction(
        model, tokenizer, device, all_tense_pairs, target_layer, 'tense'
    )
    print(f"    全部时态方向: cos_mean={cos_all:.4f}, n={len(all_tense_pairs)}")
    
    # 极性方向
    d_polarity, cos_polarity = _extract_functional_direction(
        model, tokenizer, device, POLARITY_PAIRS[:15], target_layer, 'polarity'
    )
    print(f"    极性方向: cos_mean={cos_polarity:.4f}")
    
    # 语义方向
    d_semantic, cos_semantic = _extract_functional_direction(
        model, tokenizer, device, SEMANTIC_PAIRS[:10], target_layer, 'semantic'
    )
    print(f"    语义方向: cos_mean={cos_semantic:.4f}")
    
    # --- Step 2: 规则vs不规则的cross-validation ---
    print("\n  [N1.2] 规则vs不规则方向交叉验证")
    
    if d_regular is not None and d_irregular is not None:
        cross_cos = float(np.dot(d_regular, d_irregular))
        print(f"    d_regular · d_irregular = {cross_cos:.4f}")
        results['cross_cos'] = cross_cos
    
    if d_all_tense is not None:
        if d_regular is not None:
            cos_reg_all = float(np.dot(d_regular, d_all_tense))
            print(f"    d_regular · d_all_tense = {cos_reg_all:.4f}")
            results['cos_regular_all'] = cos_reg_all
        if d_irregular is not None:
            cos_irr_all = float(np.dot(d_irregular, d_all_tense))
            print(f"    d_irregular · d_all_tense = {cos_irr_all:.4f}")
            results['cos_irregular_all'] = cos_irr_all
    
    # --- Step 3: 分别计算base_ratio ---
    print("\n  [N1.3] 计算欧氏距离base_ratio")
    
    # 规则动词 — 用规则方向测量
    if d_regular is not None:
        regular_on_regular = _compute_base_ratio_euclidean(
            model, tokenizer, device, regular_pairs, d_regular, target_layer, 'regular'
        )
        if regular_on_regular:
            results['regular'] = {
                'direction_cos': cos_regular,
                'base_ratio_on_regular': regular_on_regular['mean_base_ratio'],
                'std': regular_on_regular['std_base_ratio'],
                'energy': regular_on_regular['energy_in_direction'],
                'cos_base': regular_on_regular['mean_cos_base'],
                'n': regular_on_regular['n_samples'],
            }
            print(f"    规则动词→规则方向: base_ratio={regular_on_regular['mean_base_ratio']:.4f}, "
                  f"energy={regular_on_regular['energy_in_direction']:.1%}, n={regular_on_regular['n_samples']}")
    
    # 不规则动词 — 用不规则方向测量
    if d_irregular is not None:
        irregular_on_irregular = _compute_base_ratio_euclidean(
            model, tokenizer, device, irregular_pairs, d_irregular, target_layer, 'irregular'
        )
        if irregular_on_irregular:
            results['irregular'] = {
                'direction_cos': cos_irregular,
                'base_ratio_on_irregular': irregular_on_irregular['mean_base_ratio'],
                'std': irregular_on_irregular['std_base_ratio'],
                'energy': irregular_on_irregular['energy_in_direction'],
                'cos_base': irregular_on_irregular['mean_cos_base'],
                'n': irregular_on_irregular['n_samples'],
            }
            print(f"    不规则动词→不规则方向: base_ratio={irregular_on_irregular['mean_base_ratio']:.4f}, "
                  f"energy={irregular_on_irregular['energy_in_direction']:.1%}, n={irregular_on_irregular['n_samples']}")
    
    # 规则动词 — 用不规则方向测量(cross)
    if d_irregular is not None:
        regular_on_irregular = _compute_base_ratio_euclidean(
            model, tokenizer, device, regular_pairs, d_irregular, target_layer, 'regular'
        )
        if regular_on_irregular:
            results['regular_on_irregular'] = {
                'base_ratio': regular_on_irregular['mean_base_ratio'],
                'energy': regular_on_irregular['energy_in_direction'],
            }
            print(f"    规则动词→不规则方向: base_ratio={regular_on_irregular['mean_base_ratio']:.4f}")
    
    # 不规则动词 — 用规则方向测量(cross)
    if d_regular is not None:
        irregular_on_regular = _compute_base_ratio_euclidean(
            model, tokenizer, device, irregular_pairs, d_regular, target_layer, 'irregular'
        )
        if irregular_on_regular:
            results['irregular_on_regular'] = {
                'base_ratio': irregular_on_regular['mean_base_ratio'],
                'energy': irregular_on_regular['energy_in_direction'],
            }
            print(f"    不规则动词→规则方向: base_ratio={irregular_on_regular['mean_base_ratio']:.4f}")
    
    # 全部动词 — 用全部方向测量
    if d_all_tense is not None:
        all_on_all = _compute_base_ratio_euclidean(
            model, tokenizer, device, all_tense_pairs, d_all_tense, target_layer, 'tense'
        )
        if all_on_all:
            results['all_tense'] = {
                'direction_cos': cos_all,
                'base_ratio': all_on_all['mean_base_ratio'],
                'energy': all_on_all['energy_in_direction'],
                'n': all_on_all['n_samples'],
            }
            print(f"    全部动词→全部方向: base_ratio={all_on_all['mean_base_ratio']:.4f}, "
                  f"energy={all_on_all['energy_in_direction']:.1%}")
    
    # 极性 — 用极性方向测量
    if d_polarity is not None:
        polarity_result = _compute_base_ratio_euclidean(
            model, tokenizer, device, POLARITY_PAIRS, d_polarity, target_layer, 'polarity'
        )
        if polarity_result:
            results['polarity'] = {
                'direction_cos': cos_polarity,
                'base_ratio': polarity_result['mean_base_ratio'],
                'energy': polarity_result['energy_in_direction'],
                'n': polarity_result['n_samples'],
            }
            print(f"    极性→极性方向: base_ratio={polarity_result['mean_base_ratio']:.4f}, "
                  f"energy={polarity_result['energy_in_direction']:.1%}")
    
    # 语义 — 用语义方向测量
    if d_semantic is not None:
        semantic_result = _compute_base_ratio_euclidean(
            model, tokenizer, device, SEMANTIC_PAIRS, d_semantic, target_layer, 'semantic'
        )
        if semantic_result:
            results['semantic'] = {
                'direction_cos': cos_semantic,
                'base_ratio': semantic_result['mean_base_ratio'],
                'energy': semantic_result['energy_in_direction'],
                'n': semantic_result['n_samples'],
            }
            print(f"    语义→语义方向: base_ratio={semantic_result['mean_base_ratio']:.4f}, "
                  f"energy={semantic_result['energy_in_direction']:.1%}")
    
    # --- Step 4: Per-verb analysis ---
    print("\n  [N1.4] 逐动词base_ratio分析")
    
    per_verb_results = []
    
    # 分析规则动词的每个样本
    if d_regular is not None:
        for pair in regular_pairs[:10]:
            s1, s2, vp, vpa = pair
            h1, h2 = _get_hidden_states(model, tokenizer, device, s1, s2, target_layer)
            if h1 is None or h2 is None:
                continue
            delta = h2 - h1
            orig_dist = np.linalg.norm(delta)
            if orig_dist < 1e-8:
                continue
            proj_len = np.dot(delta, d_regular)
            proj_vec = proj_len * d_regular
            base_dist = np.linalg.norm(delta - proj_vec)
            ratio = base_dist / orig_dist
            per_verb_results.append({
                'verb': f"{vp}->{vpa}",
                'type': 'regular',
                'base_ratio': float(ratio),
                'energy': float(1 - ratio),
            })
            if len(per_verb_results) <= 12:
                print(f"    {vp}->{vpa} (regular): base_ratio={ratio:.4f}, energy={1-ratio:.1%}")
    
    # 分析不规则动词的每个样本
    if d_irregular is not None:
        for pair in irregular_pairs[:10]:
            s1, s2, vp, vpa = pair
            h1, h2 = _get_hidden_states(model, tokenizer, device, s1, s2, target_layer)
            if h1 is None or h2 is None:
                continue
            delta = h2 - h1
            orig_dist = np.linalg.norm(delta)
            if orig_dist < 1e-8:
                continue
            proj_len = np.dot(delta, d_irregular)
            proj_vec = proj_len * d_irregular
            base_dist = np.linalg.norm(delta - proj_vec)
            ratio = base_dist / orig_dist
            per_verb_results.append({
                'verb': f"{vp}->{vpa}",
                'type': 'irregular',
                'base_ratio': float(ratio),
                'energy': float(1 - ratio),
            })
            if len(per_verb_results) <= 24:
                print(f"    {vp}->{vpa} (irregular): base_ratio={ratio:.4f}, energy={1-ratio:.1%}")
    
    results['per_verb'] = per_verb_results
    
    # --- Step 5: 方向一致性 ---
    print("\n  [N1.5] 方向一致性总结")
    
    consistency = {
        'regular_cos': float(cos_regular),
        'irregular_cos': float(cos_irregular),
        'all_tense_cos': float(cos_all),
        'polarity_cos': float(cos_polarity),
        'semantic_cos': float(cos_semantic),
    }
    results['direction_consistency'] = consistency
    
    print(f"    规则动词方向一致性: cos={cos_regular:.4f}")
    print(f"    不规则动词方向一致性: cos={cos_irregular:.4f}")
    print(f"    全部时态方向一致性: cos={cos_all:.4f}")
    print(f"    极性方向一致性: cos={cos_polarity:.4f}")
    print(f"    语义方向一致性: cos={cos_semantic:.4f}")
    
    _release_model(model)
    return results


# ============================================================
# N2: 极性因果干预 — Logit Lens验证
# ============================================================

def n2_polarity_causal_intervention(model_name):
    """
    N2: 沿d_polarity干预, 用logit lens验证"not"的logit因果变化
    
    核心设计:
    1. 从极性句对提取d_polarity方向
    2. 在affirmative句子的隐藏状态上沿d_polarity干预
    3. 用lm_head解码, 检查"not"的logit是否增加
    4. 同时检查语义token是否保持不变
    
    与L1(时态干预)对比: 极性方向应该是更强的因果干预
    """
    print("\n" + "="*70)
    print("N2: 极性方向因果干预 — Logit Lens验证")
    print("="*70)
    
    model, tokenizer, device, d_model, n_layers, W_U = _load_model_full(model_name)
    if model is None:
        return {}
    
    if W_U is None:
        print("  ERROR: 无法获取lm_head权重")
        _release_model(model)
        return {}
    
    target_layer = n_layers // 2
    
    # --- Step 1: 提取极性方向 ---
    print("\n  [N2.1] 提取极性方向")
    d_polarity, cos_polarity = _extract_functional_direction(
        model, tokenizer, device, POLARITY_PAIRS[:15], target_layer, 'polarity'
    )
    if d_polarity is None:
        print("  ERROR: 无法提取极性方向")
        _release_model(model)
        return {}
    print(f"    极性方向: cos_mean={cos_polarity:.4f}")
    
    # 提取时态方向(对比)
    from causal_logit_validation import ENGLISH_TENSE_PAIRS
    d_tense, cos_tense = _extract_functional_direction(
        model, tokenizer, device, ENGLISH_TENSE_PAIRS[:15], target_layer, 'tense'
    )
    if d_tense is not None:
        print(f"    时态方向: cos_mean={cos_tense:.4f}")
        cross_cos = float(np.dot(d_polarity, d_tense))
        print(f"    d_polarity · d_tense = {cross_cos:.4f}")
    
    # --- Step 2: 获取关键token ID ---
    print("\n  [N2.2] 获取关键token ID")
    
    not_ids = tokenizer.encode("not", add_special_tokens=False)
    not_tok_id = not_ids[0] if not_ids else None
    print(f"    'not' token_id={not_tok_id}, word='{tokenizer.decode([not_tok_id]) if not_tok_id else 'N/A'}'")
    
    no_ids = tokenizer.encode("no", add_special_tokens=False)
    no_tok_id = no_ids[0] if no_ids else None
    print(f"    'no' token_id={no_tok_id}")
    
    # 获取一些控制token
    the_ids = tokenizer.encode("the", add_special_tokens=False)
    the_tok_id = the_ids[0] if the_ids else None
    is_ids = tokenizer.encode("is", add_special_tokens=False)
    is_tok_id = is_ids[0] if is_ids else None
    
    # --- Step 3: 沿d_polarity干预 ---
    print("\n  [N2.3] 沿d_polarity干预 — Logit Lens解码")
    
    alphas = [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    
    results = {
        'model': model_name,
        'target_layer': target_layer,
        'd_model': d_model,
        'polarity_cos': float(cos_polarity),
        'interventions': [],
    }
    
    not_logit_slopes = []  # "not" logit随α的变化率
    semantic_logit_slopes = []  # 语义token的变化率
    
    for aff, neg, concept in POLARITY_PAIRS[:15]:
        h_aff = _get_single_hidden(model, tokenizer, device, aff, target_layer)
        h_neg = _get_single_hidden(model, tokenizer, device, neg, target_layer)
        if h_aff is None or h_neg is None:
            continue
        
        # 原始logit
        logits_aff = h_aff @ W_U.T
        logits_neg = h_neg @ W_U.T
        
        # "not"在原始句子中的logit
        not_logit_aff = float(logits_aff[not_tok_id]) if not_tok_id else 0.0
        not_logit_neg = float(logits_neg[not_tok_id]) if not_tok_id else 0.0
        
        # 概念token
        concept_ids = tokenizer.encode(concept, add_special_tokens=False)
        concept_tok_id = concept_ids[0] if concept_ids else None
        concept_logit_aff = float(logits_aff[concept_tok_id]) if concept_tok_id else 0.0
        
        intervention_data = {
            'affirmative': aff,
            'negative': neg,
            'concept': concept,
            'not_logit_in_aff': not_logit_aff,
            'not_logit_in_neg': not_logit_neg,
            'alpha_sweep': [],
        }
        
        not_logits_list = []
        concept_logits_list = []
        the_logits_list = []
        
        for alpha in alphas:
            h_intervened = h_aff + alpha * d_polarity
            logits_intervened = h_intervened @ W_U.T
            
            # 核心指标: "not"的logit变化
            not_logit = float(logits_intervened[not_tok_id]) if not_tok_id else 0.0
            not_logits_list.append(not_logit)
            
            # 控制指标: 概念token和"the"的logit
            concept_logit = float(logits_intervened[concept_tok_id]) if concept_tok_id else 0.0
            concept_logits_list.append(concept_logit)
            
            the_logit = float(logits_intervened[the_tok_id]) if the_tok_id else 0.0
            the_logits_list.append(the_logit)
            
            # Top-1 token
            top1_token = int(np.argmax(logits_intervened))
            top1_word = tokenizer.decode([top1_token]).replace('\ufeff', '').replace('\ufffd', '?').strip()[:20]
            
            intervention_data['alpha_sweep'].append({
                'alpha': float(alpha),
                'not_logit': not_logit,
                'concept_logit': concept_logit,
                'the_logit': the_logit,
                'top1': top1_word,
            })
        
        # 计算斜率(用线性回归)
        alphas_arr = np.array(alphas)
        not_logits_arr = np.array(not_logits_list)
        concept_logits_arr = np.array(concept_logits_list)
        
        # "not" logit vs alpha的斜率
        if len(alphas) > 2:
            not_slope = float(np.polyfit(alphas_arr, not_logits_arr, 1)[0])
            concept_slope = float(np.polyfit(alphas_arr, concept_logits_arr, 1)[0])
        else:
            not_slope = 0.0
            concept_slope = 0.0
        
        not_logit_slopes.append(not_slope)
        semantic_logit_slopes.append(abs(concept_slope))
        
        intervention_data['not_logit_slope'] = not_slope
        intervention_data['concept_logit_slope'] = concept_slope
        
        results['interventions'].append(intervention_data)
        
        # 打印前5个详细结果
        if len(results['interventions']) <= 5:
            print(f"\n    [{aff}]")
            print(f"      not_logit: aff={not_logit_aff:.3f}, neg={not_logit_neg:.3f}")
            print(f"      slope: not={not_slope:+.4f}, concept={concept_slope:+.4f}")
            for sweep in intervention_data['alpha_sweep']:
                if sweep['alpha'] in [-2.0, 0.0, 2.0]:
                    print(f"      α={sweep['alpha']:+5.1f}: not_logit={sweep['not_logit']:+7.3f}, "
                          f"concept={sweep['concept_logit']:+7.3f}, top1={sweep['top1']}")
    
    # --- Step 4: 汇总 ---
    print("\n  [N2.4] 汇总结果")
    
    if not_logit_slopes:
        mean_not_slope = float(np.mean(not_logit_slopes))
        mean_concept_slope = float(np.mean(semantic_logit_slopes))
        selectivity = mean_not_slope / (mean_concept_slope + 1e-10)
        
        results['summary'] = {
            'mean_not_logit_slope': mean_not_slope,
            'mean_concept_logit_slope_abs': mean_concept_slope,
            'selectivity': float(selectivity),
            'semantic_control': 'STRONG' if selectivity > 10 else 'MODERATE' if selectivity > 3 else 'WEAK',
            'n_samples': len(not_logit_slopes),
        }
        
        print(f"    'not' logit斜率(均值): {mean_not_slope:+.4f}")
        print(f"    概念logit斜率(均值abs): {mean_concept_slope:.4f}")
        print(f"    选择性: {selectivity:.1f}x")
        print(f"    语义控制: {results['summary']['semantic_control']}")
    
    _release_model(model)
    return results


# ============================================================
# N3: 方向一致性深度分析
# ============================================================

def n3_direction_consistency_analysis(model_name):
    """
    N3: 方向一致性深度分析
    
    核心设计:
    1. 计算每对动词的时态变换方向
    2. 规则动词的方向是否更一致(cos更高)?
    3. 不规则动词的方向是否更分散(cos更低)?
    4. 方向一致性与base_ratio的相关性
    
    这直接测试"方向操作"假说:
    - 如果变换沿1维方向, 则所有样本的方向应该高度一致
    - 如果变换是多维的, 则方向应该分散
    """
    print("\n" + "="*70)
    print("N3: 方向一致性深度分析")
    print("="*70)
    
    model, tokenizer, device, d_model, n_layers, W_U = _load_model_full(model_name)
    if model is None:
        return {}
    
    target_layer = n_layers // 2
    
    results = {
        'model': model_name,
        'target_layer': target_layer,
    }
    
    # --- Step 1: 收集所有归一化差分向量 ---
    print("\n  [N3.1] 收集归一化差分向量")
    
    regular_diffs = []
    regular_metas = []
    for pair in REGULAR_VERB_PAIRS[:20]:
        s1, s2, vp, vpa = pair
        h1, h2 = _get_hidden_states(model, tokenizer, device, s1, s2, target_layer)
        if h1 is None or h2 is None:
            continue
        delta = h2 - h1
        norm = np.linalg.norm(delta)
        if norm > 1e-8:
            regular_diffs.append(delta / norm)
            regular_metas.append({'verb': f"{vp}->{vpa}", 'base_ratio': None})
    
    irregular_diffs = []
    irregular_metas = []
    for pair in IRREGULAR_VERB_PAIRS[:20]:
        s1, s2, vp, vpa = pair
        h1, h2 = _get_hidden_states(model, tokenizer, device, s1, s2, target_layer)
        if h1 is None or h2 is None:
            continue
        delta = h2 - h1
        norm = np.linalg.norm(delta)
        if norm > 1e-8:
            irregular_diffs.append(delta / norm)
            irregular_metas.append({'verb': f"{vp}->{vpa}", 'base_ratio': None})
    
    polarity_diffs = []
    for pair in POLARITY_PAIRS[:15]:
        s1, s2 = pair[0], pair[1]
        h1, h2 = _get_hidden_states(model, tokenizer, device, s1, s2, target_layer)
        if h1 is None or h2 is None:
            continue
        delta = h2 - h1
        norm = np.linalg.norm(delta)
        if norm > 1e-8:
            polarity_diffs.append(delta / norm)
    
    semantic_diffs = []
    for pair in SEMANTIC_PAIRS[:10]:
        s1, s2 = pair[0], pair[1]
        h1, h2 = _get_hidden_states(model, tokenizer, device, s1, s2, target_layer)
        if h1 is None or h2 is None:
            continue
        delta = h2 - h1
        norm = np.linalg.norm(delta)
        if norm > 1e-8:
            semantic_diffs.append(delta / norm)
    
    print(f"    规则动词: {len(regular_diffs)}差分向量")
    print(f"    不规则动词: {len(irregular_diffs)}差分向量")
    print(f"    极性: {len(polarity_diffs)}差分向量")
    print(f"    语义: {len(semantic_diffs)}差分向量")
    
    # --- Step 2: 计算成对余弦相似度矩阵 ---
    print("\n  [N3.2] 计算成对余弦相似度")
    
    def pairwise_cos_matrix(diffs, name):
        """计算差分向量的成对余弦相似度"""
        n = len(diffs)
        if n < 2:
            return None, 0.0, 0.0
        
        cos_matrix = np.zeros((n, n))
        all_cos = []
        
        for i in range(n):
            for j in range(i+1, n):
                c = float(np.dot(diffs[i], diffs[j]))
                cos_matrix[i][j] = c
                cos_matrix[j][i] = c
                all_cos.append(c)
        
        mean_cos = float(np.mean(all_cos))
        min_cos = float(np.min(all_cos))
        
        print(f"    {name}: mean_cos={mean_cos:.4f}, min_cos={min_cos:.4f}, "
              f"median={float(np.median(all_cos)):.4f}")
        
        return cos_matrix, mean_cos, min_cos
    
    reg_cos_mat, reg_mean, reg_min = pairwise_cos_matrix(regular_diffs, "规则动词")
    irr_cos_mat, irr_mean, irr_min = pairwise_cos_matrix(irregular_diffs, "不规则动词")
    pol_cos_mat, pol_mean, pol_min = pairwise_cos_matrix(polarity_diffs, "极性")
    sem_cos_mat, sem_mean, sem_min = pairwise_cos_matrix(semantic_diffs, "语义")
    
    # --- Step 3: 主成分分析 — 有效维度 ---
    print("\n  [N3.3] 主成分分析 — 有效维度")
    
    def effective_dim(diffs, name, threshold=0.9):
        """PCA分析差分向量的有效维度"""
        if len(diffs) < 3:
            return 0, []
        
        X = np.array(diffs)  # [n, d_model]
        # 中心化
        X_centered = X - X.mean(axis=0, keepdims=True)
        # SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        total_var = np.sum(S**2)
        cumvar = np.cumsum(S**2) / total_var
        
        # 找到90%方差的维度
        dim_90 = int(np.searchsorted(cumvar, threshold) + 1)
        dim_95 = int(np.searchsorted(cumvar, 0.95) + 1)
        dim_99 = int(np.searchsorted(cumvar, 0.99) + 1)
        
        # 前5个主成分的方差占比
        top5_var = float(np.sum(S[:5]**2) / total_var) if len(S) >= 5 else 1.0
        
        print(f"    {name}: dim_90={dim_90}, dim_95={dim_95}, dim_99={dim_99}, "
              f"top5_var={top5_var:.3f}, n_samples={len(diffs)}")
        
        return dim_90, [float(s**2/total_var) for s in S[:10]]
    
    reg_dim, reg_vars = effective_dim(regular_diffs, "规则动词")
    irr_dim, irr_vars = effective_dim(irregular_diffs, "不规则动词")
    pol_dim, pol_vars = effective_dim(polarity_diffs, "极性")
    sem_dim, sem_vars = effective_dim(semantic_diffs, "语义")
    
    # --- Step 4: 规则/不规则交叉相似度 ---
    print("\n  [N3.4] 规则/不规则交叉相似度")
    
    if regular_diffs and irregular_diffs:
        cross_cos_all = []
        for rd in regular_diffs:
            for id_ in irregular_diffs:
                cross_cos_all.append(float(np.dot(rd, id_)))
        
        cross_mean = float(np.mean(cross_cos_all))
        cross_max = float(np.max(cross_cos_all))
        cross_min = float(np.min(cross_cos_all))
        
        print(f"    规则x不规则: mean={cross_mean:.4f}, max={cross_max:.4f}, min={cross_min:.4f}")
        results['cross_regular_irregular'] = {
            'mean': cross_mean,
            'max': cross_max,
            'min': cross_min,
        }
    
    # --- Step 5: 汇总 ---
    results['direction_consistency'] = {
        'regular': {'mean_cos': reg_mean, 'min_cos': reg_min, 'n': len(regular_diffs)},
        'irregular': {'mean_cos': irr_mean, 'min_cos': irr_min, 'n': len(irregular_diffs)},
        'polarity': {'mean_cos': pol_mean, 'min_cos': pol_min, 'n': len(polarity_diffs)},
        'semantic': {'mean_cos': sem_mean, 'min_cos': sem_min, 'n': len(semantic_diffs)},
    }
    
    results['effective_dim'] = {
        'regular': {'dim_90': reg_dim, 'top5_var': reg_vars[:5] if reg_vars else []},
        'irregular': {'dim_90': irr_dim, 'top5_var': irr_vars[:5] if irr_vars else []},
        'polarity': {'dim_90': pol_dim, 'top5_var': pol_vars[:5] if pol_vars else []},
        'semantic': {'dim_90': sem_dim, 'top5_var': sem_vars[:5] if sem_vars else []},
    }
    
    print("\n  [N3.5] 方向一致性 vs 纤维性 对比")
    print(f"    {'类别':<12} {'方向一致性cos':<16} {'有效维度dim90':<16} {'推断纤维性'}")
    print(f"    {'-'*60}")
    
    for name, cos, dim in [
        ('极性', pol_mean, pol_dim),
        ('规则动词', reg_mean, reg_dim),
        ('不规则动词', irr_mean, irr_dim),
        ('语义', sem_mean, sem_dim),
    ]:
        if cos > 0.85:
            fib = '强纤维(1维方向操作)'
        elif cos > 0.6:
            fib = '中纤维(低维方向操作)'
        elif cos > 0.3:
            fib = '弱纤维(高维流形操作)'
        else:
            fib = '无纤维(离散映射)'
        print(f"    {name:<12} {cos:<16.4f} {dim:<16} {fib}")
    
    _release_model(model)
    return results


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='形态-纤维关联验证')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['qwen3', 'glm4', 'deepseek7b'])
    parser.add_argument('--test', type=str, required=True,
                        choices=['n1', 'n2', 'n3', 'all'])
    args = parser.parse_args()
    
    model_name = args.model
    test = args.test
    
    # 结果保存目录
    save_dir = Path("results/morphology_fiber")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    if test in ('n1', 'all'):
        print(f"\n{'='*70}")
        print(f"N1: 规则vs不规则动词纤维性对比 — {model_name}")
        print(f"{'='*70}")
        t0 = time.time()
        result = n1_regular_vs_irregular(model_name)
        elapsed = time.time() - t0
        print(f"\n  N1完成: {elapsed:.1f}s")
        all_results['n1'] = result
        # 保存
        with open(save_dir / f"{model_name}_n1.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        # 释放GPU
        gc.collect()
        torch.cuda.empty_cache()
    
    if test in ('n2', 'all'):
        print(f"\n{'='*70}")
        print(f"N2: 极性因果干预 — {model_name}")
        print(f"{'='*70}")
        t0 = time.time()
        result = n2_polarity_causal_intervention(model_name)
        elapsed = time.time() - t0
        print(f"\n  N2完成: {elapsed:.1f}s")
        all_results['n2'] = result
        with open(save_dir / f"{model_name}_n2.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        gc.collect()
        torch.cuda.empty_cache()
    
    if test in ('n3', 'all'):
        print(f"\n{'='*70}")
        print(f"N3: 方向一致性分析 — {model_name}")
        print(f"{'='*70}")
        t0 = time.time()
        result = n3_direction_consistency_analysis(model_name)
        elapsed = time.time() - t0
        print(f"\n  N3完成: {elapsed:.1f}s")
        all_results['n3'] = result
        with open(save_dir / f"{model_name}_n3.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        gc.collect()
        torch.cuda.empty_cache()
    
    # 保存合并结果
    with open(save_dir / f"{model_name}_all.json", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    print(f"\n结果已保存到: {save_dir}")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


if __name__ == '__main__':
    main()
