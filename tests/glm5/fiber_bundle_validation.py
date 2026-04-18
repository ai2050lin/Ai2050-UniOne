"""
Phase CLXXXI: 纤维丛验证 — 从假说到可证伪的理论
==================================================

基于SAE发现的纤维丛假说:
- 基空间 = 概念空间 (高维, d_eff≈90)
- 纤维 = 功能空间 (低维, d_eff≈1-3)
- 时态/极性是全局截面: 在所有概念上一致
- 语义替换是基空间本身的变化: 概念特定

验证测试:
F1: 纤维不变性 — 功能变换是否保持概念位置?
F2: 因果干预 — 沿功能方向干预是否改变功能而不改变语义?
F3: 随机基线 — 训练模型的纤维结构是否显著偏离随机?

运行方式:
  python tests/glm5/fiber_bundle_validation.py --model qwen3 --test f1
  python tests/glm5/fiber_bundle_validation.py --model glm4 --test f2
  python tests/glm5/fiber_bundle_validation.py --model deepseek7b --test all
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
# 概念-功能交叉测试集
# ============================================================

def generate_cross_concept_functional_pairs():
    """
    生成概念×功能交叉测试集
    
    核心思路: 
    - 选10个不同概念(名词)
    - 对每个概念做3种功能变换(时态/极性/句法)
    - 验证: 功能变换是否保持概念位置不变?
    
    纤维丛预测:
    - T(x_cat) = x_cat + α·d_tense (概念"cat"不变, 只沿时态纤维移动)
    - T(x_dog) = x_dog + α·d_tense (概念"dog"不变, 沿同一时态纤维移动)
    - → T(x_cat) 和 T(x_dog) 的基空间投影应与 x_cat, x_dog 相同
    """
    concepts = [
        "cat", "dog", "bird", "fish", "tree",
        "car", "book", "house", "river", "mountain",
        # 额外10个增加统计力
        "child", "woman", "star", "cloud", "door",
        "phone", "key", "flower", "bridge", "table",
    ]
    
    pairs = {
        'tense_cross': [],    # 同概念不同时态
        'polarity_cross': [], # 同概念不同极性
        'syntax_cross': [],   # 同概念不同句法
        'semantic_cross': [], # 不同概念同时态
        'dual_cross': [],     # 不同概念+不同时态(双重变化)
    }
    
    for concept in concepts:
        # Tense cross: 同概念的现在时 vs 过去时
        pairs['tense_cross'].append((
            f"The {concept} walks every day",
            f"The {concept} walked yesterday",
        ))
        pairs['tense_cross'].append((
            f"She sees the {concept}",
            f"She saw the {concept}",
        ))
        pairs['tense_cross'].append((
            f"The {concept} runs fast",
            f"The {concept} ran fast",
        ))
        
        # Polarity cross: 同概念的肯定 vs 否定
        pairs['polarity_cross'].append((
            f"The {concept} is here",
            f"The {concept} is not here",
        ))
        pairs['polarity_cross'].append((
            f"I like the {concept}",
            f"I do not like the {concept}",
        ))
        pairs['polarity_cross'].append((
            f"The {concept} was found",
            f"The {concept} was not found",
        ))
        
        # Syntax cross: 同概念的单数 vs 复数
        if concept == "child":
            plural = "children"
        elif concept == "fish":
            plural = "fish"
        else:
            plural = concept + "s"
        pairs['syntax_cross'].append((
            f"The {concept} sits on the mat",
            f"The {plural} sit on the mat",
        ))
        
        # Semantic cross: 不同概念同时态
        for other in concepts[:10]:  # 前10个概念两两配对
            if other != concept:
                pairs['semantic_cross'].append((
                    f"The {concept} walks every day",
                    f"The {other} walks every day",
                ))
    
    # Dual cross: 不同概念+不同时态
    for i, c1 in enumerate(concepts[:10]):
        for c2 in concepts[i+1:i+3]:  # 每个概念配2个其他概念
            pairs['dual_cross'].append((
                f"The {c1} walks every day",
                f"The {c2} walked yesterday",
            ))
    
    # 统计
    for dim, plist in pairs.items():
        print(f"  {dim}: {len(plist)} 对")
    
    return pairs, concepts


# ============================================================
# F1: 纤维不变性验证
# ============================================================

def f1_fiber_invariance(model_name):
    """
    F1: 验证功能变换是否保持概念位置不变
    
    核心测试:
    1. 对同一概念做功能变换(时态/极性)
    2. 提取隐藏状态的基空间投影(去掉功能方向的分量)
    3. 检查: 基空间投影是否在功能变换前后保持不变?
    
    纤维丛预测:
    - h_present = h_base + α_present · d_tense
    - h_past    = h_base + α_past · d_tense
    - → h_present和h_past去掉d_tense分量后应该相同(=h_base)
    
    反面预测:
    - 语义替换: h_cat ≠ h_dog (基空间位置不同)
    """
    print("\n" + "="*70)
    print("F1: 纤维不变性验证")
    print("="*70)
    
    # 加载模型
    if model_name in ["glm4", "deepseek7b"]:
        model, tokenizer, device = load_model_4bit(model_name)
    else:
        from model_utils import load_model
        model, tokenizer, device = load_model(model_name)
    
    d_model = model.config.hidden_size
    layers = get_layers(model)
    n_layers = len(layers)
    target_layer = n_layers // 2
    
    # 生成测试集
    pairs, concepts = generate_cross_concept_functional_pairs()
    
    # === F1.1: 先提取功能方向 ===
    print("\n  [F1.1] 提取功能方向 (从训练数据)")
    
    # 用大量句对提取方向
    functional_directions = {}
    
    # 时态方向
    tense_diffs = []
    for s1, s2 in pairs['tense_cross'][:40]:
        h1, h2 = get_hidden_states(model, tokenizer, device, s1, s2, target_layer)
        if h1 is not None and h2 is not None:
            diff = h2 - h1
            norm = np.linalg.norm(diff)
            if norm > 1e-8:
                tense_diffs.append(diff / norm)
    
    if tense_diffs:
        d_tense = np.mean(tense_diffs, axis=0)
        d_tense = d_tense / (np.linalg.norm(d_tense) + 1e-30)
        functional_directions['tense'] = d_tense
        print(f"    tense方向: {len(tense_diffs)} 样本, norm={np.linalg.norm(np.mean(tense_diffs, axis=0)):.4f}")
    
    # 极性方向
    polarity_diffs = []
    for s1, s2 in pairs['polarity_cross'][:40]:
        h1, h2 = get_hidden_states(model, tokenizer, device, s1, s2, target_layer)
        if h1 is not None and h2 is not None:
            diff = h2 - h1
            norm = np.linalg.norm(diff)
            if norm > 1e-8:
                polarity_diffs.append(diff / norm)
    
    if polarity_diffs:
        d_polarity = np.mean(polarity_diffs, axis=0)
        d_polarity = d_polarity / (np.linalg.norm(d_polarity) + 1e-30)
        functional_directions['polarity'] = d_polarity
        print(f"    polarity方向: {len(polarity_diffs)} 样本")
    
    # 语义方向(每个概念对的方向)
    semantic_diffs = []
    for s1, s2 in pairs['semantic_cross'][:40]:
        h1, h2 = get_hidden_states(model, tokenizer, device, s1, s2, target_layer)
        if h1 is not None and h2 is not None:
            diff = h2 - h1
            norm = np.linalg.norm(diff)
            if norm > 1e-8:
                semantic_diffs.append(diff / norm)
    
    if semantic_diffs:
        d_semantic = np.mean(semantic_diffs, axis=0)
        d_semantic = d_semantic / (np.linalg.norm(d_semantic) + 1e-30)
        functional_directions['semantic'] = d_semantic
        print(f"    semantic方向: {len(semantic_diffs)} 样本")
    
    # === F1.2: 纤维不变性核心测试 ===
    print("\n  [F1.2] 纤维不变性核心测试")
    
    results = {}
    
    # 测试1: 时态变换的基空间不变性
    print("\n    === 时态变换 ===")
    tense_invariance = []
    
    for concept in concepts[:10]:
        s_present = f"The {concept} walks every day"
        s_past = f"The {concept} walked yesterday"
        
        h_present, h_past = get_hidden_states(model, tokenizer, device, s_present, s_past, target_layer)
        if h_present is None or h_past is None:
            continue
        
        # 投影到时态方向的分量
        if 'tense' in functional_directions:
            d_t = functional_directions['tense']
            tense_proj_present = np.dot(h_present, d_t)
            tense_proj_past = np.dot(h_past, d_t)
            
            # 去掉时态分量后的基空间表示
            h_base_present = h_present - tense_proj_present * d_t
            h_base_past = h_past - tense_proj_past * d_t
            
            # 基空间不变性: 去掉时态分量后两个表示是否相同?
            base_cos = float(np.dot(h_base_present, h_base_past) / 
                           (np.linalg.norm(h_base_present) * np.linalg.norm(h_base_past) + 1e-30))
            
            # 原始表示的余弦(不去掉任何分量)
            orig_cos = float(np.dot(h_present, h_past) / 
                           (np.linalg.norm(h_present) * np.linalg.norm(h_past) + 1e-30))
            
            # 功能方向上的差异
            tense_shift = float(tense_proj_past - tense_proj_present)
            
            tense_invariance.append({
                'concept': concept,
                'base_cos': base_cos,
                'orig_cos': orig_cos,
                'tense_proj_present': float(tense_proj_present),
                'tense_proj_past': float(tense_proj_past),
                'tense_shift': tense_shift,
            })
            
            print(f"      {concept}: base_cos={base_cos:.6f}, orig_cos={orig_cos:.6f}, tense_shift={tense_shift:.4f}")
    
    if tense_invariance:
        avg_base_cos = np.mean([x['base_cos'] for x in tense_invariance])
        avg_orig_cos = np.mean([x['orig_cos'] for x in tense_invariance])
        print(f"\n    时态平均: base_cos={avg_base_cos:.6f}, orig_cos={avg_orig_cos:.6f}")
        results['tense_invariance'] = {
            'samples': tense_invariance,
            'avg_base_cos': float(avg_base_cos),
            'avg_orig_cos': float(avg_orig_cos),
            'base_cos_drop': float(avg_orig_cos - avg_base_cos),
        }
    
    # 测试2: 极性变换的基空间不变性
    print("\n    === 极性变换 ===")
    polarity_invariance = []
    
    for concept in concepts[:10]:
        s_aff = f"The {concept} is here"
        s_neg = f"The {concept} is not here"
        
        h_aff, h_neg = get_hidden_states(model, tokenizer, device, s_aff, s_neg, target_layer)
        if h_aff is None or h_neg is None:
            continue
        
        if 'polarity' in functional_directions:
            d_p = functional_directions['polarity']
            pol_proj_aff = np.dot(h_aff, d_p)
            pol_proj_neg = np.dot(h_neg, d_p)
            
            h_base_aff = h_aff - pol_proj_aff * d_p
            h_base_neg = h_neg - pol_proj_neg * d_p
            
            base_cos = float(np.dot(h_base_aff, h_base_neg) / 
                           (np.linalg.norm(h_base_aff) * np.linalg.norm(h_base_neg) + 1e-30))
            orig_cos = float(np.dot(h_aff, h_neg) / 
                           (np.linalg.norm(h_aff) * np.linalg.norm(h_neg) + 1e-30))
            pol_shift = float(pol_proj_neg - pol_proj_aff)
            
            polarity_invariance.append({
                'concept': concept,
                'base_cos': base_cos,
                'orig_cos': orig_cos,
                'pol_proj_aff': float(pol_proj_aff),
                'pol_proj_neg': float(pol_proj_neg),
                'pol_shift': pol_shift,
            })
            
            print(f"      {concept}: base_cos={base_cos:.6f}, orig_cos={orig_cos:.6f}, pol_shift={pol_shift:.4f}")
    
    if polarity_invariance:
        avg_base_cos = np.mean([x['base_cos'] for x in polarity_invariance])
        avg_orig_cos = np.mean([x['orig_cos'] for x in polarity_invariance])
        print(f"\n    极性平均: base_cos={avg_base_cos:.6f}, orig_cos={avg_orig_cos:.6f}")
        results['polarity_invariance'] = {
            'samples': polarity_invariance,
            'avg_base_cos': float(avg_base_cos),
            'avg_orig_cos': float(avg_orig_cos),
            'base_cos_drop': float(avg_orig_cos - avg_base_cos),
        }
    
    # 测试3: 语义替换的基空间变化(对比基准)
    print("\n    === 语义替换 (对比基准) ===")
    semantic_shift = []
    
    for i, c1 in enumerate(concepts[:10]):
        for c2 in concepts[i+1:i+2]:
            s1 = f"The {c1} walks every day"
            s2 = f"The {c2} walks every day"
            
            h1, h2 = get_hidden_states(model, tokenizer, device, s1, s2, target_layer)
            if h1 is None or h2 is None:
                continue
            
            orig_cos = float(np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-30))
            
            # 去掉时态和极性分量
            if 'tense' in functional_directions:
                d_t = functional_directions['tense']
                h1_base = h1 - np.dot(h1, d_t) * d_t
                h2_base = h2 - np.dot(h2, d_t) * d_t
                if 'polarity' in functional_directions:
                    d_p = functional_directions['polarity']
                    h1_base = h1_base - np.dot(h1_base, d_p) * d_p
                    h2_base = h2_base - np.dot(h2_base, d_p) * d_p
                base_cos = float(np.dot(h1_base, h2_base) / 
                               (np.linalg.norm(h1_base) * np.linalg.norm(h2_base) + 1e-30))
            else:
                base_cos = orig_cos
            
            semantic_shift.append({
                'concept_pair': f"{c1}-{c2}",
                'orig_cos': orig_cos,
                'base_cos': base_cos,
            })
            
            print(f"      {c1}-{c2}: orig_cos={orig_cos:.6f}, base_cos={base_cos:.6f}")
    
    if semantic_shift:
        avg_base_cos = np.mean([x['base_cos'] for x in semantic_shift])
        avg_orig_cos = np.mean([x['orig_cos'] for x in semantic_shift])
        print(f"\n    语义平均: base_cos={avg_base_cos:.6f}, orig_cos={avg_orig_cos:.6f}")
        results['semantic_shift'] = {
            'samples': semantic_shift,
            'avg_base_cos': float(avg_base_cos),
            'avg_orig_cos': float(avg_orig_cos),
        }
    
    # === F1.3: 双重变化测试 ===
    print("\n  [F1.3] 双重变化测试 (不同概念+不同时态)")
    
    dual_results = []
    for s1, s2 in pairs['dual_cross'][:20]:
        h1, h2 = get_hidden_states(model, tokenizer, device, s1, s2, target_layer)
        if h1 is None or h2 is None:
            continue
        
        orig_cos = float(np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-30))
        
        if 'tense' in functional_directions:
            d_t = functional_directions['tense']
            h1_base = h1 - np.dot(h1, d_t) * d_t
            h2_base = h2 - np.dot(h2, d_t) * d_t
            base_cos = float(np.dot(h1_base, h2_base) / 
                           (np.linalg.norm(h1_base) * np.linalg.norm(h2_base) + 1e-30))
        else:
            base_cos = orig_cos
        
        dual_results.append({
            's1': s1, 's2': s2,
            'orig_cos': orig_cos,
            'base_cos': base_cos,
        })
    
    if dual_results:
        avg_base = np.mean([x['base_cos'] for x in dual_results])
        avg_orig = np.mean([x['orig_cos'] for x in dual_results])
        print(f"    双重变化: orig_cos={avg_orig:.6f}, base_cos={avg_base:.6f}")
        results['dual_change'] = {
            'avg_orig_cos': float(avg_orig),
            'avg_base_cos': float(avg_base),
        }
    
    # === F1.4: 跨层分析 ===
    print("\n  [F1.4] 跨层纤维不变性")
    
    layer_invariance = {}
    test_concepts = concepts[:5]
    
    for layer_idx in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        layer_tense_base = []
        
        for concept in test_concepts:
            s_present = f"The {concept} walks every day"
            s_past = f"The {concept} walked yesterday"
            
            h1, h2 = get_hidden_states(model, tokenizer, device, s_present, s_past, layer_idx)
            if h1 is None or h2 is None:
                continue
            
            # 时态方向的跨层提取
            if 'tense' in functional_directions:
                d_t = functional_directions['tense']
                t1 = np.dot(h1, d_t)
                t2 = np.dot(h2, d_t)
                h1_base = h1 - t1 * d_t
                h2_base = h2 - t2 * d_t
                base_cos = float(np.dot(h1_base, h2_base) / 
                               (np.linalg.norm(h1_base) * np.linalg.norm(h2_base) + 1e-30))
                orig_cos = float(np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-30))
                layer_tense_base.append({'orig_cos': orig_cos, 'base_cos': base_cos})
        
        if layer_tense_base:
            avg = {
                'orig_cos': float(np.mean([x['orig_cos'] for x in layer_tense_base])),
                'base_cos': float(np.mean([x['base_cos'] for x in layer_tense_base])),
            }
            layer_invariance[str(layer_idx)] = avg
            print(f"    Layer {layer_idx}: orig={avg['orig_cos']:.6f}, base={avg['base_cos']:.6f}")
    
    results['layer_invariance'] = layer_invariance
    
    # === 汇总判定 ===
    print("\n  === F1 纤维不变性判定 ===")
    
    if 'tense_invariance' in results:
        ti = results['tense_invariance']
        if ti['avg_base_cos'] > 0.999:
            verdict_tense = "强纤维不变性 (时态变换几乎不改变基空间)"
        elif ti['avg_base_cos'] > 0.99:
            verdict_tense = "中等纤维不变性 (时态变换轻微扰动基空间)"
        else:
            verdict_tense = "弱/无纤维不变性 (时态变换显著改变基空间)"
        print(f"    时态: {verdict_tense} (base_cos={ti['avg_base_cos']:.6f})")
        results['tense_verdict'] = verdict_tense
    
    if 'polarity_invariance' in results:
        pi = results['polarity_invariance']
        if pi['avg_base_cos'] > 0.999:
            verdict_polarity = "强纤维不变性"
        elif pi['avg_base_cos'] > 0.99:
            verdict_polarity = "中等纤维不变性"
        else:
            verdict_polarity = "弱/无纤维不变性"
        print(f"    极性: {verdict_polarity} (base_cos={pi['avg_base_cos']:.6f})")
        results['polarity_verdict'] = verdict_polarity
    
    if 'semantic_shift' in results:
        ss = results['semantic_shift']
        print(f"    语义: base_cos={ss['avg_base_cos']:.6f} (对比基准)")
    
    release_model(model)
    return results


# ============================================================
# F2: 因果干预验证
# ============================================================

def f2_causal_intervention(model_name):
    """
    F2: 沿功能方向干预, 验证是否改变功能而不改变语义
    
    核心测试:
    1. 取一个句子的隐藏表示 h
    2. 沿时态方向干预: h' = h + α·d_tense
    3. 用干预后的表示解码, 检查:
       - 时态是否改变? (功能因果性)
       - 语义是否保持? (语义不变性)
    
    注意: 直接用隐藏状态解码需要模型支持, 这里用替代方案:
    - 测量干预后表示与目标时态表示的距离变化
    - 测量干预后表示与原始语义表示的距离变化
    """
    print("\n" + "="*70)
    print("F2: 因果干预验证")
    print("="*70)
    
    # 加载模型
    if model_name in ["glm4", "deepseek7b"]:
        model, tokenizer, device = load_model_4bit(model_name)
    else:
        from model_utils import load_model
        model, tokenizer, device = load_model(model_name)
    
    d_model = model.config.hidden_size
    layers = get_layers(model)
    n_layers = len(layers)
    target_layer = n_layers // 2
    
    # 测试句对
    test_cases = [
        # (present, past, concept)
        ("The cat walks every day", "The cat walked yesterday", "cat"),
        ("The dog runs fast", "The dog ran fast", "dog"),
        ("The bird flies high", "The bird flew high", "bird"),
        ("The tree grows tall", "The tree grew tall", "tree"),
        ("The fish swims deep", "The fish swam deep", "fish"),
    ]
    
    results = {}
    
    for present, past, concept in test_cases:
        print(f"\n  === {concept}: '{present}' → '{past}' ===")
        
        # 获取隐藏状态
        h_present = get_single_hidden(model, tokenizer, device, present, target_layer)
        h_past = get_single_hidden(model, tokenizer, device, past, target_layer)
        
        if h_present is None or h_past is None:
            continue
        
        # 计算时态方向
        d_tense = h_past - h_present
        d_norm = np.linalg.norm(d_tense)
        if d_norm < 1e-8:
            continue
        d_tense = d_tense / d_norm
        
        # 干预: 从present沿d_tense方向移动不同距离
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        
        intervention_results = []
        for alpha in alphas:
            h_intervened = h_present + alpha * d_tense
            
            # 测量1: 干预后表示与past表示的距离(越小=越接近目标时态)
            cos_to_past = float(np.dot(h_intervened, h_past) / 
                              (np.linalg.norm(h_intervened) * np.linalg.norm(h_past) + 1e-30))
            
            # 测量2: 干预后表示与present表示的距离(越大=越远离原始时态)
            cos_to_present = float(np.dot(h_intervened, h_present) / 
                                 (np.linalg.norm(h_intervened) * np.linalg.norm(h_present) + 1e-30))
            
            # 测量3: 去掉d_tense分量后与present的基空间相似度
            tense_proj = np.dot(h_intervened, d_tense)
            h_base_intervened = h_intervened - tense_proj * d_tense
            tense_proj_present = np.dot(h_present, d_tense)
            h_base_present = h_present - tense_proj_present * d_tense
            
            base_cos = float(np.dot(h_base_intervened, h_base_present) / 
                           (np.linalg.norm(h_base_intervened) * np.linalg.norm(h_base_present) + 1e-30))
            
            intervention_results.append({
                'alpha': alpha,
                'cos_to_past': cos_to_past,
                'cos_to_present': cos_to_present,
                'base_cos': base_cos,
            })
            
            print(f"    α={alpha:.2f}: cos_to_past={cos_to_past:.4f}, "
                  f"cos_to_present={cos_to_present:.4f}, base_cos={base_cos:.6f}")
        
        results[concept] = intervention_results
    
    # 汇总
    print("\n  === F2 因果干预汇总 ===")
    
    # 在α=1.0时的平均效果
    alpha1_results = []
    for concept, res in results.items():
        for r in res:
            if abs(r['alpha'] - 1.0) < 0.01:
                alpha1_results.append(r)
    
    if alpha1_results:
        avg_cos_past = np.mean([r['cos_to_past'] for r in alpha1_results])
        avg_cos_present = np.mean([r['cos_to_present'] for r in alpha1_results])
        avg_base = np.mean([r['base_cos'] for r in alpha1_results])
        
        print(f"    α=1.0: cos_to_past={avg_cos_past:.4f}, cos_to_present={avg_cos_present:.4f}, "
              f"base_cos={avg_base:.6f}")
        
        if avg_base > 0.999:
            verdict = "强因果不变性: 沿时态方向干预不改变基空间"
        elif avg_base > 0.99:
            verdict = "中等因果不变性: 干预轻微扰动基空间"
        else:
            verdict = "弱因果不变性: 干预显著改变基空间"
        print(f"    判定: {verdict}")
        results['summary'] = {
            'alpha1_avg_cos_to_past': float(avg_cos_past),
            'alpha1_avg_cos_to_present': float(avg_cos_present),
            'alpha1_avg_base_cos': float(avg_base),
            'verdict': verdict,
        }
    
    release_model(model)
    return results


# ============================================================
# F3: 随机基线对比
# ============================================================

def f3_random_baseline(model_name):
    """
    F3: 训练模型的纤维结构是否显著偏离随机?
    
    核心测试:
    1. 在同一空间生成随机高斯向量对
    2. 对随机向量做同样的"纤维不变性"测试
    3. 如果训练模型的base_cos显著高于随机 → 纤维结构是学习产物
    """
    print("\n" + "="*70)
    print("F3: 随机基线对比")
    print("="*70)
    
    # 获取训练模型的F1结果
    f1_results = f1_fiber_invariance(model_name)
    
    d_model = 2560  # 默认; 实际中从模型获取
    
    # 随机基线
    n_random = 100
    random_base_cosines = []
    
    for _ in range(n_random):
        # 生成"present"和"past"随机向量
        h1 = np.random.randn(d_model)
        h1 = h1 / np.linalg.norm(h1)
        
        # "时态方向": 随机方向
        d_t = np.random.randn(d_model)
        d_t = d_t / np.linalg.norm(d_t)
        
        # "past" = "present" + shift along d_t
        shift = np.random.randn() * 0.5  # 随机shift幅度
        h2 = h1 + shift * d_t
        h2 = h2 / np.linalg.norm(h2)
        
        # 去掉d_t分量后的基空间相似度
        proj1 = np.dot(h1, d_t)
        proj2 = np.dot(h2, d_t)
        h1_base = h1 - proj1 * d_t
        h2_base = h2 - proj2 * d_t
        
        base_cos = float(np.dot(h1_base, h2_base) / 
                        (np.linalg.norm(h1_base) * np.linalg.norm(h2_base) + 1e-30))
        random_base_cosines.append(base_cos)
    
    # 随机基线统计
    random_mean = float(np.mean(random_base_cosines))
    random_std = float(np.std(random_base_cosines))
    random_ci = float(1.96 * random_std / (n_random ** 0.5))
    
    print(f"  随机基线: base_cos = {random_mean:.6f} ± {random_ci:.6f}")
    print(f"  随机基线范围: [{min(random_base_cosines):.6f}, {max(random_base_cosines):.6f}]")
    
    # 与训练模型对比
    results = {
        'random_baseline': {
            'n_samples': n_random,
            'mean_base_cos': random_mean,
            'std_base_cos': random_std,
            'ci95': random_ci,
        },
        'trained_model': {},
    }
    
    if 'tense_invariance' in f1_results:
        trained_base = f1_results['tense_invariance']['avg_base_cos']
        z_score = (trained_base - random_mean) / (random_std + 1e-30)
        p_value = float(2 * (1 - __import__('scipy').stats.norm.cdf(abs(z_score)))) if random_std > 0 else 0.0
        
        print(f"\n  时态纤维不变性:")
        print(f"    训练模型: {trained_base:.6f}")
        print(f"    随机基线: {random_mean:.6f} ± {random_ci:.6f}")
        print(f"    Z-score: {z_score:.2f}")
        
        results['trained_model']['tense'] = {
            'base_cos': trained_base,
            'z_score': float(z_score),
            'significant': abs(z_score) > 2.0,
        }
    
    if 'polarity_invariance' in f1_results:
        trained_base = f1_results['polarity_invariance']['avg_base_cos']
        z_score = (trained_base - random_mean) / (random_std + 1e-30)
        
        print(f"\n  极性纤维不变性:")
        print(f"    训练模型: {trained_base:.6f}")
        print(f"    随机基线: {random_mean:.6f} ± {random_ci:.6f}")
        print(f"    Z-score: {z_score:.2f}")
        
        results['trained_model']['polarity'] = {
            'base_cos': trained_base,
            'z_score': float(z_score),
            'significant': abs(z_score) > 2.0,
        }
    
    return results


# ============================================================
# 辅助函数
# ============================================================

def get_hidden_states(model, tokenizer, device, s1, s2, layer_idx):
    """获取两个句子的隐藏状态"""
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


def get_single_hidden(model, tokenizer, device, text, layer_idx):
    """获取单个句子的隐藏状态"""
    try:
        toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            h = model(**toks, output_hidden_states=True).hidden_states[layer_idx]
        return h[0].mean(0).detach().cpu().float().numpy()
    except:
        return None


def load_model_4bit(model_name):
    """加载4-bit量化模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    MODEL_CONFIGS = {
        "glm4": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "deepseek7b": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    }
    
    path = MODEL_CONFIGS[model_name]
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


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(f"Cannot find layers in {type(model).__name__}")


def release_model(model):
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("[release] GPU memory released")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        return super().default(obj)


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Fiber Bundle Validation")
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--test", type=str, default="all",
                        choices=["f1", "f2", "f3", "all"],
                        help="运行哪个测试")
    args = parser.parse_args()
    
    model_name = args.model
    print(f"\n{'#'*70}")
    print(f"# 纤维丛验证: F1(不变性) + F2(因果干预) + F3(随机基线)")
    print(f"# Model: {model_name}")
    print(f"{'#'*70}")
    
    all_results = {
        'model': model_name,
        'timestamp': datetime.now().isoformat(),
    }
    
    output_dir = Path(r"d:\Ai2050\TransformerLens-Project\results\fiber_bundle")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # F1: 纤维不变性
    if args.test in ["f1", "all"]:
        f1_results = f1_fiber_invariance(model_name)
        all_results['f1_invariance'] = f1_results
    
    # F2: 因果干预
    if args.test in ["f2", "all"]:
        f2_results = f2_causal_intervention(model_name)
        all_results['f2_intervention'] = f2_results
    
    # F3: 随机基线
    if args.test in ["f3", "all"]:
        f3_results = f3_random_baseline(model_name)
        all_results['f3_random_baseline'] = f3_results
    
    # 保存结果
    output_path = output_dir / f"{model_name}_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n{'='*70}")
    print(f"所有结果已保存到 {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
