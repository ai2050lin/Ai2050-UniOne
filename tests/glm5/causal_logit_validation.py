"""
Phase CLXXXII: 因果解码验证 — 从统计相关性到因果机制
==================================================

核心目标: 用Logit Lens验证沿d_tense干预是否真的改变时态输出
这将把"时态纤维方向在统计上与基空间正交"升级为
"沿时态纤维干预因果地改变模型的时态输出"

测试:
L1: Logit Lens因果干预 — 干预后logit中动词时态token的变化
L2: 组合纤维验证 — 时态+极性复合变换是否走二维平面
L3: 欧氏距离测量 — 消除维度效应对余弦相似度的影响
L4: 中文纤维验证 — 中文时态/否定标记是否也有纤维结构

运行方式:
  python tests/glm5/causal_logit_validation.py --model qwen3 --test l1
  python tests/glm5/causal_logit_validation.py --model glm4 --test l2
  python tests/glm5/causal_logit_validation.py --model deepseek7b --test l3
  python tests/glm5/causal_logit_validation.py --model qwen3 --test l4
  python tests/glm5/causal_logit_validation.py --model qwen3 --test all
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
# 英文测试集
# ============================================================

ENGLISH_TENSE_PAIRS = [
    # (present_sentence, past_sentence, present_verb, past_verb)
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
    ("The key fits perfectly", "The key fit perfectly", "fit", "fit"),
]

ENGLISH_POLARITY_PAIRS = [
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

# 组合纤维测试: 时态+极性
COMBINED_PAIRS = [
    # (present_aff, past_aff, present_neg, past_neg, verb_present, verb_past)
    ("The cat walks", "The cat walked", "The cat does not walk", "The cat did not walk", "walk", "walked"),
    ("The dog runs", "The dog ran", "The dog does not run", "The dog did not run", "run", "ran"),
    ("The bird flies", "The bird flew", "The bird does not fly", "The bird did not fly", "fly", "flew"),
    ("She sees it", "She saw it", "She does not see it", "She did not see it", "see", "saw"),
    ("He takes it", "He took it", "He does not take it", "He did not take it", "take", "took"),
    ("The fish swims", "The fish swam", "The fish does not swim", "The fish did not swim", "swim", "swam"),
    ("The tree grows", "The tree grew", "The tree does not grow", "The tree did not grow", "grow", "grew"),
    ("I think so", "I thought so", "I do not think so", "I did not think so", "think", "thought"),
    ("She writes well", "She wrote well", "She does not write well", "She did not write well", "write", "wrote"),
    ("The star shines", "The star shone", "The star does not shine", "The star did not shine", "shine", "shone"),
]

# 中文测试集
CHINESE_TENSE_PAIRS = [
    # 中文没有时态屈折，但用助词"了"表示完成体
    ("猫走路", "猫走了路", "walk", "walked"),
    ("狗跑步", "狗跑了步", "run", "ran"),
    ("鸟飞翔", "鸟飞翔了", "fly", "flew"),
    ("树生长", "树生长了", "grow", "grew"),
    ("鱼游泳", "鱼游了泳", "swim", "swam"),
    ("她看书", "她看了书", "read", "read_past"),
    ("他吃饭", "他吃了饭", "eat", "ate"),
    ("孩子来家", "孩子来了家", "come", "came"),
    ("花开", "花开了", "bloom", "bloomed"),
    ("水流", "水流了", "flow", "flowed"),
    ("星星闪耀", "星星闪耀了", "shine", "shone"),
    ("云移动", "云移动了", "move", "moved"),
    ("门打开", "门打开了", "open", "opened"),
    ("电话响了", "电话响过了", "ring", "rang"),
    ("桥站立", "桥站立了", "stand", "stood"),
    ("我思考", "我思考了", "think", "thought"),
    ("她写字", "她写了字", "write", "wrote"),
    ("钥匙工作", "钥匙工作了", "work", "worked"),
    ("河流动", "河流了", "flow", "flowed"),
    ("山高耸", "山高耸了", "tower", "towered"),
]

CHINESE_POLARITY_PAIRS = [
    ("猫在这里", "猫不在这里", "cat"),
    ("狗很快乐", "狗不快乐", "dog"),
    ("书被找到了", "书没被找到", "book"),
    ("我喜欢车", "我不喜欢车", "car"),
    ("她知道答案", "她不知道答案", "answer"),
    ("房子很大", "房子不大", "house"),
    ("河向北流", "河不向北流", "river"),
    ("他会游泳", "他不会游泳", "swim"),
    ("鸟会来", "鸟不会来", "bird"),
    ("门关了", "门没关", "door"),
    ("电话在工作", "电话不在工作", "phone"),
    ("花开了", "花没开", "flower"),
    ("我理解计划", "我不理解计划", "plan"),
    ("她喜欢电影", "她不喜欢电影", "movie"),
    ("桥安全", "桥不安全", "bridge"),
]


# ============================================================
# L1: Logit Lens因果干预
# ============================================================

def l1_logit_lens_intervention(model_name):
    """
    L1: 用Logit Lens验证沿d_tense干预的因果效应
    
    核心设计:
    1. 对20个句对(present/past)，提取中间层隐藏状态
    2. 计算d_tense方向(从大量句对中提取)
    3. 沿d_tense干预: h' = h_present + α·d_tense
    4. 用lm_head解码h'，检查动词时态token的logit变化
    5. 同时检查语义token(名词)的logit是否保持不变
    
    Logit Lens: logits = h @ W_U.T (lm_head的线性投影)
    """
    print("\n" + "="*70)
    print("L1: Logit Lens因果干预验证")
    print("="*70)
    
    # 加载模型
    model, tokenizer, device, d_model, n_layers, W_U_np = _load_model_full(model_name)
    if model is None:
        return {}
    
    target_layer = n_layers // 2
    layers = _get_layers(model)
    
    # 获取lm_head权重矩阵 (vocab_size, d_model)
    W_U = W_U_np  # [vocab_size, d_model]
    
    # === Step 1: 提取时态方向 ===
    print("\n  [L1.1] 提取时态方向")
    d_tense = _extract_functional_direction(model, tokenizer, device, 
                                             ENGLISH_TENSE_PAIRS[:15], target_layer, 'tense')
    if d_tense is None:
        print("  ERROR: 无法提取时态方向")
        _release_model(model)
        return {}
    
    # 提取极性方向
    d_polarity = _extract_functional_direction(model, tokenizer, device,
                                                ENGLISH_POLARITY_PAIRS[:15], target_layer, 'polarity')
    
    # === Step 2: Logit Lens干预测试 ===
    print("\n  [L1.2] Logit Lens干预测试")
    
    results = {
        'tense_intervention': [],
        'polarity_intervention': [],
        'semantic_control': [],
    }
    
    # --- 时态干预 ---
    print("\n    === 时态干预 ===")
    alphas = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    for present, past, verb_pres, verb_past in ENGLISH_TENSE_PAIRS:
        # 获取隐藏状态
        h_present = _get_single_hidden(model, tokenizer, device, present, target_layer)
        h_past = _get_single_hidden(model, tokenizer, device, past, target_layer)
        if h_present is None or h_past is None:
            continue
        
        # 获取动词token ID
        pres_ids = tokenizer.encode(verb_pres, add_special_tokens=False)
        past_ids = tokenizer.encode(verb_past, add_special_tokens=False)
        if not pres_ids or not past_ids:
            continue
        pres_tok_id = pres_ids[0]
        past_tok_id = past_ids[0]
        
        # 获取名词token(第一个非特殊token)
        pres_sent_ids = tokenizer.encode(present, add_special_tokens=False)
        noun_tok_id = pres_sent_ids[1] if len(pres_sent_ids) > 1 else None  # "The"后的名词
        
        # Logit Lens解码
        logits_present = h_present @ W_U.T  # [vocab_size]
        logits_past = h_past @ W_U.T
        
        # 原始logit差
        orig_diff = logits_past[past_tok_id] - logits_present[past_tok_id]
        
        intervention_results = []
        for alpha in alphas:
            h_intervened = h_present + alpha * d_tense
            logits_intervened = h_intervened @ W_U.T
            
            # 核心指标: 动词时态token的logit变化
            pres_logit = float(logits_intervened[pres_tok_id])
            past_logit = float(logits_intervened[past_tok_id])
            logit_diff = past_logit - pres_logit
            
            # 语义控制: 名词token的logit变化
            noun_logit = float(logits_intervened[noun_tok_id]) if noun_tok_id else 0.0
            noun_logit_present = float(logits_present[noun_tok_id]) if noun_tok_id else 0.0
            noun_change = noun_logit - noun_logit_present
            
            # Top-1 token变化
            top1_token = int(np.argmax(logits_intervened))
            top1_word = tokenizer.decode([top1_token])
            # 清理可能的BOM/special字符
            top1_word = top1_word.replace('\ufeff', '').replace('\ufffd', '?').strip()[:20]
            
            intervention_results.append({
                'alpha': float(alpha),
                'pres_verb_logit': pres_logit,
                'past_verb_logit': past_logit,
                'logit_diff': logit_diff,
                'noun_logit_change': float(noun_change),
                'top1_token': top1_word,
            })
        
        results['tense_intervention'].append({
            'sentence': present,
            'verb_pres': verb_pres,
            'verb_past': verb_past,
            'interventions': intervention_results,
        })
        
        # 打印示例
        if len(results['tense_intervention']) <= 3:
            print(f"\n    [{present}]")
            for r in intervention_results:
                print(f"      α={r['alpha']:+5.1f}: pres={r['pres_verb_logit']:+7.3f}, "
                      f"past={r['past_verb_logit']:+7.3f}, diff={r['logit_diff']:+7.3f}, "
                      f"noun_Δ={r['noun_logit_change']:+7.3f}, top1={r['top1_token']}")
    
    # === Step 3: 极性干预(对比) ===
    if d_polarity is not None:
        print("\n    === 极性干预(对比) ===")
        polarity_alphas = [-2.0, -1.0, 0.0, 1.0, 2.0]
        
        for aff, neg, concept in ENGLISH_POLARITY_PAIRS[:10]:
            h_aff = _get_single_hidden(model, tokenizer, device, aff, target_layer)
            if h_aff is None:
                continue
            
            intervention_results = []
            for alpha in polarity_alphas:
                h_intervened = h_aff + alpha * d_polarity
                logits_intervened = h_intervened @ W_U.T
                
                # 检查否定词"not"的logit
                not_ids = tokenizer.encode("not", add_special_tokens=False)
                not_tok_id = not_ids[0] if not_ids else None
                
                not_logit = float(logits_intervened[not_tok_id]) if not_tok_id else 0.0
                
                intervention_results.append({
                    'alpha': float(alpha),
                    'not_logit': not_logit,
                })
            
            results['polarity_intervention'].append({
                'sentence': aff,
                'interventions': intervention_results,
            })
    
    # === Step 4: 汇总统计 ===
    print("\n  [L1.3] Logit Lens汇总统计")
    
    # 时态干预: α从-2到+2，logit_diff的变化
    tense_summary = _summarize_tense_intervention(results['tense_intervention'])
    results['tense_summary'] = tense_summary
    
    # 极性干预汇总
    if results['polarity_intervention']:
        polarity_summary = _summarize_polarity_intervention(results['polarity_intervention'])
        results['polarity_summary'] = polarity_summary
    
    # 语义控制检验
    semantic_control = _check_semantic_control(results['tense_intervention'])
    results['semantic_control_summary'] = semantic_control
    
    _release_model(model)
    return results


def _summarize_tense_intervention(tense_results):
    """汇总时态干预结果"""
    if not tense_results:
        return {}
    
    # 收集每个α下的logit_diff
    alpha_to_diffs = defaultdict(list)
    alpha_to_noun_changes = defaultdict(list)
    
    for item in tense_results:
        for iv in item['interventions']:
            alpha_to_diffs[iv['alpha']].append(iv['logit_diff'])
            alpha_to_noun_changes[iv['alpha']].append(iv['noun_logit_change'])
    
    summary = {
        'logit_diff_by_alpha': {},
        'noun_change_by_alpha': {},
    }
    
    for alpha in sorted(alpha_to_diffs.keys()):
        diffs = alpha_to_diffs[alpha]
        noun_changes = alpha_to_noun_changes[alpha]
        summary['logit_diff_by_alpha'][float(alpha)] = {
            'mean': float(np.mean(diffs)),
            'std': float(np.std(diffs)),
            'n': len(diffs),
        }
        summary['noun_change_by_alpha'][float(alpha)] = {
            'mean': float(np.mean(noun_changes)),
            'std': float(np.std(noun_changes)),
            'n': len(noun_changes),
        }
        
        print(f"    α={alpha:+5.1f}: logit_diff={np.mean(diffs):+7.3f}±{np.std(diffs):.3f}, "
              f"noun_Δ={np.mean(noun_changes):+7.3f}±{np.std(noun_changes):.3f}")
    
    # 关键指标: logit_diff的斜率(每单位α的变化)
    if -1.0 in alpha_to_diffs and 1.0 in alpha_to_diffs:
        diff_neg1 = np.mean(alpha_to_diffs[-1.0])
        diff_pos1 = np.mean(alpha_to_diffs[1.0])
        slope = (diff_pos1 - diff_neg1) / 2.0
        summary['logit_diff_slope'] = float(slope)
        print(f"\n    logit_diff斜率(每单位α): {slope:+.4f}")
    
    # 语义控制检验: noun_change是否显著<logit_diff
    if 1.0 in alpha_to_noun_changes and 1.0 in alpha_to_diffs:
        noun_mean = abs(np.mean(alpha_to_noun_changes[1.0]))
        tense_mean = abs(np.mean(alpha_to_diffs[1.0]))
        selectivity = tense_mean / max(noun_mean, 1e-6)
        summary['selectivity'] = float(selectivity)
        print(f"    选择性(时态logit变化/名词logit变化): {selectivity:.1f}x")
    
    return summary


def _summarize_polarity_intervention(polarity_results):
    """汇总极性干预结果"""
    if not polarity_results:
        return {}
    
    alpha_to_not = defaultdict(list)
    for item in polarity_results:
        for iv in item['interventions']:
            alpha_to_not[iv['alpha']].append(iv['not_logit'])
    
    summary = {}
    for alpha in sorted(alpha_to_not.keys()):
        not_logits = alpha_to_not[alpha]
        summary[float(alpha)] = {
            'mean_not_logit': float(np.mean(not_logits)),
            'std_not_logit': float(np.std(not_logits)),
        }
        print(f"    α={alpha:+5.1f}: not_logit={np.mean(not_logits):+7.3f}±{np.std(not_logits):.3f}")
    
    return summary


def _check_semantic_control(tense_results):
    """检查语义控制: 沿时态方向干预时，名词token的logit是否保持不变"""
    if not tense_results:
        return {}
    
    # α=0 vs α=2时的名词logit变化
    noun_changes_at_2 = []
    logit_diffs_at_2 = []
    
    for item in tense_results:
        for iv in item['interventions']:
            if abs(iv['alpha'] - 2.0) < 0.01:
                noun_changes_at_2.append(iv['noun_logit_change'])
                logit_diffs_at_2.append(iv['logit_diff'])
    
    if noun_changes_at_2:
        result = {
            'noun_change_at_alpha2': {
                'mean': float(np.mean(noun_changes_at_2)),
                'std': float(np.std(noun_changes_at_2)),
            },
            'logit_diff_at_alpha2': {
                'mean': float(np.mean(logit_diffs_at_2)),
                'std': float(np.std(logit_diffs_at_2)),
            },
            'verdict': '',
        }
        
        noun_mean = abs(np.mean(noun_changes_at_2))
        tense_mean = abs(np.mean(logit_diffs_at_2))
        
        if tense_mean > 5 * noun_mean:
            result['verdict'] = 'STRONG: 时态干预显著改变动词logit，几乎不影响名词logit'
        elif tense_mean > 2 * noun_mean:
            result['verdict'] = 'MODERATE: 时态干预改变动词logit，对名词logit影响较小'
        else:
            result['verdict'] = 'WEAK: 时态干预同时改变动词和名词logit'
        
        print(f"\n  语义控制检验(α=2.0):")
        print(f"    动词logit变化: {tense_mean:+.3f}")
        print(f"    名词logit变化: {noun_mean:+.3f}")
        print(f"    判定: {result['verdict']}")
        
        return result
    
    return {}


# ============================================================
# L2: 组合纤维验证
# ============================================================

def l2_combined_fiber(model_name):
    """
    L2: 验证时态+极性的复合变换是否走二维纤维平面
    
    纤维丛预测:
    - h(present, aff) = h_base + α_pres·d_tense + β_aff·d_polarity
    - h(past, neg)    = h_base + α_past·d_tense + β_neg·d_polarity
    - → 四种组合(pres/aff, pres/neg, past/aff, past/neg)应分布在2D平面上
    
    检验方法:
    1. 提取4种组合的隐藏状态
    2. 做PCA，检查前2个PC是否解释大部分方差
    3. 检查4个点是否近似在2D平面上
    """
    print("\n" + "="*70)
    print("L2: 组合纤维验证")
    print("="*70)
    
    model, tokenizer, device, d_model, n_layers, _ = _load_model_full(model_name)
    if model is None:
        return {}
    
    target_layer = n_layers // 2
    
    # 提取方向
    d_tense = _extract_functional_direction(model, tokenizer, device,
                                             ENGLISH_TENSE_PAIRS[:15], target_layer, 'tense')
    d_polarity = _extract_functional_direction(model, tokenizer, device,
                                                ENGLISH_POLARITY_PAIRS[:15], target_layer, 'polarity')
    
    if d_tense is None or d_polarity is None:
        print("  ERROR: 无法提取功能方向")
        _release_model(model)
        return {}
    
    # 检查d_tense和d_polarity的正交性
    ortho_cos = float(np.dot(d_tense, d_polarity))
    print(f"\n  d_tense · d_polarity = {ortho_cos:.6f}")
    if abs(ortho_cos) < 0.3:
        print("  → 时态和极性方向近似正交")
    else:
        print(f"  → 时态和极性方向有显著相关性(cos={ortho_cos:.3f})")
    
    results = {
        'd_tense_d_polarity_cos': ortho_cos,
        'per_concept': [],
    }
    
    # 对10个概念做组合测试
    for pres_aff, past_aff, pres_neg, past_neg, verb_p, verb_pa in COMBINED_PAIRS:
        # 获取4种组合的隐藏状态
        h_pa = _get_single_hidden(model, tokenizer, device, pres_aff, target_layer)
        h_pp = _get_single_hidden(model, tokenizer, device, past_aff, target_layer)
        h_na = _get_single_hidden(model, tokenizer, device, pres_neg, target_layer)
        h_np = _get_single_hidden(model, tokenizer, device, past_neg, target_layer)
        
        if any(x is None for x in [h_pa, h_pp, h_na, h_np]):
            continue
        
        # 4个向量堆叠
        H = np.stack([h_pa, h_pp, h_na, h_np])  # [4, d_model]
        
        # PCA: 检查前2个PC解释多少方差
        H_centered = H - H.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
        
        total_var = np.sum(S**2)
        var_explained = S**2 / total_var if total_var > 0 else np.zeros(len(S))
        cum_var_2 = float(np.sum(var_explained[:2]))
        cum_var_3 = float(np.sum(var_explained[:3])) if len(var_explained) > 2 else cum_var_2
        
        # 纤维平面性: 4个点到2D子空间的平均距离
        proj_2d = U[:, :2] @ np.diag(S[:2]) @ Vt[:2, :]
        residuals_2d = H_centered - proj_2d
        avg_residual_2d = float(np.mean(np.linalg.norm(residuals_2d, axis=1)))
        avg_norm = float(np.mean(np.linalg.norm(H_centered, axis=1)))
        planarity_ratio = avg_residual_2d / max(avg_norm, 1e-10)
        
        # 时态/极性方向的投影系数
        proj_tense = np.array([np.dot(h, d_tense) for h in [h_pa, h_pp, h_na, h_np]])
        proj_polarity = np.array([np.dot(h, d_polarity) for h in [h_pa, h_pp, h_na, h_np]])
        
        # 检查2D平面是否由d_tense和d_polarity张成
        # 用d_tense和d_polarity作为基底，看4个点在这个2D空间的表示
        D = np.stack([d_tense, d_polarity])  # [2, d_model]
        coeffs = H_centered @ D.T  # [4, 2]
        
        # 重建误差
        H_reconstructed = coeffs @ D  # [4, d_model]
        recon_error = float(np.mean(np.linalg.norm(H_centered - H_reconstructed, axis=1)))
        recon_ratio = recon_error / max(avg_norm, 1e-10)
        
        concept_result = {
            'verb': verb_p,
            'cum_var_2d': cum_var_2,
            'cum_var_3d': cum_var_3,
            'planarity_ratio': planarity_ratio,
            'recon_error_by_fiber_dirs': recon_ratio,
            'proj_tense': proj_tense.tolist(),
            'proj_polarity': proj_polarity.tolist(),
            'fiber_coeffs': coeffs.tolist(),
        }
        
        results['per_concept'].append(concept_result)
        print(f"    {verb_p}: 2D方差={cum_var_2:.4f}, 平面性={planarity_ratio:.4f}, "
              f"纤维重建误差={recon_ratio:.4f}")
    
    # 汇总
    if results['per_concept']:
        avg_var2 = np.mean([x['cum_var_2d'] for x in results['per_concept']])
        avg_planarity = np.mean([x['planarity_ratio'] for x in results['per_concept']])
        avg_recon = np.mean([x['recon_error_by_fiber_dirs'] for x in results['per_concept']])
        
        results['summary'] = {
            'avg_2d_variance': float(avg_var2),
            'avg_planarity_ratio': float(avg_planarity),
            'avg_fiber_recon_error': float(avg_recon),
            'd_tense_d_polarity_cos': ortho_cos,
        }
        
        print(f"\n  汇总: 2D方差={avg_var2:.4f}, 平面性={avg_planarity:.4f}, "
              f"纤维重建误差={avg_recon:.4f}")
        
        if avg_var2 > 0.99:
            print("  → 4种组合几乎完全在2D平面上(强纤维平面)")
        elif avg_var2 > 0.95:
            print("  → 4种组合主要在2D平面上(中等纤维平面)")
        else:
            print("  → 4种组合超出2D平面(纤维平面假说不成立)")
        
        if avg_recon < 0.1:
            print("  → d_tense+d_polarity几乎完全张成纤维平面")
        elif avg_recon < 0.3:
            print("  → d_tense+d_polarity部分张成纤维平面")
        else:
            print("  → d_tense+d_polarity不能张成纤维平面")
    
    _release_model(model)
    return results


# ============================================================
# L3: 欧氏距离测量
# ============================================================

def l3_euclidean_distance(model_name):
    """
    L3: 用欧氏距离代替余弦相似度，消除维度效应
    
    维度效应问题:
    - d_model=2560, 去掉1维后余弦变化Δcos ≈ 1/2560 ≈ 0.0004
    - 时态base_cos=0.9998 → 可能只是维度效应
    - 用欧氏距离可以更精确地测量"去掉功能方向后基空间变化了多少"
    
    核心测试:
    1. 时态变换: ||h_base_present - h_base_past|| / ||h_present - h_past||
    2. 语义替换: ||h_base_cat - h_base_dog|| / ||h_cat - h_dog||
    3. 如果时态的基空间距离比<<语义的基空间距离比 → 纤维结构真实
    """
    print("\n" + "="*70)
    print("L3: 欧氏距离测量")
    print("="*70)
    
    model, tokenizer, device, d_model, n_layers, _ = _load_model_full(model_name)
    if model is None:
        return {}
    
    target_layer = n_layers // 2
    
    # 提取方向
    d_tense = _extract_functional_direction(model, tokenizer, device,
                                             ENGLISH_TENSE_PAIRS[:15], target_layer, 'tense')
    d_polarity = _extract_functional_direction(model, tokenizer, device,
                                                ENGLISH_POLARITY_PAIRS[:15], target_layer, 'polarity')
    
    if d_tense is None:
        print("  ERROR: 无法提取时态方向")
        _release_model(model)
        return {}
    
    results = {}
    
    # --- 时态变换的欧氏距离 ---
    print("\n  [L3.1] 时态变换")
    tense_distances = []
    
    for present, past, _, _ in ENGLISH_TENSE_PAIRS:
        h1, h2 = _get_hidden_states(model, tokenizer, device, present, past, target_layer)
        if h1 is None or h2 is None:
            continue
        
        # 原始距离
        orig_dist = float(np.linalg.norm(h1 - h2))
        
        # 去掉d_tense后的距离
        proj1 = np.dot(h1, d_tense)
        proj2 = np.dot(h2, d_tense)
        h1_base = h1 - proj1 * d_tense
        h2_base = h2 - proj2 * d_tense
        base_dist = float(np.linalg.norm(h1_base - h2_base))
        
        # 基空间距离比
        base_ratio = base_dist / max(orig_dist, 1e-10)
        
        # d_tense解释的方差
        tense_shift = abs(proj2 - proj1)
        tense_energy = tense_shift / max(orig_dist, 1e-10)
        
        tense_distances.append({
            'sentence': present,
            'orig_dist': orig_dist,
            'base_dist': base_dist,
            'base_ratio': base_ratio,
            'tense_shift': float(tense_shift),
            'tense_energy': tense_energy,
        })
    
    if tense_distances:
        avg_base_ratio = np.mean([x['base_ratio'] for x in tense_distances])
        avg_tense_energy = np.mean([x['tense_energy'] for x in tense_distances])
        
        results['tense'] = {
            'samples': tense_distances,
            'avg_base_ratio': float(avg_base_ratio),
            'avg_tense_energy': float(avg_tense_energy),
        }
        
        print(f"    平均基空间距离比: {avg_base_ratio:.6f}")
        print(f"    平均时态能量比: {avg_tense_energy:.6f}")
        print(f"    → 时态方向解释了{(1-avg_base_ratio)*100:.1f}%的距离变化")
    
    # --- 极性变换的欧氏距离 ---
    print("\n  [L3.2] 极性变换")
    polarity_distances = []
    
    if d_polarity is not None:
        for aff, neg, concept in ENGLISH_POLARITY_PAIRS:
            h1, h2 = _get_hidden_states(model, tokenizer, device, aff, neg, target_layer)
            if h1 is None or h2 is None:
                continue
            
            orig_dist = float(np.linalg.norm(h1 - h2))
            
            proj1 = np.dot(h1, d_polarity)
            proj2 = np.dot(h2, d_polarity)
            h1_base = h1 - proj1 * d_polarity
            h2_base = h2 - proj2 * d_polarity
            base_dist = float(np.linalg.norm(h1_base - h2_base))
            
            base_ratio = base_dist / max(orig_dist, 1e-10)
            polarity_shift = abs(proj2 - proj1)
            polarity_energy = polarity_shift / max(orig_dist, 1e-10)
            
            polarity_distances.append({
                'concept': concept,
                'orig_dist': orig_dist,
                'base_dist': base_dist,
                'base_ratio': base_ratio,
                'polarity_shift': float(polarity_shift),
                'polarity_energy': polarity_energy,
            })
        
        if polarity_distances:
            avg_base_ratio = np.mean([x['base_ratio'] for x in polarity_distances])
            avg_pol_energy = np.mean([x['polarity_energy'] for x in polarity_distances])
            
            results['polarity'] = {
                'samples': polarity_distances,
                'avg_base_ratio': float(avg_base_ratio),
                'avg_polarity_energy': float(avg_pol_energy),
            }
            
            print(f"    平均基空间距离比: {avg_base_ratio:.6f}")
            print(f"    平均极性能量比: {avg_pol_energy:.6f}")
    
    # --- 语义替换的欧氏距离(对比基准) ---
    print("\n  [L3.3] 语义替换 (对比基准)")
    concepts = ["cat", "dog", "bird", "fish", "tree", "car", "book", "house", "river", "mountain"]
    semantic_distances = []
    
    for i, c1 in enumerate(concepts):
        for c2 in concepts[i+1:]:
            s1 = f"The {c1} walks every day"
            s2 = f"The {c2} walks every day"
            
            h1, h2 = _get_hidden_states(model, tokenizer, device, s1, s2, target_layer)
            if h1 is None or h2 is None:
                continue
            
            orig_dist = float(np.linalg.norm(h1 - h2))
            
            # 去掉时态+极性分量后的距离
            if d_tense is not None and d_polarity is not None:
                h1_base = h1 - np.dot(h1, d_tense) * d_tense - np.dot(h1, d_polarity) * d_polarity
                h2_base = h2 - np.dot(h2, d_tense) * d_tense - np.dot(h2, d_polarity) * d_polarity
            elif d_tense is not None:
                h1_base = h1 - np.dot(h1, d_tense) * d_tense
                h2_base = h2 - np.dot(h2, d_tense) * d_tense
            else:
                h1_base = h1.copy()
                h2_base = h2.copy()
            
            base_dist = float(np.linalg.norm(h1_base - h2_base))
            base_ratio = base_dist / max(orig_dist, 1e-10)
            
            semantic_distances.append({
                'pair': f"{c1}-{c2}",
                'orig_dist': orig_dist,
                'base_dist': base_dist,
                'base_ratio': base_ratio,
            })
    
    if semantic_distances:
        avg_base_ratio = np.mean([x['base_ratio'] for x in semantic_distances])
        
        results['semantic'] = {
            'samples': semantic_distances[:20],  # 保存前20个
            'avg_base_ratio': float(avg_base_ratio),
            'n_pairs': len(semantic_distances),
        }
        
        print(f"    平均基空间距离比: {avg_base_ratio:.6f}")
    
    # --- 汇总对比 ---
    print("\n  [L3.4] 欧氏距离对比")
    if 'tense' in results and 'semantic' in results:
        tense_ratio = results['tense']['avg_base_ratio']
        semantic_ratio = results['semantic']['avg_base_ratio']
        
        separation = semantic_ratio / max(tense_ratio, 1e-10)
        
        results['comparison'] = {
            'tense_base_ratio': tense_ratio,
            'semantic_base_ratio': semantic_ratio,
            'separation_factor': float(separation),
            'verdict': '',
        }
        
        print(f"    时态base_ratio: {tense_ratio:.6f}")
        print(f"    语义base_ratio: {semantic_ratio:.6f}")
        print(f"    分离因子(语义/时态): {separation:.1f}x")
        
        if separation > 10:
            print("    → 强纤维结构: 时态的基空间距离远小于语义")
            results['comparison']['verdict'] = 'STRONG_FIBER'
        elif separation > 3:
            print("    → 中等纤维结构")
            results['comparison']['verdict'] = 'MODERATE_FIBER'
        else:
            print("    → 弱/无纤维结构")
            results['comparison']['verdict'] = 'WEAK_FIBER'
    
    _release_model(model)
    return results


# ============================================================
# L4: 中文纤维验证
# ============================================================

def l4_chinese_fiber(model_name):
    """
    L4: 检查中文的时态/否定标记是否也有纤维结构
    
    核心问题:
    - 英文的时态是动词屈折(walk→walked)
    - 中文的时态用助词"了"(走→走了)
    - 如果纤维结构跨语言一致 → 语言编码的数学结构是普遍的
    - 如果不一致 → 纤维结构可能是英语特有的
    
    测试:
    1. 中文时态: "猫走路" vs "猫走了路"
    2. 中文极性: "猫在这里" vs "猫不在这里"
    3. 与英文结果对比
    """
    print("\n" + "="*70)
    print("L4: 中文纤维验证")
    print("="*70)
    
    model, tokenizer, device, d_model, n_layers, _ = _load_model_full(model_name)
    if model is None:
        return {}
    
    target_layer = n_layers // 2
    
    results = {
        'chinese_tense': {},
        'chinese_polarity': {},
    }
    
    # --- 中文时态纤维 ---
    print("\n  [L4.1] 中文时态纤维")
    
    # 提取中文时态方向
    zh_tense_diffs = []
    for s1, s2, _, _ in CHINESE_TENSE_PAIRS:
        h1, h2 = _get_hidden_states(model, tokenizer, device, s1, s2, target_layer)
        if h1 is None or h2 is None:
            continue
        diff = h2 - h1
        norm = np.linalg.norm(diff)
        if norm > 1e-8:
            zh_tense_diffs.append(diff / norm)
    
    if zh_tense_diffs:
        d_zh_tense = np.mean(zh_tense_diffs, axis=0)
        d_zh_tense = d_zh_tense / (np.linalg.norm(d_zh_tense) + 1e-30)
        
        # 纤维不变性测试
        zh_tense_invariance = []
        for s1, s2, _, _ in CHINESE_TENSE_PAIRS:
            h1, h2 = _get_hidden_states(model, tokenizer, device, s1, s2, target_layer)
            if h1 is None or h2 is None:
                continue
            
            # 原始距离
            orig_dist = float(np.linalg.norm(h1 - h2))
            
            # 去掉中文时态方向后的基空间距离
            proj1 = np.dot(h1, d_zh_tense)
            proj2 = np.dot(h2, d_zh_tense)
            h1_base = h1 - proj1 * d_zh_tense
            h2_base = h2 - proj2 * d_zh_tense
            base_dist = float(np.linalg.norm(h1_base - h2_base))
            base_ratio = base_dist / max(orig_dist, 1e-10)
            
            # 余弦相似度
            orig_cos = float(np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-30))
            base_cos = float(np.dot(h1_base, h2_base) / 
                           (np.linalg.norm(h1_base) * np.linalg.norm(h2_base) + 1e-30))
            
            zh_tense_invariance.append({
                'sentence': s1,
                'orig_dist': orig_dist,
                'base_dist': base_dist,
                'base_ratio': base_ratio,
                'orig_cos': orig_cos,
                'base_cos': base_cos,
            })
        
        if zh_tense_invariance:
            avg_base_ratio = np.mean([x['base_ratio'] for x in zh_tense_invariance])
            avg_base_cos = np.mean([x['base_cos'] for x in zh_tense_invariance])
            avg_orig_cos = np.mean([x['orig_cos'] for x in zh_tense_invariance])
            
            results['chinese_tense'] = {
                'samples': zh_tense_invariance,
                'avg_base_ratio': float(avg_base_ratio),
                'avg_base_cos': float(avg_base_cos),
                'avg_orig_cos': float(avg_orig_cos),
                'n_direction_samples': len(zh_tense_diffs),
            }
            
            print(f"    方向样本数: {len(zh_tense_diffs)}")
            print(f"    基空间距离比: {avg_base_ratio:.6f}")
            print(f"    基空间余弦: {avg_base_cos:.6f}")
            print(f"    原始余弦: {avg_orig_cos:.6f}")
            
            # 与英文对比
            en_tense_diffs = []
            for s1, s2, _, _ in ENGLISH_TENSE_PAIRS[:15]:
                h1, h2 = _get_hidden_states(model, tokenizer, device, s1, s2, target_layer)
                if h1 is None or h2 is None:
                    continue
                diff = h2 - h1
                norm = np.linalg.norm(diff)
                if norm > 1e-8:
                    en_tense_diffs.append(diff / norm)
            
            if en_tense_diffs and zh_tense_diffs:
                d_en_tense = np.mean(en_tense_diffs, axis=0)
                d_en_tense = d_en_tense / (np.linalg.norm(d_en_tense) + 1e-30)
                
                # 中英文时态方向的相关性
                cross_lang_cos = float(np.dot(d_zh_tense, d_en_tense))
                
                results['chinese_tense']['cross_language_cos'] = float(cross_lang_cos)
                print(f"\n    中英文时态方向cos: {cross_lang_cos:.6f}")
                
                if abs(cross_lang_cos) > 0.5:
                    print("    → 中英文时态方向有显著相关性(纤维跨语言共享)")
                elif abs(cross_lang_cos) > 0.2:
                    print("    → 中英文时态方向有弱相关性")
                else:
                    print("    → 中英文时态方向不相关(语言特异)")
    else:
        print("    WARNING: 无法提取中文时态方向(可能tokenization问题)")
    
    # --- 中文极性纤维 ---
    print("\n  [L4.2] 中文极性纤维")
    
    zh_polarity_diffs = []
    for aff, neg, concept in CHINESE_POLARITY_PAIRS:
        h1, h2 = _get_hidden_states(model, tokenizer, device, aff, neg, target_layer)
        if h1 is None or h2 is None:
            continue
        diff = h2 - h1
        norm = np.linalg.norm(diff)
        if norm > 1e-8:
            zh_polarity_diffs.append(diff / norm)
    
    if zh_polarity_diffs:
        d_zh_polarity = np.mean(zh_polarity_diffs, axis=0)
        d_zh_polarity = d_zh_polarity / (np.linalg.norm(d_zh_polarity) + 1e-30)
        
        # 纤维不变性
        zh_pol_invariance = []
        for aff, neg, concept in CHINESE_POLARITY_PAIRS:
            h1, h2 = _get_hidden_states(model, tokenizer, device, aff, neg, target_layer)
            if h1 is None or h2 is None:
                continue
            
            orig_dist = float(np.linalg.norm(h1 - h2))
            proj1 = np.dot(h1, d_zh_polarity)
            proj2 = np.dot(h2, d_zh_polarity)
            h1_base = h1 - proj1 * d_zh_polarity
            h2_base = h2 - proj2 * d_zh_polarity
            base_dist = float(np.linalg.norm(h1_base - h2_base))
            base_ratio = base_dist / max(orig_dist, 1e-10)
            
            base_cos = float(np.dot(h1_base, h2_base) / 
                           (np.linalg.norm(h1_base) * np.linalg.norm(h2_base) + 1e-30))
            
            zh_pol_invariance.append({
                'concept': concept,
                'base_ratio': base_ratio,
                'base_cos': base_cos,
            })
        
        if zh_pol_invariance:
            avg_base_ratio = np.mean([x['base_ratio'] for x in zh_pol_invariance])
            avg_base_cos = np.mean([x['base_cos'] for x in zh_pol_invariance])
            
            results['chinese_polarity'] = {
                'samples': zh_pol_invariance,
                'avg_base_ratio': float(avg_base_ratio),
                'avg_base_cos': float(avg_base_cos),
                'n_direction_samples': len(zh_polarity_diffs),
            }
            
            print(f"    方向样本数: {len(zh_polarity_diffs)}")
            print(f"    基空间距离比: {avg_base_ratio:.6f}")
            print(f"    基空间余弦: {avg_base_cos:.6f}")
            
            # 与英文对比
            d_polarity_en = _extract_functional_direction(
                model, tokenizer, device, ENGLISH_POLARITY_PAIRS[:15], target_layer, 'polarity'
            )
            if d_polarity_en is not None:
                cross_lang_cos = float(np.dot(d_zh_polarity, d_polarity_en))
                results['chinese_polarity']['cross_language_cos'] = float(cross_lang_cos)
                print(f"    中英文极性方向cos: {cross_lang_cos:.6f}")
    else:
        print("    WARNING: 无法提取中文极性方向")
    
    # --- 跨语言汇总 ---
    print("\n  [L4.3] 跨语言纤维结构对比")
    
    if results['chinese_tense'] and results.get('chinese_polarity'):
        zh_tense_br = results['chinese_tense'].get('avg_base_ratio', 1.0)
        zh_pol_br = results['chinese_polarity'].get('avg_base_ratio', 1.0)
        
        print(f"    中文时态 base_ratio: {zh_tense_br:.6f}")
        print(f"    中文极性 base_ratio: {zh_pol_br:.6f}")
        
        if zh_tense_br < 0.3:
            print("    → 中文时态有纤维结构(基空间距离比<30%)")
        else:
            print("    → 中文时态纤维结构弱(基空间距离比>30%)")
        
        if zh_pol_br < 0.5:
            print("    → 中文极性有纤维结构")
        else:
            print("    → 中文极性纤维结构弱")
    
    _release_model(model)
    return results


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
    
    # 获取lm_head权重
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


def _get_single_hidden(model, tokenizer, device, text, layer_idx):
    """获取单个句子的隐藏状态"""
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
    """
    diffs = []
    
    for pair in pairs:
        if direction_type == 'tense':
            s1, s2 = pair[0], pair[1]
        else:  # polarity
            s1, s2 = pair[0], pair[1]
        
        h1, h2 = _get_hidden_states(model, tokenizer, device, s1, s2, layer_idx)
        if h1 is None or h2 is None:
            continue
        
        diff = h2 - h1
        norm = np.linalg.norm(diff)
        if norm > 1e-8:
            diffs.append(diff / norm)
    
    if not diffs:
        return None
    
    d = np.mean(diffs, axis=0)
    d = d / (np.linalg.norm(d) + 1e-30)
    
    # 计算内部一致性
    if len(diffs) > 1:
        cosines = [float(np.dot(d, dd)) for dd in diffs]
        avg_cos = float(np.mean(cosines))
        print(f"    {direction_type}方向: {len(diffs)}样本, 内部一致性cos={avg_cos:.4f}")
    
    return d


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
    parser = argparse.ArgumentParser(description="Causal Logit Lens Validation")
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--test", type=str, default="all",
                        choices=["l1", "l2", "l3", "l4", "all"],
                        help="运行哪个测试")
    args = parser.parse_args()
    
    model_name = args.model
    print(f"\n{'#'*70}")
    print(f"# Phase CLXXXII: 因果解码验证")
    print(f"# Model: {model_name}")
    print(f"{'#'*70}")
    
    all_results = {
        'model': model_name,
        'timestamp': datetime.now().isoformat(),
    }
    
    output_dir = Path(r"d:\Ai2050\TransformerLens-Project\results\causal_logit")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # L1: Logit Lens因果干预
    if args.test in ["l1", "all"]:
        t0 = time.time()
        l1_results = l1_logit_lens_intervention(model_name)
        all_results['l1_logit_lens'] = l1_results
        print(f"\n  L1耗时: {time.time()-t0:.1f}s")
    
    # L2: 组合纤维验证
    if args.test in ["l2", "all"]:
        t0 = time.time()
        l2_results = l2_combined_fiber(model_name)
        all_results['l2_combined_fiber'] = l2_results
        print(f"\n  L2耗时: {time.time()-t0:.1f}s")
    
    # L3: 欧氏距离测量
    if args.test in ["l3", "all"]:
        t0 = time.time()
        l3_results = l3_euclidean_distance(model_name)
        all_results['l3_euclidean'] = l3_results
        print(f"\n  L3耗时: {time.time()-t0:.1f}s")
    
    # L4: 中文纤维验证
    if args.test in ["l4", "all"]:
        t0 = time.time()
        l4_results = l4_chinese_fiber(model_name)
        all_results['l4_chinese'] = l4_results
        print(f"\n  L4耗时: {time.time()-t0:.1f}s")
    
    # 保存结果
    output_path = output_dir / f"{model_name}_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"所有结果已保存到 {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
