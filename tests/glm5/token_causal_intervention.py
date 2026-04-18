"""
Phase CLXXXV: Token位置特异性因果干预 — 突破Mean-Pooling瓶颈
============================================================

核心问题: 之前所有logit lens干预都使用mean-pooled隐藏状态,
极性信息在"not"这个特定token位置上, mean-pooling把它混合掉了。

解决方案:
1. 用hook在特定层特定位置做真正的因果干预(forward pass)
2. 看最终输出(next token prediction)的logit变化
3. 对比不同层/不同位置的干预效果

关键改进: 不用logit lens! 直接看模型最终输出的logit!

测试:
T1: Hook-based极性因果干预 — 在中间层注入方向, 看最终输出
T2: 逐层干预扫描 — 哪些层的干预最有效
T3: 逐位置干预扫描 — 在否定句的哪个位置注入最有效
T4: Activation Patching — 用hook将否定句的残差修补到肯定句

运行方式:
  python tests/glm5/token_causal_intervention.py --model qwen3 --test t1
  python tests/glm5/token_causal_intervention.py --model qwen3 --test t2
  python tests/glm5/token_causal_intervention.py --model qwen3 --test t3
  python tests/glm5/token_causal_intervention.py --model qwen3 --test t4
  python tests/glm5/token_causal_intervention.py --model qwen3 --test all
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
# 测试集
# ============================================================

POLARITY_PAIRS = [
    ("The cat is here", "The cat is not here", "not"),
    ("The dog is happy", "The dog is not happy", "not"),
    ("The book was found", "The book was not found", "not"),
    ("I like the car", "I do not like the car", "not"),
    ("She knows the answer", "She does not know the answer", "not"),
    ("The house is big", "The house is not big", "not"),
    ("The river flows north", "The river does not flow north", "not"),
    ("He can swim", "He cannot swim", "not"),
    ("The bird will come", "The bird will not come", "not"),
    ("The door was closed", "The door was not closed", "not"),
    ("The phone is working", "The phone is not working", "not"),
    ("The flower has bloomed", "The flower has not bloomed", "not"),
    ("I understand the plan", "I do not understand the plan", "not"),
    ("She likes the movie", "She does not like the movie", "not"),
    ("The bridge is safe", "The bridge is not safe", "not"),
    ("The child was playing", "The child was not playing", "not"),
    ("The star is visible", "The star is not visible", "not"),
    ("The cloud disappeared", "The cloud did not disappear", "not"),
    ("The key works well", "The key does not work well", "not"),
    ("The table holds weight", "The table does not hold weight", "not"),
    ("The food tastes good", "The food does not taste good", "not"),
    ("The light shines bright", "The light does not shine bright", "not"),
    ("The wind blows cold", "The wind does not blow cold", "not"),
    ("The rain falls down", "The rain does not fall down", "not"),
    ("The sun rises early", "The sun does not rise early", "not"),
]

TENSE_PAIRS = [
    ("The cat walks every day", "The cat walked yesterday", "walk", "walked"),
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
    ("The man gives gifts", "The man gave gifts", "give", "gave"),
    ("She sings a song", "She sang a song", "sing", "sang"),
    ("The king fights well", "The king fought well", "fight", "fought"),
    ("The boy knows the way", "The boy knew the way", "know", "knew"),
    ("She speaks softly", "She spoke softly", "speak", "spoke"),
    ("The wheel breaks down", "The wheel broke down", "break", "broke"),
    ("He chooses wisely", "He chose wisely", "choose", "chose"),
    ("The water freezes", "The water froze", "freeze", "froze"),
    ("She rides the horse", "She rode the horse", "ride", "rode"),
    ("He draws a picture", "He drew a picture", "draw", "drew"),
    ("She wears a dress", "She wore a dress", "wear", "wore"),
    ("The candle burns out", "The candle burned out", "burn", "burned"),
    ("The flower blooms late", "The flower bloomed late", "bloom", "bloomed"),
]


# ============================================================
# 模型加载与辅助函数
# ============================================================

def _load_model(model_name):
    if model_name in ["glm4", "deepseek7b"]:
        return _load_model_4bit(model_name)
    else:
        from model_utils import load_model
        return load_model(model_name)


def _load_model_4bit(model_name):
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
    print(f"  [{model_name}] 4-bit loaded, device={device}")
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


def _find_token_position(tokenizer, text, target_token):
    """通过解码每个token来找目标token的位置"""
    toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = toks.input_ids[0].tolist()
    
    positions = []
    for i, tid in enumerate(input_ids):
        decoded = tokenizer.decode([tid]).strip().lower()
        if decoded == target_token.lower():
            positions.append((i, tid))
    
    return positions


def _get_not_tok_id(tokenizer):
    """获取'not'在句子中的实际token ID"""
    test_ids = tokenizer.encode("The cat is not here", add_special_tokens=False)
    for tid in test_ids:
        if tokenizer.decode([tid]).strip().lower() == "not":
            return tid
    # fallback
    ids = tokenizer.encode(" not", add_special_tokens=False)
    return ids[0] if ids else None


def _extract_polarity_direction(model, tokenizer, device, pairs, layer_idx):
    """用mean-pooled方法提取极性方向"""
    diffs = []
    for aff, neg, _ in pairs:
        toks1 = tokenizer(aff, return_tensors="pt", truncation=True, max_length=64).to(device)
        toks2 = tokenizer(neg, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            h1 = model(**toks1, output_hidden_states=True).hidden_states[layer_idx]
            h2 = model(**toks2, output_hidden_states=True).hidden_states[layer_idx]
        r1 = h1[0].mean(0).detach().cpu().float().numpy()
        r2 = h2[0].mean(0).detach().cpu().float().numpy()
        diffs.append(r2 - r1)
    
    if not diffs:
        return None, []
    diffs = np.array(diffs)
    d_mean = diffs.mean(axis=0)
    d_norm = np.linalg.norm(d_mean)
    if d_norm < 1e-10:
        return None, []
    d_direction = d_mean / d_norm
    
    cos_list = []
    for d in diffs:
        d_n = np.linalg.norm(d)
        if d_n > 1e-10:
            cos_list.append(float(np.dot(d, d_direction) / d_n))
    
    return d_direction, cos_list


def _get_final_logits(model, tokenizer, device, text, target_tok_id=None):
    """获取模型最终输出的logits"""
    toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
    with torch.no_grad():
        output = model(**toks)
    logits = output.logits[0, -1, :].detach().cpu().float().numpy()
    target_logit = float(logits[target_tok_id]) if target_tok_id is not None else None
    return logits, target_logit


def _hook_intervene_and_get_logits(model, tokenizer, device, text, layers, 
                                     target_layer, target_pos, direction, alpha,
                                     target_tok_id=None):
    """在指定层指定位置用hook注入方向, 返回最终输出的logits"""
    direction_t = torch.tensor(direction, dtype=torch.bfloat16, device=device)
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0].clone()
            h[0, target_pos, :] += alpha * direction_t
            return (h,) + output[1:]
        else:
            h = output.clone()
            h[0, target_pos, :] += alpha * direction_t
            return h
    
    hook_handle = layers[target_layer].register_forward_hook(hook_fn)
    
    toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
    with torch.no_grad():
        output = model(**toks)
    hook_handle.remove()
    
    logits = output.logits[0, -1, :].detach().cpu().float().numpy()
    target_logit = float(logits[target_tok_id]) if target_tok_id is not None else None
    return logits, target_logit


def _hook_patch_and_get_logits(model, tokenizer, device, 
                                source_text, target_text, layers,
                                target_layer, patch_pos,
                                target_tok_id=None):
    """将source_text在patch_pos的残差修补到target_text, 返回最终logits
    
    即: 运行target_text的forward, 但在target_layer的patch_pos位置
    用source_text的残差替换
    """
    # 先获取source在target_layer的隐藏状态
    src_toks = tokenizer(source_text, return_tensors="pt", truncation=True, max_length=64).to(device)
    with torch.no_grad():
        src_output = model(**src_toks, output_hidden_states=True)
    src_hidden = src_output.hidden_states[target_layer][0, patch_pos, :].detach().clone()
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0].clone()
            h[0, patch_pos, :] = src_hidden.to(h.dtype)
            return (h,) + output[1:]
        else:
            h = output.clone()
            h[0, patch_pos, :] = src_hidden.to(h.dtype)
            return h
    
    hook_handle = layers[target_layer].register_forward_hook(hook_fn)
    
    tgt_toks = tokenizer(target_text, return_tensors="pt", truncation=True, max_length=64).to(device)
    with torch.no_grad():
        output = model(**tgt_toks)
    hook_handle.remove()
    
    logits = output.logits[0, -1, :].detach().cpu().float().numpy()
    target_logit = float(logits[target_tok_id]) if target_tok_id is not None else None
    return logits, target_logit


# ============================================================
# T1: Hook-based极性因果干预 — 核心测试
# ============================================================

def test_t1(model_name, model, tokenizer, device, d_model, n_layers, W_U):
    """T1: 极性因果干预 — 用hook在中间层注入方向, 看最终输出"""
    
    print("\n" + "=" * 70)
    print("T1: Hook-based极性因果干预")
    print("=" * 70)
    
    target_layer = n_layers // 2
    layers = _get_layers(model)
    not_tok_id = _get_not_tok_id(tokenizer)
    print(f"  目标层: L{target_layer}/{n_layers}, 'not' tok_id={not_tok_id}")
    
    # 提取极性方向
    print("\n  [T1.1] 提取极性方向...")
    d_polarity, cos_polarity = _extract_polarity_direction(
        model, tokenizer, device, POLARITY_PAIRS, target_layer)
    
    if d_polarity is None:
        print("  ERROR: 无法提取方向")
        return {}
    
    print(f"  极性方向: cos_mean={np.mean(cos_polarity):.4f}, n={len(cos_polarity)}")
    
    # 测试不同位置上的干预
    print("\n  [T1.2] 不同位置的hook干预效果...")
    
    alphas = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    
    results = {
        'target_layer': target_layer,
        'cos_direction': float(np.mean(cos_polarity)),
        'not_tok_id': not_tok_id,
        'by_sentence': [],
    }
    
    for aff, neg, _ in POLARITY_PAIRS[:15]:
        # 找"not"在否定句中的位置
        not_positions = _find_token_position(tokenizer, neg, "not")
        if not not_positions:
            continue
        not_pos = not_positions[0][0]
        
        # 基线: 肯定句和否定句的"not" logit
        _, not_logit_aff = _get_final_logits(model, tokenizer, device, aff, not_tok_id)
        _, not_logit_neg = _get_final_logits(model, tokenizer, device, neg, not_tok_id)
        
        # 在肯定句中, 不同位置注入极性方向
        # 尝试: 在not_pos位置注入(对应否定句的not位置)
        # 以及: 在最后位置注入
        aff_seq_len = tokenizer(aff, return_tensors="pt").input_ids.shape[1]
        
        interventions = []
        for alpha in alphas:
            # 在not_pos位置注入
            _, not_logit_notpos = _hook_intervene_and_get_logits(
                model, tokenizer, device, aff, layers, 
                target_layer, not_pos, d_polarity, alpha, not_tok_id)
            
            # 在最后位置注入
            last_pos = aff_seq_len - 1
            _, not_logit_lastpos = _hook_intervene_and_get_logits(
                model, tokenizer, device, aff, layers,
                target_layer, last_pos, d_polarity, alpha, not_tok_id)
            
            interventions.append({
                'alpha': float(alpha),
                'not_pos_logit': not_logit_notpos,
                'last_pos_logit': not_logit_lastpos,
            })
        
        results['by_sentence'].append({
            'affirmative': aff,
            'negative': neg,
            'not_pos': not_pos,
            'aff_seq_len': aff_seq_len,
            'not_logit_aff': not_logit_aff,
            'not_logit_neg': not_logit_neg,
            'interventions': interventions,
        })
        
        if len(results['by_sentence']) <= 3:
            print(f"\n    [{aff}]")
            print(f"      Baseline: aff_not={not_logit_aff:+.3f}, neg_not={not_logit_neg:+.3f}")
            for iv in interventions:
                print(f"      α={iv['alpha']:+5.1f}: "
                      f"not_pos_not={iv['not_pos_logit']:+7.3f}, "
                      f"last_pos_not={iv['last_pos_logit']:+7.3f}")
    
    # 汇总
    print("\n  [T1.3] 汇总结果")
    
    alpha_to_notpos = defaultdict(list)
    alpha_to_lastpos = defaultdict(list)
    
    for item in results['by_sentence']:
        for iv in item['interventions']:
            alpha_to_notpos[iv['alpha']].append(iv['not_pos_logit'])
            alpha_to_lastpos[iv['alpha']].append(iv['last_pos_logit'])
    
    print("\n  α      | not_pos干预 'not' logit | last_pos干预 'not' logit")
    print("  " + "-" * 65)
    for alpha in sorted(alpha_to_notpos.keys()):
        np_mean = np.mean(alpha_to_notpos[alpha])
        lp_mean = np.mean(alpha_to_lastpos[alpha])
        print(f"  {alpha:+5.1f}  |  {np_mean:+8.3f} ± {np.std(alpha_to_notpos[alpha]):.3f}"
              f"  |  {lp_mean:+8.3f} ± {np.std(alpha_to_lastpos[alpha]):.3f}")
    
    # 斜率
    if -1.0 in alpha_to_notpos and 1.0 in alpha_to_notpos:
        notpos_slope = (np.mean(alpha_to_notpos[1.0]) - np.mean(alpha_to_notpos[-1.0])) / 2.0
        lastpos_slope = (np.mean(alpha_to_lastpos[1.0]) - np.mean(alpha_to_lastpos[-1.0])) / 2.0
        
        print(f"\n  ★ not_pos干预斜率:  {notpos_slope:+.4f}")
        print(f"  ★ last_pos干预斜率: {lastpos_slope:+.4f}")
        
        results['notpos_slope'] = float(notpos_slope)
        results['lastpos_slope'] = float(lastpos_slope)
    
    # 基线
    aff_logits = [item['not_logit_aff'] for item in results['by_sentence']]
    neg_logits = [item['not_logit_neg'] for item in results['by_sentence']]
    print(f"\n  基线: 肯定句'not' logit={np.mean(aff_logits):+.3f}, "
          f"否定句={np.mean(neg_logits):+.3f}, "
          f"差异={np.mean(neg_logits)-np.mean(aff_logits):+.3f}")
    
    results['aff_not_mean'] = float(np.mean(aff_logits))
    results['neg_not_mean'] = float(np.mean(neg_logits))
    
    return results


# ============================================================
# T2: 逐层干预扫描
# ============================================================

def test_t2(model_name, model, tokenizer, device, d_model, n_layers, W_U):
    """T2: 逐层干预扫描 — 哪些层的极性方向干预最有效"""
    
    print("\n" + "=" * 70)
    print("T2: 逐层极性方向干预扫描")
    print("=" * 70)
    
    layers = _get_layers(model)
    not_tok_id = _get_not_tok_id(tokenizer)
    
    # 采样层
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    results = {
        'sample_layers': sample_layers,
        'layer_results': [],
    }
    
    alpha = 3.0  # 用较大的alpha看效果
    
    for li in sample_layers:
        print(f"\n  [T2] Layer {li}/{n_layers-1}...")
        
        # 提取该层的极性方向
        d_polarity, cos_pol = _extract_polarity_direction(
            model, tokenizer, device, POLARITY_PAIRS, li)
        
        if d_polarity is None:
            print(f"    跳过 (无法提取方向)")
            continue
        
        print(f"    极性方向cos_mean={np.mean(cos_pol):.4f}")
        
        # 在肯定句的最后位置注入, 看"not" logit变化
        not_logit_changes = []
        
        for aff, neg, _ in POLARITY_PAIRS[:10]:
            aff_toks = tokenizer(aff, return_tensors="pt", truncation=True, max_length=64).to(device)
            last_pos = aff_toks.input_ids.shape[1] - 1
            
            # 基线
            _, not_logit_base = _get_final_logits(model, tokenizer, device, aff, not_tok_id)
            
            # 干预
            _, not_logit_intervened = _hook_intervene_and_get_logits(
                model, tokenizer, device, aff, layers,
                li, last_pos, d_polarity, alpha, not_tok_id)
            
            change = not_logit_intervened - not_logit_base
            not_logit_changes.append(change)
        
        mean_change = np.mean(not_logit_changes)
        std_change = np.std(not_logit_changes)
        print(f"    α={alpha:.1f} at last_pos: 'not' logit变化={mean_change:+.4f} ± {std_change:.4f}")
        
        results['layer_results'].append({
            'layer': li,
            'cos_polarity': float(np.mean(cos_pol)),
            'mean_not_logit_change': float(mean_change),
            'std_not_logit_change': float(std_change),
            'n': len(not_logit_changes),
        })
    
    # 找最有效的层
    print("\n  [T2] 逐层效果汇总:")
    for lr in results['layer_results']:
        print(f"    Layer {lr['layer']:2d}: cos={lr['cos_polarity']:.4f}, "
              f"'not' Δ={lr['mean_not_logit_change']:+.4f}")
    
    best_layer = max(results['layer_results'], key=lambda x: abs(x['mean_not_logit_change']))
    print(f"\n  ★ 最有效的层: L{best_layer['layer']} "
          f"(Δ={best_layer['mean_not_logit_change']:+.4f})")
    results['best_layer'] = best_layer
    
    return results


# ============================================================
# T3: 逐位置干预扫描
# ============================================================

def test_t3(model_name, model, tokenizer, device, d_model, n_layers, W_U):
    """T3: 逐位置干预扫描 — 在否定句的哪个位置注入极性方向最有效"""
    
    print("\n" + "=" * 70)
    print("T3: 逐位置干预扫描")
    print("=" * 70)
    
    target_layer = n_layers // 2
    layers = _get_layers(model)
    not_tok_id = _get_not_tok_id(tokenizer)
    
    # 提取极性方向
    d_polarity, cos_pol = _extract_polarity_direction(
        model, tokenizer, device, POLARITY_PAIRS, target_layer)
    
    if d_polarity is None:
        print("  ERROR: 无法提取方向")
        return {}
    
    print(f"  目标层: L{target_layer}, 极性方向cos={np.mean(cos_pol):.4f}")
    
    alpha = 3.0
    results = {
        'target_layer': target_layer,
        'alpha': alpha,
        'by_sentence': [],
    }
    
    for aff, neg, _ in POLARITY_PAIRS[:10]:
        aff_toks = tokenizer(aff, return_tensors="pt", truncation=True, max_length=64).to(device)
        aff_seq_len = aff_toks.input_ids.shape[1]
        
        # 基线
        _, not_logit_base = _get_final_logits(model, tokenizer, device, aff, not_tok_id)
        
        # 在肯定句的每个位置注入
        position_effects = []
        for pos in range(aff_seq_len):
            _, not_logit_intervened = _hook_intervene_and_get_logits(
                model, tokenizer, device, aff, layers,
                target_layer, pos, d_polarity, alpha, not_tok_id)
            
            change = not_logit_intervened - not_logit_base
            
            # 解码该位置的token
            tok_id = aff_toks.input_ids[0, pos].item()
            tok_word = tokenizer.decode([tok_id]).strip()[:10]
            
            position_effects.append({
                'pos': pos,
                'token': tok_word,
                'not_logit_change': change,
            })
        
        # 找最大效果的位置
        max_effect = max(position_effects, key=lambda x: abs(x['not_logit_change']))
        
        # 找not在否定句中的位置
        not_positions = _find_token_position(tokenizer, neg, "not")
        not_pos = not_positions[0][0] if not_positions else -1
        
        results['by_sentence'].append({
            'affirmative': aff,
            'negative': neg,
            'not_pos_in_neg': not_pos,
            'not_logit_base': not_logit_base,
            'position_effects': position_effects,
            'max_effect_pos': max_effect['pos'],
            'max_effect_change': max_effect['not_logit_change'],
        })
        
        print(f"\n    [{aff}]")
        print(f"      not_pos_in_neg={not_pos}")
        print(f"      max_effect at pos {max_effect['pos']} "
              f"({max_effect['token']}): Δ={max_effect['not_logit_change']:+.4f}")
        # 打印所有位置的效果
        for pe in position_effects:
            if abs(pe['not_logit_change']) > 0.01:
                print(f"        pos {pe['pos']:2d} ({pe['token']:8s}): "
                      f"Δ={pe['not_logit_change']:+.4f}")
    
    # 汇总: 哪个位置最常出现最大效果
    pos_to_max_count = defaultdict(int)
    for item in results['by_sentence']:
        pos_to_max_count[item['max_effect_pos']] += 1
    
    print(f"\n  [T3] 最大效果位置统计:")
    for pos, count in sorted(pos_to_max_count.items(), key=lambda x: -x[1]):
        print(f"    Pos {pos}: {count}次")
    
    return results


# ============================================================
# T4: Activation Patching — 真正的hook-based修补
# ============================================================

def test_t4(model_name, model, tokenizer, device, d_model, n_layers, W_U):
    """T4: Activation Patching — 将否定句的残差修补到肯定句"""
    
    print("\n" + "=" * 70)
    print("T4: Activation Patching (Hook-based)")
    print("=" * 70)
    
    layers = _get_layers(model)
    not_tok_id = _get_not_tok_id(tokenizer)
    
    # 采样层
    sample_layers = list(range(0, n_layers, max(1, n_layers // 6))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    results = {
        'sample_layers': sample_layers,
        'layer_results': [],
    }
    
    for li in sample_layers:
        print(f"\n  [T4] Layer {li}/{n_layers-1}...")
        
        patch_effects = []
        
        for aff, neg, _ in POLARITY_PAIRS[:8]:
            # 找"not"在否定句中的位置
            not_positions = _find_token_position(tokenizer, neg, "not")
            if not not_positions:
                continue
            not_pos = not_positions[0][0]
            
            # 肯定句的序列长度
            aff_toks = tokenizer(aff, return_tensors="pt", truncation=True, max_length=64).to(device)
            aff_seq_len = aff_toks.input_ids.shape[1]
            neg_toks = tokenizer(neg, return_tensors="pt", truncation=True, max_length=64).to(device)
            neg_seq_len = neg_toks.input_ids.shape[1]
            
            # 基线: 肯定句和否定句的"not" logit
            _, not_logit_aff = _get_final_logits(model, tokenizer, device, aff, not_tok_id)
            _, not_logit_neg = _get_final_logits(model, tokenizer, device, neg, not_tok_id)
            
            # 在肯定句中, 逐位置用否定句的残差修补
            min_len = min(aff_seq_len, neg_seq_len)
            
            position_impacts = []
            for patch_pos in range(min_len):
                # 将neg在patch_pos的残差修补到aff
                _, not_logit_patched = _hook_patch_and_get_logits(
                    model, tokenizer, device,
                    neg, aff, layers,
                    li, patch_pos, not_tok_id)
                
                impact = not_logit_patched - not_logit_aff
                
                # 解码token
                tok_id = aff_toks.input_ids[0, patch_pos].item()
                tok_word = tokenizer.decode([tok_id]).strip()[:10]
                
                position_impacts.append({
                    'pos': patch_pos,
                    'token': tok_word,
                    'not_logit_patched': not_logit_patched,
                    'impact': impact,
                })
            
            # 找最大影响
            max_impact = max(position_impacts, key=lambda x: abs(x['impact']))
            
            patch_effects.append({
                'affirmative': aff,
                'negative': neg,
                'not_pos_in_neg': not_pos,
                'not_logit_aff': not_logit_aff,
                'not_logit_neg': not_logit_neg,
                'max_impact_pos': max_impact['pos'],
                'max_impact_value': max_impact['impact'],
                'max_impact_token': max_impact['token'],
                'position_impacts': position_impacts,
            })
        
        if not patch_effects:
            continue
        
        # 汇总该层
        max_impacts = [item['max_impact_value'] for item in patch_effects]
        mean_max_impact = np.mean(max_impacts)
        
        print(f"    样本数: {len(patch_effects)}")
        print(f"    最大修补效果均值: {mean_max_impact:+.4f}")
        
        # 打印每个样本
        for item in patch_effects[:3]:
            print(f"      [{item['affirmative'][:30]}] "
                  f"max at pos {item['max_impact_pos']} ({item['max_impact_token']}): "
                  f"Δ={item['max_impact_value']:+.4f}")
        
        results['layer_results'].append({
            'layer': li,
            'mean_max_impact': float(mean_max_impact),
            'n': len(patch_effects),
            'details': patch_effects,
        })
    
    # 汇总
    print("\n  [T4] 逐层Activation Patching汇总:")
    for lr in results['layer_results']:
        print(f"    Layer {lr['layer']:2d}: mean_max_impact={lr['mean_max_impact']:+.4f}")
    
    return results


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase CLXXXV: Token位置特异性因果干预")
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--test", type=str, required=True,
                       choices=["t1", "t2", "t3", "t4", "all"])
    args = parser.parse_args()
    
    model_name = args.model
    test_name = args.test
    
    print(f"\n{'='*70}")
    print(f"Phase CLXXXV: Token位置特异性因果干预")
    print(f"Model: {model_name}, Test: {test_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    # 加载模型
    print(f"\n  Loading {model_name}...")
    model, tokenizer, device = _load_model(model_name)
    if model is None:
        print("  ERROR: Model loading failed")
        return
    
    d_model = model.config.hidden_size
    n_layers = len(_get_layers(model))
    
    # 获取W_U (部分测试需要)
    W_U = None
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        W_U = model.lm_head.weight.detach().cpu().float().numpy()
    
    print(f"  Model: {type(model).__name__}, d_model={d_model}, "
          f"n_layers={n_layers}, device={device}")
    
    # 运行测试
    all_results = {}
    tests_to_run = ['t1', 't2', 't3', 't4'] if test_name == 'all' else [test_name]
    
    for t in tests_to_run:
        t_start = time.time()
        
        if t == 't1':
            result = test_t1(model_name, model, tokenizer, device, d_model, n_layers, W_U)
        elif t == 't2':
            result = test_t2(model_name, model, tokenizer, device, d_model, n_layers, W_U)
        elif t == 't3':
            result = test_t3(model_name, model, tokenizer, device, d_model, n_layers, W_U)
        elif t == 't4':
            result = test_t4(model_name, model, tokenizer, device, d_model, n_layers, W_U)
        else:
            continue
        
        t_elapsed = time.time() - t_start
        all_results[t] = result
        print(f"\n  [{t.upper()}] elapsed: {t_elapsed:.1f}s")
    
    # 保存结果
    out_dir = Path(f"results/token_causal/{model_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for t, result in all_results.items():
        if result:
            out_file = out_dir / f"{t}_results.json"
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            print(f"  Saved: {out_file}")
    
    # 释放模型
    _release_model(model)
    print(f"\n  Done!")


if __name__ == "__main__":
    main()
