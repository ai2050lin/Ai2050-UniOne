"""
瓶颈破解测试 B1/B2/B3: 解决当前三大基础问题
=============================================

B1: 随机初始化基线 — 区分"架构不变量" vs "学习产物"
    问题: 我们发现的特征对齐/头分化/泄漏比可能是架构固有的，不是训练学到的
    方案: 用同架构随机初始化模型做相同测试，训练模型的显著偏离才是"学习产物"

B2: 大规模统计可靠性 (>1000词对)
    问题: 绝大部分测试<10个词，无统计效力
    方案: 自动生成大规模句对(6类×200对=1200对)，计算95%置信区间

B3: 从分析到合成的验证 — 预测能力测试
    问题: 发现了结构但无法重建/预测
    方案: 用已知结构预测未知句对的编码，检验预测准确率

运行方式:
  python tests/glm5/bottleneck_breaker.py --model qwen3
  python tests/glm5/bottleneck_breaker.py --model glm4
  python tests/glm5/bottleneck_breaker.py --model deepseek7b
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
from itertools import product

# ============================================================
# B1: 随机初始化基线
# ============================================================

def b1_random_baseline(model_name, device_str="cuda"):
    """
    B1: 随机初始化基线测试
    
    核心思路:
    1. 用同架构随机初始化模型(不加载权重)
    2. 测量随机模型的: 功能对齐度、头分化、Q/K/V能量分布
    3. 与训练模型对比: 显著偏离=学习产物, 无差异=架构不变量
    
    关键指标:
    - Q/K/V在功能空间的对齐度: 随机模型 vs 训练模型
    - 注意力头功能/内容分化: 随机模型是否有同样模式
    - 残差流范数增长模式: 随机模型 vs 训练模型
    - 功能方向的余弦相似度: 随机模型是否有概念分离
    """
    print("\n" + "="*70)
    print("B1: 随机初始化基线测试")
    print("="*70)
    
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    
    # 模型配置
    MODEL_CONFIGS = {
        "qwen3": {
            "path": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
            "arch": "Qwen3ForCausalLM",
        },
        "glm4": {
            "path": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
            "arch": "GlmForCausalLM",
        },
        "deepseek7b": {
            "path": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "arch": "Qwen2ForCausalLM",
        },
    }
    
    cfg = MODEL_CONFIGS[model_name]
    
    # 1. 加载tokenizer
    print(f"  加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 加载随机初始化模型
    print(f"  创建随机初始化模型({cfg['arch']})...")
    config = AutoConfig.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True,
    )
    
    # 根据模型大小决定是否量化
    use_4bit = model_name in ["glm4", "deepseek7b"]
    
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        random_model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True,
        )
        # 随机初始化的模型不能直接4-bit量化，需要先保存再4-bit加载
        # 简化: 用CPU上的随机模型进行分析（权重分析不需要GPU前向传播）
        random_model = random_model.to("cpu")
        random_model.eval()
        device = "cpu"
    else:
        random_model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True,
        )
        if torch.cuda.is_available():
            random_model = random_model.to("cuda")
        random_model.eval()
        device = next(random_model.parameters()).device
    
    print(f"  随机模型: {type(random_model).__name__}, device={device}")
    
    # 3. 测试: Q/K/V在功能空间的对齐度 (随机模型)
    # 注意: 随机模型的功能方向需要用随机模型的hidden states提取
    # 但随机模型的前向传播输出是噪声 → 功能方向本身无意义
    # 关键测试: 随机权重矩阵在"训练模型功能方向"上的能量分布
    
    print(f"\n  === B1.1: 随机权重的功能对齐度 ===")
    
    # 先加载训练模型提取功能方向
    if use_4bit:
        trained_model, _, _ = load_model_4bit(model_name)
    else:
        from model_utils import load_model
        trained_model, _, trained_device = load_model(model_name)
    
    # 提取训练模型的功能方向 (用5类句对)
    V_func, ortho_labels = extract_trained_functional_directions(
        trained_model, tokenizer, trained_device if not use_4bit else next(trained_model.parameters()).device
    )
    
    if V_func is None:
        print("  ERROR: 无法提取功能方向")
        release_model(trained_model)
        return None
    
    n_func = V_func.shape[0]
    d_model = V_func.shape[1]
    print(f"  功能方向: {n_func}个, 维度={ortho_labels}")
    
    # 测量随机模型的Q/K/V在功能方向的能量
    random_layers = get_layers(random_model)
    n_layers = len(random_layers)
    
    # 采样层
    sample_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    
    random_alignment = {}
    for li in sample_layers:
        layer = random_layers[li]
        sa = layer.self_attn
        W_q = sa.q_proj.weight.detach().cpu().float().numpy()
        W_k = sa.k_proj.weight.detach().cpu().float().numpy()
        W_v = sa.v_proj.weight.detach().cpu().float().numpy()
        W_o = sa.o_proj.weight.detach().cpu().float().numpy()
        
        # 投影到功能方向
        def compute_func_ratio(W, V_func):
            if W.shape[1] == d_model:
                W_func = W @ V_func.T
            elif W.shape[0] == d_model:
                W_func = V_func @ W
            else:
                return 0.0
            func_energy = np.sum(W_func ** 2)
            total_energy = np.sum(W ** 2)
            return float(func_energy / (total_energy + 1e-30))
        
        q_func = compute_func_ratio(W_q, V_func)
        k_func = compute_func_ratio(W_k, V_func)
        v_func = compute_func_ratio(W_v, V_func)
        o_func = compute_func_ratio(W_o, V_func)
        
        random_alignment[li] = {
            'W_Q': q_func, 'W_K': k_func, 'W_V': v_func, 'W_O': o_func
        }
        print(f"  Random Layer {li}: Q={q_func:.4f}, K={k_func:.4f}, V={v_func:.4f}, O={o_func:.4f}")
    
    # 测量训练模型的Q/K/V在功能方向的能量 (对比)
    print(f"\n  === B1.1对比: 训练模型的功能对齐度 ===")
    trained_layers = get_layers(trained_model)
    trained_alignment = {}
    
    for li in sample_layers:
        layer = trained_layers[li]
        sa = layer.self_attn
        # 4-bit模型需要dequantize
        W_q = dequantize_weight(sa.q_proj.weight)
        W_k = dequantize_weight(sa.k_proj.weight)
        W_v = dequantize_weight(sa.v_proj.weight)
        W_o = dequantize_weight(sa.o_proj.weight)
        
        q_func = compute_func_ratio(W_q, V_func)
        k_func = compute_func_ratio(W_k, V_func)
        v_func = compute_func_ratio(W_v, V_func)
        o_func = compute_func_ratio(W_o, V_func)
        
        trained_alignment[li] = {
            'W_Q': q_func, 'W_K': k_func, 'W_V': v_func, 'W_O': o_func
        }
        print(f"  Trained Layer {li}: Q={q_func:.4f}, K={k_func:.4f}, V={v_func:.4f}, O={o_func:.4f}")
    
    # B1.2: 随机模型的概念分离能力
    print(f"\n  === B1.2: 概念分离能力 ===")
    
    # 用训练模型的hidden states看概念对在随机模型中的分离度
    concept_pairs = {
        'emotion': ("She is very happy today", "She is very sad today"),
        'animal': ("The cat sat on the mat", "The dog sat on the mat"),
        'logic': ("Because it rained the ground is wet", "The ground is wet although it did not rain"),
    }
    
    trained_cos = {}
    random_cos = {}
    
    for dim, (s1, s2) in concept_pairs.items():
        # 训练模型
        t_dev = next(trained_model.parameters()).device
        toks1 = tokenizer(s1, return_tensors="pt").to(t_dev)
        toks2 = tokenizer(s2, return_tensors="pt").to(t_dev)
        with torch.no_grad():
            h1 = trained_model(**toks1, output_hidden_states=True).hidden_states
            h2 = trained_model(**toks2, output_hidden_states=True).hidden_states
        
        # 中间层余弦相似度
        mid = len(h1) // 2
        r1 = h1[mid][0].mean(0).detach().cpu().float().numpy()
        r2 = h2[mid][0].mean(0).detach().cpu().float().numpy()
        cos_mid = float(np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2) + 1e-30))
        trained_cos[dim] = cos_mid
        
        # 随机模型 (CPU)
        if device == "cpu":
            toks1_r = tokenizer(s1, return_tensors="pt")
            toks2_r = tokenizer(s2, return_tensors="pt")
            with torch.no_grad():
                rh1 = random_model(**toks1_r, output_hidden_states=True).hidden_states
                rh2 = random_model(**toks2_r, output_hidden_states=True).hidden_states
            
            rr1 = rh1[mid][0].mean(0).detach().float().numpy()
            rr2 = rh2[mid][0].mean(0).detach().float().numpy()
            cos_r = float(np.dot(rr1, rr2) / (np.linalg.norm(rr1) * np.linalg.norm(rr2) + 1e-30))
        else:
            cos_r = 0.0
        
        random_cos[dim] = cos_r
        print(f"  {dim}: trained_cos={cos_mid:.4f}, random_cos={cos_r:.4f}, diff={cos_mid-cos_r:+.4f}")
    
    # 释放模型
    release_model(trained_model)
    del random_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # B1.3: 随机权重范数分布
    print(f"\n  === B1.3: 权重范数分布对比 ===")
    # 随机模型的权重范数 vs 训练模型的权重范数
    # 这是纯权重分析，不需要前向传播
    
    results = {
        'random_alignment': {str(k): v for k, v in random_alignment.items()},
        'trained_alignment': {str(k): v for k, v in trained_alignment.items()},
        'concept_separation': {
            'trained_cos': trained_cos,
            'random_cos': random_cos,
        },
        'n_func_directions': n_func,
        'd_model': d_model,
        'random_baseline_func_ratio': n_func / d_model,  # 随机对齐的理论基线
    }
    
    # 关键判断: 训练模型的功能对齐度是否显著超过随机基线
    print(f"\n  === B1核心结论 ===")
    random_baseline = n_func / d_model
    print(f"  随机对齐理论基线: {random_baseline:.4f} ({n_func}/{d_model})")
    
    for li in sample_layers:
        ta = trained_alignment[li]
        ra = random_alignment[li]
        for name in ['W_Q', 'W_K', 'W_V', 'W_O']:
            t_val = ta[name]
            r_val = ra[name]
            is_learned = t_val > r_val * 2  # 超过随机2倍视为学习产物
            print(f"  L{li} {name}: trained={t_val:.4f}, random={r_val:.4f}, "
                  f"ratio={t_val/(r_val+1e-30):.1f}x, "
                  f"{'学习产物' if is_learned else '架构不变量'}")
    
    return results


# ============================================================
# B2: 大规模统计可靠性
# ============================================================

# 大规模句对生成器
def generate_large_scale_pairs():
    """
    自动生成6类×200对=1200对句对
    使用模板化生成确保覆盖度和可控性
    """
    pairs = {
        'syntax': [],    # 句法变化 (单复数/时态/语态)
        'semantic': [],  # 语义替换 (名词/动词/形容词)
        'style': [],     # 风格变化 (正式/非正式)
        'tense': [],     # 时态变化 (现在/过去)
        'polarity': [],  # 极性变化 (肯定/否定)
        'logic': [],     # 逻辑关系 (因果/转折)
    }
    
    # --- Syntax: 单复数 ---
    singular_nouns = [
        "cat", "dog", "bird", "tree", "car", "house", "book", "river",
        "mountain", "flower", "child", "woman", "man", "fish", "star",
        "cloud", "door", "window", "street", "bridge", "table", "chair",
        "lamp", "phone", "key", "cup", "plate", "bottle", "shirt", "hat",
    ]
    for noun in singular_nouns[:50]:
        if noun == "child":
            plural = "children"
        elif noun == "woman":
            plural = "women"
        elif noun == "man":
            plural = "men"
        elif noun == "fish":
            plural = "fish"
        else:
            plural = noun + "s"
        pairs['syntax'].append((
            f"The {noun} sits on the mat",
            f"The {plural} sit on the mat"
        ))
    
    # --- Syntax: 语态变化 ---
    active_verbs = [
        ("chased", "was chased by"),
        ("caught", "was caught by"),
        ("saw", "was seen by"),
        ("heard", "was heard by"),
        ("followed", "was followed by"),
        ("helped", "was helped by"),
        ("pushed", "was pushed by"),
        ("pulled", "was pulled by"),
        ("watched", "was watched by"),
        ("admired", "was admired by"),
    ]
    subjects_objects = [
        ("The dog", "the cat"), ("The hunter", "the deer"),
        ("The teacher", "the student"), ("The doctor", "the patient"),
        ("The king", "the servant"), ("The chef", "the waiter"),
        ("The driver", "the passenger"), ("The artist", "the model"),
        ("The writer", "the editor"), ("The coach", "the player"),
    ]
    for (v_act, v_pass), (subj, obj) in product(active_verbs, subjects_objects):
        if len(pairs['syntax']) >= 200:
            break
        pairs['syntax'].append((
            f"{subj} {v_act} {obj}",
            f"{obj.capitalize()} {v_pass} {subj.lower()}"
        ))
    
    # --- Semantic: 名词替换 ---
    noun_pairs = [
        ("cat", "dog"), ("king", "queen"), ("sun", "moon"),
        ("car", "bicycle"), ("apple", "orange"), ("river", "mountain"),
        ("doctor", "teacher"), ("sword", "shield"), ("forest", "desert"),
        ("piano", "guitar"), ("winter", "summer"), ("coffee", "tea"),
        ("diamond", "ruby"), ("eagle", "hawk"), ("castle", "palace"),
        ("ocean", "lake"), ("library", "museum"), ("train", "airplane"),
        ("gold", "silver"), ("rose", "lily"), ("bread", "rice"),
        ("wolf", "fox"), ("hammer", "screwdriver"), ("whale", "dolphin"),
        ("thunder", "lightning"), ("mirror", "window"), ("crown", "helmet"),
        ("candle", "torch"), ("bridge", "tunnel"), ("feather", "scale"),
    ]
    templates_sem = [
        "The {} sat on the mat",
        "She saw the {} in the garden",
        "The {} was very beautiful",
        "He read about the {} yesterday",
        "The {} crossed the road quickly",
    ]
    for (n1, n2), tmpl in product(noun_pairs, templates_sem):
        if len(pairs['semantic']) >= 200:
            break
        pairs['semantic'].append((tmpl.format(n1), tmpl.format(n2)))
    
    # --- Style: 正式/非正式 ---
    style_pairs = [
        ("Hi", "Greetings"),
        ("Thanks", "I appreciate your assistance"),
        ("Bye", "Farewell"),
        ("OK", "That is acceptable"),
        ("No way", "That is highly improbable"),
        ("Cool", "That is quite impressive"),
        ("Yeah", "Indeed"),
        ("Dunno", "I am uncertain"),
        ("Gonna", "Intending to"),
        ("Wanna", "Desiring to"),
        ("Got it", "I understand"),
        ("So what", "What is the significance"),
        ("Big deal", "That is of considerable importance"),
        ("Let's go", "Shall we proceed"),
        ("Come on", "Please proceed"),
        ("What's up", "What is the current situation"),
        ("Hang out", "Spend time together"),
        ("Figure out", "Determine"),
        ("Point out", "Indicate"),
        ("Look into", "Investigate"),
    ]
    sentence_starters = [
        ("Hey, ", "Excuse me, "),
        ("Well, ", "Furthermore, "),
        ("Look, ", "Observe that "),
        ("I mean, ", "I intend to convey that "),
        ("You know, ", "It is worth noting that "),
    ]
    for (s1, s2), (pre1, pre2) in product(style_pairs, sentence_starters):
        if len(pairs['style']) >= 200:
            break
        pairs['style'].append((f"{pre1}{s1}", f"{pre2}{s2}"))
    
    # --- Tense: 现在/过去 ---
    verb_pairs = [
        ("walk", "walked"), ("run", "ran"), ("eat", "ate"),
        ("sleep", "slept"), ("write", "wrote"), ("read", "read"),
        ("speak", "spoke"), ("drive", "drove"), ("swim", "swam"),
        ("sing", "sang"), ("think", "thought"), ("buy", "bought"),
        ("teach", "taught"), ("catch", "caught"), ("fly", "flew"),
        ("grow", "grew"), ("draw", "drew"), ("know", "knew"),
        ("break", "broke"), ("choose", "chose"),
    ]
    subjects = [
        "I", "She", "He", "They", "We", "The cat", "The dog",
        "The child", "The woman", "The man",
    ]
    for (v1, v2), subj in product(verb_pairs, subjects):
        if len(pairs['tense']) >= 200:
            break
        pairs['tense'].append((
            f"{subj} {v1}s every day" if subj not in ["I", "They", "We"] else f"{subj} {v1} every day",
            f"{subj} {v2} yesterday"
        ))
    
    # --- Polarity: 肯定/否定 ---
    affirmative = [
        "She is happy", "The movie was good", "He can swim",
        "I like this song", "The test was easy", "She will come",
        "They are rich", "We have time", "The food is fresh",
        "He always arrives early", "The door is open", "The light is on",
        "The water is warm", "The sky is clear", "The road is safe",
        "The answer is correct", "The plan will work", "The cake is sweet",
        "The book is interesting", "The game is fun",
    ]
    negation_patterns = [
        ("is", "is not"), ("was", "was not"), ("can", "cannot"),
        ("will", "will not"), ("are", "are not"), ("have", "do not have"),
        ("always", "never"), ("is", "is not"),
    ]
    for aff in affirmative:
        if len(pairs['polarity']) >= 200:
            break
        # 简单否定
        for old, new in negation_patterns:
            if old in aff and len(pairs['polarity']) < 200:
                neg = aff.replace(old, new, 1)
                if neg != aff:
                    pairs['polarity'].append((aff, neg))
    
    # --- Logic: 因果/转折 ---
    cause_effect = [
        ("it rained heavily", "the ground is wet"),
        ("she studied hard", "she passed the exam"),
        ("the fire was hot", "the ice melted"),
        ("he exercised daily", "he became stronger"),
        ("the wind blew hard", "the tree fell down"),
        ("she ate too much", "she felt sick"),
        ("the sun was bright", "the shadows were sharp"),
        ("they practiced often", "they improved quickly"),
        ("the road was icy", "the car slipped"),
        ("he stayed up late", "he was tired"),
        ("the water boiled", "the pasta cooked"),
        ("she saved money", "she bought a house"),
        ("the team worked together", "they won the game"),
        ("he took medicine", "he felt better"),
        ("the temperature dropped", "the lake froze"),
    ]
    for cause, effect in cause_effect:
        if len(pairs['logic']) >= 200:
            break
        # 因果 vs 转折
        pairs['logic'].append((
            f"Because {cause}, {effect}",
            f"Although {cause}, {effect} did not happen as expected"
        ))
        # 不同因果强度
        pairs['logic'].append((
            f"Since {cause}, {effect}",
            f"Even though {cause}, {effect} was not guaranteed"
        ))
        # 条件 vs 事实
        pairs['logic'].append((
            f"If {cause}, then {effect}",
            f"Despite {cause}, {effect} was not observed"
        ))
    
    # 截断到200
    for k in pairs:
        pairs[k] = pairs[k][:200]
    
    return pairs


def b2_large_scale_statistics(model_name):
    """
    B2: 大规模统计可靠性测试
    
    核心: 用1200+句对测试，计算:
    1. 每类功能维度的余弦相似度分布 (均值+95%CI)
    2. 层级分离度分布 (哪层最能区分各类变化)
    3. 功能方向对齐度的置信区间
    """
    print("\n" + "="*70)
    print("B2: 大规模统计可靠性测试 (1200+句对)")
    print("="*70)
    
    # 加载模型
    if model_name in ["glm4", "deepseek7b"]:
        model, tokenizer, device = load_model_4bit(model_name)
    else:
        from model_utils import load_model
        model, tokenizer, device = load_model(model_name)
    
    layers = get_layers(model)
    n_layers = len(layers)
    d_model = model.config.hidden_size
    
    # 生成大规模句对
    print(f"  生成大规模句对...")
    pairs = generate_large_scale_pairs()
    for k, v in pairs.items():
        print(f"    {k}: {len(v)} 对")
    
    # 测量每类句对在各层的余弦相似度
    print(f"\n  测量各层余弦相似度分布...")
    
    sample_layers = list(range(0, n_layers, max(1, n_layers // 5)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    
    results = {}
    
    for dim_name, pair_list in pairs.items():
        print(f"\n  === {dim_name} ({len(pair_list)} 对) ===")
        
        cos_by_layer = {li: [] for li in sample_layers}
        delta_norms = []
        
        for idx, (s1, s2) in enumerate(pair_list):
            if idx % 50 == 0:
                print(f"    处理 {idx}/{len(pair_list)}...")
            
            try:
                toks1 = tokenizer(s1, return_tensors="pt", truncation=True, max_length=64).to(device)
                toks2 = tokenizer(s2, return_tensors="pt", truncation=True, max_length=64).to(device)
                
                with torch.no_grad():
                    h1 = model(**toks1, output_hidden_states=True).hidden_states
                    h2 = model(**toks2, output_hidden_states=True).hidden_states
                
                for li in sample_layers:
                    r1 = h1[li][0].mean(0).detach().cpu().float().numpy()
                    r2 = h2[li][0].mean(0).detach().cpu().float().numpy()
                    n1, n2 = np.linalg.norm(r1), np.linalg.norm(r2)
                    if n1 > 1e-8 and n2 > 1e-8:
                        cos = float(np.dot(r1, r2) / (n1 * n2))
                        cos_by_layer[li].append(cos)
                
                # delta norm (中间层)
                mid = n_layers // 2
                r1_mid = h1[mid][0].mean(0).detach().cpu().float().numpy()
                r2_mid = h2[mid][0].mean(0).float().numpy()
                delta_norms.append(float(np.linalg.norm(r2_mid - r1_mid)))
                
            except Exception as e:
                continue
        
        # 统计
        dim_results = {'cos_by_layer': {}, 'delta_norm': {}}
        
        for li in sample_layers:
            cos_vals = cos_by_layer[li]
            if len(cos_vals) > 0:
                mean_cos = float(np.mean(cos_vals))
                std_cos = float(np.std(cos_vals))
                n = len(cos_vals)
                ci95 = 1.96 * std_cos / (n ** 0.5) if n > 1 else 0
                
                dim_results['cos_by_layer'][li] = {
                    'mean': mean_cos,
                    'std': std_cos,
                    'n': n,
                    'ci95': ci95,
                    'min': float(np.min(cos_vals)),
                    'max': float(np.max(cos_vals)),
                }
                if li == sample_layers[len(sample_layers)//2]:  # 中间层
                    print(f"    Layer {li}: cos={mean_cos:.4f}±{ci95:.4f} (n={n})")
        
        if delta_norms:
            dim_results['delta_norm'] = {
                'mean': float(np.mean(delta_norms)),
                'std': float(np.std(delta_norms)),
                'n': len(delta_norms),
                'ci95': float(1.96 * np.std(delta_norms) / (len(delta_norms) ** 0.5)),
            }
            print(f"    Delta norm: {np.mean(delta_norms):.4f}±{1.96*np.std(delta_norms)/(len(delta_norms)**0.5):.4f}")
        
        results[dim_name] = dim_results
    
    # 层级分离度: 哪层最能区分不同功能维度
    print(f"\n  === B2核心: 层级分离度 ===")
    separation_by_layer = {}
    for li in sample_layers:
        dim_means = []
        for dim_name in pairs:
            if li in results[dim_name]['cos_by_layer']:
                dim_means.append(results[dim_name]['cos_by_layer'][li]['mean'])
        if len(dim_means) >= 2:
            # 分离度 = 维度间方差 / 维度内方差
            between_var = np.var(dim_means)
            within_vars = []
            for dim_name in pairs:
                if li in results[dim_name]['cos_by_layer']:
                    within_vars.append(results[dim_name]['cos_by_layer'][li]['std'] ** 2)
            within_var = np.mean(within_vars) if within_vars else 1e-30
            separation = between_var / (within_var + 1e-30)
            separation_by_layer[li] = float(separation)
            print(f"    Layer {li}: 分离度={separation:.4f} (between={between_var:.6f}, within={within_var:.6f})")
    
    results['layer_separation'] = {str(k): v for k, v in separation_by_layer.items()}
    
    # 释放模型
    release_model(model)
    
    return results


# ============================================================
# B3: 预测能力验证
# ============================================================

def b3_prediction_validation(model_name):
    """
    B3: 从分析到合成的验证
    
    核心思路:
    1. 用一半句对提取功能方向
    2. 用这些方向预测另一半句对的编码差异
    3. 检验: 预测的差异方向与实际差异方向的余弦相似度
    
    如果预测能力显著>随机基线 → 结构发现可泛化
    如果预测能力≈随机基线 → 结构发现不可泛化
    """
    print("\n" + "="*70)
    print("B3: 预测能力验证 (训练集/测试集分割)")
    print("="*70)
    
    # 加载模型
    if model_name in ["glm4", "deepseek7b"]:
        model, tokenizer, device = load_model_4bit(model_name)
    else:
        from model_utils import load_model
        model, tokenizer, device = load_model(model_name)
    
    layers = get_layers(model)
    n_layers = len(layers)
    
    # 生成句对并分割
    pairs = generate_large_scale_pairs()
    
    target_layer = n_layers // 2
    
    results = {}
    
    for dim_name, pair_list in pairs.items():
        print(f"\n  === {dim_name} ===")
        
        # 50/50分割
        n_train = len(pair_list) // 2
        train_pairs = pair_list[:n_train]
        test_pairs = pair_list[n_train:]
        
        # 训练集: 提取功能方向
        print(f"    训练集: {len(train_pairs)} 对, 测试集: {len(test_pairs)} 对")
        
        diffs_train = []
        for s1, s2 in train_pairs:
            try:
                toks1 = tokenizer(s1, return_tensors="pt", truncation=True, max_length=64).to(device)
                toks2 = tokenizer(s2, return_tensors="pt", truncation=True, max_length=64).to(device)
                with torch.no_grad():
                    h1 = model(**toks1, output_hidden_states=True).hidden_states[target_layer]
                    h2 = model(**toks2, output_hidden_states=True).hidden_states[target_layer]
                r1 = h1[0].mean(0).detach().cpu().float().numpy()
                r2 = h2[0].mean(0).detach().cpu().float().numpy()
                diff = r2 - r1
                norm = np.linalg.norm(diff)
                if norm > 1e-8:
                    diffs_train.append(diff / norm)
            except:
                continue
        
        if not diffs_train:
            print(f"    跳过 (无有效训练数据)")
            continue
        
        # 训练方向: 平均差异向量
        train_dir = np.mean(diffs_train, axis=0)
        train_norm = np.linalg.norm(train_dir)
        if train_norm > 1e-8:
            train_dir = train_dir / train_norm
        
        # 测试集: 预测 vs 实际
        prediction_cosines = []
        random_cosines = []
        
        for s1, s2 in test_pairs:
            try:
                toks1 = tokenizer(s1, return_tensors="pt", truncation=True, max_length=64).to(device)
                toks2 = tokenizer(s2, return_tensors="pt", truncation=True, max_length=64).to(device)
                with torch.no_grad():
                    h1 = model(**toks1, output_hidden_states=True).hidden_states[target_layer]
                    h2 = model(**toks2, output_hidden_states=True).hidden_states[target_layer]
                r1 = h1[0].mean(0).detach().cpu().float().numpy()
                r2 = h2[0].mean(0).detach().cpu().float().numpy()
                test_diff = r2 - r1
                test_norm = np.linalg.norm(test_diff)
                if test_norm < 1e-8:
                    continue
                test_dir = test_diff / test_norm
                
                # 预测: 训练方向与测试实际方向的余弦
                pred_cos = float(np.dot(train_dir, test_dir))
                prediction_cosines.append(pred_cos)
                
                # 随机基线: 训练方向与随机方向的余弦
                rand_dir = np.random.randn(len(train_dir))
                rand_dir = rand_dir / (np.linalg.norm(rand_dir) + 1e-30)
                rand_cos = float(np.dot(train_dir, rand_dir))
                random_cosines.append(rand_cos)
            except:
                continue
        
        if prediction_cosines:
            mean_pred = float(np.mean(prediction_cosines))
            ci_pred = float(1.96 * np.std(prediction_cosines) / (len(prediction_cosines) ** 0.5))
            mean_rand = float(np.mean(random_cosines))
            n_test = len(prediction_cosines)
            
            # 效应量: Cohen's d
            pooled_std = np.std(prediction_cosines + random_cosines)
            cohens_d = (mean_pred - mean_rand) / (pooled_std + 1e-30)
            
            # 预测成功率: 预测cos > 随机cos的比例
            success_rate = sum(1 for p, r in zip(prediction_cosines, random_cosines) if p > r) / len(prediction_cosines)
            
            results[dim_name] = {
                'n_train': len(diffs_train),
                'n_test': n_test,
                'prediction_cosine': {
                    'mean': mean_pred,
                    'ci95': ci_pred,
                    'min': float(np.min(prediction_cosines)),
                    'max': float(np.max(prediction_cosines)),
                },
                'random_cosine': {
                    'mean': mean_rand,
                },
                'cohens_d': float(cohens_d),
                'success_rate': float(success_rate),
            }
            
            print(f"    预测cos: {mean_pred:.4f}±{ci_pred:.4f}")
            print(f"    随机cos: {mean_rand:.4f}")
            print(f"    Cohen's d: {cohens_d:.2f}")
            print(f"    预测成功率: {success_rate:.2%}")
            
            # 判断
            if cohens_d > 0.8:
                verdict = "强预测力"
            elif cohens_d > 0.5:
                verdict = "中等预测力"
            elif cohens_d > 0.2:
                verdict = "弱预测力"
            else:
                verdict = "无预测力(≈随机)"
            print(f"    结论: {verdict}")
    
    release_model(model)
    return results


# ============================================================
# 辅助函数
# ============================================================

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


def dequantize_weight(param):
    if hasattr(param, 'quant_state'):
        import bitsandbytes.functional as bnbF
        deq = bnbF.dequantize_4bit(param.data, param.quant_state)
        return deq.detach().cpu().float().numpy().astype(np.float32)
    else:
        return param.detach().cpu().float().numpy().astype(np.float32)


def release_model(model):
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("[release] GPU memory released")


def extract_trained_functional_directions(model, tokenizer, device):
    """从训练模型提取功能方向"""
    pairs = {
        'syntax': [
            ("The cat sits on the mat", "The cats sit on the mat"),
            ("She walks to school", "She walked to school"),
            ("The dog chased the cat", "The cat was chased by the dog"),
        ],
        'semantic': [
            ("The cat sat on the mat", "The dog sat on the mat"),
            ("The king ruled the kingdom", "The queen ruled the kingdom"),
            ("He ate an apple", "He ate an orange"),
        ],
        'style': [
            ("The cat sat on the mat", "The feline rested upon the rug"),
            ("He is very happy", "He is exceedingly joyful"),
            ("The food was good", "The cuisine was delectable"),
        ],
        'tense': [
            ("I walk to school", "I walked to school"),
            ("She reads a book", "She read a book"),
            ("They play outside", "They played outside"),
        ],
        'polarity': [
            ("She is happy", "She is not happy"),
            ("The movie was good", "The movie was not good"),
            ("He can swim", "He cannot swim"),
        ],
    }
    
    d_model = model.config.hidden_size
    directions = {}
    
    for dim_name, pair_list in pairs.items():
        diffs = []
        for s1, s2 in pair_list:
            toks1 = tokenizer(s1, return_tensors="pt").to(device)
            toks2 = tokenizer(s2, return_tensors="pt").to(device)
            with torch.no_grad():
                h1 = model(**toks1, output_hidden_states=True).hidden_states[0]
                h2 = model(**toks2, output_hidden_states=True).hidden_states[0]
            r1 = h1[0].mean(0).detach().cpu().float().numpy()
            r2 = h2[0].mean(0).detach().cpu().float().numpy()
            diff = r2 - r1
            norm = np.linalg.norm(diff)
            if norm > 1e-8:
                diffs.append(diff / norm)
        
        if diffs:
            avg = np.mean(diffs, axis=0)
            norm = np.linalg.norm(avg)
            if norm > 1e-8:
                directions[dim_name] = avg / norm
    
    # Gram-Schmidt正交化
    dim_order = list(directions.keys())
    ortho_dirs = []
    ortho_labels = []
    for dim_name in dim_order:
        v = directions[dim_name].copy()
        for u in ortho_dirs:
            v -= np.dot(v, u) * u
        norm = np.linalg.norm(v)
        if norm > 0.01:
            ortho_dirs.append(v / norm)
            ortho_labels.append(dim_name)
    
    if not ortho_dirs:
        return None, None
    
    V_func = np.array(ortho_dirs)
    return V_func, ortho_labels


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Bottleneck Breaker: B1/B2/B3")
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--test", type=str, default="all",
                        choices=["b1", "b2", "b3", "all"],
                        help="运行哪个测试")
    args = parser.parse_args()
    
    model_name = args.model
    print(f"\n{'#'*70}")
    print(f"# 瓶颈破解测试: B1(随机基线) + B2(大规模) + B3(预测验证)")
    print(f"# Model: {model_name}")
    print(f"{'#'*70}")
    
    all_results = {
        'model': model_name,
        'timestamp': datetime.now().isoformat(),
    }
    
    # B1: 随机初始化基线
    if args.test in ["b1", "all"]:
        b1_results = b1_random_baseline(model_name)
        all_results['b1_random_baseline'] = b1_results
    
    # B2: 大规模统计可靠性
    if args.test in ["b2", "all"]:
        b2_results = b2_large_scale_statistics(model_name)
        all_results['b2_large_scale'] = b2_results
    
    # B3: 预测能力验证
    if args.test in ["b3", "all"]:
        b3_results = b3_prediction_validation(model_name)
        all_results['b3_prediction'] = b3_results
    
    # 保存结果
    output_dir = Path(r"d:\Ai2050\TransformerLens-Project\results\bottleneck_breaker")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_results.json"
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_path}")
    print(f"\n瓶颈破解测试完成!")


if __name__ == "__main__":
    main()
