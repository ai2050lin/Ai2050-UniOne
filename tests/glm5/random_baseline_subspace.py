"""
Phase CXCVI: 随机基线子空间分析 — 区分语言特有 vs 高维一般性质
=================================================================

核心问题: Phase CXCV发现dim_90%≈64%和Gini≈0.42可能是高维一般性质,
而非语言特有结构。需要通过随机基线实验区分。

核心测试:
  R1: 随机标签主角度 — 随机标签的Top-K子空间是否也正交(90°)?
  R2: 随机方向因果 — 随机1维注入是否也无效? (vs极性1维注入无效)
  R3: 深层方向汇聚vs随机 — 随机特征是否也有深层cos_diff增大?
  R4: 大样本Permutation — 三个模型上重复Permutation Test

运行:
  python tests/glm5/random_baseline_subspace.py --model qwen3 --test r1
  python tests/glm5/random_baseline_subspace.py --model qwen3 --test all
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import gc
import time
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ============================================================
# 大样本测试集 — 200+对/特征 (更大样本量!)
# ============================================================

def generate_polarity_pairs_large():
    """生成200对极性对"""
    templates = [
        # "X is [not] Y"
        ("The cat is here", "The cat is not here"),
        ("The dog is happy", "The dog is not happy"),
        ("The house is big", "The house is not big"),
        ("The phone is working", "The phone is not working"),
        ("The bridge is safe", "The bridge is not safe"),
        ("The star is visible", "The star is not visible"),
        ("The door is open", "The door is not open"),
        ("The lake is deep", "The lake is not deep"),
        ("The road is clear", "The road is not clear"),
        ("The wall is strong", "The wall is not strong"),
        ("The bird is flying", "The bird is not flying"),
        ("The fish is swimming", "The fish is not swimming"),
        ("The car is fast", "The car is not fast"),
        ("The tree is tall", "The tree is not tall"),
        ("The river is wide", "The river is not wide"),
        ("The book is interesting", "The book is not interesting"),
        ("The food is fresh", "The food is not fresh"),
        ("The light is bright", "The light is not bright"),
        ("The music is loud", "The music is not loud"),
        ("The weather is cold", "The weather is not cold"),
        # "X was [not] Y"
        ("The door was closed", "The door was not closed"),
        ("The child was playing", "The child was not playing"),
        ("The man was running", "The man was not running"),
        ("The woman was singing", "The woman was not singing"),
        ("The dog was barking", "The dog was not barking"),
        ("The sun was shining", "The sun was not shining"),
        ("The wind was blowing", "The wind was not blowing"),
        ("The rain was falling", "The rain was not falling"),
        ("The snow was melting", "The snow was not melting"),
        ("The fire was burning", "The fire was not burning"),
        # "X does [not] Y"
        ("I like the car", "I do not like the car"),
        ("She knows the answer", "She does not know the answer"),
        ("The river flows north", "The river does not flow north"),
        ("I understand the plan", "I do not understand the plan"),
        ("She likes the movie", "She does not like the movie"),
        ("The key works well", "The key does not work well"),
        ("The food tastes good", "The food does not taste good"),
        ("He drives the truck", "He does not drive the truck"),
        ("The system runs smoothly", "The system does not run smoothly"),
        ("The machine operates well", "The machine does not operate well"),
        # "X can/will [not] Y"
        ("He can swim", "He cannot swim"),
        ("The bird will come", "The bird will not come"),
        ("She can help", "She cannot help"),
        ("They will agree", "They will not agree"),
        ("He can solve it", "He cannot solve it"),
        ("The team will win", "The team will not win"),
        ("She can finish it", "She cannot finish it"),
        ("They will succeed", "They will not succeed"),
        # "X did [not] Y"
        ("The cloud disappeared", "The cloud did not disappear"),
        ("He arrived early", "He did not arrive early"),
        ("She passed the test", "She did not pass the test"),
        ("They finished the work", "They did not finish the work"),
        ("The student graduated", "The student did not graduate"),
        ("He answered correctly", "He did not answer correctly"),
        ("She returned home", "She did not return home"),
        # "X has [not] Y"
        ("The flower has bloomed", "The flower has not bloomed"),
        ("He has finished", "He has not finished"),
        ("She has arrived", "She has not arrived"),
        ("The tree has grown", "The tree has not grown"),
        ("The project has started", "The project has not started"),
        # Extended diverse patterns
        ("The water is clean", "The water is not clean"),
        ("The machine works", "The machine does not work"),
        ("He can see", "He cannot see"),
        ("The plan was approved", "The plan was not approved"),
        ("She will come", "She will not come"),
        ("The road is open", "The road is not open"),
        ("The system is stable", "The system is not stable"),
        ("I trust the result", "I do not trust the result"),
        ("The door is locked", "The door is not locked"),
        ("The glass is empty", "The glass is not empty"),
        ("The room is dark", "The room is not dark"),
        ("The sky is blue", "The sky is not blue"),
        ("The grass is green", "The grass is not green"),
        ("The coffee is hot", "The coffee is not hot"),
        ("The ice is thick", "The ice is not thick"),
        ("The path is narrow", "The path is not narrow"),
        ("The cake is sweet", "The cake is not sweet"),
        ("The movie is long", "The movie is not long"),
        ("The test is hard", "The test is not hard"),
        ("The game is fair", "The game is not fair"),
        ("The price is low", "The price is not low"),
        ("The quality is high", "The quality is not high"),
        ("The speed is slow", "The speed is not slow"),
        ("The sound is clear", "The sound is not clear"),
        ("The color is bright", "The color is not bright"),
        ("The shape is round", "The shape is not round"),
        ("The size is small", "The size is not small"),
        ("The weight is heavy", "The weight is not heavy"),
        ("The distance is short", "The distance is not short"),
        ("The temperature is warm", "The temperature is not warm"),
        # More diverse
        ("The apple is ripe", "The apple is not ripe"),
        ("The ocean is calm", "The ocean is not calm"),
        ("The mountain is high", "The mountain is not high"),
        ("The forest is dense", "The forest is not dense"),
        ("The desert is dry", "The desert is not dry"),
        ("The city is noisy", "The city is not noisy"),
        ("The village is quiet", "The village is not quiet"),
        ("The boy is tall", "The boy is not tall"),
        ("The girl is short", "The girl is not short"),
        ("The man is strong", "The man is not strong"),
        ("The woman is kind", "The woman is not kind"),
        ("The cat is black", "The cat is not black"),
        ("The dog is brown", "The dog is not brown"),
        ("The flower is red", "The flower is not red"),
        ("The leaf is green", "The leaf is not green"),
        ("The stone is hard", "The stone is not hard"),
        ("The feather is soft", "The feather is not soft"),
        ("The metal is shiny", "The metal is not shiny"),
        ("The wood is rough", "The wood is not rough"),
        ("The glass is smooth", "The glass is not smooth"),
        # Sentences with "not" in different positions
        ("He is running fast", "He is not running fast"),
        ("She is reading quietly", "She is not reading quietly"),
        ("They are sleeping peacefully", "They are not sleeping peacefully"),
        ("We are learning quickly", "We are not learning quickly"),
        ("The children are playing", "The children are not playing"),
        ("The birds are singing", "The birds are not singing"),
        ("The students are studying", "The students are not studying"),
        ("The workers are building", "The workers are not building"),
        ("The doctors are helping", "The doctors are not helping"),
        ("The scientists are researching", "The scientists are not researching"),
        # More patterns
        ("This is correct", "This is not correct"),
        ("That is true", "That is not true"),
        ("It is real", "It is not real"),
        ("The answer is right", "The answer is not right"),
        ("The method is effective", "The method is not effective"),
        ("The solution is simple", "The solution is not simple"),
        ("The problem is easy", "The problem is not easy"),
        ("The task is difficult", "The task is not difficult"),
        ("The process is fast", "The process is not fast"),
        ("The result is accurate", "The result is not accurate"),
        # Additional
        ("The building is tall", "The building is not tall"),
        ("The river is long", "The river is not long"),
        ("The park is beautiful", "The park is not beautiful"),
        ("The market is busy", "The market is not busy"),
        ("The station is crowded", "The station is not crowded"),
        ("The airport is modern", "The airport is not modern"),
        ("The library is old", "The library is not old"),
        ("The museum is large", "The museum is not large"),
        ("The theater is famous", "The theater is not famous"),
        ("The restaurant is popular", "The restaurant is not popular"),
        # Past tense negatives
        ("The concert was exciting", "The concert was not exciting"),
        ("The game was close", "The game was not close"),
        ("The match was fair", "The match was not fair"),
        ("The show was entertaining", "The show was not entertaining"),
        ("The speech was inspiring", "The speech was not inspiring"),
        ("The lecture was boring", "The lecture was not boring"),
        ("The exam was challenging", "The exam was not challenging"),
        ("The project was successful", "The project was not successful"),
        ("The experiment was conclusive", "The experiment was not conclusive"),
        ("The debate was heated", "The debate was not heated"),
        # Even more
        ("The engine is powerful", "The engine is not powerful"),
        ("The battery is full", "The battery is not full"),
        ("The signal is strong", "The signal is not strong"),
        ("The connection is stable", "The connection is not stable"),
        ("The software is updated", "The software is not updated"),
        ("The hardware is compatible", "The hardware is not compatible"),
        ("The network is secure", "The network is not secure"),
        ("The system is reliable", "The system is not reliable"),
        ("The device is functional", "The device is not functional"),
        ("The screen is bright", "The screen is not bright"),
    ]
    return templates[:200]


def generate_tense_pairs_large():
    """生成200对时态对"""
    verbs = [
        ("runs", "ran"), ("walks", "walked"), ("plays", "played"),
        ("sings", "sang"), ("eats", "ate"), ("drinks", "drank"),
        ("writes", "wrote"), ("reads", "read"), ("speaks", "spoke"),
        ("drives", "drove"), ("swims", "swam"), ("flies", "flew"),
        ("grows", "grew"), ("knows", "knew"), ("thinks", "thought"),
        ("brings", "brought"), ("builds", "built"), ("catches", "caught"),
        ("teaches", "taught"), ("feels", "felt"), ("finds", "found"),
        ("gives", "gave"), ("holds", "held"), ("keeps", "kept"),
        ("leaves", "left"), ("loses", "lost"), ("meets", "met"),
        ("pays", "paid"), ("sells", "sold"), ("sends", "sent"),
        ("shuts", "shut"), ("sits", "sat"), ("sleeps", "slept"),
        ("spends", "spent"), ("stands", "stood"), ("takes", "took"),
        ("tells", "told"), ("understands", "understood"), ("wears", "wore"),
        ("wins", "won"), ("begins", "began"), ("breaks", "broke"),
        ("chooses", "chose"), ("draws", "drew"), ("falls", "fell"),
        ("forgets", "forgot"), ("gets", "got"), ("hangs", "hung"),
        ("hides", "hid"), ("hurts", "hurt"),
    ]
    subjects = [
        "The cat", "The dog", "He", "She", "The bird", "The man",
        "The woman", "The child", "The teacher", "The student",
        "The river", "The wind", "The fire", "The rain", "The sun",
        "I", "We", "They", "The team", "The group",
        "The boy", "The girl", "The king", "The queen", "The doctor",
        "The farmer", "The driver", "The artist", "The writer", "The singer",
        "The player", "The runner", "The worker", "The leader", "The speaker",
        "The teacher", "The nurse", "The chef", "The guard", "The pilot",
    ]
    
    pairs = []
    for i, subj in enumerate(subjects):
        for pres, past in verbs[i % len(verbs):i % len(verbs) + 5]:
            pairs.append((f"{subj} {pres} every day", f"{subj} {past} yesterday"))
            if len(pairs) >= 200:
                return pairs[:200]
    return pairs[:200]


def generate_semantic_pairs_large():
    """生成200对语义对 (animate vs inanimate)"""
    animate = [
        "The girl", "The boy", "She", "He", "The teacher", "The doctor",
        "The cat", "The dog", "The bird", "The fish", "The horse",
        "The man", "The woman", "The child", "The student", "The worker",
        "The artist", "The chef", "The farmer", "The player",
        "The nurse", "The driver", "The singer", "The writer", "The guard",
        "The rabbit", "The deer", "The bear", "The fox", "The wolf",
        "The lion", "The tiger", "The monkey", "The elephant", "The whale",
        "The eagle", "The snake", "The frog", "The turtle", "The ant",
    ]
    inanimate = [
        "The wind", "The storm", "The rain", "The flood", "The earthquake",
        "The tool", "The machine", "The computer", "The program", "The system",
        "The car", "The truck", "The crane", "The engine", "The motor",
        "The knife", "The saw", "The spring", "The vacuum", "The alarm",
        "The bridge", "The tower", "The wall", "The gate", "The door",
        "The rock", "The stone", "The cloud", "The wave", "The flame",
        "The ice", "The snow", "The dust", "The smoke", "The steam",
        "The magnet", "The lens", "The pipe", "The wire", "The gear",
    ]
    verbs_objs = [
        ("opened", "the door"), ("broke", "the window"), ("fixed", "the machine"),
        ("moved", "the box"), ("closed", "the gate"), ("pulled", "the cart"),
        ("lifted", "the stone"), ("dropped", "the glass"), ("cut", "the wood"),
        ("cleaned", "the room"), ("destroyed", "the nest"), ("found", "the key"),
        ("made", "the dish"), ("built", "the house"), ("solved", "the problem"),
        ("cured", "the patient"), ("grew", "the crops"), ("created", "the art"),
        ("drove", "the bus"), ("guarded", "the house"), ("chased", "the cat"),
        ("hit", "the ball"), ("caught", "the fish"), ("sang", "the song"),
        ("wrote", "the code"), ("painted", "the wall"), ("explained", "it"),
        ("discovered", "the truth"), ("followed", "the path"), ("changed", "the plan"),
        ("designed", "the system"), ("tested", "the method"), ("improved", "the design"),
        ("finished", "the task"), ("started", "the engine"), ("stopped", "the car"),
        ("pushed", "the button"), ("turned", "the wheel"), ("checked", "the result"),
        ("repaired", "the damage"), ("damaged", "the roof"), ("shattered", "the glass"),
    ]
    
    pairs = []
    for i, (verb, obj) in enumerate(verbs_objs):
        a = animate[i % len(animate)]
        ia = inanimate[i % len(inanimate)]
        pairs.append((f"{a} {verb} {obj}", f"{ia} {verb} {obj}"))
        if len(pairs) >= 200:
            break
    
    # Add more by rotating
    for i in range(len(verbs_objs), 200):
        verb, obj = verbs_objs[i % len(verbs_objs)]
        a = animate[(i + 3) % len(animate)]
        ia = inanimate[(i + 7) % len(inanimate)]
        pairs.append((f"{a} {verb} {obj}", f"{ia} {verb} {obj}"))
    
    return pairs[:200]


# ============================================================
# 模型加载
# ============================================================

def load_model_fast(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    PATHS = {
        "qwen3": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "glm4": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "deepseek7b": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    }
    
    path = PATHS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if model_name in ["glm4", "deepseek7b"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            path, quantization_config=bnb_config, device_map="auto",
            trust_remote_code=True, local_files_only=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map='cpu',
            trust_remote_code=True, local_files_only=True,
        )
        model = model.to('cuda')
    
    model.eval()
    device = next(model.parameters()).device
    return model, tokenizer, device


# ============================================================
# 核心工具: 批量提取残差流
# ============================================================

def extract_residual_stream(model, tokenizer, device, texts, layers_to_probe, pool_method='last'):
    """批量提取残差流表示"""
    captured = {}
    hooks = []
    layers = model.model.layers
    
    for li in layers_to_probe:
        captured[li] = []
    
    def make_hook(li):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0].detach().cpu().float()
            else:
                h = output.detach().cpu().float()
            captured[li].append(h[0])
        return hook_fn
    
    for li in layers_to_probe:
        hooks.append(layers[li].register_forward_hook(make_hook(li)))
    
    results = {li: [] for li in layers_to_probe}
    
    for text in texts:
        for li in layers_to_probe:
            captured[li] = []
        
        toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
        
        with torch.no_grad():
            try:
                _ = model(**toks)
            except Exception as e:
                for li in layers_to_probe:
                    results[li].append(np.zeros(model.config.hidden_size))
                continue
        
        for li in layers_to_probe:
            if captured[li]:
                h = captured[li][0].numpy()
                if pool_method == 'last':
                    results[li].append(h[-1])
                elif pool_method == 'mean':
                    results[li].append(h.mean(axis=0))
                else:
                    results[li].append(h[-1])
            else:
                results[li].append(np.zeros(model.config.hidden_size))
    
    for h in hooks:
        h.remove()
    
    for li in layers_to_probe:
        results[li] = np.array(results[li])
    
    return results


# ============================================================
# 工具函数
# ============================================================

def gini(x):
    """计算Gini系数"""
    x = np.sort(np.abs(x))
    n = len(x)
    if np.sum(x) < 1e-15:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * x) - (n + 1) * np.sum(x)) / (n * np.sum(x) + 1e-10))


def compute_principal_angles(Q1, Q2, k_angles=5):
    """计算两个子空间之间的主角度"""
    k = min(Q1.shape[1], Q2.shape[1], k_angles)
    M = Q1.T @ Q2
    try:
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        S = np.clip(S, -1, 1)
        angles = np.arccos(S[:k]) * 180 / np.pi
        return angles
    except:
        return np.array([90.0] * k)


def fit_probe_and_analyze(X, labels, d_model):
    """拟合probe并返回分析结果"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 5-fold CV
    clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
    cv_scores = cross_val_score(clf, X_scaled, labels, cv=5, scoring='accuracy')
    cv_mean = float(np.mean(cv_scores))
    
    clf.fit(X_scaled, labels)
    full_acc = float(clf.score(X_scaled, labels))
    
    w = clf.coef_[0]
    importance = np.abs(w)
    sorted_dims = np.argsort(importance)[::-1]
    sorted_importance = importance[sorted_dims]
    total = np.sum(sorted_importance)
    cumulative = np.cumsum(sorted_importance) / total
    
    dim_50 = int(np.searchsorted(cumulative, 0.5)) + 1
    dim_90 = int(np.searchsorted(cumulative, 0.9)) + 1
    dim_95 = int(np.searchsorted(cumulative, 0.95)) + 1
    
    # 差分向量
    n_half = len(labels) // 2
    mean_a = X[:n_half].mean(axis=0)
    mean_b = X[n_half:].mean(axis=0)
    diff_vec = mean_b - mean_a
    diff_norm = np.linalg.norm(diff_vec)
    if diff_norm > 1e-10:
        diff_vec = diff_vec / diff_norm
    
    w_norm = w / (np.linalg.norm(w) + 1e-10)
    
    return {
        'w': w,
        'w_norm': w_norm,
        'diff_vec': diff_vec,
        'importance': importance,
        'sorted_dims': sorted_dims,
        'dim_50': dim_50,
        'dim_90': dim_90,
        'dim_95': dim_95,
        'gini': gini(w),
        'full_acc': full_acc,
        'cv_acc': cv_mean,
        'dim_90_pct': round(dim_90 / d_model, 3),
    }


# ============================================================
# R1: 随机标签主角度
# ============================================================

def test_r1(model_name, model, tokenizer, device, d_model, n_layers):
    """
    R1: 随机标签主角度 — 随机标签的Top-K子空间是否也正交(90°)?
    
    核心逻辑:
    - 用极性数据训练probe → 得到polarity子空间
    - 用相同数据+随机标签训练probe → 得到random子空间
    - 计算两组随机子空间之间的主角度
    - 如果随机也90° → 正交性是高维一般性质
    - 如果随机≠90° → 正交性是语言特有!
    """
    
    print("\n" + "=" * 70)
    print("R1: 随机标签主角度 — 正交性是语言特有还是高维一般性质?")
    print("=" * 70)
    
    pol_pairs = generate_polarity_pairs_large()
    tense_pairs = generate_tense_pairs_large()
    sem_pairs = generate_semantic_pairs_large()
    
    n_random_probes = 10  # 生成10组随机标签probe
    
    key_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    key_layers = sorted(set(key_layers))
    
    results = {
        'key_layers': key_layers,
        'n_random_probes': n_random_probes,
        'real_vs_real': {},  # 语言特征之间的主角度
        'random_vs_random': {},  # 随机标签之间的主角度
        'real_vs_random': {},  # 语言特征 vs 随机标签
    }
    
    for li in key_layers:
        print(f"\n  --- L{li} ---")
        
        # === 提取三类特征的数据 ===
        feature_probes = {}
        
        for fname, pairs in [('polarity', pol_pairs), ('tense', tense_pairs), ('semantic', sem_pairs)]:
            cat_a = [a for a, b in pairs]
            cat_b = [b for a, b in pairs]
            all_texts = cat_a + cat_b
            labels_arr = np.array([0] * len(cat_a) + [1] * len(cat_b))
            
            repr_dict = extract_residual_stream(model, tokenizer, device, all_texts, [li], 'last')
            X = repr_dict[li]
            
            probe_info = fit_probe_and_analyze(X, labels_arr, d_model)
            feature_probes[fname] = probe_info
            
            print(f"    {fname}: CV={probe_info['cv_acc']:.3f}, dim_90={probe_info['dim_90']}({probe_info['dim_90_pct']:.1%}), gini={probe_info['gini']:.3f}")
        
        # === 生成随机标签probe ===
        # 用极性数据的表示, 但打乱标签
        pol_a = [a for a, b in pol_pairs]
        pol_b = [b for a, b in pol_pairs]
        pol_texts = pol_a + pol_b
        pol_labels = np.array([0] * len(pol_a) + [1] * len(pol_b))
        
        repr_dict = extract_residual_stream(model, tokenizer, device, pol_texts, [li], 'last')
        X_pol = repr_dict[li]
        
        random_probes = []
        np.random.seed(42)
        for r in range(n_random_probes):
            perm_labels = np.random.permutation(pol_labels)
            probe_info = fit_probe_and_analyze(X_pol, perm_labels, d_model)
            # 随机标签的CV应该≈0.5, 但我们仍记录
            random_probes.append(probe_info)
        
        # === 计算主角度 ===
        
        # 1. Real vs Real (语言特征之间)
        fnames = list(feature_probes.keys())
        for i, f1 in enumerate(fnames):
            for f2 in fnames[i+1:]:
                p1 = feature_probes[f1]
                p2 = feature_probes[f2]
                
                for k_sub in [10, 20, 50]:
                    dims1 = p1['sorted_dims'][:k_sub]
                    dims2 = p2['sorted_dims'][:k_sub]
                    
                    basis1 = np.zeros((k_sub, d_model))
                    basis2 = np.zeros((k_sub, d_model))
                    for j, d in enumerate(dims1):
                        basis1[j, d] = p1['importance'][d]
                    for j, d in enumerate(dims2):
                        basis2[j, d] = p2['importance'][d]
                    
                    Q1, _ = np.linalg.qr(basis1.T)
                    Q2, _ = np.linalg.qr(basis2.T)
                    
                    angles = compute_principal_angles(Q1, Q2, k_angles=min(5, k_sub))
                    
                    results['real_vs_real'][(f1, f2, li, k_sub)] = {
                        'min_angle': float(angles[0]),
                        'mean_angle': float(np.mean(angles)),
                        'angles': [round(float(a), 2) for a in angles],
                    }
                    
                    print(f"    REAL {f1}∩{f2}(K={k_sub}): min={angles[0]:.1f}°, mean={np.mean(angles):.1f}°")
        
        # 2. Random vs Random (随机标签之间)
        random_angle_stats = {k: [] for k in [10, 20, 50]}
        
        for i in range(n_random_probes):
            for j in range(i+1, n_random_probes):
                rp1 = random_probes[i]
                rp2 = random_probes[j]
                
                for k_sub in [10, 20, 50]:
                    dims1 = rp1['sorted_dims'][:k_sub]
                    dims2 = rp2['sorted_dims'][:k_sub]
                    
                    basis1 = np.zeros((k_sub, d_model))
                    basis2 = np.zeros((k_sub, d_model))
                    for jj, d in enumerate(dims1):
                        basis1[jj, d] = rp1['importance'][d]
                    for jj, d in enumerate(dims2):
                        basis2[jj, d] = rp2['importance'][d]
                    
                    Q1, _ = np.linalg.qr(basis1.T)
                    Q2, _ = np.linalg.qr(basis2.T)
                    
                    angles = compute_principal_angles(Q1, Q2, k_angles=min(5, k_sub))
                    random_angle_stats[k_sub].append(float(angles[0]))
        
        for k_sub in [10, 20, 50]:
            angles_list = random_angle_stats[k_sub]
            if angles_list:
                results['random_vs_random'][(li, k_sub)] = {
                    'min_angle_mean': float(np.mean(angles_list)),
                    'min_angle_std': float(np.std(angles_list)),
                    'min_angle_min': float(np.min(angles_list)),
                    'min_angle_max': float(np.max(angles_list)),
                    'n_pairs': len(angles_list),
                }
                print(f"    RANDOM vs RANDOM(K={k_sub}): min_angle={np.mean(angles_list):.1f}°±{np.std(angles_list):.1f}°, "
                      f"range=[{np.min(angles_list):.1f}°, {np.max(angles_list):.1f}°]")
        
        # 3. Real vs Random (语言特征 vs 随机标签)
        for fname in fnames:
            p = feature_probes[fname]
            
            for k_sub in [10, 20]:
                real_random_angles = []
                for rp in random_probes:
                    dims1 = p['sorted_dims'][:k_sub]
                    dims2 = rp['sorted_dims'][:k_sub]
                    
                    basis1 = np.zeros((k_sub, d_model))
                    basis2 = np.zeros((k_sub, d_model))
                    for j, d in enumerate(dims1):
                        basis1[j, d] = p['importance'][d]
                    for j, d in enumerate(dims2):
                        basis2[j, d] = rp['importance'][d]
                    
                    Q1, _ = np.linalg.qr(basis1.T)
                    Q2, _ = np.linalg.qr(basis2.T)
                    
                    angles = compute_principal_angles(Q1, Q2, k_angles=min(3, k_sub))
                    real_random_angles.append(float(angles[0]))
                
                results['real_vs_random'][(fname, li, k_sub)] = {
                    'min_angle_mean': float(np.mean(real_random_angles)),
                    'min_angle_std': float(np.std(real_random_angles)),
                }
                print(f"    REAL({fname}) vs RANDOM(K={k_sub}): min_angle={np.mean(real_random_angles):.1f}°±{np.std(real_random_angles):.1f}°")
        
        # 方向余弦: real vs random
        for fname in fnames:
            p = feature_probes[fname]
            cos_with_random = []
            for rp in random_probes:
                cos_val = float(np.dot(p['diff_vec'], rp['diff_vec']))
                cos_with_random.append(cos_val)
            results['real_vs_random'][(fname, li, 'cos_diff')] = {
                'mean': float(np.mean(cos_with_random)),
                'std': float(np.std(cos_with_random)),
                'max_abs': float(np.max(np.abs(cos_with_random))),
            }
            print(f"    REAL({fname}) vs RANDOM cos_diff: {np.mean(cos_with_random):.3f}±{np.std(cos_with_random):.3f}")
    
    # === 汇总判别 ===
    print("\n" + "=" * 70)
    print("R1 汇总: 正交性是语言特有还是高维一般性质?")
    print("=" * 70)
    
    for li in key_layers:
        for k_sub in [10, 20]:
            # Real vs Real
            real_angles = []
            for f1 in ['polarity', 'tense', 'semantic']:
                for f2 in ['polarity', 'tense', 'semantic']:
                    if f1 < f2:
                        key = (f1, f2, li, k_sub)
                        if key in results['real_vs_real']:
                            real_angles.append(results['real_vs_real'][key]['min_angle'])
            
            # Random vs Random
            rand_key = (li, k_sub)
            if rand_key in results['random_vs_random']:
                rand_mean = results['random_vs_random'][rand_key]['min_angle_mean']
                rand_std = results['random_vs_random'][rand_key]['min_angle_std']
                
                real_mean = np.mean(real_angles) if real_angles else 0
                
                print(f"  L{li} K={k_sub}: Real∩Real={real_mean:.1f}°, Random∩Random={rand_mean:.1f}°±{rand_std:.1f}°", end="")
                
                if abs(real_mean - rand_mean) < 2 * rand_std:
                    print(" → 正交性可能是高维一般性质 (无显著差异)")
                else:
                    print(" → 正交性可能是语言特有! (有显著差异)")
    
    return results


# ============================================================
# R2: 随机方向因果效应
# ============================================================

def test_r2(model_name, model, tokenizer, device, d_model, n_layers):
    """
    R2: 随机方向因果效应 — 随机1维注入是否也无效?
    
    已知: 极性1维注入无效(斜率<0.05)
    问题: 随机方向的1维注入是否也无效?
    
    如果随机也无效 → 1维注入无效是高维一般性质
    如果随机更无效 → 极性1维注入虽然弱,但仍比随机强
    """
    
    print("\n" + "=" * 70)
    print("R2: 随机方向因果效应 — 1维注入无效是语言特有还是高维一般?")
    print("=" * 70)
    
    # 极性方向(差分向量)
    pol_pairs = generate_polarity_pairs_large()[:50]  # 用50对节省时间
    aff_texts = [aff for aff, neg in pol_pairs]
    neg_texts = [neg for aff, neg in pol_pairs]
    
    # 测试模板
    test_templates = [
        ("The cat is", "happy"),
        ("The door is", "open"),
        ("The bird is", "flying"),
        ("The system is", "stable"),
        ("The road is", "clear"),
    ]
    
    key_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    key_layers = sorted(set(key_layers))
    
    results = {
        'key_layers': key_layers,
        'polarity_injection': {},
        'random_injection': {},
        'comparison': {},
    }
    
    # === 1. 提取极性差分方向 ===
    print("  提取极性差分方向...")
    all_pol_texts = aff_texts + neg_texts
    pol_labels = np.array([0] * len(aff_texts) + [1] * len(neg_texts))
    
    target_layer = n_layers // 2
    repr_dict = extract_residual_stream(model, tokenizer, device, all_pol_texts, [target_layer], 'last')
    X_pol = repr_dict[target_layer]
    
    n_half = len(aff_texts)
    mean_aff = X_pol[:n_half].mean(axis=0)
    mean_neg = X_pol[n_half:].mean(axis=0)
    polarity_dir = mean_neg - mean_aff
    polarity_norm = np.linalg.norm(polarity_dir)
    if polarity_norm > 1e-10:
        polarity_dir = polarity_dir / polarity_norm
    
    # === 2. 生成随机方向 ===
    n_random_dirs = 20
    np.random.seed(42)
    random_dirs = []
    for _ in range(n_random_dirs):
        rd = np.random.randn(d_model)
        rd = rd / np.linalg.norm(rd)
        random_dirs.append(rd)
    
    # === 3. 在embedding层注入方向 ===
    print("  注入方向并测量因果效应...")
    
    betas = [1.0, 2.0, 4.0, 8.0, 16.0]
    
    for test_prompt, _ in test_templates:
        print(f"\n    Prompt: '{test_prompt}'")
        
        # 基线前向
        toks = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=32).to(device)
        with torch.no_grad():
            base_logits = model(**toks).logits[0, -1].detach().cpu().float().numpy()
        
        # 极性注入
        pol_effects = {}
        for beta in betas:
            embed_layer = model.get_input_embeddings()
            input_ids = toks.input_ids
            embeds = embed_layer(input_ids).detach().clone()
            
            dir_tensor = torch.tensor(beta * polarity_dir, dtype=embeds.dtype, device=device)
            embeds_intervened = embeds.clone()
            embeds_intervened[0, -1, :] += dir_tensor.to(embeds.dtype)
            
            pos_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
            
            with torch.no_grad():
                try:
                    interv_logits = model(inputs_embeds=embeds_intervened, position_ids=pos_ids).logits[0, -1].detach().cpu().float().numpy()
                except:
                    interv_logits = base_logits.copy()
            
            # 因果效应 = logits变化
            delta_logits = interv_logits - base_logits
            effect_norm = float(np.linalg.norm(delta_logits))
            max_change = float(np.max(np.abs(delta_logits)))
            
            pol_effects[beta] = {
                'effect_norm': effect_norm,
                'max_change': max_change,
            }
        
        results['polarity_injection'][test_prompt] = pol_effects
        
        # 随机方向注入
        random_effect_norms = {beta: [] for beta in betas}
        random_max_changes = {beta: [] for beta in betas}
        
        for rd in random_dirs[:10]:  # 10个随机方向
            for beta in betas:
                embeds = embed_layer(input_ids).detach().clone()
                dir_tensor = torch.tensor(beta * rd, dtype=embeds.dtype, device=device)
                embeds_intervened = embeds.clone()
                embeds_intervened[0, -1, :] += dir_tensor.to(embeds.dtype)
                
                pos_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
                
                with torch.no_grad():
                    try:
                        interv_logits = model(inputs_embeds=embeds_intervened, position_ids=pos_ids).logits[0, -1].detach().cpu().float().numpy()
                    except:
                        interv_logits = base_logits.copy()
                
                delta_logits = interv_logits - base_logits
                random_effect_norms[beta].append(float(np.linalg.norm(delta_logits)))
                random_max_changes[beta].append(float(np.max(np.abs(delta_logits))))
        
        results['random_injection'][test_prompt] = {
            beta: {
                'effect_norm_mean': float(np.mean(random_effect_norms[beta])),
                'effect_norm_std': float(np.std(random_effect_norms[beta])),
                'max_change_mean': float(np.mean(random_max_changes[beta])),
                'max_change_std': float(np.std(random_max_changes[beta])),
            }
            for beta in betas
        }
        
        # 对比
        for beta in betas:
            pol_effect = pol_effects[beta]['effect_norm']
            rand_mean = results['random_injection'][test_prompt][beta]['effect_norm_mean']
            rand_std = results['random_injection'][test_prompt][beta]['effect_norm_std']
            
            ratio = pol_effect / max(rand_mean, 1e-10)
            
            results['comparison'][(test_prompt, beta)] = {
                'pol_effect': pol_effect,
                'rand_effect_mean': rand_mean,
                'rand_effect_std': rand_std,
                'ratio': ratio,
            }
            
            print(f"      beta={beta:5.1f}: pol={pol_effect:.3f}, rand={rand_mean:.3f}±{rand_std:.3f}, ratio={ratio:.2f}")
    
    # === 汇总 ===
    print("\n" + "=" * 70)
    print("R2 汇总: 1维注入无效是语言特有还是高维一般?")
    print("=" * 70)
    
    for beta in betas:
        ratios = [results['comparison'][(p, beta)]['ratio'] for p, _ in test_templates if (p, beta) in results['comparison']]
        if ratios:
            mean_ratio = np.mean(ratios)
            print(f"  beta={beta}: pol/rand ratio={mean_ratio:.2f}", end="")
            if mean_ratio > 1.5:
                print(" → 极性方向比随机更强! 1维注入无效但不是完全无效")
            elif mean_ratio > 1.0:
                print(" → 极性方向略强于随机, 但差异不大")
            else:
                print(" → 极性方向与随机无差异 → 1维注入无效是高维一般性质")
    
    return results


# ============================================================
# R3: 深层方向汇聚 vs 随机
# ============================================================

def test_r3(model_name, model, tokenizer, device, d_model, n_layers):
    """
    R3: 深层方向汇聚vs随机 — 随机特征是否也有深层cos_diff增大?
    
    已知: 语言特征的cos_diff在深层增大(0.2-0.7)
    问题: 随机特征的cos_diff是否也在深层增大?
    """
    
    print("\n" + "=" * 70)
    print("R3: 深层方向汇聚vs随机 — 深层cos_diff增大是语言特有还是高维一般?")
    print("=" * 70)
    
    pol_pairs = generate_polarity_pairs_large()
    tense_pairs = generate_tense_pairs_large()
    sem_pairs = generate_semantic_pairs_large()
    
    # 采样多个层 (5个关键层 + 更多中间层)
    key_layers = [0, n_layers // 6, n_layers // 3, n_layers // 2, 2 * n_layers // 3, 5 * n_layers // 6, n_layers - 1]
    key_layers = sorted(set(key_layers))
    
    n_random_features = 5  # 5组随机标签
    
    results = {
        'key_layers': key_layers,
        'real_cos_diff': {},  # 语言特征之间的cos_diff
        'random_cos_diff': {},  # 随机特征之间的cos_diff
        'real_vs_random': {},
    }
    
    for li in key_layers:
        print(f"\n  --- L{li} ---")
        
        feature_diffs = {}
        
        # 语言特征
        for fname, pairs in [('polarity', pol_pairs), ('tense', tense_pairs), ('semantic', sem_pairs)]:
            cat_a = [a for a, b in pairs]
            cat_b = [b for a, b in pairs]
            all_texts = cat_a + cat_b
            
            repr_dict = extract_residual_stream(model, tokenizer, device, all_texts, [li], 'last')
            X = repr_dict[li]
            
            n_half = len(cat_a)
            mean_a = X[:n_half].mean(axis=0)
            mean_b = X[n_half:].mean(axis=0)
            diff = mean_b - mean_a
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                diff = diff / norm
            feature_diffs[fname] = diff
        
        # 计算语言特征之间的cos_diff
        fnames = list(feature_diffs.keys())
        for i, f1 in enumerate(fnames):
            for f2 in fnames[i+1:]:
                cos = float(np.dot(feature_diffs[f1], feature_diffs[f2]))
                results['real_cos_diff'][(f1, f2, li)] = cos
                print(f"    REAL {f1}∩{f2}: cos_diff={cos:+.4f}")
        
        # 随机特征
        pol_a = [a for a, b in pol_pairs]
        pol_b = [b for a, b in pol_pairs]
        pol_texts = pol_a + pol_b
        pol_labels = np.array([0] * len(pol_a) + [1] * len(pol_b))
        
        repr_dict = extract_residual_stream(model, tokenizer, device, pol_texts, [li], 'last')
        X_pol = repr_dict[li]
        n_half_pol = len(pol_a)
        
        random_diffs = []
        np.random.seed(42 + li)
        for r in range(n_random_features):
            # 随机二分
            perm = np.random.permutation(len(pol_labels))
            idx_a = perm[:n_half_pol]
            idx_b = perm[n_half_pol:]
            
            mean_a_rand = X_pol[idx_a].mean(axis=0)
            mean_b_rand = X_pol[idx_b].mean(axis=0)
            diff_rand = mean_b_rand - mean_a_rand
            norm_rand = np.linalg.norm(diff_rand)
            if norm_rand > 1e-10:
                diff_rand = diff_rand / norm_rand
            random_diffs.append(diff_rand)
        
        # 随机特征之间的cos_diff
        random_cos_diffs = []
        for i in range(len(random_diffs)):
            for j in range(i+1, len(random_diffs)):
                cos = float(np.dot(random_diffs[i], random_diffs[j]))
                random_cos_diffs.append(cos)
        
        results['random_cos_diff'][li] = {
            'mean': float(np.mean(random_cos_diffs)) if random_cos_diffs else 0,
            'std': float(np.std(random_cos_diffs)) if random_cos_diffs else 0,
            'max_abs': float(np.max(np.abs(random_cos_diffs))) if random_cos_diffs else 0,
            'values': [round(v, 4) for v in random_cos_diffs],
        }
        
        # 语言特征 vs 随机特征
        for fname in fnames:
            real_random_cos = [float(np.dot(feature_diffs[fname], rd)) for rd in random_diffs]
            results['real_vs_random'][(fname, li)] = {
                'mean': float(np.mean(real_random_cos)),
                'std': float(np.std(real_random_cos)),
                'max_abs': float(np.max(np.abs(real_random_cos))),
            }
        
        rand_info = results['random_cos_diff'][li]
        print(f"    RANDOM∩RANDOM: cos_diff={rand_info['mean']:+.4f}±{rand_info['std']:.4f}, max_abs={rand_info['max_abs']:.4f}")
    
    # === 汇总: 深层cos_diff趋势 ===
    print("\n" + "=" * 70)
    print("R3 汇总: 深层cos_diff增大是语言特有还是高维一般?")
    print("=" * 70)
    
    # 检查语言特征的cos_diff是否随层增大
    for f1, f2 in [('tense', 'semantic'), ('polarity', 'tense'), ('polarity', 'semantic')]:
        real_cos_by_layer = []
        for li in key_layers:
            key = (f1, f2, li)
            if key in results['real_cos_diff']:
                real_cos_by_layer.append((li, results['real_cos_diff'][key]))
        
        if len(real_cos_by_layer) >= 2:
            early = real_cos_by_layer[0][1]
            late = real_cos_by_layer[-1][1]
            trend = abs(late) - abs(early)
            print(f"  {f1}∩{f2}: |cos| early={abs(early):.4f}, late={abs(late):.4f}, trend={trend:+.4f}", end="")
    
    # 检查随机特征的cos_diff是否随层增大
    random_cos_by_layer = []
    for li in key_layers:
        if li in results['random_cos_diff']:
            random_cos_by_layer.append((li, results['random_cos_diff'][li]['mean'], results['random_cos_diff'][li]['max_abs']))
    
    if len(random_cos_by_layer) >= 2:
        early_mean = abs(random_cos_by_layer[0][1])
        late_mean = abs(random_cos_by_layer[-1][1])
        early_max = random_cos_by_layer[0][2]
        late_max = random_cos_by_layer[-1][2]
        print(f"\n  RANDOM: |cos_mean| early={early_mean:.4f}, late={late_mean:.4f}")
        print(f"  RANDOM: max_abs early={early_max:.4f}, late={late_max:.4f}")
        
        if abs(late_mean) > 2 * abs(early_mean):
            print("  → 随机特征也在深层出现方向汇聚! 深层cos_diff增大可能是高维一般性质")
        else:
            print("  → 随机特征深层不汇聚! 深层cos_diff增大可能是语言特有!")
    
    return results


# ============================================================
# R4: 大样本Permutation (三个模型)
# ============================================================

def test_r4(model_name, model, tokenizer, device, d_model, n_layers):
    """
    R4: 大样本Permutation — 200样本, 200次置换
    """
    
    print("\n" + "=" * 70)
    print("R4: 大样本Permutation Test (200样本, 200置换)")
    print("=" * 70)
    
    pol_pairs = generate_polarity_pairs_large()
    aff_texts = [aff for aff, neg in pol_pairs]
    neg_texts = [neg for aff, neg in pol_pairs]
    all_texts = aff_texts + neg_texts
    labels = np.array([0] * len(aff_texts) + [1] * len(neg_texts))
    
    n_permutations = 200
    target_layer = n_layers // 2
    
    print(f"  目标层: L{target_layer}")
    print(f"  样本数: {len(all_texts)}")
    print(f"  置换次数: {n_permutations}")
    
    repr_dict = extract_residual_stream(model, tokenizer, device, all_texts, [target_layer], 'last')
    X = repr_dict[target_layer]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 真实probe
    clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
    clf.fit(X_scaled, labels)
    w_real = clf.coef_[0]
    importance_real = np.abs(w_real)
    sorted_dims_real = np.argsort(importance_real)[::-1]
    
    # 真实指标
    gini_real = gini(w_real)
    total = np.sum(importance_real)
    cumulative = np.cumsum(importance_real[sorted_dims_real]) / total
    dim_90_real = int(np.searchsorted(cumulative, 0.9)) + 1
    
    # 真实Top-50维度
    top50_real = set(sorted_dims_real[:50].tolist())
    
    # 真实probe的主角度分析
    # 用两组不同的随机标签作为对比
    print(f"\n  真实指标: dim_90={dim_90_real}({dim_90_real/d_model:.1%}), gini={gini_real:.3f}")
    
    # Permutation
    perm_ginis = []
    perm_dim90s = []
    perm_top50_sets = []
    # 新增: 随机标签的主角度
    perm_sorted_dims_list = []
    perm_importance_list = []
    
    np.random.seed(42)
    for p in range(n_permutations):
        perm_labels = np.random.permutation(labels)
        
        clf_perm = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
        try:
            clf_perm.fit(X_scaled, perm_labels)
            w_perm = clf_perm.coef_[0]
            imp_perm = np.abs(w_perm)
            sorted_perm = np.argsort(imp_perm)[::-1]
            
            perm_ginis.append(gini(w_perm))
            
            total_p = np.sum(imp_perm)
            cum_p = np.cumsum(imp_perm[sorted_perm]) / total_p
            perm_dim90s.append(int(np.searchsorted(cum_p, 0.9)) + 1)
            
            perm_top50_sets.append(set(sorted_perm[:50].tolist()))
            perm_sorted_dims_list.append(sorted_perm)
            perm_importance_list.append(imp_perm)
        except:
            perm_ginis.append(0)
            perm_dim90s.append(d_model)
            perm_top50_sets.append(set())
        
        if (p + 1) % 50 == 0:
            print(f"  完成 {p+1}/{n_permutations} 次置换...")
    
    # 统计显著性
    gini_pvalue = float(np.mean([g >= gini_real for g in perm_ginis]))
    dim90_pvalue = float(np.mean([d <= dim_90_real for d in perm_dim90s]))
    
    mean_perm_gini = float(np.mean(perm_ginis))
    std_perm_gini = float(np.std(perm_ginis))
    mean_perm_dim90 = float(np.mean(perm_dim90s))
    std_perm_dim90 = float(np.std(perm_dim90s))
    
    # === 随机标签的主角度 ===
    print("\n  计算随机标签probe之间的主角度...")
    random_principal_angles = {k: [] for k in [10, 20]}
    
    n_pa_pairs = min(20, len(perm_sorted_dims_list))  # 20对随机probe
    for i in range(n_pa_pairs):
        for j in range(i+1, min(i+3, n_pa_pairs)):  # 每个probe与后2个比较
            for k_sub in [10, 20]:
                dims1 = perm_sorted_dims_list[i][:k_sub]
                dims2 = perm_sorted_dims_list[j][:k_sub]
                
                basis1 = np.zeros((k_sub, d_model))
                basis2 = np.zeros((k_sub, d_model))
                for jj, d in enumerate(dims1):
                    basis1[jj, d] = perm_importance_list[i][d]
                for jj, d in enumerate(dims2):
                    basis2[jj, d] = perm_importance_list[j][d]
                
                Q1, _ = np.linalg.qr(basis1.T)
                Q2, _ = np.linalg.qr(basis2.T)
                
                angles = compute_principal_angles(Q1, Q2, k_angles=min(5, k_sub))
                random_principal_angles[k_sub].append(float(angles[0]))
    
    # 真实特征的主角度(从之前结果中获取或重新计算)
    # 简化: 用polarity vs tense
    tense_pairs = generate_tense_pairs_large()
    tense_a = [a for a, b in tense_pairs]
    tense_b = [b for a, b in tense_pairs]
    tense_texts = tense_a + tense_b
    tense_labels = np.array([0] * len(tense_a) + [1] * len(tense_b))
    
    repr_dict_tense = extract_residual_stream(model, tokenizer, device, tense_texts, [target_layer], 'last')
    X_tense = repr_dict_tense[target_layer]
    X_tense_scaled = scaler.transform(X_tense)  # 用同一个scaler
    
    clf_tense = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
    clf_tense.fit(X_tense_scaled, tense_labels)
    w_tense = clf_tense.coef_[0]
    imp_tense = np.abs(w_tense)
    sorted_dims_tense = np.argsort(imp_tense)[::-1]
    
    real_principal_angles = {}
    for k_sub in [10, 20]:
        dims1 = sorted_dims_real[:k_sub]
        dims2 = sorted_dims_tense[:k_sub]
        
        basis1 = np.zeros((k_sub, d_model))
        basis2 = np.zeros((k_sub, d_model))
        for jj, d in enumerate(dims1):
            basis1[jj, d] = importance_real[d]
        for jj, d in enumerate(dims2):
            basis2[jj, d] = imp_tense[d]
        
        Q1, _ = np.linalg.qr(basis1.T)
        Q2, _ = np.linalg.qr(basis2.T)
        
        angles = compute_principal_angles(Q1, Q2, k_angles=min(5, k_sub))
        real_principal_angles[k_sub] = {
            'min_angle': float(angles[0]),
            'mean_angle': float(np.mean(angles)),
        }
    
    # 汇总
    results = {
        'target_layer': target_layer,
        'n_samples': len(all_texts),
        'n_permutations': n_permutations,
        'real': {
            'dim_90': dim_90_real,
            'dim_90_pct': round(dim_90_real / d_model, 3),
            'gini': gini_real,
        },
        'permutation': {
            'mean_gini': mean_perm_gini,
            'std_gini': std_perm_gini,
            'mean_dim90': mean_perm_dim90,
            'std_dim90': std_perm_dim90,
            'mean_dim90_pct': round(mean_perm_dim90 / d_model, 3),
            'gini_pvalue': gini_pvalue,
            'dim90_pvalue': dim90_pvalue,
        },
        'z_scores': {
            'gini_z': float((gini_real - mean_perm_gini) / max(std_perm_gini, 1e-6)),
            'dim90_z': float((dim_90_real - mean_perm_dim90) / max(std_perm_dim90, 1e-6)),
        },
        'principal_angles': {
            'real_pol_tense': real_principal_angles,
            'random_mean': {k: float(np.mean(v)) for k, v in random_principal_angles.items() if v},
            'random_std': {k: float(np.std(v)) for k, v in random_principal_angles.items() if v},
        }
    }
    
    print(f"\n  置换检验结果:")
    print(f"    Gini: 真实={gini_real:.3f}, 随机={mean_perm_gini:.3f}±{std_perm_gini:.3f}, "
          f"z={results['z_scores']['gini_z']:.1f}, p={gini_pvalue:.3f}")
    print(f"    dim_90%: 真实={dim_90_real/d_model:.3f}, 随机={mean_perm_dim90/d_model:.3f}±{std_perm_dim90/d_model:.3f}, "
          f"z={results['z_scores']['dim90_z']:.1f}, p={dim90_pvalue:.3f}")
    
    print(f"\n  主角度对比:")
    for k_sub in [10, 20]:
        real_pa = real_principal_angles[k_sub]['min_angle']
        rand_mean = results['principal_angles']['random_mean'].get(k_sub, 0)
        rand_std = results['principal_angles']['random_std'].get(k_sub, 0)
        print(f"    K={k_sub}: Real(pol∩tense)={real_pa:.1f}°, Random={rand_mean:.1f}°±{rand_std:.1f}°", end="")
        if abs(real_pa - rand_mean) < 2 * max(rand_std, 1):
            print(" → 正交性可能是高维一般性质")
        else:
            print(" → 正交性可能不同!")
    
    return results


# ============================================================
# 主入口
# ============================================================

def run_test(model_name, test_name):
    print(f"\n{'='*70}")
    print(f"Phase CXCVI: Random Baseline Subspace - {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model_fast(model_name)
    n_layers = len(model.model.layers)
    d_model = model.config.hidden_size
    
    print(f"  n_layers={n_layers}, d_model={d_model}")
    
    result_dir = Path(f"results/random_baseline_subspace/{model_name}")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    results = {}
    
    def convert_keys(obj):
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                new_key = str(k) if isinstance(k, tuple) else k
                new_dict[new_key] = convert_keys(v)
            return new_dict
        elif isinstance(obj, list):
            return [convert_keys(item) for item in obj]
        else:
            return obj
    
    if test_name in ['r1', 'all']:
        r = test_r1(model_name, model, tokenizer, device, d_model, n_layers)
        results['r1'] = r
        with open(result_dir / "r1_results.json", 'w') as f:
            json.dump(convert_keys(r), f, indent=2, default=str)
    
    if test_name in ['r2', 'all']:
        r = test_r2(model_name, model, tokenizer, device, d_model, n_layers)
        results['r2'] = r
        with open(result_dir / "r2_results.json", 'w') as f:
            json.dump(convert_keys(r), f, indent=2, default=str)
    
    if test_name in ['r3', 'all']:
        r = test_r3(model_name, model, tokenizer, device, d_model, n_layers)
        results['r3'] = r
        with open(result_dir / "r3_results.json", 'w') as f:
            json.dump(convert_keys(r), f, indent=2, default=str)
    
    if test_name in ['r4', 'all']:
        r = test_r4(model_name, model, tokenizer, device, d_model, n_layers)
        results['r4'] = r
        with open(result_dir / "r4_results.json", 'w') as f:
            json.dump(convert_keys(r), f, indent=2, default=str)
    
    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.1f}s")
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--test", type=str, default="all", choices=["r1", "r2", "r3", "r4", "all"])
    args = parser.parse_args()
    
    run_test(args.model, args.test)
