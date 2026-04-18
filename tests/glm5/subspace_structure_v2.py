"""
Phase CXCV: 大样本子空间结构分析 — 修正版
==========================================================

修正Phase CXCIV的问题:
  1. 样本量不足(30对) → 扩大到100+对/特征
  2. 主角度=0°(子空间200维太大) → 用Top-K=10/20/50小维度计算
  3. L1-sparse过拟合(50维/60样本) → 严格交叉验证
  4. 缺少随机基线 → 加入permutation test

核心测试:
  T1: 大样本L1-Sparse Probe (100对, 5-fold CV)
  T2: 修正主角度 — 小维度子空间(10/20/50维)
  T3: Permutation Test — 子空间重叠是否超过随机?
  T4: 跨模型结构验证 — 大样本下dim_90%/Gini是否仍不变?

运行:
  python tests/glm5/subspace_structure_v2.py --model qwen3 --test t1
  python tests/glm5/subspace_structure_v2.py --model qwen3 --test all
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
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ============================================================
# 大样本测试集 — 100+对/特征
# ============================================================

def generate_polarity_pairs():
    """生成100对极性对 (affirmative vs negative)"""
    templates = [
        # "X is [not] Y" pattern
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
        # "X was [not] Y" pattern
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
        # "X does [not] Y" pattern
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
        # "X can/will [not] Y" pattern
        ("He can swim", "He cannot swim"),
        ("The bird will come", "The bird will not come"),
        ("She can help", "She cannot help"),
        ("They will agree", "They will not agree"),
        ("He can solve it", "He cannot solve it"),
        ("The team will win", "The team will not win"),
        ("She can finish it", "She cannot finish it"),
        ("They will succeed", "They will not succeed"),
        # "X did [not] Y" pattern
        ("The cloud disappeared", "The cloud did not disappear"),
        ("He arrived early", "He did not arrive early"),
        ("She passed the test", "She did not pass the test"),
        ("They finished the work", "They did not finish the work"),
        ("The student graduated", "The student did not graduate"),
        ("He answered correctly", "He did not answer correctly"),
        ("She returned home", "She did not return home"),
        # "X has [not] Y" pattern
        ("The flower has bloomed", "The flower has not bloomed"),
        ("He has finished", "He has not finished"),
        ("She has arrived", "She has not arrived"),
        ("The tree has grown", "The tree has not grown"),
        ("The project has started", "The project has not started"),
        # Extra diverse patterns
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
    ]
    return templates


def generate_tense_pairs():
    """生成100对时态对 (present vs past)"""
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
        ("wins", "won"),
    ]
    subjects = [
        "The cat", "The dog", "He", "She", "The bird", "The man",
        "The woman", "The child", "The teacher", "The student",
        "The river", "The wind", "The fire", "The rain", "The sun",
        "I", "We", "They", "The team", "The group",
    ]
    objects_or_adverbs = [
        "fast", "slowly", "well", "hard", "often", "quickly", "carefully",
        "the book", "the song", "the letter", "the message", "the food",
        "home", "away", "alone", "together", "outside", "inside",
        "every day", "every morning", "every night", "today", "now",
    ]
    
    pairs = []
    for subj in subjects:
        for pres, past in verbs[:5]:  # 5 verbs per subject
            obj = objects_or_adverbs[len(pairs) % len(objects_or_adverbs)]
            pairs.append((f"{subj} {pres} {obj}", f"{subj} {past} {obj}"))
            if len(pairs) >= 100:
                return pairs[:100]
    return pairs[:100]


def generate_semantic_pairs():
    """生成100对语义对 (animate vs inanimate subject)"""
    animate_subjects = [
        "The girl", "The boy", "She", "He", "The teacher", "The doctor",
        "The cat", "The dog", "The bird", "The fish", "The horse",
        "The man", "The woman", "The child", "The student", "The worker",
        "The artist", "The chef", "The farmer", "The player",
    ]
    inanimate_subjects = [
        "The wind", "The storm", "The rain", "The flood", "The earthquake",
        "The tool", "The machine", "The computer", "The program", "The system",
        "The car", "The truck", "The crane", "The engine", "The motor",
        "The knife", "The saw", "The spring", "The vacuum", "The alarm",
    ]
    verbs_objects = [
        ("opened", "the door"), ("broke", "the window"), ("fixed", "machine"),
        ("moved", "the box"), ("closed", "the gate"), ("pulled", "the cart"),
        ("lifted", "the stone"), ("dropped", "glass"), ("cut", "the wood"),
        ("cleaned", "the room"), ("destroyed", "the nest"), ("found", "the key"),
        ("made", "the dish"), ("built", "the house"), ("solved", "the problem"),
        ("cured", "patient"), ("grew", "the crops"), ("created", "the art"),
        ("drove", "the bus"), ("guarded", "the house"), ("chased", "the cat"),
        ("hit", "the ball"), ("caught", "the fish"), ("sang", "the song"),
        ("wrote", "the code"), ("painted", "the wall"), ("explained", "it"),
    ]
    
    pairs = []
    for i, (verb, obj) in enumerate(verbs_objects):
        anim = animate_subjects[i % len(animate_subjects)]
        inanim = inanimate_subjects[i % len(inanimate_subjects)]
        pairs.append((f"{anim} {verb} {obj}", f"{inanim} {verb} {obj}"))
        if len(pairs) >= 100:
            break
    
    # Add more by rotating
    for i in range(len(verbs_objects), 100):
        verb, obj = verbs_objects[i % len(verbs_objects)]
        anim = animate_subjects[(i + 3) % len(animate_subjects)]
        inanim = inanimate_subjects[(i + 7) % len(inanimate_subjects)]
        pairs.append((f"{anim} {verb} {obj}", f"{inanim} {verb} {obj}"))
    
    return pairs[:100]


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
    """
    计算两个子空间之间的主角度
    
    Args:
        Q1: [d_model, k1] 正交基
        Q2: [d_model, k2] 正交基
        k_angles: 返回的角度数
    
    Returns:
        angles: [min(k1,k2,k_angles)] 角度列表(度)
    """
    k = min(Q1.shape[1], Q2.shape[1], k_angles)
    M = Q1.T @ Q2  # [k1, k2]
    try:
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        S = np.clip(S, -1, 1)
        angles = np.arccos(S[:k]) * 180 / np.pi
        return angles
    except:
        return np.array([90.0] * k)


# ============================================================
# T1: 大样本L1-Sparse Probe (严格交叉验证)
# ============================================================

def test_t1(model_name, model, tokenizer, device, d_model, n_layers):
    """T1: 大样本L1-Sparse Probe"""
    
    print("\n" + "=" * 70)
    print("T1: 大样本L1-Sparse Probe (100对, 5-fold CV)")
    print("=" * 70)
    
    polarity_pairs = generate_polarity_pairs()
    aff_texts = [aff for aff, neg in polarity_pairs]
    neg_texts = [neg for aff, neg in polarity_pairs]
    all_texts = aff_texts + neg_texts
    labels = np.array([0] * len(aff_texts) + [1] * len(neg_texts))
    
    n_samples = len(all_texts)
    print(f"  样本数: {n_samples} ({len(aff_texts)} aff + {len(neg_texts)} neg)")
    print(f"  d_model: {d_model}")
    
    key_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    key_layers = sorted(set(key_layers))
    
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    
    results = {
        'key_layers': key_layers,
        'alphas': alphas,
        'n_samples': n_samples,
        'sparse_probe': {},
    }
    
    for li in key_layers:
        print(f"\n  --- L{li} ---")
        
        repr_dict = extract_residual_stream(model, tokenizer, device, all_texts, [li], 'last')
        X = repr_dict[li]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        results['sparse_probe'][li] = {}
        
        for alpha in alphas:
            y_reg = labels * 2 - 1  # ±1
            
            lasso = Lasso(alpha=alpha, max_iter=20000)
            
            # 5-fold cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_accs = []
            cv_n_nonzero = []
            
            for train_idx, test_idx in cv.split(X_scaled, labels):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y_reg[train_idx], y_reg[test_idx]
                
                lasso.fit(X_train, y_train)
                y_pred = lasso.predict(X_test)
                y_pred_label = (y_pred > 0).astype(int)
                acc = np.mean(y_pred_label == labels[test_idx])
                n_nz = int(np.sum(np.abs(lasso.coef_) > 1e-6))
                
                cv_accs.append(acc)
                cv_n_nonzero.append(n_nz)
            
            # Full data fit
            lasso.fit(X_scaled, y_reg)
            w = lasso.coef_
            n_nonzero = int(np.sum(np.abs(w) > 1e-6))
            sparsity = 1.0 - n_nonzero / d_model
            full_acc = float(np.mean((lasso.predict(X_scaled) > 0).astype(int) == labels))
            
            mean_cv_acc = float(np.mean(cv_accs))
            std_cv_acc = float(np.std(cv_accs))
            mean_cv_nz = float(np.mean(cv_n_nonzero))
            
            # 维度组分析
            nonzero_dims = np.where(np.abs(w) > 1e-6)[0]
            if len(nonzero_dims) > 1:
                sorted_dims = np.sort(nonzero_dims)
                gaps = np.diff(sorted_dims)
                groups = []
                current_group = [int(sorted_dims[0])]
                for i in range(1, len(sorted_dims)):
                    if sorted_dims[i] - sorted_dims[i-1] < 5:
                        current_group.append(int(sorted_dims[i]))
                    else:
                        groups.append(current_group)
                        current_group = [int(sorted_dims[i])]
                groups.append(current_group)
                n_groups = len(groups)
                avg_group_size = float(np.mean([len(g) for g in groups]))
                max_group_size = max(len(g) for g in groups)
                n_large_groups = sum(1 for g in groups if len(g) >= 5)
            else:
                n_groups = max(len(nonzero_dims), 0)
                avg_group_size = 1.0
                max_group_size = min(len(nonzero_dims), 1)
                n_large_groups = 0
            
            results['sparse_probe'][li][alpha] = {
                'n_nonzero': n_nonzero,
                'sparsity': float(sparsity),
                'full_acc': float(full_acc),
                'cv_acc_mean': mean_cv_acc,
                'cv_acc_std': std_cv_acc,
                'cv_n_nonzero_mean': mean_cv_nz,
                'n_groups': n_groups,
                'avg_group_size': avg_group_size,
                'max_group_size': int(max_group_size),
                'n_large_groups': n_large_groups,
            }
            
            print(f"    alpha={alpha:6.3f}: nz={n_nonzero:4d}, sparsity={sparsity:.3f}, "
                  f"full_acc={full_acc:.3f}, CV_acc={mean_cv_acc:.3f}±{std_cv_acc:.3f}, "
                  f"groups={n_groups}, large={n_large_groups}")
        
        # L2-LR baseline
        clf_l2 = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
        cv_l2 = cross_val_score(clf_l2, X_scaled, labels, cv=5, scoring='accuracy')
        clf_l2.fit(X_scaled, labels)
        full_acc_l2 = float(clf_l2.score(X_scaled, labels))
        
        print(f"    [Baseline] L2-LR: full={full_acc_l2:.3f}, CV={np.mean(cv_l2):.3f}±{np.std(cv_l2):.3f}")
        results['sparse_probe'][li]['baseline_l2'] = {
            'full_acc': float(full_acc_l2),
            'cv_acc_mean': float(np.mean(cv_l2)),
            'cv_acc_std': float(np.std(cv_l2)),
        }
    
    return results


# ============================================================
# T2: 修正主角度 + 子空间交叉 (小维度)
# ============================================================

def test_t2(model_name, model, tokenizer, device, d_model, n_layers):
    """T2: 修正主角度 — 用小维度子空间计算"""
    
    print("\n" + "=" * 70)
    print("T2: 子空间交叉分析 (修正主角度, 小维度K=10/20/50)")
    print("=" * 70)
    
    # 准备大样本数据
    pol_pairs = generate_polarity_pairs()
    tense_pairs = generate_tense_pairs()
    sem_pairs = generate_semantic_pairs()
    
    features = {
        'polarity': (pol_pairs, "极性"),
        'tense': (tense_pairs, "时态"),
        'semantic': (sem_pairs, "语义"),
    }
    
    key_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    key_layers = sorted(set(key_layers))
    
    print(f"  关键层: {key_layers}")
    for fname, (pairs, label) in features.items():
        print(f"  {label}对: {len(pairs)}")
    
    results = {
        'key_layers': key_layers,
        'feature_info': {f: len(p) for f, (p, _) in features.items()},
        'subspace_dim': {},
        'direction_cosine': {},
        'jaccard_overlap': {},
        'principal_angles': {},
    }
    
    for li in key_layers:
        print(f"\n  --- L{li} ---")
        
        feature_data = {}
        
        for fname, (pairs, label) in features.items():
            cat_a = [a for a, b in pairs]
            cat_b = [b for a, b in pairs]
            all_texts = cat_a + cat_b
            labels_arr = np.array([0] * len(cat_a) + [1] * len(cat_b))
            
            repr_dict = extract_residual_stream(model, tokenizer, device, all_texts, [li], 'last')
            X = repr_dict[li]
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 5-fold CV for accuracy
            clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
            cv_scores = cross_val_score(clf, X_scaled, labels_arr, cv=5, scoring='accuracy')
            cv_mean = float(np.mean(cv_scores))
            
            clf.fit(X_scaled, labels_arr)
            full_acc = float(clf.score(X_scaled, labels_arr))
            
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
            mean_a = X[:len(cat_a)].mean(axis=0)
            mean_b = X[len(cat_a):].mean(axis=0)
            diff_vec = mean_b - mean_a
            diff_norm = np.linalg.norm(diff_vec)
            if diff_norm > 1e-10:
                diff_vec = diff_vec / diff_norm
            
            w_norm = w / (np.linalg.norm(w) + 1e-10)
            gini_coeff = gini(w)
            
            feature_data[fname] = {
                'full_acc': full_acc,
                'cv_acc': cv_mean,
                'w': w,
                'w_norm': w_norm,
                'diff_vec': diff_vec,
                'importance': importance,
                'sorted_dims': sorted_dims,
                'dim_50': dim_50,
                'dim_90': dim_90,
                'dim_95': dim_95,
                'gini': gini_coeff,
            }
            
            results['subspace_dim'][(fname, li)] = {
                'dim_50': dim_50,
                'dim_90': dim_90,
                'dim_95': dim_95,
                'full_acc': full_acc,
                'cv_acc': cv_mean,
                'gini': gini_coeff,
                'dim_90_pct': round(dim_90 / d_model, 3),
            }
            
            print(f"    {fname:10s}: CV={cv_mean:.3f}, dim_50={dim_50}({dim_50/d_model:.1%}), "
                  f"dim_90={dim_90}({dim_90/d_model:.1%}), gini={gini_coeff:.3f}")
        
        # 子空间交叉分析
        feature_names = list(feature_data.keys())
        for i, f1 in enumerate(feature_names):
            for f2 in feature_names[i+1:]:
                r1 = feature_data[f1]
                r2 = feature_data[f2]
                
                # 1. 方向余弦
                cos_diff = float(np.dot(r1['diff_vec'], r2['diff_vec']))
                cos_probe = float(np.dot(r1['w_norm'], r2['w_norm']))
                
                results['direction_cosine'][(f1, f2, li)] = {
                    'cos_diff': cos_diff,
                    'cos_probe': cos_probe,
                }
                
                # 2. Jaccard Top-K (K=10,20,50,100)
                for k in [10, 20, 50, 100]:
                    topk1 = set(r1['sorted_dims'][:k].tolist())
                    topk2 = set(r2['sorted_dims'][:k].tolist())
                    overlap = len(topk1 & topk2)
                    union = len(topk1 | topk2)
                    jaccard = overlap / max(union, 1)
                    # 随机期望 Jaccard
                    expected_j = k * k / (d_model * d_model) * k
                    
                    results['jaccard_overlap'][(f1, f2, li, k)] = {
                        'jaccard': float(jaccard),
                        'overlap': overlap,
                        'expected_jaccard': float(expected_j),
                    }
                
                # 3. 修正主角度 — 用小维度K
                for k_sub in [10, 20, 50]:
                    # 构建子空间基
                    dims1 = r1['sorted_dims'][:k_sub]
                    dims2 = r2['sorted_dims'][:k_sub]
                    
                    basis1 = np.zeros((k_sub, d_model))
                    basis2 = np.zeros((k_sub, d_model))
                    for j, d in enumerate(dims1):
                        basis1[j, d] = r1['importance'][d]
                    for j, d in enumerate(dims2):
                        basis2[j, d] = r2['importance'][d]
                    
                    # QR正交化
                    Q1, _ = np.linalg.qr(basis1.T)  # [d_model, k_sub]
                    Q2, _ = np.linalg.qr(basis2.T)  # [d_model, k_sub]
                    
                    # 主角度
                    angles = compute_principal_angles(Q1, Q2, k_angles=min(5, k_sub))
                    
                    results['principal_angles'][(f1, f2, li, k_sub)] = {
                        'angles': [round(float(a), 2) for a in angles],
                        'min_angle': float(angles[0]),
                        'mean_angle': float(np.mean(angles)),
                    }
                
                # 打印关键结果
                j50 = results['jaccard_overlap'][(f1, f2, li, 50)]['jaccard']
                pa10 = results['principal_angles'][(f1, f2, li, 10)]
                pa20 = results['principal_angles'][(f1, f2, li, 20)]
                
                # 判定
                if pa10['min_angle'] > 45:
                    angle_relation = "正交"
                elif pa10['min_angle'] > 20:
                    angle_relation = "斜交"
                else:
                    angle_relation = "接近"
                
                print(f"    {f1}∩{f2}: cos={cos_diff:+.3f}, J50={j50:.3f}, "
                      f"PA(K=10)={pa10['min_angle']:.1f}°/{pa10['mean_angle']:.1f}°, "
                      f"PA(K=20)={pa20['min_angle']:.1f}°/{pa20['mean_angle']:.1f}° → {angle_relation}")
    
    return results


# ============================================================
# T3: Permutation Test — 子空间重叠是否超过随机?
# ============================================================

def test_t3(model_name, model, tokenizer, device, d_model, n_layers):
    """T3: Permutation Test — 验证子空间结构的统计显著性"""
    
    print("\n" + "=" * 70)
    print("T3: Permutation Test — 子空间重叠的统计显著性")
    print("=" * 70)
    
    pol_pairs = generate_polarity_pairs()
    aff_texts = [aff for aff, neg in pol_pairs]
    neg_texts = [neg for aff, neg in pol_pairs]
    all_texts = aff_texts + neg_texts
    labels = np.array([0] * len(aff_texts) + [1] * len(neg_texts))
    
    n_permutations = 100
    
    # 在中间层做permutation test
    target_layer = n_layers // 2
    print(f"  目标层: L{target_layer}")
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
    top50_real = set(sorted_dims_real[:50].tolist())
    top100_real = set(sorted_dims_real[:100].tolist())
    gini_real = gini(w_real)
    
    # dim_90
    total = np.sum(importance_real)
    cumulative = np.cumsum(importance_real[sorted_dims_real]) / total
    dim_90_real = int(np.searchsorted(cumulative, 0.9)) + 1
    
    print(f"\n  真实指标: dim_90={dim_90_real}({dim_90_real/d_model:.1%}), gini={gini_real:.3f}")
    
    # Permutation test
    perm_ginis = []
    perm_dim90s = []
    perm_top50_sets = []
    
    np.random.seed(42)
    for p in range(n_permutations):
        # 打乱标签
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
        except:
            perm_ginis.append(0)
            perm_dim90s.append(d_model)
            perm_top50_sets.append(set())
        
        if (p + 1) % 25 == 0:
            print(f"  完成 {p+1}/{n_permutations} 次置换...")
    
    # 统计显著性
    gini_pvalue = float(np.mean([g >= gini_real for g in perm_ginis]))
    dim90_pvalue = float(np.mean([d <= dim_90_real for d in perm_dim90s]))
    
    # 随机基线
    mean_perm_gini = float(np.mean(perm_ginis))
    std_perm_gini = float(np.std(perm_ginis))
    mean_perm_dim90 = float(np.mean(perm_dim90s))
    std_perm_dim90 = float(np.std(perm_dim90s))
    
    # Top-50重叠与随机基线比较
    # 真实: polarity Top-50 vs 自己的重排 — 不适用
    # 改为: 真实probe的Top-50 vs 随机选50维
    random_top50_overlaps = []
    for p50 in perm_top50_sets:
        overlap = len(top50_real & p50)
        random_top50_overlaps.append(overlap)
    mean_random_overlap = float(np.mean(random_top50_overlaps))
    
    results = {
        'target_layer': target_layer,
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
            'mean_top50_overlap': mean_random_overlap,
        },
        'z_scores': {
            'gini_z': float((gini_real - mean_perm_gini) / max(std_perm_gini, 1e-6)),
            'dim90_z': float((dim_90_real - mean_perm_dim90) / max(std_perm_dim90, 1e-6)),
        }
    }
    
    print(f"\n  置换检验结果:")
    print(f"    Gini: 真实={gini_real:.3f}, 随机={mean_perm_gini:.3f}±{std_perm_gini:.3f}, "
          f"z={results['z_scores']['gini_z']:.1f}, p={gini_pvalue:.3f}")
    print(f"    dim_90%: 真实={dim_90_real/d_model:.3f}, 随机={mean_perm_dim90/d_model:.3f}±{std_perm_dim90/d_model:.3f}, "
          f"z={results['z_scores']['dim90_z']:.1f}, p={dim90_pvalue:.3f}")
    print(f"    Top50随机重叠: {mean_random_overlap:.1f}/50")
    
    if gini_pvalue < 0.05:
        print(f"    ★ Gini显著高于随机(p<0.05) → 极性维度有集中趋势")
    else:
        print(f"    Gini不显著 → 极性维度分布接近随机")
    
    if dim90_pvalue < 0.05:
        print(f"    ★ dim_90显著低于随机(p<0.05) → 极性信息比随机更集中")
    else:
        print(f"    dim_90不显著 → 极性信息分散度接近随机")
    
    return results


# ============================================================
# 主入口
# ============================================================

def run_test(model_name, test_name):
    print(f"\n{'='*70}")
    print(f"Phase CXCV: Subspace Structure V2 (Large Sample) - {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model_fast(model_name)
    n_layers = len(model.model.layers)
    d_model = model.config.hidden_size
    
    print(f"  n_layers={n_layers}, d_model={d_model}")
    
    result_dir = Path(f"results/subspace_structure_v2/{model_name}")
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
    
    if test_name in ['t1', 'all']:
        r = test_t1(model_name, model, tokenizer, device, d_model, n_layers)
        results['t1'] = r
        with open(result_dir / "t1_results.json", 'w') as f:
            json.dump(convert_keys(r), f, indent=2, default=str)
    
    if test_name in ['t2', 'all']:
        r = test_t2(model_name, model, tokenizer, device, d_model, n_layers)
        results['t2'] = r
        with open(result_dir / "t2_results.json", 'w') as f:
            json.dump(convert_keys(r), f, indent=2, default=str)
    
    if test_name in ['t3', 'all']:
        r = test_t3(model_name, model, tokenizer, device, d_model, n_layers)
        results['t3'] = r
        with open(result_dir / "t3_results.json", 'w') as f:
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
    parser.add_argument("--test", type=str, default="all", choices=["t1", "t2", "t3", "all"])
    args = parser.parse_args()
    
    run_test(args.model, args.test)
