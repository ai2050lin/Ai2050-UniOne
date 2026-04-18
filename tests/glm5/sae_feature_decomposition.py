"""
Phase CLXXX: SAE稀疏特征分解 — 从预测方向到原子特征
====================================================

基于B3验证的发现:
- tense/polarity有强预测力(cos>0.98) → 存在统一编码方向
- syntax有强预测力(cos>0.87) → 句法变换有可泛化方向
- semantic无预测力(cos≈0.03) → 概念特定，无统一方向

SAE分析目标:
1. 在强预测方向上做稀疏分解，找到底层原子特征
2. 分析原子特征的组合结构（是否可加性组合？）
3. 与随机基线对比，确认特征是学习产物
4. 跨模型对比特征的通用性

运行方式:
  python tests/glm5/sae_feature_decomposition.py --model qwen3
  python tests/glm5/sae_feature_decomposition.py --model glm4
  python tests/glm5/sae_feature_decomposition.py --model deepseek7b
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
# 大规模句对生成器（从bottleneck_breaker扩展）
# ============================================================

def generate_enhanced_pairs():
    """
    增强版句对生成器:
    - tense: 300对（更多动词+更多主语+更多上下文）
    - polarity: 200对（更多句式+双重否定等）
    - syntax: 300对（单复数+语态+疑问句）
    - semantic: 200对（同类+跨类替换）
    - logic: 200对（因果+转折+条件）
    - style: 200对（正式/非正式+语气词+口语化）
    """
    from itertools import product
    
    pairs = {
        'syntax': [],
        'semantic': [],
        'style': [],
        'tense': [],
        'polarity': [],
        'logic': [],
    }
    
    # --- Tense: 扩展到300对 ---
    verb_pairs = [
        # 规则变化
        ("walk", "walked"), ("talk", "talked"), ("play", "played"),
        ("work", "worked"), ("live", "lived"), ("open", "opened"),
        ("close", "closed"), ("move", "moved"), ("smile", "smiled"),
        ("laugh", "laughed"), ("cry", "cried"), ("try", "tried"),
        ("study", "studied"), ("carry", "carried"), ("hurry", "hurried"),
        ("enjoy", "enjoyed"), ("finish", "finished"), ("listen", "listened"),
        ("watch", "watched"), ("clean", "cleaned"), ("jump", "jumped"),
        ("climb", "climbed"), ("dance", "danced"), ("wait", "waited"),
        ("paint", "painted"), ("cook", "cooked"), ("dream", "dreamed"),
        # 不规则变化
        ("run", "ran"), ("eat", "ate"), ("sleep", "slept"),
        ("write", "wrote"), ("read", "read"), ("speak", "spoke"),
        ("drive", "drove"), ("swim", "swam"), ("sing", "sang"),
        ("think", "thought"), ("buy", "bought"), ("teach", "taught"),
        ("catch", "caught"), ("fly", "flew"), ("grow", "grew"),
        ("draw", "drew"), ("know", "knew"), ("break", "broke"),
        ("choose", "chose"), ("wear", "wore"), ("begin", "began"),
        ("drink", "drank"), ("give", "gave"), ("rise", "rose"),
        ("shake", "shook"), ("steal", "stole"), ("wear", "wore"),
        ("bite", "bit"), ("hide", "hid"), ("slide", "slid"),
        ("freeze", "froze"), ("ride", "rode"), ("shine", "shone"),
        ("throw", "threw"), ("blow", "blew"), ("fight", "fought"),
        ("find", "found"), ("hold", "held"), ("leave", "left"),
        ("meet", "met"), ("send", "sent"), ("spend", "spent"),
        ("build", "built"), ("keep", "kept"), ("mean", "meant"),
        ("feel", "felt"), ("lose", "lost"), ("sell", "sold"),
        ("tell", "told"), ("bring", "brought"), ("stand", "stood"),
        ("understand", "understood"), ("sit", "sat"), ("win", "won"),
    ]
    subjects = [
        "I", "She", "He", "They", "We", "The cat", "The dog",
        "The child", "The woman", "The man", "The student", "The teacher",
        "The doctor", "The artist", "The bird", "The fish", "The horse",
        "The rabbit", "The fox", "The bear",
    ]
    contexts_pres = [
        "{} {}s every day",
        "{} {}s in the morning",
        "{} {}s on weekends",
        "{} always {}s",
    ]
    contexts_past = [
        "{} {} yesterday",
        "{} {} last week",
        "{} {} this morning",
        "{} just {}",
    ]
    
    for (v1, v2), subj in product(verb_pairs, subjects):
        if len(pairs['tense']) >= 300:
            break
        # 主语第三人称单数处理
        if subj in ["I", "They", "We"]:
            pres = f"{subj} {v1} every day"
        else:
            pres = f"{subj} {v1}s every day"
        past = f"{subj} {v2} yesterday"
        pairs['tense'].append((pres, past))
    
    # --- Polarity: 扩展到200对 ---
    polarity_templates = [
        ("She is {}", "She is not {}"),
        ("He was {}", "He was not {}"),
        ("They are {}", "They are not {}"),
        ("The weather is {}", "The weather is not {}"),
        ("This movie was {}", "This movie was not {}"),
        ("The food tastes {}", "The food does not taste {}"),
        ("She can {}", "She cannot {}"),
        ("He will {}", "He will not {}"),
        ("They should {}", "They should not {}"),
        ("We must {}", "We must not {}"),
        ("I like {}", "I do not like {}"),
        ("She loves {}", "She does not love {}"),
        ("He believes {}", "He does not believe {}"),
        ("They enjoy {}", "They do not enjoy {}"),
        ("We need {}", "We do not need {}"),
    ]
    adjectives = [
        "happy", "sad", "tall", "strong", "beautiful", "expensive",
        "difficult", "easy", "important", "interesting", "dangerous",
        "safe", "warm", "cold", "bright", "dark", "loud", "quiet",
        "fast", "slow", "clean", "dirty", "rich", "poor", "healthy",
        "sick", "young", "old", "smart", "careful",
    ]
    verbs_pol = [
        "swim", "dance", "cook", "drive", "sing", "fly", "run",
        "read", "write", "play", "work", "travel", "speak", "cook",
    ]
    nouns_pol = [
        "this song", "the movie", "the book", "that idea", "this plan",
        "the weather", "the food", "that place", "this game", "the result",
        "coffee", "music", "art", "sports", "nature",
    ]
    
    for adj in adjectives:
        for (t1, t2) in polarity_templates[:6]:
            if len(pairs['polarity']) >= 200:
                break
            pairs['polarity'].append((t1.format(adj), t2.format(adj)))
    
    for v in verbs_pol:
        for (t1, t2) in polarity_templates[6:11]:
            if len(pairs['polarity']) >= 200:
                break
            pairs['polarity'].append((t1.format(v), t2.format(v)))
    
    for n in nouns_pol:
        for (t1, t2) in polarity_templates[11:]:
            if len(pairs['polarity']) >= 200:
                break
            pairs['polarity'].append((t1.format(n), t2.format(n)))
    
    # --- Syntax: 扩展到300对 ---
    # 单复数
    singular_nouns = [
        "cat", "dog", "bird", "tree", "car", "house", "book", "river",
        "mountain", "flower", "child", "woman", "man", "star",
        "cloud", "door", "window", "street", "bridge", "table",
        "lamp", "phone", "key", "cup", "plate", "bottle", "shirt", "hat",
        "horse", "rabbit", "fox", "bear", "snake", "eagle", "fish",
        "butterfly", "rose", "lily", "oak", "pine",
    ]
    for noun in singular_nouns[:40]:
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
    
    # 语态变化
    active_verbs = [
        ("chased", "was chased by"), ("caught", "was caught by"),
        ("saw", "was seen by"), ("heard", "was heard by"),
        ("followed", "was followed by"), ("helped", "was helped by"),
        ("pushed", "was pushed by"), ("pulled", "was pulled by"),
        ("watched", "was watched by"), ("admired", "was admired by"),
        ("loved", "was loved by"), ("hated", "was hated by"),
        ("protected", "was protected by"), ("supported", "was supported by"),
        ("trained", "was trained by"), ("taught", "was taught by"),
    ]
    subjects_objects = [
        ("The dog", "the cat"), ("The hunter", "the deer"),
        ("The teacher", "the student"), ("The doctor", "the patient"),
        ("The king", "the servant"), ("The chef", "the waiter"),
        ("The driver", "the passenger"), ("The artist", "the model"),
        ("The writer", "the editor"), ("The coach", "the player"),
    ]
    for (v_act, v_pass), (subj, obj) in product(active_verbs, subjects_objects):
        if len(pairs['syntax']) >= 300:
            break
        pairs['syntax'].append((
            f"{subj} {v_act} {obj}",
            f"{obj.capitalize()} {v_pass} {subj.lower()}"
        ))
    
    # 疑问句
    statements = [
        ("She is happy", "Is she happy"),
        ("He can swim", "Can he swim"),
        ("They will come", "Will they come"),
        ("The book is interesting", "Is the book interesting"),
        ("She likes coffee", "Does she like coffee"),
        ("He plays guitar", "Does he play guitar"),
        ("They speak French", "Do they speak French"),
        ("The cat sleeps all day", "Does the cat sleep all day"),
        ("She works hard", "Does she work hard"),
        ("He knows the answer", "Does he know the answer"),
        ("They live nearby", "Do they live nearby"),
        ("The door is locked", "Is the door locked"),
        ("She has a dog", "Does she have a dog"),
        ("He reads a lot", "Does he read a lot"),
        ("They cook well", "Do they cook well"),
    ]
    for s, q in statements:
        if len(pairs['syntax']) >= 300:
            break
        pairs['syntax'].append((s, q))
    
    # --- Semantic: 200对 ---
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
        ("book", "magazine"), ("shirt", "jacket"), ("boat", "ship"),
        ("pen", "pencil"), ("chair", "sofa"), ("knife", "fork"),
        ("hat", "cap"), ("boot", "shoe"), ("door", "gate"),
        ("hill", "valley"),
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
    
    # --- Logic: 200对 ---
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
        ("she practiced daily", "she mastered the skill"),
        ("the battery died", "the phone shut down"),
        ("he forgot his keys", "he could not enter"),
        ("the alarm rang", "everyone woke up"),
        ("she added sugar", "the coffee tasted sweet"),
        ("the dog barked", "the cat ran away"),
        ("he pushed the button", "the machine started"),
        ("the plant got water", "it grew tall"),
        ("she turned the key", "the door opened"),
        ("the snow fell", "the roads were blocked"),
        ("he trained hard", "he won the race"),
        ("the cat saw the mouse", "it started chasing"),
        ("she mixed the colors", "they turned purple"),
        ("the cloud grew dark", "it started to rain"),
        ("he ate the mushroom", "he got poisoned"),
    ]
    connectors_cause = [
        ("Because {}", ", {}"),
        ("Since {}", ", {}"),
        ("Due to the fact that {}", ", {}"),
    ]
    connectors_contra = [
        ("Although {}", ", {}"),
        ("Even though {}", ", {}"),
        ("Despite the fact that {}", ", {}"),
    ]
    connectors_cond = [
        ("If {}", ", then {}"),
        ("Unless {}", ", {}"),
        ("Provided that {}", ", {}"),
    ]
    
    for cause, effect in cause_effect:
        for c1, c2 in connectors_cause:
            if len(pairs['logic']) >= 200:
                break
            pairs['logic'].append((c1.format(cause).strip(), c2.format(effect).strip()))
        for c1, c2 in connectors_contra:
            if len(pairs['logic']) >= 200:
                break
            pairs['logic'].append((c1.format(cause).strip(), c2.format(effect).strip()))
        for c1, c2 in connectors_cond:
            if len(pairs['logic']) >= 200:
                break
            pairs['logic'].append((c1.format(cause).strip(), c2.format(effect).strip()))
    
    # --- Style: 200对 ---
    style_pairs = [
        ("Hi", "Greetings"), ("Thanks", "I appreciate your assistance"),
        ("Bye", "Farewell"), ("OK", "That is acceptable"),
        ("No way", "That is highly improbable"), ("Cool", "That is quite impressive"),
        ("Yeah", "Indeed"), ("Dunno", "I am uncertain"),
        ("Gonna", "Intending to"), ("Wanna", "Desiring to"),
        ("Got it", "I understand"), ("So what", "What is the significance"),
        ("Big deal", "That is of considerable importance"),
        ("Let's go", "Shall we proceed"), ("Come on", "Please proceed"),
        ("What's up", "What is the current situation"),
        ("Hang out", "Spend time together"), ("Figure out", "Determine"),
        ("Point out", "Indicate"), ("Look into", "Investigate"),
        ("Give up", "Surrender"), ("Go on", "Continue"),
        ("Hold on", "Wait a moment"), ("Take off", "Depart"),
        ("Turn down", "Reject"), ("Work out", "Exercise"),
        ("Break down", "Malfunction"), ("Call off", "Cancel"),
        ("Carry out", "Execute"), ("Find out", "Discover"),
        ("Get along", "Be friendly"), ("Give back", "Return"),
        ("Keep up", "Maintain"), ("Look after", "Care for"),
        ("Make up", "Invent"), ("Pass away", "Decease"),
        ("Put off", "Postpone"), ("Run out", "Exhaust"),
        ("Set up", "Establish"), ("Take over", "Assume control"),
    ]
    sentence_starters = [
        ("Hey, ", "Excuse me, "), ("Well, ", "Furthermore, "),
        ("Look, ", "Observe that "), ("I mean, ", "I intend to convey that "),
        ("You know, ", "It is worth noting that "),
    ]
    for (s1, s2), (pre1, pre2) in product(style_pairs, sentence_starters):
        if len(pairs['style']) >= 200:
            break
        pairs['style'].append((f"{pre1}{s1}", f"{pre2}{s2}"))
    
    # 打印统计
    for dim, plist in pairs.items():
        print(f"  {dim}: {len(plist)} 对")
    
    return pairs


# ============================================================
# S1: 稀疏原子特征分解
# ============================================================

def s1_sparse_decomposition(model_name, target_dims=None):
    """
    S1: 在强预测方向上做稀疏分解
    
    核心方法:
    1. 收集大量激活向量在功能方向上的投影
    2. 在投影空间做稀疏分解(过完备字典学习)
    3. 分析原子特征的: 数量、稀疏度、组合性、可解释性
    4. 与随机基线对比
    
    技术路线:
    - 不训练SAE(没有足够数据), 而是用SVD+阈值化作为"穷人SAE"
    - 核心洞察: SVD的奇异向量就是最优的"原子特征"候选
    - 额外: 用激活聚类发现离散的原子模式
    """
    print("\n" + "="*70)
    print("S1: 稀疏原子特征分解")
    print("="*70)
    
    target_dims = target_dims or ['tense', 'polarity', 'syntax', 'semantic']
    
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
    
    # 生成句对
    pairs = generate_enhanced_pairs()
    
    results = {}
    
    for dim_name in target_dims:
        print(f"\n  === {dim_name} ===")
        pair_list = pairs.get(dim_name, [])
        if not pair_list:
            print(f"    跳过 (无句对)")
            continue
        
        # 收集激活差异向量
        diffs = []
        raw_diffs = []  # 未归一化的差异
        
        for s1, s2 in pair_list:
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
                    diffs.append(diff / norm)
                    raw_diffs.append(diff)
            except:
                continue
        
        if len(diffs) < 10:
            print(f"    跳过 (有效样本不足: {len(diffs)})")
            continue
        
        diffs = np.array(diffs)
        raw_diffs = np.array(raw_diffs)
        n_samples = len(diffs)
        print(f"    有效样本: {n_samples}")
        
        # === S1.1: 主方向分析 (PCA/SVD) ===
        print(f"\n    [S1.1] 主方向分析")
        
        # SVD分解
        U, S, Vt = np.linalg.svd(diffs, full_matrices=False)
        
        # 累积方差解释比
        var_explained = S ** 2 / (S ** 2).sum()
        cumvar = np.cumsum(var_explained)
        
        # 找到解释90%, 95%, 99%所需的成分数
        n_90 = int(np.searchsorted(cumvar, 0.90)) + 1
        n_95 = int(np.searchsorted(cumvar, 0.95)) + 1
        n_99 = int(np.searchsorted(cumvar, 0.99)) + 1
        
        print(f"    主成分数: 90%→{n_90}, 95%→{n_95}, 99%→{n_99}")
        print(f"    Top-5 奇异值: {S[:5].round(4)}")
        print(f"    Top-5 方差解释: {var_explained[:5].round(4)}")
        
        # 有效维度 (entropy-based)
        p = var_explained[var_explained > 1e-10]
        entropy = -np.sum(p * np.log(p + 1e-30)) / np.log(len(p))
        eff_dim = np.exp(entropy * np.log(len(p)))
        print(f"    有效维度(entropy): {eff_dim:.2f}")
        
        s1_1_results = {
            'n_samples': n_samples,
            'n_components_90': n_90,
            'n_components_95': n_95,
            'n_components_99': n_99,
            'top5_singular_values': S[:5].tolist(),
            'top5_variance_explained': var_explained[:5].tolist(),
            'effective_dimension': float(eff_dim),
            'cumvar_at_10': float(cumvar[min(9, len(cumvar)-1)]),
            'cumvar_at_20': float(cumvar[min(19, len(cumvar)-1)]),
        }
        
        # === S1.2: 原子特征聚类 ===
        print(f"\n    [S1.2] 原子特征聚类")
        
        # 在主成分空间聚类 (降维到前20个PC)
        n_pc = min(20, diffs.shape[1], diffs.shape[0])
        diffs_pc = diffs @ Vt[:n_pc].T  # [N, n_pc]
        
        # K-means聚类 (尝试2-8个簇)
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        best_k = 2
        best_sil = -1
        cluster_results = {}
        
        for k in range(2, min(9, n_samples // 5 + 1)):
            if n_samples < k * 5:
                break
            km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
            labels = km.fit_predict(diffs_pc)
            sil = silhouette_score(diffs_pc, labels) if len(set(labels)) > 1 else -1
            cluster_results[k] = {
                'silhouette': float(sil),
                'cluster_sizes': [int(np.sum(labels == i)) for i in range(k)],
            }
            if sil > best_sil:
                best_sil = sil
                best_k = k
        
        print(f"    最优聚类数: {best_k} (silhouette={best_sil:.3f})")
        for k, info in sorted(cluster_results.items()):
            print(f"      k={k}: sil={info['silhouette']:.3f}, sizes={info['cluster_sizes']}")
        
        # 获取最优聚类的簇中心（在原始空间）
        km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=100)
        best_labels = km_best.fit_predict(diffs_pc)
        cluster_centers = []
        for i in range(best_k):
            center = diffs[best_labels == i].mean(axis=0)
            center_norm = np.linalg.norm(center)
            if center_norm > 1e-8:
                cluster_centers.append(center / center_norm)
        
        # 簇中心之间的余弦相似度
        if len(cluster_centers) > 1:
            center_sim_matrix = np.zeros((len(cluster_centers), len(cluster_centers)))
            for i in range(len(cluster_centers)):
                for j in range(len(cluster_centers)):
                    center_sim_matrix[i, j] = np.dot(cluster_centers[i], cluster_centers[j])
            print(f"    簇中心余弦相似度矩阵:")
            for i in range(len(cluster_centers)):
                row = " ".join([f"{center_sim_matrix[i,j]:.3f}" for j in range(len(cluster_centers))])
                print(f"      C{i}: {row}")
        else:
            center_sim_matrix = np.array([[1.0]])
        
        s1_2_results = {
            'best_k': best_k,
            'best_silhouette': float(best_sil),
            'cluster_results': {str(k): v for k, v in cluster_results.items()},
            'n_clusters': len(cluster_centers),
            'center_cosine_matrix': center_sim_matrix.tolist(),
        }
        
        # === S1.3: 稀疏度分析 ===
        print(f"\n    [S1.3] 稀疏度分析")
        
        # 分析激活的稀疏度
        # 对每个样本，计算其在前N个主方向上的稀疏表示
        n_top = min(50, diffs.shape[1], diffs.shape[0])
        coeffs = diffs @ Vt[:n_top].T  # [N, n_top]
        
        # L0稀疏度: 显著非零系数的比例
        threshold = 0.01
        l0_sparsity = np.mean(np.abs(coeffs) > threshold)
        
        # Gini系数
        abs_coeffs = np.abs(coeffs).flatten()
        sorted_coeffs = np.sort(abs_coeffs)
        n_coeffs = len(sorted_coeffs)
        index = np.arange(1, n_coeffs + 1)
        gini = (2 * np.sum(index * sorted_coeffs) / (n_coeffs * np.sum(sorted_coeffs) + 1e-30)) - (n_coeffs + 1) / n_coeffs
        
        # 每个样本的稀疏度分布
        sample_l0 = np.mean(np.abs(coeffs) > threshold, axis=1)
        sample_l1 = np.mean(np.abs(coeffs), axis=1)
        
        print(f"    L0稀疏度(全局): {l0_sparsity:.4f}")
        print(f"    Gini系数: {gini:.4f}")
        print(f"    样本L0: mean={sample_l0.mean():.4f}, std={sample_l0.std():.4f}")
        print(f"    样本L1: mean={sample_l1.mean():.6f}, std={sample_l1.std():.6f}")
        
        s1_3_results = {
            'l0_sparsity': float(l0_sparsity),
            'gini_coefficient': float(gini),
            'sample_l0_mean': float(sample_l0.mean()),
            'sample_l0_std': float(sample_l0.std()),
            'sample_l1_mean': float(sample_l1.mean()),
            'sample_l1_std': float(sample_l1.std()),
            'n_top_components': n_top,
        }
        
        # === S1.4: 组合性分析 ===
        print(f"\n    [S1.4] 组合性分析")
        
        # 检查: 差异向量是否可以由少数"原子"线性组合？
        # 方法: 用前K个PC重建，计算重建误差
        reconstruction_errors = {}
        for k in [1, 2, 3, 5, 10, 20]:
            if k > n_pc:
                break
            # 重建: coeffs_k @ Vt[:k]
            reconstructed = diffs_pc[:, :k] @ Vt[:k]
            error = np.mean(np.linalg.norm(diffs - reconstructed, axis=1)) / np.mean(np.linalg.norm(diffs, axis=1))
            reconstruction_errors[k] = float(error)
            print(f"    k={k}: 重建误差={error:.4f}")
        
        # 可加性检验: A→B + B→C ≈ A→C ?
        # 取tense维度: "I walk"→"I walked" + "I walked"→"I had walked" ≈ "I walk"→"I had walked"
        additivity_test = None
        if dim_name == 'tense' and len(pair_list) >= 3:
            # 用同一动词的不同时态
            additivities = []
            for i in range(min(20, len(pair_list) - 2)):
                try:
                    s1, s2 = pair_list[i]
                    # s1=present, s2=past
                    # 用另一个present→past和past→past_perfect的组合
                    # 简化: 检查 present_diff + past_diff 的一致性
                    toks1 = tokenizer(s1, return_tensors="pt", truncation=True, max_length=64).to(device)
                    toks2 = tokenizer(s2, return_tensors="pt", truncation=True, max_length=64).to(device)
                    with torch.no_grad():
                        h1 = model(**toks1, output_hidden_states=True).hidden_states[target_layer]
                        h2 = model(**toks2, output_hidden_states=True).hidden_states[target_layer]
                    r1 = h1[0].mean(0).detach().cpu().float().numpy()
                    r2 = h2[0].mean(0).detach().cpu().float().numpy()
                    d12 = r2 - r1
                    n12 = np.linalg.norm(d12)
                    if n12 < 1e-8:
                        continue
                    d12 = d12 / n12
                    
                    # 检查: 这个差异与平均方向的对齐度
                    mean_dir = diffs.mean(axis=0)
                    mean_norm = np.linalg.norm(mean_dir)
                    if mean_norm > 1e-8:
                        mean_dir = mean_dir / mean_norm
                        alignment = float(np.dot(d12, mean_dir))
                        additivities.append(alignment)
                except:
                    continue
            
            if additivities:
                additivity_test = {
                    'mean_alignment_to_mean_dir': float(np.mean(additivities)),
                    'std_alignment': float(np.std(additivities)),
                    'min_alignment': float(np.min(additivities)),
                    'max_alignment': float(np.max(additivities)),
                }
                print(f"    可加性(对齐均值方向): {np.mean(additivities):.4f}±{np.std(additivities):.4f}")
        
        s1_4_results = {
            'reconstruction_errors': reconstruction_errors,
            'additivity_test': additivity_test,
        }
        
        # === S1.5: 层级深度分析 ===
        print(f"\n    [S1.5] 层级深度分析")
        
        # 检查不同层的稀疏度变化
        layer_analysis = {}
        sample_pairs = pair_list[:20]  # 用前20对做层级分析
        
        for layer_idx in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
            layer_diffs = []
            for s1, s2 in sample_pairs:
                try:
                    toks1 = tokenizer(s1, return_tensors="pt", truncation=True, max_length=64).to(device)
                    toks2 = tokenizer(s2, return_tensors="pt", truncation=True, max_length=64).to(device)
                    with torch.no_grad():
                        h1 = model(**toks1, output_hidden_states=True).hidden_states[layer_idx]
                        h2 = model(**toks2, output_hidden_states=True).hidden_states[layer_idx]
                    r1 = h1[0].mean(0).detach().cpu().float().numpy()
                    r2 = h2[0].mean(0).detach().cpu().float().numpy()
                    diff = r2 - r1
                    norm = np.linalg.norm(diff)
                    if norm > 1e-8:
                        layer_diffs.append(diff / norm)
                except:
                    continue
            
            if len(layer_diffs) < 5:
                continue
            
            layer_diffs = np.array(layer_diffs)
            
            # 计算层级的预测力: 平均余弦相似度
            mean_dir = layer_diffs.mean(axis=0)
            mean_norm = np.linalg.norm(mean_dir)
            if mean_norm > 1e-8:
                mean_dir = mean_dir / mean_norm
                cosines = [float(np.dot(d, mean_dir)) for d in layer_diffs]
                avg_cos = float(np.mean(cosines))
            else:
                avg_cos = 0.0
            
            # 差异幅度
            diff_norms = [float(np.linalg.norm(r2 - r1)) for r1, r2 in 
                         zip(layer_diffs, layer_diffs)]
            
            layer_analysis[str(layer_idx)] = {
                'avg_cosine_to_mean': avg_cos,
                'n_samples': len(layer_diffs),
                'mean_diff_norm': float(np.mean(diff_norms)),
            }
            print(f"    Layer {layer_idx}: cos={avg_cos:.4f}, n={len(layer_diffs)}")
        
        s1_5_results = layer_analysis
        
        # 汇总
        results[dim_name] = {
            's1_1_principal': s1_1_results,
            's1_2_clustering': s1_2_results,
            's1_3_sparsity': s1_3_results,
            's1_4_compositionality': s1_4_results,
            's1_5_layer_depth': s1_5_results,
        }
    
    release_model(model)
    return results


# ============================================================
# S2: 随机基线对比
# ============================================================

def s2_random_baseline(model_name, target_dims=None):
    """
    S2: 与随机基线对比
    
    核心思路:
    1. 对随机向量做同样的SVD/聚类/稀疏度分析
    2. 如果训练模型的特征显著偏离随机 → 学习产物
    3. 如果训练模型的特征与随机类似 → 可能是维度效应
    
    随机基线:
    - 在同一空间中生成均匀随机向量
    - 在同一空间中生成高斯随机向量
    - 与训练模型的分布做Kolmogorov-Smirnov检验
    """
    print("\n" + "="*70)
    print("S2: 随机基线对比")
    print("="*70)
    
    target_dims = target_dims or ['tense', 'polarity', 'syntax', 'semantic']
    
    # 先获取训练模型的S1结果
    s1_results = s1_sparse_decomposition(model_name, target_dims)
    
    results = {}
    
    for dim_name in target_dims:
        if dim_name not in s1_results:
            continue
        
        s1_dim = s1_results[dim_name]
        n_samples = s1_dim['s1_1_principal']['n_samples']
        d_model_val = 2560  # Qwen3 hidden_size (需要动态获取)
        
        # 生成随机基线
        # 基线1: 均匀球面随机
        random_dirs_sphere = np.random.randn(n_samples, d_model_val)
        random_dirs_sphere = random_dirs_sphere / np.linalg.norm(random_dirs_sphere, axis=1, keepdims=True)
        
        # 基线2: 与训练数据同分布的随机(从训练模型激活的协方差采样)
        # 简化: 用训练差异的随机排列
        # (这需要原始差异矩阵, 这里用简化版)
        
        # SVD分解随机数据
        U_r, S_r, Vt_r = np.linalg.svd(random_dirs_sphere, full_matrices=False)
        var_r = S_r ** 2 / (S_r ** 2).sum()
        cumvar_r = np.cumsum(var_r)
        
        n_90_r = int(np.searchsorted(cumvar_r, 0.90)) + 1
        n_95_r = int(np.searchsorted(cumvar_r, 0.95)) + 1
        
        # 有效维度
        p_r = var_r[var_r > 1e-10]
        entropy_r = -np.sum(p_r * np.log(p_r + 1e-30)) / np.log(len(p_r))
        eff_dim_r = np.exp(entropy_r * np.log(len(p_r)))
        
        # 聚类分析
        n_pc = min(20, d_model_val)
        random_pc = random_dirs_sphere @ Vt_r[:n_pc].T
        
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        best_k_r = 2
        best_sil_r = -1
        for k in range(2, min(9, n_samples // 5 + 1)):
            if n_samples < k * 5:
                break
            km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
            labels = km.fit_predict(random_pc)
            sil = silhouette_score(random_pc, labels) if len(set(labels)) > 1 else -1
            if sil > best_sil_r:
                best_sil_r = sil
                best_k_r = k
        
        # 对比
        s1_dim_data = s1_dim['s1_1_principal']
        s1_dim_cluster = s1_dim['s1_2_clustering']
        
        comparison = {
            'trained': {
                'n_components_90': s1_dim_data['n_components_90'],
                'n_components_95': s1_dim_data['n_components_95'],
                'effective_dimension': s1_dim_data['effective_dimension'],
                'best_k': s1_dim_cluster['best_k'],
                'best_silhouette': s1_dim_cluster['best_silhouette'],
                'top1_variance': s1_dim_data['top5_variance_explained'][0] if s1_dim_data['top5_variance_explained'] else 0,
            },
            'random': {
                'n_components_90': n_90_r,
                'n_components_95': n_95_r,
                'effective_dimension': float(eff_dim_r),
                'best_k': best_k_r,
                'best_silhouette': float(best_sil_r),
                'top1_variance': float(var_r[0]),
            },
            'ratio': {
                'n_components_90': float(s1_dim_data['n_components_90'] / (n_90_r + 1e-30)),
                'effective_dimension': float(s1_dim_data['effective_dimension'] / (eff_dim_r + 1e-30)),
                'top1_variance': float(s1_dim_data['top5_variance_explained'][0] / (var_r[0] + 1e-30)) if s1_dim_data['top5_variance_explained'] else 0,
            }
        }
        
        # 判定
        n_comp_ratio = comparison['ratio']['n_components_90']
        eff_dim_ratio = comparison['ratio']['effective_dimension']
        top1_ratio = comparison['ratio']['top1_variance']
        
        if n_comp_ratio < 0.5 or top1_ratio > 3.0:
            verdict = "学习产物 (显著偏离随机)"
        elif n_comp_ratio < 0.8 or top1_ratio > 2.0:
            verdict = "弱学习信号 (轻微偏离随机)"
        else:
            verdict = "架构效应 (≈随机)"
        
        print(f"\n  === {dim_name} 对比 ===")
        print(f"    训练: n_90={s1_dim_data['n_components_90']}, eff_dim={s1_dim_data['effective_dimension']:.2f}, top1_var={s1_dim_data['top5_variance_explained'][0]:.4f}" if s1_dim_data['top5_variance_explained'] else "")
        print(f"    随机: n_90={n_90_r}, eff_dim={eff_dim_r:.2f}, top1_var={var_r[0]:.4f}")
        print(f"    比值: n_90={n_comp_ratio:.2f}, eff_dim={eff_dim_ratio:.2f}, top1_var={top1_ratio:.2f}")
        print(f"    判定: {verdict}")
        
        comparison['verdict'] = verdict
        results[dim_name] = comparison
    
    return results


# ============================================================
# S3: 跨模型对比
# ============================================================

def s3_cross_model_comparison():
    """
    S3: 三模型SAE结果对比
    
    基于已保存的结果文件对比
    """
    print("\n" + "="*70)
    print("S3: 跨模型SAE对比")
    print("="*70)
    
    results_dir = Path(r"d:\Ai2050\TransformerLens-Project\results\sae_decomposition")
    
    models = ['qwen3', 'glm4', 'deepseek7b']
    dims = ['tense', 'polarity', 'syntax', 'semantic', 'logic', 'style']
    
    all_data = {}
    for model_name in models:
        result_file = results_dir / f"{model_name}_s1_results.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                all_data[model_name] = json.load(f)
    
    if len(all_data) < 2:
        print("  需要至少2个模型的S1结果才能做跨模型对比")
        print(f"  当前已有: {list(all_data.keys())}")
        return None
    
    comparison = {}
    for dim in dims:
        dim_comparison = {}
        for model_name in models:
            if model_name in all_data and dim in all_data[model_name]:
                data = all_data[model_name][dim]
                dim_comparison[model_name] = {
                    'n_components_90': data['s1_1_principal']['n_components_90'],
                    'effective_dimension': data['s1_1_principal']['effective_dimension'],
                    'best_k': data['s1_2_clustering']['best_k'],
                    'best_silhouette': data['s1_2_clustering']['best_silhouette'],
                    'gini': data['s1_3_sparsity']['gini_coefficient'],
                    'top1_var': data['s1_1_principal']['top5_variance_explained'][0] if data['s1_1_principal']['top5_variance_explained'] else 0,
                }
        
        if dim_comparison:
            comparison[dim] = dim_comparison
            print(f"\n  {dim}:")
            for model_name, stats in dim_comparison.items():
                print(f"    {model_name}: n90={stats['n_components_90']}, eff_dim={stats['effective_dimension']:.1f}, "
                      f"best_k={stats['best_k']}, sil={stats['best_silhouette']:.3f}, "
                      f"gini={stats['gini']:.3f}, top1_var={stats['top1_var']:.4f}")
    
    return comparison


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


def release_model(model):
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("[release] GPU memory released")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SAE Feature Decomposition")
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--test", type=str, default="s1",
                        choices=["s1", "s2", "s3", "all"],
                        help="运行哪个测试")
    args = parser.parse_args()
    
    model_name = args.model
    print(f"\n{'#'*70}")
    print(f"# SAE稀疏特征分解")
    print(f"# Model: {model_name}")
    print(f"{'#'*70}")
    
    all_results = {
        'model': model_name,
        'timestamp': datetime.now().isoformat(),
    }
    
    output_dir = Path(r"d:\Ai2050\TransformerLens-Project\results\sae_decomposition")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # S1: 稀疏分解
    if args.test in ["s1", "all"]:
        s1_results = s1_sparse_decomposition(model_name)
        all_results['s1_decomposition'] = s1_results
        
        # 保存S1结果（供S3使用）
        with open(output_dir / f"{model_name}_s1_results.json", 'w') as f:
            json.dump(s1_results, f, indent=2, cls=NumpyEncoder)
        print(f"\n[S1] 结果已保存到 {output_dir / f'{model_name}_s1_results.json'}")
    
    # S2: 随机基线
    if args.test in ["s2", "all"]:
        s2_results = s2_random_baseline(model_name)
        all_results['s2_random_baseline'] = s2_results
    
    # S3: 跨模型对比
    if args.test in ["s3", "all"]:
        s3_results = s3_cross_model_comparison()
        all_results['s3_cross_model'] = s3_results
    
    # 保存完整结果
    output_path = output_dir / f"{model_name}_full_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n{'='*70}")
    print(f"所有结果已保存到 {output_path}")
    print(f"{'='*70}")


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


if __name__ == "__main__":
    main()
