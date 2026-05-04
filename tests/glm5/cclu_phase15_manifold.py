"""
CCL-U(Phase 15): 语法流形的精细结构与位置-语法交互
=============================================================================
核心问题(基于Phase 14的发现):
  1. 4类语法角色在3维流形上的精确几何结构是什么?
     → nsubj/poss重合, dobj反向(127°), amod侧面
     → 是否形成四面体? 三角锥? 线段?
  
  2. 位置编码与语法编码如何分离?
     → 同一词在nsubj/dobj位置时, 减去位置编码后语法信息是否保留?
     → 位置编码是加法嵌入还是乘法调制?
  
  3. ICA(斜交分解)是否能更好地分离语法角色?
     → 正交分解(PCA)损失28-50%信息
     → ICA是否保留更多信息?
  
  4. 语法角色在多层中的演化轨迹是什么?
     → 语法信息在哪些层开始显现?
     → nsubj-poss等价在各层中是否一致?

实验:
  Exp1: ★★★★★★★ 3维语法流形的精细刻画
    → 6类语法角色(nsubj/poss/dobj/amod/advmod/pobj)在3D PCA空间中的精确位置
    → 几何关系: 角度, 距离, 体积, 形状
    → 使用主动/被动语态对来增加dobj样本

  Exp2: ★★★★★★★ 位置编码与语法编码的分离
    → 主动/被动语态: "The cat chased the dog" vs "The dog was chased by the cat"
    → 同一词(cat)在nsubj位置 vs dobj位置的表示差异
    → 减去平均位置向量后, 语法角色信息是否保留
    → 位置编码的加法假设检验

  Exp3: ★★★★★ ICA斜交分解
    → FastICA分离6类语法角色的表示
    → 对比ICA vs PCA的信息保留率
    → ICA分量的可解释性

  Exp4: ★★★★★★★ 语法角色在多层中的演化
    → 在8个采样层中收集hidden states
    → 语法角色分类CV随层变化
    → nsubj-poss等价在各层中的变化
    → 语法流形维度随层变化
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull

from model_utils import load_model, get_layers, get_model_info, release_model, safe_decode


# ===== Exp1: 3维语法流形精细刻画 =====
# 6类语法角色, 增加pobj(介词宾语)和advmod(状语修饰)

MANIFOLD_ROLES_DATA = {
    "nsubj": {
        "sentences": [
            "The king ruled the kingdom wisely",
            "The doctor treated the patient carefully",
            "The artist painted the portrait beautifully",
            "The soldier defended the castle bravely",
            "The teacher explained the lesson clearly",
            "The chef cooked the meal perfectly",
            "The cat chased the mouse quickly",
            "The dog found the bone happily",
            "The woman drove the car safely",
            "The man fixed the roof carefully",
            "The student read the book quietly",
            "The singer performed the song brilliantly",
            "The baker made the bread daily",
            "The pilot flew the plane smoothly",
            "The farmer grew the crops diligently",
            "The writer wrote the novel slowly",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "teacher",
            "chef", "cat", "dog", "woman", "man",
            "student", "singer", "baker", "pilot", "farmer", "writer",
        ],
    },
    "poss": {
        "sentences": [
            "The king's crown glittered brightly",
            "The doctor's office opened early",
            "The artist's studio looked beautiful",
            "The soldier's uniform was clean",
            "The teacher's book sold quickly",
            "The chef's restaurant opened today",
            "The cat's tail swished gently",
            "The dog's bark echoed loudly",
            "The woman's dress looked elegant",
            "The man's car drove fast",
            "The student's essay read well",
            "The singer's voice rang clearly",
            "The baker's shop smelled wonderful",
            "The pilot's license was renewed",
            "The farmer's land was fertile",
            "The writer's pen wrote smoothly",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "teacher",
            "chef", "cat", "dog", "woman", "man",
            "student", "singer", "baker", "pilot", "farmer", "writer",
        ],
    },
    "dobj": {
        "sentences": [
            "They crowned the king yesterday",
            "She visited the doctor recently",
            "He admired the artist greatly",
            "We honored the soldier today",
            "You thanked the teacher warmly",
            "The customer tipped the chef generously",
            "The hawk chased the cat swiftly",
            "The boy found the dog outside",
            "The police arrested the woman quickly",
            "The company hired the man recently",
            "I praised the student loudly",
            "They applauded the singer warmly",
            "She visited the baker often",
            "He admired the pilot greatly",
            "We thanked the farmer sincerely",
            "The editor praised the writer highly",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "teacher",
            "chef", "cat", "dog", "woman", "man",
            "student", "singer", "baker", "pilot", "farmer", "writer",
        ],
    },
    "amod": {
        "sentences": [
            "The brave king fought hard",
            "The kind doctor helped many",
            "The creative artist worked well",
            "The strong soldier marched far",
            "The wise teacher explained clearly",
            "The skilled chef cooked perfectly",
            "The quick cat ran fast",
            "The loyal dog stayed close",
            "The old woman walked slowly",
            "The tall man stood quietly",
            "The bright student read carefully",
            "The talented singer performed brilliantly",
            "The patient baker waited calmly",
            "The careful pilot landed smoothly",
            "The hardworking farmer harvested early",
            "The thoughtful writer reflected deeply",
        ],
        "target_words": [
            "brave", "kind", "creative", "strong", "wise",
            "skilled", "quick", "loyal", "old", "tall",
            "bright", "talented", "patient", "careful", "hardworking", "thoughtful",
        ],
    },
    "advmod": {
        "sentences": [
            "The king ruled wisely forever",
            "The doctor worked carefully always",
            "The artist painted beautifully daily",
            "The soldier fought bravely there",
            "The teacher spoke clearly again",
            "The chef worked quickly then",
            "The cat ran swiftly home",
            "The dog barked loudly today",
            "The woman drove slowly forward",
            "The man spoke quietly now",
            "The student studied carefully alone",
            "The singer performed brilliantly tonight",
            "The baker baked freshly daily",
            "The pilot flew steadily onward",
            "The farmer worked diligently always",
            "The writer typed quickly away",
        ],
        "target_words": [
            "wisely", "carefully", "beautifully", "bravely", "clearly",
            "quickly", "swiftly", "loudly", "slowly", "quietly",
            "carefully", "brilliantly", "freshly", "steadily", "diligently", "quickly",
        ],
    },
    "pobj": {
        "sentences": [
            "They looked at the king closely",
            "She waited for the doctor patiently",
            "He thought about the artist often",
            "We marched toward the soldier steadily",
            "You listened to the teacher attentively",
            "They paid the chef generously",
            "She played with the cat happily",
            "He walked toward the dog slowly",
            "The gift belonged to the woman originally",
            "The letter was for the man personally",
            "I read about the student recently",
            "They talked about the singer excitedly",
            "She ordered from the baker regularly",
            "He flew with the pilot recently",
            "We learned from the farmer carefully",
            "They wrote about the writer frequently",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "teacher",
            "chef", "cat", "dog", "woman", "man",
            "student", "singer", "baker", "pilot", "farmer", "writer",
        ],
    },
}


# ===== Exp2: 位置编码与语法编码的分离 =====
# 使用主动/被动语态对: 同一个词在nsubj位置 vs dobj位置

VOICE_PAIR_DATA = {
    "active_nsubj": {
        "desc": "主动语态主语: The N V the M",
        "sentences": [
            ("The cat chased the mouse quickly", "cat", "nsubj"),
            ("The dog found the bone happily", "dog", "nsubj"),
            ("The king ruled the kingdom wisely", "king", "nsubj"),
            ("The doctor treated the patient carefully", "doctor", "nsubj"),
            ("The artist painted the portrait beautifully", "artist", "nsubj"),
            ("The soldier defended the castle bravely", "soldier", "nsubj"),
            ("The teacher explained the lesson clearly", "teacher", "nsubj"),
            ("The chef cooked the meal perfectly", "chef", "nsubj"),
            ("The woman drove the car safely", "woman", "nsubj"),
            ("The man fixed the roof carefully", "man", "nsubj"),
            ("The student read the book quietly", "student", "nsubj"),
            ("The writer wrote the novel slowly", "writer", "nsubj"),
        ],
    },
    "active_dobj": {
        "desc": "主动语态宾语: They V-ed the N",
        "sentences": [
            ("The hawk chased the cat swiftly", "cat", "dobj"),
            ("The boy found the dog outside", "dog", "dobj"),
            ("They crowned the king yesterday", "king", "dobj"),
            ("She visited the doctor recently", "doctor", "dobj"),
            ("He admired the artist greatly", "artist", "dobj"),
            ("We honored the soldier today", "soldier", "dobj"),
            ("You thanked the teacher warmly", "teacher", "dobj"),
            ("The customer tipped the chef generously", "chef", "dobj"),
            ("The police arrested the woman quickly", "woman", "dobj"),
            ("The company hired the man recently", "man", "dobj"),
            ("I praised the student loudly", "student", "dobj"),
            ("The editor praised the writer highly", "writer", "dobj"),
        ],
    },
    # 被动语态: 同一个词变成主语
    "passive_nsubj": {
        "desc": "被动语态主语: The N was V-ed",
        "sentences": [
            ("The cat was chased by the hawk", "cat", "nsubj_passive"),
            ("The dog was found by the boy", "dog", "nsubj_passive"),
            ("The king was crowned by the priest", "king", "nsubj_passive"),
            ("The doctor was visited by patients", "doctor", "nsubj_passive"),
            ("The artist was admired by many", "artist", "nsubj_passive"),
            ("The soldier was honored by the nation", "soldier", "nsubj_passive"),
            ("The teacher was thanked by students", "teacher", "nsubj_passive"),
            ("The chef was tipped by customers", "chef", "nsubj_passive"),
            ("The woman was arrested by police", "woman", "nsubj_passive"),
            ("The man was hired by the company", "man", "nsubj_passive"),
            ("The student was praised by teachers", "student", "nsubj_passive"),
            ("The writer was praised by critics", "writer", "nsubj_passive"),
        ],
    },
    # 位置控制: 同一词在同一位置但不同语法角色
    "position_control": {
        "desc": "位置控制: pos=1, 不同语法角色",
        "sentences": [
            ("The cat chased the mouse", "cat", "nsubj_pos1"),
            ("The cat's tail swished gently", "cat", "poss_pos1"),
            ("The king ruled the kingdom", "king", "nsubj_pos1"),
            ("The king's crown glittered", "king", "poss_pos1"),
            ("The dog found the bone", "dog", "nsubj_pos1"),
            ("The dog's bark echoed", "dog", "poss_pos1"),
            ("The teacher explained the lesson", "teacher", "nsubj_pos1"),
            ("The teacher's book sold well", "teacher", "poss_pos1"),
            ("The chef cooked the meal", "chef", "nsubj_pos1"),
            ("The chef's restaurant opened", "chef", "poss_pos1"),
            ("The woman drove the car", "woman", "nsubj_pos1"),
            ("The woman's dress looked elegant", "woman", "poss_pos1"),
        ],
    },
}


# ===== Exp4: 多层演化 =====
# 复用MANIFOLD_ROLES_DATA, 在多层收集hidden states


def find_token_index(tokens, word):
    word_lower = word.lower()
    word_start = word_lower[:3]
    for i, tok in enumerate(tokens):
        tok_lower = tok.lower().strip()
        if word_lower in tok_lower or tok_lower.startswith(word_start):
            return i
    for i, tok in enumerate(tokens):
        if word_lower[:2] in tok.lower():
            return i
    return None


def collect_hs_at_layer(model, tokenizer, device, sentences, target_words, layer_idx):
    """在指定层收集target token的hidden states"""
    layers = get_layers(model)
    if layer_idx >= len(layers):
        layer_idx = len(layers) - 1
    target_layer = layers[layer_idx]

    all_h = []
    valid_words = []

    for sent, target_word in zip(sentences, target_words):
        toks = tokenizer(sent, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]

        dep_idx = find_token_index(tokens_list, target_word)
        if dep_idx is None:
            continue

        captured = {}
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured['h'] = output[0].detach().float().cpu().numpy()
            else:
                captured['h'] = output.detach().float().cpu().numpy()

        h_handle = target_layer.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = model(**toks)
        h_handle.remove()

        if 'h' not in captured:
            continue

        h_vec = captured['h'][0, dep_idx, :]
        all_h.append(h_vec)
        valid_words.append(target_word)

    return np.array(all_h) if all_h else None


def collect_hs_at_position(model, tokenizer, device, sent, target_word, layer_idx):
    """在指定层收集一个句子中target token和所有位置的hidden states"""
    layers = get_layers(model)
    if layer_idx >= len(layers):
        layer_idx = len(layers) - 1
    target_layer = layers[layer_idx]

    toks = tokenizer(sent, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
    seq_len = len(tokens_list)

    dep_idx = find_token_index(tokens_list, target_word)
    if dep_idx is None:
        return None, None, None

    captured = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured['h'] = output[0].detach().float().cpu().numpy()
        else:
            captured['h'] = output.detach().float().cpu().numpy()

    h_handle = target_layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(**toks)
    h_handle.remove()

    if 'h' not in captured:
        return None, None, None

    h_all = captured['h'][0]  # [seq_len, d_model]
    return h_all, dep_idx, seq_len


# ===== Exp1: 3维语法流形精细刻画 =====
def exp1_manifold_geometry(model, tokenizer, device):
    """6类语法角色在3D PCA空间中的精确几何"""
    print("\n" + "="*70)
    print("Exp1: 3维语法流形的精细刻画 ★★★★★★★")
    print("="*70)

    results = {}
    role_names = list(MANIFOLD_ROLES_DATA.keys())

    # 收集6类语法角色的hidden states (最后一层)
    all_h = []
    all_labels = []
    all_words = []

    for role_idx, role in enumerate(role_names):
        data = MANIFOLD_ROLES_DATA[role]
        H = collect_hs_at_layer(model, tokenizer, device,
                                data["sentences"], data["target_words"], -1)
        if H is not None and len(H) > 0:
            all_h.append(H)
            all_labels.extend([role_idx] * len(H))
            all_words.extend(data["target_words"][:len(H)])
            print(f"  {role}: {len(H)} samples")

    if len(all_h) < 6:
        print("  样本不足!")
        return results

    H_all = np.vstack(all_h)
    labels = np.array(all_labels)
    print(f"  Total: {len(H_all)} samples, dim={H_all.shape[1]}")

    # Part A: 全空间分类
    print(f"\n  Part A: 全空间6类语法角色分类")
    scaler = StandardScaler()
    H_scaled = scaler.fit_transform(H_all)
    probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
    cv = cross_val_score(probe, H_scaled, labels, cv=5, scoring='accuracy')
    print(f"    6类CV: {cv.mean():.4f}")
    results['full_6class_cv'] = float(cv.mean())

    # Part B: PCA 3D空间中的几何
    print(f"\n  Part B: PCA 3D空间中的几何结构")
    pca3 = PCA(n_components=3)
    H_3d = pca3.fit_transform(H_scaled)
    var_3d = sum(pca3.explained_variance_ratio_)
    print(f"    3D方差保留: {var_3d:.4f}")

    # 计算各类中心
    centers_3d = {}
    for role_idx, role in enumerate(role_names):
        mask = labels == role_idx
        center = np.mean(H_3d[mask], axis=0)
        centers_3d[role] = center
        spread = np.mean(np.linalg.norm(H_3d[mask] - center, axis=1))
        print(f"    {role}: center={center}, spread={spread:.4f}")
        results[f'{role}_center_3d'] = center.tolist()
        results[f'{role}_spread_3d'] = float(spread)

    # Part C: 中心间距离和角度
    print(f"\n  Part C: 中心间距离和角度")
    center_matrix = np.array([centers_3d[r] for r in role_names])
    
    # 欧氏距离矩阵
    dist_matrix = squareform(pdist(center_matrix))
    print(f"\n    欧氏距离矩阵:")
    header = "          " + "  ".join(f"{r:>8}" for r in role_names)
    print(f"    {header}")
    for i, r1 in enumerate(role_names):
        row = f"    {r1:>8}"
        for j, r2 in enumerate(role_names):
            row += f"  {dist_matrix[i,j]:>8.4f}"
        print(row)
    results['center_dist_matrix'] = dist_matrix.tolist()

    # 余弦相似度矩阵
    cos_matrix = cosine_similarity(center_matrix)
    print(f"\n    余弦相似度矩阵:")
    for i, r1 in enumerate(role_names):
        for j, r2 in enumerate(role_names):
            if j > i:
                angle = np.degrees(np.arccos(np.clip(cos_matrix[i,j], -1, 1)))
                print(f"    {r1} vs {r2}: cos={cos_matrix[i,j]:.4f}, angle={angle:.1f}°")
    results['center_cos_matrix'] = cos_matrix.tolist()

    # Part D: 几何形状分析
    print(f"\n  Part D: 几何形状分析")
    
    # nsubj-poss距离(应该≈0)
    nsubj_poss_dist = np.linalg.norm(centers_3d['nsubj'] - centers_3d['poss'])
    nsubj_dobj_dist = np.linalg.norm(centers_3d['nsubj'] - centers_3d['dobj'])
    nsubj_amod_dist = np.linalg.norm(centers_3d['nsubj'] - centers_3d['amod'])
    nsubj_pobj_dist = np.linalg.norm(centers_3d['nsubj'] - centers_3d['pobj'])
    nsubj_advmod_dist = np.linalg.norm(centers_3d['nsubj'] - centers_3d['advmod'])
    
    print(f"    nsubj-poss dist: {nsubj_poss_dist:.4f} (应该≈0)")
    print(f"    nsubj-dobj dist: {nsubj_dobj_dist:.4f}")
    print(f"    nsubj-amod dist: {nsubj_amod_dist:.4f}")
    print(f"    nsubj-pobj dist: {nsubj_pobj_dist:.4f}")
    print(f"    nsubj-advmod dist: {nsubj_advmod_dist:.4f}")
    
    results['nsubj_poss_dist'] = float(nsubj_poss_dist)
    results['nsubj_dobj_dist'] = float(nsubj_dobj_dist)
    results['nsubj_amod_dist'] = float(nsubj_amod_dist)
    results['nsubj_pobj_dist'] = float(nsubj_pobj_dist)
    results['nsubj_advmod_dist'] = float(nsubj_advmod_dist)

    # 名词角色(nsubj/poss/dobj/pobj)是否共面?
    noun_centers = np.array([centers_3d[r] for r in ['nsubj', 'poss', 'dobj', 'pobj']])
    # 4个点的体积(四面体)
    try:
        v1 = noun_centers[1] - noun_centers[0]
        v2 = noun_centers[2] - noun_centers[0]
        v3 = noun_centers[3] - noun_centers[0]
        volume = abs(np.dot(v1, np.cross(v2, v3))) / 6.0
        print(f"    名词角色四面体体积: {volume:.6f}")
        results['noun_tetrahedron_volume'] = float(volume)
    except:
        pass

    # nsubj/poss vs dobj的方向
    nsubj_dobj_vec = centers_3d['dobj'] - centers_3d['nsubj']
    nsubj_dobj_norm = np.linalg.norm(nsubj_dobj_vec)
    if nsubj_dobj_norm > 0:
        nsubj_dobj_dir = nsubj_dobj_vec / nsubj_dobj_norm
        # pobj在nsubj-dobj轴上的投影
        nsubj_pobj_vec = centers_3d['pobj'] - centers_3d['nsubj']
        pobj_proj = np.dot(nsubj_pobj_vec, nsubj_dobj_dir)
        pobj_perp = np.linalg.norm(nsubj_pobj_vec - pobj_proj * nsubj_dobj_dir)
        print(f"    pobj在nsubj-dobj轴上的投影: {pobj_proj:.4f}")
        print(f"    pobj垂直于轴的距离: {pobj_perp:.4f}")
        results['pobj_proj_on_nsubj_dobj'] = float(pobj_proj)
        results['pobj_perp_to_nsubj_dobj'] = float(pobj_perp)
        
        # amod和advmod的投影
        for role in ['amod', 'advmod']:
            vec = centers_3d[role] - centers_3d['nsubj']
            proj = np.dot(vec, nsubj_dobj_dir)
            perp = np.linalg.norm(vec - proj * nsubj_dobj_dir)
            print(f"    {role}在nsubj-dobj轴上的投影: {proj:.4f}, 垂直: {perp:.4f}")
            results[f'{role}_proj_on_nsubj_dobj'] = float(proj)
            results[f'{role}_perp_to_nsubj_dobj'] = float(perp)

    # Part E: 所有名词角色(nsubj/poss/dobj/pobj)的两两CV
    print(f"\n  Part E: 名词角色两两CV (target位置)")
    noun_roles = ['nsubj', 'poss', 'dobj', 'pobj']
    for i, r1 in enumerate(noun_roles):
        for j, r2 in enumerate(noun_roles):
            if j > i:
                mask = (labels == role_names.index(r1)) | (labels == role_names.index(r2))
                H_pair = H_scaled[mask]
                labels_pair = labels[mask]
                if len(H_pair) >= 10:
                    probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
                    cv = cross_val_score(probe, H_pair, labels_pair, cv=5, scoring='accuracy')
                    print(f"    {r1} vs {r2}: CV={cv.mean():.4f}")
                    results[f'cv_{r1}_vs_{r2}'] = float(cv.mean())

    # Part F: 凸包分析
    print(f"\n  Part F: 名词角色的凸包(3D)")
    # 6个中心点是否在凸包上?
    try:
        hull = ConvexHull(center_matrix)
        print(f"    凸包顶点数: {len(hull.vertices)}")
        print(f"    凸包体积: {hull.volume:.6f}")
        print(f"    凸包面积: {hull.area:.6f}")
        for v in hull.vertices:
            print(f"    顶点: {role_names[v]}")
        results['hull_vertices'] = [role_names[v] for v in hull.vertices]
        results['hull_volume'] = float(hull.volume)
        results['hull_area'] = float(hull.area)
    except Exception as e:
        print(f"    凸包计算失败: {e}")

    results['pca3_variance'] = float(var_3d)
    return results


# ===== Exp2: 位置编码与语法编码的分离 =====
def exp2_position_syntax_separation(model, tokenizer, device):
    """位置编码与语法编码的分离"""
    print("\n" + "="*70)
    print("Exp2: 位置编码与语法编码的分离 ★★★★★★★")
    print("="*70)

    results = {}
    model_info = get_model_info(model, args.model if hasattr(args, 'model') else 'qwen3')

    # Part A: 主动/被动语态中的同一词
    print(f"\n  Part A: 主动/被动语态中的同一词")

    groups = ['active_nsubj', 'active_dobj', 'passive_nsubj']
    group_h = {}

    for group in groups:
        data = VOICE_PAIR_DATA[group]
        sentences = [s[0] for s in data["sentences"]]
        target_words = [s[1] for s in data["sentences"]]
        H = collect_hs_at_layer(model, tokenizer, device, sentences, target_words, -1)
        if H is not None:
            group_h[group] = H
            print(f"  {group}: {len(H)} samples")

    # 同一词在nsubj位置 vs dobj位置的表示差异
    if 'active_nsubj' in group_h and 'active_dobj' in group_h:
        print(f"\n  Part A1: nsubj vs dobj (同一词不同位置)")
        H_nsubj = group_h['active_nsubj']
        H_dobj = group_h['active_dobj']
        min_len = min(len(H_nsubj), len(H_dobj))
        
        # 配对距离
        paired_dists = np.linalg.norm(H_nsubj[:min_len] - H_dobj[:min_len], axis=1)
        # 随机距离
        rand_dists = []
        for _ in range(200):
            i = np.random.randint(len(H_nsubj))
            j = np.random.randint(len(H_dobj))
            rand_dists.append(np.linalg.norm(H_nsubj[i] - H_dobj[j]))
        rand_dists = np.array(rand_dists)
        
        print(f"    配对距离(同词nsubj vs dobj): mean={np.mean(paired_dists):.4f}")
        print(f"    随机距离: mean={np.mean(rand_dists):.4f}")
        print(f"    距离比: {np.mean(paired_dists)/max(np.mean(rand_dists),1e-10):.4f}")
        results['nsubj_dobj_paired_dist'] = float(np.mean(paired_dists))
        results['nsubj_dobj_random_dist'] = float(np.mean(rand_dists))

        # 分类CV
        H_pair = np.vstack([H_nsubj[:min_len], H_dobj[:min_len]])
        labels_pair = np.array([0]*min_len + [1]*min_len)
        scaler = StandardScaler()
        H_scaled = scaler.fit_transform(H_pair)
        probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        cv = cross_val_score(probe, H_scaled, labels_pair, cv=5, scoring='accuracy')
        print(f"    nsubj vs dobj CV: {cv.mean():.4f}")
        results['nsubj_vs_dobj_cv'] = float(cv.mean())

    # 主动nsubj vs 被动nsubj (同一个词, 语法角色相同但句法结构不同)
    if 'active_nsubj' in group_h and 'passive_nsubj' in group_h:
        print(f"\n  Part A2: 主动nsubj vs 被动nsubj (同词同角色不同结构)")
        H_active = group_h['active_nsubj']
        H_passive = group_h['passive_nsubj']
        min_len = min(len(H_active), len(H_passive))
        
        paired_dists = np.linalg.norm(H_active[:min_len] - H_passive[:min_len], axis=1)
        print(f"    配对距离(主动nsubj vs 被动nsubj): mean={np.mean(paired_dists):.4f}")
        results['active_passive_nsubj_paired_dist'] = float(np.mean(paired_dists))
        
        # 分类
        H_pair = np.vstack([H_active[:min_len], H_passive[:min_len]])
        labels_pair = np.array([0]*min_len + [1]*min_len)
        scaler = StandardScaler()
        H_scaled = scaler.fit_transform(H_pair)
        probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        cv = cross_val_score(probe, H_scaled, labels_pair, cv=5, scoring='accuracy')
        print(f"    主动nsubj vs 被动nsubj CV: {cv.mean():.4f}")
        results['active_vs_passive_nsubj_cv'] = float(cv.mean())

    # Part B: 位置编码的加法假设检验
    print(f"\n  Part B: 位置编码的加法假设检验")
    # 假设: h = h_semantic + h_position + h_syntax
    # 如果位置编码是加法的, 则:
    #   h(nsubj, pos=1) - h(dobj, pos=3) ≈ h_position(pos1-pos3) + h_syntax(nsubj-dobj)
    # 验证: 减去平均位置向量后, 语法角色信息是否保留?

    # 收集不同位置的"平均"表示(使用同一组词)
    control_data = VOICE_PAIR_DATA['position_control']
    sentences = [s[0] for s in control_data["sentences"]]
    target_words = [s[1] for s in control_data["sentences"]]
    roles_list = [s[2] for s in control_data["sentences"]]

    # 收集最后一层的hidden states
    control_h = []
    control_labels = []
    for sent, tw, role in zip(sentences, target_words, roles_list):
        toks = tokenizer(sent, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
        
        dep_idx = find_token_index(tokens_list, tw)
        if dep_idx is None:
            continue
        
        layers = get_layers(model)
        target_layer = layers[-1]
        
        captured = {}
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured['h'] = output[0].detach().float().cpu().numpy()
            else:
                captured['h'] = output.detach().float().cpu().numpy()
        
        h_handle = target_layer.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = model(**toks)
        h_handle.remove()
        
        if 'h' not in captured:
            continue
        
        h_vec = captured['h'][0, dep_idx, :]
        control_h.append(h_vec)
        control_labels.append(role)

    if len(control_h) >= 10:
        H_control = np.array(control_h)
        control_labels = np.array(control_labels)

        # 原始CV
        scaler = StandardScaler()
        H_scaled = scaler.fit_transform(H_control)
        
        # nsubj_pos1 vs poss_pos1
        nsubj_mask = np.array([l.startswith('nsubj') for l in control_labels])
        poss_mask = np.array([l.startswith('poss') for l in control_labels])
        
        if nsubj_mask.sum() >= 5 and poss_mask.sum() >= 5:
            pair_mask = nsubj_mask | poss_mask
            H_pair = H_scaled[pair_mask]
            labels_pair = np.array([0]*nsubj_mask[pair_mask].sum() + [1]*poss_mask[pair_mask].sum())
            if len(H_pair) >= 10:
                probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
                cv = cross_val_score(probe, H_pair, labels_pair, cv=min(5, len(H_pair)//2), scoring='accuracy')
                print(f"    nsubj_pos1 vs poss_pos1 原始CV: {cv.mean():.4f}")
                results['control_nsubj_vs_poss_cv'] = float(cv.mean())

        # Part C: 位置编码减除实验
        print(f"\n  Part C: 位置编码减除实验")
        # 计算各位置的平均向量(用first token作为位置编码代理)
        # 收集first token
        first_h = []
        for sent, tw, role in zip(sentences, target_words, roles_list):
            toks = tokenizer(sent, return_tensors="pt").to(device)
            layers = get_layers(model)
            target_layer = layers[-1]
            
            captured = {}
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    captured['h'] = output[0].detach().float().cpu().numpy()
                else:
                    captured['h'] = output.detach().float().cpu().numpy()
            
            h_handle = target_layer.register_forward_hook(hook_fn)
            with torch.no_grad():
                _ = model(**toks)
            h_handle.remove()
            
            if 'h' not in captured:
                continue
            
            first_h.append(captured['h'][0, 0, :])  # first token

        if len(first_h) > 0:
            first_h = np.array(first_h)
            # 减去平均first token(作为位置编码的估计)
            mean_first = np.mean(first_h, axis=0)
            H_control_centered = H_control - mean_first
            
            # 减去后重新分类
            scaler2 = StandardScaler()
            H_centered_scaled = scaler2.fit_transform(H_control_centered)
            
            if nsubj_mask.sum() >= 5 and poss_mask.sum() >= 5:
                pair_mask = nsubj_mask | poss_mask
                H_pair_c = H_centered_scaled[pair_mask]
                labels_pair = np.array([0]*nsubj_mask[pair_mask].sum() + [1]*poss_mask[pair_mask].sum())
                if len(H_pair_c) >= 10:
                    probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
                    cv_c = cross_val_score(probe, H_pair_c, labels_pair, cv=min(5, len(H_pair_c)//2), scoring='accuracy')
                    print(f"    减去位置编码后 nsubj_pos1 vs poss_pos1 CV: {cv_c.mean():.4f}")
                    results['centered_nsubj_vs_poss_cv'] = float(cv_c.mean())

            # 用dobj位置测试(主动nsubj vs 主动dobj)
            if 'active_nsubj' in group_h and 'active_dobj' in group_h:
                H_ns = group_h['active_nsubj']
                H_do = group_h['active_dobj']
                min_len = min(len(H_ns), len(H_do))
                
                # 原始
                H_pair_raw = np.vstack([H_ns[:min_len], H_do[:min_len]])
                labels_raw = np.array([0]*min_len + [1]*min_len)
                scaler_raw = StandardScaler()
                H_pair_raw_s = scaler_raw.fit_transform(H_pair_raw)
                probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
                cv_raw = cross_val_score(probe, H_pair_raw_s, labels_raw, cv=5, scoring='accuracy')
                
                # 减去位置编码后
                H_ns_c = H_ns - mean_first
                H_do_c = H_do - mean_first
                H_pair_c = np.vstack([H_ns_c[:min_len], H_do_c[:min_len]])
                scaler_c = StandardScaler()
                H_pair_c_s = scaler_c.fit_transform(H_pair_c)
                probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
                cv_c = cross_val_score(probe, H_pair_c_s, labels_raw, cv=5, scoring='accuracy')
                
                print(f"    nsubj vs dobj 原始CV: {cv_raw.mean():.4f}")
                print(f"    nsubj vs dobj 减去位置编码后CV: {cv_c.mean():.4f}")
                print(f"    位置编码贡献: {(cv_raw.mean() - cv_c.mean()):.4f}")
                results['nsubj_dobj_raw_cv'] = float(cv_raw.mean())
                results['nsubj_dobj_centered_cv'] = float(cv_c.mean())
                results['position_coding_contribution'] = float(cv_raw.mean() - cv_c.mean())

    return results


# ===== Exp3: ICA斜交分解 =====
def exp3_ica_decomposition(model, tokenizer, device):
    """ICA斜交分解 vs PCA正交分解"""
    print("\n" + "="*70)
    print("Exp3: ICA斜交分解 ★★★★★")
    print("="*70)

    results = {}
    role_names = list(MANIFOLD_ROLES_DATA.keys())

    # 收集6类语法角色的hidden states
    all_h = []
    all_labels = []

    for role_idx, role in enumerate(role_names):
        data = MANIFOLD_ROLES_DATA[role]
        H = collect_hs_at_layer(model, tokenizer, device,
                                data["sentences"], data["target_words"], -1)
        if H is not None and len(H) > 0:
            all_h.append(H)
            all_labels.extend([role_idx] * len(H))
            print(f"  {role}: {len(H)} samples")

    if len(all_h) < 6:
        print("  样本不足!")
        return results

    H_all = np.vstack(all_h)
    labels = np.array(all_labels)

    scaler = StandardScaler()
    H_scaled = scaler.fit_transform(H_all)

    # 基准: 全空间
    probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
    cv_full = cross_val_score(probe, H_scaled, labels, cv=5, scoring='accuracy')
    baseline = float(cv_full.mean())
    print(f"  基准CV: {baseline:.4f}")
    results['baseline_cv'] = baseline

    # PCA vs ICA在不同维度下的比较
    for n_comp in [3, 5, 10, 15, 20]:
        if n_comp >= min(H_all.shape[0], H_all.shape[1]):
            continue
        
        # PCA
        pca = PCA(n_components=n_comp)
        H_pca = pca.fit_transform(H_scaled)
        probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        cv_pca = cross_val_score(probe, H_pca, labels, cv=5, scoring='accuracy')
        
        # ICA
        ica = FastICA(n_components=n_comp, random_state=42, max_iter=500)
        H_ica = ica.fit_transform(H_scaled)
        probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        cv_ica = cross_val_score(probe, H_ica, labels, cv=5, scoring='accuracy')
        
        print(f"    dim={n_comp}: PCA={cv_pca.mean():.4f}, ICA={cv_ica.mean():.4f}, "
              f"ICA-PCA={cv_ica.mean()-cv_pca.mean():.4f}")
        results[f'pca_{n_comp}_cv'] = float(cv_pca.mean())
        results[f'ica_{n_comp}_cv'] = float(cv_ica.mean())
        results[f'ica_pca_diff_{n_comp}'] = float(cv_ica.mean() - cv_pca.mean())

    # Part B: ICA分量的可解释性
    print(f"\n  Part B: ICA分量的可解释性 (dim=10)")
    n_comp = 10
    if n_comp < min(H_all.shape[0], H_all.shape[1]):
        ica = FastICA(n_components=n_comp, random_state=42, max_iter=500)
        H_ica = ica.fit_transform(H_scaled)
        
        # 每个ICA分量对各类的区分力
        for comp_idx in range(n_comp):
            comp_values = H_ica[:, comp_idx]
            role_means = {}
            for role_idx, role in enumerate(role_names):
                mask = labels == role_idx
                role_means[role] = float(np.mean(comp_values[mask]))
            
            # 找最大和最小的角色
            max_role = max(role_means, key=role_means.get)
            min_role = min(role_means, key=role_means.get)
            max_val = role_means[max_role]
            min_val = role_means[min_role]
            
            if comp_idx < 5:  # 只打印前5个
                print(f"    ICA-{comp_idx}: max={max_role}({max_val:.3f}), "
                      f"min={min_role}({min_val:.3f}), range={max_val-min_val:.3f}")
            
            results[f'ica_comp{comp_idx}_max_role'] = max_role
            results[f'ica_comp{comp_idx}_min_role'] = min_role
            results[f'ica_comp{comp_idx}_range'] = float(max_val - min_val)

    # Part C: ICA子空间投影信息保留
    print(f"\n  Part C: ICA子空间投影信息保留 (Gram-Schmidt替代)")
    # 不做Gram-Schmidt, 直接比较ICA和PCA在相同维度下的信息保留
    n_comp = 20
    if n_comp < min(H_all.shape[0], H_all.shape[1]):
        pca = PCA(n_components=n_comp)
        H_pca = pca.fit_transform(H_scaled)
        var_pca = sum(pca.explained_variance_ratio_)
        
        # ICA的"方差保留"用投影重建误差来估计
        ica = FastICA(n_components=n_comp, random_state=42, max_iter=500)
        H_ica = ica.fit_transform(H_scaled)
        # ICA重建: H_scaled ≈ H_ica @ ica.mixing_.T + ica.mean_
        H_recon = H_ica @ ica.mixing_.T + ica.mean_
        recon_error = np.mean(np.linalg.norm(H_scaled - H_recon, axis=1))
        total_norm = np.mean(np.linalg.norm(H_scaled, axis=1))
        ica_recon_rate = 1.0 - recon_error / max(total_norm, 1e-10)
        
        print(f"    PCA dim={n_comp} 方差保留: {var_pca:.4f}")
        print(f"    ICA dim={n_comp} 重建率: {ica_recon_rate:.4f}")
        results['pca20_var'] = float(var_pca)
        results['ica20_recon_rate'] = float(ica_recon_rate)

    return results


# ===== Exp4: 语法角色在多层中的演化 =====
def exp4_layer_evolution(model, tokenizer, device):
    """语法角色在多层中的演化"""
    print("\n" + "="*70)
    print("Exp4: 语法角色在多层中的演化 ★★★★★★★")
    print("="*70)

    results = {}
    role_names = list(MANIFOLD_ROLES_DATA.keys())
    model_info = get_model_info(model, args.model if hasattr(args, 'model') else 'qwen3')
    n_layers = model_info.n_layers

    # 采样层
    sample_layers = []
    if n_layers <= 8:
        sample_layers = list(range(n_layers))
    else:
        step = n_layers // 7
        sample_layers = list(range(0, n_layers, step)) + [n_layers - 1]
        sample_layers = sorted(set(sample_layers))

    print(f"  采样层: {sample_layers}")

    layer_results = {}

    for layer_idx in sample_layers:
        print(f"\n  === Layer {layer_idx} ===")
        
        # 收集hidden states
        all_h = []
        all_labels = []
        
        for role_idx, role in enumerate(role_names):
            data = MANIFOLD_ROLES_DATA[role]
            H = collect_hs_at_layer(model, tokenizer, device,
                                    data["sentences"], data["target_words"], layer_idx)
            if H is not None and len(H) > 0:
                all_h.append(H)
                all_labels.extend([role_idx] * len(H))

        if len(all_h) < 6:
            print(f"    样本不足, 跳过")
            continue

        H_all = np.vstack(all_h)
        labels = np.array(all_labels)

        # 6类分类CV
        scaler = StandardScaler()
        H_scaled = scaler.fit_transform(H_all)
        probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        cv_6class = cross_val_score(probe, H_scaled, labels, cv=5, scoring='accuracy')
        print(f"    6类CV: {cv_6class.mean():.4f}")

        # nsubj vs poss CV
        nsubj_idx = role_names.index('nsubj')
        poss_idx = role_names.index('poss')
        mask_np = (labels == nsubj_idx) | (labels == poss_idx)
        H_np = H_scaled[mask_np]
        labels_np = labels[mask_np]
        if len(H_np) >= 10:
            probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
            cv_np = cross_val_score(probe, H_np, labels_np, cv=5, scoring='accuracy')
            print(f"    nsubj vs poss CV: {cv_np.mean():.4f}")
        else:
            cv_np = np.array([0.5])

        # nsubj vs dobj CV
        dobj_idx = role_names.index('dobj')
        mask_nd = (labels == nsubj_idx) | (labels == dobj_idx)
        H_nd = H_scaled[mask_nd]
        labels_nd = labels[mask_nd]
        if len(H_nd) >= 10:
            probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
            cv_nd = cross_val_score(probe, H_nd, labels_nd, cv=5, scoring='accuracy')
            print(f"    nsubj vs dobj CV: {cv_nd.mean():.4f}")
        else:
            cv_nd = np.array([0.5])

        # 内在维度(95%)
        max_dim = min(H_all.shape[0], H_all.shape[1]) - 1
        dims_to_test = [d for d in [1, 2, 3, 5, 7, 10, 15, 20, 30] if d < max_dim]
        intrinsic_dim = None
        for dim in dims_to_test:
            pca = PCA(n_components=dim)
            H_pca = pca.fit_transform(H_scaled)
            probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
            cv_dim = cross_val_score(probe, H_pca, labels, cv=5, scoring='accuracy')
            if intrinsic_dim is None and cv_dim.mean() >= cv_6class.mean() * 0.95:
                intrinsic_dim = dim
                break
        if intrinsic_dim is None:
            intrinsic_dim = dims_to_test[-1] if dims_to_test else max_dim
        print(f"    内在维度(95%): {intrinsic_dim}")

        # nsubj-poss中心余弦相似度
        H_nsubj = H_scaled[labels == nsubj_idx]
        H_poss = H_scaled[labels == poss_idx]
        if len(H_nsubj) > 0 and len(H_poss) > 0:
            center_ns = np.mean(H_nsubj, axis=0)
            center_po = np.mean(H_poss, axis=0)
            cos_np = np.dot(center_ns, center_po) / (
                np.linalg.norm(center_ns) * np.linalg.norm(center_po) + 1e-10)
            print(f"    nsubj-poss中心cos: {cos_np:.4f}")
        else:
            cos_np = 0.0

        layer_results[layer_idx] = {
            'cv_6class': float(cv_6class.mean()),
            'cv_nsubj_poss': float(cv_np.mean()),
            'cv_nsubj_dobj': float(cv_nd.mean()),
            'intrinsic_dim': intrinsic_dim,
            'nsubj_poss_cos': float(cos_np),
        }

    results['layer_results'] = layer_results

    # Part B: 演化趋势总结
    print(f"\n  Part B: 演化趋势总结")
    layers_sorted = sorted(layer_results.keys())
    
    print(f"\n    CV随层变化:")
    for l in layers_sorted:
        r = layer_results[l]
        print(f"    L{l:2d}: 6class={r['cv_6class']:.3f}, "
              f"nsubj-poss={r['cv_nsubj_poss']:.3f}, "
              f"nsubj-dobj={r['cv_nsubj_dobj']:.3f}, "
              f"dim={r['intrinsic_dim']:2d}, "
              f"cos(nsubj-poss)={r['nsubj_poss_cos']:.3f}")

    # 找语法信息显现的关键层
    cv_trend = [layer_results[l]['cv_6class'] for l in layers_sorted]
    if len(cv_trend) > 2:
        # CV首次超过0.5的层
        syntax_emerge_layer = None
        for l in layers_sorted:
            if layer_results[l]['cv_6class'] > 0.5:
                syntax_emerge_layer = l
                break
        if syntax_emerge_layer is not None:
            print(f"\n    语法信息显现层: L{syntax_emerge_layer}")
            results['syntax_emerge_layer'] = syntax_emerge_layer
        
        # CV最高的层
        best_layer = layers_sorted[np.argmax(cv_trend)]
        print(f"    语法编码最强层: L{best_layer} (CV={max(cv_trend):.3f})")
        results['best_syntax_layer'] = best_layer
    
    # nsubj-poss等价在各层的变化
    nsubj_poss_trend = [layer_results[l]['cv_nsubj_poss'] for l in layers_sorted]
    nsubj_poss_cos_trend = [layer_results[l]['nsubj_poss_cos'] for l in layers_sorted]
    print(f"\n    nsubj-poss等价演化:")
    print(f"    CV趋势: {nsubj_poss_trend}")
    print(f"    cos趋势: {[f'{c:.3f}' for c in nsubj_poss_cos_trend]}")

    return results


# ===== 主函数 =====
def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True,
                       choices=[1, 2, 3, 4])
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"CCL-U Phase15 语法流形精细结构+位置-语法交互 | Model={args.model} | Exp={args.exp}")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    print(f"  Model: {model_info.model_class}, Layers={model_info.n_layers}, "
          f"d_model={model_info.d_model}")

    try:
        if args.exp == 1:
            results = exp1_manifold_geometry(model, tokenizer, device)
        elif args.exp == 2:
            results = exp2_position_syntax_separation(model, tokenizer, device)
        elif args.exp == 3:
            results = exp3_ica_decomposition(model, tokenizer, device)
        elif args.exp == 4:
            results = exp4_layer_evolution(model, tokenizer, device)

        # 保存结果
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'glm5_temp')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir,
                               f"cclu_exp{args.exp}_{args.model}_results.json")

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(convert(v) for v in obj)
            return obj

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(convert(results), f, indent=2, ensure_ascii=False)
        print(f"\n  结果已保存: {out_path}")

    finally:
        release_model(model)
        print(f"  模型已释放")


if __name__ == "__main__":
    main()
