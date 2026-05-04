"""
CCL-Q(Phase 11): 内在几何与信息论分析
=============================================================================
核心问题:
  Phase 10发现: 语法是极低维流形(2-10维), 冗余编码在所有频段成立
  但:
  1. 语法角色在内在维度空间中的几何结构是什么? 是否有"语法拓扑"?
  2. 冗余编码的信息论度量是什么? 互信息在各频段如何分布?
  3. 语法→逻辑推理的维度跃迁是多少? 推理的内在维度是多少?
  4. 语法角色之间的"距离"vs 语法关系是否有关联?

实验:
  Exp1: ★★★★★ 语法角色的2D/3D可视化与几何结构
    → PCA/t-SNE投影到2D/3D
    → 分析: 角色之间的距离, 角色聚类, 语法关系vs几何距离
    → 核心问题: 4个角色在2维中是否形成正四面体? 10个角色是否有拓扑结构?

  Exp2: ★★★★★ 冗余编码的信息论分析
    → 互信息 I(H; role) 在各W_U频段中的分布
    → 条件熵 H(role | H_band) = 分类不确定度
    → 冗余度 = I(H; role) / sum(I(H_band; role)) — 如果>1, 说明存在冗余
    → 最优冗余度: 信息论下界

  Exp3: ★★★★★ 从语法到推理的维度跃迁
    → 语法角色(nsubj/dobj/amod/advmod) — 语法维度
    → 逻辑推理(因果/条件/递进/转折) — 推理维度
    → 推理的内在维度是多少? vs 语法内在维度
    → 语法→推理的维度跃迁比 = dim(reasoning) / dim(syntax)

  Exp4: ★★★★★ 语法角色距离vs语法关系
    → 计算10个角色在内在空间中的欧氏距离矩阵
    → 计算10个角色的语法关系距离(基于依存语法理论)
    → Mantel test: 几何距离 vs 语法距离的相关性
    → 如果相关: 几何编码了语法关系! 这是"语法拓扑"的证据
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

from model_utils import load_model, get_layers, get_model_info, release_model, safe_decode, get_W_U


# ===== 数据集: 复用Phase 10的10角色数据 =====
EXTENDED_ROLES_DATA = {
    "nsubj": {
        "desc": "主语名词",
        "sentences": [
            "The king ruled the kingdom", "The doctor treated the patient",
            "The artist painted the portrait", "The soldier defended the castle",
            "The cat sat on the mat", "The dog ran through the park",
            "The woman drove the car", "The man fixed the roof",
            "The student read the textbook", "The teacher explained the lesson",
            "The president signed the bill", "The chef cooked the meal",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "cat", "dog",
            "woman", "man", "student", "teacher", "president", "chef",
        ],
    },
    "dobj": {
        "desc": "直接宾语",
        "sentences": [
            "They crowned the king yesterday", "She visited the doctor recently",
            "He admired the artist greatly", "We honored the soldier today",
            "She chased the cat away", "He found the dog outside",
            "The police arrested the man quickly", "The company hired the woman recently",
            "I praised the student loudly", "You thanked the teacher warmly",
            "The nation elected the president fairly", "The customer tipped the chef generously",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "cat", "dog",
            "man", "woman", "student", "teacher", "president", "chef",
        ],
    },
    "amod": {
        "desc": "形容词修饰语",
        "sentences": [
            "The brave king fought hard", "The kind doctor helped many",
            "The creative artist worked well", "The strong soldier marched far",
            "The beautiful cat sat quietly", "The large dog ran swiftly",
            "The old woman walked slowly", "The tall man stood quietly",
            "The bright student read carefully", "The wise teacher explained clearly",
            "The powerful president decided firmly", "The skilled chef cooked perfectly",
        ],
        "target_words": [
            "brave", "kind", "creative", "strong", "beautiful", "large",
            "old", "tall", "bright", "wise", "powerful", "skilled",
        ],
    },
    "advmod": {
        "desc": "副词修饰语",
        "sentences": [
            "The king ruled wisely forever", "The doctor worked carefully always",
            "The artist painted beautifully daily", "The soldier fought bravely there",
            "The cat ran quickly home", "The dog barked loudly today",
            "The woman drove slowly home", "The man spoke quietly now",
            "The student read carefully alone", "The teacher spoke clearly again",
            "The president spoke firmly today", "The chef worked quickly then",
        ],
        "target_words": [
            "wisely", "carefully", "beautifully", "bravely", "quickly", "loudly",
            "slowly", "quietly", "carefully", "clearly", "firmly", "quickly",
        ],
    },
    "poss": {
        "desc": "所有格修饰语",
        "sentences": [
            "The king's crown glittered brightly", "The doctor's office opened early",
            "The artist's studio looked beautiful", "The soldier's uniform was clean",
            "The cat's tail swished gently", "The dog's bark echoed loudly",
            "The woman's dress looked elegant", "The man's car drove fast",
            "The student's essay read well", "The teacher's book sold quickly",
            "The president's speech inspired many", "The chef's restaurant opened today",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "cat", "dog",
            "woman", "man", "student", "teacher", "president", "chef",
        ],
    },
    "aux": {
        "desc": "助动词",
        "sentences": [
            "The king has ruled for years", "The doctor will treat the patient",
            "The artist can paint very well", "The soldier should defend the fort",
            "The cat could jump very high", "The dog must run every day",
            "The woman has driven to work", "The man will fix the car",
            "The student should read the book", "The teacher can explain clearly",
            "The president must decide today", "The chef has cooked the meal",
        ],
        "target_words": [
            "has", "will", "can", "should", "could", "must",
            "has", "will", "should", "can", "must", "has",
        ],
    },
    "det": {
        "desc": "限定词(冠词/指示词)",
        "sentences": [
            "This king ruled very wisely", "That doctor helped many people",
            "This artist created beautiful works", "That soldier fought very bravely",
            "This cat jumped over the fence", "That dog chased the ball",
            "This woman led the team well", "That man built the house",
            "This student solved the problem", "That teacher taught the class",
            "This president changed the law", "That chef prepared the feast",
        ],
        "target_words": [
            "This", "That", "This", "That", "This", "That",
            "This", "That", "This", "That", "This", "That",
        ],
    },
    "mark": {
        "desc": "从句标记词",
        "sentences": [
            "He ruled because the king commanded", "She studied since the doctor advised",
            "They painted while the artist watched", "We marched although the soldier hesitated",
            "It slept because the cat was tired", "He barked while the dog played",
            "She worked because the woman needed money", "He rested since the man was exhausted",
            "They studied because the student wanted grades", "She taught while the teacher observed",
            "He decided because the president ordered", "She cooked while the chef supervised",
        ],
        "target_words": [
            "because", "since", "while", "although", "because", "while",
            "because", "since", "because", "while", "because", "while",
        ],
    },
    "nummod": {
        "desc": "数词修饰语",
        "sentences": [
            "Three kings ruled the land", "Five doctors worked at the hospital",
            "Two artists painted the mural", "Four soldiers guarded the gate",
            "Seven cats sat on the wall", "Three dogs ran in the park",
            "Two women drove to the city", "Five men stood in the line",
            "Three students passed the exam", "Four teachers attended the meeting",
            "Two presidents signed the treaty", "Six chefs prepared the banquet",
        ],
        "target_words": [
            "Three", "Five", "Two", "Four", "Seven", "Three",
            "Two", "Five", "Three", "Four", "Two", "Six",
        ],
    },
    "cc": {
        "desc": "并列连词",
        "sentences": [
            "The king and the queen ruled together", "The doctor or the nurse helped first",
            "The artist and the musician performed", "The soldier but not the civilian fought",
            "The cat and the dog played together", "The dog or the cat chased the mouse",
            "The woman and the man worked together", "The teacher or the student answered first",
            "The student and the researcher collaborated", "The chef and the waiter served dinner",
            "The president and the minister agreed", "The doctor but not the patient decided",
        ],
        "target_words": [
            "and", "or", "and", "but", "and", "or",
            "and", "or", "and", "and", "and", "but",
        ],
    },
}

ROLE_SETS = {
    "4roles": ["nsubj", "dobj", "amod", "advmod"],
    "6roles": ["nsubj", "dobj", "amod", "advmod", "poss", "aux"],
    "8roles": ["nsubj", "dobj", "amod", "advmod", "poss", "aux", "det", "mark"],
    "10roles": list(EXTENDED_ROLES_DATA.keys()),
}

# ===== 推理类型数据 =====
REASONING_DATA = {
    "causal": {
        "desc": "因果推理",
        "sentences": [
            "The rain caused the flood", "The heat melted the ice",
            "The wind broke the window", "The fire burned the forest",
            "The cold froze the lake", "The storm damaged the roof",
            "The drought killed the crops", "The earthquake destroyed the building",
            "The explosion shattered the glass", "The virus infected the patient",
            "The medicine cured the disease", "The exercise strengthened the muscle",
        ],
    },
    "conditional": {
        "desc": "条件推理",
        "sentences": [
            "If it rains the ground gets wet", "If you study you will pass",
            "If she runs she will win", "If they work they get paid",
            "If we try we can succeed", "If he rests he recovers",
            "If the sun shines the ice melts", "If you eat you feel full",
            "If she practices she improves", "If they build they create",
            "If we plan we achieve", "If he reads he learns",
        ],
    },
    "progressive": {
        "desc": "递进推理",
        "sentences": [
            "He worked hard and earned more", "She studied more and improved greatly",
            "They practiced daily and mastered the skill", "He saved money and bought a house",
            "She trained hard and won the race", "They invested wisely and grew wealthy",
            "He exercised regularly and got stronger", "She wrote daily and published books",
            "They researched deeply and discovered truth", "He practiced medicine and saved lives",
            "She painted often and created beauty", "They explored widely and found treasure",
        ],
    },
    "adversative": {
        "desc": "转折推理",
        "sentences": [
            "He tried hard but still failed", "She studied hard yet barely passed",
            "They planned carefully but things went wrong", "He was tired yet kept working",
            "She was afraid but faced the challenge", "They were outnumbered yet won the battle",
            "He was ill but attended the meeting", "She was young yet very wise",
            "They were poor but remained happy", "He was wrong but refused to admit",
            "She was hurt but kept smiling", "They were late but finished first",
        ],
    },
}

# ===== 语法关系距离矩阵(基于依存语法理论) =====
# 10个角色的语法关系距离: 基于Universal Dependencies中的角色分类
# 角色分类体系:
#   Core arguments: nsubj, dobj (核心论元)
#   Non-core dependents: advmod, aux (非核心依存)
#   Nominal modifiers: amod, poss, det, nummod (名词修饰语)
#   Clausal/coordination: mark, cc (从句/并列)
#
# 距离定义: 同类=1, 相邻类=2, 远类=3
SYNTAX_DISTANCE_MATRIX = {
    #              nsubj dobj amod advmod poss aux det mark nummod cc
    "nsubj":  [   0,    1,   2,    3,    2,   3,  2,  3,    2,    3],
    "dobj":   [   1,    0,   2,    3,    2,   3,  2,  3,    2,    3],
    "amod":   [   2,    2,   0,    2,    1,   3,  1,  3,    1,    3],
    "advmod": [   3,    3,   2,    0,    3,   1,  3,  2,    3,    2],
    "poss":   [   2,    2,   1,    3,    0,   3,  1,  3,    1,    3],
    "aux":    [   3,    3,   3,    1,    3,   0,  3,  2,    3,    2],
    "det":    [   2,    2,   1,    3,    1,   3,  0,  3,    1,    3],
    "mark":   [   3,    3,   3,    2,    3,   2,  3,  0,    3,    1],
    "nummod": [   2,    2,   1,    3,    1,   3,  1,  3,    0,    3],
    "cc":     [   3,    3,   3,    2,    3,   2,  3,  1,    3,    0],
}

ROLE_NAMES_10 = ["nsubj", "dobj", "amod", "advmod", "poss", "aux", "det", "mark", "nummod", "cc"]


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


def collect_hidden_states_multirole(model, tokenizer, device, role_names, data_dict, layer_idx=-1):
    """收集多个语法角色的hidden states"""
    all_h = []
    all_labels = []

    layers = get_layers(model)
    target_layer = layers[layer_idx] if layer_idx >= 0 else layers[-1]

    for role_idx, role in enumerate(role_names):
        data = data_dict[role]
        for sent, target_word in zip(data["sentences"], data["target_words"]):
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
            all_labels.append(role_idx)

    return np.array(all_h), np.array(all_labels)


def collect_reasoning_hidden_states(model, tokenizer, device, layer_idx=-1):
    """收集推理类别的hidden states(取句末token)"""
    all_h = []
    all_labels = []

    layers = get_layers(model)
    target_layer = layers[layer_idx] if layer_idx >= 0 else layers[-1]

    for cat_idx, (cat_name, cat_data) in enumerate(REASONING_DATA.items()):
        for sent in cat_data["sentences"]:
            toks = tokenizer(sent, return_tensors="pt").to(device)

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

            # 取句末token
            h_vec = captured['h'][0, -1, :]
            all_h.append(h_vec)
            all_labels.append(cat_idx)

    return np.array(all_h), np.array(all_labels)


def compute_W_U_eigenspectrum(W_U):
    """计算W_U的特征值谱和特征向量"""
    d_model = W_U.shape[1]
    W_U_f64 = W_U.astype(np.float64)
    WtW = W_U_f64.T @ W_U_f64
    eigenvalues, eigenvectors = np.linalg.eigh(WtW)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    return eigenvalues, eigenvectors


# ===== Exp1: 语法角色的2D/3D可视化与几何结构 =====
def exp1_syntax_geometry(model, tokenizer, device):
    """分析语法角色在内在维度空间中的几何结构"""
    print("\n" + "="*70)
    print("Exp1: 语法角色的内在几何结构 ★★★★★")
    print("="*70)

    results = {}

    for set_name, role_names in ROLE_SETS.items():
        n_roles = len(role_names)
        print(f"\n  --- {set_name} ({n_roles}角色) ---")

        # 收集hidden states
        H, labels = collect_hidden_states_multirole(
            model, tokenizer, device, role_names, EXTENDED_ROLES_DATA)
        print(f"  收集到 {len(H)} 个样本")

        if len(H) < n_roles * 2:
            continue

        # Part A: PCA降维到2D和3D
        n_components = min(3, min(H.shape[0], H.shape[1]) - 1)
        pca = PCA(n_components=n_components)
        H_pca = pca.fit_transform(H)

        explained_var = pca.explained_variance_ratio_
        print(f"\n  PCA解释方差比: {explained_var}")
        print(f"  前3维累计解释: {np.cumsum(explained_var)[:3]}")

        # Part B: 质心坐标
        centers = {}
        for ri, role in enumerate(role_names):
            mask = labels == ri
            centers[role] = H_pca[mask].mean(axis=0)

        print(f"\n  各角色质心(PCA坐标):")
        for role in role_names:
            c = centers[role]
            if len(c) >= 2:
                print(f"    {role:>8s}: ({c[0]:8.3f}, {c[1]:8.3f}" +
                      (f", {c[2]:8.3f})" if len(c) >= 3 else ")"))

        # Part C: 质心间距离矩阵
        center_vecs = np.array([centers[r] for r in role_names])
        dist_matrix = squareform(pdist(center_vecs, metric='euclidean'))

        print(f"\n  质心间欧氏距离矩阵:")
        header = "         " + "  ".join(f"{r[:5]:>6s}" for r in role_names)
        print(f"  {header}")
        for ri, role in enumerate(role_names):
            row = f"  {role:>8s}"
            for ci in range(n_roles):
                row += f"  {dist_matrix[ri, ci]:6.3f}"
            print(row)

        # Part D: 几何结构分析
        # 4角色: 检测是否接近正四面体
        if n_roles == 4:
            dists_4 = dist_matrix[np.triu_indices(4, k=1)]
            cv_dist = dists_4.std() / max(dists_4.mean(), 1e-10)
            print(f"\n  ★ 4角色距离变异系数: {cv_dist:.4f}")
            print(f"    正四面体: CV=0.000, 随机4点: CV≈0.3-0.5")
            print(f"    {'接近正四面体!' if cv_dist < 0.1 else '非正四面体'}")

        # 10角色: 层次聚类检测
        if n_roles >= 8:
            # 基于距离矩阵的简单聚类分析
            # 找最近邻
            print(f"\n  ★ 最近邻关系:")
            for ri, role in enumerate(role_names):
                dists_from_ri = dist_matrix[ri].copy()
                dists_from_ri[ri] = float('inf')
                nn_idx = np.argmin(dists_from_ri)
                print(f"    {role:>8s} -> {role_names[nn_idx]:>8s} (d={dist_matrix[ri, nn_idx]:.3f})")

        # Part E: 内在维度中的角度结构
        if n_components >= 2:
            # 计算质心相对原点的角度
            angles = []
            for role in role_names:
                c = centers[role]
                angle = np.arctan2(c[1], c[0]) * 180 / np.pi
                angles.append(angle)

            print(f"\n  质心角度分布(PCA1-2平面):")
            for role, angle in zip(role_names, angles):
                print(f"    {role:>8s}: {angle:7.1f}°")

            # 角度间隔
            sorted_angles = sorted(zip(angles, role_names))
            print(f"\n  角度排序:")
            for angle, role in sorted_angles:
                print(f"    {angle:7.1f}° - {role}")

        results[set_name] = {
            'n_roles': n_roles,
            'n_samples': len(H),
            'pca_explained_var': explained_var.tolist(),
            'centers_pca': {r: centers[r].tolist() for r in role_names},
            'distance_matrix': dist_matrix.tolist(),
            'role_names': role_names,
        }

    return results


# ===== Exp2: 冗余编码的信息论分析 =====
def exp2_information_theory(model, tokenizer, device):
    """分析语法信息在各W_U频段中的互信息分布"""
    print("\n" + "="*70)
    print("Exp2: 冗余编码的信息论分析 ★★★★★")
    print("="*70)

    # 收集4角色hidden states
    syntax_roles = ROLE_SETS["4roles"]
    H, labels = collect_hidden_states_multirole(
        model, tokenizer, device, syntax_roles, EXTENDED_ROLES_DATA)
    print(f"  收集到 {len(H)} 个样本, {len(syntax_roles)} 角色")

    n_samples = len(H)
    n_roles = len(syntax_roles)
    d_model = H.shape[1]

    # W_U特征谱
    W_U = get_W_U(model)
    eigenvalues, eigenvectors = compute_W_U_eigenspectrum(W_U)

    # Part A: 互信息估计 — 基于分类准确率
    # I(H; role) ≈ log(n_roles) + H(role|H)
    # H(role|H) ≈ -sum p(role|H) log p(role|H) 对于分类器
    # 近似: I(H; role) ≈ log2(n_roles) * accuracy (粗略)
    # 更好: 用分类概率估计条件熵

    print(f"\n  Part A: 各频段的互信息估计")
    print(f"  H(role) = log2({n_roles}) = {np.log2(n_roles):.3f} bits")

    # 全空间分类 + 概率估计
    scaler = StandardScaler()
    H_scaled = scaler.fit_transform(H)
    probe_full = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
    probe_full.fit(H_scaled, labels)

    # 计算条件熵 H(role|H_full) 用分类概率
    probs_full = probe_full.predict_proba(H_scaled)
    # 限制概率下界避免log(0)
    probs_full = np.clip(probs_full, 1e-10, 1.0)
    probs_full /= probs_full.sum(axis=1, keepdims=True)
    H_role_given_full = -np.mean(np.sum(probs_full * np.log2(probs_full), axis=1))
    I_full = np.log2(n_roles) - H_role_given_full

    print(f"  I(H_full; role) = {I_full:.3f} bits")

    # 5个频段的互信息
    n_bands = 5
    band_size = d_model // n_bands

    band_mi = {}
    print(f"\n  {'Band':>8s} {'Range':>12s} {'I(band)':>8s} {'I/full':>8s} {'Acc':>6s}")

    for bi in range(n_bands):
        start = bi * band_size
        end = min((bi + 1) * band_size, d_model)
        U_band = eigenvectors[:, start:end]
        H_band = H @ U_band

        scaler_b = StandardScaler()
        H_band_scaled = scaler_b.fit_transform(H_band)
        probe_b = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        probe_b.fit(H_band_scaled, labels)

        probs_b = probe_b.predict_proba(H_band_scaled)
        probs_b = np.clip(probs_b, 1e-10, 1.0)
        probs_b /= probs_b.sum(axis=1, keepdims=True)
        H_cond_b = -np.mean(np.sum(probs_b * np.log2(probs_b), axis=1))
        I_band = np.log2(n_roles) - H_cond_b

        # 分类准确率
        cv = cross_val_score(
            LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0),
            H_band_scaled, labels, cv=5, scoring='accuracy')

        band_name = f"band{bi+1}"
        band_mi[band_name] = float(I_band)

        lambda_range = f"[{eigenvalues[start]:.0f}-{eigenvalues[end-1]:.0f}]"
        print(f"  {band_name:>8s} {lambda_range:>12s} {I_band:8.3f} {I_band/max(I_full,0.001):8.3f} {cv.mean():6.3f}")

    # Part B: 冗余度分析
    sum_band_mi = sum(band_mi.values())
    redundancy = sum_band_mi / max(I_full, 0.001)

    print(f"\n  Part B: 冗余度分析")
    print(f"  I(H_full; role)     = {I_full:.3f} bits")
    print(f"  Σ I(H_band; role)  = {sum_band_mi:.3f} bits")
    print(f"  冗余度 = Σ/I_full  = {redundancy:.2f}")
    print(f"  如果冗余度>1: 各频段独立贡献之和超过总量 → 存在信息冗余")
    print(f"  如果冗余度≈1: 各频段贡献不重叠 → 无冗余")
    print(f"  如果冗余度<1: 各频段贡献不足 → 需要联合信息")

    # Part C: 逐维度MI贡献(PCA空间)
    print(f"\n  Part C: PCA逐维度互信息贡献")
    max_pca = min(20, min(H.shape[0], H.shape[1]) - 1)
    pca = PCA(n_components=max_pca)
    H_pca = pca.fit_transform(H)

    dim_mi = []
    cumsum_mi = 0
    print(f"  {'Dim':>5s} {'I(dim)':>8s} {'CumI':>8s} {'CumI/I_full':>12s}")

    for dim in range(max_pca):
        H_1d = H_pca[:, dim:dim+1]
        scaler_d = StandardScaler()
        H_1d_scaled = scaler_d.fit_transform(H_1d)

        probe_d = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        cv_d = cross_val_score(probe_d, H_1d_scaled, labels, cv=5, scoring='accuracy')
        # 近似MI: I ≈ log2(n_roles) * max(0, cv - 1/n_roles) / (1 - 1/n_roles)
        # 简化: I ≈ accuracy * log2(n_roles) 作为上界
        mi_approx = cv_d.mean() * np.log2(n_roles)
        dim_mi.append(float(mi_approx))
        cumsum_mi += mi_approx

        if dim < 10 or (dim + 1) % 5 == 0:
            print(f"  {dim+1:5d} {mi_approx:8.3f} {cumsum_mi:8.3f} {cumsum_mi/max(I_full,0.001):12.3f}")

    results = {
        'n_samples': n_samples,
        'n_roles': n_roles,
        'H_role': float(np.log2(n_roles)),
        'I_full': float(I_full),
        'H_role_given_full': float(H_role_given_full),
        'band_mi': band_mi,
        'sum_band_mi': float(sum_band_mi),
        'redundancy': float(redundancy),
        'pca_dim_mi': dim_mi,
        'pca_explained_var': pca.explained_variance_ratio_.tolist(),
    }

    return results


# ===== Exp3: 从语法到推理的维度跃迁 =====
def exp3_syntax_to_reasoning(model, tokenizer, device):
    """比较语法和推理的内在维度"""
    print("\n" + "="*70)
    print("Exp3: 语法→推理的维度跃迁 ★★★★★")
    print("="*70)

    results = {}

    # Part A: 语法内在维度(4角色)
    print(f"\n  Part A: 语法内在维度")
    syntax_roles = ROLE_SETS["4roles"]
    H_syn, labels_syn = collect_hidden_states_multirole(
        model, tokenizer, device, syntax_roles, EXTENDED_ROLES_DATA)
    print(f"  语法: {len(H_syn)} 样本, {len(syntax_roles)} 角色")

    max_dim_syn = min(H_syn.shape[0], H_syn.shape[1]) - 1
    dims_syn = [d for d in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50] if d < max_dim_syn]

    # 全空间准确率
    scaler_s = StandardScaler()
    H_syn_scaled = scaler_s.fit_transform(H_syn)
    probe_s = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
    cv_syn_full = cross_val_score(probe_s, H_syn_scaled, labels_syn, cv=5, scoring='accuracy')
    syn_full = cv_syn_full.mean()
    print(f"  语法全空间CV: {syn_full:.4f}")

    syn_dim_curve = {}
    syn_dim_95 = None
    syn_dim_90 = None

    for dim in dims_syn:
        pca = PCA(n_components=dim)
        H_pca = pca.fit_transform(H_syn)
        probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        cv = cross_val_score(probe, H_pca, labels_syn, cv=5, scoring='accuracy')
        syn_dim_curve[dim] = float(cv.mean())

        if syn_dim_90 is None and cv.mean() >= syn_full * 0.90:
            syn_dim_90 = dim
        if syn_dim_95 is None and cv.mean() >= syn_full * 0.95:
            syn_dim_95 = dim

    print(f"  语法内在维度(95%): {syn_dim_95}, (90%): {syn_dim_90}")

    # Part B: 推理内在维度(4类)
    print(f"\n  Part B: 推理内在维度(因果/条件/递进/转折)")
    H_reas, labels_reas = collect_reasoning_hidden_states(model, tokenizer, device)
    print(f"  推理: {len(H_reas)} 样本, 4 类别")

    max_dim_reas = min(H_reas.shape[0], H_reas.shape[1]) - 1
    dims_reas = [d for d in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50] if d < max_dim_reas]

    # 全空间准确率
    scaler_r = StandardScaler()
    H_reas_scaled = scaler_r.fit_transform(H_reas)
    probe_r = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
    cv_reas_full = cross_val_score(probe_r, H_reas_scaled, labels_reas, cv=5, scoring='accuracy')
    reas_full = cv_reas_full.mean()
    print(f"  推理全空间CV: {reas_full:.4f}")

    reas_dim_curve = {}
    reas_dim_95 = None
    reas_dim_90 = None

    for dim in dims_reas:
        pca = PCA(n_components=dim)
        H_pca = pca.fit_transform(H_reas)
        probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        cv = cross_val_score(probe, H_pca, labels_reas, cv=5, scoring='accuracy')
        reas_dim_curve[dim] = float(cv.mean())

        if reas_dim_90 is None and cv.mean() >= reas_full * 0.90:
            reas_dim_90 = dim
        if reas_dim_95 is None and cv.mean() >= reas_full * 0.95:
            reas_dim_95 = dim

    print(f"  推理内在维度(95%): {reas_dim_95}, (90%): {reas_dim_90}")

    # Part C: 维度跃迁比
    if syn_dim_95 is not None and reas_dim_95 is not None:
        ratio_95 = reas_dim_95 / syn_dim_95
        print(f"\n  ★ 维度跃迁比(95%): {ratio_95:.2f}")
    else:
        ratio_95 = None
        print(f"\n  ★ 维度跃迁比: 无法计算(某个维度为N/A)")

    if syn_dim_90 is not None and reas_dim_90 is not None:
        ratio_90 = reas_dim_90 / syn_dim_90
        print(f"  ★ 维度跃迁比(90%): {ratio_90:.2f}")
    else:
        ratio_90 = None

    # Part D: 推理的频段分析
    print(f"\n  Part D: 推理信息的频段分布")
    W_U = get_W_U(model)
    eigenvalues, eigenvectors = compute_W_U_eigenspectrum(W_U)
    d_model = H_reas.shape[1]

    n_bands = 5
    band_size = d_model // n_bands

    print(f"  {'Band':>8s} {'Syntax_CV':>10s} {'Reason_CV':>10s} {'Ratio':>7s}")

    for bi in range(n_bands):
        start = bi * band_size
        end = min((bi + 1) * band_size, d_model)
        U_band = eigenvectors[:, start:end]

        # 语法
        H_syn_band = H_syn @ U_band
        probe_syn = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        try:
            cv_syn = cross_val_score(probe_syn, H_syn_band, labels_syn, cv=5, scoring='accuracy')
        except:
            cv_syn = np.array([0.0])

        # 推理
        H_reas_band = H_reas @ U_band
        probe_reas = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        try:
            cv_reas = cross_val_score(probe_reas, H_reas_band, labels_reas, cv=5, scoring='accuracy')
        except:
            cv_reas = np.array([0.0])

        ratio = cv_syn.mean() / max(cv_reas.mean(), 0.001)
        print(f"  band{bi+1:>4d} {cv_syn.mean():10.4f} {cv_reas.mean():10.4f} {ratio:7.2f}")

    results = {
        'syntax_dim_curve': syn_dim_curve,
        'syntax_dim_95': syn_dim_95,
        'syntax_dim_90': syn_dim_90,
        'syntax_full_cv': float(syn_full),
        'reasoning_dim_curve': reas_dim_curve,
        'reasoning_dim_95': reas_dim_95,
        'reasoning_dim_90': reas_dim_90,
        'reasoning_full_cv': float(reas_full),
        'dim_transition_ratio_95': float(ratio_95) if ratio_95 else None,
        'dim_transition_ratio_90': float(ratio_90) if ratio_90 else None,
    }

    return results


# ===== Exp4: 语法角色距离vs语法关系 =====
def exp4_geometry_vs_syntax(model, tokenizer, device):
    """检验几何距离是否与语法关系距离相关"""
    print("\n" + "="*70)
    print("Exp4: 几何距离 vs 语法关系距离 ★★★★★")
    print("="*70)

    # 收集10角色hidden states
    role_names = ROLE_NAMES_10
    H, labels = collect_hidden_states_multirole(
        model, tokenizer, device, role_names, EXTENDED_ROLES_DATA)
    print(f"  收集到 {len(H)} 个样本, {len(role_names)} 角色")

    # Part A: 计算内在空间中的质心距离
    # 用PCA降到内在维度(10维)
    max_dim = min(15, min(H.shape[0], H.shape[1]) - 1)
    pca = PCA(n_components=max_dim)
    H_pca = pca.fit_transform(H)

    # 各角色的质心(PCA空间)
    centers = {}
    for ri, role in enumerate(role_names):
        mask = labels == ri
        centers[role] = H_pca[mask].mean(axis=0)

    center_vecs = np.array([centers[r] for r in role_names])

    # 尝试不同PCA维度的距离矩阵
    dims_to_test = [2, 3, 5, 7, 10]
    dims_to_test = [d for d in dims_to_test if d <= max_dim]

    best_corr = -1
    best_dim = None

    print(f"\n  Part A: 不同PCA维度下的几何-语法相关性")
    print(f"  {'PCA_dim':>8s} {'Spearman_r':>12s} {'p_value':>10s}")

    for dim in dims_to_test:
        cv_pca = center_vecs[:, :dim]
        geo_dist = squareform(pdist(cv_pca, metric='euclidean'))

        # 语法距离矩阵
        syn_dist = np.zeros((len(role_names), len(role_names)))
        for ri, role in enumerate(role_names):
            for ci in range(len(role_names)):
                syn_dist[ri, ci] = SYNTAX_DISTANCE_MATRIX[role][ci]

        # 取上三角
        geo_tri = geo_dist[np.triu_indices(len(role_names), k=1)]
        syn_tri = syn_dist[np.triu_indices(len(role_names), k=1)]

        corr, pval = spearmanr(geo_tri, syn_tri)
        print(f"  {dim:8d} {corr:12.4f} {pval:10.4f}")

        if abs(corr) > best_corr:
            best_corr = abs(corr)
            best_dim = dim

    print(f"\n  ★ 最佳PCA维度: {best_dim}, 相关性: {best_corr:.4f}")

    # Part B: 最佳维度的详细分析
    cv_best = center_vecs[:, :best_dim]
    geo_dist_best = squareform(pdist(cv_best, metric='euclidean'))
    syn_dist_full = np.zeros((len(role_names), len(role_names)))
    for ri, role in enumerate(role_names):
        for ci in range(len(role_names)):
            syn_dist_full[ri, ci] = SYNTAX_DISTANCE_MATRIX[role][ci]

    print(f"\n  Part B: 几何距离矩阵(PCA-{best_dim}D)")
    header = "         " + "  ".join(f"{r[:5]:>6s}" for r in role_names)
    print(f"  {header}")
    for ri, role in enumerate(role_names):
        row = f"  {role:>8s}"
        for ci in range(len(role_names)):
            row += f"  {geo_dist_best[ri, ci]:6.3f}"
        print(row)

    print(f"\n  语法关系距离矩阵:")
    print(f"  {header}")
    for ri, role in enumerate(role_names):
        row = f"  {role:>8s}"
        for ci in range(len(role_names)):
            row += f"  {syn_dist_full[ri, ci]:6.1f}"
        print(row)

    # Part C: 角色对距离排序对比
    geo_tri = geo_dist_best[np.triu_indices(len(role_names), k=1)]
    syn_tri = syn_dist_full[np.triu_indices(len(role_names), k=1)]

    # 按语法距离排序
    pairs = []
    idx = 0
    for ri in range(len(role_names)):
        for ci in range(ri+1, len(role_names)):
            pairs.append((role_names[ri], role_names[ci], geo_tri[idx], syn_tri[idx]))
            idx += 1

    pairs.sort(key=lambda x: x[3])  # 按语法距离排序

    print(f"\n  Part C: 角色对距离排序(按语法距离)")
    print(f"  {'Pair':>20s} {'Geo_dist':>10s} {'Syn_dist':>10s}")
    for r1, r2, gd, sd in pairs[:15]:
        print(f"  {r1+'-'+r2:>20s} {gd:10.3f} {sd:10.1f}")

    # Part D: 分类分析——同类角色几何距离更近?
    # Core: nsubj, dobj
    # Modifier: amod, poss, det, nummod
    # Functional: advmod, aux
    # Clausal: mark, cc
    groups = {
        'Core': ['nsubj', 'dobj'],
        'Modifier': ['amod', 'poss', 'det', 'nummod'],
        'Functional': ['advmod', 'aux'],
        'Clausal': ['mark', 'cc'],
    }

    print(f"\n  Part D: 同类vs跨类几何距离")
    intra_dists = []
    inter_dists = []

    for gname, grows in groups.items():
        for i in range(len(grows)):
            for j in range(i+1, len(grows)):
                ri = role_names.index(grows[i])
                ci = role_names.index(grows[j])
                intra_dists.append(geo_dist_best[ri, ci])

    group_list = list(groups.keys())
    for gi in range(len(group_list)):
        for gj in range(gi+1, len(group_list)):
            for r1 in groups[group_list[gi]]:
                for r2 in groups[group_list[gj]]:
                    ri = role_names.index(r1)
                    ci = role_names.index(r2)
                    inter_dists.append(geo_dist_best[ri, ci])

    intra_mean = np.mean(intra_dists) if intra_dists else 0
    inter_mean = np.mean(inter_dists) if inter_dists else 0
    print(f"  同类平均距离: {intra_mean:.3f}")
    print(f"  跨类平均距离: {inter_mean:.3f}")
    print(f"  跨类/同类比: {inter_mean/max(intra_mean,0.001):.2f}")
    print(f"  {'同类角色几何上更近!' if inter_mean > intra_mean else '同类角色不比跨类更近'}")

    corr_final, pval_final = spearmanr(geo_tri, syn_tri)

    results = {
        'best_pca_dim': best_dim,
        'spearman_r': float(corr_final),
        'spearman_p': float(pval_final),
        'intra_group_mean_dist': float(intra_mean),
        'inter_group_mean_dist': float(inter_mean),
        'inter_intra_ratio': float(inter_mean / max(intra_mean, 0.001)),
        'pca_explained_var': pca.explained_variance_ratio_[:10].tolist(),
    }

    return results


# ===== 主函数 =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True,
                       choices=[1, 2, 3, 4])
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"CCL-Q Phase11 内在几何与信息论 | Model={args.model} | Exp={args.exp}")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    print(f"  Model: {model_info.model_class}, Layers={model_info.n_layers}, "
          f"d_model={model_info.d_model}")

    try:
        if args.exp == 1:
            results = exp1_syntax_geometry(model, tokenizer, device)
        elif args.exp == 2:
            results = exp2_information_theory(model, tokenizer, device)
        elif args.exp == 3:
            results = exp3_syntax_to_reasoning(model, tokenizer, device)
        elif args.exp == 4:
            results = exp4_geometry_vs_syntax(model, tokenizer, device)

        # 保存结果
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'glm5_temp')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir,
                               f"cclq_exp{args.exp}_{args.model}_results.json")

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
