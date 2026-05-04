"""
CCL-T(Phase 14): 表示空间的几何结构与token位置效应
=============================================================================
核心问题:
  Phase 13发现了3个关键问题:
  1. target token vs last token给出完全不同的维度结果
  2. nsubj-poss等价: 是位置等价还是结构等价?
  3. 词性子空间近正交(70-86°): 正交分解是否有效?

实验:
  Exp1: ★★★★★★★ token位置效应的系统测量
    → 同一语法角色(target token) vs 前一个token vs 后一个token vs last token
    → 测量: 语法信息在哪些位置最集中?
    → 如果target token编码语法最集中: 局部编码假说
    → 如果语法信息扩散到多个位置: 分布式编码假说

  Exp2: ★★★★★★★ nsubj-poss等价的深层测试
    → "The king ruled" (nsubj) vs "The king's crown" (poss)
    → 测量nsubj名词在不同位置(pos=1)和target位置的表示
    → 如果nsubj和poss在pos=1位置就等价: 位置等价假说
    → 如果只在target位置等价: 结构等价假说
    → 加对照: "The king ate" vs "A king's crown" (不同限定词)

  Exp3: ★★★★★ 子空间正交性的信息论解释
    → 将4类词性投影到各自的正交子空间
    → 测量: 正交投影后信息保留多少?
    → 如果信息保留>95%: 正交分解是有效的
    → 如果信息保留<80%: 正交分解损失太多
    → 对比: 通用PCA子空间 vs 词性专属子空间

  Exp4: ★★★★★ 语法角色的几何: 线性探针 vs 流形
    → UMAP降维后的语法角色分布
    → 如果UMAP上的距离与CV一致: 线性探针是有效的
    → 如果不一致: 需要非线性分析
    → 测量: 语法角色的流形曲率和连通性
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
from scipy.spatial.distance import pdist, squareform

from model_utils import load_model, get_layers, get_model_info, release_model, safe_decode


# ===== Exp1: token位置效应 =====
# 核心设计: 对同一个句子, 提取不同位置的hidden states
# 比较: target位置 / target-1 / target+1 / last 的语法角色编码

POSITION_TEST_DATA = {
    "nsubj": {
        "sentences": [
            ("The king ruled the kingdom wisely", "king", 1),
            ("The doctor treated the patient carefully", "doctor", 1),
            ("The artist painted the portrait beautifully", "artist", 1),
            ("The soldier defended the castle bravely", "soldier", 1),
            ("The teacher explained the lesson clearly", "teacher", 1),
            ("The chef cooked the meal perfectly", "chef", 1),
            ("The student read the textbook carefully", "student", 1),
            ("The president signed the bill firmly", "president", 1),
            ("The woman drove the car safely", "woman", 1),
            ("The man fixed the roof quickly", "man", 1),
            ("The cat sat on the mat quietly", "cat", 1),
            ("The dog ran through the park swiftly", "dog", 1),
        ],
    },
    "dobj": {
        "sentences": [
            ("They crowned the king yesterday", "king", 3),
            ("She visited the doctor recently", "doctor", 3),
            ("He admired the artist greatly", "artist", 3),
            ("We honored the soldier today", "soldier", 3),
            ("You thanked the teacher warmly", "teacher", 3),
            ("The customer tipped the chef generously", "chef", 4),
            ("I praised the student loudly", "student", 3),
            ("The nation elected the president fairly", "president", 4),
            ("The police arrested the woman quickly", "woman", 4),
            ("The company hired the man recently", "man", 4),
            ("She chased the cat away", "cat", 3),
            ("He found the dog outside", "dog", 3),
        ],
    },
    "amod": {
        "sentences": [
            ("The brave king fought hard", "brave", 1),
            ("The kind doctor helped many", "kind", 1),
            ("The creative artist worked well", "creative", 1),
            ("The strong soldier marched far", "strong", 1),
            ("The wise teacher explained clearly", "wise", 1),
            ("The skilled chef cooked perfectly", "skilled", 1),
            ("The bright student read carefully", "bright", 1),
            ("The powerful president decided firmly", "powerful", 1),
            ("The old woman walked slowly", "old", 1),
            ("The tall man stood quietly", "tall", 1),
            ("The beautiful cat sat peacefully", "beautiful", 1),
            ("The large dog ran swiftly", "large", 1),
        ],
    },
    "advmod": {
        "sentences": [
            ("The king ruled wisely forever", "wisely", 2),
            ("The doctor worked carefully always", "carefully", 2),
            ("The artist painted beautifully daily", "beautifully", 2),
            ("The soldier fought bravely there", "bravely", 2),
            ("The teacher spoke clearly again", "clearly", 2),
            ("The chef worked quickly then", "quickly", 2),
            ("The student read carefully alone", "carefully", 2),
            ("The president spoke firmly today", "firmly", 2),
            ("The woman drove slowly home", "slowly", 2),
            ("The man spoke quietly now", "quietly", 2),
            ("The cat ran quickly home", "quickly", 2),
            ("The dog barked loudly today", "loudly", 2),
        ],
    },
}


# ===== Exp2: nsubj-poss等价的深层测试 =====
NSUBJ_POSS_DEEP_DATA = {
    # 组A: nsubj vs poss, 同一个名词, 同一个限定词
    "nsubj_the": {
        "desc": "nsubj: The N V",
        "sentences": [
            "The king ruled wisely",
            "The doctor worked carefully",
            "The artist painted beautifully",
            "The soldier fought bravely",
            "The teacher taught clearly",
            "The chef cooked expertly",
            "The student studied hard",
            "The president decided firmly",
            "The woman spoke softly",
            "The man walked slowly",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "teacher",
            "chef", "student", "president", "woman", "man",
        ],
        "target_pos": 1,  # The N V → N在pos=1
    },
    "poss_the": {
        "desc": "poss: The N's X V",
        "sentences": [
            "The king's crown glittered",
            "The doctor's office opened",
            "The artist's studio looked",
            "The soldier's uniform was",
            "The teacher's book sold",
            "The chef's restaurant opened",
            "The student's essay read",
            "The president's speech inspired",
            "The woman's dress looked",
            "The man's car drove",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "teacher",
            "chef", "student", "president", "woman", "man",
        ],
        "target_pos": 1,  # The N's X → N在pos=1
    },
    # 组B: nsubj with A → 换限定词
    "nsubj_a": {
        "desc": "nsubj: A N V",
        "sentences": [
            "A king ruled wisely",
            "A doctor worked carefully",
            "An artist painted beautifully",
            "A soldier fought bravely",
            "A teacher taught clearly",
            "A chef cooked expertly",
            "A student studied hard",
            "A president decided firmly",
            "A woman spoke softly",
            "A man walked slowly",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "teacher",
            "chef", "student", "president", "woman", "man",
        ],
        "target_pos": 1,  # A N V → N在pos=1
    },
    # 组C: poss without "The" → 测量's的影响
    "poss_her": {
        "desc": "poss: Her N's X V",
        "sentences": [
            "Her king's crown glittered",
            "Her doctor's office opened",
            "Her artist's studio looked",
            "Her soldier's uniform was",
            "Her teacher's book sold",
            "Her chef's restaurant opened",
            "Her student's essay read",
            "Her president's speech inspired",
            "Her woman's dress looked",
            "Her man's car drove",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "teacher",
            "chef", "student", "president", "woman", "man",
        ],
        "target_pos": 1,  # Her N's → N在pos=1
    },
    # 组D: nsubj在更复杂句中(有形容词)
    "nsubj_adj": {
        "desc": "nsubj: The ADJ N V",
        "sentences": [
            "The wise king ruled wisely",
            "The careful doctor worked hard",
            "The creative artist painted well",
            "The brave soldier fought hard",
            "The clear teacher taught well",
            "The expert chef cooked well",
            "The hard student studied hard",
            "The firm president decided well",
            "The soft woman spoke gently",
            "The slow man walked quietly",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "teacher",
            "chef", "student", "president", "woman", "man",
        ],
        "target_pos": 2,  # The ADJ N V → N在pos=2
    },
}


# ===== Exp3: 子空间正交性的信息论解释 =====
# 复用Phase 13的SUBSPACE_DATA
SUBSPACE_POS_DATA = {
    "nouns": {
        "sentences": [
            "Kings ruled the ancient kingdom", "Doctors treated the sick patient",
            "Artists painted the beautiful portrait", "Soldiers defended the mighty castle",
            "Cats sat on the soft mat", "Dogs ran through the green park",
            "Women drove the new car", "Men fixed the broken roof",
            "Students read the thick textbook", "Teachers explained the complex lesson",
            "Presidents signed the important bill", "Chefs cooked the delicious meal",
            "Bakers baked the fresh bread", "Pilots flew the large airplane",
            "Singers sang the happy song", "Farmers grew the golden crops",
        ],
        "target_words": [
            "Kings", "Doctors", "Artists", "Soldiers", "Cats", "Dogs",
            "Women", "Men", "Students", "Teachers", "Presidents", "Chefs",
            "Bakers", "Pilots", "Singers", "Farmers",
        ],
    },
    "verbs": {
        "sentences": [
            "The king ruled wisely", "The doctor treated carefully",
            "The artist painted beautifully", "The soldier fought bravely",
            "The cat jumped quickly", "The dog ran swiftly",
            "The woman drove carefully", "The man spoke loudly",
            "The student studied hard", "The teacher taught well",
            "The president decided firmly", "The chef cooked expertly",
            "The baker baked daily", "The pilot flew safely",
            "The singer performed brilliantly", "The farmer worked diligently",
        ],
        "target_words": [
            "ruled", "treated", "painted", "fought", "jumped", "ran",
            "drove", "spoke", "studied", "taught", "decided", "cooked",
            "baked", "flew", "performed", "worked",
        ],
    },
    "adjectives": {
        "sentences": [
            "The brave king fought", "The kind doctor helped",
            "The creative artist worked", "The strong soldier marched",
            "The beautiful cat sat", "The large dog ran",
            "The old woman walked", "The tall man stood",
            "The bright student read", "The wise teacher explained",
            "The powerful president decided", "The skilled chef cooked",
            "The patient baker waited", "The careful pilot landed",
            "The talented singer performed", "The hardworking farmer harvested",
        ],
        "target_words": [
            "brave", "kind", "creative", "strong", "beautiful", "large",
            "old", "tall", "bright", "wise", "powerful", "skilled",
            "patient", "careful", "talented", "hardworking",
        ],
    },
    "adverbs": {
        "sentences": [
            "He ruled wisely always", "She worked carefully forever",
            "They painted beautifully daily", "He fought bravely there",
            "It ran quickly home", "It barked loudly today",
            "She drove slowly home", "He spoke quietly now",
            "She studied carefully alone", "He explained clearly again",
            "She decided firmly today", "He cooked quickly then",
            "She baked freshly daily", "He flew steadily onward",
            "She sang softly tonight", "He worked diligently always",
        ],
        "target_words": [
            "wisely", "carefully", "beautifully", "bravely", "quickly", "loudly",
            "slowly", "quietly", "carefully", "clearly", "firmly", "quickly",
            "freshly", "steadily", "softly", "diligently",
        ],
    },
}


# ===== Exp4: 语法角色的几何 =====
# 复用Phase 13的NOMINAL_ROLES_DATA + NON_NOMINAL_ROLES_DATA
GEOMETRY_ROLES_DATA = {
    "nsubj": {
        "sentences": [
            "The king ruled the kingdom", "The doctor treated the patient",
            "The artist painted the portrait", "The soldier defended the castle",
            "The cat sat on the mat", "The dog ran through the park",
            "The woman drove the car", "The man fixed the roof",
            "The student read the textbook", "The teacher explained the lesson",
            "The president signed the bill", "The chef cooked the meal",
            "The baker baked the bread", "The pilot flew the airplane",
            "The singer sang a song", "The farmer grew crops",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "cat", "dog",
            "woman", "man", "student", "teacher", "president", "chef",
            "baker", "pilot", "singer", "farmer",
        ],
    },
    "poss": {
        "sentences": [
            "The king's crown glittered brightly", "The doctor's office opened early",
            "The artist's studio looked beautiful", "The soldier's uniform was clean",
            "The cat's tail swished gently", "The dog's bark echoed loudly",
            "The woman's dress looked elegant", "The man's car drove fast",
            "The student's essay read well", "The teacher's book sold quickly",
            "The president's speech inspired many", "The chef's restaurant opened today",
            "The baker's shop smelled wonderful", "The pilot's license was renewed",
            "The singer's voice rang clearly", "The farmer's land was fertile",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "cat", "dog",
            "woman", "man", "student", "teacher", "president", "chef",
            "baker", "pilot", "singer", "farmer",
        ],
    },
    "dobj": {
        "sentences": [
            "They crowned the king yesterday", "She visited the doctor recently",
            "He admired the artist greatly", "We honored the soldier today",
            "She chased the cat away", "He found the dog outside",
            "The police arrested the man quickly", "The company hired the woman recently",
            "I praised the student loudly", "You thanked the teacher warmly",
            "The nation elected the president fairly", "The customer tipped the chef generously",
            "They hired the baker recently", "She admired the pilot greatly",
            "We praised the singer loudly", "He visited the farmer often",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "cat", "dog",
            "man", "woman", "student", "teacher", "president", "chef",
            "baker", "pilot", "singer", "farmer",
        ],
    },
    "amod": {
        "sentences": [
            "The brave king fought hard", "The kind doctor helped many",
            "The creative artist worked well", "The strong soldier marched far",
            "The beautiful cat sat quietly", "The large dog ran swiftly",
            "The old woman walked slowly", "The tall man stood quietly",
            "The bright student read carefully", "The wise teacher explained clearly",
            "The powerful president decided firmly", "The skilled chef cooked perfectly",
            "The patient baker waited calmly", "The careful pilot landed smoothly",
            "The talented singer performed brilliantly", "The hardworking farmer harvested early",
        ],
        "target_words": [
            "brave", "kind", "creative", "strong", "beautiful", "large",
            "old", "tall", "bright", "wise", "powerful", "skilled",
            "patient", "careful", "talented", "hardworking",
        ],
    },
}


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


def collect_hidden_states_at_positions(model, tokenizer, device, role_name, data, layer_idx=-1):
    """收集同一角色在不同位置的hidden states"""
    layers = get_layers(model)
    target_layer = layers[layer_idx] if layer_idx >= 0 else layers[-1]

    results = {
        'target': [],      # target token位置
        'prev': [],        # target-1位置
        'next': [],        # target+1位置
        'last': [],        # 句末位置
        'first': [],       # 首token位置
    }
    labels = []

    for item in data["sentences"]:
        if isinstance(item, tuple):
            sent, target_word, expected_pos = item
        else:
            continue

        toks = tokenizer(sent, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
        seq_len = len(tokens_list)

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

        h_all = captured['h'][0]  # [seq_len, d_model]

        # 收集各位置
        results['target'].append(h_all[dep_idx, :])
        results['prev'].append(h_all[max(0, dep_idx - 1), :])
        results['next'].append(h_all[min(seq_len - 1, dep_idx + 1), :])
        results['last'].append(h_all[-1, :])
        results['first'].append(h_all[0, :])
        labels.append(role_name)

    for key in results:
        if len(results[key]) > 0:
            results[key] = np.array(results[key])

    return results, labels


def collect_hidden_states_target(model, tokenizer, device, role_names, data_dict, layer_idx=-1):
    """收集target token位置的hidden states"""
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


def collect_hidden_states_at_pos(model, tokenizer, device, sentences, target_words, target_pos, layer_idx=-1):
    """收集指定固定位置的hidden states"""
    all_h = []

    layers = get_layers(model)
    target_layer = layers[layer_idx] if layer_idx >= 0 else layers[-1]

    for sent, target_word in zip(sentences, target_words):
        toks = tokenizer(sent, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]

        # 找target word的实际位置
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

        h_all = captured['h'][0]
        # 返回target位置和固定pos位置的表示
        pos_idx = min(target_pos, len(h_all) - 1)
        all_h.append({
            'target_pos': h_all[dep_idx, :],
            'fixed_pos': h_all[pos_idx, :],
            'actual_target_idx': dep_idx,
        })

    return all_h


def compute_intrinsic_dim(H, labels, threshold=0.95):
    """计算内在维度"""
    scaler = StandardScaler()
    H_scaled = scaler.fit_transform(H)

    probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
    cv_full = cross_val_score(probe, H_scaled, labels, cv=5, scoring='accuracy')
    full_acc = cv_full.mean()

    max_dim = min(H.shape[0], H.shape[1]) - 1
    dims_to_test = [d for d in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50] if d < max_dim]

    dim_curve = {}
    intrinsic_dim = None

    for dim in dims_to_test:
        pca = PCA(n_components=dim)
        H_pca = pca.fit_transform(H)
        probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        cv = cross_val_score(probe, H_pca, labels, cv=5, scoring='accuracy')
        dim_curve[dim] = float(cv.mean())
        if intrinsic_dim is None and cv.mean() >= full_acc * threshold:
            intrinsic_dim = dim

    return {
        'full_acc': float(full_acc),
        'dim_curve': dim_curve,
        'intrinsic_dim': intrinsic_dim,
        'threshold': threshold,
    }


def compute_pairwise_cv(H, labels, role_names):
    """计算所有角色对之间的分类CV"""
    results = {}
    n_roles = len(role_names)
    for i in range(n_roles):
        for j in range(i + 1, n_roles):
            mask = (labels == i) | (labels == j)
            H_pair = H[mask]
            labels_pair = labels[mask]
            if len(H_pair) < 8:
                continue
            scaler = StandardScaler()
            H_scaled = scaler.fit_transform(H_pair)
            probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
            cv = cross_val_score(probe, H_scaled, labels_pair, cv=5, scoring='accuracy')
            pair_key = f"{role_names[i]}_vs_{role_names[j]}"
            results[pair_key] = float(cv.mean())
    return results


# ===== Exp1: token位置效应的系统测量 =====
def exp1_token_position_effect(model, tokenizer, device):
    """测量语法角色在不同token位置的编码强度"""
    print("\n" + "="*70)
    print("Exp1: token位置效应的系统测量 ★★★★★★★")
    print("="*70)

    results = {}
    role_names = list(POSITION_TEST_DATA.keys())

    # 收集各角色在各位置的hidden states
    pos_data = {}  # {pos_type: {role: H}}
    for pos_type in ['target', 'prev', 'next', 'last', 'first']:
        pos_data[pos_type] = {}

    for role in role_names:
        data = POSITION_TEST_DATA[role]
        h_dict, _ = collect_hidden_states_at_positions(model, tokenizer, device, role, data)
        for pos_type in h_dict:
            if len(h_dict[pos_type]) > 0:
                pos_data[pos_type][role] = h_dict[pos_type]

    # Part A: 各位置的语法角色分类CV
    print(f"\n  Part A: 各位置的4类语法角色分类CV")
    pos_cvs = {}
    for pos_type in ['target', 'prev', 'next', 'last', 'first']:
        H_list = []
        labels_list = []
        for role_idx, role in enumerate(role_names):
            if role in pos_data[pos_type]:
                H_list.append(pos_data[pos_type][role])
                labels_list.extend([role_idx] * len(pos_data[pos_type][role]))

        if len(H_list) < 4 or len(labels_list) < 20:
            print(f"    {pos_type}: 样本不足, 跳过")
            continue

        H = np.vstack(H_list)
        labels = np.array(labels_list)
        scaler = StandardScaler()
        H_scaled = scaler.fit_transform(H)
        probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        cv = cross_val_score(probe, H_scaled, labels, cv=5, scoring='accuracy')
        pos_cvs[pos_type] = float(cv.mean())
        print(f"    {pos_type}: CV={cv.mean():.4f} (n={len(labels)})")

    results['position_cvs'] = pos_cvs

    # Part B: 各位置的内在维度
    print(f"\n  Part B: 各位置的内在维度(95%)")
    pos_dims = {}
    for pos_type in ['target', 'prev', 'next', 'last', 'first']:
        H_list = []
        labels_list = []
        for role_idx, role in enumerate(role_names):
            if role in pos_data[pos_type]:
                H_list.append(pos_data[pos_type][role])
                labels_list.extend([role_idx] * len(pos_data[pos_type][role]))

        if len(H_list) < 4 or len(labels_list) < 20:
            continue

        H = np.vstack(H_list)
        labels = np.array(labels_list)
        dim_result = compute_intrinsic_dim(H, labels, threshold=0.95)
        pos_dims[pos_type] = dim_result['intrinsic_dim']
        print(f"    {pos_type}: dim={dim_result['intrinsic_dim']}, CV={dim_result['full_acc']:.4f}")

    results['position_dims'] = pos_dims

    # Part C: 各位置的两两CV (nsubj-poss重点关注)
    print(f"\n  Part C: nsubj vs 其他角色 在各位置")
    for pos_type in ['target', 'prev', 'next', 'last']:
        if 'nsubj' not in pos_data.get(pos_type, {}) or 'dobj' not in pos_data.get(pos_type, {}):
            continue
        H_list = []
        labels_list = []
        for role_idx, role in enumerate(role_names):
            if role in pos_data[pos_type]:
                H_list.append(pos_data[pos_type][role])
                labels_list.extend([role_idx] * len(pos_data[pos_type][role]))

        if len(H_list) < 4:
            continue

        H = np.vstack(H_list)
        labels = np.array(labels_list)
        pairwise = compute_pairwise_cv(H, labels, role_names)
        print(f"    {pos_type}:")
        for pair, cv in sorted(pairwise.items()):
            if 'nsubj' in pair:
                print(f"      {pair}: CV={cv:.4f}")
        results[f'pairwise_{pos_type}'] = pairwise

    # Part D: 位置效应总结
    print(f"\n  Part D: 位置效应总结")
    if pos_cvs:
        best_pos = max(pos_cvs, key=pos_cvs.get)
        print(f"    语法编码最强位置: {best_pos} (CV={pos_cvs[best_pos]:.4f})")
        worst_pos = min(pos_cvs, key=pos_cvs.get)
        print(f"    语法编码最弱位置: {worst_pos} (CV={pos_cvs[worst_pos]:.4f})")

        target_cv = pos_cvs.get('target', 0)
        last_cv = pos_cvs.get('last', 0)
        print(f"    target vs last: {target_cv:.4f} vs {last_cv:.4f}")
        if target_cv > last_cv:
            print(f"    → 语法信息在target token最集中! 局部编码假说成立")
        else:
            print(f"    → 语法信息在last token更集中! 分布式编码假说")

    return results


# ===== Exp2: nsubj-poss等价的深层测试 =====
def exp2_nsubj_poss_deep(model, tokenizer, device):
    """测试nsubj-poss等价是位置等价还是结构等价"""
    print("\n" + "="*70)
    print("Exp2: nsubj-poss等价的深层测试 ★★★★★★★")
    print("="*70)

    results = {}

    group_names = list(NSUBJ_POSS_DEEP_DATA.keys())

    # 收集各组的target位置和pos=1位置的表示
    all_target = {}  # {group: H at target token}
    all_pos1 = {}    # {group: H at position 1}

    for group in group_names:
        data = NSUBJ_POSS_DEEP_DATA[group]
        h_results = collect_hidden_states_at_pos(
            model, tokenizer, device,
            data["sentences"], data["target_words"], data["target_pos"])

        target_h = np.array([h['target_pos'] for h in h_results])
        pos1_h = np.array([h['fixed_pos'] for h in h_results])

        all_target[group] = target_h
        all_pos1[group] = pos1_h
        print(f"  {group} ({data['desc']}): {len(target_h)} samples")

    # Part A: nsubj vs poss 在target位置的CV (Phase 13结果复现)
    print(f"\n  Part A: nsubj vs poss 在target位置的CV")
    if 'nsubj_the' in all_target and 'poss_the' in all_target:
        H_pair = np.vstack([all_target['nsubj_the'], all_target['poss_the']])
        labels = np.array([0] * len(all_target['nsubj_the']) + [1] * len(all_target['poss_the']))
        if len(H_pair) >= 10:
            scaler = StandardScaler()
            H_scaled = scaler.fit_transform(H_pair)
            probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
            cv = cross_val_score(probe, H_scaled, labels, cv=5, scoring='accuracy')
            print(f"    nsubj_the vs poss_the (target pos): CV={cv.mean():.4f}")
            results['nsubj_vs_poss_target'] = float(cv.mean())

    # Part B: nsubj vs poss 在pos=1位置的CV
    print(f"\n  Part B: nsubj vs poss 在pos=1位置的CV")
    if 'nsubj_the' in all_pos1 and 'poss_the' in all_pos1:
        H_pair = np.vstack([all_pos1['nsubj_the'], all_pos1['poss_the']])
        labels = np.array([0] * len(all_pos1['nsubj_the']) + [1] * len(all_pos1['poss_the']))
        if len(H_pair) >= 10:
            scaler = StandardScaler()
            H_scaled = scaler.fit_transform(H_pair)
            probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
            cv = cross_val_score(probe, H_scaled, labels, cv=5, scoring='accuracy')
            print(f"    nsubj_the vs poss_the (pos=1): CV={cv.mean():.4f}")
            results['nsubj_vs_poss_pos1'] = float(cv.mean())

    # Part C: 限定词效应 (The vs A)
    print(f"\n  Part C: 限定词效应 (The vs A)")
    for pos_type, pos_data_dict in [('target', all_target), ('pos1', all_pos1)]:
        if 'nsubj_the' in pos_data_dict and 'nsubj_a' in pos_data_dict:
            H_pair = np.vstack([pos_data_dict['nsubj_the'], pos_data_dict['nsubj_a']])
            labels = np.array([0] * len(pos_data_dict['nsubj_the']) + [1] * len(pos_data_dict['nsubj_a']))
            if len(H_pair) >= 10:
                scaler = StandardScaler()
                H_scaled = scaler.fit_transform(H_pair)
                probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
                cv = cross_val_score(probe, H_scaled, labels, cv=5, scoring='accuracy')
                print(f"    nsubj_the vs nsubj_a ({pos_type}): CV={cv.mean():.4f}")
                results[f'nsubj_the_vs_a_{pos_type}'] = float(cv.mean())

    # Part D: poss限定词效应 (The vs Her)
    print(f"\n  Part D: poss限定词效应 (The vs Her)")
    for pos_type, pos_data_dict in [('target', all_target), ('pos1', all_pos1)]:
        if 'poss_the' in pos_data_dict and 'poss_her' in pos_data_dict:
            H_pair = np.vstack([pos_data_dict['poss_the'], pos_data_dict['poss_her']])
            labels = np.array([0] * len(pos_data_dict['poss_the']) + [1] * len(pos_data_dict['poss_her']))
            if len(H_pair) >= 10:
                scaler = StandardScaler()
                H_scaled = scaler.fit_transform(H_pair)
                probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
                cv = cross_val_score(probe, H_scaled, labels, cv=5, scoring='accuracy')
                print(f"    poss_the vs poss_her ({pos_type}): CV={cv.mean():.4f}")
                results[f'poss_the_vs_her_{pos_type}'] = float(cv.mean())

    # Part E: 形容词对nsubj的影响
    print(f"\n  Part E: 形容词对nsubj的影响")
    for pos_type, pos_data_dict in [('target', all_target), ('pos1', all_pos1)]:
        if 'nsubj_the' in pos_data_dict and 'nsubj_adj' in pos_data_dict:
            H_pair = np.vstack([pos_data_dict['nsubj_the'], pos_data_dict['nsubj_adj']])
            labels = np.array([0] * len(pos_data_dict['nsubj_the']) + [1] * len(pos_data_dict['nsubj_adj']))
            if len(H_pair) >= 10:
                scaler = StandardScaler()
                H_scaled = scaler.fit_transform(H_pair)
                probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
                cv = cross_val_score(probe, H_scaled, labels, cv=5, scoring='accuracy')
                print(f"    nsubj_the vs nsubj_adj ({pos_type}): CV={cv.mean():.4f}")
                results[f'nsubj_the_vs_adj_{pos_type}'] = float(cv.mean())

    # Part F: 等价性判断
    print(f"\n  Part F: 等价性判断")
    target_cv = results.get('nsubj_vs_poss_target', 0.5)
    pos1_cv = results.get('nsubj_vs_poss_pos1', 0.5)

    print(f"    nsubj vs poss at target: CV={target_cv:.4f}")
    print(f"    nsubj vs poss at pos=1:  CV={pos1_cv:.4f}")

    if pos1_cv < 0.55:
        print(f"    → 位置等价假说: nsubj和poss在pos=1就不可分!")
        print(f"    → 说明: 两者等价不是因为语法角色, 而是因为位置相同")
    elif target_cv < 0.55 and pos1_cv > 0.7:
        print(f"    → 结构等价假说: nsubj和poss只在target位置不可分!")
        print(f"    → 说明: 两者等价是因为语法角色在target位置的编码相同")
    else:
        print(f"    → 混合等价: nsubj和poss在多个位置都相似")
        print(f"    → 可能: 同一个词在不同语法角色中的表示变化不大")

    results['equivalence_type'] = 'position' if pos1_cv < 0.55 else ('structure' if target_cv < 0.55 and pos1_cv > 0.7 else 'mixed')

    return results


# ===== Exp3: 子空间正交性的信息论解释 =====
def exp3_subspace_information(model, tokenizer, device):
    """测试词性子空间的正交投影信息保留"""
    print("\n" + "="*70)
    print("Exp3: 子空间正交性的信息论解释 ★★★★★")
    print("="*70)

    results = {}

    pos_categories = list(SUBSPACE_POS_DATA.keys())

    # 收集4类词性的hidden states
    pos_H = {}
    for pos in pos_categories:
        data = SUBSPACE_POS_DATA[pos]
        H, _ = collect_hidden_states_target(model, tokenizer, device, [pos], {pos: data})
        pos_H[pos] = H
        print(f"  {pos}: {len(H)} samples")

    # 全部hidden states
    H_all, labels_all = collect_hidden_states_target(
        model, tokenizer, device, pos_categories, SUBSPACE_POS_DATA)
    print(f"  Total: {len(H_all)} samples")

    if len(H_all) < 20:
        print(f"  样本不足, 跳过")
        return results

    # Part A: 全空间分类准确率(基准)
    print(f"\n  Part A: 全空间分类准确率(基准)")
    scaler = StandardScaler()
    H_scaled = scaler.fit_transform(H_all)
    probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
    cv_full = cross_val_score(probe, H_scaled, labels_all, cv=5, scoring='accuracy')
    baseline_acc = float(cv_full.mean())
    print(f"    全空间4类词性CV: {baseline_acc:.4f}")
    results['baseline_acc'] = baseline_acc

    # Part B: 通用PCA子空间投影
    print(f"\n  Part B: 通用PCA子空间投影(不区分词性)")
    n_components_list = [4, 10, 20, 50]
    for nc in n_components_list:
        if nc >= min(H_all.shape[0], H_all.shape[1]):
            continue
        pca = PCA(n_components=nc)
        H_pca = pca.fit_transform(H_scaled)
        probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        cv = cross_val_score(probe, H_pca, labels_all, cv=5, scoring='accuracy')
        print(f"    PCA dim={nc}: CV={cv.mean():.4f}, 方差保留={sum(pca.explained_variance_ratio_):.4f}")
        results[f'pca_{nc}_cv'] = float(cv.mean())
        results[f'pca_{nc}_var'] = float(sum(pca.explained_variance_ratio_))

    # Part C: 词性专属子空间投影
    print(f"\n  Part C: 词性专属子空间投影")
    # 步骤:
    # 1. 对每个词性计算PCA子空间
    # 2. Gram-Schmidt正交化得到4个正交子空间
    # 3. 将所有数据投影到这些子空间
    # 4. 测量分类准确率

    n_sub = 5  # 每个词性的子空间维度
    sub_bases = {}

    for pos in pos_categories:
        H = pos_H[pos]
        if len(H) < n_sub + 1:
            continue
        scaler_pos = StandardScaler()
        H_scaled_pos = scaler_pos.fit_transform(H)
        pca = PCA(n_components=n_sub)
        pca.fit(H_scaled_pos)
        sub_bases[pos] = pca.components_.T  # [d_model, n_sub]

    # Gram-Schmidt正交化
    print(f"\n  Gram-Schmidt正交化...")
    ortho_bases = {}
    used_basis = np.zeros((H_all.shape[1], 0))

    for pos in pos_categories:
        if pos not in sub_bases:
            continue
        basis = sub_bases[pos]  # [d_model, n_sub]
        # 正交化: 减去与已有基的投影
        for i in range(basis.shape[1]):
            v = basis[:, i]
            for j in range(used_basis.shape[1]):
                v = v - np.dot(v, used_basis[:, j]) * used_basis[:, j]
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                v = v / norm
                used_basis = np.column_stack([used_basis, v])

        ortho_bases[pos] = used_basis[:, -n_sub:]  # 取最新的n_sub个
        print(f"    {pos}: 正交基维度={used_basis.shape[1]}")

    # 将所有数据投影到正交子空间
    total_dim = used_basis.shape[1]
    print(f"  正交子空间总维度: {total_dim}")

    H_ortho = H_scaled @ used_basis  # [n_samples, total_dim]
    probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
    cv_ortho = cross_val_score(probe, H_ortho, labels_all, cv=5, scoring='accuracy')
    print(f"    正交子空间投影CV: {cv_ortho.mean():.4f}")
    print(f"    信息保留: {cv_ortho.mean() / max(baseline_acc, 0.01) * 100:.1f}%")
    results['ortho_subspace_cv'] = float(cv_ortho.mean())
    results['ortho_subspace_dim'] = total_dim
    results['ortho_info_retention'] = float(cv_ortho.mean() / max(baseline_acc, 0.01))

    # Part D: 各词性子空间的重建误差
    print(f"\n  Part D: 各词性子空间的重建误差")
    for pos in pos_categories:
        H = pos_H[pos]
        if len(H) < 5:
            continue
        scaler_pos = StandardScaler()
        H_scaled_pos = scaler_pos.fit_transform(H)

        # 重建: 用该词性的PCA基重建
        if pos in sub_bases:
            basis = sub_bases[pos]
            proj = H_scaled_pos @ basis @ basis.T  # 投影+重建
            error = np.mean(np.linalg.norm(H_scaled_pos - proj, axis=1))
            total_norm = np.mean(np.linalg.norm(H_scaled_pos, axis=1))
            recon_ratio = 1.0 - error / max(total_norm, 1e-10)
            print(f"    {pos}: 自身子空间重建率={recon_ratio:.4f}")
            results[f'{pos}_self_recon'] = float(recon_ratio)

        # 用其他词性的基重建
        for other_pos in pos_categories:
            if other_pos == pos or other_pos not in sub_bases:
                continue
            other_basis = sub_bases[other_pos]
            proj = H_scaled_pos @ other_basis @ other_basis.T
            error = np.mean(np.linalg.norm(H_scaled_pos - proj, axis=1))
            total_norm = np.mean(np.linalg.norm(H_scaled_pos, axis=1))
            recon_ratio = 1.0 - error / max(total_norm, 1e-10)
            if other_pos == 'nouns' or pos == 'nouns':
                print(f"    {pos}用{other_pos}的基重建: {recon_ratio:.4f}")
            results[f'{pos}_by_{other_pos}_recon'] = float(recon_ratio)

    return results


# ===== Exp4: 语法角色的几何 =====
def exp4_syntax_geometry(model, tokenizer, device):
    """语法角色的几何结构: 线性可分性 vs 流形"""
    print("\n" + "="*70)
    print("Exp4: 语法角色的几何结构 ★★★★★")
    print("="*70)

    results = {}

    role_names = list(GEOMETRY_ROLES_DATA.keys())

    # 收集4类语法角色的hidden states
    H_all, labels_all = collect_hidden_states_target(
        model, tokenizer, device, role_names, GEOMETRY_ROLES_DATA)
    print(f"  Total: {len(H_all)} samples, dim={H_all.shape[1]}")

    if len(H_all) < 20:
        print(f"  样本不足, 跳过")
        return results

    # Part A: 线性可分性(基准)
    print(f"\n  Part A: 线性可分性(基准)")
    scaler = StandardScaler()
    H_scaled = scaler.fit_transform(H_all)
    probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
    cv = cross_val_score(probe, H_scaled, labels_all, cv=5, scoring='accuracy')
    print(f"    4类语法角色线性CV: {cv.mean():.4f}")
    results['linear_cv'] = float(cv.mean())

    # Part B: PCA降维后的几何
    print(f"\n  Part B: PCA降维后的几何结构")
    n_pca = min(50, min(H_all.shape[0], H_all.shape[1]) - 1)
    pca = PCA(n_components=n_pca)
    H_pca = pca.fit_transform(H_scaled)

    # 计算每个角色的类中心和类内散布
    class_centers = {}
    class_spreads = {}
    for role_idx, role in enumerate(role_names):
        mask = labels_all == role_idx
        H_role = H_pca[mask]
        center = np.mean(H_role, axis=0)
        spread = np.mean(np.linalg.norm(H_role - center, axis=1))
        class_centers[role] = center.tolist()
        class_spreads[role] = float(spread)
        print(f"    {role}: center_norm={np.linalg.norm(center):.4f}, spread={spread:.4f}")

    results['class_centers'] = class_centers
    results['class_spreads'] = class_spreads

    # Part C: 类间距离矩阵
    print(f"\n  Part C: 类间距离(余弦相似度)")
    center_matrix = np.array([class_centers[r] for r in role_names])
    # 余弦相似度
    from sklearn.metrics.pairwise import cosine_similarity
    cos_sim = cosine_similarity(center_matrix)
    for i, r1 in enumerate(role_names):
        for j, r2 in enumerate(role_names):
            if j > i:
                print(f"    {r1} vs {r2}: cos_sim={cos_sim[i,j]:.4f}, "
                      f"angle={np.degrees(np.arccos(np.clip(cos_sim[i,j], -1, 1))):.1f}°")
    results['center_cosine_sim'] = cos_sim.tolist()

    # Part D: UMAP降维 (尝试import)
    print(f"\n  Part D: 非线性降维(PCA proxy)")
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=10, min_dist=0.3)
        H_umap = reducer.fit_transform(H_pca[:, :20])  # 先用前20个PCA维度
        has_umap = True
        print(f"    UMAP降维成功")
    except ImportError:
        # 用PCA 2维代替
        H_umap = H_pca[:, :2]
        has_umap = False
        print(f"    UMAP不可用, 用PCA-2D代替")

    # 在2D空间中计算各类的分离度
    for role_idx, role in enumerate(role_names):
        mask = labels_all == role_idx
        H_role_2d = H_umap[mask]
        if len(H_role_2d) > 1:
            center_2d = np.mean(H_role_2d, axis=0)
            spread_2d = np.mean(np.linalg.norm(H_role_2d - center_2d, axis=1))
            print(f"    {role}: 2D spread={spread_2d:.4f}")
            results[f'{role}_2d_spread'] = float(spread_2d)

    # Part E: 流形曲率估计
    print(f"\n  Part E: 流形曲率估计")
    # 简单方法: 计算每个点与其k近邻的局部维度
    from sklearn.neighbors import NearestNeighbors
    k = min(5, len(H_pca) - 1)
    nn = NearestNeighbors(n_neighbors=k+1).fit(H_pca[:, :20])
    distances, indices = nn.kneighbors(H_pca[:, :20])

    # 局部PCA维度(在每个点的k近邻上做PCA, 看需要多少维解释95%方差)
    local_dims = []
    for i in range(len(H_pca)):
        neighbor_idx = indices[i, 1:]  # 排除自身
        H_local = H_pca[neighbor_idx, :20]
        if len(H_local) < 3:
            continue
        pca_local = PCA()
        pca_local.fit(H_local)
        cumvar = np.cumsum(pca_local.explained_variance_ratio_)
        local_dim = np.searchsorted(cumvar, 0.95) + 1
        local_dims.append(local_dim)

    if local_dims:
        avg_local_dim = np.mean(local_dims)
        print(f"    平均局部维度(95%): {avg_local_dim:.2f}")
        results['avg_local_dim'] = float(avg_local_dim)

        # 各角色的局部维度
        for role_idx, role in enumerate(role_names):
            mask = labels_all == role_idx
            role_local_dims = [local_dims[i] for i in range(len(local_dims))
                              if i < len(labels_all) and labels_all[i] == role_idx]
            if role_local_dims:
                avg_role_dim = np.mean(role_local_dims)
                print(f"    {role}: 局部维度={avg_role_dim:.2f}")
                results[f'{role}_local_dim'] = float(avg_role_dim)

    # Part F: nsubj-poss的几何关系
    print(f"\n  Part F: nsubj-poss的几何关系")
    if 'nsubj' in role_names and 'poss' in role_names:
        nsubj_idx = role_names.index('nsubj')
        poss_idx = role_names.index('poss')
        H_nsubj = H_pca[labels_all == nsubj_idx]
        H_poss = H_pca[labels_all == poss_idx]

        if len(H_nsubj) > 0 and len(H_poss) > 0:
            # 中心距离
            center_nsubj = np.mean(H_nsubj, axis=0)
            center_poss = np.mean(H_poss, axis=0)
            center_dist = np.linalg.norm(center_nsubj - center_poss)
            cos_between = np.dot(center_nsubj, center_poss) / (
                np.linalg.norm(center_nsubj) * np.linalg.norm(center_poss) + 1e-10)
            print(f"    nsubj-poss center distance: {center_dist:.4f}")
            print(f"    nsubj-poss center cosine: {cos_between:.4f}")
            results['nsubj_poss_center_dist'] = float(center_dist)
            results['nsubj_poss_center_cos'] = float(cos_between)

            # 配对样本距离: 同一词在nsubj和poss位置的表示差异
            # 使用相同的target_words
            nsubj_words = GEOMETRY_ROLES_DATA['nsubj']['target_words']
            poss_words = GEOMETRY_ROLES_DATA['poss']['target_words']

            # 计算同类词在nsubj和poss位置的平均距离
            if len(H_nsubj) > 0 and len(H_poss) > 0:
                # 随机配对(因为句子数量可能不完全匹配)
                min_len = min(len(H_nsubj), len(H_poss))
                paired_dists = np.linalg.norm(
                    H_nsubj[:min_len] - H_poss[:min_len], axis=1)
                # 跨类配对(随机)
                rand_dists = []
                for _ in range(100):
                    i = np.random.randint(len(H_nsubj))
                    j = np.random.randint(len(H_poss))
                    rand_dists.append(np.linalg.norm(H_nsubj[i] - H_poss[j]))
                rand_dists = np.array(rand_dists)

                print(f"    配对距离(同词): mean={np.mean(paired_dists):.4f}")
                print(f"    随机距离(跨词): mean={np.mean(rand_dists):.4f}")
                print(f"    距离比: {np.mean(paired_dists)/max(np.mean(rand_dists), 1e-10):.4f}")
                results['nsubj_poss_paired_dist'] = float(np.mean(paired_dists))
                results['nsubj_poss_random_dist'] = float(np.mean(rand_dists))

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
    print(f"CCL-T Phase14 表示空间几何+token位置效应 | Model={args.model} | Exp={args.exp}")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    print(f"  Model: {model_info.model_class}, Layers={model_info.n_layers}, "
          f"d_model={model_info.d_model}")

    try:
        if args.exp == 1:
            results = exp1_token_position_effect(model, tokenizer, device)
        elif args.exp == 2:
            results = exp2_nsubj_poss_deep(model, tokenizer, device)
        elif args.exp == 3:
            results = exp3_subspace_information(model, tokenizer, device)
        elif args.exp == 4:
            results = exp4_syntax_geometry(model, tokenizer, device)

        # 保存结果
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'glm5_temp')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir,
                               f"cclt_exp{args.exp}_{args.model}_results.json")

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
