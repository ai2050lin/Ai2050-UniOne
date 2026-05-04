"""
CCMD(Phase 24): 修正实验设计——控制词汇效应
=============================================================================
Phase 23发现的核心问题:
  ★★★ 1D共线可能是"同一名词集效应", 不是"语法1D效应"!
  → nsubj/dobj/iobj/poss用了同一组12个名词(king, doctor, ...)
  → prep的target全是"of"
  → 1D共线可能是因为12个名词的hidden state在中间层天然1D

Phase 24实验设计:
  Exp1: ★★★★★★★★★★★ 控制词汇的实验设计
    → 每个语法角色使用完全不同的词汇集(零重叠)
    → nsubj: 人物名词(king, doctor, ...)
    → dobj:  物品名词(book, table, ...)
    → amod:  形容词(brave, kind, ...)
    → advmod: 程度副词(extremely, very, ...)
    → det:   限定词(This, That, ...)
    → prep:  多种介词(in, on, at, with, ...)
    → iobj:  抽象名词(advice, support, ...) + 不同动词
    → poss:  不同的所有者
    → 如果1D共线消失 → Phase 23的1D是词汇效应
    → 如果1D共线仍存在 → 语法偏移确实1D

  Exp2: ★★★★★★★★★ 随机token基线
    → 收集随机token在中间层的hidden state
    → 如果随机token也1D共线 → 中间层1D是架构效应, 不是语法效应
    → 如果随机token不1D共线 → 语法角色的1D共线是有意义的

  Exp3: ★★★★★★★★ PC1方向分析
    → 中间层PC1的方向是什么?
    → 与position embedding的关系?
    → 与token norm的关系?
    → 检查PC1是否是"norm方向"(范数最大的方向)

  Exp4: ★★★★★★★ 混合词汇设计(交叉验证)
    → 同一词在不同角色: king作为nsubj vs king作为dobj
    → 不同词在同一角色: king vs book作为nsubj
    → 这可以分离: 角色效应 vs 词汇效应
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
from sklearn.decomposition import PCA
from collections import defaultdict

from model_utils import load_model, get_layers, get_model_info, release_model, safe_decode, MODEL_CONFIGS


# ===== Exp1: 控制词汇——每个角色用完全不同的词汇集 =====
# 关键设计: 零词汇重叠, 消除"同一名词集效应"

CONTROLLED_ROLES_DATA = {
    "nsubj": {
        # 主语: 人物名词(与Phase23不同的人物, 或者同组但在全新语境)
        "sentences": [
            "The king ruled wisely",
            "The doctor worked carefully",
            "The artist created beautifully",
            "The soldier fought bravely",
            "The teacher spoke clearly",
            "The chef cooked perfectly",
            "The student studied quietly",
            "The writer composed slowly",
            "The pilot navigated smoothly",
            "The farmer planted diligently",
            "The judge decided fairly",
            "The nurse cared kindly",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "teacher",
            "chef", "student", "writer", "pilot", "farmer",
            "judge", "nurse",
        ],
        "word_category": "person_nouns",
    },
    "dobj": {
        # 宾语: 物品名词(完全不同于nsubj的人物名词)
        "sentences": [
            "She found the book",
            "He broke the table",
            "They painted the wall",
            "We opened the door",
            "I read the letter",
            "You moved the chair",
            "The girl dropped the glass",
            "The boy fixed the window",
            "She closed the box",
            "He cleaned the floor",
            "They built the bridge",
            "We repaired the car",
        ],
        "target_words": [
            "book", "table", "wall", "door", "letter",
            "chair", "glass", "window", "box", "floor",
            "bridge", "car",
        ],
        "word_category": "object_nouns",
    },
    "amod": {
        # 形容词修饰语: 形容词
        "sentences": [
            "The brave soldier fought",
            "The kind woman helped",
            "The tall man walked",
            "The old house stood",
            "The young boy ran",
            "The rich king donated",
            "The cold water froze",
            "The hot sun burned",
            "The dark night fell",
            "The soft pillow rested",
            "The hard rock cracked",
            "The sweet fruit ripened",
        ],
        "target_words": [
            "brave", "kind", "tall", "old", "young",
            "rich", "cold", "hot", "dark", "soft",
            "hard", "sweet",
        ],
        "word_category": "adjectives",
    },
    "advmod": {
        # 副词修饰语: 程度副词
        "sentences": [
            "He ran extremely fast",
            "She spoke very softly",
            "They worked quite hard",
            "We sang incredibly well",
            "I slept remarkably deeply",
            "You answered absolutely correctly",
            "The dog barked very loudly",
            "The wind blew extremely strongly",
            "The rain fell quite heavily",
            "The sun shone incredibly brightly",
            "The baby cried very loudly",
            "The bird sang remarkably sweetly",
        ],
        "target_words": [
            "extremely", "very", "quite", "incredibly", "remarkably",
            "absolutely", "very", "extremely", "quite", "incredibly",
            "very", "remarkably",
        ],
        "word_category": "degree_adverbs",
    },
    "det": {
        # 限定词: 限定词
        "sentences": [
            "This book is interesting",
            "That table is broken",
            "Every student must study",
            "Each person has a chance",
            "Some animals are wild",
            "Any child can learn",
            "This road leads home",
            "That idea was brilliant",
            "Every morning was cold",
            "Each answer was correct",
            "Some music is relaxing",
            "Any time works for me",
        ],
        "target_words": [
            "This", "That", "Every", "Each", "Some",
            "Any", "This", "That", "Every", "Each",
            "Some", "Any",
        ],
        "word_category": "determiners",
    },
    "prep": {
        # 介词: 多种介词(不是全用"of"!)
        "sentences": [
            "The cat sat on the mat",
            "She walked through the garden",
            "He jumped over the fence",
            "They swam across the river",
            "We drove under the bridge",
            "I hid behind the tree",
            "The bird flew above the clouds",
            "She stood beside the window",
            "He walked along the beach",
            "They sat around the table",
            "We looked into the room",
            "I ran toward the door",
        ],
        "target_words": [
            "on", "through", "over", "across", "under",
            "behind", "above", "beside", "along", "around",
            "into", "toward",
        ],
        "word_category": "prepositions",
    },
    "iobj": {
        # 间接宾语: 抽象名词 + 不同动词
        "sentences": [
            "She gave advice to him",
            "He offered support to them",
            "They provided shelter to us",
            "We sent information to her",
            "I gave permission to the team",
            "You showed kindness to everyone",
            "The teacher gave homework to students",
            "The boss offered training to workers",
            "She provided guidance to children",
            "He sent money to family",
            "They gave attention to details",
            "We offered help to neighbors",
        ],
        "target_words": [
            "advice", "support", "shelter", "information", "permission",
            "kindness", "homework", "training", "guidance", "money",
            "attention", "help",
        ],
        "word_category": "abstract_nouns",
    },
    "poss": {
        # 所有格: 不同的所有者(非人物)
        "sentences": [
            "The city's streets were crowded",
            "The ocean's waves crashed loudly",
            "The mountain's peak was snow-covered",
            "The forest's trees grew tall",
            "The river's current flowed fast",
            "The country's flag flew proudly",
            "The school's bell rang loudly",
            "The church's tower stood high",
            "The island's beach was beautiful",
            "The planet's atmosphere was thin",
            "The company's profit increased",
            "The team's spirit was strong",
        ],
        "target_words": [
            "city", "ocean", "mountain", "forest", "river",
            "country", "school", "church", "island", "planet",
            "company", "team",
        ],
        "word_category": "collective_nouns",
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
    if len(word) > 0:
        first_char = word[0]
        for i, tok in enumerate(tokens):
            if first_char in tok:
                return i
    return None


def collect_hs_at_layer(model, tokenizer, device, sentences, target_words, layer_idx):
    """在指定层收集target token的hidden states"""
    layers = get_layers(model)
    if layer_idx >= len(layers):
        layer_idx = len(layers) - 1
    if layer_idx < 0:
        layer_idx = len(layers) + layer_idx
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


def collect_all_data_controlled(model, tokenizer, device, layers_to_test):
    """收集所有角色在所有层的hidden states (控制词汇版)"""
    all_data = {}
    role_names = list(CONTROLLED_ROLES_DATA.keys())
    print(f"  收集8个角色的数据(控制词汇版): {role_names}")
    
    for layer_idx in layers_to_test:
        layer_data = {}
        for role in role_names:
            role_info = CONTROLLED_ROLES_DATA[role]
            hs = collect_hs_at_layer(
                model, tokenizer, device,
                role_info["sentences"],
                role_info["target_words"],
                layer_idx
            )
            if hs is not None and len(hs) >= 3:
                layer_data[role] = hs
        all_data[layer_idx] = layer_data
        
        n_roles = len(layer_data)
        n_samples = sum(len(v) for v in layer_data.values())
        print(f"    Layer {layer_idx}: {n_roles} roles, {n_samples} samples")
    
    return all_data, role_names


# ==================== Exp1: 控制词汇的纤维维度 ====================

def exp1_controlled_vocab(model_name, model, tokenizer, device, layers_to_test):
    """Exp1: 控制词汇——每个角色用不同词汇集, 验证1D共线是否消失"""
    print(f"\n{'='*60}")
    print(f"Exp1: 控制词汇的纤维维度 (验证1D共线是否是词汇效应)")
    print(f"{'='*60}")
    
    all_data, role_names = collect_all_data_controlled(model, tokenizer, device, layers_to_test)
    
    results = {"model": model_name, "exp": 1, "experiment": "controlled_vocab", "layers": {}}
    
    for layer_idx in sorted(all_data.keys()):
        layer_data = all_data[layer_idx]
        if len(layer_data) < 4:
            print(f"  Layer {layer_idx}: too few roles ({len(layer_data)}), skipping")
            continue
        
        # 计算每个角色的中心
        role_centers = {}
        for role, hs in layer_data.items():
            role_centers[role] = np.mean(hs, axis=0)
        
        # grand mean
        all_centers = np.array([role_centers[r] for r in layer_data.keys()])
        grand_mean = np.mean(all_centers, axis=0)
        
        # δ̄_r = center_r - grand_mean
        delta_matrix = np.array([role_centers[r] - grand_mean for r in layer_data.keys()])
        n_roles = delta_matrix.shape[0]
        
        # PCA分析
        pca = PCA()
        pca.fit(delta_matrix)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        
        eff_dim_95 = np.searchsorted(cumvar, 0.95) + 1
        eff_dim_90 = np.searchsorted(cumvar, 0.90) + 1
        eff_dim_80 = np.searchsorted(cumvar, 0.80) + 1
        
        eigenvalues = pca.explained_variance_ratio_
        pr = (np.sum(eigenvalues))**2 / np.sum(eigenvalues**2)
        
        # ★★★ 共同PCA: 所有角色数据合并做PCA
        all_hs = np.vstack([hs for hs in layer_data.values()])
        joint_pca = PCA()
        joint_pca.fit(all_hs)
        joint_cumvar = np.cumsum(joint_pca.explained_variance_ratio_)
        
        # ★★★ 逐角色PCA: 每个角色内部做PCA
        role_internal_dims = {}
        for role, hs in layer_data.items():
            if len(hs) >= 3:
                rpca = PCA()
                rpca.fit(hs)
                rc = np.cumsum(rpca.explained_variance_ratio_)
                role_internal_dims[role] = {
                    "pc1_pct": round(float(rpca.explained_variance_ratio_[0]*100), 2),
                    "eff_dim_90": int(np.searchsorted(rc, 0.90) + 1),
                }
        
        # ★★★ 角色间PCA (去掉每个角色的中心)
        # 残差 = h - role_center, 检查残差是否也1D
        residuals_centered = []
        for role, hs in layer_data.items():
            center = role_centers[role]
            for h in hs:
                residuals_centered.append(h - center)
        residuals_matrix = np.array(residuals_centered)
        res_pca = PCA()
        res_pca.fit(residuals_matrix)
        res_cumvar = np.cumsum(res_pca.explained_variance_ratio_)
        
        # ★★★ 词汇类别信息
        vocab_categories = {role: CONTROLLED_ROLES_DATA[role]["word_category"] for role in layer_data.keys()}
        
        layer_result = {
            "n_roles": n_roles,
            "delta_pca_eigenvalues_pct": [round(float(v*100), 2) for v in pca.explained_variance_ratio_[:min(8, n_roles)]],
            "delta_cumvar_pct": [round(float(v*100), 1) for v in cumvar[:min(8, n_roles)]],
            "delta_eff_dim_80": int(eff_dim_80),
            "delta_eff_dim_90": int(eff_dim_90),
            "delta_eff_dim_95": int(eff_dim_95),
            "delta_participation_ratio": round(float(pr), 2),
            "joint_pca_pc1_pct": round(float(joint_pca.explained_variance_ratio_[0]*100), 2),
            "joint_pca_pc1_pc2_pct": [round(float(joint_pca.explained_variance_ratio_[i]*100), 2) for i in range(min(3, len(joint_pca.explained_variance_ratio_)))],
            "joint_pca_k1_var": round(float(joint_cumvar[0]*100), 2),
            "joint_pca_k3_var": round(float(joint_cumvar[min(2, len(joint_cumvar)-1)]*100), 2),
            "residual_pca_pc1_pct": round(float(res_pca.explained_variance_ratio_[0]*100), 2),
            "residual_pca_k3_var": round(float(res_cumvar[min(2, len(res_cumvar)-1)]*100), 2),
            "role_internal_dims": role_internal_dims,
            "vocab_categories": vocab_categories,
        }
        
        results["layers"][str(layer_idx)] = layer_result
        
        print(f"  Layer {layer_idx}: n_roles={n_roles}")
        print(f"    δ̄_r PCA: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, eff_dim_95={eff_dim_95}")
        print(f"    联合PCA: PC1={joint_pca.explained_variance_ratio_[0]*100:.1f}%")
        print(f"    残差PCA: PC1={res_pca.explained_variance_ratio_[0]*100:.1f}%")
        print(f"    角色内部维度: {role_internal_dims}")
    
    # ★★★ 关键对比: Phase 23(同一名词集) vs Phase 24(控制词汇)
    results["comparison_note"] = (
        "如果控制词汇后δ̄_r维度从1D变为高维 → Phase 23的1D是词汇效应。"
        "如果控制词汇后δ̄_r仍1D → 语法偏移确实低维。"
        "联合PCA的PC1如果从>99%降到<80% → 中间层1D也是词汇效应。"
    )
    
    return results


# ==================== Exp2: 随机token基线 ====================

def exp2_random_baseline(model_name, model, tokenizer, device, layers_to_test):
    """Exp2: 随机token基线——检查中间层1D是否是架构效应"""
    print(f"\n{'='*60}")
    print(f"Exp2: 随机token基线 (检查中间层1D是否是架构效应)")
    print(f"{'='*60}")
    
    # 生成随机句子(用随机常见词)
    # 5组不同长度的随机句子, 每组12句
    np.random.seed(42)
    
    random_word_lists = [
        # 组1: 随机5词句
        ["The cat sat on mats", "A dog ran very fast", "Big fish swim deep down",
         "Red birds fly high up", "Old trees grow slow now", "New stars shine bright here",
         "Dark clouds move fast", "Small ants work hard", "Long roads go far away",
         "Deep water flows cold", "High walls stand firm", "Cold wind blew hard"],
        # 组2: 随机6词句
        ["The quick brown fox jumped", "A large green frog hopped", "Some tiny blue fish swam",
         "That old gray cat slept", "Each small red bird sang", "This big white dog barked",
         "The tall dark man walked", "A young smart girl read", "Some warm soft rain fell",
         "That cold hard ice froze", "Each sweet ripe fruit hung", "This thin dry leaf fell"],
        # 组3: 随机4词句
        ["Cats eat fish", "Dogs chase cats", "Birds fly south", "Fish swim deep",
         "Trees grow tall", "Flowers bloom red", "Rivers flow fast", "Mountains stand high",
         "Stars shine bright", "Wind blows cold", "Rain falls down", "Sun rises early"],
        # 组4: 语法正确但语义随机
        ["The probability approaches zero", "His calculation revealed patterns",
         "The algorithm found solutions", "Her analysis identified trends",
         "The experiment produced data", "Their method improved accuracy",
         "The simulation generated results", "Our theory predicted outcomes",
         "The measurement confirmed values", "His observation detected signals",
         "The model processed inputs", "Her research discovered relationships"],
        # 组5: 简单常见词组合
        ["Time goes by slowly", "Life moves on forward", "Work gets done quickly",
         "Food tastes good today", "Water runs clear here", "Fire burns hot always",
         "Earth stays firm below", "Air flows free above", "Light shines far wide",
         "Sound travels fast far", "Heat spreads out slow", "Cold stays in deep"],
    ]
    
    # 每组取target word(第3个词)
    target_indices = []  # 每组中target token的位置
    for group in random_word_lists:
        # target = 句子中的第2个实词(通常是动词/形容词)
        # 简单策略: 取第3个token(通常是句子中间的词)
        target_indices.append(2)  # 大致是第3个位置
    
    results = {"model": model_name, "exp": 2, "experiment": "random_baseline", "layers": {}}
    
    for layer_idx in layers_to_test:
        layers = get_layers(model)
        if layer_idx >= len(layers):
            layer_idx = len(layers) - 1
        target_layer = layers[layer_idx]
        
        group_pcas = {}
        all_random_hs = []
        
        for g_idx, (word_list, tgt_pos) in enumerate(zip(random_word_lists, target_indices)):
            group_hs = []
            for sent in word_list:
                toks = tokenizer(sent, return_tensors="pt").to(device)
                input_ids = toks.input_ids
                tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
                
                # 取中间位置的token
                mid_pos = len(tokens_list) // 2
                
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
                
                h_vec = captured['h'][0, mid_pos, :]
                group_hs.append(h_vec)
                all_random_hs.append(h_vec)
            
            if len(group_hs) >= 3:
                gpca = PCA()
                gpca.fit(group_hs)
                group_pcas[f"group_{g_idx}"] = {
                    "n_samples": len(group_hs),
                    "pc1_pct": round(float(gpca.explained_variance_ratio_[0]*100), 2),
                    "pc2_pct": round(float(gpca.explained_variance_ratio_[1]*100), 2) if len(gpca.explained_variance_ratio_) > 1 else 0,
                    "eff_dim_90": int(np.searchsorted(np.cumsum(gpca.explained_variance_ratio_), 0.90) + 1),
                }
        
        # 合并所有随机句的hidden states
        if len(all_random_hs) >= 10:
            all_random_hs = np.array(all_random_hs)
            joint_pca = PCA()
            joint_pca.fit(all_random_hs)
            joint_cumvar = np.cumsum(joint_pca.explained_variance_ratio_)
            
            layer_result = {
                "total_random_samples": len(all_random_hs),
                "joint_pca_pc1_pct": round(float(joint_pca.explained_variance_ratio_[0]*100), 2),
                "joint_pca_pc1_pc2_pc3_pct": [round(float(joint_pca.explained_variance_ratio_[i]*100), 2) for i in range(min(3, len(joint_pca.explained_variance_ratio_)))],
                "joint_pca_k1_var": round(float(joint_cumvar[0]*100), 2),
                "joint_pca_k3_var": round(float(joint_cumvar[min(2, len(joint_cumvar)-1)]*100), 2),
                "joint_pca_k10_var": round(float(joint_cumvar[min(9, len(joint_cumvar)-1)]*100), 2),
                "group_pcas": group_pcas,
            }
        else:
            layer_result = {"total_random_samples": len(all_random_hs), "note": "too few samples"}
        
        results["layers"][str(layer_idx)] = layer_result
        
        print(f"  Layer {layer_idx}:")
        if len(all_random_hs) >= 10:
            print(f"    随机句联合PCA: PC1={joint_pca.explained_variance_ratio_[0]*100:.1f}%")
            print(f"    各组PC1: {[(k, v['pc1_pct']) for k, v in group_pcas.items()]}")
    
    results["comparison_note"] = (
        "如果随机token在中间层也1D共线(PC1>99%) → 中间层1D是架构效应, 不是语法效应。"
        "如果随机token不1D共线(PC1<80%) → 语法角色的1D共线是有意义的结构。"
    )
    
    return results


# ==================== Exp3: PC1方向分析 ====================

def exp3_pc1_direction(model_name, model, tokenizer, device, layers_to_test):
    """Exp3: 分析PC1方向——检查是否是norm方向或position方向"""
    print(f"\n{'='*60}")
    print(f"Exp3: PC1方向分析 (检查PC1是否是norm/position方向)")
    print(f"{'='*60}")
    
    all_data, role_names = collect_all_data_controlled(model, tokenizer, device, layers_to_test)
    
    results = {"model": model_name, "exp": 3, "experiment": "pc1_direction", "layers": {}}
    
    for layer_idx in sorted(all_data.keys()):
        layer_data = all_data[layer_idx]
        if len(layer_data) < 3:
            continue
        
        # 合并所有数据
        all_hs = []
        all_labels = []
        for role, hs in layer_data.items():
            for h in hs:
                all_hs.append(h)
                all_labels.append(role)
        all_hs = np.array(all_hs)
        
        # PCA
        pca = PCA()
        pca.fit(all_hs)
        pc1 = pca.components_[0]  # PC1方向
        
        # ★★★ 检查1: PC1与hidden state norm的关系
        # 如果PC1是norm方向, 那么PC1投影应该与norm高度相关
        norms = np.linalg.norm(all_hs, axis=1)
        pc1_projections = all_hs @ pc1  # 在PC1上的投影
        
        # 相关性
        norm_proj_corr = np.corrcoef(norms, pc1_projections)[0, 1]
        
        # ★★★ 检查2: 去掉norm后是否还有1D结构
        # 方法: L2归一化后再做PCA
        all_hs_normed = all_hs / (norms[:, None] + 1e-10)
        normed_pca = PCA()
        normed_pca.fit(all_hs_normed)
        
        # ★★★ 检查3: 去掉均值后PC1的方向
        centered_hs = all_hs - all_hs.mean(axis=0)
        mean_vec = all_hs.mean(axis=0)
        mean_norm = np.linalg.norm(mean_vec)
        
        # PC1与均值方向的关系
        pc1_mean_cos = np.dot(pc1, mean_vec) / (np.linalg.norm(pc1) * mean_norm + 1e-10)
        
        # ★★★ 检查4: PC1投影 vs 角色标签
        # 如果PC1能区分角色, 那么PC1投影在不同角色间应该有显著差异
        role_pc1_means = {}
        for role in layer_data.keys():
            role_hs = layer_data[role]
            role_proj = role_hs @ pc1
            role_pc1_means[role] = round(float(np.mean(role_proj)), 4)
        
        # 角色间PC1投影的范围
        pc1_mean_values = [v for v in role_pc1_means.values()]
        pc1_range = max(pc1_mean_values) - min(pc1_mean_values)
        pc1_std_between = np.std(pc1_mean_values)
        
        # ★★★ 检查5: 逐角色norm
        role_norms = {}
        for role, hs in layer_data.items():
            role_norms[role] = round(float(np.mean(np.linalg.norm(hs, axis=1))), 4)
        
        # ★★★ 检查6: 去norm后的角色可分性
        # 用normed PCA的前3维做简单分类准确率
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        normed_proj = normed_pca.transform(all_hs_normed)[:, :min(3, normed_pca.n_components_)]
        labels_idx = [list(layer_data.keys()).index(l) for l in all_labels]
        if len(set(labels_idx)) >= 2 and len(normed_proj) >= 10:
            lda = LinearDiscriminantAnalysis()
            try:
                lda.fit(normed_proj, labels_idx)
                normed_lda_acc = round(float(lda.score(normed_proj, labels_idx)), 3)
            except Exception:
                normed_lda_acc = None
        else:
            normed_lda_acc = None
        
        layer_result = {
            "pc1_var_pct": round(float(pca.explained_variance_ratio_[0]*100), 2),
            "pc2_var_pct": round(float(pca.explained_variance_ratio_[1]*100), 2) if len(pca.explained_variance_ratio_) > 1 else 0,
            "norm_proj_correlation": round(float(norm_proj_corr), 4),
            "normed_pca_pc1_pct": round(float(normed_pca.explained_variance_ratio_[0]*100), 2),
            "normed_pca_pc2_pct": round(float(normed_pca.explained_variance_ratio_[1]*100), 2) if len(normed_pca.explained_variance_ratio_) > 1 else 0,
            "pc1_mean_cosine": round(float(pc1_mean_cos), 4),
            "mean_norm": round(float(mean_norm), 4),
            "role_pc1_means": role_pc1_means,
            "pc1_range_between_roles": round(float(pc1_range), 4),
            "pc1_std_between_roles": round(float(pc1_std_between), 4),
            "role_norms": role_norms,
            "normed_lda_accuracy": normed_lda_acc,
        }
        
        results["layers"][str(layer_idx)] = layer_result
        
        print(f"  Layer {layer_idx}:")
        print(f"    PC1 var={pca.explained_variance_ratio_[0]*100:.1f}%, norm-proj corr={norm_proj_corr:.4f}")
        print(f"    Normed PCA PC1={normed_pca.explained_variance_ratio_[0]*100:.1f}%")
        print(f"    PC1-mean cosine={pc1_mean_cos:.4f}")
        print(f"    角色PC1均值: {role_pc1_means}")
        print(f"    角色norms: {role_norms}")
        if normed_lda_acc is not None:
            print(f"    Normed LDA准确率: {normed_lda_acc:.3f}")
    
    results["interpretation_guide"] = {
        "norm_proj_corr_high": "PC1是norm方向, 1D共线是因为范数差异",
        "normed_pca_still_1D": "去norm后仍1D, 说明1D结构不来自范数",
        "pc1_mean_cos_high": "PC1方向接近均值方向, 1D共线因为所有h接近均值",
        "normed_lda_high": "去norm后角色仍可分, 语法信息在方向上而非范数上",
    }
    
    return results


# ==================== Exp4: 交叉验证——同一词不同角色 ====================

def exp4_cross_validation(model_name, model, tokenizer, device, layers_to_test):
    """Exp4: 交叉验证——同一词在不同角色, 分离角色效应与词汇效应"""
    print(f"\n{'='*60}")
    print(f"Exp4: 交叉验证 (同一词不同角色, 分离角色效应vs词汇效应)")
    print(f"{'='*60}")
    
    # 设计: king出现在nsubj和dobj中, book也出现在nsubj和dobj中
    # 这样可以做2x2因素分析: 角色(nsubj/dobj) x 词汇(king/book)
    
    cross_data = {
        "nsubj_king": {
            "sentence": "The king ruled wisely",
            "target": "king",
        },
        "dobj_king": {
            "sentence": "They crowned the king yesterday",
            "target": "king",
        },
        "nsubj_book": {
            "sentence": "The book sold quickly",
            "target": "book",
        },
        "dobj_book": {
            "sentence": "She read the book carefully",
            "target": "book",
        },
        # 扩展: 更多词
        "nsubj_doctor": {
            "sentence": "The doctor worked carefully",
            "target": "doctor",
        },
        "dobj_doctor": {
            "sentence": "She visited the doctor recently",
            "target": "doctor",
        },
        "nsubj_table": {
            "sentence": "The table stood firmly",
            "target": "table",
        },
        "dobj_table": {
            "sentence": "He moved the table slowly",
            "target": "table",
        },
        # 更多角色
        "amod_king": {
            "sentence": "The brave king fought hard",
            "target": "brave",  # 形容词修饰king
        },
        "prep_king": {
            "sentence": "The crown of the king glittered",
            "target": "of",  # 介词
        },
    }
    
    results = {"model": model_name, "exp": 4, "experiment": "cross_validation", "layers": {}}
    
    for layer_idx in layers_to_test:
        layers = get_layers(model)
        if layer_idx >= len(layers):
            layer_idx = len(layers) - 1
        target_layer = layers[layer_idx]
        
        hs_dict = {}
        for cond_name, cond_info in cross_data.items():
            sent = cond_info["sentence"]
            target_word = cond_info["target"]
            
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
            hs_dict[cond_name] = h_vec
        
        # ★★★ 2x2因素分析: 角色(nsubj/dobj) x 词汇(king/book/doctor/table)
        # 计算角色偏移和词汇偏移
        
        # 角色偏移: h(nsubj, w) - h(dobj, w) 对同一词w
        role_shifts = {}
        for word in ["king", "book", "doctor", "table"]:
            ns_key = f"nsubj_{word}"
            do_key = f"dobj_{word}"
            if ns_key in hs_dict and do_key in hs_dict:
                shift = hs_dict[ns_key] - hs_dict[do_key]
                role_shifts[word] = shift
        
        # 词汇偏移: h(r, king) - h(r, book) 对同一角色r
        word_shifts = {}
        for role in ["nsubj", "dobj"]:
            k_key = f"{role}_king"
            b_key = f"{role}_book"
            d_key = f"{role}_doctor"
            t_key = f"{role}_table"
            
            pairs = []
            if k_key in hs_dict and b_key in hs_dict:
                pairs.append(("king-book", hs_dict[k_key] - hs_dict[b_key]))
            if k_key in hs_dict and d_key in hs_dict:
                pairs.append(("king-doctor", hs_dict[k_key] - hs_dict[d_key]))
            if b_key in hs_dict and t_key in hs_dict:
                pairs.append(("book-table", hs_dict[b_key] - hs_dict[t_key]))
            
            for pair_name, shift in pairs:
                word_shifts[f"{role}_{pair_name}"] = shift
        
        # ★★★ 角色偏移的一致性: 不同词的角色偏移是否同方向?
        role_shift_list = list(role_shifts.values())
        if len(role_shift_list) >= 2:
            # 两两余弦相似度
            cos_matrix = np.zeros((len(role_shift_list), len(role_shift_list)))
            for i in range(len(role_shift_list)):
                for j in range(len(role_shift_list)):
                    si = role_shift_list[i]
                    sj = role_shift_list[j]
                    cos_matrix[i, j] = np.dot(si, sj) / (np.linalg.norm(si) * np.linalg.norm(sj) + 1e-10)
            
            # 非对角线元素的平均(角色偏移一致性)
            off_diag = cos_matrix[np.triu_indices(len(role_shift_list), k=1)]
            role_shift_consistency = round(float(np.mean(off_diag)), 4) if len(off_diag) > 0 else None
        else:
            cos_matrix = None
            role_shift_consistency = None
        
        # ★★★ 角色偏移 vs 词汇偏移的范数比较
        role_shift_norms = [round(float(np.linalg.norm(v)), 4) for v in role_shifts.values()]
        word_shift_norms = [round(float(np.linalg.norm(v)), 4) for v in word_shifts.values()]
        
        mean_role_norm = round(float(np.mean(role_shift_norms)), 4) if role_shift_norms else None
        mean_word_norm = round(float(np.mean(word_shift_norms)), 4) if word_shift_norms else None
        
        # ★★★ Δ = A + B 分解(2x2设计)
        # A_role = 角色主效应 = mean_w[h(r,w) - h(r',w)]
        # B_word = 角色x词汇交互 = h(r,w) - h(r',w) - A_role
        if len(role_shifts) >= 2:
            A_role = np.mean(list(role_shifts.values()), axis=0)
            B_residuals = {w: shift - A_role for w, shift in role_shifts.items()}
            B_norms = {w: round(float(np.linalg.norm(b)), 4) for w, b in B_residuals.items()}
            A_norm = round(float(np.linalg.norm(A_role)), 4)
        else:
            A_role = None
            B_residuals = {}
            B_norms = {}
            A_norm = None
        
        layer_result = {
            "n_conditions": len(hs_dict),
            "conditions_found": list(hs_dict.keys()),
            "role_shift_consistency": role_shift_consistency,
            "role_shift_norms": {k: round(float(np.linalg.norm(v)), 4) for k, v in role_shifts.items()},
            "word_shift_norms": {k: round(float(np.linalg.norm(v)), 4) for k, v in word_shifts.items()},
            "mean_role_shift_norm": mean_role_norm,
            "mean_word_shift_norm": mean_word_norm,
            "A_role_norm": A_norm,
            "B_residual_norms": B_norms,
            "A_to_AplusB_ratio": round(float(A_norm / (A_norm + sum(B_norms.values()))), 4) if A_norm and B_norms else None,
        }
        
        results["layers"][str(layer_idx)] = layer_result
        
        print(f"  Layer {layer_idx}:")
        print(f"    条件数: {len(hs_dict)}")
        print(f"    角色偏移一致性: {role_shift_consistency}")
        print(f"    角色偏移范数: {layer_result['role_shift_norms']}")
        print(f"    词汇偏移范数: {layer_result['word_shift_norms']}")
        print(f"    A/B分解: A_norm={A_norm}, B_norms={B_norms}")
    
    results["interpretation_guide"] = {
        "role_shift_consistency_high": "不同词的角色偏移方向一致 → 角色效应是系统性的",
        "role_shift_consistency_low": "角色偏移方向不一致 → 角色效应依赖词汇",
        "A_dominant": "A_ℓ(r)主导 → 语法纤维是低维的",
        "B_dominant": "B_ℓ(r,x)主导 → 角色偏移是词依赖的",
    }
    
    return results


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="Phase 24: 控制词汇效应验证")
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"],
                       help="模型名称")
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3, 4],
                       help="实验编号(1-4)")
    args = parser.parse_args()
    
    model_name = args.model
    exp_num = args.exp
    
    # 加载模型
    if model_name == "deepseek7b":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        cfg = MODEL_CONFIGS[model_name]
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], quantization_config=bnb_config, device_map="auto",
            trust_remote_code=True, local_files_only=True,
        )
        model.eval()
        device = next(model.parameters()).device
        print(f"[ccmd] {model_name} loaded with 8bit, device={device}")
    else:
        model, tokenizer, device = load_model(model_name)
    
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    
    # 选择测试层
    layers_to_test = [0]
    if n_layers > 6:
        step = (n_layers - 1) // 6
        layers_to_test = list(range(0, n_layers, step))[:7]
    if n_layers - 1 not in layers_to_test:
        layers_to_test.append(n_layers - 1)
    layers_to_test = sorted(set(layers_to_test))
    
    print(f"\nModel: {model_name}, Layers: {n_layers}, Test layers: {layers_to_test}")
    
    # 运行实验
    if exp_num == 1:
        results = exp1_controlled_vocab(model_name, model, tokenizer, device, layers_to_test)
    elif exp_num == 2:
        results = exp2_random_baseline(model_name, model, tokenizer, device, layers_to_test)
    elif exp_num == 3:
        results = exp3_pc1_direction(model_name, model, tokenizer, device, layers_to_test)
    elif exp_num == 4:
        results = exp4_cross_validation(model_name, model, tokenizer, device, layers_to_test)
    
    # 保存结果
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "glm5_temp")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"ccmd_exp{exp_num}_{model_name}_results.json")
    
    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(x) for x in obj]
        return obj
    
    results = convert_numpy(results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")
    
    release_model(model)
    print("模型已释放")


if __name__ == "__main__":
    main()
