"""
CCMF(Phase 26): 语义属性编码机制破解
=============================================================================
Phase 25核心发现:
  ★★★ 语义类别不是低维子空间! 类别n90≈随机n90 (ratio≈1.0)
  ★★★ 类别信号在中低方差方向, top-10 PCs不含类别信号
  ★★★ 类别是"群岛"结构 (35-95个连通分量)
  ★★★ 超类(animate/inanimate)在top 3-5 PCs编码

当前最大知识空白:
  ??? 属性(颜色/大小/可食性)在hidden state中如何分布?
  ??? "苹果is-a水果"这种上位关系在神经元级别怎么实现?
  ??? "苹果可吃因为水果可吃"的推理链在神经元上如何走通?

Phase 26实验:
  Exp1: ★★★★★★★★★★★ 属性方向独立性测试
    → 选择20+概念, 每个标注5属性(颜色/大小/可食性/材质/功能)
    → 对每个属性值收集一组词(红色: apple,rose,blood...)
    → 在hidden state空间中: 每个属性是否形成独立方向?
    → 不同属性之间是否正交?

  Exp2: ★★★★★★★★★ is-a关系的精确验证
    → "X is a Y"句式中的hidden state差分分析
    → apple→fruit的偏移 vs orange→fruit的偏移是否平行?
    → 如果平行 → 上位关系是统一方向
    → 如果不平行 → 每个is-a关系是独立的

  Exp3: ★★★★★★★ 推理链的因果追踪
    → 设计三段式推理: "X是Y, Y可以Z → X可以Z"
    → 追踪X的hidden state在推理过程中如何变化
    → 关键检验: 推理结论是否已经在前提中隐含编码?

  Exp4: ★★★★★★★★ 属性干预实验
    → 如果"红色"是一个方向, 沿该方向平移是否改变概念的颜色属性?
    → 真正的因果验证, 不是统计观察
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from collections import defaultdict
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

from model_utils import (load_model, get_layers, get_model_info, release_model,
                         safe_decode, MODEL_CONFIGS, get_W_U, compute_cos,
                         compute_recoding_ratio)

# ============================================================================
# 数据定义: 概念-属性-层级-推理的完整知识网络
# ============================================================================

# 20个核心概念, 每个标注5个属性值
CONCEPTS_WITH_ATTRIBUTES = {
    # 水果类
    "apple":      {"color": "red",    "size": "medium", "edible": "yes", "material": "organic", "function": "food"},
    "orange":     {"color": "orange", "size": "medium", "edible": "yes", "material": "organic", "function": "food"},
    "banana":     {"color": "yellow", "size": "medium", "edible": "yes", "material": "organic", "function": "food"},
    "strawberry": {"color": "red",    "size": "small",  "edible": "yes", "material": "organic", "function": "food"},
    "grape":      {"color": "purple", "size": "small",  "edible": "yes", "material": "organic", "function": "food"},
    # 动物类
    "dog":        {"color": "brown",  "size": "medium", "edible": "no",  "material": "flesh",   "function": "pet"},
    "cat":        {"color": "gray",   "size": "small",  "edible": "no",  "material": "flesh",   "function": "pet"},
    "elephant":   {"color": "gray",   "size": "large",  "edible": "no",  "material": "flesh",   "function": "wild"},
    "eagle":      {"color": "brown",  "size": "medium", "edible": "no",  "material": "flesh",   "function": "wild"},
    "salmon":     {"color": "pink",   "size": "medium", "edible": "yes", "material": "flesh",   "function": "food"},
    # 工具/物品类
    "hammer":     {"color": "gray",   "size": "medium", "edible": "no",  "material": "metal",   "function": "tool"},
    "knife":      {"color": "silver", "size": "small",  "edible": "no",  "material": "metal",   "function": "tool"},
    "chair":      {"color": "brown",  "size": "medium", "edible": "no",  "material": "wood",    "function": "furniture"},
    "shirt":      {"color": "white",  "size": "medium", "edible": "no",  "material": "fabric",  "function": "clothing"},
    "car":        {"color": "black",  "size": "large",  "edible": "no",  "material": "metal",   "function": "vehicle"},
    # 自然物
    "rock":       {"color": "gray",   "size": "medium", "edible": "no",  "material": "stone",   "function": "natural"},
    "tree":       {"color": "green",  "size": "large",  "edible": "no",  "material": "organic", "function": "natural"},
    "flower":     {"color": "red",    "size": "small",  "edible": "no",  "material": "organic", "function": "decoration"},
    "water":      {"color": "clear",  "size": "none",   "edible": "yes", "material": "liquid",  "function": "drink"},
    "cloud":      {"color": "white",  "size": "large",  "edible": "no",  "material": "gas",     "function": "natural"},
}

# 按属性值分组 (用于Exp1: 属性方向独立性测试)
ATTRIBUTE_GROUPS = {
    "color": {
        "red":    ["apple", "strawberry", "flower"],
        "orange": ["orange"],
        "yellow": ["banana"],
        "green":  ["tree"],
        "brown":  ["dog", "eagle", "hammer", "chair"],
        "gray":   ["cat", "elephant", "rock"],
        "silver": ["knife"],
        "black":  ["car"],
        "white":  ["shirt", "cloud"],
        "pink":   ["salmon"],
        "purple": ["grape"],
        "clear":  ["water"],
    },
    "size": {
        "small":  ["cat", "strawberry", "grape", "knife", "flower"],
        "medium": ["apple", "orange", "banana", "dog", "eagle", "salmon", "hammer", "chair", "shirt", "rock"],
        "large":  ["elephant", "car", "tree", "cloud"],
    },
    "edible": {
        "yes": ["apple", "orange", "banana", "strawberry", "grape", "salmon", "water"],
        "no":  ["dog", "cat", "elephant", "eagle", "hammer", "knife", "chair", "shirt", "car", "rock", "tree", "flower", "cloud"],
    },
    "material": {
        "organic": ["apple", "orange", "banana", "strawberry", "grape", "tree", "flower"],
        "flesh":   ["dog", "cat", "elephant", "eagle", "salmon"],
        "metal":   ["hammer", "knife", "car"],
        "wood":    ["chair"],
        "fabric":  ["shirt"],
        "stone":   ["rock"],
        "liquid":  ["water"],
        "gas":     ["cloud"],
    },
    "function": {
        "food":       ["apple", "orange", "banana", "strawberry", "grape", "salmon"],
        "pet":        ["dog", "cat"],
        "wild":       ["elephant", "eagle"],
        "tool":       ["hammer", "knife"],
        "furniture":  ["chair"],
        "clothing":   ["shirt"],
        "vehicle":    ["car"],
        "natural":    ["rock", "tree", "cloud"],
        "decoration": ["flower"],
        "drink":      ["water"],
    },
}

# is-a层级链 (用于Exp2)
ISA_HIERARCHY = {
    # 水果链: apple/orange/banana → fruit → food → object
    "apple→fruit":      {"child": "apple",  "parent": "fruit"},
    "orange→fruit":     {"child": "orange", "parent": "fruit"},
    "banana→fruit":     {"child": "banana", "parent": "fruit"},
    "strawberry→fruit": {"child": "strawberry", "parent": "fruit"},
    "grape→fruit":      {"child": "grape",  "parent": "fruit"},
    # 动物链: dog/cat → animal → organism → object
    "dog→animal":       {"child": "dog",    "parent": "animal"},
    "cat→animal":       {"child": "cat",    "parent": "animal"},
    "eagle→animal":     {"child": "eagle",  "parent": "animal"},
    "elephant→animal":  {"child": "elephant","parent": "animal"},
    "salmon→animal":    {"child": "salmon", "parent": "animal"},
    # 工具链: hammer/knife → tool → object
    "hammer→tool":      {"child": "hammer", "parent": "tool"},
    "knife→tool":       {"child": "knife",  "parent": "tool"},
    # 高层链
    "fruit→food":       {"child": "fruit",  "parent": "food"},
    "animal→organism":  {"child": "animal", "parent": "organism"},
    "tool→object":      {"child": "tool",   "parent": "object"},
    "food→object":      {"child": "food",   "parent": "object"},
}

# 推理链 (用于Exp3)
REASONING_CHAINS = [
    # 可食性推理: X是水果, 水果可以吃 → X可以吃
    {"premise1": "Apple is a fruit",       "premise2": "Fruit can be eaten",     "conclusion": "Apple can be eaten",     "type": "edible_from_category"},
    {"premise1": "Orange is a fruit",      "premise2": "Fruit can be eaten",     "conclusion": "Orange can be eaten",    "type": "edible_from_category"},
    {"premise1": "Salmon is an animal",    "premise2": "Some animals can be eaten","conclusion": "Salmon can be eaten",  "type": "edible_from_category"},
    # 功能推理: X是工具, 工具可以用来做Y → X可以用来做Y
    {"premise1": "Hammer is a tool",       "premise2": "Tools are used for work","conclusion": "Hammer is used for work", "type": "function_from_category"},
    {"premise1": "Knife is a tool",        "premise2": "Tools are used for work","conclusion": "Knife is used for work",  "type": "function_from_category"},
    # 属性推理: X是红色的, 红色是暖色 → X是暖色
    {"premise1": "Apple is red",           "premise2": "Red is a warm color",    "conclusion": "Apple is a warm color",  "type": "color_reasoning"},
    # 反向对照: 不可食的
    {"premise1": "Dog is an animal",       "premise2": "Most animals are not eaten","conclusion": "Dog is not eaten",     "type": "not_edible_from_category"},
    {"premise1": "Rock is a natural object","premise2": "Natural objects are not eaten","conclusion": "Rock is not eaten",  "type": "not_edible_from_category"},
]

# 用于提取hidden state的句子模板
CONCEPT_SENTENCE_TEMPLATES = [
    "The {concept} is here",
    "I see the {concept}",
    "This is a {concept}",
]

ATTRIBUTE_SENTENCE_TEMPLATES = [
    "The {concept} is {attribute}",
    "A {concept} has the property of being {attribute}",
]

ISA_SENTENCE_TEMPLATES = [
    "{child} is a {parent}",
    "A {child} is a kind of {parent}",
]

REASONING_SENTENCE_TEMPLATES = [
    "{premise1}. {premise2}. Therefore {conclusion}.",
    "{premise1}, and {premise2}, so {conclusion}.",
]


# ============================================================================
# 工具函数
# ============================================================================

def find_token_index(tokens, target_word):
    """在token列表中查找目标词的位置 (模糊匹配)"""
    target_lower = target_word.lower().strip()
    # 精确匹配
    for i, t in enumerate(tokens):
        if t.lower().strip() == target_lower:
            return i
    # 前3字符匹配
    for i, t in enumerate(tokens):
        if t.lower().strip()[:3] == target_lower[:3]:
            return i
    # 前2字符匹配
    for i, t in enumerate(tokens):
        if t.lower().strip()[:2] == target_lower[:2]:
            return i
    # 首字符匹配
    for i, t in enumerate(tokens):
        if len(t.strip()) > 0 and t.lower().strip()[0] == target_lower[0]:
            return i
    return -1


def collect_hs_at_layer(model, tokenizer, device, sentences, target_words, layer_idx):
    """在指定层收集hidden states"""
    layers = get_layers(model)
    if layer_idx >= len(layers):
        return None
    
    target_layer = layers[layer_idx]
    all_hs = []
    
    for sent, target in zip(sentences, target_words):
        toks = tokenizer(sent, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
        
        dep_idx = find_token_index(tokens_list, target)
        if dep_idx < 0:
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
        
        if 'h' in captured:
            h_vec = captured['h'][0, dep_idx, :]
            all_hs.append(h_vec)
    
    if len(all_hs) == 0:
        return None
    return np.array(all_hs)


def collect_hs_for_concepts(model, tokenizer, device, concepts, layer_idx, template_idx=0):
    """收集一组概念在指定层的hidden states"""
    templates = CONCEPT_SENTENCE_TEMPLATES
    template = templates[template_idx % len(templates)]
    
    sentences = []
    targets = []
    for concept in concepts:
        sent = template.format(concept=concept)
        sentences.append(sent)
        targets.append(concept)
    
    hs = collect_hs_at_layer(model, tokenizer, device, sentences, targets, layer_idx)
    return hs


def collect_hs_for_attribute(model, tokenizer, device, concepts, attribute_value, layer_idx):
    """收集概念的属性值hidden state"""
    template = ATTRIBUTE_SENTENCE_TEMPLATES[0]
    
    sentences = []
    targets = []
    for concept in concepts:
        sent = template.format(concept=concept, attribute=attribute_value)
        sentences.append(sent)
        targets.append(concept)
    
    hs = collect_hs_at_layer(model, tokenizer, device, sentences, targets, layer_idx)
    return hs


def compute_direction_overlap(dir1, dir2):
    """计算两个方向之间的余弦相似度 (绝对值)"""
    n1 = np.linalg.norm(dir1)
    n2 = np.linalg.norm(dir2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return abs(float(np.dot(dir1, dir2) / (n1 * n2)))


def compute_attribute_direction(hs_by_value, target_value, all_values_hs=None):
    """
    计算某个属性值的特征方向
    方法: 该属性值的质心 - 全体质心 (或去掉该值后的质心)
    """
    target_hs = hs_by_value.get(target_value)
    if target_hs is None or len(target_hs) < 2:
        return None
    
    target_centroid = np.mean(target_hs, axis=0)
    
    if all_values_hs is not None:
        # 用全体质心
        all_centroid = np.mean(all_values_hs, axis=0)
    else:
        # 用其他值的质心
        other_hs = []
        for val, hs in hs_by_value.items():
            if val != target_value:
                other_hs.append(hs)
        if len(other_hs) == 0:
            return None
        other_all = np.vstack(other_hs)
        all_centroid = np.mean(other_all, axis=0)
    
    direction = target_centroid - all_centroid
    return direction


# ============================================================================
# Exp1: 属性方向独立性测试
# ============================================================================

def exp1_attribute_directions(model_name, model, tokenizer, device, layers_to_test):
    """Exp1: 测试属性是否形成独立方向"""
    print(f"\n{'='*60}")
    print(f"Exp1: 属性方向独立性测试")
    print(f"{'='*60}")
    
    results = {"model": model_name, "exp": 1, "experiment": "attribute_directions", "layers": {}}
    
    # 只测试2个最有代表性的属性维度
    test_attributes = ["edible", "size", "color"]
    
    for layer_idx in layers_to_test:
        print(f"\n  --- Layer {layer_idx} ---")
        layer_result = {}
        
        # 1. 收集所有概念的hidden states
        all_concepts = list(CONCEPTS_WITH_ATTRIBUTES.keys())
        all_hs = collect_hs_for_concepts(model, tokenizer, device, all_concepts, layer_idx)
        
        if all_hs is None or len(all_hs) < 5:
            print(f"    收集概念数据不足, 跳过")
            continue
        
        concept_to_hs = {}
        for i, concept in enumerate(all_concepts):
            if i < len(all_hs):
                concept_to_hs[concept] = all_hs[i]
        
        # 2. 对每个属性, 计算属性值方向
        attr_directions = {}  # {attr: {value: direction}}
        
        for attr in test_attributes:
            groups = ATTRIBUTE_GROUPS[attr]
            hs_by_value = {}
            all_for_attr = []
            
            for value, concepts in groups.items():
                hs_list = []
                for c in concepts:
                    if c in concept_to_hs:
                        hs_list.append(concept_to_hs[c])
                if len(hs_list) >= 1:
                    hs_by_value[value] = np.array(hs_list)
                    all_for_attr.extend(hs_list)
            
            if len(hs_by_value) < 2:
                continue
            
            all_for_attr = np.array(all_for_attr)
            
            # 计算每个属性值的方向
            directions = {}
            for value in hs_by_value:
                d = compute_attribute_direction(hs_by_value, value, all_for_attr)
                if d is not None:
                    directions[value] = d
            
            attr_directions[attr] = directions
        
        # 3. 测试1: 同一属性内不同值的方向是否正交
        within_attr_orthogonality = {}
        for attr, directions in attr_directions.items():
            values = list(directions.keys())
            if len(values) < 2:
                continue
            cosines = []
            for i in range(len(values)):
                for j in range(i+1, len(values)):
                    cos = compute_direction_overlap(directions[values[i]], directions[values[j]])
                    cosines.append(cos)
            if cosines:
                within_attr_orthogonality[attr] = {
                    "mean_cos": float(np.mean(cosines)),
                    "max_cos": float(np.max(cosines)),
                    "min_cos": float(np.min(cosines)),
                    "n_pairs": len(cosines),
                }
        
        # 4. 测试2: 不同属性之间的方向是否正交
        cross_attr_orthogonality = {}
        attr_list = list(attr_directions.keys())
        for i in range(len(attr_list)):
            for j in range(i+1, len(attr_list)):
                attr1, attr2 = attr_list[i], attr_list[j]
                cosines = []
                for v1, d1 in attr_directions[attr1].items():
                    for v2, d2 in attr_directions[attr2].items():
                        cos = compute_direction_overlap(d1, d2)
                        cosines.append(cos)
                if cosines:
                    cross_attr_orthogonality[f"{attr1}_vs_{attr2}"] = {
                        "mean_cos": float(np.mean(cosines)),
                        "max_cos": float(np.max(cosines)),
                        "n_pairs": len(cosines),
                    }
        
        # 5. 测试3: 每个属性的分类准确率 (用方向做分类)
        attr_classification = {}
        for attr, directions in attr_directions.items():
            groups = ATTRIBUTE_GROUPS[attr]
            correct = 0
            total = 0
            
            for value, concepts in groups.items():
                if value not in directions:
                    continue
                ref_dir = directions[value]
                
                for concept in concepts:
                    if concept not in concept_to_hs:
                        continue
                    hs = concept_to_hs[concept]
                    
                    # 计算与每个属性值方向的余弦, 选最大的
                    best_val = None
                    best_cos = -2
                    for v, d in directions.items():
                        cos = compute_direction_overlap(hs, d)
                        if cos > best_cos:
                            best_cos = cos
                            best_val = v
                    
                    if best_val == value:
                        correct += 1
                    total += 1
            
            if total > 0:
                attr_classification[attr] = {
                    "accuracy": float(correct / total),
                    "correct": correct,
                    "total": total,
                }
        
        # 6. 测试4: 属性方向的维度 (PCA分析)
        attr_dimension = {}
        for attr, directions in attr_directions.items():
            if len(directions) < 2:
                continue
            dir_matrix = np.array(list(directions.values()))
            if dir_matrix.shape[0] < 2:
                continue
            pca = PCA()
            pca.fit(dir_matrix)
            # n90: 解释90%方差需要的维度数
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            n90 = int(np.searchsorted(cumvar, 0.9) + 1)
            attr_dimension[attr] = {
                "n_values": len(directions),
                "n90": n90,
                "pc1_var": float(pca.explained_variance_ratio_[0]),
                "pc2_var": float(pca.explained_variance_ratio_[1]) if len(pca.explained_variance_ratio_) > 1 else 0,
            }
        
        # 7. 测试5: 概念可分性 - 基于属性的kNN分类
        # 构建属性向量 (每个概念用其属性值的one-hot编码)
        attr_names = test_attributes
        concept_attr_vecs = {}
        for concept in all_concepts:
            attrs = CONCEPTS_WITH_ATTRIBUTES[concept]
            vec = []
            for attr in attr_names:
                if attr == "edible":
                    vec.append(1.0 if attrs[attr] == "yes" else 0.0)
                elif attr == "size":
                    vec.append({"small": 0.0, "medium": 0.5, "large": 1.0}.get(attrs[attr], 0.25))
                elif attr == "color":
                    vec.append(hash(attrs[attr]) % 100 / 100.0)  # 伪编码
            concept_attr_vecs[concept] = np.array(vec)
        
        # 在hidden state空间中做kNN, 看是否属性近邻一致
        if all_hs is not None and len(all_hs) >= 10:
            from sklearn.neighbors import KNeighborsClassifier
            
            # 用edible做分类
            y_edible = [1 if CONCEPTS_WITH_ATTRIBUTES[c]["edible"] == "yes" else 0 for c in all_concepts]
            
            if len(set(y_edible)) >= 2:
                knn = KNeighborsClassifier(n_neighbors=min(3, len(all_hs)-1))
                try:
                    knn.fit(all_hs, y_edible)
                    edible_acc = float(knn.score(all_hs, y_edible))
                except:
                    edible_acc = -1
            else:
                edible_acc = -1
            
            # 用size做分类
            y_size = [CONCEPTS_WITH_ATTRIBUTES[c]["size"] for c in all_concepts]
            if len(set(y_size)) >= 2:
                knn2 = KNeighborsClassifier(n_neighbors=min(3, len(all_hs)-1))
                try:
                    knn2.fit(all_hs, y_size)
                    size_acc = float(knn2.score(all_hs, y_size))
                except:
                    size_acc = -1
            else:
                size_acc = -1
        else:
            edible_acc = -1
            size_acc = -1
        
        # 汇总
        layer_result = {
            "n_concepts": len(concept_to_hs),
            "within_attr_orthogonality": within_attr_orthogonality,
            "cross_attr_orthogonality": cross_attr_orthogonality,
            "attr_classification": attr_classification,
            "attr_dimension": attr_dimension,
            "edible_knn_acc": edible_acc,
            "size_knn_acc": size_acc,
        }
        
        results["layers"][str(layer_idx)] = layer_result
        
        # 打印摘要
        print(f"    概念数: {len(concept_to_hs)}")
        for attr, orth in within_attr_orthogonality.items():
            print(f"    {attr} 内部正交性: mean_cos={orth['mean_cos']:.4f}, max_cos={orth['max_cos']:.4f}")
        for key, orth in cross_attr_orthogonality.items():
            print(f"    {key} 交叉正交性: mean_cos={orth['mean_cos']:.4f}")
        for attr, cls in attr_classification.items():
            print(f"    {attr} 方向分类: acc={cls['accuracy']:.4f}")
        print(f"    edible kNN: {edible_acc:.4f}, size kNN: {size_acc:.4f}")
    
    return results


# ============================================================================
# Exp2: is-a关系的精确验证
# ============================================================================

def exp2_isa_relationships(model_name, model, tokenizer, device, layers_to_test):
    """Exp2: 验证is-a关系的几何结构"""
    print(f"\n{'='*60}")
    print(f"Exp2: is-a关系的精确验证")
    print(f"{'='*60}")
    
    results = {"model": model_name, "exp": 2, "experiment": "isa_relationships", "layers": {}}
    
    for layer_idx in layers_to_test:
        print(f"\n  --- Layer {layer_idx} ---")
        layer_result = {}
        
        # 收集is-a关系中所有词的hidden states
        all_words = set()
        for pair_name, pair in ISA_HIERARCHY.items():
            all_words.add(pair["child"])
            all_words.add(pair["parent"])
        all_words = sorted(all_words)
        
        # 用简单句式收集
        word_hs = {}
        for word in all_words:
            hs = collect_hs_for_concepts(model, tokenizer, device, [word], layer_idx)
            if hs is not None and len(hs) > 0:
                word_hs[word] = hs[0]
        
        if len(word_hs) < 5:
            print(f"    数据不足, 跳过")
            continue
        
        # 1. 计算is-a偏移向量
        isa_offsets = {}
        for pair_name, pair in ISA_HIERARCHY.items():
            child = pair["child"]
            parent = pair["parent"]
            if child in word_hs and parent in word_hs:
                offset = word_hs[parent] - word_hs[child]
                isa_offsets[pair_name] = offset
        
        # 2. 同类is-a偏移的平行度 (核心测试!)
        # 水果→fruit的偏移是否平行?
        fruit_offsets = {k: v for k, v in isa_offsets.items() if "→fruit" in k}
        animal_offsets = {k: v for k, v in isa_offsets.items() if "→animal" in k}
        tool_offsets = {k: v for k, v in isa_offsets.items() if "→tool" in k}
        
        def compute_pairwise_cos(offsets_dict, label):
            if len(offsets_dict) < 2:
                return None
            keys = list(offsets_dict.keys())
            cosines = []
            for i in range(len(keys)):
                for j in range(i+1, len(keys)):
                    cos = compute_direction_overlap(offsets_dict[keys[i]], offsets_dict[keys[j]])
                    cosines.append((keys[i], keys[j], cos))
            return cosines
        
        fruit_parallels = compute_pairwise_cos(fruit_offsets, "fruit")
        animal_parallels = compute_pairwise_cos(animal_offsets, "animal")
        tool_parallels = compute_pairwise_cos(tool_offsets, "tool")
        
        # 3. 不同类别is-a偏移的平行度 (对比基线)
        cross_category_cosines = []
        for f_name, f_off in fruit_offsets.items():
            for a_name, a_off in animal_offsets.items():
                cos = compute_direction_overlap(f_off, a_off)
                cross_category_cosines.append(cos)
            for t_name, t_off in tool_offsets.items():
                cos = compute_direction_overlap(f_off, t_off)
                cross_category_cosines.append(cos)
        
        # 4. 高层is-a偏移 (fruit→food, animal→organism, tool→object)
        higher_offsets = {k: v for k, v in isa_offsets.items() 
                        if any(x in k for x in ["fruit→food", "animal→organism", "tool→object", "food→object"])}
        
        # 5. 偏移的稳定性: 同一概念的偏移方向是否在不同句子模板中一致
        # (简单版本: 只用一种模板, 但计算各偏移的范数和归一化后的方向)
        offset_norms = {}
        for name, off in isa_offsets.items():
            offset_norms[name] = float(np.linalg.norm(off))
        
        # 6. 偏移能否用于预测is-a关系: 给定child的hs + 偏移方向, 是否接近parent?
        # 用同类偏移的平均方向做预测
        predict_results = {}
        for category, offsets in [("fruit", fruit_offsets), ("animal", animal_offsets), ("tool", tool_offsets)]:
            if len(offsets) < 2:
                continue
            # 计算平均偏移方向
            mean_offset = np.mean(list(offsets.values()), axis=0)
            mean_dir = mean_offset / max(np.linalg.norm(mean_offset), 1e-10)
            
            # 用leave-one-out方式测试
            correct = 0
            total = 0
            for pair_name, pair in ISA_HIERARCHY.items():
                if category not in pair_name and f"→{category}" not in pair_name:
                    continue
                child = pair["child"]
                parent = pair["parent"]
                if child not in word_hs or parent not in word_hs:
                    continue
                
                # 预测: child + 平均偏移方向 * child_to_parent的距离
                child_hs = word_hs[child]
                parent_hs = word_hs[parent]
                dist = np.linalg.norm(parent_hs - child_hs)
                predicted = child_hs + mean_dir * dist
                
                # 检查predicted是否比child更接近parent
                cos_pred_parent = compute_direction_overlap(predicted, parent_hs)
                cos_child_parent = compute_direction_overlap(child_hs, parent_hs)
                
                if cos_pred_parent > cos_child_parent:
                    correct += 1
                total += 1
            
            if total > 0:
                predict_results[category] = {
                    "accuracy": float(correct / total),
                    "correct": correct,
                    "total": total,
                }
        
        # 汇总
        same_category_mean_cos = {}
        for label, parallels in [("fruit", fruit_parallels), ("animal", animal_parallels), ("tool", tool_parallels)]:
            if parallels:
                cos_vals = [c for _, _, c in parallels]
                same_category_mean_cos[label] = {
                    "mean_cos": float(np.mean(cos_vals)),
                    "max_cos": float(np.max(cos_vals)),
                    "min_cos": float(np.min(cos_vals)),
                    "n_pairs": len(cos_vals),
                }
        
        cross_category_mean_cos = float(np.mean(cross_category_cosines)) if cross_category_cosines else -1
        
        layer_result = {
            "n_words": len(word_hs),
            "n_offsets": len(isa_offsets),
            "same_category_parallelism": same_category_mean_cos,
            "cross_category_cosine": cross_category_mean_cos,
            "predict_with_mean_direction": predict_results,
            "offset_norms_sample": {k: v for k, v in list(offset_norms.items())[:5]},
        }
        
        results["layers"][str(layer_idx)] = layer_result
        
        # 打印摘要
        print(f"    词数: {len(word_hs)}, 偏移数: {len(isa_offsets)}")
        for label, data in same_category_mean_cos.items():
            print(f"    {label}→parent 平行度: mean_cos={data['mean_cos']:.4f}")
        print(f"    跨类别偏移余弦: {cross_category_mean_cos:.4f}")
        for cat, pred in predict_results.items():
            print(f"    {cat} 偏移预测: acc={pred['accuracy']:.4f}")
    
    return results


# ============================================================================
# Exp3: 推理链的因果追踪
# ============================================================================

def exp3_reasoning_chains(model_name, model, tokenizer, device, layers_to_test):
    """Exp3: 追踪推理链在hidden state中的体现"""
    print(f"\n{'='*60}")
    print(f"Exp3: 推理链的因果追踪")
    print(f"{'='*60}")
    
    results = {"model": model_name, "exp": 3, "experiment": "reasoning_chains", "layers": {}}
    
    for layer_idx in layers_to_test:
        print(f"\n  --- Layer {layer_idx} ---")
        layer_result = {}
        
        # 收集各种推理句子中的关键概念hidden state
        # 关键设计: 比较同一概念在不同上下文中的hidden state变化
        
        reasoning_data = {}
        
        for chain in REASONING_CHAINS:
            chain_type = chain["type"]
            
            # 提取主语概念 (简化: 取第一个词)
            premise1 = chain["premise1"]
            conclusion = chain["conclusion"]
            
            # 获取主语词
            subject = premise1.split()[0].lower()
            
            # 三种上下文:
            # A: 单独概念 "The apple"
            # B: 含前提 "Apple is a fruit"
            # C: 含推理 "Apple is a fruit, and fruit can be eaten, so apple can be eaten"
            
            context_A = f"The {subject} is here"
            context_B = premise1 + "."
            context_C = f"{chain['premise1']}. {chain['premise2']}. Therefore, {chain['conclusion']}."
            
            # 也收集结论中的关键属性词的hidden state
            # 例如 "eaten", "work", "warm"
            conclusion_words = conclusion.lower().split()
            attr_word = conclusion_words[-1] if conclusion_words else ""
            
            hs_A = collect_hs_at_layer(model, tokenizer, device, [context_A], [subject], layer_idx)
            hs_B = collect_hs_at_layer(model, tokenizer, device, [context_B], [subject], layer_idx)
            hs_C = collect_hs_at_layer(model, tokenizer, device, [context_C], [subject], layer_idx)
            
            # 也收集属性词在推理上下文中的hidden state
            hs_attr = collect_hs_at_layer(model, tokenizer, device, [context_C], [attr_word], layer_idx)
            
            reasoning_data[chain_type] = {
                "subject": subject,
                "hs_A": hs_A[0] if hs_A is not None and len(hs_A) > 0 else None,
                "hs_B": hs_B[0] if hs_B is not None and len(hs_B) > 0 else None,
                "hs_C": hs_C[0] if hs_C is not None and len(hs_C) > 0 else None,
                "hs_attr": hs_attr[0] if hs_attr is not None and len(hs_attr) > 0 else None,
            }
        
        # 分析1: 推理上下文对概念表示的影响
        # A→B: 加前提后概念表示变化了多少?
        # A→C: 加推理后概念表示变化了多少?
        # B→C: 加结论后概念表示变化了多少?
        
        context_effects = {}
        for chain_type, data in reasoning_data.items():
            if data["hs_A"] is None or data["hs_B"] is None or data["hs_C"] is None:
                continue
            
            A, B, C = data["hs_A"], data["hs_B"], data["hs_C"]
            
            cos_AB = compute_direction_overlap(B - A, A)  # 加前提的影响方向 vs 原始方向
            cos_AC = compute_direction_overlap(C - A, A)  # 加推理的影响方向 vs 原始方向
            
            # 更重要: B→C的变化是否指向属性方向?
            delta_BC = C - B
            delta_AB = B - A
            cos_BC_AB = compute_direction_overlap(delta_BC, delta_AB)  # 推理变化 vs 前提变化
            
            # 如果属性词存在, 检查推理变化是否指向属性词方向
            if data["hs_attr"] is not None:
                attr_dir = data["hs_attr"] - A  # 属性词相对概念的偏移
                cos_delta_attr = compute_direction_overlap(delta_BC, attr_dir)
            else:
                cos_delta_attr = -2
            
            context_effects[chain_type] = {
                "subject": data["subject"],
                "delta_AB_norm": float(np.linalg.norm(B - A)),
                "delta_AC_norm": float(np.linalg.norm(C - A)),
                "delta_BC_norm": float(np.linalg.norm(delta_BC)),
                "cos_BC_vs_AB": float(cos_BC_AB),
                "cos_delta_BC_vs_attr": float(cos_delta_attr) if cos_delta_attr > -2 else None,
            }
        
        # 分析2: 可食性推理 vs 不可食推理 的差异
        edible_types = [k for k in context_effects.keys() if "edible" in k and "not" not in k]
        not_edible_types = [k for k in context_effects.keys() if "not_edible" in k]
        
        edible_deltas = [context_effects[k]["delta_BC_norm"] for k in edible_types if k in context_effects]
        not_edible_deltas = [context_effects[k]["delta_BC_norm"] for k in not_edible_types if k in context_effects]
        
        layer_result = {
            "n_chains": len(reasoning_data),
            "context_effects": context_effects,
            "edible_vs_not_edible": {
                "edible_mean_delta_BC": float(np.mean(edible_deltas)) if edible_deltas else None,
                "not_edible_mean_delta_BC": float(np.mean(not_edible_deltas)) if not_edible_deltas else None,
            }
        }
        
        results["layers"][str(layer_idx)] = layer_result
        
        # 打印摘要
        print(f"    推理链数: {len(reasoning_data)}")
        for chain_type, eff in context_effects.items():
            print(f"    {chain_type}: delta_BC={eff['delta_BC_norm']:.4f}, cos(BC,AB)={eff['cos_BC_vs_AB']:.4f}")
        if edible_deltas:
            print(f"    可食推理 delta_BC: {np.mean(edible_deltas):.4f}")
        if not_edible_deltas:
            print(f"    不可食推理 delta_BC: {np.mean(not_edible_deltas):.4f}")
    
    return results


# ============================================================================
# Exp4: 属性干预实验
# ============================================================================

def exp4_attribute_intervention(model_name, model, tokenizer, device, layers_to_test):
    """Exp4: 通过干预测试属性的因果性"""
    print(f"\n{'='*60}")
    print(f"Exp4: 属性干预实验")
    print(f"{'='*60}")
    
    results = {"model": model_name, "exp": 4, "experiment": "attribute_intervention", "layers": {}}
    
    # 获取W_U用于计算logit变化
    W_U = get_W_U(model)  # [vocab_size, d_model]
    
    for layer_idx in layers_to_test:
        print(f"\n  --- Layer {layer_idx} ---")
        layer_result = {}
        
        # 步骤1: 收集可食/不可食两组概念的hidden states
        edible_concepts = [c for c, a in CONCEPTS_WITH_ATTRIBUTES.items() if a["edible"] == "yes"]
        not_edible_concepts = [c for c, a in CONCEPTS_WITH_ATTRIBUTES.items() if a["edible"] == "no"]
        
        # 取子集 (避免太多)
        edible_subset = edible_concepts[:5]
        not_edible_subset = not_edible_concepts[:5]
        
        edible_hs = collect_hs_for_concepts(model, tokenizer, device, edible_subset, layer_idx)
        not_edible_hs = collect_hs_for_concepts(model, tokenizer, device, not_edible_subset, layer_idx)
        
        if edible_hs is None or not_edible_hs is None:
            print(f"    数据不足, 跳过")
            continue
        
        # 步骤2: 计算可食性方向
        edible_centroid = np.mean(edible_hs, axis=0)
        not_edible_centroid = np.mean(not_edible_hs, axis=0)
        edible_direction = edible_centroid - not_edible_centroid
        edible_dir_norm = np.linalg.norm(edible_direction)
        if edible_dir_norm > 1e-10:
            edible_direction = edible_direction / edible_dir_norm
        
        # 步骤3: 对不可食概念, 沿可食性方向干预
        # 检查: 干预后是否更"可食" (在W_U空间中更接近可食词)
        
        # 定义目标词
        edible_words = ["eat", "food", "delicious", "tasty", "fruit"]
        not_edible_words = ["tool", "use", "build", "hard", "stone"]
        
        # 获取目标词的token id
        edible_token_ids = []
        for w in edible_words:
            ids = tokenizer.encode(w, add_special_tokens=False)
            if ids:
                edible_token_ids.extend(ids)
        
        not_edible_token_ids = []
        for w in not_edible_words:
            ids = tokenizer.encode(w, add_special_tokens=False)
            if ids:
                not_edible_token_ids.extend(ids)
        
        # 对每个不可食概念进行干预
        intervention_results = []
        for i, concept in enumerate(not_edible_subset):
            if i >= len(not_edible_hs):
                break
            
            hs_original = not_edible_hs[i]
            
            # 计算原始logit
            logits_original = W_U @ hs_original
            edible_score_original = float(np.mean(logits_original[edible_token_ids])) if edible_token_ids else 0
            not_edible_score_original = float(np.mean(logits_original[not_edible_token_ids])) if not_edible_token_ids else 0
            
            # 干预: 沿可食性方向平移
            for beta in [0.5, 1.0, 2.0, 4.0]:
                hs_intervened = hs_original + beta * edible_direction * edible_dir_norm
                
                logits_intervened = W_U @ hs_intervened
                edible_score_intervened = float(np.mean(logits_intervened[edible_token_ids])) if edible_token_ids else 0
                not_edible_score_intervened = float(np.mean(logits_intervened[not_edible_token_ids])) if not_edible_token_ids else 0
                
                intervention_results.append({
                    "concept": concept,
                    "beta": beta,
                    "edible_score_original": edible_score_original,
                    "edible_score_intervened": edible_score_intervened,
                    "edible_score_delta": edible_score_intervened - edible_score_original,
                    "not_edible_score_original": not_edible_score_original,
                    "not_edible_score_intervened": not_edible_score_intervened,
                    "not_edible_score_delta": not_edible_score_intervened - not_edible_score_original,
                })
        
        # 也对可食概念做反向干预 (向不可食方向平移)
        reverse_intervention_results = []
        for i, concept in enumerate(edible_subset):
            if i >= len(edible_hs):
                break
            
            hs_original = edible_hs[i]
            
            logits_original = W_U @ hs_original
            edible_score_original = float(np.mean(logits_original[edible_token_ids])) if edible_token_ids else 0
            
            for beta in [0.5, 1.0, 2.0, 4.0]:
                hs_intervened = hs_original - beta * edible_direction * edible_dir_norm
                
                logits_intervened = W_U @ hs_intervened
                edible_score_intervened = float(np.mean(logits_intervened[edible_token_ids])) if edible_token_ids else 0
                
                reverse_intervention_results.append({
                    "concept": concept,
                    "beta": beta,
                    "edible_score_original": edible_score_original,
                    "edible_score_intervened": edible_score_intervened,
                    "edible_score_delta": edible_score_intervened - edible_score_original,
                })
        
        # 步骤4: 检查干预是否在logit空间产生正确的方向性变化
        edible_deltas = [r["edible_score_delta"] for r in intervention_results]
        not_edible_deltas = [r["not_edible_score_delta"] for r in intervention_results]
        
        # 按beta分组汇总
        by_beta = defaultdict(list)
        for r in intervention_results:
            by_beta[r["beta"]].append(r)
        
        beta_summary = {}
        for beta, res_list in sorted(by_beta.items()):
            beta_summary[str(beta)] = {
                "mean_edible_delta": float(np.mean([r["edible_score_delta"] for r in res_list])),
                "mean_not_edible_delta": float(np.mean([r["not_edible_score_delta"] for r in res_list])),
                "n_concepts": len(res_list),
            }
        
        layer_result = {
            "edible_direction_norm": float(edible_dir_norm),
            "intervention_on_not_edible": intervention_results[:10],  # 保存前10个
            "reverse_intervention_on_edible": reverse_intervention_results[:10],
            "beta_summary": beta_summary,
            "edible_words_used": edible_words,
            "not_edible_words_used": not_edible_words,
        }
        
        results["layers"][str(layer_idx)] = layer_result
        
        # 打印摘要
        print(f"    可食性方向范数: {edible_dir_norm:.4f}")
        for beta, summary in sorted(beta_summary.items(), key=lambda x: float(x[0])):
            print(f"    beta={beta}: edible_delta={summary['mean_edible_delta']:.4f}, "
                  f"not_edible_delta={summary['mean_not_edible_delta']:.4f}")
    
    return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CCMF Phase26: 属性编码机制")
    parser.add_argument("--model", type=str, default="qwen3",
                       choices=["qwen3", "glm4", "deepseek7b"],
                       help="模型名称")
    parser.add_argument("--exp", type=int, default=0,
                       help="实验编号 (0=全部, 1-4)")
    args = parser.parse_args()
    
    model_name = args.model
    exp_num = args.exp
    
    print(f"\n{'='*60}")
    print(f"CCMF Phase 26: 语义属性编码机制破解")
    print(f"模型: {model_name}, 实验: {exp_num if exp_num > 0 else '全部'}")
    print(f"{'='*60}")
    
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
    else:
        model, tokenizer, device = load_model(model_name)
    
    model_info = get_model_info(model, model_name)
    print(f"模型信息: {model_info.model_class}, {model_info.n_layers}层, d_model={model_info.d_model}")
    
    # 选择测试层
    n_layers = model_info.n_layers
    # 采样: 首层 + 均匀采样 + 断裂层附近 + 末层
    sample_layers = set()
    sample_layers.add(0)
    sample_layers.add(n_layers - 1)
    step = max(n_layers // 8, 1)
    for i in range(step, n_layers, step):
        sample_layers.add(i)
    # 加入断裂层附近 (根据之前的发现)
    fracture_layers = {"qwen3": 6, "glm4": 2, "deepseek7b": 7}
    fl = fracture_layers.get(model_name, n_layers // 3)
    for delta in [-1, 0, 1, 2, 3]:
        if 0 <= fl + delta < n_layers:
            sample_layers.add(fl + delta)
    layers_to_test = sorted(sample_layers)
    print(f"测试层: {layers_to_test}")
    
    # 运行实验
    all_results = []
    
    try:
        if exp_num in [0, 1]:
            r1 = exp1_attribute_directions(model_name, model, tokenizer, device, layers_to_test)
            all_results.append(r1)
        
        if exp_num in [0, 2]:
            r2 = exp2_isa_relationships(model_name, model, tokenizer, device, layers_to_test)
            all_results.append(r2)
        
        if exp_num in [0, 3]:
            r3 = exp3_reasoning_chains(model_name, model, tokenizer, device, layers_to_test)
            all_results.append(r3)
        
        if exp_num in [0, 4]:
            r4 = exp4_attribute_intervention(model_name, model, tokenizer, device, layers_to_test)
            all_results.append(r4)
    
    finally:
        release_model(model)
    
    # 保存结果
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
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "glm5_temp")
    os.makedirs(output_dir, exist_ok=True)
    
    for result in all_results:
        exp_id = result["exp"]
        output_file = os.path.join(output_dir, f"ccmf_exp{exp_id}_{model_name}_results.json")
        result = convert_numpy(result)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {output_file}")
    
    print(f"\n{'='*60}")
    print(f"Phase 26 完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
