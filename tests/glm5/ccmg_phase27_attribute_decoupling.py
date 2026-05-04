"""
CCMG(Phase 27): 属性是否真实存在——条件解耦与非线性检测
=============================================================================
Phase 26核心发现(经批判修正):
  ★★★ 属性方向是"混合语义方向", 不是纯属性方向!
    → d(color_red) = mean(apple,rose,blood...) 混杂了edible/material等
    → 无法从混合方向推断属性是否正交
  
  ★★★ edible不是线性方向, 是非线性结构!
    → 线性分类不稳定(10-95%), 但kNN高达85-95%
    → edible = 非线性流形 / gating结构, 不是1D方向
  
  ★★★ 推理实验无效(位置未控制, 句长变化, 上下文污染)
  
  ★★★ 因果性结论过强: 方向可影响输出 ≠ 方向是因果变量

核心未解问题:
  ??? 是否存在"纯属性方向"(一个方向="红", 一个方向="可食")?
  ??? 属性是否独立编码(正交? 耦合? 非线性)?
  ??? 属性是线性还是非线性结构?

Phase 27实验(精炼版):
  27A: ★★★★★★★★★★★ 条件解耦(最重要!)
    → 构造: 红+可食 vs 红+不可食, 绿+可食 vs 绿+不可食
    → 控制一个属性, 变化另一个属性
    → 如果edible方向在控制color后仍然一致 → 属性独立
    → 如果不一致 → 属性耦合

  27B: ★★★★★★★★★ 非线性检测
    → 线性(logistic) vs 非线性(MLP/kernel SVM)分类器
    → 如果非线性显著优于线性 → 属性边界非线性
    → 如果差异小 → 属性可能是线性结构

  27C: ★★★★★★★ 局部线性结构
    → h(apple) - h(fruit) vs h(orange) - h(fruit)
    → 在"水果"这个局部邻域内, 偏移是否一致?
    → 如果一致 → 局部线性; 不一致 → 非线性

  27D: ★★★★★★★★ 最小因果子空间
    → 找到最小维度子空间 → 干预仍能影响输出
    → 如果1维就够 → 属性是线性方向
    → 如果需要高维 → 属性是非线性结构
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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from collections import defaultdict

from model_utils import (load_model, get_layers, get_model_info, release_model,
                         safe_decode, MODEL_CONFIGS, get_W_U)

# 修正版compute_cos: 对两个向量都归一化
def compute_cos(v1, v2):
    """计算两个向量的余弦相似度 (双归一化)"""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))

# ============================================================================
# 数据定义: 条件解耦的精心设计
# ============================================================================

# 27A: 条件解耦 — 控制一个属性, 变化另一个
# 关键设计: 用虚拟/稀有概念避免语义预存偏差
CONDITIONAL_PAIRS = {
    # 控制颜色, 变化可食性
    "color_controlled_edible": {
        "description": "控制颜色=红, 变化edible",
        "red_edible":    ["apple", "strawberry", "tomato", "cherry"],
        "red_inedible":  ["hammer", "brick", "fire_engine", "stop_sign"],
        "controlled_attr": "color",
        "varied_attr": "edible",
    },
    "color_controlled_edible_v2": {
        "description": "控制颜色=绿, 变化edible",
        "green_edible":   ["grape", "kiwi", "pear", "lime"],
        "green_inedible": ["grass", "tree", "frog", "emerald"],
        "controlled_attr": "color",
        "varied_attr": "edible",
    },
    # 控制可食性, 变化颜色
    "edible_controlled_color": {
        "description": "控制edible=yes, 变化color",
        "edible_red":    ["apple", "strawberry", "cherry", "tomato"],
        "edible_yellow": ["banana", "lemon", "mango", "corn"],
        "controlled_attr": "edible",
        "varied_attr": "color",
    },
    # 控制可食性, 变化大小
    "edible_controlled_size": {
        "description": "控制edible=yes, 变化size",
        "edible_small":  ["grape", "strawberry", "cherry", "blueberry"],
        "edible_medium": ["apple", "orange", "banana", "mango"],
        "edible_large":  ["watermelon", "pineapple", "pumpkin", "coconut"],
        "controlled_attr": "edible",
        "varied_attr": "size",
    },
    # 控制大小, 变化可食性
    "size_controlled_edible": {
        "description": "控制size=medium, 变化edible",
        "medium_edible":   ["apple", "orange", "banana", "tomato"],
        "medium_inedible": ["hammer", "shoe", "book", "ball"],
        "controlled_attr": "size",
        "varied_attr": "edible",
    },
}

# 27B: 非线性检测 — 用大量概念做分类
NONLINEAR_TEST_CONCEPTS = {
    "edible": {
        "yes": ["apple", "orange", "banana", "strawberry", "grape", "salmon",
                "rice", "bread", "cheese", "egg", "carrot", "potato", "mushroom",
                "corn", "pepper", "onion", "garlic", "bean", "pea", "nut"],
        "no": ["dog", "cat", "elephant", "eagle", "hammer", "knife", "chair",
               "shirt", "car", "rock", "tree", "flower", "cloud", "book",
               "shoe", "ball", "cup", "pen", "table", "door"],
    },
    "animacy": {
        "animate": ["dog", "cat", "elephant", "eagle", "salmon", "person",
                    "bird", "fish", "snake", "frog", "bee", "ant",
                    "horse", "cow", "pig", "sheep", "monkey", "bear",
                    "lion", "tiger"],
        "inanimate": ["apple", "hammer", "knife", "chair", "shirt", "car",
                      "rock", "tree", "flower", "cloud", "book", "shoe",
                      "ball", "cup", "pen", "table", "door", "wall",
                      "window", "bridge"],
    },
    "size": {
        "small": ["ant", "bee", "grape", "cherry", "pin", "coin", "key",
                  "ring", "seed", "pebble"],
        "medium": ["apple", "cat", "dog", "book", "shoe", "cup", "hammer",
                   "ball", "flower", "bird"],
        "large": ["elephant", "car", "house", "mountain", "airplane", "ship",
                  "building", "tree", "whale", "bridge"],
    },
}

# 27C: 局部线性 — 同类概念的偏移一致性
LOCAL_LINEARITY_GROUPS = {
    "fruit_to_fruit": {
        "children": ["apple", "orange", "banana", "strawberry", "grape",
                     "mango", "pear", "peach", "cherry", "lemon"],
        "parent": "fruit",
    },
    "animal_to_animal": {
        "children": ["dog", "cat", "elephant", "eagle", "salmon",
                     "horse", "cow", "pig", "bird", "fish"],
        "parent": "animal",
    },
    "tool_to_tool": {
        "children": ["hammer", "knife", "saw", "drill", "wrench",
                     "screwdriver", "axe", "shovel", "pliers", "chisel"],
        "parent": "tool",
    },
    # 更精细的子类
    "citrus_to_fruit": {
        "children": ["orange", "lemon", "lime", "grapefruit"],
        "parent": "fruit",
    },
    "berry_to_fruit": {
        "children": ["strawberry", "blueberry", "raspberry", "blackberry"],
        "parent": "fruit",
    },
}

# 句子模板 — 所有实验统一用一种模板, 控制位置
TEMPLATE = "The {word} is here"


# ============================================================================
# 工具函数
# ============================================================================

def find_token_index(tokens, target_word):
    target_lower = target_word.lower().strip()
    for i, t in enumerate(tokens):
        if t.lower().strip() == target_lower:
            return i
    for i, t in enumerate(tokens):
        if t.lower().strip()[:3] == target_lower[:3]:
            return i
    for i, t in enumerate(tokens):
        if t.lower().strip()[:2] == target_lower[:2]:
            return i
    for i, t in enumerate(tokens):
        if len(t.strip()) > 0 and t.lower().strip()[0] == target_lower[0]:
            return i
    return -1


def collect_hs_for_words(model, tokenizer, device, words, layer_idx):
    """收集一组词在指定层的hidden states, 统一用TEMPLATE, 控制位置"""
    sentences = [TEMPLATE.format(word=w) for w in words]
    
    layers = get_layers(model)
    if layer_idx >= len(layers):
        return None, None
    
    target_layer = layers[layer_idx]
    all_hs = []
    valid_words = []
    
    for sent, word in zip(sentences, words):
        toks = tokenizer(sent, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
        
        dep_idx = find_token_index(tokens_list, word)
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
            valid_words.append(word)
    
    if len(all_hs) == 0:
        return None, None
    return np.array(all_hs), valid_words


# ============================================================================
# 27A: 条件解耦 — 控制一个属性, 变化另一个
# ============================================================================

def expA_conditional_decoupling(model_name, model, tokenizer, device, layers_to_test):
    """27A: 条件解耦验证属性独立性"""
    print(f"\n{'='*60}")
    print(f"ExpA: 条件解耦 — 属性是否独立?")
    print(f"{'='*60}")
    
    results = {"model": model_name, "exp": "A", "experiment": "conditional_decoupling", "layers": {}}
    
    for layer_idx in layers_to_test:
        print(f"\n  --- Layer {layer_idx} ---")
        layer_result = {}
        
        for pair_name, pair_data in CONDITIONAL_PAIRS.items():
            # 收集所有组的hidden states
            group_hs = {}
            for key, words in pair_data.items():
                if key in ["description", "controlled_attr", "varied_attr"]:
                    continue
                hs, valid = collect_hs_for_words(model, tokenizer, device, words, layer_idx)
                if hs is not None and len(hs) >= 2:
                    group_hs[key] = {"hs": hs, "words": valid, "centroid": np.mean(hs, axis=0)}
            
            if len(group_hs) < 2:
                continue
            
            controlled_attr = pair_data["controlled_attr"]
            varied_attr = pair_data["varied_attr"]
            
            # 核心测试: 在控制属性不变时, 变化属性的方向是否一致?
            # 例: 控制=颜色(red), 变化=edible
            #   red_edible的质心 vs red_inedible的质心 → edible方向(在红色条件下)
            #   green_edible的质心 vs green_inedible的质心 → edible方向(在绿色条件下)
            #   这两个edible方向是否平行?
            
            # 找到控制条件不同的组对
            varied_directions = {}  # {控制条件: 变化属性方向}
            
            if pair_name == "color_controlled_edible":
                # 控制color=red, 变化edible
                if "red_edible" in group_hs and "red_inedible" in group_hs:
                    d = group_hs["red_edible"]["centroid"] - group_hs["red_inedible"]["centroid"]
                    varied_directions["red"] = d
            elif pair_name == "color_controlled_edible_v2":
                if "green_edible" in group_hs and "green_inedible" in group_hs:
                    d = group_hs["green_edible"]["centroid"] - group_hs["green_inedible"]["centroid"]
                    varied_directions["green"] = d
            elif pair_name == "edible_controlled_color":
                if "edible_red" in group_hs and "edible_yellow" in group_hs:
                    d = group_hs["edible_red"]["centroid"] - group_hs["edible_yellow"]["centroid"]
                    varied_directions["edible_red_vs_yellow"] = d
            elif pair_name == "edible_controlled_size":
                if "edible_small" in group_hs and "edible_medium" in group_hs:
                    d = group_hs["edible_small"]["centroid"] - group_hs["edible_medium"]["centroid"]
                    varied_directions["small_vs_medium"] = d
                if "edible_medium" in group_hs and "edible_large" in group_hs:
                    d = group_hs["edible_medium"]["centroid"] - group_hs["edible_large"]["centroid"]
                    varied_directions["medium_vs_large"] = d
                if "edible_small" in group_hs and "edible_large" in group_hs:
                    d = group_hs["edible_small"]["centroid"] - group_hs["edible_large"]["centroid"]
                    varied_directions["small_vs_large"] = d
            elif pair_name == "size_controlled_edible":
                if "medium_edible" in group_hs and "medium_inedible" in group_hs:
                    d = group_hs["medium_edible"]["centroid"] - group_hs["medium_inedible"]["centroid"]
                    varied_directions["medium"] = d
            
            # 计算: 同一变化属性在不同控制条件下的方向一致性
            dir_keys = list(varied_directions.keys())
            consistency = {}
            if len(dir_keys) >= 2:
                for i in range(len(dir_keys)):
                    for j in range(i+1, len(dir_keys)):
                        k1, k2 = dir_keys[i], dir_keys[j]
                        d1, d2 = varied_directions[k1], varied_directions[k2]
                        cos = compute_cos(d1, d2)
                        consistency[f"{k1}_vs_{k2}"] = {
                            "cosine": float(cos),
                            "interpretation": "parallel→independent" if abs(cos) > 0.7 else ("orthogonal→coupled" if abs(cos) < 0.3 else "intermediate→partial_coupling"),
                        }
            
            # 特殊对比: red条件下的edible方向 vs green条件下的edible方向
            # 这个对比需要两个pair的结果合并, 在后面处理
            
            pair_result = {
                "description": pair_data["description"],
                "controlled_attr": controlled_attr,
                "varied_attr": varied_attr,
                "n_groups": len(group_hs),
                "group_sizes": {k: len(v["hs"]) for k, v in group_hs.items()},
                "varied_directions_norms": {k: float(np.linalg.norm(v)) for k, v in varied_directions.items()},
                "direction_consistency": consistency,
            }
            
            layer_result[pair_name] = pair_result
        
        # 跨pair对比: red条件下edible方向 vs green条件下edible方向
        # 这是最关键的测试!
        if "color_controlled_edible" in layer_result and "color_controlled_edible_v2" in layer_result:
            # 需要重新收集数据做直接对比
            red_edible_words = CONDITIONAL_PAIRS["color_controlled_edible"]["red_edible"]
            red_inedible_words = CONDITIONAL_PAIRS["color_controlled_edible"]["red_inedible"]
            green_edible_words = CONDITIONAL_PAIRS["color_controlled_edible_v2"]["green_edible"]
            green_inedible_words = CONDITIONAL_PAIRS["color_controlled_edible_v2"]["green_inedible"]
            
            hs_red_e, _ = collect_hs_for_words(model, tokenizer, device, red_edible_words, layer_idx)
            hs_red_ne, _ = collect_hs_for_words(model, tokenizer, device, red_inedible_words, layer_idx)
            hs_green_e, _ = collect_hs_for_words(model, tokenizer, device, green_edible_words, layer_idx)
            hs_green_ne, _ = collect_hs_for_words(model, tokenizer, device, green_inedible_words, layer_idx)
            
            if all(x is not None and len(x) >= 2 for x in [hs_red_e, hs_red_ne, hs_green_e, hs_green_ne]):
                edible_dir_red = np.mean(hs_red_e, axis=0) - np.mean(hs_red_ne, axis=0)
                edible_dir_green = np.mean(hs_green_e, axis=0) - np.mean(hs_green_ne, axis=0)
                
                cross_condition_cos = compute_cos(edible_dir_red, edible_dir_green)
                
                # 也计算颜色方向: 红-绿 (在edible条件下)
                color_dir_edible = np.mean(hs_red_e, axis=0) - np.mean(hs_green_e, axis=0)
                # 颜色方向: 红-绿 (在inedible条件下)
                color_dir_inedible = np.mean(hs_red_ne, axis=0) - np.mean(hs_green_ne, axis=0)
                
                cross_condition_color_cos = compute_cos(color_dir_edible, color_dir_inedible)
                
                # 关键测试: edible方向与color方向是否正交?
                edible_color_cos = compute_cos(edible_dir_red, color_dir_edible)
                
                layer_result["CRITICAL_cross_condition_test"] = {
                    "description": "红条件下edible方向 vs 绿条件下edible方向",
                    "edible_dir_cross_condition_cos": float(cross_condition_cos),
                    "color_dir_cross_condition_cos": float(cross_condition_color_cos),
                    "edible_vs_color_orthogonality": float(edible_color_cos),
                    "interpretation": {
                        "edible_independent": f"cos={cross_condition_cos:.3f}, {'独立' if abs(cross_condition_cos) > 0.6 else '耦合' if abs(cross_condition_cos) < 0.3 else '部分耦合'}",
                        "color_independent": f"cos={cross_condition_color_cos:.3f}, {'独立' if abs(cross_condition_color_cos) > 0.6 else '耦合' if abs(cross_condition_color_cos) < 0.3 else '部分耦合'}",
                        "edible_color_orthogonal": f"cos={edible_color_cos:.3f}, {'正交' if abs(edible_color_cos) < 0.3 else '非正交'}",
                    }
                }
        
        results["layers"][str(layer_idx)] = layer_result
        
        # 打印摘要
        print(f"    对比数: {len(layer_result)}")
        if "CRITICAL_cross_condition_test" in layer_result:
            crit = layer_result["CRITICAL_cross_condition_test"]
            print(f"    ★关键: edible跨条件cos={crit['edible_dir_cross_condition_cos']:.4f}")
            print(f"    ★关键: color跨条件cos={crit['color_dir_cross_condition_cos']:.4f}")
            print(f"    ★关键: edible⊥color cos={crit['edible_vs_color_orthogonality']:.4f}")
    
    return results


# ============================================================================
# 27B: 非线性检测 — 线性 vs 非线性分类器
# ============================================================================

def expB_nonlinearity_detection(model_name, model, tokenizer, device, layers_to_test):
    """27B: 检测属性编码是线性还是非线性"""
    print(f"\n{'='*60}")
    print(f"ExpB: 非线性检测 — 属性是线性还是非线性?")
    print(f"{'='*60}")
    
    results = {"model": model_name, "exp": "B", "experiment": "nonlinearity_detection", "layers": {}}
    
    for layer_idx in layers_to_test:
        print(f"\n  --- Layer {layer_idx} ---")
        layer_result = {}
        
        for attr_name, attr_data in NONLINEAR_TEST_CONCEPTS.items():
            # 收集所有概念的hidden states
            all_words = []
            all_labels = []
            label_values = list(attr_data.keys())
            
            for label_idx, (label, words) in enumerate(attr_data.items()):
                hs, valid = collect_hs_for_words(model, tokenizer, device, words, layer_idx)
                if hs is not None:
                    all_words.extend(valid)
                    all_labels.extend([label_idx] * len(valid))
                    # 存储hs供后续使用
                    if f"hs_{label}" not in layer_result:
                        layer_result[f"hs_{label}"] = None
                    if layer_result[f"hs_{label}"] is None:
                        layer_result[f"hs_{label}"] = hs
                    else:
                        layer_result[f"hs_{label}"] = np.vstack([layer_result[f"hs_{label}"], hs])
            
            # 重新收集(更简洁)
            all_hs_list = []
            all_labels_list = []
            for label_idx, (label, words) in enumerate(attr_data.items()):
                hs, valid = collect_hs_for_words(model, tokenizer, device, words, layer_idx)
                if hs is not None:
                    all_hs_list.append(hs)
                    all_labels_list.extend([label_idx] * len(hs))
            
            if len(all_hs_list) < 2:
                continue
            
            X = np.vstack(all_hs_list)
            y = np.array(all_labels_list)
            
            if len(set(y)) < 2 or len(X) < 10:
                continue
            
            # 分类器对比
            classifiers = {
                "linear_lr": LogisticRegression(max_iter=1000, C=1.0),
                "linear_lda": LinearDiscriminantAnalysis(),
                "nonlinear_rbf_svm": SVC(kernel='rbf', C=1.0),
                "nonlinear_knn": KNeighborsClassifier(n_neighbors=5),
                "nonlinear_mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
            }
            
            clf_results = {}
            for clf_name, clf in classifiers.items():
                try:
                    # 5-fold交叉验证
                    scores = cross_val_score(clf, X, y, cv=min(5, len(X)//2), scoring='accuracy')
                    clf_results[clf_name] = {
                        "mean_acc": float(np.mean(scores)),
                        "std_acc": float(np.std(scores)),
                    }
                except Exception as e:
                    clf_results[clf_name] = {"mean_acc": -1, "std_acc": -1, "error": str(e)}
            
            # 关键指标: 非线性优势 = max(非线性) - max(线性)
            linear_accs = [v["mean_acc"] for k, v in clf_results.items() if "linear" in k and v["mean_acc"] > 0]
            nonlinear_accs = [v["mean_acc"] for k, v in clf_results.items() if "nonlinear" in k and v["mean_acc"] > 0]
            
            if linear_accs and nonlinear_accs:
                best_linear = max(linear_accs)
                best_nonlinear = max(nonlinear_accs)
                nonlinearity_advantage = best_nonlinear - best_linear
            else:
                best_linear = -1
                best_nonlinear = -1
                nonlinearity_advantage = 0
            
            # 判断
            if nonlinearity_advantage > 0.10:
                structure = "NONLINEAR (advantage > 10%)"
            elif nonlinearity_advantage > 0.05:
                structure = "WEAKLY_NONLINEAR (advantage 5-10%)"
            elif nonlinearity_advantage > 0.02:
                structure = "MOSTLY_LINEAR (advantage 2-5%)"
            else:
                structure = "LINEAR (advantage < 2%)"
            
            attr_result = {
                "n_classes": len(label_values),
                "n_samples": len(X),
                "class_distribution": {label_values[i]: int(np.sum(y == i)) for i in range(len(label_values))},
                "classifiers": clf_results,
                "best_linear_acc": best_linear,
                "best_nonlinear_acc": best_nonlinear,
                "nonlinearity_advantage": float(nonlinearity_advantage),
                "structure_verdict": structure,
            }
            
            layer_result[attr_name] = attr_result
            
            print(f"    {attr_name}: linear={best_linear:.3f}, nonlinear={best_nonlinear:.3f}, "
                  f"advantage={nonlinearity_advantage:.3f} → {structure}")
        
        # 清理临时存储
        for key in list(layer_result.keys()):
            if key.startswith("hs_"):
                del layer_result[key]
        
        results["layers"][str(layer_idx)] = layer_result
    
    return results


# ============================================================================
# 27C: 局部线性结构 — 同类概念偏移一致性
# ============================================================================

def expC_local_linearity(model_name, model, tokenizer, device, layers_to_test):
    """27C: 在局部邻域内测试线性性"""
    print(f"\n{'='*60}")
    print(f"ExpC: 局部线性 — 偏移是否局部一致?")
    print(f"{'='*60}")
    
    results = {"model": model_name, "exp": "C", "experiment": "local_linearity", "layers": {}}
    
    for layer_idx in layers_to_test:
        print(f"\n  --- Layer {layer_idx} ---")
        layer_result = {}
        
        for group_name, group_data in LOCAL_LINEARITY_GROUPS.items():
            children = group_data["children"]
            parent_word = group_data["parent"]
            
            # 收集parent的hs
            parent_hs, _ = collect_hs_for_words(model, tokenizer, device, [parent_word], layer_idx)
            if parent_hs is None or len(parent_hs) == 0:
                continue
            parent_vec = parent_hs[0]
            
            # 收集children的hs
            child_hs, valid_children = collect_hs_for_words(model, tokenizer, device, children, layer_idx)
            if child_hs is None or len(child_hs) < 3:
                continue
            
            # 计算每个child→parent的偏移
            offsets = child_hs - parent_vec  # [n_children, d_model]
            
            # 测试1: 偏移之间的余弦相似度 (核心!)
            n = len(offsets)
            cos_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    cos_matrix[i, j] = compute_cos(offsets[i], offsets[j])
            
            # 上三角(不含对角线)的平均余弦
            upper_tri = cos_matrix[np.triu_indices(n, k=1)]
            mean_offset_cos = float(np.mean(upper_tri))
            min_offset_cos = float(np.min(upper_tri))
            max_offset_cos = float(np.max(upper_tri))
            
            # 测试2: 偏移范数的一致性
            offset_norms = np.linalg.norm(offsets, axis=1)
            norm_cv = float(np.std(offset_norms) / max(np.mean(offset_norms), 1e-10))  # 变异系数
            
            # 测试3: 用PCA分析偏移的维度
            if n >= 3:
                pca = PCA()
                pca.fit(offsets)
                cumvar = np.cumsum(pca.explained_variance_ratio_)
                n90 = int(np.searchsorted(cumvar, 0.9) + 1)
                pc1_var = float(pca.explained_variance_ratio_[0])
            else:
                n90 = -1
                pc1_var = -1
            
            # 测试4: 平均偏移的预测能力
            mean_offset = np.mean(offsets, axis=0)
            predict_correct = 0
            predict_total = 0
            for i in range(n):
                predicted = child_hs[i] + mean_offset  # child + 平均偏移
                actual_parent_plus_offset = parent_vec + offsets[i]
                # 检查predicted是否更接近parent方向
                cos_pred = compute_cos(predicted - child_hs[i], parent_vec - child_hs[i])
                predict_correct += 1 if cos_pred > 0 else 0
                predict_total += 1
            
            # 判断
            if mean_offset_cos > 0.7:
                linearity = "STRONGLY_LINEAR (cos>0.7)"
            elif mean_offset_cos > 0.4:
                linearity = "WEAKLY_LINEAR (cos 0.4-0.7)"
            elif mean_offset_cos > 0.2:
                linearity = "NONLINEAR (cos 0.2-0.4)"
            else:
                linearity = "STRONGLY_NONLINEAR (cos<0.2)"
            
            group_result = {
                "n_children": n,
                "valid_children": valid_children,
                "mean_offset_cos": mean_offset_cos,
                "min_offset_cos": min_offset_cos,
                "max_offset_cos": max_offset_cos,
                "offset_norm_cv": norm_cv,
                "offset_dim_n90": n90,
                "pc1_variance": pc1_var,
                "mean_offset_predict_acc": float(predict_correct / max(predict_total, 1)),
                "linearity_verdict": linearity,
            }
            
            layer_result[group_name] = group_result
            print(f"    {group_name}: mean_cos={mean_offset_cos:.4f}, n90={n90}, "
                  f"norm_cv={norm_cv:.3f} → {linearity}")
        
        results["layers"][str(layer_idx)] = layer_result
    
    return results


# ============================================================================
# 27D: 最小因果子空间
# ============================================================================

def expD_minimal_causal_subspace(model_name, model, tokenizer, device, layers_to_test):
    """27D: 找到最小维度子空间使得干预仍有效"""
    print(f"\n{'='*60}")
    print(f"ExpD: 最小因果子空间 — 属性干预需要多少维?")
    print(f"{'='*60}")
    
    results = {"model": model_name, "exp": "D", "experiment": "minimal_causal_subspace", "layers": {}}
    
    W_U = get_W_U(model)  # [vocab_size, d_model]
    
    # 目标词
    edible_words = ["eat", "food", "delicious", "tasty", "fruit", "hungry"]
    not_edible_words = ["tool", "use", "build", "hard", "stone", "metal"]
    
    edible_token_ids = []
    for w in edible_words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        edible_token_ids.extend(ids)
    
    not_edible_token_ids = []
    for w in not_edible_words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        not_edible_token_ids.extend(ids)
    
    for layer_idx in layers_to_test:
        print(f"\n  --- Layer {layer_idx} ---")
        layer_result = {}
        
        # 收集可食/不可食概念的hidden states
        edible_concepts = NONLINEAR_TEST_CONCEPTS["edible"]["yes"][:10]
        not_edible_concepts = NONLINEAR_TEST_CONCEPTS["edible"]["no"][:10]
        
        edible_hs, _ = collect_hs_for_words(model, tokenizer, device, edible_concepts, layer_idx)
        not_edible_hs, _ = collect_hs_for_words(model, tokenizer, device, not_edible_concepts, layer_idx)
        
        if edible_hs is None or not_edible_hs is None:
            print(f"    数据不足, 跳过")
            continue
        
        # 计算可食性方向
        edible_centroid = np.mean(edible_hs, axis=0)
        not_edible_centroid = np.mean(not_edible_hs, axis=0)
        edible_direction = edible_centroid - not_edible_centroid
        edible_dir_norm = np.linalg.norm(edible_direction)
        if edible_dir_norm < 1e-10:
            continue
        edible_dir_normalized = edible_direction / edible_dir_norm
        
        # 对所有概念做PCA, 找到主成分方向
        all_hs = np.vstack([edible_hs, not_edible_hs])
        pca = PCA(n_components=min(50, all_hs.shape[0]-1, all_hs.shape[1]))
        pca.fit(all_hs)
        
        # 测试: 在不同维度的子空间中干预, 看效果
        # 方法: 只保留前k个PC方向, 将edible_direction投影到该子空间
        test_concept_hs = not_edible_hs[:5]  # 对5个不可食概念做干预
        
        subspace_results = {}
        for k in [1, 2, 3, 5, 10, 20, 50]:
            if k > pca.n_components_:
                continue
            
            # 子空间基
            basis = pca.components_[:k]  # [k, d_model]
            
            # 将edible_direction投影到子空间
            proj_coeffs = basis @ edible_direction  # [k]
            projected_dir = basis.T @ proj_coeffs  # [d_model]
            proj_norm = np.linalg.norm(projected_dir)
            
            if proj_norm < 1e-10:
                subspace_results[f"k={k}"] = {
                    "projection_ratio": 0,
                    "edible_delta": 0,
                    "not_edible_delta": 0,
                }
                continue
            
            projected_dir_normalized = projected_dir / proj_norm
            
            # 干预测试
            beta = 2.0
            edible_deltas = []
            not_edible_deltas = []
            
            for hs in test_concept_hs:
                hs_intervened = hs + beta * projected_dir_normalized * edible_dir_norm
                
                logits_orig = W_U @ hs
                logits_int = W_U @ hs_intervened
                
                if edible_token_ids:
                    ed_orig = float(np.mean(logits_orig[edible_token_ids]))
                    ed_int = float(np.mean(logits_int[edible_token_ids]))
                    edible_deltas.append(ed_int - ed_orig)
                
                if not_edible_token_ids:
                    ned_orig = float(np.mean(logits_orig[not_edible_token_ids]))
                    ned_int = float(np.mean(logits_int[not_edible_token_ids]))
                    not_edible_deltas.append(ned_int - ned_orig)
            
            # 也测试完整的edible_direction (不在子空间中截断)
            edible_deltas_full = []
            not_edible_deltas_full = []
            for hs in test_concept_hs:
                hs_intervened = hs + beta * edible_dir_normalized * edible_dir_norm
                logits_int = W_U @ hs_intervened
                logits_orig = W_U @ hs
                if edible_token_ids:
                    edible_deltas_full.append(float(np.mean(logits_int[edible_token_ids])) - float(np.mean(logits_orig[edible_token_ids])))
                if not_edible_token_ids:
                    not_edible_deltas_full.append(float(np.mean(logits_int[not_edible_token_ids])) - float(np.mean(logits_orig[not_edible_token_ids])))
            
            projection_ratio = float(proj_norm / edible_dir_norm)  # 投影保留了多大比例
            
            subspace_results[f"k={k}"] = {
                "projection_ratio": projection_ratio,
                "edible_delta": float(np.mean(edible_deltas)) if edible_deltas else 0,
                "not_edible_delta": float(np.mean(not_edible_deltas)) if not_edible_deltas else 0,
                "selectivity": float(np.mean(edible_deltas) - np.mean(not_edible_deltas)) if edible_deltas and not_edible_deltas else 0,
            }
        
        # 完整方向的基线
        subspace_results["k=full"] = {
            "projection_ratio": 1.0,
            "edible_delta": float(np.mean(edible_deltas_full)) if edible_deltas_full else 0,
            "not_edible_delta": float(np.mean(not_edible_deltas_full)) if not_edible_deltas_full else 0,
            "selectivity": float(np.mean(edible_deltas_full) - np.mean(not_edible_deltas_full)) if edible_deltas_full and not_edible_deltas_full else 0,
        }
        
        # 找到最小有效子空间
        min_k = None
        for k_key, k_data in subspace_results.items():
            if k_key == "k=full":
                continue
            if k_data["selectivity"] > 0.5 * subspace_results["k=full"]["selectivity"]:
                k_val = int(k_key.split("=")[1])
                if min_k is None or k_val < min_k:
                    min_k = k_val
        
        layer_result = {
            "edible_dir_norm": float(edible_dir_norm),
            "subspace_intervention": subspace_results,
            "minimal_effective_subspace": min_k,
            "interpretation": f"属性干预至少需要{min_k}维子空间" if min_k else "无法确定",
        }
        
        results["layers"][str(layer_idx)] = layer_result
        
        print(f"    可食性方向范数: {edible_dir_norm:.2f}")
        for k_key, k_data in subspace_results.items():
            print(f"    {k_key}: proj_ratio={k_data['projection_ratio']:.3f}, "
                  f"edible_delta={k_data['edible_delta']:.4f}, "
                  f"selectivity={k_data['selectivity']:.4f}")
        print(f"    最小有效子空间维度: {min_k}")
    
    return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CCMG Phase27: 属性条件解耦与非线性检测")
    parser.add_argument("--model", type=str, default="qwen3",
                       choices=["qwen3", "glm4", "deepseek7b"],
                       help="模型名称")
    parser.add_argument("--exp", type=str, default="0",
                       help="实验编号 (0=全部, A/B/C/D)")
    args = parser.parse_args()
    
    model_name = args.model
    exp_id = args.exp
    
    print(f"\n{'='*60}")
    print(f"CCMG Phase 27: 属性是否真实存在——条件解耦与非线性检测")
    print(f"模型: {model_name}, 实验: {exp_id if exp_id != '0' else '全部'}")
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
    
    # 选择测试层 — 聚焦中层(Phase26发现中层干预最有效)
    n_layers = model_info.n_layers
    fracture_layers = {"qwen3": 6, "glm4": 2, "deepseek7b": 7}
    fl = fracture_layers.get(model_name, n_layers // 3)
    
    layers_to_test = sorted(set([
        0, fl-1, fl, fl+1, fl+2,
        n_layers // 3, n_layers // 2, 2 * n_layers // 3,
        n_layers - 2, n_layers - 1,
    ] + list(range(max(0, fl-1), min(n_layers, fl+5)))))
    layers_to_test = [l for l in layers_to_test if 0 <= l < n_layers]
    print(f"测试层: {layers_to_test}")
    
    # 运行实验
    all_results = []
    
    try:
        if exp_id in ["0", "A"]:
            rA = expA_conditional_decoupling(model_name, model, tokenizer, device, layers_to_test)
            all_results.append(rA)
        
        if exp_id in ["0", "B"]:
            rB = expB_nonlinearity_detection(model_name, model, tokenizer, device, layers_to_test)
            all_results.append(rB)
        
        if exp_id in ["0", "C"]:
            rC = expC_local_linearity(model_name, model, tokenizer, device, layers_to_test)
            all_results.append(rC)
        
        if exp_id in ["0", "D"]:
            rD = expD_minimal_causal_subspace(model_name, model, tokenizer, device, layers_to_test)
            all_results.append(rD)
    
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
        exp_label = result["exp"]
        output_file = os.path.join(output_dir, f"ccmg_exp{exp_label}_{model_name}_results.json")
        result = convert_numpy(result)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {output_file}")
    
    print(f"\n{'='*60}")
    print(f"Phase 27 完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
