"""
CCXLIX(409): 跨层语法信息扫描
=========================================
Phase 3A: 找到语法信息最丰富的层

CCXLVIII发现:
  - GLM4最后一层pair_diff仅0.524, 远低于Qwen3的0.803
  - 说明最后一层不一定是语法层
  - 需要跨层扫描找到语法信息最丰富的层

核心实验:
  Exp1: 全层LDA分类 — 每层提取pair特征, 用LDA分类8类依存
  Exp2: 全层方向分类 — 每层的差向量方向分类准确率
  Exp3: 语法vs语义分离 — 语法(依存类型)和语义(词义)的层间分布
  Exp4: 注意力头归因 — 在语法最丰富的层分析注意力头
  Exp5: 代数结构探测 — 在判别子空间中测试组合律

关键假设:
  - 中间层语法信息最强, 最后层语义信息最强
  - 方向编码在所有层都稳定
  - 类型编码在中层达到峰值
"""

import torch
import numpy as np
import json
import argparse
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_rel, pearsonr
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tests.glm5.model_utils import load_model, get_layers

_DEVICE = "cuda:0"

# 复用CCXLVIII的自然句语料和解析
from tests.glm5.ccxlviii_large_scale_natural import (
    NATURAL_SENTENCES, TARGET_DEP_TYPES, DEP_TYPE_MAP, parse_with_spacy
)


def extract_pair_features_at_layer(model, tokenizer, sentence, head_idx, dep_idx, layer_idx):
    """Extract token representations at a specific layer."""
    layers = get_layers(model)
    if layer_idx >= len(layers):
        layer_idx = len(layers) - 1
    
    toks = tokenizer(sentence, return_tensors="pt").to(_DEVICE)
    input_ids = toks["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    n_tokens = len(tokens)
    
    has_bos = tokens[0] in ['<s>', '<|begin_of_text|>', '<|im_start|>', 'Ċ', 'Ċ']
    offset = 1 if has_bos else 0
    actual_head = min(head_idx + offset, n_tokens - 1)
    actual_dep = min(dep_idx + offset, n_tokens - 1)
    
    captured = {}
    def make_hook():
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured["h"] = output[0].detach().float().cpu()
            else:
                captured["h"] = output.detach().float().cpu()
        return hook
    
    hook = layers[layer_idx].register_forward_hook(make_hook())
    with torch.no_grad():
        _ = model(**toks)
    hook.remove()
    
    if "h" not in captured:
        return None
    
    hidden = captured["h"][0].numpy()
    h_head = hidden[actual_head]
    h_dep = hidden[actual_dep]
    
    return {
        "h_head": h_head,
        "h_dep": h_dep,
        "pair_diff": h_head - h_dep,
        "pair_concat": np.concatenate([h_head, h_dep]),
    }


def extract_all_layers_for_pair(model, tokenizer, sentence, head_idx, dep_idx, sample_layers):
    """Extract features at multiple layers for one pair. Returns {layer_idx: features_dict}."""
    layers = get_layers(model)
    n_layers = len(layers)
    toks = tokenizer(sentence, return_tensors="pt").to(_DEVICE)
    input_ids = toks["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    n_tokens = len(tokens)
    
    has_bos = tokens[0] in ['<s>', '<|begin_of_text|>', '<|im_start|>', 'Ċ', 'Ċ']
    offset = 1 if has_bos else 0
    actual_head = min(head_idx + offset, n_tokens - 1)
    actual_dep = min(dep_idx + offset, n_tokens - 1)
    
    # Hook all sample layers at once
    captured = {}
    def make_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook
    
    hooks = []
    for li in sample_layers:
        if li < n_layers:
            hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))
    
    with torch.no_grad():
        _ = model(**toks)
    
    for h in hooks:
        h.remove()
    
    results = {}
    for li in sample_layers:
        key = f"L{li}"
        if key not in captured:
            continue
        hidden = captured[key][0].numpy()
        h_head = hidden[actual_head]
        h_dep = hidden[actual_dep]
        results[li] = {
            "h_head": h_head,
            "h_dep": h_dep,
            "pair_diff": h_head - h_dep,
            "pair_concat": np.concatenate([h_head, h_dep]),
        }
    
    return results


# ============================================================
# Exp1: 全层LDA分类
# ============================================================

def exp1_cross_layer_lda(model_name, model, tokenizer, max_per_type=40):
    """Layer-by-layer LDA classification of dependency types."""
    print(f"\n{'='*70}")
    print(f"Exp1: 全层LDA分类 ({model_name})")
    print(f"{'='*70}")
    
    layers = get_layers(model)
    n_layers = len(layers)
    
    # Sample layers: every 2nd + first/last
    if n_layers <= 20:
        sample_layers = list(range(n_layers))
    else:
        sample_layers = sorted(set(
            [0, 1] + list(range(0, n_layers, 2)) + [n_layers - 2, n_layers - 1]
        ))
    print(f"  Model has {n_layers} layers, sampling {len(sample_layers)}: {sample_layers}")
    
    # Parse sentences
    pairs = parse_with_spacy(NATURAL_SENTENCES)
    type_sampled = {dt: 0 for dt in TARGET_DEP_TYPES}
    balanced_pairs = []
    for p in pairs:
        _, _, _, dt = p
        if dt in TARGET_DEP_TYPES and type_sampled[dt] < max_per_type:
            balanced_pairs.append(p)
            type_sampled[dt] += 1
    
    print(f"  Parsed {len(balanced_pairs)} balanced pairs")
    type_counts = {}
    for _, _, _, dt in balanced_pairs:
        type_counts[dt] = type_counts.get(dt, 0) + 1
    print(f"  Distribution: {type_counts}")
    
    # Extract features at all sample layers for all pairs
    print(f"\n  Extracting features at {len(sample_layers)} layers for {len(balanced_pairs)} pairs...")
    
    # Data structure: {layer_idx: {feature_name: list_of_arrays}}
    layer_data = {li: {"h_head": [], "h_dep": [], "pair_diff": [], "pair_concat": [], "y": []} 
                  for li in sample_layers}
    
    valid_count = 0
    for i, (sent, h_idx, d_idx, dep_type) in enumerate(balanced_pairs):
        if i % 50 == 0:
            print(f"    Processing pair {i}/{len(balanced_pairs)}...")
        try:
            multi_layer_feats = extract_all_layers_for_pair(
                model, tokenizer, sent, h_idx, d_idx, sample_layers
            )
            for li in sample_layers:
                if li in multi_layer_feats:
                    feats = multi_layer_feats[li]
                    layer_data[li]["h_head"].append(feats["h_head"])
                    layer_data[li]["h_dep"].append(feats["h_dep"])
                    layer_data[li]["pair_diff"].append(feats["pair_diff"])
                    layer_data[li]["pair_concat"].append(feats["pair_concat"])
                    if li == sample_layers[0]:
                        layer_data[li]["y"].append(TARGET_DEP_TYPES.index(dep_type))
                    else:
                        layer_data[li]["y"].append(TARGET_DEP_TYPES.index(dep_type))
            valid_count += 1
        except Exception as e:
            continue
    
    # Use y from first layer (should be same across all)
    y = np.array(layer_data[sample_layers[0]]["y"])
    n_valid = len(y)
    n_classes = len(np.unique(y))
    print(f"  Valid pairs: {n_valid}")
    
    if n_valid < 30:
        print("  Too few samples!")
        return {"error": "insufficient_samples", "n": n_valid}
    
    results = {"n_samples": n_valid, "n_layers": n_layers, "sample_layers": sample_layers}
    
    # For each layer, classify using pair_diff, pair_concat, and LDA
    print(f"\n  === Layer-by-layer classification ===")
    layer_results = {}
    
    for li in sample_layers:
        n_this = len(layer_data[li]["y"])
        if n_this < 30:
            continue
        
        y_li = np.array(layer_data[li]["y"])
        X_diff = np.array(layer_data[li]["pair_diff"])
        X_concat = np.array(layer_data[li]["pair_concat"])
        X_dep = np.array(layer_data[li]["h_dep"])
        
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
        clf = LinearSVC(max_iter=5000, C=1.0, dual=False)
        
        # pair_diff
        scores_diff = cross_val_score(clf, X_diff, y_li, cv=cv, scoring='accuracy')
        
        # pair_concat
        scores_concat = cross_val_score(clf, X_concat, y_li, cv=cv, scoring='accuracy')
        
        # point_dep
        scores_dep = cross_val_score(clf, X_dep, y_li, cv=cv, scoring='accuracy')
        
        # LDA on pair_concat
        n_lda = min(n_classes - 1, 7)
        try:
            lda = LinearDiscriminantAnalysis(n_components=n_lda)
            X_lda = lda.fit_transform(X_concat, y_li)
            scores_lda = cross_val_score(clf, X_lda, y_li, cv=cv, scoring='accuracy')
        except:
            scores_lda = np.array([0.0])
        
        layer_results[li] = {
            "pair_diff": float(scores_diff.mean()),
            "pair_diff_std": float(scores_diff.std()),
            "pair_concat": float(scores_concat.mean()),
            "pair_concat_std": float(scores_concat.std()),
            "point_dep": float(scores_dep.mean()),
            "point_dep_std": float(scores_dep.std()),
            "lda": float(scores_lda.mean()),
            "lda_std": float(scores_lda.std()),
            "n": n_this,
        }
        
        print(f"  L{li:2d}/{n_layers-1}: diff={scores_diff.mean():.3f} concat={scores_concat.mean():.3f} "
              f"dep={scores_dep.mean():.3f} LDA={scores_lda.mean():.3f} (n={n_this})")
    
    results["layer_results"] = layer_results
    
    # Find peak layer
    if layer_results:
        best_layer_diff = max(layer_results, key=lambda k: layer_results[k]["pair_diff"])
        best_layer_concat = max(layer_results, key=lambda k: layer_results[k]["pair_concat"])
        best_layer_lda = max(layer_results, key=lambda k: layer_results[k]["lda"])
        best_layer_dep = max(layer_results, key=lambda k: layer_results[k]["point_dep"])
        
        results["best_layer_diff"] = best_layer_diff
        results["best_layer_concat"] = best_layer_concat
        results["best_layer_lda"] = best_layer_lda
        results["best_layer_dep"] = best_layer_dep
        
        print(f"\n  ★ Best layers:")
        print(f"    pair_diff:   L{best_layer_diff} ({layer_results[best_layer_diff]['pair_diff']:.3f})")
        print(f"    pair_concat: L{best_layer_concat} ({layer_results[best_layer_concat]['pair_concat']:.3f})")
        print(f"    LDA:         L{best_layer_lda} ({layer_results[best_layer_lda]['lda']:.3f})")
        print(f"    point_dep:   L{best_layer_dep} ({layer_results[best_layer_dep]['point_dep']:.3f})")
        
        # Is the peak in the middle?
        mid = n_layers // 2
        is_mid = (abs(best_layer_diff - mid) < n_layers * 0.3)
        results["peak_in_middle"] = is_mid
        print(f"    Peak in middle third? {is_mid} (mid={mid})")
    
    return results


# ============================================================
# Exp2: 全层方向分类
# ============================================================

def exp2_cross_layer_direction(model_name, model, tokenizer, max_per_type=30):
    """Layer-by-layer direction classification."""
    print(f"\n{'='*70}")
    print(f"Exp2: 全层方向分类 ({model_name})")
    print(f"{'='*70}")
    
    layers = get_layers(model)
    n_layers = len(layers)
    
    sample_layers = sorted(set(
        [0, 1] + list(range(0, n_layers, 3)) + [n_layers - 2, n_layers - 1]
    ))
    print(f"  Sampling {len(sample_layers)} layers")
    
    pairs = parse_with_spacy(NATURAL_SENTENCES)
    type_sampled = {dt: 0 for dt in TARGET_DEP_TYPES}
    balanced_pairs = []
    for p in pairs:
        _, _, _, dt = p
        if dt in TARGET_DEP_TYPES and type_sampled[dt] < max_per_type:
            balanced_pairs.append(p)
            type_sampled[dt] += 1
    
    print(f"  Using {len(balanced_pairs)} pairs")
    
    layer_data = {li: {"diff_fwd": [], "diff_rev": []} for li in sample_layers}
    valid_count = 0
    
    for i, (sent, h_idx, d_idx, dep_type) in enumerate(balanced_pairs):
        if i % 50 == 0:
            print(f"    Processing pair {i}/{len(balanced_pairs)}...")
        try:
            multi_layer_feats = extract_all_layers_for_pair(
                model, tokenizer, sent, h_idx, d_idx, sample_layers
            )
            for li in sample_layers:
                if li in multi_layer_feats:
                    h_head = multi_layer_feats[li]["h_head"]
                    h_dep = multi_layer_feats[li]["h_dep"]
                    layer_data[li]["diff_fwd"].append(h_head - h_dep)
                    layer_data[li]["diff_rev"].append(h_dep - h_head)
            valid_count += 1
        except:
            continue
    
    print(f"  Valid pairs: {valid_count}")
    
    results = {"n_samples": valid_count, "sample_layers": sample_layers}
    layer_dir_results = {}
    
    for li in sample_layers:
        n_fwd = len(layer_data[li]["diff_fwd"])
        if n_fwd < 10:
            continue
        
        X = np.array(layer_data[li]["diff_fwd"] + layer_data[li]["diff_rev"])
        y = np.array([1] * n_fwd + [0] * n_fwd)
        
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
        clf = LinearSVC(max_iter=5000, C=1.0, dual=False)
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        
        layer_dir_results[li] = {
            "direction_acc": float(scores.mean()),
            "direction_std": float(scores.std()),
            "n": n_fwd,
        }
        print(f"  L{li:2d}/{n_layers-1}: direction={scores.mean():.3f} ± {scores.std():.3f}")
    
    results["layer_direction"] = layer_dir_results
    
    # Find direction stability
    if layer_dir_results:
        accs = [layer_dir_results[li]["direction_acc"] for li in sorted(layer_dir_results)]
        results["direction_min"] = min(accs)
        results["direction_max"] = max(accs)
        results["direction_range"] = max(accs) - min(accs)
        results["direction_mean"] = np.mean(accs)
        print(f"\n  Direction stability: min={min(accs):.3f} max={max(accs):.3f} "
              f"range={max(accs)-min(accs):.3f} mean={np.mean(accs):.3f}")
    
    return results


# ============================================================
# Exp3: 语法vs语义分离 — 词义干扰控制
# ============================================================

def exp3_syntax_vs_semantic(model_name, model, tokenizer, max_per_type=40):
    """Test if syntactic classification is robust to semantic variation.
    
    Key idea: If the classifier uses syntax (not semantics), then:
    - Same word in different syntactic roles should have different representations
    - Different words in same syntactic role should have similar representations
    
    Test: word-level cross-validation (leave-one-word-out)
    """
    print(f"\n{'='*70}")
    print(f"Exp3: 语法vs语义分离 ({model_name})")
    print(f"{'='*70}")
    
    layers = get_layers(model)
    n_layers = len(layers)
    
    # Focus on key layers: early, mid, late
    key_layers = sorted(set([0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]))
    print(f"  Key layers: {key_layers}")
    
    pairs = parse_with_spacy(NATURAL_SENTENCES)
    
    # Group pairs by dependency type
    type_pairs = {dt: [] for dt in TARGET_DEP_TYPES}
    for p in pairs:
        _, _, _, dt = p
        if dt in type_pairs:
            type_pairs[dt].append(p)
    
    # Only keep types with enough pairs
    valid_types = [dt for dt in TARGET_DEP_TYPES if len(type_pairs[dt]) >= 15]
    print(f"  Types with >=15 pairs: {valid_types}")
    
    if len(valid_types) < 3:
        print("  Too few types with enough samples!")
        return {"error": "insufficient_types"}
    
    # Sample pairs
    type_sampled = {dt: 0 for dt in valid_types}
    balanced_pairs = []
    for p in pairs:
        _, _, _, dt = p
        if dt in valid_types and type_sampled[dt] < max_per_type:
            balanced_pairs.append(p)
            type_sampled[dt] += 1
    
    # Extract features at key layers
    layer_data = {li: {"pair_diff": [], "y": [], "sentences": []} for li in key_layers}
    
    for i, (sent, h_idx, d_idx, dep_type) in enumerate(balanced_pairs):
        if i % 50 == 0:
            print(f"    Processing pair {i}/{len(balanced_pairs)}...")
        try:
            multi_layer_feats = extract_all_layers_for_pair(
                model, tokenizer, sent, h_idx, d_idx, key_layers
            )
            for li in key_layers:
                if li in multi_layer_feats:
                    layer_data[li]["pair_diff"].append(multi_layer_feats[li]["pair_diff"])
                    layer_data[li]["y"].append(valid_types.index(dep_type))
                    layer_data[li]["sentences"].append(sent)
        except:
            continue
    
    results = {}
    
    for li in key_layers:
        n = len(layer_data[li]["y"])
        if n < 30:
            continue
        
        X = np.array(layer_data[li]["pair_diff"])
        y = np.array(layer_data[li]["y"])
        
        # Standard CV
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
        clf = LinearSVC(max_iter=5000, C=1.0, dual=False)
        scores_std = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        
        # Sentence-grouped CV: train on sentences with different words, test on held-out sentences
        # This controls for word-specific memorization
        unique_sents = list(set(layer_data[li]["sentences"]))
        np.random.seed(42)
        np.random.shuffle(unique_sents)
        n_test = max(1, len(unique_sents) // 5)
        test_sents = set(unique_sents[:n_test])
        train_sents = set(unique_sents[n_test:])
        
        train_mask = np.array([s in train_sents for s in layer_data[li]["sentences"]])
        test_mask = np.array([s in test_sents for s in layer_data[li]["sentences"]])
        
        if train_mask.sum() > 10 and test_mask.sum() > 5 and len(np.unique(y[train_mask])) >= 2:
            clf.fit(X[train_mask], y[train_mask])
            acc_sent = accuracy_score(y[test_mask], clf.predict(X[test_mask]))
        else:
            acc_sent = 0.0
        
        # Cosine similarity analysis
        # Within-type vs between-type similarity
        type_means = {}
        for t_idx in range(len(valid_types)):
            mask = y == t_idx
            if mask.sum() > 0:
                type_means[t_idx] = X[mask].mean(axis=0)
        
        within_sim = []
        between_sim = []
        type_list = list(type_means.keys())
        for i_t in type_list:
            mask_i = y == i_t
            X_i = X[mask_i]
            mean_i = type_means[i_t]
            for x in X_i:
                within_sim.append(np.dot(x, mean_i) / (np.linalg.norm(x) * np.linalg.norm(mean_i) + 1e-10))
            for j_t in type_list:
                if j_t != i_t:
                    mean_j = type_means[j_t]
                    between_sim.append(np.dot(mean_i, mean_j) / (np.linalg.norm(mean_i) * np.linalg.norm(mean_j) + 1e-10))
        
        results[li] = {
            "standard_cv": float(scores_std.mean()),
            "sentence_grouped": float(acc_sent),
            "within_type_cos": float(np.mean(within_sim)),
            "between_type_cos": float(np.mean(between_sim)),
            "cos_separation": float(np.mean(within_sim) - np.mean(between_sim)),
            "n": n,
        }
        
        print(f"  L{li:2d}: std_cv={scores_std.mean():.3f} sent_grouped={acc_sent:.3f} "
              f"within_cos={np.mean(within_sim):.3f} between_cos={np.mean(between_sim):.3f} "
              f"separation={np.mean(within_sim)-np.mean(between_sim):.3f}")
    
    return results


# ============================================================
# Exp4: 代数结构探测 — 在判别子空间中测试组合律
# ============================================================

def exp4_algebraic_structure(model_name, model, tokenizer, max_per_type=30):
    """Probe algebraic structure in the discriminative subspace.
    
    Test composition laws:
    1. Chain: d(s→v) + d(v→o) ≈ d(s→o)?  (subject→verb→object chain)
    2. Parallel: d(n→a1) vs d(n→a2)  (multiple modifiers of same noun)
    3. Crossover: swapping arguments changes direction
    """
    print(f"\n{'='*70}")
    print(f"Exp4: 代数结构探测 ({model_name})")
    print(f"{'='*70}")
    
    layers = get_layers(model)
    n_layers = len(layers)
    
    # Use best layer from CCXLVIII if available, otherwise use middle layer
    mid_layer = n_layers // 2
    key_layers = sorted(set([n_layers//4, mid_layer, 3*n_layers//4, n_layers-1]))
    print(f"  Key layers: {key_layers}")
    
    pairs = parse_with_spacy(NATURAL_SENTENCES)
    
    # === Test 1: Chain composition ===
    # Find sentences with s→v→o chain
    print(f"\n  === Test 1: Chain composition d(s→v) + d(v→o) vs d(s→o) ===")
    
    # Build: for each sentence, find subj_verb and verb_obj pairs
    sent_sv_vo = {}  # {sentence: {"sv": (h,d), "vo": (h,d)}}
    sv_pairs = [(s, h, d) for s, h, d, dt in pairs if dt == "subj_verb"]
    vo_pairs = [(s, h, d) for s, h, d, dt in pairs if dt == "verb_obj"]
    
    # Group by sentence
    from collections import defaultdict
    sv_by_sent = defaultdict(list)
    vo_by_sent = defaultdict(list)
    for s, h, d in sv_pairs:
        sv_by_sent[s].append((h, d))
    for s, h, d in vo_pairs:
        vo_by_sent[s].append((h, d))
    
    chain_sentences = []
    for s in sv_by_sent:
        if s in vo_by_sent:
            for sv_h, sv_d in sv_by_sent[s]:
                for vo_h, vo_d in vo_by_sent[s]:
                    # sv_d should be the verb (head of vo pair)
                    if sv_d == vo_h:
                        chain_sentences.append((s, sv_h, sv_d, vo_d))
    
    print(f"  Found {len(chain_sentences)} s→v→o chains")
    
    chain_results = {}
    for li in key_layers:
        d_sv_list = []
        d_vo_list = []
        
        for sent, subj_idx, verb_idx, obj_idx in chain_sentences[:20]:
            try:
                feats = extract_pair_features_at_layer(
                    model, tokenizer, sent, subj_idx, verb_idx, li
                )
                if feats:
                    d_sv_list.append(feats["pair_diff"])
                
                feats = extract_pair_features_at_layer(
                    model, tokenizer, sent, verb_idx, obj_idx, li
                )
                if feats:
                    d_vo_list.append(feats["pair_diff"])
            except:
                continue
        
        if len(d_sv_list) >= 5 and len(d_vo_list) >= 5:
            d_sv = np.array(d_sv_list)
            d_vo = np.array(d_vo_list)
            n_chain = min(len(d_sv), len(d_vo))
            d_sv = d_sv[:n_chain]
            d_vo = d_vo[:n_chain]
            
            # Test: d(s→v) + d(v→o) vs d(s→o)
            d_sum = d_sv + d_vo  # This would be d(s→o) if chain law holds
            
            # Measure: cosine similarity between d_sv and d_vo
            cosines = []
            for i in range(n_chain):
                c = np.dot(d_sv[i], d_vo[i]) / (np.linalg.norm(d_sv[i]) * np.linalg.norm(d_vo[i]) + 1e-10)
                cosines.append(c)
            
            # Also: norm ratio
            norm_ratios = np.linalg.norm(d_vo, axis=1) / (np.linalg.norm(d_sv, axis=1) + 1e-10)
            
            # Test linearity: d_sum norm vs d_sv norm + d_vo norm
            norms_sum = np.linalg.norm(d_sum, axis=1)
            norms_individual = np.linalg.norm(d_sv, axis=1) + np.linalg.norm(d_vo, axis=1)
            norm_ratio = norms_sum / (norms_individual + 1e-10)
            
            chain_results[li] = {
                "n_chains": n_chain,
                "cosine_sv_vo": float(np.mean(cosines)),
                "norm_ratio_vo_sv": float(np.mean(norm_ratios)),
                "sum_norm_ratio": float(np.mean(norm_ratio)),
                "cosine_std": float(np.std(cosines)),
            }
            print(f"  L{li:2d}: cos(sv,vo)={np.mean(cosines):.3f}±{np.std(cosines):.3f} "
                  f"norm_ratio={np.mean(norm_ratios):.3f} sum_norm_ratio={np.mean(norm_ratio):.3f}")
        else:
            print(f"  L{li:2d}: not enough chains ({len(d_sv_list)} sv, {len(d_vo_list)} vo)")
    
    # === Test 2: Parallel modifiers ===
    print(f"\n  === Test 2: Parallel modifiers ===")
    
    # Find nouns with multiple adjectives
    na_pairs = [(s, h, d) for s, h, d, dt in pairs if dt == "noun_adj"]
    nd_pairs = [(s, h, d) for s, h, d, dt in pairs if dt == "noun_det"]
    
    na_by_sent_head = defaultdict(list)
    for s, h, d in na_pairs:
        na_by_sent_head[(s, h)].append(d)
    
    # Find heads with 2+ adjective modifiers
    parallel_mods = []
    for (s, h), deps in na_by_sent_head.items():
        if len(deps) >= 2:
            for i in range(len(deps)):
                for j in range(i+1, len(deps)):
                    parallel_mods.append((s, h, deps[i], deps[j]))
    
    print(f"  Found {len(parallel_mods)} parallel modifier pairs")
    
    parallel_results = {}
    for li in key_layers:
        d_na1_list = []
        d_na2_list = []
        
        for sent, head_idx, adj1_idx, adj2_idx in parallel_mods[:15]:
            try:
                feats1 = extract_pair_features_at_layer(
                    model, tokenizer, sent, head_idx, adj1_idx, li
                )
                feats2 = extract_pair_features_at_layer(
                    model, tokenizer, sent, head_idx, adj2_idx, li
                )
                if feats1 and feats2:
                    d_na1_list.append(feats1["pair_diff"])
                    d_na2_list.append(feats2["pair_diff"])
            except:
                continue
        
        if len(d_na1_list) >= 3:
            d_na1 = np.array(d_na1_list)
            d_na2 = np.array(d_na2_list)
            n_par = min(len(d_na1), len(d_na2))
            
            # Test commutativity: d(n→a1) + d(n→a2) ≈ d(n→a2) + d(n→a1)
            # (Trivially true for vector addition, but test if directions are similar)
            cosines = []
            for i in range(n_par):
                c = np.dot(d_na1[i], d_na2[i]) / (np.linalg.norm(d_na1[i]) * np.linalg.norm(d_na2[i]) + 1e-10)
                cosines.append(c)
            
            # Test: same-noun modifiers should have similar directions
            # (All noun_adj pairs share the same dependency type)
            
            parallel_results[li] = {
                "n_parallel": n_par,
                "cosine_na1_na2": float(np.mean(cosines)),
                "cosine_std": float(np.std(cosines)),
            }
            print(f"  L{li:2d}: cos(na1,na2)={np.mean(cosines):.3f}±{np.std(cosines):.3f}")
        else:
            print(f"  L{li:2d}: not enough parallel pairs ({len(d_na1_list)})")
    
    # === Test 3: Type-discriminability in LDA space ===
    print(f"\n  === Test 3: LDA discriminability ===")
    
    type_sampled = {dt: 0 for dt in TARGET_DEP_TYPES}
    balanced_pairs = []
    for p in pairs:
        _, _, _, dt = p
        if dt in TARGET_DEP_TYPES and type_sampled[dt] < max_per_type:
            balanced_pairs.append(p)
            type_sampled[dt] += 1
    
    lda_results = {}
    for li in key_layers:
        X_diff = []
        y = []
        for sent, h_idx, d_idx, dep_type in balanced_pairs:
            try:
                feats = extract_pair_features_at_layer(
                    model, tokenizer, sent, h_idx, d_idx, li
                )
                if feats:
                    X_diff.append(feats["pair_diff"])
                    y.append(TARGET_DEP_TYPES.index(dep_type))
            except:
                continue
        
        if len(X_diff) < 30:
            continue
        
        X_diff = np.array(X_diff)
        y = np.array(y)
        n_classes = len(np.unique(y))
        
        # LDA
        n_lda = min(n_classes - 1, 7)
        try:
            lda = LinearDiscriminantAnalysis(n_components=n_lda)
            X_lda = lda.fit_transform(X_diff, y)
            
            # In LDA space, compute inter-type distances
            type_means_lda = {}
            for t in range(n_classes):
                mask = y == t
                if mask.sum() > 0:
                    type_means_lda[t] = X_lda[mask].mean(axis=0)
            
            # Distance matrix
            type_names = [TARGET_DEP_TYPES[t] for t in sorted(type_means_lda.keys())]
            dist_matrix = np.zeros((len(type_means_lda), len(type_means_lda)))
            for i, ti in enumerate(sorted(type_means_lda.keys())):
                for j, tj in enumerate(sorted(type_means_lda.keys())):
                    dist_matrix[i, j] = np.linalg.norm(type_means_lda[ti] - type_means_lda[tj])
            
            # Key: argument-argument distance vs modifier-modifier distance vs cross distance
            arg_types = {TARGET_DEP_TYPES.index(dt) for dt in ["subj_verb", "verb_obj", "verb_aux"]}
            mod_types = {TARGET_DEP_TYPES.index(dt) for dt in ["noun_adj", "noun_det", "noun_poss"]}
            
            arg_arg_dists = []
            mod_mod_dists = []
            arg_mod_dists = []
            
            for i, ti in enumerate(sorted(type_means_lda.keys())):
                for j, tj in enumerate(sorted(type_means_lda.keys())):
                    if i < j:
                        d = dist_matrix[i, j]
                        if ti in arg_types and tj in arg_types:
                            arg_arg_dists.append(d)
                        elif ti in mod_types and tj in mod_types:
                            mod_mod_dists.append(d)
                        else:
                            arg_mod_dists.append(d)
            
            lda_results[li] = {
                "arg_arg_dist": float(np.mean(arg_arg_dists)) if arg_arg_dists else 0,
                "mod_mod_dist": float(np.mean(mod_mod_dists)) if mod_mod_dists else 0,
                "arg_mod_dist": float(np.mean(arg_mod_dists)) if arg_mod_dists else 0,
                "cross_separation": float(np.mean(arg_mod_dists) - (np.mean(arg_arg_dists) + np.mean(mod_mod_dists)) / 2) if arg_mod_dists else 0,
            }
            print(f"  L{li:2d}: arg-arg={lda_results[li]['arg_arg_dist']:.3f} "
                  f"mod-mod={lda_results[li]['mod_mod_dist']:.3f} "
                  f"arg-mod={lda_results[li]['arg_mod_dist']:.3f} "
                  f"cross_sep={lda_results[li]['cross_separation']:.3f}")
        except Exception as e:
            print(f"  L{li:2d}: LDA failed: {e}")
    
    return {
        "chain": chain_results,
        "parallel": parallel_results,
        "lda_discriminability": lda_results,
    }


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CCXLIX: Cross-Layer Syntactic Scan")
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3, 4])
    args = parser.parse_args()
    
    print(f"Loading {args.model}...")
    model, tokenizer, device = load_model(args.model)
    model.eval()
    global _DEVICE
    _DEVICE = device
    
    t0 = time.time()
    
    if args.exp == 1:
        results = exp1_cross_layer_lda(args.model, model, tokenizer)
    elif args.exp == 2:
        results = exp2_cross_layer_direction(args.model, model, tokenizer)
    elif args.exp == 3:
        results = exp3_syntax_vs_semantic(args.model, model, tokenizer)
    elif args.exp == 4:
        results = exp4_algebraic_structure(args.model, model, tokenizer)
    
    elapsed = time.time() - t0
    results["elapsed_seconds"] = float(elapsed)
    
    # Save results
    out_dir = Path(__file__).resolve().parent.parent / "glm5_temp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"ccxlix_exp{args.exp}_{args.model}_results.json"
    
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        if isinstance(obj, set):
            return list(obj)
        return obj
    
    with open(out_file, "w") as f:
        json.dump(convert(results), f, indent=2)
    
    print(f"\nResults saved to {out_file}")
    print(f"Elapsed: {elapsed:.1f}s")
    
    # Release model
    del model
    torch.cuda.empty_cache()
    print("GPU memory released.")


if __name__ == "__main__":
    main()
