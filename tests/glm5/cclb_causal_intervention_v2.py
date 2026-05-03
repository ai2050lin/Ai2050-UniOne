"""
CCL-B(250.2): 因果干预实验V2 — 最后层干预 + 大扰动
=========================================================
CCL-V1发现:
  - 在L20(中间层)干预: KL仅0.001-0.003, 几乎无效
  - 消融LDA方向: Mean KL=0.0015, top-1改变0/16
  - 原因: 中间层干预被后续16层"修复"

关键改进:
  1. 干预最后一层 → 没有后续层可以修复!
  2. 同时干预head和dep两个位置
  3. 使用更大alpha (5, 10, 20, 50)
  4. 使用pair_diff方向而不只是class offset
  5. 直接测量: 干预后模型是否改变对语法角色的预测
"""

import torch
import numpy as np
import json
import argparse
import os
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model
from tests.glm5.ccxlix_cross_layer_scan import (
    extract_all_layers_for_pair, NATURAL_SENTENCES, TARGET_DEP_TYPES, 
    DEP_TYPE_MAP, parse_with_spacy
)

_DEVICE = "cuda:0"

# 干预测试句子
INTERVENTION_SENTENCES = [
    # nsubj sentences
    ["The cat sat on the mat", "sat", "cat", "nsubj"],
    ["The dog ran quickly", "ran", "dog", "nsubj"],
    ["The student read the book", "read", "student", "nsubj"],
    ["A bird flew overhead", "flew", "bird", "nsubj"],
    
    # dobj sentences
    ["She ate the apple", "ate", "apple", "dobj"],
    ["He wrote a letter", "wrote", "letter", "dobj"],
    ["They built the house", "built", "house", "dobj"],
    ["I found the key", "found", "key", "dobj"],
    
    # amod sentences
    ["The red car stopped", "car", "red", "amod"],
    ["A big dog barked", "dog", "big", "amod"],
    ["The old man smiled", "man", "old", "amod"],
    ["The young girl sang", "girl", "young", "amod"],
    
    # advmod sentences
    ["She ran quickly home", "ran", "quickly", "advmod"],
    ["He spoke softly", "spoke", "softly", "advmod"],
    ["They worked hard", "worked", "hard", "advmod"],
    ["The bird flew slowly", "flew", "slowly", "advmod"],
]


def safe_decode(tokenizer, token_id):
    """Safely decode a token ID, handling tokenizer quirks."""
    try:
        result = tokenizer.decode([token_id])
        if result is None:
            return f"<tok_{token_id}>"
        return result
    except Exception:
        return f"<tok_{token_id}>"


def find_token_index(tokens, word, has_bos):
    offset = 1 if has_bos else 0
    word_lower = word.lower()
    for i, t in enumerate(tokens[offset:], offset):
        t_clean = t.lower().strip("▁Ġ").strip()
        if t_clean == word_lower:
            return i
    for i, t in enumerate(tokens[offset:], offset):
        t_clean = t.lower().strip("▁Ġ").strip()
        if word_lower in t_clean or t_clean in word_lower:
            return i
    return None


def intervene_last_layer(model, tokenizer, sentence, positions_dirs, alpha, device):
    """
    Intervene at the LAST layer by adding directions to multiple positions.
    
    Args:
        positions_dirs: dict {pos_idx: direction_np_array}
        alpha: intervention strength
    
    Returns:
        base_logits, intervened_logits
    """
    toks = tokenizer(sentence, return_tensors="pt").to(device)
    
    layers = get_layers(model)
    last_layer = layers[-1]
    
    # Get base output
    with torch.no_grad():
        base_output = model(**toks)
        base_logits = base_output.logits.detach().float().cpu()
    
    # Intervene at last layer
    def make_multi_pos_hook(pos_dirs, scale):
        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0].clone()
                for pos, direction in pos_dirs.items():
                    if pos < h.shape[1]:
                        dir_t = torch.tensor(direction, dtype=torch.float32, device=device)
                        h[0, pos, :] += (scale * dir_t).to(h.dtype)
                return (h,) + output[1:]
            else:
                h = output.clone()
                for pos, direction in pos_dirs.items():
                    if pos < h.shape[1]:
                        dir_t = torch.tensor(direction, dtype=torch.float32, device=device)
                        h[0, pos, :] += (scale * dir_t).to(h.dtype)
                return h
        return hook
    
    hook = last_layer.register_forward_hook(
        make_multi_pos_hook(positions_dirs, alpha)
    )
    
    with torch.no_grad():
        interv_output = model(**toks)
        interv_logits = interv_output.logits.detach().float().cpu()
    
    hook.remove()
    
    return base_logits, interv_logits


def compute_full_kl_analysis(base_logits, interv_logits, tokenizer, 
                              positions_of_interest=None):
    """Comprehensive KL analysis at all positions."""
    n_positions = base_logits.shape[1]
    
    results = {
        "per_pos_kl": {},
        "total_kl": 0,
        "max_kl_pos": 0,
        "max_kl_val": 0,
        "top_token_changes": {},
    }
    
    for pos in range(n_positions):
        base_p = torch.softmax(base_logits[0, pos], dim=-1).numpy()
        interv_p = torch.softmax(interv_logits[0, pos], dim=-1).numpy()
        
        kl = float(np.sum(base_p * np.log(base_p / (interv_p + 1e-10) + 1e-10)))
        results["per_pos_kl"][str(pos)] = kl
        results["total_kl"] += kl
        
        if kl > results["max_kl_val"]:
            results["max_kl_val"] = kl
            results["max_kl_pos"] = pos
        
        # Top changes at key positions
        if positions_of_interest and pos in positions_of_interest:
            prob_diff = interv_p - base_p
            top_inc = np.argsort(prob_diff)[-5:][::-1]
            
            changes = []
            for idx in top_inc:
                tok_str = safe_decode(tokenizer, idx)
                changes.append({
                    "token": tok_str,
                    "base_prob": float(base_p[idx]),
                    "interv_prob": float(interv_p[idx]),
                    "delta": float(prob_diff[idx]),
                })
            results["top_token_changes"][str(pos)] = changes
    
    results["mean_kl"] = results["total_kl"] / n_positions
    return results


# ============================================================
# Exp1: 最后层干预 — 大alpha + 多位置
# ============================================================

def exp1_last_layer_intervention(model_name, model, tokenizer, device):
    """
    Intervene at the LAST layer with large alpha values.
    Use pair_diff directions computed from the training data.
    """
    print(f"\n{'='*70}")
    print(f"Exp1-V2: 最后层大扰动干预 ({model_name})")
    print(f"{'='*70}")
    
    layers = get_layers(model)
    last_layer_idx = len(layers) - 1
    d_model = model.get_input_embeddings().weight.shape[1]
    
    # Step 1: Compute mean pair_diff for each dependency type at last layer
    print(f"\n  Step 1: Computing mean directions at last layer (L{last_layer_idx})...")
    
    pairs = parse_with_spacy(NATURAL_SENTENCES)
    type_sampled = {dt: 0 for dt in TARGET_DEP_TYPES}
    balanced_pairs = []
    for p in pairs:
        _, _, _, dt = p
        if dt in TARGET_DEP_TYPES and type_sampled[dt] < 50:
            balanced_pairs.append(p)
            type_sampled[dt] += 1
    
    print(f"  Using {len(balanced_pairs)} pairs")
    
    # Extract features at last layer
    type_pair_diffs = {dt: [] for dt in TARGET_DEP_TYPES}
    type_h_heads = {dt: [] for dt in TARGET_DEP_TYPES}
    type_h_deps = {dt: [] for dt in TARGET_DEP_TYPES}
    
    for i, (sent, h_idx, d_idx, dep_type) in enumerate(balanced_pairs):
        if i % 50 == 0:
            print(f"    Extracting {i}/{len(balanced_pairs)}...")
        try:
            feats = extract_all_layers_for_pair(
                model, tokenizer, sent, h_idx, d_idx, [last_layer_idx]
            )
            if last_layer_idx in feats:
                if dep_type in type_pair_diffs:
                    type_pair_diffs[dep_type].append(feats[last_layer_idx]["pair_diff"])
                    type_h_heads[dep_type].append(feats[last_layer_idx]["h_head"])
                    type_h_deps[dep_type].append(feats[last_layer_idx]["h_dep"])
        except:
            continue
    
    # Compute mean directions
    type_mean_diff = {}
    type_mean_dep = {}
    overall_mean_dep = []
    
    for dt in TARGET_DEP_TYPES:
        if len(type_pair_diffs[dt]) >= 5:
            type_mean_diff[dt] = np.mean(type_pair_diffs[dt], axis=0)
            type_mean_dep[dt] = np.mean(type_h_deps[dt], axis=0)
            overall_mean_dep.extend(type_h_deps[dt])
            print(f"    {dt}: n={len(type_pair_diffs[dt])}, "
                  f"||mean_diff||={np.linalg.norm(type_mean_diff[dt]):.2f}")
    
    overall_mean_dep = np.mean(overall_mean_dep, axis=0) if overall_mean_dep else np.zeros(d_model)
    
    # Type-specific offsets (from overall mean)
    type_dep_offsets = {}
    for dt in type_mean_dep:
        offset = type_mean_dep[dt] - overall_mean_dep
        norm = np.linalg.norm(offset)
        if norm > 1e-10:
            type_dep_offsets[dt] = offset / norm  # Unit direction
            print(f"    {dt} offset: ||offset||={norm:.2f}, normalized")
    
    # Step 2: Intervene on test sentences
    print(f"\n  Step 2: Last-layer intervention on test sentences...")
    
    alphas = [5, 10, 20, 50]
    
    results = {
        "model": model_name,
        "last_layer": last_layer_idx,
        "alphas": alphas,
        "type_dep_offsets_norms": {dt: float(np.linalg.norm(type_mean_dep.get(dt, np.zeros(d_model)) - overall_mean_dep)) 
                                   for dt in type_dep_offsets},
        "sentence_results": [],
    }
    
    for sent_data in INTERVENTION_SENTENCES:
        sentence, head_word, dep_word, expected_dep = sent_data
        
        toks = tokenizer(sentence, return_tensors="pt").to(device)
        input_ids = toks["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        has_bos = tokens[0] in ['<s>', '<|begin_of_text|>', '<|im_start|>', 'Ċ']
        
        head_idx = find_token_index(tokens, head_word, has_bos)
        dep_idx = find_token_index(tokens, dep_word, has_bos)
        
        if head_idx is None or dep_idx is None:
            print(f"  SKIP: '{sentence}'")
            continue
        
        print(f"\n  '{sentence}' head={head_idx}({head_word}), dep={dep_idx}({dep_word}), type={expected_dep}")
        
        sent_result = {
            "sentence": sentence,
            "head_word": head_word,
            "dep_word": dep_word,
            "expected_dep": expected_dep,
            "head_idx": head_idx,
            "dep_idx": dep_idx,
            "interventions": [],
        }
        
        # Get base output
        with torch.no_grad():
            base_output = model(**toks)
            base_logits = base_output.logits.detach().float().cpu()
        
        # Try each type's direction at each alpha
        for dt in type_dep_offsets:
            direction = type_dep_offsets[dt]
            
            for alpha in alphas:
                # Intervene at dep position only
                target_positions = {dep_idx: direction}
                
                try:
                    base_log, interv_log = intervene_last_layer(
                        model, tokenizer, sentence, 
                        {dep_idx: direction}, alpha, device
                    )
                    
                    analysis = compute_full_kl_analysis(
                        base_log, interv_log, tokenizer,
                        positions_of_interest={dep_idx, head_idx, len(tokens)-1}
                    )
                    
                    interv_data = {
                        "direction": dt,
                        "alpha": alpha,
                        "total_kl": analysis["total_kl"],
                        "mean_kl": analysis["mean_kl"],
                        "max_kl_pos": analysis["max_kl_pos"],
                        "max_kl_val": analysis["max_kl_val"],
                        "dep_pos_kl": analysis["per_pos_kl"].get(str(dep_idx), 0),
                        "last_pos_kl": analysis["per_pos_kl"].get(str(len(tokens)-1), 0),
                    }
                    
                    # Add top token changes at dep and last position
                    for pos_key in [dep_idx, len(tokens)-1]:
                        if str(pos_key) in analysis["top_token_changes"]:
                            changes = analysis["top_token_changes"][str(pos_key)]
                            interv_data[f"top_changes_pos{pos_key}"] = changes[:3]
                    
                    sent_result["interventions"].append(interv_data)
                    
                    is_same = dt == expected_dep
                    marker = "★" if is_same else " "
                    print(f"    {marker} +{dt} α={alpha}: total_KL={analysis['total_kl']:.4f}, "
                          f"max_KL@{analysis['max_kl_pos']}={analysis['max_kl_val']:.4f}")
                
                except Exception as e:
                    import traceback
                    print(f"    ERROR +{dt} α={alpha}: {e}")
                    traceback.print_exc()
                    continue
        
        results["sentence_results"].append(sent_result)
    
    # Summary
    print(f"\n  === Summary ===")
    same_total_kl = []
    cross_total_kl = []
    
    for sr in results["sentence_results"]:
        for iv in sr["interventions"]:
            if iv["direction"] == sr["expected_dep"]:
                same_total_kl.append(iv["total_kl"])
            else:
                cross_total_kl.append(iv["total_kl"])
    
    if same_total_kl:
        results["same_type_mean_total_kl"] = float(np.mean(same_total_kl))
        results["cross_type_mean_total_kl"] = float(np.mean(cross_total_kl)) if cross_total_kl else 0
        print(f"  Same-type mean total KL: {np.mean(same_total_kl):.4f}")
        print(f"  Cross-type mean total KL: {np.mean(cross_total_kl):.4f}" if cross_total_kl else "")
    
    # By alpha
    for alpha in alphas:
        alpha_kls = [iv["total_kl"] for sr in results["sentence_results"] 
                     for iv in sr["interventions"] if iv["alpha"] == alpha]
        if alpha_kls:
            results[f"alpha_{alpha}_mean_kl"] = float(np.mean(alpha_kls))
            print(f"  α={alpha}: mean total KL = {np.mean(alpha_kls):.4f}")
    
    return results


# ============================================================
# Exp2: 最后层消融 — 投影掉类型特定方向
# ============================================================

def exp2_last_layer_ablation(model_name, model, tokenizer, device):
    """
    Ablate type-specific directions at the LAST layer.
    """
    print(f"\n{'='*70}")
    print(f"Exp2-V2: 最后层消融 ({model_name})")
    print(f"{'='*70}")
    
    layers = get_layers(model)
    last_layer_idx = len(layers) - 1
    d_model = model.get_input_embeddings().weight.shape[1]
    
    # Compute mean directions (same as Exp1)
    print(f"\n  Computing mean directions at L{last_layer_idx}...")
    
    pairs = parse_with_spacy(NATURAL_SENTENCES)
    type_sampled = {dt: 0 for dt in TARGET_DEP_TYPES}
    balanced_pairs = []
    for p in pairs:
        _, _, _, dt = p
        if dt in TARGET_DEP_TYPES and type_sampled[dt] < 50:
            balanced_pairs.append(p)
            type_sampled[dt] += 1
    
    type_h_deps = {dt: [] for dt in TARGET_DEP_TYPES}
    for i, (sent, h_idx, d_idx, dep_type) in enumerate(balanced_pairs):
        if i % 50 == 0:
            print(f"    Extracting {i}/{len(balanced_pairs)}...")
        try:
            feats = extract_all_layers_for_pair(
                model, tokenizer, sent, h_idx, d_idx, [last_layer_idx]
            )
            if last_layer_idx in feats and dep_type in type_h_deps:
                type_h_deps[dep_type].append(feats[last_layer_idx]["h_dep"])
        except:
            continue
    
    # Compute type-specific offsets and overall PCA subspace
    all_deps = []
    for dt in TARGET_DEP_TYPES:
        all_deps.extend(type_h_deps[dt])
    
    if len(all_deps) < 50:
        print("  Too few samples!")
        return None
    
    all_deps = np.array(all_deps)
    overall_mean = np.mean(all_deps, axis=0)
    
    # Use PCA to find the major directions in dep representation space
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(20, all_deps.shape[1], all_deps.shape[0]-1))
    centered = all_deps - overall_mean
    pca.fit(centered)
    
    print(f"  PCA top-20 explained variance: {pca.explained_variance_ratio_[:20]}")
    print(f"  Cumulative: {np.cumsum(pca.explained_variance_ratio_[:20])}")
    
    # Also compute LDA directions
    X_concats = []
    y_list = []
    type_h_heads = {dt: [] for dt in TARGET_DEP_TYPES}
    
    for i, (sent, h_idx, d_idx, dep_type) in enumerate(balanced_pairs):
        try:
            feats = extract_all_layers_for_pair(
                model, tokenizer, sent, h_idx, d_idx, [last_layer_idx]
            )
            if last_layer_idx in feats and dep_type in TARGET_DEP_TYPES:
                X_concats.append(feats[last_layer_idx]["pair_concat"])
                y_list.append(TARGET_DEP_TYPES.index(dep_type))
        except:
            continue
    
    if len(y_list) < 50:
        print("  Too few for LDA!")
        return None
    
    X_concat = np.array(X_concats)
    y = np.array(y_list)
    n_classes = len(np.unique(y))
    
    lda = LinearDiscriminantAnalysis(n_components=min(n_classes-1, 7))
    lda.fit(X_concat, y)
    
    dep_scalings = lda.scalings_[d_model:, :]  # dep part
    from scipy.linalg import orth
    U_lda = orth(dep_scalings)  # Orthonormal basis of LDA subspace
    
    # Projector to remove LDA subspace
    P_orth_lda = np.eye(d_model) - U_lda @ U_lda.T
    print(f"  LDA subspace dimension: {U_lda.shape[1]}")
    
    # Projector to remove PCA top-k subspace
    U_pca = pca.components_[:7].T  # Top 7 PCA directions (same dim as LDA)
    P_orth_pca = np.eye(d_model) - U_pca @ U_pca.T
    
    results = {
        "model": model_name,
        "last_layer": last_layer_idx,
        "lda_subspace_dim": U_lda.shape[1],
        "pca_top7_explained_var": float(np.sum(pca.explained_variance_ratio_[:7])),
        "ablation_results": [],
    }
    
    # Test ablation on sentences
    for sent_data in INTERVENTION_SENTENCES:
        sentence, head_word, dep_word, expected_dep = sent_data
        
        toks = tokenizer(sentence, return_tensors="pt").to(device)
        input_ids = toks["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        has_bos = tokens[0] in ['<s>', '<|begin_of_text|>', '<|im_start|>', 'Ċ']
        
        dep_idx = find_token_index(tokens, dep_word, has_bos)
        
        if dep_idx is None:
            continue
        
        # Base output
        with torch.no_grad():
            base_output = model(**toks)
            base_logits = base_output.logits.detach().float().cpu()
        
        # LDA ablation
        P_orth_lda_t = torch.tensor(P_orth_lda, dtype=torch.float32, device=device)
        
        def make_ablation_hook(target_pos, proj_matrix):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0].clone()
                    h_original = h[0, target_pos, :].float()
                    h_ablated = proj_matrix @ h_original
                    h[0, target_pos, :] = h_ablated.to(h.dtype)
                    return (h,) + output[1:]
                else:
                    h = output.clone()
                    h_original = h[0, target_pos, :].float()
                    h_ablated = proj_matrix @ h_original
                    h[0, target_pos, :] = h_ablated.to(h.dtype)
                    return h
            return hook
        
        # LDA ablation
        hook = layers[-1].register_forward_hook(make_ablation_hook(dep_idx, P_orth_lda_t))
        with torch.no_grad():
            lda_ablated_output = model(**toks)
            lda_ablated_logits = lda_ablated_output.logits.detach().float().cpu()
        hook.remove()
        
        # PCA ablation (control: remove same-dimensional PCA subspace)
        P_orth_pca_t = torch.tensor(P_orth_pca, dtype=torch.float32, device=device)
        hook = layers[-1].register_forward_hook(make_ablation_hook(dep_idx, P_orth_pca_t))
        with torch.no_grad():
            pca_ablated_output = model(**toks)
            pca_ablated_logits = pca_ablated_output.logits.detach().float().cpu()
        hook.remove()
        
        # Compare
        n_positions = base_logits.shape[1]
        
        def compute_mean_kl(base_l, interv_l):
            total = 0
            for pos in range(min(n_positions, 20)):
                bp = torch.softmax(base_l[0, pos], dim=-1).numpy()
                ip = torch.softmax(interv_l[0, pos], dim=-1).numpy()
                total += float(np.sum(bp * np.log(bp / (ip + 1e-10) + 1e-10)))
            return total / min(n_positions, 20)
        
        lda_kl = compute_mean_kl(base_logits, lda_ablated_logits)
        pca_kl = compute_mean_kl(base_logits, pca_ablated_logits)
        
        # Top-1 change
        base_top1 = torch.argmax(base_logits[0, -1]).item()
        lda_top1 = torch.argmax(lda_ablated_logits[0, -1]).item()
        pca_top1 = torch.argmax(pca_ablated_logits[0, -1]).item()
        
        ablation_result = {
            "sentence": sentence,
            "dep_word": dep_word,
            "expected_dep": expected_dep,
            "lda_ablation_kl": lda_kl,
            "pca_ablation_kl": pca_kl,
            "lda_vs_pca_ratio": lda_kl / max(pca_kl, 1e-10),
            "base_top1": safe_decode(tokenizer, base_top1),
            "lda_top1": safe_decode(tokenizer, lda_top1),
            "pca_top1": safe_decode(tokenizer, pca_top1),
            "lda_changed_top1": base_top1 != lda_top1,
            "pca_changed_top1": base_top1 != pca_top1,
        }
        
        results["ablation_results"].append(ablation_result)
        print(f"  '{sentence}': LDA_KL={lda_kl:.4f}, PCA_KL={pca_kl:.4f}, "
              f"ratio={lda_kl/max(pca_kl,1e-10):.2f}, "
              f"top1_change: LDA={base_top1!=lda_top1}, PCA={base_top1!=pca_top1}")
    
    # Summary
    if results["ablation_results"]:
        mean_lda_kl = np.mean([ar["lda_ablation_kl"] for ar in results["ablation_results"]])
        mean_pca_kl = np.mean([ar["pca_ablation_kl"] for ar in results["ablation_results"]])
        n_lda_changed = sum(1 for ar in results["ablation_results"] if ar["lda_changed_top1"])
        n_pca_changed = sum(1 for ar in results["ablation_results"] if ar["pca_changed_top1"])
        
        results["overall_lda_kl"] = float(mean_lda_kl)
        results["overall_pca_kl"] = float(mean_pca_kl)
        results["lda_pca_ratio"] = float(mean_lda_kl / max(mean_pca_kl, 1e-10))
        results["n_lda_top1_changed"] = n_lda_changed
        results["n_pca_top1_changed"] = n_pca_changed
        
        print(f"\n  ★ Summary:")
        print(f"    LDA ablation mean KL: {mean_lda_kl:.4f}")
        print(f"    PCA ablation mean KL: {mean_pca_kl:.4f}")
        print(f"    LDA/PCA ratio: {mean_lda_kl/max(mean_pca_kl,1e-10):.2f}")
        print(f"    Top-1 changed: LDA={n_lda_changed}, PCA={n_pca_changed}")
    
    return results


# ============================================================
# Exp3: 表示空间距离分析 — 干预效果的理论预测
# ============================================================

def exp3_representation_distance(model_name, model, tokenizer, device):
    """
    Measure how much representation space changes per layer
    to understand why middle-layer interventions fail.
    """
    print(f"\n{'='*70}")
    print(f"Exp3-V2: 表示空间逐层变化量 ({model_name})")
    print(f"{'='*70}")
    
    layers = get_layers(model)
    n_layers = len(layers)
    d_model = model.get_input_embeddings().weight.shape[1]
    
    # Sample layers
    if n_layers <= 20:
        sample_layers = list(range(n_layers))
    else:
        sample_layers = sorted(set([0, 1] + list(range(0, n_layers, 3)) + [n_layers-2, n_layers-1]))
    
    # Use a few sentences
    test_sentences = [
        "The cat sat on the mat",
        "She ate the apple",
        "The red car stopped",
    ]
    
    results = {
        "model": model_name,
        "n_layers": n_layers,
        "per_sentence": [],
    }
    
    for sentence in test_sentences:
        print(f"\n  '{sentence}'")
        
        toks = tokenizer(sentence, return_tensors="pt").to(device)
        input_ids = toks["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        n_tokens = len(tokens)
        
        # Get representations at all sample layers
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
            hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))
        
        with torch.no_grad():
            _ = model(**toks)
        
        for h in hooks:
            h.remove()
        
        # Compute layer-to-layer changes
        sent_result = {
            "sentence": sentence,
            "n_tokens": n_tokens,
            "layer_changes": {},
        }
        
        prev_repr = None
        for li in sample_layers:
            key = f"L{li}"
            if key not in captured:
                continue
            
            repr = captured[key][0].numpy()  # [n_tokens, d_model]
            
            if prev_repr is not None:
                # Per-token change
                changes = []
                for t in range(n_tokens):
                    delta = repr[t] - prev_repr[t]
                    changes.append(float(np.linalg.norm(delta)))
                
                mean_change = float(np.mean(changes))
                sent_result["layer_changes"][li] = {
                    "mean_norm_change": mean_change,
                    "per_token": changes,
                }
                print(f"    L{li}: mean ||Δh|| = {mean_change:.2f}")
            
            prev_repr = repr
        
        results["per_sentence"].append(sent_result)
    
    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, 
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True,
                       choices=[1, 2, 3])
    args = parser.parse_args()
    
    model_name = args.model
    exp_num = args.exp
    
    print(f"\n{'='*70}")
    print(f"CCL-B(250.2): 因果干预实验V2")
    print(f"Model: {model_name}, Exp: {exp_num}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"  Model: {model_info.model_class}, layers={model_info.n_layers}, d_model={model_info.d_model}")
    
    try:
        if exp_num == 1:
            result = exp1_last_layer_intervention(model_name, model, tokenizer, device)
        elif exp_num == 2:
            result = exp2_last_layer_ablation(model_name, model, tokenizer, device)
        elif exp_num == 3:
            result = exp3_representation_distance(model_name, model, tokenizer, device)
        
        if result is not None:
            out_path = f"tests/glm5_temp/cclb_exp{exp_num}_{model_name}_results.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n  Results saved to {out_path}")
    
    finally:
        release_model(model)
    
    print(f"\nDone!")


if __name__ == "__main__":
    main()
