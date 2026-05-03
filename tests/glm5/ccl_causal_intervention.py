"""
CCL(250): 因果干预实验 — 从相关性到因果性的关键跳跃
=========================================================
CCXLIX发现:
  - LDA >> LinearSVC 在所有层都成立
  - 方向编码跨层稳定(0.96-0.98)
  - 但: LDA能分类 ≠ 模型使用了这些方向!
  - 需要: 沿LDA方向干预, 观察语法输出变化

核心实验:
  Exp1: LDA方向干预 — 在最佳语法层添加/减去LDA方向, 测量token概率变化
  Exp2: 依存类型交换 — 注入不同依存类型的LDA方向, 观察输出变化
  Exp3: LDA方向消融 — 投影掉LDA方向, 测试语法生成退化

关键假设:
  - 如果模型真的使用了判别子空间编码语法:
    → 沿LDA方向扰动 → 输出的语法角色应变化
    → 消融LDA方向 → 语法生成应退化
    → 这是因果性的直接证据!
"""

import torch
import numpy as np
import json
import argparse
import os
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model
from tests.glm5.ccxlix_cross_layer_scan import (
    extract_all_layers_for_pair, NATURAL_SENTENCES, TARGET_DEP_TYPES, 
    DEP_TYPE_MAP, parse_with_spacy
)

_DEVICE = "cuda:0"

# 各模型的最佳语法层(来自CCXLIX Exp1结果)
BEST_LDA_LAYERS = {
    "qwen3": 20,      # LDA=0.900
    "glm4": 39,       # LDA=0.924
    "deepseek7b": 20, # LDA=0.981
}

# 干预测试用的句子(手动构造, 确保有明确的依存结构)
INTERVENTION_SENTENCES = [
    # [句子, head词, dep词, 期望依存类型]
    # nsubj: 名词作主语
    ["The cat sat on the mat", "sat", "cat", "nsubj"],
    ["The dog ran quickly", "ran", "dog", "nsubj"],
    ["A bird flew overhead", "flew", "bird", "nsubj"],
    ["The student read the book", "read", "student", "nsubj"],
    
    # dobj: 名词作宾语
    ["She ate the apple", "ate", "apple", "dobj"],
    ["He wrote a letter", "wrote", "letter", "dobj"],
    ["They built the house", "built", "house", "dobj"],
    ["I found the key", "found", "key", "dobj"],
    
    # amod: 形容词修饰名词
    ["The red car stopped", "car", "red", "amod"],
    ["A big dog barked", "dog", "big", "amod"],
    ["The old man smiled", "man", "old", "amod"],
    ["The young girl sang", "girl", "young", "amod"],
    
    # advmod: 副词修饰动词
    ["She ran quickly home", "ran", "quickly", "advmod"],
    ["He spoke softly", "spoke", "softly", "advmod"],
    ["They worked hard", "worked", "hard", "advmod"],
    ["The bird flew slowly", "flew", "slowly", "advmod"],
]

# 用于概率测量的语法相关token
SYNTAX_TOKENS = {
    "nsubj": ["he", "she", "it", "they", "i", "we", "who", "that", "which"],
    "dobj": ["him", "her", "it", "them", "me", "us", "what", "which", "that"],
    "amod": ["very", "quite", "really", "so", "too", "more", "most", "rather"],
    "advmod": ["very", "quite", "really", "so", "too", "much", "more", "extremely"],
}


def find_token_index(tokens, word, has_bos):
    """Find the index of a word in tokenized sequence."""
    offset = 1 if has_bos else 0
    word_lower = word.lower()
    for i, t in enumerate(tokens[offset:], offset):
        t_clean = t.lower().strip("▁Ġ").strip()
        if t_clean == word_lower:
            return i
    # Fuzzy match
    for i, t in enumerate(tokens[offset:], offset):
        t_clean = t.lower().strip("▁Ġ").strip()
        if word_lower in t_clean or t_clean in word_lower:
            return i
    return None


def train_lda_at_layer(model, tokenizer, layer_idx, max_per_type=40):
    """Train LDA at a specific layer and return the fitted LDA + data."""
    print(f"\n  Training LDA at layer {layer_idx}...")
    
    pairs = parse_with_spacy(NATURAL_SENTENCES)
    type_sampled = {dt: 0 for dt in TARGET_DEP_TYPES}
    balanced_pairs = []
    for p in pairs:
        _, _, _, dt = p
        if dt in TARGET_DEP_TYPES and type_sampled[dt] < max_per_type:
            balanced_pairs.append(p)
            type_sampled[dt] += 1
    
    print(f"  Parsed {len(balanced_pairs)} balanced pairs")
    
    # Extract features
    h_heads_list = []
    h_deps_list = []
    pair_diffs_list = []
    pair_concats_list = []
    y_list = []
    
    for i, (sent, h_idx, d_idx, dep_type) in enumerate(balanced_pairs):
        if i % 50 == 0:
            print(f"    Extracting pair {i}/{len(balanced_pairs)}...")
        try:
            feats = extract_all_layers_for_pair(
                model, tokenizer, sent, h_idx, d_idx, [layer_idx]
            )
            if layer_idx in feats:
                h_heads_list.append(feats[layer_idx]["h_head"])
                h_deps_list.append(feats[layer_idx]["h_dep"])
                pair_diffs_list.append(feats[layer_idx]["pair_diff"])
                pair_concats_list.append(feats[layer_idx]["pair_concat"])
                y_list.append(TARGET_DEP_TYPES.index(dep_type))
        except:
            continue
    
    if len(y_list) < 30:
        print(f"  Too few samples: {len(y_list)}")
        return None
    
    X_concat = np.array(pair_concats_list)
    y = np.array(y_list)
    
    # Train LDA
    n_classes = len(np.unique(y))
    n_lda = min(n_classes - 1, 7)
    lda = LinearDiscriminantAnalysis(n_components=n_lda)
    lda.fit(X_concat, y)
    
    # Verify accuracy
    X_lda = lda.transform(X_concat)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0)
    cv = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=42)
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, X_lda, y, cv=cv, scoring='accuracy')
    
    print(f"  LDA trained: {n_lda} components, accuracy={scores.mean():.3f}±{scores.std():.3f}")
    print(f"  Classes: {[TARGET_DEP_TYPES[i] for i in lda.classes_]}")
    print(f"  Explained variance ratio: {lda.explained_variance_ratio_}")
    
    return {
        "lda": lda,
        "X_concat": X_concat,
        "pair_diffs": np.array(pair_diffs_list),
        "h_heads": np.array(h_heads_list),
        "h_deps": np.array(h_deps_list),
        "y": y,
        "n_samples": len(y_list),
        "lda_acc": float(scores.mean()),
    }


def intervene_at_layer(model, tokenizer, sentence, target_token_idx, layer_idx,
                       direction, alpha, device):
    """
    Add a perturbation along 'direction' to the hidden state at target_token_idx
    at the specified layer, and return the modified logits.
    
    Returns:
        base_logits: original logits
        intervened_logits: logits after intervention
    """
    toks = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = toks["input_ids"]
    n_tokens = input_ids.shape[1]
    
    # Clamp target index
    actual_target = min(target_token_idx, n_tokens - 1)
    
    layers = get_layers(model)
    
    # First, get base output
    with torch.no_grad():
        base_output = model(**toks)
        base_logits = base_output.logits.detach().float().cpu()
    
    # Now, intervene using a hook
    direction_t = torch.tensor(direction, dtype=torch.float32, device=device)
    
    intervened_logits = None
    
    def make_intervention_hook(target_pos, dir_vec, scale):
        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0].clone()
                h[0, target_pos, :] += scale * dir_vec.to(h.dtype)
                return (h,) + output[1:]
            else:
                h = output.clone()
                h[0, target_pos, :] += scale * dir_vec.to(h.dtype)
                return h
        return hook
    
    hook = layers[layer_idx].register_forward_hook(
        make_intervention_hook(actual_target, direction_t, alpha)
    )
    
    with torch.no_grad():
        intervened_output = model(**toks)
        intervened_logits = intervened_output.logits.detach().float().cpu()
    
    hook.remove()
    
    return base_logits, intervened_logits


def get_top_token_changes(base_logits, intervened_logits, target_pos, tokenizer, k=20):
    """Get the top-k tokens with largest probability change after intervention."""
    # Get probabilities at the next position after target
    next_pos = target_pos + 1
    if next_pos >= base_logits.shape[1]:
        next_pos = target_pos
    
    base_probs = torch.softmax(base_logits[0, next_pos], dim=-1).numpy()
    interv_probs = torch.softmax(intervened_logits[0, next_pos], dim=-1).numpy()
    
    prob_diff = interv_probs - base_probs
    
    # Top increases
    top_increase_idx = np.argsort(prob_diff)[-k:][::-1]
    # Top decreases
    top_decrease_idx = np.argsort(prob_diff)[:k]
    
    results = {
        "increases": [],
        "decreases": [],
        "kl_div": float(np.sum(base_probs * np.log(base_probs / (interv_probs + 1e-10) + 1e-10))),
        "total_variation": float(np.sum(np.abs(prob_diff)) / 2),
    }
    
    for idx in top_increase_idx:
        tok = tokenizer.decode([idx])
        results["increases"].append({
            "token": tok, "base_prob": float(base_probs[idx]),
            "interv_prob": float(interv_probs[idx]),
            "delta": float(prob_diff[idx]),
        })
    
    for idx in top_decrease_idx:
        tok = tokenizer.decode([idx])
        results["decreases"].append({
            "token": tok, "base_prob": float(base_probs[idx]),
            "interv_prob": float(interv_probs[idx]),
            "delta": float(prob_diff[idx]),
        })
    
    return results


# ============================================================
# Exp1: LDA方向干预 — 添加/减去LDA方向, 测量token概率变化
# ============================================================

def exp1_lda_direction_intervention(model_name, model, tokenizer, device):
    """
    For each dependency type, add/subtract the LDA discriminant direction
    and measure token probability changes.
    """
    print(f"\n{'='*70}")
    print(f"Exp1: LDA方向干预 ({model_name})")
    print(f"{'='*70}")
    
    best_layer = BEST_LDA_LAYERS.get(model_name)
    if best_layer is None:
        print(f"  No best layer for {model_name}")
        return None
    
    # Train LDA
    lda_data = train_lda_at_layer(model, tokenizer, best_layer, max_per_type=40)
    if lda_data is None:
        return None
    
    lda = lda_data["lda"]
    n_components = lda.n_components
    
    # Get LDA scalings (direction vectors in original space)
    # lda.scalings_ shape: [2*d_model, n_components] for pair_concat
    # Each column is a discriminant direction in the concatenation space
    scalings = lda.scalings_  # [2*d_model, n_components]
    d_model = scalings.shape[0] // 2
    
    # Split into head and dep components
    head_directions = scalings[:d_model, :]   # [d_model, n_components]
    dep_directions = scalings[d_model:, :]     # [d_model, n_components]
    
    # Also get class means in LDA space for interpretation
    class_means = lda.means_  # [n_classes, 2*d_model]
    class_means_head = class_means[:, :d_model]
    class_means_dep = class_means[:, d_model:]
    
    print(f"\n  LDA scalings shape: {scalings.shape}")
    print(f"  Head directions shape: {head_directions.shape}")
    print(f"  Dep directions shape: {dep_directions.shape}")
    print(f"  Classes: {[TARGET_DEP_TYPES[i] for i in lda.classes_]}")
    
    # Compute the mean direction for each dependency type (from overall mean)
    overall_mean = np.mean(class_means, axis=0)
    overall_mean_head = overall_mean[:d_model]
    overall_mean_dep = overall_mean[d_model:]
    
    # Type-specific offset from mean (in original space)
    type_directions = {}
    for i, cls in enumerate(lda.classes_):
        dt = TARGET_DEP_TYPES[cls]
        type_directions[dt] = {
            "head_offset": class_means_head[i] - overall_mean_head,
            "dep_offset": class_means_dep[i] - overall_mean_dep,
        }
    
    # Test intervention on manual sentences
    print(f"\n  === Intervention on manual sentences ===")
    
    # Intervention strengths
    alphas = [0.5, 1.0, 2.0, 5.0]
    
    results = {
        "model": model_name,
        "best_layer": best_layer,
        "n_components": n_components,
        "lda_acc": lda_data["lda_acc"],
        "alphas": alphas,
        "sentence_results": [],
    }
    
    for sent_data in INTERVENTION_SENTENCES:
        sentence, head_word, dep_word, expected_dep = sent_data
        
        toks = tokenizer(sentence, return_tensors="pt").to(device)
        input_ids = toks["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        has_bos = tokens[0] in ['<s>', '<|begin_of_text|>', '<|im_start|>', 'Ċ']
        
        # Find indices
        head_idx = find_token_index(tokens, head_word, has_bos)
        dep_idx = find_token_index(tokens, dep_word, has_bos)
        
        if head_idx is None or dep_idx is None:
            print(f"  SKIP: '{sentence}' - head={head_idx}, dep={dep_idx}")
            continue
        
        print(f"\n  Sentence: '{sentence}'")
        print(f"    head='{head_word}'(idx={head_idx}), dep='{dep_word}'(idx={dep_idx}), expected={expected_dep}")
        
        sent_result = {
            "sentence": sentence,
            "head_word": head_word,
            "dep_word": dep_word,
            "expected_dep": expected_dep,
            "head_idx": head_idx,
            "dep_idx": dep_idx,
            "interventions": [],
        }
        
        # For each intervention direction and alpha
        for dt in TARGET_DEP_TYPES:
            if dt not in type_directions:
                continue
            
            dep_offset = type_directions[dt]["dep_offset"]
            offset_norm = np.linalg.norm(dep_offset)
            if offset_norm < 1e-10:
                continue
            # Normalize
            dep_dir_normalized = dep_offset / offset_norm
            
            for alpha in alphas:
                # Intervene on dep token position
                try:
                    base_logits, interv_logits = intervene_at_layer(
                        model, tokenizer, sentence, dep_idx, best_layer,
                        dep_dir_normalized * alpha, alpha, device
                    )
                    
                    changes = get_top_token_changes(
                        base_logits, interv_logits, dep_idx, tokenizer, k=10
                    )
                    
                    # Also compute change at the head position
                    if head_idx < base_logits.shape[1]:
                        head_base_probs = torch.softmax(base_logits[0, head_idx], dim=-1).numpy()
                        head_interv_probs = torch.softmax(interv_logits[0, head_idx], dim=-1).numpy()
                        head_kl = float(np.sum(head_base_probs * np.log(
                            head_base_probs / (head_interv_probs + 1e-10) + 1e-10)))
                    else:
                        head_kl = 0.0
                    
                    interv_data = {
                        "direction": dt,
                        "alpha": alpha,
                        "kl_div": changes["kl_div"],
                        "total_variation": changes["total_variation"],
                        "head_kl": head_kl,
                        "top_increases": changes["increases"][:5],
                        "top_decreases": changes["decreases"][:5],
                    }
                    sent_result["interventions"].append(interv_data)
                    
                    print(f"    +{dt} α={alpha}: KL={changes['kl_div']:.4f}, "
                          f"TV={changes['total_variation']:.4f}, head_KL={head_kl:.4f}")
                    if changes["increases"]:
                        top_inc = changes["increases"][0]
                        print(f"      Top↑: '{top_inc['token']}' ({top_inc['base_prob']:.4f}→{top_inc['interv_prob']:.4f})")
                
                except Exception as e:
                    print(f"    ERROR +{dt} α={alpha}: {e}")
                    continue
        
        results["sentence_results"].append(sent_result)
    
    # Summary: average effect size by direction type
    print(f"\n  === Summary: Average KL by intervention direction ===")
    direction_kl = {}
    for sr in results["sentence_results"]:
        for iv in sr["interventions"]:
            dt = iv["direction"]
            if dt not in direction_kl:
                direction_kl[dt] = []
            direction_kl[dt].append(iv["kl_div"])
    
    for dt in sorted(direction_kl.keys()):
        kls = direction_kl[dt]
        mean_kl = np.mean(kls) if kls else 0
        print(f"    {dt}: mean_KL={mean_kl:.4f} (n={len(kls)})")
        results[f"mean_kl_{dt}"] = mean_kl
    
    return results


# ============================================================
# Exp2: 依存类型交换 — 注入不同类型的LDA方向
# ============================================================

def exp2_dep_type_swap(model_name, model, tokenizer, device):
    """
    For sentences with a known dependency type, inject the LDA direction
    of a DIFFERENT type and measure the output change.
    If the model uses LDA directions for syntax, swapping should cause
    detectable changes in the output distribution.
    """
    print(f"\n{'='*70}")
    print(f"Exp2: 依存类型交换干预 ({model_name})")
    print(f"{'='*70}")
    
    best_layer = BEST_LDA_LAYERS.get(model_name)
    if best_layer is None:
        return None
    
    lda_data = train_lda_at_layer(model, tokenizer, best_layer, max_per_type=40)
    if lda_data is None:
        return None
    
    lda = lda_data["lda"]
    scalings = lda.scalings_
    d_model = scalings.shape[0] // 2
    
    class_means = lda.means_
    overall_mean = np.mean(class_means, axis=0)
    overall_mean_dep = overall_mean[d_model:]
    
    type_directions = {}
    for i, cls in enumerate(lda.classes_):
        dt = TARGET_DEP_TYPES[cls]
        offset = class_means[i, d_model:] - overall_mean_dep
        norm = np.linalg.norm(offset)
        type_directions[dt] = offset / norm if norm > 1e-10 else offset
    
    alpha = 2.0  # Fixed intervention strength
    
    results = {
        "model": model_name,
        "best_layer": best_layer,
        "alpha": alpha,
        "swap_results": [],
    }
    
    # For each sentence, try swapping to all other types
    for sent_data in INTERVENTION_SENTENCES:
        sentence, head_word, dep_word, original_dep = sent_data
        
        toks = tokenizer(sentence, return_tensors="pt").to(device)
        input_ids = toks["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        has_bos = tokens[0] in ['<s>', '<|begin_of_text|>', '<|im_start|>', 'Ċ']
        
        dep_idx = find_token_index(tokens, dep_word, has_bos)
        head_idx = find_token_index(tokens, head_word, has_bos)
        
        if dep_idx is None:
            continue
        
        print(f"\n  '{sentence}' (original_dep={original_dep})")
        
        swap_result = {
            "sentence": sentence,
            "original_dep": original_dep,
            "dep_word": dep_word,
            "swaps": {},
        }
        
        # Baseline: no intervention
        try:
            with torch.no_grad():
                base_output = model(**toks)
                base_logits = base_output.logits.detach().float().cpu()
        except:
            continue
        
        # Try each swap direction
        for target_dep in TARGET_DEP_TYPES:
            if target_dep not in type_directions:
                continue
            if target_dep == original_dep:
                # Same direction = positive intervention
                direction = type_directions[target_dep]
                label = f"same({target_dep})"
            else:
                # Different direction = swap
                direction = type_directions[target_dep]
                label = f"swap→{target_dep}"
            
            try:
                base_log, interv_log = intervene_at_layer(
                    model, tokenizer, sentence, dep_idx, best_layer,
                    direction, alpha, device
                )
                
                # Measure KL at multiple positions
                n_positions = base_log.shape[1]
                kl_by_pos = {}
                for pos in range(min(n_positions, 15)):
                    base_p = torch.softmax(base_log[0, pos], dim=-1).numpy()
                    interv_p = torch.softmax(interv_log[0, pos], dim=-1).numpy()
                    kl = float(np.sum(base_p * np.log(base_p / (interv_p + 1e-10) + 1e-10)))
                    kl_by_pos[pos] = kl
                
                swap_result["swaps"][target_dep] = {
                    "kl_by_pos": kl_by_pos,
                    "mean_kl": float(np.mean(list(kl_by_pos.values()))),
                    "max_kl": float(max(kl_by_pos.values())),
                    "is_same": target_dep == original_dep,
                }
                
                mean_kl = np.mean(list(kl_by_pos.values()))
                is_same = target_dep == original_dep
                marker = "★" if is_same else " "
                print(f"    {marker} {target_dep}: mean_KL={mean_kl:.4f}, max_KL={max(kl_by_pos.values()):.4f}")
            
            except Exception as e:
                print(f"    ERROR swap→{target_dep}: {e}")
                continue
        
        results["swap_results"].append(swap_result)
    
    # Compute key metric: is same-type KL > cross-type KL?
    same_kls = []
    cross_kls = []
    for sr in results["swap_results"]:
        for dt, sw in sr["swaps"].items():
            if sw["is_same"]:
                same_kls.append(sw["mean_kl"])
            else:
                cross_kls.append(sw["mean_kl"])
    
    if same_kls and cross_kls:
        results["same_type_mean_kl"] = float(np.mean(same_kls))
        results["cross_type_mean_kl"] = float(np.mean(cross_kls))
        results["same_vs_cross_ratio"] = float(np.mean(same_kls) / max(np.mean(cross_kls), 1e-10))
        print(f"\n  ★ Same-type mean KL: {np.mean(same_kls):.4f}")
        print(f"  ★ Cross-type mean KL: {np.mean(cross_kls):.4f}")
        print(f"  ★ Same/Cross ratio: {np.mean(same_kls)/max(np.mean(cross_kls),1e-10):.3f}")
    
    return results


# ============================================================
# Exp3: LDA方向消融 — 投影掉LDA方向, 测试语法生成退化
# ============================================================

def exp3_lda_ablation(model_name, model, tokenizer, device):
    """
    Project out the LDA discriminant directions from the hidden state
    and measure how much the next-token prediction degrades for
    syntax-relevant predictions.
    """
    print(f"\n{'='*70}")
    print(f"Exp3: LDA方向消融 ({model_name})")
    print(f"{'='*70}")
    
    best_layer = BEST_LDA_LAYERS.get(model_name)
    if best_layer is None:
        return None
    
    lda_data = train_lda_at_layer(model, tokenizer, best_layer, max_per_type=40)
    if lda_data is None:
        return None
    
    lda = lda_data["lda"]
    scalings = lda.scalings_
    d_model = scalings.shape[0] // 2
    
    # Projector: remove LDA subspace from dep token
    # P_orth = I - U @ U^T where U is orthonormal basis of LDA subspace
    dep_dirs = scalings[d_model:, :]  # [d_model, n_components]
    
    # Orthonormalize
    from scipy.linalg import orth
    U = orth(dep_dirs)  # [d_model, r] where r = rank of dep_dirs
    P_orth = np.eye(d_model) - U @ U.T  # Projection matrix to remove LDA subspace
    
    print(f"  LDA dep subspace dimension: {U.shape[1]}")
    print(f"  Projection matrix shape: {P_orth.shape}")
    
    results = {
        "model": model_name,
        "best_layer": best_layer,
        "lda_subspace_dim": U.shape[1],
        "ablation_results": [],
    }
    
    for sent_data in INTERVENTION_SENTENCES:
        sentence, head_word, dep_word, expected_dep = sent_data
        
        toks = tokenizer(sentence, return_tensors="pt").to(device)
        input_ids = toks["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        has_bos = tokens[0] in ['<s>', '<|begin_of_text|>', '<|im_start|>', 'Ċ']
        
        dep_idx = find_token_index(tokens, dep_word, has_bos)
        
        if dep_idx is None:
            continue
        
        print(f"\n  '{sentence}' (dep_idx={dep_idx})")
        
        layers = get_layers(model)
        
        # Get base output
        with torch.no_grad():
            base_output = model(**toks)
            base_logits = base_output.logits.detach().float().cpu()
        
        # Ablate: project out LDA subspace at dep position
        P_orth_t = torch.tensor(P_orth, dtype=torch.float32, device=device)
        
        def make_ablation_hook(target_pos, proj_matrix):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0].clone()
                    # Project out: h = P_orth @ h
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
        
        hook = layers[best_layer].register_forward_hook(
            make_ablation_hook(dep_idx, P_orth_t)
        )
        
        with torch.no_grad():
            ablated_output = model(**toks)
            ablated_logits = ablated_output.logits.detach().float().cpu()
        
        hook.remove()
        
        # Measure degradation
        # 1. Perplexity change
        base_probs_last = torch.softmax(base_logits[0, -1], dim=-1).numpy()
        ablated_probs_last = torch.softmax(ablated_logits[0, -1], dim=-1).numpy()
        
        # 2. KL divergence at each position
        n_positions = base_logits.shape[1]
        kl_by_pos = {}
        for pos in range(min(n_positions, 15)):
            base_p = torch.softmax(base_logits[0, pos], dim=-1).numpy()
            ablated_p = torch.softmax(ablated_logits[0, pos], dim=-1).numpy()
            kl = float(np.sum(base_p * np.log(base_p / (ablated_p + 1e-10) + 1e-10)))
            kl_by_pos[pos] = kl
        
        # 3. Top-1 token change
        base_top1 = torch.argmax(base_logits[0, -1]).item()
        ablated_top1 = torch.argmax(ablated_logits[0, -1]).item()
        top1_changed = base_top1 != ablated_top1
        
        # 4. Probability of the original top-1 token after ablation
        orig_top1_prob_base = float(base_probs_last[base_top1])
        orig_top1_prob_ablated = float(ablated_probs_last[base_top1])
        prob_drop = orig_top1_prob_base - orig_top1_prob_ablated
        
        ablation_result = {
            "sentence": sentence,
            "dep_word": dep_word,
            "expected_dep": expected_dep,
            "kl_by_pos": kl_by_pos,
            "mean_kl": float(np.mean(list(kl_by_pos.values()))),
            "max_kl": float(max(kl_by_pos.values())),
            "top1_changed": top1_changed,
            "orig_top1_prob_base": orig_top1_prob_base,
            "orig_top1_prob_ablated": orig_top1_prob_ablated,
            "prob_drop": prob_drop,
        }
        
        results["ablation_results"].append(ablation_result)
        
        base_tok = tokenizer.decode([base_top1])
        ablated_tok = tokenizer.decode([ablated_top1])
        print(f"    KL={np.mean(list(kl_by_pos.values())):.4f}, "
              f"top1: '{base_tok}'({orig_top1_prob_base:.3f}) → "
              f"'{ablated_tok}'({orig_top1_prob_ablated:.3f}), "
              f"drop={prob_drop:.3f}")
    
    # Summary
    if results["ablation_results"]:
        mean_kl = np.mean([ar["mean_kl"] for ar in results["ablation_results"]])
        mean_prob_drop = np.mean([ar["prob_drop"] for ar in results["ablation_results"]])
        n_top1_changed = sum(1 for ar in results["ablation_results"] if ar["top1_changed"])
        
        results["overall_mean_kl"] = float(mean_kl)
        results["overall_mean_prob_drop"] = float(mean_prob_drop)
        results["n_top1_changed"] = n_top1_changed
        results["n_sentences"] = len(results["ablation_results"])
        
        print(f"\n  ★ Overall ablation effect:")
        print(f"    Mean KL: {mean_kl:.4f}")
        print(f"    Mean prob drop: {mean_prob_drop:.4f}")
        print(f"    Top-1 changed: {n_top1_changed}/{len(results['ablation_results'])}")
    
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
    print(f"CCL(250): 因果干预实验")
    print(f"Model: {model_name}, Exp: {exp_num}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"  Model: {model_info.model_class}, layers={model_info.n_layers}, d_model={model_info.d_model}")
    
    try:
        if exp_num == 1:
            result = exp1_lda_direction_intervention(model_name, model, tokenizer, device)
        elif exp_num == 2:
            result = exp2_dep_type_swap(model_name, model, tokenizer, device)
        elif exp_num == 3:
            result = exp3_lda_ablation(model_name, model, tokenizer, device)
        
        if result is not None:
            out_path = f"tests/glm5_temp/ccl_exp{exp_num}_{model_name}_results.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n  Results saved to {out_path}")
    
    finally:
        release_model(model)
    
    print(f"\nDone!")


if __name__ == "__main__":
    main()
