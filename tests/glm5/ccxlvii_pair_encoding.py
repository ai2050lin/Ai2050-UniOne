"""
CCXLVII(407): Pair-based关系编码验证
=====================================
核心假说: 句法关系编码在token对之间, 而非单个token中

对比:
  Point-based: 语法角色 = f(h_token)  — 单token分类
  Pair-based:  依存关系 = φ(h_i, h_j) — token对分类

构造 pair 表示:
  φ(i,j) = [h_i; h_j; h_i - h_j; h_i ⊙ h_j]

实验:
  Exp1: pair-based依存方向分类 vs point-based
  Exp2: pair-based依存类型分类 (subj→verb, verb→obj, etc.)
  Exp3: pair-based自然句验证
  Exp4: 注意力权重 vs 依存关系相关性
  Exp5: 消融: 哪个pair组件(h_i, h_j, h_i-h_j, h_i⊙h_j)最重要?
"""

import torch
import numpy as np
import json
import argparse
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine
from pathlib import Path

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tests.glm5.model_utils import load_model

# Global device variable (set in main)
_DEVICE = "cuda:0"

# ============================================================
# 数据构造
# ============================================================

# 简单模板句 - 6种依存类型 (每类4对)
TEMPLATE_PAIRS = [
    # (sentence, head_idx, dep_idx, dep_type)
    # subject → verb
    ("The cat chases the mouse", 1, 2, "subj_verb"),
    ("The dog runs quickly", 1, 2, "subj_verb"),
    ("A bird sings beautifully", 1, 2, "subj_verb"),
    ("The student reads the book", 1, 2, "subj_verb"),
    # verb → object
    ("The cat chases the mouse", 2, 4, "verb_obj"),
    ("The student reads the book", 2, 4, "verb_obj"),
    ("The man eats the food", 2, 4, "verb_obj"),
    ("The girl finds the key", 2, 4, "verb_obj"),
    # noun → modifier (adjective)
    ("The big cat sleeps", 2, 1, "noun_adj"),
    ("The small dog barks", 2, 1, "noun_adj"),
    ("The red car moves", 2, 1, "noun_adj"),
    ("The old man walks", 2, 1, "noun_adj"),
    # verb → adverb
    ("The cat runs quickly", 2, 3, "verb_adv"),
    ("The dog barks loudly", 2, 3, "verb_adv"),
    ("The man walks slowly", 2, 3, "verb_adv"),
    ("The girl smiles brightly", 2, 3, "verb_adv"),
    # prep → prep_obj
    ("The cat sleeps on the mat", 3, 5, "prep_pobj"),
    ("The dog sits under the tree", 3, 5, "prep_pobj"),
    ("The bird flies over the house", 3, 5, "prep_pobj"),
    ("The man walks to the store", 3, 5, "prep_pobj"),
    # noun → determiner
    ("The cat chases the mouse", 1, 0, "noun_det"),
    ("The dog runs quickly", 1, 0, "noun_det"),
    ("The big cat sleeps", 2, 0, "noun_det"),
    ("The small dog barks", 2, 0, "noun_det"),
]

# 自然复杂句对
NATURAL_PAIRS = [
    # (sentence, head_idx, dep_idx, dep_type)
    # Embedded clauses
    ("The cat that chased the mouse slept", 1, 3, "subj_verb"),
    ("The cat that chased the mouse slept", 3, 5, "verb_obj"),
    ("The dog which the boy fed barked", 1, 5, "subj_verb"),
    ("The dog which the boy fed barked", 4, 1, "verb_obj"),
    # Passive
    ("The mouse was chased by the cat", 1, 3, "subj_verb"),
    ("The book was read by the student", 1, 3, "subj_verb"),
    ("The car was driven by the woman", 1, 3, "subj_verb"),
    # Coordination
    ("The cat and the dog chased the mouse", 1, 5, "subj_verb"),
    ("The cat and the dog chased the mouse", 3, 5, "subj_verb"),
    # Long distance
    ("The cat that the dog chased ran away", 1, 6, "subj_verb"),
    ("The man who the girl saw left", 1, 5, "subj_verb"),
    # Prepositional phrases
    ("The key to the door was lost", 1, 5, "subj_verb"),
    ("The book on the table is mine", 1, 5, "subj_verb"),
    # Relative clauses
    ("The student who reads books passed", 1, 5, "subj_verb"),
    ("The teacher that helped the boy smiled", 1, 6, "subj_verb"),
    # Multiple modifiers
    ("The big red car moved fast", 3, 4, "subj_verb"),
    ("The old wise man spoke slowly", 3, 4, "subj_verb"),
    # Control verbs
    ("The boy wanted to eat the food", 1, 2, "subj_verb"),
    ("The girl tried to open the door", 1, 2, "subj_verb"),
    # Raising verbs
    ("The cat seemed to sleep peacefully", 1, 3, "subj_verb"),
    ("The dog appeared to run quickly", 1, 3, "subj_verb"),
    # Tough movement
    ("The book is easy to read", 1, 4, "subj_verb"),
    ("The problem is hard to solve", 1, 4, "subj_verb"),
    # Verb_obj in complex
    ("The cat that chased the mouse ate it", 3, 5, "verb_obj"),
    ("The student who reads books passed", 3, 4, "verb_obj"),
    # Noun_adj in complex
    ("The very big cat slept", 3, 4, "subj_verb"),
    ("The extremely small dog barked", 3, 4, "subj_verb"),
    # Prep_pobj in complex
    ("The cat slept on the soft mat", 3, 6, "prep_pobj"),
    ("The dog sat under the old tree", 3, 6, "prep_pobj"),
    # Double object
    ("The man gave the girl the book", 2, 4, "verb_obj"),
    ("The teacher told the student the answer", 2, 4, "verb_obj"),
    # Dative alternation
    ("The man gave the book to the girl", 2, 4, "verb_obj"),
    ("The teacher told the answer to the student", 2, 4, "verb_obj"),
    # Cleft
    ("It was the cat that chased the mouse", 5, 6, "subj_verb"),
    ("It was the boy who fed the dog", 5, 6, "subj_verb"),
    # Questions
    ("What did the cat chase", 4, 3, "subj_verb"),
    ("Who did the dog bite", 4, 3, "subj_verb"),
    # Negation
    ("The cat did not chase the mouse", 1, 5, "subj_verb"),
    ("The dog has not eaten the food", 1, 5, "subj_verb"),
]


def extract_pair_features(model, tokenizer, sentence, head_idx, dep_idx, layer=-1):
    """Extract token representations and construct pair features.
    
    Note: head_idx and dep_idx are word-level indices. We find the corresponding
    token positions by searching for the token at that word position.
    """
    from tests.glm5.model_utils import get_layers
    layers = get_layers(model)
    target_layer = layer if layer >= 0 else len(layers) + layer
    
    toks = tokenizer(sentence, return_tensors="pt").to(_DEVICE)
    input_ids = toks["input_ids"]
    
    # Get token-level positions
    # Tokenizer may add BOS token, so word_idx may need +1 offset
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    n_tokens = len(tokens)
    
    # Adjust for BOS token offset
    # Most tokenizers add a BOS token at position 0
    has_bos = tokens[0] in ['<s>', '<|begin_of_text|>', '<|im_start|>', '<|endoftext|>', 'Ċ']
    offset = 1 if has_bos else 0
    
    actual_head = head_idx + offset
    actual_dep = dep_idx + offset
    
    if actual_head >= n_tokens or actual_dep >= n_tokens:
        raise IndexError(f"Index OOB: head={head_idx}→{actual_head}, dep={dep_idx}→{actual_dep}, n_tokens={n_tokens}, tokens={tokens[:10]}")
    
    captured = {}
    def make_hook():
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured["h"] = output[0].detach().float().cpu()
            else:
                captured["h"] = output.detach().float().cpu()
        return hook
    
    hook = layers[target_layer].register_forward_hook(make_hook())
    with torch.no_grad():
        _ = model(**toks)
    hook.remove()
    
    if "h" not in captured:
        raise ValueError(f"Failed to capture hidden states for: {sentence}")
    
    hidden = captured["h"][0].numpy()  # [seq_len, d_model]
    
    h_head = hidden[actual_head]
    h_dep = hidden[actual_dep]
    
    # Point features
    point_head = h_head
    point_dep = h_dep
    
    # Pair features: [h_i; h_j; h_i - h_j; h_i ⊙ h_j]
    pair_diff = h_head - h_dep
    pair_prod = h_head * h_dep  # element-wise product
    pair_full = np.concatenate([h_head, h_dep, pair_diff, pair_prod])
    
    return {
        "h_head": point_head,
        "h_dep": point_dep,
        "pair_diff": pair_diff,
        "pair_prod": pair_prod,
        "pair_full": pair_full,
    }


def extract_attention_weights(model, tokenizer, sentence, head_idx, dep_idx, layer=-1):
    """Extract attention weights between head and dep tokens."""
    from tests.glm5.model_utils import get_layers
    layers = get_layers(model)
    target_layer = layer if layer >= 0 else len(layers) + layer
    
    toks = tokenizer(sentence, return_tensors="pt").to(_DEVICE)
    input_ids = toks["input_ids"]
    
    # Adjust for BOS token
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    has_bos = tokens[0] in ['<s>', '<|begin_of_text|>', '<|im_start|>', 'Ċ']
    offset = 1 if has_bos else 0
    actual_head = head_idx + offset
    actual_dep = dep_idx + offset
    
    # Try to get attention from model output directly
    with torch.no_grad():
        outputs = model(**toks, output_attentions=True)
    
    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
        attn_pattern = outputs.attentions[target_layer][0].float().cpu().numpy()  # [n_heads, seq_len, seq_len]
    else:
        raise ValueError("Cannot extract attention weights - model doesn't output attentions")
    
    # Attention from dep to head (dep attends to head)
    attn_dep_to_head = attn_pattern[:, actual_dep, actual_head]  # [n_heads]
    # Attention from head to dep
    attn_head_to_dep = attn_pattern[:, actual_head, actual_dep]  # [n_heads]
    
    return {
        "attn_dep_to_head": attn_dep_to_head,
        "attn_head_to_dep": attn_head_to_dep,
        "attn_pattern": attn_pattern,
    }


# ============================================================
# Exp1: Pair-based vs Point-based 依存方向分类
# ============================================================

def exp1_pair_vs_point(model_name, model, tokenizer, layer=-1):
    """Compare pair-based vs point-based dependency TYPE classification.
    
    Key insight: dependency direction (head→dep) can only be encoded in pairs,
    not in single tokens. So we compare:
    1. Point: classify dependency type from head token alone
    2. Pair: classify dependency type from pair representation
    3. Direction: classify direction (head→dep vs dep→head) - only possible with pairs
    """
    print(f"\n{'='*60}")
    print(f"Exp1: Pair vs Point - 依存类型分类 + 方向分类 ({model_name})")
    print(f"{'='*60}")
    
    dep_types = ["subj_verb", "verb_obj", "noun_adj", "verb_adv", "prep_pobj", "noun_det"]
    
    X_point_head = []
    X_point_dep = []
    X_pair_full = []
    X_pair_diff = []
    X_pair_prod = []
    y_type = []
    
    for sent, h_idx, d_idx, dep_type in TEMPLATE_PAIRS:
        try:
            feats = extract_pair_features(model, tokenizer, sent, h_idx, d_idx, layer)
            X_point_head.append(feats["h_head"])
            X_point_dep.append(feats["h_dep"])
            X_pair_full.append(feats["pair_full"])
            X_pair_diff.append(feats["pair_diff"])
            X_pair_prod.append(feats["pair_prod"])
            y_type.append(dep_types.index(dep_type))
        except Exception as e:
            print(f"  Skip: {sent} - {e}")
            continue
    
    X_point_head = np.array(X_point_head)
    X_point_dep = np.array(X_point_dep)
    X_pair_full = np.array(X_pair_full)
    X_pair_diff = np.array(X_pair_diff)
    X_pair_prod = np.array(X_pair_prod)
    y_type = np.array(y_type)
    
    n = len(y_type)
    print(f"  Collected {n} samples")
    
    results = {"n_samples": n}
    clf = LinearSVC(max_iter=5000, C=1.0, dual=False)  # Faster than LogisticRegression for high-dim
    
    from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
    from sklearn.decomposition import PCA
    # Use StratifiedShuffleSplit for small datasets
    cv_type = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
    cv_dir = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
    
    # PCA for high-dim pair features to speed up
    d_single = X_point_head.shape[1]
    d_pair = X_pair_full.shape[1]
    
    # Reduce pair_full to min(20, n-1) dims for speed
    n_pca = min(20, n - 1)
    if d_pair > n_pca:
        pca = PCA(n_components=n_pca, random_state=42)
        X_pair_full_pca = pca.fit_transform(X_pair_full)
        print(f"  PCA: {d_pair} → {n_pca} dims (explained variance: {pca.explained_variance_ratio_.sum():.3f})")
    else:
        X_pair_full_pca = X_pair_full
    
    # --- 6-type dependency classification ---
    print("\n  === 6-type dependency classification ===")
    
    # Point-based: head token alone
    scores = cross_val_score(clf, X_point_head, y_type, cv=cv_type, scoring='accuracy')
    results["point_head_6type"] = float(scores.mean())
    print(f"  Point (head token):     {scores.mean():.3f}")
    
    # Point-based: dep token alone
    scores = cross_val_score(clf, X_point_dep, y_type, cv=cv_type, scoring='accuracy')
    results["point_dep_6type"] = float(scores.mean())
    print(f"  Point (dep token):      {scores.mean():.3f}")
    
    # Pair full (PCA-reduced)
    scores = cross_val_score(clf, X_pair_full_pca, y_type, cv=cv_type, scoring='accuracy')
    results["pair_full_6type"] = float(scores.mean())
    print(f"  Pair full [h;h;diff;prod] (PCA50): {scores.mean():.3f}")
    
    # Pair diff
    scores = cross_val_score(clf, X_pair_diff, y_type, cv=cv_type, scoring='accuracy')
    results["pair_diff_6type"] = float(scores.mean())
    print(f"  Pair diff [h_i-h_j]:   {scores.mean():.3f}")
    
    # Pair prod
    scores = cross_val_score(clf, X_pair_prod, y_type, cv=cv_type, scoring='accuracy')
    results["pair_prod_6type"] = float(scores.mean())
    print(f"  Pair prod [h_i⊙h_j]:   {scores.mean():.3f}")
    
    # Concat [h_head; h_dep]
    X_concat = np.array([np.concatenate([X_point_head[i], X_point_dep[i]]) for i in range(n)])
    scores = cross_val_score(clf, X_concat, y_type, cv=cv_type, scoring='accuracy')
    results["pair_concat_6type"] = float(scores.mean())
    print(f"  Pair concat [h_i;h_j]: {scores.mean():.3f}")
    
    # --- Direction classification (pair-only) ---
    print("\n  === Direction classification (head→dep vs dep→head) ===")
    
    # Construct direction data: forward pairs (y=1) and reverse pairs (y=0)
    d = X_point_head.shape[1]
    X_dir_full = list(X_pair_full)  # forward
    X_dir_diff = list(X_pair_diff)  # forward
    y_dir = [1] * n
    
    for i in range(n):
        # Reverse: swap head and dep
        pair_rev = np.concatenate([X_point_dep[i], X_point_head[i], 
                                   X_point_dep[i] - X_point_head[i],
                                   X_point_head[i] * X_point_dep[i]])
        X_dir_full.append(pair_rev)
        X_dir_diff.append(X_point_dep[i] - X_point_head[i])
        y_dir.append(0)
    
    X_dir_full = np.array(X_dir_full)
    X_dir_diff = np.array(X_dir_diff)
    y_dir = np.array(y_dir)
    
    # PCA for direction task
    n_pca_dir = min(20, len(y_dir) - 1)
    if X_dir_full.shape[1] > n_pca_dir:
        pca_dir = PCA(n_components=n_pca_dir, random_state=42)
        X_dir_full_pca = pca_dir.fit_transform(X_dir_full)
    else:
        X_dir_full_pca = X_dir_full
    
    # Pair full: direction classification
    scores = cross_val_score(clf, X_dir_full_pca, y_dir, cv=cv_dir, scoring='accuracy')
    results["pair_full_direction"] = float(scores.mean())
    print(f"  Pair full → direction:  {scores.mean():.3f}")
    
    # Pair diff: direction classification
    scores = cross_val_score(clf, X_dir_diff, y_dir, cv=cv_dir, scoring='accuracy')
    results["pair_diff_direction"] = float(scores.mean())
    print(f"  Pair diff → direction:  {scores.mean():.3f}")
    
    # Key comparison: can point features classify direction?
    # Answer: No! Because direction is relational, not point property
    # But let's verify: use h_head alone (all are y=1 for forward, so must add reverse)
    X_dir_point = list(X_point_head)  # forward: head token
    y_dir_point = [1] * n
    for i in range(n):
        X_dir_point.append(X_point_dep[i])  # reverse: the "dep" becomes the "head" in reverse
        y_dir_point.append(0)
    X_dir_point = np.array(X_dir_point)
    y_dir_point = np.array(y_dir_point)
    
    scores = cross_val_score(clf, X_dir_point, y_dir_point, cv=cv_dir, scoring='accuracy')
    results["point_direction"] = float(scores.mean())
    print(f"  Point (head/dep) → direction: {scores.mean():.3f}")
    
    # Improvement from pair over point
    if results["point_head_6type"] > 0:
        results["pair_improvement"] = results["pair_full_6type"] - results["point_head_6type"]
        print(f"\n  ★ Pair improvement over point: {results['pair_improvement']:+.3f}")
    
    return results


# ============================================================
# Exp2: Pair-based 依存类型分类
# ============================================================

def exp2_dep_type_classification(model_name, model, tokenizer, layer=-1):
    """Classify dependency types using pair vs point features."""
    print(f"\n{'='*60}")
    print(f"Exp2: 依存类型分类 ({model_name})")
    print(f"{'='*60}")
    
    dep_types = ["subj_verb", "verb_obj", "noun_adj", "verb_adv", "prep_pobj", "noun_det"]
    
    X_point_head = []
    X_pair_full = []
    X_pair_diff = []
    y_type = []
    
    for sent, h_idx, d_idx, dep_type in TEMPLATE_PAIRS:
        try:
            feats = extract_pair_features(model, tokenizer, sent, h_idx, d_idx, layer)
            X_point_head.append(feats["h_head"])
            X_pair_full.append(feats["pair_full"])
            X_pair_diff.append(feats["pair_diff"])
            y_type.append(dep_types.index(dep_type))
        except Exception as e:
            continue
    
    X_point_head = np.array(X_point_head)
    X_pair_full = np.array(X_pair_full)
    X_pair_diff = np.array(X_pair_diff)
    y = np.array(y_type)
    
    results = {}
    clf = LogisticRegression(max_iter=2000, C=1.0)
    
    from sklearn.model_selection import cross_val_score
    
    # Point-based
    scores = cross_val_score(clf, X_point_head, y, cv=5, scoring='accuracy')
    results["point_head_6type_acc"] = float(scores.mean())
    print(f"  Point (head) 6-type: {scores.mean():.3f}")
    
    # Pair full
    scores = cross_val_score(clf, X_pair_full, y, cv=5, scoring='accuracy')
    results["pair_full_6type_acc"] = float(scores.mean())
    print(f"  Pair full 6-type: {scores.mean():.3f}")
    
    # Pair diff
    scores = cross_val_score(clf, X_pair_diff, y, cv=5, scoring='accuracy')
    results["pair_diff_6type_acc"] = float(scores.mean())
    print(f"  Pair diff 6-type: {scores.mean():.3f}")
    
    # Confusion: subj_verb vs verb_obj (both involve verb)
    mask = (y == 0) | (y == 1)  # subj_verb or verb_obj
    if mask.sum() > 10:
        scores = cross_val_score(clf, X_pair_full[mask], y[mask], cv=min(5, mask.sum()), scoring='accuracy')
        results["pair_subj_vs_obj_acc"] = float(scores.mean())
        print(f"  Pair subj_verb vs verb_obj: {scores.mean():.3f}")
    
    return results


# ============================================================
# Exp3: 自然复杂句验证
# ============================================================

def exp3_natural_pairs(model_name, model, tokenizer, layer=-1):
    """Test pair encoding on natural complex sentences."""
    print(f"\n{'='*60}")
    print(f"Exp3: 自然复杂句 pair编码 ({model_name})")
    print(f"{'='*60}")
    
    dep_types = ["subj_verb", "verb_obj", "noun_adj", "verb_adv", "prep_pobj", "noun_det"]
    
    X_point_head = []
    X_pair_full = []
    X_pair_diff = []
    y_type = []
    y_direction = []
    
    for sent, h_idx, d_idx, dep_type in NATURAL_PAIRS:
        try:
            feats = extract_pair_features(model, tokenizer, sent, h_idx, d_idx, layer)
            X_point_head.append(feats["h_head"])
            X_pair_full.append(feats["pair_full"])
            X_pair_diff.append(feats["pair_diff"])
            y_type.append(dep_types.index(dep_type))
            y_direction.append(1)
        except Exception as e:
            continue
    
    X_point_head = np.array(X_point_head)
    X_pair_full = np.array(X_pair_full)
    X_pair_diff = np.array(X_pair_diff)
    y_type = np.array(y_type)
    y_dir = np.array(y_direction)
    
    results = {"n_samples": len(y_type)}
    
    if len(y_type) < 10:
        print("  Too few natural samples!")
        return results
    
    clf = LogisticRegression(max_iter=2000, C=1.0)
    
    # Train on template, test on natural (generalization test)
    # Template data
    X_t_point = []
    X_t_pair = []
    X_t_diff = []
    y_t = []
    for sent, h_idx, d_idx, dep_type in TEMPLATE_PAIRS:
        try:
            feats = extract_pair_features(model, tokenizer, sent, h_idx, d_idx, layer)
            X_t_point.append(feats["h_head"])
            X_t_pair.append(feats["pair_full"])
            X_t_diff.append(feats["pair_diff"])
            y_t.append(dep_types.index(dep_type))
        except:
            continue
    
    X_t_point = np.array(X_t_point)
    X_t_pair = np.array(X_t_pair)
    X_t_diff = np.array(X_t_diff)
    y_t = np.array(y_t)
    
    # Point-based: train template → test natural
    clf.fit(X_t_point, y_t)
    pred_point = clf.predict(X_point_head)
    acc_point = accuracy_score(y_type, pred_point)
    results["point_template_to_natural"] = float(acc_point)
    print(f"  Point (template→natural): {acc_point:.3f}")
    
    # Pair full: train template → test natural
    clf.fit(X_t_pair, y_t)
    pred_pair = clf.predict(X_pair_full)
    acc_pair = accuracy_score(y_type, pred_pair)
    results["pair_template_to_natural"] = float(acc_pair)
    print(f"  Pair full (template→natural): {acc_pair:.3f}")
    
    # Pair diff: train template → test natural
    clf.fit(X_t_diff, y_t)
    pred_diff = clf.predict(X_pair_diff)
    acc_diff = accuracy_score(y_type, pred_diff)
    results["diff_template_to_natural"] = float(acc_diff)
    print(f"  Pair diff (template→natural): {acc_diff:.3f}")
    
    # Per-type accuracy
    type_names = dep_types
    per_type_point = {}
    per_type_pair = {}
    for t in range(len(type_names)):
        mask = y_type == t
        if mask.sum() > 0:
            pt_acc = accuracy_score(y_type[mask], pred_point[mask])
            pr_acc = accuracy_score(y_type[mask], pred_pair[mask])
            per_type_point[type_names[t]] = float(pt_acc)
            per_type_pair[type_names[t]] = float(pr_acc)
            print(f"    {type_names[t]:12s}: point={pt_acc:.3f}  pair={pr_acc:.3f}")
    
    results["per_type_point"] = per_type_point
    results["per_type_pair"] = per_type_pair
    
    return results


# ============================================================
# Exp4: 注意力权重 vs 依存关系相关性
# ============================================================

def exp4_attention_dependency(model_name, model, tokenizer, layer=-1):
    """Correlate attention weights with dependency structure.
    
    Key question: Does the dependent token attend more to the head than vice versa?
    If yes, attention partially implements dependency parsing.
    """
    print(f"\n{'='*60}")
    print(f"Exp4: 注意力 vs 依存关系 ({model_name})")
    print(f"{'='*60}")
    
    dep_types = ["subj_verb", "verb_obj", "noun_adj", "verb_adv", "prep_pobj", "noun_det"]
    
    # Use fewer samples (2 per type) for speed with eager attention
    reduced_pairs = TEMPLATE_PAIRS[::2]  # every other pair = 2 per type
    
    attn_dep_to_head = []
    attn_head_to_dep = []
    y_type = []
    
    for sent, h_idx, d_idx, dep_type in reduced_pairs:
        try:
            attn = extract_attention_weights(model, tokenizer, sent, h_idx, d_idx, layer)
            attn_dep_to_head.append(attn["attn_dep_to_head"].mean())
            attn_head_to_dep.append(attn["attn_head_to_dep"].mean())
            y_type.append(dep_types.index(dep_type))
            print(f"  {dep_type:12s}: dep→head={attn['attn_dep_to_head'].mean():.4f}  head→dep={attn['attn_head_to_dep'].mean():.4f}")
        except Exception as e:
            print(f"  Skip: {sent} - {e}")
            continue
    
    attn_dep_to_head = np.array(attn_dep_to_head)
    attn_head_to_dep = np.array(attn_head_to_dep)
    y_type = np.array(y_type)
    
    results = {"n_samples": len(y_type)}
    
    # Average attention by dependency type
    print("\n  Average attention by dependency type:")
    type_attn = {}
    for t_idx, t_name in enumerate(dep_types):
        mask = y_type == t_idx
        if mask.sum() > 0:
            avg_d2h = attn_dep_to_head[mask].mean()
            avg_h2d = attn_head_to_dep[mask].mean()
            ratio = avg_d2h / (avg_h2d + 1e-10)
            type_attn[t_name] = {
                "dep_to_head": float(avg_d2h),
                "head_to_dep": float(avg_h2d),
                "ratio": float(ratio),
            }
            print(f"    {t_name:12s}: dep→head={avg_d2h:.4f}  head→dep={avg_h2d:.4f}  ratio={ratio:.2f}")
    
    results["type_attn"] = type_attn
    
    # Key test: does dep attend MORE to head than head to dep?
    d2h_mean = attn_dep_to_head.mean()
    h2d_mean = attn_head_to_dep.mean()
    results["overall_dep_to_head"] = float(d2h_mean)
    results["overall_head_to_dep"] = float(h2d_mean)
    results["overall_ratio"] = float(d2h_mean / (h2d_mean + 1e-10))
    print(f"\n  Overall: dep→head={d2h_mean:.4f}  head→dep={h2d_mean:.4f}  ratio={d2h_mean/(h2d_mean+1e-10):.2f}")
    
    # Statistical test
    if len(attn_dep_to_head) > 2:
        from scipy.stats import ttest_rel
        t_stat, p_val = ttest_rel(attn_dep_to_head, attn_head_to_dep)
        results["paired_t_stat"] = float(t_stat)
        results["paired_p_val"] = float(p_val)
        print(f"  Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")
    
    return results


# ============================================================
# Exp5: Pair组件消融
# ============================================================

def exp5_pair_ablation(model_name, model, tokenizer, layer=-1):
    """Ablation: which pair component is most important?"""
    print(f"\n{'='*60}")
    print(f"Exp5: Pair组件消融 ({model_name})")
    print(f"{'='*60}")
    
    dep_types = ["subj_verb", "verb_obj", "noun_adj", "verb_adv", "prep_pobj", "noun_det"]
    
    X_h_head = []  # h_head alone
    X_h_dep = []   # h_dep alone
    X_diff = []     # h_head - h_dep
    X_prod = []     # h_head ⊙ h_dep
    X_concat = []   # [h_head; h_dep]
    X_full = []     # [h_head; h_dep; diff; prod]
    y = []
    
    for sent, h_idx, d_idx, dep_type in TEMPLATE_PAIRS:
        try:
            feats = extract_pair_features(model, tokenizer, sent, h_idx, d_idx, layer)
            X_h_head.append(feats["h_head"])
            X_h_dep.append(feats["h_dep"])
            X_diff.append(feats["pair_diff"])
            X_prod.append(feats["pair_prod"])
            X_concat.append(np.concatenate([feats["h_head"], feats["h_dep"]]))
            X_full.append(feats["pair_full"])
            y.append(dep_types.index(dep_type))
        except:
            continue
    
    X_h_head = np.array(X_h_head)
    X_h_dep = np.array(X_h_dep)
    X_diff = np.array(X_diff)
    X_prod = np.array(X_prod)
    X_concat = np.array(X_concat)
    X_full = np.array(X_full)
    y = np.array(y)
    
    results = {}
    clf = LinearSVC(max_iter=5000, C=1.0, dual=False)
    
    from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
    
    configs = [
        ("h_head", X_h_head),
        ("h_dep", X_h_dep),
        ("h_head-h_dep", X_diff),
        ("h_head⊙h_dep", X_prod),
        ("[h_head;h_dep]", X_concat),
        ("[h_head;h_dep;diff;prod]", X_full),
    ]
    
    print("  Component ablation (6-type classification):")
    for name, X in configs:
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        results[name] = float(scores.mean())
        print(f"    {name:30s}: {scores.mean():.3f}")
    
    # Incremental: what does each component add?
    print("\n  Incremental contribution:")
    base = X_concat  # [h_head; h_dep]
    scores_base = cross_val_score(clf, base, y, cv=cv, scoring='accuracy')
    
    with_diff = np.concatenate([base, X_diff], axis=1)
    scores_diff = cross_val_score(clf, with_diff, y, cv=cv, scoring='accuracy')
    
    with_prod = np.concatenate([base, X_prod], axis=1)
    scores_prod = cross_val_score(clf, with_prod, y, cv=cv, scoring='accuracy')
    
    with_both = X_full
    scores_both = cross_val_score(clf, with_both, y, cv=cv, scoring='accuracy')
    
    results["incremental_base"] = float(scores_base.mean())
    results["incremental_+diff"] = float(scores_diff.mean())
    results["incremental_+prod"] = float(scores_prod.mean())
    results["incremental_+both"] = float(scores_both.mean())
    
    print(f"    [h_head;h_dep]              : {scores_base.mean():.3f}")
    print(f"    [h_head;h_dep; diff]        : {scores_diff.mean():.3f}  (Δ={scores_diff.mean()-scores_base.mean():+.3f})")
    print(f"    [h_head;h_dep; prod]        : {scores_prod.mean():.3f}  (Δ={scores_prod.mean()-scores_base.mean():+.3f})")
    print(f"    [h_head;h_dep; diff; prod]  : {scores_both.mean():.3f}  (Δ={scores_both.mean()-scores_base.mean():+.3f})")
    
    return results


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CCXLVII: Pair-based Relational Encoding")
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--layer", type=int, default=-1, help="Transformer layer (-1 = last)")
    args = parser.parse_args()
    
    # Load model
    print(f"Loading {args.model}...")
    model, tokenizer, device = load_model(args.model)
    model.eval()
    global _DEVICE
    _DEVICE = device
    
    # Run experiment
    if args.exp == 1:
        results = exp1_pair_vs_point(args.model, model, tokenizer, args.layer)
    elif args.exp == 2:
        results = exp2_dep_type_classification(args.model, model, tokenizer, args.layer)
    elif args.exp == 3:
        results = exp3_natural_pairs(args.model, model, tokenizer, args.layer)
    elif args.exp == 4:
        results = exp4_attention_dependency(args.model, model, tokenizer, args.layer)
    elif args.exp == 5:
        results = exp5_pair_ablation(args.model, model, tokenizer, args.layer)
    
    # Save results
    out_dir = Path(__file__).resolve().parent.parent / "glm5_temp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"ccxlvii_exp{args.exp}_{args.model}_results.json"
    
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
        return obj
    
    with open(out_file, "w") as f:
        json.dump(convert(results), f, indent=2)
    
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
