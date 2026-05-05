"""
CCML Phase 45: Trajectory Decomposition & Layer Dynamics — The Generative Theory
==================================================================================

User's decisive critique of Phase 44:
  "You have the phenomenology, but not the generative theory.
   You see the orbits, but don't understand the force field that generates them."

Three critical gaps identified:
  1. cos(h_A, h_B) ≈ 0.93 may reflect shared base, not semantic proximity
  2. W_U defines discriminative directions, not "holistic projection"
  3. We haven't analyzed F_l: h_{l+1} = h_l + F_l(h_l)

Core decomposition to verify:
  h(x) = h_base + h_sem(x) + h_noise(x)

  h_base: shared by all sentences (syntactic/structural/statistical)
  h_sem(x): semantic offset (low-amplitude but structured)
  h_noise: irrelevant perturbation

Key questions:
  45A: h(x) Decomposition
    - Compute h_base = mean of h_L(x) across diverse sentences
    - Compute h_sem(x) = h_L(x) - h_base
    - Is h_sem(x) low-dimensional?
    - Can h_sem(x) be linearly classified by semantic category?

  45B: Layer Dynamics F_l
    - Compute Δh_l = h_{l+1} - h_l for each layer
    - Is Δh_l contracting/expanding?
    - Does Δh_l align with h_l (radial) or orthogonal (tangential)?
    - Does F_l have fixed structures (e.g., dominant direction)?

  45C: Semantic Coordinate System
    - PCA on {h_sem(x_i)} → find semantic subspace U
    - Linear probe: can a linear classifier predict semantics from h_sem?
    - If yes → semantics is in a subspace (not holistic)
    - If no → semantics requires nonlinear structure

  45D: W_U as Discriminative Direction Family
    - Each token i defines a discriminative direction w_i = W_U[i]
    - Does h_sem(x) have high projection on semantically relevant w_i?
    - This tests: "W_U is a direction selector" vs "W_U is a holistic decoder"

Usage:
  python ccml_phase45_dynamics_decomposition.py --model deepseek7b --exp 1
  python ccml_phase45_dynamics_decomposition.py --model deepseek7b --exp 2
  python ccml_phase45_dynamics_decomposition.py --model deepseek7b --exp 3
  python ccml_phase45_dynamics_decomposition.py --model deepseek7b --exp 4
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
import gc
import time
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from model_utils import (load_model, get_layers, get_model_info, release_model,
                          safe_decode, collect_layer_outputs, get_W_U)

# ===== Sentence Sets =====
# Diverse sentences for h_base computation and semantic analysis
SEMANTIC_CATEGORIES = {
    "animals": [
        "The cat sat on the mat",
        "A dog ran through the park",
        "The bird flew over the tree",
        "My fish swam in the pond",
        "The horse galloped across the field",
        "A rabbit hopped through the garden",
        "The elephant drank from the river",
        "The snake slithered under the rock",
        "The whale dove into the ocean",
        "The bear climbed the mountain",
    ],
    "science": [
        "The scientist discovered a new element",
        "Researchers found evidence of dark matter",
        "The experiment confirmed the hypothesis",
        "Chemists synthesized a novel compound",
        "Physicists measured the particle decay",
        "Biologists identified a new species",
        "The telescope observed distant galaxies",
        "The microscope revealed cell structures",
        "Mathematicians proved the conjecture",
        "Engineers designed a new circuit",
    ],
    "emotion": [
        "She felt happy about the news",
        "He was angry at the decision",
        "They felt sad about the loss",
        "The child was excited for the trip",
        "She felt worried about the exam",
        "He was disgusted by the smell",
        "The woman felt proud of her work",
        "The man was surprised by the result",
        "She felt calm after the meditation",
        "He was confused by the instructions",
    ],
    "actions": [
        "She walked to the store yesterday",
        "He cooked dinner for the family",
        "They built a house last year",
        "She wrote a letter to her friend",
        "He drove the car to work",
        "They planted trees in the garden",
        "She painted the wall blue",
        "He fixed the broken window",
        "They cleaned the entire house",
        "She played the piano beautifully",
    ],
    "questions": [
        "What is the meaning of life?",
        "How does the brain process language?",
        "Why do people make mistakes?",
        "When will the rain stop falling?",
        "Where can I find the answer?",
        "Who discovered the new method?",
        "Which path leads to success?",
        "Can machines understand emotions?",
        "Should we explore the universe?",
        "Is knowledge always beneficial?",
    ],
    "descriptions": [
        "The sky was blue and clear",
        "The room was dark and cold",
        "The water was warm and calm",
        "The mountain was tall and steep",
        "The flower was red and beautiful",
        "The road was long and winding",
        "The building was old and crumbling",
        "The lake was still and peaceful",
        "The forest was dense and green",
        "The desert was vast and empty",
    ],
}

# Sentences for dynamics analysis (shorter, cleaner)
DYNAMICS_SENTENCES = [
    "The cat sat on the mat",
    "A dog ran through the park",
    "The scientist discovered a new element",
    "She felt happy about the news",
    "He cooked dinner for the family",
    "The sky was blue and clear",
    "What is the meaning of life?",
    "The bird flew over the tree",
    "She walked to the store yesterday",
    "Researchers found evidence of dark matter",
    "He was angry at the decision",
    "They built a house last year",
    "The room was dark and cold",
    "How does the brain process language?",
    "My fish swam in the pond",
    "The experiment confirmed the hypothesis",
    "She felt worried about the exam",
    "She wrote a letter to her friend",
    "The water was warm and calm",
    "Why do people make mistakes?",
]


def get_all_hidden_states(model, tokenizer, device, text, n_layers, d_model):
    """
    Get hidden states at ALL layers for a given text.
    Returns: [n_layers, d_model] array
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids).detach()
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    layers = get_layers(model)
    captured = {}

    def make_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook

    hooks = []
    for li in range(n_layers):
        hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))

    with torch.no_grad():
        try:
            _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids,
                     attention_mask=attention_mask)
        except Exception as e:
            print(f"  Forward failed for '{text[:30]}...': {e}")

    for h in hooks:
        h.remove()

    # Extract last token hidden state at each layer
    result = np.zeros((n_layers, d_model), dtype=np.float32)
    for li in range(n_layers):
        key = f"L{li}"
        if key in captured:
            result[li] = captured[key][0, -1, :].numpy()

    del captured
    torch.cuda.empty_cache()
    return result


# ============================================================================
# 45A: h(x) Decomposition — h_base + h_sem(x)
# ============================================================================

def run_45A(model_name):
    """
    Decompose h_L(x) = h_base + h_sem(x)
    
    Key questions:
    1. What fraction of h(x) is shared (h_base)?
    2. Is h_sem(x) low-dimensional?
    3. Can h_sem(x) be linearly classified by semantic category?
    """
    print(f"\n{'='*70}")
    print("45A: h(x) Decomposition — h_base + h_sem(x)")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  Model: {model_name}, layers={n_layers}, d_model={d_model}")

    # Collect all sentences
    all_sentences = []
    all_categories = []
    for cat, sents in SEMANTIC_CATEGORIES.items():
        all_sentences.extend(sents)
        all_categories.extend([cat] * len(sents))
    N = len(all_sentences)
    print(f"  Total sentences: {N}, categories: {len(SEMANTIC_CATEGORIES)}")

    # Collect hidden states at final layer for all sentences
    print("\n  Collecting final-layer hidden states...")
    t0 = time.time()

    H_final = np.zeros((N, d_model), dtype=np.float32)
    for i, sent in enumerate(all_sentences):
        hs = get_all_hidden_states(model, tokenizer, device, sent, n_layers, d_model)
        H_final[i] = hs[-1]  # final layer
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{N} sentences ({time.time()-t0:.1f}s)")

    print(f"  Collection complete: {time.time()-t0:.1f}s")

    # ===== Step 1: Compute h_base =====
    print("\n  ===== Step 1: Computing h_base =====")
    h_base = H_final.mean(axis=0)  # [d_model]
    h_base_norm = np.linalg.norm(h_base)
    print(f"  ||h_base|| = {h_base_norm:.2f}")

    # ===== Step 2: Compute h_sem(x) = h(x) - h_base =====
    print("\n  ===== Step 2: Computing h_sem(x) =====")
    H_sem = H_final - h_base[np.newaxis, :]  # [N, d_model]

    # Check norms
    h_norms = np.linalg.norm(H_final, axis=1)
    h_sem_norms = np.linalg.norm(H_sem, axis=1)
    h_base_proj_norms = np.array([abs(np.dot(H_final[i], h_base) / h_base_norm) for i in range(N)])

    print(f"  Mean ||h(x)|| = {np.mean(h_norms):.2f} ± {np.std(h_norms):.2f}")
    print(f"  Mean ||h_sem(x)|| = {np.mean(h_sem_norms):.2f} ± {np.std(h_sem_norms):.2f}")
    print(f"  Mean ||h_base projection|| = {np.mean(h_base_proj_norms):.2f} ± {np.std(h_base_proj_norms):.2f}")
    print(f"  ||h_sem|| / ||h|| = {np.mean(h_sem_norms) / np.mean(h_norms):.4f}")

    # ===== Step 3: Project h(x) onto h_base direction =====
    print("\n  ===== Step 3: h_base dominance test =====")
    # For each h(x), compute fraction of energy in h_base direction vs. orthogonal
    h_base_unit = h_base / max(h_base_norm, 1e-10)

    proj_on_base = np.array([np.dot(H_final[i], h_base_unit) for i in range(N)])
    proj_orthogonal = np.sqrt(np.maximum(h_norms**2 - proj_on_base**2, 0))

    print(f"  Projection on h_base: {np.mean(proj_on_base):.2f} ± {np.std(proj_on_base):.2f}")
    print(f"  Projection orthogonal: {np.mean(proj_orthogonal):.2f} ± {np.std(proj_orthogonal):.2f}")
    print(f"  Fraction along h_base: {np.mean(proj_on_base**2) / np.mean(h_norms**2):.4f}")

    # ===== Step 4: Dimensionality of h_sem =====
    print("\n  ===== Step 4: Dimensionality of h_sem(x) =====")
    U_sem, s_sem, Vt_sem = np.linalg.svd(H_sem, full_matrices=False)
    total_var_sem = np.sum(s_sem**2)

    if total_var_sem > 1e-10:
        cum_var_sem = np.cumsum(s_sem**2) / total_var_sem
        dim90_sem = int(np.searchsorted(cum_var_sem, 0.90) + 1)
        dim50_sem = int(np.searchsorted(cum_var_sem, 0.50) + 1)
        dim95_sem = int(np.searchsorted(cum_var_sem, 0.95) + 1)
        top1_pct_sem = s_sem[0]**2 / total_var_sem * 100
        top3_pct_sem = np.sum(s_sem[:3]**2) / total_var_sem * 100
        top10_pct_sem = np.sum(s_sem[:10]**2) / total_var_sem * 100

        print(f"  h_sem dim90: {dim90_sem}, dim50: {dim50_sem}, dim95: {dim95_sem}")
        print(f"  top1: {top1_pct_sem:.1f}%, top3: {top3_pct_sem:.1f}%, top10: {top10_pct_sem:.1f}%")
    else:
        dim90_sem = 0
        dim50_sem = 0
        print(f"  h_sem has near-zero variance!")

    # Compare with dimensionality of H_final (including h_base)
    U_full, s_full, Vt_full = np.linalg.svd(H_final - H_final.mean(axis=0), full_matrices=False)
    total_var_full = np.sum(s_full**2)
    if total_var_full > 1e-10:
        cum_var_full = np.cumsum(s_full**2) / total_var_full
        dim90_full = int(np.searchsorted(cum_var_full, 0.90) + 1)
        print(f"  H_final (centered) dim90: {dim90_full}")
        print(f"  → h_sem dim90 ({dim90_sem}) vs centered H dim90 ({dim90_full})")

    # ===== Step 5: Semantic classification on h_sem =====
    print("\n  ===== Step 5: Semantic Classification on h_sem =====")
    # Use logistic regression to classify semantic category from h_sem
    X = H_sem  # [N, d_model]
    y = np.array(all_categories)

    # Need enough samples per class
    unique_cats = list(SEMANTIC_CATEGORIES.keys())
    min_samples = min(len(SEMANTIC_CATEGORIES[c]) for c in unique_cats)

    if min_samples >= 5:
        # PCA reduce to top 50 components (avoid curse of dimensionality)
        n_pca = min(50, min(X.shape) - 1)
        X_pca = U_sem[:, :n_pca] * s_sem[:n_pca]  # [N, n_pca]

        # 5-fold cross-validation
        clf = LogisticRegression(max_iter=1000, C=1.0)
        scores = cross_val_score(clf, X_pca, y, cv=min(5, min_samples), scoring='accuracy')

        print(f"  Logistic regression on h_sem (PCA-{n_pca}):")
        print(f"    Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        print(f"    Chance level: {1/len(unique_cats):.4f}")
        print(f"    Above chance: {'YES' if np.mean(scores) > 2/len(unique_cats) else 'NO'}")

        # Also test on raw h(x) for comparison
        X_full = H_final - H_final.mean(axis=0)
        U_f2, s_f2, _ = np.linalg.svd(X_full, full_matrices=False)
        n_pca2 = min(50, min(X_full.shape) - 1)
        X_full_pca = U_f2[:, :n_pca2] * s_f2[:n_pca2]

        scores_full = cross_val_score(clf, X_full_pca, y, cv=min(5, min_samples), scoring='accuracy')
        print(f"\n  Logistic regression on h(x) (centered, PCA-{n_pca2}):")
        print(f"    Accuracy: {np.mean(scores_full):.4f} ± {np.std(scores_full):.4f}")

        # Test on just h_base direction (should be near chance)
        X_base_only = proj_on_base.reshape(-1, 1)
        scores_base = cross_val_score(clf, X_base_only, y, cv=min(5, min_samples), scoring='accuracy')
        print(f"\n  Logistic regression on h_base projection only:")
        print(f"    Accuracy: {np.mean(scores_base):.4f} ± {np.std(scores_base):.4f}")
    else:
        print(f"  Not enough samples per class for classification (min={min_samples})")

    # ===== Step 6: Cosine similarity analysis with h_base removed =====
    print("\n  ===== Step 6: Cosine Similarity (h_sem) =====")
    # Recompute cos similarity on h_sem (without h_base)
    for cat in unique_cats:
        cat_idx = [i for i, c in enumerate(all_categories) if c == cat]
        other_idx = [i for i, c in enumerate(all_categories) if c != cat]

        if len(cat_idx) < 2:
            continue

        # Within-category cos on h_sem
        cat_vecs = H_sem[cat_idx]
        norms = np.linalg.norm(cat_vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normed = cat_vecs / norms
        sim_within = normed @ normed.T
        n_c = len(cat_idx)
        within_sims = [sim_within[i, j] for i in range(n_c) for j in range(i+1, n_c)]

        # Cross-category cos on h_sem
        other_vecs = H_sem[other_idx]
        norms_o = np.linalg.norm(other_vecs, axis=1, keepdims=True)
        norms_o = np.maximum(norms_o, 1e-10)
        normed_o = other_vecs / norms_o
        cross_sim = normed @ normed_o.T
        cross_sims = cross_sim.flatten()

        print(f"  {cat:>12}: within_cos={np.mean(within_sims):.4f}, cross_cos={np.mean(cross_sims):.4f}, "
              f"gap={np.mean(within_sims)-np.mean(cross_sims):.4f}")

    # ===== Summary =====
    print("\n  ===== 45A SUMMARY =====")
    print(f"  ||h_base|| = {h_base_norm:.2f}")
    print(f"  ||h_sem|| / ||h|| = {np.mean(h_sem_norms) / np.mean(h_norms):.4f}")
    print(f"  Fraction along h_base: {np.mean(proj_on_base**2) / np.mean(h_norms**2):.4f}")
    print(f"  h_sem dim90: {dim90_sem}")
    if min_samples >= 5:
        print(f"  Classification accuracy (h_sem): {np.mean(scores):.4f}")
        print(f"  Classification accuracy (h_base only): {np.mean(scores_base):.4f}")

    # Key interpretation
    base_fraction = np.mean(proj_on_base**2) / np.mean(h_norms**2)
    sem_fraction = np.mean(h_sem_norms**2) / np.mean(h_norms**2)

    print(f"\n  KEY: h(x) = {base_fraction:.1%} h_base + {sem_fraction:.1%} h_sem(x)")
    if base_fraction > 0.9:
        print(f"  → h_base DOMINATES: cos(h_A, h_B) ≈ 0.93 is mainly base similarity")
    elif base_fraction > 0.5:
        print(f"  → h_base is MAJOR but h_sem is significant")
    else:
        print(f"  → h_base is NOT dominant; semantics is distributed")

    # Save results
    results = {
        'model': model_name,
        'n_sentences': N,
        'h_base_norm': float(h_base_norm),
        'h_sem_norm_mean': float(np.mean(h_sem_norms)),
        'h_norm_mean': float(np.mean(h_norms)),
        'sem_over_h_ratio': float(np.mean(h_sem_norms) / np.mean(h_norms)),
        'base_fraction': float(base_fraction),
        'h_sem_dim90': dim90_sem,
        'h_sem_dim50': dim50_sem,
    }
    if min_samples >= 5:
        results['clf_accuracy_hsem'] = float(np.mean(scores))
        results['clf_accuracy_hbase'] = float(np.mean(scores_base))
        results['clf_accuracy_hfull'] = float(np.mean(scores_full))

    out_path = f"tests/glm5_temp/phase45A_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    release_model(model)
    return results


# ============================================================================
# 45B: Layer Dynamics — Analyzing F_l: h_{l+1} = h_l + Δh_l
# ============================================================================

def run_45B(model_name):
    """
    Analyze the layer-to-layer dynamics Δh_l = h_{l+1} - h_l
    
    Key questions:
    1. Is Δh_l contracting (||Δh|| decreasing) or expanding?
    2. Does Δh_l align with h_l (radial) or is orthogonal (tangential)?
    3. Is there a dominant direction for Δh_l?
    4. Does the dynamics have fixed structures?
    """
    print(f"\n{'='*70}")
    print("45B: Layer Dynamics — h_{l+1} = h_l + F_l(h_l)")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  Model: {model_name}, layers={n_layers}, d_model={d_model}")

    # Collect full trajectories for all dynamics sentences
    print("\n  Collecting full trajectories...")
    t0 = time.time()

    N = len(DYNAMICS_SENTENCES)
    # trajectories[i] = [n_layers, d_model]
    trajectories = []

    for i, sent in enumerate(DYNAMICS_SENTENCES):
        hs = get_all_hidden_states(model, tokenizer, device, sent, n_layers, d_model)
        trajectories.append(hs)
        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{N} sentences ({time.time()-t0:.1f}s)")

    print(f"  Collection complete: {time.time()-t0:.1f}s")

    # ===== Step 1: Δh_l dynamics =====
    print("\n  ===== Step 1: Δh_l = h_{l+1} - h_l =====")
    print(f"  {'Layer':>6} {'Mean||Δh||':>12} {'Std||Δh||':>12} {'Mean||h||':>12} {'||Δh||/||h||':>14} {'Radial%':>10}")

    delta_norms_by_layer = []  # For each layer, list of ||Δh|| across sentences
    radial_fracs = []  # Fraction of Δh that is radial (aligned with h)

    for l in range(n_layers - 1):
        delta_h = np.array([trajectories[i][l+1] - trajectories[i][l] for i in range(N)])  # [N, d_model]
        h_l = np.array([trajectories[i][l] for i in range(N)])  # [N, d_model]

        delta_norms = np.linalg.norm(delta_h, axis=1)
        h_norms = np.linalg.norm(h_l, axis=1)

        mean_delta = np.mean(delta_norms)
        mean_h = np.mean(h_norms)
        ratio = mean_delta / max(mean_h, 1e-10)

        # Radial fraction: how much of Δh is in the direction of h_l
        radial_fracs_l = []
        for i in range(N):
            if h_norms[i] > 1e-10 and delta_norms[i] > 1e-10:
                h_unit = h_l[i] / h_norms[i]
                proj = np.dot(delta_h[i], h_unit)
                radial_frac = proj**2 / (delta_norms[i]**2 + 1e-20)
                radial_fracs_l.append(radial_frac)
        mean_radial = np.mean(radial_fracs_l) if radial_fracs_l else 0

        delta_norms_by_layer.append(mean_delta)
        radial_fracs.append(mean_radial)

        if l % max(1, n_layers // 10) == 0 or l == n_layers - 2:
            print(f"  {l:>6} {mean_delta:>12.2f} {np.std(delta_norms):>12.2f} "
                  f"{mean_h:>12.2f} {ratio:>14.6f} {mean_radial*100:>9.1f}%")

    # ===== Step 2: Contraction rate =====
    print("\n  ===== Step 2: Contraction Rate =====")
    # Does ||Δh|| decrease across layers?
    if len(delta_norms_by_layer) > 5:
        early_delta = np.mean(delta_norms_by_layer[:5])
        late_delta = np.mean(delta_norms_by_layer[-5:])
        print(f"  Early layers ||Δh||: {early_delta:.2f}")
        print(f"  Late layers ||Δh||:  {late_delta:.2f}")
        contraction = late_delta / max(early_delta, 1e-10)
        print(f"  Late/Early ratio: {contraction:.4f}")
        if contraction < 0.5:
            print(f"  → STRONG contraction: dynamics is contracting")
        elif contraction < 1.0:
            print(f"  → Mild contraction")
        else:
            print(f"  → Expanding or constant dynamics")

    # ===== Step 3: Mean Δh direction — is there a "force field"? =====
    print("\n  ===== Step 3: Mean Δh Direction (Force Field) =====")
    # If there's a consistent direction for Δh across sentences, it means
    # the dynamics has a "mean force" that all trajectories follow
    print(f"  {'Layer':>6} {'||mean_Δh||':>12} {'Consistency':>12} {'cos(Δh_mean, h_mean)':>22}")

    mean_delta_dirs = []
    consistencies = []

    for l in range(n_layers - 1):
        delta_h = np.array([trajectories[i][l+1] - trajectories[i][l] for i in range(N)])
        h_l = np.array([trajectories[i][l] for i in range(N)])

        mean_delta = delta_h.mean(axis=0)  # [d_model]
        mean_h = h_l.mean(axis=0)
        mean_delta_norm = np.linalg.norm(mean_delta)
        mean_h_norm = np.linalg.norm(mean_h)

        # Consistency: what fraction of individual Δh aligns with the mean Δh?
        if mean_delta_norm > 1e-10:
            mean_dir = mean_delta / mean_delta_norm
            cos_with_mean = [np.dot(delta_h[i], mean_dir) / max(np.linalg.norm(delta_h[i]), 1e-10)
                            for i in range(N)]
            consistency = np.mean(cos_with_mean)
        else:
            consistency = 0

        # Angle between mean Δh and mean h
        cos_delta_h = 0
        if mean_delta_norm > 1e-10 and mean_h_norm > 1e-10:
            cos_delta_h = np.dot(mean_delta, mean_h) / (mean_delta_norm * mean_h_norm)

        mean_delta_dirs.append(mean_delta)
        consistencies.append(consistency)

        if l % max(1, n_layers // 10) == 0 or l == n_layers - 2:
            print(f"  {l:>6} {mean_delta_norm:>12.2f} {consistency:>12.4f} {cos_delta_h:>22.4f}")

    # ===== Step 4: Δh dimensional structure =====
    print("\n  ===== Step 4: Dimensionality of Δh =====")
    # Stack all Δh across sentences and do PCA
    print(f"  {'Layer':>6} {'dim90':>6} {'dim50':>6} {'top1%':>8} {'top3%':>8}")

    delta_dim90_by_layer = []

    for l in range(n_layers - 1):
        delta_h = np.array([trajectories[i][l+1] - trajectories[i][l] for i in range(N)])
        centered = delta_h - delta_h.mean(axis=0)

        U_d, s_d, Vt_d = np.linalg.svd(centered, full_matrices=False)
        total_var = np.sum(s_d**2)

        if total_var > 1e-10:
            cum_var = np.cumsum(s_d**2) / total_var
            dim90 = int(np.searchsorted(cum_var, 0.90) + 1)
            dim50 = int(np.searchsorted(cum_var, 0.50) + 1)
            top1_pct = s_d[0]**2 / total_var * 100
            top3_pct = np.sum(s_d[:3]**2) / total_var * 100
        else:
            dim90, dim50 = 0, 0
            top1_pct, top3_pct = 0, 0

        delta_dim90_by_layer.append(dim90)

        if l % max(1, n_layers // 10) == 0 or l == n_layers - 2:
            print(f"  {l:>6} {dim90:>6} {dim50:>6} {top1_pct:>7.1f}% {top3_pct:>7.1f}%")

    # ===== Step 5: h_base per layer (evolution of shared structure) =====
    print("\n  ===== Step 5: h_base Evolution Across Layers =====")
    # At each layer, compute the mean hidden state
    print(f"  {'Layer':>6} {'||h_mean||':>12} {'Cos(h_mean_l, h_mean_{l+1})':>28}")

    h_means = []
    for l in range(n_layers):
        h_l = np.array([trajectories[i][l] for i in range(N)])
        h_mean = h_l.mean(axis=0)
        h_means.append(h_mean)

    for l in range(n_layers - 1):
        norm_l = np.linalg.norm(h_means[l])
        norm_lp1 = np.linalg.norm(h_means[l+1])
        cos_consec = 0
        if norm_l > 1e-10 and norm_lp1 > 1e-10:
            cos_consec = np.dot(h_means[l], h_means[l+1]) / (norm_l * norm_lp1)

        if l % max(1, n_layers // 10) == 0 or l == n_layers - 2:
            print(f"  {l:>6} {norm_l:>12.2f} {cos_consec:>28.6f}")

    # ===== Step 6: Layer-wise h_sem decomposition =====
    print("\n  ===== Step 6: h_sem Evolution Across Layers =====")
    print(f"  {'Layer':>6} {'||h_base||':>12} {'Mean||h_sem||':>14} {'h_sem/||h||':>12} {'h_sem_dim90':>12}")

    for l in range(n_layers):
        h_l = np.array([trajectories[i][l] for i in range(N)])  # [N, d_model]
        h_base_l = h_l.mean(axis=0)
        H_sem_l = h_l - h_base_l[np.newaxis, :]

        h_base_norm_l = np.linalg.norm(h_base_l)
        h_sem_norms_l = np.linalg.norm(H_sem_l, axis=1)
        h_norms_l = np.linalg.norm(h_l, axis=1)

        sem_over_h = np.mean(h_sem_norms_l) / max(np.mean(h_norms_l), 1e-10)

        # PCA on h_sem
        U_s, s_s, _ = np.linalg.svd(H_sem_l, full_matrices=False)
        total_var = np.sum(s_s**2)
        dim90_sem_l = 0
        if total_var > 1e-10:
            cum_var = np.cumsum(s_s**2) / total_var
            dim90_sem_l = int(np.searchsorted(cum_var, 0.90) + 1)

        if l % max(1, n_layers // 8) == 0 or l == n_layers - 1:
            print(f"  {l:>6} {h_base_norm_l:>12.2f} {np.mean(h_sem_norms_l):>14.2f} "
                  f"{sem_over_h:>12.4f} {dim90_sem_l:>12}")

    # ===== Summary =====
    print("\n  ===== 45B SUMMARY =====")
    print(f"  Contraction rate (late/early Δh): {contraction:.4f}")
    print(f"  Mean consistency of Δh: {np.mean(consistencies):.4f}")
    print(f"  Mean radial fraction: {np.mean(radial_fracs)*100:.1f}%")
    print(f"  Δh dim90 range: [{min(delta_dim90_by_layer)}, {max(delta_dim90_by_layer)}]")

    if np.mean(consistencies) > 0.5:
        print(f"  → HIGH consistency: trajectories follow a shared force field")
    elif np.mean(consistencies) > 0.2:
        print(f"  → MODERATE consistency: shared structure + individual variation")
    else:
        print(f"  → LOW consistency: dynamics is highly input-dependent")

    radial_pct = np.mean(radial_fracs) * 100
    if radial_pct > 50:
        print(f"  → Radial dynamics: Δh mainly changes ||h|| (expansion/contraction)")
    elif radial_pct > 20:
        print(f"  → Mixed: Δh changes both ||h|| and direction")
    else:
        print(f"  → Tangential dynamics: Δh mainly rotates h (direction change)")

    # Save results
    results = {
        'model': model_name,
        'n_sentences': N,
        'n_layers': n_layers,
        'contraction_rate': float(contraction),
        'mean_consistency': float(np.mean(consistencies)),
        'mean_radial_fraction': float(np.mean(radial_fracs)),
        'delta_dim90_range': [int(min(delta_dim90_by_layer)), int(max(delta_dim90_by_layer))],
    }

    out_path = f"tests/glm5_temp/phase45B_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    release_model(model)
    return results


# ============================================================================
# 45C: Semantic Coordinate System
# ============================================================================

def run_45C(model_name):
    """
    Find the semantic subspace: does there exist a subspace U such that
    semantic information ≈ projection of h_sem onto U?
    
    Method:
    1. PCA on {h_sem(x_i)} → find principal semantic directions
    2. Check if semantic categories cluster in PCA space
    3. Test if linear probe on h_sem can predict semantics
    4. Check cross-model consistency
    """
    print(f"\n{'='*70}")
    print("45C: Semantic Coordinate System")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  Model: {model_name}, layers={n_layers}, d_model={d_model}")

    # Collect sentences with semantic labels
    all_sentences = []
    all_categories = []
    for cat, sents in SEMANTIC_CATEGORIES.items():
        all_sentences.extend(sents)
        all_categories.extend([cat] * len(sents))
    N = len(all_sentences)
    unique_cats = list(SEMANTIC_CATEGORIES.keys())

    # Collect hidden states at final and mid layers
    print("\n  Collecting hidden states at multiple layers...")
    t0 = time.time()

    # We'll test at final layer and 3/4 point
    test_layers = sorted(set([n_layers - 1, n_layers * 3 // 4, n_layers // 2]))
    print(f"  Test layers: {test_layers}")

    layer_H = {li: np.zeros((N, d_model), dtype=np.float32) for li in test_layers}

    for i, sent in enumerate(all_sentences):
        hs = get_all_hidden_states(model, tokenizer, device, sent, n_layers, d_model)
        for li in test_layers:
            if li < n_layers:
                layer_H[li][i] = hs[li]
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{N} sentences ({time.time()-t0:.1f}s)")

    print(f"  Collection complete: {time.time()-t0:.1f}s")

    W_U = get_W_U(model)  # [vocab_size, d_model]

    # ===== Analysis at each test layer =====
    for li in test_layers:
        print(f"\n  ===== Layer {li} =====")
        H = layer_H[li]
        h_base = H.mean(axis=0)
        H_sem = H - h_base[np.newaxis, :]

        # PCA on h_sem
        U_sem, s_sem, Vt_sem = np.linalg.svd(H_sem, full_matrices=False)
        total_var = np.sum(s_sem**2)

        if total_var < 1e-10:
            print(f"  h_sem has near-zero variance at layer {li}")
            continue

        cum_var = np.cumsum(s_sem**2) / total_var
        dim90 = int(np.searchsorted(cum_var, 0.90) + 1)

        # Project h_sem onto top PCA components
        n_top = min(20, len(s_sem))
        projections = H_sem @ Vt_sem[:n_top].T  # [N, n_top]

        # Semantic clustering in PCA space
        print(f"\n  Semantic clustering on PCA components (h_sem):")
        print(f"  {'Category':>12} {'PCA1_mean':>10} {'PCA2_mean':>10} {'PCA3_mean':>10}")

        for cat in unique_cats:
            cat_idx = [i for i, c in enumerate(all_categories) if c == cat]
            cat_proj = projections[cat_idx]
            print(f"  {cat:>12} {np.mean(cat_proj[:,0]):>10.2f} "
                  f"{np.mean(cat_proj[:,1]):>10.2f} {np.mean(cat_proj[:,2]):>10.2f}")

        # ANOVA on top PCA components
        print(f"\n  ANOVA on PCA components:")
        for pc_idx in range(min(5, n_top)):
            groups = [projections[np.array(all_categories) == cat, pc_idx] for cat in unique_cats]
            if all(len(g) >= 2 for g in groups):
                F, p = stats.f_oneway(*groups)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"    PCA{pc_idx+1}: F={F:>8.2f}, p={p:.6f} {sig}")

        # Linear classification with increasing dimensionality
        print(f"\n  Linear probe accuracy vs dimensionality:")
        y = np.array(all_categories)
        for n_dim in [3, 5, 10, 20, dim90]:
            if n_dim > n_top:
                continue
            X_probe = projections[:, :n_dim]
            clf = LogisticRegression(max_iter=1000, C=1.0)
            min_samples = min(len(SEMANTIC_CATEGORIES[c]) for c in unique_cats)
            cv = min(5, min_samples)
            scores = cross_val_score(clf, X_probe, y, cv=cv, scoring='accuracy')
            print(f"    PCA-{n_dim:>3}: accuracy = {np.mean(scores):.4f} ± {np.std(scores):.4f}")

        # ===== W_U discriminative direction analysis =====
        print(f"\n  W_U discriminative direction analysis:")

        # For each semantic category, find the most distinguishing tokens
        # Then check if h_sem projects highly onto those token directions
        h_base_unit = h_base / max(np.linalg.norm(h_base), 1e-10)

        # Project h_sem onto W_U rows (token directions)
        # logit_change_i = w_i · h_sem
        h_sem_mean_by_cat = {}
        for cat in unique_cats:
            cat_idx = [i for i, c in enumerate(all_categories) if c == cat]
            h_sem_mean_by_cat[cat] = H_sem[cat_idx].mean(axis=0)

        # For each pair of categories, find tokens with largest logit difference
        print(f"\n  Top-5 discriminative tokens per category pair:")
        pairs_tested = 0
        for ci, cat1 in enumerate(unique_cats):
            for cat2 in unique_cats[ci+1:]:
                delta_sem = h_sem_mean_by_cat[cat1] - h_sem_mean_by_cat[cat2]
                delta_logits = W_U @ delta_sem  # [vocab_size]

                # Top tokens with positive delta (favor cat1) and negative (favor cat2)
                top_pos_idx = np.argsort(delta_logits)[-5:][::-1]
                top_neg_idx = np.argsort(delta_logits)[:5]

                top_pos_toks = [safe_decode(tokenizer, int(idx)) for idx in top_pos_idx]
                top_neg_toks = [safe_decode(tokenizer, int(idx)) for idx in top_neg_idx]

                if pairs_tested < 3:  # Only show first 3 pairs
                    print(f"    {cat1} vs {cat2}:")
                    print(f"      Favor {cat1}: {top_pos_toks}")
                    print(f"      Favor {cat2}: {top_neg_toks}")
                pairs_tested += 1

    # ===== Summary =====
    print("\n  ===== 45C SUMMARY =====")
    print(f"  Tested at layers: {test_layers}")
    print(f"  Chance level: {1/len(unique_cats):.4f}")
    print(f"  → If linear probe >> chance → semantics is in a subspace")
    print(f"  → If linear probe ≈ chance → semantics requires nonlinear structure")

    release_model(model)
    return {'model': model_name, 'test_layers': test_layers}


# ============================================================================
# 45D: W_U as Discriminative Direction Family
# ============================================================================

def run_45D(model_name):
    """
    Test whether W_U acts as a family of discriminative directions.
    
    Key insight from user:
      "Each token = a discriminative direction"
      logit_i = <w_i, h(x)>
      Semantics = combined response across multiple discriminative directions
    
    Method:
    1. For each semantic contrast pair (A, B), compute Δh = h(B) - h(A)
    2. Compute Δlogits = W_U · Δh
    3. Find which token logits change the most
    4. Check: are the top-changed tokens semantically related to the contrast?
    5. Also check: does Δh have high projection on the semantically relevant w_i?
    """
    print(f"\n{'='*70}")
    print("45D: W_U as Discriminative Direction Family")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  Model: {model_name}, layers={n_layers}, d_model={d_model}")

    W_U = get_W_U(model)  # [vocab_size, d_model]

    # Semantic contrast pairs
    contrast_pairs = {
        "tense": [
            ("The cat sat on the mat", "The cat sits on the mat"),
            ("She walked to the store", "She walks to the store"),
            ("He discovered the truth", "He discovers the truth"),
            ("They built the house", "They build the house"),
            ("The scientist published results", "The scientist publishes results"),
            ("The bird flew over the tree", "The bird flies over the tree"),
            ("She wrote a letter", "She writes a letter"),
            ("He drove the car", "He drives the car"),
        ],
        "sentiment": [
            ("She felt happy about the news", "She felt sad about the news"),
            ("The movie was excellent", "The movie was terrible"),
            ("He loved the new design", "He hated the new design"),
            ("The food was delicious", "The food was awful"),
            ("They enjoyed the concert", "They disliked the concert"),
            ("The weather was beautiful", "The weather was horrible"),
            ("She appreciated the help", "She resented the help"),
            ("He admired the painting", "He despised the painting"),
        ],
        "negation": [
            ("The cat sat on the mat", "The cat did not sit on the mat"),
            ("She was happy about it", "She was not happy about it"),
            ("He found the answer", "He did not find the answer"),
            ("They succeeded in the task", "They failed in the task"),
            ("The plan worked perfectly", "The plan failed completely"),
            ("She understood the concept", "She misunderstood the concept"),
        ],
        "number": [
            ("The cat sat on the mat", "The cats sat on the mat"),
            ("A dog ran through the park", "Dogs ran through the park"),
            ("The bird flew over the tree", "The birds flew over the tree"),
            ("My friend helped me", "My friends helped me"),
            ("This book is interesting", "These books are interesting"),
            ("The child played outside", "The children played outside"),
        ],
    }

    # For each contrast type, define "expected relevant tokens"
    # These are tokens we'd EXPECT to change if semantics is properly encoded
    expected_relevant = {
        "tense": ["sat", "sits", "walked", "walks", "discovered", "discovers",
                   "built", "build", "published", "publishes", "flew", "flies"],
        "sentiment": ["happy", "sad", "excellent", "terrible", "loved", "hated",
                       "delicious", "awful", "enjoyed", "disliked"],
        "negation": ["not", "no", "never", "did", "failed", "misunderstood"],
        "number": ["cat", "cats", "dog", "dogs", "bird", "birds", "friend", "friends",
                    "book", "books", "child", "children", "is", "are", "was", "were"],
    }

    print("\n  Analyzing contrast pairs...")

    for ctype, pairs in contrast_pairs.items():
        print(f"\n  ===== Contrast type: {ctype} =====")

        all_delta_logits = []
        all_cos_delta_margin = []

        for A, B in pairs:
            # Get final hidden states
            hs_A = get_all_hidden_states(model, tokenizer, device, A, n_layers, d_model)
            hs_B = get_all_hidden_states(model, tokenizer, device, B, n_layers, d_model)

            h_A = hs_A[-1]
            h_B = hs_B[-1]
            delta_h = h_B - h_A

            delta_logits = W_U @ delta_h  # [vocab_size]

            # Top-10 changed tokens
            abs_changes = np.abs(delta_logits)
            top_idx = np.argsort(abs_changes)[-10:][::-1]
            top_toks = [(safe_decode(tokenizer, int(idx)), float(delta_logits[idx]))
                       for idx in top_idx]

            # Check how many top tokens are in expected_relevant
            expected = expected_relevant.get(ctype, [])
            n_relevant = sum(1 for tok, _ in top_toks
                           if any(e.lower() in tok.lower() for e in expected))

            all_delta_logits.append(delta_logits)

            # Also compute cos(Δh, W_U[i]) for each relevant token
            delta_norm = np.linalg.norm(delta_h)
            if delta_norm > 1e-10:
                delta_unit = delta_h / delta_norm
                # For expected tokens, compute alignment
                for e in expected[:5]:
                    tok_ids = tokenizer.encode(e, add_special_tokens=False)
                    if tok_ids:
                        w_i = W_U[tok_ids[0]]
                        w_norm = np.linalg.norm(w_i)
                        if w_norm > 1e-10:
                            cos_align = np.dot(delta_unit, w_i / w_norm)
                            if len(all_cos_delta_margin) < 20:
                                pass  # Collected below

            if len(all_delta_logits) <= 2:  # Show details for first 2 pairs
                print(f"\n    '{A}' → '{B}'")
                print(f"    ||Δh|| = {np.linalg.norm(delta_h):.2f}")
                print(f"    Top-10 logit changes:")
                for tok, val in top_toks:
                    marker = " ✓" if any(e.lower() in tok.lower() for e in expected) else ""
                    print(f"      {tok:>20}: {val:>+8.4f}{marker}")
                print(f"    Relevant tokens in top-10: {n_relevant}/10")

        # Aggregate analysis
        all_delta_logits = np.array(all_delta_logits)  # [n_pairs, vocab_size]
        mean_abs_change = np.mean(np.abs(all_delta_logits), axis=0)

        # Top-20 most affected tokens across all pairs of this type
        top_global_idx = np.argsort(mean_abs_change)[-20:][::-1]
        top_global_toks = [(safe_decode(tokenizer, int(idx)), float(mean_abs_change[idx]))
                          for idx in top_global_idx]

        n_globally_relevant = sum(1 for tok, _ in top_global_toks
                                 if any(e.lower() in tok.lower() for e in expected))

        print(f"\n  Top-20 globally affected tokens ({ctype}):")
        for tok, val in top_global_toks[:10]:
            marker = " ✓" if any(e.lower() in tok.lower() for e in expected) else ""
            print(f"    {tok:>20}: {val:.4f}{marker}")
        print(f"  Relevant tokens in global top-20: {n_globally_relevant}/20")

        # Key metric: relevance ratio
        if n_globally_relevant > 10:
            print(f"  → HIGH relevance: W_U discriminative directions capture semantics")
        elif n_globally_relevant > 5:
            print(f"  → MODERATE relevance: partial semantic encoding")
        else:
            print(f"  → LOW relevance: W_U directions don't capture this semantic contrast")

    # ===== Summary =====
    print("\n  ===== 45D SUMMARY =====")
    print(f"  Key question: Are semantically relevant tokens among the top-changed?")
    print(f"  If YES → W_U is a discriminative direction family that captures semantics")
    print(f"  If NO  → Semantic information is encoded differently than W_U directions suggest")

    release_model(model)
    return {'model': model_name}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 45: Dynamics Decomposition")
    parser.add_argument("--model", type=str, default="deepseek7b",
                       choices=["deepseek7b", "glm4", "qwen3"])
    parser.add_argument("--exp", type=int, default=0,
                       help="Experiment number: 1=45A, 2=45B, 3=45C, 4=45D, 0=all")
    args = parser.parse_args()

    if args.exp == 1 or args.exp == 0:
        run_45A(args.model)
        gc.collect()
        torch.cuda.empty_cache()

    if args.exp == 2 or args.exp == 0:
        run_45B(args.model)
        gc.collect()
        torch.cuda.empty_cache()

    if args.exp == 3 or args.exp == 0:
        run_45C(args.model)
        gc.collect()
        torch.cuda.empty_cache()

    if args.exp == 4 or args.exp == 0:
        run_45D(args.model)
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "="*70)
    print("Phase 45 complete!")
    print("="*70)


if __name__ == "__main__":
    main()
