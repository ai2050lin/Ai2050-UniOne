"""
CCML Phase 44: From Tangent Space to Trajectory — The Manifold Geometry of h(x)
===============================================================================

User's most profound critique:
  "You've been studying the tangent space (S(x)), but language computation
   happens on the manifold trajectory (h(x)). You understand the differential
   structure but not the geometric structure."

Core insight: S(x) = Im(J(x)) is the TANGENT SPACE of h(x)'s manifold.
The main object of study should be the TRAJECTORY h_0 → h_1 → ... → h_L,
not the tangent structure S(x).

Correct three-layer model:
  Layer 1: Trajectory h_0 → h_1 → ... → h_L  [THE MAIN OBJECT]
  Layer 2: Tangent dynamics J(x) → S(x)       [LOCAL STRUCTURE]
  Layer 3: Readout y = W_U · h_L              [OBSERVATION]

Language = Trajectory(x) + TangentStructure(x) + Readout

Key conceptual upgrades from this phase:
  ❌ "S(x) is the core object" → ✅ "S(x) is the tangent space; h(x) is the main object"
  ❌ "W_U · S(x) determines output" → ✅ "W_U · h(x) determines output"
  ❌ "S(x) encodes semantics" → ✅ "h(x) encodes semantics; S(x) is its local structure"
  ❌ "Perturbation propagation is the dynamics" → ✅ "Trajectory IS the dynamics"

Experiments:

44A: Trajectory Manifold Geometry
  - 60 diverse inputs across semantic categories
  - Record h_l at each sampled layer → full trajectory
  - PCA of {h_l(x_i)} at each layer → intrinsic dimensionality evolution
  - Track trajectory "spread" across layers → convergence/divergence
  - Semantic clustering in trajectory space
  - KEY QUESTION: Does h(x) lie on a low-dimensional manifold?

44B: Tangent Space = Trajectory Tangent? (Verification)
  - Apply small perturbations at embedding layer
  - Collect h_L(x+ε) - h_L(x) for many perturbations
  - Compare the span of these trajectory differences with S(x)
  - KEY QUESTION: Is S(x) truly the tangent space of the trajectory manifold?

44C: h(x) Readout — What Does W_U·h(x) Encode?
  - Semantic contrast pairs → Δh = h_L(B) - h_L(A)
  - W_U · Δh → Δlogits
  - Which token logits change? Semantic or random?
  - KEY QUESTION: Does h(x) directly encode semantic information through W_U?

Usage:
  python ccml_phase44_trajectory_manifold.py --model deepseek7b --exp 1
  python ccml_phase44_trajectory_manifold.py --model deepseek7b --exp 2
  python ccml_phase44_trajectory_manifold.py --model deepseek7b --exp 3
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
from scipy.linalg import subspace_angles

from model_utils import (load_model, get_layers, get_model_info, release_model,
                          safe_decode, collect_layer_outputs, get_W_U)

# ===== 配置 =====
N_RANDOM_DIRS = 50

# 60 diverse sentences for trajectory manifold analysis
TRAJECTORY_SENTENCES = {
    "animals": [
        "The cat sat on the mat",
        "A dog ran through the park",
        "The bird flew over the tree",
        "My fish swam in the pond",
        "The horse galloped across the field",
        "A rabbit hopped through the garden",
    ],
    "science": [
        "The scientist discovered a new element",
        "Researchers found evidence of dark matter",
        "The experiment confirmed the hypothesis",
        "Chemists synthesized a novel compound",
        "Physicists measured the particle decay",
        "Biologists identified a new species",
    ],
    "emotion": [
        "She felt happy about the news",
        "He was angry at the decision",
        "They felt sad about the loss",
        "The child was excited for the trip",
        "She felt worried about the exam",
        "He was disgusted by the smell",
    ],
    "actions": [
        "She walked to the store yesterday",
        "He cooked dinner for the family",
        "They built a house last year",
        "She wrote a letter to her friend",
        "He drove the car to work",
        "They planted trees in the garden",
    ],
    "descriptions": [
        "The sky was blue and clear",
        "The room was dark and cold",
        "The water was warm and calm",
        "The mountain was tall and steep",
        "The flower was red and beautiful",
        "The road was long and winding",
    ],
    "abstract": [
        "Freedom is essential for progress",
        "Knowledge requires careful study",
        "Justice demands fair treatment",
        "Creativity involves new perspectives",
        "Wisdom comes from experience",
        "Truth emerges through inquiry",
    ],
    "questions": [
        "What is the meaning of life?",
        "How does the brain process language?",
        "Why do people make mistakes?",
        "When will the rain stop falling?",
        "Where can I find the answer?",
        "Who discovered the new method?",
    ],
    "negation": [
        "The cat did not sit on the mat",
        "She never walked to the store",
        "He was not happy about the news",
        "They could not find the answer",
        "The experiment did not confirm it",
        "She had no reason to complain",
    ],
    "past_tense": [
        "The cat sat quietly on the mat",
        "She walked slowly to the store",
        "He discovered the hidden treasure",
        "They built the entire structure",
        "The scientist published the results",
        "She completed the difficult task",
    ],
    "present_tense": [
        "The cat sits quietly on the mat",
        "She walks slowly to the store",
        "He discovers the hidden treasure",
        "They build the entire structure",
        "The scientist publishes the results",
        "She completes the difficult task",
    ],
}

# Semantic contrast pairs for 44C
SEMANTIC_CONTRAST_PAIRS = {
    "tense": [
        ("The cat sat on the mat", "The cat sits on the mat"),
        ("She walked to the store", "She walks to the store"),
        ("He discovered the truth", "He discovers the truth"),
        ("They built the house", "They build the house"),
        ("The scientist published results", "The scientist publishes results"),
    ],
    "sentiment": [
        ("She felt happy about the news", "She felt sad about the news"),
        ("The movie was excellent", "The movie was terrible"),
        ("He loved the new design", "He hated the new design"),
        ("The food was delicious", "The food was awful"),
        ("They enjoyed the concert", "They disliked the concert"),
    ],
    "negation": [
        ("The cat sat on the mat", "The cat did not sit on the mat"),
        ("She was happy about it", "She was not happy about it"),
        ("He found the answer", "He did not find the answer"),
        ("They succeeded in the task", "They failed in the task"),
        ("The plan worked perfectly", "The plan failed completely"),
    ],
    "number": [
        ("The cat sat on the mat", "The cats sat on the mat"),
        ("A dog ran through the park", "Dogs ran through the park"),
        ("The bird flew over the tree", "The birds flew over the tree"),
        ("My friend helped me", "My friends helped me"),
        ("This book is interesting", "These books are interesting"),
    ],
}

# For tangent space verification (44B)
TANGENT_VERIFY_TEXTS = [
    "The cat sat on the mat",
    "The scientist discovered a new element",
    "She walked to the store yesterday",
]


def get_hidden_states_at_layers(model, tokenizer, device, text, sample_layers):
    """
    Get hidden states at specified layers for a given text.
    Returns dict: {layer_idx: h[seq_len, d_model]}
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids).detach()
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    
    # Use hooks to capture layer outputs
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
    for li in sample_layers:
        if li < len(layers):
            hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))
    
    with torch.no_grad():
        try:
            _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids,
                     attention_mask=attention_mask)
        except Exception as e:
            print(f"  [get_hidden_states] Forward failed: {e}")
    
    for h in hooks:
        h.remove()
    
    # Extract last token hidden state at each layer
    result = {}
    for li in sample_layers:
        key = f"L{li}"
        if key in captured:
            result[li] = captured[key][0, -1, :].numpy()  # last token
    
    del captured
    torch.cuda.empty_cache()
    return result


# ============================================================================
# 44A: Trajectory Manifold Geometry
# ============================================================================

def run_44A(model_name):
    """
    Trajectory Manifold Geometry
    
    Key questions:
    1. Does {h_L(x)} lie on a low-dimensional manifold?
    2. Do trajectories converge or diverge across layers?
    3. Do semantically similar inputs cluster in trajectory space?
    4. What is the effective dimensionality of the "language manifold"?
    """
    print(f"\n{'='*70}")
    print("44A: Trajectory Manifold Geometry")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  Model: {model_name}, layers={n_layers}, d_model={d_model}")
    
    # Sample layers for trajectory tracking
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    print(f"  Sample layers: {sample_layers}")
    
    # Collect all sentences
    all_sentences = []
    all_categories = []
    for cat, sents in TRAJECTORY_SENTENCES.items():
        all_sentences.extend(sents)
        all_categories.extend([cat] * len(sents))
    N = len(all_sentences)
    print(f"  Total sentences: {N}")
    
    # Collect hidden states
    print("\n  Collecting hidden states...")
    t0 = time.time()
    
    # {layer_idx: [h_1, h_2, ..., h_N]} each h_i is [d_model]
    layer_hidden = {li: [] for li in sample_layers}
    layer_hidden_norms = {li: [] for li in sample_layers}
    
    for i, sent in enumerate(all_sentences):
        hs = get_hidden_states_at_layers(model, tokenizer, device, sent, sample_layers)
        for li in sample_layers:
            if li in hs:
                layer_hidden[li].append(hs[li])
                layer_hidden_norms[li].append(np.linalg.norm(hs[li]))
            else:
                layer_hidden[li].append(np.zeros(d_model))
                layer_hidden_norms[li].append(0.0)
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{N} sentences processed ({time.time()-t0:.1f}s)")
    
    print(f"  Collection complete: {time.time()-t0:.1f}s")
    
    # ===== Analysis 1: Intrinsic dimensionality at each layer =====
    print("\n  ===== Analysis 1: Intrinsic Dimensionality Across Layers =====")
    print(f"  {'Layer':>6} {'dim90':>6} {'dim50':>6} {'top1%':>8} {'top3%':>8} {'top10%':>8} {'Var':>10}")
    
    layer_dim90 = {}
    layer_var = {}
    layer_pca_Vt = {}
    
    for li in sample_layers:
        H = np.array(layer_hidden[li])  # [N, d_model]
        centered = H - H.mean(axis=0)
        
        # SVD for PCA
        U, s, Vt = np.linalg.svd(centered, full_matrices=False)
        total_var = np.sum(s**2)
        
        if total_var < 1e-10:
            print(f"  {li:>6} {'N/A':>6} {'N/A':>6} {'N/A':>8} {'N/A':>8} {'N/A':>8} {total_var:>10.2f}")
            continue
        
        cum_var = np.cumsum(s**2) / total_var
        dim90 = int(np.searchsorted(cum_var, 0.90) + 1)
        dim50 = int(np.searchsorted(cum_var, 0.50) + 1)
        
        top1_pct = s[0]**2 / total_var * 100
        top3_pct = np.sum(s[:3]**2) / total_var * 100
        top10_pct = np.sum(s[:10]**2) / total_var * 100
        
        print(f"  {li:>6} {dim90:>6} {dim50:>6} {top1_pct:>7.1f}% {top3_pct:>7.1f}% {top10_pct:>7.1f}% {total_var:>10.1f}")
        
        layer_dim90[li] = dim90
        layer_var[li] = total_var
        layer_pca_Vt[li] = Vt
    
    # ===== Analysis 2: Trajectory convergence/divergence =====
    print("\n  ===== Analysis 2: Trajectory Convergence/Divergence =====")
    print(f"  {'Layer':>6} {'Mean||h||':>10} {'Std||h||':>10} {'MeanDist':>10} {'CentroidVar':>12}")
    
    for li in sample_layers:
        H = np.array(layer_hidden[li])  # [N, d_model]
        norms = np.array(layer_hidden_norms[li])
        centroid = H.mean(axis=0)
        dists_to_centroid = np.linalg.norm(H - centroid, axis=1)
        centroid_var = np.var(dists_to_centroid)
        
        print(f"  {li:>6} {np.mean(norms):>10.2f} {np.std(norms):>10.2f} "
              f"{np.mean(dists_to_centroid):>10.2f} {centroid_var:>12.2f}")
    
    # ===== Analysis 3: Semantic clustering at final layer =====
    print("\n  ===== Analysis 3: Semantic Clustering at Final Layer =====")
    final_li = sample_layers[-1]
    H_final = np.array(layer_hidden[final_li])  # [N, d_model]
    
    # Compute pairwise cosine similarities within and across categories
    unique_cats = list(TRAJECTORY_SENTENCES.keys())
    
    # Within-category similarity
    print(f"\n  Within-category cosine similarity (at layer {final_li}):")
    for cat in unique_cats:
        cat_idx = [i for i, c in enumerate(all_categories) if c == cat]
        if len(cat_idx) < 2:
            continue
        cat_vecs = H_final[cat_idx]
        # Pairwise cosine
        norms = np.linalg.norm(cat_vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = cat_vecs / norms
        sim_matrix = normalized @ normalized.T
        # Take upper triangle (exclude diagonal)
        n = len(cat_idx)
        sims = []
        for i in range(n):
            for j in range(i+1, n):
                sims.append(sim_matrix[i, j])
        print(f"    {cat:>12}: mean_cos={np.mean(sims):.4f}, std={np.std(sims):.4f}, n_pairs={len(sims)}")
    
    # Cross-category similarity
    print(f"\n  Cross-category cosine similarity (at layer {final_li}):")
    for i, cat1 in enumerate(unique_cats):
        for cat2 in unique_cats[i+1:]:
            idx1 = [i for i, c in enumerate(all_categories) if c == cat1]
            idx2 = [i for i, c in enumerate(all_categories) if c == cat2]
            vecs1 = H_final[idx1]
            vecs2 = H_final[idx2]
            norms1 = np.linalg.norm(vecs1, axis=1, keepdims=True)
            norms2 = np.linalg.norm(vecs2, axis=1, keepdims=True)
            norms1 = np.maximum(norms1, 1e-10)
            norms2 = np.maximum(norms2, 1e-10)
            norm1 = vecs1 / norms1
            norm2 = vecs2 / norms2
            sim_matrix = norm1 @ norm2.T
            cross_sims = sim_matrix.flatten()
            print(f"    {cat1:>12} x {cat2:>12}: mean_cos={np.mean(cross_sims):.4f}")
    
    # ===== Analysis 4: PCA projection and semantic structure =====
    print("\n  ===== Analysis 4: PCA Projection — Semantic Structure =====")
    # Project onto first 3 PCA components and check clustering
    if final_li in layer_pca_Vt:
        Vt = layer_pca_Vt[final_li]
        # Project each h_L onto first 3 principal components
        centered = H_final - H_final.mean(axis=0)
        projections = centered @ Vt[:3].T  # [N, 3]
        
        print(f"\n  PCA1 range: [{projections[:,0].min():.2f}, {projections[:,0].max():.2f}]")
        print(f"  PCA2 range: [{projections[:,1].min():.2f}, {projections[:,1].max():.2f}]")
        print(f"  PCA3 range: [{projections[:,2].min():.2f}, {projections[:,2].max():.2f}]")
        
        # Check if categories separate on PCA1
        print(f"\n  Mean PCA1 by category:")
        for cat in unique_cats:
            cat_idx = [i for i, c in enumerate(all_categories) if c == cat]
            cat_pca1 = projections[cat_idx, 0]
            print(f"    {cat:>12}: mean={np.mean(cat_pca1):>8.2f}, std={np.std(cat_pca1):>8.2f}")
        
        # ANOVA test: do categories differ on PCA1?
        cat_groups = []
        for cat in unique_cats:
            cat_idx = [i for i, c in enumerate(all_categories) if c == cat]
            cat_groups.append(projections[cat_idx, 0])
        if all(len(g) >= 2 for g in cat_groups):
            F_stat, p_val = stats.f_oneway(*cat_groups)
            print(f"\n  ANOVA on PCA1: F={F_stat:.4f}, p={p_val:.6f}")
            if p_val < 0.05:
                print(f"  → Categories DO separate on PCA1 (p < 0.05)")
            else:
                print(f"  → Categories do NOT separate on PCA1")
    
    # ===== Analysis 5: Trajectory path similarity =====
    print("\n  ===== Analysis 5: Trajectory Path Similarity =====")
    # For each pair of inputs, compute the "trajectory distance"
    # (average distance between their hidden states across layers)
    
    # Select a subset for pairwise analysis
    subset_indices = list(range(0, N, 3))  # every 3rd sentence
    subset_cats = [all_categories[i] for i in subset_indices]
    
    # Compute trajectory distances
    within_cat_dists = []
    cross_cat_dists = []
    
    for i_idx in range(len(subset_indices)):
        for j_idx in range(i_idx + 1, len(subset_indices)):
            i = subset_indices[i_idx]
            j = subset_indices[j_idx]
            
            # Trajectory distance: average L2 distance across layers
            dists = []
            for li in sample_layers:
                h_i = layer_hidden[li][i]
                h_j = layer_hidden[li][j]
                dists.append(np.linalg.norm(h_i - h_j))
            avg_dist = np.mean(dists)
            
            if subset_cats[i_idx] == subset_cats[j_idx]:
                within_cat_dists.append(avg_dist)
            else:
                cross_cat_dists.append(avg_dist)
    
    if within_cat_dists and cross_cat_dists:
        print(f"  Within-category trajectory distance: {np.mean(within_cat_dists):.2f} ± {np.std(within_cat_dists):.2f}")
        print(f"  Cross-category trajectory distance:  {np.mean(cross_cat_dists):.2f} ± {np.std(cross_cat_dists):.2f}")
        ratio = np.mean(cross_cat_dists) / max(np.mean(within_cat_dists), 1e-10)
        print(f"  Cross/Within ratio: {ratio:.3f}")
        if ratio > 1.2:
            print(f"  → Trajectories DO cluster by semantic category")
        else:
            print(f"  → Trajectories do NOT cluster by semantic category")
    
    # ===== Summary =====
    print("\n  ===== 44A SUMMARY =====")
    if layer_dim90:
        final_dim90 = layer_dim90.get(sample_layers[-1], 'N/A')
        first_dim90 = layer_dim90.get(sample_layers[0], 'N/A')
        print(f"  Final-layer dim90(h_L manifold): {final_dim90}")
        print(f"  First-layer dim90(h_0 manifold): {first_dim90}")
        print(f"  d_model = {d_model}, N_samples = {N}")
        if isinstance(final_dim90, int) and isinstance(first_dim90, int):
            if final_dim90 < first_dim90:
                print(f"  → Trajectories CONVERGE: dim90 decreases from {first_dim90} to {final_dim90}")
            else:
                print(f"  → Trajectories DIVERGE or maintain dimension")
        
        print(f"\n  dim90 across layers: {[layer_dim90.get(li, '?') for li in sample_layers]}")
    
    # Save results
    results = {
        'model': model_name,
        'n_sentences': N,
        'sample_layers': sample_layers,
        'layer_dim90': {str(k): v for k, v in layer_dim90.items()},
        'layer_total_var': {str(k): float(v) for k, v in layer_var.items()},
        'within_cat_traj_dist': float(np.mean(within_cat_dists)) if within_cat_dists else None,
        'cross_cat_traj_dist': float(np.mean(cross_cat_dists)) if cross_cat_dists else None,
    }
    
    out_path = f"tests/glm5_temp/phase44A_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    
    release_model(model)
    return results


# ============================================================================
# 44B: Tangent Space Verification — Is S(x) = T_{h(x)}M?
# ============================================================================

def run_44B(model_name):
    """
    Verify whether S(x) = Im(J(x)) is truly the tangent space of the trajectory manifold.
    
    Method:
    1. For each input x, compute S(x) by injecting random perturbations at embedding
       and measuring the span of {h_L(x+ε) - h_L(x)}
    2. This directly gives us the tangent space (if perturbations are small)
    3. Compare with S(x) from Jacobian method (inject at specific layer)
    
    If the two spaces match → S(x) is a good tangent space approximation
    If they differ → nonlinear effects are significant
    """
    print(f"\n{'='*70}")
    print("44B: Tangent Space Verification — Is S(x) = T_{h(x)}M?")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    layers = get_layers(model)
    
    # We'll inject at layer 0 (embedding) and measure at final layer
    # This gives us the "full Jacobian" tangent space
    inject_layer = 0
    target_layer = n_layers - 1
    
    # For comparison, also inject at a middle layer
    mid_layer = n_layers // 2
    
    results = []
    
    for text in TANGENT_VERIFY_TEXTS:
        print(f"\n  Text: '{text}'")
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # Step 1: Get base hidden state at final layer
        base_h = [None]
        def base_hook(module, input, output):
            if isinstance(output, tuple):
                base_h[0] = output[0][0, -1, :].detach().cpu().float().numpy()
            else:
                base_h[0] = output[0, -1, :].detach().cpu().float().numpy()
        
        handle = layers[target_layer].register_forward_hook(base_hook)
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        handle.remove()
        
        if base_h[0] is None:
            print(f"    Failed to get base hidden state, skipping")
            continue
        
        base_h_val = base_h[0]
        h_norm = np.linalg.norm(base_h_val)
        print(f"    Base ||h_L|| = {h_norm:.2f}")
        
        # Step 2: Inject small perturbations at EMBEDDING layer
        # This gives us the full-chain tangent space
        print(f"    Computing embedding-level tangent space (n_dirs={N_RANDOM_DIRS})...")
        
        embed_layer = model.get_input_embeddings()
        with torch.no_grad():
            inputs_embeds_base = embed_layer(input_ids).detach().clone()
        
        # Use small perturbation scale (1% of embedding norm)
        embed_norm = inputs_embeds_base[0, -1, :].float().norm().item()
        alpha = 0.01 * embed_norm
        
        np.random.seed(42)
        random_dirs = np.random.randn(N_RANDOM_DIRS, d_model)
        random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)
        
        delta_h_embed = []  # h_L(x+ε) - h_L(x) for embedding perturbations
        
        for di in range(N_RANDOM_DIRS):
            delta = torch.tensor(alpha * random_dirs[di],
                               dtype=inputs_embeds_base.dtype, device=device)
            
            inputs_embeds_pert = inputs_embeds_base.clone()
            inputs_embeds_pert[0, -1, :] += delta
            
            pert_h = [None]
            def pert_hook(module, input, output):
                if isinstance(output, tuple):
                    pert_h[0] = output[0][0, -1, :].detach().cpu().float().numpy()
                else:
                    pert_h[0] = output[0, -1, :].detach().cpu().float().numpy()
            
            handle = layers[target_layer].register_forward_hook(pert_hook)
            with torch.no_grad():
                try:
                    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
                    _ = model(inputs_embeds=inputs_embeds_pert, position_ids=position_ids,
                             attention_mask=attention_mask)
                except:
                    pass
            handle.remove()
            
            if pert_h[0] is not None:
                diff = pert_h[0] - base_h_val
                if not np.isnan(diff).any():
                    delta_h_embed.append(diff)
            
            if (di + 1) % 10 == 0:
                torch.cuda.empty_cache()
        
        # SVD of embedding-level tangent vectors
        if len(delta_h_embed) < 5:
            print(f"    Not enough valid tangent vectors, skipping")
            continue
        
        M_embed = np.array(delta_h_embed)
        M_embed_centered = M_embed - M_embed.mean(axis=0)
        U_e, s_e, Vt_e = np.linalg.svd(M_embed_centered, full_matrices=False)
        
        total_e = np.sum(s_e**2)
        cum_e = np.cumsum(s_e**2) / total_e if total_e > 0 else np.zeros(len(s_e))
        dim90_embed = int(np.searchsorted(cum_e, 0.90) + 1) if total_e > 0 else 0
        top1_embed = float(s_e[0]**2 / total_e) if total_e > 0 else 0
        
        print(f"    Embedding tangent space: dim90={dim90_embed}, top1={top1_embed:.4f}")
        
        # Step 3: Inject at MIDDLE layer — this gives partial Jacobian
        print(f"    Computing mid-layer tangent space (inject at L{mid_layer})...")
        
        # Get hidden state norm at mid layer for scaling
        mid_h = [None]
        def mid_base_hook(module, input, output):
            if isinstance(output, tuple):
                mid_h[0] = output[0][0, -1, :].detach().cpu().float().numpy()
            else:
                mid_h[0] = output[0, -1, :].detach().cpu().float().numpy()
        
        handle = layers[mid_layer].register_forward_hook(mid_base_hook)
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        handle.remove()
        
        if mid_h[0] is None:
            print(f"    Failed to get mid-layer hidden state, skipping")
            continue
        
        mid_h_val = mid_h[0]
        mid_norm = np.linalg.norm(mid_h_val)
        alpha_mid = 0.01 * mid_norm
        
        delta_h_mid = []
        
        for di in range(N_RANDOM_DIRS):
            delta = torch.tensor(alpha_mid * random_dirs[di],
                               dtype=torch.float32, device=device)
            
            # Inject at mid layer using hook
            def make_perturb_hook(delta_vec):
                def perturb_hook(module, input, output):
                    if isinstance(output, tuple):
                        new_h = output[0].clone()
                        new_h[0, -1, :] += delta_vec.to(new_h.dtype).to(new_h.device)
                        return (new_h,) + output[1:]
                    new_h = output.clone()
                    new_h[0, -1, :] += delta_vec.to(new_h.dtype).to(new_h.device)
                    return new_h
                return perturb_hook
            
            target_h = [None]
            def target_hook(module, input, output):
                if isinstance(output, tuple):
                    target_h[0] = output[0][0, -1, :].detach().cpu().float().numpy()
                else:
                    target_h[0] = output[0, -1, :].detach().cpu().float().numpy()
            
            h1 = layers[mid_layer].register_forward_hook(make_perturb_hook(delta))
            h2 = layers[target_layer].register_forward_hook(target_hook)
            
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            h1.remove()
            h2.remove()
            
            if target_h[0] is not None:
                diff = target_h[0] - base_h_val
                if not np.isnan(diff).any():
                    delta_h_mid.append(diff)
            
            if (di + 1) % 10 == 0:
                torch.cuda.empty_cache()
        
        # SVD of mid-layer tangent vectors
        if len(delta_h_mid) >= 5:
            M_mid = np.array(delta_h_mid)
            M_mid_centered = M_mid - M_mid.mean(axis=0)
            U_m, s_m, Vt_m = np.linalg.svd(M_mid_centered, full_matrices=False)
            
            total_m = np.sum(s_m**2)
            cum_m = np.cumsum(s_m**2) / total_m if total_m > 0 else np.zeros(len(s_m))
            dim90_mid = int(np.searchsorted(cum_m, 0.90) + 1) if total_m > 0 else 0
            top1_mid = float(s_m[0]**2 / total_m) if total_m > 0 else 0
            
            print(f"    Mid-layer tangent space: dim90={dim90_mid}, top1={top1_mid:.4f}")
        else:
            Vt_m = None
            dim90_mid = None
            top1_mid = None
            print(f"    Not enough mid-layer tangent vectors")
        
        # Step 4: Compare the two tangent spaces
        print(f"\n    ===== Tangent Space Comparison =====")
        
        k = min(10, min(Vt_e.shape[0], Vt_m.shape[0] if Vt_m is not None else 100))
        
        # Compare embedding vs mid-layer tangent spaces
        if Vt_m is not None:
            # Principal angles
            sub1 = Vt_e[:k]
            sub2 = Vt_m[:k]
            M_cross = sub1 @ sub2.T
            _, cos_angles, _ = np.linalg.svd(M_cross)
            cos_angles = np.clip(np.abs(cos_angles), 0, 1)
            angles_deg = np.degrees(np.arccos(cos_angles))
            
            print(f"    Principal angles (embed vs mid, k={k}):")
            print(f"      Mean: {np.mean(angles_deg):.1f}°, Max: {np.max(angles_deg):.1f}°")
            print(f"      Angles: {[f'{a:.1f}°' for a in angles_deg[:5]]}")
            
            # Overlap (subspace similarity)
            overlap = np.sum(cos_angles**2) / k
            print(f"    Subspace overlap: {overlap:.4f}")
            
            # cos(v1_embed, v1_mid)
            cos_v1 = abs(np.dot(Vt_e[0], Vt_m[0]))
            print(f"    |cos(v1_embed, v1_mid)|: {cos_v1:.4f}")
        
        # Key comparison: is the embedding tangent space the SAME as
        # what we'd get from Jacobian chain analysis?
        # If S(x) = Im(J_full(x)) where J_full = J_L ··· J_1,
        # then embedding injection should give the same subspace
        # as single-layer injection (at layer 0) propagated to the end.
        
        # The embedding injection already IS the full Jacobian chain!
        # So S(x) from embedding = Im(J_full(x))
        
        # But the mid-layer injection gives Im(J_L ··· J_{mid+1}),
        # which is only the PARTIAL chain from mid to end.
        
        # If the full chain has the same effective dimension as the partial chain,
        # it means the early layers don't contribute much to the tangent structure.
        
        result = {
            'text': text,
            'embed_dim90': dim90_embed,
            'embed_top1': top1_embed,
            'mid_dim90': dim90_mid,
            'mid_top1': top1_mid,
            'embed_n_valid': len(delta_h_embed),
            'mid_n_valid': len(delta_h_mid) if delta_h_mid else 0,
        }
        if Vt_m is not None:
            result['principal_angles'] = angles_deg.tolist()
            result['subspace_overlap'] = float(overlap)
            result['cos_v1'] = float(cos_v1)
        
        results.append(result)
    
    # ===== Summary =====
    print(f"\n  ===== 44B SUMMARY =====")
    for r in results:
        print(f"  '{r['text'][:40]}': embed_dim90={r['embed_dim90']}, "
              f"mid_dim90={r.get('mid_dim90', 'N/A')}, "
              f"overlap={r.get('subspace_overlap', 'N/A'):.4f}" if 'subspace_overlap' in r else
              f"  '{r['text'][:40]}': embed_dim90={r['embed_dim90']}, mid_dim90={r.get('mid_dim90', 'N/A')}")
    
    print(f"\n  KEY INSIGHT:")
    print(f"  The embedding-level tangent space IS Im(J_full(x)) by definition.")
    print(f"  The mid-level tangent space is Im(J_partial(x)).")
    print(f"  If they're similar → early layers don't change the tangent structure much")
    print(f"  If they differ → each layer contributes to shaping the tangent space")
    
    # Save results
    out_path = f"tests/glm5_temp/phase44B_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    
    release_model(model)
    return results


# ============================================================================
# 44C: h(x) Readout — What Does W_U·h(x) Encode?
# ============================================================================

def run_44C(model_name):
    """
    THE critical experiment: Does h(x) encode semantic information?
    
    Previous phases studied S(x) = tangent space. But the output is determined
    by W_U · h(x), not by W_U · S(x). So we need to check:
    
    1. For semantic contrast pairs (A, B), compute Δh = h_L(B) - h_L(A)
    2. Compute Δlogits = W_U · Δh — how does the semantic difference affect logits?
    3. Check: does Δh align with the margin direction? (cos similarity)
    4. Check: which tokens' logits are most affected by Δh?
    
    If Δlogits changes semantically relevant tokens → h(x) encodes semantics
    If Δlogits changes random tokens → semantic signal is elsewhere
    """
    print(f"\n{'='*70}")
    print("44C: h(x) Readout — What Does W_U·h(x) Encode?")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    W_U = get_W_U(model)  # [vocab_size, d_model]
    vocab_size = W_U.shape[0]
    print(f"  W_U shape: {W_U.shape}")
    
    results = {}
    
    for contrast_type, pairs in SEMANTIC_CONTRAST_PAIRS.items():
        print(f"\n  ===== Contrast Type: {contrast_type} =====")
        type_results = []
        
        for sent_A, sent_B in pairs:
            # Get h_L for both sentences
            inputs_A = tokenizer(sent_A, return_tensors="pt", truncation=True, max_length=64)
            inputs_B = tokenizer(sent_B, return_tensors="pt", truncation=True, max_length=64)
            
            input_ids_A = inputs_A["input_ids"].to(device)
            attention_mask_A = inputs_A["attention_mask"].to(device)
            input_ids_B = inputs_B["input_ids"].to(device)
            attention_mask_B = inputs_B["attention_mask"].to(device)
            
            # Get hidden states at last layer
            with torch.no_grad():
                outputs_A = model(input_ids=input_ids_A, attention_mask=attention_mask_A,
                                output_hidden_states=True)
                outputs_B = model(input_ids=input_ids_B, attention_mask=attention_mask_B,
                                output_hidden_states=True)
            
            h_A = outputs_A.hidden_states[-1][0, -1, :].detach().cpu().float().numpy()
            h_B = outputs_B.hidden_states[-1][0, -1, :].detach().cpu().float().numpy()
            
            # Also get logits for reference
            logits_A = outputs_A.logits[0, -1, :].detach().cpu().float().numpy()
            logits_B = outputs_B.logits[0, -1, :].detach().cpu().float().numpy()
            
            del outputs_A, outputs_B
            torch.cuda.empty_cache()
            
            # Compute Δh and Δlogits
            delta_h = h_B - h_A
            delta_logits = logits_B - logits_A  # This IS W_U · Δh (approximately)
            
            # Also compute W_U · Δh directly to verify
            delta_logits_computed = W_U @ delta_h
            cos_delta_logits = np.dot(delta_logits, delta_logits_computed) / (
                np.linalg.norm(delta_logits) * np.linalg.norm(delta_logits_computed) + 1e-10)
            
            # ===== Analysis 1: Δh properties =====
            delta_h_norm = np.linalg.norm(delta_h)
            h_A_norm = np.linalg.norm(h_A)
            h_B_norm = np.linalg.norm(h_B)
            
            # Relative magnitude of Δh
            rel_delta = delta_h_norm / ((h_A_norm + h_B_norm) / 2)
            
            # ===== Analysis 2: Margin direction alignment =====
            # margin direction = direction in h-space that separates the top tokens
            top_token_A = np.argmax(logits_A)
            top_token_B = np.argmax(logits_B)
            
            # W_U rows for top tokens
            w_topA = W_U[top_token_A]
            w_topB = W_U[top_token_B]
            margin_dir = w_topB - w_topA
            margin_norm = np.linalg.norm(margin_dir)
            
            if margin_norm > 1e-10:
                margin_dir_norm = margin_dir / margin_norm
                cos_delta_margin = abs(np.dot(delta_h, margin_dir_norm) / (delta_h_norm + 1e-10))
            else:
                cos_delta_margin = 0.0
            
            # ===== Analysis 3: Which tokens' logits change most? =====
            top_changed_tokens = np.argsort(np.abs(delta_logits))[-10:][::-1]
            top_changed_values = np.abs(delta_logits)[top_changed_tokens]
            
            # Decode the changed tokens
            changed_token_strs = []
            for tid in top_changed_tokens[:5]:
                tok_str = safe_decode(tokenizer, int(tid))
                logit_change = delta_logits[tid]
                changed_token_strs.append(f"{tok_str}({logit_change:+.2f})")
            
            # ===== Analysis 4: h(x) norm and direction =====
            cos_hA_hB = np.dot(h_A, h_B) / (h_A_norm * h_B_norm + 1e-10)
            
            # ===== Analysis 5: Compare Δh with Δh from perturbation =====
            # From Phase 43, we know S(x) dim90 ≈ 30-40
            # If Δh is mostly in S(x), then semantic differences are "tangent"
            # If Δh is mostly in the base trajectory, they're "structural"
            
            # We can check: is Δh large relative to typical perturbation effects?
            # From Phase 42-43, typical δh_L for α=0.1*||h|| perturbation was ~1-10
            # Δh is the full semantic difference
            
            result = {
                'sent_A': sent_A,
                'sent_B': sent_B,
                'h_A_norm': float(h_A_norm),
                'h_B_norm': float(h_B_norm),
                'delta_h_norm': float(delta_h_norm),
                'rel_delta': float(rel_delta),
                'cos_hA_hB': float(cos_hA_hB),
                'cos_delta_margin': float(cos_delta_margin),
                'top_token_A': safe_decode(tokenizer, int(top_token_A)),
                'top_token_B': safe_decode(tokenizer, int(top_token_B)),
                'top5_changed': changed_token_strs[:5],
                'cos_delta_logits_verification': float(cos_delta_logits),
            }
            type_results.append(result)
            
            print(f"\n    A: '{sent_A}'")
            print(f"    B: '{sent_B}'")
            print(f"    ||h_A||={h_A_norm:.1f}, ||h_B||={h_B_norm:.1f}, ||Δh||={delta_h_norm:.1f}")
            print(f"    rel_delta(Δh/avg_h)={rel_delta:.4f}")
            print(f"    cos(h_A, h_B)={cos_hA_hB:.4f}")
            print(f"    cos(Δh, margin)={cos_delta_margin:.4f}")
            print(f"    Top token A: {safe_decode(tokenizer, int(top_token_A))}")
            print(f"    Top token B: {safe_decode(tokenizer, int(top_token_B))}")
            print(f"    Top-5 logit changes: {', '.join(changed_token_strs[:5])}")
        
        results[contrast_type] = type_results
    
    # ===== Cross-type Analysis =====
    print(f"\n  ===== 44C CROSS-TYPE ANALYSIS =====")
    
    for ctype, type_results in results.items():
        cos_margins = [r['cos_delta_margin'] for r in type_results]
        rel_deltas = [r['rel_delta'] for r in type_results]
        cos_hh = [r['cos_hA_hB'] for r in type_results]
        
        print(f"\n  {ctype}:")
        print(f"    cos(Δh, margin): mean={np.mean(cos_margins):.4f}, std={np.std(cos_margins):.4f}")
        print(f"    rel_delta(Δh/h): mean={np.mean(rel_deltas):.4f}, std={np.std(rel_deltas):.4f}")
        print(f"    cos(h_A, h_B):   mean={np.mean(cos_hh):.4f}, std={np.std(cos_hh):.4f}")
    
    # ===== KEY COMPARISON: h(x) vs S(x) as semantic carrier =====
    print(f"\n  ===== KEY QUESTION: Is semantic information in h(x) or S(x)? =====")
    print(f"  If cos(Δh, margin) is HIGH → h(x) directly encodes semantic differences")
    print(f"  If cos(Δh, margin) is LOW  → semantic signal needs W_U to extract from h(x)")
    print(f"  From Phase 43: S(x) bases had |cos(margin)| ≈ 0.02-0.08")
    
    all_cos_margins = []
    for ctype, type_results in results.items():
        all_cos_margins.extend([r['cos_delta_margin'] for r in type_results])
    
    print(f"\n  Overall cos(Δh, margin): mean={np.mean(all_cos_margins):.4f}")
    print(f"  This is {'HIGHER' if np.mean(all_cos_margins) > 0.1 else 'SIMILAR'} than S(x)'s cos(margin) ≈ 0.02-0.08")
    
    if np.mean(all_cos_margins) > 0.1:
        print(f"  → Δh (trajectory difference) aligns better with margin than S(x) bases")
        print(f"  → Semantic information is in h(x) TRAJECTORY, not in tangent structure S(x)")
    else:
        print(f"  → Δh does NOT align with margin direction")
        print(f"  → Semantic information is distributed across many directions in h(x)")
    
    # Save results
    out_path = f"tests/glm5_temp/phase44C_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    
    release_model(model)
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 44: Trajectory Manifold Geometry")
    parser.add_argument("--model", type=str, default="deepseek7b",
                       choices=["deepseek7b", "glm4", "qwen3"])
    parser.add_argument("--exp", type=int, default=1, choices=[1, 2, 3],
                       help="1=44A trajectory manifold, 2=44B tangent verification, 3=44C readout")
    args = parser.parse_args()
    
    if args.exp == 1:
        run_44A(args.model)
    elif args.exp == 2:
        run_44B(args.model)
    elif args.exp == 3:
        run_44C(args.model)


if __name__ == "__main__":
    main()
