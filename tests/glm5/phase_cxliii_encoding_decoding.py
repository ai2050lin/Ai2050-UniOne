#!/usr/bin/env python3
"""
Phase CXLIII: Encoding-Decoding Geometry Framework (P631-P636)
Focus: Mathematical formalization of the encoding-decoding geometry discovered in Phase CXLII

Key insight from Phase CXLII: W_U PCs are the key bridge for h->gap prediction.
GLM4 h@W_U_PCs->gap r=0.893, Non-oracle->prob r=0.953 (95.3% of oracle!)

P631: W_U projection manifold structure - Is W_U a low-dimensional manifold?
P632: Encoding functor - mathematical form of x->h mapping
P633: Decoding functor - W_U·h in low-dimensional subspace
P634: Geodesic distance on manifold - Is gap a geodesic distance?
P635: Cross-model encoding-decoding invariants
P636: Unified intelligence theory framework v1
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import json
import time
import argparse
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, MDS
from sklearn.metrics import pairwise_distances
from model_utils import load_model, get_model_info

TEST_TEXTS = [
    "The quantum computer solved the complex optimization problem in seconds.",
    "She walked through the ancient forest, listening to birds singing.",
    "The stock market crashed after the central bank raised interest rates.",
    "Artificial intelligence is transforming healthcare diagnostics.",
    "The musician played a haunting melody on the violin.",
    "Climate change threatens coastal cities with rising sea levels.",
    "The philosopher questioned the nature of consciousness and free will.",
    "A new vaccine was developed to combat the emerging virus.",
    "The architect designed a sustainable building with solar panels.",
    "The detective gathered clues to solve the mysterious case.",
    "The chef prepared a delicious meal using local ingredients.",
    "The spacecraft orbited Mars, collecting data for scientists.",
    "The poet wrote verses about love and loss under moonlight.",
    "The economist predicted a recession based on market indicators.",
    "The teacher encouraged students to think critically about history.",
    "The programmer debugged the code to fix the memory leak.",
    "The artist painted a vibrant landscape with bold brushstrokes.",
    "The biologist discovered a new species in the rainforest.",
    "The judge ruled in favor of the plaintiff after hearing arguments.",
    "The engineer built a bridge that could withstand earthquakes.",
    "The novelist crafted a story about time travel and paradox.",
    "The doctor diagnosed the patient with a rare genetic disorder.",
    "The astronaut floated weightlessly in the International Space Station.",
    "The historian analyzed primary sources from the Renaissance.",
    "The mathematician proved a theorem about prime numbers.",
    "The chemist synthesized a new compound in the laboratory.",
    "The diplomat negotiated a peace agreement between nations.",
    "The journalist reported on the election results from the capital.",
    "The firefighter rescued a family from the burning building.",
    "The musician composed a symphony that moved the audience to tears.",
    "The researcher published a groundbreaking paper on dark matter.",
    "The sailor navigated through the storm using celestial observations.",
    "The developer created an app that simplifies project management.",
    "The photographer captured the sunset over the mountain range.",
    "The botanist studied the rare orchid in its natural habitat.",
    "The librarian organized the archives for public access.",
    "The cyclist trained rigorously for the upcoming championship.",
    "The geologist identified a new mineral formation in the cave.",
    "The translator rendered the ancient text into modern language.",
    "The volunteer distributed food to families affected by the flood.",
]


def compute_features(model, tokenizer, device, texts):
    """Compute comprehensive features for all texts."""
    # Get W_U
    if hasattr(model, 'lm_head'):
        W_U = model.lm_head.weight.detach().cpu().float().numpy()
    else:
        W_U = model.get_output_embeddings().weight.detach().cpu().float().numpy()
    
    # Get model config
    if hasattr(model.config, 'hidden_size'):
        d_model = model.config.hidden_size
    elif hasattr(model.config, 'd_model'):
        d_model = model.config.d_model
    else:
        d_model = 2560
    
    if hasattr(model.config, 'num_hidden_layers'):
        n_layers = model.config.num_hidden_layers
    elif hasattr(model.config, 'n_layer'):
        n_layers = model.config.n_layer
    else:
        n_layers = 32
    
    # W_U PCA
    pca_wu = PCA(n_components=min(200, W_U.shape[0], W_U.shape[1]))
    pca_wu.fit(W_U)
    W_U_pcs = pca_wu.components_  # [n_components, d_model]
    W_U_explained = pca_wu.explained_variance_ratio_
    
    all_features = []
    
    for i, text in enumerate(texts):
        if (i+1) % 10 == 0:
            print(f"  Processing text {i+1}/{len(texts)}...")
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # Get all hidden states
        all_hidden = []
        for hs in outputs.hidden_states:
            all_hidden.append(hs[0, -1, :].cpu().float().numpy())
        
        h = all_hidden[-1]
        logits = outputs.logits[0, -1, :].cpu().float().numpy()
        
        # Token info
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        logit_gap = logits[top1_idx] - logits[top2_idx]
        prob = 1.0 / (1.0 + np.exp(-logit_gap))
        
        # W_U structure
        W_U_top1 = W_U[top1_idx]
        W_U_top2 = W_U[top2_idx]
        Delta_W = W_U_top1 - W_U_top2
        
        # h projection onto W_U PCs
        h_wu_proj = W_U_pcs @ h  # [n_components]
        
        # Delta_W projection onto W_U PCs
        dw_wu_proj = W_U_pcs @ Delta_W  # [n_components]
        
        all_features.append({
            'h': h,
            'all_hidden': all_hidden,
            'logit_gap': logit_gap,
            'prob': prob,
            'top1_idx': top1_idx,
            'top2_idx': top2_idx,
            'Delta_W': Delta_W,
            'W_U_top1': W_U_top1,
            'W_U_top2': W_U_top2,
            'h_wu_proj': h_wu_proj,
            'dw_wu_proj': dw_wu_proj,
            'h_norm': np.linalg.norm(h),
            'text_idx': i,
        })
    
    return all_features, W_U, W_U_pcs, W_U_explained, d_model, n_layers


def experiment_p631(all_features, W_U, W_U_pcs, W_U_explained, d_model, model_name):
    """P631: W_U Projection Manifold Structure
    Is W_U a low-dimensional manifold? Test:
    1. Intrinsic dimensionality of W_U (correlation dimension)
    2. h trajectory on W_U manifold
    3. Gap as distance on manifold
    """
    print(f"\n{'='*60}")
    print(f"P631: W_U Projection Manifold Structure ({model_name})")
    print(f"{'='*60}")
    
    n_vocab = W_U.shape[0]
    
    # 1. W_U intrinsic dimensionality via PCA curve
    cumvar = np.cumsum(W_U_explained)
    dims_50 = np.searchsorted(cumvar, 0.50)
    dims_90 = np.searchsorted(cumvar, 0.90)
    dims_95 = np.searchsorted(cumvar, 0.95)
    dims_99 = np.searchsorted(cumvar, 0.99)
    
    print(f"\n1. W_U PCA Intrinsic Dimensionality:")
    print(f"   50% variance: {dims_50} PCs ({dims_50/d_model*100:.1f}% of d_model)")
    print(f"   90% variance: {dims_90} PCs ({dims_90/d_model*100:.1f}% of d_model)")
    print(f"   95% variance: {dims_95} PCs ({dims_95/d_model*100:.1f}% of d_model)")
    print(f"   99% variance: {dims_99} PCs ({dims_99/d_model*100:.1f}% of d_model)")
    
    # 2. Correlation dimension of W_U
    # Sample subset for computational efficiency
    n_sample = min(2000, n_vocab)
    sample_idx = np.random.choice(n_vocab, n_sample, replace=False)
    W_U_sample = W_U[sample_idx]
    W_U_sample = W_U_sample / (np.linalg.norm(W_U_sample, axis=1, keepdims=True) + 1e-10)
    
    dists = pdist(W_U_sample)
    
    # Correlation dimension: C(r) ~ r^d
    r_values = np.percentile(dists, [5, 10, 20, 30, 40, 50, 60, 70, 80])
    C_values = []
    for r in r_values:
        C_values.append(np.mean(dists < r))
    
    # Log-log fit for correlation dimension
    log_r = np.log(r_values[1:-1] + 1e-10)
    log_C = np.log(np.array(C_values[1:-1]) + 1e-10)
    
    if len(log_r) > 2:
        slope, intercept, r_val, p_val, se = stats.linregress(log_r, log_C)
        corr_dim = slope
        print(f"\n2. Correlation Dimension of W_U:")
        print(f"   Correlation dimension d_corr = {corr_dim:.2f}")
        print(f"   R-squared = {r_val**2:.4f}")
        print(f"   (d_model = {d_model}, d_corr/d_model = {corr_dim/d_model:.4f})")
    else:
        corr_dim = d_model
        r_val = 0
        print(f"\n2. Correlation Dimension: Insufficient data points")
    
    # 3. h trajectory on W_U PC space
    h_projs = np.array([f['h_wu_proj'] for f in all_features])  # [n_texts, n_components]
    
    # How many PCs needed to capture h@W_U variance?
    h_proj_pca = PCA()
    h_proj_pca.fit(h_projs)
    h_cumvar = np.cumsum(h_proj_pca.explained_variance_ratio_)
    h_dims_50 = np.searchsorted(h_cumvar, 0.50)
    h_dims_90 = np.searchsorted(h_cumvar, 0.90)
    h_dims_95 = np.searchsorted(h_cumvar, 0.95)
    
    print(f"\n3. h Projection onto W_U PCs - Effective Dimensionality:")
    print(f"   h@W_U_PCs: 50% var in {h_dims_50} dims, 90% in {h_dims_90}, 95% in {h_dims_95}")
    
    # 4. Gap as distance on W_U manifold
    # Test: Is gap related to distance between h and W_U rows in PC space?
    gaps = np.array([f['logit_gap'] for f in all_features])
    
    # Distance from h to top1 and top2 in W_U PC space
    dist_to_top1 = []
    dist_to_top2 = []
    diff_dist = []
    for f in all_features:
        top1_proj = W_U_pcs @ f['W_U_top1']
        top2_proj = W_U_pcs @ f['W_U_top2']
        d1 = np.linalg.norm(f['h_wu_proj'] - top1_proj)
        d2 = np.linalg.norm(f['h_wu_proj'] - top2_proj)
        dist_to_top1.append(d1)
        dist_to_top2.append(d2)
        diff_dist.append(d2 - d1)  # diff_dist > 0 means h closer to top1
    
    dist_to_top1 = np.array(dist_to_top1)
    dist_to_top2 = np.array(dist_to_top2)
    diff_dist = np.array(diff_dist)
    
    r_diff, _ = stats.pearsonr(diff_dist, gaps)
    r_d1, _ = stats.pearsonr(dist_to_top1, gaps)
    r_d2, _ = stats.pearsonr(dist_to_top2, gaps)
    
    print(f"\n4. Gap as Distance on W_U Manifold:")
    print(f"   d(h,top2)-d(h,top1) → gap: r={r_diff:.4f}")
    print(f"   d(h,top1) → gap: r={r_d1:.4f}")
    print(f"   d(h,top2) → gap: r={r_d2:.4f}")
    
    # 5. Isomap on W_U to test manifold hypothesis
    # Use small subset for Isomap (computationally expensive)
    n_iso = min(500, n_vocab)
    iso_idx = np.random.choice(n_vocab, n_iso, replace=False)
    W_U_iso = W_U[iso_idx]
    W_U_iso_norm = W_U_iso / (np.linalg.norm(W_U_iso, axis=1, keepdims=True) + 1e-10)
    
    try:
        iso = Isomap(n_components=min(20, n_iso-1), n_neighbors=min(15, n_iso-1))
        W_U_iso_embedded = iso.fit_transform(W_U_iso_norm)
        iso_reconstruction_error = iso.reconstruction_error()
        print(f"\n5. Isomap Manifold Test:")
        print(f"   Isomap reconstruction error: {iso_reconstruction_error:.4f}")
        print(f"   (Lower = more manifold-like)")
    except Exception as e:
        iso_reconstruction_error = -1
        print(f"\n5. Isomap failed: {e}")
    
    # 6. Manifold curvature test: locally flat vs curved
    # Pick random triplets and check triangle inequality deviation
    n_triplets = 1000
    triplet_idx = np.random.choice(n_sample, (n_triplets, 3), replace=True)
    D = squareform(dists)
    
    triangle_ratios = []
    for t in triplet_idx:
        i, j, k = t
        d_ij = D[i, j] if i < n_sample and j < n_sample else 0
        d_ik = D[i, k] if i < n_sample and k < n_sample else 0
        d_jk = D[j, k] if j < n_sample and k < n_sample else 0
        if d_jk > 0 and d_ij > 0 and d_ik > 0:
            # In flat space: d_ik <= d_ij + d_jk (triangle inequality)
            # Ratio d_ik / (d_ij + d_jk) should be close to some value
            triangle_ratios.append(d_ik / (d_ij + d_jk))
    
    if triangle_ratios:
        mean_ratio = np.mean(triangle_ratios)
        std_ratio = np.std(triangle_ratios)
        print(f"\n6. Triangle Inequality Ratios (manifold curvature proxy):")
        print(f"   Mean ratio d_ik/(d_ij+d_jk): {mean_ratio:.4f} ± {std_ratio:.4f}")
        print(f"   (0.5 = flat, >0.5 = positively curved, <0.5 = negatively curved)")
    
    results = {
        'dims_50': int(dims_50), 'dims_90': int(dims_90), 
        'dims_95': int(dims_95), 'dims_99': int(dims_99),
        'correlation_dimension': float(corr_dim),
        'corr_dim_r2': float(r_val**2) if hasattr(r_val, '__float__') else 0,
        'h_proj_dims_50': int(h_dims_50), 'h_proj_dims_90': int(h_dims_90),
        'h_proj_dims_95': int(h_dims_95),
        'gap_as_dist_r': float(r_diff),
        'isomap_error': float(iso_reconstruction_error),
    }
    
    return results


def experiment_p632(all_features, W_U, W_U_pcs, W_U_explained, d_model, n_layers, model_name):
    """P632: Encoding Functor - Mathematical Form of x->h Mapping
    Test if the encoding x->h can be described as a simple mathematical operation:
    1. Layer-wise trajectory in W_U PC space
    2. Linearity of h transformation across layers
    3. Residual connection structure
    """
    print(f"\n{'='*60}")
    print(f"P632: Encoding Functor ({model_name})")
    print(f"{'='*60}")
    
    gaps = np.array([f['logit_gap'] for f in all_features])
    probs = np.array([f['prob'] for f in all_features])
    
    # 1. Layer-wise gap trajectory
    print(f"\n1. Layer-wise Gap Trajectory:")
    layer_gaps = []
    for layer_idx in range(n_layers):
        layer_gap_list = []
        for f in all_features:
            if layer_idx < len(f['all_hidden']):
                h_l = f['all_hidden'][layer_idx]
                gap_l = np.dot(h_l, f['Delta_W'])
                layer_gap_list.append(gap_l)
        layer_gaps.append(layer_gap_list)
    
    layer_gaps = np.array(layer_gaps)  # [n_layers, n_texts]
    
    # Test: Does final gap = sum of layer increments?
    if n_layers > 1:
        increments = np.diff(layer_gaps, axis=0)  # [n_layers-1, n_texts]
        sum_increments = np.sum(increments, axis=0)
        final_gaps = layer_gaps[-1]
        initial_gaps = layer_gaps[0]
        
        # gap_final = gap_initial + sum(inc)
        r_sum, _ = stats.pearsonr(sum_increments + initial_gaps, final_gaps)
        print(f"   gap_final = gap_initial + Σδ: r={r_sum:.4f}")
        
        # Per-layer increment contribution
        mean_increments = np.mean(np.abs(increments), axis=1)
        top_contrib_layers = np.argsort(mean_increments)[::-1][:5]
        print(f"   Top-5 contributing layers: {top_contrib_layers.tolist()}")
        print(f"   Their |increment|: {mean_increments[top_contrib_layers].tolist()}")
    
    # 2. Linearity test: Can h(l+1) = A·h(l) + b?
    print(f"\n2. Linearity of Layer Transformation:")
    linearity_scores = []
    for layer_idx in range(min(n_layers-1, len(all_features[0]['all_hidden'])-1)):
        h_l_list = []
        h_l1_list = []
        for f in all_features:
            if layer_idx+1 < len(f['all_hidden']):
                h_l_list.append(f['all_hidden'][layer_idx])
                h_l1_list.append(f['all_hidden'][layer_idx+1])
        
        if len(h_l_list) > 5:
            H_l = np.array(h_l_list)
            H_l1 = np.array(h_l1_list)
            
            # Fit linear model: h(l+1) = A @ h(l)
            # Use ridge regression for stability
            reg = Ridge(alpha=1.0)
            reg.fit(H_l, H_l1)
            H_pred = reg.predict(H_l)
            
            # Per-text cosine similarity
            cos_sims = []
            for j in range(len(h_l_list)):
                cs = np.dot(H_pred[j], H_l1[j]) / (np.linalg.norm(H_pred[j]) * np.linalg.norm(H_l1[j]) + 1e-10)
                cos_sims.append(cs)
            
            mean_cos = np.mean(cos_sims)
            linearity_scores.append(mean_cos)
            
            if layer_idx < 3 or layer_idx >= n_layers - 3:
                print(f"   Layer {layer_idx}->{layer_idx+1}: linear fit cosine={mean_cos:.4f}")
    
    if linearity_scores:
        print(f"   Mean linearity across layers: {np.mean(linearity_scores):.4f} ± {np.std(linearity_scores):.4f}")
        print(f"   Min/Max linearity: {np.min(linearity_scores):.4f} / {np.max(linearity_scores):.4f}")
    
    # 3. Residual connection structure
    print(f"\n3. Residual Connection Structure:")
    # In Transformer: h(l+1) = h(l) + f_l(h(l))
    # So f_l(h(l)) = h(l+1) - h(l) = residual stream update
    
    residual_norms = []
    h_norms = []
    for layer_idx in range(min(n_layers-1, len(all_features[0]['all_hidden'])-1)):
        res_norms = []
        hn_norms = []
        for f in all_features:
            if layer_idx+1 < len(f['all_hidden']):
                h_l = f['all_hidden'][layer_idx]
                h_l1 = f['all_hidden'][layer_idx+1]
                res = h_l1 - h_l
                res_norms.append(np.linalg.norm(res))
                hn_norms.append(np.linalg.norm(h_l))
        if res_norms:
            residual_norms.append(np.mean(res_norms))
            h_norms.append(np.mean(hn_norms))
    
    if residual_norms:
        ratios = np.array(residual_norms) / (np.array(h_norms) + 1e-10)
        print(f"   |residual|/|h| ratio: mean={np.mean(ratios):.4f}, std={np.std(ratios):.4f}")
        print(f"   First 5 layers ratios: {ratios[:5].tolist()}")
        print(f"   Last 5 layers ratios: {ratios[-5:].tolist()}")
    
    # 4. Encoding functor: h in W_U PC space
    print(f"\n4. Encoding Functor in W_U PC Space:")
    # Track h(l) projection onto W_U PCs across layers
    layer_wu_projections = []
    for layer_idx in range(min(n_layers, len(all_features[0]['all_hidden']))):
        projs = []
        for f in all_features:
            h_l = f['all_hidden'][layer_idx]
            proj = W_U_pcs[:50] @ h_l  # Project to top-50 W_U PCs
            projs.append(proj)
        layer_wu_projections.append(np.array(projs))
    
    # Track how well layer projections predict final gap
    layer_gap_pred = []
    for layer_idx, projs in enumerate(layer_wu_projections):
        if len(projs) > 5:
            reg = Ridge(alpha=1.0)
            reg.fit(projs, gaps)
            pred = reg.predict(projs)
            r, _ = stats.pearsonr(pred, gaps)
            layer_gap_pred.append((layer_idx, r))
    
    if layer_gap_pred:
        print(f"   Layer-wise gap prediction (h@W_U_PCs):")
        for li, r in layer_gap_pred[:5]:
            print(f"     Layer {li}: r={r:.4f}")
        print(f"     ...")
        for li, r in layer_gap_pred[-3:]:
            print(f"     Layer {li}: r={r:.4f}")
    
    results = {
        'gap_sum_r': float(r_sum) if n_layers > 1 else 0,
        'mean_linearity': float(np.mean(linearity_scores)) if linearity_scores else 0,
        'residual_h_ratio': float(np.mean(ratios)) if len(ratios) > 0 else 0,
        'layer_gap_pred': [(int(li), float(r)) for li, r in layer_gap_pred[-5:]] if layer_gap_pred else [],
    }
    
    return results


def experiment_p633(all_features, W_U, W_U_pcs, W_U_explained, d_model, model_name):
    """P633: Decoding Functor - W_U·h in Low-Dimensional Subspace
    Test if W_U·h can be approximated in a low-dimensional subspace:
    1. Low-rank approximation of W_U
    2. Delta_W structure in PC space
    3. Optimal subspace dimension for gap prediction
    """
    print(f"\n{'='*60}")
    print(f"P633: Decoding Functor ({model_name})")
    print(f"{'='*60}")
    
    gaps = np.array([f['logit_gap'] for f in all_features])
    probs = np.array([f['prob'] for f in all_features])
    
    # 1. Low-rank W_U approximation for gap prediction
    print(f"\n1. Low-Rank W_U Approximation for Gap Prediction:")
    
    # Test different numbers of PCs
    n_pc_values = [5, 10, 20, 50, 100, 200]
    pc_r_values = []
    
    for n_pc in n_pc_values:
        if n_pc > len(W_U_pcs):
            continue
        # Reconstruct gap using only n_pc components
        pred_gaps = []
        for f in all_features:
            # gap = h · Delta_W ≈ (h · W_U_PCs[:n_pc].T) @ (W_U_PCs[:n_pc] @ Delta_W)
            h_pc = f['h_wu_proj'][:n_pc]  # h projected to n_pc components
            dw_pc = f['dw_wu_proj'][:n_pc]  # Delta_W projected to n_pc components
            approx_gap = np.dot(h_pc, dw_pc)
            pred_gaps.append(approx_gap)
        
        pred_gaps = np.array(pred_gaps)
        r, _ = stats.pearsonr(pred_gaps, gaps)
        pc_r_values.append((n_pc, r))
        print(f"   n_PCs={n_pc:3d}: gap approximation r={r:.4f}")
    
    # 2. Delta_W structure in W_U PC space
    print(f"\n2. Delta_W Structure in W_U PC Space:")
    dw_projs = np.array([f['dw_wu_proj'] for f in all_features])  # [n_texts, n_components]
    
    # How concentrated is Delta_W in PC space?
    dw_norms_full = np.array([np.linalg.norm(f['Delta_W']) for f in all_features])
    dw_norms_pc = np.linalg.norm(dw_projs, axis=1)
    
    # Ratio of PC-projected norm to full norm
    norm_ratios = dw_norms_pc / (dw_norms_full + 1e-10)
    print(f"   |Delta_W_PC|/|Delta_W| ratio: {np.mean(norm_ratios):.4f} ± {np.std(norm_ratios):.4f}")
    
    # Concentration of Delta_W in top PCs
    for n_pc in [5, 10, 20, 50]:
        if n_pc <= dw_projs.shape[1]:
            partial_norm = np.linalg.norm(dw_projs[:, :n_pc], axis=1)
            concentration = partial_norm / (dw_norms_pc + 1e-10)
            print(f"   Delta_W concentration in top-{n_pc} PCs: {np.mean(concentration):.4f}")
    
    # 3. Optimal subspace dimension via cross-validation
    print(f"\n3. Optimal Subspace Dimension:")
    # Simple hold-out: use first 30 texts for training, last 10 for testing
    n_train = 30
    n_test = len(all_features) - n_train
    
    h_projs = np.array([f['h_wu_proj'] for f in all_features])
    
    best_r = 0
    best_n_pc = 0
    for n_pc in [3, 5, 10, 15, 20, 30, 50, 100]:
        if n_pc > h_projs.shape[1]:
            continue
        X_train = h_projs[:n_train, :n_pc]
        X_test = h_projs[n_train:, :n_pc]
        y_train = gaps[:n_train]
        y_test = gaps[n_train:]
        
        reg = Ridge(alpha=1.0)
        reg.fit(X_train, y_train)
        pred = reg.predict(X_test)
        
        r, _ = stats.pearsonr(pred, y_test)
        if r > best_r:
            best_r = r
            best_n_pc = n_pc
        print(f"   n_PCs={n_pc:3d}: test r={r:.4f}")
    
    print(f"   Best: n_PCs={best_n_pc}, r={best_r:.4f}")
    
    # 4. Information-theoretic view: How many bits of gap are in each PC?
    print(f"\n4. Information per W_U PC:")
    # Correlation of each PC projection with gap
    pc_gap_corrs = []
    for pc_idx in range(min(50, h_projs.shape[1])):
        r, _ = stats.pearsonr(h_projs[:, pc_idx], gaps)
        pc_gap_corrs.append(r)
    
    # Sort by absolute correlation
    sorted_corrs = sorted(enumerate(pc_gap_corrs), key=lambda x: abs(x[1]), reverse=True)
    print(f"   Top-10 PCs by |correlation| with gap:")
    for rank, (pc_idx, r) in enumerate(sorted_corrs[:10]):
        print(f"     PC {pc_idx}: r={r:.4f}")
    
    # Cumulative R-squared of top-K PCs
    cum_r2 = 0
    for rank, (pc_idx, r) in enumerate(sorted_corrs[:50]):
        cum_r2 += r**2
        if rank in [4, 9, 19, 49]:
            print(f"   Top-{rank+1} PCs cumulative R-sq: {cum_r2:.4f}")
    
    results = {
        'pc_r_values': [(int(n), float(r)) for n, r in pc_r_values],
        'dw_norm_ratio': float(np.mean(norm_ratios)),
        'best_n_pc': int(best_n_pc),
        'best_test_r': float(best_r),
        'top10_pc_indices': [int(idx) for idx, _ in sorted_corrs[:10]],
        'top10_pc_corrs': [float(r) for _, r in sorted_corrs[:10]],
    }
    
    return results


def experiment_p634(all_features, W_U, W_U_pcs, W_U_explained, d_model, model_name):
    """P634: Geodesic Distance on Manifold - Is Gap a Geodesic Distance?
    Test if logit_gap corresponds to a geodesic distance on the W_U manifold:
    1. Euclidean vs geodesic distance for gap
    2. Local vs global geometry
    3. Curvature effects on gap
    """
    print(f"\n{'='*60}")
    print(f"P634: Geodesic Distance on Manifold ({model_name})")
    print(f"{'='*60}")
    
    gaps = np.array([f['logit_gap'] for f in all_features])
    probs = np.array([f['prob'] for f in all_features])
    
    # 1. Euclidean distance analysis
    print(f"\n1. Euclidean Distance in Different Spaces:")
    
    # In full d_model space
    eucl_full = []
    for f in all_features:
        d1 = np.linalg.norm(f['h'] - f['W_U_top1'])
        d2 = np.linalg.norm(f['h'] - f['W_U_top2'])
        eucl_full.append(d2 - d1)
    eucl_full = np.array(eucl_full)
    r_eucl_full, _ = stats.pearsonr(eucl_full, gaps)
    print(f"   d(h,top2)-d(h,top1) in full space → gap: r={r_eucl_full:.4f}")
    
    # In W_U PC space (50 PCs)
    eucl_pc = []
    for f in all_features:
        h_pc = f['h_wu_proj'][:50]
        top1_pc = W_U_pcs[:50] @ f['W_U_top1']
        top2_pc = W_U_pcs[:50] @ f['W_U_top2']
        d1 = np.linalg.norm(h_pc - top1_pc)
        d2 = np.linalg.norm(h_pc - top2_pc)
        eucl_pc.append(d2 - d1)
    eucl_pc = np.array(eucl_pc)
    r_eucl_pc, _ = stats.pearsonr(eucl_pc, gaps)
    print(f"   d(h,top2)-d(h,top1) in PC-50 space → gap: r={r_eucl_pc:.4f}")
    
    # 2. Inner product (dot product) is NOT Euclidean distance
    # gap = h·Delta_W = h·W_U[top1] - h·W_U[top2]
    # This is related to: d(h,top2)^2 - d(h,top1)^2 = |h-top2|^2 - |h-top1|^2
    #   = (|h|^2-2h*top2+|top2|^2) - (|h|^2-2h*top1+|top1|^2)
    #   = 2(h*top1-h*top2) + (|top1|^2-|top2|^2)
    # So: gap = h*(top1-top2) = 0.5*(d^2(h,top2)-d^2(h,top1)) - 0.5*(|top1|^2-|top2|^2)
    
    print(f"\n2. Gap as Squared Distance Difference:")
    sq_diff = []
    for f in all_features:
        d1_sq = np.sum((f['h'] - f['W_U_top1'])**2)
        d2_sq = np.sum((f['h'] - f['W_U_top2'])**2)
        sq_diff.append(d2_sq - d1_sq)
    sq_diff = np.array(sq_diff)
    r_sq_diff, _ = stats.pearsonr(sq_diff, gaps)
    print(f"   d_sq(h,top2)-d_sq(h,top1) -> gap: r={r_sq_diff:.4f}")
    
    # Corrected: gap = 0.5*(sq_diff) - 0.5*(|top1|^2-|top2|^2)
    norm_correction = []
    for f in all_features:
        norm_diff = np.linalg.norm(f['W_U_top1'])**2 - np.linalg.norm(f['W_U_top2'])**2
        norm_correction.append(norm_diff)
    norm_correction = np.array(norm_correction)
    
    corrected = 0.5 * sq_diff - 0.5 * norm_correction
    r_corrected, _ = stats.pearsonr(corrected, gaps)
    r_norm_corr, _ = stats.pearsonr(norm_correction, gaps)
    print(f"   |top1|^2-|top2|^2 -> gap: r={r_norm_corr:.4f}")
    print(f"   0.5*sq_diff - 0.5*norm_correction → gap: r={r_corrected:.4f}")
    
    # 3. Angular distance
    print(f"\n3. Angular Distance Analysis:")
    angular_diff = []
    for f in all_features:
        cos1 = np.dot(f['h'], f['W_U_top1']) / (np.linalg.norm(f['h']) * np.linalg.norm(f['W_U_top1']) + 1e-10)
        cos2 = np.dot(f['h'], f['W_U_top2']) / (np.linalg.norm(f['h']) * np.linalg.norm(f['W_U_top2']) + 1e-10)
        angle1 = np.arccos(np.clip(cos1, -1, 1))
        angle2 = np.arccos(np.clip(cos2, -1, 1))
        angular_diff.append(angle2 - angle1)
    angular_diff = np.array(angular_diff)
    r_angular, _ = stats.pearsonr(angular_diff, gaps)
    print(f"   angle(h,top2)-angle(h,top1) → gap: r={r_angular:.4f}")
    
    # 4. Local vs global geometry
    print(f"\n4. Local vs Global Geometry:")
    # For each h, compute gap with all tokens (not just top1/top2)
    # This tests if the gap is a local or global property
    
    # Subsample tokens for efficiency
    n_tokens_sample = 1000
    sample_token_indices = np.random.choice(W_U.shape[0], n_tokens_sample, replace=False)
    
    # For each text, compute distribution of h@W_U[token]
    gap_percentiles = []
    for f in all_features:
        all_scores = f['h'] @ W_U[sample_token_indices].T
        top_score = np.max(all_scores)
        sorted_scores = np.sort(all_scores)[::-1]
        actual_gap = f['logit_gap']
        # Where does actual top1 score fall?
        percentile = np.searchsorted(np.sort(all_scores), np.dot(f['h'], f['W_U_top1'])) / len(all_scores)
        gap_percentiles.append(percentile)
    
    gap_percentiles = np.array(gap_percentiles)
    print(f"   Top1 token score percentile: {np.mean(gap_percentiles)*100:.1f}% ± {np.std(gap_percentiles)*100:.1f}%")
    print(f"   (Top1 score is at ~{np.mean(gap_percentiles)*100:.0f}th percentile of random tokens)")
    
    # 5. Riemannian metric test: Is W_U·h equivalent to a Riemannian metric?
    # If M is a metric tensor, then d(h, top_k)^2 = h^T M top_k
    # For W_U: logit_k = W_U[k] · h = h^T W_U[k]
    # This is a flat (Euclidean) metric with M = I
    print(f"\n5. Metric Structure:")
    print(f"   Logit = W_U[k] · h is a LINEAR (flat) metric")
    print(f"   No curvature effects needed - gap is purely Euclidean dot product")
    
    results = {
        'eucl_full_r': float(r_eucl_full),
        'eucl_pc50_r': float(r_eucl_pc),
        'sq_diff_r': float(r_sq_diff),
        'corrected_r': float(r_corrected),
        'angular_r': float(r_angular),
        'norm_corr_r': float(r_norm_corr),
        'top1_percentile': float(np.mean(gap_percentiles)),
    }
    
    return results


def experiment_p635(all_features, W_U, W_U_pcs, W_U_explained, d_model, model_name, 
                    all_features_other_models=None):
    """P635: Cross-Model Encoding-Decoding Invariants
    Test if different models share similar W_U PC structure:
    1. W_U PC alignment across models
    2. Shared encoding-decoding structure
    3. Universal vs model-specific features
    """
    print(f"\n{'='*60}")
    print(f"P635: Cross-Model Encoding-Decoding Invariants ({model_name})")
    print(f"{'='*60}")
    
    gaps = np.array([f['logit_gap'] for f in all_features])
    
    # 1. W_U PC structure analysis for this model
    print(f"\n1. W_U PC Structure ({model_name}):")
    cumvar = np.cumsum(W_U_explained)
    print(f"   Top-10 explained variance: {np.sum(W_U_explained[:10])*100:.1f}%")
    print(f"   Top-50 explained variance: {np.sum(W_U_explained[:50])*100:.1f}%")
    print(f"   Top-100 explained variance: {np.sum(W_U_explained[:100])*100:.1f}%")
    
    # 2. W_U PC semantic interpretation
    print(f"\n2. W_U PC Semantic Interpretation:")
    # For each top PC, find the tokens with highest/lowest projection
    for pc_idx in range(min(5, len(W_U_pcs))):
        pc = W_U_pcs[pc_idx]
        projections = W_U @ pc
        
        top_tokens = np.argsort(projections)[-5:][::-1]
        bottom_tokens = np.argsort(projections)[:5]
        
        print(f"   PC {pc_idx}:")
        print(f"     Top-5 token indices: {top_tokens.tolist()}")
        print(f"     Bottom-5 token indices: {bottom_tokens.tolist()}")
        print(f"     Variance explained: {W_U_explained[pc_idx]*100:.2f}%")
    
    # 3. h trajectory entropy in W_U PC space
    print(f"\n3. h Trajectory in W_U PC Space:")
    h_projs = np.array([f['h_wu_proj'][:50] for f in all_features])
    
    # Variance of each PC across texts
    pc_variances = np.var(h_projs, axis=0)
    total_variance = np.sum(pc_variances)
    variance_concentration = pc_variances / (total_variance + 1e-10)
    
    # Entropy of variance distribution
    entropy = -np.sum(variance_concentration * np.log(variance_concentration + 1e-10))
    max_entropy = np.log(50)
    normalized_entropy = entropy / max_entropy
    
    print(f"   Variance entropy: {entropy:.2f} / {max_entropy:.2f} = {normalized_entropy:.4f}")
    print(f"   (0 = all variance in 1 PC, 1 = uniform across all PCs)")
    print(f"   Top-5 PCs variance fraction: {np.sum(variance_concentration[:5])*100:.1f}%")
    
    # 4. Universal features: Which PCs are most predictive of gap?
    print(f"\n4. Gap-Predictive PC Structure:")
    pc_gap_r = []
    for pc_idx in range(min(50, h_projs.shape[1])):
        r, _ = stats.pearsonr(h_projs[:, pc_idx], gaps)
        pc_gap_r.append((pc_idx, r))
    
    # Sort by absolute correlation
    pc_gap_r.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"   Top-5 gap-predictive PCs: {[(idx, round(r, 4)) for idx, r in pc_gap_r[:5]]}")
    
    # 5. Decoding efficiency: How well do top-K W_U PCs reconstruct the full gap?
    print(f"\n5. Decoding Efficiency:")
    for n_pc in [5, 10, 20, 50]:
        pred_gaps = []
        for f in all_features:
            approx_gap = np.dot(f['h_wu_proj'][:n_pc], f['dw_wu_proj'][:n_pc])
            pred_gaps.append(approx_gap)
        pred_gaps = np.array(pred_gaps)
        r, _ = stats.pearsonr(pred_gaps, gaps)
        print(f"   Top-{n_pc} PCs: gap reconstruction r={r:.4f}")
    
    # 6. Encoding invariant: h norm and angle with W_U
    print(f"\n6. Encoding Invariants:")
    h_norms = np.array([f['h_norm'] for f in all_features])
    
    # Angle of h with each W_U PC
    h_angles = []
    for f in all_features:
        cos_angles = f['h_wu_proj'][:10] / (np.linalg.norm(f['h_wu_proj'][:10]) + 1e-10)
        h_angles.append(cos_angles)
    h_angles = np.array(h_angles)
    
    # Are h angles correlated with gap?
    for pc_idx in range(min(3, h_angles.shape[1])):
        r, _ = stats.pearsonr(h_angles[:, pc_idx], gaps)
        print(f"   h angle with PC {pc_idx} → gap: r={r:.4f}")
    
    r_norm, _ = stats.pearsonr(h_norms, gaps)
    print(f"   h_norm → gap: r={r_norm:.4f}")
    
    results = {
        'top10_var': float(np.sum(W_U_explained[:10])),
        'top50_var': float(np.sum(W_U_explained[:50])),
        'variance_entropy': float(normalized_entropy),
        'top5_gap_predictive_pcs': [(int(idx), float(r)) for idx, r in pc_gap_r[:5]],
        'h_norm_gap_r': float(r_norm),
    }
    
    return results


def experiment_p636(all_features, W_U, W_U_pcs, W_U_explained, d_model, n_layers, model_name):
    """P636: Unified Intelligence Theory Framework v1
    Build the complete mathematical framework:
    1. Encoding equation: x → h
    2. Decoding equation: h → logits
    3. Language behavior equation: logits → prob
    4. Full pipeline: x → prob
    5. Information flow analysis
    """
    print(f"\n{'='*60}")
    print(f"P636: Unified Intelligence Theory Framework v1 ({model_name})")
    print(f"{'='*60}")
    
    gaps = np.array([f['logit_gap'] for f in all_features])
    probs = np.array([f['prob'] for f in all_features])
    h_projs = np.array([f['h_wu_proj'] for f in all_features])
    
    # 1. Full pipeline: h → gap → prob
    print(f"\n1. Full Pipeline Analysis:")
    
    # Step 1: h → gap (via W_U projection)
    # Best non-oracle: h@W_U_PCs → gap
    reg = Ridge(alpha=1.0)
    reg.fit(h_projs[:, :50], gaps)
    pred_gaps = reg.predict(h_projs[:, :50])
    r_h_gap, _ = stats.pearsonr(pred_gaps, gaps)
    
    # Step 2: gap → prob (sigmoid)
    pred_probs_from_pred_gap = 1.0 / (1.0 + np.exp(-pred_gaps))
    r_gap_prob, _ = stats.pearsonr(pred_probs_from_pred_gap, probs)
    
    # Oracle: actual gap → prob
    oracle_probs = 1.0 / (1.0 + np.exp(-gaps))
    r_oracle, _ = stats.pearsonr(oracle_probs, probs)
    
    print(f"   h@W_U_PCs → gap: r={r_h_gap:.4f}")
    print(f"   Predicted gap → prob: r={r_gap_prob:.4f}")
    print(f"   Oracle gap → prob: r={r_oracle:.4f}")
    print(f"   Pipeline efficiency: {r_gap_prob/r_oracle*100:.1f}% of oracle")
    
    # 2. Decomposition of gap: which W_U PCs contribute most?
    print(f"\n2. Gap Decomposition by W_U PCs:")
    reg_full = Ridge(alpha=0.1)
    reg_full.fit(h_projs[:, :100], gaps)
    
    # Coefficient importance
    coef_importance = np.abs(reg_full.coef_)
    total_importance = np.sum(coef_importance)
    importance_ratio = coef_importance / (total_importance + 1e-10)
    
    # Cumulative importance
    sorted_idx = np.argsort(importance_ratio)[::-1]
    cum_importance = np.cumsum(importance_ratio[sorted_idx])
    
    for threshold in [0.50, 0.80, 0.90, 0.95]:
        n_needed = np.searchsorted(cum_importance, threshold) + 1
        print(f"   {threshold*100:.0f}% importance in top-{n_needed} PCs")
    
    # 3. Information flow: Layer-by-layer contribution
    print(f"\n3. Information Flow Analysis:")
    layer_gap_r = []
    for layer_idx in range(min(n_layers, len(all_features[0]['all_hidden']))):
        h_l_list = [f['all_hidden'][layer_idx] for f in all_features]
        H_l = np.array(h_l_list)
        
        # Project to W_U PCs
        H_l_proj = (W_U_pcs[:50] @ H_l.T).T
        
        reg_l = Ridge(alpha=1.0)
        reg_l.fit(H_l_proj, gaps)
        pred_l = reg_l.predict(H_l_proj)
        r_l, _ = stats.pearsonr(pred_l, gaps)
        layer_gap_r.append(r_l)
        
        if layer_idx < 3 or layer_idx >= n_layers - 3:
            print(f"   Layer {layer_idx}: h@W_U_PCs → gap r={r_l:.4f}")
    
    # Information gain per layer
    if len(layer_gap_r) > 1:
        info_gain = np.diff(layer_gap_r)
        max_gain_layer = np.argmax(info_gain)
        print(f"   Max information gain at layer {max_gain_layer} (Δr={info_gain[max_gain_layer]:.4f})")
        print(f"   Information gain in last 5 layers: {info_gain[-5:].tolist()}")
    
    # 4. Mathematical framework summary
    print(f"\n4. Mathematical Framework Summary:")
    print(f"   === Encoding-Decoding Geometry ===")
    print(f"   Encoding:  x → h = F(x; θ)    [F = Transformer layers]")
    print(f"   Decoding:  h → logits = W_U · h [Linear projection]")
    print(f"   Selection: prob = sigmoid(gap)   [Deterministic sigmoid]")
    print(f"   ")
    print(f"   Where gap = h · (W_U[top1] - W_U[top2])")
    print(f"         = h · Delta_W")
    print(f"         ≈ (h@W_U_PCs) · (Delta_W@W_U_PCs)  [Low-rank approx]")
    print(f"   ")
    print(f"   Key property: gap is LINEAR in h (Euclidean dot product)")
    print(f"   No quantum effects, no curvature, no interference needed")
    
    # 5. Comparison with brain
    print(f"\n5. Brain Comparison:")
    print(f"   Transformer              Brain")
    print(f"   -----------------        -----------------")
    print(f"   h (hidden state)  ≈     Neural population activity")
    print(f"   W_U (unembed)     ≈     Motor cortex mapping")
    print(f"   gap (logit diff)  ≈     Decision variable")
    print(f"   sigmoid(gap)      ≈     Choice probability")
    print(f"   W_U PCs           ≈     Principal neural modes")
    print(f"   Layer propagation ≈     Cortical hierarchy")
    
    # 6. Three competing hypotheses
    print(f"\n6. Competing Mathematical Hypotheses:")
    
    # H1: Manifold geometry (gap = geodesic distance)
    print(f"   H1: Manifold Geometry - gap as geodesic distance on W_U manifold")
    print(f"       Evidence: W_U low-rank, gap=dot product (flat metric)")
    print(f"       Prediction: W_U manifold should be approximately flat")
    
    # H2: Information geometry (gap = Fisher distance)
    print(f"   H2: Information Geometry - gap as Fisher information distance")
    print(f"       Evidence: prob=sigmoid(gap) is exponential family")
    print(f"       Prediction: Fisher metric should match W_U PC structure")
    
    # H3: Category theory (encoding-decoding as functor)
    print(f"   H3: Category Theory - encoding-decoding as functor")
    print(f"       Evidence: Linear structure, low-rank decomposition")
    print(f"       Prediction: Compositionality in W_U PC structure")
    
    # Test H1: Flatness of W_U manifold
    # If W_U is flat, then Isomap and MDS should give similar embeddings
    n_sample = min(1000, W_U.shape[0])
    sample_idx = np.random.choice(W_U.shape[0], n_sample, replace=False)
    W_U_sample = W_U[sample_idx]
    W_U_sample_norm = W_U_sample / (np.linalg.norm(W_U_sample, axis=1, keepdims=True) + 1e-10)
    
    # PCA vs Isomap reconstruction
    pca_10 = PCA(n_components=10)
    W_U_pca_10 = pca_10.fit_transform(W_U_sample_norm)
    pca_reconstruction = pca_10.inverse_transform(W_U_pca_10)
    pca_error = np.mean((W_U_sample_norm - pca_reconstruction)**2)
    
    print(f"\n7. Manifold Flatness Test:")
    print(f"   PCA-10 reconstruction error: {pca_error:.6f}")
    print(f"   (PCA assumes flat manifold; low error = manifold is flat)")
    
    results = {
        'pipeline_h_gap_r': float(r_h_gap),
        'pipeline_gap_prob_r': float(r_gap_prob),
        'pipeline_oracle_r': float(r_oracle),
        'pipeline_efficiency': float(r_gap_prob/r_oracle*100),
        'layer_gap_r': [(int(i), float(r)) for i, r in enumerate(layer_gap_r[-5:])] if layer_gap_r else [],
        'max_info_gain_layer': int(max_gain_layer) if len(layer_gap_r) > 1 else 0,
        'pca_reconstruction_error': float(pca_error),
    }
    
    return results


def run_experiment(model_name, experiment_name):
    """Run a specific experiment for a specific model."""
    print(f"\n{'#'*60}")
    print(f"# Model: {model_name}, Experiment: {experiment_name}")
    print(f"{'#'*60}")
    
    # Load model
    print(f"Loading {model_name}...")
    model, tokenizer, device = load_model(model_name)
    model.eval()
    
    # Compute features
    print("Computing features...")
    all_features, W_U, W_U_pcs, W_U_explained, d_model, n_layers = compute_features(
        model, tokenizer, device, TEST_TEXTS
    )
    
    # Run experiment
    experiments = {
        'p631': experiment_p631,
        'p632': experiment_p632,
        'p633': experiment_p633,
        'p634': experiment_p634,
        'p635': experiment_p635,
        'p636': experiment_p636,
    }
    
    if experiment_name not in experiments:
        print(f"Unknown experiment: {experiment_name}")
        return None
    
    if experiment_name == 'p632':
        results = experiments[experiment_name](
            all_features, W_U, W_U_pcs, W_U_explained, d_model, n_layers, model_name
        )
    elif experiment_name == 'p636':
        results = experiments[experiment_name](
            all_features, W_U, W_U_pcs, W_U_explained, d_model, n_layers, model_name
        )
    else:
        results = experiments[experiment_name](
            all_features, W_U, W_U_pcs, W_U_explained, d_model, model_name
        )
    
    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'phase_cxliii')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{experiment_name}_{model_name}.json')
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    # Release model
    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase CXLIII: Encoding-Decoding Geometry Framework")
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True, 
                        choices=["p631", "p632", "p633", "p634", "p635", "p636"])
    args = parser.parse_args()
    
    results = run_experiment(args.model, args.experiment)
    
    if results:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT COMPLETE: {args.experiment} ({args.model})")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
