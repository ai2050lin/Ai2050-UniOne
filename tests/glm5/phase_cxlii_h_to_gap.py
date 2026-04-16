#!/usr/bin/env python3
"""
Phase CXLII: Breaking the h->gap Ceiling (P625-P630)
Focus: Alternative routes beyond quantum acoustics for h->gap prediction

Key insight from Phase CXLI: gap->prob is perfectly solved by sigmoid (r=1.0).
The real bottleneck is h->gap (spectral mechanics ceiling: Non-oracle r<0.35).

P625: Complex h features ceiling - all features from h_complex predict gap
P626: Direction-level Non-oracle ceiling - spectral params predict |h[k]|
P627: W_U structure deep dive - low-rank Delta_k prediction
P628: Energy landscape - logit as energy function of h
P629: Sparse coding approach - top-K sparse approximation of gap
P630: Unified equation v5 - prob = sigmoid(f(spectral, W_U structure))
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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
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
        h_prev = all_hidden[-2] if len(all_hidden) > 1 else h
        logits = outputs.logits[0, -1, :].cpu().float().numpy()
        
        # Token info
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        logit_gap = logits[top1_idx] - logits[top2_idx]
        logit_max = logits[top1_idx]
        prob = 1.0 / (1.0 + np.exp(-logit_gap))
        
        # W_U structure
        W_U_top1 = W_U[top1_idx]
        W_U_top2 = W_U[top2_idx]
        Delta_W = W_U_top1 - W_U_top2
        
        # SVD of h
        U_h, S_h, Vt_h = np.linalg.svd(h.reshape(1, -1), full_matrices=False)
        h_norm = np.linalg.norm(h)
        
        # Spectral features
        cumsum = np.cumsum(S_h**2) / np.sum(S_h**2)
        ratio50 = np.searchsorted(cumsum, 0.50) / len(S_h) if len(S_h) > 0 else 0
        ratio90 = np.searchsorted(cumsum, 0.90) / len(S_h) if len(S_h) > 0 else 0
        ratio99 = np.searchsorted(cumsum, 0.99) / len(S_h) if len(S_h) > 0 else 0
        
        # Direction-level features: h[k] and h[k]*Delta_W[k]
        contributions = h * Delta_W  # element-wise
        
        # Top-K contributions
        abs_contrib = np.abs(contributions)
        top_k_idx = np.argsort(abs_contrib)[::-1]
        
        # Sign statistics
        signs = np.sign(contributions)
        n_positive = np.sum(signs > 0)
        n_negative = np.sum(signs < 0)
        sign_ratio = n_positive / (n_positive + n_negative + 1e-10)
        
        # Energy in different frequency bands
        n = len(h)
        band_size = max(1, n // 10)
        band_energies = []
        for b in range(10):
            start = b * band_size
            end = min((b+1) * band_size, n)
            band_energies.append(np.sum(h[start:end]**2) / (h_norm**2 + 1e-10))
        
        # PCA features from h
        # Use the contribution pattern
        top10_contrib = np.sum(abs_contrib[top_k_idx[:10]])
        top50_contrib = np.sum(abs_contrib[top_k_idx[:50]])
        top100_contrib = np.sum(abs_contrib[top_k_idx[:100]])
        total_contrib = np.sum(abs_contrib)
        
        # Net contribution (with signs)
        net_positive = np.sum(contributions[contributions > 0])
        net_negative = np.sum(contributions[contributions < 0])
        
        features = {
            'text_idx': i,
            'h': h,
            'h_prev': h_prev,
            'logits': logits,
            'logit_gap': logit_gap,
            'logit_max': logit_max,
            'prob': prob,
            'top1_idx': top1_idx,
            'top2_idx': top2_idx,
            'Delta_W': Delta_W,
            'W_U': W_U,
            'W_U_top1': W_U_top1,
            'W_U_top2': W_U_top2,
            'h_norm': h_norm,
            'd_model': d_model,
            'n_layers': n_layers,
            'all_hidden': all_hidden,
            # Spectral features
            'ratio50': ratio50,
            'ratio90': ratio90,
            'ratio99': ratio99,
            'S_h': S_h,
            # Direction features
            'contributions': contributions,
            'abs_contrib': abs_contrib,
            'top_k_idx': top_k_idx,
            'sign_ratio': sign_ratio,
            'n_positive': n_positive,
            'n_negative': n_negative,
            # Contribution features
            'top10_contrib': top10_contrib,
            'top50_contrib': top50_contrib,
            'top100_contrib': top100_contrib,
            'total_contrib': total_contrib,
            'net_positive': net_positive,
            'net_negative': net_negative,
            # Band energies
            'band_energies': band_energies,
        }
        
        all_features.append(features)
    
    return all_features


def experiment_p625(all_features, model_name, model=None):
    """P625: Complex h features ceiling - what is the maximum h->gap prediction?"""
    print(f"\nP625: Complex h features ceiling -- {model_name}")
    print("Testing: What is the maximum h->gap prediction using ALL features?")
    
    gaps = np.array([f['logit_gap'] for f in all_features])
    probs = np.array([f['prob'] for f in all_features])
    
    # Build comprehensive feature matrix
    # Group 1: Spectral features (known to be weak)
    spectral_features = []
    for f in all_features:
        row = [
            f['h_norm'],
            f['ratio50'],
            f['ratio90'],
            f['ratio99'],
            f['sign_ratio'],
            f['n_positive'],
            f['n_negative'],
        ]
        spectral_features.append(row)
    X_spectral = np.array(spectral_features)
    
    # Group 2: Contribution statistics (without oracle - statistical properties)
    contrib_features = []
    for f in all_features:
        ac = f['abs_contrib']
        c = f['contributions']
        row = [
            f['top10_contrib'] / (f['total_contrib'] + 1e-10),  # concentration
            f['top50_contrib'] / (f['total_contrib'] + 1e-10),
            f['top100_contrib'] / (f['total_contrib'] + 1e-10),
            f['net_positive'] / (f['total_contrib'] + 1e-10),  # sign bias
            np.std(ac) / (np.mean(ac) + 1e-10),  # CV of |contrib|
            np.max(ac) / (np.mean(ac) + 1e-10),  # max/mean ratio
            np.median(ac),  # median contribution
            np.sum(c) / (np.sum(np.abs(c)) + 1e-10),  # net/total ratio
            np.mean(c[c > 0]) if np.sum(c > 0) > 0 else 0,  # mean positive
            np.mean(c[c < 0]) if np.sum(c < 0) > 0 else 0,  # mean negative
        ]
        contrib_features.append(row)
    X_contrib = np.array(contrib_features)
    
    # Group 3: Band energy features
    band_features = []
    for f in all_features:
        band_features.append(f['band_energies'])
    X_band = np.array(band_features)
    
    # Group 4: Energy distribution statistics (replace SVD with sorted |h|)
    svd_features = []
    for f in all_features:
        h = f['h']
        sorted_abs_h = np.sort(np.abs(h))[::-1]
        total_energy = np.sum(h**2)
        row = [
            sorted_abs_h[0]**2 / (total_energy + 1e-10) if total_energy > 0 else 0,  # top-1 energy
            np.sum(sorted_abs_h[:10]**2) / (total_energy + 1e-10),  # top-10 energy
            np.sum(sorted_abs_h[:50]**2) / (total_energy + 1e-10),  # top-50 energy
            np.sum(sorted_abs_h[:100]**2) / (total_energy + 1e-10),  # top-100 energy
            sorted_abs_h[0] / (np.mean(sorted_abs_h) + 1e-10),  # max/mean ratio
            np.std(sorted_abs_h) / (np.mean(sorted_abs_h) + 1e-10),  # CV of |h|
        ]
        svd_features.append(row)
    X_svd = np.array(svd_features)
    
    # Group 5: Cross-layer features
    cross_features = []
    for f in all_features:
        h = f['h']
        h_prev = f['h_prev']
        Delta_W = f['Delta_W']
        row = [
            np.linalg.norm(h_prev),
            np.dot(h, h_prev) / (np.linalg.norm(h) * np.linalg.norm(h_prev) + 1e-10),
            np.dot(h, Delta_W),  # This IS the gap (oracle!)
            np.dot(h_prev, Delta_W),  # Previous layer gap
            np.linalg.norm(h - h_prev),  # Layer change magnitude
        ]
        cross_features.append(row)
    X_cross = np.array(cross_features)
    
    # Remove oracle features (column 2 is the gap itself)
    X_cross_no_oracle = np.delete(X_cross, 2, axis=1)
    
    # Test each group
    results = {}
    
    # Spectral only
    reg = LinearRegression().fit(X_spectral, gaps)
    r_spectral = np.sqrt(reg.score(X_spectral, gaps))
    print(f"  Spectral only -> gap r={r_spectral:.4f}")
    results['r_spectral'] = r_spectral
    
    # Contribution stats only
    reg = LinearRegression().fit(X_contrib, gaps)
    r_contrib = np.sqrt(max(0, reg.score(X_contrib, gaps)))
    print(f"  Contribution stats -> gap r={r_contrib:.4f}")
    results['r_contrib'] = r_contrib
    
    # Band energy only
    reg = LinearRegression().fit(X_band, gaps)
    r_band = np.sqrt(max(0, reg.score(X_band, gaps)))
    print(f"  Band energy -> gap r={r_band:.4f}")
    results['r_band'] = r_band
    
    # SVD stats only
    reg = LinearRegression().fit(X_svd, gaps)
    r_svd = np.sqrt(max(0, reg.score(X_svd, gaps)))
    print(f"  SVD stats -> gap r={r_svd:.4f}")
    results['r_svd'] = r_svd
    
    # Cross-layer (no oracle) only
    reg = LinearRegression().fit(X_cross_no_oracle, gaps)
    r_cross = np.sqrt(max(0, reg.score(X_cross_no_oracle, gaps)))
    print(f"  Cross-layer (no oracle) -> gap r={r_cross:.4f}")
    results['r_cross'] = r_cross
    
    # Combined: all non-oracle features
    X_all = np.column_stack([X_spectral, X_contrib, X_band, X_svd, X_cross_no_oracle])
    reg = Ridge(alpha=1.0).fit(X_all, gaps)
    r_all = np.sqrt(max(0, reg.score(X_all, gaps)))
    print(f"  ALL non-oracle features -> gap r={r_all:.4f}")
    results['r_all_nonoracle'] = r_all
    
    # Oracle baseline: just the gap
    r_oracle = 1.0  # gap predicts itself perfectly
    print(f"  Oracle (gap itself) -> gap r={r_oracle:.4f}")
    results['r_oracle'] = r_oracle
    
    # Previous layer gap as predictor
    prev_gaps = X_cross[:, 3]  # h_prev . Delta_W
    r_prev_gap, _ = stats.pearsonr(prev_gaps, gaps)
    print(f"  Previous layer gap -> gap r={r_prev_gap:.4f}")
    results['r_prev_gap'] = r_prev_gap
    
    # Gap prediction from prev_gap + spectral
    X_prev_spectral = np.column_stack([X_cross_no_oracle[:, 1], X_spectral])
    reg = LinearRegression().fit(X_prev_spectral, gaps)
    r_prev_spectral = np.sqrt(max(0, reg.score(X_prev_spectral, gaps)))
    print(f"  Prev gap + spectral -> gap r={r_prev_spectral:.4f}")
    results['r_prev_spectral'] = r_prev_spectral
    
    return results


def experiment_p626(all_features, model_name, model=None):
    """P626: Direction-level Non-oracle ceiling - can spectral params predict |h[k]|?"""
    print(f"\nP626: Direction-level Non-oracle ceiling -- {model_name}")
    print("Testing: Can spectral parameters predict |h[k]| for important directions?")
    
    d_model = all_features[0]['d_model']
    K = 50  # Top-K directions
    
    # For each text, compute |h[k]| for top-K contribution directions
    # and try to predict them from spectral features
    
    all_r_values = []
    all_r_by_rank = []
    
    for rank_k in [1, 5, 10, 20, 50]:
        abs_h_at_rank = []
        spectral_predictors = []
        
        for f in all_features:
            h = f['h']
            abs_contrib = f['abs_contrib']
            top_k_idx = f['top_k_idx']
            sorted_abs_h = np.sort(np.abs(h))[::-1]
            total_energy = np.sum(h**2)
            
            if rank_k <= len(top_k_idx):
                k = top_k_idx[rank_k - 1]  # k-th most important dimension
                abs_h_at_rank.append(np.abs(h[k]))
                
                # Spectral predictors for this |h[k]|
                row = [
                    f['h_norm'],
                    f['ratio50'],
                    f['ratio90'],
                    np.sum(sorted_abs_h[:10]**2) / (total_energy + 1e-10),  # top-10 energy
                    np.sum(sorted_abs_h[:50]**2) / (total_energy + 1e-10),  # top-50 energy
                    f['top10_contrib'] / (f['total_contrib'] + 1e-10),  # concentration
                    np.max(abs_contrib) / (np.mean(abs_contrib) + 1e-10),  # max/mean
                ]
                spectral_predictors.append(row)
        
        abs_h_at_rank = np.array(abs_h_at_rank)
        X_pred = np.array(spectral_predictors)
        
        if len(abs_h_at_rank) > 2:
            reg = LinearRegression().fit(X_pred, abs_h_at_rank)
            r_pred = np.sqrt(max(0, reg.score(X_pred, abs_h_at_rank)))
            all_r_by_rank.append((rank_k, r_pred))
            print(f"  Rank-{rank_k} |h[k]| prediction from spectral: r={r_pred:.4f}")
    
    # Overall: can we predict the contribution pattern?
    # Test: predicted_gap = sum_k predicted_|h[k]| * |Delta_W[k]|
    # vs actual_gap = sum_k h[k] * Delta_W[k]
    
    predicted_gaps = []
    actual_gaps = []
    
    for f in all_features:
        h = f['h']
        Delta_W = f['Delta_W']
        actual_gap = f['logit_gap']
        
        # Use spectral features to estimate |h[k]| * |Delta_W[k]|
        h_norm = f['h_norm']
        ratio50 = f['ratio50']
        concentration = f['top10_contrib'] / (f['total_contrib'] + 1e-10)
        
        # Simple model: |h[k]| ~ h_norm / sqrt(d) (uniform assumption)
        d = len(h)
        uniform_abs_h = h_norm / np.sqrt(d)
        predicted_gap_uniform = np.sum(uniform_abs_h * np.abs(Delta_W)) * (2 * f['sign_ratio'] - 1)
        
        predicted_gaps.append(predicted_gap_uniform)
        actual_gaps.append(actual_gap)
    
    r_uniform, _ = stats.pearsonr(predicted_gaps, actual_gaps)
    print(f"  Uniform |h[k]| model -> gap r={r_uniform:.4f}")
    
    # Concentrated model: |h[k]| ~ h_norm * sqrt(concentration) for top-K
    predicted_gaps_conc = []
    for f in all_features:
        h = f['h']
        Delta_W = f['Delta_W']
        actual_gap = f['logit_gap']
        h_norm = f['h_norm']
        conc = f['top10_contrib'] / (f['total_contrib'] + 1e-10)
        sign_bias = 2 * f['sign_ratio'] - 1
        
        # Top-K dimensions get more energy
        abs_dw = np.abs(Delta_W)
        total_dw = np.sum(abs_dw)
        
        predicted_gaps_conc.append(h_norm * np.sqrt(conc) * total_dw * sign_bias / np.sqrt(len(h)))
    
    r_conc, _ = stats.pearsonr(predicted_gaps_conc, actual_gaps)
    print(f"  Concentrated |h[k]| model -> gap r={r_conc:.4f}")
    
    results = {
        'experiment': 'P626',
        'model': model_name,
        'r_by_rank': {str(k): float(r) for k, r in all_r_by_rank},
        'r_uniform': float(r_uniform),
        'r_concentrated': float(r_conc),
    }
    
    return results


def experiment_p627(all_features, model_name, model=None):
    """P627: W_U structure deep dive - low-rank Delta_k prediction."""
    print(f"\nP627: W_U structure deep dive -- {model_name}")
    print("Testing: Can W_U structure predict Delta_W = W_U[top1]-W_U[top2]?")
    
    W_U = all_features[0]['W_U']
    d_model = W_U.shape[1]
    
    # Collect all Delta_W vectors
    Delta_Ws = []
    gaps = []
    
    for f in all_features:
        Delta_Ws.append(f['Delta_W'])
        gaps.append(f['logit_gap'])
    
    Delta_Ws = np.array(Delta_Ws)
    
    # 1. Delta_W norm statistics
    dw_norms = np.linalg.norm(Delta_Ws, axis=1)
    print(f"  Delta_W norm: mean={np.mean(dw_norms):.4f}, std={np.std(dw_norms):.4f}")
    
    # 2. Can we predict Delta_W direction from W_U structure?
    # W_U SVD (use small random sample to avoid memory)
    n_samples = min(1000, W_U.shape[0])
    sample_idx = np.random.choice(W_U.shape[0], n_samples, replace=False)
    W_U_sample = W_U[sample_idx]
    
    # PCA of W_U rows
    pca = PCA(n_components=min(50, d_model))
    pca.fit(W_U_sample)
    W_U_pcs = pca.components_  # (50, d_model)
    W_U_explained = pca.explained_variance_ratio_
    
    print(f"  W_U PCA top-10 explained: {np.sum(W_U_explained[:10]):.4f}")
    print(f"  W_U PCA top-50 explained: {np.sum(W_U_explained[:50]):.4f}")
    
    # 3. Project Delta_W onto W_U principal components
    dw_projections = Delta_Ws @ W_U_pcs.T  # (n_texts, 50)
    
    # Test: do these projections predict gap?
    r_dw_pc1_gap, _ = stats.pearsonr(dw_projections[:, 0], gaps)
    print(f"  Delta_W @ PC1 -> gap r={r_dw_pc1_gap:.4f}")
    
    # Full regression
    reg = LinearRegression().fit(dw_projections, gaps)
    r_dw_pcs_gap = np.sqrt(max(0, reg.score(dw_projections, gaps)))
    print(f"  Delta_W @ all PCs -> gap r={r_dw_pcs_gap:.4f}")
    
    # 4. Cross-token Delta_W similarity
    # Are Delta_W vectors for different texts similar?
    dw_similarity = np.corrcoef(Delta_Ws)
    np.fill_diagonal(dw_similarity, 0)
    mean_dw_sim = np.mean(dw_similarity[dw_similarity != 0])
    print(f"  Mean cross-text Delta_W similarity: {mean_dw_sim:.4f}")
    
    # 5. Delta_W alignment with W_U mean
    W_U_mean = np.mean(W_U, axis=0)
    dw_alignments = [np.dot(dw, W_U_mean) / (np.linalg.norm(dw) * np.linalg.norm(W_U_mean) + 1e-10) for dw in Delta_Ws]
    r_dw_align_gap, _ = stats.pearsonr(dw_alignments, gaps)
    print(f"  Delta_W alignment with W_U mean -> gap r={r_dw_align_gap:.4f}")
    
    # 6. Token-pair specific structure
    # For top1/top2 pairs, does W_U[top1]-W_U[top2] have a consistent structure?
    top1_indices = [f['top1_idx'] for f in all_features]
    top2_indices = [f['top2_idx'] for f in all_features]
    unique_top1 = len(set(top1_indices))
    unique_top2 = len(set(top2_indices))
    print(f"  Unique top1 tokens: {unique_top1}/{len(all_features)}")
    print(f"  Unique top2 tokens: {unique_top2}/{len(all_features)}")
    
    results = {
        'experiment': 'P627',
        'model': model_name,
        'dw_norm_mean': float(np.mean(dw_norms)),
        'dw_norm_std': float(np.std(dw_norms)),
        'W_U_PCA_top10': float(np.sum(W_U_explained[:10])),
        'W_U_PCA_top50': float(np.sum(W_U_explained[:50])),
        'r_dw_pc1_gap': float(r_dw_pc1_gap),
        'r_dw_pcs_gap': float(r_dw_pcs_gap),
        'mean_dw_similarity': float(mean_dw_sim),
        'r_dw_align_gap': float(r_dw_align_gap),
        'unique_top1': unique_top1,
        'unique_top2': unique_top2,
    }
    
    return results


def experiment_p628(all_features, model_name, model=None):
    """P628: Energy landscape - logit as energy function of h."""
    print(f"\nP628: Energy landscape -- {model_name}")
    print("Testing: Does logit computation have energy landscape properties?")
    
    W_U = all_features[0]['W_U']
    gaps = np.array([f['logit_gap'] for f in all_features])
    probs = np.array([f['prob'] for f in all_features])
    h_norms = np.array([f['h_norm'] for f in all_features])
    
    # 1. Energy function perspective: E(h) = -logit_gap = -h·(W_U[top1]-W_U[top2])
    # The "energy" is a linear function of h in the Delta_W direction
    # Prob = sigmoid(-E) = 1/(1+exp(E))
    
    # 2. Energy curvature: how sensitive is E to perturbations?
    # Compute Hessian of E(h) = -h·Delta_W
    # For linear E, Hessian = 0, so there's no curvature
    
    # 3. Instead, look at logit landscape: logits = W_U @ h
    # This is a linear map from h-space to logit-space
    # The "energy well" depth for top1 is logit[top1]
    
    # 4. Marginal energy gap: E[top1] - E[top2] = logit_gap
    # How does this relate to h_norm and W_U structure?
    
    logit_maxes = np.array([f['logit_max'] for f in all_features])
    
    # logit_max = W_U[top1] · h ≈ ||W_U[top1]|| * ||h|| * cos(theta)
    W_U_norms = np.linalg.norm(W_U, axis=1)
    
    cos_angles = []
    for f in all_features:
        top1 = f['top1_idx']
        cos_a = np.dot(f['h'], W_U[top1]) / (f['h_norm'] * W_U_norms[top1] + 1e-10)
        cos_angles.append(cos_a)
    
    r_cos_gap, _ = stats.pearsonr(cos_angles, gaps)
    print(f"  cos(angle, W_U[top1]) -> gap r={r_cos_gap:.4f}")
    
    # 5. Logit landscape smoothness: correlation between adjacent logits
    # For a smooth landscape, nearby tokens (in W_U space) should have similar logits
    all_logits = np.array([f['logits'] for f in all_features])
    
    # 6. Sparse coding: can we approximate gap with few directions?
    # Gap = sum_k h[k] * Delta_W[k]
    # How many directions are needed to capture 90% of gap?
    
    n_directions_90 = []
    for f in all_features:
        contributions = f['contributions']
        sorted_abs = np.sort(np.abs(contributions))[::-1]
        total = np.sum(sorted_abs)
        cumsum = np.cumsum(sorted_abs) / (total + 1e-10)
        n90 = np.searchsorted(cumsum, 0.90) + 1
        n_directions_90.append(n90)
    
    print(f"  N directions for 90% gap: mean={np.mean(n_directions_90):.1f}, median={np.median(n_directions_90):.1f}")
    
    # 7. Sparse reconstruction: use only top-K directions
    for K in [10, 50, 100]:
        sparse_gaps = []
        for f in all_features:
            contributions = f['contributions']
            top_k = f['top_k_idx'][:K]
            sparse_gap = np.sum(contributions[top_k])
            sparse_gaps.append(sparse_gap)
        
        r_sparse, _ = stats.pearsonr(sparse_gaps, gaps)
        print(f"  Top-{K} sparse gap -> actual gap r={r_sparse:.4f}")
    
    # 8. Energy gap decomposition
    # gap = sum_positive - |sum_negative| = net_positive + net_negative
    net_pos = np.array([f['net_positive'] for f in all_features])
    net_neg = np.array([f['net_negative'] for f in all_features])
    
    r_pos_gap, _ = stats.pearsonr(net_pos, gaps)
    r_neg_gap, _ = stats.pearsonr(net_neg, gaps)
    print(f"  net_positive -> gap r={r_pos_gap:.4f}")
    print(f"  net_negative -> gap r={r_neg_gap:.4f}")
    
    # 9. Participation ratio of gap
    PR_gaps = []
    for f in all_features:
        c = f['contributions']
        total = np.sum(np.abs(c))
        if total > 0:
            p = np.abs(c) / total
            PR = 1.0 / np.sum(p**2)
        else:
            PR = 0
        PR_gaps.append(PR)
    
    r_PR_gap, _ = stats.pearsonr(PR_gaps, gaps)
    r_PR_prob, _ = stats.pearsonr(PR_gaps, probs)
    print(f"  PR(contributions) -> gap r={r_PR_gap:.4f}")
    print(f"  PR(contributions) -> prob r={r_PR_prob:.4f}")
    print(f"  PR(contributions) mean: {np.mean(PR_gaps):.1f}")
    
    results = {
        'experiment': 'P628',
        'model': model_name,
        'r_cos_gap': float(r_cos_gap),
        'n_dir_90_mean': float(np.mean(n_directions_90)),
        'r_pos_gap': float(r_pos_gap),
        'r_neg_gap': float(r_neg_gap),
        'r_PR_gap': float(r_PR_gap),
        'r_PR_prob': float(r_PR_prob),
        'PR_mean': float(np.mean(PR_gaps)),
    }
    
    return results


def experiment_p629(all_features, model_name, model=None):
    """P629: Sparse coding approach - top-K sparse approximation of gap."""
    print(f"\nP629: Sparse coding approach -- {model_name}")
    print("Testing: Can sparse coding principles improve gap prediction?")
    
    gaps = np.array([f['logit_gap'] for f in all_features])
    probs = np.array([f['prob'] for f in all_features])
    W_U = all_features[0]['W_U']
    
    # Key idea: gap is determined by a few "important" dimensions
    # Can we identify these dimensions WITHOUT looking at the actual contributions?
    
    # Method 1: Use W_U structure to identify important dimensions
    # |Delta_W[k]| = |W_U[top1,k] - W_U[top2,k]|
    # Dimensions with large |Delta_W[k]| are potentially important
    
    predicted_gaps_WU = []
    for f in all_features:
        h = f['h']
        top1 = f['top1_idx']
        top2 = f['top2_idx']
        
        # Estimate contribution without knowing actual sign
        Delta_W = f['Delta_W']
        abs_dw = np.abs(Delta_W)
        
        # Use h_norm and uniform distribution as estimate
        h_norm = f['h_norm']
        d = len(h)
        est_abs_h = h_norm / np.sqrt(d)  # uniform estimate
        
        # Gap estimate: sum_k est_|h[k]| * |Delta_W[k]| * sign_bias
        sign_bias = 2 * f['sign_ratio'] - 1
        est_gap = est_abs_h * np.sum(abs_dw) * sign_bias
        predicted_gaps_WU.append(est_gap)
    
    r_WU, _ = stats.pearsonr(predicted_gaps_WU, gaps)
    print(f"  W_U-based estimate (uniform |h|) -> gap r={r_WU:.4f}")
    
    # Method 2: Use SVD of W_U to find "detection directions"
    # W_U's top PCs correspond to directions that many tokens use
    # These directions should have more consistent contributions
    
    n_samples = min(1000, W_U.shape[0])
    sample_idx = np.random.choice(W_U.shape[0], n_samples, replace=False)
    W_U_sample = W_U[sample_idx]
    
    pca = PCA(n_components=min(20, W_U.shape[1]))
    pca.fit(W_U_sample)
    W_U_pcs = pca.components_[:20]  # (20, d)
    
    # Project h onto W_U PCs and use these as features
    pc_projections = []
    for f in all_features:
        h = f['h']
        proj = h @ W_U_pcs.T  # (20,)
        pc_projections.append(proj)
    
    X_pc = np.array(pc_projections)
    reg = LinearRegression().fit(X_pc, gaps)
    r_pc = np.sqrt(max(0, reg.score(X_pc, gaps)))
    print(f"  h projected onto W_U PCs -> gap r={r_pc:.4f}")
    
    # Method 3: Sparse coding with OMP-like selection
    # Select dimensions one by one that maximize gap prediction
    
    # Method 4: Use h_prev layer information
    prev_gaps = []
    for f in all_features:
        h_prev = f['h_prev']
        Delta_W = f['Delta_W']
        prev_gaps.append(np.dot(h_prev, Delta_W))
    
    r_prev_gap, _ = stats.pearsonr(prev_gaps, gaps)
    print(f"  Previous layer gap -> gap r={r_prev_gap:.4f}")
    
    # Combined: prev_gap + spectral + W_U PCs
    spectral_features = []
    for f in all_features:
        row = [
            f['h_norm'],
            f['ratio50'],
            f['ratio90'],
            f['sign_ratio'],
        ]
        spectral_features.append(row)
    X_spec = np.array(spectral_features)
    
    X_combined = np.column_stack([np.array(prev_gaps).reshape(-1, 1), X_spec, X_pc])
    reg = Ridge(alpha=1.0).fit(X_combined, gaps)
    r_combined = np.sqrt(max(0, reg.score(X_combined, gaps)))
    print(f"  Combined (prev_gap + spectral + W_U PCs) -> gap r={r_combined:.4f}")
    
    # Method 5: Multi-layer trajectory
    # Track gap at each layer and predict final gap
    layer_gaps = []
    for f in all_features:
        all_h = f['all_hidden']
        Delta_W = f['Delta_W']
        lg = [np.dot(h, Delta_W) for h in all_h]
        layer_gaps.append(lg)
    
    layer_gaps = np.array(layer_gaps)  # (n_texts, n_layers)
    
    # Use early layers to predict final gap
    n_layers = layer_gaps.shape[1]
    for use_layers in [1, 5, 10, n_layers // 2]:
        if use_layers < n_layers:
            X_early = layer_gaps[:, :use_layers]
            reg = LinearRegression().fit(X_early, gaps)
            r_early = np.sqrt(max(0, reg.score(X_early, gaps)))
            print(f"  First {use_layers} layer gaps -> final gap r={r_early:.4f}")
    
    # Last layer gap before final
    if n_layers > 1:
        r_penultimate, _ = stats.pearsonr(layer_gaps[:, -2], gaps)
        print(f"  Penultimate layer gap -> final gap r={r_penultimate:.4f}")
    
    results = {
        'experiment': 'P629',
        'model': model_name,
        'r_WU_uniform': float(r_WU),
        'r_pc': float(r_pc),
        'r_prev_gap': float(r_prev_gap),
        'r_combined': float(r_combined),
    }
    
    return results


def experiment_p630(all_features, model_name, model=None):
    """P630: Unified equation v5 - prob = sigmoid(f(spectral, W_U structure))."""
    print(f"\nP630: Unified equation v5 -- {model_name}")
    print("Testing: Best Non-oracle prob prediction equation")
    
    gaps = np.array([f['logit_gap'] for f in all_features])
    probs = np.array([f['prob'] for f in all_features])
    W_U = all_features[0]['W_U']
    
    # Oracle baseline: sigmoid(gap)
    sig_oracle = 1.0 / (1.0 + np.exp(-gaps))
    r_oracle, _ = stats.pearsonr(sig_oracle, probs)
    print(f"  Oracle: sigmoid(gap) -> prob r={r_oracle:.4f}")
    
    # Model A: Pure spectral features
    X_A = []
    for f in all_features:
        row = [f['h_norm'], f['ratio50'], f['ratio90'], f['ratio99'], f['sign_ratio']]
        X_A.append(row)
    X_A = np.array(X_A)
    reg_A = LinearRegression().fit(X_A, gaps)
    pred_gaps_A = reg_A.predict(X_A)
    sig_A = 1.0 / (1.0 + np.exp(-pred_gaps_A))
    r_A, _ = stats.pearsonr(sig_A, probs)
    r_A_direct = np.sqrt(max(0, reg_A.score(X_A, gaps)))
    print(f"  Model A (spectral->gap->prob): r={r_A:.4f}, gap r={r_A_direct:.4f}")
    
    # Model B: Spectral + contribution stats
    X_B = []
    for f in all_features:
        row = [
            f['h_norm'], f['ratio50'], f['ratio90'], f['sign_ratio'],
            f['top10_contrib'] / (f['total_contrib'] + 1e-10),
            f['top50_contrib'] / (f['total_contrib'] + 1e-10),
            f['net_positive'] / (f['total_contrib'] + 1e-10),
        ]
        X_B.append(row)
    X_B = np.array(X_B)
    reg_B = LinearRegression().fit(X_B, gaps)
    pred_gaps_B = reg_B.predict(X_B)
    sig_B = 1.0 / (1.0 + np.exp(-pred_gaps_B))
    r_B, _ = stats.pearsonr(sig_B, probs)
    r_B_direct = np.sqrt(max(0, reg_B.score(X_B, gaps)))
    print(f"  Model B (spectral+contrib->gap->prob): r={r_B:.4f}, gap r={r_B_direct:.4f}")
    
    # Model C: W_U PCs
    n_samples = min(1000, W_U.shape[0])
    sample_idx = np.random.choice(W_U.shape[0], n_samples, replace=False)
    W_U_sample = W_U[sample_idx]
    pca = PCA(n_components=min(20, W_U.shape[1]))
    pca.fit(W_U_sample)
    W_U_pcs = pca.components_[:20]
    
    X_C = []
    for f in all_features:
        proj = f['h'] @ W_U_pcs.T
        X_C.append(proj)
    X_C = np.array(X_C)
    reg_C = LinearRegression().fit(X_C, gaps)
    pred_gaps_C = reg_C.predict(X_C)
    sig_C = 1.0 / (1.0 + np.exp(-pred_gaps_C))
    r_C, _ = stats.pearsonr(sig_C, probs)
    r_C_direct = np.sqrt(max(0, reg_C.score(X_C, gaps)))
    print(f"  Model C (W_U PCs->gap->prob): r={r_C:.4f}, gap r={r_C_direct:.4f}")
    
    # Model D: Previous layer gap + spectral
    prev_gaps = [np.dot(f['h_prev'], f['Delta_W']) for f in all_features]
    X_D = np.column_stack([np.array(prev_gaps).reshape(-1, 1), X_A])
    reg_D = LinearRegression().fit(X_D, gaps)
    pred_gaps_D = reg_D.predict(X_D)
    sig_D = 1.0 / (1.0 + np.exp(-pred_gaps_D))
    r_D, _ = stats.pearsonr(sig_D, probs)
    r_D_direct = np.sqrt(max(0, reg_D.score(X_D, gaps)))
    print(f"  Model D (prev_gap+spectral->prob): r={r_D:.4f}, gap r={r_D_direct:.4f}")
    
    # Model E: ALL combined
    X_E = np.column_stack([X_A, X_B[:, 5:], X_C])
    reg_E = Ridge(alpha=1.0).fit(X_E, gaps)
    pred_gaps_E = reg_E.predict(X_E)
    sig_E = 1.0 / (1.0 + np.exp(-pred_gaps_E))
    r_E, _ = stats.pearsonr(sig_E, probs)
    r_E_direct = np.sqrt(max(0, reg_E.score(X_E, gaps)))
    print(f"  Model E (ALL features->prob): r={r_E:.4f}, gap r={r_E_direct:.4f}")
    
    # Model F: Direct prob prediction (skip gap)
    reg_F = Ridge(alpha=1.0).fit(X_E, probs)
    pred_probs_F = reg_F.predict(X_E)
    r_F, _ = stats.pearsonr(pred_probs_F, probs)
    print(f"  Model F (ALL features->prob directly): r={r_F:.4f}")
    
    # Summary
    print(f"\n  === SUMMARY ===")
    print(f"  Oracle: r={r_oracle:.4f}")
    models = {'A_spectral': r_A, 'B_spectral+contrib': r_B, 'C_WU_PCs': r_C,
              'D_prev+spectral': r_D, 'E_all': r_E, 'F_direct': r_F}
    for name, r in sorted(models.items(), key=lambda x: -x[1]):
        pct = r / r_oracle * 100
        print(f"  {name}: r={r:.4f} ({pct:.1f}% of oracle)")
    
    results = {
        'experiment': 'P630',
        'model': model_name,
        'r_oracle': float(r_oracle),
        'r_A_spectral': float(r_A),
        'r_B_spectral_contrib': float(r_B),
        'r_C_WU_PCs': float(r_C),
        'r_D_prev_spectral': float(r_D),
        'r_E_all': float(r_E),
        'r_F_direct': float(r_F),
    }
    
    return results


# Experiment registry
EXPERIMENTS = {
    'p625': experiment_p625,
    'p626': experiment_p626,
    'p627': experiment_p627,
    'p628': experiment_p628,
    'p629': experiment_p629,
    'p630': experiment_p630,
}


def main():
    parser = argparse.ArgumentParser(description='Phase CXLII: Breaking h->gap Ceiling')
    parser.add_argument('--model', type=str, required=True,
                        choices=['qwen3', 'glm4', 'deepseek7b'],
                        help='Model to test')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=list(EXPERIMENTS.keys()),
                        help='Experiment to run')
    args = parser.parse_args()
    
    # Load model
    print(f"Loading {args.model}...")
    model, tokenizer, device = load_model(args.model)
    model.eval()
    
    # Compute features
    print(f"Computing features for {len(TEST_TEXTS)} texts...")
    all_features = compute_features(model, tokenizer, device, TEST_TEXTS)
    
    # Run experiment
    results = EXPERIMENTS[args.experiment](all_features, args.model, model)
    
    # Save results
    os.makedirs('results/phase_cxlii', exist_ok=True)
    result_file = f'results/phase_cxlii/{args.model}_{args.experiment}.json'
    with open(result_file, 'w') as f:
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(x) for x in obj]
            return obj
        json.dump(convert(results), f, indent=2)
    
    print(f"\nResults saved to {result_file}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
