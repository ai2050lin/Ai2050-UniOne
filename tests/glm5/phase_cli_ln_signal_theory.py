#!/usr/bin/env python3
"""
Phase CLI: LN Signal Redistribution Theory & Language Mechanism Synthesis (P664-P667)
====================================================================================

Based on Phase CL findings:
1. LN SNR gain: 164,663x (Qwen3) — LN is "signal redistributor", not whitener
2. MLP disrupts 99.3% SNR, reinjects 52x variance — MLP is "noise injector"
3. h encodes global W_U direction (r=0.41), not per-text Delta_W (r=-0.10)
4. GLM4 W_U PR=83 (concentrated), Qwen3 PR=2286 (dispersed), DS7B h PR=6.7 (too low)
5. Simple formula gap=(gamma/std)*gap_before is imprecise (r=0.29-0.78)

This phase addresses:
P664: LN Signal Redistribution — precise math of how gamma extracts gap from noise
P665: W_U Structure → Gap Encoding Efficiency — causal chain from W_U to Ridge
P666: Whitening-Noise-Rewhitening Cycle — mathematical model of the last layer
P667: Language Mechanism Synthesis — comprehensive theory of language in DNNs
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
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from model_utils import load_model, get_model_info, get_layer_weights, release_model, MODEL_CONFIGS

from phase_cxlviii_causal_intervention import ALL_TEST_TEXTS


def get_W_U(model):
    """Get unembedding matrix."""
    if hasattr(model, 'lm_head'):
        return model.lm_head.weight.detach().cpu().float().numpy()
    else:
        return model.get_output_embeddings().weight.detach().cpu().float().numpy()


def get_final_ln(model):
    """Get final layer norm module."""
    if hasattr(model.model, 'norm'):
        return model.model.norm
    elif hasattr(model.model, 'final_layernorm'):
        return model.model.final_layernorm
    return None


def get_layers(model):
    """Get transformer layers."""
    if hasattr(model.model, 'layers'):
        return model.model.layers
    return []


def experiment_p664(model, tokenizer, device, model_name):
    """P664: LN Signal Redistribution — Precise Math.
    
    Key question: How exactly does LN's gamma weight extract gap from noise?
    
    Theory development:
    gap_after_LN = LN(h) . Delta_W = sum_i gamma_i * (h_i - mu) / sigma * Delta_W_i
    
    Key insight from P660: LN does NOT whiten (variance ratio increases).
    Instead, gamma acts as a "selective amplifier" — amplifying dimensions
    where gap signal is strong and suppressing dimensions where noise dominates.
    
    Test:
    1. Per-dimension SNR: which dimensions carry gap signal before LN?
    2. Gamma alignment: does gamma correlate with SNR_before per dimension?
    3. Signal amplification: gamma * Delta_W direction correlation
    4. Decompose gap_after = gap_centering + gap_scaling
    5. Test: if we replace gamma with ones, does gap_after still correlate?
    6. Test: if we replace gamma with abs(Delta_W_mean), does correlation improve?
    """
    print(f"\n{'='*60}")
    print(f"P664: LN Signal Redistribution ({model_name})")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    layers = get_layers(model)
    d_model = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 2560
    final_ln = get_final_ln(model)
    
    n_texts = 100
    test_texts = ALL_TEST_TEXTS[:n_texts]
    
    # Get LN weight
    if final_ln is not None:
        gamma = final_ln.weight.detach().cpu().float().numpy()
        gamma_bias = final_ln.bias.detach().cpu().float().numpy() if hasattr(final_ln, 'bias') and final_ln.bias is not None else np.zeros(d_model)
    
    # Collect data
    h_before_list = []
    h_after_list = []
    logit_gaps = []
    Delta_Ws = []
    
    for i, text in enumerate(test_texts):
        if (i+1) % 20 == 0:
            print(f"  Processing text {i+1}/{n_texts}...")
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"].to(device),
                          attention_mask=inputs["attention_mask"].to(device),
                          output_hidden_states=True)
        
        h_before = outputs.hidden_states[-2][0, -1, :].cpu().float().numpy()
        h_after = outputs.hidden_states[-1][0, -1, :].cpu().float().numpy()
        
        logits = outputs.logits[0, -1, :].cpu().float().numpy()
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        
        h_before_list.append(h_before)
        h_after_list.append(h_after)
        logit_gaps.append(logits[top1_idx] - logits[top2_idx])
        Delta_Ws.append(W_U[top1_idx] - W_U[top2_idx])
    
    h_before_arr = np.array(h_before_list)  # [n_texts, d_model]
    h_after_arr = np.array(h_after_list)
    gaps = np.array(logit_gaps)
    Delta_W_arr = np.array(Delta_Ws)
    
    # ===== 1. Per-dimension SNR analysis =====
    print(f"\n1. Per-Dimension SNR Analysis:")
    
    # For each dimension i, compute:
    # signal_i = cov(h_i, gap) / var(h_i)  -- how much dimension i correlates with gap
    # noise_i = var(h_i) - signal_i^2 * var(gap)^2 / var(h_i)  -- residual variance
    
    # Per-dimension correlation with logit_gap
    dim_corr = np.zeros(d_model)
    for i in range(d_model):
        r, _ = stats.pearsonr(h_before_arr[:, i], gaps)
        dim_corr[i] = r
    
    # Per-dimension correlation with logit_gap after LN
    dim_corr_after = np.zeros(d_model)
    for i in range(d_model):
        r, _ = stats.pearsonr(h_after_arr[:, i], gaps)
        dim_corr_after[i] = r
    
    print(f"   Per-dim corr with gap BEFORE LN: mean={np.mean(np.abs(dim_corr)):.4f}, max={np.max(np.abs(dim_corr)):.4f}")
    print(f"   Per-dim corr with gap AFTER LN: mean={np.mean(np.abs(dim_corr_after)):.4f}, max={np.max(np.abs(dim_corr_after)):.4f}")
    
    # Top contributing dimensions
    top_dims_before = np.argsort(np.abs(dim_corr))[-10:]
    top_dims_after = np.argsort(np.abs(dim_corr_after))[-10:]
    print(f"   Top-10 dims BEFORE: {top_dims_before.tolist()}, corr: {[f'{dim_corr[d]:.3f}' for d in top_dims_before]}")
    print(f"   Top-10 dims AFTER:  {top_dims_after.tolist()}, corr: {[f'{dim_corr_after[d]:.3f}' for d in top_dims_after]}")
    
    # ===== 2. Gamma alignment with per-dim SNR =====
    print(f"\n2. Gamma Alignment with Per-Dim Signal:")
    
    # Does gamma align with dimensions that carry gap signal?
    corr_gamma_snr, _ = stats.pearsonr(gamma, np.abs(dim_corr))
    corr_gamma_var, _ = stats.pearsonr(gamma, np.var(h_before_arr, axis=0))
    corr_gamma_var_after, _ = stats.pearsonr(gamma, np.var(h_after_arr, axis=0))
    
    print(f"   corr(gamma, |dim_corr_before|) = {corr_gamma_snr:.4f}")
    print(f"   corr(gamma, var_before) = {corr_gamma_var:.4f}")
    print(f"   corr(gamma, var_after) = {corr_gamma_var_after:.4f}")
    
    # ===== 3. Signal amplification by gamma =====
    print(f"\n3. Signal Amplification by Gamma:")
    
    # gap_after = sum_i gamma_i * (h_i - mu) / sigma * Delta_W_i
    # Decompose into: centering contribution + scaling contribution
    
    # Compute LN manually
    h_centered = h_before_arr - np.mean(h_before_arr, axis=1, keepdims=True)
    h_std = np.std(h_before_arr, axis=1, keepdims=True) + 1e-8
    h_normalized = h_centered / h_std
    h_scaled = h_normalized * gamma[np.newaxis, :]
    
    # gap contributions
    gap_centered = np.sum(h_centered * Delta_W_arr, axis=1)
    gap_normalized = np.sum(h_normalized * Delta_W_arr, axis=1)
    gap_scaled = np.sum(h_scaled * Delta_W_arr, axis=1)
    gap_actual = np.sum(h_after_arr * Delta_W_arr, axis=1)  # Actual gap after LN
    
    r_centered, _ = stats.pearsonr(gap_centered, gaps)
    r_normalized, _ = stats.pearsonr(gap_normalized, gaps)
    r_scaled, _ = stats.pearsonr(gap_scaled, gaps)
    r_actual, _ = stats.pearsonr(gap_actual, gaps)
    
    print(f"   gap(centered h, Delta_W) -> logit_gap: r={r_centered:.4f}")
    print(f"   gap(normalized h, Delta_W) -> logit_gap: r={r_normalized:.4f}")
    print(f"   gap(gamma*normalized h, Delta_W) -> logit_gap: r={r_scaled:.4f}")
    print(f"   gap(actual h_after, Delta_W) -> logit_gap: r={r_actual:.4f}")
    
    # ===== 4. What if we remove gamma? =====
    print(f"\n4. Counterfactual: Remove/Replace Gamma:")
    
    # (a) gamma = ones
    h_no_gamma = h_normalized * 1.0
    gap_no_gamma = np.sum(h_no_gamma * Delta_W_arr, axis=1)
    r_no_gamma, _ = stats.pearsonr(gap_no_gamma, gaps)
    
    # (b) gamma = random (same distribution)
    rng = np.random.RandomState(42)
    gamma_random = rng.exponential(scale=np.mean(gamma), size=d_model)
    h_random_gamma = h_normalized * gamma_random[np.newaxis, :]
    gap_random_gamma = np.sum(h_random_gamma * Delta_W_arr, axis=1)
    r_random_gamma, _ = stats.pearsonr(gap_random_gamma, gaps)
    
    # (c) gamma = sorted by alignment with mean(Delta_W)
    mean_DW = np.mean(Delta_W_arr, axis=0)
    # Construct optimal gamma: amplify dimensions where Delta_W is strong
    gamma_optimal = np.abs(mean_DW) / (np.std(h_before_arr, axis=0) + 1e-8)
    gamma_optimal = gamma_optimal / np.mean(gamma_optimal) * np.mean(gamma)  # Same scale
    h_optimal_gamma = h_normalized * gamma_optimal[np.newaxis, :]
    gap_optimal_gamma = np.sum(h_optimal_gamma * Delta_W_arr, axis=1)
    r_optimal_gamma, _ = stats.pearsonr(gap_optimal_gamma, gaps)
    
    # (d) gamma = sign(gamma) * |Delta_W_mean| 
    gamma_dw = np.sign(gamma) * np.abs(mean_DW)
    gamma_dw = gamma_dw / np.mean(np.abs(gamma_dw)) * np.mean(gamma)
    h_dw_gamma = h_normalized * gamma_dw[np.newaxis, :]
    gap_dw_gamma = np.sum(h_dw_gamma * Delta_W_arr, axis=1)
    r_dw_gamma, _ = stats.pearsonr(gap_dw_gamma, gaps)
    
    print(f"   (a) gamma=ones:       r={r_no_gamma:.4f}")
    print(f"   (b) gamma=random:     r={r_random_gamma:.4f}")
    print(f"   (c) gamma=optimal:    r={r_optimal_gamma:.4f}")
    print(f"   (d) gamma=sign*|DW|:  r={r_dw_gamma:.4f}")
    print(f"   (e) gamma=actual:     r={r_scaled:.4f}")
    
    # ===== 5. Gamma's role: scale vs direction =====
    print(f"\n5. Gamma Scale vs Direction:")
    
    # Test: gamma magnitude only (no direction info)
    gamma_magnitude = np.abs(gamma)
    h_mag_gamma = h_normalized * gamma_magnitude[np.newaxis, :]
    gap_mag_gamma = np.sum(h_mag_gamma * Delta_W_arr, axis=1)
    r_mag_gamma, _ = stats.pearsonr(gap_mag_gamma, gaps)
    
    # Test: gamma direction only (uniform magnitude)
    gamma_sign = np.sign(gamma)
    h_sign_gamma = h_normalized * gamma_sign[np.newaxis, :]
    gap_sign_gamma = np.sum(h_sign_gamma * Delta_W_arr, axis=1)
    r_sign_gamma, _ = stats.pearsonr(gap_sign_gamma, gaps)
    
    print(f"   gamma magnitude only: r={r_mag_gamma:.4f}")
    print(f"   gamma direction only: r={r_sign_gamma:.4f}")
    print(f"   gamma (both):         r={r_scaled:.4f}")
    
    # ===== 6. Per-text variance of gap contributions =====
    print(f"\n6. Per-Text Gap Contribution Analysis:")
    
    # For each text, which dimensions contribute most to gap_after?
    gap_contribs = h_scaled * Delta_W_arr  # [n_texts, d_model]
    
    # Top-k contribution fraction
    for k in [1, 5, 10, 50]:
        topk_fracs = []
        for t in range(n_texts):
            abs_contribs = np.abs(gap_contribs[t])
            topk_sum = np.sum(np.sort(abs_contribs)[-k:])
            total_sum = np.sum(abs_contribs) + 1e-12
            topk_fracs.append(topk_sum / total_sum)
        print(f"   Top-{k:2d} dims contribute {np.mean(topk_fracs)*100:.1f}% of gap_after")
    
    # ===== 7. Key metric: effective dimension of gap information =====
    print(f"\n7. Effective Gap Information Dimension:")
    
    # PCA of h_scaled, then correlate each PC with gap
    pca = PCA(n_components=min(100, d_model))
    h_scaled_pca = pca.fit_transform(h_scaled)
    
    pc_gaps_r = []
    for pc in range(min(50, h_scaled_pca.shape[1])):
        r, _ = stats.pearsonr(h_scaled_pca[:, pc], gaps)
        pc_gaps_r.append(r)
    
    pc_gaps_r = np.array(pc_gaps_r)
    n_significant = np.sum(np.abs(pc_gaps_r) > 0.2)
    n_highly_sig = np.sum(np.abs(pc_gaps_r) > 0.5)
    
    print(f"   PCs with |r|>0.2: {n_significant}")
    print(f"   PCs with |r|>0.5: {n_highly_sig}")
    print(f"   Top-5 PC correlations: {[f'{r:.3f}' for r in pc_gaps_r[:5]]}")
    
    # Results
    results = {
        "dim_corr_mean_before": float(np.mean(np.abs(dim_corr))),
        "dim_corr_mean_after": float(np.mean(np.abs(dim_corr_after))),
        "dim_corr_max_before": float(np.max(np.abs(dim_corr))),
        "dim_corr_max_after": float(np.max(np.abs(dim_corr_after))),
        "corr_gamma_snr": float(corr_gamma_snr),
        "corr_gamma_var": float(corr_gamma_var),
        "r_centered": float(r_centered),
        "r_normalized": float(r_normalized),
        "r_scaled": float(r_scaled),
        "r_actual": float(r_actual),
        "r_no_gamma": float(r_no_gamma),
        "r_random_gamma": float(r_random_gamma),
        "r_optimal_gamma": float(r_optimal_gamma),
        "r_dw_gamma": float(r_dw_gamma),
        "r_mag_gamma": float(r_mag_gamma),
        "r_sign_gamma": float(r_sign_gamma),
        "n_significant_pcs": int(n_significant),
        "n_highly_sig_pcs": int(n_highly_sig),
        "top5_pc_corrs": [float(x) for x in pc_gaps_r[:5]],
    }
    
    return results


def experiment_p665(model, tokenizer, device, model_name):
    """P665: W_U Structure → Gap Encoding Efficiency.
    
    Key question: How does W_U structure determine gap encoding efficiency?
    
    Theory: 
    gap = h . Delta_W = h . (W_U[top1] - W_U[top2])
    
    If W_U rows are spread in many directions (high PR), then:
    - Delta_W directions are diverse → h needs many dimensions to encode
    - Ridge needs high k → overfitting
    
    If W_U rows are concentrated in few directions (low PR), then:
    - Delta_W directions are similar → h needs few dimensions
    - Ridge needs low k → less overfitting
    
    Tests:
    1. Delta_W diversity: PCA of Delta_W across texts
    2. W_U row space structure: how many directions are needed?
    3. Correlation: W_U PR vs Ridge best r across models
    4. Per-text: Delta_W alignment with top W_U directions
    5. Constructive vs destructive interference in gap
    """
    print(f"\n{'='*60}")
    print(f"P665: W_U Structure → Gap Encoding ({model_name})")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    d_model = W_U.shape[1]
    n_vocab = W_U.shape[0]
    
    n_texts = 100
    test_texts = ALL_TEST_TEXTS[:n_texts]
    
    # Collect data
    h_before_list = []
    logit_gaps = []
    Delta_Ws = []
    top1_indices = []
    top2_indices = []
    
    for i, text in enumerate(test_texts):
        if (i+1) % 20 == 0:
            print(f"  Processing text {i+1}/{n_texts}...")
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"].to(device),
                          attention_mask=inputs["attention_mask"].to(device),
                          output_hidden_states=True)
        
        h_before = outputs.hidden_states[-2][0, -1, :].cpu().float().numpy()
        logits = outputs.logits[0, -1, :].cpu().float().numpy()
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        
        h_before_list.append(h_before)
        logit_gaps.append(logits[top1_idx] - logits[top2_idx])
        Delta_Ws.append(W_U[top1_idx] - W_U[top2_idx])
        top1_indices.append(top1_idx)
        top2_indices.append(top2_idx)
    
    h_arr = np.array(h_before_list)
    gaps = np.array(logit_gaps)
    Delta_W_arr = np.array(Delta_Ws)
    
    # ===== 1. Delta_W diversity =====
    print(f"\n1. Delta_W Diversity:")
    
    # PCA of Delta_W
    pca_DW = PCA(n_components=min(50, n_texts))
    pca_DW.fit(Delta_W_arr)
    
    DW_PR = (np.sum(pca_DW.explained_variance_ratio_))**2 / np.sum(pca_DW.explained_variance_ratio_**2)
    
    print(f"   Delta_W participation ratio: {DW_PR:.1f}")
    for k in [1, 3, 5, 10, 20]:
        energy_k = np.sum(pca_DW.explained_variance_ratio_[:k]) * 100
        print(f"   Top-{k:2d} Delta_W PCs: {energy_k:.1f}% of variance")
    
    # ===== 2. W_U row space structure =====
    print(f"\n2. W_U Row Space Structure:")
    
    # Sample W_U rows for tractability
    n_sample = min(1000, n_vocab)
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(n_vocab, n_sample, replace=False)
    W_U_sample = W_U[sample_idx]
    
    pca_W = PCA(n_components=min(100, d_model))
    pca_W.fit(W_U_sample)
    
    W_PR = (np.sum(pca_W.explained_variance_ratio_))**2 / np.sum(pca_W.explained_variance_ratio_**2)
    
    print(f"   W_U participation ratio (sampled): {W_PR:.1f}")
    for k in [1, 5, 10, 20, 50]:
        energy_k = np.sum(pca_W.explained_variance_ratio_[:k]) * 100
        print(f"   Top-{k:2d} W_U PCs: {energy_k:.1f}% of variance")
    
    # ===== 3. Delta_W alignment with W_U principal directions =====
    print(f"\n3. Delta_W Alignment with W_U PCs:")
    
    # Project each Delta_W onto W_U PCs
    DW_in_WU_space = pca_W.transform(Delta_W_arr)  # [n_texts, n_components]
    
    # How much of Delta_W variance is captured by top-k W_U PCs?
    for k in [1, 5, 10, 20]:
        DW_reconstructed = DW_in_WU_space[:, :k] @ pca_W.components_[:k]
        residual = Delta_W_arr - DW_reconstructed
        capture_ratio = 1 - np.mean(np.sum(residual**2, axis=1)) / np.mean(np.sum(Delta_W_arr**2, axis=1))
        print(f"   Top-{k:2d} W_U PCs capture {capture_ratio*100:.1f}% of Delta_W variance")
    
    # ===== 4. Constructive vs destructive interference =====
    print(f"\n4. Constructive vs Destructive Interference in gap:")
    
    # gap = h . Delta_W = sum_i h_i * Delta_W_i
    # Decompose into positive (constructive) and negative (destructive) parts
    gap_contribs = h_arr * Delta_W_arr  # [n_texts, d_model]
    
    positive_contribs = np.sum(gap_contribs * (gap_contribs > 0), axis=1)
    negative_contribs = np.sum(gap_contribs * (gap_contribs < 0), axis=1)
    total_abs = np.sum(np.abs(gap_contribs), axis=1)
    
    print(f"   Mean positive contribution: {np.mean(positive_contribs):.2f}")
    print(f"   Mean negative contribution: {np.mean(negative_contribs):.2f}")
    print(f"   Mean |total|: {np.mean(total_abs):.2f}")
    print(f"   Net/Total ratio (constructive fraction): {np.mean((positive_contribs + negative_contribs) / (total_abs + 1e-12)):.4f}")
    
    # ===== 5. How many W_U directions are needed to distinguish top1 from top2? =====
    print(f"\n5. W_U Directions Needed for Top-1 vs Top-2:")
    
    # For each text, project logits onto W_U PCs and see how many PCs are needed
    # to correctly rank top1 > top2
    logit_data = []
    for i, text in enumerate(test_texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"].to(device),
                          attention_mask=inputs["attention_mask"].to(device),
                          output_hidden_states=True)
        logits = outputs.logits[0, -1, :].cpu().float().numpy()
        logit_data.append(logits)
    
    # Project full logits onto W_U PCs
    # logits = h_final @ W_U.T + bias
    # In PC space: logits_in_pc = h_final @ (W_U.T) in PC space
    # This is equivalent to: logits = sum_k (h_final . PC_k) * (W_U rows . PC_k)
    
    # Alternative: directly compute how many W_U PCs are needed
    # to maintain top1 > top2 ranking
    W_U_pcs = pca_W.components_[:50]  # [50, d_model]
    
    correct_rankings = []
    for k in [1, 3, 5, 10, 20, 50]:
        # Reconstruct W_U in k-dimensional space
        W_U_approx = np.zeros_like(W_U)
        for pc_idx in range(k):
            proj = W_U @ W_U_pcs[pc_idx]
            W_U_approx += np.outer(proj, W_U_pcs[pc_idx])
        
        # Check if top1 > top2 in approximated logits
        n_correct = 0
        for i in range(n_texts):
            h_final_approx = h_arr[i]  # Use h_{L-1} for simplicity
            logits_approx = h_final_approx @ W_U_approx.T
            if logits_approx[top1_indices[i]] > logits_approx[top2_indices[i]]:
                n_correct += 1
        correct_rankings.append(n_correct / n_texts)
        print(f"   Top-{k:2d} W_U PCs: {n_correct}/{n_texts} correct top-1>top-2 ranking ({n_correct/n_texts*100:.0f}%)")
    
    # ===== 6. Ridge prediction using W_U PCs =====
    print(f"\n6. Ridge Prediction Using W_U PC Projections:")
    
    # Project h onto W_U PCs, then use those as Ridge features
    h_in_WU = pca_W.transform(h_arr)  # [n_texts, n_components]
    
    ridge_results = []
    for k in [1, 3, 5, 10, 20, 50]:
        X = h_in_WU[:, :k]
        y = gaps
        
        # LOO Ridge
        loo = LeaveOneOut()
        preds = np.zeros(n_texts)
        for train_idx, test_idx in loo.split(X):
            reg = Ridge(alpha=1.0)
            reg.fit(X[train_idx], y[train_idx])
            preds[test_idx[0]] = reg.predict(X[test_idx])[0]
        
        r, _ = stats.pearsonr(preds, y)
        ridge_results.append({"k": k, "r_loo": float(r)})
        print(f"   k={k:2d}: Ridge LOO r={r:.4f}")
    
    best_ridge = max(ridge_results, key=lambda x: x["r_loo"])
    
    results = {
        "DW_PR": float(DW_PR),
        "W_U_PR": float(W_PR),
        "DW_top5_energy": float(np.sum(pca_DW.explained_variance_ratio_[:5]) * 100),
        "W_U_top5_energy": float(np.sum(pca_W.explained_variance_ratio_[:5]) * 100),
        "DW_top10_energy": float(np.sum(pca_DW.explained_variance_ratio_[:10]) * 100),
        "W_U_top10_energy": float(np.sum(pca_W.explained_variance_ratio_[:10]) * 100),
        "constructive_fraction": float(np.mean((positive_contribs + negative_contribs) / (total_abs + 1e-12))),
        "correct_rankings": [float(x) for x in correct_rankings],
        "ridge_results": ridge_results,
        "best_ridge_k": best_ridge["k"],
        "best_ridge_r": best_ridge["r_loo"],
    }
    
    return results


def experiment_p666(model, tokenizer, device, model_name):
    """P666: Whitening-Noise-Rewhitening Cycle Model.
    
    Key question: Mathematical model of the last layer's processing.
    
    Signal flow: h_{L-1} -> input_LN -> Attn -> post_attn_LN -> MLP -> final_LN -> h_L
    
    Key findings:
    - post-attn LN: first emergence (r=0.5 for Qwen3/GLM4)
    - MLP: disrupts (r drops to ~0.05)
    - final LN: re-emergence (r=0.997)
    
    Mathematical model:
    Let gap_t = gap at sub-step t
    gap_0 = gap(h_{L-1}, Delta_W)  -- no info
    gap_1 = gap(LN1(h_{L-1}), Delta_W)  -- some info
    gap_2 = gap(Attn(LN1(h)), Delta_W)  -- no info (Attn mixes)
    gap_3 = gap(LN2(Attn(LN1(h))), Delta_W)  -- emergence!
    gap_4 = gap(MLP(LN2(...)), Delta_W)  -- disruption
    gap_5 = gap(LN3(MLP(...)), Delta_W)  -- re-emergence!
    
    Hypothesis: Each LN acts as a "whitening step" that extracts gap info
    from the noise. MLP acts as "noise injection" that destroys gap info.
    The final LN must extract gap info that survived MLP disruption.
    
    Tests:
    1. Quantify LN "extraction power" at each LN step
    2. Quantify MLP "injection power" 
    3. Is the cycle: extraction → injection → re-extraction?
    4. How much residual gap info survives MLP?
    """
    print(f"\n{'='*60}")
    print(f"P666: Whitening-Noise-Rewhitening Cycle ({model_name})")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    layers = get_layers(model)
    last_layer = layers[-1]
    d_model = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 2560
    
    n_texts = 80
    test_texts = ALL_TEST_TEXTS[:n_texts]
    
    # Collect h at each sub-step
    # Sub-steps: h_{L-1}, after input LN, after Attn, after post-attn LN, after MLP, after final LN
    
    substep_names = ['h_Lm1', 'after_input_LN', 'after_attn', 'after_postattn_LN', 'after_mlp', 'after_final_LN']
    substep_h = {name: [] for name in substep_names}
    logit_gaps = []
    Delta_Ws = []
    
    # Hook functions
    hook_outputs = {}
    
    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hook_outputs[name] = output[0].detach().cpu().float()
            else:
                hook_outputs[name] = output.detach().cpu().float()
        return hook_fn
    
    # Register hooks on last layer
    # This is model-specific, need to handle different architectures
    hooks = []
    
    # For Qwen2/GLM/DS architectures
    if hasattr(last_layer, 'input_layernorm'):
        hooks.append(last_layer.input_layernorm.register_forward_hook(make_hook('input_ln')))
    if hasattr(last_layer, 'self_attn'):
        hooks.append(last_layer.self_attn.register_forward_hook(make_hook('attn')))
    if hasattr(last_layer, 'post_attention_layernorm'):
        hooks.append(last_layer.post_attention_layernorm.register_forward_hook(make_hook('post_attn_ln')))
    if hasattr(last_layer, 'mlp'):
        hooks.append(last_layer.mlp.register_forward_hook(make_hook('mlp')))
    
    final_ln = get_final_ln(model)
    if final_ln is not None:
        hooks.append(final_ln.register_forward_hook(make_hook('final_ln')))
    
    for i, text in enumerate(test_texts):
        if (i+1) % 20 == 0:
            print(f"  Processing text {i+1}/{n_texts}...")
        
        hook_outputs.clear()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"].to(device),
                          attention_mask=inputs["attention_mask"].to(device),
                          output_hidden_states=True)
        
        # h_{L-1}
        h_Lm1 = outputs.hidden_states[-2][0, -1, :].cpu().float().numpy()
        substep_h['h_Lm1'].append(h_Lm1)
        
        # After input LN
        if 'input_ln' in hook_outputs:
            h_input_ln = hook_outputs['input_ln'][0, -1, :].numpy()
            substep_h['after_input_LN'].append(h_input_ln)
        else:
            substep_h['after_input_LN'].append(h_Lm1)  # fallback
        
        # After attention
        if 'attn' in hook_outputs:
            h_attn = hook_outputs['attn'][0, -1, :].numpy()
            # h_after_attn = h_input_ln + h_attn (residual connection)
            h_input_ln_tensor = hook_outputs['input_ln'][0, -1, :].numpy() if 'input_ln' in hook_outputs else h_Lm1
            substep_h['after_attn'].append(h_input_ln_tensor + h_attn)
        else:
            substep_h['after_attn'].append(h_Lm1)
        
        # After post-attn LN
        if 'post_attn_ln' in hook_outputs:
            h_post_attn_ln = hook_outputs['post_attn_ln'][0, -1, :].numpy()
            substep_h['after_postattn_LN'].append(h_post_attn_ln)
        else:
            substep_h['after_postattn_LN'].append(h_Lm1)
        
        # After MLP
        if 'mlp' in hook_outputs:
            h_mlp = hook_outputs['mlp'][0, -1, :].numpy()
            # h_after_mlp = h_post_attn_ln + h_mlp (residual connection)
            h_post_ln_tensor = hook_outputs['post_attn_ln'][0, -1, :].numpy() if 'post_attn_ln' in hook_outputs else h_Lm1
            substep_h['after_mlp'].append(h_post_ln_tensor + h_mlp)
        else:
            substep_h['after_mlp'].append(h_Lm1)
        
        # After final LN
        if 'final_ln' in hook_outputs:
            h_final_ln = hook_outputs['final_ln'][0, -1, :].numpy()
            substep_h['after_final_LN'].append(h_final_ln)
        else:
            substep_h['after_final_LN'].append(outputs.hidden_states[-1][0, -1, :].cpu().float().numpy())
        
        # Logit gap
        logits = outputs.logits[0, -1, :].cpu().float().numpy()
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        logit_gaps.append(logits[top1_idx] - logits[top2_idx])
        Delta_Ws.append(W_U[top1_idx] - W_U[top2_idx])
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    gaps = np.array(logit_gaps)
    Delta_W_arr = np.array(Delta_Ws)
    
    # ===== 1. Gap correlation at each sub-step =====
    print(f"\n1. Gap Correlation at Each Sub-Step:")
    
    substep_r = {}
    for name in substep_names:
        h_arr = np.array(substep_h[name])
        gap_values = np.sum(h_arr * Delta_W_arr, axis=1)
        r, _ = stats.pearsonr(gap_values, gaps)
        substep_r[name] = r
        print(f"   {name:25s}: r = {r:.4f}")
    
    # ===== 2. LN extraction power =====
    print(f"\n2. LN Extraction Power (r_after - r_before):")
    
    # input LN: h_{L-1} -> after_input_LN
    delta_r_input_ln = substep_r['after_input_LN'] - substep_r['h_Lm1']
    print(f"   input LN:  delta_r = {delta_r_input_ln:.4f}")
    
    # post-attn LN: after_attn -> after_postattn_LN
    delta_r_post_attn_ln = substep_r['after_postattn_LN'] - substep_r['after_attn']
    print(f"   post-attn LN: delta_r = {delta_r_post_attn_ln:.4f}")
    
    # final LN: after_mlp -> after_final_LN
    delta_r_final_ln = substep_r['after_final_LN'] - substep_r['after_mlp']
    print(f"   final LN:   delta_r = {delta_r_final_ln:.4f}")
    
    # ===== 3. MLP injection power =====
    print(f"\n3. MLP Injection Power (r_after - r_before):")
    
    # Attn: after_input_LN -> after_attn
    delta_r_attn = substep_r['after_attn'] - substep_r['after_input_LN']
    print(f"   Attn:  delta_r = {delta_r_attn:.4f}")
    
    # MLP: after_postattn_LN -> after_mlp
    delta_r_mlp = substep_r['after_mlp'] - substep_r['after_postattn_LN']
    print(f"   MLP:   delta_r = {delta_r_mlp:.4f}")
    
    # ===== 4. Residual gap info surviving MLP =====
    print(f"\n4. Residual Gap Info After MLP:")
    
    # Even though gap(h_mlp, Delta_W) r≈0, does gap info survive in some form?
    # Test: Ridge on h_after_mlp features
    h_mlp_arr = np.array(substep_h['after_mlp'])
    
    pca = PCA(n_components=min(50, d_model))
    h_mlp_pca = pca.fit_transform(h_mlp_arr)
    
    ridge_loo = LeaveOneOut()
    preds_loo = np.zeros(n_texts)
    X = h_mlp_pca[:, :20]
    y = gaps
    for train_idx, test_idx in ridge_loo.split(X):
        reg = Ridge(alpha=1.0)
        reg.fit(X[train_idx], y[train_idx])
        preds_loo[test_idx[0]] = reg.predict(X[test_idx])[0]
    
    r_ridge_mlp, _ = stats.pearsonr(preds_loo, y)
    print(f"   Ridge on h_after_mlp (k=20): r = {r_ridge_mlp:.4f}")
    
    # ===== 5. SNR at each sub-step =====
    print(f"\n5. SNR at Each Sub-Step:")
    
    substep_snr = {}
    for name in substep_names:
        h_arr = np.array(substep_h[name])
        gap_values = np.sum(h_arr * Delta_W_arr, axis=1)
        r, _ = stats.pearsonr(gap_values, gaps)
        # SNR = r^2 / (1 - r^2) (Fisher's signal-to-noise ratio)
        snr = r**2 / (1 - r**2 + 1e-12)
        substep_snr[name] = snr
        print(f"   {name:25s}: SNR = {snr:.4f}")
    
    # ===== 6. Direction analysis =====
    print(f"\n6. Direction Analysis Between Sub-Steps:")
    
    for i in range(len(substep_names)-1):
        h1 = np.array(substep_h[substep_names[i]])
        h2 = np.array(substep_h[substep_names[i+1]])
        
        # Average cosine similarity between consecutive sub-steps
        cos_sims = []
        for t in range(n_texts):
            h1_norm = h1[t] / (np.linalg.norm(h1[t]) + 1e-12)
            h2_norm = h2[t] / (np.linalg.norm(h2[t]) + 1e-12)
            cos_sims.append(np.dot(h1_norm, h2_norm))
        
        print(f"   {substep_names[i]:25s} -> {substep_names[i+1]:25s}: cos = {np.mean(cos_sims):.4f}")
    
    results = {
        "substep_r": {k: float(v) for k, v in substep_r.items()},
        "substep_snr": {k: float(v) for k, v in substep_snr.items()},
        "delta_r_input_ln": float(delta_r_input_ln),
        "delta_r_post_attn_ln": float(delta_r_post_attn_ln),
        "delta_r_final_ln": float(delta_r_final_ln),
        "delta_r_attn": float(delta_r_attn),
        "delta_r_mlp": float(delta_r_mlp),
        "r_ridge_mlp_k20": float(r_ridge_mlp),
    }
    
    return results


def experiment_p667(model, tokenizer, device, model_name):
    """P667: Language Mechanism Synthesis.
    
    Key question: Comprehensive theory of how language works in DNNs.
    
    Based on all findings so far, synthesize a unified model:
    
    Language = W_U Structure × h Dimensionality × LN Signal Redistribution × MLP Noise
    
    Components:
    1. W_U provides the "vocabulary space" — its concentration determines
       how many h dimensions are needed to encode gap
    2. h dimensionality (PR) determines encoding capacity
    3. LN extracts gap signal from noise via gamma-weighted normalization
    4. MLP injects noise but gap info survives in hidden form
    
    Test the unified model:
    1. Predict Ridge r from W_U PR, h PR, W_U-h overlap
    2. Cross-model consistency check
    3. Identify the "bottleneck" for each model
    """
    print(f"\n{'='*60}")
    print(f"P667: Language Mechanism Synthesis ({model_name})")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    d_model = W_U.shape[1]
    
    n_texts = 100
    test_texts = ALL_TEST_TEXTS[:n_texts]
    
    # ===== 1. Compute all model characteristics =====
    print(f"\n1. Model Characteristics:")
    
    # W_U PR (sampled)
    n_sample = min(1000, W_U.shape[0])
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(W_U.shape[0], n_sample, replace=False)
    pca_W = PCA(n_components=min(100, d_model))
    pca_W.fit(W_U[sample_idx])
    W_PR = (np.sum(pca_W.explained_variance_ratio_))**2 / np.sum(pca_W.explained_variance_ratio_**2)
    
    # W_U top-k concentration
    W_top5 = np.sum(pca_W.explained_variance_ratio_[:5]) * 100
    W_top10 = np.sum(pca_W.explained_variance_ratio_[:10]) * 100
    W_top50 = np.sum(pca_W.explained_variance_ratio_[:50]) * 100
    
    print(f"   W_U PR = {W_PR:.1f}")
    print(f"   W_U top-5/10/50 energy: {W_top5:.1f}% / {W_top10:.1f}% / {W_top50:.1f}%")
    
    # h PR and other characteristics
    h_list = []
    for i, text in enumerate(test_texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"].to(device),
                          attention_mask=inputs["attention_mask"].to(device),
                          output_hidden_states=True)
        h_list.append(outputs.hidden_states[-2][0, -1, :].cpu().float().numpy())
    
    h_arr = np.array(h_list)
    
    pca_h = PCA(n_components=min(50, d_model))
    pca_h.fit(h_arr)
    h_PR = (np.sum(pca_h.explained_variance_ratio_))**2 / np.sum(pca_h.explained_variance_ratio_**2)
    
    print(f"   h PR = {h_PR:.1f}")
    print(f"   h top-5 energy: {np.sum(pca_h.explained_variance_ratio_[:5])*100:.1f}%")
    
    # ===== 2. W_U-h overlap =====
    print(f"\n2. W_U-h Subspace Overlap:")
    
    overlaps = {}
    for k in [5, 10, 20]:
        W_U_dirs = pca_W.components_[:k]
        h_dirs = pca_h.components_[:k]
        
        # Subspace overlap: mean of |cos(W_U_dir_i, h_dir_j)|
        cos_matrix = np.abs(W_U_dirs @ h_dirs.T)
        overlap = np.mean(cos_matrix)
        overlaps[k] = overlap
        print(f"   Top-{k:2d} overlap: {overlap:.4f}")
    
    # ===== 3. Ridge prediction using W_U PCs =====
    print(f"\n3. Ridge Prediction Using W_U PC Features:")
    
    # Project h onto W_U PCs
    h_in_WU = pca_W.transform(h_arr)  # [n_texts, n_components]
    
    # Get logit gaps
    logit_gaps = []
    for i, text in enumerate(test_texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"].to(device),
                          attention_mask=inputs["attention_mask"].to(device),
                          output_hidden_states=True)
        logits = outputs.logits[0, -1, :].cpu().float().numpy()
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        logit_gaps.append(logits[top1_idx] - logits[top2_idx])
    
    gaps = np.array(logit_gaps)
    
    # Ridge with W_U PC features
    ridge_WU_results = []
    for k in [1, 3, 5, 10, 20, 50]:
        X = h_in_WU[:, :k]
        y = gaps
        
        loo = LeaveOneOut()
        preds = np.zeros(n_texts)
        for train_idx, test_idx in loo.split(X):
            reg = Ridge(alpha=1.0)
            reg.fit(X[train_idx], y[train_idx])
            preds[test_idx[0]] = reg.predict(X[test_idx])[0]
        
        r, _ = stats.pearsonr(preds, y)
        ridge_WU_results.append({"k": k, "r_loo": float(r)})
        print(f"   W_U PC Ridge k={k:2d}: r={r:.4f}")
    
    # ===== 4. Ridge with h PCs =====
    print(f"\n4. Ridge Prediction Using h PC Features:")
    
    h_in_h = pca_h.transform(h_arr)
    
    ridge_h_results = []
    for k in [1, 3, 5, 10, 20, 50]:
        if k > h_in_h.shape[1]:
            continue
        X = h_in_h[:, :k]
        y = gaps
        
        loo = LeaveOneOut()
        preds = np.zeros(n_texts)
        for train_idx, test_idx in loo.split(X):
            reg = Ridge(alpha=1.0)
            reg.fit(X[train_idx], y[train_idx])
            preds[test_idx[0]] = reg.predict(X[test_idx])[0]
        
        r, _ = stats.pearsonr(preds, y)
        ridge_h_results.append({"k": k, "r_loo": float(r)})
        print(f"   h PC Ridge k={k:2d}: r={r:.4f}")
    
    # ===== 5. Bottleneck analysis =====
    print(f"\n5. Bottleneck Analysis:")
    
    best_WU_r = max(ridge_WU_results, key=lambda x: x["r_loo"])
    best_h_r = max(ridge_h_results, key=lambda x: x["r_loo"])
    
    print(f"   Best Ridge with W_U PCs: k={best_WU_r['k']}, r={best_WU_r['r_loo']:.4f}")
    print(f"   Best Ridge with h PCs:   k={best_h_r['k']}, r={best_h_r['r_loo']:.4f}")
    
    # Bottleneck identification
    if W_PR > 200:
        print(f"   Bottleneck: W_U too dispersed (PR={W_PR:.0f}) → gap info scattered")
    elif h_PR < 10:
        print(f"   Bottleneck: h too low-dimensional (PR={h_PR:.0f}) → insufficient encoding capacity")
    else:
        print(f"   Bottleneck: W_U-h misalignment (overlap={overlaps.get(10, 0):.3f}) → encoding inefficiency")
    
    # ===== 6. Cross-model prediction (qualitative) =====
    print(f"\n6. Unified Model Prediction:")
    
    # Ridge r ∝ (W_U concentration) × (h dimensionality) × (W_U-h overlap)
    # Simplified: Ridge_r ~ C * (1/log(W_PR)) * min(h_PR, optimal_k) * overlap
    
    # For this model:
    encoding_capacity = min(h_PR, 50)  # h can't encode more than its dimensionality
    W_concentration = 1.0 / np.log(W_PR + 1)
    predicted_r_raw = W_concentration * encoding_capacity * overlaps.get(10, 0)
    print(f"   W_U concentration factor: 1/log(PR+1) = {W_concentration:.4f}")
    print(f"   Encoding capacity: min(h_PR, 50) = {encoding_capacity:.1f}")
    print(f"   W_U-h overlap (k=10): {overlaps.get(10, 0):.4f}")
    print(f"   Predicted r (raw): {predicted_r_raw:.4f}")
    print(f"   Actual best r (W_U PCs): {best_WU_r['r_loo']:.4f}")
    
    results = {
        "W_PR": float(W_PR),
        "h_PR": float(h_PR),
        "W_top5": float(W_top5),
        "W_top10": float(W_top10),
        "W_top50": float(W_top50),
        "overlaps": {str(k): float(v) for k, v in overlaps.items()},
        "ridge_WU_results": ridge_WU_results,
        "ridge_h_results": ridge_h_results,
        "best_WU_r": best_WU_r,
        "best_h_r": best_h_r,
        "encoding_capacity": float(encoding_capacity),
        "W_concentration": float(W_concentration),
        "predicted_r_raw": float(predicted_r_raw),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase CLI: LN Signal Theory & Synthesis")
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True, choices=["p664", "p665", "p666", "p667"])
    args = parser.parse_args()
    
    model_name = args.model
    experiment = args.experiment
    
    print(f"Loading model: {model_name}")
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"Model: {model_info.name}, d_model={model_info.d_model}, n_layers={model_info.n_layers}")
    
    start_time = time.time()
    
    if experiment == "p664":
        results = experiment_p664(model, tokenizer, device, model_name)
    elif experiment == "p665":
        results = experiment_p665(model, tokenizer, device, model_name)
    elif experiment == "p666":
        results = experiment_p666(model, tokenizer, device, model_name)
    elif experiment == "p667":
        results = experiment_p667(model, tokenizer, device, model_name)
    
    elapsed = time.time() - start_time
    
    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "phase_cli")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{experiment}_{model_name}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_file}")
    print(f"Time: {elapsed:.1f}s")
    
    # Cleanup
    release_model(model)
    print("Model released.")


if __name__ == "__main__":
    main()
