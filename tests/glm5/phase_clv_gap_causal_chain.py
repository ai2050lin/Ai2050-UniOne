#!/usr/bin/env python3
"""
Phase CLV: Gap Causal Chain Repair & Gamma Nonlinear Decoding (P680-P683)
==========================================================================

Based on Phase CLIV findings:
1. Eq2(gamma+universal) boosted Qwen3 by 77%, but still r=0.495 (far from 1.0)
2. GLM4 gamma > Ridge optimal — gamma encodes nonlinear info
3. Qwen3 gamma is high-frequency (84%) — far from W_U_col_std
4. Spectrum->logit_gap causal chain is broken (r<0.12)
5. Format vs Content direction function differs across models

This phase addresses:
P680: Dual-Channel Spectrum→logit_gap Model — format spectrum + content spectrum
P681: Gamma Nonlinear Information Decoding — what does GLM4 gamma encode?
P682: Qwen3 Gamma High-Frequency Source — training/architecture origin
P683: Cross-Layer Gap Information Propagation — precise LN-MLP-LN tracking
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
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from model_utils import load_model, get_model_info, get_layer_weights, release_model, MODEL_CONFIGS

# Reuse test texts from previous phase
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


def extract_tensor(t):
    """Extract last token from tensor with flexible dim handling."""
    if t.dim() == 3:
        return t[0, -1, :].float().numpy()
    elif t.dim() == 2:
        return t[-1, :].float().numpy()
    else:
        return t.float().numpy().flatten()


def compute_spectrum_directions(W_U, n_dirs=20):
    """Compute W_U spectral directions via SVD.
    
    Returns:
        U_k: top-k left singular vectors [k, d_model]
        s_k: top-k singular values [k]
    """
    # Use TruncatedSVD for memory efficiency
    n_components = min(n_dirs, min(W_U.shape) - 1)
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(W_U)
    U_k = svd.components_  # [k, d_model]
    s_k = svd.singular_values_  # [k]
    return U_k, s_k


def classify_format_content_directions(U_k, W_U, tokenizer, model, top_k_words=50):
    """Classify spectral directions as format or content based on top-loaded words.
    
    Strategy:
    - For each direction k, find top-k words with highest |W_U @ U_k[k]|
    - Count punctuation/formatting words vs content words
    - Format direction: >50% punctuation/connectors
    - Content direction: >50% content words
    """
    n_dirs = U_k.shape[0]
    direction_types = []
    
    # Simple heuristic: project W_U rows onto each direction
    # W_U_proj[k] = W_U @ U_k[k] -> how much each word loads on direction k
    format_keywords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at',
                       'to', 'for', 'of', 'and', 'or', 'but', '.', ',', '!', '?', ';',
                       ':', '-', '(', ')', '"', "'", '\n', '\t', '##', '<', '>'}
    
    for k in range(n_dirs):
        # Project all words onto direction k
        word_loadings = W_U @ U_k[k]  # [vocab_size]
        top_indices = np.argsort(np.abs(word_loadings))[-top_k_words:]
        
        # Try to decode top words
        format_count = 0
        content_count = 0
        unknown_count = 0
        
        for idx in top_indices:
            try:
                token = tokenizer.decode([idx])
                token_lower = token.strip().lower()
                if token_lower in format_keywords or not token.strip().isalpha():
                    format_count += 1
                elif len(token.strip()) > 2:
                    content_count += 1
                else:
                    unknown_count += 1
            except:
                unknown_count += 1
        
        total = format_count + content_count + unknown_count
        if total > 0:
            format_ratio = format_count / total
        else:
            format_ratio = 0.5
        
        if format_ratio > 0.5:
            direction_types.append('format')
        elif format_ratio < 0.3:
            direction_types.append('content')
        else:
            direction_types.append('mixed')
    
    return direction_types


def experiment_p680(model, tokenizer, device, model_name):
    """P680: Dual-Channel Spectrum→logit_gap Model.
    
    Key question: Can separating format and content spectral directions 
    repair the broken spectrum→logit_gap causal chain?
    
    From Phase CXXXIV-CXXXVI:
    - Format vs content direction function differs across models
    - Qwen3: content positive (r=0.177), format weak (r=0.092)
    - GLM4: format contributes 200%, content negative (r=-0.019)
    - DS7B: content positive (r=0.221), format negative (r=-0.129)
    - sign(c_k·Delta_k) ~ random (agreement~0.5)
    
    Tests:
    1. Classify top-20 spectral directions as format/content/mixed
    2. Compute per-text format_gap and content_gap separately
    3. Fit dual-channel model: logit_gap ~ format_gap + content_gap
    4. Compare with single-channel model
    5. Analyze per-direction contribution with sign correction
    """
    print(f"\n{'='*60}")
    print(f"P680: Dual-Channel Spectrum→logit_gap ({model_name})")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    d_model = W_U.shape[1]
    final_ln = get_final_ln(model)
    
    if final_ln is not None:
        gamma = final_ln.weight.detach().cpu().float().numpy()
    else:
        print("  No final LN found, skipping.")
        return {}
    
    # Compute spectral directions
    print("  Computing W_U spectral directions...")
    n_dirs = 20
    U_k, s_k = compute_spectrum_directions(W_U, n_dirs=n_dirs)
    
    # Classify directions
    print("  Classifying format/content directions...")
    dir_types = classify_format_content_directions(U_k, W_U, tokenizer, model)
    
    format_dirs = [i for i, t in enumerate(dir_types) if t == 'format']
    content_dirs = [i for i, t in enumerate(dir_types) if t == 'content']
    mixed_dirs = [i for i, t in enumerate(dir_types) if t == 'mixed']
    
    print(f"  Direction classification: format={len(format_dirs)}, "
          f"content={len(content_dirs)}, mixed={len(mixed_dirs)}")
    print(f"  Types: {dir_types}")
    
    # Collect per-text data
    n_texts = 100
    test_texts = ALL_TEST_TEXTS[:n_texts]
    
    h_final_list = []
    logit_gaps = []
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
        
        h_final = outputs.hidden_states[-1][0, -1, :].cpu().float().numpy()
        logits = outputs.logits[0, -1, :].cpu().float().numpy()
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        
        h_final_list.append(h_final)
        logit_gaps.append(logits[top1_idx] - logits[top2_idx])
        top1_indices.append(top1_idx)
        top2_indices.append(top2_idx)
    
    h_arr = np.array(h_final_list)
    gaps = np.array(logit_gaps)
    
    # ===== 1. Per-direction spectrum analysis =====
    print(f"\n1. Per-Direction Contribution to logit_gap:")
    
    # For each text, compute c_k = h @ U_k[k] (projection coefficient)
    # and Delta_k = (W_U[top1] - W_U[top2]) @ U_k[k] (direction difference)
    c_k_all = h_arr @ U_k.T  # [n_texts, n_dirs]
    
    dir_contributions = {}
    for k in range(n_dirs):
        Delta_k = np.array([
            (W_U[top1_indices[j]] - W_U[top2_indices[j]]) @ U_k[k]
            for j in range(n_texts)
        ])
        
        # Contribution = c_k * Delta_k
        contrib_k = c_k_all[:, k] * Delta_k
        
        # Correlation with actual gap
        r_k, _ = stats.pearsonr(contrib_k, gaps)
        
        # Sign agreement
        sign_agreement = np.mean(np.sign(contrib_k) == np.sign(gaps))
        
        # |c_k| * |Delta_k| correlation
        r_abs, _ = stats.pearsonr(np.abs(c_k_all[:, k]) * np.abs(Delta_k), np.abs(gaps))
        
        dir_contributions[k] = {
            "r_contrib": float(r_k),
            "sign_agreement": float(sign_agreement),
            "r_abs_contrib": float(r_abs),
            "type": dir_types[k],
            "mean_Delta_k_cv": float(np.std(Delta_k) / (np.mean(np.abs(Delta_k)) + 1e-8)),
        }
        
        if k < 10 or dir_types[k] != 'mixed':
            print(f"   Dir {k:2d} [{dir_types[k]:7s}]: r={r_k:+.4f}, "
                  f"sign_agree={sign_agreement:.3f}, r_abs={r_abs:.4f}, "
                  f"Delta_k_CV={dir_contributions[k]['mean_Delta_k_cv']:.2f}")
    
    # ===== 2. Dual-channel model =====
    print(f"\n2. Dual-Channel Model:")
    
    # Format gap: sum of contributions from format directions
    # Content gap: sum of contributions from content directions
    format_gap = np.zeros(n_texts)
    content_gap = np.zeros(n_texts)
    mixed_gap = np.zeros(n_texts)
    
    for k in format_dirs:
        Delta_k = np.array([
            (W_U[top1_indices[j]] - W_U[top2_indices[j]]) @ U_k[k]
            for j in range(n_texts)
        ])
        format_gap += c_k_all[:, k] * Delta_k
    
    for k in content_dirs:
        Delta_k = np.array([
            (W_U[top1_indices[j]] - W_U[top2_indices[j]]) @ U_k[k]
            for j in range(n_texts)
        ])
        content_gap += c_k_all[:, k] * Delta_k
    
    for k in mixed_dirs:
        Delta_k = np.array([
            (W_U[top1_indices[j]] - W_U[top2_indices[j]]) @ U_k[k]
            for j in range(n_texts)
        ])
        mixed_gap += c_k_all[:, k] * Delta_k
    
    # Single-channel (all directions)
    all_gap = format_gap + content_gap + mixed_gap
    r_single, _ = stats.pearsonr(all_gap, gaps)
    
    # Format only
    r_format, _ = stats.pearsonr(format_gap, gaps) if len(format_dirs) > 0 else (0, 0)
    # Content only
    r_content, _ = stats.pearsonr(content_gap, gaps) if len(content_dirs) > 0 else (0, 0)
    
    # Dual-channel Ridge model
    X_dual = np.column_stack([format_gap, content_gap])
    if len(mixed_dirs) > 0:
        X_dual = np.column_stack([X_dual, mixed_gap])
    
    reg_dual = Ridge(alpha=1.0).fit(X_dual, gaps)
    r_dual = np.corrcoef(reg_dual.predict(X_dual), gaps)[0, 1]
    
    print(f"   r(all directions)    = {r_single:.4f}")
    print(f"   r(format only)       = {r_format:.4f}")
    print(f"   r(content only)      = {r_content:.4f}")
    print(f"   r(dual-channel Ridge)= {r_dual:.4f}")
    print(f"   Dual coefficients: format={reg_dual.coef_[0]:.4f}, content={reg_dual.coef_[1]:.4f}")
    
    # ===== 3. Sign-corrected model =====
    print(f"\n3. Sign-Corrected Model:")
    
    # Instead of c_k * Delta_k, use |c_k| * |Delta_k| * sign_corrected
    # where sign_corrected = sign(mean(c_k * Delta_k over texts))
    sign_corrected_gap = np.zeros(n_texts)
    for k in range(n_dirs):
        Delta_k = np.array([
            (W_U[top1_indices[j]] - W_U[top2_indices[j]]) @ U_k[k]
            for j in range(n_texts)
        ])
        # Use absolute contribution with ensemble sign
        mean_sign = np.sign(np.mean(c_k_all[:, k] * Delta_k))
        sign_corrected_gap += np.abs(c_k_all[:, k]) * np.abs(Delta_k) * mean_sign
    
    r_sign_corrected, _ = stats.pearsonr(sign_corrected_gap, gaps)
    
    # Also: |c_k| * |Delta_k| without sign (pure amplitude)
    amplitude_gap = np.zeros(n_texts)
    for k in range(n_dirs):
        Delta_k = np.array([
            (W_U[top1_indices[j]] - W_U[top2_indices[j]]) @ U_k[k]
            for j in range(n_texts)
        ])
        amplitude_gap += np.abs(c_k_all[:, k]) * np.abs(Delta_k)
    
    r_amplitude, _ = stats.pearsonr(amplitude_gap, gaps)
    
    print(f"   r(sign-corrected)    = {r_sign_corrected:.4f}")
    print(f"   r(amplitude only)    = {r_amplitude:.4f}")
    
    # ===== 4. Conditional expectation model =====
    print(f"\n4. Conditional Expectation Model:")
    
    # E[Delta_k | context] — use mean Delta_k as proxy
    # For each direction, compute E[Delta_k] across all texts
    mean_Delta_k = np.zeros(n_dirs)
    for k in range(n_dirs):
        Delta_k = np.array([
            (W_U[top1_indices[j]] - W_U[top2_indices[j]]) @ U_k[k]
            for j in range(n_texts)
        ])
        mean_Delta_k[k] = np.mean(Delta_k)
    
    # Conditional gap: c_k * E[Delta_k]
    cond_gap = c_k_all @ mean_Delta_k
    r_cond, _ = stats.pearsonr(cond_gap, gaps)
    
    # Weighted by direction importance
    # Use s_k as weights (spectral importance)
    weighted_cond_gap = np.zeros(n_texts)
    for k in range(n_dirs):
        weighted_cond_gap += s_k[k] * c_k_all[:, k] * mean_Delta_k[k]
    r_weighted_cond, _ = stats.pearsonr(weighted_cond_gap, gaps)
    
    print(f"   r(conditional E[Delta_k]) = {r_cond:.4f}")
    print(f"   r(spectral-weighted cond) = {r_weighted_cond:.4f}")
    
    # ===== 5. Gamma-mediated model =====
    print(f"\n5. Gamma-Mediated Spectrum→gap Model:")
    
    # After LN: h_LN = gamma * (h - mean) / std
    # gap = h_LN . Delta_W = sum_d gamma[d] * h_norm[d] * Delta_W[d]
    # Rewrite in spectral basis: gap = sum_k (gamma @ U_k[k]) * (h_norm @ U_k[k]) * ...
    # gamma in spectral space
    gamma_spec = gamma @ U_k.T  # [n_dirs]
    
    print(f"   Gamma spectral coefficients (top 10):")
    for k in range(min(10, n_dirs)):
        print(f"     Dir {k:2d} [{dir_types[k]:7s}]: gamma_spec={gamma_spec[k]:.4f}, "
              f"singular_val={s_k[k]:.2f}")
    
    # Gamma-weighted spectrum gap
    gamma_spec_gap = np.zeros(n_texts)
    for k in range(n_dirs):
        Delta_k = np.array([
            (W_U[top1_indices[j]] - W_U[top2_indices[j]]) @ U_k[k]
            for j in range(n_texts)
        ])
        gamma_spec_gap += gamma_spec[k] * c_k_all[:, k] * Delta_k
    
    r_gamma_spec, _ = stats.pearsonr(gamma_spec_gap, gaps)
    print(f"   r(gamma-weighted spectral gap) = {r_gamma_spec:.4f}")
    
    # ===== 6. Best combined model =====
    print(f"\n6. Best Combined Model:")
    
    # Combine dual-channel + sign-corrected + conditional
    X_combined = np.column_stack([
        format_gap, content_gap,
        sign_corrected_gap,
        amplitude_gap,
        cond_gap,
    ])
    
    # LOO-CV for combined model
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    preds_loo = np.zeros(n_texts)
    
    X_combined_std = StandardScaler().fit_transform(X_combined)
    
    for train_idx, test_idx in loo.split(X_combined_std):
        reg = Ridge(alpha=1.0).fit(X_combined_std[train_idx], gaps[train_idx])
        preds_loo[test_idx[0]] = reg.predict(X_combined_std[test_idx])[0]
    
    r_combined_loo, _ = stats.pearsonr(preds_loo, gaps)
    r_combined = np.corrcoef(Ridge(alpha=1.0).fit(X_combined_std, gaps).predict(X_combined_std), gaps)[0, 1]
    
    print(f"   r(combined, train) = {r_combined:.4f}")
    print(f"   r(combined, LOO)   = {r_combined_loo:.4f}")
    
    results = {
        "n_dirs": n_dirs,
        "format_dirs": format_dirs,
        "content_dirs": content_dirs,
        "mixed_dirs": mixed_dirs,
        "dir_types": dir_types,
        "r_single_channel": float(r_single),
        "r_format_only": float(r_format) if len(format_dirs) > 0 else 0,
        "r_content_only": float(r_content) if len(content_dirs) > 0 else 0,
        "r_dual_channel": float(r_dual),
        "r_sign_corrected": float(r_sign_corrected),
        "r_amplitude": float(r_amplitude),
        "r_conditional": float(r_cond),
        "r_weighted_conditional": float(r_weighted_cond),
        "r_gamma_spectral": float(r_gamma_spec),
        "r_combined_train": float(r_combined),
        "r_combined_loo": float(r_combined_loo),
        "dir_contributions": {str(k): v for k, v in dir_contributions.items()},
    }
    
    return results


def experiment_p681(model, tokenizer, device, model_name):
    """P681: Gamma Nonlinear Information Decoding.
    
    Key question: What does GLM4 gamma encode that Ridge cannot capture?
    
    From Phase CLII-CLIV:
    - GLM4 actual gamma r=0.819 > Ridge optimal r=0.610
    - GLM4 residual gamma is NEGATIVE (-0.375) — variance suppression only
    - Qwen3 residual gamma is POSITIVE (+0.170) — extra useful info
    - corr(actual_gamma, optimal_gamma) ≈ 0.02-0.12 — almost uncorrelated!
    
    Tests:
    1. Gamma decomposition: VS component + orthogonal component
    2. What does the orthogonal component correlate with?
    3. Nonlinear gamma model: polynomial + interaction terms
    4. Per-dimension gamma-h interaction analysis
    """
    print(f"\n{'='*60}")
    print(f"P681: Gamma Nonlinear Information Decoding ({model_name})")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    d_model = W_U.shape[1]
    final_ln = get_final_ln(model)
    
    if final_ln is not None:
        gamma = final_ln.weight.detach().cpu().float().numpy()
        beta = final_ln.bias.detach().cpu().float().numpy() if hasattr(final_ln, 'bias') and final_ln.bias is not None else np.zeros(d_model)
    else:
        print("  No final LN found, skipping.")
        return {}
    
    W_U_col_std = np.std(W_U, axis=0)
    W_U_col_mean = np.mean(W_U, axis=0)
    W_U_col_mean_abs = np.mean(np.abs(W_U), axis=0)
    
    n_texts = 100
    test_texts = ALL_TEST_TEXTS[:n_texts]
    
    # Collect per-text data
    h_before_list = []
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
        logits = outputs.logits[0, -1, :].cpu().float().numpy()
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        
        h_before_list.append(h_before)
        logit_gaps.append(logits[top1_idx] - logits[top2_idx])
        Delta_Ws.append(W_U[top1_idx] - W_U[top2_idx])
    
    h_arr = np.array(h_before_list)
    gaps = np.array(logit_gaps)
    Delta_W_arr = np.array(Delta_Ws)
    
    # Normalize h
    h_centered = h_arr - np.mean(h_arr, axis=1, keepdims=True)
    h_std = np.std(h_arr, axis=1, keepdims=True) + 1e-8
    h_normalized = h_centered / h_std
    
    # ===== 1. Gamma decomposition =====
    print(f"\n1. Gamma Decomposition:")
    
    # VS component: project gamma onto -W_U_col_std direction
    vs_direction = -W_U_col_std / (np.linalg.norm(W_U_col_std) + 1e-8)
    gamma_vs = np.dot(gamma, vs_direction) * vs_direction  # projection
    gamma_orth = gamma - gamma_vs  # orthogonal component
    
    # Quality of VS component
    gap_vs = np.sum(gamma_vs * h_normalized * Delta_W_arr, axis=1)
    gap_orth = np.sum(gamma_orth * h_normalized * Delta_W_arr, axis=1)
    gap_full = np.sum(gamma * h_normalized * Delta_W_arr, axis=1)
    
    r_vs, _ = stats.pearsonr(gap_vs, gaps)
    r_orth, _ = stats.pearsonr(gap_orth, gaps)
    r_full, _ = stats.pearsonr(gap_full, gaps)
    
    # Norm ratio
    norm_vs = np.linalg.norm(gamma_vs)
    norm_orth = np.linalg.norm(gamma_orth)
    norm_full = np.linalg.norm(gamma)
    
    print(f"   ||gamma|| = {norm_full:.4f}, ||gamma_vs|| = {norm_vs:.4f}, ||gamma_orth|| = {norm_orth:.4f}")
    print(f"   VS fraction = {norm_vs/norm_full:.4f}")
    print(f"   r(VS component)      = {r_vs:.4f}")
    print(f"   r(orthogonal comp)   = {r_orth:.4f}")
    print(f"   r(full gamma)        = {r_full:.4f}")
    
    # ===== 2. What does the orthogonal component correlate with? =====
    print(f"\n2. Orthogonal Component Correlation Analysis:")
    
    # Test correlations with various W_U statistics
    orth_candidates = {
        "W_U_col_std_sq": W_U_col_std ** 2,
        "W_U_col_mean_abs": W_U_col_mean_abs,
        "W_U_col_mean": W_U_col_mean,
        "W_U_col_skew": np.mean((W_U - W_U_col_mean) ** 3, axis=0) / (W_U_col_std ** 3 + 1e-8),
        "W_U_col_kurtosis": np.mean((W_U - W_U_col_mean) ** 4, axis=0) / (W_U_col_std ** 4 + 1e-8) - 3,
        "W_U_col_max": np.max(W_U, axis=0),
        "W_U_col_min": np.min(W_U, axis=0),
        "W_U_col_range": np.max(W_U, axis=0) - np.min(W_U, axis=0),
        "beta": beta,  # LN bias
    }
    
    # For large W_U, skip skew/kurtosis computation
    if W_U.shape[0] * W_U.shape[1] > 500_000_000:
        orth_candidates.pop("W_U_col_skew", None)
        orth_candidates.pop("W_U_col_kurtosis", None)
    
    orth_correlations = {}
    for name, candidate in orth_candidates.items():
        if candidate.shape[0] != d_model:
            continue
        r_val, p_val = stats.pearsonr(gamma_orth, candidate)
        orth_correlations[name] = {"r": float(r_val), "p": float(p_val)}
        print(f"   corr(gamma_orth, {name:20s}) = {r_val:+.4f} (p={p_val:.2e})")
    
    # ===== 3. Nonlinear gamma model =====
    print(f"\n3. Nonlinear Gamma Model:")
    
    # Model: gap ~ sum_d [a_d * gamma[d] + b_d * gamma[d]^2 + c_d * h_norm[d] * gamma[d]] * Delta_W[d]
    # Simplified: compute gamma-weighted gap with nonlinear terms
    
    # Linear term (baseline)
    gap_linear = np.sum(gamma * h_normalized * Delta_W_arr, axis=1)
    r_linear, _ = stats.pearsonr(gap_linear, gaps)
    
    # Quadratic term: gamma^2 * h_norm * Delta_W
    gap_quad = np.sum(gamma ** 2 * h_normalized * Delta_W_arr, axis=1)
    r_quad, _ = stats.pearsonr(gap_quad, gaps)
    
    # Cross-term: gamma * h_norm^2 * Delta_W (h magnitude interaction)
    gap_cross = np.sum(gamma * (h_normalized ** 2) * Delta_W_arr, axis=1)
    r_cross, _ = stats.pearsonr(gap_cross, gaps)
    
    # Combined: Ridge on [linear, quad, cross]
    X_nl = np.column_stack([gap_linear, gap_quad, gap_cross])
    reg_nl = Ridge(alpha=1.0).fit(X_nl, gaps)
    r_nl_combined = np.corrcoef(reg_nl.predict(X_nl), gaps)[0, 1]
    
    print(f"   r(linear term)       = {r_linear:.4f}")
    print(f"   r(quadratic term)    = {r_quad:.4f}")
    print(f"   r(cross term)        = {r_cross:.4f}")
    print(f"   r(nonlinear combined)= {r_nl_combined:.4f}")
    print(f"   Coefficients: linear={reg_nl.coef_[0]:.4f}, quad={reg_nl.coef_[1]:.6f}, cross={reg_nl.coef_[2]:.6f}")
    
    # ===== 4. Per-dimension gamma-h interaction =====
    print(f"\n4. Per-Dimension Gamma-h Interaction:")
    
    # For each dimension d, compute:
    # contribution_d = gamma[d] * h_norm[d] * Delta_W[d]
    # Then find which dimensions have highest correlation with gap
    
    per_dim_contrib = h_normalized * Delta_W_arr * gamma  # [n_texts, d_model]
    
    # Top dimensions by correlation with gap
    dim_correlations = np.zeros(d_model)
    for d in range(d_model):
        if np.std(per_dim_contrib[:, d]) > 1e-10:
            r_d, _ = stats.pearsonr(per_dim_contrib[:, d], gaps)
            dim_correlations[d] = r_d
        else:
            dim_correlations[d] = 0
    
    # Top-20 dimensions
    top_dims = np.argsort(np.abs(dim_correlations))[-20:][::-1]
    print(f"   Top-20 gap-predictive dimensions:")
    for rank, d in enumerate(top_dims):
        print(f"     Dim {d:5d}: r={dim_correlations[d]:+.4f}, "
              f"gamma={gamma[d]:.4f}, W_U_col_std={W_U_col_std[d]:.4f}")
    
    # ===== 5. Gamma effectiveness decomposition =====
    print(f"\n5. Gamma Effectiveness Decomposition:")
    
    # How much does each gamma property contribute to r=0.819 (GLM4) vs r=0.279 (Qwen3)?
    
    # Property 1: gamma magnitude
    r_magnitude, _ = stats.pearsonr(np.sum(np.abs(gamma) * h_normalized * Delta_W_arr, axis=1), gaps)
    
    # Property 2: gamma sign alignment
    sign_alignment = np.mean(np.sign(gamma) == np.sign(np.mean(h_normalized * Delta_W_arr, axis=0)))
    
    # Property 3: gamma-W_U_col_std alignment (variance suppression)
    r_vs_align, _ = stats.pearsonr(gamma, -W_U_col_std)
    
    # Property 4: gamma-gap_contrib alignment
    mean_gap_contrib = np.mean(h_normalized * Delta_W_arr, axis=0)
    r_gap_align, _ = stats.pearsonr(gamma, mean_gap_contrib)
    
    # Property 5: gamma frequency (low vs high frequency content)
    gamma_fft = np.abs(np.fft.fft(gamma - np.mean(gamma)))
    n_freq = len(gamma_fft) // 2
    low_freq_energy = np.sum(gamma_fft[:n_freq // 4]) / np.sum(gamma_fft[:n_freq])
    
    print(f"   r(magnitude model)       = {r_magnitude:.4f}")
    print(f"   Sign alignment           = {sign_alignment:.4f}")
    print(f"   corr(gamma, -W_U_std)    = {r_vs_align:.4f}")
    print(f"   corr(gamma, gap_contrib) = {r_gap_align:.4f}")
    print(f"   Low-freq energy fraction = {low_freq_energy:.4f}")
    
    # ===== 6. Ridge vs actual gamma comparison =====
    print(f"\n6. Ridge Optimal vs Actual Gamma:")
    
    # Compute Ridge optimal gamma
    # Target: gamma_ridge that maximizes corr(gamma * h_norm * DW, gap)
    # = Ridge regression on h_norm * DW -> gap
    X_ridge = h_normalized * Delta_W_arr  # [n_texts, d_model]
    
    # With 100 texts and 2560-4096 dims, use Ridge with high alpha
    ridge = Ridge(alpha=100.0).fit(X_ridge, gaps)
    gamma_ridge = ridge.coef_
    
    r_ridge = np.corrcoef(ridge.predict(X_ridge), gaps)[0, 1]
    
    # LOO-CV for Ridge
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    preds_loo = np.zeros(n_texts)
    for train_idx, test_idx in loo.split(X_ridge):
        reg = Ridge(alpha=100.0).fit(X_ridge[train_idx], gaps[train_idx])
        preds_loo[test_idx[0]] = reg.predict(X_ridge[test_idx])[0]
    r_ridge_loo, _ = stats.pearsonr(preds_loo, gaps)
    
    # Compare actual vs Ridge
    corr_actual_ridge, _ = stats.pearsonr(gamma, gamma_ridge)
    
    # Residual gamma = actual - projection onto Ridge direction
    ridge_dir = gamma_ridge / (np.linalg.norm(gamma_ridge) + 1e-8)
    gamma_proj_ridge = np.dot(gamma, ridge_dir) * ridge_dir
    gamma_residual = gamma - gamma_proj_ridge
    
    gap_residual = np.sum(gamma_residual * h_normalized * Delta_W_arr, axis=1)
    r_residual, _ = stats.pearsonr(gap_residual, gaps)
    
    print(f"   r(Ridge optimal, train) = {r_ridge:.4f}")
    print(f"   r(Ridge optimal, LOO)   = {r_ridge_loo:.4f}")
    print(f"   r(actual gamma)         = {r_full:.4f}")
    print(f"   corr(actual, Ridge)     = {corr_actual_ridge:.4f}")
    print(f"   r(residual gamma)       = {r_residual:.4f}")
    print(f"   Residual is {'POSITIVE' if r_residual > 0 else 'NEGATIVE'} contribution")
    
    results = {
        "norm_gamma": float(norm_full),
        "norm_vs_component": float(norm_vs),
        "norm_orth_component": float(norm_orth),
        "vs_fraction": float(norm_vs / norm_full),
        "r_vs_component": float(r_vs),
        "r_orth_component": float(r_orth),
        "r_full_gamma": float(r_full),
        "r_linear": float(r_linear),
        "r_quadratic": float(r_quad),
        "r_cross": float(r_cross),
        "r_nonlinear_combined": float(r_nl_combined),
        "r_magnitude_model": float(r_magnitude),
        "sign_alignment": float(sign_alignment),
        "corr_gamma_vs": float(r_vs_align),
        "corr_gamma_gap_contrib": float(r_gap_align),
        "low_freq_fraction": float(low_freq_energy),
        "r_ridge_train": float(r_ridge),
        "r_ridge_loo": float(r_ridge_loo),
        "corr_actual_ridge": float(corr_actual_ridge),
        "r_residual_gamma": float(r_residual),
        "orth_correlations": orth_correlations,
    }
    
    return results


def experiment_p682(model, tokenizer, device, model_name):
    """P682: Qwen3 Gamma High-Frequency Source Analysis.
    
    Key question: Why is Qwen3 gamma high-frequency (84%) while GLM4 is low-frequency (48%)?
    
    From Phase CLIV:
    - Qwen3 gamma HF=84%, GLM4 gamma LF=48%
    - GLM4 gamma closer to W_U_col_std (low-freq)
    - Qwen3 gamma far from W_U_col_std
    - Qwen3 actual gamma (r=0.278) << Ridge optimal (r=0.767)
    
    Hypotheses:
    H1: Architecture — RMSNorm vs LayerNorm produces different gamma
    H2: Training — SFT vs RLHF produces different gamma frequency
    H3: W_U structure — Qwen3 W_U is more uniform → gamma has less structure
    H4: MLP interaction — Qwen3 MLP noise forces gamma to be high-freq
    
    Tests:
    1. Gamma frequency spectrum detailed analysis
    2. W_U frequency spectrum comparison
    3. Layer-wise gamma evolution (is gamma learned layer-by-layer?)
    4. Gamma smoothing experiment (low-pass filter gamma)
    """
    print(f"\n{'='*60}")
    print(f"P682: Qwen3 Gamma High-Frequency Source ({model_name})")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    d_model = W_U.shape[1]
    final_ln = get_final_ln(model)
    
    if final_ln is not None:
        gamma = final_ln.weight.detach().cpu().float().numpy()
    else:
        print("  No final LN found, skipping.")
        return {}
    
    W_U_col_std = np.std(W_U, axis=0)
    
    n_texts = 100
    test_texts = ALL_TEST_TEXTS[:n_texts]
    
    # Collect per-text data
    h_before_list = []
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
        logits = outputs.logits[0, -1, :].cpu().float().numpy()
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        
        h_before_list.append(h_before)
        logit_gaps.append(logits[top1_idx] - logits[top2_idx])
        Delta_Ws.append(W_U[top1_idx] - W_U[top2_idx])
    
    h_arr = np.array(h_before_list)
    gaps = np.array(logit_gaps)
    Delta_W_arr = np.array(Delta_Ws)
    
    h_centered = h_arr - np.mean(h_arr, axis=1, keepdims=True)
    h_std = np.std(h_arr, axis=1, keepdims=True) + 1e-8
    h_normalized = h_centered / h_std
    
    # ===== 1. Gamma frequency spectrum detailed analysis =====
    print(f"\n1. Gamma Frequency Spectrum:")
    
    gamma_centered = gamma - np.mean(gamma)
    gamma_fft = np.fft.fft(gamma_centered)
    gamma_power = np.abs(gamma_fft[:d_model // 2]) ** 2
    total_power = np.sum(gamma_power)
    
    # Frequency bands
    n_bands = 4
    band_size = len(gamma_power) // n_bands
    band_powers = []
    for b in range(n_bands):
        band_power = np.sum(gamma_power[b * band_size:(b + 1) * band_size])
        band_powers.append(band_power / total_power)
    
    print(f"   Frequency band powers (normalized):")
    for b in range(n_bands):
        freq_range = f"{b * band_size}-{(b + 1) * band_size}"
        print(f"     Band {b} ({freq_range}): {band_powers[b]:.4f}")
    
    # Dominant frequency
    dominant_freq = np.argmax(gamma_power)
    print(f"   Dominant frequency: {dominant_freq}")
    
    # Also for W_U_col_std
    std_centered = W_U_col_std - np.mean(W_U_col_std)
    std_fft = np.fft.fft(std_centered)
    std_power = np.abs(std_fft[:d_model // 2]) ** 2
    std_total = np.sum(std_power)
    std_band_powers = []
    for b in range(n_bands):
        band_power = np.sum(std_power[b * band_size:(b + 1) * band_size])
        std_band_powers.append(band_power / std_total)
    
    print(f"   W_U_col_std frequency band powers:")
    for b in range(n_bands):
        freq_range = f"{b * band_size}-{(b + 1) * band_size}"
        print(f"     Band {b} ({freq_range}): {std_band_powers[b]:.4f}")
    
    # Cross-spectrum: how well does gamma spectrum match W_U_col_std spectrum?
    gamma_spec_normalized = gamma_power / total_power
    std_spec_normalized = std_power / std_total
    spec_corr, _ = stats.pearsonr(gamma_spec_normalized, std_spec_normalized)
    print(f"   Spectrum correlation (gamma vs W_U_col_std): {spec_corr:.4f}")
    
    # ===== 2. Gamma smoothing experiment =====
    print(f"\n2. Gamma Smoothing Experiment:")
    
    # Apply low-pass filter to gamma and measure gap quality
    gamma_fft_full = np.fft.fft(gamma_centered)
    
    smoothing_results = {}
    for cutoff_pct in [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]:
        cutoff = int(d_model * cutoff_pct / 2)
        
        # Low-pass filter
        gamma_filtered_fft = np.zeros_like(gamma_fft_full)
        gamma_filtered_fft[:cutoff] = gamma_fft_full[:cutoff]
        gamma_filtered_fft[-cutoff:] = gamma_fft_full[-cutoff:]  # symmetric
        gamma_smooth = np.real(np.fft.ifft(gamma_filtered_fft)) + np.mean(gamma)
        
        # Compute gap quality
        gap_smooth = np.sum(gamma_smooth * h_normalized * Delta_W_arr, axis=1)
        r_smooth, _ = stats.pearsonr(gap_smooth, gaps)
        
        smoothing_results[str(cutoff_pct)] = {
            "cutoff_freq": cutoff,
            "r_smooth": float(r_smooth),
            "n_freq_components": 2 * cutoff,
            "energy_retained": float(np.sum(np.abs(gamma_filtered_fft[:cutoff]) ** 2) / total_power) if total_power > 0 else 0,
        }
        
        print(f"   Cutoff {cutoff_pct:.0%} ({2*cutoff} components): r={r_smooth:.4f}")
    
    # ===== 3. Layer-wise gamma analysis =====
    print(f"\n3. Layer-wise Gamma Analysis:")
    
    # Collect gamma from all layer norms
    layers = get_layers(model)
    layer_gammas = {}
    
    for l_idx, layer in enumerate(layers):
        # Input LN gamma
        if hasattr(layer, 'input_layernorm'):
            ln = layer.input_layernorm
            if hasattr(ln, 'weight'):
                g = ln.weight.detach().cpu().float().numpy()
                layer_gammas[f"L{l_idx}_input"] = g
        
        # Post-attn LN gamma
        if hasattr(layer, 'post_attention_layernorm'):
            ln = layer.post_attention_layernorm
            if hasattr(ln, 'weight'):
                g = ln.weight.detach().cpu().float().numpy()
                layer_gammas[f"L{l_idx}_post_attn"] = g
    
    # Final LN
    layer_gammas["final"] = gamma
    
    # Analyze frequency evolution
    print(f"   Layer-wise gamma frequency (low-freq energy fraction):")
    layer_freq_data = {}
    for name, g in layer_gammas.items():
        g_centered = g - np.mean(g)
        g_fft = np.fft.fft(g_centered)
        g_power = np.abs(g_fft[:len(g) // 2]) ** 2
        g_total = np.sum(g_power) + 1e-10
        
        n_half = len(g) // 2
        lf_energy = np.sum(g_power[:n_half // 4]) / g_total
        
        # Correlation with final gamma
        if g.shape[0] == d_model:
            r_with_final, _ = stats.pearsonr(g, gamma)
            r_with_std, _ = stats.pearsonr(g, -W_U_col_std)
        else:
            r_with_final = 0
            r_with_std = 0
        
        layer_freq_data[name] = {
            "lf_fraction": float(lf_energy),
            "r_with_final": float(r_with_final),
            "r_with_neg_std": float(r_with_std),
        }
        
        if "post_attn" in name or name == "final":
            print(f"     {name:20s}: LF={lf_energy:.4f}, r(final)={r_with_final:.4f}, r(-std)={r_with_std:.4f}")
    
    # ===== 4. Optimal gamma construction =====
    print(f"\n4. Optimal Gamma Construction:")
    
    # What would the ideal gamma look like?
    # Ideal: gamma_ideal = Ridge(h_norm * DW, gap)
    X_ridge = h_normalized * Delta_W_arr
    ridge = Ridge(alpha=100.0).fit(X_ridge, gaps)
    gamma_ideal = ridge.coef_
    
    # Frequency analysis of ideal gamma
    gi_centered = gamma_ideal - np.mean(gamma_ideal)
    gi_fft = np.fft.fft(gi_centered)
    gi_power = np.abs(gi_fft[:d_model // 2]) ** 2
    gi_total = np.sum(gi_power) + 1e-10
    gi_lf = np.sum(gi_power[:d_model // 8]) / gi_total
    
    print(f"   Ideal gamma LF fraction: {gi_lf:.4f}")
    print(f"   Actual gamma LF fraction: {band_powers[0]:.4f}")
    print(f"   Ideal gamma is {'LOW' if gi_lf > 0.5 else 'HIGH'} frequency")
    
    # Correlation: actual vs ideal
    r_actual_ideal, _ = stats.pearsonr(gamma, gamma_ideal)
    print(f"   corr(actual, ideal) = {r_actual_ideal:.4f}")
    
    # ===== 5. MLP noise → gamma frequency hypothesis =====
    print(f"\n5. MLP Noise → Gamma Frequency Hypothesis:")
    
    # If MLP injects high-frequency noise (Qwen3), gamma must also be high-freq to compensate?
    # Test: compute h frequency spectrum at different stages
    
    # h after post-attn LN vs h after MLP at last layer
    last_layer = layers[-1]
    
    cache = {}
    def make_hook(name):
        def hook_fn(module, input, output):
            cache[name] = output[0].detach().cpu()
        return hook_fn
    
    hooks = []
    if hasattr(last_layer, 'post_attention_layernorm'):
        hooks.append(last_layer.post_attention_layernorm.register_forward_hook(make_hook('post_ln')))
    if hasattr(last_layer, 'mlp'):
        hooks.append(last_layer.mlp.register_forward_hook(make_hook('mlp')))
    
    h_post_ln_ffts = []
    h_mlp_ffts = []
    
    for i, text in enumerate(test_texts[:20]):
        cache.clear()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"].to(device),
                          attention_mask=inputs["attention_mask"].to(device),
                          output_hidden_states=True)
        
        if 'post_ln' in cache:
            h_post_ln = extract_tensor(cache['post_ln'])
            h_pl_centered = h_post_ln - np.mean(h_post_ln)
            h_pl_fft = np.abs(np.fft.fft(h_pl_centered))[:d_model // 2] ** 2
            h_post_ln_ffts.append(h_pl_fft)
        
        if 'mlp' in cache:
            h_mlp_out = extract_tensor(cache['mlp'])
            h_m_centered = h_mlp_out - np.mean(h_mlp_out)
            h_m_fft = np.abs(np.fft.fft(h_m_centered))[:d_model // 2] ** 2
            h_mlp_ffts.append(h_m_fft)
    
    for h in hooks:
        h.remove()
    
    if h_post_ln_ffts and h_mlp_ffts:
        mean_post_ln_fft = np.mean(h_post_ln_ffts, axis=0)
        mean_mlp_fft = np.mean(h_mlp_ffts, axis=0)
        
        pl_lf = np.sum(mean_post_ln_fft[:d_model // 8]) / (np.sum(mean_post_ln_fft) + 1e-10)
        mlp_lf = np.sum(mean_mlp_fft[:d_model // 8]) / (np.sum(mean_mlp_fft) + 1e-10)
        
        print(f"   h after post-attn LN LF fraction: {pl_lf:.4f}")
        print(f"   h after MLP LF fraction:          {mlp_lf:.4f}")
        print(f"   MLP shifts frequency: {'LOW→HIGH' if mlp_lf < pl_lf else 'HIGH→LOW'}")
    else:
        pl_lf = 0
        mlp_lf = 0
        print(f"   Could not collect hook data")
    
    results = {
        "gamma_band_powers": {f"band_{b}": float(band_powers[b]) for b in range(n_bands)},
        "W_U_std_band_powers": {f"band_{b}": float(std_band_powers[b]) for b in range(n_bands)},
        "spectrum_corr_gamma_std": float(spec_corr),
        "smoothing_results": smoothing_results,
        "gamma_ideal_lf_fraction": float(gi_lf),
        "gamma_actual_lf_fraction": float(band_powers[0]),
        "corr_actual_ideal": float(r_actual_ideal),
        "h_post_ln_lf": float(pl_lf),
        "h_mlp_lf": float(mlp_lf),
        "layer_freq_data": layer_freq_data,
    }
    
    return results


def experiment_p683(model, tokenizer, device, model_name):
    """P683: Cross-Layer Gap Information Propagation Tracking.
    
    Key question: How exactly does gap information flow through the LN-MLP-LN cycle?
    
    From Phase CXLVIII-CXLIX:
    - LN reveals gap information (doesn't create it)
    - MLP injects noise that destroys gap SNR
    - Gap emergence is sudden (Δr > 1.1)
    - But the precise mechanism is unclear
    
    Tests:
    1. Per-layer gap information tracking (SNR at each stage)
    2. LN-MLP-LN cycle at multiple layers (not just last)
    3. Information bottleneck: where exactly is gap lost/recovered?
    4. Per-component contribution to gap SNR
    """
    print(f"\n{'='*60}")
    print(f"P683: Cross-Layer Gap Information Tracking ({model_name})")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    d_model = W_U.shape[1]
    n_layers = len(get_layers(model))
    
    n_texts = 80
    test_texts = ALL_TEST_TEXTS[:n_texts]
    
    # We need to track gap information at multiple stages per layer
    # For each layer: input_ln → attn → post_attn_ln → mlp → (residual)
    
    # Strategy: hook into key points and collect h states
    layers = get_layers(model)
    
    # Hook setup for all layers
    cache = {}
    def make_hook(layer_idx, name):
        def hook_fn(module, input, output):
            cache[(layer_idx, name)] = output[0].detach().cpu()
        return hook_fn
    
    hooks = []
    for l_idx, layer in enumerate(layers):
        # Post-attention layer norm
        if hasattr(layer, 'post_attention_layernorm'):
            hooks.append(layer.post_attention_layernorm.register_forward_hook(
                make_hook(l_idx, 'post_attn_ln')))
        # MLP output
        if hasattr(layer, 'mlp'):
            hooks.append(layer.mlp.register_forward_hook(
                make_hook(l_idx, 'mlp_out')))
    
    # Collect data
    logit_gaps = []
    Delta_Ws = []
    
    # Hidden states at each stage
    h_stages = {}  # (layer, stage) -> [n_texts, d_model]
    
    for i, text in enumerate(test_texts):
        if (i+1) % 20 == 0:
            print(f"  Processing text {i+1}/{n_texts}...")
        
        cache.clear()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"].to(device),
                          attention_mask=inputs["attention_mask"].to(device),
                          output_hidden_states=True)
        
        logits = outputs.logits[0, -1, :].cpu().float().numpy()
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        
        logit_gaps.append(logits[top1_idx] - logits[top2_idx])
        Delta_Ws.append(W_U[top1_idx] - W_U[top2_idx])
        
        # Extract hidden states from model output
        for l_idx in range(n_layers):
            # From output hidden states
            h_layer = outputs.hidden_states[l_idx][0, -1, :].cpu().float().numpy()
            stage_key = (l_idx, 'output')
            if stage_key not in h_stages:
                h_stages[stage_key] = []
            h_stages[stage_key].append(h_layer)
        
        # From hooks
        for l_idx in range(n_layers):
            for stage_name in ['post_attn_ln', 'mlp_out']:
                if (l_idx, stage_name) in cache:
                    t = cache[(l_idx, stage_name)]
                    h = extract_tensor(t)
                    stage_key = (l_idx, stage_name)
                    if stage_key not in h_stages:
                        h_stages[stage_key] = []
                    h_stages[stage_key].append(h)
    
    for h in hooks:
        h.remove()
    
    gaps = np.array(logit_gaps)
    Delta_W_arr = np.array(Delta_Ws)
    
    # ===== 1. Per-layer gap information tracking =====
    print(f"\n1. Per-Layer Gap Information (Ridge LOO):")
    
    layer_gap_info = {}
    
    for l_idx in range(n_layers):
        # Get h at layer output
        if (l_idx, 'output') not in h_stages:
            continue
        
        h_layer = np.array(h_stages[(l_idx, 'output')])
        if h_layer.shape[0] != n_texts:
            continue
        
        # Simple dot product gap
        h_centered = h_layer - np.mean(h_layer, axis=1, keepdims=True)
        h_std = np.std(h_layer, axis=1, keepdims=True) + 1e-8
        h_norm = h_centered / h_std
        
        gap_dot = np.sum(h_norm * Delta_W_arr, axis=1)
        r_dot, _ = stats.pearsonr(gap_dot, gaps)
        
        # Ridge prediction
        from sklearn.model_selection import LeaveOneOut
        loo = LeaveOneOut()
        X_ridge = h_norm * Delta_W_arr
        preds_loo = np.zeros(n_texts)
        for train_idx, test_idx in loo.split(X_ridge):
            reg = Ridge(alpha=100.0).fit(X_ridge[train_idx], gaps[train_idx])
            preds_loo[test_idx[0]] = reg.predict(X_ridge[test_idx])[0]
        r_ridge_loo, _ = stats.pearsonr(preds_loo, gaps)
        
        layer_gap_info[l_idx] = {
            "r_dot": float(r_dot),
            "r_ridge_loo": float(r_ridge_loo),
        }
        
        if (l_idx + 1) % 5 == 0 or l_idx == n_layers - 1:
            print(f"   L{l_idx:2d}: r(dot)={r_dot:+.4f}, r(Ridge LOO)={r_ridge_loo:.4f}")
    
    # ===== 2. Last layer detailed stage tracking =====
    print(f"\n2. Last Layer Stage-by-Stage Gap Information:")
    
    last_l = n_layers - 1
    stage_names = ['output', 'post_attn_ln', 'mlp_out']
    
    stage_gap_info = {}
    for stage_name in stage_names:
        if (last_l, stage_name) not in h_stages:
            continue
        
        h_stage = np.array(h_stages[(last_l, stage_name)])
        if h_stage.shape[0] != n_texts:
            continue
        
        h_centered = h_stage - np.mean(h_stage, axis=1, keepdims=True)
        h_std = np.std(h_stage, axis=1, keepdims=True) + 1e-8
        h_norm = h_centered / h_std
        
        gap_dot = np.sum(h_norm * Delta_W_arr, axis=1)
        r_dot, _ = stats.pearsonr(gap_dot, gaps)
        
        # Ridge
        X_ridge = h_norm * Delta_W_arr
        loo = LeaveOneOut()
        preds_loo = np.zeros(n_texts)
        for train_idx, test_idx in loo.split(X_ridge):
            reg = Ridge(alpha=100.0).fit(X_ridge[train_idx], gaps[train_idx])
            preds_loo[test_idx[0]] = reg.predict(X_ridge[test_idx])[0]
        r_ridge_loo, _ = stats.pearsonr(preds_loo, gaps)
        
        # SNR: signal_mean / noise_std
        signal = np.mean(gap_dot)
        noise = np.std(gap_dot - gaps) if np.std(gap_dot) > 1e-10 else 0
        snr = abs(signal) / (noise + 1e-10)
        
        stage_gap_info[stage_name] = {
            "r_dot": float(r_dot),
            "r_ridge_loo": float(r_ridge_loo),
            "snr": float(snr),
        }
        
        print(f"   {stage_name:20s}: r(dot)={r_dot:+.4f}, r(Ridge LOO)={r_ridge_loo:.4f}, SNR={snr:.4f}")
    
    # ===== 3. SNR evolution across layers =====
    print(f"\n3. SNR Evolution Across Layers:")
    
    snr_evolution = {}
    for l_idx in range(n_layers):
        if (l_idx, 'output') not in h_stages:
            continue
        
        h_layer = np.array(h_stages[(l_idx, 'output')])
        if h_layer.shape[0] != n_texts:
            continue
        
        h_centered = h_layer - np.mean(h_layer, axis=1, keepdims=True)
        h_std = np.std(h_layer, axis=1, keepdims=True) + 1e-8
        h_norm = h_centered / h_std
        
        gap_dot = np.sum(h_norm * Delta_W_arr, axis=1)
        
        signal = np.mean(gap_dot)
        noise = np.std(gap_dot - gaps) if np.std(gap_dot) > 1e-10 else 0
        snr = abs(signal) / (noise + 1e-10)
        
        snr_evolution[l_idx] = float(snr)
        
        if (l_idx + 1) % 5 == 0 or l_idx == n_layers - 1:
            print(f"   L{l_idx:2d}: SNR={snr:.4f}")
    
    # ===== 4. Critical transition layer =====
    print(f"\n4. Critical Transition Layer:")
    
    # Find the layer where gap information first appears (largest r jump)
    r_values = []
    for l_idx in range(n_layers):
        if l_idx in layer_gap_info:
            r_values.append((l_idx, layer_gap_info[l_idx]["r_ridge_loo"]))
    
    if len(r_values) > 1:
        max_jump = 0
        critical_layer = 0
        for i in range(1, len(r_values)):
            jump = r_values[i][1] - r_values[i-1][1]
            if abs(jump) > abs(max_jump):
                max_jump = jump
                critical_layer = r_values[i][0]
        
        print(f"   Largest r jump: {max_jump:+.4f} at layer {critical_layer}")
        print(f"   Before: r={layer_gap_info.get(critical_layer-1, {}).get('r_ridge_loo', 'N/A')}")
        print(f"   After:  r={layer_gap_info.get(critical_layer, {}).get('r_ridge_loo', 'N/A')}")
    
    # ===== 5. Per-layer MLP impact on gap information =====
    print(f"\n5. Per-Layer MLP Impact:")
    
    mlp_impact = {}
    for l_idx in range(n_layers):
        if (l_idx, 'post_attn_ln') not in h_stages or (l_idx, 'mlp_out') not in h_stages:
            continue
        
        h_post_ln = np.array(h_stages[(l_idx, 'post_attn_ln')])
        h_mlp = np.array(h_stages[(l_idx, 'mlp_out')])
        
        if h_post_ln.shape[0] != n_texts or h_mlp.shape[0] != n_texts:
            continue
        
        # Gap info before MLP (at post_attn_ln)
        h_c1 = h_post_ln - np.mean(h_post_ln, axis=1, keepdims=True)
        h_s1 = np.std(h_post_ln, axis=1, keepdims=True) + 1e-8
        h_n1 = h_c1 / h_s1
        gap_before = np.sum(h_n1 * Delta_W_arr, axis=1)
        r_before, _ = stats.pearsonr(gap_before, gaps) if np.std(gap_before) > 1e-10 else (0, 0)
        
        # Gap info after MLP (post_ln + mlp via residual)
        h_after = h_post_ln + h_mlp
        h_c2 = h_after - np.mean(h_after, axis=1, keepdims=True)
        h_s2 = np.std(h_after, axis=1, keepdims=True) + 1e-8
        h_n2 = h_c2 / h_s2
        gap_after = np.sum(h_n2 * Delta_W_arr, axis=1)
        r_after, _ = stats.pearsonr(gap_after, gaps) if np.std(gap_after) > 1e-10 else (0, 0)
        
        delta_r = r_after - r_before
        
        # MLP ratio
        norm_post_ln = np.mean(np.linalg.norm(h_post_ln, axis=1))
        norm_mlp = np.mean(np.linalg.norm(h_mlp, axis=1))
        mlp_ratio = norm_mlp / (norm_post_ln + 1e-8)
        
        mlp_impact[l_idx] = {
            "r_before_mlp": float(r_before),
            "r_after_mlp": float(r_after),
            "delta_r": float(delta_r),
            "mlp_ratio": float(mlp_ratio),
        }
        
        if (l_idx + 1) % 5 == 0 or l_idx == n_layers - 1:
            print(f"   L{l_idx:2d}: r_before={r_before:+.4f}, r_after={r_after:+.4f}, "
                  f"delta_r={delta_r:+.4f}, mlp_ratio={mlp_ratio:.3f}")
    
    # Correlation: mlp_ratio vs delta_r
    if len(mlp_impact) > 3:
        ratios = [v["mlp_ratio"] for v in mlp_impact.values()]
        delta_rs = [v["delta_r"] for v in mlp_impact.values()]
        r_ratio_dr, _ = stats.pearsonr(ratios, delta_rs)
        print(f"\n   corr(mlp_ratio, delta_r) = {r_ratio_dr:.4f}")
    else:
        r_ratio_dr = 0
    
    results = {
        "layer_gap_info": {str(k): v for k, v in layer_gap_info.items()},
        "stage_gap_info": stage_gap_info,
        "snr_evolution": {str(k): v for k, v in snr_evolution.items()},
        "mlp_impact": {str(k): v for k, v in mlp_impact.items()},
        "corr_mlp_ratio_delta_r": float(r_ratio_dr),
    }
    
    return results


def run_all_experiments(model_name):
    """Run all experiments for a given model."""
    print(f"\n{'#'*70}")
    print(f"# Phase CLV: Gap Causal Chain & Gamma Nonlinear Decoding")
    print(f"# Model: {model_name}")
    print(f"{'#'*70}")
    
    model, tokenizer, device = load_model(model_name)
    
    try:
        results = {}
        
        # P680: Dual-Channel Spectrum→logit_gap
        r680 = experiment_p680(model, tokenizer, device, model_name)
        results["p680"] = r680
        
        # P681: Gamma Nonlinear Information Decoding
        r681 = experiment_p681(model, tokenizer, device, model_name)
        results["p681"] = r681
        
        # P682: Qwen3 Gamma High-Frequency Source
        r682 = experiment_p682(model, tokenizer, device, model_name)
        results["p682"] = r682
        
        # P683: Cross-Layer Gap Information Tracking
        r683 = experiment_p683(model, tokenizer, device, model_name)
        results["p683"] = r683
        
        # Save results
        output_dir = "results/phase_clv"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{model_name}_results.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n  Results saved to {output_path}")
        
        return results
        
    finally:
        release_model(model)


def main():
    parser = argparse.ArgumentParser(description="Phase CLV: Gap Causal Chain & Gamma Nonlinear Decoding")
    parser.add_argument("--model", type=str, default="all",
                       choices=["qwen3", "glm4", "deepseek7b", "all"],
                       help="Model to test")
    args = parser.parse_args()
    
    if args.model == "all":
        # Run models sequentially to avoid GPU memory issues
        for model_name in ["qwen3", "glm4", "deepseek7b"]:
            print(f"\n\n{'='*70}")
            print(f"Starting experiments for {model_name}...")
            print(f"{'='*70}")
            run_all_experiments(model_name)
            
            # Clear GPU cache between models
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"\n  GPU cache cleared after {model_name}")
    else:
        run_all_experiments(args.model)
    
    print(f"\n\n{'#'*70}")
    print(f"# Phase CLV Complete!")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
