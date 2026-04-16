#!/usr/bin/env python3
"""
Phase CLIV: Language Interaction Model & Causal Theory (P676-P679)
==================================================================

Based on Phase CLIII findings:
1. Product model failed (unified_score ~ 0 for all models)
2. GLM4 gamma = pure variance suppressor (residual = -0.375)
3. Qwen3 gamma = variance suppression + extra info (residual = +0.169)
4. DS7B MLP ratio = 1.28~20.68! (every layer injects massive noise)
5. Qwen3 MLP increases offdiag by +0.88; GLM4 only +0.10
6. GLM4 whitening h destroys gap info (-93%); Qwen3 whitening gamma helps (+13%)

This phase addresses:
P676: Three-Factor Interaction Model — gap_quality = f(VS, GS, LE, VS*GS, VS*LE, GS*LE)
P677: MLP Causal Mechanism — why does MLP ratio > 1 increase offdiag?
P678: Gamma Optimization Target — why GLM4 gamma only does VS, Qwen3 gamma does VS+extra?
P679: Unified Language Equation v2 — nonlinear model with interactions
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
from sklearn.linear_model import Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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


def extract_tensor(t):
    """Extract last token from tensor with flexible dim handling."""
    if t.dim() == 3:
        return t[0, -1, :].float().numpy()
    elif t.dim() == 2:
        return t[-1, :].float().numpy()
    else:
        return t.float().numpy().flatten()


def experiment_p676(model, tokenizer, device, model_name):
    """P676: Three-Factor Interaction Model.
    
    Key question: Can an interaction model predict gap quality better than product model?
    
    From P675: product model failed (unified_score ~ 0)
    
    Model: gap_quality ~ a*VS + b*GS + c*LE + d*VS*GS + e*VS*LE + f*GS*LE + g*VS*GS*LE
    
    But we only have 3 data points (3 models). We need per-text variation.
    
    Strategy: For each text, compute per-text VS, GS, LE factors and fit interaction model.
    """
    print(f"\n{'='*60}")
    print(f"P676: Three-Factor Interaction Model ({model_name})")
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
    W_U_col_norms = np.linalg.norm(W_U, axis=0)
    
    n_texts = 100
    test_texts = ALL_TEST_TEXTS[:n_texts]
    
    # Collect per-text data
    h_before_list = []
    logit_gaps = []
    Delta_Ws = []
    
    # Also collect MLP ratio data at last layer
    layers = get_layers(model)
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
    
    norm_post_ln_list = []
    norm_mlp_list = []
    
    for i, text in enumerate(test_texts):
        if (i+1) % 20 == 0:
            print(f"  Processing text {i+1}/{n_texts}...")
        
        cache.clear()
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
        
        if 'post_ln' in cache and 'mlp' in cache:
            norm_post_ln_list.append(np.linalg.norm(extract_tensor(cache['post_ln'])))
            norm_mlp_list.append(np.linalg.norm(extract_tensor(cache['mlp'])))
    
    for h in hooks:
        h.remove()
    
    h_before_arr = np.array(h_before_list)
    gaps = np.array(logit_gaps)
    Delta_W_arr = np.array(Delta_Ws)
    
    # ===== Compute per-text factors =====
    print(f"\n1. Per-Text Factor Computation:")
    
    # Factor 1: Variance Suppression (per-text)
    h_centered = h_before_arr - np.mean(h_before_arr, axis=1, keepdims=True)
    h_std = np.std(h_before_arr, axis=1, keepdims=True) + 1e-8
    h_normalized = h_centered / h_std
    
    # Per-text: corr between gamma*normalized_h*DW and gap
    per_text_vs_signal = np.sum(gamma * h_normalized * Delta_W_arr, axis=1)
    r_vs, _ = stats.pearsonr(per_text_vs_signal, gaps)
    
    # Per-text variance suppression score: how much gamma suppresses W_U high-var dims
    # VS_text = corr(gamma * h_i, -W_U_col_std_i * h_i) -- per-text alignment
    vs_scores = []
    for j in range(n_texts):
        h_n = h_normalized[j]
        DW_j = Delta_W_arr[j]
        # How much of the per-text gap signal comes from variance suppression direction
        gap_suppressed = np.sum(-W_U_col_std * h_n * DW_j)
        gap_direct = np.sum(gamma * h_n * DW_j)
        vs_scores.append(gap_direct)
    vs_scores = np.array(vs_scores)
    
    # Factor 2: Gap Survival (per-text)
    if norm_post_ln_list and norm_mlp_list:
        mlp_ratios = np.array(norm_mlp_list) / (np.array(norm_post_ln_list) + 1e-8)
        gs_scores = 1.0 / (1.0 + np.exp(mlp_ratios - 1.0))  # sigmoid
    else:
        mlp_ratios = np.ones(n_texts)
        gs_scores = np.ones(n_texts) * 0.5
    
    # Factor 3: LN Efficiency (per-text, based on h covariance structure)
    # Use per-text h norm and dimensionality as proxy
    h_norms = np.linalg.norm(h_before_arr, axis=1)
    h_pr_proxy = np.std(h_before_arr, axis=1) / (np.mean(h_before_arr, axis=1) + 1e-8)  # CV as PR proxy
    le_scores = 1.0 / (1.0 + h_pr_proxy)  # higher CV = lower efficiency
    
    print(f"   VS signal range: [{vs_scores.min():.2f}, {vs_scores.max():.2f}]")
    print(f"   GS scores range: [{gs_scores.min():.4f}, {gs_scores.max():.4f}]")
    print(f"   LE scores range: [{le_scores.min():.4f}, {le_scores.max():.4f}]")
    print(f"   MLP ratios range: [{mlp_ratios.min():.2f}, {mlp_ratios.max():.2f}]")
    
    # ===== 2. Interaction model =====
    print(f"\n2. Interaction Model:")
    
    # Standardize factors
    scaler = StandardScaler()
    X_base = np.column_stack([vs_scores, gs_scores, le_scores])
    X_base_std = scaler.fit_transform(X_base)
    
    # Add interaction terms
    VS = X_base_std[:, 0]
    GS = X_base_std[:, 1]
    LE = X_base_std[:, 2]
    
    X_interact = np.column_stack([
        VS, GS, LE,                    # main effects
        VS * GS, VS * LE, GS * LE,     # two-way interactions
        VS * GS * LE,                  # three-way interaction
    ])
    
    # Fit Ridge regression
    reg_interact = Ridge(alpha=1.0).fit(X_interact, gaps)
    r_interact = np.corrcoef(reg_interact.predict(X_interact), gaps)[0, 1]
    
    # Fit main-effects-only model for comparison
    reg_main = Ridge(alpha=1.0).fit(X_base_std, gaps)
    r_main = np.corrcoef(reg_main.predict(X_base_std), gaps)[0, 1]
    
    # Fit product model
    product_pred = vs_scores * gs_scores * le_scores
    r_product, _ = stats.pearsonr(product_pred, gaps)
    
    # Fit VS-only model
    reg_vs = Ridge(alpha=1.0).fit(VS.reshape(-1, 1), gaps)
    r_vs_only = np.corrcoef(reg_vs.predict(VS.reshape(-1, 1)), gaps)[0, 1]
    
    print(f"   r(VS only)       = {r_vs_only:.4f}")
    print(f"   r(product model) = {r_product:.4f}")
    print(f"   r(main effects)  = {r_main:.4f}")
    print(f"   r(interaction)   = {r_interact:.4f}")
    
    # ===== 3. Coefficient analysis =====
    print(f"\n3. Interaction Coefficients:")
    coef_names = ["VS", "GS", "LE", "VS*GS", "VS*LE", "GS*LE", "VS*GS*LE"]
    for name, coef in zip(coef_names, reg_interact.coef_):
        print(f"   {name:10s} = {coef:.6f}")
    
    # ===== 4. Lasso for sparsity =====
    print(f"\n4. Lasso (sparsity analysis):")
    reg_lasso = Lasso(alpha=0.01).fit(X_interact, gaps)
    r_lasso = np.corrcoef(reg_lasso.predict(X_interact), gaps)[0, 1]
    for name, coef in zip(coef_names, reg_lasso.coef_):
        print(f"   {name:10s} = {coef:.6f}")
    print(f"   r(Lasso) = {r_lasso:.4f}")
    
    # ===== 5. Nonlinear transformation =====
    print(f"\n5. Nonlinear Factor Transformations:")
    
    # Try: gap ~ a * VS^alpha + b * GS^beta + c * LE^gamma
    # Use log transformation for power law
    vs_pos = vs_scores - vs_scores.min() + 1e-8
    gs_pos = np.maximum(gs_scores, 1e-8)
    le_pos = np.maximum(le_scores, 1e-8)
    
    # Log-transformed
    X_log = np.column_stack([
        np.log(vs_pos), np.log(gs_pos), np.log(le_pos),
        np.log(vs_pos) * np.log(gs_pos),
    ])
    # Replace any NaN/inf
    X_log = np.nan_to_num(X_log, nan=0.0, posinf=0.0, neginf=0.0)
    X_log_std = StandardScaler().fit_transform(X_log)
    # Check for any remaining NaN
    if np.any(np.isnan(X_log_std)) or np.any(np.isinf(X_log_std)):
        r_log = 0.0
    else:
        reg_log = Ridge(alpha=1.0).fit(X_log_std, gaps)
        r_log = np.corrcoef(reg_log.predict(X_log_std), gaps)[0, 1]
    print(f"   r(log-transformed) = {r_log:.4f}")
    
    # ===== 6. Per-text gap decomposition =====
    print(f"\n6. Per-Text Gap Decomposition:")
    
    # gap = gamma * h_normalized . Delta_W
    # Decompose into: high-var dims (top 20% by W_U_col_std) vs low-var dims
    std_threshold = np.percentile(W_U_col_std, 80)
    high_var_mask = W_U_col_std > std_threshold
    
    gap_high_var = np.sum(gamma[high_var_mask] * h_normalized[:, high_var_mask] * Delta_W_arr[:, high_var_mask], axis=1)
    gap_low_var = np.sum(gamma[~high_var_mask] * h_normalized[:, ~high_var_mask] * Delta_W_arr[:, ~high_var_mask], axis=1)
    
    r_high_var, _ = stats.pearsonr(gap_high_var, gaps)
    r_low_var, _ = stats.pearsonr(gap_low_var, gaps)
    
    # Also: gamma-suppressed dims vs gamma-enhanced dims
    gamma_median = np.median(gamma)
    suppressed_mask = gamma < gamma_median  # gamma suppresses these
    enhanced_mask = gamma >= gamma_median
    
    gap_suppressed = np.sum(gamma[suppressed_mask] * h_normalized[:, suppressed_mask] * Delta_W_arr[:, suppressed_mask], axis=1)
    gap_enhanced = np.sum(gamma[enhanced_mask] * h_normalized[:, enhanced_mask] * Delta_W_arr[:, enhanced_mask], axis=1)
    
    r_suppressed, _ = stats.pearsonr(gap_suppressed, gaps)
    r_enhanced, _ = stats.pearsonr(gap_enhanced, gaps)
    
    print(f"   r(high-var W_U dims)    = {r_high_var:.4f}")
    print(f"   r(low-var W_U dims)     = {r_low_var:.4f}")
    print(f"   r(gamma-suppressed dims) = {r_suppressed:.4f}")
    print(f"   r(gamma-enhanced dims)   = {r_enhanced:.4f}")
    
    # Correlation between gap contributions
    r_high_low, _ = stats.pearsonr(gap_high_var, gap_low_var)
    r_supp_enh, _ = stats.pearsonr(gap_suppressed, gap_enhanced)
    print(f"   corr(high-var, low-var)    = {r_high_low:.4f}")
    print(f"   corr(suppressed, enhanced) = {r_supp_enh:.4f}")
    
    results = {
        "model_name": model_name,
        "r_vs_only": float(r_vs_only),
        "r_product": float(r_product),
        "r_main_effects": float(r_main),
        "r_interaction": float(r_interact),
        "r_lasso": float(r_lasso),
        "r_log": float(r_log),
        "r_high_var": float(r_high_var),
        "r_low_var": float(r_low_var),
        "r_suppressed": float(r_suppressed),
        "r_enhanced": float(r_enhanced),
        "corr_high_low": float(r_high_low),
        "corr_supp_enh": float(r_supp_enh),
        "interaction_coefs": {name: float(c) for name, c in zip(coef_names, reg_interact.coef_)},
    }
    
    return results


def experiment_p677(model, tokenizer, device, model_name):
    """P677: MLP Causal Mechanism.
    
    Key question: Why does MLP ratio > 1 increase offdiag (dimension correlation)?
    
    Hypotheses:
    H1: MLP output is low-rank → forces h into a low-dim subspace → high correlation
    H2: MLP weight matrix has high off-diagonal → directly creates correlations
    H3: Large MLP output drowns out the independent components of h
    
    Tests:
    1. Measure effective rank of MLP output at each layer
    2. Correlate MLP output rank with offdiag change
    3. Measure MLP weight matrix structure
    """
    print(f"\n{'='*60}")
    print(f"P677: MLP Causal Mechanism ({model_name})")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    d_model = W_U.shape[1]
    layers = get_layers(model)
    n_layers = len(layers)
    
    n_texts = 80
    test_texts = ALL_TEST_TEXTS[:n_texts]
    
    # Collect MLP output at multiple layers
    layer_data = {l: {"mlp_out": [], "post_ln": [], "logit_gaps": [], "Delta_Ws": []}
                  for l in range(0, n_layers, max(1, n_layers//6))}
    
    for i, text in enumerate(test_texts):
        if (i+1) % 20 == 0:
            print(f"  Processing text {i+1}/{n_texts}...")
        
        cache = {}
        def make_hook(layer_idx, name):
            def hook_fn(module, input, output):
                cache[(layer_idx, name)] = output[0].detach().cpu()
            return hook_fn
        
        hooks = []
        for l_idx in layer_data.keys():
            layer = layers[l_idx]
            if hasattr(layer, 'post_attention_layernorm'):
                hooks.append(layer.post_attention_layernorm.register_forward_hook(
                    make_hook(l_idx, 'post_ln')))
            if hasattr(layer, 'mlp'):
                hooks.append(layer.mlp.register_forward_hook(
                    make_hook(l_idx, 'mlp')))
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"].to(device),
                          attention_mask=inputs["attention_mask"].to(device),
                          output_hidden_states=True)
        
        logits = outputs.logits[0, -1, :].cpu().float().numpy()
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        gap = logits[top1_idx] - logits[top2_idx]
        Delta_W = W_U[top1_idx] - W_U[top2_idx]
        
        for l_idx in layer_data.keys():
            if (l_idx, 'post_ln') in cache:
                layer_data[l_idx]["post_ln"].append(extract_tensor(cache[(l_idx, 'post_ln')]))
            if (l_idx, 'mlp') in cache:
                layer_data[l_idx]["mlp_out"].append(extract_tensor(cache[(l_idx, 'mlp')]))
            layer_data[l_idx]["logit_gaps"].append(gap)
            layer_data[l_idx]["Delta_Ws"].append(Delta_W)
        
        for h in hooks:
            h.remove()
    
    # ===== Analyze each layer =====
    print(f"\n1. MLP Output Structure at Each Layer:")
    
    layer_results = []
    for l_idx in sorted(layer_data.keys()):
        data = layer_data[l_idx]
        if len(data["mlp_out"]) < 10:
            continue
        
        mlp_out = np.array(data["mlp_out"])
        post_ln = np.array(data["post_ln"])
        gaps = np.array(data["logit_gaps"])
        DW = np.array(data["Delta_Ws"])
        
        # MLP output effective rank (PR)
        mlp_centered = mlp_out - np.mean(mlp_out, axis=0)
        if mlp_centered.shape[0] > 1 and mlp_centered.shape[1] > 0:
            mlp_eigvals = np.linalg.svd(mlp_centered, compute_uv=False)[:50]
            mlp_PR = (np.sum(mlp_eigvals))**2 / (np.sum(mlp_eigvals**2) + 1e-10)
        else:
            mlp_PR = 0
        
        # Post-LN effective rank
        pln_centered = post_ln - np.mean(post_ln, axis=0)
        if pln_centered.shape[0] > 1 and pln_centered.shape[1] > 0:
            pln_eigvals = np.linalg.svd(pln_centered, compute_uv=False)[:50]
            pln_PR = (np.sum(pln_eigvals))**2 / (np.sum(pln_eigvals**2) + 1e-10)
        else:
            pln_PR = 0
        
        # MLP output offdiag
        mlp_cov = np.cov(mlp_out.T)
        if mlp_cov.shape[0] > 1:
            mlp_offdiag = 1 - np.sum(np.diag(mlp_cov)) / (np.sum(np.abs(mlp_cov)) + 1e-10)
        else:
            mlp_offdiag = 0
        
        # Post-LN offdiag
        pln_cov = np.cov(post_ln.T)
        if pln_cov.shape[0] > 1:
            pln_offdiag = 1 - np.sum(np.diag(pln_cov)) / (np.sum(np.abs(pln_cov)) + 1e-10)
        else:
            pln_offdiag = 0
        
        # Norms
        norm_mlp = np.mean(np.linalg.norm(mlp_out, axis=1))
        norm_pln = np.mean(np.linalg.norm(post_ln, axis=1))
        mlp_ratio = norm_mlp / (norm_pln + 1e-8)
        
        # h after MLP = post_ln + mlp_out (residual)
        h_after_mlp = post_ln + mlp_out
        h_after_cov = np.cov(h_after_mlp.T)
        if h_after_cov.shape[0] > 1:
            h_after_offdiag = 1 - np.sum(np.diag(h_after_cov)) / (np.sum(np.abs(h_after_cov)) + 1e-10)
        else:
            h_after_offdiag = 0
        
        layer_results.append({
            "layer": l_idx,
            "mlp_PR": float(mlp_PR),
            "pln_PR": float(pln_PR),
            "mlp_offdiag": float(mlp_offdiag),
            "pln_offdiag": float(pln_offdiag),
            "h_after_offdiag": float(h_after_offdiag),
            "offdiag_increase": float(h_after_offdiag - pln_offdiag),
            "mlp_ratio": float(mlp_ratio),
            "norm_mlp": float(norm_mlp),
            "norm_pln": float(norm_pln),
        })
        
        print(f"   L{l_idx:2d}: mlp_ratio={mlp_ratio:.3f}, mlp_PR={mlp_PR:.1f}, pln_PR={pln_PR:.1f}, "
              f"offdiag: pln={pln_offdiag:.3f} -> after={h_after_offdiag:.3f} (delta={h_after_offdiag-pln_offdiag:+.3f})")
    
    # ===== 2. Correlation analysis =====
    print(f"\n2. Correlation Analysis:")
    
    if len(layer_results) >= 3:
        ratios = [r["mlp_ratio"] for r in layer_results]
        offdiag_inc = [r["offdiag_increase"] for r in layer_results]
        mlp_prs = [r["mlp_PR"] for r in layer_results]
        
        r_ratio_offdiag, _ = stats.pearsonr(ratios, offdiag_inc)
        r_pr_offdiag, _ = stats.pearsonr(mlp_prs, offdiag_inc)
        r_ratio_pr, _ = stats.pearsonr(ratios, mlp_prs)
        
        print(f"   corr(mlp_ratio, offdiag_increase) = {r_ratio_offdiag:.4f}")
        print(f"   corr(mlp_PR, offdiag_increase)     = {r_pr_offdiag:.4f}")
        print(f"   corr(mlp_ratio, mlp_PR)            = {r_ratio_pr:.4f}")
    
    # ===== 3. MLP weight matrix analysis =====
    print(f"\n3. MLP Weight Matrix Analysis:")
    
    for l_idx in [0, n_layers//2, n_layers-1]:
        if l_idx >= len(layers):
            continue
        layer = layers[l_idx]
        if not hasattr(layer, 'mlp'):
            continue
        
        mlp = layer.mlp
        
        # Try to get the up-projection weight
        w_up = None
        if hasattr(mlp, 'up_proj') and hasattr(mlp.up_proj, 'weight'):
            w_up = mlp.up_proj.weight.detach().cpu().float().numpy()
        elif hasattr(mlp, 'fc1') and hasattr(mlp.fc1, 'weight'):
            w_up = mlp.fc1.weight.detach().cpu().float().numpy()
        elif hasattr(mlp, 'gate_proj') and hasattr(mlp.gate_proj, 'weight'):
            w_up = mlp.gate_proj.weight.detach().cpu().float().numpy()
        
        # Try to get the down-projection weight
        w_down = None
        if hasattr(mlp, 'down_proj') and hasattr(mlp.down_proj, 'weight'):
            w_down = mlp.down_proj.weight.detach().cpu().float().numpy()
        elif hasattr(mlp, 'fc2') and hasattr(mlp.fc2, 'weight'):
            w_down = mlp.fc2.weight.detach().cpu().float().numpy()
        
        if w_up is not None:
            # Effective rank of W_up
            w_up_centered = w_up - np.mean(w_up, axis=0)
            try:
                w_up_svd = np.linalg.svd(w_up_centered, compute_uv=False)[:50]
                w_up_PR = (np.sum(w_up_svd))**2 / (np.sum(w_up_svd**2) + 1e-10)
            except:
                w_up_PR = 0
            
            # Off-diagonal of W_up @ W_up.T (row correlation)
            w_cov = w_up @ w_up.T
            if w_cov.shape[0] > 1:
                w_offdiag = 1 - np.sum(np.diag(w_cov)) / (np.sum(np.abs(w_cov)) + 1e-10)
            else:
                w_offdiag = 0
            
            print(f"   L{l_idx:2d} W_up: shape={w_up.shape}, PR={w_up_PR:.1f}, offdiag={w_offdiag:.4f}")
        
        if w_down is not None:
            w_down_centered = w_down - np.mean(w_down, axis=0)
            try:
                w_down_svd = np.linalg.svd(w_down_centered, compute_uv=False)[:50]
                w_down_PR = (np.sum(w_down_svd))**2 / (np.sum(w_down_svd**2) + 1e-10)
            except:
                w_down_PR = 0
            print(f"   L{l_idx:2d} W_down: shape={w_down.shape}, PR={w_down_PR:.1f}")
    
    results = {
        "n_layers": n_layers,
        "layer_results": layer_results,
    }
    
    if len(layer_results) >= 3:
        results["r_ratio_offdiag"] = float(r_ratio_offdiag)
        results["r_pr_offdiag"] = float(r_pr_offdiag)
        results["r_ratio_pr"] = float(r_ratio_pr)
    
    return results


def experiment_p678(model, tokenizer, device, model_name):
    """P678: Gamma Optimization Target.
    
    Key question: Why does GLM4 gamma only do VS while Qwen3 gamma does VS+extra?
    
    Hypothesis: GLM4's training dynamics converged to a different local minimum
    where variance suppression alone is optimal, while Qwen3 needs additional 
    per-dimension adjustments.
    
    Tests:
    1. Compare gamma vs optimal Ridge gamma across models
    2. Compute per-dim gap contribution and see if gamma aligns
    3. Analyze gamma's frequency content (does gamma encode W_U row frequency?)
    """
    print(f"\n{'='*60}")
    print(f"P678: Gamma Optimization Target ({model_name})")
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
    
    # Collect data
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
    
    h_before_arr = np.array(h_before_list)
    gaps = np.array(logit_gaps)
    Delta_W_arr = np.array(Delta_Ws)
    
    h_centered = h_before_arr - np.mean(h_before_arr, axis=1, keepdims=True)
    h_std = np.std(h_before_arr, axis=1, keepdims=True) + 1e-8
    h_normalized = h_centered / h_std
    
    # ===== 1. Optimal gamma via Ridge =====
    print(f"\n1. Optimal Gamma Analysis:")
    
    # Find optimal gamma: maximize corr(gamma * h_normalized . Delta_W, gap)
    # This is equivalent to Ridge regression on X = h_normalized * Delta_W
    X_gap = h_normalized * Delta_W_arr  # [n_texts, d_model]
    
    reg = Ridge(alpha=1.0).fit(X_gap, gaps)
    gamma_optimal = reg.coef_
    
    r_actual, _ = stats.pearsonr(np.sum(gamma * h_normalized * Delta_W_arr, axis=1), gaps)
    r_optimal, _ = stats.pearsonr(np.sum(gamma_optimal * h_normalized * Delta_W_arr, axis=1), gaps)
    
    # Correlation between actual and optimal gamma
    r_gamma_opt, _ = stats.pearsonr(gamma, gamma_optimal)
    
    # How much of actual gamma is captured by -W_U_col_std direction?
    vs_direction = -W_U_col_std / (np.linalg.norm(W_U_col_std) + 1e-8)
    gamma_proj_vs = np.dot(gamma, vs_direction)
    opt_proj_vs = np.dot(gamma_optimal, vs_direction)
    
    print(f"   r(actual gamma)      = {r_actual:.4f}")
    print(f"   r(Ridge optimal)     = {r_optimal:.4f}")
    print(f"   corr(gamma, opt)     = {r_gamma_opt:.4f}")
    print(f"   gamma projection on VS dir = {gamma_proj_vs:.4f}")
    print(f"   opt projection on VS dir   = {opt_proj_vs:.4f}")
    
    # ===== 2. Per-dimension analysis =====
    print(f"\n2. Per-Dimension Gap Contribution:")
    
    # Per-dim contribution: c_i = mean(h_normalized_i * Delta_W_i)
    mean_gap_contrib = np.mean(h_normalized * Delta_W_arr, axis=0)
    abs_gap_contrib = np.abs(mean_gap_contrib)
    
    # How does gamma align with gap contribution?
    r_gamma_contrib, _ = stats.pearsonr(gamma, mean_gap_contrib)
    r_gamma_abs_contrib, _ = stats.pearsonr(gamma, abs_gap_contrib)
    r_opt_contrib, _ = stats.pearsonr(gamma_optimal, mean_gap_contrib)
    r_opt_abs_contrib, _ = stats.pearsonr(gamma_optimal, abs_gap_contrib)
    
    print(f"   corr(gamma, gap_contrib)          = {r_gamma_contrib:.4f}")
    print(f"   corr(gamma, |gap_contrib|)        = {r_gamma_abs_contrib:.4f}")
    print(f"   corr(optimal_gamma, gap_contrib)   = {r_opt_contrib:.4f}")
    print(f"   corr(optimal_gamma, |gap_contrib|) = {r_opt_abs_contrib:.4f}")
    
    # ===== 3. Decompose gamma into VS and residual =====
    print(f"\n3. Gamma Decomposition:")
    
    # Fit gamma = a * (-W_U_col_std) + residual
    slope, intercept, r_value, _, _ = stats.linregress(-W_U_col_std, gamma)
    gamma_vs = slope * (-W_U_col_std) + intercept
    gamma_residual = gamma - gamma_vs
    
    r_vs_component, _ = stats.pearsonr(
        np.sum(gamma_vs * h_normalized * Delta_W_arr, axis=1), gaps)
    r_res_component, _ = stats.pearsonr(
        np.sum(gamma_residual * h_normalized * Delta_W_arr, axis=1), gaps)
    
    print(f"   r(VS component)     = {r_vs_component:.4f}")
    print(f"   r(residual component) = {r_res_component:.4f}")
    print(f"   VS fraction of gamma var = {r_value**2:.4f}")
    
    # ===== 4. Residual gamma structure =====
    print(f"\n4. Residual Gamma Structure:")
    
    # What does residual gamma correlate with?
    residual_features = {
        "W_U_col_norms": np.linalg.norm(W_U, axis=0),
        "W_U_col_mean_abs": np.mean(np.abs(W_U), axis=0),
        "W_U_col_max": np.max(np.abs(W_U), axis=0),
        "|gap_contrib|": abs_gap_contrib,
        "gap_contrib_signed": mean_gap_contrib,
        "optimal_gamma": gamma_optimal,
    }
    
    for name, vec in residual_features.items():
        r, _ = stats.pearsonr(gamma_residual, vec)
        print(f"   corr(residual_gamma, {name:20s}) = {r:.4f}")
    
    # ===== 5. Gamma's frequency content =====
    print(f"\n5. Gamma Frequency Content:")
    
    # Sort dimensions by W_U_col_std and look at gamma pattern
    sorted_indices = np.argsort(W_U_col_std)
    gamma_sorted = gamma[sorted_indices]
    
    # FFT of gamma (as a function of W_U_col_std rank)
    gamma_fft = np.abs(np.fft.fft(gamma_sorted))[:50]
    gamma_fft[0] = 0  # remove DC
    total_power = np.sum(gamma_fft**2)
    low_freq_power = np.sum(gamma_fft[:5]**2)
    high_freq_power = np.sum(gamma_fft[5:]**2)
    
    print(f"   Low freq (1-5) power fraction = {low_freq_power/total_power:.4f}")
    print(f"   High freq (6+) power fraction = {high_freq_power/total_power:.4f}")
    
    # ===== 6. Per-sample optimal gamma =====
    print(f"\n6. Per-Sample vs Global Optimal:")
    
    # For each text, compute the "local optimal gamma" (just 1 sample, so it's trivial)
    # Instead, compute per-text gamma contribution
    per_text_gap = np.sum(gamma * h_normalized * Delta_W_arr, axis=1)
    per_text_gap_opt = np.sum(gamma_optimal * h_normalized * Delta_W_arr, axis=1)
    
    # How consistent is optimal gamma across texts?
    # Use LOO: train on n-1, test on 1
    loo_preds = []
    loo_actual = []
    for j in range(min(20, n_texts)):
        mask = np.ones(n_texts, dtype=bool)
        mask[j] = False
        reg_loo = Ridge(alpha=1.0).fit(X_gap[mask], gaps[mask])
        pred = reg_loo.predict(X_gap[j:j+1])[0]
        loo_preds.append(pred)
        loo_actual.append(gaps[j])
    
    r_loo, _ = stats.pearsonr(loo_preds, loo_actual[:len(loo_preds)])
    print(f"   LOO Ridge r = {r_loo:.4f} (n={len(loo_preds)})")
    
    results = {
        "model_name": model_name,
        "r_actual": float(r_actual),
        "r_optimal": float(r_optimal),
        "r_gamma_opt_corr": float(r_gamma_opt),
        "r_vs_component": float(r_vs_component),
        "r_res_component": float(r_res_component),
        "vs_var_fraction": float(r_value**2),
        "r_gamma_contrib": float(r_gamma_contrib),
        "r_gamma_abs_contrib": float(r_gamma_abs_contrib),
        "r_opt_contrib": float(r_opt_contrib),
        "r_opt_abs_contrib": float(r_opt_abs_contrib),
        "low_freq_fraction": float(low_freq_power/total_power),
        "r_loo": float(r_loo),
    }
    
    return results


def experiment_p679(model, tokenizer, device, model_name):
    """P679: Unified Language Equation v2.
    
    Key question: Can we find a unified equation that works for all three models?
    
    From P676: Interaction model improved over product but still limited
    From P678: gamma can be decomposed into VS + residual components
    
    New approach: Instead of three abstract factors, use the actual gap prediction
    signal (gamma * h_normalized . Delta_W) and measure its "quality" via 
    information-theoretic metrics.
    
    gap_quality = InformationQuality(gamma, h, Delta_W)
    
    where InformationQuality measures how much gap-relevant information
    is preserved through the LN transformation.
    """
    print(f"\n{'='*60}")
    print(f"P679: Unified Language Equation v2 ({model_name})")
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
    
    h_before_arr = np.array(h_before_list)
    h_after_arr = np.array(h_after_list)
    gaps = np.array(logit_gaps)
    Delta_W_arr = np.array(Delta_Ws)
    
    h_centered = h_before_arr - np.mean(h_before_arr, axis=1, keepdims=True)
    h_std = np.std(h_before_arr, axis=1, keepdims=True) + 1e-8
    h_normalized = h_centered / h_std
    
    # ===== 1. Information Quality Metrics =====
    print(f"\n1. Information Quality Metrics:")
    
    # A. SNR of gap signal before and after LN
    gap_before = np.sum(h_centered * Delta_W_arr, axis=1)
    gap_after_ln = np.sum(gamma * h_normalized * Delta_W_arr, axis=1)
    gap_actual = np.sum(h_after_arr * Delta_W_arr, axis=1)
    
    snr_before = np.mean(gap_before)**2 / (np.var(gap_before) + 1e-10)
    snr_after_ln = np.mean(gap_after_ln)**2 / (np.var(gap_after_ln) + 1e-10)
    snr_actual = np.mean(gap_actual)**2 / (np.var(gap_actual) + 1e-10)
    
    print(f"   SNR before LN = {snr_before:.4f}")
    print(f"   SNR after LN (gamma*normalized) = {snr_after_ln:.4f}")
    print(f"   SNR actual (h_after) = {snr_actual:.4f}")
    print(f"   LN SNR improvement = {snr_after_ln/snr_before:.1f}x")
    
    # B. Correlation metrics
    r_gamma_norm, _ = stats.pearsonr(gap_after_ln, gaps)
    r_actual, _ = stats.pearsonr(gap_actual, gaps)
    
    print(f"   r(gamma*normalized, gap) = {r_gamma_norm:.4f}")
    print(f"   r(h_after, gap) = {r_actual:.4f}")
    
    # C. Mutual information proxy: I(gap_signal; gap) ~ 0.5 * log(1 + r^2)
    mi_proxy_before = 0.5 * np.log(1 + stats.pearsonr(gap_before, gaps)[0]**2)
    mi_proxy_after = 0.5 * np.log(1 + r_gamma_norm**2)
    mi_proxy_actual = 0.5 * np.log(1 + r_actual**2)
    
    print(f"   MI proxy before LN = {mi_proxy_before:.4f}")
    print(f"   MI proxy after LN  = {mi_proxy_after:.4f}")
    print(f"   MI proxy actual    = {mi_proxy_actual:.4f}")
    
    # ===== 2. Gap decomposition =====
    print(f"\n2. Gap Signal Decomposition:")
    
    # gap_after = Σ_i γ_i * (h_i - μ)/σ * ΔW_i
    # = Σ_i γ_i * h_norm_i * ΔW_i
    # Decompose by gamma magnitude groups
    
    gamma_abs = np.abs(gamma)
    q25 = np.percentile(gamma_abs, 25)
    q50 = np.percentile(gamma_abs, 50)
    q75 = np.percentile(gamma_abs, 75)
    
    groups = {
        "bottom_25": gamma_abs <= q25,
        "mid_25_50": (gamma_abs > q25) & (gamma_abs <= q50),
        "mid_50_75": (gamma_abs > q50) & (gamma_abs <= q75),
        "top_25": gamma_abs > q75,
    }
    
    for name, mask in groups.items():
        gap_group = np.sum(gamma[mask] * h_normalized[:, mask] * Delta_W_arr[:, mask], axis=1)
        r_group, _ = stats.pearsonr(gap_group, gaps)
        var_explained = np.var(gap_group) / (np.var(gap_after_ln) + 1e-10)
        print(f"   {name:12s}: n_dims={np.sum(mask):4d}, r={r_group:.4f}, var_frac={var_explained:.4f}")
    
    # ===== 3. VS-adjusted gap signal =====
    print(f"\n3. VS-Adjusted Gap Signal:")
    
    # Instead of gamma, use -W_U_col_std as "universal" gamma
    gamma_universal = -W_U_col_std / (np.linalg.norm(W_U_col_std) + 1e-8) * np.linalg.norm(gamma)
    gap_universal = np.sum(gamma_universal * h_normalized * Delta_W_arr, axis=1)
    r_universal, _ = stats.pearsonr(gap_universal, gaps)
    
    print(f"   r(universal gamma = -W_U_col_std) = {r_universal:.4f}")
    print(f"   r(actual gamma)                   = {r_gamma_norm:.4f}")
    print(f"   Universal/Actual ratio             = {r_universal/r_gamma_norm:.4f}")
    
    # ===== 4. Unified equation candidates =====
    print(f"\n4. Unified Equation Candidates:")
    
    # Candidate 1: gap ~ a * (gamma * h_norm . DW) + b
    reg1 = Ridge(alpha=1.0).fit(gap_after_ln.reshape(-1, 1), gaps)
    r_eq1 = np.corrcoef(reg1.predict(gap_after_ln.reshape(-1, 1)), gaps)[0, 1]
    
    # Candidate 2: gap ~ a * (gamma * h_norm . DW) + b * (universal * h_norm . DW)
    X_eq2 = np.column_stack([gap_after_ln, gap_universal])
    reg2 = Ridge(alpha=1.0).fit(X_eq2, gaps)
    r_eq2 = np.corrcoef(reg2.predict(X_eq2), gaps)[0, 1]
    
    # Candidate 3: gap ~ a * h_after . DW + b (baseline)
    reg3 = Ridge(alpha=1.0).fit(gap_actual.reshape(-1, 1), gaps)
    r_eq3 = np.corrcoef(reg3.predict(gap_actual.reshape(-1, 1)), gaps)[0, 1]
    
    print(f"   Eq1: gap ~ gamma*h*DW:              r = {r_eq1:.4f}")
    print(f"   Eq2: gap ~ gamma*h*DW + uni*h*DW:   r = {r_eq2:.4f}")
    print(f"   Eq3: gap ~ h_after*DW (baseline):   r = {r_eq3:.4f}")
    
    # ===== 5. Cross-model prediction potential =====
    print(f"\n5. Model Characteristic Summary:")
    
    # Summary metrics for cross-model comparison
    model_info = get_model_info(model, model_name)
    
    summary = {
        "model_name": model_name,
        "d_model": d_model,
        "n_layers": model_info.n_layers,
        "gamma_std": float(np.std(gamma)),
        "gamma_mean": float(np.mean(gamma)),
        "gamma_cv": float(np.std(gamma) / (np.abs(np.mean(gamma)) + 1e-8)),
        "vs_corr": float(stats.pearsonr(gamma, W_U_col_std)[0]),
        "snr_before": float(snr_before),
        "snr_after_ln": float(snr_after_ln),
        "snr_actual": float(snr_actual),
        "r_gamma_norm": float(r_gamma_norm),
        "r_actual": float(r_actual),
        "r_universal": float(r_universal),
        "r_eq1": float(r_eq1),
        "r_eq2": float(r_eq2),
        "r_eq3": float(r_eq3),
        "mi_proxy_before": float(mi_proxy_before),
        "mi_proxy_after": float(mi_proxy_after),
        "mi_proxy_actual": float(mi_proxy_actual),
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase CLIV: Language Interaction Model")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["p676", "p677", "p678", "p679"])
    args = parser.parse_args()
    
    model_name = args.model
    experiment = args.experiment
    
    print(f"Loading model: {model_name}")
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"Model: {model_info.name}, d_model={model_info.d_model}, n_layers={model_info.n_layers}")
    
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'phase_cliv')
    os.makedirs(results_dir, exist_ok=True)
    
    start_time = time.time()
    
    if experiment == "p676":
        results = experiment_p676(model, tokenizer, device, model_name)
    elif experiment == "p677":
        results = experiment_p677(model, tokenizer, device, model_name)
    elif experiment == "p678":
        results = experiment_p678(model, tokenizer, device, model_name)
    elif experiment == "p679":
        results = experiment_p679(model, tokenizer, device, model_name)
    else:
        print(f"Unknown experiment: {experiment}")
        return
    
    elapsed = time.time() - start_time
    
    results["elapsed_seconds"] = elapsed
    
    # Save results
    result_file = os.path.join(results_dir, f"{experiment}_{model_name}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nResults saved to {result_file}")
    print(f"Elapsed: {elapsed:.1f}s")
    
    # Cleanup
    release_model(model)
    print("Model released.")


if __name__ == "__main__":
    main()
