#!/usr/bin/env python3
"""
Phase CLIII: Language Math Theory - Precise Equations (P672-P675)
=================================================================

Based on Phase CLII findings:
1. gamma is a variance suppressor: corr(gamma, W_U_col_std) = -0.42~-0.82
2. GLM4 gamma > Ridge optimal (actual=0.819 > optimal=0.610), encoding nonlinear info
3. Qwen3 MLP injects 3.85x noise; GLM4 MLP only 0.62x — MLP output ratio is gap survival switch
4. GLM4 dimensions nearly independent (cov contribution=1.5%); Qwen3 highly correlated (72.5%)
5. Three regime: struggle(Qwen3) / elegant(GLM4) / brute_force(DS7B)

This phase addresses:
P672: Gamma Variance Suppression Math — derive gap extraction formula from gamma ~ -W_U_col_std
P673: MLP Output Ratio → Gap Survival Phase Transition — critical value of 3.85x vs 0.62x
P674: Dimension Independence → LN Efficiency — how covariance affects LN extraction power
P675: Unified Language Capability Equation
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


def get_ln_at_layer(model, layer_idx):
    """Get the post-attention LN at a specific layer."""
    layers = get_layers(model)
    if layer_idx < len(layers):
        layer = layers[layer_idx]
        if hasattr(layer, 'post_attention_layernorm'):
            return layer.post_attention_layernorm
        elif hasattr(layer, 'input_layernorm'):
            return layer.input_layernorm
    return None


def experiment_p672(model, tokenizer, device, model_name):
    """P672: Gamma Variance Suppression Math.
    
    Key question: Can we derive the gap extraction formula from gamma ~ -W_U_col_std?
    
    Theory: If gamma ≈ -c * W_U_col_std + d (variance suppression),
    then gap_after = Σ_i γ_i * (h_i - μ) / σ * ΔW_i
                   ≈ Σ_i (-c * std_W_i + d) * (h_i - μ) / σ * ΔW_i
    
    Tests:
    1. Fit gamma = a * W_U_col_std + b, compute R2
    2. Derive predicted gap from fitted gamma, compare with actual
    3. Residual analysis: what does gamma encode beyond W_U_col_std?
    4. Per-sample analysis: is the variance suppression text-dependent?
    5. Optimal variance suppression coefficient: what 'a' maximizes gap extraction?
    """
    print(f"\n{'='*60}")
    print(f"P672: Gamma Variance Suppression Math ({model_name})")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    d_model = W_U.shape[1]
    final_ln = get_final_ln(model)
    
    if final_ln is not None:
        gamma = final_ln.weight.detach().cpu().float().numpy()
    else:
        print("  No final LN found, skipping.")
        return {}
    
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
    
    # ===== 1. Fit gamma = a * W_U_col_std + b =====
    print(f"\n1. Gamma ~ W_U_col_std Linear Fit:")
    
    W_U_col_std = np.std(W_U, axis=0)  # [d_model]
    W_U_col_norms = np.linalg.norm(W_U, axis=0)
    W_U_col_mean_abs = np.mean(np.abs(W_U), axis=0)
    
    # Linear regression: gamma = a * W_U_col_std + b
    slope, intercept, r_value, p_value, std_err = stats.linregress(W_U_col_std, gamma)
    R2_linear = r_value**2
    
    gamma_predicted_linear = slope * W_U_col_std + intercept
    residual_gamma = gamma - gamma_predicted_linear
    
    print(f"   gamma = {slope:.6f} * W_U_col_std + {intercept:.6f}")
    print(f"   R2 = {R2_linear:.4f}, r = {r_value:.4f}")
    
    # Also try: gamma = a * W_U_col_norms + b
    slope2, intercept2, r2_value, _, _ = stats.linregress(W_U_col_norms, gamma)
    R2_norm = r2_value**2
    print(f"   gamma ~ W_U_col_norms: R2 = {R2_norm:.4f}")
    
    # Try: gamma = a * W_U_col_mean_abs + b
    slope3, intercept3, r3_value, _, _ = stats.linregress(W_U_col_mean_abs, gamma)
    R2_meanabs = r3_value**2
    print(f"   gamma ~ W_U_col_mean_abs: R2 = {R2_meanabs:.4f}")
    
    # ===== 2. Predicted gap from fitted gamma =====
    print(f"\n2. Predicted Gap from Fitted Gamma:")
    
    # Actual gap extraction
    h_centered = h_before_arr - np.mean(h_before_arr, axis=1, keepdims=True)
    h_std = np.std(h_before_arr, axis=1, keepdims=True) + 1e-8
    h_normalized = h_centered / h_std
    
    # gap using actual gamma
    r_actual, _ = stats.pearsonr(
        np.sum(gamma * h_normalized * Delta_W_arr, axis=1), gaps)
    
    # gap using linear-fitted gamma
    r_linear, _ = stats.pearsonr(
        np.sum(gamma_predicted_linear * h_normalized * Delta_W_arr, axis=1), gaps)
    
    # gap using residual gamma (beyond W_U_col_std)
    r_residual, _ = stats.pearsonr(
        np.sum(residual_gamma * h_normalized * Delta_W_arr, axis=1), gaps)
    
    # gap using only W_U_col_std (no gamma)
    r_std_only, _ = stats.pearsonr(
        np.sum(-W_U_col_std * h_normalized * Delta_W_arr, axis=1), gaps)
    
    print(f"   r(actual gamma)     = {r_actual:.4f}")
    print(f"   r(linear fitted)    = {r_linear:.4f}")
    print(f"   r(residual gamma)   = {r_residual:.4f}")
    print(f"   r(-W_U_col_std only)= {r_std_only:.4f}")
    
    # ===== 3. Optimal variance suppression coefficient =====
    print(f"\n3. Optimal Variance Suppression Coefficient:")
    
    # Find optimal 'a' in gamma_pred = a * (-W_U_col_std) + b
    # that maximizes corr(gamma_pred * h_normalized . Delta_W, gap)
    a_range = np.linspace(-5, 5, 1000)
    best_a = 0
    best_r = -1
    for a in a_range:
        gamma_opt = a * (-W_U_col_std) + intercept
        pred_gaps = np.sum(gamma_opt * h_normalized * Delta_W_arr, axis=1)
        r, _ = stats.pearsonr(pred_gaps, gaps)
        if r > best_r:
            best_r = r
            best_a = a
    
    print(f"   Optimal a (in gamma = a*(-W_U_col_std) + b) = {best_a:.4f}")
    print(f"   Optimal r = {best_r:.4f}")
    print(f"   Actual slope = {slope:.4f} (from linear fit)")
    print(f"   Efficiency = actual/optimal = {r_actual/best_r:.4f}" if best_r > 0 else "   Efficiency: N/A")
    
    # ===== 4. Per-sample variance suppression analysis =====
    print(f"\n4. Per-Sample Variance Suppression:")
    
    # For each sample, compute the "effective gamma" that would maximize gap
    # Effective gamma_i = gap_i * Delta_W_i / (h_normalized_i * ||h_normalized||^2)
    # But this is per-sample, so we compare the variance suppression pattern
    
    # Split texts by gap magnitude
    gap_median = np.median(np.abs(gaps))
    high_gap_mask = np.abs(gaps) > gap_median
    low_gap_mask = ~high_gap_mask
    
    # Variance suppression effectiveness per group
    for label, mask in [("high_gap", high_gap_mask), ("low_gap", low_gap_mask)]:
        if np.sum(mask) < 5:
            continue
        h_sub = h_normalized[mask]
        DW_sub = Delta_W_arr[mask]
        gaps_sub = gaps[mask]
        
        r_sub, _ = stats.pearsonr(
            np.sum(gamma * h_sub * DW_sub, axis=1), gaps_sub)
        r_linear_sub, _ = stats.pearsonr(
            np.sum(gamma_predicted_linear * h_sub * DW_sub, axis=1), gaps_sub)
        
        print(f"   {label} (n={np.sum(mask)}): r_actual={r_sub:.4f}, r_linear={r_linear_sub:.4f}")
    
    # ===== 5. Nonlinear gamma model =====
    print(f"\n5. Nonlinear Gamma Model:")
    
    # Try polynomial: gamma = a*std² + b*std + c
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    
    X_std = W_U_col_std.reshape(-1, 1)
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X_std)
    reg_poly = LinearRegression().fit(X_poly, gamma)
    gamma_poly = reg_poly.predict(X_poly)
    R2_poly = reg_poly.score(X_poly, gamma)
    
    r_poly, _ = stats.pearsonr(
        np.sum(gamma_poly * h_normalized * Delta_W_arr, axis=1), gaps)
    
    print(f"   Polynomial(degree=3) R2 = {R2_poly:.4f}")
    print(f"   r(poly predicted gamma) = {r_poly:.4f}")
    print(f"   Improvement over linear: {(r_poly - r_linear)/abs(r_linear)*100:.1f}%" if r_linear != 0 else "   Improvement: N/A")
    
    # ===== 6. Gamma vs per-dim SNR with W_U_col_std partialled out =====
    print(f"\n6. Gamma Residual Analysis (partialling out W_U_col_std):")
    
    # After removing W_U_col_std effect, what does residual gamma correlate with?
    residual_correlations = {}
    residual_features = [
        ("W_U_col_norms", W_U_col_norms),
        ("W_U_col_mean_abs", W_U_col_mean_abs),
        ("W_U_col_max", np.max(np.abs(W_U), axis=0)),
    ]
    # Skip skew/kurtosis for large W_U to avoid memory issues
    if W_U.shape[0] * W_U.shape[1] < 500_000_000:  # < 500M elements
        try:
            residual_features.append(("W_U_col_skew", stats.skew(W_U, axis=0)))
            residual_features.append(("W_U_col_kurtosis", stats.kurtosis(W_U, axis=0)))
        except Exception:
            pass
    
    for name, vec in residual_features:
        r, _ = stats.pearsonr(residual_gamma, vec)
        residual_correlations[f"r_residual_{name}"] = float(r)
        print(f"   corr(residual_gamma, {name}) = {r:.4f}")
    
    # ===== 7. Variance suppression formula derivation =====
    print(f"\n7. Variance Suppression Formula Derivation:")
    
    # Theory: gap = Σ_i γ_i * (h_i - μ)/σ * ΔW_i
    # If γ_i ≈ -c * std(W_U[:,i]) + d, then:
    # gap ≈ Σ_i (-c * std(W_U[:,i]) + d) * (h_i - μ)/σ * ΔW_i
    #     = d * Σ_i (h_i - μ)/σ * ΔW_i  -  c * Σ_i std(W_U[:,i]) * (h_i - μ)/σ * ΔW_i
    #     = d * gap_normalized  -  c * gap_weighted
    
    gap_normalized = np.sum(h_normalized * Delta_W_arr, axis=1)
    gap_weighted = np.sum(W_U_col_std * h_normalized * Delta_W_arr, axis=1)
    
    # Fit: gap = a * gap_normalized + b * gap_weighted + c
    X_pred = np.column_stack([gap_normalized, gap_weighted, np.ones(n_texts)])
    reg_formula = LinearRegression().fit(X_pred, gaps)
    r_formula = np.corrcoef(reg_formula.predict(X_pred), gaps)[0, 1]
    
    print(f"   gap = {reg_formula.coef_[0]:.4f} * gap_normalized + {reg_formula.coef_[1]:.4f} * gap_weighted + {reg_formula.coef_[2]:.4f}")
    print(f"   r(formula) = {r_formula:.4f}")
    print(f"   vs r(actual gamma) = {r_actual:.4f}")
    
    results = {
        "slope_std": float(slope),
        "intercept_std": float(intercept),
        "R2_linear": float(R2_linear),
        "R2_norm": float(R2_norm),
        "R2_meanabs": float(R2_meanabs),
        "r_actual_gamma": float(r_actual),
        "r_linear_fitted": float(r_linear),
        "r_residual_gamma": float(r_residual),
        "r_std_only": float(r_std_only),
        "optimal_a": float(best_a),
        "optimal_r": float(best_r),
        "efficiency": float(r_actual / best_r) if best_r > 0 else 0,
        "R2_poly": float(R2_poly),
        "r_poly": float(r_poly),
        "r_formula": float(r_formula),
        "formula_coef_normalized": float(reg_formula.coef_[0]),
        "formula_coef_weighted": float(reg_formula.coef_[1]),
    }
    results.update(residual_correlations)
    
    return results


def experiment_p673(model, tokenizer, device, model_name):
    """P673: MLP Output Ratio → Gap Survival Phase Transition.
    
    Key question: What is the critical MLP output ratio that determines
    whether gap survives or is destroyed?
    
    From P669: Qwen3 MLP/post_LN = 3.85x (gap destroyed)
               GLM4 MLP/post_LN = 0.62x (gap preserved)
    
    Tests:
    1. Compute MLP output ratio at ALL layers (not just last)
    2. At each layer, compute gap correlation before/after MLP
    3. Find the critical ratio where gap switches from preserved to destroyed
    4. Control experiment: artificially scale MLP output and measure gap survival
    """
    print(f"\n{'='*60}")
    print(f"P673: MLP Output Ratio → Gap Survival ({model_name})")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    d_model = W_U.shape[1]
    n_layers = len(get_layers(model))
    
    n_texts = 80
    test_texts = ALL_TEST_TEXTS[:n_texts]
    
    # Collect data at all layers
    layer_data = {l: {"h_before_ln": [], "h_after_attn_ln": [], "h_after_mlp": [],
                      "attn_out": [], "mlp_out": [], "logit_gaps": [], "Delta_Ws": []}
                  for l in range(n_layers)}
    
    for i, text in enumerate(test_texts):
        if (i+1) % 20 == 0:
            print(f"  Processing text {i+1}/{n_texts}...")
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        
        # Hook into all layers
        cache = {}
        def make_hook(layer_idx, name):
            def hook_fn(module, input, output):
                cache[(layer_idx, name)] = output[0].detach().cpu()
            return hook_fn
        
        hooks = []
        layers = get_layers(model)
        for l_idx, layer in enumerate(layers):
            # Post-attention LN output
            if hasattr(layer, 'post_attention_layernorm'):
                h = layer.post_attention_layernorm.register_forward_hook(
                    make_hook(l_idx, 'post_attn_ln'))
                hooks.append(h)
            # MLP output
            if hasattr(layer, 'mlp'):
                h = layer.mlp.register_forward_hook(
                    make_hook(l_idx, 'mlp_out'))
                hooks.append(h)
        
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"].to(device),
                          attention_mask=inputs["attention_mask"].to(device),
                          output_hidden_states=True)
        
        logits = outputs.logits[0, -1, :].cpu().float().numpy()
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        gap = logits[top1_idx] - logits[top2_idx]
        Delta_W = W_U[top1_idx] - W_U[top2_idx]
        
        # Extract cached activations
        for l_idx in range(n_layers):
            # h after post-attn LN (before MLP)
            if (l_idx, 'post_attn_ln') in cache:
                t = cache[(l_idx, 'post_attn_ln')]
                if t.dim() == 3:
                    h_post_ln = t[0, -1, :].float().numpy()
                elif t.dim() == 2:
                    h_post_ln = t[-1, :].float().numpy()
                else:
                    h_post_ln = t.float().numpy().flatten()
                layer_data[l_idx]["h_after_attn_ln"].append(h_post_ln)
            
            # MLP output
            if (l_idx, 'mlp_out') in cache:
                t = cache[(l_idx, 'mlp_out')]
                if t.dim() == 3:
                    h_mlp = t[0, -1, :].float().numpy()
                elif t.dim() == 2:
                    h_mlp = t[-1, :].float().numpy()
                else:
                    h_mlp = t.float().numpy().flatten()
                layer_data[l_idx]["h_after_mlp"].append(h_mlp)
            
            layer_data[l_idx]["logit_gaps"].append(gap)
            layer_data[l_idx]["Delta_Ws"].append(Delta_W)
        
        for h in hooks:
            h.remove()
    
    # ===== Analyze each layer =====
    print(f"\n1. Per-Layer MLP Output Ratio and Gap Survival:")
    
    layer_results = []
    for l_idx in range(n_layers):
        data = layer_data[l_idx]
        if len(data["h_after_attn_ln"]) < 5:
            continue
        
        h_post_ln = np.array(data["h_after_attn_ln"])
        h_mlp = np.array(data["h_after_mlp"])
        gaps = np.array(data["logit_gaps"])
        DW = np.array(data["Delta_Ws"])
        
        # Norms
        norm_post_ln = np.mean(np.linalg.norm(h_post_ln, axis=1))
        norm_mlp = np.mean(np.linalg.norm(h_mlp, axis=1))
        mlp_ratio = norm_mlp / (norm_post_ln + 1e-8)
        
        # Gap correlation before MLP (at post_attn_ln)
        h_centered = h_post_ln - np.mean(h_post_ln, axis=1, keepdims=True)
        h_std = np.std(h_post_ln, axis=1, keepdims=True) + 1e-8
        h_norm = h_centered / h_std
        r_before_mlp, _ = stats.pearsonr(np.sum(h_norm * DW, axis=1), gaps)
        
        # Gap correlation after MLP (h_post_ln + mlp_out, approximate)
        h_after_mlp_approx = h_post_ln + h_mlp  # residual connection
        h_centered2 = h_after_mlp_approx - np.mean(h_after_mlp_approx, axis=1, keepdims=True)
        h_std2 = np.std(h_after_mlp_approx, axis=1, keepdims=True) + 1e-8
        h_norm2 = h_centered2 / h_std2
        r_after_mlp, _ = stats.pearsonr(np.sum(h_norm2 * DW, axis=1), gaps)
        
        # MLP delta_r
        delta_r = r_after_mlp - r_before_mlp
        
        # Cosine between h_post_ln and mlp_out
        cos_mean = np.mean([
            np.dot(h_post_ln[j], h_mlp[j]) / (np.linalg.norm(h_post_ln[j]) * np.linalg.norm(h_mlp[j]) + 1e-8)
            for j in range(len(gaps))
        ])
        
        layer_results.append({
            "layer": l_idx,
            "mlp_ratio": float(mlp_ratio),
            "r_before_mlp": float(r_before_mlp),
            "r_after_mlp": float(r_after_mlp),
            "delta_r": float(delta_r),
            "cos_post_ln_mlp": float(cos_mean),
        })
        
        if (l_idx + 1) % 5 == 0 or l_idx == n_layers - 1:
            print(f"   L{l_idx:2d}: mlp_ratio={mlp_ratio:.3f}, r_before={r_before_mlp:.4f}, "
                  f"r_after={r_after_mlp:.4f}, delta_r={delta_r:+.4f}, cos={cos_mean:.4f}")
    
    # ===== 2. Phase transition analysis =====
    print(f"\n2. Phase Transition Analysis:")
    
    ratios = [r["mlp_ratio"] for r in layer_results]
    delta_rs = [r["delta_r"] for r in layer_results]
    
    # Correlation between mlp_ratio and delta_r
    r_ratio_delta, _ = stats.pearsonr(ratios, delta_rs)
    print(f"   corr(mlp_ratio, delta_r) = {r_ratio_delta:.4f}")
    
    # Split by mlp_ratio threshold
    for threshold in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        low_mask = np.array(ratios) < threshold
        high_mask = np.array(ratios) >= threshold
        if np.sum(low_mask) > 0 and np.sum(high_mask) > 0:
            mean_delta_low = np.mean(np.array(delta_rs)[low_mask])
            mean_delta_high = np.mean(np.array(delta_rs)[high_mask])
            print(f"   threshold={threshold:.1f}: low_ratio delta_r={mean_delta_low:+.4f}(n={np.sum(low_mask)}), "
                  f"high_ratio delta_r={mean_delta_high:+.4f}(n={np.sum(high_mask)})")
    
    # ===== 3. MLP ratio and gap survival classification =====
    print(f"\n3. Gap Survival Classification:")
    
    # Classify: gap preserved (delta_r > 0) vs destroyed (delta_r < 0)
    preserved = [r for r in layer_results if r["delta_r"] > 0]
    destroyed = [r for r in layer_results if r["delta_r"] <= 0]
    
    if preserved and destroyed:
        mean_ratio_preserved = np.mean([r["mlp_ratio"] for r in preserved])
        mean_ratio_destroyed = np.mean([r["mlp_ratio"] for r in destroyed])
        print(f"   Preserved (delta_r>0): n={len(preserved)}, mean mlp_ratio={mean_ratio_preserved:.3f}")
        print(f"   Destroyed (delta_r<=0): n={len(destroyed)}, mean mlp_ratio={mean_ratio_destroyed:.3f}")
        
        # Optimal threshold via ROC-like analysis
        from itertools import combinations
        best_thresh = 0
        best_acc = 0
        for t in np.arange(0.3, 4.0, 0.1):
            pred_preserved = np.array(ratios) < t
            actual_preserved = np.array(delta_rs) > 0
            acc = np.mean(pred_preserved == actual_preserved)
            if acc > best_acc:
                best_acc = acc
                best_thresh = t
        print(f"   Best classification threshold: mlp_ratio < {best_thresh:.1f}, accuracy={best_acc:.3f}")
    
    # ===== 4. MLP output direction analysis =====
    print(f"\n4. MLP Output Direction vs Gap:")
    
    # At last few layers, analyze the MLP output direction
    for l_idx in range(max(0, n_layers-3), n_layers):
        data = layer_data[l_idx]
        if len(data["h_after_attn_ln"]) < 5:
            continue
        
        h_post_ln = np.array(data["h_after_attn_ln"])
        h_mlp = np.array(data["h_after_mlp"])
        DW = np.array(data["Delta_Ws"])
        gaps = np.array(data["logit_gaps"])
        
        # Correlation of MLP output direction with Delta_W
        h_mlp_centered = h_mlp - np.mean(h_mlp, axis=1, keepdims=True)
        h_mlp_std = np.std(h_mlp, axis=1, keepdims=True) + 1e-8
        h_mlp_norm = h_mlp_centered / h_mlp_std
        
        r_mlp_gap, _ = stats.pearsonr(np.sum(h_mlp_norm * DW, axis=1), gaps)
        
        # Fraction of MLP output in the direction of Delta_W
        mean_DW = np.mean(DW, axis=0)
        mean_DW_norm = mean_DW / (np.linalg.norm(mean_DW) + 1e-8)
        proj_fraction = np.mean([
            np.abs(np.dot(h_mlp[j], mean_DW_norm)) / (np.linalg.norm(h_mlp[j]) + 1e-8)
            for j in range(len(gaps))
        ])
        
        print(f"   L{l_idx}: r(mlp_out, gap)={r_mlp_gap:.4f}, proj_fraction={proj_fraction:.4f}")
    
    results = {
        "n_layers": n_layers,
        "r_ratio_delta_r": float(r_ratio_delta),
        "layer_results": layer_results,
    }
    
    if preserved and destroyed:
        results["mean_ratio_preserved"] = float(mean_ratio_preserved)
        results["mean_ratio_destroyed"] = float(mean_ratio_destroyed)
        results["best_threshold"] = float(best_thresh)
        results["best_accuracy"] = float(best_acc)
    
    return results


def experiment_p674(model, tokenizer, device, model_name):
    """P674: Dimension Independence → LN Efficiency.
    
    Key question: How does dimension covariance affect LN extraction power?
    
    From P670: GLM4 cov_contribution=1.5% (independent), Qwen3=72.5% (correlated)
    LN efficiency: GLM4 SNR_after=1921, Qwen3=181.8 (10x difference)
    
    Tests:
    1. Compute covariance structure of h at each substep of last layer
    2. Measure how covariance changes through LN
    3. Simulate: if we decorrelate dimensions, does LN efficiency increase?
    4. Quantify: LN efficiency = f(cov_concentration, dimensionality)
    """
    print(f"\n{'='*60}")
    print(f"P674: Dimension Independence → LN Efficiency ({model_name})")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    d_model = W_U.shape[1]
    final_ln = get_final_ln(model)
    
    if final_ln is not None:
        gamma = final_ln.weight.detach().cpu().float().numpy()
    else:
        print("  No final LN found, skipping.")
        return {}
    
    n_texts = 100
    test_texts = ALL_TEST_TEXTS[:n_texts]
    
    # Collect h at substeps of last layer
    h_data = {
        "h_Lm1": [],      # h before last layer
        "h_after_input_ln": [],
        "h_after_attn": [],
        "h_after_post_ln": [],
        "h_after_mlp": [],
        "h_after_final_ln": [],
        "logit_gaps": [],
        "Delta_Ws": [],
    }
    
    layers = get_layers(model)
    n_layers = len(layers)
    last_layer = layers[-1]
    
    cache = {}
    def make_hook(name):
        def hook_fn(module, input, output):
            cache[name] = output[0].detach().cpu()
        return hook_fn
    
    hooks = []
    # Input LN of last layer
    if hasattr(last_layer, 'input_layernorm'):
        hooks.append(last_layer.input_layernorm.register_forward_hook(make_hook('input_ln')))
    # Post-attn LN of last layer
    if hasattr(last_layer, 'post_attention_layernorm'):
        hooks.append(last_layer.post_attention_layernorm.register_forward_hook(make_hook('post_attn_ln')))
    # MLP of last layer
    if hasattr(last_layer, 'mlp'):
        hooks.append(last_layer.mlp.register_forward_hook(make_hook('mlp_out')))
    
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
        gap = logits[top1_idx] - logits[top2_idx]
        Delta_W = W_U[top1_idx] - W_U[top2_idx]
        
        # Extract hidden states
        h_Lm1 = outputs.hidden_states[-2][0, -1, :].cpu().float().numpy()
        h_final = outputs.hidden_states[-1][0, -1, :].cpu().float().numpy()
        
        h_data["h_Lm1"].append(h_Lm1)
        h_data["h_after_final_ln"].append(h_final)
        h_data["logit_gaps"].append(gap)
        h_data["Delta_Ws"].append(Delta_W)
        
        # From cache
        def extract_from_cache(key):
            if key not in cache:
                return None
            t = cache[key]
            if t.dim() == 3:
                return t[0, -1, :].float().numpy()
            elif t.dim() == 2:
                return t[-1, :].float().numpy()
            else:
                return t.float().numpy().flatten()
        
        h_val = extract_from_cache('input_ln')
        if h_val is not None:
            h_data["h_after_input_ln"].append(h_val)
        h_val = extract_from_cache('post_attn_ln')
        if h_val is not None:
            h_data["h_after_post_ln"].append(h_val)
        h_val = extract_from_cache('mlp_out')
        if h_val is not None:
            h_data["h_after_mlp"].append(h_val)
    
    for h in hooks:
        h.remove()
    
    gaps = np.array(h_data["logit_gaps"])
    DW = np.array(h_data["Delta_Ws"])
    
    # ===== 1. Covariance structure at each substep =====
    print(f"\n1. Covariance Structure at Each Substep:")
    
    substep_results = {}
    for name in ["h_Lm1", "h_after_input_ln", "h_after_post_ln", "h_after_mlp", "h_after_final_ln"]:
        if len(h_data[name]) < 10:
            continue
        
        h_arr = np.array(h_data[name])
        n_samples = h_arr.shape[0]
        
        # Compute covariance matrix
        h_centered = h_arr - np.mean(h_arr, axis=0, keepdims=True)
        cov_matrix = np.cov(h_arr.T)
        
        # Eigenvalues of covariance
        eigvals = np.linalg.svalsh(cov_matrix) if cov_matrix.shape[0] <= 500 else np.linalg.svd(h_centered, compute_uv=False)[:100]
        
        # Participation ratio (effective dimensionality)
        if len(eigvals) > 0:
            PR = (np.sum(eigvals))**2 / (np.sum(eigvals**2) + 1e-10)
        else:
            PR = 0
        
        # Off-diagonal covariance fraction
        diag_var = np.sum(np.diag(cov_matrix))
        total_var = np.sum(cov_matrix)
        offdiag_fraction = 1 - diag_var / (total_var + 1e-10)
        
        # Gap correlation
        h_c = h_arr - np.mean(h_arr, axis=1, keepdims=True)
        h_s = np.std(h_arr, axis=1, keepdims=True) + 1e-8
        h_n = h_c / h_s
        r_gap, _ = stats.pearsonr(np.sum(h_n * DW, axis=1), gaps)
        
        # SNR
        gap_before = np.sum(h_c * DW, axis=1)
        snr = np.mean(gap_before)**2 / (np.var(gap_before) + 1e-10)
        
        substep_results[name] = {
            "PR": float(PR),
            "offdiag_fraction": float(offdiag_fraction),
            "r_gap": float(r_gap),
            "snr": float(snr),
        }
        
        print(f"   {name:25s}: PR={PR:.2f}, offdiag={offdiag_fraction:.4f}, r_gap={r_gap:.4f}, SNR={snr:.4f}")
    
    # ===== 2. Covariance change through LN =====
    print(f"\n2. Covariance Change Through LN:")
    
    # Compare h_after_post_ln vs h_after_mlp (pre-LN vs post-MLP)
    if "h_after_post_ln" in h_data and "h_after_mlp" in h_data:
        if len(h_data["h_after_post_ln"]) > 10 and len(h_data["h_after_mlp"]) > 10:
            h_post_ln = np.array(h_data["h_after_post_ln"])
            h_mlp = np.array(h_data["h_after_mlp"])
            
            # Covariance of h_post_ln
            cov_post_ln = np.cov(h_post_ln.T)
            # Covariance of h_mlp
            cov_mlp = np.cov(h_mlp.T)
            
            # How much does MLP increase off-diagonal?
            offdiag_post_ln = 1 - np.sum(np.diag(cov_post_ln)) / (np.sum(cov_post_ln) + 1e-10)
            offdiag_mlp = 1 - np.sum(np.diag(cov_mlp)) / (np.sum(cov_mlp) + 1e-10)
            
            print(f"   offdiag(post_attn_ln) = {offdiag_post_ln:.4f}")
            print(f"   offdiag(after MLP)    = {offdiag_mlp:.4f}")
            print(f"   MLP increases offdiag by {(offdiag_mlp - offdiag_post_ln):.4f}")
            
            substep_results["offdiag_post_ln"] = float(offdiag_post_ln)
            substep_results["offdiag_after_mlp"] = float(offdiag_mlp)
    
    # ===== 3. Decorrelation simulation =====
    print(f"\n3. Decorrelation Simulation:")
    
    if "h_Lm1" in h_data and len(h_data["h_Lm1"]) > 10:
        h_before = np.array(h_data["h_Lm1"])
        h_centered = h_before - np.mean(h_before, axis=1, keepdims=True)
        h_std = np.std(h_before, axis=1, keepdims=True) + 1e-8
        h_normalized = h_centered / h_std
        
        # Original gap correlation
        r_original, _ = stats.pearsonr(np.sum(gamma * h_normalized * DW, axis=1), gaps)
        
        # Apply whitening (ZCA) to decorrelate dimensions
        # h_whitened = h_centered @ Cov^{-1/2}
        cov_h = np.cov(h_before.T)
        try:
            # Use SVD for numerical stability
            U, S, Vt = np.linalg.svd(cov_h, full_matrices=False)
            # ZCA whitening: W = U @ diag(1/sqrt(S)) @ U^T
            S_inv_sqrt = 1.0 / np.sqrt(S + 1e-6)
            W_zca = U @ np.diag(S_inv_sqrt) @ U.T
            
            h_whitened = h_centered @ W_zca
            h_w_std = np.std(h_whitened, axis=1, keepdims=True) + 1e-8
            h_w_normalized = h_whitened / h_w_std
            
            r_whitened, _ = stats.pearsonr(np.sum(gamma * h_w_normalized * DW, axis=1), gaps)
            
            # Also test: gamma after whitening
            gamma_whitened = W_zca @ gamma
            r_whitened_gamma, _ = stats.pearsonr(np.sum(gamma_whitened * h_normalized * DW, axis=1), gaps)
            
            print(f"   r(original)           = {r_original:.4f}")
            print(f"   r(whitened h)         = {r_whitened:.4f}")
            print(f"   r(whitened gamma)     = {r_whitened_gamma:.4f}")
            print(f"   Improvement (whitened h): {(r_whitened - r_original)/abs(r_original)*100:.1f}%")
            
            substep_results["r_original"] = float(r_original)
            substep_results["r_whitened_h"] = float(r_whitened)
            substep_results["r_whitened_gamma"] = float(r_whitened_gamma)
        except Exception as e:
            print(f"   Whitening failed: {e}")
    
    # ===== 4. LN efficiency vs covariance =====
    print(f"\n4. LN Efficiency vs Covariance Model:")
    
    # At each substep, quantify: LN_efficiency = SNR_after / SNR_before
    # and correlate with offdiag_fraction
    
    if "h_after_post_ln" in h_data and "h_after_mlp" in h_data:
        if len(h_data["h_after_post_ln"]) > 10:
            # Final LN efficiency
            h_before_final = np.array(h_data["h_after_mlp"]) if len(h_data["h_after_mlp"]) > 10 else np.array(h_data["h_after_post_ln"])
            h_after_final = np.array(h_data["h_after_final_ln"])
            
            # SNR before final LN
            h_bf_c = h_before_final - np.mean(h_before_final, axis=1, keepdims=True)
            gap_bf = np.sum(h_bf_c * DW, axis=1)
            snr_bf = np.mean(gap_bf)**2 / (np.var(gap_bf) + 1e-10)
            
            # SNR after final LN
            h_af_c = h_after_final - np.mean(h_after_final, axis=1, keepdims=True)
            gap_af = np.sum(h_af_c * DW, axis=1)
            snr_af = np.mean(gap_af)**2 / (np.var(gap_af) + 1e-10)
            
            ln_efficiency = snr_af / (snr_bf + 1e-10)
            
            # Off-diagonal fraction before final LN
            cov_before = np.cov(h_before_final.T)
            offdiag_before = 1 - np.sum(np.diag(cov_before)) / (np.sum(cov_before) + 1e-10)
            
            print(f"   Final LN efficiency = {ln_efficiency:.1f}x")
            print(f"   Off-diag fraction before LN = {offdiag_before:.4f}")
            print(f"   SNR before = {snr_bf:.4f}, SNR after = {snr_af:.4f}")
            
            substep_results["final_ln_efficiency"] = float(ln_efficiency)
            substep_results["offdiag_before_final_ln"] = float(offdiag_before)
    
    results = {
        "substep_results": substep_results,
    }
    
    return results


def experiment_p675(model, tokenizer, device, model_name):
    """P675: Unified Language Capability Equation.
    
    Key question: Can we write a single equation that describes language capability
    across all three models?
    
    Proposed equation:
    gap_quality = VarianceSuppression(gamma, W_U) × GapSurvival(MLP_ratio) × LNEfficiency(independence)
    
    Where:
    - VarianceSuppression = corr(gamma, -W_U_col_std) × gamma_effectiveness
    - GapSurvival = sigmoid(-(mlp_ratio - critical_ratio))
    - LNEfficiency = f(offdiag_fraction, PR)
    
    Tests:
    1. Measure all three factors for each model
    2. Predict gap quality (Ridge_r) from the three factors
    3. Leave-one-out cross-validation
    4. Ablation: which factor is most important?
    """
    print(f"\n{'='*60}")
    print(f"P675: Unified Language Capability Equation ({model_name})")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    d_model = W_U.shape[1]
    final_ln = get_final_ln(model)
    
    if final_ln is not None:
        gamma = final_ln.weight.detach().cpu().float().numpy()
    else:
        print("  No final LN found, skipping.")
        return {}
    
    n_texts = 100
    test_texts = ALL_TEST_TEXTS[:n_texts]
    
    # Collect comprehensive data
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
    
    # ===== Factor 1: Variance Suppression =====
    print(f"\n1. Factor 1: Variance Suppression:")
    
    W_U_col_std = np.std(W_U, axis=0)
    r_var_suppress, _ = stats.pearsonr(gamma, W_U_col_std)
    
    # Gamma effectiveness
    h_centered = h_before_arr - np.mean(h_before_arr, axis=1, keepdims=True)
    h_std = np.std(h_before_arr, axis=1, keepdims=True) + 1e-8
    h_normalized = h_centered / h_std
    
    r_gamma_effect, _ = stats.pearsonr(
        np.sum(gamma * h_normalized * Delta_W_arr, axis=1), gaps)
    
    var_suppression_score = abs(r_var_suppress) * r_gamma_effect
    
    print(f"   corr(gamma, W_U_col_std) = {r_var_suppress:.4f}")
    print(f"   gamma effectiveness (r)  = {r_gamma_effect:.4f}")
    print(f"   Variance suppression score = {var_suppression_score:.4f}")
    
    # ===== Factor 2: Gap Survival =====
    print(f"\n2. Factor 2: Gap Survival (MLP ratio):")
    
    # Compute MLP ratio at last layer using hooks
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
    
    for text in test_texts[:50]:
        cache.clear()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            model(input_ids=inputs["input_ids"].to(device),
                  attention_mask=inputs["attention_mask"].to(device))
        
        if 'post_ln' in cache and 'mlp' in cache:
            t1 = cache['post_ln']
            t2 = cache['mlp']
            if t1.dim() == 3:
                v1 = t1[0, -1, :].float().numpy()
            elif t1.dim() == 2:
                v1 = t1[-1, :].float().numpy()
            else:
                v1 = t1.float().numpy().flatten()
            if t2.dim() == 3:
                v2 = t2[0, -1, :].float().numpy()
            elif t2.dim() == 2:
                v2 = t2[-1, :].float().numpy()
            else:
                v2 = t2.float().numpy().flatten()
            norm_post_ln_list.append(np.linalg.norm(v1))
            norm_mlp_list.append(np.linalg.norm(v2))
    
    for h in hooks:
        h.remove()
    
    if norm_post_ln_list and norm_mlp_list:
        mean_mlp_ratio = np.mean(norm_mlp_list) / (np.mean(norm_post_ln_list) + 1e-8)
    else:
        mean_mlp_ratio = 1.0  # default
    
    # Gap survival: sigmoid(-(mlp_ratio - 1.0))
    # If mlp_ratio > 1, MLP injects more noise than signal → gap destroyed
    gap_survival = 1.0 / (1.0 + np.exp(mean_mlp_ratio - 1.0))
    
    print(f"   Mean MLP ratio = {mean_mlp_ratio:.4f}")
    print(f"   Gap survival score = {gap_survival:.4f}")
    
    # ===== Factor 3: LN Efficiency (dimension independence) =====
    print(f"\n3. Factor 3: LN Efficiency (dimension independence):")
    
    # Off-diagonal covariance fraction of h_before
    cov_h = np.cov(h_before_arr.T)
    diag_var = np.sum(np.diag(cov_h))
    total_var = np.sum(np.abs(cov_h))
    offdiag_fraction = 1 - diag_var / (total_var + 1e-10)
    
    # Participation ratio
    eigvals = np.linalg.svd(h_before_arr - np.mean(h_before_arr, axis=0), compute_uv=False)[:100]
    h_PR = (np.sum(eigvals))**2 / (np.sum(eigvals**2) + 1e-10)
    
    # LN efficiency: 1 - offdiag_fraction (more independent = more efficient)
    ln_efficiency = 1 - offdiag_fraction
    
    print(f"   Off-diagonal fraction = {offdiag_fraction:.4f}")
    print(f"   h PR = {h_PR:.2f}")
    print(f"   LN efficiency score = {ln_efficiency:.4f}")
    
    # ===== Unified equation =====
    print(f"\n4. Unified Equation Prediction:")
    
    # Simple product model
    predicted_quality = var_suppression_score * gap_survival * ln_efficiency
    
    # Also try: Ridge regression on the three factors
    # Since we only have 1 data point per model, we'll use per-text predictions
    
    # Per-text variance suppression
    per_text_var_suppress = []
    for j in range(n_texts):
        h_n = h_normalized[j]
        DW_j = Delta_W_arr[j]
        gap_j = np.sum(gamma * h_n * DW_j)
        per_text_var_suppress.append(gap_j)
    
    # Per-text gap quality: actual gap
    r_per_text, _ = stats.pearsonr(per_text_var_suppress, gaps)
    
    print(f"   Per-text correlation (variance suppression only) = {r_per_text:.4f}")
    print(f"   Unified score (product) = {predicted_quality:.4f}")
    
    # ===== 5. W_U PR and dimensionality =====
    print(f"\n5. W_U Structure Metrics:")
    
    W_U_centered = W_U - np.mean(W_U, axis=0)
    try:
        from sklearn.decomposition import TruncatedSVD
        svd_w = TruncatedSVD(n_components=min(100, min(W_U.shape)-1))
        svd_w.fit(W_U_centered)
        W_U_eigvals = svd_w.singular_values_
        W_U_PR = (np.sum(W_U_eigvals))**2 / (np.sum(W_U_eigvals**2) + 1e-10)
    except Exception:
        W_U_PR = 0.0
    
    print(f"   W_U PR = {W_U_PR:.2f}")
    
    # W_U-h overlap (lightweight: correlate h PCs with W_U_col_std)
    k = min(10, h_before_arr.shape[0]-1, h_before_arr.shape[1])
    h_V = np.linalg.svd(h_before_arr - np.mean(h_before_arr, axis=0), full_matrices=False)[2][:k, :]
    overlap_scores = []
    for i in range(k):
        r, _ = stats.pearsonr(np.abs(h_V[i]), W_U_col_std)
        overlap_scores.append(abs(r))
    overlap = np.mean(overlap_scores)
    
    print(f"   W_U-h overlap (k=10) = {overlap:.4f}")
    
    # ===== 6. Comprehensive model =====
    print(f"\n6. Comprehensive Gap Prediction Model:")
    
    # Features: gamma*h_normalized*DW, h_PR, W_U_PR, offdiag, mlp_ratio
    X_features = np.column_stack([
        per_text_var_suppress,  # variance suppression signal
        np.ones(n_texts),       # intercept
    ])
    
    reg = Ridge(alpha=1.0).fit(X_features, gaps)
    r_ridge, _ = stats.pearsonr(reg.predict(X_features), gaps)
    
    print(f"   Ridge(gamma*h*DW only): r = {r_ridge:.4f}")
    
    results = {
        "model_name": model_name,
        "var_suppression_corr": float(r_var_suppress),
        "gamma_effectiveness": float(r_gamma_effect),
        "var_suppression_score": float(var_suppression_score),
        "mlp_ratio": float(mean_mlp_ratio),
        "gap_survival_score": float(gap_survival),
        "offdiag_fraction": float(offdiag_fraction),
        "h_PR": float(h_PR),
        "ln_efficiency": float(ln_efficiency),
        "W_U_PR": float(W_U_PR),
        "W_U_h_overlap": float(overlap),
        "unified_score_product": float(predicted_quality),
        "r_per_text": float(r_per_text),
        "r_ridge": float(r_ridge),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase CLIII: Language Math Theory")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["p672", "p673", "p674", "p675"])
    args = parser.parse_args()
    
    model_name = args.model
    experiment = args.experiment
    
    print(f"Loading model: {model_name}")
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"Model: {model_info.name}, d_model={model_info.d_model}, n_layers={model_info.n_layers}")
    
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'phase_cliii')
    os.makedirs(results_dir, exist_ok=True)
    
    start_time = time.time()
    
    if experiment == "p672":
        results = experiment_p672(model, tokenizer, device, model_name)
    elif experiment == "p673":
        results = experiment_p673(model, tokenizer, device, model_name)
    elif experiment == "p674":
        results = experiment_p674(model, tokenizer, device, model_name)
    elif experiment == "p675":
        results = experiment_p675(model, tokenizer, device, model_name)
    else:
        print(f"Unknown experiment: {experiment}")
        return
    
    elapsed = time.time() - start_time
    
    results["elapsed_seconds"] = elapsed
    results["model_name"] = model_name
    results["experiment"] = experiment
    
    # Save results
    result_file = os.path.join(results_dir, f"{experiment}_{model_name}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {result_file}")
    print(f"Elapsed: {elapsed:.1f}s")
    
    # Cleanup
    release_model(model)
    print("Model released.")


if __name__ == "__main__":
    main()
