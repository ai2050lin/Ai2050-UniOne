#!/usr/bin/env python3
"""
Phase CLII: Gamma Alignment Target & Language Unified Theory (P668-P671)
========================================================================

Based on Phase CLI findings:
1. gamma is necessary and sufficient for LN to extract gap (no gamma → r<0, gamma → r=0.28~0.82)
2. GLM4 gamma*normalized r=0.819, 3x better than Qwen3(0.279)
3. gamma does NOT align with per-dim SNR (corr=0.01~0.02)
4. Three models have three fundamentally different modes:
   - Qwen3: whitening-noise-rewhitening (MLP disrupts, final LN re-extracts)
   - GLM4: progressive enhancement (Attn creates, MLP enhances)
   - DS7B: all-on-final-LN (intermediate steps destroy gap)
5. Gap is interference residue: positive=121, negative=-121, net=-0.24 (ratio=-0.0016)

This phase addresses:
P668: Gamma Alignment Target — what does gamma actually align with?
P669: GLM4 Attn Gap Creation — why does GLM4's Attn create gap info?
P670: Interference Residue Model — mathematical conditions for gap as residue
P671: Unified Language Theory — can three modes be described by one equation?
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


def experiment_p668(model, tokenizer, device, model_name):
    """P668: Gamma Alignment Target.
    
    Key question: What does gamma align with?
    
    From P664: gamma does NOT align with per-dim SNR (corr=0.01~0.02)
    But gamma*normalized h predicts gap (r=0.28~0.82)
    
    Hypotheses:
    H1: gamma aligns with W_U row space structure (gamma ~ projection of W_U onto h dims)
    H2: gamma aligns with the "Delta_W manifold" direction in h space
    H3: gamma is learned to maximize gap extraction (optimization target)
    H4: gamma encodes frequency information (high-freq W_U rows get high gamma)
    
    Tests:
    1. Correlate gamma with W_U column norms (H1)
    2. Correlate gamma with mean(Delta_W) direction (H2)
    3. Correlate gamma with W_U row frequency (H4)
    4. Optimize gamma: what gamma maximizes gap(gamma*normalized_h, Delta_W)?
    5. Compare optimized gamma with actual gamma
    """
    print(f"\n{'='*60}")
    print(f"P668: Gamma Alignment Target ({model_name})")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    d_model = W_U.shape[1]
    n_vocab = W_U.shape[0]
    final_ln = get_final_ln(model)
    
    if final_ln is not None:
        gamma = final_ln.weight.detach().cpu().float().numpy()
    
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
    
    # Compute normalized h
    h_centered = h_before_arr - np.mean(h_before_arr, axis=1, keepdims=True)
    h_std = np.std(h_before_arr, axis=1, keepdims=True) + 1e-8
    h_normalized = h_centered / h_std
    
    # ===== 1. Gamma vs W_U column norms (H1) =====
    print(f"\n1. Gamma vs W_U Column Norms:")
    
    W_U_col_norms = np.linalg.norm(W_U, axis=0)  # [d_model]
    W_U_col_means = np.mean(W_U, axis=0)  # [d_model]
    W_U_col_std = np.std(W_U, axis=0)  # [d_model]
    
    r_col_norm, _ = stats.pearsonr(gamma, W_U_col_norms)
    r_col_mean, _ = stats.pearsonr(gamma, W_U_col_means)
    r_col_std, _ = stats.pearsonr(gamma, W_U_col_std)
    
    print(f"   corr(gamma, W_U_col_norms) = {r_col_norm:.4f}")
    print(f"   corr(gamma, W_U_col_means) = {r_col_mean:.4f}")
    print(f"   corr(gamma, W_U_col_std)   = {r_col_std:.4f}")
    
    # ===== 2. Gamma vs mean(Delta_W) direction (H2) =====
    print(f"\n2. Gamma vs Delta_W Direction:")
    
    mean_DW = np.mean(Delta_W_arr, axis=0)
    abs_mean_DW = np.abs(mean_DW)
    
    r_mean_DW, _ = stats.pearsonr(gamma, abs_mean_DW)
    r_signed_DW, _ = stats.pearsonr(gamma, mean_DW)
    
    # Also: gamma vs per-dim contribution to gap
    # gap_contrib_i = mean(h_normalized_i * Delta_W_i)
    gap_contribs = np.mean(h_normalized * Delta_W_arr, axis=0)  # [d_model]
    abs_gap_contribs = np.abs(gap_contribs)
    
    r_gap_contrib, _ = stats.pearsonr(gamma, abs_gap_contribs)
    
    print(f"   corr(gamma, |mean(Delta_W)|) = {r_mean_DW:.4f}")
    print(f"   corr(gamma, mean(Delta_W))   = {r_signed_DW:.4f}")
    print(f"   corr(gamma, |gap_contrib|)   = {r_gap_contrib:.4f}")
    
    # ===== 3. Gamma vs W_U row frequency (H4) =====
    print(f"\n3. Gamma vs W_U Row Frequency:")
    
    # Compute "frequency" of each W_U dimension: how many rows have high projection onto dim i
    # Approximate: project W_U rows onto dim i, compute fraction above threshold
    # More efficient: use PCA of W_U
    
    pca_W = PCA(n_components=min(50, d_model))
    pca_W.fit(W_U[:min(5000, n_vocab)])
    
    # Project gamma onto W_U PCs
    gamma_in_WU = pca_W.transform(gamma.reshape(1, -1))[0]  # [n_components]
    
    print(f"   gamma projection onto W_U PCs (top-5): {[f'{x:.3f}' for x in gamma_in_WU[:5]]}")
    
    # How much of gamma's variance is captured by W_U top-k PCs?
    gamma_reconstructed = pca_W.inverse_transform(gamma_in_WU.reshape(1, -1))[0]
    gamma_capture = 1 - np.sum((gamma - gamma_reconstructed)**2) / np.sum(gamma**2)
    print(f"   W_U top-50 PCs capture {gamma_capture*100:.1f}% of gamma variance")
    
    # ===== 4. Optimize gamma =====
    print(f"\n4. Optimize Gamma for Gap Extraction:")
    
    # Objective: find gamma that maximizes corr(gamma * h_normalized . Delta_W, logit_gap)
    # gap(gamma) = sum_i gamma_i * h_normalized_i * Delta_W_i
    # For each text t: gap_t = sum_i gamma_i * h_norm[t,i] * DW[t,i]
    
    # This is a linear regression problem:
    # X[t,i] = h_norm[t,i] * DW[t,i], y[t] = logit_gap[t]
    # Find gamma that minimizes ||X @ gamma - y||^2
    
    X_gap = h_normalized * Delta_W_arr  # [n_texts, d_model]
    y_gap = gaps
    
    # Ridge regression to find optimal gamma
    reg = Ridge(alpha=1.0)
    reg.fit(X_gap, y_gap)
    gamma_optimal = reg.coef_
    
    # Compute correlation with optimal gamma
    gap_optimal = X_gap @ gamma_optimal
    r_optimal, _ = stats.pearsonr(gap_optimal, y_gap)
    
    # Compute correlation with actual gamma
    gap_actual_gamma = X_gap @ gamma
    r_actual, _ = stats.pearsonr(gap_actual_gamma, y_gap)
    
    # Compute correlation between gamma and gamma_optimal
    r_gamma_opt, _ = stats.pearsonr(gamma, gamma_optimal)
    
    print(f"   Optimal gamma -> gap: r = {r_optimal:.4f}")
    print(f"   Actual gamma  -> gap: r = {r_actual:.4f}")
    print(f"   corr(actual_gamma, optimal_gamma) = {r_gamma_opt:.4f}")
    
    # ===== 5. Optimal gamma analysis =====
    print(f"\n5. Optimal Gamma Analysis:")
    
    # What does optimal gamma align with?
    r_opt_col_norm, _ = stats.pearsonr(gamma_optimal, W_U_col_norms)
    r_opt_gap_contrib, _ = stats.pearsonr(np.abs(gamma_optimal), abs_gap_contribs)
    r_opt_mean_DW, _ = stats.pearsonr(np.abs(gamma_optimal), abs_mean_DW)
    
    print(f"   corr(|optimal_gamma|, W_U_col_norms) = {r_opt_col_norm:.4f}")
    print(f"   corr(|optimal_gamma|, |gap_contrib|) = {r_opt_gap_contrib:.4f}")
    print(f"   corr(|optimal_gamma|, |mean(DW)|)    = {r_opt_mean_DW:.4f}")
    
    # ===== 6. Gamma decomposition =====
    print(f"\n6. Gamma Decomposition (signal vs noise):")
    
    # Split gamma into: component that aligns with gap_contrib (signal) and orthogonal (noise)
    gap_contrib_dir = abs_gap_contribs / (np.linalg.norm(abs_gap_contribs) + 1e-12)
    gamma_signal_proj = np.dot(gamma, gap_contrib_dir) * gap_contrib_dir
    gamma_noise = gamma - gamma_signal_proj
    
    gap_signal = X_gap @ gamma_signal_proj
    gap_noise = X_gap @ gamma_noise
    
    r_signal, _ = stats.pearsonr(gap_signal, y_gap)
    r_noise, _ = stats.pearsonr(gap_noise, y_gap)
    
    print(f"   gamma_signal (aligned with gap_contrib) -> gap: r = {r_signal:.4f}")
    print(f"   gamma_noise (orthogonal to gap_contrib)  -> gap: r = {r_noise:.4f}")
    print(f"   ||gamma_signal|| / ||gamma|| = {np.linalg.norm(gamma_signal_proj) / np.linalg.norm(gamma):.4f}")
    
    # ===== 7. Per-text gamma effectiveness =====
    print(f"\n7. Per-Text Gamma Effectiveness:")
    
    # For each text, compute: how much does gamma help vs no gamma?
    gap_no_gamma = np.sum(h_normalized * Delta_W_arr, axis=1)  # gamma=1
    gap_with_gamma = X_gap @ gamma
    
    r_no_gamma, _ = stats.pearsonr(gap_no_gamma, y_gap)
    r_with_gamma, _ = stats.pearsonr(gap_with_gamma, y_gap)
    
    print(f"   gap(no_gamma) -> logit_gap: r = {r_no_gamma:.4f}")
    print(f"   gap(with_gamma) -> logit_gap: r = {r_with_gamma:.4f}")
    print(f"   Gamma improvement: {r_with_gamma - r_no_gamma:.4f}")
    
    # Texts where gamma helps most vs least
    per_text_help = gap_with_gamma * np.sign(y_gap) - gap_no_gamma * np.sign(y_gap)
    top_help = np.argsort(per_text_help)[-5:]
    bottom_help = np.argsort(per_text_help)[:5]
    
    print(f"   Top-5 helped texts (indices): {top_help.tolist()}")
    print(f"   Bottom-5 helped texts (indices): {bottom_help.tolist()}")
    
    results = {
        "r_col_norm": float(r_col_norm),
        "r_col_mean": float(r_col_mean),
        "r_col_std": float(r_col_std),
        "r_mean_DW": float(r_mean_DW),
        "r_signed_DW": float(r_signed_DW),
        "r_gap_contrib": float(r_gap_contrib),
        "gamma_WU_capture_pct": float(gamma_capture * 100),
        "r_optimal": float(r_optimal),
        "r_actual": float(r_actual),
        "r_gamma_opt": float(r_gamma_opt),
        "r_opt_col_norm": float(r_opt_col_norm),
        "r_opt_gap_contrib": float(r_opt_gap_contrib),
        "r_opt_mean_DW": float(r_opt_mean_DW),
        "r_signal": float(r_signal),
        "r_noise": float(r_noise),
        "gamma_signal_fraction": float(np.linalg.norm(gamma_signal_proj) / np.linalg.norm(gamma)),
        "r_no_gamma": float(r_no_gamma),
        "r_with_gamma": float(r_with_gamma),
        "gamma_improvement": float(r_with_gamma - r_no_gamma),
    }
    
    return results


def experiment_p669(model, tokenizer, device, model_name):
    """P669: GLM4 Attn Gap Creation Mechanism.
    
    Key question: Why does GLM4's Attn create gap info (delta_r=+0.51)
    while Qwen3's Attn does not (delta_r=+0.06)?
    
    Hypotheses:
    H1: GLM4's attention heads directly compute Delta_W-related projections
    H2: GLM4's attention patterns are more gap-informative
    H3: GLM4's residual connection preserves gap info better
    H4: GLM4's input LN positions h better for gap extraction
    
    Tests:
    1. Compare attention head outputs: which heads contribute most to gap?
    2. Analyze residual vs attention contribution
    3. Test: if we remove attention, how much gap info is lost?
    4. Compare Qwen3 vs GLM4 attention patterns
    """
    print(f"\n{'='*60}")
    print(f"P669: Attn Gap Creation Mechanism ({model_name})")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    layers = get_layers(model)
    d_model = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 2560
    n_layers = len(layers)
    
    n_texts = 80
    test_texts = ALL_TEST_TEXTS[:n_texts]
    
    # Collect data at the last layer
    hook_outputs = {}
    
    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hook_outputs[name] = output[0].detach().cpu().float()
            else:
                hook_outputs[name] = output.detach().cpu().float()
        return hook_fn
    
    last_layer = layers[-1]
    
    # Register hooks
    hooks = []
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
    
    # Collect per-text data
    h_Lm1_list = []
    h_input_ln_list = []
    h_attn_list = []
    h_post_attn_ln_list = []
    h_mlp_list = []
    h_final_ln_list = []
    logit_gaps = []
    Delta_Ws = []
    
    for i, text in enumerate(test_texts):
        if (i+1) % 20 == 0:
            print(f"  Processing text {i+1}/{n_texts}...")
        
        hook_outputs.clear()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"].to(device),
                          attention_mask=inputs["attention_mask"].to(device),
                          output_hidden_states=True)
        
        h_Lm1 = outputs.hidden_states[-2][0, -1, :].cpu().float().numpy()
        h_L = outputs.hidden_states[-1][0, -1, :].cpu().float().numpy()
        
        h_Lm1_list.append(h_Lm1)
        
        # Extract hook outputs
        if 'input_ln' in hook_outputs:
            h_input_ln_list.append(hook_outputs['input_ln'][0, -1, :].numpy())
        else:
            h_input_ln_list.append(h_Lm1)
        
        if 'attn' in hook_outputs:
            h_attn_list.append(hook_outputs['attn'][0, -1, :].numpy())
        else:
            h_attn_list.append(np.zeros(d_model))
        
        if 'post_attn_ln' in hook_outputs:
            h_post_attn_ln_list.append(hook_outputs['post_attn_ln'][0, -1, :].numpy())
        else:
            h_post_attn_ln_list.append(h_Lm1)
        
        if 'mlp' in hook_outputs:
            h_mlp_list.append(hook_outputs['mlp'][0, -1, :].numpy())
        else:
            h_mlp_list.append(np.zeros(d_model))
        
        h_final_ln_list.append(h_L)
        
        logits = outputs.logits[0, -1, :].cpu().float().numpy()
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        logit_gaps.append(logits[top1_idx] - logits[top2_idx])
        Delta_Ws.append(W_U[top1_idx] - W_U[top2_idx])
    
    for h in hooks:
        h.remove()
    
    gaps = np.array(logit_gaps)
    Delta_W_arr = np.array(Delta_Ws)
    
    h_Lm1_arr = np.array(h_Lm1_list)
    h_input_ln_arr = np.array(h_input_ln_list)
    h_attn_arr = np.array(h_attn_list)
    h_post_ln_arr = np.array(h_post_attn_ln_list)
    h_mlp_arr = np.array(h_mlp_list)
    h_final_arr = np.array(h_final_ln_list)
    
    # ===== 1. Residual decomposition =====
    print(f"\n1. Residual Decomposition:")
    
    # h_after_attn = h_input_ln + attn_output (residual connection)
    # gap(h_after_attn, Delta_W) = gap(h_input_ln, Delta_W) + gap(attn_output, Delta_W)
    
    gap_input_ln = np.sum(h_input_ln_arr * Delta_W_arr, axis=1)
    gap_attn_out = np.sum(h_attn_arr * Delta_W_arr, axis=1)
    gap_after_attn = gap_input_ln + gap_attn_out  # residual
    
    gap_post_ln = np.sum(h_post_ln_arr * Delta_W_arr, axis=1)
    gap_mlp_out = np.sum(h_mlp_arr * Delta_W_arr, axis=1)
    
    r_input_ln, _ = stats.pearsonr(gap_input_ln, gaps)
    r_attn_out, _ = stats.pearsonr(gap_attn_out, gaps)
    r_after_attn, _ = stats.pearsonr(gap_after_attn, gaps)
    r_post_ln, _ = stats.pearsonr(gap_post_ln, gaps)
    r_mlp_out, _ = stats.pearsonr(gap_mlp_out, gaps)
    
    print(f"   gap(input_ln, DW) -> logit_gap:   r = {r_input_ln:.4f}")
    print(f"   gap(attn_output, DW) -> logit_gap: r = {r_attn_out:.4f}")
    print(f"   gap(after_attn, DW) -> logit_gap:  r = {r_after_attn:.4f}")
    print(f"   gap(post_attn_LN, DW) -> logit_gap: r = {r_post_ln:.4f}")
    print(f"   gap(mlp_output, DW) -> logit_gap:   r = {r_mlp_out:.4f}")
    
    # ===== 2. Attention output analysis =====
    print(f"\n2. Attention Output Analysis:")
    
    # How much of the attn output is in the Delta_W direction?
    # cos(attn_out, Delta_W) per text
    cos_attn_DW = []
    for t in range(n_texts):
        attn_norm = h_attn_arr[t] / (np.linalg.norm(h_attn_arr[t]) + 1e-12)
        dw_norm = Delta_W_arr[t] / (np.linalg.norm(Delta_W_arr[t]) + 1e-12)
        cos_attn_DW.append(np.dot(attn_norm, dw_norm))
    
    print(f"   cos(attn_output, Delta_W): mean={np.mean(cos_attn_DW):.4f}, std={np.std(cos_attn_DW):.4f}")
    
    # ===== 3. Norm analysis =====
    print(f"\n3. Norm Analysis:")
    
    norm_Lm1 = np.mean(np.linalg.norm(h_Lm1_arr, axis=1))
    norm_input_ln = np.mean(np.linalg.norm(h_input_ln_arr, axis=1))
    norm_attn = np.mean(np.linalg.norm(h_attn_arr, axis=1))
    norm_post_ln = np.mean(np.linalg.norm(h_post_ln_arr, axis=1))
    norm_mlp = np.mean(np.linalg.norm(h_mlp_arr, axis=1))
    
    print(f"   ||h_Lm1||:      {norm_Lm1:.1f}")
    print(f"   ||input_ln||:   {norm_input_ln:.1f}")
    print(f"   ||attn_out||:   {norm_attn:.1f}")
    print(f"   ||post_attn_LN||: {norm_post_ln:.1f}")
    print(f"   ||mlp_out||:    {norm_mlp:.1f}")
    
    print(f"   ||attn_out|| / ||input_ln||: {norm_attn/norm_input_ln:.3f}")
    print(f"   ||mlp_out|| / ||post_ln||:   {norm_mlp/norm_post_ln:.3f}")
    
    # ===== 4. Attn as "gap amplifier" =====
    print(f"\n4. Attn as Gap Amplifier:")
    
    # If Attn creates gap info, then: gap(attn_output) should correlate with gap
    # And the correlation should be positive
    
    # Also check: is the attn output orthogonal to h_input_ln?
    cos_input_attn = []
    for t in range(n_texts):
        input_norm = h_input_ln_arr[t] / (np.linalg.norm(h_input_ln_arr[t]) + 1e-12)
        attn_norm = h_attn_arr[t] / (np.linalg.norm(h_attn_arr[t]) + 1e-12)
        cos_input_attn.append(np.dot(input_norm, attn_norm))
    
    print(f"   cos(input_ln, attn_out): mean={np.mean(cos_input_attn):.4f}")
    
    # ===== 5. Layer-specific: also check penultimate layer =====
    print(f"\n5. Layer-Level Gap Correlation:")
    
    # Check all layers for gap correlation
    layer_gap_r = []
    # Sample a few layers for efficiency
    layer_indices = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]
    
    for layer_idx in layer_indices:
        r_list = []
        for text in test_texts[:30]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(input_ids=inputs["input_ids"].to(device),
                              attention_mask=inputs["attention_mask"].to(device),
                              output_hidden_states=True)
            
            h = outputs.hidden_states[layer_idx][0, -1, :].cpu().float().numpy()
            logits = outputs.logits[0, -1, :].cpu().float().numpy()
            top1 = np.argmax(logits)
            top2 = np.argsort(logits)[-2]
            gap = logits[top1] - logits[top2]
            dw = W_U[top1] - W_U[top2]
            
            r_list.append((np.dot(h, dw), gap))
        
        r_vals = np.array(r_list)
        r, _ = stats.pearsonr(r_vals[:, 0], r_vals[:, 1])
        layer_gap_r.append(r)
        print(f"   Layer {layer_idx:3d}: gap(h, DW) -> logit_gap r = {r:.4f}")
    
    results = {
        "r_input_ln": float(r_input_ln),
        "r_attn_out": float(r_attn_out),
        "r_after_attn": float(r_after_attn),
        "r_post_ln": float(r_post_ln),
        "r_mlp_out": float(r_mlp_out),
        "cos_attn_DW_mean": float(np.mean(cos_attn_DW)),
        "norm_Lm1": float(norm_Lm1),
        "norm_input_ln": float(norm_input_ln),
        "norm_attn": float(norm_attn),
        "norm_post_ln": float(norm_post_ln),
        "norm_mlp": float(norm_mlp),
        "attn_to_input_ratio": float(norm_attn/norm_input_ln),
        "mlp_to_postln_ratio": float(norm_mlp/norm_post_ln),
        "cos_input_attn_mean": float(np.mean(cos_input_attn)),
        "layer_gap_r": {str(idx): float(r) for idx, r in zip(layer_indices, layer_gap_r)},
    }
    
    return results


def experiment_p670(model, tokenizer, device, model_name):
    """P670: Interference Residue Model.
    
    Key question: Under what conditions is gap an interference residue?
    
    From P665: positive=121, negative=-121, net=-0.24 (ratio=-0.0016)
    
    Theory:
    gap = h . Delta_W = sum_i h_i * Delta_W_i
    
    If h_i and Delta_W_i are approximately random with mean 0:
    E[gap] = 0, Var[gap] = sum_i Var[h_i] * Var[Delta_W_i]
    
    But we observe gap correlates with logit_gap (r~1 after LN).
    This means: despite being a small residue, gap contains information.
    
    Mathematical condition for "interference residue that carries information":
    gap = signal + noise, where signal << |noise| but signal is systematic
    
    Tests:
    1. Is gap_before_LN truly random (Gaussian)?
    2. After sorting by gap magnitude, is there systematic structure?
    3. How much of gap_before_LN variance is explained by logit_gap?
    4. Constructive vs destructive interference patterns
    """
    print(f"\n{'='*60}")
    print(f"P670: Interference Residue Model ({model_name})")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    d_model = W_U.shape[1]
    
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
    
    # ===== 1. Gap distribution analysis =====
    print(f"\n1. Gap Distribution Analysis:")
    
    gap_before = np.sum(h_before_arr * Delta_W_arr, axis=1)
    gap_after = np.sum(h_after_arr * Delta_W_arr, axis=1)
    
    # Normality test
    from scipy.stats import normaltest
    stat_before, p_before = normaltest(gap_before)
    stat_after, p_after = normaltest(gap_after)
    
    print(f"   gap_before: mean={np.mean(gap_before):.2f}, std={np.std(gap_before):.2f}")
    print(f"   gap_before: normality p={p_before:.4f} ({'Gaussian' if p_before > 0.05 else 'non-Gaussian'})")
    print(f"   gap_after:  mean={np.mean(gap_after):.2f}, std={np.std(gap_after):.2f}")
    print(f"   gap_after:  normality p={p_after:.4f} ({'Gaussian' if p_after > 0.05 else 'non-Gaussian'})")
    
    # ===== 2. Interference decomposition =====
    print(f"\n2. Interference Decomposition:")
    
    # gap = positive + negative
    contribs = h_before_arr * Delta_W_arr  # [n_texts, d_model]
    positive = np.sum(contribs * (contribs > 0), axis=1)
    negative = np.sum(contribs * (contribs < 0), axis=1)
    total_abs = np.sum(np.abs(contribs), axis=1)
    
    print(f"   Mean positive: {np.mean(positive):.2f}")
    print(f"   Mean negative: {np.mean(negative):.2f}")
    print(f"   Mean |total|: {np.mean(total_abs):.2f}")
    print(f"   Net/Total ratio: {np.mean((positive + negative) / (total_abs + 1e-12)):.4f}")
    
    # ===== 3. Systematic vs random component =====
    print(f"\n3. Systematic vs Random Component:")
    
    # If gap = signal + noise, where signal = f(logit_gap),
    # then: Var(gap) = Var(signal) + Var(noise)
    # signal variance = r^2 * Var(gap), noise = (1-r^2) * Var(gap)
    
    r_before, _ = stats.pearsonr(gap_before, gaps)
    r_after, _ = stats.pearsonr(gap_after, gaps)
    
    signal_var_before = r_before**2 * np.var(gap_before)
    noise_var_before = (1 - r_before**2) * np.var(gap_before)
    
    signal_var_after = r_after**2 * np.var(gap_after)
    noise_var_after = (1 - r_after**2) * np.var(gap_after)
    
    print(f"   Before LN: signal_var={signal_var_before:.2f}, noise_var={noise_var_before:.2f}")
    print(f"   Before LN: SNR = {signal_var_before/noise_var_before:.6f}")
    print(f"   After LN:  signal_var={signal_var_after:.2f}, noise_var={noise_var_after:.2f}")
    print(f"   After LN:  SNR = {signal_var_after/noise_var_after:.6f}")
    
    # ===== 4. Per-dimension interference =====
    print(f"\n4. Per-Dimension Interference Pattern:")
    
    # For each dimension, compute: is it consistently constructive or destructive?
    # I.e., is h_i * Delta_W_i consistently positive or negative across texts?
    
    sign_consistency = np.mean(np.sign(contribs) == np.sign(np.mean(contribs, axis=0)), axis=0)
    
    print(f"   Mean sign consistency: {np.mean(sign_consistency):.4f}")
    print(f"   Dims with consistency > 0.7: {np.sum(sign_consistency > 0.7)}/{d_model}")
    print(f"   Dims with consistency > 0.9: {np.sum(sign_consistency > 0.9)}/{d_model}")
    
    # Are consistent dimensions more correlated with gap?
    dim_corr = np.zeros(d_model)
    for i in range(d_model):
        r, _ = stats.pearsonr(h_before_arr[:, i] * Delta_W_arr[:, i], gaps)
        dim_corr[i] = r
    
    r_consist_corr, _ = stats.pearsonr(sign_consistency, np.abs(dim_corr))
    print(f"   corr(sign_consistency, |dim_corr|) = {r_consist_corr:.4f}")
    
    # ===== 5. Central limit theorem check =====
    print(f"\n5. Central Limit Theorem Check:")
    
    # If h_i * Delta_W_i are iid, gap should be Gaussian (CLT)
    # But if they're correlated, gap may be non-Gaussian
    
    # Per-dimension correlations
    dim_cross_corr = np.corrcoef(contribs.T)  # [d_model, d_model]
    # Mean absolute off-diagonal correlation
    mask = ~np.eye(d_model, dtype=bool)
    mean_cross_corr = np.mean(np.abs(dim_cross_corr[mask]))
    
    print(f"   Mean |cross-dim correlation|: {mean_cross_corr:.4f}")
    print(f"   Max |cross-dim correlation|: {np.max(np.abs(dim_cross_corr[mask])):.4f}")
    
    # ===== 6. Gap as sum of random variables =====
    print(f"\n6. Gap as Sum of Random Variables:")
    
    # If gap = sum_i X_i where X_i = h_i * Delta_W_i
    # E[gap] = sum E[X_i], Var[gap] = sum Var[X_i] + 2*sum_{i<j} Cov[X_i, X_j]
    
    E_gap = np.sum(np.mean(contribs, axis=0))
    Var_indep = np.sum(np.var(contribs, axis=0))
    Var_actual = np.var(gap_before)
    
    # Covariance contribution
    Var_cov = Var_actual - Var_indep
    
    print(f"   E[gap] = {E_gap:.2f}")
    print(f"   Var(gap) if independent: {Var_indep:.2f}")
    print(f"   Var(gap) actual: {Var_actual:.2f}")
    print(f"   Covariance contribution: {Var_cov:.2f} ({Var_cov/Var_actual*100:.1f}% of total)")
    
    # ===== 7. After LN interference =====
    print(f"\n7. After LN Interference:")
    
    contribs_after = h_after_arr * Delta_W_arr
    positive_after = np.sum(contribs_after * (contribs_after > 0), axis=1)
    negative_after = np.sum(contribs_after * (contribs_after < 0), axis=1)
    total_abs_after = np.sum(np.abs(contribs_after), axis=1)
    
    print(f"   Mean positive (after LN): {np.mean(positive_after):.2f}")
    print(f"   Mean negative (after LN): {np.mean(negative_after):.2f}")
    print(f"   Net/Total ratio (after LN): {np.mean((positive_after + negative_after) / (total_abs_after + 1e-12)):.4f}")
    
    # Change in interference pattern
    ratio_before = np.mean((positive + negative) / (total_abs + 1e-12))
    ratio_after = np.mean((positive_after + negative_after) / (total_abs_after + 1e-12))
    print(f"   Net/Total ratio change: {ratio_before:.4f} -> {ratio_after:.4f}")
    
    results = {
        "gap_before_mean": float(np.mean(gap_before)),
        "gap_before_std": float(np.std(gap_before)),
        "gap_before_normal_p": float(p_before),
        "gap_after_mean": float(np.mean(gap_after)),
        "gap_after_std": float(np.std(gap_after)),
        "r_before": float(r_before),
        "r_after": float(r_after),
        "snr_before": float(signal_var_before/noise_var_before),
        "snr_after": float(signal_var_after/noise_var_after),
        "positive_before": float(np.mean(positive)),
        "negative_before": float(np.mean(negative)),
        "net_total_ratio_before": float(np.mean((positive + negative) / (total_abs + 1e-12))),
        "net_total_ratio_after": float(np.mean((positive_after + negative_after) / (total_abs_after + 1e-12))),
        "sign_consistency_mean": float(np.mean(sign_consistency)),
        "r_consist_corr": float(r_consist_corr),
        "mean_cross_corr": float(mean_cross_corr),
        "E_gap": float(E_gap),
        "Var_indep": float(Var_indep),
        "Var_actual": float(Var_actual),
        "cov_pct": float(Var_cov/Var_actual*100),
    }
    
    return results


def experiment_p671(model, tokenizer, device, model_name):
    """P671: Unified Language Theory.
    
    Key question: Can three modes (Qwen3/GLM4/DS7B) be described by one equation?
    
    Proposed unified equation:
    gap_after = alpha * LN(gap_before + noise_attn + noise_mlp)
    
    Where:
    - alpha depends on gamma effectiveness (P664: 0.28~0.82)
    - noise_attn depends on Attn's gap contribution (P666: -0.08~+0.51)
    - noise_mlp depends on MLP's gap contribution (P666: -0.37~+0.12)
    
    Three regimes:
    1. alpha low, noise_mlp negative → "struggle" mode (Qwen3)
    2. alpha high, noise_attn positive → "elegant" mode (GLM4)
    3. alpha low, noise_attn/ln negative → "brute force" mode (DS7B)
    
    Tests:
    1. Fit unified model parameters for each model
    2. Cross-validate: can we predict one model's behavior from another's?
    3. Parameter sensitivity analysis
    """
    print(f"\n{'='*60}")
    print(f"P671: Unified Language Theory ({model_name})")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    d_model = W_U.shape[1]
    final_ln = get_final_ln(model)
    
    if final_ln is not None:
        gamma = final_ln.weight.detach().cpu().float().numpy()
    
    n_texts = 100
    test_texts = ALL_TEST_TEXTS[:n_texts]
    
    # ===== 1. Gamma effectiveness =====
    print(f"\n1. Gamma Effectiveness (alpha):")
    
    # alpha = corr(gamma*normalized_h . Delta_W, logit_gap)
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
    h_scaled = h_normalized * gamma[np.newaxis, :]
    
    gap_no_gamma = np.sum(h_normalized * Delta_W_arr, axis=1)
    gap_with_gamma = np.sum(h_scaled * Delta_W_arr, axis=1)
    
    alpha_no_gamma, _ = stats.pearsonr(gap_no_gamma, gaps)
    alpha_with_gamma, _ = stats.pearsonr(gap_with_gamma, gaps)
    
    print(f"   alpha (no gamma):   {alpha_no_gamma:.4f}")
    print(f"   alpha (with gamma): {alpha_with_gamma:.4f}")
    print(f"   Gamma amplification: {alpha_with_gamma / (alpha_no_gamma + 1e-12):.2f}x")
    
    # ===== 2. W_U structure parameters =====
    print(f"\n2. W_U Structure Parameters:")
    
    pca_W = PCA(n_components=min(50, d_model))
    pca_W.fit(W_U[:min(5000, W_U.shape[0])])
    W_PR = (np.sum(pca_W.explained_variance_ratio_))**2 / np.sum(pca_W.explained_variance_ratio_**2)
    
    pca_h = PCA(n_components=min(50, d_model))
    pca_h.fit(h_before_arr)
    h_PR = (np.sum(pca_h.explained_variance_ratio_))**2 / np.sum(pca_h.explained_variance_ratio_**2)
    
    W_top10 = np.sum(pca_W.explained_variance_ratio_[:10]) * 100
    h_top10 = np.sum(pca_h.explained_variance_ratio_[:10]) * 100
    
    print(f"   W_U PR: {W_PR:.1f}, top-10: {W_top10:.1f}%")
    print(f"   h PR: {h_PR:.1f}, top-10: {h_top10:.1f}%")
    
    # ===== 3. Unified model parameters =====
    print(f"\n3. Unified Model Parameters:")
    
    # Three key parameters:
    # 1. gamma_effectiveness: how well gamma extracts gap (0.28~0.82)
    # 2. W_U_concentration: how concentrated W_U is (PR 44~89)
    # 3. h_dimensionality: how many dims h uses (PR 1.8~28)
    
    params = {
        "gamma_effectiveness": float(alpha_with_gamma),
        "W_U_PR": float(W_PR),
        "h_PR": float(h_PR),
        "W_U_top10": float(W_top10),
        "h_top10": float(h_top10),
        "model_name": model_name,
    }
    
    for k, v in params.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")
    
    # ===== 4. Model regime classification =====
    print(f"\n4. Model Regime Classification:")
    
    if alpha_with_gamma > 0.6:
        regime = "elegant (high gamma effectiveness)"
    elif h_PR < 5:
        regime = "brute_force (low h dimensionality)"
    else:
        regime = "struggle (moderate gamma effectiveness)"
    
    print(f"   Regime: {regime}")
    
    # ===== 5. Predictive model =====
    print(f"\n5. Predictive Model (from P666/P667 data):")
    
    # From P666: sub-step delta_r values
    # We need to re-compute or use cached data
    # For now, compute from h_before and h_after
    
    gap_before = np.sum(h_before_arr * Delta_W_arr, axis=1)
    r_before, _ = stats.pearsonr(gap_before, gaps)
    
    gap_after = np.sum(h_before_arr * Delta_W_arr, axis=1)  # Will be replaced with actual after-LN
    h_after_list = []
    for i, text in enumerate(test_texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"].to(device),
                          attention_mask=inputs["attention_mask"].to(device),
                          output_hidden_states=True)
        h_after_list.append(outputs.hidden_states[-1][0, -1, :].cpu().float().numpy())
    
    h_after_arr = np.array(h_after_list)
    gap_after = np.sum(h_after_arr * Delta_W_arr, axis=1)
    r_after, _ = stats.pearsonr(gap_after, gaps)
    
    # LN amplification factor
    ln_amplification = r_after / (abs(r_before) + 1e-12)
    
    print(f"   r_before_LN: {r_before:.4f}")
    print(f"   r_after_LN: {r_after:.4f}")
    print(f"   LN amplification: {ln_amplification:.1f}x")
    
    # ===== 6. Gap prediction quality =====
    print(f"\n6. Gap Prediction Quality:")
    
    # Use unified model: gap_predicted = gamma_effectiveness * gap_before_LN
    # Where gap_before_LN = sum(h_normalized * Delta_W)
    
    # Direct prediction
    gap_pred_direct = gap_with_gamma
    r_pred_direct, _ = stats.pearsonr(gap_pred_direct, gaps)
    
    # Optimal prediction (Ridge on X_gap features)
    X_gap = h_normalized * Delta_W_arr
    loo = LeaveOneOut()
    preds_loo = np.zeros(n_texts)
    for train_idx, test_idx in loo.split(X_gap):
        reg = Ridge(alpha=1.0)
        reg.fit(X_gap[train_idx], gaps[train_idx])
        preds_loo[test_idx[0]] = reg.predict(X_gap[test_idx])[0]
    
    r_pred_ridge, _ = stats.pearsonr(preds_loo, gaps)
    
    print(f"   Direct prediction (gamma*normalized): r = {r_pred_direct:.4f}")
    print(f"   Ridge LOO prediction: r = {r_pred_ridge:.4f}")
    print(f"   Gap between direct and Ridge: {r_pred_ridge - r_pred_direct:.4f}")
    
    params.update({
        "regime": regime,
        "r_before_LN": float(r_before),
        "r_after_LN": float(r_after),
        "ln_amplification": float(ln_amplification),
        "r_pred_direct": float(r_pred_direct),
        "r_pred_ridge": float(r_pred_ridge),
    })
    
    return params


def main():
    parser = argparse.ArgumentParser(description="Phase CLII: Gamma Alignment & Unified Theory")
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True, choices=["p668", "p669", "p670", "p671"])
    args = parser.parse_args()
    
    model_name = args.model
    experiment = args.experiment
    
    print(f"Loading model: {model_name}")
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"Model: {model_info.name}, d_model={model_info.d_model}, n_layers={model_info.n_layers}")
    
    start_time = time.time()
    
    if experiment == "p668":
        results = experiment_p668(model, tokenizer, device, model_name)
    elif experiment == "p669":
        results = experiment_p669(model, tokenizer, device, model_name)
    elif experiment == "p670":
        results = experiment_p670(model, tokenizer, device, model_name)
    elif experiment == "p671":
        results = experiment_p671(model, tokenizer, device, model_name)
    
    elapsed = time.time() - start_time
    
    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "phase_clii")
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
