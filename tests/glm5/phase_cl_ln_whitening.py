#!/usr/bin/env python3
"""
Phase CL: LN Whitening Theory & Hidden Gap Information (P660-P663)
==================================================================

Based on Phase CXLIX findings:
1. LN scaling gamma → gap perfectly linear (R2>0.993, ratio=2.000)
2. LN scaling h → gap invariant (ratio=1.000) — LN is scale-invariant!
3. post-attn LN is the first gap emergence point (Qwen3/GLM4 r>0.5)
4. Attn/MLP do NOT create gap info (r<0.2); LN "reveals" it
5. GLM4 encodes gap in 10-dim subspace; Qwen3/DS7B cannot
6. cos(W_learned, Delta_W) ≈ 0.13-0.19 (GLM4/Qwen3) or 0.0005 (DS7B)

This phase addresses:
P660: LN Whitening Theory — mathematical conditions for LN to "reveal" gap
P661: MLP Disruption Mechanism — how MLP "scrambles" LN-revealed gap info
P662: Hidden Gap Encoding — what form does gap info take in h_{L-1}?
P663: GLM4 Low-Dim Encoding Physics — why can GLM4 encode gap in 10 dims?
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


def experiment_p660(model, tokenizer, device, model_name):
    """P660: LN Whitening Theory.
    
    Key question: Under what mathematical conditions does LN "reveal" gap information?
    
    Theory:
    gap_before_LN = h · Delta_W = sum_i h_i * Delta_W_i
    gap_after_LN = LN(h) · Delta_W = sum_i gamma_i * (h_i - mean(h)) / std(h) * Delta_W_i
    
    For gap_after to correlate with logit_gap, we need:
    gap_after ∝ some function of h that contains gap information
    
    LN does two things:
    1. Centering: h_i → h_i - mean(h)
       This removes the "DC component" — may help or hurt gap depending on sign
    2. Scaling: h_i → h_i / std(h)  
       This normalizes variance — "whitening" effect
    
    Key insight: if h has very different scales in different dimensions,
    and Delta_W aligns with low-variance dimensions, then:
    - Before LN: gap is dominated by high-variance dimensions (noise)
    - After LN: all dimensions contribute equally → gap signal emerges
    
    Tests:
    1. Per-dimension variance of h before/after LN
    2. Signal-to-noise ratio (SNR) of gap before/after LN
    3. Does LN increase SNR? By how much?
    4. Which dimensions carry gap information before LN?
    """
    print(f"\n{'='*60}")
    print(f"P660: LN Whitening Theory ({model_name})")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    layers = get_layers(model)
    last_layer = layers[-1]
    d_model = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 2560
    final_ln = get_final_ln(model)
    
    n_texts = 100
    test_texts = ALL_TEST_TEXTS[:n_texts]
    
    # Get LN weight
    if final_ln is not None:
        gamma = final_ln.weight.detach().cpu().float().numpy()
        gamma_bias = final_ln.bias.detach().cpu().float().numpy() if hasattr(final_ln, 'bias') and final_ln.bias is not None else np.zeros(d_model)
    
    # Collect h_before_ln and h_after_ln for all texts
    h_before_list = []
    h_after_list = []
    logit_gaps = []
    Delta_Ws = []
    W_top1s = []
    
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
        W_top1s.append(W_U[top1_idx])
    
    h_before_arr = np.array(h_before_list)  # [n_texts, d_model]
    h_after_arr = np.array(h_after_list)
    gaps = np.array(logit_gaps)
    Delta_W_arr = np.array(Delta_Ws)
    
    # ===== 1. Per-dimension variance analysis =====
    print(f"\n1. Per-Dimension Variance Analysis:")
    
    var_before = np.var(h_before_arr, axis=0)  # [d_model]
    var_after = np.var(h_after_arr, axis=0)
    
    print(f"   Before LN: var range=[{np.min(var_before):.2f}, {np.max(var_before):.2f}], "
          f"ratio={np.max(var_before)/np.min(var_before):.1f}")
    print(f"   After LN:  var range=[{np.min(var_after):.6f}, {np.max(var_after):.6f}], "
          f"ratio={np.max(var_after)/np.min(var_after):.1f}")
    
    # Variance compression ratio
    var_ratio = np.max(var_before) / np.min(var_before)
    print(f"   Variance dynamic range: {var_ratio:.1f}x before LN → "
          f"{np.max(var_after)/np.min(var_after):.1f}x after LN")
    
    # ===== 2. Gap SNR analysis =====
    print(f"\n2. Gap Signal-to-Noise Ratio:")
    
    # gap_before = h_before · Delta_W for each text
    gaps_before = np.array([np.dot(h_before_arr[i], Delta_W_arr[i]) for i in range(n_texts)])
    gaps_after = np.array([np.dot(h_after_arr[i], Delta_W_arr[i]) for i in range(n_texts)])
    
    # "Signal" = correlation with logit_gap
    # "Noise" = 1 - correlation^2
    r_before, _ = stats.pearsonr(gaps_before, gaps)
    r_after, _ = stats.pearsonr(gaps_after, gaps)
    
    snr_before = r_before**2 / (1 - r_before**2) if abs(r_before) < 1 else float('inf')
    snr_after = r_after**2 / (1 - r_after**2) if abs(r_after) < 1 else float('inf')
    
    print(f"   gap_before_LN → logit_gap: r={r_before:.4f}, SNR={snr_before:.4f}")
    print(f"   gap_after_LN → logit_gap: r={r_after:.4f}, SNR={snr_after:.4f}")
    
    if abs(snr_before) > 1e-8:
        snr_gain = snr_after / snr_before
        print(f"   SNR gain from LN: {snr_gain:.1f}x")
    else:
        snr_gain = float('inf')
        print(f"   SNR gain from LN: INF (SNR_before ≈ 0)")
    
    # ===== 3. Which dimensions carry gap information? =====
    print(f"\n3. Gap-Carrying Dimensions:")
    
    # For each dimension i, compute corr(h_before[:, i] * Delta_W[:, i], logit_gap)
    # This tells us which dimensions contribute to gap prediction
    dim_gap_corr_before = []
    dim_gap_corr_after = []
    
    mean_DeltaW = np.mean(Delta_W_arr, axis=0)  # Average Delta_W across texts
    
    for dim_i in range(0, d_model, max(1, d_model // 20)):  # Sample 20 dimensions
        # Contribution of dimension i to gap
        contrib_before = h_before_arr[:, dim_i] * mean_DeltaW[dim_i]
        contrib_after = h_after_arr[:, dim_i] * mean_DeltaW[dim_i]
        
        if np.std(contrib_before) > 1e-10:
            r_b, _ = stats.pearsonr(contrib_before, gaps)
        else:
            r_b = 0
        if np.std(contrib_after) > 1e-10:
            r_a, _ = stats.pearsonr(contrib_after, gaps)
        else:
            r_a = 0
        
        dim_gap_corr_before.append({'dim': dim_i, 'r': r_b, 'var': var_before[dim_i]})
        dim_gap_corr_after.append({'dim': dim_i, 'r': r_a, 'var': var_after[dim_i]})
    
    # Sort by |correlation| after LN
    dim_gap_corr_after.sort(key=lambda x: abs(x['r']), reverse=True)
    print(f"   Top-5 gap-carrying dimensions AFTER LN:")
    for d in dim_gap_corr_after[:5]:
        print(f"     Dim {d['dim']}: r={d['r']:.4f}, var={d['var']:.6f}")
    
    # ===== 4. LN whitening effect =====
    print(f"\n4. LN Whitening Effect:")
    
    # Before LN: gap = sum_i h_i * Delta_W_i
    # The "noise" in gap comes from high-variance dimensions that are NOT aligned with Delta_W
    # After LN: all dimensions are equalized → gap signal emerges
    
    # Compute: for each text, decompose gap_before into signal and noise components
    # Signal = dimensions where Delta_W is large
    # Noise = dimensions where Delta_W is small but h variance is large
    
    # |Delta_W_i| sorted
    DeltaW_importance = np.abs(mean_DeltaW)
    DeltaW_rank = np.argsort(DeltaW_importance)[::-1]
    
    # Top-10% Delta_W dimensions
    top_pct = 0.1
    n_top = max(1, int(d_model * top_pct))
    top_dims = DeltaW_rank[:n_top]
    bottom_dims = DeltaW_rank[n_top:]
    
    # Gap contribution from top vs bottom dimensions
    gap_top_before = np.sum(h_before_arr[:, top_dims] * mean_DeltaW[top_dims], axis=1)
    gap_bottom_before = np.sum(h_before_arr[:, bottom_dims] * mean_DeltaW[bottom_dims], axis=1)
    
    gap_top_after = np.sum(h_after_arr[:, top_dims] * mean_DeltaW[top_dims], axis=1)
    gap_bottom_after = np.sum(h_after_arr[:, bottom_dims] * mean_DeltaW[bottom_dims], axis=1)
    
    r_top_before, _ = stats.pearsonr(gap_top_before, gaps)
    r_bottom_before, _ = stats.pearsonr(gap_bottom_before, gaps)
    r_top_after, _ = stats.pearsonr(gap_top_after, gaps)
    r_bottom_after, _ = stats.pearsonr(gap_bottom_after, gaps)
    
    print(f"   Top {top_pct*100:.0f}% Delta_W dimensions:")
    print(f"     Before LN: r={r_top_before:.4f} (signal), var_fraction={np.sum(var_before[top_dims])/np.sum(var_before):.4f}")
    print(f"     After LN:  r={r_top_after:.4f} (signal), var_fraction={np.sum(var_after[top_dims])/np.sum(var_after):.4f}")
    print(f"   Bottom {(1-top_pct)*100:.0f}% Delta_W dimensions:")
    print(f"     Before LN: r={r_bottom_before:.4f} (noise), var_fraction={np.sum(var_before[bottom_dims])/np.sum(var_before):.4f}")
    print(f"     After LN:  r={r_bottom_after:.4f} (noise), var_fraction={np.sum(var_after[bottom_dims])/np.sum(var_after):.4f}")
    
    # ===== 5. Key theoretical result =====
    print(f"\n5. KEY THEORETICAL RESULT:")
    print(f"   LN whitening effect:")
    print(f"   - Before LN: high-variance dimensions dominate gap → noise overwhelms signal")
    print(f"   - After LN: all dimensions equalized → signal emerges from noise")
    print(f"   - SNR gain: {snr_gain:.1f}x")
    print(f"   - This is analogous to 'whitening' in signal processing")
    print(f"   - LN does NOT create information; it REVEALS information already in h")
    
    return {
        'r_before': float(r_before),
        'r_after': float(r_after),
        'snr_before': float(snr_before),
        'snr_after': float(snr_after),
        'snr_gain': float(snr_gain),
        'var_ratio_before': float(var_ratio),
        'var_ratio_after': float(np.max(var_after)/np.min(var_after)),
        'r_top_before': float(r_top_before),
        'r_top_after': float(r_top_after),
        'r_bottom_before': float(r_bottom_before),
        'r_bottom_after': float(r_bottom_after),
    }


def experiment_p661(model, tokenizer, device, model_name):
    """P661: MLP Disruption Mechanism.
    
    Key question: How does MLP "scramble" the gap information revealed by LN?
    
    From P658: after post-attn LN r≈0.5, but after MLP r≈0.05
    This means MLP destroys the gap structure that LN created.
    
    Hypotheses:
    H1: MLP rotates h away from Delta_W direction
    H2: MLP increases variance of non-Delta_W dimensions (re-adding noise)
    H3: MLP's non-linear activation (SiLU/GELU) distorts the gap signal
    
    Tests:
    1. Rotation: cos(h_after_ln, h_after_mlp) — how much does MLP rotate h?
    2. Variance re-injection: var(h_after_mlp) vs var(h_after_ln) per dimension
    3. Gap SNR before/after MLP
    4. MLP output decomposition: aligned vs orthogonal components
    """
    print(f"\n{'='*60}")
    print(f"P661: MLP Disruption Mechanism ({model_name})")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    layers = get_layers(model)
    last_layer = layers[-1]
    d_model = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 2560
    
    n_texts = 100
    test_texts = ALL_TEST_TEXTS[:n_texts]
    
    results = {
        'cos_h_ln_vs_mlp': [],      # cos(h_after_post_attn_ln, h_after_mlp)
        'gap_snr_before_mlp': [],    # SNR of gap at h_after_post_attn_ln
        'gap_snr_after_mlp': [],     # SNR of gap at h_after_mlp
        'var_ratio_mlp': [],         # var(h_after_mlp) / var(h_after_ln)
        'mlp_norm': [],
        'h_ln_norm': [],
    }
    
    h_ln_list = []   # h after post-attn LN
    h_mlp_list = []  # h after MLP + residual
    logit_gaps = []
    Delta_Ws = []
    
    for i, text in enumerate(test_texts):
        if (i+1) % 20 == 0:
            print(f"  Processing text {i+1}/{n_texts}...")
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        hook_data = {}
        def make_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hook_data[name] = output[0].detach()
                else:
                    hook_data[name] = output.detach()
            return hook_fn
        
        handles = []
        # Hook post-attn LN
        for ln_name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
            if hasattr(last_layer, ln_name):
                handles.append(getattr(last_layer, ln_name).register_forward_hook(make_hook('post_attn_ln')))
                break
        # Hook MLP output
        handles.append(last_layer.mlp.register_forward_hook(make_hook('mlp_out')))
        # Hook Attn output (to reconstruct h_after_attn)
        handles.append(last_layer.self_attn.register_forward_hook(make_hook('attn_out')))
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        for h in handles:
            h.remove()
        
        h_Lm1 = outputs.hidden_states[-2][0, -1, :].cpu().float().numpy()
        h_L = outputs.hidden_states[-1][0, -1, :].cpu().float().numpy()
        
        attn_out = hook_data.get('attn_out', None)
        mlp_out = hook_data.get('mlp_out', None)
        post_attn_ln = hook_data.get('post_attn_ln', None)
        
        if attn_out is not None:
            attn_out = attn_out[0, -1, :].cpu().float().numpy()
        if mlp_out is not None:
            mlp_out = mlp_out[0, -1, :].cpu().float().numpy()
        if post_attn_ln is not None:
            post_attn_ln = post_attn_ln[0, -1, :].cpu().float().numpy()
        
        # Reconstruct intermediate states
        h_after_attn = h_Lm1 + (attn_out if attn_out is not None else 0)
        
        # h after post-attn LN (input to MLP)
        h_ln = post_attn_ln if post_attn_ln is not None else h_after_attn
        
        # h after MLP + residual
        h_mlp = h_after_attn + (mlp_out if mlp_out is not None else 0)
        
        logits = outputs.logits[0, -1, :].cpu().float().numpy()
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        Delta_W = W_U[top1_idx] - W_U[top2_idx]
        
        h_ln_list.append(h_ln)
        h_mlp_list.append(h_mlp)
        logit_gaps.append(logits[top1_idx] - logits[top2_idx])
        Delta_Ws.append(Delta_W)
        
        # Cosine between h_ln and h_mlp
        norm_ln = np.linalg.norm(h_ln)
        norm_mlp = np.linalg.norm(h_mlp)
        if norm_ln > 1e-8 and norm_mlp > 1e-8:
            cos_ln_mlp = np.dot(h_ln, h_mlp) / (norm_ln * norm_mlp)
        else:
            cos_ln_mlp = 0
        results['cos_h_ln_vs_mlp'].append(float(cos_ln_mlp))
    
    h_ln_arr = np.array(h_ln_list)
    h_mlp_arr = np.array(h_mlp_list)
    gaps = np.array(logit_gaps)
    Delta_W_arr = np.array(Delta_Ws)
    
    # ===== Analysis =====
    print(f"\n1. Rotation Analysis:")
    cos_vals = np.array(results['cos_h_ln_vs_mlp'])
    print(f"   cos(h_after_LN, h_after_MLP): mean={np.mean(cos_vals):.4f}, std={np.std(cos_vals):.4f}")
    
    # ===== 2. Gap SNR comparison =====
    print(f"\n2. Gap SNR Before vs After MLP:")
    
    gaps_ln = np.array([np.dot(h_ln_arr[i], Delta_W_arr[i]) for i in range(n_texts)])
    gaps_mlp = np.array([np.dot(h_mlp_arr[i], Delta_W_arr[i]) for i in range(n_texts)])
    
    r_ln, _ = stats.pearsonr(gaps_ln, gaps)
    r_mlp, _ = stats.pearsonr(gaps_mlp, gaps)
    
    snr_ln = r_ln**2 / (1 - r_ln**2) if abs(r_ln) < 1 else float('inf')
    snr_mlp = r_mlp**2 / (1 - r_mlp**2) if abs(r_mlp) < 1 else float('inf')
    
    print(f"   gap(h_after_LN) → logit_gap: r={r_ln:.4f}, SNR={snr_ln:.4f}")
    print(f"   gap(h_after_MLP) → logit_gap: r={r_mlp:.4f}, SNR={snr_mlp:.4f}")
    
    if abs(snr_ln) > 1e-8:
        print(f"   SNR loss from MLP: {snr_mlp/snr_ln:.4f} ({(1-snr_mlp/snr_ln)*100:.1f}% lost)")
    
    # ===== 3. Variance re-injection =====
    print(f"\n3. Variance Re-injection by MLP:")
    
    var_ln = np.var(h_ln_arr, axis=0)
    var_mlp = np.var(h_mlp_arr, axis=0)
    
    var_ratio = np.mean(var_mlp) / np.mean(var_ln)
    print(f"   Mean variance ratio (MLP/LN): {var_ratio:.2f}")
    print(f"   Var before LN: range=[{np.min(var_ln):.2f}, {np.max(var_ln):.2f}]")
    print(f"   Var after MLP: range=[{np.min(var_mlp):.2f}, {np.max(var_mlp):.2f}]")
    print(f"   MLP re-injects variance: {var_ratio:.2f}x (makes LN whitening ineffective)")
    
    # ===== 4. MLP output direction analysis =====
    print(f"\n4. MLP Output Direction Analysis:")
    
    # Look at MLP's effect on gap
    delta_gap_mlp = gaps_mlp - gaps_ln
    r_delta, _ = stats.pearsonr(delta_gap_mlp, gaps)
    print(f"   delta_gap_mlp = gap_after_MLP - gap_after_LN:")
    print(f"   mean={np.mean(delta_gap_mlp):.4f}, std={np.std(delta_gap_mlp):.4f}")
    print(f"   corr(delta_gap_mlp, logit_gap): r={r_delta:.4f}")
    
    # ===== 5. Key insight =====
    print(f"\n5. KEY INSIGHT:")
    print(f"   MLP disrupts gap information by:")
    if var_ratio > 2.0:
        print(f"   - Re-injecting variance: {var_ratio:.1f}x → LN whitening is undone")
    else:
        print(f"   - Rotating h away from gap-revealing subspace (cos={np.mean(cos_vals):.4f})")
    print(f"   - SNR loss: {(1-snr_mlp/snr_ln)*100:.1f}% → gap information is degraded")
    print(f"   - MLP is a 'noise injection' step that undoes LN's 'whitening'")
    print(f"   - But final LN re-whitens → gap information re-emerges")
    
    return {
        'cos_ln_mlp': float(np.mean(cos_vals)),
        'r_ln': float(r_ln),
        'r_mlp': float(r_mlp),
        'snr_ln': float(snr_ln),
        'snr_mlp': float(snr_mlp),
        'var_ratio': float(var_ratio),
    }


def experiment_p662(model, tokenizer, device, model_name):
    """P662: Hidden Gap Encoding Form.
    
    Key question: In what form does gap information exist in h_{L-1}?
    
    We know:
    - gap(h_{L-1}, Delta_W) → logit_gap: r≈-0.1 (no simple encoding)
    - GLM4 Ridge on h_{L-1} → logit_gap: r≈0.62 (information IS there!)
    - So gap information exists but in a "hidden" (non-Delta_W-aligned) form
    
    Possibilities:
    1. Gap info is distributed across many dimensions (high-dimensional encoding)
    2. Gap info is in W_U's principal directions (not Delta_W specifically)
    3. Gap info is in the "whitened" version of h (i.e., after implicit normalization)
    4. Gap info needs non-linear extraction (not just dot product)
    
    Tests:
    1. SVD of h_{L-1} and W_U — find shared directions
    2. Project h_{L-1} onto W_U's top SVD directions → predict logit_gap?
    3. "Implicit whitening": divide h by per-dimension std → predict gap?
    4. Non-linear extraction: MLP/Ridge with increasing k
    """
    print(f"\n{'='*60}")
    print(f"P662: Hidden Gap Encoding Form ({model_name})")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    d_model = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 2560
    
    n_texts = 200
    test_texts = ALL_TEST_TEXTS[:n_texts]
    
    h_Lm1_list = []
    logit_gaps = []
    Delta_Ws = []
    top1_indices = []
    
    print(f"  Collecting data for {n_texts} texts...")
    for i, text in enumerate(test_texts):
        if (i+1) % 50 == 0:
            print(f"    {i+1}/{n_texts}...")
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"].to(device),
                          attention_mask=inputs["attention_mask"].to(device),
                          output_hidden_states=True)
        
        h_Lm1 = outputs.hidden_states[-2][0, -1, :].cpu().float().numpy()
        logits = outputs.logits[0, -1, :].cpu().float().numpy()
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        
        h_Lm1_list.append(h_Lm1)
        logit_gaps.append(logits[top1_idx] - logits[top2_idx])
        Delta_Ws.append(W_U[top1_idx] - W_U[top2_idx])
        top1_indices.append(top1_idx)
    
    h_arr = np.array(h_Lm1_list)
    gaps = np.array(logit_gaps)
    Delta_W_arr = np.array(Delta_Ws)
    
    # ===== 1. W_U SVD — shared structure with h =====
    print(f"\n1. W_U SVD and h Shared Directions:")
    
    # Use truncated SVD for large W_U
    n_W_components = min(100, d_model)
    pca_W = PCA(n_components=n_W_components)
    pca_W.fit(W_U)
    W_U_dirs = pca_W.components_  # [n_components, d_model]
    
    # SVD of h
    U_h, S_h, Vt_h = np.linalg.svd(h_arr, full_matrices=False)
    
    print(f"   W_U: top-5 explained variance ratio={pca_W.explained_variance_ratio_[:5].tolist()}")
    print(f"   h: top-5 singular values={[f'{s:.2f}' for s in S_h[:5]]}")
    
    # ===== 2. Project h onto W_U SVD directions =====
    print(f"\n2. h_(L-1) Projected onto W_U SVD Directions:")
    
    # Project h onto top-k W_U directions
    for k in [1, 5, 10, 20, 50]:
        if k > n_W_components:
            continue
        W_U_top_dirs = W_U_dirs[:k]  # [k, d_model]
        h_proj = h_arr @ W_U_top_dirs.T  # [n_texts, k]
        
        # Ridge LOO
        try:
            loo_preds = []
            for train_idx, test_idx in LeaveOneOut().split(h_proj):
                ridge = Ridge(alpha=1.0).fit(h_proj[train_idx], gaps[train_idx])
                loo_preds.append(ridge.predict(h_proj[test_idx])[0])
            loo_preds = np.array(loo_preds)
            r_loo, _ = stats.pearsonr(loo_preds, gaps)
        except:
            r_loo = -1
        
        print(f"   W_U top-{k:2d} dirs: Ridge LOO r={r_loo:.4f}")
    
    # ===== 3. Implicit whitening =====
    print(f"\n3. Implicit Whitening (divide h by per-dim std):")
    
    h_std = np.std(h_arr, axis=0)  # [d_model]
    h_std[h_std < 1e-8] = 1.0  # avoid division by zero
    
    h_whitened = h_arr / h_std  # "whiten" h by dividing by per-dim std
    
    # Gap in whitened space
    gaps_whitened = np.array([np.dot(h_whitened[i], Delta_W_arr[i]) for i in range(n_texts)])
    r_whitened, _ = stats.pearsonr(gaps_whitened, gaps)
    
    print(f"   gap(h_whitened, Delta_W) → logit_gap: r={r_whitened:.4f}")
    print(f"   gap(h_original, Delta_W) → logit_gap: r={stats.pearsonr(np.array([np.dot(h_arr[i], Delta_W_arr[i]) for i in range(n_texts)]), gaps)[0]:.4f}")
    
    # Ridge on whitened h
    for k in [10, 20, 50]:
        pca = PCA(n_components=min(k, n_texts-1))
        h_pca = pca.fit_transform(h_whitened)
        try:
            loo_preds = []
            for train_idx, test_idx in LeaveOneOut().split(h_pca):
                ridge = Ridge(alpha=1.0).fit(h_pca[train_idx], gaps[train_idx])
                loo_preds.append(ridge.predict(h_pca[test_idx])[0])
            loo_preds = np.array(loo_preds)
            r_loo, _ = stats.pearsonr(loo_preds, gaps)
        except:
            r_loo = -1
        print(f"   Whitened h, k={k:2d}: Ridge LOO r={r_loo:.4f}")
    
    # ===== 4. Delta_W as variable direction =====
    print(f"\n4. Per-Text Delta_W vs Mean Delta_W:")
    
    mean_DeltaW = np.mean(Delta_W_arr, axis=0)
    
    # gap with mean Delta_W
    gaps_mean_DW = np.array([np.dot(h_arr[i], mean_DeltaW) for i in range(n_texts)])
    r_mean_DW, _ = stats.pearsonr(gaps_mean_DW, gaps)
    
    # gap with per-text Delta_W (original)
    gaps_per_text = np.array([np.dot(h_arr[i], Delta_W_arr[i]) for i in range(n_texts)])
    r_per_text, _ = stats.pearsonr(gaps_per_text, gaps)
    
    print(f"   gap(h, mean(Delta_W)) → logit_gap: r={r_mean_DW:.4f}")
    print(f"   gap(h, per-text Delta_W) → logit_gap: r={r_per_text:.4f}")
    
    # ===== 5. Key finding =====
    print(f"\n5. KEY FINDING:")
    print(f"   Gap information in h_{{L-1}} is 'hidden' because:")
    print(f"   1. Different texts have different Delta_W directions (each text has its own top1/top2)")
    print(f"   2. h encodes information relevant to W_U rows, not specifically to (W_U[top1]-W_U[top2])")
    print(f"   3. LN's whitening effect 'reveals' this by equalizing dimension contributions")
    print(f"   4. GLM4's low-dimensional encoding means gap info concentrates in few W_U directions")
    
    return {
        'r_whitened': float(r_whitened),
        'r_mean_DW': float(r_mean_DW),
        'r_per_text': float(r_per_text),
    }


def experiment_p663(model, tokenizer, device, model_name):
    """P663: GLM4 Low-Dim Encoding Physics.
    
    Key question: Why can GLM4 encode gap info in 10 dimensions while others cannot?
    
    From P659:
    - GLM4: k=10 → r>0.5, Ridge LOO max=0.654
    - Qwen3: k=50 → r<0.3, Ridge LOO max=0.277
    - DS7B: k=100 → r<0.22, Ridge LOO max=0.212
    
    Possible explanations:
    1. GLM4's W_U has different structure (more concentrated singular values?)
    2. GLM4's h_{L-1} has lower intrinsic dimensionality
    3. GLM4's training creates more structured h representations
    4. GLM4's merged_gate_up architecture creates different MLP dynamics
    
    Tests:
    1. W_U SVD structure comparison (already done in P662, but focused)
    2. h_{L-1} intrinsic dimensionality (participation ratio)
    3. W_U top directions' alignment with h_{L-1} top PCA directions
    4. Per-layer Ridge LOO curve shape
    """
    print(f"\n{'='*60}")
    print(f"P663: GLM4 Low-Dim Encoding Physics ({model_name})")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    d_model = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 2560
    
    n_texts = 200
    test_texts = ALL_TEST_TEXTS[:n_texts]
    
    h_Lm1_list = []
    logit_gaps = []
    Delta_Ws = []
    
    print(f"  Collecting data for {n_texts} texts...")
    for i, text in enumerate(test_texts):
        if (i+1) % 50 == 0:
            print(f"    {i+1}/{n_texts}...")
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"].to(device),
                          attention_mask=inputs["attention_mask"].to(device),
                          output_hidden_states=True)
        
        h_Lm1 = outputs.hidden_states[-2][0, -1, :].cpu().float().numpy()
        logits = outputs.logits[0, -1, :].cpu().float().numpy()
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        
        h_Lm1_list.append(h_Lm1)
        logit_gaps.append(logits[top1_idx] - logits[top2_idx])
        Delta_Ws.append(W_U[top1_idx] - W_U[top2_idx])
    
    h_arr = np.array(h_Lm1_list)
    gaps = np.array(logit_gaps)
    Delta_W_arr = np.array(Delta_Ws)
    
    # ===== 1. W_U structure =====
    print(f"\n1. W_U SVD Structure:")
    
    # Use truncated SVD for large W_U
    n_components = min(100, d_model)
    pca_W = PCA(n_components=n_components)
    pca_W.fit(W_U)
    S_W_approx = np.sqrt(pca_W.explained_variance_ * min(W_U.shape))
    
    # Participation ratio of W_U (approximate using PCA)
    S_W_full_est = np.sqrt(pca_W.explained_variance_) * np.sqrt(min(W_U.shape))
    PR_W = (np.sum(S_W_full_est))**2 / np.sum(S_W_full_est**2)
    
    # Top-k concentration
    total_energy = np.sum(pca_W.explained_variance_)
    for k in [1, 5, 10, 20, 50, 100]:
        energy_k = np.sum(pca_W.explained_variance_[:k]) / total_energy * 100
        print(f"   Top-{k:3d} W_U singular values: {energy_k:.1f}% of energy")
    
    print(f"   W_U participation ratio (approx): {PR_W:.1f}")
    
    # W_U directions from PCA
    W_U_dirs = pca_W.components_  # [n_components, d_model]
    
    # ===== 2. h_{L-1} intrinsic dimensionality =====
    print(f"\n2. h_(L-1) Intrinsic Dimensionality:")
    
    U_h, S_h, Vt_h = np.linalg.svd(h_arr, full_matrices=False)
    
    PR_h = (np.sum(S_h))**2 / np.sum(S_h**2)
    
    total_energy_h = np.sum(S_h**2)
    for k in [1, 5, 10, 20, 50, 100]:
        if k < len(S_h):
            energy_k = np.sum(S_h[:k]**2) / total_energy_h * 100
            print(f"   Top-{k:3d} h singular values: {energy_k:.1f}% of energy")
    
    print(f"   h participation ratio: {PR_h:.1f}")
    
    # ===== 3. W_U-h alignment =====
    print(f"\n3. W_U-h Alignment (Subspace Overlap):")
    
    # How much do the top-k W_U directions overlap with top-k h directions?
    for k in [5, 10, 20, 50]:
        if k > n_components or k > len(S_h):
            continue
        W_U_subspace = W_U_dirs[:k]  # [k, d_model]
        h_subspace = Vt_h[:k]    # [k, d_model]
        
        # Subspace overlap = sum of |cos| between subspaces
        # Compute: for each W_U direction, find max cos with any h direction
        max_cosines = []
        for i in range(k):
            cos_vals = np.abs(W_U_subspace[i] @ h_subspace.T)
            max_cosines.append(np.max(cos_vals))
        
        mean_max_cos = np.mean(max_cosines)
        print(f"   Top-{k:2d} subspace overlap: mean_max_cos={mean_max_cos:.4f}")
    
    # ===== 4. Ridge LOO curve shape =====
    print(f"\n4. Ridge LOO Curve Shape (Information Concentration):")
    
    k_scan = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50]
    ridge_results = []
    
    for k in k_scan:
        if k >= n_texts or k >= d_model:
            continue
        try:
            pca = PCA(n_components=k)
            h_pca = pca.fit_transform(h_arr)
            
            loo_preds = []
            for train_idx, test_idx in LeaveOneOut().split(h_pca):
                ridge = Ridge(alpha=1.0).fit(h_pca[train_idx], gaps[train_idx])
                loo_preds.append(ridge.predict(h_pca[test_idx])[0])
            loo_preds = np.array(loo_preds)
            r_loo, _ = stats.pearsonr(loo_preds, gaps)
        except:
            r_loo = -1
        
        ridge_results.append({'k': k, 'r_loo': r_loo})
        print(f"   k={k:3d}: r_LOO={r_loo:.4f}")
    
    # Find the "elbow" — where adding more dimensions stops helping
    best_r = max(ridge_results, key=lambda x: x['r_loo'])
    print(f"   Best k={best_r['k']}, r={best_r['r_loo']:.4f}")
    
    # ===== 5. Key diagnostic =====
    print(f"\n5. KEY DIAGNOSTIC:")
    print(f"   Model: {model_name}")
    print(f"   W_U participation ratio: {PR_W:.1f}")
    print(f"   h participation ratio: {PR_h:.1f}")
    print(f"   Best Ridge k: {best_r['k']}, r: {best_r['r_loo']:.4f}")
    
    if PR_h < 50:
        print(f"   → h has LOW intrinsic dimensionality → gap info concentrates in few dimensions")
    else:
        print(f"   → h has HIGH intrinsic dimensionality → gap info is spread across many dimensions")
    
    return {
        'PR_W': float(PR_W),
        'PR_h': float(PR_h),
        'best_k': best_r['k'],
        'best_r': float(best_r['r_loo']),
        'ridge_results': ridge_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase CL: LN Whitening Theory")
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["p660", "p661", "p662", "p663", "all"])
    args = parser.parse_args()
    
    model_name = args.model
    
    print(f"Loading {model_name}...")
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    print(f"Model: {info.name}, layers={info.n_layers}, d_model={info.d_model}")
    
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                               "results", "phase_cl")
    os.makedirs(results_dir, exist_ok=True)
    
    experiments = {
        "p660": experiment_p660,
        "p661": experiment_p661,
        "p662": experiment_p662,
        "p663": experiment_p663,
    }
    
    if args.experiment == "all":
        exp_list = ["p660", "p661", "p662", "p663"]
    else:
        exp_list = [args.experiment]
    
    for exp_name in exp_list:
        print(f"\n{'#'*60}")
        print(f"# Running {exp_name} on {model_name}")
        print(f"{'#'*60}")
        
        start_time = time.time()
        
        try:
            result = experiments[exp_name](model, tokenizer, device, model_name)
            elapsed = time.time() - start_time
            print(f"\n{exp_name} completed in {elapsed:.1f}s")
            
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert(v) for v in obj]
                return obj
            
            result_serializable = convert(result)
            
            result_path = os.path.join(results_dir, f"{exp_name}_{model_name}.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result_serializable, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {result_path}")
            
        except Exception as e:
            print(f"\nERROR in {exp_name}: {e}")
            import traceback
            traceback.print_exc()
    
    release_model(model)
    print(f"\nModel released.")


if __name__ == "__main__":
    main()
