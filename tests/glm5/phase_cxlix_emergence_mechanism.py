#!/usr/bin/env python3
"""
Phase CXLIX: Last-Layer Emergence Mechanism Analysis (P656-P659)
================================================================

Based on Phase CXLVIII findings:
1. LN causal intervention is perfectly linear (R2>0.992, gap(2x)/gap(1x)=2.000)
2. MLP causal intervention has NEGATIVE slope (Qwen3=-0.193, GLM4=-0.496)
3. Gap emergence is SUDDEN (Delta_r>1.10 in all 3 models)
4. GLM4 intermediate layers can encode gap info (Ridge LOO=0.62) but simple dot product r=-0.14

This phase addresses:
P656: MLP Negative Causality Physics — which directions does MLP push h toward?
P657: LN Scaling First-Principles — why is gap(2x)/gap(1x) exactly 2.000?
P658: Sudden Emergence Minimal Mechanism — what exactly happens at the last layer?
P659: GLM4 vs Qwen3/DS7B Information Encoding — why can GLM4 encode gap in middle layers?
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

# Reuse text corpus from Phase CXLVIII
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


def experiment_p656(model, tokenizer, device, model_name):
    """P656: MLP Negative Causality Physics.
    
    Key question: Why does scaling up MLP output DECREASE gap?
    
    Hypotheses:
    H1: MLP pushes h AWAY from Delta_W direction (anti-alignment)
    H2: MLP increases variance of h, causing LN to compress Delta_W component
    H3: MLP output has negative cosine with Delta_W on average
    
    Tests:
    1. cos(mlp_output, Delta_W) distribution — is it systematically negative?
    2. After MLP, does h move toward or away from W_U[top1]?
    3. MLP's effect on LN normalization: does MLP increase ||h|| so LN compresses gap?
    4. Decompose MLP output into Delta_W-aligned and Delta_W-orthogonal components
    """
    print(f"\n{'='*60}")
    print(f"P656: MLP Negative Causality Physics ({model_name})")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    layers = get_layers(model)
    last_layer = layers[-1]
    
    n_texts = 100
    test_texts = ALL_TEST_TEXTS[:n_texts]
    
    results = {
        'cos_mlp_DeltaW': [],      # cos(mlp_output, Delta_W)
        'cos_mlp_Wtop1': [],       # cos(mlp_output, W_U[top1])
        'cos_mlp_Wtop2': [],       # cos(mlp_output, W_U[top2])
        'mlp_proj_DeltaW': [],     # mlp_output · Delta_W
        'mlp_proj_Wtop1': [],      # mlp_output · W_U[top1]
        'mlp_proj_Wtop2': [],      # mlp_output · W_U[top2]
        'gap_before_mlp': [],       # gap at h_after_attn (before MLP)
        'gap_after_mlp': [],        # gap at h_after_mlp (after MLP)
        'delta_gap_mlp': [],        # gap_after_mlp - gap_before_mlp
        'norm_mlp': [],             # ||mlp_output||
        'norm_h_before_mlp': [],    # ||h_before_mlp||
        'norm_h_after_mlp': [],     # ||h_after_mlp||
        'norm_DeltaW': [],          # ||Delta_W||
        # LN compression analysis
        'gap_before_ln': [],        # gap at h_after_mlp (before final LN)
        'gap_after_ln': [],         # gap at h_final (after final LN)
        'std_h_before_ln': [],      # std(h_before_ln) across dimensions
        'ln_scale_factor': [],      # 1/std(h_before_ln) * LN_weight
        # MLP direction decomposition
        'mlp_aligned_frac': [],     # fraction of mlp_output along Delta_W
        'mlp_orthogonal_frac': [],  # fraction of mlp_output orthogonal to Delta_W
    }
    
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
        
        # Hook last layer attn, mlp
        handles = []
        handles.append(last_layer.self_attn.register_forward_hook(make_hook('attn_out')))
        handles.append(last_layer.mlp.register_forward_hook(make_hook('mlp_out')))
        # Hook LN before MLP (post_attention_layernorm)
        for ln_name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
            if hasattr(last_layer, ln_name):
                handles.append(getattr(last_layer, ln_name).register_forward_hook(make_hook('pre_mlp_ln')))
                break
        # Hook final LN
        final_ln = get_final_ln(model)
        if final_ln is not None:
            handles.append(final_ln.register_forward_hook(make_hook('final_ln_out')))
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        for h in handles:
            h.remove()
        
        # Extract data
        h_L_minus_1 = outputs.hidden_states[-2][0, -1, :].cpu().float().numpy()
        h_L = outputs.hidden_states[-1][0, -1, :].cpu().float().numpy()
        
        attn_out = hook_data.get('attn_out', None)
        mlp_out = hook_data.get('mlp_out', None)
        pre_mlp_ln = hook_data.get('pre_mlp_ln', None)
        final_ln_out = hook_data.get('final_ln_out', None)
        
        if attn_out is not None:
            attn_out = attn_out[0, -1, :].cpu().float().numpy()
        if mlp_out is not None:
            mlp_out = mlp_out[0, -1, :].cpu().float().numpy()
        if pre_mlp_ln is not None:
            pre_mlp_ln = pre_mlp_ln[0, -1, :].cpu().float().numpy()
        if final_ln_out is not None:
            final_ln_out = final_ln_out[0, -1, :].cpu().float().numpy()
        
        # Logits and Delta_W
        logits = outputs.logits[0, -1, :].cpu().float().numpy()
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        logit_gap = logits[top1_idx] - logits[top2_idx]
        
        W_top1 = W_U[top1_idx]
        W_top2 = W_U[top2_idx]
        Delta_W = W_top1 - W_top2
        
        # Key measurements
        if mlp_out is not None:
            norm_mlp = np.linalg.norm(mlp_out)
            norm_DeltaW = np.linalg.norm(Delta_W)
            
            # Cosines
            if norm_mlp > 1e-8 and norm_DeltaW > 1e-8:
                cos_mlp_DeltaW = np.dot(mlp_out, Delta_W) / (norm_mlp * norm_DeltaW)
                cos_mlp_Wtop1 = np.dot(mlp_out, W_top1) / (norm_mlp * np.linalg.norm(W_top1))
                cos_mlp_Wtop2 = np.dot(mlp_out, W_top2) / (norm_mlp * np.linalg.norm(W_top2))
            else:
                cos_mlp_DeltaW = 0
                cos_mlp_Wtop1 = 0
                cos_mlp_Wtop2 = 0
            
            # Projections
            proj_DeltaW = np.dot(mlp_out, Delta_W)
            proj_Wtop1 = np.dot(mlp_out, W_top1)
            proj_Wtop2 = np.dot(mlp_out, W_top2)
            
            # Gap before/after MLP
            h_after_attn = h_L_minus_1 + (attn_out if attn_out is not None else 0)
            gap_before_mlp = np.dot(h_after_attn, Delta_W)
            gap_after_mlp_raw = np.dot(h_after_attn + mlp_out, Delta_W)
            delta_gap_mlp = gap_after_mlp_raw - gap_before_mlp
            
            # MLP direction decomposition
            # Delta_W direction unit vector
            if norm_DeltaW > 1e-8:
                u_DeltaW = Delta_W / norm_DeltaW
                mlp_aligned = np.dot(mlp_out, u_DeltaW)
                mlp_aligned_frac = mlp_aligned**2 / (norm_mlp**2 + 1e-12)
            else:
                mlp_aligned_frac = 0
            
            results['cos_mlp_DeltaW'].append(float(cos_mlp_DeltaW))
            results['cos_mlp_Wtop1'].append(float(cos_mlp_Wtop1))
            results['cos_mlp_Wtop2'].append(float(cos_mlp_Wtop2))
            results['mlp_proj_DeltaW'].append(float(proj_DeltaW))
            results['mlp_proj_Wtop1'].append(float(proj_Wtop1))
            results['mlp_proj_Wtop2'].append(float(proj_Wtop2))
            results['gap_before_mlp'].append(float(gap_before_mlp))
            results['gap_after_mlp'].append(float(gap_after_mlp_raw))
            results['delta_gap_mlp'].append(float(delta_gap_mlp))
            results['norm_mlp'].append(float(norm_mlp))
            results['norm_h_before_mlp'].append(float(np.linalg.norm(h_after_attn)))
            results['norm_h_after_mlp'].append(float(np.linalg.norm(h_after_attn + mlp_out)))
            results['norm_DeltaW'].append(float(norm_DeltaW))
            results['mlp_aligned_frac'].append(float(mlp_aligned_frac))
        
        # LN analysis
        if final_ln_out is not None:
            gap_before_ln_val = np.dot(h_after_attn + (mlp_out if mlp_out is not None else 0), Delta_W)
            gap_after_ln_val = np.dot(h_L, Delta_W)
            
            # h before LN
            h_before_ln = h_after_attn + (mlp_out if mlp_out is not None else 0)
            std_h_before = np.std(h_before_ln)
            
            # LN scale: gap_after / gap_before
            if abs(gap_before_ln_val) > 1e-8:
                ln_scale = gap_after_ln_val / gap_before_ln_val
            else:
                ln_scale = 0
            
            results['gap_before_ln'].append(float(gap_before_ln_val))
            results['gap_after_ln'].append(float(gap_after_ln_val))
            results['std_h_before_ln'].append(float(std_h_before))
            results['ln_scale_factor'].append(float(ln_scale))
    
    # ===== Analysis =====
    print(f"\n1. MLP Direction Analysis:")
    
    cos_vals = np.array(results['cos_mlp_DeltaW'])
    print(f"   cos(mlp_output, Delta_W): mean={np.mean(cos_vals):.4f}, "
          f"median={np.median(cos_vals):.4f}, std={np.std(cos_vals):.4f}")
    print(f"   Fraction with cos<0: {np.mean(cos_vals < 0):.3f}")
    
    cos_top1 = np.array(results['cos_mlp_Wtop1'])
    cos_top2 = np.array(results['cos_mlp_Wtop2'])
    print(f"   cos(mlp, W_top1): mean={np.mean(cos_top1):.4f}")
    print(f"   cos(mlp, W_top2): mean={np.mean(cos_top2):.4f}")
    print(f"   cos(mlp, W_top1) - cos(mlp, W_top2): mean={np.mean(cos_top1 - cos_top2):.4f}")
    
    proj_vals = np.array(results['mlp_proj_DeltaW'])
    print(f"   mlp · Delta_W: mean={np.mean(proj_vals):.4f}, "
          f"positive frac={np.mean(proj_vals > 0):.3f}")
    
    print(f"\n2. Gap Before/After MLP:")
    gap_before = np.array(results['gap_before_mlp'])
    gap_after = np.array(results['gap_after_mlp'])
    delta_gap = np.array(results['delta_gap_mlp'])
    
    print(f"   gap_before_mlp: mean={np.mean(gap_before):.4f}, std={np.std(gap_before):.4f}")
    print(f"   gap_after_mlp: mean={np.mean(gap_after):.4f}, std={np.std(gap_after):.4f}")
    print(f"   delta_gap_mlp: mean={np.mean(delta_gap):.4f}, std={np.std(delta_gap):.4f}")
    print(f"   Fraction where MLP DECREASES gap: {np.mean(delta_gap < 0):.3f}")
    
    # Correlation: does mlp_proj_DeltaW predict delta_gap?
    r_proj, _ = stats.pearsonr(results['mlp_proj_DeltaW'], results['delta_gap_mlp'])
    print(f"   corr(mlp·DeltaW, delta_gap_mlp): r={r_proj:.4f}")
    
    print(f"\n3. LN Compression Analysis:")
    gap_before_ln = np.array(results['gap_before_ln'])
    gap_after_ln = np.array(results['gap_after_ln'])
    ln_scales = np.array(results['ln_scale_factor'])
    std_before = np.array(results['std_h_before_ln'])
    
    print(f"   gap_before_ln: mean={np.mean(gap_before_ln):.4f}")
    print(f"   gap_after_ln: mean={np.mean(gap_after_ln):.4f}")
    print(f"   LN scale factor (gap_after/before): mean={np.mean(ln_scales):.4f}, "
          f"std={np.std(ln_scales):.4f}")
    print(f"   std(h_before_ln): mean={np.mean(std_before):.4f}")
    
    # Key test: is LN scale factor predictable from std?
    r_scale_std, _ = stats.pearsonr(ln_scales, std_before)
    print(f"   corr(LN_scale, std_before_ln): r={r_scale_std:.4f}")
    
    # Does increasing MLP decrease LN scale?
    norm_mlp = np.array(results['norm_mlp'])
    r_mlp_scale, _ = stats.pearsonr(norm_mlp, ln_scales)
    print(f"   corr(||mlp||, LN_scale): r={r_mlp_scale:.4f}")
    
    print(f"\n4. MLP Direction Fraction:")
    aligned_frac = np.array(results['mlp_aligned_frac'])
    print(f"   Fraction of MLP energy along Delta_W: mean={np.mean(aligned_frac):.6f}")
    print(f"   (Random baseline: ~1/d_model = {1.0/len(Delta_W):.6f})")
    
    # Hypothesis test
    print(f"\n5. Hypothesis Test:")
    if np.mean(cos_vals) < 0:
        print(f"   H1 SUPPORTED: cos(mlp, DeltaW) < 0 on average ({np.mean(cos_vals):.4f})")
        print(f"   -> MLP pushes h AWAY from Delta_W direction")
    else:
        print(f"   H1 REJECTED: cos(mlp, DeltaW) >= 0 on average ({np.mean(cos_vals):.4f})")
    
    if r_mlp_scale < -0.3:
        print(f"   H2 SUPPORTED: corr(||mlp||, LN_scale) < -0.3 ({r_mlp_scale:.4f})")
        print(f"   -> Larger MLP → smaller LN scale → LN compresses gap more")
    else:
        print(f"   H2 INCONCLUSIVE: corr(||mlp||, LN_scale) = {r_mlp_scale:.4f}")
    
    return results


def experiment_p657(model, tokenizer, device, model_name):
    """P657: LN Scaling First-Principles Derivation.
    
    Key question: Why is gap(2x)/gap(1x) EXACTLY 2.000?
    
    LN(x) = gamma * (x - mean(x)) / std(x)
    
    For gap:
    gap_after_LN = LN(h) · Delta_W
                  = gamma * (h - mean(h)) / std(h) · Delta_W
                  = gamma / std(h) * [(h · Delta_W) - mean(h) * (1·Delta_W)]
    
    If mean(h) ≈ 0 (which LN enforces):
    gap_after_LN ≈ gamma / std(h) * (h · Delta_W) = scale * gap_before_LN
    
    When we scale gamma by s:
    gap_after_LN(s*gamma) = s * gap_after_LN(gamma)
    
    So gap(2x)/gap(1x) = 2.000 PERFECTLY!
    
    This is because LN is a HOMOGENEOUS function of its weight parameter.
    
    Tests:
    1. Verify mean(h) ≈ 0 before LN
    2. Verify LN scale = gamma / std(h_before_LN)
    3. Test: scaling gamma by s gives gap scaled by s (exact)
    4. Test: scaling h_before_LN by s does NOT give gap scaled by s (LN is non-linear in x)
    """
    print(f"\n{'='*60}")
    print(f"P657: LN Scaling First-Principles ({model_name})")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    final_ln = get_final_ln(model)
    d_model = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 2560
    
    n_texts = 50
    test_texts = ALL_TEST_TEXTS[:n_texts]
    
    # Get original LN weight
    orig_ln_weight = final_ln.weight.data.clone()
    orig_ln_bias = final_ln.bias.data.clone() if hasattr(final_ln, 'bias') and final_ln.bias is not None else None
    
    # ===== Part 1: Verify LN formula =====
    print(f"\n1. Verify LN Formula: gap_after = (gamma/std(h)) * gap_before")
    
    results_ln = {
        'mean_h_before_ln': [],
        'gap_before_ln': [],
        'gap_after_ln': [],
        'std_h_before_ln': [],
        'predicted_gap_after': [],
        'ln_weight_norm': [],
    }
    
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # Hook before and after final LN
        hook_data = {}
        
        def make_hook(name):
            def hook_fn(module, input, output):
                if isinstance(input, tuple):
                    hook_data[name + '_input'] = input[0].detach()
                if isinstance(output, tuple):
                    hook_data[name + '_output'] = output[0].detach()
                else:
                    hook_data[name + '_output'] = output.detach()
            return hook_fn
        
        handle = final_ln.register_forward_hook(make_hook('final_ln'))
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        handle.remove()
        
        # h before LN (hidden state -2, before final LN)
        h_before_ln = outputs.hidden_states[-2][0, -1, :].cpu().float().numpy()
        # Actually, hidden_states[-2] is h after last transformer block but BEFORE final LN
        # hidden_states[-1] is h after final LN
        h_after_ln = outputs.hidden_states[-1][0, -1, :].cpu().float().numpy()
        
        logits = outputs.logits[0, -1, :].cpu().float().numpy()
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        Delta_W = W_U[top1_idx] - W_U[top2_idx]
        
        gap_before = np.dot(h_before_ln, Delta_W)
        gap_after = np.dot(h_after_ln, Delta_W)
        
        mean_h = np.mean(h_before_ln)
        std_h = np.std(h_before_ln)
        
        # LN formula: gap_after = sum(gamma_i * (h_i - mean) / std * Delta_W_i)
        gamma = orig_ln_weight.cpu().float().numpy()
        
        # Predicted gap after LN
        h_normalized = (h_before_ln - mean_h) / std_h
        predicted_gap = np.dot(gamma * h_normalized, Delta_W)
        
        results_ln['mean_h_before_ln'].append(float(mean_h))
        results_ln['gap_before_ln'].append(float(gap_before))
        results_ln['gap_after_ln'].append(float(gap_after))
        results_ln['std_h_before_ln'].append(float(std_h))
        results_ln['predicted_gap_after'].append(float(predicted_gap))
        results_ln['ln_weight_norm'].append(float(np.linalg.norm(gamma)))
    
    # Verify prediction
    gaps_after = np.array(results_ln['gap_after_ln'])
    gaps_predicted = np.array(results_ln['predicted_gap_after'])
    r_pred, _ = stats.pearsonr(gaps_predicted, gaps_after)
    rmse = np.sqrt(np.mean((gaps_predicted - gaps_after)**2))
    print(f"   Predicted vs actual gap_after_LN: r={r_pred:.6f}, RMSE={rmse:.6f}")
    print(f"   Mean absolute gap_after: {np.mean(np.abs(gaps_after)):.4f}")
    print(f"   Relative RMSE: {rmse / np.mean(np.abs(gaps_after)) * 100:.2f}%")
    
    # Mean h before LN
    means = np.array(results_ln['mean_h_before_ln'])
    print(f"\n2. Mean(h_before_LN): mean={np.mean(means):.6f}, std={np.std(means):.6f}")
    print(f"   (LN centering removes this, so gap_after ≈ (gamma/std) * gap_before)")
    
    # ===== Part 2: Scaling gamma vs scaling h =====
    print(f"\n3. Scale gamma (LN weight) vs Scale h_before_LN:")
    
    scale_factors = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    
    # Test 1: Scale gamma
    print(f"\n   3a. Scale gamma (should give EXACT linear gap scaling):")
    gamma_results = []
    for scale in scale_factors:
        final_ln.weight.data = orig_ln_weight * scale
        if orig_ln_bias is not None:
            final_ln.bias.data = orig_ln_bias * scale
        
        gap_values = []
        for text in test_texts[:10]:  # Use 10 texts for speed
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(input_ids=inputs["input_ids"].to(device), 
                               attention_mask=inputs["attention_mask"].to(device))
            logits = outputs.logits[0, -1, :].cpu().float().numpy()
            top1_idx = np.argmax(logits)
            top2_idx = np.argsort(logits)[-2]
            gap_values.append(logits[top1_idx] - logits[top2_idx])
        
        mean_gap = np.mean(gap_values)
        gamma_results.append({'scale': scale, 'mean_gap': float(mean_gap)})
        print(f"      scale={scale:.2f}: mean_gap={mean_gap:.6f}")
    
    # Restore
    final_ln.weight.data = orig_ln_weight.clone()
    if orig_ln_bias is not None:
        final_ln.bias.data = orig_ln_bias.clone()
    
    # Check linearity
    scales = np.array([r['scale'] for r in gamma_results])
    gaps_gamma = np.array([r['mean_gap'] for r in gamma_results])
    slope, intercept, r_val, p_val, _ = stats.linregress(scales, gaps_gamma)
    gap_at_1 = gaps_gamma[scales == 1.0][0]
    gap_at_2 = gaps_gamma[scales == 2.0][0]
    ratio = gap_at_2 / gap_at_1 if abs(gap_at_1) > 1e-8 else float('inf')
    print(f"      Linear fit: R2={r_val**2:.6f}, gap(2x)/gap(1x)={ratio:.6f}")
    
    # Test 2: Scale h_before_LN via hook
    print(f"\n   3b. Scale h before LN (should NOT give exact linear gap scaling):")
    
    h_scale_results = []
    for scale in scale_factors:
        gap_values = []
        for text in test_texts[:10]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # Hook to scale input to final LN
            def make_input_scale_hook(s):
                def hook_fn(module, input, output):
                    if isinstance(input, tuple):
                        scaled_input = input[0] * s
                        # Re-compute LN output with scaled input
                        # LN(x) = gamma * (x - mean) / std
                        # If we scale x by s: LN(s*x) = gamma * (s*x - s*mean) / (s*std) = gamma * (x - mean) / std = LN(x)
                        # Wait! LN is scale-invariant in x! So scaling x before LN changes NOTHING!
                        # This is the key insight!
                        pass
                    return output
                return hook_fn
            
            # Instead, let's directly test by modifying hidden_states
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            # Manually apply scaled LN
            h_before = outputs.hidden_states[-2][0, -1, :].cpu().float().numpy()
            h_scaled = h_before * scale
            
            # Apply LN manually
            mean_scaled = np.mean(h_scaled)
            std_scaled = np.std(h_scaled)
            gamma = orig_ln_weight.cpu().float().numpy()
            h_after_ln_manual = gamma * (h_scaled - mean_scaled) / std_scaled
            
            # Compute gap
            logits_orig = outputs.logits[0, -1, :].cpu().float().numpy()
            top1_idx = np.argmax(logits_orig)
            top2_idx = np.argsort(logits_orig)[-2]
            Delta_W = W_U[top1_idx] - W_U[top2_idx]
            
            gap = np.dot(h_after_ln_manual, Delta_W)
            gap_values.append(gap)
        
        mean_gap = np.mean(gap_values)
        h_scale_results.append({'scale': scale, 'mean_gap': float(mean_gap)})
        print(f"      scale={scale:.2f}: mean_gap={mean_gap:.6f}")
    
    # Check if scaling h gives same gap
    gaps_h_scale = np.array([r['mean_gap'] for r in h_scale_results])
    slope2, intercept2, r_val2, _, _ = stats.linregress(scales, gaps_h_scale)
    gap_at_1_h = gaps_h_scale[scales == 1.0][0]
    gap_at_2_h = gaps_h_scale[scales == 2.0][0]
    ratio_h = gap_at_2_h / gap_at_1_h if abs(gap_at_1_h) > 1e-8 else float('inf')
    print(f"      Linear fit: R2={r_val2**2:.6f}, gap(2x)/gap(1x)={ratio_h:.6f}")
    
    print(f"\n4. KEY INSIGHT:")
    print(f"   Scaling gamma by s → gap scales by s (ratio={ratio:.4f})")
    print(f"   Scaling h by s → gap is INVARIANT (ratio={ratio_h:.4f})")
    print(f"   This is because LN(x) = gamma * (x-mean)/std is:")
    print(f"     - HOMOGENEOUS of degree 1 in gamma → gap ∝ gamma")
    print(f"     - INVARIANT under scaling of x → gap ∝ gap(h)")
    print(f"   So gap = (gamma/std(h)) * gap_before_LN (approximately)")
    
    return {
        'r_prediction': float(r_pred),
        'rmse': float(rmse),
        'gamma_scaling_ratio': float(ratio),
        'h_scaling_ratio': float(ratio_h),
    }


def experiment_p658(model, tokenizer, device, model_name):
    """P658: Sudden Emergence Minimal Mechanism.
    
    Key question: What EXACTLY happens at the last layer that creates gap?
    
    We know:
    - gap(h_{L-1}, Delta_W) → logit_gap: r ≈ -0.1 (no information!)
    - gap(h_L, Delta_W) → logit_gap: r ≈ 1.0 (perfect prediction!)
    
    This means the last layer TRANSFORMS h so that gap(h, Delta_W) goes from
    uninformative to perfectly predictive.
    
    The last layer consists of:
    1. Input LN
    2. Self-Attention (+ residual)
    3. Post-Attention LN
    4. MLP (+ residual)
    5. Final LN
    
    Tests:
    1. Track gap(h, Delta_W) at each sub-step within the last layer
    2. Identify which sub-step creates the gap information
    3. Is it LN's scale-invariance that "reveals" gap?
    4. Does the last layer have special weight structure?
    """
    print(f"\n{'='*60}")
    print(f"P658: Sudden Emergence Minimal Mechanism ({model_name})")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    layers = get_layers(model)
    last_layer = layers[-1]
    d_model = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 2560
    
    n_texts = 100
    test_texts = ALL_TEST_TEXTS[:n_texts]
    
    # We need to track h at EVERY sub-step within the last layer
    # The computation is:
    # h_input = LN_input(h_{L-1})
    # h_attn = h_{L-1} + Attn(h_input)
    # h_pre_mlp = LN_post_attn(h_attn)
    # h_mlp = h_attn + MLP(h_pre_mlp)
    # h_final = LN_final(h_mlp)
    
    substep_data = {
        'gap_h_Lm1': [],       # gap at h_{L-1}
        'gap_input_ln': [],    # gap after input LN
        'gap_after_attn': [],  # gap after attention + residual
        'gap_post_attn_ln': [], # gap after post-attn LN
        'gap_after_mlp': [],   # gap after MLP + residual
        'gap_final_ln': [],    # gap after final LN (= gap at h_L)
        'logit_gap': [],       # ground truth
        # Cosine with Delta_W at each step
        'cos_h_Lm1_DeltaW': [],
        'cos_after_attn_DeltaW': [],
        'cos_after_mlp_DeltaW': [],
        'cos_final_ln_DeltaW': [],
        # Norm at each step
        'norm_h_Lm1': [],
        'norm_after_attn': [],
        'norm_after_mlp': [],
        'norm_final_ln': [],
        # Delta_W projection at each step
        'proj_h_Lm1': [],
        'proj_after_attn': [],
        'proj_after_mlp': [],
        'proj_final_ln': [],
    }
    
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
        
        # Register hooks at every sub-step
        handles = []
        
        # Input LN
        for ln_name in ["input_layernorm", "ln_1", "layernorm"]:
            if hasattr(last_layer, ln_name):
                handles.append(getattr(last_layer, ln_name).register_forward_hook(make_hook('input_ln')))
                break
        
        # Attention output
        handles.append(last_layer.self_attn.register_forward_hook(make_hook('attn_out')))
        
        # Post-attention LN
        for ln_name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
            if hasattr(last_layer, ln_name):
                handles.append(getattr(last_layer, ln_name).register_forward_hook(make_hook('post_attn_ln')))
                break
        
        # MLP output
        handles.append(last_layer.mlp.register_forward_hook(make_hook('mlp_out')))
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        for h in handles:
            h.remove()
        
        # Extract
        h_Lm1 = outputs.hidden_states[-2][0, -1, :].cpu().float().numpy()
        h_L = outputs.hidden_states[-1][0, -1, :].cpu().float().numpy()
        
        logits = outputs.logits[0, -1, :].cpu().float().numpy()
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        logit_gap = logits[top1_idx] - logits[top2_idx]
        Delta_W = W_U[top1_idx] - W_U[top2_idx]
        
        # Get hook outputs
        input_ln = hook_data.get('input_ln', None)
        attn_out = hook_data.get('attn_out', None)
        post_attn_ln = hook_data.get('post_attn_ln', None)
        mlp_out = hook_data.get('mlp_out', None)
        
        # Convert to numpy
        if input_ln is not None:
            input_ln = input_ln[0, -1, :].cpu().float().numpy()
        if attn_out is not None:
            attn_out = attn_out[0, -1, :].cpu().float().numpy()
        if post_attn_ln is not None:
            post_attn_ln = post_attn_ln[0, -1, :].cpu().float().numpy()
        if mlp_out is not None:
            mlp_out = mlp_out[0, -1, :].cpu().float().numpy()
        
        # Reconstruct intermediate states
        h_after_input_ln = input_ln if input_ln is not None else h_Lm1
        
        h_after_attn = h_Lm1 + (attn_out if attn_out is not None else 0)
        
        h_after_post_attn_ln = post_attn_ln if post_attn_ln is not None else h_after_attn
        
        h_after_mlp = h_after_attn + (mlp_out if mlp_out is not None else 0)
        
        h_after_final_ln = h_L
        
        # Compute gaps at each step
        norm_DeltaW = np.linalg.norm(Delta_W)
        
        gaps = {
            'h_Lm1': np.dot(h_Lm1, Delta_W),
            'input_ln': np.dot(h_after_input_ln, Delta_W),
            'after_attn': np.dot(h_after_attn, Delta_W),
            'post_attn_ln': np.dot(h_after_post_attn_ln, Delta_W),
            'after_mlp': np.dot(h_after_mlp, Delta_W),
            'final_ln': np.dot(h_after_final_ln, Delta_W),
        }
        
        norms = {
            'h_Lm1': np.linalg.norm(h_Lm1),
            'after_attn': np.linalg.norm(h_after_attn),
            'after_mlp': np.linalg.norm(h_after_mlp),
            'final_ln': np.linalg.norm(h_after_final_ln),
        }
        
        substep_data['gap_h_Lm1'].append(gaps['h_Lm1'])
        substep_data['gap_input_ln'].append(gaps['input_ln'])
        substep_data['gap_after_attn'].append(gaps['after_attn'])
        substep_data['gap_post_attn_ln'].append(gaps['post_attn_ln'])
        substep_data['gap_after_mlp'].append(gaps['after_mlp'])
        substep_data['gap_final_ln'].append(gaps['final_ln'])
        substep_data['logit_gap'].append(logit_gap)
        
        # Cosines with Delta_W
        for name, h_vec in [('h_Lm1', h_Lm1), ('after_attn', h_after_attn), 
                           ('after_mlp', h_after_mlp), ('final_ln', h_after_final_ln)]:
            n = np.linalg.norm(h_vec)
            cos = np.dot(h_vec, Delta_W) / (n * norm_DeltaW) if n > 1e-8 and norm_DeltaW > 1e-8 else 0
            substep_data[f'cos_{name}_DeltaW'].append(float(cos))
            substep_data[f'norm_{name}'].append(float(n))
            substep_data[f'proj_{name}'].append(float(np.dot(h_vec, Delta_W)))
    
    # ===== Analysis =====
    print(f"\n1. Gap at Each Sub-Step (correlation with logit_gap):")
    
    logit_gaps = np.array(substep_data['logit_gap'])
    
    step_names = ['gap_h_Lm1', 'gap_input_ln', 'gap_after_attn', 
                  'gap_post_attn_ln', 'gap_after_mlp', 'gap_final_ln']
    step_labels = ['h_{L-1}', 'after input LN', 'after Attn', 
                   'after post-attn LN', 'after MLP', 'after final LN']
    
    for name, label in zip(step_names, step_labels):
        r, p = stats.pearsonr(substep_data[name], logit_gaps)
        print(f"   {label:20s}: r={r:.4f}, p={p:.4e}, mean_gap={np.mean(np.abs(substep_data[name])):.4f}")
    
    print(f"\n2. Gap Change at Each Sub-Step:")
    for i in range(1, len(step_names)):
        prev_gaps = np.array(substep_data[step_names[i-1]])
        curr_gaps = np.array(substep_data[step_names[i]])
        delta = curr_gaps - prev_gaps
        r_delta, _ = stats.pearsonr(delta, logit_gaps)
        print(f"   {step_labels[i]:20s} - {step_labels[i-1]:20s}: "
              f"mean_delta={np.mean(delta):.4f}, corr(delta, logit_gap)={r_delta:.4f}")
    
    print(f"\n3. Cos(h, Delta_W) at Each Sub-Step:")
    for name in ['h_Lm1', 'after_attn', 'after_mlp', 'final_ln']:
        cos_vals = np.array(substep_data[f'cos_{name}_DeltaW'])
        print(f"   cos(h_{name}, Delta_W): mean={np.mean(cos_vals):.6f}, "
              f"std={np.std(cos_vals):.6f}, |mean|={np.mean(np.abs(cos_vals)):.6f}")
    
    print(f"\n4. Norm at Each Sub-Step:")
    for name in ['h_Lm1', 'after_attn', 'after_mlp', 'final_ln']:
        norms = np.array(substep_data[f'norm_{name}'])
        print(f"   ||h_{name}||: mean={np.mean(norms):.4f}, std={np.std(norms):.4f}")
    
    # ===== KEY ANALYSIS: Which step creates the gap information? =====
    print(f"\n5. CRITICAL: Which Step Creates Gap Information?")
    
    # The step where gap(h, Delta_W) first correlates with logit_gap
    for i in range(len(step_names)):
        r, _ = stats.pearsonr(substep_data[step_names[i]], logit_gaps)
        if abs(r) > 0.5:
            print(f"   >>> FIRST high correlation at: {step_labels[i]} (r={r:.4f})")
            break
    
    # The step with the largest jump in correlation
    max_jump = 0
    max_jump_step = ""
    for i in range(1, len(step_names)):
        r_prev, _ = stats.pearsonr(substep_data[step_names[i-1]], logit_gaps)
        r_curr, _ = stats.pearsonr(substep_data[step_names[i]], logit_gaps)
        jump = abs(r_curr) - abs(r_prev)
        if jump > max_jump:
            max_jump = jump
            max_jump_step = step_labels[i]
    
    print(f"   >>> Largest correlation jump at: {max_jump_step} (|Δr|={max_jump:.4f})")
    
    return substep_data


def experiment_p659(model, tokenizer, device, model_name):
    """P659: GLM4 vs Qwen3/DS7B Information Encoding Difference.
    
    Key question: Why can GLM4 encode gap info in middle layers (Ridge LOO=0.62)
    while Qwen3/DS7B cannot (Ridge LOO<0.29)?
    
    Possibilities:
    1. GLM4's h has higher-dimensional gap encoding (needs more PCA components)
    2. GLM4's W_U structure is different (more aligned with h directions)
    3. GLM4's intermediate layers preserve more Delta_W-relevant information
    4. The encoding is in a different basis — not Delta_W direction but some rotation
    
    Tests:
    1. Effective dimensionality: how many PCA components needed for Ridge r>0.5?
    2. Alignment: cos(PCA_components, Delta_W) for GLM4 vs others
    3. SVD of W_U restricted to top PCA directions of h
    4. Cross-prediction: can GLM4's PCA directions predict Qwen3's gaps?
    """
    print(f"\n{'='*60}")
    print(f"P659: Information Encoding Analysis ({model_name})")
    print(f"{'='*60}")
    
    W_U = get_W_U(model)
    d_model = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 2560
    n_layers = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else 32
    
    n_texts = 200
    test_texts = ALL_TEST_TEXTS[:n_texts]
    
    # Collect h_{L-1} and logit_gaps
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
    
    h_Lm1_arr = np.array(h_Lm1_list)
    gaps = np.array(logit_gaps)
    Delta_W_arr = np.array(Delta_Ws)
    
    # ===== 1. Effective Dimensionality =====
    print(f"\n1. Effective Dimensionality (Ridge LOO vs k):")
    
    k_scan = [1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200]
    dim_results = []
    
    for k in k_scan:
        if k >= n_texts or k >= d_model:
            continue
        try:
            pca = PCA(n_components=k)
            h_pca = pca.fit_transform(h_Lm1_arr)
            
            # LOO-CV
            loo_preds = []
            for train_idx, test_idx in LeaveOneOut().split(h_pca):
                ridge = Ridge(alpha=1.0).fit(h_pca[train_idx], gaps[train_idx])
                loo_preds.append(ridge.predict(h_pca[test_idx])[0])
            loo_preds = np.array(loo_preds)
            r_loo, _ = stats.pearsonr(loo_preds, gaps)
            
            dim_results.append({'k': k, 'r_loo': float(r_loo)})
            print(f"   k={k:3d}: r_LOO={r_loo:.4f}")
        except:
            pass
    
    # Find effective dimension (first k where r > 0.5)
    eff_dim = None
    for dr in dim_results:
        if dr['r_loo'] > 0.5:
            eff_dim = dr['k']
            break
    
    if eff_dim is None:
        # Find k with maximum r
        best = max(dim_results, key=lambda x: x['r_loo'])
        eff_dim = best['k']
    
    print(f"   Effective dimensionality (first r>0.5): k={eff_dim}")
    
    # ===== 2. PCA-ΔW Alignment =====
    print(f"\n2. PCA Components vs Delta_W Alignment:")
    
    # Use best k from above
    pca = PCA(n_components=min(eff_dim + 10, n_texts - 1, d_model))
    h_pca = pca.fit_transform(h_Lm1_arr)
    components = pca.components_  # [k, d_model]
    
    # For each PCA component, compute cos with mean Delta_W
    mean_DeltaW = np.mean(Delta_W_arr, axis=0)
    norm_mean_DeltaW = np.linalg.norm(mean_DeltaW)
    
    print(f"   ||mean(Delta_W)|| = {norm_mean_DeltaW:.4f}")
    print(f"   Top-10 PCA component alignments with mean(Delta_W):")
    
    alignment_scores = []
    for i in range(min(10, len(components))):
        cos = np.dot(components[i], mean_DeltaW) / (np.linalg.norm(components[i]) * norm_mean_DeltaW)
        alignment_scores.append(cos)
        print(f"     PC{i}: cos={cos:.6f}, variance_explained={pca.explained_variance_ratio_[i]:.4f}")
    
    # ===== 3. Per-Direction Information =====
    print(f"\n3. Per-Text Delta_W Projection Analysis:")
    
    # For each text, project h_{L-1} onto that text's Delta_W
    projections = np.array([np.dot(h_Lm1_arr[i], Delta_W_arr[i]) for i in range(n_texts)])
    r_proj, _ = stats.pearsonr(projections, gaps)
    
    # Also try: project onto W_U[top1] and W_U[top2] separately
    # Need to re-extract top1/top2 indices
    print(f"   gap(h_{{L-1}}, Delta_W) -> logit_gap: r={r_proj:.4f}")
    
    # ===== 4. Rotated Gap Encoding =====
    print(f"\n4. Rotated Gap Encoding Test:")
    
    # Hypothesis: GLM4 encodes gap in a ROTATED basis
    # Test: instead of projecting onto Delta_W, project onto learned directions
    
    # Use Ridge coefficients as "learned directions"
    best_k = max(dim_results, key=lambda x: x['r_loo'])['k']
    pca = PCA(n_components=best_k)
    h_pca = pca.fit_transform(h_Lm1_arr)
    ridge = Ridge(alpha=1.0).fit(h_pca, gaps)
    
    # Ridge predicts: gap = sum(ridge_coef[i] * pca_component[i] · h + intercept)
    # This is equivalent to: gap = W_learned · h + intercept
    # where W_learned = sum(ridge_coef[i] * pca_component[i])
    
    W_learned = np.zeros(d_model)
    for i in range(best_k):
        W_learned += ridge.coef_[i] * pca.components_[i]
    
    # Compute learned gap
    learned_gaps = h_Lm1_arr @ W_learned + ridge.intercept_
    r_learned, _ = stats.pearsonr(learned_gaps, gaps)
    
    print(f"   Learned direction prediction: r={r_learned:.4f}")
    
    # Compare: cos(W_learned, mean_DeltaW)
    cos_learned_DeltaW = np.dot(W_learned, mean_DeltaW) / (np.linalg.norm(W_learned) * norm_mean_DeltaW)
    print(f"   cos(W_learned, mean(Delta_W)): {cos_learned_DeltaW:.4f}")
    
    # Key question: is W_learned similar to Delta_W?
    # If cos is low, the encoding is in a ROTATED basis
    # If cos is high, the encoding is along Delta_W direction
    
    print(f"\n5. Cross-Text Delta_W Variance:")
    
    # How different are Delta_W vectors across texts?
    Delta_W_norms = np.linalg.norm(Delta_W_arr, axis=1)
    print(f"   ||Delta_W||: mean={np.mean(Delta_W_norms):.4f}, std={np.std(Delta_W_norms):.4f}")
    
    # Pairwise cosines
    n_sample = min(100, n_texts)
    sample_indices = np.random.choice(n_texts, n_sample, replace=False)
    pairwise_cos = []
    for i in range(n_sample):
        for j in range(i+1, min(i+20, n_sample)):
            cos = np.dot(Delta_W_arr[sample_indices[i]], Delta_W_arr[sample_indices[j]]) / \
                  (Delta_W_norms[sample_indices[i]] * Delta_W_norms[sample_indices[j]])
            pairwise_cos.append(cos)
    
    print(f"   Pairwise cos(Delta_W_i, Delta_W_j): mean={np.mean(pairwise_cos):.6f}, "
          f"std={np.std(pairwise_cos):.6f}")
    print(f"   (Near-zero means Delta_Ws are nearly orthogonal — W_U rows are nearly orthogonal)")
    
    return {
        'effective_dimension': eff_dim,
        'r_learned': float(r_learned),
        'cos_learned_DeltaW': float(cos_learned_DeltaW),
        'r_proj_DeltaW': float(r_proj),
        'dim_results': dim_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase CXLIX: Emergence Mechanism")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["p656", "p657", "p658", "p659", "all"])
    parser.add_argument("--n_texts", type=int, default=100)
    args = parser.parse_args()
    
    model_name = args.model
    
    # Load model
    print(f"Loading {model_name}...")
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    print(f"Model: {info.name}, layers={info.n_layers}, d_model={info.d_model}")
    
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                               "results", "phase_cxlix")
    os.makedirs(results_dir, exist_ok=True)
    
    experiments = {
        "p656": experiment_p656,
        "p657": experiment_p657,
        "p658": experiment_p658,
        "p659": experiment_p659,
    }
    
    if args.experiment == "all":
        exp_list = ["p656", "p657", "p658", "p659"]
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
            
            # Save results
            # Convert numpy arrays to lists for JSON serialization
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
    
    # Release model
    release_model(model)
    print(f"\nModel released.")


if __name__ == "__main__":
    main()
