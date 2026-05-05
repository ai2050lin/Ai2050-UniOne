"""
CCML Phase 43: Causal Structure Identification — Closing the Evidence Gaps
===========================================================================

DEEP CRITIQUE from user on Phase 42:
1. S(x) is NOT an "object created by MLP" — it's the RESULT of the Jacobian chain
   S(x) = Im(J(x)), where J(x) = J_L(x) ··· J_1(x)
   MLP changes J_l, which changes the chain, which changes S(x)

2. "Syntax → S(x)" is a skipped inference. The real chain is:
   Syntax → Attention routing → Jacobian → S(x)
   We need to test whether Attention routing (not syntax per se) drives S(x)

3. Top-1 direction "stability" (cos=0.9) may be inflated by high-dimensional effects.
   Need Grassmann distance / principal angle distributions for rigorous comparison.

4. CRITICAL: S(x) being low-dimensional may be a "compression artifact" of Jacobian
   chain spectral decay, NOT learned semantic structure. If random networks also
   show dim90 << d with depth, then our entire theory is built on architectural artifact.

Phase 43 Experiments:
43A: Spectral Decay vs Depth — Is low-dim S(x) an artifact?
  - Measure dim90(S(x)) at different propagation depths (1, 2, 5, 10, 20 layers)
  - Compare trained model vs randomized model
  - If both show dim90 decreasing with depth → architectural effect
  - If only trained model shows decrease → learned structure

43B: Attention Routing → S(x) Causal Test
  - Fix input text, manually modify attention masks/patterns
  - Measure how S(x) changes with different attention routing
  - If attention routing drives S(x) → confirms Syntax→Attn→J→S(x) chain
  - If not → S(x) is determined by MLP weights, not routing

43C: S(x) → Token Prediction (Closing the Last Gap)
  - Find the basis vectors of S(x)
  - Project them through W_U to see which affect top token prediction
  - This connects "structure" to "function"

Key conceptual correction throughout:
  ❌ "MLP creates S(x)" → ✅ "MLP changes J_l, which changes the Jacobian chain,
                                which determines S(x) = Im(J(x))"
  ❌ "Syntax determines S(x)" → ✅ "Syntax → Attention routing → Jacobian → S(x)"

Usage:
  python ccml_phase43_causal_structure.py --model deepseek7b --exp 1
  python ccml_phase43_causal_structure.py --model deepseek7b --exp 2
  python ccml_phase43_causal_structure.py --model deepseek7b --exp 3
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

from model_utils import load_model, get_layers, get_model_info, release_model, safe_decode

# ===== 配置 =====
N_RANDOM_DIRS = 50

TEST_TEXTS = [
    "The cat sat on the mat",
    "The scientist discovered a new element",
    "She walked to the store yesterday",
]


def inject_and_get_hidden_at_target(model, input_ids, attention_mask, hook_target,
                                     delta, layers, target_layer_idx, token_pos=-1):
    """
    Inject delta at hook_target, return hidden state perturbation at target_layer_idx.
    Returns Δh_target (NOT logits).
    """
    base_hs = [None]
    perturbed_hs = [None]
    
    def base_hook(module, input, output):
        if isinstance(output, tuple):
            base_hs[0] = output[0][0, token_pos, :].detach().cpu().float()
        else:
            base_hs[0] = output[0, token_pos, :].detach().cpu().float()
    
    handle_base = layers[target_layer_idx].register_forward_hook(base_hook)
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    handle_base.remove()
    
    def perturb_hook(module, input, output):
        if isinstance(output, tuple):
            new_h = output[0].clone()
            new_h[0, token_pos, :] += delta.to(new_h.dtype).to(new_h.device)
            return (new_h,) + output[1:]
        new_h = output.clone()
        new_h[0, token_pos, :] += delta.to(new_h.dtype).to(new_h.device)
        return new_h
    
    def target_hook(module, input, output):
        if isinstance(output, tuple):
            perturbed_hs[0] = output[0][0, token_pos, :].detach().cpu().float()
        else:
            perturbed_hs[0] = output[0, token_pos, :].detach().cpu().float()
    
    handle_perturb = hook_target.register_forward_hook(perturb_hook)
    handle_target = layers[target_layer_idx].register_forward_hook(target_hook)
    
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    handle_perturb.remove()
    handle_target.remove()
    
    if base_hs[0] is None or perturbed_hs[0] is None:
        return None
    
    return (perturbed_hs[0] - base_hs[0]).numpy()


def compute_subspace_at_depth(model, tokenizer, device, layers, n_layers,
                               text, inject_layer, target_depth, n_dirs=50,
                               alpha=0.1, seed=42):
    """
    Compute propagation subspace S(x) for a specific propagation depth.
    
    inject_layer: where to inject perturbation
    target_depth: how many layers to propagate (1 = next layer, etc.)
    
    Returns: (Vt, s, subspace_info)
    """
    info = get_model_info(model, model_name_ref[0])
    d_model = info.d_model
    
    target_layer = min(inject_layer + target_depth, n_layers - 1)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Get hidden state norm at injection layer
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                      output_hidden_states=True)
    
    hs_norm = outputs.hidden_states[inject_layer+1][0, -1].float().norm().item()
    del outputs
    torch.cuda.empty_cache()
    
    delta_mag = alpha * hs_norm
    
    np.random.seed(seed)
    random_dirs = np.random.randn(n_dirs, d_model)
    random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)
    
    delta_h_list = []
    for di in range(n_dirs):
        delta = torch.tensor(delta_mag * random_dirs[di],
                           dtype=torch.float32, device=device)
        
        delta_h = inject_and_get_hidden_at_target(
            model, input_ids, attention_mask, layers[inject_layer],
            delta, layers, target_layer, token_pos=-1)
        
        if delta_h is not None and not np.isnan(delta_h).any():
            delta_h_list.append(delta_h)
        torch.cuda.empty_cache()
    
    if len(delta_h_list) < 5:
        return None, None, None
    
    matrix = np.array(delta_h_list)
    centered = matrix - matrix.mean(axis=0)
    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    
    total_energy = np.sum(s**2)
    if total_energy < 1e-10:
        return Vt, s, None
    
    cum_energy = np.cumsum(s**2) / total_energy
    eff_dim_90 = int(np.searchsorted(cum_energy, 0.90) + 1)
    eff_dim_50 = int(np.searchsorted(cum_energy, 0.50) + 1)
    
    subspace_info = {
        'effective_dim_90': int(eff_dim_90),
        'effective_dim_50': int(eff_dim_50),
        'top1_fraction': float(s[0]**2 / total_energy),
        'top3_fraction': float(np.sum(s[:3]**2) / total_energy),
        'top10_fraction': float(np.sum(s[:10]**2) / total_energy),
        'singular_values': s[:30].tolist(),
        'n_valid_dirs': len(delta_h_list),
        'inject_layer': inject_layer,
        'target_layer': target_layer,
        'target_depth': target_depth,
    }
    
    return Vt, s, subspace_info


def grassmann_distance(Vt1, Vt2, k=10):
    """
    Compute Grassmann distance between two subspaces.
    
    Uses principal angles: d_G = sqrt(sum(theta_i^2))
    where theta_i are principal angles.
    
    Returns:
    - grassmann_dist: Grassmann distance
    - principal_angles: array of principal angles in degrees
    - max_angle: maximum principal angle
    """
    sub1 = Vt1[:k]  # [k, d]
    sub2 = Vt2[:k]  # [k, d]
    
    # Compute principal angles using SVD of sub1 @ sub2.T
    M = sub1 @ sub2.T  # [k, k]
    U, cos_angles, Vt = np.linalg.svd(M)
    
    # cos_angles are the cosines of principal angles
    cos_angles = np.clip(cos_angles, -1, 1)
    angles_rad = np.arccos(np.abs(cos_angles))
    angles_deg = np.degrees(angles_rad)
    
    # Grassmann distance
    grassmann_dist = np.sqrt(np.sum(angles_rad**2))
    
    return float(grassmann_dist), angles_deg.tolist(), float(np.max(angles_deg))


def randomize_model_weights(model, model_name):
    """Randomize attention+MLP weights, keep norm+embed+W_U"""
    layers = get_layers(model)
    n_randomized = 0
    for layer in layers:
        for attr in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if hasattr(layer.self_attn, attr):
                w = getattr(layer.self_attn, attr)
                if hasattr(w, 'weight'):
                    torch.nn.init.normal_(w.weight.data, std=0.02)
                    n_randomized += 1
                if hasattr(w, 'bias') and w.bias is not None:
                    torch.nn.init.zeros_(w.bias.data)
        mlp = layer.mlp
        for attr in ['gate_proj', 'up_proj', 'down_proj', 'gate_up_proj',
                     'dense_h_to_4h', 'dense_4h_to_h']:
            if hasattr(mlp, attr):
                w = getattr(mlp, attr)
                if hasattr(w, 'weight'):
                    torch.nn.init.normal_(w.weight.data, std=0.02)
                    n_randomized += 1
                if hasattr(w, 'bias') and w.bias is not None:
                    torch.nn.init.zeros_(w.bias.data)
    print(f"  Randomized {n_randomized} weight matrices")
    return model


# Global reference for model name in compute_subspace_at_depth
model_name_ref = ['deepseek7b']


# ============================================================================
# 43A: Spectral Decay vs Depth — Is low-dim S(x) an artifact?
# ============================================================================

def run_43A(model_name):
    """
    THE most critical experiment:
    Does dim90(S(x)) naturally decrease with depth in RANDOM networks?
    If yes → low-dim S(x) is architectural, not learned
    If no → low-dim S(x) is a genuine training effect
    """
    print(f"\n{'='*70}")
    print(f"Phase 43A: Spectral Decay vs Depth ({model_name})")
    print(f"{'='*70}")
    print(f"CRITICAL TEST: Is low-dim S(x) an artifact of Jacobian chain spectral decay?\n")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    layers = get_layers(model)
    model_name_ref[0] = model_name
    
    inject_layer = 7  # inject at layer 7, propagate to different depths
    alpha = 0.1
    
    # Propagation depths to test
    depths = [1, 2, 5, 10, 15, 20, min(25, n_layers - inject_layer - 1)]
    depths = sorted(set([d for d in depths if inject_layer + d < n_layers]))
    
    text = TEST_TEXTS[0]  # "The cat sat on the mat"
    
    # ===== Part 1: TRAINED model =====
    print(f"=== TRAINED MODEL ===")
    print(f"Inject at L{inject_layer}, text: '{text[:40]}'\n")
    
    trained_results = {}
    for depth in depths:
        Vt, s, sinfo = compute_subspace_at_depth(
            model, tokenizer, device, layers, n_layers,
            text, inject_layer, depth, N_RANDOM_DIRS, alpha, seed=42)
        
        if sinfo is not None:
            trained_results[depth] = sinfo
            print(f"  Depth {depth:2d} → L{inject_layer+depth}: "
                  f"dim90={sinfo['effective_dim_90']:2d}, "
                  f"top1={sinfo['top1_fraction']:.3f}, "
                  f"top3={sinfo['top3_fraction']:.3f}")
        else:
            print(f"  Depth {depth:2d}: FAILED")
        
        torch.cuda.empty_cache()
    
    # ===== Part 2: RANDOMIZED model =====
    print(f"\n=== RANDOMIZING MODEL ===")
    model = randomize_model_weights(model, model_name)
    layers = get_layers(model)
    
    # Re-compute hidden state norms for randomized model
    print(f"\n=== RANDOMIZED MODEL ===")
    
    untrained_results = {}
    for depth in depths:
        Vt, s, sinfo = compute_subspace_at_depth(
            model, tokenizer, device, layers, n_layers,
            text, inject_layer, depth, N_RANDOM_DIRS, alpha, seed=42)
        
        if sinfo is not None:
            untrained_results[depth] = sinfo
            print(f"  Depth {depth:2d} → L{inject_layer+depth}: "
                  f"dim90={sinfo['effective_dim_90']:2d}, "
                  f"top1={sinfo['top1_fraction']:.3f}, "
                  f"top3={sinfo['top3_fraction']:.3f}")
        else:
            print(f"  Depth {depth:2d}: FAILED")
        
        torch.cuda.empty_cache()
    
    # ===== Comparison =====
    print(f"\n{'='*70}")
    print(f"Phase 43A SUMMARY: Spectral Decay vs Depth")
    print(f"{'='*70}")
    
    print(f"\n--- dim90(S(x)) vs Propagation Depth ---")
    print(f"{'Depth':<8} {'Trained dim90':<16} {'Random dim90':<16} {'T top1':<12} {'R top1':<12} {'Conclusion'}")
    print("-" * 80)
    
    for depth in depths:
        t = trained_results.get(depth)
        r = untrained_results.get(depth)
        
        if t and r:
            t_dim = t['effective_dim_90']
            r_dim = r['effective_dim_90']
            t_top1 = t['top1_fraction']
            r_top1 = r['top1_fraction']
            
            dim_ratio = t_dim / (r_dim + 1e-10)
            
            if dim_ratio < 0.7:
                conclusion = "★ Training compresses S(x)"
            elif dim_ratio > 1.3:
                conclusion = "Training expands S(x)"
            else:
                conclusion = "Similar dim (architectural)"
            
            print(f"{depth:<8} {t_dim:<16} {r_dim:<16} {t_top1:<12.3f} {r_top1:<12.3f} {conclusion}")
        elif t:
            print(f"{depth:<8} {t['effective_dim_90']:<16} {'N/A':<16} {t['top1_fraction']:<12.3f} {'N/A':<12} ")
        elif r:
            print(f"{depth:<8} {'N/A':<16} {r['effective_dim_90']:<16} {'N/A':<12} {r['top1_fraction']:<12.3f}")
    
    # KEY CONCLUSION
    print(f"\n--- KEY CONCLUSION ---")
    
    t_dims = [trained_results[d]['effective_dim_90'] for d in depths if d in trained_results]
    r_dims = [untrained_results[d]['effective_dim_90'] for d in depths if d in untrained_results]
    t_top1s = [trained_results[d]['top1_fraction'] for d in depths if d in trained_results]
    r_top1s = [untrained_results[d]['top1_fraction'] for d in depths if d in untrained_results]
    
    if t_dims and r_dims:
        # Check if dim90 decreases with depth in BOTH
        if len(t_dims) > 2:
            t_corr = np.corrcoef(depths[:len(t_dims)], t_dims)[0, 1]
            r_corr = np.corrcoef(depths[:len(r_dims)], r_dims)[0, 1]
            
            print(f"Correlation(depth, dim90) — Trained: {t_corr:.3f}, Random: {r_corr:.3f}")
        
        # Average dim at max depth
        max_depth = max(depths)
        if max_depth in trained_results and max_depth in untrained_results:
            t_max = trained_results[max_depth]['effective_dim_90']
            r_max = untrained_results[max_depth]['effective_dim_90']
            ratio = t_max / (r_max + 1e-10)
            
            print(f"\nAt max depth ({max_depth} layers):")
            print(f"  Trained dim90 = {t_max}")
            print(f"  Random  dim90 = {r_max}")
            print(f"  Ratio = {ratio:.2f}")
            
            if ratio < 0.6:
                print(f"\n★★★ DEFINITIVE: Training creates genuinely lower-dimensional S(x)")
                print(f"  Random network at same depth has {r_max - t_max} more dimensions")
                print(f"  Low-dim S(x) is NOT just spectral decay — it's learned structure")
            elif ratio < 0.85:
                print(f"\n★ Training moderately compresses S(x) beyond architectural effect")
                print(f"  Architectural spectral decay contributes, but training adds further compression")
            else:
                print(f"\n⚠ S(x) dimensionality is primarily ARCHITECTURAL")
                print(f"  Random network shows similar dim90 → low-dim is spectral decay artifact")
                print(f"  Training effect on subspace dimension is WEAK")
    
    # Save results
    out_data = {
        'model': model_name, 'd_model': d_model, 'n_layers': n_layers,
        'inject_layer': inject_layer, 'alpha': alpha,
        'text': text, 'depths': depths,
        'trained_results': {str(k): v for k, v in trained_results.items()},
        'untrained_results': {str(k): v for k, v in untrained_results.items()},
    }
    out_path = f'tests/glm5_temp/phase43A_{model_name}_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}")
    
    release_model(model)
    return out_data


# ============================================================================
# 43B: Attention Routing → S(x) Causal Test
# ============================================================================

def run_43B(model_name):
    """
    Test: Does attention routing causally determine S(x)?
    
    Method: Fix input, modify attention by:
    1. Shuffling attention mask (breaks position-based routing)
    2. Masking specific positions (removes information from certain tokens)
    3. Using causal mask only (prevents attending to future tokens)
    
    If S(x) changes with attention modification → routing drives S(x)
    """
    print(f"\n{'='*70}")
    print(f"Phase 43B: Attention Routing → S(x) Causal Test ({model_name})")
    print(f"{'='*70}")
    print(f"Does attention routing causally determine S(x)?\n")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    layers = get_layers(model)
    model_name_ref[0] = model_name
    
    inject_layer = min(14, n_layers - 2)
    alpha = 0.1
    
    text = "The cat sat on the mat"
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids_orig = inputs["input_ids"].to(device)
    attention_mask_orig = inputs["attention_mask"].to(device)
    seq_len = input_ids_orig.shape[1]
    
    # Attention modification conditions:
    # 1. Normal (baseline)
    # 2. Reversed mask (attend to wrong positions)
    # 3. Only-first-token mask (only attend to first token)
    # 4. Only-last-token mask (only attend to last token)
    # 5. Random mask (random attention pattern)
    
    conditions = {}
    
    # Condition 1: Normal
    conditions['normal'] = {
        'input_ids': input_ids_orig,
        'attention_mask': attention_mask_orig,
        'description': 'Normal attention',
    }
    
    # Condition 2: Shuffled input tokens (breaks position-based routing)
    shuffled_ids = input_ids_orig.clone()
    # Shuffle all tokens except first and last
    if seq_len > 2:
        perm = torch.randperm(seq_len - 2) + 1
        shuffled_ids[0, 1:-1] = input_ids_orig[0, perm]
    conditions['shuffled_tokens'] = {
        'input_ids': shuffled_ids,
        'attention_mask': attention_mask_orig,
        'description': 'Shuffled token order',
    }
    
    # Condition 3: Repeated first token (all tokens same)
    repeat_ids = input_ids_orig.clone()
    repeat_ids[0, 1:] = input_ids_orig[0, 0]  # all tokens = first token
    conditions['repeated_token'] = {
        'input_ids': repeat_ids,
        'attention_mask': attention_mask_orig,
        'description': 'All tokens = first token',
    }
    
    # Condition 4: Masked last token (hide final token from attention)
    # We can't truly mask a specific token from attention without modifying model internals,
    # but we can replace the last token with a padding token
    masked_ids = input_ids_orig.clone()
    masked_ids[0, -1] = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
    conditions['masked_last'] = {
        'input_ids': masked_ids,
        'attention_mask': attention_mask_orig,
        'description': 'Last token replaced with padding',
    }
    
    # Condition 5: Different sentence with same length
    alt_text = "She ran fast on the track"
    alt_inputs = tokenizer(alt_text, return_tensors="pt", truncation=True, max_length=64)
    # Pad/truncate to same length
    alt_ids = alt_inputs["input_ids"].to(device)
    if alt_ids.shape[1] < seq_len:
        padding = torch.full((1, seq_len - alt_ids.shape[1]), 
                           tokenizer.pad_token_id, device=device)
        alt_ids = torch.cat([alt_ids, padding], dim=1)
    elif alt_ids.shape[1] > seq_len:
        alt_ids = alt_ids[:, :seq_len]
    alt_mask = torch.ones_like(alt_ids)
    conditions['different_text'] = {
        'input_ids': alt_ids,
        'attention_mask': alt_mask,
        'description': f'Different text: "{alt_text}"',
    }
    
    # For each condition, compute S(x)
    all_Vt = {}
    all_info = {}
    
    for cond_name, cond in conditions.items():
        print(f"\n--- Condition: {cond_name} ({cond['description']}) ---")
        
        # Get hidden state norm at injection layer
        with torch.no_grad():
            outputs = model(input_ids=cond['input_ids'], 
                          attention_mask=cond['attention_mask'],
                          output_hidden_states=True)
        
        hs_norm = outputs.hidden_states[inject_layer+1][0, -1].float().norm().item()
        del outputs
        torch.cuda.empty_cache()
        
        delta_mag = alpha * hs_norm
        
        np.random.seed(42)
        random_dirs = np.random.randn(N_RANDOM_DIRS, d_model)
        random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)
        
        delta_h_list = []
        for di in range(N_RANDOM_DIRS):
            delta = torch.tensor(delta_mag * random_dirs[di],
                               dtype=torch.float32, device=device)
            
            delta_h = inject_and_get_hidden_at_target(
                model, cond['input_ids'], cond['attention_mask'],
                layers[inject_layer], delta, layers, n_layers - 1, token_pos=-1)
            
            if delta_h is not None and not np.isnan(delta_h).any():
                delta_h_list.append(delta_h)
            torch.cuda.empty_cache()
        
        if len(delta_h_list) < 5:
            print(f"  Not enough valid directions: {len(delta_h_list)}")
            continue
        
        matrix = np.array(delta_h_list)
        centered = matrix - matrix.mean(axis=0)
        U, s, Vt = np.linalg.svd(centered, full_matrices=False)
        
        total_energy = np.sum(s**2)
        cum_energy = np.cumsum(s**2) / total_energy
        eff_dim_90 = int(np.searchsorted(cum_energy, 0.90) + 1)
        
        sinfo = {
            'effective_dim_90': int(eff_dim_90),
            'top1_fraction': float(s[0]**2 / total_energy),
            'top3_fraction': float(np.sum(s[:3]**2) / total_energy),
            'top10_fraction': float(np.sum(s[:10]**2) / total_energy),
        }
        
        all_Vt[cond_name] = Vt
        all_info[cond_name] = sinfo
        
        print(f"  dim90={sinfo['effective_dim_90']}, top1={sinfo['top1_fraction']:.3f}, "
              f"top3={sinfo['top3_fraction']:.3f}")
    
    # Compare with normal using Grassmann distance
    print(f"\n{'='*70}")
    print(f"Phase 43B SUMMARY: Attention Routing → S(x)")
    print(f"{'='*70}")
    
    if 'normal' not in all_Vt:
        print("Normal condition failed, cannot compare")
        release_model(model)
        return None
    
    Vt_normal = all_Vt['normal']
    
    print(f"\n--- Grassmann Distance from Normal ---")
    print(f"{'Condition':<20} {'dim90':<8} {'top1':<10} {'G_dist':<10} {'Max_angle':<12} {'|cos(v1)|':<10}")
    print("-" * 70)
    
    for cond_name, Vt_cond in all_Vt.items():
        sinfo = all_info[cond_name]
        
        if cond_name == 'normal':
            print(f"{'normal':<20} {sinfo['effective_dim_90']:<8} {sinfo['top1_fraction']:<10.3f} "
                  f"{'0.000':<10} {'0.0':<12} {'1.000':<10}")
            continue
        
        # Grassmann distance
        g_dist, angles, max_angle = grassmann_distance(Vt_normal, Vt_cond, k=10)
        cos_v1 = abs(np.dot(Vt_normal[0], Vt_cond[0]))
        
        print(f"{cond_name:<20} {sinfo['effective_dim_90']:<8} {sinfo['top1_fraction']:<10.3f} "
              f"{g_dist:<10.3f} {max_angle:<12.1f} {cos_v1:<10.4f}")
    
    # Key interpretation
    print(f"\n--- INTERPRETATION ---")
    
    if 'shuffled_tokens' in all_info and 'normal' in all_info:
        n_dim = all_info['normal']['effective_dim_90']
        s_dim = all_info['shuffled_tokens']['effective_dim_90']
        g_dist_s, _, _ = grassmann_distance(Vt_normal, all_Vt['shuffled_tokens'], k=10)
        
        if g_dist_s > 1.0:
            print(f"★ Token order (attention routing) CAUSALLY affects S(x)")
            print(f"  Grassmann distance = {g_dist_s:.3f} (significant)")
            print(f"  → Confirms: Syntax → Attention routing → J → S(x)")
        else:
            print(f"⚠ Token order has WEAK effect on S(x)")
            print(f"  Grassmann distance = {g_dist_s:.3f} (small)")
            print(f"  → S(x) may be more determined by MLP weights than routing")
    
    if 'repeated_token' in all_info and 'normal' in all_info:
        g_dist_r, _, _ = grassmann_distance(Vt_normal, all_Vt['repeated_token'], k=10)
        
        if g_dist_r > 1.0:
            print(f"★ Token diversity CAUSALLY affects S(x)")
            print(f"  Repeated token → Grassmann distance = {g_dist_r:.3f}")
        else:
            print(f"⚠ Token diversity has WEAK effect on S(x)")
            print(f"  Repeated token → Grassmann distance = {g_dist_r:.3f}")
    
    # Save results
    out_data = {
        'model': model_name, 'd_model': d_model, 'n_layers': n_layers,
        'inject_layer': inject_layer, 'alpha': alpha,
        'text': text,
        'subspace_info': {k: v for k, v in all_info.items()},
    }
    out_path = f'tests/glm5_temp/phase43B_{model_name}_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}")
    
    release_model(model)
    return out_data


# ============================================================================
# 43C: S(x) → Token Prediction (Closing the Function Gap)
# ============================================================================

def run_43C(model_name):
    """
    Test: How does S(x) relate to token prediction?
    
    Method:
    1. Compute S(x) for an input
    2. Find the top-k basis vectors of S(x)
    3. For each basis vector, measure how much it affects the top token logit
    4. This connects "subspace structure" to "functional output"
    """
    print(f"\n{'='*70}")
    print(f"Phase 43C: S(x) → Token Prediction ({model_name})")
    print(f"{'='*70}")
    print(f"How does the propagation subspace determine token prediction?\n")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    layers = get_layers(model)
    model_name_ref[0] = model_name
    
    # Get W_U
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        W_U = model.lm_head.weight.detach().cpu().float().numpy()
    else:
        W_U = model.get_output_embeddings().weight.detach().cpu().float().numpy()
    
    inject_layer = min(14, n_layers - 2)
    alpha = 0.1
    
    for text in TEST_TEXTS[:2]:
        print(f"\n--- Text: '{text[:50]}' ---")
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # Get base logits and top tokens
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True)
        
        base_logits = outputs.logits[0, -1].cpu().float().numpy()
        top_tokens = np.argsort(base_logits)[-5:][::-1]
        top_token = top_tokens[0]
        second_token = top_tokens[1]
        margin_dir = W_U[top_token] - W_U[second_token]
        margin_dir = margin_dir / (np.linalg.norm(margin_dir) + 1e-10)
        
        hs_norm = outputs.hidden_states[inject_layer+1][0, -1].float().norm().item()
        h_L = outputs.hidden_states[-1][0, -1].cpu().float().numpy()
        del outputs
        torch.cuda.empty_cache()
        
        # Compute S(x)
        delta_mag = alpha * hs_norm
        np.random.seed(42)
        random_dirs = np.random.randn(N_RANDOM_DIRS, d_model)
        random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)
        
        delta_h_list = []
        for di in range(N_RANDOM_DIRS):
            delta = torch.tensor(delta_mag * random_dirs[di],
                               dtype=torch.float32, device=device)
            
            delta_h = inject_and_get_hidden_at_target(
                model, input_ids, attention_mask, layers[inject_layer],
                delta, layers, n_layers - 1, token_pos=-1)
            
            if delta_h is not None and not np.isnan(delta_h).any():
                delta_h_list.append(delta_h)
            torch.cuda.empty_cache()
        
        if len(delta_h_list) < 5:
            print(f"  Not enough valid directions")
            continue
        
        matrix = np.array(delta_h_list)
        centered = matrix - matrix.mean(axis=0)
        U, s, Vt = np.linalg.svd(centered, full_matrices=False)
        
        total_energy = np.sum(s**2)
        cum_energy = np.cumsum(s**2) / total_energy
        eff_dim_90 = int(np.searchsorted(cum_energy, 0.90) + 1)
        
        print(f"  S(x): dim90={eff_dim_90}, top1={s[0]**2/total_energy:.3f}")
        print(f"  Top predicted token: {safe_decode(tokenizer, top_token)} (logit={base_logits[top_token]:.2f})")
        
        # For each top basis vector of S(x), measure its functional impact
        print(f"\n  --- Basis Vectors of S(x) → Logit Impact ---")
        print(f"  {'Mode':<8} {'SV_frac':<10} {'|cos(margin)|':<14} {'Δlogit_top':<14} {'Δlogit_2nd':<14} {'Net_margin':<12}")
        
        for k in range(min(15, len(Vt))):
            v_k = Vt[k]  # [d_model] — k-th principal direction of S(x)
            sv_frac = s[k]**2 / total_energy
            
            # Alignment with margin direction
            cos_margin = abs(np.dot(v_k, margin_dir))
            
            # Logit impact: W_U @ v_k gives logit change per unit of v_k
            logit_change = W_U @ v_k  # [vocab_size]
            
            delta_logit_top = logit_change[top_token]
            delta_logit_second = logit_change[second_token]
            net_margin = delta_logit_top - delta_logit_second
            
            # Also: which token is most affected by this direction?
            most_affected_token = np.argmax(np.abs(logit_change))
            
            if k < 10:
                print(f"  v_{k:<5} {sv_frac:<10.4f} {cos_margin:<14.4f} "
                      f"{delta_logit_top:<14.4f} {delta_logit_second:<14.4f} {net_margin:<12.4f}")
        
        # Summary: how much of S(x) is "aligned" with the margin direction?
        print(f"\n  --- S(x) → Margin Alignment Summary ---")
        
        # Project margin_dir onto S(x)'s top-k subspace
        for k_test in [3, 5, 10, min(20, len(Vt))]:
            Vt_k = Vt[:k_test]  # [k, d]
            proj = Vt_k.T @ Vt_k @ margin_dir  # project margin onto S(x) top-k
            proj_frac = np.linalg.norm(proj) / (np.linalg.norm(margin_dir) + 1e-10)
            print(f"  Top-{k_test:2d} modes capture {proj_frac:.3f} of margin direction")
        
        # Which token is most affected by the DOMINANT mode of S(x)?
        v1 = Vt[0]
        logit_impact_v1 = W_U @ v1  # [vocab_size]
        top_affected = np.argsort(np.abs(logit_impact_v1))[-5:][::-1]
        
        print(f"\n  Dominant mode v_1 most affects tokens:")
        for rank, tok_id in enumerate(top_affected):
            print(f"    {rank+1}. {safe_decode(tokenizer, tok_id):>15} "
                  f"(impact={logit_impact_v1[tok_id]:.4f}, "
                  f"base_logit={base_logits[tok_id]:.2f})")
    
    # Save results
    out_data = {
        'model': model_name, 'd_model': d_model, 'n_layers': n_layers,
        'inject_layer': inject_layer, 'alpha': alpha,
    }
    out_path = f'tests/glm5_temp/phase43C_{model_name}_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}")
    
    release_model(model)
    return out_data


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 43: Causal Structure')
    parser.add_argument('--model', type=str, required=True,
                       choices=['deepseek7b', 'glm4', 'qwen3'])
    parser.add_argument('--exp', type=int, required=True, choices=[1, 2, 3],
                       help='1=43A (spectral decay), 2=43B (attn routing), 3=43C (token prediction)')
    args = parser.parse_args()

    if args.exp == 1:
        run_43A(args.model)
    elif args.exp == 2:
        run_43B(args.model)
    elif args.exp == 3:
        run_43C(args.model)
