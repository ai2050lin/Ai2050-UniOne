"""
CCML Phase 42: Conditional Propagation Subspace — Semantic Coordinate & Source Decomposition
=============================================================================================

CRITICAL CORRECTIONS from user critique of Phase 41:
1. "Attractor" is WRONG term — we measure local Jacobian propagation through finite layers,
   NOT convergence to fixed points. Correct term: "conditional propagation subspace S(x)"
2. HS-DER≈1 doesn't mean isotropic dynamics — it means margin direction is NOT in the
   principal subspace of J. We probed the wrong direction, not that there's no structure.
3. Trained vs Untrained comparison confounds distribution × Jacobian.

Unified Theory (user's correct formulation):
  Language = Conditional Subspace Dynamics × Readout Geometry

  Training learns an input-dependent low-dimensional propagation subspace S(x).
  Output is just the projection of S(x) under W_U.

Phase 42 Experiments:
42A: Subspace Semantic Coordinate Mapping
  - Use semantic contrast pairs (tense, sentiment, number)
  - Measure how S(x) rotates/changes when semantics change
  - If S(x) rotates systematically → subspace encodes semantics
  - If S(x) is stable → subspace encodes computation structure

42B: Subspace Source Decomposition (Attention vs MLP)
  - Ablate attention or MLP at a layer, re-measure S(x)
  - Determine which component creates the propagation subspace
  - If attention creates S(x) → routing-based subspace
  - If MLP creates S(x) → transformation-based subspace

42C: Subspace Stability (True Dynamical Systems Test)
  - Small perturbation to input → does S(x) change?
  - If stable → S(x) is a robust property (approaching attractor-like)
  - If unstable → S(x) is fragile, no dynamical structure
  - This is the closest we can get to "attractor" verification

Key terminology correction throughout:
  ❌ "attractor" → ✅ "propagation subspace S(x)"
  ❌ "attractor direction" → ✅ "dominant mode of S(x)"
  ❌ "convergence to attractor" → ✅ "concentration into low-dim subspace"

Usage:
  python ccml_phase42_subspace_theory.py --model deepseek7b --exp 1
  python ccml_phase42_subspace_theory.py --model deepseek7b --exp 2
  python ccml_phase42_subspace_theory.py --model deepseek7b --exp 3
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

from model_utils import load_model, get_layers, get_model_info, release_model, safe_decode

# ===== 配置 =====
N_RANDOM_DIRS = 50  # increased from 40 for better subspace resolution

# Semantic contrast pairs for 42A
SEMANTIC_PAIRS = {
    'tense': [
        ("The cat walks to the park", "The cat walked to the park"),
        ("She reads the book every day", "She read the book yesterday"),
        ("He runs quickly home", "He ran quickly home"),
    ],
    'number': [
        ("The cat sits on the mat", "The cats sit on the mat"),
        ("A dog barks loudly", "Dogs bark loudly"),
        ("The bird flies south", "The birds fly south"),
    ],
    'sentiment': [
        ("The movie was excellent and thrilling", "The movie was terrible and boring"),
        ("She felt happy about the result", "She felt sad about the result"),
        ("The food was delicious and warm", "The food was awful and cold"),
    ],
}


def inject_and_get_final_hidden(model, input_ids, attention_mask, hook_target,
                                 delta, layers, n_layers, token_pos=-1):
    """Same as Phase 41 — inject delta, return Δh_L (hidden state, NOT logits)"""
    base_hs_final = [None]
    perturbed_hs_final = [None]
    
    def base_hook(module, input, output):
        if isinstance(output, tuple):
            base_hs_final[0] = output[0][0, token_pos, :].detach().cpu().float()
        else:
            base_hs_final[0] = output[0, token_pos, :].detach().cpu().float()
    
    handle_base = layers[-1].register_forward_hook(base_hook)
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
    
    def final_hook(module, input, output):
        if isinstance(output, tuple):
            perturbed_hs_final[0] = output[0][0, token_pos, :].detach().cpu().float()
        else:
            perturbed_hs_final[0] = output[0, token_pos, :].detach().cpu().float()
    
    handle_perturb = hook_target.register_forward_hook(perturb_hook)
    handle_final = layers[-1].register_forward_hook(final_hook)
    
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    handle_perturb.remove()
    handle_final.remove()
    
    if base_hs_final[0] is None or perturbed_hs_final[0] is None:
        return None, None
    
    delta_h_L = perturbed_hs_final[0] - base_hs_final[0]
    return delta_h_L.numpy(), base_hs_final[0].numpy()


def compute_propagation_subspace(model, tokenizer, device, layers, n_layers,
                                  text, inject_layer, n_dirs=50, alpha=0.1,
                                  seed=42):
    """
    Compute the propagation subspace S(x) for a given input text.
    
    Returns:
    - Vt: [min(n_dirs, d), d_model] — principal directions (right singular vectors)
    - singular_values: array of singular values
    - subspace_info: dict with dim90, top1_fraction, etc.
    """
    d_model = layers[0].output_features if hasattr(layers[0], 'output_features') else \
              next(p for n, p in layers[0].named_parameters() if 'weight' in n).shape[0]
    # Get d_model from the layer's output dimension
    try:
        d_model = layers[0].mlp.down_proj.out_features
    except:
        try:
            d_model = layers[0].self_attn.o_proj.out_features
        except:
            d_model = 3584  # fallback
    
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
    
    # Generate random probe directions
    np.random.seed(seed)
    random_dirs = np.random.randn(n_dirs, d_model)
    random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)
    
    # Propagate each direction
    delta_h_list = []
    for di in range(n_dirs):
        delta = torch.tensor(delta_mag * random_dirs[di],
                           dtype=torch.float32, device=device)
        
        delta_h_L, _ = inject_and_get_final_hidden(
            model, input_ids, attention_mask, layers[inject_layer],
            delta, layers, n_layers, token_pos=-1)
        
        if delta_h_L is not None and not np.isnan(delta_h_L).any():
            delta_h_list.append(delta_h_L)
        torch.cuda.empty_cache()
    
    if len(delta_h_list) < 5:
        return None, None, None
    
    matrix = np.array(delta_h_list)  # [n_dirs, d_model]
    centered = matrix - matrix.mean(axis=0)
    
    # SVD
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
        'cumulative_energy': cum_energy[:30].tolist(),
        'n_valid_dirs': len(delta_h_list),
    }
    
    return Vt, s, subspace_info


def subspace_overlap(Vt1, s1, Vt2, s2, k=10):
    """
    Compute overlap between two subspaces using their principal directions.
    
    Vt1, Vt2: [rank, d_model] — right singular vectors
    k: number of top components to use
    
    Returns:
    - overlap: fraction of Vt1's energy captured by Vt2's top-k subspace
    - principal_cos: |cos(v1_1, v2_1)| — alignment of top directions
    """
    sub1 = Vt1[:k]  # [k, d_model]
    sub2 = Vt2[:k]  # [k, d_model]
    
    # Project sub1 onto sub2's subspace
    # For each vector in sub1, project onto sub2
    proj = sub1 @ sub2.T @ sub2  # [k, d_model]
    proj_energy = np.sum(proj**2)
    total_energy = np.sum(sub1**2)
    overlap = proj_energy / (total_energy + 1e-10)
    
    # Principal direction alignment
    principal_cos = abs(np.dot(Vt1[0], Vt2[0]))
    
    return float(overlap), float(principal_cos)


def principal_subspace_rotation(Vt1, Vt2, k=5):
    """
    Measure how much the top-k subspace rotates between two conditions.
    
    Returns rotation angle in degrees (0 = identical, 90 = orthogonal).
    Uses the Frobenius norm of the difference between projection matrices.
    """
    P1 = Vt1[:k].T @ Vt1[:k]  # [d, d] projection matrix
    P2 = Vt2[:k].T @ Vt2[:k]  # [d, d] projection matrix
    
    # Subspace distance: ||P1 - P2||_F / sqrt(2k)
    diff = np.linalg.norm(P1 - P2, 'fro') / np.sqrt(2 * k)
    
    # Convert to angle-like measure: 0 = same, 1 = orthogonal
    # diff ranges from 0 (identical) to 1 (orthogonal)
    angle_rad = np.arcsin(min(diff, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    return float(angle_deg), float(diff)


# ============================================================================
# 42A: Subspace Semantic Coordinate Mapping
# ============================================================================

def run_42A(model_name):
    """
    Test: Does the propagation subspace S(x) encode semantic information?
    
    Method: Compare S(x) across semantic contrast pairs.
    - If S(x) rotates with semantic change → subspace encodes semantics
    - If S(x) is stable → subspace encodes computation structure
    """
    print(f"\n{'='*70}")
    print(f"Phase 42A: Subspace Semantic Coordinate Mapping ({model_name})")
    print(f"{'='*70}")
    print(f"Question: Does S(x) rotate when semantics change?\n")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    layers = get_layers(model)
    
    inject_layer = min(14, n_layers - 2)  # middle layer
    alpha = 0.1
    
    all_results = {}
    
    for pair_type, pairs in SEMANTIC_PAIRS.items():
        print(f"\n{'='*50}")
        print(f"Semantic Contrast: {pair_type.upper()}")
        print(f"{'='*50}")
        
        pair_results = []
        
        for pi, (text_A, text_B) in enumerate(pairs):
            print(f"\n  Pair {pi+1}:")
            print(f"    A: '{text_A}'")
            print(f"    B: '{text_B}'")
            
            # Compute S(x) for each text
            Vt_A, s_A, info_A = compute_propagation_subspace(
                model, tokenizer, device, layers, n_layers,
                text_A, inject_layer, N_RANDOM_DIRS, alpha, seed=42)
            
            Vt_B, s_B, info_B = compute_propagation_subspace(
                model, tokenizer, device, layers, n_layers,
                text_B, inject_layer, N_RANDOM_DIRS, alpha, seed=42)
            
            if info_A is None or info_B is None:
                print(f"    FAILED to compute subspace")
                continue
            
            # Measure subspace rotation
            overlap_10, cos_top1 = subspace_overlap(Vt_A, s_A, Vt_B, s_B, k=10)
            overlap_5, _ = subspace_overlap(Vt_A, s_A, Vt_B, s_B, k=5)
            overlap_3, _ = subspace_overlap(Vt_A, s_A, Vt_B, s_B, k=3)
            angle_deg, subspace_dist = principal_subspace_rotation(Vt_A, Vt_B, k=5)
            
            # Also measure pairwise cosines for top-5 directions
            top5_cosines = []
            for i in range(min(5, len(Vt_A), len(Vt_B))):
                top5_cosines.append(float(abs(np.dot(Vt_A[i], Vt_B[i]))))
            
            print(f"    A: dim90={info_A['effective_dim_90']}, top1={info_A['top1_fraction']:.3f}")
            print(f"    B: dim90={info_B['effective_dim_90']}, top1={info_B['top1_fraction']:.3f}")
            print(f"    Overlap(top-10): {overlap_10:.3f}")
            print(f"    Overlap(top-5):  {overlap_5:.3f}")
            print(f"    Overlap(top-3):  {overlap_3:.3f}")
            print(f"    |cos(v1_A, v1_B)|: {cos_top1:.4f}")
            print(f"    Subspace rotation: {angle_deg:.1f}°")
            print(f"    Top-5 diagonal cosines: {[f'{c:.3f}' for c in top5_cosines]}")
            
            pair_results.append({
                'text_A': text_A,
                'text_B': text_B,
                'info_A': info_A,
                'info_B': info_B,
                'overlap_10': overlap_10,
                'overlap_5': overlap_5,
                'overlap_3': overlap_3,
                'cos_top1': cos_top1,
                'rotation_deg': angle_deg,
                'subspace_dist': subspace_dist,
                'top5_cosines': top5_cosines,
            })
            
            gc.collect()
            torch.cuda.empty_cache()
        
        all_results[pair_type] = pair_results
    
    # Summary across all pairs
    print(f"\n{'='*70}")
    print(f"Phase 42A SUMMARY: Semantic Coordinate Mapping")
    print(f"{'='*70}")
    
    print(f"\n--- Subspace Rotation by Semantic Type ---")
    print(f"{'Type':<12} {'Avg Overlap(10)':<16} {'Avg |cos(v1)|':<16} {'Avg Rotation':<14} {'Interpretation'}")
    print("-" * 80)
    
    for pair_type, results in all_results.items():
        if not results:
            continue
        
        avg_overlap = np.mean([r['overlap_10'] for r in results])
        avg_cos = np.mean([r['cos_top1'] for r in results])
        avg_rot = np.mean([r['rotation_deg'] for r in results])
        
        if avg_cos > 0.9:
            interp = "S(x) STABLE → computation structure"
        elif avg_cos > 0.5:
            interp = "S(x) PARTIALLY rotates → mixed"
        else:
            interp = "S(x) ROTATES → encodes semantics"
        
        print(f"{pair_type:<12} {avg_overlap:<16.3f} {avg_cos:<16.4f} {avg_rot:<14.1f} {interp}")
    
    # Compare: within-semantic-type vs across-types
    # For each type, compute average rotation
    print(f"\n--- Key Question: Does S(x) encode semantic content? ---")
    
    all_overlaps = [r['overlap_10'] for results in all_results.values() for r in results]
    all_cosines = [r['cos_top1'] for results in all_results.values() for r in results]
    all_rotations = [r['rotation_deg'] for results in all_results.values() for r in results]
    
    avg_overlap = np.mean(all_overlaps)
    avg_cos = np.mean(all_cosines)
    avg_rot = np.mean(all_rotations)
    
    print(f"Average overlap (top-10): {avg_overlap:.3f}")
    print(f"Average |cos(v1_A, v1_B)|: {avg_cos:.4f}")
    print(f"Average rotation angle: {avg_rot:.1f}°")
    
    if avg_cos > 0.8:
        print(f"\n★ CONCLUSION: S(x) is SEMANTICALLY STABLE")
        print(f"  Top propagation direction doesn't change with semantics")
        print(f"  → S(x) encodes COMPUTATION STRUCTURE, not semantic content")
        print(f"  → Different inputs use the same 'computational pathway'")
    elif avg_cos > 0.4:
        print(f"\n★ CONCLUSION: S(x) is PARTIALLY SEMANTIC")
        print(f"  Some modes change with semantics, some are stable")
        print(f"  → Mixed: computation structure + semantic encoding")
    else:
        print(f"\n★ CONCLUSION: S(x) is SEMANTICALLY DEPENDENT")
        print(f"  Top direction changes significantly with semantics")
        print(f"  → S(x) encodes SEMANTIC CONTENT directly")
    
    # Also test: do SAME-SEMANTIC different-surface-forms share subspace?
    # (This tests whether subspace is about meaning or surface form)
    print(f"\n--- Within-type variation ---")
    for pair_type, results in all_results.items():
        if len(results) >= 2:
            # Compare pair1_A with pair2_A (same semantic pole, different text)
            Vt_sets = {}
            for pi, r in enumerate(results):
                text = r['text_A']
                Vt, s, _ = compute_propagation_subspace(
                    model, tokenizer, device, layers, n_layers,
                    text, inject_layer, N_RANDOM_DIRS, alpha, seed=42)
                Vt_sets[f"{pair_type}_A_{pi}"] = (Vt, s)
                torch.cuda.empty_cache()
            
            # Cross-pair overlap for same semantic pole
            keys = list(Vt_sets.keys())
            if len(keys) >= 2:
                cross_cos = abs(np.dot(Vt_sets[keys[0]][0][0], Vt_sets[keys[1]][0][0]))
                cross_overlap, _ = subspace_overlap(Vt_sets[keys[0]][0], Vt_sets[keys[0]][1],
                                                    Vt_sets[keys[1]][0], Vt_sets[keys[1]][1], k=10)
                print(f"  {pair_type}: same-pole cross-pair |cos(v1)|={cross_cos:.4f}, "
                      f"overlap(10)={cross_overlap:.3f}")
    
    # Save results
    out_data = {
        'model': model_name, 'd_model': d_model, 'n_layers': n_layers,
        'inject_layer': inject_layer, 'alpha': alpha,
        'n_random_dirs': N_RANDOM_DIRS,
        'results': all_results,
        'summary': {
            'avg_overlap_10': float(avg_overlap),
            'avg_cos_top1': float(avg_cos),
            'avg_rotation_deg': float(avg_rot),
        }
    }
    out_path = f'tests/glm5_temp/phase42A_{model_name}_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}")
    
    release_model(model)
    return out_data


# ============================================================================
# 42B: Subspace Source Decomposition (Attention vs MLP)
# ============================================================================

def run_42B(model_name):
    """
    Test: Which component (attention or MLP) creates the propagation subspace?
    
    Method: Zero-out attention or MLP output at a layer, re-measure S(x).
    - If attention creates S(x) → routing-based subspace selection
    - If MLP creates S(x) → transformation-based subspace creation
    """
    print(f"\n{'='*70}")
    print(f"Phase 42B: Subspace Source Decomposition ({model_name})")
    print(f"{'='*70}")
    print(f"Question: Does attention or MLP create S(x)?\n")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    layers = get_layers(model)
    
    inject_layer = min(14, n_layers - 2)
    alpha = 0.1
    
    test_texts = [
        "The cat sat on the mat",
        "The scientist discovered a new element",
    ]
    
    # For each text, compute S(x) under 3 conditions:
    # 1. Normal (baseline)
    # 2. Zero-attention at layers after injection
    # 3. Zero-MLP at layers after injection
    
    conditions = ['normal', 'zero_attn', 'zero_mlp']
    
    all_results = {}
    
    for text in test_texts:
        print(f"\n--- Text: '{text[:50]}' ---")
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        text_results = {}
        
        for condition in conditions:
            print(f"\n  Condition: {condition}")
            
            # Register hooks for ablation
            hooks = []
            
            if condition == 'zero_attn':
                # Zero out attention output at layers after injection
                for li in range(inject_layer, n_layers):
                    layer = layers[li]
                    attn = layer.self_attn
                    
                    def zero_hook(module, input, output):
                        if isinstance(output, tuple):
                            return (torch.zeros_like(output[0]),) + output[1:]
                        return torch.zeros_like(output)
                    
                    # Hook on attention output (after o_proj)
                    if hasattr(attn, 'o_proj'):
                        hooks.append(attn.o_proj.register_forward_hook(zero_hook))
                    # For models where attention output is directly from the attention module
                    # We need a different approach — hook the residual after attention
                    # Actually, let's hook the attention module's output and zero the residual contribution
                    # This is tricky because the residual addition happens inside the layer
                    pass
                
                # Better approach: hook the layer's self_attn module and zero its output contribution
                hooks = []
                for li in range(inject_layer, n_layers):
                    layer = layers[li]
                    
                    # We need to intercept the self-attention output and zero it
                    # The challenge is that in most transformer implementations,
                    # the residual connection is inside the layer forward method
                    
                    # Strategy: hook into the layer's forward and zero attention's contribution
                    # For simplicity, we'll modify the approach:
                    # Instead of hooking, we'll temporarily replace the self_attn's forward
                    
                    # Actually, let's use a cleaner approach:
                    # Hook the residual stream right after attention addition but before MLP
                    
                    # This is model-specific. Let me use a simpler method:
                    # Replace the entire attention output with zeros by hooking o_proj
                    
                    if hasattr(layer.self_attn, 'o_proj'):
                        def make_zero_hook():
                            def zero_hook(module, input, output):
                                return torch.zeros_like(output)
                            return zero_hook
                        hooks.append(layer.self_attn.o_proj.register_forward_hook(make_zero_hook()))
                    
            elif condition == 'zero_mlp':
                hooks = []
                for li in range(inject_layer, n_layers):
                    layer = layers[li]
                    
                    # Zero out MLP output
                    if hasattr(layer.mlp, 'down_proj'):
                        def make_zero_hook():
                            def zero_hook(module, input, output):
                                return torch.zeros_like(output)
                            return zero_hook
                        hooks.append(layer.mlp.down_proj.register_forward_hook(make_zero_hook()))
            
            # Compute S(x) under this condition
            np.random.seed(42)
            random_dirs = np.random.randn(N_RANDOM_DIRS, d_model)
            random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)
            
            # Get hidden state norm at injection layer
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=True)
            
            if outputs.hidden_states[inject_layer+1] is not None:
                hs_norm = outputs.hidden_states[inject_layer+1][0, -1].float().norm().item()
            else:
                hs_norm = 10.0  # fallback
            
            del outputs
            torch.cuda.empty_cache()
            
            delta_mag = alpha * hs_norm
            
            delta_h_list = []
            for di in range(N_RANDOM_DIRS):
                delta = torch.tensor(delta_mag * random_dirs[di],
                                   dtype=torch.float32, device=device)
                
                delta_h_L, _ = inject_and_get_final_hidden(
                    model, input_ids, attention_mask, layers[inject_layer],
                    delta, layers, n_layers, token_pos=-1)
                
                if delta_h_L is not None and not np.isnan(delta_h_L).any():
                    delta_h_list.append(delta_h_L)
                torch.cuda.empty_cache()
            
            # Remove hooks
            for h in hooks:
                h.remove()
            
            if len(delta_h_list) < 5:
                print(f"    Not enough valid directions: {len(delta_h_list)}")
                text_results[condition] = None
                continue
            
            matrix = np.array(delta_h_list)
            centered = matrix - matrix.mean(axis=0)
            U, s, Vt = np.linalg.svd(centered, full_matrices=False)
            
            total_energy = np.sum(s**2)
            if total_energy < 1e-10:
                text_results[condition] = None
                continue
            
            cum_energy = np.cumsum(s**2) / total_energy
            eff_dim_90 = int(np.searchsorted(cum_energy, 0.90) + 1)
            
            subspace_info = {
                'effective_dim_90': int(eff_dim_90),
                'top1_fraction': float(s[0]**2 / total_energy),
                'top3_fraction': float(np.sum(s[:3]**2) / total_energy),
                'top10_fraction': float(np.sum(s[:10]**2) / total_energy),
                'n_valid_dirs': len(delta_h_list),
            }
            
            # Store Vt for cross-condition comparison
            text_results[condition] = {
                'subspace': subspace_info,
                'Vt_top10': Vt[:10].tolist(),  # for overlap computation
            }
            
            print(f"    dim90={subspace_info['effective_dim_90']}, "
                  f"top1={subspace_info['top1_fraction']:.3f}, "
                  f"top3={subspace_info['top3_fraction']:.3f}")
        
        all_results[text] = text_results
    
    # Summary: compare conditions
    print(f"\n{'='*70}")
    print(f"Phase 42B SUMMARY: Subspace Source Decomposition")
    print(f"{'='*70}")
    
    print(f"\n--- Subspace Properties by Condition ---")
    print(f"{'Condition':<12} {'Avg dim90':<12} {'Avg top1':<12} {'Avg top3':<12} {'Interpretation'}")
    print("-" * 70)
    
    for condition in conditions:
        infos = [r[condition]['subspace'] for r in all_results.values() 
                 if r.get(condition) and r[condition] is not None]
        if not infos:
            continue
        
        avg_dim90 = np.mean([i['effective_dim_90'] for i in infos])
        avg_top1 = np.mean([i['top1_fraction'] for i in infos])
        avg_top3 = np.mean([i['top3_fraction'] for i in infos])
        
        print(f"{condition:<12} {avg_dim90:<12.1f} {avg_top1:<12.3f} {avg_top3:<12.3f}")
    
    # Cross-condition subspace overlap
    print(f"\n--- Subspace Overlap: Normal vs Ablated ---")
    for text in test_texts:
        r = all_results.get(text, {})
        if not all(r.get(c) for c in conditions):
            continue
        
        Vt_normal = np.array(r['normal']['Vt_top10'])
        Vt_zero_attn = np.array(r['zero_attn']['Vt_top10'])
        Vt_zero_mlp = np.array(r['zero_mlp']['Vt_top10'])
        
        # Normal vs Zero-Attention
        cos_attn = abs(np.dot(Vt_normal[0], Vt_zero_attn[0]))
        overlap_attn, _ = subspace_overlap(Vt_normal, np.ones(10), Vt_zero_attn, np.ones(10), k=10)
        
        # Normal vs Zero-MLP
        cos_mlp = abs(np.dot(Vt_normal[0], Vt_zero_mlp[0]))
        overlap_mlp, _ = subspace_overlap(Vt_normal, np.ones(10), Vt_zero_mlp, np.ones(10), k=10)
        
        print(f"  '{text[:30]}':")
        print(f"    Normal vs Zero-Attn: |cos(v1)|={cos_attn:.4f}, overlap(10)={overlap_attn:.3f}")
        print(f"    Normal vs Zero-MLP:  |cos(v1)|={cos_mlp:.4f}, overlap(10)={overlap_mlp:.3f}")
        
        if cos_attn > cos_mlp:
            print(f"    → MLP more important for S(x) structure (zeroing MLP changes subspace more)")
        else:
            print(f"    → Attention more important for S(x) structure (zeroing Attn changes subspace more)")
    
    # Key conclusion
    print(f"\n--- KEY CONCLUSION ---")
    normal_infos = [r['normal']['subspace'] for r in all_results.values() 
                    if r.get('normal')]
    attn_infos = [r['zero_attn']['subspace'] for r in all_results.values() 
                  if r.get('zero_attn')]
    mlp_infos = [r['zero_mlp']['subspace'] for r in all_results.values() 
                 if r.get('zero_mlp')]
    
    if normal_infos and attn_infos and mlp_infos:
        n_dim = np.mean([i['effective_dim_90'] for i in normal_infos])
        a_dim = np.mean([i['effective_dim_90'] for i in attn_infos])
        m_dim = np.mean([i['effective_dim_90'] for i in mlp_infos])
        
        n_top1 = np.mean([i['top1_fraction'] for i in normal_infos])
        a_top1 = np.mean([i['top1_fraction'] for i in attn_infos])
        m_top1 = np.mean([i['top1_fraction'] for i in mlp_infos])
        
        print(f"Normal:     dim90={n_dim:.1f}, top1={n_top1:.3f}")
        print(f"Zero-Attn:  dim90={a_dim:.1f}, top1={a_top1:.3f}")
        print(f"Zero-MLP:   dim90={m_dim:.1f}, top1={m_top1:.3f}")
        
        if a_top1 < n_top1 * 0.5:
            print("→ Attention is CRITICAL for subspace concentration")
        if m_top1 < n_top1 * 0.5:
            print("→ MLP is CRITICAL for subspace concentration")
        
        if a_dim > n_dim * 1.5:
            print("→ Without attention, subspace becomes high-dimensional (isotropic)")
            print("  ★ Attention CREATES the low-dimensional propagation subspace")
        elif m_dim > n_dim * 1.5:
            print("→ Without MLP, subspace becomes high-dimensional (isotropic)")
            print("  ★ MLP CREATES the low-dimensional propagation subspace")
        else:
            print("→ Both components contribute to subspace structure")
    
    # Save results
    out_data = {
        'model': model_name, 'd_model': d_model, 'n_layers': n_layers,
        'inject_layer': inject_layer, 'alpha': alpha,
        'n_random_dirs': N_RANDOM_DIRS,
        'results': {k: v for k, v in all_results.items()},
    }
    out_path = f'tests/glm5_temp/phase42B_{model_name}_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}")
    
    release_model(model)
    return out_data


# ============================================================================
# 42C: Subspace Stability — True Dynamical Systems Test
# ============================================================================

def run_42C(model_name):
    """
    Test: Is the propagation subspace S(x) stable under small input perturbations?
    
    This is the closest we can get to "attractor" verification:
    - Small perturbation to input tokens → does S(x) change?
    - If S(x) is stable → the system has genuine dynamical structure
    - If S(x) is fragile → no robust dynamical structure
    
    Method:
    1. Compute S(x) for original input
    2. Perturb input (change one token, add one word, paraphrase slightly)
    3. Compute S(x') for perturbed input
    4. Measure subspace stability
    
    Also test: norm-scale stability (scale hidden state, check S(x))
    """
    print(f"\n{'='*70}")
    print(f"Phase 42C: Subspace Stability Test ({model_name})")
    print(f"{'='*70}")
    print(f"Question: Is S(x) stable under small perturbations?\n")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    layers = get_layers(model)
    
    inject_layer = min(14, n_layers - 2)
    alpha = 0.1
    
    # Perturbation types
    test_cases = [
        {
            'name': 'token_swap',
            'base': "The cat sat on the mat",
            'perturbed': "The dog sat on the mat",
            'pert_type': 'single token change (cat→dog)',
        },
        {
            'name': 'determiner',
            'base': "The cat sat on the mat",
            'perturbed': "A cat sat on the mat",
            'pert_type': 'determiner change (The→A)',
        },
        {
            'name': 'position',
            'base': "The cat sat on the mat",
            'perturbed': "On the mat sat the cat",
            'pert_type': 'word order change',
        },
        {
            'name': 'addition',
            'base': "The cat sat on the mat",
            'perturbed': "The black cat sat on the mat",
            'pert_type': 'word addition (black)',
        },
        {
            'name': 'synonym',
            'base': "She walked to the store",
            'perturbed': "She went to the shop",
            'pert_type': 'synonym substitution',
        },
    ]
    
    all_results = {}
    
    for tc in test_cases:
        print(f"\n--- {tc['name']}: {tc['pert_type']} ---")
        print(f"  Base:     '{tc['base']}'")
        print(f"  Perturbed: '{tc['perturbed']}'")
        
        # Compute S(x) for base
        Vt_base, s_base, info_base = compute_propagation_subspace(
            model, tokenizer, device, layers, n_layers,
            tc['base'], inject_layer, N_RANDOM_DIRS, alpha, seed=42)
        
        # Compute S(x') for perturbed
        Vt_pert, s_pert, info_pert = compute_propagation_subspace(
            model, tokenizer, device, layers, n_layers,
            tc['perturbed'], inject_layer, N_RANDOM_DIRS, alpha, seed=42)
        
        if info_base is None or info_pert is None:
            print(f"  FAILED to compute subspace")
            all_results[tc['name']] = None
            continue
        
        # Measure stability
        overlap_10, cos_top1 = subspace_overlap(Vt_base, s_base, Vt_pert, s_pert, k=10)
        overlap_5, _ = subspace_overlap(Vt_base, s_base, Vt_pert, s_pert, k=5)
        overlap_3, _ = subspace_overlap(Vt_base, s_base, Vt_pert, s_pert, k=3)
        angle_deg, subspace_dist = principal_subspace_rotation(Vt_base, Vt_pert, k=5)
        
        # Top-5 direction alignment
        top5_cos = [float(abs(np.dot(Vt_base[i], Vt_pert[i]))) for i in range(min(5, len(Vt_base), len(Vt_pert)))]
        
        print(f"  Base:  dim90={info_base['effective_dim_90']}, top1={info_base['top1_fraction']:.3f}")
        print(f"  Pert:  dim90={info_pert['effective_dim_90']}, top1={info_pert['top1_fraction']:.3f}")
        print(f"  Overlap(10): {overlap_10:.3f}, |cos(v1)|: {cos_top1:.4f}")
        print(f"  Rotation: {angle_deg:.1f}°")
        print(f"  Top-5 cos: {[f'{c:.3f}' for c in top5_cos]}")
        
        all_results[tc['name']] = {
            'info_base': info_base,
            'info_pert': info_pert,
            'overlap_10': overlap_10,
            'overlap_5': overlap_5,
            'overlap_3': overlap_3,
            'cos_top1': cos_top1,
            'rotation_deg': angle_deg,
            'subspace_dist': subspace_dist,
            'top5_cosines': top5_cos,
        }
        
        torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Phase 42C SUMMARY: Subspace Stability")
    print(f"{'='*70}")
    
    print(f"\n--- Stability by Perturbation Type ---")
    print(f"{'Perturbation':<25} {'Overlap(10)':<14} {'|cos(v1)|':<12} {'Rotation':<12} {'Stability'}")
    print("-" * 80)
    
    valid_results = {k: v for k, v in all_results.items() if v is not None}
    
    for name, r in valid_results.items():
        if r['cos_top1'] > 0.9:
            stability = "VERY STABLE"
        elif r['cos_top1'] > 0.7:
            stability = "STABLE"
        elif r['cos_top1'] > 0.4:
            stability = "MODERATE"
        else:
            stability = "UNSTABLE"
        
        tc = [t for t in test_cases if t['name'] == name][0]
        print(f"{tc['pert_type']:<25} {r['overlap_10']:<14.3f} {r['cos_top1']:<12.4f} "
              f"{r['rotation_deg']:<12.1f} {stability}")
    
    # Overall stability
    if valid_results:
        avg_cos = np.mean([r['cos_top1'] for r in valid_results.values()])
        avg_overlap = np.mean([r['overlap_10'] for r in valid_results.values()])
        
        print(f"\n--- OVERALL STABILITY ---")
        print(f"Average |cos(v1)|: {avg_cos:.4f}")
        print(f"Average overlap(10): {avg_overlap:.3f}")
        
        if avg_cos > 0.8:
            print(f"\n★ S(x) is HIGHLY STABLE under input perturbations")
            print(f"  This is genuine dynamical structure — approaching attractor-like behavior")
            print(f"  The propagation subspace is a robust property of the model's dynamics")
            print(f"  ★ Language = stable conditional propagation subspace × readout")
        elif avg_cos > 0.5:
            print(f"\n★ S(x) has MODERATE stability")
            print(f"  Core structure is preserved, but details change")
            print(f"  → Some directions are robust, some are input-sensitive")
        else:
            print(f"\n★ S(x) is FRAGILE")
            print(f"  Small perturbations change the propagation subspace significantly")
            print(f"  → No robust dynamical structure, subspace is input-dependent")
    
    # Save results
    out_data = {
        'model': model_name, 'd_model': d_model, 'n_layers': n_layers,
        'inject_layer': inject_layer, 'alpha': alpha,
        'n_random_dirs': N_RANDOM_DIRS,
        'results': all_results,
        'summary': {
            'avg_cos_top1': float(np.mean([r['cos_top1'] for r in valid_results.values()])) if valid_results else 0,
            'avg_overlap_10': float(np.mean([r['overlap_10'] for r in valid_results.values()])) if valid_results else 0,
        }
    }
    out_path = f'tests/glm5_temp/phase42C_{model_name}_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}")
    
    release_model(model)
    return out_data


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 42: Subspace Theory')
    parser.add_argument('--model', type=str, required=True,
                       choices=['deepseek7b', 'glm4', 'qwen3'])
    parser.add_argument('--exp', type=int, required=True, choices=[1, 2, 3],
                       help='1=42A (semantic coords), 2=42B (attn vs mlp), 3=42C (stability)')
    args = parser.parse_args()

    if args.exp == 1:
        run_42A(args.model)
    elif args.exp == 2:
        run_42B(args.model)
    elif args.exp == 3:
        run_42C(args.model)
