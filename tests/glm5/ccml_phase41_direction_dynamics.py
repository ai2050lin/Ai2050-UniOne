"""
CCML Phase 41: Direction Dynamics and Attractor Structure
=========================================================

CRITICAL CORRECTION from Phase 40 critique:
Phase 40 showed DER is dominated by W_U readout geometry.
This does NOT mean dynamics is unimportant — it means DER is the
wrong metric for dynamics.

The correct framework:
  Effect = Dynamics(J) × Readout Geometry(W_U)
  
  Phase 40 measured: Effect (via DER)
  Phase 40 showed: Readout Geometry dominates DER
  Phase 40 did NOT show: Dynamics is unimportant

This phase measures DYNAMICS DIRECTLY, bypassing W_U readout:
  - Measure Δh_L (hidden state perturbation at final layer)
  - NOT Δlogits (which goes through W_U)

Key questions:
1. Does the Jacobian converge directions to low-dimensional subspaces?
2. Is the propagation map input-dependent (trained) or universal (architecture)?
3. Are trained vs untrained Jacobians actually different?

Experiments:
41A: Direction Propagation Through Layers
  - Inject random directions at layer l, measure Δh_L (hidden state, NOT logits)
  - PCA on output perturbations → subspace dimensionality
  - If top-k PCs explain most variance → directions converge to k-dim subspace
  
41B: Conditional Jacobian — Input Dependence
  - Same injection directions, different input texts
  - Compare propagation maps across inputs
  - If maps differ → Jacobian is input-dependent → training effect
  
41C: Trained vs Untrained Jacobian Direct Comparison
  - Same setup as 41A but on untrained model
  - Compare: subspace dimensionality, convergence rate, input dependence
  - If trained ≠ untrained → training shapes dynamics

Usage:
  python ccml_phase41_direction_dynamics.py --model deepseek7b --exp 1
  python ccml_phase41_direction_dynamics.py --model deepseek7b --exp 2
  python ccml_phase41_direction_dynamics.py --model deepseek7b --exp 3
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
TEST_SENTENCES = [
    "The cat sat on the mat",
    "She walked to the store yesterday",
    "The scientist discovered a new element",
    "Music fills the quiet room",
    "The river flows through the valley",
]

N_RANDOM_DIRS = 40  # number of random probe directions


def inject_and_get_final_hidden(model, input_ids, attention_mask, hook_target,
                                 delta, layers, n_layers, token_pos=-1):
    """
    Inject delta at hook_target, return:
    - final hidden state perturbation Δh_L (NOT logits!)
    - base final hidden state h_L
    """
    base_hs_final = [None]
    perturbed_hs_final = [None]
    
    # Get base forward
    def base_hook(module, input, output):
        if isinstance(output, tuple):
            base_hs_final[0] = output[0][0, token_pos, :].detach().cpu().float()
        else:
            base_hs_final[0] = output[0, token_pos, :].detach().cpu().float()
    
    handle_base = layers[-1].register_forward_hook(base_hook)
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    handle_base.remove()
    
    # Get perturbed forward
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


def inject_and_get_layer_hidden(model, input_ids, attention_mask, hook_target,
                                 delta, layers, target_layer_idx, token_pos=-1):
    """
    Inject delta at hook_target, return hidden state perturbation at target_layer_idx.
    This is more efficient than going all the way to final layer for intermediate analysis.
    """
    base_hs = [None]
    perturbed_hs = [None]
    
    # Get base forward
    def base_hook(module, input, output):
        if isinstance(output, tuple):
            base_hs[0] = output[0][0, token_pos, :].detach().cpu().float()
        else:
            base_hs[0] = output[0, token_pos, :].detach().cpu().float()
    
    handle_base = layers[target_layer_idx].register_forward_hook(base_hook)
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    handle_base.remove()
    
    # Get perturbed forward
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
            perturbed_hs[0] = output[0][0, token_pos, :].detach().cpu().float()
        else:
            perturbed_hs[0] = output[0, token_pos, :].detach().cpu().float()
    
    handle_perturb = hook_target.register_forward_hook(perturb_hook)
    handle_target = layers[target_layer_idx].register_forward_hook(final_hook)
    
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    handle_perturb.remove()
    handle_target.remove()
    
    if base_hs[0] is None or perturbed_hs[0] is None:
        return None, None
    
    delta_h = perturbed_hs[0] - base_hs[0]
    return delta_h.numpy(), base_hs[0].numpy()


def compute_subspace_analysis(delta_h_matrix):
    """
    Analyze the subspace structure of propagated perturbations.
    
    delta_h_matrix: [n_directions, d_model] — each row is Δh_L for one injection
    
    Returns:
    - singular values
    - cumulative energy (how many dimensions explain most variance)
    - effective dimensionality (number of SVs needed for 90% energy)
    """
    # Center the data
    mean = delta_h_matrix.mean(axis=0)
    centered = delta_h_matrix - mean
    
    # SVD
    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    
    # Cumulative energy
    total_energy = np.sum(s**2)
    if total_energy < 1e-10:
        return {
            'singular_values': s.tolist(),
            'cumulative_energy': np.cumsum(s**2) / (total_energy + 1e-10),
            'effective_dim_90': len(s),
            'effective_dim_50': len(s),
            'top1_fraction': 0.0,
            'top3_fraction': 0.0,
            'top10_fraction': 0.0,
        }
    
    cum_energy = np.cumsum(s**2) / total_energy
    
    # Effective dimensionality
    eff_dim_90 = int(np.searchsorted(cum_energy, 0.90) + 1)
    eff_dim_50 = int(np.searchsorted(cum_energy, 0.50) + 1)
    
    return {
        'singular_values': s[:30].tolist(),  # top 30
        'cumulative_energy': cum_energy[:30].tolist(),
        'effective_dim_90': int(eff_dim_90),
        'effective_dim_50': int(eff_dim_50),
        'top1_fraction': float(s[0]**2 / total_energy),
        'top3_fraction': float(np.sum(s[:3]**2) / total_energy),
        'top10_fraction': float(np.sum(s[:10]**2) / total_energy),
    }


def get_W_U_np(model):
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        return model.lm_head.weight.detach().cpu().float().numpy()
    return model.get_output_embeddings().weight.detach().cpu().float().numpy()


def randomize_model_weights(model, model_name):
    """Randomize attention+MLP weights, keep norm+embed+W_U"""
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    n_randomized = 0
    for li, layer in enumerate(layers):
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


# ============================================================================
# 41A: Direction Propagation — Subspace Analysis
# ============================================================================

def run_41A(model_name):
    """Direction propagation through layers — measure Δh_L, NOT Δlogits"""
    print(f"\n{'='*70}")
    print(f"Phase 41A: Direction Propagation & Subspace Analysis ({model_name})")
    print(f"{'='*70}")
    print(f"Key: Measure Δh_L (hidden state perturbation), NOT Δlogits")
    print(f"This bypasses W_U readout geometry entirely!\n")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    layers = get_layers(model)
    W_U = get_W_U_np(model)

    alpha = 0.1  # perturbation scale relative to ||h||
    
    # Injection layers to test
    inject_layers = [7, 14, 21, min(26, n_layers-2)]
    inject_layers = sorted(set([li for li in inject_layers if li < n_layers]))
    
    all_results = {}
    
    for text_idx, text in enumerate(TEST_SENTENCES[:3]):
        print(f"\n--- Text {text_idx+1}: '{text[:50]}' ---")
        
        # Get input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # Get base hidden state norms at injection layers
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True)
        
        hs_norms = {}
        for li in inject_layers:
            hs_norms[li] = outputs.hidden_states[li+1][0, -1].float().norm().item()
        
        base_logits = outputs.logits[0, -1].cpu().float().numpy()
        top_tok = int(np.argmax(base_logits))
        second_tok = int(np.argsort(base_logits)[-2])
        
        del outputs
        torch.cuda.empty_cache()
        
        text_results = {}
        
        for li in inject_layers:
            print(f"\n  === Inject at L{li}, propagate to L{n_layers-1} ===")
            
            hs_norm = hs_norms[li]
            delta_mag = alpha * hs_norm
            
            # Generate random injection directions
            np.random.seed(42 + text_idx * 100 + li)
            random_dirs = np.random.randn(N_RANDOM_DIRS, d_model)
            random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)
            
            # Also test margin direction (for comparison)
            margin_dir = W_U[top_tok] - W_U[second_tok]
            margin_dir = margin_dir / (np.linalg.norm(margin_dir) + 1e-10)
            
            # Propagate each direction
            delta_h_L_list = []
            gain_logit_list = []  # also measure logit gain for comparison
            
            for di in range(N_RANDOM_DIRS):
                delta = torch.tensor(delta_mag * random_dirs[di], 
                                    dtype=torch.float32, device=device)
                
                # Get hidden state perturbation at final layer
                delta_h_L, h_L_base = inject_and_get_final_hidden(
                    model, input_ids, attention_mask, layers[li],
                    delta, layers, n_layers, token_pos=-1)
                
                if delta_h_L is None:
                    continue
                
                # Also get logit perturbation for comparison
                with torch.no_grad():
                    out_base = model(input_ids=input_ids, attention_mask=attention_mask)
                base_logits_full = out_base.logits[0, -1].cpu().float().numpy()
                
                # Perturbed forward (reuse the injection function for logits)
                def perturb_fn(module, input, output):
                    if isinstance(output, tuple):
                        new_h = output[0].clone()
                        new_h[0, -1, :] += delta.to(new_h.dtype).to(new_h.device)
                        return (new_h,) + output[1:]
                    new_h = output.clone()
                    new_h[0, -1, :] += delta.to(new_h.dtype).to(new_h.device)
                    return new_h
                
                handle = layers[li].register_forward_hook(perturb_fn)
                with torch.no_grad():
                    out_perturbed = model(input_ids=input_ids, attention_mask=attention_mask)
                handle.remove()
                
                perturbed_logits = out_perturbed.logits[0, -1].cpu().float().numpy()
                
                delta_h_L_list.append(delta_h_L)
                gain_logit_list.append(
                    perturbed_logits[top_tok] - perturbed_logits[second_tok] - 
                    (base_logits_full[top_tok] - base_logits_full[second_tok])
                )
                
                del out_base, out_perturbed
                torch.cuda.empty_cache()
            
            # Also propagate margin direction
            delta_margin = torch.tensor(delta_mag * margin_dir,
                                       dtype=torch.float32, device=device)
            delta_h_L_margin, h_L_base_margin = inject_and_get_final_hidden(
                model, input_ids, attention_mask, layers[li],
                delta_margin, layers, n_layers, token_pos=-1)
            
            if delta_h_L_margin is not None and len(delta_h_L_list) > 0:
                delta_h_L_list.append(delta_h_L_margin)
                # margin logit gain
                with torch.no_grad():
                    out_base2 = model(input_ids=input_ids, attention_mask=attention_mask)
                base_logits2 = out_base2.logits[0, -1].cpu().float().numpy()
                
                def perturb_fn2(module, input, output):
                    if isinstance(output, tuple):
                        new_h = output[0].clone()
                        new_h[0, -1, :] += delta_margin.to(new_h.dtype).to(new_h.device)
                        return (new_h,) + output[1:]
                    new_h = output.clone()
                    new_h[0, -1, :] += delta_margin.to(new_h.dtype).to(new_h.device)
                    return new_h
                
                handle2 = layers[li].register_forward_hook(perturb_fn2)
                with torch.no_grad():
                    out_pert2 = model(input_ids=input_ids, attention_mask=attention_mask)
                handle2.remove()
                pert_logits2 = out_pert2.logits[0, -1].cpu().float().numpy()
                
                gain_margin_logit = (pert_logits2[top_tok] - pert_logits2[second_tok]) - \
                                   (base_logits2[top_tok] - base_logits2[second_tok])
                gain_logit_list.append(gain_margin_logit)
                
                del out_base2, out_pert2
                torch.cuda.empty_cache()
            
            if len(delta_h_L_list) == 0:
                print(f"  No valid propagations at L{li}")
                continue
            
            # Stack into matrix
            delta_h_matrix = np.array(delta_h_L_list)  # [n_dirs, d_model]
            
            # Subspace analysis
            subspace = compute_subspace_analysis(delta_h_matrix[:N_RANDOM_DIRS])  # only random dirs
            
            # Compute norms
            norms_random = np.linalg.norm(delta_h_matrix[:N_RANDOM_DIRS], axis=1)
            norm_margin = np.linalg.norm(delta_h_L_margin) if delta_h_L_margin is not None else 0
            
            # Norm amplification: ||Δh_L|| / (alpha * ||h_l||)
            norm_amp_random = norms_random.mean() / (delta_mag + 1e-10)
            norm_amp_margin = norm_margin / (delta_mag + 1e-10)
            
            # Compare hidden-state-space DER (not logit DER)
            # HS-DER = ||Δh_L_margin|| / mean(||Δh_L_random||)
            hs_der = norm_margin / (norms_random.mean() + 1e-10)
            
            # Traditional logit DER for comparison
            logit_gains_random = np.abs(gain_logit_list[:N_RANDOM_DIRS])
            logit_der = abs(gain_logit_list[-1]) / (logit_gains_random.mean() + 1e-10) if len(gain_logit_list) > N_RANDOM_DIRS else 0
            
            print(f"  Subspace dim (90%): {subspace['effective_dim_90']}")
            print(f"  Subspace dim (50%): {subspace['effective_dim_50']}")
            print(f"  Top-1 SV fraction:  {subspace['top1_fraction']:.3f}")
            print(f"  Top-3 SV fraction:  {subspace['top3_fraction']:.3f}")
            print(f"  Top-10 SV fraction: {subspace['top10_fraction']:.3f}")
            print(f"  Norm amplification (random): {norm_amp_random:.3f}")
            print(f"  Norm amplification (margin): {norm_amp_margin:.3f}")
            print(f"  HS-DER (hidden state): {hs_der:.2f}x")
            print(f"  Logit-DER (via W_U):  {logit_der:.1f}x")
            print(f"  ||Δh_L||/||δ_in|| (random): {norm_amp_random:.3f}")
            print(f"  ||Δh_L||/||δ_in|| (margin): {norm_amp_margin:.3f}")
            
            text_results[li] = {
                'subspace': subspace,
                'norm_amp_random': float(norm_amp_random),
                'norm_amp_margin': float(norm_amp_margin),
                'hs_der': float(hs_der),
                'logit_der': float(logit_der),
                'mean_random_norm': float(norms_random.mean()),
                'margin_norm': float(norm_margin),
                'delta_mag': float(delta_mag),
            }
        
        all_results[text] = text_results
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Phase 41A SUMMARY ({model_name})")
    print(f"{'='*70}")
    print(f"\n--- Subspace Dimensionality ---")
    print(f"{'Layer':<8} {'EffDim(90%)':<14} {'EffDim(50%)':<14} {'Top1%':<10} {'Top3%':<10} {'Top10%':<10}")
    
    for li in inject_layers:
        subspace_vals = []
        for text, res in all_results.items():
            if li in res:
                subspace_vals.append(res[li]['subspace'])
        
        if subspace_vals:
            avg_dim90 = np.mean([s['effective_dim_90'] for s in subspace_vals])
            avg_dim50 = np.mean([s['effective_dim_50'] for s in subspace_vals])
            avg_top1 = np.mean([s['top1_fraction'] for s in subspace_vals])
            avg_top3 = np.mean([s['top3_fraction'] for s in subspace_vals])
            avg_top10 = np.mean([s['top10_fraction'] for s in subspace_vals])
            print(f"L{li:<6} {avg_dim90:<14.1f} {avg_dim50:<14.1f} {avg_top1:<10.3f} {avg_top3:<10.3f} {avg_top10:<10.3f}")
    
    print(f"\n--- HS-DER vs Logit-DER ---")
    print(f"{'Layer':<8} {'HS-DER':<12} {'Logit-DER':<12} {'Ratio':<10} {'Interpretation'}")
    
    for li in inject_layers:
        hs_ders = []
        logit_ders = []
        for text, res in all_results.items():
            if li in res:
                hs_ders.append(res[li]['hs_der'])
                logit_ders.append(res[li]['logit_der'])
        
        if hs_ders:
            avg_hs = np.mean(hs_ders)
            avg_logit = np.mean(logit_ders)
            ratio = avg_logit / (avg_hs + 1e-10)
            
            if ratio > 5:
                interp = "W_U amplifies direction bias"
            elif ratio > 2:
                interp = "Moderate W_U amplification"
            else:
                interp = "Dynamics dominant"
            
            print(f"L{li:<6} {avg_hs:<12.2f} {avg_logit:<12.1f} {ratio:<10.1f} {interp}")
    
    # Key conclusion
    print(f"\n--- KEY CONCLUSION ---")
    avg_hs_der_all = np.mean([res[li]['hs_der'] for text, res in all_results.items() 
                              for li in res])
    avg_logit_der_all = np.mean([res[li]['logit_der'] for text, res in all_results.items() 
                                  for li in res])
    
    if avg_hs_der_all < 3:
        print(f"HS-DER ≈ {avg_hs_der_all:.2f} → Dynamics is NEARLY ISOTROPIC")
        print(f"  Direction bias at logit level comes from W_U, not Jacobian")
    elif avg_hs_der_all < 10:
        print(f"HS-DER ≈ {avg_hs_der_all:.2f} → Weak direction bias in dynamics")
        print(f"  But Logit-DER = {avg_logit_der_all:.1f} → W_U amplifies it significantly")
    else:
        print(f"HS-DER ≈ {avg_hs_der_all:.2f} → Strong direction bias in dynamics!")
        print(f"  Jacobian itself has anisotropic structure")
    
    # Save results
    out_path = f'tests/glm5_temp/phase41A_{model_name}_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model': model_name, 'd_model': d_model, 'n_layers': n_layers,
            'alpha': alpha, 'n_random_dirs': N_RANDOM_DIRS,
            'results': {k: {str(kk): vv for kk, vv in v.items()} 
                       for k, v in all_results.items()},
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}")
    
    release_model(model)
    return all_results


# ============================================================================
# 41B: Conditional Jacobian — Input Dependence
# ============================================================================

def run_41B(model_name):
    """Test if the propagation map is input-dependent"""
    print(f"\n{'='*70}")
    print(f"Phase 41B: Conditional Jacobian — Input Dependence ({model_name})")
    print(f"{'='*70}")
    print(f"If propagation map differs across inputs → Jacobian is input-dependent")
    print(f"Input-dependent Jacobian = training effect (architecture is fixed)\n")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    layers = get_layers(model)

    alpha = 0.1
    
    # Use a single injection layer (middle of model)
    inject_layer = min(14, n_layers - 2)
    
    # Generate FIXED random directions
    np.random.seed(999)
    random_dirs = np.random.randn(N_RANDOM_DIRS, d_model)
    random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)
    
    # For each text, propagate the SAME directions
    all_delta_h = {}  # {text: [n_dirs, d_model]}
    
    for text_idx, text in enumerate(TEST_SENTENCES):
        print(f"\n--- Text {text_idx+1}: '{text[:50]}' ---")
        
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
        delta_h_list = []
        
        for di in range(N_RANDOM_DIRS):
            delta = torch.tensor(delta_mag * random_dirs[di],
                               dtype=torch.float32, device=device)
            
            delta_h_L, h_L_base = inject_and_get_final_hidden(
                model, input_ids, attention_mask, layers[inject_layer],
                delta, layers, n_layers, token_pos=-1)
            
            if delta_h_L is not None:
                delta_h_list.append(delta_h_L)
            torch.cuda.empty_cache()
        
        if len(delta_h_list) > 0:
            all_delta_h[text] = np.array(delta_h_list)
            # Subspace analysis for this text
            subspace = compute_subspace_analysis(all_delta_h[text])
            print(f"  Subspace dim(90%): {subspace['effective_dim_90']}, "
                  f"Top-1: {subspace['top1_fraction']:.3f}, "
                  f"Top-3: {subspace['top3_fraction']:.3f}")
    
    # Compare propagation maps across texts
    print(f"\n{'='*70}")
    print(f"Phase 41B: Cross-Input Subspace Comparison")
    print(f"{'='*70}")
    
    texts = list(all_delta_h.keys())
    
    if len(texts) < 2:
        print("Not enough valid texts for comparison")
        release_model(model)
        return None
    
    # For each pair of texts, compute subspace overlap
    # Method: project one text's delta_h onto another text's principal subspace
    print(f"\n--- Subspace Overlap Matrix ---")
    print(f"Measures: how much of text_i's propagation space is captured by text_j's space\n")
    
    # Compute principal subspaces (top-k components)
    # SVD of delta_h_matrix [n_dirs, d_model]: Vt[:k] spans the principal subspace in R^d_model
    k = 10  # top-10 components
    subspaces = {}
    for text in texts:
        centered = all_delta_h[text] - all_delta_h[text].mean(axis=0)
        U, s, Vt = np.linalg.svd(centered, full_matrices=False)
        subspaces[text] = Vt[:k]  # [k, d_model] — principal directions
    
    overlap_matrix = np.zeros((len(texts), len(texts)))
    for i, t1 in enumerate(texts):
        for j, t2 in enumerate(texts):
            # Project t1's data onto t2's subspace
            Vt2 = subspaces[t2]  # [k, d_model]
            data1 = all_delta_h[t1]
            mean1 = data1.mean(axis=0)
            centered1 = data1 - mean1  # [n_dirs, d_model]
            # Projection: proj = centered1 @ Vt2.T @ Vt2 → [n_dirs, d_model]
            proj = centered1 @ Vt2.T @ Vt2  # [n_dirs, d_model]
            proj_energy = np.sum(proj**2)
            total_energy = np.sum(centered1**2)
            overlap_matrix[i, j] = proj_energy / (total_energy + 1e-10)
    
    # Print overlap matrix
    header = "         " + "  ".join([f"T{i+1:<6}" for i in range(len(texts))])
    print(header)
    for i, t1 in enumerate(texts):
        row = f"T{i+1:<6}   "
        row += "  ".join([f"{overlap_matrix[i,j]:.3f} " for j in range(len(texts))])
        print(row)
    
    # Compute average cross-input overlap
    cross_overlaps = []
    for i in range(len(texts)):
        for j in range(len(texts)):
            if i != j:
                cross_overlaps.append(overlap_matrix[i, j])
    
    self_overlaps = [overlap_matrix[i, i] for i in range(len(texts))]
    avg_cross = np.mean(cross_overlaps) if cross_overlaps else 0
    avg_self = np.mean(self_overlaps) if self_overlaps else 0
    
    print(f"\nAverage self-overlap:   {avg_self:.3f}")
    print(f"Average cross-overlap:  {avg_cross:.3f}")
    print(f"Self/Cross ratio:       {avg_self / (avg_cross + 1e-10):.2f}")
    
    # Also compute: do top principal directions align across inputs?
    print(f"\n--- Top Principal Direction Alignment ---")
    top_dirs = {}
    for text in texts:
        U, s, Vt = np.linalg.svd(all_delta_h[text], full_matrices=False)
        # Actually Vt[0] is the top right singular vector (in d_model space)
        # But for our centered data, we want the top direction
        centered = all_delta_h[text] - all_delta_h[text].mean(axis=0)
        U2, s2, Vt2 = np.linalg.svd(centered, full_matrices=False)
        top_dirs[text] = Vt2[0]  # [d_model]
    
    print(f"Pairwise |cos(v1, v2)| between top propagation directions:")
    for i, t1 in enumerate(texts):
        for j, t2 in enumerate(texts):
            if i < j:
                cos = abs(np.dot(top_dirs[t1], top_dirs[t2]))
                print(f"  T{i+1} vs T{j+1}: |cos| = {cos:.4f}")
    
    # Interpretation
    print(f"\n--- INTERPRETATION ---")
    if avg_cross > 0.8:
        print("Cross-overlap > 80% → Propagation map is INPUT-INDEPENDENT")
        print("  Jacobian structure is dominated by architecture, not input")
        print("  → Training effect on dynamics is WEAK")
    elif avg_cross > 0.5:
        print("Cross-overlap 50-80% → Mixed: architecture + input-dependent component")
        print("  Some direction preference is shared, some is input-specific")
        print("  → Training shapes part of the dynamics")
    else:
        print("Cross-overlap < 50% → Propagation map is STRONGLY INPUT-DEPENDENT")
        print("  Different inputs activate very different Jacobian structures")
        print("  → Training creates input-conditional dynamics")
    
    # Save results
    out_data = {
        'model': model_name, 'd_model': d_model, 'inject_layer': inject_layer,
        'alpha': alpha, 'n_random_dirs': N_RANDOM_DIRS,
        'avg_self_overlap': float(avg_self),
        'avg_cross_overlap': float(avg_cross),
        'overlap_matrix': overlap_matrix.tolist(),
    }
    out_path = f'tests/glm5_temp/phase41B_{model_name}_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
    
    release_model(model)
    return out_data


# ============================================================================
# 41C: Trained vs Untrained Jacobian Direct Comparison
# ============================================================================

def run_41C(model_name):
    """Compare Jacobian dynamics between trained and untrained models"""
    print(f"\n{'='*70}")
    print(f"Phase 41C: Trained vs Untrained Jacobian ({model_name})")
    print(f"{'='*70}")
    print(f"Direct comparison: subspace dimensionality, norm amplification, input dependence\n")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    layers = get_layers(model)
    W_U = get_W_U_np(model)

    alpha = 0.1
    inject_layers = [7, 14, 21, min(26, n_layers-2)]
    inject_layers = sorted(set([li for li in inject_layers if li < n_layers]))
    
    # ===== Step 1: TRAINED model =====
    print(f"=== TRAINED MODEL ===")
    
    # Get trained model's top tokens
    trained_data = {}
    for text_idx, text in enumerate(TEST_SENTENCES[:2]):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True)
        
        base_logits = outputs.logits[0, -1].cpu().float().numpy()
        top_tok = int(np.argmax(base_logits))
        
        hs_norms = {}
        for li in inject_layers:
            hs_norms[li] = outputs.hidden_states[li+1][0, -1].float().norm().item()
        
        del outputs
        torch.cuda.empty_cache()
        
        trained_data[text] = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'top_tok': top_tok,
            'hs_norms': hs_norms,
        }
    
    # Propagate random directions through TRAINED model
    trained_results = {}
    for text in TEST_SENTENCES[:2]:
        td = trained_data[text]
        text_res = {}
        
        for li in inject_layers:
            np.random.seed(42)
            random_dirs = np.random.randn(N_RANDOM_DIRS, d_model)
            random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)
            
            delta_mag = alpha * td['hs_norms'][li]
            delta_h_list = []
            
            for di in range(N_RANDOM_DIRS):
                delta = torch.tensor(delta_mag * random_dirs[di],
                                   dtype=torch.float32, device=device)
                delta_h_L, _ = inject_and_get_final_hidden(
                    model, td['input_ids'], td['attention_mask'],
                    layers[li], delta, layers, n_layers, token_pos=-1)
                
                if delta_h_L is not None:
                    delta_h_list.append(delta_h_L)
                torch.cuda.empty_cache()
            
            if len(delta_h_list) > 0:
                matrix = np.array(delta_h_list)
                subspace = compute_subspace_analysis(matrix)
                norms = np.linalg.norm(matrix, axis=1)
                norm_amp = norms.mean() / (delta_mag + 1e-10)
                
                text_res[li] = {
                    'subspace': subspace,
                    'norm_amp': float(norm_amp),
                    'mean_norm': float(norms.mean()),
                }
                print(f"  T L{li}: dim90={subspace['effective_dim_90']}, "
                      f"top1={subspace['top1_fraction']:.3f}, "
                      f"norm_amp={norm_amp:.3f}")
        
        trained_results[text] = text_res
    
    # ===== Step 2: RANDOMIZE model =====
    print(f"\n=== RANDOMIZING MODEL ===")
    model = randomize_model_weights(model, model_name)
    
    # Sanity check
    for text in TEST_SENTENCES[:1]:
        td = trained_data[text]
        with torch.no_grad():
            out = model(input_ids=td['input_ids'], attention_mask=td['attention_mask'])
        logits = out.logits[0, -1].cpu().float().numpy()
        if np.isnan(logits).any():
            print("  WARNING: NaN in untrained model, re-initializing with smaller std")
            for layer in get_layers(model):
                for attr in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    if hasattr(layer.self_attn, attr):
                        w = getattr(layer.self_attn, attr)
                        if hasattr(w, 'weight'):
                            torch.nn.init.normal_(w.weight.data, std=0.01)
                for attr in ['gate_proj', 'up_proj', 'down_proj', 'gate_up_proj']:
                    if hasattr(layer.mlp, attr):
                        w = getattr(layer.mlp, attr)
                        if hasattr(w, 'weight'):
                            torch.nn.init.normal_(w.weight.data, std=0.01)
        del out
        torch.cuda.empty_cache()
    
    # ===== Step 3: UNTRAINED model =====
    print(f"\n=== UNTRAINED MODEL ===")
    
    layers = get_layers(model)  # refresh after randomization
    
    # Re-compute hs_norms for untrained model
    untrained_data = {}
    for text in TEST_SENTENCES[:2]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True)
        
        hs_norms = {}
        for li in inject_layers:
            hs_norms[li] = outputs.hidden_states[li+1][0, -1].float().norm().item()
        
        del outputs
        torch.cuda.empty_cache()
        
        untrained_data[text] = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'hs_norms': hs_norms,
        }
    
    # Propagate random directions through UNTRAINED model
    untrained_results = {}
    for text in TEST_SENTENCES[:2]:
        ud = untrained_data[text]
        text_res = {}
        
        for li in inject_layers:
            np.random.seed(42)  # SAME random directions
            random_dirs = np.random.randn(N_RANDOM_DIRS, d_model)
            random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)
            
            delta_mag = alpha * ud['hs_norms'][li]
            delta_h_list = []
            
            for di in range(N_RANDOM_DIRS):
                delta = torch.tensor(delta_mag * random_dirs[di],
                                   dtype=torch.float32, device=device)
                delta_h_L, _ = inject_and_get_final_hidden(
                    model, ud['input_ids'], ud['attention_mask'],
                    layers[li], delta, layers, n_layers, token_pos=-1)
                
                if delta_h_L is not None and not np.isnan(delta_h_L).any():
                    delta_h_list.append(delta_h_L)
                torch.cuda.empty_cache()
            
            if len(delta_h_list) > 0:
                matrix = np.array(delta_h_list)
                subspace = compute_subspace_analysis(matrix)
                norms = np.linalg.norm(matrix, axis=1)
                norm_amp = norms.mean() / (delta_mag + 1e-10)
                
                text_res[li] = {
                    'subspace': subspace,
                    'norm_amp': float(norm_amp),
                    'mean_norm': float(norms.mean()),
                }
                print(f"  U L{li}: dim90={subspace['effective_dim_90']}, "
                      f"top1={subspace['top1_fraction']:.3f}, "
                      f"norm_amp={norm_amp:.3f}")
        
        untrained_results[text] = text_res
    
    # ===== Step 4: Comparison =====
    print(f"\n{'='*70}")
    print(f"Phase 41C SUMMARY: Trained vs Untrained Jacobian Dynamics")
    print(f"{'='*70}")
    print(f"\n{'Layer':<8} {'T-dim90':<10} {'U-dim90':<10} {'T-top1':<10} {'U-top1':<10} "
          f"{'T-norm_amp':<12} {'U-norm_amp':<12} {'Conclusion'}")
    print("-" * 90)
    
    for li in inject_layers:
        t_dims = [trained_results[t][li]['subspace']['effective_dim_90'] 
                  for t in trained_results if li in trained_results[t]]
        u_dims = [untrained_results[t][li]['subspace']['effective_dim_90'] 
                  for t in untrained_results if li in untrained_results[t]]
        t_top1s = [trained_results[t][li]['subspace']['top1_fraction'] 
                   for t in trained_results if li in trained_results[t]]
        u_top1s = [untrained_results[t][li]['subspace']['top1_fraction'] 
                   for t in untrained_results if li in untrained_results[t]]
        t_amps = [trained_results[t][li]['norm_amp'] 
                  for t in trained_results if li in trained_results[t]]
        u_amps = [untrained_results[t][li]['norm_amp'] 
                  for t in untrained_results if li in untrained_results[t]]
        
        if t_dims and u_dims:
            t_dim90 = np.mean(t_dims)
            u_dim90 = np.mean(u_dims)
            t_top1 = np.mean(t_top1s)
            u_top1 = np.mean(u_top1s)
            t_amp = np.mean(t_amps)
            u_amp = np.mean(u_amps)
            
            dim_ratio = t_dim90 / (u_dim90 + 1e-10)
            
            if dim_ratio < 0.7:
                conclusion = "✓ Training REDUCES subspace dim (convergence)"
            elif dim_ratio > 1.3:
                conclusion = "✗ Training INCREASES subspace dim (diversification)"
            else:
                conclusion = "≈ Similar subspace dimensionality"
            
            print(f"L{li:<6} {t_dim90:<10.1f} {u_dim90:<10.1f} {t_top1:<10.3f} {u_top1:<10.3f} "
                  f"{t_amp:<12.3f} {u_amp:<12.3f} {conclusion}")
    
    # Key conclusion
    print(f"\n--- KEY CONCLUSION ---")
    all_t_dims = [trained_results[t][li]['subspace']['effective_dim_90'] 
                  for t in trained_results for li in inject_layers if li in trained_results[t]]
    all_u_dims = [untrained_results[t][li]['subspace']['effective_dim_90'] 
                  for t in untrained_results for li in inject_layers if li in untrained_results[t]]
    all_t_top1 = [trained_results[t][li]['subspace']['top1_fraction'] 
                  for t in trained_results for li in inject_layers if li in trained_results[t]]
    all_u_top1 = [untrained_results[t][li]['subspace']['top1_fraction'] 
                  for t in untrained_results for li in inject_layers if li in untrained_results[t]]
    
    if all_t_dims and all_u_dims:
        avg_t_dim = np.mean(all_t_dims)
        avg_u_dim = np.mean(all_u_dims)
        avg_t_top1 = np.mean(all_t_top1)
        avg_u_top1 = np.mean(all_u_top1)
        
        print(f"Trained   avg dim(90%) = {avg_t_dim:.1f}, avg top-1 fraction = {avg_t_top1:.3f}")
        print(f"Untrained avg dim(90%) = {avg_u_dim:.1f}, avg top-1 fraction = {avg_u_top1:.3f}")
        
        if avg_t_dim < avg_u_dim * 0.8:
            print("→ Training creates MORE ANISOTROPIC dynamics (lower effective dim)")
            print("  Random directions converge to fewer principal directions in trained model")
            print("  ★ This IS a training effect on dynamics!")
        elif avg_t_top1 > avg_u_top1 * 1.2:
            print("→ Training creates STRONGER top-direction concentration")
            print("  ★ This IS a training effect on dynamics!")
        else:
            print("→ Trained and untrained have SIMILAR subspace structure")
            print("  Dynamics structure is largely architectural")
    
    # Save results
    out_data = {
        'model': model_name, 'd_model': d_model,
        'alpha': alpha, 'n_random_dirs': N_RANDOM_DIRS,
        'trained_results': {k: {str(kk): vv for kk, vv in v.items()} 
                           for k, v in trained_results.items()},
        'untrained_results': {k: {str(kk): vv for kk, vv in v.items()} 
                             for k, v in untrained_results.items()},
    }
    out_path = f'tests/glm5_temp/phase41C_{model_name}_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}")
    
    release_model(model)
    return out_data


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 41: Direction Dynamics')
    parser.add_argument('--model', type=str, required=True,
                       choices=['deepseek7b', 'glm4', 'qwen3'])
    parser.add_argument('--exp', type=int, required=True, choices=[1, 2, 3],
                       help='1=41A (subspace), 2=41B (input dependence), 3=41C (trained vs untrained)')
    args = parser.parse_args()

    if args.exp == 1:
        run_41A(args.model)
    elif args.exp == 2:
        run_41B(args.model)
    elif args.exp == 3:
        run_41C(args.model)
