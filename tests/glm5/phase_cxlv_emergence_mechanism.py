#!/usr/bin/env python3
"""
Phase CXLV: Last-Layer Emergence Mechanism (P641-P644)
Focus: Why does gap information only emerge at the last layer?

Critical finding from Phase CXLIV:
- h(0)->h(L) linear cos=0.79-0.96 but Pipeline efficiency only 32-46% (DS7B=-12%)
- gap information is NOT predictable at intermediate layers (r<0.41)
- gap only "emerges" at the last layer (r>0.99)
- This contradicts the "flat linear geometry" framework

Hypothesis: The last LayerNorm (or final residual connection) rotates h 
into alignment with Delta_W, creating the gap.

P641: Last LayerNorm precise effect - how does it rotate h to align with Delta_W?
P642: Gap emergence physical mechanism - why is gap unpredictable at intermediate layers?
P643: Corrected mathematical framework - per-layer linear + last-layer nonlinear emergence
P644: Quantitative validation of corrected framework
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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
from model_utils import load_model, get_model_info, release_model

TEST_TEXTS = [
    "The quantum computer solved the complex optimization problem in seconds.",
    "She walked through the ancient forest, listening to birds singing.",
    "The stock market crashed after the central bank raised interest rates.",
    "Artificial intelligence is transforming healthcare diagnostics.",
    "The musician played a haunting melody on the violin.",
    "Climate change threatens coastal cities with rising sea levels.",
    "The philosopher questioned the nature of consciousness and free will.",
    "A new vaccine was developed to combat the emerging virus.",
    "The architect designed a sustainable building with solar panels.",
    "The detective gathered clues to solve the mysterious case.",
    "The chef prepared a delicious meal using local ingredients.",
    "The spacecraft orbited Mars, collecting data for scientists.",
    "The poet wrote verses about love and loss under moonlight.",
    "The economist predicted a recession based on market indicators.",
    "The teacher encouraged students to think critically about history.",
    "The programmer debugged the code to fix the memory leak.",
    "The artist painted a vibrant landscape with bold brushstrokes.",
    "The biologist discovered a new species in the rainforest.",
    "The judge ruled in favor of the plaintiff after hearing arguments.",
    "The engineer built a bridge that could withstand earthquakes.",
    "The novelist crafted a story about time travel and paradox.",
    "The doctor diagnosed the patient with a rare genetic disorder.",
    "The astronaut floated weightlessly in the International Space Station.",
    "The historian analyzed primary sources from the Renaissance.",
    "The mathematician proved a theorem about prime numbers.",
    "The chemist synthesized a new compound in the laboratory.",
    "The diplomat negotiated a peace agreement between nations.",
    "The journalist reported on the election results from the capital.",
    "The firefighter rescued a family from the burning building.",
    "The musician composed a symphony that moved the audience to tears.",
    "The researcher published a groundbreaking paper on dark matter.",
    "The sailor navigated through the storm using celestial observations.",
    "The developer created an app that simplifies project management.",
    "The photographer captured the sunset over the mountain range.",
    "The botanist studied the rare orchid in its natural habitat.",
    "The librarian organized the archives for public access.",
    "The cyclist trained rigorously for the upcoming championship.",
    "The geologist identified a new mineral formation in the cave.",
    "The translator rendered the ancient text into modern language.",
    "The volunteer distributed food to families affected by the flood.",
]


def compute_features(model, tokenizer, device, texts):
    """Compute comprehensive features with attention to last-layer operations."""
    if hasattr(model, 'lm_head'):
        W_U = model.lm_head.weight.detach().cpu().float().numpy()
    else:
        W_U = model.get_output_embeddings().weight.detach().cpu().float().numpy()
    
    if hasattr(model.config, 'hidden_size'):
        d_model = model.config.hidden_size
    elif hasattr(model.config, 'd_model'):
        d_model = model.config.d_model
    else:
        d_model = 2560
    
    if hasattr(model.config, 'num_hidden_layers'):
        n_layers = model.config.num_hidden_layers
    elif hasattr(model.config, 'n_layer'):
        n_layers = model.config.n_layer
    else:
        n_layers = 32
    
    # Check if model has final LayerNorm
    has_final_ln = False
    final_ln_weight = None
    final_ln_bias = None
    
    if hasattr(model.model, 'norm'):
        has_final_ln = True
        final_ln_weight = model.model.norm.weight.detach().cpu().float().numpy()
        if hasattr(model.model.norm, 'bias') and model.model.norm.bias is not None:
            final_ln_bias = model.model.norm.bias.detach().cpu().float().numpy()
        else:
            final_ln_bias = np.zeros(d_model)
    elif hasattr(model.model, 'final_layernorm'):
        has_final_ln = True
        final_ln_weight = model.model.final_layernorm.weight.detach().cpu().float().numpy()
        if hasattr(model.model.final_layernorm, 'bias') and model.model.final_layernorm.bias is not None:
            final_ln_bias = model.model.final_layernorm.bias.detach().cpu().float().numpy()
        else:
            final_ln_bias = np.zeros(d_model)
    
    print(f"  Model has final LayerNorm: {has_final_ln}")
    if has_final_ln:
        print(f"  Final LN weight norm: {np.linalg.norm(final_ln_weight):.4f}")
        print(f"  Final LN bias norm: {np.linalg.norm(final_ln_bias):.4f}")
    
    # Get layer norm weights from each layer
    layers = model.model.layers if hasattr(model.model, 'layers') else []
    ln_weights = []
    for layer in layers:
        if hasattr(layer, 'input_layernorm'):
            ln_w = layer.input_layernorm.weight.detach().cpu().float().numpy()
        elif hasattr(layer, 'ln_1'):
            ln_w = layer.ln_1.weight.detach().cpu().float().numpy()
        else:
            ln_w = np.ones(d_model)
        ln_weights.append(ln_w)
    
    all_features = []
    
    for i, text in enumerate(texts):
        if (i+1) % 10 == 0:
            print(f"  Processing text {i+1}/{len(texts)}...")
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        all_hidden = []
        for hs in outputs.hidden_states:
            all_hidden.append(hs[0, -1, :].cpu().float().numpy())
        
        h = all_hidden[-1]
        logits = outputs.logits[0, -1, :].cpu().float().numpy()
        
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        logit_gap = logits[top1_idx] - logits[top2_idx]
        prob = 1.0 / (1.0 + np.exp(-logit_gap))
        
        W_U_top1 = W_U[top1_idx]
        W_U_top2 = W_U[top2_idx]
        Delta_W = W_U_top1 - W_U_top2
        
        # Gap at each layer
        gaps_at_layers = [np.dot(all_hidden[l], Delta_W) for l in range(len(all_hidden))]
        
        # Pre-final-LN hidden state (second to last)
        h_pre_ln = all_hidden[-2] if len(all_hidden) > 1 else all_hidden[0]
        
        # Simulate LayerNorm effect
        if has_final_ln and np.linalg.norm(h_pre_ln) > 0:
            # LayerNorm: (h - mean) / std * gamma + beta
            h_mean = np.mean(h_pre_ln)
            h_std = np.std(h_pre_ln)
            if h_std > 1e-10:
                h_ln_normalized = (h_pre_ln - h_mean) / h_std
                h_ln_scaled = h_ln_normalized * final_ln_weight + final_ln_bias
            else:
                h_ln_scaled = h_pre_ln.copy()
        else:
            h_ln_scaled = None
        
        # Delta_W direction alignment at each layer
        cos_with_dw = []
        for l in range(len(all_hidden)):
            h_l = all_hidden[l]
            cos = np.dot(h_l, Delta_W) / (np.linalg.norm(h_l) * np.linalg.norm(Delta_W) + 1e-10)
            cos_with_dw.append(cos)
        
        all_features.append({
            'h': h,
            'all_hidden': all_hidden,
            'logit_gap': logit_gap,
            'prob': prob,
            'top1_idx': top1_idx,
            'top2_idx': top2_idx,
            'Delta_W': Delta_W,
            'W_U_top1': W_U_top1,
            'W_U_top2': W_U_top2,
            'gaps_at_layers': gaps_at_layers,
            'h_pre_ln': h_pre_ln,
            'h_ln_scaled': h_ln_scaled,
            'cos_with_dw': cos_with_dw,
            'h_norm': np.linalg.norm(h),
            'text_idx': i,
        })
    
    extra = {
        'W_U': W_U,
        'd_model': d_model,
        'n_layers': n_layers,
        'has_final_ln': has_final_ln,
        'final_ln_weight': final_ln_weight,
        'final_ln_bias': final_ln_bias,
        'ln_weights': ln_weights,
    }
    
    return all_features, extra


def experiment_p641(all_features, extra, model_name):
    """P641: Last LayerNorm Precise Effect
    How does the final LayerNorm rotate h into alignment with Delta_W?
    
    Key test: Compare h before and after final LayerNorm
    - Does LayerNorm increase cos(h, Delta_W)?
    - Does LayerNorm increase gap?
    - What fraction of gap is created by LayerNorm?
    """
    print(f"\n{'='*60}")
    print(f"P641: Last LayerNorm Precise Effect ({model_name})")
    print(f"{'='*60}")
    
    W_U = extra['W_U']
    d_model = extra['d_model']
    n_layers = extra['n_layers']
    has_final_ln = extra['has_final_ln']
    final_ln_weight = extra['final_ln_weight']
    final_ln_bias = extra['final_ln_bias']
    
    n_texts = len(all_features)
    gaps = np.array([f['logit_gap'] for f in all_features])
    
    # 1. Before vs After final LayerNorm
    print(f"\n1. Before vs After Final LayerNorm:")
    print(f"   Model has final LN: {has_final_ln}")
    
    if has_final_ln:
        # Compare h_pre_ln (before LN) with h (after LN)
        h_pre = np.array([f['h_pre_ln'] for f in all_features])
        h_post = np.array([f['h'] for f in all_features])
        
        # Gap before and after LN
        Delta_Ws = np.array([f['Delta_W'] for f in all_features])
        gaps_pre = np.sum(h_pre * Delta_Ws, axis=1)
        gaps_post = np.sum(h_post * Delta_Ws, axis=1)
        
        # Cos with Delta_W before and after
        cos_pre = []
        cos_post = []
        for i in range(n_texts):
            dw = all_features[i]['Delta_W']
            c_pre = np.dot(h_pre[i], dw) / (np.linalg.norm(h_pre[i]) * np.linalg.norm(dw) + 1e-10)
            c_post = np.dot(h_post[i], dw) / (np.linalg.norm(h_post[i]) * np.linalg.norm(dw) + 1e-10)
            cos_pre.append(c_pre)
            cos_post.append(c_post)
        
        cos_pre = np.array(cos_pre)
        cos_post = np.array(cos_post)
        
        print(f"   Gap before LN: mean={np.mean(gaps_pre):.4f}, std={np.std(gaps_pre):.4f}")
        print(f"   Gap after LN:  mean={np.mean(gaps_post):.4f}, std={np.std(gaps_post):.4f}")
        print(f"   Gap change (after - before): mean={np.mean(gaps_post - gaps_pre):.4f}")
        print(f"   cos(h, Delta_W) before LN: mean={np.mean(cos_pre):.4f}")
        print(f"   cos(h, Delta_W) after LN:  mean={np.mean(cos_post):.4f}")
        print(f"   cos change: mean={np.mean(cos_post - cos_pre):.4f}")
        
        # Fraction of gap created by LN
        gap_ratio_pre = np.mean(np.abs(gaps_pre)) / (np.mean(np.abs(gaps_post)) + 1e-10)
        gap_ratio_ln = 1 - gap_ratio_pre
        print(f"   Gap fraction before LN: {gap_ratio_pre:.4f}")
        print(f"   Gap fraction from LN: {gap_ratio_ln:.4f}")
        
        # Correlation: does LN increase gap?
        r_pre_post, _ = stats.pearsonr(gaps_pre, gaps_post)
        print(f"   gap(pre-LN) -> gap(post-LN): r={r_pre_post:.4f}")
        
        # 2. LN weight analysis
        print(f"\n2. Final LayerNorm Weight Analysis:")
        
        # Is LN weight aligned with Delta_W direction?
        # For each text, compute cos(LN_weight, Delta_W)
        cos_ln_dw = []
        for f in all_features:
            c = np.dot(final_ln_weight, f['Delta_W']) / (np.linalg.norm(final_ln_weight) * np.linalg.norm(f['Delta_W']) + 1e-10)
            cos_ln_dw.append(c)
        
        cos_ln_dw = np.array(cos_ln_dw)
        print(f"   cos(LN_weight, Delta_W): mean={np.mean(cos_ln_dw):.4f}, std={np.std(cos_ln_dw):.4f}")
        
        # What does LN do geometrically?
        # LN: h' = (h - mean(h)) / std(h) * gamma + beta
        # = gamma/std(h) * h + (beta - gamma*mean(h)/std(h))
        # So LN is a linear transformation + shift per sample
        
        # Compute LN effective matrix for each sample
        print(f"\n3. LN as Per-Sample Linear Transformation:")
        
        ln_gaps_predicted = []
        ln_cos_predicted = []
        
        for f in all_features:
            h_raw = f['h_pre_ln']
            dw = f['Delta_W']
            
            h_mean = np.mean(h_raw)
            h_std = np.std(h_raw)
            
            if h_std > 1e-10:
                # LN gap = (gamma/std) * (h.Delta_W) + (beta - gamma*mean/std) . Delta_W
                # = (gamma/std) * gap_raw + (beta.Delta_W - gamma*mean/std * sum(Delta_W))
                
                scale = final_ln_weight / h_std
                shift = final_ln_bias - final_ln_weight * h_mean / h_std
                
                gap_ln = np.dot(scale * h_raw + shift, dw)
                ln_gaps_predicted.append(gap_ln)
                
                h_ln = scale * h_raw + shift
                cos_ln = np.dot(h_ln, dw) / (np.linalg.norm(h_ln) * np.linalg.norm(dw) + 1e-10)
                ln_cos_predicted.append(cos_ln)
        
        if ln_gaps_predicted:
            r_ln_pred, _ = stats.pearsonr(ln_gaps_predicted, gaps)
            print(f"   LN-predicted gap vs actual gap: r={r_ln_pred:.6f}")
            print(f"   LN-predicted cos vs actual cos: r_diff={np.mean(np.abs(np.array(ln_cos_predicted) - cos_post)):.6f}")
        
        # 4. Does LN amplify the gap?
        print(f"\n4. LN Gap Amplification:")
        
        # gap_after = scale_factor * gap_before + correction
        # where scale_factor = gamma.Delta_W / (std * |Delta_W|)
        # This is approximate
        
        gap_amplifications = []
        for i in range(n_texts):
            if abs(gaps_pre[i]) > 1e-10:
                amp = gaps_post[i] / gaps_pre[i]
                gap_amplifications.append(amp)
        
        if gap_amplifications:
            print(f"   Gap amplification (gap_after/gap_before):")
            print(f"     mean={np.mean(gap_amplifications):.4f}")
            print(f"     median={np.median(gap_amplifications):.4f}")
            print(f"     >1 fraction: {np.mean(np.array(gap_amplifications) > 1):.4f}")
        
        # 5. What if we remove LN? (Simulate)
        print(f"\n5. Simulated LN Removal:")
        
        # h_no_ln = h_pre_ln (raw, unnormalized)
        gaps_no_ln = gaps_pre
        probs_no_ln = 1.0 / (1.0 + np.exp(-gaps_no_ln))
        probs_actual = np.array([f['prob'] for f in all_features])
        
        r_no_ln, _ = stats.pearsonr(probs_no_ln, probs_actual)
        r_with_ln, _ = stats.pearsonr(1.0 / (1.0 + np.exp(-gaps_post)), probs_actual)
        
        print(f"   prob from gap(pre-LN) vs actual: r={r_no_ln:.4f}")
        print(f"   prob from gap(post-LN) vs actual: r={r_with_ln:.6f}")
        print(f"   LN improvement: {(r_with_ln - r_no_ln):.4f}")
    
    else:
        print(f"   No final LayerNorm found in this model")
        # Still compute layer-wise gap analysis
        h_pre = np.array([f['all_hidden'][-2] for f in all_features])
        h_post = np.array([f['h'] for f in all_features])
        Delta_Ws = np.array([f['Delta_W'] for f in all_features])
        gaps_pre = np.sum(h_pre * Delta_Ws, axis=1)
        gaps_post = np.sum(h_post * Delta_Ws, axis=1)
        r_pre_post, _ = stats.pearsonr(gaps_pre, gaps_post)
        print(f"   gap(layer L-1) -> gap(layer L): r={r_pre_post:.4f}")
    
    # 6. Layer-wise cos(h_l, Delta_W) trajectory
    print(f"\n6. cos(h_l, Delta_W) Trajectory Across Layers:")
    
    all_cos_traj = np.array([f['cos_with_dw'] for f in all_features])  # [n_texts, n_layers+1]
    mean_cos_traj = np.mean(all_cos_traj, axis=0)
    std_cos_traj = np.std(all_cos_traj, axis=0)
    
    for l in [0, 1, 2, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1, n_layers]:
        if l < len(mean_cos_traj):
            print(f"   Layer {l:2d}: cos={mean_cos_traj[l]:.4f} +/- {std_cos_traj[l]:.4f}")
    
    # Find the "emergence point" - where cos first becomes significant
    cos_threshold = 0.3
    for l in range(len(mean_cos_traj)):
        if abs(mean_cos_traj[l]) > cos_threshold:
            print(f"   First layer with |cos|>{cos_threshold}: Layer {l}")
            break
    
    return {
        'has_final_ln': has_final_ln,
        'gap_ratio_pre_ln': gap_ratio_pre if has_final_ln else None,
        'gap_ratio_from_ln': gap_ratio_ln if has_final_ln else None,
        'r_pre_post': r_pre_post if has_final_ln else None,
        'ln_gap_pred_r': r_ln_pred if has_final_ln and ln_gaps_predicted else None,
    }


def experiment_p642(all_features, extra, model_name):
    """P642: Gap Emergence Physical Mechanism
    Why is gap unpredictable at intermediate layers but emerges at the last?
    
    Test: 
    1. Is there a "rotation" happening at the last layer?
    2. Does h rotate into the Delta_W subspace?
    3. Is the rotation gradual or sudden?
    """
    print(f"\n{'='*60}")
    print(f"P642: Gap Emergence Physical Mechanism ({model_name})")
    print(f"{'='*60}")
    
    W_U = extra['W_U']
    d_model = extra['d_model']
    n_layers = extra['n_layers']
    
    n_texts = len(all_features)
    gaps = np.array([f['logit_gap'] for f in all_features])
    
    # 1. Gap accumulation trajectory
    print(f"\n1. Gap Accumulation Trajectory:")
    
    all_gap_traj = np.array([f['gaps_at_layers'] for f in all_features])  # [n_texts, n_layers+1]
    mean_gap_traj = np.mean(all_gap_traj, axis=0)
    std_gap_traj = np.std(all_gap_traj, axis=0)
    
    for l in [0, 1, 2, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1, n_layers]:
        if l < len(mean_gap_traj):
            print(f"   Layer {l:2d}: gap={mean_gap_traj[l]:.4f} +/- {std_gap_traj[l]:.4f}")
    
    # Correlation of gap_at_layer_l with gap_at_layer_L
    print(f"\n2. Gap Correlation with Final Gap:")
    
    gap_corr_with_final = []
    for l in range(n_layers + 1):
        gaps_l = all_gap_traj[:, l]
        r_l, _ = stats.pearsonr(gaps_l, gaps)
        gap_corr_with_final.append(r_l)
    
    for l in [0, 1, 2, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1, n_layers]:
        if l < len(gap_corr_with_final):
            print(f"   Layer {l:2d}: r(gap_l, gap_L)={gap_corr_with_final[l]:.4f}")
    
    # Find the "emergence layer" where correlation first exceeds 0.8
    emergence_layer = -1
    for l in range(n_layers, -1, -1):
        if abs(gap_corr_with_final[l]) < 0.8:
            emergence_layer = l + 1
            break
    if emergence_layer < 0:
        emergence_layer = 0
    
    print(f"   Emergence layer (first r>0.8): Layer {emergence_layer}")
    print(f"   (out of {n_layers} total layers)")
    
    # 3. Is the emergence gradual or sudden?
    print(f"\n3. Emergence: Gradual or Sudden?")
    
    # Compute gap correlation change rate
    corr_changes = []
    for l in range(1, n_layers + 1):
        change = gap_corr_with_final[l] - gap_corr_with_final[l-1]
        corr_changes.append(change)
    
    max_change_layer = np.argmax(np.abs(corr_changes))
    max_change_val = corr_changes[max_change_layer]
    
    print(f"   Largest correlation jump: Layer {max_change_layer}->{max_change_layer+1}")
    print(f"   Jump magnitude: {max_change_val:.4f}")
    print(f"   Mean correlation change: {np.mean(np.abs(corr_changes)):.4f}")
    
    if abs(max_change_val) > 3 * np.mean(np.abs(corr_changes)):
        print(f"   => EMERGENCE IS SUDDEN (single-layer jump)")
    else:
        print(f"   => EMERGENCE IS GRADUAL")
    
    # 4. h rotation analysis
    print(f"\n4. h Rotation Analysis:")
    
    # Compute h trajectory in Delta_W subspace
    # Project each h onto the Delta_W direction
    Delta_Ws = np.array([f['Delta_W'] for f in all_features])
    
    # For each text, compute the "gap component" and "non-gap component" of h
    gap_fracs = []
    for f in all_features:
        dw = f['Delta_W']
        dw_norm = np.linalg.norm(dw)
        if dw_norm > 1e-10:
            dw_unit = dw / dw_norm
        else:
            dw_unit = np.zeros(d_model)
        
        # At each layer, what fraction of |h| is in the Delta_W direction?
        fracs = []
        for l in range(len(f['all_hidden'])):
            h_l = f['all_hidden'][l]
            h_l_norm = np.linalg.norm(h_l)
            if h_l_norm > 1e-10:
                proj_on_dw = np.dot(h_l, dw_unit)
                frac = abs(proj_on_dw) / h_l_norm
            else:
                frac = 0
            fracs.append(frac)
        gap_fracs.append(fracs)
    
    gap_fracs = np.array(gap_fracs)  # [n_texts, n_layers+1]
    mean_frac = np.mean(gap_fracs, axis=0)
    
    for l in [0, 1, n_layers//2, n_layers-1, n_layers]:
        if l < len(mean_frac):
            print(f"   Layer {l:2d}: |h.Delta_W_unit|/|h| = {mean_frac[l]:.4f}")
    
    # 5. W_U subspace analysis
    print(f"\n5. W_U Subspace Analysis:")
    
    # PCA of W_U
    pca_wu = PCA(n_components=min(100, W_U.shape[0], W_U.shape[1]))
    pca_wu.fit(W_U)
    W_U_pcs = pca_wu.components_
    
    # At each layer, how much of h is in the W_U subspace?
    wu_subspace_fracs = []
    for l in range(n_layers + 1):
        h_l_all = np.array([f['all_hidden'][l] for f in all_features])
        
        # Project h onto top-K W_U PCs
        for k in [10, 50, 100]:
            proj = h_l_all @ W_U_pcs[:k, :].T @ W_U_pcs[:k, :]
            frac = np.mean([np.linalg.norm(proj[i])**2 / (np.linalg.norm(h_l_all[i])**2 + 1e-10) for i in range(n_texts)])
            if k == 10 and l in [0, n_layers//2, n_layers]:
                wu_subspace_fracs.append(frac)
                print(f"   Layer {l:2d}: h in top-10 W_U PCs = {frac:.4f}")
    
    # 6. Gap sign consistency across layers
    print(f"\n6. Gap Sign Consistency:")
    
    # Is the sign of gap_at_layer consistent with gap_at_final?
    sign_consistency = []
    for l in range(n_layers + 1):
        gaps_l = all_gap_traj[:, l]
        consistency = np.mean(np.sign(gaps_l) == np.sign(gaps))
        sign_consistency.append(consistency)
    
    for l in [0, 1, n_layers//2, n_layers-1, n_layers]:
        if l < len(sign_consistency):
            print(f"   Layer {l:2d}: sign consistency with final = {sign_consistency[l]:.4f}")
    
    # 7. Key insight: "Rotation into alignment"
    print(f"\n7. Rotation into Alignment Summary:")
    
    # Does the angle between h and Delta_W decrease at later layers?
    cos_final = np.array([f['cos_with_dw'][-1] for f in all_features])
    cos_early = np.array([f['cos_with_dw'][1] for f in all_features])
    
    print(f"   cos(h, Delta_W) at Layer 1: mean={np.mean(np.abs(cos_early)):.4f}")
    print(f"   cos(h, Delta_W) at Layer L: mean={np.mean(np.abs(cos_final)):.4f}")
    print(f"   Alignment increase: {np.mean(np.abs(cos_final)) - np.mean(np.abs(cos_early)):.4f}")
    
    return {
        'emergence_layer': emergence_layer,
        'max_corr_jump_layer': max_change_layer,
        'max_corr_jump_val': max_change_val,
        'is_sudden': abs(max_change_val) > 3 * np.mean(np.abs(corr_changes)),
        'sign_consistency_first': sign_consistency[1] if len(sign_consistency) > 1 else None,
        'sign_consistency_last': sign_consistency[-1] if sign_consistency else None,
    }


def experiment_p643(all_features, extra, model_name):
    """P643: Corrected Mathematical Framework
    Per-layer linear + last-layer nonlinear emergence = language behavior
    
    Core equation:
    h(l+1) = A_l * h(l) + b_l          (per-layer linear, cos>0.99)
    h(L) = LN(A_{L-1} * h(L-1) + b_{L-1})  (last layer: LN creates alignment)
    gap = h(L) . Delta_W                (linear projection)
    prob = sigmoid(gap)                 (exponential family)
    
    Test: Can we predict gap from h(L-1) + LN model?
    """
    print(f"\n{'='*60}")
    print(f"P643: Corrected Mathematical Framework ({model_name})")
    print(f"{'='*60}")
    
    W_U = extra['W_U']
    d_model = extra['d_model']
    n_layers = extra['n_layers']
    has_final_ln = extra['has_final_ln']
    final_ln_weight = extra['final_ln_weight']
    final_ln_bias = extra['final_ln_bias']
    
    n_texts = len(all_features)
    gaps = np.array([f['logit_gap'] for f in all_features])
    probs = np.array([f['prob'] for f in all_features])
    
    # 1. Corrected pipeline: h(L-1) + LN -> h(L) -> gap
    print(f"\n1. Corrected Pipeline: h(L-1) + LN -> gap")
    
    if has_final_ln:
        h_pre = np.array([f['h_pre_ln'] for f in all_features])
        
        # Apply LN manually
        h_ln_applied = []
        for i in range(n_texts):
            h_raw = h_pre[i]
            h_mean = np.mean(h_raw)
            h_std = np.std(h_raw)
            if h_std > 1e-10:
                h_ln = (h_raw - h_mean) / h_std * final_ln_weight + final_ln_bias
            else:
                h_ln = h_raw.copy()
            h_ln_applied.append(h_ln)
        
        h_ln_applied = np.array(h_ln_applied)
        h_actual = np.array([f['h'] for f in all_features])
        
        # Compare LN-applied with actual h(L)
        cos_ln_vs_actual = []
        for i in range(n_texts):
            cos = np.dot(h_ln_applied[i], h_actual[i]) / (np.linalg.norm(h_ln_applied[i]) * np.linalg.norm(h_actual[i]) + 1e-10)
            cos_ln_vs_actual.append(cos)
        
        print(f"   cos(LN_applied, h_actual): mean={np.mean(cos_ln_vs_actual):.6f}")
        
        # Gap from LN-applied h
        gaps_ln = [np.dot(h_ln_applied[i], all_features[i]['Delta_W']) for i in range(n_texts)]
        r_ln_gap, _ = stats.pearsonr(gaps_ln, gaps)
        print(f"   gap from LN_applied h -> actual gap: r={r_ln_gap:.6f}")
        
        # Prob from LN-applied
        probs_ln = 1.0 / (1.0 + np.exp(-np.array(gaps_ln)))
        r_ln_prob, _ = stats.pearsonr(probs_ln, probs)
        print(f"   prob from LN_applied -> actual prob: r={r_ln_prob:.6f}")
    
    # 2. Alternative: Skip LN, use h(L-1) directly
    print(f"\n2. Alternative: h(L-1) Direct (No LN):")
    
    h_pre_ln = np.array([f['h_pre_ln'] for f in all_features])
    Delta_Ws = np.array([f['Delta_W'] for f in all_features])
    
    gaps_no_ln = np.sum(h_pre_ln * Delta_Ws, axis=1)
    r_no_ln, _ = stats.pearsonr(gaps_no_ln, gaps)
    print(f"   gap from h(L-1) -> actual gap: r={r_no_ln:.4f}")
    
    probs_no_ln = 1.0 / (1.0 + np.exp(-gaps_no_ln))
    r_no_ln_prob, _ = stats.pearsonr(probs_no_ln, probs)
    print(f"   prob from h(L-1) -> actual prob: r={r_no_ln_prob:.4f}")
    
    # 3. h(L-1) -> h(L) transformation analysis
    print(f"\n3. h(L-1) -> h(L) Transformation:")
    
    hL_minus1 = np.array([f['all_hidden'][-2] for f in all_features])
    hL = np.array([f['h'] for f in all_features])
    
    # Linear fit
    reg = Ridge(alpha=1.0)
    reg.fit(hL_minus1, hL)
    hL_pred = reg.predict(hL_minus1)
    
    cos_linear = []
    for i in range(n_texts):
        cos = np.dot(hL[i], hL_pred[i]) / (np.linalg.norm(hL[i]) * np.linalg.norm(hL_pred[i]) + 1e-10)
        cos_linear.append(cos)
    
    print(f"   Linear h(L-1)->h(L) cos: {np.mean(cos_linear):.4f}")
    
    # But gap from linear prediction
    gaps_from_linear = [np.dot(hL_pred[i], all_features[i]['Delta_W']) for i in range(n_texts)]
    r_linear_gap, _ = stats.pearsonr(gaps_from_linear, gaps)
    print(f"   gap from linear h(L-1)->h(L) -> actual gap: r={r_linear_gap:.4f}")
    
    # 4. What does LN add that linear cannot?
    print(f"\n4. What Does LN Add Beyond Linear?")
    
    if has_final_ln:
        # Residual: h(L) - linear_pred
        residuals = hL - hL_pred
        residual_norms = np.linalg.norm(residuals, axis=1)
        
        # Gap contribution from residual
        gaps_from_residual = [np.dot(residuals[i], all_features[i]['Delta_W']) for i in range(n_texts)]
        
        r_residual_gap, _ = stats.pearsonr(gaps_from_residual, gaps)
        print(f"   Residual norm: mean={np.mean(residual_norms):.4f}")
        print(f"   Gap from residual -> actual gap: r={r_residual_gap:.4f}")
        print(f"   Gap from linear -> actual gap: r={r_linear_gap:.4f}")
        
        # Combined
        gaps_combined = np.array(gaps_from_linear) + np.array(gaps_from_residual)
        r_combined, _ = stats.pearsonr(gaps_combined, gaps)
        print(f"   Gap from linear + residual: r={r_combined:.6f}")
        
        # Fraction of gap from residual
        total_gap_sq = np.var(gaps)
        residual_gap_sq = np.var(gaps_from_residual)
        linear_gap_sq = np.var(gaps_from_linear)
        print(f"   Var(gap) = {total_gap_sq:.4f}")
        print(f"   Var(gap_linear) = {linear_gap_sq:.4f}")
        print(f"   Var(gap_residual) = {residual_gap_sq:.4f}")
    
    # 5. Layer-wise gap prediction accuracy
    print(f"\n5. Layer-wise Gap Prediction Accuracy:")
    
    gap_r_by_layer = []
    for l in range(n_layers + 1):
        h_l = np.array([f['all_hidden'][l] for f in all_features])
        gaps_l = np.sum(h_l * Delta_Ws, axis=1)
        r_l, _ = stats.pearsonr(gaps_l, gaps)
        gap_r_by_layer.append(r_l)
    
    # Last 5 layers
    for l in range(max(0, n_layers-4), n_layers+1):
        print(f"   Layer {l:2d}: r={gap_r_by_layer[l]:.4f}")
    
    # 6. Corrected framework prediction
    print(f"\n6. Corrected Framework Prediction:")
    print(f"   Framework: h(l+1) = A_l*h(l) + b_l (per-layer linear)")
    print(f"              h(L) = LN(h(L-1)) (last layer LN)")
    print(f"              gap = h(L) . Delta_W")
    print(f"              prob = sigmoid(gap)")
    
    # Using h(L-1) + LN
    if has_final_ln:
        print(f"   h(L-1) + LN -> gap: r={r_ln_gap:.4f}" if 'r_ln_gap' in dir() else "   N/A")
        print(f"   h(L-1) + LN -> prob: r={r_ln_prob:.4f}" if 'r_ln_prob' in dir() else "   N/A")
        print(f"   h(L-1) direct -> gap: r={r_no_ln:.4f}")
        print(f"   h(L-1) direct -> prob: r={r_no_ln_prob:.4f}")
        print(f"   LN improvement (gap): {r_ln_gap - r_no_ln:.4f}" if 'r_ln_gap' in dir() else "   N/A")
    
    return {
        'r_ln_gap': r_ln_gap if has_final_ln and 'r_ln_gap' in dir() else None,
        'r_ln_prob': r_ln_prob if has_final_ln and 'r_ln_prob' in dir() else None,
        'r_no_ln_gap': r_no_ln,
        'r_no_ln_prob': r_no_ln_prob,
        'r_linear_gap': r_linear_gap,
        'last5_gap_r': gap_r_by_layer[-5:] if len(gap_r_by_layer) >= 5 else gap_r_by_layer,
    }


def experiment_p644(all_features, extra, model_name):
    """P644: Quantitative Validation of Corrected Framework
    Compare prediction accuracy of:
    1. Pure linear: h(0) -> h(L) linear -> gap
    2. Per-layer linear: h(0) -> h(1) -> ... -> h(L) -> gap
    3. Corrected: h(L-1) + LN -> gap
    4. Oracle: h(L) -> gap (upper bound)
    """
    print(f"\n{'='*60}")
    print(f"P644: Quantitative Validation of Corrected Framework ({model_name})")
    print(f"{'='*60}")
    
    W_U = extra['W_U']
    d_model = extra['d_model']
    n_layers = extra['n_layers']
    has_final_ln = extra['has_final_ln']
    final_ln_weight = extra['final_ln_weight']
    final_ln_bias = extra['final_ln_bias']
    
    n_texts = len(all_features)
    gaps = np.array([f['logit_gap'] for f in all_features])
    probs = np.array([f['prob'] for f in all_features])
    Delta_Ws = np.array([f['Delta_W'] for f in all_features])
    
    # 1. Oracle: h(L) -> gap
    print(f"\n1. Oracle (Upper Bound): h(L) -> gap")
    gaps_oracle = np.sum(np.array([f['h'] for f in all_features]) * Delta_Ws, axis=1)
    r_oracle, _ = stats.pearsonr(gaps_oracle, gaps)
    probs_oracle = 1.0 / (1.0 + np.exp(-gaps_oracle))
    r_oracle_prob, _ = stats.pearsonr(probs_oracle, probs)
    print(f"   gap: r={r_oracle:.6f}")
    print(f"   prob: r={r_oracle_prob:.6f}")
    
    # 2. h(L-1) + LN -> gap
    print(f"\n2. Corrected: h(L-1) + LN -> gap")
    
    if has_final_ln:
        h_pre = np.array([f['h_pre_ln'] for f in all_features])
        h_ln = []
        for i in range(n_texts):
            h_raw = h_pre[i]
            h_mean = np.mean(h_raw)
            h_std = np.std(h_raw)
            if h_std > 1e-10:
                h_l = (h_raw - h_mean) / h_std * final_ln_weight + final_ln_bias
            else:
                h_l = h_raw.copy()
            h_ln.append(h_l)
        h_ln = np.array(h_ln)
        
        gaps_ln = np.sum(h_ln * Delta_Ws, axis=1)
        r_ln, _ = stats.pearsonr(gaps_ln, gaps)
        probs_ln = 1.0 / (1.0 + np.exp(-gaps_ln))
        r_ln_prob, _ = stats.pearsonr(probs_ln, probs)
        print(f"   gap: r={r_ln:.6f}")
        print(f"   prob: r={r_ln_prob:.6f}")
        ln_efficiency = r_ln_prob / (r_oracle_prob + 1e-10) * 100
    else:
        print(f"   No final LN, using h(L-1) direct")
        h_pre = np.array([f['all_hidden'][-2] for f in all_features])
        gaps_ln = np.sum(h_pre * Delta_Ws, axis=1)
        r_ln, _ = stats.pearsonr(gaps_ln, gaps)
        probs_ln = 1.0 / (1.0 + np.exp(-gaps_ln))
        r_ln_prob, _ = stats.pearsonr(probs_ln, probs)
        print(f"   gap: r={r_ln:.4f}")
        print(f"   prob: r={r_ln_prob:.4f}")
        ln_efficiency = r_ln_prob / (r_oracle_prob + 1e-10) * 100
    
    # 3. h(L-1) direct (no LN)
    print(f"\n3. h(L-1) Direct (No LN):")
    h_pre_ln = np.array([f['h_pre_ln'] for f in all_features])
    gaps_pre = np.sum(h_pre_ln * Delta_Ws, axis=1)
    r_pre, _ = stats.pearsonr(gaps_pre, gaps)
    probs_pre = 1.0 / (1.0 + np.exp(-gaps_pre))
    r_pre_prob, _ = stats.pearsonr(probs_pre, probs)
    print(f"   gap: r={r_pre:.4f}")
    print(f"   prob: r={r_pre_prob:.4f}")
    pre_efficiency = r_pre_prob / (r_oracle_prob + 1e-10) * 100
    
    # 4. Per-layer propagation
    print(f"\n4. Per-Layer Propagation: h(0) -> h(1) -> ... -> h(L) -> gap")
    
    # Use actual intermediate hidden states (ground truth per layer)
    # But predict next layer from current
    h0 = np.array([f['all_hidden'][0] for f in all_features])
    
    # Propagate through layers using actual h (no prediction error)
    # This tests if the per-layer structure itself preserves gap info
    gap_r_perfect_prop = []
    for l in range(n_layers + 1):
        h_l = np.array([f['all_hidden'][l] for f in all_features])
        gaps_l = np.sum(h_l * Delta_Ws, axis=1)
        r_l, _ = stats.pearsonr(gaps_l, gaps)
        gap_r_perfect_prop.append(r_l)
    
    print(f"   Using actual h(l) at each layer:")
    for l in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1, n_layers]:
        if l < len(gap_r_perfect_prop):
            print(f"     Layer {l:2d}: r={gap_r_perfect_prop[l]:.4f}")
    
    # 5. Cross-validation test
    print(f"\n5. Cross-Validation: h(L-1) + LN -> gap (train/test split)")
    
    n_train = n_texts // 2
    
    if has_final_ln:
        # Feature: h(L-1) with LN applied
        X_ln = h_ln  # Already computed
        y = gaps
        
        # Train Ridge on h(L-1) with LN
        reg_cv = Ridge(alpha=1.0)
        reg_cv.fit(X_ln[:n_train], y[:n_train])
        gaps_cv = reg_cv.predict(X_ln[n_train:])
        r_cv, _ = stats.pearsonr(gaps_cv, y[n_train:])
        print(f"   LN h(L-1) Ridge -> gap (test): r={r_cv:.4f}")
        
        # Train Ridge on h(L-1) without LN
        reg_cv2 = Ridge(alpha=1.0)
        reg_cv2.fit(h_pre_ln[:n_train], y[:n_train])
        gaps_cv2 = reg_cv2.predict(h_pre_ln[n_train:])
        r_cv2, _ = stats.pearsonr(gaps_cv2, y[n_train:])
        print(f"   Raw h(L-1) Ridge -> gap (test): r={r_cv2:.4f}")
        
        # Train Ridge on h(L) (oracle-ish)
        hL = np.array([f['h'] for f in all_features])
        reg_cv3 = Ridge(alpha=1.0)
        reg_cv3.fit(hL[:n_train], y[:n_train])
        gaps_cv3 = reg_cv3.predict(hL[n_train:])
        r_cv3, _ = stats.pearsonr(gaps_cv3, y[n_train:])
        print(f"   h(L) Ridge -> gap (test): r={r_cv3:.4f}")
    
    # 6. Summary comparison
    print(f"\n6. Framework Comparison Summary:")
    print(f"   Oracle (h(L) -> gap):           r={r_oracle:.6f}")
    print(f"   Corrected (h(L-1)+LN -> gap):   r={r_ln:.6f}")
    print(f"   No LN (h(L-1) -> gap):          r={r_pre:.4f}")
    print(f"   LN efficiency: {ln_efficiency:.1f}%")
    print(f"   No-LN efficiency: {pre_efficiency:.1f}%")
    print(f"   LN improvement over no-LN: {ln_efficiency - pre_efficiency:.1f}%")
    
    # 7. Emergence quantification
    print(f"\n7. Emergence Quantification:")
    
    # How much does the last layer improve gap prediction?
    if len(gap_r_perfect_prop) >= 2:
        r_before_last = gap_r_perfect_prop[-2]
        r_at_last = gap_r_perfect_prop[-1]
        emergence_boost = r_at_last - r_before_last
        print(f"   r(gap at L-1, gap at L) = {r_before_last:.4f}")
        print(f"   r(gap at L, gap at L)   = {r_at_last:.4f}")
        print(f"   Emergence boost: {emergence_boost:.4f}")
        print(f"   => Last layer adds {emergence_boost/abs(r_at_last)*100:.1f}% of gap predictability")
    
    return {
        'r_oracle': r_oracle,
        'r_oracle_prob': r_oracle_prob,
        'r_ln': r_ln,
        'r_ln_prob': r_ln_prob if has_final_ln else r_ln,
        'r_pre': r_pre,
        'r_pre_prob': r_pre_prob,
        'ln_efficiency': ln_efficiency,
        'pre_efficiency': pre_efficiency,
        'emergence_boost': emergence_boost if 'emergence_boost' in dir() else None,
    }


def save_results(results, model_name, experiment_name, output_dir="results/phase_cxlv"):
    """Save experiment results."""
    os.makedirs(output_dir, exist_ok=True)
    
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    results = convert(results)
    
    filename = f"{output_dir}/{model_name}_{experiment_name}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"   Results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Phase CXLV: Last-Layer Emergence")
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True, 
                       choices=["p641", "p642", "p643", "p644"])
    args = parser.parse_args()
    
    model_name = args.model
    experiment = args.experiment
    
    print(f"\n{'#'*60}")
    print(f"# Phase CXLV: {experiment.upper()} - {model_name}")
    print(f"# Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")
    
    # Load model
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"Model: {model_info.name}, d_model={model_info.d_model}, n_layers={model_info.n_layers}")
    
    # Compute features
    print(f"\nComputing features for {len(TEST_TEXTS)} texts...")
    all_features, extra = compute_features(model, tokenizer, device, TEST_TEXTS)
    
    # Run experiment
    if experiment == "p641":
        results = experiment_p641(all_features, extra, model_name)
    elif experiment == "p642":
        results = experiment_p642(all_features, extra, model_name)
    elif experiment == "p643":
        results = experiment_p643(all_features, extra, model_name)
    elif experiment == "p644":
        results = experiment_p644(all_features, extra, model_name)
    
    # Save results
    save_results(results, model_name, experiment)
    
    # Release model
    release_model(model)
    
    print(f"\n{'#'*60}")
    print(f"# {experiment.upper()} completed for {model_name}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
