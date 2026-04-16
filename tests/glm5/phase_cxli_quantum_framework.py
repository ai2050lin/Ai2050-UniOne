#!/usr/bin/env python3
"""
Phase CXLI: Quantum Acoustics Mathematical Framework Rigorization (P619-P624)
Focus: Establishing precise mathematical framework based on P613-P618 evidence

P619: Complex h representation - h_complex(l) = h(l) + i*h(l-1)
      Test: Is h_complex propagating via a unitary-like transformation?
P620: Phase propagation equation - phase(l+1) = f(phase(l), W_down)
      Test: Can we predict phase(l+1) from phase(l) and W_down?
P621: Coherence term analytical form - C_kj = <psi_k|psi_j>
      Test: What is the exact mathematical form of sign correlation C_kj?
P622: Collapse mechanism - from complex h to real logit
      Test: Is there a deterministic "measurement" process?
P623: Quantum-like interference in logit computation
      Test: Do cross-direction interference terms improve prob prediction?
P624: Unified quantum acoustics equation
      Test: prob = f(h_complex, W_U) - what is the best functional form?
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
from sklearn.linear_model import LinearRegression
from model_utils import load_model, get_model_info

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

def compute_all_hidden_states(model, tokenizer, device, text, max_length=128):
    """Get hidden states from ALL layers for a single text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    
    # Get all hidden states: tuple of (1, seq_len, d_model)
    all_hidden_states = []
    for hs in outputs.hidden_states:
        all_hidden_states.append(hs[0, -1, :].cpu().float().numpy())
    
    logits = outputs.logits[0, -1, :].cpu().float().numpy()
    
    return all_hidden_states, logits

def compute_features_multi_layer(model, tokenizer, device, texts):
    """Compute multi-layer features for all texts."""
    # Get W_U
    if hasattr(model, 'lm_head'):
        W_U = model.lm_head.weight.detach().cpu().float().numpy()
    else:
        W_U = model.get_output_embeddings().weight.detach().cpu().float().numpy()
    
    # Get model config directly from model
    if hasattr(model.config, 'hidden_size'):
        d_model = model.config.hidden_size
    elif hasattr(model.config, 'd_model'):
        d_model = model.config.d_model
    else:
        d_model = model.config.num_hidden_states
    
    if hasattr(model.config, 'num_hidden_layers'):
        n_layers = model.config.num_hidden_layers
    elif hasattr(model.config, 'n_layer'):
        n_layers = model.config.n_layer
    else:
        n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 32
    
    all_features = []
    
    for i, text in enumerate(texts):
        if (i+1) % 10 == 0:
            print(f"  Processing text {i+1}/{len(texts)}...")
        
        all_hidden, logits = compute_all_hidden_states(model, tokenizer, device, text)
        
        # Last layer h
        h = all_hidden[-1]
        
        # Token info
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        logit_gap = logits[top1_idx] - logits[top2_idx]
        logit_max = logits[top1_idx]
        prob = 1.0 / (1.0 + np.exp(-logit_gap))
        
        # W_U structure
        W_U_top1 = W_U[top1_idx]
        W_U_top2 = W_U[top2_idx]
        Delta_W = W_U_top1 - W_U_top2
        
        # SVD of h
        U_h, S_h, Vt_h = np.linalg.svd(h.reshape(1, -1), full_matrices=False)
        # Actually for a vector, SVD is trivial. Let's compute directional decomposition.
        # h = sum_k c_k * e_k where e_k are standard basis... 
        # For SVD of h as a vector, it's just h/||h|| * ||h||
        h_norm = np.linalg.norm(h)
        
        # Complex h representation: h_complex(l) = h(l) + i*h(l-1)
        # For last two layers
        h_last = all_hidden[-1]
        h_prev = all_hidden[-2]
        h_complex = h_last + 1j * h_prev  # Complex vector
        
        # Complex inner product with Delta_W
        Delta_W_complex = Delta_W.astype(complex)
        complex_dot = np.vdot(h_complex, Delta_W_complex)  # <h_complex | Delta_W>
        complex_amplitude = np.abs(complex_dot)
        complex_phase = np.angle(complex_dot)
        
        # Compute h(l) and h(l-1) projections onto Delta_W
        proj_last = np.dot(h_last, Delta_W)
        proj_prev = np.dot(h_prev, Delta_W)
        
        # Phase from layer pair
        phase_from_pair = np.arctan2(proj_prev, proj_last)
        
        features = {
            'text_idx': i,
            'h': h,
            'h_last': h_last,
            'h_prev': h_prev,
            'h_complex': h_complex,
            'all_hidden': all_hidden,
            'logits': logits,
            'logit_gap': logit_gap,
            'logit_max': logit_max,
            'prob': prob,
            'top1_idx': top1_idx,
            'top2_idx': top2_idx,
            'Delta_W': Delta_W,
            'h_norm': h_norm,
            'W_U': W_U,
            'd_model': d_model,
            'n_layers': n_layers,
            # Complex features
            'complex_amplitude': complex_amplitude,
            'complex_phase': complex_phase,
            'proj_last': proj_last,
            'proj_prev': proj_prev,
            'phase_from_pair': phase_from_pair,
        }
        
        all_features.append(features)
    
    return all_features


def experiment_p619(all_features, model_name, model=None):
    """P619: Complex h representation and unitary-like propagation."""
    print(f"\nP619: Complex h representation -- {model_name}")
    print("Testing: Does h_complex(l) = h(l) + i*h(l-1) propagate unitarily?")
    
    n_layers = all_features[0]['n_layers']
    
    # For each text, compute complex propagation ratios
    all_norm_ratios = []
    all_phase_shifts = []
    all_real_to_imag_ratios = []
    all_norm_preservations = []
    
    for feat in all_features:
        all_h = feat['all_hidden']
        
        # Compute ||h(l)|| and ||h(l-1)|| for all layers
        norms = [np.linalg.norm(h) for h in all_h]
        
        # Complex norm: ||h_complex|| = sqrt(||h_last||^2 + ||h_prev||^2)
        # Compare with individual norms
        for l in range(2, n_layers):
            h_l = all_h[l]
            h_l_prev = all_h[l-1]
            h_l_prev2 = all_h[l-2]
            
            norm_l = np.linalg.norm(h_l)
            norm_l_prev = np.linalg.norm(h_l_prev)
            
            # Real-to-imaginary norm ratio
            if norm_l_prev > 0:
                r2i = norm_l / norm_l_prev
            else:
                r2i = 0
            all_real_to_imag_ratios.append(r2i)
            
            # Norm preservation: ||h(l+1)|| should relate to ||h(l)|| if unitary
            if l > 2:
                norm_pres = norm_l / np.linalg.norm(all_h[l-1]) if np.linalg.norm(all_h[l-1]) > 0 else 0
                all_norm_preservations.append(norm_pres)
    
    # Statistics
    r2i_mean = np.mean(all_real_to_imag_ratios)
    r2i_std = np.std(all_real_to_imag_ratios)
    r2i_cv = r2i_std / r2i_mean if r2i_mean > 0 else float('inf')
    
    norm_pres_mean = np.mean(all_norm_preservations) if all_norm_preservations else 0
    norm_pres_std = np.std(all_norm_preservations) if all_norm_preservations else 0
    
    # Test: Complex amplitude -> prob correlation
    amplitudes = [f['complex_amplitude'] for f in all_features]
    phases = [f['complex_phase'] for f in all_features]
    probs = [f['prob'] for f in all_features]
    gaps = [f['logit_gap'] for f in all_features]
    
    r_amp_prob, _ = stats.pearsonr(amplitudes, probs)
    r_amp_gap, _ = stats.pearsonr(amplitudes, gaps)
    r_phase_prob, _ = stats.pearsonr(phases, probs)
    r_phase_gap, _ = stats.pearsonr(phases, gaps)
    
    # Complex projection: Re(<h_complex|Delta_W>) and Im(<h_complex|Delta_W>)
    re_dots = [f['proj_last'] for f in all_features]
    im_dots = [f['proj_prev'] for f in all_features]
    
    r_re_gap, _ = stats.pearsonr(re_dots, gaps)
    r_im_gap, _ = stats.pearsonr(im_dots, gaps)
    
    # Combined: amplitude and phase predict gap
    X = np.column_stack([amplitudes, phases])
    y = np.array(gaps)
    reg = LinearRegression().fit(X, y)
    r_combined = reg.score(X, y)
    
    # Complex modulus prediction: |<h_complex|Delta_W>| vs |<h_last|Delta_W>|
    moduli = [np.abs(f['proj_last'] + 1j * f['proj_prev']) for f in all_features]
    r_modulus_gap, _ = stats.pearsonr(moduli, gaps)
    
    print(f"  Real/imag norm ratio: mean={r2i_mean:.4f}, std={r2i_std:.4f}, CV={r2i_cv:.4f}")
    print(f"  Norm preservation: mean={norm_pres_mean:.4f}, std={norm_pres_std:.4f}")
    print(f"  Complex amplitude->prob r={r_amp_prob:.4f}")
    print(f"  Complex amplitude->gap r={r_amp_gap:.4f}")
    print(f"  Complex phase->prob r={r_phase_prob:.4f}")
    print(f"  Complex phase->gap r={r_phase_gap:.4f}")
    print(f"  Re(<h|Delta>)=proj_last->gap r={r_re_gap:.4f}")
    print(f"  Im(<h|Delta>)=proj_prev->gap r={r_im_gap:.4f}")
    print(f"  |complex proj|->gap r={r_modulus_gap:.4f}")
    print(f"  Combined(amp+phase)->gap R2={r_combined:.4f}")
    
    results = {
        'experiment': 'P619',
        'model': model_name,
        'r2i_mean': r2i_mean, 'r2i_cv': r2i_cv,
        'norm_pres_mean': norm_pres_mean,
        'r_amp_prob': r_amp_prob, 'r_amp_gap': r_amp_gap,
        'r_phase_prob': r_phase_prob, 'r_phase_gap': r_phase_gap,
        'r_re_gap': r_re_gap, 'r_im_gap': r_im_gap,
        'r_modulus_gap': r_modulus_gap,
        'r_combined_amp_phase': r_combined,
    }
    
    return results


def experiment_p620(all_features, model_name, model=None):
    """P620: Phase propagation equation across layers."""
    print(f"\nP620: Phase propagation equation -- {model_name}")
    print("Testing: Can we predict phase(l+1) from phase(l)?")
    
    n_layers = all_features[0]['n_layers']
    d_model = all_features[0]['d_model']
    
    # For each text, compute phase at each layer relative to Delta_W
    # Phase(l) = angle of projection of h(l) onto a fixed reference direction
    # Use Delta_W as reference direction (but it depends on text - problem!)
    # Instead, use the first SVD direction of h at last layer as reference
    
    all_phase_trajectories = []
    all_phase_diffs = []  # phase(l+1) - phase(l)
    
    for feat in all_features:
        all_h = feat['all_hidden']
        Delta_W = feat['Delta_W']
        
        phases = []
        for l in range(n_layers):
            proj_re = np.dot(all_h[l], Delta_W)
            # For imaginary part, we need a perpendicular direction
            # Use the component of h(l) perpendicular to Delta_W
            proj_perp = np.linalg.norm(all_h[l] - proj_re * Delta_W / (np.dot(Delta_W, Delta_W) + 1e-10))
            phase_l = np.arctan2(proj_perp, proj_re)
            phases.append(phase_l)
        
        all_phase_trajectories.append(phases)
        
        # Phase differences
        for l in range(1, n_layers):
            all_phase_diffs.append(phases[l] - phases[l-1])
    
    # Statistics of phase differences
    phase_diff_mean = np.mean(all_phase_diffs)
    phase_diff_std = np.std(all_phase_diffs)
    phase_diff_cv = phase_diff_std / abs(phase_diff_mean) if abs(phase_diff_mean) > 1e-10 else float('inf')
    
    # Test: Is phase(l+1) = phase(l) + delta (constant shift)?
    # Compute R2 of this prediction
    predicted_phases = []
    actual_phases = []
    for phases in all_phase_trajectories:
        for l in range(1, len(phases)):
            predicted_phases.append(phases[l-1] + phase_diff_mean)
            actual_phases.append(phases[l])
    
    r_phase_prop, _ = stats.pearsonr(predicted_phases, actual_phases)
    
    # Better: use linear regression phase(l) = a * phase(l-1) + b
    # across all texts and layers
    X_phase = []
    y_phase = []
    for phases in all_phase_trajectories:
        for l in range(1, len(phases)):
            X_phase.append(phases[l-1])
            y_phase.append(phases[l])
    
    X_phase = np.array(X_phase).reshape(-1, 1)
    y_phase = np.array(y_phase)
    reg = LinearRegression().fit(X_phase, y_phase)
    r_phase_reg = reg.score(X_phase, y_phase)
    phase_a = reg.coef_[0]
    phase_b = reg.intercept_
    
    # Test: Complex norm propagation
    # ||h_complex(l)|| = sqrt(||h(l)||^2 + ||h(l-1)||^2)
    # Does ||h_complex(l+1)||/||h_complex(l)|| remain constant?
    complex_norm_ratios = []
    for feat in all_features:
        all_h = feat['all_hidden']
        for l in range(2, n_layers - 1):
            cn_l = np.sqrt(np.linalg.norm(all_h[l])**2 + np.linalg.norm(all_h[l-1])**2)
            cn_l1 = np.sqrt(np.linalg.norm(all_h[l+1])**2 + np.linalg.norm(all_h[l])**2)
            if cn_l > 0:
                complex_norm_ratios.append(cn_l1 / cn_l)
    
    cnr_mean = np.mean(complex_norm_ratios) if complex_norm_ratios else 0
    cnr_std = np.std(complex_norm_ratios) if complex_norm_ratios else 0
    cnr_cv = cnr_std / cnr_mean if cnr_mean > 0 else float('inf')
    
    print(f"  Phase diff mean={phase_diff_mean:.4f}, std={phase_diff_std:.4f}, CV={phase_diff_cv:.4f}")
    print(f"  Phase propagation r(constant shift)={r_phase_prop:.4f}")
    print(f"  Phase propagation R2(linear reg)={r_phase_reg:.4f}, a={phase_a:.4f}, b={phase_b:.4f}")
    print(f"  Complex norm ratio mean={cnr_mean:.4f}, std={cnr_std:.4f}, CV={cnr_cv:.4f}")
    
    results = {
        'experiment': 'P620',
        'model': model_name,
        'phase_diff_mean': phase_diff_mean, 'phase_diff_cv': phase_diff_cv,
        'r_phase_prop': r_phase_prop, 'r_phase_reg': r_phase_reg,
        'phase_a': phase_a, 'phase_b': phase_b,
        'cnr_mean': cnr_mean, 'cnr_cv': cnr_cv,
    }
    
    return results


def experiment_p621(all_features, model_name, model=None):
    """P621: Coherence term analytical form."""
    print(f"\nP621: Coherence term C_kj analytical form -- {model_name}")
    print("Testing: What is the exact mathematical form of sign correlation C_kj?")
    
    # From P613: sign correlation matrix has structure
    # Now: decompose C_kj into contributions from different mechanisms
    
    n_texts = len(all_features)
    d_model = all_features[0]['d_model']
    
    # Compute sign vectors for top-K directions
    K = 30  # Use top-30 directions for tractability
    
    sign_matrix = np.zeros((n_texts, K))  # (texts, directions)
    c_k_matrix = np.zeros((n_texts, K))   # |c_k| values
    delta_k_matrix = np.zeros((n_texts, K))  # |Delta_k| values
    
    for i, feat in enumerate(all_features):
        h = feat['h']
        Delta_W = feat['Delta_W']
        h_norm = feat['h_norm']
        
        # Compute per-direction contributions
        # h[k] * Delta_W[k] for each dimension k
        contributions = h * Delta_W  # element-wise
        
        # Sort by |contribution| and take top-K
        abs_contrib = np.abs(contributions)
        top_k_idx = np.argsort(abs_contrib)[-K:]
        
        for j, k in enumerate(top_k_idx):
            sign_matrix[i, j] = np.sign(contributions[k])
            c_k_matrix[i, j] = np.abs(contributions[k])
            delta_k_matrix[i, j] = np.abs(Delta_W[k])
    
    # Compute sign correlation matrix C_kj
    C = np.corrcoef(sign_matrix.T)  # (K, K)
    
    # Decompose C_kj:
    # Hypothesis 1: C_kj = f(|c_k|, |c_j|) - correlation from amplitude coupling
    # Hypothesis 2: C_kj = f(h_norm) - correlation from overall scale
    # Hypothesis 3: C_kj = f(Delta_W structure) - correlation from W_U geometry
    
    # Test H1: |c_k| * |c_j| predicts C_kj
    amp_products = []
    c_values = []
    for k in range(K):
        for j in range(k+1, K):
            amp_products.append(np.mean(c_k_matrix[:, k]) * np.mean(c_k_matrix[:, j]))
            c_values.append(C[k, j])
    
    r_amp_c, _ = stats.pearsonr(amp_products, c_values)
    
    # Test H2: h_norm predicts mean |C_kj|
    h_norms = [f['h_norm'] for f in all_features]
    # For each text, compute mean sign correlation
    mean_sign_corr_per_text = []
    for i in range(n_texts):
        signs_i = sign_matrix[i]
        # Mean correlation between all pairs of directions for this text
        # Since sign is +/-1, correlation between two signs is just sign_k * sign_j
        pairwise = []
        for k in range(K):
            for j in range(k+1, K):
                pairwise.append(signs_i[k] * signs_i[j])
        mean_sign_corr_per_text.append(np.mean(pairwise))
    
    r_hnorm_signcorr, _ = stats.pearsonr(h_norms, mean_sign_corr_per_text)
    
    # Test H3: Delta_W structure predicts C_kj
    # Delta_W correlation matrix
    delta_corr = np.corrcoef(delta_k_matrix.T)
    upper_delta = []
    upper_C = []
    for k in range(K):
        for j in range(k+1, K):
            upper_delta.append(delta_corr[k, j])
            upper_C.append(C[k, j])
    
    r_delta_c, _ = stats.pearsonr(upper_delta, upper_C)
    
    # Test H4: Phase coherence predicts C_kj
    # Use complex representation to compute phase correlations
    phase_corr_values = []
    for k in range(K):
        for j in range(k+1, K):
            # Phase coherence: how often do signs of k and j agree?
            agreement = np.mean(sign_matrix[:, k] * sign_matrix[:, j])
            phase_corr_values.append(agreement)
    
    # Eigenvalue spectrum of C
    eigenvalues = np.linalg.eigvalsh(C)
    eigenvalues = eigenvalues[::-1]  # Descending order
    top_3_ratio = np.sum(eigenvalues[:3]) / np.sum(eigenvalues) if np.sum(eigenvalues) > 0 else 0
    
    # Participation ratio of eigenvalues
    PR = np.sum(eigenvalues)**2 / np.sum(eigenvalues**2) if np.sum(eigenvalues**2) > 0 else 0
    
    print(f"  H1: |c_k|*|c_j| predicts C_kj: r={r_amp_c:.4f}")
    print(f"  H2: h_norm predicts mean sign corr: r={r_hnorm_signcorr:.4f}")
    print(f"  H3: Delta_W corr predicts C_kj: r={r_delta_c:.4f}")
    print(f"  Top-3 eigenvalue ratio: {top_3_ratio:.4f}")
    print(f"  Participation ratio: {PR:.4f}")
    print(f"  Top-5 eigenvalues: {eigenvalues[:5].tolist()}")
    
    results = {
        'experiment': 'P621',
        'model': model_name,
        'r_amp_c': r_amp_c,
        'r_hnorm_signcorr': r_hnorm_signcorr,
        'r_delta_c': r_delta_c,
        'top3_ratio': top_3_ratio,
        'PR': PR,
        'top5_eigenvalues': eigenvalues[:5].tolist(),
    }
    
    return results


def experiment_p622(all_features, model_name, model=None):
    """P622: Collapse mechanism from complex h to real logit."""
    print(f"\nP622: Collapse mechanism -- {model_name}")
    print("Testing: Is there a deterministic measurement process?")
    
    # Quantum analogy: |psi> -> measurement -> classical outcome
    # In Transformer: h_complex -> logit computation -> argmax
    # 
    # Test: Can we model the "collapse" as projection onto W_U directions?
    
    probs = [f['prob'] for f in all_features]
    gaps = [f['logit_gap'] for f in all_features]
    
    # Model 1: Classical projection (already known)
    # logit_gap = <h, Delta_W> = Re(<h_complex, Delta_W>)
    proj_last = [f['proj_last'] for f in all_features]
    r_classical, _ = stats.pearsonr(proj_last, gaps)
    
    # Model 2: Complex projection
    # logit_gap = |<h_complex, Delta_W>| * cos(phase)
    complex_dots = [f['proj_last'] + 1j * f['proj_prev'] for f in all_features]
    complex_moduli = [np.abs(cd) for cd in complex_dots]
    complex_phases = [np.angle(cd) for cd in complex_dots]
    
    r_modulus_gap, _ = stats.pearsonr(complex_moduli, gaps)
    
    # Model 3: Born rule analog
    # prob = |<h_complex|top1>|^2 / (|<h_complex|top1>|^2 + |<h_complex|top2>|^2)
    W_U = all_features[0]['W_U']
    
    born_probs = []
    for feat in all_features:
        h_complex = feat['h_complex']
        top1_idx = feat['top1_idx']
        top2_idx = feat['top2_idx']
        
        # Complex inner products
        amp1 = np.abs(np.vdot(h_complex, W_U[top1_idx].astype(complex)))
        amp2 = np.abs(np.vdot(h_complex, W_U[top2_idx].astype(complex)))
        
        born_prob = amp1**2 / (amp1**2 + amp2**2 + 1e-10)
        born_probs.append(born_prob)
    
    r_born_prob, _ = stats.pearsonr(born_probs, probs)
    
    # Model 4: Cosine-squared rule
    # prob = cos^2(angle between h_complex and W_U[top1])
    cos2_probs = []
    for feat in all_features:
        h_complex = feat['h_complex']
        top1_idx = feat['top1_idx']
        
        amp1 = np.vdot(h_complex, W_U[top1_idx].astype(complex))
        h_norm_c = np.linalg.norm(h_complex)
        w_norm = np.linalg.norm(W_U[top1_idx])
        
        if h_norm_c > 0 and w_norm > 0:
            cos_angle = np.abs(amp1) / (h_norm_c * w_norm + 1e-10)
            cos2_prob = cos_angle**2
        else:
            cos2_prob = 0.5
        cos2_probs.append(cos2_prob)
    
    r_cos2_prob, _ = stats.pearsonr(cos2_probs, probs)
    
    # Model 5: Sigmoid of complex amplitude
    # prob = sigmoid(|<h_complex|Delta_W>|)
    sig_complex_probs = [1.0 / (1.0 + np.exp(-np.abs(cd))) for cd in complex_dots]
    r_sig_complex, _ = stats.pearsonr(sig_complex_probs, probs)
    
    # Model 6: Phase-weighted sigmoid
    # prob = sigmoid(Re(<h_complex|Delta_W>) * cos(phase))
    phase_weighted = [np.real(cd) * np.cos(np.angle(cd)) for cd in complex_dots]
    r_phase_weighted, _ = stats.pearsonr(phase_weighted, gaps)
    
    # Compare all models
    print(f"  Classical Re(<h|Delta>)->gap r={r_classical:.4f}")
    print(f"  |<h_complex|Delta>|->gap r={r_modulus_gap:.4f}")
    print(f"  Born rule -> prob r={r_born_prob:.4f}")
    print(f"  cos^2(angle) -> prob r={r_cos2_prob:.4f}")
    print(f"  sigmoid(|complex_dot|) -> prob r={r_sig_complex:.4f}")
    print(f"  Phase-weighted -> gap r={r_phase_weighted:.4f}")
    
    # Best collapse model
    models = {
        'classical': r_classical,
        'modulus': r_modulus_gap,
        'born': r_born_prob,
        'cos2': r_cos2_prob,
        'sig_complex': r_sig_complex,
        'phase_weighted': r_phase_weighted,
    }
    best_model = max(models, key=lambda k: abs(models[k]))
    print(f"  Best collapse model: {best_model} (r={models[best_model]:.4f})")
    
    results = {
        'experiment': 'P622',
        'model': model_name,
        'r_classical': r_classical,
        'r_modulus_gap': r_modulus_gap,
        'r_born_prob': r_born_prob,
        'r_cos2_prob': r_cos2_prob,
        'r_sig_complex': r_sig_complex,
        'r_phase_weighted': r_phase_weighted,
        'best_model': best_model,
    }
    
    return results


def experiment_p623(all_features, model_name, model=None):
    """P623: Quantum-like interference in logit computation."""
    print(f"\nP623: Quantum-like interference -- {model_name}")
    print("Testing: Do cross-direction interference terms improve prob prediction?")
    
    # In quantum mechanics: |a+b|^2 = |a|^2 + |b|^2 + 2*Re(a*·b)
    # The cross term 2*Re(a*·b) is the interference term
    # 
    # In Transformer: logit_gap = sum_k h[k] * Delta_W[k]
    # = sum_k |h[k]|*|Delta_W[k]| * sign(h[k]*Delta_W[k])
    # 
    # If we group directions into "channels", we can look for interference
    
    probs = [f['prob'] for f in all_features]
    gaps = [f['logit_gap'] for f in all_features]
    W_U = all_features[0]['W_U']
    d_model = all_features[0]['d_model']
    
    # Split h into frequency groups based on SVD
    # Group 1: top-10% dimensions, Group 2: next 20%, Group 3: remaining
    
    K1 = max(1, d_model // 10)  # top 10%
    K2 = max(1, d_model // 5)   # next 20%
    
    group1_gaps = []
    group2_gaps = []
    group3_gaps = []
    interference_12 = []
    interference_13 = []
    interference_23 = []
    
    for feat in all_features:
        h = feat['h']
        Delta_W = feat['Delta_W']
        
        contrib = h * Delta_W  # element-wise
        abs_contrib = np.abs(contrib)
        sorted_idx = np.argsort(abs_contrib)[::-1]
        
        # Group 1: top K1 dimensions
        idx1 = sorted_idx[:K1]
        # Group 2: next K2 dimensions
        idx2 = sorted_idx[K1:K1+K2]
        # Group 3: remaining
        idx3 = sorted_idx[K1+K2:]
        
        gap1 = np.sum(contrib[idx1])
        gap2 = np.sum(contrib[idx2])
        gap3 = np.sum(contrib[idx3])
        
        group1_gaps.append(gap1)
        group2_gaps.append(gap2)
        group3_gaps.append(gap3)
        
        # Interference terms: sign coherence between groups
        # If signs in group1 and group2 are correlated, interference is constructive
        signs1 = np.sign(contrib[idx1])
        signs2 = np.sign(contrib[idx2])
        signs3 = np.sign(contrib[idx3])
        
        # Mean sign agreement between groups
        int12 = np.mean(signs1) * np.mean(signs2)  # If both have same bias, constructive
        int13 = np.mean(signs1) * np.mean(signs3)
        int23 = np.mean(signs2) * np.mean(signs3)
        
        interference_12.append(int12)
        interference_13.append(int13)
        interference_23.append(int23)
    
    # Test: Do group gaps predict total gap?
    X_groups = np.column_stack([group1_gaps, group2_gaps, group3_gaps])
    y_gaps = np.array(gaps)
    reg_groups = LinearRegression().fit(X_groups, y_gaps)
    r_groups = reg_groups.score(X_groups, y_gaps)
    
    # Test: Does adding interference terms improve prediction?
    X_with_int = np.column_stack([group1_gaps, group2_gaps, group3_gaps, 
                                   interference_12, interference_13, interference_23])
    reg_with_int = LinearRegression().fit(X_with_int, y_gaps)
    r_with_int = reg_with_int.score(X_with_int, y_gaps)
    
    # Test: Interference -> prob
    r_int12_prob, _ = stats.pearsonr(interference_12, probs)
    r_int13_prob, _ = stats.pearsonr(interference_13, probs)
    r_int23_prob, _ = stats.pearsonr(interference_23, probs)
    
    # Test: Phase coherence across groups
    # Use complex representation to compute cross-group coherence
    cross_group_coherence = []
    for feat in all_features:
        h_complex = feat['h_complex']
        h_last = feat['h_last']
        h_prev = feat['h_prev']
        Delta_W = feat['Delta_W']
        
        contrib_last = h_last * Delta_W
        contrib_prev = h_prev * Delta_W
        abs_contrib = np.abs(contrib_last)
        sorted_idx = np.argsort(abs_contrib)[::-1]
        
        idx1 = sorted_idx[:K1]
        idx2 = sorted_idx[K1:K1+K2]
        
        # Complex contributions in each group
        c1 = np.sum(contrib_last[idx1]) + 1j * np.sum(contrib_prev[idx1])
        c2 = np.sum(contrib_last[idx2]) + 1j * np.sum(contrib_prev[idx2])
        
        # Cross-group coherence: phase alignment
        if np.abs(c1) > 0 and np.abs(c2) > 0:
            coherence = np.cos(np.angle(c1) - np.angle(c2))
        else:
            coherence = 0
        cross_group_coherence.append(coherence)
    
    r_coherence_prob, _ = stats.pearsonr(cross_group_coherence, probs)
    r_coherence_gap, _ = stats.pearsonr(cross_group_coherence, gaps)
    
    print(f"  Groups->gap R2 (without interference): {r_groups:.4f}")
    print(f"  Groups+interference->gap R2: {r_with_int:.4f}")
    print(f"  Improvement from interference: {r_with_int - r_groups:.4f}")
    print(f"  Interference 1-2 -> prob r={r_int12_prob:.4f}")
    print(f"  Interference 1-3 -> prob r={r_int13_prob:.4f}")
    print(f"  Interference 2-3 -> prob r={r_int23_prob:.4f}")
    print(f"  Cross-group coherence -> prob r={r_coherence_prob:.4f}")
    print(f"  Cross-group coherence -> gap r={r_coherence_gap:.4f}")
    
    results = {
        'experiment': 'P623',
        'model': model_name,
        'r_groups': r_groups,
        'r_with_int': r_with_int,
        'int_improvement': r_with_int - r_groups,
        'r_int12_prob': r_int12_prob,
        'r_int13_prob': r_int13_prob,
        'r_int23_prob': r_int23_prob,
        'r_coherence_prob': r_coherence_prob,
        'r_coherence_gap': r_coherence_gap,
    }
    
    return results


def experiment_p624(all_features, model_name, model=None):
    """P624: Unified quantum acoustics equation."""
    print(f"\nP624: Unified quantum acoustics equation -- {model_name}")
    print("Testing: prob = f(h_complex, W_U) - best functional form")
    
    probs = np.array([f['prob'] for f in all_features])
    gaps = np.array([f['logit_gap'] for f in all_features])
    W_U = all_features[0]['W_U']
    
    # Model 1: Classical sigmoid(gap) - baseline
    sig_probs = 1.0 / (1.0 + np.exp(-gaps))
    r_sigmoid, _ = stats.pearsonr(sig_probs, probs)
    
    # Model 2: Complex sigmoid
    complex_dots = np.array([f['proj_last'] + 1j * f['proj_prev'] for f in all_features])
    complex_amps = np.abs(complex_dots)
    complex_phases = np.angle(complex_dots)
    sig_complex = 1.0 / (1.0 + np.exp(-complex_amps))
    r_sig_complex, _ = stats.pearsonr(sig_complex, probs)
    
    # Model 3: Born rule
    born_probs = []
    for feat in all_features:
        h_complex = feat['h_complex']
        top1_idx = feat['top1_idx']
        top2_idx = feat['top2_idx']
        amp1 = np.abs(np.vdot(h_complex, W_U[top1_idx].astype(complex)))
        amp2 = np.abs(np.vdot(h_complex, W_U[top2_idx].astype(complex)))
        born_probs.append(amp1**2 / (amp1**2 + amp2**2 + 1e-10))
    r_born, _ = stats.pearsonr(born_probs, probs)
    
    # Model 4: Hybrid sigmoid + quantum phase
    # prob = sigmoid(gap + w_q * quantum_correction)
    quantum_corr = np.real(complex_dots) - gaps  # Difference between complex and real projection
    # Actually, real part of complex_dot IS the gap, so this is zero. Let me think differently.
    
    # Phase correction: use cos(phase) * |complex_dot| as a corrected gap
    phase_corrected_gap = np.real(complex_dots) * np.cos(complex_phases)
    sig_phase = 1.0 / (1.0 + np.exp(-phase_corrected_gap))
    r_sig_phase, _ = stats.pearsonr(sig_phase, probs)
    
    # Model 5: Multi-feature quantum regression
    features_list = []
    for feat in all_features:
        h_complex = feat['h_complex']
        h_norm = feat['h_norm']
        top1_idx = feat['top1_idx']
        top2_idx = feat['top2_idx']
        
        amp1 = np.abs(np.vdot(h_complex, W_U[top1_idx].astype(complex)))
        amp2 = np.abs(np.vdot(h_complex, W_U[top2_idx].astype(complex)))
        
        row = [
            feat['logit_gap'],          # Classical gap
            feat['complex_amplitude'],   # Complex amplitude
            feat['complex_phase'],       # Complex phase
            h_norm,                      # h L2 norm
            amp1 / (amp2 + 1e-10),      # Amplitude ratio
            np.cos(feat['complex_phase']),  # Phase cosine
            np.sin(feat['complex_phase']),  # Phase sine
        ]
        features_list.append(row)
    
    X_full = np.array(features_list)
    y_prob = probs
    
    # Full model
    reg_full = LinearRegression().fit(X_full, y_prob)
    r_full = reg_full.score(X_full, y_prob)
    
    # Without classical gap (pure quantum)
    X_quantum = X_full[:, 1:]  # Remove gap
    reg_quantum = LinearRegression().fit(X_quantum, y_prob)
    r_quantum = reg_quantum.score(X_quantum, y_prob)
    
    # Without quantum features (pure classical)
    X_classical = X_full[:, :1]  # Only gap
    reg_classical = LinearRegression().fit(X_classical, y_prob)
    r_classical_only = reg_classical.score(X_classical, y_prob)
    
    # Classical + phase
    X_class_phase = X_full[:, [0, 5, 6]]  # gap + cos(phase) + sin(phase)
    reg_class_phase = LinearRegression().fit(X_class_phase, y_prob)
    r_class_phase = reg_class_phase.score(X_class_phase, y_prob)
    
    print(f"  Baseline sigmoid(gap)->prob r={r_sigmoid:.4f}")
    print(f"  sigmoid(|complex_dot|)->prob r={r_sig_complex:.4f}")
    print(f"  Born rule->prob r={r_born:.4f}")
    print(f"  sigmoid(phase_corrected)->prob r={r_sig_phase:.4f}")
    print(f"  Classical only R2={r_classical_only:.4f}")
    print(f"  Quantum only (no gap) R2={r_quantum:.4f}")
    print(f"  Classical+phase R2={r_class_phase:.4f}")
    print(f"  Full model R2={r_full:.4f}")
    print(f"  Phase contribution: {r_class_phase - r_classical_only:.4f}")
    print(f"  Full quantum contribution: {r_full - r_classical_only:.4f}")
    
    # Feature importance
    feature_names = ['gap', 'complex_amp', 'complex_phase', 'h_norm', 'amp_ratio', 'cos_phase', 'sin_phase']
    print(f"  Feature coefficients (full model):")
    for name, coef in zip(feature_names, reg_full.coef_):
        print(f"    {name}: {coef:.6f}")
    
    results = {
        'experiment': 'P624',
        'model': model_name,
        'r_sigmoid': r_sigmoid,
        'r_sig_complex': r_sig_complex,
        'r_born': r_born,
        'r_sig_phase': r_sig_phase,
        'r_classical_only': r_classical_only,
        'r_quantum': r_quantum,
        'r_class_phase': r_class_phase,
        'r_full': r_full,
        'phase_contribution': r_class_phase - r_classical_only,
        'quantum_contribution': r_full - r_classical_only,
        'coefficients': {name: float(coef) for name, coef in zip(feature_names, reg_full.coef_)},
    }
    
    return results


# Experiment registry
EXPERIMENTS = {
    'p619': experiment_p619,
    'p620': experiment_p620,
    'p621': experiment_p621,
    'p622': experiment_p622,
    'p623': experiment_p623,
    'p624': experiment_p624,
}


def main():
    parser = argparse.ArgumentParser(description='Phase CXLI: Quantum Framework Rigorization')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['qwen3', 'glm4', 'deepseek7b'],
                        help='Model to test')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=list(EXPERIMENTS.keys()),
                        help='Experiment to run')
    args = parser.parse_args()
    
    # Load model
    print(f"Loading {args.model}...")
    model, tokenizer, device = load_model(args.model)
    model.eval()
    
    # Compute features
    print(f"Computing multi-layer features for {len(TEST_TEXTS)} texts...")
    all_features = compute_features_multi_layer(model, tokenizer, device, TEST_TEXTS)
    
    # Run experiment
    results = EXPERIMENTS[args.experiment](all_features, args.model, model)
    
    # Save results
    os.makedirs('results/phase_cxli', exist_ok=True)
    result_file = f'results/phase_cxli/{args.model}_{args.experiment}.json'
    with open(result_file, 'w') as f:
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(x) for x in obj]
            return obj
        json.dump(convert(results), f, indent=2)
    
    print(f"\nResults saved to {result_file}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
