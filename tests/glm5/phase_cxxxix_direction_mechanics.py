#!/usr/bin/env python3
"""
Phase CXXXIX: Direction-level mechanics vs Spectral-level mechanics (P607-P612)
Focus: Quantitative comparison of prediction power at different granularity levels

P607: Direction-level vs Spectral-level prediction power comparison
P608: c_k cross-text predictability
P609: W_U structure equation for Delta_k
P610: Semantic direction definition (top contributors)
P611: Spectral features of semantic directions
P612: Complete causal chain evaluation
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
from model_utils import load_model, get_model_info

# ============================================================
# Test texts
# ============================================================
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
    "The mathematician proved a theorem about prime number distribution.",
    "The journalist reported on the political scandal unfolding in Congress.",
    "The psychologist studied the effects of meditation on anxiety.",
    "The environmentalist campaigned against deforestation in the Amazon.",
    "The composer arranged a symphony for orchestra and choir.",
    "The entrepreneur launched a startup focused on renewable energy.",
    "The linguist documented an endangered language in remote villages.",
    "The geologist examined rock formations to understand plate tectonics.",
    "The philosopher argued that knowledge is justified true belief.",
    "The chemist synthesized a compound with unusual properties.",
    "The physicist proposed a theory unifying gravity and quantum mechanics.",
    "The sociologist researched patterns of urban migration.",
    "The veterinarian treated injured wildlife after the oil spill.",
    "The librarian curated a collection of rare manuscripts.",
    "The firefighter rescued people from the burning building.",
    "The mechanic repaired the engine using specialized tools.",
    "The photographer captured stunning images of the northern lights.",
    "The dancer performed a contemporary piece expressing grief.",
    "The filmmaker directed a documentary about ocean pollution.",
    "The botanist classified plants according to their evolutionary relationships.",
    "The astronomer detected signals from a distant galaxy cluster.",
    "The pharmacist dispensed medication with careful dosage instructions.",
    "The carpenter built furniture from reclaimed wood.",
    "The pilot navigated through turbulent weather conditions.",
    "The sculptor carved marble into a figure expressing joy.",
    "The nutritionist recommended a balanced diet rich in vegetables.",
    "The oceanographer mapped the currents of the deep sea.",
    "The archaeologist excavated artifacts from an ancient civilization.",
    "The meteorologist predicted severe thunderstorms for the weekend.",
    "The criminologist studied patterns in serial offender behavior.",
    "The theologian explored concepts of grace across religious traditions.",
    "The agronomist developed drought-resistant crop varieties.",
    "The epidemiologist tracked the spread of the infectious disease.",
    "The paralegal prepared documents for the upcoming trial.",
    "The optometrist prescribed corrective lenses for the patient.",
    "The zoologist observed mating behaviors in wild chimpanzees.",
    "The anthropologist studied rituals in a tribal community.",
    "The horticulturist cultivated exotic orchids in the greenhouse.",
    "The cryptographer developed a new encryption algorithm.",
    "The actuary calculated risk probabilities for the insurance company.",
    "The cartographer updated the map with newly discovered roads.",
    "The curator organized an exhibition of contemporary art.",
    "The demographer analyzed population trends in developing nations.",
    "The ecologist studied the impact of invasive species.",
    "The endocrinologist researched hormonal imbalances in adolescents.",
    "The forensic scientist analyzed DNA evidence from the crime scene.",
    "The geneticist mapped the genome of a rare plant species.",
    "The immunologist developed a treatment for autoimmune disorders.",
    "The lexicographer updated the dictionary with new vocabulary.",
    "The microbiologist cultured bacteria to study antibiotic resistance.",
    "The neuropathologist examined brain tissue for signs of disease.",
    "The ornithologist tracked migration patterns of Arctic terns.",
    "The paleontologist discovered fossils of a prehistoric marine reptile.",
    "The seismologist monitored earthquake activity along the fault line.",
    "The toxicologist assessed the safety of chemical additives in food.",
    "The virologist studied the mutation rate of the influenza virus.",
]

def compute_h_features(model, tokenizer, device, texts):
    """Compute h features at last layer for all texts."""
    # Get W_U
    if hasattr(model, 'lm_head'):
        W_U = model.lm_head.weight.detach().cpu().float().numpy()
    else:
        W_U = model.get_output_embeddings().weight.detach().cpu().float().numpy()
    
    all_features = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        last_hidden = outputs.hidden_states[-1][0, -1, :].cpu().float().numpy()
        logits = outputs.logits[0, -1, :].cpu().float().numpy()
        
        # SVD of h
        h = last_hidden
        d = len(h)
        h_2d = h.reshape(1, -1)
        U, S, Vt = np.linalg.svd(h_2d, full_matrices=False)
        U = U.flatten()
        
        # Get c_k (projections onto SVD directions of h)
        # We need the SVD of h itself: h = sum c_k * v_k where v_k are right singular vectors
        _, S_h, Vt_h = np.linalg.svd(h_2d, full_matrices=False)
        # For a 1xd matrix, SVD gives: h = U[0] * S[0] * Vt[0,:]
        # c_k = h . v_k = S_h[0] if k=0, 0 otherwise for 1xd
        # Actually for vector: c_k = <h, v_k> where v_k = Vt_h[k]
        c_k = Vt_h @ h  # This gives S_h * U[0,k] which is just S_h[0] for k=0
        
        # Better: do full SVD decomposition
        # h = sum_k c_k * e_k where e_k are orthonormal basis
        # Use the h's own structure
        h_norm = np.linalg.norm(h)
        c_k_norms = np.abs(h)  # Component magnitudes in standard basis
        
        # Get top token info
        top2_indices = np.argsort(logits)[-2:][::-1]
        top1_idx, top2_idx = top2_indices[0], top2_indices[1]
        prob_top1 = torch.softmax(outputs.logits[0, -1, :], dim=-1)[top1_idx].item()
        logit_gap = logits[top1_idx] - logits[top2_idx]
        logit_max = logits[top1_idx]
        
        # W_U already loaded at function start
        W_U_top1 = W_U[top1_idx]
        W_U_top2 = W_U[top2_idx]
        Delta_W = W_U_top1 - W_U_top2  # (d,)
        
        # Compute c_k in the standard basis direction
        # contrib_k = h[k] * Delta_W[k]
        contrib_k = h * Delta_W  # Per-component contribution
        logit_gap_check = np.sum(contrib_k)
        
        # Delta_k for each standard basis direction = Delta_W[k]
        # |c_k| = |h[k]|, |Delta_k| = |Delta_W[k]|
        abs_contrib_k = np.abs(h) * np.abs(Delta_W)
        
        # Ratio of top-50 components
        sorted_abs_contrib = np.sort(abs_contrib_k)[::-1]
        ratio50 = np.sum(sorted_abs_contrib[:50]) / (np.sum(abs_contrib_k) + 1e-10)
        ratio10 = np.sum(sorted_abs_contrib[:10]) / (np.sum(abs_contrib_k) + 1e-10)
        
        # Alpha (spectral exponent) - approximate from SVD of h
        # Use power law fit on |h[k]| sorted
        sorted_h = np.sort(np.abs(h))[::-1]
        n_fit = min(100, d)
        ranks = np.arange(1, n_fit + 1)
        log_ranks = np.log(ranks)
        log_vals = np.log(sorted_h[:n_fit] + 1e-10)
        if len(log_vals) > 2 and np.std(log_vals) > 0:
            slope, _, r_val, _, _ = stats.linregress(log_ranks, log_vals)
            alpha_approx = -slope
            alpha_r = r_val ** 2
        else:
            alpha_approx = 1.0
            alpha_r = 0.0
        
        # Sign statistics
        signs = np.sign(h * Delta_W)
        sign_pos_ratio = np.mean(signs > 0)
        sign_agreement = np.abs(np.mean(signs))
        
        # Format vs content directions
        # Use top-5 abs_contrib components as "format", rest as "content"
        top5_idx = np.argsort(abs_contrib_k)[-5:][::-1]
        format_gap = np.sum(contrib_k[top5_idx])
        content_gap = np.sum(contrib_k) - format_gap
        
        # |c_k| level: can we predict logit_gap from |h[k]| and |Delta_W[k]|?
        # Using cross-text mean of |h[k]| and |Delta_W[k]|
        
        # h features for prediction
        features = {
            'h_norm': h_norm,
            'ratio50': ratio50,
            'ratio10': ratio10,
            'alpha_approx': alpha_approx,
            'alpha_r': alpha_r,
            'logit_gap': logit_gap,
            'logit_max': logit_max,
            'prob_top1': prob_top1,
            'logit_gap_check': logit_gap_check,
            'sign_pos_ratio': sign_pos_ratio,
            'sign_agreement': sign_agreement,
            'format_gap': format_gap,
            'content_gap': content_gap,
            'format_fraction': format_gap / (np.abs(logit_gap) + 1e-10),
        }
        
        # Store per-component data
        features['h'] = h
        features['Delta_W'] = Delta_W
        features['contrib_k'] = contrib_k
        features['abs_contrib_k'] = abs_contrib_k
        features['abs_h'] = np.abs(h)
        features['abs_Delta_W'] = np.abs(Delta_W)
        features['signs'] = signs
        features['top1_idx'] = int(top1_idx)
        features['top2_idx'] = int(top2_idx)
        
        all_features.append(features)
    
    return all_features


def experiment_p607(all_features, model_name):
    """P607: Direction-level vs Spectral-level prediction power comparison."""
    print(f"\nP607: Direction-level vs Spectral-level -- {model_name}")
    
    n = len(all_features)
    
    # Collect scalar features
    h_norms = np.array([f['h_norm'] for f in all_features])
    ratio50s = np.array([f['ratio50'] for f in all_features])
    ratio10s = np.array([f['ratio10'] for f in all_features])
    alphas = np.array([f['alpha_approx'] for f in all_features])
    logit_gaps = np.array([f['logit_gap'] for f in all_features])
    logit_maxs = np.array([f['logit_max'] for f in all_features])
    probs = np.array([f['prob_top1'] for f in all_features])
    
    # Spectral-level prediction
    # ratio50 -> logit_gap
    r_ratio50_gap, _ = stats.pearsonr(ratio50s, logit_gaps)
    # ratio50 -> logit_max
    r_ratio50_max, _ = stats.pearsonr(ratio50s, logit_maxs)
    # h_norm -> logit_max
    r_hnorm_max, _ = stats.pearsonr(h_norms, logit_maxs)
    # Multi spectral -> logit_gap
    X_spec = np.column_stack([h_norms, ratio50s, ratio10s, alphas])
    from numpy.linalg import lstsq
    coeff, _, _, _ = lstsq(X_spec, logit_gaps, rcond=None)
    pred_spec = X_spec @ coeff
    r_multi_spec_gap, _ = stats.pearsonr(pred_spec, logit_gaps)
    # Multi spectral -> prob
    coeff2, _, _, _ = lstsq(X_spec, probs, rcond=None)
    pred_spec2 = X_spec @ coeff2
    r_multi_spec_prob, _ = stats.pearsonr(pred_spec2, probs)
    
    # Direction-level prediction
    # Compute mean |h[k]| and mean |Delta_W[k]| across texts
    d = len(all_features[0]['h'])
    mean_abs_h = np.zeros(d)
    mean_abs_Delta = np.zeros(d)
    for f in all_features:
        mean_abs_h += f['abs_h']
        mean_abs_Delta += f['abs_Delta_W']
    mean_abs_h /= n
    mean_abs_Delta /= n
    
    # For each text, predict logit_gap using mean |h[k]| * actual |Delta_W[k]|
    dir_preds = []
    for f in all_features:
        # Use mean |h[k]| * actual |Delta_W[k]| with sign from actual h
        pred = np.sum(mean_abs_h * f['abs_Delta_W'] * f['signs'])
        dir_preds.append(pred)
    dir_preds = np.array(dir_preds)
    r_dir_gap, _ = stats.pearsonr(dir_preds, logit_gaps)
    
    # Use mean |h[k]| * mean |Delta_W[k]| (full cross-text mean)
    dir_preds2 = []
    for f in all_features:
        pred = np.sum(mean_abs_h * mean_abs_Delta * np.sign(np.mean([ff['signs'] for ff in all_features], axis=0)))
        dir_preds2.append(pred)
    # This is a constant, so correlation is 0
    r_dir2_gap = 0.0
    
    # Use actual |h[k]| * actual |Delta_W[k]| without sign (magnitude only)
    dir_preds3 = []
    for f in all_features:
        pred = np.sum(f['abs_h'] * f['abs_Delta_W'])
        dir_preds3.append(pred)
    dir_preds3 = np.array(dir_preds3)
    r_dir3_gap, _ = stats.pearsonr(dir_preds3, logit_gaps)
    r_dir3_prob, _ = stats.pearsonr(dir_preds3, probs)
    
    # Direction-level multi regression
    # Use per-component |h[k]| as features (too many, use top-50)
    top50_indices = np.argsort(mean_abs_h * mean_abs_Delta)[-50:][::-1]
    X_dir = np.column_stack([np.array([f['abs_h'][top50_indices] for f in all_features])])
    # Also add sign features for top-50
    X_dir_sign = np.column_stack([
        np.array([f['abs_h'][top50_indices] for f in all_features]),
        np.array([f['signs'][top50_indices] for f in all_features]),
    ])
    coeff_dir, _, _, _ = lstsq(X_dir, logit_gaps, rcond=None)
    pred_dir = X_dir @ coeff_dir
    r_dir_multi_gap, _ = stats.pearsonr(pred_dir, logit_gaps)
    coeff_dir2, _, _, _ = lstsq(X_dir_sign, logit_gaps, rcond=None)
    pred_dir2 = X_dir_sign @ coeff_dir2
    r_dir_multi2_gap, _ = stats.pearsonr(pred_dir2, logit_gaps)
    # Direction-level -> prob
    coeff_dir3, _, _, _ = lstsq(X_dir_sign, probs, rcond=None)
    pred_dir3 = X_dir_sign @ coeff_dir3
    r_dir_multi2_prob, _ = stats.pearsonr(pred_dir3, probs)
    
    results = {
        'spectral_ratio50_gap': float(r_ratio50_gap),
        'spectral_ratio50_max': float(r_ratio50_max),
        'spectral_hnorm_max': float(r_hnorm_max),
        'spectral_multi_gap': float(r_multi_spec_gap),
        'spectral_multi_prob': float(r_multi_spec_prob),
        'direction_mean_h_actual_DeltaW_gap': float(r_dir_gap),
        'direction_abs_h_abs_DeltaW_gap': float(r_dir3_gap),
        'direction_abs_h_abs_DeltaW_prob': float(r_dir3_prob),
        'direction_top50_nosign_gap': float(r_dir_multi_gap),
        'direction_top50_withsign_gap': float(r_dir_multi2_gap),
        'direction_top50_withsign_prob': float(r_dir_multi2_prob),
    }
    
    print(f"  Spectral-level:")
    print(f"    ratio50->gap r={r_ratio50_gap:.4f}")
    print(f"    ratio50->logit_max r={r_ratio50_max:.4f}")
    print(f"    h_norm->logit_max r={r_hnorm_max:.4f}")
    print(f"    multi spec->gap r={r_multi_spec_gap:.4f}")
    print(f"    multi spec->prob r={r_multi_spec_prob:.4f}")
    print(f"  Direction-level:")
    print(f"    mean|h|*actual|DeltaW|->gap r={r_dir_gap:.4f}")
    print(f"    |h|*|DeltaW|->gap r={r_dir3_gap:.4f}")
    print(f"    |h|*|DeltaW|->prob r={r_dir3_prob:.4f}")
    print(f"    top50 nosign->gap r={r_dir_multi_gap:.4f}")
    print(f"    top50 withsign->gap r={r_dir_multi2_gap:.4f}")
    print(f"    top50 withsign->prob r={r_dir_multi2_prob:.4f}")
    
    return results


def experiment_p608(all_features, model_name):
    """P608: c_k cross-text predictability."""
    print(f"\nP608: c_k cross-text predictability -- {model_name}")
    
    d = len(all_features[0]['h'])
    n = len(all_features)
    
    # Compute CV and sign agreement for each component
    h_matrix = np.array([f['h'] for f in all_features])  # (n, d)
    
    cv_h = np.std(h_matrix, axis=0) / (np.mean(np.abs(h_matrix), axis=0) + 1e-10)
    sign_h = np.mean(np.sign(h_matrix), axis=0)
    
    # Mean CV and sign agreement
    mean_cv = np.mean(cv_h)
    median_cv = np.median(cv_h)
    mean_sign_agreement = np.mean(np.abs(sign_h))
    sign_pos_ratio = np.mean(sign_h > 0)
    
    # Can we predict |h[k]| from spectral features?
    # For each component k, regress |h[k]| on h_norm, ratio50
    abs_h_matrix = np.abs(h_matrix)
    h_norms = np.array([f['h_norm'] for f in all_features])
    ratio50s = np.array([f['ratio50'] for f in all_features])
    
    # Top-20 components by mean |h[k]|
    top20_idx = np.argsort(np.mean(abs_h_matrix, axis=0))[-20:][::-1]
    
    pred_r_nosign = []
    pred_r_withsign = []
    for k in top20_idx:
        y = h_matrix[:, k]
        # From h_norm alone
        r1, _ = stats.pearsonr(h_norms, np.abs(y))
        pred_r_nosign.append(r1)
        # From h_norm + ratio50
        X = np.column_stack([h_norms, ratio50s])
        from numpy.linalg import lstsq
        coeff, _, _, _ = lstsq(X, y, rcond=None)
        pred_y = X @ coeff
        r2, _ = stats.pearsonr(pred_y, y)
        pred_r_withsign.append(r2)
    
    # Overall: can we predict h from spectral features?
    # Use PCA: project h to top-10 PCs, predict from spectral features
    from numpy.linalg import lstsq
    h_centered = h_matrix - np.mean(h_matrix, axis=0)
    U_pca, S_pca, Vt_pca = np.linalg.svd(h_centered, full_matrices=False)
    top10_pcs = U_pca[:, :10] * S_pca[:10]  # (n, 10)
    
    X_spec = np.column_stack([h_norms, ratio50s, np.array([f['alpha_approx'] for f in all_features])])
    coeff_pca, _, _, _ = lstsq(X_spec, top10_pcs, rcond=None)
    pred_pcs = X_spec @ coeff_pca
    
    pca_pred_r = []
    for pc in range(10):
        r, _ = stats.pearsonr(pred_pcs[:, pc], top10_pcs[:, pc])
        pca_pred_r.append(r)
    
    results = {
        'mean_cv': float(mean_cv),
        'median_cv': float(median_cv),
        'mean_sign_agreement': float(mean_sign_agreement),
        'sign_pos_ratio': float(sign_pos_ratio),
        'mean_pred_r_nosign': float(np.mean(pred_r_nosign)),
        'mean_pred_r_withsign': float(np.mean(pred_r_withsign)),
        'pca_pred_r_mean': float(np.mean(pca_pred_r)),
        'pca_pred_r_max': float(np.max(pca_pred_r)),
    }
    
    print(f"  h component CV: mean={mean_cv:.4f}, median={median_cv:.4f}")
    print(f"  Sign agreement: mean={mean_sign_agreement:.4f}, pos_ratio={sign_pos_ratio:.4f}")
    print(f"  |h[k]| prediction from h_norm: r={np.mean(pred_r_nosign):.4f}")
    print(f"  h[k] prediction from h_norm+ratio50: r={np.mean(pred_r_withsign):.4f}")
    print(f"  PCA prediction from spectral: mean_r={np.mean(pca_pred_r):.4f}, max_r={np.max(pca_pred_r):.4f}")
    
    return results


def experiment_p609(all_features, model_name, model):
    """P609: W_U structure equation for Delta_k."""
    print(f"\nP609: W_U structure equation for Delta_k -- {model_name}")
    
    if hasattr(model, 'lm_head'):
        W_U = model.lm_head.weight.detach().cpu().float().numpy()
    else:
        W_U = model.get_output_embeddings().weight.detach().cpu().float().numpy()
    d = W_U.shape[1]
    n = len(all_features)
    
    # For each text, compute Delta_W = W_U[top1] - W_U[top2]
    Delta_Ws = []
    top_pairs = []
    for f in all_features:
        Delta_W = f['Delta_W']
        Delta_Ws.append(Delta_W)
        top_pairs.append((f['top1_idx'], f['top2_idx']))
    
    Delta_Ws = np.array(Delta_Ws)
    
    # SVD of W_U - skip for large vocab (memory issue)
    # U_wu, S_wu, Vt_wu = np.linalg.svd(W_U, full_matrices=False)
    
    # Can we predict |Delta_W[k]| from W_U structure?
    # |Delta_W[k]| = |W_U[top1,k] - W_U[top2,k]|
    # For random top1/top2, E[|Delta_W[k]|] ≈ sqrt(2) * std(W_U[:,k])
    col_stds = np.std(W_U, axis=0)
    mean_abs_Delta = np.mean(np.abs(Delta_Ws), axis=0)
    
    r_std_pred, _ = stats.pearsonr(col_stds, mean_abs_Delta)
    
    # Per-text: predict |Delta_W[k]| from S_wu and Vt_wu
    # Delta_W = W_U[top1] - W_U[top2]
    # In SVD basis: Delta_W = (U_wu[top1] - U_wu[top2]) @ diag(S_wu) @ Vt_wu
    # |Delta_W[k]| ≈ sum_j |U_wu[top1,j] - U_wu[top2,j]| * S_wu[j] * |Vt_wu[j,k]|
    
    # Simpler: predict mean |Delta_W| from W_U column norms
    col_norms = np.linalg.norm(W_U, axis=0)
    r_colnorm_pred, _ = stats.pearsonr(col_norms, mean_abs_Delta)
    
    # Cross-text stability of Delta_W
    cv_Delta = np.std(Delta_Ws, axis=0) / (np.mean(np.abs(Delta_Ws), axis=0) + 1e-10)
    mean_cv_Delta = np.mean(cv_Delta)
    
    # Sign stability of Delta_W
    sign_Delta = np.mean(np.sign(Delta_Ws), axis=0)
    mean_sign_Delta = np.mean(np.abs(sign_Delta))
    
    # Predict logit_gap from Delta_W structure
    logit_gaps = np.array([f['logit_gap'] for f in all_features])
    probs = np.array([f['prob_top1'] for f in all_features])
    
    # Use mean Delta_W * actual h
    mean_Delta = np.mean(Delta_Ws, axis=0)
    pred_gaps = [np.dot(mean_Delta, f['h']) for f in all_features]
    r_mean_Delta_gap, _ = stats.pearsonr(pred_gaps, logit_gaps)
    
    # Predict |Delta_W| from W_U SVD structure
    # |Delta_W[k]| should correlate with W_U column activity
    abs_Delta_Ws = np.abs(Delta_Ws)
    mean_abs_Delta_per_comp = np.mean(abs_Delta_Ws, axis=0)
    
    results = {
        'W_U_col_std_predict_Delta_abs_r': float(r_std_pred),
        'W_U_col_norm_predict_Delta_abs_r': float(r_colnorm_pred),
        'Delta_W_cv_mean': float(mean_cv_Delta),
        'Delta_W_sign_agreement': float(mean_sign_Delta),
        'mean_Delta_dot_h_gap_r': float(r_mean_Delta_gap),
    }
    
    print(f"  W_U col_std -> mean|Delta_W| r={r_std_pred:.4f}")
    print(f"  W_U col_norm -> mean|Delta_W| r={r_colnorm_pred:.4f}")
    print(f"  Delta_W CV: {mean_cv_Delta:.4f}")
    print(f"  Delta_W sign agreement: {mean_sign_Delta:.4f}")
    print(f"  mean(Delta_W)·h -> gap r={r_mean_Delta_gap:.4f}")
    
    return results


def experiment_p610(all_features, model_name):
    """P610: Semantic direction definition (top contributors)."""
    print(f"\nP610: Semantic direction definition -- {model_name}")
    
    d = len(all_features[0]['h'])
    n = len(all_features)
    
    # For each text, rank components by |contrib_k|
    # Find "semantic directions" = components that consistently contribute most
    abs_contrib_matrix = np.array([f['abs_contrib_k'] for f in all_features])
    
    # Mean |contrib_k| across texts
    mean_abs_contrib = np.mean(abs_contrib_matrix, axis=0)
    top10_semantic_idx = np.argsort(mean_abs_contrib)[-10:][::-1]
    top20_semantic_idx = np.argsort(mean_abs_contrib)[-20:][::-1]
    
    # Consistency: how often is each component in top-10?
    consistency = np.zeros(d)
    for f in all_features:
        top10 = np.argsort(f['abs_contrib_k'])[-10:]
        consistency[top10] += 1
    consistency /= n
    
    # Top semantic directions consistency
    top10_consistency = consistency[top10_semantic_idx]
    mean_top10_consistency = np.mean(top10_consistency)
    
    # How much of total |contrib| do top-10 cover?
    mean_total_contrib = np.sum(mean_abs_contrib)
    top10_coverage = np.sum(mean_abs_contrib[top10_semantic_idx]) / mean_total_contrib
    top20_coverage = np.sum(mean_abs_contrib[top20_semantic_idx]) / mean_total_contrib
    
    # Are semantic directions format or content?
    # Check: are top semantic directions the same as format directions?
    logit_gaps = np.array([f['logit_gap'] for f in all_features])
    
    # Contribution of top-10 semantic directions vs rest
    semantic_gaps = []
    rest_gaps = []
    for f in all_features:
        sem_gap = np.sum(f['contrib_k'][top10_semantic_idx])
        rest_gap = np.sum(f['contrib_k']) - sem_gap
        semantic_gaps.append(sem_gap)
        rest_gaps.append(rest_gap)
    
    r_sem_gap_prob, _ = stats.pearsonr(semantic_gaps, np.array([f['prob_top1'] for f in all_features]))
    r_rest_gap_prob, _ = stats.pearsonr(rest_gaps, np.array([f['prob_top1'] for f in all_features]))
    r_sem_gap_gap, _ = stats.pearsonr(semantic_gaps, logit_gaps)
    
    results = {
        'top10_semantic_idx': [int(x) for x in top10_semantic_idx],
        'top10_consistency_mean': float(mean_top10_consistency),
        'top10_coverage': float(top10_coverage),
        'top20_coverage': float(top20_coverage),
        'semantic_gap_prob_r': float(r_sem_gap_prob),
        'rest_gap_prob_r': float(r_rest_gap_prob),
        'semantic_gap_gap_r': float(r_sem_gap_gap),
    }
    
    print(f"  Top-10 semantic indices: {top10_semantic_idx}")
    print(f"  Top-10 consistency: {mean_top10_consistency:.4f}")
    print(f"  Top-10 coverage: {top10_coverage:.4f}")
    print(f"  Top-20 coverage: {top20_coverage:.4f}")
    print(f"  Semantic gap->prob r={r_sem_gap_prob:.4f}")
    print(f"  Rest gap->prob r={r_rest_gap_prob:.4f}")
    print(f"  Semantic gap->gap r={r_sem_gap_gap:.4f}")
    
    return results


def experiment_p611(all_features, model_name):
    """P611: Spectral features of semantic directions."""
    print(f"\nP611: Spectral features of semantic directions -- {model_name}")
    
    d = len(all_features[0]['h'])
    n = len(all_features)
    
    # Get semantic directions from P610
    abs_contrib_matrix = np.array([f['abs_contrib_k'] for f in all_features])
    mean_abs_contrib = np.mean(abs_contrib_matrix, axis=0)
    top10_idx = np.argsort(mean_abs_contrib)[-10:][::-1]
    
    # Spectral features: ratio50, alpha, h_norm
    h_norms = np.array([f['h_norm'] for f in all_features])
    ratio50s = np.array([f['ratio50'] for f in all_features])
    probs = np.array([f['prob_top1'] for f in all_features])
    
    # For semantic directions: can we predict |h[k]| for k in top10 from spectral features?
    h_matrix = np.array([f['h'] for f in all_features])
    
    pred_r_semantic = []
    for k in top10_idx:
        y = np.abs(h_matrix[:, k])
        r, _ = stats.pearsonr(h_norms, y)
        pred_r_semantic.append(r)
    
    # For non-semantic directions
    non_semantic_idx = [i for i in range(d) if i not in top10_idx]
    # Sample 10 random non-semantic
    if len(non_semantic_idx) > 10:
        np.random.seed(42)
        sample_idx = np.random.choice(non_semantic_idx, 10, replace=False)
    else:
        sample_idx = non_semantic_idx[:10]
    
    pred_r_nonsemantic = []
    for k in sample_idx:
        y = np.abs(h_matrix[:, k])
        r, _ = stats.pearsonr(h_norms, y)
        pred_r_nonsemantic.append(r)
    
    # Can ratio50 predict semantic |h[k]|?
    pred_r_semantic_ratio50 = []
    for k in top10_idx:
        y = np.abs(h_matrix[:, k])
        r, _ = stats.pearsonr(ratio50s, y)
        pred_r_semantic_ratio50.append(r)
    
    # Are semantic directions' h values more predictable?
    # Compare variance of |h[k]| for semantic vs non-semantic
    cv_semantic = np.std(np.abs(h_matrix[:, top10_idx]), axis=0) / (np.mean(np.abs(h_matrix[:, top10_idx]), axis=0) + 1e-10)
    cv_nonsemantic = np.std(np.abs(h_matrix[:, sample_idx]), axis=0) / (np.mean(np.abs(h_matrix[:, sample_idx]), axis=0) + 1e-10)
    
    results = {
        'semantic_h_predict_from_hnorm_r_mean': float(np.mean(pred_r_semantic)),
        'nonsemantic_h_predict_from_hnorm_r_mean': float(np.mean(pred_r_nonsemantic)),
        'semantic_h_predict_from_ratio50_r_mean': float(np.mean(pred_r_semantic_ratio50)),
        'semantic_cv_mean': float(np.mean(cv_semantic)),
        'nonsemantic_cv_mean': float(np.mean(cv_nonsemantic)),
    }
    
    print(f"  Semantic |h[k]| <- h_norm: r={np.mean(pred_r_semantic):.4f}")
    print(f"  Non-semantic |h[k]| <- h_norm: r={np.mean(pred_r_nonsemantic):.4f}")
    print(f"  Semantic |h[k]| <- ratio50: r={np.mean(pred_r_semantic_ratio50):.4f}")
    print(f"  Semantic CV: {np.mean(cv_semantic):.4f}")
    print(f"  Non-semantic CV: {np.mean(cv_nonsemantic):.4f}")
    
    return results


def experiment_p612(all_features, model_name, model=None):
    """P612: Complete causal chain evaluation."""
    print(f"\nP612: Complete causal chain evaluation -- {model_name}")
    
    n = len(all_features)
    d = len(all_features[0]['h'])
    
    logit_gaps = np.array([f['logit_gap'] for f in all_features])
    probs = np.array([f['prob_top1'] for f in all_features])
    h_norms = np.array([f['h_norm'] for f in all_features])
    ratio50s = np.array([f['ratio50'] for f in all_features])
    
    # Path 1: spectral -> gap -> prob (classical)
    r_ratio50_gap, _ = stats.pearsonr(ratio50s, logit_gaps)
    r_gap_prob, _ = stats.pearsonr(logit_gaps, probs)
    
    # Path 2: spectral -> h features -> gap -> prob
    alphas = np.array([f['alpha_approx'] for f in all_features])
    X_h = np.column_stack([h_norms, ratio50s, alphas])
    from numpy.linalg import lstsq
    coeff, _, _, _ = lstsq(X_h, logit_gaps, rcond=None)
    pred_gap = X_h @ coeff
    r_h_gap, _ = stats.pearsonr(pred_gap, logit_gaps)
    pred_prob_from_gap = 1.0 / (1.0 + np.exp(-pred_gap))
    r_h_gap_prob, _ = stats.pearsonr(pred_prob_from_gap, probs)
    
    # Path 3: direction-level -> gap -> prob
    abs_contrib_matrix = np.array([f['abs_contrib_k'] for f in all_features])
    mean_abs_contrib = np.mean(abs_contrib_matrix, axis=0)
    top10_idx = np.argsort(mean_abs_contrib)[-10:][::-1]
    
    h_matrix = np.array([f['h'] for f in all_features])
    sign_matrix = np.array([f['signs'] for f in all_features])
    
    # Use top-10 direction values (with sign) as features
    X_dir = np.column_stack([
        h_matrix[:, top10_idx],
        sign_matrix[:, top10_idx],
    ])
    coeff_dir, _, _, _ = lstsq(X_dir, logit_gaps, rcond=None)
    pred_dir_gap = X_dir @ coeff_dir
    r_dir_gap, _ = stats.pearsonr(pred_dir_gap, logit_gaps)
    pred_dir_prob = 1.0 / (1.0 + np.exp(-pred_dir_gap))
    r_dir_prob, _ = stats.pearsonr(pred_dir_prob, probs)
    
    # Path 4: direction-level -> prob (direct)
    coeff_dir2, _, _, _ = lstsq(X_dir, probs, rcond=None)
    pred_dir2_prob = X_dir @ coeff_dir2
    r_dir2_prob, _ = stats.pearsonr(pred_dir2_prob, probs)
    
    # Path 5: Oracle (actual gap -> prob)
    oracle_prob = 1.0 / (1.0 + np.exp(-logit_gaps))
    r_oracle, _ = stats.pearsonr(oracle_prob, probs)
    
    # Path 6: Full h -> gap (all components, oracle level)
    X_full = h_matrix
    coeff_full, _, _, _ = lstsq(X_full, logit_gaps, rcond=None)
    pred_full_gap = X_full @ coeff_full
    r_full_gap, _ = stats.pearsonr(pred_full_gap, logit_gaps)
    pred_full_prob = 1.0 / (1.0 + np.exp(-pred_full_gap))
    r_full_prob, _ = stats.pearsonr(pred_full_prob, probs)
    
    # Path ranking
    paths = [
        ('P1_spectral_ratio50', r_ratio50_gap, 'No oracle'),
        ('P2_spectral_multi_h', r_h_gap, 'No oracle'),
        ('P3_direction_top10', r_dir_gap, 'No oracle'),
        ('P4_direction_direct', r_dir2_prob, 'No oracle'),
        ('P5_full_h', r_full_gap, 'Semi-oracle'),
        ('P6_oracle_gap', r_oracle, 'Oracle'),
    ]
    paths.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"  === Path Ranking ===")
    for i, (name, r, level) in enumerate(paths):
        print(f"  #{i+1}: {name} r={abs(r):.4f} ({level})")
    
    # Key comparison: spectral vs direction vs oracle
    results = {
        'spectral_ratio50_gap_r': float(r_ratio50_gap),
        'spectral_multi_gap_r': float(r_h_gap),
        'direction_top10_gap_r': float(r_dir_gap),
        'direction_top10_prob_r': float(r_dir2_prob),
        'full_h_gap_r': float(r_full_gap),
        'oracle_gap_prob_r': float(r_oracle),
        'gap_prob_r': float(r_gap_prob),
        'spectral_vs_direction_ratio': float(abs(r_dir_gap) / (abs(r_h_gap) + 1e-10)),
        'direction_vs_oracle_ratio': float(abs(r_dir_gap) / (abs(r_oracle) + 1e-10)),
    }
    
    return results


# ============================================================
# Main
# ============================================================
EXPERIMENTS = {
    'p607': experiment_p607,
    'p608': experiment_p608,
    'p609': experiment_p609,
    'p610': experiment_p610,
    'p611': experiment_p611,
    'p612': experiment_p612,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'glm4', 'deepseek7b'])
    parser.add_argument('--experiment', type=str, required=True, choices=list(EXPERIMENTS.keys()))
    args = parser.parse_args()
    
    t0 = time.time()
    
    # Load model
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    
    # Compute features
    print(f"Computing h features for {len(TEST_TEXTS)} texts...")
    all_features = compute_h_features(model, tokenizer, device, TEST_TEXTS)
    
    # Run experiment
    if args.experiment in ['p609']:
        results = EXPERIMENTS[args.experiment](all_features, args.model, model)
    elif args.experiment in ['p612']:
        results = EXPERIMENTS[args.experiment](all_features, args.model)
    else:
        results = EXPERIMENTS[args.experiment](all_features, args.model)
    
    # Save results
    os.makedirs(f'results/phase_cxxxix', exist_ok=True)
    out_path = f'results/phase_cxxxix/{args.model}_{args.experiment}.json'
    
    # Convert non-serializable types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    
    results_serializable = {k: convert(v) for k, v in results.items()}
    with open(out_path, 'w') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\n  Results saved to {out_path}")
    print(f"  Total time: {time.time()-t0:.1f}s")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
