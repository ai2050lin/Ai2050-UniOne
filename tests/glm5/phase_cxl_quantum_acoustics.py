#!/usr/bin/env python3
"""
Phase CXL: Quantum Acoustics Framework Verification (P613-P618)
Focus: Testing the quantum acoustics hypothesis

P613: Sign correlation matrix - structured or random?
P614: Pseudo-phase construction from layer pairs
P615: Sign dynamics across layers
P616: Coherence structure of semantic directions
P617: Quantum projection prob = |Σ c_k · <top1|U_k>|²
P618: Cross-model quantum invariants
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


def get_hidden_states_all_layers(model, tokenizer, device, text, max_layers=None):
    """Get hidden states from all layers."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    all_hidden = []
    for hs in outputs.hidden_states:
        all_hidden.append(hs[0, -1, :].cpu().float().numpy())
    
    logits = outputs.logits[0, -1, :].cpu().float().numpy()
    return all_hidden, logits


def experiment_p613(all_features, model_name):
    """P613: Sign correlation matrix - structured or random?"""
    print(f"\nP613: Sign correlation matrix -- {model_name}")
    
    n = len(all_features)
    
    # Build sign matrix: sign_matrix[i, k] = sign(h[i,k] * Delta_W[i,k])
    d = len(all_features[0]['h'])
    sign_matrix = np.array([f['signs'] for f in all_features])  # (n, d)
    
    # Compute sign correlation matrix: C_{kj} = corr(sign_k, sign_j)
    # Only compute for top-50 most important components (by mean |contrib_k|)
    abs_contrib_matrix = np.array([f['abs_contrib_k'] for f in all_features])
    mean_abs_contrib = np.mean(abs_contrib_matrix, axis=0)
    top50_idx = np.argsort(mean_abs_contrib)[-50:][::-1]
    
    sign_top50 = sign_matrix[:, top50_idx]  # (n, 50)
    
    # Correlation matrix of signs
    # Replace 0 signs with small positive (to avoid zero std)
    sign_top50_clean = sign_top50.copy().astype(float)
    for k in range(50):
        if np.std(sign_top50_clean[:, k]) < 1e-10:
            sign_top50_clean[:, k] = np.random.randn(n) * 0.01
    
    C_sign = np.corrcoef(sign_top50_clean.T)  # (50, 50)
    
    # Statistics of C_sign
    # Diagonal is 1, so exclude
    off_diag = C_sign[np.triu_indices(50, k=1)]
    mean_off_diag = np.mean(off_diag)
    std_off_diag = np.std(off_diag)
    max_off_diag = np.max(np.abs(off_diag))
    
    # How many off-diagonal entries have |r| > 0.3?
    n_significant = np.sum(np.abs(off_diag) > 0.3)
    frac_significant = n_significant / len(off_diag)
    
    # Expected under null (random signs): each sign is independent
    # E[corr(sign_k, sign_j)] = 0 for k≠j (by CLT for large n)
    # Under null, |corr| ~ Beta(0.5, (n-3)/2 - 0.5) approximately
    # For n=80, E[|corr|] ≈ 0.11, P(|corr|>0.3) ≈ 0.04
    expected_frac = 0.04
    
    # Eigenvalue structure of C_sign
    eigenvalues = np.linalg.eigvalsh(C_sign)
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Ratio of top eigenvalue to sum (measures concentration)
    top_eigenvalue_ratio = eigenvalues[0] / np.sum(eigenvalues)
    
    # Compare with random sign matrix
    np.random.seed(42)
    random_signs = np.sign(np.random.randn(n, 50))
    C_random = np.corrcoef(random_signs.T)
    off_diag_random = C_random[np.triu_indices(50, k=1)]
    
    results = {
        'sign_corr_mean': float(mean_off_diag),
        'sign_corr_std': float(std_off_diag),
        'sign_corr_max_abs': float(max_off_diag),
        'frac_significant_r03': float(frac_significant),
        'expected_frac_random': float(expected_frac),
        'top_eigenvalue_ratio': float(top_eigenvalue_ratio),
        'eigenvalues_top5': [float(x) for x in eigenvalues[:5]],
        'random_corr_mean': float(np.mean(off_diag_random)),
        'random_corr_max_abs': float(np.max(np.abs(off_diag_random))),
        'random_frac_significant': float(np.sum(np.abs(off_diag_random) > 0.3) / len(off_diag_random)),
    }
    
    print(f"  Sign correlation matrix (top-50 components):")
    print(f"    Mean off-diag r={mean_off_diag:.4f} (random: {np.mean(off_diag_random):.4f})")
    print(f"    Std off-diag r={std_off_diag:.4f}")
    print(f"    Max |r|={max_off_diag:.4f} (random: {np.max(np.abs(off_diag_random)):.4f})")
    print(f"    Frac |r|>0.3={frac_significant:.4f} (random: {np.sum(np.abs(off_diag_random) > 0.3) / len(off_diag_random):.4f})")
    print(f"    Top eigenvalue ratio={top_eigenvalue_ratio:.4f}")
    print(f"    Top-5 eigenvalues: {eigenvalues[:5]}")
    
    # Verdict: is C_sign structured?
    is_structured = frac_significant > 2 * expected_frac or top_eigenvalue_ratio > 0.1
    print(f"    ** Verdict: {'STRUCTURED' if is_structured else 'RANDOM'} **")
    results['is_structured'] = bool(is_structured)
    
    return results


def experiment_p614(model, tokenizer, device, model_name, texts):
    """P614: Pseudo-phase construction from layer pairs."""
    print(f"\nP614: Pseudo-phase construction -- {model_name}")
    
    n = len(texts)
    
    # Get W_U
    if hasattr(model, 'lm_head'):
        W_U = model.lm_head.weight.detach().cpu().float().numpy()
    else:
        W_U = model.get_output_embeddings().weight.detach().cpu().float().numpy()
    
    all_results = []
    
    for i, text in enumerate(texts):
        all_hidden, logits = get_hidden_states_all_layers(model, tokenizer, device, text)
        
        n_layers = len(all_hidden)
        h_last = all_hidden[-1]
        
        # Top-2 tokens
        top2 = np.argsort(logits)[-2:][::-1]
        top1, top2_idx = top2[0], top2[1]
        prob_top1 = torch.softmax(torch.tensor(logits), dim=-1)[top1].item()
        logit_gap = logits[top1] - logits[top2_idx]
        
        # Delta_W
        Delta_W = W_U[top1] - W_U[top2_idx]
        
        # Standard sign: sign(h_last * Delta_W)
        signs_last = np.sign(h_last * Delta_W)
        sign_agreement_last = np.abs(np.mean(signs_last))
        
        # Pseudo-phase: use consecutive layer pair
        # h(l) = h_real, h(l-1) = h_imag
        # c_k_complex = h_last[k] + i * h_prev[k]
        # phase_k = angle(c_k_complex * Delta_W[k])
        
        best_layer = None
        best_r = 0
        
        for l in range(1, min(n_layers, 10)):  # Try layers 1-9
            h_real = all_hidden[-1]
            h_imag = all_hidden[-1 - l]
            
            # Complex contribution
            contrib_real = h_real * Delta_W
            contrib_imag = h_imag * Delta_W
            contrib_complex = contrib_real + 1j * contrib_imag
            
            # Phase-predicted sign: sign = sign(cos(phase))
            phases = np.angle(contrib_complex)
            predicted_signs = np.sign(np.cos(phases))
            
            # How well does phase-predicted sign match actual sign?
            match_rate = np.mean(predicted_signs == signs_last)
            
            # Use phase-weighted contribution
            phase_weights = np.cos(phases)  # Real part of e^{i*phase}
            phase_weighted_gap = np.sum(np.abs(h_real * Delta_W) * phase_weights)
            
            all_results.append({
                'layer_gap': l,
                'match_rate': float(match_rate),
                'sign_agreement_last': float(sign_agreement_last),
            })
            
            if match_rate > best_r:
                best_r = match_rate
                best_layer = l
        
        if i % 20 == 0:
            print(f"  Processed {i+1}/{n} texts...")
    
    # Aggregate results by layer gap
    layer_gaps = sorted(set(r['layer_gap'] for r in all_results))
    agg_results = {}
    
    for lg in layer_gaps:
        layer_data = [r for r in all_results if r['layer_gap'] == lg]
        match_rates = [r['match_rate'] for r in layer_data]
        sign_agreements = [r['sign_agreement_last'] for r in layer_data]
        agg_results[lg] = {
            'mean_match_rate': float(np.mean(match_rates)),
            'mean_sign_agreement': float(np.mean(sign_agreements)),
            'improvement': float(np.mean(match_rates) - np.mean(sign_agreements)),
        }
    
    # Best pseudo-phase result
    best_lg = max(agg_results.keys(), key=lambda lg: agg_results[lg]['mean_match_rate'])
    
    results = {
        'best_layer_gap': int(best_lg),
        'best_match_rate': float(agg_results[best_lg]['mean_match_rate']),
        'baseline_sign_agreement': float(agg_results[best_lg]['mean_sign_agreement']),
        'improvement': float(agg_results[best_lg]['improvement']),
        'all_layer_gaps': {str(k): v for k, v in agg_results.items()},
    }
    
    print(f"  Baseline sign agreement: {agg_results[best_lg]['mean_sign_agreement']:.4f}")
    print(f"  Best pseudo-phase (gap={best_lg}): match_rate={agg_results[best_lg]['mean_match_rate']:.4f}")
    print(f"  Improvement: {agg_results[best_lg]['improvement']:.4f}")
    
    return results


def experiment_p615(model, tokenizer, device, model_name, texts):
    """P615: Sign dynamics across layers."""
    print(f"\nP615: Sign dynamics across layers -- {model_name}")
    
    n_texts = min(len(texts), 30)  # Limit for speed
    
    # Get W_U
    if hasattr(model, 'lm_head'):
        W_U = model.lm_head.weight.detach().cpu().float().numpy()
    else:
        W_U = model.get_output_embeddings().weight.detach().cpu().float().numpy()
    
    d = W_U.shape[1]
    
    # Track sign evolution across layers
    all_sign_profiles = []
    
    for i, text in enumerate(texts[:n_texts]):
        all_hidden, logits = get_hidden_states_all_layers(model, tokenizer, device, text)
        
        top2 = np.argsort(logits)[-2:][::-1]
        top1, top2_idx = top2[0], top2[1]
        Delta_W = W_U[top1] - W_U[top2_idx]
        
        # Compute sign(c_k * Delta_k) at each layer
        n_layers = len(all_hidden)
        sign_profile = []
        for l in range(n_layers):
            h_l = all_hidden[l]
            signs_l = np.sign(h_l * Delta_W)
            sign_agreement_l = np.abs(np.mean(signs_l))
            sign_pos_ratio_l = np.mean(signs_l > 0)
            sign_profile.append({
                'agreement': float(sign_agreement_l),
                'pos_ratio': float(sign_pos_ratio_l),
            })
        
        all_sign_profiles.append(sign_profile)
        
        if (i+1) % 10 == 0:
            print(f"  Processed {i+1}/{n_texts} texts...")
    
    # Average sign profile across texts
    n_layers = len(all_sign_profiles[0])
    avg_agreement = []
    avg_pos_ratio = []
    
    for l in range(n_layers):
        agreements = [sp[l]['agreement'] for sp in all_sign_profiles]
        pos_ratios = [sp[l]['pos_ratio'] for sp in all_sign_profiles]
        avg_agreement.append(np.mean(agreements))
        avg_pos_ratio.append(np.mean(pos_ratios))
    
    # Trend: early layers vs late layers
    early_mean = np.mean(avg_agreement[:n_layers//3])
    mid_mean = np.mean(avg_agreement[n_layers//3:2*n_layers//3])
    late_mean = np.mean(avg_agreement[2*n_layers//3:])
    
    results = {
        'n_layers': int(n_layers),
        'early_sign_agreement': float(early_mean),
        'mid_sign_agreement': float(mid_mean),
        'late_sign_agreement': float(late_mean),
        'agreement_profile': [float(x) for x in avg_agreement],
        'pos_ratio_profile': [float(x) for x in avg_pos_ratio],
    }
    
    print(f"  Sign agreement across layers:")
    print(f"    Early (0-{n_layers//3}): {early_mean:.4f}")
    print(f"    Mid ({n_layers//3}-{2*n_layers//3}): {mid_mean:.4f}")
    print(f"    Late ({2*n_layers//3}-{n_layers}): {late_mean:.4f}")
    print(f"    Trend: {'INCREASING' if late_mean > early_mean else 'DECREASING' if late_mean < early_mean else 'FLAT'}")
    
    # Quantum acoustics prediction: early layers should have more stable signs
    # (less decoherence in lower-dimensional space)
    if early_mean > late_mean:
        print(f"    ** Supports quantum acoustics: early layers more coherent **")
        results['supports_quantum'] = True
    else:
        print(f"    ** Does NOT support quantum acoustics **")
        results['supports_quantum'] = False
    
    return results


def experiment_p616(all_features, model_name):
    """P616: Coherence structure of semantic directions."""
    print(f"\nP616: Coherence structure of semantic directions -- {model_name}")
    
    n = len(all_features)
    d = len(all_features[0]['h'])
    
    # Get semantic directions (top-10 by mean |contrib_k|)
    abs_contrib_matrix = np.array([f['abs_contrib_k'] for f in all_features])
    mean_abs_contrib = np.mean(abs_contrib_matrix, axis=0)
    top10_idx = np.argsort(mean_abs_contrib)[-10:][::-1]
    next10_idx = np.argsort(mean_abs_contrib)[-20:-10][::-1]
    
    # Sign matrix for semantic directions
    sign_matrix = np.array([f['signs'] for f in all_features])
    
    # Compute pairwise sign correlation for semantic directions
    sign_semantic = sign_matrix[:, top10_idx].astype(float)
    sign_next10 = sign_matrix[:, next10_idx].astype(float)
    
    # Clean zero-variance columns
    for k in range(10):
        if np.std(sign_semantic[:, k]) < 1e-10:
            sign_semantic[:, k] = np.random.randn(n) * 0.01
        if np.std(sign_next10[:, k]) < 1e-10:
            sign_next10[:, k] = np.random.randn(n) * 0.01
    
    C_semantic = np.corrcoef(sign_semantic.T)  # (10, 10)
    C_next10 = np.corrcoef(sign_next10.T)  # (10, 10)
    
    # Compare with random directions
    np.random.seed(42)
    random_idx = np.random.choice(d, 10, replace=False)
    sign_random = sign_matrix[:, random_idx].astype(float)
    for k in range(10):
        if np.std(sign_random[:, k]) < 1e-10:
            sign_random[:, k] = np.random.randn(n) * 0.01
    C_random = np.corrcoef(sign_random.T)
    
    # Statistics
    off_diag_semantic = C_semantic[np.triu_indices(10, k=1)]
    off_diag_next10 = C_next10[np.triu_indices(10, k=1)]
    off_diag_random = C_random[np.triu_indices(10, k=1)]
    
    # Fraction of positive correlations (coherence)
    frac_pos_semantic = np.mean(off_diag_semantic > 0)
    frac_pos_next10 = np.mean(off_diag_next10 > 0)
    frac_pos_random = np.mean(off_diag_random > 0)
    
    # Mean |correlation|
    mean_abs_semantic = np.mean(np.abs(off_diag_semantic))
    mean_abs_next10 = np.mean(np.abs(off_diag_next10))
    mean_abs_random = np.mean(np.abs(off_diag_random))
    
    results = {
        'semantic_frac_pos_corr': float(frac_pos_semantic),
        'next10_frac_pos_corr': float(frac_pos_next10),
        'random_frac_pos_corr': float(frac_pos_random),
        'semantic_mean_abs_corr': float(mean_abs_semantic),
        'next10_mean_abs_corr': float(mean_abs_next10),
        'random_mean_abs_corr': float(mean_abs_random),
        'semantic_vs_random_ratio': float(mean_abs_semantic / (mean_abs_random + 1e-10)),
    }
    
    print(f"  Semantic directions (top-10):")
    print(f"    Frac positive corr: {frac_pos_semantic:.4f} (random: {frac_pos_random:.4f})")
    print(f"    Mean |corr|: {mean_abs_semantic:.4f} (random: {mean_abs_random:.4f})")
    print(f"    Semantic/random ratio: {mean_abs_semantic / (mean_abs_random + 1e-10):.4f}")
    print(f"  Next-10 directions:")
    print(f"    Frac positive corr: {frac_pos_next10:.4f}")
    print(f"    Mean |corr|: {mean_abs_next10:.4f}")
    
    # Verdict: semantic directions more coherent than random?
    is_coherent = mean_abs_semantic > 1.5 * mean_abs_random
    print(f"    ** Verdict: {'COHERENT' if is_coherent else 'NOT COHERENT'} **")
    results['is_coherent'] = bool(is_coherent)
    
    return results


def experiment_p617(model, tokenizer, device, model_name, texts):
    """P617: Quantum projection prob = |Σ c_k · <top1|U_k>|²"""
    print(f"\nP617: Quantum projection prob -- {model_name}")
    
    n = len(texts)
    
    # Get W_U
    if hasattr(model, 'lm_head'):
        W_U = model.lm_head.weight.detach().cpu().float().numpy()
    else:
        W_U = model.get_output_embeddings().weight.detach().cpu().float().numpy()
    
    d = W_U.shape[1]
    
    # For computational efficiency, use SVD of W_U (top-d components)
    # W_U ≈ U_wu @ diag(S_wu) @ Vt_wu
    # But W_U is (vocab, d), so we can use Vt_wu as the projection basis
    # Actually, we need SVD of h, not W_U
    
    all_results = []
    
    for i, text in enumerate(texts):
        all_hidden, logits = get_hidden_states_all_layers(model, tokenizer, device, text)
        h = all_hidden[-1]
        
        top2 = np.argsort(logits)[-2:][::-1]
        top1, top2_idx = top2[0], top2[1]
        prob_top1 = torch.softmax(torch.tensor(logits), dim=-1)[top1].item()
        logit_gap = logits[top1] - logits[top2_idx]
        
        # Standard sigmoid prediction
        sigmoid_prob = 1.0 / (1.0 + np.exp(-logit_gap))
        
        # SVD of h
        h_2d = h.reshape(1, -1)
        U_h, S_h, Vt_h = np.linalg.svd(h_2d, full_matrices=False)
        # h = U_h[0] * S_h[0] * Vt_h[0]  (rank-1 decomposition)
        
        # Quantum projection:
        # prob(top1) = |<top1|h>|² / Σ_j |<j|h>|²
        # = (W_U[top1]·h)² / Σ_j (W_U[j]·h)²
        # But this is just softmax, so let's try a different approach
        
        # Instead, try: prob = |Σ_k c_k · W_U[top1,k]|²
        # where c_k = h[k] (component values in standard basis)
        # This is just (W_U[top1] · h)², which is logit² not prob
        
        # Better: use SVD basis of h to construct "quantum state"
        # h = Σ_k s_k * v_k where v_k are right singular vectors
        # prob(top1) ∝ |Σ_k s_k * <top1|v_k>|²
        
        # In standard basis: h = Σ_k h[k] * e_k
        # "Quantum state": |ψ> = Σ_k h[k] * |k>
        # "Observation": prob(top1) = |<top1|ψ>|² / <ψ|ψ>
        #               = |W_U[top1]·h|² / ||h||² * ||W_U[top1]||²
        
        # This gives: prob = logit² / (||h||² * ||W_U[top1]||²)
        # Which is NOT the same as softmax
        
        # Try: prob = cos²(angle between h and W_U[top1])
        cos_angle = np.dot(W_U[top1], h) / (np.linalg.norm(W_U[top1]) * np.linalg.norm(h) + 1e-10)
        quantum_prob = cos_angle ** 2
        
        # Compare: standard, sigmoid, quantum
        all_results.append({
            'prob_actual': float(prob_top1),
            'prob_sigmoid': float(sigmoid_prob),
            'prob_quantum_cos2': float(quantum_prob),
            'logit_gap': float(logit_gap),
            'cos_angle': float(cos_angle),
        })
        
        if (i+1) % 20 == 0:
            print(f"  Processed {i+1}/{n} texts...")
    
    # Correlations
    actual_probs = np.array([r['prob_actual'] for r in all_results])
    sigmoid_probs = np.array([r['prob_sigmoid'] for r in all_results])
    quantum_probs = np.array([r['prob_quantum_cos2'] for r in all_results])
    
    r_sigmoid, _ = stats.pearsonr(sigmoid_probs, actual_probs)
    r_quantum, _ = stats.pearsonr(quantum_probs, actual_probs)
    
    # Also try: quantum + sigmoid hybrid
    # prob = a * sigmoid(gap) + b * cos²(angle)
    from numpy.linalg import lstsq
    X_hybrid = np.column_stack([sigmoid_probs, quantum_probs, np.ones(n)])
    coeff, _, _, _ = lstsq(X_hybrid, actual_probs, rcond=None)
    pred_hybrid = X_hybrid @ coeff
    r_hybrid, _ = stats.pearsonr(pred_hybrid, actual_probs)
    
    results = {
        'sigmoid_prob_r': float(r_sigmoid),
        'quantum_cos2_prob_r': float(r_quantum),
        'hybrid_prob_r': float(r_hybrid),
        'hybrid_coeff_sigmoid': float(coeff[0]),
        'hybrid_coeff_quantum': float(coeff[1]),
        'hybrid_coeff_bias': float(coeff[2]),
    }
    
    print(f"  Sigmoid(gap)->prob r={r_sigmoid:.4f}")
    print(f"  Quantum cos^2(angle)->prob r={r_quantum:.4f}")
    print(f"  Hybrid (sigmoid+quantum)->prob r={r_hybrid:.4f}")
    print(f"    Coefficients: sigmoid={coeff[0]:.4f}, quantum={coeff[1]:.4f}, bias={coeff[2]:.4f}")
    
    return results


def experiment_p618(model, tokenizer, device, model_name, texts):
    """P618: Cross-model quantum invariants."""
    print(f"\nP618: Cross-model quantum invariants -- {model_name}")
    
    n = min(len(texts), 30)
    
    # Get W_U
    if hasattr(model, 'lm_head'):
        W_U = model.lm_head.weight.detach().cpu().float().numpy()
    else:
        W_U = model.get_output_embeddings().weight.detach().cpu().float().numpy()
    
    d = W_U.shape[1]
    
    # Compute quantum features for each text
    all_results = []
    
    for i, text in enumerate(texts[:n]):
        all_hidden, logits = get_hidden_states_all_layers(model, tokenizer, device, text)
        h = all_hidden[-1]
        
        top2 = np.argsort(logits)[-2:][::-1]
        top1, top2_idx = top2[0], top2[1]
        prob_top1 = torch.softmax(torch.tensor(logits), dim=-1)[top1].item()
        logit_gap = logits[top1] - logits[top2_idx]
        
        Delta_W = W_U[top1] - W_U[top2_idx]
        
        # Quantum features
        h_norm = np.linalg.norm(h)
        
        # 1. Sign agreement
        signs = np.sign(h * Delta_W)
        sign_agreement = np.abs(np.mean(signs))
        
        # 2. Sign entropy (Shannon entropy of positive/negative sign ratio)
        p_pos = np.mean(signs > 0)
        p_neg = 1 - p_pos
        if p_pos > 0 and p_neg > 0:
            sign_entropy = -p_pos * np.log2(p_pos) - p_neg * np.log2(p_neg)
        else:
            sign_entropy = 0
        
        # 3. Coherence: |Σ signs| / n (order parameter)
        coherence = np.abs(np.sum(signs)) / d
        
        # 4. Phase concentration: von Mises concentration parameter
        # Using h as "phase": angle = atan2(h_imag, h_real) where we use h and h_prev
        # For simplicity, use the angle of each h[k]*Delta_W[k]
        contrib = h * Delta_W
        phases = np.arctan2(0, contrib)  # All phases are 0 or pi (real only)
        # Better: use h[k] as amplitude, sign(h[k]) as phase
        R = np.abs(np.sum(np.sign(h) * np.exp(1j * 0))) / d  # Trivial for real h
        
        # 5. Participation ratio (effective dimensionality)
        h_squared = h ** 2
        PR = (np.sum(h_squared)) ** 2 / (np.sum(h_squared ** 2) + 1e-10)
        PR_ratio = PR / d
        
        # 6. Energy concentration in top-k
        h_sorted = np.sort(np.abs(h))[::-1]
        top10_energy = np.sum(h_sorted[:10]**2) / (np.sum(h_squared) + 1e-10)
        
        all_results.append({
            'sign_agreement': float(sign_agreement),
            'sign_entropy': float(sign_entropy),
            'coherence': float(coherence),
            'PR_ratio': float(PR_ratio),
            'top10_energy': float(top10_energy),
            'prob': float(prob_top1),
            'logit_gap': float(logit_gap),
        })
        
        if (i+1) % 10 == 0:
            print(f"  Processed {i+1}/{n} texts...")
    
    # Compute statistics
    sign_agreements = [r['sign_agreement'] for r in all_results]
    sign_entropies = [r['sign_entropy'] for r in all_results]
    coherences = [r['coherence'] for r in all_results]
    PR_ratios = [r['PR_ratio'] for r in all_results]
    top10_energies = [r['top10_energy'] for r in all_results]
    probs = [r['prob'] for r in all_results]
    logit_gaps = [r['logit_gap'] for r in all_results]
    
    # Correlations with prob
    r_sign_prob, _ = stats.pearsonr(sign_agreements, probs)
    r_entropy_prob, _ = stats.pearsonr(sign_entropies, probs)
    r_coherence_prob, _ = stats.pearsonr(coherences, probs)
    r_PR_prob, _ = stats.pearsonr(PR_ratios, probs)
    r_energy_prob, _ = stats.pearsonr(top10_energies, probs)
    
    results = {
        'mean_sign_agreement': float(np.mean(sign_agreements)),
        'mean_sign_entropy': float(np.mean(sign_entropies)),
        'mean_coherence': float(np.mean(coherences)),
        'mean_PR_ratio': float(np.mean(PR_ratios)),
        'mean_top10_energy': float(np.mean(top10_energies)),
        'sign_agreement_prob_r': float(r_sign_prob),
        'sign_entropy_prob_r': float(r_entropy_prob),
        'coherence_prob_r': float(r_coherence_prob),
        'PR_ratio_prob_r': float(r_PR_prob),
        'top10_energy_prob_r': float(r_energy_prob),
    }
    
    print(f"  Quantum features (mean):")
    print(f"    Sign agreement: {np.mean(sign_agreements):.4f}")
    print(f"    Sign entropy: {np.mean(sign_entropies):.4f} bits (max=1.0)")
    print(f"    Coherence: {np.mean(coherences):.4f}")
    print(f"    PR ratio: {np.mean(PR_ratios):.4f}")
    print(f"    Top-10 energy: {np.mean(top10_energies):.4f}")
    print(f"  Correlations with prob:")
    print(f"    Sign agreement->prob r={r_sign_prob:.4f}")
    print(f"    Sign entropy->prob r={r_entropy_prob:.4f}")
    print(f"    Coherence->prob r={r_coherence_prob:.4f}")
    print(f"    PR ratio->prob r={r_PR_prob:.4f}")
    print(f"    Top-10 energy->prob r={r_energy_prob:.4f}")
    
    return results


# ============================================================
# Main
# ============================================================
EXPERIMENTS = {
    'p613': lambda all_feat, mn, model=None, tok=None, dev=None, texts=None: experiment_p613(all_feat, mn),
    'p614': lambda all_feat, mn, model=None, tok=None, dev=None, texts=None: experiment_p614(model, tok, dev, mn, texts),
    'p615': lambda all_feat, mn, model=None, tok=None, dev=None, texts=None: experiment_p615(model, tok, dev, mn, texts),
    'p616': lambda all_feat, mn, model=None, tok=None, dev=None, texts=None: experiment_p616(all_feat, mn),
    'p617': lambda all_feat, mn, model=None, tok=None, dev=None, texts=None: experiment_p617(model, tok, dev, mn, texts),
    'p618': lambda all_feat, mn, model=None, tok=None, dev=None, texts=None: experiment_p618(model, tok, dev, mn, texts),
}


def compute_features(model, tokenizer, device, texts):
    """Compute h features at last layer for all texts."""
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
        
        h = outputs.hidden_states[-1][0, -1, :].cpu().float().numpy()
        logits = outputs.logits[0, -1, :].cpu().float().numpy()
        
        top2 = np.argsort(logits)[-2:][::-1]
        top1, top2_idx = top2[0], top2[1]
        prob_top1 = torch.softmax(outputs.logits[0, -1, :], dim=-1)[top1].item()
        
        Delta_W = W_U[top1] - W_U[top2_idx]
        contrib_k = h * Delta_W
        signs = np.sign(contrib_k)
        abs_contrib_k = np.abs(contrib_k)
        
        features = {
            'h': h,
            'Delta_W': Delta_W,
            'signs': signs,
            'abs_contrib_k': abs_contrib_k,
            'abs_h': np.abs(h),
            'abs_Delta_W': np.abs(Delta_W),
            'h_norm': np.linalg.norm(h),
            'logit_gap': logits[top1] - logits[top2_idx],
            'logit_max': logits[top1],
            'prob_top1': prob_top1,
            'top1_idx': int(top1),
            'top2_idx': int(top2_idx),
        }
        all_features.append(features)
    
    return all_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'glm4', 'deepseek7b'])
    parser.add_argument('--experiment', type=str, required=True, choices=list(EXPERIMENTS.keys()))
    args = parser.parse_args()
    
    t0 = time.time()
    
    model, tokenizer, device = load_model(args.model)
    
    print(f"Computing features for {len(TEST_TEXTS)} texts...")
    all_features = compute_features(model, tokenizer, device, TEST_TEXTS)
    
    results = EXPERIMENTS[args.experiment](
        all_features, args.model,
        model=model, tok=tokenizer, dev=device, texts=TEST_TEXTS
    )
    
    os.makedirs('results/phase_cxl', exist_ok=True)
    out_path = f'results/phase_cxl/{args.model}_{args.experiment}.json'
    
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
    
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
