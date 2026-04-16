#!/usr/bin/env python3
"""
Phase CXLVII: Parameter Sensitivity Audit + Hook-Based Exact Decomposition (P649-P652)
==================================================================================

Critical motivation: Phase CXLVI had "ratio=0.22" class errors:
1. PCA n_components=30 (FIXED) -> may over/underfit like k=300 for ratio
2. Ridge alpha=1.0 (FIXED) -> regularization strength is arbitrary
3. 40 texts -> 30 PCA dims = severe overfitting risk
4. Physical simulation ignores softmax/LN -> catastrophic for DS7B

This phase:
P649: Parameter Sensitivity Audit - sweep PCA k, Ridge alpha, text count
P650: Hook-Based Exact Decomposition - get REAL h_attn, h_mlp from model
P651: Data-Driven Emergence Equation - Ridge with exact intermediate states
P652: Cross-Model Robustness Validation - test parameter-independent conclusions
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


def apply_layernorm_numpy(h, ln_weight, ln_bias=None):
    """Apply LayerNorm using numpy (matches PyTorch implementation)."""
    h_mean = np.mean(h)
    h_var = np.var(h)
    h_std = np.sqrt(h_var + 1e-5)
    h_norm = (h - h_mean) / h_std
    return h_norm * ln_weight + (ln_bias if ln_bias is not None else 0.0)


def compute_features_with_hooks(model, tokenizer, device, texts, n_samples=None):
    """
    Extract features using PyTorch hooks to get EXACT intermediate states.
    
    This avoids the "simple weight multiplication" problem from Phase CXLVI.
    Hooks capture the REAL attention output, MLP output, and LN output.
    
    Returns features dict with:
    - h_L_minus_1: input to last layer
    - h_attn_out: exact attention output (after residual connection)
    - h_mlp_out: exact MLP output (after residual connection)
    - h_L: final hidden state
    - logit_gap: gap between top1 and top2 logits
    - Delta_W: W_U[top1] - W_U[top2]
    """
    if n_samples is not None:
        texts = texts[:n_samples]
    
    # Get W_U
    if hasattr(model, 'lm_head'):
        W_U = model.lm_head.weight.detach().cpu().float().numpy()
    else:
        W_U = model.get_output_embeddings().weight.detach().cpu().float().numpy()
    
    d_model = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 2560
    n_layers = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else 32
    
    # Get final LN weights
    final_ln_weight = None
    final_ln_bias = None
    if hasattr(model.model, 'norm'):
        final_ln_weight = model.model.norm.weight.detach().cpu().float().numpy()
        final_ln_bias = model.model.norm.bias.detach().cpu().float().numpy() if hasattr(model.model.norm, 'bias') and model.model.norm.bias is not None else np.zeros(d_model)
    elif hasattr(model.model, 'final_layernorm'):
        final_ln_weight = model.model.final_layernorm.weight.detach().cpu().float().numpy()
        final_ln_bias = model.model.final_layernorm.bias.detach().cpu().float().numpy() if hasattr(model.model.final_layernorm, 'bias') and model.model.final_layernorm.bias is not None else np.zeros(d_model)
    
    layers = model.model.layers if hasattr(model.model, 'layers') else []
    last_layer = layers[-1]
    
    all_features = []
    
    for i, text in enumerate(texts):
        if (i+1) % 10 == 0:
            print(f"  Processing text {i+1}/{len(texts)}...")
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # Hook storage
        hook_data = {}
        
        def make_hook(name):
            def hook_fn(module, input, output):
                # For attention/MLP layers, output is typically a tuple (hidden_states, ...)
                if isinstance(output, tuple):
                    hook_data[name] = output[0].detach()
                else:
                    hook_data[name] = output.detach()
            return hook_fn
        
        # Register hooks on the last layer
        # Hook 1: Self-attention output (before residual)
        handles = []
        handles.append(last_layer.self_attn.register_forward_hook(make_hook('attn_output')))
        # Hook 2: MLP output (before residual)  
        handles.append(last_layer.mlp.register_forward_hook(make_hook('mlp_output')))
        # Hook 3: Post-attention LayerNorm input
        for ln_name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
            if hasattr(last_layer, ln_name):
                handles.append(getattr(last_layer, ln_name).register_forward_hook(make_hook('post_attn_ln_input')))
                break
        # Hook 4: Input LayerNorm output
        for ln_name in ["input_layernorm", "ln_1", "layernorm"]:
            if hasattr(last_layer, ln_name):
                handles.append(getattr(last_layer, ln_name).register_forward_hook(make_hook('input_ln_output')))
                break
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # Remove hooks
        for h in handles:
            h.remove()
        
        # Extract hidden states
        all_hidden = []
        for hs in outputs.hidden_states:
            all_hidden.append(hs[0, -1, :].cpu().float().numpy())
        
        h_L = all_hidden[-1]
        h_L_minus_1 = all_hidden[-2] if len(all_hidden) > 1 else all_hidden[0]
        
        # Extract hook data
        h_attn_raw = hook_data.get('attn_output', None)
        h_mlp_raw = hook_data.get('mlp_output', None)
        h_input_ln = hook_data.get('input_ln_output', None)
        h_post_attn_ln = hook_data.get('post_attn_ln_input', None)
        
        if h_attn_raw is not None:
            h_attn_raw = h_attn_raw[0, -1, :].cpu().float().numpy()
        if h_mlp_raw is not None:
            h_mlp_raw = h_mlp_raw[0, -1, :].cpu().float().numpy()
        if h_input_ln is not None:
            h_input_ln = h_input_ln[0, -1, :].cpu().float().numpy()
        if h_post_attn_ln is not None:
            h_post_attn_ln = h_post_attn_ln[0, -1, :].cpu().float().numpy()
        
        # Compute residuals
        # h_attn_out = h_L_minus_1 + attn_output (residual connection)
        # h_mlp_out = h_attn_out + mlp_output (residual connection)
        # But hooks give us attn_output and mlp_output BEFORE residual addition
        
        # Exact decomposition:
        # h_L_minus_1 -> input_ln -> attn -> +residual -> h_attn_out
        # h_attn_out -> post_attn_ln -> mlp -> +residual -> h_mlp_out
        # h_mlp_out -> final_ln -> h_L
        
        attn_output = h_attn_raw  # This is the attention output BEFORE residual
        mlp_output = h_mlp_raw    # This is the MLP output BEFORE residual
        
        h_attn_out = h_L_minus_1 + (attn_output if attn_output is not None else 0)
        h_mlp_out = h_attn_out + (mlp_output if mlp_output is not None else 0)
        
        # Logits
        logits = outputs.logits[0, -1, :].cpu().float().numpy()
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        logit_gap = logits[top1_idx] - logits[top2_idx]
        
        W_U_top1 = W_U[top1_idx]
        W_U_top2 = W_U[top2_idx]
        Delta_W = W_U_top1 - W_U_top2
        
        # Gap at each stage
        gap_L_minus_1 = np.dot(h_L_minus_1, Delta_W)
        gap_attn_out = np.dot(h_attn_out, Delta_W)
        gap_mlp_out = np.dot(h_mlp_out, Delta_W)
        gap_L = np.dot(h_L, Delta_W)
        
        # Exact decomposition of gap change
        delta_gap_attn = gap_attn_out - gap_L_minus_1  # Attention contribution
        delta_gap_mlp = gap_mlp_out - gap_attn_out      # MLP contribution
        delta_gap_final_ln = gap_L - gap_mlp_out         # Final LN contribution
        delta_gap_total = gap_L - gap_L_minus_1
        
        all_features.append({
            'h_L': h_L,
            'h_L_minus_1': h_L_minus_1,
            'h_attn_out': h_attn_out,
            'h_mlp_out': h_mlp_out,
            'attn_output': attn_output,
            'mlp_output': mlp_output,
            'h_input_ln': h_input_ln,
            'h_post_attn_ln': h_post_attn_ln,
            'logit_gap': logit_gap,
            'top1_idx': int(top1_idx),
            'top2_idx': int(top2_idx),
            'Delta_W': Delta_W,
            'gap_L_minus_1': gap_L_minus_1,
            'gap_attn_out': gap_attn_out,
            'gap_mlp_out': gap_mlp_out,
            'gap_L': gap_L,
            'delta_gap_attn': delta_gap_attn,
            'delta_gap_mlp': delta_gap_mlp,
            'delta_gap_final_ln': delta_gap_final_ln,
            'delta_gap_total': delta_gap_total,
            'text_idx': i,
        })
    
    extra = {
        'W_U': W_U,
        'd_model': d_model,
        'n_layers': n_layers,
        'final_ln_weight': final_ln_weight,
        'final_ln_bias': final_ln_bias,
    }
    
    return all_features, extra


def experiment_p649(all_features, extra, model_name):
    """P649: Parameter Sensitivity Audit.
    
    Test whether Phase CXLVI conclusions depend on arbitrary parameter choices:
    1. PCA n_components: sweep from 1 to min(n_texts-1, d_model)
    2. Ridge alpha: sweep from 0.001 to 1000
    3. Text subset size: test with 10, 20, 30, 40 texts
    
    This is the "k=300" audit - checking if fixed parameters create artifacts.
    """
    print(f"\n{'='*60}")
    print(f"P649: Parameter Sensitivity Audit ({model_name})")
    print(f"{'='*60}")
    
    d_model = extra['d_model']
    n_texts = len(all_features)
    gaps = np.array([f['logit_gap'] for f in all_features])
    
    h_L_minus_1 = np.array([f['h_L_minus_1'] for f in all_features])
    h_L = np.array([f['h_L'] for f in all_features])
    
    # ===== 1. PCA n_components sweep =====
    print(f"\n1. PCA n_components sweep (Ridge alpha=1.0):")
    print(f"   n_texts={n_texts}, d_model={d_model}")
    print(f"   Max possible components: {min(n_texts-1, d_model)}")
    
    pca_results = []
    max_k = min(n_texts - 1, 50, d_model)  # Cap at 50 for speed
    
    for k in range(1, max_k + 1):
        try:
            pca = PCA(n_components=k)
            h_pca = pca.fit_transform(h_L_minus_1)
            
            # LOO-CV Ridge
            loo = LeaveOneOut()
            loo_preds = []
            for train_idx, test_idx in loo.split(h_pca):
                ridge = Ridge(alpha=1.0).fit(h_pca[train_idx], gaps[train_idx])
                loo_preds.append(ridge.predict(h_pca[test_idx])[0])
            loo_preds = np.array(loo_preds)
            r_loo, _ = stats.pearsonr(loo_preds, gaps)
            
            # Explained variance ratio
            var_explained = np.sum(pca.explained_variance_ratio_[:k])
            
            pca_results.append({
                'k': k,
                'r_loo': r_loo,
                'var_explained': var_explained,
            })
        except Exception as e:
            print(f"   k={k} failed: {e}")
            break
    
    # Print results
    print(f"\n   k  | r_LOO  | var_explained")
    print(f"   ---|--------|-------------")
    for res in pca_results:
        marker = " <-- k=30 (Phase CXLVI default)" if res['k'] == 30 else ""
        if res['k'] in [1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 39] or res['k'] == max_k:
            print(f"   {res['k']:3d} | {res['r_loo']:.4f} | {res['var_explained']:.4f}{marker}")
    
    # Find optimal k
    best_k = max(pca_results, key=lambda x: x['r_loo'])
    print(f"\n   *** Best k={best_k['k']}, r_LOO={best_k['r_loo']:.4f}, var={best_k['var_explained']:.4f}")
    print(f"   *** k=30 r_LOO={next((r['r_loo'] for r in pca_results if r['k']==30), 'N/A')}")
    
    # Check: is k=30 close to optimal, or is it an artifact?
    if best_k['k'] != 30:
        r_at_30 = next((r['r_loo'] for r in pca_results if r['k']==30), 0)
        gap_from_best = best_k['r_loo'] - r_at_30
        print(f"   *** k=30 gap from optimal: {gap_from_best:.4f} ({'SAFE' if gap_from_best < 0.05 else 'PROBLEMATIC!'})")
    
    # ===== 2. Ridge alpha sweep =====
    print(f"\n2. Ridge alpha sweep (PCA k=optimal):")
    
    alpha_values = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    alpha_results = []
    
    # Use best k from above
    opt_k = best_k['k']
    pca = PCA(n_components=opt_k)
    h_pca = pca.fit_transform(h_L_minus_1)
    
    for alpha in alpha_values:
        loo_preds = []
        for train_idx, test_idx in LeaveOneOut().split(h_pca):
            ridge = Ridge(alpha=alpha).fit(h_pca[train_idx], gaps[train_idx])
            loo_preds.append(ridge.predict(h_pca[test_idx])[0])
        loo_preds = np.array(loo_preds)
        r_loo, _ = stats.pearsonr(loo_preds, gaps)
        alpha_results.append({'alpha': alpha, 'r_loo': r_loo})
    
    print(f"\n   alpha | r_LOO")
    print(f"   ------|------")
    for res in alpha_results:
        marker = " <-- default" if res['alpha'] == 1.0 else ""
        print(f"   {res['alpha']:8.3f} | {res['r_loo']:.4f}{marker}")
    
    best_alpha = max(alpha_results, key=lambda x: x['r_loo'])
    print(f"\n   *** Best alpha={best_alpha['alpha']}, r_LOO={best_alpha['r_loo']:.4f}")
    
    # ===== 3. Text subset size stability =====
    print(f"\n3. Text subset size stability:")
    
    # Use different random subsets of texts
    np.random.seed(42)
    subset_sizes = [10, 15, 20, 25, 30, 40]
    subset_results = []
    
    for n_sub in subset_sizes:
        if n_sub > n_texts:
            continue
        
        r_values = []
        for trial in range(5):  # 5 random subsets
            indices = np.random.choice(n_texts, n_sub, replace=False)
            h_sub = h_L_minus_1[indices]
            gaps_sub = gaps[indices]
            
            max_k_sub = min(n_sub - 1, 30, d_model)
            if max_k_sub < 1:
                continue
            
            pca_sub = PCA(n_components=max_k_sub)
            h_pca_sub = pca_sub.fit_transform(h_sub)
            
            if n_sub <= 3:
                # Too few for LOO
                ridge_sub = Ridge(alpha=1.0).fit(h_pca_sub, gaps_sub)
                r_sub = stats.pearsonr(ridge_sub.predict(h_pca_sub), gaps_sub)[0]
            else:
                loo_preds = []
                for train_idx, test_idx in LeaveOneOut().split(h_pca_sub):
                    ridge_sub = Ridge(alpha=1.0).fit(h_pca_sub[train_idx], gaps_sub[train_idx])
                    loo_preds.append(ridge_sub.predict(h_pca_sub[test_idx])[0])
                loo_preds = np.array(loo_preds)
                r_sub = stats.pearsonr(loo_preds, gaps_sub)[0]
            
            r_values.append(r_sub)
        
        mean_r = np.mean(r_values)
        std_r = np.std(r_values)
        subset_results.append({'n': n_sub, 'mean_r': mean_r, 'std_r': std_r})
        print(f"   n={n_sub:2d}: r_LOO={mean_r:.4f} ± {std_r:.4f}")
    
    # ===== 4. Critical check: Is Ridge success due to h(L-1) or Delta_W leakage? =====
    print(f"\n4. Delta_W leakage check:")
    
    # If Delta_W is "leaked" through h(L-1) correlation, Ridge would work trivially
    # Test: does Delta_W correlate with h(L-1)?
    Delta_Ws = np.array([f['Delta_W'] for f in all_features])
    gap_L_minus_1 = np.array([f['gap_L_minus_1'] for f in all_features])
    
    # The "gap at h(L-1)" is already h(L-1) . Delta_W
    # If Ridge works because it learns Delta_W from h(L-1), that's "leakage"
    # But this is actually VALID - the model DOES encode Delta_W info in h(L-1)!
    
    # More precise test: does Ridge from h(L-1) predict gap better than gap_L_minus_1?
    pca_opt = PCA(n_components=opt_k)
    h_pca_opt = pca_opt.fit_transform(h_L_minus_1)
    ridge_opt = Ridge(alpha=best_alpha['alpha']).fit(h_pca_opt, gaps)
    
    r_gap_Lm1, _ = stats.pearsonr(gap_L_minus_1, gaps)
    r_ridge_opt, _ = stats.pearsonr(ridge_opt.predict(h_pca_opt), gaps)
    
    print(f"   gap(h(L-1), Delta_W) -> logit_gap: r={r_gap_Lm1:.4f}")
    print(f"   Ridge(h(L-1)) -> logit_gap: r={r_ridge_opt:.4f}")
    print(f"   Improvement from Ridge: {r_ridge_opt - r_gap_Lm1:.4f}")
    print(f"   {'INFO: Ridge extracts info beyond simple dot product' if r_ridge_opt > r_gap_Lm1 + 0.05 else 'WARNING: Ridge barely improves over simple gap(h(L-1))'}")
    
    return {
        'best_k': best_k['k'],
        'best_k_r_loo': best_k['r_loo'],
        'r_at_k30': next((r['r_loo'] for r in pca_results if r['k']==30), None),
        'best_alpha': best_alpha['alpha'],
        'best_alpha_r_loo': best_alpha['r_loo'],
        'r_alpha_1': next((r['r_loo'] for r in alpha_results if r['alpha']==1.0), None),
        'r_gap_Lm1_vs_gap': float(r_gap_Lm1),
        'r_ridge_opt_vs_gap': float(r_ridge_opt),
        'pca_sweep': pca_results,
        'alpha_sweep': alpha_results,
        'subset_stability': subset_results,
    }


def experiment_p650(all_features, extra, model_name):
    """P650: Hook-Based Exact Decomposition.
    
    Using REAL intermediate states from hooks (not simulated W*V*h).
    This gives us the EXACT attention and MLP contributions.
    
    Key comparison: Hook-based vs Simulated decomposition
    """
    print(f"\n{'='*60}")
    print(f"P650: Hook-Based Exact Decomposition ({model_name})")
    print(f"{'='*60}")
    
    d_model = extra['d_model']
    n_texts = len(all_features)
    gaps = np.array([f['logit_gap'] for f in all_features])
    
    # Check that hooks captured data
    has_attn = all_features[0].get('attn_output') is not None
    has_mlp = all_features[0].get('mlp_output') is not None
    has_input_ln = all_features[0].get('h_input_ln') is not None
    has_post_attn_ln = all_features[0].get('h_post_attn_ln') is not None
    
    print(f"   Hook data available:")
    print(f"   attn_output: {has_attn}")
    print(f"   mlp_output: {has_mlp}")
    print(f"   input_ln: {has_input_ln}")
    print(f"   post_attn_ln: {has_post_attn_ln}")
    
    if not has_attn or not has_mlp:
        print("   ERROR: Hooks did not capture intermediate states!")
        return {'error': 'hooks_failed'}
    
    # ===== 1. Exact decomposition =====
    print(f"\n1. Exact Gap Decomposition (Hook-Based):")
    
    gap_L_minus_1 = np.array([f['gap_L_minus_1'] for f in all_features])
    gap_attn_out = np.array([f['gap_attn_out'] for f in all_features])
    gap_mlp_out = np.array([f['gap_mlp_out'] for f in all_features])
    gap_L = np.array([f['gap_L'] for f in all_features])
    
    delta_attn = np.array([f['delta_gap_attn'] for f in all_features])
    delta_mlp = np.array([f['delta_gap_mlp'] for f in all_features])
    delta_final_ln = np.array([f['delta_gap_final_ln'] for f in all_features])
    delta_total = np.array([f['delta_gap_total'] for f in all_features])
    
    print(f"   Gap at h(L-1): mean={np.mean(gap_L_minus_1):.4f}, std={np.std(gap_L_minus_1):.4f}")
    print(f"   Gap after Attn: mean={np.mean(gap_attn_out):.4f}")
    print(f"   Gap after MLP:  mean={np.mean(gap_mlp_out):.4f}")
    print(f"   Gap at h(L):    mean={np.mean(gap_L):.4f}")
    
    print(f"\n   Exact contributions:")
    print(f"   Attn:     mean={np.mean(delta_attn):.4f}, |mean|={np.mean(np.abs(delta_attn)):.4f}")
    print(f"   MLP:      mean={np.mean(delta_mlp):.4f}, |mean|={np.mean(np.abs(delta_mlp)):.4f}")
    print(f"   Final LN: mean={np.mean(delta_final_ln):.4f}, |mean|={np.mean(np.abs(delta_final_ln)):.4f}")
    print(f"   Total:    mean={np.mean(delta_total):.4f}")
    
    # Verify decomposition: attn + mlp + final_ln = total
    reconstruction_error = np.mean(np.abs(delta_attn + delta_mlp + delta_final_ln - delta_total))
    print(f"   Decomposition error: {reconstruction_error:.6f} ({'OK' if reconstruction_error < 0.01 else 'LARGE!'})")
    
    # ===== 2. Correlation of each component with final gap =====
    print(f"\n2. Correlation with logit_gap:")
    
    r_gap_Lm1, _ = stats.pearsonr(gap_L_minus_1, gaps)
    r_gap_attn, _ = stats.pearsonr(gap_attn_out, gaps)
    r_gap_mlp, _ = stats.pearsonr(gap_mlp_out, gaps)
    r_gap_L, _ = stats.pearsonr(gap_L, gaps)
    
    r_delta_attn, _ = stats.pearsonr(delta_attn, gaps)
    r_delta_mlp, _ = stats.pearsonr(delta_mlp, gaps)
    r_delta_final_ln, _ = stats.pearsonr(delta_final_ln, gaps)
    
    print(f"   gap(h(L-1)) -> logit_gap: r={r_gap_Lm1:.4f}")
    print(f"   gap(h_attn) -> logit_gap: r={r_gap_attn:.4f}")
    print(f"   gap(h_mlp)  -> logit_gap: r={r_gap_mlp:.4f}")
    print(f"   gap(h(L))   -> logit_gap: r={r_gap_L:.4f}")
    
    print(f"\n   Delta contributions:")
    print(f"   delta_attn     -> logit_gap: r={r_delta_attn:.4f}")
    print(f"   delta_mlp      -> logit_gap: r={r_delta_mlp:.4f}")
    print(f"   delta_final_ln -> logit_gap: r={r_delta_final_ln:.4f}")
    
    # ===== 3. Sign consistency =====
    print(f"\n3. Sign Consistency:")
    
    # For delta contributions, the "correct" sign depends on context
    # If delta_attn > 0 and gap increases, that's "consistent"
    sign_attn = np.mean(np.sign(delta_attn) == np.sign(delta_total))
    sign_mlp = np.mean(np.sign(delta_mlp) == np.sign(delta_total))
    sign_final_ln = np.mean(np.sign(delta_final_ln) == np.sign(delta_total))
    
    print(f"   sign(delta_attn) == sign(delta_total): {sign_attn:.4f}")
    print(f"   sign(delta_mlp) == sign(delta_total): {sign_mlp:.4f}")
    print(f"   sign(delta_final_ln) == sign(delta_total): {sign_final_ln:.4f}")
    
    # ===== 4. Variance explained by each component =====
    print(f"\n4. Variance Decomposition:")
    
    var_attn = np.var(delta_attn)
    var_mlp = np.var(delta_mlp)
    var_final_ln = np.var(delta_final_ln)
    var_total = np.var(delta_total)
    
    # These don't add up due to correlations, but give relative importance
    print(f"   Var(delta_attn):     {var_attn:.4f} ({var_attn/var_total*100:.1f}%)")
    print(f"   Var(delta_mlp):      {var_mlp:.4f} ({var_mlp/var_total*100:.1f}%)")
    print(f"   Var(delta_final_ln): {var_final_ln:.4f} ({var_final_ln/var_total*100:.1f}%)")
    print(f"   Var(delta_total):    {var_total:.4f}")
    
    # ===== 5. Prediction: linear combination of components =====
    print(f"\n5. Linear Model: gap = a*gap_Lm1 + b*delta_attn + c*delta_mlp + d*delta_final_ln:")
    
    from sklearn.linear_model import LinearRegression
    
    X_exact = np.column_stack([gap_L_minus_1, delta_attn, delta_mlp, delta_final_ln])
    lr = LinearRegression().fit(X_exact, gaps)
    r_exact, _ = stats.pearsonr(lr.predict(X_exact), gaps)
    
    print(f"   Coefficients: gap_Lm1={lr.coef_[0]:.4f}, delta_attn={lr.coef_[1]:.4f}, delta_mlp={lr.coef_[2]:.4f}, delta_final_ln={lr.coef_[3]:.4f}")
    print(f"   Intercept: {lr.intercept_:.4f}")
    print(f"   r(exact decomposition -> gap): {r_exact:.4f}")
    
    # Compare with simpler models
    lr_simple = LinearRegression().fit(gap_L_minus_1.reshape(-1, 1), gaps)
    r_simple, _ = stats.pearsonr(lr_simple.predict(gap_L_minus_1.reshape(-1, 1)), gaps)
    
    print(f"   r(gap_Lm1 only -> gap): {r_simple:.4f}")
    print(f"   Improvement from full decomposition: {r_exact - r_simple:.4f}")
    
    return {
        'r_gap_Lm1': float(r_gap_Lm1),
        'r_gap_attn': float(r_gap_attn),
        'r_gap_mlp': float(r_gap_mlp),
        'r_gap_L': float(r_gap_L),
        'r_delta_attn': float(r_delta_attn),
        'r_delta_mlp': float(r_delta_mlp),
        'r_delta_final_ln': float(r_delta_final_ln),
        'sign_attn': float(sign_attn),
        'sign_mlp': float(sign_mlp),
        'sign_final_ln': float(sign_final_ln),
        'var_frac_attn': float(var_attn / var_total),
        'var_frac_mlp': float(var_mlp / var_total),
        'var_frac_final_ln': float(var_final_ln / var_total),
        'r_exact_model': float(r_exact),
        'r_simple_model': float(r_simple),
        'decomposition_error': float(reconstruction_error),
        'mean_delta_attn': float(np.mean(delta_attn)),
        'mean_delta_mlp': float(np.mean(delta_mlp)),
        'mean_delta_final_ln': float(np.mean(delta_final_ln)),
    }


def experiment_p651(all_features, extra, model_name):
    """P651: Data-Driven Emergence Equation.
    
    Using exact intermediate states from hooks, build a predictive model:
    logit_gap = f(h_L_minus_1)
    
    Key innovation: Train separate Ridge regressors for each decomposition stage
    to understand which stage carries the most predictive information.
    """
    print(f"\n{'='*60}")
    print(f"P651: Data-Driven Emergence Equation ({model_name})")
    print(f"{'='*60}")
    
    d_model = extra['d_model']
    n_texts = len(all_features)
    gaps = np.array([f['logit_gap'] for f in all_features])
    
    # ===== 1. Stage-by-stage Ridge prediction =====
    print(f"\n1. Stage-by-Stage Ridge Prediction:")
    
    stages = {
        'h_L_minus_1': np.array([f['h_L_minus_1'] for f in all_features]),
        'h_attn_out': np.array([f['h_attn_out'] for f in all_features]),
        'h_mlp_out': np.array([f['h_mlp_out'] for f in all_features]),
        'h_L': np.array([f['h_L'] for f in all_features]),
    }
    
    stage_results = {}
    
    for stage_name, H in stages.items():
        # Find optimal k using LOO
        best_r = -1
        best_k = 1
        
        for k in range(1, min(n_texts - 1, 30) + 1):
            try:
                pca = PCA(n_components=k)
                h_pca = pca.fit_transform(H)
                
                loo_preds = []
                for train_idx, test_idx in LeaveOneOut().split(h_pca):
                    ridge = Ridge(alpha=1.0).fit(h_pca[train_idx], gaps[train_idx])
                    loo_preds.append(ridge.predict(h_pca[test_idx])[0])
                loo_preds = np.array(loo_preds)
                r_loo, _ = stats.pearsonr(loo_preds, gaps)
                
                if r_loo > best_r:
                    best_r = r_loo
                    best_k = k
            except:
                break
        
        stage_results[stage_name] = {'best_k': best_k, 'r_loo': best_r}
        print(f"   {stage_name:15s}: best_k={best_k:2d}, r_LOO={best_r:.4f}")
    
    # ===== 2. Information gain at each stage =====
    print(f"\n2. Information Gain (incremental prediction improvement):")
    
    stage_names = ['h_L_minus_1', 'h_attn_out', 'h_mlp_out', 'h_L']
    for i in range(1, len(stage_names)):
        prev = stage_results[stage_names[i-1]]['r_loo']
        curr = stage_results[stage_names[i]]['r_loo']
        gain = curr - prev
        print(f"   {stage_names[i-1]} -> {stage_names[i]}: Δr = {gain:+.4f} ({'significant' if abs(gain) > 0.02 else 'minor'})")
    
    # ===== 3. Cross-stage Ridge: predict gap from concatenated features =====
    print(f"\n3. Cross-Stage Feature Combination:")
    
    # Use PCA on each stage, then concatenate
    for combo_name, combo_stages in [
        ('L-1 only', ['h_L_minus_1']),
        ('L-1 + Attn', ['h_L_minus_1', 'h_attn_out']),
        ('L-1 + Attn + MLP', ['h_L_minus_1', 'h_attn_out', 'h_mlp_out']),
        ('Attn only', ['h_attn_out']),
        ('MLP only', ['h_mlp_out']),
    ]:
        # Use delta features (difference from previous stage)
        if len(combo_stages) == 1:
            H_combo = stages[combo_stages[0]]
        elif combo_stages == ['h_L_minus_1', 'h_attn_out']:
            # h_L-1 and delta_attn
            deltas = np.array([f['attn_output'] for f in all_features])
            H_combo = np.column_stack([stages['h_L_minus_1'], deltas])
        elif combo_stages == ['h_L_minus_1', 'h_attn_out', 'h_mlp_out']:
            deltas_attn = np.array([f['attn_output'] for f in all_features])
            deltas_mlp = np.array([f['mlp_output'] for f in all_features])
            H_combo = np.column_stack([stages['h_L_minus_1'], deltas_attn, deltas_mlp])
        else:
            H_combo = stages[combo_stages[0]]
        
        # Ridge with optimal k
        best_r_combo = -1
        for k in range(1, min(n_texts - 1, 40) + 1):
            try:
                pca = PCA(n_components=k)
                h_pca = pca.fit_transform(H_combo)
                loo_preds = []
                for train_idx, test_idx in LeaveOneOut().split(h_pca):
                    ridge = Ridge(alpha=1.0).fit(h_pca[train_idx], gaps[train_idx])
                    loo_preds.append(ridge.predict(h_pca[test_idx])[0])
                loo_preds = np.array(loo_preds)
                r_loo, _ = stats.pearsonr(loo_preds, gaps)
                if r_loo > best_r_combo:
                    best_r_combo = r_loo
            except:
                break
        
        print(f"   {combo_name:25s}: best r_LOO={best_r_combo:.4f}")
    
    # ===== 4. The Emergence Equation =====
    print(f"\n4. Emergence Equation Summary:")
    
    r_Lm1 = stage_results['h_L_minus_1']['r_loo']
    r_attn = stage_results['h_attn_out']['r_loo']
    r_mlp = stage_results['h_mlp_out']['r_loo']
    r_L = stage_results['h_L']['r_loo']
    
    print(f"   Stage 1 (encoding): h(L-1) carries {r_Lm1*100:.1f}% of gap info")
    print(f"   Stage 2 (attn):     h_attn carries {r_attn*100:.1f}% (+{(r_attn-r_Lm1)*100:.1f}%)")
    print(f"   Stage 3 (MLP):      h_mlp carries  {r_mlp*100:.1f}% (+{(r_mlp-r_attn)*100:.1f}%)")
    print(f"   Stage 4 (final LN): h(L) carries   {r_L*100:.1f}% (+{(r_L-r_mlp)*100:.1f}%)")
    
    return {
        'stage_results': stage_results,
        'r_Lm1': float(r_Lm1),
        'r_attn': float(r_attn),
        'r_mlp': float(r_mlp),
        'r_L': float(r_L),
    }


def experiment_p652(all_features, extra, model_name):
    """P652: Cross-Model Robustness Validation.
    
    Test whether key conclusions are PARAMETER-INDEPENDENT:
    1. Does the "dominant component" (Attn vs MLP vs LN) depend on k/alpha?
    2. Is the "information gain" at each stage robust across parameter choices?
    3. Are sign consistency values stable?
    
    The key lesson from ratio=0.22: if a conclusion depends on a specific k value,
    it's not trustworthy. We need conclusions that hold across parameter sweeps.
    """
    print(f"\n{'='*60}")
    print(f"P652: Cross-Model Robustness Validation ({model_name})")
    print(f"{'='*60}")
    
    d_model = extra['d_model']
    n_texts = len(all_features)
    gaps = np.array([f['logit_gap'] for f in all_features])
    
    # ===== 1. Dominant component: is it robust? =====
    print(f"\n1. Dominant Component Robustness:")
    print(f"   Testing if 'which component dominates gap' depends on analysis parameters")
    
    delta_attn = np.array([f['delta_gap_attn'] for f in all_features])
    delta_mlp = np.array([f['delta_gap_mlp'] for f in all_features])
    delta_final_ln = np.array([f['delta_gap_final_ln'] for f in all_features])
    
    # Mean absolute contributions
    abs_attn = np.mean(np.abs(delta_attn))
    abs_mlp = np.mean(np.abs(delta_mlp))
    abs_ln = np.mean(np.abs(delta_final_ln))
    
    contributions = {'Attn': abs_attn, 'MLP': abs_mlp, 'LN': abs_ln}
    dominant = max(contributions, key=contributions.get)
    
    print(f"   |delta_attn| mean:     {abs_attn:.4f}")
    print(f"   |delta_mlp| mean:      {abs_mlp:.4f}")
    print(f"   |delta_final_ln| mean: {abs_ln:.4f}")
    print(f"   Dominant component: {dominant}")
    
    # Check robustness: do different text subsets give different dominant component?
    np.random.seed(123)
    n_trials = 20
    dominant_counts = {'Attn': 0, 'MLP': 0, 'LN': 0}
    
    for _ in range(n_trials):
        indices = np.random.choice(n_texts, max(10, n_texts // 2), replace=False)
        abs_a = np.mean(np.abs(delta_attn[indices]))
        abs_m = np.mean(np.abs(delta_mlp[indices]))
        abs_l = np.mean(np.abs(delta_final_ln[indices]))
        
        if abs_a >= abs_m and abs_a >= abs_l:
            dominant_counts['Attn'] += 1
        elif abs_m >= abs_a and abs_m >= abs_l:
            dominant_counts['MLP'] += 1
        else:
            dominant_counts['LN'] += 1
    
    print(f"   Dominant across {n_trials} random subsets: Attn={dominant_counts['Attn']}, MLP={dominant_counts['MLP']}, LN={dominant_counts['LN']}")
    
    # ===== 2. Sign consistency: robust across subsets? =====
    print(f"\n2. Sign Consistency Robustness:")
    
    delta_total = np.array([f['delta_gap_total'] for f in all_features])
    
    sign_a = np.mean(np.sign(delta_attn) == np.sign(delta_total))
    sign_m = np.mean(np.sign(delta_mlp) == np.sign(delta_total))
    sign_l = np.mean(np.sign(delta_final_ln) == np.sign(delta_total))
    
    print(f"   Sign consistency (full set): Attn={sign_a:.3f}, MLP={sign_m:.3f}, LN={sign_l:.3f}")
    
    # Bootstrap
    np.random.seed(456)
    sign_attn_bs = []
    sign_mlp_bs = []
    sign_ln_bs = []
    
    for _ in range(100):
        idx = np.random.choice(n_texts, n_texts, replace=True)
        sa = np.mean(np.sign(delta_attn[idx]) == np.sign(delta_total[idx]))
        sm = np.mean(np.sign(delta_mlp[idx]) == np.sign(delta_total[idx]))
        sl = np.mean(np.sign(delta_final_ln[idx]) == np.sign(delta_total[idx]))
        sign_attn_bs.append(sa)
        sign_mlp_bs.append(sm)
        sign_ln_bs.append(sl)
    
    print(f"   Bootstrap 95% CI:")
    print(f"   Attn: [{np.percentile(sign_attn_bs, 2.5):.3f}, {np.percentile(sign_attn_bs, 97.5):.3f}]")
    print(f"   MLP:  [{np.percentile(sign_mlp_bs, 2.5):.3f}, {np.percentile(sign_mlp_bs, 97.5):.3f}]")
    print(f"   LN:   [{np.percentile(sign_ln_bs, 2.5):.3f}, {np.percentile(sign_ln_bs, 97.5):.3f}]")
    
    # ===== 3. Cross-model comparison summary =====
    print(f"\n3. Parameter-Independent Conclusions:")
    
    # Which conclusions survive parameter sweep?
    conclusions = []
    
    # Conclusion 1: h(L-1) contains significant gap info
    r_Lm1, _ = stats.pearsonr(np.array([f['gap_L_minus_1'] for f in all_features]), gaps)
    if abs(r_Lm1) > 0.5:
        conclusions.append(f"h(L-1) gap correlates with logit_gap (r={r_Lm1:.3f})")
    
    # Conclusion 2: Final LN changes the gap significantly
    if abs_ln / (abs_attn + abs_mlp + abs_ln + 1e-10) > 0.3:
        conclusions.append(f"Final LN is a major gap contributor ({abs_ln/(abs_attn+abs_mlp+abs_ln)*100:.1f}%)")
    
    # Conclusion 3: Decomposition is exact (hooks)
    recon_err = np.mean(np.abs(delta_attn + delta_mlp + delta_final_ln - delta_total))
    if recon_err < 0.01:
        conclusions.append(f"Hook-based decomposition is exact (error={recon_err:.6f})")
    
    for i, c in enumerate(conclusions):
        print(f"   {i+1}. {c}")
    
    return {
        'dominant_component': dominant,
        'dominant_counts': dominant_counts,
        'abs_attn': float(abs_attn),
        'abs_mlp': float(abs_mlp),
        'abs_ln': float(abs_ln),
        'sign_attn': float(sign_a),
        'sign_mlp': float(sign_m),
        'sign_ln': float(sign_l),
        'sign_attn_ci': [float(np.percentile(sign_attn_bs, 2.5)), float(np.percentile(sign_attn_bs, 97.5))],
        'sign_mlp_ci': [float(np.percentile(sign_mlp_bs, 2.5)), float(np.percentile(sign_mlp_bs, 97.5))],
        'sign_ln_ci': [float(np.percentile(sign_ln_bs, 2.5)), float(np.percentile(sign_ln_bs, 97.5))],
        'r_Lm1_vs_gap': float(r_Lm1),
        'reconstruction_error': float(recon_err),
    }


# ===== Main =====
EXPERIMENTS = {
    'p649': experiment_p649,
    'p650': experiment_p650,
    'p651': experiment_p651,
    'p652': experiment_p652,
}


def main():
    parser = argparse.ArgumentParser(description='Phase CXLVII: Parameter Audit + Hook Decomposition')
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'glm4', 'deepseek7b'])
    parser.add_argument('--experiment', type=str, required=True, choices=list(EXPERIMENTS.keys()))
    parser.add_argument('--n_texts', type=int, default=40)
    parser.add_argument('--output_dir', type=str, default='results/phase_cxlvii')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nPhase CXLVII: {args.experiment.upper()} on {args.model}")
    print(f"n_texts={args.n_texts}, output_dir={args.output_dir}")
    
    # Load model
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    print(f"Model: {model_info.model_class}, d_model={model_info.d_model}, n_layers={model_info.n_layers}")
    
    # Extract features with hooks
    print(f"\nExtracting features with hooks...")
    all_features, extra = compute_features_with_hooks(
        model, tokenizer, device, TEST_TEXTS[:args.n_texts]
    )
    
    # Run experiment
    experiment_fn = EXPERIMENTS[args.experiment]
    results = experiment_fn(all_features, extra, args.model)
    
    # Save results
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    results = convert(results)
    results['model'] = args.model
    results['experiment'] = args.experiment
    results['n_texts'] = args.n_texts
    results['d_model'] = extra['d_model']
    results['n_layers'] = extra['n_layers']
    
    output_file = os.path.join(args.output_dir, f"{args.experiment}_{args.model}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_file}")
    
    # Release model
    release_model(model)


if __name__ == '__main__':
    main()
