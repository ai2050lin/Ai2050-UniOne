#!/usr/bin/env python3
"""
Phase CXLVI: Attention-MLP-LN Synergistic Emergence (P645-P648)
Focus: Decompose the last layer's attention, MLP, and LN contributions to gap emergence.

Critical finding from Phase CXLV:
- Linear h(L-1)->h(L) cos=1.0, gap r>0.994
- But h(L-1)+simple LN->gap only r=0.23-0.76
- LN efficiency only 22.8-62.0%
- Emergence enhancement 107-141%

This means: The last layer's attention+MLP+LN together create the emergence.
We need to decompose the exact contribution of each component.

P645: Last-layer attention precise effect - how Q/K/V weights select Delta_W-relevant info
P646: Last-layer MLP precise effect - how gate/up/down transform representations
P647: Attention-MLP-LN synergistic emergence mathematical model
P648: Quantitative validation - predicted precision should >90%
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


def detect_head_structure(W_q, W_v, d_model):
    """Detect head structure from weight matrices, handling GQA."""
    head_dim = d_model // 8  # Default fallback
    for hd in [64, 80, 96, 128, 256]:
        if W_q.shape[0] % hd == 0 and W_v.shape[0] % hd == 0:
            head_dim = hd
            break
    
    n_q_heads = W_q.shape[0] // head_dim
    n_kv_heads = W_v.shape[0] // head_dim
    kv_group_size = n_q_heads // n_kv_heads if n_kv_heads > 0 else 1
    
    return head_dim, n_q_heads, n_kv_heads, kv_group_size


def compute_attn_output_gqa(h_in, W_v, W_o, n_kv_heads, n_q_heads, head_dim, kv_group_size):
    """Compute attention output with GQA expansion: W_o @ expand(W_v @ h_in)."""
    v_out = W_v @ h_in  # [n_kv_heads * head_dim]
    if n_q_heads != n_kv_heads and kv_group_size > 1:
        # Expand KV heads to match Q heads
        v_reshaped = v_out.reshape(n_kv_heads, head_dim)
        v_expanded = np.repeat(v_reshaped, kv_group_size, axis=0).flatten()
    else:
        v_expanded = v_out
    return W_o @ v_expanded  # [d_model]


def apply_layernorm(h, ln_weight, ln_bias):
    """Apply LayerNorm transformation."""
    h_mean = np.mean(h)
    h_std = np.std(h)
    if h_std > 1e-10:
        return (h - h_mean) / h_std * ln_weight + ln_bias
    return h.copy()


def compute_last_layer_features(model, tokenizer, device, texts):
    """Extract detailed last-layer decomposition features."""
    # Get W_U
    if hasattr(model, 'lm_head'):
        W_U = model.lm_head.weight.detach().cpu().float().numpy()
    else:
        W_U = model.get_output_embeddings().weight.detach().cpu().float().numpy()
    
    d_model = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 2560
    n_layers = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else 32
    
    # Get final LN weights
    has_final_ln = False
    final_ln_weight = None
    final_ln_bias = None
    
    if hasattr(model.model, 'norm'):
        has_final_ln = True
        final_ln_weight = model.model.norm.weight.detach().cpu().float().numpy()
        final_ln_bias = model.model.norm.bias.detach().cpu().float().numpy() if hasattr(model.model.norm, 'bias') and model.model.norm.bias is not None else np.zeros(d_model)
    elif hasattr(model.model, 'final_layernorm'):
        has_final_ln = True
        final_ln_weight = model.model.final_layernorm.weight.detach().cpu().float().numpy()
        final_ln_bias = model.model.final_layernorm.bias.detach().cpu().float().numpy() if hasattr(model.model.final_layernorm, 'bias') and model.model.final_layernorm.bias is not None else np.zeros(d_model)
    
    # Get last layer weights
    layers = model.model.layers if hasattr(model.model, 'layers') else []
    last_layer = layers[-1]
    
    # Determine mlp_type from model structure
    mlp_type = "split_gate_up"
    if hasattr(last_layer.mlp, 'gate_up_proj'):
        mlp_type = "merged_gate_up"
    
    last_layer_weights = get_layer_weights(last_layer, d_model, mlp_type)
    
    # Get LN weights
    post_attn_ln_weight = None
    for ln_name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
        if hasattr(last_layer, ln_name):
            ln = getattr(last_layer, ln_name)
            if hasattr(ln, 'weight'):
                post_attn_ln_weight = ln.weight.detach().cpu().float().numpy()
            break
    
    input_ln_weight = None
    for ln_name in ["input_layernorm", "ln_1", "layernorm"]:
        if hasattr(last_layer, ln_name):
            ln = getattr(last_layer, ln_name)
            if hasattr(ln, 'weight'):
                input_ln_weight = ln.weight.detach().cpu().float().numpy()
            break
    
    # Detect head structure
    head_dim, n_q_heads, n_kv_heads, kv_group_size = detect_head_structure(
        last_layer_weights.W_q, last_layer_weights.W_v, d_model
    )
    
    print(f"  Model: d_model={d_model}, n_layers={n_layers}")
    print(f"  Has final LN: {has_final_ln}")
    print(f"  Last layer mlp_type: {mlp_type}")
    print(f"  Head structure: n_q={n_q_heads}, n_kv={n_kv_heads}, head_dim={head_dim}, gqa_group={kv_group_size}")
    
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
        
        h_L = all_hidden[-1]
        h_L_minus_1 = all_hidden[-2] if len(all_hidden) > 1 else all_hidden[0]
        
        logits = outputs.logits[0, -1, :].cpu().float().numpy()
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        logit_gap = logits[top1_idx] - logits[top2_idx]
        
        W_U_top1 = W_U[top1_idx]
        W_U_top2 = W_U[top2_idx]
        Delta_W = W_U_top1 - W_U_top2
        
        # Gap at each layer
        gaps_at_layers = [np.dot(all_hidden[l], Delta_W) for l in range(len(all_hidden))]
        
        gap_input = np.dot(h_L_minus_1, Delta_W)
        gap_output = np.dot(h_L, Delta_W)
        delta_gap = gap_output - gap_input
        
        all_features.append({
            'h_L': h_L,
            'h_L_minus_1': h_L_minus_1,
            'all_hidden': all_hidden,
            'logit_gap': logit_gap,
            'top1_idx': int(top1_idx),
            'top2_idx': int(top2_idx),
            'Delta_W': Delta_W,
            'gap_input': gap_input,
            'gap_output': gap_output,
            'delta_gap': delta_gap,
            'gaps_at_layers': gaps_at_layers,
            'text_idx': i,
        })
    
    extra = {
        'W_U': W_U,
        'd_model': d_model,
        'n_layers': n_layers,
        'has_final_ln': has_final_ln,
        'final_ln_weight': final_ln_weight,
        'final_ln_bias': final_ln_bias,
        'last_layer_weights': last_layer_weights,
        'post_attn_ln_weight': post_attn_ln_weight,
        'input_ln_weight': input_ln_weight,
        'mlp_type': mlp_type,
        'head_dim': head_dim,
        'n_q_heads': n_q_heads,
        'n_kv_heads': n_kv_heads,
        'kv_group_size': kv_group_size,
    }
    
    return all_features, extra


def experiment_p645(all_features, extra, model_name):
    """P645: Last-Layer Attention Precise Effect."""
    print(f"\n{'='*60}")
    print(f"P645: Last-Layer Attention Precise Effect ({model_name})")
    print(f"{'='*60}")
    
    d_model = extra['d_model']
    last_w = extra['last_layer_weights']
    head_dim = extra['head_dim']
    n_q_heads = extra['n_q_heads']
    n_kv_heads = extra['n_kv_heads']
    kv_group_size = extra['kv_group_size']
    
    W_o = last_w.W_o
    W_q = last_w.W_q
    W_k = last_w.W_k
    W_v = last_w.W_v
    
    n_texts = len(all_features)
    gaps = np.array([f['logit_gap'] for f in all_features])
    
    # 1. Attention output alignment with Delta_W
    print(f"\n1. Attention Output Alignment with Delta_W:")
    
    cos_attn_dw_list = []
    norm_attn_dw_list = []
    
    for f in all_features:
        dw = f['Delta_W']
        h_in = f['h_L_minus_1']
        attn_out = compute_attn_output_gqa(h_in, W_v, W_o, n_kv_heads, n_q_heads, head_dim, kv_group_size)
        cos_val = np.dot(attn_out, dw) / (np.linalg.norm(attn_out) * np.linalg.norm(dw) + 1e-10)
        norm_val = np.linalg.norm(attn_out) / (np.linalg.norm(dw) + 1e-10)
        cos_attn_dw_list.append(cos_val)
        norm_attn_dw_list.append(norm_val)
    
    print(f"   cos(Attn(h), Delta_W): mean={np.mean(cos_attn_dw_list):.4f}, std={np.std(cos_attn_dw_list):.4f}")
    print(f"   |Attn(h)| / |Delta_W|: mean={np.mean(norm_attn_dw_list):.4f}")
    
    # 2. Attention gap contribution
    print(f"\n2. Attention Gap Contribution:")
    
    attn_gap_preds = []
    for f in all_features:
        h_in = f['h_L_minus_1']
        dw = f['Delta_W']
        attn_out = compute_attn_output_gqa(h_in, W_v, W_o, n_kv_heads, n_q_heads, head_dim, kv_group_size)
        attn_gap_preds.append(np.dot(attn_out, dw))
    
    attn_gap_preds = np.array(attn_gap_preds)
    delta_gaps = np.array([f['delta_gap'] for f in all_features])
    
    r_attn_delta, _ = stats.pearsonr(attn_gap_preds, delta_gaps)
    r_attn_gap, _ = stats.pearsonr(attn_gap_preds, gaps)
    
    print(f"   r(Attn_gap, delta_gap): {r_attn_delta:.4f}")
    print(f"   r(Attn_gap, final_gap): {r_attn_gap:.4f}")
    print(f"   Attn_gap: mean={np.mean(attn_gap_preds):.4f}, std={np.std(attn_gap_preds):.4f}")
    
    # 3. Relative Attention vs MLP gap potential
    print(f"\n3. Relative Attention vs MLP Gap Potential:")
    
    W_down = last_w.W_down
    
    attn_potentials = []
    mlp_potentials = []
    
    for f in all_features:
        dw = f['Delta_W']
        attn_out = compute_attn_output_gqa(dw, W_v, W_o, n_kv_heads, n_q_heads, head_dim, kv_group_size)
        attn_pot = np.linalg.norm(attn_out)
        
        Wd_T_dw = W_down.T @ dw
        mlp_pot = np.linalg.norm(Wd_T_dw)
        
        attn_potentials.append(attn_pot)
        mlp_potentials.append(mlp_pot)
    
    print(f"   Attn gap potential: mean={np.mean(attn_potentials):.4f}")
    print(f"   MLP gap potential: mean={np.mean(mlp_potentials):.4f}")
    print(f"   Attn/MLP ratio: {np.mean(attn_potentials)/np.mean(mlp_potentials):.4f}")
    
    # 4. Head-level analysis
    print(f"\n4. Head-Level Analysis:")
    
    n_attn_dim = W_o.shape[1]
    n_eff_heads = n_attn_dim // head_dim
    
    head_gap_alignment = []
    for h_idx in range(n_eff_heads):
        W_o_h = W_o[:, h_idx*head_dim:(h_idx+1)*head_dim]
        alignments = []
        for f in all_features:
            dw = f['Delta_W']
            dw_Wo_h = dw @ W_o_h
            alignment = np.linalg.norm(dw_Wo_h) / (np.linalg.norm(dw) + 1e-10)
            alignments.append(alignment)
        head_gap_alignment.append(np.mean(alignments))
    
    sorted_heads = np.argsort(head_gap_alignment)[::-1]
    print(f"   Number of heads: {n_eff_heads}")
    print(f"   Top-5 heads: {sorted_heads[:5].tolist()}, values: {[f'{head_gap_alignment[h]:.4f}' for h in sorted_heads[:5]]}")
    print(f"   Bottom-5 heads: {sorted_heads[-5:].tolist()}, values: {[f'{head_gap_alignment[h]:.4f}' for h in sorted_heads[-5:]]}")
    
    # 5. Sign consistency of attention contribution
    print(f"\n5. Attention Sign Consistency:")
    sign_consistent = np.mean(np.sign(attn_gap_preds) == np.sign(delta_gaps))
    print(f"   sign(Attn_gap) == sign(delta_gap): {sign_consistent:.4f}")
    
    return {
        'cos_attn_dw_mean': float(np.mean(cos_attn_dw_list)),
        'r_attn_delta': float(r_attn_delta),
        'r_attn_gap': float(r_attn_gap),
        'attn_mlp_ratio': float(np.mean(attn_potentials) / np.mean(mlp_potentials)),
        'n_heads': int(n_eff_heads),
        'sign_consistent': float(sign_consistent),
        'head_gap_alignment': [float(x) for x in head_gap_alignment],
    }


def experiment_p646(all_features, extra, model_name):
    """P646: Last-Layer MLP Precise Effect."""
    print(f"\n{'='*60}")
    print(f"P646: Last-Layer MLP Precise Effect ({model_name})")
    print(f"{'='*60}")
    
    d_model = extra['d_model']
    last_w = extra['last_layer_weights']
    head_dim = extra['head_dim']
    n_q_heads = extra['n_q_heads']
    n_kv_heads = extra['n_kv_heads']
    kv_group_size = extra['kv_group_size']
    
    n_texts = len(all_features)
    gaps = np.array([f['logit_gap'] for f in all_features])
    
    W_o = last_w.W_o
    W_v = last_w.W_v
    W_down = last_w.W_down
    W_up = last_w.W_up
    W_gate = last_w.W_gate
    
    if W_up is None or W_gate is None:
        print("   WARNING: MLP weights not fully available")
        return {}
    
    intermediate_size = W_down.shape[1]
    
    # 1. MLP gap sensitivity distribution
    print(f"\n1. MLP Gap Sensitivity (W_down^T . Delta_W):")
    
    top_neuron_fracs = []
    for f in all_features:
        dw = f['Delta_W']
        sensitivity = W_down.T @ dw
        abs_sens = np.abs(sensitivity)
        total = np.sum(abs_sens)
        if total > 0:
            sorted_sens = np.sort(abs_sens)[::-1]
            cum_frac = np.cumsum(sorted_sens) / total
            top_neuron_fracs.append({
                'n_50pct': np.searchsorted(cum_frac, 0.5) + 1,
                'n_90pct': np.searchsorted(cum_frac, 0.9) + 1,
                'n_99pct': np.searchsorted(cum_frac, 0.99) + 1,
            })
    
    if top_neuron_fracs:
        mean_fracs = {k: np.mean([f[k] for f in top_neuron_fracs]) for k in top_neuron_fracs[0]}
        print(f"   Neurons for 50%: {mean_fracs['n_50pct']:.0f}/{intermediate_size} ({mean_fracs['n_50pct']/intermediate_size*100:.1f}%)")
        print(f"   Neurons for 90%: {mean_fracs['n_90pct']:.0f}/{intermediate_size} ({mean_fracs['n_90pct']/intermediate_size*100:.1f}%)")
        print(f"   Neurons for 99%: {mean_fracs['n_99pct']:.0f}/{intermediate_size} ({mean_fracs['n_99pct']/intermediate_size*100:.1f}%)")
    
    # 2. Simulated MLP gap contribution
    print(f"\n2. Simulated MLP Gap Contribution:")
    
    mlp_gap_preds = []
    for f in all_features:
        h_in = f['h_L_minus_1']
        dw = f['Delta_W']
        
        # Simplified MLP (ignoring LN2)
        gate_out = W_gate @ h_in
        up_out = W_up @ h_in
        activated = gate_out * (1.0 / (1.0 + np.exp(-gate_out)))  # SiLU
        mlp_out = W_down @ (activated * up_out)
        mlp_gap_preds.append(np.dot(mlp_out, dw))
    
    mlp_gap_preds = np.array(mlp_gap_preds)
    delta_gaps = np.array([f['delta_gap'] for f in all_features])
    
    r_mlp_delta, _ = stats.pearsonr(mlp_gap_preds, delta_gaps)
    r_mlp_gap, _ = stats.pearsonr(mlp_gap_preds, gaps)
    
    print(f"   r(MLP_gap, delta_gap): {r_mlp_delta:.4f}")
    print(f"   r(MLP_gap, final_gap): {r_mlp_gap:.4f}")
    print(f"   MLP_gap: mean={np.mean(mlp_gap_preds):.4f}, actual delta: mean={np.mean(delta_gaps):.4f}")
    
    # 3. Combined Attn + MLP prediction
    print(f"\n3. Combined Attn + MLP Prediction:")
    
    attn_gap_preds = []
    for f in all_features:
        h_in = f['h_L_minus_1']
        dw = f['Delta_W']
        attn_out = compute_attn_output_gqa(h_in, W_v, W_o, n_kv_heads, n_q_heads, head_dim, kv_group_size)
        attn_gap_preds.append(np.dot(attn_out, dw))
    
    attn_gap_preds = np.array(attn_gap_preds)
    combined_preds = attn_gap_preds + mlp_gap_preds
    
    r_combined, _ = stats.pearsonr(combined_preds, delta_gaps)
    r_combined_gap, _ = stats.pearsonr(combined_preds, gaps)
    
    print(f"   r(Attn+MLP, delta_gap): {r_combined:.4f}")
    print(f"   r(Attn+MLP, final_gap): {r_combined_gap:.4f}")
    
    # 4. Sign consistency
    print(f"\n4. Sign Consistency:")
    sign_mlp = np.mean(np.sign(mlp_gap_preds) == np.sign(delta_gaps))
    sign_attn = np.mean(np.sign(attn_gap_preds) == np.sign(delta_gaps))
    print(f"   sign(MLP) == sign(delta_gap): {sign_mlp:.4f}")
    print(f"   sign(Attn) == sign(delta_gap): {sign_attn:.4f}")
    
    # 5. LN2 effect
    print(f"\n5. Post-Attention LayerNorm Effect:")
    post_attn_ln = extra.get('post_attn_ln_weight')
    if post_attn_ln is not None:
        cos_ln2_dw = []
        for f in all_features:
            dw = f['Delta_W']
            cos_val = np.dot(post_attn_ln, dw) / (np.linalg.norm(post_attn_ln) * np.linalg.norm(dw) + 1e-10)
            cos_ln2_dw.append(cos_val)
        print(f"   cos(LN2_weight, Delta_W): mean={np.mean(cos_ln2_dw):.4f}")
    else:
        print(f"   No post-attention LN weight found")
    
    return {
        'intermediate_size': int(intermediate_size),
        'neurons_50pct': float(np.mean([f['n_50pct'] for f in top_neuron_fracs])) if top_neuron_fracs else 0,
        'neurons_90pct': float(np.mean([f['n_90pct'] for f in top_neuron_fracs])) if top_neuron_fracs else 0,
        'r_mlp_delta': float(r_mlp_delta),
        'r_mlp_gap': float(r_mlp_gap),
        'r_combined_delta': float(r_combined),
        'r_combined_gap': float(r_combined_gap),
        'sign_mlp': float(sign_mlp),
        'sign_attn': float(sign_attn),
    }


def experiment_p647(all_features, extra, model_name):
    """P647: Attention-MLP-LN Synergistic Emergence Mathematical Model."""
    print(f"\n{'='*60}")
    print(f"P647: Attention-MLP-LN Synergy Mathematical Model ({model_name})")
    print(f"{'='*60}")
    
    d_model = extra['d_model']
    has_final_ln = extra['has_final_ln']
    final_ln_weight = extra['final_ln_weight']
    final_ln_bias = extra['final_ln_bias']
    last_w = extra['last_layer_weights']
    post_attn_ln = extra.get('post_attn_ln_weight')
    input_ln = extra.get('input_ln_weight')
    head_dim = extra['head_dim']
    n_q_heads = extra['n_q_heads']
    n_kv_heads = extra['n_kv_heads']
    kv_group_size = extra['kv_group_size']
    
    n_texts = len(all_features)
    gaps = np.array([f['logit_gap'] for f in all_features])
    
    W_o = last_w.W_o
    W_v = last_w.W_v
    W_down = last_w.W_down
    W_up = last_w.W_up
    W_gate = last_w.W_gate
    
    # 1. Full simulated last-layer transformation
    print(f"\n1. Simulated Last-Layer Transformation:")
    
    sim_results = {
        'gap_input': [],
        'gap_after_attn': [],
        'gap_after_mlp': [],
        'gap_after_ln': [],
        'attn_contribution': [],
        'mlp_contribution': [],
        'ln_contribution': [],
    }
    
    for f in all_features:
        h_in = f['h_L_minus_1']
        dw = f['Delta_W']
        
        # Step 1: Input LN + Attention
        if input_ln is not None:
            h_ln1 = apply_layernorm(h_in, input_ln, np.zeros(d_model))
        else:
            h_ln1 = h_in
        
        attn_out = compute_attn_output_gqa(h_ln1, W_v, W_o, n_kv_heads, n_q_heads, head_dim, kv_group_size)
        h_attn = h_in + attn_out  # Residual connection
        
        gap_after_attn = np.dot(h_attn, dw)
        attn_contribution = gap_after_attn - np.dot(h_in, dw)
        
        # Step 2: Post-Attn LN + MLP
        if post_attn_ln is not None:
            h_ln2 = apply_layernorm(h_attn, post_attn_ln, np.zeros(d_model))
        else:
            h_ln2 = h_attn
        
        gate_out = W_gate @ h_ln2
        up_out = W_up @ h_ln2
        activated = gate_out * (1.0 / (1.0 + np.exp(-gate_out)))  # SiLU
        mlp_out = W_down @ (activated * up_out)
        h_mlp = h_attn + mlp_out  # Residual connection
        
        gap_after_mlp = np.dot(h_mlp, dw)
        mlp_contribution = gap_after_mlp - gap_after_attn
        
        # Step 3: Final LN
        if has_final_ln:
            h_ln = apply_layernorm(h_mlp, final_ln_weight, final_ln_bias)
        else:
            h_ln = h_mlp
        
        gap_after_ln = np.dot(h_ln, dw)
        ln_contribution = gap_after_ln - gap_after_mlp
        
        sim_results['gap_input'].append(np.dot(h_in, dw))
        sim_results['gap_after_attn'].append(gap_after_attn)
        sim_results['gap_after_mlp'].append(gap_after_mlp)
        sim_results['gap_after_ln'].append(gap_after_ln)
        sim_results['attn_contribution'].append(attn_contribution)
        sim_results['mlp_contribution'].append(mlp_contribution)
        sim_results['ln_contribution'].append(ln_contribution)
    
    for k in sim_results:
        sim_results[k] = np.array(sim_results[k])
    
    print(f"   Mean gap at each stage:")
    print(f"     Input (L-1):    {np.mean(sim_results['gap_input']):.4f}")
    print(f"     After Attn:     {np.mean(sim_results['gap_after_attn']):.4f}")
    print(f"     After MLP:      {np.mean(sim_results['gap_after_mlp']):.4f}")
    print(f"     After LN:       {np.mean(sim_results['gap_after_ln']):.4f}")
    print(f"   Mean contributions:")
    print(f"     Attn:  {np.mean(sim_results['attn_contribution']):.4f}")
    print(f"     MLP:   {np.mean(sim_results['mlp_contribution']):.4f}")
    print(f"     LN:    {np.mean(sim_results['ln_contribution']):.4f}")
    
    r_sim, _ = stats.pearsonr(sim_results['gap_after_ln'], gaps)
    print(f"   r(simulated_gap, actual_gap): {r_sim:.4f}")
    
    # 2. Component-wise correlation
    print(f"\n2. Component Correlations with Final Gap:")
    
    r_input, _ = stats.pearsonr(sim_results['gap_input'], gaps)
    r_attn, _ = stats.pearsonr(sim_results['attn_contribution'], gaps)
    r_mlp, _ = stats.pearsonr(sim_results['mlp_contribution'], gaps)
    r_ln, _ = stats.pearsonr(sim_results['ln_contribution'], gaps)
    
    print(f"   r(gap_input, final_gap): {r_input:.4f}")
    print(f"   r(attn_contrib, final_gap): {r_attn:.4f}")
    print(f"   r(mlp_contrib, final_gap): {r_mlp:.4f}")
    print(f"   r(ln_contrib, final_gap): {r_ln:.4f}")
    
    # 3. Full linear model
    print(f"\n3. Full Linear Model:")
    
    X = np.column_stack([
        sim_results['gap_input'],
        sim_results['attn_contribution'],
        sim_results['mlp_contribution'],
        sim_results['ln_contribution'],
    ])
    
    reg = Ridge(alpha=1.0).fit(X, gaps)
    y_pred = reg.predict(X)
    r_full, _ = stats.pearsonr(y_pred, gaps)
    
    print(f"   Coefficients: input={reg.coef_[0]:.4f}, attn={reg.coef_[1]:.4f}, mlp={reg.coef_[2]:.4f}, ln={reg.coef_[3]:.4f}")
    print(f"   r(full_model, actual_gap): {r_full:.4f}")
    
    # 4. Synergy analysis
    print(f"\n4. Synergy Analysis:")
    
    r_attn_only, _ = stats.pearsonr(sim_results['gap_input'] + sim_results['attn_contribution'], gaps)
    r_mlp_only, _ = stats.pearsonr(sim_results['gap_input'] + sim_results['mlp_contribution'], gaps)
    r_ln_only, _ = stats.pearsonr(sim_results['gap_input'] + sim_results['ln_contribution'], gaps)
    r_attn_mlp, _ = stats.pearsonr(sim_results['gap_after_mlp'], gaps)
    r_all, _ = stats.pearsonr(sim_results['gap_after_ln'], gaps)
    
    print(f"   r(input+attn): {r_attn_only:.4f}")
    print(f"   r(input+mlp): {r_mlp_only:.4f}")
    print(f"   r(input+ln): {r_ln_only:.4f}")
    print(f"   r(input+attn+mlp): {r_attn_mlp:.4f}")
    print(f"   r(full_simulation): {r_all:.4f}")
    
    # 5. Refined: LN(h(L-1)) only
    print(f"\n5. LN(h(L-1)) Only:")
    
    refined_gaps = []
    for f in all_features:
        h_in = f['h_L_minus_1']
        dw = f['Delta_W']
        if has_final_ln:
            h_ln = apply_layernorm(h_in, final_ln_weight, final_ln_bias)
        else:
            h_ln = h_in
        refined_gaps.append(np.dot(h_ln, dw))
    
    refined_gaps = np.array(refined_gaps)
    r_refined, _ = stats.pearsonr(refined_gaps, gaps)
    print(f"   r(LN(h(L-1)), gap): {r_refined:.4f}")
    print(f"   Comparison: LN(h(L-1))={r_refined:.4f}, Full sim={r_all:.4f}, Oracle=1.0")
    
    return {
        'r_sim': float(r_sim),
        'r_full_model': float(r_full),
        'r_refined': float(r_refined),
        'r_attn_only': float(r_attn_only),
        'r_mlp_only': float(r_mlp_only),
        'r_ln_only': float(r_ln_only),
        'r_attn_mlp': float(r_attn_mlp),
        'r_all': float(r_all),
        'mean_attn_contribution': float(np.mean(sim_results['attn_contribution'])),
        'mean_mlp_contribution': float(np.mean(sim_results['mlp_contribution'])),
        'mean_ln_contribution': float(np.mean(sim_results['ln_contribution'])),
    }


def experiment_p648(all_features, extra, model_name):
    """P648: Quantitative Validation of Synergistic Framework."""
    print(f"\n{'='*60}")
    print(f"P648: Quantitative Validation ({model_name})")
    print(f"{'='*60}")
    
    d_model = extra['d_model']
    has_final_ln = extra['has_final_ln']
    final_ln_weight = extra['final_ln_weight']
    final_ln_bias = extra['final_ln_bias']
    last_w = extra['last_layer_weights']
    post_attn_ln = extra.get('post_attn_ln_weight')
    input_ln = extra.get('input_ln_weight')
    head_dim = extra['head_dim']
    n_q_heads = extra['n_q_heads']
    n_kv_heads = extra['n_kv_heads']
    kv_group_size = extra['kv_group_size']
    
    n_texts = len(all_features)
    gaps = np.array([f['logit_gap'] for f in all_features])
    probs = np.array([1.0 / (1.0 + np.exp(-g)) for g in gaps])
    
    W_o = last_w.W_o
    W_v = last_w.W_v
    W_down = last_w.W_down
    W_up = last_w.W_up
    W_gate = last_w.W_gate
    
    # Framework A: LN(h(L-1)) only
    gaps_ln_only = []
    for f in all_features:
        h_in = f['h_L_minus_1']
        dw = f['Delta_W']
        if has_final_ln:
            h_ln = apply_layernorm(h_in, final_ln_weight, final_ln_bias)
        else:
            h_ln = h_in
        gaps_ln_only.append(np.dot(h_ln, dw))
    gaps_ln_only = np.array(gaps_ln_only)
    
    # Framework B: Full simulation
    gaps_full_sim = []
    for f in all_features:
        h_in = f['h_L_minus_1']
        dw = f['Delta_W']
        
        # Input LN + Attention
        if input_ln is not None:
            h_ln1 = apply_layernorm(h_in, input_ln, np.zeros(d_model))
        else:
            h_ln1 = h_in
        
        attn_out = compute_attn_output_gqa(h_ln1, W_v, W_o, n_kv_heads, n_q_heads, head_dim, kv_group_size)
        h_attn = h_in + attn_out
        
        # Post-Attn LN + MLP
        if post_attn_ln is not None:
            h_ln2 = apply_layernorm(h_attn, post_attn_ln, np.zeros(d_model))
        else:
            h_ln2 = h_attn
        
        gate_out = W_gate @ h_ln2
        up_out = W_up @ h_ln2
        activated = gate_out * (1.0 / (1.0 + np.exp(-gate_out)))
        mlp_out = W_down @ (activated * up_out)
        h_mlp = h_attn + mlp_out
        
        # Final LN
        if has_final_ln:
            h_ln = apply_layernorm(h_mlp, final_ln_weight, final_ln_bias)
        else:
            h_ln = h_mlp
        
        gaps_full_sim.append(np.dot(h_ln, dw))
    gaps_full_sim = np.array(gaps_full_sim)
    
    # Framework C: Oracle
    gaps_oracle = np.array([np.dot(f['h_L'], f['Delta_W']) for f in all_features])
    
    # Framework D: h(L-1) direct
    gaps_h_input = np.array([f['gap_input'] for f in all_features])
    
    # Framework E: Ridge from h(L-1)
    h_L_minus_1 = np.array([f['h_L_minus_1'] for f in all_features])
    pca = PCA(n_components=min(30, n_texts - 1))
    h_pca = pca.fit_transform(h_L_minus_1)
    ridge = Ridge(alpha=1.0).fit(h_pca, gaps)
    gaps_ridge = ridge.predict(h_pca)
    
    # Results
    r_ln_only, _ = stats.pearsonr(gaps_ln_only, gaps)
    r_full_sim, _ = stats.pearsonr(gaps_full_sim, gaps)
    r_oracle, _ = stats.pearsonr(gaps_oracle, gaps)
    r_h_input, _ = stats.pearsonr(gaps_h_input, gaps)
    r_ridge, _ = stats.pearsonr(gaps_ridge, gaps)
    
    print(f"\n1. Framework Comparison:")
    print(f"   A: LN(h(L-1)) only:      r={r_ln_only:.4f}")
    print(f"   B: Full simulation:       r={r_full_sim:.4f}")
    print(f"   C: Oracle h(L).Delta_W:   r={r_oracle:.6f}")
    print(f"   D: h(L-1) direct:         r={r_h_input:.4f}")
    print(f"   E: Ridge from h(L-1):     r={r_ridge:.4f}")
    
    # 2. Probability prediction
    probs_ln = 1.0 / (1.0 + np.exp(-gaps_ln_only))
    probs_sim = 1.0 / (1.0 + np.exp(-gaps_full_sim))
    probs_oracle = 1.0 / (1.0 + np.exp(-gaps_oracle))
    probs_input = 1.0 / (1.0 + np.exp(-gaps_h_input))
    
    r_prob_ln, _ = stats.pearsonr(probs_ln, probs)
    r_prob_sim, _ = stats.pearsonr(probs_sim, probs)
    r_prob_oracle, _ = stats.pearsonr(probs_oracle, probs)
    r_prob_input, _ = stats.pearsonr(probs_input, probs)
    
    print(f"\n2. Probability Prediction:")
    print(f"   LN(h(L-1)):  r={r_prob_ln:.4f}")
    print(f"   Full sim:    r={r_prob_sim:.4f}")
    print(f"   Oracle:      r={r_prob_oracle:.6f}")
    print(f"   h(L-1):      r={r_prob_input:.4f}")
    
    # 3. Sign prediction
    sign_actual = np.sign(gaps)
    acc_ln = np.mean(np.sign(gaps_ln_only) == sign_actual)
    acc_sim = np.mean(np.sign(gaps_full_sim) == sign_actual)
    acc_input = np.mean(np.sign(gaps_h_input) == sign_actual)
    
    print(f"\n3. Sign Prediction Accuracy:")
    print(f"   LN(h(L-1)):  {acc_ln:.4f}")
    print(f"   Full sim:    {acc_sim:.4f}")
    print(f"   h(L-1):      {acc_input:.4f}")
    
    # 4. LOO-CV for Ridge
    loo_preds_ridge = []
    for leave_out in range(n_texts):
        train_idx = [j for j in range(n_texts) if j != leave_out]
        ridge_cv = Ridge(alpha=1.0).fit(h_pca[train_idx], gaps[train_idx])
        loo_preds_ridge.append(ridge_cv.predict(h_pca[leave_out:leave_out+1])[0])
    
    loo_preds_ridge = np.array(loo_preds_ridge)
    r_loo_ridge, _ = stats.pearsonr(loo_preds_ridge, gaps)
    mse_ridge = np.mean((loo_preds_ridge - gaps)**2)
    
    print(f"\n4. Cross-Validation:")
    print(f"   Full sim (no CV needed): r={r_full_sim:.4f}")
    print(f"   Ridge LOO-CV: r={r_loo_ridge:.4f}, MSE={mse_ridge:.4f}")
    
    # 5. Improvement summary
    print(f"\n5. Improvement Summary:")
    print(f"   Phase CXLIV Pipeline: ~32-46%")
    print(f"   Phase CXLV LN-only: {r_ln_only:.4f} ({r_ln_only*100:.1f}%)")
    print(f"   Phase CXLVI Full sim: {r_full_sim:.4f} ({r_full_sim*100:.1f}%)")
    print(f"   Oracle: {r_oracle:.6f}")
    
    # 6. Error analysis
    errors = gaps_full_sim - gaps
    abs_errors = np.abs(errors)
    print(f"\n6. Error Analysis:")
    print(f"   Mean |error|: {np.mean(abs_errors):.4f}")
    print(f"   r(|error|, |gap|): {stats.pearsonr(abs_errors, np.abs(gaps))[0]:.4f}")
    
    return {
        'r_ln_only': float(r_ln_only),
        'r_full_sim': float(r_full_sim),
        'r_oracle': float(r_oracle),
        'r_h_input': float(r_h_input),
        'r_ridge': float(r_ridge),
        'r_loo_ridge': float(r_loo_ridge),
        'acc_sign_ln': float(acc_ln),
        'acc_sign_sim': float(acc_sim),
        'acc_sign_input': float(acc_input),
        'mse_ridge': float(mse_ridge),
        'r_prob_sim': float(r_prob_sim),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase CXLVI: Attention-MLP-LN Synergy")
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True, choices=["p645", "p646", "p647", "p648"])
    args = parser.parse_args()
    
    model_name = args.model
    experiment = args.experiment
    
    print(f"\n{'#'*70}")
    print(f"# Phase CXLVI: Attention-MLP-LN Synergistic Emergence")
    print(f"# Model: {model_name}, Experiment: {experiment}")
    print(f"# Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"  Model info: {model_info.n_layers} layers, d_model={model_info.d_model}, vocab={model_info.vocab_size}")
    
    print(f"\nComputing last-layer features...")
    all_features, extra = compute_last_layer_features(model, tokenizer, device, TEST_TEXTS)
    print(f"  Features computed for {len(all_features)} texts")
    
    if experiment == "p645":
        results = experiment_p645(all_features, extra, model_name)
    elif experiment == "p646":
        results = experiment_p646(all_features, extra, model_name)
    elif experiment == "p647":
        results = experiment_p647(all_features, extra, model_name)
    elif experiment == "p648":
        results = experiment_p648(all_features, extra, model_name)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    
    output_dir = "d:/develop/TransformerLens-main/results/phase_cxlvi"
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
    
    results_converted = convert(results)
    
    output_file = f"{output_dir}/{experiment}_{model_name}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_converted, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    release_model(model)
    
    print(f"\n{'#'*70}")
    print(f"# Phase CXLVI {experiment} ({model_name}) COMPLETE")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
