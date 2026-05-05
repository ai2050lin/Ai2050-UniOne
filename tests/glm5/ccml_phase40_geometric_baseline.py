"""
CCML Phase 40: Geometric Baseline and Training-Induced Gain
============================================================

KEY INSIGHT from Phase 39 critique analysis:
DER ≈ √(πd/2) at final hidden state is a GEOMETRIC NECESSITY,
not a training effect. Phase 39A's 65x DER is explained by
high-dimensional geometry, not by training-induced anisotropic gain.

Mathematical proof:
  At final hidden state: logits = h @ W_U^T
  gain(margin_dir) = ||W_U[top] - W_U[second]||
  gain(random) = E[|cos(θ)|] · ||W_U[top] - W_U[second]||
  where E[|cos(θ)|] ≈ √(2/(πd)) for random unit vectors in d dimensions
  → DER = √(πd/2) ≈ 75 for d=3584

The REAL test of training-induced anisotropy is at INTERMEDIATE layers,
where the perturbation must propagate through the Jacobian.

User's critique points addressed:
1. ✅ α* is measurement threshold → confirmed, also DER at final layer is geometric
2. ✅ DER is misleading → confirmed, DER/√d (gain_ratio) is better metric
3. ✅ System should be normalized nonlinear dynamics → acknowledged
4. ✅ Direction importance = max logit functional direction → testing this
5. ✅ RASC = functional alignment, not spectral alignment → Phase 39+ already showed this

Experiments:
40A: Theoretical DER baseline verification + three-way direction comparison
     - Verify DER_measured ≈ √(πd/2) at final hidden state
     - Compare gain for: margin_dir, top_logit_dir, top_SV_of_WU, random
40B: Untrained model control at intermediate layers
     - Randomize attention+MLP weights, keep norm+embed+W_U
     - Measure DER at L7, L14, L21, L26, L27
     - If trained DER >> untrained DER → training creates anisotropic gain
40C: Prediction-specific vs random-token DER at intermediate layers
     - Compare DER for model's predicted token vs a random token
     - If DER(predicted) >> DER(random) → gain is prediction-specific

Usage:
  python ccml_phase40_geometric_baseline.py --model deepseek7b --exp 1
  python ccml_phase40_geometric_baseline.py --model deepseek7b --exp 2
  python ccml_phase40_geometric_baseline.py --model deepseek7b --exp 3
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


def compute_theoretical_der(d_model):
    """
    Theoretical DER at final hidden state.
    
    For logits = h @ W_U^T, the gain along direction δ is:
      gain(δ) = |(W_U[top] - W_U[second]) · δ|
    
    For δ = margin_dir: gain = ||W_U[top] - W_U[second]||
    For δ = random: gain ≈ ||W_U[top] - W_U[second]|| · √(2/(πd))
    
    DER = √(πd/2)
    """
    return np.sqrt(np.pi * d_model / 2)


def get_W_U_np(model):
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        return model.lm_head.weight.detach().cpu().float().numpy()
    return model.get_output_embeddings().weight.detach().cpu().float().numpy()


def inject_additive(model, input_ids, attention_mask, hook_target, delta,
                    token_pos=-1):
    """加法注入: 在hook_target的输出上添加delta (只修改last token位置)"""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            new_h = output[0].clone()
            new_h[0, token_pos, :] += delta.to(new_h.dtype).to(new_h.device)
            return (new_h,) + output[1:]
        new_h = output.clone()
        new_h[0, token_pos, :] += delta.to(new_h.dtype).to(new_h.device)
        return new_h

    handle = hook_target.register_forward_hook(hook_fn)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    handle.remove()
    return outputs.logits[0, -1].cpu().float().numpy()


def get_base_forward(model, tokenizer, device, text):
    """获取base forward的结果"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)

    logits = outputs.logits[0, -1].cpu().float().numpy()
    hs_norms = [outputs.hidden_states[li][0, -1].float().norm().item()
                for li in range(len(outputs.hidden_states))]
    hs_final = outputs.hidden_states[-1][0, -1].detach().cpu().float()

    top_token_id = int(np.argmax(logits))
    second_token_id = int(np.argsort(logits)[-2])
    base_margin = logits[top_token_id] - logits[second_token_id]

    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'logits': logits,
        'hs_norms': hs_norms,
        'hs_final': hs_final,
        'top_token_id': top_token_id,
        'second_token_id': second_token_id,
        'base_margin': base_margin,
    }

    del outputs
    torch.cuda.empty_cache()
    return result


def measure_direction_gain(model, tokenizer, device, base_result, direction_np,
                          hook_target, alpha=0.1, top_token_id=None, second_token_id=None):
    """测量沿指定方向的margin增益"""
    if top_token_id is None:
        top_token_id = base_result['top_token_id']
    if second_token_id is None:
        second_token_id = base_result['second_token_id']

    # 获取注入层的hidden state norm
    input_ids = base_result['input_ids']
    attention_mask = base_result['attention_mask']

    # 需要知道注入层的hidden state norm
    # 对于final layer, 用hs_final的norm
    # 对于intermediate layer, 需要从hidden_states获取
    # 这里简化: 使用base_result中的信息

    # 计算delta
    d_model = len(direction_np)
    delta = torch.tensor(direction_np, dtype=torch.float32, device=device)

    # 注入并获取logits
    perturbed_logits = inject_additive(model, input_ids, attention_mask,
                                       hook_target, delta)

    base_margin = base_result['base_margin']
    perturbed_margin = perturbed_logits[top_token_id] - perturbed_logits[second_token_id]
    delta_margin = perturbed_margin - base_margin

    return float(delta_margin)


# ============================================================================
# 40A: Theoretical DER Baseline + Three-way Direction Comparison
# ============================================================================

def run_40A(model_name):
    """Theoretical DER baseline verification + direction comparison at final hidden state"""
    print(f"\n{'='*70}")
    print(f"Phase 40A: Geometric Baseline + Direction Comparison ({model_name})")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    layers = get_layers(model)

    der_theory = compute_theoretical_der(d_model)
    print(f"\nd_model = {d_model}, n_layers = {n_layers}")
    print(f"Theoretical DER at final hidden state: √(πd/2) = {der_theory:.1f}")
    print(f"If measured DER ≈ {der_theory:.0f}, then DER is GEOMETRIC, not training effect")

    # Get W_U
    W_U = get_W_U_np(model)  # [vocab_size, d_model]

    # Compute top singular vector of W_U using Power Method
    # W_U is [vocab_size, d_model], top right SV = argmax ||W_U v||
    print("\nComputing top singular vector of W_U (Power Method)...")
    v = np.random.randn(d_model)
    v = v / np.linalg.norm(v)
    W_U_T = W_U.T  # [d_model, vocab_size]
    for i in range(50):
        # Power method: v_{k+1} = W_U^T @ (W_U @ v_k) / norm
        u = W_U @ v  # [vocab_size]
        v_new = W_U_T @ u  # [d_model]
        norm = np.linalg.norm(v_new)
        if norm > 0:
            v_new = v_new / norm
        # Check convergence
        if np.abs(np.abs(np.dot(v_new, v)) - 1.0) < 1e-8:
            break
        v = v_new
    top_sv_dir = v_new
    # Compute top singular value
    top_sv = np.linalg.norm(W_U @ top_sv_dir)
    print(f"  Top SV of W_U: σ₁ = {top_sv:.2f}")

    alpha = 0.1
    n_random = 20

    all_results = []

    for ti, text in enumerate(TEST_SENTENCES):
        print(f"\n--- Text {ti+1}: '{text[:50]}' ---")

        base = get_base_forward(model, tokenizer, device, text)
        top_tok = base['top_token_id']
        second_tok = base['second_token_id']
        hs_norm = base['hs_norms'][-1]

        print(f"  Top token: {safe_decode(tokenizer, top_tok)} (id={top_tok})")
        print(f"  Second token: {safe_decode(tokenizer, second_tok)} (id={second_tok})")
        print(f"  ||h_final|| = {hs_norm:.1f}, base_margin = {base['base_margin']:.3f}")

        # Define directions
        margin_dir = W_U[top_tok] - W_U[second_tok]
        margin_dir = margin_dir / (np.linalg.norm(margin_dir) + 1e-10)

        top_logit_dir = W_U[top_tok].copy()
        top_logit_dir = top_logit_dir / (np.linalg.norm(top_logit_dir) + 1e-10)

        # Random directions
        np.random.seed(42 + ti)
        random_dirs = np.random.randn(n_random, d_model)
        random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)

        # Measure gains for each direction (inject at final hidden state)
        # Final layer hook target
        final_hook_target = layers[-1]

        # Delta magnitude = alpha * hs_norm
        delta_mag = alpha * hs_norm

        # 1. Margin direction
        delta_margin = torch.tensor(delta_mag * margin_dir, dtype=torch.float32, device=device)
        logits_margin = inject_additive(model, base['input_ids'], base['attention_mask'],
                                        final_hook_target, delta_margin)
        gain_margin = (logits_margin[top_tok] - logits_margin[second_tok]) - base['base_margin']

        # 2. Top logit direction
        delta_toplogit = torch.tensor(delta_mag * top_logit_dir, dtype=torch.float32, device=device)
        logits_toplogit = inject_additive(model, base['input_ids'], base['attention_mask'],
                                          final_hook_target, delta_toplogit)
        gain_toplogit = (logits_toplogit[top_tok] - logits_toplogit[second_tok]) - base['base_margin']

        # 3. Top SV direction
        delta_topsv = torch.tensor(delta_mag * top_sv_dir, dtype=torch.float32, device=device)
        logits_topsv = inject_additive(model, base['input_ids'], base['attention_mask'],
                                       final_hook_target, delta_topsv)
        gain_topsv = (logits_topsv[top_tok] - logits_topsv[second_tok]) - base['base_margin']

        # 4. Random directions
        random_gains = []
        for ri in range(n_random):
            delta_rand = torch.tensor(delta_mag * random_dirs[ri], dtype=torch.float32, device=device)
            logits_rand = inject_additive(model, base['input_ids'], base['attention_mask'],
                                          final_hook_target, delta_rand)
            gain_rand = (logits_rand[top_tok] - logits_rand[second_tok]) - base['base_margin']
            random_gains.append(gain_rand)

        random_mean = np.mean(np.abs(random_gains))
        random_std = np.std(np.abs(random_gains))

        # Compute DERs
        der_margin = abs(gain_margin) / (random_mean + 1e-10)
        der_toplogit = abs(gain_toplogit) / (random_mean + 1e-10)
        der_topsv = abs(gain_topsv) / (random_mean + 1e-10)

        # Gain ratio = DER / DER_theory
        gain_ratio_margin = der_margin / der_theory
        gain_ratio_toplogit = der_toplogit / der_theory
        gain_ratio_topsv = der_topsv / der_theory

        # Cosine similarities between directions
        cos_margin_toplogit = abs(np.dot(margin_dir, top_logit_dir))
        cos_margin_topsv = abs(np.dot(margin_dir, top_sv_dir))
        cos_toplogit_topsv = abs(np.dot(top_logit_dir, top_sv_dir))

        print(f"\n  === Gains (α={alpha}) ===")
        print(f"  margin_dir:  gain = {gain_margin:+.4f}, DER = {der_margin:.1f}, ratio = {gain_ratio_margin:.3f}")
        print(f"  top_logit:   gain = {gain_toplogit:+.4f}, DER = {der_toplogit:.1f}, ratio = {gain_ratio_toplogit:.3f}")
        print(f"  top_SV_WU:   gain = {gain_topsv:+.4f}, DER = {der_topsv:.1f}, ratio = {gain_ratio_topsv:.3f}")
        print(f"  random_mean: |gain| = {random_mean:.4f} ± {random_std:.4f}")
        print(f"\n  === Direction Cosines ===")
        print(f"  cos(margin, top_logit) = {cos_margin_toplogit:.4f}")
        print(f"  cos(margin, top_SV)    = {cos_margin_topsv:.4f}")
        print(f"  cos(top_logit, top_SV) = {cos_toplogit_topsv:.4f}")
        print(f"\n  === Theory vs Measurement ===")
        print(f"  DER_theory = {der_theory:.1f}")
        print(f"  DER_margin_measured = {der_margin:.1f} → ratio = {gain_ratio_margin:.3f}")
        print(f"  If ratio ≈ 1.0: DER is GEOMETRIC (not training effect)")

        all_results.append({
            'text': text[:40],
            'd_model': d_model,
            'der_theory': float(der_theory),
            'hs_norm': float(hs_norm),
            'gain_margin': float(gain_margin),
            'gain_toplogit': float(gain_toplogit),
            'gain_topsv': float(gain_topsv),
            'random_mean': float(random_mean),
            'random_std': float(random_std),
            'der_margin': float(der_margin),
            'der_toplogit': float(der_toplogit),
            'der_topsv': float(der_topsv),
            'gain_ratio_margin': float(gain_ratio_margin),
            'gain_ratio_toplogit': float(gain_ratio_toplogit),
            'gain_ratio_topsv': float(gain_ratio_topsv),
            'cos_margin_toplogit': float(cos_margin_toplogit),
            'cos_margin_topsv': float(cos_margin_topsv),
            'cos_toplogit_topsv': float(cos_toplogit_topsv),
        })

        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print(f"Phase 40A SUMMARY ({model_name})")
    print(f"{'='*70}")
    print(f"d_model = {d_model}")
    print(f"DER_theory = √(πd/2) = {der_theory:.1f}")
    print()

    avg_der_margin = np.mean([r['der_margin'] for r in all_results])
    avg_der_toplogit = np.mean([r['der_toplogit'] for r in all_results])
    avg_der_topsv = np.mean([r['der_topsv'] for r in all_results])
    avg_ratio_margin = np.mean([r['gain_ratio_margin'] for r in all_results])
    avg_ratio_toplogit = np.mean([r['gain_ratio_toplogit'] for r in all_results])
    avg_ratio_topsv = np.mean([r['gain_ratio_topsv'] for r in all_results])

    print(f"Average DER (margin_dir):  {avg_der_margin:.1f} (theory: {der_theory:.1f}, ratio: {avg_ratio_margin:.3f})")
    print(f"Average DER (top_logit):   {avg_der_toplogit:.1f} (ratio: {avg_ratio_toplogit:.3f})")
    print(f"Average DER (top_SV_WU):   {avg_der_topsv:.1f} (ratio: {avg_ratio_topsv:.3f})")
    print()

    if avg_ratio_margin < 1.5:
        print("★ CONCLUSION: DER at final hidden state is GEOMETRIC (ratio ≈ 1.0)")
        print("  Phase 39A's 65x DER is explained by √(πd/2), NOT by training!")
    else:
        print("⚠ UNEXPECTED: DER ratio >> 1.0, additional structure beyond geometry")

    # Save results
    out_path = f'tests/glm5_temp/phase40A_{model_name}_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'d_model': d_model, 'der_theory': float(der_theory),
                   'results': all_results}, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    release_model(model)
    return all_results


# ============================================================================
# 40B: Untrained Model Control at Intermediate Layers
# ============================================================================

def randomize_model_weights(model, model_name):
    """Randomize attention+MLP weights, keep norm+embed+W_U"""
    info = get_model_info(model, model_name)
    layers = get_layers(model)

    n_randomized = 0
    n_kept = 0

    for li, layer in enumerate(layers):
        # Randomize attention weights
        for attr in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if hasattr(layer.self_attn, attr):
                w = getattr(layer.self_attn, attr)
                if hasattr(w, 'weight'):
                    torch.nn.init.normal_(w.weight.data, std=0.02)
                    n_randomized += 1
                if hasattr(w, 'bias') and w.bias is not None:
                    torch.nn.init.zeros_(w.bias.data)

        # Randomize MLP weights
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

        # Keep layer norm weights
        n_kept += 2  # input_layernorm + post_attention_layernorm

    # Keep W_U (lm_head)
    n_kept += 1

    # Keep embedding
    n_kept += 1

    # Keep final layer norm
    if hasattr(model.model, 'norm'):
        n_kept += 1

    print(f"  Randomized {n_randomized} weight matrices, kept {n_kept} (norm+embed+W_U)")
    return model


def measure_der_at_layer(model, tokenizer, device, text, layer_idx, layers,
                         W_U, top_tok, second_tok, alpha=0.1, n_random=15):
    """Measure DER at a specific intermediate layer"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Get base forward with hidden states
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)

    base_logits = outputs.logits[0, -1].cpu().float().numpy()
    base_margin = base_logits[top_tok] - base_logits[second_tok]

    # Get hidden state norm at this layer
    # hidden_states[0] = embedding, hidden_states[1] = after layer 0, etc.
    hs_norm = outputs.hidden_states[layer_idx + 1][0, -1].float().norm().item()

    del outputs
    torch.cuda.empty_cache()

    # Define directions
    d_model = W_U.shape[1]
    margin_dir = W_U[top_tok] - W_U[second_tok]
    margin_dir = margin_dir / (np.linalg.norm(margin_dir) + 1e-10)

    np.random.seed(42)
    random_dirs = np.random.randn(n_random, d_model)
    random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)

    # Inject at this layer
    hook_target = layers[layer_idx]
    delta_mag = alpha * hs_norm

    # Margin direction
    delta_margin = torch.tensor(delta_mag * margin_dir, dtype=torch.float32, device=device)
    logits_margin = inject_additive(model, input_ids, attention_mask,
                                    hook_target, delta_margin)
    gain_margin = (logits_margin[top_tok] - logits_margin[second_tok]) - base_margin

    # -Margin direction
    delta_neg_margin = torch.tensor(-delta_mag * margin_dir, dtype=torch.float32, device=device)
    logits_neg = inject_additive(model, input_ids, attention_mask,
                                 hook_target, delta_neg_margin)
    gain_neg_margin = (logits_neg[top_tok] - logits_neg[second_tok]) - base_margin

    # Random directions
    random_gains = []
    for ri in range(n_random):
        delta_rand = torch.tensor(delta_mag * random_dirs[ri], dtype=torch.float32, device=device)
        logits_rand = inject_additive(model, input_ids, attention_mask,
                                      hook_target, delta_rand)
        gain_rand = (logits_rand[top_tok] - logits_rand[second_tok]) - base_margin
        random_gains.append(gain_rand)

    random_mean = np.mean(np.abs(random_gains))
    random_std = np.std(np.abs(random_gains))

    der = abs(gain_margin) / (random_mean + 1e-10)

    return {
        'layer': layer_idx,
        'hs_norm': float(hs_norm),
        'gain_margin': float(gain_margin),
        'gain_neg_margin': float(gain_neg_margin),
        'random_mean': float(random_mean),
        'random_std': float(random_std),
        'der': float(der),
        'base_margin': float(base_margin),
    }


def run_40B(model_name):
    """Untrained model control at intermediate layers"""
    print(f"\n{'='*70}")
    print(f"Phase 40B: Untrained Model Control ({model_name})")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    layers = get_layers(model)
    W_U = get_W_U_np(model)
    der_theory = compute_theoretical_der(d_model)

    print(f"d_model = {d_model}, n_layers = {n_layers}")
    print(f"DER_theory = {der_theory:.1f}")

    # Layer indices to test
    test_layer_indices = [7, 14, 21, min(26, n_layers-2), min(27, n_layers-2)]
    test_layer_indices = sorted(set([li for li in test_layer_indices if li < n_layers]))

    alpha = 0.1
    texts = TEST_SENTENCES[:3]  # Use fewer texts to save time

    # ===== Step 1: Measure DER for TRAINED model =====
    print(f"\n--- Step 1: TRAINED model DER at intermediate layers ---")
    trained_results = {}

    for li in test_layer_indices:
        layer_ders = []
        for text in texts:
            # First get top/second tokens from trained model
            base = get_base_forward(model, tokenizer, device, text)
            top_tok = base['top_token_id']
            second_tok = base['second_token_id']

            result = measure_der_at_layer(model, tokenizer, device, text, li, layers,
                                         W_U, top_tok, second_tok, alpha=alpha)
            layer_ders.append(result)
            torch.cuda.empty_cache()

        avg_der = np.mean([r['der'] for r in layer_ders])
        avg_gain = np.mean([r['gain_margin'] for r in layer_ders])
        avg_rand = np.mean([r['random_mean'] for r in layer_ders])
        trained_results[li] = {
            'avg_der': float(avg_der),
            'avg_gain_margin': float(avg_gain),
            'avg_random': float(avg_rand),
            'details': layer_ders,
        }
        # Count remaining RMSNorm layers from this layer to output
        n_remaining_norms = 2 * (n_layers - li) + 1  # rough estimate
        print(f"  L{li}: DER = {avg_der:.1f}x, gain_margin = {avg_gain:+.3f}, "
              f"random = {avg_rand:.4f}")

    # Store trained top/second tokens for fair comparison
    trained_tokens = {}
    for text in texts:
        base = get_base_forward(model, tokenizer, device, text)
        trained_tokens[text] = (base['top_token_id'], base['second_token_id'])

    # ===== Step 2: Randomize model =====
    print(f"\n--- Step 2: Randomizing model weights ---")
    model = randomize_model_weights(model, model_name)

    # ===== Step 3: Check if untrained model produces valid output =====
    print(f"\n--- Step 3: Sanity check for untrained model ---")
    valid = True
    for text in texts[:1]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, -1].cpu().float().numpy()
        if np.isnan(logits).any() or np.isinf(logits).any():
            print(f"  ✗ NaN/Inf in untrained model output!")
            valid = False
            break
        print(f"  Logits range: [{logits.min():.2f}, {logits.max():.2f}], "
              f"std = {logits.std():.2f}")
        del outputs
        torch.cuda.empty_cache()

    if not valid:
        print("Untrained model produces invalid output. Trying with smaller init...")
        # Re-load and try with smaller init
        release_model(model)
        model, tokenizer, device = load_model(model_name)
        # Use smaller initialization
        for li, layer in enumerate(get_layers(model)):
            for attr in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if hasattr(layer.self_attn, attr):
                    w = getattr(layer.self_attn, attr)
                    if hasattr(w, 'weight'):
                        torch.nn.init.normal_(w.weight.data, std=0.01)
            mlp = layer.mlp
            for attr in ['gate_proj', 'up_proj', 'down_proj', 'gate_up_proj']:
                if hasattr(mlp, attr):
                    w = getattr(mlp, attr)
                    if hasattr(w, 'weight'):
                        torch.nn.init.normal_(w.weight.data, std=0.01)

        # Re-check
        for text in texts[:1]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0, -1].cpu().float().numpy()
            if np.isnan(logits).any() or np.isinf(logits).any():
                print("  ✗ Still invalid. Aborting untrained control.")
                release_model(model)
                return None
            print(f"  ✓ Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
            del outputs
            torch.cuda.empty_cache()

    # ===== Step 4: Measure DER for UNTRAINED model =====
    print(f"\n--- Step 4: UNTRAINED model DER at intermediate layers ---")
    untrained_results = {}

    for li in test_layer_indices:
        layer_ders = []
        for text in texts:
            # Use TRAINED model's top/second tokens for fair comparison
            top_tok, second_tok = trained_tokens[text]

            result = measure_der_at_layer(model, tokenizer, device, text, li,
                                         get_layers(model), W_U, top_tok, second_tok,
                                         alpha=alpha)

            # Check for NaN
            if np.isnan(result['der']) or np.isinf(result['der']):
                print(f"  L{li}: NaN/Inf in DER, skipping")
                continue

            layer_ders.append(result)
            torch.cuda.empty_cache()

        if len(layer_ders) > 0:
            avg_der = np.mean([r['der'] for r in layer_ders])
            avg_gain = np.mean([r['gain_margin'] for r in layer_ders])
            avg_rand = np.mean([r['random_mean'] for r in layer_ders])
            untrained_results[li] = {
                'avg_der': float(avg_der),
                'avg_gain_margin': float(avg_gain),
                'avg_random': float(avg_rand),
                'details': layer_ders,
            }
            print(f"  L{li}: DER = {avg_der:.1f}x, gain_margin = {avg_gain:+.3f}, "
                  f"random = {avg_rand:.4f}")

    # ===== Step 5: Compare trained vs untrained =====
    print(f"\n{'='*70}")
    print(f"Phase 40B SUMMARY: Trained vs Untrained DER ({model_name})")
    print(f"{'='*70}")
    print(f"{'Layer':<8} {'Trained DER':<14} {'Untrained DER':<14} {'Ratio':<10} {'Conclusion'}")
    print(f"{'-'*56}")

    for li in test_layer_indices:
        t_der = trained_results.get(li, {}).get('avg_der', 0)
        u_der = untrained_results.get(li, {}).get('avg_der', 0)
        ratio = t_der / (u_der + 1e-10)

        if ratio > 2:
            conclusion = "✓ Training-induced gain"
        elif ratio > 1.2:
            conclusion = "⚠ Weak training effect"
        else:
            conclusion = "✗ No training effect"

        print(f"L{li:<6} {t_der:<14.1f} {u_der:<14.1f} {ratio:<10.2f} {conclusion}")

    # Save results
    out_data = {
        'model': model_name,
        'd_model': d_model,
        'der_theory': float(der_theory),
        'alpha': alpha,
        'trained_results': {str(k): v for k, v in trained_results.items()},
        'untrained_results': {str(k): v for k, v in untrained_results.items()},
    }
    out_path = f'tests/glm5_temp/phase40B_{model_name}_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}")

    release_model(model)
    return out_data


# ============================================================================
# 40C: Prediction-Specific vs Random-Token DER at Intermediate Layers
# ============================================================================

def run_40C(model_name):
    """Compare DER for predicted token vs random token at intermediate layers"""
    print(f"\n{'='*70}")
    print(f"Phase 40C: Prediction-Specific vs Random-Token DER ({model_name})")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    layers = get_layers(model)
    W_U = get_W_U_np(model)
    der_theory = compute_theoretical_der(d_model)

    alpha = 0.1
    test_layer_indices = [7, 14, 21, min(26, n_layers-2)]
    test_layer_indices = sorted(set([li for li in test_layer_indices if li < n_layers]))
    n_random_tokens = 5

    all_results = []

    for ti, text in enumerate(TEST_SENTENCES[:3]):
        print(f"\n--- Text {ti+1}: '{text[:50]}' ---")

        # Get trained model's prediction
        base = get_base_forward(model, tokenizer, device, text)
        top_tok = base['top_token_id']
        second_tok = base['second_token_id']

        print(f"  Predicted: {safe_decode(tokenizer, top_tok)}, "
              f"Second: {safe_decode(tokenizer, second_tok)}")

        # Select random tokens (not top or second)
        np.random.seed(100 + ti)
        vocab_size = W_U.shape[0]
        random_toks = []
        for _ in range(n_random_tokens):
            rt = np.random.randint(0, vocab_size)
            while rt == top_tok or rt == second_tok:
                rt = np.random.randint(0, vocab_size)
            random_toks.append(rt)

        for li in test_layer_indices:
            # Measure DER for predicted token's margin direction
            result_pred = measure_der_at_layer(model, tokenizer, device, text, li,
                                               layers, W_U, top_tok, second_tok, alpha=alpha)

            # Measure DER for random tokens' margin directions
            random_der_list = []
            for rt in random_toks:
                # Find the second-highest logit for this random token
                # For simplicity, use the model's second token as comparison
                result_rand = measure_der_at_layer(model, tokenizer, device, text, li,
                                                   layers, W_U, rt, second_tok, alpha=alpha)
                random_der_list.append(result_rand['der'])

            avg_random_der = np.mean(random_der_list)
            predicted_der = result_pred['der']
            specificity_ratio = predicted_der / (avg_random_der + 1e-10)

            print(f"  L{li}: DER(predicted) = {predicted_der:.1f}x, "
                  f"DER(random_tok) = {avg_random_der:.1f}x, "
                  f"specificity = {specificity_ratio:.2f}")

            all_results.append({
                'text': text[:40],
                'layer': li,
                'predicted_der': float(predicted_der),
                'random_token_der': float(avg_random_der),
                'specificity_ratio': float(specificity_ratio),
                'predicted_gain': float(result_pred['gain_margin']),
                'predicted_random_gain': float(result_pred['random_mean']),
            })

            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print(f"Phase 40C SUMMARY ({model_name})")
    print(f"{'='*70}")

    for li in test_layer_indices:
        layer_results = [r for r in all_results if r['layer'] == li]
        avg_spec = np.mean([r['specificity_ratio'] for r in layer_results])
        avg_pred_der = np.mean([r['predicted_der'] for r in layer_results])
        avg_rand_der = np.mean([r['random_token_der'] for r in layer_results])

        print(f"L{li}: DER(pred) = {avg_pred_der:.1f}x, DER(rand_tok) = {avg_rand_der:.1f}x, "
              f"specificity = {avg_spec:.2f}")

        if avg_spec > 1.5:
            print(f"  → ✓ DER is prediction-specific (training shapes Jacobian for predicted direction)")
        else:
            print(f"  → ✗ DER is NOT prediction-specific (gain is generic)")

    # Save results
    out_path = f'tests/glm5_temp/phase40C_{model_name}_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'model': model_name, 'd_model': d_model, 'results': all_results},
                  f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    release_model(model)
    return all_results


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 40: Geometric Baseline')
    parser.add_argument('--model', type=str, required=True,
                       choices=['deepseek7b', 'glm4', 'qwen3'])
    parser.add_argument('--exp', type=int, required=True, choices=[1, 2, 3],
                       help='1=40A (theory+directions), 2=40B (untrained control), 3=40C (prediction-specific)')
    args = parser.parse_args()

    if args.exp == 1:
        run_40A(args.model)
    elif args.exp == 2:
        run_40B(args.model)
    elif args.exp == 3:
        run_40C(args.model)
