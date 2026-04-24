"""
Phase CCXLV: Local Geometry Direct Measurement
================================================
CRITICAL BREAKTHROUGH: Previous 14 phases all used perturbation-response through
the full network, which mixes "local nonlinearity" with "asymmetric propagation".
This cannot be fully separated by any normalization trick.

NEW APPROACH: Measure each layer's LOCAL Jacobian and Hessian norms directly,
by perturbing the INPUT to that layer and measuring the OUTPUT of that SAME layer.
No cross-layer propagation involved!

Key metrics per layer l:
  1. ||J_l|| ≈ ||f_l(h+εv) - f_l(h-εv)|| / (2ε)  — Jacobian (linear sensitivity)
  2. ||H_l|| ≈ ||f_l(h+εv) - 2f_l(h) + f_l(h-εv)|| / ε²  — Hessian (curvature)
  3. κ_l = ||H_l|| / ||J_l||  — curvature density (intrinsic nonlinearity)

If Corr(layer, ||H_l||) < 0 → shallow layers genuinely more nonlinear
If Corr(layer, ||J_l||) < 0 → shallow layers have larger Jacobian (consistent with CCXLIV)
If Corr(layer, κ_l) < 0 → shallow layers have higher curvature DENSITY

Also: Weight matrix spectral analysis (no forward pass needed, very fast)
  - ||W_attn||_spectral, ||W_mlp||_spectral per layer
  - Condition numbers
  - Correlate with layer depth
"""

import argparse
import json
import os
import numpy as np
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_CONFIGS = {
    'qwen3': {
        'name': 'Qwen3-4B',
        'path': 'D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c',
        'n_layers': 36,
        'd_model': 2560,
        'dtype': 'bf16',
    },
    'deepseek7b': {
        'name': 'DeepSeek-R1-Distill-Qwen-7B',
        'path': 'D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'n_layers': 28,
        'd_model': 3584,
        'dtype': '8bit',
    },
    'glm4': {
        'name': 'GLM4-9B-Chat',
        'path': 'D:/develop/model/hub/models--zai-org--glm-4-9b-chat-hf/download',
        'n_layers': 40,
        'd_model': 4096,
        'dtype': '8bit',
    }
}

SENTENCES = [
    "She walks to school",
    "He runs in the park",
    "They play football",
    "The cat sleeps on the mat",
    "She sings beautifully",
    "He writes a letter",
    "They travel abroad",
    "The dog barks loudly",
    "She cooks dinner",
    "He drives carefully",
    "The bird flies south",
    "She reads the book",
    "They build houses",
    "The river flows north",
    "She teaches mathematics",
]


def load_model(model_key, device='cuda'):
    config = MODEL_CONFIGS[model_key]
    path = config['path']
    dtype = config['dtype']

    print(f"  Loading: {config['name']} ({dtype})")

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dtype == '8bit':
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, quantization_config=bnb_config, device_map="auto",
            trust_remote_code=True, local_files_only=True
        )
    elif dtype == 'bf16':
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, device_map="auto",
            trust_remote_code=True, local_files_only=True
        )

    model.eval()
    return model, tokenizer


def get_target_layer(model, layer_idx):
    model_type = model.config.model_type
    if model_type in ['qwen2', 'qwen3']:
        return model.model.layers[layer_idx]
    elif model_type in ['chatglm', 'glm4']:
        return model.transformer.encoder.layers[layer_idx]
    else:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return model.model.layers[layer_idx]
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'encoder'):
            return model.transformer.encoder.layers[layer_idx]
        else:
            raise ValueError(f"Cannot find layers for model type {model_type}")


def analyze_weight_matrices(model, model_key):
    """Analyze weight matrices without any forward pass."""
    cfg = MODEL_CONFIGS[model_key]
    n_layers = cfg['n_layers']

    results = []

    for l in range(n_layers):
        layer = get_target_layer(model, l)

        # Extract weight matrices
        weight_norms = {}
        condition_numbers = {}

        for name, param in layer.named_parameters():
            if 'weight' in name and param.ndim == 2:
                if param.is_meta:
                    continue
                try:
                    w = param.detach().float()
                    frobenius = torch.norm(w).item()
                    if min(w.shape) <= 100:
                        svs = torch.linalg.svdvals(w)
                        spectral = svs[0].item()
                        sv_min = svs[-1].item()
                    else:
                        spectral = frobenius / max(w.shape)**0.5
                        sv_min = 0.01
                    cond = spectral / (sv_min + 1e-10)
                except RuntimeError:
                    continue

                short_name = name.split('.')[-2] + '.' + name.split('.')[-1]
                weight_norms[short_name] = {
                    'frobenius': float(frobenius),
                    'spectral': float(spectral),
                }
                condition_numbers[short_name] = float(min(cond, 1e6))

        # Compute summary statistics
        if not weight_norms:
            results.append({
                'layer': l,
                'mean_spectral': 0,
                'max_spectral': 0,
                'mean_frobenius': 0,
                'mean_condition': 0,
                'weight_norms': {},
                'condition_numbers': {},
            })
            continue

        all_spectrals = [v['spectral'] for v in weight_norms.values()]
        all_frobenius = [v['frobenius'] for v in weight_norms.values()]
        all_conds = list(condition_numbers.values())

        results.append({
            'layer': l,
            'mean_spectral': float(np.mean(all_spectrals)),
            'max_spectral': float(np.max(all_spectrals)),
            'mean_frobenius': float(np.mean(all_frobenius)),
            'mean_condition': float(np.mean(all_conds)),
            'weight_norms': weight_norms,
            'condition_numbers': condition_numbers,
        })

    return results


def measure_local_geometry(model, tokenizer, model_key, n_samples=10, n_dirs=5, eps=1.0):
    """Measure local Jacobian and Hessian at each layer by perturbing layer INPUT."""
    cfg = MODEL_CONFIGS[model_key]
    n_layers = cfg['n_layers']
    d_model = cfg['d_model']

    # Measure at every 3rd layer + key positions
    test_layers = sorted(set(
        list(range(1, n_layers, 3)) +
        [max(1, n_layers // 7), max(2, n_layers // 3),
         max(3, n_layers // 2), max(4, 2 * n_layers // 3)]
    ))
    test_layers = [l for l in test_layers if 1 <= l < n_layers]

    sentences = SENTENCES[:n_samples]

    results = {}

    for layer_idx in test_layers:
        target_layer = get_target_layer(model, layer_idx)

        hess_vals = []
        jac_vals = []
        act_norms = []

        for sent_idx, sentence in enumerate(sentences):
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
            input_ids = inputs['input_ids'].to(model.device)
            attention_mask = inputs['attention_mask'].to(model.device)
            seq_len = attention_mask.sum().item()
            last_pos = seq_len - 1

            # Step 1: Clean forward pass - capture input and output of target layer
            clean_input = [None]
            clean_output = [None]

            def make_capture_hooks(li):
                def pre_hook(module, args):
                    hidden = args[0]
                    clean_input[0] = hidden[0, last_pos, :].detach().cpu().float().clone()
                    return args

                def post_hook(module, args, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    clean_output[0] = hidden[0, last_pos, :].detach().cpu().float().clone()
                    return output

                return pre_hook, post_hook

            pre_h, post_h = make_capture_hooks(layer_idx)
            h1 = target_layer.register_forward_pre_hook(pre_h)
            h2 = target_layer.register_forward_hook(post_h)

            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

            h1.remove()
            h2.remove()

            h_clean = clean_input[0].numpy()
            f_clean = clean_output[0].numpy()
            act_norms.append(np.linalg.norm(h_clean))

            # Step 2: Perturbations for Jacobian and Hessian
            for d_idx in range(n_dirs):
                v = np.random.randn(d_model)
                v = v / (np.linalg.norm(v) + 1e-10)
                v_tensor = torch.tensor(v, dtype=torch.float32, device=model.device)

                # Use relative epsilon
                h_norm = np.linalg.norm(h_clean) + 1e-6
                local_eps = eps  # Fixed eps for simplicity

                # Forward with +eps*v at layer input
                output_plus = [None]

                def make_plus_hook(ve, ep, lp):
                    def pre_hook_plus(module, args):
                        hidden = args[0]
                        perturbed = hidden.clone()
                        perturbed[0, lp, :] = perturbed[0, lp, :].float() + ep * ve.to(perturbed.dtype)
                        return (perturbed,) + args[1:]

                    def post_hook_plus(module, args, output):
                        hidden = output[0] if isinstance(output, tuple) else output
                        output_plus[0] = hidden[0, lp, :].detach().cpu().float().clone()
                        return output

                    return pre_hook_plus, post_hook_plus

                ph_plus, poh_plus = make_plus_hook(v_tensor, local_eps, last_pos)
                h3 = target_layer.register_forward_pre_hook(ph_plus)
                h4 = target_layer.register_forward_hook(poh_plus)

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

                h3.remove()
                h4.remove()

                f_plus = output_plus[0].numpy()

                # Forward with -eps*v at layer input
                output_minus = [None]

                def make_minus_hook(ve, ep, lp):
                    def pre_hook_minus(module, args):
                        hidden = args[0]
                        perturbed = hidden.clone()
                        perturbed[0, lp, :] = perturbed[0, lp, :].float() - ep * ve.to(perturbed.dtype)
                        return (perturbed,) + args[1:]

                    def post_hook_minus(module, args, output):
                        hidden = output[0] if isinstance(output, tuple) else output
                        output_minus[0] = hidden[0, lp, :].detach().cpu().float().clone()
                        return output

                    return pre_hook_minus, post_hook_minus

                ph_minus, poh_minus = make_minus_hook(v_tensor, local_eps, last_pos)
                h5 = target_layer.register_forward_pre_hook(ph_minus)
                h6 = target_layer.register_forward_hook(poh_minus)

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

                h5.remove()
                h6.remove()

                f_minus = output_minus[0].numpy()

                # Compute local Jacobian and Hessian
                # Jacobian: Jv ≈ (f(h+εv) - f(h-εv)) / (2ε)
                jac_v = (f_plus - f_minus) / (2 * local_eps)
                jac_norm = np.linalg.norm(jac_v)

                # Hessian: v^T H v ≈ (f(h+εv) - 2f(h) + f(h-εv)) / ε²
                hess_v = (f_plus - 2 * f_clean + f_minus) / (local_eps ** 2)
                hess_norm = np.linalg.norm(hess_v)

                hess_vals.append(hess_norm)
                jac_vals.append(jac_norm)

        # Aggregate results for this layer
        mean_hess = np.mean(hess_vals)
        mean_jac = np.mean(jac_vals)
        mean_act = np.mean(act_norms)
        kappa = mean_hess / (mean_jac + 1e-10)  # curvature density

        results[layer_idx] = {
            'layer': layer_idx,
            'n_measurements': len(hess_vals),
            'hessian_mean': float(mean_hess),
            'hessian_std': float(np.std(hess_vals)),
            'jacobian_mean': float(mean_jac),
            'jacobian_std': float(np.std(jac_vals)),
            'activation_norm_mean': float(mean_act),
            'kappa': float(kappa),  # H/J = curvature density
        }

        print(f"  Layer {layer_idx:>2}: ||H||={mean_hess:.6f}, ||J||={mean_jac:.4f}, "
              f"||h||={mean_act:.2f}, kappa=H/J={kappa:.6f}")

    return results


def run_ccxlv(model, tokenizer, model_key, n_samples=10, n_dirs=5, eps=1.0):
    cfg = MODEL_CONFIGS[model_key]
    n_layers = cfg['n_layers']
    d_model = cfg['d_model']

    print(f"\n=== Phase CCXLV: Local Geometry Direct Measurement ===")
    print(f"Model: {cfg['name']}, n_layers={n_layers}, d_model={d_model}")

    # Part 1: Weight matrix analysis (very fast, no forward pass)
    print(f"\n--- Part 1: Weight Matrix Spectral Analysis ---")
    weight_results = analyze_weight_matrices(model, model_key)

    # Print weight analysis
    print(f"\n  {'Layer':>5} {'MaxSpectral':>12} {'MeanSpectral':>13} {'MeanFrobenius':>14} {'MeanCond':>10}")
    for wr in weight_results:
        print(f"  L{wr['layer']:>4} {wr['max_spectral']:>12.2f} {wr['mean_spectral']:>13.2f} "
              f"{wr['mean_frobenius']:>14.2f} {wr['mean_condition']:>10.1f}")

    # Correlation of weight properties with layer depth
    layers = [wr['layer'] for wr in weight_results]
    max_specs = [wr['max_spectral'] for wr in weight_results]
    mean_specs = [wr['mean_spectral'] for wr in weight_results]
    mean_frobs = [wr['mean_frobenius'] for wr in weight_results]
    mean_conds = [wr['mean_condition'] for wr in weight_results]

    corr_max_spec = np.corrcoef(layers, max_specs)[0, 1]
    corr_mean_spec = np.corrcoef(layers, mean_specs)[0, 1]
    corr_mean_frob = np.corrcoef(layers, mean_frobs)[0, 1]
    corr_mean_cond = np.corrcoef(layers, mean_conds)[0, 1]

    print(f"\n  Weight-Depth Correlations:")
    print(f"    Corr(layer, max_spectral)  = {corr_max_spec:>7.3f}")
    print(f"    Corr(layer, mean_spectral) = {corr_mean_spec:>7.3f}")
    print(f"    Corr(layer, mean_frobenius)= {corr_mean_frob:>7.3f}")
    print(f"    Corr(layer, mean_condition)= {corr_mean_cond:>7.3f}")

    # Part 2: Local Jacobian and Hessian measurement
    print(f"\n--- Part 2: Local Jacobian & Hessian Measurement ---")
    print(f"  (Perturbing layer INPUT, measuring layer OUTPUT — NO cross-layer propagation!)")
    print(f"  n_samples={n_samples}, n_dirs={n_dirs}, eps={eps}")

    geometry_results = measure_local_geometry(model, tokenizer, model_key,
                                               n_samples=n_samples, n_dirs=n_dirs, eps=eps)

    # Print geometry analysis
    print(f"\n  {'Layer':>5} {'||H||':>12} {'||J||':>10} {'||h||':>8} {'kappa=H/J':>10} {'Pos':>6}")
    geo_layers = []
    geo_hess = []
    geo_jac = []
    geo_act = []
    geo_kappa = []

    for l in sorted(geometry_results.keys()):
        d = geometry_results[l]
        pos = f"L/{n_layers / (l + 1):.1f}"
        print(f"  L{l:>4} {d['hessian_mean']:>12.6f} {d['jacobian_mean']:>10.4f} "
              f"{d['activation_norm_mean']:>8.2f} {d['kappa']:>10.6f} {pos:>6}")
        geo_layers.append(l)
        geo_hess.append(d['hessian_mean'])
        geo_jac.append(d['jacobian_mean'])
        geo_act.append(d['activation_norm_mean'])
        geo_kappa.append(d['kappa'])

    # Correlation analysis
    if len(geo_layers) > 3:
        corr_hess = np.corrcoef(geo_layers, geo_hess)[0, 1]
        corr_jac = np.corrcoef(geo_layers, geo_jac)[0, 1]
        corr_act = np.corrcoef(geo_layers, geo_act)[0, 1]
        corr_kappa = np.corrcoef(geo_layers, geo_kappa)[0, 1]

        print(f"\n  LOCAL GEOMETRY-DEPTH CORRELATIONS:")
        print(f"    Corr(layer, ||H||)     = {corr_hess:>7.3f}  {'Hessian (curvature)' :<25}")
        print(f"    Corr(layer, ||J||)     = {corr_jac:>7.3f}  {'Jacobian (sensitivity)' :<25}")
        print(f"    Corr(layer, ||h||)     = {corr_act:>7.3f}  {'Activation norm' :<25}")
        print(f"    Corr(layer, kappa=H/J) = {corr_kappa:>7.3f}  {'Curvature density' :<25}")

        print(f"\n  INTERPRETATION:")
        if corr_hess < -0.3:
            print(f"    ||H|| DECREASES with depth -> Shallow layers genuinely more nonlinear!")
        elif corr_hess > 0.3:
            print(f"    ||H|| INCREASES with depth -> Deep layers more nonlinear!")
        else:
            print(f"    ||H|| roughly CONSTANT across layers -> No depth dependence in local curvature")

        if corr_jac < -0.3:
            print(f"    ||J|| DECREASES with depth -> Consistent with J_forward from CCXLIV!")
        elif corr_jac > 0.3:
            print(f"    ||J|| INCREASES with depth -> Opposite of CCXLIV finding!")
        else:
            print(f"    ||J|| roughly CONSTANT -> No depth dependence in local Jacobian")

        if corr_kappa < -0.3:
            print(f"    kappa=H/J DECREASES with depth -> Shallow layers have higher curvature DENSITY!")
            print(f"    -> This is the INTRINSIC nonlinearity difference, stripped of propagation effects!")
        elif corr_kappa > 0.3:
            print(f"    kappa=H/J INCREASES with depth -> Deep layers have higher curvature density!")
        else:
            print(f"    kappa=H/J roughly CONSTANT -> No intrinsic nonlinearity difference across layers!")

    # Part 3: Decompose the "shallow advantage" from CCXLIV
    print(f"\n--- Part 3: Decomposing the 'Shallow Advantage' ---")

    # From CCXLIV, we know:
    # - J_{L1->out} decreases with depth (Corr < -0.5)
    # - H_true = ||[A,B]||_fix / A_eff_fix has weak negative Corr with L1 (-0.38 to -0.59)
    #
    # Now we can decompose:
    # - ||[A,B]|| ≈ ||J_{L2->out}|| * ||H_{L2}|| * ||J_{L1->L2}|| * ||delta_A|| * ||delta_B||
    # - A_eff = ||J_{L1->out}|| * ||delta_A||
    # - H_true ≈ ||J_{L2->out}|| * ||H_{L2}|| * ||J_{L1->L2}|| / ||J_{L1->out}||
    #         = ||H_{L2}|| * ||J_{L1->L2}|| / ||J_{L1->L2}||  (if product structure holds)
    #         = ||H_{L2}||
    #
    # But we found H_true varies with L1, which means ||H_{L2}|| alone doesn't explain it.
    # The remaining L1 dependence must come from the Jacobian structure (non-scalar).

    if len(geo_layers) > 3:
        # Predicted H_true from local measurements
        # H_true_predicted ≈ ||H_{L2}|| (if pure product structure)
        # If H_true still varies with L1 after fixing L2, it means the Jacobian
        # product J_{L1->L2} has directional structure that varies with L1

        print(f"  Local ||H_l|| profile (Hessian at each layer):")
        for i, l in enumerate(geo_layers):
            print(f"    L{l:>2}: ||H||={geo_hess[i]:.6f}")

        print(f"\n  Local ||J_l|| profile (Jacobian at each layer):")
        for i, l in enumerate(geo_layers):
            print(f"    L{l:>2}: ||J||={geo_jac[i]:.4f}")

        print(f"\n  CRITICAL QUESTION: Does ||H_l|| explain H_true from CCXLIV?")
        print(f"  If ||H_l|| is roughly constant, then H_true variation with L1")
        print(f"  must come from Jacobian directional structure (not captured by norms).")
        print(f"  If ||H_l|| varies with depth in the same direction as H_true,")
        print(f"  then part of the 'shallow advantage' is genuine local nonlinearity.")

    # Save results
    all_results = {
        'model': cfg['name'],
        'n_layers': n_layers,
        'd_model': d_model,
        'weight_analysis': weight_results,
        'geometry_analysis': {str(k): v for k, v in geometry_results.items()},
        'correlations': {
            'weight_max_spectral_vs_depth': float(corr_max_spec),
            'weight_mean_spectral_vs_depth': float(corr_mean_spec),
            'weight_mean_frobenius_vs_depth': float(corr_mean_frob),
            'weight_mean_condition_vs_depth': float(corr_mean_cond),
            'local_hessian_vs_depth': float(corr_hess) if len(geo_layers) > 3 else None,
            'local_jacobian_vs_depth': float(corr_jac) if len(geo_layers) > 3 else None,
            'local_activation_norm_vs_depth': float(corr_act) if len(geo_layers) > 3 else None,
            'local_kappa_vs_depth': float(corr_kappa) if len(geo_layers) > 3 else None,
        },
    }

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'deepseek7b', 'glm4'])
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--n_dirs', type=int, default=5)
    parser.add_argument('--eps', type=float, default=1.0)
    args = parser.parse_args()

    model_key = args.model
    cfg = MODEL_CONFIGS[model_key]

    out_dir = f"results/causal_fiber/{model_key}_ccxlv"
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, 'run.log')

    def log(msg):
        print(msg, flush=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log(f"=== Phase CCXLV: Local Geometry Direct Measurement ===")
    log(f"Model: {cfg['name']}, n_samples: {args.n_samples}, n_dirs: {args.n_dirs}, eps: {args.eps}")
    log(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    model, tokenizer = load_model(model_key)
    log(f"Model loaded successfully")

    results = run_ccxlv(model, tokenizer, model_key,
                        n_samples=args.n_samples, n_dirs=args.n_dirs, eps=args.eps)

    results['timestamp'] = datetime.now().isoformat()
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Final summary
    log(f"\n{'='*70}")
    log(f"FINAL SUMMARY: {cfg['name']}")
    log(f"{'='*70}")

    corrs = results['correlations']
    log(f"\n  Weight Matrix Correlations with Depth:")
    log(f"    Corr(layer, max_spectral)  = {corrs['weight_max_spectral_vs_depth']:.3f}")
    log(f"    Corr(layer, mean_spectral) = {corrs['weight_mean_spectral_vs_depth']:.3f}")
    log(f"    Corr(layer, mean_frobenius)= {corrs['weight_mean_frobenius_vs_depth']:.3f}")
    log(f"    Corr(layer, mean_condition)= {corrs['weight_mean_condition_vs_depth']:.3f}")

    if corrs['local_hessian_vs_depth'] is not None:
        log(f"\n  Local Geometry Correlations with Depth:")
        log(f"    Corr(layer, ||H||)     = {corrs['local_hessian_vs_depth']:.3f}  [Hessian/curvature]")
        log(f"    Corr(layer, ||J||)     = {corrs['local_jacobian_vs_depth']:.3f}  [Jacobian/sensitivity]")
        log(f"    Corr(layer, ||h||)     = {corrs['local_activation_norm_vs_depth']:.3f}  [Activation norm]")
        log(f"    Corr(layer, kappa=H/J) = {corrs['local_kappa_vs_depth']:.3f}  [Curvature density]")

        # Final verdict
        h_corr = corrs['local_hessian_vs_depth']
        k_corr = corrs['local_kappa_vs_depth']
        j_corr = corrs['local_jacobian_vs_depth']

        log(f"\n  FINAL VERDICT:")
        if h_corr < -0.3:
            log(f"    Local Hessian DECREASES with depth -> Shallow layers genuinely more curved")
        elif h_corr > 0.3:
            log(f"    Local Hessian INCREASES with depth -> Deep layers more curved")
        else:
            log(f"    Local Hessian roughly CONSTANT -> No depth-dependent local curvature")

        if k_corr < -0.3:
            log(f"    Curvature DENSITY decreases with depth -> INTRINSIC nonlinearity at shallow layers!")
        elif k_corr > 0.3:
            log(f"    Curvature DENSITY increases with depth -> Deep layers intrinsically more nonlinear")
        else:
            log(f"    Curvature DENSITY constant -> No intrinsic nonlinearity difference")

        if j_corr < -0.3:
            log(f"    Local Jacobian DECREASES with depth -> Shallow layers amplify perturbations more")
            log(f"    -> This explains the J_forward finding from CCXLIV!")
        elif j_corr > 0.3:
            log(f"    Local Jacobian INCREASES with depth -> Opposite of CCXLIV!")
        else:
            log(f"    Local Jacobian constant -> No depth-dependent amplification")

    del model
    torch.cuda.empty_cache()

    log(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
