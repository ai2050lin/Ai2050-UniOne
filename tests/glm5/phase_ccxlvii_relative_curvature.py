"""
Phase CCXLVII: Relative Curvature & Multi-eps Hessian Verification
====================================================================
CRITICAL: CCXLVI showed that normalizing by ||h|| REVERSES CCXLV conclusion.
But is dividing by ||h|| the right normalization?

The question: what is the "intrinsic" nonlinearity of a layer?

Key insight: If we perturb the input RELATIVE to its norm (i.e., ε% perturbation),
the Hessian response should be measured relative to the function's output scale.

Two ways to think about it:
1. Absolute perturbation: add δ to h, measure ||f(h+δ) - 2f(h) + f(h-δ)||/ε²
   → This is what CCXLV did with fixed ε=1.0
   → Deep layers have larger ||h||, so ε=1.0 is a smaller RELATIVE perturbation
   → Hessian appears larger because the function's scale is larger

2. Relative perturbation: add ε% * ||h|| to h
   → Same relative perturbation at every layer
   → More fair comparison across layers of different scales

Also: Multi-eps verification to check if Hessian estimate is stable.
If Hessian estimate depends strongly on ε, it's not reliable.

This test:
- Uses relative eps = ε_rel * ||h|| (e.g., 1%, 5%, 10%)
- Compares absolute vs relative Hessian
- Tests multiple eps values for stability
- Only runs on a subset of layers (speed)
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
    "She walks to school", "He runs in the park", "They play football",
    "The cat sleeps on the mat", "She sings beautifully", "He writes a letter",
    "They travel abroad", "The dog barks loudly", "She cooks dinner",
    "He drives carefully", "The bird flies south", "She reads the book",
    "They build houses", "The river flows north", "She teaches mathematics",
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


def measure_relative_geometry(model, tokenizer, model_key, n_samples=8, n_dirs=3,
                              eps_rel_list=None):
    """Measure Hessian with RELATIVE perturbation (eps proportional to ||h||)."""
    if eps_rel_list is None:
        eps_rel_list = [0.01, 0.05, 0.10, 0.20]  # 1%, 5%, 10%, 20% of ||h||

    cfg = MODEL_CONFIGS[model_key]
    n_layers = cfg['n_layers']
    d_model = cfg['d_model']

    # Test at key positions: shallow, mid-shallow, mid, mid-deep, deep
    test_layers = [
        max(1, n_layers // 8),
        max(2, n_layers // 4),
        max(3, n_layers // 2),
        max(4, 3 * n_layers // 4),
        max(5, n_layers - 2),
    ]
    test_layers = sorted(set([l for l in test_layers if 1 <= l < n_layers]))

    sentences = SENTENCES[:n_samples]

    results = {}

    for layer_idx in test_layers:
        target_layer = get_target_layer(model, layer_idx)

        layer_results = {
            'layer': layer_idx,
            'abs_hess': {},  # eps_abs -> {eps_rel: hess_val}
            'rel_hess': {},
            'jac_vals': [],
            'act_norms': [],
        }

        for sent_idx, sentence in enumerate(sentences):
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
            input_ids = inputs['input_ids'].to(model.device)
            attention_mask = inputs['attention_mask'].to(model.device)
            seq_len = attention_mask.sum().item()
            last_pos = seq_len - 1

            # Clean forward pass
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

            h1 = target_layer.register_forward_pre_hook(make_capture_hooks(layer_idx)[0])
            h2 = target_layer.register_forward_hook(make_capture_hooks(layer_idx)[1])

            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

            h1.remove()
            h2.remove()

            h_clean = clean_input[0].numpy()
            f_clean = clean_output[0].numpy()
            h_norm = np.linalg.norm(h_clean)
            f_norm = np.linalg.norm(f_clean)
            act_norms_val = h_norm

            layer_results['act_norms'].append(act_norms_val)

            for d_idx in range(n_dirs):
                v = np.random.randn(d_model)
                v = v / (np.linalg.norm(v) + 1e-10)
                v_tensor = torch.tensor(v, dtype=torch.float32, device=model.device)

                # Jacobian
                eps_jac = max(0.01 * h_norm, 0.01)  # 1% relative
                output_plus_j = [None]
                output_minus_j = [None]

                def make_jac_hooks(ve, ep, lp):
                    def pre_plus(module, args):
                        hidden = args[0]
                        perturbed = hidden.clone()
                        perturbed[0, lp, :] = perturbed[0, lp, :].float() + ep * ve.to(perturbed.dtype)
                        return (perturbed,) + args[1:]

                    def post_plus(module, args, output):
                        hidden = output[0] if isinstance(output, tuple) else output
                        output_plus_j[0] = hidden[0, lp, :].detach().cpu().float().clone()
                        return output

                    def pre_minus(module, args):
                        hidden = args[0]
                        perturbed = hidden.clone()
                        perturbed[0, lp, :] = perturbed[0, lp, :].float() - ep * ve.to(perturbed.dtype)
                        return (perturbed,) + args[1:]

                    def post_minus(module, args, output):
                        hidden = output[0] if isinstance(output, tuple) else output
                        output_minus_j[0] = hidden[0, lp, :].detach().cpu().float().clone()
                        return output

                    return pre_plus, post_plus, pre_minus, post_minus

                ph_p, poh_p, ph_m, poh_m = make_jac_hooks(v_tensor, eps_jac, last_pos)
                h3 = target_layer.register_forward_pre_hook(ph_p)
                h4 = target_layer.register_forward_hook(poh_p)
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                h3.remove()
                h4.remove()

                h5 = target_layer.register_forward_pre_hook(ph_m)
                h6 = target_layer.register_forward_hook(poh_m)
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                h5.remove()
                h6.remove()

                f_plus_j = output_plus_j[0].numpy()
                f_minus_j = output_minus_j[0].numpy()
                jac_v = (f_plus_j - f_minus_j) / (2 * eps_jac)
                jac_norm = np.linalg.norm(jac_v)
                layer_results['jac_vals'].append(jac_norm)

                # Hessian with multiple relative eps values
                for eps_rel in eps_rel_list:
                    eps_hess = eps_rel * h_norm  # Relative perturbation

                    output_plus_h = [None]
                    output_minus_h = [None]

                    def make_hess_hooks(ve, ep, lp):
                        def pre_plus(module, args):
                            hidden = args[0]
                            perturbed = hidden.clone()
                            perturbed[0, lp, :] = perturbed[0, lp, :].float() + ep * ve.to(perturbed.dtype)
                            return (perturbed,) + args[1:]

                        def post_plus(module, args, output):
                            hidden = output[0] if isinstance(output, tuple) else output
                            output_plus_h[0] = hidden[0, lp, :].detach().cpu().float().clone()
                            return output

                        def pre_minus(module, args):
                            hidden = args[0]
                            perturbed = hidden.clone()
                            perturbed[0, lp, :] = perturbed[0, lp, :].float() - ep * ve.to(perturbed.dtype)
                            return (perturbed,) + args[1:]

                        def post_minus(module, args, output):
                            hidden = output[0] if isinstance(output, tuple) else output
                            output_minus_h[0] = hidden[0, lp, :].detach().cpu().float().clone()
                            return output

                        return pre_plus, post_plus, pre_minus, post_minus

                    ph_p2, poh_p2, ph_m2, poh_m2 = make_hess_hooks(v_tensor, eps_hess, last_pos)
                    h7 = target_layer.register_forward_pre_hook(ph_p2)
                    h8 = target_layer.register_forward_hook(poh_p2)
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    h7.remove()
                    h8.remove()

                    h9 = target_layer.register_forward_pre_hook(ph_m2)
                    h10 = target_layer.register_forward_hook(poh_m2)
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    h9.remove()
                    h10.remove()

                    f_plus_h = output_plus_h[0].numpy()
                    f_minus_h = output_minus_h[0].numpy()

                    # Absolute Hessian: ||f(h+eps*v) - 2f(h) + f(h-eps*v)|| / eps^2
                    hess_abs = np.linalg.norm(f_plus_h - 2 * f_clean + f_minus_h) / (eps_hess ** 2)

                    # Relative Hessian: normalize by output norm
                    # This measures "fractional curvature" = how much the function curves
                    # relative to its own scale
                    hess_rel = hess_abs * (h_norm / (f_norm + 1e-10))

                    if eps_rel not in layer_results['abs_hess']:
                        layer_results['abs_hess'][eps_rel] = []
                        layer_results['rel_hess'][eps_rel] = []

                    layer_results['abs_hess'][eps_rel].append(hess_abs)
                    layer_results['rel_hess'][eps_rel].append(hess_rel)

        # Aggregate
        mean_act = np.mean(layer_results['act_norms'])
        mean_jac = np.mean(layer_results['jac_vals'])

        agg = {
            'layer': layer_idx,
            'activation_norm': float(mean_act),
            'jacobian': float(mean_jac),
        }

        for eps_rel in eps_rel_list:
            abs_vals = layer_results['abs_hess'][eps_rel]
            rel_vals = layer_results['rel_hess'][eps_rel]
            agg[f'abs_hess_eps{eps_rel}'] = float(np.mean(abs_vals))
            agg[f'rel_hess_eps{eps_rel}'] = float(np.mean(rel_vals))
            # kappa = H/J (CCXLV style)
            agg[f'kappa_eps{eps_rel}'] = float(np.mean(abs_vals) / (mean_jac + 1e-10))
            # kappa_norm = H/(J*h) (CCXLVI style)
            agg[f'kappa_norm_eps{eps_rel}'] = float(np.mean(abs_vals) / ((mean_jac + 1e-10) * (mean_act + 1e-10)))
            # relative curvature = H*h/f (fractional curvature)
            agg[f'rel_curv_eps{eps_rel}'] = float(np.mean(rel_vals))

        results[layer_idx] = agg

        # Print progress
        print(f"  Layer {layer_idx:>2}: ||h||={mean_act:.2f}, ||J||={mean_jac:.4f}", end='')
        for eps_rel in eps_rel_list:
            abs_h = agg[f'abs_hess_eps{eps_rel}']
            kn = agg[f'kappa_norm_eps{eps_rel}']
            rc = agg[f'rel_curv_eps{eps_rel}']
            print(f", H({eps_rel:.0%})={abs_h:.4f}, kn={kn:.6f}, rc={rc:.6f}", end='')
        print()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'deepseek7b', 'glm4'])
    parser.add_argument('--n_samples', type=int, default=8)
    parser.add_argument('--n_dirs', type=int, default=3)
    args = parser.parse_args()

    model_key = args.model
    cfg = MODEL_CONFIGS[model_key]

    out_dir = f"results/causal_fiber/{model_key}_ccxlvii"
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, 'run.log')

    def log(msg):
        print(msg, flush=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log(f"=== Phase CCXLVII: Relative Curvature & Multi-eps Verification ===")
    log(f"Model: {cfg['name']}, n_samples: {args.n_samples}, n_dirs: {args.n_dirs}")
    log(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    model, tokenizer = load_model(model_key)
    log(f"Model loaded successfully")

    eps_rel_list = [0.01, 0.05, 0.10, 0.20]
    results = measure_relative_geometry(model, tokenizer, model_key,
                                        n_samples=args.n_samples, n_dirs=args.n_dirs,
                                        eps_rel_list=eps_rel_list)

    # Analysis
    log(f"\n{'='*70}")
    log(f"ANALYSIS")
    log(f"{'='*70}")

    layers = []
    act_norms = []
    jac_norms = []

    for l in sorted(results.keys()):
        d = results[l]
        layers.append(d['layer'])
        act_norms.append(d['activation_norm'])
        jac_norms.append(d['jacobian'])

    layers = np.array(layers, dtype=float)
    act_norms = np.array(act_norms)
    jac_norms = np.array(jac_norms)

    # For each eps, compute correlations
    log(f"\n  Multi-eps Correlation Analysis:")
    log(f"  {'eps_rel':>8} {'Corr(L,H_abs)':>14} {'Corr(L,kappa)':>14} "
        f"{'Corr(L,kn)':>12} {'Corr(L,rc)':>12} {'Stability':>10}")

    for eps_rel in eps_rel_list:
        key_abs = f'abs_hess_eps{eps_rel}'
        key_kappa = f'kappa_eps{eps_rel}'
        key_kn = f'kappa_norm_eps{eps_rel}'
        key_rc = f'rel_curv_eps{eps_rel}'

        abs_vals = np.array([results[l][key_abs] for l in sorted(results.keys())])
        kappa_vals = np.array([results[l][key_kappa] for l in sorted(results.keys())])
        kn_vals = np.array([results[l][key_kn] for l in sorted(results.keys())])
        rc_vals = np.array([results[l][key_rc] for l in sorted(results.keys())])

        corr_abs = np.corrcoef(layers, abs_vals)[0, 1] if len(layers) > 2 else 0
        corr_kappa = np.corrcoef(layers, kappa_vals)[0, 1] if len(layers) > 2 else 0
        corr_kn = np.corrcoef(layers, kn_vals)[0, 1] if len(layers) > 2 else 0
        corr_rc = np.corrcoef(layers, rc_vals)[0, 1] if len(layers) > 2 else 0

        # Stability: compare with next eps
        if eps_rel < max(eps_rel_list):
            next_eps = [e for e in eps_rel_list if e > eps_rel][0]
            next_abs = np.array([results[l][f'abs_hess_eps{next_eps}'] for l in sorted(results.keys())])
            # Ratio of Hessians at different eps (should be ~1 for stable estimate)
            ratio = np.mean(abs_vals / (next_abs + 1e-10))
            stability = f"ratio={ratio:.2f}"
        else:
            stability = "N/A"

        log(f"  {eps_rel:>8.2f} {corr_abs:>14.3f} {corr_kappa:>14.3f} "
            f"{corr_kn:>12.3f} {corr_rc:>12.3f} {stability:>10}")

    # Key comparison: CCXLV vs CCXLVI vs CCXLVII
    log(f"\n  KEY COMPARISON (using eps_rel=5%):")
    eps_main = 0.05
    key_abs = f'abs_hess_eps{eps_main}'
    key_kappa = f'kappa_eps{eps_main}'
    key_kn = f'kappa_norm_eps{eps_main}'
    key_rc = f'rel_curv_eps{eps_main}'

    abs_vals = np.array([results[l][key_abs] for l in sorted(results.keys())])
    kappa_vals = np.array([results[l][key_kappa] for l in sorted(results.keys())])
    kn_vals = np.array([results[l][key_kn] for l in sorted(results.keys())])
    rc_vals = np.array([results[l][key_rc] for l in sorted(results.keys())])

    corr_abs = np.corrcoef(layers, abs_vals)[0, 1] if len(layers) > 2 else 0
    corr_kappa = np.corrcoef(layers, kappa_vals)[0, 1] if len(layers) > 2 else 0
    corr_kn = np.corrcoef(layers, kn_vals)[0, 1] if len(layers) > 2 else 0
    corr_rc = np.corrcoef(layers, rc_vals)[0, 1] if len(layers) > 2 else 0

    log(f"  Corr(layer, ||H||)        = {corr_abs:>7.3f}  [Absolute Hessian — CCXLV]")
    log(f"  Corr(layer, kappa=H/J)    = {corr_kappa:>7.3f}  [Curvature density — CCXLV]")
    log(f"  Corr(layer, H/(J*h))     = {corr_kn:>7.3f}  [Normalized — CCXLVI]")
    log(f"  Corr(layer, rel_curv)    = {corr_rc:>7.3f}  [Relative curvature — CCXLVII]")

    log(f"\n  INTERPRETATION:")
    if corr_kn < -0.3 and corr_rc < -0.3:
        log(f"  Both normalized metrics agree: SHALLOW layers have higher intrinsic nonlinearity!")
        log(f"  The 'deep nonlinearity' from CCXLV was entirely an activation norm artifact.")
    elif corr_kn < -0.3 and corr_rc > -0.3:
        log(f"  Normalized curvature (H/Jh) says shallow > deep, but relative curvature disagrees.")
        log(f"  Need to think about which normalization is correct.")
    elif corr_kn > 0.3 and corr_rc > 0.3:
        log(f"  Both metrics say DEEP layers have higher nonlinearity even after normalization.")
        log(f"  CCXLV conclusion holds even with relative perturbation!")

    # Print layer-by-layer
    log(f"\n  Layer-by-Layer (eps_rel=5%):")
    log(f"  {'L':>4} {'||h||':>8} {'||J||':>8} {'||H||':>10} {'kappa':>8} {'kappa_n':>10} {'rel_c':>10}")
    for l in sorted(results.keys()):
        d = results[l]
        log(f"  {d['layer']:>4} {d['activation_norm']:>8.2f} {d['jacobian']:>8.4f} "
            f"{d[key_abs]:>10.4f} {d[key_kappa]:>8.4f} {d[key_kn]:>10.6f} {d[key_rc]:>10.6f}")

    # Save results
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump({'model': cfg['name'], 'results': {str(k): v for k, v in results.items()},
                   'eps_rel_list': eps_rel_list}, f, indent=2, default=str)

    del model
    torch.cuda.empty_cache()

    log(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
