"""
Phase CCXLVIII: Hessian Directional Spectrum Analysis
=====================================================
CCXLVII confirmed: shallow layers have higher RELATIVE curvature.
But is this curvature uniformly distributed, or concentrated in specific directions?

Key questions:
1. Is shallow curvature "spread out" or "concentrated"?
2. Are there specific directions with extremely high curvature?
3. How does the curvature distribution differ between shallow and deep layers?

This test:
- At each layer, sample many random directions (e.g., 20-30)
- For each direction, compute relative curvature
- Analyze the distribution: mean, std, skew, kurtosis, max/min ratio
- Compare distribution shapes across layers
- Also: compute curvature along PRINCIPAL COMPONENTS of the activation space
  (if shallow layers have more "structured" curvature, it should align with PCs)
"""

import argparse
import json
import os
import numpy as np
from datetime import datetime
from scipy import stats

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
    "The wind blows hard", "He paints pictures", "They dance together",
    "The sun rises early", "She plays piano",
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


def measure_hessian_spectrum(model, tokenizer, model_key, n_samples=6, n_dirs=25,
                             eps_rel=0.05):
    """Measure Hessian curvature along many directions at each layer."""
    cfg = MODEL_CONFIGS[model_key]
    n_layers = cfg['n_layers']
    d_model = cfg['d_model']

    # Test at key positions
    test_layers = [
        max(1, n_layers // 8),      # shallow
        max(2, n_layers // 4),      # mid-shallow
        max(3, n_layers // 2),      # mid
        max(4, 3 * n_layers // 4),  # mid-deep
        max(5, n_layers - 2),       # deep
    ]
    test_layers = sorted(set([l for l in test_layers if 1 <= l < n_layers]))

    sentences = SENTENCES[:n_samples]
    results = {}

    for layer_idx in test_layers:
        target_layer = get_target_layer(model, layer_idx)
        all_curvatures = []  # [sample_idx * n_dirs]
        all_jac_norms = []
        all_act_norms = []
        all_jac_vectors = []  # Store Jacobian projections for each direction

        for sent_idx, sentence in enumerate(sentences):
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
            input_ids = inputs['input_ids'].to(model.device)
            attention_mask = inputs['attention_mask'].to(model.device)
            seq_len = attention_mask.sum().item()
            last_pos = seq_len - 1

            # Clean forward pass
            clean_input = [None]
            clean_output = [None]

            def make_capture_hooks():
                def pre_hook(module, args):
                    hidden = args[0]
                    clean_input[0] = hidden[0, last_pos, :].detach().cpu().float().clone()
                    return args

                def post_hook(module, args, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    clean_output[0] = hidden[0, last_pos, :].detach().cpu().float().clone()
                    return output

                return pre_hook, post_hook

            h1 = target_layer.register_forward_pre_hook(make_capture_hooks()[0])
            h2 = target_layer.register_forward_hook(make_capture_hooks()[1])

            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

            h1.remove()
            h2.remove()

            h_clean = clean_input[0].numpy()
            f_clean = clean_output[0].numpy()
            h_norm = np.linalg.norm(h_clean)
            f_norm = np.linalg.norm(f_clean)

            all_act_norms.append(h_norm)

            # Generate random directions (orthogonal via QR decomposition)
            raw_dirs = np.random.randn(n_dirs, d_model)
            Q, _ = np.linalg.qr(raw_dirs.T)
            directions = Q.T[:n_dirs]  # n_dirs x d_model, orthonormal

            sample_curvatures = []
            sample_jac_norms = []
            sample_jac_vectors = []

            for d_idx in range(n_dirs):
                v = directions[d_idx]
                v_tensor = torch.tensor(v, dtype=torch.float32, device=model.device)

                eps_hess = eps_rel * h_norm  # Relative perturbation

                # Forward +perturbation
                output_plus = [None]
                output_minus = [None]

                def make_hess_hooks(ve, ep, lp):
                    def pre_plus(module, args):
                        hidden = args[0]
                        perturbed = hidden.clone()
                        perturbed[0, lp, :] = perturbed[0, lp, :].float() + ep * ve.to(perturbed.dtype)
                        return (perturbed,) + args[1:]

                    def post_plus(module, args, output):
                        hidden = output[0] if isinstance(output, tuple) else output
                        output_plus[0] = hidden[0, lp, :].detach().cpu().float().clone()
                        return output

                    def pre_minus(module, args):
                        hidden = args[0]
                        perturbed = hidden.clone()
                        perturbed[0, lp, :] = perturbed[0, lp, :].float() - ep * ve.to(perturbed.dtype)
                        return (perturbed,) + args[1:]

                    def post_minus(module, args, output):
                        hidden = output[0] if isinstance(output, tuple) else output
                        output_minus[0] = hidden[0, lp, :].detach().cpu().float().clone()
                        return output

                    return pre_plus, post_plus, pre_minus, post_minus

                ph_p, poh_p, ph_m, poh_m = make_hess_hooks(v_tensor, eps_hess, last_pos)

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

                f_plus = output_plus[0].numpy()
                f_minus = output_minus[0].numpy()

                # Jacobian: (f+ - f-) / (2*eps)
                jac_v = (f_plus - f_minus) / (2 * eps_hess)
                jac_norm = np.linalg.norm(jac_v)

                # Hessian: (f+ - 2f0 + f-) / eps^2
                hess_v = (f_plus - 2 * f_clean + f_minus) / (eps_hess ** 2)
                hess_norm = np.linalg.norm(hess_v)

                # Relative curvature = ||H*v|| / ||J*v|| (dimensionless)
                rel_curv = hess_norm / (jac_norm + 1e-10)

                # Jacobian projection onto perturbation direction
                jac_proj = np.dot(jac_v, v)  # How much Jacobian aligns with v

                sample_curvatures.append(rel_curv)
                sample_jac_norms.append(jac_norm)
                sample_jac_vectors.append(jac_proj)

            all_curvatures.extend(sample_curvatures)
            all_jac_norms.extend(sample_jac_norms)
            all_jac_vectors.extend(sample_jac_vectors)

        # Analyze curvature distribution
        curvs = np.array(all_curvatures)
        jac_norms_arr = np.array(all_jac_norms)
        act_norm = np.mean(all_act_norms)

        # Distribution statistics
        curv_mean = np.mean(curvs)
        curv_std = np.std(curvs)
        curv_cv = curv_std / (curv_mean + 1e-10)  # Coefficient of variation
        curv_skew = stats.skew(curvs)
        curv_kurt = stats.kurtosis(curvs)  # Excess kurtosis (0 = normal)
        curv_max = np.max(curvs)
        curv_min = np.min(curvs)
        curv_range_ratio = curv_max / (curv_min + 1e-10)
        curv_p90 = np.percentile(curvs, 90)
        curv_p10 = np.percentile(curvs, 10)
        curv_p90_p10 = curv_p90 / (curv_p10 + 1e-10)  # P90/P10 ratio

        # Jacobian alignment analysis
        jac_proj_arr = np.array(all_jac_vectors)
        jac_proj_std = np.std(jac_proj_arr)
        jac_proj_mean = np.mean(np.abs(jac_proj_arr))

        results[layer_idx] = {
            'layer': layer_idx,
            'activation_norm': float(act_norm),
            'curv_mean': float(curv_mean),
            'curv_std': float(curv_std),
            'curv_cv': float(curv_cv),
            'curv_skew': float(curv_skew),
            'curv_kurtosis': float(curv_kurt),
            'curv_max': float(curv_max),
            'curv_min': float(curv_min),
            'curv_range_ratio': float(curv_range_ratio),
            'curv_p90_p10': float(curv_p90_p10),
            'jac_norm_mean': float(np.mean(jac_norms_arr)),
            'jac_proj_std': float(jac_proj_std),
            'jac_proj_mean': float(jac_proj_mean),
            'all_curvatures': curvs.tolist(),
        }

        print(f"  L{layer_idx:>2}: ||h||={act_norm:.1f}, curv_mean={curv_mean:.4f}, "
              f"CV={curv_cv:.3f}, skew={curv_skew:.2f}, kurt={curv_kurt:.2f}, "
              f"P90/P10={curv_p90_p10:.1f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'deepseek7b', 'glm4'])
    parser.add_argument('--n_samples', type=int, default=6)
    parser.add_argument('--n_dirs', type=int, default=25)
    parser.add_argument('--eps_rel', type=float, default=0.05)
    args = parser.parse_args()

    model_key = args.model
    cfg = MODEL_CONFIGS[model_key]

    out_dir = f"results/causal_fiber/{model_key}_ccxlviii"
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, 'run.log')

    def log(msg):
        print(msg, flush=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log(f"=== Phase CCXLVIII: Hessian Directional Spectrum ===")
    log(f"Model: {cfg['name']}, n_samples: {args.n_samples}, n_dirs: {args.n_dirs}, eps_rel: {args.eps_rel}")
    log(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    model, tokenizer = load_model(model_key)
    log(f"Model loaded successfully")

    results = measure_hessian_spectrum(model, tokenizer, model_key,
                                       n_samples=args.n_samples,
                                       n_dirs=args.n_dirs,
                                       eps_rel=args.eps_rel)

    # Analysis
    log(f"\n{'='*70}")
    log(f"ANALYSIS")
    log(f"{'='*70}")

    layers = []
    curv_means = []
    curv_cvs = []
    curv_skews = []
    curv_kurts = []
    curv_range_ratios = []
    curv_p90_p10s = []

    for l in sorted(results.keys()):
        d = results[l]
        layers.append(d['layer'])
        curv_means.append(d['curv_mean'])
        curv_cvs.append(d['curv_cv'])
        curv_skews.append(d['curv_skew'])
        curv_kurts.append(d['curv_kurtosis'])
        curv_range_ratios.append(d['curv_range_ratio'])
        curv_p90_p10s.append(d['curv_p90_p10'])

    layers_arr = np.array(layers, dtype=float)
    curv_means_arr = np.array(curv_means)
    curv_cvs_arr = np.array(curv_cvs)
    curv_skews_arr = np.array(curv_skews)
    curv_kurts_arr = np.array(curv_kurts)
    curv_range_arr = np.array(curv_range_ratios)
    curv_p90p10_arr = np.array(curv_p90_p10s)

    # Correlations with depth
    corr_mean = np.corrcoef(layers_arr, curv_means_arr)[0, 1] if len(layers_arr) > 2 else 0
    corr_cv = np.corrcoef(layers_arr, curv_cvs_arr)[0, 1] if len(layers_arr) > 2 else 0
    corr_skew = np.corrcoef(layers_arr, curv_skews_arr)[0, 1] if len(layers_arr) > 2 else 0
    corr_kurt = np.corrcoef(layers_arr, curv_kurts_arr)[0, 1] if len(layers_arr) > 2 else 0
    corr_range = np.corrcoef(layers_arr, curv_range_arr)[0, 1] if len(layers_arr) > 2 else 0
    corr_p90p10 = np.corrcoef(layers_arr, curv_p90p10_arr)[0, 1] if len(layers_arr) > 2 else 0

    log(f"\n  Correlation with Layer Depth:")
    log(f"  {'Metric':>20} {'Corr(L, metric)':>15} {'Interpretation':>40}")
    log(f"  {'curv_mean':>20} {corr_mean:>15.3f} {'Mean curvature (confirm CCXLVII)':>40}")
    log(f"  {'curv_CV':>20} {corr_cv:>15.3f} {'Curvature variability across dirs':>40}")
    log(f"  {'curv_skew':>20} {corr_skew:>15.3f} {'Asymmetry of curvature dist':>40}")
    log(f"  {'curv_kurtosis':>20} {corr_kurt:>15.3f} {'Heavy tails of curvature dist':>40}")
    log(f"  {'curv_range':>20} {corr_range:>15.3f} {'Max/Min curvature ratio':>40}")
    log(f"  {'curv_P90/P10':>20} {corr_p90p10:>15.3f} {'Interquartile range ratio':>40}")

    log(f"\n  Layer-by-Layer Distribution Summary:")
    log(f"  {'L':>4} {'||h||':>8} {'mean':>8} {'CV':>6} {'skew':>6} {'kurt':>6} {'P90/P10':>8} {'max/min':>8}")

    for l in sorted(results.keys()):
        d = results[l]
        log(f"  {d['layer']:>4} {d['activation_norm']:>8.1f} {d['curv_mean']:>8.4f} "
            f"{d['curv_cv']:>6.3f} {d['curv_skew']:>6.2f} {d['curv_kurtosis']:>6.2f} "
            f"{d['curv_p90_p10']:>8.1f} {d['curv_range_ratio']:>8.1f}")

    # Interpretation
    log(f"\n  INTERPRETATION:")

    if corr_cv > 0.3:
        log(f"  CV (curvature variability) INCREASES with depth => shallow curvature is MORE UNIFORM")
        log(f"  Shallow layers: curvature similar across all directions (isotropic)")
        log(f"  Deep layers: curvature varies a lot across directions (anisotropic)")
    elif corr_cv < -0.3:
        log(f"  CV (curvature variability) DECREASES with depth => shallow curvature is MORE ANISOTROPIC")
        log(f"  Shallow layers: curvature concentrated in specific directions")
        log(f"  Deep layers: curvature more uniform across directions")
    else:
        log(f"  CV roughly constant => curvature anisotropy similar across layers")

    if corr_kurt > 0.3:
        log(f"  Kurtosis INCREASES with depth => deep layers have heavier tails (more extreme curvatures)")
    elif corr_kurt < -0.3:
        log(f"  Kurtosis DECREASES with depth => shallow layers have heavier tails")
        log(f"  => Shallow layers have more 'outlier' high-curvature directions!")

    if corr_skew > 0.3:
        log(f"  Skewness INCREASES with depth => deep curvature distribution more right-skewed")
    elif corr_skew < -0.3:
        log(f"  Skewness DECREASES with depth => shallow curvature distribution more right-skewed")

    # Quantitative check: is shallow curvature truly "isotropic"?
    log(f"\n  Key Question: Is shallow curvature isotropic or anisotropic?")
    shallow = results[min(results.keys())]
    deep = results[max(results.keys())]
    log(f"  Shallow L{shallow['layer']}: CV={shallow['curv_cv']:.3f}, P90/P10={shallow['curv_p90_p10']:.1f}, range={shallow['curv_range_ratio']:.1f}")
    log(f"  Deep    L{deep['layer']}: CV={deep['curv_cv']:.3f}, P90/P10={deep['curv_p90_p10']:.1f}, range={deep['curv_range_ratio']:.1f}")

    if shallow['curv_cv'] < deep['curv_cv']:
        log(f"  => Shallow curvature MORE isotropic (uniform across directions)")
    else:
        log(f"  => Shallow curvature MORE anisotropic (concentrated in specific directions)")

    # Save results (without full curvature lists for compactness)
    save_results = {}
    for l, d in results.items():
        save_results[str(l)] = {k: v for k, v in d.items() if k != 'all_curvatures'}

    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump({'model': cfg['name'], 'results': save_results,
                   'n_dirs': args.n_dirs, 'eps_rel': args.eps_rel}, f, indent=2, default=str)

    # Also save full curvature distributions
    full_dists = {}
    for l, d in results.items():
        full_dists[str(l)] = d['all_curvatures']
    with open(os.path.join(out_dir, 'curvature_distributions.json'), 'w') as f:
        json.dump(full_dists, f)

    del model
    torch.cuda.empty_cache()

    log(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
