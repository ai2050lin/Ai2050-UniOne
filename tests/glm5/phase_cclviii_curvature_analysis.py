"""
Phase CCLVIII: Curvature Structure & Linear Range Analysis
============================================================
Exp1: Linear Range Quantification - find eps where response becomes non-linear
Exp2: Directional Hessian (2nd derivative) along feature vs random directions
Exp3: GLM4 SA Verification with 500+ random samples

Key hypothesis from CCLVII: Feature directions have LARGER linear range than
random directions, which explains E/R growth at large eps. This experiment
tests whether this is due to LOWER CURVATURE along feature directions.
"""
import argparse, os, torch, numpy as np
from pathlib import Path
from datetime import datetime
import json
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_CONFIGS = {
    'deepseek7b': {
        'name': 'DeepSeek-R1-Distill-Qwen-7B',
        'path': 'D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'n_layers': 28, 'd_model': 3584, 'dtype': '8bit',
    },
    'qwen3': {
        'name': 'Qwen3-4B',
        'path': 'D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c',
        'n_layers': 36, 'd_model': 2560, 'dtype': 'bf16',
    },
    'glm4': {
        'name': 'GLM4-9B-Chat',
        'path': 'D:/develop/model/hub/models--zai-org--glm-4-9b-chat-hf/download',
        'n_layers': 40, 'd_model': 4096, 'dtype': '8bit',
    },
}

TENSE_PAIRS = [
    ("She walks to the park","She walked to the park"),
    ("He reads books in the library","He read books in the library"),
    ("They play soccer after school","They played soccer after school"),
    ("The cat sleeps on the sofa","The cat slept on the sofa"),
    ("We cook dinner together","We cooked dinner together"),
    ("I think about this problem","I thought about this problem"),
    ("She catches the early train","She caught the early train"),
    ("The dog runs around the yard","The dog ran around the yard"),
]

QUESTION_PAIRS = [
    ("She walks to the park","Does she walk to the park"),
    ("He reads books in the library","Does he read books in the library"),
    ("They play soccer after school","Do they play soccer after school"),
    ("The cat sleeps on the sofa","Does the cat sleep on the sofa"),
    ("We cook dinner together","Do we cook dinner together"),
    ("She is happy today","Is she happy today"),
    ("He can swim very well","Can he swim very well"),
]


def load_model(model_key):
    config = MODEL_CONFIGS[model_key]
    path, dtype = config['path'], config['dtype']
    print(f"  Loading: {config['name']} ({dtype})")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if dtype == '8bit':
        bnb = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb, device_map="auto", trust_remote_code=True, local_files_only=True)
    elif dtype == 'bf16':
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, local_files_only=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True, local_files_only=True)
    model.eval()
    return model, tokenizer


def get_target_layer(model, layer_idx):
    mt = model.config.model_type
    if mt in ['qwen2','qwen3']:
        return model.model.layers[layer_idx]
    elif mt in ['chatglm','glm4']:
        return model.transformer.encoder.layers[layer_idx]
    return model.model.layers[layer_idx]


def get_layer_output(model, input_ids, layer_idx):
    h = None
    def hook_fn(module, input, output):
        nonlocal h
        h = (output[0] if isinstance(output, tuple) else output).detach().clone()
    handle = get_target_layer(model, layer_idx).register_forward_hook(hook_fn)
    with torch.no_grad():
        model(input_ids)
    handle.remove()
    return h


def measure_logits_delta(model, input_ids, layer_idx, direction, eps_abs, pos=-1):
    """Return full logits change vector from perturbation."""
    def hook_perturb(module, input, output):
        hs = (output[0] if isinstance(output, tuple) else output).clone()
        hs[0, pos, :] = hs[0, pos, :] + eps_abs * direction.to(hs.dtype)
        return (hs,) + output[1:] if isinstance(output, tuple) else hs
    with torch.no_grad():
        orig = model(input_ids).logits[0, pos, :].detach().clone()
    handle = get_target_layer(model, layer_idx).register_forward_hook(hook_perturb)
    with torch.no_grad():
        pert = model(input_ids).logits[0, pos, :].detach().clone()
    handle.remove()
    return pert - orig


def get_feat_direction(model, tokenizer, sa, sb, layer, device):
    enc_a = tokenizer(sa, return_tensors='pt').to(device)
    enc_b = tokenizer(sb, return_tensors='pt').to(device)
    h_a = get_layer_output(model, enc_a['input_ids'], layer)
    h_b = get_layer_output(model, enc_b['input_ids'], layer)
    pa, pb = h_a.shape[1]-1, h_b.shape[1]-1
    d = h_b[0,pb,:] - h_a[0,pa,:]
    if d.norm() < 1e-8:
        return None, None, None, None
    d = d / d.norm()
    return d, enc_a['input_ids'], pa, h_a[0,pa,:].norm().item()


# ====================================================================
# Experiment 1: Linear Range Quantification
# ====================================================================

def measure_linear_range(model, input_ids, layer_idx, direction, h_norm,
                         eps_grid=None, pos=-1, threshold=0.10):
    """Find the eps at which the response deviates >threshold from linearity.
    
    For a linear function: ||Dlogits|| / eps = constant
    Non-linearity: ||Dlogits|| / eps changes with eps
    
    Returns:
        linear_range_eps: the eps at which deviation exceeds threshold
        response_curve: dict of eps -> (eff, eff/eps, deviation)
    """
    if eps_grid is None:
        eps_grid = [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    
    # Measure at smallest eps as baseline
    eps_base = eps_grid[0]
    eps_abs_base = eps_base * h_norm
    delta_base = measure_logits_delta(model, input_ids, layer_idx, direction, eps_abs_base, pos)
    eff_base = delta_base.norm().item()
    sensitivity_base = eff_base / eps_base  # ||J·d|| estimate
    
    response_curve = {}
    linear_range_eps = eps_grid[-1]  # default to max
    
    for eps in eps_grid:
        eps_abs = eps * h_norm
        delta = measure_logits_delta(model, input_ids, layer_idx, direction, eps_abs, pos)
        eff = delta.norm().item()
        sensitivity = eff / eps
        
        # Deviation from linear prediction
        if sensitivity_base > 1e-10:
            deviation = abs(sensitivity - sensitivity_base) / sensitivity_base
        else:
            deviation = 0.0
        
        response_curve[eps] = {
            'eff': eff,
            'sensitivity': sensitivity,
            'deviation': deviation,
        }
        
        if deviation > threshold and eps < linear_range_eps:
            linear_range_eps = eps
    
    return linear_range_eps, response_curve


# ====================================================================
# Experiment 2: Directional Hessian
# ====================================================================

def measure_directional_hessian(model, input_ids, layer_idx, direction, h_norm, pos=-1):
    """Measure directional second derivative along a direction.
    
    For a smooth function f(h):
    f(h + eps*d) ~ f(h) + eps*J*d + (eps^2/2)*d^T*H*d

    Using finite differences:
    d^T*H*d ~ [f(h+2*eps*d) - 2*f(h+eps*d) + f(h)] / eps^2

    But since we measure Dlogits = f(h+eps*d) - f(h):
    D(2*eps) - 2*D(eps) = eps^2 * d^T*H*d + O(eps^3)

    So: d^T*H*d ~ [D(2*eps) - 2*D(eps)] / eps^2
    
    Returns:
        hess_norm: ||d^T H d|| (norm of the directional Hessian vector)
        linear_term: ||J·d|| (the linear response)
        relative_curvature: hess_norm / linear_term (dimensionless curvature measure)
    """
    eps = 0.01 * h_norm
    eps2 = 2 * eps
    
    # Linear response at eps
    delta1 = measure_logits_delta(model, input_ids, layer_idx, direction, eps, pos)
    linear_term = delta1.norm().item()
    
    # Response at 2*eps
    delta2 = measure_logits_delta(model, input_ids, layer_idx, direction, eps2, pos)
    
    # Directional Hessian: (delta2 - 2*delta1) / eps^2
    # Note: eps here is in absolute units
    hess_vec = (delta2 - 2 * delta1).float()
    hess_norm = hess_vec.norm().item() / (eps ** 2) if eps > 1e-10 else 0
    
    # Relative curvature: how much curvature relative to linear term
    # Normalized so it's dimensionless
    relative_curvature = hess_norm / linear_term * eps if linear_term > 1e-10 else 0
    # This is essentially ||d^T H d|| / ||J·d|| * eps
    # If < 1: linear dominates; if > 1: curvature dominates
    
    return hess_norm, linear_term, relative_curvature


# ====================================================================
# Experiment 3: GLM4 SA Verification with Large Sampling
# ====================================================================

def measure_sa_large_sampling(model, input_ids, layer_idx, d_feat, h_norm,
                              n_samples=500, eps_rel=0.01, pos=-1):
    """Precisely estimate SA = feat_eff / ||J||_2 using large random sampling.
    
    For 8bit models, we can't use autograd, so we use many random samples
    to estimate the true ||J||_2.
    """
    device = input_ids.device
    d_model = model.config.hidden_size
    eps_abs = eps_rel * h_norm
    
    # Feature efficiency
    fd = measure_logits_delta(model, input_ids, layer_idx, d_feat, eps_abs, pos)
    feat_eff = fd.norm().item()
    
    # Random directions
    rand_effs = []
    for i in range(n_samples):
        dr = torch.randn(d_model, device=device)
        dr = dr / dr.norm()
        rd = measure_logits_delta(model, input_ids, layer_idx, dr, eps_abs, pos)
        rand_effs.append(rd.norm().item())
        
        if (i+1) % 100 == 0:
            current_max = max(rand_effs)
            current_sa = feat_eff / current_max if current_max > 1e-10 else 0
            print(f"    Sample {i+1}/{n_samples}: max_rand={current_max:.2f}, SA={current_sa:.3f}")
    
    max_rand = max(rand_effs)
    mean_rand = np.mean(rand_effs)
    top10_rand = sorted(rand_effs, reverse=True)[:10]
    avg_top10 = np.mean(top10_rand)
    
    sa_max = feat_eff / max_rand if max_rand > 1e-10 else 0
    er = feat_eff / mean_rand if mean_rand > 1e-10 else 0
    
    # Also estimate ||J||_F from mean: ||J||_F ~ E[||J*d||] * sqrt(d)
    # And ||J||_2 / ||J||_F tells us spectral concentration
    
    return {
        'feat_eff': feat_eff,
        'max_rand': max_rand,
        'mean_rand': mean_rand,
        'avg_top10': avg_top10,
        'SA_max': sa_max,
        'E/R': er,
        'n_samples': n_samples,
        # Estimate of true ||J||_2 using extreme value theory
        # For n samples from |J·d| (which follows ||J||_2 * sin(theta) distribution),
        # the expected maximum approaches ||J||_2 as n → ∞
    }


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['deepseek7b','qwen3','glm4'])
    parser.add_argument('--experiments', nargs='+', default=['1','2','3'])
    parser.add_argument('--n_random', type=int, default=20)
    parser.add_argument('--n_sentences', type=int, default=3)
    parser.add_argument('--n_sa_samples', type=int, default=500)
    args = parser.parse_args()

    out_dir = Path(f'results/causal_fiber/{args.model}_cclviii')
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / 'console.log'

    def log(msg):
        print(msg, flush=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log(f"Phase CCLVIII: Curvature Structure and Linear Range Analysis")
    log(f"Model: {args.model}, Experiments: {args.experiments}")
    log(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    log("Loading model...")
    model, tokenizer = load_model(args.model)
    cfg = MODEL_CONFIGS[args.model]
    n_layers, d_model = cfg['n_layers'], cfg['d_model']
    device = next(model.parameters()).device
    is_bf16 = (cfg['dtype'] == 'bf16')
    log(f"Loaded: {cfg['name']}, {n_layers}L, d={d_model}, dev={device}, bf16={is_bf16}")

    # Select layers
    if n_layers == 36: layers = [5, 15, 24, 34]
    elif n_layers == 28: layers = [4, 11, 18, 26]
    elif n_layers == 40: layers = [6, 16, 27, 38]
    else: layers = sorted(set(np.linspace(1, n_layers-2, 4, dtype=int).tolist()))
    log(f"Layers: {layers}")

    tense_pairs = TENSE_PAIRS[:args.n_sentences]
    question_pairs = QUESTION_PAIRS[:min(args.n_sentences, len(QUESTION_PAIRS))]
    features = [('tense', tense_pairs), ('question', question_pairs)]

    all_results = {}

    # ==================================================================
    # EXPERIMENT 1: Linear Range Quantification
    # ==================================================================
    if '1' in args.experiments:
        log(f"\n{'='*70}")
        log(f"EXPERIMENT 1: Linear Range Quantification")
        log(f"Find eps where ||Dlogits||/eps deviates >10% from baseline")
        log(f"{'='*70}")

        exp1 = {}
        eps_grid = [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]

        for feat_name, pairs in features:
            log(f"\n--- {feat_name.upper()} ---")
            exp1[feat_name] = {}

            for layer in layers:
                feat_ranges, rand_ranges = [], []
                feat_curves, rand_curves = [], []

                for si, (sa, sb) in enumerate(pairs):
                    d_feat, ids_a, pa, h_norm = get_feat_direction(model, tokenizer, sa, sb, layer, device)
                    if d_feat is None: continue

                    # Feature direction linear range
                    lr_feat, rc_feat = measure_linear_range(
                        model, ids_a, layer, d_feat, h_norm, eps_grid, pos=pa, threshold=0.10)
                    feat_ranges.append(lr_feat)
                    feat_curves.append(rc_feat)
                    log(f"  L{layer} P{si+1} feat: linear_range={lr_feat:.3f} ({lr_feat*100:.1f}%)")

                    # Random direction linear ranges
                    for ri in range(args.n_random):
                        dr = torch.randn(d_model, device=device)
                        dr = dr / dr.norm()
                        lr_rand, rc_rand = measure_linear_range(
                            model, ids_a, layer, dr, h_norm, eps_grid, pos=pa, threshold=0.10)
                        rand_ranges.append(lr_rand)
                        rand_curves.append(rc_rand)

                if feat_ranges:
                    avg_feat_lr = np.mean(feat_ranges)
                    avg_rand_lr = np.mean(rand_ranges) if rand_ranges else 0
                    std_rand_lr = np.std(rand_ranges) if rand_ranges else 0
                    lr_ratio = avg_feat_lr / avg_rand_lr if avg_rand_lr > 1e-10 else float('inf')

                    exp1[feat_name][layer] = {
                        'feat_linear_range': float(avg_feat_lr),
                        'rand_linear_range_mean': float(avg_rand_lr),
                        'rand_linear_range_std': float(std_rand_lr),
                        'lr_ratio': float(lr_ratio),
                        'n_rand': len(rand_ranges),
                    }

                    log(f"  => L{layer}: feat_LR={avg_feat_lr:.3f}, rand_LR={avg_rand_lr:.3f}±{std_rand_lr:.3f}, ratio={lr_ratio:.2f}")

                    # Print detailed response curve for first pair
                    if feat_curves:
                        log(f"  Response curve (first pair):")
                        log(f"    {'eps':>6} {'feat_eff':>10} {'feat/eps':>10} {'feat_dev':>10} {'rand_eff':>10} {'rand/eps':>10} {'rand_dev':>10}")
                        for eps in eps_grid:
                            fc = feat_curves[0][eps]
                            # Average rand curve
                            rc_avg = {}
                            for key in ['eff','sensitivity','deviation']:
                                rc_avg[key] = np.mean([rc[eps][key] for rc in rand_curves[:min(5,len(rand_curves))]])
                            log(f"    {eps:>6.3f} {fc['eff']:>10.4f} {fc['sensitivity']:>10.2f} {fc['deviation']:>10.3f} "
                                f"{rc_avg['eff']:>10.4f} {rc_avg['sensitivity']:>10.2f} {rc_avg['deviation']:>10.3f}")

        log(f"\n  EXP1 SUMMARY: Linear Range (eps at 10% deviation)")
        log(f"  {'Feat':>8} {'L':>4} {'feat_LR':>10} {'rand_LR':>10} {'ratio':>8} {'Interpretation':>20}")
        for fn, _ in features:
            for l in layers:
                if l in exp1[fn]:
                    r = exp1[fn][l]
                    interp = 'feat>rand' if r['lr_ratio'] > 1.2 else ('feat<rand' if r['lr_ratio'] < 0.8 else 'equal')
                    log(f"  {fn:>8} {l:>4} {r['feat_linear_range']:>10.3f} {r['rand_linear_range_mean']:>10.3f} {r['lr_ratio']:>8.2f} {interp:>20}")

        all_results['exp1'] = exp1
        with open(out_dir / 'exp1_linear_range.json', 'w') as f:
            json.dump(exp1, f, indent=2, default=str)

    # ==================================================================
    # EXPERIMENT 2: Directional Hessian
    # ==================================================================
    if '2' in args.experiments:
        log(f"\n{'='*70}")
        log(f"EXPERIMENT 2: Directional Hessian (Curvature Analysis)")
        log(f"Measure d^T H d = [D(2e) - 2*D(e)] / e^2 along directions")
        log(f"{'='*70}")

        exp2 = {}

        for feat_name, pairs in features:
            log(f"\n--- {feat_name.upper()} ---")
            exp2[feat_name] = {}

            for layer in layers:
                feat_hess, feat_lin = [], []
                rand_hess, rand_lin, rand_relcurv = [], [], []

                for si, (sa, sb) in enumerate(pairs):
                    d_feat, ids_a, pa, h_norm = get_feat_direction(model, tokenizer, sa, sb, layer, device)
                    if d_feat is None: continue

                    # Feature direction Hessian
                    hn, lt, rc = measure_directional_hessian(
                        model, ids_a, layer, d_feat, h_norm, pos=pa)
                    feat_hess.append(hn)
                    feat_lin.append(lt)
                    log(f"  L{layer} P{si+1} feat: hess={hn:.2f}, linear={lt:.4f}, rel_curv={rc:.4f}")

                    # Random directions
                    for ri in range(args.n_random):
                        dr = torch.randn(d_model, device=device)
                        dr = dr / dr.norm()
                        hn_r, lt_r, rc_r = measure_directional_hessian(
                            model, ids_a, layer, dr, h_norm, pos=pa)
                        rand_hess.append(hn_r)
                        rand_lin.append(lt_r)
                        rand_relcurv.append(rc_r)

                if feat_hess:
                    avg_fh = np.mean(feat_hess)
                    avg_fl = np.mean(feat_lin)
                    avg_rh = np.mean(rand_hess) if rand_hess else 0
                    avg_rl = np.mean(rand_lin) if rand_lin else 0
                    std_rh = np.std(rand_hess) if rand_hess else 0
                    avg_frc = np.mean([rc for rc in [feat_hess[i]/feat_lin[i] if feat_lin[i] > 0 else 0 for i in range(len(feat_hess))]])
                    avg_rrc = np.mean(rand_relcurv) if rand_relcurv else 0

                    hess_ratio = avg_fh / avg_rh if avg_rh > 1e-10 else float('inf')

                    exp2[feat_name][layer] = {
                        'feat_hess': float(avg_fh),
                        'feat_linear': float(avg_fl),
                        'rand_hess_mean': float(avg_rh),
                        'rand_hess_std': float(std_rh),
                        'rand_linear_mean': float(avg_rl),
                        'hess_ratio': float(hess_ratio),
                        'feat_rel_curv': float(avg_frc),
                        'rand_rel_curv': float(avg_rrc),
                    }

                    log(f"  => L{layer}: feat_hess={avg_fh:.2f}, rand_hess={avg_rh:.2f}±{std_rh:.2f}, "
                        f"ratio={hess_ratio:.3f}, rel_curv: feat={avg_frc:.4f} rand={avg_rrc:.4f}")

        log(f"\n  EXP2 SUMMARY: Directional Hessian")
        log(f"  {'Feat':>8} {'L':>4} {'feat_H':>10} {'rand_H':>10} {'H_ratio':>10} {'feat_RC':>10} {'rand_RC':>10} {'Interp':>15}")
        for fn, _ in features:
            for l in layers:
                if l in exp2[fn]:
                    r = exp2[fn][l]
                    interp = 'feat<rand' if r['hess_ratio'] < 0.8 else ('feat>rand' if r['hess_ratio'] > 1.2 else 'equal')
                    log(f"  {fn:>8} {l:>4} {r['feat_hess']:>10.2f} {r['rand_hess_mean']:>10.2f} "
                        f"{r['hess_ratio']:>10.3f} {r['feat_rel_curv']:>10.4f} {r['rand_rel_curv']:>10.4f} {interp:>15}")

        all_results['exp2'] = exp2
        with open(out_dir / 'exp2_directional_hessian.json', 'w') as f:
            json.dump(exp2, f, indent=2, default=str)

    # ==================================================================
    # EXPERIMENT 3: GLM4 SA Verification with Large Sampling
    # ==================================================================
    if '3' in args.experiments:
        log(f"\n{'='*70}")
        log(f"EXPERIMENT 3: SA Verification with {args.n_sa_samples} Random Samples")
        log(f"{'='*70}")

        exp3 = {}

        for feat_name, pairs in features:
            log(f"\n--- {feat_name.upper()} ---")
            exp3[feat_name] = {}

            for layer in layers:
                for si, (sa, sb) in enumerate(pairs[:1]):  # Use first pair only (expensive)
                    d_feat, ids_a, pa, h_norm = get_feat_direction(model, tokenizer, sa, sb, layer, device)
                    if d_feat is None: continue

                    log(f"  L{layer}: Running {args.n_sa_samples} random samples...")
                    sa_data = measure_sa_large_sampling(
                        model, ids_a, layer, d_feat, h_norm,
                        n_samples=args.n_sa_samples, eps_rel=0.01, pos=pa)

                    exp3[feat_name][layer] = sa_data
                    log(f"  => L{layer}: feat={sa_data['feat_eff']:.4f}, max_rand={sa_data['max_rand']:.4f}, "
                        f"SA={sa_data['SA_max']:.3f}, E/R={sa_data['E/R']:.2f}, "
                        f"avg_top10={sa_data['avg_top10']:.4f}")

        log(f"\n  EXP3 SUMMARY: SA with Large Sampling")
        log(f"  {'Feat':>8} {'L':>4} {'feat_eff':>10} {'max_rand':>10} {'avg_top10':>10} {'SA':>8} {'E/R':>8} {'n':>6}")
        for fn, _ in features:
            for l in layers:
                if l in exp3[fn]:
                    r = exp3[fn][l]
                    log(f"  {fn:>8} {l:>4} {r['feat_eff']:>10.4f} {r['max_rand']:>10.4f} "
                        f"{r['avg_top10']:>10.4f} {r['SA_max']:>8.3f} {r['E/R']:>8.2f} {r['n_samples']:>6}")

        all_results['exp3'] = exp3
        with open(out_dir / 'exp3_sa_verification.json', 'w') as f:
            json.dump(exp3, f, indent=2, default=str)

    log(f"\nAll experiments done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
