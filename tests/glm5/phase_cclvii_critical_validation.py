"""
Phase CCLVII: Critical Validation Experiments
==============================================
Exp1: True Jacobian SVD via Power Iteration (bf16) / Large Sampling (8bit)
Exp2: Task Subspace Projection (PCA decomposition)
Exp3: Epsilon Sweep (Linearity Check)
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

PCA_SENTENCES = [
    "The weather is nice today","I love reading books","She went to the store",
    "They are playing outside","The movie was interesting","We should leave early",
    "He finished his homework","The food tastes great","She speaks three languages",
    "The train arrived late","Birds fly in the sky","The house is very old",
    "I need more time","They built a new school","The concert starts at eight",
    "Programming requires patience","Rivers flow to the sea","Mountains rise above clouds",
    "Knowledge grows with study","Art reflects human experience",
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
# Experiment 1: True Jacobian SVD
# ====================================================================

def power_iteration_jacobian(model, input_ids, layer_idx, n_iter=15, eps_rel=0.01, pos=-1, is_bf16=True):
    """Compute top singular value ||J||_2 via power iteration.
    bf16: uses autograd for J^T @ u (precise).
    8bit: uses large random sampling (200 samples).
    """
    device = input_ids.device
    d_model = model.config.hidden_size
    h_orig = get_layer_output(model, input_ids, layer_idx)
    h_norm = h_orig[0, pos, :].norm().item()
    eps_abs = eps_rel * h_norm

    if is_bf16:
        # Power iteration with autograd
        v = torch.randn(d_model, device=device, dtype=torch.float32)
        v = v / v.norm()
        sigma = 0.0

        for it in range(n_iter):
            # J @ v via finite difference
            jv = measure_logits_delta(model, input_ids, layer_idx, v, eps_abs, pos).float()
            sigma_new = jv.norm().item()
            if sigma_new < 1e-10:
                break
            u = jv / sigma_new
            sigma = sigma_new

            # J^T @ u via autograd hook-replacement trick
            h_rep_holder = [None]
            def replace_hook(module, input, output):
                h_cap = (output[0] if isinstance(output, tuple) else output).detach().clone()
                h_rep = h_cap.clone().detach().requires_grad_(True)
                h_rep_holder[0] = h_rep
                return (h_rep,) + output[1:] if isinstance(output, tuple) else h_rep

            handle = get_target_layer(model, layer_idx).register_forward_hook(replace_hook)
            logits = model(input_ids).logits
            handle.remove()

            target = logits[0, pos, :].float()
            loss = (u.to(target.device) @ target).sum()
            loss.backward()

            vjp = h_rep_holder[0].grad[0, pos, :].float().detach().clone()
            vjp_norm = vjp.norm().item()
            if vjp_norm < 1e-10:
                break
            v = vjp / vjp_norm
            model.zero_grad()

        return sigma
    else:
        # Large random sampling for 8bit
        max_norm = 0.0
        for _ in range(200):
            dr = torch.randn(d_model, device=device)
            dr = dr / dr.norm()
            delta = measure_logits_delta(model, input_ids, layer_idx, dr, eps_abs, pos)
            n = delta.norm().item()
            if n > max_norm:
                max_norm = n
        return max_norm


# ====================================================================
# Experiment 2: PCA Task Subspace
# ====================================================================

def collect_activations(model, tokenizer, sentences, layer_idx, pos=-1):
    device = next(model.parameters()).device
    acts = []
    for s in sentences:
        enc = tokenizer(s, return_tensors='pt').to(device)
        h = get_layer_output(model, enc['input_ids'], layer_idx)
        acts.append(h[0, pos, :].float().cpu().numpy())
    return np.array(acts)


def compute_pca(activations, top_k=None):
    mean = activations.mean(axis=0)
    centered = activations - mean
    n, d = centered.shape
    if n < d:
        cov = centered @ centered.T / max(n-1, 1)
        evals, evecs_small = np.linalg.eigh(cov)
        evecs = centered.T @ evecs_small
        norms = np.linalg.norm(evecs, axis=0, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        evecs = evecs / norms
    else:
        cov = centered.T @ centered / max(n-1, 1)
        evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]
    if top_k:
        evals, evecs = evals[:top_k], evecs[:, :top_k]
    return evals, evecs, mean


# ====================================================================
# Experiment 3: Eps Sweep
# ====================================================================

def measure_er_at_eps(model, input_ids, layer_idx, d_feat, eps_rel, n_random, pos=-1):
    device = input_ids.device
    d_model = model.config.hidden_size
    h = get_layer_output(model, input_ids, layer_idx)
    h_norm = h[0, pos, :].norm().item()
    eps_abs = eps_rel * h_norm

    fd = measure_logits_delta(model, input_ids, layer_idx, d_feat, eps_abs, pos)
    feat_eff = fd.norm().item()

    rand_effs = []
    for _ in range(n_random):
        dr = torch.randn(d_model, device=device)
        dr = dr / dr.norm()
        rd = measure_logits_delta(model, input_ids, layer_idx, dr, eps_abs, pos)
        rand_effs.append(rd.norm().item())

    mr = np.mean(rand_effs)
    mx = np.max(rand_effs)
    er = feat_eff / mr if mr > 1e-10 else 0
    sa = feat_eff / mx if mx > 1e-10 else 0
    return er, sa, feat_eff, mr


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['deepseek7b','qwen3','glm4'])
    parser.add_argument('--experiments', nargs='+', default=['1','2','3'])
    parser.add_argument('--n_random', type=int, default=30)
    parser.add_argument('--n_sentences', type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(f'results/causal_fiber/{args.model}_cclvii')
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / 'console.log'

    def log(msg):
        print(msg, flush=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log(f"Phase CCLVII: Critical Validation Experiments")
    log(f"Model: {args.model}, Experiments: {args.experiments}")
    log(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    log("Loading model...")
    model, tokenizer = load_model(args.model)
    cfg = MODEL_CONFIGS[args.model]
    n_layers, d_model = cfg['n_layers'], cfg['d_model']
    device = next(model.parameters()).device
    is_bf16 = (cfg['dtype'] == 'bf16')
    log(f"Loaded: {cfg['name']}, {n_layers}L, d={d_model}, dev={device}, bf16={is_bf16}")

    # Select 4 representative layers
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
    # EXPERIMENT 1: True Jacobian SVD
    # ==================================================================
    if '1' in args.experiments:
        log(f"\n{'='*70}")
        log(f"EXPERIMENT 1: True Jacobian SVD via {'Power Iteration' if is_bf16 else 'Large Sampling (200)'}")
        log(f"{'='*70}")

        exp1 = {}
        for feat_name, pairs in features:
            log(f"\n--- {feat_name.upper()} ---")
            exp1[feat_name] = {}

            for layer in layers:
                feat_effs, sigma1s = [], []
                for si, (sa, sb) in enumerate(pairs):
                    d_feat, ids_a, pa, h_norm = get_feat_direction(model, tokenizer, sa, sb, layer, device)
                    if d_feat is None: continue
                    eps_abs = 0.05 * h_norm

                    # feat_eff
                    fd = measure_logits_delta(model, ids_a, layer, d_feat, eps_abs, pos=pa)
                    feat_eff = fd.norm().item()

                    # ||J||_2 via power iteration or large sampling
                    sigma1 = power_iteration_jacobian(model, ids_a, layer, n_iter=15, eps_rel=0.01, pos=pa, is_bf16=is_bf16)

                    sa_val = feat_eff / sigma1 if sigma1 > 1e-10 else 0
                    feat_effs.append(feat_eff)
                    sigma1s.append(sigma1)
                    log(f"  L{layer} P{si+1}: feat={feat_eff:.4f} ||J||_2={sigma1:.4f} SA={sa_val:.3f}")

                if feat_effs:
                    avg_fe, avg_s1 = np.mean(feat_effs), np.mean(sigma1s)
                    avg_sa = avg_fe / avg_s1 if avg_s1 > 1e-10 else 0
                    verdict = 'GENUINE' if avg_sa > 1.0 else 'NOT>1'
                    exp1[feat_name][layer] = {'feat_eff': avg_fe, 'sigma1': avg_s1, 'SA': avg_sa, 'verdict': verdict}
                    log(f"  => L{layer}: feat={avg_fe:.4f} ||J||_2={avg_s1:.4f} SA={avg_sa:.3f} {verdict}")

        log(f"\n  EXP1 SUMMARY: SA > 1.0 means feat_eff > ||J||_2 (genuine encoding)")
        log(f"  {'Feat':>8} {'L':>4} {'feat_eff':>10} {'||J||_2':>10} {'SA':>8} {'Verdict':>8}")
        for fn, _ in features:
            for l in layers:
                if l in exp1[fn]:
                    r = exp1[fn][l]
                    log(f"  {fn:>8} {l:>4} {r['feat_eff']:>10.4f} {r['sigma1']:>10.4f} {r['SA']:>8.3f} {r['verdict']:>8}")

        all_results['exp1'] = exp1
        with open(out_dir / 'exp1_jacobian_svd.json', 'w') as f:
            json.dump(exp1, f, indent=2, default=str)

    # ==================================================================
    # EXPERIMENT 2: PCA Task Subspace
    # ==================================================================
    if '2' in args.experiments:
        log(f"\n{'='*70}")
        log(f"EXPERIMENT 2: Task Subspace Projection (PCA)")
        log(f"{'='*70}")

        # Collect sentences for PCA
        all_sents = list(set([s for p in TENSE_PAIRS+QUESTION_PAIRS for s in p] + PCA_SENTENCES))
        log(f"  PCA sentences: {len(all_sents)}")

        # Max PCA components = min(n_sentences, d_model) - 1
        max_pca_comp = min(len(all_sents), d_model) - 1
        pca_k_values = [k for k in [10, 50, 100, 200] if k <= max_pca_comp]
        if not pca_k_values:
            pca_k_values = [max_pca_comp]
        log(f"  Max PCA components: {max_pca_comp}, testing k={pca_k_values}")
        exp2 = {}

        for layer in layers:
            log(f"\n  Layer {layer}:")
            acts = collect_activations(model, tokenizer, all_sents, layer)
            evals, evecs, mean = compute_pca(acts)
            total_var = evals.sum()
            cum_var = np.cumsum(evals) / total_var
            log(f"    Shape: {acts.shape}, Top10 var: {cum_var[9]:.3f}, Top50: {cum_var[min(49,len(cum_var)-1)]:.3f}, Top100: {cum_var[min(99,len(cum_var)-1)]:.3f}")

            layer_data = {'cum_var_top10': float(cum_var[9]),
                          'cum_var_top50': float(cum_var[min(49,len(cum_var)-1)]),
                          'features': {}}

            for feat_name, pairs in features:
                pca_ers, pca_sas = {k: [] for k in pca_k_values}, {k: [] for k in pca_k_values}
                residual_ratios = []

                for si, (sa, sb) in enumerate(pairs[:2]):
                    d_feat, ids_a, pa, h_norm = get_feat_direction(model, tokenizer, sa, sb, layer, device)
                    if d_feat is None: continue
                    eps_abs = 0.05 * h_norm

                    d_feat_np = d_feat.float().cpu().numpy()
                    d_feat_np = d_feat_np / np.linalg.norm(d_feat_np)

                    # Decompose feat direction: PCA component + residual
                    for k in pca_k_values:
                        pca_comp = evecs[:, :k]  # [d_model, k]
                        # Project d_feat onto PCA subspace
                        d_pca_np = pca_comp @ (pca_comp.T @ d_feat_np)
                        d_res_np = d_feat_np - d_pca_np

                        d_pca_norm = np.linalg.norm(d_pca_np)
                        d_res_norm = np.linalg.norm(d_res_np)
                        pca_frac = d_pca_norm**2  # fraction of variance in PCA subspace

                        # Measure feat_eff for PCA component
                        if d_pca_norm > 1e-8:
                            d_pca_t = torch.tensor(d_pca_np / d_pca_norm, device=device, dtype=torch.float32)
                            fd_pca = measure_logits_delta(model, ids_a, layer, d_pca_t, eps_abs, pos=pa)
                            feat_eff_pca = fd_pca.norm().item()
                        else:
                            feat_eff_pca = 0

                        # Measure feat_eff for residual component
                        if d_res_norm > 1e-8:
                            d_res_t = torch.tensor(d_res_np / d_res_norm, device=device, dtype=torch.float32)
                            fd_res = measure_logits_delta(model, ids_a, layer, d_res_t, eps_abs, pos=pa)
                            feat_eff_res = fd_res.norm().item()
                        else:
                            feat_eff_res = 0

                        # Random directions in PCA subspace
                        rand_pca_effs = []
                        for _ in range(args.n_random):
                            coeffs = np.random.randn(k)
                            r_np = pca_comp @ coeffs
                            r_norm = np.linalg.norm(r_np)
                            if r_norm < 1e-8: continue
                            r_t = torch.tensor(r_np / r_norm, device=device, dtype=torch.float32)
                            rd = measure_logits_delta(model, ids_a, layer, r_t, eps_abs, pos=pa)
                            rand_pca_effs.append(rd.norm().item())

                        mean_rand_pca = np.mean(rand_pca_effs) if rand_pca_effs else 0
                        er_pca = feat_eff_pca / mean_rand_pca if mean_rand_pca > 1e-10 else 0

                        # Residual efficacy ratio
                        res_ratio = feat_eff_res / feat_eff_pca if feat_eff_pca > 1e-10 else float('inf')

                        pca_ers[k].append(er_pca)
                        residual_ratios.append(res_ratio)

                avg_ers = {k: float(np.mean(v)) for k, v in pca_ers.items() if v}
                avg_res_ratio = float(np.mean(residual_ratios)) if residual_ratios else 0

                log(f"    {feat_name}: E/R in PCA: " + ", ".join([f"k={k}={v:.2f}" for k,v in sorted(avg_ers.items())]))
                log(f"    {feat_name}: feat_eff_residual / feat_eff_PCA = {avg_res_ratio:.3f} (<1: PCA dominates, >1: residual dominates)")

                layer_data['features'][feat_name] = {'er_pca': avg_ers, 'residual_ratio': avg_res_ratio}

            exp2[layer] = layer_data

        all_results['exp2'] = exp2
        with open(out_dir / 'exp2_pca_projection.json', 'w') as f:
            json.dump(exp2, f, indent=2, default=str)

    # ==================================================================
    # EXPERIMENT 3: Epsilon Sweep
    # ==================================================================
    if '3' in args.experiments:
        log(f"\n{'='*70}")
        log(f"EXPERIMENT 3: Epsilon Sweep (Linearity Check)")
        log(f"{'='*70}")

        eps_values = [0.01, 0.03, 0.05, 0.10]
        exp3 = {}

        for feat_name, pairs in features:
            log(f"\n--- {feat_name.upper()} ---")
            exp3[feat_name] = {}

            for layer in layers:
                eps_data = {e: {'er':[],'sa':[],'feat_eff':[],'mean_rand':[],'norm_feat':[],'norm_rand':[]} for e in eps_values}

                for si, (sa, sb) in enumerate(pairs[:2]):
                    d_feat, ids_a, pa, h_norm = get_feat_direction(model, tokenizer, sa, sb, layer, device)
                    if d_feat is None: continue

                    for eps in eps_values:
                        er, sa_v, fe, mr = measure_er_at_eps(model, ids_a, layer, d_feat, eps, args.n_random, pos=pa)
                        eps_data[eps]['er'].append(er)
                        eps_data[eps]['sa'].append(sa_v)
                        eps_data[eps]['feat_eff'].append(fe)
                        eps_data[eps]['mean_rand'].append(mr)
                        eps_data[eps]['norm_feat'].append(fe / eps)  # Should be constant if linear
                        eps_data[eps]['norm_rand'].append(mr / eps)  # Should be constant if linear

                # Average
                avg = {}
                for eps in eps_values:
                    d = eps_data[eps]
                    avg[eps] = {
                        'er': float(np.mean(d['er'])),
                        'sa': float(np.mean(d['sa'])),
                        'feat_eff': float(np.mean(d['feat_eff'])),
                        'mean_rand': float(np.mean(d['mean_rand'])),
                        'norm_feat': float(np.mean(d['norm_feat'])),
                        'norm_rand': float(np.mean(d['norm_rand'])),
                    }

                # Linearity check: norm_feat and norm_rand should be constant
                nf = [avg[e]['norm_feat'] for e in eps_values]
                nr = [avg[e]['norm_rand'] for e in eps_values]
                nf_cv = np.std(nf) / np.mean(nf) if np.mean(nf) > 0 else float('inf')
                nr_cv = np.std(nr) / np.mean(nr) if np.mean(nr) > 0 else float('inf')
                ers = [avg[e]['er'] for e in eps_values]
                er_cv = np.std(ers) / np.mean(ers) if np.mean(ers) > 0 else float('inf')

                exp3[feat_name][layer] = {'per_eps': avg, 'norm_feat_cv': float(nf_cv), 'norm_rand_cv': float(nr_cv), 'er_cv': float(er_cv)}

                log(f"  L{layer}:")
                log(f"    {'eps':>6} {'E/R':>8} {'SA':>8} {'feat_eff':>10} {'meanR':>10} {'feat/eps':>10} {'rand/eps':>10}")
                for eps in eps_values:
                    d = avg[eps]
                    log(f"    {eps:>6.0%} {d['er']:>8.2f} {d['sa']:>8.3f} {d['feat_eff']:>10.4f} {d['mean_rand']:>10.4f} {d['norm_feat']:>10.2f} {d['norm_rand']:>10.2f}")
                lin_verdict = 'LINEAR' if nf_cv < 0.15 and nr_cv < 0.15 else 'NONLINEAR'
                log(f"    Linearity: feat/eps CV={nf_cv:.3f}, rand/eps CV={nr_cv:.3f}, E/R CV={er_cv:.3f} => {lin_verdict}")

        log(f"\n  EXP3 SUMMARY:")
        log(f"  {'Feat':>8} {'L':>4} {'E/R@1%':>8} {'E/R@3%':>8} {'E/R@5%':>8} {'E/R@10%':>8} {'feat_cv':>8} {'rand_cv':>8} {'Linear?':>8}")
        for fn, _ in features:
            for l in layers:
                if l in exp3[fn]:
                    r = exp3[fn][l]
                    pe = r['per_eps']
                    lin = 'YES' if r['norm_feat_cv'] < 0.15 and r['norm_rand_cv'] < 0.15 else 'NO'
                    log(f"  {fn:>8} {l:>4} {pe[0.01]['er']:>8.2f} {pe[0.03]['er']:>8.2f} {pe[0.05]['er']:>8.2f} {pe[0.10]['er']:>8.2f} {r['norm_feat_cv']:>8.3f} {r['norm_rand_cv']:>8.3f} {lin:>8}")

        all_results['exp3'] = exp3
        with open(out_dir / 'exp3_eps_sweep.json', 'w') as f:
            json.dump(exp3, f, indent=2, default=str)

    log(f"\nAll experiments done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
