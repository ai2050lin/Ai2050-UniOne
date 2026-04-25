"""
Phase CCLVI: Layer-wise Jacobian Norm Control Analysis
=====================================================
Key question: is E/R growth just ||d logits/d h|| growth?

If ||J|| grows exponentially, BOTH feat and rand effects grow,
so their RATIO (E/R) stays CONSTANT unless features grow FASTER.

Measurements per layer:
  feat_eff = ||Δlogits_feat||/eps   (feature sensitivity)
  mean_rand = E[||Δlogits_rand||]/eps ≈ ||J||_F/sqrt(d)  (average sensitivity)
  max_rand = max ||Δlogits_rand||/eps ≈ ||J||_2  (top singular direction)
  E/R = feat_eff / mean_rand
  spectral_alignment = feat_eff / max_rand
  eff_rank ≈ (||J||_F/||J||_2)^2

Exponential growth rates: log(y) = alpha * idx + c
  alpha_feat > alpha_max => GENUINE encoding (beats ALL directions)
  alpha_feat ~ alpha_max > alpha_mean => Jacobian concentration + feature alignment
  alpha_feat ~ alpha_mean => NO special encoding (just sensitivity growth)
"""
import argparse, os, torch, numpy as np
from pathlib import Path
from datetime import datetime
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
            trust_remote_code=True, local_files_only=True)
    elif dtype == 'bf16':
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, device_map="auto", trust_remote_code=True, local_files_only=True)
    model.eval()
    return model, tokenizer

def get_target_layer(model, layer_idx):
    model_type = model.config.model_type
    if model_type in ['qwen2', 'qwen3']:
        return model.model.layers[layer_idx]
    elif model_type in ['chatglm', 'glm4']:
        return model.transformer.encoder.layers[layer_idx]
    else:
        return model.model.layers[layer_idx]

def get_layer_output(model, input_ids, layer_idx):
    """Get hidden state at a specific transformer layer output."""
    h = None
    def hook_fn(module, input, output):
        nonlocal h
        if isinstance(output, tuple):
            h = output[0].detach().clone()
        else:
            h = output.detach().clone()
    layer = get_target_layer(model, layer_idx)
    handle = layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(input_ids)
    handle.remove()
    return h

def measure_effect(model, input_ids, layer_idx, direction, eps_abs, pos=-1):
    """Measure logit change from perturbation along direction at layer."""
    d = direction
    def hook_perturb(module, input, output):
        if isinstance(output, tuple):
            hs = output[0].clone()
            hs[0, pos, :] = hs[0, pos, :] + eps_abs * d
            return (hs,) + output[1:]
        else:
            hs = output.clone()
            hs[0, pos, :] = hs[0, pos, :] + eps_abs * d
            return hs
    
    with torch.no_grad():
        orig = model(input_ids).logits[0, pos, :].detach().clone()
    
    layer = get_target_layer(model, layer_idx)
    handle = layer.register_forward_hook(hook_perturb)
    with torch.no_grad():
        pert = model(input_ids).logits[0, pos, :].detach().clone()
    handle.remove()
    
    return (pert - orig).norm().item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['deepseek7b','qwen3','glm4'])
    parser.add_argument('--n_random', type=int, default=40)
    parser.add_argument('--n_sentences', type=int, default=6)
    parser.add_argument('--eps_rel', type=float, default=0.05)
    args = parser.parse_args()

    out_dir = Path(f'results/causal_fiber/{args.model}_cclvi')
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / 'console.log'

    def log(msg):
        print(msg, flush=True)
        with open(log_file, 'a', encoding='utf-8') as f: f.write(msg + '\n')

    log(f"Phase CCLVI: Jacobian Norm Control")
    log(f"Model: {args.model}, n_rand={args.n_random}, n_sent={args.n_sentences}, eps={args.eps_rel}")
    log(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    log("Loading model...")
    model, tokenizer = load_model(args.model)
    cfg = MODEL_CONFIGS[args.model]
    n_layers, d_model = cfg['n_layers'], cfg['d_model']
    device = next(model.parameters()).device
    log(f"Loaded: {cfg['name']}, {n_layers}L, d={d_model}, dev={device}")

    layers = np.linspace(1, n_layers-2, 8, dtype=int).tolist()
    log(f"Layers: {layers}")

    tense_pairs = TENSE_PAIRS[:args.n_sentences]
    question_pairs = QUESTION_PAIRS[:min(args.n_sentences, len(QUESTION_PAIRS))]
    features = [('tense', tense_pairs), ('question', question_pairs)]

    all_results = {}
    for feat_name, pairs in features:
        log(f"\n=== {feat_name.upper()} ===")
        layer_data = {l: {'feat_effs':[],'mean_rands':[],'max_rands':[],'ers':[],'sas':[],'effrs':[],'concs':[]} for l in layers}

        for si, (sa, sb) in enumerate(pairs):
            log(f"  Pair {si+1}: '{sa[:30]}...'")
            enc_a = tokenizer(sa, return_tensors='pt').to(device)
            enc_b = tokenizer(sb, return_tensors='pt').to(device)
            input_ids_a = enc_a['input_ids']
            input_ids_b = enc_b['input_ids']

            for layer in layers:
                h_a = get_layer_output(model, input_ids_a, layer)
                h_b = get_layer_output(model, input_ids_b, layer)
                pos_a = h_a.shape[1] - 1
                pos_b = h_b.shape[1] - 1

                d_feat = h_b[0, pos_b, :] - h_a[0, pos_a, :]
                if d_feat.norm() < 1e-8:
                    log(f"    L{layer}: zero direction, skip")
                    continue
                d_feat = d_feat / d_feat.norm()

                h_norm = h_a[0, pos_a, :].norm().item()
                eps_abs = args.eps_rel * h_norm

                feat_eff = measure_effect(model, input_ids_a, layer, d_feat, eps_abs, pos=pos_a)

                rand_effs = []
                for _ in range(args.n_random):
                    dr = torch.randn(d_model, device=device)
                    dr = dr / dr.norm()
                    re = measure_effect(model, input_ids_a, layer, dr, eps_abs, pos=pos_a)
                    rand_effs.append(re)

                re_arr = np.array(rand_effs)
                mr, mx = np.mean(re_arr), np.max(re_arr)
                er = feat_eff/mr if mr>1e-10 else 0
                sa_val = feat_eff/mx if mx>1e-10 else 0
                jf = mr * np.sqrt(d_model)
                effr = (jf/mx)**2 if mx>1e-10 else 0
                conc = mx/mr if mr>1e-10 else 0

                r = layer_data[layer]
                r['feat_effs'].append(feat_eff); r['mean_rands'].append(mr)
                r['max_rands'].append(mx); r['ers'].append(er)
                r['sas'].append(sa_val); r['effrs'].append(effr); r['concs'].append(conc)

                log(f"    L{layer}: feat={feat_eff:.4f} meanR={mr:.4f} maxR={mx:.4f} E/R={er:.2f} SA={sa_val:.3f} effR={effr:.0f}")

        all_results[feat_name] = layer_data

    # ---- AGGREGATE ----
    log(f"\n{'='*70}\nAGGREGATE ANALYSIS\n{'='*70}")

    for feat_name, _ in features:
        log(f"\n--- {feat_name.upper()} ---")
        header = f"  {'L':>4} {'feat':>8} {'meanR':>8} {'maxR':>8} {'E/R':>6} {'SA':>5} {'effR':>5} {'conc':>5}"
        log(header)

        agg = {k:[] for k in ['fe','mr','mx','er','sa','efr','conc']}
        vl = []
        for l in layers:
            r = all_results[feat_name][l]
            if not r['feat_effs']: continue
            vl.append(l)
            fe,mr,mx = np.mean(r['feat_effs']),np.mean(r['mean_rands']),np.mean(r['max_rands'])
            er,sa_val = np.mean(r['ers']),np.mean(r['sas'])
            efr,conc = np.mean(r['effrs']),np.mean(r['concs'])
            agg['fe'].append(fe); agg['mr'].append(mr); agg['mx'].append(mx)
            agg['er'].append(er); agg['sa'].append(sa_val); agg['efr'].append(efr); agg['conc'].append(conc)
            log(f"  {l:>4} {fe:>8.4f} {mr:>8.4f} {mx:>8.4f} {er:>6.2f} {sa_val:>5.3f} {efr:>5.0f} {conc:>5.2f}")

        if len(vl) < 3:
            log("  Too few layers"); continue

        idx = np.arange(len(vl))
        def exp_fit(vals):
            v = np.array(vals) + 1e-12
            lv = np.log(v)
            alpha, c = np.polyfit(idx, lv, 1)
            corr = np.corrcoef(idx, lv)[0,1]
            pred = alpha*idx + c
            ss_res = np.sum((lv-pred)**2); ss_tot = np.sum((lv-np.mean(lv))**2)
            r2 = 1-ss_res/ss_tot if ss_tot>0 else 0
            return alpha, corr, r2

        log(f"\n  Exponential growth rates (log(y) = alpha * layer_index + c):")
        log(f"  {'Quantity':>35} {'alpha':>8} {'Corr':>7} {'R-sq':>7}")
        results = {}
        for name, vals in [('feat_eff', agg['fe']),
                           ('mean_rand (||J||_F/sqrt(d))', agg['mr']),
                           ('max_rand (||J||_2)', agg['mx']),
                           ('E/R', agg['er']),
                           ('spectral_align', agg['sa']),
                           ('||J||_F', [m*np.sqrt(d_model) for m in agg['mr']]),
                           ('||J||_2', agg['mx'])]:
            alpha, corr, r2 = exp_fit(vals)
            results[name] = alpha
            log(f"  {name:>35} {alpha:>8.4f} {corr:>7.3f} {r2:>7.3f}")

        alpha_feat = results['feat_eff']
        alpha_mean = results['mean_rand (||J||_F/sqrt(d))']
        alpha_max = results['max_rand (||J||_2)']

        log(f"\n  KEY COMPARISON:")
        log(f"  alpha_feat - alpha_mean = {alpha_feat - alpha_mean:+.4f}")
        log(f"    (>0: features grow faster than average -> E/R grows)")
        log(f"  alpha_feat - alpha_max  = {alpha_feat - alpha_max:+.4f}")
        log(f"    (>0: features beat ALL directions -> GENUINE encoding!)")
        log(f"  alpha_max  - alpha_mean = {alpha_max - alpha_mean:+.4f}")
        log(f"    (>0: Jacobian concentrating)")

        if abs(alpha_mean) > 1e-6:
            log(f"  alpha_feat / alpha_mean = {alpha_feat/alpha_mean:.3f} (1.0=no special; >1.0=genuine)")
        if abs(alpha_max) > 1e-6:
            log(f"  alpha_feat / alpha_max  = {alpha_feat/alpha_max:.3f} (>1.0=GENUINE encoding)")
            log(f"  alpha_max  / alpha_mean = {alpha_max/alpha_mean:.3f} (>1.0=Jacobian concentrating)")

        log(f"\n  CONCLUSION for {feat_name}:")
        if alpha_feat > alpha_max + 0.02:
            log(f"  >>> GENUINE FEATURE ENCODING <<<")
            log(f"  Features grow FASTER than max random! Not just Jacobian growth.")
        elif alpha_feat > alpha_mean + 0.02:
            log(f"  >>> JACOBIAN CONCENTRATION + ALIGNMENT <<<")
            log(f"  Features ~ max random > mean random. E/R from Jacobian going low-rank.")
            log(f"  Features align with top singular. Real but linear-algebraic, not 'semantic'.")
        else:
            log(f"  >>> NO SPECIAL ENCODING <<<")
            log(f"  E/R growth is just overall sensitivity growth.")

    log(f"\nDone: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()
