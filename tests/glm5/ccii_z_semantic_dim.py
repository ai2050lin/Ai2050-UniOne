"""
CCI(302): z维度语义映射 — ANOVA + 定向perturb
=============================================
数据驱动, 不预设理论:
  1. 对每层的z(presigmoid), 计算每个维度对4个超类的ANOVA F值
  2. 找到"最有语义区分力"的维度
  3. Hook perturb这些维度 vs 随机维度(对照)
  4. 积累原始数据, 让规律自然浮现

用法:
  python ccii_z_semantic_dim.py --model qwen3
  python ccii_z_semantic_dim.py --model glm4
  python ccii_z_semantic_dim.py --model deepseek7b
"""
import argparse, os, sys, time, gc, json
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from collections import defaultdict
from scipy import stats

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, get_W_U,
    get_layer_weights, MODEL_CONFIGS,
)

TEMP_DIR = Path("tests/glm5_temp")
LOG_FILE = TEMP_DIR / "ccii_z_semantic_dim_log.txt"

CONCEPTS = {
    "animals": ["dog", "cat", "horse", "eagle", "shark"],
    "food":    ["apple", "rice", "bread", "cheese", "pizza"],
    "tools":   ["hammer", "knife", "saw", "drill", "wrench"],
    "nature":  ["mountain", "river", "ocean", "forest", "desert"],
}

TEMPLATE = "The {} is"


def run_experiment(model_name):
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log_f(msg=""):
        print(msg)
        log_lines.append(msg)

    log_f(f"\n{'#'*70}")
    log_f(f"CCI(302): z-dim Semantic Mapping")
    log_f(f"Model: {model_name}")
    log_f(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_f(f"{'#'*70}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    n_inter = model_info.intermediate_size
    layers_list = get_layers(model)

    log_f(f"  n_layers={n_layers}, d_model={d_model}, n_inter={n_inter}")

    # ===== Step 1: Collect z for all words =====
    log_f(f"\n--- Step 1: Collecting z (presigmoid) activations ---")

    all_words = []
    all_cats = []
    word_cat = {}
    for cat, words in CONCEPTS.items():
        all_words.extend(words)
        all_cats.extend([cat] * len(words))
        for w in words:
            word_cat[w] = cat

    word_z = {}
    word_h_tilde = {}
    word_base_logits = {}

    for word in all_words:
        text = TEMPLATE.format(word)
        input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
        last_pos = input_ids.shape[1] - 1

        with torch.no_grad():
            base_logits = model(input_ids).logits[0, last_pos].detach().float().cpu().numpy()
        word_base_logits[word] = base_logits

        mlp_input = {}
        hooks = []
        for li in range(n_layers):
            layer = layers_list[li]
            if hasattr(layer, 'mlp'):
                def make_hook(key):
                    def hook(module, args):
                        if isinstance(args, tuple):
                            mlp_input[key] = args[0][0, last_pos].detach().float().cpu().numpy()
                        else:
                            mlp_input[key] = args[0, last_pos].detach().float().cpu().numpy()
                    return hook
                hooks.append(layer.mlp.register_forward_pre_hook(make_hook(f"L{li}")))

        with torch.no_grad():
            _ = model(input_ids)

        for h in hooks:
            h.remove()

        word_z[word] = {}
        word_h_tilde[word] = {}
        for li in range(n_layers):
            key = f"L{li}"
            if key not in mlp_input:
                continue
            lw = get_layer_weights(layers_list[li], d_model, mlp_type)
            if lw.W_gate is None:
                continue
            h_tilde = mlp_input[key]
            z = lw.W_gate @ h_tilde

            word_z[word][li] = z
            word_h_tilde[word][li] = h_tilde

    log_f(f"  Collected z for {len(all_words)} words across {n_layers} layers")

    # ===== Step 2: Per-dim ANOVA =====
    log_f(f"\n--- Step 2: Per-dimension ANOVA for category discrimination ---")

    layer_indices = sorted(set([
        n_layers // 4,
        n_layers // 3,
        n_layers // 2,
        2 * n_layers // 3,
        3 * n_layers // 4,
    ]))
    log_f(f"  Testing layers: {layer_indices}")

    anova_results = {}  # li -> {dim_idx: F_value, ...}

    for li in layer_indices:
        valid_words = [w for w in all_words if li in word_z[w]]
        if len(valid_words) < 4:
            continue

        # Build Z matrix: [n_words, n_inter]
        Z_matrix = np.array([word_z[w][li] for w in valid_words])
        cats_for_valid = [word_cat[w] for w in valid_words]

        # ANOVA for each dimension
        cat_groups = defaultdict(list)
        for i, cat in enumerate(cats_for_valid):
            cat_groups[cat].append(i)

        F_values = np.zeros(n_inter)
        p_values = np.ones(n_inter)

        for dim in range(n_inter):
            groups = [Z_matrix[indices, dim] for cat, indices in cat_groups.items()
                      if len(indices) >= 2]
            if len(groups) >= 2:
                try:
                    F, p = stats.f_oneway(*groups)
                    F_values[dim] = F if not np.isnan(F) else 0
                    p_values[dim] = p if not np.isnan(p) else 1.0
                except:
                    F_values[dim] = 0
                    p_values[dim] = 1.0

        anova_results[li] = {
            'F_values': F_values,
            'p_values': p_values,
            'valid_words': valid_words,
        }

        # Top discriminating dimensions
        top_dims = np.argsort(-F_values)[:20]
        top_F = F_values[top_dims]
        
        # How many dims are significant (p < 0.05)?
        n_sig = np.sum(p_values < 0.05)
        n_high_sig = np.sum(p_values < 0.001)
        
        # Distribution of F values
        F_nonzero = F_values[F_values > 0]
        
        log_f(f"\n  L{li}: n_valid={len(valid_words)}")
        log_f(f"    n_sig(p<0.05)={n_sig}, n_high_sig(p<0.001)={n_high_sig}")
        log_f(f"    F stats: mean={np.mean(F_nonzero):.2f}, median={np.median(F_nonzero):.2f}, "
              f"max={np.max(F_values):.2f}")
        log_f(f"    Top-10 dims: {top_dims[:10].tolist()}")
        log_f(f"    Top-10 F: {top_F[:10].round(2).tolist()}")
        
        # Also compute eta-squared (effect size) for top dims
        total_n = len(valid_words)
        for rank, dim in enumerate(top_dims[:5]):
            groups = [Z_matrix[indices, dim] for cat, indices in cat_groups.items()
                      if len(indices) >= 2]
            grand_mean = np.mean(Z_matrix[:, dim])
            ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
            ss_total = np.sum((Z_matrix[:, dim] - grand_mean)**2)
            eta_sq = ss_between / max(ss_total, 1e-20)
            log_f(f"    Dim {dim}: F={F_values[dim]:.2f}, p={p_values[dim]:.4e}, eta_sq={eta_sq:.4f}")

    # ===== Step 3: Perturb top-discriminating dims vs random dims =====
    log_f(f"\n{'='*70}")
    log_f(f"Step 3: Hook perturb — top-semantic dims vs random dims")
    log_f(f"{'='*70}")

    mid_li = n_layers // 2
    perturb_results = {}

    for li in [mid_li]:
        if li not in anova_results:
            continue

        F_values = anova_results[li]['F_values']
        top_dims = np.argsort(-F_values)[:5]  # top-5 semantic dims
        # Random dims (not in top-50)
        bottom_dims_pool = np.where(F_values < np.percentile(F_values[F_values > 0], 30))[0]
        rng = np.random.RandomState(42)
        random_dims = rng.choice(bottom_dims_pool, min(5, len(bottom_dims_pool)), replace=False)

        log_f(f"\n  === Layer L{li} ===")
        log_f(f"  Top semantic dims: {top_dims.tolist()}")
        log_f(f"  Random dims: {random_dims.tolist()}")

        lw = get_layer_weights(layers_list[li], d_model, mlp_type)
        W_down_t = torch.tensor(lw.W_down, dtype=torch.bfloat16, device=device)
        W_up = lw.W_up

        valid_words = anova_results[li]['valid_words']

        for dim_type, dims in [("semantic", top_dims), ("random", random_dims)]:
            perturb_results.setdefault(li, {})[dim_type] = {}

            for dim_idx, dim in enumerate(dims):
                perturb_results[li][dim_type][int(dim)] = {}

                for eps in [3.0, 5.0]:  # perturb z value by this amount
                    test_words = valid_words[:8]
                    top1_changes = 0
                    total_tests = 0
                    delta_logit_norms = []
                    logit_coss = []
                    top_logit_changes = defaultdict(float)
                    category_shifts = defaultdict(int)  # track category-level changes

                    for word in test_words:
                        if li not in word_z[word]:
                            continue

                        z_orig = word_z[word][li]
                        z_perturbed = z_orig.copy()
                        z_perturbed[dim] += eps  # only perturb one dimension

                        z_clip = np.clip(z_perturbed, -500, 500)
                        g_perturbed = 1.0 / (1.0 + np.exp(-z_clip))

                        h_tilde = word_h_tilde[word][li]
                        u = W_up @ h_tilde

                        g_pert_t = torch.tensor(g_perturbed, dtype=torch.bfloat16, device=device)
                        u_t = torch.tensor(u, dtype=torch.bfloat16, device=device)

                        base_logits = word_base_logits[word]
                        base_top1 = int(np.argmax(base_logits))
                        base_top1_tok = tokenizer.decode([base_top1])

                        text = TEMPLATE.format(word)
                        input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
                        last_pos = input_ids.shape[1] - 1

                        intervention_done = [False]

                        def make_perturb_hook(g_p, u_p):
                            def hook(module, input, output):
                                if intervention_done[0]:
                                    return output
                                intervention_done[0] = True
                                if isinstance(output, tuple):
                                    out = output[0]
                                else:
                                    out = output
                                new_out = (W_down_t @ (g_p * u_p)).unsqueeze(0).unsqueeze(0)
                                new_out = new_out.expand_as(out)
                                if isinstance(output, tuple):
                                    return (new_out,) + output[1:]
                                return new_out
                            return hook

                        hook_handle = layers_list[li].mlp.register_forward_hook(
                            make_perturb_hook(g_pert_t, u_t))

                        with torch.no_grad():
                            interv_logits = model(input_ids).logits[0, last_pos].detach().float().cpu().numpy()

                        hook_handle.remove()

                        interv_top1 = int(np.argmax(interv_logits))
                        interv_top1_tok = tokenizer.decode([interv_top1])
                        delta_logits = interv_logits - base_logits
                        delta_norm = float(np.linalg.norm(delta_logits))

                        if delta_norm > 1e-6:
                            base_norm = float(np.linalg.norm(base_logits))
                            logit_cos = float(np.dot(base_logits, interv_logits) /
                                             (base_norm * float(np.linalg.norm(interv_logits)) + 1e-20))
                        else:
                            logit_cos = 1.0

                        if interv_top1 != base_top1:
                            top1_changes += 1

                        total_tests += 1
                        delta_logit_norms.append(delta_norm)
                        logit_coss.append(logit_cos)

                        # Category shift
                        word_cat_name = word_cat[word]
                        if interv_top1 != base_top1:
                            category_shifts[f"{word_cat_name}->changed"] += 1

                        top_changes_idx = np.argsort(np.abs(delta_logits))[-5:]
                        for idx in top_changes_idx:
                            top_logit_changes[int(idx)] = max(
                                top_logit_changes[int(idx)],
                                abs(float(delta_logits[int(idx)])))

                        del g_pert_t, u_t

                    # Aggregate
                    top1_rate = top1_changes / max(total_tests, 1)
                    avg_delta_norm = float(np.mean(delta_logit_norms)) if delta_logit_norms else 0.0
                    avg_logit_cos = float(np.mean(logit_coss)) if logit_coss else 1.0

                    sorted_tokens = sorted(top_logit_changes.items(), key=lambda x: -x[1])[:5]
                    top5_info = [(int(tid), tokenizer.decode([tid]).strip(), round(ch, 3))
                                 for tid, ch in sorted_tokens]

                    perturb_results[li][dim_type][int(dim)][eps] = {
                        'top1_rate': top1_rate,
                        'avg_delta_norm': avg_delta_norm,
                        'avg_logit_cos': avg_logit_cos,
                        'n_tests': total_tests,
                        'top5_affected': top5_info,
                        'category_shifts': dict(category_shifts),
                    }

                    log_f(f"    {dim_type} dim{dim} eps={eps}: top1_rate={top1_rate:.3f} "
                          f"||dL||={avg_delta_norm:.2f} cos={avg_logit_cos:.4f} "
                          f"top3={[t[1] for t in top5_info[:3]]}")

                    torch.cuda.empty_cache()

    # ===== Step 4: ANOVA profile across layers =====
    log_f(f"\n{'='*70}")
    log_f(f"Step 4: ANOVA profile across layers")
    log_f(f"{'='*70}")

    log_f(f"\n  {'Layer':<8} {'n_sig(p<0.05)':<15} {'n_sig(p<0.001)':<18} {'max_F':<10} {'median_F':<12}")
    for li in sorted(anova_results.keys()):
        F = anova_results[li]['F_values']
        p = anova_results[li]['p_values']
        n_sig = int(np.sum(p < 0.05))
        n_high = int(np.sum(p < 0.001))
        F_nz = F[F > 0]
        log_f(f"  L{li:<6} {n_sig:<15} {n_high:<18} {np.max(F):<10.2f} {np.median(F_nz):<12.2f}")

    # ===== Step 5: Top-semantic dims' per-category means =====
    log_f(f"\n{'='*70}")
    log_f(f"Step 5: Top-semantic dims per-category z-means")
    log_f(f"{'='*70}")

    for li in sorted(anova_results.keys()):
        F = anova_results[li]['F_values']
        top5_dims = np.argsort(-F)[:5]
        valid_words = anova_results[li]['valid_words']

        Z_matrix = np.array([word_z[w][li] for w in valid_words])
        cats_for_valid = [word_cat[w] for w in valid_words]

        log_f(f"\n  L{li} top-5 semantic dims:")
        for dim in top5_dims:
            cat_means = {}
            for cat in CONCEPTS:
                cat_indices = [i for i, c in enumerate(cats_for_valid) if c == cat]
                if cat_indices:
                    cat_means[cat] = float(np.mean(Z_matrix[cat_indices, dim]))
            means_str = ", ".join([f"{c}={v:.2f}" for c, v in sorted(cat_means.items())])
            log_f(f"    dim{dim}: F={F[dim]:.2f}, means: {means_str}")

    # ===== Save results =====
    log_f(f"\n{'='*70}")
    log_f(f"Saving results")
    log_f(f"{'='*70}")

    save_data = {
        'model': model_name,
        'n_layers': n_layers,
        'd_model': d_model,
        'n_inter': n_inter,
        'anova_profile': {},
        'perturb_summary': {},
    }

    for li in anova_results:
        F = anova_results[li]['F_values']
        p = anova_results[li]['p_values']
        top20_dims = np.argsort(-F)[:20].tolist()
        save_data['anova_profile'][str(li)] = {
            'n_sig_p005': int(np.sum(p < 0.05)),
            'n_sig_p001': int(np.sum(p < 0.001)),
            'max_F': float(np.max(F)),
            'top20_dims': top20_dims,
            'top20_F': [float(F[d]) for d in top20_dims],
        }

    for li in perturb_results:
        save_data['perturb_summary'][str(li)] = perturb_results[li]

    json_path = TEMP_DIR / f"ccii_z_semantic_dim_{model_name}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    log_f(f"  Results saved to {json_path}")

    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write('\n'.join(log_lines) + '\n')

    del model
    torch.cuda.empty_cache()
    gc.collect()

    log_f(f"\nCCI(302) {model_name} done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    run_experiment(args.model)
