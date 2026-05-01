"""
CCI(301): Δg增益旋钮语义图谱 v2 — z空间(presigmoid)扰动
=========================================================
核心修正:
  1. 在z空间(presigmoid)做PCA和perturb, 保持sigmoid后g∈[0,1]
  2. eps用z的std为单位: eps=0.1, 0.5, 1.0, 2.0
  3. 只测中间层(mid)和深层(late), 避免浅层扰动过强
  4. 对比perturb z_PC vs perturb h_PC(对照)

目标: 积累数据, 让规律自然浮现

用法:
  python cci_pc_gain_map.py --model qwen3
  python cci_pc_gain_map.py --model glm4
  python cci_pc_gain_map.py --model deepseek7b
"""
import argparse, os, sys, time, gc, json
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from collections import defaultdict

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, get_W_U,
    get_layer_weights, MODEL_CONFIGS,
)

TEMP_DIR = Path("tests/glm5_temp")
LOG_FILE = TEMP_DIR / "cci_pc_gain_map_log.txt"

CONCEPTS = {
    "animals": ["dog", "cat", "horse", "eagle", "shark"],
    "food":    ["apple", "rice", "bread", "cheese", "pizza"],
    "tools":   ["hammer", "knife", "saw", "drill", "wrench"],
    "nature":  ["mountain", "river", "ocean", "forest", "desert"],
}

TEMPLATE = "The {} is"

# Perturbation strengths (in z-std units)
EPS_LIST = [0.5, 1.0, 2.0]
# Number of PC directions to test per layer
N_PC_TEST = 8


def run_experiment(model_name):
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log_f(msg=""):
        print(msg)
        log_lines.append(msg)

    log_f(f"\n{'#'*70}")
    log_f(f"CCI(301): z-space PC Gain Map v2")
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

    # ===== Step 1: Collect z (presigmoid) for all words =====
    log_f(f"\n--- Step 1: Collecting z (presigmoid) activations ---")

    all_words = []
    all_cats = []
    for cat, words in CONCEPTS.items():
        all_words.extend(words)
        all_cats.extend([cat] * len(words))

    word_z = {}        # word -> {li: z_array}
    word_h_tilde = {}  # word -> {li: h_tilde_array (MLP input)}
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
            z = lw.W_gate @ h_tilde  # presigmoid

            word_z[word][li] = z
            word_h_tilde[word][li] = h_tilde

    log_f(f"  Collected z for {len(all_words)} words across {n_layers} layers")

    # ===== Step 2: Per-layer PCA of z =====
    log_f(f"\n--- Step 2: Per-layer PCA of z (presigmoid) ---")

    # Select layers: mid and late
    layer_indices = sorted(set([
        n_layers // 3,
        n_layers // 2,
        2 * n_layers // 3,
        n_layers - 1,
    ]))
    log_f(f"  Testing layers: {layer_indices}")

    pca_results = {}

    for li in layer_indices:
        valid_words = [w for w in all_words if li in word_z[w]]
        if len(valid_words) < 2:
            continue

        Z_matrix = np.array([word_z[w][li] for w in valid_words])  # [n_words, n_inter]

        # Center
        Z_mean = Z_matrix.mean(axis=0)
        Z_std = Z_matrix.std(axis=0) + 1e-10  # per-dim std for later scaling
        Z_centered = Z_matrix - Z_mean

        # PCA via SVD
        try:
            U_pca, s_pca, Vt_pca = np.linalg.svd(Z_centered, full_matrices=False)
            k_svd = min(19, 50, n_inter)
            pc_directions = Vt_pca[:k_svd]  # [k_svd, n_inter]
            eigenvalues = s_pca[:k_svd] ** 2 / (len(valid_words) - 1)
            total_var = np.sum(eigenvalues)
            explained_var_ratio = eigenvalues / max(total_var, 1e-20)

            # PC std (how much z varies along each PC)
            pc_std = s_pca[:k_svd] / np.sqrt(len(valid_words) - 1)

            pca_results[li] = {
                'pc_directions': pc_directions,
                'eigenvalues': eigenvalues,
                'explained_var_ratio': explained_var_ratio,
                'pc_std': pc_std,
                'Z_mean': Z_mean,
                'Z_std': Z_std,
                'valid_words': valid_words,
            }

            log_f(f"  L{li}: n_valid={len(valid_words)}, n_pc={k_svd}")
            log_f(f"    Top-5 explained var: {explained_var_ratio[:5].round(4).tolist()}")
            log_f(f"    PC std (top-5): {pc_std[:5].round(4).tolist()}")
            log_f(f"    Z global stats: mean={Z_mean.mean():.4f}, std={Z_centered.std():.4f}")
        except Exception as e:
            log_f(f"  L{li}: PCA failed: {e}")

    # ===== Step 3: Hook perturb z along PC directions =====
    log_f(f"\n{'='*70}")
    log_f(f"Step 3: Hook perturb z along PC directions (via MLP output replacement)")
    log_f(f"{'='*70}")

    perturb_results = {}

    for li in layer_indices:
        if li not in pca_results:
            continue

        pc_dirs = pca_results[li]['pc_directions']
        pc_std = pca_results[li]['pc_std']
        Z_mean = pca_results[li]['Z_mean']
        valid_words = pca_results[li]['valid_words']
        n_pc_avail = min(N_PC_TEST, pc_dirs.shape[0])

        log_f(f"\n  === Layer L{li} ===")

        perturb_results[li] = {}
        lw = get_layer_weights(layers_list[li], d_model, mlp_type)
        W_down_t = torch.tensor(lw.W_down, dtype=torch.bfloat16, device=device)
        W_up = lw.W_up
        W_gate = lw.W_gate

        for pc_idx in range(n_pc_avail):
            pc_dir = pc_dirs[pc_idx]  # [n_inter]
            pc_s = float(pc_std[pc_idx])

            perturb_results[li][pc_idx] = {}

            for eps in EPS_LIST:
                # Actual perturbation magnitude = eps * pc_std[pc_idx]
                delta_z = eps * pc_s * pc_dir  # [n_inter]

                test_words = valid_words[:8]  # Use first 8 words (2 per category)
                top1_changes = 0
                total_tests = 0
                delta_logit_norms = []
                logit_coss = []
                top_logit_changes = defaultdict(float)

                for word in test_words:
                    if li not in word_z[word]:
                        continue

                    z_orig = word_z[word][li]
                    z_perturbed = z_orig + delta_z

                    # Apply sigmoid to get g
                    z_clip = np.clip(z_perturbed, -500, 500)
                    g_perturbed = 1.0 / (1.0 + np.exp(-z_clip))

                    # Get u
                    h_tilde = word_h_tilde[word][li]
                    u = W_up @ h_tilde

                    # Convert to torch
                    g_pert_t = torch.tensor(g_perturbed, dtype=torch.bfloat16, device=device)
                    u_t = torch.tensor(u, dtype=torch.bfloat16, device=device)

                    base_logits = word_base_logits[word]
                    base_top1 = int(np.argmax(base_logits))

                    # Hook intervention
                    text = TEMPLATE.format(word)
                    input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
                    last_pos = input_ids.shape[1] - 1

                    intervention_done = [False]

                    def make_perturb_hook(g_pert_t, u_t):
                        def hook(module, input, output):
                            if intervention_done[0]:
                                return output
                            intervention_done[0] = True

                            if isinstance(output, tuple):
                                out = output[0]
                            else:
                                out = output

                            new_out = (W_down_t @ (g_pert_t * u_t)).unsqueeze(0).unsqueeze(0)
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
                    delta_logits = interv_logits - base_logits
                    delta_norm = float(np.linalg.norm(delta_logits))

                    # Logit cosine
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

                    # Top-5 token changes
                    top_changes_idx = np.argsort(np.abs(delta_logits))[-5:]
                    for idx in top_changes_idx:
                        top_logit_changes[int(idx)] = max(
                            top_logit_changes[int(idx)],
                            abs(float(delta_logits[int(idx)]))
                        )

                    del g_pert_t, u_t

                # Aggregate
                top1_rate = top1_changes / max(total_tests, 1)
                avg_delta_norm = float(np.mean(delta_logit_norms)) if delta_logit_norms else 0.0
                avg_logit_cos = float(np.mean(logit_coss)) if logit_coss else 1.0

                sorted_tokens = sorted(top_logit_changes.items(), key=lambda x: -x[1])[:5]
                top5_info = []
                for tok_id, change in sorted_tokens:
                    tok_str = tokenizer.decode([tok_id]).strip()
                    top5_info.append((tok_id, tok_str, round(change, 3)))

                perturb_results[li][pc_idx][eps] = {
                    'top1_rate': top1_rate,
                    'avg_delta_norm': avg_delta_norm,
                    'avg_logit_cos': avg_logit_cos,
                    'n_tests': total_tests,
                    'top5_affected': top5_info,
                }

                log_f(f"    PC{pc_idx} eps={eps}: top1_rate={top1_rate:.3f} "
                      f"||Δlogits||={avg_delta_norm:.2f} cos={avg_logit_cos:.4f} "
                      f"top3={top5_info[:3]}")

                torch.cuda.empty_cache()

        log_f(f"  Layer L{li} done")

    # ===== Step 4: Cross-PC comparison (data table) =====
    log_f(f"\n{'='*70}")
    log_f(f"Step 4: Data tables")
    log_f(f"{'='*70}")

    for eps in EPS_LIST:
        log_f(f"\n  --- eps={eps} (in z-std units) ---")
        log_f(f"  {'Layer':<8} {'PC':<4} {'top1_rate':<10} {'||Δlogits||':<12} {'logit_cos':<10}")
        for li in sorted(perturb_results.keys()):
            for pc_idx in sorted(perturb_results[li].keys()):
                if eps in perturb_results[li][pc_idx]:
                    d = perturb_results[li][pc_idx][eps]
                    log_f(f"  L{li:<6} {pc_idx:<4} {d['top1_rate']:<10.3f} "
                          f"{d['avg_delta_norm']:<12.2f} {d['avg_logit_cos']:<10.4f}")

    # ===== Step 5: PCA variance structure =====
    log_f(f"\n{'='*70}")
    log_f(f"Step 5: PCA variance structure")
    log_f(f"{'='*70}")

    for li in sorted(pca_results.keys()):
        evr = pca_results[li]['explained_var_ratio']
        log_f(f"  L{li}: top-5 var={evr[:5].round(4).tolist()} "
              f"cum10={evr[:10].cumsum().round(4).tolist()}")

    # ===== Step 6: Per-PC semantic map =====
    log_f(f"\n{'='*70}")
    log_f(f"Step 6: Per-PC top affected tokens (mid layer)")
    log_f(f"{'='*70}")

    mid_li = n_layers // 2
    if mid_li in perturb_results:
        eps_show = EPS_LIST[-1]  # Use largest eps for clearer signal
        log_f(f"\n  L{mid_li}, eps={eps_show}:")
        for pc_idx in range(min(N_PC_TEST, len(perturb_results[mid_li]))):
            if eps_show in perturb_results[mid_li][pc_idx]:
                info = perturb_results[mid_li][pc_idx][eps_show]
                top5 = info['top5_affected']
                tokens_str = ", ".join([f"'{t}'({c:.1f})" for _, t, c in top5])
                log_f(f"    PC{pc_idx}: {tokens_str}")

    # ===== Save results =====
    log_f(f"\n{'='*70}")
    log_f(f"Saving results")
    log_f(f"{'='*70}")

    save_data = {
        'model': model_name,
        'n_layers': n_layers,
        'd_model': d_model,
        'n_inter': n_inter,
        'layer_indices': layer_indices,
        'pca_variance': {},
        'perturb_summary': {},
    }

    for li in layer_indices:
        if li in pca_results:
            save_data['pca_variance'][str(li)] = {
                'explained_var_ratio': pca_results[li]['explained_var_ratio'][:20].tolist(),
                'eigenvalues': pca_results[li]['eigenvalues'][:20].tolist(),
                'pc_std': pca_results[li]['pc_std'][:20].tolist(),
            }
        if li in perturb_results:
            save_data['perturb_summary'][str(li)] = {}
            for pc_idx in perturb_results[li]:
                save_data['perturb_summary'][str(li)][str(pc_idx)] = {}
                for eps in perturb_results[li][pc_idx]:
                    d = perturb_results[li][pc_idx][eps]
                    save_data['perturb_summary'][str(li)][str(pc_idx)][str(eps)] = {
                        'top1_rate': d['top1_rate'],
                        'avg_delta_norm': d['avg_delta_norm'],
                        'avg_logit_cos': d['avg_logit_cos'],
                        'n_tests': d['n_tests'],
                        'top5_affected': [(int(tid), tstr, float(ch)) for tid, tstr, ch in d['top5_affected']],
                    }

    json_path = TEMP_DIR / f"cci_pc_gain_map_{model_name}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    log_f(f"  Results saved to {json_path}")

    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write('\n'.join(log_lines) + '\n')

    del model
    torch.cuda.empty_cache()
    gc.collect()

    log_f(f"\nCCI(301) {model_name} done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    run_experiment(args.model)
