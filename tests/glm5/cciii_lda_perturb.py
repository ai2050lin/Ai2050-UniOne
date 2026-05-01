"""
CCI(303): 类别判别方向(LDA) perturb — 多维度组合效应
===================================================
之前发现:
  - PC方向perturb: 所有PC影响相同token(非特异)
  - 单维度perturb: 语义维度=随机维度(非特异)
  
假设: 单维度不足以产生语义效应, 需要多维度组合

本实验:
  1. 用LDA找到最佳4类别判别方向(多维组合)
  2. Hook perturb沿LDA方向 vs 沿随机方向
  3. 测量: 是否LDA方向更能改变top-1预测到目标类别?
  
数据驱动, 不预设结论

用法:
  python cciii_lda_perturb.py --model qwen3
  python cciii_lda_perturb.py --model glm4
  python cciii_lda_perturb.py --model deepseek7b
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
    load_model, get_layers, get_model_info, release_model,
    get_layer_weights, MODEL_CONFIGS,
)

TEMP_DIR = Path("tests/glm5_temp")
LOG_FILE = TEMP_DIR / "cciii_lda_perturb_log.txt"

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
    log_f(f"CCI(303): LDA Direction Perturb")
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

    cat_names = sorted(CONCEPTS.keys())
    cat_to_idx = {c: i for i, c in enumerate(cat_names)}

    word_z = {}
    word_h_tilde = {}
    word_base_logits = {}
    word_base_top1 = {}

    for word in all_words:
        text = TEMPLATE.format(word)
        input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
        last_pos = input_ids.shape[1] - 1

        with torch.no_grad():
            base_logits = model(input_ids).logits[0, last_pos].detach().float().cpu().numpy()
        word_base_logits[word] = base_logits
        word_base_top1[word] = int(np.argmax(base_logits))

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

    # ===== Step 2: Compute LDA directions =====
    log_f(f"\n--- Step 2: Computing LDA directions ---")

    target_layers = sorted(set([
        n_layers // 3,
        n_layers // 2,
        2 * n_layers // 3,
    ]))

    lda_results = {}

    for li in target_layers:
        valid_words = [w for w in all_words if li in word_z[w]]
        if len(valid_words) < 4:
            continue

        Z_matrix = np.array([word_z[w][li] for w in valid_words])  # [n, n_inter]
        cats_for_valid = np.array([cat_to_idx[word_cat[w]] for w in valid_words])

        # Use top-50 ANOVA dims for LDA (dimensionality reduction first)
        from scipy import stats
        cat_groups = defaultdict(list)
        for i, c in enumerate(cats_for_valid):
            cat_groups[int(c)].append(i)

        F_values = np.zeros(n_inter)
        for dim in range(n_inter):
            groups = [Z_matrix[indices, dim] for c, indices in cat_groups.items() if len(indices) >= 2]
            if len(groups) >= 2:
                try:
                    F, _ = stats.f_oneway(*groups)
                    F_values[dim] = F if not np.isnan(F) else 0
                except:
                    pass

        top50_dims = np.argsort(-F_values)[:50]
        Z_reduced = Z_matrix[:, top50_dims]  # [n, 50]

        # LDA: compute class means and within-class scatter
        n_classes = len(cat_names)
        class_means = np.zeros((n_classes, 50))
        overall_mean = Z_reduced.mean(axis=0)

        Sw = np.zeros((50, 50))  # within-class scatter
        for c in range(n_classes):
            indices = [i for i, cat_idx in enumerate(cats_for_valid) if cat_idx == c]
            if len(indices) < 2:
                continue
            class_data = Z_reduced[indices]
            class_means[c] = class_data.mean(axis=0)
            diff = class_data - class_means[c]
            Sw += diff.T @ diff

        # Between-class scatter
        Sb = np.zeros((50, 50))
        for c in range(n_classes):
            n_c = sum(1 for cat_idx in cats_for_valid if cat_idx == c)
            diff = (class_means[c] - overall_mean).reshape(-1, 1)
            Sb += n_c * (diff @ diff.T)

        # LDA directions: eigenvectors of Sw^-1 @ Sb
        try:
            Sw_inv = np.linalg.pinv(Sw + 1e-6 * np.eye(50))
            S = Sw_inv @ Sb
            eigenvalues, eigenvectors = np.linalg.eig(S)
            # Take real parts
            eigenvalues = np.real(eigenvalues)
            eigenvectors = np.real(eigenvectors)
            # Sort by eigenvalue
            sort_idx = np.argsort(-eigenvalues)
            eigenvalues = eigenvalues[sort_idx]
            eigenvectors = eigenvectors[:, sort_idx]

            # LDA directions in reduced space -> project back to full z space
            lda_dirs_reduced = eigenvectors[:, :3]  # top-3 LDA directions [50, 3]
            lda_dirs_full = np.zeros((n_inter, 3))
            for k in range(3):
                for j, dim in enumerate(top50_dims):
                    lda_dirs_full[dim, k] = lda_dirs_reduced[j, k]

            # Normalize
            for k in range(3):
                lda_dirs_full[:, k] /= (np.linalg.norm(lda_dirs_full[:, k]) + 1e-20)

            lda_results[li] = {
                'lda_dirs': lda_dirs_full,  # [n_inter, 3]
                'eigenvalues': eigenvalues[:3],
                'class_means_reduced': class_means,
                'top50_dims': top50_dims,
                'valid_words': valid_words,
            }

            log_f(f"\n  L{li}: LDA eigenvalues={eigenvalues[:3].round(4).tolist()}")
            for k in range(3):
                dir_k = lda_dirs_full[:, k]
                # Project each class mean along this direction
                class_projs = {}
                for c in range(n_classes):
                    indices = [i for i, cat_idx in enumerate(cats_for_valid) if cat_idx == c]
                    if indices:
                        proj = np.mean([np.dot(Z_matrix[idx], dir_k) for idx in indices])
                        class_projs[cat_names[c]] = round(proj, 3)
                log_f(f"    LDA{k}: {class_projs}")
        except Exception as e:
            log_f(f"  L{li}: LDA failed: {e}")

    # ===== Step 3: Hook perturb along LDA directions =====
    log_f(f"\n{'='*70}")
    log_f(f"Step 3: Hook perturb along LDA directions")
    log_f(f"{'='*70}")

    perturb_results = {}

    for li in sorted(lda_results.keys()):
        lda_dirs = lda_results[li]['lda_dirs']  # [n_inter, 3]
        valid_words = lda_results[li]['valid_words']
        
        lw = get_layer_weights(layers_list[li], d_model, mlp_type)
        W_down_t = torch.tensor(lw.W_down, dtype=torch.bfloat16, device=device)
        W_up = lw.W_up

        # Generate random directions for comparison (same norm as LDA)
        rng = np.random.RandomState(42)
        random_dirs = rng.randn(n_inter, 3)
        for k in range(3):
            random_dirs[:, k] /= np.linalg.norm(random_dirs[:, k])

        log_f(f"\n  === Layer L{li} ===")

        perturb_results[li] = {'lda': {}, 'random': {}}

        for dir_type, dirs in [("lda", lda_dirs), ("random", random_dirs)]:
            for k in range(3):  # 3 directions
                dir_k = dirs[:, k]

                for eps in [1.0, 3.0]:  # perturbation magnitude
                    test_words = valid_words[:8]
                    top1_changes = 0
                    total_tests = 0
                    delta_logit_norms = []
                    logit_coss = []
                    # Track: does perturbation shift prediction toward target category?
                    cat_shifts = defaultdict(int)

                    for word in test_words:
                        if li not in word_z[word]:
                            continue

                        z_orig = word_z[word][li]
                        z_perturbed = z_orig + eps * dir_k

                        z_clip = np.clip(z_perturbed, -500, 500)
                        g_perturbed = 1.0 / (1.0 + np.exp(-z_clip))

                        h_tilde = word_h_tilde[word][li]
                        u = W_up @ h_tilde

                        g_pert_t = torch.tensor(g_perturbed, dtype=torch.bfloat16, device=device)
                        u_t = torch.tensor(u, dtype=torch.bfloat16, device=device)

                        base_logits = word_base_logits[word]
                        base_top1 = word_base_top1[word]

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
                            # Which category does the new top-1 belong to?
                            new_tok = tokenizer.decode([interv_top1]).strip().lower()
                            for cat in cat_names:
                                if any(w in new_tok for w in CONCEPTS[cat]):
                                    cat_shifts[f"to_{cat}"] += 1
                                    break
                            cat_shifts["total_changed"] += 1

                        total_tests += 1
                        delta_logit_norms.append(delta_norm)
                        logit_coss.append(logit_cos)

                        del g_pert_t, u_t

                    top1_rate = top1_changes / max(total_tests, 1)
                    avg_delta_norm = float(np.mean(delta_logit_norms)) if delta_logit_norms else 0.0
                    avg_logit_cos = float(np.mean(logit_coss)) if logit_coss else 1.0

                    perturb_results[li][dir_type][f"dir{k}_eps{eps}"] = {
                        'top1_rate': top1_rate,
                        'avg_delta_norm': avg_delta_norm,
                        'avg_logit_cos': avg_logit_cos,
                        'cat_shifts': dict(cat_shifts),
                    }

                    log_f(f"    {dir_type} dir{k} eps={eps}: top1={top1_rate:.3f} "
                          f"||dL||={avg_delta_norm:.1f} cos={avg_logit_cos:.4f} "
                          f"shifts={dict(cat_shifts)}")

                    torch.cuda.empty_cache()

    # ===== Step 4: Summary comparison =====
    log_f(f"\n{'='*70}")
    log_f(f"Step 4: LDA vs Random direction comparison")
    log_f(f"{'='*70}")

    for li in sorted(perturb_results.keys()):
        log_f(f"\n  L{li}:")
        for dir_type in ['lda', 'random']:
            for key in sorted(perturb_results[li][dir_type].keys()):
                d = perturb_results[li][dir_type][key]
                log_f(f"    {dir_type} {key}: top1={d['top1_rate']:.3f} "
                      f"||dL||={d['avg_delta_norm']:.1f} cos={d['avg_logit_cos']:.4f}")

    # ===== Save =====
    json_path = TEMP_DIR / f"cciii_lda_perturb_{model_name}.json"
    save_data = {
        'model': model_name,
        'n_layers': n_layers,
        'perturb_results': {},
    }
    for li in perturb_results:
        save_data['perturb_results'][str(li)] = perturb_results[li]
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    log_f(f"  Results saved to {json_path}")

    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write('\n'.join(log_lines) + '\n')

    del model
    torch.cuda.empty_cache()
    gc.collect()

    log_f(f"\nCCI(303) {model_name} done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    run_experiment(args.model)
