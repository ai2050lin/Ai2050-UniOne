"""
Phase CCLXXV: Up向量ū的方向编码 — 门控-调制分离的后半部分
=========================================================
核心发现(CCLXXIV): Δg是选择器(W_down@Δg cos≈0), ū是信号载体(Δg⊙ū才对齐概念)
  - Δg子空间跨层正交(cos≈0.04)
  - W_down@Δg与概念方向无关(cos≈0.003)
  - 但gate_term = W_down@(Δg⊙ū)的cos=+0.37 (CCLXXIII)

本Phase要回答:
1. W_down@(ḡ⊙Δu)与概念方向的对齐度是多少? up_term的方向能力?
2. ū(Up向量均值)是否携带类别信息? 不同类别ū的方向差异?
3. Δu子空间几何: 跨层是否正交? 与Δg子空间的关系?
4. Δg与Δu的协同: Δg大的神经元对应的Δu是否也大? 协同还是独立?

实验:
  Exp1: Δu的方向分析 — W_down@(ḡ⊙Δu) vs W_down@(Δg⊙ū) vs 概念方向
  Exp2: ū的类别编码 — ū是否携带类别信息? 类别间ū的cos
  Exp3: Δu子空间几何 — 跨层正交? 与Δg子空间关系?
  Exp4: Δg与Δu的协同 — 逐神经元相关 + 概念选择协同度

用法:
  python phase_cclxxv_up_direction.py --model qwen3 --exp 1
  python phase_cclxxv_up_direction.py --model qwen3 --exp all
"""
import argparse, os, sys, json, time, gc
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, get_W_U,
    get_layer_weights, LayerWeights, compute_cos, MODEL_CONFIGS,
)

OUTPUT_DIR = ROOT / "results" / "causal_fiber"

CONCEPTS = {
    "animal": ["dog", "cat", "horse", "bird", "fish", "lion", "bear", "wolf", "deer", "rabbit"],
    "food": ["apple", "bread", "cheese", "rice", "cake", "pizza", "pasta", "salad", "soup", "steak"],
    "tool": ["hammer", "knife", "scissors", "saw", "drill", "wrench", "shovel", "axe", "chisel", "pliers"],
    "vehicle": ["car", "bus", "train", "plane", "boat", "truck", "bike", "ship", "subway", "tractor"],
}


def proper_cos(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def json_serialize(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: json_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_serialize(x) for x in obj]
    return obj


def collect_data(model_name, n_words_per_cat=8):
    """Collect gate and up activations. Only store minimal data."""
    print(f"\n  Loading model {model_name}...")
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers_list = get_layers(model)
    W_U = get_W_U(model)

    template = "The {} is"

    rng = np.random.RandomState(42)
    all_words = []
    all_cats = []
    for cat, words in CONCEPTS.items():
        sel = rng.choice(words, min(n_words_per_cat, len(words)), replace=False)
        all_words.extend(sel.tolist() if hasattr(sel, 'tolist') else list(sel))
        all_cats.extend([cat] * len(sel))

    word_gates = {}   # word -> {li: gate_vec}
    word_ups = {}     # word -> {li: up_vec}
    word_residuals = {}  # word -> {li: residual_vec}

    t0 = time.time()
    for wi, word in enumerate(all_words):
        text = template.format(word)
        input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
        last_pos = input_ids.shape[1] - 1

        ln_out = {}
        res_out = {}
        hooks = []
        for li in range(n_layers):
            layer = layers_list[li]
            if hasattr(layer, 'mlp'):
                def make_ffn_pre(key):
                    def hook(module, args):
                        if isinstance(args, tuple):
                            ln_out[key] = args[0][0, last_pos].detach().float().cpu().numpy()
                        else:
                            ln_out[key] = args[0, last_pos].detach().float().cpu().numpy()
                    return hook
                hooks.append(layer.mlp.register_forward_pre_hook(make_ffn_pre(f"L{li}")))
            def make_layer_out(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        res_out[key] = output[0][0, last_pos].detach().float().cpu().numpy()
                    else:
                        res_out[key] = output[0, last_pos].detach().float().cpu().numpy()
                return hook
            hooks.append(layer.register_forward_hook(make_layer_out(f"L{li}")))

        with torch.no_grad():
            _ = model(input_ids)

        for h in hooks:
            h.remove()

        g_dict = {}
        u_dict = {}
        r_dict = {}
        for li in range(n_layers):
            key = f"L{li}"
            lw = get_layer_weights(layers_list[li], d_model, mlp_type)
            if lw.W_gate is None or key not in ln_out:
                continue
            h_input = ln_out[key]
            z = lw.W_gate @ h_input
            z_clipped = np.clip(z, -500, 500)
            g = 1.0 / (1.0 + np.exp(-z_clipped))
            u = lw.W_up @ h_input
            g_dict[li] = g
            u_dict[li] = u
            r_dict[li] = res_out.get(key, None)

        word_gates[word] = g_dict
        word_ups[word] = u_dict
        word_residuals[word] = r_dict

        if (wi + 1) % 4 == 0 or wi == len(all_words) - 1:
            elapsed = time.time() - t0
            print(f"  Collected {wi+1}/{len(all_words)} words ({elapsed:.0f}s)")

    # Free GPU memory (keep model for W_down access)
    return (word_gates, word_ups, word_residuals, all_words, all_cats,
            model_info, layers_list, W_U, model)


# ============================================================
# Exp1: Δu的方向分析 — up项 vs gate项 vs 概念方向
# ============================================================
def exp1_delta_u_direction(model_name):
    print(f"\n{'='*70}")
    print(f"Exp1: Δu Direction Analysis — Gate vs Up vs Cross Terms")
    print(f"  Model: {model_name}")
    print(f"{'='*70}")

    (word_gates, word_ups, word_residuals, all_words, all_cats,
     model_info, layers_list, W_U, model) = collect_data(model_name)

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    cat_names = list(CONCEPTS.keys())

    target_layers = list(range(0, n_layers, max(1, n_layers // 6)))
    if n_layers - 1 not in target_layers:
        target_layers.append(n_layers - 1)
    target_layers = sorted(set(target_layers))

    layer_results = []

    for li in target_layers:
        valid_words = [w for w in all_words if li in word_gates[w] and li in word_ups[w]]
        if len(valid_words) < 4:
            continue

        lw = get_layer_weights(layers_list[li], d_model, mlp_type)
        if lw.W_gate is None:
            continue
        W_down = lw.W_down  # [d_model, n_inter]

        # Compute concept directions from residuals
        cat_means = {}
        for cat in cat_names:
            cat_words = [w for w, c in zip(all_words, all_cats) if c == cat and li in word_residuals[w]]
            res_vecs = [word_residuals[w][li] for w in cat_words
                        if word_residuals[w].get(li) is not None]
            if len(res_vecs) < 2:
                continue
            cat_means[cat] = np.mean(res_vecs, axis=0)

        if len(cat_means) < 2:
            continue

        gate_cos_list, up_cos_list, cross_cos_list, total_cos_list = [], [], [], []
        dgu_cos_list, dg_cos_list = [], []
        gate_norm_list, up_norm_list, cross_norm_list = [], [], []

        cat_list = sorted(cat_means.keys())
        for i, cA in enumerate(cat_list):
            for j, cB in enumerate(cat_list):
                if i >= j:
                    continue
                concept_dir = cat_means[cA] - cat_means[cB]
                if np.linalg.norm(concept_dir) < 1e-8:
                    continue

                words_A = [w for w, c in zip(all_words, all_cats) if c == cA and li in word_gates[w]]
                words_B = [w for w, c in zip(all_words, all_cats) if c == cB and li in word_gates[w]]

                g_A = np.mean([word_gates[w][li] for w in words_A], axis=0)
                g_B = np.mean([word_gates[w][li] for w in words_B], axis=0)
                u_A = np.mean([word_ups[w][li] for w in words_A], axis=0)
                u_B = np.mean([word_ups[w][li] for w in words_B], axis=0)

                Dg = g_A - g_B
                Du = u_A - u_B
                g_bar = (g_A + g_B) / 2
                u_bar = (u_A + u_B) / 2

                gate_vec = W_down @ (Dg * u_bar)
                up_vec = W_down @ (g_bar * Du)
                cross_vec = W_down @ (Dg * Du)
                total_vec = gate_vec + up_vec + cross_vec

                gate_cos_list.append(proper_cos(gate_vec, concept_dir))
                up_cos_list.append(proper_cos(up_vec, concept_dir))
                cross_cos_list.append(proper_cos(cross_vec, concept_dir))
                total_cos_list.append(proper_cos(total_vec, concept_dir))
                dgu_cos_list.append(proper_cos(W_down @ Du, concept_dir))
                dg_cos_list.append(proper_cos(W_down @ Dg, concept_dir))

                gate_norm_list.append(float(np.linalg.norm(gate_vec)))
                up_norm_list.append(float(np.linalg.norm(up_vec)))
                cross_norm_list.append(float(np.linalg.norm(cross_vec)))

        if not gate_cos_list:
            continue

        total_norm = max(sum(gate_norm_list) + sum(up_norm_list) + sum(cross_norm_list), 1e-10)
        lr = {
            "layer": li,
            "n_pairs": len(gate_cos_list),
            "gate_cos_mean": float(np.mean(gate_cos_list)),
            "gate_cos_abs_mean": float(np.mean(np.abs(gate_cos_list))),
            "up_cos_mean": float(np.mean(up_cos_list)),
            "up_cos_abs_mean": float(np.mean(np.abs(up_cos_list))),
            "cross_cos_mean": float(np.mean(cross_cos_list)),
            "cross_cos_abs_mean": float(np.mean(np.abs(cross_cos_list))),
            "total_cos_mean": float(np.mean(total_cos_list)),
            "total_cos_abs_mean": float(np.mean(np.abs(total_cos_list))),
            "dgu_cos_mean": float(np.mean(dgu_cos_list)),
            "dgu_cos_abs_mean": float(np.mean(np.abs(dgu_cos_list))),
            "dg_cos_mean": float(np.mean(dg_cos_list)),
            "dg_cos_abs_mean": float(np.mean(np.abs(dg_cos_list))),
            "gate_norm_frac": float(sum(gate_norm_list) / total_norm),
            "up_norm_frac": float(sum(up_norm_list) / total_norm),
        }
        layer_results.append(lr)

        print(f"  L{li}: gate_cos={lr['gate_cos_mean']:+.3f}, up_cos={lr['up_cos_mean']:+.3f}, "
              f"cross_cos={lr['cross_cos_mean']:+.3f}, total_cos={lr['total_cos_mean']:+.3f}")
        print(f"         dgu_cos(W_down@Δu)={lr['dgu_cos_mean']:+.3f}, dg_cos(W_down@Δg)={lr['dg_cos_mean']:+.3f}")
        print(f"         norm_frac: gate={lr['gate_norm_frac']:.3f}, up={lr['up_norm_frac']:.3f}")

    # Summary
    print(f"\n  Summary:")
    mid = [lr for lr in layer_results if n_layers * 0.3 <= lr["layer"] < n_layers * 0.7]
    if mid:
        print(f"    Mid-layer avg:")
        print(f"      gate_cos = {np.mean([l['gate_cos_mean'] for l in mid]):+.3f} "
              f"(abs={np.mean([l['gate_cos_abs_mean'] for l in mid]):.3f})")
        print(f"      up_cos   = {np.mean([l['up_cos_mean'] for l in mid]):+.3f} "
              f"(abs={np.mean([l['up_cos_abs_mean'] for l in mid]):.3f})")
        print(f"      cross_cos= {np.mean([l['cross_cos_mean'] for l in mid]):+.3f}")
        print(f"      total_cos= {np.mean([l['total_cos_mean'] for l in mid]):+.3f}")
        print(f"      dgu_cos  = {np.mean([l['dgu_cos_mean'] for l in mid]):+.3f}")
        print(f"      dg_cos   = {np.mean([l['dg_cos_mean'] for l in mid]):+.3f}")
        print(f"      gate_norm_frac={np.mean([l['gate_norm_frac'] for l in mid]):.3f}, "
              f"up_norm_frac={np.mean([l['up_norm_frac'] for l in mid]):.3f}")

    out_dir = OUTPUT_DIR / f"{model_name}_cclxxv"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "exp1_delta_u_direction",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "layer_results": layer_results,
    }
    with open(out_dir / "exp1_delta_u_direction.json", 'w', encoding='utf-8') as f:
        json.dump(json_serialize(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_dir / 'exp1_delta_u_direction.json'}")

    release_model(model)
    return summary


# ============================================================
# Exp2: ū的类别编码
# ============================================================
def exp2_u_bar_category_encoding(model_name):
    print(f"\n{'='*70}")
    print(f"Exp2: ū Category Encoding")
    print(f"  Model: {model_name}")
    print(f"{'='*70}")

    (word_gates, word_ups, word_residuals, all_words, all_cats,
     model_info, layers_list, W_U, model) = collect_data(model_name)

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    cat_names = list(CONCEPTS.keys())

    target_layers = list(range(0, n_layers, max(1, n_layers // 6)))
    if n_layers - 1 not in target_layers:
        target_layers.append(n_layers - 1)
    target_layers = sorted(set(target_layers))

    layer_results = []

    for li in target_layers:
        valid_words = [w for w in all_words if li in word_ups[w]]
        if len(valid_words) < 4:
            continue

        lw = get_layer_weights(layers_list[li], d_model, mlp_type)
        if lw.W_gate is None:
            continue
        W_down = lw.W_down

        # W_down @ u for each word
        word_up_proj = {}
        word_u_raw = {}
        for w in valid_words:
            u = word_ups[w][li]
            word_up_proj[w] = W_down @ u
            word_u_raw[w] = u

        # Pairwise cos
        same_cat_cos, diff_cat_cos = [], []
        for i in range(len(valid_words)):
            for j in range(i + 1, len(valid_words)):
                w1, w2 = valid_words[i], valid_words[j]
                c1 = all_cats[all_words.index(w1)]
                c2 = all_cats[all_words.index(w2)]
                c = proper_cos(word_up_proj[w1], word_up_proj[w2])
                if c1 == c2:
                    same_cat_cos.append(c)
                else:
                    diff_cat_cos.append(c)

        # Category separation
        cat_u_means = {}
        for cat in cat_names:
            cat_words = [w for w, c in zip(all_words, all_cats) if c == cat and li in word_ups[w]]
            if len(cat_words) < 2:
                continue
            cat_u_means[cat] = np.mean([word_up_proj[w] for w in cat_words], axis=0)

        if len(cat_u_means) >= 2:
            cat_list = sorted(cat_u_means.keys())
            inter_dists = [float(np.linalg.norm(cat_u_means[cA] - cat_u_means[cB]))
                          for i, cA in enumerate(cat_list) for j, cB in enumerate(cat_list) if i < j]
            intra_dists = []
            for cat in cat_list:
                cw = [w for w, c in zip(all_words, all_cats) if c == cat and li in word_ups[w]]
                for i2 in range(len(cw)):
                    for j2 in range(i2 + 1, len(cw)):
                        intra_dists.append(float(np.linalg.norm(word_up_proj[cw[i2]] - word_up_proj[cw[j2]])))
            avg_inter = np.mean(inter_dists) if inter_dists else 0
            avg_intra = np.mean(intra_dists) if intra_dists else 1e-10
            u_sep_ratio = avg_inter / max(avg_intra, 1e-10)
        else:
            u_sep_ratio = -1

        # Δu PCA: n90
        delta_u_list = []
        pair_cats_du = []
        for i in range(len(valid_words)):
            for j in range(i + 1, len(valid_words)):
                w1, w2 = valid_words[i], valid_words[j]
                c1 = all_cats[all_words.index(w1)]
                c2 = all_cats[all_words.index(w2)]
                delta_u_list.append(word_u_raw[w1] - word_u_raw[w2])
                pair_cats_du.append((c1, c2))

        n90_du, top5_var_du = -1, -1
        classification_acc = {}
        if len(delta_u_list) >= 10:
            delta_u_arr = np.array(delta_u_list, dtype=np.float32)
            du_centered = delta_u_arr - delta_u_arr.mean(axis=0)
            from scipy.sparse.linalg import svds
            n_pca = min(50, du_centered.shape[0] - 1, du_centered.shape[1] - 1)
            if n_pca >= 2:
                U_du, s_du, Vt_du = svds(du_centered.astype(np.float32), k=n_pca)
                sort_idx = np.argsort(-s_du)
                s_du = s_du[sort_idx]
                total_var = np.sum(s_du ** 2)
                cum_var = np.cumsum(s_du ** 2) / total_var
                n90_du = int(np.searchsorted(cum_var, 0.90)) + 1
                top5_var_du = float(np.sum(s_du[:5] ** 2) / total_var)

                # Classification
                proj_du = U_du[:, :min(20, n_pca)]
                is_diff_cat = np.array([c1 != c2 for c1, c2 in pair_cats_du])
                from sklearn.linear_model import LogisticRegression
                from sklearn.model_selection import cross_val_score
                for k in [5, 10, 20]:
                    if k > min(20, n_pca):
                        continue
                    X = proj_du[:, :k]
                    y = is_diff_cat.astype(int)
                    if len(np.unique(y)) < 2 or len(y) < 10:
                        continue
                    try:
                        clf = LogisticRegression(max_iter=500, solver='lbfgs')
                        scores = cross_val_score(clf, X, y, cv=min(5, len(y) // 3), scoring='accuracy')
                        classification_acc[k] = float(np.mean(scores))
                    except:
                        classification_acc[k] = -1

        lr = {
            "layer": li,
            "same_cat_cos_mean": float(np.mean(same_cat_cos)) if same_cat_cos else 0,
            "diff_cat_cos_mean": float(np.mean(diff_cat_cos)) if diff_cat_cos else 0,
            "cos_separation": float(np.mean(same_cat_cos) - np.mean(diff_cat_cos)) if same_cat_cos and diff_cat_cos else 0,
            "u_sep_ratio": u_sep_ratio,
            "n90_du": n90_du,
            "top5_var_du": top5_var_du,
            "classification_acc": classification_acc,
        }
        layer_results.append(lr)

        print(f"  L{li}: same_cos={lr['same_cat_cos_mean']:+.3f}, diff_cos={lr['diff_cat_cos_mean']:+.3f}, "
              f"sep={lr['cos_separation']:+.3f}, u_sep_ratio={lr['u_sep_ratio']:.2f}")
        print(f"         n90(Δu)={lr['n90_du']}, top5_var={lr['top5_var_du']:.3f}")

    mid = [lr for lr in layer_results if n_layers * 0.3 <= lr["layer"] < n_layers * 0.7]
    if mid:
        avg_sep = np.mean([lr["cos_separation"] for lr in mid])
        avg_u_sep = np.mean([lr["u_sep_ratio"] for lr in mid if lr["u_sep_ratio"] > 0])
        avg_n90 = np.mean([lr["n90_du"] for lr in mid if lr["n90_du"] > 0])
        print(f"\n  Summary: cos_sep={avg_sep:+.3f}, u_sep_ratio={avg_u_sep:.2f}, n90(Δu)={avg_n90:.1f}")

    out_dir = OUTPUT_DIR / f"{model_name}_cclxxv"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "exp2_u_bar_category_encoding",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "layer_results": layer_results,
    }
    with open(out_dir / "exp2_u_bar_category.json", 'w', encoding='utf-8') as f:
        json.dump(json_serialize(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_dir / 'exp2_u_bar_category.json'}")

    release_model(model)
    return summary


# ============================================================
# Exp3: Δu子空间几何
# ============================================================
def exp3_delta_u_subspace_geometry(model_name):
    print(f"\n{'='*70}")
    print(f"Exp3: Δu Subspace Geometry")
    print(f"  Model: {model_name}")
    print(f"{'='*70}")

    (word_gates, word_ups, word_residuals, all_words, all_cats,
     model_info, layers_list, W_U, model) = collect_data(model_name)

    n_layers = model_info.n_layers

    target_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    if n_layers - 1 not in target_layers:
        target_layers.append(n_layers - 1)
    target_layers = sorted(set(target_layers))

    from scipy.sparse.linalg import svds

    layer_pcas = {}
    for li in target_layers:
        valid_words = [w for w in all_words if li in word_gates[w] and li in word_ups[w]]
        if len(valid_words) < 4:
            continue

        delta_g_list, delta_u_list = [], []
        for i in range(len(valid_words)):
            for j in range(i + 1, len(valid_words)):
                w1, w2 = valid_words[i], valid_words[j]
                delta_g_list.append(word_gates[w1][li] - word_gates[w2][li])
                delta_u_list.append(word_ups[w1][li] - word_ups[w2][li])

        if len(delta_g_list) < 10:
            continue

        dg_arr = np.array(delta_g_list, dtype=np.float32)
        du_arr = np.array(delta_u_list, dtype=np.float32)

        n_pca = min(30, len(delta_g_list) - 1, dg_arr.shape[1] - 1)
        n_pca_u = min(30, len(delta_u_list) - 1, du_arr.shape[1] - 1)
        if n_pca < 2 or n_pca_u < 2:
            continue

        # PCA Δg
        dg_c = dg_arr - dg_arr.mean(axis=0)
        _, s_dg, Vt_dg = svds(dg_c.astype(np.float32), k=n_pca)
        idx = np.argsort(-s_dg); s_dg = s_dg[idx]; Vt_dg = Vt_dg[idx]

        # PCA Δu
        du_c = du_arr - du_arr.mean(axis=0)
        _, s_du, Vt_du = svds(du_c.astype(np.float32), k=n_pca_u)
        idx2 = np.argsort(-s_du); s_du = s_du[idx2]; Vt_du = Vt_du[idx2]

        n_sub = min(20, len(s_dg), len(s_du))
        layer_pcas[li] = {
            "Vt_dg": Vt_dg[:n_sub], "Vt_du": Vt_du[:n_sub],
            "s_dg": s_dg[:n_sub], "s_du": s_du[:n_sub],
        }

        tv_dg = np.sum(s_dg ** 2)
        tv_du = np.sum(s_du ** 2)
        n90_dg = int(np.searchsorted(np.cumsum(s_dg ** 2) / tv_dg, 0.90)) + 1
        n90_du = int(np.searchsorted(np.cumsum(s_du ** 2) / tv_du, 0.90)) + 1
        print(f"  L{li}: n90(Δg)={n90_dg}, n90(Δu)={n90_du}")

    print(f"  Computed PCA for {len(layer_pcas)} layers")

    # Part A: Cross-layer Δu rotation
    layer_list = sorted(layer_pcas.keys())
    du_rotation = []
    for i, li1 in enumerate(layer_list):
        for j, li2 in enumerate(layer_list):
            if i >= j:
                continue
            V1, V2 = layer_pcas[li1]["Vt_du"], layer_pcas[li2]["Vt_du"]
            _, cos_a, _ = np.linalg.svd(V1 @ V2.T)
            mean_cos = float(np.mean(cos_a))
            mean_angle = float(np.mean(np.degrees(np.arccos(np.clip(cos_a, -1, 1)))))
            cap = float(np.mean(np.sum((V2 @ V1.T) ** 2, axis=0)))
            du_rotation.append({"layer1": li1, "layer2": li2, "mean_cos": mean_cos, "mean_angle_deg": mean_angle, "captured_frac": cap})
            if abs(li2 - li1) <= 2 or li1 == 0 or li2 == n_layers - 1:
                print(f"  Δu L{li1}->L{li2}: cos={mean_cos:.3f}, angle={mean_angle:.1f}°, captured={cap:.3f}")

    # Part B: Δg vs Δu at same layer
    same_layer = []
    for li in layer_list:
        Vt_dg, Vt_du = layer_pcas[li]["Vt_dg"], layer_pcas[li]["Vt_du"]
        _, cos_a, _ = np.linalg.svd(Vt_dg @ Vt_du.T)
        mean_cos = float(np.mean(cos_a))
        mean_angle = float(np.mean(np.degrees(np.arccos(np.clip(cos_a, -1, 1)))))
        cap = float(np.mean(np.sum((Vt_du @ Vt_dg.T) ** 2, axis=0)))
        same_layer.append({"layer": li, "dg_du_cos": mean_cos, "dg_du_angle": mean_angle, "dg_du_captured": cap})
        print(f"  L{li}: Δg↔Δu cos={mean_cos:.3f}, angle={mean_angle:.1f}°, captured={cap:.3f}")

    # Summary
    adj = [r for r in du_rotation if abs(r["layer2"] - r["layer1"]) <= max(2, n_layers // 8)]
    if adj:
        print(f"\n  Δu adjacent: avg cos={np.mean([r['mean_cos'] for r in adj]):.3f}, "
              f"avg angle={np.mean([r['mean_angle_deg'] for r in adj]):.1f}°")
    mid_same = [r for r in same_layer if n_layers * 0.3 <= r["layer"] < n_layers * 0.7]
    if mid_same:
        avg_dg_du = np.mean([r["dg_du_cos"] for r in mid_same])
        print(f"  Δg↔Δu same-layer mid: avg cos={avg_dg_du:.3f}")
        if avg_dg_du < 0.1:
            print(f"  >>> ★ Δg and Δu subspaces ORTHOGONAL!")
        elif avg_dg_du > 0.5:
            print(f"  >>> Δg and Δu subspaces ALIGNED!")
        else:
            print(f"  >>> Δg and Δu subspaces PARTIALLY related")

    out_dir = OUTPUT_DIR / f"{model_name}_cclxxv"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "exp3_delta_u_subspace_geometry",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "du_cross_layer_rotation": du_rotation,
        "dg_du_same_layer": same_layer,
    }
    with open(out_dir / "exp3_delta_u_subspace.json", 'w', encoding='utf-8') as f:
        json.dump(json_serialize(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_dir / 'exp3_delta_u_subspace.json'}")

    release_model(model)
    return summary


# ============================================================
# Exp4: Δg与Δu的协同
# ============================================================
def exp4_dg_du_coordination(model_name):
    print(f"\n{'='*70}")
    print(f"Exp4: Δg-Δu Coordination")
    print(f"  Model: {model_name}")
    print(f"{'='*70}")

    (word_gates, word_ups, word_residuals, all_words, all_cats,
     model_info, layers_list, W_U, model) = collect_data(model_name)

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type

    target_layers = list(range(0, n_layers, max(1, n_layers // 6)))
    if n_layers - 1 not in target_layers:
        target_layers.append(n_layers - 1)
    target_layers = sorted(set(target_layers))

    layer_results = []

    for li in target_layers:
        valid_words = [w for w in all_words if li in word_gates[w] and li in word_ups[w]]
        if len(valid_words) < 4:
            continue

        dg_list, du_list, pair_cat_list = [], [], []
        for i in range(len(valid_words)):
            for j in range(i + 1, len(valid_words)):
                w1, w2 = valid_words[i], valid_words[j]
                c1 = all_cats[all_words.index(w1)]
                c2 = all_cats[all_words.index(w2)]
                dg_list.append(word_gates[w1][li] - word_gates[w2][li])
                du_list.append(word_ups[w1][li] - word_ups[w2][li])
                pair_cat_list.append((c1, c2))

        if len(dg_list) < 10:
            continue

        dg_arr = np.array(dg_list, dtype=np.float32)
        du_arr = np.array(du_list, dtype=np.float32)

        # Test 1: neuron-level correlation
        pair_corrs = []
        for k in range(len(dg_list)):
            abs_dg, abs_du = np.abs(dg_arr[k]), np.abs(du_arr[k])
            if np.std(abs_dg) < 1e-10 or np.std(abs_du) < 1e-10:
                continue
            pair_corrs.append(float(np.corrcoef(abs_dg, abs_du)[0, 1]))

        # Test 2: top-k overlap
        k_values = [50, 100, 200, 500]
        avg_overlaps = {}
        for kv in k_values:
            overlaps = []
            for k in range(len(dg_list)):
                top_dg = set(np.argsort(np.abs(dg_arr[k]))[-kv:])
                top_du = set(np.argsort(np.abs(du_arr[k]))[-kv:])
                overlaps.append(len(top_dg & top_du) / kv)
            avg_overlaps[kv] = float(np.mean(overlaps)) if overlaps else 0

        n_inter = dg_arr.shape[1]
        random_overlaps = {kv: float(kv / n_inter) for kv in k_values}

        # Test 3: category-dependent
        same_cat_corrs = [pair_corrs[k] for k, (c1, c2) in enumerate(pair_cat_list) if c1 == c2 and k < len(pair_corrs)]
        diff_cat_corrs = [pair_corrs[k] for k, (c1, c2) in enumerate(pair_cat_list) if c1 != c2 and k < len(pair_corrs)]

        # Test 4: W_down weighted
        lw = get_layer_weights(layers_list[li], d_model, mlp_type)
        W_down = lw.W_down
        w_col_norms = np.linalg.norm(W_down, axis=0)

        weighted_corrs = []
        for k in range(len(dg_list)):
            abs_dg_w = np.abs(dg_arr[k]) * w_col_norms
            abs_du_w = np.abs(du_arr[k]) * w_col_norms
            if np.std(abs_dg_w) < 1e-10 or np.std(abs_du_w) < 1e-10:
                continue
            weighted_corrs.append(float(np.corrcoef(abs_dg_w, abs_du_w)[0, 1]))

        lr = {
            "layer": li,
            "n_inter": n_inter,
            "neuron_corr_mean": float(np.mean(pair_corrs)) if pair_corrs else 0,
            "neuron_corr_std": float(np.std(pair_corrs)) if pair_corrs else 0,
            "weighted_corr_mean": float(np.mean(weighted_corrs)) if weighted_corrs else 0,
            "top_k_overlap": avg_overlaps,
            "random_overlap": random_overlaps,
            "overlap_ratio_100": float(avg_overlaps.get(100, 0) / max(random_overlaps.get(100, 1e-10), 1e-10)),
            "same_cat_corr": float(np.mean(same_cat_corrs)) if same_cat_corrs else 0,
            "diff_cat_corr": float(np.mean(diff_cat_corrs)) if diff_cat_corrs else 0,
        }
        layer_results.append(lr)

        print(f"  L{li}: corr={lr['neuron_corr_mean']:.3f}±{lr['neuron_corr_std']:.3f}, "
              f"weighted={lr['weighted_corr_mean']:.3f}")
        print(f"         overlap(100)={avg_overlaps.get(100,0):.3f} vs random={random_overlaps.get(100,0):.3f} "
              f"({lr['overlap_ratio_100']:.2f}x)")

    mid = [lr for lr in layer_results if n_layers * 0.3 <= lr["layer"] < n_layers * 0.7]
    if mid:
        avg_corr = np.mean([l["neuron_corr_mean"] for l in mid])
        avg_wcorr = np.mean([l["weighted_corr_mean"] for l in mid])
        avg_ratio = np.mean([l["overlap_ratio_100"] for l in mid])
        print(f"\n  Summary: corr={avg_corr:.3f}, weighted={avg_wcorr:.3f}, overlap_ratio={avg_ratio:.2f}x")
        if avg_corr > 0.3:
            print(f"  >>> ★ Δg and Δu CORRELATED!")
        elif avg_corr > 0.1:
            print(f"  >>> Δg and Δu WEAKLY correlated")
        else:
            print(f"  >>> Δg and Δu LARGELY INDEPENDENT")

    out_dir = OUTPUT_DIR / f"{model_name}_cclxxv"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "exp4_dg_du_coordination",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "layer_results": layer_results,
    }
    with open(out_dir / "exp4_dg_du_coordination.json", 'w', encoding='utf-8') as f:
        json.dump(json_serialize(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_dir / 'exp4_dg_du_coordination.json'}")

    release_model(model)
    return summary


# ============================================================
# Main
# ============================================================
EXPS = {
    1: exp1_delta_u_direction,
    2: exp2_u_bar_category_encoding,
    3: exp3_delta_u_subspace_geometry,
    4: exp4_dg_du_coordination,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, required=True,
                        help="Experiment number (1-4) or 'all'")
    args = parser.parse_args()

    if args.exp == "all":
        for exp_id in sorted(EXPS.keys()):
            print(f"\n{'#'*70}")
            print(f"# Running Exp{exp_id}")
            print(f"{'#'*70}")
            EXPS[exp_id](args.model)
            gc.collect()
    else:
        exp_id = int(args.exp)
        if exp_id not in EXPS:
            print(f"Invalid exp: {args.exp}. Must be 1-4 or 'all'")
            return
        EXPS[exp_id](args.model)


if __name__ == "__main__":
    main()
