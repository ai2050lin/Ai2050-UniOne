"""
Phase CCLXXV Batch: Up向量ū方向编码 — 一次加载完成全部4个实验
================================================================
优化: 避免重复加载模型和前向传播, 所有实验共享同一数据收集
"""
import os, sys, json, time, gc
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

# Setup dual output (console + log file)
LOG_FILE = Path(__file__).resolve().parent.parent / "glm5_temp" / "cclxxv_batch_log.txt"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

class TeeWriter:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except:
                pass
    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except:
                pass

_log_fh = open(LOG_FILE, 'w', encoding='utf-8', buffering=1)
sys.stdout = TeeWriter(sys.stdout, _log_fh)
sys.stderr = TeeWriter(sys.stderr, _log_fh)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, get_W_U,
    get_layer_weights, LayerWeights, compute_cos, MODEL_CONFIGS,
)

OUTPUT_DIR = ROOT / "results" / "causal_fiber"

CONCEPTS = {
    "animal": ["dog", "cat", "horse", "bird", "fish", "lion", "bear", "wolf"],
    "food": ["apple", "bread", "cheese", "rice", "cake", "pizza", "pasta", "salad"],
    "tool": ["hammer", "knife", "scissors", "saw", "drill", "wrench", "shovel", "axe"],
    "vehicle": ["car", "bus", "train", "plane", "boat", "truck", "bike", "ship"],
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


def collect_data(model_name, n_words_per_cat=4):
    """Collect gate/up/residual for all words — single model load."""
    print(f"\n  Loading model {model_name}...", flush=True)
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers_list = get_layers(model)
    W_U = get_W_U(model)

    template = "The {} is"
    rng = np.random.RandomState(42)
    all_words, all_cats = [], []
    for cat, words in CONCEPTS.items():
        sel = rng.choice(words, min(n_words_per_cat, len(words)), replace=False)
        all_words.extend(sel.tolist() if hasattr(sel, 'tolist') else list(sel))
        all_cats.extend([cat] * len(sel))

    word_gates, word_ups, word_residuals = {}, {}, {}
    cat_names = list(CONCEPTS.keys())

    t0 = time.time()
    for wi, word in enumerate(all_words):
        text = template.format(word)
        input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
        last_pos = input_ids.shape[1] - 1

        ln_out, res_out = {}, {}
        hooks = []
        for li in range(n_layers):
            layer = layers_list[li]
            if hasattr(layer, 'mlp'):
                def make_ffn_pre(key):
                    def hook(module, args):
                        a = args[0] if not isinstance(args, tuple) else args[0]
                        ln_out[key] = a[0, last_pos].detach().float().cpu().numpy()
                    return hook
                hooks.append(layer.mlp.register_forward_pre_hook(make_ffn_pre(f"L{li}")))
            def make_layer_out(key):
                def hook(module, input, output):
                    o = output[0] if isinstance(output, tuple) else output
                    res_out[key] = o[0, last_pos].detach().float().cpu().numpy()
                return hook
            hooks.append(layer.register_forward_hook(make_layer_out(f"L{li}")))

        with torch.no_grad():
            _ = model(input_ids)
        for h in hooks:
            h.remove()

        g_dict, u_dict, r_dict = {}, {}, {}
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

        elapsed = time.time() - t0
        print(f"  Word {wi+1}/{len(all_words)}: '{word}' done ({elapsed:.0f}s)", flush=True)

    print(f"  Data collection done in {time.time()-t0:.0f}s", flush=True)
    return (word_gates, word_ups, word_residuals, all_words, all_cats,
            model_info, layers_list, W_U, model, d_model, mlp_type, n_layers, cat_names)


def save_json(data, filepath):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(json_serialize(data), f, indent=2, ensure_ascii=False)
    print(f"  Saved to {filepath}")


def run_all_experiments(model_name):
    """Run all 4 experiments with a single data collection."""
    data = collect_data(model_name)
    (word_gates, word_ups, word_residuals, all_words, all_cats,
     model_info, layers_list, W_U, model, d_model, mlp_type, n_layers, cat_names) = data

    target_layers_6 = list(range(0, n_layers, max(1, n_layers // 6)))
    if n_layers - 1 not in target_layers_6:
        target_layers_6.append(n_layers - 1)
    target_layers_6 = sorted(set(target_layers_6))

    out_dir = OUTPUT_DIR / f"{model_name}_cclxxv"
    all_results = {}

    # ================================================================
    # Exp1: Δu Direction Analysis
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Exp1: Δu Direction Analysis")
    print(f"{'='*60}")
    exp1_results = []
    for li in target_layers_6:
        valid_words = [w for w in all_words if li in word_gates[w] and li in word_ups[w]]
        if len(valid_words) < 4:
            continue
        lw = get_layer_weights(layers_list[li], d_model, mlp_type)
        if lw.W_gate is None:
            continue
        W_down = lw.W_down

        cat_means = {}
        for cat in cat_names:
            cat_words = [w for w, c in zip(all_words, all_cats) if c == cat and li in word_residuals[w]]
            res_vecs = [word_residuals[w][li] for w in cat_words if word_residuals[w].get(li) is not None]
            if len(res_vecs) >= 2:
                cat_means[cat] = np.mean(res_vecs, axis=0)
        if len(cat_means) < 2:
            continue

        gate_cos_list, up_cos_list, cross_cos_list, total_cos_list = [], [], [], []
        dgu_cos_list, dg_cos_list = [], []
        gate_norm_list, up_norm_list = [], []

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

        if not gate_cos_list:
            continue
        total_norm = max(sum(gate_norm_list) + sum(up_norm_list), 1e-10)
        lr = {
            "layer": li, "n_pairs": len(gate_cos_list),
            "gate_cos": float(np.mean(gate_cos_list)),
            "up_cos": float(np.mean(up_cos_list)),
            "cross_cos": float(np.mean(cross_cos_list)),
            "total_cos": float(np.mean(total_cos_list)),
            "W_down_Du_cos": float(np.mean(dgu_cos_list)),
            "W_down_Dg_cos": float(np.mean(dg_cos_list)),
            "gate_norm_frac": float(sum(gate_norm_list) / total_norm),
            "up_norm_frac": float(sum(up_norm_list) / total_norm),
        }
        exp1_results.append(lr)
        print(f"  L{li}: gate_cos={lr['gate_cos']:+.3f}, up_cos={lr['up_cos']:+.3f}, "
              f"W_down@Δu={lr['W_down_Du_cos']:+.3f}, W_down@Δg={lr['W_down_Dg_cos']:+.3f}")

    save_json({"experiment": "exp1", "model": model_name, "timestamp": datetime.now().isoformat(),
               "layer_results": exp1_results}, out_dir / "exp1_delta_u_direction.json")
    all_results["exp1"] = exp1_results

    # ================================================================
    # Exp2: ū Category Encoding
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Exp2: ū Category Encoding")
    print(f"{'='*60}")
    exp2_results = []
    for li in target_layers_6:
        valid_words = [w for w in all_words if li in word_ups[w]]
        if len(valid_words) < 4:
            continue
        lw = get_layer_weights(layers_list[li], d_model, mlp_type)
        if lw.W_gate is None:
            continue
        W_down = lw.W_down

        word_up_proj = {w: W_down @ word_ups[w][li] for w in valid_words}

        same_cat_cos, diff_cat_cos = [], []
        for i in range(len(valid_words)):
            for j in range(i + 1, len(valid_words)):
                w1, w2 = valid_words[i], valid_words[j]
                c1 = all_cats[all_words.index(w1)]
                c2 = all_cats[all_words.index(w2)]
                c = proper_cos(word_up_proj[w1], word_up_proj[w2])
                (same_cat_cos if c1 == c2 else diff_cat_cos).append(c)

        cat_u_means = {}
        for cat in cat_names:
            cw = [w for w, c in zip(all_words, all_cats) if c == cat and li in word_ups[w]]
            if len(cw) >= 2:
                cat_u_means[cat] = np.mean([word_up_proj[w] for w in cw], axis=0)

        u_sep_ratio = -1
        if len(cat_u_means) >= 2:
            cl = sorted(cat_u_means.keys())
            inter = [float(np.linalg.norm(cat_u_means[a] - cat_u_means[b]))
                     for i, a in enumerate(cl) for j, b in enumerate(cl) if i < j]
            intra = []
            for cat in cl:
                cw = [w for w, c in zip(all_words, all_cats) if c == cat and li in word_ups[w]]
                for i2 in range(len(cw)):
                    for j2 in range(i2 + 1, len(cw)):
                        intra.append(float(np.linalg.norm(word_up_proj[cw[i2]] - word_up_proj[cw[j2]])))
            u_sep_ratio = float(np.mean(inter) / max(np.mean(intra), 1e-10))

        # Δu PCA dimensionality
        delta_u_list = []
        for i in range(len(valid_words)):
            for j in range(i + 1, len(valid_words)):
                delta_u_list.append(word_ups[valid_words[i]][li] - word_ups[valid_words[j]][li])

        n90_du, top5_var_du = -1, -1
        if len(delta_u_list) >= 10:
            du_arr = np.array(delta_u_list, dtype=np.float32)
            du_c = du_arr - du_arr.mean(axis=0)
            from scipy.sparse.linalg import svds
            n_pca = min(50, du_c.shape[0] - 1, du_c.shape[1] - 1)
            if n_pca >= 2:
                U_du, s_du, _ = svds(du_c.astype(np.float32), k=n_pca)
                s_du = s_du[np.argsort(-s_du)]
                tv = np.sum(s_du ** 2)
                n90_du = int(np.searchsorted(np.cumsum(s_du ** 2) / tv, 0.90)) + 1
                top5_var_du = float(np.sum(s_du[:5] ** 2) / tv)

        lr = {
            "layer": li,
            "same_cat_cos": float(np.mean(same_cat_cos)) if same_cat_cos else 0,
            "diff_cat_cos": float(np.mean(diff_cat_cos)) if diff_cat_cos else 0,
            "cos_sep": float(np.mean(same_cat_cos) - np.mean(diff_cat_cos)) if same_cat_cos and diff_cat_cos else 0,
            "u_sep_ratio": u_sep_ratio,
            "n90_du": n90_du,
            "top5_var_du": top5_var_du,
        }
        exp2_results.append(lr)
        print(f"  L{li}: same_cos={lr['same_cat_cos']:+.3f}, diff_cos={lr['diff_cat_cos']:+.3f}, "
              f"sep={lr['cos_sep']:+.3f}, n90(Δu)={lr['n90_du']}, u_sep={lr['u_sep_ratio']:.2f}")

    save_json({"experiment": "exp2", "model": model_name, "timestamp": datetime.now().isoformat(),
               "layer_results": exp2_results}, out_dir / "exp2_u_bar_category.json")
    all_results["exp2"] = exp2_results

    # ================================================================
    # Exp3: Δu Subspace Geometry (cross-layer + Δg↔Δu)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Exp3: Δu Subspace Geometry")
    print(f"{'='*60}")
    target_layers_8 = list(range(0, n_layers, max(1, n_layers // 8)))
    if n_layers - 1 not in target_layers_8:
        target_layers_8.append(n_layers - 1)
    target_layers_8 = sorted(set(target_layers_8))

    from scipy.sparse.linalg import svds
    layer_pcas = {}
    for li in target_layers_8:
        valid_words = [w for w in all_words if li in word_gates[w] and li in word_ups[w]]
        if len(valid_words) < 4:
            continue
        dg_list, du_list = [], []
        for i in range(len(valid_words)):
            for j in range(i + 1, len(valid_words)):
                dg_list.append(word_gates[valid_words[i]][li] - word_gates[valid_words[j]][li])
                du_list.append(word_ups[valid_words[i]][li] - word_ups[valid_words[j]][li])
        if len(dg_list) < 10:
            continue
        dg_arr = np.array(dg_list, dtype=np.float32)
        du_arr = np.array(du_list, dtype=np.float32)
        n_pca = min(30, len(dg_list) - 1, dg_arr.shape[1] - 1)
        n_pca_u = min(30, len(du_list) - 1, du_arr.shape[1] - 1)
        if n_pca < 2 or n_pca_u < 2:
            continue
        dg_c = dg_arr - dg_arr.mean(axis=0)
        _, s_dg, Vt_dg = svds(dg_c.astype(np.float32), k=n_pca)
        idx = np.argsort(-s_dg); s_dg = s_dg[idx]; Vt_dg = Vt_dg[idx]
        du_c = du_arr - du_arr.mean(axis=0)
        _, s_du, Vt_du = svds(du_c.astype(np.float32), k=n_pca_u)
        idx2 = np.argsort(-s_du); s_du = s_du[idx2]; Vt_du = Vt_du[idx2]
        n_sub = min(20, len(s_dg), len(s_du))
        layer_pcas[li] = {"Vt_dg": Vt_dg[:n_sub], "Vt_du": Vt_du[:n_sub]}

    # Cross-layer Δu rotation
    layer_list = sorted(layer_pcas.keys())
    du_rotation, same_layer = [], []
    for i, li1 in enumerate(layer_list):
        for j, li2 in enumerate(layer_list):
            if i >= j:
                continue
            V1, V2 = layer_pcas[li1]["Vt_du"], layer_pcas[li2]["Vt_du"]
            _, cos_a, _ = np.linalg.svd(V1 @ V2.T)
            du_rotation.append({
                "layer1": li1, "layer2": li2,
                "mean_cos": float(np.mean(cos_a)),
                "mean_angle": float(np.mean(np.degrees(np.arccos(np.clip(cos_a, -1, 1)))))
            })

    # Same-layer Δg↔Δu
    for li in layer_list:
        Vt_dg, Vt_du = layer_pcas[li]["Vt_dg"], layer_pcas[li]["Vt_du"]
        _, cos_a, _ = np.linalg.svd(Vt_dg @ Vt_du.T)
        same_layer.append({
            "layer": li,
            "dg_du_cos": float(np.mean(cos_a)),
            "dg_du_angle": float(np.mean(np.degrees(np.arccos(np.clip(cos_a, -1, 1)))))
        })
        print(f"  L{li}: Δg↔Δu cos={float(np.mean(cos_a)):.3f}, angle={float(np.mean(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))):.1f}°")

    # Summary
    adj = [r for r in du_rotation if abs(r["layer2"] - r["layer1"]) <= max(2, n_layers // 8)]
    if adj:
        print(f"  Δu cross-layer adjacent: avg cos={np.mean([r['mean_cos'] for r in adj]):.3f}")
    mid_sl = [r for r in same_layer if n_layers * 0.3 <= r["layer"] < n_layers * 0.7]
    if mid_sl:
        avg_dgdu = np.mean([r["dg_du_cos"] for r in mid_sl])
        print(f"  Δg↔Δu mid-layer: avg cos={avg_dgdu:.3f}")

    save_json({"experiment": "exp3", "model": model_name, "timestamp": datetime.now().isoformat(),
               "du_cross_layer": du_rotation, "dg_du_same_layer": same_layer},
              out_dir / "exp3_delta_u_subspace.json")
    all_results["exp3"] = {"du_cross_layer": du_rotation, "dg_du_same_layer": same_layer}

    # ================================================================
    # Exp4: Δg-Δu Coordination
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Exp4: Δg-Δu Coordination")
    print(f"{'='*60}")
    exp4_results = []
    for li in target_layers_6:
        valid_words = [w for w in all_words if li in word_gates[w] and li in word_ups[w]]
        if len(valid_words) < 4:
            continue
        dg_list, du_list, pair_cats = [], [], []
        for i in range(len(valid_words)):
            for j in range(i + 1, len(valid_words)):
                dg_list.append(word_gates[valid_words[i]][li] - word_gates[valid_words[j]][li])
                du_list.append(word_ups[valid_words[i]][li] - word_ups[valid_words[j]][li])
                pair_cats.append((all_cats[all_words.index(valid_words[i])],
                                  all_cats[all_words.index(valid_words[j])]))
        if len(dg_list) < 10:
            continue

        dg_arr = np.array(dg_list, dtype=np.float32)
        du_arr = np.array(du_list, dtype=np.float32)

        # Neuron-level correlation
        pair_corrs = []
        for k in range(len(dg_list)):
            abs_dg, abs_du = np.abs(dg_arr[k]), np.abs(du_arr[k])
            if np.std(abs_dg) < 1e-10 or np.std(abs_du) < 1e-10:
                continue
            pair_corrs.append(float(np.corrcoef(abs_dg, abs_du)[0, 1]))

        # Top-k overlap
        k_val = 100
        overlaps = []
        for k in range(len(dg_list)):
            top_dg = set(np.argsort(np.abs(dg_arr[k]))[-k_val:])
            top_du = set(np.argsort(np.abs(du_arr[k]))[-k_val:])
            overlaps.append(len(top_dg & top_du) / k_val)
        n_inter = dg_arr.shape[1]
        random_overlap = k_val / n_inter

        # W_down weighted correlation
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

        # Category-dependent
        same_cat_corrs = [pair_corrs[k] for k, (c1, c2) in enumerate(pair_cats)
                          if c1 == c2 and k < len(pair_corrs)]
        diff_cat_corrs = [pair_corrs[k] for k, (c1, c2) in enumerate(pair_cats)
                          if c1 != c2 and k < len(pair_corrs)]

        lr = {
            "layer": li, "n_inter": n_inter,
            "neuron_corr": float(np.mean(pair_corrs)) if pair_corrs else 0,
            "weighted_corr": float(np.mean(weighted_corrs)) if weighted_corrs else 0,
            "overlap_100": float(np.mean(overlaps)),
            "random_overlap": float(random_overlap),
            "overlap_ratio": float(np.mean(overlaps) / random_overlap) if random_overlap > 0 else 0,
            "same_cat_corr": float(np.mean(same_cat_corrs)) if same_cat_corrs else 0,
            "diff_cat_corr": float(np.mean(diff_cat_corrs)) if diff_cat_corrs else 0,
        }
        exp4_results.append(lr)
        print(f"  L{li}: corr={lr['neuron_corr']:.3f}, weighted={lr['weighted_corr']:.3f}, "
              f"overlap(100)={lr['overlap_100']:.3f} vs rand={lr['random_overlap']:.3f} ({lr['overlap_ratio']:.2f}x)")

    save_json({"experiment": "exp4", "model": model_name, "timestamp": datetime.now().isoformat(),
               "layer_results": exp4_results}, out_dir / "exp4_dg_du_coordination.json")
    all_results["exp4"] = exp4_results

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'#'*60}")
    print(f"ALL EXPERIMENTS COMPLETE for {model_name}")
    print(f"{'#'*60}")

    mid1 = [r for r in exp1_results if n_layers * 0.3 <= r["layer"] < n_layers * 0.7]
    if mid1:
        print(f"\n  Exp1 Mid-layer summary:")
        print(f"    gate_cos     = {np.mean([r['gate_cos'] for r in mid1]):+.3f}")
        print(f"    up_cos       = {np.mean([r['up_cos'] for r in mid1]):+.3f}")
        print(f"    W_down@Δu    = {np.mean([r['W_down_Du_cos'] for r in mid1]):+.3f}")
        print(f"    W_down@Δg    = {np.mean([r['W_down_Dg_cos'] for r in mid1]):+.3f}")
        print(f"    gate_norm%   = {np.mean([r['gate_norm_frac'] for r in mid1]):.3f}")
        print(f"    up_norm%     = {np.mean([r['up_norm_frac'] for r in mid1]):.3f}")

    mid2 = [r for r in exp2_results if n_layers * 0.3 <= r["layer"] < n_layers * 0.7]
    if mid2:
        print(f"\n  Exp2 Mid-layer summary:")
        print(f"    cos_sep      = {np.mean([r['cos_sep'] for r in mid2]):+.3f}")
        print(f"    u_sep_ratio  = {np.mean([r['u_sep_ratio'] for r in mid2 if r['u_sep_ratio'] > 0]):.2f}")
        print(f"    n90(Δu)      = {np.mean([r['n90_du'] for r in mid2 if r['n90_du'] > 0]):.1f}")

    if mid_sl:
        print(f"\n  Exp3 Mid-layer: Δg↔Δu cos = {avg_dgdu:.3f}")

    mid4 = [r for r in exp4_results if n_layers * 0.3 <= r["layer"] < n_layers * 0.7]
    if mid4:
        print(f"\n  Exp4 Mid-layer summary:")
        print(f"    neuron_corr  = {np.mean([r['neuron_corr'] for r in mid4]):.3f}")
        print(f"    weighted_corr= {np.mean([r['weighted_corr'] for r in mid4]):.3f}")
        print(f"    overlap_ratio= {np.mean([r['overlap_ratio'] for r in mid4]):.2f}x")

    release_model(model)
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    run_all_experiments(args.model)
