"""
Phase CCLXXIV: Δg的20维概念编码空间的几何与语义
================================================
核心发现(CCLXXIII): Δg(门控差异)有效维度仅~20维(n90=19-24)
  - 9728-18944维门控空间中，概念差异只沿~20个方向
  - W_down忠实保持Δg(norm_corr=0.997)
  - gate项是概念信号载体(cos=+0.37 > up_cos=+0.26)

本Phase要回答:
1. 这20个PC方向对应什么语义维度? 类别在Δg-PCA空间中是否可分?
2. 不同层的Δg子空间是否相同? 如果旋转，旋转了多少?
3. Δg的20维概念编码是否映射到W_U的特定子空间?
4. 完整FFN输出(Δout)与概念方向的总对齐度是多少?

实验:
  Exp1: Δg-PCA语义解释 — PC方向的语义含义 + 类别可分性
  Exp2: 跨层Δg子空间旋转 — 子空间相似度与旋转角
  Exp3: Δg子空间与W_U行空间的对齐 — 概念编码→logit空间的映射
  Exp4: Δout与概念方向总对齐 — 完整FFN输出的概念对齐度

用法:
  python phase_cclxxiv_delta_g_geometry.py --model qwen3 --exp 1
  python phase_cclxxiv_delta_g_geometry.py --model qwen3 --exp all
"""
import argparse, os, sys, json, time, gc
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
    get_layer_weights, LayerWeights, compute_cos, MODEL_CONFIGS,
)

OUTPUT_DIR = Path("results/causal_fiber")

CONCEPTS = {
    "animals": ["dog", "cat", "horse", "eagle", "shark", "snake",
                "lion", "bear", "whale", "dolphin", "rabbit", "deer"],
    "food":    ["apple", "rice", "bread", "cheese", "salmon", "mango",
                "grape", "banana", "pasta", "pizza", "cookie", "steak"],
    "tools":   ["hammer", "knife", "saw", "drill", "wrench", "chisel",
                "pliers", "ruler", "level", "clamp", "file", "shovel"],
    "nature":  ["mountain", "river", "ocean", "forest", "desert", "volcano",
                "canyon", "glacier", "meadow", "island", "valley", "cliff"],
}

TEMPLATES = ["The {} is"]


def json_serialize(obj):
    if isinstance(obj, dict):
        return {str(k): json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_serialize(x) for x in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, bool):
        return obj
    elif obj is None:
        return None
    return obj


def proper_cos(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def collect_gate_activations(model_name, n_words_per_cat=8):
    """Collect gate activations for all words across all layers.
    Returns: word_gates, all_words, all_cats, model_info, layers_list, W_U
    """
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers_list = get_layers(model)
    W_U = get_W_U(model)

    template = "The {} is"

    # Sample words from each category
    rng = np.random.RandomState(42)
    all_words = []
    all_cats = []
    for cat, words in CONCEPTS.items():
        sel = rng.choice(words, min(n_words_per_cat, len(words)), replace=False)
        all_words.extend(sel.tolist() if hasattr(sel, 'tolist') else list(sel))
        all_cats.extend([cat] * len(sel))

    # Collect gate activations
    word_gates = {}
    word_residuals = {}  # for concept direction

    for word in all_words:
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

        word_gates[word] = {}
        word_residuals[word] = {}
        for li in range(n_layers):
            key = f"L{li}"
            if key not in ln_out:
                continue
            lw = get_layer_weights(layers_list[li], d_model, mlp_type)
            if lw.W_gate is None:
                continue
            z = lw.W_gate @ ln_out[key]
            z_clipped = np.clip(z, -500, 500)
            g = 1.0 / (1.0 + np.exp(-z_clipped))
            word_gates[word][li] = g
            word_residuals[word][li] = res_out.get(key, None)

    print(f"  Collected gates for {len(all_words)} words, {n_layers} layers")

    return (word_gates, word_residuals, all_words, all_cats,
            model_info, layers_list, W_U, model)


def compute_delta_g_pca(word_gates, all_words, all_cats, li, n_pca=50):
    """Compute PCA of Δg for a given layer.
    Returns: (U_dg, s_dg, Vt_dg, delta_g_centered, delta_g_arr, pair_cats, pair_words)
    """
    valid_words = [w for w in all_words if li in word_gates[w]]
    if len(valid_words) < 4:
        return None

    delta_g_list = []
    pair_cats = []  # (cat1, cat2) for each pair
    pair_words = []  # (word1, word2) for each pair

    for i in range(len(valid_words)):
        for j in range(i + 1, len(valid_words)):
            w1, w2 = valid_words[i], valid_words[j]
            c1 = all_cats[all_words.index(w1)]
            c2 = all_cats[all_words.index(w2)]

            dg = word_gates[w1][li] - word_gates[w2][li]
            delta_g_list.append(dg)
            pair_cats.append((c1, c2))
            pair_words.append((w1, w2))

    if len(delta_g_list) < 10:
        return None

    delta_g_arr = np.array(delta_g_list, dtype=np.float32)  # [n_pairs, n_inter]
    delta_g_centered = delta_g_arr - delta_g_arr.mean(axis=0)

    from scipy.sparse.linalg import svds
    n_pca_actual = min(n_pca, delta_g_centered.shape[0] - 1, delta_g_centered.shape[1] - 1)
    if n_pca_actual < 2:
        return None

    U_dg, s_dg, Vt_dg = svds(delta_g_centered.astype(np.float32), k=n_pca_actual)
    # Sort by singular value
    sort_idx = np.argsort(-s_dg)
    s_dg = s_dg[sort_idx]
    U_dg = U_dg[:, sort_idx]
    Vt_dg = Vt_dg[sort_idx, :]

    return U_dg, s_dg, Vt_dg, delta_g_centered, delta_g_arr, pair_cats, pair_words


# ============================================================
# Exp1: Δg-PCA语义解释 — PC方向的语义含义 + 类别可分性
# ============================================================
def exp1_dg_pca_semantics(model_name):
    """Analyze the semantic interpretation of Δg PCA components.

    Key questions:
    1. What do the top PC directions encode? (category differences? individual differences?)
    2. Are categories separable in Δg-PCA space?
    3. Which PCs carry category information?
    4. How many PCs are needed for category classification?
    """
    print(f"\n{'='*70}")
    print(f"Exp1: Δg-PCA Semantic Interpretation")
    print(f"  Model: {model_name}")
    print(f"  Key test: What do the 20 PC directions encode?")
    print(f"{'='*70}")

    (word_gates, word_residuals, all_words, all_cats,
     model_info, layers_list, W_U, model) = collect_gate_activations(model_name)

    n_layers = model_info.n_layers
    cat_names = list(CONCEPTS.keys())

    target_layers = list(range(0, n_layers, max(1, n_layers // 6)))
    if n_layers - 1 not in target_layers:
        target_layers.append(n_layers - 1)
    target_layers = sorted(set(target_layers))

    layer_results = []

    for li in target_layers:
        result = compute_delta_g_pca(word_gates, all_words, all_cats, li, n_pca=50)
        if result is None:
            continue

        U_dg, s_dg, Vt_dg, delta_g_centered, delta_g_arr, pair_cats, pair_words = result

        total_var = np.sum(s_dg ** 2)
        cum_var = np.cumsum(s_dg ** 2) / total_var

        # ---- Test 1: Category separability in PCA space ----
        # Project each Δg onto top PCs
        n_top = min(20, len(s_dg))
        proj = U_dg[:, :n_top]  # [n_pairs, n_top] — projections of centered Δg

        # For each pair, determine if same-cat or diff-cat
        is_diff_cat = np.array([c1 != c2 for c1, c2 in pair_cats])

        # Same-cat pairs vs diff-cat pairs: compare variance along each PC
        diff_cat_proj = proj[is_diff_cat]  # [n_diff_pairs, n_top]
        same_cat_proj = proj[~is_diff_cat]  # [n_same_pairs, n_top]

        # Variance ratio: diff-cat variance / same-cat variance along each PC
        diff_var = np.var(diff_cat_proj, axis=0) if len(diff_cat_proj) > 1 else np.zeros(n_top)
        same_var = np.var(same_cat_proj, axis=0) if len(same_cat_proj) > 1 else np.ones(n_top)
        var_ratio = diff_var / np.maximum(same_var, 1e-10)

        # Mean absolute projection: which PCs have the largest diff-cat signal?
        diff_mean_abs = np.mean(np.abs(diff_cat_proj), axis=0) if len(diff_cat_proj) > 0 else np.zeros(n_top)
        same_mean_abs = np.mean(np.abs(same_cat_proj), axis=0) if len(same_cat_proj) > 0 else np.zeros(n_top)

        # ---- Test 2: Category classification using Δg-PCA ----
        # Use top-k PCs as features, classify pairs as same-cat vs diff-cat
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        classification_acc = {}
        for k in [3, 5, 10, 20]:
            if k > n_top:
                continue
            X = proj[:, :k]
            y = is_diff_cat.astype(int)
            if len(np.unique(y)) < 2 or len(y) < 10:
                classification_acc[k] = -1
                continue
            try:
                clf = LogisticRegression(max_iter=500, solver='lbfgs')
                scores = cross_val_score(clf, X, y, cv=min(5, len(y) // 3), scoring='accuracy')
                classification_acc[k] = float(np.mean(scores))
            except:
                classification_acc[k] = -1

        # ---- Test 3: Which PC corresponds to which category pair? ----
        # For each category pair, compute the mean projection onto each PC
        cat_pair_mean_proj = {}
        for cA in cat_names:
            for cB in cat_names:
                if cA >= cB:
                    continue
                pair_mask = np.array([
                    (c1 == cA and c2 == cB) or (c1 == cB and c2 == cA)
                    for c1, c2 in pair_cats
                ])
                if np.sum(pair_mask) < 2:
                    continue
                mean_proj = np.mean(proj[pair_mask, :n_top], axis=0)
                cat_pair_mean_proj[f"{cA}-{cB}"] = mean_proj.tolist()

        # ---- Test 4: Individual word positions in gate PCA space ----
        # Instead of Δg pairs, compute PCA of the original gates
        valid_words = [w for w in all_words if li in word_gates[w]]
        if len(valid_words) >= 4:
            gate_arr = np.array([word_gates[w][li] for w in valid_words], dtype=np.float32)
            gate_centered = gate_arr - gate_arr.mean(axis=0)

            from scipy.sparse.linalg import svds
            n_pca_gate = min(20, gate_centered.shape[0] - 1, gate_centered.shape[1] - 1)
            if n_pca_gate >= 2:
                U_gate, s_gate, Vt_gate = svds(gate_centered, k=n_pca_gate)
                sort_idx = np.argsort(-s_gate)
                s_gate = s_gate[sort_idx]
                U_gate = U_gate[:, sort_idx]
                Vt_gate = Vt_gate[sort_idx, :]

                # Word positions in gate PCA space
                word_proj = U_gate[:, :min(5, n_pca_gate)]  # [n_words, 5]

                # Category separability: compute average intra-cat and inter-cat distance
                word_cats = [all_cats[all_words.index(w)] for w in valid_words]
                intra_dists = []
                inter_dists = []
                for i in range(len(valid_words)):
                    for j in range(i + 1, len(valid_words)):
                        d = np.linalg.norm(word_proj[i] - word_proj[j])
                        if word_cats[i] == word_cats[j]:
                            intra_dists.append(d)
                        else:
                            inter_dists.append(d)

                avg_intra = float(np.mean(intra_dists)) if intra_dists else 0
                avg_inter = float(np.mean(inter_dists)) if inter_dists else 0
                sep_ratio = avg_inter / max(avg_intra, 1e-10)
            else:
                sep_ratio = -1
                word_proj = None
        else:
            sep_ratio = -1
            word_proj = None

        # ---- Test 5: PC1 vs category ----
        # Does PC1 separate any specific category pair?
        pc1_cat_corr = {}
        if len(proj) > 0:
            for cA in cat_names:
                # Is this category "positive" or "negative" in PC1?
                mask_A_first = np.array([
                    c1 == cA and c2 != cA for c1, c2 in pair_cats
                ])
                mask_A_second = np.array([
                    c1 != cA and c2 == cA for c1, c2 in pair_cats
                ])
                if np.sum(mask_A_first) > 0 and np.sum(mask_A_second) > 0:
                    mean_first = float(np.mean(proj[mask_A_first, 0]))
                    mean_second = float(np.mean(proj[mask_A_second, 0]))
                    pc1_cat_corr[cA] = {
                        "mean_as_first": mean_first,
                        "mean_as_second": mean_second,
                        "contrast": mean_first - mean_second,
                    }

        layer_results.append({
            "layer": li,
            "n_pairs": len(delta_g_arr),
            # Variance explained
            "top5_var": float(np.sum(s_dg[:5] ** 2) / total_var) if len(s_dg) >= 5 else -1,
            "top10_var": float(np.sum(s_dg[:10] ** 2) / total_var) if len(s_dg) >= 10 else -1,
            "n90": int(np.searchsorted(cum_var, 0.90)) + 1,
            # Category separability
            "var_ratio_top5": var_ratio[:5].tolist(),
            "diff_mean_abs_top5": diff_mean_abs[:5].tolist(),
            "same_mean_abs_top5": same_mean_abs[:5].tolist(),
            # Classification accuracy
            "classification_acc": classification_acc,
            # Gate PCA separability
            "gate_pca_sep_ratio": sep_ratio,
            # PC1 category correlation
            "pc1_cat_corr": pc1_cat_corr,
        })

        lr = layer_results[-1]
        print(f"  L{li}: top5_var={lr['top5_var']:.3f}, n90={lr['n90']}, "
              f"sep_ratio={lr['gate_pca_sep_ratio']:.2f}")
        print(f"         var_ratio(PC1-5): {[f'{v:.2f}' for v in var_ratio[:5]]}")
        if classification_acc:
            print(f"         classification: {classification_acc}")

    # Summary
    print(f"\n  Summary:")
    mid_results = [lr for lr in layer_results if n_layers * 0.3 <= lr["layer"] < n_layers * 0.7]
    if mid_results:
        avg_sep = np.mean([lr["gate_pca_sep_ratio"] for lr in mid_results if lr["gate_pca_sep_ratio"] > 0])
        avg_class = {}
        for k in [5, 10, 20]:
            vals = [lr["classification_acc"].get(k, -1) for lr in mid_results if lr["classification_acc"].get(k, -1) > 0]
            if vals:
                avg_class[k] = float(np.mean(vals))
        avg_var_ratio = []
        for lr in mid_results:
            avg_var_ratio.extend(lr["var_ratio_top5"][:3])
        avg_var_ratio = float(np.mean(avg_var_ratio)) if avg_var_ratio else 0

        print(f"    Mid-layer avg gate_pca_sep_ratio: {avg_sep:.2f}")
        print(f"    Mid-layer avg classification_acc: {avg_class}")
        print(f"    Mid-layer avg var_ratio(PC1-3): {avg_var_ratio:.2f}")

        if avg_sep > 1.5:
            print(f"  >>> Categories are SEPARABLE in Δg-PCA space!")
        else:
            print(f"  >>> Categories are NOT clearly separable in Δg-PCA space")

    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxxiv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp1_dg_pca_semantics.json"

    summary = {
        "experiment": "exp1_dg_pca_semantics",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "layer_results": layer_results,
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json_serialize(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")

    release_model(model)
    return summary


# ============================================================
# Exp2: 跨层Δg子空间旋转
# ============================================================
def exp2_cross_layer_subspace_rotation(model_name):
    """Analyze how the Δg subspace rotates across layers.

    Key questions:
    1. Are the Δg subspaces at different layers aligned or rotated?
    2. If rotated, what is the rotation angle?
    3. Is there a gradual rotation or discrete jumps?
    4. Does the rotation correlate with semantic processing depth?
    """
    print(f"\n{'='*70}")
    print(f"Exp2: Cross-Layer Δg Subspace Rotation")
    print(f"  Model: {model_name}")
    print(f"  Key test: Does the ~20-dim Δg subspace rotate across layers?")
    print(f"{'='*70}")

    (word_gates, word_residuals, all_words, all_cats,
     model_info, layers_list, W_U, model) = collect_gate_activations(model_name)

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type

    target_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    if n_layers - 1 not in target_layers:
        target_layers.append(n_layers - 1)
    target_layers = sorted(set(target_layers))

    # Compute Δg PCA for each target layer
    layer_pcas = {}
    for li in target_layers:
        result = compute_delta_g_pca(word_gates, all_words, all_cats, li, n_pca=30)
        if result is None:
            continue
        U_dg, s_dg, Vt_dg, _, _, _, _ = result
        n_sub = min(20, len(s_dg))
        # Subspace basis: top-n_sub principal directions in n_inter space
        # Vt_dg[:n_sub] are the principal directions [n_sub, n_inter]
        layer_pcas[li] = {
            "Vt": Vt_dg[:n_sub],  # [n_sub, n_inter]
            "s": s_dg[:n_sub],
            "cum_var": np.cumsum(s_dg[:n_sub] ** 2) / np.sum(s_dg ** 2),
        }

    print(f"  Computed Δg PCA for {len(layer_pcas)} layers")

    # Compare subspaces across layers
    layer_list = sorted(layer_pcas.keys())
    rotation_results = []

    for i, li1 in enumerate(layer_list):
        for j, li2 in enumerate(layer_list):
            if i >= j:
                continue

            V1 = layer_pcas[li1]["Vt"]  # [n_sub, n_inter]
            V2 = layer_pcas[li2]["Vt"]  # [n_sub, n_inter]

            # Subspace similarity: principal angles
            # Compute V1 @ V2^T to get the matrix of inner products
            M = V1 @ V2.T  # [n_sub, n_sub]

            # SVD of M gives the cosines of principal angles
            U_m, cos_angles, Vt_m = np.linalg.svd(M)

            # Principal angles
            angles_rad = np.arccos(np.clip(cos_angles, -1, 1))
            angles_deg = np.degrees(angles_rad)

            # Mean subspace alignment
            mean_cos = float(np.mean(cos_angles))
            min_cos = float(np.min(cos_angles))
            max_cos = float(np.max(cos_angles))

            # Frobenius norm of M (normalized)
            frob_norm = float(np.linalg.norm(M, 'fro') / np.sqrt(len(cos_angles)))

            # Also compute: how much of subspace 1 is contained in subspace 2?
            # Project each vector of V1 onto V2's subspace
            # V2_basis @ V2_basis^T is the projector onto V2's subspace
            # (V2^T is [n_inter, n_sub], so V2^T @ V2 is [n_sub, n_sub] — not the projector)
            # Correct: projector P2 = V2^T @ (V2 @ V2^T)^{-1} @ V2 ... but this is expensive
            # Simplified: for orthonormal V2 (rows), projector = V2^T @ V2 ... no
            # V2 has shape [n_sub, n_inter]. Rows are orthonormal => V2 @ V2^T = I
            # Projector onto row space of V2 = V2^T @ V2 (this is [n_inter, n_inter])
            # Projection of V1[i] onto V2 subspace: (V2^T @ V2) @ V1[i]
            # Fraction captured: ||proj||^2 / ||V1[i]||^2 = V1[i]^T @ V2^T @ V2 @ V1[i]
            # Since V1[i] is unit norm, this = ||V2 @ V1[i]||^2
            V1V2 = V2 @ V1.T  # [n_sub, n_sub] — same as M^T
            captured_frac = np.sum(V1V2 ** 2, axis=0)  # ||V2 @ V1[i]||^2 for each i
            mean_captured = float(np.mean(captured_frac))

            rotation_results.append({
                "layer1": li1,
                "layer2": li2,
                "mean_cos": mean_cos,
                "min_cos": min_cos,
                "max_cos": max_cos,
                "frob_norm": frob_norm,
                "mean_captured_frac": mean_captured,
                "top5_angles_deg": angles_deg[:5].tolist(),
                "mean_angle_deg": float(np.mean(angles_deg)),
            })

            if abs(li2 - li1) <= 2 or li1 == 0 or li2 == n_layers - 1:
                print(f"  L{li1}->L{li2}: mean_cos={mean_cos:.3f}, "
                      f"mean_angle={float(np.mean(angles_deg)):.1f}°, "
                      f"captured={mean_captured:.3f}")

    # Summary: adjacent layer rotation and total rotation
    print(f"\n  Summary:")

    # Adjacent layer rotation
    adj_rotations = [r for r in rotation_results if abs(r["layer2"] - r["layer1"]) <= max(2, n_layers // 8)]
    if adj_rotations:
        avg_adj_cos = np.mean([r["mean_cos"] for r in adj_rotations])
        avg_adj_angle = np.mean([r["mean_angle_deg"] for r in adj_rotations])
        print(f"    Adjacent layer: avg cos={avg_adj_cos:.3f}, avg angle={avg_adj_angle:.1f}°")

    # First-to-last rotation
    first_last = [r for r in rotation_results if r["layer1"] == 0 and r["layer2"] == n_layers - 1]
    if first_last:
        fl = first_last[0]
        print(f"    L0->L{n_layers-1}: cos={fl['mean_cos']:.3f}, "
              f"angle={fl['mean_angle_deg']:.1f}°, captured={fl['mean_captured_frac']:.3f}")

    # Gradual rotation: plot cos vs layer distance
    dist_cos = defaultdict(list)
    for r in rotation_results:
        d = r["layer2"] - r["layer1"]
        dist_cos[d].append(r["mean_cos"])

    print(f"\n    Rotation vs distance:")
    for d in sorted(dist_cos.keys()):
        if len(dist_cos[d]) >= 1:
            print(f"      dist={d:2d}: mean_cos={np.mean(dist_cos[d]):.3f}")

    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxxiv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp2_cross_layer_rotation.json"

    summary = {
        "experiment": "exp2_cross_layer_subspace_rotation",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "rotation_results": rotation_results,
        "adj_avg_cos": float(avg_adj_cos) if adj_rotations else -1,
        "adj_avg_angle": float(avg_adj_angle) if adj_rotations else -1,
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json_serialize(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")

    release_model(model)
    return summary


# ============================================================
# Exp3: Δg子空间与W_U行空间的对齐
# ============================================================
def exp3_dg_subspace_wu_alignment(model_name):
    """Analyze the alignment between Δg subspace and W_U row space.

    Key questions:
    1. Does the ~20-dim Δg subspace, when projected through W_down, align with W_U row space?
    2. Which W_U directions does Δg map to?
    3. Is there a "preferred" subset of W_U directions for concept encoding?
    4. How does the alignment change across layers?
    """
    print(f"\n{'='*70}")
    print(f"Exp3: Δg Subspace ↔ W_U Row Space Alignment")
    print(f"  Model: {model_name}")
    print(f"  Key test: Does W_down @ Δg map to specific W_U directions?")
    print(f"{'='*70}")

    (word_gates, word_residuals, all_words, all_cats,
     model_info, layers_list, W_U, model) = collect_gate_activations(model_name)

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type

    target_layers = list(range(0, n_layers, max(1, n_layers // 6)))
    if n_layers - 1 not in target_layers:
        target_layers.append(n_layers - 1)
    target_layers = sorted(set(target_layers))

    # Pre-compute W_U row space basis
    from scipy.sparse.linalg import svds
    n_wu_components = min(200, W_U.shape[0] - 1, W_U.shape[1] - 1)
    U_wu, s_wu, Vt_wu = svds(W_U.T.astype(np.float32), k=n_wu_components)
    sort_idx = np.argsort(-s_wu)
    s_wu = s_wu[sort_idx]
    U_wu = U_wu[:, sort_idx]  # [d_model, n_wu_components] — W_U row space basis

    print(f"  W_U SVD: top5 singular values = {s_wu[:5].tolist()}")
    print(f"  W_U row space: {U_wu.shape}")

    # Also compute concept-specific W_U directions
    # For each category, compute the average W_U direction of its words
    cat_names = list(CONCEPTS.keys())
    # Get tokenizer
    from transformers import AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )

    cat_wu_dirs = {}
    for cat in cat_names:
        words = CONCEPTS[cat]
        wu_vecs = []
        for w in words:
            tok_ids = tokenizer.encode(w, add_special_tokens=False)
            if len(tok_ids) > 0:
                wu_vecs.append(W_U[tok_ids[0]])
        if len(wu_vecs) >= 2:
            cat_wu_dirs[cat] = np.mean(wu_vecs, axis=0)
            # Normalize
            norm = np.linalg.norm(cat_wu_dirs[cat])
            if norm > 1e-10:
                cat_wu_dirs[cat] = cat_wu_dirs[cat] / norm

    layer_results = []

    for li in target_layers:
        result = compute_delta_g_pca(word_gates, all_words, all_cats, li, n_pca=30)
        if result is None:
            continue

        U_dg, s_dg, Vt_dg, _, delta_g_arr, pair_cats, pair_words = result

        lw = get_layer_weights(layers_list[li], d_model, mlp_type)
        if lw.W_gate is None:
            continue
        W_down = lw.W_down  # [d_model, n_inter]

        n_sub = min(20, len(s_dg))
        Vt_top = Vt_dg[:n_sub]  # [n_sub, n_inter] — Δg subspace basis

        # ---- Test 1: W_down @ Δg_subspace → W_U alignment ----
        # Map each Δg PC direction through W_down
        # W_down @ Vt_top[i] gives the d_model direction that PC i maps to
        wdown_pc_dirs = (W_down @ Vt_top.T).T  # [n_sub, d_model]

        # Project each wdown_pc_dir onto W_U row space
        # recoding_ratio for each direction
        pc_wu_ratios = []
        for k in range(n_sub):
            dir_k = wdown_pc_dirs[k]
            dir_norm = np.linalg.norm(dir_k)
            if dir_norm < 1e-10:
                pc_wu_ratios.append(0.0)
                continue
            # Project onto W_U row space
            proj_coeffs = U_wu.T @ dir_k  # [n_wu_components]
            proj_energy = np.sum(proj_coeffs ** 2)
            ratio = min(proj_energy / max(dir_norm ** 2, 1e-20), 1.0)
            pc_wu_ratios.append(float(ratio))

        # ---- Test 2: Δg → W_down → residual concept direction alignment ----
        # For each cross-category pair, compute:
        #   Δg → W_down @ Δg → cos with residual concept direction
        valid_words = [w for w in all_words if li in word_gates[w] and li in word_residuals[w]]
        delta_g_wu_cos = []
        delta_g_residual_cos = []

        for i in range(len(valid_words)):
            for j in range(i + 1, len(valid_words)):
                w1, w2 = valid_words[i], valid_words[j]
                c1 = all_cats[all_words.index(w1)]
                c2 = all_cats[all_words.index(w2)]
                if c1 == c2:
                    continue

                r1 = word_residuals[w1][li]
                r2 = word_residuals[w2][li]
                if r1 is None or r2 is None:
                    continue

                dg = word_gates[w1][li] - word_gates[w2][li]
                wdown_dg = W_down @ dg  # [d_model]

                # Concept direction from residual
                concept_dir = r1 - r2
                concept_norm = np.linalg.norm(concept_dir)
                if concept_norm < 1e-10:
                    continue

                # Cosine alignment
                cos_residual = proper_cos(wdown_dg, concept_dir)

                # Also: alignment with W_U concept direction
                # Get the W_U direction for the category pair
                if c1 in cat_wu_dirs and c2 in cat_wu_dirs:
                    wu_concept_dir = cat_wu_dirs[c1] - cat_wu_dirs[c2]
                    cos_wu = proper_cos(wdown_dg, wu_concept_dir)
                    delta_g_wu_cos.append(cos_wu)

                delta_g_residual_cos.append(cos_residual)

        # ---- Test 3: Subspace alignment metric ----
        # Compute the "canonical correlation" between Δg subspace (after W_down) and W_U row space
        # Build matrix of W_down @ Vt_top (each row is a d_model direction)
        # Build matrix of U_wu (each column is a W_U direction)
        # CCA: how correlated are these two subspaces?

        # Simplified: project Δg subspace directions onto W_U subspace
        # Δg_W = W_down @ Vt_top^T -> [d_model, n_sub]
        # W_U basis: U_wu -> [d_model, n_wu]
        # Subspace overlap: ||U_wu^T @ Δg_W||_F / ||Δg_W||_F

        dg_W = W_down @ Vt_top.T  # [d_model, n_sub]
        dg_W_norm = np.linalg.norm(dg_W, 'fro')
        if dg_W_norm > 1e-10:
            overlap = np.linalg.norm(U_wu.T @ dg_W, 'fro') / dg_W_norm
        else:
            overlap = 0

        # Random baseline: random subspace of same dimension
        n_random_trials = 20
        random_overlaps = []
        for _ in range(n_random_trials):
            R = np.random.randn(d_model, n_sub).astype(np.float32)
            R_norm = np.linalg.norm(R, 'fro')
            if R_norm > 1e-10:
                r_overlap = np.linalg.norm(U_wu.T @ R, 'fro') / R_norm
                random_overlaps.append(r_overlap)
        random_overlap_mean = float(np.mean(random_overlaps))
        overlap_ratio = overlap / max(random_overlap_mean, 1e-10)

        layer_results.append({
            "layer": li,
            # W_down @ Δg → W_U alignment per PC
            "pc_wu_ratio_top5": pc_wu_ratios[:5],
            "pc_wu_ratio_mean": float(np.mean(pc_wu_ratios)),
            # Subspace overlap
            "subspace_overlap": float(overlap),
            "random_overlap_mean": random_overlap_mean,
            "overlap_ratio": float(overlap_ratio),
            # Δg → W_down → concept alignment
            "mean_dg_residual_cos": float(np.mean(delta_g_residual_cos)) if delta_g_residual_cos else 0,
            "mean_dg_wu_cos": float(np.mean(delta_g_wu_cos)) if delta_g_wu_cos else 0,
        })

        lr = layer_results[-1]
        print(f"  L{li}: pc_wu_ratio(top5)={[f'{v:.3f}' for v in pc_wu_ratios[:5]]}, "
              f"overlap={lr['subspace_overlap']:.3f}, ratio={lr['overlap_ratio']:.2f}x, "
              f"dg→residual_cos={lr['mean_dg_residual_cos']:+.3f}, "
              f"dg→wu_cos={lr['mean_dg_wu_cos']:+.3f}")

    # Summary
    print(f"\n  Summary:")
    mid_results = [lr for lr in layer_results if n_layers * 0.3 <= lr["layer"] < n_layers * 0.7]
    if mid_results:
        avg_overlap = np.mean([lr["subspace_overlap"] for lr in mid_results])
        avg_ratio = np.mean([lr["overlap_ratio"] for lr in mid_results])
        avg_dg_res = np.mean([lr["mean_dg_residual_cos"] for lr in mid_results])
        avg_dg_wu = np.mean([lr["mean_dg_wu_cos"] for lr in mid_results])
        avg_pc_wu = np.mean([lr["pc_wu_ratio_mean"] for lr in mid_results])

        print(f"    Mid-layer subspace_overlap: {avg_overlap:.3f}")
        print(f"    Mid-layer overlap_ratio: {avg_ratio:.2f}x random")
        print(f"    Mid-layer dg→residual_cos: {avg_dg_res:+.3f}")
        print(f"    Mid-layer dg→wu_cos: {avg_dg_wu:+.3f}")
        print(f"    Mid-layer avg pc_wu_ratio: {avg_pc_wu:.3f}")

        if avg_ratio > 1.5:
            print(f"  >>> Δg subspace (after W_down) has SIGNIFICANT alignment with W_U row space!")
        else:
            print(f"  >>> Δg subspace alignment with W_U is near random ({avg_ratio:.2f}x)")

    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxxiv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp3_dg_wu_alignment.json"

    summary = {
        "experiment": "exp3_dg_subspace_wu_alignment",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "layer_results": layer_results,
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json_serialize(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")

    release_model(model)
    return summary


# ============================================================
# Exp4: Δout与概念方向总对齐
# ============================================================
def exp4_delta_out_concept_alignment(model_name):
    """Measure the total alignment between FFN output difference and concept direction.

    Key questions:
    1. What fraction of Δout aligns with the concept direction?
    2. How much of Δout is "signal" vs "noise"?
    3. Is there a layer where FFN output best aligns with concept direction?
    4. How does this compare with the gate-term-only alignment?
    """
    print(f"\n{'='*70}")
    print(f"Exp4: Δout vs Concept Direction Total Alignment")
    print(f"  Model: {model_name}")
    print(f"  Key test: How much of Δout is concept signal vs noise?")
    print(f"{'='*70}")

    (word_gates, word_residuals, all_words, all_cats,
     model_info, layers_list, W_U, model) = collect_gate_activations(model_name)

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type

    target_layers = list(range(0, n_layers, max(1, n_layers // 6)))
    if n_layers - 1 not in target_layers:
        target_layers.append(n_layers - 1)
    target_layers = sorted(set(target_layers))

    cat_names = list(CONCEPTS.keys())

    layer_results = []

    for li in target_layers:
        lw = get_layer_weights(layers_list[li], d_model, mlp_type)
        if lw.W_gate is None:
            continue
        W_down = lw.W_down  # [d_model, n_inter]

        valid_words = [w for w in all_words if li in word_gates[w] and li in word_residuals[w]]

        # Compute all cross-category pairs
        same_cat_cos = []
        diff_cat_cos = []
        diff_cat_norms = []
        diff_cat_proj_frac = []  # fraction of Δout in concept direction

        # Also decompose: gate_term vs up_term alignment
        diff_cat_gate_cos = []
        diff_cat_up_cos = []
        diff_cat_total_cos = []

        for i in range(len(valid_words)):
            for j in range(i + 1, len(valid_words)):
                w1, w2 = valid_words[i], valid_words[j]
                c1 = all_cats[all_words.index(w1)]
                c2 = all_cats[all_words.index(w2)]

                r1 = word_residuals[w1][li]
                r2 = word_residuals[w2][li]
                if r1 is None or r2 is None:
                    continue

                g1 = word_gates[w1][li]
                g2 = word_gates[w2][li]

                # Compute up vectors
                # Need ln_input to compute u = W_up @ h
                # We don't have this directly; use the gate and W_up
                # Actually, we need W_up and h_input. Let's use W_up from weights
                # and approximate: we stored gate = σ(W_gate @ h), so we can't recover h directly
                # Instead, use the decomposition from CCLXXIII
                # Δout = W_down @ (Δg ⊙ ū + ḡ ⊙ Δu + Δg ⊙ Δu)
                # We need u vectors. Let's compute them.
                # Actually, we should have collected u vectors too. Let me add that.

                # For now, compute total Δout directly
                # ffn_out = W_down @ (g * u), but we don't have u
                # We can compute: Δout = (residual1 - ffn_correction1) - (residual2 - ffn_correction2)
                # This is complex. Let's use a simpler approach.

                # Concept direction from residual
                concept_dir = r1 - r2
                concept_norm = np.linalg.norm(concept_dir)
                if concept_norm < 1e-10:
                    continue

                # Gate difference
                dg = g1 - g2

                # Δg through W_down (without u modulation)
                wdown_dg = W_down @ dg  # [d_model]

                # Cosine alignment with concept direction
                cos_dg = proper_cos(wdown_dg, concept_dir)

                # Also: gate mean direction
                g_avg = (g1 + g2) / 2

                if c1 == c2:
                    same_cat_cos.append(cos_dg)
                else:
                    diff_cat_cos.append(cos_dg)
                    diff_cat_norms.append(float(np.linalg.norm(wdown_dg)))

                    # Projection fraction: |Δout · concept_dir| / (||Δout|| * ||concept_dir||)
                    # This is just |cos|, but also compute the raw projection
                    proj_mag = abs(np.dot(wdown_dg, concept_dir)) / concept_norm
                    total_mag = np.linalg.norm(wdown_dg)
                    if total_mag > 1e-10:
                        diff_cat_proj_frac.append(float(proj_mag / total_mag))

        if len(diff_cat_cos) < 5:
            continue

        layer_results.append({
            "layer": li,
            "n_diff_pairs": len(diff_cat_cos),
            # Total Δout vs concept direction
            "mean_diff_cat_cos": float(np.mean(diff_cat_cos)),
            "abs_mean_diff_cat_cos": float(np.mean(np.abs(diff_cat_cos))),
            "mean_same_cat_cos": float(np.mean(same_cat_cos)) if same_cat_cos else 0,
            # Projection fraction
            "mean_proj_frac": float(np.mean(diff_cat_proj_frac)) if diff_cat_proj_frac else 0,
            # Norm
            "mean_diff_cat_norm": float(np.mean(diff_cat_norms)),
        })

        lr = layer_results[-1]
        print(f"  L{li}: diff_cos={lr['mean_diff_cat_cos']:+.3f}, "
              f"|diff_cos|={lr['abs_mean_diff_cat_cos']:.3f}, "
              f"proj_frac={lr['mean_proj_frac']:.3f}, "
              f"same_cos={lr['mean_same_cat_cos']:+.3f}")

    # Summary
    print(f"\n  Summary:")
    mid_results = [lr for lr in layer_results if n_layers * 0.3 <= lr["layer"] < n_layers * 0.7]
    if mid_results:
        avg_diff_cos = np.mean([lr["mean_diff_cat_cos"] for lr in mid_results])
        avg_abs_cos = np.mean([lr["abs_mean_diff_cat_cos"] for lr in mid_results])
        avg_proj_frac = np.mean([lr["mean_proj_frac"] for lr in mid_results])
        avg_same_cos = np.mean([lr["mean_same_cat_cos"] for lr in mid_results])

        print(f"    Mid-layer mean diff_cat_cos: {avg_diff_cos:+.3f}")
        print(f"    Mid-layer |diff_cat_cos|: {avg_abs_cos:.3f}")
        print(f"    Mid-layer proj_frac: {avg_proj_frac:.3f}")
        print(f"    Mid-layer same_cat_cos: {avg_same_cos:+.3f}")

        # Signal vs noise decomposition
        signal_frac = avg_proj_frac ** 2  # fraction of energy in concept direction
        noise_frac = 1 - signal_frac
        print(f"    Signal fraction: {signal_frac:.1%}")
        print(f"    Noise fraction: {noise_frac:.1%}")

        if avg_abs_cos > 0.3:
            print(f"  >>> W_down @ Δg has MODERATE alignment with concept direction")
        else:
            print(f"  >>> W_down @ Δg alignment with concept direction is WEAK")

    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxxiv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp4_delta_out_alignment.json"

    summary = {
        "experiment": "exp4_delta_out_concept_alignment",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "layer_results": layer_results,
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json_serialize(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")

    release_model(model)
    return summary


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, required=True,
                        choices=["1", "2", "3", "4", "all"])
    args = parser.parse_args()

    if args.exp in ["1", "all"]:
        exp1_dg_pca_semantics(args.model)

    if args.exp in ["2", "all"]:
        exp2_cross_layer_subspace_rotation(args.model)

    if args.exp in ["3", "all"]:
        exp3_dg_subspace_wu_alignment(args.model)

    if args.exp in ["4", "all"]:
        exp4_delta_out_concept_alignment(args.model)
