"""
Phase CCLXX: W_gate行向量的几何结构与概念映射
================================================
核心问题: W_gate的行向量在d_model空间中有什么几何结构?
         这种结构如何映射到概念类别?

背景:
  INV-50: 门控切换是FFN概念差异的主导来源(cos_gate=+0.47)
  INV-48: 门控中度稀疏(20-35% σ>0.5)
  INV-49: 同类概念门控更相似(W/X=1.2-1.25)
  INV-47: W_down不与W_U对齐(≈1.0x随机)

关键假说: W_gate行向量形成某种几何结构(如聚类/流形),
  不同的概念类别对应W_gate行的不同子集,
  这种映射关系是FFN"概念选择性"的数学基底.

验证:
  Exp1: W_gate行的几何结构 — PCA/聚类分析
        -> W_gate行是否有低维结构?
        -> 是否存在自然聚类?
        -> 聚类数与概念类别数是否相关?
  Exp2: 概念→门控神经元的因果映射
        -> 哪些门控神经元对哪些概念最敏感?
        -> 消融概念特异性神经元, 概念区分能力是否下降?
        -> 因果验证: 选择性消融 vs 随机消融
  Exp3: W_gate行与W_U行的关系
        -> W_gate行是否与W_U行有结构关系?
        -> 概念特异性W_gate行是否指向对应概念的W_U行?
        -> 门控→输出的"概念通道"是否存在?

用法:
  python phase_cclxx_gate_geometry.py --model qwen3 --exp 1
  python phase_cclxx_gate_geometry.py --model qwen3 --exp all
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
    get_layer_weights, LayerWeights,
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

TEMPLATES = [
    "The {} is",
]


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


# ============================================================
# Exp1: W_gate行的几何结构
# ============================================================
def exp1_gate_geometry(model_name):
    """Analyze geometric structure of W_gate row vectors.

    Key measures:
      - PCA variance explained: how many PCs needed for 90% variance?
      - K-means clustering: natural cluster count (via silhouette)
      - Row norm distribution
      - Cross-layer consistency of cluster structure
    """
    print(f"\n{'='*70}")
    print(f"Exp1: W_gate Row Vector Geometry")
    print(f"  Model: {model_name}")
    print(f"  Key test: What geometric structure do W_gate rows have?")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers_list = get_layers(model)

    layer_results = []

    sample_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))

    for li in sample_layers:
        lw = get_layer_weights(layers_list[li], d_model, mlp_type)
        if lw.W_gate is None:
            continue

        W_gate = lw.W_gate  # [intermediate, d_model]
        n_inter, d = W_gate.shape

        # 1. Row norm distribution
        row_norms = np.linalg.norm(W_gate, axis=1)
        norm_mean = float(np.mean(row_norms))
        norm_std = float(np.std(row_norms))
        norm_min = float(np.min(row_norms))
        norm_max = float(np.max(row_norms))
        norm_cv = norm_std / max(norm_mean, 1e-10)

        # 2. PCA on W_gate rows (subsample if too large for SVD)
        from scipy.sparse.linalg import svds
        W_centered = W_gate - W_gate.mean(axis=0, keepdims=True)

        # Use randomized SVD for large matrices
        max_k = min(100, min(n_inter, d) - 2)
        max_k = max(max_k, 10)

        try:
            # SVD of W_centered: rows are in d_model space
            # W_centered.T @ W_centered / n gives covariance in d_model
            # But we want variance of rows along PCs
            # SVD: W = U S Vt, where Vt columns are PCs in d_model space
            # Variance explained by each PC = s_i^2 / sum(s_i^2)
            U_svd, s_svd, Vt_svd = np.linalg.svd(W_centered, full_matrices=False)

            # Cumulative variance explained
            total_var = np.sum(s_svd ** 2)
            cum_var = np.cumsum(s_svd ** 2) / max(total_var, 1e-10)

            # Number of PCs for 50%, 80%, 90%, 95% variance
            n_pcs_50 = int(np.searchsorted(cum_var, 0.50) + 1)
            n_pcs_80 = int(np.searchsorted(cum_var, 0.80) + 1)
            n_pcs_90 = int(np.searchsorted(cum_var, 0.90) + 1)
            n_pcs_95 = int(np.searchsorted(cum_var, 0.95) + 1)

            # Top-5 variance fractions
            top5_var = float(cum_var[4]) if len(cum_var) >= 5 else float(cum_var[-1])
            top10_var = float(cum_var[9]) if len(cum_var) >= 10 else float(cum_var[-1])
            top20_var = float(cum_var[19]) if len(cum_var) >= 20 else float(cum_var[-1])

        except Exception as e:
            print(f"  L{li}: SVD failed: {e}")
            n_pcs_50 = n_pcs_80 = n_pcs_90 = n_pcs_95 = 0
            top5_var = top10_var = top20_var = 0.0
            cum_var = np.array([])

        # 3. K-means clustering (on top-50 PCs)
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        n_pca_use = min(50, len(cum_var))
        if n_pca_use < 2:
            continue

        V_pcs = Vt_svd[:n_pca_use].T  # [d_model, n_pca_use]
        W_pca = W_centered @ V_pcs  # [n_inter, n_pca_use]

        # Subsample for silhouette (expensive for large n_inter)
        rng = np.random.RandomState(42)
        if n_inter > 5000:
            sub_idx = rng.choice(n_inter, 5000, replace=False)
            W_sub = W_pca[sub_idx]
        else:
            W_sub = W_pca
            sub_idx = np.arange(n_inter)

        # Test k=2,3,4,5,6,8,10 for silhouette
        best_k = 2
        best_sil = -1
        sil_scores = {}
        for k in [2, 3, 4, 5, 6, 8, 10]:
            if k >= len(W_sub):
                continue
            try:
                km = KMeans(n_clusters=k, random_state=42, n_init=3, max_iter=100)
                labels = km.fit_predict(W_sub)
                sil = float(silhouette_score(W_sub, labels, sample_size=min(2000, len(W_sub))))
                sil_scores[k] = sil
                if sil > best_sil:
                    best_sil = sil
                    best_k = k
            except:
                pass

        # 4. Pairwise cosine between W_gate rows (subsample)
        n_sample_cos = min(500, n_inter)
        idx_sample = rng.choice(n_inter, n_sample_cos, replace=False)
        W_sample = W_gate[idx_sample]
        # Compute pairwise cosine
        W_norm = W_sample / np.linalg.norm(W_sample, axis=1, keepdims=True).clip(1e-10)
        cos_matrix = W_norm @ W_norm.T
        # Upper triangle
        triu_idx = np.triu_indices(n_sample_cos, k=1)
        pairwise_cos = cos_matrix[triu_idx]
        cos_mean = float(np.mean(pairwise_cos))
        cos_std = float(np.std(pairwise_cos))
        cos_frac_positive = float(np.mean(pairwise_cos > 0))

        layer_results.append({
            "layer": li,
            "n_intermediate": n_inter,
            "d_model": d,
            "norm_mean": norm_mean,
            "norm_std": norm_std,
            "norm_cv": norm_cv,
            "n_pcs_50": n_pcs_50,
            "n_pcs_80": n_pcs_80,
            "n_pcs_90": n_pcs_90,
            "n_pcs_95": n_pcs_95,
            "top5_var": top5_var,
            "top10_var": top10_var,
            "top20_var": top20_var,
            "best_k": best_k,
            "best_silhouette": best_sil,
            "silhouette_scores": sil_scores,
            "pairwise_cos_mean": cos_mean,
            "pairwise_cos_std": cos_std,
            "pairwise_cos_frac_positive": cos_frac_positive,
        })

        print(f"  L{li}: n_inter={n_inter}, PCA(n50={n_pcs_50}, n80={n_pcs_80}, n90={n_pcs_90}), "
              f"best_k={best_k}(sil={best_sil:.3f}), "
              f"top5_var={top5_var:.3f}, top10_var={top10_var:.3f}, "
              f"cos_mean={cos_mean:.4f}, cos_std={cos_std:.4f}")

    # Print summary
    print(f"\n  Summary across layers:")
    mid_results = [lr for lr in layer_results if n_layers * 0.3 <= lr["layer"] < n_layers * 0.7]
    if mid_results:
        avg_n90 = np.mean([lr["n_pcs_90"] for lr in mid_results])
        avg_top5 = np.mean([lr["top5_var"] for lr in mid_results])
        avg_top10 = np.mean([lr["top10_var"] for lr in mid_results])
        avg_best_k = np.mean([lr["best_k"] for lr in mid_results])
        avg_sil = np.mean([lr["best_silhouette"] for lr in mid_results])
        avg_cos_mean = np.mean([lr["pairwise_cos_mean"] for lr in mid_results])

        print(f"    Mid-layer avg PCA(n90): {avg_n90:.1f}")
        print(f"    Mid-layer avg top5_var: {avg_top5:.3f}")
        print(f"    Mid-layer avg top10_var: {avg_top10:.3f}")
        print(f"    Mid-layer avg best_k: {avg_best_k:.1f}")
        print(f"    Mid-layer avg best_sil: {avg_sil:.3f}")
        print(f"    Mid-layer avg cos_mean: {avg_cos_mean:.4f}")

        if avg_top5 > 0.3:
            print(f"  >>> W_gate rows have LOW-DIMENSIONAL structure (top5 > 30%)")
        elif avg_top5 > 0.15:
            print(f"  >>> W_gate rows have MODERATE low-dim structure (top5 = 15-30%)")
        else:
            print(f"  >>> W_gate rows are HIGH-DIMENSIONAL (top5 < 15%)")

        if avg_sil > 0.1:
            print(f"  >>> W_gate rows have natural CLUSTER structure (sil > 0.1)")
        else:
            print(f"  >>> W_gate rows do NOT have strong cluster structure (sil < 0.1)")

    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxx"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp1_gate_geometry.json"

    summary = {
        "experiment": "exp1_gate_geometry",
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
# Exp2: 概念→门控神经元的因果映射
# ============================================================
def exp2_concept_gate_mapping(model_name):
    """Map concept categories to specific gate neurons and test causality.

    Key measures:
      - For each category, find "discriminative neurons": 
        neurons where σ(cat_A) >> σ(cat_B) for all other categories B
      - Test causality: ablate top-k discriminative neurons, 
        measure effect on concept discrimination
      - Compare selective ablation vs random ablation
    """
    print(f"\n{'='*70}")
    print(f"Exp2: Concept -> Gate Neuron Causal Mapping")
    print(f"  Model: {model_name}")
    print(f"  Key test: Which gate neurons are concept-specific? Causal?")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers_list = get_layers(model)

    template = "The {} is"

    # Collect gate activations for all concepts
    all_gate_acts = {}  # (cat, word) -> {layer: sigma_z}
    categories = list(CONCEPTS.keys())

    rng = np.random.RandomState(42)
    words_per_cat = {}
    for cat, wlist in CONCEPTS.items():
        words_per_cat[cat] = rng.choice(wlist, min(8, len(wlist)), replace=False).tolist()

    for cat, words in words_per_cat.items():
        for word in words:
            text = template.format(word)
            input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
            last_pos = input_ids.shape[1] - 1

            ln_out = {}
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

            with torch.no_grad():
                _ = model(input_ids)

            for h in hooks:
                h.remove()

            word_gate = {}
            for li in range(n_layers):
                key = f"L{li}"
                if key not in ln_out:
                    continue
                lw = get_layer_weights(layers_list[li], d_model, mlp_type)
                if lw.W_gate is None:
                    continue
                z = lw.W_gate @ ln_out[key]
                z_clipped = np.clip(z, -500, 500)
                sigma_z = 1.0 / (1.0 + np.exp(-z_clipped))
                word_gate[li] = sigma_z

            all_gate_acts[(cat, word)] = word_gate

    print(f"  Collected gate activations for {len(all_gate_acts)} words")

    # For each layer, compute category-discriminative neurons
    target_layers = list(range(0, n_layers, max(1, n_layers // 6)))
    if n_layers - 1 not in target_layers:
        target_layers.append(n_layers - 1)
    target_layers = sorted(set(target_layers))

    layer_results = []

    for li in target_layers:
        # Collect gate activations per category
        cat_acts = defaultdict(list)  # cat -> [sigma_z arrays]
        for (cat, word), gate in all_gate_acts.items():
            if li in gate:
                cat_acts[cat].append(gate[li])

        if len(cat_acts) < 2:
            continue

        # Compute mean activation per category
        cat_means = {}
        for cat in categories:
            if cat in cat_acts and len(cat_acts[cat]) > 0:
                cat_means[cat] = np.mean(cat_acts[cat], axis=0)  # [n_inter]

        n_inter = list(cat_means.values())[0].shape[0]

        # For each category, find discriminative neurons:
        # neurons where this category has highest mean σ
        cat_disc_neurons = {}  # cat -> [(neuron_idx, selectivity)]
        for cat in categories:
            if cat not in cat_means:
                continue
            mean_this = cat_means[cat]

            # Selectivity = mean_this - max(mean_others)
            other_means = [cat_means[c] for c in categories if c != cat and c in cat_means]
            if not other_means:
                continue
            max_other = np.maximum.reduce(other_means)
            selectivity = mean_this - max_other  # [n_inter]

            # Top discriminative neurons
            top_indices = np.argsort(selectivity)[::-1]
            cat_disc_neurons[cat] = [
                (int(idx), float(selectivity[idx]))
                for idx in top_indices[:200]
                if selectivity[idx] > 0.01  # only positive selectivity
            ]

        # Count discriminative neurons per category
        disc_counts = {cat: len(cat_disc_neurons.get(cat, [])) for cat in categories}

        # Overlap: do discriminative neurons overlap across categories?
        disc_sets = {cat: set(n[0] for n in cat_disc_neurons.get(cat, [])) for cat in categories}
        max_overlap = 0
        for i, c1 in enumerate(categories):
            for c2 in categories[i+1:]:
                if disc_sets[c1] and disc_sets[c2]:
                    overlap = len(disc_sets[c1] & disc_sets[c2])
                    max_overlap = max(max_overlap, overlap)

        # Causal test: ablate top-k discriminative neurons
        # Measure: concept discrimination before and after ablation
        # Discrimination = σ(cat_target) - mean(σ(cat_other)) for top discriminative neurons
        k_ablate = 50  # ablate top-50 most discriminative neurons

        abl_results = {}
        for cat in categories:
            if cat not in cat_disc_neurons or len(cat_disc_neurons[cat]) < k_ablate:
                continue

            ablate_neurons = [n[0] for n in cat_disc_neurons[cat][:k_ablate]]

            # Before ablation: discrimination on all words
            before_disc_scores = []
            for (c, w), gate in all_gate_acts.items():
                if li in gate:
                    sigma = gate[li]
                    # Mean activation of target category neurons
                    target_act = np.mean(sigma[ablate_neurons])
                    before_disc_scores.append((c, w, float(target_act)))

            # Random ablation baseline: same number of random neurons
            rng_abl = np.random.RandomState(42)
            random_neurons = rng_abl.choice(n_inter, k_ablate, replace=False).tolist()

            # Compute discrimination for selective vs random
            selective_target = []  # target cat words, selective ablation
            selective_other = []   # other cat words, selective ablation
            random_target = []     # target cat words, random ablation
            random_other = []      # other cat words, random ablation

            for c, w, act in before_disc_scores:
                if c == cat:
                    selective_target.append(act)
                    random_target.append(act)  # same before ablation
                else:
                    selective_other.append(act)
                    random_other.append(act)

            # After ablation (simulated): set σ(ablated_neurons) = 0
            # This means those neurons contribute nothing to the FFN output
            # We measure: does the discrimination (target - other) change?
            # Actually, we measure the discrimination based on ablated neuron activations
            # Before: target neurons have high σ for target cat, low for others
            # After ablation: those neurons are forced to 0

            # Simpler causal measure:
            # For each word, compute "discriminative gate mass" = sum of σ on disc neurons
            # Discrimination = mean(target_cat_disc_mass) - mean(other_cat_disc_mass)
            # Random comparison: same with random neurons

            # Compute discriminative mass for selective neurons
            sel_mass_target = []
            sel_mass_other = []
            for (c, w), gate in all_gate_acts.items():
                if li in gate:
                    sigma = gate[li]
                    mass = np.mean(sigma[ablate_neurons])
                    if c == cat:
                        sel_mass_target.append(mass)
                    else:
                        sel_mass_other.append(mass)

            # Compute discriminative mass for random neurons
            rnd_mass_target = []
            rnd_mass_other = []
            for (c, w), gate in all_gate_acts.items():
                if li in gate:
                    sigma = gate[li]
                    mass = np.mean(sigma[random_neurons])
                    if c == cat:
                        rnd_mass_target.append(mass)
                    else:
                        rnd_mass_other.append(mass)

            # Selectivity ratio: (target - other) for selective / (target - other) for random
            sel_disc = np.mean(sel_mass_target) - np.mean(sel_mass_other)
            rnd_disc = np.mean(rnd_mass_target) - np.mean(rnd_mass_other)
            sel_ratio = sel_disc / max(abs(rnd_disc), 1e-6)

            abl_results[cat] = {
                "n_disc_neurons": len(cat_disc_neurons[cat]),
                "selective_disc": float(sel_disc),
                "random_disc": float(rnd_disc),
                "selective_random_ratio": float(sel_ratio),
                "sel_mass_target": float(np.mean(sel_mass_target)),
                "sel_mass_other": float(np.mean(sel_mass_other)),
                "rnd_mass_target": float(np.mean(rnd_mass_target)),
                "rnd_mass_other": float(np.mean(rnd_mass_other)),
            }

        # Average across categories
        avg_sel_disc = np.mean([r["selective_disc"] for r in abl_results.values()])
        avg_rnd_disc = np.mean([r["random_disc"] for r in abl_results.values()])
        avg_sel_ratio = np.mean([r["selective_random_ratio"] for r in abl_results.values()])

        layer_results.append({
            "layer": li,
            "n_intermediate": n_inter,
            "disc_counts": disc_counts,
            "max_overlap": max_overlap,
            "abl_results": abl_results,
            "avg_selective_disc": float(avg_sel_disc),
            "avg_random_disc": float(avg_rnd_disc),
            "avg_selective_random_ratio": float(avg_sel_ratio),
        })

        print(f"  L{li}: disc_counts={disc_counts}, "
              f"sel_disc={avg_sel_disc:.4f}, rnd_disc={avg_rnd_disc:.4f}, "
              f"sel/rnd={avg_sel_ratio:.2f}")

    # Summary
    print(f"\n  Summary:")
    mid_results = [lr for lr in layer_results if n_layers * 0.3 <= lr["layer"] < n_layers * 0.7]
    if mid_results:
        avg_sel = np.mean([lr["avg_selective_disc"] for lr in mid_results])
        avg_rnd = np.mean([lr["avg_random_disc"] for lr in mid_results])
        avg_ratio = np.mean([lr["avg_selective_random_ratio"] for lr in mid_results])

        print(f"    Mid-layer avg selective discrimination: {avg_sel:.4f}")
        print(f"    Mid-layer avg random discrimination: {avg_rnd:.4f}")
        print(f"    Mid-layer avg sel/rnd ratio: {avg_ratio:.2f}")

        if avg_ratio > 3.0:
            print(f"  >>> Strong concept-specific gate neurons (sel/rnd > 3x)")
        elif avg_ratio > 1.5:
            print(f"  >>> Moderate concept-specific gate neurons (sel/rnd = 1.5-3x)")
        else:
            print(f"  >>> Weak concept-specificity (sel/rnd < 1.5x)")

    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxx"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp2_concept_gate_mapping.json"

    summary = {
        "experiment": "exp2_concept_gate_mapping",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "k_ablate": k_ablate,
        "layer_results": layer_results,
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json_serialize(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")

    release_model(model)
    return summary


# ============================================================
# Exp3: W_gate行与W_U行的关系
# ============================================================
def exp3_gate_output_relation(model_name):
    """Analyze relationship between W_gate rows and W_U rows.

    Key measures:
      - Project W_gate rows onto W_U row space: how much variance is captured?
      - For concept-specific W_gate rows, do they align with W_U rows 
        of the corresponding concept words?
      - Is there a "concept channel" from gate -> output?
    """
    print(f"\n{'='*70}")
    print(f"Exp3: W_gate Rows vs W_U Rows")
    print(f"  Model: {model_name}")
    print(f"  Key test: Do W_gate rows relate to W_U rows? Concept channels?")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers_list = get_layers(model)

    W_U = get_W_U(model)  # [vocab_size, d_model]

    # SVD of W_U for row space projection
    from scipy.sparse.linalg import svds
    k_svd = min(200, min(W_U.shape) - 2)
    k_svd = max(k_svd, 10)
    U_wu, s_wu, Vt_wu = svds(W_U.T.astype(np.float32), k=k_svd)
    U_wu = np.asarray(U_wu, dtype=np.float64)  # [d_model, k_svd] — W_U row space basis

    # W_U row space projection of a vector v: U_wu @ (U_wu.T @ v)
    # Fraction of v's norm in W_U row space: ||U_wu @ U_wu.T @ v||^2 / ||v||^2

    # Get token IDs for concept words
    concept_token_ids = {}
    for cat, words in CONCEPTS.items():
        for word in words:
            tok_ids = tokenizer.encode(word, add_special_tokens=False)
            if tok_ids:
                concept_token_ids[(cat, word)] = tok_ids[0]

    target_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    if n_layers - 1 not in target_layers:
        target_layers.append(n_layers - 1)
    target_layers = sorted(set(target_layers))

    layer_results = []

    for li in target_layers:
        lw = get_layer_weights(layers_list[li], d_model, mlp_type)
        if lw.W_gate is None:
            continue

        W_gate = lw.W_gate  # [n_inter, d_model]
        n_inter = W_gate.shape[0]

        # 1. Project W_gate rows onto W_U row space
        # For each W_gate row w_i, compute fraction in W_U row space
        # proj_frac = ||U_wu @ U_wu.T @ w_i||^2 / ||w_i||^2

        # Batch computation for efficiency
        # U_wu.T @ W_gate.T = [k, n_inter]
        coeffs = U_wu.T @ W_gate.T  # [k_svd, n_inter]
        # Reconstructed in W_U space: U_wu @ coeffs = [d_model, n_inter]
        W_gate_proj = U_wu @ coeffs  # [d_model, n_inter]

        # Norms
        orig_norms_sq = np.sum(W_gate ** 2, axis=1)  # [n_inter]
        proj_norms_sq = np.sum(W_gate_proj ** 2, axis=0)  # [n_inter]

        # Fraction in W_U row space
        proj_fracs = proj_norms_sq / np.maximum(orig_norms_sq, 1e-10)
        proj_fracs = np.clip(proj_fracs, 0, 1)

        mean_proj_frac = float(np.mean(proj_fracs))
        median_proj_frac = float(np.median(proj_fracs))

        # Random baseline: project random vectors of same norm distribution
        rng = np.random.RandomState(42)
        n_random = min(2000, n_inter)
        random_vecs = rng.randn(n_random, d_model)
        # Match norm distribution
        orig_norms = np.sqrt(orig_norms_sq)
        random_norms = np.linalg.norm(random_vecs, axis=1)
        scale = orig_norms[rng.choice(n_inter, n_random, replace=True)] / np.maximum(random_norms, 1e-10)
        random_vecs = random_vecs * scale[:, None]

        rand_coeffs = U_wu.T @ random_vecs.T
        rand_proj = U_wu @ rand_coeffs
        rand_proj_norms_sq = np.sum(rand_proj ** 2, axis=0)
        rand_orig_norms_sq = np.sum(random_vecs ** 2, axis=1)
        rand_proj_fracs = rand_proj_norms_sq / np.maximum(rand_orig_norms_sq, 1e-10)
        mean_random_proj_frac = float(np.mean(rand_proj_fracs))

        proj_ratio = mean_proj_frac / max(mean_random_proj_frac, 1e-6)

        # 2. For concept-specific W_gate rows, do they align with W_U rows 
        #    of corresponding concept words?
        # This requires knowing which W_gate rows are concept-specific
        # We'll use a simpler approach:
        # For each concept word w, compute cos(W_gate[i], W_U[w]) for all i
        # Then check if the top-aligned W_gate row tends to be discriminative for that category

        concept_alignment = {}
        for cat in CONCEPTS:
            cat_words = list(CONCEPTS[cat])[:4]  # top 4 words per category
            cat_align_scores = []

            for word in cat_words:
                key = (cat, word)
                if key not in concept_token_ids:
                    continue
                tok_id = concept_token_ids[key]
                w_u_row = W_U[tok_id]  # [d_model]
                w_u_norm = np.linalg.norm(w_u_row)
                if w_u_norm < 1e-10:
                    continue
                w_u_normalized = w_u_row / w_u_norm

                # Cosine of each W_gate row with this W_U row
                W_gate_norms = np.sqrt(orig_norms_sq)
                valid = W_gate_norms > 1e-10
                cos_vals = np.zeros(n_inter)
                cos_vals[valid] = (W_gate[valid] @ w_u_normalized) / W_gate_norms[valid]

                # Top-aligned gate neuron
                top_idx = int(np.argmax(cos_vals))
                top_cos = float(cos_vals[top_idx])

                cat_align_scores.append({
                    "word": word,
                    "top_gate_neuron": top_idx,
                    "top_cosine": top_cos,
                    "mean_cosine": float(np.mean(np.abs(cos_vals))),
                    "frac_positive": float(np.mean(cos_vals > 0)),
                })

            if cat_align_scores:
                concept_alignment[cat] = cat_align_scores

        # 3. Cross-category alignment overlap
        # Do top-aligned neurons for different categories overlap?
        cat_top_neurons = {}
        for cat, scores in concept_alignment.items():
            top_set = set()
            for s in scores:
                top_set.add(s["top_gate_neuron"])
            cat_top_neurons[cat] = top_set

        max_overlap = 0
        cat_list = list(cat_top_neurons.keys())
        for i in range(len(cat_list)):
            for j in range(i+1, len(cat_list)):
                if cat_top_neurons[cat_list[i]] and cat_top_neurons[cat_list[j]]:
                    overlap = len(cat_top_neurons[cat_list[i]] & cat_top_neurons[cat_list[j]])
                    max_overlap = max(max_overlap, overlap)

        # Average top_cosine across all concepts
        all_top_cos = []
        for cat, scores in concept_alignment.items():
            for s in scores:
                all_top_cos.append(s["top_cosine"])
        avg_top_cos = float(np.mean(all_top_cos)) if all_top_cos else 0.0

        layer_results.append({
            "layer": li,
            "n_intermediate": n_inter,
            "mean_proj_frac": mean_proj_frac,
            "median_proj_frac": median_proj_frac,
            "mean_random_proj_frac": mean_random_proj_frac,
            "proj_ratio": proj_ratio,
            "concept_alignment": concept_alignment,
            "avg_top_cosine": avg_top_cos,
            "max_cross_cat_overlap": max_overlap,
        })

        print(f"  L{li}: proj_frac={mean_proj_frac:.4f}, random={mean_random_proj_frac:.4f}, "
              f"ratio={proj_ratio:.2f}, avg_top_cos={avg_top_cos:.4f}")

    # Summary
    print(f"\n  Summary:")
    mid_results = [lr for lr in layer_results if n_layers * 0.3 <= lr["layer"] < n_layers * 0.7]
    if mid_results:
        avg_proj = np.mean([lr["mean_proj_frac"] for lr in mid_results])
        avg_rand = np.mean([lr["mean_random_proj_frac"] for lr in mid_results])
        avg_ratio = np.mean([lr["proj_ratio"] for lr in mid_results])
        avg_topcos = np.mean([lr["avg_top_cosine"] for lr in mid_results])

        print(f"    Mid-layer avg W_gate proj in W_U: {avg_proj:.4f}")
        print(f"    Mid-layer avg random proj: {avg_rand:.4f}")
        print(f"    Mid-layer avg ratio: {avg_ratio:.2f}x")
        print(f"    Mid-layer avg top_cosine: {avg_topcos:.4f}")

        if avg_ratio > 1.5:
            print(f"  >>> W_gate rows are ALIGNED with W_U row space (>{1.5}x random)")
        else:
            print(f"  >>> W_gate rows are NOT aligned with W_U row space (≈random)")

        if avg_topcos > 0.1:
            print(f"  >>> Some W_gate rows align with concept W_U rows (top_cos > 0.1)")
        else:
            print(f"  >>> No strong concept-specific gate-output alignment")

    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxx"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp3_gate_output_relation.json"

    summary = {
        "experiment": "exp3_gate_output_relation",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "k_svd": k_svd,
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
                        choices=["1", "2", "3", "all"])
    args = parser.parse_args()

    if args.exp in ["1", "all"]:
        exp1_gate_geometry(args.model)

    if args.exp in ["2", "all"]:
        exp2_concept_gate_mapping(args.model)

    if args.exp in ["3", "all"]:
        exp3_gate_output_relation(args.model)
