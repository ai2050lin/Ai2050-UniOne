"""
CCLXXXIX: 类别内部几何结构分析
基于CCLXXXVIII发现: cross-superclass最1D可分(0.47), same最不可分(0.25)
目标: 分析每个类别集群的内部结构, 积累类别级几何数据

Exp1: 类别紧凑度 — 每个类别的intra-pairwise distance分布 (均值/std/min/max)
Exp2: 类别各向异性 — 每个类别在PC空间中的elongation方向和比率
Exp3: 类别偏心率 — 每个类别质心到全局均值的距离在各PC上的分解
Exp4: 类别子聚类 — 每个类别内部是否存在2-3个子集群
Exp5: 跨层紧凑度演化 — 类别紧凑度在各层如何变化? 与结晶层的关系?
"""
import sys, os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxxix_log.txt"

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        import time
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()

log("=== CCLXXXIX Script started ===")

import json, time, gc, traceback
from pathlib import Path
import numpy as np
from collections import defaultdict

log("Importing torch...")
import torch

log("Importing model_utils...")
sys.path.insert(0, r"d:\Ai2050\TransformerLens-Project")
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, get_W_U,
    get_layer_weights, MODEL_CONFIGS,
)

# ===== 13类别 × 20词 =====
CATEGORIES_13 = {
    "animal": [
        "dog", "cat", "horse", "cow", "pig", "sheep", "goat", "donkey",
        "lion", "tiger", "bear", "wolf", "fox", "deer", "rabbit",
        "elephant", "giraffe", "zebra", "monkey", "camel",
    ],
    "bird": [
        "eagle", "hawk", "owl", "crow", "swan", "goose", "duck",
        "penguin", "parrot", "robin", "sparrow", "pigeon", "seagull",
        "falcon", "vulture", "crane", "stork", "heron", "peacock", "flamingo",
    ],
    "fish": [
        "shark", "whale", "dolphin", "salmon", "trout", "tuna",
        "cod", "bass", "carp", "catfish", "perch", "pike", "eel",
        "herring", "sardine", "anchovy", "flounder", "sole", "mackerel", "swordfish",
    ],
    "insect": [
        "ant", "bee", "spider", "butterfly", "mosquito", "fly", "wasp",
        "beetle", "cockroach", "grasshopper", "cricket", "dragonfly",
        "ladybug", "moth", "flea", "tick", "mantis", "caterpillar", "worm", "snail",
    ],
    "plant": [
        "tree", "flower", "grass", "bush", "shrub", "vine", "fern",
        "moss", "algae", "weed", "oak", "pine", "maple", "birch",
        "willow", "cactus", "bamboo", "palm", "rose", "lily",
    ],
    "fruit": [
        "apple", "orange", "banana", "grape", "pear", "peach",
        "cherry", "plum", "mango", "lemon", "lime", "melon",
        "berry", "strawberry", "blueberry", "raspberry", "fig", "date",
        "coconut", "pineapple",
    ],
    "vegetable": [
        "carrot", "potato", "tomato", "onion", "garlic", "cabbage",
        "lettuce", "spinach", "celery", "pea", "bean", "corn",
        "mushroom", "pepper", "cucumber", "pumpkin", "squash",
        "radish", "turnip", "broccoli",
    ],
    "body_part": [
        "hand", "foot", "head", "heart", "brain", "eye", "ear",
        "nose", "mouth", "tooth", "neck", "shoulder", "arm",
        "finger", "knee", "chest", "back", "hip", "ankle", "wrist",
    ],
    "tool": [
        "hammer", "knife", "scissors", "saw", "drill", "wrench",
        "screwdriver", "plier", "axe", "chisel", "ruler", "file",
        "clamp", "level", "shovel", "rake", "hoe", "trowel",
        "spade", "mallet",
    ],
    "vehicle": [
        "car", "bus", "truck", "train", "bicycle", "motorcycle",
        "airplane", "helicopter", "boat", "ship", "submarine",
        "rocket", "tractor", "van", "taxi", "ambulance",
        "sled", "canoe", "wagon", "cart",
    ],
    "clothing": [
        "shirt", "dress", "hat", "coat", "shoe", "belt", "scarf",
        "glove", "jacket", "sweater", "vest", "skirt", "pants",
        "jeans", "sock", "boot", "sandal", "tie", "uniform", "cape",
    ],
    "weapon": [
        "sword", "spear", "bow", "arrow", "shield", "axe_w",
        "dagger", "mace", "pike_w", "lance", "crossbow", "catapult",
        "pistol", "rifle", "cannon", "grenade", "dynamite",
        "knife_w", "club", "whip",
    ],
    "furniture": [
        "chair", "table", "desk", "bed", "sofa", "couch", "shelf",
        "cabinet", "drawer", "wardrobe", "dresser", "bench",
        "stool", "armchair", "bookcase", "mirror", "lamp",
        "rug", "curtain", "pillow",
    ],
}

ANIMATE_CATS = {"animal", "bird", "fish", "insect"}
PLANT_CATS = {"plant", "fruit", "vegetable"}
BODY_CATS = {"body_part"}
ARTIFACT_CATS = {"tool", "vehicle", "clothing", "weapon", "furniture"}

def get_superclass(cat):
    if cat in ANIMATE_CATS: return "biological_animate"
    if cat in PLANT_CATS: return "biological_plant"
    if cat in BODY_CATS: return "body"
    if cat in ARTIFACT_CATS: return "artifact"
    return "unknown"


def run_model(model_name):
    log(f"=== Starting {model_name} ===")
    try:
        log("Loading model...")
        model, tokenizer, device = load_model(model_name)
        model_info = get_model_info(model, model_name)
        n_layers = model_info.n_layers
        d_model = model_info.d_model
        mlp_type = model_info.mlp_type
        layers_list = get_layers(model)
        log(f"Model loaded: {n_layers}L, d={d_model}, mlp={mlp_type}")

        # ===== 收集词 =====
        all_words, all_cats = [], []
        for cat, words in CATEGORIES_13.items():
            all_words.extend(words[:20])
            all_cats.extend([cat] * 20)

        n_total = len(all_words)
        log(f"Total words: {n_total}")

        # ===== 前向推理: 收集各层残差流 =====
        template = "The {} is"
        word_layer_acts = {}  # word -> {li: h_vector[d_model]}

        t0 = time.time()
        for wi, word in enumerate(all_words):
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
                            a = args[0] if not isinstance(args, tuple) else args[0]
                            ln_out[key] = a[0, last_pos].detach().float().cpu().numpy()
                        return hook
                    hooks.append(layer.mlp.register_forward_pre_hook(make_ffn_pre(f"L{li}")))

            with torch.no_grad():
                _ = model(input_ids)
            for h in hooks:
                h.remove()

            word_layer_acts[word] = {}
            for li in range(n_layers):
                key = f"L{li}"
                if key in ln_out:
                    word_layer_acts[word][li] = ln_out[key]

            if (wi + 1) % 50 == 0 or wi == 0:
                log(f"  Word {wi+1}/{n_total} ({time.time()-t0:.0f}s)")

        log(f"Data collection done ({time.time()-t0:.0f}s)")

        # ===== 采样层 =====
        if n_layers <= 20:
            sample_layers = list(range(n_layers))
        elif n_layers <= 40:
            sample_layers = list(range(0, n_layers, 2)) + [n_layers - 1]
        else:
            sample_layers = list(range(0, n_layers, 3)) + [n_layers - 1]
        sample_layers = sorted(set(sample_layers))
        log(f"Sample layers: {sample_layers}")

        main_cats = sorted(CATEGORIES_13.keys())

        # ===== 跨层分析 =====
        from scipy.sparse.linalg import svds

        # 存储结果
        results = {
            "model": model_name,
            "n_layers": n_layers,
            "d_model": d_model,
            "exp1_compactness": {},   # {li: {cat: {mean, std, min, max, median}}}
            "exp2_anisotropy": {},    # {li: {cat: {top1_ratio, top3_ratio, elongation_pc}}}
            "exp3_eccentricity": {},  # {li: {cat: {total_dist, pc_decomposition}}}
            "exp4_subcluster": {},    # {li: {cat: {n_sub, sub_sizes, sub_gap}}}
            "exp5_evolution": {},     # compactness evolution across layers
        }

        for li in sample_layers:
            log(f"\n--- Layer {li} ---")

            # 收集该层所有词的激活
            cat_vectors = {}  # cat -> list of vectors
            all_vecs = []
            all_vec_cats = []

            for cat in main_cats:
                words = CATEGORIES_13[cat][:20]
                vecs = []
                for w in words:
                    if w in word_layer_acts and li in word_layer_acts[w]:
                        vecs.append(word_layer_acts[w][li])
                if len(vecs) >= 5:
                    cat_vectors[cat] = np.array(vecs)
                    all_vecs.extend(vecs)
                    all_vec_cats.extend([cat] * len(vecs))

            if len(all_vecs) < 10:
                log(f"  L{li}: too few vectors ({len(all_vecs)}), skip")
                continue

            all_vecs = np.array(all_vecs)

            # SVD for this layer
            d_proj = 30  # use more PCs for intra-category analysis
            k = min(d_proj, min(all_vecs.shape) - 1)
            if k < 2:
                continue
            _U_full, S, Vt = np.linalg.svd(all_vecs - all_vecs.mean(axis=0), full_matrices=False)
            Vt = Vt[:k]  # [k, d_model] — right singular vectors (PC directions)
            # Projection basis: Vt.T shape = [d_model, k]

            # ===== Exp1: 类别紧凑度 =====
            log(f"\n=== Exp1: Category Compactness at L{li} ===")
            exp1_data = {}

            for cat in main_cats:
                if cat not in cat_vectors:
                    continue
                vecs = cat_vectors[cat]
                n = len(vecs)
                if n < 3:
                    continue

                # Project to SVD space
                proj = vecs @ Vt.T  # [n, k]

                # Intra-pairwise distances (original space)
                from scipy.spatial.distance import pdist
                dists = pdist(vecs, metric='euclidean')

                # Intra-pairwise distances in projected space
                proj_dists = pdist(proj, metric='euclidean')

                # Mean distance to centroid
                centroid = vecs.mean(axis=0)
                dist_to_centroid = np.linalg.norm(vecs - centroid, axis=1)

                exp1_data[cat] = {
                    "n_words": n,
                    "intra_dist_mean": float(np.mean(dists)),
                    "intra_dist_std": float(np.std(dists)),
                    "intra_dist_min": float(np.min(dists)),
                    "intra_dist_max": float(np.max(dists)),
                    "intra_dist_median": float(np.median(dists)),
                    "intra_proj_dist_mean": float(np.mean(proj_dists)),
                    "dist_to_centroid_mean": float(np.mean(dist_to_centroid)),
                    "dist_to_centroid_std": float(np.std(dist_to_centroid)),
                }

            # Print compactness ranking
            if exp1_data:
                sorted_cats = sorted(exp1_data.items(), key=lambda x: x[1]["intra_dist_mean"])
                log(f"  Compactness ranking (intra_dist_mean, ascending=most compact):")
                for rank, (cat, d) in enumerate(sorted_cats, 1):
                    log(f"    {rank}. {cat}: mean={d['intra_dist_mean']:.3f}, "
                        f"std={d['intra_dist_std']:.3f}, median={d['intra_dist_median']:.3f}, "
                        f"to_centroid={d['dist_to_centroid_mean']:.3f}")

                # Most compact vs most spread
                most_compact = sorted_cats[0]
                most_spread = sorted_cats[-1]
                log(f"  Most compact: {most_compact[0]} ({most_compact[1]['intra_dist_mean']:.3f})")
                log(f"  Most spread: {most_spread[0]} ({most_spread[1]['intra_dist_mean']:.3f})")
                log(f"  Spread ratio: {most_spread[1]['intra_dist_mean']/max(most_compact[1]['intra_dist_mean'],1e-10):.2f}")

            results["exp1_compactness"][li] = exp1_data

            # ===== Exp2: 类别各向异性 =====
            log(f"\n=== Exp2: Category Anisotropy at L{li} ===")
            exp2_data = {}

            for cat in main_cats:
                if cat not in cat_vectors:
                    continue
                vecs = cat_vectors[cat]
                n = len(vecs)
                if n < 5:
                    continue

                # SVD of category-internal spread
                cat_centered = vecs - vecs.mean(axis=0)
                try:
                    cat_U, cat_S, _ = np.linalg.svd(cat_centered, full_matrices=False)
                except:
                    continue

                # Anisotropy: how much of variance is in top PCs
                total_var = np.sum(cat_S[:min(len(cat_S), n-1)]**2)
                if total_var < 1e-10:
                    continue

                n_components = min(5, len(cat_S))
                top1_ratio = cat_S[0]**2 / total_var
                top3_ratio = np.sum(cat_S[:min(3, n_components)]**2) / total_var
                top5_ratio = np.sum(cat_S[:min(5, n_components)]**2) / total_var

                # Elongation ratio: S[0]/S[1] — how elongated along first PC
                if len(cat_S) > 1 and cat_S[1] > 1e-10:
                    elongation = cat_S[0] / cat_S[1]
                else:
                    elongation = float('inf')

                exp2_data[cat] = {
                    "top1_ratio": float(top1_ratio),
                    "top3_ratio": float(top3_ratio),
                    "top5_ratio": float(top5_ratio),
                    "elongation": float(elongation) if elongation != float('inf') else 999.0,
                    "singular_values": [float(s) for s in cat_S[:5]],
                }

            # Print anisotropy ranking
            if exp2_data:
                sorted_aniso = sorted(exp2_data.items(), key=lambda x: x[1]["top1_ratio"], reverse=True)
                log(f"  Anisotropy ranking (top1_ratio, descending=most anisotropic):")
                for rank, (cat, d) in enumerate(sorted_aniso, 1):
                    log(f"    {rank}. {cat}: top1={d['top1_ratio']:.3f}, top3={d['top3_ratio']:.3f}, "
                        f"elongation={d['elongation']:.2f}, SV={d['singular_values'][:3]}")

            results["exp2_anisotropy"][li] = exp2_data

            # ===== Exp3: 类别偏心率 =====
            log(f"\n=== Exp3: Category Eccentricity at L{li} ===")
            exp3_data = {}

            global_mean = all_vecs.mean(axis=0)

            for cat in main_cats:
                if cat not in cat_vectors:
                    continue
                vecs = cat_vectors[cat]
                centroid = vecs.mean(axis=0)
                diff = centroid - global_mean  # [d_model]

                total_dist = np.linalg.norm(diff)
                if total_dist < 1e-10:
                    continue

                # Project diff onto U (SVD basis)
                diff_proj = diff @ Vt.T  # [k]
                diff_proj_sq = diff_proj**2

                # PC decomposition of eccentricity
                total_sq = np.sum(diff_proj_sq)
                if total_sq < 1e-10:
                    continue

                # Top contributing PCs
                pc_contributions = diff_proj_sq / total_sq
                top_pc_idx = np.argsort(pc_contributions)[::-1][:5]

                pc_decomp = {}
                cum = 0.0
                for rank, pci in enumerate(top_pc_idx):
                    contrib = float(pc_contributions[pci])
                    cum += contrib
                    pc_decomp[f"PC{pci}"] = {
                        "contrib": contrib,
                        "cum_contrib": cum,
                    }

                exp3_data[cat] = {
                    "total_dist": float(total_dist),
                    "superclass": get_superclass(cat),
                    "top_pcs": pc_decomp,
                    "top3_cum": sum(float(pc_contributions[i]) for i in top_pc_idx[:3]),
                }

            # Print eccentricity ranking
            if exp3_data:
                sorted_ecc = sorted(exp3_data.items(), key=lambda x: x[1]["total_dist"], reverse=True)
                log(f"  Eccentricity ranking (total_dist, descending=most eccentric):")
                for rank, (cat, d) in enumerate(sorted_ecc, 1):
                    top_pcs_str = ", ".join([f"{k}({v['contrib']:.2f})" for k, v in list(d["top_pcs"].items())[:3]])
                    log(f"    {rank}. {cat}({d['superclass']}): dist={d['total_dist']:.3f}, "
                        f"top3_cum={d['top3_cum']:.3f}, top_PCs=[{top_pcs_str}]")

            results["exp3_eccentricity"][li] = exp3_data

            # ===== Exp4: 类别子聚类 =====
            log(f"\n=== Exp4: Category Sub-clustering at L{li} ===")
            exp4_data = {}

            for cat in main_cats:
                if cat not in cat_vectors:
                    continue
                vecs = cat_vectors[cat]
                n = len(vecs)
                if n < 6:
                    continue

                # Use projected vectors for clustering
                proj = vecs @ Vt.T  # [n, k]

                # Try k=2 and k=3 sub-clusters
                from scipy.cluster.hierarchy import fcluster, linkage
                from scipy.spatial.distance import pdist as sp_pdist

                best_k = 0
                best_silhouette = -1
                best_labels = None

                for k_try in [2, 3]:
                    if n < k_try * 2:
                        continue
                    try:
                        Z = linkage(sp_pdist(proj), method='ward')
                        labels = fcluster(Z, t=k_try, criterion='maxclust')

                        # Compute simple intra/inter ratio
                        # (approximate silhouette)
                        intra_dists = []
                        inter_dists = []
                        for i in range(n):
                            same_cluster = [j for j in range(n) if labels[j] == labels[i] and j != i]
                            diff_cluster = [j for j in range(n) if labels[j] != labels[i]]
                            if same_cluster and diff_cluster:
                                intra = np.mean([np.linalg.norm(proj[i] - proj[j]) for j in same_cluster])
                                inter = np.mean([np.linalg.norm(proj[i] - proj[j]) for j in diff_cluster])
                                intra_dists.append(intra)
                                inter_dists.append(inter)

                        if intra_dists and inter_dists:
                            mean_intra = np.mean(intra_dists)
                            mean_inter = np.mean(inter_dists)
                            # Silhouette-like score
                            sil = (mean_inter - mean_intra) / max(mean_inter, 1e-10)

                            if sil > best_silhouette:
                                best_silhouette = sil
                                best_k = k_try
                                best_labels = labels
                    except:
                        continue

                if best_labels is not None:
                    cluster_sizes = [int(np.sum(best_labels == c+1)) for c in range(best_k)]
                    exp4_data[cat] = {
                        "n_subclusters": best_k,
                        "sub_sizes": cluster_sizes,
                        "silhouette": float(best_silhouette),
                    }

            # Print sub-clustering results
            if exp4_data:
                sorted_sub = sorted(exp4_data.items(), key=lambda x: x[1]["silhouette"], reverse=True)
                log(f"  Sub-clustering ranking (silhouette, descending):")
                for rank, (cat, d) in enumerate(sorted_sub, 1):
                    log(f"    {rank}. {cat}: k={d['n_subclusters']}, sizes={d['sub_sizes']}, "
                        f"silhouette={d['silhouette']:.3f}")

            results["exp4_subcluster"][li] = exp4_data

        # ===== Exp5: 跨层紧凑度演化 =====
        log(f"\n=== Exp5: Cross-Layer Compactness Evolution ===")

        # Track intra_dist_mean for each category across layers
        evolution_data = {}
        for cat in main_cats:
            evolution_data[cat] = {
                "layers": [],
                "intra_dist_mean": [],
                "dist_to_centroid_mean": [],
                "top1_ratio": [],
                "eccentricity": [],
            }

        for li in sorted(results["exp1_compactness"].keys()):
            for cat in main_cats:
                if cat in results["exp1_compactness"][li]:
                    evolution_data[cat]["layers"].append(li)
                    evolution_data[cat]["intra_dist_mean"].append(
                        results["exp1_compactness"][li][cat]["intra_dist_mean"])
                    evolution_data[cat]["dist_to_centroid_mean"].append(
                        results["exp1_compactness"][li][cat]["dist_to_centroid_mean"])
                if cat in results["exp2_anisotropy"].get(li, {}):
                    evolution_data[cat]["top1_ratio"].append(
                        results["exp2_anisotropy"][li][cat]["top1_ratio"])
                if cat in results["exp3_eccentricity"].get(li, {}):
                    evolution_data[cat]["eccentricity"].append(
                        results["exp3_eccentricity"][li][cat]["total_dist"])

        # Print evolution for key categories
        key_cats = ["body_part", "tool", "weapon", "fruit", "vegetable", "animal", "insect"]
        log("  Category compactness evolution (intra_dist_mean):")
        header = f"  {'Layer':>6}"
        for cat in key_cats:
            header += f"  {cat:>12}"
        log(header)

        for li in sorted(results["exp1_compactness"].keys()):
            row = f"  L{li:>4}"
            for cat in key_cats:
                if cat in results["exp1_compactness"][li]:
                    row += f"  {results['exp1_compactness'][li][cat]['intra_dist_mean']:>12.3f}"
                else:
                    row += f"  {'N/A':>12}"
            log(row)

        # Print anisotropy evolution
        log("\n  Category anisotropy evolution (top1_ratio):")
        header = f"  {'Layer':>6}"
        for cat in key_cats:
            header += f"  {cat:>12}"
        log(header)

        for li in sorted(results["exp2_anisotropy"].keys()):
            row = f"  L{li:>4}"
            for cat in key_cats:
                if cat in results["exp2_anisotropy"][li]:
                    row += f"  {results['exp2_anisotropy'][li][cat]['top1_ratio']:>12.3f}"
                else:
                    row += f"  {'N/A':>12}"
            log(row)

        # ===== Cross-category correlations =====
        log(f"\n=== Cross-Category Correlations at mid-layer ===")

        # Find the middle layer
        mid_li = sample_layers[len(sample_layers)//3]  # roughly L8-L10
        log(f"  Using layer {mid_li} for correlation analysis")

        if mid_li in results["exp1_compactness"] and mid_li in results["exp2_anisotropy"]:
            compactness = {}
            anisotropy = {}
            eccentricity = {}

            for cat in main_cats:
                if cat in results["exp1_compactness"][mid_li]:
                    compactness[cat] = results["exp1_compactness"][mid_li][cat]["intra_dist_mean"]
                if cat in results["exp2_anisotropy"][mid_li]:
                    anisotropy[cat] = results["exp2_anisotropy"][mid_li][cat]["top1_ratio"]
                if cat in results["exp3_eccentricity"][mid_li]:
                    eccentricity[cat] = results["exp3_eccentricity"][mid_li][cat]["total_dist"]

            # Correlation: compactness vs eccentricity
            common_cats = set(compactness.keys()) & set(eccentricity.keys())
            if len(common_cats) >= 5:
                comp_vals = [compactness[c] for c in common_cats]
                ecc_vals = [eccentricity[c] for c in common_cats]
                corr = np.corrcoef(comp_vals, ecc_vals)[0, 1]
                log(f"  Compactness vs Eccentricity correlation: {corr:.3f}")

                # Scatter data
                log(f"  Scatter data (cat: compactness, eccentricity):")
                for cat in sorted(common_cats):
                    log(f"    {cat}: compact={compactness[cat]:.3f}, ecc={eccentricity[cat]:.3f}")

            # Correlation: anisotropy vs compactness
            common_cats2 = set(compactness.keys()) & set(anisotropy.keys())
            if len(common_cats2) >= 5:
                comp_vals2 = [compactness[c] for c in common_cats2]
                aniso_vals = [anisotropy[c] for c in common_cats2]
                corr2 = np.corrcoef(comp_vals2, aniso_vals)[0, 1]
                log(f"  Compactness vs Anisotropy correlation: {corr2:.3f}")

            # Correlation: anisotropy vs eccentricity
            common_cats3 = set(anisotropy.keys()) & set(eccentricity.keys())
            if len(common_cats3) >= 5:
                aniso_vals2 = [anisotropy[c] for c in common_cats3]
                ecc_vals2 = [eccentricity[c] for c in common_cats3]
                corr3 = np.corrcoef(aniso_vals2, ecc_vals2)[0, 1]
                log(f"  Anisotropy vs Eccentricity correlation: {corr3:.3f}")

        # ===== Inter-category distance matrix (for reference) =====
        log(f"\n=== Inter-category Distance Matrix at L{mid_li} ===")

        if mid_li in results["exp1_compactness"]:
            cat_centroids = {}
            for cat in main_cats:
                if cat in cat_vectors:
                    cat_centroids[cat] = cat_vectors[cat].mean(axis=0)

            if len(cat_centroids) >= 5:
                # Compute pairwise centroid distances
                cats_with_centroids = sorted(cat_centroids.keys())
                n_cats = len(cats_with_centroids)
                dist_matrix = np.zeros((n_cats, n_cats))

                for i in range(n_cats):
                    for j in range(i+1, n_cats):
                        d = np.linalg.norm(cat_centroids[cats_with_centroids[i]] - cat_centroids[cats_with_centroids[j]])
                        dist_matrix[i, j] = d
                        dist_matrix[j, i] = d

                log(f"  Centroid distance matrix (abbreviated, top-5 closest pairs):")
                pairs = []
                for i in range(n_cats):
                    for j in range(i+1, n_cats):
                        pairs.append((cats_with_centroids[i], cats_with_centroids[j], dist_matrix[i, j]))
                pairs.sort(key=lambda x: x[2])
                for rank, (c1, c2, d) in enumerate(pairs[:10], 1):
                    sc1, sc2 = get_superclass(c1), get_superclass(c2)
                    cross = "CROSS" if sc1 != sc2 else "same"
                    log(f"    {rank}. {c1}-{c2}: {d:.3f} ({cross} {sc1}/{sc2})")

                log(f"\n  Top-5 most distant pairs:")
                for rank, (c1, c2, d) in enumerate(pairs[-5:], 1):
                    sc1, sc2 = get_superclass(c1), get_superclass(c2)
                    cross = "CROSS" if sc1 != sc2 else "same"
                    log(f"    {rank}. {c1}-{c2}: {d:.3f} ({cross} {sc1}/{sc2})")

        # ===== Summary =====
        log(f"\n=== Summary ===")
        log(f"Model: {model_name}, n_layers={n_layers}, d_model={d_model}")

        # Key layer data
        key_layers = [sample_layers[0], sample_layers[len(sample_layers)//3],
                      sample_layers[2*len(sample_layers)//3], sample_layers[-1]]
        key_layers = sorted(set(key_layers))

        for li in key_layers:
            log(f"\n  Layer {li}:")

            # Most/least compact
            if li in results["exp1_compactness"]:
                compact = results["exp1_compactness"][li]
                if compact:
                    most_compact = min(compact.items(), key=lambda x: x[1]["intra_dist_mean"])
                    least_compact = max(compact.items(), key=lambda x: x[1]["intra_dist_mean"])
                    log(f"    Most compact: {most_compact[0]} ({most_compact[1]['intra_dist_mean']:.3f})")
                    log(f"    Least compact: {least_compact[0]} ({least_compact[1]['intra_dist_mean']:.3f})")

            # Most/least anisotropic
            if li in results["exp2_anisotropy"]:
                aniso = results["exp2_anisotropy"][li]
                if aniso:
                    most_aniso = max(aniso.items(), key=lambda x: x[1]["top1_ratio"])
                    least_aniso = min(aniso.items(), key=lambda x: x[1]["top1_ratio"])
                    log(f"    Most anisotropic: {most_aniso[0]} (top1={most_aniso[1]['top1_ratio']:.3f})")
                    log(f"    Least anisotropic: {least_aniso[0]} (top1={least_aniso[1]['top1_ratio']:.3f})")

            # Most/least eccentric
            if li in results["exp3_eccentricity"]:
                ecc = results["exp3_eccentricity"][li]
                if ecc:
                    most_ecc = max(ecc.items(), key=lambda x: x[1]["total_dist"])
                    least_ecc = min(ecc.items(), key=lambda x: x[1]["total_dist"])
                    log(f"    Most eccentric: {most_ecc[0]} ({most_ecc[1]['total_dist']:.3f})")
                    log(f"    Least eccentric: {least_ecc[0]} ({least_ecc[1]['total_dist']:.3f})")

            # Sub-clustering
            if li in results["exp4_subcluster"]:
                sub = results["exp4_subcluster"][li]
                if sub:
                    cats_with_sub = [c for c, d in sub.items() if d["silhouette"] > 0.3]
                    log(f"    Categories with sub-clusters (sil>0.3): {cats_with_sub}")

        # ===== Save results =====
        save_dir = Path(rf"d:\Ai2050\TransformerLens-Project\results\causal_fiber\{model_name}_cclxxxix")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        clean_results = json.loads(json.dumps(results, default=convert))
        with open(save_dir / "results.json", 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)

        log(f"Results saved to {save_dir}")
        log(f"=== {model_name} COMPLETE ===")

    except Exception as e:
        log(f"ERROR in {model_name}: {e}")
        traceback.print_exc(file=open(LOG, 'a', encoding='utf-8'))

    finally:
        # Release GPU memory
        gc.collect()
        try:
            release_model(model)
        except:
            pass
        torch.cuda.empty_cache()
        log(f"GPU memory released for {model_name}")


if __name__ == "__main__":
    # 清空日志
    with open(LOG, 'w', encoding='utf-8') as f:
        f.write("")

    # 顺序运行三个模型
    for model_name in ["qwen3", "glm4", "deepseek7b"]:
        run_model(model_name)
        gc.collect()
        torch.cuda.empty_cache()
        log(f"Waiting 10s before next model...")
        time.sleep(10)

    log("\n=== ALL MODELS COMPLETE ===")
