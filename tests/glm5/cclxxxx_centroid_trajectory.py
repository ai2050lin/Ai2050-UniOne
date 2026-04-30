"""
CCLXXXX(290): 类别质心轨迹与成对距离演化
基于CCLXXXIX发现: body_part偏心率最高, weapon最低; L6紧凑度极小值
目标: 积累质心运动数据, 不做理论分析

Exp1: 质心轨迹 — 13个类别质心在各层的位置(投影到全局PC)
Exp2: 成对距离演化 — 78对类别质心距离的跨层变化
Exp3: 质心位移方向 — 相邻层间质心移动向量, 与超类方向对齐度
Exp4: 分离时序 — 每对类别何时达到最大/最小距离
Exp5: 超类内vs超类间距离演化 — superclass-level聚合
"""
import sys, os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxxx_log.txt"

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        import time
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()

log("=== CCLXXXX Script started ===")

import json, time, gc, traceback
from pathlib import Path
import numpy as np
from collections import defaultdict
from itertools import combinations

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
    if cat in ANIMATE_CATS: return "animate"
    if cat in PLANT_CATS: return "plant"
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

        # ===== 计算每层每个类别的质心 =====
        layer_cat_centroids = {}  # li -> {cat: centroid[d_model]}
        layer_global_mean = {}    # li -> mean[d_model]

        for li in sample_layers:
            cat_centroids = {}
            all_vecs_for_mean = []
            for cat in main_cats:
                words = CATEGORIES_13[cat][:20]
                vecs = []
                for w in words:
                    if w in word_layer_acts and li in word_layer_acts[w]:
                        vecs.append(word_layer_acts[w][li])
                if len(vecs) >= 5:
                    centroid = np.mean(vecs, axis=0)
                    cat_centroids[cat] = centroid
                    all_vecs_for_mean.extend(vecs)

            layer_cat_centroids[li] = cat_centroids
            if all_vecs_for_mean:
                layer_global_mean[li] = np.mean(all_vecs_for_mean, axis=0)
            else:
                layer_global_mean[li] = np.zeros(d_model)

        # ===== 全局SVD (用中间层的所有向量) =====
        mid_li = sample_layers[len(sample_layers)//2]
        all_vecs_mid = []
        for cat in main_cats:
            words = CATEGORIES_13[cat][:20]
            for w in words:
                if w in word_layer_acts and mid_li in word_layer_acts[w]:
                    all_vecs_mid.append(word_layer_acts[w][mid_li])
        all_vecs_mid = np.array(all_vecs_mid)
        global_mean_mid = all_vecs_mid.mean(axis=0)

        n_pc = 10
        k_svd = min(n_pc, min(all_vecs_mid.shape) - 1)
        _U_svd, S_svd, Vt_svd = np.linalg.svd(all_vecs_mid - global_mean_mid, full_matrices=False)
        Vt_global = Vt_svd[:k_svd]  # [k, d_model] — global PC directions
        log(f"Global SVD: k={k_svd}, top5 singular values: {S_svd[:5].tolist()}")

        # ================================================================
        # Exp1: 质心轨迹 — 13个类别质心在各层的位置(投影到全局PC)
        # ================================================================
        log("\n" + "="*70)
        log("Exp1: Category Centroid Trajectory in Global PC Space")
        log("="*70)

        centroid_trajectory = {}  # cat -> {li: projection[k]}

        for cat in main_cats:
            centroid_trajectory[cat] = {}
            log(f"\n--- {cat} (superclass: {get_superclass(cat)}) ---")
            for li in sample_layers:
                if cat in layer_cat_centroids[li]:
                    centroid = layer_cat_centroids[li][cat]
                    proj = (centroid - global_mean_mid) @ Vt_global.T  # [k]
                    centroid_trajectory[cat][li] = proj
                    log(f"  L{li}: PC0={proj[0]:.4f} PC1={proj[1]:.4f} PC2={proj[2]:.4f} | "
                        f"dist_to_mean={np.linalg.norm(centroid - global_mean_mid):.4f}")

        # 按超类分组显示轨迹
        log("\n--- Trajectory grouped by superclass ---")
        for sc_name, sc_cats in [("animate", ANIMATE_CATS), ("plant", PLANT_CATS),
                                  ("body", BODY_CATS), ("artifact", ARTIFACT_CATS)]:
            log(f"\n  [{sc_name}]")
            # 找公共的sample层
            common_layers = sample_layers
            header = f"  {'Cat':>12s}"
            for li in common_layers:
                header += f"  {'L'+str(li):>7s}"
            log(header)

            for cat in sorted(sc_cats):
                if cat not in centroid_trajectory:
                    continue
                row = f"  {cat:>12s}"
                for li in common_layers:
                    if li in centroid_trajectory[cat]:
                        proj = centroid_trajectory[cat][li]
                        row += f"  {proj[0]:7.3f}"
                    else:
                        row += f"  {'---':>7s}"
                log(row)

        # ================================================================
        # Exp2: 成对距离演化 — 78对类别质心距离的跨层变化
        # ================================================================
        log("\n" + "="*70)
        log("Exp2: Pairwise Centroid Distance Evolution")
        log("="*70)

        pair_dist_evolution = {}  # (cat1, cat2) -> {li: distance}
        all_pairs = list(combinations(main_cats, 2))

        for cat1, cat2 in all_pairs:
            pair_dist_evolution[(cat1, cat2)] = {}
            for li in sample_layers:
                if cat1 in layer_cat_centroids[li] and cat2 in layer_cat_centroids[li]:
                    c1 = layer_cat_centroids[li][cat1]
                    c2 = layer_cat_centroids[li][cat2]
                    dist = np.linalg.norm(c1 - c2)
                    pair_dist_evolution[(cat1, cat2)][li] = dist

        # 显示top-10最近和最远对 (在中间层)
        mid_layer = sample_layers[len(sample_layers)//2]
        log(f"\n--- Top-10 Closest Pairs at L{mid_layer} ---")
        pair_dists_mid = []
        for (c1, c2), dists in pair_dist_evolution.items():
            if mid_layer in dists:
                pair_dists_mid.append((c1, c2, dists[mid_layer]))
        pair_dists_mid.sort(key=lambda x: x[2])

        for i, (c1, c2, d) in enumerate(pair_dists_mid[:10]):
            sc1, sc2 = get_superclass(c1), get_superclass(c2)
            cross_sc = "CROSS" if sc1 != sc2 else "same"
            log(f"  {i+1:2d}. {c1:>12s} - {c2:<12s} dist={d:.4f}  [{sc1}/{sc2}] {cross_sc}")

        log(f"\n--- Top-10 Farthest Pairs at L{mid_layer} ---")
        for i, (c1, c2, d) in enumerate(pair_dists_mid[-10:]):
            sc1, sc2 = get_superclass(c1), get_superclass(c2)
            cross_sc = "CROSS" if sc1 != sc2 else "same"
            log(f"  {i+1:2d}. {c1:>12s} - {c2:<12s} dist={d:.4f}  [{sc1}/{sc2}] {cross_sc}")

        # 每对的跨层演化
        log(f"\n--- Pairwise Distance Evolution (all 78 pairs) ---")
        for (c1, c2), dists in sorted(pair_dist_evolution.items()):
            if not dists:
                continue
            sc1, sc2 = get_superclass(c1), get_superclass(c2)
            cross_sc = "CROSS" if sc1 != sc2 else "same"
            vals = [f"L{li}={dists[li]:.2f}" for li in sorted(dists.keys())]
            log(f"  {c1:>12s}-{c2:<12s} [{sc1[:4]}/{sc2[:4]}] {cross_sc:5s}: {' | '.join(vals[:8])}...")

        # ================================================================
        # Exp3: 质心位移方向 — 相邻层间质心移动向量
        # ================================================================
        log("\n" + "="*70)
        log("Exp3: Centroid Displacement Between Consecutive Layers")
        log("="*70)

        displacement = {}  # cat -> {li: displacement[d_model]}

        for cat in main_cats:
            displacement[cat] = {}
            prev_li = None
            for li in sample_layers:
                if cat in layer_cat_centroids[li]:
                    if prev_li is not None and cat in layer_cat_centroids[prev_li]:
                        disp = layer_cat_centroids[li][cat] - layer_cat_centroids[prev_li][cat]
                        displacement[cat][li] = disp
                prev_li = li

        # 计算位移向量在全局PC上的投影
        log("\n--- Displacement projected to Global PCs ---")
        for cat in main_cats:
            log(f"\n  {cat} (superclass: {get_superclass(cat)})")
            for li in sorted(displacement[cat].keys()):
                disp = displacement[cat][li]
                disp_proj = disp @ Vt_global.T  # [k]
                disp_norm = np.linalg.norm(disp)
                log(f"    L{li}: |disp|={disp_norm:.4f}  "
                    f"PC0={disp_proj[0]:.4f} PC1={disp_proj[1]:.4f} PC2={disp_proj[2]:.4f}")

        # 同超类质心位移方向的对齐度
        log("\n--- Same-superclass centroid displacement alignment ---")
        for sc_name, sc_cats in [("animate", ANIMATE_CATS), ("plant", PLANT_CATS),
                                  ("artifact", ARTIFACT_CATS)]:
            sc_cats_sorted = sorted(sc_cats)
            log(f"\n  [{sc_name}]")
            for li in sorted(sample_layers[1:], key=lambda x: x):
                # 计算该超类所有类别的位移
                sc_disps = []
                for cat in sc_cats_sorted:
                    if cat in displacement and li in displacement[cat]:
                        sc_disps.append((cat, displacement[cat][li]))

                if len(sc_disps) < 2:
                    continue

                # 计算所有对的余弦相似度
                cos_vals = []
                for i in range(len(sc_disps)):
                    for j in range(i+1, len(sc_disps)):
                        c1, d1 = sc_disps[i]
                        c2, d2 = sc_disps[j]
                        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
                        if n1 > 1e-6 and n2 > 1e-6:
                            cos_val = np.dot(d1, d2) / (n1 * n2)
                            cos_vals.append(cos_val)

                if cos_vals:
                    mean_cos = np.mean(cos_vals)
                    log(f"    L{li}: mean_cos={mean_cos:.4f} (n_pairs={len(cos_vals)})")

        # ================================================================
        # Exp4: 分离时序 — 每对类别何时达到最大/最小距离
        # ================================================================
        log("\n" + "="*70)
        log("Exp4: Separation Timing — When Each Pair Reaches Max/Min Distance")
        log("="*70)

        pair_timing = []
        for (c1, c2), dists in pair_dist_evolution.items():
            if len(dists) < 3:
                continue
            sorted_layers = sorted(dists.keys())
            dist_list = [dists[li] for li in sorted_layers]

            max_dist = max(dist_list)
            min_dist = min(dist_list)
            max_layer = sorted_layers[dist_list.index(max_dist)]
            min_layer = sorted_layers[dist_list.index(min_dist)]

            # 距离变化: 最终/最初比率
            first_d = dist_list[0]
            last_d = dist_list[-1]
            ratio = last_d / first_d if first_d > 1e-6 else 0

            sc1, sc2 = get_superclass(c1), get_superclass(c2)
            cross_sc = sc1 != sc2

            pair_timing.append({
                "pair": (c1, c2),
                "superclass": (sc1, sc2),
                "cross_superclass": cross_sc,
                "max_dist": max_dist,
                "max_layer": max_layer,
                "min_dist": min_dist,
                "min_layer": min_layer,
                "first_dist": first_d,
                "last_dist": last_d,
                "ratio_last_first": ratio,
            })

        # 按max_layer排序
        pair_timing.sort(key=lambda x: x["max_layer"])
        log(f"\n--- Pairs reaching MAX distance earliest → latest ---")
        for pt in pair_timing[:15]:
            c1, c2 = pt["pair"]
            log(f"  L{pt['max_layer']:2d} max: {c1:>12s}-{c2:<12s} "
                f"[{pt['superclass'][0][:4]}/{pt['superclass'][1][:4]}] "
                f"max_d={pt['max_dist']:.2f}  ratio={pt['ratio_last_first']:.2f}")

        log(f"\n--- Pairs reaching MAX distance latest ---")
        for pt in pair_timing[-15:]:
            c1, c2 = pt["pair"]
            log(f"  L{pt['max_layer']:2d} max: {c1:>12s}-{c2:<12s} "
                f"[{pt['superclass'][0][:4]}/{pt['superclass'][1][:4]}] "
                f"max_d={pt['max_dist']:.2f}  ratio={pt['ratio_last_first']:.2f}")

        # 按min_layer排序
        pair_timing.sort(key=lambda x: x["min_layer"])
        log(f"\n--- Pairs reaching MIN distance earliest → latest ---")
        for pt in pair_timing[:15]:
            c1, c2 = pt["pair"]
            log(f"  L{pt['min_layer']:2d} min: {c1:>12s}-{c2:<12s} "
                f"[{pt['superclass'][0][:4]}/{pt['superclass'][1][:4]}] "
                f"min_d={pt['min_dist']:.2f}")

        # 同超类 vs 跨超类 的 max_layer 分布
        log(f"\n--- Max-layer distribution: same-superclass vs cross-superclass ---")
        same_max_layers = [pt["max_layer"] for pt in pair_timing if not pt["cross_superclass"]]
        cross_max_layers = [pt["max_layer"] for pt in pair_timing if pt["cross_superclass"]]
        if same_max_layers:
            log(f"  Same-superclass: mean_max_L={np.mean(same_max_layers):.1f} ± {np.std(same_max_layers):.1f}")
        if cross_max_layers:
            log(f"  Cross-superclass: mean_max_L={np.mean(cross_max_layers):.1f} ± {np.std(cross_max_layers):.1f}")

        # ================================================================
        # Exp5: 超类级别距离演化
        # ================================================================
        log("\n" + "="*70)
        log("Exp5: Superclass-level Distance Evolution")
        log("="*70)

        superclasses = {"animate": ANIMATE_CATS, "plant": PLANT_CATS, "body": BODY_CATS, "artifact": ARTIFACT_CATS}

        # 计算超类质心
        layer_sc_centroids = {}  # li -> {sc: centroid}
        for li in sample_layers:
            layer_sc_centroids[li] = {}
            for sc_name, sc_cats in superclasses.items():
                sc_centroids = []
                for cat in sc_cats:
                    if cat in layer_cat_centroids[li]:
                        sc_centroids.append(layer_cat_centroids[li][cat])
                if sc_centroids:
                    layer_sc_centroids[li][sc_name] = np.mean(sc_centroids, axis=0)

        # 超类间成对距离
        sc_names = sorted(superclasses.keys())
        sc_pairs = list(combinations(sc_names, 2))
        log(f"\n--- Superclass inter-centroid distances across layers ---")
        header = f"  {'Pair':>25s}"
        for li in sample_layers:
            header += f"  {'L'+str(li):>7s}"
        log(header)

        for sc1, sc2 in sc_pairs:
            row = f"  {sc1:>12s}-{sc2:<12s}"
            for li in sample_layers:
                if sc1 in layer_sc_centroids[li] and sc2 in layer_sc_centroids[li]:
                    d = np.linalg.norm(layer_sc_centroids[li][sc1] - layer_sc_centroids[li][sc2])
                    row += f"  {d:7.2f}"
                else:
                    row += f"  {'---':>7s}"
            log(row)

        # 超类内类别间平均距离
        log(f"\n--- Intra-superclass average pairwise distance across layers ---")
        header = f"  {'Superclass':>12s}"
        for li in sample_layers:
            header += f"  {'L'+str(li):>7s}"
        log(header)

        for sc_name, sc_cats in superclasses.items():
            row = f"  {sc_name:>12s}"
            for li in sample_layers:
                sc_centroids = []
                for cat in sc_cats:
                    if cat in layer_cat_centroids[li]:
                        sc_centroids.append(layer_cat_centroids[li][cat])
                if len(sc_centroids) >= 2:
                    # 平均成对距离
                    dists = [np.linalg.norm(sc_centroids[i] - sc_centroids[j])
                             for i in range(len(sc_centroids))
                             for j in range(i+1, len(sc_centroids))]
                    avg_dist = np.mean(dists)
                    row += f"  {avg_dist:7.2f}"
                else:
                    row += f"  {'---':>7s}"
            log(row)

        # body_part到各超类质心的距离
        log(f"\n--- body_part distance to each superclass centroid across layers ---")
        header = f"  {'To':>12s}"
        for li in sample_layers:
            header += f"  {'L'+str(li):>7s}"
        log(header)

        for sc_name in ["animate", "plant", "artifact"]:
            row = f"  {sc_name:>12s}"
            for li in sample_layers:
                if "body_part" in layer_cat_centroids[li] and sc_name in layer_sc_centroids[li]:
                    d = np.linalg.norm(layer_cat_centroids[li]["body_part"] - layer_sc_centroids[li][sc_name])
                    row += f"  {d:7.2f}"
                else:
                    row += f"  {'---':>7s}"
            log(row)

        # weapon到各超类质心的距离
        log(f"\n--- weapon distance to each superclass centroid across layers ---")
        header = f"  {'To':>12s}"
        for li in sample_layers:
            header += f"  {'L'+str(li):>7s}"
        log(header)

        for sc_name in ["animate", "plant", "artifact"]:
            row = f"  {sc_name:>12s}"
            for li in sample_layers:
                if "weapon" in layer_cat_centroids[li] and sc_name in layer_sc_centroids[li]:
                    d = np.linalg.norm(layer_cat_centroids[li]["weapon"] - layer_sc_centroids[li][sc_name])
                    row += f"  {d:7.2f}"
                else:
                    row += f"  {'---':>7s}"
            log(row)

        # ===== 保存结果 =====
        results_dir = Path(r"d:\Ai2050\TransformerLens-Project\results\causal_fiber")
        model_results_dir = results_dir / f"{model_name}_cclxxxx"
        model_results_dir.mkdir(parents=True, exist_ok=True)

        # 保存质心轨迹 (numpy格式)
        for cat in main_cats:
            traj_data = {}
            for li in sample_layers:
                if cat in layer_cat_centroids[li]:
                    traj_data[str(li)] = layer_cat_centroids[li][cat].tolist()
            with open(model_results_dir / f"centroid_{cat}.json", 'w') as f:
                json.dump(traj_data, f)

        # 保存成对距离演化
        pair_dist_save = {}
        for (c1, c2), dists in pair_dist_evolution.items():
            pair_dist_save[f"{c1}-{c2}"] = {str(li): float(d) for li, d in dists.items()}
        with open(model_results_dir / "pair_distances.json", 'w') as f:
            json.dump(pair_dist_save, f)

        # 保存分离时序
        with open(model_results_dir / "pair_timing.json", 'w') as f:
            json.dump(pair_timing, f, default=str)

        log(f"\nResults saved to {model_results_dir}")

        # ===== Summary =====
        log("\n" + "="*70)
        log(f"=== Summary for {model_name} ===")
        log("="*70)

        # 1. body_part/weapon轨迹特征
        for cat in ["body_part", "weapon", "animal", "tool"]:
            if cat in centroid_trajectory:
                log(f"\n{cat} trajectory (PC0):")
                for li in sorted(centroid_trajectory[cat].keys()):
                    log(f"  L{li}: PC0={centroid_trajectory[cat][li][0]:.4f}")

        # 2. 最稳定最近对 (各层距离变化最小的top5)
        pair_stability = []
        for (c1, c2), dists in pair_dist_evolution.items():
            if len(dists) >= 5:
                dist_vals = list(dists.values())
                cv = np.std(dist_vals) / max(np.mean(dist_vals), 1e-6)  # coefficient of variation
                pair_stability.append((c1, c2, cv, np.mean(dist_vals)))
        pair_stability.sort(key=lambda x: x[2])

        log(f"\nMost stable pairs (lowest CV across layers):")
        for c1, c2, cv, mean_d in pair_stability[:5]:
            log(f"  {c1}-{c2}: CV={cv:.3f}, mean_dist={mean_d:.2f}")

        log(f"\nLeast stable pairs (highest CV across layers):")
        for c1, c2, cv, mean_d in pair_stability[-5:]:
            log(f"  {c1}-{c2}: CV={cv:.3f}, mean_dist={mean_d:.2f}")

        # 3. 距离变化比 (最终/最初) top5
        log(f"\nPairs with largest distance increase (last/first ratio):")
        pair_timing.sort(key=lambda x: x["ratio_last_first"], reverse=True)
        for pt in pair_timing[:5]:
            c1, c2 = pt["pair"]
            log(f"  {c1}-{c2}: ratio={pt['ratio_last_first']:.2f} "
                f"({pt['superclass'][0][:4]}/{pt['superclass'][1][:4]})")

        log(f"\nPairs with largest distance decrease (last/first ratio < 1):")
        pair_timing.sort(key=lambda x: x["ratio_last_first"])
        for pt in pair_timing[:5]:
            c1, c2 = pt["pair"]
            log(f"  {c1}-{c2}: ratio={pt['ratio_last_first']:.2f} "
                f"({pt['superclass'][0][:4]}/{pt['superclass'][1][:4]})")

        # 释放模型
        release_model(model)
        log(f"=== {model_name} done ===\n")
        gc.collect()

    except Exception as e:
        log(f"ERROR in {model_name}: {e}")
        traceback.print_exc(file=open(LOG, 'a', encoding='utf-8'))
        try:
            release_model(model)
        except:
            pass
        gc.collect()


if __name__ == "__main__":
    # 清空日志
    with open(LOG, 'w', encoding='utf-8') as f:
        f.write("")

    for model_name in ["qwen3", "glm4", "deepseek7b"]:
        run_model(model_name)
        gc.collect()

    log("\n=== ALL MODELS DONE ===")
