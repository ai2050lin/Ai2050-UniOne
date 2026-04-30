"""
CCLXXXXIII(293): 超类正交残差结构与逐层细化分析
基于CCLXXXXII发现: dim_90=8-10(不是4维), 超类重建率L8骤降(0.99→0.86), body_part独占PC0
目标:
  1. 超类正交残差中编码了什么? (9个额外维度)
  2. L8-L10超类失效的精确机制是什么?
  3. 超类内部的位移结构: 同一超类内的类别是否同方向运动?

Exp1: 位移分解 — 超类平行分量 vs 超类正交分量, 然后SVD残差
Exp2: 逐层分析 L0→L1→L2→...→L6 (精确到每层, 定位断裂点)
Exp3: 超类内部位移结构 — 同超类内的类别位移cos矩阵
"""
import sys, os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxxxiii_log.txt"

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        import time
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()

log("=== CCLXXXXIII Script started ===")

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
SUPERCLASS_MAP = {
    "animal": "animate", "bird": "animate", "fish": "animate", "insect": "animate",
    "plant": "plant", "fruit": "plant", "vegetable": "plant",
    "body_part": "body",
    "tool": "artifact", "vehicle": "artifact", "clothing": "artifact",
    "weapon": "artifact", "furniture": "artifact",
}

SUPERCLASS_NAMES = ["animate", "plant", "body", "artifact"]
SUPERCLASS_CATS_MAP = {
    "animate": sorted(ANIMATE_CATS),
    "plant": sorted(PLANT_CATS),
    "body": sorted(BODY_CATS),
    "artifact": sorted(ARTIFACT_CATS),
}


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
        word_layer_acts = {}

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

        main_cats = sorted(CATEGORIES_13.keys())

        # ===== 计算每层每个类别的质心 =====
        layer_cat_centroids = {}
        for li in range(n_layers):
            cat_centroids = {}
            for cat in main_cats:
                words = CATEGORIES_13[cat][:20]
                vecs = []
                for w in words:
                    if w in word_layer_acts and li in word_layer_acts[w]:
                        vecs.append(word_layer_acts[w][li])
                if len(vecs) >= 5:
                    centroid = np.mean(vecs, axis=0)
                    cat_centroids[cat] = centroid
            layer_cat_centroids[li] = cat_centroids

        # ===== 计算位移向量(逐层和隔2层) =====
        # 隔2层采样(与CCLXXXXII一致)
        if n_layers <= 20:
            sample_layers_2 = list(range(n_layers))
        elif n_layers <= 40:
            sample_layers_2 = list(range(0, n_layers, 2)) + [n_layers - 1]
        else:
            sample_layers_2 = list(range(0, n_layers, 3)) + [n_layers - 1]
        sample_layers_2 = sorted(set(sample_layers_2))

        displacement_2 = {}  # cat -> {(li1,li2): vector}
        for cat in main_cats:
            displacement_2[cat] = {}
            for i in range(len(sample_layers_2) - 1):
                li1, li2 = sample_layers_2[i], sample_layers_2[i+1]
                if cat in layer_cat_centroids[li1] and cat in layer_cat_centroids[li2]:
                    displacement_2[cat][(li1, li2)] = layer_cat_centroids[li2][cat] - layer_cat_centroids[li1][cat]

        # 逐层采样(L0→L10精确分析)
        fine_layers = list(range(min(12, n_layers)))
        displacement_1 = {}
        for cat in main_cats:
            displacement_1[cat] = {}
            for i in range(len(fine_layers) - 1):
                li1, li2 = fine_layers[i], fine_layers[i+1]
                if cat in layer_cat_centroids[li1] and cat in layer_cat_centroids[li2]:
                    displacement_1[cat][(li1, li2)] = layer_cat_centroids[li2][cat] - layer_cat_centroids[li1][cat]

        # ================================================================
        # Exp1: 位移分解 — 超类平行 vs 超类正交
        # ================================================================
        log("\n" + "="*70)
        log("Exp1: Displacement Decomposition (Superclass-parallel vs Orthogonal)")
        log("="*70)

        for i in range(len(sample_layers_2) - 1):
            li1, li2 = sample_layers_2[i], sample_layers_2[i+1]

            # 计算超类平均位移
            sc_mean_disp = {}
            for sc_name in SUPERCLASS_NAMES:
                sc_cats = SUPERCLASS_CATS_MAP[sc_name]
                disps = [displacement_2[cat][(li1, li2)] for cat in sc_cats
                         if (li1, li2) in displacement_2.get(cat, {})]
                if len(disps) >= 1:
                    sc_mean_disp[sc_name] = np.mean(disps, axis=0)

            if len(sc_mean_disp) < 3:
                continue

            # 构建超类方向基
            sc_matrix = np.array([sc_mean_disp[sc] for sc in SUPERCLASS_NAMES
                                  if sc in sc_mean_disp])  # [n_sc, d_model]
            n_sc = sc_matrix.shape[0]

            # 对超类方向做QR分解得到正交基
            Q, R = np.linalg.qr(sc_matrix.T)  # Q: [d_model, n_sc]
            # Q的列就是超类子空间的正交基

            # 收集所有类别的位移
            disp_vectors = []
            disp_cats_list = []
            for cat in main_cats:
                if (li1, li2) in displacement_2.get(cat, {}):
                    disp_vectors.append(displacement_2[cat][(li1, li2)])
                    disp_cats_list.append(cat)

            if len(disp_vectors) < 3:
                continue

            disp_matrix = np.array(disp_vectors)  # [n_cats, d_model]

            # 分解: v = v_parallel + v_orthogonal
            # v_parallel = Q @ Q^T @ v
            # v_orthogonal = v - v_parallel

            parallel_matrix = disp_matrix @ Q @ Q.T  # [n_cats, d_model]
            orth_matrix = disp_matrix - parallel_matrix  # [n_cats, d_model]

            # 统计各分量的能量
            total_energy = np.sum(disp_matrix ** 2)
            parallel_energy = np.sum(parallel_matrix ** 2)
            orth_energy = np.sum(orth_matrix ** 2)

            para_ratio = parallel_energy / total_energy if total_energy > 0 else 0
            orth_ratio = orth_energy / total_energy if total_energy > 0 else 0

            log(f"\n--- L{li1}-L{li2}: Decomposition ---")
            log(f"  Total displacement energy: {total_energy:.2f}")
            log(f"  Superclass-parallel energy: {parallel_energy:.2f} ({para_ratio:.4f})")
            log(f"  Superclass-orthogonal energy: {orth_energy:.2f} ({orth_ratio:.4f})")

            # 逐类别的分解
            log(f"  Per-category decomposition (||parallel|| / ||total||):")
            for j, cat in enumerate(disp_cats_list):
                v = disp_matrix[j]
                v_par = parallel_matrix[j]
                v_orth = orth_matrix[j]
                v_total = np.linalg.norm(v)
                v_par_norm = np.linalg.norm(v_par)
                v_orth_norm = np.linalg.norm(v_orth)
                par_frac = v_par_norm ** 2 / (v_total ** 2) if v_total > 1e-10 else 0
                sc = SUPERCLASS_MAP[cat]
                log(f"    {cat:>12s} [{sc[:4]}]: ||total||={v_total:.2f}, ||par||={v_par_norm:.2f} ({par_frac:.4f}), ||orth||={v_orth_norm:.2f}")

            # SVD of orthogonal residuals
            if orth_energy > 1e-10:
                orth_centered = orth_matrix - orth_matrix.mean(axis=0)
                _U, S, Vt = np.linalg.svd(orth_centered, full_matrices=False)
                total_var = np.sum(S ** 2)
                if total_var > 1e-10:
                    cumvar = np.cumsum(S ** 2) / total_var
                    dim_90 = np.searchsorted(cumvar, 0.90) + 1
                    dim_95 = np.searchsorted(cumvar, 0.95) + 1
                    log(f"  Orthogonal residual SVD:")
                    log(f"    dim_90={dim_90}, dim_95={dim_95}")
                    log(f"    Top-8 singular values: {[f'{s:.2f}' for s in S[:8]]}")
                    log(f"    S[0]/total={S[0]**2/total_var:.4f}, S[0]/S[1]={S[0]/S[1]:.3f}" if len(S) > 1 else "")

                    # PC0负载 of orthogonal residuals
                    u0 = _U[:, 0]
                    log(f"    Orthogonal PC0 loadings (top-5):")
                    sorted_idx = np.argsort(np.abs(u0))[::-1]
                    for j in sorted_idx[:5]:
                        cat = disp_cats_list[j]
                        sc = SUPERCLASS_MAP[cat]
                        log(f"      {cat:>12s} [{sc[:4]}]: u0={u0[j]:+.4f}")

                    # PC1负载
                    if len(S) > 1:
                        u1 = _U[:, 1]
                        log(f"    Orthogonal PC1 loadings (top-5):")
                        sorted_idx1 = np.argsort(np.abs(u1))[::-1]
                        for j in sorted_idx1[:5]:
                            cat = disp_cats_list[j]
                            sc = SUPERCLASS_MAP[cat]
                            log(f"      {cat:>12s} [{sc[:4]}]: u1={u1[j]:+.4f}")

        # ================================================================
        # Exp2: 逐层分析 L0→L1→L2→...→L11
        # ================================================================
        log("\n" + "="*70)
        log("Exp2: Per-Layer Fine-Grained Analysis (L0→L11)")
        log("="*70)

        for li_start in range(min(11, n_layers - 1)):
            li1, li2 = li_start, li_start + 1

            disp_vectors = []
            disp_cats_list = []
            for cat in main_cats:
                if (li1, li2) in displacement_1.get(cat, {}):
                    disp_vectors.append(displacement_1[cat][(li1, li2)])
                    disp_cats_list.append(cat)

            if len(disp_vectors) < 3:
                continue

            disp_matrix = np.array(disp_vectors)
            n_cats_seg = len(disp_cats_list)

            # 超类重建率
            sc_mean_disp = {}
            for sc_name in SUPERCLASS_NAMES:
                sc_cats = SUPERCLASS_CATS_MAP[sc_name]
                disps = [displacement_1[cat][(li1, li2)] for cat in sc_cats
                         if (li1, li2) in displacement_1.get(cat, {})]
                if len(disps) >= 1:
                    sc_mean_disp[sc_name] = np.mean(disps, axis=0)

            sc_recon_rate = None
            if len(sc_mean_disp) >= 3:
                sc_matrix = np.array([sc_mean_disp[sc] for sc in SUPERCLASS_NAMES
                                      if sc in sc_mean_disp])
                Q_sc, _ = np.linalg.qr(sc_matrix.T)
                proj = disp_matrix @ Q_sc @ Q_sc.T
                total_e = np.sum(disp_matrix ** 2)
                orth_e = np.sum((disp_matrix - proj) ** 2)
                sc_recon_rate = 1 - orth_e / total_e if total_e > 0 else 0

            # SVD of displacements
            disp_centered = disp_matrix - disp_matrix.mean(axis=0)
            _U, S, Vt = np.linalg.svd(disp_centered, full_matrices=False)
            total_var = np.sum(S ** 2)
            cumvar = np.cumsum(S ** 2) / total_var if total_var > 0 else np.zeros(len(S))
            dim_90 = np.searchsorted(cumvar, 0.90) + 1 if total_var > 0 else 0

            # 所有类别间的平均cos
            cos_sum, cos_count = 0.0, 0
            for j1 in range(n_cats_seg):
                for j2 in range(j1 + 1, n_cats_seg):
                    v1 = disp_matrix[j1]
                    v2 = disp_matrix[j2]
                    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if n1 > 1e-8 and n2 > 1e-8:
                        cos_sum += np.dot(v1, v2) / (n1 * n2)
                        cos_count += 1
            avg_cos = cos_sum / cos_count if cos_count > 0 else 0

            # 速度(平均位移norm)
            avg_speed = np.mean([np.linalg.norm(disp_matrix[j]) for j in range(n_cats_seg)])

            # PC0负载
            u0 = _U[:, 0]
            sorted_idx = np.argsort(np.abs(u0))[::-1]
            pc0_top2 = [(disp_cats_list[j], u0[j]) for j in sorted_idx[:2]]

            log(f"\n--- L{li1}→L{li2} (per-layer) ---")
            log(f"  avg_speed={avg_speed:.2f}, avg_cos={avg_cos:.4f}, dim_90={dim_90}, sc_recon={sc_recon_rate:.4f}" if sc_recon_rate is not None else f"  avg_speed={avg_speed:.2f}, avg_cos={avg_cos:.4f}, dim_90={dim_90}")
            log(f"  PC0 top-2: {[(c, f'{v:+.3f}') for c, v in pc0_top2]}")

            # 关键层对: 跨超类的最小cos和最大cos
            cross_cos_list = []
            within_cos_list = []
            for j1 in range(n_cats_seg):
                for j2 in range(j1 + 1, n_cats_seg):
                    v1, v2 = disp_matrix[j1], disp_matrix[j2]
                    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if n1 > 1e-8 and n2 > 1e-8:
                        c = np.dot(v1, v2) / (n1 * n2)
                        sc1 = SUPERCLASS_MAP[disp_cats_list[j1]]
                        sc2 = SUPERCLASS_MAP[disp_cats_list[j2]]
                        if sc1 != sc2:
                            cross_cos_list.append((disp_cats_list[j1], disp_cats_list[j2], c))
                        else:
                            within_cos_list.append((disp_cats_list[j1], disp_cats_list[j2], c))

            if cross_cos_list:
                cross_cos_list.sort(key=lambda x: x[2])
                log(f"  Lowest cross-superclass cos: {cross_cos_list[0][0]}-{cross_cos_list[0][1]}={cross_cos_list[0][2]:.4f}")
                cross_cos_list.sort(key=lambda x: x[2], reverse=True)
                log(f"  Highest cross-superclass cos: {cross_cos_list[0][0]}-{cross_cos_list[0][1]}={cross_cos_list[0][2]:.4f}")

            if within_cos_list:
                within_cos_list.sort(key=lambda x: x[2])
                log(f"  Lowest within-superclass cos: {within_cos_list[0][0]}-{within_cos_list[0][1]}={within_cos_list[0][2]:.4f}")
                within_cos_avg = np.mean([c for _, _, c in within_cos_list])
                log(f"  Average within-superclass cos: {within_cos_avg:.4f}")

            # 逐类别位移norm
            log(f"  Per-category speed:")
            speeds = [(disp_cats_list[j], np.linalg.norm(disp_matrix[j])) for j in range(n_cats_seg)]
            speeds.sort(key=lambda x: x[1], reverse=True)
            for cat, spd in speeds:
                sc = SUPERCLASS_MAP[cat]
                log(f"    {cat:>12s} [{sc[:4]}]: {spd:.2f}")

        # ================================================================
        # Exp3: 超类内部位移结构
        # ================================================================
        log("\n" + "="*70)
        log("Exp3: Within-Superclass Displacement Structure")
        log("="*70)

        for i in range(len(sample_layers_2) - 1):
            li1, li2 = sample_layers_2[i], sample_layers_2[i+1]

            log(f"\n--- L{li1}-L{li2}: Within-superclass analysis ---")

            for sc_name in SUPERCLASS_NAMES:
                sc_cats = SUPERCLASS_CATS_MAP[sc_name]
                if len(sc_cats) < 2:
                    # body只有1个类别, 跳过
                    log(f"  {sc_name}: only {len(sc_cats)} category, skipping")
                    continue

                # 收集该超类所有类别的位移
                sc_disps = {}
                for cat in sc_cats:
                    if (li1, li2) in displacement_2.get(cat, {}):
                        sc_disps[cat] = displacement_2[cat][(li1, li2)]

                if len(sc_disps) < 2:
                    continue

                # 超类内部的cos矩阵
                cats_in_sc = sorted(sc_disps.keys())
                log(f"  {sc_name} ({len(cats_in_sc)} cats): within-superclass cos matrix")

                cos_pairs = []
                for c1, c2 in combinations(cats_in_sc, 2):
                    v1, v2 = sc_disps[c1], sc_disps[c2]
                    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if n1 > 1e-8 and n2 > 1e-8:
                        cos_val = np.dot(v1, v2) / (n1 * n2)
                        cos_pairs.append((c1, c2, cos_val))

                if not cos_pairs:
                    continue

                cos_pairs.sort(key=lambda x: x[2])
                avg_cos = np.mean([c for _, _, c in cos_pairs])
                min_cos = cos_pairs[0]
                max_cos = cos_pairs[-1]

                log(f"    avg_cos={avg_cos:.4f}")
                log(f"    Most different: {min_cos[0]}-{min_cos[1]} cos={min_cos[2]:.4f}")
                log(f"    Most similar: {max_cos[0]}-{max_cos[1]} cos={max_cos[2]:.4f}")

                # 如果超类有3+类别, SVD看内在维度
                if len(sc_disps) >= 3:
                    sc_disp_matrix = np.array([sc_disps[c] for c in cats_in_sc])
                    sc_disp_centered = sc_disp_matrix - sc_disp_matrix.mean(axis=0)
                    _U_sc, S_sc, _ = np.linalg.svd(sc_disp_centered, full_matrices=False)
                    total_var_sc = np.sum(S_sc ** 2)
                    if total_var_sc > 1e-10:
                        cumvar_sc = np.cumsum(S_sc ** 2) / total_var_sc
                        log(f"    Intrinsic dim: dim_90={np.searchsorted(cumvar_sc, 0.90)+1}, S[0]/total={S_sc[0]**2/total_var_sc:.4f}")

                        # PC0负载
                        u0_sc = _U_sc[:, 0]
                        sorted_idx_sc = np.argsort(np.abs(u0_sc))[::-1]
                        log(f"    PC0 loadings:")
                        for j in sorted_idx_sc:
                            log(f"      {cats_in_sc[j]:>12s}: u0={u0_sc[j]:+.4f}")

        # ===== 保存结果 =====
        results_dir = f"d:/Ai2050/TransformerLens-Project/results/causal_fiber/{model_name}_cclxxxxiii"
        os.makedirs(results_dir, exist_ok=True)

        # 保存关键数据
        summary_data = {
            "per_layer_analysis": {},
            "decomposition": {},
        }

        # 逐层数据
        for li_start in range(min(11, n_layers - 1)):
            li1, li2 = li_start, li_start + 1
            disp_vectors = []
            for cat in main_cats:
                if (li1, li2) in displacement_1.get(cat, {}):
                    disp_vectors.append(displacement_1[cat][(li1, li2)])
            if len(disp_vectors) < 3:
                continue
            disp_matrix = np.array(disp_vectors)
            disp_centered = disp_matrix - disp_matrix.mean(axis=0)
            _U, S, Vt = np.linalg.svd(disp_centered, full_matrices=False)
            total_var = np.sum(S ** 2)
            cumvar = np.cumsum(S ** 2) / total_var if total_var > 0 else np.zeros(len(S))
            dim_90 = int(np.searchsorted(cumvar, 0.90) + 1) if total_var > 0 else 0

            # 超类重建率
            sc_mean_disp = {}
            for sc_name in SUPERCLASS_NAMES:
                sc_cats = SUPERCLASS_CATS_MAP[sc_name]
                disps = [displacement_1[cat][(li1, li2)] for cat in sc_cats
                         if (li1, li2) in displacement_1.get(cat, {})]
                if len(disps) >= 1:
                    sc_mean_disp[sc_name] = np.mean(disps, axis=0)

            sc_recon_rate = None
            if len(sc_mean_disp) >= 3:
                sc_matrix = np.array([sc_mean_disp[sc] for sc in SUPERCLASS_NAMES if sc in sc_mean_disp])
                Q_sc, _ = np.linalg.qr(sc_matrix.T)
                proj = disp_matrix @ Q_sc @ Q_sc.T
                total_e = np.sum(disp_matrix ** 2)
                orth_e = np.sum((disp_matrix - proj) ** 2)
                sc_recon_rate = float(1 - orth_e / total_e) if total_e > 0 else 0

            avg_speed = float(np.mean([np.linalg.norm(disp_matrix[j]) for j in range(disp_matrix.shape[0])]))

            cos_vals = []
            for j1 in range(disp_matrix.shape[0]):
                for j2 in range(j1+1, disp_matrix.shape[0]):
                    v1, v2 = disp_matrix[j1], disp_matrix[j2]
                    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if n1 > 1e-8 and n2 > 1e-8:
                        cos_vals.append(float(np.dot(v1, v2) / (n1 * n2)))

            summary_data["per_layer_analysis"][f"L{li1}_L{li2}"] = {
                "dim_90": dim_90,
                "sc_recon_rate": sc_recon_rate,
                "avg_speed": avg_speed,
                "avg_cos": float(np.mean(cos_vals)) if cos_vals else 0,
            }

        with open(os.path.join(results_dir, "summary.json"), 'w') as f:
            json.dump(summary_data, f, indent=2)

        log(f"Results saved to {results_dir}")

        # ===== 清理 =====
        del word_layer_acts, layer_cat_centroids, displacement_2, displacement_1
        gc.collect()

        log(f"=== {model_name} done ===\n")
        return True

    except Exception as e:
        log(f"ERROR in {model_name}: {e}")
        log(traceback.format_exc())
        return False
    finally:
        gc.collect()
        try:
            release_model(model)
        except:
            pass


if __name__ == "__main__":
    # 清空日志
    with open(LOG, 'w', encoding='utf-8') as f:
        f.write("")

    models = ["qwen3", "glm4", "deepseek7b"]
    for m in models:
        success = run_model(m)
        if not success:
            log(f"WARNING: {m} failed, continuing to next model")
        time.sleep(5)
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

    log("=== ALL MODELS DONE ===")
