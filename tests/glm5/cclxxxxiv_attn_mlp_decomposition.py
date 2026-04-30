"""
CCLXXXXIV(294): 断裂层的Attention vs MLP贡献分解 + 逐头分析
基于CCLXXXXIII发现: 精确断裂层 Qwen3=L6→L7, GLM4=L2→L3, DS7B=L7→L8

Exp1: 每层Attention vs MLP位移分解
  - Hook attn output和MLP output
  - 计算per-category attn_delta, mlp_delta
  - 关键: 断裂层哪个组件主导?

Exp2: 断裂层逐头分析
  - Hook o_proj input获取per-head outputs
  - 计算每个头的输出向量
  - 哪些头贡献最大? 哪些头的category selectivity最高?

Exp3: 断裂层前后的头方向变化
  - 比较断裂层和前一层的头输出
  - 哪些头变化最大?
"""
import sys, os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxxxiv_log.txt"

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        import time
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()

log("=== CCLXXXXIV Script started ===")

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

# 断裂层配置 (来自CCLXXXXIII)
FRACTURE_LAYERS = {
    "qwen3": 6,       # L6→L7
    "glm4": 2,        # L2→L3
    "deepseek7b": 7,  # L7→L8
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

        # Get number of attention heads from model config
        n_heads = model.config.num_attention_heads
        d_head = d_model // n_heads

        log(f"Model: {n_layers}L, d={d_model}, n_heads={n_heads}, d_head={d_head}")

        fracture_layer = FRACTURE_LAYERS.get(model_name, 6)
        log(f"Fracture layer for {model_name}: L{fracture_layer}")

        # Collect words
        all_words, all_cats = [], []
        for cat, words in CATEGORIES_13.items():
            all_words.extend(words[:20])
            all_cats.extend([cat] * 20)
        n_total = len(all_words)
        log(f"Total words: {n_total}")

        # ===== Forward pass: collect attn/MLP outputs =====
        template = "The {} is"

        # Storage for each word's layer components
        word_attn = {}    # word -> {li: attn_delta_vector}
        word_mlp = {}     # word -> {li: mlp_delta_vector}
        word_head_concat = {}  # word -> {li: concat_head_vector} (only at fracture region)

        # Which layers to collect head_concat for
        head_collect_layers = set(range(max(0, fracture_layer - 2), min(n_layers, fracture_layer + 3)))

        t0 = time.time()
        for wi, word in enumerate(all_words):
            text = template.format(word)
            input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
            last_pos = input_ids.shape[1] - 1

            attn_out = {}
            mlp_out = {}
            head_concat = {}

            hooks = []
            for li in range(n_layers):
                layer = layers_list[li]

                # Hook self_attn output
                def make_attn_hook(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            a = output[0]
                        else:
                            a = output
                        attn_out[key] = a[0, last_pos].detach().float().cpu().numpy()
                    return hook
                hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(f"L{li}")))

                # Hook MLP output
                def make_mlp_hook(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            a = output[0]
                        else:
                            a = output
                        mlp_out[key] = a[0, last_pos].detach().float().cpu().numpy()
                    return hook
                hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(f"L{li}")))

                # Hook o_proj input for per-head analysis (only at fracture region)
                if li in head_collect_layers:
                    def make_o_proj_hook(key):
                        def hook(module, args):
                            a = args[0] if not isinstance(args, tuple) else args[0]
                            head_concat[key] = a[0, last_pos].detach().float().cpu().numpy()
                        return hook
                    hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(make_o_proj_hook(f"L{li}")))

            with torch.no_grad():
                _ = model(input_ids)
            for h in hooks:
                h.remove()

            word_attn[word] = {li: attn_out.get(f"L{li}", None) for li in range(n_layers)}
            word_mlp[word] = {li: mlp_out.get(f"L{li}", None) for li in range(n_layers)}
            word_head_concat[word] = {li: head_concat.get(f"L{li}", None) for li in head_collect_layers}

            if (wi + 1) % 50 == 0 or wi == 0:
                log(f"  Word {wi+1}/{n_total} ({time.time()-t0:.0f}s)")

        log(f"Data collection done ({time.time()-t0:.0f}s)")

        main_cats = sorted(CATEGORIES_13.keys())

        # ===== Compute per-category centroids for each component =====
        attn_centroids = defaultdict(dict)
        mlp_centroids = defaultdict(dict)
        total_centroids = defaultdict(dict)

        for li in range(n_layers):
            for cat in main_cats:
                words = CATEGORIES_13[cat][:20]
                attn_vecs, mlp_vecs, total_vecs = [], [], []
                for w in words:
                    a = word_attn.get(w, {}).get(li)
                    m = word_mlp.get(w, {}).get(li)
                    if a is not None:
                        attn_vecs.append(a)
                        total_vecs.append(a + (m if m is not None else 0))
                    if m is not None:
                        mlp_vecs.append(m)

                if len(attn_vecs) >= 5:
                    attn_centroids[li][cat] = np.mean(attn_vecs, axis=0)
                if len(mlp_vecs) >= 5:
                    mlp_centroids[li][cat] = np.mean(mlp_vecs, axis=0)
                if len(total_vecs) >= 5:
                    total_centroids[li][cat] = np.mean(total_vecs, axis=0)

        # ================================================================
        # Exp1: Attention vs MLP displacement decomposition
        # ================================================================
        log("\n" + "="*70)
        log("Exp1: Attention vs MLP Displacement Decomposition (per layer)")
        log("="*70)

        exp1_summary = {}

        for li in range(min(16, n_layers)):  # First 16 layers for detailed analysis
            cats_present = [c for c in main_cats if c in attn_centroids[li] and c in mlp_centroids[li]]
            if len(cats_present) < 3:
                continue

            attn_disp = np.array([attn_centroids[li][c] for c in cats_present])
            mlp_disp = np.array([mlp_centroids[li][c] for c in cats_present])
            total_disp = attn_disp + mlp_disp

            # Energy decomposition
            total_energy = np.sum(total_disp ** 2)
            attn_energy = np.sum(attn_disp ** 2)
            mlp_energy = np.sum(mlp_disp ** 2)
            cross_energy = 2 * np.sum(attn_disp * mlp_disp)

            attn_ratio = attn_energy / total_energy if total_energy > 0 else 0
            mlp_ratio = mlp_energy / total_energy if total_energy > 0 else 0
            cross_ratio = cross_energy / total_energy if total_energy > 0 else 0

            # avg_cos for each component separately
            def compute_avg_cos(disp_matrix):
                cos_vals = []
                for j1 in range(disp_matrix.shape[0]):
                    for j2 in range(j1+1, disp_matrix.shape[0]):
                        v1, v2 = disp_matrix[j1], disp_matrix[j2]
                        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                        if n1 > 1e-8 and n2 > 1e-8:
                            cos_vals.append(np.dot(v1, v2) / (n1 * n2))
                return np.mean(cos_vals) if cos_vals else 0

            attn_avg_cos = compute_avg_cos(attn_disp)
            mlp_avg_cos = compute_avg_cos(mlp_disp)
            total_avg_cos = compute_avg_cos(total_disp)

            # Attn-MLP alignment
            attn_mlp_cos_vals = []
            for j in range(len(cats_present)):
                a, m = attn_disp[j], mlp_disp[j]
                na, nm = np.linalg.norm(a), np.linalg.norm(m)
                if na > 1e-8 and nm > 1e-8:
                    attn_mlp_cos_vals.append(np.dot(a, m) / (na * nm))
            attn_mlp_align = np.mean(attn_mlp_cos_vals) if attn_mlp_cos_vals else 0

            # Per-superclass breakdown
            sc_breakdown = {}
            for sc_name in SUPERCLASS_NAMES:
                sc_cats = [c for c in SUPERCLASS_CATS_MAP[sc_name] if c in cats_present]
                if len(sc_cats) >= 2:
                    sc_attn = np.array([attn_centroids[li][c] for c in sc_cats])
                    sc_mlp = np.array([mlp_centroids[li][c] for c in sc_cats])
                    sc_attn_cos = compute_avg_cos(sc_attn)
                    sc_mlp_cos = compute_avg_cos(sc_mlp)
                    sc_breakdown[sc_name] = {
                        "attn_cos": sc_attn_cos,
                        "mlp_cos": sc_mlp_cos,
                    }

            marker = ""
            if li == fracture_layer:
                marker = " *** FRACTURE LAYER ***"
            elif li == fracture_layer - 1:
                marker = " *** Pre-fracture ***"
            elif li == fracture_layer + 1:
                marker = " *** Post-fracture ***"

            log(f"\n--- L{li}: Attention vs MLP decomposition ---{marker}")
            log(f"  Energy: attn={attn_energy:.1f} ({attn_ratio:.4f}), mlp={mlp_energy:.1f} ({mlp_ratio:.4f}), cross={cross_ratio:.4f}")
            log(f"  avg_cos: attn={attn_avg_cos:.4f}, mlp={mlp_avg_cos:.4f}, total={total_avg_cos:.4f}")
            log(f"  attn-mlp alignment: {attn_mlp_align:.4f}")

            if sc_breakdown:
                log(f"  Per-superclass avg_cos:")
                for sc_name, bd in sc_breakdown.items():
                    log(f"    {sc_name:>10s}: attn_cos={bd['attn_cos']:.4f}, mlp_cos={bd['mlp_cos']:.4f}")

            # Per-category speed breakdown
            log(f"  Per-category speed (attn/total):")
            for j, cat in enumerate(cats_present):
                attn_norm = np.linalg.norm(attn_disp[j])
                mlp_norm = np.linalg.norm(mlp_disp[j])
                total_norm = np.linalg.norm(total_disp[j])
                sc = SUPERCLASS_MAP[cat]
                log(f"    {cat:>12s} [{sc[:4]}]: attn={attn_norm:.2f}, mlp={mlp_norm:.2f}, total={total_norm:.2f}")

            exp1_summary[f"L{li}"] = {
                "attn_ratio": float(attn_ratio),
                "mlp_ratio": float(mlp_ratio),
                "attn_avg_cos": float(attn_avg_cos),
                "mlp_avg_cos": float(mlp_avg_cos),
                "total_avg_cos": float(total_avg_cos),
                "attn_mlp_align": float(attn_mlp_align),
                "is_fracture": li == fracture_layer,
            }

        # ===== Key comparison: Pre-fracture vs Fracture vs Post-fracture =====
        log("\n" + "="*70)
        log("Exp1 Summary: Pre-fracture vs Fracture vs Post-fracture")
        log("="*70)

        for label, li in [("Pre-fracture", fracture_layer - 1),
                          ("FRACTURE", fracture_layer),
                          ("Post-fracture", fracture_layer + 1)]:
            if li < 0 or li >= n_layers:
                continue
            cats_present = [c for c in main_cats if c in attn_centroids[li] and c in mlp_centroids[li]]
            if len(cats_present) < 3:
                continue

            attn_disp = np.array([attn_centroids[li][c] for c in cats_present])
            mlp_disp = np.array([mlp_centroids[li][c] for c in cats_present])

            attn_avg_cos = compute_avg_cos(attn_disp)
            mlp_avg_cos = compute_avg_cos(mlp_disp)

            log(f"\n  {label} (L{li}):")
            log(f"    attn_avg_cos = {attn_avg_cos:.4f}")
            log(f"    mlp_avg_cos  = {mlp_avg_cos:.4f}")
            log(f"    diff (attn - mlp) = {attn_avg_cos - mlp_avg_cos:.4f}")

            # Which component drives the category differentiation?
            if label == "FRACTURE":
                # Compare with pre-fracture
                li_prev = fracture_layer - 1
                if li_prev >= 0:
                    cats_prev = [c for c in main_cats if c in attn_centroids[li_prev] and c in mlp_centroids[li_prev]]
                    if len(cats_prev) >= 3:
                        attn_prev = np.array([attn_centroids[li_prev][c] for c in cats_prev])
                        mlp_prev = np.array([mlp_centroids[li_prev][c] for c in cats_prev])
                        attn_prev_cos = compute_avg_cos(attn_prev)
                        mlp_prev_cos = compute_avg_cos(mlp_prev)

                        log(f"\n  Cos drop from Pre-fracture to Fracture:")
                        log(f"    attn: {attn_prev_cos:.4f} -> {attn_avg_cos:.4f} (drop={attn_prev_cos - attn_avg_cos:.4f})")
                        log(f"    mlp:  {mlp_prev_cos:.4f} -> {mlp_avg_cos:.4f} (drop={mlp_prev_cos - mlp_avg_cos:.4f})")

                        if (attn_prev_cos - attn_avg_cos) > (mlp_prev_cos - mlp_avg_cos):
                            log(f"    >>> ATTENTION drops more — attn drives the fracture <<<")
                        else:
                            log(f"    >>> MLP drops more — mlp drives the fracture <<<")

        # ================================================================
        # Exp2: Per-head analysis at fracture layers
        # ================================================================
        log("\n" + "="*70)
        log("Exp2: Per-Head Analysis at Fracture Layers")
        log(f"Fracture layer: L{fracture_layer}, n_heads={n_heads}, d_head={d_head}")
        log("="*70)

        for li in sorted(head_collect_layers):
            if li >= n_layers:
                continue

            # Get W_o for this layer
            lw = get_layer_weights(layers_list[li], d_model, mlp_type)
            W_o = lw.W_o  # [d_model, d_model]

            # Compute per-head centroids
            cat_head_centroids = {}  # cat -> {h: centroid_vector}

            for cat in main_cats:
                words = CATEGORIES_13[cat][:20]
                head_outputs = defaultdict(list)

                for w in words:
                    concat = word_head_concat.get(w, {}).get(li)
                    if concat is None:
                        continue
                    # concat shape: [n_heads * d_head] = [d_model]
                    for h in range(n_heads):
                        h_out = concat[h*d_head:(h+1)*d_head]  # [d_head]
                        W_o_h = W_o[:, h*d_head:(h+1)*d_head]  # [d_model, d_head]
                        h_contribution = W_o_h @ h_out  # [d_model]
                        head_outputs[h].append(h_contribution)

                if len(next(iter(head_outputs.values()), [])) >= 5:
                    cat_head_centroids[cat] = {}
                    for h in range(n_heads):
                        if len(head_outputs[h]) >= 5:
                            cat_head_centroids[cat][h] = np.mean(head_outputs[h], axis=0)

            cats_with_heads = [c for c in main_cats if c in cat_head_centroids]
            if len(cats_with_heads) < 3:
                continue

            marker = " *** FRACTURE ***" if li == fracture_layer else ""
            log(f"\n--- L{li}: Per-head analysis ---{marker}")

            # For each head: norm and category differentiation
            head_norms = {}
            head_cat_diff = {}
            head_pc0_loadings = {}

            for h in range(n_heads):
                h_vecs = []
                h_cats = []
                for cat in cats_with_heads:
                    if h in cat_head_centroids[cat]:
                        h_vecs.append(cat_head_centroids[cat][h])
                        h_cats.append(cat)

                if len(h_vecs) < 3:
                    continue

                norms = [np.linalg.norm(v) for v in h_vecs]
                mean_norm = np.mean(norms)
                head_norms[h] = mean_norm

                # Category differentiation
                cos_vals = []
                for j1 in range(len(h_vecs)):
                    for j2 in range(j1+1, len(h_vecs)):
                        v1, v2 = h_vecs[j1], h_vecs[j2]
                        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                        if n1 > 1e-8 and n2 > 1e-8:
                            cos_vals.append(np.dot(v1, v2) / (n1 * n2))

                avg_cos = np.mean(cos_vals) if cos_vals else 0
                head_cat_diff[h] = 1 - avg_cos

                # PC0 of head's category outputs
                h_matrix = np.array(h_vecs)
                h_centered = h_matrix - h_matrix.mean(axis=0)
                try:
                    _U, S, _ = np.linalg.svd(h_centered, full_matrices=False)
                    u0 = _U[:, 0]
                    # Top-3 loadings
                    sorted_idx = np.argsort(np.abs(u0))[::-1]
                    top3 = [(h_cats[j], float(u0[j])) for j in sorted_idx[:3]]
                    head_pc0_loadings[h] = top3
                except:
                    pass

            # Sort heads
            heads_by_norm = sorted(head_norms.items(), key=lambda x: x[1], reverse=True)
            heads_by_diff = sorted(head_cat_diff.items(), key=lambda x: x[1], reverse=True)

            log(f"  Top-10 heads by output norm:")
            for h, norm in heads_by_norm[:10]:
                diff = head_cat_diff.get(h, 0)
                log(f"    Head {h:2d}: norm={norm:.2f}, cat_diff={diff:.4f}")

            log(f"  Top-10 heads by category differentiation (1 - avg_cos):")
            for h, diff in heads_by_diff[:10]:
                norm = head_norms.get(h, 0)
                pc0 = head_pc0_loadings.get(h, [])
                pc0_str = ", ".join([f"{c}({v:+.2f})" for c, v in pc0[:2]])
                log(f"    Head {h:2d}: cat_diff={diff:.4f}, norm={norm:.2f}, PC0=[{pc0_str}]")

            # For fracture layer: detailed analysis of top differentiated heads
            if li == fracture_layer:
                log(f"\n  === Fracture layer detailed head analysis ===")
                for h, diff in heads_by_diff[:5]:
                    h_vecs = []
                    h_cats = []
                    for cat in cats_with_heads:
                        if h in cat_head_centroids[cat]:
                            h_vecs.append(cat_head_centroids[cat][h])
                            h_cats.append(cat)

                    if len(h_vecs) >= 3:
                        h_matrix = np.array(h_vecs)
                        h_centered = h_matrix - h_matrix.mean(axis=0)
                        _U, S, _ = np.linalg.svd(h_centered, full_matrices=False)
                        u0 = _U[:, 0]
                        sorted_idx = np.argsort(np.abs(u0))[::-1]
                        log(f"    Head {h} PC0 loadings (all categories):")
                        for j in sorted_idx:
                            cat = h_cats[j]
                            sc = SUPERCLASS_MAP[cat]
                            log(f"      {cat:>12s} [{sc[:4]}]: u0={u0[j]:+.4f}")

        # ================================================================
        # Exp3: Head direction change at fracture layer
        # ================================================================
        log("\n" + "="*70)
        log("Exp3: Head Direction Change at Fracture Layer")
        log("="*70)

        li_prev = fracture_layer - 1
        li_curr = fracture_layer

        if li_prev >= 0 and li_prev in head_collect_layers and li_curr in head_collect_layers:
            lw_prev = get_layer_weights(layers_list[li_prev], d_model, mlp_type)
            lw_curr = get_layer_weights(layers_list[li_curr], d_model, mlp_type)
            W_o_prev = lw_prev.W_o
            W_o_curr = lw_curr.W_o

            log(f"\n--- Head direction change: L{li_prev} → L{li_curr} ---")

            # Compute mean direction for each head at each layer
            head_dir_prev = {}
            head_dir_curr = {}

            for li, W_o_l, storage in [(li_prev, W_o_prev, head_dir_prev),
                                        (li_curr, W_o_curr, head_dir_curr)]:
                for h in range(n_heads):
                    all_h_vecs = []
                    for cat in main_cats:
                        words = CATEGORIES_13[cat][:20]
                        for w in words:
                            concat = word_head_concat.get(w, {}).get(li)
                            if concat is None:
                                continue
                            h_out = concat[h*d_head:(h+1)*d_head]
                            W_o_h = W_o_l[:, h*d_head:(h+1)*d_head]
                            h_contribution = W_o_h @ h_out
                            all_h_vecs.append(h_contribution)

                    if all_h_vecs:
                        mean_vec = np.mean(all_h_vecs, axis=0)
                        norm = np.linalg.norm(mean_vec)
                        if norm > 1e-8:
                            storage[h] = mean_vec / norm

            # Compute direction change for each head
            head_changes = {}
            for h in range(n_heads):
                if h in head_dir_prev and h in head_dir_curr:
                    cos_val = np.dot(head_dir_prev[h], head_dir_curr[h])
                    head_changes[h] = cos_val

            if head_changes:
                sorted_changes = sorted(head_changes.items(), key=lambda x: x[1])
                log(f"  Heads with LARGEST direction change (lowest cos):")
                for h, cos_val in sorted_changes[:10]:
                    log(f"    Head {h:2d}: cos(L{li_prev},L{li_curr})={cos_val:.4f}")

                log(f"  Heads with SMALLEST direction change (highest cos):")
                for h, cos_val in sorted_changes[-5:]:
                    log(f"    Head {h:2d}: cos(L{li_prev},L{li_curr})={cos_val:.4f}")

                # Also: per-head norm change
                head_norm_prev = {}
                head_norm_curr = {}
                for li, W_o_l, storage in [(li_prev, W_o_prev, head_norm_prev),
                                            (li_curr, W_o_curr, head_norm_curr)]:
                    for h in range(n_heads):
                        all_h_norms = []
                        for cat in main_cats:
                            words = CATEGORIES_13[cat][:20]
                            for w in words:
                                concat = word_head_concat.get(w, {}).get(li)
                                if concat is None:
                                    continue
                                h_out = concat[h*d_head:(h+1)*d_head]
                                W_o_h = W_o_l[:, h*d_head:(h+1)*d_head]
                                h_contribution = W_o_h @ h_out
                                all_h_norms.append(np.linalg.norm(h_contribution))
                        if all_h_norms:
                            storage[h] = np.mean(all_h_norms)

                norm_changes = {}
                for h in range(n_heads):
                    if h in head_norm_prev and h in head_norm_curr:
                        if head_norm_prev[h] > 1e-8:
                            norm_changes[h] = head_norm_curr[h] / head_norm_prev[h]

                if norm_changes:
                    sorted_norm = sorted(norm_changes.items(), key=lambda x: x[1], reverse=True)
                    log(f"\n  Heads with LARGEST norm increase (curr/prev ratio):")
                    for h, ratio in sorted_norm[:10]:
                        dir_cos = head_changes.get(h, 0)
                        log(f"    Head {h:2d}: norm_ratio={ratio:.2f}, dir_cos={dir_cos:.4f}")

                    sorted_norm_down = sorted(norm_changes.items(), key=lambda x: x[1])
                    log(f"\n  Heads with LARGEST norm decrease (curr/prev ratio):")
                    for h, ratio in sorted_norm_down[:10]:
                        dir_cos = head_changes.get(h, 0)
                        log(f"    Head {h:2d}: norm_ratio={ratio:.2f}, dir_cos={dir_cos:.4f}")

        # ===== Exp4: Extended layer analysis — full layer range =====
        log("\n" + "="*70)
        log("Exp4: Extended Attn vs MLP Energy Ratio (all layers)")
        log("="*70)

        for li in range(n_layers):
            cats_present = [c for c in main_cats if c in attn_centroids[li] and c in mlp_centroids[li]]
            if len(cats_present) < 3:
                continue

            attn_disp = np.array([attn_centroids[li][c] for c in cats_present])
            mlp_disp = np.array([mlp_centroids[li][c] for c in cats_present])
            total_disp = attn_disp + mlp_disp

            total_energy = np.sum(total_disp ** 2)
            attn_energy = np.sum(attn_disp ** 2)
            mlp_energy = np.sum(mlp_disp ** 2)

            attn_ratio = attn_energy / total_energy if total_energy > 0 else 0
            mlp_ratio = mlp_energy / total_energy if total_energy > 0 else 0

            attn_avg_cos = compute_avg_cos(attn_disp)
            mlp_avg_cos = compute_avg_cos(mlp_disp)
            total_avg_cos = compute_avg_cos(total_disp)

            marker = " ***" if li == fracture_layer else ""
            log(f"  L{li:2d}: attn_ratio={attn_ratio:.4f}, mlp_ratio={mlp_ratio:.4f}, "
                f"attn_cos={attn_avg_cos:.4f}, mlp_cos={mlp_avg_cos:.4f}, total_cos={total_avg_cos:.4f}{marker}")

        # ===== Save results =====
        results_dir = f"d:/Ai2050/TransformerLens-Project/results/causal_fiber/{model_name}_cclxxxxiv"
        os.makedirs(results_dir, exist_ok=True)

        with open(os.path.join(results_dir, "exp1_summary.json"), 'w') as f:
            json.dump(exp1_summary, f, indent=2)

        log(f"Results saved to {results_dir}")

        # Cleanup
        del word_attn, word_mlp, word_head_concat
        del attn_centroids, mlp_centroids, total_centroids
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
