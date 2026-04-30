"""
CCLXXXXI(291): 层间几何稳定性与质心运动学分析
基于CCLXXXX发现: 三阶段运动模式(超类同步→类别分道扬镳→超类恢复)
目标: 纯数据积累, 不做理论分析

Exp1: 层间距离相关性矩阵 — 每对层间78对类别距离的Pearson相关
      揭示几何结构的"相位边界"
Exp2: 质心速度与加速度 — 相邻层质心位移大小、方向变化
      揭示类别何时"急转"、"减速"
Exp3: 逐类别关键层 — 每个类别最大速度/加速度/角变化的层号
      揭示各类别的"转折点"是否一致
"""
import sys, os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxxxi_log.txt"

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        import time
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()

log("=== CCLXXXXI Script started ===")

import json, time, gc, traceback
from pathlib import Path
import numpy as np
from collections import defaultdict
from itertools import combinations
from scipy.stats import pearsonr

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

        # ===== 采样层 (用更密的采样以捕捉运动细节) =====
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
        layer_cat_centroids = {}

        for li in sample_layers:
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

        # ===== 计算78对类别间距离的跨层数据 =====
        all_pairs = list(combinations(main_cats, 2))
        pair_dist_matrix = {}  # (cat1,cat2) -> {li: distance}

        for cat1, cat2 in all_pairs:
            pair_dist_matrix[(cat1, cat2)] = {}
            for li in sample_layers:
                if cat1 in layer_cat_centroids[li] and cat2 in layer_cat_centroids[li]:
                    c1 = layer_cat_centroids[li][cat1]
                    c2 = layer_cat_centroids[li][cat2]
                    pair_dist_matrix[(cat1, cat2)][li] = np.linalg.norm(c1 - c2)

        # ================================================================
        # Exp1: 层间距离相关性矩阵
        # ================================================================
        log("\n" + "="*70)
        log("Exp1: Cross-Layer Pairwise Distance Correlation Matrix")
        log("="*70)

        # 构建距离向量: 每层一个78维向量
        layer_dist_vectors = {}
        for li in sample_layers:
            dist_vec = []
            for (c1, c2) in all_pairs:
                if (c1, c2) in pair_dist_matrix and li in pair_dist_matrix[(c1, c2)]:
                    dist_vec.append(pair_dist_matrix[(c1, c2)][li])
                else:
                    dist_vec.append(np.nan)
            layer_dist_vectors[li] = np.array(dist_vec)

        # 计算层间Pearson相关
        n_sample = len(sample_layers)
        corr_matrix = np.zeros((n_sample, n_sample))

        log(f"\nDistance vector length: {len(all_pairs)}")
        log(f"Number of sampled layers: {n_sample}")

        for i, li1 in enumerate(sample_layers):
            for j, li2 in enumerate(sample_layers):
                v1 = layer_dist_vectors[li1]
                v2 = layer_dist_vectors[li2]
                # 去除nan
                valid = ~(np.isnan(v1) | np.isnan(v2))
                if valid.sum() > 10:
                    r, p = pearsonr(v1[valid], v2[valid])
                    corr_matrix[i, j] = r
                else:
                    corr_matrix[i, j] = np.nan

        # 打印相关矩阵 (只打印关键行)
        log("\n--- Correlation Matrix (selected rows) ---")
        log(f"{'':>6s}" + "".join(f"  L{li:>3d}" for li in sample_layers))
        for i, li1 in enumerate(sample_layers):
            row = f"L{li1:>4d}"
            for j, li2 in enumerate(sample_layers):
                row += f"  {corr_matrix[i,j]:5.3f}"
            log(row)

        # 相邻层间的相关 — 找"断裂点"
        log("\n--- Adjacent-layer correlation (phase boundary detection) ---")
        adj_corrs = []
        for i in range(len(sample_layers) - 1):
            li1, li2 = sample_layers[i], sample_layers[i+1]
            r = corr_matrix[i, i+1]
            gap = sample_layers[i+1] - sample_layers[i]
            adj_corrs.append((li1, li2, r, gap))
            log(f"  L{li1}-L{li2}: r={r:.4f} (gap={gap})")

        # 找相关最低的相邻对
        adj_corrs_sorted = sorted(adj_corrs, key=lambda x: x[2])
        log("\n--- Lowest adjacent correlations (potential phase boundaries) ---")
        for li1, li2, r, gap in adj_corrs_sorted[:5]:
            log(f"  L{li1}-L{li2}: r={r:.4f}")

        # L0与其他层的相关衰减
        log("\n--- Correlation with L0 (how fast geometry changes from initial) ---")
        for j, li in enumerate(sample_layers):
            log(f"  corr(L0, L{li}) = {corr_matrix[0, j]:.4f}")

        # 中间层与其他层的相关
        mid_idx = len(sample_layers) // 2
        mid_li = sample_layers[mid_idx]
        log(f"\n--- Correlation with L{mid_li} (middle layer) ---")
        for j, li in enumerate(sample_layers):
            log(f"  corr(L{mid_li}, L{li}) = {corr_matrix[mid_idx, j]:.4f}")

        # ================================================================
        # Exp2: 质心速度与加速度
        # ================================================================
        log("\n" + "="*70)
        log("Exp2: Centroid Velocity and Acceleration")
        log("="*70)

        # 质心位移向量
        centroid_displacement = {}  # cat -> {(li1,li2): displacement_vector}
        centroid_speed = {}         # cat -> {(li1,li2): speed}

        for cat in main_cats:
            centroid_displacement[cat] = {}
            centroid_speed[cat] = {}
            for i in range(len(sample_layers) - 1):
                li1, li2 = sample_layers[i], sample_layers[i+1]
                if cat in layer_cat_centroids[li1] and cat in layer_cat_centroids[li2]:
                    disp = layer_cat_centroids[li2][cat] - layer_cat_centroids[li1][cat]
                    centroid_displacement[cat][(li1, li2)] = disp
                    gap = li2 - li1
                    centroid_speed[cat][(li1, li2)] = np.linalg.norm(disp) / gap

        # 全局平均速度 (所有类别)
        log("\n--- Average centroid speed across all categories ---")
        for i in range(len(sample_layers) - 1):
            li1, li2 = sample_layers[i], sample_layers[i+1]
            speeds = [centroid_speed[cat][(li1, li2)] for cat in main_cats
                      if (li1, li2) in centroid_speed[cat]]
            if speeds:
                avg_speed = np.mean(speeds)
                max_speed = np.max(speeds)
                min_speed = np.min(speeds)
                # 找最快和最慢的类别
                fastest_cat = max(main_cats, key=lambda c: centroid_speed[c].get((li1, li2), 0))
                slowest_cat = min(main_cats, key=lambda c: centroid_speed[c].get((li1, li2), 1e10))
                log(f"  L{li1}-L{li2}: avg={avg_speed:.4f} max={max_speed:.4f}({fastest_cat}) "
                    f"min={min_speed:.4f}({slowest_cat})")

        # 逐类别速度曲线
        log("\n--- Per-category speed profile ---")
        for cat in main_cats:
            sc = get_superclass(cat)
            row = f"  {cat:>12s} [{sc[:4]}]: "
            for i in range(len(sample_layers) - 1):
                li1, li2 = sample_layers[i], sample_layers[i+1]
                if (li1, li2) in centroid_speed[cat]:
                    row += f"L{li1}={centroid_speed[cat][(li1,li2)]:.3f} "
            log(row)

        # 加速度 = 速度变化
        log("\n--- Per-category acceleration (speed change) ---")
        centroid_accel = {}  # cat -> {(li1,li2,li3): accel}
        for cat in main_cats:
            centroid_accel[cat] = {}
            speed_keys = sorted(centroid_speed[cat].keys())
            for k in range(len(speed_keys) - 1):
                (li1, li2) = speed_keys[k]
                (li3, li4) = speed_keys[k+1]
                s1 = centroid_speed[cat][(li1, li2)]
                s2 = centroid_speed[cat][(li3, li4)]
                # 归一化加速度
                accel = (s2 - s1) / ((li3 + li4 - li1 - li2) / 2)  # per layer
                centroid_accel[cat][(li1, li2, li3, li4)] = accel

        for cat in main_cats:
            sc = get_superclass(cat)
            row = f"  {cat:>12s} [{sc[:4]}]: "
            for key in sorted(centroid_accel[cat].keys())[:8]:
                accel = centroid_accel[cat][key]
                row += f"L{key[0]}-{key[2]}={accel:.4f} "
            log(row)

        # 角变化 — 位移方向的转折
        log("\n--- Per-category angular change (direction change in degrees) ---")
        centroid_angle_change = {}  # cat -> {(li1,li2,li3,li4): angle_deg}

        for cat in main_cats:
            centroid_angle_change[cat] = {}
            disp_keys = sorted(centroid_displacement[cat].keys())
            for k in range(len(disp_keys) - 1):
                (li1, li2) = disp_keys[k]
                (li3, li4) = disp_keys[k+1]
                d1 = centroid_displacement[cat][(li1, li2)]
                d2 = centroid_displacement[cat][(li3, li4)]
                n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
                if n1 > 1e-8 and n2 > 1e-8:
                    cos_angle = np.clip(np.dot(d1, d2) / (n1 * n2), -1, 1)
                    angle_deg = np.degrees(np.arccos(cos_angle))
                    centroid_angle_change[cat][(li1, li2, li3, li4)] = angle_deg

        for cat in main_cats:
            sc = get_superclass(cat)
            row = f"  {cat:>12s} [{sc[:4]}]: "
            for key in sorted(centroid_angle_change[cat].keys()):
                angle = centroid_angle_change[cat][key]
                row += f"L{key[0]}-{key[2]}={angle:.1f}° "
            log(row)

        # ================================================================
        # Exp3: 逐类别关键层
        # ================================================================
        log("\n" + "="*70)
        log("Exp3: Per-Category Critical Layers")
        log("="*70)

        log("\n--- Max speed layer ---")
        for cat in main_cats:
            sc = get_superclass(cat)
            if centroid_speed[cat]:
                max_key = max(centroid_speed[cat], key=centroid_speed[cat].get)
                max_val = centroid_speed[cat][max_key]
                log(f"  {cat:>12s} [{sc[:4]}]: max_speed={max_val:.4f} at L{max_key[0]}-L{max_key[1]}")
            else:
                log(f"  {cat:>12s} [{sc[:4]}]: no data")

        log("\n--- Min speed layer ---")
        for cat in main_cats:
            sc = get_superclass(cat)
            if centroid_speed[cat]:
                min_key = min(centroid_speed[cat], key=centroid_speed[cat].get)
                min_val = centroid_speed[cat][min_key]
                log(f"  {cat:>12s} [{sc[:4]}]: min_speed={min_val:.4f} at L{min_key[0]}-L{min_key[1]}")
            else:
                log(f"  {cat:>12s} [{sc[:4]}]: no data")

        log("\n--- Max angular change layer (direction reversal) ---")
        for cat in main_cats:
            sc = get_superclass(cat)
            if centroid_angle_change[cat]:
                max_key = max(centroid_angle_change[cat], key=centroid_angle_change[cat].get)
                max_val = centroid_angle_change[cat][max_key]
                log(f"  {cat:>12s} [{sc[:4]}]: max_angle={max_val:.1f}° at L{max_key[0]}-L{max_key[2]}")
            else:
                log(f"  {cat:>12s} [{sc[:4]}]: no data")

        # ===== 超类级别的速度聚合 =====
        log("\n--- Superclass average speed ---")
        for sc_name, sc_cats in [("animate", ANIMATE_CATS), ("plant", PLANT_CATS),
                                  ("body", BODY_CATS), ("artifact", ARTIFACT_CATS)]:
            row = f"  [{sc_name:>9s}]: "
            for i in range(len(sample_layers) - 1):
                li1, li2 = sample_layers[i], sample_layers[i+1]
                speeds = [centroid_speed[cat][(li1, li2)] for cat in sc_cats
                          if (li1, li2) in centroid_speed.get(cat, {})]
                if speeds:
                    row += f"L{li1}={np.mean(speeds):.3f} "
            log(row)

        # ===== 位移方向与PC方向的对齐 =====
        log("\n--- Displacement-PC alignment (cosine of displacement with top PCs) ---")
        # 用中间层的全局SVD方向
        mid_li = sample_layers[len(sample_layers)//2]
        all_vecs_mid = []
        for cat in main_cats:
            words = CATEGORIES_13[cat][:20]
            for w in words:
                if w in word_layer_acts and mid_li in word_layer_acts[w]:
                    all_vecs_mid.append(word_layer_acts[w][mid_li])
        all_vecs_mid = np.array(all_vecs_mid)
        global_mean_mid = all_vecs_mid.mean(axis=0)
        _U, S, Vt = np.linalg.svd(all_vecs_mid - global_mean_mid, full_matrices=False)
        top_pcs = Vt[:5]  # [5, d_model]

        for cat in main_cats:
            sc = get_superclass(cat)
            row = f"  {cat:>12s} [{sc[:4]}]: "
            for i in range(len(sample_layers) - 1):
                li1, li2 = sample_layers[i], sample_layers[i+1]
                if (li1, li2) in centroid_displacement[cat]:
                    disp = centroid_displacement[cat][(li1, li2)]
                    disp_norm = np.linalg.norm(disp)
                    if disp_norm > 1e-8:
                        cos_pc0 = np.dot(disp, top_pcs[0]) / (disp_norm * np.linalg.norm(top_pcs[0]))
                        cos_pc1 = np.dot(disp, top_pcs[1]) / (disp_norm * np.linalg.norm(top_pcs[1]))
                        row += f"L{li1}(PC0={cos_pc0:+.2f},PC1={cos_pc1:+.2f}) "
            log(row)

        # ===== 保存结果 =====
        results_dir = f"d:/Ai2050/TransformerLens-Project/results/causal_fiber/{model_name}_cclxxxxi"
        os.makedirs(results_dir, exist_ok=True)

        # 保存相关矩阵
        np.save(os.path.join(results_dir, "corr_matrix.npy"), corr_matrix)
        np.save(os.path.join(results_dir, "sample_layers.npy"), np.array(sample_layers))

        # 保存速度数据
        speed_data = {}
        for cat in main_cats:
            speed_data[cat] = {f"L{li1}-L{li2}": v for (li1, li2), v in centroid_speed[cat].items()}

        angle_data = {}
        for cat in main_cats:
            angle_data[cat] = {f"L{k[0]}-L{k[2]}": v for k, v in centroid_angle_change[cat].items()}

        with open(os.path.join(results_dir, "kinematics.json"), 'w') as f:
            json.dump({"speed": speed_data, "angle_change": angle_data}, f, indent=2)

        log(f"Results saved to {results_dir}")

        # ===== 清理 =====
        del word_layer_acts, layer_cat_centroids, pair_dist_matrix
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
        # 等待GPU释放
        time.sleep(5)
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        time.sleep(3)

    log("=== ALL MODELS DONE ===")
