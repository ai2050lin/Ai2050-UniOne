"""
CCLXXXXII(292): 运动自由度与位移子空间分析
基于CCLXXXXI发现: 速度U形, 角变化在L0-L2最大(158°), body_part方向最稳定
目标: 13个质心的运动到底用了几个"自由度"?

Exp1: 每个层段位移向量的SVD → 运动内在维度(几个奇异值解释90%/95%/99%方差)
Exp2: 超类平均位移方向是否解释大部分运动(投影重建率)
Exp3: body_part位移的正交性 → body_part是否独占一个运动维度
"""
import sys, os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxxxii_log.txt"

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        import time
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()

log("=== CCLXXXXII Script started ===")

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

        # ===== 计算每段位移向量 =====
        displacement = {}  # cat -> {(li1,li2): vector}
        for cat in main_cats:
            displacement[cat] = {}
            for i in range(len(sample_layers) - 1):
                li1, li2 = sample_layers[i], sample_layers[i+1]
                if cat in layer_cat_centroids[li1] and cat in layer_cat_centroids[li2]:
                    displacement[cat][(li1, li2)] = layer_cat_centroids[li2][cat] - layer_cat_centroids[li1][cat]

        # ================================================================
        # Exp1: 运动内在维度 — 位移向量的SVD
        # ================================================================
        log("\n" + "="*70)
        log("Exp1: Intrinsic Dimension of Centroid Displacements (SVD)")
        log("="*70)

        for i in range(len(sample_layers) - 1):
            li1, li2 = sample_layers[i], sample_layers[i+1]
            # 收集所有类别的位移向量
            disp_vectors = []
            disp_cats = []
            for cat in main_cats:
                if (li1, li2) in displacement[cat]:
                    disp_vectors.append(displacement[cat][(li1, li2)])
                    disp_cats.append(cat)

            if len(disp_vectors) < 3:
                continue

            disp_matrix = np.array(disp_vectors)  # [n_cats, d_model]
            n_cats_seg = disp_matrix.shape[0]

            # 去均值
            disp_mean = disp_matrix.mean(axis=0)
            disp_centered = disp_matrix - disp_mean

            # SVD
            _U, S, Vt = np.linalg.svd(disp_centered, full_matrices=False)

            # 计算累积方差解释比
            total_var = np.sum(S**2)
            cumvar = np.cumsum(S**2) / total_var

            # 找达到各阈值的维度数
            dim_90 = np.searchsorted(cumvar, 0.90) + 1
            dim_95 = np.searchsorted(cumvar, 0.95) + 1
            dim_99 = np.searchsorted(cumvar, 0.99) + 1

            log(f"\n--- L{li1}-L{li2}: {n_cats_seg} categories ---")
            log(f"  Top-10 singular values: {[f'{s:.2f}' for s in S[:10]]}")
            log(f"  Cumulative variance: 90% at dim-{dim_90}, 95% at dim-{dim_95}, 99% at dim-{dim_99}")
            log(f"  S[0]/S[1] = {S[0]/S[1]:.3f}, S[0]/S_sum = {S[0]**2/total_var:.4f}")
            log(f"  First 5 cumvar: {[f'{c:.4f}' for c in cumvar[:5]]}")

            # 第一个左奇异向量: 哪些类别贡献最大?
            u0 = _U[:, 0]  # [n_cats]
            log(f"  PC0 loadings (top contributors):")
            sorted_idx = np.argsort(np.abs(u0))[::-1]
            for j in sorted_idx[:5]:
                cat = disp_cats[j]
                sc = SUPERCLASS_MAP[cat]
                log(f"    {cat:>12s} [{sc[:4]}]: u0={u0[j]:+.4f} (|u0|={abs(u0[j]):.4f})")

            # 第二个左奇异向量
            if len(S) > 1:
                u1 = _U[:, 1]
                log(f"  PC1 loadings (top contributors):")
                sorted_idx1 = np.argsort(np.abs(u1))[::-1]
                for j in sorted_idx1[:5]:
                    cat = disp_cats[j]
                    sc = SUPERCLASS_MAP[cat]
                    log(f"    {cat:>12s} [{sc[:4]}]: u1={u1[j]:+.4f} (|u1|={abs(u1[j]):.4f})")

        # ===== 汇总: 各层段的内在维度 =====
        log("\n--- Summary: Intrinsic dimensions across layers ---")
        log(f"  {'Layer':>10s}  {'dim_90':>6s}  {'dim_95':>6s}  {'dim_99':>6s}  {'S[0]/total':>10s}  {'S[0]/S[1]':>10s}")
        for i in range(len(sample_layers) - 1):
            li1, li2 = sample_layers[i], sample_layers[i+1]
            disp_vectors = []
            for cat in main_cats:
                if (li1, li2) in displacement[cat]:
                    disp_vectors.append(displacement[cat][(li1, li2)])
            if len(disp_vectors) < 3:
                continue
            disp_matrix = np.array(disp_vectors)
            disp_centered = disp_matrix - disp_matrix.mean(axis=0)
            _U, S, Vt = np.linalg.svd(disp_centered, full_matrices=False)
            total_var = np.sum(S**2)
            cumvar = np.cumsum(S**2) / total_var
            dim_90 = np.searchsorted(cumvar, 0.90) + 1
            dim_95 = np.searchsorted(cumvar, 0.95) + 1
            dim_99 = np.searchsorted(cumvar, 0.99) + 1
            s0_ratio = S[0]**2 / total_var
            s0_s1 = S[0]/S[1] if len(S) > 1 else float('inf')
            log(f"  L{li1:>4d}-L{li2:<4d}  {dim_90:>6d}  {dim_95:>6d}  {dim_99:>6d}  {s0_ratio:>10.4f}  {s0_s1:>10.3f}")

        # ================================================================
        # Exp2: 超类平均位移方向的重建率
        # ================================================================
        log("\n" + "="*70)
        log("Exp2: Superclass Mean Displacement Reconstruction")
        log("="*70)

        # 计算4个超类的平均位移方向
        superclass_names = ["animate", "plant", "body", "artifact"]
        superclass_cats_map = {
            "animate": sorted(ANIMATE_CATS),
            "plant": sorted(PLANT_CATS),
            "body": sorted(BODY_CATS),
            "artifact": sorted(ARTIFACT_CATS),
        }

        for i in range(len(sample_layers) - 1):
            li1, li2 = sample_layers[i], sample_layers[i+1]

            # 计算每个超类的平均位移
            sc_mean_disp = {}
            for sc_name in superclass_names:
                sc_cats = superclass_cats_map[sc_name]
                disps = [displacement[cat][(li1, li2)] for cat in sc_cats
                         if (li1, li2) in displacement.get(cat, {})]
                if len(disps) >= 1:
                    sc_mean_disp[sc_name] = np.mean(disps, axis=0)

            if len(sc_mean_disp) < 3:
                continue

            # 构建超类方向矩阵
            sc_matrix = np.array([sc_mean_disp[sc] for sc in superclass_names
                                  if sc in sc_mean_disp])  # [n_sc, d_model]

            # 对每个类别的位移，计算在超类方向上的投影重建率
            total_norm_sq = 0
            residual_norm_sq = 0
            n_reconstructed = 0

            cat_recon = {}  # cat -> reconstruction_rate
            for cat in main_cats:
                if (li1, li2) not in displacement.get(cat, {}):
                    continue
                v = displacement[cat][(li1, li2)]
                v_norm_sq = np.dot(v, v)
                if v_norm_sq < 1e-10:
                    continue

                # 投影到超类子空间
                # 用最小二乘: v ≈ sc_matrix.T @ alpha
                # alpha = (sc_matrix @ sc_matrix.T)^{-1} @ sc_matrix @ v
                try:
                    alpha = np.linalg.lstsq(sc_matrix.T, v, rcond=None)[0]
                    v_recon = sc_matrix.T @ alpha
                    residual = v - v_recon
                    recon_rate = 1 - np.dot(residual, residual) / v_norm_sq
                except:
                    recon_rate = 0

                cat_recon[cat] = recon_rate
                total_norm_sq += v_norm_sq
                residual_norm_sq += np.dot(v - sc_matrix.T @ np.linalg.lstsq(sc_matrix.T, v, rcond=None)[0],
                                            v - sc_matrix.T @ np.linalg.lstsq(sc_matrix.T, v, rcond=None)[0])
                n_reconstructed += 1

            if n_reconstructed > 0:
                overall_recon = 1 - residual_norm_sq / total_norm_sq if total_norm_sq > 0 else 0
                log(f"\n--- L{li1}-L{li2}: Superclass reconstruction ---")
                log(f"  Overall reconstruction rate: {overall_recon:.4f}")
                for cat in sorted(cat_recon.keys()):
                    sc = SUPERCLASS_MAP[cat]
                    log(f"  {cat:>12s} [{sc[:4]}]: recon={cat_recon[cat]:.4f}")

        # ================================================================
        # Exp3: body_part位移的正交性分析
        # ================================================================
        log("\n" + "="*70)
        log("Exp3: Body_part Displacement Orthogonality")
        log("="*70)

        for i in range(len(sample_layers) - 1):
            li1, li2 = sample_layers[i], sample_layers[i+1]
            if "body_part" not in displacement or (li1, li2) not in displacement["body_part"]:
                continue

            bp_disp = displacement["body_part"][(li1, li2)]
            bp_norm = np.linalg.norm(bp_disp)
            if bp_norm < 1e-8:
                continue

            log(f"\n--- L{li1}-L{li2}: body_part vs others ---")
            log(f"  body_part displacement norm: {bp_norm:.4f}")

            # body_part位移与其他类别位移的cos similarity
            cos_sims = []
            for cat in main_cats:
                if cat == "body_part":
                    continue
                if (li1, li2) not in displacement.get(cat, {}):
                    continue
                other_disp = displacement[cat][(li1, li2)]
                other_norm = np.linalg.norm(other_disp)
                if other_norm < 1e-8:
                    continue
                cos_sim = np.dot(bp_disp, other_disp) / (bp_norm * other_norm)
                cos_sims.append((cat, cos_sim))

            cos_sims.sort(key=lambda x: abs(x[1]))
            log(f"  Most orthogonal to body_part:")
            for cat, cos_val in cos_sims[:5]:
                sc = SUPERCLASS_MAP[cat]
                log(f"    {cat:>12s} [{sc[:4]}]: cos={cos_val:+.4f}")

            log(f"  Most aligned with body_part:")
            for cat, cos_val in sorted(cos_sims, key=lambda x: abs(x[1]), reverse=True)[:5]:
                sc = SUPERCLASS_MAP[cat]
                log(f"    {cat:>12s} [{sc[:4]}]: cos={cos_val:+.4f}")

            # body_part位移与Exp1的PC0/PC1方向的对齐
            # 重新计算该层段的SVD
            disp_vectors = []
            disp_cats_list = []
            for cat in main_cats:
                if (li1, li2) in displacement.get(cat, {}):
                    disp_vectors.append(displacement[cat][(li1, li2)])
                    disp_cats_list.append(cat)
            if len(disp_vectors) >= 3:
                disp_matrix = np.array(disp_vectors)
                disp_centered = disp_matrix - disp_matrix.mean(axis=0)
                _U_seg, S_seg, Vt_seg = np.linalg.svd(disp_centered, full_matrices=False)

                # PC0方向
                pc0 = Vt_seg[0]
                cos_pc0 = np.dot(bp_disp, pc0) / (bp_norm * np.linalg.norm(pc0))
                log(f"  body_part alignment with displacement-PC0: cos={cos_pc0:+.4f}")
                if len(Vt_seg) > 1:
                    pc1 = Vt_seg[1]
                    cos_pc1 = np.dot(bp_disp, pc1) / (bp_norm * np.linalg.norm(pc1))
                    log(f"  body_part alignment with displacement-PC1: cos={cos_pc1:+.4f}")

        # ===== 保存结果 =====
        results_dir = f"d:/Ai2050/TransformerLens-Project/results/causal_fiber/{model_name}_cclxxxxii"
        os.makedirs(results_dir, exist_ok=True)

        # 保存各层段的内在维度
        dim_data = {}
        for i in range(len(sample_layers) - 1):
            li1, li2 = sample_layers[i], sample_layers[i+1]
            disp_vectors = []
            for cat in main_cats:
                if (li1, li2) in displacement[cat]:
                    disp_vectors.append(displacement[cat][(li1, li2)])
            if len(disp_vectors) < 3:
                continue
            disp_matrix = np.array(disp_vectors)
            disp_centered = disp_matrix - disp_matrix.mean(axis=0)
            _U, S, Vt = np.linalg.svd(disp_centered, full_matrices=False)
            total_var = np.sum(S**2)
            cumvar = np.cumsum(S**2) / total_var
            dim_data[f"L{li1}-L{li2}"] = {
                "singular_values": S[:15].tolist(),
                "cumulative_variance": cumvar[:15].tolist(),
                "dim_90": int(np.searchsorted(cumvar, 0.90) + 1),
                "dim_95": int(np.searchsorted(cumvar, 0.95) + 1),
                "dim_99": int(np.searchsorted(cumvar, 0.99) + 1),
            }

        with open(os.path.join(results_dir, "intrinsic_dims.json"), 'w') as f:
            json.dump(dim_data, f, indent=2)

        log(f"Results saved to {results_dir}")

        # ===== 清理 =====
        del word_layer_acts, layer_cat_centroids, displacement
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
        time.sleep(3)

    log("=== ALL MODELS DONE ===")
