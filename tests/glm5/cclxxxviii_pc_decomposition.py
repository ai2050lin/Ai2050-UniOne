"""
CCLXXXVIII: 语义距离的维度分解
基于CCLXXXVII发现: 超类L2-L4结晶, 类别L6-L17结晶, 差距6.5层
目标: 哪些PC编码超类? 哪些PC编码细类? 积累维度级数据

Exp1: 每对类别质心距离的PC分解 — 哪些PC贡献最大?
Exp2: 超类对的PC贡献谱 vs 非超类对的PC贡献谱
Exp3: 每个PC的"类别区分力" — 单个PC能区分多少对类别?
Exp4: 边界词在各PC上的位置 — 哪些PC把tomato推向fruit, 哪些推向vegetable?
Exp5: 跨层PC贡献演化 — 超类PC和类别PC是否在不同层出现?
"""
import sys, os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxxviii_log.txt"

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        import time
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()

log("=== CCLXXXVIII Script started ===")

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

# ===== 13类别 × 20词 (与之前相同) =====
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

# 边界词
BOUNDARY_WORDS = {
    "fruit_veg": ["tomato", "cucumber", "avocado", "eggplant", "pepper"],
    "weapon_tool": ["knife", "axe", "hammer", "scissors", "chisel"],
    "bird_animal": ["penguin", "duck", "goose", "chicken", "turkey"],
    "body_plant": ["trunk", "bark", "crown", "branch", "leaf"],
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

# 超类对(不同超类之间的类别对)
def is_cross_superclass(c1, c2):
    return get_superclass(c1) != get_superclass(c2)

def is_same_superclass(c1, c2):
    return get_superclass(c1) == get_superclass(c2) and c1 != c2


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

        # ===== 收集所有类别词 =====
        all_words, all_cats = [], []
        for cat, words in CATEGORIES_13.items():
            all_words.extend(words[:20])
            all_cats.extend([cat] * 20)
        
        # 添加边界词
        for btype, words in BOUNDARY_WORDS.items():
            for w in words:
                if w not in all_words:
                    all_words.append(w)
                    all_cats.append(f"boundary_{btype}")
        
        n_total = len(all_words)
        log(f"Total words: {n_total}")

        word2cat = {w: c for w, c in zip(all_words, all_cats)}

        # ===== 前向推理 =====
        template = "The {} is"
        word_wdu = {}
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

            u_dict = {}
            for li in range(n_layers):
                key = f"L{li}"
                lw = get_layer_weights(layers_list[li], d_model, mlp_type)
                if lw.W_gate is None or key not in ln_out:
                    continue
                h_input = ln_out[key]
                u = lw.W_up @ h_input
                wdu = lw.W_down @ u
                u_dict[li] = wdu

            word_wdu[word] = u_dict
            if (wi + 1) % 50 == 0 or wi == 0:
                log(f"  Word {wi+1}/{n_total} ({time.time()-t0:.0f}s)")

        log(f"Data collection done ({time.time()-t0:.0f}s)")

        from scipy.sparse.linalg import svds

        main_cats = sorted(CATEGORIES_13.keys())
        
        # ===== 选择层 — 覆盖超类结晶层(L2-L4)和类别结晶层(L6-L17) =====
        if n_layers <= 30:
            sample_layers = [0, 2, 4, 6, 8, 10, n_layers // 2, n_layers - 2, n_layers - 1]
        else:
            sample_layers = [0, 2, 4, 6, 8, 10, 14, 18, n_layers // 2, 
                           3 * n_layers // 4, n_layers - 2, n_layers - 1]
        # 去重并排序
        sample_layers = sorted(set(li for li in sample_layers if li < n_layers))
        log(f"Sample layers: {sample_layers}")

        # =================================================================
        # Exp1: 每对类别质心距离的PC分解
        # =================================================================
        log("\n=== Exp1: PC Decomposition of Category Pair Distances ===")
        
        # 选取关键类别对
        key_pairs = [
            ("tool", "weapon"),      # 最近对
            ("fruit", "vegetable"),  # 第二近
            ("animal", "bird"),      # 同超类
            ("clothing", "furniture"), # 第三混淆对
            ("body_part", "animal"), # 最远对之一
            ("body_part", "tool"),   # body_part vs artifact
        ]
        
        # 对每层, 做SVD然后分解每对质心距离
        for li in sample_layers:
            wdu_vecs, wnames = [], []
            for w in all_words:
                if li in word_wdu.get(w, {}):
                    wdu_vecs.append(word_wdu[w][li])
                    wnames.append(w)
            
            if len(wdu_vecs) < 50:
                continue
            
            X = np.array(wdu_vecs, dtype=np.float32)
            X_c = X - X.mean(axis=0)
            k_svd = min(100, len(X_c) - 1, X_c.shape[1] - 1)
            try:
                U, S, Vt = svds(X_c.astype(np.float32), k=k_svd)
                order = np.argsort(S)[::-1]
                U, S, Vt = U[:, order], S[order], Vt[order]
            except:
                continue
            
            # 投影到top PCs
            n_pcs = min(30, len(S))
            proj = X_c @ Vt[:n_pcs].T  # (n, n_pcs)
            
            # 计算各类别质心(在PC空间中)
            centroids = {}
            for cat in main_cats:
                cat_words = CATEGORIES_13[cat][:20]
                cat_idx = [i for i, w in enumerate(wnames) if w in cat_words]
                if len(cat_idx) >= 3:
                    centroids[cat] = proj[cat_idx].mean(axis=0)
            
            if len(centroids) < 10:
                continue
            
            log(f"  L{li}:")
            
            # 对每对关键类别对, 分解质心距离到各PC
            for c1, c2 in key_pairs:
                if c1 not in centroids or c2 not in centroids:
                    continue
                
                diff = centroids[c1] - centroids[c2]  # (n_pcs,)
                total_dist_sq = np.sum(diff ** 2)
                
                if total_dist_sq < 1e-10:
                    continue
                
                # 每个PC的贡献 = diff[pc]^2 / total_dist_sq
                contributions = diff ** 2 / total_dist_sq
                
                # Top-5贡献PC
                top_pcs = np.argsort(contributions)[::-1][:5]
                
                log(f"    {c1:12s}-{c2:12s}: total_dist^2={total_dist_sq:.4f}")
                for pc in top_pcs:
                    log(f"      PC{pc:2d}: {contributions[pc]:.4f} ({contributions[pc]*100:.1f}%), "
                        f"diff={diff[pc]:.4f}")
                
                # 累积贡献
                sorted_contribs = np.sort(contributions)[::-1]
                cumsum = np.cumsum(sorted_contribs)
                log(f"      Cumulative: top3={cumsum[2]:.3f}, top5={cumsum[4]:.3f}, "
                    f"top10={cumsum[min(9,n_pcs-1)]:.3f}")

        # =================================================================
        # Exp2: 超类对 vs 非超类对的PC贡献谱
        # =================================================================
        log("\n=== Exp2: Cross-Superclass vs Same-Superclass PC Spectra ===")
        
        # 选中间层
        mid_li = sample_layers[min(len(sample_layers) // 2, len(sample_layers) - 1)]
        
        for li in [sample_layers[1] if len(sample_layers) > 1 else 0,  # 早期层
                   mid_li]:  # 中间层
            wdu_vecs, wnames = [], []
            for w in all_words:
                if li in word_wdu.get(w, {}):
                    wdu_vecs.append(word_wdu[w][li])
                    wnames.append(w)
            
            if len(wdu_vecs) < 50:
                continue
            
            X = np.array(wdu_vecs, dtype=np.float32)
            X_c = X - X.mean(axis=0)
            k_svd = min(100, len(X_c) - 1, X_c.shape[1] - 1)
            try:
                U, S, Vt = svds(X_c.astype(np.float32), k=k_svd)
                order = np.argsort(S)[::-1]
                U, S, Vt = U[:, order], S[order], Vt[order]
            except:
                continue
            
            n_pcs = min(30, len(S))
            proj = X_c @ Vt[:n_pcs].T
            
            centroids = {}
            for cat in main_cats:
                cat_words = CATEGORIES_13[cat][:20]
                cat_idx = [i for i, w in enumerate(wnames) if w in cat_words]
                if len(cat_idx) >= 3:
                    centroids[cat] = proj[cat_idx].mean(axis=0)
            
            if len(centroids) < 10:
                continue
            
            log(f"  L{li}:")
            
            # 所有对: 分为cross-superclass和same-superclass
            cross_sc_contribs = []  # 每个cross-superclass对的PC贡献谱
            same_sc_contribs = []   # 每个same-superclass对的PC贡献谱
            
            for i, c1 in enumerate(main_cats):
                for j, c2 in enumerate(main_cats):
                    if i >= j or c1 not in centroids or c2 not in centroids:
                        continue
                    
                    diff = centroids[c1] - centroids[c2]
                    total_dist_sq = np.sum(diff ** 2)
                    if total_dist_sq < 1e-10:
                        continue
                    
                    contribs = diff ** 2 / total_dist_sq  # (n_pcs,)
                    
                    if is_cross_superclass(c1, c2):
                        cross_sc_contribs.append(contribs)
                    elif is_same_superclass(c1, c2):
                        same_sc_contribs.append(contribs)
            
            if cross_sc_contribs:
                cross_mean = np.mean(cross_sc_contribs, axis=0)
                log(f"    Cross-superclass ({len(cross_sc_contribs)} pairs):")
                log(f"      Mean PC contribution spectrum (top 10 PCs):")
                top_pcs = np.argsort(cross_mean)[::-1][:10]
                for pc in top_pcs:
                    log(f"        PC{pc:2d}: {cross_mean[pc]:.4f} ({cross_mean[pc]*100:.1f}%)")
                # 前3/5/10个PC的累积贡献
                sorted_cross = np.sort(cross_mean)[::-1]
                cum_cross = np.cumsum(sorted_cross)
                log(f"      Cumulative: top3={cum_cross[2]:.3f}, top5={cum_cross[4]:.3f}")
            
            if same_sc_contribs:
                same_mean = np.mean(same_sc_contribs, axis=0)
                log(f"    Same-superclass ({len(same_sc_contribs)} pairs):")
                log(f"      Mean PC contribution spectrum (top 10 PCs):")
                top_pcs = np.argsort(same_mean)[::-1][:10]
                for pc in top_pcs:
                    log(f"        PC{pc:2d}: {same_mean[pc]:.4f} ({same_mean[pc]*100:.1f}%)")
                sorted_same = np.sort(same_mean)[::-1]
                cum_same = np.cumsum(sorted_same)
                log(f"      Cumulative: top3={cum_same[2]:.3f}, top5={cum_same[4]:.3f}")
            
            # 直接比较: 对每个PC, cross vs same的贡献差异
            if cross_sc_contribs and same_sc_contribs:
                log(f"    PC-level comparison (cross - same):")
                diff_spectrum = cross_mean - same_mean
                # 哪些PC对cross-superclass更重要?
                top_diff_pcs = np.argsort(diff_spectrum)[::-1][:5]
                for pc in top_diff_pcs:
                    log(f"      PC{pc:2d}: cross={cross_mean[pc]:.4f}, same={same_mean[pc]:.4f}, "
                        f"diff={diff_spectrum[pc]:+.4f} (MORE cross)")
                # 哪些PC对same-superclass更重要?
                bot_diff_pcs = np.argsort(diff_spectrum)[:5]
                for pc in bot_diff_pcs:
                    log(f"      PC{pc:2d}: cross={cross_mean[pc]:.4f}, same={same_mean[pc]:.4f}, "
                        f"diff={diff_spectrum[pc]:+.4f} (MORE same)")

        # =================================================================
        # Exp3: 每个PC的类别区分力
        # =================================================================
        log("\n=== Exp3: Per-PC Category Discriminability ===")
        
        for li in [sample_layers[min(2, len(sample_layers)-1)],  # ~L4
                   sample_layers[min(4, len(sample_layers)-1)],  # ~L8
                   mid_li]:  # 中间层
            wdu_vecs, wnames = [], []
            for w in all_words:
                if li in word_wdu.get(w, {}):
                    wdu_vecs.append(word_wdu[w][li])
                    wnames.append(w)
            
            if len(wdu_vecs) < 50:
                continue
            
            X = np.array(wdu_vecs, dtype=np.float32)
            X_c = X - X.mean(axis=0)
            k_svd = min(100, len(X_c) - 1, X_c.shape[1] - 1)
            try:
                U, S, Vt = svds(X_c.astype(np.float32), k=k_svd)
                order = np.argsort(S)[::-1]
                U, S, Vt = U[:, order], S[order], Vt[order]
            except:
                continue
            
            n_pcs = min(30, len(S))
            proj = X_c @ Vt[:n_pcs].T
            
            # 对每个PC, 做简单的kNN分类(只用该PC)
            from collections import Counter
            
            # 准备标签
            labels = []
            for w in wnames:
                cat = word2cat.get(w, "")
                if not cat.startswith("boundary"):
                    labels.append(cat)
                else:
                    labels.append("boundary")
            
            log(f"  L{li}: Per-PC kNN accuracy (k=5, 13-class):")
            
            # 4-class (superclass) accuracy per PC
            sc_labels = []
            for w in wnames:
                cat = word2cat.get(w, "")
                if cat.startswith("boundary"):
                    sc_labels.append("boundary")
                else:
                    sc_labels.append(get_superclass(cat))
            
            pc_sc_acc = []
            pc_cat_acc = []
            
            for pc in range(min(20, n_pcs)):
                pc_vals = proj[:, pc]
                
                # Superclass accuracy
                sc_correct = 0
                sc_total = 0
                for i in range(len(pc_vals)):
                    if sc_labels[i] == "boundary":
                        continue
                    # Find k nearest neighbors
                    dists = np.abs(pc_vals - pc_vals[i])
                    dists[i] = float('inf')
                    nn_idx = np.argsort(dists)[:5]
                    nn_labels = [sc_labels[j] for j in nn_idx if sc_labels[j] != "boundary"]
                    if nn_labels:
                        majority = Counter(nn_labels).most_common(1)[0][0]
                        if majority == sc_labels[i]:
                            sc_correct += 1
                        sc_total += 1
                
                sc_acc = sc_correct / max(sc_total, 1)
                pc_sc_acc.append((pc, sc_acc))
                
                # Category accuracy
                cat_correct = 0
                cat_total = 0
                for i in range(len(pc_vals)):
                    if labels[i] == "boundary":
                        continue
                    dists = np.abs(pc_vals - pc_vals[i])
                    dists[i] = float('inf')
                    nn_idx = np.argsort(dists)[:5]
                    nn_labels = [labels[j] for j in nn_idx if labels[j] != "boundary"]
                    if nn_labels:
                        majority = Counter(nn_labels).most_common(1)[0][0]
                        if majority == labels[i]:
                            cat_correct += 1
                        cat_total += 1
                
                cat_acc = cat_correct / max(cat_total, 1)
                pc_cat_acc.append((pc, cat_acc))
            
            # 排序打印
            pc_sc_acc.sort(key=lambda x: -x[1])
            pc_cat_acc.sort(key=lambda x: -x[1])
            
            log(f"    Top-5 PCs for SUPERCLASS discrimination:")
            for pc, acc in pc_sc_acc[:5]:
                log(f"      PC{pc:2d}: {acc:.3f}")
            
            log(f"    Top-5 PCs for CATEGORY discrimination:")
            for pc, acc in pc_cat_acc[:5]:
                log(f"      PC{pc:2d}: {acc:.3f}")
            
            # 检查: superclass-top PCs 和 category-top PCs 是否不同?
            sc_top_set = set(pc for pc, _ in pc_sc_acc[:5])
            cat_top_set = set(pc for pc, _ in pc_cat_acc[:5])
            overlap = sc_top_set & cat_top_set
            log(f"    Overlap between top-5 SC and top-5 CAT PCs: {len(overlap)}/5 = {overlap if overlap else 'none'}")

        # =================================================================
        # Exp4: 边界词在各PC上的位置
        # =================================================================
        log("\n=== Exp4: Boundary Word PC Decomposition ===")
        
        for li in [sample_layers[min(4, len(sample_layers)-1)]]:  # ~L8
            wdu_vecs, wnames = [], []
            for w in all_words:
                if li in word_wdu.get(w, {}):
                    wdu_vecs.append(word_wdu[w][li])
                    wnames.append(w)
            
            if len(wdu_vecs) < 50:
                continue
            
            X = np.array(wdu_vecs, dtype=np.float32)
            X_c = X - X.mean(axis=0)
            k_svd = min(100, len(X_c) - 1, X_c.shape[1] - 1)
            try:
                U, S, Vt = svds(X_c.astype(np.float32), k=k_svd)
                order = np.argsort(S)[::-1]
                U, S, Vt = U[:, order], S[order], Vt[order]
            except:
                continue
            
            n_pcs = min(30, len(S))
            proj = X_c @ Vt[:n_pcs].T
            
            centroids = {}
            for cat in main_cats:
                cat_words = CATEGORIES_13[cat][:20]
                cat_idx = [i for i, w in enumerate(wnames) if w in cat_words]
                if len(cat_idx) >= 3:
                    centroids[cat] = proj[cat_idx].mean(axis=0)
            
            # 对每个边界词, 分解它到两个相关类别质心的距离差异
            boundary_pairs = {
                "fruit_veg": ("fruit", "vegetable"),
                "weapon_tool": ("weapon", "tool"),
                "bird_animal": ("bird", "animal"),
                "body_plant": ("body_part", "plant"),
            }
            
            log(f"  L{li}:")
            for btype, (cat1, cat2) in boundary_pairs.items():
                if cat1 not in centroids or cat2 not in centroids:
                    continue
                
                log(f"    --- {btype}: {cat1} vs {cat2} ---")
                
                # 质心差异分解
                centroid_diff = centroids[cat1] - centroids[cat2]  # (n_pcs,)
                total_sq = np.sum(centroid_diff ** 2)
                if total_sq < 1e-10:
                    continue
                
                # Top PCs for this pair
                centroid_contribs = centroid_diff ** 2 / total_sq
                top_pair_pcs = np.argsort(centroid_contribs)[::-1][:5]
                log(f"      Pair top PCs: {[(f'PC{pc}', f'{centroid_contribs[pc]:.3f}') for pc in top_pair_pcs]}")
                
                # 边界词分析
                for w in BOUNDARY_WORDS.get(btype, []):
                    w_idx = None
                    for i, wn in enumerate(wnames):
                        if wn == w:
                            w_idx = i
                            break
                    if w_idx is None:
                        continue
                    
                    # 词到两个质心的距离差异
                    d1 = np.linalg.norm(proj[w_idx] - centroids[cat1])
                    d2 = np.linalg.norm(proj[w_idx] - centroids[cat2])
                    
                    # 差异分解: 在每个PC上, 词更接近cat1还是cat2?
                    word_diff = proj[w_idx] - centroids[cat2]  # 词到cat2质心的向量
                    cat1_diff = centroids[cat1] - centroids[cat2]  # cat1到cat2的向量
                    
                    # 每个PC上的"cat1倾向" = (word_pos - cat2_centroid) * sign(cat1 - cat2)
                    # 正值=更接近cat1, 负值=更接近cat2
                    pc_alignment = word_diff * np.sign(cat1_diff)
                    
                    # 对齐最强的5个PC
                    aligned_pcs = np.argsort(pc_alignment)[::-1][:5]
                    anti_aligned_pcs = np.argsort(pc_alignment)[:5]
                    
                    closer = cat1 if d1 < d2 else cat2
                    ratio = max(d1, d2) / max(min(d1, d2), 1e-10)
                    
                    log(f"      {w:12s}: closer to {closer:10s} (d1={d1:.3f}, d2={d2:.3f}, ratio={ratio:.2f})")
                    log(f"        Pro-cat1 PCs: {[(f'PC{pc}', f'{pc_alignment[pc]:+.3f}') for pc in aligned_pcs[:3]]}")
                    log(f"        Pro-cat2 PCs: {[(f'PC{pc}', f'{pc_alignment[pc]:+.3f}') for pc in anti_aligned_pcs[:3]]}")

        # =================================================================
        # Exp5: 跨层PC贡献演化
        # =================================================================
        log("\n=== Exp5: Cross-Layer PC Contribution Evolution ===")
        
        # 对tool-weapon和fruit-vegetable, 追踪top贡献PC在各层的变化
        focus_pairs = [("tool", "weapon"), ("fruit", "vegetable"), 
                       ("body_part", "tool"), ("animal", "vehicle")]
        
        for c1, c2 in focus_pairs:
            log(f"\n  --- {c1} vs {c2} cross-layer ---")
            
            for li in sample_layers:
                wdu_vecs, wnames = [], []
                for w in all_words:
                    if li in word_wdu.get(w, {}):
                        wdu_vecs.append(word_wdu[w][li])
                        wnames.append(w)
                
                if len(wdu_vecs) < 50:
                    continue
                
                X = np.array(wdu_vecs, dtype=np.float32)
                X_c = X - X.mean(axis=0)
                k_svd = min(100, len(X_c) - 1, X_c.shape[1] - 1)
                try:
                    U, S, Vt = svds(X_c.astype(np.float32), k=k_svd)
                    order = np.argsort(S)[::-1]
                    U, S, Vt = U[:, order], S[order], Vt[order]
                except:
                    continue
                
                n_pcs = min(30, len(S))
                proj = X_c @ Vt[:n_pcs].T
                
                centroids = {}
                for cat in main_cats:
                    cat_words = CATEGORIES_13[cat][:20]
                    cat_idx = [i for i, w in enumerate(wnames) if w in cat_words]
                    if len(cat_idx) >= 3:
                        centroids[cat] = proj[cat_idx].mean(axis=0)
                
                if c1 not in centroids or c2 not in centroids:
                    continue
                
                diff = centroids[c1] - centroids[c2]
                total_sq = np.sum(diff ** 2)
                if total_sq < 1e-10:
                    log(f"    L{li}: dist^2 ≈ 0")
                    continue
                
                contribs = diff ** 2 / total_sq
                top3 = np.argsort(contribs)[::-1][:3]
                
                log(f"    L{li:3d}: top3 PCs = [{top3[0]},{top3[1]},{top3[2]}] "
                    f"({contribs[top3[0]]:.3f}, {contribs[top3[1]]:.3f}, {contribs[top3[2]]:.3f}) "
                    f"cum3={sum(contribs[top3]):.3f}")

        # =================================================================
        # 关键数据汇总
        # =================================================================
        log("\n=== Summary ===")
        
        # 对中间层, 统计关键数据
        for li in [sample_layers[min(4, len(sample_layers)-1)]]:
            wdu_vecs, wnames = [], []
            for w in all_words:
                if li in word_wdu.get(w, {}):
                    wdu_vecs.append(word_wdu[w][li])
                    wnames.append(w)
            
            if len(wdu_vecs) < 50:
                continue
            
            X = np.array(wdu_vecs, dtype=np.float32)
            X_c = X - X.mean(axis=0)
            k_svd = min(100, len(X_c) - 1, X_c.shape[1] - 1)
            try:
                U, S, Vt = svds(X_c.astype(np.float32), k=k_svd)
                order = np.argsort(S)[::-1]
                U, S, Vt = U[:, order], S[order], Vt[order]
            except:
                continue
            
            n_pcs = min(30, len(S))
            proj = X_c @ Vt[:n_pcs].T
            
            centroids = {}
            for cat in main_cats:
                cat_words = CATEGORIES_13[cat][:20]
                cat_idx = [i for i, w in enumerate(wnames) if w in cat_words]
                if len(cat_idx) >= 3:
                    centroids[cat] = proj[cat_idx].mean(axis=0)
            
            # 对所有78对, 计算top-1 PC贡献
            log(f"  L{li}: Top-1 PC contribution for all 78 pairs:")
            pair_top1 = []
            for i, c1 in enumerate(main_cats):
                for j, c2 in enumerate(main_cats):
                    if i >= j or c1 not in centroids or c2 not in centroids:
                        continue
                    diff = centroids[c1] - centroids[c2]
                    total_sq = np.sum(diff ** 2)
                    if total_sq < 1e-10:
                        continue
                    contribs = diff ** 2 / total_sq
                    top1_pc = np.argmax(contribs)
                    top1_val = contribs[top1_pc]
                    pair_top1.append((c1, c2, top1_pc, top1_val))
            
            # 按top1贡献排序
            pair_top1.sort(key=lambda x: -x[3])
            log(f"    Highest top-1 PC contribution (most 1D-separable):")
            for c1, c2, pc, val in pair_top1[:10]:
                sc1, sc2 = get_superclass(c1), get_superclass(c2)
                log(f"      {c1:12s}-{c2:12s}: PC{pc} ({val:.3f}) [{sc1}/{sc2}]")
            
            log(f"    Lowest top-1 PC contribution (least 1D-separable):")
            for c1, c2, pc, val in pair_top1[-10:]:
                sc1, sc2 = get_superclass(c1), get_superclass(c2)
                log(f"      {c1:12s}-{c2:12s}: PC{pc} ({val:.3f}) [{sc1}/{sc2}]")
            
            # 统计: 多少对的主要贡献PC是PC0?
            pc0_pairs = sum(1 for _, _, pc, _ in pair_top1 if pc == 0)
            log(f"    Pairs where PC0 is dominant: {pc0_pairs}/{len(pair_top1)}")
            
            # 统计: 对superclass对 vs same-class对, top1贡献的均值
            cross_top1 = [val for c1, c2, _, val in pair_top1 if is_cross_superclass(c1, c2)]
            same_top1 = [val for c1, c2, _, val in pair_top1 if is_same_superclass(c1, c2)]
            if cross_top1:
                log(f"    Cross-superclass mean top1: {np.mean(cross_top1):.3f}")
            if same_top1:
                log(f"    Same-superclass mean top1: {np.mean(same_top1):.3f}")

        # 保存结果
        result_dir = Path(f"d:/Ai2050/TransformerLens-Project/results/causal_fiber/{model_name}_cclxxxviii")
        result_dir.mkdir(parents=True, exist_ok=True)
        
        save_data = {"model": model_name, "n_layers": n_layers, "sample_layers": sample_layers}
        with open(result_dir / "config.json", 'w') as f:
            json.dump(save_data, f, indent=2)
        
        log(f"Results saved to {result_dir}")

        release_model(model)
        log(f"=== {model_name} COMPLETE ===")
        
    except Exception as e:
        log(f"ERROR in {model_name}: {e}")
        traceback.print_exc(file=open(LOG, 'a', encoding='utf-8'))
        try:
            release_model(model)
        except:
            pass


if __name__ == "__main__":
    models = ['qwen3', 'glm4', 'deepseek7b']
    
    for model_name in models:
        run_model(model_name)
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        time.sleep(5)
    
    log("=== ALL MODELS COMPLETE ===")
