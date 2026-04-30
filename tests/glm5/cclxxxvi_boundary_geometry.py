"""
CCLXXXVI: 语义边界几何结构分析
基于CCLXXXV发现: fruit↔vegetable, weapon↔tool是最强混淆对
目标: 积累类别边界的几何数据，不预设理论

Exp1: 13×13类别间质心距离矩阵 (全矩阵，各层，top-d PCs)
Exp2: 每个词的跨类别最近邻 — 构建混淆图
Exp3: 边界词定位 — 语义模糊词在几何空间中位于何处？
Exp4: 类别可分性指数 — centroid_dist / avg_within_spread
Exp5: 跨入侵率 — % cat1词更接近cat2质心
"""
import sys, os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxxvi_log.txt"

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        import time
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()

log("=== CCLXXXVI Script started ===")

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

# ===== 13类别 × 20词 (与CCLXXXV完全相同) =====
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

# 边界词 — 语义上介于两个类别之间
BOUNDARY_WORDS = {
    "fruit_veg": [
        "tomato", "cucumber", "avocado", "eggplant", "zucchini",
        "olive", "pumpkin", "squash", "pepper", "corn",
    ],
    "weapon_tool": [
        "knife", "axe", "machete", "sickle", "pickaxe",
        "chisel", "scissors", "shovel", "hammer", "screwdriver",
    ],
    "bird_animal": [
        "penguin", "ostrich", "chicken", "turkey", "duck",
        "goose", "swan", "emu", "flamingo", "peacock",
    ],
    "body_plant": [
        "trunk", "bark", "crown", "branch", "thorn",
        "root", "stem", "leaf", "seed", "bud",
    ],
}

ANIMATE_CATS = {"animal", "bird", "fish", "insect"}
PLANT_CATS = {"plant", "fruit", "vegetable"}
BODY_CATS = {"body_part"}
ARTIFACT_CATS = {"tool", "vehicle", "clothing", "weapon", "furniture"}


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

        # ===== 收集主类别词 =====
        all_words, all_cats = [], []
        for cat, words in CATEGORIES_13.items():
            all_words.extend(words[:20])
            all_cats.extend([cat] * 20)
        
        # 添加边界词
        boundary_word_type = {}
        for btype, words in BOUNDARY_WORDS.items():
            for w in words:
                if w not in all_words:
                    all_words.append(w)
                    all_cats.append(f"boundary_{btype}")
                    boundary_word_type[w] = btype
        
        cat_names = sorted(set(all_cats))
        n_total = len(all_words)
        log(f"Total words: {n_total} (including {sum(len(v) for v in BOUNDARY_WORDS.values())} boundary)")

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

        def get_wdu_for_layer(li):
            wdu_list, wnames = [], []
            for w in all_words:
                if li in word_wdu.get(w, {}):
                    wdu_list.append(word_wdu[w][li])
                    wnames.append(w)
            return wdu_list, wnames

        # ===== 选择测试层 =====
        if n_layers <= 30:
            sample_layers = [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]
        else:
            sample_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
        log(f"Sample layers: {sample_layers}")

        from scipy.sparse.linalg import svds

        # =================================================================
        # Exp1: 13×13 Category Pairwise Centroid Distance Matrix
        # =================================================================
        log("\n=== Exp1: Category Pairwise Centroid Distance Matrix ===")
        
        main_cats = sorted(CATEGORIES_13.keys())
        
        for li in sample_layers:
            wdu_vecs, wnames = get_wdu_for_layer(li)
            if len(wdu_vecs) < 100:
                continue
            
            X = np.array(wdu_vecs, dtype=np.float32)
            n = len(wdu_vecs)
            
            # 中心化 + SVD
            X_c = X - X.mean(axis=0)
            k_svd = min(100, n - 1, X_c.shape[1] - 1)
            try:
                U, S, Vt = svds(X_c.astype(np.float32), k=k_svd)
                order = np.argsort(S)[::-1]
                U, S, Vt = U[:, order], S[order], Vt[order]
            except:
                continue
            
            # Project to top-d PCs
            for d_proj in [3, 5, 10]:
                proj = X_c @ Vt[:d_proj].T  # (n, d_proj)
                
                # Compute centroids for each main category
                centroids = {}
                for cat in main_cats:
                    cat_words = CATEGORIES_13[cat][:20]
                    cat_idx = [i for i, w in enumerate(wnames) if w in cat_words]
                    if len(cat_idx) == 0:
                        continue
                    centroids[cat] = proj[cat_idx].mean(axis=0)
                
                # Pairwise distances
                dist_matrix = {}
                pairs_list = []
                for i, c1 in enumerate(main_cats):
                    for j, c2 in enumerate(main_cats):
                        if c1 in centroids and c2 in centroids:
                            d = np.linalg.norm(centroids[c1] - centroids[c2])
                            dist_matrix[(c1, c2)] = d
                            if i < j:
                                pairs_list.append((c1, c2, d))
                
                pairs_list.sort(key=lambda x: x[2])
                
                log(f"  L{li} d={d_proj}:")
                log(f"    CLOSEST 5 pairs:")
                for c1, c2, d in pairs_list[:5]:
                    log(f"      {c1:12s} - {c2:12s}: {d:.4f}")
                log(f"    FARTHEST 5 pairs:")
                for c1, c2, d in pairs_list[-5:]:
                    log(f"      {c1:12s} - {c2:12s}: {d:.4f}")
                
                # Same-superclass vs cross-superclass distances
                same_sc_dists = []
                cross_sc_dists = []
                for c1, c2, d in pairs_list:
                    sc1 = None
                    sc2 = None
                    for sc, cats in [("animate", ANIMATE_CATS), ("plant", PLANT_CATS), 
                                     ("body", BODY_CATS), ("artifact", ARTIFACT_CATS)]:
                        if c1 in cats: sc1 = sc
                        if c2 in cats: sc2 = sc
                    if sc1 and sc2:
                        if sc1 == sc2:
                            same_sc_dists.append(d)
                        else:
                            cross_sc_dists.append(d)
                
                if same_sc_dists and cross_sc_dists:
                    log(f"    Same-superclass mean dist: {np.mean(same_sc_dists):.4f}")
                    log(f"    Cross-superclass mean dist: {np.mean(cross_sc_dists):.4f}")
                    log(f"    Ratio (cross/same): {np.mean(cross_sc_dists)/max(np.mean(same_sc_dists),1e-10):.4f}")

        # =================================================================
        # Exp2: Cross-category Nearest Neighbor Confusion Graph
        # =================================================================
        log("\n=== Exp2: Cross-category NN Confusion Graph ===")
        
        for li in sample_layers:
            wdu_vecs, wnames = get_wdu_for_layer(li)
            if len(wdu_vecs) < 100:
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
            
            for d_proj in [5, 50]:
                proj = X_c @ Vt[:d_proj].T
                
                # For each main-category word, find nearest neighbor from each other category
                confusion_counts = defaultdict(lambda: defaultdict(int))
                confusion_total = defaultdict(int)
                
                main_word_set = set()
                for cat in main_cats:
                    for w in CATEGORIES_13[cat][:20]:
                        main_word_set.add(w)
                
                for i, w in enumerate(wnames):
                    if w not in main_word_set:
                        continue
                    w_cat = word2cat.get(w)
                    if w_cat is None or w_cat.startswith("boundary"):
                        continue
                    
                    # Distance to all other words
                    dists = np.linalg.norm(proj - proj[i], axis=1)
                    dists[i] = float('inf')  # exclude self
                    
                    # Find nearest word from each other category
                    for j, wj in enumerate(wnames):
                        wj_cat = word2cat.get(wj)
                        if wj_cat is None or wj_cat.startswith("boundary") or wj_cat == w_cat:
                            continue
                        # Track closest word from each other category
                        if f"_nn_{wj_cat}" not in confusion_counts[w_cat] or dists[j] < confusion_counts[w_cat][f"_nn_{wj_cat}_dist"]:
                            confusion_counts[w_cat][f"_nn_{wj_cat}"] = wj
                            confusion_counts[w_cat][f"_nn_{wj_cat}_dist"] = dists[j]
                    
                    # Count which category the overall nearest neighbor belongs to
                    nn_idx = np.argmin(dists)
                    nn_word = wnames[nn_idx]
                    nn_cat = word2cat.get(nn_word, "")
                    if not nn_cat.startswith("boundary") and nn_cat != w_cat:
                        confusion_counts[w_cat][nn_cat] += 1
                        confusion_total[w_cat] += 1
                
                log(f"  L{li} d={d_proj}: Cross-category NN confusion:")
                for cat in main_cats:
                    if cat not in confusion_total or confusion_total[cat] == 0:
                        continue
                    total = confusion_total[cat]
                    sorted_conf = sorted(
                        [(k, v) for k, v in confusion_counts[cat].items() if not k.startswith("_")],
                        key=lambda x: -x[1]
                    )[:4]
                    conf_str = ", ".join([f"{k}:{v/total:.0%}" for k, v in sorted_conf])
                    log(f"    {cat:12s} -> {conf_str}")

        # =================================================================
        # Exp3: Boundary Word Positioning
        # =================================================================
        log("\n=== Exp3: Boundary Word Positioning ===")
        
        for li in sample_layers[1:-1]:  # skip L0 and last layer
            wdu_vecs, wnames = get_wdu_for_layer(li)
            if len(wdu_vecs) < 100:
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
            
            d_proj = 5
            proj = X_c @ Vt[:d_proj].T
            
            # Compute main category centroids
            centroids = {}
            for cat in main_cats:
                cat_words = CATEGORIES_13[cat][:20]
                cat_idx = [i for i, w in enumerate(wnames) if w in cat_words]
                if len(cat_idx) == 0:
                    continue
                centroids[cat] = proj[cat_idx].mean(axis=0)
            
            # For each boundary word, compute distance to all centroids
            log(f"  L{li} d={d_proj}:")
            for btype, bwords in BOUNDARY_WORDS.items():
                log(f"    {btype} boundary:")
                for bw in bwords:
                    bw_idx = [i for i, w in enumerate(wnames) if w == bw]
                    if not bw_idx:
                        continue
                    bw_vec = proj[bw_idx[0]]
                    
                    dists = {}
                    for cat, cent in centroids.items():
                        dists[cat] = np.linalg.norm(bw_vec - cent)
                    
                    sorted_d = sorted(dists.items(), key=lambda x: x[1])
                    nearest = sorted_d[0]
                    second = sorted_d[1]
                    ratio = second[1] / max(nearest[1], 1e-10)
                    
                    # Is this word actually in a main category?
                    actual_cat = None
                    for cat, words in CATEGORIES_13.items():
                        if bw in words:
                            actual_cat = cat
                            break
                    
                    match_str = f"actual={actual_cat}" if actual_cat else "not_in_main"
                    log(f"      {bw:15s}: nearest={nearest[0]:12s}({nearest[1]:.3f}), "
                        f"2nd={second[0]:12s}({second[1]:.3f}), ratio={ratio:.3f} [{match_str}]")

        # =================================================================
        # Exp4: Category Separability Index
        # =================================================================
        log("\n=== Exp4: Category Separability Index ===")
        
        for li in sample_layers[1:-1]:
            wdu_vecs, wnames = get_wdu_for_layer(li)
            if len(wdu_vecs) < 100:
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
            
            d_proj = 5
            proj = X_c @ Vt[:d_proj].T
            
            # For each category pair, compute separability
            cat_vecs = {}
            for cat in main_cats:
                cat_words = CATEGORIES_13[cat][:20]
                cat_idx = [i for i, w in enumerate(wnames) if w in cat_words]
                if len(cat_idx) >= 2:
                    cat_vecs[cat] = proj[cat_idx]
            
            sep_results = []
            for i, c1 in enumerate(main_cats):
                for j, c2 in enumerate(main_cats):
                    if i >= j or c1 not in cat_vecs or c2 not in cat_vecs:
                        continue
                    
                    v1 = cat_vecs[c1]
                    v2 = cat_vecs[c2]
                    cent1 = v1.mean(axis=0)
                    cent2 = v2.mean(axis=0)
                    
                    cent_dist = np.linalg.norm(cent1 - cent2)
                    spread1 = np.mean([np.linalg.norm(v - cent1) for v in v1])
                    spread2 = np.mean([np.linalg.norm(v - cent2) for v in v2])
                    avg_spread = (spread1 + spread2) / 2
                    separability = cent_dist / max(avg_spread, 1e-10)
                    
                    # Cross-encroachment
                    enc1 = np.mean([1 for v in v1 if np.linalg.norm(v - cent2) < np.linalg.norm(v - cent1)])
                    enc2 = np.mean([1 for v in v2 if np.linalg.norm(v - cent1) < np.linalg.norm(v - cent2)])
                    
                    sep_results.append((c1, c2, separability, enc1, enc2, max(enc1, enc2), cent_dist))
            
            sep_results.sort(key=lambda x: x[2])
            
            log(f"  L{li} d={d_proj}:")
            log(f"    HARDEST to separate (sep < 2.0):")
            for c1, c2, sep, e1, e2, emax, cd in sep_results:
                if sep >= 2.0:
                    break
                log(f"      {c1:12s} vs {c2:12s}: sep={sep:.3f}, encroach={emax:.3f}, cent_d={cd:.3f}")
            
            log(f"    EASIEST to separate (top 5):")
            for c1, c2, sep, e1, e2, emax, cd in sep_results[-5:]:
                log(f"      {c1:12s} vs {c2:12s}: sep={sep:.3f}, encroach={emax:.3f}, cent_d={cd:.3f}")

        # =================================================================
        # Exp5: Within-category spread analysis
        # =================================================================
        log("\n=== Exp5: Within-category Spread ===")
        
        for li in sample_layers[1:-1]:
            wdu_vecs, wnames = get_wdu_for_layer(li)
            if len(wdu_vecs) < 100:
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
            
            d_proj = 5
            proj = X_c @ Vt[:d_proj].T
            
            log(f"  L{li} d={d_proj}:")
            for cat in main_cats:
                cat_words = CATEGORIES_13[cat][:20]
                cat_idx = [i for i, w in enumerate(wnames) if w in cat_words]
                if len(cat_idx) < 2:
                    continue
                
                cat_proj = proj[cat_idx]
                cent = cat_proj.mean(axis=0)
                
                # Spread: mean distance to centroid
                spreads = [np.linalg.norm(v - cent) for v in cat_proj]
                mean_spread = np.mean(spreads)
                std_spread = np.std(spreads)
                
                # Max pairwise distance within category
                max_pair = 0
                for ii in range(len(cat_idx)):
                    for jj in range(ii+1, len(cat_idx)):
                        d = np.linalg.norm(cat_proj[ii] - cat_proj[jj])
                        if d > max_pair:
                            max_pair = d
                
                log(f"    {cat:12s}: spread_mean={mean_spread:.3f} ± {std_spread:.3f}, "
                    f"max_pair_dist={max_pair:.3f}, n={len(cat_idx)}")

        log(f"\n=== {model_name} COMPLETE ===")
        
        # Save results
        out_dir = Path(rf"d:\Ai2050\TransformerLens-Project\results\causal_fiber\{model_name}_cclxxxvi")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "status.json", 'w') as sf:
            json.dump({"model": model_name, "status": "complete", 
                       "layers": sample_layers, "time": time.strftime('%Y-%m-%d %H:%M:%S')}, sf, indent=2)
        
        # Free GPU
        del model, word_wdu
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        log(f"ERROR in {model_name}: {e}")
        traceback.print_exc()
        try:
            del model
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass


def main():
    models = ['qwen3', 'glm4', 'deepseek7b']
    
    # Check completed
    completed = set()
    if os.path.exists(LOG):
        with open(LOG, 'r', encoding='utf-8') as f:
            content = f.read()
            for m in models:
                if f"{m} COMPLETE" in content:
                    completed.add(m)
    
    log(f"\n{'#'*70}")
    log(f"# CCLXXXVI: Semantic Boundary Geometry Analysis")
    log(f"# Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"# Completed: {completed}")
    log(f"{'#'*70}")
    
    for model_name in models:
        if model_name in completed:
            log(f"Skipping {model_name} (already completed)")
            continue
        run_model(model_name)

if __name__ == '__main__':
    main()
