"""
CCLXXXVII: 跨层语义轨迹追踪
基于CCLXXXVI发现: tool-weapon/fruit-vegetable是最近类别对, 边界词ratio≈1.0-1.5
目标: 追踪每个词从L0到L_last的类别归属演化, 积累跨层数据

Exp1: 每个词在各层的最近类别质心 — 类别轨迹图
Exp2: 类别归属"结晶层" — 从哪层开始类别分配稳定?
Exp3: 边界词 vs 典型词的轨迹对比
Exp4: 类别置信度(coscore = nearest/2nd_nearest distance ratio)跨层演化
Exp5: 超类归属(biological/artifact/body)的跨层演化
"""
import sys, os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxxvii_log.txt"

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        import time
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()

log("=== CCLXXXVII Script started ===")

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

# ===== 13类别 × 20词 (与CCLXXXV/VI相同) =====
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
    "fruit_veg": ["tomato", "cucumber", "avocado", "eggplant", "zucchini",
                  "olive", "pumpkin", "squash", "pepper", "corn"],
    "weapon_tool": ["knife", "axe", "machete", "sickle", "pickaxe",
                    "chisel", "scissors", "shovel", "hammer", "screwdriver"],
    "bird_animal": ["penguin", "ostrich", "chicken", "turkey", "duck",
                    "goose", "swan", "emu", "flamingo", "peacock"],
    "body_plant": ["trunk", "bark", "crown", "branch", "thorn",
                   "root", "stem", "leaf", "seed", "bud"],
}

# 典型词 — 每个类别选5个最典型的
TYPICAL_WORDS = {
    "animal": ["dog", "cat", "horse", "lion", "elephant"],
    "bird": ["eagle", "hawk", "owl", "sparrow", "falcon"],
    "fish": ["shark", "salmon", "tuna", "trout", "cod"],
    "insect": ["ant", "bee", "butterfly", "mosquito", "spider"],
    "plant": ["tree", "flower", "grass", "oak", "pine"],
    "fruit": ["apple", "orange", "banana", "grape", "cherry"],
    "vegetable": ["carrot", "potato", "onion", "cabbage", "broccoli"],
    "body_part": ["hand", "foot", "head", "eye", "heart"],
    "tool": ["hammer", "screwdriver", "wrench", "drill", "saw"],
    "vehicle": ["car", "bus", "train", "airplane", "bicycle"],
    "clothing": ["shirt", "dress", "hat", "shoe", "coat"],
    "weapon": ["sword", "spear", "bow", "dagger", "rifle"],
    "furniture": ["chair", "table", "bed", "sofa", "desk"],
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
        # 典型词 + 边界词 (不需要全部260词, 减少推理时间)
        test_words = set()
        word_expected_cat = {}
        word_type = {}  # "typical" or "boundary_xxx"
        
        for cat, words in TYPICAL_WORDS.items():
            for w in words:
                test_words.add(w)
                word_expected_cat[w] = cat
                word_type[w] = "typical"
        
        for btype, words in BOUNDARY_WORDS.items():
            for w in words:
                test_words.add(w)
                word_type[w] = f"boundary_{btype}"
                # boundary词没有单一expected_cat
        
        # 也加入每个类别的质心词(用于计算质心)
        centroid_words = set()
        for cat, words in CATEGORIES_13.items():
            for w in words[:20]:
                centroid_words.add(w)
                if w not in word_expected_cat:
                    word_expected_cat[w] = cat
        
        all_words = sorted(test_words | centroid_words)
        n_total = len(all_words)
        log(f"Total words: {n_total} (typical={sum(1 for v in word_type.values() if v=='typical')}, "
            f"boundary={sum(1 for v in word_type.values() if v.startswith('boundary'))}, "
            f"centroid={len(centroid_words - test_words)})")

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

        # ===== 各层SVD + 类别质心 + 轨迹追踪 =====
        from scipy.sparse.linalg import svds

        main_cats = sorted(CATEGORIES_13.keys())
        
        # 存储每个词在各层的类别归属
        word_trajectory = {}  # word -> {layer_idx: (nearest_cat, 2nd_cat, nearest_dist, 2nd_dist)}
        word_sc_trajectory = {}  # word -> {layer_idx: nearest_superclass}

        # 选择d_proj = 5 (CCLXXXVI发现d=5是最佳中间层维度)
        d_proj = 5

        log(f"\n=== Computing trajectories with d_proj={d_proj} ===")

        # 用每3层采样以减少计算量 (如果层数多)
        if n_layers <= 20:
            layer_indices = list(range(n_layers))
        elif n_layers <= 40:
            layer_indices = list(range(0, n_layers, 2))
        else:
            layer_indices = list(range(0, n_layers, 3))
        # 确保最后层在列表中
        if n_layers - 1 not in layer_indices:
            layer_indices.append(n_layers - 1)
        log(f"Layer indices: {layer_indices}")

        for li in layer_indices:
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
            
            proj = X_c @ Vt[:d_proj].T  # (n, d_proj)
            
            # 计算各类别质心
            centroids = {}
            for cat in main_cats:
                cat_words = CATEGORIES_13[cat][:20]
                cat_idx = [i for i, w in enumerate(wnames) if w in cat_words]
                if len(cat_idx) >= 3:
                    centroids[cat] = proj[cat_idx].mean(axis=0)
            
            if len(centroids) < 10:
                continue
            
            # 对每个测试词, 计算到各类质心的距离
            for i, w in enumerate(wnames):
                if w not in test_words and w not in centroid_words:
                    continue
                
                cat_dists = {}
                for cat, centroid in centroids.items():
                    cat_dists[cat] = np.linalg.norm(proj[i] - centroid)
                
                sorted_cats = sorted(cat_dists.items(), key=lambda x: x[1])
                nearest_cat = sorted_cats[0][0]
                nearest_dist = sorted_cats[0][1]
                second_cat = sorted_cats[1][0]
                second_dist = sorted_cats[1][1]
                
                if w not in word_trajectory:
                    word_trajectory[w] = {}
                word_trajectory[w][li] = (nearest_cat, second_cat, nearest_dist, second_dist)
                
                # 超类归属
                sc_nearest = get_superclass(nearest_cat)
                if w not in word_sc_trajectory:
                    word_sc_trajectory[w] = {}
                word_sc_trajectory[w][li] = sc_nearest

        # =================================================================
        # Exp1: 类别轨迹图 — 每个词在各层的最近类别
        # =================================================================
        log("\n=== Exp1: Category Trajectory ===")
        
        # 边界词轨迹
        for btype in ["fruit_veg", "weapon_tool", "bird_animal", "body_plant"]:
            log(f"\n  --- Boundary: {btype} ---")
            for w in BOUNDARY_WORDS[btype]:
                if w not in word_trajectory:
                    continue
                traj = word_trajectory[w]
                layers_sorted = sorted(traj.keys())
                # 只打印每隔几层的轨迹
                step = max(1, len(layers_sorted) // 8)
                cat_seq = []
                for li in layers_sorted[::step]:
                    cat_seq.append(f"L{li}:{traj[li][0]}")
                log(f"    {w:15s}: {' -> '.join(cat_seq)}")
        
        # 典型词轨迹 (每类1个)
        log(f"\n  --- Typical words ---")
        for cat in ["animal", "bird", "fruit", "vegetable", "tool", "weapon", "body_part"]:
            words = TYPICAL_WORDS[cat]
            w = words[0]  # 取第一个典型词
            if w not in word_trajectory:
                continue
            traj = word_trajectory[w]
            layers_sorted = sorted(traj.keys())
            step = max(1, len(layers_sorted) // 8)
            cat_seq = []
            for li in layers_sorted[::step]:
                cat_seq.append(f"L{li}:{traj[li][0]}")
            log(f"    {cat:12s} {w:15s}: {' -> '.join(cat_seq)}")

        # =================================================================
        # Exp2: 结晶层 — 从哪层开始类别分配稳定?
        # =================================================================
        log("\n=== Exp2: Crystallization Layer ===")
        
        crystallization = {}  # word -> first layer where category stabilizes
        
        for w in word_trajectory:
            traj = word_trajectory[w]
            layers_sorted = sorted(traj.keys())
            
            if len(layers_sorted) < 3:
                continue
            
            # 找到从哪层开始, 后续5层(或到最后)类别分配一致
            for start_idx in range(len(layers_sorted)):
                start_li = layers_sorted[start_idx]
                cat_at_start = traj[start_li][0]
                
                # 检查后续是否稳定
                stable = True
                check_end = min(start_idx + 5, len(layers_sorted))
                for j in range(start_idx, check_end):
                    if traj[layers_sorted[j]][0] != cat_at_start:
                        stable = False
                        break
                
                if stable and check_end - start_idx >= 3:
                    crystallization[w] = start_li
                    break
        
        # 按类别统计结晶层
        cat_cryst = defaultdict(list)
        for w, cry_li in crystallization.items():
            cat = word_expected_cat.get(w, "unknown")
            if cat != "unknown":
                cat_cryst[cat].append(cry_li)
        
        log("  Category crystallization layer (mean ± std):")
        for cat in main_cats:
            if cat in cat_cryst and len(cat_cryst[cat]) > 0:
                vals = cat_cryst[cat]
                log(f"    {cat:12s}: L{np.mean(vals):.1f} ± {np.std(vals):.1f} "
                    f"(n={len(vals)}, range L{min(vals)}-L{max(vals)})")
        
        # 边界词结晶层
        log("  Boundary word crystallization:")
        for btype in ["fruit_veg", "weapon_tool", "bird_animal", "body_plant"]:
            btype_cry = []
            for w in BOUNDARY_WORDS[btype]:
                if w in crystallization:
                    btype_cry.append((w, crystallization[w]))
            if btype_cry:
                vals = [v[1] for v in btype_cry]
                log(f"    {btype:15s}: L{np.mean(vals):.1f} ± {np.std(vals):.1f} "
                    f"(range L{min(vals)}-L{max(vals)})")
                for w, v in btype_cry:
                    final_cat = word_trajectory[w].get(max(word_trajectory[w].keys()), ("?",))[0]
                    log(f"      {w:15s}: crystallize at L{v}, final_cat={final_cat}")

        # =================================================================
        # Exp3: 边界词 vs 典型词的轨迹对比
        # =================================================================
        log("\n=== Exp3: Boundary vs Typical Trajectory Comparison ===")
        
        # 计算每个词在每层的"正确率" — 最近类别==期望类别?
        correct_by_layer_boundary = defaultdict(list)
        correct_by_layer_typical = defaultdict(list)
        
        for w in test_words:
            if w not in word_trajectory or w not in word_expected_cat:
                continue
            traj = word_trajectory[w]
            expected = word_expected_cat[w]
            wt = word_type.get(w, "unknown")
            
            for li, (nearest, second, nd, sd) in traj.items():
                correct = 1.0 if nearest == expected else 0.0
                if wt.startswith("boundary"):
                    correct_by_layer_boundary[li].append(correct)
                elif wt == "typical":
                    correct_by_layer_typical[li].append(correct)
        
        log("  Layer | Typical_acc | Boundary_acc | Gap")
        log("  " + "-" * 50)
        for li in sorted(correct_by_layer_typical.keys()):
            if li not in correct_by_layer_boundary:
                continue
            t_acc = np.mean(correct_by_layer_typical[li])
            b_acc = np.mean(correct_by_layer_boundary[li])
            log(f"  L{li:3d}  | {t_acc:.3f}      | {b_acc:.3f}        | {t_acc - b_acc:+.3f}")

        # =================================================================
        # Exp4: 类别置信度 (nearest/2nd_nearest ratio) 跨层演化
        # =================================================================
        log("\n=== Exp4: Category Confidence Across Layers ===")
        
        # confidence = 2nd_dist / nearest_dist (越大越自信)
        # boundary词的confidence应该更低
        
        confidence_by_layer_boundary = defaultdict(list)
        confidence_by_layer_typical = defaultdict(list)
        confidence_by_layer_cat = defaultdict(lambda: defaultdict(list))
        
        for w in test_words:
            if w not in word_trajectory:
                continue
            traj = word_trajectory[w]
            wt = word_type.get(w, "unknown")
            expected = word_expected_cat.get(w, "unknown")
            
            for li, (nearest, second, nd, sd) in traj.items():
                conf = sd / max(nd, 1e-10)  # 2nd_dist / nearest_dist
                if wt.startswith("boundary"):
                    confidence_by_layer_boundary[li].append(conf)
                elif wt == "typical":
                    confidence_by_layer_typical[li].append(conf)
                    confidence_by_layer_cat[expected][li].append(conf)
        
        log("  Layer | Typical_conf | Boundary_conf | Ratio(T/B)")
        log("  " + "-" * 55)
        for li in sorted(confidence_by_layer_typical.keys()):
            if li not in confidence_by_layer_boundary:
                continue
            t_conf = np.mean(confidence_by_layer_typical[li])
            b_conf = np.mean(confidence_by_layer_boundary[li])
            ratio = t_conf / max(b_conf, 1e-10)
            log(f"  L{li:3d}  | {t_conf:.3f}       | {b_conf:.3f}         | {ratio:.3f}")
        
        # 各类别置信度
        log("\n  Category confidence at mid-layer:")
        mid_layers = sorted(confidence_by_layer_typical.keys())
        if mid_layers:
            mid_li = mid_layers[len(mid_layers) // 2]
            log(f"  (at layer L{mid_li})")
            for cat in main_cats:
                if cat in confidence_by_layer_cat and mid_li in confidence_by_layer_cat[cat]:
                    vals = confidence_by_layer_cat[cat][mid_li]
                    log(f"    {cat:12s}: {np.mean(vals):.3f} (n={len(vals)})")

        # =================================================================
        # Exp5: 超类归属的跨层演化
        # =================================================================
        log("\n=== Exp5: Superclass Trajectory ===")
        
        # 对典型词, 超类从哪层开始稳定?
        sc_crystallization = {}
        for w in word_sc_trajectory:
            if word_type.get(w) != "typical":
                continue
            traj = word_sc_trajectory[w]
            layers_sorted = sorted(traj.keys())
            expected_sc = get_superclass(word_expected_cat.get(w, "unknown"))
            
            if len(layers_sorted) < 3:
                continue
            
            # 找到从哪层开始超类稳定
            for start_idx in range(len(layers_sorted)):
                start_li = layers_sorted[start_idx]
                sc_at_start = traj[start_li]
                
                stable = True
                check_end = min(start_idx + 5, len(layers_sorted))
                for j in range(start_idx, check_end):
                    if traj[layers_sorted[j]] != sc_at_start:
                        stable = False
                        break
                
                if stable and check_end - start_idx >= 3:
                    sc_crystallization[w] = (start_li, expected_sc, sc_at_start)
                    break
        
        # 按超类统计
        sc_cry = defaultdict(list)
        for w, (cry_li, expected, actual) in sc_crystallization.items():
            sc_cry[expected].append((cry_li, actual == expected))
        
        log("  Superclass crystallization:")
        for sc in ["biological_animate", "biological_plant", "body", "artifact"]:
            if sc in sc_cry:
                vals = [v[0] for v in sc_cry[sc]]
                correct = sum(1 for v in sc_cry[sc] if v[1])
                total = len(sc_cry[sc])
                log(f"    {sc:20s}: L{np.mean(vals):.1f} ± {np.std(vals):.1f} "
                    f"(n={total}, correct={correct}/{total}={correct/max(total,1):.2f})")
        
        # 边界词的超类轨迹
        log("\n  Boundary word superclass trajectory:")
        for btype in ["fruit_veg", "weapon_tool", "bird_animal"]:
            for w in BOUNDARY_WORDS[btype]:
                if w not in word_sc_trajectory:
                    continue
                traj = word_sc_trajectory[w]
                layers_sorted = sorted(traj.keys())
                step = max(1, len(layers_sorted) // 6)
                sc_seq = []
                for li in layers_sorted[::step]:
                    sc_seq.append(f"L{li}:{traj[li][:8]}")  # 截短显示
                log(f"    {w:15s}: {' -> '.join(sc_seq)}")

        # =================================================================
        # 关键数据汇总
        # =================================================================
        log("\n=== Summary ===")
        
        # 1. 典型词最终类别正确率 (最后层)
        last_li = max(layer_indices)
        typical_correct = 0
        typical_total = 0
        for w in test_words:
            if word_type.get(w) == "typical" and w in word_trajectory and last_li in word_trajectory[w]:
                typical_total += 1
                if word_trajectory[w][last_li][0] == word_expected_cat.get(w):
                    typical_correct += 1
        
        log(f"  Typical word final accuracy: {typical_correct}/{typical_total} = "
            f"{typical_correct/max(typical_total,1):.3f}")
        
        # 2. 边界词在最后层的类别分配
        log("  Boundary word final category assignment:")
        for btype in ["fruit_veg", "weapon_tool", "bird_animal", "body_plant"]:
            for w in BOUNDARY_WORDS[btype]:
                if w in word_trajectory and last_li in word_trajectory[w]:
                    nearest = word_trajectory[w][last_li][0]
                    second = word_trajectory[w][last_li][1]
                    conf = word_trajectory[w][last_li][3] / max(word_trajectory[w][last_li][2], 1e-10)
                    log(f"    {w:15s}: nearest={nearest:12s} (2nd={second:12s}, conf={conf:.2f})")
        
        # 3. 最早期(L0-L3)类别正确率
        early_layers = [li for li in layer_indices if li <= max(3, n_layers // 10)]
        if early_layers:
            early_li = early_layers[0]
            early_correct = 0
            early_total = 0
            for w in test_words:
                if word_type.get(w) == "typical" and w in word_trajectory and early_li in word_trajectory[w]:
                    early_total += 1
                    if word_trajectory[w][early_li][0] == word_expected_cat.get(w):
                        early_correct += 1
            log(f"  Early layer (L{early_li}) accuracy: {early_correct}/{early_total} = "
                f"{early_correct/max(early_total,1):.3f}")
        
        # 4. 所有词的类别变更次数
        log("  Category change count (typical words):")
        change_counts = defaultdict(list)
        for w in test_words:
            if word_type.get(w) != "typical" or w not in word_trajectory:
                continue
            traj = word_trajectory[w]
            layers_sorted = sorted(traj.keys())
            changes = 0
            for i in range(1, len(layers_sorted)):
                if traj[layers_sorted[i]][0] != traj[layers_sorted[i-1]][0]:
                    changes += 1
            cat = word_expected_cat.get(w, "unknown")
            change_counts[cat].append(changes)
        
        for cat in main_cats:
            if cat in change_counts:
                vals = change_counts[cat]
                log(f"    {cat:12s}: mean={np.mean(vals):.1f}, max={max(vals)}, "
                    f"n={len(vals)}")
        
        # 5. 类别变更方向矩阵 (cat_A -> cat_B的变更次数)
        log("  Category transition matrix (top 15):")
        transition_counts = defaultdict(int)
        for w in all_words:
            if w not in word_trajectory:
                continue
            traj = word_trajectory[w]
            layers_sorted = sorted(traj.keys())
            for i in range(1, len(layers_sorted)):
                prev_cat = traj[layers_sorted[i-1]][0]
                curr_cat = traj[layers_sorted[i]][0]
                if prev_cat != curr_cat:
                    transition_counts[(prev_cat, curr_cat)] += 1
        
        top_transitions = sorted(transition_counts.items(), key=lambda x: -x[1])[:15]
        for (c1, c2), count in top_transitions:
            log(f"    {c1:12s} -> {c2:12s}: {count}")

        # 保存结果
        result_dir = Path(f"d:/Ai2050/TransformerLens-Project/results/causal_fiber/{model_name}_cclxxxvii")
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存轨迹数据
        trajectory_data = {}
        for w in word_trajectory:
            trajectory_data[w] = {}
            for li, (nc, sc, nd, sd) in word_trajectory[w].items():
                trajectory_data[w][str(li)] = {
                    "nearest_cat": nc,
                    "second_cat": sc,
                    "nearest_dist": float(nd),
                    "second_dist": float(sd),
                    "confidence": float(sd / max(nd, 1e-10)),
                }
        
        with open(result_dir / "trajectories.json", 'w', encoding='utf-8') as f:
            json.dump(trajectory_data, f, indent=2, ensure_ascii=False)
        
        # 保存结晶层数据
        cryst_data = {}
        for w, cry_li in crystallization.items():
            cryst_data[w] = {
                "crystallization_layer": int(cry_li),
                "expected_cat": word_expected_cat.get(w, "unknown"),
                "word_type": word_type.get(w, "unknown"),
            }
        
        with open(result_dir / "crystallization.json", 'w', encoding='utf-8') as f:
            json.dump(cryst_data, f, indent=2, ensure_ascii=False)
        
        log(f"Results saved to {result_dir}")

        # 释放模型
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
