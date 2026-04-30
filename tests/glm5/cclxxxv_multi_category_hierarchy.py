"""
Phase CCLXXXV: 多类别层次编码验证
=========================================
核心问题: CCLXXXIV用5个类别发现top PCs编码animate/inanimate超类
          这个层次编码是否普遍？用13个类别验证

类别设计(13个, ~260词):
- Animate: animal(20), bird(20), fish(20), insect(20)
- Plant: plant(20), fruit(20), vegetable(20)
- Body: body_part(20)
- Artifact: tool(20), vehicle(20), clothing(20), weapon(20), furniture(20)

超类层次:
- Level 0: all words (1 class)
- Level 1: animate vs inanimate (2 classes)
- Level 2: biological vs artifact (3 classes: animate+plant, body, artifact)
- Level 3: domain (6 classes: animal, bird/fish, insect, plant/fruit/veg, body, artifact)
- Level 4: 13 fine categories

Exp1: kNN accuracy at each hierarchy level, in top-d PCs space
  - 在不同d_proj下, 各层级的kNN准确率
  - 找到每个层级需要的最小PC数

Exp2: Individual PC semantic decoding
  - 对PC1-20, 分析各PC最强的语义区分
  - 哪些PC编码animate/inanimate? 哪些编码更细的区分?
  - PC之间的语义正交性

Exp3: Centroid structure analysis
  - 13个类别质心在PC空间中的位置
  - 质心之间的距离层次
  - 是否存在非animate/inanimate的超类?

Exp4: Cross-category confusion analysis
  - 在top-5 PCs空间中, 哪些类别最常被混淆?
  - 混淆矩阵的层次结构
  - 是否存在比animate/inanimate更强的超类?
"""

import sys, os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxxv_log.txt"

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        import time
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()

log("=== CCLXXXV Script started ===")

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

# ===== 超类层次定义 =====
# Level 1: animate vs inanimate
ANIMATE_CATS = {"animal", "bird", "fish", "insect"}
PLANT_CATS = {"plant", "fruit", "vegetable"}
BODY_CATS = {"body_part"}
ARTIFACT_CATS = {"tool", "vehicle", "clothing", "weapon", "furniture"}

SUPERCLASSES = {
    "level1_animate": {c: 0 if c in ANIMATE_CATS else 1 for c in CATEGORIES_13},
    "level2_bio_artifact": {},
    "level3_domain": {},
    "level4_13cat": {c: i for i, c in enumerate(sorted(CATEGORIES_13.keys()))},
}

# Level 2: biological(body included) vs artifact
for c in CATEGORIES_13:
    if c in ANIMATE_CATS or c in PLANT_CATS or c in BODY_CATS:
        SUPERCLASSES["level2_bio_artifact"][c] = 0
    else:
        SUPERCLASSES["level2_bio_artifact"][c] = 1

# Level 3: domain groups
domain_map = {}
for c in CATEGORIES_13:
    if c in ANIMATE_CATS:
        domain_map[c] = 0  # animal domain
    elif c in PLANT_CATS:
        domain_map[c] = 1  # plant domain
    elif c in BODY_CATS:
        domain_map[c] = 2  # body
    else:
        domain_map[c] = 3  # artifact domain
SUPERCLASSES["level3_domain"] = domain_map

# 词频估算(基于常见英文词频，取log10)
WORD_FREQ = {}
# Common words get higher freq
HIGH_FREQ = {"dog": 4.2, "cat": 4.1, "horse": 3.9, "cow": 3.2, "pig": 3.3,
             "bird": 3.8, "fish": 3.8, "tree": 3.7, "car": 4.5, "hand": 4.2,
             "head": 4.1, "eye": 3.9, "heart": 3.8, "apple": 3.9, "knife": 3.5,
             "shirt": 3.6, "chair": 3.5, "sword": 3.0, "shoe": 3.5}
for cat, words in CATEGORIES_13.items():
    for w in words:
        WORD_FREQ[w] = HIGH_FREQ.get(w, 2.5 + np.random.uniform(0, 1.0))


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

        # ===== 收集260词数据 =====
        all_words, all_cats = [], []
        for cat, words in CATEGORIES_13.items():
            all_words.extend(words[:20])
            all_cats.extend([cat] * 20)
        cat_names = sorted(set(all_cats))
        n_cats = len(cat_names)
        n_total = len(all_words)
        log(f"Total words: {n_total}, categories: {n_cats}")

        word2cat = {w: c for w, c in zip(all_words, all_cats)}
        cat2words = defaultdict(list)
        for w, c in zip(all_words, all_cats):
            cat2words[c].append(w)

        # ===== 前向推理收集数据 =====
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

        # ===== 辅助函数 =====
        def get_wdu_for_layer(li):
            lw = get_layer_weights(layers_list[li], d_model, mlp_type)
            if lw.W_gate is None:
                return [], []
            wdu_list, wnames = [], []
            for w in all_words:
                if li in word_wdu.get(w, {}):
                    wdu_list.append(word_wdu[w][li])
                    wnames.append(w)
            return wdu_list, wnames

        # ===== 结果保存 =====
        out_dir = Path(rf"d:\Ai2050\TransformerLens-Project\results\causal_fiber\{model_name}_cclxxxv")
        out_dir.mkdir(parents=True, exist_ok=True)

        # 选择3-4个关键层
        if n_layers <= 30:
            sample_layers = [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]
        else:
            sample_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
        log(f"Sample layers: {sample_layers}")

        from scipy.sparse.linalg import svds
        from scipy.stats import spearmanr

        # ===================================================================
        # Exp1: 各层级kNN准确率 vs PC维度
        # ===================================================================
        log("Exp1: Hierarchical kNN accuracy vs PC dimensions...")
        exp1_results = []

        for li in sample_layers:
            wdu_vecs, wnames = get_wdu_for_layer(li)
            if len(wdu_vecs) < 100:
                continue

            X = np.array(wdu_vecs, dtype=np.float32)
            n = len(wdu_vecs)
            fine_labels = [cat_names.index(word2cat[w]) for w in wnames]

            # 中心化 + SVD
            X_c = X - X.mean(axis=0)
            k_svd = min(100, n - 1, X_c.shape[1] - 1)
            if k_svd < 20:
                continue
            try:
                U, S, Vt = svds(X_c.astype(np.float32), k=k_svd)
                order = np.argsort(S)[::-1]
                U, S, Vt = U[:, order], S[order], Vt[order]
            except:
                continue

            # PC scores
            pc_scores = U @ np.diag(S)  # [n, k_svd]

            # 各超类标签
            sc_labels = {}
            for sc_name, sc_map in SUPERCLASSES.items():
                sc_labels[sc_name] = np.array([sc_map[word2cat[w]] for w in wnames])

            # 在不同PC子空间中测试各层级kNN
            for d_proj in [1, 2, 3, 5, 10, 20, 50, 100]:
                if d_proj > k_svd:
                    continue
                X_proj = pc_scores[:, :d_proj]

                # 计算距离矩阵
                dist_mat = np.zeros((n, n), dtype=np.float32)
                for i in range(n):
                    diff = X_proj[i] - X_proj
                    dist_mat[i] = np.sum(diff ** 2, axis=1)
                np.fill_diagonal(dist_mat, np.inf)

                # 各层级kNN
                level_knn = {}
                for sc_name, sc_lab in sc_labels.items():
                    n_sc = len(set(sc_lab))
                    correct = 0
                    for i in range(n):
                        nn_idx = np.argmin(dist_mat[i])
                        if sc_lab[nn_idx] == sc_lab[i]:
                            correct += 1
                    level_knn[sc_name] = round(correct / n, 4)

                # 随机基线
                random_baselines = {}
                for sc_name, sc_lab in sc_labels.items():
                    n_sc = len(set(sc_lab))
                    random_baselines[sc_name] = round(1.0 / n_sc, 4)

                exp1_results.append({
                    "layer": li,
                    "d_proj": d_proj,
                    "level_knn": level_knn,
                    "random_baselines": random_baselines,
                    "var_in_top": round(float(np.sum(S[:d_proj]**2) / np.sum(S**2)), 4),
                })

                log(f"  L{li} d={d_proj}: " +
                    ", ".join(f"{k}={v}" for k, v in level_knn.items()))

        with open(out_dir / "exp1_hierarchical_knn.json", 'w') as f:
            json.dump({"model": model_name, "results": exp1_results}, f, indent=2)
        log(f"Exp1 done: {len(exp1_results)} entries")

        # ===================================================================
        # Exp2: Individual PC semantic decoding
        # ===================================================================
        log("Exp2: Individual PC semantic decoding...")
        exp2_results = []

        # 使用中间层做详细分析
        mid_layer = sample_layers[len(sample_layers) // 2]
        for li in [mid_layer]:
            wdu_vecs, wnames = get_wdu_for_layer(li)
            if len(wdu_vecs) < 100:
                continue

            X = np.array(wdu_vecs, dtype=np.float32)
            n = len(wdu_vecs)
            fine_labels = [cat_names.index(word2cat[w]) for w in wnames]

            X_c = X - X.mean(axis=0)
            k_svd = min(100, n - 1, X_c.shape[1] - 1)
            try:
                U, S, Vt = svds(X_c.astype(np.float32), k=k_svd)
                order = np.argsort(S)[::-1]
                U, S, Vt = U[:, order], S[order], Vt[order]
            except:
                continue

            pc_scores = U @ np.diag(S)

            # 各超类标签
            sc_labels = {}
            for sc_name, sc_map in SUPERCLASSES.items():
                sc_labels[sc_name] = np.array([sc_map[word2cat[w]] for w in wnames])

            # 逐PC分析
            pc_semantic = []
            for pc_idx in range(min(30, k_svd)):
                scores = pc_scores[:, pc_idx]

                # 各类别在该PC上的均值
                cat_means = {}
                for cn in cat_names:
                    mask = np.array([word2cat[w] == cn for w in wnames])
                    cat_means[cn] = float(np.mean(scores[mask]))

                # 各超类在该PC上的区分度
                sc_separation = {}
                for sc_name, sc_lab in sc_labels.items():
                    n_sc = len(set(sc_lab))
                    # 类间方差 / 类内方差
                    grand_mean = np.mean(scores)
                    between_var = sum(
                        np.sum(sc_lab == c) * (np.mean(scores[sc_lab == c]) - grand_mean) ** 2
                        for c in range(n_sc)
                    ) / n
                    within_var = np.mean([
                        np.var(scores[sc_lab == c])
                        for c in range(n_sc)
                    ])
                    f_ratio = between_var / within_var if within_var > 0 else 0

                    # 单PC kNN
                    dist_1d = np.abs(scores[:, None] - scores[None, :])
                    np.fill_diagonal(dist_1d, np.inf)
                    correct = sum(
                        1 for i in range(n)
                        if sc_lab[np.argmin(dist_1d[i])] == sc_lab[i]
                    )
                    knn_1d = correct / n

                    sc_separation[sc_name] = {
                        "f_ratio": round(float(f_ratio), 4),
                        "knn_1d": round(float(knn_1d), 4),
                    }

                # 词频相关
                freqs = np.array([WORD_FREQ.get(w, 2.5) for w in wnames])
                freq_corr, _ = spearmanr(scores, freqs)

                # 范数相关
                norms = np.linalg.norm(X, axis=1)
                norm_corr, _ = spearmanr(scores, norms)

                # 找到该PC最强的语义区分
                best_sc = max(sc_separation.keys(), key=lambda k: sc_separation[k]["f_ratio"])
                best_f = sc_separation[best_sc]["f_ratio"]

                pc_semantic.append({
                    "pc_idx": pc_idx,
                    "singular_value": round(float(S[pc_idx]), 2),
                    "var_explained": round(float(S[pc_idx]**2 / (S**2).sum()), 6),
                    "best_superclass": best_sc,
                    "best_f_ratio": round(float(best_f), 4),
                    "best_knn_1d": round(float(sc_separation[best_sc]["knn_1d"]), 4),
                    "freq_corr": round(float(freq_corr), 4),
                    "norm_corr": round(float(norm_corr), 4),
                    "superclass_separation": sc_separation,
                    "cat_means": {k: round(v, 4) for k, v in sorted(cat_means.items(), key=lambda x: x[1])},
                })

                log(f"  PC{pc_idx}: best={best_sc}(F={best_f:.2f}), "
                    f"|freq|={abs(freq_corr):.3f}, |norm|={abs(norm_corr):.3f}")

            exp2_results.append({
                "layer": li,
                "pc_analysis": pc_semantic,
            })

        with open(out_dir / "exp2_pc_semantic.json", 'w') as f:
            json.dump({"model": model_name, "results": exp2_results}, f, indent=2)
        log(f"Exp2 done: {len(exp2_results)} layers")

        # ===================================================================
        # Exp3: 质心结构与超类距离
        # ===================================================================
        log("Exp3: Centroid structure analysis...")
        exp3_results = []

        for li in sample_layers:
            wdu_vecs, wnames = get_wdu_for_layer(li)
            if len(wdu_vecs) < 100:
                continue

            X = np.array(wdu_vecs, dtype=np.float32)
            n = len(wdu_vecs)

            X_c = X - X.mean(axis=0)
            k_svd = min(100, n - 1, X_c.shape[1] - 1)
            try:
                U, S, Vt = svds(X_c.astype(np.float32), k=k_svd)
                order = np.argsort(S)[::-1]
                U, S, Vt = U[:, order], S[order], Vt[order]
            except:
                continue

            pc_scores = U @ np.diag(S)

            # 13个类别质心
            centroids = {}
            for cn in cat_names:
                mask = np.array([word2cat[w] == cn for w in wnames])
                centroids[cn] = np.mean(X[mask], axis=0)

            # 质心之间的距离矩阵
            cent_list = [centroids[c] for c in cat_names]
            cent_dist = np.zeros((n_cats, n_cats))
            for i in range(n_cats):
                for j in range(n_cats):
                    cent_dist[i, j] = np.linalg.norm(cent_list[i] - cent_list[j])

            # 质心在PC空间中的投影
            cent_proj = {}
            for d_proj in [3, 5, 10]:
                cent_pc = {}
                for cn in cat_names:
                    mask = np.array([word2cat[w] == cn for w in wnames])
                    cent_orig = np.mean(X_c[mask], axis=0)
                    # 投影到top-d PCs
                    proj = Vt[:d_proj] @ cent_orig
                    cent_pc[cn] = proj.tolist()
                cent_proj[f"d{d_proj}"] = cent_pc

            # 超类内部距离 vs 超类之间距离
            sc_dist_stats = {}
            for sc_name, sc_map in SUPERCLASSES.items():
                n_sc = len(set(sc_map.values()))
                # 超类内距离
                within_dists = []
                between_dists = []
                for i in range(n_cats):
                    for j in range(i + 1, n_cats):
                        d = cent_dist[i, j]
                        if sc_map[cat_names[i]] == sc_map[cat_names[j]]:
                            within_dists.append(d)
                        else:
                            between_dists.append(d)
                wb_ratio = np.mean(within_dists) / np.mean(between_dists) if between_dists else 0
                sc_dist_stats[sc_name] = {
                    "within_mean": round(float(np.mean(within_dists)), 4) if within_dists else 0,
                    "between_mean": round(float(np.mean(between_dists)), 4) if between_dists else 0,
                    "wb_ratio": round(float(wb_ratio), 4),
                    "n_within": len(within_dists),
                    "n_between": len(between_dists),
                }

            # 层次聚类: 在不同d下
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import pdist

            hier_clusters = {}
            for d_proj in [3, 5, 10]:
                X_proj = pc_scores[:, :d_proj]
                Z = linkage(X_proj, method='ward')

                # 切成2, 3, 4, 6, 13类
                for n_cl in [2, 3, 4, 6, 13]:
                    cl = fcluster(Z, t=n_cl, criterion='maxclust')
                    # 计算聚类纯度
                    purity_sum = 0
                    for c in range(1, n_cl + 1):
                        mask = cl == c
                        if mask.sum() == 0:
                            continue
                        cat_counts = defaultdict(int)
                        for idx in np.where(mask)[0]:
                            cat_counts[word2cat[wnames[idx]]] += 1
                        purity_sum += max(cat_counts.values())
                    purity = purity_sum / n

                    hier_clusters[f"d{d_proj}_n{n_cl}"] = {
                        "purity": round(float(purity), 4),
                        "clusters": [int(x) for x in cl],
                    }

            exp3_results.append({
                "layer": li,
                "cent_dist_matrix": {cat_names[i]: {cat_names[j]: round(float(cent_dist[i, j]), 4)
                                                     for j in range(n_cats)}
                                     for i in range(n_cats)},
                "cent_proj": cent_proj,
                "sc_dist_stats": sc_dist_stats,
                "hier_clusters": hier_clusters,
            })

            log(f"  L{li}: sc_dist_stats = " +
                ", ".join(f"{k}:wb={v['wb_ratio']:.3f}" for k, v in sc_dist_stats.items()))

        with open(out_dir / "exp3_centroid_structure.json", 'w') as f:
            json.dump({"model": model_name, "results": exp3_results}, f, indent=2)
        log(f"Exp3 done: {len(exp3_results)} layers")

        # ===================================================================
        # Exp4: 混淆矩阵与层次结构
        # ===================================================================
        log("Exp4: Confusion matrix analysis...")
        exp4_results = []

        for li in sample_layers:
            wdu_vecs, wnames = get_wdu_for_layer(li)
            if len(wdu_vecs) < 100:
                continue

            X = np.array(wdu_vecs, dtype=np.float32)
            n = len(wdu_vecs)
            fine_labels = [cat_names.index(word2cat[w]) for w in wnames]

            X_c = X - X.mean(axis=0)
            k_svd = min(100, n - 1, X_c.shape[1] - 1)
            try:
                U, S, Vt = svds(X_c.astype(np.float32), k=k_svd)
                order = np.argsort(S)[::-1]
                U, S, Vt = U[:, order], S[order], Vt[order]
            except:
                continue

            pc_scores = U @ np.diag(S)

            # 在不同PC维度下的混淆矩阵
            for d_proj in [3, 5, 10, 50]:
                if d_proj > k_svd:
                    continue
                X_proj = pc_scores[:, :d_proj]

                # 距离矩阵
                dist_mat = np.zeros((n, n), dtype=np.float32)
                for i in range(n):
                    diff = X_proj[i] - X_proj
                    dist_mat[i] = np.sum(diff ** 2, axis=1)
                np.fill_diagonal(dist_mat, np.inf)

                # kNN混淆矩阵
                conf_mat = np.zeros((n_cats, n_cats), dtype=int)
                for i in range(n):
                    nn_idx = np.argmin(dist_mat[i])
                    conf_mat[fine_labels[i], fine_labels[nn_idx]] += 1

                # 转为概率
                conf_prob = conf_mat.astype(float)
                for i in range(n_cats):
                    row_sum = conf_prob[i].sum()
                    if row_sum > 0:
                        conf_prob[i] /= row_sum

                # 对角线均值 = kNN准确率
                knn_acc = np.mean(np.diag(conf_prob))

                # 最大混淆对(非对角)
                max_conf_val = 0
                max_conf_pair = ""
                for i in range(n_cats):
                    for j in range(n_cats):
                        if i != j and conf_prob[i, j] > max_conf_val:
                            max_conf_val = conf_prob[i, j]
                            max_conf_pair = f"{cat_names[i]}->{cat_names[j]}"

                exp4_results.append({
                    "layer": li,
                    "d_proj": d_proj,
                    "knn_acc": round(float(knn_acc), 4),
                    "max_confusion_pair": max_conf_pair,
                    "max_confusion_val": round(float(max_conf_val), 4),
                    "confusion_matrix": {
                        cat_names[i]: {cat_names[j]: round(float(conf_prob[i, j]), 4)
                                       for j in range(n_cats)}
                        for i in range(n_cats)
                    },
                })

                log(f"  L{li} d={d_proj}: knn={knn_acc:.3f}, "
                    f"max_conf={max_conf_pair}({max_conf_val:.3f})")

        with open(out_dir / "exp4_confusion_matrix.json", 'w') as f:
            json.dump({"model": model_name, "results": exp4_results}, f, indent=2)
        log(f"Exp4 done: {len(exp4_results)} entries")

        # ===== 释放模型 =====
        release_model(model)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        log(f"=== {model_name} done ===")

    except Exception as e:
        log(f"ERROR in {model_name}: {e}")
        traceback.print_exc(file=open(LOG, 'a'))
        try:
            release_model(model)
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # 清空日志
    with open(LOG, 'w') as f:
        f.write("")

    for model_name in ["qwen3", "glm4", "deepseek7b"]:
        run_model(model_name)
        gc.collect()
        time.sleep(5)

    log("=== All models done ===")
