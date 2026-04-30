"""
Phase CCLXXXIV: Top PCs语义解码 — 最大方差方向编码了什么？
=============================================================
核心问题: CCLXXXIII发现类别信号不在top PCs中(top-10 kNN≈20%)
关键实验: 如果top PCs不含类别信号，那它们编码了什么？

Exp1: Top PCs与词频/位置/范数的相关性
  - 每个词的top-PC投影值 vs 词频(在语料中)
  - top-PC投影值 vs 向量范数
  - top-PC投影值 vs 位置信息
  - 逐PC分析: PC1, PC2, ...各编码什么？

Exp2: Top PCs的语义超类分析
  - 在top PCs空间中做层次聚类
  - 是否存在跨类别的"超类"(如: animate vs inanimate)?
  - top PCs是否编码了更抽象的语义维度?

Exp3: 质心偏移方向分析
  - 5个类别质心偏移向量的n90
  - 质心偏移是否在top PCs方向上？还是正交的？
  - 质心偏移之间的夹角结构

Exp4: 去除top PCs后的精细结构
  - 去除top-10/30/50 PCs后，剩余空间的类别编码
  - 剩余空间中每个类别的n90
  - 剩余空间中kNN准确率随去除PC数的变化
"""

import sys, os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxxiv_log.txt"

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        import time
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()

log("=== CCLXXXIV Script started ===")

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

# ===== 5类别×50词 =====
LARGE_CATEGORIES = {
    "animal": [
        "dog", "cat", "horse", "bird", "fish", "lion", "bear", "deer",
        "wolf", "fox", "rabbit", "snake", "eagle", "whale", "shark",
        "tiger", "monkey", "elephant", "giraffe", "zebra", "penguin",
        "dolphin", "owl", "hawk", "crow", "swan", "goose", "duck",
        "frog", "turtle", "crab", "ant", "bee", "spider", "worm",
        "mouse", "rat", "cow", "pig", "sheep", "goat", "donkey",
        "camel", "gorilla", "leopard", "cheetah", "salmon", "trout",
        "parrot", "robin",
    ],
    "food": [
        "apple", "bread", "cheese", "rice", "meat", "cake", "soup", "salt",
        "milk", "egg", "butter", "sugar", "flour", "honey", "cream",
        "pepper", "vinegar", "oil", "pork", "beef", "lamb", "ham",
        "bacon", "sausage", "chicken", "turkey", "duck_f", "salmon_f",
        "tuna", "shrimp", "lobster", "oyster", "mushroom", "onion",
        "garlic", "carrot", "potato", "tomato", "cabbage", "lettuce",
        "spinach", "celery", "pea", "bean", "lentil", "corn_f",
        "wheat_f", "oat_f", "barley_f", "rye_f",
    ],
    "body": [
        "hand", "foot", "head", "heart", "brain", "lung", "bone", "skin",
        "eye", "ear", "nose", "mouth", "tooth", "tongue", "lip",
        "neck", "shoulder", "arm", "elbow", "wrist", "finger", "thumb",
        "chest", "back", "spine", "hip", "knee", "ankle", "heel",
        "toe", "rib", "muscle", "vein", "nerve", "blood", "sweat",
        "tear", "hair", "nail", "palm", "fist", "belly", "throat",
        "jaw", "chin", "cheek", "brow", "lash", "pupil", "cornea",
    ],
    "tool": [
        "hammer", "knife", "scissors", "saw", "drill", "wrench", "chisel", "ruler",
        "screwdriver", "plier", "axe_t", "mallet", "clamp", "vise",
        "level", "square", "compass_t", "protractor", "caliper",
        "file", "rasp", "plane_t", "lathe", "mill", "press",
        "anvil", "tongs", "pliers", "crimper", "stapler",
        "tape_m", "welder", "solder", "torch_t", "grinder",
        "polisher", "sander", "shovel", "spade", "rake",
        "hoe", "trowel", "pruner", "shears", "hedge_cl",
        "bolt_cut", "wire_cut", "pipe_w", "socket", "hex_k",
    ],
    "clothing": [
        "shirt", "dress", "hat", "coat", "shoe", "belt", "scarf", "glove",
        "jacket", "sweater", "vest", "cape", "robe", "blouse",
        "skirt", "pants", "jeans", "shorts", "sock", "stocking",
        "boot", "sandal", "slipper", "sneaker", "loafer", "heel_c",
        "tie", "bow", "badge", "pin", "brooch", "collar",
        "cuff", "pocket", "zipper", "button", "buckle", "clasp",
        "hood", "visor", "brim", "crown_c", "apron", "overall",
        "uniform", "gown", "tuxedo", "kimono", "sari", "poncho",
    ],
}

# 语义属性标注
WORD_PROPERTIES = {
    # concreteness: 1=abstract, 5=concrete (rough estimates)
    # animacy: 0=inanimate, 1=animate
    # size: 1=small, 5=large
    # all animal words get animacy=1, others=0
}

# 词频估算(基于常见英文词频，取log10)
WORD_FREQ = {
    "dog": 4.2, "cat": 4.1, "horse": 3.9, "bird": 3.8, "fish": 3.8,
    "lion": 3.5, "bear": 3.6, "deer": 3.3, "wolf": 3.3, "fox": 3.4,
    "rabbit": 3.2, "snake": 3.4, "eagle": 3.1, "whale": 3.2, "shark": 3.2,
    "tiger": 3.4, "monkey": 3.3, "elephant": 3.4, "giraffe": 2.8, "zebra": 2.7,
    "penguin": 3.0, "dolphin": 3.1, "owl": 3.0, "hawk": 2.9, "crow": 3.0,
    "swan": 2.9, "goose": 3.0, "duck": 3.3, "frog": 3.0, "turtle": 3.0,
    "crab": 2.9, "ant": 3.2, "bee": 3.2, "spider": 3.1, "worm": 3.0,
    "mouse": 3.3, "rat": 3.2, "cow": 3.2, "pig": 3.3, "sheep": 3.1,
    "goat": 2.9, "donkey": 2.7, "camel": 2.8, "gorilla": 2.6, "leopard": 2.5,
    "cheetah": 2.4, "salmon": 2.8, "trout": 2.5, "parrot": 2.6, "robin": 2.5,
    "apple": 3.9, "bread": 3.7, "cheese": 3.5, "rice": 3.6, "meat": 3.6,
    "cake": 3.5, "soup": 3.5, "salt": 3.5, "milk": 3.7, "egg": 3.7,
    "butter": 3.3, "sugar": 3.4, "flour": 2.9, "honey": 3.2, "cream": 3.3,
    "pepper": 3.2, "vinegar": 2.6, "oil": 3.5, "pork": 3.0, "beef": 3.1,
    "lamb": 2.9, "ham": 3.0, "bacon": 3.0, "sausage": 2.9, "chicken": 3.6,
    "turkey": 3.1, "shrimp": 2.9, "mushroom": 2.9, "onion": 3.0, "garlic": 3.0,
    "carrot": 2.9, "potato": 3.1, "tomato": 3.1, "cabbage": 2.7, "lettuce": 2.7,
    "spinach": 2.6, "celery": 2.5, "pea": 2.8, "bean": 3.0,
    "hand": 4.2, "foot": 3.9, "head": 4.1, "heart": 3.8, "brain": 3.5,
    "lung": 2.9, "bone": 3.3, "skin": 3.5, "eye": 3.9, "ear": 3.7,
    "nose": 3.6, "mouth": 3.7, "tooth": 3.3, "tongue": 3.2, "lip": 3.0,
    "neck": 3.3, "shoulder": 3.3, "arm": 3.5, "elbow": 2.8, "wrist": 2.8,
    "finger": 3.3, "thumb": 2.9, "chest": 3.2, "back": 4.0, "spine": 2.7,
    "hip": 2.9, "knee": 3.0, "ankle": 2.6, "heel": 2.7, "toe": 2.7,
    "rib": 2.5, "muscle": 3.1, "vein": 2.7, "nerve": 2.9, "blood": 3.6,
    "sweat": 2.8, "tear": 3.1, "hair": 3.7, "nail": 3.0, "palm": 3.0,
    "fist": 2.8, "belly": 2.7, "throat": 3.0, "jaw": 2.6, "chin": 2.6,
    "cheek": 2.8, "brow": 2.6, "lash": 2.2, "pupil": 2.5, "cornea": 2.0,
    "hammer": 3.2, "knife": 3.5, "scissors": 3.0, "saw": 3.0, "drill": 2.9,
    "wrench": 2.5, "chisel": 2.2, "ruler": 2.9, "screwdriver": 2.4,
    "shovel": 2.6, "spade": 2.3, "rake": 2.3,
    "shirt": 3.6, "dress": 3.5, "hat": 3.4, "coat": 3.4, "shoe": 3.5,
    "belt": 3.1, "scarf": 2.7, "glove": 2.9, "jacket": 3.3, "sweater": 2.9,
    "vest": 2.6, "skirt": 3.0, "pants": 3.2, "jeans": 3.2, "sock": 2.9,
    "boot": 3.0, "sandal": 2.4, "slipper": 2.4, "sneaker": 2.5,
    "tie": 3.0, "button": 3.0, "zipper": 2.5, "hood": 2.5, "apron": 2.4,
    "uniform": 3.0, "gown": 2.5, "kimono": 2.0, "sari": 1.9, "poncho": 1.9,
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

        # ===== 收集250词数据 =====
        all_words, all_cats = [], []
        for cat, words in LARGE_CATEGORIES.items():
            all_words.extend(words[:50])
            all_cats.extend([cat] * 50)
        cat_names = sorted(set(all_cats))
        n_cats = len(cat_names)
        n_total = len(all_words)
        log(f"Total words: {n_total}, categories: {n_cats}")

        word2cat = {w: c for w, c in zip(all_words, all_cats)}
        cat2words = {}
        for w, c in zip(all_words, all_cats):
            if c not in cat2words:
                cat2words[c] = []
            cat2words[c].append(w)

        # ===== 前向推理收集数据 =====
        template = "The {} is"
        word_wdu = {}  # W_down @ W_up @ h (d_model维)
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
            if lw.W_gate is None: return [], []
            wdu_list, wnames = [], []
            for w in all_words:
                if li in word_wdu.get(w, {}):
                    wdu_list.append(word_wdu[w][li])
                    wnames.append(w)
            return wdu_list, wnames

        # ===== 结果保存 =====
        out_dir = Path(rf"d:\Ai2050\TransformerLens-Project\results\causal_fiber\{model_name}_cclxxxiv")
        out_dir.mkdir(parents=True, exist_ok=True)

        # 选择3个关键层(浅、中、深)
        if n_layers <= 30:
            sample_layers = [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]
        else:
            sample_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
        log(f"Sample layers: {sample_layers}")

        from scipy.sparse.linalg import svds
        from scipy.stats import spearmanr, pearsonr

        # ===================================================================
        # Exp1: Top PCs与词频/范数/位置的相关性
        # ===================================================================
        log("Exp1: Top PCs correlation with word properties...")
        exp1_results = []

        for li in sample_layers:
            wdu_vecs, wnames = get_wdu_for_layer(li)
            if len(wdu_vecs) < 50:
                continue

            X = np.array(wdu_vecs, dtype=np.float32)
            n = len(wdu_vecs)
            labels = [cat_names.index(word2cat[w]) for w in wnames]

            # 中心化 + SVD
            X_c = X - X.mean(axis=0)
            k_svd = min(100, n - 1, X_c.shape[1] - 1)
            if k_svd < 20:
                continue
            try:
                U, S, Vt = svds(X_c.astype(np.float32), k=k_svd)
            except:
                continue

            # 按奇异值排序
            order = np.argsort(S)[::-1]
            U = U[:, order]
            S = S[order]
            Vt = Vt[order]

            # 每个词在top PCs上的投影
            pc_scores = U @ np.diag(S)  # [n, k_svd]

            # 词频
            freqs = np.array([WORD_FREQ.get(w, 2.5) for w in wnames])

            # 向量范数
            norms = np.linalg.norm(X, axis=1)

            # 类别label
            label_arr = np.array(labels)

            # 动物=1, 其他=0 (animacy)
            animacy = np.array([1.0 if word2cat[w] == "animal" else 0.0 for w in wnames])

            # 对每个PC分析相关性
            pc_analysis = []
            for pc_idx in range(min(30, k_svd)):
                scores = pc_scores[:, pc_idx]

                # 与词频的相关
                freq_corr, freq_p = spearmanr(scores, freqs)

                # 与范数的相关
                norm_corr, norm_p = spearmanr(scores, norms)

                # 与类别的ANOVA-like: 各类别在该PC上的均值差异
                cat_means = {}
                for ci, cn in enumerate(cat_names):
                    mask = label_arr == ci
                    cat_means[cn] = float(np.mean(scores[mask]))

                # 类别可分性: 最大类间差异 / 类内std
                cat_mean_vals = list(cat_means.values())
                between_range = max(cat_mean_vals) - min(cat_mean_vals)
                within_std = float(np.mean([np.std(scores[label_arr == ci]) for ci in range(n_cats)]))
                cat_separability = between_range / within_std if within_std > 0 else 0

                # kNN准确率只用这个PC
                dist_pc = np.abs(scores[:, None] - scores[None, :])
                np.fill_diagonal(dist_pc, np.inf)
                correct_pc = 0
                for i in range(n):
                    nn = np.argmin(dist_pc[i])
                    if labels[nn] == labels[i]:
                        correct_pc += 1
                knn_acc_pc = correct_pc / n

                pc_analysis.append({
                    "pc_idx": pc_idx,
                    "singular_value": round(float(S[pc_idx]), 2),
                    "var_explained": round(float(S[pc_idx] ** 2 / (S ** 2).sum()), 6),
                    "freq_corr": round(float(freq_corr), 4),
                    "freq_p": round(float(freq_p), 6),
                    "norm_corr": round(float(norm_corr), 4),
                    "norm_p": round(float(norm_p), 6),
                    "cat_separability": round(float(cat_separability), 4),
                    "knn_acc": round(float(knn_acc_pc), 4),
                    "cat_means": {k: round(v, 4) for k, v in cat_means.items()},
                })

            # 汇总: 前10个PC的相关性
            top10_freq_corr = [p["freq_corr"] for p in pc_analysis[:10]]
            top10_norm_corr = [p["norm_corr"] for p in pc_analysis[:10]]
            top10_cat_sep = [p["cat_separability"] for p in pc_analysis[:10]]
            top10_knn = [p["knn_acc"] for p in pc_analysis[:10]]

            # 对比: 11-30个PC
            mid_freq_corr = [p["freq_corr"] for p in pc_analysis[10:30]]
            mid_norm_corr = [p["norm_corr"] for p in pc_analysis[10:30]]
            mid_cat_sep = [p["cat_separability"] for p in pc_analysis[10:30]]
            mid_knn = [p["knn_acc"] for p in pc_analysis[10:30]]

            exp1_results.append({
                "layer": li,
                "n_words": n,
                "pc_analysis": pc_analysis,
                "top10_mean_freq_corr": round(float(np.mean(np.abs(top10_freq_corr))), 4),
                "top10_mean_norm_corr": round(float(np.mean(np.abs(top10_norm_corr))), 4),
                "top10_mean_cat_sep": round(float(np.mean(top10_cat_sep)), 4),
                "top10_mean_knn": round(float(np.mean(top10_knn)), 4),
                "mid_mean_freq_corr": round(float(np.mean(np.abs(mid_freq_corr))) if mid_freq_corr else 0, 4),
                "mid_mean_norm_corr": round(float(np.mean(np.abs(mid_norm_corr))) if mid_norm_corr else 0, 4),
                "mid_mean_cat_sep": round(float(np.mean(mid_cat_sep)) if mid_cat_sep else 0, 4),
                "mid_mean_knn": round(float(np.mean(mid_knn)) if mid_knn else 0, 4),
            })

            log(f"  L{li}: top10 |freq|={np.mean(np.abs(top10_freq_corr)):.3f}, "
                f"|norm|={np.mean(np.abs(top10_norm_corr)):.3f}, "
                f"cat_sep={np.mean(top10_cat_sep):.3f}, knn={np.mean(top10_knn):.3f}")
            log(f"         mid   |freq|={np.mean(np.abs(mid_freq_corr)) if mid_freq_corr else 0:.3f}, "
                f"|norm|={np.mean(np.abs(mid_norm_corr)) if mid_norm_corr else 0:.3f}, "
                f"cat_sep={np.mean(mid_cat_sep) if mid_cat_sep else 0:.3f}, knn={np.mean(mid_knn) if mid_knn else 0:.3f}")

        with open(out_dir / "exp1_pc_correlations.json", 'w') as f:
            json.dump({"model": model_name, "results": exp1_results}, f, indent=2)
        log(f"Exp1 done: {len(exp1_results)} layers")

        # ===================================================================
        # Exp2: Top PCs的语义超类分析
        # ===================================================================
        log("Exp2: Top PCs semantic super-category analysis...")
        exp2_results = []

        for li in sample_layers:
            wdu_vecs, wnames = get_wdu_for_layer(li)
            if len(wdu_vecs) < 50:
                continue

            X = np.array(wdu_vecs, dtype=np.float32)
            n = len(wdu_vecs)
            labels = [cat_names.index(word2cat[w]) for w in wnames]

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

            # 在不同PC子空间中做kNN
            for d_proj in [3, 5, 10, 20, 50]:
                if d_proj > k_svd:
                    continue
                # 投影到top-d PCs
                X_top = U[:, :d_proj] @ np.diag(S[:d_proj])

                # kNN
                dist_top = np.zeros((n, n), dtype=np.float32)
                for i in range(n):
                    diff = X_top[i] - X_top
                    dist_top[i] = np.linalg.norm(diff, axis=1)
                np.fill_diagonal(dist_top, np.inf)

                correct = 0
                for i in range(n):
                    nn = np.argmin(dist_top[i])
                    if labels[nn] == labels[i]:
                        correct += 1
                knn_top = correct / n

                # 层次聚类: 在top-d PCs中
                from scipy.cluster.hierarchy import linkage, fcluster
                from scipy.spatial.distance import pdist

                # 用前d_proj个PC做聚类
                Z = linkage(X_top, method='ward')

                # 切成5类
                clusters_5 = fcluster(Z, t=5, criterion='maxclust')

                # 聚类纯度: 每个聚类中最多类别的比例
                cluster_purities = []
                cluster_dominant = []
                for cl in range(1, 6):
                    mask = clusters_5 == cl
                    if mask.sum() == 0:
                        continue
                    cl_labels = [labels[i] for i in range(n) if mask[i]]
                    unique, counts = np.unique(cl_labels, return_counts=True)
                    dominant = unique[np.argmax(counts)]
                    purity = counts.max() / len(cl_labels)
                    cluster_purities.append(purity)
                    cluster_dominant.append(cat_names[dominant])

                # NMI (normalized mutual information)
                from scipy.stats import contingency
                # 简单的聚类-标签匹配
                # 看top PCs是否发现了"超类"
                # animate (animal) vs inanimate (food, body, tool, clothing)
                animacy_labels = [1 if word2cat[wnames[i]] == "animal" else 0 for i in range(n)]

                # 在top-d PCs中，animacy的kNN准确率
                correct_anim = 0
                for i in range(n):
                    nn = np.argmin(dist_top[i])
                    if animacy_labels[nn] == animacy_labels[i]:
                        correct_anim += 1
                knn_anim = correct_anim / n

                # concrete vs less concrete (body+tool+clothing vs animal+food, roughly)
                # 用超类: biological (animal+body+food) vs artifact (tool+clothing)
                bio_labels = [1 if word2cat[wnames[i]] in ("animal", "body", "food") else 0 for i in range(n)]
                correct_bio = 0
                for i in range(n):
                    nn = np.argmin(dist_top[i])
                    if bio_labels[nn] == bio_labels[i]:
                        correct_bio += 1
                knn_bio = correct_bio / n

                exp2_results.append({
                    "layer": li,
                    "d_proj": d_proj,
                    "knn_5cat": round(knn_top, 4),
                    "knn_animacy": round(knn_anim, 4),
                    "knn_biological": round(knn_bio, 4),
                    "cluster_5_purity_mean": round(float(np.mean(cluster_purities)), 4) if cluster_purities else -1,
                    "cluster_dominant_cats": cluster_dominant,
                })

                log(f"  L{li} d={d_proj}: 5cat_knn={knn_top:.3f}, anim_knn={knn_anim:.3f}, bio_knn={knn_bio:.3f}, "
                    f"clust_purity={np.mean(cluster_purities):.3f}" if cluster_purities else "")

        with open(out_dir / "exp2_super_category.json", 'w') as f:
            json.dump({"model": model_name, "results": exp2_results}, f, indent=2)
        log(f"Exp2 done: {len(exp2_results)} entries")

        # ===================================================================
        # Exp3: 质心偏移方向分析
        # ===================================================================
        log("Exp3: Centroid shift direction analysis...")
        exp3_results = []

        for li in sample_layers:
            wdu_vecs, wnames = get_wdu_for_layer(li)
            if len(wdu_vecs) < 50:
                continue

            X = np.array(wdu_vecs, dtype=np.float32)
            n = len(wdu_vecs)

            # 全局质心
            global_centroid = X.mean(axis=0)

            # 每个类别的质心
            cat_centroids = {}
            cat_vecs = defaultdict(list)
            for v, w in zip(wdu_vecs, wnames):
                c = word2cat[w]
                cat_vecs[c].append(v)
            for c, vecs in cat_vecs.items():
                cat_centroids[c] = np.mean(vecs, axis=0)

            # 质心偏移向量 (相对于全局质心)
            centroid_shifts = {}
            for c in cat_names:
                centroid_shifts[c] = cat_centroids[c] - global_centroid

            # 质心偏移矩阵
            shift_matrix = np.array([centroid_shifts[c] for c in cat_names], dtype=np.float32)

            # 质心偏移的n90
            shift_c = shift_matrix - shift_matrix.mean(axis=0)
            k_shift = min(4, shift_c.shape[0] - 1, shift_c.shape[1] - 1)  # 只有5个质心
            if k_shift >= 2:
                try:
                    _, s_shift, Vt_shift = svds(shift_c.astype(np.float32), k=k_shift)
                    s_shift_sorted = np.sort(s_shift)[::-1]
                    var_shift = s_shift_sorted ** 2
                    total_shift = var_shift.sum()
                    if total_shift > 1e-10:
                        cum_shift = np.cumsum(var_shift) / total_shift
                        n90_shift = int(np.searchsorted(cum_shift, 0.90)) + 1
                    else:
                        n90_shift = -1
                except:
                    n90_shift = -1
            else:
                n90_shift = -1

            # 质心偏移在top PCs上的投影
            X_c = X - X.mean(axis=0)
            k_svd = min(100, n - 1, X_c.shape[1] - 1)
            if k_svd >= 20:
                try:
                    U, S, Vt = svds(X_c.astype(np.float32), k=k_svd)
                    order = np.argsort(S)[::-1]
                    U, S, Vt = U[:, order], S[order], Vt[order]

                    # 每个质心偏移在top PCs上的投影比例
                    shift_proj_in_top = {}
                    for d_top in [10, 30, 50]:
                        if d_top > k_svd:
                            continue
                        # Vt[:d_top]是top-d PCs的方向
                        # 质心偏移在这些方向上的投影
                        total_shift_norm = 0
                        projected_norm = 0
                        for c in cat_names:
                            shift = centroid_shifts[c]
                            total_shift_norm += np.linalg.norm(shift) ** 2
                            # 投影到top-d PCs
                            proj = Vt[:d_top] @ shift
                            projected_norm += np.linalg.norm(proj) ** 2
                        ratio = projected_norm / total_shift_norm if total_shift_norm > 0 else 0
                        shift_proj_in_top[f"top_{d_top}"] = round(float(ratio), 4)

                except:
                    shift_proj_in_top = {}
            else:
                shift_proj_in_top = {}

            # 质心偏移之间的夹角
            shift_angles = {}
            for i, c1 in enumerate(cat_names):
                for j, c2 in enumerate(cat_names):
                    if i >= j:
                        continue
                    v1 = centroid_shifts[c1]
                    v2 = centroid_shifts[c2]
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle_deg = float(np.degrees(np.arccos(cos_angle)))
                    shift_angles[f"{c1}_vs_{c2}"] = round(angle_deg, 1)

            exp3_results.append({
                "layer": li,
                "n90_centroid_shifts": n90_shift,
                "shift_proj_in_top_pcs": shift_proj_in_top,
                "shift_angles": shift_angles,
            })

            log(f"  L{li}: n90_shifts={n90_shift}, proj_in_top10={shift_proj_in_top.get('top_10', -1):.3f}")

        with open(out_dir / "exp3_centroid_shift.json", 'w') as f:
            json.dump({"model": model_name, "results": exp3_results}, f, indent=2)
        log(f"Exp3 done: {len(exp3_results)} layers")

        # ===================================================================
        # Exp4: 去除top PCs后的精细结构
        # ===================================================================
        log("Exp4: Fine structure after removing top PCs...")
        exp4_results = []

        for li in sample_layers:
            wdu_vecs, wnames = get_wdu_for_layer(li)
            if len(wdu_vecs) < 50:
                continue

            X = np.array(wdu_vecs, dtype=np.float32)
            n = len(wdu_vecs)
            labels = [cat_names.index(word2cat[w]) for w in wnames]

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

            # 逐个添加PC: d=1,2,...,100
            # kNN准确率随PC数的变化
            for d_proj in [1, 2, 3, 5, 10, 20, 30, 50, 100]:
                if d_proj > k_svd:
                    continue

                # 使用top-d PCs
                X_top = U[:, :d_proj] @ np.diag(S[:d_proj])

                dist_top = np.zeros((n, n), dtype=np.float32)
                for i in range(n):
                    diff = X_top[i] - X_top
                    dist_top[i] = np.linalg.norm(diff, axis=1)
                np.fill_diagonal(dist_top, np.inf)

                correct = 0
                for i in range(n):
                    nn = np.argmin(dist_top[i])
                    if labels[nn] == labels[i]:
                        correct += 1
                knn_top = correct / n

                # 使用去除top-d后的residual
                X_resid = X_c - (U[:, :d_proj] @ np.diag(S[:d_proj]) @ Vt[:d_proj])

                dist_resid = np.zeros((n, n), dtype=np.float32)
                for i in range(n):
                    diff = X_resid[i] - X_resid
                    dist_resid[i] = np.linalg.norm(diff, axis=1)
                np.fill_diagonal(dist_resid, np.inf)

                correct_r = 0
                for i in range(n):
                    nn = np.argmin(dist_resid[i])
                    if labels[nn] == labels[i]:
                        correct_r += 1
                knn_resid = correct_r / n

                # residual中每个类别的n90
                cat_n90s_resid = {}
                for c in cat_names:
                    c_words = [w for w in wnames if word2cat[w] == c]
                    c_idx = [wnames.index(w) for w in c_words]
                    if len(c_idx) < 20:
                        continue
                    arr = X_resid[c_idx]
                    arr_c = arr - arr.mean(axis=0)
                    k_cat = min(len(c_idx) - 1, arr_c.shape[1] - 1, 100)
                    if k_cat < 5:
                        continue
                    try:
                        _, s_c, _ = svds(arr_c.astype(np.float32), k=k_cat)
                        s_c_sorted = np.sort(s_c)[::-1]
                        var_c = s_c_sorted ** 2
                        total_c = var_c.sum()
                        if total_c > 1e-10:
                            cum_c = np.cumsum(var_c) / total_c
                            cat_n90s_resid[c] = int(np.searchsorted(cum_c, 0.90)) + 1
                    except:
                        pass

                var_in_top = float(S[:d_proj].sum() ** 2 / (S ** 2).sum()) if d_proj <= len(S) else 1.0

                exp4_results.append({
                    "layer": li,
                    "d_proj": d_proj,
                    "knn_top_d": round(knn_top, 4),
                    "knn_residual": round(knn_resid, 4),
                    "var_in_top_d": round(var_in_top, 4),
                    "cat_n90_in_residual": {k: v for k, v in cat_n90s_resid.items()},
                })

                log(f"  L{li} d={d_proj}: top_knn={knn_top:.3f}, resid_knn={knn_resid:.3f}")

        with open(out_dir / "exp4_remove_top_pcs.json", 'w') as f:
            json.dump({"model": model_name, "results": exp4_results}, f, indent=2)
        log(f"Exp4 done: {len(exp4_results)} entries")

        # ===== 释放模型 =====
        log("Releasing model...")
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        log(f"=== {model_name} done ===")

    except Exception as e:
        log(f"ERROR: {e}")
        traceback.print_exc()
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_model(model_name)
