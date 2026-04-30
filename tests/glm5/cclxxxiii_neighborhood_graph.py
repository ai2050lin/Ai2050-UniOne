"""
Phase CCLXXXIII: 邻域图结构与距离分解
=============================================
核心问题: CCLXXXII发现类别不是低维子空间(ratio≈1.0)，但kNN仍有91-94%准确率
关键实验: 如果类别不是子空间，那语义编码的几何结构是什么？

Exp1: kNN图与社区结构
  - 构建250词的kNN图(k=5,10)
  - 图的连通分量 vs 语义类别
  - 类内边 vs 类间边的比例
  - Community detection (Louvain) → 检测社区是否对应类别

Exp2: 距离精细分解
  - within距离 vs between距离的来源
  - 质心偏移贡献(centroid shift)
  - 协方差结构差异贡献
  - 对齐到全局主成分后是否仍有类别信号

Exp3: residual stream空间对比
  - 在residual stream空间中做同样的类别n90 vs 随机n90测试
  - residual stream中的类别编码结构是否与W_down@ū不同？
  - 两个空间中kNN准确率的对比

Exp4: 局部邻域纯度
  - 对每个词，计算其k近邻中同类别词的比例(邻域纯度)
  - 高纯度词 vs 低纯度词的特征
  - 邻域纯度随深度如何变化
"""

import sys, os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxxiii_log.txt"

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        import time
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()

log("=== CCLXXXIII Script started ===")

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

# ===== 5类别×50词 (与CCLXXXII一致) =====
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
        word_ups = {}     # W_down @ u (d_model维)
        word_resid = {}   # residual stream (d_model维)
        t0 = time.time()

        for wi, word in enumerate(all_words):
            text = template.format(word)
            input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
            last_pos = input_ids.shape[1] - 1

            ln_out = {}
            resid_out = {}
            hooks = []
            for li in range(n_layers):
                layer = layers_list[li]
                # MLP前(即residual stream在LN后的值)
                if hasattr(layer, 'mlp'):
                    def make_ffn_pre(key):
                        def hook(module, args):
                            a = args[0] if not isinstance(args, tuple) else args[0]
                            ln_out[key] = a[0, last_pos].detach().float().cpu().numpy()
                        return hook
                    hooks.append(layer.mlp.register_forward_pre_hook(make_ffn_pre(f"L{li}")))
                # post-attention layernorm input = residual stream
                if hasattr(layer, 'post_attention_layernorm') or hasattr(layer, 'input_layernorm'):
                    ln_type = 'post_attention_layernorm' if hasattr(layer, 'post_attention_layernorm') else 'input_layernorm'
                    def make_resid_hook(key):
                        def hook(module, args):
                            a = args[0] if not isinstance(args, tuple) else args[0]
                            resid_out[key] = a[0, last_pos].detach().float().cpu().numpy()
                        return hook
                    # hook on the layernorm
                    ln_module = getattr(layer, ln_type)
                    hooks.append(ln_module.register_forward_pre_hook(make_resid_hook(f"L{li}")))

            with torch.no_grad():
                _ = model(input_ids)
            for h in hooks:
                h.remove()

            u_dict = {}
            resid_dict = {}
            for li in range(n_layers):
                key = f"L{li}"
                lw = get_layer_weights(layers_list[li], d_model, mlp_type)
                if lw.W_gate is None or key not in ln_out:
                    continue
                h_input = ln_out[key]
                u = lw.W_up @ h_input
                wdu = lw.W_down @ u
                u_dict[li] = wdu

                # residual stream
                if key in resid_out:
                    resid_dict[li] = resid_out[key]

            word_ups[word] = u_dict
            word_resid[word] = resid_dict
            if (wi + 1) % 50 == 0 or wi == 0:
                log(f"  Word {wi+1}/{n_total} ({time.time()-t0:.0f}s)")

        log(f"Data collection done ({time.time()-t0:.0f}s)")

        # ===== 计算每层均值 =====
        layer_mean_u = {}
        layer_mean_resid = {}
        for li in range(n_layers):
            us = [word_ups[w][li] for w in all_words if li in word_ups.get(w, {})]
            if us:
                layer_mean_u[li] = np.mean(us, axis=0)
            rs = [word_resid[w][li] for w in all_words if li in word_resid.get(w, {})]
            if rs:
                layer_mean_resid[li] = np.mean(rs, axis=0)
        log("Layer means computed")

        # 辅助函数
        def get_wdu_for_layer(li):
            lw = get_layer_weights(layers_list[li], d_model, mlp_type)
            if lw.W_gate is None: return [], []
            wdu_list, wnames = [], []
            for w in all_words:
                if li in word_ups.get(w, {}):
                    wdu_list.append(word_ups[w][li])
                    wnames.append(w)
            return wdu_list, wnames

        def get_resid_for_layer(li):
            resid_list, wnames = [], []
            for w in all_words:
                if li in word_resid.get(w, {}):
                    resid_list.append(word_resid[w][li])
                    wnames.append(w)
            return resid_list, wnames

        # ===== 结果保存目录 =====
        out_dir = Path(rf"d:\Ai2050\TransformerLens-Project\results\causal_fiber\{model_name}_cclxxxiii")
        out_dir.mkdir(parents=True, exist_ok=True)

        # 选择采样层
        sample_layers = list(range(0, n_layers, max(1, n_layers // 8)))
        if n_layers - 1 not in sample_layers:
            sample_layers.append(n_layers - 1)
        log(f"Sample layers: {sample_layers}")

        from scipy.sparse.linalg import svds

        # ===================================================================
        # Exp1: kNN图与社区结构
        # ===================================================================
        log("Exp1: kNN graph and community structure...")
        exp1_results = []

        for li in sample_layers:
            wdu_vecs, wnames = get_wdu_for_layer(li)
            if len(wdu_vecs) < 50:
                continue

            X = np.array(wdu_vecs, dtype=np.float32)
            n = len(wdu_vecs)
            labels = np.array([cat_names.index(word2cat[w]) for w in wnames])

            # 距离矩阵
            dist_mat = np.zeros((n, n), dtype=np.float32)
            for i in range(n):
                diff = X[i] - X
                dist_mat[i] = np.linalg.norm(diff, axis=1)
            np.fill_diagonal(dist_mat, np.inf)

            # kNN图分析 (k=5和k=10)
            for k_nn in [5, 10]:
                # 构建邻接表
                adj = {}
                within_edges = 0
                between_edges = 0
                for i in range(n):
                    nn_idx = np.argpartition(dist_mat[i], k_nn)[:k_nn]
                    adj[i] = nn_idx.tolist()
                    for j in nn_idx:
                        if labels[i] == labels[j]:
                            within_edges += 1
                        else:
                            between_edges += 1

                total_edges = within_edges + between_edges
                within_ratio = within_edges / total_edges if total_edges > 0 else 0

                # 随机基线: 随机k近邻的within_ratio
                n_random = 100
                random_within_ratios = []
                rng = np.random.RandomState(42)
                for _ in range(n_random):
                    r_within = 0
                    r_between = 0
                    for i in range(n):
                        random_nn = rng.choice(n, k_nn, replace=False)
                        for j in random_nn:
                            if labels[i] == labels[j]:
                                r_within += 1
                            else:
                                r_between += 1
                    r_total = r_within + r_between
                    random_within_ratios.append(r_within / r_total if r_total > 0 else 0)

                # 连通分量分析 (简单BFS)
                visited = [False] * n
                components = []
                for start in range(n):
                    if visited[start]:
                        continue
                    comp = []
                    queue = [start]
                    visited[start] = True
                    while queue:
                        node = queue.pop(0)
                        comp.append(node)
                        for neighbor in adj.get(node, []):
                            if not visited[neighbor]:
                                visited[neighbor] = True
                                queue.append(neighbor)
                    components.append(comp)

                # 每个连通分量的类别纯度
                comp_purities = []
                comp_dominant_cats = []
                for comp in components:
                    comp_labels = labels[comp]
                    if len(comp) == 0:
                        continue
                    unique, counts = np.unique(comp_labels, return_counts=True)
                    dominant = unique[np.argmax(counts)]
                    purity = counts.max() / len(comp)
                    comp_purities.append(purity)
                    comp_dominant_cats.append(int(dominant))

                # 跨类别桥接边分析
                # 哪些类别对之间有最多的边
                cross_cat_edges = defaultdict(int)
                for i in range(n):
                    for j in adj.get(i, []):
                        if labels[i] != labels[j]:
                            pair = tuple(sorted([int(labels[i]), int(labels[j])]))
                            cross_cat_edges[pair] += 1

                exp1_results.append({
                    "layer": li,
                    "k_nn": k_nn,
                    "n_nodes": n,
                    "n_components": len(components),
                    "largest_comp_size": max(len(c) for c in components) if components else 0,
                    "within_edges": within_edges,
                    "between_edges": between_edges,
                    "within_ratio": round(within_ratio, 4),
                    "random_within_ratio_mean": round(float(np.mean(random_within_ratios)), 4),
                    "random_within_ratio_std": round(float(np.std(random_within_ratios)), 4),
                    "within_vs_random_ratio": round(within_ratio / np.mean(random_within_ratios), 2) if np.mean(random_within_ratios) > 0 else -1,
                    "comp_purity_mean": round(float(np.mean(comp_purities)), 4) if comp_purities else -1,
                    "comp_purity_std": round(float(np.std(comp_purities)), 4) if comp_purities else -1,
                    "comp_sizes": [len(c) for c in sorted(components, key=len, reverse=True)[:10]],
                    "cross_cat_edges_top5": {f"{cat_names[k[0]]}_vs_{cat_names[k[1]]}": v for k, v in sorted(cross_cat_edges.items(), key=lambda x: -x[1])[:5]},
                    "comp_dominant_cats": comp_dominant_cats[:20],
                })

                log(f"  L{li} k={k_nn}: within_ratio={within_ratio:.3f} "
                    f"(random={np.mean(random_within_ratios):.3f}), "
                    f"n_comp={len(components)}, purity={np.mean(comp_purities):.3f}")

        with open(out_dir / "exp1_knn_graph.json", 'w') as f:
            json.dump({"model": model_name, "results": exp1_results}, f, indent=2)
        log(f"Exp1 done: {len(exp1_results)} entries")

        # ===================================================================
        # Exp2: 距离精细分解
        # ===================================================================
        log("Exp2: Distance decomposition...")
        exp2_results = []

        for li in sample_layers:
            wdu_vecs, wnames = get_wdu_for_layer(li)
            if len(wdu_vecs) < 50:
                continue

            X = np.array(wdu_vecs, dtype=np.float32)
            n = len(wdu_vecs)
            labels = [cat_names.index(word2cat[w]) for w in wnames]

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

            # within距离 vs between距离
            within_dists = []
            between_dists = []
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(X[i] - X[j])
                    if labels[i] == labels[j]:
                        within_dists.append(d)
                    else:
                        between_dists.append(d)

            within_mean = np.mean(within_dists) if within_dists else 0
            between_mean = np.mean(between_dists) if between_dists else 0
            wb_ratio = within_mean / between_mean if between_mean > 0 else -1

            # 质心间距离 vs within距离
            centroid_dists = []
            for c1 in cat_names:
                for c2 in cat_names:
                    if c1 >= c2:
                        continue
                    cd = np.linalg.norm(cat_centroids[c1] - cat_centroids[c2])
                    centroid_dists.append(cd)
            centroid_mean = np.mean(centroid_dists) if centroid_dists else 0

            # 分解: within = within_to_centroid + centroid_to_centroid
            # 类内距离的期望 = E[||x_i - x_j||] where x_i, x_j in same cat
            # = E[||x_i - mu_c||] + E[||x_j - mu_c||] - correlated part
            # 简化: within_to_centroid (类内点到质心的平均距离)
            within_to_centroid = []
            for c, vecs in cat_vecs.items():
                mu = cat_centroids[c]
                for v in vecs:
                    within_to_centroid.append(np.linalg.norm(v - mu))
            wtc_mean = np.mean(within_to_centroid) if within_to_centroid else 0

            # 对齐到全局主成分后是否仍有类别信号
            X_c = X - global_centroid
            k_svd = min(100, n - 1, X_c.shape[1] - 1)
            if k_svd >= 5:
                try:
                    U, S, Vt = svds(X_c.astype(np.float32), k=k_svd)
                    # 投影到前d个主成分
                    for d_proj in [10, 30, 50, 100]:
                        if d_proj > k_svd:
                            continue
                        X_proj = U[:, :d_proj] @ np.diag(S[:d_proj])

                        # kNN准确率在投影空间
                        dist_proj = np.zeros((n, n), dtype=np.float32)
                        for i in range(n):
                            diff = X_proj[i] - X_proj
                            dist_proj[i] = np.linalg.norm(diff, axis=1)
                        np.fill_diagonal(dist_proj, np.inf)

                        correct = 0
                        for i in range(n):
                            nn_idx = np.argpartition(dist_proj[i], 1)[:1]
                            pred = labels[nn_idx[0]]
                            if pred == labels[i]:
                                correct += 1
                        knn_acc_proj = correct / n

                        # 去除前d个主成分后
                        # X_residual = X_c - X_proj (即用后k-d个主成分)
                        X_residual = X_c - (U[:, :d_proj] @ np.diag(S[:d_proj]) @ Vt[:d_proj])
                        dist_resid = np.zeros((n, n), dtype=np.float32)
                        for i in range(n):
                            diff = X_residual[i] - X_residual
                            dist_resid[i] = np.linalg.norm(diff, axis=1)
                        np.fill_diagonal(dist_resid, np.inf)

                        correct_r = 0
                        for i in range(n):
                            nn_idx = np.argpartition(dist_resid[i], 1)[:1]
                            pred = labels[nn_idx[0]]
                            if pred == labels[i]:
                                correct_r += 1
                        knn_acc_resid = correct_r / n

                        exp2_results.append({
                            "layer": li,
                            "type": "pc_alignment",
                            "d_proj": d_proj,
                            "knn_acc_top_d": round(knn_acc_proj, 4),
                            "knn_acc_residual": round(knn_acc_resid, 4),
                            "var_in_top_d": round(float(S[:d_proj].sum() / S.sum()), 4),
                        })

                        log(f"  L{li} d={d_proj}: top_d_acc={knn_acc_proj:.3f}, "
                            f"resid_acc={knn_acc_resid:.3f}, var={S[:d_proj].sum()/S.sum():.3f}")
                except Exception as e:
                    log(f"  L{li} PC alignment SVD failed: {e}")

            # 基础距离统计
            exp2_results.append({
                "layer": li,
                "type": "distance_stats",
                "within_mean": round(float(within_mean), 4),
                "between_mean": round(float(between_mean), 4),
                "wb_ratio": round(float(wb_ratio), 4),
                "centroid_mean_dist": round(float(centroid_mean), 4),
                "within_to_centroid_mean": round(float(wtc_mean), 4),
                "centroid_to_within_ratio": round(float(centroid_mean / wtc_mean), 4) if wtc_mean > 0 else -1,
            })

            log(f"  L{li}: within={within_mean:.3f}, between={between_mean:.3f}, "
                f"w/b={wb_ratio:.3f}, cent/within={centroid_mean/wtc_mean:.3f}" if wtc_mean > 0 else f"  L{li}: within={within_mean:.3f}, between={between_mean:.3f}")

        with open(out_dir / "exp2_distance_decomp.json", 'w') as f:
            json.dump({"model": model_name, "results": exp2_results}, f, indent=2)
        log(f"Exp2 done: {len(exp2_results)} entries")

        # ===================================================================
        # Exp3: Residual stream空间对比
        # ===================================================================
        log("Exp3: Residual stream space comparison...")
        exp3_results = []

        for li in sample_layers:
            # W_down@ū空间
            wdu_vecs, wdu_names = get_wdu_for_layer(li)
            # Residual stream空间
            resid_vecs, resid_names = get_resid_for_layer(li)

            if len(wdu_vecs) < 50 or len(resid_vecs) < 50:
                continue

            # 确保两个空间用的是同一批词
            common_words = set(wdu_names) & set(resid_names)
            if len(common_words) < 50:
                continue

            common_words = sorted(common_words)
            n_common = len(common_words)
            labels = [cat_names.index(word2cat[w]) for w in common_words]

            # 构建两个空间的向量矩阵
            wdu_idx = {w: i for i, w in enumerate(wdu_names)}
            resid_idx = {w: i for i, w in enumerate(resid_names)}

            X_wdu = np.array([wdu_vecs[wdu_idx[w]] for w in common_words], dtype=np.float32)
            X_resid = np.array([resid_vecs[resid_idx[w]] for w in common_words], dtype=np.float32)

            # 两个空间的kNN准确率
            for space_name, X_space in [("wdu", X_wdu), ("resid", X_resid)]:
                n = len(X_space)
                dist_mat = np.zeros((n, n), dtype=np.float32)
                for i in range(n):
                    diff = X_space[i] - X_space
                    dist_mat[i] = np.linalg.norm(diff, axis=1)
                np.fill_diagonal(dist_mat, np.inf)

                # kNN k=1
                correct = 0
                for i in range(n):
                    nn_idx = np.argpartition(dist_mat[i], 1)[:1]
                    pred = labels[nn_idx[0]]
                    if pred == labels[i]:
                        correct += 1
                knn_acc = correct / n

                # Per-category n90 vs random n90
                X_c = X_space - X_space.mean(axis=0)
                k_svd = min(100, n_common - 1, X_c.shape[1] - 1)

                # 真实类别n90
                cat_n90s = {}
                for c in cat_names:
                    c_words = [w for w in common_words if word2cat[w] == c]
                    c_idx = [common_words.index(w) for w in c_words]
                    if len(c_idx) < 20:
                        continue
                    arr = X_space[c_idx]
                    arr_c = arr - arr.mean(axis=0)
                    k_cat = min(len(c_idx) - 1, arr_c.shape[1] - 1, 100)
                    if k_cat < 5:
                        continue
                    try:
                        _, s, _ = svds(arr_c.astype(np.float32), k=k_cat)
                        s_sorted = np.sort(s)[::-1]
                        var_exp = s_sorted ** 2
                        total = var_exp.sum()
                        if total > 1e-10:
                            cum = np.cumsum(var_exp) / total
                            cat_n90s[c] = int(np.searchsorted(cum, 0.90)) + 1
                    except:
                        pass

                # 随机50点n90
                random_n90s = []
                rng = np.random.RandomState(42)
                for _ in range(10):
                    idx = rng.choice(n_common, 50, replace=False)
                    rand_vecs = X_space[idx]
                    rand_c = rand_vecs - rand_vecs.mean(axis=0)
                    k_rand = min(49, rand_c.shape[1] - 1, 100)
                    if k_rand < 5:
                        continue
                    try:
                        _, s_r, _ = svds(rand_c.astype(np.float32), k=k_rand)
                        s_r_sorted = np.sort(s_r)[::-1]
                        var_r = s_r_sorted ** 2
                        total_r = var_r.sum()
                        if total_r > 1e-10:
                            cum_r = np.cumsum(var_r) / total_r
                            random_n90s.append(int(np.searchsorted(cum_r, 0.90)) + 1)
                    except:
                        pass

                real_n90_mean = float(np.mean(list(cat_n90s.values()))) if cat_n90s else -1
                rand_n90_mean = float(np.mean(random_n90s)) if random_n90s else -1
                ratio = real_n90_mean / rand_n90_mean if rand_n90_mean > 0 else -1

                exp3_results.append({
                    "layer": li,
                    "space": space_name,
                    "n_words": n_common,
                    "knn_k1_acc": round(knn_acc, 4),
                    "cat_n90_mean": round(real_n90_mean, 1) if real_n90_mean > 0 else -1,
                    "random_n90_mean": round(rand_n90_mean, 1) if rand_n90_mean > 0 else -1,
                    "cat_vs_random_ratio": round(ratio, 4) if ratio > 0 else -1,
                })

                log(f"  L{li} {space_name}: knn={knn_acc:.3f}, cat_n90={real_n90_mean:.1f}, "
                    f"rand_n90={rand_n90_mean:.1f}, ratio={ratio:.3f}" if rand_n90_mean > 0 else f"  L{li} {space_name}: knn={knn_acc:.3f}")

        with open(out_dir / "exp3_resid_comparison.json", 'w') as f:
            json.dump({"model": model_name, "results": exp3_results}, f, indent=2)
        log(f"Exp3 done: {len(exp3_results)} entries")

        # ===================================================================
        # Exp4: 局部邻域纯度
        # ===================================================================
        log("Exp4: Local neighborhood purity...")
        exp4_results = []

        for li in sample_layers:
            wdu_vecs, wnames = get_wdu_for_layer(li)
            if len(wdu_vecs) < 50:
                continue

            X = np.array(wdu_vecs, dtype=np.float32)
            n = len(wdu_vecs)
            labels = np.array([cat_names.index(word2cat[w]) for w in wnames])

            dist_mat = np.zeros((n, n), dtype=np.float32)
            for i in range(n):
                diff = X[i] - X
                dist_mat[i] = np.linalg.norm(diff, axis=1)
            np.fill_diagonal(dist_mat, np.inf)

            for k_nn in [5, 10]:
                # 每个词的邻域纯度
                purities = []
                word_purities = {}
                for i in range(n):
                    nn_idx = np.argpartition(dist_mat[i], k_nn)[:k_nn]
                    nn_labels = labels[nn_idx]
                    same = np.sum(nn_labels == labels[i])
                    purity = same / k_nn
                    purities.append(purity)
                    word_purities[wnames[i]] = round(float(purity), 4)

                # 按类别统计
                cat_purities = defaultdict(list)
                for i, w in enumerate(wnames):
                    cat_purities[word2cat[w]].append(purities[i])

                # 找出纯度最低和最高的词
                sorted_by_purity = sorted(word_purities.items(), key=lambda x: x[1])
                lowest_10 = sorted_by_purity[:10]
                highest_10 = sorted_by_purity[-10:]

                exp4_results.append({
                    "layer": li,
                    "k_nn": k_nn,
                    "purity_mean": round(float(np.mean(purities)), 4),
                    "purity_std": round(float(np.std(purities)), 4),
                    "purity_median": round(float(np.median(purities)), 4),
                    "cat_purity_means": {c: round(float(np.mean(ps)), 4) for c, ps in cat_purities.items()},
                    "lowest_10": [{w: p} for w, p in lowest_10],
                    "highest_10": [{w: p} for w, p in highest_10],
                    # 随机基线纯度
                    "random_purity": round(1.0 / n_cats, 4),  # 5类→0.2
                })

                log(f"  L{li} k={k_nn}: purity={np.mean(purities):.3f}±{np.std(purities):.3f} "
                    f"(random={1/n_cats:.3f})")

        with open(out_dir / "exp4_neighborhood_purity.json", 'w') as f:
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
