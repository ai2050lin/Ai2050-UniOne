"""
Phase CCLXXXII: 大样本类别测试 — 打破per-category n90天花板
============================================================
核心问题: CCLXXXI发现per-category n90≈6是8词的天花板效应
关键实验: 5类别×50词=250词，测试类别子空间假说

Exp1: 大样本per-category维度
  - 5类别×50词，每类50个W_down@ū向量
  - per-category n90, n50, n95
  - 50个样本最大秩49，n90若=6则是真发现，若=40+则无子空间

Exp2: 类别子空间vs随机基线
  - 同一空间中随机选50个点，计算n90
  - 真实类别n90 vs 随机50点n90
  - 如果类别子空间存在：类别n90 << 随机n90

Exp3: kNN分类与错误分析
  - 5类×50词的leave-one-out kNN
  - 哪些词被分错？错误集中在类别边界？
  - k=1 vs k=5 vs k=11

Exp4: 跨层kNN稳定性
  - 同一个词在不同层的最近邻是否相同？
  - 邻居稳定性随深度如何变化？
  - 类别内邻居稳定性 vs 类别间邻居变化
"""

import sys, os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxxii_log.txt"

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        import time
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()

log("=== CCLXXXII Script started ===")

import json, time, gc, traceback
from pathlib import Path
import numpy as np

log("Importing torch...")
import torch

log("Importing model_utils...")
sys.path.insert(0, r"d:\Ai2050\TransformerLens-Project")
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, get_W_U,
    get_layer_weights, MODEL_CONFIGS,
)

# ===== 5类别×50词 =====
# 选5个语义边界清晰、词量丰富的类别
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
            all_words.extend(words[:50])  # 每类50词
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
        word_ups = {}
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
                u_dict[li] = u

            word_ups[word] = u_dict
            if (wi + 1) % 50 == 0 or wi == 0:
                log(f"  Word {wi+1}/{n_total} ({time.time()-t0:.0f}s)")

        log(f"Data collection done ({time.time()-t0:.0f}s)")

        # ===== 计算每层均值 =====
        layer_mean_u = {}
        for li in range(n_layers):
            us = [word_ups[w][li] for w in all_words if li in word_ups.get(w, {})]
            if us:
                layer_mean_u[li] = np.mean(us, axis=0)
        log("Layer means computed")

        # 辅助函数
        def get_wdu_for_layer(li):
            lw = get_layer_weights(layers_list[li], d_model, mlp_type)
            if lw.W_gate is None: return [], []
            wdu_list, wnames = [], []
            for w in all_words:
                if li in word_ups.get(w, {}):
                    wdu_list.append(lw.W_down @ word_ups[w][li])
                    wnames.append(w)
            return wdu_list, wnames

        def get_wd_du_for_layer(li):
            if li not in layer_mean_u: return [], []
            lw = get_layer_weights(layers_list[li], d_model, mlp_type)
            if lw.W_gate is None: return [], []
            du_list, wnames = [], []
            mu = layer_mean_u[li]
            for w in all_words:
                if li in word_ups.get(w, {}):
                    du = word_ups[w][li] - mu
                    du_list.append(lw.W_down @ du)
                    wnames.append(w)
            return du_list, wnames

        # ===== 结果保存目录 =====
        out_dir = Path(rf"d:\Ai2050\TransformerLens-Project\results\causal_fiber\{model_name}_cclxxxii")
        out_dir.mkdir(parents=True, exist_ok=True)

        # 选择采样层
        sample_layers = list(range(0, n_layers, max(1, n_layers // 10)))
        if n_layers - 1 not in sample_layers:
            sample_layers.append(n_layers - 1)
        log(f"Sample layers: {sample_layers}")

        from scipy.sparse.linalg import svds

        # ===== Exp1: 大样本per-category维度 =====
        log("Exp1: Large-sample per-category dimensionality...")
        exp1_results = []

        for li in sample_layers:
            wdu_vecs, wnames = get_wdu_for_layer(li)
            if len(wdu_vecs) < 50:
                continue

            # 按类别分组
            cat_vecs = {}
            for v, w in zip(wdu_vecs, wnames):
                c = word2cat[w]
                if c not in cat_vecs:
                    cat_vecs[c] = []
                cat_vecs[c].append(v)

            # 每个类别的子空间维度(50个样本, 最大秩49)
            cat_dims = {}
            for c, vecs in cat_vecs.items():
                if len(vecs) < 20:  # 至少20个样本
                    cat_dims[c] = {"n50": -1, "n90": -1, "n95": -1, "n": len(vecs)}
                    continue
                arr = np.array(vecs, dtype=np.float32)
                arr_c = arr - arr.mean(axis=0)
                k = min(len(vecs) - 1, arr_c.shape[1] - 1, 100)
                if k < 5:
                    cat_dims[c] = {"n50": -1, "n90": -1, "n95": -1, "n": len(vecs)}
                    continue
                try:
                    _, s, _ = svds(arr_c.astype(np.float32), k=k)
                    s_sorted = np.sort(s)[::-1]
                    var_exp = s_sorted ** 2
                    total = var_exp.sum()
                    if total < 1e-10:
                        cat_dims[c] = {"n50": -1, "n90": -1, "n95": -1, "n": len(vecs)}
                        continue
                    cum = np.cumsum(var_exp) / total
                    n50 = int(np.searchsorted(cum, 0.50)) + 1
                    n90 = int(np.searchsorted(cum, 0.90)) + 1
                    n95 = int(np.searchsorted(cum, 0.95)) + 1
                    # 前5个主成分的方差占比
                    top5 = float(var_exp[:5].sum() / total)
                    cat_dims[c] = {"n50": n50, "n90": n90, "n95": n95, "n": len(vecs), "top5_var": round(top5, 4)}
                except Exception as e:
                    log(f"  L{li} cat={c} SVD failed: {e}")
                    cat_dims[c] = {"n50": -1, "n90": -1, "n95": -1, "n": len(vecs)}

            # 全局维度(所有250词)
            X_all = np.array(wdu_vecs, dtype=np.float32)
            X_all_c = X_all - X_all.mean(axis=0)
            k_global = min(100, X_all_c.shape[0] - 1, X_all_c.shape[1] - 1)
            global_n50 = global_n90 = global_n95 = -1
            global_top5 = -1
            if k_global >= 5:
                try:
                    _, s_g, _ = svds(X_all_c.astype(np.float32), k=k_global)
                    s_g_sorted = np.sort(s_g)[::-1]
                    var_g = s_g_sorted ** 2
                    total_g = var_g.sum()
                    if total_g > 1e-10:
                        cum_g = np.cumsum(var_g) / total_g
                        global_n50 = int(np.searchsorted(cum_g, 0.50)) + 1
                        global_n90 = int(np.searchsorted(cum_g, 0.90)) + 1
                        global_n95 = int(np.searchsorted(cum_g, 0.95)) + 1
                        global_top5 = float(var_g[:5].sum() / total_g)
                except:
                    pass

            exp1_results.append({
                "layer": li,
                "cat_dims": cat_dims,
                "global_n50": global_n50,
                "global_n90": global_n90,
                "global_n95": global_n95,
                "global_top5_var": round(global_top5, 4) if global_top5 >= 0 else -1,
            })

            # 打印
            cat_n90s = [v["n90"] for v in cat_dims.values() if v["n90"] > 0]
            if cat_n90s:
                log(f"  L{li}: global_n90={global_n90}, cat_n90_mean={np.mean(cat_n90s):.1f} "
                    f"(min={min(cat_n90s)}, max={max(cat_n90s)})")
            else:
                log(f"  L{li}: insufficient data")

        with open(out_dir / "exp1_large_cat_dims.json", 'w') as f:
            json.dump({"model": model_name, "results": exp1_results}, f, indent=2)
        log(f"Exp1 done: {len(exp1_results)} layers")

        # ===== Exp2: 类别子空间 vs 随机基线 =====
        log("Exp2: Category subspace vs random baseline...")
        exp2_results = []

        for li in sample_layers:
            wdu_vecs, wnames = get_wdu_for_layer(li)
            if len(wdu_vecs) < 50:
                continue

            X = np.array(wdu_vecs, dtype=np.float32)
            n = len(wdu_vecs)

            # 真实类别的n90
            cat_vecs = {}
            for v, w in zip(wdu_vecs, wnames):
                c = word2cat[w]
                if c not in cat_vecs:
                    cat_vecs[c] = []
                cat_vecs[c].append(v)

            real_cat_n90s = {}
            for c, vecs in cat_vecs.items():
                if len(vecs) < 20:
                    continue
                arr = np.array(vecs, dtype=np.float32)
                arr_c = arr - arr.mean(axis=0)
                k = min(len(vecs) - 1, arr_c.shape[1] - 1, 100)
                try:
                    _, s, _ = svds(arr_c.astype(np.float32), k=k)
                    s_sorted = np.sort(s)[::-1]
                    var_exp = s_sorted ** 2
                    total = var_exp.sum()
                    if total > 1e-10:
                        cum = np.cumsum(var_exp) / total
                        real_cat_n90s[c] = int(np.searchsorted(cum, 0.90)) + 1
                except:
                    pass

            # 随机基线: 随机选50个点，重复10次
            n_random_trials = 10
            random_n90s = []
            rng = np.random.RandomState(42)
            for trial in range(n_random_trials):
                # 从所有点中随机选50个
                idx = rng.choice(n, 50, replace=False)
                rand_vecs = X[idx]
                rand_c = rand_vecs - rand_vecs.mean(axis=0)
                k_rand = min(49, rand_c.shape[1] - 1, 100)
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

            # 同类别子空间间的Grassmann距离
            # 计算每对类别的子空间重叠度
            cat_subspace_overlap = {}
            cat_list = [c for c in cat_names if c in real_cat_n90s]
            for i, c1 in enumerate(cat_list):
                for j, c2 in enumerate(cat_list):
                    if i >= j:
                        continue
                    v1 = cat_vecs[c1]
                    v2 = cat_vecs[c2]
                    if len(v1) < 20 or len(v2) < 20:
                        continue
                    a1 = np.array(v1, dtype=np.float32)
                    a2 = np.array(v2, dtype=np.float32)
                    a1c = a1 - a1.mean(axis=0)
                    a2c = a2 - a2.mean(axis=0)
                    # 各取top-10主成分方向
                    k_sub = min(10, len(v1) - 1, len(v2) - 1, a1c.shape[1] - 1)
                    if k_sub < 3:
                        continue
                    try:
                        _, s1, Vt1 = svds(a1c.astype(np.float32), k=k_sub)
                        _, s2, Vt2 = svds(a2c.astype(np.float32), k=k_sub)
                        Vt1_top = Vt1[np.argsort(s1)[-k_sub:]]
                        Vt2_top = Vt2[np.argsort(s2)[-k_sub:]]
                        # 子空间重叠: |V1^T V2| 的Frobenius范数 / sqrt(k)
                        overlap_mat = Vt1_top @ Vt2_top.T
                        grassmann_dist = float(np.linalg.norm(overlap_mat, 'fro') / np.sqrt(k_sub))
                        cat_subspace_overlap[f"{c1}_vs_{c2}"] = round(grassmann_dist, 4)
                    except:
                        pass

            real_vals = list(real_cat_n90s.values())
            exp2_results.append({
                "layer": li,
                "real_cat_n90s": real_cat_n90s,
                "real_cat_n90_mean": float(np.mean(real_vals)) if real_vals else -1,
                "real_cat_n90_std": float(np.std(real_vals)) if real_vals else -1,
                "random_n90_mean": float(np.mean(random_n90s)) if random_n90s else -1,
                "random_n90_std": float(np.std(random_n90s)) if random_n90s else -1,
                "ratio_real_to_random": round(float(np.mean(real_vals) / np.mean(random_n90s)), 4) if real_vals and random_n90s else -1,
                "n_random_trials": len(random_n90s),
                "subspace_overlap": cat_subspace_overlap,
            })

            if real_vals and random_n90s:
                log(f"  L{li}: real_n90={np.mean(real_vals):.1f}, random_n90={np.mean(random_n90s):.1f}, "
                    f"ratio={np.mean(real_vals)/np.mean(random_n90s):.3f}")

        with open(out_dir / "exp2_cat_vs_random.json", 'w') as f:
            json.dump({"model": model_name, "results": exp2_results}, f, indent=2)
        log(f"Exp2 done: {len(exp2_results)} layers")

        # ===== Exp3: kNN分类与错误分析 =====
        log("Exp3: kNN classification with error analysis...")
        exp3_results = []

        for li in sample_layers:
            wdu_vecs, wnames = get_wdu_for_layer(li)
            if len(wdu_vecs) < 50:
                continue

            X = np.array(wdu_vecs, dtype=np.float32)
            n = len(wdu_vecs)
            labels = [cat_names.index(word2cat[w]) for w in wnames]

            # 距离矩阵(一次性计算)
            # 分块计算以节省内存
            dist_mat = np.zeros((n, n), dtype=np.float32)
            for i in range(n):
                diff = X[i] - X
                dist_mat[i] = np.linalg.norm(diff, axis=1)
            np.fill_diagonal(dist_mat, np.inf)

            for k in [1, 3, 5, 11, 21]:
                correct = 0
                errors = []  # (word, true_cat, pred_cat, dist_to_nn)
                for i in range(n):
                    nn_idx = np.argpartition(dist_mat[i], k)[:k]
                    nn_labels = [labels[j] for j in nn_idx]
                    from collections import Counter
                    vote = Counter(nn_labels)
                    predicted = vote.most_common(1)[0][0]
                    if predicted == labels[i]:
                        correct += 1
                    else:
                        pred_cat = cat_names[predicted]
                        true_cat = cat_names[labels[i]]
                        # 找最近邻的实际距离
                        nn1_idx = np.argmin(dist_mat[i])
                        errors.append({
                            "word": wnames[i],
                            "true_cat": true_cat,
                            "pred_cat": pred_cat,
                            "dist_to_nn1": float(dist_mat[i][nn1_idx]),
                        })

                acc = correct / n
                random_acc = 1.0 / n_cats

                # 错误模式分析
                error_by_true_cat = {}
                error_by_pred_cat = {}
                for e in errors:
                    error_by_true_cat[e["true_cat"]] = error_by_true_cat.get(e["true_cat"], 0) + 1
                    error_by_pred_cat[e["pred_cat"]] = error_by_pred_cat.get(e["pred_cat"], 0) + 1

                exp3_results.append({
                    "layer": li,
                    "k": k,
                    "knn_acc": round(acc, 4),
                    "random_acc": round(random_acc, 4),
                    "lift": round(acc / random_acc, 2) if random_acc > 0 else -1,
                    "n_errors": len(errors),
                    "error_by_true_cat": error_by_true_cat,
                    "error_by_pred_cat": error_by_pred_cat,
                    "top_errors": errors[:10],  # 只保存前10个错误
                })
                log(f"  L{li} k={k}: acc={acc:.3f} ({correct}/{n}), errors: {error_by_true_cat}")

        with open(out_dir / "exp3_knn_errors.json", 'w') as f:
            json.dump({"model": model_name, "results": exp3_results}, f, indent=2)
        log(f"Exp3 done: {len(exp3_results)} entries")

        # ===== Exp4: 跨层kNN稳定性 =====
        log("Exp4: Cross-layer kNN stability...")
        exp4_results = []

        # 选2个参考层做详细分析
        ref_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]
        ref_layers = [l for l in ref_layers if l < n_layers]

        # 先收集所有层的kNN邻居(每个词的top-5邻居)
        layer_neighbors = {}  # layer -> {word_idx: [top5_neighbor_indices]}
        for li in sample_layers:
            wdu_vecs, wnames = get_wdu_for_layer(li)
            if len(wdu_vecs) < 50:
                continue
            X = np.array(wdu_vecs, dtype=np.float32)
            n = len(wdu_vecs)
            neighbors = {}
            for i in range(n):
                dists = np.linalg.norm(X - X[i], axis=1)
                dists[i] = np.inf
                top5 = np.argsort(dists)[:5]
                neighbors[i] = top5.tolist()
            layer_neighbors[li] = neighbors

        # 计算跨层邻居重叠度
        layer_list = sorted(layer_neighbors.keys())
        for i, li1 in enumerate(layer_list):
            for li2 in layer_list[i+1:]:
                if li2 - li1 < 3:  # 只比较相隔至少3层的
                    continue
                # 计算每个词的邻居重叠度
                overlaps = []
                cat_stability = {c: [] for c in cat_names}
                for idx in range(len(layer_neighbors[li1])):
                    n1 = set(layer_neighbors[li1][idx])
                    n2 = set(layer_neighbors[li2].get(idx, []))
                    if not n2:
                        continue
                    overlap = len(n1 & n2) / len(n1)
                    overlaps.append(overlap)
                    w = all_words[idx]
                    c = word2cat[w]
                    if c in cat_stability:
                        cat_stability[c].append(overlap)

                if not overlaps:
                    continue

                cat_mean_stab = {}
                for c, vals in cat_stability.items():
                    if vals:
                        cat_mean_stab[c] = round(float(np.mean(vals)), 4)

                exp4_results.append({
                    "layer1": li1,
                    "layer2": li2,
                    "layer_gap": li2 - li1,
                    "mean_overlap": round(float(np.mean(overlaps)), 4),
                    "std_overlap": round(float(np.std(overlaps)), 4),
                    "cat_stability": cat_mean_stab,
                })
                log(f"  L{li1}-L{li2}: mean_overlap={np.mean(overlaps):.3f}")

        with open(out_dir / "exp4_cross_layer_stability.json", 'w') as f:
            json.dump({"model": model_name, "results": exp4_results}, f, indent=2)
        log(f"Exp4 done: {len(exp4_results)} pairs")

        # ===== 释放 =====
        del word_ups
        del layer_mean_u
        gc.collect()
        release_model(model)

        log(f"=== {model_name} ALL DONE ===")

    except Exception as e:
        log(f"ERROR: {e}")
        traceback.print_exc()
        try:
            gc.collect()
            release_model(model)
        except:
            pass


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        sys.exit(1)

    # 清空日志
    with open(LOG, 'w') as f:
        f.write("")

    run_model(model_name)
