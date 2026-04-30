"""
Phase CCLXXXI: 偏移向量精细结构 — 纯经验数据积累
===================================================
核心目标: 积累偏移向量(W_down@Δu)的精细结构数据

Exp1: 完整PCA特征谱 — 每层top-100特征值(方差占比)
  - 不只看n50/n90/n95，保存完整的特征值曲线
  - 特征值衰减模式: 指数? 幂律? 阶梯?

Exp2: 词间距离分布 — within/between类别
  - 256词的完整距离矩阵统计
  - within-category和between-category距离的均值/中位数/分位数
  - 两种距离分布的重叠度

Exp3: kNN分类 — k=1,3,5,7
  - Leave-one-out kNN在W_down@ū空间
  - 对比最近质心(euc_acc)和kNN
  - 局部结构 vs 全局结构

Exp4: 每类别的子空间维度
  - 32个类别各自8个δ向量的n90
  - 每个类别是否占据不同维度的子空间？
  - 类别间子空间重叠度
"""

import sys, os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxxi_log.txt"

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        import time
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()

log("=== CCLXXXI Script started ===")

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

# ===== 256词表 (同CCLXXX) =====
CONCEPTS_256 = {
    "animal":   ["dog", "cat", "horse", "bird", "fish", "lion", "bear", "deer"],
    "food":     ["apple", "bread", "cheese", "rice", "meat", "cake", "soup", "salt"],
    "tool":     ["hammer", "knife", "scissors", "saw", "drill", "wrench", "chisel", "ruler"],
    "vehicle":  ["car", "bus", "train", "plane", "boat", "truck", "bike", "ship"],
    "clothing": ["shirt", "dress", "hat", "coat", "shoe", "belt", "scarf", "glove"],
    "weather":  ["rain", "snow", "wind", "storm", "fog", "hail", "frost", "cloud"],
    "emotion":  ["joy", "fear", "anger", "hope", "love", "grief", "pride", "shame"],
    "building": ["house", "church", "tower", "bridge", "castle", "temple", "museum", "palace"],
    "color":    ["red", "blue", "green", "gold", "silver", "pink", "brown", "gray"],
    "plant":    ["tree", "flower", "grass", "bush", "vine", "weed", "moss", "fern"],
    "metal":    ["iron", "copper", "steel", "bronze", "brass", "tin", "zinc", "lead"],
    "sport":    ["soccer", "tennis", "boxing", "golf", "rugby", "skiing", "rowing", "fencing"],
    "music":    ["piano", "violin", "drum", "flute", "guitar", "harp", "trumpet", "organ"],
    "science":  ["atom", "cell", "gene", "orbit", "force", "mass", "wave", "ray"],
    "body":     ["hand", "foot", "head", "heart", "brain", "lung", "bone", "skin"],
    "time":     ["dawn", "noon", "dusk", "night", "spring", "summer", "autumn", "winter"],
    "furniture": ["chair", "table", "desk", "bed", "sofa", "shelf", "cabinet", "bench"],
    "weapon":   ["sword", "bow", "spear", "shield", "axe", "dart", "lance", "dagger"],
    "gem":      ["ruby", "pearl", "jade", "opal", "amber", "topaz", "onyx", "coral"],
    "fabric":   ["silk", "wool", "cotton", "linen", "velvet", "nylon", "lace", "denim"],
    "container": ["box", "cup", "bowl", "jar", "pot", "barrel", "basket", "crate"],
    "terrain":  ["hill", "lake", "river", "cliff", "valley", "cave", "desert", "island"],
    "fruit":    ["grape", "peach", "lemon", "plum", "melon", "cherry", "mango", "olive"],
    "insect":   ["ant", "bee", "fly", "moth", "wasp", "beetle", "spider", "worm"],
    "profession": ["doctor", "lawyer", "chef", "pilot", "nurse", "judge", "artist", "poet"],
    "material":  ["stone", "glass", "wood", "paper", "clay", "cement", "rubber", "wax"],
    "light":     ["lamp", "candle", "torch", "flare", "beacon", "lantern", "prism", "lens"],
    "season":    ["January", "March", "May", "July", "August", "October", "April", "June"],
    "ocean":     ["whale", "shark", "dolphin", "seal", "crab", "squid", "turtle", "eel"],
    "space":     ["star", "moon", "comet", "mars", "venus", "nebula", "quasar", "pulsar"],
    "sound":     ["bell", "horn", "chime", "echo", "boom", "whisper", "thunder", "hum"],
    "grain":     ["wheat", "corn", "oat", "barley", "rye", "millet", "rice_g", "sorghum"],
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
        W_U = get_W_U(model)
        log(f"Model loaded: {n_layers}L, d={d_model}, mlp={mlp_type}")

        # ===== 收集256词数据 =====
        template = "The {} is"
        rng = np.random.RandomState(42)
        all_words, all_cats = [], []
        for cat, words in CONCEPTS_256.items():
            sel = rng.choice(words, min(8, len(words)), replace=False)
            all_words.extend(sel.tolist() if hasattr(sel, 'tolist') else list(sel))
            all_cats.extend([cat] * len(sel))
        cat_names = sorted(set(all_cats))
        n_cats = len(cat_names)
        log(f"Total words: {len(all_words)}, categories: {n_cats}")

        # 建立word->category映射和category->words映射
        word2cat = {w: c for w, c in zip(all_words, all_cats)}
        cat2words = {}
        for w, c in zip(all_words, all_cats):
            if c not in cat2words:
                cat2words[c] = []
            cat2words[c].append(w)

        word_gates, word_ups = {}, {}
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

            g_dict, u_dict = {}, {}
            for li in range(n_layers):
                key = f"L{li}"
                lw = get_layer_weights(layers_list[li], d_model, mlp_type)
                if lw.W_gate is None or key not in ln_out:
                    continue
                h_input = ln_out[key]
                z = lw.W_gate @ h_input
                z_clipped = np.clip(z, -500, 500)
                g = 1.0 / (1.0 + np.exp(-z_clipped))
                u = lw.W_up @ h_input
                g_dict[li] = g
                u_dict[li] = u

            word_gates[word] = g_dict
            word_ups[word] = u_dict
            if (wi + 1) % 32 == 0 or wi == 0:
                log(f"  Word {wi+1}/{len(all_words)} ({time.time()-t0:.0f}s)")

        log(f"Data collection done ({time.time()-t0:.0f}s)")

        # 计算每层均值
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
        out_dir = Path(rf"d:\Ai2050\TransformerLens-Project\results\causal_fiber\{model_name}_cclxxxi")
        out_dir.mkdir(parents=True, exist_ok=True)

        # ===== Exp1: 完整PCA特征谱 =====
        log("Exp1: Full PCA spectrum...")
        from scipy.sparse.linalg import svds
        exp1_results = []
        # 每隔几层采样（减少计算量）
        sample_layers = list(range(0, n_layers, max(1, n_layers // 12)))
        if n_layers - 1 not in sample_layers:
            sample_layers.append(n_layers - 1)

        for li in sample_layers:
            du_vecs, _ = get_wd_du_for_layer(li)
            if len(du_vecs) < 16:
                continue

            X = np.array(du_vecs, dtype=np.float32)
            # 中心化
            X_c = X - X.mean(axis=0)

            # SVD获取特征值
            k = min(100, X_c.shape[0]-1, X_c.shape[1]-1)
            if k < 5:
                continue
            try:
                _, s, Vt = svds(X_c.astype(np.float32), k=k)
            except Exception as e:
                log(f"  L{li} SVD failed: {e}")
                continue

            s_sorted = np.sort(s)[::-1]
            var_explained = s_sorted ** 2
            total_var = var_explained.sum()
            if total_var < 1e-10:
                continue
            var_ratio = (var_explained / total_var).tolist()

            # 累积方差
            cum_var = np.cumsum(var_explained) / total_var
            n50 = int(np.searchsorted(cum_var, 0.50)) + 1
            n90 = int(np.searchsorted(cum_var, 0.90)) + 1
            n95 = int(np.searchsorted(cum_var, 0.95)) + 1

            # 特征值比: λ1/λ2, λ2/λ3, ..., λ10/λ50
            eig_ratios = []
            for i in range(min(10, len(s_sorted)-1)):
                if s_sorted[i+1] > 1e-10:
                    eig_ratios.append(float(s_sorted[i] / s_sorted[i+1]))
                else:
                    eig_ratios.append(-1.0)

            # 幂律指数拟合: var_i ∝ i^α (对log(var)做线性回归)
            top_k_fit = min(50, len(var_ratio))
            if top_k_fit >= 5:
                log_indices = np.log(np.arange(1, top_k_fit+1))
                log_vars = np.log(np.array(var_ratio[:top_k_fit]) + 1e-15)
                # 线性回归
                A = np.vstack([log_indices, np.ones(top_k_fit)]).T
                try:
                    slope, intercept = np.linalg.lstsq(A, log_vars, rcond=None)[0]
                    power_law_exp = float(slope)
                except:
                    power_law_exp = -1.0
            else:
                power_law_exp = -1.0

            exp1_results.append({
                "layer": li,
                "n_samples": len(du_vecs),
                "n50": n50,
                "n90": n90,
                "n95": n95,
                "top5_var_ratio": [round(v, 6) for v in var_ratio[:5]],
                "top10_var_ratio": [round(v, 6) for v in var_ratio[:10]],
                "top50_var_ratio": [round(v, 6) for v in var_ratio[:50]],
                "top100_var_ratio": [round(v, 6) for v in var_ratio[:100]],
                "eig_ratios_top10": [round(v, 4) for v in eig_ratios],
                "power_law_exp": round(power_law_exp, 4),
            })
            log(f"  L{li}: n50={n50}, n90={n90}, n95={n95}, power_exp={power_law_exp:.3f}")

        with open(out_dir / "exp1_pca_spectrum.json", 'w') as f:
            json.dump({"model": model_name, "results": exp1_results}, f, indent=2)
        log(f"Exp1 done: {len(exp1_results)} layers")

        # ===== Exp2: 词间距离分布 =====
        log("Exp2: Distance distribution...")
        exp2_results = []
        sample_layers2 = list(range(0, n_layers, max(1, n_layers // 10)))
        if n_layers - 1 not in sample_layers2:
            sample_layers2.append(n_layers - 1)

        for li in sample_layers2:
            wdu_vecs, wnames = get_wdu_for_layer(li)
            if len(wdu_vecs) < 16:
                continue

            X = np.array(wdu_vecs, dtype=np.float32)
            n = len(wnames)

            # 计算所有pairwise欧氏距离
            # 为节省内存，分块计算
            within_dists = []
            between_dists = []
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(X[i] - X[j])
                    ci = word2cat[wnames[i]]
                    cj = word2cat[wnames[j]]
                    if ci == cj:
                        within_dists.append(float(d))
                    else:
                        between_dists.append(float(d))

            if not within_dists or not between_dists:
                continue

            within_arr = np.array(within_dists)
            between_arr = np.array(between_dists)

            # 距离分布统计
            def dist_stats(arr, name):
                return {
                    f"{name}_mean": float(np.mean(arr)),
                    f"{name}_median": float(np.median(arr)),
                    f"{name}_std": float(np.std(arr)),
                    f"{name}_p10": float(np.percentile(arr, 10)),
                    f"{name}_p25": float(np.percentile(arr, 25)),
                    f"{name}_p75": float(np.percentile(arr, 75)),
                    f"{name}_p90": float(np.percentile(arr, 90)),
                    f"{name}_n": len(arr),
                }

            stats = {"layer": li}
            stats.update(dist_stats(within_arr, "within"))
            stats.update(dist_stats(between_arr, "between"))

            # 重叠度: within的第90分位 < between的第10分位?
            # 重叠度: within中有多大比例低于between的p10
            p10_between = np.percentile(between_arr, 10)
            stats["overlap_90_10"] = float((within_arr <= p10_between).mean()) if len(between_arr) > 0 else -1
            # 更有意义的重叠度量: 两个分布的Wasserstein距离
            try:
                from scipy.stats import wasserstein_distance
                stats["wasserstein"] = float(wasserstein_distance(within_arr, between_arr))
            except:
                stats["wasserstein"] = -1

            # Ratio: within_median / between_median
            if np.median(between_arr) > 0:
                stats["within_between_ratio"] = float(np.median(within_arr) / np.median(between_arr))
            else:
                stats["within_between_ratio"] = -1

            exp2_results.append(stats)
            log(f"  L{li}: within_median={np.median(within_arr):.2f}, between_median={np.median(between_arr):.2f}, ratio={stats['within_between_ratio']:.3f}")

        with open(out_dir / "exp2_distance_dist.json", 'w') as f:
            json.dump({"model": model_name, "results": exp2_results}, f, indent=2)
        log(f"Exp2 done: {len(exp2_results)} layers")

        # ===== Exp3: kNN分类 =====
        log("Exp3: kNN classification...")
        exp3_results = []
        sample_layers3 = list(range(0, n_layers, max(1, n_layers // 10)))
        if n_layers - 1 not in sample_layers3:
            sample_layers3.append(n_layers - 1)

        for li in sample_layers3:
            wdu_vecs, wnames = get_wdu_for_layer(li)
            if len(wdu_vecs) < 16:
                continue

            X = np.array(wdu_vecs, dtype=np.float32)
            n = len(wnames)
            labels = [cat_names.index(word2cat[w]) for w in wnames]

            # Leave-one-out kNN
            for k in [1, 3, 5, 7]:
                correct = 0
                for i in range(n):
                    # 计算到所有其他点的距离
                    dists = np.linalg.norm(X - X[i], axis=1)
                    dists[i] = np.inf  # 排除自身
                    # 找k个最近邻
                    nn_indices = np.argpartition(dists, k)[:k]
                    nn_labels = [labels[j] for j in nn_indices]
                    # 多数投票
                    from collections import Counter
                    vote = Counter(nn_labels)
                    predicted = vote.most_common(1)[0][0]
                    if predicted == labels[i]:
                        correct += 1

                acc = correct / n
                random_acc = 1.0 / n_cats

                exp3_results.append({
                    "layer": li,
                    "k": k,
                    "knn_acc": round(acc, 4),
                    "random_acc": round(random_acc, 4),
                    "lift": round(acc / random_acc, 2) if random_acc > 0 else -1,
                })
                log(f"  L{li} k={k}: knn_acc={acc:.3f} (random={random_acc:.3f}, lift={acc/random_acc:.1f}x)")

        with open(out_dir / "exp3_knn_probe.json", 'w') as f:
            json.dump({"model": model_name, "results": exp3_results}, f, indent=2)
        log(f"Exp3 done: {len(exp3_results)} entries")

        # ===== Exp4: 每类别的子空间维度 =====
        log("Exp4: Per-category subspace dimensionality...")
        exp4_results = []
        # 每隔几层采样
        sample_layers4 = list(range(0, n_layers, max(1, n_layers // 8)))
        if n_layers - 1 not in sample_layers4:
            sample_layers4.append(n_layers - 1)

        for li in sample_layers4:
            du_vecs, wnames = get_wd_du_for_layer(li)
            if len(du_vecs) < 16:
                continue

            # 按类别分组
            cat_vecs = {}
            for v, w in zip(du_vecs, wnames):
                c = word2cat[w]
                if c not in cat_vecs:
                    cat_vecs[c] = []
                cat_vecs[c].append(v)

            # 每个类别的子空间维度(n90需要>=5个样本,8个词够了)
            cat_n90s = {}
            for c, vecs in cat_vecs.items():
                if len(vecs) < 5:
                    cat_n90s[c] = -1
                    continue
                arr = np.array(vecs, dtype=np.float32)
                arr_c = arr - arr.mean(axis=0)
                k = min(len(vecs)-1, arr_c.shape[1]-1, 50)
                if k < 2:
                    cat_n90s[c] = -1
                    continue
                try:
                    _, s, _ = svds(arr_c.astype(np.float32), k=k)
                    s_sorted = np.sort(s)[::-1]
                    var_exp = s_sorted ** 2
                    total = var_exp.sum()
                    if total < 1e-10:
                        cat_n90s[c] = -1
                        continue
                    cum = np.cumsum(var_exp) / total
                    n90 = int(np.searchsorted(cum, 0.90)) + 1
                    n50 = int(np.searchsorted(cum, 0.50)) + 1
                    cat_n90s[c] = {"n50": n50, "n90": n90}
                except:
                    cat_n90s[c] = -1

            # 类别子空间重叠度: 每个类别的top-N PCA维度有多少是全局top-N的？
            # 先计算全局PCA
            X_all = np.array(du_vecs, dtype=np.float32)
            X_all_c = X_all - X_all.mean(axis=0)
            k_global = min(80, X_all_c.shape[0]-1, X_all_c.shape[1]-1)
            if k_global < 5:
                continue
            try:
                _, s_global, Vt_global = svds(X_all_c.astype(np.float32), k=k_global)
            except:
                continue

            # 全局top-50维度的索引
            global_top50_dims = set(range(min(50, k_global)))

            # 对每个类别，计算其top-20维度中有多少在全局top-50中
            overlap_stats = {}
            for c, vecs in cat_vecs.items():
                if len(vecs) < 5:
                    overlap_stats[c] = -1
                    continue
                arr = np.array(vecs, dtype=np.float32)
                arr_c = arr - arr.mean(axis=0)
                k_cat = min(len(vecs)-1, arr_c.shape[1]-1, 40)
                if k_cat < 3:
                    overlap_stats[c] = -1
                    continue
                try:
                    _, s_cat, Vt_cat = svds(arr_c.astype(np.float32), k=k_cat)
                except:
                    overlap_stats[c] = -1
                    continue

                # 类别top-20维度的方向
                # 通过计算类别向量在全局Vt上的投影来找重叠
                # 更简单：计算类别主成分和全局主成分的相关性
                # 用前5个主成分的平均余弦相似度
                cat_top5 = Vt_cat[np.argsort(s_cat)[-5:]]  # top-5方向
                global_top5 = Vt_global[np.argsort(s_global)[-5:]]  # top-5方向

                # 平均绝对余弦相似度
                cos_sims = []
                for cv in cat_top5:
                    for gv in global_top5:
                        cs = abs(np.dot(cv, gv) / (np.linalg.norm(cv) * np.linalg.norm(gv) + 1e-10))
                        cos_sims.append(cs)
                avg_cos = float(np.mean(cos_sims))
                overlap_stats[c] = round(avg_cos, 4)

            # 有效类别数 (n90从低到高排列)
            valid_cats = {c: v for c, v in cat_n90s.items() if isinstance(v, dict)}
            if valid_cats:
                n90_values = [v["n90"] for v in valid_cats.values()]
                n50_values = [v["n50"] for v in valid_cats.values()]
                overlap_values = [v for v in overlap_stats.values() if isinstance(v, float)]
            else:
                n90_values, n50_values, overlap_values = [], [], []

            exp4_results.append({
                "layer": li,
                "n_categories_with_data": len(valid_cats),
                "cat_n90_mean": float(np.mean(n90_values)) if n90_values else -1,
                "cat_n90_std": float(np.std(n90_values)) if n90_values else -1,
                "cat_n90_min": int(np.min(n90_values)) if n90_values else -1,
                "cat_n90_max": int(np.max(n90_values)) if n90_values else -1,
                "cat_n50_mean": float(np.mean(n50_values)) if n50_values else -1,
                "per_cat_n90": {c: v for c, v in cat_n90s.items() if isinstance(v, dict)},
                "subspace_overlap_mean": float(np.mean(overlap_values)) if overlap_values else -1,
                "subspace_overlap_std": float(np.std(overlap_values)) if overlap_values else -1,
                "per_cat_overlap": overlap_stats,
            })
            log(f"  L{li}: cat_n90_mean={np.mean(n90_values):.1f}, overlap_mean={np.mean(overlap_values):.3f}" if n90_values else f"  L{li}: no data")

        with open(out_dir / "exp4_cat_subspace.json", 'w') as f:
            json.dump({"model": model_name, "results": exp4_results}, f, indent=2)
        log(f"Exp4 done: {len(exp4_results)} layers")

        # ===== 释放 =====
        del word_gates, word_ups
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
