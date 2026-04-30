"""
Phase CCLXXIX: 数据积累 — 4组纯经验实验
=====================================
目标: 不做理论推演，纯积累数据填补缺口

Exp1: Δg/Δu差异向量Probe — 类别信息到底在哪里？
  - 对Δg⊙ū做8分类probe（最近质心+余弦分类器）
  - 对Δu做8分类probe
  - 对W_down@Δu做8分类probe
  - 对比三类差异向量的分类准确率

Exp2: 非相邻层Procrustes — 多步旋转是累积还是独立？
  - L0→Lk (k=2,5,10,15,20,...)
  - 对比: 单步fit_cos vs 累积fit_cos
  - 如果独立旋转，多步fit_cos应快速下降

Exp3: R_l与权重矩阵的经验关联
  - W_down的奇异值谱 vs R_l的结构
  - W_gate的行范数分布 vs R_l对角线
  - W_up的条件数 vs R_l的fit_cos
  - 纯相关性数据，不做因果推断

Exp4: 扩大词表到128词 — n90是否继续增长？
  - 16类别 × 8词 = 128词
  - 在n90=34-46的基础上是否继续增长
  - 新增: color, plant, metal, sport, music, science, body, time
"""

import sys, os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxix_log.txt"

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        import time
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()

log("=== CCLXXIX Script started ===")

import json, time, gc, traceback
from pathlib import Path
from datetime import datetime
import numpy as np

log("Importing torch...")
import torch

log("Importing model_utils...")
sys.path.insert(0, r"d:\Ai2050\TransformerLens-Project")
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, get_W_U,
    get_layer_weights, MODEL_CONFIGS,
)

log("All imports done")

# ===== 128词表: 16类别 × 8词 =====
CONCEPTS_128 = {
    # 原始8类
    "animal":   ["dog", "cat", "horse", "bird", "fish", "lion", "bear", "deer"],
    "food":     ["apple", "bread", "cheese", "rice", "meat", "cake", "soup", "salt"],
    "tool":     ["hammer", "knife", "scissors", "saw", "drill", "wrench", "chisel", "ruler"],
    "vehicle":  ["car", "bus", "train", "plane", "boat", "truck", "bike", "ship"],
    "clothing": ["shirt", "dress", "hat", "coat", "shoe", "belt", "scarf", "glove"],
    "weather":  ["rain", "snow", "wind", "storm", "fog", "hail", "frost", "cloud"],
    "emotion":  ["joy", "fear", "anger", "hope", "love", "grief", "pride", "shame"],
    "building": ["house", "church", "tower", "bridge", "castle", "temple", "museum", "palace"],
    # 新增8类
    "color":    ["red", "blue", "green", "gold", "silver", "pink", "brown", "gray"],
    "plant":    ["tree", "flower", "grass", "bush", "vine", "weed", "moss", "fern"],
    "metal":    ["iron", "copper", "steel", "gold_m", "brass", "tin", "zinc", "lead"],
    "sport":    ["soccer", "tennis", "boxing", "golf", "rugby", "skiing", "rowing", "fencing"],
    "music":    ["piano", "violin", "drum", "flute", "guitar", "harp", "trumpet", "organ"],
    "science":  ["atom", "cell", "gene", "orbit", "force", "mass", "wave", "ray"],
    "body":     ["hand", "foot", "head", "heart", "brain", "lung", "bone", "skin"],
    "time":     ["dawn", "noon", "dusk", "night", "spring", "summer", "autumn", "winter"],
}

# 修正metal中的gold_m → gold (避免与color重叠, 使用不同token)
CONCEPTS_128["metal"] = ["iron", "copper", "steel", "bronze", "brass", "tin", "zinc", "lead"]


def proper_cos(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def compute_n90(vectors, max_k=80):
    """Compute n90 from a list of difference vectors."""
    if len(vectors) < 10:
        return -1, -1, -1
    a = np.array(vectors, dtype=np.float32)
    c = a - a.mean(axis=0)
    from scipy.sparse.linalg import svds
    np_ = min(max_k, c.shape[0]-1, c.shape[1]-1)
    if np_ < 2:
        return -1, -1, -1
    try:
        _, s, _ = svds(c.astype(np.float32), k=np_)
    except Exception:
        return -1, -1, -1
    s_sorted = np.sort(s)[::-1]
    var_explained = s_sorted ** 2
    total_var = var_explained.sum()
    if total_var < 1e-10:
        return -1, -1, -1
    cum_var = np.cumsum(var_explained) / total_var
    n90 = int(np.searchsorted(cum_var, 0.90)) + 1
    top5_var = float(var_explained[:5].sum() / total_var)
    return n90, top5_var, int(c.shape[0])


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

        # ===== 收集128词数据 =====
        template = "The {} is"
        rng = np.random.RandomState(42)
        all_words, all_cats = [], []
        for cat, words in CONCEPTS_128.items():
            sel = rng.choice(words, min(8, len(words)), replace=False)
            all_words.extend(sel.tolist() if hasattr(sel, 'tolist') else list(sel))
            all_cats.extend([cat] * len(sel))
        cat_names = sorted(set(all_cats))
        n_cats = len(cat_names)
        log(f"Total words: {len(all_words)}, categories: {n_cats}")

        word_gates, word_ups, word_residuals = {}, {}, {}
        t0 = time.time()
        for wi, word in enumerate(all_words):
            text = template.format(word)
            input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
            last_pos = input_ids.shape[1] - 1

            ln_out, res_out = {}, {}
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
                def make_layer_out(key):
                    def hook(module, input, output):
                        o = output[0] if isinstance(output, tuple) else output
                        res_out[key] = o[0, last_pos].detach().float().cpu().numpy()
                    return hook
                hooks.append(layer.register_forward_hook(make_layer_out(f"L{li}")))

            with torch.no_grad():
                _ = model(input_ids)
            for h in hooks:
                h.remove()

            g_dict, u_dict, r_dict = {}, {}, {}
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
                r_dict[li] = res_out.get(key, None)

            word_gates[word] = g_dict
            word_ups[word] = u_dict
            word_residuals[word] = r_dict
            if (wi + 1) % 32 == 0 or wi == 0:
                log(f"  Word {wi+1}/{len(all_words)} ({time.time()-t0:.0f}s)")

        log(f"Data collection done ({time.time()-t0:.0f}s)")

        # 预计算 W_down @ ū, W_down @ Δu, Δg⊙ū
        log("Precomputing W_down@ū, W_down@Δu, Δg⊙ū...")
        word_wdu = {}  # W_down @ ū
        word_wd_du = {}  # W_down @ Δu (Δu = u_word - u_mean)
        word_dg_u = {}   # Δg ⊙ ū (Δg = g_word - g_mean)

        # 计算每层的均值ū和均值g
        layer_mean_u = {}
        layer_mean_g = {}
        for li in range(n_layers):
            us = [word_ups[w][li] for w in all_words if li in word_ups.get(w, {})]
            gs = [word_gates[w][li] for w in all_words if li in word_gates.get(w, {})]
            if us:
                layer_mean_u[li] = np.mean(us, axis=0)
            if gs:
                layer_mean_g[li] = np.mean(gs, axis=0)

        for word in all_words:
            word_wdu[word] = {}
            word_wd_du[word] = {}
            word_dg_u[word] = {}
            for li in range(n_layers):
                if li not in word_ups.get(word, {}): continue
                lw = get_layer_weights(layers_list[li], d_model, mlp_type)
                if lw.W_gate is None: continue

                u = word_ups[word][li]
                g = word_gates[word][li]
                wdu = lw.W_down @ u
                word_wdu[word][li] = wdu

                if li in layer_mean_u:
                    du = u - layer_mean_u[li]
                    word_wd_du[word][li] = lw.W_down @ du

                if li in layer_mean_g:
                    dg = g - layer_mean_g[li]
                    # Δg ⊙ ū: 用均值ū作为载体
                    dg_u = dg * layer_mean_u[li]  # 逐元素乘
                    word_dg_u[word][li] = lw.W_down @ dg_u

        log("Precomputation done")

        # ================================================================
        # Exp1: Δg/Δu差异向量Probe — 类别信息在哪里？
        # ================================================================
        log("=== Exp1: Difference Vector Probe ===")

        from scipy.sparse.linalg import svds

        exp1_results = []
        # 选择几个代表性层
        probe_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
        probe_layers = [l for l in probe_layers if l < n_layers]

        for li in probe_layers:
            # 收集三类差异向量
            feat_sets = {}
            # 1) W_down @ Δu
            wd_du_vecs = [word_wd_du[w][li] for w in all_words if li in word_wd_du.get(w, {})]
            if len(wd_du_vecs) >= 16:
                feat_sets["Wd_du"] = np.array(wd_du_vecs, dtype=np.float32)

            # 2) W_down @ (Δg ⊙ ū)
            dg_u_vecs = [word_dg_u[w][li] for w in all_words if li in word_dg_u.get(w, {})]
            if len(dg_u_vecs) >= 16:
                feat_sets["Wd_dg_u"] = np.array(dg_u_vecs, dtype=np.float32)

            # 3) Δg 本身 (高维)
            dg_vecs = [word_gates[w][li] - layer_mean_g[li] for w in all_words if li in word_gates.get(w, {})]
            if len(dg_vecs) >= 16:
                feat_sets["dg_raw"] = np.array(dg_vecs, dtype=np.float32)

            # 4) Δu 本身 (高维)
            du_vecs = [word_ups[w][li] - layer_mean_u[li] for w in all_words if li in word_ups.get(w, {})]
            if len(du_vecs) >= 16:
                feat_sets["du_raw"] = np.array(du_vecs, dtype=np.float32)

            # 5) W_down @ ū (绝对方向)
            wdu_vecs = [word_wdu[w][li] for w in all_words if li in word_wdu.get(w, {})]
            if len(wdu_vecs) >= 16:
                feat_sets["Wd_u_abs"] = np.array(wdu_vecs, dtype=np.float32)

            layer_result = {"layer": li, "n_words": len(all_words), "n_cats": n_cats}

            for feat_name, X in feat_sets.items():
                if X.shape[0] < 16:
                    continue

                # PCA降维 (用于分类)
                n_comp = min(50, X.shape[0]-1, X.shape[1]-1)
                X_c = X - X.mean(axis=0)
                try:
                    _, s, Vt = svds(X_c, k=n_comp)
                    Vt = Vt[np.argsort(-s)]
                    s = s[np.argsort(-s)]
                    X_pca = X_c @ Vt.T
                except Exception:
                    continue

                # ===== 分类器1: 最近质心 (余弦距离) =====
                # Leave-one-category-out: 每次留一个类别做测试
                cat_labels = np.array([cat_names.index(c) for c in all_cats])
                accs_cos = []
                for test_cat_idx in range(n_cats):
                    test_mask = cat_labels == test_cat_idx
                    train_mask = ~test_mask
                    if test_mask.sum() < 2 or train_mask.sum() < 16:
                        continue

                    # 计算每个训练类别的质心
                    centroids = []
                    for ci in range(n_cats):
                        ci_mask = cat_labels == ci
                        ci_train = ci_mask & train_mask
                        if ci_train.sum() > 0:
                            centroids.append(X_pca[ci_train].mean(axis=0))
                        else:
                            centroids.append(np.zeros(n_comp))
                    centroids = np.array(centroids)

                    # 对测试样本分类
                    for ti in np.where(test_mask)[0]:
                        v = X_pca[ti]
                        # 余弦相似度
                        sims = [proper_cos(v, c) for c in centroids]
                        pred = np.argmax(sims)
                        accs_cos.append(int(pred == cat_labels[ti]))

                # ===== 分类器2: 最近质心 (欧氏距离) =====
                accs_euc = []
                for test_cat_idx in range(n_cats):
                    test_mask = cat_labels == test_cat_idx
                    train_mask = ~test_mask
                    if test_mask.sum() < 2 or train_mask.sum() < 16:
                        continue
                    centroids = []
                    for ci in range(n_cats):
                        ci_mask = cat_labels == ci
                        ci_train = ci_mask & train_mask
                        if ci_train.sum() > 0:
                            centroids.append(X_pca[ci_train].mean(axis=0))
                        else:
                            centroids.append(np.zeros(n_comp))
                    centroids = np.array(centroids)
                    for ti in np.where(test_mask)[0]:
                        v = X_pca[ti]
                        dists = [np.linalg.norm(v - c) for c in centroids]
                        pred = np.argmin(dists)
                        accs_euc.append(int(pred == cat_labels[ti]))

                # ===== n90 for this feature set =====
                n90, top5v, nsamp = compute_n90(list(X), max_k=80)

                cos_acc = float(np.mean(accs_cos)) if accs_cos else 0.0
                euc_acc = float(np.mean(accs_euc)) if accs_euc else 0.0

                layer_result[feat_name] = {
                    "cos_acc": round(cos_acc, 4),
                    "euc_acc": round(euc_acc, 4),
                    "n90": n90,
                    "top5_var": round(top5v, 4) if top5v >= 0 else -1,
                    "n_samp": nsamp,
                }
                log(f"  L{li} {feat_name}: cos_acc={cos_acc:.3f} euc_acc={euc_acc:.3f} n90={n90}")

            exp1_results.append(layer_result)

        out_dir = f"d:\\Ai2050\\TransformerLens-Project\\results\\causal_fiber\\{model_name}_cclxxix"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "exp1_diff_probe.json"), 'w') as f:
            json.dump({"model": model_name, "n_words": len(all_words), "n_cats": n_cats,
                        "layer_results": exp1_results}, f, indent=2)
        log(f"Exp1 saved ({len(exp1_results)} layers)")

        # ================================================================
        # Exp2: 非相邻层Procrustes — 多步旋转
        # ================================================================
        log("=== Exp2: Non-adjacent Procrustes ===")

        n_dim = min(50, len(all_words)-1, d_model-1)

        # 构建PCA子空间
        layer_pca = {}
        for li in range(n_layers):
            vw = [w for w in all_words if li in word_wdu[w]]
            if len(vw) < 8: continue
            X = np.array([word_wdu[w][li] for w in vw], dtype=np.float32)
            X_c = X - X.mean(axis=0)
            try:
                _, s, Vt = svds(X_c, k=n_dim)
                Vt = Vt[np.argsort(-s)]
                s = s[np.argsort(-s)]
            except Exception:
                continue
            X_pca = X_c @ Vt.T
            layer_pca[li] = {"Vt": Vt, "s": s, "X_pca": X_pca, "words": vw}

        exp2_results = []

        # 测试不同的步长
        steps = [1, 2, 3, 5, 10, 15, 20]
        start_layers = [0, n_layers//4, n_layers//2]

        for start_l in start_layers:
            if start_l not in layer_pca: continue

            for step in steps:
                end_l = start_l + step
                if end_l >= n_layers or end_l not in layer_pca: continue

                # 对齐词表
                vw_common = [w for w in all_words
                             if w in layer_pca[start_l]["words"] and w in layer_pca[end_l]["words"]]
                if len(vw_common) < 8: continue

                # 计算配对的PCA坐标
                X_start = np.array([word_wdu[w][start_l] for w in vw_common], dtype=np.float32)
                X_end = np.array([word_wdu[w][end_l] for w in vw_common], dtype=np.float32)
                X_start_c = X_start - X_start.mean(axis=0)
                X_end_c = X_end - X_end.mean(axis=0)

                Vt_s = layer_pca[start_l]["Vt"]
                Vt_e = layer_pca[end_l]["Vt"]
                X_proj = X_start_c @ Vt_s.T
                Y_proj = X_end_c @ Vt_e.T

                # Procrustes
                M = Y_proj.T @ X_proj
                U, S, Vt_m = np.linalg.svd(M)
                R = U @ Vt_m
                fit_cos = float(np.mean([proper_cos(X_proj[i] @ R.T, Y_proj[i])
                                          for i in range(len(vw_common))]))

                # 随机基线: 随机排列Y_proj的行
                rng_bl = np.random.RandomState(42)
                n_bl = 20
                rand_fits = []
                for _ in range(n_bl):
                    perm = rng_bl.permutation(len(vw_common))
                    Y_rand = Y_proj[perm]
                    M_r = Y_rand.T @ X_proj
                    U_r, S_r, Vt_r = np.linalg.svd(M_r)
                    R_r = U_r @ Vt_r
                    rand_fits.append(float(np.mean([proper_cos(X_proj[i] @ R_r.T, Y_rand[i])
                                                     for i in range(len(vw_common))])))
                rand_fit = float(np.mean(rand_fits))

                # 累积旋转预测: R_cumul = R_{end-1} @ ... @ R_{start}
                # 如果每步都是独立旋转，累积旋转的fit_cos应该约等于单步^step
                # 但我们直接测量累积旋转是否比单步Procrustes差

                r = {
                    "start_layer": start_l,
                    "end_layer": end_l,
                    "step": step,
                    "n_words": len(vw_common),
                    "fit_cos": round(fit_cos, 4),
                    "rand_fit_cos": round(rand_fit, 4),
                    "excess_over_rand": round(fit_cos - rand_fit, 4),
                    "det_R": round(float(np.linalg.det(R)), 4),
                }
                exp2_results.append(r)
                if step in [1, 5, 10, 20]:
                    log(f"  L{start_l}->L{end_l} (step={step}): fit_cos={fit_cos:.4f} "
                        f"rand={rand_fit:.4f} excess={fit_cos-rand_fit:.4f}")

        with open(os.path.join(out_dir, "exp2_multi_step_procrustes.json"), 'w') as f:
            json.dump({"model": model_name, "results": exp2_results}, f, indent=2)
        log(f"Exp2 saved ({len(exp2_results)} pairs)")

        # ================================================================
        # Exp3: R_l与权重矩阵的经验关联
        # ================================================================
        log("=== Exp3: R_l vs Weight Matrix Correlation ===")

        # 先计算所有相邻层的R_l
        layer_R = {}
        sorted_layers = sorted(layer_pca.keys())
        for idx in range(len(sorted_layers) - 1):
            li = sorted_layers[idx]
            lj = sorted_layers[idx + 1]
            if lj != li + 1: continue

            vw_common = [w for w in all_words
                         if w in layer_pca[li]["words"] and w in layer_pca[lj]["words"]]
            if len(vw_common) < 8: continue

            X_li = np.array([word_wdu[w][li] for w in vw_common], dtype=np.float32)
            Y_lj = np.array([word_wdu[w][lj] for w in vw_common], dtype=np.float32)
            X_li_c = X_li - X_li.mean(axis=0)
            Y_lj_c = Y_lj - Y_lj.mean(axis=0)

            Vt_l = layer_pca[li]["Vt"]
            Vt_r = layer_pca[lj]["Vt"]
            X_proj = X_li_c @ Vt_l.T
            Y_proj = Y_lj_c @ Vt_r.T

            M = Y_proj.T @ X_proj
            U, S, Vt_m = np.linalg.svd(M)
            R = U @ Vt_m
            fit_cos = float(np.mean([proper_cos(X_proj[i] @ R.T, Y_proj[i])
                                      for i in range(len(vw_common))]))

            R_dev = float(np.linalg.norm(R - np.eye(n_dim), 'fro') / np.sqrt(n_dim))
            n_sig = int(np.sum(np.abs(np.diag(R)) < 0.99))
            diag_mean = float(np.mean(np.abs(np.diag(R))))
            det_R = float(np.linalg.det(R))

            layer_R[li] = {
                "R": R, "fit_cos": fit_cos, "R_dev": R_dev,
                "n_sig_rot": n_sig, "diag_mean": diag_mean, "det_R": det_R,
            }

        # 收集权重矩阵的统计量 — 只在6个采样层做完整SVD避免OOM
        exp3_results = []
        exp3_sample_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]
        exp3_sample_layers = sorted(set([l for l in exp3_sample_layers if l < n_layers]))
        
        for li in sorted(layer_R.keys()):
            R_info = layer_R[li]

            # 激活统计（轻量，所有层都做）
            g_all = np.array([word_gates[w][li] for w in all_words if li in word_gates.get(w, {})])
            u_all = np.array([word_ups[w][li] for w in all_words if li in word_ups.get(w, {})])

            g_mean = float(g_all.mean()) if len(g_all) > 0 else 0
            g_std = float(g_all.std()) if len(g_all) > 0 else 0
            g_sparsity = float((g_all < 0.1).mean()) if len(g_all) > 0 else 0
            u_norm_mean = float(np.mean(np.linalg.norm(u_all, axis=1))) if len(u_all) > 0 else 0

            r = {
                "layer": li,
                "R_dev": round(R_info["R_dev"], 4),
                "R_fit_cos": round(R_info["fit_cos"], 4),
                "R_n_sig_rot": R_info["n_sig_rot"],
                "R_diag_mean": round(R_info["diag_mean"], 4),
                "R_det": round(R_info["det_R"], 4),
                "gate_mean": round(g_mean, 4),
                "gate_std": round(g_std, 4),
                "gate_sparsity": round(g_sparsity, 4),
                "up_norm_mean": round(u_norm_mean, 1),
            }

            # 只在采样层做权重矩阵SVD（避免OOM）
            if li in exp3_sample_layers:
                lw = get_layer_weights(layers_list[li], d_model, mlp_type)
                if lw.W_gate is not None:
                    # W_down统计 — 用scipy.sparse.linalg.svds只取top20
                    W_d = lw.W_down  # (d_model, intermediate)
                    wd_fro = float(np.linalg.norm(W_d, 'fro'))
                    try:
                        from scipy.sparse.linalg import svds as sparse_svds
                        k_svd = min(20, min(W_d.shape)-1)
                        wd_svals = sparse_svds(W_d.astype(np.float32), k=k_svd, return_singular_vectors=False)
                        wd_svals = np.sort(wd_svals)[::-1]
                        wd_cond = float(wd_svals[0] / max(wd_svals[-1], 1e-10))
                    except Exception:
                        wd_svals = np.array([0]*5)
                        wd_cond = -1

                    # W_gate统计
                    W_g = lw.W_gate
                    wg_row_norms = np.linalg.norm(W_g, axis=1)
                    wg_row_norm_mean = float(wg_row_norms.mean())
                    wg_row_norm_std = float(wg_row_norms.std())

                    # W_up统计
                    W_u = lw.W_up
                    wu_fro = float(np.linalg.norm(W_u, 'fro'))
                    try:
                        k_svd = min(20, min(W_u.shape)-1)
                        wu_svals = sparse_svds(W_u.astype(np.float32), k=k_svd, return_singular_vectors=False)
                        wu_svals = np.sort(wu_svals)[::-1]
                        wu_cond = float(wu_svals[0] / max(wu_svals[-1], 1e-10))
                    except Exception:
                        wu_svals = np.array([0]*5)
                        wu_cond = -1

                    # layernorm
                    ln_w = lw.input_layernorm_weight
                    ln_std = float(np.std(ln_w)) if ln_w is not None else 0.0

                    r.update({
                        "Wd_cond": round(wd_cond, 2),
                        "Wd_fro": round(wd_fro, 1),
                        "Wd_sval_top5": [round(float(x), 4) for x in wd_svals[:5]],
                        "Wg_row_norm_mean": round(wg_row_norm_mean, 4),
                        "Wg_row_norm_std": round(wg_row_norm_std, 4),
                        "Wu_cond": round(wu_cond, 2),
                        "Wu_fro": round(wu_fro, 1),
                        "Wu_sval_top5": [round(float(x), 4) for x in wu_svals[:5]],
                        "LN_std": round(ln_std, 4),
                    })

                    # 释放权重矩阵
                    del W_d, W_g, W_u, lw
                    gc.collect()

            exp3_results.append(r)
            if li % 6 == 0 or li == n_layers - 1:
                log(f"  L{li}: R_dev={R_info['R_dev']:.3f} fit={R_info['fit_cos']:.3f} "
                    f"gate_sparsity={g_sparsity:.3f}")

        with open(os.path.join(out_dir, "exp3_R_vs_weights.json"), 'w') as f:
            json.dump({"model": model_name, "results": exp3_results}, f, indent=2)
        log(f"Exp3 saved ({len(exp3_results)} layers)")

        # 释放大数组
        del layer_R, layer_pca, word_wdu, word_wd_du, word_dg_u
        gc.collect()

        # ================================================================
        # Exp4: 128词n90收敛性 — 关键数据缺口
        # ================================================================
        log("=== Exp4: 128-word n90 Convergence ===")

        exp4_results = []
        sample_sizes = [16, 32, 48, 64, 96, 128]  # 词数

        for n_sample in sample_sizes:
            # 随机选n_sample个词
            rng_s = np.random.RandomState(123)
            idx = rng_s.choice(len(all_words), min(n_sample, len(all_words)), replace=False)
            sel_words = [all_words[i] for i in idx]
            sel_cats = [all_cats[i] for i in idx]

            for li in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
                if li >= n_layers: continue
                # Δg差异向量
                dg_diffs = []
                for i in range(len(sel_words)):
                    for j in range(i+1, len(sel_words)):
                        w1, w2 = sel_words[i], sel_words[j]
                        if li in word_gates.get(w1, {}) and li in word_gates.get(w2, {}):
                            dg_diffs.append(word_gates[w1][li] - word_gates[w2][li])

                # Δu差异向量
                du_diffs = []
                for i in range(len(sel_words)):
                    for j in range(i+1, len(sel_words)):
                        w1, w2 = sel_words[i], sel_words[j]
                        if li in word_ups.get(w1, {}) and li in word_ups.get(w2, {}):
                            du_diffs.append(word_ups[w1][li] - word_ups[w2][li])

                n90_dg, top5_dg, ns_dg = compute_n90(dg_diffs, max_k=80)
                n90_du, top5_du, ns_du = compute_n90(du_diffs, max_k=80)

                r = {
                    "n_words": n_sample,
                    "n_pairs_dg": ns_dg,
                    "n_pairs_du": ns_du,
                    "layer": li,
                    "n90_dg": n90_dg,
                    "n90_du": n90_du,
                    "top5_var_dg": round(top5_dg, 4) if top5_dg >= 0 else -1,
                    "top5_var_du": round(top5_du, 4) if top5_du >= 0 else -1,
                }
                exp4_results.append(r)
                if n_sample in [16, 64, 128] and li in [0, n_layers//2, n_layers-1]:
                    log(f"  n={n_sample} L{li}: n90_dg={n90_dg} n90_du={n90_du}")

        with open(os.path.join(out_dir, "exp4_n90_convergence_128.json"), 'w') as f:
            json.dump({"model": model_name, "results": exp4_results}, f, indent=2)
        log(f"Exp4 saved ({len(exp4_results)} entries)")

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
        except Exception:
            pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3")
    args = parser.parse_args()
    model_arg = args.model
    if model_arg == "all":
        for m in ["qwen3", "glm4", "deepseek7b"]:
            run_model(m)
    else:
        run_model(model_arg)
