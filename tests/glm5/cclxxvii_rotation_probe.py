"""
Phase CCLXXVII: 旋转群检测 + Δg干预实验 + Probe信息追踪
核心目标:
1. 检验FFN层间变换是否接近正交旋转(Procrustes分析)
2. Δg/Δu swap干预验证选择器假说
3. 线性Probe追踪每层概念信息量
"""
import sys, os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxvii_log.txt"

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        import time
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()

log("=== CCLXXVII Script started ===")

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

CONCEPTS_EXPANDED = {
    "animal":   ["dog", "cat", "horse", "bird", "fish", "lion", "bear", "deer"],
    "food":     ["apple", "bread", "cheese", "rice", "meat", "cake", "soup", "salt"],
    "tool":     ["hammer", "knife", "scissors", "saw", "drill", "wrench", "chisel", "ruler"],
    "vehicle":  ["car", "bus", "train", "plane", "boat", "truck", "bike", "ship"],
    "clothing": ["shirt", "dress", "hat", "coat", "shoe", "belt", "scarf", "glove"],
    "weather":  ["rain", "snow", "wind", "storm", "fog", "hail", "frost", "cloud"],
    "emotion":  ["joy", "fear", "anger", "hope", "love", "grief", "pride", "shame"],
    "building": ["house", "church", "tower", "bridge", "castle", "temple", "museum", "palace"],
}


def proper_cos(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


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

        # ===== 收集64词数据 =====
        template = "The {} is"
        rng = np.random.RandomState(42)
        all_words, all_cats = [], []
        for cat, words in CONCEPTS_EXPANDED.items():
            sel = rng.choice(words, min(8, len(words)), replace=False)
            all_words.extend(sel.tolist() if hasattr(sel, 'tolist') else list(sel))
            all_cats.extend([cat] * len(sel))
        cat_names = list(CONCEPTS_EXPANDED.keys())
        log(f"Total words: {len(all_words)}, categories: {len(cat_names)}")

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
            elapsed = time.time() - t0
            if (wi + 1) % 16 == 0 or wi == 0:
                log(f"  Word {wi+1}/{len(all_words)} ({elapsed:.0f}s)")

        log(f"Data collection done ({time.time()-t0:.0f}s)")

        # 预计算 W_down @ ū 和 W_down @ (g ⊙ u) for each word/layer
        log("Precomputing W_down@ū and W_down@(g⊙u)...")
        word_wdu = {}  # W_down @ ū
        word_wdg_u = {}  # W_down @ (Δg ⊙ ū) = W_down @ ((g-ḡ) ⊙ ū)
        word_wg_du = {}  # W_down @ (ḡ ⊙ Δu)

        # 计算每层每类的均值
        layer_cat_mean_g = {}
        layer_cat_mean_u = {}
        layer_mean_g = {}
        layer_mean_u = {}
        for li in range(n_layers):
            vw = [w for w in all_words if li in word_gates[w] and li in word_ups[w]]
            if not vw: continue
            layer_mean_g[li] = np.mean([word_gates[w][li] for w in vw], axis=0)
            layer_mean_u[li] = np.mean([word_ups[w][li] for w in vw], axis=0)

        for word in all_words:
            word_wdu[word] = {}
            word_wdg_u[word] = {}
            word_wg_du[word] = {}
            for li in range(n_layers):
                if li not in word_gates.get(word, {}) or li not in word_ups.get(word, {}):
                    continue
                lw = get_layer_weights(layers_list[li], d_model, mlp_type)
                if lw.W_gate is None: continue
                W_down = lw.W_down
                g = word_gates[word][li]
                u = word_ups[word][li]
                g_bar = layer_mean_g[li]
                u_bar = layer_mean_u[li]
                word_wdu[word][li] = W_down @ u
                word_wdg_u[word][li] = W_down @ ((g - g_bar) * u)  # gate项差异
                word_wg_du[word][li] = W_down @ (g_bar * (u - u_bar))  # up项差异

        log("Precomputation done")

        # ================================================================
        # Exp1: Procrustes旋转检测 (★★★★★)
        # 检验层间变换是否接近正交旋转
        # ================================================================
        log("=== Exp1: Procrustes Rotation Detection ===")
        exp1 = []

        for li in range(n_layers - 1):
            vw = [w for w in all_words if li in word_wdu[w] and (li+1) in word_wdu[w]]
            if len(vw) < 8: continue

            # 构建数据矩阵: X = [v_l(w1), ...], Y = [v_{l+1}(w1), ...]
            X = np.array([word_wdu[w][li] for w in vw], dtype=np.float32)  # (n, d)
            Y = np.array([word_wdu[w][li+1] for w in vw], dtype=np.float32)  # (n, d)

            # 中心化
            X_c = X - X.mean(axis=0)
            Y_c = Y - Y.mean(axis=0)

            # 方法1: 直接Procrustes (在有效子空间中)
            # 用PCA降维到有效维度
            from scipy.sparse.linalg import svds
            n_dim = min(50, len(vw)-1, d_model-1)
            if n_dim < 2: continue

            try:
                _, sx, Vx = svds(X_c, k=n_dim)
                _, sy, Vy = svds(Y_c, k=n_dim)
            except Exception:
                continue

            idx_x = np.argsort(-sx); Vx = Vx[idx_x]
            idx_y = np.argsort(-sy); Vy = Vy[idx_y]

            # 在PCA子空间中的坐标
            X_pca = X_c @ Vx.T  # (n, n_dim)
            Y_pca = Y_c @ Vy.T  # (n, n_dim)

            # Procrustes: 找最优正交变换R使得 ||Y_pca - X_pca @ R^T||_F 最小
            # 解: U,S,Vt = SVD(Y_pca^T @ X_pca), R = U @ Vt
            M = Y_pca.T @ X_pca  # (n_dim, n_dim)
            U, S, Vt = np.linalg.svd(M)

            # 最优正交变换
            R = U @ Vt  # (n_dim, n_dim)

            # 检验1: R是否正交? ||R^T R - I||_F / sqrt(n_dim)
            orth_err = float(np.linalg.norm(R.T @ R - np.eye(n_dim)) / np.sqrt(n_dim))

            # 检验2: det(R) 接近+1还是-1?
            det_R = float(np.linalg.det(R))

            # 检验3: 拟合质量 — R解释了多少方差?
            X_rot = X_pca @ R.T  # 旋转后的X
            # 逐词余弦
            fit_cos = []
            for i in range(len(vw)):
                fit_cos.append(proper_cos(X_rot[i], Y_pca[i]))
            mean_fit_cos = float(np.mean(fit_cos))

            # 检验4: S的分布 (Procrustes中的奇异值)
            # 如果R=U@Vt完美正交, S应全为1
            s_ratio = S / S[0] if S[0] > 1e-10 else np.zeros_like(S)
            s_top5 = float(np.mean(s_ratio[:5]))

            # 检验5: 与随机基线比较
            # 打乱词序后做同样的Procrustes
            rng_perm = np.random.RandomState(li * 100 + 42)
            perm = rng_perm.permutation(len(vw))
            Y_perm = Y_pca[perm]
            M_perm = Y_perm.T @ X_pca
            U_p, S_p, Vt_p = np.linalg.svd(M_perm)
            R_p = U_p @ Vt_p
            X_rot_p = X_pca @ R_p.T
            fit_cos_perm = [proper_cos(X_rot_p[i], Y_pca[perm[i]]) for i in range(len(vw))]

            lr = {
                "layer_pair": f"L{li}->L{li+1}",
                "l1": li, "l2": li+1,
                "n_words": len(vw),
                "pca_dim": n_dim,
                "orth_error": orth_err,
                "det_R": det_R,
                "mean_fit_cos": mean_fit_cos,
                "mean_fit_cos_random": float(np.mean(fit_cos_perm)),
                "s_top5_ratio": s_top5,
                "fit_vs_random_ratio": float(mean_fit_cos / max(np.mean(fit_cos_perm), 1e-6)),
                "singular_value_ratios": s_ratio[:10].tolist(),
            }
            exp1.append(lr)
            if li % 4 == 0 or li == n_layers - 2:
                log(f"  L{li}->L{li+1}: orth_err={orth_err:.4f} det={det_R:+.3f} "
                    f"fit_cos={mean_fit_cos:.3f} vs random={np.mean(fit_cos_perm):.3f} "
                    f"ratio={lr['fit_vs_random_ratio']:.2f}x")

        # ================================================================
        # Exp2: Δg/Δu Swap干预 (★★★★)
        # 核心假说: Δg是选择器(swap Δg应改变类别方向)
        #           Δu是调制器(swap Δu应改变幅度/方向细节)
        # ================================================================
        log("=== Exp2: Δg/Δu Swap Intervention ===")
        exp2 = []

        # 选择跨类别的swap对
        swap_pairs = []
        for i, cA in enumerate(cat_names):
            for j, cB in enumerate(cat_names):
                if i >= j: continue
                wA_list = [w for w, c in zip(all_words, all_cats) if c == cA]
                wB_list = [w for w, c in zip(all_words, all_cats) if c == cB]
                # 每对取前2个词
                for wA in wA_list[:2]:
                    for wB in wB_list[:2]:
                        swap_pairs.append((wA, wB, cA, cB))
        log(f"  Swap pairs: {len(swap_pairs)}")

        tl = list(range(0, n_layers, max(1, n_layers // 6)))
        if n_layers - 1 not in tl: tl.append(n_layers - 1)
        tl = sorted(set(tl))

        for li in tl:
            # 类别中心
            cm = {}
            for cat in cat_names:
                cw = [w for w, c in zip(all_words, all_cats) if c == cat and li in word_residuals[w]]
                rv = [word_residuals[w][li] for w in cw if word_residuals[w].get(li) is not None]
                if len(rv) >= 2: cm[cat] = np.mean(rv, axis=0)
            if len(cm) < 2: continue

            # 选8对swap（避免太多）
            test_pairs = swap_pairs[:8]
            dg_swap_cos, du_swap_cos, dg_swap_norm, du_swap_norm = [], [], [], []

            for wA, wB, cA, cB in test_pairs:
                if li not in word_gates.get(wA, {}) or li not in word_gates.get(wB, {}):
                    continue
                if cA not in cm or cB not in cm: continue

                cat_dir = cm[cB] - cm[cA]  # 从A类别指向B类别的方向
                if np.linalg.norm(cat_dir) < 1e-8: continue

                gA = word_gates[wA][li]
                gB = word_gates[wB][li]
                uA = word_ups[wA][li]
                uB = word_ups[wB][li]
                g_bar = layer_mean_g[li]
                u_bar = layer_mean_u[li]

                lw = get_layer_weights(layers_list[li], d_model, mlp_type)
                W_down = lw.W_down

                # 原始MLP输出差异
                orig_diff = W_down @ (gA * uA) - W_down @ (gA * uA)  # = 0 (baseline)

                # Swap Δg: 用gB替换gA (保持uA)
                # 效果: Δ_swap_g = W_down @ (gB ⊙ uA) - W_down @ (gA ⊙ uA) = W_down @ ((gB-gA) ⊙ uA)
                delta_swap_g = W_down @ ((gB - gA) * uA)
                cos_swap_g = proper_cos(delta_swap_g, cat_dir)
                norm_swap_g = float(np.linalg.norm(delta_swap_g))

                # Swap Δu: 用uB替换uA (保持gA)
                # 效果: Δ_swap_u = W_down @ (gA ⊙ uB) - W_down @ (gA ⊙ uA) = W_down @ (gA ⊙ (uB-uA))
                delta_swap_u = W_down @ (gA * (uB - uA))
                cos_swap_u = proper_cos(delta_swap_u, cat_dir)
                norm_swap_u = float(np.linalg.norm(delta_swap_u))

                dg_swap_cos.append(cos_swap_g)
                du_swap_cos.append(cos_swap_u)
                dg_swap_norm.append(norm_swap_g)
                du_swap_norm.append(norm_swap_u)

            if not dg_swap_cos: continue

            lr = {
                "layer": li,
                "n_swaps": len(dg_swap_cos),
                "dg_swap_mean_cos": float(np.mean(dg_swap_cos)),
                "du_swap_mean_cos": float(np.mean(du_swap_cos)),
                "dg_swap_mean_norm": float(np.mean(dg_swap_norm)),
                "du_swap_mean_norm": float(np.mean(du_swap_norm)),
                "dg_vs_du_cos_ratio": float(np.mean(np.abs(dg_swap_cos)) / max(np.mean(np.abs(du_swap_cos)), 1e-6)),
                "dg_vs_du_norm_ratio": float(np.mean(dg_swap_norm) / max(np.mean(du_swap_norm), 1e-6)),
            }
            exp2.append(lr)
            if li in [0, tl[len(tl)//2], tl[-1]]:
                log(f"  L{li}: dg_swap_cos={lr['dg_swap_mean_cos']:+.3f} du_swap_cos={lr['du_swap_mean_cos']:+.3f} "
                    f"norm_ratio={lr['dg_vs_du_norm_ratio']:.2f}")

        # ================================================================
        # Exp3: 线性Probe类别信息追踪 (★★★)
        # 用leave-one-out CV检测每层ū/residual中的类别信息
        # ================================================================
        log("=== Exp3: Linear Probe Category Info ===")
        exp3 = []

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        cat_to_idx = {c: i for i, c in enumerate(cat_names)}
        y = np.array([cat_to_idx[c] for c in all_cats])

        for li in tl:
            vw = [w for w in all_words if li in word_wdu.get(w, {})]
            if len(vw) < 16: continue

            # --- Probe on W_down@ū ---
            X_u = np.array([word_wdu[w][li] for w in all_words], dtype=np.float32)
            # PCA降维（避免过拟合）
            from scipy.sparse.linalg import svds
            n_comp = min(40, len(all_words)-1, d_model-1)
            if n_comp < 2: continue
            try:
                X_u_c = X_u - X_u.mean(axis=0)
                _, s_u, Vt_u = svds(X_u_c, k=n_comp)
                Vt_u = Vt_u[np.argsort(-s_u)]
                X_u_pca = X_u_c @ Vt_u.T
            except Exception:
                continue

            # Leave-one-out per category (8-fold CV)
            acc_u = []
            for fold_cat in cat_names:
                test_idx = [i for i, c in enumerate(all_cats) if c == fold_cat]
                train_idx = [i for i in range(len(all_words)) if i not in test_idx]
                if len(test_idx) < 2 or len(train_idx) < 8: continue

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_u_pca[train_idx])
                X_test = scaler.transform(X_u_pca[test_idx])
                y_train = y[train_idx]
                y_test = y[test_idx]

                clf = LogisticRegression(max_iter=500, C=1.0)
                clf.fit(X_train, y_train)
                acc_u.append(float(clf.score(X_test, y_test)))

            # --- Probe on residual stream ---
            X_r = np.array([word_residuals[w].get(li, np.zeros(d_model)) for w in all_words], dtype=np.float32)
            try:
                X_r_c = X_r - X_r.mean(axis=0)
                _, s_r, Vt_r = svds(X_r_c, k=n_comp)
                Vt_r = Vt_r[np.argsort(-s_r)]
                X_r_pca = X_r_c @ Vt_r.T
            except Exception:
                X_r_pca = X_u_pca  # fallback

            acc_r = []
            for fold_cat in cat_names:
                test_idx = [i for i, c in enumerate(all_cats) if c == fold_cat]
                train_idx = [i for i in range(len(all_words)) if i not in test_idx]
                if len(test_idx) < 2 or len(train_idx) < 8: continue

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_r_pca[train_idx])
                X_test = scaler.transform(X_r_pca[test_idx])
                y_train = y[train_idx]
                y_test = y[test_idx]

                clf = LogisticRegression(max_iter=500, C=1.0)
                clf.fit(X_train, y_train)
                acc_r.append(float(clf.score(X_test, y_test)))

            # --- Probe on W_down@(g⊙u) = full MLP output contribution ---
            X_mlp = np.array([word_wdg_u[w][li] + word_wg_du[w][li] for w in all_words], dtype=np.float32)
            # 这里用Δg⊙ū + ḡ⊙Δu近似（忽略了Δg⊙Δu和ḡ⊙ū的常数部分）
            # 实际上应该用 W_down @ (g⊙u) - W_down @ (ḡ⊙ū)
            # 但word_wdg_u已经是 W_down@((g-ḡ)⊙ū), word_wg_du是 W_down@(ḡ⊙(u-ū))
            # 所以 X_mlp ≈ W_down@((g-ḡ)⊙ū + ḡ⊙(u-ū)) = 概念差异部分

            try:
                X_mlp_c = X_mlp - X_mlp.mean(axis=0)
                _, s_m, Vt_m = svds(X_mlp_c, k=min(n_comp, X_mlp_c.shape[1]-1))
                Vt_m = Vt_m[np.argsort(-s_m)]
                X_mlp_pca = X_mlp_c @ Vt_m.T
            except Exception:
                X_mlp_pca = X_u_pca

            acc_mlp = []
            for fold_cat in cat_names:
                test_idx = [i for i, c in enumerate(all_cats) if c == fold_cat]
                train_idx = [i for i in range(len(all_words)) if i not in test_idx]
                if len(test_idx) < 2 or len(train_idx) < 8: continue

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_mlp_pca[train_idx])
                X_test = scaler.transform(X_mlp_pca[test_idx])
                y_train = y[train_idx]
                y_test = y[test_idx]

                clf = LogisticRegression(max_iter=500, C=1.0)
                clf.fit(X_train, y_train)
                acc_mlp.append(float(clf.score(X_test, y_test)))

            lr = {
                "layer": li,
                "probe_u_acc": float(np.mean(acc_u)) if acc_u else 0,
                "probe_r_acc": float(np.mean(acc_r)) if acc_r else 0,
                "probe_mlp_acc": float(np.mean(acc_mlp)) if acc_mlp else 0,
                "n_folds": len(acc_u),
                "random_baseline": 1.0 / len(cat_names),
            }
            exp3.append(lr)
            log(f"  L{li}: ū_acc={lr['probe_u_acc']:.3f} res_acc={lr['probe_r_acc']:.3f} "
                f"mlp_acc={lr['probe_mlp_acc']:.3f} (rand={lr['random_baseline']:.3f})")

        # ===== 保存结果 =====
        out_dir = f"d:\\Ai2050\\TransformerLens-Project\\results\\causal_fiber\\{model_name}_cclxxvii"
        os.makedirs(out_dir, exist_ok=True)

        def js(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)): return float(obj)
            if isinstance(obj, (np.int32, np.int64)): return int(obj)
            return obj

        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize(x) for x in obj]
            if isinstance(obj, (np.ndarray, np.float32, np.float64, np.int32, np.int64)):
                return js(obj)
            return obj

        for name, data in [("exp1_procrustes", exp1), ("exp2_swap_intervention", exp2), ("exp3_probe", exp3)]:
            with open(os.path.join(out_dir, f"{name}.json"), 'w', encoding='utf-8') as f:
                json.dump(sanitize({"experiment": name, "model": model_name, "n_words": len(all_words),
                                    "timestamp": datetime.now().isoformat(), "results": data}), f, indent=2)
        log(f"Saved to {out_dir}")

        # ===== 摘要 =====
        log("=== SUMMARY ===")
        # Exp1 Procrustes
        if exp1:
            mid = [r for r in exp1 if n_layers*0.3 <= r["l1"] < n_layers*0.7]
            if mid:
                log(f"  Procrustes mid: orth_err={np.mean([r['orth_error'] for r in mid]):.4f} "
                    f"det={np.mean([r['det_R'] for r in mid]):+.3f} "
                    f"fit_cos={np.mean([r['mean_fit_cos'] for r in mid]):.3f} "
                    f"vs random={np.mean([r['mean_fit_cos_random'] for r in mid]):.3f}")
        # Exp2 Swap
        if exp2:
            mid2 = [r for r in exp2 if n_layers*0.3 <= r["layer"] < n_layers*0.7]
            if mid2:
                log(f"  Swap mid: dg_cos={np.mean([r['dg_swap_mean_cos'] for r in mid2]):+.3f} "
                    f"du_cos={np.mean([r['du_swap_mean_cos'] for r in mid2]):+.3f} "
                    f"norm_ratio={np.mean([r['dg_vs_du_norm_ratio'] for r in mid2]):.2f}")
        # Exp3 Probe
        if exp3:
            mid3 = [r for r in exp3 if n_layers*0.3 <= r["layer"] < n_layers*0.7]
            if mid3:
                log(f"  Probe mid: u_acc={np.mean([r['probe_u_acc'] for r in mid3]):.3f} "
                    f"r_acc={np.mean([r['probe_r_acc'] for r in mid3]):.3f} "
                    f"mlp_acc={np.mean([r['probe_mlp_acc'] for r in mid3]):.3f}")

        release_model(model)
        log(f"=== {model_name} COMPLETE ===")
    except Exception as e:
        log(f"ERROR: {e}")
        log(traceback.format_exc())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    with open(LOG, 'w', encoding='utf-8') as f:
        import time as t
        f.write(f"[{t.strftime('%H:%M:%S')}] === CCLXXVII NEW RUN: {args.model} ===\n")
    run_model(args.model)
