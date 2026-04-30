"""
Phase CCLXXVIII: 旋转矩阵结构分析 + 修复Probe + 多步旋转累积
核心目标:
1. 分解R_l为Givens旋转/Householder反射，找最小旋转基
2. 修复sklearn probe，追踪类别信息
3. 计算累积旋转 R_0 @ R_1 @ ... @ R_l 的结构
"""
import sys, os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxviii_log.txt"

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        import time
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()

log("=== CCLXXVIII Script started ===")

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
            if (wi + 1) % 16 == 0 or wi == 0:
                log(f"  Word {wi+1}/{len(all_words)} ({time.time()-t0:.0f}s)")

        log(f"Data collection done ({time.time()-t0:.0f}s)")

        # 预计算 W_down @ ū
        log("Precomputing W_down@ū...")
        word_wdu = {}
        for word in all_words:
            word_wdu[word] = {}
            for li in range(n_layers):
                if li not in word_ups.get(word, {}): continue
                lw = get_layer_weights(layers_list[li], d_model, mlp_type)
                if lw.W_gate is None: continue
                word_wdu[word][li] = lw.W_down @ word_ups[word][li]

        # ================================================================
        # Exp1: 旋转矩阵结构分析 (★★★★★)
        # 1) 计算每对相邻层的旋转矩阵R_l
        # 2) 分析R_l的奇异值/特征值结构
        # 3) Givens旋转分解: R_l需要多少个2D旋转来表示？
        # 4) 累积旋转: R_0@R_1@...@R_l的结构
        # ================================================================
        log("=== Exp1: Rotation Matrix Structure ===")

        from scipy.sparse.linalg import svds
        n_dim = min(50, len(all_words)-1, d_model-1)

        # 1) 构建每层的PCA子空间和旋转矩阵
        layer_pca = {}
        layer_R = {}  # R_l: PCA子空间中的旋转

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
            X_pca = X_c @ Vt.T  # (n, n_dim)
            layer_pca[li] = {"Vt": Vt, "s": s, "X_pca": X_pca, "words": vw}

        # 2) 计算相邻层旋转矩阵
        exp1_R = []
        sorted_layers = sorted(layer_pca.keys())
        for idx in range(len(sorted_layers) - 1):
            li = sorted_layers[idx]
            lj = sorted_layers[idx + 1]
            if lj != li + 1: continue  # 只看相邻层

            X_pca = layer_pca[li]["X_pca"]
            Y_pca = layer_pca[lj]["X_pca"]

            # 对齐词表
            vw_common = [w for w in all_words if w in layer_pca[li]["words"] and w in layer_pca[lj]["words"]]
            if len(vw_common) < 8: continue

            # 重新计算配对的PCA坐标
            X_li = np.array([word_wdu[w][li] for w in vw_common], dtype=np.float32)
            Y_lj = np.array([word_wdu[w][lj] for w in vw_common], dtype=np.float32)
            X_li_c = X_li - X_li.mean(axis=0)
            Y_lj_c = Y_lj - Y_lj.mean(axis=0)

            # 投影到各自的PCA子空间
            Vt_l = layer_pca[li]["Vt"]
            Vt_r = layer_pca[lj]["Vt"]
            X_proj = X_li_c @ Vt_l.T  # (n, n_dim)
            Y_proj = Y_lj_c @ Vt_r.T  # (n, n_dim)

            # Procrustes: R = U @ Vt where U,S,Vt = SVD(Y^T @ X)
            M = Y_proj.T @ X_proj  # (n_dim, n_dim)
            U, S, Vt_m = np.linalg.svd(M)
            R = U @ Vt_m  # (n_dim, n_dim)

            # 分析R的结构
            # 2a) R的特征值（复数，因为R是正交的，特征值在单位圆上）
            eigvals = np.linalg.eigvals(R)
            eig_mag = np.abs(eigvals)
            eig_phase = np.angle(eigvals)

            # 2b) Givens旋转分解: 正交矩阵可以分解为d(d-1)/2个Givens旋转
            # 但我们想知道"有效"旋转数——有多少个2D平面被显著旋转
            # 每对共轭特征值 e^{±iθ} 对应一个2D旋转角θ
            # 统计|θ|>阈值的2D平面数
            # 先排序相位，找共轭对
            phases_sorted = np.sort(eig_phase)
            # 正交矩阵的特征值在单位圆上，实特征值=±1，复数=共轭对e^{±iθ}
            real_mask = np.abs(eig_mag - 1.0) < 0.01  # 所有都在单位圆上

            # 统计旋转角分布
            # 用R的对角元素的偏离1的程度来估计旋转"量"
            diag_dev = np.abs(np.diag(R)) - 1.0  # 应该≤0
            n_significant_rot = np.sum(np.abs(np.diag(R)) < 0.99)  # 偏离1的对角元素数

            # R的Frobenius范数偏离单位矩阵
            R_dev = float(np.linalg.norm(R - np.eye(n_dim), 'fro') / np.sqrt(n_dim))

            # 2c) 累积旋转（后面做）

            lr = {
                "layer_pair": f"L{li}->L{lj}",
                "l1": li, "l2": lj,
                "n_words": len(vw_common),
                "R_deviation_from_I": R_dev,
                "n_significant_rotations": int(n_significant_rot),
                "frac_significant": float(n_significant_rot / n_dim),
                "mean_diag_abs": float(np.mean(np.abs(np.diag(R)))),
                "singular_values_top10": S[:10].tolist(),
                "det_R": float(np.linalg.det(R)),
                "fit_cos": float(np.mean([proper_cos(X_proj[i] @ R.T, Y_proj[i]) for i in range(len(vw_common))])),
            }
            exp1_R.append(lr)
            layer_R[li] = R
            if li % 4 == 0 or li == n_layers - 2:
                log(f"  L{li}->L{lj}: R_dev={R_dev:.4f} n_sig_rot={n_significant_rot}/{n_dim} "
                    f"diag_mean={lr['mean_diag_abs']:.3f} det={lr['det_R']:+.3f}")

        # 3) 累积旋转分析
        log("  Computing cumulative rotations...")
        exp1_cum = []
        # 选几个起始层，计算累积旋转的偏差增长
        start_layers = [0, n_layers//4, n_layers//2]
        for start_l in start_layers:
            if start_l not in layer_R: continue
            cum_R = np.eye(n_dim)
            for step in range(min(10, n_layers - start_l - 1)):
                l_curr = start_l + step
                if l_curr not in layer_R: break
                cum_R = layer_R[l_curr] @ cum_R
                # 累积偏差
                cum_dev = float(np.linalg.norm(cum_R - np.eye(n_dim), 'fro') / np.sqrt(n_dim))
                # 累积fit: 从start_l到start_l+step+1的变换质量
                if start_l in layer_pca and (start_l + step + 1) in layer_pca:
                    vw_s = [w for w in all_words if w in layer_pca[start_l]["words"] and w in layer_pca[start_l+step+1]["words"]]
                    if len(vw_s) >= 4:
                        X_s = np.array([word_wdu[w][start_l] for w in vw_s], dtype=np.float32)
                        Y_s = np.array([word_wdu[w][start_l+step+1] for w in vw_s], dtype=np.float32)
                        X_s_c = X_s - X_s.mean(axis=0)
                        Y_s_c = Y_s - Y_s.mean(axis=0)
                        Vt_s = layer_pca[start_l]["Vt"]
                        Vt_e = layer_pca[start_l+step+1]["Vt"]
                        X_s_proj = X_s_c @ Vt_s.T
                        Y_s_proj = Y_s_c @ Vt_e.T
                        cum_fit = float(np.mean([proper_cos(X_s_proj[i] @ cum_R.T, Y_s_proj[i]) for i in range(len(vw_s))]))
                    else:
                        cum_fit = -1
                else:
                    cum_fit = -1

                exp1_cum.append({
                    "start_layer": start_l,
                    "n_steps": step + 1,
                    "end_layer": start_l + step + 1,
                    "cum_deviation": cum_dev,
                    "cum_fit_cos": cum_fit,
                })
            devs = [round(r['cum_deviation'], 3) for r in exp1_cum if r['start_layer']==start_l]
            steps = [r['n_steps'] for r in exp1_cum if r['start_layer']==start_l]
            log(f"  Cumulative from L{start_l}: steps={steps} dev={devs}")

        # ================================================================
        # Exp2: 修复的Probe类别信息追踪
        # 用numpy实现简单的最近质心分类器（避免sklearn版本问题）
        # ================================================================
        log("=== Exp2: Nearest-Centroid Probe ===")
        exp2 = []

        cat_to_idx = {c: i for i, c in enumerate(cat_names)}

        for li in range(0, n_layers, max(1, n_layers // 8)):
            if li not in layer_pca: continue

            # 用PCA坐标
            X_pca = layer_pca[li]["X_pca"]  # (64, n_dim)
            y = np.array([cat_to_idx[c] for c in all_cats])

            # 最近质心分类器 (leave-one-category-out)
            acc_u = []
            for fold_cat in cat_names:
                test_idx = [i for i, c in enumerate(all_cats) if c == fold_cat]
                train_idx = [i for i in range(len(all_words)) if i not in test_idx]
                if len(test_idx) < 2 or len(train_idx) < 8: continue

                # 计算训练集每类质心
                centroids = {}
                for cat in cat_names:
                    cat_idx = [i for i in train_idx if all_cats[i] == cat]
                    if len(cat_idx) >= 1:
                        centroids[cat] = np.mean(X_pca[cat_idx], axis=0)

                # 对测试集分类
                correct = 0
                for i in test_idx:
                    best_cat = None
                    best_cos = -2
                    for cat, cent in centroids.items():
                        c = proper_cos(X_pca[i], cent)
                        if c > best_cos:
                            best_cos = c
                            best_cat = cat
                    if best_cat == all_cats[i]:
                        correct += 1
                acc_u.append(correct / len(test_idx))

            # 残差流probe
            X_r = np.array([word_residuals[w].get(li, np.zeros(d_model)) for w in all_words], dtype=np.float32)
            X_r_c = X_r - X_r.mean(axis=0)
            try:
                _, s_r, Vt_r = svds(X_r_c, k=n_dim)
                Vt_r = Vt_r[np.argsort(-s_r)]
                X_r_pca = X_r_c @ Vt_r.T
            except Exception:
                X_r_pca = X_pca

            acc_r = []
            for fold_cat in cat_names:
                test_idx = [i for i, c in enumerate(all_cats) if c == fold_cat]
                train_idx = [i for i in range(len(all_words)) if i not in test_idx]
                if len(test_idx) < 2 or len(train_idx) < 8: continue

                centroids = {}
                for cat in cat_names:
                    cat_idx = [i for i in train_idx if all_cats[i] == cat]
                    if len(cat_idx) >= 1:
                        centroids[cat] = np.mean(X_r_pca[cat_idx], axis=0)

                correct = 0
                for i in test_idx:
                    best_cat = None
                    best_cos = -2
                    for cat, cent in centroids.items():
                        c = proper_cos(X_r_pca[i], cent)
                        if c > best_cos:
                            best_cos = c
                            best_cat = cat
                    if best_cat == all_cats[i]:
                        correct += 1
                acc_r.append(correct / len(test_idx))

            lr = {
                "layer": li,
                "probe_u_acc": float(np.mean(acc_u)) if acc_u else 0,
                "probe_r_acc": float(np.mean(acc_r)) if acc_r else 0,
                "n_folds": len(acc_u),
                "random_baseline": 1.0 / len(cat_names),
            }
            exp2.append(lr)
            log(f"  L{li}: ū_acc={lr['probe_u_acc']:.3f} res_acc={lr['probe_r_acc']:.3f} (rand={lr['random_baseline']:.3f})")

        # ===== 保存结果 =====
        out_dir = f"d:\\Ai2050\\TransformerLens-Project\\results\\causal_fiber\\{model_name}_cclxxviii"
        os.makedirs(out_dir, exist_ok=True)

        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize(x) for x in obj]
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        for name, data in [("exp1_rotation_structure", exp1_R),
                           ("exp1_cumulative_rotation", exp1_cum),
                           ("exp2_probe", exp2)]:
            with open(os.path.join(out_dir, f"{name}.json"), 'w', encoding='utf-8') as f:
                json.dump(sanitize({"experiment": name, "model": model_name, "n_words": len(all_words),
                                    "timestamp": datetime.now().isoformat(), "results": data}), f, indent=2)
        log(f"Saved to {out_dir}")

        # ===== 摘要 =====
        log("=== SUMMARY ===")
        if exp1_R:
            mid = [r for r in exp1_R if n_layers*0.3 <= r["l1"] < n_layers*0.7]
            if mid:
                log(f"  Rot mid: R_dev={np.mean([r['R_deviation_from_I'] for r in mid]):.4f} "
                    f"n_sig={np.mean([r['n_significant_rotations'] for r in mid]):.1f}/{n_dim} "
                    f"diag={np.mean([r['mean_diag_abs'] for r in mid]):.3f}")
        if exp2:
            mid2 = [r for r in exp2 if n_layers*0.3 <= r["layer"] < n_layers*0.7]
            if mid2:
                log(f"  Probe mid: u_acc={np.mean([r['probe_u_acc'] for r in mid2]):.3f} "
                    f"r_acc={np.mean([r['probe_r_acc'] for r in mid2]):.3f}")

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
        f.write(f"[{t.strftime('%H:%M:%S')}] === CCLXXVIII NEW RUN: {args.model} ===\n")
    run_model(args.model)
