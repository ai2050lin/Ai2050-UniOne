"""
Phase CCLXXVI: 扩大词表(16→64词)验证n90可靠性 + 语义脊线追踪
核心目标: 1) n90是否随词表扩大而收敛 2) Δu子空间结构是否稳健
3) 同一概念在各层Δu子空间的投影轨迹(语义脊线)
"""
import sys, os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxvi_log.txt"

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        import time
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()

log("=== CCLXXVI Script started ===")

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

# ===== 扩展词表: 8类别 × 8词 = 64词 =====
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

# 原始16词对照（用于收敛性分析）
CONCEPTS_ORIGINAL = {
    "animal": ["dog", "cat", "horse", "bird"],
    "food": ["apple", "bread", "cheese", "rice"],
    "tool": ["hammer", "knife", "scissors", "saw"],
    "vehicle": ["car", "bus", "train", "plane"],
}


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
    s = s[np.argsort(-s)]
    tv = np.sum(s**2)
    if tv < 1e-10:
        return -1, -1, -1
    cumvar = np.cumsum(s**2) / tv
    n90 = int(np.searchsorted(cumvar, 0.90)) + 1
    n95 = int(np.searchsorted(cumvar, 0.95)) + 1
    top5 = float(np.sum(s[:5]**2) / tv)
    return n90, n95, top5


def compute_n90_subsampled(vectors, sample_sizes, rng, n_trials=3):
    """Compute n90 at different subsample sizes to test convergence."""
    results = {}
    for ss in sample_sizes:
        n90s, n95s, t5s = [], [], []
        for trial in range(n_trials):
            idx = rng.choice(len(vectors), min(ss, len(vectors)), replace=False)
            sub = [vectors[i] for i in idx]
            n90, n95, t5 = compute_n90(sub)
            if n90 > 0:
                n90s.append(n90)
                n95s.append(n95)
                t5s.append(t5)
        if n90s:
            results[ss] = {
                "n90_mean": float(np.mean(n90s)),
                "n90_std": float(np.std(n90s)),
                "n95_mean": float(np.mean(n95s)) if n95s else -1,
                "top5_mean": float(np.mean(t5s)) if t5s else -1,
                "n_trials": len(n90s),
            }
    return results


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
            log(f"  Word {wi+1}/{len(all_words)}: '{word}' ({elapsed:.0f}s)")

        log(f"Data collection done ({time.time()-t0:.0f}s)")

        # Target layers (约6-8层)
        tl = list(range(0, n_layers, max(1, n_layers // 6)))
        if n_layers - 1 not in tl: tl.append(n_layers - 1)
        tl = sorted(set(tl))
        log(f"Target layers: {tl}")

        # ================================================================
        # Exp1: n90收敛性分析 (★★★★★ 最关键的新实验)
        # ================================================================
        log("=== Exp1: n90 Convergence ===")
        exp1 = []
        sample_sizes = [16, 24, 32, 48, 64]
        rng_sub = np.random.RandomState(123)

        for li in tl:
            vw = [w for w in all_words if li in word_gates[w] and li in word_ups[w]]
            if len(vw) < 16: continue

            # 生成所有差异向量
            dg_all = [word_gates[vw[i]][li] - word_gates[vw[j]][li]
                      for i in range(len(vw)) for j in range(i+1, len(vw))]
            du_all = [word_ups[vw[i]][li] - word_ups[vw[j]][li]
                      for i in range(len(vw)) for j in range(i+1, len(vw))]

            # 全量n90
            n90_dg_full, n95_dg_full, top5_dg = compute_n90(dg_all)
            n90_du_full, n95_du_full, top5_du = compute_n90(du_all)

            # 子采样n90（检查收敛性）
            dg_conv = compute_n90_subsampled(dg_all, sample_sizes, rng_sub, n_trials=3)
            du_conv = compute_n90_subsampled(du_all, sample_sizes, rng_sub, n_trials=3)

            lr = {
                "layer": li,
                "n_words": len(vw),
                "n_pairs": len(dg_all),
                "n90_dg_full": n90_dg_full,
                "n95_dg_full": n95_dg_full,
                "n90_du_full": n90_du_full,
                "n95_du_full": n95_du_full,
                "top5_var_dg": top5_dg,
                "top5_var_du": top5_du,
                "dg_convergence": dg_conv,
                "du_convergence": du_conv,
            }
            exp1.append(lr)
            # 关键日志：n90是否收敛
            conv_str = ""
            for ss in sample_sizes:
                if ss in dg_conv and ss in du_conv:
                    conv_str += f" n={ss}:dg_n90={dg_conv[ss]['n90_mean']:.1f}±{dg_conv[ss]['n90_std']:.1f}/du_n90={du_conv[ss]['n90_mean']:.1f}±{du_conv[ss]['n90_std']:.1f}"
            log(f"  L{li}: dg_n90={n90_dg_full} du_n90={n90_du_full}{conv_str}")

        # ================================================================
        # Exp2: 方向分解 (与CCLXXV Exp1相同，但64词)
        # ================================================================
        log("=== Exp2: Direction Decomposition (64 words) ===")
        exp2 = []
        for li in tl:
            vw = [w for w in all_words if li in word_gates[w] and li in word_ups[w]]
            if len(vw) < 8: continue
            lw = get_layer_weights(layers_list[li], d_model, mlp_type)
            if lw.W_gate is None: continue
            W_down = lw.W_down

            cm = {}
            for cat in cat_names:
                cw = [w for w, c in zip(all_words, all_cats) if c == cat and li in word_residuals[w]]
                rv = [word_residuals[w][li] for w in cw if word_residuals[w].get(li) is not None]
                if len(rv) >= 2: cm[cat] = np.mean(rv, axis=0)
            if len(cm) < 2: continue

            gc_l, uc_l, tc_l, dgu_l, dg_l = [], [], [], [], []
            gn_l, un_l = [], []
            cl = sorted(cm.keys())
            for i, cA in enumerate(cl):
                for j, cB in enumerate(cl):
                    if i >= j: continue
                    cdir = cm[cA] - cm[cB]
                    if np.linalg.norm(cdir) < 1e-8: continue
                    wA = [w for w, c in zip(all_words, all_cats) if c == cA and li in word_gates[w]]
                    wB = [w for w, c in zip(all_words, all_cats) if c == cB and li in word_gates[w]]
                    gA = np.mean([word_gates[w][li] for w in wA], axis=0)
                    gB = np.mean([word_gates[w][li] for w in wB], axis=0)
                    uA = np.mean([word_ups[w][li] for w in wA], axis=0)
                    uB = np.mean([word_ups[w][li] for w in wB], axis=0)
                    Dg, Du = gA - gB, uA - uB
                    gb, ub = (gA + gB) / 2, (uA + uB) / 2
                    gv = W_down @ (Dg * ub)
                    uv = W_down @ (gb * Du)
                    tv = gv + uv + W_down @ (Dg * Du)
                    gc_l.append(proper_cos(gv, cdir))
                    uc_l.append(proper_cos(uv, cdir))
                    tc_l.append(proper_cos(tv, cdir))
                    dgu_l.append(proper_cos(W_down @ Du, cdir))
                    dg_l.append(proper_cos(W_down @ Dg, cdir))
                    gn_l.append(float(np.linalg.norm(gv)))
                    un_l.append(float(np.linalg.norm(uv)))
            if not gc_l: continue
            tn = max(sum(gn_l) + sum(un_l), 1e-10)
            lr = {"layer": li, "n_pairs": len(gc_l),
                  "gate_cos": float(np.mean(gc_l)), "up_cos": float(np.mean(uc_l)),
                  "total_cos": float(np.mean(tc_l)), "W_down_Du_cos": float(np.mean(dgu_l)),
                  "W_down_Dg_cos": float(np.mean(dg_l)),
                  "gate_norm_frac": float(sum(gn_l)/tn), "up_norm_frac": float(sum(un_l)/tn)}
            exp2.append(lr)
            log(f"  L{li}: gate={lr['gate_cos']:+.3f} up={lr['up_cos']:+.3f} Du={lr['W_down_Du_cos']:+.3f}")

        # ================================================================
        # Exp3: Δu子空间正交性 (64词版)
        # ================================================================
        log("=== Exp3: Subspace Orthogonality (64 words) ===")
        tl8 = list(range(0, n_layers, max(1, n_layers // 8)))
        if n_layers - 1 not in tl8: tl8.append(n_layers - 1)
        tl8 = sorted(set(tl8))

        from scipy.sparse.linalg import svds
        lpcas = {}
        for li in tl8:
            vw = [w for w in all_words if li in word_gates[w] and li in word_ups[w]]
            if len(vw) < 8: continue
            dg_l = [word_gates[vw[i]][li] - word_gates[vw[j]][li]
                    for i in range(len(vw)) for j in range(i+1, len(vw))]
            du_l = [word_ups[vw[i]][li] - word_ups[vw[j]][li]
                    for i in range(len(vw)) for j in range(i+1, len(vw))]
            if len(dg_l) < 10: continue
            dg_a = np.array(dg_l, dtype=np.float32)
            du_a = np.array(du_l, dtype=np.float32)
            np_dg = min(30, len(dg_l)-1, dg_a.shape[1]-1)
            np_du = min(30, len(du_l)-1, du_a.shape[1]-1)
            if np_dg < 2 or np_du < 2: continue
            dg_c = dg_a - dg_a.mean(axis=0)
            _, s_dg, Vt_dg = svds(dg_c.astype(np.float32), k=np_dg)
            idx = np.argsort(-s_dg); s_dg = s_dg[idx]; Vt_dg = Vt_dg[idx]
            du_c = du_a - du_a.mean(axis=0)
            _, s_du, Vt_du = svds(du_c.astype(np.float32), k=np_du)
            idx2 = np.argsort(-s_du); s_du = s_du[idx2]; Vt_du = Vt_du[idx2]
            ns = min(20, len(s_dg), len(s_du))
            lpcas[li] = {"Vt_dg": Vt_dg[:ns], "Vt_du": Vt_du[:ns]}

        du_rot, same_l = [], []
        ll = sorted(lpcas.keys())
        for i, l1 in enumerate(ll):
            for j, l2 in enumerate(ll):
                if i >= j: continue
                _, ca, _ = np.linalg.svd(lpcas[l1]["Vt_du"] @ lpcas[l2]["Vt_du"].T)
                du_rot.append({"l1": l1, "l2": l2, "mean_cos": float(np.mean(ca))})
        for li in ll:
            _, ca_dg, _ = np.linalg.svd(lpcas[li]["Vt_dg"] @ lpcas[li]["Vt_du"].T)
            same_l.append({"layer": li, "dg_du_cos": float(np.mean(ca_dg))})
            log(f"  L{li}: dg_du_cos={float(np.mean(ca_dg)):.4f}")

        # ================================================================
        # Exp4: 语义脊线追踪 (★★★ 新实验)
        # 追踪同一概念在各层Δu子空间中的投影轨迹
        # ================================================================
        log("=== Exp4: Semantic Ridge Tracking ===")
        exp4 = []
        for li in tl:
            vw = [w for w in all_words if li in word_ups[w] and li in word_gates[w]]
            if len(vw) < 8: continue
            lw = get_layer_weights(layers_list[li], d_model, mlp_type)
            if lw.W_gate is None: continue
            W_down = lw.W_down

            # 1) 类内一致性: 同类词的W_down@ū余弦
            cat_consistency = {}
            for cat in cat_names:
                cw = [w for w, c in zip(all_words, all_cats) if c == cat and li in word_ups[w]]
                if len(cw) < 2: continue
                wu = [W_down @ word_ups[w][li] for w in cw]
                pairs_cos = []
                for i in range(len(wu)):
                    for j in range(i+1, len(wu)):
                        pairs_cos.append(proper_cos(wu[i], wu[j]))
                cat_consistency[cat] = {
                    "n_words": len(cw),
                    "mean_intra_cos": float(np.mean(pairs_cos)) if pairs_cos else 0,
                }

            # 2) 类间分离: 不同类别平均ū的方向余弦
            cat_means = {}
            for cat in cat_names:
                cw = [w for w, c in zip(all_words, all_cats) if c == cat and li in word_ups[w]]
                if len(cw) < 1: continue
                wu = [W_down @ word_ups[w][li] for w in cw]
                cat_means[cat] = np.mean(wu, axis=0)

            inter_cos = []
            cl = sorted(cat_means.keys())
            for i, cA in enumerate(cl):
                for j, cB in enumerate(cl):
                    if i >= j: continue
                    inter_cos.append(proper_cos(cat_means[cA], cat_means[cB]))

            # 3) 语义脊线: 追踪每个词在各层W_down@ū的投影
            # (这个在主循环外做，因为需要跨层数据)

            lr = {
                "layer": li,
                "cat_consistency": cat_consistency,
                "inter_cat_mean_cos": float(np.mean(inter_cos)) if inter_cos else 0,
                "n_categories": len(cat_means),
            }
            exp4.append(lr)
            intra_avg = np.mean([v["mean_intra_cos"] for v in cat_consistency.values()])
            log(f"  L{li}: intra_cos={intra_avg:.3f} inter_cos={lr['inter_cat_mean_cos']:.3f} n_cat={len(cat_means)}")

        # ================================================================
        # Exp5: 语义脊线跨层追踪
        # 追踪每个词在各层W_down@ū的投影余弦变化
        # ================================================================
        log("=== Exp5: Cross-layer Semantic Ridge ===")
        exp5 = []
        # 对每个词，计算相邻层W_down@ū的余弦相似度
        for word in all_words:
            if word not in word_ups: continue
            layers_available = sorted([li for li in word_ups[word] if li in word_ups[word]])
            if len(layers_available) < 3: continue

            ridge = []
            for idx_l in range(len(layers_available) - 1):
                l1 = layers_available[idx_l]
                l2 = layers_available[idx_l + 1]
                lw1 = get_layer_weights(layers_list[l1], d_model, mlp_type)
                lw2 = get_layer_weights(layers_list[l2], d_model, mlp_type)
                if lw1.W_gate is None or lw2.W_gate is None: continue
                wu1 = lw1.W_down @ word_ups[word][l1]
                wu2 = lw2.W_down @ word_ups[word][l2]
                ridge.append({
                    "l1": l1, "l2": l2,
                    "adjacent_cos": proper_cos(wu1, wu2),
                })

            if ridge:
                cat = all_cats[all_words.index(word)]
                exp5.append({
                    "word": word, "category": cat,
                    "mean_adj_cos": float(np.mean([r["adjacent_cos"] for r in ridge])),
                    "min_adj_cos": float(np.min([r["adjacent_cos"] for r in ridge])),
                    "ridge": ridge,
                })

        # 按类别汇总脊线
        cat_ridge_summary = {}
        for entry in exp5:
            cat = entry["category"]
            if cat not in cat_ridge_summary:
                cat_ridge_summary[cat] = []
            cat_ridge_summary[cat].append(entry["mean_adj_cos"])
        for cat in cat_ridge_summary:
            vals = cat_ridge_summary[cat]
            cat_ridge_summary[cat] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n_words": len(vals),
            }
        log(f"Ridge: n_words={len(exp5)}, categories={len(cat_ridge_summary)}")
        for cat, v in sorted(cat_ridge_summary.items()):
            log(f"  {cat}: adj_cos={v['mean']:.3f}±{v['std']:.3f}")

        # ===== 保存结果 =====
        out_dir = f"d:\\Ai2050\\TransformerLens-Project\\results\\causal_fiber\\{model_name}_cclxxvi"
        os.makedirs(out_dir, exist_ok=True)

        def js(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)): return float(obj)
            if isinstance(obj, (np.int32, np.int64)): return int(obj)
            if isinstance(obj, dict): return {k: js(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)): return [js(x) for x in obj]
            return obj

        for name, data in [("exp1_n90_convergence", exp1), ("exp2_direction_decomposition", exp2),
                           ("exp4_semantic_consistency", exp4)]:
            with open(os.path.join(out_dir, f"{name}.json"), 'w', encoding='utf-8') as f:
                json.dump(js({"experiment": name, "model": model_name, "n_words": len(all_words),
                              "timestamp": datetime.now().isoformat(), "layer_results": data}), f, indent=2)

        with open(os.path.join(out_dir, "exp3_subspace_orthogonality.json"), 'w', encoding='utf-8') as f:
            json.dump(js({"experiment": "exp3", "model": model_name, "n_words": len(all_words),
                          "du_cross_layer": du_rot, "dg_du_same_layer": same_l}), f, indent=2)

        with open(os.path.join(out_dir, "exp5_semantic_ridge.json"), 'w', encoding='utf-8') as f:
            json.dump(js({"experiment": "exp5", "model": model_name, "n_words": len(all_words),
                          "word_ridges": exp5, "cat_ridge_summary": cat_ridge_summary}), f, indent=2)

        log(f"Saved to {out_dir}")

        # ===== 摘要 =====
        log("=== SUMMARY ===")
        # Exp1 n90收敛性
        for r in exp1:
            li = r["layer"]
            full_dg = r["n90_dg_full"]
            full_du = r["n90_du_full"]
            # 检查n16 vs n64
            n16_dg = r["dg_convergence"].get("16", {}).get("n90_mean", -1)
            n64_dg = r["dg_convergence"].get("64", {}).get("n90_mean", -1)
            n16_du = r["du_convergence"].get("16", {}).get("n90_mean", -1)
            n64_du = r["du_convergence"].get("64", {}).get("n90_mean", -1)
            log(f"  L{li} n90: dg[16]={n16_dg:.1f}→64={n64_dg:.1f}(full={full_dg}) du[16]={n16_du:.1f}→64={n64_du:.1f}(full={full_du})")

        # Exp2 方向分解
        mid2 = [r for r in exp2 if n_layers*0.3 <= r["layer"] < n_layers*0.7]
        if mid2:
            log(f"  Mid-layer: gate_cos={np.mean([r['gate_cos'] for r in mid2]):+.3f} "
                f"up_cos={np.mean([r['up_cos'] for r in mid2]):+.3f} "
                f"Du_cos={np.mean([r['W_down_Du_cos'] for r in mid2]):+.3f}")

        # Exp3 正交性
        if same_l:
            mid3 = [r for r in same_l if n_layers*0.3 <= r["layer"] < n_layers*0.7]
            if mid3:
                log(f"  Mid dg_du_cos={np.mean([r['dg_du_cos'] for r in mid3]):.4f}")
        if du_rot:
            near = [r for r in du_rot if abs(r["l2"]-r["l1"]) <= max(2, n_layers//8)]
            if near:
                log(f"  Near-layer du_rot={np.mean([r['mean_cos'] for r in near]):.4f}")

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
        f.write(f"[{t.strftime('%H:%M:%S')}] === CCLXXVI NEW RUN: {args.model} ===\n")
    run_model(args.model)
