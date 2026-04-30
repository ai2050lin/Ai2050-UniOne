"""
Phase CCLXXX: 256词数据积累 — 纯经验实验
=========================================
优先级: 数据积累 >> 理论分析

Exp1: 256词n90收敛 — 确认n90∝ln(n)假说
  - 32类别 × 8词 = 256词
  - 在8/16/32/64/128/192/256处采样n90
  - 7个数据点拟合增长曲线

Exp2: 偏移空间维度分析 — δ = W_down@ū - mean(W_down@ū)的有效维度
  - 对δ做PCA，计算n50/n90/n95
  - 对比δ与原始W_down@ū的维度差异
  - 这是纯数据，不做理论解读

Exp3: 32类Leave-One-Category-Out probe
  - 32类probe vs 16类probe的准确率对比
  - 类别更多时probe是否更难

Exp4: 类内/类间方差分析
  - 对每层计算within-category variance和between-category variance
  - 计算F-ratio = between/within
  - 纯数据积累
"""

import sys, os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxx_log.txt"

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        import time
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()

log("=== CCLXXX Script started ===")

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

# ===== 256词表: 32类别 × 8词 =====
CONCEPTS_256 = {
    # 原始16类 (from CCLXXIX)
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
    # 新增16类
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


def proper_cos(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def compute_n90(vectors, max_k=100):
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
    n50 = int(np.searchsorted(cum_var, 0.50)) + 1
    n95 = int(np.searchsorted(cum_var, 0.95)) + 1
    top5_var = float(var_explained[:5].sum() / total_var)
    return n90, n50, n95, top5_var, int(c.shape[0])


def compute_nXX(vectors, threshold=0.90, max_k=100):
    """Compute n{threshold} from a list of vectors."""
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
    n_t = int(np.searchsorted(cum_var, threshold)) + 1
    top5_var = float(var_explained[:5].sum() / total_var)
    return n_t, top5_var, int(c.shape[0])


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

        # 计算每层均值（轻量）
        layer_mean_u = {}
        layer_mean_g = {}
        for li in range(n_layers):
            us = [word_ups[w][li] for w in all_words if li in word_ups.get(w, {})]
            gs = [word_gates[w][li] for w in all_words if li in word_gates.get(w, {})]
            if us:
                layer_mean_u[li] = np.mean(us, axis=0)
            if gs:
                layer_mean_g[li] = np.mean(gs, axis=0)

        log("Layer means computed")

        # 辅助函数：逐层计算W_down投影（避免一次性缓存所有层的W_down结果）
        def get_wdu_for_layer(li):
            """返回该层所有词的W_down@ū列表和词名"""
            lw = get_layer_weights(layers_list[li], d_model, mlp_type)
            if lw.W_gate is None: return [], []
            wdu_list, wnames = [], []
            for w in all_words:
                if li in word_ups.get(w, {}):
                    wdu_list.append(lw.W_down @ word_ups[w][li])
                    wnames.append(w)
            return wdu_list, wnames

        def get_wd_du_for_layer(li):
            """返回该层所有词的W_down@Δu列表"""
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

        def get_dg_u_for_layer(li):
            """返回该层所有词的W_down@(Δg⊙ū)列表"""
            if li not in layer_mean_u or li not in layer_mean_g: return [], []
            lw = get_layer_weights(layers_list[li], d_model, mlp_type)
            if lw.W_gate is None: return [], []
            dgu_list, wnames = [], []
            mu = layer_mean_u[li]
            mg = layer_mean_g[li]
            for w in all_words:
                if li in word_ups.get(w, {}) and li in word_gates.get(w, {}):
                    dg = word_gates[w][li] - mg
                    dgu_list.append(lw.W_down @ (dg * mu))
                    wnames.append(w)
            return dgu_list, wnames

        # ================================================================
        # Exp1: 256词n90收敛 — 在不同词数处采样
        # ================================================================
        log("=== Exp1: 256-word n90 convergence ===")

        # 采样点: 使用前N个词(保持类别平衡)
        sample_sizes = [8, 16, 32, 64, 128, 192, 256]
        # 按类别均匀采样
        def sample_words(n_target, rng_seed=42):
            """均匀采样n_target个词, 尽量每类相同数量"""
            rng_local = np.random.RandomState(rng_seed)
            cats_sorted = sorted(CONCEPTS_256.keys())
            per_cat = max(1, n_target // len(cats_sorted))
            selected = []
            for cat in cats_sorted:
                ws = [w for w, c in zip(all_words, all_cats) if c == cat]
                n_pick = min(per_cat, len(ws))
                sel = rng_local.choice(ws, n_pick, replace=False)
                selected.extend(sel.tolist() if hasattr(sel, 'tolist') else list(sel))
            # 如果不够，从剩余词中随机补
            if len(selected) < n_target:
                remaining = [w for w in all_words if w not in selected]
                extra = rng_local.choice(remaining, min(n_target - len(selected), len(remaining)), replace=False)
                selected.extend(extra.tolist() if hasattr(extra, 'tolist') else list(extra))
            return selected[:n_target]

        # 选择代表性层
        probe_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
        probe_layers = [l for l in probe_layers if l < n_layers]

        exp1_results = []
        for n_w in sample_sizes:
            if n_w > len(all_words): continue
            sampled = sample_words(n_w)
            log(f"  n_words={n_w}, sampled {len(sampled)} words")

            for li in probe_layers:
                # Δg差异向量
                dg_pairs = []
                for w in sampled:
                    if li in word_gates.get(w, {}):
                        dg_pairs.append(word_gates[w][li] - layer_mean_g[li])

                # Δu差异向量
                du_pairs = []
                for w in sampled:
                    if li in word_ups.get(w, {}):
                        du_pairs.append(word_ups[w][li] - layer_mean_u[li])

                # W_down@Δu差异向量
                wdu_pairs = []
                if li in layer_mean_u:
                    lw = get_layer_weights(layers_list[li], d_model, mlp_type)
                    if lw.W_gate is not None:
                        for w in sampled:
                            if li in word_ups.get(w, {}):
                                du = word_ups[w][li] - layer_mean_u[li]
                                wdu_pairs.append(lw.W_down @ du)

                n90_dg, n50_dg, n95_dg, top5v_dg, ns_dg = compute_n90(dg_pairs, max_k=100) if len(dg_pairs) >= 10 else (-1,-1,-1,-1,-1)
                n90_du, n50_du, n95_du, top5v_du, ns_du = compute_n90(du_pairs, max_k=100) if len(du_pairs) >= 10 else (-1,-1,-1,-1,-1)
                n90_wdu, n50_wdu, n95_wdu, top5v_wdu, ns_wdu = compute_n90(wdu_pairs, max_k=100) if len(wdu_pairs) >= 10 else (-1,-1,-1,-1,-1)

                r = {
                    "n_words": n_w,
                    "layer": li,
                    "n90_dg": n90_dg if n90_dg != -1 else -1,
                    "n50_dg": n50_dg if n50_dg != -1 else -1,
                    "n95_dg": n95_dg if n95_dg != -1 else -1,
                    "n90_du": n90_du if n90_du != -1 else -1,
                    "n50_du": n50_du if n50_du != -1 else -1,
                    "n95_du": n95_du if n95_du != -1 else -1,
                    "n90_wdu": n90_wdu if n90_wdu != -1 else -1,
                    "n50_wdu": n50_wdu if n50_wdu != -1 else -1,
                    "n95_wdu": n95_wdu if n95_wdu != -1 else -1,
                }
                exp1_results.append(r)
                log(f"    L{li}: n90_dg={n90_dg} n90_du={n90_du} n90_wdu={n90_wdu}")

        out_dir = f"d:\\Ai2050\\TransformerLens-Project\\results\\causal_fiber\\{model_name}_cclxxx"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "exp1_n90_convergence_256.json"), 'w') as f:
            json.dump({"model": model_name, "sample_sizes": sample_sizes,
                        "probe_layers": probe_layers, "results": exp1_results}, f, indent=2)
        log(f"Exp1 saved ({len(exp1_results)} entries)")

        # ================================================================
        # Exp2: 偏移空间维度分析
        # ================================================================
        log("=== Exp2: Offset space dimensionality ===")

        exp2_results = []
        for li in range(n_layers):
            # 使用辅助函数逐层计算
            wdu_all, wnames = get_wdu_for_layer(li)
            if len(wdu_all) < 16: continue

            X_wdu = np.array(wdu_all, dtype=np.float32)
            mean_wdu = X_wdu.mean(axis=0)

            # δ = W_down@ū - mean(W_down@ū) — 偏移向量
            delta_vecs = [wdu - mean_wdu for wdu in wdu_all]

            # n50/n90/n95 of δ
            n90_d, n50_d, n95_d, top5v_d, ns_d = compute_n90(delta_vecs, max_k=100)
            if n90_d == -1: continue

            # 原始W_down@ū的n50/n90/n95
            n90_abs, n50_abs, n95_abs, top5v_abs, ns_abs = compute_n90(wdu_all, max_k=100)

            # Δu差异向量的n50/n90/n95
            du_vecs, _ = get_wd_du_for_layer(li)
            if len(du_vecs) >= 16:
                n90_du, n50_du, n95_du, top5v_du, ns_du = compute_n90(du_vecs, max_k=100)
            else:
                n90_du = n50_du = n95_du = top5v_du = ns_du = -1

            # Δg⊙ū差异向量的n50/n90/n95
            dgu_vecs, _ = get_dg_u_for_layer(li)
            if len(dgu_vecs) >= 16:
                n90_dgu, n50_dgu, n95_dgu, top5v_dgu, ns_dgu = compute_n90(dgu_vecs, max_k=100)
            else:
                n90_dgu = n50_dgu = n95_dgu = top5v_dgu = ns_dgu = -1

            r = {
                "layer": li,
                "n_samp": ns_d,
                # 偏移δ的维度
                "delta_n50": n50_d, "delta_n90": n90_d, "delta_n95": n95_d,
                "delta_top5_var": round(top5v_d, 4) if top5v_d >= 0 else -1,
                # 绝对W_down@ū的维度
                "abs_n50": n50_abs, "abs_n90": n90_abs, "abs_n95": n95_abs,
                "abs_top5_var": round(top5v_abs, 4) if top5v_abs >= 0 else -1,
                # Δu差异的维度
                "du_n50": n50_du, "du_n90": n90_du, "du_n95": n95_du,
                "du_top5_var": round(top5v_du, 4) if top5v_du >= 0 else -1,
                # Δg⊙ū差异的维度
                "dgu_n50": n50_dgu, "dgu_n90": n90_dgu, "dgu_n95": n95_dgu,
                "dgu_top5_var": round(top5v_dgu, 4) if top5v_dgu >= 0 else -1,
            }
            exp2_results.append(r)
            if li % 6 == 0 or li == n_layers - 1:
                log(f"  L{li}: delta_n90={n90_d} abs_n90={n90_abs} du_n90={n90_du} dgu_n90={n90_dgu}")

        with open(os.path.join(out_dir, "exp2_offset_dimensionality.json"), 'w') as f:
            json.dump({"model": model_name, "results": exp2_results}, f, indent=2)
        log(f"Exp2 saved ({len(exp2_results)} layers)")

        # ================================================================
        # Exp3: 32类Leave-One-Category-Out probe
        # ================================================================
        log("=== Exp3: 32-category probe ===")

        from scipy.sparse.linalg import svds

        exp3_results = []
        for li in probe_layers:
            # 使用W_down@Δu作为特征
            wd_du_vecs, _ = get_wd_du_for_layer(li)
            if len(wd_du_vecs) < 16: continue

            X = np.array(wd_du_vecs, dtype=np.float32)
            X_c = X - X.mean(axis=0)
            n_comp = min(50, X.shape[0]-1, X.shape[1]-1)
            try:
                _, s, Vt = svds(X_c, k=n_comp)
                Vt = Vt[np.argsort(-s)]
                s = s[np.argsort(-s)]
                X_pca = X_c @ Vt.T
            except Exception:
                continue

            cat_labels = np.array([cat_names.index(c) for c in all_cats])

            # 32类probe
            accs_euc_32 = []
            accs_cos_32 = []
            for test_cat_idx in range(n_cats):
                test_mask = cat_labels == test_cat_idx
                train_mask = ~test_mask
                if test_mask.sum() < 1 or train_mask.sum() < 16:
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
                    # 欧氏
                    dists = [np.linalg.norm(v - c) for c in centroids]
                    pred = np.argmin(dists)
                    accs_euc_32.append(int(pred == cat_labels[ti]))
                    # 余弦
                    sims = [proper_cos(v, c) for c in centroids]
                    pred_c = np.argmax(sims)
                    accs_cos_32.append(int(pred_c == cat_labels[ti]))

            # 16类probe (只用前16类)
            first16_cats = cat_names[:16]
            mask16 = np.array([c in first16_cats for c in all_cats])
            accs_euc_16 = []
            accs_cos_16 = []
            if mask16.sum() >= 32:
                X_pca_16 = X_pca[mask16]
                cat_labels_16 = np.array([first16_cats.index(c) for c in np.array(all_cats)[mask16]])
                for test_cat_idx in range(16):
                    test_mask = cat_labels_16 == test_cat_idx
                    train_mask = ~test_mask
                    if test_mask.sum() < 1 or train_mask.sum() < 16:
                        continue
                    centroids = []
                    for ci in range(16):
                        ci_mask = cat_labels_16 == ci
                        ci_train = ci_mask & train_mask
                        if ci_train.sum() > 0:
                            centroids.append(X_pca_16[ci_train].mean(axis=0))
                        else:
                            centroids.append(np.zeros(n_comp))
                    centroids = np.array(centroids)
                    for ti in np.where(test_mask)[0]:
                        v = X_pca_16[ti]
                        dists = [np.linalg.norm(v - c) for c in centroids]
                        pred = np.argmin(dists)
                        accs_euc_16.append(int(pred == cat_labels_16[ti]))
                        sims = [proper_cos(v, c) for c in centroids]
                        pred_c = np.argmax(sims)
                        accs_cos_16.append(int(pred_c == cat_labels_16[ti]))

            euc32 = float(np.mean(accs_euc_32)) if accs_euc_32 else 0.0
            cos32 = float(np.mean(accs_cos_32)) if accs_cos_32 else 0.0
            euc16 = float(np.mean(accs_euc_16)) if accs_euc_16 else 0.0
            cos16 = float(np.mean(accs_cos_16)) if accs_cos_16 else 0.0

            r = {
                "layer": li,
                "n_cats": n_cats,
                "n_words": len(all_words),
                "euc_acc_32cat": round(euc32, 4),
                "cos_acc_32cat": round(cos32, 4),
                "euc_acc_16cat": round(euc16, 4),
                "cos_acc_16cat": round(cos16, 4),
                "random_32cat": round(1.0/32, 4),
                "random_16cat": round(1.0/16, 4),
            }
            exp3_results.append(r)
            log(f"  L{li}: euc32={euc32:.3f} cos32={cos32:.3f} euc16={euc16:.3f} cos16={cos16:.3f}")

        with open(os.path.join(out_dir, "exp3_32cat_probe.json"), 'w') as f:
            json.dump({"model": model_name, "results": exp3_results}, f, indent=2)
        log(f"Exp3 saved ({len(exp3_results)} layers)")

        # ================================================================
        # Exp4: 类内/类间方差分析
        # ================================================================
        log("=== Exp4: Within/Between category variance ===")

        exp4_results = []
        for li in range(n_layers):
            # 使用辅助函数获取该层的W_down@ū
            wdu_all_li, wnames_li = get_wdu_for_layer(li)
            if len(wdu_all_li) < 16: continue

            # 按类别分组
            wdu_vecs = {}
            for v, w in zip(wdu_all_li, wnames_li):
                # 找到该词的类别
                idx = all_words.index(w)
                c = all_cats[idx]
                if c not in wdu_vecs:
                    wdu_vecs[c] = []
                wdu_vecs[c].append(v)

            if len(wdu_vecs) < 4: continue

            # 计算within-category variance (平均)
            within_vars = []
            for cat, vecs in wdu_vecs.items():
                if len(vecs) < 2: continue
                arr = np.array(vecs, dtype=np.float32)
                cat_mean = arr.mean(axis=0)
                var = np.mean(np.sum((arr - cat_mean)**2, axis=1))
                within_vars.append(var)

            # 计算between-category variance
            all_vecs = []
            for vecs in wdu_vecs.values():
                all_vecs.extend(vecs)
            grand_mean = np.mean(all_vecs, axis=0)
            between_vars = []
            for cat, vecs in wdu_vecs.items():
                if len(vecs) < 2: continue
                arr = np.array(vecs, dtype=np.float32)
                cat_mean = arr.mean(axis=0)
                var = np.sum((cat_mean - grand_mean)**2)
                between_vars.append(var)

            if not within_vars or not between_vars:
                continue

            mean_within = float(np.mean(within_vars))
            mean_between = float(np.mean(between_vars))
            f_ratio = mean_between / max(mean_within, 1e-10)

            r = {
                "layer": li,
                "n_categories": len(wdu_vecs),
                "mean_within_var": round(mean_within, 4),
                "mean_between_var": round(mean_between, 4),
                "f_ratio": round(f_ratio, 4),
            }
            exp4_results.append(r)
            if li % 6 == 0 or li == n_layers - 1:
                log(f"  L{li}: within={mean_within:.2f} between={mean_between:.2f} f_ratio={f_ratio:.2f}")

        with open(os.path.join(out_dir, "exp4_within_between_var.json"), 'w') as f:
            json.dump({"model": model_name, "results": exp4_results}, f, indent=2)
        log(f"Exp4 saved ({len(exp4_results)} layers)")

        # 释放大数组
        del word_gates, word_ups, word_residuals
        del layer_mean_u, layer_mean_g
        gc.collect()
        release_model(model)
        log(f"Model {model_name} released")

    except Exception as e:
        log(f"ERROR in {model_name}: {e}")
        traceback.print_exc(file=open(LOG, 'a', encoding='utf-8'))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    log("=== CCLXXX All done ===")
