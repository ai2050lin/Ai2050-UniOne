"""
CCVIII(308): Attention vs MLP语义分解 + 跨模型语义轴对齐
=========================================================
用户关键审视:
1. 当前只分析MLP(Δg, Δu), 忽略了Attention(Δa)
2. 完整层更新: h_{l+1} = h_l + Attn(h_l) + MLP(h_l)
3. 核心问题: 语义断裂首先出现在Attention还是MLP?

实验设计:
  Exp1: 收集每层的Attention输出Δa和MLP输出Δm
    - 对4类别词, 收集attn_output和mlp_output
    - 计算Δa和Δm的语义方向(类别质心差)
    - 比较: 语义信息首先在哪个模块出现?

  Exp2: Δa和Δm的W_U对齐度
    - 语义方向分别在attn和mlp输出中
    - 哪个模块的语义方向更对齐W_U大奇异值?

  Exp3: 精细因果控制 — 单一SVD模式perturb
    - 不swap整个Δa/Δm, 而是沿单一SVD模式perturb
    - 连续调节: 找到最小控制变量集

  Exp4: 跨模型语义轴对齐
    - 在不同模型的d_model空间, 计算语义方向
    - 测量跨模型语义方向的余弦相似度
    - 如果语义轴跨模型一致, 说明是普适数学结构

用法:
  python ccviii_attn_mlp_decompose.py --model qwen3
  python ccviii_attn_mlp_decompose.py --model glm4
  python ccviii_attn_mlp_decompose.py --model deepseek7b
"""
import argparse, os, sys, time, gc, json
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from collections import defaultdict
from scipy import stats
from scipy.sparse.linalg import svds

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, get_W_U,
    get_layer_weights, MODEL_CONFIGS,
)

TEMP_DIR = Path("tests/glm5_temp")
LOG_FILE = TEMP_DIR / "ccviii_attn_mlp_decompose_log.txt"

CONCEPTS = {
    "animals": ["dog", "cat", "horse", "eagle", "shark", "lion", "bear", "fish", "snake", "whale",
                "rabbit", "deer", "fox", "wolf", "tiger", "monkey", "elephant", "dolphin", "parrot", "duck"],
    "food":    ["apple", "rice", "bread", "cheese", "pizza", "banana", "mango", "pasta", "salad", "steak",
                "soup", "cake", "cookie", "grape", "lemon", "peach", "corn", "bean", "pepper", "onion"],
    "tools":   ["hammer", "knife", "saw", "drill", "wrench", "screw", "pliers", "chisel", "level", "ruler",
                "shovel", "axe", "clamp", "welder", "plane", "anvil", "lathe", "forge", "drill", "mallet"],
    "nature":  ["mountain", "river", "ocean", "forest", "desert", "valley", "canyon", "island", "meadow", "glacier",
                "volcano", "waterfall", "swamp", "tundra", "prairie", "cliff", "reef", "lagoon", "cave", "ridge"],
}

TEMPLATE = "The {} is"
N_WORDS_PER_CAT = 15  # 每类别词数
N_RANDOM_DIRS = 5


def log_f(msg="", end="\n"):
    print(msg, end=end, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + end)


def run_experiment(model_name):
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    log_f(f"\n{'#'*70}")
    log_f(f"CCVIII(308): Attention vs MLP Semantic Decomposition")
    log_f(f"Model: {model_name}")
    log_f(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_f(f"{'#'*70}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers_list = get_layers(model)

    # ===== Step 1: Collect Attention + MLP outputs per layer =====
    log_f("\n--- Step 1: Collecting Attn/MLP outputs ---")

    rng = np.random.RandomState(42)
    all_words = []
    all_cats = []
    for cat, words in CONCEPTS.items():
        sel = rng.choice(words, min(N_WORDS_PER_CAT, len(words)), replace=False)
        all_words.extend(sel.tolist() if hasattr(sel, 'tolist') else list(sel))
        all_cats.extend([cat] * min(N_WORDS_PER_CAT, len(words)))

    categories = list(CONCEPTS.keys())
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    # 采样层
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    log_f(f"  Sample layers: {sample_layers}")

    # 收集各词在各层的attn_output和mlp_output
    # shape: {li: {word: {"attn": [d_model], "mlp": [d_model], "residual": [d_model]}}}
    word_outputs = {li: {} for li in sample_layers}

    for wi, word in enumerate(all_words):
        text = TEMPLATE.format(word)
        input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
        last_pos = input_ids.shape[1] - 1

        # 用hook收集attn_output和mlp_output
        captured = {}

        def make_hook(key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured[key] = output[0][0, last_pos].detach().float().cpu().numpy()
                else:
                    captured[key] = output[0, last_pos].detach().float().cpu().numpy() if output.ndim > 1 else output.detach().float().cpu().numpy()
            return hook

        hooks = []
        for li in sample_layers:
            layer = layers_list[li]
            # Hook attention output
            hooks.append(layer.self_attn.register_forward_hook(make_hook(f"attn_L{li}")))
            # Hook MLP output
            hooks.append(layer.mlp.register_forward_hook(make_hook(f"mlp_L{li}")))
            # Hook residual stream (layer output)
            hooks.append(layer.register_forward_hook(make_hook(f"resid_L{li}")))

        with torch.no_grad():
            try:
                _ = model(input_ids)
            except Exception as e:
                log_f(f"  Forward failed for '{word}': {e}")
                for h in hooks:
                    h.remove()
                continue

        for h in hooks:
            h.remove()

        for li in sample_layers:
            attn_key = f"attn_L{li}"
            mlp_key = f"mlp_L{li}"
            resid_key = f"resid_L{li}"
            if attn_key in captured and mlp_key in captured:
                resid_data = captured.get(resid_key, None)
                word_outputs[li][word] = {
                    "attn": captured[attn_key],
                    "mlp": captured[mlp_key],
                    "resid": resid_data,
                }

        if (wi + 1) % 5 == 0:
            log_f(f"  Processed {wi+1}/{len(all_words)} words")

    log_f(f"  Collected outputs for {sum(1 for li in sample_layers for w in word_outputs[li])} word-layer pairs")

    # ===== Step 2: Compute semantic directions in Attn/MLP/Residual space =====
    log_f(f"\n{'='*70}")
    log_f(f"Step 2: Semantic Direction Decomposition (Attn vs MLP)")
    log_f(f"{'='*70}")

    # 类别质心
    cat_centroids = {li: {"attn": {}, "mlp": {}, "resid": {}} for li in sample_layers}

    for li in sample_layers:
        for space in ["attn", "mlp", "resid"]:
            for cat in categories:
                cat_words_list = [w for w, c in zip(all_words, all_cats) if c == cat and w in word_outputs[li]]
                if cat_words_list:
                    valid_vecs = [word_outputs[li][w][space] for w in cat_words_list if word_outputs[li][w].get(space) is not None]
                    if valid_vecs:
                        vecs = np.array(valid_vecs)
                        cat_centroids[li][space][cat] = vecs.mean(axis=0)

    # 语义方向: 类别质心差
    cat_pairs = []
    for i in range(len(categories)):
        for j in range(i+1, len(categories)):
            cat_pairs.append((categories[i], categories[j]))

    results = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "sample_layers": sample_layers,
        "decomposition": {},  # per-layer decomposition results
        "cross_model_alignment": {},  # will be filled after all models run
    }

    for li in sample_layers:
        log_f(f"\n  === Layer {li} ===")
        li_data = {"attn_semantic": {}, "mlp_semantic": {}, "resid_semantic": {}}

        for cat_a, cat_b in cat_pairs:
            pair_name = f"{cat_a}->{cat_b}"

            for space in ["attn", "mlp", "resid"]:
                if cat_a not in cat_centroids[li][space] or cat_b not in cat_centroids[li][space]:
                    continue

                centroid_a = cat_centroids[li][space][cat_a]
                centroid_b = cat_centroids[li][space][cat_b]
                direction = centroid_b - centroid_a
                norm = np.linalg.norm(direction)

                if norm < 1e-10:
                    continue

                direction_normalized = direction / norm

                # ANOVA: 各维度对类别的区分力
                cat_a_words = [w for w, c in zip(all_words, all_cats) if c == cat_a and w in word_outputs[li]]
                cat_b_words = [w for w, c in zip(all_words, all_cats) if c == cat_b and w in word_outputs[li]]

                if len(cat_a_words) < 2 or len(cat_b_words) < 2:
                    continue

                vecs_a = np.array([word_outputs[li][w][space] for w in cat_a_words])
                vecs_b = np.array([word_outputs[li][w][space] for w in cat_b_words])

                # Welch t-test per dimension (快速ANOVA for 2 groups)
                n_sig = 0
                max_t = 0
                for dim in range(min(d_model, vecs_a.shape[1])):
                    t_stat, p_val = stats.ttest_ind(vecs_a[:, dim], vecs_b[:, dim], equal_var=False)
                    if not np.isnan(p_val) and p_val < 0.05:
                        n_sig += 1
                    if not np.isnan(abs(t_stat)):
                        max_t = max(max_t, abs(t_stat))

                sig_ratio = n_sig / min(d_model, vecs_a.shape[1]) if vecs_a.shape[1] > 0 else 0

                # Category classification accuracy (质心差方向投影)
                all_vecs = np.vstack([vecs_a, vecs_b])
                all_labels = np.array([0]*len(cat_a_words) + [1]*len(cat_b_words))
                projections = all_vecs @ direction_normalized
                threshold = (projections[:len(cat_a_words)].mean() + projections[len(cat_a_words):].mean()) / 2
                pred_labels = (projections > threshold).astype(int)
                accuracy = np.mean(pred_labels == all_labels)

                key = space + "_semantic"
                li_data[key][pair_name] = {
                    "direction_norm": float(norm),
                    "sig_ratio": float(sig_ratio),
                    "max_t": float(max_t),
                    "accuracy": float(accuracy),
                }

        # 汇总: Attention vs MLP 语义能力对比
        attn_norms = [li_data["attn_semantic"][p]["direction_norm"] for p in li_data["attn_semantic"]]
        mlp_norms = [li_data["mlp_semantic"][p]["direction_norm"] for p in li_data["mlp_semantic"]]
        resid_norms = [li_data["resid_semantic"][p]["direction_norm"] for p in li_data["resid_semantic"]]

        attn_sig = [li_data["attn_semantic"][p]["sig_ratio"] for p in li_data["attn_semantic"]]
        mlp_sig = [li_data["mlp_semantic"][p]["sig_ratio"] for p in li_data["mlp_semantic"]]
        resid_sig = [li_data["resid_semantic"][p]["sig_ratio"] for p in li_data["resid_semantic"]]

        attn_acc = [li_data["attn_semantic"][p]["accuracy"] for p in li_data["attn_semantic"]]
        mlp_acc = [li_data["mlp_semantic"][p]["accuracy"] for p in li_data["mlp_semantic"]]
        resid_acc = [li_data["resid_semantic"][p]["accuracy"] for p in li_data["resid_semantic"]]

        summary = {
            "attn_avg_norm": float(np.mean(attn_norms)) if attn_norms else 0,
            "mlp_avg_norm": float(np.mean(mlp_norms)) if mlp_norms else 0,
            "resid_avg_norm": float(np.mean(resid_norms)) if resid_norms else 0,
            "attn_avg_sig": float(np.mean(attn_sig)) if attn_sig else 0,
            "mlp_avg_sig": float(np.mean(mlp_sig)) if mlp_sig else 0,
            "resid_avg_sig": float(np.mean(resid_sig)) if resid_sig else 0,
            "attn_avg_acc": float(np.mean(attn_acc)) if attn_acc else 0,
            "mlp_avg_acc": float(np.mean(mlp_acc)) if mlp_acc else 0,
            "resid_avg_acc": float(np.mean(resid_acc)) if resid_acc else 0,
            "attn_mlp_norm_ratio": float(np.mean(attn_norms) / np.mean(mlp_norms)) if mlp_norms and np.mean(mlp_norms) > 0 else 0,
            "attn_mlp_sig_ratio": float(np.mean(attn_sig) / np.mean(mlp_sig)) if mlp_sig and np.mean(mlp_sig) > 0 else 0,
            "attn_mlp_acc_ratio": float(np.mean(attn_acc) / np.mean(mlp_acc)) if mlp_acc and np.mean(mlp_acc) > 0 else 0,
        }
        li_data["_summary"] = summary

        log_f(f"  Attn: norm={summary['attn_avg_norm']:.3f}, sig={summary['attn_avg_sig']:.3f}, acc={summary['attn_avg_acc']:.3f}")
        log_f(f"  MLP:  norm={summary['mlp_avg_norm']:.3f}, sig={summary['mlp_avg_sig']:.3f}, acc={summary['mlp_avg_acc']:.3f}")
        log_f(f"  Resid: norm={summary['resid_avg_norm']:.3f}, sig={summary['resid_avg_sig']:.3f}, acc={summary['resid_avg_acc']:.3f}")
        log_f(f"  Attn/MLP ratio: norm={summary['attn_mlp_norm_ratio']:.2f}, sig={summary['attn_mlp_sig_ratio']:.2f}, acc={summary['attn_mlp_acc_ratio']:.2f}")

        results["decomposition"][str(li)] = li_data

    # ===== Step 3: W_U alignment for Attn vs MLP semantic directions =====
    log_f(f"\n{'='*70}")
    log_f(f"Step 3: W_U Alignment (Attn vs MLP)")
    log_f(f"{'='*70}")

    W_U = get_W_U(model)  # [vocab_size, d_model]

    # W_U SVD
    log_f("  Computing W_U SVD...")
    W_U_T = W_U.T.astype(np.float32)  # [d_model, vocab_size]
    k_svd = min(500, min(W_U_T.shape) - 2)
    U_wu, s_wu, Vt_wu = svds(W_U_T, k=k_svd)
    sort_idx = np.argsort(-s_wu)
    U_wu = U_wu[:, sort_idx]  # [d_model, k], sorted by singular value
    s_wu = s_wu[sort_idx]

    log_f(f"  W_U top-5 singular values: {s_wu[:5]}")

    wu_alignment = {}

    for li in sample_layers:
        li_align = {"attn": {}, "mlp": {}, "resid": {}}

        for space in ["attn", "mlp", "resid"]:
            sem_dirs = []
            for pair_name in cat_pairs:
                p_name = f"{pair_name[0]}->{pair_name[1]}"
                if p_name in results["decomposition"][str(li)][space + "_semantic"]:
                    cat_a, cat_b = pair_name
                    if cat_a in cat_centroids[li][space] and cat_b in cat_centroids[li][space]:
                        direction = cat_centroids[li][space][cat_b] - cat_centroids[li][space][cat_a]
                        norm = np.linalg.norm(direction)
                        if norm > 1e-10:
                            sem_dirs.append((p_name, direction / norm))

            # 随机方向
            rng_temp = np.random.RandomState(li * 100)
            rnd_dirs = [rng_temp.randn(d_model) for _ in range(N_RANDOM_DIRS)]
            rnd_dirs = [d / np.linalg.norm(d) for d in rnd_dirs]

            # 计算W_U投影
            sem_wu_gains = []
            sem_top10_energy = []
            rnd_wu_gains = []
            rnd_top10_energy = []

            for p_name, d in sem_dirs:
                proj = U_wu.T @ d  # [k] - coefficients on W_U SVD modes
                # W_U gain: ||W_U * d|| / ||d||
                wu_gain = np.sqrt(np.sum((s_wu * proj) ** 2))
                sem_wu_gains.append(float(wu_gain))

                # Top-10 energy ratio
                energy = proj ** 2
                total_e = np.sum(energy)
                top10_e = np.sum(np.sort(energy)[-10:]) if len(energy) >= 10 else np.sum(energy)
                sem_top10_energy.append(float(top10_e / total_e) if total_e > 0 else 0)

                li_align[space][p_name] = {
                    "wu_gain": float(wu_gain),
                    "top10_energy_ratio": float(top10_e / total_e) if total_e > 0 else 0,
                    "weighted_mode_idx": float(np.average(np.arange(len(energy)), weights=energy + 1e-20)),
                }

            for d in rnd_dirs:
                proj = U_wu.T @ d
                wu_gain = np.sqrt(np.sum((s_wu * proj) ** 2))
                rnd_wu_gains.append(float(wu_gain))

                energy = proj ** 2
                total_e = np.sum(energy)
                top10_e = np.sum(np.sort(energy)[-10:]) if len(energy) >= 10 else np.sum(energy)
                rnd_top10_energy.append(float(top10_e / total_e) if total_e > 0 else 0)

            li_align[space]["_summary"] = {
                "sem_avg_wu_gain": float(np.mean(sem_wu_gains)) if sem_wu_gains else 0,
                "rnd_avg_wu_gain": float(np.mean(rnd_wu_gains)) if rnd_wu_gains else 0,
                "sem_avg_top10": float(np.mean(sem_top10_energy)) if sem_top10_energy else 0,
                "rnd_avg_top10": float(np.mean(rnd_top10_energy)) if rnd_top10_energy else 0,
                "sem_rnd_gain_ratio": float(np.mean(sem_wu_gains) / np.mean(rnd_wu_gains)) if rnd_wu_gains and np.mean(rnd_wu_gains) > 0 else 0,
            }

        # 对比
        attn_s = li_align["attn"]["_summary"]
        mlp_s = li_align["mlp"]["_summary"]
        resid_s = li_align["resid"]["_summary"]

        log_f(f"  L{li}:")
        log_f(f"    Attn:  WU_gain={attn_s['sem_avg_wu_gain']:.2f} (RND={attn_s['rnd_avg_wu_gain']:.2f}), "
              f"ratio={attn_s['sem_rnd_gain_ratio']:.2f}, top10={attn_s['sem_avg_top10']:.3f}")
        log_f(f"    MLP:   WU_gain={mlp_s['sem_avg_wu_gain']:.2f} (RND={mlp_s['rnd_avg_wu_gain']:.2f}), "
              f"ratio={mlp_s['sem_rnd_gain_ratio']:.2f}, top10={mlp_s['sem_avg_top10']:.3f}")
        log_f(f"    Resid: WU_gain={resid_s['sem_avg_wu_gain']:.2f} (RND={resid_s['rnd_avg_wu_gain']:.2f}), "
              f"ratio={resid_s['sem_rnd_gain_ratio']:.2f}, top10={resid_s['sem_avg_top10']:.3f}")

        wu_alignment[str(li)] = li_align

    results["wu_alignment"] = wu_alignment

    # ===== Step 4: Fine-grained causal control — SVD mode perturb =====
    log_f(f"\n{'='*70}")
    log_f(f"Step 4: SVD Mode Perturb (Fine-grained Causal Control)")
    log_f(f"{'='*70}")

    # 选择后半层做精细perturb
    target_layers = [li for li in sample_layers if li >= n_layers // 2]
    if not target_layers:
        target_layers = [sample_layers[-1]]

    svd_perturb_results = {}

    # 只用3个类别对, 减少计算量
    test_pairs = [("animals", "tools"), ("food", "nature")]

    for li in target_layers[-3:]:  # 只测最后3个采样层
        log_f(f"\n  === L{li} SVD Mode Perturb ===")
        li_perturb = {}

        for space in ["attn", "mlp"]:
            # 收集所有词的attn/mlp输出
            all_vecs = []
            all_labels_cat = []
            for cat in categories:
                cat_words_list = [w for w, c in zip(all_words, all_cats) if c == cat and w in word_outputs[li]]
                for w in cat_words_list:
                    all_vecs.append(word_outputs[li][w][space])
                    all_labels_cat.append(cat)

            if len(all_vecs) < 4:
                continue

            all_vecs = np.array(all_vecs)  # [n_words, d_model]

            # SVD of the output matrix
            log_f(f"    Computing SVD of {space} output matrix ({all_vecs.shape})...")
            try:
                # 中心化
                mean_vec = all_vecs.mean(axis=0)
                centered = all_vecs - mean_vec

                # SVD
                k_svd_local = min(50, min(centered.shape) - 1)
                U_local, s_local, Vt_local = svds(centered, k=k_svd_local)
                sort_idx = np.argsort(-s_local)
                U_local = U_local[:, sort_idx]
                s_local = s_local[sort_idx]
                Vt_local = Vt_local[sort_idx]

                log_f(f"    Top-5 singular values: {s_local[:5]}")

                # 每个SVD模式上各类别的区分力
                mode_discrimination = []
                for mi in range(min(20, k_svd_local)):
                    mode_coeffs = centered @ Vt_local[mi]  # [n_words]
                    # ANOVA F-statistic for 4 categories
                    cat_groups = defaultdict(list)
                    for coeff, cat in zip(mode_coeffs, all_labels_cat):
                        cat_groups[cat].append(coeff)

                    group_data = [cat_groups[c] for c in categories if c in cat_groups]
                    if len(group_data) >= 2:
                        f_stat, p_val = stats.f_oneway(*group_data)
                        mode_discrimination.append((mi, float(f_stat) if not np.isnan(f_stat) else 0,
                                                    float(p_val) if not np.isnan(p_val) else 1.0))

                # 排序: 最具语义区分力的SVD模式
                mode_discrimination.sort(key=lambda x: -x[1])
                top_discriminative = mode_discrimination[:5]

                log_f(f"    Top-5 discriminative SVD modes:")
                for mi, f, p in top_discriminative:
                    log_f(f"      Mode {mi}: F={f:.2f}, p={p:.4f}")

                li_perturb[space] = {
                    "top_singular_values": s_local[:10].tolist(),
                    "mode_discrimination": [(mi, f, p) for mi, f, p in mode_discrimination[:20]],
                    "top_discriminative_modes": [(mi, f, p) for mi, f, p in top_discriminative],
                }

            except Exception as e:
                log_f(f"    SVD failed for {space}: {e}")
                li_perturb[space] = {"error": str(e)}

        svd_perturb_results[str(li)] = li_perturb

    results["svd_perturb"] = svd_perturb_results

    # ===== Step 5: Cross-model semantic axis alignment =====
    log_f(f"\n{'='*70}")
    log_f(f"Step 5: Store semantic directions for cross-model alignment")
    log_f(f"{'='*70}")

    # 保存每个模型在每个层的语义方向(归一化)
    semantic_directions = {}
    for li in sample_layers:
        li_dirs = {}
        for space in ["attn", "mlp", "resid"]:
            space_dirs = {}
            for cat_a, cat_b in cat_pairs:
                p_name = f"{cat_a}->{cat_b}"
                if cat_a in cat_centroids[li][space] and cat_b in cat_centroids[li][space]:
                    direction = cat_centroids[li][space][cat_b] - cat_centroids[li][space][cat_a]
                    norm = np.linalg.norm(direction)
                    if norm > 1e-10:
                        space_dirs[p_name] = (direction / norm).tolist()
            li_dirs[space] = space_dirs
        semantic_directions[str(li)] = li_dirs

    results["semantic_directions"] = semantic_directions

    # ===== Save results =====
    elapsed = time.time() - start_time
    results["elapsed_seconds"] = elapsed

    out_path = TEMP_DIR / f"ccviii_attn_mlp_decompose_{model_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    log_f(f"\n  Results saved to {out_path}")
    log_f(f"  Total time: {elapsed:.1f}s")

    # ===== Summary =====
    log_f(f"\n{'='*70}")
    log_f(f"SUMMARY: Attention vs MLP Semantic Decomposition")
    log_f(f"{'='*70}")

    for li in sample_layers:
        li_str = str(li)
        if li_str in results["decomposition"] and "_summary" in results["decomposition"][li_str]:
            s = results["decomposition"][li_str]["_summary"]
            log_f(f"  L{li}: Attn/MLP norm_ratio={s['attn_mlp_norm_ratio']:.2f}, "
                  f"sig_ratio={s['attn_mlp_sig_ratio']:.2f}, acc_ratio={s['attn_mlp_acc_ratio']:.2f}")

    log_f(f"\n  ★★★ KEY QUESTION: Where does semantic fracture first appear? ★★★")

    # 找到第一个Attn sig > 0.2的层
    first_attn_layer = None
    first_mlp_layer = None
    for li in sample_layers:
        li_str = str(li)
        if li_str in results["decomposition"] and "_summary" in results["decomposition"][li_str]:
            s = results["decomposition"][li_str]["_summary"]
            if s["attn_avg_sig"] > 0.2 and first_attn_layer is None:
                first_attn_layer = li
            if s["mlp_avg_sig"] > 0.2 and first_mlp_layer is None:
                first_mlp_layer = li

    if first_attn_layer is not None and first_mlp_layer is not None:
        if first_attn_layer <= first_mlp_layer:
            log_f(f"  → Attention first shows semantic structure at L{first_attn_layer}")
            log_f(f"  → MLP first shows semantic structure at L{first_mlp_layer}")
            log_f(f"  → ★★★ Attention is the TRIGGER, MLP is the RECODER ★★★")
        else:
            log_f(f"  → MLP first shows semantic structure at L{first_mlp_layer}")
            log_f(f"  → Attention first shows semantic structure at L{first_attn_layer}")
            log_f(f"  → ★★★ MLP encodes semantics BEFORE Attention ★★★")
    else:
        log_f(f"  → Could not determine order (attn_layer={first_attn_layer}, mlp_layer={first_mlp_layer})")

    release_model(model)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    run_experiment(args.model)
