"""
CCXI(311): 语义偏移方向分析
==========================
CCX发现: 深层nosem模式效果≈sem模式(72-97%), 语义特异性在哪?
关键洞察: 语义特异性不在"效果强度", 而在"效果方向"!

本实验:
  对每个SVD模式perturb后, 追踪:
  1. 输出logit偏向哪个类别? (语义偏移方向)
  2. sem模式是否让输出偏向特定类别? (方向性)
  3. nosem模式是否让输出偏向随机类别? (无方向性)

核心假设:
  - sem模式perturb → 输出沿语义力线方向偏移(可预测)
  - nosem模式perturb → 输出随机偏移(不可预测)
  - 如果成立 → 语义特异性的关键是偏移方向, 不是偏移幅度

指标:
  - direction_consistency: 同一SVD模式perturb不同词, 偏移方向是否一致?
  - category_preference: perturb后输出最常偏向哪个类别?
  - sem_vs_nosem_directionality: sem模式的direction_consistency是否>nosem?

用法:
  python ccxi_shift_direction.py --model qwen3
  python ccxi_shift_direction.py --model glm4
  python ccxi_shift_direction.py --model deepseek7b
"""
import argparse, os, sys, time, gc, json
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from collections import defaultdict, Counter
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
LOG_FILE = TEMP_DIR / "ccxi_shift_direction_log.txt"

CONCEPTS = {
    "animals": ["dog", "cat", "horse", "eagle", "shark", "lion", "bear", "fish", "snake", "whale"],
    "food":    ["apple", "rice", "bread", "cheese", "pizza", "banana", "mango", "pasta", "salad", "steak"],
    "tools":   ["hammer", "knife", "saw", "drill", "wrench", "screw", "pliers", "chisel", "level", "ruler"],
    "nature":  ["mountain", "river", "ocean", "forest", "desert", "valley", "canyon", "island", "meadow", "glacier"],
}

TEMPLATE = "The {} is"
N_WORDS_PER_CAT = 8


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def get_baseline_prediction(model, tokenizer, device, prompt):
    """获取baseline预测: top1 token id + 类别logits"""
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**toks)
    logits = out.logits[0, -1, :].float().cpu().numpy()
    top1_id = int(np.argmax(logits))
    return top1_id, logits


def get_perturbed_prediction(model, tokenizer, device, prompt, target_module, perturb_vec_np, last_pos):
    """获取perturbed预测"""
    intervention_done = [False]

    def make_hook(pv_np, done_flag, lp):
        def hook(module, input, output):
            if done_flag[0]:
                return output
            done_flag[0] = True
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            perturb_tensor = torch.tensor(pv_np, dtype=out.dtype, device=device)
            new_out = out.clone()
            new_out[0, lp, :] += perturb_tensor
            if isinstance(output, tuple):
                return (new_out,) + output[1:]
            return new_out
        return hook

    hook_handle = target_module.register_forward_hook(make_hook(perturb_vec_np, intervention_done, last_pos))
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**toks)
    logits = out.logits[0, -1, :].float().cpu().numpy()
    top1_id = int(np.argmax(logits))
    hook_handle.remove()
    return top1_id, logits


def get_category_from_token(tokenizer, token_id, word_to_cat):
    """从token id判断属于哪个类别"""
    word = tokenizer.decode([token_id]).strip().lower()
    return word_to_cat.get(word, None)


def build_word_to_cat():
    """构建word→category映射"""
    mapping = {}
    for cat, words in CONCEPTS.items():
        for w in words:
            mapping[w] = cat
    # 加入复数和常见关联词
    for cat, words in CONCEPTS.items():
        for w in words:
            mapping[w + "s"] = cat
            mapping[w + "es"] = cat
    return mapping


def run_experiment(model_name):
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model)
    word_to_cat = build_word_to_cat()

    # 构建类别token集合 — 用于计算类别logit
    cat_token_ids = {}
    for cat, words in CONCEPTS.items():
        ids = []
        for w in words:
            tok_ids = tokenizer.encode(" " + w, add_special_tokens=False)
            ids.extend(tok_ids)
        cat_token_ids[cat] = list(set(ids))

    n_layers = info.n_layers
    d_model = info.d_model

    # 选择测试层
    if n_layers <= 6:
        sample_layers = list(range(n_layers))
    else:
        # 浅层、中层、深层
        sample_layers = [
            0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1,
            n_layers // 8, n_layers - 2
        ]
        sample_layers = sorted(set(sample_layers))

    log(f"\n{'='*70}")
    log(f"CCXI(311): 语义偏移方向分析 - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}")
    log(f"  测试层: {sample_layers}")
    log(f"{'='*70}")

    # Step 1: 收集Attention/MLP输出
    log("\n--- Step 1: 收集各词在各层的Attn/MLP输出 ---")
    all_words = []
    all_cats = []
    for cat, words in CONCEPTS.items():
        for w in words[:N_WORDS_PER_CAT]:
            all_words.append(w)
            all_cats.append(cat)
    categories = list(CONCEPTS.keys())

    word_outputs = {li: {} for li in sample_layers}

    for wi, word in enumerate(all_words):
        prompt = TEMPLATE.format(word)
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        last_pos = toks.input_ids.shape[1] - 1

        captured = {}

        def make_capture_hook(key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured[key] = output[0][0, last_pos, :].detach().float().cpu().numpy()
                else:
                    captured[key] = output[0, last_pos, :].detach().float().cpu().numpy()
            return hook

        hooks = []
        for li in sample_layers:
            layer = layers_list[li]
            attn_key = f"attn_L{li}"
            mlp_key = f"mlp_L{li}"
            resid_key = f"resid_L{li}"
            hooks.append(layer.self_attn.register_forward_hook(make_capture_hook(attn_key)))
            hooks.append(layer.mlp.register_forward_hook(make_capture_hook(mlp_key)))
            hooks.append(layer.register_forward_hook(make_capture_hook(resid_key)))

        with torch.no_grad():
            _ = model(**toks)

        for h in hooks:
            h.remove()

        for li in sample_layers:
            attn_data = captured.get(f"attn_L{li}", None)
            mlp_data = captured.get(f"mlp_L{li}", None)
            resid_data = captured.get(f"resid_L{li}", None)
            word_outputs[li][word] = {
                "attn": attn_data,
                "mlp": mlp_data,
                "resid": resid_data,
            }

        if (wi + 1) % 10 == 0:
            log(f"  收集 {wi+1}/{len(all_words)} 词完成")

    # Step 2: 计算类别质心和SVD模式
    log("\n--- Step 2: 计算类别质心和SVD模式 ---")

    layer_svd = {}
    layer_centroids = {}
    layer_mode_labels = {}  # sem vs nosem

    for li in sample_layers:
        for space in ["attn", "mlp"]:
            # 收集该层该空间所有词的向量
            vecs = []
            valid_words = []
            for w in all_words:
                v = word_outputs[li][w].get(space)
                if v is not None:
                    vecs.append(v)
                    valid_words.append(w)

            if len(vecs) < 4:
                continue

            vecs = np.array(vecs)

            # 计算类别质心
            centroids = {}
            for cat in categories:
                cat_vecs = [vecs[i] for i, w in enumerate(valid_words) if all_cats[all_words.index(w)] == cat]
                if cat_vecs:
                    centroids[cat] = np.mean(cat_vecs, axis=0)

            layer_centroids[(li, space)] = centroids

            # SVD
            mean_vec = vecs.mean(axis=0)
            centered = vecs - mean_vec
            n_svd = min(30, centered.shape[0] - 1, centered.shape[1] - 1)
            if n_svd < 3:
                continue
            U, s, Vt = np.linalg.svd(centered, full_matrices=False)
            U = U[:, :n_svd]
            s = s[:n_svd]
            Vt = Vt[:n_svd, :]

            # 计算每个SVD模式的语义区分力(F值)
            mode_f_scores = []
            projections = centered @ Vt.T  # [n_words, n_svd]

            for mi in range(n_svd):
                proj_mi = projections[:, mi]
                groups = []
                for cat in categories:
                    cat_idx = [i for i, w in enumerate(valid_words) if all_cats[all_words.index(w)] == cat]
                    if cat_idx:
                        groups.append(proj_mi[cat_idx])
                if len(groups) >= 2:
                    f_stat, _ = stats.f_oneway(*groups)
                    mode_f_scores.append(f_stat if not np.isnan(f_stat) else 0)
                else:
                    mode_f_scores.append(0)

            # 标记semantic vs non-semantic模式
            f_arr = np.array(mode_f_scores)
            median_f = np.median(f_arr[f_arr > 0]) if np.any(f_arr > 0) else 0

            mode_labels = {}
            for mi in range(n_svd):
                if f_arr[mi] > median_f * 2:  # F值 > 2倍中位数 → 语义模式
                    mode_labels[mi] = "sem"
                elif f_arr[mi] < median_f * 0.3:  # F值 < 0.3倍中位数 → 非语义模式
                    mode_labels[mi] = "nosem"

            # 选择最有语义区分力的3个模式和最无区分力的3个模式
            sorted_modes = np.argsort(f_arr)[::-1]
            top3_sem = sorted_modes[:3].tolist()
            bottom3_nosem = sorted_modes[-3:].tolist()

            layer_svd[(li, space)] = {
                "Vt": Vt, "s": s, "mean": mean_vec,
                "f_scores": f_arr, "top3_sem": top3_sem, "bottom3_nosem": bottom3_nosem,
            }
            layer_mode_labels[(li, space)] = mode_labels

    # Step 3: 语义偏移方向分析
    log("\n--- Step 3: 语义偏移方向分析 ---")

    ALPHA = 1.0  # 扰动强度
    results = {}

    # 选择关键层进行详细分析
    key_layers = [sample_layers[-1], sample_layers[-2]] if len(sample_layers) >= 2 else [sample_layers[-1]]
    # 加一个中层
    mid_idx = len(sample_layers) // 2
    if mid_idx not in key_layers:
        key_layers.append(sample_layers[mid_idx])
    # 加一个浅层
    if sample_layers[0] not in key_layers:
        key_layers.append(sample_layers[0])

    for li in key_layers:
        for space in ["attn", "mlp"]:
            svd_data = layer_svd.get((li, space))
            if svd_data is None:
                continue

            Vt = svd_data["Vt"]
            centroids = layer_centroids.get((li, space), {})

            log(f"\n  === L{li} {space} ===")

            # 对每个SVD模式进行perturb
            for mode_type, mode_list in [("sem", svd_data["top3_sem"]), ("nosem", svd_data["bottom3_nosem"])]:
                for mi in mode_list:
                    if mi >= Vt.shape[0]:
                        continue
                    mode_dir = Vt[mi]  # [d_model], 已归一化

                    # 对每个词进行perturb, 记录偏移方向
                    shift_cats = []  # perturb后输出偏向的类别
                    shift_logit_changes = {cat: [] for cat in categories}

                    for wi, word in enumerate(all_words):
                        if word not in word_outputs[li]:
                            continue
                        v = word_outputs[li][word].get(space)
                        if v is None:
                            continue

                        output_norm = np.linalg.norm(v)
                        if output_norm < 1e-10:
                            continue

                        # 计算perturbation向量
                        perturb_vec = ALPHA * output_norm * mode_dir

                        # 获取baseline
                        prompt = TEMPLATE.format(word)
                        base_top1, base_logits = get_baseline_prediction(model, tokenizer, device, prompt)

                        # 计算baseline各类别logit
                        base_cat_logits = {}
                        for cat in categories:
                            cat_logit = float(np.max(base_logits[cat_token_ids[cat]])) if len(cat_token_ids[cat]) > 0 else 0
                            base_cat_logits[cat] = cat_logit

                        # 获取perturbed
                        target_module = layers_list[li].self_attn if space == "attn" else layers_list[li].mlp
                        toks = tokenizer(prompt, return_tensors="pt").to(device)
                        last_pos = toks.input_ids.shape[1] - 1

                        pert_top1, pert_logits = get_perturbed_prediction(
                            model, tokenizer, device, prompt, target_module, perturb_vec, last_pos
                        )

                        # 计算perturbed各类别logit
                        pert_cat_logits = {}
                        for cat in categories:
                            cat_logit = float(np.max(pert_logits[cat_token_ids[cat]])) if len(cat_token_ids[cat]) > 0 else 0
                            pert_cat_logits[cat] = cat_logit

                        # 计算各类别logit变化
                        for cat in categories:
                            delta = pert_cat_logits[cat] - base_cat_logits[cat]
                            shift_logit_changes[cat].append(delta)

                        # 判断偏移方向: 哪个类别logit增加最多?
                        max_increase_cat = max(categories, key=lambda c: pert_cat_logits[c] - base_cat_logits[c])
                        shift_cats.append(max_increase_cat)

                    if not shift_cats:
                        continue

                    # 分析偏移方向的一致性
                    cat_counter = Counter(shift_cats)
                    total = len(shift_cats)
                    dominant_cat = cat_counter.most_common(1)[0]
                    dominant_ratio = dominant_cat[1] / total

                    # 随机基线: 4个类别均匀分布 → dominant_ratio = 0.25
                    # 如果 dominant_ratio >> 0.25 → 有方向性

                    # 计算direction_consistency (信息熵)
                    probs = np.array([cat_counter[c] / total for c in categories])
                    entropy = -np.sum(probs * np.log2(probs + 1e-10))
                    max_entropy = np.log2(len(categories))  # 2.0 for 4 categories
                    consistency = 1 - entropy / max_entropy  # 0=随机, 1=完全一致

                    # 平均logit变化
                    avg_delta = {cat: np.mean(shift_logit_changes[cat]) for cat in categories}

                    key = f"L{li}_{space}_{mode_type}_M{mi}"
                    results[key] = {
                        "layer": li, "space": space, "mode_type": mode_type, "mode_idx": mi,
                        "consistency": consistency,
                        "dominant_cat": dominant_cat[0],
                        "dominant_ratio": dominant_ratio,
                        "shift_distribution": {c: cat_counter.get(c, 0) for c in categories},
                        "avg_logit_change": avg_delta,
                    }

                    log(f"    {mode_type}_M{mi}: consistency={consistency:.3f}, "
                        f"dominant={dominant_cat[0]}({dominant_ratio:.2f}), "
                        f"entropy={entropy:.2f}/{max_entropy:.2f}")

    # Step 4: 汇总比较 sem vs nosem
    log("\n--- Step 4: sem vs nosem 方向性比较 ---")

    sem_consistencies = []
    nosem_consistencies = []
    sem_dominant_ratios = []
    nosem_dominant_ratios = []

    for key, val in results.items():
        if val["mode_type"] == "sem":
            sem_consistencies.append(val["consistency"])
            sem_dominant_ratios.append(val["dominant_ratio"])
        else:
            nosem_consistencies.append(val["consistency"])
            nosem_dominant_ratios.append(val["dominant_ratio"])

    sem_mean_cons = np.mean(sem_consistencies) if sem_consistencies else 0
    nosem_mean_cons = np.mean(nosem_consistencies) if nosem_consistencies else 0
    sem_mean_dom = np.mean(sem_dominant_ratios) if sem_dominant_ratios else 0
    nosem_mean_dom = np.mean(nosem_dominant_ratios) if nosem_dominant_ratios else 0

    log(f"\n  sem模式:   mean_consistency={sem_mean_cons:.3f}, mean_dominant_ratio={sem_mean_dom:.3f}")
    log(f"  nosem模式: mean_consistency={nosem_mean_cons:.3f}, mean_dominant_ratio={nosem_mean_dom:.3f}")
    log(f"  sem/nosem consistency ratio: {sem_mean_cons/max(nosem_mean_cons, 1e-6):.2f}x")
    log(f"  随机基线: dominant_ratio=0.250, consistency=0.000")

    # 逐层分析
    log("\n--- 逐层方向性分析 ---")
    for li in key_layers:
        for space in ["attn", "mlp"]:
            sem_c = [results[k]["consistency"] for k in results
                     if results[k]["layer"] == li and results[k]["space"] == space and results[k]["mode_type"] == "sem"]
            nosem_c = [results[k]["consistency"] for k in results
                       if results[k]["layer"] == li and results[k]["space"] == space and results[k]["mode_type"] == "nosem"]
            sem_d = [results[k]["dominant_ratio"] for k in results
                     if results[k]["layer"] == li and results[k]["space"] == space and results[k]["mode_type"] == "sem"]
            nosem_d = [results[k]["dominant_ratio"] for k in results
                       if results[k]["layer"] == li and results[k]["space"] == space and results[k]["mode_type"] == "nosem"]

            if sem_c and nosem_c:
                log(f"  L{li} {space}: sem_cons={np.mean(sem_c):.3f} vs nosem_cons={np.mean(nosem_c):.3f} "
                    f"({np.mean(sem_c)/max(np.mean(nosem_c),1e-6):.2f}x), "
                    f"sem_dom={np.mean(sem_d):.3f} vs nosem_dom={np.mean(nosem_d):.3f}")

    # Step 5: 语义模式偏移方向与类别质心差的对齐
    log("\n--- Step 5: 语义模式偏移方向分析 ---")

    for li in key_layers:
        for space in ["attn", "mlp"]:
            svd_data = layer_svd.get((li, space))
            if svd_data is None:
                continue
            centroids = layer_centroids.get((li, space), {})

            log(f"\n  === L{li} {space} 语义模式偏移方向 ===")

            for mi in svd_data["top3_sem"][:2]:  # 只看top2语义模式
                if mi >= svd_data["Vt"].shape[0]:
                    continue
                mode_dir = svd_data["Vt"][mi]

                # 该模式对各词的偏移偏好
                r_key = f"L{li}_{space}_sem_M{mi}"
                if r_key in results:
                    r = results[r_key]
                    log(f"    M{mi}: dominant_shift→{r['dominant_cat']}({r['dominant_ratio']:.2f}), "
                        f"consistency={r['consistency']:.3f}")

                    # 检查偏移方向是否对齐类别质心差
                    if len(centroids) >= 2:
                        cat_list = list(centroids.keys())
                        logit_changes = r["avg_logit_change"]
                        sorted_cats = sorted(cat_list, key=lambda c: logit_changes.get(c, 0), reverse=True)
                        log(f"           logit变化排序: {[(c, f'{logit_changes.get(c,0):.2f}') for c in sorted_cats]}")

    # 保存结果
    out_file = TEMP_DIR / f"ccxi_shift_direction_{model_name}.json"
    with open(out_file, "w") as f:
        json.dump({
            "model": model_name,
            "d_model": d_model,
            "n_layers": n_layers,
            "key_layers": key_layers,
            "alpha": ALPHA,
            "sem_mean_consistency": sem_mean_cons,
            "nosem_mean_consistency": nosem_mean_cons,
            "sem_mean_dominant_ratio": sem_mean_dom,
            "nosem_mean_dominant_ratio": nosem_mean_dom,
            "random_baseline_dominant": 0.25,
            "results": results,
        }, f, indent=2, default=str)

    log(f"\n结果已保存: {out_file}")

    release_model(model)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    run_experiment(args.model)
