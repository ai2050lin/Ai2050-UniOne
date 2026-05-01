"""
CCX(310): 精细因果控制 — 单SVD模式Perturb + 剂量反应曲线
==========================================================
用户关键审视: 当前swap整个模块太粗, 需要更精细干预。
CCVIII发现: Attention语义方向97%能量在前10个W_U模式(DS7B)。

本实验:
  Exp1: 沿Attention/MLP输出空间的单个SVD模式perturb
    - 选择最有语义区分力的SVD模式(F值top-3)
    - 沿该模式加不同强度的扰动(alpha=0.1-2.0)
    - 测量: top-1是否改变? 语义是否偏移?

  Exp2: 剂量-反应曲线
    - alpha从0到2, 连续调节
    - 观察: 语义偏移是渐变还是相变?
    - 如果是相变 → 语义编码是离散的
    - 如果是渐变 → 语义编码是连续的

  Exp3: 最小控制变量集
    - 只perturb top-K个SVD模式, K=1,3,5,10
    - 看K=1是否足够? K=5是否饱和?
    - 目标: 找到控制语义输出的最小维度

用法:
  python ccx_svd_mode_perturb.py --model qwen3
  python ccx_svd_mode_perturb.py --model glm4
  python ccx_svd_mode_perturb.py --model deepseek7b
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
LOG_FILE = TEMP_DIR / "ccx_svd_mode_perturb_log.txt"

CONCEPTS = {
    "animals": ["dog", "cat", "horse", "eagle", "shark", "lion", "bear", "fish", "snake", "whale"],
    "food":    ["apple", "rice", "bread", "cheese", "pizza", "banana", "mango", "pasta", "salad", "steak"],
    "tools":   ["hammer", "knife", "saw", "drill", "wrench", "screw", "pliers", "chisel", "level", "ruler"],
    "nature":  ["mountain", "river", "ocean", "forest", "desert", "valley", "canyon", "island", "meadow", "glacier"],
}

TEMPLATE = "The {} is"
N_WORDS_PER_CAT = 8  # 减少词数, 加快实验速度


def log_f(msg="", end="\n"):
    print(msg, end=end, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + end)


def run_experiment(model_name):
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    log_f(f"\n{'#'*70}")
    log_f(f"CCX(310): SVD Mode Perturb + Dose-Response Curve")
    log_f(f"Model: {model_name}")
    log_f(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_f(f"{'#'*70}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers_list = get_layers(model)

    categories = list(CONCEPTS.keys())

    # 采样词
    rng = np.random.RandomState(42)
    all_words = []
    all_cats = []
    for cat, words in CONCEPTS.items():
        sel = rng.choice(words, min(N_WORDS_PER_CAT, len(words)), replace=False)
        all_words.extend(sel.tolist() if hasattr(sel, 'tolist') else list(sel))
        all_cats.extend([cat] * min(N_WORDS_PER_CAT, len(words)))

    # 选择3个测试层: 1/4, 1/2, 3/4
    test_layers = sorted(set([
        n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1
    ]))
    log_f(f"  Test layers: {test_layers}")

    # ===== Step 1: Collect Attn/MLP outputs + SVD =====
    log_f("\n--- Step 1: Collecting Attn/MLP outputs ---")

    # 收集各词在各层的attn/mlp输出
    word_outputs = {li: {} for li in test_layers}

    for wi, word in enumerate(all_words):
        text = TEMPLATE.format(word)
        input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
        last_pos = input_ids.shape[1] - 1

        captured = {}

        def make_hook(key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured[key] = output[0][0, last_pos].detach().float().cpu().numpy()
                else:
                    captured[key] = output[0, last_pos].detach().float().cpu().numpy() if output.ndim > 1 else output.detach().float().cpu().numpy()
            return hook

        hooks = []
        for li in test_layers:
            layer = layers_list[li]
            hooks.append(layer.self_attn.register_forward_hook(make_hook(f"attn_L{li}")))
            hooks.append(layer.mlp.register_forward_hook(make_hook(f"mlp_L{li}")))

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

        for li in test_layers:
            for space in ["attn", "mlp"]:
                key = f"{space}_L{li}"
                if key in captured:
                    word_outputs[li].setdefault(word, {})[space] = captured[key]

        if (wi + 1) % 5 == 0:
            log_f(f"  Processed {wi+1}/{len(all_words)} words")

    # ===== Step 2: SVD of Attn/MLP outputs + semantic discrimination =====
    log_f(f"\n{'='*70}")
    log_f(f"Step 2: SVD + Semantic Mode Identification")
    log_f(f"{'='*70}")

    svd_results = {}  # {li: {space: {U, s, Vt, mode_discrim}}}

    for li in test_layers:
        svd_results[li] = {}

        for space in ["attn", "mlp"]:
            all_vecs = []
            all_labels_cat = []
            for cat in categories:
                cat_words_list = [w for w, c in zip(all_words, all_cats) if c == cat and w in word_outputs[li] and space in word_outputs[li][w]]
                for w in cat_words_list:
                    all_vecs.append(word_outputs[li][w][space])
                    all_labels_cat.append(cat)

            if len(all_vecs) < 8:
                continue

            all_vecs = np.array(all_vecs)  # [n_words, d_model]
            mean_vec = all_vecs.mean(axis=0)
            centered = all_vecs - mean_vec

            # SVD
            k_svd = min(30, min(centered.shape) - 1)
            U_local, s_local, Vt_local = svds(centered, k=k_svd)
            sort_idx = np.argsort(-s_local)
            U_local = U_local[:, sort_idx]
            s_local = s_local[sort_idx]
            Vt_local = Vt_local[sort_idx]

            # 每个SVD模式的语义区分力
            mode_discrim = []
            for mi in range(k_svd):
                mode_coeffs = centered @ Vt_local[mi]
                cat_groups = defaultdict(list)
                for coeff, cat in zip(mode_coeffs, all_labels_cat):
                    cat_groups[cat].append(coeff)

                group_data = [cat_groups[c] for c in categories if c in cat_groups]
                if len(group_data) >= 2:
                    f_stat, p_val = stats.f_oneway(*group_data)
                    mode_discrim.append({
                        "mode_idx": mi,
                        "f_stat": float(f_stat) if not np.isnan(f_stat) else 0,
                        "p_val": float(p_val) if not np.isnan(p_val) else 1.0,
                    })

            # 排序
            mode_discrim.sort(key=lambda x: -x["f_stat"])

            svd_results[li][space] = {
                "U": U_local,
                "s": s_local,
                "Vt": Vt_local,
                "mean": mean_vec,
                "mode_discrim": mode_discrim,
                "all_vecs": all_vecs,
                "all_labels": all_labels_cat,
            }

            top3 = mode_discrim[:3]
            log_f(f"  L{li} {space}: top-3 modes = {[(m['mode_idx'], m['f_stat']) for m in top3]}")

    # ===== Step 3: Single-Mode Perturb + Dose-Response =====
    log_f(f"\n{'='*70}")
    log_f(f"Step 3: Single-Mode Perturb + Dose-Response")
    log_f(f"{'='*70}")

    # Alpha值(剂量)
    alpha_list = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]

    # 只测试最有语义区分力的3个模式
    n_top_modes = 3

    perturb_results = {}

    for li in test_layers:
        perturb_results[str(li)] = {}

        for space in ["attn", "mlp"]:
            if space not in svd_results[li]:
                continue

            svd_data = svd_results[li][space]
            Vt = svd_data["Vt"]  # [k, d_model]
            s = svd_data["s"]
            mode_discrim = svd_data["mode_discrim"]
            all_vecs = svd_data["all_vecs"]
            all_labels = svd_data["all_labels"]

            # 选择top-3语义模式 + 1个非语义模式(对比)
            top_modes = [m["mode_idx"] for m in mode_discrim[:n_top_modes]]
            # 找一个低区分力模式
            low_disc_modes = [m for m in mode_discrim if m["f_stat"] < 1.0]
            low_mode = low_disc_modes[-1]["mode_idx"] if low_disc_modes else mode_discrim[-1]["mode_idx"]

            test_modes = top_modes + [low_mode]
            mode_names = [f"sem_M{m}" for m in top_modes] + [f"nosem_M{low_mode}"]

            log_f(f"\n  L{li} {space}: testing modes {test_modes}")

            # 对每个词做perturb
            space_results = {}

            for mode_idx, mode_name in zip(test_modes, mode_names):
                mode_result = {"alphas": alpha_list, "top1_changed": [], "logit_cos": [], "margin_change": []}

                # 模式方向(在d_model空间)
                mode_dir = Vt[mode_idx]  # [d_model]

                for alpha in alpha_list:
                    n_changed = 0
                    logit_cos_sum = 0
                    margin_change_sum = 0
                    n_tested = 0

                    for word_idx, word in enumerate(all_words):
                        cat = all_cats[word_idx]
                        if word not in word_outputs[li] or space not in word_outputs[li][word]:
                            continue

                        text = TEMPLATE.format(word)
                        input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
                        last_pos = input_ids.shape[1] - 1

                        # Baseline logits
                        with torch.no_grad():
                            base_logits = model(input_ids).logits[0, last_pos].detach().float().cpu().numpy()
                        base_top1 = np.argmax(base_logits)

                        # Perturb: 在目标层的attn/mlp输出中沿SVD模式加扰动
                        # 用hook修改输出
                        intervention_done = [False]

                        # 计算perturbation向量 — norm-relative缩放
                        # 扰动大小 = alpha * ||output||, 沿SVD模式方向
                        output_norm = np.linalg.norm(word_outputs[li][word][space])
                        if output_norm > 1e-10 and alpha > 0:
                            # mode_dir已归一化(Vt的行), 扰动向量 = alpha * output_norm * mode_dir
                            perturb_vec_np = alpha * output_norm * mode_dir
                        else:
                            perturb_vec_np = np.zeros(d_model)

                        def make_perturb_hook(pv_np, done_flag, lp):
                            def hook(module, input, output):
                                if done_flag[0]:
                                    return output
                                done_flag[0] = True

                                if isinstance(output, tuple):
                                    out = output[0]
                                else:
                                    out = output

                                # 在输出上加扰动
                                perturb_tensor = torch.tensor(pv_np, dtype=out.dtype, device=device)
                                new_out = out.clone()
                                new_out[0, lp, :] += perturb_tensor

                                if isinstance(output, tuple):
                                    return (new_out,) + output[1:]
                                return new_out
                            return hook

                        # Register hook on target module
                        if space == "attn":
                            target_module = layers_list[li].self_attn
                        else:
                            target_module = layers_list[li].mlp

                        hook_handle = target_module.register_forward_hook(make_perturb_hook(perturb_vec_np, intervention_done, last_pos))

                        with torch.no_grad():
                            interv_logits = model(input_ids).logits[0, last_pos].detach().float().cpu().numpy()

                        hook_handle.remove()

                        interv_top1 = np.argmax(interv_logits)
                        n_changed += int(base_top1 != interv_top1)

                        # Logit cosine
                        logit_cos = float(np.dot(base_logits, interv_logits) /
                                         (np.linalg.norm(base_logits) * np.linalg.norm(interv_logits) + 1e-10))
                        logit_cos_sum += logit_cos

                        # Margin change
                        word_tok_ids = tokenizer.encode(word, add_special_tokens=False)
                        word_tok_id = word_tok_ids[0] if word_tok_ids else -1
                        if word_tok_id >= 0:
                            base_margin = base_logits[word_tok_id] - np.max(np.delete(base_logits, word_tok_id))
                            interv_margin = interv_logits[word_tok_id] - np.max(np.delete(interv_logits, word_tok_id))
                            margin_change_sum += interv_margin - base_margin

                        n_tested += 1

                    if n_tested > 0:
                        mode_result["top1_changed"].append(float(n_changed / n_tested))
                        mode_result["logit_cos"].append(float(logit_cos_sum / n_tested))
                        mode_result["margin_change"].append(float(margin_change_sum / n_tested))
                    else:
                        mode_result["top1_changed"].append(0)
                        mode_result["logit_cos"].append(1.0)
                        mode_result["margin_change"].append(0)

                # 判断: 渐变 vs 相变
                top1_rates = mode_result["top1_changed"]
                if len(top1_rates) >= 3:
                    # 相变特征: top1率从<5%突然跳到>20%
                    has_phase_transition = False
                    for i in range(1, len(top1_rates)):
                        if top1_rates[i] - top1_rates[i-1] > 0.15:  # 15%跳变
                            has_phase_transition = True
                            break
                    mode_result["phase_transition"] = has_phase_transition
                else:
                    mode_result["phase_transition"] = False

                log_f(f"    {mode_name}: top1_changed={mode_result['top1_changed']}, "
                      f"phase_transition={mode_result.get('phase_transition', 'N/A')}")

                space_results[mode_name] = mode_result

            perturb_results[str(li)][space] = space_results

    # ===== Step 4: Minimal Control Variable Set =====
    log_f(f"\n{'='*70}")
    log_f(f"Step 4: Minimal Control Variable Set (Top-K SVD Modes)")
    log_f(f"{'='*70}")

    k_values = [1, 3, 5, 10]
    alpha_fixed = 1.0  # 固定扰动强度

    mcv_results = {}

    for li in test_layers:
        mcv_results[str(li)] = {}

        for space in ["attn", "mlp"]:
            if space not in svd_results[li]:
                continue

            svd_data = svd_results[li][space]
            Vt = svd_data["Vt"]
            s = svd_data["s"]
            mode_discrim = svd_data["mode_discrim"]

            for k in k_values:
                # 扰动前k个语义SVD模式
                top_k_modes = [m["mode_idx"] for m in mode_discrim[:k]]

                n_changed = 0
                n_tested = 0

                for word_idx, word in enumerate(all_words):
                    if word not in word_outputs[li] or space not in word_outputs[li][word]:
                        continue

                    text = TEMPLATE.format(word)
                    input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
                    last_pos = input_ids.shape[1] - 1

                    # Baseline
                    with torch.no_grad():
                        base_logits = model(input_ids).logits[0, last_pos].detach().float().cpu().numpy()
                    base_top1 = np.argmax(base_logits)

                    # 计算多模式perturbation — norm-relative缩放
                    output_norm = np.linalg.norm(word_outputs[li][word][space])
                    perturb_vec_np = np.zeros(d_model)
                    if output_norm > 1e-10:
                        for mi in top_k_modes:
                            perturb_vec_np += alpha_fixed * output_norm * Vt[mi]

                    intervention_done = [False]

                    def make_perturb_hook_mc(pv_np, done_flag, lp):
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

                    if space == "attn":
                        target_module = layers_list[li].self_attn
                    else:
                        target_module = layers_list[li].mlp

                    hook_handle = target_module.register_forward_hook(make_perturb_hook_mc(perturb_vec_np, intervention_done, last_pos))

                    with torch.no_grad():
                        interv_logits = model(input_ids).logits[0, last_pos].detach().float().cpu().numpy()

                    hook_handle.remove()

                    interv_top1 = np.argmax(interv_logits)
                    n_changed += int(base_top1 != interv_top1)
                    n_tested += 1

                top1_rate = n_changed / n_tested if n_tested > 0 else 0
                mcv_results[str(li)].setdefault(space, {})[f"top{k}"] = {
                    "top1_rate": float(top1_rate),
                    "n_modes": k,
                }

                log_f(f"  L{li} {space} top-{k}: top1_changed={top1_rate:.3f}")

    # ===== Save results =====
    results = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "test_layers": test_layers,
        "perturb_results": perturb_results,
        "mcv_results": mcv_results,
        "elapsed_seconds": time.time() - start_time,
    }

    # 转换numpy类型
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    results = convert_numpy(results)

    out_path = TEMP_DIR / f"ccx_svd_mode_perturb_{model_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log_f(f"\n  Results saved to {out_path}")

    # ===== Summary =====
    log_f(f"\n{'='*70}")
    log_f(f"SUMMARY: SVD Mode Perturb + Dose-Response")
    log_f(f"{'='*70}")

    for li in test_layers:
        li_str = str(li)
        log_f(f"\n  L{li}:")
        for space in ["attn", "mlp"]:
            if li_str in perturb_results and space in perturb_results[li_str]:
                for mode_name, mode_data in perturb_results[li_str][space].items():
                    top1_at_1 = mode_data["top1_changed"][4] if len(mode_data["top1_changed"]) > 4 else 0  # alpha=1.0
                    top1_at_2 = mode_data["top1_changed"][6] if len(mode_data["top1_changed"]) > 6 else 0  # alpha=2.0
                    phase = mode_data.get("phase_transition", False)
                    log_f(f"    {space} {mode_name}: top1@1.0={top1_at_1:.3f}, top1@2.0={top1_at_2:.3f}, "
                          f"phase={phase}")

    log_f(f"\n  Minimal Control Variable Set:")
    for li in test_layers:
        li_str = str(li)
        for space in ["attn", "mlp"]:
            if li_str in mcv_results and space in mcv_results[li_str]:
                k_results = mcv_results[li_str][space]
                k_line = ", ".join([f"K{k_data['n_modes']}={k_data['top1_rate']:.3f}" for k_data in k_results.values()])
                log_f(f"    L{li} {space}: {k_line}")

    release_model(model)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    run_experiment(args.model)
