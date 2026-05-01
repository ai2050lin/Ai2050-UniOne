"""
CCXII(312): 扩展类别测试 — 验证低维控制的普适性
=================================================
CCX发现: 4类别时1-5个SVD模式可100%控制输出。
但这可能只是4类别的特殊情况(4类→1-2维足够)。

本实验:
  1. 扩展到8个类别, 验证低维控制是否仍然成立
  2. 如果8类别需要更多维度 → 低维控制是4类别的artifacts
  3. 如果8类别仍只需1-5维 → 低维控制是普适的

8个类别:
  animals, food, tools, nature, vehicles, clothing, body, weather

指标:
  - MCV(K): top-K个SVD模式perturb后top1改变率
  - K_min: 达到90%改变率所需的最小K
  - 与4类别时的K_min对比

用法:
  python ccxii_extended_categories.py --model qwen3
  python ccxii_extended_categories.py --model glm4
  python ccxii_extended_categories.py --model deepseek7b
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
LOG_FILE = TEMP_DIR / "ccxii_extended_categories_log.txt"

# 8个类别, 每个8个词
CONCEPTS_8 = {
    "animals":   ["dog", "cat", "horse", "eagle", "shark", "lion", "bear", "whale"],
    "food":      ["apple", "rice", "bread", "cheese", "pizza", "banana", "mango", "steak"],
    "tools":     ["hammer", "knife", "saw", "drill", "wrench", "pliers", "chisel", "ruler"],
    "nature":    ["mountain", "river", "ocean", "forest", "desert", "valley", "canyon", "glacier"],
    "vehicles":  ["car", "bus", "train", "plane", "boat", "truck", "bicycle", "helicopter"],
    "clothing":  ["shirt", "pants", "dress", "jacket", "shoes", "hat", "coat", "socks"],
    "body":      ["head", "hand", "foot", "arm", "leg", "eye", "heart", "brain"],
    "weather":   ["rain", "snow", "wind", "storm", "cloud", "fog", "thunder", "frost"],
}

TEMPLATE = "The {} is"


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run_experiment(model_name):
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)

    n_layers = info.n_layers
    d_model = info.d_model
    n_cats = len(CONCEPTS_8)

    # 构建词列表
    all_words = []
    all_cats = []
    for cat, words in CONCEPTS_8.items():
        for w in words:
            all_words.append(w)
            all_cats.append(cat)
    categories = list(CONCEPTS_8.keys())

    # 测试层: 最后一层
    target_layer = n_layers - 1
    # 也测试中间层和浅层
    test_layers = sorted(set([0, n_layers // 4, n_layers // 2, n_layers - 2, n_layers - 1]))

    log(f"\n{'='*70}")
    log(f"CCXII(312): 扩展类别测试(8类) - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}, categories={n_cats}")
    log(f"  测试层: {test_layers}")
    log(f"  总词数: {len(all_words)}")
    log(f"{'='*70}")

    # Step 1: 收集各词在各层的Attn/MLP输出
    log("\n--- Step 1: 收集各词在各层的Attn/MLP输出 ---")

    word_outputs = {li: {} for li in test_layers}

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
        for li in test_layers:
            layer = layers_list[li]
            hooks.append(layer.self_attn.register_forward_hook(make_capture_hook(f"attn_L{li}")))
            hooks.append(layer.mlp.register_forward_hook(make_capture_hook(f"mlp_L{li}")))

        with torch.no_grad():
            _ = model(**toks)

        for h in hooks:
            h.remove()

        for li in test_layers:
            attn_data = captured.get(f"attn_L{li}", None)
            mlp_data = captured.get(f"mlp_L{li}", None)
            word_outputs[li][word] = {
                "attn": attn_data,
                "mlp": mlp_data,
            }

        if (wi + 1) % 16 == 0:
            log(f"  收集 {wi+1}/{len(all_words)} 词完成")

    # Step 2: 对每层计算SVD, 然后做MCV实验
    log("\n--- Step 2: MCV实验 (8类别) ---")

    results = {}

    for li in test_layers:
        for space in ["attn", "mlp"]:
            # 收集向量
            vecs = []
            valid_words = []
            valid_cats = []
            for w, c in zip(all_words, all_cats):
                v = word_outputs[li][w].get(space)
                if v is not None:
                    vecs.append(v)
                    valid_words.append(w)
                    valid_cats.append(c)

            if len(vecs) < n_cats + 1:
                continue

            vecs = np.array(vecs)

            # SVD
            mean_vec = vecs.mean(axis=0)
            centered = vecs - mean_vec
            n_svd = min(50, centered.shape[0] - 1, centered.shape[1] - 1)
            if n_svd < 3:
                continue

            U, s, Vt = np.linalg.svd(centered, full_matrices=False)
            Vt = Vt[:n_svd, :]
            s = s[:n_svd]

            # 计算每个SVD模式的语义区分力
            projections = centered @ Vt.T
            mode_f_scores = []
            for mi in range(n_svd):
                proj_mi = projections[:, mi]
                groups = []
                for cat in categories:
                    cat_idx = [i for i, c in enumerate(valid_cats) if c == cat]
                    if cat_idx:
                        groups.append(proj_mi[cat_idx])
                if len(groups) >= 2:
                    f_stat, _ = stats.f_oneway(*groups)
                    mode_f_scores.append(f_stat if not np.isnan(f_stat) else 0)
                else:
                    mode_f_scores.append(0)

            # 按F值排序
            sorted_modes = np.argsort(mode_f_scores)[::-1]
            top10_modes = sorted_modes[:10].tolist()

            log(f"\n  === L{li} {space} (8类) ===")
            log(f"    top-10 F值: {[f'{mode_f_scores[m]:.1f}' for m in top10_modes]}")

            # MCV实验: top-K个语义SVD模式perturb
            ALPHA = 1.0
            for K in [1, 3, 5, 10, 15, 20]:
                if K > len(top10_modes) and K > n_svd:
                    continue

                top_k_modes = sorted_modes[:K].tolist()

                # 计算perturbation向量
                # 对每个词用不同的perturb_vec (基于该词的output norm)
                n_changed = 0
                n_total = 0

                for wi, word in enumerate(valid_words):
                    v = word_outputs[li][word].get(space)
                    if v is None:
                        continue

                    output_norm = np.linalg.norm(v)
                    if output_norm < 1e-10:
                        continue

                    # 合成perturbation向量
                    perturb_vec_np = np.zeros(d_model)
                    for mi in top_k_modes:
                        perturb_vec_np += ALPHA * output_norm * Vt[mi]

                    # 获取baseline预测
                    prompt = TEMPLATE.format(word)
                    toks = tokenizer(prompt, return_tensors="pt").to(device)
                    last_pos = toks.input_ids.shape[1] - 1

                    with torch.no_grad():
                        base_out = model(**toks)
                    base_top1 = int(np.argmax(base_out.logits[0, -1, :].float().cpu().numpy()))

                    # Perturbed预测
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

                    target_module = layers_list[li].self_attn if space == "attn" else layers_list[li].mlp
                    hook_handle = target_module.register_forward_hook(make_hook(perturb_vec_np, intervention_done, last_pos))

                    with torch.no_grad():
                        pert_out = model(**toks)
                    pert_top1 = int(np.argmax(pert_out.logits[0, -1, :].float().cpu().numpy()))

                    hook_handle.remove()

                    if pert_top1 != base_top1:
                        n_changed += 1
                    n_total += 1

                change_rate = n_changed / n_total if n_total > 0 else 0
                key = f"L{li}_{space}_K{K}"
                results[key] = {
                    "layer": li, "space": space, "K": K,
                    "n_changed": n_changed, "n_total": n_total,
                    "change_rate": change_rate,
                }
                log(f"    K={K:2d}: {n_changed}/{n_total} = {change_rate:.1%}")

    # Step 3: 汇总 — K_min(90%阈值)
    log("\n--- Step 3: K_min(90%阈值)汇总 ---")

    k_min_90 = {}
    for li in test_layers:
        for space in ["attn", "mlp"]:
            k_list = sorted([results[k]["K"] for k in results
                           if results[k]["layer"] == li and results[k]["space"] == space])
            for k in k_list:
                key = f"L{li}_{space}_K{k}"
                if results[key]["change_rate"] >= 0.9:
                    k_min_90[(li, space)] = k
                    break
            else:
                k_min_90[(li, space)] = ">max"

            log(f"  L{li} {space}: K_min(90%)={k_min_90.get((li, space), 'N/A')}")

    # 对比: 4类别时的K_min(从CCX数据)
    log("\n--- 对比4类别 vs 8类别 ---")
    log(f"  4类别 (CCX数据): DS7B Attn K1=97%, GLM4 MLP K1=78%, Qwen3 MLP K5=97%")
    log(f"  8类别 K_min(90%):")

    last_li = n_layers - 1
    for space in ["attn", "mlp"]:
        k = k_min_90.get((last_li, space), "N/A")
        log(f"    {space}: K_min={k}")

    # 保存结果
    out_file = TEMP_DIR / f"ccxii_extended_categories_{model_name}.json"
    with open(out_file, "w") as f:
        json.dump({
            "model": model_name,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_categories": n_cats,
            "test_layers": test_layers,
            "alpha": ALPHA,
            "k_min_90": {f"L{k[0]}_{k[1]}": v for k, v in k_min_90.items()},
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
