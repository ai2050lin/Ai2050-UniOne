"""
CCXVI(316): 传递性推理测试 — A>B + B>C = A>C 是否线性?
======================================================================
如果语义编码的核心是关系, 那么逻辑推理也应该是关系操作。
传递性是最基本的逻辑关系: A>B, B>C → A>C。

核心问题:
  1. 传递性推理在语义空间中是否是线性的?
  2. vec(A>B) + vec(B>C) ≈ vec(A>C)?
  3. 如果是 → 语言=关系代数, 推理=向量运算
  4. 不同类型推理(大小/重量/速度)是否共享几何结构?

设计:
  - 三元组推理链: A > B > C (大小/重量/速度)
  - 收集 "A is bigger than B" 的残差流
  - 计算 vec(A>B) = resid(A>B) - resid(baseline)
  - 测试: vec(A>B) + vec(B>C) ≈ vec(A>C)?
  - 线性度 = cos(vec(A>B)+vec(B>C), vec(A>C))

关键指标:
  - 线性度(余弦相似度): >0.5 = 强线性, <0.3 = 非线性
  - 蒸馏vs非蒸馏: DS7B(1维)是否更线性?

用法:
  python ccxvi_transitive_reasoning.py --model qwen3
  python ccxvi_transitive_reasoning.py --model glm4
  python ccxvi_transitive_reasoning.py --model deepseek7b
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

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, get_W_U,
    MODEL_CONFIGS,
)

TEMP_DIR = Path("tests/glm5_temp")
LOG_FILE = TEMP_DIR / "ccxvi_transitive_reasoning_log.txt"

# 推理链三元组
# 每种推理维度10条链
SIZE_CHAINS = [
    ("elephant", "dog", "mouse"),
    ("whale", "dolphin", "fish"),
    ("mountain", "hill", "rock"),
    ("house", "car", "bicycle"),
    ("tree", "bush", "flower"),
    ("lion", "cat", "kitten"),
    ("horse", "goat", "rabbit"),
    ("eagle", "sparrow", "butterfly"),
    ("shark", "salmon", "shrimp"),
    ("bear", "fox", "squirrel"),
]

WEIGHT_CHAINS = [
    ("elephant", "cow", "cat"),
    ("car", "bicycle", "shoe"),
    ("rock", "book", "feather"),
    ("tree", "chair", "pen"),
    ("whale", "seal", "crab"),
    ("mountain", "house", "phone"),
    ("bear", "dog", "mouse"),
    ("horse", "sheep", "rabbit"),
    ("table", "plate", "spoon"),
    ("brick", "apple", "grape"),
]

SPEED_CHAINS = [
    ("cheetah", "horse", "turtle"),
    ("falcon", "eagle", "penguin"),
    ("car", "bicycle", "walker"),
    ("rocket", "plane", "boat"),
    ("tiger", "deer", "snail"),
    ("shark", "dolphin", "crab"),
    ("motorcycle", "car", "tractor"),
    ("lion", "zebra", "sloth"),
    ("train", "bus", "tractor"),
    ("wind", "rain", "snow"),
]

# 模板
COMPARISON_TEMPLATES = {
    "size": "The {} is bigger than the",
    "weight": "The {} is heavier than the",
    "speed": "The {} is faster than the",
}

# 基线模板(无比较)
BASELINE_TEMPLATE = "The {} is"


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

    # 采样层
    test_layers = sorted(set([
        0, n_layers // 4, n_layers // 2, 3 * n_layers // 4,
        n_layers - 2, n_layers - 1
    ]))

    log(f"\n{'='*70}")
    log(f"CCXVI(316): 传递性推理测试 - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}")
    log(f"  测试层: {test_layers}")
    log(f"  推理维度: {list(COMPARISON_TEMPLATES.keys())}")
    log(f"{'='*70}")

    results = {}

    # ============================================================
    # Step 1: 收集比较句和基线句的残差流
    # ============================================================
    log("\n--- Step 1: 收集残差流 ---")

    # resid_cache[template_type][li][word] = residual vector
    resid_cache = defaultdict(lambda: defaultdict(dict))

    all_chains = {
        "size": SIZE_CHAINS,
        "weight": WEIGHT_CHAINS,
        "speed": SPEED_CHAINS,
    }

    # 收集所有需要测试的词
    all_words_set = set()
    for dim, chains in all_chains.items():
        for a, b, c in chains:
            all_words_set.update([a, b, c])

    all_words = sorted(all_words_set)
    log(f"  总词数: {len(all_words)}")

    # 先收集基线
    log(f"\n  收集基线残差...")
    for word in all_words:
        prompt = BASELINE_TEMPLATE.format(word)
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
            hooks.append(layers_list[li].register_forward_hook(make_capture_hook(f"L{li}")))

        with torch.no_grad():
            _ = model(**toks)

        for h in hooks:
            h.remove()

        for li in test_layers:
            if f"L{li}" in captured:
                resid_cache["baseline"][li][word] = captured[f"L{li}"]

    log(f"  基线收集完成")

    # 收集比较句
    for dim, chains in all_chains.items():
        template = COMPARISON_TEMPLATES[dim]
        log(f"\n  收集 {dim} 比较残差...")

        for a, b, c in chains:
            for x in [a, b, c]:
                if x in resid_cache[dim][test_layers[-1]]:
                    continue

                prompt = template.format(x)
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
                    hooks.append(layers_list[li].register_forward_hook(make_capture_hook(f"L{li}")))

                with torch.no_grad():
                    _ = model(**toks)

                for h in hooks:
                    h.remove()

                for li in test_layers:
                    if f"L{li}" in captured:
                        resid_cache[dim][li][x] = captured[f"L{li}"]

        log(f"    {dim} 收集完成")

    # ============================================================
    # Step 2: 传递性线性度测试
    # ============================================================
    log("\n--- Step 2: 传递性线性度测试 ---")

    for dim, chains in all_chains.items():
        log(f"\n  === {dim} 推理 ===")

        for li in test_layers:
            linearities = []

            for a, b, c in chains:
                # vec(A>B) = resid(A in "A is bigger than") - resid(A in "A is")
                r_a_comp = resid_cache[dim][li].get(a)
                r_a_base = resid_cache["baseline"][li].get(a)
                r_b_comp = resid_cache[dim][li].get(b)
                r_b_base = resid_cache["baseline"][li].get(b)
                r_c_comp = resid_cache[dim][li].get(c)
                r_c_base = resid_cache["baseline"][li].get(c)

                if any(v is None for v in [r_a_comp, r_a_base, r_b_comp, r_b_base, r_c_comp, r_c_base]):
                    continue

                # 比较方向向量
                vec_ab = r_a_comp - r_a_base  # "A is bigger than" 的方向
                vec_bc = r_b_comp - r_b_base  # "B is bigger than" 的方向
                vec_ac = r_a_comp - r_a_base  # "A is bigger than" 的方向 (同vec_ab)

                # 传递性测试: vec(A>B) + vec(B>C) vs vec(A>C)
                # 但vec(A>C)不是直接的, 我们用A在比较模板中的残差 - 基线
                # 更好的方式: vec(A>B) 和 vec(B>C) 的和 是否接近 vec(A>C)?

                # 方案: 计算vec_ab + vec_bc 与 vec_ac 的余弦
                # vec_ab ≈ "A比B大"的方向, vec_bc ≈ "B比C大"的方向
                # 如果传递性成立: vec_ab + vec_bc ≈ "A比C大"的方向
                # 但我们没有 "A比C大" 的残差... 

                # 修正方案: 用3个词的比较方向测试线性性
                # vec_A = resid(A, "A is bigger") - baseline(A) → "A大"的方向
                # vec_B = resid(B, "B is bigger") - baseline(B) → "B大"的方向  
                # vec_C = resid(C, "C is bigger") - baseline(C) → "C大"的方向

                # 如果语义空间线性: vec_A > vec_B > vec_C 应该共线
                # 即: vec_A - vec_B 和 vec_B - vec_C 应该共线

                delta_ab = vec_ab - vec_bc  # A的"大"方向 - B的"大"方向
                delta_bc = vec_bc - (r_c_comp - r_c_base)  # B的"大"方向 - C的"大"方向

                norm_ab = np.linalg.norm(delta_ab)
                norm_bc = np.linalg.norm(delta_bc)

                if norm_ab < 1e-10 or norm_bc < 1e-10:
                    continue

                cos_linear = float(np.dot(delta_ab, delta_bc) / (norm_ab * norm_bc))

                linearities.append(cos_linear)

            if linearities:
                mean_cos = float(np.mean(linearities))
                median_cos = float(np.median(linearities))
                pos_ratio = float(np.mean([1 if c > 0 else 0 for c in linearities]))

                log(f"    L{li}: mean_cos={mean_cos:.3f}, median={median_cos:.3f}, "
                    f"pos_ratio={pos_ratio:.2f}, n={len(linearities)}")

                key = f"transitivity_{dim}_L{li}"
                results[key] = {
                    "dimension": dim,
                    "layer": li,
                    "mean_linearity": mean_cos,
                    "median_linearity": median_cos,
                    "pos_ratio": pos_ratio,
                    "n_chains": len(linearities),
                    "all_cosines": [float(x) for x in linearities],
                }

    # ============================================================
    # Step 3: 推理方向的可加性
    # ============================================================
    log("\n--- Step 3: 推理方向可加性 ---")

    # 更直接的测试: "X is bigger" - baseline(X) = "big" direction
    # 如果传递性: big_dir(A) - big_dir(B) + big_dir(B) - big_dir(C) = big_dir(A) - big_dir(C)
    # 即: delta(A,B) + delta(B,C) = delta(A,C) — 恒等式!
    # 这总是成立的, 所以需要更有意义的测试

    # 更好的测试: 推理方向在不同链之间是否一致?
    # 如果 "A>B" 的方向和 "D>E" 的方向共线 → 推理是1维的

    for dim, chains in all_chains.items():
        log(f"\n  === {dim} 方向一致性 ===")

        for li in test_layers:
            directions = []  # "X is bigger" - baseline(X) 的方向

            for a, b, c in chains:
                r_x_comp = resid_cache[dim][li].get(a)
                r_x_base = resid_cache["baseline"][li].get(a)

                if r_x_comp is None or r_x_base is None:
                    continue

                dir_vec = r_x_comp - r_x_base
                norm = np.linalg.norm(dir_vec)
                if norm > 1e-10:
                    directions.append(dir_vec / norm)

            if len(directions) < 3:
                continue

            # 计算方向间的平均余弦
            n_dir = len(directions)
            cos_matrix = np.zeros((n_dir, n_dir))
            for i in range(n_dir):
                for j in range(i + 1, n_dir):
                    cos_val = float(np.dot(directions[i], directions[j]))
                    cos_matrix[i, j] = cos_val
                    cos_matrix[j, i] = cos_val

            # 平均余弦(排除对角线)
            upper_tri = cos_matrix[np.triu_indices(n_dir, k=1)]
            mean_cos = float(np.mean(upper_tri))

            # SVD分析方向的维度
            dir_matrix = np.array(directions)
            n_svd = min(len(directions) - 1, dir_matrix.shape[1] - 1, 20)
            if n_svd > 1:
                _, s_dirs, _ = np.linalg.svd(dir_matrix, full_matrices=False)
                total_var = np.sum(s_dirs[:n_svd] ** 2)
                cumvar = np.cumsum(s_dirs[:n_svd] ** 2) / total_var
                eff_dim = int(np.searchsorted(cumvar, 0.95)) + 1
                first_var = float(s_dirs[0] ** 2 / total_var)
            else:
                eff_dim = 1
                first_var = 1.0

            log(f"    L{li}: mean_cos_between_dirs={mean_cos:.3f}, eff_dim={eff_dim}, "
                f"first_var={first_var:.3f}")

            key = f"direction_consistency_{dim}_L{li}"
            results[key] = {
                "dimension": dim,
                "layer": li,
                "mean_cos_between_directions": mean_cos,
                "eff_dim": eff_dim,
                "first_var_ratio": first_var,
            }

    # ============================================================
    # Step 4: 跨推理维度的一致性
    # ============================================================
    log("\n--- Step 4: 跨推理维度一致性 ---")

    for li in test_layers:
        dim_directions = {}
        for dim in COMPARISON_TEMPLATES:
            dirs = []
            for a, b, c in all_chains[dim]:
                r_comp = resid_cache[dim][li].get(a)
                r_base = resid_cache["baseline"][li].get(a)
                if r_comp is not None and r_base is not None:
                    dv = r_comp - r_base
                    norm = np.linalg.norm(dv)
                    if norm > 1e-10:
                        dirs.append(dv / norm)

            if dirs:
                dim_directions[dim] = np.mean(dirs, axis=0)

        # 计算不同推理维度的方向间余弦
        dim_names = list(dim_directions.keys())
        log(f"\n  L{li}:")
        for i in range(len(dim_names)):
            for j in range(i + 1, len(dim_names)):
                d1 = dim_directions[dim_names[i]]
                d2 = dim_directions[dim_names[j]]
                cos_val = float(np.dot(d1, d2))
                log(f"    {dim_names[i]} vs {dim_names[j]}: cos={cos_val:.3f}")

                key = f"cross_dim_{dim_names[i]}_vs_{dim_names[j]}_L{li}"
                results[key] = {
                    "dim1": dim_names[i],
                    "dim2": dim_names[j],
                    "layer": li,
                    "cos_between_dims": cos_val,
                }

    # ============================================================
    # Step 5: 推理perturb测试
    # ============================================================
    log("\n--- Step 5: 推理方向perturb — 输出是否偏向比较词? ---")

    li = n_layers - 1

    for dim in COMPARISON_TEMPLATES:
        log(f"\n  === {dim} perturb ===")

        # 计算"大/快/重"方向
        dirs = []
        for a, b, c in all_chains[dim]:
            r_comp = resid_cache[dim][li].get(a)
            r_base = resid_cache["baseline"][li].get(a)
            if r_comp is not None and r_base is not None:
                dirs.append(r_comp - r_base)

        if not dirs:
            continue

        mean_dir = np.mean(dirs, axis=0)
        norm = np.linalg.norm(mean_dir)
        if norm < 1e-10:
            continue
        mean_dir = mean_dir / norm

        # 比较词token
        if dim == "size":
            target_words = ["bigger", "larger", "huge", "enormous", "giant", "massive", "smaller", "tiny", "little", "small"]
        elif dim == "weight":
            target_words = ["heavier", "lighter", "massive", "weight", "light", "heavy", "thin", "dense"]
        else:  # speed
            target_words = ["faster", "quicker", "slower", "rapid", "swift", "slow", "quick", "speed"]

        target_tok_ids = []
        for w in target_words:
            tok_ids = tokenizer.encode(" " + w, add_special_tokens=False)
            if tok_ids:
                target_tok_ids.append((w, tok_ids[0]))

        if not target_tok_ids:
            continue

        # 对5个测试词做perturb
        for alpha in [1.0, 2.0, 4.0]:
            big_shifts = []
            small_shifts = []

            for a, b, c in all_chains[dim][:5]:
                prompt = COMPARISON_TEMPLATES[dim].format(a)
                toks = tokenizer(prompt, return_tensors="pt").to(device)
                last_pos = toks.input_ids.shape[1] - 1

                with torch.no_grad():
                    base_out = model(**toks)
                base_logits = base_out.logits[0, -1, :].float().cpu().numpy()

                # Perturb
                perturb_vec = alpha * mean_dir
                intervention_done = [False]

                def make_hook(pv, done, lp):
                    def hook(module, input, output):
                        if done[0]:
                            return output
                        done[0] = True
                        if isinstance(output, tuple):
                            out = output[0]
                        else:
                            out = output
                        pt = torch.tensor(pv, dtype=out.dtype, device=device)
                        new_out = out.clone()
                        new_out[0, lp, :] += pt
                        if isinstance(output, tuple):
                            return (new_out,) + output[1:]
                        return new_out
                    return hook

                h_handle = layers_list[li].register_forward_hook(
                    make_hook(perturb_vec, intervention_done, last_pos))

                with torch.no_grad():
                    pert_out = model(**toks)
                pert_logits = pert_out.logits[0, -1, :].float().cpu().numpy()
                h_handle.remove()

                delta_logits = pert_logits - base_logits

                # 分别统计"正向"和"反向"词的shift
                positive_words = target_words[:len(target_words)//2]
                negative_words = target_words[len(target_words)//2:]

                pos_shift = np.mean([delta_logits[tid] for w, tid in target_tok_ids
                                     if w in positive_words and tid < len(delta_logits)])
                neg_shift = np.mean([delta_logits[tid] for w, tid in target_tok_ids
                                     if w in negative_words and tid < len(delta_logits)])

                big_shifts.append(pos_shift)
                small_shifts.append(neg_shift)

            mean_pos = float(np.mean(big_shifts))
            mean_neg = float(np.mean(small_shifts))
            asymmetry = mean_pos - mean_neg

            log(f"    alpha={alpha}: pos_shift={mean_pos:.3f}, neg_shift={mean_neg:.3f}, "
                f"asymmetry={asymmetry:.3f}")

            key = f"perturb_{dim}_L{li}_a{alpha}"
            results[key] = {
                "dimension": dim,
                "layer": li,
                "alpha": alpha,
                "positive_shift": mean_pos,
                "negative_shift": mean_neg,
                "asymmetry": float(asymmetry),
            }

    # 保存结果
    out_file = TEMP_DIR / f"ccxvi_transitive_reasoning_{model_name}.json"
    with open(out_file, "w") as f:
        json.dump({
            "model": model_name,
            "d_model": d_model,
            "n_layers": n_layers,
            "test_layers": test_layers,
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
