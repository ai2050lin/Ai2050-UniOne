"""
CCXIV(314): 关系perturb控制实验 — Habitat SVD模式能否控制输出?
======================================================================
CCXIII发现Habitat是最强语义关系(28-375x vs category), 但只是观察性的。
本实验: 沿Habitat SVD模式perturb → 输出是否偏向特定栖息地?
与Category perturb对比 → 关系控制力 vs 分类控制力?

核心问题:
  1. Habitat SVD perturb能否让输出偏向特定栖息地?
  2. Habitat perturb效果是否远>Category perturb?
  3. 关系控制的语义特异性: perturb只影响栖息地, 不影响类别?

设计:
  - 20个词, 3种栖息地(land/ocean/sky)
  - 在最后层perturb Habitat SVD模式
  - 测量logit空间中栖息地词的shift量
  - 与Category perturb对比

用法:
  python ccxiv_relation_perturb.py --model qwen3
  python ccxiv_relation_perturb.py --model glm4
  python ccxiv_relation_perturb.py --model deepseek7b
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

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, get_W_U,
    get_layer_weights, MODEL_CONFIGS,
)

TEMP_DIR = Path("tests/glm5_temp")
LOG_FILE = TEMP_DIR / "ccxiv_relation_perturb_log.txt"

# 词汇和属性
WORDS_BY_HABITAT = {
    "land": ["dog", "cat", "lion", "tiger", "horse", "cow", "sheep", "rabbit", "fox", "deer"],
    "ocean": ["whale", "shark", "dolphin", "octopus", "salmon", "turtle", "crab", "seal", "squid", "lobster"],
    "sky": ["eagle", "hawk", "owl", "parrot", "crow", "sparrow", "swallow", "falcon", "pigeon", "robin"],
}

CATEGORY_OF = {
    "dog":"animal", "cat":"animal", "lion":"animal", "tiger":"animal", "horse":"animal",
    "cow":"animal", "sheep":"animal", "rabbit":"animal", "fox":"animal", "deer":"animal",
    "whale":"animal", "shark":"animal", "dolphin":"animal", "octopus":"animal", "salmon":"animal",
    "turtle":"animal", "crab":"animal", "seal":"animal", "squid":"animal", "lobster":"animal",
    "eagle":"animal", "hawk":"animal", "owl":"animal", "parrot":"animal", "crow":"animal",
    "sparrow":"animal", "swallow":"animal", "falcon":"animal", "pigeon":"animal", "robin":"animal",
}

TEMPLATES = {
    "habitat": "The {} lives in the",
    "category": "The {} is a",
}

# 栖息地/类别相关的logit目标词
HABITAT_TOKENS = {
    "land": ["land", "ground", "earth", "field", "forest", "plains", "jungle", "savanna"],
    "ocean": ["ocean", "sea", "water", "river", "lake", "deep", "marine", "coastal"],
    "sky": ["sky", "air", "trees", "mountains", "heights", "clouds", "nests", "branches"],
}

CATEGORY_TOKENS = {
    "animal": ["animal", "creature", "beast", "mammal", "bird", "fish", "species", "pet"],
}


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def get_token_ids(tokenizer, words):
    """获取词的token id列表(跳过特殊token)"""
    ids = []
    for w in words:
        tok_ids = tokenizer.encode(" " + w, add_special_tokens=False)
        if tok_ids:
            ids.append((w, tok_ids[0]))
    return ids


def run_experiment(model_name):
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)

    n_layers = info.n_layers
    d_model = info.d_model
    W_U = get_W_U(model)  # [vocab, d_model]

    # 测试层
    test_layers = sorted(set([
        n_layers // 4, n_layers // 2, 3 * n_layers // 4,
        n_layers - 2, n_layers - 1
    ]))

    all_words = []
    word_habitat = {}
    for hab, ws in WORDS_BY_HABITAT.items():
        all_words.extend(ws)
        for w in ws:
            word_habitat[w] = hab

    log(f"\n{'='*70}")
    log(f"CCXIV(314): 关系perturb控制实验 - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}")
    log(f"  测试层: {test_layers}")
    log(f"  总词数: {len(all_words)} (3栖息地, 每栖息地10词)")
    log(f"{'='*70}")

    results = {}

    # ============================================================
    # Step 1: 收集各关系模板的残差流
    # ============================================================
    log("\n--- Step 1: 收集残差流 ---")

    # word_outputs[rel_type][li][word] = residual vector
    word_outputs = {rel: {li: {} for li in test_layers} for rel in TEMPLATES}

    for rel_type, template in TEMPLATES.items():
        log(f"\n  关系: {rel_type} (模板: '{template}')")
        for wi, word in enumerate(all_words):
            prompt = template.format(word)
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
                resid = captured.get(f"L{li}", None)
                if resid is not None:
                    word_outputs[rel_type][li][word] = resid

        log(f"    收集完成")

    # ============================================================
    # Step 2: 计算Habitat/Category的SVD模式
    # ============================================================
    log("\n--- Step 2: 计算SVD模式 ---")

    svd_modes = {}  # svd_modes[rel_type][li] = (Vt, best_mode_idx, f_scores)

    for rel_type in TEMPLATES:
        svd_modes[rel_type] = {}
        for li in test_layers:
            vecs = []
            valid_words = []
            for w in all_words:
                v = word_outputs[rel_type][li].get(w)
                if v is not None:
                    vecs.append(v)
                    valid_words.append(w)

            if len(vecs) < 4:
                continue

            vecs = np.array(vecs)
            mean_vec = vecs.mean(axis=0)
            centered = vecs - mean_vec

            n_svd = min(30, centered.shape[0] - 1, centered.shape[1] - 1)
            if n_svd < 3:
                continue

            U, s, Vt = np.linalg.svd(centered, full_matrices=False)
            Vt = Vt[:n_svd, :]

            # 找最有区分力的模式
            projections = centered @ Vt.T
            f_scores = []

            # 根据关系类型选择属性
            if rel_type == "habitat":
                attr_of = word_habitat
            else:  # category
                attr_of = CATEGORY_OF

            attr_values = set(attr_of.get(w, "") for w in valid_words)
            attr_values.discard("")

            for mi in range(n_svd):
                proj_mi = projections[:, mi]
                groups = []
                for av in attr_values:
                    av_idx = [i for i, w in enumerate(valid_words) if attr_of.get(w) == av]
                    if av_idx:
                        groups.append(proj_mi[av_idx])
                if len(groups) >= 2:
                    f_stat, _ = stats.f_oneway(*groups)
                    f_scores.append(f_stat if not np.isnan(f_stat) else 0)
                else:
                    f_scores.append(0)

            best_mode = int(np.argmax(f_scores))
            svd_modes[rel_type][li] = (Vt, best_mode, f_scores)

            log(f"  {rel_type} L{li}: best_mode={best_mode}, F={f_scores[best_mode]:.1f}, "
                f"top3_F={sorted(f_scores, reverse=True)[:3]}")

    # ============================================================
    # Step 3: Habitat perturb → 输出logit分析
    # ============================================================
    log("\n--- Step 3: Habitat perturb → logit分析 ---")

    # 获取栖息地/类别token的id
    habitat_tok_ids = {}
    for hab, words in HABITAT_TOKENS.items():
        habitat_tok_ids[hab] = get_token_ids(tokenizer, words)

    category_tok_ids = {}
    for cat, words in CATEGORY_TOKENS.items():
        category_tok_ids[cat] = get_token_ids(tokenizer, words)

    # 对最后2层做perturb测试
    perturb_layers = [n_layers - 2, n_layers - 1]
    alphas = [0.5, 1.0, 2.0]

    for li in perturb_layers:
        if "habitat" not in svd_modes or li not in svd_modes["habitat"]:
            continue

        Vt_hab, best_mode_hab, _ = svd_modes["habitat"][li]
        mode_dir = Vt_hab[best_mode_hab]

        log(f"\n  === L{li} Habitat perturb (best_mode={best_mode_hab}) ===")

        for alpha in alphas:
            log(f"\n  alpha={alpha}:")

            # 对每个栖息地词做perturb
            # 统计: perturb后栖息地token logit增量
            hab_logit_shifts = defaultdict(list)  # {habitat: [shift_values]}
            cat_logit_shifts = defaultdict(list)   # {category: [shift_values]}

            for word in all_words:
                true_hab = word_habitat[word]
                prompt = TEMPLATES["habitat"].format(word)
                toks = tokenizer(prompt, return_tensors="pt").to(device)
                last_pos = toks.input_ids.shape[1] - 1

                # Baseline logits
                with torch.no_grad():
                    base_out = model(**toks)
                base_logits = base_out.logits[0, -1, :].float().cpu().numpy()

                # Perturb: +alpha 沿habitat mode
                perturb_vec = alpha * mode_dir
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

                # 测量栖息地token的logit shift
                for hab, tok_list in habitat_tok_ids.items():
                    hab_shift = float(np.mean([delta_logits[tid] for _, tid in tok_list if tid < len(delta_logits)]))
                    hab_logit_shifts[hab].append(hab_shift)

                # 测量类别token的logit shift (特异性测试)
                for cat, tok_list in category_tok_ids.items():
                    cat_shift = float(np.mean([delta_logits[tid] for _, tid in tok_list if tid < len(delta_logits)]))
                    cat_logit_shifts[cat].append(cat_shift)

            # 汇总
            log(f"    栖息地token logit shift:")
            for hab in ["land", "ocean", "sky"]:
                shifts = hab_logit_shifts.get(hab, [0])
                mean_s = np.mean(shifts) if shifts else 0
                log(f"      {hab}: mean_shift={mean_s:.3f}")

            log(f"    类别token logit shift:")
            for cat, shifts in cat_logit_shifts.items():
                mean_s = np.mean(shifts) if shifts else 0
                log(f"      {cat}: mean_shift={mean_s:.3f}")

            # 关系特异性: habitat_shift / category_shift
            hab_shift_abs = np.mean([abs(s) for ss in hab_logit_shifts.values() for s in ss])
            cat_shift_abs = np.mean([abs(s) for ss in cat_logit_shifts.values() for s in ss])
            specificity = hab_shift_abs / max(cat_shift_abs, 1e-6)

            log(f"    ★ 关系特异性(hab_shift/cat_shift)={specificity:.2f}x")

            key = f"habitat_perturb_L{li}_a{alpha}"
            results[key] = {
                "relation": "habitat",
                "layer": li,
                "alpha": alpha,
                "habitat_shifts": {h: float(np.mean(v)) for h, v in hab_logit_shifts.items()},
                "category_shifts": {c: float(np.mean(v)) for c, v in cat_logit_shifts.items()},
                "specificity": float(specificity),
            }

    # ============================================================
    # Step 4: Category perturb → 对比
    # ============================================================
    log("\n--- Step 4: Category perturb → 对比 ---")

    for li in perturb_layers:
        if "category" not in svd_modes or li not in svd_modes["category"]:
            continue

        Vt_cat, best_mode_cat, _ = svd_modes["category"][li]
        mode_dir = Vt_cat[best_mode_cat]

        log(f"\n  === L{li} Category perturb (best_mode={best_mode_cat}) ===")

        for alpha in alphas:
            log(f"\n  alpha={alpha}:")

            hab_logit_shifts = defaultdict(list)
            cat_logit_shifts = defaultdict(list)

            for word in all_words:
                prompt = TEMPLATES["category"].format(word)
                toks = tokenizer(prompt, return_tensors="pt").to(device)
                last_pos = toks.input_ids.shape[1] - 1

                with torch.no_grad():
                    base_out = model(**toks)
                base_logits = base_out.logits[0, -1, :].float().cpu().numpy()

                perturb_vec = alpha * mode_dir
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

                for hab, tok_list in habitat_tok_ids.items():
                    hab_shift = float(np.mean([delta_logits[tid] for _, tid in tok_list if tid < len(delta_logits)]))
                    hab_logit_shifts[hab].append(hab_shift)

                for cat, tok_list in category_tok_ids.items():
                    cat_shift = float(np.mean([delta_logits[tid] for _, tid in tok_list if tid < len(delta_logits)]))
                    cat_logit_shifts[cat].append(cat_shift)

            log(f"    栖息地token logit shift:")
            for hab in ["land", "ocean", "sky"]:
                shifts = hab_logit_shifts.get(hab, [0])
                mean_s = np.mean(shifts) if shifts else 0
                log(f"      {hab}: mean_shift={mean_s:.3f}")

            log(f"    类别token logit shift:")
            for cat, shifts in cat_logit_shifts.items():
                mean_s = np.mean(shifts) if shifts else 0
                log(f"      {cat}: mean_shift={mean_s:.3f}")

            cat_shift_abs = np.mean([abs(s) for ss in cat_logit_shifts.values() for s in ss])
            hab_shift_abs = np.mean([abs(s) for ss in hab_logit_shifts.values() for s in ss])
            specificity = cat_shift_abs / max(hab_shift_abs, 1e-6)

            log(f"    ★ 分类特异性(cat_shift/hab_shift)={specificity:.2f}x")

            key = f"category_perturb_L{li}_a{alpha}"
            results[key] = {
                "relation": "category",
                "layer": li,
                "alpha": alpha,
                "habitat_shifts": {h: float(np.mean(v)) for h, v in hab_logit_shifts.items()},
                "category_shifts": {c: float(np.mean(v)) for c, v in cat_logit_shifts.items()},
                "specificity": float(specificity),
            }

    # ============================================================
    # Step 5: 定向perturb — 让输出偏向特定栖息地
    # ============================================================
    log("\n--- Step 5: 定向perturb — 输出偏向目标栖息地 ---")

    # 对每个栖息地, 计算该栖息地词的均值 - 全体均值
    li = n_layers - 1
    if "habitat" in svd_modes and li in svd_modes["habitat"]:
        Vt_hab, _, f_scores_hab = svd_modes["habitat"][li]
        n_top = min(5, len(f_scores_hab))

        # 找区分各栖息地的模式
        # 对top-5模式, 计算3个栖息地的组均值
        projections_all = {}
        valid_words_all = []
        for w in all_words:
            v = word_outputs["habitat"][li].get(w)
            if v is not None:
                projections_all[w] = (v - np.mean([word_outputs["habitat"][li].get(w2, np.zeros(d_model))
                                                     for w2 in all_words], axis=0)) @ Vt_hab[:n_top].T
                valid_words_all.append(w)

        # 计算每个模式对各栖息地的区分方向
        hab_directions = {}  # {habitat: direction_in_d_model}
        for hab in ["land", "ocean", "sky"]:
            hab_words = WORDS_BY_HABITAT[hab]
            other_words = [w for h, ws in WORDS_BY_HABITAT.items() if h != hab for w in ws]

            hab_vecs = [word_outputs["habitat"][li].get(w) for w in hab_words
                        if w in word_outputs["habitat"][li]]
            other_vecs = [word_outputs["habitat"][li].get(w) for w in other_words
                          if w in word_outputs["habitat"][li]]

            if hab_vecs and other_vecs:
                hab_mean = np.mean(hab_vecs, axis=0)
                other_mean = np.mean(other_vecs, axis=0)
                direction = hab_mean - other_mean
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction = direction / norm
                hab_directions[hab] = direction

        log(f"  计算出 {len(hab_directions)} 个栖息地方向")

        # 对每个栖息地方向做perturb
        for target_hab, direction in hab_directions.items():
            log(f"\n  → 目标栖息地: {target_hab}")

            for alpha in [1.0, 2.0, 4.0]:
                success_count = 0
                total_count = 0

                for word in all_words[:15]:  # 15个测试词
                    prompt = TEMPLATES["habitat"].format(word)
                    toks = tokenizer(prompt, return_tensors="pt").to(device)
                    last_pos = toks.input_ids.shape[1] - 1

                    # Baseline
                    with torch.no_grad():
                        base_out = model(**toks)
                    base_logits = base_out.logits[0, -1, :].float().cpu().numpy()

                    # Perturb
                    perturb_vec = alpha * direction
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

                    # 检查栖息地token的logit变化
                    target_shift = 0
                    other_shift = 0
                    for hab, tok_list in habitat_tok_ids.items():
                        shift = np.mean([pert_logits[tid] - base_logits[tid]
                                         for _, tid in tok_list if tid < len(base_logits)])
                        if hab == target_hab:
                            target_shift = shift
                        else:
                            other_shift += shift

                    other_shift /= max(len(habitat_tok_ids) - 1, 1)

                    total_count += 1
                    if target_shift > other_shift + 0.1:
                        success_count += 1

                success_rate = success_count / max(total_count, 1)
                log(f"    alpha={alpha}: 定向成功率={success_rate:.2%} ({success_count}/{total_count})")

                key = f"directed_{target_hab}_L{li}_a{alpha}"
                results[key] = {
                    "target_habitat": target_hab,
                    "layer": li,
                    "alpha": alpha,
                    "success_rate": float(success_rate),
                    "success_count": success_count,
                    "total_count": total_count,
                }

    # 保存结果
    out_file = TEMP_DIR / f"ccxiv_relation_perturb_{model_name}.json"
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
