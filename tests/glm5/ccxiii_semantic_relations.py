"""
CCXIII(313): 语义关系测试 — 关系是否也有低维控制通道?
======================================================
之前所有实验只测分类(词属于哪个类别), 但语言编码的核心是关系和组合。

本实验测试3种语义关系:
  1. 属性关系: "The X is [color/size/material]" — X的颜色/大小/材质
  2. 层级关系: "The X is a kind of [animal/food/tool]" — 上位词
  3. 对比关系: "X is [bigger/smaller] than Y" — 比较

核心问题:
  - 关系表示是否也存在于d_model空间?
  - 关系表示是否也是低维的?
  - 关系表示和分类表示是否共享子空间?

设计:
  对比三组提示词:
  A. 分类: "The dog is" → 期待animal类token
  B. 属性: "The dog is [colored]" → 期待颜色token
  C. 层级: "The dog is a kind of" → 期待animal上位词

  测量:
  1. 不同关系类型的SVD模式结构
  2. 关系SVD模式vs分类SVD模式的重叠度
  3. 沿关系SVD模式perturb → 输出是否按关系方向偏移?

用法:
  python ccxiii_semantic_relations.py --model qwen3
  python ccxiii_semantic_relations.py --model glm4
  python ccxiii_semantic_relations.py --model deepseek7b
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
LOG_FILE = TEMP_DIR / "ccxiii_semantic_relations_log.txt"

# 关系测试设计
# 每个词有多种关系, 期待不同的输出
RELATION_ITEMS = {
    # 词: {关系类型: 期待输出类别/方向}
    "dog":     {"category": "animal", "size": "small", "habitat": "land", "color": "brown"},
    "cat":     {"category": "animal", "size": "small", "habitat": "land", "color": "orange"},
    "whale":   {"category": "animal", "size": "large", "habitat": "ocean", "color": "blue"},
    "eagle":   {"category": "animal", "size": "medium", "habitat": "sky", "color": "brown"},
    "shark":   {"category": "animal", "size": "large", "habitat": "ocean", "color": "gray"},
    "apple":   {"category": "food", "size": "small", "color": "red", "taste": "sweet"},
    "bread":   {"category": "food", "size": "medium", "color": "brown", "taste": "neutral"},
    "pizza":   {"category": "food", "size": "medium", "color": "yellow", "taste": "savory"},
    "mango":   {"category": "food", "size": "small", "color": "yellow", "taste": "sweet"},
    "steak":   {"category": "food", "size": "large", "color": "red", "taste": "savory"},
    "hammer":  {"category": "tool", "size": "medium", "material": "metal", "use": "build"},
    "knife":   {"category": "tool", "size": "small", "material": "metal", "use": "cut"},
    "drill":   {"category": "tool", "size": "medium", "material": "metal", "use": "build"},
    "chisel":  {"category": "tool", "size": "small", "material": "metal", "use": "carve"},
    "ruler":   {"category": "tool", "size": "small", "material": "plastic", "use": "measure"},
    "mountain":{"category": "nature", "size": "large", "color": "green", "terrain": "rocky"},
    "river":   {"category": "nature", "size": "large", "color": "blue", "terrain": "watery"},
    "desert":  {"category": "nature", "size": "large", "color": "yellow", "terrain": "sandy"},
    "glacier": {"category": "nature", "size": "large", "color": "white", "terrain": "icy"},
    "forest":  {"category": "nature", "size": "large", "color": "green", "terrain": "wooded"},
}

# 不同关系模板
TEMPLATES = {
    "category": "The {} is a",        # → animal, food, tool, nature
    "size":     "The {} is very",      # → big, small, large, tiny
    "color":    "The {} is colored",   # → red, blue, green, yellow
    "habitat":  "The {} lives in",     # → water, land, sky, forest
    "material": "The {} is made of",   # → metal, wood, plastic, stone
}


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

    # 测试层: 最后层 + 中层
    test_layers = sorted(set([n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]))

    log(f"\n{'='*70}")
    log(f"CCXIII(313): 语义关系测试 - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}")
    log(f"  测试层: {test_layers}")
    log(f"  关系类型: {list(TEMPLATES.keys())}")
    log(f"{'='*70}")

    # Step 1: 对每个关系模板, 收集残差流输出
    log("\n--- Step 1: 收集各关系模板的残差流 ---")

    words = list(RELATION_ITEMS.keys())
    relation_types = list(TEMPLATES.keys())

    # word_relation_outputs[relation_type][li][word] = residual vector
    word_relation_outputs = {rel: {li: {} for li in test_layers} for rel in relation_types}

    for rel_type, template in TEMPLATES.items():
        log(f"\n  关系类型: {rel_type} (模板: '{template}')")

        for wi, word in enumerate(words):
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
                layer = layers_list[li]
                hooks.append(layer.register_forward_hook(make_capture_hook(f"L{li}")))

            with torch.no_grad():
                _ = model(**toks)

            for h in hooks:
                h.remove()

            for li in test_layers:
                resid = captured.get(f"L{li}", None)
                if resid is not None:
                    word_relation_outputs[rel_type][li][word] = resid

        log(f"    收集完成: {len([w for w in words if w in word_relation_outputs[rel_type][test_layers[-1]]])} 词")

    # Step 2: 分析各关系类型的语义区分力
    log("\n--- Step 2: 各关系类型的语义区分力 ---")

    results = {}

    for rel_type in relation_types:
        log(f"\n  === 关系类型: {rel_type} ===")

        for li in test_layers:
            # 收集该关系类型的向量
            vecs = []
            valid_words = []
            for w in words:
                v = word_relation_outputs[rel_type][li].get(w)
                if v is not None:
                    vecs.append(v)
                    valid_words.append(w)

            if len(vecs) < 4:
                continue

            vecs = np.array(vecs)

            # SVD
            mean_vec = vecs.mean(axis=0)
            centered = vecs - mean_vec
            n_svd = min(30, centered.shape[0] - 1, centered.shape[1] - 1)
            if n_svd < 3:
                continue

            U, s, Vt = np.linalg.svd(centered, full_matrices=False)
            s = s[:n_svd]
            Vt = Vt[:n_svd, :]

            # 计算每个SVD模式对该关系属性的区分力
            # 例如: "size"关系 → 按size属性(大/中/小)分组, 看SVD模式能否区分
            rel_attr = rel_type
            attr_values = set()
            for w in valid_words:
                if rel_attr in RELATION_ITEMS[w]:
                    attr_values.add(RELATION_ITEMS[w][rel_attr])

            if len(attr_values) < 2:
                continue

            # 按属性值分组
            projections = centered @ Vt.T
            mode_f_scores = []

            for mi in range(n_svd):
                proj_mi = projections[:, mi]
                groups = []
                for av in attr_values:
                    av_idx = [i for i, w in enumerate(valid_words)
                             if rel_attr in RELATION_ITEMS[w] and RELATION_ITEMS[w][rel_attr] == av]
                    if av_idx:
                        groups.append(proj_mi[av_idx])

                if len(groups) >= 2:
                    f_stat, _ = stats.f_oneway(*groups)
                    mode_f_scores.append(f_stat if not np.isnan(f_stat) else 0)
                else:
                    mode_f_scores.append(0)

            # 也计算分类区分力(按category属性)
            cat_f_scores = []
            categories = set(RELATION_ITEMS[w]["category"] for w in valid_words)
            for mi in range(n_svd):
                proj_mi = projections[:, mi]
                groups = []
                for cat in categories:
                    cat_idx = [i for i, w in enumerate(valid_words)
                              if RELATION_ITEMS[w]["category"] == cat]
                    if cat_idx:
                        groups.append(proj_mi[cat_idx])
                if len(groups) >= 2:
                    f_stat, _ = stats.f_oneway(*groups)
                    cat_f_scores.append(f_stat if not np.isnan(f_stat) else 0)
                else:
                    cat_f_scores.append(0)

            # 语义区分力: 关系属性 vs 分类属性
            top_rel_f = max(mode_f_scores)
            top_cat_f = max(cat_f_scores)
            top3_rel_f = sorted(mode_f_scores, reverse=True)[:3]
            top3_cat_f = sorted(cat_f_scores, reverse=True)[:3]

            # 关系区分力/分类区分力 比值
            rel_cat_ratio = top_rel_f / max(top_cat_f, 1e-6)

            key = f"{rel_type}_L{li}"
            results[key] = {
                "relation": rel_type,
                "layer": li,
                "top_rel_f": float(top_rel_f),
                "top_cat_f": float(top_cat_f),
                "top3_rel_f": [float(x) for x in top3_rel_f],
                "top3_cat_f": [float(x) for x in top3_cat_f],
                "rel_cat_ratio": float(rel_cat_ratio),
                "n_attr_values": len(attr_values),
            }

            log(f"    L{li}: top_rel_F={top_rel_f:.1f}, top_cat_F={top_cat_f:.1f}, "
                f"ratio={rel_cat_ratio:.2f}, n_attrs={len(attr_values)}")

    # Step 3: 关系SVD模式与分类SVD模式的重叠度
    log("\n--- Step 3: 关系SVD模式与分类SVD模式的重叠度 ---")

    for li in test_layers:
        log(f"\n  L{li}:")

        # 获取category关系的SVD基
        cat_key = f"category_L{li}"
        if cat_key not in results:
            continue

        # 收集category模板的向量
        cat_vecs = []
        cat_words_valid = []
        for w in words:
            v = word_relation_outputs["category"][li].get(w)
            if v is not None:
                cat_vecs.append(v)
                cat_words_valid.append(w)

        if len(cat_vecs) < 4:
            continue

        cat_vecs = np.array(cat_vecs)
        cat_mean = cat_vecs.mean(axis=0)
        cat_centered = cat_vecs - cat_mean

        n_svd = min(20, cat_centered.shape[0] - 1, cat_centered.shape[1] - 1)
        _, s_cat, Vt_cat = np.linalg.svd(cat_centered, full_matrices=False)
        Vt_cat = Vt_cat[:n_svd, :]

        # 对其他关系类型计算重叠度
        for rel_type in ["size", "color", "habitat", "material"]:
            if rel_type not in word_relation_outputs:
                continue

            rel_vecs = []
            rel_words_valid = []
            for w in words:
                v = word_relation_outputs[rel_type][li].get(w)
                if v is not None:
                    rel_vecs.append(v)
                    rel_words_valid.append(w)

            if len(rel_vecs) < 4:
                continue

            rel_vecs = np.array(rel_vecs)
            rel_mean = rel_vecs.mean(axis=0)
            rel_centered = rel_vecs - rel_mean

            _, s_rel, Vt_rel = np.linalg.svd(rel_centered, full_matrices=False)
            Vt_rel = Vt_rel[:n_svd, :]

            # 子空间重叠度: ||Vt_cat @ Vt_rel^T||_F / sqrt(k)
            # 即两个子空间之间的Grassmann距离
            overlap_matrix = Vt_cat[:10] @ Vt_rel[:10].T  # [10, 10]
            # SVD of overlap matrix → principal angles
            _, cos_angles, _ = np.linalg.svd(overlap_matrix)
            # cos_angles = principal angle cosines
            mean_cos = float(np.mean(np.abs(cos_angles)))
            max_cos = float(np.max(np.abs(cos_angles)))

            log(f"    {rel_type} vs category: mean_subspace_cos={mean_cos:.3f}, "
                f"max_cos={max_cos:.3f}")

            key = f"overlap_{rel_type}_cat_L{li}"
            results[key] = {
                "relation": rel_type,
                "vs": "category",
                "layer": li,
                "mean_subspace_cos": mean_cos,
                "max_subspace_cos": max_cos,
            }

    # Step 4: 关系perturb测试 — 沿size SVD模式perturb, 输出是否偏向大/小?
    log("\n--- Step 4: 关系perturb测试 ---")

    # 只测试最后一层的size关系
    li = n_layers - 1
    rel_type = "size"

    size_vecs = []
    size_words_valid = []
    size_labels = []
    for w in words:
        v = word_relation_outputs[rel_type][li].get(w)
        if v is not None and "size" in RELATION_ITEMS[w]:
            size_vecs.append(v)
            size_words_valid.append(w)
            size_labels.append(RELATION_ITEMS[w]["size"])

    if len(size_vecs) >= 4:
        size_vecs = np.array(size_vecs)
        size_mean = size_vecs.mean(axis=0)
        size_centered = size_vecs - size_mean

        n_svd = min(20, size_centered.shape[0] - 1, size_centered.shape[1] - 1)
        _, s_size, Vt_size = np.linalg.svd(size_centered, full_matrices=False)
        Vt_size = Vt_size[:n_svd, :]

        # 找最有size区分力的SVD模式
        projections = size_centered @ Vt_size.T
        mode_f = []
        for mi in range(n_svd):
            proj_mi = projections[:, mi]
            groups = []
            for sv in set(size_labels):
                sv_idx = [i for i, l in enumerate(size_labels) if l == sv]
                groups.append(proj_mi[sv_idx])
            if len(groups) >= 2:
                f_stat, _ = stats.f_oneway(*groups)
                mode_f.append(f_stat if not np.isnan(f_stat) else 0)
            else:
                mode_f.append(0)

        best_mode = int(np.argmax(mode_f))
        log(f"  L{li} size: best_mode={best_mode}, F={mode_f[best_mode]:.1f}")

        # 沿best_mode方向perturb, 看输出偏向哪个size
        ALPHA = 1.0
        mode_dir = Vt_size[best_mode]

        # +alpha 和 -alpha 两个方向
        for direction_label, sign in [("positive", 1.0), ("negative", -1.0)]:
            size_shifts = Counter()

            for wi, word in enumerate(size_words_valid):
                v = word_relation_outputs[rel_type][li].get(word)
                if v is None:
                    continue

                output_norm = np.linalg.norm(v)
                if output_norm < 1e-10:
                    continue

                perturb_vec = sign * ALPHA * output_norm * mode_dir

                # Baseline
                prompt = TEMPLATES["size"].format(word)
                toks = tokenizer(prompt, return_tensors="pt").to(device)
                last_pos = toks.input_ids.shape[1] - 1

                with torch.no_grad():
                    base_out = model(**toks)
                base_logits = base_out.logits[0, -1, :].float().cpu().numpy()
                base_top1 = int(np.argmax(base_logits))

                # Perturbed
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

                hook_handle = layers_list[li].register_forward_hook(
                    make_hook(perturb_vec, intervention_done, last_pos))

                with torch.no_grad():
                    pert_out = model(**toks)
                pert_logits = pert_out.logits[0, -1, :].float().cpu().numpy()
                pert_top1 = int(np.argmax(pert_logits))

                hook_handle.remove()

                # 检查输出词的size属性
                base_word = tokenizer.decode([base_top1]).strip().lower()
                pert_word = tokenizer.decode([pert_top1]).strip().lower()

                # 简单检测: 输出是否包含大/小相关词
                big_words = {"large", "big", "huge", "giant", "massive", "enormous", "tall", "long", "wide"}
                small_words = {"small", "tiny", "little", "short", "narrow", "miniature", "petite"}

                base_size = "big" if any(bw in base_word for bw in big_words) else \
                           "small" if any(sw in base_word for sw in small_words) else "other"
                pert_size = "big" if any(bw in pert_word for bw in big_words) else \
                           "small" if any(sw in pert_word for sw in small_words) else "other"

                if pert_size != "other":
                    size_shifts[pert_size] += 1

            log(f"    {direction_label}方向: size_shifts={dict(size_shifts)}")

    # 保存结果
    out_file = TEMP_DIR / f"ccxiii_semantic_relations_{model_name}.json"
    with open(out_file, "w") as f:
        json.dump({
            "model": model_name,
            "d_model": d_model,
            "n_layers": n_layers,
            "test_layers": test_layers,
            "relation_types": relation_types,
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
