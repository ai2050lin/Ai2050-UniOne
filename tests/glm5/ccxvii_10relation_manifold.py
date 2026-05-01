"""
CCXVII(317): 10+关系语义流形验证 — 3维流形是否普适?
======================================================================
CCXV发现4种关系→3维流形, 但可能是4类的artifact(4类只需3维分离)。
本实验: 扩展到10种关系, 验证3维流形是否普适。

核心问题:
  1. 10种关系仍只需3维? → 3维流形是语言的本质结构
  2. 10种关系需要>3维? → 3维只是4类的artifact
  3. 有效维度是否随关系数线性增长? → 语义流形的拓扑性质

设计:
  - 10种关系模板, 40个词(4类别x10词)
  - 计算全关系联合子空间的有效维度
  - ANOVA F检验每个SVD模式区分关系的能力
  - Grassmann距离检验子空间正交性

用法:
  python ccxvii_10relation_manifold.py --model qwen3
  python ccxvii_10relation_manifold.py --model glm4
  python ccxvii_10relation_manifold.py --model deepseek7b
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
    MODEL_CONFIGS,
)

TEMP_DIR = Path("tests/glm5_temp")
LOG_FILE = TEMP_DIR / "ccxvii_10relation_manifold_log.txt"

# 10种关系模板
RELATION_TEMPLATES = {
    "habitat":     "The {} lives in the",
    "category":    "The {} is a",
    "material":    "The {} is made of",
    "size":        "The {} is very",
    "color":       "The {} is colored",
    "shape":       "The {} has a",
    "function":    "The {} is used for",
    "taste":       "The {} tastes",
    "temperature": "The {} feels",
    "weight":      "The {} weighs",
}

# 40个词, 4类别x10词
WORDS = {
    "animal_land": ["dog", "cat", "lion", "tiger", "horse", "cow", "fox", "deer", "sheep", "rabbit"],
    "animal_ocean": ["whale", "shark", "dolphin", "octopus", "salmon", "seal", "crab", "squid", "turtle", "lobster"],
    "food_fruit": ["apple", "banana", "mango", "cherry", "peach", "grape", "lemon", "orange", "melon", "berry"],
    "tool_metal": ["hammer", "knife", "drill", "chisel", "wrench", "saw", "axe", "pliers", "nail", "bolt"],
}

WORD_ATTRS = {}
for ws in WORDS.values():
    for w in ws:
        if w in ["dog","cat","lion","tiger","horse","cow","fox","deer","sheep","rabbit",
                  "whale","shark","dolphin","octopus","salmon","seal","crab","squid","turtle","lobster"]:
            WORD_ATTRS[w] = "animal"
        elif w in ["apple","banana","mango","cherry","peach","grape","lemon","orange","melon","berry"]:
            WORD_ATTRS[w] = "food"
        else:
            WORD_ATTRS[w] = "tool"


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def compute_principal_angles(V1, V2, k=10):
    """计算两个子空间之间的主角(principal angles)"""
    k1 = min(k, V1.shape[0], V2.shape[0])
    overlap = V1[:k1] @ V2[:k1].T
    _, cos_vals, _ = np.linalg.svd(overlap)
    return cos_vals


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

    all_words = []
    for ws in WORDS.values():
        all_words.extend(ws)
    all_words = list(set(all_words))

    log(f"\n{'='*70}")
    log(f"CCXVII(317): 10+关系语义流形验证 - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}")
    log(f"  测试层: {test_layers}")
    log(f"  总词数: {len(all_words)}")
    log(f"  关系数: {len(RELATION_TEMPLATES)}")
    log(f"{'='*70}")

    results = {}

    # Step 1: 收集残差流
    log("\n--- Step 1: 收集残差流 ---")

    # word_outputs[rel_type][li][word] = residual vector
    word_outputs = {rel: {li: {} for li in test_layers} for rel in RELATION_TEMPLATES}

    for rel_type, template in RELATION_TEMPLATES.items():
        log(f"  关系: {rel_type}")
        for word in all_words:
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

        log(f"    完成 ({len(all_words)} 词)")

    # Step 2: 逐步增加关系数, 观察有效维度变化
    log("\n--- Step 2: 关系数 vs 有效维度 ---")

    # 关系子集: 从4到10逐步增加
    relation_subsets = {
        "4rel": ["habitat", "category", "material", "size"],  # CCXV的4种
        "6rel": ["habitat", "category", "material", "size", "color", "shape"],
        "8rel": ["habitat", "category", "material", "size", "color", "shape", "function", "taste"],
        "10rel": list(RELATION_TEMPLATES.keys()),
    }

    for subset_name, rel_list in relation_subsets.items():
        log(f"\n  === {subset_name} ({len(rel_list)}种关系) ===")

        for li in test_layers:
            # 合并所有关系的向量
            all_rel_vecs = []
            all_rel_labels = []

            for rel_type in rel_list:
                for w in all_words:
                    v = word_outputs[rel_type][li].get(w)
                    if v is not None:
                        all_rel_vecs.append(v)
                        all_rel_labels.append(rel_type)

            if len(all_rel_vecs) < 10:
                continue

            all_rel_vecs = np.array(all_rel_vecs)
            mean_vec = all_rel_vecs.mean(axis=0)
            centered = all_rel_vecs - mean_vec

            n_svd = min(60, centered.shape[0] - 1, centered.shape[1] - 1)
            U, s, Vt = np.linalg.svd(centered, full_matrices=False)

            # 每个SVD模式的关系区分力
            projections = centered @ Vt[:n_svd].T
            rel_types_set = list(set(all_rel_labels))

            mode_rel_f = []
            for mi in range(min(n_svd, 40)):
                proj_mi = projections[:, mi]
                groups = []
                for rt in rel_types_set:
                    rt_idx = [i for i, l in enumerate(all_rel_labels) if l == rt]
                    if rt_idx:
                        groups.append(proj_mi[rt_idx])
                if len(groups) >= 2:
                    f_stat, _ = stats.f_oneway(*groups)
                    mode_rel_f.append(f_stat if not np.isnan(f_stat) else 0)
                else:
                    mode_rel_f.append(0)

            # 关系有效维度(F>10的显著模式数)
            significant_modes_f10 = sum(1 for f in mode_rel_f if f > 10)
            # 更严格: F>100
            significant_modes_f100 = sum(1 for f in mode_rel_f if f > 100)
            # 方差解释: 95%方差需要的维度
            total_var = np.sum(s[:n_svd] ** 2)
            cumvar = np.cumsum(s[:n_svd] ** 2) / total_var
            eff_dim_95 = int(np.searchsorted(cumvar, 0.95)) + 1

            log(f"    L{li}: n_rel={len(rel_list)}, "
                f"F>10 modes={significant_modes_f10}, "
                f"F>100 modes={significant_modes_f100}, "
                f"95%var dim={eff_dim_95}, "
                f"top5_F={[float(f) for f in sorted(mode_rel_f, reverse=True)[:5]]}")

            key = f"scaling_{subset_name}_L{li}"
            results[key] = {
                "subset": subset_name,
                "n_relations": len(rel_list),
                "layer": li,
                "significant_modes_F10": significant_modes_f10,
                "significant_modes_F100": significant_modes_f100,
                "eff_dim_95var": eff_dim_95,
                "top5_rel_F": [float(f) for f in sorted(mode_rel_f, reverse=True)[:5]],
                "singular_values": [float(x) for x in s[:15]],
            }

    # Step 3: 10关系的子空间两两正交性
    log("\n--- Step 3: 10关系子空间两两正交性(最后层) ---")

    li_last = n_layers - 1

    # 计算每种关系的SVD子空间
    rel_subspaces = {}
    for rel_type in RELATION_TEMPLATES:
        vecs = []
        for w in all_words:
            v = word_outputs[rel_type][li_last].get(w)
            if v is not None:
                vecs.append(v)
        if len(vecs) < 4:
            continue
        vecs = np.array(vecs)
        mean_vec = vecs.mean(axis=0)
        centered = vecs - mean_vec
        n_svd = min(15, centered.shape[0] - 1, centered.shape[1] - 1)
        _, s, Vt = np.linalg.svd(centered, full_matrices=False)
        rel_subspaces[rel_type] = Vt[:n_svd, :]

    rel_names = sorted(rel_subspaces.keys())
    log(f"\n  最后层 L{li_last}, {len(rel_names)}种关系有子空间")

    # 两两计算主角
    orthogonal_pairs = 0
    overlapping_pairs = 0
    total_pairs = 0
    max_cos_matrix = np.zeros((len(rel_names), len(rel_names)))

    for i, r1 in enumerate(rel_names):
        for j, r2 in enumerate(rel_names):
            if i >= j:
                continue
            V1 = rel_subspaces[r1]
            V2 = rel_subspaces[r2]
            k = min(5, V1.shape[0], V2.shape[0])
            cos_vals = compute_principal_angles(V1, V2, k=k)
            max_cos = float(np.max(np.abs(cos_vals)))
            mean_cos = float(np.mean(np.abs(cos_vals)))

            max_cos_matrix[i, j] = max_cos
            max_cos_matrix[j, i] = max_cos

            total_pairs += 1
            if max_cos < 0.5:
                orthogonal_pairs += 1
            elif max_cos > 0.7:
                overlapping_pairs += 1

            if max_cos > 0.7 or max_cos < 0.3:
                log(f"    {r1} vs {r2}: max_cos={max_cos:.3f}, mean_cos={mean_cos:.3f} "
                     f"{'★★★ 重叠' if max_cos > 0.7 else '★★★ 正交'}")

            key = f"pair_{r1}_vs_{r2}_L{li_last}"
            results[key] = {
                "rel1": r1, "rel2": r2, "layer": li_last,
                "max_cos": max_cos, "mean_cos": mean_cos,
            }

    log(f"\n  正交对(max_cos<0.5): {orthogonal_pairs}/{total_pairs}")
    log(f"  重叠对(max_cos>0.7): {overlapping_pairs}/{total_pairs}")

    results["orthogonality_summary"] = {
        "layer": li_last,
        "n_relations": len(rel_names),
        "total_pairs": total_pairs,
        "orthogonal_pairs": orthogonal_pairs,
        "overlapping_pairs": overlapping_pairs,
    }

    # Step 4: 每种关系的有效维度
    log("\n--- Step 4: 每种关系的有效维度 ---")

    for li in test_layers:
        log(f"\n  L{li}:")
        for rel_type in RELATION_TEMPLATES:
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

            n_svd = min(20, centered.shape[0] - 1, centered.shape[1] - 1)
            _, s, Vt = np.linalg.svd(centered, full_matrices=False)

            total_var = np.sum(s[:n_svd] ** 2)
            cumvar = np.cumsum(s[:n_svd] ** 2) / total_var
            eff_dim_95 = int(np.searchsorted(cumvar, 0.95)) + 1

            # ANOVA: 类别区分力
            projections = centered @ Vt[:n_svd].T
            cat_values = set(WORD_ATTRS.get(w, "") for w in valid_words)
            cat_values.discard("")

            max_cat_f = 0
            if len(cat_values) >= 2:
                for mi in range(min(n_svd, 10)):
                    proj_mi = projections[:, mi]
                    groups = []
                    for cv in cat_values:
                        cv_idx = [i for i, w in enumerate(valid_words) if WORD_ATTRS.get(w) == cv]
                        if cv_idx:
                            groups.append(proj_mi[cv_idx])
                    if len(groups) >= 2:
                        f_stat, _ = stats.f_oneway(*groups)
                        max_cat_f = max(max_cat_f, f_stat if not np.isnan(f_stat) else 0)

            log(f"    {rel_type}: eff_dim_95={eff_dim_95}, top_s=[{s[0]:.1f},{s[1]:.1f},{s[2]:.1f}], max_cat_F={max_cat_f:.1f}")

            key = f"single_rel_{rel_type}_L{li}"
            results[key] = {
                "relation": rel_type, "layer": li,
                "eff_dim_95": eff_dim_95,
                "top3_sv": [float(x) for x in s[:3]],
                "max_cat_F": float(max_cat_f),
            }

    # Step 5: 关系数vs维度的核心分析
    log("\n--- Step 5: 关系数 vs 维度 核心分析 ---")

    scaling_data = {}
    for li in test_layers:
        scaling_data[li] = {}
        for subset_name, rel_list in relation_subsets.items():
            key = f"scaling_{subset_name}_L{li}"
            if key in results:
                scaling_data[li][len(rel_list)] = results[key]["significant_modes_F10"]

    log("\n  关系数 → 有效维度(F>10):")
    header = "  层    "
    for n_rel in [4, 6, 8, 10]:
        header += f"  {n_rel}rel"
    log(header)

    for li in test_layers:
        row = f"  L{li:>4d}"
        for n_rel in [4, 6, 8, 10]:
            dim = scaling_data.get(li, {}).get(n_rel, "?")
            row += f"  {str(dim):>5s}"
        log(row)

    # 判断: 3维是否普适?
    log("\n--- 核心判断 ---")
    last_layer_dims = {}
    for n_rel in [4, 6, 8, 10]:
        dim = scaling_data.get(li_last, {}).get(n_rel, None)
        last_layer_dims[n_rel] = dim
        if dim is not None:
            if dim <= 3:
                log(f"  ★★★★★ {n_rel}种关系 → {dim}维: 3维流形是普适的!")
            elif dim <= 5:
                log(f"  ★★★ {n_rel}种关系 → {dim}维: 3维不普适, 但仍低维")
            else:
                log(f"  ★ {n_rel}种关系 → {dim}维: 高维, 3维是4类的artifact")

    results["core_judgment"] = {
        "model": model_name,
        "last_layer_dims": last_layer_dims,
        "is_3dim_universal": all(v is not None and v <= 3 for v in last_layer_dims.values()),
    }

    # 保存结果
    out_file = TEMP_DIR / f"ccxvii_10relation_manifold_{model_name}.json"
    with open(out_file, "w") as f:
        json.dump({
            "model": model_name,
            "d_model": d_model,
            "n_layers": n_layers,
            "test_layers": test_layers,
            "n_relations": len(RELATION_TEMPLATES),
            "n_words": len(all_words),
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
