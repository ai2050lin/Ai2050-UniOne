"""
CCXV(315): 多维语义流形分析 — 不同关系子空间是否正交?
======================================================================
CCXIII发现DS7B所有关系共享1维子空间(max_cos≈0.95), 但非蒸馏模型呢?

核心问题:
  1. 非蒸馏模型(Qwen3/GLM4)的关系子空间是否正交?
  2. 正交 → 语义空间是多维流形, 不同关系占据不同维度
  3. 重叠 → 关系共享编码, 维度低
  4. 蒸馏 = 子空间坍缩(多维→1维)?

设计:
  - 5种关系(habitat/category/material/color/size)
  - 对每种关系计算SVD子空间(top-k模式)
  - 计算所有关系对的Grassmann距离/主角
  - 三模型对比: Qwen3(5维) vs GLM4(15维) vs DS7B(1维)

关键指标:
  - 主角余弦: 小=正交(独立编码), 大=重叠(共享编码)
  - 有效维度: 子空间的本质维数

用法:
  python ccxv_relation_manifold.py --model qwen3
  python ccxv_relation_manifold.py --model glm4
  python ccxv_relation_manifold.py --model deepseek7b
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
LOG_FILE = TEMP_DIR / "ccxv_relation_manifold_log.txt"

# 扩展词汇表 — 每类别8词, 确保统计可靠性
RELATION_WORDS = {
    "animal_land": ["dog", "cat", "lion", "tiger", "horse", "cow", "fox", "deer"],
    "animal_ocean": ["whale", "shark", "dolphin", "octopus", "salmon", "seal", "crab", "squid"],
    "animal_sky": ["eagle", "hawk", "owl", "parrot", "crow", "sparrow", "falcon", "swallow"],
    "food_fruit": ["apple", "banana", "mango", "cherry", "peach", "grape", "lemon", "orange"],
    "food_meal": ["bread", "pizza", "steak", "rice", "pasta", "soup", "salad", "cheese"],
    "tool_hand": ["hammer", "knife", "drill", "chisel", "ruler", "wrench", "saw", "axe"],
    "nature_land": ["mountain", "river", "desert", "glacier", "forest", "canyon", "valley", "hill"],
}

# 每个词的属性
WORD_ATTRS = {
    "dog": {"category":"animal", "habitat":"land", "size":"medium", "material":"flesh"},
    "cat": {"category":"animal", "habitat":"land", "size":"small", "material":"flesh"},
    "lion": {"category":"animal", "habitat":"land", "size":"large", "material":"flesh"},
    "tiger": {"category":"animal", "habitat":"land", "size":"large", "material":"flesh"},
    "horse": {"category":"animal", "habitat":"land", "size":"large", "material":"flesh"},
    "cow": {"category":"animal", "habitat":"land", "size":"large", "material":"flesh"},
    "fox": {"category":"animal", "habitat":"land", "size":"medium", "material":"flesh"},
    "deer": {"category":"animal", "habitat":"land", "size":"medium", "material":"flesh"},
    "whale": {"category":"animal", "habitat":"ocean", "size":"large", "material":"flesh"},
    "shark": {"category":"animal", "habitat":"ocean", "size":"large", "material":"flesh"},
    "dolphin": {"category":"animal", "habitat":"ocean", "size":"large", "material":"flesh"},
    "octopus": {"category":"animal", "habitat":"ocean", "size":"medium", "material":"flesh"},
    "salmon": {"category":"animal", "habitat":"ocean", "size":"medium", "material":"flesh"},
    "seal": {"category":"animal", "habitat":"ocean", "size":"medium", "material":"flesh"},
    "crab": {"category":"animal", "habitat":"ocean", "size":"small", "material":"flesh"},
    "squid": {"category":"animal", "habitat":"ocean", "size":"medium", "material":"flesh"},
    "eagle": {"category":"animal", "habitat":"sky", "size":"large", "material":"flesh"},
    "hawk": {"category":"animal", "habitat":"sky", "size":"medium", "material":"flesh"},
    "owl": {"category":"animal", "habitat":"sky", "size":"medium", "material":"flesh"},
    "parrot": {"category":"animal", "habitat":"sky", "size":"small", "material":"flesh"},
    "crow": {"category":"animal", "habitat":"sky", "size":"medium", "material":"flesh"},
    "sparrow": {"category":"animal", "habitat":"sky", "size":"small", "material":"flesh"},
    "falcon": {"category":"animal", "habitat":"sky", "size":"medium", "material":"flesh"},
    "swallow": {"category":"animal", "habitat":"sky", "size":"small", "material":"flesh"},
    "apple": {"category":"food", "habitat":"land", "size":"small", "material":"organic"},
    "banana": {"category":"food", "habitat":"land", "size":"medium", "material":"organic"},
    "mango": {"category":"food", "habitat":"land", "size":"small", "material":"organic"},
    "cherry": {"category":"food", "habitat":"land", "size":"small", "material":"organic"},
    "peach": {"category":"food", "habitat":"land", "size":"small", "material":"organic"},
    "grape": {"category":"food", "habitat":"land", "size":"small", "material":"organic"},
    "lemon": {"category":"food", "habitat":"land", "size":"small", "material":"organic"},
    "orange": {"category":"food", "habitat":"land", "size":"small", "material":"organic"},
    "bread": {"category":"food", "habitat":"land", "size":"medium", "material":"organic"},
    "pizza": {"category":"food", "habitat":"land", "size":"medium", "material":"organic"},
    "steak": {"category":"food", "habitat":"land", "size":"medium", "material":"organic"},
    "rice": {"category":"food", "habitat":"land", "size":"small", "material":"organic"},
    "pasta": {"category":"food", "habitat":"land", "size":"medium", "material":"organic"},
    "soup": {"category":"food", "habitat":"land", "size":"medium", "material":"organic"},
    "salad": {"category":"food", "habitat":"land", "size":"medium", "material":"organic"},
    "cheese": {"category":"food", "habitat":"land", "size":"medium", "material":"organic"},
    "hammer": {"category":"tool", "habitat":"land", "size":"medium", "material":"metal"},
    "knife": {"category":"tool", "habitat":"land", "size":"small", "material":"metal"},
    "drill": {"category":"tool", "habitat":"land", "size":"medium", "material":"metal"},
    "chisel": {"category":"tool", "habitat":"land", "size":"small", "material":"metal"},
    "ruler": {"category":"tool", "habitat":"land", "size":"small", "material":"plastic"},
    "wrench": {"category":"tool", "habitat":"land", "size":"medium", "material":"metal"},
    "saw": {"category":"tool", "habitat":"land", "size":"medium", "material":"metal"},
    "axe": {"category":"tool", "habitat":"land", "size":"medium", "material":"metal"},
    "mountain": {"category":"nature", "habitat":"land", "size":"large", "material":"rock"},
    "river": {"category":"nature", "habitat":"land", "size":"large", "material":"water"},
    "desert": {"category":"nature", "habitat":"land", "size":"large", "material":"sand"},
    "glacier": {"category":"nature", "habitat":"land", "size":"large", "material":"ice"},
    "forest": {"category":"nature", "habitat":"land", "size":"large", "material":"wood"},
    "canyon": {"category":"nature", "habitat":"land", "size":"large", "material":"rock"},
    "valley": {"category":"nature", "habitat":"land", "size":"large", "material":"earth"},
    "hill": {"category":"nature", "habitat":"land", "size":"medium", "material":"earth"},
}

TEMPLATES = {
    "habitat": "The {} lives in the",
    "category": "The {} is a",
    "material": "The {} is made of",
    "size": "The {} is very",
}


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def compute_principal_angles(V1, V2, k=10):
    """
    计算两个子空间之间的主角(principal angles)
    V1, V2: [n, d] 的正交基行(每行是一个基向量)
    返回: cos值列表 (1=平行, 0=正交)
    """
    k1 = min(k, V1.shape[0], V2.shape[0])
    overlap = V1[:k1] @ V2[:k1].T  # [k1, k1]
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
    for ws in RELATION_WORDS.values():
        all_words.extend(ws)
    all_words = list(set(all_words))

    log(f"\n{'='*70}")
    log(f"CCXV(315): 多维语义流形分析 - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}")
    log(f"  测试层: {test_layers}")
    log(f"  总词数: {len(all_words)}")
    log(f"  关系类型: {list(TEMPLATES.keys())}")
    log(f"{'='*70}")

    results = {}

    # Step 1: 收集残差流
    log("\n--- Step 1: 收集残差流 ---")

    word_outputs = {rel: {li: {} for li in test_layers} for rel in TEMPLATES}

    for rel_type, template in TEMPLATES.items():
        log(f"\n  关系: {rel_type}")
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

        log(f"    完成")

    # Step 2: 计算每种关系的SVD子空间
    log("\n--- Step 2: 计算关系SVD子空间 ---")

    relation_subspaces = {}  # {rel_type: {li: Vt_top_k}}

    for rel_type in TEMPLATES:
        relation_subspaces[rel_type] = {}

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

            n_svd = min(20, centered.shape[0] - 1, centered.shape[1] - 1)
            if n_svd < 3:
                continue

            U, s, Vt = np.linalg.svd(centered, full_matrices=False)
            Vt = Vt[:n_svd, :]

            # 每个关系的有效维度(95%方差)
            total_var = np.sum(s[:n_svd] ** 2)
            cumvar = np.cumsum(s[:n_svd] ** 2) / total_var
            eff_dim = int(np.searchsorted(cumvar, 0.95)) + 1

            relation_subspaces[rel_type][li] = Vt

            # 计算该关系各属性的区分力
            attr_name = rel_type
            attr_values = set()
            for w in valid_words:
                if attr_name in WORD_ATTRS.get(w, {}):
                    attr_values.add(WORD_ATTRS[w][attr_name])

            # 也计算category区分力
            cat_values = set(WORD_ATTRS.get(w, {}).get("category", "") for w in valid_words)
            cat_values.discard("")

            projections = centered @ Vt.T
            max_rel_f = 0
            max_cat_f = 0

            if len(attr_values) >= 2:
                for mi in range(min(n_svd, 10)):
                    proj_mi = projections[:, mi]
                    groups = []
                    for av in attr_values:
                        av_idx = [i for i, w in enumerate(valid_words)
                                 if WORD_ATTRS.get(w, {}).get(attr_name) == av]
                        if av_idx:
                            groups.append(proj_mi[av_idx])
                    if len(groups) >= 2:
                        f_stat, _ = stats.f_oneway(*groups)
                        max_rel_f = max(max_rel_f, f_stat if not np.isnan(f_stat) else 0)

            if len(cat_values) >= 2:
                for mi in range(min(n_svd, 10)):
                    proj_mi = projections[:, mi]
                    groups = []
                    for cv in cat_values:
                        cv_idx = [i for i, w in enumerate(valid_words)
                                 if WORD_ATTRS.get(w, {}).get("category") == cv]
                        if cv_idx:
                            groups.append(proj_mi[cv_idx])
                    if len(groups) >= 2:
                        f_stat, _ = stats.f_oneway(*groups)
                        max_cat_f = max(max_cat_f, f_stat if not np.isnan(f_stat) else 0)

            ratio = max_rel_f / max(max_cat_f, 1e-6)

            log(f"  {rel_type} L{li}: eff_dim={eff_dim}, top_rel_F={max_rel_f:.1f}, "
                f"top_cat_F={max_cat_f:.1f}, ratio={ratio:.1f}")

            key = f"subspace_{rel_type}_L{li}"
            results[key] = {
                "relation": rel_type,
                "layer": li,
                "eff_dim": eff_dim,
                "top_rel_F": float(max_rel_f),
                "top_cat_F": float(max_cat_f),
                "rel_cat_ratio": float(ratio),
                "singular_values": [float(x) for x in s[:10]],
            }

    # Step 3: 关系子空间两两对比(Grassmann距离)
    log("\n--- Step 3: 关系子空间两两主角 ---")

    rel_pairs = [("habitat", "category"), ("habitat", "material"), ("habitat", "size"),
                 ("category", "material"), ("category", "size"), ("material", "size")]

    for li in test_layers:
        log(f"\n  L{li}:")

        for rel1, rel2 in rel_pairs:
            if rel1 not in relation_subspaces or rel2 not in relation_subspaces:
                continue
            if li not in relation_subspaces[rel1] or li not in relation_subspaces[rel2]:
                continue

            V1 = relation_subspaces[rel1][li]
            V2 = relation_subspaces[rel2][li]

            k = min(10, V1.shape[0], V2.shape[0])
            cos_vals = compute_principal_angles(V1, V2, k=k)

            mean_cos = float(np.mean(np.abs(cos_vals)))
            max_cos = float(np.max(np.abs(cos_vals)))

            # Grassmann距离
            grassmann_dist = float(np.sqrt(np.sum(np.arccos(np.clip(np.abs(cos_vals), 0, 1)) ** 2)))

            log(f"    {rel1} vs {rel2}: mean_cos={mean_cos:.3f}, max_cos={max_cos:.3f}, "
                f"grassmann_dist={grassmann_dist:.3f}")

            key = f"angle_{rel1}_vs_{rel2}_L{li}"
            results[key] = {
                "rel1": rel1,
                "rel2": rel2,
                "layer": li,
                "mean_cos": mean_cos,
                "max_cos": max_cos,
                "grassmann_dist": grassmann_dist,
                "cos_values": [float(x) for x in cos_vals],
            }

    # Step 4: 全关系联合子空间分析
    log("\n--- Step 4: 全关系联合子空间 ---")

    for li in test_layers:
        # 合并所有关系的向量
        all_rel_vecs = []
        all_rel_labels = []

        for rel_type in TEMPLATES:
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

        n_svd = min(50, centered.shape[0] - 1, centered.shape[1] - 1)
        U, s, Vt = np.linalg.svd(centered, full_matrices=False)

        # 每个SVD模式的关系区分力
        projections = centered @ Vt[:n_svd].T
        rel_types_set = list(set(all_rel_labels))

        mode_rel_f = []
        for mi in range(min(n_svd, 30)):
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

        # 关系总维度(区分不同关系的有效维度)
        significant_modes = sum(1 for f in mode_rel_f if f > 10)

        log(f"  L{li}: 关系总有效维度={significant_modes}, "
            f"top5_rel_F={[float(f) for f in sorted(mode_rel_f, reverse=True)[:5]]}")

        key = f"joint_subspace_L{li}"
        results[key] = {
            "layer": li,
            "relation_eff_dim": significant_modes,
            "top5_rel_F": [float(f) for f in sorted(mode_rel_f, reverse=True)[:5]],
            "singular_values": [float(x) for x in s[:20]],
        }

    # Step 5: 各关系的子空间独立性(与联合子空间的重叠)
    log("\n--- Step 5: 各关系子空间独立性 ---")

    for li in test_layers:
        # 联合子空间(取top-20)
        all_rel_vecs = []
        for rel_type in TEMPLATES:
            for w in all_words:
                v = word_outputs[rel_type][li].get(w)
                if v is not None:
                    all_rel_vecs.append(v)

        if len(all_rel_vecs) < 10:
            continue

        all_rel_vecs = np.array(all_rel_vecs)
        mean_vec = all_rel_vecs.mean(axis=0)
        centered = all_rel_vecs - mean_vec

        n_svd = min(30, centered.shape[0] - 1, centered.shape[1] - 1)
        _, s_joint, Vt_joint = np.linalg.svd(centered, full_matrices=False)
        Vt_joint_top = Vt_joint[:n_svd, :]

        log(f"\n  L{li} (联合子空间维度={n_svd}):")

        for rel_type in TEMPLATES:
            if rel_type not in relation_subspaces or li not in relation_subspaces[rel_type]:
                continue

            V_rel = relation_subspaces[rel_type][li]
            k = min(10, V_rel.shape[0], Vt_joint_top.shape[0])

            cos_vals = compute_principal_angles(V_rel, Vt_joint_top, k=k)
            mean_cos = float(np.mean(np.abs(cos_vals)))
            max_cos = float(np.max(np.abs(cos_vals)))

            # 独立性 = 1 - max_cos (越接近1越独立)
            independence = 1 - max_cos

            log(f"    {rel_type}: 与联合子空间 max_cos={max_cos:.3f}, "
                f"independence={independence:.3f}")

            key = f"independence_{rel_type}_L{li}"
            results[key] = {
                "relation": rel_type,
                "layer": li,
                "max_cos_with_joint": max_cos,
                "independence": float(independence),
            }

    # 保存结果
    out_file = TEMP_DIR / f"ccxv_relation_manifold_{model_name}.json"
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
