"""
CCXX(370): 50概念暗物质PCA — 验证维度饱和点与分块正交跨模型一致性
==================================================================

CCXIX发现:
  eff_rank≈0.83×n, 暗物质维度随概念数线性增长
  但只有20个概念, 无法确定是否饱和

CCXX目标:
  Exp1: 50概念暗物质PCA — 确定eff_rank在n=50时是否饱和
  Exp2: 5类语义的跨组正交性 — 验证分块正交是否跨模型一致
  Exp3: 增量维度曲线 — 5→10→15→...→50的eff_rank轨迹
  Exp4: 子空间夹角vs语义距离 — 正交性与语义距离是否相关

用法:
  python ccxx_50concept_dark_matter_pca.py --model qwen3 --exp 1
  python ccxx_50concept_dark_matter_pca.py --model qwen3 --exp 2
  python ccxx_50concept_dark_matter_pca.py --model qwen3 --exp 3
  python ccxx_50concept_dark_matter_pca.py --model qwen3 --exp all
"""

import argparse, os, sys, json, gc, warnings, time
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import torch

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANS_TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, MODEL_CONFIGS, get_W_U
)

TEMP = Path("tests/glm5_temp")

# ================================================================
# 50概念定义 — 5类语义, 每类10概念
# ================================================================
CONCEPTS_50 = {
    # === 具体物体 (Concrete Objects) ===
    "apple": ["The word is apple", "I ate an apple", "A red apple", "The apple fell", "Apple is a fruit"],
    "dog": ["The word is dog", "A big dog", "The dog barked", "My pet dog", "Dog is an animal"],
    "mountain": ["The word is mountain", "A tall mountain", "The mountain peak", "Climb the mountain", "Mountain is high"],
    "ocean": ["The word is ocean", "The deep ocean", "Ocean waves", "Swim in the ocean", "Ocean is vast"],
    "house": ["The word is house", "A big house", "The house stood", "Build a house", "House is a building"],
    "car": ["The word is car", "A fast car", "The car stopped", "Drive the car", "Car is a vehicle"],
    "tree": ["The word is tree", "A tall tree", "The tree grew", "Plant a tree", "Tree has leaves"],
    "book": ["The word is book", "Read a book", "A thick book", "Open the book", "Book has pages"],
    "flower": ["The word is flower", "A beautiful flower", "The flower bloomed", "Pick the flower", "Flower has petals"],
    "bird": ["The word is bird", "A small bird", "The bird flew", "Watch the bird", "Bird can sing"],

    # === 抽象概念 (Abstract Concepts) ===
    "love": ["The word is love", "Feel the love", "Love is strong", "Show your love", "Love and care"],
    "science": ["The word is science", "Study of science", "Science advances", "Modern science", "Science is knowledge"],
    "freedom": ["The word is freedom", "Fight for freedom", "Freedom is precious", "Value freedom", "Freedom to choose"],
    "justice": ["The word is justice", "Seek justice", "Justice is fair", "Demand justice", "Justice and law"],
    "beauty": ["The word is beauty", "Appreciate beauty", "Beauty is deep", "Find beauty", "Beauty and art"],
    "truth": ["The word is truth", "Seek the truth", "Truth is clear", "Tell the truth", "Truth and honesty"],
    "wisdom": ["The word is wisdom", "Gain wisdom", "Wisdom is valuable", "Share wisdom", "Wisdom and age"],
    "courage": ["The word is courage", "Show courage", "Courage is brave", "Need courage", "Courage and fear"],
    "hope": ["The word is hope", "Feel hope", "Hope is bright", "Keep hope", "Hope and faith"],
    "peace": ["The word is peace", "Find peace", "Peace is calm", "Want peace", "Peace and quiet"],

    # === 关系角色 (Relational Roles) ===
    "king": ["The word is king", "The king ruled", "A wise king", "The king and queen", "King is a ruler"],
    "doctor": ["The word is doctor", "The doctor helped", "A good doctor", "Visit the doctor", "Doctor treats patients"],
    "teacher": ["The word is teacher", "The teacher taught", "A kind teacher", "Ask the teacher", "Teacher guides students"],
    "mother": ["The word is mother", "My dear mother", "The mother cared", "Love your mother", "Mother and child"],
    "enemy": ["The word is enemy", "A fierce enemy", "The enemy attacked", "Defeat the enemy", "Enemy is hostile"],
    "friend": ["The word is friend", "A good friend", "The friend helped", "Trust your friend", "Friend is loyal"],
    "leader": ["The word is leader", "A strong leader", "The leader decided", "Follow the leader", "Leader guides others"],
    "child": ["The word is child", "A young child", "The child played", "Teach the child", "Child is innocent"],
    "hero": ["The word is hero", "A brave hero", "The hero saved", "Be a hero", "Hero is courageous"],
    "judge": ["The word is judge", "A fair judge", "The judge decided", "Respect the judge", "Judge is impartial"],

    # === 时间/变化 (Temporal/Change) ===
    "time": ["The word is time", "Passage of time", "Time flies", "Save time", "Time is precious"],
    "change": ["The word is change", "Embrace change", "Change happens", "Adapt to change", "Change and growth"],
    "history": ["The word is history", "Study history", "History repeats", "Learn from history", "History is past"],
    "future": ["The word is future", "Plan the future", "Future is bright", "Shape the future", "Future and hope"],
    "memory": ["The word is memory", "Recall a memory", "Memory fades", "Keep the memory", "Memory and past"],
    "birth": ["The word is birth", "A new birth", "Celebrate birth", "Birth and life", "Birth is beginning"],
    "death": ["The word is death", "Mourn the death", "Death is final", "Face death", "Death and loss"],
    "growth": ["The word is growth", "Experience growth", "Growth takes time", "Encourage growth", "Growth and progress"],
    "war": ["The word is war", "A terrible war", "The war ended", "Prevent war", "War and conflict"],
    "season": ["The word is season", "A new season", "Season changes", "Enjoy the season", "Season and weather"],

    # === 空间/物理 (Spatial/Physical) ===
    "river": ["The word is river", "A long river", "The river flows", "Cross the river", "River is water"],
    "sky": ["The word is sky", "A clear sky", "The sky is blue", "Look at the sky", "Sky is above"],
    "forest": ["The word is forest", "A dense forest", "The forest hides", "Walk through forest", "Forest has trees"],
    "city": ["The word is city", "A big city", "The city grows", "Live in city", "City is urban"],
    "road": ["The word is road", "A long road", "The road bends", "Travel the road", "Road leads somewhere"],
    "bridge": ["The word is bridge", "A tall bridge", "The bridge spans", "Cross the bridge", "Bridge connects"],
    "wall": ["The word is wall", "A high wall", "The wall stands", "Build a wall", "Wall is solid"],
    "island": ["The word is island", "A small island", "The island floats", "Visit the island", "Island is isolated"],
    "desert": ["The word is desert", "A vast desert", "The desert burns", "Cross the desert", "Desert is dry"],
    "star": ["The word is star", "A bright star", "The star shines", "Watch the star", "Star is distant"],
}

# 语义分组
SEMANTIC_GROUPS = {
    "concrete": ["apple", "dog", "mountain", "ocean", "house", "car", "tree", "book", "flower", "bird"],
    "abstract": ["love", "science", "freedom", "justice", "beauty", "truth", "wisdom", "courage", "hope", "peace"],
    "relational": ["king", "doctor", "teacher", "mother", "enemy", "friend", "leader", "child", "hero", "judge"],
    "temporal": ["time", "change", "history", "future", "memory", "birth", "death", "growth", "war", "season"],
    "spatial": ["river", "sky", "forest", "city", "road", "bridge", "wall", "island", "desert", "star"],
}

BASELINE_TEXT = "The word is the"


# ================================================================
# 基础工具函数
# ================================================================
def collect_states_at_layers(model, tokenizer, device, text, capture_layers):
    """用hooks收集指定层的残差流状态"""
    captured = {}
    all_layers = get_layers(model)
    def make_hook(li):
        def hook(module, inp, output):
            if isinstance(output, tuple):
                captured[li] = output[0][0, -1, :].detach().float().cpu().numpy()
            else:
                captured[li] = output[0, -1, :].detach().float().cpu().numpy()
        return hook
    hooks = []
    for li in capture_layers:
        if li < len(all_layers):
            hooks.append(all_layers[li].register_forward_hook(make_hook(li)))
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids)
        except Exception as e:
            print(f"  Forward failed: {e}")
            for h in hooks: h.remove()
            return {}, None
    for h in hooks: h.remove()
    logits = outputs.logits[0, -1, :].detach().float().cpu().numpy()
    gc.collect()
    return captured, logits


def compute_wu_basis(model):
    """计算W_U行空间基"""
    W_U = get_W_U(model)
    from scipy.sparse.linalg import svds
    k_wu = min(200, min(W_U.shape) - 2)
    U_wu, s_wu, _ = svds(W_U.T.astype(np.float32), k=k_wu)
    U_wu = np.asarray(U_wu, dtype=np.float64)
    del W_U
    gc.collect()
    return U_wu


def decompose_delta(delta, U_wu):
    """将delta分解为W_U分量和暗物质"""
    proj_wu = U_wu.T @ delta
    delta_wu = U_wu @ proj_wu
    delta_dark = delta - delta_wu
    return delta_wu, delta_dark, proj_wu


def compute_pca_stats(vectors):
    """对向量集合做PCA, 返回eff_rank和方差分布"""
    X = np.array(vectors)
    n_samples, d = X.shape
    X_c = X - X.mean(axis=0, keepdims=True)

    if n_samples < d:
        XXt = X_c @ X_c.T
        eigenvalues, _ = np.linalg.eigh(XXt)
        eigenvalues = eigenvalues[::-1]
        s = np.sqrt(np.maximum(eigenvalues, 0))
    else:
        _, s, _ = np.linalg.svd(X_c, full_matrices=False)

    total_var = np.sum(s ** 2)
    if total_var < 1e-20:
        return None

    var_ratio = s ** 2 / total_var
    cum_var = np.cumsum(var_ratio)

    p = var_ratio[var_ratio > 1e-10]
    entropy = -np.sum(p * np.log2(p))
    eff_rank = 2 ** entropy

    n_90 = int(np.searchsorted(cum_var, 0.90)) + 1
    n_95 = int(np.searchsorted(cum_var, 0.95)) + 1

    return {
        "n_samples": n_samples,
        "effective_rank": float(eff_rank),
        "n_for_90": n_90,
        "n_for_95": n_95,
        "var_ratio_top10": [float(x) for x in var_ratio[:10]],
        "cum_var_top10": [float(x) for x in cum_var[:10]],
        "singular_values_top10": [float(x) for x in s[:10]],
    }


def compute_pca_and_get_PC1(vectors):
    """做PCA并返回PC1方向"""
    X = np.array(vectors)
    n_samples, d = X.shape
    X_c = X - X.mean(axis=0, keepdims=True)

    if n_samples < d:
        XXt = X_c @ X_c.T
        eigenvalues, eigvecs = np.linalg.eigh(XXt)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigvecs = eigvecs[:, idx]
        # 转换回原始空间
        pc1 = X_c.T @ eigvecs[:, 0]
        pc1 = pc1 / (np.linalg.norm(pc1) + 1e-12)
    else:
        U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
        pc1 = Vt[0]

    return pc1


# ================================================================
# Exp1: 50概念暗物质PCA全量分析
# ================================================================
def run_exp1(model_name):
    """收集50概念的暗物质向量, 做全量PCA分析"""
    print(f"\n{'='*70}")
    print(f"  Exp1: 50-Concept Dark Matter PCA ({model_name})")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    # 关键层
    key_layers = [l for l in [6, 12, 18, 24, 30, 36] if l < n_layers]
    print(f"  Model: {model_name}, n_layers={n_layers}, d_model={d_model}")
    print(f"  Key layers: {key_layers}")

    U_wu = compute_wu_basis(model)

    # 收集baseline
    bl_all, _ = collect_states_at_layers(model, tokenizer, device, BASELINE_TEXT, key_layers)

    # 收集所有概念的暗物质
    all_dark = {l: {} for l in key_layers}  # {layer: {concept_name: dark_vec}}
    all_dark_by_group = {l: {g: [] for g in SEMANTIC_GROUPS} for l in key_layers}

    for cname, templates in CONCEPTS_50.items():
        concept_states = {l: [] for l in key_layers}
        for template in templates:
            states, _ = collect_states_at_layers(model, tokenizer, device, template, key_layers)
            for l in key_layers:
                if l in states and l in bl_all:
                    concept_states[l].append(states[l])

        # 找到概念所属的语义组
        concept_group = None
        for g, glist in SEMANTIC_GROUPS.items():
            if cname in glist:
                concept_group = g
                break

        for l in key_layers:
            if concept_states[l]:
                delta = np.mean(concept_states[l], axis=0) - bl_all[l]
                _, delta_dark, _ = decompose_delta(delta, U_wu)

                if np.linalg.norm(delta_dark) > 1e-8:
                    all_dark[l][cname] = delta_dark
                    if concept_group:
                        all_dark_by_group[l][concept_group].append(delta_dark)

        print(f"  Collected: {cname}", end="\r")
        del concept_states
        gc.collect()

    print(f"\n  Collected all 50 concepts.")

    # ---- Part 1: 全量PCA ----
    print(f"\n  --- Part 1: Full 50-Concept PCA ---")
    results = {"model": model_name, "n_layers": n_layers, "d_model": d_model}

    for l in key_layers:
        dark_vecs = list(all_dark[l].values())
        if len(dark_vecs) < 3:
            continue
        stats = compute_pca_stats(dark_vecs)
        if stats:
            results[f"all_L{l}"] = stats
            print(f"  L{l}: n={stats['n_samples']}, eff_rank={stats['effective_rank']:.1f}, "
                  f"n_90={stats['n_for_90']}, n_95={stats['n_for_95']}, "
                  f"cum@1={stats['cum_var_top10'][0]:.3f}, cum@5={stats['cum_var_top10'][4]:.3f}")

    # ---- Part 2: 分组PCA ----
    print(f"\n  --- Part 2: Group PCA ---")
    for l in key_layers:
        for g, dark_vecs in all_dark_by_group[l].items():
            if len(dark_vecs) < 3:
                continue
            stats = compute_pca_stats(dark_vecs)
            if stats:
                results[f"{g}_L{l}"] = stats
                print(f"  L{l} ({g}, n={stats['n_samples']}): eff_rank={stats['effective_rank']:.1f}, "
                      f"n_90={stats['n_for_90']}, cum@1={stats['cum_var_top10'][0]:.3f}")

    # ---- Part 3: 增量维度曲线 ----
    print(f"\n  --- Part 3: Scaling Curve (n_concepts → eff_rank) ---")
    for l in key_layers:
        dark_vecs = list(all_dark[l].values())
        n_total = len(dark_vecs)
        if n_total < 5:
            continue

        scaling_data = []
        np.random.seed(42)
        indices = list(range(n_total))
        np.random.shuffle(indices)

        for n_concepts in [3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
            if n_concepts > n_total:
                continue
            subset = [dark_vecs[i] for i in indices[:n_concepts]]
            stats = compute_pca_stats(subset)
            if stats:
                scaling_data.append({
                    "n_concepts": n_concepts,
                    "eff_rank": stats["effective_rank"],
                    "n_90": stats["n_for_90"],
                })
                print(f"  L{l}: n={n_concepts}, eff_rank={stats['effective_rank']:.1f}")

        results[f"scaling_L{l}"] = scaling_data

    # ---- Part 4: 饱和分析 ----
    print(f"\n  --- Part 4: Saturation Analysis ---")
    for l in key_layers:
        key = f"scaling_L{l}"
        if key not in results:
            continue
        data = results[key]
        if len(data) < 3:
            continue

        # 计算增量比: eff_rank[n] / n_concepts[n]
        ratios = [d["eff_rank"] / d["n_concepts"] for d in data]
        # 如果ratios随n递减 → 开始饱和
        # 如果ratios稳定 → 线性增长
        trend = "saturating" if ratios[-1] < ratios[0] * 0.85 else "linear" if ratios[-1] > ratios[0] * 0.95 else "unclear"

        results[f"saturation_L{l}"] = {
            "ratio_first": ratios[0],
            "ratio_last": ratios[-1],
            "ratio_trend": trend,
            "ratios": ratios,
        }
        print(f"  L{l}: ratio_first={ratios[0]:.3f}, ratio_last={ratios[-1]:.3f}, trend={trend}")

    # 保存
    out_path = TEMP / f"ccxx_exp1_{model_name}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")

    release_model(model)
    return results


# ================================================================
# Exp2: 跨组正交性分析
# ================================================================
def run_exp2(model_name):
    """分析5类语义之间暗物质子空间的正交性"""
    print(f"\n{'='*70}")
    print(f"  Exp2: Cross-Group Orthogonality Analysis ({model_name})")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    key_layers = [l for l in [6, 12, 18, 24, 30, 36] if l < n_layers]

    U_wu = compute_wu_basis(model)
    bl_all, _ = collect_states_at_layers(model, tokenizer, device, BASELINE_TEXT, key_layers)

    # 收集各组暗物质
    all_dark_by_group = {l: {g: [] for g in SEMANTIC_GROUPS} for l in key_layers}

    for cname, templates in CONCEPTS_50.items():
        concept_group = None
        for g, glist in SEMANTIC_GROUPS.items():
            if cname in glist:
                concept_group = g
                break
        if not concept_group:
            continue

        concept_states = {l: [] for l in key_layers}
        for template in templates:
            states, _ = collect_states_at_layers(model, tokenizer, device, template, key_layers)
            for l in key_layers:
                if l in states and l in bl_all:
                    concept_states[l].append(states[l])

        for l in key_layers:
            if concept_states[l]:
                delta = np.mean(concept_states[l], axis=0) - bl_all[l]
                _, delta_dark, _ = decompose_delta(delta, U_wu)
                if np.linalg.norm(delta_dark) > 1e-8:
                    all_dark_by_group[l][concept_group].append(delta_dark)

        print(f"  Collected: {cname}", end="\r")
        del concept_states
        gc.collect()

    print(f"\n  All concepts collected.")

    results = {"model": model_name}

    # 对每层: 计算组间PC1余弦相似度 + 子空间距离
    group_names = list(SEMANTIC_GROUPS.keys())

    for l in key_layers:
        # 计算每组的PC1和前k个主成分
        group_pcs = {}
        group_centroids = {}

        for g in group_names:
            vecs = all_dark_by_group[l][g]
            if len(vecs) < 3:
                continue

            X = np.array(vecs)
            group_centroids[g] = X.mean(axis=0)
            group_pcs[g] = compute_pca_and_get_PC1(vecs)

        if len(group_pcs) < 2:
            continue

        # PC1余弦相似度矩阵
        pc1_sim = {}
        for g1 in group_pcs:
            for g2 in group_pcs:
                if g1 >= g2:
                    continue
                cos_sim = abs(np.dot(group_pcs[g1], group_pcs[g2]) /
                             (np.linalg.norm(group_pcs[g1]) * np.linalg.norm(group_pcs[g2]) + 1e-12))
                pc1_sim[f"{g1}_vs_{g2}"] = float(cos_sim)
                print(f"  L{l} PC1 cos({g1}, {g2}) = {cos_sim:.4f}")

        # 质心余弦相似度
        centroid_sim = {}
        for g1 in group_centroids:
            for g2 in group_centroids:
                if g1 >= g2:
                    continue
                cos_sim = abs(np.dot(group_centroids[g1], group_centroids[g2]) /
                             (np.linalg.norm(group_centroids[g1]) * np.linalg.norm(group_centroids[g2]) + 1e-12))
                centroid_sim[f"{g1}_vs_{g2}"] = float(cos_sim)

        # 子空间正交性 (Grassmann距离近似)
        subspace_dist = {}
        for g1 in group_names:
            for g2 in group_names:
                if g1 >= g2:
                    continue
                v1 = all_dark_by_group[l][g1]
                v2 = all_dark_by_group[l][g2]
                if len(v1) < 3 or len(v2) < 3:
                    continue

                # 取前5个主成分构成子空间
                k = min(5, len(v1)-1, len(v2)-1)
                if k < 1:
                    continue

                X1 = np.array(v1)
                X2 = np.array(v2)
                X1_c = X1 - X1.mean(axis=0, keepdims=True)
                X2_c = X2 - X2.mean(axis=0, keepdims=True)

                _, s1, Vt1 = np.linalg.svd(X1_c, full_matrices=False)
                _, s2, Vt2 = np.linalg.svd(X2_c, full_matrices=False)
                U1 = Vt1[:k]  # k × d
                U2 = Vt2[:k]  # k × d

                # 子空间距离: U1(k,d), U2(k,d) → 正交投影重叠
                # P1 = U1^T U1 (d×d), P2 = U2^T U2 (d×d)
                # overlap = ||P1 P2||_F^2 / k = ||U1^T U1 U2^T U2||_F^2 / k
                P1 = U1.T @ U1  # d × d
                P2 = U2.T @ U2  # d × d
                overlap = np.linalg.norm(P1 @ P2, 'fro')**2 / k
                subspace_dist[f"{g1}_vs_{g2}"] = float(overlap)

        # 零假设测试: 随机分组的PC1对齐度
        all_vecs = []
        for g in group_names:
            all_vecs.extend(all_dark_by_group[l][g])

        if len(all_vecs) >= 10:
            n_rand = 100
            rand_pc1_cos = []
            np.random.seed(42)
            for _ in range(n_rand):
                idx = np.random.permutation(len(all_vecs))
                mid = len(all_vecs) // 2
                g1_vecs = [all_vecs[i] for i in idx[:mid]]
                g2_vecs = [all_vecs[i] for i in idx[mid:]]

                if len(g1_vecs) < 3 or len(g2_vecs) < 3:
                    continue

                pc1_a = compute_pca_and_get_PC1(g1_vecs)
                pc1_b = compute_pca_and_get_PC1(g2_vecs)
                cos_sim = abs(np.dot(pc1_a, pc1_b) /
                             (np.linalg.norm(pc1_a) * np.linalg.norm(pc1_b) + 1e-12))
                rand_pc1_cos.append(cos_sim)

            rand_mean = np.mean(rand_pc1_cos)
            rand_std = np.std(rand_pc1_cos)

            # 检验实际PC1对齐是否显著低于随机
            actual_pc1_cos = [v for v in pc1_sim.values()]
            actual_mean = np.mean(actual_pc1_cos) if actual_pc1_cos else 0

            results[f"null_test_L{l}"] = {
                "random_pc1_cos_mean": float(rand_mean),
                "random_pc1_cos_std": float(rand_std),
                "actual_pc1_cos_mean": float(actual_mean),
                "significance": "orthogonal" if actual_mean < rand_mean - 2*rand_std else "not_significant",
            }
            print(f"  L{l} null test: random_mean={rand_mean:.4f}, actual_mean={actual_mean:.4f}, "
                  f"significance={results[f'null_test_L{l}']['significance']}")

        results[f"pc1_sim_L{l}"] = pc1_sim
        results[f"centroid_sim_L{l}"] = centroid_sim
        results[f"subspace_overlap_L{l}"] = subspace_dist

    # 保存
    out_path = TEMP / f"ccxx_exp2_{model_name}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")

    release_model(model)
    return results


# ================================================================
# Exp3: 增量维度曲线的精细分析
# ================================================================
def run_exp3(model_name):
    """增量维度曲线 + 饱和点检测 + 与线性/对数/幂律拟合"""
    print(f"\n{'='*70}")
    print(f"  Exp3: Incremental Dimensionality Curve ({model_name})")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    key_layers = [l for l in [12, 24, 36] if l < n_layers]

    U_wu = compute_wu_basis(model)
    bl_all, _ = collect_states_at_layers(model, tokenizer, device, BASELINE_TEXT, key_layers)

    # 收集所有暗物质向量
    all_dark = {l: {} for l in key_layers}

    for cname, templates in CONCEPTS_50.items():
        concept_states = {l: [] for l in key_layers}
        for template in templates:
            states, _ = collect_states_at_layers(model, tokenizer, device, template, key_layers)
            for l in key_layers:
                if l in states and l in bl_all:
                    concept_states[l].append(states[l])

        for l in key_layers:
            if concept_states[l]:
                delta = np.mean(concept_states[l], axis=0) - bl_all[l]
                _, delta_dark, _ = decompose_delta(delta, U_wu)
                if np.linalg.norm(delta_dark) > 1e-8:
                    all_dark[l][cname] = delta_dark

        print(f"  Collected: {cname}", end="\r")
        del concept_states
        gc.collect()

    print(f"\n  All concepts collected.")

    results = {"model": model_name}

    for l in key_layers:
        dark_vecs = list(all_dark[l].values())
        n_total = len(dark_vecs)
        if n_total < 5:
            continue

        # 精细增量曲线: 每2个概念一个点
        scaling_data = []
        np.random.seed(42)
        indices = list(range(n_total))
        np.random.shuffle(indices)

        for n_concepts in list(range(3, n_total+1, 2)) + [n_total]:
            if n_concepts > n_total:
                continue
            subset = [dark_vecs[i] for i in indices[:n_concepts]]
            stats = compute_pca_stats(subset)
            if stats:
                scaling_data.append({
                    "n_concepts": n_concepts,
                    "eff_rank": stats["effective_rank"],
                    "n_90": stats["n_for_90"],
                    "n_95": stats["n_for_95"],
                    "var_ratio_top3": stats["var_ratio_top10"][:3],
                })

        # 拟合三种模型
        ns = np.array([d["n_concepts"] for d in scaling_data])
        ers = np.array([d["eff_rank"] for d in scaling_data])

        fits = {}

        # 线性: eff_rank = a * n + b
        if len(ns) >= 3:
            coeffs_lin = np.polyfit(ns, ers, 1)
            pred_lin = np.polyval(coeffs_lin, ns)
            ss_res = np.sum((ers - pred_lin)**2)
            ss_tot = np.sum((ers - ers.mean())**2)
            r2_lin = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            fits["linear"] = {"a": float(coeffs_lin[0]), "b": float(coeffs_lin[1]), "R2": float(r2_lin)}

        # 对数: eff_rank = a * log(n) + b
        if len(ns) >= 3 and all(n > 1 for n in ns):
            log_ns = np.log(ns)
            coeffs_log = np.polyfit(log_ns, ers, 1)
            pred_log = np.polyval(coeffs_log, log_ns)
            ss_res = np.sum((ers - pred_log)**2)
            r2_log = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            fits["log"] = {"a": float(coeffs_log[0]), "b": float(coeffs_log[1]), "R2": float(r2_log)}

        # 幂律: eff_rank = a * n^b
        if len(ns) >= 3 and all(n > 0 for n in ns) and all(e > 0 for e in ers):
            try:
                log_ns_p = np.log(ns)
                log_ers = np.log(ers)
                coeffs_pow = np.polyfit(log_ns_p, log_ers, 1)
                b_pow = coeffs_pow[0]
                a_pow = np.exp(coeffs_pow[1])
                pred_pow = a_pow * ns**b_pow
                ss_res = np.sum((ers - pred_pow)**2)
                r2_pow = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                fits["power"] = {"a": float(a_pow), "b": float(b_pow), "R2": float(r2_pow)}
            except:
                pass

        # 饱和: eff_rank = D * (1 - exp(-n/D)) (渐进饱和)
        if len(ns) >= 5:
            from scipy.optimize import curve_fit
            try:
                def sat_func(n, D):
                    return D * (1 - np.exp(-n / D))
                popt, _ = curve_fit(sat_func, ns.astype(float), ers.astype(float),
                                    p0=[max(ers)*2], maxfev=5000)
                D_sat = popt[0]
                pred_sat = sat_func(ns, D_sat)
                ss_res = np.sum((ers - pred_sat)**2)
                r2_sat = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                fits["saturation"] = {"D_asymptote": float(D_sat), "R2": float(r2_sat)}
            except:
                pass

        # 找最佳拟合
        best_fit = max(fits.keys(), key=lambda k: fits[k]["R2"]) if fits else "none"

        results[f"scaling_L{l}"] = scaling_data
        results[f"fits_L{l}"] = fits
        results[f"best_fit_L{l}"] = best_fit

        print(f"\n  L{l} scaling curve:")
        for d in scaling_data:
            print(f"    n={d['n_concepts']:2d}, eff_rank={d['eff_rank']:.1f}, n_90={d['n_90']}")
        print(f"  Best fit: {best_fit}")
        for fname, fdata in fits.items():
            print(f"    {fname}: R2={fdata['R2']:.4f}, params={ {k:v for k,v in fdata.items() if k!='R2'} }")

    # 保存
    out_path = TEMP / f"ccxx_exp3_{model_name}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")

    release_model(model)
    return results


# ================================================================
# Exp4: 跨模型正交性比较 (需3个模型都已运行Exp2)
# ================================================================
def run_exp4():
    """跨模型正交性比较 — 验证分块正交是否跨模型一致"""
    print(f"\n{'='*70}")
    print(f"  Exp4: Cross-Model Orthogonality Comparison")
    print(f"{'='*70}")

    models = ["qwen3", "glm4", "deepseek7b"]
    exp2_data = {}

    for mn in models:
        path = TEMP / f"ccxx_exp2_{mn}_results.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                exp2_data[mn] = json.load(f)
        else:
            print(f"  WARNING: {mn} Exp2 data not found at {path}")

    if len(exp2_data) < 2:
        print("  Need at least 2 models with Exp2 data. Run Exp2 first.")
        return None

    results = {"models_compared": list(exp2_data.keys())}

    # 比较PC1余弦相似度跨模型
    for l in [6, 12, 18, 24, 30]:
        key = f"pc1_sim_L{l}"
        model_pc1s = {}
        for mn, data in exp2_data.items():
            if key in data:
                model_pc1s[mn] = data[key]

        if len(model_pc1s) < 2:
            continue

        # 跨模型比较同组的PC1对齐值
        all_pairs = set()
        for mn in model_pc1s:
            all_pairs.update(model_pc1s[mn].keys())

        pair_comparison = {}
        for pair in sorted(all_pairs):
            values = {mn: model_pc1s[mn].get(pair, None) for mn in model_pc1s}
            valid = {k: v for k, v in values.items() if v is not None}
            if len(valid) >= 2:
                pair_comparison[pair] = valid

        results[f"pc1_cross_model_L{l}"] = pair_comparison

        print(f"\n  L{l} cross-model PC1 cos comparison:")
        for pair, vals in pair_comparison.items():
            vals_str = ", ".join(f"{mn}={v:.4f}" for mn, v in vals.items())
            # 跨模型变异系数
            v_arr = list(vals.values())
            cv = np.std(v_arr) / np.mean(v_arr) if np.mean(v_arr) > 0 else 0
            print(f"    {pair}: {vals_str} (CV={cv:.3f})")

    # 统计: 跨模型PC1对齐是否一致
    all_cv = []
    for l in [6, 12, 18, 24, 30]:
        key = f"pc1_cross_model_L{l}"
        if key in results:
            for pair, vals in results[key].items():
                v_arr = list(vals.values())
                if len(v_arr) >= 2 and np.mean(v_arr) > 0.01:
                    cv = np.std(v_arr) / np.mean(v_arr)
                    all_cv.append(cv)

    if all_cv:
        results["cross_model_cv_stats"] = {
            "mean": float(np.mean(all_cv)),
            "std": float(np.std(all_cv)),
            "median": float(np.median(all_cv)),
        }
        print(f"\n  Cross-model CV: mean={np.mean(all_cv):.3f}, median={np.median(all_cv):.3f}")
        print(f"  → {'Consistent across models' if np.median(all_cv) < 0.3 else 'Inconsistent across models'}")

    # null test比较
    print(f"\n  Null test comparison:")
    for l in [6, 12, 18, 24]:
        null_data = {}
        for mn, data in exp2_data.items():
            nkey = f"null_test_L{l}"
            if nkey in data:
                null_data[mn] = data[nkey]

        if len(null_data) >= 2:
            results[f"null_test_cross_L{l}"] = null_data
            for mn, nd in null_data.items():
                print(f"    {mn} L{l}: random={nd['random_pc1_cos_mean']:.4f}, "
                      f"actual={nd['actual_pc1_cos_mean']:.4f}, "
                      f"significance={nd['significance']}")

    out_path = TEMP / "ccxx_exp4_cross_model_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")

    return results


# ================================================================
# 主入口
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3",
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="1",
                       choices=["1", "2", "3", "4", "all"])
    args = parser.parse_args()

    TEMP.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.exp == "all":
        r1 = run_exp1(args.model)
        r2 = run_exp2(args.model)
        r3 = run_exp3(args.model)
    elif args.exp == "1":
        run_exp1(args.model)
    elif args.exp == "2":
        run_exp2(args.model)
    elif args.exp == "3":
        run_exp3(args.model)
    elif args.exp == "4":
        run_exp4()

    print(f"\n  Total time: {time.time()-t0:.1f}s")
