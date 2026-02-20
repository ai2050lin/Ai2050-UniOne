# -*- coding: utf-8 -*-
"""
Qwen3 编码结构四维度提取器
=========================
从 Qwen3-4B 中提取编码，验证四个关键数学特性：
  1. 高维抽象 — 语义收敛能力
  2. 低维精确 — 细粒度区分能力
  3. 特异性 — 概念子空间正交性
  4. 系统性 — 类比关系一致性

输出: tempdata/qwen3_structure_report.json + 4 张可视化图
"""

import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")  # 无头模式，兼容服务器
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 第零部分：模型加载（复用已验证的 import_trace.py 逻辑）
# ============================================================

SNAPSHOT_PATH = r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"

# 环境变量
os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"


def load_qwen3():
    """加载 Qwen3-4B 为 HookedTransformer"""
    import transformers.configuration_utils as config_utils
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from transformer_lens import HookedTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] 加载 Qwen3-4B，设备: {device}")
    print(f"    路径: {SNAPSHOT_PATH}")

    t0 = time.time()

    # 步骤 1: 在 CPU 上加载 HF 模型 (HookedTransformer 会自行处理设备迁移)
    hf_model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT_PATH, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        SNAPSHOT_PATH, local_files_only=True, add_bos_token=False
    )

    # 修复1: Qwen3 tokenizer 缺少 bos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
        tokenizer.bos_token_id = tokenizer.eos_token_id
        print(f"    [fix] 设置 bos_token = eos_token ({tokenizer.bos_token})")

    # 修复2: Monkey-patch PretrainedConfig 以修复 rope_theta
    _orig_getattr = config_utils.PretrainedConfig.__getattribute__

    def _patched_getattr(self, key):
        if key == "rope_theta":
            try:
                return _orig_getattr(self, key)
            except AttributeError:
                try:
                    rs = _orig_getattr(self, "rope_scaling")
                    if isinstance(rs, dict) and "rope_theta" in rs:
                        return rs["rope_theta"]
                except (AttributeError, TypeError):
                    pass
                return 1000000
        return _orig_getattr(self, key)

    config_utils.PretrainedConfig.__getattribute__ = _patched_getattr

    # 修复3: Monkey-patch get_tokenizer_with_bos 避免重新加载 tokenizer
    import transformer_lens.utils as tl_utils
    _orig_get_tok_bos = tl_utils.get_tokenizer_with_bos

    def _patched_get_tok_bos(tok):
        # 直接返回已修复的 tokenizer，避免重新 from_pretrained
        return tok

    tl_utils.get_tokenizer_with_bos = _patched_get_tok_bos
    print("    [fix] 已 monkey-patch rope_theta + get_tokenizer_with_bos")

    try:
        model = HookedTransformer.from_pretrained(
            "Qwen/Qwen3-4B", hf_model=hf_model, device=device, tokenizer=tokenizer,
            fold_ln=False, center_writing_weights=False, center_unembed=False,
            dtype=torch.float16, default_prepend_bos=False
        )
    finally:
        config_utils.PretrainedConfig.__getattribute__ = _orig_getattr
        tl_utils.get_tokenizer_with_bos = _orig_get_tok_bos
        print("    [fix] 已恢复所有 monkey-patch")

    model.eval()
    print(f"[+] 模型加载完成 ({time.time() - t0:.1f}s)")
    print(f"    层数: {model.cfg.n_layers}, 维度: {model.cfg.d_model}")
    return model


def extract_activations(model, prompts, token_idx=-1):
    """
    批量提取指定 token 位置的各层 residual stream 激活。
    返回: dict[layer_idx] -> np.ndarray [N_prompts, d_model]
    """
    n_layers = model.cfg.n_layers
    act_dict = {i: [] for i in range(n_layers)}

    # 小批量处理，避免显存溢出
    batch_size = 4
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        with torch.no_grad():
            _, cache = model.run_with_cache(batch)
            for layer in range(n_layers):
                acts = cache[f"blocks.{layer}.hook_resid_post"]
                acts = acts[:, token_idx, :].cpu().float().numpy()
                act_dict[layer].append(acts)
            del cache
            torch.cuda.empty_cache()

    for i in range(n_layers):
        act_dict[i] = np.concatenate(act_dict[i], axis=0)

    return act_dict


# ============================================================
# 维度 1: 高维抽象 — 语义收敛检测
# ============================================================

ABSTRACTION_PROMPTS = {
    "chase": [
        "A dog chased a cat across the yard",
        "The puppy ran after the kitten quickly",
        "The canine pursued the feline through the grass",
    ],
    "sunrise": [
        "The sun rose in the east this morning",
        "Dawn broke over the distant horizon",
        "Sunrise illuminated the entire sky beautifully",
    ],
    "cooking": [
        "She cooked dinner in the kitchen tonight",
        "The chef prepared a meal using fresh ingredients",
        "Food was being made on the stove carefully",
    ],
    "reading": [
        "He read a book in the library quietly",
        "The student studied the text very carefully",
        "She was absorbed in the written pages deeply",
    ],
}


def analyze_abstraction(model):
    """维度1: 高维抽象分析"""
    print("\n" + "=" * 60)
    print("[维度1] 高维抽象 — 语义收敛检测")
    print("=" * 60)

    # 收集所有 prompt 和标签
    all_prompts = []
    group_labels = []
    for group_name, prompts in ABSTRACTION_PROMPTS.items():
        for p in prompts:
            all_prompts.append(p)
            group_labels.append(group_name)

    acts_by_layer = extract_activations(model, all_prompts)
    n_layers = len(acts_by_layer)
    group_names = list(ABSTRACTION_PROMPTS.keys())

    abstraction_scores = []

    for layer in range(n_layers):
        acts = acts_by_layer[layer]
        sim_matrix = cosine_similarity(acts)

        # 组内相似度（同语义，不同表面）
        intra_sims = []
        inter_sims = []
        for i in range(len(all_prompts)):
            for j in range(i + 1, len(all_prompts)):
                if group_labels[i] == group_labels[j]:
                    intra_sims.append(sim_matrix[i, j])
                else:
                    inter_sims.append(sim_matrix[i, j])

        intra_mean = np.mean(intra_sims)
        inter_mean = np.mean(inter_sims)
        ratio = intra_mean / (inter_mean + 1e-9)
        abstraction_scores.append({
            "layer": layer,
            "intra_sim": float(intra_mean),
            "inter_sim": float(inter_mean),
            "ratio": float(ratio),
        })

    # 打印关键层
    print(f"\n{'层':>4} | {'组内相似':>10} | {'组间相似':>10} | {'抽象指标':>10}")
    print("-" * 50)
    for s in abstraction_scores:
        if s["layer"] % 4 == 0 or s["layer"] == n_layers - 1:
            print(f"{s['layer']:>4} | {s['intra_sim']:>10.4f} | {s['inter_sim']:>10.4f} | {s['ratio']:>10.4f}")

    # PCA 可视化 (浅层 vs 深层)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_idx, layer_idx in enumerate([0, n_layers - 1]):
        acts = acts_by_layer[layer_idx]
        pca = PCA(n_components=2)
        proj = pca.fit_transform(acts)
        ax = axes[ax_idx]
        colors = {"chase": "red", "sunrise": "orange", "cooking": "green", "reading": "blue"}
        for i, label in enumerate(group_labels):
            ax.scatter(proj[i, 0], proj[i, 1], c=colors[label], s=100, label=label if i < 4 else "")
        ax.set_title(f"Layer {layer_idx}", fontsize=14)
        ax.legend()
    fig.suptitle("Dimension 1: Abstraction (Shallow vs Deep)", fontsize=16)
    plt.tight_layout()
    os.makedirs("tempdata", exist_ok=True)
    plt.savefig("tempdata/qwen3_abstraction_curve.png", dpi=150)
    plt.close()
    print("[+] 已保存 tempdata/qwen3_abstraction_curve.png")

    return abstraction_scores


# ============================================================
# 维度 2: 低维精确 — 细粒度区分能力
# ============================================================

PRECISION_PAIRS = [
    # (正确/正面, 错误/负面)
    ("2 + 3 = 5", "2 + 3 = 6"),
    ("7 * 8 = 56", "7 * 8 = 54"),
    ("The capital of France is Paris", "The capital of France is London"),
    ("Water boils at 100 degrees", "Water boils at 50 degrees"),
    ("The cat is alive", "The cat is not alive"),
    ("I agree with you", "I disagree with you"),
    ("The light is on", "The light is off"),
    ("She is happy today", "She is sad today"),
]


def analyze_precision(model):
    """维度2: 低维精确分析"""
    print("\n" + "=" * 60)
    print("[维度2] 低维精确 — 细粒度区分能力")
    print("=" * 60)

    all_prompts = []
    labels = []  # 0 = 正确/正面, 1 = 错误/负面
    for pos, neg in PRECISION_PAIRS:
        all_prompts.append(pos)
        labels.append(0)
        all_prompts.append(neg)
        labels.append(1)

    acts_by_layer = extract_activations(model, all_prompts)
    n_layers = len(acts_by_layer)
    labels_arr = np.array(labels)

    precision_results = []

    for layer in range(n_layers):
        acts = acts_by_layer[layer]

        # 对不同投影维度训练线性探针
        dim_accuracies = {}
        n_samples = acts.shape[0]
        for k in [1, 2, 4, 8, 16, 32]:
            if k >= n_samples or k > acts.shape[1]:
                break
            pca = PCA(n_components=k)
            proj = pca.fit_transform(acts)
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(proj, labels_arr)
            acc = clf.score(proj, labels_arr)
            dim_accuracies[k] = float(acc)

        # 找到 >=95% 的最小维度
        min_k = None
        for k, acc in sorted(dim_accuracies.items()):
            if acc >= 0.95:
                min_k = k
                break

        precision_results.append({
            "layer": layer,
            "dim_accuracies": dim_accuracies,
            "min_separable_dim": min_k,
        })

    # 打印结果
    print(f"\n{'层':>4} | {'k=1':>6} | {'k=2':>6} | {'k=4':>6} | {'k=8':>6} | {'k=16':>6} | {'最小k':>6}")
    print("-" * 60)
    for r in precision_results:
        if r["layer"] % 4 == 0 or r["layer"] == n_layers - 1:
            da = r["dim_accuracies"]
            print(f"{r['layer']:>4} | {da.get(1,0):>6.2f} | {da.get(2,0):>6.2f} | {da.get(4,0):>6.2f} | {da.get(8,0):>6.2f} | {da.get(16,0):>6.2f} | {str(r['min_separable_dim']):>6}")

    # 可视化: 线性探针准确率热力图
    fig, ax = plt.subplots(figsize=(10, 6))
    dims = [1, 2, 4, 8, 16]
    heatmap_data = []
    for r in precision_results:
        row = [r["dim_accuracies"].get(k, 0) for k in dims]
        heatmap_data.append(row)
    heatmap_data = np.array(heatmap_data)
    im = ax.imshow(heatmap_data.T, aspect="auto", cmap="YlOrRd", vmin=0.5, vmax=1.0)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Projection Dim (k)", fontsize=12)
    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels(dims)
    ax.set_title("Dimension 2: Precision (Linear Probe Accuracy)", fontsize=14)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("tempdata/qwen3_precision_probe.png", dpi=150)
    plt.close()
    print("[+] 已保存 tempdata/qwen3_precision_probe.png")

    return precision_results


# ============================================================
# 维度 3: 特异性 — 概念子空间正交性
# ============================================================

SPECIFICITY_GROUPS = {
    "animals": ["cat", "dog", "bird", "fish", "horse"],
    "colors": ["red", "blue", "green", "yellow", "black"],
    "numbers": ["one", "two", "three", "four", "five"],
    "emotions": ["happy", "sad", "angry", "calm", "excited"],
    "spatial": ["up", "down", "left", "right", "center"],
    "temporal": ["past", "present", "future", "always", "never"],
}


def analyze_specificity(model):
    """维度3: 特异性分析"""
    print("\n" + "=" * 60)
    print("[维度3] 特异性 — 概念子空间正交性")
    print("=" * 60)

    # 将每个概念包装成短句以获得更好的上下文激活
    all_prompts = []
    group_labels = []
    group_names = list(SPECIFICITY_GROUPS.keys())

    for g_name, concepts in SPECIFICITY_GROUPS.items():
        for c in concepts:
            all_prompts.append(f"The concept of {c}")
            group_labels.append(g_name)

    acts_by_layer = extract_activations(model, all_prompts)
    n_layers = len(acts_by_layer)

    specificity_results = []

    # 选几个关键层进行分析
    key_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]

    for layer in range(n_layers):
        acts = acts_by_layer[layer]
        sim_matrix = cosine_similarity(acts)

        # 组内 vs 组间相似度
        intra_sims = []
        inter_sims = []
        for i in range(len(all_prompts)):
            for j in range(i + 1, len(all_prompts)):
                if group_labels[i] == group_labels[j]:
                    intra_sims.append(sim_matrix[i, j])
                else:
                    inter_sims.append(sim_matrix[i, j])

        # SVD 子空间正交性分析
        # 对每个概念组计算主方向，检查不同组的主方向是否正交
        group_directions = {}
        for g_name in group_names:
            indices = [i for i, g in enumerate(group_labels) if g == g_name]
            group_acts = acts[indices]
            centered = group_acts - group_acts.mean(axis=0)
            if centered.shape[0] > 1:
                u, s, vh = np.linalg.svd(centered, full_matrices=False)
                group_directions[g_name] = vh[0]  # 第一主方向

        # 计算主方向间的绝对余弦值（正交性: 接近 0 为好）
        orthogonality_scores = []
        dir_names = list(group_directions.keys())
        for i in range(len(dir_names)):
            for j in range(i + 1, len(dir_names)):
                cos_val = abs(np.dot(group_directions[dir_names[i]], group_directions[dir_names[j]]))
                orthogonality_scores.append(cos_val)

        avg_orthogonality = 1.0 - np.mean(orthogonality_scores)  # 接近 1 = 更正交

        specificity_results.append({
            "layer": layer,
            "intra_sim": float(np.mean(intra_sims)),
            "inter_sim": float(np.mean(inter_sims)),
            "orthogonality": float(avg_orthogonality),
        })

    # 打印
    print(f"\n{'层':>4} | {'组内相似':>10} | {'组间相似':>10} | {'正交性':>10}")
    print("-" * 50)
    for r in specificity_results:
        if r["layer"] % 4 == 0 or r["layer"] == n_layers - 1:
            print(f"{r['layer']:>4} | {r['intra_sim']:>10.4f} | {r['inter_sim']:>10.4f} | {r['orthogonality']:>10.4f}")

    # 可视化: 选择深层的相似度热力图
    deep_layer = n_layers - 1
    acts = acts_by_layer[deep_layer]
    sim_matrix = cosine_similarity(acts)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-0.2, vmax=1.0)

    # 画组分割线
    cumulative = 0
    for g_name in group_names:
        size = len(SPECIFICITY_GROUPS[g_name])
        ax.axhline(cumulative - 0.5, color="black", linewidth=0.5)
        ax.axvline(cumulative - 0.5, color="black", linewidth=0.5)
        ax.text(cumulative + size / 2, -1.5, g_name, ha="center", fontsize=8)
        cumulative += size

    ax.set_title(f"Dimension 3: Specificity - Concept Similarity (Layer {deep_layer})", fontsize=13)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("tempdata/qwen3_specificity_heatmap.png", dpi=150)
    plt.close()
    print("[+] 已保存 tempdata/qwen3_specificity_heatmap.png")

    return specificity_results


# ============================================================
# 维度 4: 系统性 — 类比关系一致性
# ============================================================

ANALOGY_GROUPS = {
    "gender": [
        ("king", "queen"), ("man", "woman"), ("boy", "girl"),
        ("father", "mother"), ("son", "daughter"),
    ],
    "capital": [
        ("France", "Paris"), ("Japan", "Tokyo"), ("Germany", "Berlin"),
        ("Italy", "Rome"), ("China", "Beijing"),
    ],
    "antonym": [
        ("hot", "cold"), ("tall", "short"), ("fast", "slow"),
        ("big", "small"), ("light", "dark"),
    ],
    "young": [
        ("dog", "puppy"), ("cat", "kitten"), ("cow", "calf"),
        ("horse", "foal"), ("goat", "kid"),
    ],
}


def analyze_systematicity(model):
    """维度4: 系统性分析"""
    print("\n" + "=" * 60)
    print("[维度4] 系统性 — 类比关系一致性")
    print("=" * 60)

    # 收集所有概念词
    all_words = set()
    for group_pairs in ANALOGY_GROUPS.values():
        for a, b in group_pairs:
            all_words.add(a)
            all_words.add(b)
    all_words = sorted(all_words)
    word_prompts = [f"The word {w}" for w in all_words]
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    acts_by_layer = extract_activations(model, word_prompts)
    n_layers = len(acts_by_layer)

    systematicity_results = []

    for layer in range(n_layers):
        acts = acts_by_layer[layer]

        # 计算每组关系的差异向量
        group_consistency = {}
        for group_name, pairs in ANALOGY_GROUPS.items():
            diff_vectors = []
            for a, b in pairs:
                idx_a = word_to_idx[a]
                idx_b = word_to_idx[b]
                diff = acts[idx_b] - acts[idx_a]
                diff_vectors.append(diff / (np.linalg.norm(diff) + 1e-9))

            # 组内差异向量一致性: 任意两个差异向量的余弦相似度
            pairwise_cos = []
            for i in range(len(diff_vectors)):
                for j in range(i + 1, len(diff_vectors)):
                    cos_val = np.dot(diff_vectors[i], diff_vectors[j])
                    pairwise_cos.append(cos_val)

            group_consistency[group_name] = float(np.mean(pairwise_cos))

        # 类比完成准确率测试 (A:B = C:?)
        analogy_correct = 0
        analogy_total = 0

        for group_name, pairs in ANALOGY_GROUPS.items():
            if len(pairs) < 2:
                continue
            for i in range(len(pairs)):
                for j in range(len(pairs)):
                    if i == j:
                        continue
                    a1, b1 = pairs[i]
                    a2, b2_true = pairs[j]

                    # 预测: b2_pred = a2 + (b1 - a1)
                    idx_a1 = word_to_idx[a1]
                    idx_b1 = word_to_idx[b1]
                    idx_a2 = word_to_idx[a2]
                    idx_b2_true = word_to_idx[b2_true]

                    pred_vec = acts[idx_a2] + (acts[idx_b1] - acts[idx_a1])

                    # 在所有候选 b 中找最近的
                    best_sim = -1
                    best_idx = -1
                    for candidate_b in [p[1] for p in pairs]:
                        c_idx = word_to_idx[candidate_b]
                        sim = np.dot(pred_vec, acts[c_idx]) / (
                            np.linalg.norm(pred_vec) * np.linalg.norm(acts[c_idx]) + 1e-9
                        )
                        if sim > best_sim:
                            best_sim = sim
                            best_idx = c_idx

                    if best_idx == idx_b2_true:
                        analogy_correct += 1
                    analogy_total += 1

        analogy_accuracy = analogy_correct / (analogy_total + 1e-9)

        systematicity_results.append({
            "layer": layer,
            "group_consistency": group_consistency,
            "analogy_accuracy": float(analogy_accuracy),
        })

    # 打印
    print(f"\n{'层':>4} | {'gender':>8} | {'capital':>8} | {'antonym':>8} | {'young':>8} | {'类比准确率':>10}")
    print("-" * 60)
    for r in systematicity_results:
        if r["layer"] % 4 == 0 or r["layer"] == n_layers - 1:
            gc = r["group_consistency"]
            print(f"{r['layer']:>4} | {gc.get('gender', 0):>8.4f} | {gc.get('capital', 0):>8.4f} | {gc.get('antonym', 0):>8.4f} | {gc.get('young', 0):>8.4f} | {r['analogy_accuracy']:>10.2%}")

    # 可视化: 系统性得分随层变化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    layers = [r["layer"] for r in systematicity_results]
    for group_name in ANALOGY_GROUPS:
        vals = [r["group_consistency"][group_name] for r in systematicity_results]
        ax1.plot(layers, vals, label=group_name, linewidth=2)
    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("Direction Consistency", fontsize=12)
    ax1.set_title("Relation Vector Consistency", fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)

    accuracies = [r["analogy_accuracy"] for r in systematicity_results]
    ax2.plot(layers, accuracies, color="red", linewidth=2)
    ax2.axhline(0.7, color="green", linestyle="--", alpha=0.5, label="目标: 70%")
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_ylabel("Analogy Accuracy", fontsize=12)
    ax2.set_title("Analogy Completion Accuracy", fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("Dimension 4: Systematicity", fontsize=16)
    plt.tight_layout()
    plt.savefig("tempdata/qwen3_systematicity_graph.png", dpi=150)
    plt.close()
    print("[+] 已保存 tempdata/qwen3_systematicity_graph.png")

    return systematicity_results


# ============================================================
# 汇总报告
# ============================================================

def generate_report(abstraction, precision, specificity, systematicity):
    """生成最终结构报告"""
    n_layers = len(abstraction)

    # 判断通过标准
    deep_layer = n_layers - 1

    # 1. 抽象: 深层 ratio > 2.0
    abs_pass = abstraction[deep_layer]["ratio"] > 2.0
    abs_best_ratio = max(s["ratio"] for s in abstraction)

    # 2. 精确: 最小可分离维度 <= 4
    prec_pass = any(r["min_separable_dim"] is not None and r["min_separable_dim"] <= 4
                   for r in precision)
    prec_best_k = min(
        (r["min_separable_dim"] for r in precision if r["min_separable_dim"] is not None),
        default=None
    )

    # 3. 特异性: 深层正交性 > 0.7
    spec_pass = specificity[deep_layer]["orthogonality"] > 0.7
    spec_best_orth = max(s["orthogonality"] for s in specificity)

    # 4. 系统性: 最佳类比准确率 > 0.7
    sys_best_acc = max(r["analogy_accuracy"] for r in systematicity)
    sys_pass = sys_best_acc > 0.7

    report = {
        "model": "Qwen3-4B",
        "n_layers": n_layers,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "abstraction": {
                "passed": abs_pass,
                "best_ratio": float(abs_best_ratio),
                "criterion": "ratio > 2.0 in deep layers",
            },
            "precision": {
                "passed": prec_pass,
                "min_separable_dim": prec_best_k,
                "criterion": "min k <= 4 for 95% accuracy",
            },
            "specificity": {
                "passed": spec_pass,
                "best_orthogonality": float(spec_best_orth),
                "criterion": "orthogonality > 0.7 in deep layers",
            },
            "systematicity": {
                "passed": sys_pass,
                "best_analogy_accuracy": float(sys_best_acc),
                "criterion": "analogy accuracy > 70%",
            },
            "all_passed": all([abs_pass, prec_pass, spec_pass, sys_pass]),
        },
        "detail": {
            "abstraction": abstraction,
            "precision": precision,
            "specificity": specificity,
            "systematicity": systematicity,
        },
    }

    # 保存报告
    os.makedirs("tempdata", exist_ok=True)
    report_path = "tempdata/qwen3_structure_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 打印总结
    print("\n" + "=" * 60)
    print("                 结构提取总报告")
    print("=" * 60)
    print(f"模型: Qwen3-4B ({n_layers} 层)")
    print(f"时间: {report['timestamp']}")
    print()
    for dim_name, result in report["summary"].items():
        if dim_name == "all_passed":
            continue
        status = "✅ 通过" if result["passed"] else "❌ 未通过"
        print(f"  {dim_name:15s}: {status}")
    print()
    verdict = "✅ 全部通过 — 统一数学结构假说获得实证支撑" if report["summary"]["all_passed"] else "⚠️ 部分未通过 — 需进一步分析"
    print(f"  最终判定: {verdict}")
    print(f"\n[+] 完整报告已保存: {report_path}")

    return report


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("  Qwen3 编码结构四维度提取器 v1.0")
    print("  目标: 验证高维抽象 / 低维精确 / 特异性 / 系统性")
    print("=" * 60)

    t_start = time.time()

    # 1. 加载模型
    model = load_qwen3()

    # 2. 运行四个维度分析
    print(f"\n[进度 1/4] 高维抽象分析...")
    abstraction = analyze_abstraction(model)

    print(f"\n[进度 2/4] 低维精确分析...")
    precision = analyze_precision(model)

    print(f"\n[进度 3/4] 特异性分析...")
    specificity = analyze_specificity(model)

    print(f"\n[进度 4/4] 系统性分析...")
    systematicity = analyze_systematicity(model)

    # 3. 生成报告
    report = generate_report(abstraction, precision, specificity, systematicity)

    elapsed = time.time() - t_start
    print(f"\n[*] 全流程耗时: {elapsed:.1f}s ({elapsed/60:.1f} 分钟)")

    return report


if __name__ == "__main__":
    main()
