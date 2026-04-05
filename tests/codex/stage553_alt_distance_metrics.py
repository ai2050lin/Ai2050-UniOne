"""
stage553: 替代距离度量 + 句式版四模型Pearson r
目标：
1. 用Wasserstein距离替代cosine距离计算编码差异
2. 在句式版中重新计算四模型Pearson r
3. 用MMD(Maximum Mean Discrepancy)等核方法
4. 比较cosine/Wasserstein/MMD三种度量的跨模型一致性

使用已有的缓存数据(stage548)，并补充句式版Qwen3数据。
"""
import sys, os, json
import numpy as np
from scipy.stats import spearmanr

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'glm5', 'temp')


def cosine_dist_matrix(vectors):
    """计算cosine距离矩阵"""
    n = len(vectors)
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = 1 - np.dot(vectors[i], vectors[j]) / max(np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]), 1e-10)
            dm[i, j] = d
            dm[j, i] = d
    return dm


def euclidean_dist_matrix(vectors):
    """计算欧氏距离矩阵"""
    n = len(vectors)
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dm[i, j] = np.linalg.norm(vectors[i] - vectors[j])
            dm[j, i] = dm[i, j]
    return dm


def wasserstein_dist_per_layer(vectors1, vectors2):
    """
    简化版Wasserstein距离：不求解完整OT，而是用
    各维度的Wasserstein-1距离之和（按分量独立计算）
    vectors: [n_samples, dim]
    """
    if len(vectors1) != len(vectors2):
        raise ValueError("需要相同数量的样本")
    # 对每个维度独立计算W1距离并求和
    sorted1 = np.sort(vectors1, axis=0)
    sorted2 = np.sort(vectors2, axis=0)
    # W1 = sum_i |F^{-1}_1(i/n) - F^{-1}_2(i/n)|
    w1_per_dim = np.mean(np.abs(sorted1 - sorted2), axis=0)
    return float(np.sum(w1_per_dim))


def mmd_rbf(X, Y, gamma=None):
    """MMD with RBF kernel"""
    def pairwise_sq_dists(A, B):
        AA = np.sum(A**2, axis=1, keepdims=True)
        BB = np.sum(B**2, axis=1, keepdims=True)
        AB = A @ B.T
        return np.maximum(AA - 2*AB + BB.T, 0)
    
    if gamma is None:
        # 自动选择gamma
        XX = pairwise_sq_dists(X, X)
        YY = pairwise_sq_dists(Y, Y)
        gamma = 1.0 / max(np.median(XX[XX > 0]), np.median(YY[YY > 0]), 1e-10)
    
    K_XX = np.exp(-gamma * pairwise_sq_dists(X, X))
    K_YY = np.exp(-gamma * pairwise_sq_dists(Y, Y))
    K_XY = np.exp(-gamma * pairwise_sq_dists(X, Y))
    
    m = X.shape[0]
    n = Y.shape[0]
    mmd = np.sqrt(np.sum(K_XX) / (m*(m-1)) + np.sum(K_YY) / (n*(n-1)) - 2*np.sum(K_XY) / (m*n))
    return float(mmd)


def compare_metrics_across_models(dm1_cos, dm2_cos, vectors1, vectors2):
    """比较不同度量下的跨模型一致性"""
    triu_idx = np.triu_indices_from(dm1_cos, k=1)
    v1_cos = dm1_cos[triu_idx]
    v2_cos = dm2_cos[triu_idx]
    
    # Cosine Pearson r
    cos_r = np.corrcoef(v1_cos, v2_cos)[0, 1] if np.std(v1_cos) > 1e-10 else 0
    
    # Euclidean distance matrix
    dm1_euc = euclidean_dist_matrix(vectors1)
    dm2_euc = euclidean_dist_matrix(vectors2)
    v1_euc = dm1_euc[triu_idx]
    v2_euc = dm2_euc[triu_idx]
    euc_r = np.corrcoef(v1_euc, v2_euc)[0, 1] if np.std(v1_euc) > 1e-10 else 0
    
    # Spearman rank correlation (cosine)
    try:
        cos_rho, _ = spearmanr(v1_cos, v2_cos)
    except:
        cos_rho = 0
    
    # Spearman rank correlation (euclidean)
    try:
        euc_rho, _ = spearmanr(v1_euc, v2_euc)
    except:
        euc_rho = 0
    
    return {
        "cosine_pearson": round(cos_r, 6),
        "cosine_spearman": round(cos_rho, 6),
        "euclidean_pearson": round(euc_r, 6),
        "euclidean_spearman": round(euc_rho, 6),
    }


def load_stage548_data():
    """加载stage548的四模型距离矩阵"""
    models = {}
    for mname in ["Qwen3", "DeepSeek7B", "GLM4", "Gemma4"]:
        p = os.path.join(OUTPUT_DIR, f"stage548_{mname}_dist.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                models[mname] = json.load(f)
    return models


def main():
    print(f"{'='*60}")
    print(f"  stage553: 替代距离度量分析")
    print(f"{'='*60}")

    # 加载stage548数据
    models = load_stage548_data()
    print(f"  已加载: {list(models.keys())}")

    if len(models) < 2:
        print("  数据不足，请先运行stage548")
        return

    all_words = models[list(models.keys())[0]]["words"]
    print(f"  词汇: {all_words}")

    # ========== 实验1: Cosine vs Euclidean 跨模型比较 ==========
    print(f"\n{'='*60}")
    print(f"  [实验1] Cosine vs Euclidean 跨模型Pearson/Spearman")
    print(f"{'='*60}")

    model_names = list(models.keys())
    # 用L0和末层
    for m1 in model_names:
        for m2 in model_names:
            if m2 <= m1:
                continue
            print(f"\n  {m1} vs {m2}:")
            
            # 找共同层（L0）
            layers1 = [int(l) for l in models[m1]["dist_matrices"].keys()]
            layers2 = [int(l) for l in models[m2]["dist_matrices"].keys()]
            common = sorted(set(layers1) & set(layers2))
            
            if not common:
                print(f"    无共同层")
                continue
            
            # 取L0和末层
            for li in [common[0], common[-1]]:
                dm1 = np.array(models[m1]["dist_matrices"][str(li)])
                dm2 = np.array(models[m2]["dist_matrices"][str(li)])
                triu = np.triu_indices(len(all_words), k=1)
                v1, v2 = dm1[triu], dm2[triu]
                
                cos_p = np.corrcoef(v1, v2)[0, 1] if np.std(v1) > 1e-10 else 0
                try:
                    cos_s, _ = spearmanr(v1, v2)
                except:
                    cos_s = 0
                
                print(f"    L{li}: cosine_Pearson={cos_p:.4f}, cosine_Spearman={cos_s:.4f}")

    # ========== 实验2: 归一化层位置对齐 - 多度量比较 ==========
    print(f"\n{'='*60}")
    print(f"  [实验2] 归一化层对齐 - Euclidean/Spearman比较")
    print(f"{'='*60}")

    norm_positions = [0.0, 0.25, 0.5, 0.75, 1.0]
    all_pair_results = {}

    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            if j <= i:
                continue
            pair_key = f"{m1} vs {m2}"
            n1 = models[m1]["n_layers"]
            n2 = models[m2]["n_layers"]
            layers1 = [int(l) for l in models[m1]["dist_matrices"].keys()]
            layers2 = [int(l) for l in models[m2]["dist_matrices"].keys()]

            print(f"\n  {pair_key}:")
            for pos in norm_positions:
                li1 = min(layers1, key=lambda x: abs(x / max(n1 - 1, 1) - pos))
                li2 = min(layers2, key=lambda x: abs(x / max(n2 - 1, 1) - pos))
                dm1 = np.array(models[m1]["dist_matrices"][str(li1)])
                dm2 = np.array(models[m2]["dist_matrices"][str(li2)])
                triu = np.triu_indices(len(all_words), k=1)
                v1, v2 = dm1[triu], dm2[triu]

                cos_p = np.corrcoef(v1, v2)[0, 1] if np.std(v1) > 1e-10 else 0
                try:
                    cos_s, cos_sp = spearmanr(v1, v2)
                except:
                    cos_s, cos_sp = 0, 1.0

                print(f"    {int(pos*100)}% ({m1}.L{li1} vs {m2}.L{li2}): "
                      f"cos_P={cos_p:.4f}, cos_S={cos_s:.4f}, p={cos_sp:.2e}")

            all_pair_results[pair_key] = True

    # ========== 实验3: Spearman排名比较（不依赖具体距离值） ==========
    print(f"\n{'='*60}")
    print(f"  [实验3] 纯排名一致性 (Spearman)")
    print(f"{'='*60}")

    # 所有模型对在末层的Spearman
    print(f"\n  末层距离排名Spearman:")
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            if j <= i:
                continue
            layers1 = [int(l) for l in models[m1]["dist_matrices"].keys()]
            layers2 = [int(l) for l in models[m2]["dist_matrices"].keys()]
            dm1 = np.array(models[m1]["dist_matrices"][str(layers1[-1])])
            dm2 = np.array(models[m2]["dist_matrices"][str(layers2[-1])])
            triu = np.triu_indices(len(all_words), k=1)
            v1, v2 = dm1[triu], dm2[triu]
            try:
                rho, p = spearmanr(v1, v2)
                print(f"  {m1} vs {m2}: rho={rho:.4f}, p={p:.2e}")
            except:
                print(f"  {m1} vs {m2}: 无法计算")

    # ========== 实验4: Top-K距离对的一致性 ==========
    print(f"\n{'='*60}")
    print(f"  [实验4] Top-K最近/最远词对一致性")
    print(f"{'='*60}")

    n_pairs = len(all_words) * (len(all_words) - 1) // 2  # 153

    # 取末层
    for m1 in model_names:
        dm = np.array(models[m1]["dist_matrices"][str([int(l) for l in models[m1]["dist_matrices"].keys()][-1])])
        triu = np.triu_indices(len(all_words), k=1)
        dists = dm[triu]
        
        # Top-10 最近
        sorted_idx = np.argsort(dists)
        top10_closest = sorted_idx[:10]
        top10_farthest = sorted_idx[-10:]
        
        # 转为词对
        pairs = [(all_words[i], all_words[j]) for i, j in zip(triu[0], triu[1])]
        close_pairs = [pairs[k] for k in top10_closest]
        far_pairs = [pairs[k] for k in top10_farthest]
        
        print(f"\n  {m1} Top-5 最近:")
        for p in close_pairs[:5]:
            print(f"    {p[0]} - {p[1]}")
        print(f"  {m1} Top-5 最远:")
        for p in far_pairs[-5:]:
            print(f"    {p[0]} - {p[1]}")

    # ========== 总结 ==========
    print(f"\n{'='*60}")
    print(f"  拼图总结")
    print(f"{'='*60}")
    print(f"""
  关键问题：cosine距离是否是最佳度量？
  
  如果Spearman rho显著高于Pearson r → 排序一致但绝对值不同
  → 编码拓扑结构跨模型一致，但具体"尺度"不同
  → 需要使用秩统计量而非原始距离
""")


if __name__ == "__main__":
    main()
