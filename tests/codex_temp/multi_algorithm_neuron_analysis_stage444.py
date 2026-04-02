# -*- coding: utf-8 -*-
"""
Stage444: 多种算法分析神经元编码特征
目标: 使用多种算法深入分析神经元级别的编码特征
"""
import json
import numpy as np
from collections import defaultdict
from datetime import datetime
from scipy import stats
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, mutual_info_score, adjusted_rand_score

print("=" * 60)
print("Stage444: Multi-Algorithm Neuron Encoding Analysis")
print("=" * 60)

# 加载Stage432数据
RESULT_FILE = r"d:\develop\TransformerLens-main\tests\codex_temp\neuron_extraction_extended_qwen3_4b_stage432.json"
CLUSTER_FILE = r"d:\develop\TransformerLens-main\tests\codex_temp\neuron_activation_clustering_stage438.json"

print("\nLoading data...")
with open(RESULT_FILE, "r", encoding="utf-8") as f:
    neuron_data = json.load(f)
with open(CLUSTER_FILE, "r", encoding="utf-8") as f:
    cluster_data = json.load(f)

print(f"[OK] Data loaded!")
print(f"Words per POS: {neuron_data['word_count_per_pos']}")

# 构建神经元激活矩阵
print("\n" + "=" * 60)
print("Building Activation Matrix")
print("=" * 60)

# 结构: results -> pos -> word_activations -> word -> layer_activations -> layer -> top_neurons
results = neuron_data['results']
pos_tags = list(results.keys())
print(f"POS tags: {pos_tags}")

# 构建神经元激活向量
neuron_vectors = {}  # (layer, neuron_idx) -> [activation_per_pos]
neuron_word_count = {}  # (layer, neuron_idx) -> word_count

for pos_tag in pos_tags:
    pos_data = results[pos_tag]
    if 'word_activations' in pos_data:
        for word, word_data in pos_data['word_activations'].items():
            if 'layer_activations' in word_data:
                for layer_act in word_data['layer_activations']:
                    layer_idx = layer_act['layer_idx']
                    if 'top_neurons' in layer_act:
                        for neuron_idx in layer_act['top_neurons']:
                            key = (layer_idx, neuron_idx)
                            if key not in neuron_vectors:
                                neuron_vectors[key] = np.zeros(len(pos_tags))
                                neuron_word_count[key] = np.zeros(len(pos_tags))
                            # 激活强度用top_values的平均
                            if 'top_values' in layer_act:
                                avg_activation = np.mean(layer_act['top_values'][:5])
                                neuron_vectors[key][pos_tags.index(pos_tag)] += avg_activation
                                neuron_word_count[key][pos_tags.index(pos_tag)] += 1

# 归一化
for key in neuron_vectors:
    count = neuron_word_count[key]
    count[count == 0] = 1  # 避免除零
    neuron_vectors[key] = neuron_vectors[key] / count

print(f"Total neurons: {len(neuron_vectors)}")
print(f"Vector dimension: {len(pos_tags)}")

# 转换为矩阵
neurons = list(neuron_vectors.keys())
X = np.array([neuron_vectors[n] for n in neurons])
print(f"Activation matrix shape: {X.shape}")

# ============================================================
# 算法1: PCA降维分析
# ============================================================
print("\n" + "=" * 60)
print("Algorithm 1: PCA Analysis")
print("=" * 60)

pca = PCA(n_components=min(6, X.shape[1]))
X_pca = pca.fit_transform(X)

print(f"\nExplained variance ratio:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")
print(f"  Total: {sum(pca.explained_variance_ratio_):.4f}")

# 分析各主成分与词性的关系
print(f"\nPC1 vs POS correlation:")
pc1_loadings = pca.components_[0]
for i, pos in enumerate(pos_tags):
    print(f"  {pos}: {pc1_loadings[i]:.4f}")

# ============================================================
# 算法2: NMF非负矩阵分解
# ============================================================
print("\n" + "=" * 60)
print("Algorithm 2: NMF Analysis")
print("=" * 60)

# NMF requires non-negative values
X_pos = np.maximum(X, 0)
nmf = NMF(n_components=min(6, X.shape[1]), init='random', random_state=42)
W_nmf = nmf.fit_transform(X_pos)
H_nmf = nmf.components_

print(f"\nNMF reconstruction error: {nmf.reconstruction_err_:.4f}")
print(f"\nW matrix shape: {W_nmf.shape} (neurons x components)")
print(f"H matrix shape: {H_nmf.shape} (components x features)")

# 分析NMF组件与词性的关系
print(f"\nNMF component loadings on POS:")
for i in range(H_nmf.shape[0]):
    print(f"\n  Component {i+1}:")
    for j, pos in enumerate(pos_tags):
        print(f"    {pos}: {H_nmf[i, j]:.4f}")

# ============================================================
# 算法3: ICA独立成分分析
# ============================================================
print("\n" + "=" * 60)
print("Algorithm 3: ICA Analysis")
print("=" * 60)

ica = FastICA(n_components=min(6, X.shape[1]), random_state=42, max_iter=500)
X_ica = ica.fit_transform(X)

print(f"\nICA components shape: {X_ica.shape}")
print(f"\nICA component statistics:")
for i in range(X_ica.shape[1]):
    print(f"  IC{i+1}: mean={X_ica[:, i].mean():.4f}, std={X_ica[:, i].std():.4f}")

# 计算各成分与词性的互信息
print(f"\nMutual Information with POS:")
mi_scores = []
for i in range(X_ica.shape[1]):
    # 离散化ICA输出
    ic_discrete = np.digitize(X_ica[:, i], np.percentile(X_ica[:, i], [25, 50, 75]))
    # 计算与每个POS的互信息
    pos_labels = np.argmax(X, axis=1)
    mi = mutual_info_score(ic_discrete, pos_labels)
    mi_scores.append(mi)
    print(f"  IC{i+1}: MI={mi:.4f}")

# ============================================================
# 算法4: 多重聚类对比
# ============================================================
print("\n" + "=" * 60)
print("Algorithm 4: Multi-Clustering Comparison")
print("=" * 60)

k_range = range(5, 21)
silhouette_scores = []
inertias = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    silhouette_scores.append(silhouette_score(X, labels))
    inertias.append(kmeans.inertia_)

print(f"\nSilhouette scores (k=5 to 20):")
for k, score in zip(k_range, silhouette_scores):
    print(f"  k={k:2d}: {score:.4f}")

best_k = list(k_range)[np.argmax(silhouette_scores)]
print(f"\nBest k by silhouette: {best_k}")

# 层次聚类
print("\nHierarchical Clustering:")
hier = AgglomerativeClustering(n_clusters=15, linkage='ward')
hier_labels = hier.fit_predict(X)
hier_silhouette = silhouette_score(X, hier_labels)
print(f"  n_clusters=15, silhouette: {hier_silhouette:.4f}")

# 谱聚类
print("\nSpectral Clustering:")
spec = SpectralClustering(n_clusters=15, random_state=42, affinity='nearest_neighbors')
spec_labels = spec.fit_predict(X)
spec_silhouette = silhouette_score(X, spec_labels)
print(f"  n_clusters=15, silhouette: {spec_silhouette:.4f}")

# GMM聚类
print("\nGaussian Mixture Model:")
gmm = GaussianMixture(n_components=15, random_state=42, covariance_type='full')
gmm_labels = gmm.fit_predict(X)
gmm_silhouette = silhouette_score(X, gmm_labels)
print(f"  n_clusters=15, silhouette: {gmm_silhouette:.4f}")

# ============================================================
# 算法5: 神经元特异性分析（多维度）
# ============================================================
print("\n" + "=" * 60)
print("Algorithm 5: Multi-Dimensional Specificity Analysis")
print("=" * 60)

# 计算多种特异性指标
neuron_metrics = {}

for i, key in enumerate(neurons):
    vec = X[i]
    pos_idx = np.argmax(vec)  # 最强激活的POS
    max_val = vec[pos_idx]
    sum_val = np.sum(vec)

    # 特异性指标
    specificity = max_val / sum_val if sum_val > 0 else 0

    # 熵（多功能性）
    vec_pos = vec[vec > 0]
    if len(vec_pos) > 0:
        vec_norm = vec_pos / np.sum(vec_pos)
        entropy = -np.sum(vec_norm * np.log(vec_norm + 1e-10))
    else:
        entropy = 0

    # 激活稀疏度
    sparsity = 1 - np.count_nonzero(vec) / len(vec)

    # 激活均匀度
    uniformity = np.std(vec) / (np.mean(vec) + 1e-10)

    neuron_metrics[key] = {
        'specificity': specificity,
        'entropy': entropy,
        'sparsity': sparsity,
        'uniformity': uniformity,
        'dominant_pos': pos_tags[pos_idx],
        'activation_strength': max_val
    }

# 统计特性分布
print("\nNeuron metric distributions:")
spec_values = [m['specificity'] for m in neuron_metrics.values()]
print(f"  Specificity: mean={np.mean(spec_values):.4f}, std={np.std(spec_values):.4f}")

ent_values = [m['entropy'] for m in neuron_metrics.values()]
print(f"  Entropy: mean={np.mean(ent_values):.4f}, std={np.std(ent_values):.4f}")

spar_values = [m['sparsity'] for m in neuron_metrics.values()]
print(f"  Sparsity: mean={np.mean(spar_values):.4f}, std={np.std(spar_values):.4f}")

# 神经元分类
specialized = [(k, v) for k, v in neuron_metrics.items() if v['specificity'] > 0.5]
multifunctional = [(k, v) for k, v in neuron_metrics.items() if v['entropy'] > 1.5]
sparse = [(k, v) for k, v in neuron_metrics.items() if v['sparsity'] > 0.7]

print(f"\nNeuron classification:")
print(f"  Specialized (spec > 0.5): {len(specialized)} ({len(specialized)/len(neurons)*100:.2f}%)")
print(f"  Multifunctional (ent > 1.5): {len(multifunctional)} ({len(multifunctional)/len(neurons)*100:.2f}%)")
print(f"  Sparse (sparsity > 0.7): {len(sparse)} ({len(sparse)/len(neurons)*100:.2f}%)")

# ============================================================
# 算法6: 层间激活模式分析
# ============================================================
print("\n" + "=" * 60)
print("Algorithm 6: Layer Activation Pattern Analysis")
print("=" * 60)

# 按层聚合神经元
layer_vectors = defaultdict(list)
for key, vec in neuron_vectors.items():
    layer_vectors[key[0]].append(vec)

# 计算每层的平均激活模式
layer_avg_patterns = {}
for layer, vecs in layer_vectors.items():
    layer_avg_patterns[layer] = np.mean(vecs, axis=0)

# 计算层间相关性
layer_corr = np.zeros((len(layer_avg_patterns), len(layer_avg_patterns)))
layers = sorted(layer_avg_patterns.keys())
for i, l1 in enumerate(layers):
    for j, l2 in enumerate(layers):
        if i != j:
            corr, _ = stats.pearsonr(layer_avg_patterns[l1], layer_avg_patterns[l2])
            layer_corr[i, j] = corr

print(f"\nLayer activation pattern correlations:")
print(f"\nLayer pair similarities (top 10 most similar):")
similar_pairs = []
for i, l1 in enumerate(layers):
    for j, l2 in enumerate(layers):
        if i < j:
            similar_pairs.append((l1, l2, layer_corr[i, j]))

similar_pairs.sort(key=lambda x: x[2], reverse=True)
for l1, l2, corr in similar_pairs[:10]:
    print(f"  Layer {l1} <-> Layer {l2}: {corr:.4f}")

print(f"\nMost dissimilar pairs (indicating functional shift):")
for l1, l2, corr in similar_pairs[-5:]:
    print(f"  Layer {l1} <-> Layer {l2}: {corr:.4f}")

# ============================================================
# 算法7: 激活值分布分析
# ============================================================
print("\n" + "=" * 60)
print("Algorithm 7: Activation Distribution Analysis")
print("=" * 60)

# 收集所有激活值
all_activations = X.flatten()
all_activations = all_activations[all_activations != 0]

print(f"\nActivation value statistics:")
print(f"  Total values: {len(all_activations)}")
print(f"  Mean: {np.mean(all_activations):.4f}")
print(f"  Std: {np.std(all_activations):.4f}")
print(f"  Min: {np.min(all_activations):.4f}")
print(f"  Max: {np.max(all_activations):.4f}")

# 按词性分析激活分布
print(f"\nActivation by POS:")
for i, pos in enumerate(pos_tags):
    pos_activations = X[:, i]
    pos_activations = pos_activations[pos_activations != 0]
    if len(pos_activations) > 0:
        print(f"  {pos:12s}: mean={np.mean(pos_activations):.4f}, std={np.std(pos_activations):.4f}")

# ============================================================
# 综合分析：编码特征发现
# ============================================================
print("\n" + "=" * 60)
print("Comprehensive Encoding Feature Discovery")
print("=" * 60)

# 使用KMeans识别主要编码模式
kmeans_final = KMeans(n_clusters=15, random_state=42, n_init=10)
final_labels = kmeans_final.fit_predict(X)
centers = kmeans_final.cluster_centers_

# 分析每个聚类的特征
cluster_features = []
for i in range(15):
    mask = final_labels == i
    cluster_size = np.sum(mask)
    center = centers[i]
    dominant_pos = pos_tags[np.argmax(center)]
    entropy = -np.sum(center * np.log(center + 1e-10))
    specificity = np.max(center) / np.sum(center) if np.sum(center) > 0 else 0

    cluster_features.append({
        'cluster_id': i,
        'size': int(cluster_size),
        'dominant_pos': dominant_pos,
        'specificity': float(specificity),
        'entropy': float(entropy),
        'center': center.tolist(),
        'neurons': [(neurons[j][0], neurons[j][1]) for j in range(len(mask)) if mask[j]][:5]
    })

# 按特异性排序
cluster_features.sort(key=lambda x: x['specificity'], reverse=True)

print("\nTop 10 most specific clusters:")
for i, cf in enumerate(cluster_features[:10]):
    print(f"\n  Cluster {cf['cluster_id']}:")
    print(f"    Size: {cf['size']}")
    print(f"    Dominant POS: {cf['dominant_pos']}")
    print(f"    Specificity: {cf['specificity']:.4f}")
    print(f"    Entropy: {cf['entropy']:.4f}")
    print(f"    Example neurons: {cf['neurons']}")

# 保存结果
print("\n" + "=" * 60)
print("Saving Results")
print("=" * 60)

results = {
    'experiment_id': 'multi_algorithm_neuron_analysis_stage444',
    'timestamp': datetime.now().isoformat(),
    'pca_analysis': {
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'total_explained': float(sum(pca.explained_variance_ratio_)),
        'components': pca.components_.tolist()
    },
    'nmf_analysis': {
        'reconstruction_error': float(nmf.reconstruction_err_),
        'W_shape': list(W_nmf.shape),
        'H_shape': list(H_nmf.shape)
    },
    'ica_analysis': {
        'components_shape': list(X_ica.shape),
        'mutual_info_scores': mi_scores
    },
    'clustering_comparison': {
        'kmeans_silhouette_scores': {f'k_{k}': float(s) for k, s in zip(k_range, silhouette_scores)},
        'hierarchical_silhouette': float(hier_silhouette),
        'spectral_silhouette': float(spec_silhouette),
        'gmm_silhouette': float(gmm_silhouette),
        'best_k': int(best_k)
    },
    'neuron_metrics': {
        'total_neurons': len(neurons),
        'specialized_count': len(specialized),
        'multifunctional_count': len(multifunctional),
        'sparse_count': len(sparse)
    },
    'layer_correlations': {
        'layers': layers,
        'correlation_matrix': layer_corr.tolist()
    },
    'activation_stats': {
        'mean': float(np.mean(all_activations)),
        'std': float(np.std(all_activations)),
        'min': float(np.min(all_activations)),
        'max': float(np.max(all_activations))
    },
    'cluster_features': cluster_features
}

output_file = r"d:\develop\TransformerLens-main\tests\codex_temp\multi_algorithm_analysis_stage444.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n[OK] Results saved to: {output_file}")

print("\n" + "=" * 60)
print("Stage444 Completed!")
print("=" * 60)
print("\nKey Findings from Multi-Algorithm Analysis:")
print(f"1. PCA: {sum(pca.explained_variance_ratio_)*100:.1f}% variance explained by 6 components")
print(f"2. Best cluster number by silhouette: k={best_k}")
print(f"3. Specialized neurons: {len(specialized)} ({len(specialized)/len(neurons)*100:.1f}%)")
print(f"4. Multifunctional neurons: {len(multifunctional)} ({len(multifunctional)/len(neurons)*100:.1f}%)")
print(f"5. Layer activation patterns show correlations across layers")
print("\nNext Steps:")
print("- Stage445: Functional module validation experiments")
print("- Stage446: Cross-model comparison validation")
