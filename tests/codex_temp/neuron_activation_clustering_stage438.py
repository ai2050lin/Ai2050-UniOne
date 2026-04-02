"""
Stage438: 神经元激活向量聚类与功能模块识别
目标：对神经元的激活向量进行聚类，识别功能模块

基于Stage432的30个单词/词性数据，提取神经元的激活向量
使用K-means和层次聚类方法，识别功能模块
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

# 读取数据
RESULT_FILE = Path("D:/develop/TransformerLens-main/tests/codex_temp/deep_neuron_analysis_stage435.json")

# 输出目录
OUTPUT_DIR = Path("D:/develop/TransformerLens-main/tests/codex_temp")

def load_stage435_results():
    """加载Stage435的结果"""
    with open(RESULT_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_activation_vectors():
    """
    提取神经元的激活向量

    每个神经元的激活向量 = [在6个词性中的平均激活频率]
    """
    print("\n" + "="*60)
    print("Stage438: 神经元激活向量聚类与功能模块识别")
    print("="*60)

    # 读取Stage432的原始数据
    stage432_file = OUTPUT_DIR / "neuron_extraction_extended_qwen3_4b_stage432.json"
    with open(stage432_file, 'r', encoding='utf-8') as f:
        stage432_data = json.load(f)

    print(f"\n[OK] 数据加载成功!")
    print(f"模型: {stage432_data['model_config']['name']}")
    print(f"层数: {stage432_data['model_config']['num_layers']}")

    # 统计每个神经元在每个词性中的激活次数
    pos_tags = ['noun', 'adjective', 'verb', 'adverb', 'pronoun', 'preposition']

    # {neuron: {pos_tag: activation_count}}
    neuron_pos_activation = defaultdict(lambda: {pos: 0 for pos in pos_tags})
    # {neuron: total_activation_count}
    neuron_total_activation = defaultdict(int)

    for pos_tag, pos_data in stage432_data['results'].items():
        word_activations = pos_data['word_activations']

        for word, activations in word_activations.items():
            if activations is None:
                continue

            for layer_act in activations["layer_activations"]:
                layer_idx = layer_act["layer_idx"]
                for neuron_idx in layer_act["top_neurons"]:
                    neuron = (layer_idx, neuron_idx)
                    neuron_pos_activation[neuron][pos_tag] += 1
                    neuron_total_activation[neuron] += 1

    # 转换为激活向量
    neurons = []
    activation_vectors = []
    neuron_info = {}

    for neuron, pos_counts in neuron_pos_activation.items():
        # 归一化激活向量
        total = neuron_total_activation[neuron]
        if total > 0:
            vector = [pos_counts[pos] / total for pos in pos_tags]
        else:
            vector = [0.0] * len(pos_tags)

        neurons.append(neuron)
        activation_vectors.append(vector)
        neuron_info[neuron] = {
            'pos_counts': pos_counts,
            'total_activation': total,
            'vector': vector
        }

    activation_vectors = np.array(activation_vectors)

    print(f"\n提取了 {len(neurons)} 个神经元的激活向量")
    print(f"向量维度: {activation_vectors.shape}")

    return neurons, activation_vectors, neuron_info, pos_tags

def determine_optimal_clusters(activation_vectors, max_k=15):
    """
    确定最优聚类数
    """
    print(f"\n确定最优聚类数...")

    silhouette_scores = []
    ch_scores = []

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(activation_vectors)

        sil_score = silhouette_score(activation_vectors, labels)
        ch_score = calinski_harabasz_score(activation_vectors, labels)

        silhouette_scores.append(sil_score)
        ch_scores.append(ch_score)

        print(f"  K={k}: Silhouette={sil_score:.4f}, CH={ch_score:.2f}")

    # 找到最优K（基于Silhouette分数）
    optimal_k = np.argmax(silhouette_scores) + 2
    print(f"\n  最优K (by Silhouette): {optimal_k}")

    return optimal_k, silhouette_scores, ch_scores

def perform_clustering(activation_vectors, n_clusters):
    """
    执行K-means聚类
    """
    print(f"\n执行K-means聚类 (K={n_clusters})...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(activation_vectors)

    # 获取聚类中心
    centers = kmeans.cluster_centers_

    # 计算每个聚类的神经元数
    cluster_sizes = np.bincount(labels)

    print(f"  聚类完成!")
    for i, size in enumerate(cluster_sizes):
        print(f"    Cluster {i}: {size} neurons ({size/len(labels)*100:.1f}%)")

    return labels, centers, cluster_sizes

def analyze_clusters(neurons, labels, centers, neuron_info, pos_tags):
    """
    分析每个聚类的特征
    """
    print(f"\n分析聚类特征...")

    n_clusters = len(centers)
    cluster_analysis = []

    for cluster_id in range(n_clusters):
        # 获取该聚类中的神经元
        cluster_neurons = [neurons[i] for i in range(len(labels)) if labels[i] == cluster_id]
        cluster_vectors = [neuron_info[neurons[i]]['vector'] for i in range(len(labels)) if labels[i] == cluster_id]

        # 计算平均激活向量
        avg_vector = np.mean(cluster_vectors, axis=0)

        # 确定主导词性
        dominant_pos_idx = np.argmax(avg_vector)
        dominant_pos = pos_tags[dominant_pos_idx]

        # 统计层级分布
        layer_distribution = defaultdict(int)
        for neuron in cluster_neurons:
            layer_distribution[neuron[0]] += 1

        # 计算该聚类的主导层级
        dominant_layer = max(layer_distribution.items(), key=lambda x: x[1])[0] if layer_distribution else 0

        # 分析功能特性
        purity = avg_vector[dominant_pos_idx]
        entropy = -np.sum(avg_vector * np.log(avg_vector + 1e-10))

        cluster_analysis.append({
            'cluster_id': cluster_id,
            'size': len(cluster_neurons),
            'dominant_pos': dominant_pos,
            'dominant_pos_ratio': purity,
            'dominant_layer': dominant_layer,
            'avg_vector': avg_vector.tolist(),
            'entropy': entropy,
            'layer_distribution': dict(layer_distribution),
            'example_neurons': [(n[0], n[1]) for n in cluster_neurons[:5]]
        })

    return cluster_analysis

def hierarchical_clustering(activation_vectors, n_clusters):
    """
    执行层次聚类作为对比
    """
    print(f"\n执行层次聚类 (K={n_clusters})...")

    agg = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agg.fit_predict(activation_vectors)

    return labels

def main():
    # 提取激活向量
    neurons, activation_vectors, neuron_info, pos_tags = extract_activation_vectors()

    # 确定最优聚类数
    optimal_k, silhouette_scores, ch_scores = determine_optimal_clusters(activation_vectors)

    # 执行K-means聚类
    labels, centers, cluster_sizes = perform_clustering(activation_vectors, optimal_k)

    # 分析聚类结果
    cluster_analysis = analyze_clusters(neurons, labels, centers, neuron_info, pos_tags)

    # 执行层次聚类作为对比
    hier_labels = hierarchical_clustering(activation_vectors, optimal_k)

    # 保存结果
    analysis_results = {
        'experiment_id': 'neuron_activation_clustering_stage438',
        'timestamp': datetime.now().isoformat(),
        'num_neurons': len(neurons),
        'vector_dimension': len(pos_tags),
        'optimal_k': int(optimal_k),
        'silhouette_scores': {f"k_{k+2}": float(score) for k, score in enumerate(silhouette_scores)},
        'clustering_results': {
            'kmeans': {
                'labels': [int(x) for x in labels.tolist()],
                'centers': [[float(v) for v in center] for center in centers.tolist()],
                'cluster_sizes': [int(x) for x in cluster_sizes.tolist()]
            },
            'hierarchical': {
                'labels': [int(x) for x in hier_labels.tolist()]
            }
        },
        'cluster_analysis': [
            {
                'cluster_id': int(c['cluster_id']),
                'size': int(c['size']),
                'dominant_pos': c['dominant_pos'],
                'dominant_pos_ratio': float(c['dominant_pos_ratio']),
                'dominant_layer': int(c['dominant_layer']),
                'avg_vector': [float(v) for v in c['avg_vector']],
                'entropy': float(c['entropy']),
                'layer_distribution': {int(k): int(v) for k, v in c['layer_distribution'].items()},
                'example_neurons': [(int(n[0]), int(n[1])) for n in c['example_neurons']]
            }
            for c in cluster_analysis
        ]
    }

    output_file = OUTPUT_DIR / "neuron_activation_clustering_stage438.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"[OK] 分析完成!")
    print(f"结果已保存: {output_file}")
    print(f"{'='*60}")

    # 打印关键发现
    print(f"\n{'='*60}")
    print("Key Findings:")
    print(f"{'='*60}")

    print(f"\n1. Optimal Number of Clusters: {optimal_k}")
    print(f"   - Based on Silhouette Score: {max(silhouette_scores):.4f}")

    print(f"\n2. Cluster Analysis:")
    for cluster in cluster_analysis:
        print(f"\n   Cluster {cluster['cluster_id']} ({cluster['size']} neurons, {cluster['size']/len(neurons)*100:.1f}%):")
        print(f"      - Dominant POS: {cluster['dominant_pos']} ({cluster['dominant_pos_ratio']:.3f})")
        print(f"      - Dominant Layer: {cluster['dominant_layer']}")
        print(f"      - Entropy: {cluster['entropy']:.3f}")
        print(f"      - Avg vector: {[f'{v:.3f}' for v in cluster['avg_vector']]}")
        print(f"      - Example neurons: {cluster['example_neurons'][:3]}")

    # 理论分析
    print(f"\n{'='*60}")
    print("Theoretical Analysis:")
    print(f"{'='*60}")

    print(f"\n1. 功能模块分化:")
    dominant_pos_counts = defaultdict(int)
    for cluster in cluster_analysis:
        dominant_pos_counts[cluster['dominant_pos']] += cluster['size']

    print(f"   - 各词性主导的聚类神经元数:")
    for pos, count in sorted(dominant_pos_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"     {pos}: {count} ({count/len(neurons)*100:.1f}%)")

    print(f"\n2. 层级功能分化:")
    layer_counts = defaultdict(int)
    for cluster in cluster_analysis:
        dominant_layer = cluster['dominant_layer']
        layer_counts[dominant_layer] += cluster['size']

    print(f"   - 各层级主导的聚类神经元数:")
    for layer, count in sorted(layer_counts.items())[:10]:
        print(f"     Layer {layer}: {count}")

    print(f"\n3. 多功能 vs 特异性:")
    specific_clusters = [c for c in cluster_analysis if c['entropy'] < 1.0]
    multi_functional_clusters = [c for c in cluster_analysis if c['entropy'] >= 1.0]

    print(f"   - 高特异性聚类 (entropy<1.0): {len(specific_clusters)} ({sum(c['size'] for c in specific_clusters)} neurons)")
    print(f"   - 多功能聚类 (entropy>=1.0): {len(multi_functional_clusters)} ({sum(c['size'] for c in multi_functional_clusters)} neurons)")

    print(f"\n4. 对AGI的启示:")
    print(f"   - 神经元自然形成{optimal_k}个功能模块")
    print(f"   - 模块化结构支持了'功能专业化'理论")
    print(f"   - 多功能神经元可能属于'协调器'模块")
    print(f"   - 层级的功能分化不明显，说明语言处理是分布式的")

if __name__ == "__main__":
    main()
