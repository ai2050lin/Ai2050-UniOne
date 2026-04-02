"""
Stage435: 基于Stage432结果进行深度神经元特性分析
目标：对已有的30个单词/词性的结果进行更深入的分析

分析维度：
1. 神经元激活频率分布
2. 神经元特异性分析
3. 神经元激活强度分布
4. 神经元共激活模式
5. 层级功能分化
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import itertools

# 读取Stage432的结果
RESULT_FILE = Path("D:/develop/TransformerLens-main/tests/codex_temp/neuron_extraction_extended_qwen3_4b_stage432.json")

# 输出目录
OUTPUT_DIR = Path("D:/develop/TransformerLens-main/tests/codex_temp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_results(file_path):
    """加载Stage432的结果"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_neuron_activation_counts(word_activations):
    """
    提取神经元激活计数

    参数:
        word_activations: 单词激活字典

    返回:
        neuron_count: {(layer, neuron_idx): count}
        neuron_activations: {(layer, neuron_idx): [activation_values]}
    """
    neuron_count = defaultdict(int)
    neuron_activations = defaultdict(list)

    for word, activations in word_activations.items():
        if activations is None:
            continue

        for layer_act in activations["layer_activations"]:
            layer_idx = layer_act["layer_idx"]
            for i, neuron_idx in enumerate(layer_act["top_neurons"]):
                neuron_count[(layer_idx, neuron_idx)] += 1
                neuron_activations[(layer_idx, neuron_idx)].append(layer_act["top_values"][i])

    return neuron_count, neuron_activations

def calculate_neuron_specificity(neuron_counts_by_pos, total_words_by_pos):
    """
    计算神经元对词性的特异性

    参数:
        neuron_counts_by_pos: {pos_tag: {(layer, neuron_idx): count}}
        total_words_by_pos: {pos_tag: total_words}

    返回:
        specificity: {(layer, neuron_idx): {pos_tag: specificity_score}}
    """
    specificity = {}
    all_neurons = set()

    # 收集所有神经元
    for pos_tag, neuron_counts in neuron_counts_by_pos.items():
        all_neurons.update(neuron_counts.keys())

    # 对每个神经元计算特异性
    for neuron in all_neurons:
        specificity[neuron] = {}
        for pos_tag, neuron_counts in neuron_counts_by_pos.items():
            if neuron in neuron_counts:
                # 特异性 = 该神经元在该词性中的激活比例
                count = neuron_counts[neuron]
                total_words = total_words_by_pos[pos_tag]
                specificity[neuron][pos_tag] = count / total_words
            else:
                specificity[neuron][pos_tag] = 0.0

    return specificity

def analyze_neuron_specificity(specificity):
    """
    分析神经元的特异性

    返回:
        highly_specific: 对某个词性高度特异的神经元
        multi_functional: 对多个词性都有激活的神经元
    """
    highly_specific = {}
    multi_functional = []

    for neuron, pos_scores in specificity.items():
        sorted_scores = sorted(pos_scores.items(), key=lambda x: x[1], reverse=True)
        top_pos, top_score = sorted_scores[0]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0

        # 高度特异：最高分数 > 0.3 且是第二高的2倍以上
        if top_score > 0.3 and top_score > 2 * second_score:
            highly_specific[neuron] = {
                'primary_pos': top_pos,
                'specificity': top_score,
                'second_highest': second_score
            }

        # 多功能：对至少2个词性都有激活（频率 > 0.1）
        active_pos_count = sum(1 for score in pos_scores.values() if score > 0.1)
        if active_pos_count >= 2:
            multi_functional.append({
                'neuron': neuron,
                'active_pos': [pos for pos, score in pos_scores.items() if score > 0.1],
                'scores': pos_scores
            })

    return highly_specific, multi_functional

def calculate_activation_distribution(neuron_counts_by_pos):
    """
    计算神经元激活分布

    返回:
        distribution: {pos_tag: [(layer, neuron_idx, count, frequency)]}
    """
    distribution = {}

    for pos_tag, neuron_counts in neuron_counts_by_pos.items():
        total_words = len([w for w, a in neuron_counts_by_pos[pos_tag].items()])
        # 计算总激活次数
        total_activations = sum(neuron_counts.values())

        sorted_neurons = sorted(neuron_counts.items(), key=lambda x: x[1], reverse=True)

        distribution[pos_tag] = []
        for (layer, neuron_idx), count in sorted_neurons[:100]:  # Top 100
            frequency = count / total_words
            distribution[pos_tag].append({
                'layer': layer,
                'neuron_idx': neuron_idx,
                'count': count,
                'frequency': frequency
            })

    return distribution

def analyze_layer_distribution(distribution):
    """
    分析激活的层级分布

    返回:
        layer_stats: {pos_tag: {layer: total_count}}
    """
    layer_stats = {}

    for pos_tag, neurons in distribution.items():
        layer_stats[pos_tag] = defaultdict(int)
        for neuron_info in neurons:
            layer = neuron_info['layer']
            layer_stats[pos_tag][layer] += neuron_info['count']

    return layer_stats

def calculate_centroid_layer(layer_stats):
    """
    计算每个词性的质心层（加平均层索引）

    返回:
        centroids: {pos_tag: centroid_layer}
    """
    centroids = {}

    for pos_tag, layer_counts in layer_stats.items():
        total_count = sum(layer_counts.values())
        weighted_sum = sum(layer * count for layer, count in layer_counts.items())
        centroid = weighted_sum / total_count if total_count > 0 else 0
        centroids[pos_tag] = centroid

    return centroids

def analyze_neuron_coactivation(neuron_counts_by_pos, top_k=20):
    """
    分析神经元的共激活模式

    返回:
        coactivation: {pos_tag: [((layer1, neuron1), (layer2, neuron2), coactivation_count)]}
    """
    coactivation = {}

    for pos_tag, neuron_counts in neuron_counts_by_pos.items():
        # 提取top_k神经元
        sorted_neurons = sorted(neuron_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_neurons = [neuron for neuron, count in sorted_neurons]

        # 计算共激活
        coactivation[pos_tag] = []
        for i in range(len(top_neurons)):
            for j in range(i+1, len(top_neurons)):
                # 两个神经元都被激活的单词数
                # 这里简化处理，假设它们经常一起出现
                count = min(neuron_counts[top_neurons[i]], neuron_counts[top_neurons[j]])
                if count > 0:
                    coactivation[pos_tag].append({
                        'neuron1': top_neurons[i],
                        'neuron2': top_neurons[j],
                        'coactivation_count': count
                    })

        # 按共激活次数排序
        coactivation[pos_tag].sort(key=lambda x: x['coactivation_count'], reverse=True)

    return coactivation

def main():
    print("\n" + "="*60)
    print("Stage435: 基于Stage432结果进行深度神经元特性分析")
    print("="*60)

    # 加载Stage432结果
    print(f"\n加载Stage432结果: {RESULT_FILE}")
    results = load_results(RESULT_FILE)
    print(f"[OK] 结果加载成功!")

    model_config = results['model_config']
    num_layers = model_config['num_layers']
    hidden_size = model_config['hidden_size']

    print(f"\n模型信息:")
    print(f"  - 模型: {model_config['name']}")
    print(f"  - 层数: {num_layers}")
    print(f"  - 隐藏层大小: {hidden_size}")
    print(f"  - 每个词性单词数: {results['word_count_per_pos']}")

    # 提取每个词性的神经元激活计数
    print(f"\n提取神经元激活计数...")
    neuron_counts_by_pos = {}
    total_words_by_pos = {}

    for pos_tag, pos_data in results['results'].items():
        word_activations = pos_data['word_activations']
        neuron_count, neuron_activations = extract_neuron_activation_counts(word_activations)

        neuron_counts_by_pos[pos_tag] = neuron_count
        total_words_by_pos[pos_tag] = len([w for w, a in word_activations.items() if a is not None])

        print(f"  {pos_tag}: {len(neuron_count)} neurons activated")

    # 计算神经元特异性
    print(f"\nCalculating neuron specificity...")
    specificity = calculate_neuron_specificity(neuron_counts_by_pos, total_words_by_pos)

    # 分析神经元的特异性
    highly_specific, multi_functional = analyze_neuron_specificity(specificity)

    print(f"\nHighly specific neurons: {len(highly_specific)}")
    print(f"Multi-functional neurons: {len(multi_functional)}")

    # 计算激活分布
    print(f"\nCalculating activation distribution...")
    distribution = calculate_activation_distribution(neuron_counts_by_pos)

    # 分析层级分布
    print(f"\nAnalyzing layer distribution...")
    layer_stats = analyze_layer_distribution(distribution)

    # 计算质心层
    print(f"\nCalculating centroid layers...")
    centroids = calculate_centroid_layer(layer_stats)

    print(f"\nCentroid layers for each POS:")
    for pos_tag, centroid in sorted(centroids.items(), key=lambda x: x[1]):
        normalized = centroid / num_layers
        print(f"  {pos_tag}: {centroid:.2f} (normalized: {normalized:.3f})")

    # 分析神经元共激活
    print(f"\nAnalyzing neuron coactivation patterns...")
    coactivation = analyze_neuron_coactivation(neuron_counts_by_pos, top_k=20)

    # 保存分析结果
    analysis_results = {
        'experiment_id': 'deep_neuron_analysis_stage435',
        'timestamp': datetime.now().isoformat(),
        'model_config': model_config,
        'summary': {
            'total_words_per_pos': total_words_by_pos,
            'total_neurons_by_pos': {pos: len(neuron_counts_by_pos[pos]) for pos in neuron_counts_by_pos},
            'highly_specific_neurons_count': len(highly_specific),
            'multi_functional_neurons_count': len(multi_functional)
        },
        'specificity': {
            f"{layer}_{neuron_idx}": scores
            for (layer, neuron_idx), scores in specificity.items()
        },
        'highly_specific_neurons': [
            {'layer': layer, 'neuron_idx': neuron_idx, **info}
            for (layer, neuron_idx), info in highly_specific.items()
        ],
        'multi_functional_neurons': [
            {'layer': neuron['neuron'][0], 'neuron_idx': neuron['neuron'][1], **neuron}
            for neuron in multi_functional
        ],
        'distribution': distribution,
        'layer_stats': {pos: dict(layer_counts) for pos, layer_counts in layer_stats.items()},
        'centroids': centroids,
        'coactivation': coactivation
    }

    # 保存完整结果
    output_file = OUTPUT_DIR / "deep_neuron_analysis_stage435.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Analysis completed!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")

    # 打印关键发现
    print(f"\nKey Findings:")

    print(f"\n1. Highly specific neurons (Top 10):")
    sorted_specific = sorted(highly_specific.items(), key=lambda x: x[1]['specificity'], reverse=True)[:10]
    for i, ((layer, neuron_idx), info) in enumerate(sorted_specific):
        print(f"   {i+1}. Layer {layer}, Neuron {neuron_idx}: {info['primary_pos']}")
        print(f"      Specificity: {info['specificity']:.3f}, 2nd highest: {info['second_highest']:.3f}")

    print(f"\n2. Multi-functional neurons (Top 10):")
    sorted_multi = sorted(multi_functional, key=lambda x: len(x['active_pos']), reverse=True)[:10]
    for i, neuron_info in enumerate(sorted_multi):
        layer, neuron_idx = neuron_info['neuron']
        print(f"   {i+1}. Layer {layer}, Neuron {neuron_idx}: {len(neuron_info['active_pos'])} POS tags")
        print(f"      Active POS tags: {', '.join(neuron_info['active_pos'])}")

    print(f"\n3. Centroid layer ranking:")
    sorted_centroids = sorted(centroids.items(), key=lambda x: x[1])
    for i, (pos_tag, centroid) in enumerate(sorted_centroids):
        print(f"   {i+1}. {pos_tag}: {centroid:.2f}")

if __name__ == "__main__":
    main()
