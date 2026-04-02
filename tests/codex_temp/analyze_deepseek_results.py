"""分析DeepSeek-7B的神经元提取结果"""
import json
from pathlib import Path

# 读取结果文件
result_file = Path("D:/develop/TransformerLens-main/tests/codex_temp/neuron_extraction_deepseek_7b_stage431.json")
with open(result_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 80)
print("DeepSeek-7B 神经元提取结果分析")
print("=" * 80)

# 分析每个词性的关键神经元分布
for pos, pos_data in data['results'].items():
    print(f"\n词性: {pos} ({pos_data['description']})")
    print("-" * 80)
    
    key_neurons = pos_data['key_neurons']
    top_100 = key_neurons['top_100_neurons']
    layer_dist = key_neurons['layer_distribution']
    weighted_center = key_neurons['weighted_center']
    
    # 显示前10个关键神经元
    print(f"前10个关键神经元:")
    for i, neuron in enumerate(top_100[:10], 1):
        print(f"  {i}. Layer {neuron['layer']}, Neuron {neuron['neuron']} - 激活次数: {neuron['activation_count']}")
    
    # 分析层分布
    print(f"\n层分布统计:")
    print(f"  质心层: {weighted_center:.2f} (normalized: {weighted_center / 27:.3f})")
    
    # 早期层（0-9）、中期层（10-18）、晚期层（19-27）
    early = sum(layer_dist[str(i)] for i in range(10))
    mid = sum(layer_dist[str(i)] for i in range(10, 19))
    late = sum(layer_dist[str(i)] for i in range(19, 28))
    total = early + mid + late
    
    print(f"  早期层 (0-9): {early} ({early/total*100:.1f}%)")
    print(f"  中期层 (10-18): {mid} ({mid/total*100:.1f}%)")
    print(f"  晚期层 (19-27): {late} ({late/total*100:.1f}%)")
    
    # 显示激活最多的层
    sorted_layers = sorted(layer_dist.items(), key=lambda x: x[1], reverse=True)
    print(f"\n激活最密集的层:")
    for layer, count in sorted_layers[:5]:
        if count > 0:
            print(f"  Layer {layer}: {count} 次激活")

# 对比不同词性的质心层
print("\n" + "=" * 80)
print("词性质心层对比")
print("=" * 80)

centroids = {}
for pos, pos_data in data['results'].items():
    centroids[pos] = pos_data['key_neurons']['weighted_center']

# 按质心层排序
sorted_pos = sorted(centroids.items(), key=lambda x: x[1])
for pos, center in sorted_pos:
    print(f"  {pos:15s}: {center:6.2f} (normalized: {center/27:.3f})")

print("\n结论:")
print("- 早期层激活的词性通常与基础语义处理相关")
print("- 晚期层激活的词性通常与上下文整合和任务导向相关")
