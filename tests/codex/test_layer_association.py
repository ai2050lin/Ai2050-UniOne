#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试多层3D可视化关联分析API
"""

import sys
from pathlib import Path

# 添加API目录到路径
api_dir = Path(__file__).parent / 'api'
sys.path.insert(0, str(api_dir))

from layer_association_analyzer import LayerAssociationAnalyzer


def test_single_concept_analysis():
    """测试单个概念分析"""
    print("=" * 60)
    print("测试1: 单个概念分析")
    print("=" * 60)

    analyzer = LayerAssociationAnalyzer(model_name='deepseek7b')

    # 分析概念"apple"
    print("\n分析概念 'apple':")
    result = analyzer.analyze_concept_layers('apple')

    print(f"  概念: {result['concept']}")
    print(f"  模型: {result['model']}")
    print(f"  时间戳: {result['timestamp']}")

    # 检查各层数据
    print(f"\n层数据:")
    for layer_id, layer_data in result['layers'].items():
        stats = layer_data['statistics']
        print(f"  {layer_id}:")
        print(f"    名称: {layer_data['layer_name']}")
        print(f"    节点数: {stats['num_nodes']}")
        print(f"    边数: {stats['num_edges']}")
        print(f"    平均激活度: {stats['avg_activation']:.3f}")
        print(f"    最大激活度: {stats['max_activation']:.3f}")

    # 检查关联数据
    print(f"\n关联数据:")
    print(f"  关联数: {len(result['associations'])}")
    for i, assoc in enumerate(result['associations']):
        print(f"  关联 {i+1}:")
        print(f"    {assoc['source_layer']} -> {assoc['target_layer']}")
        print(f"    强度: {assoc['strength']:.3f}")
        print(f"    共享特征: {len(assoc['shared_features'])}")
        print(f"    跨层连接: {len(assoc['cross_layer_edges'])}")

    # 检查流向路径
    print(f"\n流向路径:")
    print(f"  路径数: {len(result['flow_paths'])}")
    for i, path in enumerate(result['flow_paths']):
        print(f"  路径 {i+1}: {path['path_name']}")
        print(f"    涉及层数: {len(path['layers'])}")
        print(f"    总强度: {path['total_strength']:.3f}")

    print("\n✅ 单个概念分析测试通过")
    return result


def test_concept_comparison():
    """测试概念比较"""
    print("\n" + "=" * 60)
    print("测试2: 概念比较")
    print("=" * 60)

    analyzer = LayerAssociationAnalyzer(model_name='deepseek7b')

    # 比较概念"apple"和"banana"
    print("\n比较概念 'apple' 和 'banana':")
    comparison = analyzer.compare_concepts('apple', 'banana')

    print(f"  概念1: {comparison['concept1']}")
    print(f"  概念2: {comparison['concept2']}")
    print(f"  整体相似度: {comparison['overall_similarity']:.3f}")

    # 逐层比较
    print(f"\n逐层相似度:")
    for layer_id, sim in comparison['layer_comparisons'].items():
        print(f"  {layer_id}:")
        print(f"    激活度相关性: {sim['activation_correlation']:.3f}")
        print(f"    结构相似度: {sim['structure_similarity']:.3f}")
        print(f"    特征重叠度: {sim['feature_overlap']:.3f}")
        print(f"    整体相似度: {sim['overall']:.3f}")

    print("\n✅ 概念比较测试通过")
    return comparison


def test_multiple_concepts():
    """测试多个概念分析"""
    print("\n" + "=" * 60)
    print("测试3: 多个概念分析")
    print("=" * 60)

    analyzer = LayerAssociationAnalyzer(model_name='deepseek7b')

    concepts = ['apple', 'banana', 'orange', 'dog', 'cat']

    print(f"\n分析 {len(concepts)} 个概念:")
    results = {}

    for concept in concepts:
        print(f"\n分析概念 '{concept}':")
        result = analyzer.analyze_concept_layers(concept)
        results[concept] = result

        # 显示摘要信息
        print(f"  层数: {len(result['layers'])}")
        print(f"  关联数: {len(result['associations'])}")
        print(f"  流向路径: {len(result['flow_paths'])}")

        # 计算总体激活度
        avg_activations = [
            layer['statistics']['avg_activation']
            for layer in result['layers'].values()
        ]
        overall_activation = sum(avg_activations) / len(avg_activations)
        print(f"  总体激活度: {overall_activation:.3f}")

    # 比较所有概念
    print(f"\n概念间相似度矩阵:")
    print(f"{'':10}", end='')
    for c1 in concepts:
        print(f"{c1:10}", end='')
    print()

    for c1 in concepts:
        print(f"{c1:10}", end='')
        for c2 in concepts:
            if c1 == c2:
                print(f"{'1.000':10}", end='')
            else:
                # 临时比较（实际应该使用缓存）
                comp = analyzer.compare_concepts(c1, c2)
                print(f"{comp['overall_similarity']:10.3f}", end='')
        print()

    print("\n✅ 多个概念分析测试通过")
    return results


def test_layer_details():
    """测试层详细信息"""
    print("\n" + "=" * 60)
    print("测试4: 层详细信息")
    print("=" * 60)

    analyzer = LayerAssociationAnalyzer(model_name='deepseek7b')

    result = analyzer.analyze_concept_layers('apple')

    # 分析每个层的详细信息
    for layer_id, layer_data in result['layers'].items():
        print(f"\n{'='*60}")
        print(f"层: {layer_data['layer_name']} ({layer_id})")
        print(f"{'='*60}")
        print(f"描述: {layer_data['layer_description']}")
        print(f"位置: {layer_data['position']}")
        print(f"颜色: {layer_data['color']}")

        print(f"\n关键特征 (Top 5):")
        for i, feature in enumerate(layer_data['key_features'][:5]):
            print(f"  {i+1}. {feature['name']}")
            print(f"     重要性: {feature['importance']:.3f}")
            print(f"     激活度: {feature['activation']:.3f}")

        print(f"\n统计信息:")
        stats = layer_data['statistics']
        print(f"  节点数: {stats['num_nodes']}")
        print(f"  边数: {stats['num_edges']}")
        print(f"  平均激活度: {stats['avg_activation']:.3f}")
        print(f"  最大激活度: {stats['max_activation']:.3f}")

        # 分析节点分布
        activations = [n['activation'] for n in layer_data['nodes']]
        high_activation = sum(1 for a in activations if a > 0.7)
        medium_activation = sum(1 for a in activations if 0.4 <= a <= 0.7)
        low_activation = sum(1 for a in activations if a < 0.4)

        print(f"\n节点激活度分布:")
        print(f"  高激活 (>0.7): {high_activation} ({high_activation/len(activations)*100:.1f}%)")
        print(f"  中激活 (0.4-0.7): {medium_activation} ({medium_activation/len(activations)*100:.1f}%)")
        print(f"  低激活 (<0.4): {low_activation} ({low_activation/len(activations)*100:.1f}%)")

    print("\n✅ 层详细信息测试通过")


def test_association_details():
    """测试关联详细信息"""
    print("\n" + "=" * 60)
    print("测试5: 关联详细信息")
    print("=" * 60)

    analyzer = LayerAssociationAnalyzer(model_name='deepseek7b')

    result = analyzer.analyze_concept_layers('apple')

    print("\n层间关联详细分析:")
    for i, assoc in enumerate(result['associations']):
        source_layer = result['layers'][assoc['source_layer']]
        target_layer = result['layers'][assoc['target_layer']]

        print(f"\n关联 {i+1}: {source_layer['layer_name']} -> {target_layer['layer_name']}")
        print(f"  关联强度: {assoc['strength']:.3f}")

        # 信息流
        print(f"\n  信息流:")
        info_flow = assoc['information_flow']
        print(f"    正向流: {info_flow['forward_flow']:.3f}")
        print(f"    反向流: {info_flow['backward_flow']:.3f}")
        print(f"    双向流: {info_flow['bidirectional_flow']:.3f}")
        print(f"    总流量: {info_flow['total_flow']:.3f}")

        # 共享特征
        print(f"\n  共享特征 ({len(assoc['shared_features'])}):")
        for j, feature in enumerate(assoc['shared_features'][:3]):
            print(f"    {j+1}. {feature['source_feature']} <-> {feature['target_feature']}")
            print(f"       相似度: {feature['similarity']:.3f}")
            print(f"       保留率: {feature['preservation']:.3f}")

        # 跨层连接
        print(f"\n  跨层连接 ({len(assoc['cross_layer_edges'])}):")
        for j, edge in enumerate(assoc['cross_layer_edges'][:3]):
            print(f"    {j+1}. {edge['source']} -> {edge['target']}")
            print(f"       强度: {edge['strength']:.3f}")

    print("\n✅ 关联详细信息测试通过")


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("多层3D可视化关联分析API测试套件")
    print("="*60)

    try:
        # 运行测试
        test_single_concept_analysis()
        test_concept_comparison()
        test_multiple_concepts()
        test_layer_details()
        test_association_details()

        # 总结
        print("\n" + "="*60)
        print("所有测试通过！")
        print("="*60)
        print("\n测试总结:")
        print("✅ 单个概念分析")
        print("✅ 概念比较")
        print("✅ 多个概念分析")
        print("✅ 层详细信息")
        print("✅ 关联详细信息")
        print("\n所有功能正常工作！")

        return 0

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
