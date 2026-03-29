# -*- coding: utf-8 -*-
"""
测试层间关联分析API
验证所有层（包括静态编码层）是否正确返回数据
"""

import sys
import os
import json

# 添加API目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

from layer_association_analyzer import LayerAssociationAnalyzer

def test_all_layers_have_data():
    """测试所有层是否都有数据"""
    print("=" * 60)
    print("测试：验证所有层是否正确返回数据")
    print("=" * 60)

    analyzer = LayerAssociationAnalyzer(model_name='deepseek7b')

    # 测试概念
    test_concepts = ['apple', 'banana', 'orange', 'dog', 'cat']

    for concept in test_concepts:
        print(f"\n{'=' * 60}")
        print(f"概念: {concept}")
        print(f"{'=' * 60}")

        result = analyzer.analyze_concept_layers(concept)

        # 检查所有层是否存在
        expected_layers = ['static_encoding', 'dynamic_path', 'result_recovery',
                          'propagation_encoding', 'semantic_role']

        for layer_id in expected_layers:
            if layer_id not in result['layers']:
                print(f"[错误] {layer_id} 层不存在")
                continue

            layer = result['layers'][layer_id]
            nodes_count = layer['statistics']['num_nodes']
            edges_count = layer['statistics']['num_edges']
            avg_activation = layer['statistics']['avg_activation']

            # 检查数据完整性
            has_data = nodes_count > 0 and edges_count >= 0

            if has_data:
                status = "[OK]"
            else:
                status = "[警告]"

            print(f"{status} {layer['layer_name']}:")
            print(f"   层ID: {layer_id}")
            print(f"   节点数: {nodes_count}")
            print(f"   边数: {edges_count}")
            print(f"   平均激活度: {avg_activation:.3f}")
            print(f"   位置: {layer['position']}")

            # 检查节点数据
            if len(layer['nodes']) > 0:
                first_node = layer['nodes'][0]
                print(f"   第一个节点: {first_node['id']}")
                print(f"   节点位置: {first_node['position']}")
                print(f"   节点激活度: {first_node['activation']:.3f}")
            else:
                print(f"   [警告] 没有节点数据!")

        print(f"\n关联性分析: {len(result['associations'])} 个关联")
        print(f"流向路径: {len(result['flow_paths'])} 条路径")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

def test_static_encoding_specific():
    """专门测试静态编码层"""
    print("\n" + "=" * 60)
    print("专项测试：静态编码层数据验证")
    print("=" * 60)

    analyzer = LayerAssociationAnalyzer(model_name='deepseek7b')
    concept = 'apple'

    result = analyzer.analyze_concept_layers(concept)
    static_layer = result['layers']['static_encoding']

    print(f"\n概念: {concept}")
    print(f"静态编码层详细信息:")
    print(f"  层名称: {static_layer['layer_name']}")
    print(f"  层描述: {static_layer['layer_description']}")
    print(f"  层颜色: {static_layer['color']}")
    print(f"  层位置: {static_layer['position']}")

    print(f"\n节点数据 ({len(static_layer['nodes'])} 个):")
    for i, node in enumerate(static_layer['nodes'][:5]):  # 显示前5个节点
        print(f"  节点 {i+1}:")
        print(f"    ID: {node['id']}")
        print(f"    位置: {node['position']}")
        print(f"    激活度: {node['activation']:.3f}")
        print(f"    大小: {node['size']:.1f}")

    if len(static_layer['nodes']) > 5:
        print(f"  ... 还有 {len(static_layer['nodes']) - 5} 个节点")

    print(f"\n边数据 ({len(static_layer['edges'])} 条):")
    for i, edge in enumerate(static_layer['edges'][:5]):  # 显示前5条边
        print(f"  边 {i+1}:")
        print(f"    源: {edge['source']}")
        print(f"    目标: {edge['target']}")
        print(f"    强度: {edge['strength']:.3f}")

    if len(static_layer['edges']) > 5:
        print(f"  ... 还有 {len(static_layer['edges']) - 5} 条边")

    print(f"\n关键特征 ({len(static_layer['key_features'])} 个):")
    for i, feature in enumerate(static_layer['key_features']):
        print(f"  特征 {i+1}:")
        print(f"    ID: {feature['id']}")
        print(f"    名称: {feature['name']}")
        print(f"    重要性: {feature['importance']:.3f}")

    print(f"\n统计信息:")
    stats = static_layer['statistics']
    print(f"  节点总数: {stats['num_nodes']}")
    print(f"  边总数: {stats['num_edges']}")
    print(f"  平均激活度: {stats['avg_activation']:.3f}")
    print(f"  最大激活度: {stats['max_activation']:.3f}")

    # 验证数据完整性
    print(f"\n数据完整性检查:")
    checks = [
        ("节点存在", len(static_layer['nodes']) > 0),
        ("边存在", len(static_layer['edges']) >= 0),
        ("关键特征存在", len(static_layer['key_features']) > 0),
        ("激活度有效", 0 <= stats['avg_activation'] <= 1),
        ("统计信息完整", all(key in stats for key in ['num_nodes', 'num_edges', 'avg_activation', 'max_activation']))
    ]

    all_passed = True
    for check_name, check_result in checks:
        status = "[OK]" if check_result else "[错误]"
        print(f"{status} {check_name}")
        if not check_result:
            all_passed = False

    if all_passed:
        print("\n[成功] 静态编码层数据完整且有效")
    else:
        print("\n[失败] 静态编码层数据存在问题")

if __name__ == '__main__':
    test_all_layers_have_data()
    test_static_encoding_specific()
