#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据API框架
验证基础功能是否正常工作
"""
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from api.data_api import get_api
from api.collectors import get_collector
from api.quality_checker import get_checker


def test_data_api():
    """测试数据API"""
    print("=" * 60)
    print("测试数据API功能")
    print("=" * 60)
    
    api = get_api()
    
    # 测试获取数据源列表
    print("\n1. 获取数据源列表...")
    data_sources = api.get_data_source_list()
    print(f"✓ 数据源数量: {len(data_sources)}")
    for source_name, source_info in data_sources.items():
        print(f"  - {source_info['name']}: {source_info['count']} ({source_info['status']})")
    
    # 测试获取数据拼图分类
    print("\n2. 获取数据拼图分类...")
    categories = api.get_data_puzzle_categories()
    print(f"✓ 分类数量: {len(categories)}")
    for category_name, category_info in categories.items():
        print(f"  - {category_info['name']}: {category_info['count']}")
    
    print("\n✓ 数据API测试通过")


def test_collector():
    """测试结果收集器"""
    print("\n" + "=" * 60)
    print("测试结果收集器功能")
    print("=" * 60)
    
    collector = get_collector()
    
    # 测试收集测试结果
    print("\n1. 收集测试结果...")
    import numpy as np
    
    test_data = {
        "activations": np.random.randn(10, 100),
        "metrics": {
            "accuracy": 0.85,
            "loss": 0.15
        }
    }
    
    result = collector.collect_from_stage(
        stage_id=999,
        result_data=test_data,
        additional_metadata={
            "model": "deepseek7b",
            "test_name": "demo_test"
        }
    )
    
    print(f"✓ 结果已保存到: {result['result_path']}")
    print(f"✓ 元数据已保存到: {result['metadata_path']}")
    
    # 测试获取结果
    print("\n2. 获取测试结果...")
    retrieved = collector.get_result_by_stage(999)
    print(f"✓ 检索到的激活数据形状: {retrieved['result']['activations'].shape}")
    
    print("\n✓ 结果收集器测试通过")


def test_quality_checker():
    """测试质量检查器"""
    print("\n" + "=" * 60)
    print("测试质量检查器功能")
    print("=" * 60)
    
    checker = get_checker()
    
    # 创建测试数据
    test_data = {
        "activations": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "metadata": {
            "layer": 1,
            "concept_id": "apple",
            "model": "deepseek7b",
            "random_seed": 42,
            "model_version": "v1.0"
        }
    }
    
    # 测试完整性检查
    print("\n1. 检查完整性...")
    completeness = checker.check_completeness(test_data)
    print(f"✓ 完整性分数: {completeness['score']:.2f}")
    print(f"✓ 状态: {completeness['status']}")
    
    # 测试一致性检查
    print("\n2. 检查一致性...")
    consistency = checker.check_consistency(test_data)
    print(f"✓ 一致性分数: {consistency['score']:.2f}")
    print(f"✓ 状态: {consistency['status']}")
    
    # 测试可复现性检查
    print("\n3. 检查可复现性...")
    reproducibility = checker.check_reproducibility(test_data)
    print(f"✓ 可复现性分数: {reproducibility['score']:.2f}")
    print(f"✓ 状态: {reproducibility['status']}")
    
    # 测试综合质量报告
    print("\n4. 生成综合质量报告...")
    report = checker.check_all_metrics(test_data)
    print(f"✓ 综合分数: {report['overall_score']:.2f}")
    print(f"✓ 总问题数: {report['total_issues']}")
    print(f"✓ 总警告数: {report['total_warnings']}")
    print(f"✓ 状态: {report['status']}")
    
    print("\n✓ 质量检查器测试通过")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AGI研究数据API框架测试")
    print("=" * 60)
    
    try:
        test_data_api()
        test_collector()
        test_quality_checker()
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
        print("\n下一步：")
        print("1. 启动API服务: python tests/codex/api/server.py")
        print("2. 访问API文档: http://localhost:8000/docs")
        print("3. 集成前端可视化组件")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
