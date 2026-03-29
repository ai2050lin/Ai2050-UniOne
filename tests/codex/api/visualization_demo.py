#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成可视化示例数据
用于前端可视化测试
"""
import numpy as np
import json
from pathlib import Path


def generate_shared_bearing_demo_data():
    """生成共享承载机制示例数据"""
    
    # 模拟跨家族共享数据
    np.random.seed(42)
    
    data = {
        "activations": np.random.randn(120, 256) * 0.1 + 0.2,  # 120个概念，256个参数
        "base_load": np.random.rand(120) * 0.5 + 0.3,
        "family_hit_count": np.random.randint(10, 100, 120),
        "concept_ids": [f"concept_{i}" for i in range(120)],
        "adjacency_matrix": np.random.rand(120, 120) * 0.2
    }
    
    # 使邻接矩阵对称
    data["adjacency_matrix"] = (data["adjacency_matrix"] + data["adjacency_matrix"].T) / 2
    np.fill_diagonal(data["adjacency_matrix"], 0)
    
    return data


def generate_cross_model_demo_data():
    """生成跨模型对比示例数据"""
    
    # 模拟3个模型的数据
    concept_ids = ["apple", "banana", "orange", "grape", "pear"]
    
    data = {
        "concepts": concept_ids,
        "models": {
            "gpt2": {
                "apple": {"activations": np.random.randn(10, 100)},
                "banana": {"activations": np.random.randn(10, 100)},
                "orange": {"activations": np.random.randn(10, 100)},
                "grape": {"activations": np.random.randn(10, 100)},
                "pear": {"activations": np.random.randn(10, 100)}
            },
            "qwen3": {
                "apple": {"activations": np.random.randn(10, 100)},
                "banana": {"activations": np.random.randn(10, 100)},
                "orange": {"activations": np.random.randn(10, 100)},
                "grape": {"activations": np.random.randn(10, 100)},
                "pear": {"activations": np.random.randn(10, 100)}
            },
            "deepseek7b": {
                "apple": {"activations": np.random.randn(10, 100)},
                "banana": {"activations": np.random.randn(10, 100)},
                "orange": {"activations": np.random.randn(10, 100)},
                "grape": {"activations": np.random.randn(10, 100)},
                "pear": {"activations": np.random.randn(10, 100)}
            }
        }
    }
    
    return data


def generate_temporal_demo_data():
    """生成时间演化示例数据"""
    
    # 模拟10个checkpoint的演化
    concept_id = "apple"
    
    data = {
        "concept_id": concept_id,
        "checkpoints": list(range(1, 11)),
        "activations": [
            np.random.randn(10, 100) * 0.1 + 0.1 + i * 0.05  # 逐渐增强
            for i in range(10)
        ]
    }
    
    return data


def generate_intervention_demo_data():
    """生成干预结果示例数据"""
    
    data = {
        "param_id": "param_123",
        "intervention_type": "ablation",
        "performance_before": 0.85,
        "performance_after": 0.45,
        "delta": -0.40,
        "metadata": {
            "param_name": "layer_5_neuron_42",
            "model": "deepseek7b"
        }
    }
    
    return data


def save_demo_data():
    """保存示例数据"""
    
    # 创建目录
    base_path = Path("d:/develop/TransformerLens-main/tests/codex/data")
    processed_path = base_path / "processed" / "deepseek7b" / "mechanisms"
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # 保存共享承载数据
    shared_bearing_data = generate_shared_bearing_demo_data()
    np.savez_compressed(
        processed_path / "cross_family_bearing.npz",
        **{k: v if isinstance(v, np.ndarray) else v for k, v in shared_bearing_data.items()}
    )
    
    # 保存元数据
    with open(processed_path / "cross_family_metadata.json", 'w', encoding='utf-8') as f:
        json.dump({
            "family_type": "cross_family",
            "model": "deepseek7b",
            "num_concepts": 120,
            "num_parameters": 256,
            "generation_date": "2026-03-28"
        }, f, indent=2)
    
    # 保存家族共享数据
    family_shared_data = generate_shared_bearing_demo_data()
    np.savez_compressed(
        processed_path / "family_shared_bearing.npz",
        **{k: v if isinstance(v, np.ndarray) else v for k, v in family_shared_data.items()}
    )
    
    # 保存元数据
    with open(processed_path / "family_shared_metadata.json", 'w', encoding='utf-8') as f:
        json.dump({
            "family_type": "family_shared",
            "model": "deepseek7b",
            "num_concepts": 120,
            "num_parameters": 256,
            "generation_date": "2026-03-28"
        }, f, indent=2)
    
    # 保存跨模型数据
    cross_model_path = base_path / "processed" / "cross_model"
    cross_model_path.mkdir(parents=True, exist_ok=True)
    
    cross_model_data = generate_cross_model_demo_data()
    with open(cross_model_path / "cross_model_demo.json", 'w', encoding='utf-8') as f:
        # 转换numpy数组为列表以便JSON序列化
        serializable_data = json.dumps(cross_model_data, default=str)
        f.write(serializable_data)
    
    # 保存时间演化数据
    temporal_path = base_path / "raw_scans" / "temporal"
    temporal_path.mkdir(parents=True, exist_ok=True)
    
    temporal_data = generate_temporal_demo_data()
    for i, (checkpoint, activations) in enumerate(zip(temporal_data["checkpoints"], temporal_data["activations"])):
        np.savez_compressed(
            temporal_path / f"apple_ckpt{checkpoint}.npz",
            activations=activations
        )
    
    # 保存干预结果数据
    intervention_path = base_path / "processed" / "interventions"
    intervention_path.mkdir(parents=True, exist_ok=True)
    
    intervention_data = generate_intervention_demo_data()
    np.savez_compressed(
        intervention_path / "param_123_ablation.npz",
        performance_before=intervention_data["performance_before"],
        performance_after=intervention_data["performance_after"],
        delta=intervention_data["delta"]
    )
    
    # 保存元数据
    metadata_path = base_path / "metadata" / "interventions"
    metadata_path.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_path / "param_123_ablation_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(intervention_data["metadata"], f, indent=2)
    
    print("✓ 示例数据已生成")
    print(f"  - 共享承载数据: {processed_path}")
    print(f"  - 跨模型数据: {cross_model_path}")
    print(f"  - 时间演化数据: {temporal_path}")
    print(f"  - 干预结果数据: {intervention_path}")


if __name__ == "__main__":
    print("生成可视化示例数据...")
    save_demo_data()
    print("\nOK 完成！现在可以测试可视化API了。")
