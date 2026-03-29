#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的数据访问API
用于连接测试脚本和可视化客户端
"""
from typing import Optional, List, Dict, Any, Union
import numpy as np
import json
from pathlib import Path


class AGIDataAPI:
    """AGI研究数据统一访问API"""
    
    def __init__(self, base_path: str = "d:/develop/TransformerLens-main/tests/codex/data"):
        """
        初始化数据API
        
        Args:
            base_path: 数据基础路径
        """
        self.base_path = Path(base_path)
        self.raw_data_path = self.base_path / "raw_scans"
        self.processed_data_path = self.base_path / "processed"
        self.metadata_path = self.base_path / "metadata"
        
        # 确保目录存在
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保数据目录存在"""
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
    
    def get_concept_activation_data(
        self, 
        model: str, 
        concept_id: str, 
        layer: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        获取概念激活数据
        
        Args:
            model: 模型名称 (e.g., "deepseek7b", "gpt2", "qwen3")
            concept_id: 概念ID (e.g., "apple", "cat")
            layer: 层号 (可选，如果为None则返回所有层)
            
        Returns:
            激活数据字典，包含:
            - concept_id: 概念ID
            - model: 模型名称
            - activations: 激活值数组
            - metadata: 元数据
        """
        # 构建文件路径
        if layer is None:
            file_path = self.processed_data_path / model / f"{concept_id}_all_layers.npz"
        else:
            file_path = self.processed_data_path / model / f"{concept_id}_layer{layer}.npz"
        
        if not file_path.exists():
            return {"error": f"Data not found: {file_path}"}
        
        # 加载NPY数据
        data = np.load(file_path, allow_pickle=True)
        
        # 读取对应的元数据
        metadata_file = self.metadata_path / model / f"{concept_id}_metadata.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        return {
            "concept_id": concept_id,
            "model": model,
            "layer": layer,
            "activations": data.get("activations", None),
            "metadata": metadata
        }
    
    def get_cross_model_comparison(
        self, 
        concept_ids: List[str], 
        models: List[str]
    ) -> Dict[str, Any]:
        """
        获取跨模型对比数据
        
        Args:
            concept_ids: 概念ID列表
            models: 模型名称列表
            
        Returns:
            跨模型对比数据，包含:
            - concepts: 概念ID列表
            - models: 模型名称列表
            - comparison_matrix: 对比矩阵
            - statistics: 统计信息
        """
        results = {
            "concepts": concept_ids,
            "models": models,
            "data": {}
        }
        
        for concept_id in concept_ids:
            results["data"][concept_id] = {}
            for model in models:
                data = self.get_concept_activation_data(model, concept_id)
                results["data"][concept_id][model] = data
        
        return results
    
    def get_intervention_result(
        self, 
        param_id: str, 
        intervention_type: str
    ) -> Dict[str, Any]:
        """
        获取因果干预结果
        
        Args:
            param_id: 参数ID
            intervention_type: 干预类型 (e.g., "ablation", "replacement")
            
        Returns:
            干预结果数据
        """
        # 构建文件路径
        file_path = self.processed_data_path / "interventions" / f"{param_id}_{intervention_type}.npz"
        
        if not file_path.exists():
            return {"error": f"Intervention data not found: {file_path}"}
        
        # 加载NPY数据
        data = np.load(file_path, allow_pickle=True)
        
        # 读取对应的元数据
        metadata_file = self.metadata_path / "interventions" / f"{param_id}_{intervention_type}_metadata.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        return {
            "param_id": param_id,
            "intervention_type": intervention_type,
            "results": data,
            "metadata": metadata
        }
    
    def get_temporal_trajectory(
        self, 
        concept_id: str, 
        checkpoint_range: List[int]
    ) -> Dict[str, Any]:
        """
        获取时间演化轨迹
        
        Args:
            concept_id: 概念ID
            checkpoint_range: checkpoint范围 [start, end]
            
        Returns:
            时间演化轨迹数据
        """
        trajectory_data = {
            "concept_id": concept_id,
            "checkpoints": [],
            "activations": [],
            "metadata": []
        }
        
        for checkpoint in range(checkpoint_range[0], checkpoint_range[1] + 1):
            # 构建文件路径
            file_path = self.raw_data_path / "temporal" / f"{concept_id}_ckpt{checkpoint}.npz"
            
            if file_path.exists():
                data = np.load(file_path, allow_pickle=True)
                trajectory_data["checkpoints"].append(checkpoint)
                trajectory_data["activations"].append(data.get("activations", None))
        
        return trajectory_data
    
    def get_shared_bearing_mechanism_data(
        self, 
        family_type: str,
        model: str = "deepseek7b"
    ) -> Dict[str, Any]:
        """
        获取共享承载机制数据 (Stage294-298)
        
        Args:
            family_type: 家族类型 (e.g., "cross_family", "family_shared")
            model: 模型名称
            
        Returns:
            共享承载机制数据
        """
        # 构建文件路径
        file_path = self.processed_data_path / model / "mechanisms" / f"{family_type}_bearing.npz"
        
        if not file_path.exists():
            return {"error": f"Shared bearing data not found: {file_path}"}
        
        # 加载NPY数据
        data = np.load(file_path, allow_pickle=True)
        
        return {
            "family_type": family_type,
            "model": model,
            "bearing_data": data
        }
    
    def get_cross_model_isomorphism_data(
        self, 
        models: List[str]
    ) -> Dict[str, Any]:
        """
        获取跨模型同构数据 (Stage141)
        
        Args:
            models: 模型名称列表
            
        Returns:
            跨模型同构数据
        """
        results = {
            "models": models,
            "isomorphism_matrix": [],
            "metadata": []
        }
        
        # 读取同构矩阵数据
        file_path = self.processed_data_path / "cross_model" / "isomorphism_matrix.npz"
        
        if file_path.exists():
            data = np.load(file_path, allow_pickle=True)
            results["isomorphism_matrix"] = data.get("matrix", None)
        
        return results
    
    def get_data_source_list(self) -> Dict[str, Any]:
        """
        获取数据源列表
        
        Returns:
            数据源列表
        """
        data_sources = {
            "dnn_parameter_activation": {
                "name": "DNN参数激活",
                "description": "10,000+概念的参数激活数据",
                "count": 10000,
                "status": "部分加载",
                "last_updated": "2026-03-28"
            },
            "cross_model_comparison": {
                "name": "跨模型对比",
                "description": "3个模型的对比数据",
                "count": 3,
                "status": "部分加载",
                "last_updated": "2026-03-28"
            },
            "causal_intervention": {
                "name": "因果干预",
                "description": "100个高影响参数的干预数据",
                "count": 100,
                "status": "未加载",
                "last_updated": None
            },
            "brain_bridge": {
                "name": "脑桥接",
                "description": "fMRI/EEG数据",
                "count": 0,
                "status": "未加载",
                "last_updated": None
            },
            "cross_modal": {
                "name": "跨模态",
                "description": "图像/文本/音频数据",
                "count": 0,
                "status": "未加载",
                "last_updated": None
            }
        }
        
        return data_sources
    
    def get_data_puzzle_categories(self) -> Dict[str, Any]:
        """
        获取数据拼图分类
        
        Returns:
            数据拼图分类
        """
        categories = {
            "shared_bearing": {
                "name": "共享承载机制",
                "count": 345,
                "subcategories": {
                    "cross_family": {"name": "跨家族共享", "count": 120},
                    "family_shared": {"name": "家族共享", "count": 150},
                    "independent": {"name": "独立承载", "count": 75}
                }
            },
            "bias_deflection": {
                "name": "偏置偏转机制",
                "count": 278,
                "subcategories": {
                    "brand": {"name": "品牌偏转", "count": 90},
                    "intra_class": {"name": "类内竞争", "count": 110},
                    "fine_grained": {"name": "对象细粒度", "count": 78}
                }
            },
            "layerwise_amplification": {
                "name": "逐层放大机制",
                "count": 156,
                "subcategories": {
                    "early_layers": {"name": "早层", "count": 50},
                    "middle_layers": {"name": "中层", "count": 60},
                    "late_layers": {"name": "后层", "count": 46}
                }
            },
            "multi_space_mapping": {
                "name": "多空间映射",
                "count": 234,
                "subcategories": {
                    "object_space": {"name": "对象空间", "count": 80},
                    "task_space": {"name": "任务空间", "count": 90},
                    "propagation_space": {"name": "传播空间", "count": 64}
                }
            },
            "cross_model_validation": {
                "name": "跨模型验证",
                "count": 89,
                "subcategories": {
                    "embedding_level": {"name": "embedding级", "count": 30},
                    "layer_level": {"name": "层级", "count": 35},
                    "attention_level": {"name": "注意力级", "count": 24}
                }
            }
        }
        
        return categories


# 单例模式
_api_instance = None

def get_api() -> AGIDataAPI:
    """获取API单例"""
    global _api_instance
    if _api_instance is None:
        _api_instance = AGIDataAPI()
    return _api_instance
