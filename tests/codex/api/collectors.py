#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试结果收集器
自动收集测试脚本结果并保存到数据库
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import time
import numpy as np
from datetime import datetime


class TestResultCollector:
    """测试结果收集器"""
    
    def __init__(self, base_path: str = "d:/develop/TransformerLens-main/tests/codex/data"):
        """
        初始化收集器
        
        Args:
            base_path: 数据基础路径
        """
        self.base_path = Path(base_path)
        self.results_path = self.base_path / "results"
        self.metadata_path = self.base_path / "metadata"
        
        # 确保目录存在
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
    
    def collect_from_stage(
        self, 
        stage_id: int, 
        result_data: Dict[str, Any],
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        从stage脚本收集结果
        
        Args:
            stage_id: 测试stage ID
            result_data: 结果数据
            additional_metadata: 额外的元数据
            
        Returns:
            保存结果
        """
        # 生成元数据
        metadata = {
            "stage_id": stage_id,
            "timestamp": datetime.now().isoformat(),
            "collection_time": time.time()
        }
        
        # 合并额外的元数据
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # 保存结果数据
        result_filename = f"stage{stage_id}_result.npz"
        result_path = self.results_path / result_filename
        
        # 保存NPY数据
        np.savez_compressed(result_path, **result_data)
        
        # 保存元数据
        metadata_filename = f"stage{stage_id}_metadata.json"
        metadata_path = self.metadata_path / metadata_filename
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True,
            "stage_id": stage_id,
            "result_path": str(result_path),
            "metadata_path": str(metadata_path),
            "timestamp": metadata["timestamp"]
        }
    
    def save_to_database(self, result: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        保存到数据库
        
        Args:
            result: 结果数据
            metadata: 元数据
            
        Returns:
            保存结果
        """
        # TODO: 集成实际的数据库
        # 目前先使用文件系统存储
        
        return {
            "success": True,
            "message": "结果已保存（当前使用文件系统，后续将集成PostgreSQL）"
        }
    
    def generate_metadata(
        self, 
        stage_id: int,
        model: str,
        concept_ids: List[str],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        生成元数据
        
        Args:
            stage_id: 测试stage ID
            model: 模型名称
            concept_ids: 概念ID列表
            parameters: 参数配置
            
        Returns:
            元数据字典
        """
        metadata = {
            "stage_id": stage_id,
            "model": model,
            "concepts": concept_ids,
            "parameters": parameters,
            "timestamp": datetime.now().isoformat(),
            "collection_method": "auto"
        }
        
        return metadata
    
    def get_result_by_stage(self, stage_id: int) -> Dict[str, Any]:
        """
        根据stage ID获取结果
        
        Args:
            stage_id: 测试stage ID
            
        Returns:
            结果数据
        """
        # 加载结果数据
        result_path = self.results_path / f"stage{stage_id}_result.npz"
        
        if not result_path.exists():
            return {"error": f"Result not found for stage{stage_id}"}
        
        # 加载NPY数据
        data = np.load(result_path, allow_pickle=True)
        result_data = {k: data[k] for k in data.files}
        
        # 加载元数据
        metadata_path = self.metadata_path / f"stage{stage_id}_metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        return {
            "result": result_data,
            "metadata": metadata
        }
    
    def list_all_results(self) -> List[Dict[str, Any]]:
        """
        列出所有已收集的结果
        
        Returns:
            结果列表
        """
        results = []
        
        for metadata_file in self.metadata_path.glob("stage*_metadata.json"):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                results.append(metadata)
        
        return results


# 单例模式
_collector_instance = None

def get_collector() -> TestResultCollector:
    """获取收集器单例"""
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = TestResultCollector()
    return _collector_instance
