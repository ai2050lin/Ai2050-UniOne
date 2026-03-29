#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化助手函数
为前端提供可视化所需的数据处理
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path


class VisualizationHelper:
    """可视化助手"""
    
    def __init__(self, data_api=None):
        """
        初始化可视化助手
        
        Args:
            data_api: 数据API实例
        """
        if data_api is None:
            from api.data_api import get_api
            data_api = get_api()
        
        self.data_api = data_api
    
    def prepare_shared_bearing_heatmap_data(
        self, 
        family_type: str,
        model: str = "deepseek7b"
    ) -> Dict[str, Any]:
        """
        准备共享承载机制热图数据
        
        Args:
            family_type: 家族类型
            model: 模型名称
            
        Returns:
            热图数据
        """
        # 获取原始数据
        result = self.data_api.get_shared_bearing_mechanism_data(
            family_type=family_type,
            model=model
        )
        
        if "error" in result:
            return {"error": result["error"]}
        
        bearing_data = result["bearing_data"]
        
        # 提取激活矩阵
        activations = bearing_data.get("activations", None)
        
        if activations is None:
            return {"error": "No activation data found"}
        
        # 准备热图数据
        heatmap_data = {
            "x": list(range(activations.shape[1])),
            "y": list(range(activations.shape[0])),
            "z": activations.tolist() if isinstance(activations, np.ndarray) else activations,
            "type": "heatmap",
            "colorscale": "Viridis",
            "reversescale": False,
            "showscale": True,
            "colorbar": {
                "title": "激活强度"
            }
        }
        
        return {
            "title": f"{family_type} 共享承载激活热图",
            "data": [heatmap_data],
            "layout": {
                "xaxis": {"title": "参数索引"},
                "yaxis": {"title": "概念索引"},
                "margin": {"l": 60, "r": 20, "t": 40, "b": 60}
            }
        }
    
    def prepare_bearing_scatter_plot_data(
        self,
        family_type: str,
        model: str = "deepseek7b"
    ) -> Dict[str, Any]:
        """
        准备承载机制散点图数据 (base_load vs family_hit_count)
        
        Args:
            family_type: 家族类型
            model: 模型名称
            
        Returns:
            散点图数据
        """
        # 获取原始数据
        result = self.data_api.get_shared_bearing_mechanism_data(
            family_type=family_type,
            model=model
        )
        
        if "error" in result:
            return {"error": result["error"]}
        
        bearing_data = result["bearing_data"]
        
        # 提取基础加载和家族命中数
        base_load = bearing_data.get("base_load", None)
        family_hit_count = bearing_data.get("family_hit_count", None)
        concept_ids = bearing_data.get("concept_ids", None)
        
        if base_load is None or family_hit_count is None:
            return {"error": "Missing required data fields"}
        
        # 准备散点图数据
        scatter_data = {
            "x": base_load.tolist() if isinstance(base_load, np.ndarray) else base_load,
            "y": family_hit_count.tolist() if isinstance(family_hit_count, np.ndarray) else family_hit_count,
            "mode": "markers",
            "type": "scatter",
            "marker": {
                "size": 8,
                "color": "rgba(100, 100, 255, 0.6)",
                "line": {"width": 1, "color": "rgba(100, 100, 255, 1)"}
            },
            "text": concept_ids if concept_ids else None
        }
        
        return {
            "title": f"{family_type} base_load vs family_hit_count",
            "data": [scatter_data],
            "layout": {
                "xaxis": {"title": "base_load"},
                "yaxis": {"title": "family_hit_count"},
                "hovermode": "closest"
            }
        }
    
    def prepare_cross_model_comparison_data(
        self,
        concept_ids: List[str],
        models: List[str]
    ) -> Dict[str, Any]:
        """
        准备跨模型对比数据
        
        Args:
            concept_ids: 概念ID列表
            models: 模型名称列表
            
        Returns:
            跨模型对比数据
        """
        # 获取跨模型对比数据
        result = self.data_api.get_cross_model_comparison(
            concept_ids=concept_ids,
            models=models
        )
        
        # 准备对比数据
        comparison_traces = []
        
        for model in models:
            model_data = []
            for concept_id in concept_ids:
                data = result["data"][concept_id][model]
                if "activations" in data and data["activations"] is not None:
                    activations = data["activations"]
                    # 计算平均激活
                    avg_activation = np.mean(activations)
                    model_data.append(avg_activation)
                else:
                    model_data.append(0)
            
            # 创建柱状图
            comparison_traces.append({
                "x": concept_ids,
                "y": model_data,
                "name": model,
                "type": "bar"
            })
        
        return {
            "title": "跨模型激活对比",
            "data": comparison_traces,
            "layout": {
                "barmode": "group",
                "xaxis": {"title": "概念"},
                "yaxis": {"title": "平均激活"},
                "hovermode": "closest"
            }
        }
    
    def prepare_temporal_trajectory_data(
        self,
        concept_id: str,
        checkpoint_range: List[int]
    ) -> Dict[str, Any]:
        """
        准备时间演化轨迹数据
        
        Args:
            concept_id: 概念ID
            checkpoint_range: checkpoint范围 [start, end]
            
        Returns:
            时间演化轨迹数据
        """
        # 获取时间演化数据
        result = self.data_api.get_temporal_trajectory(
            concept_id=concept_id,
            checkpoint_range=checkpoint_range
        )
        
        if len(result["activations"]) == 0:
            return {"error": "No temporal data found"}
        
        # 准备时间序列图数据
        # 计算每个checkpoint的平均激活
        avg_activations = []
        for activations in result["activations"]:
            if activations is not None and isinstance(activations, np.ndarray):
                avg_activation = np.mean(activations)
                avg_activations.append(avg_activation)
            else:
                avg_activations.append(0)
        
        trajectory_data = {
            "x": result["checkpoints"],
            "y": avg_activations,
            "mode": "lines+markers",
            "type": "scatter",
            "line": {"width": 2},
            "marker": {"size": 8}
        }
        
        return {
            "title": f"概念 '{concept_id}' 的时间演化轨迹",
            "data": [trajectory_data],
            "layout": {
                "xaxis": {"title": "Checkpoint"},
                "yaxis": {"title": "平均激活"},
                "hovermode": "closest"
            }
        }
    
    def prepare_network_graph_data(
        self,
        family_type: str,
        model: str = "deepseek7b"
    ) -> Dict[str, Any]:
        """
        准备网络图数据（参数间承载关系）
        
        Args:
            family_type: 家族类型
            model: 模型名称
            
        Returns:
            网络图数据
        """
        # 获取原始数据
        result = self.data_api.get_shared_bearing_mechanism_data(
            family_type=family_type,
            model=model
        )
        
        if "error" in result:
            return {"error": result["error"]}
        
        bearing_data = result["bearing_data"]
        
        # 提取邻接矩阵
        adjacency_matrix = bearing_data.get("adjacency_matrix", None)
        concept_ids = bearing_data.get("concept_ids", None)
        
        if adjacency_matrix is None:
            return {"error": "No adjacency matrix found"}
        
        # 准备网络图数据（使用Plotly的简化网络图）
        # 创建节点和边
        nodes = []
        edges = []
        
        if concept_ids:
            for i, concept_id in enumerate(concept_ids):
                nodes.append({
                    "id": i,
                    "label": concept_id,
                    "value": np.sum(adjacency_matrix[i, :]) if isinstance(adjacency_matrix, np.ndarray) else 0
                })
        
        if isinstance(adjacency_matrix, np.ndarray):
            rows, cols = np.where(adjacency_matrix > 0.1)  # 阈值
            for row, col in zip(rows, cols):
                if row < col:  # 避免重复边
                    edges.append({
                        "from": int(row),
                        "to": int(col),
                        "value": float(adjacency_matrix[row, col])
                    })
        
        return {
            "title": f"{family_type} 承载关系网络图",
            "nodes": nodes,
            "edges": edges,
            "layout": {
                "showlegend": False,
                "margin": {"l": 20, "r": 20, "t": 40, "b": 20},
                "xaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
                "yaxis": {"showgrid": False, "zeroline": False, "showticklabels": False}
            }
        }
    
    def prepare_intervention_result_data(
        self,
        param_id: str,
        intervention_type: str = "ablation"
    ) -> Dict[str, Any]:
        """
        准备干预结果数据
        
        Args:
            param_id: 参数ID
            intervention_type: 干预类型
            
        Returns:
            干预结果数据
        """
        # 获取干预结果
        result = self.data_api.get_intervention_result(
            param_id=param_id,
            intervention_type=intervention_type
        )
        
        if "error" in result:
            return {"error": result["error"]}
        
        intervention_data = result["results"]
        
        # 提取干预前后的性能对比
        performance_before = intervention_data.get("performance_before", None)
        performance_after = intervention_data.get("performance_after", None)
        
        if performance_before is None or performance_after is None:
            return {"error": "Missing performance data"}
        
        # 准备对比柱状图
        performance_data = [
            {
                "x": ["干预前", "干预后"],
                "y": [performance_before, performance_after],
                "type": "bar",
                "name": "性能"
            }
        ]
        
        return {
            "title": f"参数 {param_id} 的 {intervention_type} 干预效果",
            "data": performance_data,
            "layout": {
                "xaxis": {"title": "状态"},
                "yaxis": {"title": "性能"},
                "hovermode": "closest"
            }
        }


# 单例模式
_viz_helper_instance = None

def get_visualization_helper(data_api=None) -> VisualizationHelper:
    """获取可视化助手单例"""
    global _viz_helper_instance
    if _viz_helper_instance is None:
        _viz_helper_instance = VisualizationHelper(data_api)
    return _viz_helper_instance
