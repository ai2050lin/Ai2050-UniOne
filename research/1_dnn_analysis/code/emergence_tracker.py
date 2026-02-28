"""
涌现过程追踪器
==============

追踪特征在训练过程中的涌现

核心功能:
1. 训练过程监控: 记录关键指标变化
2. 关键转变点识别: Grokking时刻
3. 因果分析: 干预实验
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class TrackingConfig:
    """追踪配置"""
    checkpoint_dir: str = "checkpoints"
    metrics: List[str] = None
    save_interval: int = 100
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                "loss", "accuracy", "intrinsic_dimension",
                "sparsity", "curvature", "betti_numbers"
            ]


class EmergenceTracker:
    """
    涌现过程追踪器
    
    监控训练过程中特征的涌现
    """
    
    def __init__(self, config: Optional[TrackingConfig] = None):
        self.config = config or TrackingConfig()
        self.trajectory = []
    
    def record_checkpoint(
        self,
        step: int,
        model_state: Optional[Dict] = None,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        记录检查点
        
        Args:
            step: 训练步数
            model_state: 模型状态
            metrics: 指标字典
        """
        checkpoint = {
            "step": step,
            "metrics": metrics or {}
        }
        
        self.trajectory.append(checkpoint)
        
        logging.info(f"Recorded checkpoint at step {step}")
    
    def identify_critical_transitions(
        self,
        metric: str = "loss",
        threshold: float = 0.1
    ) -> List[int]:
        """
        识别关键转变点
        
        Args:
            metric: 要分析的指标
            threshold: 变化阈值
        
        Returns:
            关键转变点的步数列表
        """
        if len(self.trajectory) < 3:
            return []
        
        values = [t["metrics"].get(metric, 0) for t in self.trajectory]
        
        # 计算一阶差分
        diffs = np.diff(values)
        
        # 找到突变点
        critical_points = []
        
        for i, diff in enumerate(diffs):
            if abs(diff) > threshold:
                critical_points.append(self.trajectory[i+1]["step"])
        
        logging.info(f"Found {len(critical_points)} critical transitions")
        return critical_points
    
    def analyze_emergence_pattern(
        self,
        metric: str = "intrinsic_dimension"
    ) -> Dict[str, Any]:
        """
        分析涌现模式
        
        识别: 压缩 → 重组 → 结晶
        """
        if len(self.trajectory) < 5:
            return {"status": "insufficient_data"}
        
        values = [t["metrics"].get(metric, 0) for t in self.trajectory]
        steps = [t["step"] for t in self.trajectory]
        
        # 识别趋势
        # 1. 压缩阶段: ID下降
        # 2. 重组阶段: ID反弹
        # 3. 结晶阶段: ID稳定
        
        pattern = {
            "metric": metric,
            "phases": []
        }
        
        # 简单趋势检测
        values_arr = np.array(values)
        
        # 找最小值 (压缩结束)
        min_idx = np.argmin(values_arr)
        min_step = steps[min_idx]
        
        # 找最大值 (重组峰值)
        after_min = values_arr[min_idx:]
        if len(after_min) > 0:
            max_after_min_idx = np.argmax(after_min) + min_idx
            max_step = steps[max_after_min_idx]
        else:
            max_step = min_step
        
        pattern["phases"] = [
            {"name": "compression", "end_step": min_step, "value": float(values[min_idx])},
            {"name": "reorganization", "peak_step": max_step, "value": float(values[max_after_min_idx] if len(after_min) > 0 else 0)}
        ]
        
        return pattern
    
    def save_trajectory(self, path: str):
        """保存轨迹"""
        with open(path, 'w') as f:
            json.dump(self.trajectory, f, indent=2)
        logging.info(f"Saved trajectory to {path}")
    
    def load_trajectory(self, path: str):
        """加载轨迹"""
        with open(path, 'r') as f:
            self.trajectory = json.load(f)
        logging.info(f"Loaded trajectory from {path}")
