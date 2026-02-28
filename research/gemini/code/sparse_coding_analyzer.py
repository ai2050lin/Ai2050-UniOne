"""
稀疏编码分析器
==============

分析DNN中稀疏编码的性质

核心分析:
1. 稀疏度测量: L0, L1, Gini系数
2. 编码效率: 信息论分析
3. 特征选择性: 每个神经元对特征的响应
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SparseCodingAnalyzer:
    """
    稀疏编码分析器
    """
    
    def __init__(self):
        pass
    
    def analyze(
        self,
        activations: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        完整的稀疏编码分析
        
        Args:
            activations: [N, d] 激活矩阵
            features: [k, d] 特征矩阵 (SAE学到的)
        
        Returns:
            分析结果
        """
        results = {}
        
        # 1. 激活稀疏度
        results["activation_sparsity"] = self._analyze_sparsity(activations)
        
        # 2. 如果有特征矩阵，分析特征稀疏度
        if features is not None:
            results["feature_sparsity"] = self._analyze_sparsity(features)
            
            # 3. 特征选择性
            results["feature_selectivity"] = self._analyze_selectivity(activations, features)
        
        return results
    
    def _analyze_sparsity(
        self,
        matrix: torch.Tensor,
        threshold: float = 0.01
    ) -> Dict[str, float]:
        """
        分析矩阵的稀疏度
        
        指标:
        - L0稀疏度: 非零元素比例
        - L1稀疏度: 平均绝对值
        - Gini系数: 不均匀程度
        - 熵: 分布的随机性
        """
        matrix_np = matrix.cpu().numpy()
        
        # L0稀疏度
        l0 = np.mean(np.abs(matrix_np) > threshold)
        
        # L1稀疏度
        l1 = np.mean(np.abs(matrix_np))
        
        # Gini系数
        gini = self._compute_gini(np.abs(matrix_np).flatten())
        
        # 熵 (归一化)
        flat = np.abs(matrix_np).flatten() + 1e-10
        flat = flat / flat.sum()
        entropy = -np.sum(flat * np.log(flat + 1e-10))
        max_entropy = np.log(len(flat))
        entropy_normalized = entropy / max_entropy
        
        return {
            "l0_sparsity": float(l0),
            "l1_sparsity": float(l1),
            "gini_coefficient": float(gini),
            "entropy_normalized": float(entropy_normalized)
        }
    
    def _compute_gini(self, x: np.ndarray) -> float:
        """计算Gini系数"""
        x = np.sort(x)
        n = len(x)
        if n == 0 or np.sum(x) == 0:
            return 0.0
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))
    
    def _analyze_selectivity(
        self,
        activations: torch.Tensor,
        features: torch.Tensor
    ) -> Dict[str, Any]:
        """
        分析特征选择性
        
        问题: 每个特征是否只对特定输入响应？
        """
        # 确保在同一设备上
        device = activations.device
        features = features.to(device)
        
        # 计算每个特征对每个激活的响应
        # features: [k, d], activations: [N, d]
        # responses: [k, N]
        responses = torch.matmul(features, activations.T)
        
        # 对于每个特征，找出它最响应的输入
        top_responses = torch.topk(responses, k=min(5, responses.shape[1]), dim=1)
        
        # 计算选择性指数
        # 高选择性 = 少数输入高响应，其他低响应
        selectivity_scores = []
        
        for i in range(responses.shape[0]):
            feat_responses = responses[i].cpu().numpy()
            
            # 选择性 = max / (mean + std)
            max_resp = np.max(feat_responses)
            mean_resp = np.mean(feat_responses)
            std_resp = np.std(feat_responses)
            
            selectivity = max_resp / (mean_resp + std_resp + 1e-10)
            selectivity_scores.append(selectivity)
        
        return {
            "mean_selectivity": float(np.mean(selectivity_scores)),
            "std_selectivity": float(np.std(selectivity_scores)),
            "high_selectivity_ratio": float(np.mean(np.array(selectivity_scores) > 2.0))
        }
