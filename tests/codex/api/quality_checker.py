#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据质量检查器
自动评估数据完整性、一致性、可复现性
"""
from typing import Dict, Any, List, Tuple
import numpy as np
from pathlib import Path


class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self, base_path: str = "d:/develop/TransformerLens-main/tests/codex/data"):
        """
        初始化检查器
        
        Args:
            base_path: 数据基础路径
        """
        self.base_path = Path(base_path)
        self.results_path = self.base_path / "results"
        self.metadata_path = self.base_path / "metadata"
        self.quality_reports_path = self.base_path / "quality_reports"
        
        # 确保目录存在
        self.quality_reports_path.mkdir(parents=True, exist_ok=True)
    
    def check_completeness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查数据完整性
        
        Args:
            data: 待检查的数据
            
        Returns:
            完整性检查结果
        """
        issues = []
        warnings = []
        
        # 检查必要字段是否存在
        required_fields = ["activations", "metadata"]
        for field in required_fields:
            if field not in data:
                issues.append(f"缺少必要字段: {field}")
        
        # 检查激活数据是否为空
        if "activations" in data:
            activations = data["activations"]
            if activations is None or (isinstance(activations, np.ndarray) and activations.size == 0):
                warnings.append("激活数据为空")
        
        # 计算完整性分数
        score = 1.0 - len(issues) * 0.3 - len(warnings) * 0.1
        score = max(0.0, min(1.0, score))
        
        return {
            "metric": "completeness",
            "score": score,
            "issues": issues,
            "warnings": warnings,
            "status": "passed" if len(issues) == 0 else "failed"
        }
    
    def check_consistency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查数据一致性
        
        Args:
            data: 待检查的数据
            
        Returns:
            一致性检查结果
        """
        issues = []
        warnings = []
        
        # 检查维度一致性
        if "activations" in data and "metadata" in data:
            activations = data["activations"]
            metadata = data["metadata"]
            
            if isinstance(activations, np.ndarray):
                # 检查维度是否匹配元数据
                if "layer" in metadata and activations.ndim == 2:
                    expected_layers = metadata.get("expected_layers", activations.shape[0])
                    if activations.shape[0] != expected_layers:
                        warnings.append(f"激活层数({activations.shape[0]})与预期({expected_layers})不匹配")
        
        # 检查数值范围
        if "activations" in data:
            activations = data["activations"]
            if isinstance(activations, np.ndarray):
                if activations.max() > 1000 or activations.min() < -1000:
                    warnings.append("激活值超出正常范围")
        
        # 计算一致性分数
        score = 1.0 - len(issues) * 0.3 - len(warnings) * 0.1
        score = max(0.0, min(1.0, score))
        
        return {
            "metric": "consistency",
            "score": score,
            "issues": issues,
            "warnings": warnings,
            "status": "passed" if len(issues) == 0 else "failed"
        }
    
    def check_reproducibility(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查数据可复现性
        
        Args:
            data: 待检查的数据
            
        Returns:
            可复现性检查结果
        """
        issues = []
        warnings = []
        
        # 检查是否有随机种子信息
        if "metadata" in data:
            metadata = data["metadata"]
            if "random_seed" not in metadata:
                warnings.append("缺少随机种子信息，可能影响可复现性")
        
        # 检查是否有模型版本信息
        if "metadata" in data:
            metadata = data["metadata"]
            if "model_version" not in metadata:
                warnings.append("缺少模型版本信息")
        
        # 计算可复现性分数
        score = 1.0 - len(issues) * 0.3 - len(warnings) * 0.15
        score = max(0.0, min(1.0, score))
        
        return {
            "metric": "reproducibility",
            "score": score,
            "issues": issues,
            "warnings": warnings,
            "status": "passed" if len(issues) == 0 else "failed"
        }
    
    def check_all_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查所有质量指标
        
        Args:
            data: 待检查的数据
            
        Returns:
            综合质量报告
        """
        completeness = self.check_completeness(data)
        consistency = self.check_consistency(data)
        reproducibility = self.check_reproducibility(data)
        
        # 计算综合分数
        overall_score = (
            completeness["score"] * 0.4 +
            consistency["score"] * 0.4 +
            reproducibility["score"] * 0.2
        )
        
        return {
            "overall_score": overall_score,
            "metrics": {
                "completeness": completeness,
                "consistency": consistency,
                "reproducibility": reproducibility
            },
            "status": "passed" if overall_score >= 0.7 else "needs_improvement",
            "total_issues": len(completeness["issues"]) + len(consistency["issues"]) + len(reproducibility["issues"]),
            "total_warnings": len(completeness["warnings"]) + len(consistency["warnings"]) + len(reproducibility["warnings"])
        }
    
    def generate_quality_report(self, stage_id: int, data: Dict[str, Any]) -> str:
        """
        生成质量报告
        
        Args:
            stage_id: 测试stage ID
            data: 待检查的数据
            
        Returns:
            报告文件路径
        """
        quality_result = self.check_all_metrics(data)
        
        # 保存报告
        report_filename = f"stage{stage_id}_quality_report.json"
        report_path = self.quality_reports_path / report_filename
        
        import json
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(quality_result, f, ensure_ascii=False, indent=2)
        
        return str(report_path)


# 单例模式
_checker_instance = None

def get_checker() -> DataQualityChecker:
    """获取检查器单例"""
    global _checker_instance
    if _checker_instance is None:
        _checker_instance = DataQualityChecker()
    return _checker_instance
