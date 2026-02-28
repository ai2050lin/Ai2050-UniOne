"""
特征编码分析框架
================

目标: 从DNN中提取特征编码结构，还原大脑编码机制

核心模块:
- FeatureExtractor: 特征提取
- SparseCodingAnalyzer: 稀疏编码分析
- FourPropertiesEvaluator: 四特性评估
- EmergenceTracker: 涌现过程追踪
- BrainMechanismInference: 大脑机制推断

Author: AGI Research Team
Date: 2026-02-20
"""

from .feature_extractor import FeatureExtractor
from .sparse_coding_analyzer import SparseCodingAnalyzer
from .four_properties_evaluator import FourPropertiesEvaluator
from .emergence_tracker import EmergenceTracker
from .brain_mechanism_inference import BrainMechanismInference

__all__ = [
    'FeatureExtractor',
    'SparseCodingAnalyzer',
    'FourPropertiesEvaluator',
    'EmergenceTracker',
    'BrainMechanismInference',
]
