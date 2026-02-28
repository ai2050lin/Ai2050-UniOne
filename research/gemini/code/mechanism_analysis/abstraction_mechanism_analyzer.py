"""
抽象机制分析器

目标: 理解高维抽象特征如何形成

核心问题:
1. 抽象特征是什么？
2. 抽象层级如何编码？
3. 抽象与具体的关系？

方法:
1. 概念层级探测
2. 抽象方向向量提取
3. 层间抽象程度量化
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class AbstractionConfig:
    """抽象机制分析配置"""
    # 概念层级定义
    concept_hierarchy: Dict[str, List[str]] = field(default_factory=lambda: {
        # 具体到抽象的层级
        "animals": [
            ["cat", "dog", "bird"],           # Level 0: 具体
            ["mammal", "bird", "fish"],       # Level 1: 基本类别
            ["animal", "plant"],              # Level 2: 上位类别
            ["living", "non-living"]          # Level 3: 最抽象
        ],
        "objects": [
            ["chair", "table", "bed"],
            ["furniture", "appliance"],
            ["artifact", "natural object"],
            ["physical", "abstract"]
        ],
        "actions": [
            ["run", "walk", "jump"],
            ["move", "stay", "act"],
            ["action", "state"],
            ["event", "property"]
        ]
    })
    
    # 分析层
    target_layers: List[int] = field(default_factory=lambda: [0, 3, 6, 9, 11])
    
    # 输出目录
    output_dir: str = "results/abstraction_mechanism"


class AbstractionMechanismAnalyzer:
    """
    抽象机制分析器
    
    核心洞察:
    - 抽象不是简单的"模糊"
    - 抽象是信息压缩的层级结构
    - 抽象特征具有方向性
    
    关键问题:
    1. 抽象程度如何量化？
    2. 抽象方向向量是什么？
    3. 层级如何编码？
    """
    
    def __init__(self, model: nn.Module, config: AbstractionConfig):
        self.model = model
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储分析结果
        self.abstraction_vectors: Dict[int, torch.Tensor] = {}
        self.hierarchy_scores: Dict[str, Dict[int, float]] = {}
    
    def analyze_concept_hierarchy(
        self,
        domain: str = "animals"
    ) -> Dict[str, Any]:
        """
        分析概念层级
        
        问题: 不同抽象层级的概念在激活空间中如何分布？
        
        方法:
        1. 获取各层级概念的激活
        2. 计算层级间的距离
        3. 发现抽象方向
        """
        if domain not in self.config.concept_hierarchy:
            return {"error": f"Unknown domain: {domain}"}
        
        hierarchy = self.config.concept_hierarchy[domain]
        results = {
            "domain": domain,
            "levels": {},
            "abstraction_trajectory": None,
            "key_findings": []
        }
        
        # 1. 获取每个层级的激活
        level_activations = []
        for level_idx, concepts in enumerate(hierarchy):
            acts = self._get_concept_activations(concepts)
            level_activations.append(acts)
            
            # 计算层内特征
            results["levels"][level_idx] = {
                "concepts": concepts,
                "mean_activation": acts.mean().item() if acts is not None else 0,
                "spread": acts.std().item() if acts is not None else 0
            }
        
        # 2. 分析层级间关系
        if len(level_activations) >= 2:
            # 计算抽象方向向量
            concrete_act = level_activations[0]  # 最具体
            abstract_act = level_activations[-1]  # 最抽象
            
            if concrete_act is not None and abstract_act is not None:
                # 抽象方向 = 抽象中心 - 具体中心
                abstraction_direction = abstract_act.mean(0) - concrete_act.mean(0)
                
                results["abstraction_trajectory"] = {
                    "direction_norm": abstraction_direction.norm().item(),
                    "direction_dim": abstraction_direction.shape[0]
                }
                
                # 保存抽象方向向量
                self.abstraction_vectors[domain] = abstraction_direction
        
        # 3. 关键发现
        results["key_findings"] = self._extract_findings(results)
        
        return results
    
    def _get_concept_activations(
        self,
        concepts: List[str]
    ) -> Optional[torch.Tensor]:
        """
        获取概念的激活
        
        使用模型获取概念词的激活表示
        """
        activations = []
        
        for concept in concepts:
            try:
                # 获取词嵌入
                if hasattr(self.model, 'to_tokens'):
                    tokens = self.model.to_tokens(concept)
                    with torch.no_grad():
                        # 获取最后一层的激活
                        _, cache = self.model.run_with_cache(tokens)
                        # 获取残差流激活
                        last_layer_act = cache["resid_post", -1]
                        activations.append(last_layer_act.mean(dim=1))
            except Exception as e:
                print(f"Error getting activation for {concept}: {e}")
        
        if activations:
            return torch.cat(activations, dim=0)
        return None
    
    def extract_abstraction_direction(
        self,
        concrete_examples: List[str],
        abstract_examples: List[str],
        layer_idx: int = 11
    ) -> torch.Tensor:
        """
        提取抽象方向向量
        
        问题: 从具体到抽象的"方向"是什么？
        
        方法:
        1. 获取具体概念的激活中心
        2. 获取抽象概念的激活中心
        3. 方向 = 抽象中心 - 具体中心
        
        这个方向向量可以:
        - 判断任意概念的抽象程度
        - 干预使概念更抽象/更具体
        """
        concrete_act = self._get_concept_activations(concrete_examples)
        abstract_act = self._get_concept_activations(abstract_examples)
        
        if concrete_act is None or abstract_act is None:
            return torch.zeros(768)  # 默认维度
        
        # 计算中心
        concrete_center = concrete_act.mean(dim=0)
        abstract_center = abstract_act.mean(dim=0)
        
        # 抽象方向
        abstraction_direction = abstract_center - concrete_center
        
        # 归一化
        abstraction_direction = abstraction_direction / (abstraction_direction.norm() + 1e-8)
        
        return abstraction_direction
    
    def measure_abstraction_degree(
        self,
        concept: str,
        abstraction_direction: torch.Tensor
    ) -> float:
        """
        测量概念的抽象程度
        
        方法:
        抽象程度 = 激活向量在抽象方向上的投影
        
        优点:
        - 可以量化任意概念的抽象程度
        - 不依赖人工标注
        """
        concept_act = self._get_concept_activations([concept])
        
        if concept_act is None:
            return 0.0
        
        # 投影到抽象方向
        projection = (concept_act.mean(dim=0) @ abstraction_direction).item()
        
        return projection
    
    def analyze_layer_abstraction(
        self,
        test_concepts: List[str],
        abstraction_direction: torch.Tensor
    ) -> Dict[int, float]:
        """
        分析各层的抽象程度
        
        关键问题: 哪一层最抽象？
        
        预期:
        - 深层比浅层更抽象
        - 抽象程度单调递增
        """
        layer_abstraction = {}
        
        for layer_idx in self.config.target_layers:
            # 获取该层的激活
            layer_abstractions = []
            for concept in test_concepts:
                try:
                    if hasattr(self.model, 'to_tokens'):
                        tokens = self.model.to_tokens(concept)
                        with torch.no_grad():
                            _, cache = self.model.run_with_cache(tokens)
                            layer_act = cache["resid_post", layer_idx]
                            
                            # 计算投影
                            proj = (layer_act.mean(dim=1) @ abstraction_direction).item()
                            layer_abstractions.append(proj)
                except:
                    pass
            
            if layer_abstractions:
                layer_abstraction[layer_idx] = np.mean(layer_abstractions)
        
        return layer_abstraction
    
    def _extract_findings(self, results: Dict) -> List[str]:
        """提取关键发现"""
        findings = []
        
        # 分析层级间的距离变化
        levels = results.get("levels", {})
        if len(levels) >= 2:
            spreads = [levels[i].get("spread", 0) for i in sorted(levels.keys())]
            if spreads[0] < spreads[-1]:
                findings.append("抽象概念的激活分散度更大（覆盖范围更广）")
            else:
                findings.append("具体概念的激活分散度更大")
        
        # 分析抽象方向
        trajectory = results.get("abstraction_trajectory", {})
        if trajectory:
            norm = trajectory.get("direction_norm", 0)
            if norm > 1.0:
                findings.append(f"抽象方向显著，模长={norm:.2f}")
        
        return findings
    
    def save_results(self, results: Dict):
        """保存分析结果"""
        output_path = self.output_dir / "abstraction_analysis.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"Results saved to {output_path}")


class PrecisionMechanismAnalyzer:
    """
    精确机制分析器
    
    目标: 理解低维精确预测如何实现
    
    核心问题:
    1. 精确预测需要什么条件？
    2. 注意力如何贡献精确性？
    3. 精确vs模糊的激活差异？
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        
    def analyze_precision_vs_fuzzy(
        self,
        precise_input: str,
        fuzzy_input: str,
        target_token: str
    ) -> Dict[str, Any]:
        """
        对比精确预测vs模糊预测
        
        问题: 为什么有时能精确预测，有时不能？
        
        方法:
        1. 选择精确预测的例子（如 "1+1=2"）
        2. 选择模糊预测的例子（如 "明天会..."）
        3. 对比激活模式差异
        """
        results = {
            "precise": {"input": precise_input, "target": target_token},
            "fuzzy": {"input": fuzzy_input, "target": target_token},
            "differences": {}
        }
        
        # 获取精确预测的激活
        try:
            precise_tokens = self.model.to_tokens(precise_input)
            with torch.no_grad():
                _, precise_cache = self.model.run_with_cache(precise_tokens)
                precise_act = precise_cache["resid_post", -1]
        except:
            precise_act = None
        
        # 获取模糊预测的激活
        try:
            fuzzy_tokens = self.model.to_tokens(fuzzy_input)
            with torch.no_grad():
                _, fuzzy_cache = self.model.run_with_cache(fuzzy_tokens)
                fuzzy_act = fuzzy_cache["resid_post", -1]
        except:
            fuzzy_act = None
        
        # 对比分析
        if precise_act is not None and fuzzy_act is not None:
            # 激活范数差异
            results["differences"]["activation_norm"] = {
                "precise": precise_act.norm().item(),
                "fuzzy": fuzzy_act.norm().item()
            }
            
            # 稀疏度差异
            precise_sparsity = (precise_act.abs() < 0.01).float().mean().item()
            fuzzy_sparsity = (fuzzy_act.abs() < 0.01).float().mean().item()
            results["differences"]["sparsity"] = {
                "precise": precise_sparsity,
                "fuzzy": fuzzy_sparsity
            }
        
        return results
    
    def identify_precision_conditions(
        self,
        test_cases: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        识别精确预测的条件
        
        问题: 什么情况下能精确预测？
        
        假设:
        1. 有明确的模式（如数学、语法）
        2. 上下文信息充分
        3. 注意力聚焦关键位置
        """
        results = {
            "precise_conditions": [],
            "fuzzy_conditions": []
        }
        
        for case in test_cases:
            analysis = self._analyze_single_case(case)
            if case.get("is_precise", False):
                results["precise_conditions"].append(analysis)
            else:
                results["fuzzy_conditions"].append(analysis)
        
        # 总结条件
        results["summary"] = self._summarize_conditions(results)
        
        return results
    
    def _analyze_single_case(self, case: Dict[str, str]) -> Dict[str, Any]:
        """分析单个测试用例"""
        return {
            "input": case.get("input", ""),
            "context_length": len(case.get("input", "").split()),
            "has_pattern": self._detect_pattern(case.get("input", ""))
        }
    
    def _detect_pattern(self, text: str) -> bool:
        """检测是否有明确模式"""
        # 简单的模式检测
        patterns = [
            lambda t: any(c.isdigit() for c in t),  # 包含数字
            lambda t: "?" in t,  # 问句
            lambda t: any(kw in t.lower() for kw in ["what", "where", "when"])  # 疑问词
        ]
        return any(p(text) for p in patterns)
    
    def _summarize_conditions(self, results: Dict) -> List[str]:
        """总结精确预测的条件"""
        conditions = []
        
        precise_cases = results.get("precise_conditions", [])
        fuzzy_cases = results.get("fuzzy_conditions", [])
        
        if precise_cases:
            avg_context_precise = np.mean([c["context_length"] for c in precise_cases])
            conditions.append(f"精确预测平均上下文长度: {avg_context_precise:.1f}")
        
        if fuzzy_cases:
            avg_context_fuzzy = np.mean([c["context_length"] for c in fuzzy_cases])
            conditions.append(f"模糊预测平均上下文长度: {avg_context_fuzzy:.1f}")
        
        return conditions


def run_abstraction_analysis():
    """
    运行抽象机制分析
    
    这是理解"高维抽象如何形成"的第一步
    """
    print("=" * 60)
    print("抽象机制分析")
    print("=" * 60)
    print()
    print("核心问题:")
    print("1. 抽象特征是什么？")
    print("2. 抽象层级如何编码？")
    print("3. 哪一层最抽象？")
    print()
    print("方法:")
    print("1. 定义概念层级（具体→抽象）")
    print("2. 提取抽象方向向量")
    print("3. 测量各层的抽象程度")
    print()
    
    # 示例配置
    config = AbstractionConfig()
    
    print("概念层级定义:")
    for domain, levels in config.concept_hierarchy.items():
        print(f"\n{domain}:")
        for i, level in enumerate(levels):
            print(f"  Level {i}: {level}")
    
    print()
    print("分析框架已创建，需要接入模型运行")
    print()
    print("预期发现:")
    print("  - 深层激活比浅层更抽象")
    print("  - 抽象方向向量可以量化抽象程度")
    print("  - 抽象概念覆盖更大的激活空间")


if __name__ == "__main__":
    run_abstraction_analysis()
