"""
特征涌现追踪器 - 解决"只有统计描述"问题的核心工具

目标: 理解神经网络如何从信号流中提取特征并形成稳定编码

方法:
1. 训练过程追踪 - 看特征如何涌现
2. 因果干预实验 - 改变某特征看效果
3. 对比实验 - 相同输入不同输出
4. 反向工程 - 从输出反推机制
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
from collections import defaultdict


@dataclass
class EmergenceConfig:
    """特征涌现追踪配置"""
    model_type: str = "gpt2"  # 模型类型
    model_size: str = "small"  # small/medium/large
    save_interval: int = 100  # 每100步保存一次
    total_steps: int = 10000  # 总训练步数
    eval_interval: int = 500  # 评估间隔
    
    # 分析配置
    track_layers: List[int] = field(default_factory=lambda: [0, 3, 6, 9, 11])
    num_concepts: int = 50  # 追踪的概念数量
    concept_types: List[str] = field(default_factory=lambda: ["concrete", "abstract", "syntactic"])
    
    # 输出配置
    output_dir: str = "results/feature_emergence"


class FeatureEmergenceTracker:
    """
    特征涌现追踪器
    
    核心问题: 神经网络如何从信号流中提取特征？
    
    方法论:
    1. 训练开始时: 特征不存在
    2. 训练过程中: 追踪特征何时出现、如何演化
    3. 训练结束: 特征稳定
    
    关键指标:
    - 涌现时间: 特征何时首次出现
    - 涌现速度: 特征形成需要多少步
    - 稳定性: 特征是否稳定
    - 依赖关系: 特征之间的依赖关系
    """
    
    def __init__(self, config: EmergenceConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储训练过程中的数据
        self.emergence_history: Dict[int, Dict[str, Any]] = defaultdict(dict)
        self.feature_trajectory: Dict[str, List[Dict]] = defaultdict(list)
        
        # 概念测试集
        self.concept_probes = self._create_concept_probes()
        
    def _create_concept_probes(self) -> Dict[str, Dict]:
        """
        创建概念探针 - 用于追踪特定概念的特征
        
        关键洞察: 
        - 如果模型理解"猫"，它应该有专门的"猫"特征
        - 通过对比"猫"vs"狗"的激活，找到猫特征
        """
        probes = {
            # 具体概念
            "concrete": {
                "animals": ["cat", "dog", "bird", "fish", "lion"],
                "objects": ["table", "chair", "car", "house", "book"],
                "colors": ["red", "blue", "green", "yellow", "black"],
            },
            # 抽象概念
            "abstract": {
                "emotions": ["happy", "sad", "angry", "fear", "love"],
                "qualities": ["good", "bad", "big", "small", "fast"],
                "relations": ["cause", "effect", "before", "after", "if"],
            },
            # 语法概念
            "syntactic": {
                "pos_tags": ["noun", "verb", "adj", "adv", "prep"],
                "number": ["singular", "plural"],
                "tense": ["past", "present", "future"],
            }
        }
        return probes
    
    def track_training_step(
        self,
        model: nn.Module,
        step: int,
        batch_data: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, Any]:
        """
        在训练过程中追踪特征涌现
        
        这是核心方法 - 回答"特征如何涌现"
        
        方法:
        1. 前向传播获取激活
        2. 检测特征是否出现
        3. 记录特征演化轨迹
        """
        results = {
            "step": step,
            "layers": {},
            "emergence_events": []
        }
        
        # 1. 获取各层激活
        with torch.no_grad():
            activations = self._get_layer_activations(model, batch_data)
        
        # 2. 分析每个追踪层
        for layer_idx in self.config.track_layers:
            if layer_idx not in activations:
                continue
                
            layer_acts = activations[layer_idx]
            
            # 检测特征涌现
            emergence = self._detect_feature_emergence(layer_acts, layer_idx, step)
            results["layers"][layer_idx] = emergence
            
            # 记录涌现事件
            if emergence["new_features_detected"]:
                results["emergence_events"].extend(emergence["new_features_detected"])
        
        # 3. 存储历史
        self.emergence_history[step] = results
        
        return results
    
    def _get_layer_activations(
        self, 
        model: nn.Module, 
        batch_data: Dict[str, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        """
        获取各层激活
        
        使用hook机制提取中间层激活
        """
        activations = {}
        hooks = []
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                activations[layer_idx] = output.detach()
            return hook
        
        # 注册hooks
        for layer_idx in self.config.track_layers:
            # 这里需要根据具体模型结构调整
            # 对于GPT-2，blocks[layer_idx]
            try:
                layer = model.blocks[layer_idx]
                hooks.append(layer.register_forward_hook(make_hook(layer_idx)))
            except:
                pass
        
        # 前向传播
        try:
            _ = model(batch_data["input_ids"])
        except Exception as e:
            print(f"Forward pass error: {e}")
        
        # 移除hooks
        for hook in hooks:
            hook.remove()
        
        return activations
    
    def _detect_feature_emergence(
        self,
        activations: torch.Tensor,
        layer_idx: int,
        step: int
    ) -> Dict[str, Any]:
        """
        检测特征涌现 - 核心算法
        
        方法:
        1. 特征分离度: 不同概念的激活是否分离
        2. 特征一致性: 相同概念的激活是否聚集
        3. 特征稳定性: 特征是否持续存在
        
        关键指标:
        - 涌现阈值: 分离度 > 0.5 认为特征出现
        - 稳定阈值: 连续3次出现认为特征稳定
        """
        results = {
            "layer": layer_idx,
            "step": step,
            "feature_separation": {},
            "feature_consistency": {},
            "new_features_detected": [],
            "stable_features": []
        }
        
        # 计算激活的统计特性
        mean_act = activations.mean(dim=0)
        std_act = activations.std(dim=0)
        
        # 检测稀疏激活模式
        sparsity = (activations.abs() < 0.01).float().mean()
        
        # 检测特征簇
        # 使用简单的聚类检测
        if activations.shape[0] > 10:
            try:
                from sklearn.cluster import KMeans
                n_clusters = min(10, activations.shape[0] // 5)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
                cluster_labels = kmeans.fit_predict(activations.cpu().numpy())
                
                # 计算簇内/簇间距离比
                intra_dist = 0
                inter_dist = 0
                for i in range(n_clusters):
                    cluster_points = activations[cluster_labels == i]
                    if len(cluster_points) > 1:
                        intra_dist += cluster_points.std().item()
                intra_dist /= n_clusters
                inter_dist = activations.std().item()
                
                separation_ratio = inter_dist / (intra_dist + 1e-6)
                results["feature_separation"]["cluster_ratio"] = separation_ratio
                
                # 如果分离度超过阈值，认为有新特征涌现
                if separation_ratio > 2.0:
                    results["new_features_detected"].append({
                        "layer": layer_idx,
                        "step": step,
                        "type": "cluster_emergence",
                        "separation": separation_ratio
                    })
            except Exception as e:
                pass
        
        results["sparsity"] = sparsity.item()
        results["activation_dim"] = activations.shape[-1]
        
        return results
    
    def analyze_emergence_pattern(self) -> Dict[str, Any]:
        """
        分析特征涌现模式
        
        回答关键问题:
        1. 哪些特征先出现？哪些后出现？
        2. 具体概念vs抽象概念的出现顺序？
        3. 不同层的特征涌现顺序？
        """
        if not self.emergence_history:
            return {"error": "No training history"}
        
        analysis = {
            "emergence_timeline": {},
            "layer_emergence_order": [],
            "concept_emergence_order": [],
            "key_findings": []
        }
        
        # 1. 分析涌现时间线
        for step, data in sorted(self.emergence_history.items()):
            for event in data.get("emergence_events", []):
                layer = event["layer"]
                event_type = event["type"]
                
                key = f"layer_{layer}_{event_type}"
                if key not in analysis["emergence_timeline"]:
                    analysis["emergence_timeline"][key] = {
                        "first_appearance": step,
                        "layer": layer,
                        "type": event_type
                    }
        
        # 2. 分析层涌现顺序
        layer_emergence = {}
        for key, data in analysis["emergence_timeline"].items():
            layer = data["layer"]
            if layer not in layer_emergence:
                layer_emergence[layer] = data["first_appearance"]
        
        analysis["layer_emergence_order"] = sorted(
            layer_emergence.items(), 
            key=lambda x: x[1]
        )
        
        # 3. 关键发现
        if analysis["layer_emergence_order"]:
            first_layer = analysis["layer_emergence_order"][0][0]
            last_layer = analysis["layer_emergence_order"][-1][0]
            
            analysis["key_findings"].append(
                f"特征最先在Layer {first_layer}涌现，最后在Layer {last_layer}稳定"
            )
        
        return analysis
    
    def save_results(self):
        """保存追踪结果"""
        results = {
            "config": self.config.__dict__,
            "emergence_history": dict(self.emergence_history),
            "analysis": self.analyze_emergence_pattern()
        }
        
        output_path = self.output_dir / "emergence_tracking_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Results saved to {output_path}")


class CausalInterventionAnalyzer:
    """
    因果干预分析器
    
    目标: 理解特征之间的因果关系
    
    方法:
    1. 干预: 改变某个特征的激活
    2. 观察: 其他特征如何变化
    3. 推断: 因果关系
    
    示例问题:
    - "猫"特征的激活是否导致"动物"特征的激活？
    - 破坏"性别"特征，是否影响"代词"预测？
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.intervention_cache = {}
    
    def intervene_feature(
        self,
        activations: torch.Tensor,
        feature_direction: torch.Tensor,
        intervention_type: str = "suppress",  # suppress/enhance/replace
        strength: float = 1.0
    ) -> torch.Tensor:
        """
        干预特征
        
        Args:
            activations: 原始激活
            feature_direction: 要干预的特征方向
            intervention_type: 干预类型
                - suppress: 抑制特征
                - enhance: 增强特征
                - replace: 替换特征
            strength: 干预强度
        """
        if intervention_type == "suppress":
            # 投影到特征方向的补空间
            projection = (activations @ feature_direction) / (feature_direction @ feature_direction + 1e-8)
            intervened = activations - projection * feature_direction * strength
            
        elif intervention_type == "enhance":
            # 增强特征方向
            projection = (activations @ feature_direction) / (feature_direction @ feature_direction + 1e-8)
            intervened = activations + projection * feature_direction * strength
            
        elif intervention_type == "replace":
            # 替换为特征方向
            intervened = feature_direction * strength * torch.ones_like(activations)
            
        else:
            intervened = activations
        
        return intervened
    
    def analyze_causal_effect(
        self,
        input_text: str,
        source_feature: torch.Tensor,
        target_concept: str,
        layer_idx: int = 11
    ) -> Dict[str, Any]:
        """
        分析因果效应
        
        实验设计:
        1. 获取正常激活
        2. 干预源特征
        3. 观察目标概念的变化
        """
        results = {
            "input": input_text,
            "source_feature": source_feature.shape,
            "target_concept": target_concept,
            "layer": layer_idx
        }
        
        # 这里需要实际实现干预和观察逻辑
        # 伪代码：
        # 1. 正常前向传播，获取激活
        # 2. 干预后前向传播，获取激活
        # 3. 对比差异
        
        return results


class ContrastiveMechanismAnalyzer:
    """
    对比机制分析器
    
    目标: 通过对比理解机制
    
    方法:
    1. 相同输入，不同输出 → 发现选择性机制
    2. 不同输入，相同输出 → 发现不变性机制
    3. 相似输入，不同输出 → 发现敏感特征
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def analyze_selectivity(
        self,
        same_input: str,
        different_outputs: List[str]
    ) -> Dict[str, Any]:
        """
        分析选择性机制
        
        问题: 相同输入如何产生不同输出？
        
        方法:
        - 对比不同输出的激活模式
        - 发现决定输出的关键特征
        """
        results = {
            "input": same_input,
            "outputs": different_outputs,
            "selective_features": []
        }
        
        # 获取每个输出的激活
        activations = []
        for output in different_outputs:
            # 这里需要实际实现
            pass
        
        return results
    
    def analyze_invariance(
        self,
        different_inputs: List[str],
        same_output: str
    ) -> Dict[str, Any]:
        """
        分析不变性机制
        
        问题: 不同输入如何产生相同输出？
        
        方法:
        - 对比不同输入的激活模式
        - 发现不变的公共特征
        """
        results = {
            "inputs": different_inputs,
            "output": same_output,
            "invariant_features": []
        }
        
        return results


def run_feature_emergence_tracking():
    """
    运行特征涌现追踪实验
    
    这是解决"只有统计描述"问题的第一步
    """
    print("=" * 60)
    print("特征涌现追踪实验")
    print("=" * 60)
    print()
    print("目标: 理解神经网络如何从信号流中提取特征")
    print()
    print("方法:")
    print("1. 训练模型，每100步保存激活")
    print("2. 追踪特征何时出现")
    print("3. 分析特征涌现规律")
    print()
    
    config = EmergenceConfig(
        model_type="gpt2",
        model_size="small",
        save_interval=100,
        total_steps=5000,
        track_layers=[0, 3, 6, 9, 11]
    )
    
    tracker = FeatureEmergenceTracker(config)
    
    print("追踪器初始化完成")
    print(f"追踪层: {config.track_layers}")
    print(f"保存间隔: {config.save_interval}步")
    print()
    
    # 这里需要实际的训练循环
    # 伪代码：
    # for step in range(total_steps):
    #     batch = get_batch()
    #     loss = model(batch)
    #     loss.backward()
    #     optimizer.step()
    #     
    #     if step % save_interval == 0:
    #         tracker.track_training_step(model, step, batch, optimizer)
    
    print("实验框架已创建，需要接入实际训练循环")
    print()
    print("输出:")
    print("  - results/feature_emergence/emergence_tracking_results.json")
    print("  - 包含: 涌现时间线、层涌现顺序、关键发现")


if __name__ == "__main__":
    run_feature_emergence_tracking()
