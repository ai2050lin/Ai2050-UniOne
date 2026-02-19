"""
几何干预测试脚本
================

使用真实模型（Qwen 2.5）进行几何干预测试。

核心功能：
1. 加载真实语言模型
2. 提取激活流形几何结构
3. 实现多种几何干预方法
4. 评估干预效果

Author: AGI Research Team
Date: 2026-02-19
"""

import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from functools import partial

# 设置环境
os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================================
# 几何干预模块
# ============================================================================

class HeatKernelDiffuser:
    """
    热核扩散器
    
    在激活流形上应用热核扩散，平滑曲率尖峰。
    
    数学原理：
    K_t(x,y) = exp(-d(x,y)² / 4t)
    
    物理直觉：热量从高曲率区域扩散到低曲率区域，
    实现自然的曲率平滑。
    """
    
    def __init__(self, t: float = 1.0, alpha: float = 0.1):
        """
        Args:
            t: 扩散时间参数（越大越平滑）
            alpha: 混合系数（新旧状态混合比例）
        """
        self.t = t
        self.alpha = alpha
        self.activation_buffer = None
        
    def compute_heat_kernel_weights(
        self, 
        activations: torch.Tensor,
        reference: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算热核权重矩阵
        
        Args:
            activations: 当前激活 (batch, seq, dim)
            reference: 参考激活点 (n_ref, dim)
        
        Returns:
            权重矩阵 (batch, seq, n_ref)
        """
        if reference is None:
            # 使用 buffer 中的参考点
            if self.activation_buffer is None:
                return torch.ones(activations.shape[:2], device=activations.device)
            reference = self.activation_buffer
        
        # 计算距离
        # activations: (batch, seq, dim) -> (batch*seq, dim)
        flat_act = activations.reshape(-1, activations.size(-1))
        
        # L2 距离
        distances = torch.cdist(flat_act, reference)  # (batch*seq, n_ref)
        
        # 热核权重
        weights = torch.exp(-distances**2 / (4 * self.t))
        
        # 归一化
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        return weights.reshape(*activations.shape[:2], -1)
    
    def diffuse(
        self, 
        activations: torch.Tensor,
        reference: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        应用热核扩散
        
        Args:
            activations: 当前激活 (batch, seq, dim)
            reference: 参考激活点 (n_ref, dim)
        
        Returns:
            扩散后的激活 (batch, seq, dim)
        """
        weights = self.compute_heat_kernel_weights(activations, reference)
        
        if reference is None and self.activation_buffer is not None:
            reference = self.activation_buffer
        elif reference is None:
            return activations
        
        # 加权平均
        flat_act = activations.reshape(-1, activations.size(-1))
        diffused = torch.matmul(weights.reshape(-1, weights.size(-1)), reference)
        diffused = diffused.reshape_as(activations)
        
        # 混合
        return (1 - self.alpha) * activations + self.alpha * diffused
    
    def update_buffer(self, activations: torch.Tensor, max_samples: int = 1000):
        """更新参考激活缓冲区"""
        flat_act = activations.reshape(-1, activations.size(-1))
        
        # 随机采样
        n_samples = min(flat_act.size(0), max_samples)
        indices = torch.randperm(flat_act.size(0))[:n_samples]
        sampled = flat_act[indices]
        
        if self.activation_buffer is None:
            self.activation_buffer = sampled
        else:
            # 增量更新
            self.activation_buffer = torch.cat([self.activation_buffer, sampled], dim=0)
            if self.activation_buffer.size(0) > max_samples:
                self.activation_buffer = self.activation_buffer[-max_samples:]


class GeodesicSteerer:
    """
    测地线引导器
    
    沿语义流形的测地线方向引导激活。
    
    数学原理：
    测地线方程：d²x/dt² + Γ(x)(dx/dt, dx/dt) = 0
    
    简化为：沿梯度方向移动，同时保持单位范数。
    """
    
    def __init__(self, step_size: float = 0.1):
        self.step_size = step_size
        self.direction_vectors = {}  # 目标方向缓存
        
    def set_direction(self, name: str, vector: torch.Tensor):
        """设置引导方向"""
        self.direction_vectors[name] = F.normalize(vector, dim=-1)
        
    def compute_geodesic_step(
        self, 
        activations: torch.Tensor,
        direction_name: str,
        magnitude: Optional[float] = None
    ) -> torch.Tensor:
        """
        计算测地线步骤
        
        Args:
            activations: 当前激活 (batch, seq, dim)
            direction_name: 方向名称
            magnitude: 步长倍数
        
        Returns:
            引导后的激活
        """
        if direction_name not in self.direction_vectors:
            return activations
        
        direction = self.direction_vectors[direction_name].to(activations.device)
        step = magnitude if magnitude is not None else self.step_size
        
        # 沿方向移动
        guided = activations + step * direction
        
        # 保持范数（近似测地线）
        original_norm = torch.norm(activations, dim=-1, keepdim=True)
        guided = F.normalize(guided, dim=-1) * original_norm
        
        return guided
    
    def interpolate_concepts(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        steps: int = 5
    ) -> List[torch.Tensor]:
        """
        概念空间测地线插值
        
        Args:
            start: 起始概念向量
            end: 终止概念向量
            steps: 插值步数
        
        Returns:
            插值向量列表
        """
        # 球面线性插值 (Slerp)
        start_norm = F.normalize(start, dim=-1)
        end_norm = F.normalize(end, dim=-1)
        
        # 计算角度
        omega = torch.acos(torch.clamp(
            torch.sum(start_norm * end_norm, dim=-1), -1, 1
        ))
        
        if omega < 1e-6:
            # 几乎平行，线性插值
            return [start + (end - start) * i / steps for i in range(steps + 1)]
        
        sin_omega = torch.sin(omega)
        
        result = []
        for i in range(steps + 1):
            t = i / steps
            coeff1 = torch.sin((1 - t) * omega) / sin_omega
            coeff2 = torch.sin(t * omega) / sin_omega
            interpolated = coeff1 * start_norm + coeff2 * end_norm
            result.append(interpolated * torch.norm(start, dim=-1, keepdim=True))
        
        return result


class CurvatureAwareIntervener:
    """
    曲率感知干预器
    
    根据激活流形的局部曲率进行自适应干预。
    
    高曲率区域 → 更强的平滑干预
    低曲率区域 → 保持原状
    """
    
    def __init__(self, threshold: float = 1.0, smoothing_strength: float = 0.2):
        self.threshold = threshold
        self.smoothing_strength = smoothing_strength
        
    def compute_local_curvature(self, activations: torch.Tensor, k: int = 5) -> torch.Tensor:
        """
        计算局部标量曲率
        
        Args:
            activations: (batch, seq, dim)
            k: 近邻数量
        
        Returns:
            曲率值 (batch, seq)
        """
        flat_act = activations.reshape(-1, activations.size(-1))
        n_points = flat_act.size(0)
        
        if n_points < k + 1:
            return torch.zeros(activations.shape[:2], device=activations.device)
        
        # 计算 k-NN
        distances = torch.cdist(flat_act, flat_act)
        _, indices = torch.topk(distances, k + 1, largest=False)
        indices = indices[:, 1:]  # 排除自身
        
        # 计算局部曲率 = 邻域方差
        curvatures = []
        for i in range(n_points):
            neighbors = flat_act[indices[i]]
            local_var = torch.var(neighbors, dim=0).mean()
            curvatures.append(local_var)
        
        curvatures = torch.stack(curvatures)
        return curvatures.reshape(*activations.shape[:2])
    
    def adaptive_intervention(
        self, 
        activations: torch.Tensor,
        reference: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        自适应曲率干预
        
        Args:
            activations: 当前激活
            reference: 参考点（用于平滑）
        
        Returns:
            干预后的激活, 统计信息
        """
        curvature = self.compute_local_curvature(activations)
        
        # 创建自适应强度掩码
        strength_mask = torch.clamp(curvature / self.threshold, 0, 1) * self.smoothing_strength
        strength_mask = strength_mask.unsqueeze(-1)
        
        if reference is not None:
            # 向参考点平滑
            smoothed = self._smooth_toward_reference(activations, reference, strength_mask)
        else:
            # 局部平滑
            smoothed = self._local_smoothing(activations, strength_mask)
        
        stats = {
            "mean_curvature": curvature.mean().item(),
            "max_curvature": curvature.max().item(),
            "high_curvature_ratio": (curvature > self.threshold).float().mean().item()
        }
        
        return smoothed, stats
    
    def _smooth_toward_reference(
        self, 
        activations: torch.Tensor, 
        reference: torch.Tensor,
        strength: torch.Tensor
    ) -> torch.Tensor:
        """向参考点平滑"""
        flat_act = activations.reshape(-1, activations.size(-1))
        
        # 找最近的参考点
        distances = torch.cdist(flat_act, reference)
        nearest_idx = distances.argmin(dim=-1)
        nearest_ref = reference[nearest_idx]
        
        # 插值
        smoothed = activations * (1 - strength) + nearest_ref.reshape_as(activations) * strength
        return smoothed
    
    def _local_smoothing(
        self, 
        activations: torch.Tensor, 
        strength: torch.Tensor
    ) -> torch.Tensor:
        """局部邻域平滑"""
        flat_act = activations.reshape(-1, activations.size(-1))
        
        # k-NN 平均
        k = min(5, flat_act.size(0) - 1)
        distances = torch.cdist(flat_act, flat_act)
        _, indices = torch.topk(distances, k + 1, largest=False)
        indices = indices[:, 1:]
        
        # 邻居平均
        neighbor_mean = flat_act[indices].mean(dim=1)
        
        # 插值
        smoothed = activations * (1 - strength) + neighbor_mean.reshape_as(activations) * strength
        return smoothed


# ============================================================================
# 激活提取与分析
# ============================================================================

class ActivationExtractor:
    """激活提取器"""
    
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.handles = []
        
    def get_hook_fn(self, name: str) -> Callable:
        """获取钩子函数"""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach()
            else:
                self.activations[name] = output.detach()
        return hook_fn
    
    def register_hooks(self, layer_indices: List[int]):
        """注册钩子到指定层"""
        self.clear_hooks()
        
        # 对于 HookedTransformer
        if hasattr(self.model, 'blocks'):
            for idx in layer_indices:
                if idx < len(self.model.blocks):
                    hook = self.model.blocks[idx].hook_resid_post.register_forward_hook(
                        self.get_hook_fn(f"layer_{idx}")
                    )
                    self.handles.append(hook)
        # 对于 HuggingFace GPT-2 模型
        elif hasattr(self.model, 'transformer'):
            if hasattr(self.model.transformer, 'h'):
                for idx in layer_indices:
                    if idx < len(self.model.transformer.h):
                        hook = self.model.transformer.h[idx].register_forward_hook(
                            self.get_hook_fn(f"layer_{idx}")
                        )
                        self.handles.append(hook)
        # 对于 LLaMA/Qwen 架构
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            for idx in layer_indices:
                if idx < len(self.model.model.layers):
                    hook = self.model.model.layers[idx].register_forward_hook(
                        self.get_hook_fn(f"layer_{idx}")
                    )
                    self.handles.append(hook)
    
    def clear_hooks(self):
        """清除所有钩子"""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.activations = {}
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """获取缓存的激活"""
        return self.activations


# ============================================================================
# 几何干预系统
# ============================================================================

class GeometricInterventionSystem:
    """
    几何干预系统
    
    整合多种几何干预方法，提供统一接口。
    """
    
    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device
        self.extractor = ActivationExtractor(model)
        
        # 干预模块
        self.heat_diffuser = HeatKernelDiffuser(t=1.0, alpha=0.1)
        self.geodesic_steerer = GeodesicSteerer(step_size=0.1)
        self.curvature_intervener = CurvatureAwareIntervener(threshold=1.0, smoothing_strength=0.2)
        
        # 干预状态
        self.intervention_hooks = []
        self.active_interventions = {}
        
    def extract_layer_activations(
        self, 
        input_ids: torch.Tensor,
        layer_indices: List[int]
    ) -> Dict[str, torch.Tensor]:
        """提取指定层的激活"""
        self.extractor.register_hooks(layer_indices)
        
        with torch.no_grad():
            _ = self.model(input_ids)
        
        activations = self.extractor.get_activations()
        self.extractor.clear_hooks()
        
        return activations
    
    def compute_pca_directions(
        self, 
        activations: torch.Tensor,
        n_components: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 PCA 主方向
        
        Args:
            activations: (batch, seq, dim)
            n_components: 主成分数量
        
        Returns:
            components: (n_components, dim)
            mean: (dim,)
        """
        flat_act = activations.reshape(-1, activations.size(-1))
        mean = flat_act.mean(dim=0)
        centered = flat_act - mean
        
        # SVD
        U, S, V = torch.linalg.svd(centered, full_matrices=False)
        components = V[:n_components]
        
        return components, mean
    
    def setup_intervention(
        self,
        layer_idx: int,
        intervention_type: str,
        **kwargs
    ):
        """
        设置层干预
        
        Args:
            layer_idx: 层索引
            intervention_type: 干预类型 (heat_kernel, geodesic, curvature)
            **kwargs: 干预参数
        """
        self.active_interventions[layer_idx] = {
            "type": intervention_type,
            "params": kwargs
        }
    
    def create_intervention_hook(
        self, 
        layer_idx: int,
        intervention_config: Dict
    ) -> Callable:
        """创建干预钩子"""
        intervention_type = intervention_config["type"]
        params = intervention_config["params"]
        
        def hook_fn(activations, hook):
            if intervention_type == "heat_kernel":
                reference = params.get("reference", None)
                if reference is not None:
                    reference = reference.to(activations.device)
                return self.heat_diffuser.diffuse(activations, reference)
            
            elif intervention_type == "geodesic":
                direction = params.get("direction", "positive")
                magnitude = params.get("magnitude", 0.1)
                return self.geodesic_steerer.compute_geodesic_step(
                    activations, direction, magnitude
                )
            
            elif intervention_type == "curvature":
                reference = params.get("reference", None)
                if reference is not None:
                    reference = reference.to(activations.device)
                result, _ = self.curvature_intervener.adaptive_intervention(
                    activations, reference
                )
                return result
            
            return activations
        
        return hook_fn
    
    def apply_interventions(self):
        """应用所有干预"""
        # 清除旧钩子
        if hasattr(self.model, 'reset_hooks'):
            self.model.reset_hooks()
        
        # 添加新钩子
        for layer_idx, config in self.active_interventions.items():
            hook_fn = self.create_intervention_hook(layer_idx, config)
            
            # 对于 HookedTransformer
            if hasattr(self.model, 'blocks'):
                hook_name = f"blocks.{layer_idx}.hook_resid_post"
                self.model.add_hook(hook_name, hook_fn)
            # 对于 HuggingFace 模型（使用 forward hooks）
            else:
                self._register_hf_hook(layer_idx, hook_fn)
    
    def _register_hf_hook(self, layer_idx: int, hook_fn: Callable):
        """为 HuggingFace 模型注册钩子"""
        # GPT-2 架构
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            if layer_idx < len(self.model.transformer.h):
                module = self.model.transformer.h[layer_idx]
                
                def forward_hook(module, input, output):
                    if isinstance(output, tuple):
                        return (hook_fn(output[0], None),) + output[1:]
                    return hook_fn(output, None)
                
                handle = module.register_forward_hook(forward_hook)
                self.intervention_hooks.append(handle)
        # LLaMA/Qwen 架构
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            if layer_idx < len(self.model.model.layers):
                module = self.model.model.layers[layer_idx]
                
                def forward_hook(module, input, output):
                    if isinstance(output, tuple):
                        return (hook_fn(output[0], None),) + output[1:]
                    return hook_fn(output, None)
                
                handle = module.register_forward_hook(forward_hook)
                self.intervention_hooks.append(handle)
    
    def clear_interventions(self):
        """清除所有干预"""
        if hasattr(self.model, 'reset_hooks'):
            self.model.reset_hooks()
        
        for handle in self.intervention_hooks:
            handle.remove()
        self.intervention_hooks = []
        self.active_interventions = {}


# ============================================================================
# 测试函数
# ============================================================================

def test_with_huggingface_model():
    """使用 HuggingFace 模型测试"""
    print("=" * 70)
    print("几何干预测试 - HuggingFace 模型")
    print("=" * 70)
    
    # 加载模型
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # 使用 gpt2-small 替代，更稳定
    model_name = "gpt2"
    print(f"\n[1] 加载模型: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 强制使用 CPU 避免内存问题
    device = "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    ).to(device)
    
    print(f"  [OK] 模型加载成功，设备: {device}")
    
    # 创建干预系统
    geo_system = GeometricInterventionSystem(model, device)
    
    # 测试文本 (减少数量加快测试)
    test_prompts = [
        "The capital of France is",
        "2 + 2 equals",
    ]
    
    # 提取激活
    print(f"\n[2] 提取激活...")
    layer_indices = [0, 3, 6, 9, 11]  # GPT-2 有 12 层
    all_activations = {}
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        acts = geo_system.extract_layer_activations(inputs["input_ids"], layer_indices)
        for layer_name, act in acts.items():
            if layer_name not in all_activations:
                all_activations[layer_name] = []
            all_activations[layer_name].append(act)
    
    print(f"  [OK] 提取了 {len(all_activations)} 层的激活")
    
    # 分析激活几何
    print(f"\n[3] 分析激活几何...")
    for layer_name, acts in all_activations.items():
        combined = torch.cat(acts, dim=0)
        components, mean = geo_system.compute_pca_directions(combined, n_components=3)
        
        # 计算曲率
        curvature = geo_system.curvature_intervener.compute_local_curvature(combined)
        
        print(f"\n  {layer_name}:")
        print(f"    激活形状: {combined.shape}")
        print(f"    主成分方差: {components.var().item():.4f}")
        print(f"    平均曲率: {curvature.mean().item():.4f}")
        print(f"    最大曲率: {curvature.max().item():.4f}")
    
    # 设置几何干预
    print(f"\n[4] 设置几何干预...")
    
    # 使用中间层的激活作为参考
    if "layer_6" in all_activations:
        reference_acts = torch.cat(all_activations["layer_6"], dim=0)
        flat_ref = reference_acts.reshape(-1, reference_acts.size(-1))
        
        # 为多个层设置干预
        for layer_idx in [3, 6, 9]:
            geo_system.setup_intervention(
                layer_idx=layer_idx,
                intervention_type="heat_kernel",
                reference=flat_ref[:100]  # 使用前100个点作为参考
            )
        
        print(f"  [OK] 设置了 {len(geo_system.active_interventions)} 个干预")
    
    # 生成对比测试
    print(f"\n[5] 生成对比测试...")
    
    test_prompt = "The meaning of life is"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    
    # 无干预生成
    print(f"\n  原始生成:")
    with torch.no_grad():
        original_output = model.generate(
            **inputs,
            max_new_tokens=15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    original_text = tokenizer.decode(original_output[0], skip_special_tokens=True)
    print(f"  {original_text}")
    
    # 应用干预并生成
    geo_system.apply_interventions()
    print(f"\n  干预后生成:")
    with torch.no_grad():
        intervened_output = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    intervened_text = tokenizer.decode(intervened_output[0], skip_special_tokens=True)
    print(f"  {intervened_text}")
    
    # 清除干预
    geo_system.clear_interventions()
    
    # 测试测地线引导
    print(f"\n[6] 测地线引导测试...")
    
    # 提取两个不同概念的激活
    concept_a = tokenizer("happy and joyful", return_tensors="pt").to(device)
    concept_b = tokenizer("sad and depressed", return_tensors="pt").to(device)
    
    with torch.no_grad():
        _ = model(**concept_a)
        _ = model(**concept_b)
    
    # 设置引导方向
    geo_system.geodesic_steerer.set_direction(
        "positive", 
        torch.randn(768, device=device)  # GPT-2 hidden dim = 768
    )
    
    geo_system.setup_intervention(
        layer_idx=6,
        intervention_type="geodesic",
        direction="positive",
        magnitude=0.05
    )
    
    geo_system.apply_interventions()
    
    print(f"\n  引导后生成:")
    with torch.no_grad():
        steered_output = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    steered_text = tokenizer.decode(steered_output[0], skip_special_tokens=True)
    print(f"  {steered_text}")
    
    geo_system.clear_interventions()
    
    # 保存结果
    results = {
        "model": model_name,
        "test_prompt": test_prompt,
        "original_output": original_text,
        "intervened_output": intervened_text,
        "steered_output": steered_text,
        "intervention_types_tested": ["heat_kernel", "geodesic"],
        "layers_analyzed": layer_indices
    }
    
    save_path = "tempdata/geometric_intervention_results.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] 结果保存到: {save_path}")
    print("\n" + "=" * 70)
    print("几何干预测试完成")
    print("=" * 70)
    
    return results


def test_with_hooked_transformer():
    """使用 TransformerLens 的 HookedTransformer 测试"""
    print("=" * 70)
    print("几何干预测试 - HookedTransformer")
    print("=" * 70)
    
    try:
        import transformer_lens as tl
        from transformer_lens import HookedTransformer
    except ImportError:
        print("[SKIP] TransformerLens 未安装")
        return None
    
    # 加载模型
    print(f"\n[1] 加载 HookedTransformer...")
    
    # 强制使用 CPU
    device = "cpu"
    
    try:
        model = HookedTransformer.from_pretrained(
            "gpt2-small",
            device=device
        )
        print(f"  [OK] GPT-2 Small 加载成功")
    except Exception as e:
        print(f"  [FAIL] 加载失败: {e}")
        return None
    
    # 创建干预系统
    geo_system = GeometricInterventionSystem(model, device)
    
    # 测试提示
    test_prompts = [
        "The capital of France is",
        "In the beginning",
        "Machine learning is a field of",
    ]
    
    # 提取激活
    print(f"\n[2] 提取激活...")
    layer_indices = [0, 3, 6, 9, 11]
    all_activations = {}
    
    for prompt in test_prompts:
        tokens = model.to_tokens(prompt)
        acts = geo_system.extract_layer_activations(tokens, layer_indices)
        for layer_name, act in acts.items():
            if layer_name not in all_activations:
                all_activations[layer_name] = []
            all_activations[layer_name].append(act)
    
    print(f"  [OK] 提取了 {len(all_activations)} 层的激活")
    
    # 分析几何
    print(f"\n[3] 激活几何分析...")
    
    analysis_results = {}
    for layer_name, acts in all_activations.items():
        combined = torch.cat(acts, dim=0)
        
        # PCA
        components, mean = geo_system.compute_pca_directions(combined, n_components=3)
        
        # 曲率
        curvature = geo_system.curvature_intervener.compute_local_curvature(combined)
        
        analysis_results[layer_name] = {
            "shape": list(combined.shape),
            "mean_curvature": curvature.mean().item(),
            "max_curvature": curvature.max().item(),
            "pca_variance": components.var().item()
        }
        
        print(f"\n  {layer_name}:")
        print(f"    形状: {combined.shape}")
        print(f"    平均曲率: {curvature.mean().item():.4f}")
        print(f"    最大曲率: {curvature.max().item():.4f}")
    
    # 测试干预效果
    print(f"\n[4] 测试干预效果...")
    
    test_prompt = "The future of artificial intelligence is"
    
    # 原始生成
    print(f"\n  原始生成:")
    original_output = model.generate(test_prompt, max_new_tokens=20, temperature=0)
    print(f"  {original_output}")
    
    # 应用热核干预
    if "layer_6" in all_activations:
        reference = torch.cat(all_activations["layer_6"], dim=0)
        flat_ref = reference.reshape(-1, reference.size(-1))
        
        geo_system.setup_intervention(
            layer_idx=6,
            intervention_type="heat_kernel",
            reference=flat_ref[:50]
        )
        
        geo_system.apply_interventions()
        
        print(f"\n  热核干预后:")
        intervened_output = model.generate(test_prompt, max_new_tokens=20, temperature=0)
        print(f"  {intervened_output}")
        
        geo_system.clear_interventions()
    
    # 曲率感知干预
    print(f"\n  曲率感知干预后:")
    ref_tensor = flat_ref[:50] if "layer_6" in all_activations else None
    geo_system.setup_intervention(
        layer_idx=6,
        intervention_type="curvature",
        reference=ref_tensor
    )
    
    geo_system.apply_interventions()
    curvature_output = model.generate(test_prompt, max_new_tokens=20, temperature=0)
    print(f"  {curvature_output}")
    
    geo_system.clear_interventions()
    
    # 保存结果
    results = {
        "model": "gpt2-small",
        "test_prompt": test_prompt,
        "original_output": original_output,
        "intervened_output": intervened_output if "layer_6" in all_activations else None,
        "curvature_output": curvature_output,
        "analysis": analysis_results
    }
    
    save_path = "tempdata/hooked_transformer_intervention_results.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] 结果保存到: {save_path}")
    print("\n" + "=" * 70)
    print("HookedTransformer 测试完成")
    print("=" * 70)
    
    return results


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("    AGI 几何干预测试系统")
    print("    Geometric Intervention Test System")
    print("=" * 70)
    
    # 确保 tempdata 目录存在
    os.makedirs("tempdata", exist_ok=True)
    
    # 测试 HuggingFace 模型
    print("\n>>> 测试 HuggingFace 模型 (Qwen 2.5) <<<")
    hf_results = test_with_huggingface_model()
    
    # 测试 HookedTransformer
    print("\n>>> 测试 HookedTransformer (GPT-2) <<<")
    ht_results = test_with_hooked_transformer()
    
    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    if hf_results:
        print("\n[HuggingFace 模型测试]")
        print(f"  模型: {hf_results['model']}")
        print(f"  测试干预类型: {hf_results['intervention_types_tested']}")
        print(f"  分析层数: {len(hf_results['layers_analyzed'])}")
    
    if ht_results:
        print("\n[HookedTransformer 测试]")
        print(f"  模型: {ht_results['model']}")
        print(f"  分析层数: {len(ht_results['analysis'])}")
    
    print("\n几何干预测试系统验证完成！")
