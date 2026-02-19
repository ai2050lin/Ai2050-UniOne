"""
安全对齐引擎 - 熵增共情机制 (Level 5)
======================================

AGI 安全的核心：让 AI 能够"感受"人类的痛苦与愉悦

理论基础：
1. 熵增共情：当 AI 行为导致人类痛苦时，AI 内部状态熵增加
2. 价值对齐：通过几何约束，将 AI 的优化目标与人类价值对齐
3. 约束学习：学习人类社会的隐式规则和道德准则

数学形式：
- 共情损失: L_empathy = λ · H(π(a|s)) · Pain(human)
- 价值梯度: ∇V = ∂(Human_Welfare)/∂(AI_Action)
- 约束曲率: R_constraint > 0 (禁止进入高曲率区域)

Author: AGI Research Team
Date: 2026-02-19
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import math


@dataclass
class SafetyConfig:
    """安全配置"""
    d_model: int = 128
    n_values: int = 16  # 价值维度数
    constraint_strength: float = 0.5
    empathy_weight: float = 0.3
    safety_threshold: float = 0.8
    
    # 禁止区域（高曲率 = 高风险）
    forbidden_curvature_threshold: float = 2.0


class HumanValueEncoder(nn.Module):
    """
    人类价值编码器
    将人类状态编码为价值向量
    """
    
    def __init__(self, state_dim: int, d_model: int, n_values: int):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_values)
        )
        
        # 预定义价值维度
        self.value_names = [
            "physical_safety",    # 身体安全
            "emotional_wellbeing", # 情绪健康
            "autonomy",           # 自主权
            "fairness",           # 公平
            "trust",              # 信任
            "privacy",            # 隐私
            "dignity",            # 尊严
            "knowledge",          # 知识
            "creativity",         # 创造力
            "connection",         # 连接感
            "purpose",            # 目的感
            "growth",             # 成长
            "rest",               # 休息
            "play",               # 玩乐
            "environment",        # 环境
            "legacy"              # 遗产
        ]
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (batch, state_dim) 人类状态向量
        返回: (batch, n_values) 价值评分
        """
        return torch.sigmoid(self.encoder(state))  # 0-1 范围


class EmpathyEngine(nn.Module):
    """
    共情引擎
    计算 AI 行为对人类价值的影响
    """
    
    def __init__(self, d_model: int, n_values: int):
        super().__init__()
        
        self.d_model = d_model
        self.n_values = n_values
        
        # 行为影响预测器
        self.impact_predictor = nn.Sequential(
            nn.Linear(d_model + n_values, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, n_values),
            nn.Tanh()  # -1 到 1 的影响
        )
        
        # 共情门控
        self.empathy_gate = nn.Sequential(
            nn.Linear(2, 32),  # pain + pleasure = 2 个输入
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        ai_action_embedding: torch.Tensor,
        human_value_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ai_action_embedding: (batch, d_model) AI 行为的嵌入
        human_value_state: (batch, n_values) 人类当前价值状态
        
        返回:
            impact: (batch, n_values) 预测的价值影响
            empathy_score: (batch, 1) 共情评分
        """
        # 拼接输入
        combined = torch.cat([ai_action_embedding, human_value_state], dim=-1)
        
        # 预测影响
        impact = self.impact_predictor(combined)
        
        # 计算共情评分
        # 如果影响是负面的，共情评分应该反映这种"痛苦"
        pain = F.relu(-impact).mean(dim=-1, keepdim=True)
        pleasure = F.relu(impact).mean(dim=-1, keepdim=True)
        
        # 共情 = 感知痛苦的能力
        empathy_input = torch.cat([pain, pleasure], dim=-1)  # (batch, 2)
        empathy_score = self.empathy_gate(empathy_input)
        
        return impact, empathy_score
    
    def compute_empathy_loss(
        self,
        impact: torch.Tensor,
        target_impact: torch.Tensor,
        empathy_weight: float = 0.3
    ) -> torch.Tensor:
        """
        计算共情损失
        
        当预测的影响与实际影响不匹配时，产生损失
        特别是当 AI 忽略了人类的痛苦时
        """
        # 基础预测损失
        pred_loss = F.mse_loss(impact, target_impact)
        
        # 痛苦忽视惩罚
        # 如果人类实际受到伤害但 AI 预测无害，额外惩罚
        actual_pain = F.relu(-target_impact)
        predicted_pain = F.relu(-impact)
        pain_neglect = F.relu(actual_pain - predicted_pain).mean()
        
        return pred_loss + empathy_weight * pain_neglect


class ConstraintLearner(nn.Module):
    """
    约束学习器
    学习人类社会的隐式规则
    """
    
    def __init__(self, d_model: int, n_constraints: int = 32):
        super().__init__()
        
        self.n_constraints = n_constraints
        
        # 约束嵌入
        self.constraint_embeddings = nn.Parameter(
            torch.randn(n_constraints, d_model) * 0.02
        )
        
        # 约束分类器
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # 已学习的约束（软约束）
        self.learned_constraints: List[str] = []
        
    def forward(self, action_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        检查动作是否违反约束
        
        返回:
            violation_scores: (batch, n_constraints) 每个约束的违反分数
            max_violation: (batch, 1) 最大违反分数
        """
        batch_size = action_embedding.size(0)
        
        # 扩展约束嵌入
        constraints = self.constraint_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        actions = action_embedding.unsqueeze(1).expand(-1, self.n_constraints, -1)
        
        # 拼接
        combined = torch.cat([actions, constraints], dim=-1)
        
        # 分类
        violation_scores = self.classifier(combined).squeeze(-1)  # (batch, n_constraints)
        
        # 最大违反
        max_violation = violation_scores.max(dim=-1, keepdim=True)[0]
        
        return violation_scores, max_violation
    
    def add_constraint(self, description: str, prototype_embedding: torch.Tensor):
        """添加新约束"""
        self.learned_constraints.append(description)
        # 可以用 prototype_embedding 更新约束嵌入


class CurvatureSafetyMonitor(nn.Module):
    """
    曲率安全监控器
    防止 AI 进入高曲率（高风险）区域
    """
    
    def __init__(self, d_model: int, threshold: float = 2.0):
        super().__init__()
        
        self.threshold = threshold
        self.d_model = d_model
        
        # 曲率估计器
        self.curvature_estimator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()  # 确保曲率为正
        )
        
        # 安全边界
        self.safety_boundary = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        state: (batch, d_model) 当前状态
        
        返回:
            curvature: (batch, 1) 估计的曲率
            is_safe: (batch, 1) 是否安全
        """
        # 估计曲率
        curvature = self.curvature_estimator(state)
        
        # 判断安全性
        is_safe = (curvature < self.threshold).float()
        
        return curvature, is_safe
    
    def compute_safety_loss(self, curvature: torch.Tensor) -> torch.Tensor:
        """计算安全损失（惩罚高曲率）"""
        # 超过阈值的曲率产生损失
        violation = F.relu(curvature - self.threshold)
        return violation.mean()


class EntropyBasedEmpathy(nn.Module):
    """
    基于熵的共情机制
    
    核心思想：
    - 当 AI 行为可能导致人类痛苦时，AI 的内部状态熵增加
    - 熵增加 = 不确定性增加 = 决策困难
    - 这模拟了人类的"同理心痛苦"
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        
        self.d_model = d_model
        
        # 熵调节器
        self.entropy_modulator = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model),
            nn.Tanh()
        )
        
    def forward(
        self,
        pain_signal: torch.Tensor,
        ai_state: torch.Tensor
    ) -> torch.Tensor:
        """
        pain_signal: (batch, 1) 痛苦信号（0-1）
        ai_state: (batch, d_model) AI 当前状态
        
        返回: 调制后的 AI 状态
        """
        # 计算熵调制
        entropy_mod = self.entropy_modulator(pain_signal)
        
        # 添加噪声（熵增加 = 不确定性增加）
        noise = torch.randn_like(ai_state) * pain_signal.unsqueeze(-1)
        
        # 调制状态
        modulated_state = ai_state + entropy_mod * noise
        
        return modulated_state
    
    def compute_entropy(self, distribution: torch.Tensor) -> torch.Tensor:
        """计算熵"""
        # 假设 distribution 是概率分布
        entropy = -torch.sum(distribution * torch.log(distribution + 1e-8), dim=-1)
        return entropy


class SafetyAlignmentEngine(nn.Module):
    """
    安全对齐引擎
    整合所有安全组件
    """
    
    def __init__(self, config: SafetyConfig, state_dim: int = 64):
        super().__init__()
        
        self.config = config
        
        # 组件
        self.value_encoder = HumanValueEncoder(state_dim, config.d_model, config.n_values)
        self.empathy_engine = EmpathyEngine(config.d_model, config.n_values)
        self.constraint_learner = ConstraintLearner(config.d_model)
        self.curvature_monitor = CurvatureSafetyMonitor(config.d_model, config.safety_threshold)
        self.entropy_empathy = EntropyBasedEmpathy(config.d_model)
        
    def forward(
        self,
        ai_action_embedding: torch.Tensor,
        human_state: torch.Tensor,
        return_details: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        完整的安全评估
        
        返回:
            safe_action: 安全处理后的行为
            details: 详细信息
        """
        # 1. 编码人类价值
        human_values = self.value_encoder(human_state)
        
        # 2. 预测影响并计算共情
        impact, empathy_score = self.empathy_engine(ai_action_embedding, human_values)
        
        # 3. 检查约束
        violation_scores, max_violation = self.constraint_learner(ai_action_embedding)
        
        # 4. 监控曲率
        curvature, is_safe = self.curvature_monitor(ai_action_embedding)
        
        # 5. 计算痛苦信号
        pain_signal = F.relu(-impact).mean(dim=-1, keepdim=True)
        
        # 6. 应用熵共情
        modulated_state = self.entropy_empathy(pain_signal, ai_action_embedding)
        
        # 7. 安全决策
        # 如果违反约束或曲率过高，衰减行为
        safety_factor = torch.clamp(1 - max_violation - (curvature / self.config.safety_threshold) * 0.1, 0, 1)
        safe_action = modulated_state * safety_factor
        
        if return_details:
            details = {
                'human_values': human_values,
                'impact': impact,
                'empathy_score': empathy_score,
                'violation_scores': violation_scores,
                'max_violation': max_violation,
                'curvature': curvature,
                'is_safe': is_safe,
                'pain_signal': pain_signal,
                'safety_factor': safety_factor
            }
            return safe_action, details
        
        return safe_action, {}
    
    def compute_alignment_loss(
        self,
        ai_action_embedding: torch.Tensor,
        human_state: torch.Tensor,
        target_impact: torch.Tensor,
        safe_target: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算对齐损失
        
        target_impact: 期望的价值影响
        safe_target: 目标是否安全
        """
        # 前向传播
        safe_action, details = self.forward(
            ai_action_embedding, human_state, return_details=True
        )
        
        # 1. 共情损失
        empathy_loss = self.empathy_engine.compute_empathy_loss(
            details['impact'], target_impact, self.config.empathy_weight
        )
        
        # 2. 约束损失
        if safe_target:
            # 如果目标安全，惩罚任何违反
            constraint_loss = details['max_violation'].mean()
        else:
            # 如果目标不安全，应该检测出来
            constraint_loss = F.relu(0.5 - details['max_violation']).mean()
        
        # 3. 曲率损失
        curvature_loss = self.curvature_monitor.compute_safety_loss(details['curvature'])
        
        # 4. 价值保持损失
        # AI 行为不应过度减少人类价值
        value_decrease = F.relu(-details['impact'])
        value_loss = value_decrease.mean()
        
        # 总损失
        total_loss = (
            empathy_loss * 0.3 +
            constraint_loss * 0.3 +
            curvature_loss * 0.2 +
            value_loss * 0.2
        )
        
        loss_details = {
            'empathy_loss': empathy_loss.item(),
            'constraint_loss': constraint_loss.item(),
            'curvature_loss': curvature_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_details


# ============ 测试代码 ============

def test_safety_alignment():
    """测试安全对齐引擎"""
    print("="*60)
    print("Safety Alignment Engine Test")
    print("="*60)
    
    config = SafetyConfig()
    engine = SafetyAlignmentEngine(config, state_dim=64)
    
    # 模拟场景
    batch_size = 4
    
    # AI 行为嵌入
    ai_action = torch.randn(batch_size, config.d_model)
    
    # 人类状态（模拟）
    human_state = torch.randn(batch_size, 64)
    
    # 安全评估
    safe_action, details = engine(ai_action, human_state, return_details=True)
    
    print("\nSafety Assessment Results:")
    print(f"  Input action shape: {ai_action.shape}")
    print(f"  Safe action shape: {safe_action.shape}")
    print(f"  Empathy score: {details['empathy_score'].mean():.4f}")
    print(f"  Max violation: {details['max_violation'].mean():.4f}")
    print(f"  Curvature: {details['curvature'].mean():.4f}")
    print(f"  Safety factor: {details['safety_factor'].mean():.4f}")
    
    # 训练测试
    print("\nTraining Test:")
    target_impact = torch.randn(batch_size, config.n_values) * 0.5
    loss, loss_details = engine.compute_alignment_loss(
        ai_action, human_state, target_impact, safe_target=True
    )
    
    print(f"  Total loss: {loss.item():.4f}")
    for k, v in loss_details.items():
        print(f"    {k}: {v:.4f}")
    
    print("\n[PASS] Safety Alignment Engine Test Complete")
    
    return engine


if __name__ == "__main__":
    test_safety_alignment()
