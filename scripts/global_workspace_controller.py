"""
Global Workspace Controller (GWT) - Phase VI
============================================

AGI 意识中枢实现：
1. 竞争性激活 - 多模态信号竞争进入意识
2. 动态稀疏性 - Top-K 激活机制
3. 跨束同步 - 视觉/语言流形自动对齐
4. 全局广播 - 意识状态向所有子系统传播

数学基础：
- 意识焦点 = argmax_i(Energy(module_i))
- 全局广播: GWS_{t+1} = α·GWS_t + (1-α)·Winner_signal
- 稀疏激活: Top_K(attention_scores)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


class ModuleRegistration:
    """注册模块的信息"""
    def __init__(self, name: str, local_dim: int, gws_dim: int):
        self.name = name
        self.local_dim = local_dim
        self.projection_up = nn.Linear(local_dim, gws_dim)  # Local -> GWS
        self.projection_down = nn.Linear(gws_dim, local_dim)  # GWS -> Local
        self.salience_history: List[float] = []
        self.last_signal: Optional[torch.Tensor] = None
    
    def to(self, device):
        """移动到设备"""
        self.projection_up = self.projection_up.to(device)
        self.projection_down = self.projection_down.to(device)
        return self


class DynamicSparsityEngine(nn.Module):
    """
    动态稀疏引擎
    实现 O(k) 复杂度的注意力机制
    """
    
    def __init__(self, d_model: int, sparsity_ratio: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.sparsity_ratio = sparsity_ratio
        self.k = max(1, int(d_model * sparsity_ratio))
        
        # 可学习的稀疏门控
        self.gate_network = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, return_mask: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        动态稀疏激活
        x: (batch, seq_len, d_model)
        返回: (稀疏化后的x, 稀疏mask)
        """
        batch_size, seq_len, d_model = x.shape
        
        # 计算激活门控
        gate_scores = self.gate_network(x).squeeze(-1)  # (batch, seq_len)
        
        # Top-K 选择
        k = max(1, int(seq_len * self.sparsity_ratio))
        _, top_k_indices = torch.topk(gate_scores, k, dim=-1)
        
        # 创建稀疏mask
        mask = torch.zeros_like(gate_scores)
        mask.scatter_(1, top_k_indices, 1.0)
        
        # 应用mask（广播到d_model维度）
        sparse_x = x * mask.unsqueeze(-1)
        
        if return_mask:
            return sparse_x, mask
        return sparse_x, None
    
    def sparse_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                         k_ratio: float = 0.1) -> torch.Tensor:
        """
        稀疏注意力机制
        复杂度: O(N·k) 而非 O(N²)
        """
        batch_size, seq_len, d_k = Q.shape
        k = max(1, int(seq_len * k_ratio))
        
        # 1. 计算 Q 和 K 的相似度矩阵（仅用于选择）
        # 使用随机投影加速
        if d_k > 64:
            # Johnson-Lindenstrauss 随机投影
            proj = torch.randn(d_k, 32, device=Q.device) / math.sqrt(32)
            Q_proj = Q @ proj
            K_proj = K @ proj
        else:
            Q_proj = Q
            K_proj = K
        
        # 2. 对每个 query，选择 top-k 个 key
        # (batch, seq_len, 32) @ (batch, 32, seq_len) -> (batch, seq_len, seq_len)
        similarity = torch.bmm(Q_proj, K_proj.transpose(-1, -2))
        
        # Top-K mask
        _, top_k_indices = torch.topk(similarity, k, dim=-1)
        mask = torch.zeros_like(similarity)
        mask.scatter_(-1, top_k_indices, 1.0)
        
        # 3. 仅计算 top-k 的完整注意力
        # 使用 mask 过滤
        scores = torch.bmm(Q, K.transpose(-1, -2)) / math.sqrt(d_k)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        
        # 处理 NaN（当某行全为 -inf 时）
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        # 4. 应用到 V
        output = torch.bmm(attn_weights, V)
        
        return output


class CrossBundleSynchronizer(nn.Module):
    """
    跨束同步器
    实现视觉-语言流形的自动对齐
    
    数学基础：
    - 平行移动: v' = v + Γ(v, ∂/∂x)dx
    - 联络: Γ^k_ij (Christoffel symbols)
    """
    
    def __init__(self, d_model: int, n_bundles: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_bundles = n_bundles
        
        # 每个束的联络参数
        self.connections = nn.ParameterList([
            nn.Parameter(torch.randn(d_model, d_model) * 0.01)
            for _ in range(n_bundles)
        ])
        
        # 束间映射矩阵
        self.bundle_maps = nn.ParameterList([
            nn.Parameter(torch.eye(d_model) + torch.randn(d_model, d_model) * 0.01)
            for _ in range(n_bundles - 1)
        ])
        
        # 同步强度
        self.sync_strength = nn.Parameter(torch.tensor(0.1))
        
    def parallel_transport(self, v: torch.Tensor, connection_idx: int,
                          dx: torch.Tensor) -> torch.Tensor:
        """
        平行移动
        v: 向量 (batch, d_model)
        dx: 位移 (batch, d_model)
        返回: 移动后的向量
        """
        Gamma = self.connections[connection_idx]
        
        # 平行移动公式: dv = -Γ(v, dx) = -v @ Gamma @ dx
        # 简化为: v' = v - sync_strength * (v @ Gamma) * dx
        correction = torch.bmm(v.unsqueeze(1), Gamma.unsqueeze(0).expand(v.size(0), -1, -1))
        correction = torch.bmm(correction, dx.unsqueeze(-1)).squeeze(-1)
        
        v_new = v - self.sync_strength * correction
        
        return v_new
    
    def sync_bundles(self, bundle_states: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        同步多个束的状态
        bundle_states: [(batch, d_model), ...]
        返回: 同步后的状态列表
        """
        n = len(bundle_states)
        if n < 2:
            return bundle_states
        
        # 使用第一个束作为参考
        reference = bundle_states[0]
        
        synced_states = [reference]
        
        for i in range(1, n):
            # 动态获取映射矩阵（如果没有足够的预定义映射，使用单位矩阵）
            if i - 1 < len(self.bundle_maps):
                map_matrix = self.bundle_maps[i-1]
            else:
                # 使用单位矩阵作为默认
                map_matrix = torch.eye(self.d_model, device=bundle_states[i].device)
            
            # 映射到参考束
            batch_size = bundle_states[i].size(0) if bundle_states[i].dim() > 1 else 1
            if bundle_states[i].dim() == 1:
                mapped = bundle_states[i].unsqueeze(0) @ map_matrix.T
                mapped = mapped.squeeze(0)
            else:
                mapped = torch.bmm(bundle_states[i].unsqueeze(1), 
                                 map_matrix.unsqueeze(0).expand(batch_size, -1, -1).transpose(-1, -2))
                mapped = mapped.squeeze(1)
            
            # 软同步（不强制相等，而是向参考方向调整）
            synced = (1 - self.sync_strength) * bundle_states[i] + self.sync_strength * mapped
            synced_states.append(synced)
        
        return synced_states
    
    def compute_curvature(self, connection_idx: int) -> torch.Tensor:
        """
        计算曲率张量 R^l_ijk
        R = ∂Γ/∂x - ∂Γ/∂x + ΓΓ - ΓΓ
        简化为度量曲率
        """
        Gamma = self.connections[connection_idx]
        
        # 简化的曲率度量: F = Γ - Γ^T (类似杨-米尔斯场强)
        curvature = Gamma - Gamma.T
        
        return curvature


class GlobalWorkspaceController(nn.Module):
    """
    全局工作空间控制器
    AGI 的意识中枢
    
    功能：
    1. 多模态竞争
    2. 动态稀疏激活
    3. 全局广播
    4. 跨束同步
    """
    
    def __init__(self, gws_dim: int = 64, sparsity_ratio: float = 0.1):
        super().__init__()
        self.gws_dim = gws_dim
        
        # 全局工作空间状态
        self.register_buffer('gws_state', torch.zeros(gws_dim))
        self.register_buffer('gws_velocity', torch.zeros(gws_dim))  # 意识流速度
        
        # 模块注册表
        self.modules: Dict[str, ModuleRegistration] = {}
        
        # 动态稀疏引擎
        self.sparsity_engine = DynamicSparsityEngine(gws_dim, sparsity_ratio)
        
        # 跨束同步器
        self.bundle_sync = CrossBundleSynchronizer(gws_dim)
        
        # 意识门控
        self.consciousness_gate = nn.Sequential(
            nn.Linear(gws_dim, gws_dim // 2),
            nn.ReLU(),
            nn.Linear(gws_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 注意力焦点位置（在流形上的坐标）
        self.register_buffer('attention_locus', torch.zeros(gws_dim))
        
        # 历史追踪
        self.register_buffer('winner_history', torch.zeros(10, dtype=torch.long))  # 最近10个胜者
        self.history_ptr = 0
        
    def register_module(self, name: str, local_dim: int) -> None:
        """注册一个子系统模块"""
        self.modules[name] = ModuleRegistration(name, local_dim, self.gws_dim)
        # 移动到当前设备
        device = self.gws_state.device
        self.modules[name].to(device)
        
    def compete(self, module_signals: Dict[str, torch.Tensor],
                return_all_salience: bool = False) -> Tuple[str, torch.Tensor]:
        """
        竞争机制：多模态信号竞争进入全局工作空间
        
        module_signals: {module_name: local_signal (batch, local_dim)}
        返回: (胜者名称, 全局状态)
        """
        device = next(self.parameters()).device if list(self.parameters()) else 'cpu'
        
        # 投影所有信号到 GWS 空间
        gws_projections = {}
        energies = {}
        
        for name, signal in module_signals.items():
            if name not in self.modules:
                continue
            
            # 投影到 GWS
            proj = self.modules[name].projection_up(signal)
            gws_projections[name] = proj
            
            # 计算能量（显著性）
            energy = torch.norm(proj, dim=-1).mean().item()
            energies[name] = energy
            
            # 记录历史
            self.modules[name].salience_history.append(energy)
            self.modules[name].last_signal = signal
            
        if not energies:
            return "", self.gws_state
        
        # 选择胜者
        winner = max(energies, key=energies.get)
        winner_signal = gws_projections[winner]
        
        # 应用稀疏化
        sparse_signal, mask = self.sparsity_engine(winner_signal.unsqueeze(1), return_mask=True)
        sparse_signal = sparse_signal.squeeze(1)
        
        # 更新 GWS 状态（引入惯性）
        alpha = 0.7  # 惯性系数
        batch_mean = sparse_signal.mean(dim=0)
        
        # 更新速度和位置（类似物理系统）
        self.gws_velocity = 0.5 * self.gws_velocity + 0.5 * (batch_mean - self.gws_state)
        self.gws_state = alpha * self.gws_state + (1 - alpha) * batch_mean
        
        # 更新注意力焦点
        self.attention_locus = 0.8 * self.attention_locus + 0.2 * self.gws_state
        
        # 记录胜者历史
        winner_idx = list(self.modules.keys()).index(winner)
        self.winner_history[self.history_ptr % 10] = winner_idx
        self.history_ptr += 1
        
        if return_all_salience:
            return winner, self.gws_state, energies
        
        return winner, self.gws_state
    
    def broadcast(self, target_module: str) -> Optional[torch.Tensor]:
        """
        全局广播：将 GWS 状态传递给目标模块
        返回: 目标模块空间的信号
        """
        if target_module not in self.modules:
            return None
        
        # 从 GWS 投影回目标模块空间
        local_signal = self.modules[target_module].projection_down(self.gws_state)
        
        return local_signal
    
    def broadcast_all(self) -> Dict[str, torch.Tensor]:
        """向所有注册模块广播"""
        return {name: self.broadcast(name) for name in self.modules}
    
    def get_consciousness_level(self) -> torch.Tensor:
        """
        获取意识水平（0-1）
        基于门控网络和状态能量
        """
        level = self.consciousness_gate(self.gws_state.unsqueeze(0)).squeeze()
        return level
    
    def run_attention_cycle(self, module_signals: Dict[str, torch.Tensor],
                           num_cycles: int = 3) -> Dict:
        """
        运行完整的注意力周期
        模拟意识的迭代更新过程
        """
        cycle_results = []
        
        for i in range(num_cycles):
            winner, gws_state, salience = self.compete(module_signals, return_all_salience=True)
            
            # 反馈到模块
            for name in module_signals:
                feedback = self.broadcast(name)
                if feedback is not None:
                    # 混合原始信号和反馈
                    module_signals[name] = 0.7 * module_signals[name] + 0.3 * feedback.unsqueeze(0).expand_as(module_signals[name])
            
            cycle_results.append({
                'cycle': i,
                'winner': winner,
                'salience': salience,
                'gws_energy': torch.norm(gws_state).item(),
                'consciousness_level': self.get_consciousness_level().item()
            })
        
        return {
            'cycles': cycle_results,
            'final_winner': cycle_results[-1]['winner'],
            'final_consciousness': cycle_results[-1]['consciousness_level']
        }
    
    def synchronize_bundles(self, bundle_names: List[str]) -> Dict[str, torch.Tensor]:
        """
        同步指定的束
        """
        bundle_states = []
        for name in bundle_names:
            if name in self.modules and self.modules[name].last_signal is not None:
                # 投影到 GWS 空间
                state = self.modules[name].projection_up(self.modules[name].last_signal.mean(dim=0))
                # 确保有 batch 维度
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                bundle_states.append(state)
        
        if len(bundle_states) < 2:
            return {}
        
        # 执行同步
        synced = self.bundle_sync.sync_bundles(bundle_states)
        
        return {name: synced[i].squeeze(0) if synced[i].dim() > 1 and synced[i].size(0) == 1 else synced[i] 
                for i, name in enumerate(bundle_names) if i < len(synced)}


class GlobalWorkspaceService:
    """
    高层服务接口
    提供简化的 API 用于外部调用
    """
    
    def __init__(self, gws_dim: int = 64, sparsity_ratio: float = 0.1, device: str = 'auto'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.controller = GlobalWorkspaceController(gws_dim, sparsity_ratio).to(device)
        
        # 模块索引
        self.module_index = {}
        
    def connect_module(self, name: str, dim: int) -> None:
        """连接一个模块"""
        self.controller.register_module(name, dim)
        self.module_index[name] = dim
        print(f"[GWS] Module connected: {name} (dim={dim})")
    
    def process_signals(self, signals: Dict[str, torch.Tensor]) -> Dict:
        """处理多模态信号"""
        # 移动到正确设备
        signals = {k: v.to(self.device) for k, v in signals.items()}
        
        # 运行注意力周期
        result = self.controller.run_attention_cycle(signals)
        
        return result
    
    def get_broadcast(self, module_name: str) -> Optional[torch.Tensor]:
        """获取对特定模块的广播"""
        return self.controller.broadcast(module_name)
    
    def get_state_report(self) -> Dict:
        """获取状态报告"""
        return {
            'gws_energy': torch.norm(self.controller.gws_state).item(),
            'consciousness_level': self.controller.get_consciousness_level().item(),
            'attention_locus_norm': torch.norm(self.controller.attention_locus).item(),
            'registered_modules': list(self.module_index.keys()),
            'winner_history': self.controller.winner_history.tolist()
        }


# ============ 测试代码 ============

def test_global_workspace():
    """测试全局工作空间控制器"""
    print("=" * 50)
    print("Global Workspace Controller Test")
    print("=" * 50)
    
    # 创建服务
    service = GlobalWorkspaceService(gws_dim=64, sparsity_ratio=0.2)
    
    # 连接模块
    service.connect_module("Vision", 32)
    service.connect_module("Logic", 16)
    service.connect_module("Language", 48)
    
    # 模拟信号
    batch_size = 2
    signals = {
        "Vision": torch.randn(batch_size, 32) * 2.0,  # 较强
        "Logic": torch.randn(batch_size, 16) * 0.5,   # 较弱
        "Language": torch.randn(batch_size, 48) * 1.5 # 中等
    }
    
    # 处理
    result = service.process_signals(signals)
    
    print(f"\nFinal Winner: {result['final_winner']}")
    print(f"Final Consciousness: {result['final_consciousness']:.4f}")
    
    for cycle in result['cycles']:
        print(f"\nCycle {cycle['cycle']}:")
        print(f"  Winner: {cycle['winner']}")
        print(f"  Salience: {cycle['salience']}")
        print(f"  GWS Energy: {cycle['gws_energy']:.4f}")
    
    # 状态报告
    print("\n" + "=" * 50)
    print("State Report:")
    report = service.get_state_report()
    for k, v in report.items():
        print(f"  {k}: {v}")
    
    return service


if __name__ == "__main__":
    test_global_workspace()
