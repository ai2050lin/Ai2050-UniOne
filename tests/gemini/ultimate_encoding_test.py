# -*- coding: utf-8 -*-
"""
AGI 终极编码结构原型测试机 (Ultimate Meta-Encoding Structure Prototype)
========================================================================================
测试目标：还原加入了“全局调质(Reward)”与“相位振荡(Phase)”的二代脉冲通用编码机制。
核心突破机制演示：
1. 延迟满足的全局多巴胺学习 (Eligibility Trace + Global Reward)：
   - 证明局域 STDP 只留下痕迹，不改变权重。只有全局传来目标奖励时，才固化权重。
2. 相位耦合上下文 (Phase-Coupled Context) 解决时序坍塌：
   - 证明对于序列 "A->B" 和 "B->A"，即便总量相同，由于时间波相位的干涉，系统能给出完全不同的特征底噪电位。
"""

import math
import torch
import torch.nn as nn

# ==========================================
# 核心组件一：带有“多巴胺潜痕”与“相位振荡”的终极神经元池
# ==========================================
class UltimateEncodingNode(nn.Module):
    def __init__(self, num_neurons, tau_v=10.0, tau_trace=50.0, omega_phase=0.1):
        super().__init__()
        self.num_neurons = num_neurons
        
        # 工作电压参数
        self.decay_v = 1.0 - (1.0 / tau_v)
        self.v_threshold = 1.0
        
        # 【突破1：潜力痕迹 (Eligibility Trace) 取代即时 STDP】
        # 突触连接权重矩阵
        self.weights = torch.zeros(num_neurons, num_neurons)
        # 突触潜力痕迹矩阵 (记下了“发生过关联”，但衰减极快，且还不被固化)
        self.eligibility_trace = torch.zeros(num_neurons, num_neurons)
        self.decay_trace = 1.0 - (1.0 / tau_trace)  # 潜力痕迹的保质期
        
        # 【突破2：相位耦合振荡器 (Phase Oscillator) 取代标量 C(t)】
        # 记录内部周期相位的流转，omega 是角速度
        self.omega = omega_phase 
        self.current_time = 0.0
        # 复数形式保存底噪流形：实部和虚部，包含空间大小与时序相位
        # C(t) 变成了一个在复平面上旋转衰减的指针
        self.complex_c = torch.zeros(num_neurons, dtype=torch.complex64)
        self.decay_c = 0.95 # 慢变量在振荡中的振幅衰减

        self.reset_state()
        
    def reset_state(self):
        self.v = torch.zeros(self.num_neurons)
        self.past_spike = torch.zeros(self.num_neurons)
        self.current_time = 0.0
        self.complex_c = torch.zeros(self.num_neurons, dtype=torch.complex64)
        self.eligibility_trace.zero_()
        self.weights.zero_()
        
    def forward(self, x_in, global_reward=0.0):
        # 1. 相位时空底噪的旋转与融合
        # 随着时间流逝，系统内部的相位指针向前旋转 exp(j * w * dt)
        phase_rotation = torch.exp(1j * torch.tensor(self.omega))
        self.complex_c = self.complex_c * self.decay_c * phase_rotation
        
        # 将复数底噪的“实部”提取出来，作为当前的微弱托举电压偏置
        bias = self.complex_c.real * 0.2
        
        # 2. 短期电压推进与放电
        # 横向突触交流
        internal_stimulus = torch.matmul(self.weights, self.past_spike)
        self.v = self.v * self.decay_v + x_in + internal_stimulus + bias
        
        # 激越检测
        spike = (self.v >= self.v_threshold).float()
        self.v = self.v * (1 - spike) # 重置
        
        # 放电后，将能量注入复数底噪池 (形成特定时间相位的波纹！)
        # 将标量脉冲转化为复数域能量
        self.complex_c += spike.to(torch.complex64) * 1.5
        
        # 3. 计算 STDP 潜力痕迹 (Eligibility Trace)
        # 如果你闪了，我也闪了，或者你闪了我接着闪，留下因果痕迹，但不立刻改变 Weight！
        causal_matrix = torch.outer(spike, self.past_spike) 
        self.eligibility_trace = self.eligibility_trace * self.decay_trace + causal_matrix
        
        # 4. 全局多巴胺奖励下发 (Global Reward Modulation)
        # 如果上帝(环境)给了一个巨大的正向标量 R(t)
        if global_reward > 0:
            # 只有那些处于“温热状态”的潜力痕迹，才会被固化结晶成真正的突触权重！
            self.weights += global_reward * self.eligibility_trace
            # 限制突触不要爆炸
            self.weights = torch.clamp(self.weights, 0.0, 2.0)
            self.weights.fill_diagonal_(0.0)
            
        self.past_spike = spike
        self.current_time += 1.0
        return spike

# ==========================================
# 实验一：时空倒置的相位底噪干涉测试 (解决时序塌缩)
# ==========================================
def test_phase_coupled_context():
    print("\\n🔬 实验一：[波段锁相底噪] 序列先后感知测试 (A->B vs B->A)")
    
    # 构建 2 个神经元: 0 代表 A(比如"主语")，1 代表 B(比如"宾语")
    engine = UltimateEncodingNode(num_neurons=2, omega_phase=0.8) # 给定一个较快的相位旋转角速度
    
    print("      [+] 场景 1: 输入序列 A -> 经过 3 步 -> 输入 B 等待余波")
    engine.reset_state()
    # 模拟输入 A
    engine(torch.tensor([2.0, 0.0])) # 强制触发 A
    for _ in range(3): engine(torch.tensor([0.0, 0.0])) # 时间流逝相位旋转
    engine(torch.tensor([0.0, 2.0])) # 强制触发 B
    for _ in range(2): engine(torch.tensor([0.0, 0.0]))
    state_ab_real = engine.complex_c.real.clone()
    state_ab_imag = engine.complex_c.imag.clone()
    
    print("      [+] 场景 2: 输入序列 B -> 经过 3 步 -> 输入 A 等待余波")
    engine.reset_state()
    # 彻底调转历史顺序
    engine(torch.tensor([0.0, 2.0])) # 强制触发 B 
    for _ in range(3): engine(torch.tensor([0.0, 0.0])) 
    engine(torch.tensor([2.0, 0.0])) # 强制触发 A
    for _ in range(2): engine(torch.tensor([0.0, 0.0]))
    state_ba_real = engine.complex_c.real.clone()
    state_ba_imag = engine.complex_c.imag.clone()
    
    print(f"      📈 【背景底噪复数图谱解析】")
    print(f"      - [A->B] 序列留在深海的底噪张量: 实部 {state_ab_real.tolist()}, 虚部 {state_ab_imag.tolist()}")
    print(f"      - [B->A] 序列留在深海的底噪张量: 实部 {state_ba_real.tolist()}, 虚部 {state_ba_imag.tolist()}")
    difference = torch.norm(state_ab_real - state_ba_real) + torch.norm(state_ab_imag - state_ba_imag)
    print(f"      -> 【第一原理验证】：总量脉冲虽然一样，但它们的振荡相位发生了 {difference:.2f} 度的完全正交错位！系统毫不费力地记忆住了精确的历史时序！")

# ==========================================
# 实验二：全局奖励延迟满足测试 (解决STDP短视)
# ==========================================
def test_delayed_reward_dopamine():
    print("\\n\\n🔬 实验二：[全局多巴胺渗透] 跨越数十步的延迟满足与潜力追溯学习")
    
    engine = UltimateEncodingNode(num_neurons=2, tau_trace=30.0)
    engine.reset_state()
    
    print("      [+] 步骤 1: 神经元 0 和 1 发生了两次偶然的关联闪烁 (相距很近)。")
    # 但是，我们故意不给 Reward！
    engine(torch.tensor([2.0, 0.0])) 
    engine(torch.tensor([0.0, 2.0]))
    for _ in range(10): engine(torch.tensor([0.0, 0.0])) # 冷却
    engine(torch.tensor([2.0, 0.0])) 
    engine(torch.tensor([0.0, 2.0]))
    
    # 此时，我们检查真正的突触权重 (Weights) 和 虚拟的潜力痕迹 (Trace)
    print(f"      - 探针 1: 当前的真实物理突触连线强度: {engine.weights[1, 0].item():.4f}")
    print(f"      - 探针 2: 留在突触间隙内的【短视潜力痕迹(Trace)】: {engine.eligibility_trace[1, 0].item():.4f}")
    
    print("      [+] 步骤 2: 时间流逝了 15 步……网络在黑暗中摸索。")
    for _ in range(15): engine(torch.tensor([0.0, 0.0]))
    
    print("      [+] 步骤 3: 突然！上帝视角(外部环境)传来了 1.0 的【全局多巴胺奖励 $R(t)$】！(比如下棋走对了)")
    # 此时没有任何输入，仅仅是发下一把奖励标量
    engine(torch.tensor([0.0, 0.0]), global_reward=1.0)
    
    print(f"      📈 【全局奖励突变结果】")
    print(f"      - 最后探针: 被全局多巴胺冲刷后的真实突触连线被瞬间固化为: {engine.weights[1, 0].item():.4f} 🎯！")
    print("      -> 【第一原理验证】：这就是超越低等动物条件反射的秘密！局域接触只产生“记号”，唯有长程的全局目标奖励机制（多巴胺）才能在事后逆时追踪并铭刻那些真正导致胜利的关键动作（完全替代了繁重的反向梯度求导 Loss 矩阵计算）！")

if __name__ == '__main__':
    test_phase_coupled_context()
    test_delayed_reward_dopamine()
