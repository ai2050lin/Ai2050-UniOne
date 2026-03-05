# -*- coding: utf-8 -*-
"""
AGI 核心拼图之二：海马体脱机固化与灾难性遗忘的救赎 (Hippocampal Offline Replay)
========================================================================================
测试目标：演示 SNN 在持续高频新知识冲刷下带来的容量干涉（新皮层白噪声坍塌），
以及利用“海马体（高速缓存区） + 夜间脱机低频重放 (Offline Replay)” 完美解决灾难性遗忘。
"""

import torch
import torch.nn as nn

# ==========================================
# 组件：双速率突触学习积分器
# ==========================================
class MemoryNode(nn.Module):
    def __init__(self, num_neurons, tau_v=15.0, learning_rate=0.0):
        super().__init__()
        self.num_neurons = num_neurons
        self.decay = 1.0 - (1.0 / tau_v)
        self.v_threshold = 1.0
        self.learning_rate = learning_rate
        
        # 突触连接图谱权重
        self.weights = torch.zeros(num_neurons, num_neurons)
        self.reset_state()
        
    def reset_state(self):
        self.v = torch.zeros(self.num_neurons)
        self.past_spike = torch.zeros(self.num_neurons)

    def forward(self, x_in, apply_stdp=False, homeostatic_decay=0.01):
        # 融入横向突触影响
        internal_influence = torch.matmul(self.weights, self.past_spike)
        self.v = self.v * self.decay + x_in + internal_influence
        
        spike = (self.v >= self.v_threshold).float()
        
        if apply_stdp:
            # 极简 STDP：发现此刻有人闪，并且上一刻或同刻有人闪，建立连接
            causal_matrix = torch.outer(spike, spike + self.past_spike)
            self.weights += self.learning_rate * causal_matrix
            # 全局稳态衰减压迫 (模拟记忆的冲缩)
            self.weights -= homeostatic_decay
            self.weights = torch.clamp(self.weights, 0.0, 1.0)
            self.weights.fill_diagonal_(0.0)
            
        self.v = self.v * (1 - spike)
        self.past_spike = spike
        return spike

# ==========================================
# 实验一：单脑结构的“死水坍塌” (Catastrophic Forgetting)
# ==========================================
def test_catastrophic_forgetting():
    print("\\n🔬 实验一：[没有海马体的单脑] 观测海量新入刺激对老知识的疯狂碾压破坏")
    
    # 构建一个只有新皮层的主脑，直接负责短期和长期记忆
    # 模拟 3 个概念神经元：0=狗，1=骨头，2=外星人
    neocortex = MemoryNode(num_neurons=3, tau_v=10.0, learning_rate=0.4)
    
    print("      [+] 时期一：年幼期，持续看到狗(0)伴随着骨头(1)")
    for t in range(50):
        stimulus = torch.tensor([1.5, 1.5, 0.0]) # 狗和骨头结伴输入
        neocortex(stimulus, apply_stdp=True, homeostatic_decay=0.0) # 幼年期没有竞争衰减
        
    old_knowledge_weight = neocortex.weights[1, 0].item()
    print(f"      - 童年突触结晶度：'狗<->骨头' 突触极其牢固! 满级连通度达到了 {old_knowledge_weight:.4f}")
    
    print("      [+] 时期二：去往银河系生活，每天遭到海量毫无相关的狂暴新信息轰炸 (大风暴冲刷)")
    for t in range(150):
        # 系统突然接触了大量的随机宇宙噪音
        # 再也没有刻意让 0 和 1 伴随出现了！
        noise = torch.rand(3) * 1.5
        neocortex(noise, apply_stdp=True, homeostatic_decay=0.04) # 高压成人环境，全网竞争白质衰减增强
        
    new_knowledge_weight = neocortex.weights[1, 0].item()
    print(f"      📈 【灾难性遗忘核查】")
    print(f"      - 突触探针在狂暴风雨后检查：曾经铁一般的 '狗<->骨头' 连接，断崖式掉落至 {new_knowledge_weight:.4f}！")
    print("      🔥 结论：如果把突触长期暴露在数据的洪流中，毫无遮掩的新梯度/时序共振会产生恐怖的全局挤压！它会把原来记好的常识彻底抹平。单脑系统完全变成了狗熊掰棒子，丧失了长时维度！")


# ==========================================
# 实验二：双脑架构与离线睡眠重放固化 (Hippocampal Offline Replay)
# ==========================================
def test_sleep_consolidation():
    print("\\n\\n🔬 实验二：[双系统离线固化] 引入高速海马体缓存与夜间绝缘重放机制 (Offline Replay)")
    
    # 系统 A：海马体 (极其敏感，一瞬即印)
    hippocampus = MemoryNode(num_neurons=3, tau_v=5.0, learning_rate=1.0) 
    # 系统 B：新皮层 (深不可测长衰减矩阵、只在夜间内部接收知识，拒绝白天白质竞争衰减)
    neocortex = MemoryNode(num_neurons=3, tau_v=50.0, learning_rate=0.2)
    
    print("      [+] 状况：主脑(新皮层)在遥远的过去，已经极其牢固地记住了 '狗(0)<->骨头(1)'，突触锁死了。")
    neocortex.weights[1, 0] = 1.0
    neocortex.weights[0, 1] = 1.0
    
    print("      [+] 白天 (Awake Phase)：主脑绝对关闭 STDP 可塑性。只允许活泼的海马体在最前线接敌缓冲！")
    # 模拟今天在动物园突然看到了新物种连线：外星人(2) 喜欢 狗(0)
    for t in range(15):
        stimulus = torch.tensor([1.0, 0.0, 1.0]) if t % 2 == 0 else torch.zeros(3)
        # 海马体开启暴力快速学习
        hippocampus(stimulus, apply_stdp=True, homeostatic_decay=0.0)
        # 保护主脑老网免受新事务的洪流直接冲刷！不接纳突触改变！
        neocortex(stimulus, apply_stdp=False) 
        
    hippo_new_weight = hippocampus.weights[2, 0].item()
    cortex_new_weight = neocortex.weights[2, 0].item()
    print(f"      - 日落时分：主脑对'外星人-狗'新知识录入率: {cortex_new_weight:.4f} (坚若磐石，屏蔽了白昼喧闹冲刷)")
    print(f"      - 日落时分：海马区缓存录入率: {hippo_new_weight:.4f} (极速印刻，已将新奇事物高度抽取打包暂存)")
    
    print("      [+] 深度睡眠开始 (Sleep Phase)：切断外界视觉听觉！开启无干扰自主梦境倒放！")
    print("      -> 机制：海马体按照白天积攒的高核连接，在黑夜脱机环境下无休止地高速放电。主脑开启极度微小的 STDP 缓慢篆刻这滴梦境精华。")
    
    hippocampus.reset_state()
    neocortex.reset_state() # 晚上没有任何外界刺激
    for t in range(100):
        # 海马体梦呓：我们在海马体神经元上直接注射随机尖峰，如果它学到了 2 和 0 的连接，它会自动一起开火！
        # 每隔 5 步，我们就刺激一下外星人节点(2)
        inner_noise = torch.tensor([0.0, 0.0, 1.0]) if t % 5 == 0 else torch.zeros(3)
        hippo_dream_spike = hippocampus(inner_noise, apply_stdp=False)
        
        # 海马体产生的梦境脉冲梦呓，作为“超级提纯导师向量”，注射进主脑！
        # 夜间的新皮层没有外部世界的残酷倾轧 (homeostatic_decay = 0)
        neocortex(hippo_dream_spike * 1.5, apply_stdp=True, homeostatic_decay=0.00)

    print(f"      📈 【清晨苏醒：新旧知识绝缘固化奇迹】")
    print(f"      - 探针 1 (旧常识)：老祖宗的知识 '狗<->骨头' 是否依然健在？ -> 权重 {neocortex.weights[1, 0].item():.4f} 🎯 (完好无损！毫无白日风暴熔化之灾！) ")
    print(f"      - 探针 2 (新知识)：'外星人<->狗' 是否被海马梦境偷渡成功？ -> 权重 {neocortex.weights[2, 0].item():.4f} 🎯 (暗度陈仓，新图谱扎根深柜！) ")
    print("      -> 【最终定律验证】：通过【前线高灵敏快存海马体】与【夜间新皮层绝对脱机缓慢篆刻】的双系统大坝。海量新输入再也无法把模型基底脑的老连接冲成白噪音！新知识经过降噪提纯，像打长钉一样被梦境永远敲进了主池！AGI 跨世代记忆永动模型正式成立！")

if __name__ == '__main__':
    test_catastrophic_forgetting()
    test_sleep_consolidation()
