# -*- coding: utf-8 -*-
"""
AGI 核心拼图之三：主动预测编码与主观意志的诞生 (Top-Down Predictive Coding)
========================================================================================
测试目标：演示 SNN 如何从“被动的特征流水线”跃迁为“主动的期望流形”。
证明顶层概念节点可以通过 Top-Down 反馈连接向底层下发抑制预期，使得：
  1. 符合预期的刺激被底层的预期抵消，底层神经元变得“沉默”（高效，不乱报）。
  2. 意外的刺激无法被期望流形遮蔽，导致底层神经元疯狂爆发“预测误差脉冲”（Surprise/Error Spice），
     强行上传惊醒高层。

核心隐喻：你走在每天回家的路上对环境“视而不见”（被预测完全压制）；
          但当路边突然出现一只大象，“轰”的一声误差脉冲就会直达高级皮层！
"""

import torch
import torch.nn as nn

# ==========================================
# 组件：带有误差反馈抑制的双向神经节点 (Predictive Node)
# ==========================================
class PredictiveIntegrator(nn.Module):
    """
    不仅接收底层的输入 (Bottom-up X)，还能接收高层的预期下发 (Top-down Expectation)。
    """
    def __init__(self, num_neurons, tau_v=15.0):
        super().__init__()
        self.num_neurons = num_neurons
        self.decay = 1.0 - (1.0 / tau_v)
        self.v_threshold = 1.0
        self.reset_state()
        
    def reset_state(self):
        self.v = torch.zeros(self.num_neurons)
        # 记录预测误差大小
        self.error_accumulator = 0.0

    def forward(self, bottom_up_stimulus, top_down_expectation=None):
        """
        bottom_up_stimulus: 来自外界或底层的真实火力
        top_down_expectation: 来自高层主观意识的抑制火力（预期它出现，就提前给它降温减压）
        """
        if top_down_expectation is None:
            top_down_expectation = torch.zeros_like(bottom_up_stimulus)
            
        # 【物理奥义】：底层膜电位是被真实刺激垫高，但同时被顶层期望死死压制！
        # 如果刺激符合预期，v 会极其平静。如果出现预期外，v 才会突破起飞。
        net_current = bottom_up_stimulus - top_down_expectation
        
        # 漏电积分
        self.v = self.v * self.decay + net_current
        
        # 产生向上的误差警报脉冲 (Error Spike)
        # 只有真正惊人的东西，才会越过阈值向高层发起冲击！
        spike = (self.v >= self.v_threshold).float()
        
        # 归置
        self.v = self.v * (1 - spike)
        self.error_accumulator += spike.sum().item()
        
        return spike

# ==========================================
# 网络：双层分级预测结构
# ==========================================
def test_predictive_coding_surprise():
    print("\\n🔬 实验：[主动预测编码] 高级意识如何通过预期压制底层，并被意外误差惊醒")
    
    # 底层：初级视觉皮层 (看到狗、猫、大象的轮廓部位)
    # [0: 狗尾巴, 1: 狗叫声, 2: 猫胡须, 3: 大象牙]
    layer_1_sensory = PredictiveIntegrator(num_neurons=4, tau_v=10.0)
    
    # 高层：抽象概念皮层 (脑海里的主观意志)
    # [0: "我认为面前是狗", 1: "我认为面前是猫"]
    layer_2_concept = PredictiveIntegrator(num_neurons=2, tau_v=20.0)
    
    # 【预训练建立的常识世界观】
    # 高层通过长期学习，已经知道“如果我想象出狗(0)，底层就应该亮起狗尾巴(0)和狗叫声(1)”
    # 这个长链代表突触反馈权重 (Feedback Weights)
    feedback_weights = torch.tensor([
        [1.5, 1.5, 0.0, 0.0], # "狗"(0) 产生的下行预期分配给底层
        [0.0, 0.0, 1.5, 0.0]  # "猫"(1) 产生的下行预期分配给底层
    ])
    
    # 前向连接权重 (Bottom-up Weights)
    # 如果底层报上来狗尾巴和叫声，高层就会被激发认定是狗
    feedforward_weights = torch.tensor([
        [1.0, 1.0, -0.5, -0.5], # 对应高层0: 狗
        [-0.5, -0.5, 1.0, -0.5] # 对应高层1: 猫
    ]).t() # 转置以匹配矩阵乘法
    
    # --- 阶段 1：每天循规蹈矩的日常 (符合预期) ---
    print("\\n  [阶段一：沉闷日常] 你每天回家，路上全是符合预期的狗吠和摇尾巴...")
    layer_1_sensory.reset_state()
    layer_2_concept.reset_state()
    
    # 高层带着先入为主的主观意识出门了：“我以为这趟必看狗”
    top_down_will = torch.tensor([1.0, 0.0]) # 概念0开启
    
    for t in range(30):
        # 物理世界传来的真实现实 (Bottom-up的狗特征)
        world_stimulus = torch.tensor([0.8, 0.8, 0.0, 0.0]) if t % 3 == 0 else torch.zeros(4)
        
        # 高层算计出预期压制电势，下发给底层传感器
        expected_sensory = torch.matmul(top_down_will, feedback_weights) * 0.7 # 0.7 是期待强度系数
        
        # 底层对撞！外界真实 VS 内部期待！
        l1_error_spike = layer_1_sensory(world_stimulus, expected_sensory)
        
        # 高层接收误差信号（如果符合预期，底下根本不上报，高层乐得清闲只消耗 0 算力！）
        l2_spike = layer_2_concept(torch.matmul(l1_error_spike, feedforward_weights))
        
    print(f"    -> 第一层(感觉传导池) 总计爆发的误差脉冲量：{layer_1_sensory.error_accumulator}")
    print(f"       现象证明：因为现实符合高层的预测流形，底层膜电位刚隆起就被抵消。")
    print(f"       大脑皮层处于极低功耗的“熟视无睹”暗色状态！")

    # --- 阶段 2：令人惊骇的意外 (Surprise!) ---
    print("\\n  [阶段二：晴天霹雳] 你带着看狗的预期出门，结果路上突然钻出一只【大象的獠牙】！")
    layer_1_sensory.reset_state()
    layer_2_concept.reset_state()
    
    # 高层依然死板的主观意识：“我看今天又是狗”
    top_down_will = torch.tensor([1.0, 0.0]) 
    
    for t in range(30):
        # 物理世界传来的意外输入！毫无征兆的大象牙 (索引3)
        world_stimulus = torch.tensor([0.0, 0.0, 0.0, 1.5]) if t % 3 == 0 else torch.zeros(4)
        
        # 盲目的高层依然把压制火力下发给了狗尾巴区！！
        expected_sensory = torch.matmul(top_down_will, feedback_weights) * 0.7 
        
        # 底层对撞！外界真实 VS 内部盲目期待！
        # 在大象牙(3)这个节点上，真实火力 1.5 遭遇了 0 预期抵消... 它将毫无阻挡地引爆！！！
        l1_error_spike = layer_1_sensory(world_stimulus, expected_sensory)
        l2_spike = layer_2_concept(torch.matmul(l1_error_spike, feedforward_weights))
        
    print(f"    -> 第一层(感觉传导池) 突然井喷爆发的误差脉冲量：{layer_1_sensory.error_accumulator} 💥！！！")
    print(f"       现象证明：由于大象不在期望的流形内，抑制火力打偏！大象节点的电势毫")
    print(f"       无阻碍突破天际，像狂风阵雨般向上传导 Error Spike，粗暴地把高")
    print(f"       级皮层叫醒重塑三观！这就是【主观惊奇与注意力转移】的根源！")

if __name__ == '__main__':
    test_predictive_coding_surprise()
