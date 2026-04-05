# stage479_apple_switch_mixed_circuit_search

## 实验设置
- 时间戳: 2026-04-03T00:02:25Z
- 是否使用 CUDA: False
- 批大小: 2
- 目标: 在苹果切换任务中搜索 attention head + 神经元的混合回路
- 头候选: 敏感层全部注意力头
- 神经元候选: stage478 敏感层差分神经元 + stage446 全局切换偏置神经元
- 算法: 单元快筛 + 混合贪心搜索 + 反向剪枝

## 模型 qwen3
- 模型名: Qwen/Qwen3-4B
- 敏感层: L5
- 原始候选数: 56（头 32，神经元 24）
- shortlist 数: 14
- 最终子集大小: 3
- 最终子集组成: {'attention_head': 3}
- 最终子集: H:5:2, H:5:29, H:5:9
- search_drop: +0.1370
- heldout_drop: +0.0000
- control_abs_shift: +0.0000
- utility: +0.0685
- 相对 stage478 utility 倍数: 30.0714

## 模型 deepseek7b
- 模型名: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- 敏感层: L2
- 原始候选数: 52（头 28，神经元 24）
- shortlist 数: 14
- 最终子集大小: 6
- 最终子集组成: {'mlp_neuron': 1, 'attention_head': 5}
- 最终子集: N:2:16785, H:2:2, H:2:22, H:2:10, H:2:26, H:2:5
- search_drop: +0.1545
- heldout_drop: +0.1185
- control_abs_shift: +0.0059
- utility: +0.1336
- 相对 stage478 utility 倍数: 1.6045
