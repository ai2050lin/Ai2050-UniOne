# stage478_apple_switch_minimal_subcircuit

## 实验设置
- 时间戳: 2026-04-02T18:05:26Z
- 是否使用 CUDA: False
- 批大小: 2
- 目标: 在苹果多义切换敏感层搜索最小神经元子回路
- 候选池: 敏感层中品牌偏置与水果偏置神经元
- 算法: 单神经元快筛 + 贪心组合搜索 + 反向剪枝

## 模型 qwen3
- 模型名: Qwen/Qwen3-4B
- 敏感层: L5
- 原始候选数: 16
- shortlist 数: 8
- 最终子集大小: 1
- 最终子集: N:5:9059
- search_drop: +0.0046
- heldout_drop: +0.0000
- control_abs_shift: +0.0000
- utility: +0.0023
- 相对 stage448 最强层消融恢复比例: 0.0833

## 模型 deepseek7b
- 模型名: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- 敏感层: L2
- 原始候选数: 16
- shortlist 数: 8
- 最终子集大小: 2
- 最终子集: N:2:16785, N:2:17269
- search_drop: +0.1099
- heldout_drop: +0.0615
- control_abs_shift: +0.0049
- utility: +0.0833
- 相对 stage448 最强层消融恢复比例: 4.1698
