# stage426_pronoun_minimal_causal_mechanism

## 实验设置
- 时间戳: 2026-03-30T05:46:40Z
- 是否使用 CUDA: True
- 批大小: 1
- 任务: 围绕代词句法角色，比较顶部层 MLP 整层消融、顶部层 attention 整层消融、前缀神经元子集消融。

## 模型 qwen3
- 模型名: Qwen/Qwen3-4B
- 代词顶部层: [35, 34]
- 基线代词概率: 0.2578
- MLP 顶部层消融 target_prob_delta: -0.1748
- attention 顶部层消融 target_prob_delta: +0.0506
- 最小前缀子集: 未达到 50% 整层效果

## 模型 deepseek7b
- 模型名: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- 代词顶部层: [2, 1]
- 基线代词概率: 0.6382
- MLP 顶部层消融 target_prob_delta: -0.4263
- attention 顶部层消融 target_prob_delta: -0.5895
- 最小前缀子集: 未达到 50% 整层效果
