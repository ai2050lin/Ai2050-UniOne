# stage427_pronoun_mixed_circuit_search

## 实验设置
- 时间戳: 2026-03-30T06:03:54Z
- 是否使用 CUDA: True
- 批大小: 1
- 算法: 单元快筛 + 混合贪心搜索 + 反向剪枝
- 候选池: pronoun 顶部层 attention heads + pronoun 高分 MLP neurons

## 模型 qwen3
- 模型名: Qwen/Qwen3-4B
- head 搜索层: [35, 34, 21]
- 最终子集大小: 5
- 最终子集组成: {'attention_head': 4, 'mlp_neuron': 1}
- 搜索阶段 target_drop: +0.2418
- 搜索阶段 utility: +0.2402
- 全量复核 target_prob_delta: -0.2418
- 相对 stage426 最强模块恢复比例: 1.3837051422280295

## 模型 deepseek7b
- 模型名: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- head 搜索层: [2, 1, 3]
- 最终子集大小: 6
- 最终子集组成: {'attention_head': 6}
- 搜索阶段 target_drop: +0.4353
- 搜索阶段 utility: +0.4054
- 全量复核 target_prob_delta: -0.4353
- 相对 stage426 最强模块恢复比例: 0.7384604765931013
