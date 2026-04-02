# stage428_deepseek7b_pronoun_head_group_stability

## 实验设置
- 时间戳: 2026-04-02T05:34:06Z
- 是否使用 CUDA: True
- 批大小: 1
- 对 stage427 找到的 DeepSeek7B 六头代词回路做精确穷举扫描
- 统计对象: 所有非空头组子集、Shapley 贡献、留一剔除必要性、两两协同

## 核心结果
- 最强搜索子集: ['H:2:0', 'H:2:10', 'H:2:3', 'H:2:8', 'H:3:1', 'H:3:27']
- 最强搜索 utility: +0.4038
- 最强全量复核子集: ['H:2:0', 'H:2:10', 'H:2:3', 'H:2:8', 'H:3:1', 'H:3:27']
- 最强全量复核 target_prob_delta: -0.4371

## 稳定核心排序
- H:3:1: layer=3, head=1, shapley_utility=0.1023, loo_full_drop_loss=0.1101
- H:2:0: layer=2, head=0, shapley_utility=0.0964, loo_full_drop_loss=0.1305
- H:2:3: layer=2, head=3, shapley_utility=0.0917, loo_full_drop_loss=0.1042
- H:2:8: layer=2, head=8, shapley_utility=0.0396, loo_full_drop_loss=0.0464
- H:3:27: layer=3, head=27, shapley_utility=0.0389, loo_full_drop_loss=0.0419
- H:2:10: layer=2, head=10, shapley_utility=0.0349, loo_full_drop_loss=0.0412

## 最优阈值子集
- ratio=0.50: subset=['H:2:0', 'H:2:3', 'H:3:1'], size=3, full_target_prob_delta=-0.2960
- ratio=0.70: subset=['H:2:0', 'H:2:3', 'H:3:1', 'H:3:27'], size=4, full_target_prob_delta=-0.3555
- ratio=0.80: subset=['H:2:0', 'H:2:3', 'H:3:1', 'H:3:27'], size=4, full_target_prob_delta=-0.3555
- ratio=0.90: subset=['H:2:0', 'H:2:10', 'H:2:3', 'H:2:8', 'H:3:1'], size=5, full_target_prob_delta=-0.3953