# 三模型连续密度前沿分析

## 共享结果
- 共享最稳定轴: logic
- 共享最广重排维度: mixed
- 平均 25% 汇合点: 0.3967
- 平均 50% 汇合点: 0.7333
- 平均前沿拐点: 0.2433

## 模型摘要
- DeepSeek-7B: total=530432, cross_merge25=0.4400, cross_merge50=0.7000, cross_min_jaccard=0.0051@0.01
  style: auc=0.1463, knee=0.2400, full_coverage=0.2300, pair_cos=0.5473, specific_ratio=0.4269
  logic: auc=0.1556, knee=0.2500, full_coverage=0.3200, pair_cos=0.6008, specific_ratio=0.2882
  syntax: auc=0.1308, knee=0.2200, full_coverage=0.2000, pair_cos=0.3146, specific_ratio=0.6576
- GLM-4-9B: total=547840, cross_merge25=0.2600, cross_merge50=0.7000, cross_min_jaccard=0.2124@0.03
  style: auc=0.1868, knee=0.2500, full_coverage=0.3800, pair_cos=0.5505, specific_ratio=0.5469
  logic: auc=0.2111, knee=0.2500, full_coverage=0.7000, pair_cos=0.6818, specific_ratio=0.2625
  syntax: auc=0.1591, knee=0.2400, full_coverage=0.4400, pair_cos=0.3454, specific_ratio=0.5996
- Qwen3-4B: total=350208, cross_merge25=0.4900, cross_merge50=0.8000, cross_min_jaccard=0.0386@0.01
  style: auc=0.1407, knee=0.2500, full_coverage=0.2200, pair_cos=0.5960, specific_ratio=0.4858
  logic: auc=0.1663, knee=0.2500, full_coverage=0.6500, pair_cos=0.6791, specific_ratio=0.4529
  syntax: auc=0.1128, knee=0.2400, full_coverage=0.1500, pair_cos=0.3884, specific_ratio=0.4179
