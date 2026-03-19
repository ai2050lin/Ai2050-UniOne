# 三模型连续密度前沿分析

## 共享结果
- 共享最稳定轴: logic
- 共享最广重排维度: mixed
- 平均 25% 汇合点: 0.4033
- 平均 50% 汇合点: 0.7500
- 平均前沿拐点: 0.2433

## 模型摘要
- DeepSeek-7B: total=530432, cross_merge25=0.4400, cross_merge50=0.7500, cross_min_jaccard=0.0054@0.01
  style: auc=0.1471, knee=0.2400, full_coverage=0.2300, pair_cos=0.5459, specific_ratio=0.4340
  logic: auc=0.1566, knee=0.2500, full_coverage=0.3100, pair_cos=0.6048, specific_ratio=0.2859
  syntax: auc=0.1354, knee=0.2200, full_coverage=0.2000, pair_cos=0.3336, specific_ratio=0.6558
- GLM-4-9B: total=547840, cross_merge25=0.2700, cross_merge50=0.7000, cross_min_jaccard=0.2110@0.03
  style: auc=0.1874, knee=0.2500, full_coverage=0.3700, pair_cos=0.5622, specific_ratio=0.5366
  logic: auc=0.2117, knee=0.2500, full_coverage=0.7000, pair_cos=0.6839, specific_ratio=0.2641
  syntax: auc=0.1620, knee=0.2400, full_coverage=0.4600, pair_cos=0.3509, specific_ratio=0.6065
- Qwen3-4B: total=350208, cross_merge25=0.5000, cross_merge50=0.8000, cross_min_jaccard=0.0388@0.02
  style: auc=0.1418, knee=0.2500, full_coverage=0.2100, pair_cos=0.6005, specific_ratio=0.4797
  logic: auc=0.1681, knee=0.2500, full_coverage=0.6500, pair_cos=0.6882, specific_ratio=0.4538
  syntax: auc=0.1138, knee=0.2400, full_coverage=0.1600, pair_cos=0.3843, specific_ratio=0.4262
