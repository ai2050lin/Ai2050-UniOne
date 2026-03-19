# 三模型连续密度前沿分析

## 共享结果
- 共享最稳定轴: logic
- 共享最广重排维度: style
- 平均 25% 汇合点: 0.3033
- 平均 50% 汇合点: 0.6833
- 平均前沿拐点: 0.2333

## 模型摘要
- DeepSeek-7B: total=530432, cross_merge25=0.3600, cross_merge50=0.7000, cross_min_jaccard=0.0670@0.01
  style: auc=0.1293, knee=0.2500, full_coverage=0.0400, pair_cos=0.2074, specific_ratio=0.6970
  logic: auc=0.1446, knee=0.2300, full_coverage=0.0200, pair_cos=0.4722, specific_ratio=0.3665
  syntax: auc=0.1303, knee=0.2200, full_coverage=0.1300, pair_cos=0.2002, specific_ratio=0.3346
- GLM-4-9B: total=547840, cross_merge25=0.1100, cross_merge50=0.6000, cross_min_jaccard=0.1547@0.01
  style: auc=0.1717, knee=0.2300, full_coverage=0.5000, pair_cos=0.2139, specific_ratio=0.7073
  logic: auc=0.1628, knee=0.2400, full_coverage=0.0900, pair_cos=0.3865, specific_ratio=0.3934
  syntax: auc=0.1481, knee=0.2300, full_coverage=0.1100, pair_cos=0.1258, specific_ratio=0.2792
- Qwen3-4B: total=350208, cross_merge25=0.4400, cross_merge50=0.7500, cross_min_jaccard=0.0220@0.01
  style: auc=0.1313, knee=0.2400, full_coverage=0.1800, pair_cos=0.2478, specific_ratio=0.6307
  logic: auc=0.1412, knee=0.2300, full_coverage=0.0200, pair_cos=0.5407, specific_ratio=0.4086
  syntax: auc=0.1052, knee=0.2300, full_coverage=0.0400, pair_cos=0.1490, specific_ratio=0.3467
