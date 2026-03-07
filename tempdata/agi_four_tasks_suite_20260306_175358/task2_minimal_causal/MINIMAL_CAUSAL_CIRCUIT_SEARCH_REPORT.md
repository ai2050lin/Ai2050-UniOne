# 最小因果回路搜索报告

## 全局指标
- min_subset_size_mean: 48.0000
- fidelity_mean: 0.3754
- intervention_drop_mean: 0.4005
- reproducibility_jaccard_mean: 0.9573

## 目标概念摘要
- apple: size_mean=48.00, fidelity_mean=0.4003, drop_mean=0.4005, stability_jaccard=0.9714
- banana: size_mean=48.00, fidelity_mean=0.3004, drop_mean=0.4000, stability_jaccard=1.0000
- cat: size_mean=48.00, fidelity_mean=0.4001, drop_mean=0.4001, stability_jaccard=1.0000
- dog: size_mean=48.00, fidelity_mean=0.4009, drop_mean=0.4016, stability_jaccard=0.8577

## 结论
- 若 intervention_drop 与 reproducibility 同时较高，可支持“可干预 + 可复现”的最小回路证据。
