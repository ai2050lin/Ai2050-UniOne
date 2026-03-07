# 三元组定向因果多 seed 稳定性报告

- 运行数: 5
- Seeds: 101, 202, 303, 404, 505

## 核心稳定性
- triplet_minimal_records: mean=3.0000 std=0.0000
- triplet_counterfactual_records: mean=4.0000 std=0.0000
- global_mean_causal_margin_seq_logprob: mean=0.009158 std=0.010966
- global_positive_causal_margin_ratio: mean=0.333333 std=0.235702

## 解释
- 若 seq_logprob 边际均值 > 0 且方差较小，说明三元组因果信号具跨 seed 稳定性。
