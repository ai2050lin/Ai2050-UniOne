# 变量绑定硬验证报告

## 基线 vs 增强
- rewrite_accuracy: baseline=0.0000, enhanced=0.7208, delta=+0.7208
- role_swap_accuracy: baseline=0.0042, enhanced=0.7208, delta=+0.7167
- cross_sentence_chain_accuracy: baseline=0.0000, enhanced=0.6875, delta=+0.6875
- role_decode_accuracy: baseline=0.0135, enhanced=0.4281, delta=+0.4146

- mean_delta: +0.6349
- improved_dimension_count: 4

## 结论
- 若 mean_delta 明显为正，说明引入显式变量绑定机制可以显著缓解角色混淆与跨句退化。
