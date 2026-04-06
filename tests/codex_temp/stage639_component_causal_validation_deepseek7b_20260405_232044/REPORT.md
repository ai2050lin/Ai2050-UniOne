# Stage639 组件级因果验证报告

- 时间: 2026-04-05 23:21:07
- 模型: deepseek7b
- 抽样层: [0, 7, 14, 20, 27]
- 样本数: 9

## 分能力摘要
### syntax
- baseline_pair_accuracy: 1.0000
- baseline_margin: 4.6875
- attn_margin_drop: 1.1766
- mlp_margin_drop: 1.0117
- attn_pair_accuracy_drop: 0.2000
- mlp_pair_accuracy_drop: 0.2667

### relation
- baseline_pair_accuracy: 0.6667
- baseline_margin: 5.7708
- attn_margin_drop: 1.2391
- mlp_margin_drop: 0.2163
- attn_pair_accuracy_drop: 0.0667
- mlp_pair_accuracy_drop: 0.0000

### coref
- baseline_pair_accuracy: 1.0000
- baseline_margin: 2.5000
- attn_margin_drop: 0.8410
- mlp_margin_drop: -0.6707
- attn_pair_accuracy_drop: 0.6667
- mlp_pair_accuracy_drop: 0.4667

## 单样本摘要
### syntax / subject_verb_number
- baseline_margin: 7.9688
- best_attn_layer: 0
- best_attn_margin_drop: 11.2188
- best_mlp_layer: 0
- best_mlp_margin_drop: 7.9277

### syntax / distance_agreement
- baseline_margin: 2.5000
- best_attn_layer: 0
- best_attn_margin_drop: 2.8057
- best_mlp_layer: 0
- best_mlp_margin_drop: 2.6088

### syntax / collective_local_noun
- baseline_margin: 3.5938
- best_attn_layer: 0
- best_attn_margin_drop: 4.1094
- best_mlp_layer: 0
- best_mlp_margin_drop: 4.0835

### relation / capital_relation
- baseline_margin: 8.1562
- best_attn_layer: 0
- best_attn_margin_drop: 7.1572
- best_mlp_layer: 0
- best_mlp_margin_drop: 3.5625

### relation / currency_relation
- baseline_margin: 6.5313
- best_attn_layer: 0
- best_attn_margin_drop: 4.2666
- best_mlp_layer: 0
- best_mlp_margin_drop: 1.9102

### relation / inventor_relation
- baseline_margin: 2.6250
- best_attn_layer: 0
- best_attn_margin_drop: 3.8971
- best_mlp_layer: 0
- best_mlp_margin_drop: 3.5156

### coref / winner_reference
- baseline_margin: 3.0312
- best_attn_layer: 0
- best_attn_margin_drop: 3.3262
- best_mlp_layer: 7
- best_mlp_margin_drop: 0.3125

### coref / help_reference
- baseline_margin: 2.6562
- best_attn_layer: 14
- best_attn_margin_drop: 1.9375
- best_mlp_layer: 7
- best_mlp_margin_drop: 1.7148

### coref / blame_reference
- baseline_margin: 1.8125
- best_attn_layer: 0
- best_attn_margin_drop: 1.7422
- best_mlp_layer: 14
- best_mlp_margin_drop: 1.6563

