# Stage639 组件级因果验证报告

- 时间: 2026-04-05 23:20:01
- 模型: qwen3
- 抽样层: [0, 9, 18, 26, 35]
- 样本数: 9

## 分能力摘要
### syntax
- baseline_pair_accuracy: 1.0000
- baseline_margin: 5.4479
- attn_margin_drop: 1.1137
- mlp_margin_drop: 1.1547
- attn_pair_accuracy_drop: 0.1333
- mlp_pair_accuracy_drop: 0.2000

### relation
- baseline_pair_accuracy: 1.0000
- baseline_margin: 9.4479
- attn_margin_drop: 1.1604
- mlp_margin_drop: 1.9001
- attn_pair_accuracy_drop: 0.0000
- mlp_pair_accuracy_drop: 0.0667

### coref
- baseline_pair_accuracy: 0.6667
- baseline_margin: 0.6042
- attn_margin_drop: -0.3958
- mlp_margin_drop: -0.2844
- attn_pair_accuracy_drop: 0.0667
- mlp_pair_accuracy_drop: 0.0667

## 单样本摘要
### syntax / subject_verb_number
- baseline_margin: 7.5937
- best_attn_layer: 0
- best_attn_margin_drop: 3.0312
- best_mlp_layer: 0
- best_mlp_margin_drop: 6.8750

### syntax / distance_agreement
- baseline_margin: 4.7188
- best_attn_layer: 0
- best_attn_margin_drop: 4.9531
- best_mlp_layer: 0
- best_mlp_margin_drop: 5.7031

### syntax / collective_local_noun
- baseline_margin: 4.0312
- best_attn_layer: 0
- best_attn_margin_drop: 8.5937
- best_mlp_layer: 0
- best_mlp_margin_drop: 5.8750

### relation / capital_relation
- baseline_margin: 14.1562
- best_attn_layer: 26
- best_attn_margin_drop: 2.3437
- best_mlp_layer: 0
- best_mlp_margin_drop: 11.9268

### relation / currency_relation
- baseline_margin: 8.1875
- best_attn_layer: 0
- best_attn_margin_drop: 4.0000
- best_mlp_layer: 0
- best_mlp_margin_drop: 6.0547

### relation / inventor_relation
- baseline_margin: 6.0000
- best_attn_layer: 0
- best_attn_margin_drop: 3.4375
- best_mlp_layer: 0
- best_mlp_margin_drop: 5.5713

### coref / winner_reference
- baseline_margin: 0.8750
- best_attn_layer: 9
- best_attn_margin_drop: 0.5000
- best_mlp_layer: 26
- best_mlp_margin_drop: 0.1875

### coref / help_reference
- baseline_margin: 1.6250
- best_attn_layer: 9
- best_attn_margin_drop: 0.4063
- best_mlp_layer: 18
- best_mlp_margin_drop: 0.9375

### coref / blame_reference
- baseline_margin: -0.6875
- best_attn_layer: 9
- best_attn_margin_drop: 0.8125
- best_mlp_layer: 18
- best_mlp_margin_drop: 0.0313

