# Stage639 组件级因果验证报告

- 时间: 2026-04-05 23:21:50
- 模型: gemma4
- 抽样层: [0, 8, 17, 26, 34]
- 样本数: 9

## 分能力摘要
### syntax
- baseline_pair_accuracy: 0.0000
- baseline_margin: 0.3750
- attn_margin_drop: 0.1490
- mlp_margin_drop: 0.0141
- attn_pair_accuracy_drop: -0.2000
- mlp_pair_accuracy_drop: -0.0667

### relation
- baseline_pair_accuracy: 0.6667
- baseline_margin: 0.7695
- attn_margin_drop: -0.2481
- mlp_margin_drop: -0.4602
- attn_pair_accuracy_drop: 0.0667
- mlp_pair_accuracy_drop: 0.0667

### coref
- baseline_pair_accuracy: 1.0000
- baseline_margin: 1.3125
- attn_margin_drop: 0.5250
- mlp_margin_drop: -0.2680
- attn_pair_accuracy_drop: 0.4667
- mlp_pair_accuracy_drop: 0.4000

## 单样本摘要
### syntax / subject_verb_number
- baseline_margin: 1.8125
- best_attn_layer: 0
- best_attn_margin_drop: 2.7500
- best_mlp_layer: 0
- best_mlp_margin_drop: 1.0000

### syntax / distance_agreement
- baseline_margin: 0.5625
- best_attn_layer: 26
- best_attn_margin_drop: 0.1250
- best_mlp_layer: 34
- best_mlp_margin_drop: 0.5117

### syntax / collective_local_noun
- baseline_margin: -1.2500
- best_attn_layer: 0
- best_attn_margin_drop: 0.4688
- best_mlp_layer: 34
- best_mlp_margin_drop: 2.0156

### relation / capital_relation
- baseline_margin: 2.4336
- best_attn_layer: 34
- best_attn_margin_drop: 2.0753
- best_mlp_layer: 26
- best_mlp_margin_drop: 0.6211

### relation / currency_relation
- baseline_margin: 0.8750
- best_attn_layer: 26
- best_attn_margin_drop: 0.0625
- best_mlp_layer: 26
- best_mlp_margin_drop: 0.0625

### relation / inventor_relation
- baseline_margin: -1.0000
- best_attn_layer: 8
- best_attn_margin_drop: 0.5000
- best_mlp_layer: 34
- best_mlp_margin_drop: 7.3633

### coref / winner_reference
- baseline_margin: 2.3750
- best_attn_layer: 34
- best_attn_margin_drop: 2.1250
- best_mlp_layer: 26
- best_mlp_margin_drop: 1.7500

### coref / help_reference
- baseline_margin: 0.8750
- best_attn_layer: 34
- best_attn_margin_drop: 0.8438
- best_mlp_layer: 0
- best_mlp_margin_drop: 0.5000

### coref / blame_reference
- baseline_margin: 0.6875
- best_attn_layer: 34
- best_attn_margin_drop: 0.8125
- best_mlp_layer: 0
- best_mlp_margin_drop: 2.5625

