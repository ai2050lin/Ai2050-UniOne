# Stage639 组件级因果验证报告

- 时间: 2026-04-05 23:22:34
- 模型: glm4
- 抽样层: [0, 10, 20, 29, 39]
- 样本数: 9

## 分能力摘要
### syntax
- baseline_pair_accuracy: 1.0000
- baseline_margin: 4.1875
- attn_margin_drop: 0.4771
- mlp_margin_drop: 1.1471
- attn_pair_accuracy_drop: 0.1333
- mlp_pair_accuracy_drop: 0.2000

### relation
- baseline_pair_accuracy: 1.0000
- baseline_margin: 7.3568
- attn_margin_drop: 0.7151
- mlp_margin_drop: 1.5119
- attn_pair_accuracy_drop: 0.0667
- mlp_pair_accuracy_drop: 0.2000

### coref
- baseline_pair_accuracy: 1.0000
- baseline_margin: 4.5938
- attn_margin_drop: 0.8035
- mlp_margin_drop: 1.4346
- attn_pair_accuracy_drop: 0.1333
- mlp_pair_accuracy_drop: 0.2667

## 单样本摘要
### syntax / subject_verb_number
- baseline_margin: 4.9688
- best_attn_layer: 0
- best_attn_margin_drop: 3.1250
- best_mlp_layer: 0
- best_mlp_margin_drop: 4.1875

### syntax / distance_agreement
- baseline_margin: 3.9688
- best_attn_layer: 0
- best_attn_margin_drop: 2.5859
- best_mlp_layer: 0
- best_mlp_margin_drop: 4.4053

### syntax / collective_local_noun
- baseline_margin: 3.6250
- best_attn_layer: 0
- best_attn_margin_drop: 1.4297
- best_mlp_layer: 0
- best_mlp_margin_drop: 4.5156

### relation / capital_relation
- baseline_margin: 6.7656
- best_attn_layer: 0
- best_attn_margin_drop: 6.6443
- best_mlp_layer: 0
- best_mlp_margin_drop: 5.8545

### relation / currency_relation
- baseline_margin: 8.6250
- best_attn_layer: 0
- best_attn_margin_drop: 2.0430
- best_mlp_layer: 0
- best_mlp_margin_drop: 8.0859

### relation / inventor_relation
- baseline_margin: 6.6797
- best_attn_layer: 0
- best_attn_margin_drop: 3.0078
- best_mlp_layer: 0
- best_mlp_margin_drop: 4.7187

### coref / winner_reference
- baseline_margin: 4.6250
- best_attn_layer: 10
- best_attn_margin_drop: 3.1719
- best_mlp_layer: 10
- best_mlp_margin_drop: 3.2812

### coref / help_reference
- baseline_margin: 6.2813
- best_attn_layer: 0
- best_attn_margin_drop: 1.9121
- best_mlp_layer: 0
- best_mlp_margin_drop: 7.3047

### coref / blame_reference
- baseline_margin: 2.8750
- best_attn_layer: 0
- best_attn_margin_drop: 3.3594
- best_mlp_layer: 0
- best_mlp_margin_drop: 3.9685

