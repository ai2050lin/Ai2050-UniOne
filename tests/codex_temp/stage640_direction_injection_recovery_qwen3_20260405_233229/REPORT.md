# Stage640 差异方向注入恢复报告

- 时间: 2026-04-05 23:32:44
- 模型: qwen3
- 样本数: 6

## 分能力摘要
### coref
- baseline_margin: 1.2869
- ablated_margin: 0.5312
- recovered_margin: 1.4844
- mean_recovery_gain: 0.9531
- mean_recovery_ratio: 0.9425

### relation
- baseline_margin: 11.1719
- ablated_margin: 2.1836
- recovered_margin: 2.1656
- mean_recovery_gain: -0.0180
- mean_recovery_ratio: -0.0009

### syntax
- baseline_margin: 6.1562
- ablated_margin: -0.1328
- recovered_margin: 1.0073
- mean_recovery_gain: 1.1401
- mean_recovery_ratio: 0.1980

## 单样本摘要
### syntax / subject_verb_number
- best_damage_component: mlp
- best_damage_layer: 0
- baseline_margin: 7.5937
- ablated_margin: 0.7188
- best_recovered_margin: 0.8438
- recovery_gain: 0.1250
- recovery_ratio: 0.0182
- best_scale: 0.25

### syntax / distance_agreement
- best_damage_component: mlp
- best_damage_layer: 0
- baseline_margin: 4.7188
- ablated_margin: -0.9844
- best_recovered_margin: 1.1709
- recovery_gain: 2.1553
- recovery_ratio: 0.3779
- best_scale: 0.5

### relation / capital_relation
- best_damage_component: mlp
- best_damage_layer: 0
- baseline_margin: 14.1562
- ablated_margin: 2.2344
- best_recovered_margin: 2.1828
- recovery_gain: -0.0516
- recovery_ratio: -0.0043
- best_scale: 0.25

### relation / currency_relation
- best_damage_component: mlp
- best_damage_layer: 0
- baseline_margin: 8.1875
- ablated_margin: 2.1328
- best_recovered_margin: 2.1484
- recovery_gain: 0.0156
- recovery_ratio: 0.0026
- best_scale: 0.5

### coref / winner_reference
- best_damage_component: attn
- best_damage_layer: 9
- baseline_margin: 0.8750
- ablated_margin: 0.3750
- best_recovered_margin: 0.3750
- recovery_gain: 0.0000
- recovery_ratio: 0.0000
- best_scale: 1.0

### coref / help_reference
- best_damage_component: mlp
- best_damage_layer: 18
- baseline_margin: 1.6987
- ablated_margin: 0.6875
- best_recovered_margin: 2.5938
- recovery_gain: 1.9063
- recovery_ratio: 1.8851
- best_scale: 1.0

