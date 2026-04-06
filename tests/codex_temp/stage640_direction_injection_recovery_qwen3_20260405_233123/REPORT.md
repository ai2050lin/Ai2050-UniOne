# Stage640 差异方向注入恢复报告

- 时间: 2026-04-05 23:31:41
- 模型: qwen3
- 样本数: 6

## 分能力摘要
### coref
- baseline_margin: 1.2500
- ablated_margin: 0.5312
- recovered_margin: 0.5312
- mean_recovery_gain: 0.0000
- mean_recovery_ratio: 0.0000

### relation
- baseline_margin: 11.1719
- ablated_margin: 2.1836
- recovered_margin: 2.1836
- mean_recovery_gain: 0.0000
- mean_recovery_ratio: 0.0000

### syntax
- baseline_margin: 6.1562
- ablated_margin: -0.1328
- recovered_margin: -0.1328
- mean_recovery_gain: 0.0000
- mean_recovery_ratio: 0.0000

## 单样本摘要
### syntax / subject_verb_number
- best_damage_component: mlp
- best_damage_layer: 0
- baseline_margin: 7.5937
- ablated_margin: 0.7188
- best_recovered_margin: 0.7188
- recovery_gain: 0.0000
- recovery_ratio: 0.0000
- best_scale: 0.5

### syntax / distance_agreement
- best_damage_component: mlp
- best_damage_layer: 0
- baseline_margin: 4.7188
- ablated_margin: -0.9844
- best_recovered_margin: -0.9844
- recovery_gain: 0.0000
- recovery_ratio: 0.0000
- best_scale: 0.25

### relation / capital_relation
- best_damage_component: mlp
- best_damage_layer: 0
- baseline_margin: 14.1562
- ablated_margin: 2.2344
- best_recovered_margin: 2.2344
- recovery_gain: 0.0000
- recovery_ratio: 0.0000
- best_scale: 0.25

### relation / currency_relation
- best_damage_component: mlp
- best_damage_layer: 0
- baseline_margin: 8.1875
- ablated_margin: 2.1328
- best_recovered_margin: 2.1328
- recovery_gain: 0.0000
- recovery_ratio: 0.0000
- best_scale: 0.25

### coref / winner_reference
- best_damage_component: attn
- best_damage_layer: 9
- baseline_margin: 0.8750
- ablated_margin: 0.3750
- best_recovered_margin: 0.3750
- recovery_gain: 0.0000
- recovery_ratio: 0.0000
- best_scale: 0.25

### coref / help_reference
- best_damage_component: mlp
- best_damage_layer: 18
- baseline_margin: 1.6250
- ablated_margin: 0.6875
- best_recovered_margin: 0.6875
- recovery_gain: 0.0000
- recovery_ratio: 0.0000
- best_scale: 0.25

