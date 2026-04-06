# Stage640 差异方向注入恢复报告

- 时间: 2026-04-05 23:34:32
- 模型: glm4
- 样本数: 6

## 分能力摘要
### coref
- baseline_margin: 5.4531
- ablated_margin: 0.1531
- recovered_margin: 0.2578
- mean_recovery_gain: 0.1047
- mean_recovery_ratio: 0.0143

### relation
- baseline_margin: 7.6953
- ablated_margin: 0.3302
- recovered_margin: 0.3890
- mean_recovery_gain: 0.0588
- mean_recovery_ratio: 0.0076

### syntax
- baseline_margin: 4.4688
- ablated_margin: 0.1724
- recovered_margin: 1.3848
- mean_recovery_gain: 1.2124
- mean_recovery_ratio: 0.2760

## 单样本摘要
### syntax / subject_verb_number
- best_damage_component: mlp
- best_damage_layer: 0
- baseline_margin: 4.9688
- ablated_margin: 0.7812
- best_recovered_margin: 0.9062
- recovery_gain: 0.1250
- recovery_ratio: 0.0299
- best_scale: 0.25

### syntax / distance_agreement
- best_damage_component: mlp
- best_damage_layer: 0
- baseline_margin: 3.9688
- ablated_margin: -0.4365
- best_recovered_margin: 1.8633
- recovery_gain: 2.2998
- recovery_ratio: 0.5221
- best_scale: 1.0

### relation / capital_relation
- best_damage_component: attn
- best_damage_layer: 0
- baseline_margin: 6.7656
- ablated_margin: 0.1213
- best_recovered_margin: 0.1453
- recovery_gain: 0.0239
- recovery_ratio: 0.0036
- best_scale: 1.0

### relation / currency_relation
- best_damage_component: mlp
- best_damage_layer: 0
- baseline_margin: 8.6250
- ablated_margin: 0.5391
- best_recovered_margin: 0.6328
- recovery_gain: 0.0937
- recovery_ratio: 0.0116
- best_scale: 0.5

### coref / winner_reference
- best_damage_component: mlp
- best_damage_layer: 10
- baseline_margin: 4.6250
- ablated_margin: 1.3437
- best_recovered_margin: 1.3438
- recovery_gain: 0.0000
- recovery_ratio: 0.0000
- best_scale: 1.0

### coref / help_reference
- best_damage_component: mlp
- best_damage_layer: 0
- baseline_margin: 6.2813
- ablated_margin: -1.0375
- best_recovered_margin: -0.8281
- recovery_gain: 0.2094
- recovery_ratio: 0.0286
- best_scale: 1.0

