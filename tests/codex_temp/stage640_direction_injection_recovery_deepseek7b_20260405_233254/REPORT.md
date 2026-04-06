# Stage640 差异方向注入恢复报告

- 时间: 2026-04-05 23:33:14
- 模型: deepseek7b
- 样本数: 6

## 分能力摘要
### coref
- baseline_margin: 2.8437
- ablated_margin: 0.2119
- recovered_margin: 0.4850
- mean_recovery_gain: 0.2731
- mean_recovery_ratio: 0.1254

### relation
- baseline_margin: 7.3438
- ablated_margin: 1.6351
- recovered_margin: 1.8762
- mean_recovery_gain: 0.2411
- mean_recovery_ratio: 0.0509

### syntax
- baseline_margin: 5.2344
- ablated_margin: -1.7778
- recovered_margin: -1.3516
- mean_recovery_gain: 0.4263
- mean_recovery_ratio: 0.0350

## 单样本摘要
### syntax / subject_verb_number
- best_damage_component: attn
- best_damage_layer: 0
- baseline_margin: 7.9688
- ablated_margin: -3.2500
- best_recovered_margin: -2.3750
- recovery_gain: 0.8750
- recovery_ratio: 0.0780
- best_scale: 1.0

### syntax / distance_agreement
- best_damage_component: attn
- best_damage_layer: 0
- baseline_margin: 2.5000
- ablated_margin: -0.3057
- best_recovered_margin: -0.3281
- recovery_gain: -0.0225
- recovery_ratio: -0.0080
- best_scale: 1.0

### relation / capital_relation
- best_damage_component: attn
- best_damage_layer: 0
- baseline_margin: 8.1562
- ablated_margin: 1.0055
- best_recovered_margin: 1.1250
- recovery_gain: 0.1195
- recovery_ratio: 0.0167
- best_scale: 0.25

### relation / currency_relation
- best_damage_component: attn
- best_damage_layer: 0
- baseline_margin: 6.5313
- ablated_margin: 2.2646
- best_recovered_margin: 2.6274
- recovery_gain: 0.3628
- recovery_ratio: 0.0850
- best_scale: 1.0

### coref / winner_reference
- best_damage_component: attn
- best_damage_layer: 0
- baseline_margin: 3.0312
- ablated_margin: -0.2949
- best_recovered_margin: -0.1504
- recovery_gain: 0.1445
- recovery_ratio: 0.0435
- best_scale: 0.5

### coref / help_reference
- best_damage_component: attn
- best_damage_layer: 14
- baseline_margin: 2.6562
- ablated_margin: 0.7187
- best_recovered_margin: 1.1203
- recovery_gain: 0.4016
- recovery_ratio: 0.2073
- best_scale: 0.25

