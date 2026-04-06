# Stage640 差异方向注入恢复报告

- 时间: 2026-04-05 23:33:54
- 模型: gemma4
- 样本数: 6

## 分能力摘要
### coref
- baseline_margin: 1.6250
- ablated_margin: 0.1406
- recovered_margin: 3.5313
- mean_recovery_gain: 3.3906
- mean_recovery_ratio: 2.2879

### relation
- baseline_margin: 1.6543
- ablated_margin: 0.5854
- recovered_margin: 1.5469
- mean_recovery_gain: 0.9615
- mean_recovery_ratio: 3.3729

### syntax
- baseline_margin: 1.1875
- ablated_margin: -0.4434
- recovered_margin: 0.8828
- mean_recovery_gain: 1.3262
- mean_recovery_ratio: 2.5916

## 单样本摘要
### syntax / subject_verb_number
- best_damage_component: attn
- best_damage_layer: 0
- baseline_margin: 1.8125
- ablated_margin: -0.9375
- best_recovered_margin: -0.9375
- recovery_gain: -0.0000
- recovery_ratio: -0.0000
- best_scale: 0.25

### syntax / distance_agreement
- best_damage_component: mlp
- best_damage_layer: 34
- baseline_margin: 0.5625
- ablated_margin: 0.0508
- best_recovered_margin: 2.7031
- recovery_gain: 2.6523
- recovery_ratio: 5.1832
- best_scale: 1.0

### relation / capital_relation
- best_damage_component: attn
- best_damage_layer: 34
- baseline_margin: 2.4336
- ablated_margin: 0.3583
- best_recovered_margin: 1.9063
- recovery_gain: 1.5480
- recovery_ratio: 0.7459
- best_scale: 1.0

### relation / currency_relation
- best_damage_component: attn
- best_damage_layer: 26
- baseline_margin: 0.8750
- ablated_margin: 0.8125
- best_recovered_margin: 1.1875
- recovery_gain: 0.3750
- recovery_ratio: 6.0000
- best_scale: 1.0

### coref / winner_reference
- best_damage_component: attn
- best_damage_layer: 34
- baseline_margin: 2.3750
- ablated_margin: 0.2500
- best_recovered_margin: 5.0938
- recovery_gain: 4.8438
- recovery_ratio: 2.2794
- best_scale: 1.0

### coref / help_reference
- best_damage_component: attn
- best_damage_layer: 34
- baseline_margin: 0.8750
- ablated_margin: 0.0312
- best_recovered_margin: 1.9688
- recovery_gain: 1.9375
- recovery_ratio: 2.2963
- best_scale: 1.0

