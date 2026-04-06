# Stage638 多能力统一编码协议报告

- 时间: 2026-04-05 22:48:37
- 模型: qwen3
- 能力范围: all
- 样本数: 5

## 总览
- overall_pair_accuracy: 0.6000
- overall_mean_margin: 4.6750
- overall_mean_d_norm_growth: 343.1921
- overall_mean_rotation_deg: 37.9392
- overall_mean_final_alignment: 0.0000

## 分能力摘要
### coref
- count: 1
- pair_accuracy: 1.0000
- mean_margin: 0.8750
- mean_d_norm_growth: 366.7414
- mean_rotation_deg: 41.4248
- mean_final_alignment: 0.0000
- mean_peak_layer: 35.0000

### disamb
- count: 1
- pair_accuracy: 0.0000
- mean_margin: 3.9688
- mean_d_norm_growth: 251.4921
- mean_rotation_deg: 36.4744
- mean_final_alignment: 0.0000
- mean_peak_layer: 35.0000

### relation
- count: 1
- pair_accuracy: 1.0000
- mean_margin: 14.1562
- mean_d_norm_growth: 572.7042
- mean_rotation_deg: 40.9455
- mean_final_alignment: 0.0000
- mean_peak_layer: 35.0000

### style
- count: 1
- pair_accuracy: 0.0000
- mean_margin: -3.2188
- mean_d_norm_growth: 370.0060
- mean_rotation_deg: 36.9062
- mean_final_alignment: 0.0000
- mean_peak_layer: 35.0000

### syntax
- count: 1
- pair_accuracy: 1.0000
- mean_margin: 7.5937
- mean_d_norm_growth: 155.0167
- mean_rotation_deg: 33.9451
- mean_final_alignment: 0.0000
- mean_peak_layer: 35.0000

## 单样本明细
### disamb / bank_meaning
- avg_margin: 3.9688
- preference_correct_pair: False
- d_norm_growth: 251.4921
- mean_rotation_deg: 36.4744
- avg_final_alignment: 0.0000
- layer_peak: 35

### syntax / subject_verb_number
- avg_margin: 7.5937
- preference_correct_pair: True
- d_norm_growth: 155.0167
- mean_rotation_deg: 33.9451
- avg_final_alignment: 0.0000
- layer_peak: 35

### relation / capital_relation
- avg_margin: 14.1562
- preference_correct_pair: True
- d_norm_growth: 572.7042
- mean_rotation_deg: 40.9455
- avg_final_alignment: 0.0000
- layer_peak: 35

### coref / winner_reference
- avg_margin: 0.8750
- preference_correct_pair: True
- d_norm_growth: 366.7414
- mean_rotation_deg: 41.4248
- avg_final_alignment: 0.0000
- layer_peak: 35

### style / formal_rewrite
- avg_margin: -3.2188
- preference_correct_pair: False
- d_norm_growth: 370.0060
- mean_rotation_deg: 36.9062
- avg_final_alignment: 0.0000
- layer_peak: 35

