# Stage638 多能力统一编码协议报告

- 时间: 2026-04-05 22:53:54
- 模型: qwen3
- 能力范围: all
- 样本数: 10

## 总览
- overall_pair_accuracy: 0.7000
- overall_mean_margin: 5.2824
- overall_mean_d_norm_growth: 385.8776
- overall_mean_rotation_deg: 37.6692
- overall_mean_abs_final_alignment: 0.1316
- overall_mean_signed_final_alignment: -0.0005

## 分能力摘要
### coref
- count: 2
- pair_accuracy: 1.0000
- mean_margin: 1.8281
- mean_d_norm_growth: 504.6158
- mean_rotation_deg: 41.8295
- mean_abs_final_alignment: 0.1375
- mean_signed_final_alignment: 0.0000
- mean_peak_layer: 35.0000

### disamb
- count: 2
- pair_accuracy: 0.5000
- mean_margin: 4.2500
- mean_d_norm_growth: 736.4427
- mean_rotation_deg: 41.6540
- mean_abs_final_alignment: 0.1182
- mean_signed_final_alignment: 0.0000
- mean_peak_layer: 35.0000

### relation
- count: 2
- pair_accuracy: 1.0000
- mean_margin: 13.6932
- mean_d_norm_growth: 428.0909
- mean_rotation_deg: 40.4643
- mean_abs_final_alignment: 0.1346
- mean_signed_final_alignment: -0.0027
- mean_peak_layer: 35.0000

### style
- count: 2
- pair_accuracy: 0.0000
- mean_margin: 0.9062
- mean_d_norm_growth: 160.0264
- mean_rotation_deg: 34.6049
- mean_abs_final_alignment: 0.0416
- mean_signed_final_alignment: 0.0000
- mean_peak_layer: 35.0000

### syntax
- count: 2
- pair_accuracy: 1.0000
- mean_margin: 5.7344
- mean_d_norm_growth: 100.2119
- mean_rotation_deg: 29.7931
- mean_abs_final_alignment: 0.2259
- mean_signed_final_alignment: 0.0000
- mean_peak_layer: 35.0000

## 单样本明细
### disamb / bank_meaning
- avg_margin: 5.1562
- preference_correct_pair: True
- d_norm_growth: 755.1219
- mean_rotation_deg: 41.6693
- avg_abs_final_alignment: 0.1165
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### disamb / plant_meaning
- avg_margin: 3.3437
- preference_correct_pair: False
- d_norm_growth: 717.7636
- mean_rotation_deg: 41.6386
- avg_abs_final_alignment: 0.1198
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### syntax / subject_verb_number
- avg_margin: 7.5937
- preference_correct_pair: True
- d_norm_growth: 155.0167
- mean_rotation_deg: 33.9451
- avg_abs_final_alignment: 0.3559
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### syntax / agreement_with_clause
- avg_margin: 3.8750
- preference_correct_pair: True
- d_norm_growth: 45.4071
- mean_rotation_deg: 25.6412
- avg_abs_final_alignment: 0.0959
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### relation / capital_relation
- avg_margin: 14.1562
- preference_correct_pair: True
- d_norm_growth: 572.7042
- mean_rotation_deg: 40.9455
- avg_abs_final_alignment: 0.1912
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### relation / author_relation
- avg_margin: 13.2302
- preference_correct_pair: True
- d_norm_growth: 283.4777
- mean_rotation_deg: 39.9831
- avg_abs_final_alignment: 0.0780
- signed_avg_final_alignment: -0.0054
- layer_peak: 35

### coref / winner_reference
- avg_margin: 0.8750
- preference_correct_pair: True
- d_norm_growth: 366.7414
- mean_rotation_deg: 41.4248
- avg_abs_final_alignment: 0.0698
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### coref / apology_reference
- avg_margin: 2.7813
- preference_correct_pair: True
- d_norm_growth: 642.4902
- mean_rotation_deg: 42.2342
- avg_abs_final_alignment: 0.2052
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### style / formal_rewrite
- avg_margin: 0.0625
- preference_correct_pair: False
- d_norm_growth: 191.6799
- mean_rotation_deg: 36.7501
- avg_abs_final_alignment: 0.0042
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### style / formal_word_choice
- avg_margin: 1.7500
- preference_correct_pair: False
- d_norm_growth: 128.3730
- mean_rotation_deg: 32.4596
- avg_abs_final_alignment: 0.0790
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

