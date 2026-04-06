# Stage638 多能力统一编码协议报告

- 时间: 2026-04-05 22:54:04
- 模型: deepseek7b
- 能力范围: all
- 样本数: 10

## 总览
- overall_pair_accuracy: 0.7000
- overall_mean_margin: 4.7858
- overall_mean_d_norm_growth: 226.7319
- overall_mean_rotation_deg: 35.5507
- overall_mean_abs_final_alignment: 0.0842
- overall_mean_signed_final_alignment: -0.0038

## 分能力摘要
### coref
- count: 2
- pair_accuracy: 1.0000
- mean_margin: 2.9688
- mean_d_norm_growth: 406.1486
- mean_rotation_deg: 40.2608
- mean_abs_final_alignment: 0.0799
- mean_signed_final_alignment: 0.0000
- mean_peak_layer: 27.0000

### disamb
- count: 2
- pair_accuracy: 0.5000
- mean_margin: 4.8535
- mean_d_norm_growth: 427.5318
- mean_rotation_deg: 38.9695
- mean_abs_final_alignment: 0.0515
- mean_signed_final_alignment: 0.0000
- mean_peak_layer: 27.0000

### relation
- count: 2
- pair_accuracy: 1.0000
- mean_margin: 8.3569
- mean_d_norm_growth: 106.6550
- mean_rotation_deg: 36.5456
- mean_abs_final_alignment: 0.0898
- mean_signed_final_alignment: -0.0189
- mean_peak_layer: 27.0000

### style
- count: 2
- pair_accuracy: 0.0000
- mean_margin: 0.9531
- mean_d_norm_growth: 120.6021
- mean_rotation_deg: 33.9292
- mean_abs_final_alignment: 0.0321
- mean_signed_final_alignment: 0.0000
- mean_peak_layer: 27.0000

### syntax
- count: 2
- pair_accuracy: 1.0000
- mean_margin: 6.7969
- mean_d_norm_growth: 72.7221
- mean_rotation_deg: 28.0485
- mean_abs_final_alignment: 0.1679
- mean_signed_final_alignment: 0.0000
- mean_peak_layer: 27.0000

## 单样本明细
### disamb / bank_meaning
- avg_margin: 6.6602
- preference_correct_pair: True
- d_norm_growth: 325.8409
- mean_rotation_deg: 38.1260
- avg_abs_final_alignment: 0.0599
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### disamb / plant_meaning
- avg_margin: 3.0469
- preference_correct_pair: False
- d_norm_growth: 529.2227
- mean_rotation_deg: 39.8130
- avg_abs_final_alignment: 0.0431
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### syntax / subject_verb_number
- avg_margin: 7.9688
- preference_correct_pair: True
- d_norm_growth: 107.7829
- mean_rotation_deg: 31.8280
- avg_abs_final_alignment: 0.2352
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### syntax / agreement_with_clause
- avg_margin: 5.6250
- preference_correct_pair: True
- d_norm_growth: 37.6612
- mean_rotation_deg: 24.2690
- avg_abs_final_alignment: 0.1006
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### relation / capital_relation
- avg_margin: 8.1562
- preference_correct_pair: True
- d_norm_growth: 114.1991
- mean_rotation_deg: 37.0378
- avg_abs_final_alignment: 0.1187
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### relation / author_relation
- avg_margin: 8.5575
- preference_correct_pair: True
- d_norm_growth: 99.1108
- mean_rotation_deg: 36.0534
- avg_abs_final_alignment: 0.0609
- signed_avg_final_alignment: -0.0377
- layer_peak: 27

### coref / winner_reference
- avg_margin: 3.0312
- preference_correct_pair: True
- d_norm_growth: 462.0536
- mean_rotation_deg: 38.8294
- avg_abs_final_alignment: 0.0472
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### coref / apology_reference
- avg_margin: 2.9063
- preference_correct_pair: True
- d_norm_growth: 350.2437
- mean_rotation_deg: 41.6922
- avg_abs_final_alignment: 0.1126
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### style / formal_rewrite
- avg_margin: 0.1250
- preference_correct_pair: False
- d_norm_growth: 177.1401
- mean_rotation_deg: 36.1788
- avg_abs_final_alignment: 0.0178
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### style / formal_word_choice
- avg_margin: 1.7812
- preference_correct_pair: False
- d_norm_growth: 64.0641
- mean_rotation_deg: 31.6797
- avg_abs_final_alignment: 0.0463
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

