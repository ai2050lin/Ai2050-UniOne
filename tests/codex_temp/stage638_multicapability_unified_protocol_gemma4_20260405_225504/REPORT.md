# Stage638 多能力统一编码协议报告

- 时间: 2026-04-05 22:55:21
- 模型: gemma4
- 能力范围: all
- 样本数: 10

## 总览
- overall_pair_accuracy: 0.2000
- overall_mean_margin: 0.7918
- overall_mean_d_norm_growth: 4.2232
- overall_mean_rotation_deg: 52.9053
- overall_mean_abs_final_alignment: 0.0288
- overall_mean_signed_final_alignment: -0.0013

## 分能力摘要
### coref
- count: 2
- pair_accuracy: 0.5000
- mean_margin: 1.8125
- mean_d_norm_growth: 11.2915
- mean_rotation_deg: 54.2000
- mean_abs_final_alignment: 0.0669
- mean_signed_final_alignment: 0.0000
- mean_peak_layer: 19.0000

### disamb
- count: 2
- pair_accuracy: 0.0000
- mean_margin: -0.1797
- mean_d_norm_growth: 2.8627
- mean_rotation_deg: 53.2980
- mean_abs_final_alignment: 0.0190
- mean_signed_final_alignment: 0.0000
- mean_peak_layer: 17.0000

### relation
- count: 2
- pair_accuracy: 0.5000
- mean_margin: 1.2480
- mean_d_norm_growth: 2.7357
- mean_rotation_deg: 51.0147
- mean_abs_final_alignment: 0.0140
- mean_signed_final_alignment: -0.0067
- mean_peak_layer: 17.0000

### style
- count: 2
- pair_accuracy: 0.0000
- mean_margin: 0.5000
- mean_d_norm_growth: 2.7323
- mean_rotation_deg: 53.7795
- mean_abs_final_alignment: 0.0171
- mean_signed_final_alignment: 0.0000
- mean_peak_layer: 17.0000

### syntax
- count: 2
- pair_accuracy: 0.0000
- mean_margin: 0.5781
- mean_d_norm_growth: 1.4937
- mean_rotation_deg: 52.2342
- mean_abs_final_alignment: 0.0270
- mean_signed_final_alignment: 0.0000
- mean_peak_layer: 16.0000

## 单样本明细
### disamb / bank_meaning
- avg_margin: 0.0781
- preference_correct_pair: False
- d_norm_growth: 2.0727
- mean_rotation_deg: 53.0521
- avg_abs_final_alignment: 0.0241
- signed_avg_final_alignment: 0.0000
- layer_peak: 10

### disamb / plant_meaning
- avg_margin: -0.4375
- preference_correct_pair: False
- d_norm_growth: 3.6526
- mean_rotation_deg: 53.5438
- avg_abs_final_alignment: 0.0139
- signed_avg_final_alignment: 0.0000
- layer_peak: 24

### syntax / subject_verb_number
- avg_margin: 1.8125
- preference_correct_pair: False
- d_norm_growth: 2.6042
- mean_rotation_deg: 53.3761
- avg_abs_final_alignment: 0.0181
- signed_avg_final_alignment: 0.0000
- layer_peak: 8

### syntax / agreement_with_clause
- avg_margin: -0.6562
- preference_correct_pair: False
- d_norm_growth: 0.3832
- mean_rotation_deg: 51.0924
- avg_abs_final_alignment: 0.0358
- signed_avg_final_alignment: 0.0000
- layer_peak: 24

### relation / capital_relation
- avg_margin: 2.4336
- preference_correct_pair: True
- d_norm_growth: 5.0308
- mean_rotation_deg: 50.4925
- avg_abs_final_alignment: 0.0007
- signed_avg_final_alignment: 0.0000
- layer_peak: 25

### relation / author_relation
- avg_margin: 0.0625
- preference_correct_pair: False
- d_norm_growth: 0.4407
- mean_rotation_deg: 51.5370
- avg_abs_final_alignment: 0.0273
- signed_avg_final_alignment: -0.0133
- layer_peak: 9

### coref / winner_reference
- avg_margin: 2.3750
- preference_correct_pair: True
- d_norm_growth: 5.3593
- mean_rotation_deg: 53.8120
- avg_abs_final_alignment: 0.1336
- signed_avg_final_alignment: 0.0000
- layer_peak: 19

### coref / apology_reference
- avg_margin: 1.2500
- preference_correct_pair: False
- d_norm_growth: 17.2236
- mean_rotation_deg: 54.5879
- avg_abs_final_alignment: 0.0003
- signed_avg_final_alignment: 0.0000
- layer_peak: 19

### style / formal_rewrite
- avg_margin: 0.4375
- preference_correct_pair: False
- d_norm_growth: 4.9145
- mean_rotation_deg: 55.1952
- avg_abs_final_alignment: 0.0293
- signed_avg_final_alignment: 0.0000
- layer_peak: 9

### style / formal_word_choice
- avg_margin: 0.5625
- preference_correct_pair: False
- d_norm_growth: 0.5500
- mean_rotation_deg: 52.3638
- avg_abs_final_alignment: 0.0049
- signed_avg_final_alignment: 0.0000
- layer_peak: 25

