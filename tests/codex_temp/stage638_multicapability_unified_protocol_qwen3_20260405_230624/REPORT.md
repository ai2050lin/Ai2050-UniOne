# Stage638 多能力统一编码协议报告

- 时间: 2026-04-05 23:06:38
- 模型: qwen3
- 能力范围: all
- 样本数: 25

## 总览
- overall_pair_accuracy: 0.7600
- overall_mean_margin: 4.4263
- overall_mean_d_norm_growth: 401.5988
- overall_mean_rotation_deg: 35.9652
- overall_mean_mid_d_norm: 20.9993
- overall_mean_abs_mid_alignment: 0.0226
- overall_mean_abs_final_alignment: 0.1204
- overall_mean_signed_final_alignment: -0.0018

## 分能力摘要
### coref
- count: 5
- pair_accuracy: 0.6000
- mean_margin: 1.0688
- mean_d_norm_growth: 804.4026
- mean_rotation_deg: 41.8759
- mean_mid_d_norm: 8.5573
- mean_abs_mid_alignment: 0.0134
- mean_abs_final_alignment: 0.0809
- mean_signed_final_alignment: 0.0000
- mean_mid_layer: 18.0000
- mean_peak_layer: 35.0000

### disamb
- count: 5
- pair_accuracy: 0.8000
- mean_margin: 5.2094
- mean_d_norm_growth: 721.3418
- mean_rotation_deg: 40.4987
- mean_mid_d_norm: 12.0721
- mean_abs_mid_alignment: 0.0117
- mean_abs_final_alignment: 0.1254
- mean_signed_final_alignment: 0.0000
- mean_mid_layer: 18.0000
- mean_peak_layer: 35.0000

### relation
- count: 5
- pair_accuracy: 1.0000
- mean_margin: 9.2457
- mean_d_norm_growth: 282.5202
- mean_rotation_deg: 36.4305
- mean_mid_d_norm: 27.9797
- mean_abs_mid_alignment: 0.0222
- mean_abs_final_alignment: 0.1454
- mean_signed_final_alignment: -0.0092
- mean_mid_layer: 18.0000
- mean_peak_layer: 35.0000

### style
- count: 5
- pair_accuracy: 0.4000
- mean_margin: 1.9200
- mean_d_norm_growth: 130.9741
- mean_rotation_deg: 32.3056
- mean_mid_d_norm: 22.3160
- mean_abs_mid_alignment: 0.0248
- mean_abs_final_alignment: 0.0729
- mean_signed_final_alignment: 0.0000
- mean_mid_layer: 18.0000
- mean_peak_layer: 35.0000

### syntax
- count: 5
- pair_accuracy: 1.0000
- mean_margin: 4.6875
- mean_d_norm_growth: 68.7553
- mean_rotation_deg: 28.7155
- mean_mid_d_norm: 34.0715
- mean_abs_mid_alignment: 0.0410
- mean_abs_final_alignment: 0.1774
- mean_signed_final_alignment: 0.0000
- mean_mid_layer: 18.0000
- mean_peak_layer: 35.0000

## 单样本明细
### disamb / bank_meaning
- avg_margin: 5.1562
- preference_correct_pair: True
- d_norm_growth: 755.1219
- mean_rotation_deg: 41.6693
- mid_layer: 18
- mid_d_norm: 12.2103
- avg_abs_mid_alignment: 0.0216
- avg_abs_final_alignment: 0.1165
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### disamb / plant_meaning
- avg_margin: 3.3437
- preference_correct_pair: False
- d_norm_growth: 717.7636
- mean_rotation_deg: 41.6386
- mid_layer: 18
- mid_d_norm: 10.7526
- avg_abs_mid_alignment: 0.0078
- avg_abs_final_alignment: 0.1198
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### disamb / bat_meaning
- avg_margin: 5.7188
- preference_correct_pair: True
- d_norm_growth: 833.8307
- mean_rotation_deg: 40.5819
- mid_layer: 18
- mid_d_norm: 10.1007
- avg_abs_mid_alignment: 0.0098
- avg_abs_final_alignment: 0.1126
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### disamb / watch_meaning
- avg_margin: 6.8281
- preference_correct_pair: True
- d_norm_growth: 634.8647
- mean_rotation_deg: 38.9807
- mid_layer: 18
- mid_d_norm: 16.4732
- avg_abs_mid_alignment: 0.0128
- avg_abs_final_alignment: 0.1491
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### disamb / light_meaning
- avg_margin: 5.0000
- preference_correct_pair: True
- d_norm_growth: 665.1284
- mean_rotation_deg: 39.6229
- mid_layer: 18
- mid_d_norm: 10.8239
- avg_abs_mid_alignment: 0.0061
- avg_abs_final_alignment: 0.1288
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### syntax / subject_verb_number
- avg_margin: 7.5937
- preference_correct_pair: True
- d_norm_growth: 155.0207
- mean_rotation_deg: 33.9451
- mid_layer: 18
- mid_d_norm: 22.5058
- avg_abs_mid_alignment: 0.0448
- avg_abs_final_alignment: 0.3560
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### syntax / agreement_with_clause
- avg_margin: 3.8750
- preference_correct_pair: True
- d_norm_growth: 45.4071
- mean_rotation_deg: 25.6412
- mid_layer: 18
- mid_d_norm: 32.8109
- avg_abs_mid_alignment: 0.0847
- avg_abs_final_alignment: 0.0959
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### syntax / plural_with_pp
- avg_margin: 3.2188
- preference_correct_pair: True
- d_norm_growth: 63.5155
- mean_rotation_deg: 29.2184
- mid_layer: 18
- mid_d_norm: 31.2640
- avg_abs_mid_alignment: 0.0701
- avg_abs_final_alignment: 0.1057
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### syntax / distance_agreement
- avg_margin: 4.7188
- preference_correct_pair: True
- d_norm_growth: 20.8766
- mean_rotation_deg: 25.8675
- mid_layer: 18
- mid_d_norm: 55.3919
- avg_abs_mid_alignment: 0.0023
- avg_abs_final_alignment: 0.1715
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### syntax / collective_local_noun
- avg_margin: 4.0312
- preference_correct_pair: True
- d_norm_growth: 58.9567
- mean_rotation_deg: 28.9053
- mid_layer: 18
- mid_d_norm: 28.3848
- avg_abs_mid_alignment: 0.0030
- avg_abs_final_alignment: 0.1581
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### relation / capital_relation
- avg_margin: 14.1562
- preference_correct_pair: True
- d_norm_growth: 572.7042
- mean_rotation_deg: 40.9455
- mid_layer: 18
- mid_d_norm: 19.6454
- avg_abs_mid_alignment: 0.0410
- avg_abs_final_alignment: 0.1912
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### relation / author_relation
- avg_margin: 13.2302
- preference_correct_pair: True
- d_norm_growth: 283.4777
- mean_rotation_deg: 39.9831
- mid_layer: 18
- mid_d_norm: 27.7800
- avg_abs_mid_alignment: 0.0143
- avg_abs_final_alignment: 0.0780
- signed_avg_final_alignment: -0.0054
- layer_peak: 35

### relation / chemical_symbol_relation
- avg_margin: 4.6547
- preference_correct_pair: True
- d_norm_growth: 191.5388
- mean_rotation_deg: 36.4523
- mid_layer: 18
- mid_d_norm: 15.8338
- avg_abs_mid_alignment: 0.0267
- avg_abs_final_alignment: 0.2493
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### relation / currency_relation
- avg_margin: 8.1875
- preference_correct_pair: True
- d_norm_growth: 296.9853
- mean_rotation_deg: 36.1996
- mid_layer: 18
- mid_d_norm: 24.2855
- avg_abs_mid_alignment: 0.0243
- avg_abs_final_alignment: 0.1538
- signed_avg_final_alignment: -0.0406
- layer_peak: 35

### relation / inventor_relation
- avg_margin: 6.0000
- preference_correct_pair: True
- d_norm_growth: 67.8952
- mean_rotation_deg: 28.5719
- mid_layer: 18
- mid_d_norm: 52.3539
- avg_abs_mid_alignment: 0.0049
- avg_abs_final_alignment: 0.0549
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### coref / winner_reference
- avg_margin: 0.8750
- preference_correct_pair: True
- d_norm_growth: 366.7414
- mean_rotation_deg: 41.4248
- mid_layer: 18
- mid_d_norm: 7.0271
- avg_abs_mid_alignment: 0.0171
- avg_abs_final_alignment: 0.0698
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### coref / apology_reference
- avg_margin: 2.7813
- preference_correct_pair: True
- d_norm_growth: 642.4902
- mean_rotation_deg: 42.2342
- mid_layer: 18
- mid_d_norm: 10.7050
- avg_abs_mid_alignment: 0.0067
- avg_abs_final_alignment: 0.2052
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### coref / help_reference
- avg_margin: 1.6250
- preference_correct_pair: True
- d_norm_growth: 1782.0237
- mean_rotation_deg: 41.8651
- mid_layer: 18
- mid_d_norm: 8.7413
- avg_abs_mid_alignment: 0.0217
- avg_abs_final_alignment: 0.0155
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### coref / congratulation_reference
- avg_margin: 0.7500
- preference_correct_pair: False
- d_norm_growth: 345.6340
- mean_rotation_deg: 41.4871
- mid_layer: 18
- mid_d_norm: 7.4790
- avg_abs_mid_alignment: 0.0187
- avg_abs_final_alignment: 0.0703
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### coref / blame_reference
- avg_margin: -0.6875
- preference_correct_pair: False
- d_norm_growth: 885.1237
- mean_rotation_deg: 42.3682
- mid_layer: 18
- mid_d_norm: 8.8339
- avg_abs_mid_alignment: 0.0027
- avg_abs_final_alignment: 0.0437
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### style / formal_rewrite
- avg_margin: -0.9062
- preference_correct_pair: False
- d_norm_growth: 195.1051
- mean_rotation_deg: 35.8198
- mid_layer: 18
- mid_d_norm: 13.9373
- avg_abs_mid_alignment: 0.0414
- avg_abs_final_alignment: 0.0784
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### style / formal_word_choice
- avg_margin: 1.7500
- preference_correct_pair: False
- d_norm_growth: 128.3730
- mean_rotation_deg: 32.4596
- mid_layer: 18
- mid_d_norm: 18.8337
- avg_abs_mid_alignment: 0.0361
- avg_abs_final_alignment: 0.0790
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### style / formal_request
- avg_margin: 2.0374
- preference_correct_pair: True
- d_norm_growth: 214.6425
- mean_rotation_deg: 33.9475
- mid_layer: 18
- mid_d_norm: 19.6876
- avg_abs_mid_alignment: 0.0291
- avg_abs_final_alignment: 0.0823
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### style / formal_apology
- avg_margin: 6.2187
- preference_correct_pair: True
- d_norm_growth: 48.3185
- mean_rotation_deg: 25.9765
- mid_layer: 18
- mid_d_norm: 40.1649
- avg_abs_mid_alignment: 0.0091
- avg_abs_final_alignment: 0.1090
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

### style / formal_departure
- avg_margin: 0.5000
- preference_correct_pair: False
- d_norm_growth: 68.4315
- mean_rotation_deg: 33.3249
- mid_layer: 18
- mid_d_norm: 18.9565
- avg_abs_mid_alignment: 0.0084
- avg_abs_final_alignment: 0.0157
- signed_avg_final_alignment: 0.0000
- layer_peak: 35

