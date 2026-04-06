# Stage638 多能力统一编码协议报告

- 时间: 2026-04-05 23:07:07
- 模型: deepseek7b
- 能力范围: all
- 样本数: 25

## 总览
- overall_pair_accuracy: 0.7600
- overall_mean_margin: 4.2954
- overall_mean_d_norm_growth: 182.7594
- overall_mean_rotation_deg: 33.6304
- overall_mean_mid_d_norm: 103.6339
- overall_mean_abs_mid_alignment: 0.0198
- overall_mean_abs_final_alignment: 0.0786
- overall_mean_signed_final_alignment: -0.0020

## 分能力摘要
### coref
- count: 5
- pair_accuracy: 0.8000
- mean_margin: 2.3375
- mean_d_norm_growth: 320.0827
- mean_rotation_deg: 39.7649
- mean_mid_d_norm: 32.0543
- mean_abs_mid_alignment: 0.0103
- mean_abs_final_alignment: 0.0771
- mean_signed_final_alignment: 0.0000
- mean_mid_layer: 14.0000
- mean_peak_layer: 27.0000

### disamb
- count: 5
- pair_accuracy: 0.8000
- mean_margin: 6.3677
- mean_d_norm_growth: 359.6425
- mean_rotation_deg: 39.0516
- mean_mid_d_norm: 61.2868
- mean_abs_mid_alignment: 0.0215
- mean_abs_final_alignment: 0.0699
- mean_signed_final_alignment: 0.0000
- mean_mid_layer: 14.0000
- mean_peak_layer: 27.0000

### relation
- count: 5
- pair_accuracy: 0.8000
- mean_margin: 5.8965
- mean_d_norm_growth: 85.2248
- mean_rotation_deg: 33.2146
- mean_mid_d_norm: 103.4546
- mean_abs_mid_alignment: 0.0181
- mean_abs_final_alignment: 0.0877
- mean_signed_final_alignment: -0.0102
- mean_mid_layer: 14.0000
- mean_peak_layer: 27.0000

### style
- count: 5
- pair_accuracy: 0.4000
- mean_margin: 2.0750
- mean_d_norm_growth: 100.4902
- mean_rotation_deg: 29.0535
- mean_mid_d_norm: 202.1859
- mean_abs_mid_alignment: 0.0204
- mean_abs_final_alignment: 0.0381
- mean_signed_final_alignment: 0.0000
- mean_mid_layer: 14.0000
- mean_peak_layer: 27.0000

### syntax
- count: 5
- pair_accuracy: 1.0000
- mean_margin: 4.8000
- mean_d_norm_growth: 48.3568
- mean_rotation_deg: 27.0674
- mean_mid_d_norm: 119.1879
- mean_abs_mid_alignment: 0.0290
- mean_abs_final_alignment: 0.1202
- mean_signed_final_alignment: 0.0000
- mean_mid_layer: 14.0000
- mean_peak_layer: 27.0000

## 单样本明细
### disamb / bank_meaning
- avg_margin: 6.6602
- preference_correct_pair: True
- d_norm_growth: 325.8409
- mean_rotation_deg: 38.1260
- mid_layer: 14
- mid_d_norm: 67.4025
- avg_abs_mid_alignment: 0.0271
- avg_abs_final_alignment: 0.0599
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### disamb / plant_meaning
- avg_margin: 3.0469
- preference_correct_pair: False
- d_norm_growth: 529.2227
- mean_rotation_deg: 39.8130
- mid_layer: 14
- mid_d_norm: 54.7717
- avg_abs_mid_alignment: 0.0231
- avg_abs_final_alignment: 0.0431
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### disamb / bat_meaning
- avg_margin: 6.2227
- preference_correct_pair: True
- d_norm_growth: 379.5491
- mean_rotation_deg: 39.9954
- mid_layer: 14
- mid_d_norm: 48.3447
- avg_abs_mid_alignment: 0.0185
- avg_abs_final_alignment: 0.0647
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### disamb / watch_meaning
- avg_margin: 9.3816
- preference_correct_pair: True
- d_norm_growth: 223.3180
- mean_rotation_deg: 37.8657
- mid_layer: 14
- mid_d_norm: 81.4768
- avg_abs_mid_alignment: 0.0254
- avg_abs_final_alignment: 0.1063
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### disamb / light_meaning
- avg_margin: 6.5273
- preference_correct_pair: True
- d_norm_growth: 340.2820
- mean_rotation_deg: 39.4581
- mid_layer: 14
- mid_d_norm: 54.4383
- avg_abs_mid_alignment: 0.0135
- avg_abs_final_alignment: 0.0757
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### syntax / subject_verb_number
- avg_margin: 7.9688
- preference_correct_pair: True
- d_norm_growth: 107.7829
- mean_rotation_deg: 31.8280
- mid_layer: 14
- mid_d_norm: 94.9338
- avg_abs_mid_alignment: 0.0203
- avg_abs_final_alignment: 0.2352
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### syntax / agreement_with_clause
- avg_margin: 5.6250
- preference_correct_pair: True
- d_norm_growth: 37.6612
- mean_rotation_deg: 24.2690
- mid_layer: 14
- mid_d_norm: 98.4416
- avg_abs_mid_alignment: 0.0513
- avg_abs_final_alignment: 0.1006
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### syntax / plural_with_pp
- avg_margin: 4.3125
- preference_correct_pair: True
- d_norm_growth: 26.6452
- mean_rotation_deg: 26.4512
- mid_layer: 14
- mid_d_norm: 110.7187
- avg_abs_mid_alignment: 0.0469
- avg_abs_final_alignment: 0.1689
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### syntax / distance_agreement
- avg_margin: 2.5000
- preference_correct_pair: True
- d_norm_growth: 24.4329
- mean_rotation_deg: 24.1920
- mid_layer: 14
- mid_d_norm: 187.0451
- avg_abs_mid_alignment: 0.0061
- avg_abs_final_alignment: 0.0423
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### syntax / collective_local_noun
- avg_margin: 3.5938
- preference_correct_pair: True
- d_norm_growth: 45.2619
- mean_rotation_deg: 28.5966
- mid_layer: 14
- mid_d_norm: 104.8001
- avg_abs_mid_alignment: 0.0201
- avg_abs_final_alignment: 0.0541
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### relation / capital_relation
- avg_margin: 8.1562
- preference_correct_pair: True
- d_norm_growth: 114.1991
- mean_rotation_deg: 37.0378
- mid_layer: 14
- mid_d_norm: 51.5900
- avg_abs_mid_alignment: 0.0183
- avg_abs_final_alignment: 0.1187
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### relation / author_relation
- avg_margin: 8.3402
- preference_correct_pair: True
- d_norm_growth: 99.1108
- mean_rotation_deg: 36.0534
- mid_layer: 14
- mid_d_norm: 99.9743
- avg_abs_mid_alignment: 0.0163
- avg_abs_final_alignment: 0.0609
- signed_avg_final_alignment: -0.0377
- layer_peak: 27

### relation / chemical_symbol_relation
- avg_margin: 3.8299
- preference_correct_pair: True
- d_norm_growth: 71.3773
- mean_rotation_deg: 33.4817
- mid_layer: 14
- mid_d_norm: 68.2377
- avg_abs_mid_alignment: 0.0254
- avg_abs_final_alignment: 0.1849
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### relation / currency_relation
- avg_margin: 6.5313
- preference_correct_pair: True
- d_norm_growth: 88.1711
- mean_rotation_deg: 32.8677
- mid_layer: 14
- mid_d_norm: 96.6485
- avg_abs_mid_alignment: 0.0006
- avg_abs_final_alignment: 0.0608
- signed_avg_final_alignment: -0.0132
- layer_peak: 27

### relation / inventor_relation
- avg_margin: 2.6250
- preference_correct_pair: False
- d_norm_growth: 53.2659
- mean_rotation_deg: 26.6324
- mid_layer: 14
- mid_d_norm: 200.8224
- avg_abs_mid_alignment: 0.0299
- avg_abs_final_alignment: 0.0131
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### coref / winner_reference
- avg_margin: 3.0312
- preference_correct_pair: True
- d_norm_growth: 285.0316
- mean_rotation_deg: 38.5186
- mid_layer: 14
- mid_d_norm: 34.5226
- avg_abs_mid_alignment: 0.0140
- avg_abs_final_alignment: 0.0782
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### coref / apology_reference
- avg_margin: 2.9063
- preference_correct_pair: True
- d_norm_growth: 350.2437
- mean_rotation_deg: 41.6922
- mid_layer: 14
- mid_d_norm: 30.5863
- avg_abs_mid_alignment: 0.0126
- avg_abs_final_alignment: 0.1126
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### coref / help_reference
- avg_margin: 2.6562
- preference_correct_pair: True
- d_norm_growth: 353.8605
- mean_rotation_deg: 39.6291
- mid_layer: 14
- mid_d_norm: 30.2752
- avg_abs_mid_alignment: 0.0101
- avg_abs_final_alignment: 0.0708
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### coref / congratulation_reference
- avg_margin: 1.2812
- preference_correct_pair: False
- d_norm_growth: 267.0781
- mean_rotation_deg: 39.2374
- mid_layer: 14
- mid_d_norm: 34.4054
- avg_abs_mid_alignment: 0.0051
- avg_abs_final_alignment: 0.0602
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### coref / blame_reference
- avg_margin: 1.8125
- preference_correct_pair: True
- d_norm_growth: 344.1995
- mean_rotation_deg: 39.7471
- mid_layer: 14
- mid_d_norm: 30.4819
- avg_abs_mid_alignment: 0.0095
- avg_abs_final_alignment: 0.0638
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### style / formal_rewrite
- avg_margin: -0.7812
- preference_correct_pair: False
- d_norm_growth: 191.6390
- mean_rotation_deg: 23.6879
- mid_layer: 14
- mid_d_norm: 555.2184
- avg_abs_mid_alignment: 0.0425
- avg_abs_final_alignment: 0.0326
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### style / formal_word_choice
- avg_margin: 1.7812
- preference_correct_pair: False
- d_norm_growth: 64.0641
- mean_rotation_deg: 31.6797
- mid_layer: 14
- mid_d_norm: 102.6861
- avg_abs_mid_alignment: 0.0148
- avg_abs_final_alignment: 0.0463
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### style / formal_request
- avg_margin: 4.0625
- preference_correct_pair: True
- d_norm_growth: 164.1944
- mean_rotation_deg: 33.0382
- mid_layer: 14
- mid_d_norm: 90.9666
- avg_abs_mid_alignment: 0.0223
- avg_abs_final_alignment: 0.0426
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### style / formal_apology
- avg_margin: 6.0000
- preference_correct_pair: True
- d_norm_growth: 39.0381
- mean_rotation_deg: 25.3921
- mid_layer: 14
- mid_d_norm: 161.8083
- avg_abs_mid_alignment: 0.0197
- avg_abs_final_alignment: 0.0663
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

### style / formal_departure
- avg_margin: -0.6875
- preference_correct_pair: False
- d_norm_growth: 43.5152
- mean_rotation_deg: 31.4695
- mid_layer: 14
- mid_d_norm: 100.2500
- avg_abs_mid_alignment: 0.0027
- avg_abs_final_alignment: 0.0027
- signed_avg_final_alignment: 0.0000
- layer_peak: 27

