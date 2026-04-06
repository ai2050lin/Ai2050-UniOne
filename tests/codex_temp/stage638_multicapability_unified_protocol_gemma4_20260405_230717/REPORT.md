# Stage638 多能力统一编码协议报告

- 时间: 2026-04-05 23:07:36
- 模型: gemma4
- 能力范围: all
- 样本数: 25

## 总览
- overall_pair_accuracy: 0.3200
- overall_mean_margin: 0.5762
- overall_mean_d_norm_growth: 3.4841
- overall_mean_rotation_deg: 52.8960
- overall_mean_mid_d_norm: 13.7919
- overall_mean_abs_mid_alignment: 0.0215
- overall_mean_abs_final_alignment: 0.0296
- overall_mean_signed_final_alignment: -0.0015

## 分能力摘要
### coref
- count: 5
- pair_accuracy: 0.6000
- mean_margin: 1.2375
- mean_d_norm_growth: 10.7001
- mean_rotation_deg: 53.8502
- mean_mid_d_norm: 8.6852
- mean_abs_mid_alignment: 0.0231
- mean_abs_final_alignment: 0.0617
- mean_signed_final_alignment: 0.0000
- mean_mid_layer: 17.0000
- mean_peak_layer: 16.4000

### disamb
- count: 5
- pair_accuracy: 0.0000
- mean_margin: 0.5531
- mean_d_norm_growth: 2.8828
- mean_rotation_deg: 53.4060
- mean_mid_d_norm: 15.0749
- mean_abs_mid_alignment: 0.0092
- mean_abs_final_alignment: 0.0156
- mean_signed_final_alignment: 0.0000
- mean_mid_layer: 17.0000
- mean_peak_layer: 13.2000

### relation
- count: 5
- pair_accuracy: 0.6000
- mean_margin: 0.6652
- mean_d_norm_growth: 1.7225
- mean_rotation_deg: 52.0591
- mean_mid_d_norm: 16.7842
- mean_abs_mid_alignment: 0.0256
- mean_abs_final_alignment: 0.0278
- mean_signed_final_alignment: -0.0075
- mean_mid_layer: 17.0000
- mean_peak_layer: 16.4000

### style
- count: 5
- pair_accuracy: 0.4000
- mean_margin: 0.6188
- mean_d_norm_growth: 1.0331
- mean_rotation_deg: 52.9731
- mean_mid_d_norm: 14.3274
- mean_abs_mid_alignment: 0.0276
- mean_abs_final_alignment: 0.0095
- mean_signed_final_alignment: 0.0000
- mean_mid_layer: 17.0000
- mean_peak_layer: 15.6000

### syntax
- count: 5
- pair_accuracy: 0.0000
- mean_margin: -0.1938
- mean_d_norm_growth: 1.0818
- mean_rotation_deg: 52.1916
- mean_mid_d_norm: 14.0879
- mean_abs_mid_alignment: 0.0220
- mean_abs_final_alignment: 0.0333
- mean_signed_final_alignment: 0.0000
- mean_mid_layer: 17.0000
- mean_peak_layer: 20.8000

## 单样本明细
### disamb / bank_meaning
- avg_margin: 0.0781
- preference_correct_pair: False
- d_norm_growth: 2.0727
- mean_rotation_deg: 53.0521
- mid_layer: 17
- mid_d_norm: 15.5851
- avg_abs_mid_alignment: 0.0053
- avg_abs_final_alignment: 0.0241
- signed_avg_final_alignment: 0.0000
- layer_peak: 10

### disamb / plant_meaning
- avg_margin: -0.4375
- preference_correct_pair: False
- d_norm_growth: 3.6526
- mean_rotation_deg: 53.5438
- mid_layer: 17
- mid_d_norm: 10.4943
- avg_abs_mid_alignment: 0.0059
- avg_abs_final_alignment: 0.0139
- signed_avg_final_alignment: 0.0000
- layer_peak: 24

### disamb / bat_meaning
- avg_margin: 1.0312
- preference_correct_pair: False
- d_norm_growth: 2.4702
- mean_rotation_deg: 53.4353
- mid_layer: 17
- mid_d_norm: 14.1798
- avg_abs_mid_alignment: 0.0104
- avg_abs_final_alignment: 0.0032
- signed_avg_final_alignment: 0.0000
- layer_peak: 11

### disamb / watch_meaning
- avg_margin: 1.4688
- preference_correct_pair: False
- d_norm_growth: 2.9261
- mean_rotation_deg: 53.5117
- mid_layer: 17
- mid_d_norm: 18.5387
- avg_abs_mid_alignment: 0.0211
- avg_abs_final_alignment: 0.0139
- signed_avg_final_alignment: 0.0000
- layer_peak: 11

### disamb / light_meaning
- avg_margin: 0.6250
- preference_correct_pair: False
- d_norm_growth: 3.2925
- mean_rotation_deg: 53.4871
- mid_layer: 17
- mid_d_norm: 16.5766
- avg_abs_mid_alignment: 0.0032
- avg_abs_final_alignment: 0.0230
- signed_avg_final_alignment: 0.0000
- layer_peak: 10

### syntax / subject_verb_number
- avg_margin: 1.8125
- preference_correct_pair: False
- d_norm_growth: 2.6042
- mean_rotation_deg: 53.3761
- mid_layer: 17
- mid_d_norm: 11.0858
- avg_abs_mid_alignment: 0.0238
- avg_abs_final_alignment: 0.0181
- signed_avg_final_alignment: 0.0000
- layer_peak: 8

### syntax / agreement_with_clause
- avg_margin: -0.6562
- preference_correct_pair: False
- d_norm_growth: 0.3832
- mean_rotation_deg: 51.0924
- mid_layer: 17
- mid_d_norm: 10.0418
- avg_abs_mid_alignment: 0.0145
- avg_abs_final_alignment: 0.0358
- signed_avg_final_alignment: 0.0000
- layer_peak: 24

### syntax / plural_with_pp
- avg_margin: -1.4375
- preference_correct_pair: False
- d_norm_growth: 0.4899
- mean_rotation_deg: 52.4755
- mid_layer: 17
- mid_d_norm: 13.1965
- avg_abs_mid_alignment: 0.0079
- avg_abs_final_alignment: 0.0339
- signed_avg_final_alignment: 0.0000
- layer_peak: 24

### syntax / distance_agreement
- avg_margin: 0.5625
- preference_correct_pair: False
- d_norm_growth: 0.9323
- mean_rotation_deg: 51.4643
- mid_layer: 17
- mid_d_norm: 23.8403
- avg_abs_mid_alignment: 0.0425
- avg_abs_final_alignment: 0.0319
- signed_avg_final_alignment: 0.0000
- layer_peak: 24

### syntax / collective_local_noun
- avg_margin: -1.2500
- preference_correct_pair: False
- d_norm_growth: 0.9993
- mean_rotation_deg: 52.5500
- mid_layer: 17
- mid_d_norm: 12.2751
- avg_abs_mid_alignment: 0.0214
- avg_abs_final_alignment: 0.0469
- signed_avg_final_alignment: 0.0000
- layer_peak: 24

### relation / capital_relation
- avg_margin: 2.4336
- preference_correct_pair: True
- d_norm_growth: 5.0308
- mean_rotation_deg: 50.4925
- mid_layer: 17
- mid_d_norm: 15.6444
- avg_abs_mid_alignment: 0.0366
- avg_abs_final_alignment: 0.0007
- signed_avg_final_alignment: 0.0000
- layer_peak: 25

### relation / author_relation
- avg_margin: 0.0625
- preference_correct_pair: False
- d_norm_growth: 0.4407
- mean_rotation_deg: 51.5370
- mid_layer: 17
- mid_d_norm: 18.7236
- avg_abs_mid_alignment: 0.0450
- avg_abs_final_alignment: 0.0273
- signed_avg_final_alignment: -0.0133
- layer_peak: 9

### relation / chemical_symbol_relation
- avg_margin: 0.9551
- preference_correct_pair: True
- d_norm_growth: 0.5861
- mean_rotation_deg: 53.9443
- mid_layer: 17
- mid_d_norm: 11.8370
- avg_abs_mid_alignment: 0.0071
- avg_abs_final_alignment: 0.0251
- signed_avg_final_alignment: 0.0000
- layer_peak: 29

### relation / currency_relation
- avg_margin: 0.8750
- preference_correct_pair: True
- d_norm_growth: 1.9813
- mean_rotation_deg: 53.3336
- mid_layer: 17
- mid_d_norm: 15.8632
- avg_abs_mid_alignment: 0.0264
- avg_abs_final_alignment: 0.0243
- signed_avg_final_alignment: -0.0243
- layer_peak: 15

### relation / inventor_relation
- avg_margin: -1.0000
- preference_correct_pair: False
- d_norm_growth: 0.5734
- mean_rotation_deg: 50.9880
- mid_layer: 17
- mid_d_norm: 21.8529
- avg_abs_mid_alignment: 0.0128
- avg_abs_final_alignment: 0.0615
- signed_avg_final_alignment: 0.0000
- layer_peak: 4

### coref / winner_reference
- avg_margin: 2.3750
- preference_correct_pair: True
- d_norm_growth: 5.3593
- mean_rotation_deg: 53.8120
- mid_layer: 17
- mid_d_norm: 10.2136
- avg_abs_mid_alignment: 0.0061
- avg_abs_final_alignment: 0.1336
- signed_avg_final_alignment: 0.0000
- layer_peak: 19

### coref / apology_reference
- avg_margin: 1.2500
- preference_correct_pair: False
- d_norm_growth: 17.2236
- mean_rotation_deg: 54.5879
- mid_layer: 17
- mid_d_norm: 7.3473
- avg_abs_mid_alignment: 0.0278
- avg_abs_final_alignment: 0.0003
- signed_avg_final_alignment: 0.0000
- layer_peak: 19

### coref / help_reference
- avg_margin: 0.8750
- preference_correct_pair: True
- d_norm_growth: 20.1005
- mean_rotation_deg: 53.5997
- mid_layer: 17
- mid_d_norm: 10.0548
- avg_abs_mid_alignment: 0.0252
- avg_abs_final_alignment: 0.0375
- signed_avg_final_alignment: 0.0000
- layer_peak: 9

### coref / congratulation_reference
- avg_margin: 1.0000
- preference_correct_pair: False
- d_norm_growth: 3.2806
- mean_rotation_deg: 53.1842
- mid_layer: 17
- mid_d_norm: 7.1630
- avg_abs_mid_alignment: 0.0029
- avg_abs_final_alignment: 0.0717
- signed_avg_final_alignment: 0.0000
- layer_peak: 11

### coref / blame_reference
- avg_margin: 0.6875
- preference_correct_pair: True
- d_norm_growth: 7.5366
- mean_rotation_deg: 54.0672
- mid_layer: 17
- mid_d_norm: 8.6471
- avg_abs_mid_alignment: 0.0534
- avg_abs_final_alignment: 0.0656
- signed_avg_final_alignment: 0.0000
- layer_peak: 24

### style / formal_rewrite
- avg_margin: -1.2188
- preference_correct_pair: False
- d_norm_growth: 2.7970
- mean_rotation_deg: 56.0984
- mid_layer: 17
- mid_d_norm: 10.4451
- avg_abs_mid_alignment: 0.0794
- avg_abs_final_alignment: 0.0099
- signed_avg_final_alignment: 0.0000
- layer_peak: 11

### style / formal_word_choice
- avg_margin: 0.5625
- preference_correct_pair: False
- d_norm_growth: 0.5500
- mean_rotation_deg: 52.3638
- mid_layer: 17
- mid_d_norm: 11.9701
- avg_abs_mid_alignment: 0.0341
- avg_abs_final_alignment: 0.0049
- signed_avg_final_alignment: 0.0000
- layer_peak: 25

### style / formal_request
- avg_margin: 1.0000
- preference_correct_pair: True
- d_norm_growth: 0.8066
- mean_rotation_deg: 52.8728
- mid_layer: 17
- mid_d_norm: 15.6691
- avg_abs_mid_alignment: 0.0082
- avg_abs_final_alignment: 0.0140
- signed_avg_final_alignment: 0.0000
- layer_peak: 9

### style / formal_apology
- avg_margin: 2.0000
- preference_correct_pair: True
- d_norm_growth: 0.5130
- mean_rotation_deg: 49.7340
- mid_layer: 17
- mid_d_norm: 18.9284
- avg_abs_mid_alignment: 0.0143
- avg_abs_final_alignment: 0.0017
- signed_avg_final_alignment: 0.0000
- layer_peak: 24

### style / formal_departure
- avg_margin: 0.7500
- preference_correct_pair: False
- d_norm_growth: 0.4988
- mean_rotation_deg: 53.7964
- mid_layer: 17
- mid_d_norm: 14.6244
- avg_abs_mid_alignment: 0.0019
- avg_abs_final_alignment: 0.0171
- signed_avg_final_alignment: 0.0000
- layer_peak: 9

