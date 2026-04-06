# Stage638 多能力统一编码协议报告

- 时间: 2026-04-05 23:08:12
- 模型: glm4
- 能力范围: all
- 样本数: 25

## 总览
- overall_pair_accuracy: 0.8000
- overall_mean_margin: 4.5284
- overall_mean_d_norm_growth: 5848.2547
- overall_mean_rotation_deg: 33.5151
- overall_mean_mid_d_norm: 5.0533
- overall_mean_abs_mid_alignment: 0.0131
- overall_mean_abs_final_alignment: 0.0974
- overall_mean_signed_final_alignment: -0.0020

## 分能力摘要
### coref
- count: 5
- pair_accuracy: 0.8000
- mean_margin: 3.7813
- mean_d_norm_growth: 9184.3429
- mean_rotation_deg: 37.9360
- mean_mid_d_norm: 3.2665
- mean_abs_mid_alignment: 0.0050
- mean_abs_final_alignment: 0.1072
- mean_signed_final_alignment: 0.0000
- mean_mid_layer: 20.0000
- mean_peak_layer: 39.0000

### disamb
- count: 5
- pair_accuracy: 0.8000
- mean_margin: 5.5336
- mean_d_norm_growth: 14645.3129
- mean_rotation_deg: 37.8909
- mean_mid_d_norm: 4.4095
- mean_abs_mid_alignment: 0.0106
- mean_abs_final_alignment: 0.0607
- mean_signed_final_alignment: 0.0000
- mean_mid_layer: 20.0000
- mean_peak_layer: 39.0000

### relation
- count: 5
- pair_accuracy: 1.0000
- mean_margin: 7.6803
- mean_d_norm_growth: 2609.8683
- mean_rotation_deg: 33.5620
- mean_mid_d_norm: 5.1147
- mean_abs_mid_alignment: 0.0154
- mean_abs_final_alignment: 0.1307
- mean_signed_final_alignment: -0.0098
- mean_mid_layer: 20.0000
- mean_peak_layer: 39.0000

### style
- count: 5
- pair_accuracy: 0.4000
- mean_margin: 1.8281
- mean_d_norm_growth: 2428.6435
- mean_rotation_deg: 31.8021
- mean_mid_d_norm: 6.0151
- mean_abs_mid_alignment: 0.0147
- mean_abs_final_alignment: 0.0540
- mean_signed_final_alignment: 0.0000
- mean_mid_layer: 20.0000
- mean_peak_layer: 39.0000

### syntax
- count: 5
- pair_accuracy: 1.0000
- mean_margin: 3.8188
- mean_d_norm_growth: 373.1056
- mean_rotation_deg: 26.3844
- mean_mid_d_norm: 6.4609
- mean_abs_mid_alignment: 0.0197
- mean_abs_final_alignment: 0.1344
- mean_signed_final_alignment: 0.0000
- mean_mid_layer: 20.0000
- mean_peak_layer: 39.0000

## 单样本明细
### disamb / bank_meaning
- avg_margin: 5.1211
- preference_correct_pair: True
- d_norm_growth: 12006.6763
- mean_rotation_deg: 36.8234
- mid_layer: 20
- mid_d_norm: 4.6274
- avg_abs_mid_alignment: 0.0222
- avg_abs_final_alignment: 0.0514
- signed_avg_final_alignment: 0.0000
- layer_peak: 39

### disamb / plant_meaning
- avg_margin: 7.9043
- preference_correct_pair: True
- d_norm_growth: 24088.9395
- mean_rotation_deg: 39.0432
- mid_layer: 20
- mid_d_norm: 4.0545
- avg_abs_mid_alignment: 0.0016
- avg_abs_final_alignment: 0.0955
- signed_avg_final_alignment: 0.0000
- layer_peak: 39

### disamb / bat_meaning
- avg_margin: 4.5156
- preference_correct_pair: True
- d_norm_growth: 13200.1256
- mean_rotation_deg: 37.5490
- mid_layer: 20
- mid_d_norm: 4.2429
- avg_abs_mid_alignment: 0.0137
- avg_abs_final_alignment: 0.0399
- signed_avg_final_alignment: 0.0000
- layer_peak: 39

### disamb / watch_meaning
- avg_margin: 3.8846
- preference_correct_pair: False
- d_norm_growth: 8787.1079
- mean_rotation_deg: 37.5687
- mid_layer: 20
- mid_d_norm: 4.5320
- avg_abs_mid_alignment: 0.0066
- avg_abs_final_alignment: 0.0483
- signed_avg_final_alignment: 0.0000
- layer_peak: 39

### disamb / light_meaning
- avg_margin: 6.2422
- preference_correct_pair: True
- d_norm_growth: 15143.7155
- mean_rotation_deg: 38.4704
- mid_layer: 20
- mid_d_norm: 4.5905
- avg_abs_mid_alignment: 0.0087
- avg_abs_final_alignment: 0.0685
- signed_avg_final_alignment: 0.0000
- layer_peak: 39

### syntax / subject_verb_number
- avg_margin: 4.9688
- preference_correct_pair: True
- d_norm_growth: 1028.8767
- mean_rotation_deg: 29.8063
- mid_layer: 20
- mid_d_norm: 5.3825
- avg_abs_mid_alignment: 0.0032
- avg_abs_final_alignment: 0.2569
- signed_avg_final_alignment: 0.0000
- layer_peak: 39

### syntax / agreement_with_clause
- avg_margin: 3.6875
- preference_correct_pair: True
- d_norm_growth: 128.7849
- mean_rotation_deg: 23.8874
- mid_layer: 20
- mid_d_norm: 5.2627
- avg_abs_mid_alignment: 0.0117
- avg_abs_final_alignment: 0.1407
- signed_avg_final_alignment: 0.0000
- layer_peak: 39

### syntax / plural_with_pp
- avg_margin: 2.8438
- preference_correct_pair: True
- d_norm_growth: 221.9313
- mean_rotation_deg: 27.3137
- mid_layer: 20
- mid_d_norm: 5.2308
- avg_abs_mid_alignment: 0.0449
- avg_abs_final_alignment: 0.1006
- signed_avg_final_alignment: 0.0000
- layer_peak: 39

### syntax / distance_agreement
- avg_margin: 3.9688
- preference_correct_pair: True
- d_norm_growth: 113.4701
- mean_rotation_deg: 22.7764
- mid_layer: 20
- mid_d_norm: 10.6487
- avg_abs_mid_alignment: 0.0266
- avg_abs_final_alignment: 0.0762
- signed_avg_final_alignment: 0.0000
- layer_peak: 39

### syntax / collective_local_noun
- avg_margin: 3.6250
- preference_correct_pair: True
- d_norm_growth: 372.4650
- mean_rotation_deg: 28.1384
- mid_layer: 20
- mid_d_norm: 5.7799
- avg_abs_mid_alignment: 0.0119
- avg_abs_final_alignment: 0.0977
- signed_avg_final_alignment: 0.0000
- layer_peak: 39

### relation / capital_relation
- avg_margin: 6.7656
- preference_correct_pair: True
- d_norm_growth: 2257.0541
- mean_rotation_deg: 35.6020
- mid_layer: 20
- mid_d_norm: 1.8861
- avg_abs_mid_alignment: 0.0128
- avg_abs_final_alignment: 0.2488
- signed_avg_final_alignment: 0.0000
- layer_peak: 39

### relation / author_relation
- avg_margin: 11.7668
- preference_correct_pair: True
- d_norm_growth: 2064.3949
- mean_rotation_deg: 34.8701
- mid_layer: 20
- mid_d_norm: 4.3554
- avg_abs_mid_alignment: 0.0268
- avg_abs_final_alignment: 0.0989
- signed_avg_final_alignment: -0.0167
- layer_peak: 39

### relation / chemical_symbol_relation
- avg_margin: 4.5644
- preference_correct_pair: True
- d_norm_growth: 3289.8000
- mean_rotation_deg: 34.1482
- mid_layer: 20
- mid_d_norm: 4.1109
- avg_abs_mid_alignment: 0.0075
- avg_abs_final_alignment: 0.1289
- signed_avg_final_alignment: 0.0000
- layer_peak: 39

### relation / currency_relation
- avg_margin: 8.6250
- preference_correct_pair: True
- d_norm_growth: 4308.9856
- mean_rotation_deg: 34.6431
- mid_layer: 20
- mid_d_norm: 4.2122
- avg_abs_mid_alignment: 0.0201
- avg_abs_final_alignment: 0.1236
- signed_avg_final_alignment: -0.0325
- layer_peak: 39

### relation / inventor_relation
- avg_margin: 6.6797
- preference_correct_pair: True
- d_norm_growth: 1129.1071
- mean_rotation_deg: 28.5468
- mid_layer: 20
- mid_d_norm: 11.0090
- avg_abs_mid_alignment: 0.0098
- avg_abs_final_alignment: 0.0534
- signed_avg_final_alignment: 0.0000
- layer_peak: 39

### coref / winner_reference
- avg_margin: 4.6250
- preference_correct_pair: True
- d_norm_growth: 9956.1300
- mean_rotation_deg: 36.5576
- mid_layer: 20
- mid_d_norm: 2.7728
- avg_abs_mid_alignment: 0.0074
- avg_abs_final_alignment: 0.0446
- signed_avg_final_alignment: 0.0000
- layer_peak: 39

### coref / apology_reference
- avg_margin: 2.6875
- preference_correct_pair: True
- d_norm_growth: 12470.9372
- mean_rotation_deg: 38.9049
- mid_layer: 20
- mid_d_norm: 4.0624
- avg_abs_mid_alignment: 0.0025
- avg_abs_final_alignment: 0.1121
- signed_avg_final_alignment: 0.0000
- layer_peak: 39

### coref / help_reference
- avg_margin: 6.2813
- preference_correct_pair: True
- d_norm_growth: 8959.9512
- mean_rotation_deg: 38.7309
- mid_layer: 20
- mid_d_norm: 3.0501
- avg_abs_mid_alignment: 0.0055
- avg_abs_final_alignment: 0.1389
- signed_avg_final_alignment: 0.0000
- layer_peak: 39

### coref / congratulation_reference
- avg_margin: 2.4375
- preference_correct_pair: False
- d_norm_growth: 6972.1377
- mean_rotation_deg: 37.5427
- mid_layer: 20
- mid_d_norm: 3.2033
- avg_abs_mid_alignment: 0.0009
- avg_abs_final_alignment: 0.1845
- signed_avg_final_alignment: 0.0000
- layer_peak: 39

### coref / blame_reference
- avg_margin: 2.8750
- preference_correct_pair: True
- d_norm_growth: 7562.5584
- mean_rotation_deg: 37.9437
- mid_layer: 20
- mid_d_norm: 3.2437
- avg_abs_mid_alignment: 0.0086
- avg_abs_final_alignment: 0.0560
- signed_avg_final_alignment: 0.0000
- layer_peak: 39

### style / formal_rewrite
- avg_margin: -2.0625
- preference_correct_pair: False
- d_norm_growth: 4555.2297
- mean_rotation_deg: 36.0288
- mid_layer: 20
- mid_d_norm: 5.0020
- avg_abs_mid_alignment: 0.0148
- avg_abs_final_alignment: 0.0206
- signed_avg_final_alignment: 0.0000
- layer_peak: 39

### style / formal_word_choice
- avg_margin: 0.8750
- preference_correct_pair: False
- d_norm_growth: 1580.9102
- mean_rotation_deg: 32.2001
- mid_layer: 20
- mid_d_norm: 4.8074
- avg_abs_mid_alignment: 0.0055
- avg_abs_final_alignment: 0.0513
- signed_avg_final_alignment: 0.0000
- layer_peak: 39

### style / formal_request
- avg_margin: 3.4688
- preference_correct_pair: True
- d_norm_growth: 4595.7604
- mean_rotation_deg: 32.6484
- mid_layer: 20
- mid_d_norm: 5.6103
- avg_abs_mid_alignment: 0.0080
- avg_abs_final_alignment: 0.1190
- signed_avg_final_alignment: 0.0000
- layer_peak: 39

### style / formal_apology
- avg_margin: 6.1094
- preference_correct_pair: True
- d_norm_growth: 475.2391
- mean_rotation_deg: 26.6135
- mid_layer: 20
- mid_d_norm: 9.1709
- avg_abs_mid_alignment: 0.0269
- avg_abs_final_alignment: 0.0687
- signed_avg_final_alignment: 0.0000
- layer_peak: 39

### style / formal_departure
- avg_margin: 0.7500
- preference_correct_pair: False
- d_norm_growth: 936.0780
- mean_rotation_deg: 31.5198
- mid_layer: 20
- mid_d_norm: 5.4847
- avg_abs_mid_alignment: 0.0183
- avg_abs_final_alignment: 0.0106
- signed_avg_final_alignment: 0.0000
- layer_peak: 39

