# Stage56 style 细化摘要

- row_count: 72
- main_judgment: style 已经从粗总项拆成中段、对齐、重排压力和差值等细通道，可以直接检查风格到底在哪些局部子通道上表现出稳定方向。

## Stable Features
- style_compaction_mid: negative
- style_coverage_mid: negative
- style_delta_l2: negative
- style_delta_mean_abs: positive
- style_role_align_compaction: positive
- style_role_align_coverage: negative
- style_midfield: negative
- style_alignment: positive
- style_reorder_pressure: positive
- style_gap: positive

## Fits
- target: union_joint_adv
  intercept: -74.309486
  style_compaction_mid: -4.978767
  style_coverage_mid: -1.731222
  style_delta_l2: -0.232468
  style_delta_mean_abs: +1.154779
  style_role_align_compaction: +62.811781
  style_role_align_coverage: -12.690671
  style_midfield: -3.354994
  style_alignment: +25.060556
  style_reorder_pressure: +0.462122
  style_gap: +3.247546
- target: union_synergy_joint
  intercept: -30.445899
  style_compaction_mid: -1.610226
  style_coverage_mid: -0.809590
  style_delta_l2: -0.192381
  style_delta_mean_abs: +0.956014
  style_role_align_compaction: +26.118256
  style_role_align_coverage: -5.361745
  style_midfield: -1.209908
  style_alignment: +10.378256
  style_reorder_pressure: +0.382703
  style_gap: +0.800635
- target: strict_positive_synergy
  intercept: -50.999742
  style_compaction_mid: -16.856311
  style_coverage_mid: -6.058009
  style_delta_l2: -1.667834
  style_delta_mean_abs: +8.286319
  style_role_align_compaction: +45.937437
  style_role_align_coverage: -9.944756
  style_midfield: -11.457156
  style_alignment: +17.996344
  style_reorder_pressure: +3.317458
  style_gap: +10.798293
