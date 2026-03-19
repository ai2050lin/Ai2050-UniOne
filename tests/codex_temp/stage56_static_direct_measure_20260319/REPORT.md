# Stage56 静态项直测摘要

- row_count: 72
- mean_family_patch_direct: +0.735762
- mean_concept_offset_direct: +0.008322
- mean_identity_margin_direct: +0.727440
- main_judgment: 静态本体层已经从摘要代理进一步推进到类别局部直测，现在可以直接检查 family patch 与 concept offset 在样本级的局部测度。

## Fits
- target: union_joint_adv
  intercept: +0.085779
  family_patch_direct: -1.741366
  concept_offset_direct: -3.459856
  identity_margin_direct: +1.718490
- target: union_synergy_joint
  intercept: +0.183358
  family_patch_direct: -1.247720
  concept_offset_direct: -2.235795
  identity_margin_direct: +0.988074
- target: strict_positive_synergy
  intercept: -2.937479
  family_patch_direct: -0.293062
  concept_offset_direct: -4.951524
  identity_margin_direct: +4.658462
