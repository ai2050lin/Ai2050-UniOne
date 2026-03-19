# Stage56 层间耦合规范化摘要

- row_count: 72
- equation_text: gs_load = -gs_coupling; gd_drive = gd_coupling; sd_load = -sd_coupling
- main_judgment: 层间耦合在规范化之后开始显式分成负载通道与驱动通道，这比直接使用原始乘积项更接近当前分层双主式的真实结构。

## Stable Features
- gd_drive_channel_term: positive

## Fits
- target: union_joint_adv
  intercept: +0.563979
  gs_load_channel_term: +0.107183
  gd_drive_channel_term: +0.000001
  sd_load_channel_term: -0.108119
- target: union_synergy_joint
  intercept: +0.292259
  gs_load_channel_term: -0.226442
  gd_drive_channel_term: +0.000003
  sd_load_channel_term: +0.224203
- target: strict_positive_synergy
  intercept: -0.596824
  gs_load_channel_term: -1.520807
  gd_drive_channel_term: +0.000005
  sd_load_channel_term: +1.516108
