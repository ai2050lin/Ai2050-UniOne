# Stage56 层间耦合重拟合摘要

- row_count: 72
- equation_text: gs_coupling = kernel_v4 * strict_module_final; gd_coupling = kernel_v4 * dual_gap_final; sd_coupling = strict_module_final * dual_gap_final
- main_judgment: 层间耦合已经开始分化成稳定正耦合与稳定负耦合，可以直接判断三层不是松散并列，而是存在方向不同的结构性交互。

## Stable Features
- gd_coupling_term: positive

## Fits
- target: union_joint_adv
  intercept: +0.563979
  gs_coupling_term: -0.107183
  gd_coupling_term: +0.000001
  sd_coupling_term: +0.108119
- target: union_synergy_joint
  intercept: +0.292259
  gs_coupling_term: +0.226442
  gd_coupling_term: +0.000003
  sd_coupling_term: -0.224203
- target: strict_positive_synergy
  intercept: -0.596824
  gs_coupling_term: +1.520807
  gd_coupling_term: +0.000005
  sd_coupling_term: -1.516108
