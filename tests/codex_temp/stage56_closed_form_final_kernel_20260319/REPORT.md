# Stage56 闭式核最终候选摘要

- row_count: 72
- equation_text: core_balance_v3 = positive_mass_v2 - destructive_core - alignment_load_v2; closed_form_kernel_v3 = core_balance_v3 + style_structure_gain; strict_kernel_v3 = closed_form_kernel_v3 + strict_load
- main_judgment: 闭式核已进一步压成核心平衡、严格负载和风格微修正三层，现在可以直接比较一般闭包核和严格闭包核是否需要分成两条主式。

## Stable Features
- core_balance_v3_term: positive
- strict_load_term: negative
- style_structure_gain_term: negative
- closed_form_kernel_v3_term: positive

## Fits
- target: union_joint_adv
  intercept: +0.874940
  core_balance_v3_term: +0.668457
  strict_load_term: -0.921733
  style_structure_gain_term: -0.138113
  closed_form_kernel_v3_term: +0.530344
  strict_kernel_v3_term: -0.391389
- target: union_synergy_joint
  intercept: +0.338896
  core_balance_v3_term: +0.393868
  strict_load_term: -0.505598
  style_structure_gain_term: -0.076780
  closed_form_kernel_v3_term: +0.317088
  strict_kernel_v3_term: -0.188510
- target: strict_positive_synergy
  intercept: -2.480207
  core_balance_v3_term: +2.539459
  strict_load_term: -1.207560
  style_structure_gain_term: -1.222002
  closed_form_kernel_v3_term: +1.317456
  strict_kernel_v3_term: +0.109896
