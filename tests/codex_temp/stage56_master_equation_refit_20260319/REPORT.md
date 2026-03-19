# Stage56 主方程重拟合摘要

- row_count: 72
- equation_text: U_refit(pair) = a1 * identity_margin + a2 * frontier + a3 * logic_prototype + a4 * logic_fragile_bridge + a5 * syntax_constraint_conflict + a6 * window_dominance + a7 * style_alignment + a8 * style_midfield + a9 * logic_control
- main_judgment: 主方程重拟合已经把静态边距、窗口主导性、稳定子场和细化后的 style 子通道并到同一条样本级方程里，可以直接检查新变量是否比旧粗代理更稳定。

## Stable Features
- identity_margin_term: positive
- logic_fragile_bridge_term: negative
- syntax_constraint_conflict_term: positive
- style_alignment_term: negative

## Fits
- target: union_joint_adv
  intercept: +0.777185
  identity_margin_term: +2.651744
  frontier_term: -3.645925
  logic_prototype_term: -0.993690
  logic_fragile_bridge_term: -5.586713
  syntax_constraint_conflict_term: +124.850250
  window_dominance_term: -0.562983
  style_alignment_term: -1.236739
  style_midfield_term: +0.926652
  logic_control_term: +3.009734
- target: union_synergy_joint
  intercept: +0.490411
  identity_margin_term: +2.122987
  frontier_term: -4.345204
  logic_prototype_term: -1.107829
  logic_fragile_bridge_term: -8.751957
  syntax_constraint_conflict_term: +12.812786
  window_dominance_term: -0.158695
  style_alignment_term: -0.795941
  style_midfield_term: +1.385599
  logic_control_term: +4.528377
- target: strict_positive_synergy
  intercept: -6.000322
  identity_margin_term: +4.089743
  frontier_term: +8.322268
  logic_prototype_term: +11.409293
  logic_fragile_bridge_term: -39.050580
  syntax_constraint_conflict_term: +99.186423
  window_dominance_term: +1.324597
  style_alignment_term: -1.042620
  style_midfield_term: -1.354224
  logic_control_term: -0.014112
