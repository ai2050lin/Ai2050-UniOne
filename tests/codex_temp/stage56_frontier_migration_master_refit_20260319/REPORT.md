# Stage56 前沿迁移并场重拟合摘要

- row_count: 72
- equation_text: U_gate_refit(pair) = b1 * identity_margin + b2 * syntax_constraint_conflict + b3 * logic_fragile_bridge + b4 * style_alignment + b5 * frontier_positive_migration + b6 * frontier_negative_base + b7 * window_gate_positive + b8 * window_gate_negative
- main_judgment: 主方程已把粗前沿项和粗窗口项替换成迁移型前沿与条件门型窗口子项，可以直接判断主方程是否进一步变短、变稳。

## Stable Features
- identity_margin_term: negative
- syntax_constraint_conflict_term: positive
- style_alignment_term: positive
- frontier_negative_base_term: negative
- window_gate_positive_term: positive
- window_gate_negative_term: negative

## Fits
- target: union_joint_adv
  intercept: +0.558867
  identity_margin_term: -0.784457
  syntax_constraint_conflict_term: +116.075312
  logic_fragile_bridge_term: +1.219418
  style_alignment_term: +0.946315
  frontier_positive_migration_term: -0.282825
  frontier_negative_base_term: -0.314955
  window_gate_positive_term: +10.234542
  window_gate_negative_term: -8.578716
- target: union_synergy_joint
  intercept: +0.203220
  identity_margin_term: -0.336679
  syntax_constraint_conflict_term: +8.922763
  logic_fragile_bridge_term: -5.277581
  style_alignment_term: +0.374493
  frontier_positive_migration_term: -0.172834
  frontier_negative_base_term: -0.109840
  window_gate_positive_term: +5.504783
  window_gate_negative_term: -4.258378
- target: strict_positive_synergy
  intercept: -5.491633
  identity_margin_term: -7.373646
  syntax_constraint_conflict_term: +56.369925
  logic_fragile_bridge_term: -14.874498
  style_alignment_term: +6.425834
  frontier_positive_migration_term: +4.185869
  frontier_negative_base_term: -0.727246
  window_gate_positive_term: +44.684985
  window_gate_negative_term: -30.035894
