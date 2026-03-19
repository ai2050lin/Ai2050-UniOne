# Stage56 混合项拆分摘要

- row_count: 72
- main_judgment: logic_prototype 与 window_dominance 已被拆成耦合子项，可以直接检查它们到底是通过身份边距、前沿还是风格子通道进入混合态。

## Stable Features
- logic_prototype_margin_term: positive
- logic_prototype_frontier_term: negative
- logic_prototype_syntax_term: positive

## Fits
- target: union_joint_adv
  intercept: -0.160478
  logic_prototype_margin_term: +8.640942
  logic_prototype_frontier_term: -7.663138
  logic_prototype_syntax_term: +2.034723
  window_dominance_style_alignment_term: +1.240275
  window_dominance_style_midfield_term: +0.487850
  window_dominance_frontier_term: -2.209517
- target: union_synergy_joint
  intercept: -0.101223
  logic_prototype_margin_term: +13.757961
  logic_prototype_frontier_term: -14.578746
  logic_prototype_syntax_term: +0.238309
  window_dominance_style_alignment_term: +0.569141
  window_dominance_style_midfield_term: +1.745978
  window_dominance_frontier_term: -2.601522
- target: strict_positive_synergy
  intercept: -1.059376
  logic_prototype_margin_term: +50.600380
  logic_prototype_frontier_term: -5.763927
  logic_prototype_syntax_term: +1.591972
  window_dominance_style_alignment_term: -3.341070
  window_dominance_style_midfield_term: -4.226102
  window_dominance_frontier_term: +15.248317
