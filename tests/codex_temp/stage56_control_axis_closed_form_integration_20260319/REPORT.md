# Stage56 控制轴并入闭式核摘要

- row_count: 72
- equation_text: U_closed_ctrl(pair) = closed_form_balance_v2 + alignment_load_v2 + logic_structure_gain + syntax_structure_gain + style_structure_gain
- main_judgment: 闭式核已开始吸收逻辑、句法、风格的稳定控制子通道，可以直接判断控制轴是否更适合作为闭式核的微修正，而不是独立粗总项。

## Stable Features
- closed_form_balance_v2_term: positive
- alignment_load_v2_term: negative
- style_structure_gain_term: positive

## Fits
- target: union_joint_adv
  intercept: -0.287388
  closed_form_balance_v2_term: +0.784607
  alignment_load_v2_term: -0.122275
  logic_structure_gain_term: -0.000056
  syntax_structure_gain_term: +0.000165
  style_structure_gain_term: +0.200427
- target: union_synergy_joint
  intercept: -0.041150
  closed_form_balance_v2_term: +0.608165
  alignment_load_v2_term: -0.383618
  logic_structure_gain_term: +0.000219
  syntax_structure_gain_term: -0.000109
  style_structure_gain_term: +0.062764
- target: strict_positive_synergy
  intercept: +2.725165
  closed_form_balance_v2_term: +1.734152
  alignment_load_v2_term: -2.570933
  logic_structure_gain_term: +0.002709
  syntax_structure_gain_term: -0.000287
  style_structure_gain_term: +0.702820
