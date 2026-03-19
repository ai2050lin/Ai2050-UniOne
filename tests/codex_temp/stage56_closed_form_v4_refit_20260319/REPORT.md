# Stage56 第四版闭式核重拟合摘要

- row_count: 72
- equation_text: style_penalty = -style_structure_gain; general_balance_v4 = core_balance_v3 + logic_strictload; kernel_v4 = general_balance_v4 - style_penalty
- main_judgment: 第四版一般闭包核已经被压成核心平衡、逻辑严格负载修正和风格惩罚三层，当前最关键的问题是这一版短核是否足够替代上一版更长的闭式候选。

## Stable Features
- style_penalty_term: negative
- general_balance_v4_term: negative
- kernel_v4_term: positive

## Fits
- target: union_joint_adv
  intercept: +0.322663
  style_penalty_term: -0.130348
  general_balance_v4_term: -0.064963
  kernel_v4_term: +0.065353
- target: union_synergy_joint
  intercept: +0.137972
  style_penalty_term: -0.088619
  general_balance_v4_term: -0.044195
  kernel_v4_term: +0.044402
- target: strict_positive_synergy
  intercept: +1.371156
  style_penalty_term: -0.352075
  general_balance_v4_term: -0.175101
  kernel_v4_term: +0.176890
