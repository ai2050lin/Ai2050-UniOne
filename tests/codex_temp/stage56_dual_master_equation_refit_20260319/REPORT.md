# Stage56 双主式重拟合摘要

- row_count: 72
- equation_text: U_general = kernel_v4; U_strict_module = strict_load + logic_strictload; U_gap = U_general - U_strict_module
- main_judgment: 双主式已经可以写成一般闭包核、严格闭包模块和二者的边距三层，当前最关键的问题是一般核与严格模块是否已经足够分离到能独立成式。

## Stable Features
- kernel_v4_term: positive
- strict_module_term: negative
- dual_gap_term: positive

## Fits
- target: union_joint_adv
  intercept: +0.416152
  kernel_v4_term: +0.087612
  strict_module_term: -0.087227
  dual_gap_term: +0.174063
- target: union_synergy_joint
  intercept: +0.197953
  kernel_v4_term: +0.058361
  strict_module_term: -0.058158
  dual_gap_term: +0.116001
- target: strict_positive_synergy
  intercept: +1.585040
  kernel_v4_term: +0.224351
  strict_module_term: -0.222594
  dual_gap_term: +0.444966
