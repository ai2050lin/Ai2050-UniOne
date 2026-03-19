# Stage56 闭式核重拟合摘要

- row_count: 72
- equation_text: C_v2(pair) = positive_mass - destructive_negative; S_v2(pair) = positive_mass - destructive_negative + alignment_load
- main_judgment: 闭式核已被重写成一般闭包边距和严格闭包边距两条核，可以直接判断是否需要把对齐负荷从负质量中单独拆出。

## Stable Features
- positive_mass_v2_term: positive
- alignment_load_v2_term: negative
- closed_form_balance_v2_term: positive

## Fits
- target: union_joint_adv
  intercept: +0.155987
  positive_mass_v2_term: +0.079135
  destructive_negative_v2_term: -0.509600
  alignment_load_v2_term: -0.400985
  closed_form_balance_v2_term: +0.588736
  strict_balance_v2_term: +0.187751
- target: union_synergy_joint
  intercept: +0.355641
  positive_mass_v2_term: +0.080100
  destructive_negative_v2_term: -0.311650
  alignment_load_v2_term: -0.412351
  closed_form_balance_v2_term: +0.391750
  strict_balance_v2_term: -0.020602
- target: strict_positive_synergy
  intercept: -6.029893
  positive_mass_v2_term: +2.210314
  destructive_negative_v2_term: +0.813427
  alignment_load_v2_term: -1.035818
  closed_form_balance_v2_term: +1.396887
  strict_balance_v2_term: +0.361068
