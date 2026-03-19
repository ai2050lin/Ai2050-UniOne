# Stage56 稳定核压缩摘要

- row_count: 72
- mean_positive_core_term: +0.363800
- mean_negative_core_term: +0.492773
- mean_stable_core_balance: -0.128973
- compressed_equation_text: U_core(pair) = b1 * positive_core + b2 * negative_core + b3 * stable_core_balance
- main_judgment: 当前稳定核已经被压成正核、负核与核间边距三项，可以直接检查主方程是否开始从多变量收缩到更短的核结构。

## Fits
- target: union_joint_adv
  intercept: -1.378700
  positive_core_term: +2.810002
  negative_core_term: +1.197741
  stable_core_balance: +1.612261
- target: union_synergy_joint
  intercept: -0.371601
  positive_core_term: +1.151176
  negative_core_term: +0.107384
  stable_core_balance: +1.043792
- target: strict_positive_synergy
  intercept: -1.856018
  positive_core_term: +7.585010
  negative_core_term: +0.433134
  stable_core_balance: +7.151876
