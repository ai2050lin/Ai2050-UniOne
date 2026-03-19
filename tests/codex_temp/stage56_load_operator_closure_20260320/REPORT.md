# Stage56 负载算子收口摘要

- row_count: 72
- main_judgment: gs 与 sd 可以继续压成两个更一般的算子：基础负载算子 L_base 在三目标上稳定为负，严格选择算子 L_select 只在严格闭包目标上转成正。

## Operator Equations
- load_base_operator: L_base(gs, sd) = (gs + sd) / 2
- strict_select_operator: L_select(gs, sd) = (sd - gs) / 2

## Stable Negative Features
- load_mean_term
- load_abs_sum_term

## Strict Selective Features
- load_contrast_term
