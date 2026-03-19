# Stage56 一般化公式精炼摘要

- main_judgment: 系统级一般化公式现在可以进一步缩短成：一般目标由 G、D、gd 和基础负载算子共同决定，严格目标则在一般目标之上额外叠加严格选择算子。

## Refined Formulas
- general_state_observation: y_general(pair) ~= a * G(pair) + d * D(pair) + p * gd(pair) - l * L_base(pair) + eps(pair)
- strict_state_observation: y_strict(pair) ~= y_general(pair) + s * L_select(pair) + eta(pair)
- base_load_operator: L_base(pair) = (gs(pair) + sd(pair)) / 2
- strict_select_operator: L_select(pair) = (sd(pair) - gs(pair)) / 2
- operator_form_judgment: 当前更一般化的系统公式已经可以从“层级状态 + 通道向量 + 目标条件负载算子”继续压成“基础负载算子 + 严格选择算子”的双算子结构。
