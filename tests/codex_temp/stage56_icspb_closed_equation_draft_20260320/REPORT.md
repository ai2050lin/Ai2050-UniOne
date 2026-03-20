# Stage56 ICSPB 闭式方程草案摘要

- main_judgment: 当前 ICSPB 闭式方程草案已经可以写成“主核层 + 严格层 + 判别层”的三层结构，并且在自然语料口径下已经找到与 G / L_base / L_select 对应的自然代理。

## Closed Equations
- general_equation: U_general(pair) ~= a * G(pair) + d * D(pair) + p * gd(pair) - l * L_base(pair)
- strict_equation: U_strict(pair) ~= U_general(pair) + b * S(pair) + s * L_select(pair)
- discriminator_equation: D_strict(pair) = dual_gap_final(pair)
- state_dictionary: {"G": "kernel_v4", "S": "strict_module_base", "D": "dual_gap_final", "gd": "主驱动通道", "L_base": "(gs + sd) / 2", "L_select": "(sd - gs) / 2"}
