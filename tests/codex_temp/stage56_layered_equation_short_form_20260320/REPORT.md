# Stage56 分层方程短式摘要

- main_judgment: 当前分层双主式已经可以进一步压成更短的正式写法：一般层由 G、D、gd 与 L_base 决定，严格层则在其基础上再叠加 S 与 L_select。

## Short Form
- general_short_form: U_general(pair) ~= a * G(pair) + d * D(pair) + p * gd(pair) - l * L_base(pair)
- strict_short_form: U_strict(pair) ~= U_general(pair) + b * S(pair) + s * L_select(pair)
- discriminator_short_form: D_strict(pair) = dual_gap_final(pair)
- state_dictionary: {"G": "kernel_v4", "S": "strict_module_base", "D": "dual_gap_final", "L_base": "(gs + sd) / 2", "L_select": "(sd - gs) / 2"}
