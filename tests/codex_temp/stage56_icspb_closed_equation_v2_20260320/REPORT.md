# Stage56 ICSPB 闭式方程第二版摘要

- main_judgment: ICSPB 闭式方程第二版已经把阶段最终的一般主核 G_final、阶段最终严格核心 S_final 和两类负载算子并进同一套分层短式。

## Equations
{
  "general_equation": "U_general(pair) ~= a * G_final(pair) + d * D(pair) + p * gd(pair) - l * L_base(pair)",
  "strict_equation": "U_strict(pair) ~= U_general(pair) + b * S_final(pair) + s * L_select(pair)",
  "discriminator_equation": "D_strict(pair) = dual_gap_final(pair)"
}

## State Dictionary
{
  "G_final": "kernel_v4",
  "S_final": "strict_module_base_term",
  "D": "dual_gap_final",
  "gd": "主驱动通道",
  "L_base": "(gs + sd) / 2",
  "L_select": "(sd - gs) / 2"
}
