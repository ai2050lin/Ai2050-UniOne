# Stage56 双主式正式方程系统摘要

- main_judgment: 双主式已经可以写成正式的分层方程系统：一般层负责主闭包，严格层负责严格负载，判别层负责区分严格性，而层间耦合则负责三层之间的方向性连接。

## Formal Equations
- general_layer: U_general(pair) = kernel_v4(pair)
- strict_layer: U_strict(pair) = strict_module_base(pair)
- discriminator_layer: D_strict(pair) = dual_gap_final(pair)

## Coupling Signs
- gs_coupling_term: {"union_joint_adv": "negative", "union_synergy_joint": "positive", "strict_positive_synergy": "positive"}
- gd_coupling_term: {"union_joint_adv": "positive", "union_synergy_joint": "positive", "strict_positive_synergy": "positive"}
- sd_coupling_term: {"union_joint_adv": "positive", "union_synergy_joint": "negative", "strict_positive_synergy": "negative"}
