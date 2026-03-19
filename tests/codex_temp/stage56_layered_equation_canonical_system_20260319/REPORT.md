# Stage56 分层双主式规范化系统摘要

- main_judgment: 分层双主式已经可以用主核层、严格层、判别层加上规范化通道系统来描述，其中 gd 是主驱动通道，gs 和 sd 更像目标特异的负载通道。

## Formal Equations
- general_layer: U_general(pair) = kernel_v4(pair)
- strict_layer: U_strict(pair) = strict_module_base(pair)
- discriminator_layer: D_strict(pair) = dual_gap_final(pair)

## Canonical Channels
- gs_load_channel_term: {"union_joint_adv": "positive", "union_synergy_joint": "negative", "strict_positive_synergy": "negative"}
- gd_drive_channel_term: {"union_joint_adv": "positive", "union_synergy_joint": "positive", "strict_positive_synergy": "positive"}
- sd_load_channel_term: {"union_joint_adv": "negative", "union_synergy_joint": "positive", "strict_positive_synergy": "positive"}
