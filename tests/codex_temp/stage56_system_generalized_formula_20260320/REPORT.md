# Stage56 系统级一般化公式摘要

- main_judgment: 从系统角度整理后，当前最一般化的公式已经可以写成“层级状态向量 + 通道向量 + 目标条件负载算子”的结构；其中 G 与 gd 是跨目标稳定主通道，S 与 D 是分层专用状态，而 gs 与 sd 更像目标特异负载算子。

## Invariants
- general_kernel_positive: True
- discriminator_positive: True
- gd_drive_positive: True
- gs_load_target_specific: True
- sd_load_target_specific: True
- strict_choice_target_specific: True

## Generalized Formulas
- layer_state_vector: z(pair) = [G(pair), S(pair), D(pair)]^T
- channel_vector: c(pair) = [gd(pair), gs(pair), sd(pair)]^T
- general_layer: G(pair) = kernel_v4(pair)
- strict_layer: S(pair) = strict_module_base(pair)
- discriminator_layer: D(pair) = dual_gap_final(pair)
- generalized_observation: y_t(pair) = W_t * z(pair) + V_t * c(pair) + eps_t(pair)
- compressed_general_system: y_t(pair) ~= a_t * G(pair) + b_t * S(pair) + d_t * D(pair) + p_t * gd(pair) + L_t(gs(pair), sd(pair)) + eps_t(pair)
- load_operator: L_t(gs, sd) = q_t * gs + r_t * sd; 其中 q_t, r_t 随目标变化，因此它们更像目标特异负载算子
- system_judgment: 当前最一般化的系统结构已经不再是一条单方程，而是“分层状态向量 + 通道向量 + 目标条件负载算子”的形式。
