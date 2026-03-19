# Stage56 双主式层级化摘要

- main_judgment: 双主式更适合写成分层结构：一般闭包核负责主核层，严格闭包模块负责严格层，dual_gap 负责判别层，而不是与前两层同层并场。

## General Layer
- equation: U_general = kernel_v4
- stable_signs: {"union_joint_adv": "positive", "union_synergy_joint": "positive", "general_mean_target": "positive", "strict_positive_synergy": "positive", "strictness_delta_vs_general": "positive"}

## Strict Layer
- equation: U_strict = strict_module_base
- final_choice: {"feature": "strict_module_base_term", "selectivity_score": 0.4780334532260895, "simplicity_bonus": 0.01, "final_score": 0.4880334532260895, "weights": {"union_joint_adv": -0.22536253929138184, "union_synergy_joint": -0.10061115026473999, "strict_positive_synergy": 0.6410202980041504}}

## Discriminator Layer
- equation: D_strict = dual_gap_final
- stable_signs: {"strictness_delta_vs_union": "positive", "strictness_delta_vs_synergy": "positive", "strict_positive_synergy": "positive", "union_synergy_joint": "positive"}
