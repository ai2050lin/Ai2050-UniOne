# Stage56 B/X/M Rewrite Report

- row_count: 72

## Definitions
- stable_bridge: B>0，且 union_joint_adv>0，且 union_synergy_joint>0
- fragile_bridge: B>0，但未满足稳定桥接闭包条件
- constraint_conflict: X>0，且 union_synergy_joint>0，或 strict_positive_synergy 为真
- destructive_conflict: X>0，且不属于约束型冲突
- mismatch_exposure: M>0，且 union_joint_adv>0，且 union_synergy_joint>=0
- mismatch_damage: M>0，且不属于失配暴露

## Per Axis
- style: stable_bridge=0.000069, fragile_bridge=0.000062, constraint_conflict=0.000034, destructive_conflict=0.000297, mismatch_exposure=0.000124, mismatch_damage=0.000513
- logic: stable_bridge=0.000053, fragile_bridge=0.000689, constraint_conflict=0.000131, destructive_conflict=0.000331, mismatch_exposure=0.000109, mismatch_damage=0.000624
- syntax: stable_bridge=0.000063, fragile_bridge=0.000142, constraint_conflict=0.000160, destructive_conflict=0.000054, mismatch_exposure=0.000418, mismatch_damage=0.000258

## Top Positive Closure Components
- syntax / stable_bridge: mean=0.000063, share=0.3081, corr_synergy=0.4038, corr_joint_adv=0.2811
- syntax / mismatch_exposure: mean=0.000418, share=0.6181, corr_synergy=0.3351, corr_joint_adv=0.4731
- logic / constraint_conflict: mean=0.000131, share=0.2828, corr_synergy=0.2156, corr_joint_adv=0.0805
- style / stable_bridge: mean=0.000069, share=0.5274, corr_synergy=0.2148, corr_joint_adv=0.1568
- style / constraint_conflict: mean=0.000034, share=0.1037, corr_synergy=0.1933, corr_joint_adv=0.1344
- syntax / constraint_conflict: mean=0.000160, share=0.7469, corr_synergy=0.1792, corr_joint_adv=0.7811
- logic / mismatch_exposure: mean=0.000109, share=0.1484, corr_synergy=0.1768, corr_joint_adv=0.0934
- style / mismatch_exposure: mean=0.000124, share=0.1950, corr_synergy=0.1533, corr_joint_adv=0.0969
