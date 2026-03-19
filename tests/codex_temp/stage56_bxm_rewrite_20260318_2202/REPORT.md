# Stage56 B/X/M Rewrite Report

- row_count: 36

## Definitions
- stable_bridge: B>0 且 union_joint_adv>0 且 union_synergy_joint>0
- fragile_bridge: B>0 但未满足稳定闭包条件
- constraint_conflict: X>0 且 union_synergy_joint>0，或 strict_positive_synergy 为真
- destructive_conflict: X>0 且不属于约束型冲突
- mismatch_exposure: M>0 且 union_joint_adv>0 且 union_synergy_joint>=0
- mismatch_damage: M>0 且不属于失配暴露

## Per Axis
- style: stable_bridge=0.000090, fragile_bridge=0.000078, constraint_conflict=0.000062, destructive_conflict=0.000341, mismatch_exposure=0.000208, mismatch_damage=0.000776
- logic: stable_bridge=0.000100, fragile_bridge=0.000924, constraint_conflict=0.000177, destructive_conflict=0.000347, mismatch_exposure=0.000150, mismatch_damage=0.000805
- syntax: stable_bridge=0.000021, fragile_bridge=0.000249, constraint_conflict=0.000281, destructive_conflict=0.000056, mismatch_exposure=0.000173, mismatch_damage=0.000229

## Top Positive Closure Components
- style / constraint_conflict: mean=0.000062, share=0.1539, corr_synergy=0.2841, corr_joint_adv=0.1311
- syntax / mismatch_exposure: mean=0.000173, share=0.4314, corr_synergy=0.2752, corr_joint_adv=0.8929
- syntax / constraint_conflict: mean=0.000281, share=0.8347, corr_synergy=0.2728, corr_joint_adv=0.8931
- logic / constraint_conflict: mean=0.000177, share=0.3379, corr_synergy=0.2604, corr_joint_adv=0.0629
- logic / mismatch_exposure: mean=0.000150, share=0.1572, corr_synergy=0.2435, corr_joint_adv=0.0739
- style / mismatch_exposure: mean=0.000208, share=0.2111, corr_synergy=0.2220, corr_joint_adv=0.0753
- syntax / stable_bridge: mean=0.000021, share=0.0769, corr_synergy=0.1704, corr_joint_adv=0.0636
- style / stable_bridge: mean=0.000090, share=0.5361, corr_synergy=0.1475, corr_joint_adv=0.0612
