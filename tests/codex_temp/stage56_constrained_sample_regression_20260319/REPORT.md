# Stage56 约束样本回归摘要

- row_count: 72
- main_judgment: 当前第一版约束回归已经把稳定符号先验压进样本级回归，可以直接检查主方程在符号约束下是否更接近可解释形式。

## Sign Priors
- logic_prototype_proxy: positive
- logic_fragile_bridge_proxy: negative
- syntax_constraint_conflict_proxy: positive
- logic_control_proxy: negative

## Fits
- target: union_joint_adv
  constrained_weights:
    intercept: -1.298492
    atlas_static_proxy: -0.468811
    offset_static_proxy: -2.553081
    frontier_dynamic_proxy: +1.084867
    logic_prototype_proxy: +3.708567
    logic_fragile_bridge_proxy: -7.042737
    syntax_constraint_conflict_proxy: +124.555010
    window_hidden_proxy: -0.129024
    window_mlp_proxy: +0.234423
    style_control_proxy: +0.611261
    logic_control_proxy: -2.861354
    syntax_control_proxy: -1.680801
- target: union_synergy_joint
  constrained_weights:
    intercept: -1.515546
    atlas_static_proxy: +0.057026
    offset_static_proxy: -0.836363
    frontier_dynamic_proxy: -1.280556
    logic_prototype_proxy: +1.442071
    logic_fragile_bridge_proxy: -9.328095
    syntax_constraint_conflict_proxy: +15.546320
    window_hidden_proxy: -0.157859
    window_mlp_proxy: +0.284898
    style_control_proxy: +3.393837
    logic_control_proxy: +0.000000
    syntax_control_proxy: -0.203345
- target: strict_positive_synergy
  constrained_weights:
    intercept: -4.467674
    atlas_static_proxy: -1.868126
    offset_static_proxy: -7.039846
    frontier_dynamic_proxy: +10.950980
    logic_prototype_proxy: +6.195045
    logic_fragile_bridge_proxy: -44.139717
    syntax_constraint_conflict_proxy: +112.954071
    window_hidden_proxy: -0.181709
    window_mlp_proxy: +0.321777
    style_control_proxy: -13.734287
    logic_control_proxy: -3.056865
    syntax_control_proxy: +1.944476
