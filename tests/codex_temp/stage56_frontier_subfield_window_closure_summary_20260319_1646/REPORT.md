# Stage56 四概念联立摘要

## 统一原则
- 密度前沿回答高质量支撑在哪里，内部子场回答是谁在起作用，词元窗口回答它在什么时候起作用，闭包量回答这些作用最后有没有形成稳定输出。

## 密度前沿
- broad_support_base: +0.1899
- long_separation_frontier: +0.2335
- strongest_positive_frontier: syntax / pair_coverage_middle_mean / corr=+0.3098
- strongest_negative_frontier: logic / pair_delta_l2 / corr=-0.3868

## 内部子场
- logic_fragile_bridge: best_positive=hidden_layer_center(+0.1744), best_negative=complete_prompt_energy(-0.2223)
- logic_prototype: best_positive=hidden_layer_center(+0.1744), best_negative=preferred_density(-0.1003)
- syntax_constraint_conflict: best_positive=complete_prompt_energy(+0.1857), best_negative=preferred_density(-0.0931)

## 词元窗口
- logic_fragile_bridge: hidden=tail_pos_-5, mlp=tail_pos_-5, mean_synergy=-0.0614
- logic_prototype: hidden=tail_pos_-5, mlp=tail_pos_-5, mean_synergy=-0.0162
- syntax_constraint_conflict: hidden=tail_pos_-6, mlp=tail_pos_-5, mean_synergy=+0.0388

## 闭包量
- pair_positive_ratio: +0.1944
- mean_union_joint_adv: +0.0259
- mean_union_synergy_joint: -0.0345
- strongest_positive_field: logic / prototype_field_proxy / corr=+0.2548
- strongest_negative_field: style / prototype_field_proxy / corr=-0.2479
