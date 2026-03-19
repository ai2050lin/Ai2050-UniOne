# Stage56 一般编码机制摘要

- dominant_form: anchor_fiber_path_bundle_with_windowed_closure
- structure_scores: {'local_linear_ratio': 0.076, 'path_bundle_ratio': 0.896, 'control_mixed_ratio': 0.028}
- shared_closure_categories: ['fruit', 'action', 'weather', 'object', 'nature', 'vehicle']

## 闭包动力学
- logic_prototype_window: {'joint_adv_hidden_window': 'tail_pos_-9..tail_pos_-8', 'joint_adv_hidden_corr': 0.4941753813683336, 'joint_adv_mlp_window': 'tail_pos_-9..tail_pos_-8', 'joint_adv_mlp_corr': 0.5016062216321406}
- logic_fragile_bridge_window: {'synergy_hidden_window': 'tail_pos_-2..tail_pos_-1', 'synergy_hidden_corr': -0.26838847200236665, 'joint_adv_hidden_window': 'tail_pos_-2..tail_pos_-1', 'joint_adv_hidden_corr': -0.4304910299531608}
- syntax_constraint_conflict_window: {'synergy_hidden_window': 'tail_pos_-8..tail_pos_-5', 'synergy_hidden_corr': 0.8944486933294248, 'synergy_mlp_window': 'tail_pos_-6..tail_pos_-3', 'synergy_mlp_corr': 0.8530189005728579}

## 自然生成解耦
- natural_generation_decoupling: {'style_hidden_generated_share': 0.6531785321136827, 'logic_hidden_generated_share': 0.6108294490272518, 'syntax_hidden_generated_share': 0.2583617908155026, 'syntax_hidden_prompt_share': 0.7416382091844974, 'syntax_component_hidden_prompt_share': 0.521573044856195, 'syntax_component_hidden_generated_share': 0.47842695514380496, 'judgement': 'syntax_prompt_contaminated'}

## 一般原则
- 语言系统主结构不是统一线性空间，而是锚点、纤维、关系束与控制轴的复合体。
- 局部线性只在少数关系家族成立，系统主结构由路径束主导。
- 闭包动力学必须在连续窗口上建模，粗总量变量已经失效。
- 自然生成分析必须先把提示骨架区和新增生成区解耦，否则会混入伪早窗信号。
