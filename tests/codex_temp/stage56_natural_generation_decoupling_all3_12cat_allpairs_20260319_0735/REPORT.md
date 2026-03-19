# Stage56 自然生成解耦报告

- case_count: 216
- component_joined_row_count: 216

## 按轴汇总
- logic: hidden_prompt_share=0.3892, hidden_generated_share=0.6108, mlp_prompt_share=0.3984, mlp_generated_share=0.6016, hidden_zone_counts={'generated': 70, 'prompt': 2}
- style: hidden_prompt_share=0.3468, hidden_generated_share=0.6532, mlp_prompt_share=0.3507, mlp_generated_share=0.6493, hidden_zone_counts={'prompt': 6, 'generated': 66}
- syntax: hidden_prompt_share=0.7416, hidden_generated_share=0.2584, mlp_prompt_share=0.4803, mlp_generated_share=0.5197, hidden_zone_counts={'prompt': 66, 'generated': 6}

## 按组件汇总
- logic_fragile_bridge: hidden_prompt_share=0.3770, hidden_generated_share=0.6230, corr_hidden_prompt_to_synergy=-0.2183, corr_hidden_generated_to_synergy=-0.2320
- logic_prototype: hidden_prompt_share=0.4079, hidden_generated_share=0.5921, corr_hidden_prompt_to_synergy=-0.2385, corr_hidden_generated_to_synergy=-0.2694
- syntax_constraint_conflict: hidden_prompt_share=0.5216, hidden_generated_share=0.4784, corr_hidden_prompt_to_synergy=-0.3723, corr_hidden_generated_to_synergy=-0.1178
