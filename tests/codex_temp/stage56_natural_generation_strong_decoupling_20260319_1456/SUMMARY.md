# 自然生成强解耦摘要

- case_count: 216
- component_joined_row_count: 216

## Per Axis
- logic: hidden_generated_share=0.6108, mlp_generated_share=0.6016, hidden_generated_dominant_ratio=0.9306, mlp_generated_dominant_ratio=0.9167
- style: hidden_generated_share=0.6532, mlp_generated_share=0.6493, hidden_generated_dominant_ratio=0.9167, mlp_generated_dominant_ratio=0.9028
- syntax: hidden_generated_share=0.2584, mlp_generated_share=0.5197, hidden_generated_dominant_ratio=0.1111, mlp_generated_dominant_ratio=0.8056

## Per Component
- logic_fragile_bridge: signal_origin=prompt_contaminated, hidden_prompt_corr=-0.2183, hidden_generated_corr=-0.2320, mlp_prompt_corr=-0.0378, mlp_generated_corr=-0.1792
- logic_prototype: signal_origin=mixed, hidden_prompt_corr=-0.2385, hidden_generated_corr=-0.2694, mlp_prompt_corr=-0.2089, mlp_generated_corr=-0.1270
- syntax_constraint_conflict: signal_origin=generated_dominant, hidden_prompt_corr=-0.3723, hidden_generated_corr=-0.1178, mlp_prompt_corr=+0.5438, mlp_generated_corr=+0.7051
