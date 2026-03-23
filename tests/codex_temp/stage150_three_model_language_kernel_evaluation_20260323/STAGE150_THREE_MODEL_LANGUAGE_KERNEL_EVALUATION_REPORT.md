# Stage150: 三模型语言主核评估块

## 核心结果
- 模型数: 3
- 现象数: 7
- 稳定主核数: 3
- 过渡主核数: 2
- 弱主核数: 2
- 总体主核分数: 0.6819
- 层同构分数: 0.6405
- 三模型联合反演式: triple_inversion = 0.10*a + 0.20*r + 0.10*q + 0.00*g + 0.30*f + 0.30*b

## 模型汇总
- GPT-2: test_mode=历史真实测试快照复用; transfer=reference_anchor; pass_rate=1.0000; field=field = 0.10*q + 0.10*b + 0.80*g
- Qwen3-4B: test_mode=本轮真实实跑; transfer=theory_transfer_weak; pass_rate=0.4286; field=field = 0.40*q + 0.00*b + 0.60*g
- DeepSeek-R1-Distill-Qwen-7B: test_mode=本轮真实实跑; transfer=theory_transfer_weak; pass_rate=0.2857; field=field = 0.50*q + 0.10*b + 0.40*g

## 现象汇总
- adverb_route_shift: verdict=stable_core; mean=0.6936; min=0.6131; scores=GPT-2=0.6131, Qwen3-4B=0.7873, DeepSeek-R1-Distill-Qwen-7B=0.6805
- conditional_field_fit: verdict=stable_core; mean=0.9761; min=0.9389; scores=GPT-2=0.9389, Qwen3-4B=0.9964, DeepSeek-R1-Distill-Qwen-7B=0.9928
- discourse_remention: verdict=stable_core; mean=0.8168; min=0.7414; scores=GPT-2=0.9023, Qwen3-4B=0.8066, DeepSeek-R1-Distill-Qwen-7B=0.7414
- syntax_anchor_band: verdict=partial_core; mean=0.5000; min=0.0000; scores=GPT-2=1.0000, Qwen3-4B=0.0000, DeepSeek-R1-Distill-Qwen-7B=0.5000
- anaphora_repair: verdict=partial_core; mean=0.5286; min=0.5011; scores=GPT-2=0.5613, Qwen3-4B=0.5234, DeepSeek-R1-Distill-Qwen-7B=0.5011
- noun_verb_joint_chain: verdict=weak_core; mean=0.3309; min=0.2250; scores=GPT-2=0.5426, Qwen3-4B=0.2250, DeepSeek-R1-Distill-Qwen-7B=0.2250
- noun_verb_result_chain: verdict=weak_core; mean=0.4893; min=0.4500; scores=GPT-2=0.5679, Qwen3-4B=0.4500, DeepSeek-R1-Distill-Qwen-7B=0.4500