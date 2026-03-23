# Stage149: 滚动扩量调度块

## 核心结果
- 变量数: 6
- 平均优先级分数: 0.3731
- 最高优先变量: b
- 最低优先变量: f
- 建议新增样本总数: 560
- 跨模型层同构分数: 0.6405
- 三模型联合反演分数: 0.9948

## 调度表
- b: priority=0.7945; weakest_model=Qwen3-4B; families=context_bias_family,late_repair_family; contrasts=substitute,weaken,break; difficulties=hard,adversarial; add_cases=256
- a: priority=0.4287; weakest_model=Qwen3-4B; families=anchor_subject_family; contrasts=weaken,break; difficulties=medium,hard; add_cases=32
- g: priority=0.3431; weakest_model=Qwen3-4B; families=adverb_route_family,context_bias_family,late_repair_family; contrasts=substitute,break; difficulties=medium,hard; add_cases=80
- q: priority=0.2807; weakest_model=Qwen3-4B; families=adverb_route_family,context_bias_family; contrasts=substitute,weaken; difficulties=medium,hard; add_cases=48
- r: priority=0.2195; weakest_model=DeepSeek-R1-Distill-Qwen-7B; families=pronoun_recovery_family,ellipsis_recovery_family; contrasts=substitute,break; difficulties=medium,hard; add_cases=48
- f: priority=0.1718; weakest_model=DeepSeek-R1-Distill-Qwen-7B; families=anchor_subject_family,pronoun_recovery_family,ellipsis_recovery_family,late_repair_family; contrasts=weaken,break; difficulties=medium,hard; add_cases=96