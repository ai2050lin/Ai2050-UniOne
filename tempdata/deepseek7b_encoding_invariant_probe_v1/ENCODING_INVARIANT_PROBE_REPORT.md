# 编码不变量探针报告

## 输入
- 三轴结构: `tempdata/deepseek7b_triaxial_param_structure_v1/triaxial_param_structure.json`
- 可证伪报告: `tempdata/deepseek7b_stage_falsifiable_report_v3_bilingual.json`

## 总览
- 概念数: `5`
- 轴隔离比较数: `15`
- 不变量通过/失败: `5` / `2`
- 阶段就绪度(readiness): `0.7143`

## 轴间隔离（越低越接近分工编码）
- `micro_attr__same_type`: gate_dim_jaccard_mean=0.0270, neuron_jaccard_mean=0.0000, layer_jaccard_mean=0.1733
- `micro_attr__super_type`: gate_dim_jaccard_mean=0.0300, neuron_jaccard_mean=0.0000, layer_jaccard_mean=0.1733
- `same_type__super_type`: gate_dim_jaccard_mean=0.0153, neuron_jaccard_mean=0.0000, layer_jaccard_mean=0.0667

## 组内共享骨架（按super_type）
- `fruit`: super=4, micro=0, same=0, super_dims=[2041, 2427, 2467, 2524]
- `animal`: super=1, micro=0, same=0, super_dims=[2727]

## 判定
- `axis_isolation_micro_attr__same_type`: pass (metric=gate_dim_jaccard_mean, value=0.0270, criterion=<= 0.08)
- `axis_isolation_micro_attr__super_type`: pass (metric=gate_dim_jaccard_mean, value=0.0300, criterion=<= 0.08)
- `axis_isolation_same_type__super_type`: pass (metric=gate_dim_jaccard_mean, value=0.0153, criterion=<= 0.08)
- `group_super_shared_backbone_fruit`: pass (metric=super_type_count_minus_max_other, value=4.0000, criterion=> 0)
- `group_super_shared_backbone_animal`: pass (metric=super_type_count_minus_max_other, value=1.0000, criterion=> 0)
- `global_causal_strength_ready`: fail (metric=causal_seq_z_mean, value=0.9645, criterion=>= 1.96)
- `global_mechanism_score_ready`: fail (metric=overall_score_mean, value=0.2903, criterion=>= 0.42)

## 解释
- 该探针重点检查“参数维共享骨架 + 轴间低重叠”是否同时成立。
- 若局部不变量成立但全局强证据未过阈值，意味着编码结构线索已出现，但统一原理仍需更强因果验证。
