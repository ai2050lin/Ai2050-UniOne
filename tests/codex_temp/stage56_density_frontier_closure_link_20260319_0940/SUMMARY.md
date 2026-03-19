# 密度前沿到闭包联立

- joint_rows: 9

## Top Correlations
- pair_delta_cosine_mean -> corr_prototype_to_union_synergy: corr=+0.7859
- pair_delta_cosine_mean -> mean_bridge_field_proxy: corr=+0.7157
- mass25_compaction_ratio -> mean_prototype_field_proxy: corr=+0.6128
- mass10_compaction_ratio -> mean_mismatch_field_proxy: corr=+0.5950
- pair_delta_cosine_mean -> corr_bridge_to_union_synergy: corr=-0.5696
- frontier_sharpness_auc -> mean_prototype_field_proxy: corr=-0.5627
- pair_delta_cosine_mean -> corr_mismatch_to_union_synergy: corr=-0.5496
- pair_delta_cosine_mean -> mean_prototype_field_proxy: corr=-0.5432
- mass10_compaction_ratio -> mean_prototype_field_proxy: corr=+0.5116
- frontier_sharpness_auc -> corr_bridge_to_union_synergy: corr=+0.4913
- mass10_compaction_ratio -> share_mismatch_exposure: corr=+0.4864
- mass25_compaction_ratio -> share_mismatch_exposure: corr=+0.4836

## Model-Axis Rows
- DeepSeek-7B / logic: mass10=0.0338, auc=0.1446, pair_cos=0.4722, proto_corr=+0.1345, bridge_corr=-0.1119, conflict_corr=+0.0987
- DeepSeek-7B / style: mass10=0.0416, auc=0.1293, pair_cos=0.2074, proto_corr=-0.3073, bridge_corr=+0.2946, conflict_corr=-0.0259
- DeepSeek-7B / syntax: mass10=0.0255, auc=0.1303, pair_cos=0.2002, proto_corr=-0.1946, bridge_corr=+0.3515, conflict_corr=+0.0044
- GLM-4-9B / logic: mass10=0.0308, auc=0.1628, pair_cos=0.3865, proto_corr=+0.2473, bridge_corr=-0.0446, conflict_corr=+0.0728
- GLM-4-9B / style: mass10=0.0281, auc=0.1717, pair_cos=0.2139, proto_corr=-0.3953, bridge_corr=+0.6481, conflict_corr=-0.0191
- GLM-4-9B / syntax: mass10=0.0350, auc=0.1481, pair_cos=0.1258, proto_corr=-0.5675, bridge_corr=+0.7334, conflict_corr=+0.2238
- Qwen3-4B / logic: mass10=0.0318, auc=0.1412, pair_cos=0.5407, proto_corr=+0.5733, bridge_corr=-0.4491, conflict_corr=-0.0208
- Qwen3-4B / style: mass10=0.0402, auc=0.1313, pair_cos=0.2478, proto_corr=-0.0167, bridge_corr=+0.0084, conflict_corr=+0.0717
- Qwen3-4B / syntax: mass10=0.0466, auc=0.1052, pair_cos=0.1490, proto_corr=+0.1643, bridge_corr=-0.4639, conflict_corr=-0.1203
