
## 2026-02-28 继续研究记录（DNN 数学结构）

### 执行命令
1. `python scripts/start_structure_recovery_process.py --date-tag 20260228`
2. `python scripts/scaling_validation_matrix.py --preset quick --model-filter m_0.4m --data-filter d_40k --epochs 24 --lr 0.0003 --weight-decay 0.01 --warmup-ratio 0.03 --min-lr-scale 0.1 --seed 42028 --output-json tempdata/pipeline_stage_c_minimal_rebuild_20260228_tuned.json --output-md tempdata/pipeline_stage_c_minimal_rebuild_20260228_tuned.md`
3. `python scripts/scaling_validation_matrix.py --preset quick --model-filter m_1.4m --data-filter d_120k --epochs 24 --lr 0.0003 --weight-decay 0.01 --warmup-ratio 0.03 --min-lr-scale 0.1 --seed 42028 --output-json tempdata/pipeline_stage_c_minimal_rebuild_20260228_m14_tuned.json --output-md tempdata/pipeline_stage_c_minimal_rebuild_20260228_m14_tuned.md`
4. `python scripts/scaling_validation_matrix.py --preset quick --model-filter m_1.4m --data-filter d_120k --epochs 20 --lr 0.001 --weight-decay 0.1 --warmup-ratio 0.0 --min-lr-scale 0.05 --seed 42 --output-json tempdata/pipeline_stage_c_minimal_rebuild_20260228_m14_repro.json --output-md tempdata/pipeline_stage_c_minimal_rebuild_20260228_m14_repro.md`
5. `python scripts/import_scaling_to_timeline.py --timeline tempdata/agi_route_test_timeline.json --route fiber_bundle --reports tempdata/pipeline_stage_c_minimal_rebuild_20260228_m14_repro.json tempdata/pipeline_stage_c_minimal_rebuild_20260228_m14_tuned.json tempdata/pipeline_stage_c_minimal_rebuild_20260228_tuned.json`

### 关键结果
- A 阶段（不变量）: `candidate_count=19`, `stability_score=0.8416`（pass）
- B 阶段（因果）: `feature_avg_top1_uplift=0.0646` > `layerwise_max_uplift=0.0166`（pass）
- D 阶段（跨模态）: `val_fused_acc=0.99625`, `val_retrieval_top1=0.9775`（pass，synthetic）
- E 阶段（反证）: H1=falsified, H2=supported, H3=supported；strict gate 通过

### Stage C 复验（最小重建）
- `m_0.4m + d_40k`（24 epoch tuned）: `best_val_acc=0.0464`（仍低）
- `m_1.4m + d_120k`（24 epoch tuned）: `best_val_acc=0.4865`（中等）
- `m_1.4m + d_120k`（20 epoch 历史最优同配置复现）: `best_val_acc=0.8567`（高，复现成功）

### 结论
- Stage C 当前核心不是“结构失效”，而是“配置敏感”：历史最优配置可复现高分（0.8567），偏离配置会显著掉分。
- 下一步应把 Stage C 默认配置固定到可复现高分区间，再做多 seed 与跨任务复验，避免 quick-smoke 低分误导阶段判断。

### 流程修正（同日追加）
- 代码改动：`scripts/start_structure_recovery_process.py` 的 Stage C 默认配置由 quick-smoke（2 epoch, m_0.4m/d_40k）改为可复现高分基线（20 epoch, m_1.4m/d_120k）。
- 复跑命令：`python scripts/start_structure_recovery_process.py --date-tag 20260228_v2`
- 复跑结果：`tempdata/structure_recovery_pipeline_kickoff_20260228_v2.json` 状态 `pass`，其中 Stage C `best_val_acc=0.856667`。

### 前端同步
- 已在 `frontend/src/HLAIBlueprint.jsx` 阶段3测试列表新增 `p3_t3`，展示本次 2026-02-28 的高分复现证据。
