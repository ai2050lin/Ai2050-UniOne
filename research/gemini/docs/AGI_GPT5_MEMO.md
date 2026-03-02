
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

### 5-seed 稳定性复验（继续）
#### 执行命令
1. 5-seed 复验：
`python scripts/scaling_validation_matrix.py --preset quick --model-filter m_1.4m --data-filter d_120k --epochs 20 --lr 0.001 --weight-decay 0.1 --warmup-ratio 0.0 --min-lr-scale 0.05 --seed 42031..42035`
2. 汇总生成：
`tempdata/pipeline_stage_c_minimal_rebuild_20260228_m14_multiseed_summary.json/.md`
3. 时间线导入：
`python scripts/import_scaling_to_timeline.py --timeline tempdata/agi_route_test_timeline.json --route fiber_bundle --reports <seed jsons + summary json>`

#### 结果
- 5 seed 的 best_val_acc: [0.8413, 0.8476, 0.8737, 0.8455, 0.8475]
- `best_val_acc_mean=0.8511242`, `best_val_acc_std=0.01150055`
- `best_val_acc_min=0.8413`, `best_val_acc_max=0.8737`
- 结论：`multiseed_stable`

#### 同步
- 已将该结果写入 `frontend/src/HLAIBlueprint.jsx` 阶段3测试 `p3_t4`。
- 阶段3总结更新为“已完成 5-seed 稳定性复验”。

## 2026-02-28 继续研究记录（跨模态与 H3 新种子块）

### 执行命令
1. `python scripts/train_fiber_multimodal_connector.py --dataset mnist --mnist-root tempdata/data --mnist-download --total-samples 12000 --val-ratio 0.2 --batch-size 128 --epochs 6 --d-model 96 --lr 0.001 --weight-decay 0.0001 --temperature 0.2 --w-fused-cls 1.2 --w-contrastive 0.1 --w-alignment 0.1 --w-smoothness 0.05 --w-curvature 0.05 --seed 42028 --report-json tempdata/pipeline_stage_d_multimodal_20260228_mnist_v1.json --report-md tempdata/pipeline_stage_d_multimodal_20260228_mnist_v1.md --timeline tempdata/agi_route_test_timeline.json --route fiber_bundle --analysis-type multimodal_connector_mnist_v1`
2. `python scripts/h3_holdout_validation.py --models gpt2,distilgpt2,gpt2-medium --task-profile expanded --max-per-category 64 --lock-mode per_category --locked-configs-from tempdata/h3_category_adaptive_search_20260224_v2.json --adapter-failure-report tempdata/h3_failure_localizer_20260224_v2_multi.json --adapter-profile hybrid_support_boost_v6 --adapter-strength 1.0 --support-models-min 2 --falsify-models-max 0 --seed 20260421 --device auto --output tempdata/h3_holdout_validation_20260228_v3_seed20260421.json`
3. `python scripts/h3_holdout_validation.py ... --max-per-category 64 --seed 20260422 --output tempdata/h3_holdout_validation_20260228_v3_seed20260422.json`
4. `python scripts/h3_holdout_validation.py ... --max-per-category 64 --seed 20260423 --output tempdata/h3_holdout_validation_20260228_v3_seed20260423.json`
5. `python scripts/h3_holdout_validation.py ... --max-per-category 64 --seed-block-reports tempdata/h3_holdout_validation_20260228_v3_seed20260421.json,tempdata/h3_holdout_validation_20260228_v3_seed20260422.json,tempdata/h3_holdout_validation_20260228_v3_seed20260423.json --seed 20260424 --output tempdata/h3_holdout_validation_20260228_v3_gatecheck_seed20260424.json`
6. `python scripts/h3_holdout_validation.py ... --max-per-category 96 --seed 20260422 --output tempdata/h3_holdout_validation_20260228_v4_n96_seed20260422.json`
7. `python scripts/h3_holdout_validation.py ... --max-per-category 96 --seed 20260425 --output tempdata/h3_holdout_validation_20260228_v4_n96_seed20260425.json`
8. `python scripts/h3_holdout_validation.py ... --max-per-category 96 --seed 20260426 --output tempdata/h3_holdout_validation_20260228_v4_n96_seed20260426.json`
9. `python scripts/h3_holdout_validation.py ... --max-per-category 96 --seed-block-reports tempdata/h3_holdout_validation_20260228_v4_n96_seed20260422.json,tempdata/h3_holdout_validation_20260228_v4_n96_seed20260425.json,tempdata/h3_holdout_validation_20260228_v4_n96_seed20260426.json --seed 20260427 --output tempdata/h3_holdout_validation_20260228_v4_n96_gatecheck_seed20260427.json`

### 关键结果
- Stage D（MNIST）:
  - `val_fused_acc=1.0000`
  - `val_retrieval_top1=0.9883`
  - `val_alignment_cos=0.6184`
  - 相比 synthetic 结果，真实视觉数据证据更强。
- H3（n=64）新种子块：
  - seed20260421: `support_models=2, falsify_models=0`
  - seed20260422: `support_models=0, falsify_models=0`（全 open）
  - seed20260423: `support_models=3, falsify_models=0`
  - gatecheck：`seed_block_support_min=0`，strict gate 不通过（goal_1_pending）
- H3（n=96）功效修正后：
  - seed20260422/25/26: 均为 `support_models=2, falsify_models=0`
  - gatecheck(seed20260427): `current_run_pass=true`, `seed_block_pass=true`,
    `seed_block_support_min=2`, `seed_block_falsify_max=0`
  - layered goals: `progression=goal_3_pending`

### 结论
- H3 的核心短板从“反证失败”转为“任务家族覆盖不均（goal_3）”。
- 提升每类样本数（64->96）显著降低了“统计功效不足导致的 open”。
- 下一步优先做类别级补强（尤其 fact/antonym 的 support_categories 提升），而不是继续盲目扩大 seed。

### 流程一致性修正（同日追加）
- 代码改动：`scripts/start_structure_recovery_process.py`
  - Stage D 默认配置升级为 `dataset=auto (MNIST优先), total_samples=12000, epochs=6`
  - Stage E 的 `goal_3` 改为双条件：`task_level_eval` 与 `latest_holdout goal_3` 同时通过才标记 pass
- 回归运行：`python scripts/start_structure_recovery_process.py --date-tag 20260228_v4`
- 回归结果：
  - `stageD.dataset=mnist`
  - `stageE.layered_goals.progression=goal_3_pending`
  - 总流程状态仍 `pass`（A/B/C/D 均通过），但明确暴露 H3 的当前卡点在 goal_3。
