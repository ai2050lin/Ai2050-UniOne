# DNN 分析与测试续跑报告（2026-02-20）

## 本轮目标
1. 继续执行深度神经网络结构/规模测试。
2. 将结果写入固定格式 JSON 报告并导入时间线。
3. 对异常结果进行对照实验，确认是否为超参数问题。

## 执行内容
- 快速矩阵：`scripts/scaling_validation_matrix.py --preset quick --epochs 8`
  - 输出：
    - `tempdata/scaling_validation_report_quick_20260220.json`
    - `tempdata/scaling_validation_report_quick_20260220.md`
- 大规模压力测试：`m_8.5m + 2M/3M`（默认超参）
  - 输出：
    - `tempdata/scaling_validation_report_m85_2m_3m_20260220.json`
    - `tempdata/scaling_validation_report_m85_2m_3m_20260220.md`
- 大规模对照测试：`m_8.5m + 2M`（tuned 超参）
  - 输出：
    - `tempdata/scaling_validation_report_m85_2m_tuned_20260220.json`
    - `tempdata/scaling_validation_report_m85_2m_tuned_20260220.md`

## 核心结果
- quick 矩阵（4 runs）
  - `best_val_acc_max = 0.324546`
  - `best_val_acc_avg = 0.143398`
- 大规模默认超参（2 runs）
  - `best_val_acc_max = 0.0089`
  - 接近随机（1/113 ≈ 0.00885）
- 大规模 tuned 超参（1 run）
  - `best_val_acc = 1.0000`
  - `final_val_acc = 1.0000`

## 结论
1. 大规模低分不是结构路线失效，主要是优化超参数不匹配导致。
2. 在 2M 样本下，`m_8.5m` 通过 tuned 参数可达到满分，说明路线具备规模可行性。
3. 后续应将大规模默认配置迁移到 tuned 区间（低 LR、较低 weight decay、warmup）。

## 时间线落盘
已导入：`tempdata/agi_route_test_timeline.json`
- 本轮新增 `scaling_validation` 记录：7 条
- 导入脚本：`scripts/import_scaling_to_timeline.py`

## 继续执行：3-seed 复现实验（2026-02-20）

- 配置：`m_8.5m + d_2000k + tuned`（epochs=20, lr=3e-4, wd=0.01, warmup=0.03）
- seeds: 10001, 20002, 30003（实际 run_seed: +17）

结果：
- seed10001: best=1.0000, final=1.0000
- seed20002: best=1.0000, final=1.0000
- seed30003: best=1.0000, final=1.0000

多种子汇总：
- `best_val_acc_mean=1.0, std=0.0`
- `final_val_acc_mean=1.0, std=0.0`

新增产物：
- `tempdata/scaling_validation_report_m85_2m_tuned_seed10001_20260220.json`
- `tempdata/scaling_validation_report_m85_2m_tuned_seed20002_20260220.json`
- `tempdata/scaling_validation_report_m85_2m_tuned_seed30003_20260220.json`
- `tempdata/scaling_validation_m85_2m_tuned_multiseed_20260220.json`
- `tempdata/scaling_validation_m85_2m_tuned_multiseed_summary_20260220.json`

时间线导入：
- `scaling_validation` 新增 3 条
- `scaling_validation_multiseed` 新增 1 条

## 继续执行：4M/6M 扩展点位（2026-02-20）

- 配置：`m_8.5m + tuned`（epochs=12, lr=3e-4, wd=0.01, warmup=0.03）
- 数据规模：d_4000k, d_6000k

结果：
- d_4000k: best=1.0000, final=1.0000, sps=5959.57
- d_6000k: best=1.0000, final=1.0000, sps=5443.99

新增产物：
- `tempdata/scaling_validation_report_m85_4m_6m_tuned_20260220.json`
- `tempdata/scaling_validation_m85_4m_6m_tuned_summary_20260220.json`

时间线补充导入后计数：
- unified_conscious_field: 200
- scaling_validation: 12
- scaling_validation_multiseed: 1

备注：
- 已在 `server/runtime/experiment_store.py` 增加按 analysis_type 保留策略。
- 需重启后端进程后该策略才会持续生效。

## 继续执行：S1 因果干预首轮（2026-02-20）

### 1) 单样本验证
- 脚本：`scripts/geometric_intervention_simple.py`
- 产物：`tempdata/geometric_intervention_results.json`
- 结果：在固定 prompt 上，干预前后输出发生变化（changed=1）

### 2) 批量验证（5 prompts）
- 脚本：`scripts/geometric_intervention_batch.py`
- 产物：`tempdata/geometric_intervention_batch_results_20260220.json`
- 结果：changed_rate = 0.60（3/5）

### 3) 时间线落盘
- analysis_type=causal_intervention: 1
- analysis_type=causal_intervention_batch: 1

### 4) 下一步
- 增加随机方向对照组（random direction / shuffled reference）
- 扩展到 50+ prompts，输出效应量区间与置信区间

## 继续执行：S1 批量干预随机对照补证（2026-02-20）

- 同脚本加入随机参考对照后复跑：`scripts/geometric_intervention_batch.py`
- treatment_changed_rate = 0.60
- control_changed_rate = 0.60
- causal_uplift = 0.00

解读：
- 当前 5 prompts 规模下，尚未体现几何参考相对随机对照的提升。
- 该结果将 S1 状态从“有方向信号”更新为“证据不足，需扩样本与分层扫描”。

## 继续执行：S1 多层扫描（n=60, 2026-02-20）

- 扫描层位：L3 / L6 / L9
- 对照：random_reference + shuffled_reference
- 汇总：`tempdata/geometric_intervention_layer_scan_20260220.json`

结果：
- L3 uplift(random)=+0.0167（微弱）
- L6 uplift(random)=0.0000
- L9 uplift(random)=0.0000

结论：
- 目前仅见弱信号，尚不足以确认稳健因果效应。
- 下一步应提升样本规模（n>=200）并扩展跨模型复验。
