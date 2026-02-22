# m_8.5m + 2M tuned 多种子复现实验（2026-02-20）

- 目标：验证 tuned 配置在大规模点位（2M）下的复现稳定性
- 配置：epochs=20, lr=0.0003, weight_decay=0.01, warmup_ratio=0.03, min_lr_scale=0.1

## 结果汇总
- runs: 3
- best_val_acc(mean±std): 1 ± 0
- final_val_acc(mean±std): 1 ± 0
- samples_per_second(mean±std): 6149.94 ± 27.88
- train_seconds(mean±std): 650.42 ± 2.95

## 单次运行
- seed=10018, best=1, final=1, sps=6150.41, report=tempdata/scaling_validation_report_m85_2m_tuned_seed10001_20260220.json
- seed=20019, best=1, final=1, sps=6177.59, report=tempdata/scaling_validation_report_m85_2m_tuned_seed20002_20260220.json
- seed=30020, best=1, final=1, sps=6121.83, report=tempdata/scaling_validation_report_m85_2m_tuned_seed30003_20260220.json

## 结论
3-seed tuned 配置下结果稳定（val_acc 全部为 1.0），当前大规模训练瓶颈主要不是结构表达能力，而是优化配置。

