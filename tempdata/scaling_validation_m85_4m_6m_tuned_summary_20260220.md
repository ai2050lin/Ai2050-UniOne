# m_8.5m 4M/6M tuned 扩展测试总结（2026-02-20）

来源：`tempdata/scaling_validation_report_m85_4m_6m_tuned_20260220.json`

## 结果
- d_4000k: best=1.0000, final=1.0000, sps=5959.57
- d_6000k: best=1.0000, final=1.0000, sps=5443.99

## 聚合
- best_val_acc_mean=1.0000
- final_val_acc_mean=1.0000
- samples_per_second_mean=5701.78

## 结论
在 tuned 配置下，m_8.5m 在 4M/6M 扩展点位继续稳定收敛，支持进入下一阶段（S1 因果干预批处理 + S3 一致性补证）。
