# stage490_cpu_gpu_dual_path_formal_protocol

## 核心回答
当前正式协议不再把 CPU 与 GPU 当作可随意替换的同一路径。它们在趋势级结论上可以互相支撑，但在峰值层位和小信号上仍然存在明显漂移，因此研究工作必须采用双路径治理。

## 阈值
- trend_max_late_mean_abs_diff = 0.01
- magnitude_max_peak_abs_diff = 0.01
- peak_layer_match_rate = 0.75

## 对照结果
- cpu_vs_gpu_original:
  - max_late_mean_abs_diff = 0.011670
  - max_forward_peak_abs_diff = 0.016190
  - max_reverse_peak_abs_diff = 0.006991
  - peak_layer_match_rate = 0.3750
  - recommendation = 当前不应单独信任 GPU 路径
- cpu_vs_gpu_retest:
  - max_late_mean_abs_diff = 0.007875
  - max_forward_peak_abs_diff = 0.007878
  - max_reverse_peak_abs_diff = 0.006976
  - peak_layer_match_rate = 0.3750
  - recommendation = 可用于趋势验证，但峰值层位必须保留 CPU 复核
- gpu_original_vs_gpu_retest:
  - max_late_mean_abs_diff = 0.006446
  - max_forward_peak_abs_diff = 0.016190
  - max_reverse_peak_abs_diff = 0.000396
  - peak_layer_match_rate = 0.7500
  - recommendation = 当前不应单独信任 GPU 路径

## 正式协议
- 先在 GPU 上跑趋势级实验，用于快速筛选方向。
- 如果趋势级差异通过阈值，再把关键结论在 CPU 上复核。
- 所有峰值层位、最强单元、细粒度方向耦合结论，默认必须保留 CPU 复核。
- 如果 GPU 与 GPU 重跑之间漂移超过阈值，则暂停把 GPU 结果当作研究基准。
