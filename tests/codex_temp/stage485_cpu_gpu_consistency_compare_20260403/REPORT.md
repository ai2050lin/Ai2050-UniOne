# stage485_cpu_gpu_consistency_compare

## 汇总
- max_late_mean_abs_diff = 0.011670
- max_forward_peak_abs_diff = 0.016190
- max_reverse_peak_abs_diff = 0.006991
- peak_layer_match_rate = 0.3750

## 模型 qwen3
- 单元 H:5:2 (skeleton_head_1):
  forward_layer cpu/gpu = L5 / L5
  reverse_layer cpu/gpu = L35 / L35
  late_mean_abs_diff = 0.001162
  forward_peak_abs_diff = 0.001987
  reverse_peak_abs_diff = 0.000087
- 单元 H:5:29 (skeleton_head_2):
  forward_layer cpu/gpu = L14 / L24
  reverse_layer cpu/gpu = L35 / L35
  late_mean_abs_diff = 0.000346
  forward_peak_abs_diff = 0.002524
  reverse_peak_abs_diff = 0.001617
- 单元 H:5:9 (bridge_head):
  forward_layer cpu/gpu = L17 / L4
  reverse_layer cpu/gpu = L35 / L5
  late_mean_abs_diff = 0.000245
  forward_peak_abs_diff = 0.002007
  reverse_peak_abs_diff = 0.004178
- 单元 H:5:8 (heldout_booster):
  forward_layer cpu/gpu = L0 / L4
  reverse_layer cpu/gpu = L35 / L9
  late_mean_abs_diff = 0.001777
  forward_peak_abs_diff = 0.000531
  reverse_peak_abs_diff = 0.005046

## 模型 deepseek7b
- 单元 N:2:16785 (anchor_neuron):
  forward_layer cpu/gpu = L0 / L27
  reverse_layer cpu/gpu = L4 / L4
  late_mean_abs_diff = 0.011670
  forward_peak_abs_diff = 0.016190
  reverse_peak_abs_diff = 0.001057
- 单元 H:2:22 (main_booster_1):
  forward_layer cpu/gpu = L8 / L3
  reverse_layer cpu/gpu = L27 / L19
  late_mean_abs_diff = 0.000657
  forward_peak_abs_diff = 0.001020
  reverse_peak_abs_diff = 0.002061
- 单元 H:2:10 (main_booster_2):
  forward_layer cpu/gpu = L0 / L27
  reverse_layer cpu/gpu = L27 / L5
  late_mean_abs_diff = 0.006343
  forward_peak_abs_diff = 0.000280
  reverse_peak_abs_diff = 0.006991
- 单元 H:2:26 (heldout_booster):
  forward_layer cpu/gpu = L27 / L27
  reverse_layer cpu/gpu = L2 / L2
  late_mean_abs_diff = 0.001389
  forward_peak_abs_diff = 0.002523
  reverse_peak_abs_diff = 0.000483
