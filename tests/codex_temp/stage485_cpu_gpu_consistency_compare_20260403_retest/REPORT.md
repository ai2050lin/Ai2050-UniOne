# stage485_cpu_gpu_consistency_compare

## 汇总
- max_late_mean_abs_diff = 0.007875
- max_forward_peak_abs_diff = 0.007878
- max_reverse_peak_abs_diff = 0.006976
- peak_layer_match_rate = 0.3750

## 模型 qwen3
- 单元 H:5:2 (skeleton_head_1):
  forward_layer cpu/gpu = L5 / L5
  reverse_layer cpu/gpu = L35 / L35
  late_mean_abs_diff = 0.000857
  forward_peak_abs_diff = 0.001064
  reverse_peak_abs_diff = 0.000400
- 单元 H:5:29 (skeleton_head_2):
  forward_layer cpu/gpu = L14 / L24
  reverse_layer cpu/gpu = L35 / L35
  late_mean_abs_diff = 0.000707
  forward_peak_abs_diff = 0.002239
  reverse_peak_abs_diff = 0.001221
- 单元 H:5:9 (bridge_head):
  forward_layer cpu/gpu = L17 / L0
  reverse_layer cpu/gpu = L35 / L5
  late_mean_abs_diff = 0.001748
  forward_peak_abs_diff = 0.002538
  reverse_peak_abs_diff = 0.004505
- 单元 H:5:8 (heldout_booster):
  forward_layer cpu/gpu = L0 / L17
  reverse_layer cpu/gpu = L35 / L9
  late_mean_abs_diff = 0.002828
  forward_peak_abs_diff = 0.000027
  reverse_peak_abs_diff = 0.005115

## 模型 deepseek7b
- 单元 N:2:16785 (anchor_neuron):
  forward_layer cpu/gpu = L0 / L0
  reverse_layer cpu/gpu = L4 / L4
  late_mean_abs_diff = 0.005224
  forward_peak_abs_diff = 0.000000
  reverse_peak_abs_diff = 0.001060
- 单元 H:2:22 (main_booster_1):
  forward_layer cpu/gpu = L8 / L3
  reverse_layer cpu/gpu = L27 / L19
  late_mean_abs_diff = 0.000645
  forward_peak_abs_diff = 0.001020
  reverse_peak_abs_diff = 0.001682
- 单元 H:2:10 (main_booster_2):
  forward_layer cpu/gpu = L0 / L27
  reverse_layer cpu/gpu = L27 / L5
  late_mean_abs_diff = 0.007875
  forward_peak_abs_diff = 0.007015
  reverse_peak_abs_diff = 0.006976
- 单元 H:2:26 (heldout_booster):
  forward_layer cpu/gpu = L27 / L15
  reverse_layer cpu/gpu = L2 / L2
  late_mean_abs_diff = 0.002701
  forward_peak_abs_diff = 0.007878
  reverse_peak_abs_diff = 0.000483
