# stage484_apple_switch_signed_residual_basis

## 实验设置
- 时间戳: 2026-04-03T01:46:05Z
- 是否使用 CUDA: False
- 目标: 判断苹果切换核心单元的主残差方向，是顺着切换轴推进，还是反着切换轴抵消

## 模型 qwen3
- 单元 H:5:2 (skeleton_head_1):
  forward_peak_layer = L5
  forward_peak_signed_contrast_switch_coupling = +0.0140
  forward_peak_signed_contrast_switch_cos = +0.2251
  reverse_peak_layer = L35
  reverse_peak_signed_contrast_switch_coupling = -0.0160
  reverse_peak_signed_contrast_switch_cos = -0.4330
  late_mean_signed_contrast_switch_coupling = -0.0057
- 单元 H:5:29 (skeleton_head_2):
  forward_peak_layer = L14
  forward_peak_signed_contrast_switch_coupling = +0.0049
  forward_peak_signed_contrast_switch_cos = +0.1972
  reverse_peak_layer = L35
  reverse_peak_signed_contrast_switch_coupling = -0.0147
  reverse_peak_signed_contrast_switch_cos = -0.4927
  late_mean_signed_contrast_switch_coupling = -0.0044
- 单元 H:5:9 (bridge_head):
  forward_peak_layer = L17
  forward_peak_signed_contrast_switch_coupling = +0.0025
  forward_peak_signed_contrast_switch_cos = +0.0852
  reverse_peak_layer = L35
  reverse_peak_signed_contrast_switch_coupling = -0.0182
  reverse_peak_signed_contrast_switch_cos = -0.5002
  late_mean_signed_contrast_switch_coupling = -0.0059
- 单元 H:5:8 (heldout_booster):
  forward_peak_layer = L0
  forward_peak_signed_contrast_switch_coupling = +0.0000
  forward_peak_signed_contrast_switch_cos = +0.0000
  reverse_peak_layer = L35
  reverse_peak_signed_contrast_switch_coupling = -0.0129
  reverse_peak_signed_contrast_switch_cos = -0.4129
  late_mean_signed_contrast_switch_coupling = -0.0058

## 模型 deepseek7b
- 单元 N:2:16785 (anchor_neuron):
  forward_peak_layer = L0
  forward_peak_signed_contrast_switch_coupling = +0.0000
  forward_peak_signed_contrast_switch_cos = +0.0000
  reverse_peak_layer = L4
  reverse_peak_signed_contrast_switch_coupling = -0.1526
  reverse_peak_signed_contrast_switch_cos = -0.3919
  late_mean_signed_contrast_switch_coupling = -0.0439
- 单元 H:2:22 (main_booster_1):
  forward_peak_layer = L8
  forward_peak_signed_contrast_switch_coupling = +0.0029
  forward_peak_signed_contrast_switch_cos = +0.0599
  reverse_peak_layer = L27
  reverse_peak_signed_contrast_switch_coupling = -0.0093
  reverse_peak_signed_contrast_switch_cos = -0.0576
  late_mean_signed_contrast_switch_coupling = -0.0039
- 单元 H:2:10 (main_booster_2):
  forward_peak_layer = L0
  forward_peak_signed_contrast_switch_coupling = +0.0000
  forward_peak_signed_contrast_switch_cos = +0.0000
  reverse_peak_layer = L27
  reverse_peak_signed_contrast_switch_coupling = -0.0193
  reverse_peak_signed_contrast_switch_cos = -0.1839
  late_mean_signed_contrast_switch_coupling = -0.0077
- 单元 H:2:26 (heldout_booster):
  forward_peak_layer = L27
  forward_peak_signed_contrast_switch_coupling = +0.0101
  forward_peak_signed_contrast_switch_cos = +0.0566
  reverse_peak_layer = L2
  reverse_peak_signed_contrast_switch_coupling = -0.0226
  reverse_peak_signed_contrast_switch_cos = -0.2027
  late_mean_signed_contrast_switch_coupling = +0.0041
