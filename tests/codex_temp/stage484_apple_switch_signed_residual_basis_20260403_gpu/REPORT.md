# stage484_apple_switch_signed_residual_basis

## 实验设置
- 时间戳: 2026-04-03T01:46:36Z
- 是否使用 CUDA: True
- 目标: 判断苹果切换核心单元的主残差方向，是顺着切换轴推进，还是反着切换轴抵消

## 模型 qwen3
- 单元 H:5:2 (skeleton_head_1):
  forward_peak_layer = L5
  forward_peak_signed_contrast_switch_coupling = +0.0121
  forward_peak_signed_contrast_switch_cos = +0.2001
  reverse_peak_layer = L35
  reverse_peak_signed_contrast_switch_coupling = -0.0161
  reverse_peak_signed_contrast_switch_cos = -0.3873
  late_mean_signed_contrast_switch_coupling = -0.0069
- 单元 H:5:29 (skeleton_head_2):
  forward_peak_layer = L24
  forward_peak_signed_contrast_switch_coupling = +0.0024
  forward_peak_signed_contrast_switch_cos = +0.1153
  reverse_peak_layer = L35
  reverse_peak_signed_contrast_switch_coupling = -0.0131
  reverse_peak_signed_contrast_switch_cos = -0.4477
  late_mean_signed_contrast_switch_coupling = -0.0047
- 单元 H:5:9 (bridge_head):
  forward_peak_layer = L4
  forward_peak_signed_contrast_switch_coupling = +0.0005
  forward_peak_signed_contrast_switch_cos = +0.1637
  reverse_peak_layer = L5
  reverse_peak_signed_contrast_switch_coupling = -0.0140
  reverse_peak_signed_contrast_switch_cos = -0.2761
  late_mean_signed_contrast_switch_coupling = -0.0057
- 单元 H:5:8 (heldout_booster):
  forward_peak_layer = L4
  forward_peak_signed_contrast_switch_coupling = +0.0005
  forward_peak_signed_contrast_switch_cos = +0.1637
  reverse_peak_layer = L9
  reverse_peak_signed_contrast_switch_coupling = -0.0078
  reverse_peak_signed_contrast_switch_cos = -0.1823
  late_mean_signed_contrast_switch_coupling = -0.0040

## 模型 deepseek7b
- 单元 N:2:16785 (anchor_neuron):
  forward_peak_layer = L27
  forward_peak_signed_contrast_switch_coupling = +0.0162
  forward_peak_signed_contrast_switch_cos = +0.0506
  reverse_peak_layer = L4
  reverse_peak_signed_contrast_switch_coupling = -0.1537
  reverse_peak_signed_contrast_switch_cos = -0.3862
  late_mean_signed_contrast_switch_coupling = -0.0322
- 单元 H:2:22 (main_booster_1):
  forward_peak_layer = L3
  forward_peak_signed_contrast_switch_coupling = +0.0039
  forward_peak_signed_contrast_switch_cos = +0.0638
  reverse_peak_layer = L19
  reverse_peak_signed_contrast_switch_coupling = -0.0072
  reverse_peak_signed_contrast_switch_cos = -0.2333
  late_mean_signed_contrast_switch_coupling = -0.0033
- 单元 H:2:10 (main_booster_2):
  forward_peak_layer = L27
  forward_peak_signed_contrast_switch_coupling = +0.0003
  forward_peak_signed_contrast_switch_cos = +0.0023
  reverse_peak_layer = L5
  reverse_peak_signed_contrast_switch_coupling = -0.0123
  reverse_peak_signed_contrast_switch_cos = -0.1408
  late_mean_signed_contrast_switch_coupling = -0.0014
- 单元 H:2:26 (heldout_booster):
  forward_peak_layer = L27
  forward_peak_signed_contrast_switch_coupling = +0.0076
  forward_peak_signed_contrast_switch_cos = +0.0331
  reverse_peak_layer = L2
  reverse_peak_signed_contrast_switch_coupling = -0.0221
  reverse_peak_signed_contrast_switch_cos = -0.2157
  late_mean_signed_contrast_switch_coupling = +0.0027
