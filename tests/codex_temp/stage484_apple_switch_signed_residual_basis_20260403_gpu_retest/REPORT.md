# stage484_apple_switch_signed_residual_basis

## 实验设置
- 时间戳: 2026-04-03T13:26:51Z
- 是否使用 CUDA: True
- 目标: 判断苹果切换核心单元的主残差方向，是顺着切换轴推进，还是反着切换轴抵消

## 模型 qwen3
- 单元 H:5:2 (skeleton_head_1):
  forward_peak_layer = L5
  forward_peak_signed_contrast_switch_coupling = +0.0130
  forward_peak_signed_contrast_switch_cos = +0.2162
  reverse_peak_layer = L35
  reverse_peak_signed_contrast_switch_coupling = -0.0164
  reverse_peak_signed_contrast_switch_cos = -0.3719
  late_mean_signed_contrast_switch_coupling = -0.0066
- 单元 H:5:29 (skeleton_head_2):
  forward_peak_layer = L24
  forward_peak_signed_contrast_switch_coupling = +0.0027
  forward_peak_signed_contrast_switch_cos = +0.1159
  reverse_peak_layer = L35
  reverse_peak_signed_contrast_switch_coupling = -0.0135
  reverse_peak_signed_contrast_switch_cos = -0.3879
  late_mean_signed_contrast_switch_coupling = -0.0037
- 单元 H:5:9 (bridge_head):
  forward_peak_layer = L0
  forward_peak_signed_contrast_switch_coupling = +0.0000
  forward_peak_signed_contrast_switch_cos = +0.0000
  reverse_peak_layer = L5
  reverse_peak_signed_contrast_switch_coupling = -0.0137
  reverse_peak_signed_contrast_switch_cos = -0.2719
  late_mean_signed_contrast_switch_coupling = -0.0042
- 单元 H:5:8 (heldout_booster):
  forward_peak_layer = L17
  forward_peak_signed_contrast_switch_coupling = +0.0000
  forward_peak_signed_contrast_switch_cos = +0.0009
  reverse_peak_layer = L9
  reverse_peak_signed_contrast_switch_coupling = -0.0078
  reverse_peak_signed_contrast_switch_cos = -0.1808
  late_mean_signed_contrast_switch_coupling = -0.0030

## 模型 deepseek7b
- 单元 N:2:16785 (anchor_neuron):
  forward_peak_layer = L0
  forward_peak_signed_contrast_switch_coupling = +0.0000
  forward_peak_signed_contrast_switch_cos = +0.0000
  reverse_peak_layer = L4
  reverse_peak_signed_contrast_switch_coupling = -0.1537
  reverse_peak_signed_contrast_switch_cos = -0.3862
  late_mean_signed_contrast_switch_coupling = -0.0387
- 单元 H:2:22 (main_booster_1):
  forward_peak_layer = L3
  forward_peak_signed_contrast_switch_coupling = +0.0039
  forward_peak_signed_contrast_switch_cos = +0.0638
  reverse_peak_layer = L19
  reverse_peak_signed_contrast_switch_coupling = -0.0076
  reverse_peak_signed_contrast_switch_cos = -0.2424
  late_mean_signed_contrast_switch_coupling = -0.0046
- 单元 H:2:10 (main_booster_2):
  forward_peak_layer = L27
  forward_peak_signed_contrast_switch_coupling = +0.0070
  forward_peak_signed_contrast_switch_cos = +0.0657
  reverse_peak_layer = L5
  reverse_peak_signed_contrast_switch_coupling = -0.0123
  reverse_peak_signed_contrast_switch_cos = -0.1410
  late_mean_signed_contrast_switch_coupling = +0.0002
- 单元 H:2:26 (heldout_booster):
  forward_peak_layer = L15
  forward_peak_signed_contrast_switch_coupling = +0.0022
  forward_peak_signed_contrast_switch_cos = +0.0494
  reverse_peak_layer = L2
  reverse_peak_signed_contrast_switch_coupling = -0.0221
  reverse_peak_signed_contrast_switch_cos = -0.2157
  late_mean_signed_contrast_switch_coupling = +0.0014
