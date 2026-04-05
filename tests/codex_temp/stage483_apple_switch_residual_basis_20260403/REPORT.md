# stage483_apple_switch_residual_basis

## 实验设置
- 时间戳: 2026-04-03T01:12:39Z
- 是否使用 CUDA: False
- 目标: 提取苹果切换核心单元改写的主残差方向，并比较其与水果义/品牌义切换轴的对齐程度

## 模型 qwen3
- 单元 H:5:2 (skeleton_head_1):
  peak_contrast_layer = L35
  peak_contrast_switch_coupling = +0.0160
  peak_contrast_switch_abs_cos = +0.4330
  peak_pc1_layer = L35
  peak_pc1_switch_coupling = +0.0163
  peak_pc1_switch_abs_cos = +0.4405
  peak_pc1_explained_variance_ratio = +0.7496
  late_mean_contrast_switch_coupling = +0.0057
  late_mean_pc1_switch_coupling = +0.0061
- 单元 H:5:29 (skeleton_head_2):
  peak_contrast_layer = L35
  peak_contrast_switch_coupling = +0.0147
  peak_contrast_switch_abs_cos = +0.4927
  peak_pc1_layer = L35
  peak_pc1_switch_coupling = +0.0150
  peak_pc1_switch_abs_cos = +0.5020
  peak_pc1_explained_variance_ratio = +0.6875
  late_mean_contrast_switch_coupling = +0.0044
  late_mean_pc1_switch_coupling = +0.0048
- 单元 H:5:9 (bridge_head):
  peak_contrast_layer = L35
  peak_contrast_switch_coupling = +0.0182
  peak_contrast_switch_abs_cos = +0.5002
  peak_pc1_layer = L35
  peak_pc1_switch_coupling = +0.0182
  peak_pc1_switch_abs_cos = +0.5017
  peak_pc1_explained_variance_ratio = +0.7436
  late_mean_contrast_switch_coupling = +0.0059
  late_mean_pc1_switch_coupling = +0.0064
- 单元 H:5:8 (heldout_booster):
  peak_contrast_layer = L35
  peak_contrast_switch_coupling = +0.0129
  peak_contrast_switch_abs_cos = +0.4129
  peak_pc1_layer = L35
  peak_pc1_switch_coupling = +0.0129
  peak_pc1_switch_abs_cos = +0.4146
  peak_pc1_explained_variance_ratio = +0.6498
  late_mean_contrast_switch_coupling = +0.0058
  late_mean_pc1_switch_coupling = +0.0077

## 模型 deepseek7b
- 单元 N:2:16785 (anchor_neuron):
  peak_contrast_layer = L4
  peak_contrast_switch_coupling = +0.1526
  peak_contrast_switch_abs_cos = +0.3919
  peak_pc1_layer = L4
  peak_pc1_switch_coupling = +0.1525
  peak_pc1_switch_abs_cos = +0.3916
  peak_pc1_explained_variance_ratio = +0.9845
  late_mean_contrast_switch_coupling = +0.0439
  late_mean_pc1_switch_coupling = +0.0357
- 单元 H:2:22 (main_booster_1):
  peak_contrast_layer = L27
  peak_contrast_switch_coupling = +0.0093
  peak_contrast_switch_abs_cos = +0.0576
  peak_pc1_layer = L19
  peak_pc1_switch_coupling = +0.0085
  peak_pc1_switch_abs_cos = +0.2481
  peak_pc1_explained_variance_ratio = +0.4570
  late_mean_contrast_switch_coupling = +0.0039
  late_mean_pc1_switch_coupling = +0.0022
- 单元 H:2:10 (main_booster_2):
  peak_contrast_layer = L27
  peak_contrast_switch_coupling = +0.0193
  peak_contrast_switch_abs_cos = +0.1839
  peak_pc1_layer = L5
  peak_pc1_switch_coupling = +0.0124
  peak_pc1_switch_abs_cos = +0.1421
  peak_pc1_explained_variance_ratio = +0.8785
  late_mean_contrast_switch_coupling = +0.0077
  late_mean_pc1_switch_coupling = +0.0051
- 单元 H:2:26 (heldout_booster):
  peak_contrast_layer = L2
  peak_contrast_switch_coupling = +0.0226
  peak_contrast_switch_abs_cos = +0.2027
  peak_pc1_layer = L2
  peak_pc1_switch_coupling = +0.0238
  peak_pc1_switch_abs_cos = +0.2139
  peak_pc1_explained_variance_ratio = +0.7214
  late_mean_contrast_switch_coupling = +0.0041
  late_mean_pc1_switch_coupling = +0.0038
