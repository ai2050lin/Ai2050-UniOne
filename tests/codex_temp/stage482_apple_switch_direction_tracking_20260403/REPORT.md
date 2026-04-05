# stage482_apple_switch_direction_tracking

## 实验设置
- 时间戳: 2026-04-03T01:03:29Z
- 是否使用 CUDA: False
- 目标: 追踪苹果切换核心单元对后续层切换轴分离度的影响

## 模型 qwen3
- 单元 H:5:2 (skeleton_head_1):
  peak_layer = L35
  peak_drop = +2.5112
  peak_relative_drop = +0.0160
  late_mean_relative_drop = +0.0057
- 单元 H:5:29 (skeleton_head_2):
  peak_layer = L35
  peak_drop = +2.3066
  peak_relative_drop = +0.0147
  late_mean_relative_drop = +0.0044
- 单元 H:5:9 (bridge_head):
  peak_layer = L35
  peak_drop = +2.8520
  peak_relative_drop = +0.0182
  late_mean_relative_drop = +0.0059
- 单元 H:5:8 (heldout_booster):
  peak_layer = L35
  peak_drop = +2.0186
  peak_relative_drop = +0.0129
  late_mean_relative_drop = +0.0058

## 模型 deepseek7b
- 单元 N:2:16785 (anchor_neuron):
  peak_layer = L26
  peak_drop = +39.1103
  peak_relative_drop = +0.0433
  late_mean_relative_drop = +0.0439
- 单元 H:2:22 (main_booster_1):
  peak_layer = L22
  peak_drop = +2.7946
  peak_relative_drop = +0.0064
  late_mean_relative_drop = +0.0039
- 单元 H:2:10 (main_booster_2):
  peak_layer = L27
  peak_drop = +3.8466
  peak_relative_drop = +0.0193
  late_mean_relative_drop = +0.0077
- 单元 H:2:26 (heldout_booster):
  peak_layer = L26
  peak_drop = -2.9201
  peak_relative_drop = -0.0032
  late_mean_relative_drop = -0.0041
