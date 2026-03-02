# 数学编码原理测试报告 (MEPT)

## 1) 原理指标总览
- invariance_mean: 0.650705
- additivity_error_mean: 0.386335
- transport_error_mean: 0.264442
- lowrank_effective_rank_mean: 4.736828
- causal_target_delta_mean: -0.086182
- causal_control_delta_mean: -0.140625

## 2) 属性方向不变性
- color_red: mean_cos=0.691926, std=0.112262
- taste_sweet: mean_cos=0.640653, std=0.201267
- weight_heavy: mean_cos=0.604707, std=0.216073
- size_big: mean_cos=0.665534, std=0.162375

## 3) 可加性误差
- color_red+taste_sweet: mean_rel_err=0.392909, std=0.040826
- weight_heavy+size_big: mean_rel_err=0.379761, std=0.042665

## 4) 传输性误差 (apple->others)
- color_red: mean_rel_err=0.229766, std=0.038252
- taste_sweet: mean_rel_err=0.291259, std=0.083073
- weight_heavy: mean_rel_err=0.303241, std=0.113046
- size_big: mean_rel_err=0.233500, std=0.059808

## 5) 低秩性
- color_red: effective_rank=6.2754, k95=9
- taste_sweet: effective_rank=3.9125, k95=8
- weight_heavy: effective_rank=3.9166, k95=8
- size_big: effective_rank=4.8428, k95=9

## 6) 因果可控性 (apple对齐)
- color_red: target_delta=+0.042969, control_delta=+0.125000, layers={1: 2, 2: 1, 20: 1, 21: 2, 22: 4, 23: 4, 24: 3, 25: 5, 26: 2}
- taste_sweet: target_delta=-0.391602, control_delta=+0.054688, layers={2: 9, 3: 4, 5: 1, 22: 2, 23: 4, 24: 2, 26: 1, 27: 1}
- weight_heavy: target_delta=-0.019531, control_delta=-0.664062, layers={27: 24}
- size_big: target_delta=+0.023438, control_delta=-0.078125, layers={1: 1, 2: 1, 3: 1, 22: 4, 23: 4, 24: 3, 25: 3, 26: 1, 27: 6}