# Stage123: Route Shift 层定位分析

## 核心结果
- 来源样本对数量: 144
- 来源动态分数: 0.6131
- 主导层: L3
- 主导层带: early
- 主导峰值层: L3
- 最早稳定层: L3
- 最强层带: early
- 层定位分数: 0.6386

## 逐层前五名
- L3 (early): route_adv=0.004937, peak_hit=0.5903, attn_adv=0.069963
- L4 (middle): route_adv=0.002835, peak_hit=0.2222, attn_adv=0.100792
- L2 (early): route_adv=0.001492, peak_hit=0.0278, attn_adv=0.061182
- L10 (late): route_adv=0.000939, peak_hit=0.0625, attn_adv=0.000214
- L11 (late): route_adv=0.000499, peak_hit=0.0069, attn_adv=0.028268

## 层带汇总
- early: route_adv=0.001368, attn_adv=0.060800, peak_hit_mean=0.1545
- middle: route_adv=-0.000009, attn_adv=0.056409, peak_hit_mean=0.0608
- late: route_adv=-0.000540, attn_adv=0.019369, peak_hit_mean=0.0347

## 理论提示
- 如果偏移集中在中后层，说明副词更像在上下文整合后改变选路，而不是只在最早词法层起作用。
- 如果最早稳定层明显早于主导峰值层，说明副词效应可能先萌发，再在更深层完成放大。
