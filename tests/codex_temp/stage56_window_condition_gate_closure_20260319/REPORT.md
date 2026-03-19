# Stage56 窗口条件门收口摘要

- row_count: 72
- main_judgment: 窗口条件门已被继续拆成风格正核/负核和前沿迁移/基础两类子门，可以直接判断窗口层到底在放大正迁移，还是在放大旧的负核基础。

## Stable Features
- window_style_positive_term: positive
- window_style_negative_term: negative
- window_frontier_positive_term: negative
- window_frontier_negative_term: negative

## Fits
- target: union_joint_adv
  intercept: -0.127748
  window_style_positive_term: +14.100872
  window_style_negative_term: -7.068951
  window_frontier_positive_term: -4.802209
  window_frontier_negative_term: -5.306829
- target: union_synergy_joint
  intercept: -0.061375
  window_style_positive_term: +10.727556
  window_style_negative_term: -5.377914
  window_frontier_positive_term: -5.261135
  window_frontier_negative_term: -1.895959
- target: strict_positive_synergy
  intercept: -0.762602
  window_style_positive_term: +38.836858
  window_style_negative_term: -19.475098
  window_frontier_positive_term: -17.073493
  window_frontier_negative_term: -4.043859
