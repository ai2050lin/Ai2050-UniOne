# Stage138: 条件门控场重建块

## 核心结果
- 家族数量: 5
- 最优律: `field = 0.10*q + 0.10*b + 0.80*g`
- 相关系数: 0.8332
- 平均绝对误差: 0.0196
- 条件门控场分数: 0.9389
- 最强代理量: q_proxy_mean
- 最弱代理量: b_proxy_mean

## 代理量均值
- q_proxy_mean = 0.7273
- b_proxy_mean = 0.1054
- g_proxy_mean = 0.5462

## 各语篇家族
- causal_remention: target=0.5276, pred=0.5421, q=0.7256, b=0.1375, g=0.5697
- contrastive_remention: target=0.5396, pred=0.5187, q=0.7208, b=0.1006, g=0.5456
- cross_sentence_bridge: target=0.5377, pred=0.5587, q=0.7174, b=0.1505, g=0.5899
- discourse_remention: target=0.5204, pred=0.5190, q=0.6976, b=0.1065, g=0.5483
- nested_memory: target=0.5025, pred=0.4625, q=0.7750, b=0.0317, g=0.4773