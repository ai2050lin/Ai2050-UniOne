# Stage132: 名词统一变量拟合

## 核心结果
- 家族数量: 6
- 参考律: `noun_core = 0.40*s + 0.40*c + 0.20*b`
- 最优律: `noun_proxy = 0.50*a + 0.10*q + 0.20*g + 0.20*f`
- 相关系数: 0.9983
- 平均绝对误差: 0.0137
- 拟合分数: 0.9947
- 最弱代理量: g_proxy_mean

## 代理量均值
- a_proxy_mean = 0.5506
- q_proxy_mean = 0.6734
- g_proxy_mean = 0.4288
- f_proxy_mean = 0.4665

## 各句法簇
- subject_copula: target=0.5192, pred=0.5434, a=0.5603, q=0.6659, g=0.5065, f=0.4768
- object_transitive: target=0.5065, pred=0.5161, a=0.5417, q=0.7078, g=0.3815, f=0.4910
- preposition_about: target=0.4986, pred=0.4944, a=0.5383, q=0.7257, g=0.3298, f=0.4334
- relative_clause: target=0.5192, pred=0.5434, a=0.5603, q=0.6659, g=0.5065, f=0.4768
- possessive_frame: target=0.5140, pred=0.5333, a=0.5631, q=0.5749, g=0.5144, f=0.4568
- evaluation_frame: target=0.5002, pred=0.4996, a=0.5399, q=0.7002, g=0.3344, f=0.4638

## 理论提示
- a 代理量对应句中名词位的早层定锚强度。
- q 代理量对应早层到后层的保持性，也就是上下文保形能力。
- g 代理量对应 L1 到 L3 的前向选路耦合。
- f 代理量对应 L3 到 L11 的跨层续接强度。
