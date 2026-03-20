# 异质资产阶段顺序诊断报告

- 混合资产顺序支持: 0.200000
- 长程同质资产顺序支持: 1.000000

## 个案
- icspb_phasea: ordered=True, diagnosis=样本太短，三阶段几乎同时触发
- toy_transformer: ordered=False, diagnosis=任务过于简化，图册与边界信号混叠
- toy_fibernet: ordered=False, diagnosis=任务过于简化，图册与边界信号混叠
- glm5_z113_visuals: ordered=False, diagnosis=可视化训练日志以泛化曲线为主，边界代理和前沿代理不共尺度
- openwebtext_backbone_v2: ordered=False, diagnosis=只含短骨干块，缺少长程边界和图册后段
