# stage126_verb_neuron_basic_probe: Verb 神经元基础探针

## 核心结果
- 目标词类: verb
- 目标数量: 866
- 控制组数量: 29675
- 层数: 12
- 每层神经元数: 3072
- 目标语义组数量: 4
- 主导通用层: L11
- 主导通用层分数: 0.5370
- 主要 band（尺度带）: macro
- 次要 band（尺度带）: meso
- 词类神经元基础探针分数: 0.9836

## 顶级通用神经元
- L0 N1528: rule=0.7298, effect=1.2756, support=1.0000, dominant_group=macro_action
- L1 N1680: rule=0.7209, effect=1.2767, support=1.0000, dominant_group=macro_action
- L1 N2807: rule=0.6470, effect=1.1062, support=1.0000, dominant_group=meso_tech
- L11 N785: rule=0.6421, effect=1.0098, support=1.0000, dominant_group=macro_action
- L11 N2913: rule=0.6128, effect=0.9085, support=1.0000, dominant_group=macro_action
- L11 N863: rule=0.5907, effect=1.0076, support=0.7500, dominant_group=macro_action
- L11 N1122: rule=0.5799, effect=0.8315, support=1.0000, dominant_group=macro_action
- L11 N1916: rule=0.5769, effect=0.8403, support=1.0000, dominant_group=macro_action
- L7 N909: rule=0.5732, effect=0.9514, support=1.0000, dominant_group=meso_tech
- L8 N1177: rule=0.5618, effect=0.8111, support=1.0000, dominant_group=macro_action
- L11 N1780: rule=0.5541, effect=0.8213, support=1.0000, dominant_group=meso_tech
- L6 N1878: rule=0.5539, effect=0.8307, support=1.0000, dominant_group=macro_action

## macro / meso 分裂神经元
- macro L10 N689: bias=0.2509, effect=0.8273, dominant_group=macro_action
- macro L9 N2985: bias=0.1575, effect=0.7877, dominant_group=macro_action
- macro L11 N44: bias=0.1226, effect=1.0005, dominant_group=macro_action
- macro L10 N2355: bias=0.2155, effect=0.5134, dominant_group=macro_action
- macro L6 N2902: bias=0.1119, effect=0.9092, dominant_group=macro_action
- macro L8 N1684: bias=0.1151, effect=0.8297, dominant_group=macro_action
- meso L2 N666: bias=13.8766, effect=0.2209, dominant_group=meso_tech
- meso L2 N1825: bias=12.9771, effect=0.2209, dominant_group=meso_tech
- meso L1 N1120: bias=4.1801, effect=0.2454, dominant_group=meso_tech
- meso L2 N3034: bias=4.2866, effect=0.2040, dominant_group=meso_tech
- meso L0 N566: bias=0.8170, effect=0.8613, dominant_group=meso_tech
- meso L3 N1751: bias=2.4332, effect=0.2118, dominant_group=meso_tech

## 理论提示
- 如果顶级神经元同时具有较高 effect（效应量）和较高 group support（组覆盖），说明该词类内部存在跨家族共享的较一般编码规则。
- 如果主要 band 与次要 band 的分裂明显，说明该词类内部至少包含两条不同尺度的编码链，而不是单一平面。
