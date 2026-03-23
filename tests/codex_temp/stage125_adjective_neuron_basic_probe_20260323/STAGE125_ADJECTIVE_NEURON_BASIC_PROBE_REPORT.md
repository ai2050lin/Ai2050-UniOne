# stage125_adjective_neuron_basic_probe: Adjective 神经元基础探针

## 核心结果
- 目标词类: adjective
- 目标数量: 1034
- 控制组数量: 29507
- 层数: 12
- 每层神经元数: 3072
- 目标语义组数量: 10
- 主导通用层: L11
- 主导通用层分数: 0.4980
- 主要 band（尺度带）: micro
- 次要 band（尺度带）: macro
- 词类神经元基础探针分数: 0.9200

## 顶级通用神经元
- L11 N2714: rule=0.6327, effect=1.0803, support=1.0000, dominant_group=micro_material
- L11 N2605: rule=0.6065, effect=1.1876, support=1.0000, dominant_group=micro_taste
- L7 N2602: rule=0.5793, effect=1.0374, support=1.0000, dominant_group=micro_taste
- L11 N1117: rule=0.5692, effect=0.9632, support=0.9000, dominant_group=micro_taste
- L8 N1581: rule=0.5662, effect=0.9937, support=0.8000, dominant_group=micro_taste
- L7 N2942: rule=0.5499, effect=0.7982, support=1.0000, dominant_group=micro_taste
- L9 N569: rule=0.5438, effect=0.7755, support=1.0000, dominant_group=micro_taste
- L6 N752: rule=0.5424, effect=0.8665, support=0.9000, dominant_group=micro_taste
- L7 N1444: rule=0.5418, effect=0.7100, support=0.8000, dominant_group=macro_abstract
- L7 N1655: rule=0.5346, effect=0.8192, support=0.9000, dominant_group=micro_taste
- L11 N588: rule=0.5330, effect=0.8197, support=1.0000, dominant_group=micro_taste
- L7 N250: rule=0.5309, effect=0.8361, support=0.9000, dominant_group=micro_taste

## micro / macro 分裂神经元
- micro L0 N125: bias=0.1922, effect=0.3607, dominant_group=micro_color
- micro L0 N1542: bias=0.1594, effect=0.3940, dominant_group=micro_shape
- micro L6 N752: bias=0.0664, effect=0.8665, dominant_group=micro_taste
- micro L9 N636: bias=0.0533, effect=1.0440, dominant_group=micro_taste
- micro L5 N1948: bias=0.0797, effect=0.6287, dominant_group=micro_material
- micro L0 N1155: bias=0.1522, effect=0.2765, dominant_group=micro_size
- macro L2 N666: bias=3.3866, effect=0.1312, dominant_group=macro_system
- macro L2 N1825: bias=3.1662, effect=0.1301, dominant_group=macro_system
- macro L1 N1120: bias=1.0175, effect=0.1482, dominant_group=macro_system
- macro L0 N566: bias=0.2155, effect=0.6085, dominant_group=macro_system
- macro L2 N3034: bias=1.0277, effect=0.1103, dominant_group=macro_system
- macro L0 N171: bias=0.3496, effect=0.2752, dominant_group=macro_system

## 理论提示
- 如果顶级神经元同时具有较高 effect（效应量）和较高 group support（组覆盖），说明该词类内部存在跨家族共享的较一般编码规则。
- 如果主要 band 与次要 band 的分裂明显，说明该词类内部至少包含两条不同尺度的编码链，而不是单一平面。
