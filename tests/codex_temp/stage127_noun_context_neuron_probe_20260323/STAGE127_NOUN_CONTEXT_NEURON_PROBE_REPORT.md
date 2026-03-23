# stage127_noun_context_neuron_probe: Noun 神经元基础探针

## 核心结果
- 目标词类: noun
- 目标数量: 27702
- 控制组数量: 2839
- 层数: 12
- 每层神经元数: 3072
- 目标语义组数量: 16
- 主导通用层: L1
- 主导通用层分数: 0.5427
- 主要 band（尺度带）: meso
- 次要 band（尺度带）: macro
- 词类神经元基础探针分数: 0.9740

## 顶级通用神经元
- L0 N1344: rule=0.6966, effect=1.3085, support=1.0000, dominant_group=meso_fruit
- L9 N2722: rule=0.6792, effect=1.1188, support=0.9375, dominant_group=meso_animal
- L1 N2693: rule=0.6642, effect=1.2187, support=1.0000, dominant_group=meso_food
- L2 N2380: rule=0.6186, effect=1.1925, support=1.0000, dominant_group=micro_material
- L1 N2519: rule=0.6128, effect=1.1492, support=1.0000, dominant_group=meso_vehicle
- L7 N2211: rule=0.6128, effect=0.8598, support=0.9375, dominant_group=macro_system
- L2 N385: rule=0.6051, effect=0.9785, support=1.0000, dominant_group=meso_fruit
- L11 N1383: rule=0.6048, effect=0.9138, support=0.9375, dominant_group=micro_material
- L10 N2816: rule=0.6038, effect=0.8361, support=0.9375, dominant_group=meso_fruit
- L5 N359: rule=0.5993, effect=0.9604, support=0.9375, dominant_group=meso_vehicle
- L1 N541: rule=0.5966, effect=0.9848, support=1.0000, dominant_group=meso_food
- L1 N1404: rule=0.5747, effect=1.0637, support=1.0000, dominant_group=micro_material

## meso / macro 分裂神经元
- meso L0 N1981: bias=0.3601, effect=0.9892, dominant_group=meso_fruit
- meso L9 N2722: bias=0.2195, effect=1.1188, dominant_group=meso_animal
- meso L0 N1368: bias=0.2997, effect=0.8109, dominant_group=meso_fruit
- meso L1 N1490: bias=0.2644, effect=0.7578, dominant_group=meso_fruit
- meso L11 N1383: bias=0.2052, effect=0.9138, dominant_group=micro_material
- meso L10 N428: bias=0.2343, effect=0.6779, dominant_group=meso_celestial
- macro L11 N1202: bias=0.1746, effect=0.5135, dominant_group=macro_abstract
- macro L10 N73: bias=0.2969, effect=0.2766, dominant_group=macro_abstract
- macro L11 N1660: bias=0.1458, effect=0.4941, dominant_group=macro_system
- macro L11 N1049: bias=0.1646, effect=0.4325, dominant_group=macro_system
- macro L10 N536: bias=0.1525, effect=0.4059, dominant_group=macro_system
- macro L11 N2207: bias=0.0909, effect=0.6709, dominant_group=macro_system

## 理论提示
- 如果顶级神经元同时具有较高 effect（效应量）和较高 group support（组覆盖），说明该词类内部存在跨家族共享的较一般编码规则。
- 如果主要 band 与次要 band 的分裂明显，说明该词类内部至少包含两条不同尺度的编码链，而不是单一平面。
