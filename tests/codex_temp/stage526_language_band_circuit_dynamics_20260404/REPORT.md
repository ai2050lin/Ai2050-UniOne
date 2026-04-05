# stage526 层带位置图谱与最小因果回路统一动力学

## 核心结论
把层带图谱和最小因果回路压到一起之后，当前最稳的统一动力学是：早层主要负责路由和引用建立，中层主要负责桥接与绑定，晚层主要负责名词、属性、固定搭配等内容的收束读出。不同模型的具体层号不同，但“早路由、中绑定、晚读出”这个大框架已经越来越清楚。

## 更新方程
h_{t}^{l+1} = h_{t}^{l} + E_l(route_t) + M_l(bind_t) + L_l(readout_t)，其中早层 E_l 更偏路由与功能词控制，中层 M_l 更偏桥接与绑定，晚层 L_l 更偏名词/属性收束与最终读出。

## qwen3
- 早层主功能：`无`
- 中层主功能：`noun, pronoun, verb`
- 晚层主功能：`attribute_channel, connective, fixed_phrase, locative, noun_attribute_bridge, pronoun, quantity, time`
- 跨任务共享骨干子集：`33, 34`，对应层带 `late`
- `noun_relation`：子集层 `[3, 35]`，层带 `middle`，效用 `0.009684`
- `noun_syntax_role`：子集层 `[3, 33, 34]`，层带 `late`，效用 `0.047526`
- `noun_association_network`：子集层 `[35]`，层带 `late`，效用 `0.001414`

## deepseek7b
- 早层主功能：`pronoun`
- 中层主功能：`noun, verb`
- 晚层主功能：`attribute_channel, connective, fixed_phrase, locative, noun_attribute_bridge, pronoun, quantity, time`
- 跨任务共享骨干子集：`27, 27`，对应层带 `late`
- `noun_relation`：子集层 `[]`，层带 `unknown`，效用 `0.000000`
- `noun_syntax_role`：子集层 `[]`，层带 `unknown`，效用 `0.000000`
- `noun_association_network`：子集层 `[27]`，层带 `late`，效用 `0.003581`

## glm4
- 早层主功能：`无`
- 中层主功能：`无`
- 晚层主功能：`attribute_channel, noun, noun_attribute_bridge`
- 跨任务共享骨干子集：`39`，对应层带 `late`
- `noun_relation`：子集层 `[39]`，层带 `late`，效用 `0.001302`
- `noun_syntax_role`：子集层 `[37]`，层带 `late`，效用 `0.003906`
- `noun_association_network`：子集层 `[39]`，层带 `late`，效用 `0.015625`

## gemma4
- 早层主功能：`attribute_channel, noun, noun_attribute_bridge`
- 中层主功能：`无`
- 晚层主功能：`无`
- 跨任务共享骨干子集：`4, 27, 1`，对应层带 `early`
- `noun_relation`：子集层 `[1, 4]`，层带 `early`，效用 `0.147949`
- `noun_syntax_role`：子集层 `[1, 27]`，层带 `middle`，效用 `0.031687`
- `noun_association_network`：子集层 `[27, 1, 0]`，层带 `early`，效用 `0.047689`

