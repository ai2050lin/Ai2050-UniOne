# stage524 完整语言功能层区地图

## 核心结论
语言功能相关有效神经元并不是全层均匀散开，而是呈现明显层带聚集。代词、连接词、时间词等功能性模式更容易出现在早中层路由带；名词、属性与固定搭配更容易在中晚层或晚层形成收束带；不同模型的层带坐标不同，但“功能层区”这件事本身是清楚存在的。

## qwen3
- 早层功能区：`无`
- 中层功能区：`noun, pronoun, verb`
- 晚层功能区：`attribute_channel, connective, fixed_phrase, locative, noun_attribute_bridge, pronoun, quantity, time`

- `noun`：峰值层 `[22, 20, 21]`，质心 `17.67`，层带 `middle`，来源 `stage423_wordclass_distribution`
- `verb`：峰值层 `[2, 4, 3]`，质心 `15.11`，层带 `middle`，来源 `stage423_wordclass_distribution`
- `pronoun`：峰值层 `[35, 34, 21]`，质心 `22.13`，层带 `middle`，来源 `stage423_wordclass_distribution`
- `connective`：峰值层 `[28, 27, 33, 27]`，质心 `28.75`，层带 `late`，来源 `stage493_chinese_pattern_atlas`
- `fixed_phrase`：峰值层 `[34, 32, 34]`，质心 `33.33`，层带 `late`，来源 `stage493_chinese_pattern_atlas`
- `time`：峰值层 `[34, 35, 34]`，质心 `34.33`，层带 `late`，来源 `stage493_chinese_pattern_atlas`
- `quantity`：峰值层 `[27, 28, 32]`，质心 `29.00`，层带 `late`，来源 `stage493_chinese_pattern_atlas`
- `locative`：峰值层 `[33, 32]`，质心 `32.50`，层带 `late`，来源 `stage493_chinese_pattern_atlas`
- `pronoun`：峰值层 `[33]`，质心 `33.00`，层带 `late`，来源 `stage493_chinese_pattern_atlas`
- `attribute_channel`：峰值层 `[34, 35, 33]`，质心 `31.91`，层带 `late`，来源 `stage519_noun_attribute_bridge_layer_atlas`
- `noun_attribute_bridge`：峰值层 `[34, 35, 33]`，质心 `31.93`，层带 `late`，来源 `stage519_noun_attribute_bridge_layer_atlas`

## deepseek7b
- 早层功能区：`pronoun`
- 中层功能区：`noun, verb`
- 晚层功能区：`attribute_channel, connective, fixed_phrase, locative, noun_attribute_bridge, pronoun, quantity, time`

- `noun`：峰值层 `[1, 20, 19]`，质心 `14.60`，层带 `middle`，来源 `stage423_wordclass_distribution`
- `verb`：峰值层 `[3, 2, 1]`，质心 `12.69`，层带 `middle`，来源 `stage423_wordclass_distribution`
- `pronoun`：峰值层 `[2, 1, 3]`，质心 `6.49`，层带 `early`，来源 `stage423_wordclass_distribution`
- `connective`：峰值层 `[19, 24, 25, 25]`，质心 `23.25`，层带 `late`，来源 `stage493_chinese_pattern_atlas`
- `fixed_phrase`：峰值层 `[26, 26, 25]`，质心 `25.67`，层带 `late`，来源 `stage493_chinese_pattern_atlas`
- `time`：峰值层 `[26, 26, 25]`，质心 `25.67`，层带 `late`，来源 `stage493_chinese_pattern_atlas`
- `quantity`：峰值层 `[22, 22, 25]`，质心 `23.00`，层带 `late`，来源 `stage493_chinese_pattern_atlas`
- `locative`：峰值层 `[27, 26]`，质心 `26.50`，层带 `late`，来源 `stage493_chinese_pattern_atlas`
- `pronoun`：峰值层 `[25]`，质心 `25.00`，层带 `late`，来源 `stage493_chinese_pattern_atlas`
- `attribute_channel`：峰值层 `[27, 26, 3]`，质心 `24.58`，层带 `late`，来源 `stage519_noun_attribute_bridge_layer_atlas`
- `noun_attribute_bridge`：峰值层 `[27, 26, 3]`，质心 `24.07`，层带 `late`，来源 `stage519_noun_attribute_bridge_layer_atlas`

## glm4
- 早层功能区：`无`
- 中层功能区：`无`
- 晚层功能区：`attribute_channel, noun, noun_attribute_bridge`

- `noun`：峰值层 `[39, 38, 37]`，质心 `37.81`，层带 `late`，来源 `stage519_noun_attribute_bridge_layer_atlas`
- `attribute_channel`：峰值层 `[39, 38, 37]`，质心 `37.83`，层带 `late`，来源 `stage519_noun_attribute_bridge_layer_atlas`
- `noun_attribute_bridge`：峰值层 `[39, 38, 37]`，质心 `37.82`，层带 `late`，来源 `stage519_noun_attribute_bridge_layer_atlas`

## gemma4
- 早层功能区：`attribute_channel, noun, noun_attribute_bridge`
- 中层功能区：`无`
- 晚层功能区：`无`

- `noun`：峰值层 `[4, 1, 0]`，质心 `8.15`，层带 `early`，来源 `stage519_noun_attribute_bridge_layer_atlas`
- `attribute_channel`：峰值层 `[4, 0, 1]`，质心 `6.24`，层带 `early`，来源 `stage519_noun_attribute_bridge_layer_atlas`
- `noun_attribute_bridge`：峰值层 `[4, 1, 0]`，质心 `6.70`，层带 `early`，来源 `stage519_noun_attribute_bridge_layer_atlas`

