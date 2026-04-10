# stage573 水果最小因果编码结构实测

## 总结
本实验用四模型串行检验水果骨干、概念偏置、属性通道与绑定残差四个可观测量。如果苹果在结构上稳定表现为“更靠近水果骨干 + 偏置较小 + 属性可迁移 + 绑定残差非零”，则最小因果编码结构得到第一轮实证支持。

## Qwen3-4B
- 水果骨干范数：`18.8123`
- apple 对水果/动物/物体骨干余弦：`0.1301` / `-0.0450` / `-0.1038`
- apple 概念偏置比例：`0.5626`
- `red` 属性共享余弦/绑定残差：`0.2964` / `0.1267`
- `sweet` 属性共享余弦/绑定残差：`0.4671` / `0.1131`
- apple 末层 fruit-animal 分离：`0.0250`
- `An apple is a type of` → ` fruit` (top1-top2=`9.7500`)
- `A banana is a type of` → ` fruit` (top1-top2=`10.2500`)
- `A pear is usually` → ` sweet` (top1-top2=`6.5312`)

## DeepSeek-R1-Distill-Qwen-7B
- 水果骨干范数：`28.1461`
- apple 对水果/动物/物体骨干余弦：`0.1707` / `-0.2814` / `0.0585`
- apple 概念偏置比例：`0.8406`
- `red` 属性共享余弦/绑定残差：`0.4697` / `0.2501`
- `sweet` 属性共享余弦/绑定残差：`0.3974` / `0.2537`
- apple 末层 fruit-animal 分离：`0.0285`
- `An apple is a type of` → ` fruit` (top1-top2=`7.7500`)
- `A banana is a type of` → ` fruit` (top1-top2=`9.3750`)
- `A pear is usually` → ` sweet` (top1-top2=`4.3750`)

## GLM-4-9B-Chat-HF
- 水果骨干范数：`38.2631`
- apple 对水果/动物/物体骨干余弦：`0.3862` / `0.1009` / `-0.4318`
- apple 概念偏置比例：`0.7159`
- `red` 属性共享余弦/绑定残差：`0.2943` / `0.2357`
- `sweet` 属性共享余弦/绑定残差：`0.4361` / `0.3292`
- apple 末层 fruit-animal 分离：`0.0598`
- `An apple is a type of` → ` fruit` (top1-top2=`12.1562`)
- `A banana is a type of` → ` fruit` (top1-top2=`13.9297`)
- `A pear is usually` → ` sweet` (top1-top2=`10.3125`)

## Gemma-4-E2B-it
- 水果骨干范数：`18.6038`
- apple 对水果/动物/物体骨干余弦：`0.2240` / `0.0682` / `-0.2797`
- apple 概念偏置比例：`0.8064`
- `red` 属性共享余弦/绑定残差：`0.5376` / `0.2741`
- `sweet` 属性共享余弦/绑定残差：`0.7082` / `0.2532`
- apple 末层 fruit-animal 分离：`0.0236`
- `An apple is a type of` → ` fruit` (top1-top2=`2.2500`)
- `A banana is a type of` → ` fruit` (top1-top2=`0.5000`)
- `A pear is usually` → ` loud` (top1-top2=`0.2500`)

