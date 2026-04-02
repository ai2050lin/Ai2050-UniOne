# stage435_apple_feature_binding_neuron_channels

## 核心回答
苹果与香蕉等水果更像共享一条 fruit backbone channel（水果骨干通道），而 red（红色）、sweet（甜）、sour（酸）和 fist（拳头大小锚点）更像独立的 attribute modifier channels（属性修饰通道）。当模型处理 apple-red 或 apple-sweet 这类组合时，主要是把名词骨干和属性修饰在同一残差工作区里叠加，再用一小部分 bridge neurons（桥接神经元）完成绑定。

## Qwen/Qwen3-4B
- apple vs banana backbone overlap: 0.2427
- apple vs color overlap: 0.1228
- apple vs taste overlap: 0.1179
- apple-color union coverage: 0.7422
- apple-taste union coverage: 0.7305
- apple-size union coverage: 0.7266
- apple-color bridge only ratio: 0.2578
- apple-taste bridge only ratio: 0.2695
- apple-size bridge only ratio: 0.2734

## deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- apple vs banana backbone overlap: 0.3333
- apple vs color overlap: 0.3438
- apple vs taste overlap: 0.2962
- apple-color union coverage: 0.8047
- apple-taste union coverage: 0.7773
- apple-size union coverage: 0.7773
- apple-color bridge only ratio: 0.1953
- apple-taste bridge only ratio: 0.2227
- apple-size bridge only ratio: 0.2227

## 理论解释
- 苹果和香蕉的神经元编码相交更多，说明水果名词共享骨干通道。
- 红色、酸甜等属性词与水果名词的重叠较低，说明它们不是同一类通道。
- 组合句中的苹果表示主要落在“水果骨干 + 属性通道”的并集里，外加少量桥接神经元，这支持神经元级别的绑定而不是整块重写。
