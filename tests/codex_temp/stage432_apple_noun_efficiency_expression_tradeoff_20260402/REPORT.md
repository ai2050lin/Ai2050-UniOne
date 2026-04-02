# stage432_apple_noun_efficiency_expression_tradeoff

## 核心回答
苹果之所以既能有足够表达力又能保持高性能，更像是因为它不是被完全单独编码，而是复用了共享名词底座与类别偏置，再通过较小但结构化的上下文增量完成语义切换。

## Qwen/Qwen3-4B
- 最佳平衡层: L34
- 共享压缩分数: 0.8810
- 表达能力分数: 0.7464
- 综合平衡分数: 0.7323
- fruit explained ratio: 0.8560
- brand explained ratio: 0.9988
- delta structured ratio: 1.0000
- sense readout accuracy: 0.7500
- sense delta cost ratio: 7.4766

## deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- 最佳平衡层: L26
- 共享压缩分数: 0.9212
- 表达能力分数: 0.7132
- 综合平衡分数: 0.7355
- fruit explained ratio: 0.8680
- brand explained ratio: 0.9918
- delta structured ratio: 1.0000
- sense readout accuracy: 0.7500
- sense delta cost ratio: 3.0297

## 理论解释
- 共享名词底座提供复用，所以不用给每个具体名词都配一整套独立参数。
- 水果偏置让 apple 可以快速落到同类语义区，不必从零构造整套意义。
- 上下文切换主要表现为结构化增量，而不是整向量重写，所以同一编码能服务多种表达场景。
- 当读出层能够稳定区分 fruit/brand 两种语境，而底座仍然高度共享时，就同时解释了表达力和效率。
