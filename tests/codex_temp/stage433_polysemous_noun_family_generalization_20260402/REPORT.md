# stage433_polysemous_noun_family_generalization

## 核心回答
多义名词更像共享名词底座上的低成本定向切换系统。模型主要复用公共名词结构，再借助类别偏置和结构化上下文增量完成语义分叉，因此能同时保持较强表达力和较高性能。

## Qwen/Qwen3-4B
- 多义名词数量: 4
- shared base support count: 4
- structured switch support count: 4
- reliable readout support count: 3
- average compact score: 0.8755
- average expressive score: 0.7151
- average balance score: 0.7158

### apple
- 最佳平衡层: L34
- fruit explained ratio: 0.8560
- brand explained ratio: 0.9988
- delta structured ratio: 1.0000
- sense readout accuracy: 0.7500
- sense delta cost ratio: 7.4766
- balance score: 0.7323

### amazon
- 最佳平衡层: L34
- river explained ratio: 0.7151
- company explained ratio: 0.9985
- delta structured ratio: 1.0000
- sense readout accuracy: 0.7500
- sense delta cost ratio: 7.3498
- balance score: 0.6835

### python
- 最佳平衡层: L34
- animal explained ratio: 0.8727
- programming_language explained ratio: 0.9989
- delta structured ratio: 1.0000
- sense readout accuracy: 0.7500
- sense delta cost ratio: 6.9638
- balance score: 0.7358

### java
- 最佳平衡层: L34
- coffee explained ratio: 0.8475
- programming_language explained ratio: 0.9889
- delta structured ratio: 1.0000
- sense readout accuracy: 0.6250
- sense delta cost ratio: 3.5812
- balance score: 0.7114

## deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- 多义名词数量: 4
- shared base support count: 4
- structured switch support count: 4
- reliable readout support count: 3
- average compact score: 0.8909
- average expressive score: 0.6949
- average balance score: 0.7230

### apple
- 最佳平衡层: L26
- fruit explained ratio: 0.8680
- brand explained ratio: 0.9914
- delta structured ratio: 1.0000
- sense readout accuracy: 0.7500
- sense delta cost ratio: 3.0297
- balance score: 0.7354

### amazon
- 最佳平衡层: L26
- river explained ratio: 0.7513
- company explained ratio: 0.9928
- delta structured ratio: 1.0000
- sense readout accuracy: 0.7500
- sense delta cost ratio: 2.8918
- balance score: 0.7071

### python
- 最佳平衡层: L27
- animal explained ratio: 0.8045
- programming_language explained ratio: 0.8280
- delta structured ratio: 1.0000
- sense readout accuracy: 0.7500
- sense delta cost ratio: 0.6226
- balance score: 0.7433

### java
- 最佳平衡层: L26
- coffee explained ratio: 0.8295
- programming_language explained ratio: 0.9751
- delta structured ratio: 1.0000
- sense readout accuracy: 0.6250
- sense delta cost ratio: 1.8686
- balance score: 0.7062

## 理论解释
- 如果多个多义名词都表现出高共享底座比例，说明模型没有为每个词义独立重建一整块表示。
- 如果语义切换主要体现为结构化增量，说明性能优势来自复用而不是粗暴压缩。
- 如果读出在两种语境下仍然可靠，说明共享结构并没有牺牲表达能力。
