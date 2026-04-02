# stage434_apple_polysemy_factorized_switch

## 核心回答
苹果的水果义与品牌义更像建立在同一共享名词底座上的两种稳定基底切换。模型先沿着相对独立的 sense switch axis（词义切换轴）区分水果与品牌，再在每个词义内部用少数低秩 modifier directions（修饰方向）完成颜色、味道、产品、价格等组合扩展。

## Qwen/Qwen3-4B
- 最佳层: L5
- factorized polysemy score: 0.6928
- fruit base reuse: 0.9346
- brand base reuse: 1.0000
- fruit modifier rank2 ratio: 0.0000
- brand modifier rank2 ratio: 0.0000
- switch axis orthogonality: 0.9998
- switch cost ratio: 0.9702

## deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- 最佳层: L2
- factorized polysemy score: 0.6698
- fruit base reuse: 0.8579
- brand base reuse: 1.0000
- fruit modifier rank2 ratio: 0.0000
- brand modifier rank2 ratio: 0.0000
- switch axis orthogonality: 0.8962
- switch cost ratio: 1.2164

## 理论解释
- 水果义和品牌义不是分别重建整套独立表示，而是先走一条较稳定的词义切换轴。
- 每个词义内部的具体变化更像低秩修饰，因此可以用少数方向覆盖许多组合。
- 这解释了为什么二义性不会导致组合爆炸，因为系统复用的是基底和修饰方向，不是为每个组合单独分配一块参数。 
