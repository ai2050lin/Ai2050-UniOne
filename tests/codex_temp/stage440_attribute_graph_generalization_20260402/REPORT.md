# stage440_attribute_graph_generalization

## 核心回答
如果多数水果-属性组合都能用 noun backbone（名词骨干）与 attribute modifier（属性修饰）并集解释大部分绑定神经元，同时还保留一块稳定但不占主导的 bridge-only 区域，那么“骨干 + 修饰 + 桥接”就不再只是苹果个案，而是更一般的属性绑定定律。

## Qwen/Qwen3-4B
- law_support_rate: 1.0000
- mean_union_coverage: 0.7177
- mean_bridge_only_ratio: 0.2823
- mean_fruit_fruit_overlap: 0.1869
- mean_fruit_attribute_overlap: 0.1194

- color: union=0.7135, bridge=0.2865
- taste: union=0.7227, bridge=0.2773
- size: union=0.7168, bridge=0.2832

## deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- law_support_rate: 1.0000
- mean_union_coverage: 0.7680
- mean_bridge_only_ratio: 0.2320
- mean_fruit_fruit_overlap: 0.3285
- mean_fruit_attribute_overlap: 0.2755

- color: union=0.7949, bridge=0.2051
- taste: union=0.7585, bridge=0.2415
- size: union=0.7507, bridge=0.2493

