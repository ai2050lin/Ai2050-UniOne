# stage446_polysemy_neuron_overlap_and_switch_axis_ablation

## 核心回答
苹果的水果义与品牌义不是两套完全独立神经元，而是共享一部分名词骨干、再由一条可干预的切换轴把状态推向不同词义盆地。普通名词更像同一骨干内的上下文扰动，多义词则多出一条跨词义盆地的稳定切换结构。

## Qwen/Qwen3-4B
- best_switch_layer: 5
- fruit_brand_active_jaccard: 0.0199
- banana_context_mean_active_jaccard: 0.2897
- ordinary_vs_polysemy_gap: 0.2698
- switch_axis_prob_drop: 0.0674
- control_axis_prob_drop: 0.0219
- switch_axis_accuracy_drop: 0.2000
- control_axis_accuracy_drop: 0.1000

