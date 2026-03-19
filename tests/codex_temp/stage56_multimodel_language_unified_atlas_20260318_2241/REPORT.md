# Stage56 三模型统一语言图谱块

- model_count: 3
- global_category_frontier: {'top': ['fruit', 'food', 'object', 'weather'], 'bottom': ['abstract', 'animal', 'food', 'food']}

## 共享规律
- shared_closure_categories: ['fruit', 'action', 'weather', 'object', 'nature', 'vehicle']
- relation_system_split: {'local_linear_count': 15, 'path_bundle_count': 222}
- control_laws: {'logic_prototype_to_synergy_corr': 0.4390413643209007, 'syntax_conflict_to_synergy_corr': 0.2381580965405652, 'logic_fragile_bridge_to_synergy_corr': -0.40184649563959923}

## 模型私有实现
- deepseek-ai/DeepSeek-R1-Distill-Qwen-7B: reading=闭包友好型实现, strict_positive_pair_ratio=0.0800, prototype_layer_band=middle, strict_positive_categories=['action', 'weather']
- zai-org/GLM-4-9B-Chat-HF: reading=类别异常放大型实现, strict_positive_pair_ratio=0.0870, prototype_layer_band=late, strict_positive_categories=['fruit']
- Qwen/Qwen3-4B: reading=闭包友好型实现, strict_positive_pair_ratio=0.4167, prototype_layer_band=late, strict_positive_categories=['abstract', 'action', 'food', 'fruit', 'nature', 'object', 'vehicle', 'weather']
