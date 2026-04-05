# stage487_polysemy_unified_switch_protocol

## 跨模型统一结论
- shared_large_gap_nouns = amazon, apple, python
- shared_large_gap_support_rate = 0.7500
- mean_gap_qwen3 = 0.2067
- mean_gap_deepseek7b = 0.1644
- core_answer = 当前最稳的统一结论是：多义词不是普通上下文扰动的放大版，而更像共享底座之上的低重合切换。四个多义词家族在两个模型里都出现了稳定的“普通上下文高重合、多义切换低重合”结构；苹果则额外提供了最深的机制证据，显示切换可以进一步压成头骨架或神经元锚点。

## 模型 qwen3
- mean_polysemy_active_jaccard = 0.0892
- mean_ordinary_active_jaccard = 0.2959
- mean_gap = 0.2067
- mean_shared_core_similarity = 0.6490
- large_gap_nouns = apple, amazon, python
- apple: switch=L34, balance=L34, poly_jaccard=0.0644, ordinary_jaccard=0.2897, gap=0.2252, shared_core=0.6228
- amazon: switch=L34, balance=L34, poly_jaccard=0.0802, ordinary_jaccard=0.3586, gap=0.2784, shared_core=0.6137
- python: switch=L34, balance=L34, poly_jaccard=0.0894, ordinary_jaccard=0.3219, gap=0.2325, shared_core=0.6458
- java: switch=L34, balance=L34, poly_jaccard=0.1228, ordinary_jaccard=0.2134, gap=0.0906, shared_core=0.7138
- apple_best_sensitive_layer = L5
- apple_mixed_circuit_final_subset_ids = H:5:2, H:5:29, H:5:9
- apple_exact_core_shapley_order = H:5:2, H:5:29, H:5:8, H:5:9, H:5:0
- apple_best_order = H:5:2 -> H:5:29 -> H:5:9

## 模型 deepseek7b
- mean_polysemy_active_jaccard = 0.1202
- mean_ordinary_active_jaccard = 0.2846
- mean_gap = 0.1644
- mean_shared_core_similarity = 0.8141
- large_gap_nouns = apple, amazon, python, java
- apple: switch=L26, balance=L26, poly_jaccard=0.0894, ordinary_jaccard=0.2800, gap=0.1906, shared_core=0.8165
- amazon: switch=L26, balance=L26, poly_jaccard=0.1034, ordinary_jaccard=0.2788, gap=0.1753, shared_core=0.8029
- python: switch=L27, balance=L27, poly_jaccard=0.1082, ordinary_jaccard=0.2856, gap=0.1774, shared_core=0.7963
- java: switch=L26, balance=L26, poly_jaccard=0.1797, ordinary_jaccard=0.2939, gap=0.1142, shared_core=0.8407
- apple_best_sensitive_layer = L2
- apple_mixed_circuit_final_subset_ids = N:2:16785, H:2:2, H:2:22, H:2:10, H:2:26, H:2:5
- apple_exact_core_shapley_order = N:2:16785, H:2:22, H:2:26, H:2:10, H:2:5, H:2:2
- apple_best_order = N:2:16785 -> H:2:22 -> H:2:10

