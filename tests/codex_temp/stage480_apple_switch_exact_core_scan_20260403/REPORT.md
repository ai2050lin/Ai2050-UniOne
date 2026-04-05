# stage480_apple_switch_exact_core_scan

## 实验设置
- 时间戳: 2026-04-03T00:20:34Z
- 是否使用 CUDA: False
- 批大小: 2
- 目标: 对苹果切换核心做精确子集穷举，区分骨架与增强器

## 模型 qwen3
- 候选数: 5
- utility 最优子集: H:5:0, H:5:2, H:5:29, H:5:8, H:5:9
- utility 最优值: +0.0747
- heldout_drop 最优子集: H:5:0, H:5:2, H:5:29, H:5:8, H:5:9
- heldout_drop 最优值: +0.0029
- 50% utility 最小子集: {'subset_ids': ['H:5:2', 'H:5:29'], 'subset_size': 2, 'subset_kind_counts': {'attention_head': 2, 'mlp_neuron': 0}, 'effect': {'search_drop': 0.11360677083333337, 'heldout_drop': 0.0009765625, 'control_shift': 0.00341796875, 'control_abs_shift': 0.00341796875, 'search_accuracy_before': 1.0, 'search_accuracy_after': 0.8333333333333334, 'heldout_accuracy_before': 1.0, 'heldout_accuracy_after': 1.0, 'control_accuracy_before': 0.75, 'control_accuracy_after': 0.75, 'utility': 0.055582682291666685}}
- 70% utility 最小子集: {'subset_ids': ['H:5:2', 'H:5:29'], 'subset_size': 2, 'subset_kind_counts': {'attention_head': 2, 'mlp_neuron': 0}, 'effect': {'search_drop': 0.11360677083333337, 'heldout_drop': 0.0009765625, 'control_shift': 0.00341796875, 'control_abs_shift': 0.00341796875, 'search_accuracy_before': 1.0, 'search_accuracy_after': 0.8333333333333334, 'heldout_accuracy_before': 1.0, 'heldout_accuracy_after': 1.0, 'control_accuracy_before': 0.75, 'control_accuracy_after': 0.75, 'utility': 0.055582682291666685}}
- 90% utility 最小子集: {'subset_ids': ['H:5:2', 'H:5:29', 'H:5:9'], 'subset_size': 3, 'subset_kind_counts': {'attention_head': 3, 'mlp_neuron': 0}, 'effect': {'search_drop': 0.13704427083333337, 'heldout_drop': 0.0, 'control_shift': 0.0, 'control_abs_shift': 0.0, 'search_accuracy_before': 1.0, 'search_accuracy_after': 0.8333333333333334, 'heldout_accuracy_before': 1.0, 'heldout_accuracy_after': 1.0, 'control_accuracy_before': 0.75, 'control_accuracy_after': 0.75, 'utility': 0.06852213541666669}}

## 模型 deepseek7b
- 候选数: 6
- utility 最优子集: H:2:10, H:2:2, H:2:22, H:2:26, H:2:5, N:2:16785
- utility 最优值: +0.1336
- heldout_drop 最优子集: H:2:10, H:2:2, H:2:22, H:2:26, H:2:5, N:2:16785
- heldout_drop 最优值: +0.1185
- 50% utility 最小子集: {'subset_ids': ['N:2:16785'], 'subset_size': 1, 'subset_kind_counts': {'attention_head': 0, 'mlp_neuron': 1}, 'effect': {'search_drop': 0.09114583333333337, 'heldout_drop': 0.0640869140625, 'control_shift': -0.001953125, 'control_abs_shift': 0.001953125, 'search_accuracy_before': 0.6666666666666666, 'search_accuracy_after': 0.5, 'heldout_accuracy_before': 0.75, 'heldout_accuracy_after': 0.75, 'control_accuracy_before': 1.0, 'control_accuracy_after': 1.0, 'utility': 0.07663981119791669}}
- 70% utility 最小子集: {'subset_ids': ['H:2:10', 'H:2:22', 'N:2:16785'], 'subset_size': 3, 'subset_kind_counts': {'attention_head': 2, 'mlp_neuron': 1}, 'effect': {'search_drop': 0.135498046875, 'heldout_drop': 0.09112548828125, 'control_shift': -0.0078125, 'control_abs_shift': 0.0078125, 'search_accuracy_before': 0.6666666666666666, 'search_accuracy_after': 0.5, 'heldout_accuracy_before': 0.75, 'heldout_accuracy_after': 0.5, 'control_accuracy_before': 1.0, 'control_accuracy_after': 1.0, 'utility': 0.109405517578125}}
- 90% utility 最小子集: {'subset_ids': ['H:2:10', 'H:2:2', 'H:2:22', 'H:2:26', 'H:2:5', 'N:2:16785'], 'subset_size': 6, 'subset_kind_counts': {'attention_head': 5, 'mlp_neuron': 1}, 'effect': {'search_drop': 0.15452067057291674, 'heldout_drop': 0.1185302734375, 'control_shift': -0.005859375, 'control_abs_shift': 0.005859375, 'search_accuracy_before': 0.6666666666666666, 'search_accuracy_after': 0.5, 'heldout_accuracy_before': 0.75, 'heldout_accuracy_after': 0.5, 'control_accuracy_before': 1.0, 'control_accuracy_after': 1.0, 'utility': 0.13359578450520837}}
