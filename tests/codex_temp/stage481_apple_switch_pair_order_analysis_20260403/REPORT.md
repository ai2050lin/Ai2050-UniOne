# stage481_apple_switch_pair_order_analysis

## 实验设置
- 时间戳: 2026-04-03T00:41:23Z
- 输入来源: stage480 的精确子集穷举结果
- 目标: 从已有全子集数据中抽取骨架、增强器、配对协同和最佳加入顺序

## 模型 qwen3
- 角色划分: {'skeleton': ['H:5:2', 'H:5:29'], 'bridge_to_90pct_utility': ['H:5:9'], 'heldout_boosters': ['H:5:0', 'H:5:8'], 'max_utility_boosters': ['H:5:0', 'H:5:8']}
- utility 焦点集合: ['H:5:2', 'H:5:29', 'H:5:9']
- utility 最优顺序: ['H:5:2', 'H:5:29', 'H:5:9']
- heldout 焦点集合: ['H:5:0', 'H:5:2', 'H:5:29', 'H:5:8']
- heldout 最优顺序: ['H:5:0', 'H:5:2', 'H:5:29', 'H:5:8']

## 模型 deepseek7b
- 角色划分: {'anchor': ['N:2:16785'], 'main_boosters': ['H:2:10', 'H:2:22'], 'heldout_boosters': ['H:2:2', 'H:2:26'], 'max_utility_boosters': ['H:2:2', 'H:2:26', 'H:2:5']}
- utility 焦点集合: ['H:2:10', 'H:2:22', 'N:2:16785']
- utility 最优顺序: ['N:2:16785', 'H:2:22', 'H:2:10']
- heldout 焦点集合: ['H:2:10', 'H:2:2', 'H:2:26', 'N:2:16785']
- heldout 最优顺序: ['N:2:16785', 'H:2:26', 'H:2:10', 'H:2:2']
