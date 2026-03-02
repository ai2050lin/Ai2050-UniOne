# DeepSeek-7B 多水果神经元差异与水果概念编码报告

## 1) 关键结论
本轮未得到强共性因果证据，但可见稳定的水果共享簇与水果特异子簇分化，建议扩大评估任务后复核因果强度。

## 2) 水果共性编码结构
- 共性神经元数: 50
- 层分布: {0: 4, 1: 2, 3: 9, 4: 19, 5: 9, 6: 5, 8: 2}
- 三段分布: {'early': 50, 'middle': 0, 'late': 0}
- 模块数: 1
- 主模块: size=50, mean|corr|=0.616, fruit-minus-nonfruit=+1.5862

## 3) 因果消融 (共性神经元)
| Group | Baseline | Fruit-General Ablation | Random Ablation | FG Δ | Random Δ |
|---|---:|---:|---:|---:|---:|
| fruit_recall | 0.000000 | 0.000000 | 0.000000 | +0.000000 | -0.000000 |
| fruit_general | 0.000256 | 0.000270 | 0.000255 | +0.000014 | -0.000001 |
| nonfruit_control | 0.000051 | 0.000051 | 0.000051 | +0.000000 | +0.000000 |
| banana_specific | 0.000009 | 0.000009 | 0.000009 | +0.000000 | +0.000000 |
| orange_specific | 0.000054 | 0.000061 | 0.000055 | +0.000006 | +0.000000 |
| apple_specific | 0.000092 | 0.000096 | 0.000092 | +0.000004 | -0.000001 |

## 4) 各水果特异神经元
- apple:
  - top5: L3N13834(1.86), L0N3167(1.84), L0N6511(1.78), L3N10117(1.76), L3N18787(1.75)
  - layer_distribution: {0: 9, 1: 3, 3: 25, 4: 2, 24: 1}
  - band_distribution: {'early': 39, 'middle': 0, 'late': 1}
  - targeted_ablation_delta(apple_specific): +0.000047
  - recall_ablation_delta(apple_recall): -0.000059
- banana:
  - top5: L0N17808(1.70), L0N7250(1.68), L0N9767(1.63), L3N11960(1.61), L0N15284(1.60)
  - layer_distribution: {0: 19, 1: 2, 3: 12, 4: 5, 7: 1, 13: 1}
  - band_distribution: {'early': 39, 'middle': 1, 'late': 0}
  - targeted_ablation_delta(banana_specific): -0.000000
  - recall_ablation_delta(banana_recall): -0.000000
- grape:
  - top5: L0N15727(1.87), L0N16560(1.73), L0N4714(1.65), L5N10366(1.65), L0N17870(1.64)
  - layer_distribution: {0: 28, 1: 4, 3: 5, 4: 2, 5: 1}
  - band_distribution: {'early': 40, 'middle': 0, 'late': 0}
  - targeted_ablation_delta(grape_specific): +0.000004
  - recall_ablation_delta(grape_recall): +0.000000
- orange:
  - top5: L3N7491(3.19), L0N3600(2.93), L13N8936(2.78), L0N12643(2.74), L0N17566(2.74)
  - layer_distribution: {0: 21, 1: 4, 3: 10, 4: 3, 13: 1, 27: 1}
  - band_distribution: {'early': 38, 'middle': 1, 'late': 1}
  - targeted_ablation_delta(orange_specific): +0.000006
  - recall_ablation_delta(orange_recall): -0.019074
- peach:
  - top5: L0N17900(6.68), L0N4933(5.27), L0N10747(4.40), L0N17821(4.38), L0N15675(4.07)
  - layer_distribution: {0: 36, 2: 3, 8: 1}
  - band_distribution: {'early': 40, 'middle': 0, 'late': 0}
  - targeted_ablation_delta(peach_specific): -0.000005
  - recall_ablation_delta(peach_recall): +0.000000
- pear:
  - top5: L1N11985(1.91), L1N18591(1.80), L1N13406(1.79), L3N14496(1.77), L1N17007(1.76)
  - layer_distribution: {0: 4, 1: 16, 2: 5, 3: 13, 4: 1, 5: 1}
  - band_distribution: {'early': 40, 'middle': 0, 'late': 0}
  - targeted_ablation_delta(pear_specific): -0.000000
  - recall_ablation_delta(pear_recall): -0.000005

## 5) 水果间编码重叠 (Jaccard, top-30)
| fruit | apple | banana | grape | orange | peach | pear |
|---|---:|---:|---:|---:|---:|---:|
| apple | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.017 |
| banana | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| grape | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| orange | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| peach | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| pear | 0.017 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |