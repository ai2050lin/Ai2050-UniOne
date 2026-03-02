# DeepSeek-7B 动物概念编码与动物-水果差异分析报告

## 1) 总结
当前任务集下因果特异性仍偏弱，但结构上已出现清晰分化：动物与水果共享簇低重叠，且猫/狗等实例具备独立特异子簇。建议下一步扩大行为任务对齐因果信号。

## 2) 动物共性编码结构
- 动物共性神经元数: 50
- 层分布: {0: 2, 1: 1, 3: 2, 4: 1, 7: 3, 8: 3, 9: 1, 11: 1, 19: 1, 21: 4, 22: 6, 23: 10, 24: 4, 25: 7, 26: 4}
- 三段分布: {'early': 12, 'middle': 2, 'late': 36}
- 模块数: 1
- 主模块: size=50, mean|corr|=0.630

## 3) 动物 vs 水果结构差异
- 动物-水果共性重叠(Jaccard, top-50): 0.0000
- 概念质心相似度(cosine): 0.9974
- 概念质心距离(L2): 175.4033

## 4) 交叉因果消融
| Group | Baseline | Animal Ablation | Fruit Ablation | Random Ablation | ΔAnimal | ΔFruit | ΔRandom |
|---|---:|---:|---:|---:|---:|---:|---:|
| animal_recall | 0.000005 | 0.000006 | 0.000005 | 0.000005 | +0.000001 | -0.000000 | +0.000000 |
| fruit_recall | 0.000001 | 0.000001 | 0.000001 | 0.000001 | +0.000000 | +0.000000 | -0.000000 |
| animal_attribute | 0.000030 | 0.000030 | 0.000030 | 0.000030 | +0.000000 | +0.000001 | +0.000000 |
| fruit_attribute | 0.000117 | 0.000117 | 0.000121 | 0.000119 | +0.000000 | +0.000004 | +0.000002 |
| cat_recall | 0.000000 | 0.000001 | 0.000001 | 0.000001 | +0.000000 | +0.000000 | +0.000000 |
| dog_recall | 0.000004 | 0.000005 | 0.000004 | 0.000004 | +0.000001 | -0.000000 | +0.000000 |
| nonconcept_control | 0.000051 | 0.000051 | 0.000050 | 0.000051 | -0.000000 | -0.000000 | +0.000000 |

## 5) 猫狗等特异编码
- animal_bird:
  - top5: L24N7531(5.16), L0N16928(4.94), L0N12132(4.47), L23N10856(4.14), L26N8346(4.14)
  - layer_distribution: {0: 10, 1: 1, 2: 5, 3: 11, 4: 3, 9: 1, 23: 4, 24: 1, 25: 1, 26: 3}
  - band_distribution: {'early': 30, 'middle': 1, 'late': 9}
  - recall_delta(animal_recall): +0.000000
- animal_cat:
  - top5: L2N5452(4.92), L3N16210(4.34), L0N15408(4.27), L1N11699(4.26), L0N12392(4.17)
  - layer_distribution: {0: 2, 1: 2, 2: 7, 3: 22, 4: 4, 6: 1, 9: 1, 23: 1}
  - band_distribution: {'early': 38, 'middle': 1, 'late': 1}
  - recall_delta(cat_recall): -0.000000
- animal_dog:
  - top5: L0N4913(5.33), L0N17072(4.89), L0N4803(4.55), L0N13320(4.18), L0N4603(4.12)
  - layer_distribution: {0: 14, 1: 1, 2: 2, 3: 15, 4: 4, 6: 1, 7: 1, 13: 1, 23: 1}
  - band_distribution: {'early': 38, 'middle': 1, 'late': 1}
  - recall_delta(dog_recall): +0.000000
- animal_fish:
  - top5: L3N2854(3.04), L4N13370(2.96), L3N7180(2.89), L3N5216(2.89), L3N337(2.83)
  - layer_distribution: {0: 1, 1: 5, 2: 1, 3: 27, 4: 6}
  - band_distribution: {'early': 40, 'middle': 0, 'late': 0}
  - recall_delta(animal_recall): +0.000000
- animal_horse:
  - top5: L24N13573(4.25), L0N7576(4.10), L9N6067(3.71), L9N11402(3.59), L0N11085(3.47)
  - layer_distribution: {0: 14, 3: 5, 4: 2, 9: 6, 23: 4, 24: 2, 25: 4, 26: 3}
  - band_distribution: {'early': 21, 'middle': 6, 'late': 13}
  - recall_delta(animal_recall): +0.000000
- animal_tiger:
  - top5: L1N18421(3.11), L1N2601(3.10), L1N16954(3.03), L1N4157(2.98), L4N5544(2.92)
  - layer_distribution: {0: 2, 1: 17, 3: 9, 4: 12}
  - band_distribution: {'early': 40, 'middle': 0, 'late': 0}
  - recall_delta(animal_recall): +0.000000