# DeepSeek-7B 苹果概念关键神经元消融报告

## 1) 实验配置
- 模型: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- 设备: `cpu`
- 发现集样本数: `120`
- 候选关键神经元(top-k): `80`
- 消融神经元数: `40`

## 2) 关键神经元层级结构
- 三段层分布: `{'early': 39, 'middle': 1, 'late': 0}`
- 选择性Top神经元 (前10):
  - L3 N18787 | z=1.748 | Δapple-control=2.8845
  - L3 N11143 | z=1.732 | Δapple-control=1.5266
  - L3 N16049 | z=1.720 | Δapple-control=3.3688
  - L1 N4835 | z=1.648 | Δapple-control=1.1198
  - L3 N8092 | z=1.646 | Δapple-control=2.3733
  - L3 N13331 | z=1.584 | Δapple-control=1.5116
  - L3 N6997 | z=1.580 | Δapple-control=1.6174
  - L3 N6047 | z=1.576 | Δapple-control=2.1511
  - L3 N18805 | z=1.563 | Δapple-control=2.8175
  - L0 N18075 | z=1.551 | Δapple-control=0.4847
- 因果筛选后用于组合消融的神经元 (前10):
  - L3 N15701 | single Δ(apple_recall)=-0.000233
  - L3 N11752 | single Δ(apple_recall)=-0.000140
  - L3 N9813 | single Δ(apple_recall)=-0.000113
  - L3 N14827 | single Δ(apple_recall)=-0.000089
  - L3 N11873 | single Δ(apple_recall)=-0.000086
  - L3 N7847 | single Δ(apple_recall)=-0.000069
  - L3 N13874 | single Δ(apple_recall)=-0.000063
  - L3 N6997 | single Δ(apple_recall)=-0.000062
  - L3 N2829 | single Δ(apple_recall)=-0.000061
  - L13 N18207 | single Δ(apple_recall)=-0.000060

## 3) 消融因果效应 (目标概率质量均值)
| Group | Baseline | Target Ablation | Random Ablation | Target Δ | Random Δ |
|---|---:|---:|---:|---:|---:|
| apple_attribute | 0.002036 | 0.003498 | 0.002003 | +0.001462 | -0.000033 |
| apple_recall | 0.003501 | 0.003245 | 0.003745 | -0.000256 | +0.000244 |
| fruit_attribute | 0.016713 | 0.016637 | 0.017066 | -0.000076 | +0.000353 |
| nonfruit_attribute | 0.000060 | 0.000059 | 0.000060 | -0.000001 | -0.000000 |

## 4) 编码模块 (按相关性聚类)
- 模块数: `1`
- Module 0: size=40, mean|corr|=0.627, apple-control=+1.6015, layers={0: 3, 2: 1, 3: 33, 4: 2, 13: 1}

## 5) 结论
本轮未观察到强苹果特异消融效应，建议扩大发现集并增加任务约束。