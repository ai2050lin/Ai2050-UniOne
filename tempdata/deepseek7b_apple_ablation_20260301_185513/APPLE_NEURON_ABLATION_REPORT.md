# DeepSeek-7B 苹果概念关键神经元消融报告

## 1) 实验配置
- 模型: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- 设备: `cpu`
- 发现集样本数: `120`
- 候选关键神经元(top-k): `60`
- 消融神经元数: `20`

## 2) 关键神经元层级结构
- 三段层分布: `{'early': 19, 'middle': 1, 'late': 0}`
- Top 神经元 (前10):
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

## 3) 消融因果效应 (目标概率质量均值)
| Group | Baseline | Target Ablation | Random Ablation | Target Δ | Random Δ |
|---|---:|---:|---:|---:|---:|
| apple_attribute | 0.002036 | 0.003471 | 0.002046 | +0.001435 | +0.000010 |
| apple_recall | 0.003501 | 0.003434 | 0.003468 | -0.000067 | -0.000033 |
| fruit_attribute | 0.016713 | 0.016661 | 0.016253 | -0.000052 | -0.000460 |
| nonfruit_attribute | 0.000060 | 0.000059 | 0.000059 | -0.000001 | -0.000001 |

## 4) 编码模块 (按相关性聚类)
- 模块数: `1`
- Module 0: size=20, mean|corr|=0.694, apple-control=+1.9042, layers={0: 1, 1: 1, 3: 16, 4: 1, 13: 1}

## 5) 结论
本轮未观察到强苹果特异消融效应，建议扩大发现集并增加任务约束。