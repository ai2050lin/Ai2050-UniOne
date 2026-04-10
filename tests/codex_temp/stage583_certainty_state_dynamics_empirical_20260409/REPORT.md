# stage583 确定性状态动力学实测

## 核心结论
Q_certainty 三类子任务接近，当前样本下它可以暂时视作较统一的判断状态。

## Qwen3-4B
- `certainty_commitment` 准确率：`0.5000`
- `certainty_commitment` 平均 top1-top2 间隔：`2.8281`
- `certainty_guarantee` 准确率：`0.7500`
- `certainty_guarantee` 平均 top1-top2 间隔：`2.4062`
- `certainty_opposite_compatibility` 准确率：`0.7500`
- `certainty_opposite_compatibility` 平均 top1-top2 间隔：`2.2813`

## DeepSeek-R1-Distill-Qwen-7B
- `certainty_commitment` 准确率：`0.5000`
- `certainty_commitment` 平均 top1-top2 间隔：`3.6526`
- `certainty_guarantee` 准确率：`0.7500`
- `certainty_guarantee` 平均 top1-top2 间隔：`1.3592`
- `certainty_opposite_compatibility` 准确率：`0.5000`
- `certainty_opposite_compatibility` 平均 top1-top2 间隔：`2.4688`

## GLM-4-9B-Chat-HF
- `certainty_commitment` 准确率：`0.5000`
- `certainty_commitment` 平均 top1-top2 间隔：`1.9531`
- `certainty_guarantee` 准确率：`0.5000`
- `certainty_guarantee` 平均 top1-top2 间隔：`2.7734`
- `certainty_opposite_compatibility` 准确率：`0.5000`
- `certainty_opposite_compatibility` 平均 top1-top2 间隔：`1.5938`

## Gemma-4-E2B-it
- `certainty_commitment` 准确率：`0.5000`
- `certainty_commitment` 平均 top1-top2 间隔：`0.3438`
- `certainty_guarantee` 准确率：`0.2500`
- `certainty_guarantee` 平均 top1-top2 间隔：`1.4062`
- `certainty_opposite_compatibility` 准确率：`0.7500`
- `certainty_opposite_compatibility` 平均 top1-top2 间隔：`0.7344`

