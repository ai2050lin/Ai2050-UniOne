# stage579 认识论副词与不确定性实测

## 核心结论
认识论副词只达到中等稳定度，说明 M_epistemic 既不是普通修饰项，也还没有被当前理论完整吸收到 Q_t。

## Qwen3-4B
- `epistemic_force` 准确率：`0.7500`
- `epistemic_force` 平均 top1-top2 间隔：`3.3125`
- `epistemic_force` 首个样本层间 margin 首层/末层：`2.9375` / `2.7500`
- `epistemic_entailment` 准确率：`1.0000`
- `epistemic_entailment` 平均 top1-top2 间隔：`1.5781`
- `epistemic_entailment` 首个样本层间 margin 首层/末层：`15.1641` / `4.3750`

## DeepSeek-R1-Distill-Qwen-7B
- `epistemic_force` 准确率：`0.5000`
- `epistemic_force` 平均 top1-top2 间隔：`4.2032`
- `epistemic_force` 首个样本层间 margin 首层/末层：`0.5254` / `3.3750`
- `epistemic_entailment` 准确率：`1.0000`
- `epistemic_entailment` 平均 top1-top2 间隔：`1.0416`
- `epistemic_entailment` 首个样本层间 margin 首层/末层：`8.5312` / `4.1250`

## GLM-4-9B-Chat-HF
- `epistemic_force` 准确率：`0.7500`
- `epistemic_force` 平均 top1-top2 间隔：`2.5312`
- `epistemic_force` 首个样本层间 margin 首层/末层：`-0.8789` / `0.0625`
- `epistemic_entailment` 准确率：`0.7500`
- `epistemic_entailment` 平均 top1-top2 间隔：`2.4453`
- `epistemic_entailment` 首个样本层间 margin 首层/末层：`-0.5469` / `6.5312`

## Gemma-4-E2B-it
- `epistemic_force` 准确率：`0.2500`
- `epistemic_force` 平均 top1-top2 间隔：`1.1875`
- `epistemic_force` 首个样本层间 margin 首层/末层：`-1.3438` / `4.0312`
- `epistemic_entailment` 准确率：`0.7500`
- `epistemic_entailment` 平均 top1-top2 间隔：`1.6250`
- `epistemic_entailment` 首个样本层间 margin 首层/末层：`-0.4688` / `-0.5625`

