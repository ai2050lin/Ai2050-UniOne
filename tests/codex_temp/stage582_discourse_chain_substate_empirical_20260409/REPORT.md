# stage582 跨句链路子状态实测

## 核心结论
长链回收没有更难，当前样本下 P_discourse 尚未表现出明显内部分裂。

## Qwen3-4B
- `discourse_recent_subject` 准确率：`0.5000`
- `discourse_recent_subject` 平均 top1-top2 间隔：`1.0313`
- `discourse_long_chain` 准确率：`1.0000`
- `discourse_long_chain` 平均 top1-top2 间隔：`4.7188`

## DeepSeek-R1-Distill-Qwen-7B
- `discourse_recent_subject` 准确率：`0.7500`
- `discourse_recent_subject` 平均 top1-top2 间隔：`1.7520`
- `discourse_long_chain` 准确率：`1.0000`
- `discourse_long_chain` 平均 top1-top2 间隔：`1.3851`

## GLM-4-9B-Chat-HF
- `discourse_recent_subject` 准确率：`0.7500`
- `discourse_recent_subject` 平均 top1-top2 间隔：`1.9050`
- `discourse_long_chain` 准确率：`1.0000`
- `discourse_long_chain` 平均 top1-top2 间隔：`3.5918`

## Gemma-4-E2B-it
- `discourse_recent_subject` 准确率：`0.0000`
- `discourse_recent_subject` 平均 top1-top2 间隔：`1.5088`
- `discourse_long_chain` 准确率：`0.5000`
- `discourse_long_chain` 平均 top1-top2 间隔：`3.3867`

