# stage585 自然叙事跨句链路实测

## 核心结论
自然叙事口径下，P_discourse 两类子任务接近，至少没有出现更强的内部裂解证据。

## Qwen3-4B
- `narrative_recent_actor` 准确率：`0.2500`
- `narrative_recent_actor` 平均 top1-top2 间隔：`1.8956`
- `narrative_reactivated_actor` 准确率：`0.2500`
- `narrative_reactivated_actor` 平均 top1-top2 间隔：`1.6094`

## DeepSeek-R1-Distill-Qwen-7B
- `narrative_recent_actor` 准确率：`0.7500`
- `narrative_recent_actor` 平均 top1-top2 间隔：`1.0836`
- `narrative_reactivated_actor` 准确率：`0.2500`
- `narrative_reactivated_actor` 平均 top1-top2 间隔：`0.8318`

## GLM-4-9B-Chat-HF
- `narrative_recent_actor` 准确率：`0.7500`
- `narrative_recent_actor` 平均 top1-top2 间隔：`2.2314`
- `narrative_reactivated_actor` 准确率：`0.5000`
- `narrative_reactivated_actor` 平均 top1-top2 间隔：`0.4255`

## Gemma-4-E2B-it
- `narrative_recent_actor` 准确率：`1.0000`
- `narrative_recent_actor` 平均 top1-top2 间隔：`1.4062`
- `narrative_reactivated_actor` 准确率：`1.0000`
- `narrative_reactivated_actor` 平均 top1-top2 间隔：`1.8594`

