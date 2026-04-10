# stage586 认识论副词与确定性耦合实测

## 核心结论
显式范围识别与确定性后果判断接近，当前样本下耦合优势还不明显。

## Qwen3-4B
- `scope_reading` 准确率：`0.5000`
- `scope_reading` 平均 top1-top2 间隔：`5.7188`
- `certainty_consequence` 准确率：`0.5000`
- `certainty_consequence` 平均 top1-top2 间隔：`1.4609`
- `counterclaim_compatibility` 准确率：`0.5000`
- `counterclaim_compatibility` 平均 top1-top2 间隔：`3.2031`

## DeepSeek-R1-Distill-Qwen-7B
- `scope_reading` 准确率：`0.7500`
- `scope_reading` 平均 top1-top2 间隔：`4.4002`
- `certainty_consequence` 准确率：`0.5000`
- `certainty_consequence` 平均 top1-top2 间隔：`0.9395`
- `counterclaim_compatibility` 准确率：`0.5000`
- `counterclaim_compatibility` 平均 top1-top2 间隔：`2.0817`

## GLM-4-9B-Chat-HF
- `scope_reading` 准确率：`0.5000`
- `scope_reading` 平均 top1-top2 间隔：`2.4688`
- `certainty_consequence` 准确率：`0.2500`
- `certainty_consequence` 平均 top1-top2 间隔：`0.5547`
- `counterclaim_compatibility` 准确率：`0.5000`
- `counterclaim_compatibility` 平均 top1-top2 间隔：`1.4062`

## Gemma-4-E2B-it
- `scope_reading` 准确率：`0.2500`
- `scope_reading` 平均 top1-top2 间隔：`2.8632`
- `certainty_consequence` 准确率：`0.2500`
- `certainty_consequence` 平均 top1-top2 间隔：`3.4688`
- `counterclaim_compatibility` 准确率：`0.7500`
- `counterclaim_compatibility` 平均 top1-top2 间隔：`1.0430`

