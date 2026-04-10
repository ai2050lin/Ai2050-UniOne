# stage574 统一语言状态更新实测

## 总结
本实验用四类最小实验检验统一状态更新方程是否具备实证支撑。如果代词、介词、副词和逻辑推理都能在同一套状态变量上被解释，且逻辑结论不是只在末层突然出现，那么语言理论就开始从概念编码走向计算理论。

## Qwen3-4B
- `pronoun_coreference` 准确率：`1.0000`
- `preposition_relation` 准确率：`1.0000`
- `adverb_scope` 准确率：`1.0000`
- `logic_reasoning` 准确率：`1.0000`
  - 逻辑 margin 首层/末层：`-12.8359` / `21.1875`
  - 逻辑 margin 首层/末层：`24.1562` / `12.7500`

## DeepSeek-R1-Distill-Qwen-7B
- `pronoun_coreference` 准确率：`0.5000`
- `preposition_relation` 准确率：`1.0000`
- `adverb_scope` 准确率：`0.5000`
- `logic_reasoning` 准确率：`1.0000`
  - 逻辑 margin 首层/末层：`-5.9375` / `27.4375`
  - 逻辑 margin 首层/末层：`4.4941` / `-3.1250`

## GLM-4-9B-Chat-HF
- `pronoun_coreference` 准确率：`1.0000`
- `preposition_relation` 准确率：`1.0000`
- `adverb_scope` 准确率：`0.5000`
- `logic_reasoning` 准确率：`1.0000`
  - 逻辑 margin 首层/末层：`-1.0410` / `17.9414`
  - 逻辑 margin 首层/末层：`5.5703` / `5.3125`

## Gemma-4-E2B-it
- `pronoun_coreference` 准确率：`0.5000`
- `preposition_relation` 准确率：`1.0000`
- `adverb_scope` 准确率：`0.5000`
- `logic_reasoning` 准确率：`0.6667`
  - 逻辑 margin 首层/末层：`-1.9961` / `-4.5000`
  - 逻辑 margin 首层/末层：`5.8125` / `12.0312`

