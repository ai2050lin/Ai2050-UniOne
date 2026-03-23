# Stage118 GPT-2 词表层级注册报告

- 时间: 2026-03-23 01:04:49
- 词表大小: 50257
- 干净概念词数量: 30541
- 已注册概念词数量: 468
- Micro（微观）数量: 46
- Meso（中观）数量: 357
- Macro（宏观）数量: 65
- Unknown（未确定）数量: 30073
- 家族覆盖率: 1.000000
- 注册总分: 0.707725

## 锚点词检查
- apple: label=meso, scores={'micro': 0.0, 'meso': 4.0, 'macro': 0.0}, reason=meso_seed:fruit, variants=4
- banana: label=meso, scores={'micro': 0.0, 'meso': 4.0, 'macro': 0.0}, reason=meso_seed:fruit, variants=2
- fruit: label=meso, scores={'micro': 0.0, 'meso': 2.0, 'macro': 0.0}, reason=meso_anchor, variants=3
- red: label=micro, scores={'micro': 3.0, 'meso': 0.0, 'macro': 1.0}, reason=micro_seed,verb_suffix, variants=6
- sweet: label=micro, scores={'micro': 3.0, 'meso': 0.0, 'macro': 0.0}, reason=micro_seed, variants=4
- justice: label=macro, scores={'micro': 0.0, 'meso': 0.0, 'macro': 6.0}, reason=macro_abstract_seed,macro_anchor, variants=5
- truth: label=macro, scores={'micro': 0.0, 'meso': 0.0, 'macro': 6.0}, reason=macro_abstract_seed,macro_anchor, variants=4
- run: label=macro, scores={'micro': 0.0, 'meso': 0.0, 'macro': 3.0}, reason=macro_verb_seed, variants=5
- jump: label=macro, scores={'micro': 0.0, 'meso': 0.0, 'macro': 3.0}, reason=macro_verb_seed, variants=4

## 当前结论
- 这一步解决的是“GPT-2 全词表中哪些词可以进入理论分析底座”。
- 它不是最终理论，而是为后续家族共享基底、实例偏置、关系偏置提供干净注册表。
- 当前最关键对象是可注册的 Meso（中观）实体词，因为水果族共享基底分析首先依赖它们。