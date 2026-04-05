# stage510_gemma4_polysemy_prompt_calibration

## 核心结论
Gemma4 在当前多义词判别里最适合的提示格式是 `digit_marked`，因此后续需要先按模型偏好的回答接口重标定，再讨论切换机制本身。

## digit_marked
- 总准确率：`0.5625`
- 平均正确分数：`-11.8835`

- apple: accuracy=`0.5000`, mean_correct_score=`-14.2931`
- amazon: accuracy=`0.7500`, mean_correct_score=`-12.2714`
- python: accuracy=`0.3750`, mean_correct_score=`-10.6571`
- java: accuracy=`0.6250`, mean_correct_score=`-10.3125`

## word_lower
- 总准确率：`0.4688`
- 平均正确分数：`-14.7574`

- apple: accuracy=`0.7500`, mean_correct_score=`-15.7696`
- amazon: accuracy=`0.5000`, mean_correct_score=`-14.7018`
- python: accuracy=`0.1250`, mean_correct_score=`-14.2262`
- java: accuracy=`0.5000`, mean_correct_score=`-14.3318`

## letter_ab
- 总准确率：`0.5312`
- 平均正确分数：`-16.1650`

- apple: accuracy=`0.5000`, mean_correct_score=`-15.2817`
- amazon: accuracy=`0.6250`, mean_correct_score=`-17.1803`
- python: accuracy=`0.5000`, mean_correct_score=`-16.2731`
- java: accuracy=`0.5000`, mean_correct_score=`-15.9247`

## semantic_tag
- 总准确率：`0.3750`
- 平均正确分数：`-14.0983`

- apple: accuracy=`0.5000`, mean_correct_score=`-14.0404`
- amazon: accuracy=`0.5000`, mean_correct_score=`-14.2316`
- python: accuracy=`0.2500`, mean_correct_score=`-13.8294`
- java: accuracy=`0.2500`, mean_correct_score=`-14.2917`

