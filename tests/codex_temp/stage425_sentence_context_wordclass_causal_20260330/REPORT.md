# stage425_sentence_context_wordclass_causal

## 实验设置
- 时间戳: 2026-03-30T04:50:31Z
- 是否使用 CUDA: True
- 批大小: 1
- 任务: 在句子中标记目标词，让模型依据上下文判断其词性，再消融 stage423 的顶部层与底部层，比较正确答案概率变化。

## 模型 qwen3
- 模型名: Qwen/Qwen3-4B
- 层数: 36

### 基线
- noun: prob=1.0000, acc=1.0000, margin=9.2344
- adjective: prob=0.9990, acc=1.0000, margin=7.0000
- verb: prob=1.0000, acc=1.0000, margin=7.6146
- adverb: prob=0.9285, acc=0.9167, margin=7.8594
- pronoun: prob=0.2571, acc=0.3333, margin=-3.1302
- preposition: prob=0.3772, acc=0.4167, margin=-1.5312

### 顶部层消融
- noun: layers=[22, 20], target_prob_delta=-0.0003, other_prob_delta_mean=-0.2691, specificity_gap_prob=+0.2687
- adjective: layers=[4, 5], target_prob_delta=-0.9875, other_prob_delta_mean=-0.3241, specificity_gap_prob=-0.6634
- verb: layers=[2, 4], target_prob_delta=-0.0075, other_prob_delta_mean=-0.2791, specificity_gap_prob=+0.2717
- adverb: layers=[23, 22], target_prob_delta=-0.8371, other_prob_delta_mean=-0.0841, specificity_gap_prob=-0.7530
- pronoun: layers=[35, 34], target_prob_delta=-0.1741, other_prob_delta_mean=-0.0241, specificity_gap_prob=-0.1500
- preposition: layers=[35, 21], target_prob_delta=-0.1559, other_prob_delta_mean=-0.0399, specificity_gap_prob=-0.1160

## 模型 deepseek7b
- 模型名: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- 层数: 28

### 基线
- noun: prob=0.7656, acc=0.8333, margin=2.0104
- adjective: prob=0.8242, acc=1.0000, margin=1.7604
- verb: prob=0.3386, acc=0.4167, margin=-0.1562
- adverb: prob=0.0442, acc=0.0000, margin=-3.6354
- pronoun: prob=0.6382, acc=0.8333, margin=0.9062
- preposition: prob=0.0947, acc=0.0000, margin=-1.8854

### 顶部层消融
- noun: layers=[1, 20], target_prob_delta=+0.1790, other_prob_delta_mean=-0.1927, specificity_gap_prob=+0.3718
- adjective: layers=[2, 3], target_prob_delta=-0.8113, other_prob_delta_mean=-0.1773, specificity_gap_prob=-0.6340
- verb: layers=[3, 2], target_prob_delta=-0.3246, other_prob_delta_mean=-0.2746, specificity_gap_prob=-0.0499
- adverb: layers=[19, 20], target_prob_delta=+0.0339, other_prob_delta_mean=-0.2255, specificity_gap_prob=+0.2595
- pronoun: layers=[2, 1], target_prob_delta=-0.4263, other_prob_delta_mean=-0.1291, specificity_gap_prob=-0.2972
- preposition: layers=[2, 3], target_prob_delta=+0.0463, other_prob_delta_mean=-0.3488, specificity_gap_prob=+0.3952
