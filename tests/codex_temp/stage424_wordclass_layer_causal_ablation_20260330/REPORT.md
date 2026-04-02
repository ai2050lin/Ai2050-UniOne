# stage424_wordclass_layer_causal_ablation

## 实验设置
- 时间戳: 2026-03-30T04:29:51Z
- 是否使用 CUDA: True
- 批大小: 1
- 任务: 让模型对英文单词做六分类词性判断，再消融对应词类的高分层 MLP 通道，观察正确答案概率是否定向下降。

## 模型 qwen3
- 模型名: Qwen/Qwen3-4B
- 层数: 36

### 基线
- noun: prob=1.0000, acc=1.0000, margin=8.4479
- adjective: prob=0.8614, acc=0.8750, margin=3.5859
- verb: prob=0.9956, acc=1.0000, margin=6.1901
- adverb: prob=0.9888, acc=1.0000, margin=6.9974
- pronoun: prob=0.3074, acc=0.4000, margin=-2.9438
- preposition: prob=0.8892, acc=0.9500, margin=3.9875

### 顶层消融结果
- noun: layers=[22, 20], target_prob_delta=-0.0005, other_prob_delta_mean=-0.3991, specificity_gap_prob=+0.3986
- adjective: layers=[4, 5], target_prob_delta=-0.8444, other_prob_delta_mean=-0.4522, specificity_gap_prob=-0.3923
- verb: layers=[2, 4], target_prob_delta=-0.0099, other_prob_delta_mean=-0.3439, specificity_gap_prob=+0.3340
- adverb: layers=[23, 22], target_prob_delta=-0.4041, other_prob_delta_mean=-0.0880, specificity_gap_prob=-0.3161
- pronoun: layers=[35, 34], target_prob_delta=-0.1602, other_prob_delta_mean=-0.0466, specificity_gap_prob=-0.1136
- preposition: layers=[35, 21], target_prob_delta=-0.2775, other_prob_delta_mean=-0.0790, specificity_gap_prob=-0.1985

## 模型 deepseek7b
- 模型名: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- 层数: 28

### 基线
- noun: prob=0.7419, acc=1.0000, margin=1.9583
- adjective: prob=0.8447, acc=1.0000, margin=2.2526
- verb: prob=0.7338, acc=0.9167, margin=1.7083
- adverb: prob=0.0912, acc=0.0833, margin=-2.1224
- pronoun: prob=0.8131, acc=1.0000, margin=2.2219
- preposition: prob=0.2041, acc=0.1000, margin=-1.1438

### 顶层消融结果
- noun: layers=[1, 20], target_prob_delta=+0.0388, other_prob_delta_mean=-0.1844, specificity_gap_prob=+0.2233
- adjective: layers=[2, 3], target_prob_delta=-0.7858, other_prob_delta_mean=-0.3347, specificity_gap_prob=-0.4511
- verb: layers=[3, 2], target_prob_delta=-0.6731, other_prob_delta_mean=-0.3572, specificity_gap_prob=-0.3159
- adverb: layers=[19, 20], target_prob_delta=-0.0251, other_prob_delta_mean=-0.2630, specificity_gap_prob=+0.2379
- pronoun: layers=[2, 1], target_prob_delta=-0.7908, other_prob_delta_mean=-0.3331, specificity_gap_prob=-0.4577
- preposition: layers=[2, 3], target_prob_delta=-0.0574, other_prob_delta_mean=-0.4803, specificity_gap_prob=+0.4230
