# Stage133: 复杂语篇名词传播块

## 核心结果
- 家族数量: 5
- 样本数量: 2048
- 早层重提相关均值: 0.9588
- 后层重提相关均值: 0.6361
- 早层符号一致率均值: 0.9583
- 后层符号一致率均值: 0.8572
- 复杂语篇名词传播分数: 0.9023

## 各语篇家族
- discourse_remention: early_corr=0.9445, late_corr=0.5408, family_score=0.8507
- causal_remention: early_corr=0.9636, late_corr=0.7104, family_score=0.9279
- nested_memory: early_corr=0.9697, late_corr=0.7146, family_score=0.9374
- contrastive_remention: early_corr=0.9570, late_corr=0.4832, family_score=0.8777
- cross_sentence_bridge: early_corr=0.9591, late_corr=0.7316, family_score=0.9181

## 理论提示
- 如果复杂语篇里第一次名词和后续重提名词在早层仍保持正相关，说明早层定锚不是一次性闪现，而是可被语篇重新调起。
- 如果后层相关也保持为正，说明后层聚合并没有在复杂语篇里完全散掉，而是在重提位置上继续参与闭合。
