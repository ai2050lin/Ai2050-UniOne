# 深度神经网络多维编码探针报告

## 实验目标
- 分析风格/逻辑/句法三个维度在参数激活空间中的编码分布与分工。

## 维度摘要
- style: n_pairs=10, mean_delta_l2=742.6520, pair_cos_mean=0.2002, top1_energy=0.4284, pr=3.8490
- logic: n_pairs=10, mean_delta_l2=591.7314, pair_cos_mean=0.4326, top1_energy=0.3854, pr=4.9232
- syntax: n_pairs=10, mean_delta_l2=566.3731, pair_cos_mean=0.1518, top1_energy=0.4160, pr=4.0334

## 维度间关系
- style__logic: top_neuron_jaccard=0.0000, layer_profile_corr=0.8650
- style__syntax: top_neuron_jaccard=0.0667, layer_profile_corr=0.8740
- logic__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.5692

## 维度特异性
- style: own_mean=4.143932, other_mean=1.476091, margin=2.667841
- logic: own_mean=5.652661, other_mean=0.753955, margin=4.898706
- syntax: own_mean=5.069836, other_mean=2.233049, margin=2.836787

## 解释
- 若 top_neuron_jaccard 低且 specificity_margin 为正，说明维度存在相对分工编码。
- 若 layer_profile_corr 较高，说明不同维度共享部分层级通道（可能为通用语义骨架）。
- 若 top1_energy 高且 participation_ratio 低，说明该维编码更接近低秩控制方向。
