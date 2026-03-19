# 深度神经网络多维编码探针报告

## 实验目标
- 分析风格/逻辑/句法三个维度在参数激活空间中的编码分布与分工。

## 维度摘要
- style: n_pairs=53, mean_delta_l2=282.8525, pair_cos_mean=0.5741, top1_energy=0.4536, pr=4.5145
- logic: n_pairs=53, mean_delta_l2=284.7320, pair_cos_mean=0.6710, top1_energy=0.2423, pr=11.5344
- syntax: n_pairs=53, mean_delta_l2=252.6671, pair_cos_mean=0.3610, top1_energy=0.3271, pr=7.8043

## 维度间关系
- style__logic: top_neuron_jaccard=0.0020, layer_profile_corr=0.9362
- style__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.6843
- logic__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.4833

## 维度特异性
- style: own_mean=0.452717, other_mean=0.294199, margin=0.158518
- logic: own_mean=0.515412, other_mean=0.316088, margin=0.199324
- syntax: own_mean=0.384915, other_mean=0.257329, margin=0.127586

## 解释
- 若 top_neuron_jaccard 低且 specificity_margin 为正，说明维度存在相对分工编码。
- 若 layer_profile_corr 较高，说明不同维度共享部分层级通道（可能为通用语义骨架）。
- 若 top1_energy 高且 participation_ratio 低，说明该维编码更接近低秩控制方向。
