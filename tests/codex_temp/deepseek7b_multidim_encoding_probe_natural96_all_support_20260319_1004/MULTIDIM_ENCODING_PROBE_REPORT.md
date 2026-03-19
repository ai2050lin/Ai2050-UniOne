# 深度神经网络多维编码探针报告

## 实验目标
- 分析风格/逻辑/句法三个维度在参数激活空间中的编码分布与分工。

## 维度摘要
- style: n_pairs=96, mean_delta_l2=521.9089, pair_cos_mean=0.5459, top1_energy=0.2826, pr=10.5506
- logic: n_pairs=96, mean_delta_l2=459.3672, pair_cos_mean=0.6048, top1_energy=0.1131, pr=29.4398
- syntax: n_pairs=96, mean_delta_l2=646.5841, pair_cos_mean=0.3336, top1_energy=0.3351, pr=7.7750

## 维度间关系
- style__logic: top_neuron_jaccard=0.0000, layer_profile_corr=0.9758
- style__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.6569
- logic__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.6341

## 维度特异性
- style: own_mean=0.708260, other_mean=0.475008, margin=0.233252
- logic: own_mean=0.729478, other_mean=0.493718, margin=0.235760
- syntax: own_mean=0.689748, other_mean=0.420955, margin=0.268792

## 解释
- 若 top_neuron_jaccard 低且 specificity_margin 为正，说明维度存在相对分工编码。
- 若 layer_profile_corr 较高，说明不同维度共享部分层级通道（可能为通用语义骨架）。
- 若 top1_energy 高且 participation_ratio 低，说明该维编码更接近低秩控制方向。
