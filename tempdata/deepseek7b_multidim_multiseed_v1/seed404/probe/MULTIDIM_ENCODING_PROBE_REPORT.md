# 深度神经网络多维编码探针报告

## 实验目标
- 分析风格/逻辑/句法三个维度在参数激活空间中的编码分布与分工。

## 维度摘要
- style: n_pairs=10, mean_delta_l2=723.3351, pair_cos_mean=0.2173, top1_energy=0.5237, pr=3.1145
- logic: n_pairs=10, mean_delta_l2=577.7673, pair_cos_mean=0.4917, top1_energy=0.3338, pr=5.5964
- syntax: n_pairs=10, mean_delta_l2=621.9118, pair_cos_mean=0.2035, top1_energy=0.4483, pr=3.6838

## 维度间关系
- style__logic: top_neuron_jaccard=0.0000, layer_profile_corr=0.8305
- style__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.8270
- logic__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.5114

## 维度特异性
- style: own_mean=3.275257, other_mean=0.580759, margin=2.694498
- logic: own_mean=5.723713, other_mean=0.758316, margin=4.965397
- syntax: own_mean=6.726231, other_mean=2.275586, margin=4.450645

## 解释
- 若 top_neuron_jaccard 低且 specificity_margin 为正，说明维度存在相对分工编码。
- 若 layer_profile_corr 较高，说明不同维度共享部分层级通道（可能为通用语义骨架）。
- 若 top1_energy 高且 participation_ratio 低，说明该维编码更接近低秩控制方向。
