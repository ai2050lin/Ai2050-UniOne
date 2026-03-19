# 深度神经网络多维编码探针报告

## 实验目标
- 分析风格/逻辑/句法三个维度在参数激活空间中的编码分布与分工。

## 维度摘要
- style: n_pairs=288, mean_delta_l2=488.5001, pair_cos_mean=0.5505, top1_energy=0.2991, pr=9.1778
- logic: n_pairs=288, mean_delta_l2=423.1642, pair_cos_mean=0.6818, top1_energy=0.1833, pr=16.3974
- syntax: n_pairs=288, mean_delta_l2=452.9731, pair_cos_mean=0.3454, top1_energy=0.1404, pr=20.1714

## 维度间关系
- style__logic: top_neuron_jaccard=0.0059, layer_profile_corr=0.9951
- style__syntax: top_neuron_jaccard=0.0020, layer_profile_corr=0.9954
- logic__syntax: top_neuron_jaccard=0.0039, layer_profile_corr=0.9937

## 维度特异性
- style: own_mean=0.537753, other_mean=0.355953, margin=0.181800
- logic: own_mean=0.640702, other_mean=0.433821, margin=0.206881
- syntax: own_mean=0.445471, other_mean=0.303760, margin=0.141710

## 解释
- 若 top_neuron_jaccard 低且 specificity_margin 为正，说明维度存在相对分工编码。
- 若 layer_profile_corr 较高，说明不同维度共享部分层级通道（可能为通用语义骨架）。
- 若 top1_energy 高且 participation_ratio 低，说明该维编码更接近低秩控制方向。
