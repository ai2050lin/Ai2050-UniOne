# 深度神经网络多维编码探针报告

## 实验目标
- 分析风格/逻辑/句法三个维度在参数激活空间中的编码分布与分工。

## 维度摘要
- style: n_pairs=53, mean_delta_l2=490.6328, pair_cos_mean=0.5249, top1_energy=0.3643, pr=6.4526
- logic: n_pairs=53, mean_delta_l2=426.1706, pair_cos_mean=0.6750, top1_energy=0.1812, pr=13.1266
- syntax: n_pairs=53, mean_delta_l2=446.5234, pair_cos_mean=0.3225, top1_energy=0.1865, pr=13.8294

## 维度间关系
- style__logic: top_neuron_jaccard=0.0039, layer_profile_corr=0.9938
- style__syntax: top_neuron_jaccard=0.0039, layer_profile_corr=0.9946
- logic__syntax: top_neuron_jaccard=0.0039, layer_profile_corr=0.9931

## 维度特异性
- style: own_mean=0.537789, other_mean=0.356005, margin=0.181783
- logic: own_mean=0.647891, other_mean=0.438292, margin=0.209600
- syntax: own_mean=0.439981, other_mean=0.301564, margin=0.138416

## 解释
- 若 top_neuron_jaccard 低且 specificity_margin 为正，说明维度存在相对分工编码。
- 若 layer_profile_corr 较高，说明不同维度共享部分层级通道（可能为通用语义骨架）。
- 若 top1_energy 高且 participation_ratio 低，说明该维编码更接近低秩控制方向。
