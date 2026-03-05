# 深度神经网络多维编码探针报告

## 实验目标
- 分析风格/逻辑/句法三个维度在参数激活空间中的编码分布与分工。

## 维度摘要
- style: n_pairs=10, mean_delta_l2=726.7782, pair_cos_mean=0.1907, top1_energy=0.3979, pr=4.0609
- logic: n_pairs=10, mean_delta_l2=592.6209, pair_cos_mean=0.4380, top1_energy=0.3843, pr=5.0127
- syntax: n_pairs=10, mean_delta_l2=640.0502, pair_cos_mean=0.2393, top1_energy=0.4040, pr=4.1191

## 维度间关系
- style__logic: top_neuron_jaccard=0.0000, layer_profile_corr=0.8886
- style__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.7686
- logic__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.4901

## 维度特异性
- style: own_mean=3.082006, other_mean=0.569816, margin=2.512191
- logic: own_mean=5.735450, other_mean=0.762240, margin=4.973209
- syntax: own_mean=6.886046, other_mean=2.180772, margin=4.705274

## 解释
- 若 top_neuron_jaccard 低且 specificity_margin 为正，说明维度存在相对分工编码。
- 若 layer_profile_corr 较高，说明不同维度共享部分层级通道（可能为通用语义骨架）。
- 若 top1_energy 高且 participation_ratio 低，说明该维编码更接近低秩控制方向。
