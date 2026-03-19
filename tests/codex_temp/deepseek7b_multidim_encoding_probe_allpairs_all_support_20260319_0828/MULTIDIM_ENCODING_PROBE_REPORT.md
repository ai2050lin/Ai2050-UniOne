# 深度神经网络多维编码探针报告

## 实验目标
- 分析风格/逻辑/句法三个维度在参数激活空间中的编码分布与分工。

## 维度摘要
- style: n_pairs=36, mean_delta_l2=770.8139, pair_cos_mean=0.2074, top1_energy=0.2766, pr=6.6204
- logic: n_pairs=36, mean_delta_l2=631.4593, pair_cos_mean=0.4722, top1_energy=0.2910, pr=9.2623
- syntax: n_pairs=36, mean_delta_l2=657.0449, pair_cos_mean=0.2002, top1_energy=0.3149, pr=6.7822

## 维度间关系
- style__logic: top_neuron_jaccard=0.0000, layer_profile_corr=0.7016
- style__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.8329
- logic__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.5991

## 维度特异性
- style: own_mean=0.870845, other_mean=0.578058, margin=0.292787
- logic: own_mean=0.896599, other_mean=0.607621, margin=0.288977
- syntax: own_mean=0.826075, other_mean=0.597308, margin=0.228766

## 解释
- 若 top_neuron_jaccard 低且 specificity_margin 为正，说明维度存在相对分工编码。
- 若 layer_profile_corr 较高，说明不同维度共享部分层级通道（可能为通用语义骨架）。
- 若 top1_energy 高且 participation_ratio 低，说明该维编码更接近低秩控制方向。
