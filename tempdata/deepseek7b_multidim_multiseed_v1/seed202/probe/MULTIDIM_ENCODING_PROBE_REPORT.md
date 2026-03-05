# 深度神经网络多维编码探针报告

## 实验目标
- 分析风格/逻辑/句法三个维度在参数激活空间中的编码分布与分工。

## 维度摘要
- style: n_pairs=10, mean_delta_l2=713.3544, pair_cos_mean=0.1934, top1_energy=0.4108, pr=4.1062
- logic: n_pairs=10, mean_delta_l2=587.9165, pair_cos_mean=0.4417, top1_energy=0.3631, pr=5.2403
- syntax: n_pairs=10, mean_delta_l2=645.0155, pair_cos_mean=0.1566, top1_energy=0.4245, pr=3.7778

## 维度间关系
- style__logic: top_neuron_jaccard=0.0000, layer_profile_corr=0.9105
- style__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.7236
- logic__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.5141

## 维度特异性
- style: own_mean=2.884695, other_mean=0.628979, margin=2.255716
- logic: own_mean=5.665844, other_mean=0.719750, margin=4.946094
- syntax: own_mean=6.830512, other_mean=1.466023, margin=5.364490

## 解释
- 若 top_neuron_jaccard 低且 specificity_margin 为正，说明维度存在相对分工编码。
- 若 layer_profile_corr 较高，说明不同维度共享部分层级通道（可能为通用语义骨架）。
- 若 top1_energy 高且 participation_ratio 低，说明该维编码更接近低秩控制方向。
