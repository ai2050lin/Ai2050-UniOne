# 深度神经网络多维编码探针报告

## 实验目标
- 分析风格/逻辑/句法三个维度在参数激活空间中的编码分布与分工。

## 维度摘要
- style: n_pairs=2, mean_delta_l2=393.8056, pair_cos_mean=0.7442, top1_energy=1.0000, pr=1.0000
- logic: n_pairs=2, mean_delta_l2=366.9618, pair_cos_mean=0.3409, top1_energy=1.0000, pr=1.0000
- syntax: n_pairs=2, mean_delta_l2=395.7650, pair_cos_mean=0.7611, top1_energy=1.0000, pr=1.0000

## 维度间关系
- style__logic: top_neuron_jaccard=0.0000, layer_profile_corr=0.8345
- style__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.6039
- logic__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.6523

## 维度特异性
- style: own_mean=0.743916, other_mean=0.388438, margin=0.355479
- logic: own_mean=0.644897, other_mean=0.366843, margin=0.278053
- syntax: own_mean=0.717368, other_mean=0.366857, margin=0.350511

## 解释
- 若 top_neuron_jaccard 低且 specificity_margin 为正，说明维度存在相对分工编码。
- 若 layer_profile_corr 较高，说明不同维度共享部分层级通道（可能为通用语义骨架）。
- 若 top1_energy 高且 participation_ratio 低，说明该维编码更接近低秩控制方向。
