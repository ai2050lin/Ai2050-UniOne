# 深度神经网络多维编码探针报告

## 实验目标
- 分析风格/逻辑/句法三个维度在参数激活空间中的编码分布与分工。

## 维度摘要
- style: n_pairs=96, mean_delta_l2=485.3806, pair_cos_mean=0.5622, top1_energy=0.3138, pr=8.3132
- logic: n_pairs=96, mean_delta_l2=424.0098, pair_cos_mean=0.6839, top1_energy=0.1842, pr=15.1223
- syntax: n_pairs=96, mean_delta_l2=459.6314, pair_cos_mean=0.3509, top1_energy=0.1462, pr=18.1264

## 维度间关系
- style__logic: top_neuron_jaccard=0.0079, layer_profile_corr=0.9951
- style__syntax: top_neuron_jaccard=0.0039, layer_profile_corr=0.9961
- logic__syntax: top_neuron_jaccard=0.0039, layer_profile_corr=0.9938

## 维度特异性
- style: own_mean=0.536196, other_mean=0.353824, margin=0.182372
- logic: own_mean=0.638132, other_mean=0.431189, margin=0.206943
- syntax: own_mean=0.455105, other_mean=0.307316, margin=0.147789

## 解释
- 若 top_neuron_jaccard 低且 specificity_margin 为正，说明维度存在相对分工编码。
- 若 layer_profile_corr 较高，说明不同维度共享部分层级通道（可能为通用语义骨架）。
- 若 top1_energy 高且 participation_ratio 低，说明该维编码更接近低秩控制方向。
