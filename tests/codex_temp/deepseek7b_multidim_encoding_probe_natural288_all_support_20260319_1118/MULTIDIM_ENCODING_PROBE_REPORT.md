# 深度神经网络多维编码探针报告

## 实验目标
- 分析风格/逻辑/句法三个维度在参数激活空间中的编码分布与分工。

## 维度摘要
- style: n_pairs=288, mean_delta_l2=520.2481, pair_cos_mean=0.5473, top1_energy=0.2779, pr=11.2247
- logic: n_pairs=288, mean_delta_l2=462.4313, pair_cos_mean=0.6008, top1_energy=0.1089, pr=37.0153
- syntax: n_pairs=288, mean_delta_l2=632.3500, pair_cos_mean=0.3146, top1_energy=0.3137, pr=8.8297

## 维度间关系
- style__logic: top_neuron_jaccard=0.0000, layer_profile_corr=0.9744
- style__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.6843
- logic__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.6564

## 维度特异性
- style: own_mean=0.709134, other_mean=0.477627, margin=0.231506
- logic: own_mean=0.731468, other_mean=0.495565, margin=0.235903
- syntax: own_mean=0.679273, other_mean=0.421834, margin=0.257440

## 解释
- 若 top_neuron_jaccard 低且 specificity_margin 为正，说明维度存在相对分工编码。
- 若 layer_profile_corr 较高，说明不同维度共享部分层级通道（可能为通用语义骨架）。
- 若 top1_energy 高且 participation_ratio 低，说明该维编码更接近低秩控制方向。
