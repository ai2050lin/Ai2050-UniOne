# 深度神经网络多维编码探针报告

## 实验目标
- 分析风格/逻辑/句法三个维度在参数激活空间中的编码分布与分工。

## 维度摘要
- style: n_pairs=2, mean_delta_l2=619.8059, pair_cos_mean=0.5887, top1_energy=1.0000, pr=1.0000
- logic: n_pairs=2, mean_delta_l2=553.9078, pair_cos_mean=0.1435, top1_energy=1.0000, pr=1.0000
- syntax: n_pairs=2, mean_delta_l2=518.0164, pair_cos_mean=0.6589, top1_energy=1.0000, pr=1.0000

## 维度间关系
- style__logic: top_neuron_jaccard=0.0000, layer_profile_corr=0.9808
- style__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.8584
- logic__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.9225

## 维度特异性
- style: own_mean=0.845517, other_mean=0.438012, margin=0.407505
- logic: own_mean=0.740049, other_mean=0.422114, margin=0.317935
- syntax: own_mean=0.720002, other_mean=0.388038, margin=0.331964

## 解释
- 若 top_neuron_jaccard 低且 specificity_margin 为正，说明维度存在相对分工编码。
- 若 layer_profile_corr 较高，说明不同维度共享部分层级通道（可能为通用语义骨架）。
- 若 top1_energy 高且 participation_ratio 低，说明该维编码更接近低秩控制方向。
