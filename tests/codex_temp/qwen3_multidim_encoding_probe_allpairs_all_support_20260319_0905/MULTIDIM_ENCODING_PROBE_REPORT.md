# 深度神经网络多维编码探针报告

## 实验目标
- 分析风格/逻辑/句法三个维度在参数激活空间中的编码分布与分工。

## 维度摘要
- style: n_pairs=36, mean_delta_l2=403.1884, pair_cos_mean=0.2478, top1_energy=0.3675, pr=4.5334
- logic: n_pairs=36, mean_delta_l2=351.5416, pair_cos_mean=0.5407, top1_energy=0.3783, pr=5.9664
- syntax: n_pairs=36, mean_delta_l2=314.3773, pair_cos_mean=0.1490, top1_energy=0.3795, pr=5.7768

## 维度间关系
- style__logic: top_neuron_jaccard=0.0000, layer_profile_corr=0.7687
- style__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.8731
- logic__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.7580

## 维度特异性
- style: own_mean=0.594427, other_mean=0.383598, margin=0.210829
- logic: own_mean=0.604227, other_mean=0.395087, margin=0.209139
- syntax: own_mean=0.482184, other_mean=0.362431, margin=0.119753

## 解释
- 若 top_neuron_jaccard 低且 specificity_margin 为正，说明维度存在相对分工编码。
- 若 layer_profile_corr 较高，说明不同维度共享部分层级通道（可能为通用语义骨架）。
- 若 top1_energy 高且 participation_ratio 低，说明该维编码更接近低秩控制方向。
