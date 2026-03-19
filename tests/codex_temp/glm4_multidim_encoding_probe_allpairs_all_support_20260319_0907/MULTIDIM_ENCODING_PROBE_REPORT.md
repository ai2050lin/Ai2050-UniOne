# 深度神经网络多维编码探针报告

## 实验目标
- 分析风格/逻辑/句法三个维度在参数激活空间中的编码分布与分工。

## 维度摘要
- style: n_pairs=36, mean_delta_l2=684.7931, pair_cos_mean=0.2139, top1_energy=0.3519, pr=5.9413
- logic: n_pairs=36, mean_delta_l2=527.1739, pair_cos_mean=0.3865, top1_energy=0.2865, pr=8.9353
- syntax: n_pairs=36, mean_delta_l2=472.9274, pair_cos_mean=0.1258, top1_energy=0.2927, pr=8.2528

## 维度间关系
- style__logic: top_neuron_jaccard=0.0000, layer_profile_corr=0.9865
- style__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.9881
- logic__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.9866

## 维度特异性
- style: own_mean=0.719404, other_mean=0.466585, margin=0.252819
- logic: own_mean=0.617465, other_mean=0.438677, margin=0.178787
- syntax: own_mean=0.519974, other_mean=0.406380, margin=0.113593

## 解释
- 若 top_neuron_jaccard 低且 specificity_margin 为正，说明维度存在相对分工编码。
- 若 layer_profile_corr 较高，说明不同维度共享部分层级通道（可能为通用语义骨架）。
- 若 top1_energy 高且 participation_ratio 低，说明该维编码更接近低秩控制方向。
