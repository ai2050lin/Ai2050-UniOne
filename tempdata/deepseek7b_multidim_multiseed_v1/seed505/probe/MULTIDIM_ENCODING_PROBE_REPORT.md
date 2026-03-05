# 深度神经网络多维编码探针报告

## 实验目标
- 分析风格/逻辑/句法三个维度在参数激活空间中的编码分布与分工。

## 维度摘要
- style: n_pairs=10, mean_delta_l2=769.6455, pair_cos_mean=0.1958, top1_energy=0.3753, pr=4.0592
- logic: n_pairs=10, mean_delta_l2=582.2082, pair_cos_mean=0.4277, top1_energy=0.3449, pr=5.3822
- syntax: n_pairs=10, mean_delta_l2=744.5015, pair_cos_mean=0.3183, top1_energy=0.3832, pr=4.5761

## 维度间关系
- style__logic: top_neuron_jaccard=0.0000, layer_profile_corr=0.8259
- style__syntax: top_neuron_jaccard=0.0119, layer_profile_corr=0.8813
- logic__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.5615

## 维度特异性
- style: own_mean=6.277970, other_mean=2.680089, margin=3.597881
- logic: own_mean=5.664200, other_mean=0.795591, margin=4.868609
- syntax: own_mean=6.288156, other_mean=2.026823, margin=4.261334

## 解释
- 若 top_neuron_jaccard 低且 specificity_margin 为正，说明维度存在相对分工编码。
- 若 layer_profile_corr 较高，说明不同维度共享部分层级通道（可能为通用语义骨架）。
- 若 top1_energy 高且 participation_ratio 低，说明该维编码更接近低秩控制方向。
