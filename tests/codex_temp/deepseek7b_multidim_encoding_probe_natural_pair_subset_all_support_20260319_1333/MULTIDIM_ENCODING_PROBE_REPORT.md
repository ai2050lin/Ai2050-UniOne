# 深度神经网络多维编码探针报告

## 实验目标
- 分析风格/逻辑/句法三个维度在参数激活空间中的编码分布与分工。

## 维度摘要
- style: n_pairs=53, mean_delta_l2=527.8049, pair_cos_mean=0.5186, top1_energy=0.3425, pr=7.5060
- logic: n_pairs=53, mean_delta_l2=464.6780, pair_cos_mean=0.5827, top1_energy=0.1443, pr=21.1568
- syntax: n_pairs=53, mean_delta_l2=622.2512, pair_cos_mean=0.2855, top1_energy=0.3441, pr=7.2606

## 维度间关系
- style__logic: top_neuron_jaccard=0.0000, layer_profile_corr=0.9717
- style__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.7239
- logic__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.6815

## 维度特异性
- style: own_mean=0.709319, other_mean=0.480438, margin=0.228881
- logic: own_mean=0.733451, other_mean=0.501485, margin=0.231966
- syntax: own_mean=0.678643, other_mean=0.426420, margin=0.252223

## 解释
- 若 top_neuron_jaccard 低且 specificity_margin 为正，说明维度存在相对分工编码。
- 若 layer_profile_corr 较高，说明不同维度共享部分层级通道（可能为通用语义骨架）。
- 若 top1_energy 高且 participation_ratio 低，说明该维编码更接近低秩控制方向。
