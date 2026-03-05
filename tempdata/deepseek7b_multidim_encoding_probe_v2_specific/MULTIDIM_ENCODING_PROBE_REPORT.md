# 深度神经网络多维编码探针报告

## 实验目标
- 分析风格/逻辑/句法三个维度在参数激活空间中的编码分布与分工。

## 维度摘要
- style: n_pairs=3, mean_delta_l2=1086.3445, pair_cos_mean=-0.1417, top1_energy=0.7627, pr=1.5675
- logic: n_pairs=3, mean_delta_l2=489.3272, pair_cos_mean=0.2535, top1_energy=0.6222, pr=1.8872
- syntax: n_pairs=3, mean_delta_l2=277.9172, pair_cos_mean=0.0350, top1_energy=0.5147, pr=1.9983

## 维度间关系
- style__logic: top_neuron_jaccard=0.0000, layer_profile_corr=0.7279
- style__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.6982
- logic__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.6728

## 维度特异性
- style: own_mean=11.636468, other_mean=0.754301, margin=10.882167
- logic: own_mean=4.625371, other_mean=0.600814, margin=4.024557
- syntax: own_mean=1.035625, other_mean=0.396072, margin=0.639553

## 解释
- 若 top_neuron_jaccard 低且 specificity_margin 为正，说明维度存在相对分工编码。
- 若 layer_profile_corr 较高，说明不同维度共享部分层级通道（可能为通用语义骨架）。
- 若 top1_energy 高且 participation_ratio 低，说明该维编码更接近低秩控制方向。
