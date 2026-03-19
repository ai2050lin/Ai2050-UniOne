# 深度神经网络多维编码探针报告

## 实验目标
- 分析风格/逻辑/句法三个维度在参数激活空间中的编码分布与分工。

## 维度摘要
- style: n_pairs=288, mean_delta_l2=276.4915, pair_cos_mean=0.5960, top1_energy=0.3836, pr=6.2244
- logic: n_pairs=288, mean_delta_l2=284.5276, pair_cos_mean=0.6791, top1_energy=0.2052, pr=16.0385
- syntax: n_pairs=288, mean_delta_l2=250.7372, pair_cos_mean=0.3884, top1_energy=0.2606, pr=11.3266

## 维度间关系
- style__logic: top_neuron_jaccard=0.0020, layer_profile_corr=0.9417
- style__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.7263
- logic__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.5400

## 维度特异性
- style: own_mean=0.447672, other_mean=0.291692, margin=0.155981
- logic: own_mean=0.511391, other_mean=0.312111, margin=0.199280
- syntax: own_mean=0.380999, other_mean=0.256080, margin=0.124919

## 解释
- 若 top_neuron_jaccard 低且 specificity_margin 为正，说明维度存在相对分工编码。
- 若 layer_profile_corr 较高，说明不同维度共享部分层级通道（可能为通用语义骨架）。
- 若 top1_energy 高且 participation_ratio 低，说明该维编码更接近低秩控制方向。
