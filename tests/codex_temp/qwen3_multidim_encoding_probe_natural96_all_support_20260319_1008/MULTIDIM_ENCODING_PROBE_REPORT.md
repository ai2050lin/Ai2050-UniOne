# 深度神经网络多维编码探针报告

## 实验目标
- 分析风格/逻辑/句法三个维度在参数激活空间中的编码分布与分工。

## 维度摘要
- style: n_pairs=96, mean_delta_l2=275.1370, pair_cos_mean=0.6005, top1_energy=0.3949, pr=5.8646
- logic: n_pairs=96, mean_delta_l2=285.2202, pair_cos_mean=0.6882, top1_energy=0.2017, pr=15.0728
- syntax: n_pairs=96, mean_delta_l2=251.7211, pair_cos_mean=0.3843, top1_energy=0.2684, pr=10.3080

## 维度间关系
- style__logic: top_neuron_jaccard=0.0020, layer_profile_corr=0.9357
- style__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.7540
- logic__syntax: top_neuron_jaccard=0.0000, layer_profile_corr=0.5628

## 维度特异性
- style: own_mean=0.448098, other_mean=0.290975, margin=0.157123
- logic: own_mean=0.514658, other_mean=0.311700, margin=0.202958
- syntax: own_mean=0.381857, other_mean=0.256611, margin=0.125247

## 解释
- 若 top_neuron_jaccard 低且 specificity_margin 为正，说明维度存在相对分工编码。
- 若 layer_profile_corr 较高，说明不同维度共享部分层级通道（可能为通用语义骨架）。
- 若 top1_energy 高且 participation_ratio 低，说明该维编码更接近低秩控制方向。
