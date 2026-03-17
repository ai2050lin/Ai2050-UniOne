# fruit 家族严格闭合缺口报告

## 发现式轨道
- strict_positive_pair_ratio: 0.200000
- mean_union_joint_adv: 0.016173
- mean_union_synergy_joint: -0.014930

## 严格真实类别词轨道
- qwen3_4b selected_in_stage3: False
- deepseek_7b selected_in_stage3: False
- qwen3_4b stage6 strict_positive_synergy_pair_count: 1
- deepseek_7b stage6 strict_positive_synergy_pair_count: 0

## apple 家族探针
- micro_to_meso_jaccard_mean: 0.020803
- meso_to_macro_jaccard_mean: 0.375000
- shared_base_ratio_mean: 0.027083

## 缺口判断
- fruit 在严格真实类别词管线里首先卡在 stage3 选择层，说明早期聚焦机制就没有稳定抓住它。
- fruit 在发现式轨道里并非完全不存在，但联合协同平均值仍为负，属于“有局部信号、无稳定协同”。
- apple 侧已经显示出 fruit 家族锚点和宏观桥接，但属性到实体的压缩仍弱，说明问题不在“完全没有家族结构”，而在“家族结构还不够硬”。

## 下一步
- 把 fruit 直接放入严格真实类别词主战场，不再让它停留在发现式轨道边缘。
- 优先用 fruit / food / nature / object 的对照块检查 fruit 的早期聚焦失败是否来自类别词本体弱。
- 如果 strict 轨仍失败，再比较真实类别词 fruit 与 proxy 原型之间的性能差距，定位原型词缺口。
