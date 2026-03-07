# Apple 多轴编码规律档案

## 核心指标
- style/logic/syntax 信号强度: 0.5786
- 交叉维度解耦指数: 0.6852
- apple micro->meso jaccard: 0.0208
- apple meso->macro jaccard: 0.3750
- apple shared_base_ratio: 0.0271
- triplet 可分离指数: 0.0959
- 轴特异性指数: 0.6297
- king_queen vs apple_king jaccard: 0.0959 vs 0.0000
- apple 层峰值区段: late (peak=0.1667)

## 规律判定
- H1_parallel_axes_exist: PASS
- H2_axes_not_collapsed: PASS
- H3_apple_hierarchy_closure: PASS
- H4_local_linearity_triplet: PASS
- H5_apple_has_layer_anchor: PASS

## 解读
- 若 meso_to_macro > micro_to_meso，说明苹果编码从属性层到实体层再到系统层形成层级闭包。
- 若 axis_specificity 与 triplet_separability 同时为正，说明关系轴与实体轴可并存而不完全塌缩。
