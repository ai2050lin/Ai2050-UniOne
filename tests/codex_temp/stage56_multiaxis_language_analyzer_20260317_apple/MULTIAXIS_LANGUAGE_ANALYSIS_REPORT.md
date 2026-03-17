# 多轴语言结构分析块

## 核心结论
- 概念轴类型: macro_bridge_dominant
- 生成轴类型: parallel_decoupled_axes
- 跨模型严格正协同比例: 0.239130
- 最强类别: tech, nature, object, abstract, vehicle
- 最弱类别: animal, food, fruit, human

## Apple 结构
- micro->meso: 0.020803
- meso->macro: 0.375000
- shared_base_ratio: 0.027083
- hierarchy_gain: 0.354197

## 生成控制轴
- style_logic_syntax_signal: 0.578644
- cross_dim_decoupling_index: 0.685167
- axis_specificity_index: 0.629672
- triplet_separability_index: 0.095890

## 统一编码定律
- 公式: `h_t = B_family + Delta_micro + Delta_meso + Delta_macro + G_style + G_logic + G_syntax + R_relation`
- 具体概念先落在 family basis 上，再由局部 offset 决定 apple 与 banana、pear 的实例差分。
- Micro/Meso/Macro 更像概念内容的层级展开，不是简单的词性分类。
- Style/Logic/Syntax 更像生成时的并行控制轴，它们调制同一概念底座的读出方式。
- 词嵌入类比关系说明局部线性结构存在，但真实生成还叠加了上下文约束与层级门控。

## 硬伤
- fruit 类目前只有边缘正协同，说明 apple 所在家族仍未达到严格强闭合。
- 原型通道仍依赖 proxy，尚不能宣称真实类别词本体已经完全锁定。
- 跨模型类别共现不等于跨模型同机制共现，层段差异仍然很大。

## 下一步大块
- 做 apple/banana/pear 的真实类别词闭合强化块，把 fruit 从边缘正向推到稳定正协同。
- 做 concept-axis 与 generation-axis 的交叉干预块，检查改 style 是否改变逻辑轴和语法轴的选路。
- 做强类 tech 与弱类 fruit/animal 的对照块，判断失败是原型弱、实例弱还是联合冲突弱。
