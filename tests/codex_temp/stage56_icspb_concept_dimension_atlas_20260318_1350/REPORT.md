# ICSPB 三尺度概念图谱报告

## 全局结论
- 已覆盖锚点数: 3
- 缺失锚点数: 2
- micro->meso 平均重合: 0.026688
- meso->macro 平均重合: 0.244444
- 共享基底平均比率: 0.050000
- 风格/逻辑/语法信号: 0.578644
- 交叉维度解耦指数: 0.685167
- 关系轴特异性: 0.629672

## 概念行
- apple: category=fruit, peak=late, anchor->meso=0.030043, micro->meso=0.025641, meso->macro=0.333333, shared_base_ratio=0.058333
- cat: category=animal, peak=early, anchor->meso=0.048035, micro->meso=0.054422, meso->macro=0.200000, shared_base_ratio=0.091667
- king: category=human, peak=late, anchor->meso=0.000000, micro->meso=0.000000, meso->macro=0.200000, shared_base_ratio=0.000000
- justice: missing_anchor, peers=
- run: missing_anchor, peers=

## 生成门控共识
- style: {'P': 'mixed', 'I': 'mixed', 'B': 'positive', 'X': 'mixed', 'M': 'positive'}
- logic: {'P': 'positive', 'I': 'mixed', 'B': 'positive', 'X': 'positive', 'M': 'positive'}
- syntax: {'P': 'negative', 'I': 'negative', 'B': 'mixed', 'X': 'positive', 'M': 'positive'}

## 关系轴
- triplet_separability_index: 0.095890
- axis_specificity_index: 0.629672
- king_queen_jaccard: 0.095890
- apple_king_jaccard: 0.000000

## ICSPB 解释
- 当前真实结果支持一种中观锚点主导的联合编码：概念先以中观实体原型稳定锚定，再通过微观属性纤维与宏观协议路径展开；风格、逻辑、语法主要调制读出而不是替代概念本体。
- 当前实证已覆盖 3 个锚点：apple, cat, king。
- 当前词表没有覆盖 justice 与 run，这意味着抽象名词和动作词仍未进入同口径实证闭环。
