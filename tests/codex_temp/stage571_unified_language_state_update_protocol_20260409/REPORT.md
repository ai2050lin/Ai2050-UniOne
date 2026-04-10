# stage571 统一语言状态更新协议

## 核心结论
代词、介词、副词和逻辑推理不应再作为零散补丁加入理论，而应统一进入对象、属性、关系、指代、修饰、推理状态的同一套状态更新方程；这样理论才能从概念编码猜想升级成真正的语言计算理论。

## 主状态方程
- `S_t,l = (O_t,l, A_t,l, R_t,l, P_t,l, M_t,l, Q_t,l, G_t,l, C_t,l)`
- `S_t,l+1 = F_l(S_t,l)`
- `Q_t,l+1 = F_reason(Q_t,l, O_t,l, R_t,l, P_t,l, M_t,l, C_t,l)`
- `readout_t = R(S_t,L)`

## 状态组件
- `O_t` / 对象状态：noun/entity/object content
- `A_t` / 属性状态：color/taste/size and other object properties
- `R_t` / 关系框架：preposition, role assignment, spatial or event frame
- `P_t` / 指代状态：pronoun reference, coreference pointer, discourse target
- `M_t` / 修饰状态：adverbial scope, degree, frequency, manner, time
- `Q_t` / 推理状态：intermediate conclusions, constraints, candidate worlds
- `G_t` / 绑定状态：non-additive glue between objects, relations, and modifiers
- `C_t` / 上下文条件：task frame, style, negation, discourse history

## 四类最小实验
- E1 / 代词共指实验：代词是否主要编码为指向前文对象的引用指针，而不是独立名词语义
  - 聚焦：`P_t / Q_t`
  - 例子：`John thanked Bob because he smiled.`
  - 例子：`John thanked Bob because he helped.`
  - 例子：`Alice met Mary after she arrived.`
  - 观测量：`candidate antecedent margin`
  - 观测量：`layerwise hidden-state swap effect`
  - 观测量：`coreference classifier shift`
  - 因果预测：ablate reference units and the antecedent choice should destabilize while object words remain readable
- E2 / 介词关系框架实验：介词是否主要建立对象之间的关系框架，而不是给对象加属性
  - 聚焦：`R_t / G_t`
  - 例子：`The apple is on the table.`
  - 例子：`The apple is under the table.`
  - 例子：`The apple is near the table.`
  - 观测量：`relation-specific logit margins`
  - 观测量：`object-role swap effect`
  - 观测量：`frame reconstruction score`
  - 因果预测：ablate relation-frame units and object identity survives better than the on/under/near distinction
- E3 / 副词修饰范围实验：副词是在修饰动作、整句置信度，还是对象局部属性
  - 聚焦：`M_t / G_t / Q_t`
  - 例子：`The boy quickly opened the door.`
  - 例子：`The boy probably opened the door.`
  - 例子：`The very ripe apple fell.`
  - 观测量：`predicate readout shift`
  - 观测量：`scope-sensitive token trajectory`
  - 观测量：`confidence or manner probe response`
  - 因果预测：different adverbs should move different targets: manner adverbs mostly change event trajectory, epistemic adverbs mostly change Q_t
- E4 / 逻辑链路逐层追踪实验：中间结论是否能作为稳定推理状态沿层传播，而不是只在末层突然出现
  - 聚焦：`Q_t / R_t / P_t / C_t`
  - 例子：`All fruits are edible. Apple is a fruit. Therefore apple is edible.`
  - 例子：`If the key is in the box, then the box is on the shelf. The key is in the box. Therefore the key is on the shelf.`
  - 例子：`No red fruit is green. This apple is red. Therefore this apple is not green.`
  - 观测量：`intermediate conclusion probe`
  - 观测量：`constraint satisfaction score`
  - 观测量：`layerwise contradiction resolution`
  - 因果预测：a stable Q_t trajectory should emerge before final token readout and should be perturbable by targeted ablation

## 成功标准
- Pronoun disambiguation can be isolated as a reference-state effect rather than a generic noun effect.
- Preposition edits change relation frames more than object identity.
- Adverb classes separate into event-level and proposition-level modifiers.
- Intermediate logical conclusions appear as layerwise states before final-token readout.
- The four experiments can be described in the same state-update vocabulary without ad hoc exceptions.

## 失败信号
- Pronoun behavior cannot be separated from generic semantic similarity.
- Preposition changes are explained entirely by surface token identity with no stable relation frame.
- Adverb effects collapse into one undifferentiated context residual.
- No stable intermediate reasoning trajectory is detectable before final readout.
- The unified state-update law becomes a loose naming scheme rather than a predictive structure.
