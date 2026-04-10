# stage570 水果最小因果编码结构协议

## 核心方程
- `h_concept ~= B_global + B_family + E_concept + sum(A_attr) + sum(G_bind(concept, attr_i)) + C_context + eps`
- `apple: h_apple ~= B_global + B_fruit + E_apple + A_red + A_sweet + G_bind(apple, red) + G_bind(apple, sweet) + C_context + eps`

## 核心结论
苹果的神经元级编码不应再被理解成单个神经元标签，而应被重建成‘水果家族骨干 + 苹果概念偏置 + 可复用属性通道 + 组合绑定桥接 + 上下文修正’的最小因果结构；同一套结构也应能外推出香蕉、梨、橙子的编码。

## 组件定义
- `B_global` / 全局名词骨干：carry noun-like shared structure across families
  - 估计：`mean(h(noun_set))`
  - 因果预测：ablating this term should damage noun identity broadly, not only fruit concepts
- `B_fruit` / 水果家族骨干：carry shared fruit-family structure
  - 估计：`mean(h(fruit_set)) - B_global`
  - 因果预测：ablating this term should remove fruitness while preserving some local apple residue
- `E_concept` / 概念偏置：separate apple, banana, pear, orange within the same family
  - 估计：`mean(h(concept_prompts)) - B_global - B_fruit`
  - 因果预测：ablating only this term should collapse instance identity toward generic fruit
- `A_attr` / 属性通道：carry reusable attribute directions such as red, sweet, sour, round
  - 估计：`shared mean(h(attr + object)) - mean(h(object)) over multiple objects`
  - 因果预测：injecting this term should transfer partially across compatible objects
- `G_bind` / 绑定桥接项：bind object and attribute into a valid combination
  - 估计：`h(object, attr) - [B_global + B_family + E_concept + A_attr + C_context]`
  - 因果预测：ablating this term should preserve object and attribute traces but break their composition
- `C_context` / 上下文修正项：condition the whole representation on sentence frame, task, negation, discourse, and style
  - 估计：`residual after matched-frame substitution and context-controlled averaging`
  - 因果预测：replacing context should move this term strongly even when object and attribute stay fixed

## 估计步骤
- Step 1 / 配平提示词集合：build balanced prompts for fruit identity, attributes, and context frames
  - 例子：`This apple is red.`
  - 例子：`This banana is yellow.`
  - 例子：`The pear tastes sweet.`
  - 例子：`The orange is on the table.`
- Step 2 / 家族骨干估计：estimate B_global and B_fruit from balanced noun pools
  - `B_global ~= mean(h(noun_set))`
  - `B_fruit ~= mean(h(fruit_set)) - B_global`
- Step 3 / 概念偏置估计：estimate E_apple, E_banana, E_pear, E_orange
  - `E_apple ~= mean(h(apple_prompts)) - B_global - B_fruit`
  - `E_banana ~= mean(h(banana_prompts)) - B_global - B_fruit`
- Step 4 / 属性通道估计：extract reusable color, taste, shape, and texture channels
  - `A_red ~= shared_part(mean(h(red object_prompts)) - mean(h(object_prompts)))`
  - `A_sweet ~= shared_part(mean(h(sweet object_prompts)) - mean(h(object_prompts)))`
- Step 5 / 绑定残差估计：estimate G_bind as the non-additive composition remainder
  - `G(apple, red) ~= h(red apple) - [B_global + B_fruit + E_apple + A_red + C_context]`
  - `G(banana, sweet) ~= h(sweet banana) - [B_global + B_fruit + E_banana + A_sweet + C_context]`
- Step 6 / 因果投影回神经元级：project each component back to neuron, head, and residual candidates
  - `score(unit, component) = corr(unit_activation, component_projection)`
  - `retain units that are stable across prompts and causal under ablation/injection`

## 关键干预
- `ablate B_fruit`：apple/banana/pear/orange lose fruit-family similarity and generic fruit readout
  - 若不出现：family backbone is not a useful causal object
- `ablate E_apple`：apple drifts toward generic fruit but should not fully become animal or tool
  - 若不出现：apple identity is not localized to a compact family-internal offset
- `inject A_red`：redness should transfer partially to compatible objects such as apple, car, flower
  - 若不出现：attribute channels are not reusable across objects
- `ablate G(apple, red)`：apple and red traces remain but the red-apple composition weakens strongly
  - 若不出现：binding is close to pure addition rather than a causal bridge
- `replace frame C_context`：same apple concept moves under recipe/news/poetry/negation frames
  - 若不出现：context term is overestimated or not separable

## 跨水果外推
- `apple`：`h_apple ~= B_global + B_fruit + E_apple + A_red + A_sweet + A_round + G_bind(apple, attrs) + C_context`
- `banana`：`h_banana ~= B_global + B_fruit + E_banana + A_yellow + A_sweet + A_long + G_bind(banana, attrs) + C_context`
- `pear`：`h_pear ~= B_global + B_fruit + E_pear + A_green + A_sweet + A_soft + G_bind(pear, attrs) + C_context`
- `orange`：`h_orange ~= B_global + B_fruit + E_orange + A_orange + A_citrus + A_round + G_bind(orange, attrs) + C_context`

## 成功标准
- B_fruit ablation hurts fruit-family predictions far more than matched non-fruit controls.
- E_apple ablation weakens apple specificity while preserving a generic fruit trace.
- A_attr transfers across compatible objects but not uniformly across incompatible ones.
- G_bind ablation breaks composition more than either object-only or attribute-only ablation.
- The same decomposition predicts held-out fruit prompts and held-out attribute combinations.

## 可证伪点
- Fruit-family ablation has no stronger effect than random matched ablation.
- Concept offsets are as large as the family backbone and do not cluster within the family.
- Attribute directions fail to transfer outside the source noun.
- Composition is explained almost perfectly by additive terms and leaves no stable binding residual.
- Context replacement dominates all other terms and destroys decomposition stability.
