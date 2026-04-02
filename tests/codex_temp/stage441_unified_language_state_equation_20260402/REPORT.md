# stage441_unified_language_state_equation

## 核心回答
当前最合理的统一图景是：语言编码不是一张静态词典，而是 route（路由）- backbone（骨干）- switch（切换）- attribute（属性）- bridge（桥接）- readout（读出） 六段耦合系统。代词主线告诉我们谁先路由、谁后整合；多义名词主线告诉我们共享底座与词义切换如何避免组合爆炸；属性绑定主线告诉我们概念不是整块重写，而是骨干与修饰在桥接项上完成因果绑定。

## 统一状态方程
`h_{t}^{l+1} = h_{t}^{l} + R_l(x_{<=t}) + B_l(lemma_t) + S_l(lemma_t, context_t) + A_l(attr_t, context_t) + G_l(B_l, A_l, route_t) + O_l(h_t^l)`

## 各项含义
- R_l: 早层 route field（路由场），主要由 pronoun route pair（代词路由对）和 integrator head（整合头）承载，负责先路由后整合。
- B_l: noun backbone（名词骨干），负责公共名词底座与家族复用。
- S_l: sense switch axis（词义切换轴），在共享底座上完成 fruit/brand 等多义切换。
- A_l: attribute modifier channel（属性修饰通道），承载颜色、味道、大小等修饰方向。
- G_l: binding bridge term（绑定桥接项），把名词骨干与属性修饰真正绑定到同一对象上。
- O_l: late readout term（晚层读出项），把前面形成的内部状态压成可输出答案。

## 机制解读
- 先由 R_l 决定谁与谁需要建立关系，尤其是功能词和指代线索如何被提前路由。
- 再由 B_l 提供共享名词底座，由 S_l 在底座上完成多义切换。
- 随后 A_l 写入颜色、味道、大小等修饰方向。
- 最后由 G_l 把名词底座与属性修饰绑定成 apple-red、apple-sweet、apple-fist 这类复合概念，并交给 O_l 读出。

## 证据分数
- route_first_support: 0.0358
- shared_base_support: 1.0000
- sense_switch_support: 1.0000
- binding_reuse_support: 1.0000
- binding_causal_support: 0.0000
- readout_support: 0.7500
