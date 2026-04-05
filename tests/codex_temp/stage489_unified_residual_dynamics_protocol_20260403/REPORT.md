# stage489_unified_residual_dynamics_protocol

## 核心回答
当前最合理的统一残差动力学是：R_l 决定路由，B_l 提供共享底座，S_l 负责多义切换，A_l 写入属性方向，G_l 承担对象-属性绑定，O_l 完成晚层读出。Qwen3 与 DeepSeek7B 共享这套抽象分工，但在 S_l 的实现拓扑上出现分叉：前者偏头骨架写入与晚读出放大，后者偏神经元锚点钉住与头群持续增益。

## 方程
- base = h_{t}^{l+1} = h_{t}^{l} + R_l(x_{<=t}) + B_l(lemma_t) + S_l(lemma_t, context_t) + A_l(attr_t, context_t) + G_l(B_l, A_l, route_t) + O_l(h_t^l)
- updated = h_{t}^{l+1} = h_{t}^{l} + R_l + B_l + S_l + A_l + G_l + O_l, 其中 S_l 可由两类控制杆实现：Qwen3 偏 head skeleton（头骨架），DeepSeek7B 偏 neuron anchor + head boosters（神经元锚点加头增强器）。

## 两种拓扑模式
- qwen3: head_skeleton_write_then_late_readout
  - best_order = H:5:2 -> H:5:29 -> H:5:9
  - reading = Qwen3 更像由敏感层头骨架先写入切换偏置，随后在晚层读出中被放大并收束成清晰语义分叉。
- deepseek7b: anchor_neuron_pin_then_head_boost
  - best_order = N:2:16785 -> H:2:22 -> H:2:10
  - reading = DeepSeek7B 更像由早层神经元锚点先钉住切换主方向，再由同层头群持续增强并把影响传播到中后层。

## 机制约束
- 多义词切换不能再被视为普通上下文扰动，统一协议要求它表现出稳定的低重合切换结构。
- 桥接项 G_l 不能被直接删掉，但也不能先验假定为纯神经元集合；当前更合理的约束是结构规律强、最小因果回路家族差异大。
- 残差动力学统一时，必须允许不同模型用不同拓扑实现同一抽象状态变量。
