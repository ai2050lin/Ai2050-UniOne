# stage495 统一语言控制变量协议

## 总结

- 更新后的统一方程：`h_{t}^{l+1} = h_{t}^{l} + C_l(form_t, context_t) + R_l(ref_t, context_t) + B_l(lemma_t) + S_l(lemma_t, context_t) + A_l(attr_t, context_t) + G_l(B_l, A_l, C_l, R_l) + O_l(h_t^l)`
- 重复出现的控制变量：`route_heads, write_neurons, mixed_binding_circuits, late_readout_amplifiers`

### qwen3

- 核心拓扑：`head_skeleton_write_then_late_readout`
- 词元补全图谱：`{"neuron_dominant": 16, "distributed_weak": 2, "mixed": 1}`
- 强控制特异性支持率：`1.0000`

### deepseek7b

- 核心拓扑：`anchor_neuron_pin_then_head_boost`
- 词元补全图谱：`{"head_dominant": 4, "mixed": 7, "distributed_weak": 1, "neuron_dominant": 7}`
- 强控制特异性支持率：`1.0000`

