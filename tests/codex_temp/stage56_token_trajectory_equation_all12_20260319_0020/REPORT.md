# Stage56 词元轨迹方程块

- case_count: 36
- model_count: 3
- tail_tokens: 16

## per_model
- Qwen/Qwen3-4B
  - logic: hidden=tail_pos_-5, mlp=tail_pos_-5, layer=layer_34, head=layer_1_head_13
  - style: hidden=tail_pos_-5, mlp=tail_pos_-5, layer=layer_34, head=layer_1_head_13
  - syntax: hidden=tail_pos_-5, mlp=tail_pos_-4, layer=layer_34, head=layer_4_head_8
- deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
  - logic: hidden=tail_pos_-5, mlp=tail_pos_-5, layer=layer_23, head=layer_6_head_17
  - style: hidden=tail_pos_-5, mlp=tail_pos_-5, layer=layer_24, head=layer_4_head_26
  - syntax: hidden=tail_pos_-3, mlp=tail_pos_-3, layer=layer_17, head=layer_20_head_19
- zai-org/GLM-4-9B-Chat-HF
  - logic: hidden=tail_pos_-7, mlp=tail_pos_-6, layer=layer_39, head=layer_8_head_2
  - style: hidden=tail_pos_-7, mlp=tail_pos_-6, layer=layer_39, head=layer_39_head_30
  - syntax: hidden=tail_pos_-7, mlp=tail_pos_-5, layer=layer_39, head=layer_4_head_16

## symbolic_form
- Qwen/Qwen3-4B: T*(axis,model)=argmax_t [HiddenShift(axis,t)+MLPDelta(axis,t)+Closure(axis)]
- deepseek-ai/DeepSeek-R1-Distill-Qwen-7B: T*(axis,model)=argmax_t [HiddenShift(axis,t)+MLPDelta(axis,t)+Closure(axis)]
- zai-org/GLM-4-9B-Chat-HF: T*(axis,model)=argmax_t [HiddenShift(axis,t)+MLPDelta(axis,t)+Closure(axis)]

## axis_stage6_link
- style: hidden->proto=0.0472, hidden->synergy=-0.2421, mlp->synergy=-0.0673, hidden_late->synergy=-0.1350
- logic: hidden->proto=0.0441, hidden->synergy=-0.2373, mlp->synergy=-0.0560, hidden_late->synergy=-0.1165
- syntax: hidden->proto=0.0600, hidden->synergy=-0.2580, mlp->synergy=-0.0660, hidden_late->synergy=-0.1046
