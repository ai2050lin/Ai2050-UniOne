# Stage56 词元轨迹方程块

- case_count: 36
- model_count: 3
- tail_tokens: 16

## per_model
- Qwen/Qwen3-4B
  - logic: hidden=tail_pos_-1, mlp=tail_pos_-1, layer=layer_0, head=layer_0_head_0
  - style: hidden=tail_pos_-1, mlp=tail_pos_-1, layer=layer_0, head=layer_0_head_0
  - syntax: hidden=tail_pos_-1, mlp=tail_pos_-1, layer=layer_0, head=layer_0_head_0
- deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
  - logic: hidden=tail_pos_-1, mlp=tail_pos_-1, layer=layer_0, head=layer_0_head_0
  - style: hidden=tail_pos_-1, mlp=tail_pos_-1, layer=layer_0, head=layer_0_head_0
  - syntax: hidden=tail_pos_-1, mlp=tail_pos_-1, layer=layer_0, head=layer_0_head_0
- zai-org/GLM-4-9B-Chat-HF
  - logic: hidden=tail_pos_-1, mlp=tail_pos_-1, layer=layer_0, head=layer_0_head_0
  - style: hidden=tail_pos_-1, mlp=tail_pos_-1, layer=layer_0, head=layer_0_head_0
  - syntax: hidden=tail_pos_-1, mlp=tail_pos_-1, layer=layer_0, head=layer_0_head_0

## symbolic_form
- Qwen/Qwen3-4B: T*(axis,model)=argmax_t [HiddenShift(axis,t)+MLPDelta(axis,t)+Closure(axis)]
- deepseek-ai/DeepSeek-R1-Distill-Qwen-7B: T*(axis,model)=argmax_t [HiddenShift(axis,t)+MLPDelta(axis,t)+Closure(axis)]
- zai-org/GLM-4-9B-Chat-HF: T*(axis,model)=argmax_t [HiddenShift(axis,t)+MLPDelta(axis,t)+Closure(axis)]

## axis_stage6_link
- style: hidden->proto=0.0472, hidden->synergy=-0.2421, mlp->synergy=-0.0673, hidden_late->synergy=-0.1350
- logic: hidden->proto=0.0441, hidden->synergy=-0.2373, mlp->synergy=-0.0560, hidden_late->synergy=-0.1165
- syntax: hidden->proto=0.0600, hidden->synergy=-0.2580, mlp->synergy=-0.0660, hidden_late->synergy=-0.1046
