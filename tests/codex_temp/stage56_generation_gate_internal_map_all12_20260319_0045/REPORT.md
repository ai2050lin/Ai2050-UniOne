# Stage56 Generation Gate Internal Map Report

- Case count: 36
- Model count: 3

## Per Model
- Qwen/Qwen3-4B: cases=12
  - style: proto=-0.438585, inst=-0.446181, strong=-0.354612, mixed=-0.337848, bridge=0.016763, hidden=layer_35, mlp=layer_4, head=layer_1_head_28
  - logic: proto=-0.198676, inst=-0.320936, strong=-0.112435, mixed=-0.003252, bridge=0.109183, hidden=layer_35, mlp=layer_35, head=layer_0_head_9
  - syntax: proto=-0.659831, inst=-0.370260, strong=-0.424991, mixed=-0.396791, bridge=0.028200, hidden=layer_35, mlp=layer_4, head=layer_19_head_31
- deepseek-ai/DeepSeek-R1-Distill-Qwen-7B: cases=12
  - style: proto=-1.029622, inst=-0.848524, strong=0.096842, mixed=-0.330078, bridge=-0.426921, hidden=layer_27, mlp=layer_27, head=layer_0_head_4
  - logic: proto=-0.400825, inst=0.039063, strong=1.237739, mixed=0.258073, bridge=-0.979666, hidden=layer_27, mlp=layer_27, head=layer_0_head_4
  - syntax: proto=-1.062636, inst=-0.897542, strong=-1.060470, mixed=-1.655914, bridge=-0.595444, hidden=layer_17, mlp=layer_27, head=layer_0_head_10
- zai-org/GLM-4-9B-Chat-HF: cases=12
  - style: proto=3.743815, inst=2.814019, strong=3.862558, mixed=4.011574, bridge=0.149016, hidden=layer_40, mlp=layer_39, head=layer_28_head_15
  - logic: proto=2.202257, inst=1.586914, strong=2.077836, mixed=2.679253, bridge=0.601418, hidden=layer_40, mlp=layer_39, head=layer_4_head_29
  - syntax: proto=3.104384, inst=1.935872, strong=2.868634, mixed=3.691406, bridge=0.822772, hidden=layer_40, mlp=layer_39, head=layer_0_head_19
