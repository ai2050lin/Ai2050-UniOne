# Stage56 Generation Gate Internal Map Report

- Case count: 24
- Model count: 3

## Per Model
- Qwen/Qwen3-4B: cases=8
  - style: proto=-0.084361, inst=0.016083, strong=0.365261, mixed=0.044079, bridge=-0.321182, hidden=layer_35, mlp=layer_4, head=layer_0_head_12
  - logic: proto=-0.078298, inst=0.085063, strong=0.030172, mixed=-0.016398, bridge=-0.046570, hidden=layer_35, mlp=layer_35, head=layer_0_head_12
  - syntax: proto=-0.183285, inst=0.139079, strong=0.104639, mixed=-0.049033, bridge=-0.153672, hidden=layer_35, mlp=layer_4, head=layer_6_head_7
- deepseek-ai/DeepSeek-R1-Distill-Qwen-7B: cases=8
  - style: proto=0.017456, inst=-1.243660, strong=-0.493782, mixed=-0.293603, bridge=0.200179, hidden=layer_27, mlp=layer_27, head=layer_7_head_25
  - logic: proto=0.280802, inst=-0.717529, strong=-0.030884, mixed=0.236959, bridge=0.267843, hidden=layer_27, mlp=layer_27, head=layer_7_head_25
  - syntax: proto=-0.283681, inst=-0.528442, strong=-0.522021, mixed=-0.531738, bridge=-0.009717, hidden=layer_17, mlp=layer_27, head=layer_0_head_10
- zai-org/GLM-4-9B-Chat-HF: cases=8
  - style: proto=0.451823, inst=0.079590, strong=0.140560, mixed=0.267136, bridge=0.126576, hidden=layer_40, mlp=layer_39, head=layer_9_head_22
  - logic: proto=-0.025391, inst=-0.162272, strong=-0.561230, mixed=-0.371512, bridge=0.189718, hidden=layer_40, mlp=layer_39, head=layer_9_head_22
  - syntax: proto=-0.187174, inst=-0.226400, strong=-0.315007, mixed=-0.073149, bridge=0.241857, hidden=layer_40, mlp=layer_39, head=layer_0_head_19
