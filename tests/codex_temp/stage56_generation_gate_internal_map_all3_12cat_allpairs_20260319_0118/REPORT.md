# Stage56 Generation Gate Internal Map Report

- Case count: 72
- Model count: 3

## Per Model
- Qwen/Qwen3-4B: cases=24
  - style: proto=-0.333171, inst=-0.416165, strong=-0.323211, mixed=-0.287219, bridge=0.035992, hidden=layer_35, mlp=layer_4, head=layer_0_head_9
  - logic: proto=-0.202528, inst=-0.344238, strong=-0.191987, mixed=-0.097973, bridge=0.094013, hidden=layer_35, mlp=layer_35, head=layer_0_head_9
  - syntax: proto=-0.567980, inst=-0.362765, strong=-0.422384, mixed=-0.386378, bridge=0.036006, hidden=layer_35, mlp=layer_4, head=layer_1_head_10
- deepseek-ai/DeepSeek-R1-Distill-Qwen-7B: cases=25
  - style: proto=-1.150625, inst=-0.827708, strong=0.055755, mixed=-0.772240, bridge=-0.827995, hidden=layer_27, mlp=layer_27, head=layer_0_head_4
  - logic: proto=-0.972500, inst=-0.337500, strong=0.726979, mixed=-0.709823, bridge=-1.436802, hidden=layer_27, mlp=layer_27, head=layer_0_head_4
  - syntax: proto=-1.284552, inst=-1.299760, strong=-1.455022, mixed=-1.691277, bridge=-0.236256, hidden=layer_17, mlp=layer_27, head=layer_0_head_10
- zai-org/GLM-4-9B-Chat-HF: cases=23
  - style: proto=3.364357, inst=2.588089, strong=3.865225, mixed=3.529778, bridge=-0.335447, hidden=layer_40, mlp=layer_39, head=layer_4_head_29
  - logic: proto=2.360507, inst=1.801008, strong=2.684915, mixed=2.635719, bridge=-0.049196, hidden=layer_40, mlp=layer_39, head=layer_4_head_29
  - syntax: proto=3.011322, inst=1.977072, strong=3.278042, mixed=3.379019, bridge=0.100978, hidden=layer_40, mlp=layer_39, head=layer_0_head_19
