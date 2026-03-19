# Stage56 Generation Gate Multimodel Compare Report

- Case count: 12
- Model count: 3
- style consensus:
  - prototype_field_proxy: mixed
  - instance_field_proxy: mixed
  - bridge_field_proxy: positive
  - conflict_field_proxy: positive
  - mismatch_field_proxy: mixed
- logic consensus:
  - prototype_field_proxy: positive
  - instance_field_proxy: mixed
  - bridge_field_proxy: mixed
  - conflict_field_proxy: positive
  - mismatch_field_proxy: positive
- syntax consensus:
  - prototype_field_proxy: negative
  - instance_field_proxy: negative
  - bridge_field_proxy: mixed
  - conflict_field_proxy: positive
  - mismatch_field_proxy: positive

## Per Model
- Qwen/Qwen3-4B: cases=4 / categories=food,fruit,nature,object
  - style: P=-0.000273, I=-0.002754, B=0.000159, X=0.000399, M=0.001963
  - logic: P=0.005531, I=0.001077, B=-0.000076, X=0.001143, M=0.000793
  - syntax: P=-0.001949, I=-0.003838, B=-0.000052, X=0.004004, M=0.002785
- deepseek-ai/DeepSeek-R1-Distill-Qwen-7B: cases=4 / categories=food,fruit,nature,object
  - style: P=0.019919, I=0.015108, B=0.001171, X=0.000127, M=0.000137
  - logic: P=0.003964, I=0.007661, B=0.000233, X=0.004014, M=0.016577
  - syntax: P=-0.003594, I=-0.003082, B=0.000000, X=0.000390, M=0.000330
- zai-org/GLM-4-9B-Chat-HF: cases=4 / categories=food,fruit,nature,object
  - style: P=-0.000274, I=0.002764, B=0.000862, X=0.000003, M=-0.000022
  - logic: P=0.001881, I=-0.000870, B=0.000025, X=0.000425, M=0.001760
  - syntax: P=-0.000651, I=-0.001171, B=0.001094, X=0.000055, M=0.000028
