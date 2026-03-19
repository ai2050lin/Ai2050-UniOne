# Stage56 Generation Gate Multimodel Compare Report

- Case count: 16
- Model count: 2
- style consensus:
  - prototype_field_proxy: mixed
  - instance_field_proxy: mixed
  - bridge_field_proxy: positive
  - conflict_field_proxy: positive
  - mismatch_field_proxy: positive
- logic consensus:
  - prototype_field_proxy: positive
  - instance_field_proxy: positive
  - bridge_field_proxy: positive
  - conflict_field_proxy: positive
  - mismatch_field_proxy: positive
- syntax consensus:
  - prototype_field_proxy: negative
  - instance_field_proxy: negative
  - bridge_field_proxy: mixed
  - conflict_field_proxy: positive
  - mismatch_field_proxy: positive

## Per Model
- Qwen/Qwen3-4B: cases=8 / categories=animal,food,fruit,human,nature,object,tech,vehicle
  - style: P=-0.000121, I=-0.001410, B=0.000101, X=0.000217, M=0.000982
  - style norm: P=-0.086, I=-1.000, B=0.072, X=0.154, M=0.697
  - logic: P=0.002359, I=0.000138, B=0.000112, X=0.000922, M=0.000750
  - logic norm: P=1.000, I=0.058, B=0.047, X=0.391, M=0.318
  - syntax: P=-0.001103, I=-0.001921, B=0.000108, X=0.002023, M=0.001510
  - syntax norm: P=-0.545, I=-0.950, B=0.053, X=1.000, M=0.747
- deepseek-ai/DeepSeek-R1-Distill-Qwen-7B: cases=8 / categories=animal,food,fruit,human,nature,object,tech,vehicle
  - style: P=0.009740, I=0.007604, B=0.000619, X=0.000510, M=0.000067
  - style norm: P=1.000, I=0.781, B=0.064, X=0.052, M=0.007
  - logic: P=0.002076, I=0.003947, B=0.000184, X=0.002057, M=0.008423
  - logic norm: P=0.247, I=0.469, B=0.022, X=0.244, M=1.000
  - syntax: P=-0.001699, I=-0.001446, B=-0.000002, X=0.000284, M=0.000318
  - syntax norm: P=-1.000, I=-0.851, B=-0.001, X=0.167, M=0.187
