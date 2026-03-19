# Stage56 自然生成窗口探针报告

- case_count: 72
- model_count: 3
- tail_tokens: 16

## Per Model
- Qwen/Qwen3-4B: cases=24
  - logic: hidden=tail_pos_-1, mlp=tail_pos_-1, layer=layer_34, mlp_layer=layer_35
  - style: hidden=tail_pos_-1, mlp=tail_pos_-1, layer=layer_35, mlp_layer=layer_35
  - syntax: hidden=tail_pos_-15, mlp=tail_pos_-15, layer=layer_34, mlp_layer=layer_35
- deepseek-ai/DeepSeek-R1-Distill-Qwen-7B: cases=25
  - logic: hidden=tail_pos_-7, mlp=tail_pos_-2, layer=layer_26, mlp_layer=layer_4
  - style: hidden=tail_pos_-4, mlp=tail_pos_-3, layer=layer_27, mlp_layer=layer_26
  - syntax: hidden=tail_pos_-15, mlp=tail_pos_-15, layer=layer_26, mlp_layer=layer_4
- zai-org/GLM-4-9B-Chat-HF: cases=23
  - logic: hidden=tail_pos_-4, mlp=tail_pos_-4, layer=layer_39, mlp_layer=layer_39
  - style: hidden=tail_pos_-1, mlp=tail_pos_-1, layer=layer_39, mlp_layer=layer_39
  - syntax: hidden=tail_pos_-16, mlp=tail_pos_-16, layer=layer_39, mlp_layer=layer_39

## Sample Generations
- Qwen/Qwen3-4B / abstract / style / glory: tokens=8 / suffix=things that are not things. What is
- Qwen/Qwen3-4B / abstract / logic / glory: tokens=8 / suffix=the good. But is it the case
- Qwen/Qwen3-4B / abstract / syntax / glory: tokens=8 / suffix=the category of the good, and the
- Qwen/Qwen3-4B / action / style / help: tokens=8 / suffix=things that are not things. What is
- Qwen/Qwen3-4B / action / logic / help: tokens=8 / suffix="things that are good." But if
- Qwen/Qwen3-4B / action / syntax / help: tokens=8 / suffix=the category of "help and support"
- Qwen/Qwen3-4B / animal / style / dog: tokens=8 / suffix=animals. Then, the same person asked
- Qwen/Qwen3-4B / animal / logic / dog: tokens=8 / suffix=animals. So, the statement "dog
- Qwen/Qwen3-4B / animal / syntax / dog: tokens=8 / suffix=____
A. Mammal
- Qwen/Qwen3-4B / food / style / milk: tokens=8 / suffix=liquids. Then, the same person asked
- Qwen/Qwen3-4B / food / logic / milk: tokens=8 / suffix=dairy products. Similarly, a car is
- Qwen/Qwen3-4B / food / syntax / milk: tokens=8 / suffix=____
A. Liquid
B.
