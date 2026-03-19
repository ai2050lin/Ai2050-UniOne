# Stage56 自然生成窗口探针报告

- case_count: 36
- model_count: 3
- tail_tokens: 16

## Per Model
- Qwen/Qwen3-4B: cases=12
  - logic: hidden=tail_pos_-1, mlp=tail_pos_-1, layer=layer_34, mlp_layer=layer_35
  - style: hidden=tail_pos_-1, mlp=tail_pos_-1, layer=layer_35, mlp_layer=layer_35
  - syntax: hidden=tail_pos_-15, mlp=tail_pos_-15, layer=layer_34, mlp_layer=layer_35
- deepseek-ai/DeepSeek-R1-Distill-Qwen-7B: cases=12
  - logic: hidden=tail_pos_-7, mlp=tail_pos_-2, layer=layer_26, mlp_layer=layer_4
  - style: hidden=tail_pos_-4, mlp=tail_pos_-3, layer=layer_27, mlp_layer=layer_26
  - syntax: hidden=tail_pos_-15, mlp=tail_pos_-15, layer=layer_26, mlp_layer=layer_4
- zai-org/GLM-4-9B-Chat-HF: cases=12
  - logic: hidden=tail_pos_-5, mlp=tail_pos_-5, layer=layer_39, mlp_layer=layer_39
  - style: hidden=tail_pos_-1, mlp=tail_pos_-1, layer=layer_39, mlp_layer=layer_39
  - syntax: hidden=tail_pos_-16, mlp=tail_pos_-16, layer=layer_39, mlp_layer=layer_39

## Sample Generations
- Qwen/Qwen3-4B / abstract / style / glory: In a casual conversation, someone pointed to glory and asked what kind of thing it was. The reply was that glory belongs to the category of things that are not things. What is
- Qwen/Qwen3-4B / abstract / logic / glory: Because glory is one member of a broader class, we can say that glory belongs to the category of the good. But is it the case
- Qwen/Qwen3-4B / abstract / syntax / glory: The category to which glory belongs is the category of the good, and the
- Qwen/Qwen3-4B / action / style / help: In a casual conversation, someone pointed to help and asked what kind of thing it was. The reply was that help belongs to the category of things that are not things. What is
- Qwen/Qwen3-4B / action / logic / help: Because help is one member of a broader class, we can say that help belongs to the category of "things that are good." But if
- Qwen/Qwen3-4B / action / syntax / help: The category to which help belongs is the category of "help and support"
- Qwen/Qwen3-4B / animal / style / dog: In a casual conversation, someone pointed to dog and asked what kind of thing it was. The reply was that dog belongs to the category of animals. Then, the same person asked
- Qwen/Qwen3-4B / animal / logic / dog: Because dog is one member of a broader class, we can say that dog belongs to the category of animals. So, the statement "dog
- Qwen/Qwen3-4B / animal / syntax / dog: The category to which dog belongs is ____
A. Mammal

- Qwen/Qwen3-4B / food / style / milk: In a casual conversation, someone pointed to milk and asked what kind of thing it was. The reply was that milk belongs to the category of liquids. Then, the same person asked
- Qwen/Qwen3-4B / food / logic / milk: Because milk is one member of a broader class, we can say that milk belongs to the category of dairy products. Similarly, a car is
- Qwen/Qwen3-4B / food / syntax / milk: The category to which milk belongs is ____
A. Liquid
B.
