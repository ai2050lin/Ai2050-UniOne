# stage508 长距离跨词元路由四模型内部机制协议

- 本轮在 stage501 的基础上，将 Gemma4 纳入同一层内协议。

## qwen3

- 模型：`Qwen3-4B`
- 层数：`36`
- route_heads 主导：`0/5`
- write_neurons 主导：`5/5`
- mixed：`0/5`
- 平均 attn/mlp 比：`0.0684`

## deepseek7b

- 模型：`DeepSeek-R1-Distill-Qwen-7B`
- 层数：`28`
- route_heads 主导：`5/5`
- write_neurons 主导：`0/5`
- mixed：`0/5`
- 平均 attn/mlp 比：`3.4747`

## glm4

- 模型：`GLM-4-9B-Chat-HF`
- 层数：`40`
- route_heads 主导：`0/5`
- write_neurons 主导：`5/5`
- mixed：`0/5`
- 平均 attn/mlp 比：`-0.0188`

## gemma4

- 模型：`Gemma-4-E2B-it`
- 层数：`35`
- route_heads 主导：`2/5`
- write_neurons 主导：`3/5`
- mixed：`0/5`
- 平均 attn/mlp 比：`0.6751`

