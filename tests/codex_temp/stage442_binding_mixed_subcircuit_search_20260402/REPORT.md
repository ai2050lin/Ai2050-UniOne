# stage442_binding_mixed_subcircuit_search

## 核心回答
如果加入 attention head（注意力头）之后，绑定消融效果明显强于只消融桥接神经元，就说明桥接项不是单独的稀疏神经元集合，而更像“头负责跨对象配对，神经元负责局部绑定写入”的混合子回路。

## 失败模型

- deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (deepseek7b): CUDA error: unknown error
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.


## deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- mixed_support_rate: 0.0000

- error: CUDA error: unknown error
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.


