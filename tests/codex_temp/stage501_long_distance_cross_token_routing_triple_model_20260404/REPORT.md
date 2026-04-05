# stage501 长距离跨词元路由三模型测试

## qwen3

- route_heads 主导数: `0/5`
- write_neurons 主导数: `5/5`
- mixed 主导数: `0/5`
- 平均 attn/mlp 比: `0.0684`

### long_pronoun

- 主导机制: `write_neurons`
- 平均 attn 影响: `-0.052549`
- 平均 mlp 影响: `0.877927`
- 峰值 attn 层: `L35`
- 峰值 mlp 层: `L0`

### long_reference_chain

- 主导机制: `write_neurons`
- 平均 attn 影响: `0.379166`
- 平均 mlp 影响: `1.452495`
- 峰值 attn 层: `L0`
- 峰值 mlp 层: `L0`

### long_quantity

- 主导机制: `write_neurons`
- 平均 attn 影响: `0.094345`
- 平均 mlp 影响: `1.134601`
- 峰值 attn 层: `L0`
- 峰值 mlp 层: `L9`

### long_logic_chain

- 主导机制: `write_neurons`
- 平均 attn 影响: `0.037366`
- 平均 mlp 影响: `0.805027`
- 峰值 attn 层: `L18`
- 峰值 mlp 层: `L0`

### long_attribute

- 主导机制: `write_neurons`
- 平均 attn 影响: `0.045953`
- 平均 mlp 影响: `4.107543`
- 峰值 attn 层: `L0`
- 峰值 mlp 层: `L0`

## deepseek7b

- route_heads 主导数: `5/5`
- write_neurons 主导数: `0/5`
- mixed 主导数: `0/5`
- 平均 attn/mlp 比: `3.4747`

### long_pronoun

- 主导机制: `route_heads`
- 平均 attn 影响: `1.852419`
- 平均 mlp 影响: `0.827156`
- 峰值 attn 层: `L0`
- 峰值 mlp 层: `L27`

### long_reference_chain

- 主导机制: `route_heads`
- 平均 attn 影响: `2.084169`
- 平均 mlp 影响: `1.169918`
- 峰值 attn 层: `L0`
- 峰值 mlp 层: `L0`

### long_quantity

- 主导机制: `route_heads`
- 平均 attn 影响: `1.334217`
- 平均 mlp 影响: `0.49911`
- 峰值 attn 层: `L0`
- 峰值 mlp 层: `L0`

### long_logic_chain

- 主导机制: `route_heads`
- 平均 attn 影响: `2.194948`
- 平均 mlp 影响: `0.345911`
- 峰值 attn 层: `L0`
- 峰值 mlp 层: `L0`

### long_attribute

- 主导机制: `route_heads`
- 平均 attn 影响: `3.200782`
- 平均 mlp 影响: `0.738545`
- 峰值 attn 层: `L0`
- 峰值 mlp 层: `L0`

## glm4

- route_heads 主导数: `0/5`
- write_neurons 主导数: `5/5`
- mixed 主导数: `0/5`
- 平均 attn/mlp 比: `-0.0188`

### long_pronoun

- 主导机制: `write_neurons`
- 平均 attn 影响: `-0.016531`
- 平均 mlp 影响: `2.040214`
- 峰值 attn 层: `L39`
- 峰值 mlp 层: `L0`

### long_reference_chain

- 主导机制: `write_neurons`
- 平均 attn 影响: `0.224465`
- 平均 mlp 影响: `2.935732`
- 峰值 attn 层: `L0`
- 峰值 mlp 层: `L0`

### long_quantity

- 主导机制: `write_neurons`
- 平均 attn 影响: `-0.251778`
- 平均 mlp 影响: `2.06329`
- 峰值 attn 层: `L10`
- 峰值 mlp 层: `L0`

### long_logic_chain

- 主导机制: `write_neurons`
- 平均 attn 影响: `-0.118155`
- 平均 mlp 影响: `1.985087`
- 峰值 attn 层: `L29`
- 峰值 mlp 层: `L0`

### long_attribute

- 主导机制: `write_neurons`
- 平均 attn 影响: `0.050225`
- 平均 mlp 影响: `2.603462`
- 峰值 attn 层: `L0`
- 峰值 mlp 层: `L0`

## 综合结论

- 在更长距离的跨词元任务中，如果 route_heads 仍然没有稳定单独抬头，就说明语言搬运机制更可能是头与神经元共同组成的混合回路，而不是简单的“注意力单独接管”。
