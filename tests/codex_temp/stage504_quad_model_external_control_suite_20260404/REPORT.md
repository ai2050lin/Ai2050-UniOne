# stage504 四模型统一外部行为强控制协议

- 说明：本轮将 Gemma4 加入统一测试，但 Gemma4 当前是 GGUF/Ollama 形态，因此属于外部行为协议，不属于层内挂钩协议。

## qwen3

- 显示名：`Qwen3-4B`
- 后端：`ollama`
- 总准确率：`1.0`
- `attribute_binding`：`1.0` (1 probes)
- `concept_hierarchy`：`1.0` (1 probes)
- `long_route`：`1.0` (2 probes)
- `pattern_completion`：`1.0` (3 probes)
- `polysemy`：`1.0` (3 probes)

## deepseek7b

- 显示名：`DeepSeek-R1-7B`
- 后端：`ollama`
- 总准确率：`0.8`
- `attribute_binding`：`1.0` (1 probes)
- `concept_hierarchy`：`1.0` (1 probes)
- `long_route`：`0.5` (2 probes)
- `pattern_completion`：`0.6667` (3 probes)
- `polysemy`：`1.0` (3 probes)

## glm4

- 显示名：`GLM-4-9B`
- 后端：`hf`
- 总准确率：`0.7`
- `attribute_binding`：`1.0` (1 probes)
- `concept_hierarchy`：`0.0` (1 probes)
- `long_route`：`1.0` (2 probes)
- `pattern_completion`：`0.3333` (3 probes)
- `polysemy`：`1.0` (3 probes)

## gemma4

- 显示名：`Gemma4-e2b`
- 后端：`ollama`
- 总准确率：`0.8`
- `attribute_binding`：`1.0` (1 probes)
- `concept_hierarchy`：`1.0` (1 probes)
- `long_route`：`1.0` (2 probes)
- `pattern_completion`：`0.3333` (3 probes)
- `polysemy`：`1.0` (3 probes)

