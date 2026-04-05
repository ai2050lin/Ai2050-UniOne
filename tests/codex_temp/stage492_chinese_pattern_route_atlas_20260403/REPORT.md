# stage492 中文模式路线图谱

## 总结

### qwen3

- 使用 CUDA（图形处理器）: `True`
- 总体拓扑统计: `{"neuron_dominant": 8}`
- `concrete_noun`: 平均基线概率 `0.1216`，平均最终下降 `0.0812`，拓扑 `{"neuron_dominant": 3}`
- `device_noun`: 平均基线概率 `0.0006`，平均最终下降 `0.0005`，拓扑 `{"neuron_dominant": 1}`
- `abstract_noun`: 平均基线概率 `0.0002`，平均最终下降 `0.0001`，拓扑 `{"neuron_dominant": 2}`
- `connective`: 平均基线概率 `0.0100`，平均最终下降 `0.0082`，拓扑 `{"neuron_dominant": 2}`

### deepseek7b

- 使用 CUDA（图形处理器）: `True`
- 总体拓扑统计: `{"head_dominant": 2, "mixed": 3, "neuron_dominant": 3}`
- `concrete_noun`: 平均基线概率 `0.0299`，平均最终下降 `0.0284`，拓扑 `{"head_dominant": 1, "mixed": 2}`
- `device_noun`: 平均基线概率 `0.0012`，平均最终下降 `0.0011`，拓扑 `{"neuron_dominant": 1}`
- `abstract_noun`: 平均基线概率 `0.0002`，平均最终下降 `0.0001`，拓扑 `{"head_dominant": 1, "neuron_dominant": 1}`
- `connective`: 平均基线概率 `0.0011`，平均最终下降 `0.0007`，拓扑 `{"neuron_dominant": 1, "mixed": 1}`

