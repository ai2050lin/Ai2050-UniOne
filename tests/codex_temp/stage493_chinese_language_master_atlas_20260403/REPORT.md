# stage493 中文语言模式总图谱

## 总结

- 本轮覆盖模式数：`19`，家族包括 `concrete_noun / pronoun / locative / time / quantity / connective / fixed_phrase`。
- 路线搜索参数：`TOP_CONTEXTS=2`，`TOP_HEAD_CANDIDATES=8`，`TOP_NEURON_CANDIDATES=12`，`MAX_MIXED_SUBSET=4`。

### qwen3

- 使用 CUDA（图形处理器）: `True`
- 加载模式: `prefer_cuda`
- 总体拓扑统计: `{"neuron_dominant": 16, "distributed_weak": 2, "mixed": 1}`
- `concrete_noun`: 平均基线概率 `0.1216`，平均最终下降 `0.0812`，拓扑 `{"neuron_dominant": 3}`
- `pronoun`: 平均基线概率 `0.0008`，平均最终下降 `0.0007`，拓扑 `{"neuron_dominant": 1}`
- `locative`: 平均基线概率 `0.0024`，平均最终下降 `0.0021`，拓扑 `{"neuron_dominant": 2}`
- `time`: 平均基线概率 `0.0063`，平均最终下降 `0.0056`，拓扑 `{"neuron_dominant": 3}`
- `quantity`: 平均基线概率 `0.0107`，平均最终下降 `0.0093`，拓扑 `{"distributed_weak": 2, "neuron_dominant": 1}`
- `connective`: 平均基线概率 `0.0052`，平均最终下降 `0.0043`，拓扑 `{"neuron_dominant": 4}`
- `fixed_phrase`: 平均基线概率 `0.0133`，平均最终下降 `0.0121`，拓扑 `{"mixed": 1, "neuron_dominant": 2}`

### deepseek7b

- 使用 CUDA（图形处理器）: `True`
- 加载模式: `prefer_cuda`
- 总体拓扑统计: `{"head_dominant": 4, "mixed": 7, "distributed_weak": 1, "neuron_dominant": 7}`
- `concrete_noun`: 平均基线概率 `0.0299`，平均最终下降 `0.0284`，拓扑 `{"head_dominant": 1, "mixed": 2}`
- `pronoun`: 平均基线概率 `0.0005`，平均最终下降 `0.0005`，拓扑 `{"mixed": 1}`
- `locative`: 平均基线概率 `0.0003`，平均最终下降 `0.0003`，拓扑 `{"distributed_weak": 1, "head_dominant": 1}`
- `time`: 平均基线概率 `0.0571`，平均最终下降 `0.0549`，拓扑 `{"neuron_dominant": 2, "mixed": 1}`
- `quantity`: 平均基线概率 `0.0078`，平均最终下降 `0.0060`，拓扑 `{"neuron_dominant": 2, "head_dominant": 1}`
- `connective`: 平均基线概率 `0.0008`，平均最终下降 `0.0004`，拓扑 `{"mixed": 2, "head_dominant": 1, "neuron_dominant": 1}`
- `fixed_phrase`: 平均基线概率 `0.0076`，平均最终下降 `0.0070`，拓扑 `{"neuron_dominant": 2, "mixed": 1}`

