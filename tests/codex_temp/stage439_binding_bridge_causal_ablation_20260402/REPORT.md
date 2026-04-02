# stage439_binding_bridge_causal_ablation

## 核心回答
如果 bridge neurons（桥接神经元）被打掉后，apple-red、apple-sweet、apple-fist 这类绑定任务明显下降，而 apple 本身的 noun readout（名词读出）和 red 或 sweet 本身的 attribute readout（属性读出）下降更小，就说明桥接神经元不是附带噪声，而是承担了真正的绑定因果角色。

## Qwen/Qwen3-4B
- 因果支持率: 0.0000

### color
- best_bridge_size: 16
- binding_drop: 0.0000
- heldout_binding_drop: 0.0000
- noun_shift: +0.0000
- attribute_shift: +0.0000
- utility: 0.0000
- causal_support: False

### taste
- best_bridge_size: 8
- binding_drop: 0.0000
- heldout_binding_drop: 0.0000
- noun_shift: +0.0000
- attribute_shift: +0.0000
- utility: 0.0000
- causal_support: False

### size
- best_bridge_size: 48
- binding_drop: 0.0013
- heldout_binding_drop: 0.0000
- noun_shift: +0.0000
- attribute_shift: +0.0000
- utility: 0.0013
- causal_support: False

## deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- 因果支持率: 0.0000

### color
- best_bridge_size: 32
- binding_drop: 0.0047
- heldout_binding_drop: 0.0000
- noun_shift: +0.0010
- attribute_shift: +0.0020
- utility: 0.0033
- causal_support: False

### taste
- best_bridge_size: 8
- binding_drop: 0.0005
- heldout_binding_drop: 0.0075
- noun_shift: +0.0000
- attribute_shift: +0.0000
- utility: 0.0005
- causal_support: False

### size
- best_bridge_size: 32
- binding_drop: 0.0052
- heldout_binding_drop: -0.0039
- noun_shift: +0.0054
- attribute_shift: -0.0117
- utility: -0.0033
- causal_support: False

