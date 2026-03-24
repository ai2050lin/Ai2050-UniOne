# Stage268：完整测试到参数级原理桥

- 最强模型：DeepSeek-R1-14B
- 最弱模型：Qwen3-4B
- 分差：0.1212
- 头号缺口：完整测试差异已经能压回参数层，但天然来源保真仍然是两边共同主断点

## Qwen3-4B
- 桥总分：0.6723
- 最强部件：parameter_hook_score
- 最弱部件：source_fidelity_parameter_score

## DeepSeek-R1-14B
- 桥总分：0.7935
- 最强部件：parameter_hook_score
- 最弱部件：source_fidelity_parameter_score