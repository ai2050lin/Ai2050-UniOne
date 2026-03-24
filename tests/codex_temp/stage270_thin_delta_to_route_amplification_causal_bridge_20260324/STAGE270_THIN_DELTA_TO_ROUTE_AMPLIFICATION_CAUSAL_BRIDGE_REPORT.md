# Stage270 薄差分到路径放大因果桥

- 最强模型：DeepSeek-R1-Distill-Qwen-7B
- 最弱模型：Qwen3-4B
- 关键结论：前面很薄的局部差分一旦落在高杠杆位，后续路径差异会被继续放大，而且早层差分位被压低后，后续路径差异会明显回落

## Qwen3-4B
- 因果桥总分：0.3150
- 最强对象对：lion_vs_tiger
- 最弱对象对：apple_vs_pear

## DeepSeek-R1-Distill-Qwen-7B
- 因果桥总分：0.3356
- 最强对象对：apple_vs_pear
- 最弱对象对：banana_vs_peach