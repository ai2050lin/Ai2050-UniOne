# Vision Training Report

- Generated At: 2026-02-19T13:56:14.157700+00:00
- Device: cuda
- Dataset: synthetic
- Anchor Source: logic

## Best Validation
- Epoch: 3
- Val Accuracy: 0.954333
- Val Loss: 0.183605
- Val Anchor Cosine: 0.841024

## Epoch Table
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val Anchor Cos |
|---|---:|---:|---:|---:|---:|
| 1 | 0.520976 | 0.861083 | 0.191293 | 0.953500 | 0.814021 |
| 2 | 0.184603 | 0.953417 | 0.178559 | 0.954167 | 0.842542 |
| 3 | 0.174677 | 0.955125 | 0.183605 | 0.954333 | 0.841024 |
| 4 | 0.171019 | 0.956458 | 0.179468 | 0.952167 | 0.854933 |
| 5 | 0.168495 | 0.955500 | 0.174142 | 0.953333 | 0.862193 |
| 6 | 0.165531 | 0.957583 | 0.174908 | 0.953333 | 0.866972 |

## Config
- dataset_requested: synthetic
- dataset_selected: synthetic
- total_samples: 30000
- train_size: 24000
- val_size: 6000
- batch_size: 256
- epochs: 6
- d_model: 128
- lr: 0.001
- weight_decay: 0.0001
- align_weight: 0.4
- temperature: 0.15
- seed: 42
