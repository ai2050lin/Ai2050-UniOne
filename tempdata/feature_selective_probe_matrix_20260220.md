# Feature-Selective Probe Matrix (2026-02-20)

| model | layer | alpha | top_k | n | top1_treatment | top1_control | top1_uplift | kl_uplift | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gpt2 | 3 | 0.35 | 32 | 240 | 0.1417 | 0.0875 | 0.0542 | 0.075862 | weak_signal |
| distilgpt2 | 3 | 0.35 | 32 | 240 | 0.0875 | 0.0125 | 0.0750 | 0.065312 | weak_signal |

- avg_top1_uplift: 0.0646
- avg_kl_uplift: 0.070587
- min_top1_uplift: 0.0542

Feature-selective intervention shows consistent positive uplift on forward metrics across two models, stronger than layer-wide smoothing baseline.
