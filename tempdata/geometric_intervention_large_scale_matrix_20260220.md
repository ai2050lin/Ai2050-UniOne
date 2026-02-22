# Geometric Intervention Large-Scale Matrix (2026-02-20)

| model | layer | alpha | n | treatment | random | uplift_random | pvalue_random | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| gpt2 | 3 | 0.15 | 240 | 0.3208 | 0.3042 | 0.0166 | 0.693660 | weak_signal |
| gpt2 | 3 | 0.35 | 240 | 0.6125 | 0.6208 | -0.0083 | 0.851067 | no_signal |
| distilgpt2 | 3 | 0.15 | 240 | 0.0583 | 0.0583 | 0.0000 | 1.000000 | no_signal |
| distilgpt2 | 3 | 0.35 | 240 | 0.2000 | 0.2000 | 0.0000 | 1.000000 | no_signal |

- avg_uplift_random: 0.0021
- max_uplift_random: 0.0166
- min_pvalue_random: 0.693660
- stage_judgement: weak_signal

Large-sample intervention matrix does not yet show robust positive causal uplift. Current evidence remains weak/no-signal; next step is feature-level intervention and task-metric effect evaluation beyond text-difference rate.
