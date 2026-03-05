# Plasticity Efficiency Benchmark

## Setup
- Classes: 24 (6 categories x 4 nouns)
- Feature dim (input): 530432
- Feature dim (projected): 4096
- SGD steps tested: [1, 5, 20, 100, 300, 1000]
- SGD trials: 5

## Result
- Hebbian one-shot accuracy: 0.572917
- SGD step=1: mean_acc=0.114583, std=0.000000
- SGD step=5: mean_acc=0.062500, std=0.000000
- SGD step=20: mean_acc=0.072917, std=0.000000
- SGD step=100: mean_acc=0.395833, std=0.000000
- SGD step=300: mean_acc=0.395833, std=0.000000
- SGD step=1000: mean_acc=0.395833, std=0.000000
- Steps to match Hebbian: not reached
- Efficiency ratio (steps vs one-shot): inf

## Interpretation
- If Steps to match Hebbian > 1, one-shot write is more update-efficient than repeated SGD.
- This benchmark evaluates readout-level plasticity efficiency on frozen DeepSeek features.