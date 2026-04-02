# stage431_deepseek7b_preposition_head_group_stability

## Setup
- Timestamp UTC: 2026-04-02T08:13:20Z
- Use CUDA: True
- Batch size: 1
- Candidate heads: ['H:3:4', 'H:3:20', 'H:1:14', 'H:1:7']
- Best full subset: ['H:1:14']
- Best full target drop: 0.0163
- Best heldout subset: ['H:1:14', 'H:1:7', 'H:3:20', 'H:3:4']
- Best heldout target drop: 0.0534

## Stable Core
- H:3:4: full_shapley=0.0209, heldout_shapley=0.0155, loo_full=0.0000
- H:3:20: full_shapley=0.0133, heldout_shapley=0.0147, loo_full=0.0000
- H:1:7: full_shapley=0.0112, heldout_shapley=0.0135, loo_full=0.0000
- H:1:14: full_shapley=0.0085, heldout_shapley=0.0096, loo_full=0.0163