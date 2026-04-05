# stage447_polysemy_family_switch_protocol

## Core Answer
Across a family of polysemous nouns, ordinary noun context variation keeps a much larger neuron-set overlap, while true polysemy uses a lower-overlap split plus a causal switch axis.

## Qwen/Qwen3-4B
- noun_count: 4
- polysemy_split_support_count: 4
- switch_causality_support_count: 0
- mean_polysemy_jaccard: 0.0892
- mean_ordinary_jaccard: 0.2959
- mean_switch_prob_drop: 0.0014
- mean_control_prob_drop: -0.0009

### apple
- best_switch_layer: 34
- sense_active_jaccard: 0.0644
- ordinary_control_mean_active_jaccard: 0.2897
- ordinary_vs_polysemy_gap: 0.2252
- switch_axis_prob_drop: 0.0057
- control_axis_prob_drop: 0.0015

### amazon
- best_switch_layer: 34
- sense_active_jaccard: 0.0802
- ordinary_control_mean_active_jaccard: 0.3586
- ordinary_vs_polysemy_gap: 0.2784
- switch_axis_prob_drop: 0.0037
- control_axis_prob_drop: -0.0059

### python
- best_switch_layer: 34
- sense_active_jaccard: 0.0894
- ordinary_control_mean_active_jaccard: 0.3219
- ordinary_vs_polysemy_gap: 0.2325
- switch_axis_prob_drop: -0.0020
- control_axis_prob_drop: -0.0013

### java
- best_switch_layer: 34
- sense_active_jaccard: 0.1228
- ordinary_control_mean_active_jaccard: 0.2134
- ordinary_vs_polysemy_gap: 0.0906
- switch_axis_prob_drop: -0.0017
- control_axis_prob_drop: 0.0020

