# DeepSeek7B Mass Noun Multiseed Summary

- Runs: 5
- Pattern: `tempdata/deepseek7b_mass_noun_scan_n120_multiseed_v3fix_seed*/mass_noun_encoding_scan.json`
- Grade distribution: {'insufficient_evidence': 5}

## Aggregated Stats
- overall_score: mean=0.290324, std=0.005064, min=0.285873, max=0.298801
- structure_separation: mean=0.044685, std=0.000000, min=0.044685, max=0.044685
- reuse_sparsity_structure: mean=0.596449, std=0.000000, min=0.596449, max=0.596449
- low_rank_compactness: mean=0.635449, std=0.000000, min=0.635449, max=0.635449
- causal_evidence: mean=0.065012, std=0.016881, min=0.050177, max=0.093270
- mean_causal_margin_prob: mean=-0.000000, std=0.000000, min=-0.000000, max=0.000000
- mean_causal_margin_logprob: mean=-0.001117, std=0.010974, min=-0.015879, max=0.008326
- mean_causal_margin_rank_worse: mean=-223.452963, std=26.596439, min=-252.177778, max=-198.081481
- mean_causal_margin_seq_logprob: mean=0.027050, std=0.000871, min=0.026096, max=0.028047
- mean_causal_margin_seq_avg_logprob: mean=0.008965, std=0.000820, min=0.007875, max=0.009721
- causal_margin_prob_z: mean=-0.155279, std=0.544288, min=-0.558871, max=0.713313
- causal_margin_seq_logprob_z: mean=0.964476, std=0.027076, min=0.929262, max=0.992975
- positive_causal_margin_ratio: mean=0.576667, std=0.034561, min=0.550000, max=0.633333

## Per-Run

- seed=101, grade=insufficient_evidence, overall=0.2859, causal=0.0502, z=-0.5589, logprob_margin=0.003433, seq_logprob_margin=0.026326
- seed=202, grade=insufficient_evidence, overall=0.2898, causal=0.0634, z=-0.5373, logprob_margin=-0.009495, seq_logprob_margin=0.026957
- seed=303, grade=insufficient_evidence, overall=0.2871, causal=0.0542, z=-0.4399, logprob_margin=-0.015879, seq_logprob_margin=0.026096
- seed=404, grade=insufficient_evidence, overall=0.2900, causal=0.0641, z=0.7133, logprob_margin=0.008032, seq_logprob_margin=0.028047
- seed=505, grade=insufficient_evidence, overall=0.2988, causal=0.0933, z=0.0464, logprob_margin=0.008326, seq_logprob_margin=0.027824