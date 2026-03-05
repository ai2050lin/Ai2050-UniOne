# Prompt Bootstrap Causal Stability Report

- n_runs: 5
- bootstrap_seq_margin_mean: 0.027144 (95%CI [0.026322, 0.027965])
- bootstrap_positive_ratio: 0.9939
- prompt_std_seq_margin: 0.059814
- necessity_ratio: 0.5833
- sufficiency_ratio: 1.0000
- overshoot_ratio: 0.8333
- counterfactual_positive_ratio: 0.7857
- causal_seq_z_reported: 0.9645
- overall_score_reported: 0.2903

## Interpretation
- If bootstrap_seq_margin CI stays >0, prompt-level causal direction is stable.
- High necessity+sufficiency supports a compact causal subset hypothesis.
- If global z/overall remain low, local mechanism exists but system-level evidence is not yet strong.
