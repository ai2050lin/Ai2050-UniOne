# Falsifiable Stage Report

- Runs: 5
- Pattern: `tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed*/mass_noun_encoding_scan.json`

## Metric Summary (95% CI)
- overall_score: mean=0.290324, std=0.005064, 95%CI=[0.285885, 0.294763]
- causal_seq_margin: mean=0.027050, std=0.000871, 95%CI=[0.026286, 0.027814]
- causal_seq_z: mean=0.964476, std=0.027076, 95%CI=[0.940743, 0.988210]
- counterfactual_margin: mean=0.051339, std=0.000000, 95%CI=[0.051339, 0.051339]
- counterfactual_z: mean=1.761678, std=0.000000, 95%CI=[1.761678, 1.761678]
- mcs_recovery: mean=1.524439, std=0.000000, 95%CI=[1.524439, 1.524439]
- mcs_subset_size: mean=0.666667, std=0.000000, 95%CI=[0.666667, 0.666667]

## Hypothesis Check
- H1_seq_causal_margin_positive: pass (causal_seq_margin | 95% CI lower bound > 0 | threshold=0.0)
- H2_counterfactual_margin_positive: pass (counterfactual_margin | 95% CI lower bound > 0 | threshold=0.0)
- H3_mcs_recovery_ge_0_8: pass (mcs_recovery | 95% CI lower bound > 0.8 | threshold=0.8)
- H4_overall_score_ge_0_42: fail (overall_score | 95% CI lower bound > 0.42 | threshold=0.42)
- H5_seq_causal_z_ge_1_96: fail (causal_seq_z | seed mean >= 1.96 | threshold=1.96)
- H6_plasticity_not_reached_ratio_ge_0_8: pass (plasticity.not_reached_ratio | ratio >= 0.8 | threshold=0.8)
- H7_plasticity_hebb_gt_sgd1000: pass (plasticity.hebbian_minus_sgd1000 | mean(hebbian) > mean(sgd@1000) | threshold=0.0)
- H8_prompt_bootstrap_seq_margin_positive: pass (prompt_bootstrap.bootstrap_seq_margin_mean | 95% CI lower bound > 0 | threshold=0.0)
- H9_prompt_bootstrap_positive_ratio_ge_0_95: pass (prompt_bootstrap.bootstrap_positive_ratio | seed mean >= 0.95 | threshold=0.95)

## Plasticity
- source: `tempdata/deepseek7b_plasticity_efficiency_bilingual_v2_multiseed_summary.json`
- hebbian_one_shot_acc_mean: 0.581250
- sgd_step1000_mean: 0.406250
- not_reached_ratio: 1.0000

## Prompt Bootstrap
- source: `tempdata/deepseek7b_prompt_bootstrap_causal_stability_v1/prompt_bootstrap_causal_stability.json`
- bootstrap_seq_margin_mean: 0.027144
- bootstrap_seq_margin_CI_low: 0.026322
- bootstrap_positive_ratio_mean: 0.9939
- necessity_ratio_mean: 0.5833
- sufficiency_ratio_mean: 1.0000
- overshoot_ratio_mean: 0.8333

## Per Run
- seed=101, overall=0.2859, seq_margin=0.026326, cf_margin=0.051339, mcs_recovery=1.5244
- seed=202, overall=0.2898, seq_margin=0.026957, cf_margin=0.051339, mcs_recovery=1.5244
- seed=303, overall=0.2871, seq_margin=0.026096, cf_margin=0.051339, mcs_recovery=1.5244
- seed=404, overall=0.2900, seq_margin=0.028047, cf_margin=0.051339, mcs_recovery=1.5244
- seed=505, overall=0.2988, seq_margin=0.027824, cf_margin=0.051339, mcs_recovery=1.5244