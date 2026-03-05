# Multidim Multi-seed Stability Report

- n_runs: 5
- max_pairs_per_dim: 10 (total prompts >= 30)

## Aggregate (95% CI)
- specificity_margin_style: mean=2.745625, 95%CI=[2.300937, 3.190314]
- specificity_margin_logic: mean=4.930403, 95%CI=[4.890895, 4.969911]
- specificity_margin_syntax: mean=4.323706, 95%CI=[3.508506, 5.138905]
- diag_adv_style: mean=0.023012, 95%CI=[0.012232, 0.033792]
- diag_adv_logic: mean=0.040570, 95%CI=[0.037365, 0.043776]
- diag_adv_syntax: mean=0.000731, 95%CI=[-0.000702, 0.002164]
- cross_jaccard_style_logic: mean=0.000000, 95%CI=[0.000000, 0.000000]
- cross_jaccard_style_syntax: mean=0.015705, 95%CI=[-0.009669, 0.041078]
- cross_jaccard_logic_syntax: mean=0.000000, 95%CI=[0.000000, 0.000000]

## Hypotheses
- H1_style_specificity_positive: pass (specificity_margin_style | 95% CI lower bound > 0)
- H2_logic_specificity_positive: pass (specificity_margin_logic | 95% CI lower bound > 0)
- H3_syntax_specificity_positive: pass (specificity_margin_syntax | 95% CI lower bound > 0)
- H4_style_diag_adv_positive: pass (diag_adv_style | 95% CI lower bound > 0)
- H5_logic_diag_adv_positive: pass (diag_adv_logic | 95% CI lower bound > 0)
- H6_syntax_diag_adv_positive: fail (diag_adv_syntax | 95% CI lower bound > 0)
