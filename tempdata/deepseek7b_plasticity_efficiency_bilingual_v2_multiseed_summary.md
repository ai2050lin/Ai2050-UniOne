# Plasticity Efficiency Multiseed Summary

- Runs: 5
- Pattern: `tempdata/deepseek7b_plasticity_efficiency_bilingual_v2_seed*/plasticity_efficiency_benchmark.json`
- Hebbian one-shot acc (mean): 0.581250
- Not reached ratio (SGD fails to match Hebbian): 1.0000

## SGD Curve Stats
- step=1: mean=0.093750, std=0.032106, 95%CI=[0.065608, 0.121892]
- step=5: mean=0.085417, std=0.032443, 95%CI=[0.056980, 0.113854]
- step=20: mean=0.097917, std=0.030010, 95%CI=[0.071612, 0.124222]
- step=100: mean=0.406250, std=0.035325, 95%CI=[0.375287, 0.437213]
- step=300: mean=0.406250, std=0.035325, 95%CI=[0.375287, 0.437213]
- step=1000: mean=0.406250, std=0.035325, 95%CI=[0.375287, 0.437213]

## Per Run
- seed=101, hebbian=0.572917, steps_to_match=None, sgd=[1:0.1146, 5:0.0938, 20:0.1042, 100:0.4271, 300:0.4271, 1000:0.4271]
- seed=202, hebbian=0.583333, steps_to_match=None, sgd=[1:0.1146, 5:0.0521, 20:0.0938, 100:0.4062, 300:0.4062, 1000:0.4062]
- seed=303, hebbian=0.593750, steps_to_match=None, sgd=[1:0.0417, 5:0.1354, 20:0.0729, 100:0.4479, 300:0.4479, 1000:0.4479]
- seed=404, hebbian=0.572917, steps_to_match=None, sgd=[1:0.1146, 5:0.0625, 20:0.0729, 100:0.3958, 300:0.3958, 1000:0.3958]
- seed=505, hebbian=0.583333, steps_to_match=None, sgd=[1:0.0833, 5:0.0833, 20:0.1458, 100:0.3542, 300:0.3542, 1000:0.3542]