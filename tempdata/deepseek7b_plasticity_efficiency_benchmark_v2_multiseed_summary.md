# Plasticity Efficiency Multiseed Summary

- Runs: 5
- Pattern: `tempdata/deepseek7b_plasticity_efficiency_benchmark_v2_seed*/plasticity_efficiency_benchmark.json`
- Hebbian one-shot acc (mean): 0.618750
- Not reached ratio (SGD fails to match Hebbian): 1.0000

## SGD Curve Stats
- step=1: mean=0.087500, std=0.018923, 95%CI=[0.070913, 0.104087]
- step=5: mean=0.064583, std=0.023754, 95%CI=[0.043762, 0.085404]
- step=20: mean=0.081250, std=0.013582, 95%CI=[0.069345, 0.093155]
- step=100: mean=0.506250, std=0.020306, 95%CI=[0.488451, 0.524049]
- step=300: mean=0.500000, std=0.019488, 95%CI=[0.482918, 0.517082]
- step=1000: mean=0.506250, std=0.020306, 95%CI=[0.488451, 0.524049]

## Per Run
- seed=101, hebbian=0.625000, steps_to_match=None, sgd=[1:0.0833, 5:0.0625, 20:0.0833, 100:0.5000, 300:0.5000, 1000:0.5000]
- seed=202, hebbian=0.604167, steps_to_match=None, sgd=[1:0.0938, 5:0.0938, 20:0.0938, 100:0.5312, 300:0.5312, 1000:0.5312]
- seed=303, hebbian=0.604167, steps_to_match=None, sgd=[1:0.0625, 5:0.0417, 20:0.0729, 100:0.4792, 300:0.4792, 1000:0.4792]
- seed=404, hebbian=0.656250, steps_to_match=None, sgd=[1:0.1146, 5:0.0833, 20:0.0625, 100:0.5208, 300:0.4896, 1000:0.5208]
- seed=505, hebbian=0.604167, steps_to_match=None, sgd=[1:0.0833, 5:0.0417, 20:0.0938, 100:0.5000, 300:0.5000, 1000:0.5000]