# Apple 三尺度微观因果编码报告

- micro: size=2, drop=0.00140040, off=0.00005156, layers={8: 1, 9: 1}
- meso: size=0, drop=0.00000000, off=0.00000000, layers={}
- macro: size=1, drop=0.00000211, off=0.00001055, layers={26: 1}

- overlap: {'micro_meso': 0.0, 'micro_macro': 0.0, 'meso_macro': 0.0}
- knowledge graph: nodes=3, edges=3

## Cross-Scale Causal Delta
- ablate micro: {'micro': -0.0014004027470946312, 'meso': -2.0987013922280084e-06, 'macro': -4.945797536493046e-05}
- ablate meso: {'micro': 0.0, 'meso': 0.0, 'macro': 0.0}
- ablate macro: {'micro': -9.427923941984773e-06, 'meso': 1.1265865609288994e-06, 'macro': -2.1058031052234583e-06}

## Apple Neighbors in Tri-Scale Space (Top-10)
- mango (fruit): dist=0.508289, micro=0.2026, meso=0.0000, macro=0.5908
- watermelon (fruit): dist=0.690083, micro=-0.2971, meso=0.0000, macro=0.3005
- flower (nature): dist=0.794284, micro=-0.4305, meso=0.0000, macro=0.2341
- tea (food): dist=0.938159, micro=-1.0544, meso=0.0000, macro=1.1553
- strawberry (fruit): dist=1.093351, micro=-0.8276, meso=0.0000, macro=0.1256
- lemon (fruit): dist=1.112218, micro=-1.0947, meso=0.0000, macro=0.4099
- leaf (nature): dist=1.174988, micro=-0.5272, meso=0.0000, macro=-0.1362
- soup (food): dist=1.252418, micro=-0.4370, meso=0.0000, macro=-0.2441
- rain (weather): dist=1.260365, micro=-1.3140, meso=0.0000, macro=0.5269
- software (tech): dist=1.287231, micro=-1.1973, meso=0.0000, macro=0.2449