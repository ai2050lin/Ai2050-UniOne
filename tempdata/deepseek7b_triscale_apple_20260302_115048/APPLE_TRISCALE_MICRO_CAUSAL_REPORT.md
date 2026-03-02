# Apple 三尺度微观因果编码报告

- micro: size=2, drop=0.00140040, off=0.00005017, layers={8: 1, 9: 1}
- meso: size=0, drop=0.00000000, off=0.00000000, layers={}
- macro: size=2, drop=0.00011685, off=0.00148261, layers={9: 1, 23: 1}

- overlap: {'micro_meso': 0.0, 'micro_macro': 0.3333333333333333, 'meso_macro': 0.0}
- knowledge graph: nodes=3, edges=3

## Cross-Scale Causal Delta
- ablate micro: {'micro': -0.0014004027470946312, 'meso': -7.131888644096307e-07, 'macro': -4.945797536493046e-05}
- ablate meso: {'micro': 0.0, 'meso': 0.0, 'macro': 0.0}
- ablate macro: {'micro': -0.0014821256918366998, 'meso': -4.823783150698091e-07, 'macro': -0.00011684628680086462}

## Apple Neighbors in Tri-Scale Space (Top-10)
- peach (fruit): dist=0.900669, micro=-0.2267, meso=0.0000, macro=2.0496
- strawberry (fruit): dist=1.499189, micro=-0.8276, meso=0.0000, macro=1.6162
- mango (fruit): dist=1.847223, micro=0.2026, meso=0.0000, macro=1.1294
- watermelon (fruit): dist=2.082286, micro=-0.2971, meso=0.0000, macro=0.8696
- soup (food): dist=2.682549, micro=-0.4370, meso=0.0000, macro=0.2801
- rice (food): dist=2.794455, micro=-0.5614, meso=0.0000, macro=0.1839
- flower (nature): dist=2.802179, micro=-0.4305, meso=0.0000, macro=0.1591
- leaf (nature): dist=3.218566, micro=-0.5272, meso=0.0000, macro=-0.2490
- lemon (fruit): dist=3.296274, micro=-1.0947, meso=0.0000, macro=-0.2078
- meat (food): dist=3.516076, micro=-0.9988, meso=0.0000, macro=-0.4626