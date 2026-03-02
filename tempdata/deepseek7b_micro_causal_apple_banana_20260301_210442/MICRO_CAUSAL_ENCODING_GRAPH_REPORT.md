# 微观因果编码图 (MCEG) 报告

- 概念A: apple, 概念B: banana
- fruit概念子集大小: 3
- fruit概念层分布: {1: 1, 4: 1, 26: 1}

## apple
- entity: size=4, drop=0.00000001, off=0.00000138, layers={0: 3, 7: 1}
- size: size=9, drop=0.00003328, off=0.00003233, layers={0: 5, 1: 1, 2: 1, 7: 1, 23: 1}
- weight: size=6, drop=0.00002303, off=0.00003153, layers={1: 3, 2: 1, 23: 2}
- fruit: size=2, drop=0.00002799, off=0.00003073, layers={1: 1, 26: 1}
- knowledge network: nodes=15, edges=54

## banana
- entity: size=1, drop=0.00000001, off=0.00000022, layers={20: 1}
- size: size=1, drop=0.00017013, off=0.00001708, layers={5: 1}
- weight: size=3, drop=0.00000189, off=0.00000467, layers={4: 1, 7: 1, 22: 1}
- fruit: size=1, drop=0.00000005, off=0.00000022, layers={4: 1}
- knowledge network: nodes=5, edges=10

## 跨概念共享比
- entity: 0.0000
- size: 0.0000
- weight: 0.0000
- fruit: 0.0000

## fruit概念消融效应
- apple: {'entity': 3.5771024216776024e-08, 'size': -1.416249263760011e-05, 'weight': -1.555480412207544e-05, 'fruit': -2.819162968847877e-05}
- banana: {'entity': 1.955948414344988e-08, 'size': 8.175511538865976e-07, 'weight': -1.086124636155254e-07, 'fruit': -1.0115581972058862e-07}