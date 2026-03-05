# Tri-axial Parameter Structure Report

## Concepts
- apple, banana, pineapple, cat, dog

## Apple vs Cat (Axis-level Structural Overlap)
- micro_attr: neuron=0.0000, layer=0.0000, gate_dim=0.0000, down_dim=0.0000
- same_type: neuron=0.0000, layer=0.2500, gate_dim=0.0000, down_dim=0.0000
- super_type: neuron=0.0000, layer=0.0000, gate_dim=0.0000, down_dim=0.0000

## Group Shared Gate Input Dims
- group=fruit
  - micro_attr: common_gate_dims=[]
  - same_type: common_gate_dims=[]
  - super_type: common_gate_dims=[2041, 2427, 2467, 2524]
- group=animal
  - micro_attr: common_gate_dims=[]
  - same_type: common_gate_dims=[]
  - super_type: common_gate_dims=[2727]

## Per-Concept Axis Signatures
- concept=apple (group=fruit)
  - axis=micro_attr, subset=['L23N5458', 'L27N12646', 'L27N1011', 'L27N3077']
    gate_dims=[2718, 1026, 3503, 3453, 2376, 775, 26, 2557], down_dims=[1801, 1367, 1788, 2652, 487, 1419, 3003, 1707]
  - axis=same_type, subset=['L27N2747', 'L26N4172', 'L27N7733', 'L27N4210']
    gate_dims=[3053, 3327, 879, 209, 956, 2795, 2040, 2326], down_dims=[2906, 471, 1182, 3046, 2561, 2123, 2570, 3252]
  - axis=super_type, subset=['L2N8981', 'L23N11613', 'L22N16034', 'L21N2199']
    gate_dims=[2524, 2467, 2427, 3577, 3402, 2041, 2953, 2747], down_dims=[763, 2524, 2467, 2427, 2041, 1703, 1502, 1854]
- concept=banana (group=fruit)
  - axis=micro_attr, subset=['L25N5723', 'L23N5458', 'L27N278', 'L27N17821']
    gate_dims=[1430, 2418, 183, 1864, 1205, 2630, 1026, 3503], down_dims=[2863, 409, 3151, 1770, 1511, 3343, 1801, 1367]
  - axis=same_type, subset=['L27N4327', 'L27N1704']
    gate_dims=[565, 3285, 1797, 2979, 1715, 3361, 1474, 2369], down_dims=[2451, 1422, 819, 2561, 1860, 2041, 792, 2123]
  - axis=super_type, subset=['L1N13387', 'L24N17006', 'L23N12028', 'L22N16034']
    gate_dims=[2467, 2041, 2524, 2427, 2599, 2127, 670, 3206], down_dims=[822, 3090, 1068, 2477, 1470, 1717, 3251, 244]
- concept=pineapple (group=fruit)
  - axis=micro_attr, subset=['L23N4356', 'L27N15791', 'L27N4001', 'L3N4278']
    gate_dims=[2718, 1422, 2117, 1068, 959, 763, 2561, 2730], down_dims=[2348, 1029, 2906, 2053, 3003, 822, 2467, 1865]
  - axis=same_type, subset=['L5N6513', 'L2N18502', 'L27N11121', 'L27N13465']
    gate_dims=[2030, 230, 2718, 2175, 3474, 3564, 2913, 594], down_dims=[2738, 2737, 2107, 2914, 1491, 3290, 1009, 1003]
  - axis=super_type, subset=['L1N13387', 'L24N17006', 'L23N12028', 'L22N16034']
    gate_dims=[2467, 2041, 2524, 2427, 2599, 2127, 670, 3206], down_dims=[822, 3090, 1068, 2477, 1470, 1717, 3251, 244]
- concept=cat (group=animal)
  - axis=micro_attr, subset=['L22N2369', 'L1N9692', 'L22N8252', 'L5N16166']
    gate_dims=[362, 456, 3270, 130, 523, 3046, 2927, 877], down_dims=[130, 2927, 523, 362, 650, 877, 1092, 1460]
  - axis=same_type, subset=['L21N17511', 'L2N4830', 'L27N303', 'L27N11609']
    gate_dims=[1571, 3003, 1572, 1865, 3090, 3550, 1929, 3263], down_dims=[1571, 3003, 1865, 1572, 3550, 3090, 1853, 1891]
  - axis=super_type, subset=['L1N3007']
    gate_dims=[1432, 2727, 1768, 1511, 822, 1699], down_dims=[1130, 3223, 953, 325, 2279, 2147]
- concept=dog (group=animal)
  - axis=micro_attr, subset=['L22N16034', 'L24N8185', 'L23N1346', 'L3N12006']
    gate_dims=[2467, 2524, 3577, 2427, 3402, 2041, 957, 1943], down_dims=[3090, 822, 763, 1068, 2053, 1562, 1540, 3208]
  - axis=same_type, subset=['L27N16559', 'L27N7120']
    gate_dims=[2024, 1790, 1781, 1833, 1152, 39, 2140, 307], down_dims=[775, 2123, 624, 2524, 2348, 1860, 2906, 222]
  - axis=super_type, subset=['L2N12447', 'L26N12098', 'L27N2396', 'L27N17229']
    gate_dims=[2727, 711, 1806, 1803, 46, 1447, 1266, 3482], down_dims=[868, 2828, 945, 1287, 1044, 1458, 532, 3219]