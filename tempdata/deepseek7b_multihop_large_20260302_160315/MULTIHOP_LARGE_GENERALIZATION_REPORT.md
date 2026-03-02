# Multi-hop Large-Sample Generalization Report

## 1) Global Baseline
- hop1_valid: 0.00501052
- hop1_invalid: 0.00000353
- hop1_selectivity: 0.00500698
- hop2_valid: 0.03766620
- hop2_invalid: 0.00015798
- hop2_selectivity: 0.03750821
- hop3_valid: 0.05323563
- hop3_invalid: 0.00000117
- hop3_selectivity: 0.05323445
- route_index: 0.04822747

## 2) Bootstrap CI (95%)
- hop1_selectivity: mean=0.00512639, std=0.00313018, ci95=[0.00076998, 0.01195904]
- hop2_selectivity: mean=0.03814568, std=0.00802055, ci95=[0.02511714, 0.05535454]
- hop3_selectivity: mean=0.05161628, std=0.01415118, ci95=[0.02772272, 0.08292811]
- route_index: mean=0.04648989, std=0.01475967, ci95=[0.02046789, 0.07852395]

## 3) Minimal Subset
- size: 2
- layers: {24: 1, 27: 1}
- progress: {'best_drop': 0.0017821248886374375, 'goal': 0.00142569991090995, 'achieved': 0.0017012550962077974}

## 4) After-Ablation Global
- hop1_valid: 0.00496525
- hop1_invalid: 0.00000352
- hop1_selectivity: 0.00496173
- hop2_valid: 0.03635180
- hop2_invalid: 0.00015855
- hop2_selectivity: 0.03619325
- hop3_valid: 0.05153432
- hop3_invalid: 0.00000112
- hop3_selectivity: 0.05153320
- route_index: 0.04657147

## 5) Domain Route Drops (hop3_selectivity_drop)
- animal_mammal: hop3_drop=0.01227273, route_drop=0.01227273
- animal_insect: hop3_drop=0.00321800, route_drop=0.00321816
- animal_fish: hop3_drop=0.00172476, route_drop=0.00172477
- animal_bird: hop3_drop=0.00163160, route_drop=0.00163161
- geo_city: hop3_drop=0.00045238, route_drop=0.00016883
- geo_country: hop3_drop=0.00005313, route_drop=-0.00048552
- food_fruit: hop3_drop=0.00002263, route_drop=0.00002267
- cosmos_body: hop3_drop=0.00000000, route_drop=0.00036149
- food_vegetable: hop3_drop=-0.00000082, route_drop=-0.00000082
- artifact_tool: hop3_drop=-0.00011091, route_drop=-0.00011123

## 6) Candidate Layer Concentration
- entropy=1.55958116, effective_layers=4.7568