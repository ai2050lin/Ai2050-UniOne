# Multi-hop Large-Sample Generalization Report

## 1) Global Baseline
- hop1_valid: 0.00540859
- hop1_invalid: 0.00002891
- hop1_selectivity: 0.00537968
- hop2_valid: 0.03938724
- hop2_invalid: 0.00006993
- hop2_selectivity: 0.03931731
- hop3_valid: 0.04318823
- hop3_invalid: 0.00000078
- hop3_selectivity: 0.04318745
- route_index: 0.03780777

## 2) Bootstrap CI (95%)
- hop1_selectivity: mean=0.00517265, std=0.00298865, ci95=[0.00075247, 0.01214014]
- hop2_selectivity: mean=0.03900529, std=0.00752397, ci95=[0.02576983, 0.05624988]
- hop3_selectivity: mean=0.04379134, std=0.01388140, ci95=[0.02410567, 0.07470749]
- route_index: mean=0.03861869, std=0.01431016, ci95=[0.01602952, 0.07143222]

## 3) Minimal Subset
- size: 4
- layers: {23: 1, 24: 1, 27: 2}
- progress: {'best_drop': 0.0034736561101051527, 'goal': 0.002778924888084122, 'achieved': 0.00326649995889456}

## 4) After-Ablation Global
- hop1_valid: 0.00533760
- hop1_invalid: 0.00002858
- hop1_selectivity: 0.00530902
- hop2_valid: 0.03797986
- hop2_invalid: 0.00006963
- hop2_selectivity: 0.03791023
- hop3_valid: 0.03992167
- hop3_invalid: 0.00000072
- hop3_selectivity: 0.03992095
- route_index: 0.03461193

## 5) Domain Route Drops (hop3_selectivity_drop)
- animal_mammal: hop3_drop=0.02856252, route_drop=0.02856254
- animal_bird: hop3_drop=0.00443238, route_drop=0.00443180
- animal_insect: hop3_drop=0.00396704, route_drop=0.00396738
- geo_city: hop3_drop=0.00338570, route_drop=0.00297347
- food_fruit: hop3_drop=0.00278183, route_drop=0.00278182
- geo_country: hop3_drop=0.00054682, route_drop=0.00001225
- artifact_vehicle: hop3_drop=0.00016870, route_drop=0.00016872
- animal_fish: hop3_drop=0.00006810, route_drop=0.00006776
- food_vegetable: hop3_drop=0.00000306, route_drop=0.00000309
- cosmos_body: hop3_drop=0.00000011, route_drop=0.00023828

## 6) Candidate Layer Concentration
- entropy=1.55711310, effective_layers=4.7451