# Multi-hop Reasoning Route Report (A->B->C)

## 1) Baseline
- hop1_valid: 0.00004080
- hop1_invalid: 0.00000494
- hop1_selectivity: 0.00003587
- hop2_valid: 0.01849136
- hop2_invalid: 0.00004906
- hop2_selectivity: 0.01844229
- hop3_valid: 0.02203602
- hop3_invalid: 0.00778743
- hop3_selectivity: 0.01424859
- route_index: 0.01421272

## 2) Minimal Route-Cut Subset
- size: 1
- layers: {27: 1}
- progress: {'best_drop': 0.0069037968113339065, 'goal': 0.005523037449067125, 'achieved': 0.00594661226642712}

## 3) Metrics After Ablation
- hop1_valid: 0.00004029
- hop1_invalid: 0.00000490
- hop1_selectivity: 0.00003538
- hop2_valid: 0.01851706
- hop2_invalid: 0.00004899
- hop2_selectivity: 0.01846807
- hop3_valid: 0.01487683
- hop3_invalid: 0.00657485
- hop3_selectivity: 0.00830198
- route_index: 0.00826659

## 4) Subset Neurons
- L27N16936 util=0.00594637 drop_h3=0.00594661 drop_h1=0.00000049