# 多跳推理链路测试报告 (A->B->C)

## 1) 基线
- hop1_valid: 0.71393949
- hop1_invalid: 0.00005586
- hop1_selectivity: 0.71388363
- hop2_valid: 0.14205316
- hop2_invalid: 0.00003079
- hop2_selectivity: 0.14202237
- hop3_valid: 0.03539443
- hop3_invalid: 0.01052803
- hop3_selectivity: 0.02486641
- route_index: -0.68901722

## 2) 最小链路切断子集
- size: 3
- layers: {26: 1, 27: 2}
- progress: {'best_drop': 0.01903735753213398, 'goal': 0.015229886025707184, 'achieved': 0.01693335667933349}

## 3) 消融后指标
- hop1_valid: 0.71262767
- hop1_invalid: 0.00005638
- hop1_selectivity: 0.71257129
- hop2_valid: 0.14114112
- hop2_invalid: 0.00003112
- hop2_selectivity: 0.14111000
- hop3_valid: 0.01259866
- hop3_invalid: 0.00466561
- hop3_selectivity: 0.00793305
- route_index: -0.70463824

## 4) 子集神经元
- L27N16936 util=0.00920083 drop_h3=0.00929091 drop_h1=0.00018015
- L27N6702 util=0.00690433 drop_h3=0.00693864 drop_h1=0.00006862
- L26N8166 util=0.00579719 drop_h3=0.00644279 drop_h1=0.00129120