# Stage56 Knowledge Multifield Coupling Report

- Case count: 24
- Regimes: {'bridge_compensated_mismatch': 2, 'conflict_locked_mismatch': 9, 'cooperative_multifield': 3, 'mixed_transitional': 10}
- Dominant fields: {'bridge_field': 1, 'conflict_field': 1, 'instance_field': 8, 'prototype_field': 5, 'route_mismatch_field': 9}
- Mean prototype field: 0.004943
- Mean instance field: 0.073826
- Mean bridge field: 0.002718
- Mean conflict field: 0.003071
- Mean route mismatch field: 0.084478

## Top Balance Cases
- glm_fruit / food / proto=food / inst=bread / regime=mixed_transitional / dominant=instance_field / P=-0.182064 / I=1.473691 / B=0.000000 / X=0.000000 / M=1.058316 / balance=0.508469
- deepseek_fruit / food / proto=food / inst=pizza / regime=mixed_transitional / dominant=instance_field / P=-0.000104 / I=0.314977 / B=0.000000 / X=0.000000 / M=0.000000 / balance=0.419568
- qwen_fruit / food / proto=food / inst=bread / regime=cooperative_multifield / dominant=instance_field / P=0.032896 / I=0.048167 / B=0.032312 / X=0.000000 / M=0.000000 / balance=0.160007
- glm_fruit / object / proto=object / inst=bowl / regime=mixed_transitional / dominant=instance_field / P=-0.045984 / I=0.084951 / B=0.030064 / X=0.000000 / M=0.000000 / balance=0.108575
- deepseek_real / animal / proto=animal / inst=rabbit / regime=mixed_transitional / dominant=route_mismatch_field / P=0.002771 / I=0.031091 / B=0.000000 / X=0.000000 / M=0.031964 / balance=0.044276
- qwen_real / animal / proto=animal / inst=rabbit / regime=mixed_transitional / dominant=prototype_field / P=-0.055673 / I=0.047777 / B=0.000000 / X=0.016208 / M=0.000000 / balance=0.041155
- glm_real / human / proto=human / inst=teacher / regime=bridge_compensated_mismatch / dominant=instance_field / P=0.005525 / I=0.030305 / B=0.000521 / X=0.000000 / M=0.022311 / balance=0.032241
- qwen_real / human / proto=human / inst=teacher / regime=cooperative_multifield / dominant=prototype_field / P=0.007862 / I=0.001904 / B=0.000787 / X=0.000000 / M=0.000000 / balance=0.013247
- glm_real / vehicle / proto=vehicle / inst=car / regime=mixed_transitional / dominant=route_mismatch_field / P=0.002327 / I=0.005131 / B=0.000000 / X=0.000000 / M=0.005614 / balance=0.006222
- deepseek_fruit / nature / proto=nature / inst=tree / regime=cooperative_multifield / dominant=bridge_field / P=0.000130 / I=0.001168 / B=0.001497 / X=0.000000 / M=0.000000 / balance=0.005493
- glm_real / tech / proto=tech / inst=software / regime=mixed_transitional / dominant=conflict_field / P=0.000241 / I=0.000052 / B=0.000000 / X=0.000636 / M=0.000000 / balance=0.002777
- deepseek_fruit / object / proto=object / inst=bowl / regime=conflict_locked_mismatch / dominant=route_mismatch_field / P=0.000558 / I=0.000754 / B=0.000000 / X=0.000326 / M=0.000819 / balance=0.000596
- glm_fruit / nature / proto=nature / inst=river / regime=conflict_locked_mismatch / dominant=route_mismatch_field / P=0.000468 / I=0.000468 / B=0.000000 / X=0.000291 / M=0.000527 / balance=0.000586
- qwen_real / tech / proto=tech / inst=database / regime=conflict_locked_mismatch / dominant=route_mismatch_field / P=0.000349 / I=0.000014 / B=0.000000 / X=0.000136 / M=0.000463 / balance=0.000166
- deepseek_real / vehicle / proto=vehicle / inst=car / regime=mixed_transitional / dominant=prototype_field / P=-0.003216 / I=0.002022 / B=0.000000 / X=0.000002 / M=0.000000 / balance=0.000089
- qwen_fruit / object / proto=object / inst=bowl / regime=mixed_transitional / dominant=prototype_field / P=-0.000143 / I=0.000008 / B=0.000000 / X=0.000000 / M=0.000000 / balance=0.000003
- qwen_fruit / nature / proto=nature / inst=tree / regime=bridge_compensated_mismatch / dominant=prototype_field / P=-0.000276 / I=0.000013 / B=0.000055 / X=0.000000 / M=0.000027 / balance=-0.000235
- qwen_fruit / fruit / proto=fruit / inst=apple / regime=conflict_locked_mismatch / dominant=route_mismatch_field / P=0.035830 / I=0.026291 / B=0.000000 / X=0.028743 / M=0.048220 / balance=-0.001485
- qwen_real / vehicle / proto=vehicle / inst=cart / regime=conflict_locked_mismatch / dominant=instance_field / P=-0.000262 / I=-0.002995 / B=0.000000 / X=0.000870 / M=0.001143 / balance=-0.005270
- deepseek_real / human / proto=human / inst=librarian / regime=conflict_locked_mismatch / dominant=route_mismatch_field / P=-0.000529 / I=-0.001851 / B=0.000000 / X=0.000003 / M=0.004518 / balance=-0.006902
