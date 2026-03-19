# Stage56 严格负载模块摘要

- row_count: 72
- equation_text: strict_module_base = strict_load; strict_module_logic = logic_strictload; strict_module_combined = strict_load + logic_strictload; strict_module_residual = strict_load - logic_strictload
- main_judgment: 严格负载已经被重写成基础负载、逻辑耦合、组合项和残差项，当前最关键的问题是严格闭包更依赖正向组合模块，还是依赖被扣除后的残差负担。

## Stable Features

## Fits
- target: union_joint_adv
  intercept: +1.041074
  strict_module_base_term: -0.225363
  strict_module_logic_term: -0.000483
  strict_module_combined_term: -0.223724
  strict_module_residual_term: -0.224484
- target: union_synergy_joint
  intercept: +0.418269
  strict_module_base_term: -0.100611
  strict_module_logic_term: -0.000216
  strict_module_combined_term: -0.099881
  strict_module_residual_term: -0.100219
- target: strict_positive_synergy
  intercept: -1.631274
  strict_module_base_term: +0.641020
  strict_module_logic_term: +0.002063
  strict_module_combined_term: +0.637061
  strict_module_residual_term: +0.637828
