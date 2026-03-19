# Stage56 逻辑/句法微修正压缩摘要

- row_count: 72
- equation_text: logic_syntax_support = logic_core + logic_strictload + syntax_core + syntax_strictload; logic_syntax_interference = logic_structure_gain * syntax_structure_gain; logic_net_support = logic_core - logic_strictload; syntax_net_support = syntax_strictload - syntax_core
- main_judgment: 逻辑/句法微修正已开始压缩成支持项和干涉项，可以直接判断它们更适合并入正质量，还是保留为独立的闭式微修正。

## Stable Features
- logic_strictload_term: positive

## Fits
- target: union_joint_adv
  intercept: +0.271779
  logic_core_term: +0.000530
  logic_strictload_term: +0.000005
  syntax_core_term: -0.000614
  syntax_strictload_term: +0.000248
  logic_syntax_support_term: +0.000166
  logic_syntax_interference_term: +0.000001
  logic_net_support_term: +0.000529
  syntax_net_support_term: +0.000859
  logic_syntax_net_support_term: +0.001389
- target: union_synergy_joint
  intercept: +0.234384
  logic_core_term: +0.000187
  logic_strictload_term: +0.000224
  syntax_core_term: -0.000367
  syntax_strictload_term: +0.000152
  logic_syntax_support_term: +0.000184
  logic_syntax_interference_term: +0.000001
  logic_net_support_term: -0.000031
  syntax_net_support_term: +0.000519
  logic_syntax_net_support_term: +0.000487
- target: strict_positive_synergy
  intercept: +0.134742
  logic_core_term: -0.003665
  logic_strictload_term: +0.002146
  syntax_core_term: +0.001112
  syntax_strictload_term: -0.000302
  logic_syntax_support_term: -0.000620
  logic_syntax_interference_term: -0.000005
  logic_net_support_term: -0.005825
  syntax_net_support_term: -0.001384
  logic_syntax_net_support_term: -0.007209
