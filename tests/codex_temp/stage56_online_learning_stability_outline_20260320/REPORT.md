# Stage56 在线学习稳定性摘要

- main_judgment: 当前在线学习稳定性最主要的风险不在一般主核，而在严格选择结构；一旦选择不稳定过高，实时更新最容易造成严格层漂移和遗忘。

{
  "strict_confidence": 0.5065163159606211,
  "select_instability": 0.0593952853824368,
  "strict_negative_count": 4
}

{
  "online_update_budget": "Budget ~ strict_confidence - select_instability",
  "forgetting_risk": "Risk ~ strict_negative_count + select_instability",
  "safe_update_condition": "strict_confidence > select_instability and strict_negative_count <= 3"
}
