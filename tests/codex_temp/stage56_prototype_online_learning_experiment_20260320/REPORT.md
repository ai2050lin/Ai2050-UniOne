# Stage56 原型网络与在线学习实验摘要

- main_judgment: 小型原型网络已经能形成一般路径、严格路径和判别门三层结构；在线注入后新知识能力会提升，但旧知识保真和严格门稳定性会同时受到压力。

## Before Injection
{
  "accuracy": 1.0,
  "general_norm_mean": 15.596066157023111,
  "strict_norm_mean": 7.817583243052165,
  "disc_mean": 0.9175214966138204,
  "novel_accuracy": 0.011764705882352941
}

## After Injection
{
  "accuracy": 0.8333333333333334,
  "general_norm_mean": 15.376869201660156,
  "strict_norm_mean": 6.471279462178548,
  "disc_mean": 0.8556161721547445,
  "novel_accuracy": 0.9176470588235294
}

## Deltas
{
  "base_accuracy_delta": -0.16666666666666663,
  "novel_accuracy_delta": 0.9058823529411765,
  "forgetting": 0.16666666666666663,
  "strict_gate_shift": -0.2992785573005676
}
