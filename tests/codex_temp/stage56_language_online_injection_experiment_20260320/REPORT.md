# Stage56 语言原型网络在线注入实验

- main_judgment: 小型语言原型网络已经能在真实语料片段上形成一般路径、严格路径和判别门三层结构；在线注入后新知识预测能力会明显抬升，但基础语言保真、困惑度和严格门稳定性会同时承压。

## Before Injection
{
  "base_valid": {
    "loss": 4.737144390741984,
    "accuracy": 0.20512820512820512,
    "perplexity": 114.10789489746094,
    "general_norm_mean": 31.705740928649902,
    "strict_norm_mean": 34.296648025512695,
    "disc_mean": 0.9999310672283173,
    "hidden_mean_norm": 6.3901801109313965
  },
  "novel_valid": {
    "loss": 4.763176855140693,
    "accuracy": 0.27732610659439927,
    "perplexity": 117.1174087524414,
    "general_norm_mean": 34.521065606011284,
    "strict_norm_mean": 37.611810472276474,
    "disc_mean": 0.999942766295539,
    "hidden_mean_norm": 6.164852062861125
  }
}

## After Injection
{
  "base_valid": {
    "loss": 5.72345337500939,
    "accuracy": 0.20512820512820512,
    "perplexity": 305.9597473144531,
    "general_norm_mean": 43.47534942626953,
    "strict_norm_mean": 44.404212951660156,
    "disc_mean": 0.9993986189365387,
    "hidden_mean_norm": 6.437045574188232
  },
  "novel_valid": {
    "loss": 1.6477171500308512,
    "accuracy": 0.5691056910569106,
    "perplexity": 5.195106506347656,
    "general_norm_mean": 48.651617685953774,
    "strict_norm_mean": 49.86518330044217,
    "disc_mean": 0.9983879990047879,
    "hidden_mean_norm": 6.243307563993666
  }
}

## Deltas
{
  "base_accuracy_delta": 0.0,
  "novel_accuracy_delta": 0.2917795844625113,
  "base_perplexity_delta": 191.8518524169922,
  "novel_perplexity_delta": -111.92230224609375,
  "forgetting": 0.0,
  "strict_gate_shift": -0.0015547672907511023
}
