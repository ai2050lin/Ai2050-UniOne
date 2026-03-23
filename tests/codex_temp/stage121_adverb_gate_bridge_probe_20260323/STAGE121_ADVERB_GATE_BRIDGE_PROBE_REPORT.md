# Stage121: Adverb 门控桥探针

## 核心结果
- 核心副词样本数: 430
- adverb（副词）门控桥分数: 0.4772
- adverb（副词）门控均值: 0.2649
- control（控制原型）均值: 0.3343
- content（内容原型）均值: 0.2477
- adverb（副词）中点位置: 0.1983
- adverb（副词）动作-功能平衡均值: 0.8954

## 解释
- 如果 adverb（副词）均值高于 noun（名词）/ adjective（形容词），但低于 verb（动词）/ function（功能词），就说明它更像桥而不是核心控制块。
- 如果动作-功能平衡值很高，说明副词不是只贴动作，也不是只贴功能词，而是同时沾到两边。

## Top Adverbs
- also: gate=0.4009, margin=0.1892, balance=0.7941, macro/macro_action
- actually: gate=0.3868, margin=0.1162, balance=0.8893, macro/macro_action
- therefore: gate=0.3860, margin=0.0612, balance=0.9893, macro/macro_action
- eventually: gate=0.3839, margin=0.0541, balance=0.9965, macro/macro_action
- thus: gate=0.3814, margin=0.0700, balance=0.9596, macro/macro_action
- simply: gate=0.3743, margin=0.0868, balance=0.9082, macro/macro_action
- always: gate=0.3742, margin=0.0877, balance=0.9062, macro/macro_action
- finally: gate=0.3718, margin=0.0573, balance=0.9560, macro/macro_action
- ultimately: gate=0.3692, margin=0.0373, balance=0.9856, macro/macro_action
- immediately: gate=0.3662, margin=0.0568, balance=0.9410, macro/macro_action
- directly: gate=0.3645, margin=0.0512, balance=0.9464, meso/meso_tech
- possibly: gate=0.3639, margin=0.0337, balance=0.9772, macro/macro_action

## 理论提示
- 副词若稳定站在 route/control（路由/控制）与 content（内容）之间，就可以成为 q / b / g 链的优先静态入口。
- 这还不是动态证明，但已经把“副词像桥”从印象推进成了可计算指标。
