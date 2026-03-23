# Stage120: GPT-2 词类投影图谱

## 核心结果
- 词类投影图谱分数: 0.7913
- noun（名词）-> meso（中观）锚定率: 0.7943
- verb（动词）-> macro（宏观）锚定率: 0.9630
- adjective（形容词）-> micro（微观）锚定率: 0.5513
- function（功能词）-> macro（宏观）锚定率: 0.7290
- adverb（副词）桥接熵: 0.9187
- 最强桥接词类: adverb

## 结构解释
- 名词主要投到 meso（中观）对象层，说明对象家族仍是词嵌入里的主体块。
- 动词高度投到 macro（宏观）动作层，说明动作更接近系统级变换而不是静态对象。
- 形容词主要投到 micro（微观）属性层，但仍有一部分漂到 macro（宏观）抽象层，说明属性词里混有评价与抽象风格分量。
- 副词不是简单附属物，而是桥接型词类，它在 meso（中观）/ macro（宏观）之间分布更散。
- 功能词偏 macro（宏观），这支持“语言控制/路由层不等同于对象层”的判断。

## 词类行
- noun: dominant_band=meso (0.7943), entropy=0.5255, top_groups=[meso_tech:0.208, meso_celestial:0.146, macro_abstract:0.125]
- verb: dominant_band=macro (0.9630), entropy=0.1544, top_groups=[macro_action:0.880, macro_abstract:0.054, macro_system:0.029]
- adjective: dominant_band=micro (0.5513), entropy=0.9022, top_groups=[micro_taste:0.213, macro_abstract:0.202, micro_size:0.151]
- adverb: dominant_band=meso (0.4498), entropy=0.9187, top_groups=[meso_tech:0.272, macro_action:0.194, macro_abstract:0.191]
- function: dominant_band=macro (0.7290), entropy=0.7027, top_groups=[macro_action:0.305, macro_abstract:0.283, macro_system:0.140]

## 理论提示
- 词类不是简单语法标签，而更像统一编码系统中的不同投影叶层。
- noun（名词）/ verb（动词）/ adjective（形容词）/ function（功能词）之间已经出现稳定的层级偏置。
- adverb（副词）若持续表现为高熵桥接层，后续应重点看它与上下文门控、路由调制的关系。
