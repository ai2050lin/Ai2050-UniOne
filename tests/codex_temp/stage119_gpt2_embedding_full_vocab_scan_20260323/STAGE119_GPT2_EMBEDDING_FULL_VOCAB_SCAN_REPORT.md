# Stage119: GPT-2 词嵌入全词有效编码扫描

## 核心结果
- 词表总规模: 50257
- 干净词变体数: 46788
- 干净唯一词数: 30541
- 种子家族覆盖率: 0.8156
- 词类种子覆盖率: 0.8494
- 平均有效编码分数: 0.5500
- 锚点基线分数: 0.6320

## 三层解释
- 全局几何强度: 用范数、能量集中度和主成分残差衡量词向量是否只是落在公共方向上。
- 语义家族对齐: 用种子家族质心和子空间捕获率衡量词是否进入稳定语义簇。
- 词类结构层: 额外输出 noun（名词）/ verb（动词）/ adjective（形容词）/ adverb（副词）/ function（功能词）坐标，方便后续做统一数学压缩。
- 个案继承接口: apple（苹果）类分析后续可以直接读取该扫描表，不必重新做大范围清洗。

## 锚点词
- apple: band=meso, group=meso_fruit, score=0.7168, margin=0.1058, seeds=[apple:1.000, peach:0.533, lemon:0.518]
- fruit: band=meso, group=meso_fruit, score=0.5409, margin=0.0467, seeds=[mango:0.533, strawberry:0.511, apple:0.511]
- justice: band=macro, group=macro_system, score=0.6409, margin=0.0086, seeds=[justice:1.000, truth:0.373, freedom:0.362]
- red: band=micro, group=micro_color, score=0.6857, margin=0.3227, seeds=[red:1.000, blue:0.640, yellow:0.636]
- language: band=macro, group=macro_system, score=0.5609, margin=0.0356, seeds=[language:1.000, logic:0.379, mathematics:0.365]
- run: band=macro, group=macro_action, score=0.6467, margin=0.1387, seeds=[run:1.000, build:0.425, walk:0.387]

## 词类锚点
- apple: lexical_type=noun, type_score=0.5496, type_margin=0.1049, matches=[apple:1.000, peach:0.533, lemon:0.518]
- build: lexical_type=verb, type_score=0.5699, type_margin=0.1393, matches=[build:1.000, create:0.523, make:0.470]
- beautiful: lexical_type=adjective, type_score=0.6262, type_margin=0.1128, matches=[beautiful:1.000, ugly:0.529, huge:0.470]
- quickly: lexical_type=adverb, type_score=0.6260, type_margin=0.1741, matches=[quickly:1.000, slowly:0.654, soon:0.621]
- and: lexical_type=function, type_score=0.6999, type_margin=0.3181, matches=[and:1.000, in:0.690, or:0.685]

## 下一步用途
- 从 word_rows.csv 里挑出 apple（苹果）及其近邻簇，继续做 fruit（水果）家族偏移与子空间裂缝分析。
- 从 most_ambiguous_words 里筛出边界词，优先看哪些词天然跨 micro（微观）/ meso（中观）/ macro（宏观）三层。
- 从 lexical_type_counts 和词类锚点出发，继续做“词类投影是否对应统一动力系统中的不同观测叶层”这条数学主线。
