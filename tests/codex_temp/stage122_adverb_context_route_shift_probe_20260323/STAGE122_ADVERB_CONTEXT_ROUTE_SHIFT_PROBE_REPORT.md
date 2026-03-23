# Stage122: Adverb 上下文选路偏移探针

## 核心结果
- 样本对数量: 144
- adverb（副词）动词位选路偏移均值: 0.000273
- adjective（形容词）控制组动词位选路偏移均值: 0.000000
- 动词位优势: 0.000273
- 动词位峰值优势: 0.006106
- 修饰词注意力优势: 0.045526
- 正向案例比率: 0.5347
- 峰值正向案例比率: 1.0000
- 动态选路偏移分数: 0.6131

## 解读
- 如果副词组在动词位的 route（选路）偏移高于形容词控制组，说明副词并不只是普通修饰词。
- 如果修饰词注意力优势也为正，说明模型确实在动态处理中吸收了副词位置信号。

## 最强案例
- simply / change: peak_adv=0.013244, attn_adv=0.052627
- therefore / change: peak_adv=0.013021, attn_adv=0.042716
- finally / change: peak_adv=0.012729, attn_adv=0.048548
- simply / move: peak_adv=0.011237, attn_adv=0.056459
- finally / compare: peak_adv=0.011127, attn_adv=0.039918
- simply / build: peak_adv=0.010996, attn_adv=0.056016
- finally / move: peak_adv=0.010554, attn_adv=0.053443
- therefore / explain: peak_adv=0.010491, attn_adv=0.041023
- eventually / change: peak_adv=0.010394, attn_adv=0.048210
- simply / compare: peak_adv=0.010325, attn_adv=0.051901
- simply / change: peak_adv=0.010119, attn_adv=0.046793
- always / explain: peak_adv=0.009770, attn_adv=0.047782

## 理论提示
- 这不是最终的 q / b / g 动态闭合，但已经给出第一份“副词插入会改动上下文选路代理量”的前向证据。
- 下一步应把这一效应推进到层级定位与词位定位，看看偏移主要发生在早层、中层还是后层。
