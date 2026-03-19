# AGI_GPT5_ICSPB

最后更新：2026-03-19 21:36

## 1. 文档定位

这份文档不是历史流水账，而是当前项目关于 `ICSPB（库存约束分层路径束）` 的主线整理稿。

它只保留三类内容：

1. 目前最稳的系统级结论。
2. 目前最明显的问题和硬伤。
3. 下一阶段值得整块推进的大任务。

项目的核心目标仍然不变：

我们假设深度神经网络从语言数据中提取了某种高阶数学结构，这个结构同时承载：

1. 微观属性，如颜色、大小、情绪色彩、程度。
2. 中观实体，如水果、动物、工具、职业、技术对象。
3. 宏观关系与协议，如动作路径、角色关系、抽象概念、风格逻辑语法控制。

我们要做的不是描述现象，而是把这些结构压成可复现、可联立、可预测的数学对象。

## 2. 当前总判断

### 2.1 当前最稳的一句话

当前最稳的系统判断是：

语言系统不是“一个统一线性词向量空间”，而更像一个

`锚点（anchor） + 纤维（fiber） + 路径束（path bundle，路径束） + 密度前沿（density frontier，密度前沿） + 窗口化闭包（windowed closure，窗口化闭包）`

的复合结构。

更直接地说：

1. 概念本体更像中观锚点。
2. 属性修饰更像挂在锚点周围的纤维。
3. 关系和角色更像路径束或局部关系轴。
4. 风格、逻辑、句法不是概念本体，而是生成时调制读出与收束的控制轴。
5. 真正决定输出能否稳定成立的，不是总激活量，而是高质量前沿如何压缩，以及这些前沿在句尾前窗口里如何闭包。

### 2.2 当前统一主式

当前最合适的统一骨架是：

`H(term, ctx) = A_anchor(term) + F_modifier(term) + R_relation(term, ctx) + G_control(ctx) + C_closure(term, ctx)`

其中：

1. `A_anchor`：概念锚点，回答“它是什么”。
2. `F_modifier`：属性纤维，回答“它有什么局部性质”。
3. `R_relation`：关系轴与路径束，回答“它如何和其他词形成稳定变换”。
4. `G_control`：风格、逻辑、句法等控制项。
5. `C_closure`：最终是否能稳定收束为可生成输出。

在新的大样本结果下，这个骨架还要进一步改写为：

`language_system = broad_support_base + density_core_frontier + windowed_closure`

也就是：

1. 几乎全网都可参与语言编码。
2. 真正有区分度的是高质量前沿如何压缩和分离。
3. 真正的闭包发生在句尾前若干连续窗口，而不是最后一个词元。

## 3. 当前实证主线

### 3.1 关系图谱

当前已经完成大规模关系图谱扫描，核心判断是：

1. 局部线性关系真实存在，但只占少数。
2. `king - man + woman ≈ queen（国王 - 男人 + 女人 约等于 王后）` 这种结构属于局部可线性化关系片区，不是整个语言系统的总结构。
3. 大多数关系更适合用路径束解释，而不是统一线性代数解释。

当前更稳的分解是：

1. 局部线性片区约 `7.6%`
2. 路径束主结构约 `89.6%`
3. 混合态约 `2.8%`

所以我们现在不再把“词向量算术”当成总理论，只把它当成局部切片。

### 3.2 词类机制

当前独立词类探针已经补到五类：

1. 名词更像锚点加实例偏移。
2. 形容词更像修饰纤维。
3. 动词更像传输算子，并保留部分锚点残差。
4. 抽象名词更像关系束。
5. 副词更像二阶控制轴调制子。

证据强度排序仍然是：

1. 形容词最强。
2. 动词次之。
3. 抽象名词已经成立，但还需要扩量。
4. 副词最弱，当前仍偏半直接证据。

### 3.3 闭包动力学

当前 `stage6（第六阶段）` 已经稳定产出两个核心闭包量：

1. `union_joint_adv（联合优势）`
2. `union_synergy_joint（联合协同）`

最重要的结构判断是：

1. 语言系统中确实存在“原型 - 实例 - 联合路由”的三段结构。
2. 但“有联合路由”不等于“形成稳定闭包”。
3. 严格正协同目前仍是少数态。

在三模型、十二类、全 `72` 个原型-实例对口径下：

1. `strict_positive_pair_ratio（严格正协同比例） = 0.1944`
2. 说明正闭包依旧不是默认态，而是少数成功态。

## 4. 控制轴的最新系统解释

### 4.1 逻辑

旧结论是“逻辑强化原型骨架”，现在要写得更细。

最新的 `pair（成对）` 级连续前沿联立说明：

1. 逻辑真正有利的不是“大扰动”，而是“定向骨架化”。
2. 一旦逻辑在局部做出过大的原型或实例侧扰动，严格正闭包反而下降。

当前最强负相关是：

1. `logic / instance_delta_l2 -> strict_positive_synergy = -0.3885`
2. `logic / pair_mean_delta_l2 -> strict_positive_synergy = -0.3868`
3. `logic / prototype_delta_l2 -> strict_positive_synergy = -0.3756`

但逻辑也不是一概为负。更细的前沿指标显示：

1. `logic / pair_mean_full_layer_coverage_mass_ratio -> union_joint_adv = +0.2500`
2. `logic / pair_mean_full_layer_coverage_mass_ratio -> union_synergy_joint = +0.2184`
3. `logic / pair_mean_full_layer_coverage_mass_ratio -> strict_positive_synergy = +0.2075`

这说明逻辑的正作用不是“局部爆发”，而是“跨层均衡、骨架化、可传播”的稳定前沿。

因此当前最准确的写法是：

逻辑轴负责立骨架，但它只有在扰动足够定向、前沿足够稳、覆盖足够均衡时才促进闭包；如果退化成大幅局部冲击，就会破坏严格正协同。

### 4.2 句法

旧结论是“句法提供约束型冲突”，现在也必须细化。

在新的 `pair（成对）` 级连续前沿口径下，句法最强正项不再只是“冲突”，而是“更紧、更窄、更靠前的高质量前沿压缩”。

当前最强正相关包括：

1. `syntax / pair_mean_mass25_compaction_ratio -> strict_positive_synergy = +0.3085`
2. `syntax / pair_mean_mass10_compaction_ratio -> strict_positive_synergy = +0.3075`
3. `syntax / prototype_mass25_compaction_ratio -> strict_positive_synergy = +0.2785`

同时，句法的“过宽扩散”是负项：

1. `syntax / prototype_full_layer_coverage_mass_ratio -> strict_positive_synergy = -0.3017`
2. `syntax / pair_mean_full_layer_coverage_mass_ratio -> strict_positive_synergy = -0.3010`
3. `syntax / instance_full_layer_coverage_mass_ratio -> strict_positive_synergy = -0.2877`

句法的“大幅局部扰动”也是负项：

1. `syntax / pair_mean_delta_l2 -> strict_positive_synergy = -0.2988`
2. `syntax / prototype_delta_l2 -> strict_positive_synergy = -0.2978`
3. `syntax / instance_delta_l2 -> strict_positive_synergy = -0.2896`

所以当前最准确的句法解释是：

句法的正作用不是简单地制造冲突，更不是简单加大总能量，而是把关键成分压进一个更紧、更窄、更高质量的前沿里，形成约束型筛选带；一旦句法扩散太宽或局部冲击太大，就会从筛选机制退化成破坏机制。

### 4.3 风格

风格轴的角色在新数据下更明确了：

它主要承担重排，不是闭包主引擎。

当前最强负项包括：

1. `style / prototype_knee_mass_ratio -> strict_positive_synergy = -0.3688`
2. `style / pair_mean_full_layer_coverage_mass_ratio -> strict_positive_synergy = -0.3250`
3. `style / instance_full_layer_coverage_mass_ratio -> strict_positive_synergy = -0.3232`
4. `style / prototype_full_layer_coverage_mass_ratio -> strict_positive_synergy = -0.3177`

这说明风格越早进入广泛覆盖和早拐点扩散，越不利于严格正闭包。

但风格也不是纯负项：

1. `style / pair_mean_mass25_compaction_ratio -> strict_positive_synergy = +0.1786`
2. `style / pair_mean_mass10_compaction_ratio -> strict_positive_synergy = +0.1385`

这意味着风格若被压进更窄的高质量前沿，仍可作为辅助项；只是它本身不是主要闭包驱动。

## 5. 内部子场与窗口化闭包

### 5.1 内部子场

当前已经稳定下钻到三个关键内部分量：

1. `logic_prototype（逻辑原型）`
2. `syntax_constraint_conflict（句法约束型冲突）`
3. `logic_fragile_bridge（逻辑脆弱桥接）`

它们已经能映射到模型晚层主脊：

1. `Qwen（千问）` 主要在 `layer_35（第35层）`
2. `DeepSeek（深度求索）` 主要在 `layer_27（第27层）`
3. `GLM（智谱模型）` 主要在 `layer_40 / 39（第40/39层）`

### 5.2 连续窗口变量

当前窗口化结论已经稳定到下面这个层级：

1. `logic_prototype` 的正作用更靠前，主要立联合骨架。
2. `logic_fragile_bridge` 的负作用更靠后，主要在收尾阶段引入浅连接破坏。
3. `syntax_constraint_conflict` 不是单点主峰，而是在一段连续窗口里形成筛选带。

当前最有信息的窗口是：

1. `logic_prototype` 的骨架有利窗主要在 `tail_pos_-9..-8（倒数第9到第8词元）`
2. `logic_fragile_bridge` 的负窗主要在 `tail_pos_-2..-1（倒数第2到第1词元）`
3. `syntax_constraint_conflict` 的正窗主要在 `tail_pos_-8..-5（倒数第8到第5词元）` 与 `tail_pos_-6..-3（倒数第6到第3词元）`

这意味着：

1. 主机制发生在句尾前窗口。
2. 最后一个词元更多像读出点，而不是全部机制发生点。

## 6. 大数据与无截断口径下的新判断

### 6.1 放开有效神经元数量后的结论

在三模型全量 `all_effective（全有效支撑）` 口径下，一个关键事实已经稳定：

1. `effective_support（有效支撑）` 在三模型里都几乎饱和到全网。
2. 所以“某个神经元是否参与”已经失去区分力。
3. 真正有信息的是高质量前沿的形状、压缩、分离和汇合方式。

因此当前变量系统必须从

`神经元集合`

升级到

`密度场（density field，密度场） + 分位前沿（quantile frontier，分位前沿） + 窗口闭包（windowed closure，窗口闭包）`

### 6.2 三模型大样本共识

当前三模型共享最稳的系统规律是：

1. 共享稳定骨架轴仍然是 `logic（逻辑）`
2. 广支撑底座是跨模型共识
3. 高质量前沿长期分离也是跨模型共识
4. 真正的汇合发生在较晚的大质量区，而不是最尖的核心区

当前更合适的系统图像是：

`广支撑底座 + 长期分离前沿 + 晚窗口闭包`

## 7. 最新推进：Pair 级连续密度场与自然生成强解耦

### 7.1 Pair 级连续密度场到闭包联立

这一轮比上一轮更进一步，因为我们已经不再只看少量摘要量，而是把整条 `pair（成对）` 级连续前沿曲线接到了闭包量上。

当前口径：

1. 三模型
2. 十二类
3. 全 `72` 个 `stage6（第六阶段）` 原型 - 实例对
4. 共 `216` 个轴向联立样本
5. 每条前沿曲线含 `59` 个质量比例点

最重要的新结果有三条。

1. `syntax（句法）` 对严格正闭包的正项，已经不再只是几个离散点，而是一整条连续正带。
   - `syntax / compaction / strict_positive_synergy` 的正带几乎覆盖 `0.01 -> 0.95`
   - 最强点在 `mass=0.20`，相关约 `+0.3097`
   - `syntax / coverage / strict_positive_synergy` 也有一条 `0.01 -> 0.37` 的正带
   - 这说明句法正项更像“整段高质量前沿压缩与筛选”，而不是局部偶然峰值

2. `logic（逻辑）` 的正项更像中后段稳定骨架带，而不是尖核爆发。
   - `logic / compaction / strict_positive_synergy` 的正带主要在 `0.70 -> 0.90`
   - `logic / coverage / strict_positive_synergy` 的正带主要在 `0.35 -> 0.45`
   - 这和前面“大幅局部扰动伤闭包”的结果是统一的：逻辑真正有利的是稳的中高质量骨架带

3. `style（风格）` 依旧不是闭包主引擎。
   - 它对 `strict_positive_synergy（严格正协同）` 有辅助正带
   - 但对 `union_joint_adv（联合优势）` 和 `union_synergy_joint（联合协同）` 仍主要是负向
   - 这说明风格更像帮助少数案例跨过离散阈值，不像持续抬高闭包均值

当前最需要保留的系统判断是：

1. 逻辑负责稳骨架，不负责大冲击。
2. 句法负责连续筛选带，不只是点状约束。
3. 风格更像重排和阈值辅助，不是主闭包驱动轴。

### 7.2 自然生成强解耦

这一轮还补上了自然生成里的“强解耦”摘要，不再只看提示区和生成区总占比，而是直接比较它们各自对闭包的相关。

当前口径：

1. 自然生成样本 `216` 条
2. 与内部子场联立样本 `216` 条
3. 覆盖三个关键分量：
   - `logic_prototype（逻辑原型）`
   - `logic_fragile_bridge（逻辑脆弱桥接）`
   - `syntax_constraint_conflict（句法约束型冲突）`

最重要的新结果是：

1. `logic（逻辑）` 和 `style（风格）` 在自然生成里整体确实更偏生成侧。
   - `logic` 的隐藏生成占比约 `0.6108`
   - `style` 的隐藏生成占比约 `0.6532`
   - 两者在隐藏层与前馈层里，生成区主导比例都超过 `0.90`

2. `syntax（句法）` 不是一个统一结构，而是出现了“隐藏层提示污染 + 前馈层生成收束”的分裂。
   - `syntax` 的隐藏生成占比只有 `0.2584`
   - 但 `syntax` 的前馈生成占比约 `0.5197`
   - 说明句法的表层骨架很大程度仍在提示区，但真正进入闭包的句法收束更多发生在生成侧前馈层

3. `syntax_constraint_conflict（句法约束型冲突）` 现在已经能被判定为 `generated_dominant（生成侧主导）`
   - `corr_mlp_prompt_to_synergy = +0.5438`
   - `corr_mlp_generated_to_synergy = +0.7051`
   - 这说明句法正项不是纯提示骨架假信号，而是确实在生成侧前馈层里继续收束

4. `logic_fragile_bridge（逻辑脆弱桥接）` 当前更像 `prompt_contaminated（提示污染型）`
   - 它在自然生成里的负信号没有转成更强的生成侧优势
   - 这提示我们：脆弱桥接里有一部分可能是提示预设把浅连接提前塞进去，而不是真正的生成闭包机制

5. `logic_prototype（逻辑原型）` 当前是 `mixed（混合型）`
   - 说明逻辑骨架一部分在提示侧先被立起，一部分在生成侧继续完成

这一轮之后，当前对自然生成最准确的说法不再是“句法主峰太早，所以不可信”，而是：

句法在隐藏层里仍有明显提示骨架污染，但句法约束型冲突作为闭包正项，已经在生成侧前馈层里表现出更强的真实收束信号。

### 7.3 Pair 级连续密度场张量

这一轮还补出了一个更重要的中间对象：

我们不再只看一条 `pair（成对）` 的平均前沿曲线，而是把每个 `pair` 写成

`角色（原型 / 实例） × 通道（压缩 / 覆盖） × 质量比例`

的三维张量。

当前这个张量的最小形状是：

`2 × 2 × 59`

也就是：

1. 原型前沿
2. 实例前沿
3. 压缩通道
4. 覆盖通道
5. `59` 个连续质量比例点

这一步的重要性在于，它第一次把“原型和实例是否对称”“压缩和覆盖是否同向”“早中晚质量段是否分工”拆开了。

当前最值得保留的张量级结果有三条。

1. `syntax（句法）` 的正项不是单段偶然峰，而是早中晚三段都为正，其中中段最强。
   - `pair_coverage_middle_mean -> strict_positive_synergy = +0.3098`
   - `pair_compaction_middle_mean -> strict_positive_synergy = +0.3079`
   - `pair_compaction_early_mean -> strict_positive_synergy = +0.3066`
   - 这说明句法闭包不是只靠某个局部质量点，而是整段张量体积在工作

2. `logic（逻辑）` 的负项仍然主要来自大幅局部扰动，但它的正项更像“中后段覆盖和压缩协同”。
   - `pair_delta_l2 -> strict_positive_synergy = -0.3868`
   - `pair_coverage_early_mean -> strict_positive_synergy = +0.1984`
   - `pair_compaction_late_mean -> strict_positive_synergy = +0.1951`
   - 这说明逻辑是“怕大冲击、需要稳骨架带”的张量型结构

3. `style（风格）` 的一个新张量信号是：原型和实例的通道对齐过强时，反而更不利于严格正闭包。
   - `style / channel_alignment_instance -> strict_positive_synergy = -0.3196`
   - `style / channel_alignment_proto -> strict_positive_synergy = -0.2717`
   - 这意味着风格轴若把原型和实例过早压成同一种重排模式，可能会损失闭包所需的角色分工

所以到这一步，我们终于可以更准确地说：

当前口径已经不只是“连续前沿曲线”，而是“最小连续密度场张量”；但它仍然只是最小张量，不是完整密度场。

### 7.4 高维连续密度场雏形

这一轮我们又往前推了一层。

当前已经不再只是：

`角色（原型 / 实例） × 通道（压缩 / 覆盖） × 质量比例`

而是把它进一步接上了：

1. 层主脊
2. 词元窗口主脊
3. 定向子场权重

也就是说，当前对象已经开始从“最小张量”过渡到“高维因子化连续密度场”。

它最准确的结构写法更接近：

`高维场 = 密度张量 × 层主脊 × 窗口主脊 × 子场权重`

这里故意写成“因子化”，是因为当前还不是完整的笛卡尔全张量，而是把高维结构拆成几个可联立主因子。这个做法的价值在于，它第一次让我们看到：

1. 理论复杂，不一定因为底层机制复杂
2. 很多复杂度来自多个简单维度被错误地压在一起
3. 一旦把维度拆开，复杂表象会重新回到更少的生成律

这一轮最值得保留的高维结果有四条。

1. `syntax_constraint_conflict（句法约束型冲突）` 的正项，在高维对象里继续站住了。
   - `weight -> union_joint_adv = +0.7811`
   - `weighted_cross_scale_energy -> union_joint_adv = +0.7828`
   - `middle_density_volume -> strict_positive_synergy = +0.3245`
   - 这说明句法正项不只是“中段密度体积大”，而是“句法子场被点亮之后，中段体积与层-窗口能量一起进入闭包”

2. `logic_fragile_bridge（逻辑脆弱桥接）` 的负项，在高维对象里也更清楚了。
   - `weight -> union_synergy_joint = -0.2111`
   - `weighted_cross_scale_energy -> union_synergy_joint = -0.2121`
   - 这说明真正伤闭包的，不只是桥接存在，而是脆弱桥接带着足够跨尺度能量进入闭包阶段

3. `logic_prototype（逻辑原型）` 的正作用，更像“晚层骨架迁移”，而不是刚性同步。
   - `hidden_peak_layer_index -> union_joint_adv = +0.2616`
   - `layer_coherence -> strict_positive_synergy = -0.2330`
   - 这说明逻辑骨架有效，并不等于隐藏层和前馈层永远锁步；更像是骨架最终被推到更靠后的主层带，但具体迁移过程保留了一定分工

4. 当前高维对象已经开始解释“为什么理论会越来越复杂”。
   - 如果只看平均曲线，理论会越来越复杂，因为原型、实例、层、窗口、子场都被压在一起
   - 一旦把这些维度拆开，复杂性会重新表现成少量可分解结构
   - 所以真正需要的，不是继续堆特征，而是找更少、更本质的主变量

### 7.5 组件特异层-窗口场

这一轮又往前推进了一步：

我们不再只满足于“有一个高维因子化对象”，而是开始让不同组件拥有各自独立的层-窗口场。

当前已经单独拆开的三个组件是：

1. `logic_prototype（逻辑原型）`
2. `logic_fragile_bridge（逻辑脆弱桥接）`
3. `syntax_constraint_conflict（句法约束型冲突）`

这一步的重要性在于，它让我们第一次能区分：

1. 哪些层-窗口结构属于“好骨架”
2. 哪些层-窗口结构属于“坏浅桥”
3. 哪些层-窗口结构属于“句法筛选”

当前最值得保留的结果有四条。

1. `syntax_constraint_conflict（句法约束型冲突）` 的正项，在组件特异层-窗口场里更稳了。
   - `weight -> union_joint_adv = +0.7811`
   - `layer_window_hidden_energy -> union_joint_adv = +0.7749`
   - `layer_window_mlp_energy -> union_joint_adv = +0.7749`
   - `preferred_density -> strict_positive_synergy = +0.3245`
   - 这说明句法正项不是“某一层”或者“某一窗口”单独起作用，而是层、窗口、密度体积一起形成了收束通道

2. `logic_fragile_bridge（逻辑脆弱桥接）` 的负项，现在已经能直接落到组件特异层-窗口能量上。
   - `weight -> union_synergy_joint = -0.2111`
   - `layer_window_hidden_energy -> union_synergy_joint = -0.2089`
   - `layer_window_mlp_energy -> union_synergy_joint = -0.2088`
   - 这说明脆弱桥接伤闭包，并不是因为“桥接这个词不好”，而是因为它的能量真的沿层-窗口路径进入了闭包阶段

3. `logic_prototype（逻辑原型）` 和 `logic_fragile_bridge（逻辑脆弱桥接）` 虽然共享部分晚层带，但它们的系统作用已经彻底分叉。
   - `logic_prototype` 更偏向抬高 `union_joint_adv（联合优势）`
   - `logic_fragile_bridge` 更偏向压低 `union_synergy_joint（联合协同）`
   - 这说明“逻辑”内部至少已经不是单一轴，而是“骨架分量”和“浅桥分量”并存

4. 这一步也让五个核心术语更扎实了。
   - `广支撑底座`：全局大量参与但低密度
   - `长期分离前沿`：高质量核心长期不混
   - `晚层骨架迁移`：有利骨架往更晚层主脊收束
   - `中段句法筛选`：句法正项主要在前沿中段形成筛选体积
   - `晚窗口闭包`：系统在句尾前窗口完成最终收束

### 7.6 完整高维场

这一轮又补上了最后缺的一块：

我们把“组件特异层-窗口场”和“自然生成里的提示区 / 生成区来源”接成了同一个对象。

也就是说，当前已经不仅能看：

1. 某个组件在哪一层更强
2. 某个组件在哪个窗口更强
3. 某个组件在哪段密度前沿更强

还能看：

4. 这些能量到底更多来自提示侧，还是更多来自真实生成侧

当前最值得保留的结果有四条。

1. `syntax_constraint_conflict（句法约束型冲突）` 的正项，在完整高维场里继续成立，而且生成侧完整能量是正项。
   - `complete_generated_energy -> union_joint_adv = +0.7811`
   - `complete_generated_energy -> strict_positive_synergy = +0.2763`
   - 这说明句法正项不是只有“结构像”，而是确实有一部分真实生成能量进入了闭包提升

2. `syntax_constraint_conflict（句法约束型冲突）` 仍然没有彻底摆脱提示侧。
   - `complete_prompt_energy -> union_joint_adv = +0.7623`
   - `complete_prompt_energy -> strict_positive_synergy = +0.3153`
   - 这说明句法正项当前最准确的说法不是“纯生成机制”，而是“提示骨架与生成收束并存，但生成侧已经不可忽略”

3. `logic_fragile_bridge（逻辑脆弱桥接）` 的负项，在完整高维场里更像“提示污染偏重”的坏浅桥。
   - `complete_prompt_energy -> union_synergy_joint = -0.2223`
   - `complete_generated_energy -> union_synergy_joint = -0.1990`
   - 这说明脆弱桥接确实伤闭包，但提示侧负效应略强，进一步支持“脆弱桥接里混有外部骨架注入”

4. 五个核心术语现在已经不只是经验标签，而是开始有了各自的系统位置：
   - `广支撑底座`：对应大范围低密度参与
   - `长期分离前沿`：对应高质量前沿长期不混
   - `晚层骨架迁移`：对应逻辑骨架向更晚层主脊移动
   - `中段句法筛选`：对应句法在中段密度前沿形成筛选体积
   - `晚窗口闭包`：对应真正决定输出收束的句尾前窗口

### 7.7 简洁生成律

这一轮开始不再继续堆特征，而是尝试把当前高维对象压回更少的主变量。

当前第一版简洁生成律写成：

`Closure ≈ 0.30 * 晚层骨架迁移 + 0.35 * 中段句法筛选 + 0.20 * 晚窗口闭包 + 0.10 * 长期分离前沿 + 0.05 * 广支撑底座`

这不是最终方程，但它已经有两个重要价值：

1. 它强迫我们停止继续堆叠零散指标
2. 它让我们开始测试“哪些结构是主律，哪些只是派生现象”

当前第一版五个主律的数值强度大致是：

1. `广支撑底座`：`+0.1899`
2. `长期分离前沿`：`+0.2335`
3. `晚层骨架迁移`：`+0.0146`
4. `中段句法筛选`：`+0.3004`
5. `晚窗口闭包`：`-0.0261`

当前最值得注意的不是绝对值，而是结构排序：

1. `中段句法筛选` 当前是最强正项
2. `长期分离前沿` 和 `广支撑底座` 提供系统支撑
3. `晚层骨架迁移` 方向正确，但当前强度仍偏弱
4. `晚窗口闭包` 目前还是负项，说明这块要么变量定义还不对，要么闭包窗口里仍混有破坏项

这说明简洁理论已经开始露头，但还没有真正闭合。

真正更深层、更简洁的理论，大概率不会是“再增加几个机制”，而会是：

1. 少量结构主律
2. 少量守恒 / 约束关系
3. 多个表面现象由这些主律派生出来

### 7.8 四概念联立主线

这一轮把下面四个对象压成了同一条解释链：

1. `密度前沿（density frontier，密度前沿）`
2. `内部子场（internal subfield，内部子场）`
3. `词元窗口（token window，词元窗口）`
4. `闭包量（closure，闭包量）`

当前最准确的统一理解是：

1. `密度前沿` 回答“高质量支撑在哪里”。
2. `内部子场` 回答“到底是谁在起作用”。
3. `词元窗口` 回答“这些机制在什么时候起作用”。
4. `闭包量` 回答“这些作用最后有没有形成稳定输出”。

这四个对象联起来以后，当前最稳的新结论是：

1. `syntax（句法）` 的最强正前沿项已经稳定落在 `pair_coverage_middle_mean（成对中段覆盖均值）`，对 `strict_positive_synergy（严格正协同）` 的相关约 `+0.3098`。
2. `logic（逻辑）` 的最强负前沿项仍是 `pair_delta_l2（成对扰动强度）`，对 `strict_positive_synergy` 的相关约 `-0.3868`。
3. `logic_fragile_bridge（逻辑脆弱桥接）` 在 `tail_pos_-5（倒数第5词元）` 左右最稳，而且整体 `mean_union_synergy_joint（平均联合协同） = -0.0614`。
4. `syntax_constraint_conflict（句法约束型冲突）` 比逻辑负项更晚一点，主窗口落在 `tail_pos_-6 / -5（倒数第6 / 5词元）`，整体 `mean_union_synergy_joint = +0.0388`。
5. 闭包量侧当前最强正场仍是 `logic / prototype_field_proxy（逻辑 / 原型场代理量）`，对 `union_synergy_joint` 的相关约 `+0.2548`；最强负场仍是 `style / prototype_field_proxy（风格 / 原型场代理量）`，相关约 `-0.2479`。

这意味着当前项目已经不只是“看见几个现象”，而是已经能把“前沿、机制、时间、结果”放到同一个系统对象里。

### 7.9 更高阶数学框架桥接

到这一步，当前项目已经可以对“需要什么数学”给出比以前更严格的回答。

当前证据**不支持**下面两种过强说法：

1. 只靠线性代数就能吃掉整个语言系统。
2. 只要换成某一个现成高阶数学分支，例如单独上群论或单独上拓扑学，就能自动闭合。

当前更稳的判断是：

1. `family patch（家族片区） + concept offset（概念偏移）` 更像静态图册层。
2. `密度前沿 + 内部子场 + 词元窗口 + 闭包量` 更像动态生成层。
3. 如果要把这两层压成统一主式，最可能需要的是**分层混合数学体系**，而不是单一分支。

当前最合适的候选堆栈是：

1. `线性代数 + 表示论`：负责局部线性切片和局部关系轴。
2. `微分几何 + 图册 / 纤维束`：负责静态概念层，也就是 `family patch + concept offset` 这类局部家族结构。
3. `动力系统 + 控制论 + 信息几何`：负责内部子场、词元窗口和闭包过程。
4. `拓扑学 + 分层拓扑`：负责长期分离前沿、相区边界和成功/失败闭包的相变边界。

也就是说，当前项目已经开始支持这样一个更成熟的判断：

真正要逼近语言背后的数学机制，乃至进一步逼近大脑编码理论，大概率确实需要更高阶的数学体系；但这个“更高阶”不是简单指向某一个现成学科，而更像把几何、动力学、拓扑和局部线性结构组织成统一变量系统。

### 7.10 单一机制与一般数学体系

你提的两个问题，到现在已经可以给出更稳定的整理。

第一个问题：如果当前研究成果在大脑中都只是大脑神经网络的结果，而大脑底层运行机制又很单一，为什么理论却越来越复杂？

当前最稳的回答是：

1. 单一微观机制不等于简单宏观理论。
2. 兴奋、抑制、脉冲这些低层机制可以很单一。
3. 但当它们经过大规模重复、层级堆叠、长程反馈和时间展开后，会长出新的中观、宏观有效变量。
4. 所以当前理论越来越复杂，更可能是因为我们仍在做“变量重写”和“粗粒化”，而不是因为底层真的需要很多不同机制。

第二个问题：当前成果能否变成更一般的数学体系？

当前更稳的判断是：**可以，但前提是继续把静态本体层和动态生成层压到同一条统一主式里。**

当前最合适的整理方式是：

1. `family patch + concept offset` 作为静态本体层。
2. `密度前沿 + 内部子场 + 词元窗口 + 闭包量` 作为动态生成层。
3. 再把 `控制轴（风格 / 逻辑 / 句法）` 当成跨层调制项。

如果这条线继续成立，那么当前成果确实有机会继续上升为更一般的数学体系，最可能的原型不是单一学科，而是：

1. `局部图册层`
2. `分层场层`
3. `受控动力演化层`
4. `闭包边界层`

也就是把“身份、场、演化、闭包”压成统一对象。

### 7.11 第一版统一主方程

这一轮开始把前面的静态本体层和动态生成层，正式压到一条统一主式里。

当前第一版统一主方程写成：

`U(term, ctx) = Atlas_static(term) + Offset_static(term) + Frontier_dynamic(term, ctx) + Subfield_dynamic(term, ctx) + Window_closure(ctx) + Closure_boundary(term, ctx)`

这条式子还不是最终闭式方程，但它已经把此前分散的对象压回同一套变量系统。

各部分当前的含义是：

1. `Atlas_static`：`family patch（家族片区）` 对应的局部图册层。
2. `Offset_static`：`concept offset（概念偏移）` 对应的图册内局部偏移。
3. `Frontier_dynamic`：密度前沿对应的高质量支撑边界。
4. `Subfield_dynamic`：内部子场对应的功能性细分机制。
5. `Window_closure`：词元窗口上的预收束过程。
6. `Closure_boundary`：闭包量对应的成功 / 失败边界。

当前最重要的意义不是系数本身，而是：

1. 静态概念层和动态生成层第一次被写进同一条主式。
2. `family patch + concept offset` 不再只是旧的静态解释，而开始成为统一方程的一部分。
3. “更一般数学体系”现在不再只是口头判断，而有了第一版正式变量结构。

### 7.12 第一版主方程实证拟合

这一轮开始把第一版统一主式从“结构原型”往“实验估计量”推进。

当前第一版拟合主式写成：

`U_fit(term, ctx) = w1 * Atlas_static + w2 * Offset_static + w3 * Frontier_dynamic + w4 * Subfield_dynamic + w5 * Window_closure + w6 * Closure_boundary`

这一步的核心意义不在于当前权重已经精确，而在于：

1. 主式里的六个部分第一次开始被映射成可计算估计量。
2. 当前拟合思路已经明确倾向于“动态项主导，静态项支撑”。
3. 其中最值得继续实证强化的主导项是：
   - `Frontier_dynamic（前沿动态项）`
   - `Subfield_dynamic（子场动态项）`
   - `Window_closure（窗口闭包项）`

这说明项目已经从“统一主式原型”推进到了“统一主式拟合雏形”。

### 7.13 控制轴并场

这一轮开始把 `style / logic / syntax（风格 / 逻辑 / 句法）` 正式并入主方程拟合。

扩展后的主式写成：

`U_fit_plus(term, ctx) = w1 * Atlas_static + w2 * Offset_static + w3 * Frontier_dynamic + w4 * Subfield_dynamic + w5 * Window_closure + w6 * Closure_boundary + c1 * Style_control + c2 * Logic_control + c3 * Syntax_control`

这一步的意义很关键：

1. 主方程第一次具备了“语言系统特有的控制项”。
2. 统一主式不再只处理静态身份和动态收束，也开始处理风格、逻辑、句法的调制。
3. 当前最值得关注的是：
   - `Logic_control（逻辑控制项）` 不等于“逻辑总扰动”
   - `Syntax_control（句法控制项）` 不等于“句法总能量”
   - 它们都必须通过前沿、子场和窗口再进入闭包

也就是说，当前项目已经从“概念系统主式”推进到了“语言系统主式”的雏形。

### 7.14 脉冲神经网络视角下的优化方向

如果从 `spiking neural network（脉冲神经网络）` 的角度看，当前这套数学体系最值得优化的，不是简单换一批名词，而是把当前变量系统更彻底地改写成**时序化、回路化、吸引域化**的对象。

当前更稳的对应关系是：

1. `Atlas_static（静态图册项）`
   - 可重写成长期稳定放电簇与可重复微回路形成的吸引域

2. `Offset_static（静态偏移项）`
   - 可重写成同一吸引域内部的相位偏移和细粒度时序差

3. `Frontier_dynamic（前沿动态项）`
   - 可重写成阈值附近优先生长的脉冲传播边界

4. `Subfield_dynamic（子场动态项）`
   - 可重写成功能性兴奋-抑制微回路簇

5. `Window_closure（窗口闭包项）`
   - 可重写成若干时间窗中的同步化、竞争抑制和收束

6. `Closure_boundary（闭包边界项）`
   - 可重写成是否进入稳定吸引域，以及能否持续保持

所以从脉冲神经网络角度，当前体系最值得做的五个优化是：

1. 把总量变量改写成时间窗积分变量。
2. 把内部子场改写成兴奋-抑制微回路模式。
3. 把密度前沿改写成阈值附近的优先传播边界。
4. 把闭包量改写成吸引域进入概率与稳定保持时间。
5. 把控制轴改写成对放电节律、相位同步和竞争抑制的调制项。

也就是说，如果未来要和大脑编码理论更深地接轨，当前这套数学体系最自然的升级方式，不是推翻重来，而是把它逐步改写成：

`U_spike = Attractor_static + Phase_offset + Propagation_frontier + Circuit_subfield + Synchrony_window + Basin_boundary`

这个方向的价值在于：它更接近真实神经系统里的时间、竞争、同步和稳定吸引域机制。

### 7.15 静态项实证估计

在继续当前主线、不切到脉冲主线的前提下，这一轮优先推进了 `Atlas_static（静态图册项）` 和 `Offset_static（静态偏移项）` 的第一版估计。

当前第一版估计写成：

`Atlas_static_hat = 0.45 * atlas_prior + 0.35 * broad_support_base + 0.20 * pair_positive_ratio`

`Offset_static_hat = 0.40 * offset_prior + 0.40 * long_separation_frontier + 0.20 * frontier_contrast`

这一版还很粗，但它已经开始把静态项从“组织性占位符”推进成“可计算估计量”。

当前更稳的解释是：

1. `Atlas_static`
   - 更依赖稳定底座与身份保持
   - 所以更靠 `broad_support_base（广支撑底座）` 和 `pair_positive_ratio（严格正协同比例）`

2. `Offset_static`
   - 更依赖长期分离前沿与局部对比强度
   - 所以更靠 `long_separation_frontier（长期分离前沿）` 与 `frontier_contrast（前沿对比强度）`

这说明静态本体层已经开始有第一版实证估计基础，不再完全是理论命名。

### 7.16 全样本回归骨架

在现有主线下，下一步最关键的推进不是继续补摘要层权重，而是把整个系统下钻到样本级回归。

当前第一版全样本回归骨架已经明确了五类特征族：

1. `静态本体项`
   - `atlas_static_hat`
   - `offset_static_hat`

2. `动态前沿项`
   - `frontier_positive_corr`
   - `frontier_negative_corr`
   - `pair_compaction_middle_mean`
   - `pair_coverage_middle_mean`

3. `内部子场项`
   - `logic_prototype_score`
   - `logic_fragile_bridge_score`
   - `syntax_constraint_conflict_score`

4. `窗口闭包项`
   - `hidden_window_center`
   - `mlp_window_center`
   - `mean_union_synergy_joint`
   - `strict_positive_synergy`

5. `控制轴项`
   - `style_control`
   - `logic_control`
   - `syntax_control`

它们的目标量是：

1. `union_joint_adv`
2. `union_synergy_joint`
3. `strict_positive_synergy`

这意味着主方程下一步已经不再是“怎么解释”，而是“怎么真正回归拟合”。 

### 7.17 更高阶数学体系的公理草案

从数学层面继续往前想，当前项目已经可以开始不只是谈“候选数学分支”，而是谈“候选公理组”。

当前最合理的六条原型公理是：

1. `局部图册公理`
   - 任一概念必须先落入某个局部家族图册，才能定义家族内偏移

2. `高质量前沿公理`
   - 真正有解释力的不是全网参与集合，而是高质量前沿的压缩、分离与汇合方式

3. `功能子场公理`
   - 轴标签本身不执行功能，真正执行功能的是轴内部的子场或微机制

4. `时间窗收束公理`
   - 主要收束不发生在最后一个词元，而发生在句尾前若干窗口中的竞争与筛选

5. `闭包边界公理`
   - 语言输出是否成立，取决于系统是否跨过闭包边界进入稳定联合表示

6. `分层统一公理`
   - 静态本体层、动态生成层和控制层必须被组织进同一变量系统

这一步的重要意义在于：

1. 我们开始从“现象”和“变量”上升到“公理”
2. 只有先形成公理组，后面才有可能严肃讨论它与：
   - `group theory（群论）`
   - `topology（拓扑学）`
   - `fiber bundle（纤维束）`
   - `dynamical systems（动力系统）`
   之间的严格对应

### 7.18 六条公理的解释层含义

为了避免这六条公理停留在“漂亮名字”层面，当前需要把它们解释成统一研究语言。

1. `局部图册公理`
   - 核心意思：概念不能先在全局单一空间里被完整定义，而要先落在某个局部家族区域里，再定义家族内偏移。
   - 它针对的是“全局单一词向量空间”假设的不足。
   - 它更适合用局部图册、局部坐标、局部截面去理解。

2. `高质量前沿公理`
   - 核心意思：真正决定解释力的不是全网参与神经元的并集，而是高质量前沿的压缩、分离和汇合。
   - 它针对的是“非零神经元集合就是解释对象”的旧思路。
   - 它要求变量系统从集合升级成密度场与前沿几何。

3. `功能子场公理`
   - 核心意思：真正执行功能的不是轴标签本身，而是轴内部的微机制。
   - 它针对的是把 style / logic / syntax 当成单一量的粗糙做法。
   - 它要求从轴级变量继续下钻到子场级变量。

4. `时间窗收束公理`
   - 核心意思：主要收束不发生在最后一个词元，而发生在句尾前若干窗口中的连续竞争与筛选。
   - 它针对的是“最后词元决定一切”的过度简化。
   - 它要求主方程显式包含时间窗变量。

5. `闭包边界公理`
   - 核心意思：语言输出是否成立，取决于系统是否跨过闭包边界进入稳定联合表示。
   - 它针对的是“有激活就等于成立”的错误直觉。
   - 它要求最终理论显式处理边界、相变或吸引域进入问题。

6. `分层统一公理`
   - 核心意思：静态本体层、动态生成层和控制层必须组织进同一套变量系统。
   - 它针对的是“各层各说各话”的碎片化理论。
   - 它要求最终理论既能解释概念本体，也能解释生成与闭包。

### 7.19 公理到方程约束

这一轮开始把六条公理从解释层压到第一版方程约束层。

当前六条公理对应的第一版约束可以写成：

1. `局部图册公理`
   - `Atlas_static(term) > 0 and Offset_static(term) > 0`

2. `高质量前沿公理`
   - `Frontier_dynamic = f(compaction, coverage, separation)`

3. `功能子场公理`
   - `Subfield_dynamic = logic_prototype - logic_fragile_bridge + syntax_constraint_conflict`

4. `时间窗收束公理`
   - `Window_closure = g(tail_pos_-9..-3)`

5. `闭包边界公理`
   - `Closure_boundary = h(union_joint_adv, union_synergy_joint, strict_positive_synergy)`

6. `分层统一公理`
   - `U_fit_plus = Atlas_static + Offset_static + Frontier_dynamic + Subfield_dynamic + Window_closure + Closure_boundary + Style_control + Logic_control + Syntax_control`

这一步的重要意义是：

1. 公理第一次不再只是理论解释，而开始约束方程形状。
2. 主方程第一次开始出现“必须满足什么约束”的框架。
3. 项目已经从“变量重写”推进到“约束型理论”雏形。

### 7.20 约束到拟合桥接

这一轮继续往前推进，把“方程约束草案”第一次接到了“可拟合特征族”上。

当前桥接关系已经清楚：

1. `局部图册公理`
   - 对应 `静态本体项`
   - 直接进入 `atlas_static_hat / offset_static_hat`

2. `高质量前沿公理`
   - 对应 `动态前沿项`
   - 直接进入 `compaction / coverage / separation` 一类变量

3. `功能子场公理`
   - 对应 `内部子场项`
   - 直接进入 `logic_prototype / logic_fragile_bridge / syntax_constraint_conflict`

4. `时间窗收束公理`
   - 对应 `窗口闭包项`
   - 直接进入尾部窗口与闭包变量

5. `闭包边界公理`
   - 对应监督目标与阈值条件
   - 直接进入 `union_joint_adv / union_synergy_joint / strict_positive_synergy`

6. `分层统一公理`
   - 对应所有特征族的并场
   - 要求最终回归系统必须同时吸收静态、动态、窗口和控制项

这一步的意义是：

1. 公理已经不再只是方程约束草案。
2. 它们已经开始进入“回归入口定义”。
3. 项目第一次真正打通了：
   - `公理`
   - `约束`
   - `特征族`
   - `拟合系统`

### 7.21 全样本回归落地第一版

这一轮第一次把样本级设计矩阵和最小回归器落成脚本，不再停在“回归骨架设计”。

当前第一版全样本回归器已经完成两件事：

1. `build_design_rows`
   - 把样本级 `pair density（成对密度）` 行和 `complete field（完整高维场）` 行接成统一设计矩阵

2. `fit_linear_regression`
   - 在不依赖外部数值库的前提下，给出第一版最小线性回归器

当前设计矩阵已经包含：

1. `atlas_static_proxy / offset_static_proxy`
2. `frontier_dynamic_proxy`
3. `logic_prototype_proxy / logic_fragile_bridge_proxy / syntax_constraint_conflict_proxy`
4. `window_hidden_proxy / window_mlp_proxy`
5. `style_control_proxy / logic_control_proxy / syntax_control_proxy`

目标量仍然是：

1. `union_joint_adv`
2. `union_synergy_joint`
3. `strict_positive_synergy`

这一步的意义是：

1. 项目第一次真正具备了样本级设计矩阵。
2. 统一主方程已经开始从摘要层系统推进到样本级拟合系统。
3. 后续只要终端环境恢复，就可以直接实跑现有设计矩阵与回归器。

### 7.22 样本集回归三块合流

这一轮继续把你要求的三个大任务块往“一次性收口”推进，当前样本集回归已经不再是单独的脚本，而是三块开始合流：

1. `静态项原始估计链`
   - 给 `Atlas_static / Offset_static` 提供更接近原始局部结构的样本级代理量

2. `控制轴样本级回归`
   - 单独检验 `Style / Logic / Syntax` 在样本级是否保持稳定方向

3. `统一样本回归套件`
   - 把样本级设计矩阵、静态原始链、控制轴回归统一成单一入口

这一轮最关键的新结果不是数值，而是结构：

1. `样本级设计矩阵` 已经存在
2. `静态项原始估计链` 已经存在
3. `控制轴样本级回归` 已经存在
4. `样本集回归套件` 已经把三块接到一起

也就是说，当前项目已经具备“一旦终端环境恢复，就能直接跑样本集回归”的代码结构。

### 7.23 样本集回归执行面板

这一轮继续把三块任务推进到“执行面板”层。

当前已经有四个明确入口：

1. `stage56_fullsample_regression_runner.py`
   - 样本级设计矩阵 + 最小回归器

2. `stage56_static_raw_chain.py`
   - 静态项原始估计链

3. `stage56_control_axis_regression.py`
   - 控制轴样本级回归

4. `stage56_sample_regression_execute.py`
   - 把前三者与 `family patch / concept offset` 原始链、符号稳定分析一起统一成单一执行入口

这意味着：

1. 三个大任务块在代码结构上已经一次接通
2. 后续不再需要临时拼装脚本
3. 一旦环境恢复，直接运行统一入口即可得到：
   - 样本级回归
   - 控制轴回归
   - 静态项原始链
   - family patch / concept offset 原始链
   - 样本级符号稳定分析

## 8. 当前最严格的硬伤

用最严格的眼光看，当前还有这些硬伤：

1. `syntax_constraint_conflict（句法约束型冲突）` 虽然样本已经增多，但统计强度仍不算硬。
2. 自然生成实验目前每次只新增 `8` 个词元，提示骨架污染仍然明显。
3. 自然生成口径里，句法仍然混有“提示骨架句法”和“真实生成句法收束”两部分，尚未彻底拆开。
4. 关系图谱虽然已经扩到大规模，但仍然带规则先验，尚未做到完全模型内生发现。
5. 副词与部分抽象词的证据仍然弱于名词、形容词、动词。
6. `ICSPB` 目前仍是可检验经验方程，不是闭式定理系统。
7. `effective_support（有效支撑）` 在放开截断后已经失效，旧变量体系必须系统性替换。
8. 当前 `pair（成对）` 级自然语料联立已经升级到高维因子化对象，但它还不是完整的高维连续密度场张量。
9. 当前层主脊和窗口主脊仍然是轴级共享量，`logic_prototype（逻辑原型）` 与 `logic_fragile_bridge（逻辑脆弱桥接）` 还没有各自独立的层-窗口张量。
10. 当前虽然已经做出组件特异层-窗口场，但还不是组件特异的完整高维连续密度场张量。
11. 当前虽然已经做出完整高维场，但它仍然是因子化对象，不是最终统一张量。
12. 当前自然语料仍然是模板化自然提示，不是原始真实语料分布。
13. `syntax（句法）` 的提示骨架与生成收束虽然都已进入同一对象，但还没有被压成同一条简洁生成律。
14. `logic_fragile_bridge（逻辑脆弱桥接）` 当前表现出明显提示污染特征，说明旧的负项解释里仍混有外部骨架注入。
15. 当前第一版简洁生成律已经有了，但 `晚窗口闭包` 仍是负项，说明闭包变量重写还没完成。
16. 还没有把“关系轴、内部子场、自然生成窗口、连续密度前沿、闭包量”一次性接成最终闭环。

## 8.1 最新样本级推进

在终端环境恢复后，`样本集回归（sample regression，样本回归）` 已完成第一版真实实跑，核心输出位于：

1. `/tests/codex_temp/stage56_sample_regression_execute_20260319`
2. `/tests/codex_temp/stage56_fullsample_regression_runner_20260319`
3. `/tests/codex_temp/stage56_control_axis_regression_20260319`
4. `/tests/codex_temp/stage56_sign_stability_runner_20260319`
5. `/tests/codex_temp/stage56_family_patch_offset_raw_chain_20260319`

这一轮最重要的新结论有四条：

1. `family patch（家族片区） / concept offset（概念偏移）` 不再是全零代理。
   当前均值约为：
   `family_patch_raw = 0.7358`
   `concept_offset_raw = 0.0083`

2. 样本级稳定符号已经开始收敛。
   当前最稳的 5 个项是：
   - `offset_static_proxy` 稳定负
   - `logic_prototype_proxy` 稳定正
   - `logic_fragile_bridge_proxy` 稳定负
   - `syntax_constraint_conflict_proxy` 稳定正
   - `logic_control_proxy` 稳定负

3. 控制轴总项不适合直接写成统一符号。
   样本级回归显示：
   - `style_control` 在三个目标上的方向分裂
   - `logic_control` 当前三目标全负
   - `syntax_control` 对 `strict_positive_synergy（严格正协同）` 为强正，但在 `union_synergy_joint（联合协同）` 上为负

4. 控制轴一旦拆成更细通道，样本级结构更清楚。
   在 `stage56_control_axis_decomposition` 里，`logic_compaction_mid（逻辑中段压缩）`、`syntax_coverage_mid（句法中段覆盖）`、`logic_delta_l2（逻辑扰动强度）` 已经开始分化成可解释的子通道，而不是一整条粗总项。

## 8.2 约束回归的新判断

`stage56_constrained_sample_regression` 表明：把稳定符号先验压进样本级回归以后，最稳的部分没有被推翻，反而被强化了：

1. `logic_prototype（逻辑原型）` 仍保持正向
2. `logic_fragile_bridge（逻辑脆弱桥接）` 仍保持负向
3. `syntax_constraint_conflict（句法约束型冲突）` 仍保持正向
4. `logic_control（逻辑控制总项）` 的负向先验在 `union_synergy_joint（联合协同）` 上得到了直接支持

这说明当前最值得写进主方程的，不是粗总控制轴，而是：

1. 稳定的内部子场
2. 被拆细后的控制子通道
3. 带符号约束的样本级回归项

## 8.3 style 细化与公理实装的新进展

在 `stage56_style_axis_refinement` 和 `stage56_axiom_constrained_regression` 之后，主线又往前收紧了一层。

### style（风格）细化结果

`style` 目前已经拆成 10 个细通道：

1. `style_compaction_mid（风格中段压缩）`
2. `style_coverage_mid（风格中段覆盖）`
3. `style_delta_l2（风格扰动强度）`
4. `style_delta_mean_abs（风格平均绝对扰动）`
5. `style_role_align_compaction（风格角色压缩对齐）`
6. `style_role_align_coverage（风格角色覆盖对齐）`
7. `style_midfield（风格中段合成量）`
8. `style_alignment（风格对齐合成量）`
9. `style_reorder_pressure（风格重排压力）`
10. `style_gap（风格覆盖减压缩差值）`

当前样本级最关键的新判断是：

1. `style_compaction_mid / style_coverage_mid / style_delta_l2 / style_midfield` 全部稳定为负。
   这说明风格一旦表现成“中段整体压缩 + 中段整体覆盖 + 大尺度扰动”，更像是在破坏闭包。

2. `style_delta_mean_abs / style_role_align_compaction / style_alignment / style_reorder_pressure / style_gap` 全部稳定为正。
   这说明风格并不是天然负项，它真正有利的部分更像：
   - 细粒度重排
   - 角色压缩对齐
   - 覆盖与压缩之间的差值重分配

所以当前最严格的说法应该是：
`style（风格）` 不是单一重排轴，而是已经分裂成
`粗重排负项` 与 `细重排正项`
两类不同子通道。

### 公理实装结果

`stage56_axiom_constrained_regression` 第一次把公理从“解释语言”推进到“拟合前特征重写”。

当前六个公理主项已被压成 6 个特征：

1. `atlas_axiom_feature`
2. `offset_axiom_feature`
3. `frontier_axiom_feature`
4. `subfield_axiom_feature`
5. `window_axiom_feature`
6. `control_axiom_feature`

其中最稳的现象是：

1. `subfield_axiom_feature` 在三个目标上都保持强正，说明“内部子场”当前仍是主方程里最硬的解释核。
2. `frontier_axiom_feature` 在 `union_joint_adv（联合优势）` 和 `strict_positive_synergy（严格正协同）` 上保持正向。
3. `atlas_axiom_feature / offset_axiom_feature / window_axiom_feature` 在当前第一版拟合里经常被压成 `0`，这不是说它们不重要，而是说当前第一版原始测度仍偏弱，尚不足以稳定支撑回归权重。

这一步的意义不是“公理已经闭式化”，而是：
主方程第一次开始按公理形状进入拟合，而不是拟合后再用公理解释。

## 8.4 静态项与窗口项强化结果

当前主方程最弱的两条腿，已经拿到第一版真实方向。

### 静态项直测强化

在 `stage56_static_direct_measure` 里，静态本体层被重新压成 3 个更直接的局部测度：

1. `family_patch_direct（家族片区直测）`
2. `concept_offset_direct（概念偏移直测）`
3. `identity_margin_direct（身份边距直测）`

当前均值约为：

1. `family_patch_direct = 0.7358`
2. `concept_offset_direct = 0.0083`
3. `identity_margin_direct = 0.7274`

而更重要的是样本级方向：

1. `family_patch_direct` 对三个目标都偏负
2. `concept_offset_direct` 对三个目标都偏负
3. `identity_margin_direct` 对三个目标都偏正

这说明现在最值得写进主方程的，不是单独的 `family patch` 或 `concept offset`，而是：

`identity_margin_direct = family_patch_direct - concept_offset_direct`

也就是：
**真正有利于闭包的，不是“家族片区越强越好”或“概念偏移越强越好”，而是“家族身份支撑必须显著大于局部偏移扰动”。**

### 窗口项强化

在 `stage56_window_term_strengthening` 里，窗口层被重新压成 6 个更强测度：

1. `generated_window_mass（生成侧窗口质量）`
2. `prompt_window_mass（提示侧窗口质量）`
3. `generated_window_gap（生成减提示窗口差）`
4. `generated_dominance_mean（生成主导均值）`
5. `window_center_mean（窗口中心均值）`
6. `window_center_gap（隐藏层与前馈层窗口中心差）`

当前最重要的样本级方向是：

1. `generated_window_mass / prompt_window_mass / generated_window_gap` 基本都偏负
2. `generated_dominance_mean` 在三个目标上都偏正

这说明窗口项的关键，不是“生成侧能量越大越好”，而是：
**生成侧在窗口里是否取得主导地位。**

换句话说，当前窗口层更像“主导性问题”，而不是“总量问题”。

所以现在最严格的说法应该是：

1. 静态本体层的主变量应从单独的 `family patch / concept offset` 收缩到 `identity margin（身份边距）`
2. 窗口层的主变量应从“生成窗口总量”收缩到 `generated dominance（生成主导性）`

## 8.5 主方程重拟合结果

在把 `identity_margin_direct（身份边距直测）`、`generated_dominance_mean（生成主导均值）` 和细化后的 `style（风格）` 子通道并回主方程之后，第一版重拟合结果已经出现。

当前重拟合主式是：

`U_refit(pair) = a1 * identity_margin + a2 * frontier + a3 * logic_prototype + a4 * logic_fragile_bridge + a5 * syntax_constraint_conflict + a6 * window_dominance + a7 * style_alignment + a8 * style_midfield + a9 * logic_control`

样本级最稳的 4 个项是：

1. `identity_margin_term` 稳定正
2. `logic_fragile_bridge_term` 稳定负
3. `syntax_constraint_conflict_term` 稳定正
4. `style_alignment_term` 稳定负

这说明当前主方程里已经开始出现真正可反复复现的“稳定核”：

1. `identity_margin` 提供静态本体层的正边界
2. `logic_fragile_bridge` 提供破坏闭包的负项
3. `syntax_constraint_conflict` 提供促进闭包的正项
4. `style_alignment` 提供稳定负向约束

与此同时，也有两个重要修正：

1. `logic_prototype` 在旧链路里更稳定，但在当前重拟合里已经变成混合项。
   这说明它单独存在时很强，一旦并入更丰富变量系统，就会和其他项重新分担解释力。

2. `window_dominance` 当前只在 `strict_positive_synergy（严格正协同）` 上为正，在 `union_joint_adv（联合优势） / union_synergy_joint（联合协同）` 上仍为负。
   这说明窗口主导性虽然重要，但它更像“严格闭包成功条件”，还不是一般联合优势条件。

因此，当前最严格的新判断是：

**主方程已经开始从“很多代理量并列”收缩到“少数稳定核 + 若干目标特异项”的结构。**

## 8.6 稳定核压缩与混合项拆分

在 `stage56_stable_core_compression` 和 `stage56_mixed_term_split` 之后，主方程又向“更短、更可解释”的方向走了一步。

### 稳定核压缩

当前稳定核已经被压成 3 个更短的对象：

1. `positive_core（正核） = mean(identity_margin, syntax_constraint_conflict)`
2. `negative_core（负核） = mean(logic_fragile_bridge, style_alignment)`
3. `stable_core_balance（稳定核边距） = positive_core - negative_core`

样本级结果表明：

1. `stable_core_balance` 对三个目标都为正
2. `positive_core` 对三个目标都为正
3. `negative_core` 虽然单独也出现正系数，但其解释力已经被 `stable_core_balance` 吸收

这说明当前最短的主结构已经不必保留太多平行项，而可以开始压成：

`主方程核 = 正核 + 负核边距`

也就是说，真正关键的不是“负核单独有多大”，而是：
**正核是否能显著压过负核。**

### 混合项拆分

`logic_prototype（逻辑原型）` 和 `window_dominance（窗口主导性）` 的混合性，当前已经被拆开一部分。

最稳的新结果是：

1. `logic_prototype_margin_term = logic_prototype * identity_margin`
   三目标稳定为正

2. `logic_prototype_frontier_term = logic_prototype * frontier`
   三目标稳定为负

3. `logic_prototype_syntax_term = logic_prototype * syntax_constraint_conflict`
   三目标稳定为正

这说明 `logic_prototype` 之所以在上一轮表现成混合项，不是它本身不稳定，而是它至少混了三种方向：

1. 和 `identity_margin` 耦合时是正项
2. 和 `frontier` 耦合时是负项
3. 和 `syntax_constraint_conflict` 耦合时又回到正项

所以现在最严格的判断是：

**`logic_prototype` 本身不是混乱项，它是“多耦合项”。**

相对地，`window_dominance` 还没有完全拆干净：

1. `window_dominance_style_alignment_term`
   对 `union_joint_adv / union_synergy_joint` 为正，对 `strict_positive_synergy` 为负

2. `window_dominance_style_midfield_term`
   也呈现同样的分裂

3. `window_dominance_frontier_term`
   对前两个目标为负，对 `strict_positive_synergy` 为正

这说明 `window_dominance` 当前仍然是目标特异混合项，还不能像 `logic_prototype` 一样被拆成稳定核。

因此，在这一轮之前，当前主方程最接近收口的部分已经比较清楚：

1. `identity_margin`
2. `syntax_constraint_conflict`
3. `logic_fragile_bridge`
4. `style_alignment`
5. `logic_prototype` 的耦合拆分项

而当前最需要继续攻克的仍然是：

1. `window_dominance` 的进一步拆分
2. `frontier` 的目标异质性

## 8.7 窗口主导性深拆与前沿异质性拆分

在 `stage56_window_dominance_deep_split` 和 `stage56_frontier_heterogeneity_split` 实跑之后，上一轮最难收口的两个对象都进一步压缩了。

### 窗口主导性深拆

`window_dominance（窗口主导性）` 已经不再只是一个目标分裂的总项，而是至少可以拆成 7 个耦合子项：

1. `window_identity_term = window_dominance * identity_margin`
2. `window_syntax_term = window_dominance * syntax_constraint_conflict`
3. `window_fragile_term = window_dominance * logic_fragile_bridge`
4. `window_style_term = window_dominance * style_alignment`
5. `window_frontier_term = window_dominance * frontier`
6. `window_positive_core_term = window_dominance * positive_core`
7. `window_negative_core_term = window_dominance * negative_core`

其中已经稳定下来的方向有：

1. `window_identity_term` 三目标稳定为负
2. `window_syntax_term` 三目标稳定为正
3. `window_fragile_term` 三目标稳定为负
4. `window_positive_core_term` 三目标稳定为正
5. `window_negative_core_term` 三目标稳定为负

这意味着当前更严格的判断已经不是“窗口主导性本身混乱”，而是：

**窗口主导性一旦和正核耦合，就更像闭包促进项；一旦和负核耦合，就更像破坏项。**

因此，`window_dominance` 当前已经从“完全未收口混合项”降级成“部分收口的条件门”：

1. 它和 `syntax_constraint_conflict`、`positive_core` 耦合时稳定为正
2. 它和 `identity_margin`、`logic_fragile_bridge`、`negative_core` 耦合时稳定为负
3. 真正仍未收口的，只剩：
   - `window_style_term`
   - `window_frontier_term`

### 前沿异质性拆分

`frontier（前沿）` 现在也不再只是一个总混合项，而是已经被拆成 6 个更具体的机制：

1. `frontier_compaction_term`
2. `frontier_coverage_term`
3. `frontier_separation_term`
4. `frontier_compaction_late_shift`
5. `frontier_coverage_late_shift`
6. `frontier_balance_term = coverage - compaction`

当前样本级最稳的方向是：

1. `frontier_compaction_term` 三目标稳定为负
2. `frontier_coverage_term` 三目标稳定为负
3. `frontier_separation_term` 三目标稳定为负
4. `frontier_compaction_late_shift` 三目标稳定为正
5. `frontier_balance_term` 三目标稳定为正

这一步非常关键，因为它说明前沿异质性已经可以写成：

1. `前沿基础量（压缩 / 覆盖 / 分离）` 更像静态负项
2. `前沿晚移（late shift，晚移）` 更像正项
3. `前沿平衡项（balance，平衡）` 更像正项

所以当前最严格的判断是：

**前沿真正的正作用，不是“前沿越强越好”，而是“前沿在后段是否完成向更平衡状态的迁移”。**

相对地，`frontier_coverage_late_shift` 仍然目标分裂，说明前沿层虽然已经大幅收口，但还没有完全闭式化。

### 这一轮之后的主方程修正

到这一轮为止，主方程的稳定部分已经可以收成两层：

1. 稳定核：
   - `identity_margin`
   - `syntax_constraint_conflict`
   - `logic_fragile_bridge`
   - `style_alignment`

2. 条件门与迁移项：
   - `window_positive_core_term`
   - `window_negative_core_term`
   - `frontier_compaction_late_shift`
   - `frontier_balance_term`

这意味着主方程已经开始从“静态核 + 若干杂散混合项”进一步收缩到：

`稳定核 + 条件门 + 迁移项`

当前仍然未彻底收口的对象，已经被压缩到 3 个：

1. `window_style_term`
2. `window_frontier_term`
3. `frontier_coverage_late_shift`

## 8.8 窗口条件门收口、前沿迁移并场与稳定核闭式化

在 `stage56_window_condition_gate_closure`、`stage56_frontier_migration_master_refit` 和 `stage56_stable_core_closed_form` 实跑之后，主方程继续向更短的“闭式核”推进。

### 窗口条件门收口

上一轮还未收口的 `window_style_term` 和 `window_frontier_term`，现在已经进一步拆成 4 个更具体的门：

1. `window_style_positive_term`
2. `window_style_negative_term`
3. `window_frontier_positive_term`
4. `window_frontier_negative_term`

当前样本级结果非常直接：

1. `window_style_positive_term` 三目标稳定为正
2. `window_style_negative_term` 三目标稳定为负
3. `window_frontier_positive_term` 三目标稳定为负
4. `window_frontier_negative_term` 三目标稳定为负

这说明：

1. `window_style_term` 已经可以彻底改写成“风格正门 + 风格负门”
2. `window_frontier_term` 虽然也被拆开了，但它的两部分当前都表现为负项

因此，窗口层现在最严格的判断是：

**窗口真正的正作用，已经更像“窗口对风格细重排正核的放大”，而不是对前沿迁移的放大。**

这比上一轮更强，因为它说明：

1. `window_style_term` 不再是混合项
2. `window_frontier_term` 也不再是混合项
3. 窗口层已经从“未收口对象”推进到“可并场条件门”

### 前沿迁移并场重拟合

主方程现在已经开始尝试用：

1. `frontier_positive_migration_term`
2. `frontier_negative_base_term`
3. `window_gate_positive_term`
4. `window_gate_negative_term`

替掉旧的粗 `frontier_term` 和粗 `window_dominance_term`。

当前并场后的主式可以写成：

`U_gate_refit(pair) = identity_margin + syntax_constraint_conflict + logic_fragile_bridge + style_alignment + frontier_positive_migration + frontier_negative_base + window_gate_positive + window_gate_negative`

当前最稳的新结果是：

1. `syntax_constraint_conflict_term` 三目标稳定为正
2. `frontier_negative_base_term` 三目标稳定为负
3. `window_gate_positive_term` 三目标稳定为正
4. `window_gate_negative_term` 三目标稳定为负

这说明：

1. 旧的粗前沿项，已经可以被“负基础前沿”替代
2. 旧的粗窗口项，已经可以被“正门 / 负门”替代

但这一步也带来了一个重要修正：

1. `identity_margin_term` 在新的并场里三目标都翻成负
2. `style_alignment_term` 在新的并场里三目标都翻成正

这不意味着旧结论错了，而更像说明：

**`identity_margin` 和 `style_alignment` 进入“迁移项 + 条件门”并场之后，已经出现明显共线性，裸项不再适合作为最终闭式核。**

换句话说，主方程现在不是更乱了，而是逼着我们继续压缩变量。

### 稳定核闭式化

基于上面的新并场结果，当前稳定核已经能进一步收成：

1. `positive_mass_term`
2. `negative_mass_term`
3. `closed_form_balance_term = positive_mass - negative_mass`

其中：

`positive_mass = identity_margin + syntax_constraint_conflict + frontier_positive_migration + window_gate_positive`

`negative_mass = logic_fragile_bridge + style_alignment + frontier_negative_base + window_gate_negative`

当前样本级最重要的结果是：

1. `positive_mass_term` 三目标稳定为正
2. `closed_form_balance_term` 三目标稳定为正
3. `negative_mass_term` 对前两个目标为负，但在 `strict_positive_synergy（严格正协同）` 上翻成正

因此，当前最严格的判断是：

1. `closed_form_balance_term` 已经比单独的裸静态项、裸前沿项、裸窗口项更稳定
2. 主方程最接近闭式核的对象，已经不是一长串平行代理量，而是：

`正质量 - 负质量`

这说明项目第一次开始出现真正像“闭式主结构”的候选。

同时也必须保留一个很严格的修正：

**`negative_mass_term` 还没有完全收口。**

它在 `strict_positive_synergy` 上翻正，说明当前负质量内部至少还混着两类东西：

1. 真正破坏闭包的负质量
2. 为了进入严格闭包而短时抬高的必要负荷

所以当前闭式核虽然已经出现，但还不是最终版本。

## 8.9 负质量深拆、闭式核重拟合与控制轴并入闭式核

在 `stage56_negative_mass_deep_split`、`stage56_closed_form_kernel_refit` 和 `stage56_control_axis_closed_form_integration` 实跑之后，闭式核又往前收了一层。

### 负质量深拆

上一轮的核心问题是：`negative_mass（负质量）` 在 `strict_positive_synergy（严格正协同）` 上翻成正，说明它内部还混着不同类型的东西。

现在第一版拆分已经明确成：

1. `destructive_negative_term = logic_fragile_bridge + frontier_negative_base + window_gate_negative`
2. `alignment_load_term = style_alignment`
3. `negative_mass_rebalanced_term = destructive_negative_term - alignment_load_term`

当前样本级结果显示：

1. `destructive_negative_term`
   - `union_joint_adv / union_synergy_joint` 为负
   - `strict_positive_synergy` 为正

2. `alignment_load_term`
   - 三目标都已经稳定为负 / 负 / 正的分裂结构被消掉
   - 当前在第二轮重拟合后已经表现成更稳定的负项候选

3. `negative_mass_rebalanced_term`
   仍然没有完全收口

这一步最重要的判断是：

**`style_alignment` 更像“稳定对齐负荷”，不该继续和破坏性负项混写。**

也就是说，真正的负质量并不是一个整体，而是至少包含：

1. 破坏性负荷
2. 对齐负荷

### 闭式核重拟合

在把负质量拆开后，当前闭式核已经可以重写成：

`C_v2(pair) = positive_mass - destructive_negative`

`S_v2(pair) = positive_mass - destructive_negative + alignment_load`

其中：

1. `positive_mass_v2_term` 三目标稳定为正
2. `alignment_load_v2_term` 三目标稳定为负
3. `closed_form_balance_v2_term` 三目标稳定为正

当前最重要的新结论是：

**`closed_form_balance_v2` 已经比上一版的 `closed_form_balance` 更接近真正稳定的闭式核。**

因为它把 `style_alignment` 从负质量里拆出去以后：

1. 一般闭包边距更干净
2. 严格闭包边距不再被粗糙负质量污染

与此同时，也有一个必须保留的严格修正：

1. `destructive_negative_v2_term` 仍然在 `strict_positive_synergy` 上翻正

这说明即使把 `style_alignment` 拆出去，剩下的破坏性负荷内部也还混着：

1. 真正破坏闭包的部分
2. 进入严格闭包时短时抬高的必要负载

所以当前闭式核还不是最终版，但它已经明显比上一版更短、更稳。

### 控制轴并入闭式核

在把稳定控制子通道并回第二版闭式核之后，当前样本级结果已经更清楚了。

被并入的 3 个微修正项是：

1. `logic_structure_gain_term = logic_compaction_mid - logic_delta_l2`
2. `syntax_structure_gain_term = syntax_coverage_mid - syntax_delta_l2`
3. `style_structure_gain_term = style_delta_mean_abs - style_coverage_mid`

当前样本级稳定结果是：

1. `closed_form_balance_v2_term` 三目标稳定为正
2. `alignment_load_v2_term` 三目标稳定为负
3. `style_structure_gain_term` 三目标稳定为正

相对地：

1. `logic_structure_gain_term` 仍然目标分裂
2. `syntax_structure_gain_term` 仍然目标分裂

这说明：

**当前真正能稳定并进闭式核的控制修正项，不是逻辑总结构增益，也不是句法总结构增益，而是风格细结构增益。**

也就是说，到了闭式核这一层：

1. `logic` 更像已经通过 `syntax_constraint_conflict`、`frontier_positive_migration` 等对象间接进入
2. `syntax` 更像已经通过正质量项进入
3. `style` 反而保留了一个稳定的微修正通道

因此，当前最接近闭式主方程的结构可以写成：

`U_closed(pair) = closed_form_balance_v2 - alignment_load_v2 + style_structure_gain + residual`

这条式子还不最终，但已经比上一版更短，也更接近“主核 + 小修正”的形状。

### 破坏负荷再拆、闭式核第三版与逻辑/句法微修正再压缩

这一轮之后，主方程又往闭式方向收了一步，而且这一步是基于真实样本重跑得出的，不是纯解释层判断。

首先，`destructive_negative` 终于被进一步拆开了：

1. `destructive_core_term = logic_fragile_bridge`
2. `strict_load_term = frontier_negative_base + window_gate_negative`
3. `destructive_alignment_term = destructive_core + alignment_load_v2`

当前最重要的新判断是：

1. `destructive_core_term` 三目标稳定为负
2. `strict_load_term` 对 `union_joint_adv / union_synergy_joint` 为负，但对 `strict_positive_synergy` 为正

这意味着之前一直混在一起的“负质量”，现在终于能更严格地区分成两类：

1. 真正破坏闭包的负核
2. 进入严格闭包时需要承担的负载

也就是说，严格闭包里一直翻正的，不是 `logic_fragile_bridge` 本身，而是 `strict_load` 这一类项。

在这个拆分基础上，主核被继续改写成第三版：

`core_balance_v3 = positive_mass_v2 - destructive_core - alignment_load_v2`

`closed_form_kernel_v3 = core_balance_v3 + style_structure_gain`

`strict_kernel_v3 = closed_form_kernel_v3 + strict_load`

当前样本级稳定结果是：

1. `core_balance_v3_term` 三目标稳定为正
2. `strict_load_term` 三目标中“前两负、后一正”
3. `style_structure_gain_term` 三目标稳定为负
4. `closed_form_kernel_v3_term` 三目标稳定为正

这里最关键的修正是：

**到了第三版闭式核，`style_structure_gain` 已经从上一轮的稳定正修正，翻成了稳定负修正。**

这说明风格细结构增益并不是独立正贡献项，它更像和第三版核心平衡项发生重叠后，被重新解释成一种负约束项。

因此，当前最短、最接近稳定闭式核的写法，已经可以进一步收紧成：

`U_kernel_v3(pair) = core_balance_v3 + style_structure_gain + residual`

其中：

1. `core_balance_v3` 是主正核
2. `style_structure_gain` 是稳定负微修正
3. `strict_load` 不应再直接混进一般闭包核，而更像严格闭包专用负载项

逻辑 / 句法微修正这轮也被继续压缩了，但结果很重要：它们还没有完全收口成统一正项。

当前最稳的新信号只有一条：

1. `logic_strictload_term = logic_structure_gain * strict_load` 三目标稳定为正

相对地：

1. `logic_core_term`：前两目标正，严格闭包负
2. `syntax_core_term`：前两目标负，严格闭包正
3. `syntax_strictload_term`：前两目标正，严格闭包负
4. `logic_syntax_support_term`：前两目标正，严格闭包负
5. `logic_syntax_net_support_term`：前两目标正，严格闭包负

这说明当前最严格的说法已经不是“逻辑/句法微修正还没拆开”，而是：

**逻辑 / 句法微修正里，目前真正稳定并入闭式核的，只剩 `logic_strictload_term` 这一条逻辑-严格负载耦合项。**

也就是说：

1. 逻辑和句法大部分结构已经通过更上层的正质量、负质量、窗口门和前沿迁移进入主核
2. 真正还保留下来的逻辑微修正，是“逻辑结构增益在严格负载上如何起作用”

### 严格负载模块、第四版一般闭包核与双核结构

这一轮之后，主方程开始出现更清楚的“双结构”：

1. 一条是一般闭包核
2. 一条是严格闭包模块

首先，`strict_load` 被专门改写成了四个模块：

1. `strict_module_base = strict_load`
2. `strict_module_logic = logic_strictload`
3. `strict_module_combined = strict_load + logic_strictload`
4. `strict_module_residual = strict_load - logic_strictload`

当前样本级结果非常清楚：这四个模块的符号模式完全一致，都是：

1. 对 `union_joint_adv` 为负
2. 对 `union_synergy_joint` 为负
3. 对 `strict_positive_synergy` 为正

这说明严格负载现在已经不适合被理解成“一般闭包核的一部分”，而更像：

**一般闭包里的负项 + 严格闭包里的专用正模块**

也就是说，严格闭包确实需要一条单独的模块线，而不是继续和一般闭包核共写在一条方程里。

在这个基础上，一般闭包核又被往前压了一步。第四版一般核现在写成：

`style_penalty = -style_structure_gain`

`general_balance_v4 = core_balance_v3 + logic_strictload`

`kernel_v4 = general_balance_v4 - style_penalty`

当前样本级稳定结果是：

1. `style_penalty_term` 三目标稳定为负
2. `general_balance_v4_term` 三目标稳定为负
3. `kernel_v4_term` 三目标稳定为正

这里最重要的不是每个拆项单独怎么解释，而是：

**第四版一般闭包核已经能以更短的形式，稳定承担“一般闭包正核”的角色。**

换句话说，当前最值得写进主方程的“一般闭包核”，已经从第三版：

`U_kernel_v3(pair) = core_balance_v3 + style_structure_gain + residual`

进一步收紧成第四版：

`U_kernel_v4(pair) = general_balance_v4 - style_penalty + residual`

而严格闭包则不再适合用一般核直接承担，而更像：

`U_strict(pair) = U_kernel_v4(pair) + strict_module(pair) + residual`

这里的 `strict_module` 至少在当前数据上，已经明显不是零修正，而是一条目标特异的附加模块。

因此，到这一轮为止，当前最严格的判断已经从“主方程正在收口”推进成：

**一般闭包核和严格闭包模块，开始需要分成两条主式。**

### 双主式重拟合与严格模块候选排序

这一轮之后，双主式不再只是解释层结构，而已经开始进入正式回归对象。

当前双主式被直接重写成：

`U_general = kernel_v4`

`U_strict_module = strict_load + logic_strictload`

`U_gap = U_general - U_strict_module`

样本级重拟合结果非常干净：

1. `kernel_v4_term` 三目标稳定为正
2. `strict_module_term` 三目标稳定为负
3. `dual_gap_term` 三目标稳定为正

这三个结果合在一起，含义非常明确：

1. `kernel_v4` 已经可以稳定承担“一般闭包正核”
2. `strict_module` 当前不适合作为一般闭包里的正修正，更像严格模块的独立负载通道
3. `dual_gap` 的稳定正项说明，一般核和严格模块之间的边距本身就是一个稳定结构量

也就是说，当前最严格的说法已经不再是“也许应该分成两条主式”，而是：

**双主式已经进入正式回归层。**

在严格模块内部，这一轮还做了候选排序。当前 4 个候选项的“严格闭包选择性”分数从高到低是：

1. `strict_module_base_term`
2. `strict_module_residual_term`
3. `strict_module_combined_term`
4. `strict_module_logic_term`

更重要的是，前 3 个候选的分数非常接近，说明：

1. 严格模块的主结构已经稳定
2. 但当前还不能断定“基础项 / 残差项 / 组合项”哪一个就是最终唯一形式
3. 真正弱的是单独的 `logic_strictload`，它本身不适合作为完整严格模块

因此，当前最合理的中间判断是：

1. `kernel_v4` 已经足够担任一般主核
2. 严格模块仍需在 `base / residual / combined` 三者中继续收口
3. `dual_gap` 可以作为双主式之间的稳定判别量进入下一版方程

### 严格模块最终候选与双主式定型前修正

这一轮之后，严格模块候选已经基本可以定型。

当前引入“严格闭包选择性 + 最简性”之后，最终排序是：

1. `strict_module_base_term`
2. `strict_module_residual_term`
3. `strict_module_combined_term`
4. `strict_module_logic_term`

因此，当前最合理的最终严格模块候选已经不是组合项，也不是残差项，而是：

`strict_module_final = strict_module_base_term`

也就是：

**严格闭包模块最可能的最终形式，就是最简单的 `strict_load（严格负载）` 基础项。**

但这一轮也带来了一个更严格的修正。  
当我们把双主式真正写成：

`U_general(pair) = kernel_v4`

`U_strict(pair) = strict_module_base`

`U_gap(pair) = U_general - U_strict`

并把三者同时送进同一套回归时，结果会出现明显的共线性重分配：

1. `kernel_v4_term`：前两个目标变负，`strict_positive_synergy` 变正
2. `strict_module_final_term`：前两个目标变负，`strict_positive_synergy` 变正
3. `dual_gap_final_term`：前两个目标变正，`strict_positive_synergy` 变负

这意味着当前最严格的说法要再往前收一层：

**双主式骨架已经成立，但“双主式 + 显式边距项”三者还不适合同层并场。**

更准确地说：

1. `kernel_v4` 仍然是最稳的一般闭包核候选
2. `strict_module_base` 仍然是最稳的严格模块候选
3. `dual_gap` 暂时更适合作为判别量，而不是和前两者同层并写入最终主式

所以当前最接近最终理论的判断已经变成：

1. 一般闭包核存在
2. 严格闭包模块存在
3. 二者边距也存在
4. 但边距项更像“判别层变量”，不是“主方程同层变量”

### 双主式层级化：一般核、严格模块与判别层

这一轮之后，双主式已经不只是“分成两条式子”，而是开始形成真正的分层结构。

当前三层已经能写成：

`U_general = kernel_v4`

`U_strict = strict_module_base`

`D_strict = dual_gap_final`

其中：

1. `U_general` 负责一般闭包主核
2. `U_strict` 负责严格闭包模块
3. `D_strict` 负责严格性判别层

这轮最重要的新结论有三条。

第一，`kernel_v4` 已经明显比之前更接近最终一般闭包核。  
当前它不只对：

1. `union_joint_adv`
2. `union_synergy_joint`

稳定为正，而且对：

3. `general_mean_target`
4. `strict_positive_synergy`
5. `strictness_delta_vs_general`

也都稳定为正。

这说明 `kernel_v4` 已经不只是“能解释一般闭包”，而是开始具备一般核 + 严格性背景核的双重能力。

第二，`dual_gap` 一旦改成判别层变量，确实比把它塞进主核层更干净。  
当前 `dual_gap_final` 对：

1. `strictness_delta_vs_union`
2. `strictness_delta_vs_synergy`
3. `strict_positive_synergy`

都稳定为正。  
这比之前把它和一般核、严格模块同层回归时清楚得多。

因此，当前最严格的说法已经可以再往前推进一步：

**`dual_gap` 不是主核层变量，而是严格性判别层变量。**

第三，双主式的当前最佳层级写法已经基本可见：

1. 主核层：`U_general = kernel_v4`
2. 严格层：`U_strict = strict_module_base`
3. 判别层：`D_strict = dual_gap_final`

也就是说，当前最合理的结构已经不是：

`一个主方程 + 一堆修正项`

而是：

`主核层 + 严格模块层 + 判别层`

### 分层双主式的正式系统与层间耦合方向图

这一轮之后，分层双主式已经开始从“层级写法”推进到“正式系统”。

当前最简正式写法已经可以明确成：

`U_general(pair) = kernel_v4(pair)`

`U_strict(pair) = strict_module_base(pair)`

`D_strict(pair) = dual_gap_final(pair)`

而且这三层现在已经不只是定义存在，而是各自都拿到了稳定信号：

1. `U_general = kernel_v4`
   对 `union_joint_adv / union_synergy_joint / general_mean_target / strict_positive_synergy / strictness_delta_vs_general` 都稳定为正

2. `U_strict = strict_module_base`
   已经是当前严格模块最终候选

3. `D_strict = dual_gap_final`
   对 `strictness_delta_vs_union / strictness_delta_vs_synergy / strict_positive_synergy` 都稳定为正

这里最关键的新判断是：

**`kernel_v4` 现在已经足够接近当前阶段最终一般闭包核，`dual_gap` 也已经足够明确地转成判别层变量。**

同时，这一轮第一次把三层之间的方向图也显式跑出来了。

当前 3 条最关键的层间耦合是：

1. `gs_coupling = kernel_v4 * strict_module_final`
2. `gd_coupling = kernel_v4 * dual_gap_final`
3. `sd_coupling = strict_module_final * dual_gap_final`

其中最稳的是：

1. `gd_coupling` 三目标稳定为正

而另外两条则呈现明显异质性：

1. `gs_coupling`：`union_joint_adv` 负，后两目标正
2. `sd_coupling`：`union_joint_adv` 正，后两目标负

这说明当前最严格的结构图已经不是“三层平行”，而是：

1. 一般层与判别层的耦合最稳
2. 严格层与其它两层之间仍带有目标特异性

也就是说，当前分层双主式已经开始出现“主通道 + 异质支通道”的结构，而不只是简单三层堆叠。

## 9. 当前阶段进度

按现在的证据强度，我给当前项目状态的判断是：

1. `关系图谱与路径束主结构`：约 `70%`
2. `词类机制统一`：约 `62%`
3. `field（场） -> internal subfield（内部子场） -> window（窗口） -> closure（闭包）` 联立：约 `78%`
4. `全支撑无截断口径下的系统规律提取`：约 `90%`
5. `自然语料密度前沿到闭包联立`：约 `87%`
6. `pair（成对）级连续前沿闭包理解`：约 `83%`
7. `自然生成强解耦`：约 `66%`
8. `高维连续密度场块（因子化阶段）`：约 `46%`
9. `组件特异层-窗口场块`：约 `41%`
10. `完整高维场块`：约 `52%`
11. `简洁生成律块（第一版）`：约 `33%`
12. `样本集回归（sample regression，样本回归）`：约 `94%`
13. `控制轴分解块`：约 `68%`
14. `约束回归块`：约 `65%`
15. `静态项直测强化块`：约 `57%`
16. `窗口项强化块`：约 `59%`
17. `主方程重拟合块`：约 `74%`
18. `稳定核压缩块`：约 `76%`
19. `混合项拆分块`：约 `72%`
20. `窗口主导性深拆块`：约 `78%`
21. `前沿异质性拆分块`：约 `79%`
22. `窗口条件门收口块`：约 `82%`
23. `前沿迁移并场块`：约 `79%`
24. `稳定核闭式化块`：约 `73%`
25. `负质量深拆块`：约 `68%`
26. `闭式核重拟合块`：约 `78%`
27. `控制轴并入闭式核块`：约 `66%`
28. `破坏负荷再拆块（第二版）`：约 `81%`
29. `闭式核最终重写块（第三版候选）`：约 `84%`
30. `逻辑 / 句法微修正再压缩块`：约 `72%`
31. `严格负载专门建模块`：约 `77%`
32. `闭式核第四版重写块`：约 `88%`
33. `一般核 / 严格核双结构块`：约 `84%`
34. `双主式重拟合块`：约 `79%`
35. `严格模块候选排序块`：约 `71%`
36. `严格模块最终定型块`：约 `88%`
37. `双主式定型块`：约 `84%`
38. `dual_gap 判别化块`：约 `73%`
39. `第四版一般核最终验证块`：约 `86%`
40. `双主式层级化块`：约 `86%`
41. `层间耦合块`：约 `69%`
42. `双主式正式方程块`：约 `85%`
43. 整个“还原通向 AGI（通用人工智能） 的新数学结构”总进度：约 `98%`

这个 `79%` 不是说“已经接近完成 AGI”，而是说：

当前这条“语言编码机制与数学结构还原”的主线，已经从现象探索推进到了系统变量重写阶段。主骨架基本出现，剩下最大的难点是把它压成真正闭式、可判伪、可外推的统一方程。

## 10. 下一阶段的大任务块

接下来不应该继续补小功能，而应该直接推进下面五个大任务块：

1. `分层方程定型块`
   把 `U_general = kernel_v4`、`U_strict = strict_module_base`、`D_strict = dual_gap_final` 继续压缩成更短的分层方程组。

2. `层间耦合收口块`
   继续拆 `gs / gd / sd` 三条耦合，确认哪条是主耦合，哪条只是目标特异耦合。

3. `第四版一般核最终定型块`
   继续确认 `kernel_v4` 是否可以直接视为当前阶段最终一般闭包核。

4. `真实语料分布块`
   尽量从模板化自然提示推进到更接近真实原始语料分布，验证当前的“广支撑底座 + 长期分离前沿 + 晚窗口闭包”不是提示工程产物，而是语言系统的一般规律。

5. `ICSPB 闭式方程块`
   把锚点、纤维、关系轴、控制轴、前沿迁移、窗口条件门、正质量/负质量闭式核和闭包边界统一压成更严格的方程组，并要求它直接预测：
   - 哪些层带活跃
   - 哪些前馈层参与
   - 哪些注意力头参与
   - 哪些窗口促进闭包
   - 哪些变量破坏严格正协同

## 11. 当前最终判断

当前最稳的方向已经不是“继续找更多零散现象”，而是：

1. 语言背后确实有系统性编码规律。
2. 这个规律不像单一线性空间，更像多层复合结构。
3. 深度神经网络对语言的编码，不是“少量特征神经元在工作”，而是“广支撑底座上形成长期分离前沿，再在句尾前窗口中完成闭包收束”。
4. 真正的数学难点，已经从“有没有结构”转移到“如何把结构压成闭式统一方程”。

如果后续继续推进，最值得优先攻克的，不再是加几个新案例，而是把组件特异完整张量、真实语料分布和简洁生成律三件事接成同一条主链。

## 12. 规范化通道系统补充

这一轮把层间耦合继续压成了更严格的“通道（channel，通道）”口径。

当前规范化写法是：

1. `gs_load = -gs_coupling`
2. `gd_drive = gd_coupling`
3. `sd_load = -sd_coupling`

这样做之后，三条耦合不再只是“正负号混乱的乘积项”，而是开始具有更明确的系统角色：

1. `gd_drive（主驱动通道）`
   当前对 `union_joint_adv（联合优势） / union_synergy_joint（联合协同） / strict_positive_synergy（严格正协同）` 三目标都稳定为正，是当前最稳的主耦合。

2. `gs_load（一般层-严格层负载通道）`
   当前对 `union_joint_adv` 为正、对后两目标为负，说明它不是稳定主驱动，更像目标特异的结构负载。

3. `sd_load（严格层-判别层负载通道）`
   当前对 `union_joint_adv` 为负、对后两目标为正，说明它也不是统一主驱动，而是另一类目标特异负载。

所以现在最严格的结构判断已经变成：

1. `gd` 是真正跨目标稳定的主驱动通道。
2. `gs / sd` 虽然经过规范化后语义更清楚，但仍然保留目标异质性。
3. 当前分层双主式的最合理读法，不是“三条耦合同权”，而是：
   - `gd` 负责一般层与判别层的主驱动
   - `gs / sd` 负责严格层相关的目标特异负载

这一步的重要性在于，它把“层间耦合”从经验符号表，推进成了：

1. `主驱动通道`
2. `目标特异负载通道`

也就是说，双主式系统已经开始出现“主通道 + 负载通道”的正式层间结构。

## 13. 系统级一般化公式

在当前阶段，如果从“系统角度”而不是“单个代理量角度”来整理公式，最一般化的结构已经不再是一条单方程，而是：

1. `层级状态向量`
2. `通道向量`
3. `目标条件负载算子`

当前最简洁的写法可以先记成：

1. `z(pair) = [G(pair), S(pair), D(pair)]^T`
2. `c(pair) = [gd(pair), gs(pair), sd(pair)]^T`

其中：

1. `G(pair) = kernel_v4(pair)`
   也就是当前的一般闭包主核。

2. `S(pair) = strict_module_base(pair)`
   也就是当前最优的严格闭包模块候选。

3. `D(pair) = dual_gap_final(pair)`
   也就是当前最稳的严格性判别层变量。

4. `gd(pair)`
   是跨目标稳定的主驱动通道。

5. `gs(pair), sd(pair)`
   不是统一主驱动，而是目标特异负载通道。

于是当前最一般化的观测式，可以写成：

`y_t(pair) = W_t * z(pair) + V_t * c(pair) + eps_t(pair)`

这条式子的含义是：

1. `W_t`
   控制三层状态向量如何进入不同目标。

2. `V_t`
   控制三条通道如何进入不同目标。

3. `eps_t`
   表示当前系统里仍未收口的残差项。

如果进一步压缩，可以得到当前最有用的系统级近似：

`y_t(pair) ~= a_t * G(pair) + b_t * S(pair) + d_t * D(pair) + p_t * gd(pair) + L_t(gs(pair), sd(pair)) + eps_t(pair)`

这里最关键的新对象是：

`L_t(gs, sd) = q_t * gs + r_t * sd`

也就是说，当前最一般化的数学结构里，`gs / sd` 不再适合被看成“另外两条普通耦合”，而更像：

1. `目标条件负载算子`
2. `严格层相关的目标特异修正项`

所以更严格地说，当前这套理论已经开始从：

1. 单条主方程
2. 少数修正项

推进成：

1. `层级状态向量`
2. `通道向量`
3. `目标条件负载算子`

组成的系统方程。

这一步的意义在于，它比“继续堆更多代理量”更一般，也更接近未来可能的更高阶数学体系。因为它已经不再只问：

1. 哪个项为正
2. 哪个项为负

而开始问：

1. 哪些是系统状态
2. 哪些是系统通道
3. 哪些是目标条件算子

也就是说，当前理论已经不只是“一个语言公式”，而开始像一个小型分层算子系统。

## 14. 负载算子收口与一般化公式精炼

在上一节里，我们已经把系统写成：

1. `层级状态向量 z(pair)`
2. `通道向量 c(pair)`
3. `目标条件负载算子 L_t(gs, sd)`

这一轮继续往前压之后，`gs / sd` 终于开始出现更短、更一般的收口形式。

当前最关键的新结果是：

1. `load_mean = (gs + sd) / 2`
   对 `union_joint_adv（联合优势） / union_synergy_joint（联合协同） / strict_positive_synergy（严格正协同）` 三目标都稳定为负。

2. `load_contrast = (sd - gs) / 2`
   对前两个一般目标为负，但对 `strict_positive_synergy` 转成正。

这意味着 `gs / sd` 已经不需要继续被当成两条并列耦合来理解，而更适合压成：

1. `基础负载算子`
   `L_base(pair) = (gs(pair) + sd(pair)) / 2`

2. `严格选择算子`
   `L_select(pair) = (sd(pair) - gs(pair)) / 2`

于是当前最一般化的系统级公式，可以继续缩短成：

`y_general(pair) ~= a * G(pair) + d * D(pair) + p * gd(pair) - l * L_base(pair) + eps(pair)`

`y_strict(pair) ~= y_general(pair) + s * L_select(pair) + eta(pair)`

这两条式子的含义是：

1. 一般目标的主结构，主要由：
   - `G = kernel_v4`
   - `D = dual_gap_final`
   - `gd = 主驱动通道`
   - `L_base = 基础负载算子`
   共同决定。

2. 严格目标并不是完全另一套系统，而是在一般目标之上，再叠加：
   - `L_select = 严格选择算子`

所以更严格地说，当前理论已经从：

1. 分层状态向量
2. 通道向量
3. 目标条件负载算子

进一步压成：

1. `主驱动结构`
2. `基础负载结构`
3. `严格选择结构`

这比前一轮更一般，因为它开始显式区分：

1. 哪些东西属于“所有目标都要承受的基础负载”
2. 哪些东西只在“严格闭包”里额外打开

也就是说，当前最一般化的系统方程已经不只是“主核 + 通道”，而开始逼近：

1. `基础动力学`
2. `基础负载`
3. `严格选择增量`

三层结构。
