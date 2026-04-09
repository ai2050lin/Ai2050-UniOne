# AGI_GLM5_LANGUAGE

## 0. 文档定位

这份文档是"深度神经网络语言编码机制"研究的**系统性总览**。

核心目标：讲清楚——
1. 语言信息在网络内部如何编码
2. 已发现的数学结构是什么
3. 哪些结论跨模型稳健，哪些已被推翻
4. 当前瓶颈和下一步计划

**最核心的背景判断**：

1. 深度神经网络提取了某种高度压缩的语言数学结构
2. 这套结构不是"词典式存储"，而是"语义基底方向上的微小偏移"
3. 四模型（Qwen3/DS7B/GLM4/Gemma4）共享几何不变量，但编码策略模型特异
4. 研究路线：拼图积累 -> 稳定结构压缩 -> 因果打击 -> 跨模型复核

---

## 1. 研究原则

### 1.1 先拼图，后理论

1. 先积累可复核的实验拼图
2. 再压缩出稳定结构
3. 再做因果打击和跨模型复核
4. 最后才讨论第一性原理

**先机制，后定理；先还原，后抽象。**

### 1.2 先因果，后修辞

结论保留标准：
1. 能在原始激活或行为中直接看见
2. 能被消融或干预打中
3. 能跨样本复现
4. 能跨模型复现
5. 能被更强实验推翻或修正

### 1.3 先统一变量，后局部故事

1. 不把局部亮点误当成全局机制
2. 不把工程噪声误当成理论信号
3. 不把单个模型风格误当成普遍定律
4. 优先寻找统一控制变量

### 1.4 严格区分"描述"和"因果"

描述方程（精确描述发生了什么）≠ 因果方程（预测改变X后Y如何变）
当前已有精确描述方程，因果方程仍是薄弱环节。

---

## 2. 统一理论 v8.0（最新版本）

基于 P15-P65（Stage 662-707）+ Phase XV-XXXIX（Stage 720-744）全部实验结果。v7.0→v8.0新增：
- 非线性映射定理(定理7): 非线性仅集中在L0→L1! L1+后全部近线性(R2>0.87)
- 修饰方向复用定理(定理5.5): 修饰空间dim90=7维(三模型一致!)
- 二次/三次多项式完全无用: 简单多项式无法捕捉L0→L1非线性
- FFN输出与W_lm行cos≈0: FFN不直接映射到词汇
- DS7B最线性(R2=0.918), GLM4最非线性(R2=0.026)

基于 P15-P65（Stage 662-707）+ Phase XV-XXX（Stage 720-735）全部实验结果。v6.0→v7.0新增：
- 精细因果分析系统完成（Phase XXVII-XXX，P176-P194，4阶段×4实验）
- float32因果消融修复（KL从0→0.3~6.7）
- 概念分离关键层=L3，推理跃迁层模型特异
- 单头消融几乎无影响（Qwen3/GLM4 KL<1e-8），DS7B敏感80倍
- Embedding扰动Jacobian全层曲线（DS7B L4饱和~500K，Qwen3增长到77K）
- Jacobian拉伸指数增长，信息放大646-8057倍
- SAE训练确认不够稀疏（100%存活），需更大L1系数
- Qwen3头功能最分化（后期91%=retrieval）

### 2.1 五个公理

**第一公理：高维几何公理**（P28修正）
- 不同文本方向在高维空间趋向正交：E[|cos(Δ_l, h_l)|] ≈ O(1/√d)
- 需要d > d*（高维阈值，d* >> 512）
- 小模型(d=128): angle≈15°，不满足正交性
- 旋转角≈90°是高维空间的几何性质（INV-273, 4/4确认）

**第二公理：信息域公理**（P29/P35修正）
- 纠缠度 ≈ C · √(K_domain/d) · f(RL, data, arch)
- 单域模型（纯语言）→ SEPARATED策略
- 多域模型（语言+视觉+听觉）→ DENSE策略
- RL训练→能力融合（DS7B纠缠度最高，std最大）
- f(RL)使DS7B偏离理论预测16x

**第三公理：归一化隔离公理**
- 残存 ≈ (1-ρ_avg)^N_norms
- 归一化是隔离的核心机制（INV-305, 确认）
- Gemma4多模态模型的norm密度14.5/层(506个) >> 纯语言模型2-4/层

**第四公理：信号聚焦公理**（P28修正）
- 架构固有属性（训练开始即成立，聚焦比≈2.3）
- 后期层信号更强（INV-332, 4/4确认）
- 聚焦比 = cos_avg(后期)/cos_avg(前期) > 1.5

**第五公理：Logit精确公理**（P27确认）
- margin = ||h_L|| · ||Δw|| · cos(h_L, Δw)，误差<0.3%
- 数学恒等式（训练开始即成立）
- 之前"低估100倍"是方法论错误（avg_logprob vs 单步logit）

### 2.2 核心发现：旋转动力学框架（P44-P49修正）

> **重要修正**：P43的"语义基底方向"已被P44-P49实验**大幅修正**。

**P43原始结论**：语言编码在"语义基底方向"上通过微小偏移区分语义。

**修正后的理解**：

1. **L0"语义基底"= embedding质心方向**（P44/P45）：不是学习到的语义结构，是高维空间中embedding方向的自然趋同。tiny模型(d=256)上L0对齐仅0.747，大模型(d>>256)趋近1.0。

2. **L0→L1旋转~90度**（P47/P49）：>92%的旋转发生在L0→L1之间，是第一层transform的数学结果，不是逐层渐进的语义操作。

3. **旋转平面高度一致，但与语义无关**（P47/P48）：
   - 旋转方向跨文本一致（pairwise cos=0.76，INV-350确认）
   - 但旋转角度与文本类别无关（ANOVA F<2.1，INV-351推翻）
   - 旋转是架构特性，不是语义编码机制

4. **旋转仅需11维**（P47）：旋转平面PCA90=11，占总维度(d=1536-4096)的极小比例。

5. **信息瓶颈不是DS7B特有**（P46）：四模型PCA90范围相似(12-20)，瓶颈层在L1-L3而非"深度融合区"。

**修正后的核心图景**：

残差流的宏观几何（方向对齐、旋转角度、旋转平面）主要由**架构+初始化**决定，不由语义内容决定。真正的语义编码发生在**旋转之后的微调层**中——这些微调层仅贡献<8%的角度变化，但承载了所有语义区分信息。

### 2.3 方向+norm协同编码（P36）

- **方向承载几乎所有域区分信息**（Silhouette方向 > Norm）
- RMSNorm使norm域间差异仅2.5-2.7%（Qwen3/GLM4）
- DS7B例外：Norm域间差异19.5%，norm也携带大量信息

### 2.4 层动力学模型（P38）

| 模型 | 模式 | 域分离最强层 | 特征 |
|------|------|------------|------|
| Qwen3 | 倒拱形 | L8(0.14) | 早期峰值→缓慢衰减 |
| DS7B | U形 | L3(0.04) | L5-L26深度融合区(sil≈-0.21) |
| GLM4 | 平稳型 | L1(0.09) | 全程变化极小 |
| Gemma4 | U形(轻) | -- | 与DS7B类似但幅度小 |

### 2.5 信息瓶颈理论（P37/P65修正）

- DS7B: PCA90=13维（超级压缩，Top1方差45.7%）
- Qwen3: PCA90=45维（均匀分布）
- 压缩度与纠缠度正相关
- **P65修正**：P37的PCA90是在小样本下测量的。P65大样本确认：100文本下DS7B rank=89，其余rank=100。DS7B确实有真实的信息压缩，但维度远大于13（接近89维），13维的PCA90可能是旧版stage的测量限制。

### 2.6 多步累积方程（P34/P58修正）

- avg_logprob ≈ log(avg_margin) - C(task)，**C不是常数！**
- P58推翻了C≈1.5的普适性：C范围1.3-15.2
- 代码/数学C低(1-3)，通用英文C中(4-6)，中文C高(3-14)

### 2.7 注意力-方向耦合（P41）

- DS7B: 强耦合(r=0.528)——注意力分布变化与方向变化强正相关
- Qwen3/GLM4/Gemma4: 弱耦合(r≈0.06)
- GLM4后期自关注增加(6%→19%)
- Gemma4注意力最不稳定(CV=30.7%)

### 2.8 层敏感性地图（P42）

| 模型 | 最敏感层 | delta | 模式 |
|------|---------|-------|------|
| Qwen3 | L15 | -13.5 | 拱形 |
| DS7B | L15 | -8.2 | 中偏前 |
| GLM4 | L39 | -12.1 | 后期极敏感 |
| Gemma4 | -- | 0.00 | hook失效 |

---

## 3. 跨模型对比总览

### 3.1 四模型编码策略分类

| 维度 | Qwen3 | DS7B | GLM4 | Gemma4 |
|------|-------|------|------|--------|
| 信息域 | 纯语言 | 纯语言+RL | 纯语言 | 多模态 |
| 编码策略 | 弱SEPARATED | 极端DENSE | DENSE | DENSE(多模态) |
| Silhouette | 0.064(最高) | -0.065(最低) | 0.050 | -0.003 |
| PCA90 | 45维 | **13维** | 52维 | 52维 |
| 纠缠度std | 0.065 | **0.206** | 0.083 | 0.125 |
| 方向对齐 | 0.934 | 0.903 | 0.788 | **0.972** |
| 基底捕获 | 93.8% | 92.1% | 81.6% | 89.3% |
| Norm CV | 3.3% | 8.0% | 3.9% | **22.5%** |
| Attn-Dir r | 0.057 | **0.528** | -- | -- |

### 3.2 Gemma4"例外"的统一解释（P26-P31多模态假说）

Gemma4不是"编码更差"，而是"编码不同"——它是多模态模型：

| 现象 | 多模态假说的解释 |
|------|----------------|
| 抵消率>90% | 多模态信号在共享空间中必然高抵消 |
| 信号效率13% | 大量维度用于视觉/听觉编码 |
| PCA90较高(16-22维) | 需编码视觉特征+语言特征 |
| Norm密度14.5/层(506个) | 模态内+模态间归一化 |
| 鲁棒性极高(3.86) | 多模态冗余带来的天然鲁棒性 |
| 方向对齐最高(0.972) | 多模态训练要求更强的基底对齐 |
| d_norm不增长(1.1x) | delta_angle=119°导致47.7%能量抵消 |

### 3.3 DS7B"深度融合区"（P38/P42/P65确认）

- L5-L26共22层处于持续DENSE状态(sil≈-0.21)
- 恰好是推理能力最强的区域
- RL训练创造了"深度融合区"，让不同能力深度交互
- 中间层平均敏感性(-7.0) >> 边缘层(-2.3)（P42确认）
- 信息瓶颈：PCA90=13维，Top1=45.7%（P37）
- **P65新确认**：100文本下DS7B是唯一不满秩的模型(rank=89)，其余三模型rank=100。这证实DS7B确实在内部压缩了信息，与其DENSE编码策略和RL训练一致。

---

## 4. 不变量体系（三层分类）

### 4.1 第一层：几何不变量（所有模型，不依赖训练）

| 不变量 | 内容 | 确认率 | 本质 |
|--------|------|--------|------|
| INV-286 | z-score>1.96（抵消有序） | 4/4 | 中心极限定理 |
| INV-273 | 旋转角≈90° | 4/4 | 高维几何 |

### 4.2 第二层：信息域不变量（单域模型上成立）

| 不变量 | 内容 | 单域确认 | 多域确认 |
|--------|------|---------|---------|
| INV-293 | 抵消率30-90% | 3/3 | 0/1(Gemma4>90%) |
| INV-285 | 信号效率>50% | 3/3 | 0/1(Gemma4 13%) |
| INV-305 | 归一化=隔离核心 | 4/4 | 4/4 |
| INV-306 | 归一化自适应压缩 | 4/4 | 4/4 |
| INV-308 | 信息域决定策略 | 4/4 | 4/4 |
| INV-313 | 多步信号放大 | 4/4 | 4/4 |
| INV-314 | 首token锚定 | 4/4 | 4/4 |
| INV-315 | 维度分布式贡献 | 4/4 | 4/4 |
| INV-332 | 后期信号更强 | 4/4 | 4/4 |

### 4.3 第三层：编码机制不变量（部分确认）

| 不变量 | 内容 | 判定 | 备注 |
|--------|------|------|------|
| INV-338 | DS7B mid层敏感 | ✅ | P42确认 |
| INV-339 | L0对齐>最终层 | ✅(反转) | 原预测反了 |
| INV-333 | Δ/h<0.1残差约束 | ❌ | Δ/h=0.64-1.46 |
| INV-320 | 难度↑→纠缠↑ | ❌ | 不同模型不一致 |
| INV-321 | 密度↑→纠缠↑ | ❌ | 不同模型不一致 |
| INV-326 | Norm区分力>方向 | ❌(反转) | 方向>Norm |
| INV-331 | 中间层域分离最低 | ⚠️ | 仅DS7B/Gemma4确认 |
| INV-322 | avg_lp≈log(margin)-C | ⚠️修正(P58) | C=1.5被推翻，C=1.3-15.2任务依赖 |

### 4.4 已推翻的重大结论

| 原结论 | 推翻时间 | 推翻原因 |
|--------|---------|---------|
| 80%歧义词走attention路径 | P26(stage573) | GLM4=0%, Gemma4=10% |
| 维度坍缩在15-17%层 | P25(stage577) | 单token秩=1是数学限制 |
| bank走attention, apple走MLP | P26(stage573) | 二元对立被打破 |
| 编码距离矩阵跨架构一致 | P25(stage548) | Qwen3-Gemma4 r=0.03 |
| 维度坍缩是突变 | P25(stage577) | 单token固有秩=1，无坍缩 |
| Norm区分力>方向 | P37(stage682) | 方向Silhouette > Norm |
| 残差约束导致方向对齐 | P41(stage686) | Δ/h远>0.1 |
| INV-220推理=低维编码 | P22(stage667) | 恢复率<4% |
| INV-296门控模式 | P22 | 恢复率1% |
| INV-124正交性=隔离 | P21 | Pearson r≈0 |
| 隐藏→logit分段非线性 | P28(stage593) | 末层完美线性cos>0.9999 |
| 家族结构在L0即完备 | P29(stage590) | LOO在L0即达峰值，但cos=1.0是伪像 |

---

## 5. 关键硬伤与瓶颈

### 5.1 当前最关键的未解决问题

1. ~~"语义基底方向"的存在性和唯一性未严格证明~~ → 已修正：=embedding质心(P44)，非语义结构
2. **修正因子f(RL, data, arch)的数学形式未知**——纠缠度公式中最大的未知项
3. **多步生成方程缺失**——能精确描述单步决策(margin)，但不能预测多步生成过程
4. **RL训练如何改变"基底选择"策略**——DS7B的深度融合区是RL创造的，但机制不明
5. **跨架构编码拓扑不一致**——Qwen3-Gemma4编码距离r=0.03，不同架构几乎是不同的"语言"
6. **因果效应量偏小**——跨任务因果子集效用0.001-0.025，方向正确但效应弱
7. **推理能力分布式编码**——恢复率<4%，无法用低维捕捉
8. **文本→方向偏移的可预测映射缺失**——P54 logit r=0.78-0.90但margin r≈0
9. **真实有效维度未知**——P65确认>89-100(100文本)，需要更大样本(>1000)精确定位
10. ~~语义方向的因果验证缺失~~ → P176 float32修复后KL=0.3~6.7，因果验证部分完成
11. **跨模型语义方向对齐未知**——不同模型的PC0是否编码相同语义？P67待修复
12. **SAE稀疏度不足**——λ=1e-3太低，100%特征存活，需要λ=0.01-0.1或TopK替代
13. **多头联合消融未做**——单头KL极小，需要top5/top10联合消融才能测出累积效应
14. **信息放大倍数差异巨大**——Qwen3=646x, DS7B=饱和, GLM4=8057x, 物理机制不明
15. **DS7B灵敏度L4饱和的原因**——RL训练导致早期层信息耗尽还是非线性饱和？

### 5.2 方法论限制

1. **Gemma4 hook机制不兼容**——架构差异导致部分消融实验无法执行
2. **GPU内存限制**——四模型必须串行测试
3. **单token vs 多token的方法论鸿沟**——avg_logprob vs 单步logit差7.4x
4. **高维阈值d*未精确定位**——已知d*>512，但精确值未知
5. **简单消融对RMSNorm模型无效**——GLM4/DS7B的norm放大量大，直接替换h产生非物理结果
6. **P50的ANOVA F>3阈值偏宽松**——9类别4样本的自由度下，F(8,27)的p<0.05阈值约2.3，F>3约p<0.01
7. **有效维度被样本数限制**——P61报告"rank=9"仅因10条文本上限，P65确认100文本时rank=89-100
8. **跨模型hidden_dim不同**（1536/2560/3584/4096）——无法直接比较方向向量，需要CCA或类似方法
9. **GBK编码问题**——GLM4/Gemma4的token解码在某些Unicode字符上失败

### 5.3 操控研究核心发现（Phase XV-XVIII，Stage 720-723）

Phase XV-XVIII系统性地研究了"能否通过修改hidden state来操控模型输出"的问题，在三个模型（GLM4/Qwen3/DS7B）上进行了28个实验。

**操控方法的系统测试**:

| 方法 | Phase | 核心结果 | 效果 |
|------|-------|---------|------|
| Centroid差方向注入 | XV | cos_shift<0, 远离目标 | 弱 |
| Fisher/LDA方向 | XVI | rank_improve 62-88%, 但cos_shift仍<0 | 中 |
| PC0主成分操控 | XVI | GLM4 1-cos=0.056 (最大) | 中 |
| 前半层注入 | XVI | 效果是后半层3-28x | 验证层级结构 |
| 精确梯度方向 | XVII | cos(grad,centroid)≈0.17, 不同目标梯度cos>0.94 | 强 |
| 跨层差异化注入 | XVIII | 所有策略效果完全相同 | **无效** |
| Logit空间直接操控 | XVIII | tpc=+0.013, 效果是hs的11.5倍 | **唯一有效** |
| Norm bypass | XVIII | 三种策略差异<5% | 弱 |
| Hessian对角线 | XVIII | GLM4 99%为负, Qwen3 100%为正 | **根本发现** |
| PGD对抗优化 | XVIII | 多步与单步相同, 无改善 | **无效** |

**三个核心方程**:

1. **操控不可能三角**: 方向性-质量-成功率不可能同时满足。高方向性需要大scale(>0.30)，但大scale破坏质量(1-cos>0.5)。scale∈[0.08,0.15]是"甜蜜窗口"但方向性弱。

2. **Hessian符号方程**: GLM4的log P(target|h)是局部凹函数（99% H对角线<0），Qwen3是局部凸函数（100% H>0），DS7B是混合符号。这解释了为什么GLM4操控后总是远离目标——它在"倒坡"上。

3. **范数指数稀释**: ||h_L||/||h_0|| = 699x(Qwen3)~3234x(GLM4)，外部注入信号被指数级稀释。DS7B的L5 norm(68.65)已超过GLM4 L39(248)的27%。

**已验证的结论**:
- 表征空间(centroid/fisher方向)与操控空间(梯度方向)弱正交(cos≈0.17)
- 操控子空间维度: DS7B(2维50%方差) << GLM4(5维50%方差)
- 累积注入近似线性(GLM4: ~0.002/层)，DS7B完全拮抗
- 最后3层注入无效，前半层>>后半层
- logit空间操控是唯一有效入口
- **P52"线性累积"被Phase XXI推翻**: 中间cos=0.15~-0.64, P144正确
- **后续层修复94~97%消融效果**: 直接消融vs传播消融ratio=16~39x
- **最后1层断裂**: 相邻层cos从>0.93骤降到0.09~0.47(信息质变层)

**Phase XIX新发现（Logit/Token/Unembed/Attention操控）**:
- boost_adaptive是最优logit操控策略(tpc=0.77~0.94, KL≈3.0)
- boost_scaled_20达到tpc饱和(>0.93), 继续增加scale无收益
- suppress策略适得其反(压制源类别降低目标概率)
- prefix操控是"自然操控"(h_shift=0.01~0.10), 保持输出质量
- self-generation方向与文本centroid无本质区别
- Unembed矩阵判别维度极少(前5维F-score即显著)
- 三模型(GLM4/Qwen3/DS7B)共享操控响应模式, 但GLM4需要更大boost力度

**Phase XX新发现（消融方法革新）**:
- **减法消融(维度/方向移除)top1_chg≈100%** — 远超加法操控的0%(三模型确认)
- 信息高度冗余: 零化1维 = 零化500维(KL饱和)
- 方向移除效应与方向无关: centroid/PC/随机方向KL≈19(相同)
- 信息在最后1层急剧收敛: 前N-1层cos<0.3 → 最后层cos>0.9999
- 多步操控衰减: tpc每步衰减3-5%, 前5步后效果减半
- Token因果追踪: 中间token贡献>首位token>末位token

**Phase XXI新发现（消融精细化+矛盾解决）**:
- **P52"线性累积cos>0.995"被推翻!** — 实测中间cos=0.15~-0.64(GLM4=-0.64)
- **后续层修复94~97%的消融效果** — 直接消融vs传播消融ratio=16~39x
- **早期层维度完全不贡献logit信息** — 缩放10维KL变化<0.03
- **最后1层断裂** — 相邻层cos从>0.93骤降到0.09~0.47(三模型一致)
- **保留全部维度也无法维持top-1** — h_l@W^T ≠ logits_final(层间传播改变分布)
- **GLM4 norm膨胀2923x是极端异常** — 远超Qwen3(143x)和DS7B(161x)

**Phase XXIII新发现（语言流形几何测量）**:
- **残差流路径高度弯曲** — 步角70~84°(近正交), 路径比0.745~0.830(25%弯路)
- **56~66%的delta_h正交于h_l** — 残差连接=方向旋转而非幅度缩放
- **GLM4 Menger曲率异常高** — 0.374 vs Qwen3(0.040), 弯曲极端
- **GLM4校正方向远离logits** — cos(delta@W^T, logits)=-0.40
- **操控方向无关** — tangent/random/orthogonal KL几乎相同(ratio≈1.0)
- **10维切空间只捕获66~88%** — 流形高维, 中间层膨胀到31~55%

**Phase XXV新发现（概念探测+属性编码）**:
- **句子级概念间cos=0.94~0.99** — 不同概念高度相似(但token级有区分)
- **属性间cos=0.89~0.999** — 属性不是独立维度, 是概念主方向的微小扰动
- **GLM4最后层抽象层级有序** — apple-fruit(0.89)>apple-entity(0.78)
- **词嵌入类比在h中完全不成立** — king+queen-man≠woman(accuracy=0%)

**Phase XXIX新发现（Jacobian动力学+注意力头+信息流追踪）**:
- **Jacobian拉伸指数增长** — Qwen3: L0≈0.6 → L33≈64.5, 后期层是信息放大器
- **极端专注头普遍存在** — Qwen3每层1头(ent≈0.004), DS7B早期43%, GLM4早期13%
- **后期层注意力均匀化** — entropy>1.5的头从早期30%增至后期66%
- **Gamma深度增长差异大** — Qwen3(×184) > GLM4(×133) >> DS7B(×4.4)
- **L0是信息瓶颈** — Qwen3保留仅3%, DS7B/GLM4保留11-32%
- **层间信息流相互抵消** — final_cumulative≈0, 正负交替
- **F-stat跨模型<12** — 即使gamma空间, 单维度仍不能判别8类别

**Phase XXVIII新发现（跨模型全层追踪+SAE分解+单维度语义+概念分离层）**:
- **三模型概念分离层一致=早期(L3-L8)** — cos≈0.40, 这是语言通用编码阶段
- **cos(EMB_dir, HS_last)≈0.02** — 三模型一致, embedding方向被完全重写
- **DS7B中间层Top1=98.7%** — RL创造的极端信息压缩, 几乎1维编码
- **GLM4分布更均匀(Top1=10-16%)** — 与DS7B形成鲜明对比
- **语义编码高度分布式** — F-stat<3.5, 没有语义主维度, 每类别用不同维度
- **稀疏度低(0.22-0.33)** — PCA+Varimax不够稀疏, 需要真正的SAE
- **推理跃迁层模型特异** — Qwen3=L34, DS7B=L27, GLM4=L39
- **最后层概念融合** — sep=-0.012~-0.023, 类别内>类别间(但差距极微弱)

**Phase XXVII新发现（float32因果消融+全层EMB→HS+逻辑推理+属性因果+概念图谱）**:
- **float32修复了KL=0!** — bfloat16是P173因果消融失败的根本原因; float32 KL=0.3~6.7
- **概念分离关键层=L3** — cos=0.446; 最差层L15-L20(cos=0.78); 最后层恢复cos=0.70
- **推理跃迁层一致=L34** — 所有21个推理文本的最后第2层; 实际推理准确率86%
- **属性因果效应中等** — KL(modified)=0.43, KL(ablated)=0.20; 某些上下文为零
- **类别分离仅0.041** — intra-cos=0.735 vs inter-cos=0.694; 所有层sep<0
- **DS7B因果敏感性最高** — KL max=6.68("dog"), 远超Qwen3(1.35)
- **cos(EMB_dir, HS_last)约0.04** — embedding方向被Transformer完全重写

**Phase XXVI新发现（Token级概念分析）**:
- **Token级概念间cos=0.50~0.71** — 远优于句子级0.94~0.99, 有明确区分!
- **Embedding空间概念几乎正交** — cos=0.04~0.08(独立编码)
- **Transformer将独立编码变为分布式编码** — EMB cos=0.05→HS cos=0.63
- **属性token级cos=0.75~0.80** — 比句子级改善但仍不独立
- **概念token获得21~41%注意力** — GLM4最高(41%), Qwen3最低(26%)
- **因果消融KL=0** — bfloat16精度不足, 需float32重新验证

**Phase XIX新发现（Logit/Token/Unembed/Attention操控）**:
- boost_adaptive是最优logit操控策略(tpc=0.77~0.94, KL≈3.0)
- boost_scaled_20达到tpc饱和(>0.93), 继续增加scale无收益
- suppress策略适得其反(压制源类别降低目标概率)
- prefix操控是"自然操控"(h_shift=0.01~0.10), 保持输出质量
- self-generation方向与文本centroid无本质区别
- Unembed矩阵判别维度极少(前5维F-score即显著)
- 三模型(GLM4/Qwen3/DS7B)共享操控响应模式, 但GLM4需要更大boost力度

**Phase XX新发现（消融方法革新）**:
- **减法消融(维度/方向移除)top1_chg≈100%** — 远超加法操控的0%(三模型确认)
- 信息高度冗余: 零化1维 = 零化500维(KL饱和)
- 方向移除效应与方向无关: centroid/PC/随机方向KL≈19(相同)
- 信息在最后1层急剧收敛: 前N-1层cos<0.3 → 最后层cos>0.9999
- 多步操控衰减: tpc每步衰减3-5%, 前5步后效果减半
- Token因果追踪: 中间token贡献>首位token>末位token

**Phase XXI新发现（消融精细化+矛盾解决）**:
- **P52"线性累积cos>0.995"被推翻!** — 实测中间cos=0.15~-0.64(GLM4=-0.64)
- **后续层修复94~97%的消融效果** — 直接消融vs传播消融ratio=16~39x
- **早期层维度完全不贡献logit信息** — 缩放10维KL变化<0.03
- **最后1层断裂** — 相邻层cos从>0.93骤降到0.09~0.47(三模型一致)
- **保留全部维度也无法维持top-1** — h_l@W^T ≠ logits_final(层间传播改变分布)
- **GLM4 norm膨胀2923x是极端异常** — 远超Qwen3(143x)和DS7B(161x)

**Phase XXII新发现（精确Activation Patching+流形投影）**:
- **真实patching ratio=4~509x** — Qwen3/GLM4早期层修复极强(509x/135x), DS7B弱(4~7x)
- **RMSNorm不是纯缩放!** — cos(h, RMSNorm(h))=0.46~0.57(最后层), gamma各维不同改变了方向
- **三模型最后层机制完全不同** — GLM4的RMSNorm是断裂主因, Qwen3/DS7B是LM Head
- **DS7B因果流与其他模型相反** — 中间层最敏感(非后期层)
- **早期层零化几乎不影响最终输出** — cos_h_final(L0)>0.98

**Phase XXIII新发现（语言流形几何测量）**:
- **残差流路径高度弯曲** — 步角70~84°(近正交), 路径比0.745~0.830(25%弯路)
- **56~66%的delta_h正交于h_l** — 残差连接=方向旋转而非幅度缩放
- **GLM4 Menger曲率异常高** — 0.374 vs Qwen3(0.040), 弯曲极端
- **GLM4校正方向远离logits** — cos(delta@W^T, logits)=-0.40
- **操控方向无关** — tangent/random/orthogonal KL几乎相同(ratio≈1.0)
- **10维切空间只捕获66~88%** — 流形高维, 中间层膨胀到31~55%

**Phase XXV新发现（概念探测+属性编码）**:
- **句子级概念间cos=0.94~0.99** — 不同概念高度相似(但token级有区分)
- **属性间cos=0.89~0.999** — 属性不是独立维度, 是概念主方向的微小扰动
- **GLM4最后层抽象层级有序** — apple-fruit(0.89)>apple-entity(0.78)
- **词嵌入类比在h中完全不成立** — king+queen-man≠woman(accuracy=0%)

**Phase XXVI新发现（Token级概念分析）**:
- **Token级概念间cos=0.50~0.71** — 远优于句子级0.94~0.99, 有明确区分!
- **Embedding空间概念几乎正交** — cos=0.04~0.08(独立编码)
- **Transformer将独立编码变为分布式编码** — EMB cos=0.05→HS cos=0.63
- **属性token级cos=0.75~0.80** — 比句子级改善但仍不独立
- **概念token获得21~41%注意力** — GLM4最高(41%), Qwen3最低(26%)
- **因果消融KL=0** — bfloat16精度不足, 需float32重新验证

**Phase XXVII新发现（float32因果消融+全层EMB→HS+逻辑推理+属性因果+概念图谱）**:
- **float32修复了KL=0!** — bfloat16是P173因果消融失败的根本原因; float32 KL=0.3~6.7
- **概念分离关键层=L3** — cos=0.446; 最差层L15-L20(cos=0.78); 最后层恢复cos=0.70
- **推理跃迁层一致=L34** — 所有21个推理文本的最后第2层; 实际推理准确率86%
- **属性因果效应中等** — KL(modified)=0.43, KL(ablated)=0.20; 某些上下文为零
- **类别分离仅0.041** — intra-cos=0.735 vs inter-cos=0.694; 所有层sep<0
- **DS7B因果敏感性最高** — KL max=6.68("dog"), 远超Qwen3(1.35)
- **cos(EMB_dir, HS_last)约0.04** — embedding方向被Transformer完全重写

**Phase XXVIII新发现（跨模型全层追踪+SAE分解+单维度语义+概念分离层）**:
- **三模型概念分离层一致=早期(L3-L8)** — cos≈0.40, 这是语言通用编码阶段
- **cos(EMB_dir, HS_last)≈0.02** — 三模型一致, embedding方向被完全重写
- **DS7B中间层Top1=98.7%** — RL创造的极端信息压缩, 几乎1维编码
- **GLM4分布更均匀(Top1=10-16%)** — 与DS7B形成鲜明对比
- **语义编码高度分布式** — F-stat<3.5, 没有语义主维度, 每类别用不同维度
- **稀疏度低(0.22-0.33)** — PCA+Varimax不够稀疏, 需要真正的SAE
- **推理跃迁层模型特异** — Qwen3=L34, DS7B=L27, GLM4=L39
- **最后层概念融合** — sep=-0.012~-0.023, 类别内>类别间(但差距极微弱)

**Phase XXIX新发现（Jacobian动力学+注意力头+信息流追踪）**:
- **Jacobian拉伸指数增长** — Qwen3: L0≈0.6 → L33≈64.5, 后期层是信息放大器
- **极端专注头普遍存在** — Qwen3每层1头(ent≈0.004), DS7B早期43%, GLM4早期13%
- **后期层注意力均匀化** — entropy>1.5的头从早期30%增至后期66%
- **Gamma深度增长差异大** — Qwen3(×184) > GLM4(×133) >> DS7B(×4.4)
- **L0是信息瓶颈** — Qwen3保留仅3%, DS7B/GLM4保留11-32%
- **层间信息流相互抵消** — final_cumulative≈0, 正负交替
- **F-stat跨模型<12** — 即使gamma空间, 单维度仍不能判别8类别

**Phase XXX新发现（头因果消融+Embedding扰动Jacobian+SAE+头功能聚类）**:
- **单头消融几乎无影响** — Qwen3/GLM4: KL<1e-8, top1_change<4%; DS7B敏感80倍
- **DS7B灵敏度L4饱和** — L4即达~500K, 后续层不再增长; Qwen3增长到77K; GLM4到100K
- **Qwen3 L35灵敏度骤降** — 最后层灵敏度下降3.5倍(output projection截断)
- **GLM4灵敏度最稳定增长** — 从12(L0)到100K(L39), 持续8000倍增长
- **SAE不够稀疏** — 所有特征100%存活, Qwen3 L17激活199/512, GLM4 L10激活247/512
- **Qwen3头功能最分化** — 后期91%=retrieval; DS7B/GLM4几乎全mixed

---

## 6. 已完成阶段总结与下一步计划

### 6.1 已完成（P44-P65结果摘要）

**P44（跨模型基底对齐）**→ 基底=embedding质心方向，四模型L0对齐=1.0但互相不同
**P45（训练动力学）**→ L0对齐0.747从随机初始化就存在，大模型趋近1.0是高维效应
**P46（信息瓶颈）**→ 非DS7B特有，四模型PCA90=12-20相似
**P47（旋转平面）**→ 11维旋转平面跨文本一致(cos=0.76)，旋转~90度
**P48（旋转语义）**→ 旋转角度与类别(F<2.1)和长度(|r|<0.28)无关
**P49（RL旋转效应）**→ >92%旋转在L0→L1，DS7B无特殊
**P50（微调层语义定位）**→ ANOVA F>3，35/36层有语义编码
**P51（因果消融）**→ 早期层消融margin下降95-99%
**P52（线性累积）**→ logits=sum(delta_h@U.T), cos>0.995 ← **Phase XXI推翻: 实测cos=0.15~-0.64**
**P53（单token贡献）**→ 敏感层分布因模型而异
**P54（预测模型）**→ logit r=0.78-0.90, margin r≈0
**P55（Norm因果）**→ 方向完全决定语义，P1确认
**P56（旋转干预）**→ K=1维保持top-1，P2确认
**P58（任务依赖）**→ C=1.3-15.2，P4推翻
**P60（基元消融）**→ E_gate>E_subspace>A_direction, C/D无影响
**P61/P63（信息容量）**→ "rank=9"是样本限制(P64推翻)
**P62（组合基元）**→ A+E无协同，E单独主导
**P64（第一性原理）**→ h⊥unembed, PC0=语言类型, rank=N-1
**P65（大样本维度）**→ 100文本rank=89-100, DS7B独特压缩

### 6.2 三大战略任务（详细计划见 `AGI_GLM5_NEXT_PHASE_PLAN.md`）

**大任务一：生成化（P50-P54）✅ 已完成**
- 核心：从"事后解释"到"事前预测"
- 结果：text→delta→logit链在原理上可行(P52线性累积精确)，但delta预测精度不足

**大任务二：判伪化（P55-P59）✅ 已完成（P57/P59待外部依赖）**
- 核心：为v4.1设定明确推翻条件
- 结果：3/5命题已验证（P1/P2确认，P4推翻），2个待执行（P57/P59）

**大任务三：基元化（P60-P64）✅ 已完成**
- 核心：确定最小编码基元
- 结果：编码基元=方向+门控幅度(门控主导)，rank=9被推翻(大样本后>89-100)

**大任务四：操控研究（Phase XV-XVIII）✅ 已完成**
- 核心：测试能否通过修改hidden state操控模型输出
- 结果：操控不可能三角被确认——所有方法(cos_shift<0)。Hessian符号反转是根本发现。Logit空间是唯一有效入口。
- 脚本：`stage720_phase15.py` ~ `stage723_phase18.py`

**Phase IV：大样本验证（P65-P67）⏳ 部分完成**
- 核心：用大样本重新评估有效维度
- 结果：100文本rank=89-100，Top-5 PC解释82-90%方差

**Phase XIX：Logit/Token/Unembed/Attention操控（P136-P140）✅ 已完成**
- 核心：放弃hidden state操控, 转向logit/token/unembed/attention全新入口
- 结果：boost_adaptive最优(tpc=0.77~0.94), suppress策略无效, prefix是自然操控
- 脚本：`stage724_phase19.py`

**Phase XXI：消融精细化+P52矛盾验证（P146-P150）✅ 已完成**
- 核心：层传播消融+渐进维度+最小维度集+P52/P144矛盾彻底解决
- 结果：P52被推翻(cos=0.15~-0.64), 后续层修复94~97%消融效果, 最后1层断裂
- 脚本：`stage726_phase21.py`

**Phase XXII：精确Activation Patching+流形投影（P151-P155）✅ 已完成**
- 核心：register_forward_hook精确修改+传播, RMSNorm几何分析, 因果流精确图
- 结果：真实ratio=4~509x, RMSNorm不是纯缩放(cos=0.46~0.57), 三模型最后层机制完全不同
- 脚本：`stage727_phase22.py`

**Phase XXIII：语言流形几何测量（P156-P160）✅ 已完成**
- 核心：残差流曲率+校正方向+类别流+操控对比+切空间投影
- 结果：步角70~84°(近正交), 56~66%的delta正交于h, 10维切空间只捕获66~88%
- 脚本：`stage728_phase23.py`

**Phase XXIV：K维扫描+参与比+流形平滑度（P161-P165）✅ 已完成**
- 核心：K维饱和扫描, 参与比维度, 流形平滑度, 层级结构
- 结果：PR=3~19维, K=200仍不饱和(中间层38~60%), DS7B最不平滑
- 脚本：`stage729_phase24.py`

**Phase XXV：概念探测+属性编码+抽象层级（P166-P170）✅ 已完成**
- 核心：概念共享方向, 属性编码, 抽象层级几何, 词嵌入类比验证, 概念正交性
- 结果：概念cos=0.88~0.99(句子级), 属性cos=0.89~0.999(非独立维度), 类比accuracy=0%
- 脚本：`stage730_phase25.py`
- 注：句子级结论在Phase XXVI被修正

**Phase XXVI：Token级概念分析+W_E对比（P171-P175）✅ 已完成**
- 核心：Token级概念cos, 属性替换差异, 因果消融, 注意力模式, Embedding vs Hidden State
- 结果：token级inter-cos=0.50~0.71, Emb cos=0.04~0.08, HS cos=0.63
- 脚本：`stage731_phase26.py`
- 核心修正：Phase XXV"概念不区分"是句子级artifact, token级有明确区分

**Phase XXVII：float32因果消融+逻辑推理追踪+属性因果+概念图谱（P176-P180）✅ 已完成**
- 核心：float32精度修复+全层EMB→HS变换+逻辑推理逐层追踪+属性因果+大规模概念图谱
- 结果：float32修复KL从0→0.3~6.7; 概念分离关键层=L3; 推理跃迁层=L34
- 脚本：`stage732_phase27.py`

**Phase XXVIII：跨模型全层追踪+SAE分解+单维度语义+概念分离层（P181-P185）✅ 已完成**
- 核心：跨模型EMB→HS追踪+推理跃迁验证+SAE稀疏分解+单维度语义+概念分离层
- 结果：三模型概念分离一致(L3-L8); DS7B中间层Top1=98.7%; F<3.5无语义主维度
- 脚本：`stage733_phase28.py`

**Phase XXIX：Jacobian动力学+注意力头+信息流追踪（P186-P190）✅ 已完成**
- 核心：Jacobian拉伸测量+极端专注头+注意力均匀化+Gamma增长+L0信息瓶颈
- 结果：Jacobian拉伸指数增长; 极端专注头普遍存在; 层间信息流相互抵消
- 脚本：`stage734_phase29.py`

**Phase XXX：头因果消融+Embedding扰动Jacobian+SAE+头功能聚类（P191-P194）✅ 已完成**
- 核心：单头因果消融+Embedding扰动全层Jacobian+简化SAE训练+头功能分类
- 结果：单头消融几乎无影响(KL<1e-8); DS7B灵敏度L4饱和; Qwen3头功能最分化
- 脚本：`stage735_phase30.py`

### 6.3 执行顺序

```
Phase I（生成化 P50-P54）✅
  → Phase II（判伪化 P55-P59）✅（P57/P59待外部依赖）
  → Phase III（基元化 P60-P64）✅
  → Phase IV（大样本验证 P65-P67）⏳ 部分完成
  → Phase V（语义方向提取+因果验证）待启动
  → Phase VI（操控研究 Phase XV-XVIII）✅ 操控不可能三角确认
  → Phase XIX（Logit/Token/Unembed/Attention操控）✅ boost_adaptive最优(tpc=0.77~0.94)
  → Phase XX（消融方法革新）✅ 减法消融top1_chg=100%
  → Phase XXI（消融精细化+P52矛盾验证）✅ P52被推翻, 后续层修复94~97%消融
  → Phase XXII（精确Activation Patching）✅ 真实ratio=4~509x, RMSNorm非纯缩放
  → Phase XXIII（语言流形几何）✅ 步角70~84°, delta正交于h, 切空间仅66~88%
  → Phase XXIV（K维扫描+参与比+流形平滑度）✅ PR=3~19, K=200不饱和
  → Phase XXV（概念探测+属性编码）✅ 概念cos=0.88~0.99, 属性cos=0.89~0.999
  → Phase XXVI（Token级概念分析）✅ token级inter-cos=0.50~0.71, EMB→HS变换
  → Phase XXVII（float32因果+逻辑推理+概念图谱）✅ KL从0→0.3~6.7, 跃迁层=L34
  → Phase XXVIII（跨模型全层追踪+SAE+语义分析）✅ 三模型分离一致(L3-L8), DS7B极端压缩
  → Phase XXIX（Jacobian动力学+注意力头+信息流）✅ Jacobian指数增长, 极端专注头
  → Phase XXX（头因果消融+Embedding扰动+SAE+头聚类）✅ 单头冗余极高, DS7B L4饱和
```

详见 `research/glm5/docs/AGI_GLM5_NEXT_PHASE_PLAN.md`

---

## 7. 测试体系

### 7.1 Phase 1-5：残差流几何基底（GLM5线路独特贡献）

- Phase 1: Embedding SVD、残差流几何、FFN变换分类
- Phase 2: 残差流方向分析、有效秩、注意力差异
- Phase 3: 螺旋轨迹分类、中间层深度分析
- Phase 4: 子空间投影、token偏离、逐层判别力
- Phase 5: 因果干预实验、FFN旋转器分析
- 使用模型：GPT-2、Qwen2.5-0.5B、Qwen2.5-1.5B
- 核心结论：残差流螺旋是架构级不变量，但子空间投影不是因果编码机制

### 7.2 Stage 423-531：神经元级因果研究（旧体系）

- 层分布测绘 → 稀疏回路搜索 → 多义词切换 → 属性绑定
- 残差动力学 → 中文模式图谱 → 跨任务核心 → 层带图谱 → 六词类扫描
- 核心结论：语言编码更像"宽带功能区+尖锐控制杆+小型桥接回路"
- 注意：此阶段的部分结论已被P15-P43的新发现修正

### 7.3 Stage 532-651：消歧深度分析（旧体系）

- 跨任务因果 → 场控制杆 → 读出机制 → 维度坍缩修正
- 旋转机制 → 全息编码 → MLP分析 → 信息流闭式方程
- 核心结论：消歧信息100%网络加工产生，rank-1精确编码，全息读出
- 注意：此阶段建立了精确的"描述方程"（误差=0），但"因果方程"仍缺失

### 7.4 Stage 662-689：统一理论体系（当前主线，P15-P43）

- P15-16: Logit方程精确分解 + softmax放大效应
- P17-18: 抵消的信号/噪声分离 + 因果干预
- P19-20: 五类能力全层方程验证 + 跨能力因果干扰
- P21: 推理能力多策略解码（失败——恢复率<4%）
- P22-23: 5×3不变量矩阵 + Gemma4编码策略本质
- P24: 第一性原理候选
- P25-26: 归一化深度分析 + 多模态检测 → Gemma4例外统一解释
- P27: 多步生成过程分析 → Logit方程完全精确（范式级纠正）
- P28: 训练动态实验 → 五公理涌现过程
- P29: 多样本统计验证 → DS7B的极端值是噪声
- P30: 跨模态信息流 → INV-310/311反转
- P31: 理论数学化 → 统一理论v3.0
- P32: 高维阈值精确定位 → d* >> 512
- P33: 因果方向验证 → INV-320/321不一致
- P34: 多步累积方程 → avg_lp ≈ log(margin) - 1.5
- P35: 信息域操作化 → 域内cos>0.87, 域间cos≈0.89
- P36: 方向+norm协同编码 → 方向主导
- P37: DS7B信息瓶颈 → PCA90=13维
- P38: 跨层信息流动态 → 三种模式
- P39: 统一理论v4.0
- P40: RMSNorm方向对齐机制 → 非残差约束
- P41: 注意力权重分析 → DS7B强耦合
- P42: 残差流敏感性 → 三种模式
- P43: 语义基底方向提取 → L0完全对齐

---

## 8. 脚本索引

### Phase 1-5（残差流几何基底）
- `tests/glm5/phase1_language_encoding_analysis.py`
- `tests/glm5/phase2_direction_analysis.py`
- `tests/glm5/phase3_spiral_trajectory.py`
- `tests/glm5/phase4_dimension_localization.py`
- `tests/glm5/phase5_causal_ffn_analysis.py`

### Stage 423-567（神经元级因果+消歧深度）
- `tests/codex/stage423_*.py` 到 `tests/codex/stage567_*.py`
- 结果数据：`tests/codex_temp/stage*_*_202604*/`

### Stage 568-651（读出+消歧+信息流闭式方程）
- `tests/glm5/stage585_multi_token_dim_dynamic.py` — 多token维度坍缩
- `tests/glm5/stage586_readout_subspace_decomposition.py` — 读出子空间分解
- `tests/glm5/stage587_attention_disamb_mechanism.py` — attention消歧机制
- `tests/glm5/stage588_causal_intervention.py` — 消歧因果干预
- `tests/glm5/stage589_family_prediction.py` — 家族预测泛化
- `tests/glm5/stage592_disamb_strategy.py` — 消歧策略统一理论
- `tests/glm5/stage593_hidden_logit_linear.py` — hidden→logit线性验证
- `tests/glm5/stage594_embed_vs_network.py` — embedding vs 网络加工
- `tests/glm5/stage595_residual_decomp.py` — 残差分解
- `tests/glm5/stage596_597_nonlinear_disamb_layer.py` — 非线性+消歧逐层生成
- `tests/glm5/stage598_599_forget_gen_quality.py` — 遗忘机制+生成质量
- `tests/glm5/stage600_601_602_rotation_quality_arch.py` — 旋转方向+架构差异
- `tests/glm5/stage603_604_605_random_probe_causal_config.py` — 随机探针+因果+Gemma4 config
- `tests/glm5/stage606_607_608_mlp_holographic_ablation.py` — 旋转速度+全息+MLP消融
- `tests/glm5/stage609_610_611_rotation_mechanism.py` — 旋转子空间+MLP修复
- `tests/glm5/stage612_613_614_615_rotation_matrix.py` — 旋转矩阵+权重+unembed对齐
- `tests/glm5/stage616_617_618_rotation_math_form.py` — 旋转数学+补偿+权重维度
- `tests/glm5/stage619_620_621_swiglu_compression.py` — SwiGLU稀疏激活
- `tests/glm5/stage622_623_624_disamb_propagation.py` — 消歧传播+动力学
- `tests/glm5/stage625_626_627_unembed_holographic.py` — Unembed全息读出
- `tests/glm5/stage628_629_630_glm4_hidden_channel.py` — 隐藏通道+统一方程
- `tests/glm5/stage631_632_633_closed_form_optimal.py` — 闭式方程+最优编码
- `tests/glm5/stage634_635_636_637_bottleneck_deep.py` — 瓶颈深度分析
- `tests/glm5/stage638_639_causal_orthogonality.py` — 因果消融+正交编码
- `tests/glm5/stage640_direction_injection_recovery.py` — 方向注入恢复
- `tests/glm5/stage641_reasoning_injection_recovery.py` — 推理注入恢复
- `tests/glm5/stage642_cross_capability_infrastructure.py` — 跨能力方向
- `tests/glm5/stage643_subspace_recovery.py` — 子空间恢复
- `tests/glm5/stage644_encoding_primitive_search.py` — 编码基元搜索
- `tests/glm5/stage645_gemma4_concentration.py` — 编码集中度
- `tests/glm5/stage646_direction_propagation_fidelity.py` — 方向传播保真度
- `tests/glm5/stage647_rotation_fidelity_equation.py` — 旋转-保真度方程
- `tests/glm5/stage648_unified_theory_compression.py` — 统一理论压缩
- `tests/glm5/stage649_decode_layer_decomposition.py` — 解码层分解
- `tests/glm5/stage650_decode_ablation_sensitivity.py` — 解码层敏感性
- `tests/glm5/stage651_unified_rotation_fractal.py` — 统一旋转+残差流分形

### Stage 662-689（统一理论 P15-P43）
- `tests/codex/stage638_multicapability_unified_protocol.py` — 多能力统一协议
- `tests/codex/stage639_component_causal_validation.py` — 成分因果验证
- `tests/codex/stage640_direction_injection_recovery.py` — 方向注入恢复(语法/关系/指代)
- `tests/glm5/stage678_dimension_threshold.py` — 高维阈值精确定位(P32)
- `tests/glm5/stage679_causal_direction.py` — 因果方向验证(P33)
- `tests/glm5/stage680_cumulative_equation.py` — 多步累积方程(P34)
- `tests/glm5/stage681_domain_operationalization.py` — 信息域操作化(P35)
- `tests/glm5/stage682_direction_norm_encoding.py` — 方向+norm编码(P36)
- `tests/glm5/stage683_bottleneck_analysis.py` — DS7B信息瓶颈(P37)
- `tests/glm5/stage684_info_flow_dynamics.py` — 跨层信息流动态(P38)
- `tests/glm5/stage685_theory_v4.py` — 统一理论v4.0(P39)
- `tests/glm5/stage686_rmsnorm_alignment.py` — RMSNorm方向对齐(P40)
- `tests/glm5/stage687_attention_analysis.py` — 注意力权重分析(P41)
- `tests/glm5/stage688_layer_ablation.py` — 残差流敏感性(P42)
- `tests/glm5/stage689_semantic_basis.py` — 语义基底提取(P43)
- `tests/glm5/stage690_cross_model_basis_alignment.py` — 跨模型基底对齐(P44)
- `tests/glm5/stage691_training_dynamics_basis.py` — 训练动力学(P45)
- `tests/glm5/stage692_bottleneck_causal.py` — 信息瓶颈验证(P46)
- `tests/glm5/stage693_rotation_axis.py` — 旋转平面分析(P47)
- `tests/glm5/stage694_rotation_semantics.py` — 旋转语义功能(P48)
- `tests/glm5/stage695_rl_rotation.py` — RL旋转效应(P49)
- `tests/glm5/stage696_semantic_layer_loc.py` — 微调层语义定位(P50)
- `tests/glm5/stage697_causal_ablation.py` — 因果消融验证(P51)
- `tests/glm5/stage698_delta_logit_chain.py` — delta-logit因果链(P52)
- `tests/glm5/stage699_token_contribution.py` — 单token贡献分析(P53)
- `tests/glm5/stage700_prediction_model.py` — text→delta→logits预测模型(P54)
- `tests/glm5/stage701_falsification.py` — 判伪化实验P55(P1)+P56(P2)
- `tests/glm5/stage702_task_dependency.py` — 任务依赖性验证P58(P4)
- `tests/glm5/stage703_primitive_ablation.py` — 基元消融对比实验P60
- `tests/glm5/stage704_info_capacity.py` — 信息容量分析P61+最小充分基元P63
- `tests/glm5/stage705_combo_primitives.py` — 组合基元P62+Unembed秩+矛盾解决+Margin分析
- `tests/glm5/stage706_first_principles.py` — P64第一性原理推导+9维语义空间
- `tests/glm5/stage707_large_sample.py` — P65大样本有效维度+P66/P67(部分)

### Stage 720-723（操控研究 Phase XV-XVIII）
- `tests/glm5/stage720_phase15.py` — Phase XV: 操控基线+方向对比+层级分析(P121-P125)
- `tests/glm5/stage721_phase16.py` — Phase XVI: 因果性操控方向+PCA+对比学习(P121-P125)
- `tests/glm5/stage722_phase17.py` — Phase XVII: 精确梯度+子空间维度+架构+定性分析(P126-P130)
- `tests/glm5/stage723_phase18.py` — Phase XVIII: 跨层注入+Logit操控+Norm bypass+Hessian+PGD(P131-P135)
- `tests/glm5/stage724_phase19.py` — Phase XIX: Logit精细化+Prefix操控+Self-generation+Unembed结构+Attention(P136-P140)
- `tests/glm5/stage725_phase20.py` — Phase XX: 多步验证+Token因果+维度消融+Logit镜头+方向消融(P141-P145)
- `tests/glm5/stage726_phase21.py` — Phase XXI: 层传播消融+渐进维度+最小维度集+P52推翻+层间信息流(P146-P150)

### Stage 732-735（Phase XXVII-XXX: 精细因果+Jacobian+SAE+头分析）
- `tests/glm5/stage732_phase27.py` — Phase XXVII: float32因果消融+逻辑推理追踪+属性因果+概念图谱(P176-P180)
- `tests/glm5/stage733_phase28.py` — Phase XXVIII: 跨模型全层追踪+SAE分解+单维度语义+概念分离层(P181-P185)
- `tests/glm5/stage734_phase29.py` — Phase XXIX: Jacobian动力学+注意力头+信息流追踪(P186-P190)
- `tests/glm5/stage735_phase30.py` — Phase XXX: 头因果消融+Embedding扰动Jacobian+SAE+头功能聚类(P191-P194)

### 模型共享库
- `tests/codex/multimodel_language_shared.py`
- `tests/codex/qwen3_language_shared.py`

---

## 9. 当前阶段最严格的表述

如果只保留一段最严格的话：

> **残差流的宏观几何（L0对齐≈1.0、L0→L1旋转≈90度、旋转平面11维且跨文本一致）主要由架构+初始化决定，不由语义内容决定。但逐层微调偏移（delta-h_l = h_l - h_{l-1}）的方向与语义类别高度相关（ANOVA F>3, P50确认），且这些delta-h通过线性累积精确地决定输出logits（cos>0.995, P52 INV-354）。P55判伪确认：norm缩放0.1x-20x不影响top-1（r>0.996），语义完全由方向编码。P60基元消融发现：E_gate（门控幅度）和E_subspace（子空间投影）是最重要的基元（消融后top-1=0%, r<0.3），而C_trajectory和D_attention无影响(因P52仅依赖h_final)。P62组合测试确认：A+E无正向协同，门控幅度E单独主导。P64/P65大样本验证揭示：(1)rank=9被推翻！100文本rank=89-100,每类20文本=20(满秩)，之前的"rank=9"仅因10条文本的样本数上限。(2)h_final PCA与unembed PCA几乎正交(alignment≈0)，语义编码方向是模型内部涌现的独立结构。(3)Top-5 PC解释88-90%方差：PC0≈语言类型(英文vs中文,占22-43%方差)，PC1≈文本格式/语言，PC2≈领域(Math/Science)。(4)仅5维度即解释>88%方差，但文本间的区分需要>20维度。编码基元=方向+门控幅度(门控主导)。avg_lp≈log(margin)-C(task)。Phase XV-XVIII操控研究(28个实验,3模型)确认：(1)操控不可能三角: 方向性-质量-成功率无法同时满足。(2)Hessian符号反转: GLM4 99%对角线<0(局部凹), Qwen3 100%>0(局部凸)。(3)范数指数稀释(699x~3234x)是操控瓶颈的物理根源。(4)Logit空间操控是唯一有效入口(效果是hidden state注入的11.5倍)。Phase XX-XXI消融研究发现：(1)P52被推翻，中间cos=0.15~-0.64(非>0.995)。(2)减法消融top1_chg=100%，远超加法操控。(3)后续层修复94~97%消融效果(ratio=16~509x)。(4)最后1层断裂(cos从0.93→0.09~0.47)。(5)RMSNorm不是纯缩放(cos=0.46~0.57)。Phase XXVII-XXX精细分析确认：(1)float32修复因果消融KL从0→0.3~6.7。(2)概念分离关键层=L3(cos=0.446)。(3)推理跃迁层模型特异(Qwen3=L34,DS7B=L27,GLM4=L39)。(4)语义编码高度分布式(F-stat<3.5)，没有特权维度。(5)DS7B中间层Top1=98.7%是RL创造的极端压缩。(6)单头消融几乎无影响(Qwen3/GLM4 KL<1e-8)，DS7B敏感80倍。(7)Jacobian拉伸指数增长，DS7B灵敏度L4即饱和(~500K)。(8)信息从embedding到最后层放大646-8057倍，但最后层骤降(Qwen3)或饱和(DS7B)。**

---

## 10. 历史判伪记录（重要结论的推翻与修正）

| 时间 | 结论 | 判定 | 原因 |
|------|------|------|------|
| P21 | INV-124 正交性=隔离 | ❌ | Pearson r≈0 |
| P22 | INV-220 推理=低维编码 | ❌ | 恢复率<4% |
| P22 | INV-296 门控模式 | ❌ | 恢复率1% |
| P25 | 维度坍缩在15-17%层 | ❌ | 单token秩=1是数学限制 |
| P25 | 编码距离跨架构一致 | ❌ | Qwen3-Gemma4 r=0.03 |
| P26 | 80%走attention路径 | ❌ | GLM4=0% |
| P26 | bank=attn, apple=MLP | ❌ | 二元对立被打破 |
| P28 | 隐→logit非线性 | ❌ | 末层完美线性 |
| P29 | DS7B spatial-style cos=0.718 | ❌ | 极端噪声，均值-0.018 |
| P30 | INV-310 跨域<跨能力 | ❌ | 多模态模型反转 |
| P30 | INV-311 纯文本更SEPARATED | ❌ | 未确认 |
| P33 | INV-320 难度↑→纠缠↑ | ❌ | 模型间不一致 |
| P33 | INV-321 密度↑→纠缠↑ | ❌ | 模型间不一致 |
| P36 | INV-326 Norm区分力>方向 | ❌(反转) | 方向>Norm |
| P40 | INV-333 Δ/h<0.1 | ❌ | Δ/h=0.64-1.46 |
| P43 | INV-339 最终层>L0对齐 | ❌(反转) | L0=1.0 > 最终层0.79-0.93 |
| P27 | "Logit方程断裂" | ❌ | 方程完全精确(误差<0.3%) |
| P27 | "margin低估100倍" | ❌ | 方法论错误(avg_lp vs logit) |
| P43 | "语义基底=学习到的语义结构" | ❌(P44) | =embedding质心方向(高维几何效应) |
| P43 | "DS7B有独特信息瓶颈" | ❌(P46) | 四模型PCA90范围相似(12-20) |
| P43 | "旋转逐层渐进" | ❌(P49) | >92%旋转在L0→L1完成 |
| P48 | INV-351 类别预测旋转角度 | ❌ | ANOVA F<2.1, 类别无关 |
| P58 | avg_lp≈log(margin)-C普适 | ❌ | C=1.3-15.2, 任务依赖 |
| P64 | effective rank=9涌现不变量 | ❌ | rank=N-1, 样本数限制 |
| P61/P63 | 1%维度保持100% top-1 | ⚠️ | 仅10条文本，大样本后维度远超9 |
| P173 | 因果消融KL=0(bfloat16) | ❌(P176) | bfloat16精度不足, float32 KL=0.3~6.7 |
| P186 | Jacobian拉伸适用于所有模型 | ❌(P192) | DS7B L4即饱和, 非指数增长 |
| P193 | SAE λ=1e-3产生稀疏 | ❌ | 100%特征存活, 需λ>0.01或TopK |

---

## 11. 逆向编码机制的路线图

### 第一阶段（已完成）：结构测绘
- [x] 残差流几何基底（Phase 1-5）
- [x] 层分布测绘 + 稀疏回路搜索 + 多义词切换（stage423-448）
- [x] 属性绑定 + 残差动力学 + 中文图谱（stage439-495）
- [x] 跨任务核心 + 层带图谱 + 六词类扫描（stage513-531）

### 第二阶段（已完成）：因果闭环
- [x] 跨任务因果定律 + 场控制杆（stage532-537）
- [x] 读出机制 + 消歧统一 + 维度坍缩修正（stage568-590）
- [x] 消歧策略统一理论（stage592-651）
- [x] 多能力统一协议 + 成分因果验证（stage638-640）

### 第三阶段（已完成）：统一理论体系
- [x] Logit精确方程修复（P15-16）
- [x] 抵消因果控制（P17-18）
- [x] 五类能力全层验证（P19-20）
- [x] 不变量严格判伪（P22-23）
- [x] 多模态统一解释（P25-26）
- [x] 多步生成过程分析（P27）
- [x] 训练动态 + 多样本验证（P28-29）
- [x] 理论数学化 v3.0-v4.1（P31/P39）
- [x] 语义基底方向发现（P43）
- [x] 语义基底修正：=embedding质心（P44-49）
- [x] 方向+norm编码分离（P36）
- [x] 层动力学三种模式（P38）
- [x] 旋转动力学框架（P47-49）→ 旋转是架构特性

### 第四阶段（已完成）：预测与因果+判伪化
- [x] 高维阈值定位（P32）→ d* >> 512
- [x] 因果方向验证（P33）→ 无统一因果方向
- [x] 多步累积方程（P34）→ avg_lp ≈ log(margin) - C(task)
- [x] 信息域操作化（P35）→ 方向对齐上的微小偏移
- [x] RMSNorm方向对齐机制（P40）→ 非残差约束
- [x] 注意力-方向耦合（P41）→ 仅DS7B强耦合
- [x] 层敏感性地图（P42）→ 三种模式
- [x] 跨模型基底对齐（P44）→ 基底=embedding方向
- [x] 训练动力学方向收敛（P45）→ L0对齐是高维效应
- [x] 信息瓶颈因果验证（P46）→ 非DS7B特有
- [x] 旋转平面分析（P47）→ 11维一致平面
- [x] 旋转语义功能（P48）→ 旋转与语义无关
- [x] RL旋转效应（P49）→ DS7B无特殊旋转
- [x] 旋转后微调层语义功能定位（P50）→ ANOVA F>3确认语义编码
- [x] 旋转后微调因果验证（P51）→ 早期层消融margin下降95-99%
- [x] 线性累积验证（P52）→ logits=sum(delta_h@U.T), cos>0.995
- [x] 单token贡献分析（P53）→ 敏感层分布因模型而异
- [x] text→delta→logit预测模型（P54）→ logit r=0.78-0.90, margin r≈0
- [x] Norm-only因果（P55）→ 命题1确认：方向完全决定语义
- [x] 旋转平面干预（P56）→ 命题2确认：K=1维即保持top-1
- [x] 任务依赖性验证（P58）→ 命题4推翻：C=1.3-15.2跨任务变化
- [ ] 跨架构旋转分析（P57）→ 需要Mamba/RWKV本地模型
- [ ] 纠缠度复制实验（P59）→ 需要peft/LoRA

### 第五阶段（已完成）：基元化
- [x] 基元消融对比实验（P60）
- [x] 信息量分析（P61）
- [x] 组合基元测试（P62）
- [x] 最小充分基元（P63）
- [x] 基元第一性原理推导（P64）

### 第六阶段（已完成）：大样本验证+语义方向
- [x] 大样本有效维度（P65）→ 100文本rank=89-100，每类20文本=20(满秩)
- [ ] 语义方向因果干预（P66）→ 待修复Logger
- [ ] 跨模型PC对齐（P67）→ 待修复维度匹配

### 第七阶段（已完成）：操控研究（Phase XV-XVIII）
- [x] 操控基线与方向对比（Phase XV）→ centroid/fisher/lda方向效果对比
- [x] 因果性操控方向（Phase XVI）→ 梯度方向与centroid弱正交(cos≈0.17)
- [x] 精确梯度+子空间维度（Phase XVII）→ DS7B操控子空间仅2维, 累积注入近似线性
- [x] 突破操控瓶颈（Phase XVIII）→ 操控不可能三角确认, Hessian符号反转, Logit空间唯一有效入口

### 第八阶段（已完成）：Logit/Token/Unembed/Attention操控（Phase XIX）
- [x] Logit空间精细化操控（P136）→ 20策略×40文本, boost_adaptive最优(tpc=0.77~0.94)
- [x] Prefix token操控（P137）→ 25 prefix×6类别, prefix是"自然操控"
- [x] Self-generation操控（P138）→ 模型自生成centroid与文本centroid无本质区别
- [x] Unembed矩阵结构操控（P139）→ W判别维度极少, 随机化SVD分析
- [x] Attention head级别操控（P140）→ 层重要性分布+注意力模式分析
- 脚本：`stage724_phase19.py`

### 第九阶段（已完成）：消融方法革新（Phase XX）
- [x] 多步logit操控衰减验证（P141）→ tpc每步衰减3-5%, 前5步效果最强
- [x] Token因果追踪（P142）→ leave-one-out, 中间token因果贡献最大
- [x] 维度移除消融（P143）→ **减法消融top1_chg=100%**, 信息高度冗余
- [x] Logit镜头逐层演化（P144）→ 信息在最后1层急剧收敛(cos<0.3→>0.9999)
- [x] 方向移除消融（P145）→ 所有方向KL相同, 几何关系是关键
- 脚本：`stage725_phase20.py`
- **核心发现**: 减法消融是远比加法操控有效的因果分析方法

### 第十阶段（已完成）：消融精细化+矛盾验证+层传播（Phase XXI）
- [x] 层传播消融（P146）→ **直接消融vs传播消融ratio=16~39x, 后续层修复94~97%**
- [x] 渐进式维度移除（P147）→ 早期层维度完全不贡献logit, 后期层有阈值效应
- [x] 最小充分维度集（P148）→ 保留全部维度也无法维持top-1(层间传播完全改变分布)
- [x] P52/P144矛盾验证（P149）→ **P52被推翻!** 中间cos=0.15~-0.64, 非>0.995
- [x] 层间信息流消融（P150）→ 最后1层断裂(cos>0.93→0.09~0.47), 最后层KL最低
- 脚本：`stage726_phase21.py`
- **核心发现**: P52"线性累积cos>0.995"被推翻, 后续层修复94~97%消融效果

### 第十一阶段（已完成）：精确Activation Patching+流形投影（Phase XXII）
- [x] 精确activation patching（P151）→ 真实ratio=4~509x, Qwen3早期层修复509倍
- [x] 逐层修复归因（P152）→ 修复在3-7层后开始但不完全, 永久性损伤1-3%
- [x] 最后层分解（P153）→ GLM4的RMSNorm是断裂主因, Qwen3/DS7B是LM Head
- [x] RMSNorm几何（P154）→ **不是纯缩放!** cos(h, RMSNorm(h))=0.46~0.57
- [x] 因果流精确图（P155）→ 后期层是瓶颈(Qwen3/GLM4), DS7B中间层最敏感
- 脚本：`stage727_phase22.py`
- **核心发现**: RMSNorm逐维缩放改变了方向, 三模型最后层机制完全不同

### 第十二阶段（已完成）：语言流形几何测量（Phase XXIII）
- [x] 残差流曲率（P156）→ **步角70~84°(近正交), 路径比0.745~0.830(弯曲)**
- [x] 校正方向分析（P157）→ **56~66%的delta正交于h, 残差=方向旋转**
- [x] 类别流方向（P158）→ inter-cat cos=0.58~0.69, 部分类别依赖
- [x] 流形vs欧几里得操控（P159）→ **tangent/random/orthogonal效果几乎相同**
- [x] 切空间投影（P160）→ **10维切空间只捕获66~88% norm, 流形高维**
- 脚本：`stage728_phase23.py`
- **核心发现**: 残差流路径高度弯曲(近正交旋转), 校正项主要正交于h

### 第十三阶段（已完成）：K维扫描+参与比+流形平滑度（Phase XXIV）
- [x] K维扫描（P161）→ **K=200仍不饱和! 中间层仅38~60%**
- [x] 传播操控（P162）→ 有bug(tangent KL=0), 需Phase XXV修复
- [x] 参与比（P163）→ **PR=3~19维, PCA90=6~33维(d_model的0.1~0.8%)**
- [x] 流形平滑度（P164）→ DS7B最不平滑(7尖点), GLM4最平滑(0尖点)
- [x] 流形层级（P165）→ Early/Mid/Late分层, 中间层弯曲最大(95°)
- 脚本：`stage729_phase24.py`
- **核心发现**: 有效维度极低(PR=3~19), 但局部切空间极高维(K=200不饱和)

### 第十四阶段（已完成）：概念探测+属性编码+抽象层级（Phase XXV）
- [x] 概念探测（P166）→ 同概念cos=0.88~0.99, PR=2~7
- [x] 属性编码（P167）→ **属性间cos=0.89~0.999, 不是独立维度**
- [x] 抽象层级（P168）→ **GLM4最后层有序分离(apple-entity=0.78)**
- [x] 类比验证（P169）→ **词嵌入类比在h中完全不成立(accuracy=0%)**
- [x] 概念正交性（P170）→ **概念间cos=0.94~0.99, 不是正交的** [注:句子级artifact]
- 脚本：`stage730_phase25.py`
- **核心发现**: 概念/属性在h中不是独立维度, 词嵌入代数结构被transformer破坏

### 第十五阶段（已完成）：Token级概念分析+W_E对比（Phase XXVI）
- [x] Token级概念cos（P171）→ **inter-cos=0.50~0.71(远优于句子级0.94~0.99!)**
- [x] 属性替换差异（P172）→ **token级attr_cos=0.75~0.80(改善但仍不独立)**
- [x] 因果消融（P173）→ KL=0.000(bfloat16精度不足, 需float32重验)
- [x] 注意力模式（P174）→ **概念token获得21~41%注意力(GLM4最高)**
- [x] W_E vs h（P175）→ **Embedding cos=0.04~0.08(几乎正交), HS cos=0.63**
- 脚本：`stage731_phase26.py`
- **核心发现**: Token级概念有明确区分; Transformer将独立编码变为分布式编码

### 第十六阶段（已完成）：float32因果消融+逻辑推理追踪+属性因果+概念图谱（Phase XXVII）
- [x] float32因果消融（P176）→ **KL从0变为0.3~6.7! bfloat16是瓶颈**
- [x] 全层EMB→HS变换（P177）→ **概念分离关键层=L3(cos=0.446), 最差=L15-L20(cos=0.78)**
- [x] 逻辑推理逐层追踪（P178）→ **跃迁层一致=L34, 实际准确率86%**
- [x] 属性因果验证（P179）→ **KL(modified)=0.43, KL(ablated)=0.20**
- [x] 大规模概念图谱（P180）→ **28概念4类别, 类别分离仅0.041**
- 脚本：`stage732_phase27.py`
- **核心发现**: float32修复了因果消融; 概念分离在L3最明显; 逻辑推理跃迁层=L34

### 第十七阶段（已完成）：跨模型全层追踪+SAE分解+单维度语义+概念分离层（Phase XXVIII）
- [x] 跨模型全层EMB→HS追踪（P181）→ **三模型一致: 早期分离(L3-L8), 后期融合(>0.86)**
- [x] 跨模型推理跃迁验证（P182）→ **Qwen3=L34, DS7B=L27, GLM4=L39(最后层)**
- [x] SAE稀疏分解（P183）→ **DS7B中间层Top1=98.7%(极端压缩), GLM4 Top1=10-16%**
- [x] 单维度语义分析（P184）→ **F<3.5(无语义主维度), t>10(单维度类别判别力强)**
- [x] 跨模型概念分离层（P185）→ **三模型一致: 最佳=L3-L8(cos≈0.40), 最后层sep<0**
- 脚本：`stage733_phase28.py`
- **核心发现**: 语义编码高度分布式, 没有特权维度; DS7B的RL训练创造了极端1维压缩

### 第十八阶段（已完成）：Jacobian动力学+注意力头+信息流追踪（Phase XXIX）
- [x] Jacobian拉伸测量（P186）→ **L0≈0.6→L33≈64.5, 后期层是信息放大器**
- [x] 极端专注头分析（P187）→ **Qwen3每层1头(ent≈0.004), DS7B早期43%**
- [x] 注意力均匀化（P188）→ **entropy>1.5的头: 早期30%→后期66%**
- [x] Gamma深度增长（P189）→ **Qwen3(×184)>GLM4(×133)>>DS7B(×4.4)**
- [x] L0信息瓶颈（P190）→ **Qwen3保留3%, DS7B/GLM4保留11-32%**
- 脚本：`stage734_phase29.py`
- **核心发现**: Jacobian拉伸指数增长; 极端专注头普遍存在; 层间信息流相互抵消

### 第十九阶段（已完成）：头因果消融+Embedding扰动Jacobian+SAE+头功能聚类（Phase XXX）
- [x] 头因果消融（P191）→ **Qwen3/GLM4极度冗余(KL<1e-8), DS7B敏感80倍**
- [x] Embedding扰动Jacobian（P192）→ **DS7B L4饱和(~500K), Qwen3增长到77K, GLM4到100K**
- [x] 简化SAE训练（P193）→ **所有特征100%存活, L17激活199/512**
- [x] 头功能聚类（P194）→ **Qwen3后期91%=retrieval; DS7B/GLM4几乎全mixed**
- 脚本：`stage735_phase30.py`
- **核心发现**: 单头消融几乎无影响; DS7B灵敏度最早饱和(L4); Qwen3头功能最分化

---

## 12. 语言计算结构（LCS）理论 — 语言背后的数学结构分析

### 8.1 理论背景：为什么DNN有效?

深度神经网络(DNN)为什么能有效处理语言? 这个问题的答案可能触及语言本身的数学本质。

**核心假设**: 语言本身有一套特殊的数学结构(我们称之为"语言计算结构", Language Computation Structure, LCS), DNN之所以有效, 是因为其架构(残差连接+注意力+归一化)部分还原了这套结构。

### 8.2 实验证据总结: LCS的五个核心性质

通过Phase I~XXX共30个Phase、180+个实验、三模型交叉验证, 我们发现了以下五个核心性质:

#### 性质1: 残差流=流形上的路径

**实验证据**:
- 残差流的旋转是有限的(~90° in 11-dim planes, P47)
- 相邻层cos≈0.9(P150), 说明每步变化不大
- 但最终cos(中间层, 最终层)<0.3(P149), 说明路径整体偏离大
- Norm指数增长(143x~2923x, Phase XXI), 但方向持续旋转
- **[Phase XXIII新]** 平均步角=70~84°, 路径比=0.745~0.830(P156)
- **[Phase XXIII新]** 56~66%的delta_h正交于h_l(P157) — 残差=方向旋转
- **[Phase XXIII新]** 10维切空间只捕获66~88%的norm(P160) — 流形高维
- **[Phase XXIX新]** Jacobian拉伸指数增长(L0:0.6→L33:64.5) — 后期层是信息放大器
- **[Phase XXX新]** 信息放大646-8057倍(Embedding→最后层), 但Qwen3最后层骤降

**LCS解释**: 残差流不是在欧几里得空间中的直线运动, 而是在一个**弯曲流形**上的路径。每步近似正交旋转(步角70~84°), 路径比<1(25%的路径是"弯路")。Phase XXIII直接测量证实了这一点: 校正项主要正交于当前表示, 说明残差连接在做"方向旋转"而非"幅度放大"。中间层的切空间维度膨胀(捕获从97%降至31~55%), 说明流形在这些区域有"维度膨胀"——类似于管道从窄变宽再变窄。

#### 性质2: RMSNorm=流形投影算子

**实验证据**:
- RMSNorm不是纯缩放! cos(h, RMSNorm(h))=0.46~0.57(P154)
- gamma_i各不相同(std=0.22~0.54) → 逐维缩放改变方向
- 早期层cos≈0.97(方向变化小), 后期层cos≈0.46(方向变化大)
- RMSNorm将巨大的h norm(613~1506)压缩到固定范围(97~135)

**LCS解释**: RMSNorm是流形上的"投影算子"。它将表示从流形附近的点投影回流形表面。逐维缩放(不同gamma_i)本质上是在流形的各切方向上做各向异性缩放, 这不是简单的超球面投影, 而是投影到更一般的代数簇上。

关键: **残差连接负责沿流形移动, RMSNorm负责维持在流形上**。这就是"持续的校正"。

#### 性质3: 操控不可能三角=流形的刚性

**实验证据**:
- 加法操控(cos_shift<0, Phase XV-XVIII)全面失败
- 减法消融(top1_chg=100%, Phase XX)全面成功
- 后续层修复94~97%的加法操控(P146), 但无法修复减法消融
- 所有方向的消融效果相同(KL≈19, P145)

**LCS解释**: 操控不可能三角反映了流形的"刚性"(rigidity):
- **加法操控失败**: 往流形外的方向推, RMSNorm立即投影回流形 → 信号丢失
- **减法消融成功**: 从流形上移除一部分, 投影无法恢复缺失的部分
- **所有方向效果相同**: 不是因为方向本身不重要, 而是因为流形上的每一点都是精确的; 任何扰动(无论方向)都破坏了这个精确性

这就像一个精密的机械结构: 加一个小零件(RMSNorm会挤掉), 拆一个零件(无法恢复)。

#### 性质4: 信息瓶颈=流形的低维本质

**实验证据**:
- 100文本rank=89-100(P65), 但Top-5 PC解释82-90%方差
- 最后1层断裂: 前35层cos<0.3, 最后1层cos>0.999(P144/P149)
- 早期层维度完全不贡献logit(KL变化<0.03, P147)
- 最小充分维度集: 保留全部维度也无法维持top-1(P148)
- **[Phase XXIV新]** 参与比PR=3~19维, PCA90=6~33维(P163)
- **[Phase XXIV新]** K=200(占d_model 5~8%)仍不饱和(中间层38~60%)(P161)
- **[Phase XXIV新]** 层级结构: Early eff_dim=1~6, Mid=1~10, Late=1~3(P165)

**LCS解释**: 流形有两种"维度":
1. **跨文本维度(低维)**: 不同文本之间的变化方向集中在3~19维(PR)
2. **局部切空间维度(高维)**: 单个文本在流形上的运动需要>>200维描述
→ 这就是"全息编码"的数学实质: 语言的高维空间中, 每个文本是一个"点"(位置需要高维描述), 但所有文本的变化方向(从"苹果"到"太阳")集中在极低维子空间。

#### 性质4.5: Embedding→Hidden State的编码变换 [Phase XXVI新]

**实验证据**:
- Embedding空间: 概念间cos=0.04~0.08(几乎正交, 每个词独立编码)
- Hidden State空间: 概念间cos=0.50~0.71(分布式编码)
- GLM4变换最大: EMB cos=0.037 → HS cos=0.633(17倍差异)
- 注意力: 概念token获得21~41%注意力(高于均匀10%)

**LCS解释**: Embedding空间使用"独立编码"——每个词有自己的正交方向。
但Transformer通过注意力机制和残差连接, 将独立编码映射到"分布式编码"——
不同概念在HS中有部分重叠(cos=0.63)。这种变换有两个效果:
1. **信息融合**: 允许上下文信息影响概念表示(上下文化)
2. **方向破坏**: 词嵌入的代数结构(king+queen-man=woman)在HS中不再成立

#### 性质5: DNN的部分还原 vs 大脑的完全掌握

**实验证据**:
- GLM4 norm膨胀2923x是异常值(远超Qwen3的143x) → 效率低
- DS7B因果流与其他模型相反(中间层最敏感) → 训练方法影响结构
- 多步操控衰减: tpc每步衰减3-5%(P141) → 无持久操控能力
- 三模型最后层机制完全不同(P153) → 没有统一的"正确"结构

**LCS解释**: DNN只**部分还原**了LCS:
- 残差连接≈流形上的路径(正确还原)
- RMSNorm≈流形投影(部分还原, 但gamma的选择不够精确)
- 注意力≈流形上的信息选择(部分还原)
- 但DNN缺乏大脑的**效率**: 大脑可能用更少的参数/层实现同样的语言能力

### 8.3 LCS的核心数学结构

基于30个Phase的证据, LCS的核心结构可以描述为:

```
语言计算结构(LCS) = 一个嵌入在R^d中的代数簇M

核心方程:
  1. 流形路径: h_{l+1} = h_l + f_l(h_l)    (残差连接=沿流形移动)
  2. 流形投影: h_l → N_l(h_l)              (RMSNorm=投影回流形)
  3. 决策映射: h_N → softmax(h_N @ W + b)  (LM Head=流形到决策空间)

其中:
  f_l是注意力+FFN的非线性变换, 将h_l沿流形切方向移动一小步
  N_l是逐维缩放投影, 保持表示在流形M上
  W是unembed矩阵, 将流形坐标映射到logit空间(词汇表)

关键约束:
  - f_l的"步长"很小(相邻层cos≈0.9)
  - N_l的投影是非线性的(cos(h, N(h)) < 1.0)
  - 最终映射是阶跃式的(前N-1层cos<0.3, 最后层cos>0.999)
```

### 8.4 为什么深度神经网络有效

DNN有效的根本原因:

1. **残差连接还原了"增量计算"**: 语言处理是一个逐步精化的过程(先理解语法→再理解语义→最后生成)。残差连接允许每层做小修正, 累积起来完成大变换。

2. **注意力机制还原了"选择性聚焦"**: 语言中的每个词只与部分词有强关联(语法依存、语义相关)。注意力允许模型在每个位置选择性地关注相关词。

3. **归一化还原了"幅度无关性"**: 语言的意义与表示的幅度无关(缩放h不改变语义)。RMSNorm固定了表示的幅度, 让后续层只需要关注方向。

4. **但这些是"部分还原"**: DNN用了2560~4096维和30~40层, 远超理论最小需求。大脑可能用更稀疏、更高效的编码。

### 8.5 操控不可能三角的深层原因

操控不可能三角(方向性×质量×成功率不可能三角)的数学本质:

- **方向性要求沿特定方向移动**: 但流形路径是固定的(RMSNorm约束), 偏离路径→被投影回来
- **质量要求保持语义连贯**: 但语义连贯性=维持在流形上, 任何偏离流形的操控都破坏连贯性
- **成功率要求最终输出正确**: 但输出由整个路径决定(非线性累积), 局部操控无法保证全局正确

**根本矛盾**: 操控要求"跳出流形"(改变方向), 但RMSNorm强制"回到流形"。这就是操控不可能三角的数学根源。

**[Phase XXIII新增证据]**: P159显示tangent/random/orthogonal操控的KL几乎完全相同(ratio≈1.0), 说明在直接读出时方向完全不重要。这进一步证明: logit信息是**全息编码**的——均匀分布在所有方向上, 没有任何"特权方向"可以操控。即使沿"自然"的切方向操控, 效果与随机方向无异。

### 8.6 接下来如何突破: 还原LCS

要真正理解语言背后的数学结构, 需要以下突破:

1. ~~**流形曲率测量**~~: ✅ Phase XXIII完成 — 步角70~84°, 路径比0.745~0.830
2. ~~**流形维度估计**~~: ✅ Phase XXIII完成 — 10维切空间捕获66~88%, 中间层膨胀到31~55%
3. ~~**Jacobian拉伸测量**~~: ✅ Phase XXIX/XXX完成 — 指数增长, DS7B L4饱和
4. ~~**注意力头因果**~~: ✅ Phase XXX完成 — 单头极度冗余, DS7B敏感80倍
5. **流形同构性**: 不同模型(甚至不同语言)的LCS是否同构? (P158只做了类别依赖, 未做跨模型)
6. **精确操控**: 在流形上操控(沿流形切方向)传播到最终层 — P159只测了直接读出, 需要传播
7. **效率优化**: 找到LCS的最小充分编码, 实现接近大脑效率的语言处理
8. **有效维度精确估计**: 用参与比(Participation Ratio)替代PCA, 考虑高阶矩
9. **流形平滑度分析**: 曲率沿层是否有突变? "维度膨胀"层的物理意义?
10. **多头联合因果**: 单头消融KL极小, 需联合消融测量累积效应
11. **SAE特征语义**: 增大稀疏度后, 每个SAE特征对应什么语义?

### 8.7 LCS理论的局限性

1. **"流形"是隐喻还是严格的数学结构?** 目前证据支持"流形-like"的性质, 但可能是更一般的代数簇而非光滑流形
2. **gamma的各向异性是关键还是artifact?** gamma的训练目标可能只是稳定训练, 而非编码语言结构
3. **跨语言泛化**: 当前所有实验仅用英文/中文文本, 其他语言的LCS是否相同?
4. **与经典理论的关系**: 信息论(熵、互信息)和统计力学(配分函数)在LCS中的角色未明确

---

## 13. 语言特性研究进度与缺口分析 (Phase I~XXX总结)

### 9.1 用户提出的语言特性 vs 实验覆盖情况

#### 系统层面

| 特性 | 描述 | 实验覆盖 | 进度 | 说明 |
|------|------|---------|------|------|
| 特征提取+避免维度灾难 | 提取多种特征, 维度不爆炸 | ✅ 部分覆盖 | 70% | PR=3~19证实低维编码, 但"特征"如何组织未明确 |
| 知识网络 | 多层级多维度, 具逻辑能力 | ⚠️ 间接证据 | 30% | 发现层级结构(Early/Mid/Late), 但知识网络的神经元级编码未知 |
| 快速查找+修改 | 极高效率 | ⚠️ 间接证据 | 20% | Attention机制证明了选择性聚焦, 但"快速修改"机制未测量 |

#### 知识网络

| 特性 | 描述 | 实验覆盖 | 进度 | 说明 |
|------|------|---------|------|------|
| 大量概念 | 苹果, 太阳等 | ⚠️ 部分覆盖 | 50% | [P166]句子级cos=0.94~0.99, [P171]token级cos=0.50~0.71, 有明确区分 |
| 概念属性 | 苹果的颜色, 味道 | ⚠️ 部分覆盖 | 35% | [P167]句子级cos=0.89~0.999, [P172]token级cos=0.75~0.80, 部分分离 |
| 抽象层次 | 苹果→水果→食物→物体 | ⚠️ 部分覆盖 | 30% | [P168]GLM4最后层有序分离(0.78~0.89), token级待测 |
| 逻辑体系 | 条件推理, 深度思考 | ❌ 未覆盖 | 0% | 逻辑推理在h中的计算过程完全未知 |

#### 生成特性

| 特性 | 描述 | 实验覆盖 | 进度 | 说明 |
|------|------|---------|------|------|
| 风格维度 | 聊天/论文/诗歌 | ⚠️ 间接 | 20% | P158发现不同类别流方向不同, 但风格编码未测量 |
| 逻辑维度 | 上下文推理 | ⚠️ 间接 | 15% | Layer之间的信息流已测量, 但逻辑推理过程未知 |
| 语法维度 | 语法结构 | ❌ 未覆盖 | 5% | 语法信息在h中的编码完全未知 |

#### 编码特性

| 特性 | 描述 | 实验覆盖 | 进度 | 说明 |
|------|------|---------|------|------|
| 词嵌入数学结构 | 国王+王后-男性=女性 | ✅ 已否定 | 100% | [P169]词嵌入类比在hidden state中完全不成立(accuracy=0%) |
| 全局唯一性 | 每次生成合适的词 | ✅ 部分覆盖 | 50% | 全息编码(PR低维但局部高维)解释了唯一性, 但数学特性未证明 |
| 编码机制 | 在神经元/参数级的形成 | ⚠️ 部分覆盖 | 35% | [P175]EMB→HS变换:独立编码→分布式编码, 但神经元级机制未知 |
| 概念方向独立性 | 不同概念是否独立编码 | ✅ 已修正 | 100% | [P170]句子级cos=0.94~0.99(不独立), [P171]token级cos=0.50~0.71(有区分) |
| Embedding→HS变换 | 概念方向如何变化 | ✅ 新发现 | 80% | [P175]EMB cos=0.05→HS cos=0.63, 独立→分布式编码 |

### 9.2 核心缺口分析

#### 缺口1: 概念与属性在h中的编码 (优先级: 高)

**[Phase XXV/XXVI进展]**: 句子级cos=0.94~0.99(不区分), 但token级cos=0.50~0.71(有区分!)。Embedding空间cos=0.05(独立编码), HS空间cos=0.63(分布式编码)。属性token级cos=0.75~0.80(部分分离)。
**当前状态**: Token级概念已可区分, 但编码的神经元级机制未知。
**仍需要**:
1. 全层EMB→HS追踪: 概念分离在哪一层最明显?
2. float32因果消融: P173 KL=0是bfloat16精度不足
3. 单维度语义分析: h的第i维对应什么语义特征?

#### 缺口2: 逻辑推理的计算过程 (优先级: 高)

**问题**: 模型如何从"如果A则B"推理出"A→B"? 在哪些层、哪些维度上计算?
**当前状态**: 间接证据。知道信息在中间层整合(cos下降, 维度膨胀), 但推理的具体计算过程未知。
**需要**:
1. 逻辑探测文本: "All A are B. X is A. Therefore X is B."
2. 追踪推理过程中h的变化: 从前提到结论, h如何变化?
3. 因果干预: 修改前提中的h, 观察结论是否改变

#### 缺口3: 风格/逻辑/语法的多维度同时生成 (优先级: 高)

**问题**: 模型如何在生成一个词时同时考虑风格(诗歌/论文)、逻辑(上下文)、语法?
**当前状态**: P158发现不同类别的流方向有差异(cos=0.58), P170发现概念间cos=0.94~0.99。
**需要**:
1. 设计控制变量实验: 同一内容+不同风格, 比较h的差异方向
2. 同一风格+不同内容, 比较h的差异方向
3. 测量风格/内容/语法方向的相互关系(正交?平行?)

#### 缺口4: 词嵌入空间的代数结构 (优先级: 已解决)

**[Phase XXV已解决]**: P169验证了king+queen-man≈woman在hidden state中完全不成立(accuracy=0%)。
embedding space的代数结构被transformer layers彻底"重写"。
**结论**: hidden state的编码机制与embedding space完全不同。

#### 缺口5: 编码的神经元级机制 (优先级: 最高)

**问题**: 语言信息在单个神经元级别是如何编码的?
**当前状态**: Phase XXVIII确认高度分布式编码(F<3.5), Phase XXIX发现Jacobian拉伸和注意力头分工。
**[Phase XXVIII部分解决]**: F-stat<3.5, 没有语义主维度, 每类别用不同维度编码。
**[Phase XXIX部分解决]**: Jacobian拉伸指数增长(L0:0.6→L33:64.5), 极端专注头普遍存在。
**[Phase XXX部分解决]**: 单头消融KL<1e-8(极度冗余); DS7B灵敏度L4饱和(~500K); SAE训练成功但不够稀疏。
**仍需**:
1. ✅ Jacobian跨模型对比 — DS7B L4饱和(~500K), Qwen3增长到77K, GLM4到100K
2. [部分解决] SAE稀疏特征训练 — 训练成功但不够稀疏(100%存活), 需更大L1系数
3. ✅ 注意力头因果消融 — Qwen3/GLM4极度冗余(KL<1e-8), DS7B敏感80倍
4. ❌ 多头联合消融 — top5/top10同时消融的累积效应未测量
5. ❌ 语义方向vs随机方向灵敏度 — 未区分两种方向的灵敏度差异

### 9.3 已有完整解释的特性

1. ✅ **残差流的弯曲路径**: 步角70~84°, 路径比0.745~0.830, 每步近似正交旋转
2. ✅ **信息低维编码**: PR=3~19, PCA90=6~33, 远低于d_model
3. ✅ **操控不可能三角的数学根源**: RMSNorm投影回流形, 全息编码使方向无关
4. ✅ **层级结构**: Early(提取)→Mid(整合)→Late(决策)的维度和曲率变化
5. ✅ **最后层断裂**: cos从0.93→0.09~0.47, 信息从特征空间到决策空间的跃迁
6. ✅ **后续层修复能力**: 94~97%的消融效果被修复(ratio=16~509x)
7. ✅ **RMSNorm非纯缩放**: gamma各维不同改变了方向(cos=0.46~0.57)
8. ✅ **校正方向**: 56~66%正交于h, 残差连接=方向旋转
9. ✅ **词嵌入类比不成立**: hidden state中king+queen-man≠woman(accuracy=0%), embedding结构被破坏
10. ✅ **概念非独立方向(句子级)**: cos=0.94~0.99, 不是正交编码 [注:token级有区分]
11. ✅ **属性非独立维度**: cos=0.89~0.999, 只是概念主方向上的微小扰动
12. ✅ **Token级概念有区分**: inter-cos=0.50~0.71, intra-cos=0.92~0.94
13. ✅ **Embedding→HS变换**: EMB cos=0.04~0.08(独立) → HS cos=0.63(分布式)
14. ✅ **概念token注意力聚焦**: 21~41%(vs 均匀10%)
15. ✅ **float32修复因果消融**: bfloat16 KL=0是精度不足, float32 KL=0.3~6.7
16. ✅ **概念分离关键层=L3**: cos=0.446, 后期融合(cos>0.78)
17. ✅ **推理跃迁层模型特异**: Qwen3=L34, DS7B=L27, GLM4=L39
18. ✅ **DS7B RL极端压缩**: 中间层Top1=98.7%, 几乎1维编码
19. ✅ **单头极度冗余**: Qwen3/GLM4 KL<1e-8; DS7B敏感80倍(RL降低冗余)
20. ✅ **信息灵敏度跨模型差异**: DS7B L4饱和, Qwen3增长到77K, GLM4增长到100K
21. ✅ **Qwen3头功能最分化**: 后期91%=retrieval; DS7B/GLM4几乎全mixed

### 9.4 接下来的大任务规划

**阶段目标**: 从"单头分析"转向"多头联合因果+SAE特征语义+层级信息瓶颈精确定位"。

#### Phase XXVII: float32因果+逻辑推理+概念图谱 ✅ 已完成
- [x] float32因果消融(P176): KL从0变为0.3~6.7
- [x] 全层EMB→HS变换(P177): 概念分离关键层=L3
- [x] 逻辑推理逐层追踪(P178): 跃迁层=L34, 准确率86%
- [x] 属性因果验证(P179): KL(modified)=0.43
- [x] 大规模概念图谱(P180): 28概念4类别, sep=0.041

#### Phase XXVIII: 跨模型全层追踪+SAE分解+单维度语义分析 ✅ 已完成
- [x] 跨模型全层EMB→HS追踪(P181): 三模型一致, 早期分离(L3-L8)
- [x] 跨模型推理跃迁验证(P182): Qwen3=L34, DS7B=L27, GLM4=L39
- [x] SAE稀疏分解(P183): DS7B Top1=98.7%
- [x] 单维度语义分析(P184): F<3.5, 无语义主维度
- [x] 跨模型概念分离层(P185): 最佳=L3-L8

#### Phase XXIX: Jacobian动力学+注意力头+信息流 ✅ 已完成
- [x] Jacobian拉伸测量(P186): L0≈0.6→L33≈64.5
- [x] 极端专注头分析(P187): 每层1-43%头
- [x] 注意力均匀化(P188): 30%→66%
- [x] Gamma深度增长(P189): Qwen3(×184)
- [x] L0信息瓶颈(P190): Qwen3保留3%

#### Phase XXX: 头因果消融+Embedding扰动+SAE+头聚类 ✅ 已完成
- [x] 头因果消融(P191): KL<1e-8(Qwen3/GLM4), DS7B敏感80倍
- [x] Embedding扰动Jacobian(P192): DS7B L4饱和, Qwen3到77K, GLM4到100K
- [x] 简化SAE训练(P193): 100%存活, L17激活199/512
- [x] 头功能聚类(P194): Qwen3后期91%=retrieval

#### Phase XXXI: 概念-属性编码机制破解协议 ✅ 已完成
- [x] 家族骨干提取(P195): L0 intra=0.15~0.22, L1-L3 intra=0.50~0.63, 最佳分离层=早期
- [x] 名词独有残差(P196): 骨干占60~80%, 残差占20~40%, 残差间cos低(独立)
- [x] 属性通道提取(P197): 同类属性intra_cos=0.10~0.18, 跨类cos=-0.18~-0.20(反相关!)
- [x] 因果消融验证(P198): LM Head投影消融KL=0, 属性方向太弱被RMSNorm压回
- [x] 属性注入迁移(P199): sweet属性注入Qwen3可命中top5, red属性无效
- [x] 桥接回路搜索(P200): cos(noun+attr, noun)>0.94, 桥接项很弱, 属性是名词方向的微扰

**核心发现**: 
1. **属性通道反相关**: 跨类型属性cos=-0.18~-0.20(颜色vs味道vs纹理), 说明属性不是独立维度, 而是反方向
2. **属性=名词方向微扰**: cos(noun+attr, noun)>0.94, 属性只是名词方向的微小偏移
3. **最终层投影消融无效**: 在LM Head上移除属性方向KL=0, 因为属性方向范数远小于名词方向
4. **家族骨干在早期层最强**: L0-L3 intra_cos最高, 后期层跨家族cos>0.7(融合)

#### Phase XXXII: 大规模多词类编码机制系统性破解 ✅ 已完成
- [x] 名词编码一般性规律(P201): 10家族全层追踪, 最佳分离层=最终层(sep=0.96)
- [x] 形容词编码机制(P202): 6类形容词, 跨类型cos≈-0.1(反相关), 调制方向cos=0.3-0.7
- [x] 动词编码机制(P203): 5类动词, 动词intra最高(0.87), 主语调制cos>0.89
- [x] 代词编码机制(P204): 6类代词, 代词极度坍缩(dim90=1-6, top1=57-92%)
- [x] 副词编码机制(P205): 5类副词, 副词-形容词cos=0.76(高重叠)
- [x] 介词编码机制(P206): 5类介词, 空间关系cos=0.80-0.83(高度重叠)
- [x] 跨词类统一理论(P207): HAEM模型, 编码维度dim90: noun=8, adj=5, verb=4, pron=1-6, adv=4, prep=5

**核心发现**:
1. **编码维度反比于词类紧凑性**: 动词最紧凑(dim90=4)但intra最高(0.87)
2. **代词1维坍缩**: Qwen3最终层代词dim90=1, top1=92.1%, 由格(case)维度主导(F=76.4)
3. **后期层融合**: 最终层所有词类cos>0.8, 但质心分类准确率=100%
4. **跨类型属性反相关**: 形容词跨类型cos≈-0.1, 暗示属性编码的对抗结构

#### Phase XXXIII: 编码理论精确定量化与因果验证 ✅ 已完成
- [x] 代词坍缩维度解码(P208): 主成分=格(case), F(case)=73.8-76.4, PC1解释90%方差
- [x] 词类分离定量(P209): L0 sil=0.05→L3 sil=0.30→L35 sil=0.24; 质心分类100%准确
- [x] 编码层级范数(P210): ||B_global||=1322, avg||B_pos||=1587, avg||E_word||=348; B_pos/E_word=4.56x
- [x] 词类骨干消融(P211): 自消融KL=0.0001, 交叉消融KL≈0(后期层方向重叠)
- [x] 修饰几何结构(P212): delta_mod≈95-100%正交于h_noun! 属性不是平行缩放而是正交旋转
- [x] 数学理论精炼(P213): HAEM(Hierarchical Additive Encoding Model)完成

**关键修正**:
1. **P212修正了"属性微扰律"**: delta_mod并非50-70%平行于h_noun, 而是95-100%正交! 之前Phase XXXI的结论是cos(noun+attr, noun)>0.94, 但那是指组合后与名词的cos, 不是delta与名词的cos。真实几何: 属性修饰=正交旋转(≈90°), 不是平行缩放。
2. **代词1维=格(case)**: ANOVA F(case)=76.4远超F(person)=0.7, 代词主维度是主格vs宾格vs所有格
3. **B_pos占比97%**: 词类骨干方向占编码总量的97%, 词汇独有残差仅占22%(范数比4.56x)

#### Phase XXXIV: 语言编码机制的神经元与参数级分析 ✅ 已完成
- [x] LM Head几何分解(P214): W_lm SVD S[0]=126.7, S[0]/S[1]=5.2x; 语义差异84%在尾部(>256维)
- [x] 单神经元选择性(P215): F值最高=121(代词维度), 828个神经元F>10; 稀疏激活9.5%
- [x] 注意力子空间(P216): W_k有效秩757(最低), W_q=1738(最高); 层间cos=0.9-0.96
- [x] Logit空间几何(P217): avg margin=0.73, top1 prob=0.21; W_lm行间cos=0.087(JL几乎正交)
- [x] 抽象层级轨迹(P218): apple→entity cos从L0=0.05增至L35=0.76; 抽象方向跨链cos=0.57
- [x] 维度效率(P219): scaling exponent=0.85(DS7B,亚线性), 1.06(Qwen3,线性); 组合dim90=12

#### Phase XXXV: 编码机制的数学本体论 ✅ 已完成
- [x] W_lm首奇异方向解码(P220): S[0]=126.7分量占67%("in"), 1.8%("apple"); u_0对应格式token
- [x] 正交补空间编码(P221): **语义差异84-97%在尾部奇异方向!** 前256维仅13-21%差异能量
- [x] W_lm行向量结构(P222): 星期cos=0.67(最高), 颜色cos=0.25, 反义词cos=0.27; 语义结构显著
- [x] 抽象方向溯源(P223): L0→L35抽象cos从0.05→0.76; 抽象方向85%在W_lm尾部奇异方向
- [x] 旋转矩阵估计(P224): avg旋转角=25-33°, cos(delta,noun)=-0.29; 纯加法误差52-61%
- [x] SHEM统一理论(P225): **SHEM = HAEM + W_lm SVD + JL引理**, 5大数学定律

**SHEM(子空间层级编码模型)五大定律**:
1. **骨干-差异分解律**: W_lm SVD将logit分为骨干子空间(共性)和差异子空间(区分性), 差异84-97%在尾部
2. **几乎正交律(JL)**: 151936个token在d维空间几乎正交, cos≈0, P(误选)≈exp(-dε²/2)
3. **正交旋转律**: 属性修饰δ_mod≈90°正交于被修饰词, 非平行缩放
4. **抽象共享律**: 具体概念共享抽象方向(cos=0.57), 跨词类不共享(cos<0.12)
5. **维度效率律**: 骨干共享(97%)+正交旋转+层级编码+高维JL → 避免维度灾难

---

详见 `research/glm5/docs/AGI_GLM5_MEMO.md` 的完整实验记录。

#### Phase XXXVI: 非线性组合编码与层间传递函数 ✅ 已完成
- [x] 非线性组合模型(P226): 加法误差0.99, 双线性0.98, 二次0.98, 完全0.97; **组合不是简单线性叠加!**
- [x] 层间传递函数(P227): L1是关键断裂层(cos=0.14); 传递矩阵A远非单位矩阵(||A-I||>>||I||)
- [x] FFN kNN机制(P228): **跨词激活完全不重叠(0/10)**, 极端稀疏(3.7%激活); d_ff=9728
- [x] 注意力头特化(P229): 归纳头L1H19=0.95, 位置头L0H1=0.97, 语义头L5H6=0.80
- [x] 因果方程(P230): ||B_global||=109.8, cos(h,B_global)=0.87; R2(B_global+B_pos)=0.61-0.73

#### Phase XXXVII: 跨模型CCA对齐与全局唯一性证明 ✅ 已完成
- [x] CCA子空间对齐(P231): L5-L10 CCA_sim=0.886, L10-L20=0.930; 名词-动词Fisher=3.5→6.4
- [x] JL全局唯一性证明(P232): **P(误选|e=0.2)<=8.83e-18**; 球面容量(cos<0.1)=3.84e5>vocab
- [x] 词嵌入对齐(P233): **W_emb=W_lm(tied)!** cos=1.0; 最终hidden与嵌入cos=0.05-0.09
- [x] 层间传递分解(P234): L0=1.1→L35=541→L36=123(归一化); 增量PCA 90%=5维
- [x] 第一性原理(P235): 5大定理确立语言编码机制的统一方程

**语言编码5大定理(第一性原理)**:
1. **SHEM子空间层级编码定理**: h(w,C) = B_global + a_pos*B_pos + E_word + d_mod + d_ctx + e
2. **几乎正交唯一性定理(JL)**: P(误选) <= N*exp(-d*e^2/2), d=2560时P<1e-18
3. **骨干-差异分解定理**: logit = 骨干logit(80%能量,13%差异) + 差异logit(84-97%差异)
4. **正交旋转编码定理**: h(n+adj) ~ R(25-33°)*h(noun) + e, 修饰=正交旋转
5. **层级构建定理**: L0嵌入→L1-5关系→L6-20抽象→L21-35精细→L36归一化