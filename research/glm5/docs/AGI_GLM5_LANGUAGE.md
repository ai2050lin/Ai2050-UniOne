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

## 2. 统一理论 v5.0（最新版本）

基于 P15-P65（Stage 662-707）全部实验结果。v4.1→v5.0主要修正：
- C常数被推翻：C=1.3-15.2（任务依赖，非1.5）
- rank=9被推翻：真实维度>89-100（样本数限制）
- 编码基元确认：方向+门控幅度（门控主导）
- 新发现：h PCA ⊥ unembed PCA，语义空间是涌现结构

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
10. **语义方向的因果验证缺失**——P66待修复，需要验证干预PC坐标的因果效应
11. **跨模型语义方向对齐未知**——不同模型的PC0是否编码相同语义？P67待修复

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
**P52（线性累积）**→ logits=sum(delta_h@U.T), cos>0.995
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

**Phase IV：大样本验证（P65-P67）⏳ 部分完成**
- 核心：用大样本重新评估有效维度
- 结果：100文本rank=89-100，Top-5 PC解释82-90%方差

### 6.3 执行顺序

```
Phase I（生成化 P50-P54）✅
  → Phase II（判伪化 P55-P59）✅（P57/P59待外部依赖）
  → Phase III（基元化 P60-P64）✅
  → Phase IV（大样本验证 P65-P67）⏳ 部分完成
  → Phase V（语义方向提取+因果验证）待启动
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

### 模型共享库
- `tests/codex/multimodel_language_shared.py`
- `tests/codex/qwen3_language_shared.py`

---

## 9. 当前阶段最严格的表述

如果只保留一段最严格的话：

> **残差流的宏观几何（L0对齐≈1.0、L0→L1旋转≈90度、旋转平面11维且跨文本一致）主要由架构+初始化决定，不由语义内容决定。但逐层微调偏移（delta-h_l = h_l - h_{l-1}）的方向与语义类别高度相关（ANOVA F>3, P50确认），且这些delta-h通过线性累积精确地决定输出logits（cos>0.995, P52 INV-354）。P55判伪确认：norm缩放0.1x-20x不影响top-1（r>0.996），语义完全由方向编码。P60基元消融发现：E_gate（门控幅度）和E_subspace（子空间投影）是最重要的基元（消融后top-1=0%, r<0.3），而C_trajectory和D_attention无影响(因P52仅依赖h_final)。P62组合测试确认：A+E无正向协同，门控幅度E单独主导。P64/P65大样本验证揭示：(1)rank=9被推翻！100文本rank=89-100,每类20文本=20(满秩)，之前的"rank=9"仅因10条文本的样本数上限。(2)h_final PCA与unembed PCA几乎正交(alignment≈0)，语义编码方向是模型内部涌现的独立结构。(3)Top-5 PC解释88-90%方差：PC0≈语言类型(英文vs中文,占22-43%方差)，PC1≈文本格式/语言，PC2≈领域(Math/Science)。(4)仅5维度即解释>88%方差，但文本间的区分需要>20维度。编码基元=方向+门控幅度(门控主导)。avg_lp≈log(margin)-C(task)。**

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