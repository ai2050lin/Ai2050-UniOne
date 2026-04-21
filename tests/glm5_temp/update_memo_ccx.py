"""Update MEMO with Phase CCX results"""
from datetime import datetime
from pathlib import Path

now = datetime.now().strftime('%Y%m%d%H%M')

memo = """
=================================================================
Phase CCX: 语法/语义因果对偶验证 ({now})
=================================================================

核心改进: number扩充到100模板, 新增negation/question, 全部120对
7特征 x 8层 x 120对 = 6720组patching/模型

S1. Residual全层扫描 (median_l2, n=120)
=================================================================

DS7B (28层):
  Layer | tense | polarity | number | negation | question | sentiment | topic
  L0    |  140  |   134    |  131   |   116    |   109    |   674     |  722
  L7    |  246  |   319    |  202   |   311    |   281    |   711     |  818
  L15   |  306  |   463    |  269   |   420    |   591    |   803     |  954
  L27   |  480  |   743    |  383   |  1000    |  1073    |   869     |  962

Qwen3 (36层):
  Layer | tense | polarity | number | negation | question | sentiment | topic
  L0    |   18  |    16    |   21   |    16    |    17    |   404     |  509
  L10   |  102  |   103    |  107   |   111    |   112    |   394     |  537
  L20   |  135  |   151    |  176   |   175    |   177    |   362     |  526
  L35   |  241  |   259    |  303   |   286    |   382    |   421     |  629

GLM4 (40层):
  Layer | tense | polarity | number | negation | question | sentiment | topic
  L0    |   53  |    54    |   54   |    53    |    53    |   418     |  501
  L11   |  109  |   133    |  102   |   138    |   131    |   386     |  520
  L22   |  148  |   209    |  182   |   228    |   343    |   368     |  532
  L39   |  180  |   249    |  251   |   260    |   412    |   407     |  576

S2. 对偶分类 (growth_ratio > 1.5 = SYNTACTIC)
=================================================================

DS7B:
  SYNTACTIC: tense=3.43x, polarity=5.53x, number=2.92x, negation=8.64x, question=9.85x
  SEMANTIC:  sentiment=1.29x, semantic_topic=1.33x

Qwen3:
  SYNTACTIC: tense=13.43x, polarity=15.86x, number=14.35x, negation=17.88x, question=23.04x
  SEMANTIC:  sentiment=1.04x, semantic_topic=1.24x

GLM4:
  SYNTACTIC: tense=3.38x, polarity=4.63x, number=4.63x, negation=4.89x, question=7.83x
  SEMANTIC:  sentiment=0.97x(!), semantic_topic=1.15x

S3. 代数拟合 (BEST R2)
=================================================================

DS7B:
  tense=linear(0.970), polarity=linear(0.846), number=linear(0.939)
  negation=exponential(0.913), question=linear(0.979)
  sentiment=linear(0.837), semantic_topic=power_law(0.860)

Qwen3:
  tense=linear(0.985), polarity=linear(0.988), number=linear(0.993)
  negation=power_law(0.993), question=linear(0.978)
  sentiment=power_law(0.062)!!!, semantic_topic=exponential(0.808)

GLM4:
  tense=power_law(0.987), polarity=power_law(0.974), number=linear(0.984)
  negation=power_law(0.969), question=linear(0.937)
  sentiment=power_law(0.273)!!!, semantic_topic=exponential(0.947)

S4. Attn vs MLP
=================================================================

DS7B: L0=50:50, L27=50:50 (异常, Distill影响)
Qwen3: L0=52:48, L35=24:76 (MLP主导末层)
GLM4: L0=50:50, L39=23:77 (MLP主导末层)

GLM4 L0 sentiment: attn=13%, mlp=87% (语义在L0就由MLP主导!)

S5. 对偶假设统计检验 (Mann-Whitney U)
=================================================================

三模型一致:
  Growth ratio: U=10.0, p=0.0476 (syntactic > semantic, 显著!)
  L0 value: U=10.0, p=0.0476 (semantic > syntactic, 显著!)

=================================================================
Phase CCX 核心发现 (三模型一致, n=120)
=================================================================

1. 语法/语义因果对偶: 统计显著 (p<0.05)
   - 语法: growth_ratio=2.9-23x, L0=16-140
   - 语义: growth_ratio=0.97-1.33x, L0=394-722
   - 两种完全不同的因果动力学!

2. number扩充到100模板后确认: 也是SYNTACTIC!
   - DS7B: growth=2.92x, linear R2=0.939
   - Qwen3: growth=14.35x, linear R2=0.993
   - GLM4: growth=4.63x, linear R2=0.984
   - 之前number只有20对, R2低=0.81, 现在120对确认!

3. negation和question也是SYNTACTIC
   - negation: growth=4.89-17.88x, R2>0.91
   - question: growth=7.83-23.04x, R2>0.93
   - 全部5种语法特征一致!

4. sentiment R2极低: Qwen3=0.062, GLM4=0.273
   - 几乎完全随机, 没有任何增长趋势
   - GLM4: growth=0.97x (负增长!)
   - 语义因果效应是常数函数!

5. question增长最大: 7.83-23.04x
   - 疑问句变换需要更多层间累积
   - negation增长也很大: 4.89-17.88x

=================================================================
Phase CCX 硬伤与瓶颈
=================================================================

硬伤1: p=0.0476刚好在0.05边界
  - 语法5特征 vs 语义2特征, 样本不平衡
  - 需要更多语义特征(如: 主动/被动, 人称变换, 语域变换)
  - Mann-Whitney U检验统计力不足

硬伤2: 语义特征只有2种
  - sentiment和semantic_topic都偏向"情感/主题"类型
  - 缺少: 句法角色(主语/宾语), 信息结构(焦点/预设), 语用特征
  - 二分法可能只是这2种语义的特殊性质

硬伤3: DS7B的Attn/MLP比异常
  - 50:50不变, 与Qwen3/GLM4的77:23完全不同
  - DeepSeek-R1-Distill的知识蒸馏改变了内部结构

硬伤4: negation的exponential增长(DS7B)
  - DS7B negation=exponential, 其他模型=power_law/linear
  - 可能是Distill模型的特殊性质

=================================================================
新型SAE分析
=================================================================

默认SAE的困境:
  标准SAE在残差流上训练, 重建全局稠密表示
  问题: 残差流是所有信息的叠加, SAE挖出的是统计频率特征
  无法区分: 语法贡献 vs 语义贡献 vs 噪声

新型SAE方案 (7种):

1. Head-wise独立头SAE
   - 每个Attention Head单独训练SAE
   - 优势: 捕捉Head的局部因果功能
   - 与本研究对齐: 已发现Head特异性(phase CCVI)
   - 实现: 对每个Head output做SAE分解

2. 去全局基线SAE (Subtract-then-SAE)
   - 先减去全局均值, 再对残差做SAE
   - 优势: 去掉"背景辐射"(L0就存在的语义信息)
   - 与本研究对齐: 语义L0就大, 是全局基线
   - 实现: h_resid - mean(h_resid) -> SAE

3. 差分Delta因果SAE (Delta-SAE)
   - 对因果差分向量 delta = h(source) - h(clean) 做SAE
   - 优势: 直接分解因果效应的组成
   - 与本研究完美对齐: 我们已经计算了差分向量!
   - 关键洞察: 语法delta逐层增长, 语义delta恒定
   - Delta-SAE可以发现: 语法原子 vs 语义原子

4. 分块低维SAE (Block-SAE)
   - 按因果子空间分组, 每组独立训练小SAE
   - 优势: 低维更稀疏, 特征更可解释
   - 与本研究对齐: 因果空间约5个正交子空间(phase CCV)

5. 单Token局部SAE (Local-SAE)
   - 在单个token位置的局部窗口训练SAE
   - 优势: 捕捉位置特异的因果功能
   - 与本研究对齐: 不同token位置的因果贡献不同

6. 正交正则SAE (Ortho-SAE)
   - 在SAE损失中加入正交性约束
   - 优势: 强制不同特征正交, 避免特征纠缠
   - 与本研究对齐: 因果子空间近似正交(phase CCV)

7. 分层纤维SAE (Fiber-SAE)
   - 按层分组, 每层(或层组)训练独立SAE
   - 优势: 捕捉层间因果纤维的演化
   - 与本研究完美对齐: 语法因果纤维逐层增长
   - 关键: 可以追踪同一个因果原子在不同层的"生长"

最推荐: Delta-SAE + Fiber-SAE 组合
  Delta-SAE: 分解因果效应的原子组成
  Fiber-SAE: 追踪原子在层间的演化路径
  组合: Delta-Fiber-SAE = 因果纤维的原子分解

这与本研究的全部发现完全对齐:
  - 语法Delta逐层增长 => Fiber-SAE追踪增长
  - 语义Delta恒定 => Delta-SAE发现不变原子
  - 因果空间5子空间 => Block-SAE分组分解
  - Head特异性 => Head-wise SAE

=================================================================
第一性原理: 语言背后数学原理更新
=================================================================

核心洞察 (Phase CCX确认):
  1. 因果效应的二分法:
     语法: C(l) = a*l + b (线性/幂律增长)
     语义: C(l) = C_0 (常数)
     这是Transformer内部的两类根本不同的算子!

  2. 类比信号处理:
     语法 = 时变信号 (逐层积分): S(l) = integral_0^l f(t) dt
     语义 = 直流偏置 (embedding直接注入): D(l) = D_0
     总因果效应 = S(l) + D_0

  3. Transformer的信息流模型:
     h_l = h_0 + sum_{k=0}^{l-1} [Attn_k(h_k) + MLP_k(h_k)]
     因果效应 = delta_h_l = delta_h_0 + sum delta_transforms
     语法: delta_h_0 ~ 0, sum delta_transforms >> 0
     语义: delta_h_0 >> 0, sum delta_transforms ~ 0

  4. 这解释了为什么MLP主导末层:
     MLP是"积分器", 逐层累积语法信息
     语法因果效应 = sum MLP_k的贡献
     语义因果效应 = embedding层的投影, 不需要MLP累积

下一步突破:
  1. Delta-SAE: 对差分向量做SAE, 分解因果原子
  2. Fiber-SAE: 追踪因果原子在层间的演化
  3. 更多语义特征: 主动/被动, 人称, 语域, 信息结构
  4. 非Distill模型: 原版Qwen-7B, Llama-7B
  5. 数学证明: 积分算子+投影算子的精确代数结构
"""

with open('research/glm5/docs/AGI_GLM5_MEMO.md', 'a', encoding='utf-8') as f:
    f.write(memo)
print(f'MEMO updated at ' + now)
