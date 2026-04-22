"""Update AGI_GLM5_MEMO.md with CCXVI results"""
import os

memo_path = "d:/Ai2050/TransformerLens-Project/research/glm5/docs/AGI_GLM5_MEMO.md"

entry = """

====================================================================
Phase CCXVI: 直接Head Hook因果分析 — 2026/04/22 22:23
====================================================================

核心改进: 用register_forward_hook直接获取Head输出(不用W_o投影),
跨层PC追踪, 条件互信息分析
样本: 12特征 x 3模型 x 3关键层 x 40对 = 1440差分向量


三模型核心结果

1. DS7B (8bit, 28层, d=3584)
   Delta Norms: tense(S) 19→863, question(S) 40→1802, voice(E) 32→2038
   PCA: L0 PC1=10.1%, cum5=20.8%
        L14 PC1=98.6%, cum5=98.9% (!!!几乎1维!)
        L27 PC1=53.9%, cum5=76.9%
   Cohen_d: L0=-0.21, L14=0.02, L27=0.10 (极低!)
   centroid_cos: L0=0.52, L14=0.998, L27=0.92
   Growth: syn=67x vs sem=44x, p=0.14
   Head: 全部MIXED(28/28), spec range=[-0.12,-0.02]
   跨层PC: cos(L0_PC1, L27_PC1)=0.40, 一致性LOW

2. Qwen3 (BF16, 36层, d=2560)
   Delta Norms: tense(S) 3→168, question(S) 11→434, voice(E) 8→436
   PCA: L0 PC1=13.5%, cum5=26.3%
        L18 PC1=5.3%, cum5=17.5%
        L35 PC1=16.2%, cum5=31.1%
   Cohen_d: L0=0.15, L18=-0.16, L35=-0.21
   centroid_cos: L0=0.53, L18=0.19, L35=0.087 (!!!几乎正交!)
   Growth: syn=85x vs sem=44x, p=0.14
   Head: 全部MIXED(32/32), spec range=[-0.23,-0.10]
   跨层PC: cos(L0_PC1, L35_PC1)=0.37, 一致性LOW

3. GLM4 (8bit, 40层, d=4096)
   Delta Norms: tense(S) 0→78, question(S) 0→251, voice(E) 0→234
   PCA: L0 PC1=4.6%, cum5=15.8%
        L20 PC1=5.2%, cum5=16.2%
        L39 PC1=14.1%, cum5=33.6%
   Cohen_d: L0=0.09, L20=0.33, L39=0.005
   centroid_cos: L0=0.51, L20=0.15, L39=0.54
   Growth: syn=794x vs sem=528x, p=0.34
   Head: 全部MIXED(32/32), spec range=[-0.16,-0.00]
   跨层PC: cos(L0_PC1, L39_PC1)=0.022 (!!!几乎无关!)
   definiteness growth=2751x (最大)


★★★ 关键发现 ★★★

1. DS7B L14因果空间几乎1维! PC1=98.6%, cum5=98.9%
   → 这意味着Distill模型在中间层有"信息瓶颈"
   → 所有语法/语义差异被压缩到1个维度上
   → L14的centroid_cos=0.998 — 语法/语义完全不可分
   → 这是Distill退化的核心证据

2. Qwen3 L35 centroid_cos=0.087 — 语法/语义方向几乎正交!
   → 非Distill模型在后期层保持子空间分离
   → 与DS7B L27的0.92形成鲜明对比

3. GLM4 L0_PC1 vs L39_PC1 cos=0.022 — 跨层PC几乎无关!
   → GLM4的PC方向在层间剧烈旋转
   → "纤维丛"的平行运输(Parallel Transport)效果极差
   → 这说明因果方向在层间不保守

4. Head全部MIXED(三模型一致, 3层x3模型=9个测试)
   → 直接Hook也确认: Head不按语法/语义专化
   → spec range很小([-0.23, -0.00]), 所有Head贡献均匀
   → 这是一个**否定性结论**: Head不是因果原子的载体

5. Feature separability排名一致:
   question(S)和voice(E)始终排前两位
   → question和voice是最"独立"的因果原子候选

6. 条件互信息: 三模型三层全部MIXED
   → Intra-SYN ≈ Intra-SEM ≈ Cross
   → 语法/语义在PC空间中不形成分离的簇


与CCXV对比

CCXV: centroid_cos(Qwen3 L35)=0.198, GLM4 L39=0.220
CCXVI: centroid_cos(Qwen3 L35)=0.087, GLM4 L39=0.541
→ 不一致! 因为CCXVI用3层(CCXV用5层), 采样数不同

CCXV: Growth p<0.01 (三模型显著)
CCXVI: Growth p>0.14 (三模型不显著)
→ 样本数增加后growth差异反而不显著了!
→ CCXV的显著性可能是假阳性(5层采样=3个数据点太少)

CCXV: Head全部MIXED (W_o投影)
CCXVI: Head全部MIXED (直接Hook)
→ 两种方法一致确认: Head不专化


最严格审视

1. DS7B L14 PC1=98.6%: 这可能是8bit量化导致的异常!
   8bit模型的中间层表示可能被压缩到极低精度
   → 需要用FP16模型验证

2. Growth p>0.14: CCXV的p<0.01可能是假阳性
   → 只有3-5个层的数据点做MW检验,统计力不足
   → 需要更多层的数据(如全28/36/40层)

3. Head全部MIXED: 可能是分析粒度问题
   → Head可能不是按"语法/语义"二分的
   → 而是按更细粒度的功能(如tense/person/question)分
   → 需要做per-feature的Head贡献分析

4. GLM4 L0 delta norm=0: 8bit截断导致L0分析完全不可靠

5. centroid_cos不稳定: 同一模型不同层/不同采样得到不同结果
   → 需要更稳健的子空间度量(如CCA, CKA)


数学框架更新

V_causal(L) = span(PC1, PC2, ..., PC_k)

DS7B: dim(V_causal) = 1 (L14) → 5 (L27)  [信息瓶颈→恢复]
Qwen3: dim(V_causal) ≈ 5-10 (稳定)
GLM4: dim(V_causal) ≈ 5-10 (稳定)

关键洞察: 
- 因果空间是低维的(5-10PC解释20-34%方差)
- 但因果方向在层间不保守(平行运输失效)
- Head是"民主"的贡献者,不是专化的功能单元
- question/voice是最独立的因果原子


下一步突破

1. Per-feature Head贡献分析: 不按语法/语义二分,
   而是分析每个Head对每个特征(tense/person/question...)的贡献
   → 可能发现Head按"具体语法功能"专化

2. 全层扫描: 用更高效的方法扫描所有28/36/40层,
   验证DS7B L14的信息瓶颈是否真实

3. CCA/CKA子空间度量: 替代centroid_cos,
   提供更稳健的子空间相似性度量

4. 因果原子词典: 基于separability排名,构建
   {question, voice, definiteness, ...}的因果原子词典

5. SAE训练: 对L27/35/39差分向量训练SAE,
   发现可解释的因果方向(这是通向"数学结构"的关键)
"""

with open(memo_path, 'a', encoding='utf-8') as f:
    f.write(entry)

print("MEMO updated!")
