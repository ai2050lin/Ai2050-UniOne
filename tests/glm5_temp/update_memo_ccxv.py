"""Update AGI_GLM5_MEMO.md with Phase CCXV results"""
import os
from datetime import datetime

now = datetime.now().strftime('%Y-%m-%d %H:%M')

memo = """
=================================================================
Phase CCXV: 因果原子分解 (差分向量+PCA+W_o投影) (TIMESTAMP)
=================================================================

三模型: DS7B(8bit,28层), Qwen3(BF16,36层), GLM4(8bit,40层)
n_pairs=120, 5层/模型, 12特征(8语法+4语义)

=== S1: Delta Norms ===

DS7B: tense=16,54,87,161,744 | sentiment=32,106,145,234,928
Qwen3: tense=3,17,23,57,164 | sentiment=9,35,39,96,207
GLM4: tense=0,2,4,18,68 | sentiment=1,3,8,31,106

=== S2: PCA原子分解 ===

DS7B:
  L0: PC1=9.1%, cum5=17.8%, centroid_cos=0.157
  L27: PC1=50.7%, cum5=75.4%, centroid_cos=0.943
  → L27极度低维! 5PC解释75%方差, 但语法/语义完全融合

Qwen3:
  L0: PC1=6.1%, cum5=16.7%, centroid_cos=0.142
  L35: PC1=11.9%, cum5=32.3%, centroid_cos=0.198
  → 语法/语义方向更分离(cos=0.14-0.33)

GLM4:
  L0: PC1=3.7%, cum5=14.6%, centroid_cos=0.202
  L39: PC1=14.3%, cum5=33.3%, centroid_cos=0.220
  → 8bit量化导致L0值极小(0-1), 但PCA仍有信号

=== S3: W_o Head贡献 ===

DS7B (dequantize成功):
  L0: Top SYN=[H6,H5,H1], Top SEM=[H21,H3,H10]
  L27: Top SYN=[H3,H13,H11], Top SEM=[H22,H6,H20] (spec很弱)
  → 所有Head都是MIXED, 但L0有更强的专化信号

Qwen3: W_o shape=(2560,4096) — GQA结构, 不兼容
GLM4: L39 dequantize失败, L0-L30可用但Head都是MIXED

=== S4: 因果原子发现 ===

跨模型一致: question是最具分离度的语法特征 (sep=5.0-10.8)
跨模型一致: semantic_topic是最具分离度的语义特征 (sep=1.3-3.2)

DS7B L27: 语法PC1_spread=379.8, 语义=294.6 (后期层差异最大)
GLM4: 语法PC1_spread增长: 0.04→56.6, 语义: 0.05→13.7 (语法增长远超语义)

=== S5: 统计检验 ===

Growth MW检验:
  DS7B: p=0.0020 (语法62.7x vs 语义35.7x)
  Qwen3: p=0.0081 (语法89.5x vs 语义32.4x)
  GLM4: p=0.0040 (语法700.2x vs 语义268.3x)
  → 三模型一致: 语法growth显著大于语义growth!

Centroid余弦相似度:
  DS7B: L0=0.157, L27=0.943 (后期融合)
  Qwen3: L0=0.142, L35=0.198 (方向保持分离)
  GLM4: L0=0.202, L39=0.220 (方向保持分离)

=== ★核心发现 ===

1. DS7B L27的因果空间极度低维! PC1=50.7%, 5PC=75.4%
   但centroid_cos=0.943 — 语法/语义方向在后期层融合

2. Qwen3和GLM4的语法/语义方向保持低相似度(0.14-0.36)
   → 非Distill模型保持了更好的子空间分离

3. 语法growth >> 语义growth (三模型p<0.01)
   → 这是最稳健的统计发现

4. question是最具分离度的特征 (语法原子的候选)
   semantic_topic是最具分离度的语义特征 (语义原子候选)

5. definiteness growth最大: Qwen3=245x, GLM4=1544x
   → definiteness是"最语法"的语法特征

=================================================================
最严格审视
=================================================================

1. DS7B L27 centroid_cos=0.943: Distill后期层信息融合, 
   语法/语义方向不再可分 — 这是Distill的退化还是普遍现象?
2. Qwen3/GLM4的centroid_cos=0.14-0.36: 低于CCXIII的
   cross_overlap=0.000, 因为CCXIII用的是k维子空间overlap,
   这里用的是PC1-5 centroid余弦, 度量不同
3. W_o Head分解在GQA模型(Qwen3)中不兼容
4. GLM4 L0 delta norms为0: 8bit截断
5. Head全部MIXED(spec<0.3): W_o投影方法可能不够敏感,
   或者Head确实不是按语法/语义专化的

=================================================================
下一步突破
=================================================================

1. 直接Head hook: 不用W_o投影, 直接hook每个Head的输出,
   计算Head对差分向量的贡献 — 避免GQA/8bit兼容问题
2. 因果原子SAE: 对L27/35/39的差分向量训练SAE,
   发现可解释的因果方向
3. 跨层纤维丛: 追踪同一PC成分在层间的演变轨迹
4. 非Distill验证: 更多Qwen3/GLM4分析,
   确认子空间分离是普遍性质
"""

with open('research/glm5/docs/AGI_GLM5_MEMO.md', 'a', encoding='utf-8') as f:
    f.write(memo.replace('TIMESTAMP', now))
print('MEMO updated at ' + now)
