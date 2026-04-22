"""Update AGI_GLM5_MEMO.md with CCXVII results"""
import os

memo_path = "d:/Ai2050/TransformerLens-Project/research/glm5/docs/AGI_GLM5_MEMO.md"

entry = """

====================================================================
Phase CCXVII: Per-feature Head贡献矩阵 + 全层PCA扫描 — 2026/04/22 23:30
====================================================================

核心改进: 
1. 全层PCA扫描(每4层采样), 验证信息瓶颈
2. [Head x Feature]贡献矩阵, 发现Head按功能专化
3. Head聚类分析, 检查语法/语义专化
4. Hook方式替代output_hidden_states(避免8bit OOM)

样本: 12特征 x 3模型 x ~11采样层 x 40对


★★★ 最重要的发现: 所有模型都有信息瓶颈! ★★★

DS7B: 信息瓶颈在L4! PC1=99.33%
  L0: PC1=9.3%  → L4: PC1=99.3% → L8: PC1=98.5% → ... → L27: PC1=40.4%
  L12最正交: cos=0.0486
  Growth: syn=3.6x vs sem=3.9x, p=0.77 (不显著)

Qwen3: 信息瓶颈在L8! PC1=99.54%
  L0: PC1=11.9% → L4: PC1=80.9% → L8: PC1=99.5% → ... → L35: PC1=9.7%
  L8最正交: cos=0.0436
  Growth: syn=8.9x vs sem=5.6x, p=0.024 (显著!)

GLM4: 无信息瓶颈! PC1始终<14%
  L0: PC1=3.9% → L20: PC1=4.1% → L32: PC1=6.7% → L39: PC1=13.2%
  L32最正交: cos=0.0624
  Growth: syn=71x vs sem=63x, p=0.23 (不显著)
  definiteness growth=110x


信息瓶颈的详细演变

DS7B:
  L0:  PC1=9.3%,  cum5=18.6%, cos=0.38
  L4:  PC1=99.3%, cum5=99.8%, cos=0.93  ← 瓶颈!
  L8:  PC1=98.5%, cum5=99.6%, cos=0.83
  L12: PC1=98.0%, cum5=99.6%, cos=0.05  ← 最正交!
  L16: PC1=97.8%, cum5=99.4%, cos=0.21
  L20: PC1=97.2%, cum5=98.9%, cos=0.50
  L24: PC1=93.5%, cum5=96.2%, cos=0.53
  L27: PC1=40.4%, cum5=64.7%, cos=0.76

Qwen3:
  L0:  PC1=11.9%, cum5=21.7%, cos=0.40
  L4:  PC1=80.9%, cum5=82.9%, cos=0.08
  L8:  PC1=99.5%, cum5=99.6%, cos=0.04  ← 瓶颈!
  L12: PC1=99.3%, cum5=99.6%, cos=0.05
  L16: PC1=99.1%, cum5=99.5%, cos=0.26
  L20: PC1=99.2%, cum5=99.3%, cos=0.11
  L24: PC1=97.5%, cum5=97.9%, cos=0.13
  L28: PC1=92.9%, cum5=93.9%, cos=0.09
  L32: PC1=81.8%, cum5=84.8%, cos=0.15
  L35: PC1=9.7%,  cum5=27.1%, cos=0.44  ← 恢复高维!

GLM4:
  L0:  PC1=3.9%,  cum5=14.3%, cos=0.78
  L4:  PC1=4.0%,  cum5=12.8%, cos=0.35
  L8:  PC1=4.2%,  cum5=14.4%, cos=0.35
  L12: PC1=4.5%,  cum5=14.6%, cos=0.36
  L16: PC1=3.8%,  cum5=15.8%, cos=0.28
  L20: PC1=4.1%,  cum5=16.2%, cos=0.22
  L24: PC1=5.0%,  cum5=16.1%, cos=0.15
  L28: PC1=5.0%,  cum5=14.8%, cos=0.09
  L32: PC1=6.7%,  cum5=15.9%, cos=0.06  ← 最正交!
  L36: PC1=7.8%,  cum5=16.5%, cos=0.19
  L39: PC1=13.2%, cum5=34.9%, cos=0.35


Per-feature Head贡献矩阵 (最后一层)

DS7B L27: 所有特征的Top2 Heads都是H20和H3
  H20贡献≈2.3x H3, H3贡献≈2.3x H21
  question: H20=1784, H3=741
  formality: H20=1475, H3=650
  → DS7B有2个超级Head(H20, H3), 对所有特征贡献最大

Qwen3 L35: 所有特征的Top3 Heads都是H0, H1, H2
  H0贡献≈1.8x H1
  question: H0=35.6, H1=20.3
  → Qwen3有3个超级Head(H0, H1, H2)

GLM4 L39: 所有特征的Top Head都是H18
  question: H18=5.9, H12=3.3
  → GLM4有1个超级Head(H18)


Head聚类分析

DS7B: syn_prefs在0.65-0.83之间 → Head不按语法/语义分
Qwen3: syn_prefs在0.54-0.88之间 → Head不按语法/语义分
GLM4: syn_prefs在0.62-0.88之间 → Head不按语法/语义分

→ 三模型一致: Head聚类无法区分语法/语义!
→ Head是"民主"的, 对所有特征贡献比例相似
→ 但存在"超级Head"(贡献量远超平均), 超级Head是通用的


因果原子词典 (最后一层, 按PC1 centroid排序)

DS7B L27: question > definiteness > polarity > voice > info_structure
Qwen3 L35: question > voice > formality > person > sentiment
GLM4 L39: question > voice > info_structure > negation > formality

→ question始终排第一! (最强的因果原子)
→ voice始终排前二! (最强的语义原子)
→ definiteness在DS7B排第2, Qwen3排第9, GLM4排第8 (不稳定)


★★★ 核心数学发现 ★★★

1. 因果空间的信息瓶颈结构:
   DS7B/Qwen3: V_causal(L_early) → 1维瓶颈 → V_causal(L_late)
   GLM4: V_causal(L) 始终保持~5-15维
   
   解释: Distill模型在早期层压缩因果信息到1维,
   非Distill模型(GLM4)保持高维因果空间

2. 瓶颈后正交性先升后降:
   DS7B: cos=0.05(L12) → 0.76(L27)
   Qwen3: cos=0.04(L8) → 0.44(L35)
   GLM4: cos=0.06(L32) → 0.35(L39)
   
   → 瓶颈后语法/语义最分离, 后期逐渐融合

3. 超级Head现象:
   每个模型有2-3个超级Head, 贡献远超其他Head
   但超级Head对所有特征的贡献比例相同 → 不是按功能专化的
   
   数学解释: 超级Head可能编码了"因果流的主要通道"
   类似于网络中的"中枢节点"

4. Qwen3 Growth p=0.024 (唯一显著的!)
   → Qwen3的语法growth显著大于语义growth
   → 但DS7B(p=0.77)和GLM4(p=0.23)不显著
   → 可能是Qwen3特有效应, 不是普遍性质
"""

with open(memo_path, 'a', encoding='utf-8') as f:
    f.write(entry)

print("MEMO updated!")
