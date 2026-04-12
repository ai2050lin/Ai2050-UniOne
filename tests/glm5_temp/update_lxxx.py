"""Update AGI_GLM5_LANGUAGE.md with Phase LXXX results and AGI_GLM5_MEMO.md"""

# Phase LXXX new section
new_section = """
#### Phase LXXX: LCS精确化 + DS7B非线性机制 + 7维语言空间 (P399-P402)

**P399: proj(dh, W_diff)的完整投影分布** (★★★信号分配效率极低★★★)

★★★ **核心发现：信号分配效率(signal_efficiency)仅为百万分之一！** ★★★

| 模型 | signal_efficiency | total_energy | ratio_actual/proj |
|------|-------------------|--------------|-------------------|
| Qwen3 | 1e-6 ~ 3e-6 | 1-2.3M | 0.12-0.21 |
| GLM4 | 1.6e-5 ~ 4.4e-5 | 256K-332K | 0.98-1.12 |
| DS7B | ~0 | 3.9M-50M | -0.05~0.19 |

signal_efficiency = |Δlogit(pos-neg)| / Σ|Δlogit_i| (对所有词)

关键发现：
1. **GLM4 ratio≈1.0**：proj(dh, W_diff) ≈ Δlogit，信号分配最精确
2. **Qwen3 ratio≈0.15**：proj是Δlogit的5-7倍，85%的投影"泄漏"到其他词
3. **DS7B ratio极不稳定**：从-0.05到0.19，信号分配最不可预测
4. **DS7B total_energy=5000万**！dh_norm≈2000在15万词上产生巨大总能量
5. **Top tokens与目标词无关**：style维度注入，top tokens是"heavier/falling/sinking"

**信号分配的物理解释**：
```
dh (hidden state变化) 在 lm_head 映射后:
  Δlogit_i = W_lm[i] · dh  (对所有151936个词)

  只有 ~1e-6 到 4e-5 的总能量分配给了目标维度(pos-neg差值)
  其余能量分配给了其他15万+词

  这意味着: W_lm矩阵像一个"极度弥散的透镜"
  注入的信号被W_lm"散射"到整个词空间
  只有极小比例恰好落在目标维度上
```

**P400: 非线性映射机制**

| 指标 | Qwen3 R² | GLM4 R² | DS7B R² |
|------|----------|---------|---------|
| proj(线性) | 0.813 | 0.973 | 0.354 |
| proj²(二次) | 0.852 | 0.979 | 0.351 |
| proj×cos(交叉) | 0.847 | 0.978 | 0.580 |
| 多特征top3 | 0.861 | 0.979 | 0.680 |

关键发现：
1. **GLM4近乎完美线性**：proj R²=0.973，最高0.979
2. **Qwen3需要轻微非线性**：proj² R²=0.852 > proj R²=0.813
3. **DS7B高度非线性**：单特征最大R²仅0.580(proj×cos)，多特征0.680
4. **DS7B的Δlogit/beta不稳定**：logic维度beta=4→0.17, beta=16→0.53

**P401: LCS精确矩阵估计**

**R_rotate逐层旋转**：

| 层段 | Qwen3 mean_diag_cos | GLM4 mean_diag_cos | DS7B mean_diag_cos |
|------|---------------------|--------------------|--------------------|
| L0→L4 | 0.659 | 0.972 | 0.538 |
| L8→L12 | 0.571 | 0.881 | 0.779 |
| L16→L20 | 0.783 | 0.833 | 0.819 |
| L24→L28 | 0.787 | 0.790 | 0.278(L27) |
| L32→L36 | - | 0.487(L39) | - |
| L32→L35 | 0.786 | - | - |

★★★ **GLM4 L36→L39: diag_cos暴跌到0.487！** 末2层信号方向剧烈旋转！★★★
★★★ **DS7B L24→L27: diag_cos=0.278！** 最后4层维度方向几乎完全重组！★★★

**C_compete精确竞争矩阵**（4×4，style/logic/grammar/sentiment）：

GLM4竞争矩阵（非对角线=对dim1的交互效果）：
```
        style   logic   grammar  sentiment
style      -    -1.320   +1.289    +4.004
logic   +4.359     -    +8.449    +4.408
grammar -6.014  +7.242     -     -0.641
sentiment +2.068 +8.025 +4.324      -
```
→ GLM4: 逻辑维度在联合注入时被增强(8.449, 8.025)！逻辑维度是"增强型"

Qwen3竞争矩阵：
```
        style   logic   grammar  sentiment
style      -    -1.750   +0.836    +0.461
logic   +1.354     -    +4.439    -0.428
grammar -2.395  -0.102     -     -6.590
sentiment -0.852 -2.546 -5.074      -
```
→ Qwen3: grammar-sentiment互相抑制(-6.590, -5.074)——表达形式和情感的"冲突"

**P402: 7维语言空间验证** (★★★LCS预测力★★★)

| 模型 | quantity实际Δlogit | proj(dh, W_diff) | ratio | 7维有效秩 |
|------|-------------------|-------------------|-------|----------|
| Qwen3 | 2.375 | 21.453 | 0.111 | 7 |
| GLM4 | 12.016 | 10.270 | 1.170 | 7 |
| DS7B | -1.703 | -0.056 | -30.6 | 7 |

★★★ **三模型全部：7维SVD有效秩=7！quantity构成第7维！** ★★★
→ 7维正交语言空间跨模型确认：style/logic/grammar/sentiment/tense/certainty/quantity
→ quantity与已知6维的cos全部<0.055

**LCS预测力**：
- GLM4: Δlogit ≈ proj(dh, W_diff)，ratio=1.17，预测力优秀
- Qwen3: Δlogit ≈ 0.11 × proj(dh, W_diff)，比例稳定
- DS7B: proj≈0但Δlogit=-1.7，预测力极差——信号完全被"吸收"到其他方向

#### 信号分配效率：语言计算的核心瓶颈

★★★ **最重要的新认知** ★★★

P399揭示了语言计算的一个根本性质——**信号分配效率极低**：

```
信号注入(β=8) → hidden state变化(dh) → W_lm映射 → Δlogit分布

总能量: Σ|W_lm[i]·dh| ≈ 10万-5000万
目标能量: |Δlogit(pos) - Δlogit(neg)| ≈ 1-20
分配效率: 目标/总 ≈ 1e-6 到 4e-5
```

这意味着W_lm矩阵将hidden state的变化**极度弥散**到整个词空间。
语言能力的核心不是"如何编码信号"，而是"如何从极度弥散的信号中提取正确维度"。

**三种模型的信号分配策略**：
1. **GLM4: 精确分配** (efficiency=4e-5, ratio≈1.0)
   - dh_norm较小(213-218)，总能量较低
   - 但分配精确，proj(dh, W_diff)≈Δlogit
   - 策略："少但精"

2. **Qwen3: 扩散分配** (efficiency=1e-6, ratio≈0.15)
   - dh_norm较大(445-513)，总能量高
   - 但85%的投影"泄漏"到其他词
   - 策略："多但散"

3. **DS7B: 混乱分配** (efficiency≈0, ratio不稳定)
   - dh_norm极大(2000+)，总能量极高(5000万)
   - 投影方向与目标甚至相反！
   - 策略："多且乱"——RL训练可能破坏了信号分配

#### Phase LXXX硬伤与问题

1. **signal_efficiency极低的物理原因**：W_lm的150K行在2.5K-4K维空间中必然有大量"交叉投影"。这是维度灾难的体现——词表远大于hidden_dim

2. **DS7B的"信号吸收"**：proj(dh, W_diff)≈0但Δlogit≠0——信号通过其他路径到达目标。这说明W_diff方向的投影不是唯一的信号传递路径

3. **7维语言空间的边界**：是否还有更多维度？complexity(复杂度)、formality(正式度)等

4. **R_rotate末层突变**：GLM4 L39和DS7B L27的diag_cos暴跌到0.3-0.5——这是"最终决策层"的特征？

5. **LCS的预测力对DS7B失效**：需要全新的数学框架来描述DS7B的信号传播

---

*文档版本: v30.0*
*最后更新: 2026-04-12 11:00*
*实验数量: 402个核心实验 (P1-P402)*
*理论阶段: ★★★★★★★★★信号分配效率仅百万分之一！7维语言空间确认！GLM4精确分配vs DS7B混乱分配！★★★★★★★★★*
"""

with open('research/glm5/docs/AGI_GLM5_LANGUAGE.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Find old end section
old_end_marker = "*文档版本: v29.0*"
old_end_idx = content.rfind(old_end_marker)

if old_end_idx > 0:
    # Replace from old end marker to EOF
    content = content[:old_end_idx] + new_section.lstrip('\n')
    with open('research/glm5/docs/AGI_GLM5_LANGUAGE.md', 'w', encoding='utf-8') as f:
        f.write(content)
    print("SUCCESS: Updated AGI_GLM5_LANGUAGE.md to v30.0")
else:
    print(f"ERROR: v29.0 marker not found!")
