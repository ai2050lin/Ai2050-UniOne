"""更新AGI_GLM5_MEMO - CCXVIII结果"""
import os
from datetime import datetime

MEMO_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "research", "glm5", "docs", "AGI_GLM5_MEMO.md")

text = """

---

## CCXVIII(368): 暗物质级联转导机制

### Exp1: 暗物质转导追踪 (3模型验证)

在L12注入delta_full/delta_wu/delta_dark, 追踪后续6层的W_U投影比变化:

**W_U注入 (init_wu=1.0): W_U投影比逐步衰减**
  k(层数)  Qwen3  GLM4   DS7B
  k=1:     0.69   0.80   0.76
  k=2:     0.50   0.65   0.60
  k=3:     0.39   0.54   0.50
  k=6:     0.25   0.34   0.33

  ★ W_U分量被Attn/MLP旋转出W_U空间! 每层损失约10-15%
  ★ 6层后只保留25-34%的W_U投影

**暗物质注入 (init_wu=0.0): W_U投影比逐步增长**
  k(层数)  Qwen3  GLM4   DS7B
  k=1:     0.028  0.069  0.021
  k=2:     0.047  0.049  0.021
  k=3:     0.057  0.084  0.028
  k=6:     0.073  0.109  0.045

  ★ 暗物质被Attn/MLP转导进W_U空间! 每层增长约1-2%
  ★ 暗物质方向保持性好: cos(dark_injected, delta_at_k) = 0.83→0.61
  ★ W_U方向保持性差: cos(wu_injected, delta_at_k) = 0.31→0.25

**Full注入 (init_wu~0.09): W_U投影比保持稳定**
  k=1: 0.096/0.161/0.065  (Qwen3/GLM4/DS7B)
  k=6: 0.098/0.139/0.072

  ★ Full delta的W_U投影比几乎不变(~0.06-0.16)
  ★ 这是因为W_U衰减和暗物质转导互相平衡

### 转导的数学解释

设delta = delta_wu + delta_dark, 注入后经过一层:

  delta' = J @ delta = J @ delta_wu + J @ delta_dark

  delta_wu分量:
  - 在W_U空间中, 但Attn/MLP将其旋转出W_U空间
  - 每层保留率: lambda_wu ~ 0.85 (从1.0→0.85→0.72...)
  - 6层后: 0.85^6 = 0.38, 实际0.25-0.34 (更差因为cos衰减)

  delta_dark分量:
  - 不在W_U空间中, 但Attn/MLP将其部分转导进W_U空间
  - 每层转导率: tau_dark ~ 0.02-0.07 (从0→0.03→0.05...)
  - 暗物质方向更"鲁棒": cos(dark_inj, delta') = 0.83-0.61

  稳态平衡:
  - W_U分量的衰减 ~ 暗物质的转导增长
  - 所以full delta的W_U投影比保持稳定(~0.09)

### Exp2: 暗物质PCA + alpha缩放 (Qwen3)

**暗物质的有效维度:**
  层    暗物质有效秩  n_90%  cum@1   cum@5
  L12:  6.7          6      0.205   0.817
  L18:  6.6          6      0.220   0.834
  L24:  6.7          6      0.204   0.812

  ★★★ 暗物质有效秩只有6.6-6.7! 比完整概念(28-31)低4倍!
  ★★★ 第1个PC解释20-22%暗物质方差 (vs 完整概念8%)
  ★★★ 5个PC解释81-83%暗物质方差 (vs 完整概念55%)
  → 暗物质比完整概念更集中、更低维、更有结构!

**alpha缩放: W_U vs Dark steering效果**
  alpha  full_mean  wu_mean  dark_mean  wu_eff   dark_eff
  0.5    1.464      -0.041   1.172      -0.038   0.804
  1.0    1.501       0.016   1.932      -1.018   -2.318
  2.0    1.053       0.339   1.984       0.165    1.701
  5.0   -1.809      -0.083   0.018       0.225    1.854

  ★ W_U-only在alpha=0.5时几乎无效(eff=-0.04)
  ★ alpha=1.0时暗物质overwhelms(full和dark都有负效果)
  ★ alpha=2.0暗物质仍优于W_U(1.70 vs 0.17)
  ★ alpha=5.0全面崩溃

### 统一图景: Transformer概念传播的完整数学

  概念在残差流中: delta_l = delta_wu_l + delta_dark_l

  传播规则:
  delta_wu_(l+1) = lambda_wu * delta_wu_l + tau_dark * delta_dark_l
  delta_dark_(l+1) = (1 - tau_dark) * delta_dark_l + (1 - lambda_wu) * delta_wu_l

  参数估计:
  - lambda_wu ~ 0.85 (W_U分量每层保留85%)
  - tau_dark ~ 0.02-0.07 (暗物质每层转导2-7%进W_U空间)
  - 暗物质有效维度 ~ 7 (6-7个PC解释90%方差)
  - 概念有效维度 ~ 28 (27-31个PC解释90%方差)

  为什么暗物质steering有效:
  1. 暗物质占86-92%的delta范数
  2. 暗物质每层转导2-7%进W_U空间
  3. 残差连接保证暗物质方向在6层后仍保持cos=0.61
  4. 累积转导: 6层后暗物质贡献了7-11%的W_U投影
  5. 虽然转导率低, 但暗物质范数大(~43-103 vs W_U ~14-36)

  为什么W_U steering无效:
  1. W_U分量只占8-14%的delta范数
  2. W_U分量每层损失15%的W_U投影(被旋转出)
  3. W_U方向在6层后cos降到0.25 (方向严重漂移)
  4. 范数小+方向漂移=几乎零效果

### 严格审视

1. 暗物质转导率tau=0.02-0.07看起来很小, 6层后W_U投影只有7-11%
   → 但steering效果中暗物质贡献80%! 这怎么解释?
   → 可能: 最终层的lm_head不需要高W_U投影比, 只需要绝对值足够大
   → delta_dark的范数(~43-103) * 0.07 = 3-7 的W_U分量, 足以改变logit

2. 暗物质有效维度6.6-7, 但只有8个概念x1=8个样本
   → 样本量等于概念数, 有效维度可能被低估
   → 需要更多概念(20+)来精确估计

3. W_U衰减率lambda_wu=0.85的假设需要更精细验证
   → 非线性效应: alpha=0.5时JVP~1.0, 但W_U衰减0.85
   → 可能是因为W_U分量与Attn/MLP的交互导致旋转

4. 转导机制的具体通路不清:
   → 暗物质不在当前层MLP/Attn空间中(1-13%)
   → 但暗物质被转导进W_U空间 — 通过什么路径?
   → 可能: 暗物质与MLP的gate_proj交互, 被silu非线性"折射"进W_U空间

### 核心洞察与第一性原理

暗物质有效维度约7, 这是目前为止发现的最强的"数学结构"信号:
  - d_model = 2560-4096 (全空间)
  - W_U行空间 = 200维 (5-8%的d_model)
  - 概念空间 = 28维 (1%的d_model)
  - 暗物质空间 = 7维 (0.3%的d_model!) ← 极度浓缩

这暗示: 概念在残差流中的真正表示只有7维的暗物质 + 少量W_U分量
7维暗物质可能是"概念核" — 最本质的语义编码

[CCXVIII 暗物质级联转导/W_U注入6层后W_U投影从100%衰减到25-34%/暗物质注入6层后从0%增长到7-11%/暗物质方向保持cos=0.83-0.61远优于W_U的0.31-0.25/暗物质有效秩6.6-7(极度浓缩)/W_U衰减率lambda_wu~0.85/暗物质转导率tau~0.02-0.07/概念传播=delta_wu衰减+delta_dark转导/暗物质是"概念核"约7维 时间标记: 2026年05月03日01时40分]
"""

with open(MEMO_PATH, "a", encoding="utf-8") as f:
    f.write(text)

print(f"MEMO updated at {datetime.now()}")
