"""更新AGI_GLM5_MEMO with CCXIX results"""
import os
from datetime import datetime

memo_path = os.path.join(os.path.dirname(__file__), '..', '..', 'research', 'glm5', 'docs', 'AGI_GLM5_MEMO.md')

now = datetime.now().strftime("%Y年%m月%d日%H时%M分")

content = f"""

---

## CCXIX(369): 范数控制下的暗物质转导验证

### 五个硬伤的验证结果

**问题一: 范数混淆 ★★★★★ — 部分成立**

范数补偿实验: alpha_wu_comp = alpha * (||delta_dark|| / ||delta_wu||) 使注入范数相等

  层/模型    WU_orig  WU_comp  Dark    判决
  Qwen3 L12: 0.010    0.025    0.747   转导成立
  Qwen3 L18: -0.025   0.154    0.806   转导成立(但WU_comp开始有影响)
  Qwen3 L24: 0.012    0.371    0.847   范数混淆有贡献
  GLM4  L12: 0.033    0.171    0.873   转导成立
  GLM4  L18: 0.083    0.258    0.846   范数混淆有贡献
  GLM4  L24: 0.074    0.350    0.838   范数混淆有贡献
  DS7B  L24: 0.682    0.868    0.865   范数补偿后效果相当

  ★ 修正结论: 
  - 中浅层(L12): 暗物质steering优势主要来自转导机制(范数补偿后WU仍<0.2)
  - 深层(L24): 范数混淆贡献约30-40%(WU_comp从0.01涨到0.37)
  - 之前的"暗物质承载75-87%效果"需要修正:
    → 中浅层: ~70%来自转导, ~30%来自范数
    → 深层: ~50%来自转导, ~50%来自范数

**问题三: 暗物质维度7 — 完全推翻!**

  n_concepts  eff_rank(L12)  eff_rank(L18)
  3           1.9            1.9
  5           3.8            3.7
  8           6.6            6.4     ← 之前认为7维!
  10          8.4            8.2
  12          9.9            9.5
  15          12.3           11.7
  20          15.9           15.0    ← 实际约16维!

  ★★★ "7维概念核"完全错误! eff_rank ≈ 0.8 × n_concepts, 几乎线性增长
  ★★★ 暗物质维度远大于7, 只是在8个概念时受限于样本量
  ★★★ 20概念时暗物质eff_rank=15-17, 还在增长, 真实维度未知

  但组内维度稳定:
  - concrete(n=7): eff_rank=5.5-5.8
  - abstract(n=7): eff_rank=5.7-5.9  
  - relational(n=6): eff_rank=4.8-4.9
  → 同类概念的暗物质共享约5-6维子空间

**问题四: 线性假设 — 基本成立**

  alpha    lambda_wu  tau_dark
  0.1      0.762      0.023
  0.25     0.768      0.025
  0.5      0.767      0.028

  ★ lambda_wu跨alpha几乎不变(0.762-0.768) → W_U衰减是线性的
  ★ tau_dark有微弱增长(0.023→0.028, +22%) → 轻微非线性
  ★ 线性近似在alpha≤0.5下基本成立

**问题五: "概念核=7维" — 完全推翻(见问题三)**

  跨组暗物质PC1对齐极低(0.04-0.21) → 不同类型概念使用不同暗物质方向
  暗物质可加性: add_ratio≈0.9, cos≈0.6-0.78 → 概念间暗物质有中等相关性

**问题二: alpha=1.0崩溃 — 待更细粒度分析**

  alpha=0.4-0.6: WU_comp eff=0.17-0.29 (线性区, 暗物质优势明显)
  alpha=0.7-0.9: WU_comp eff=0.36-0.61 (非线性区, W_U补偿变有效)
  alpha≥1.0: 全面不稳定, eff值剧烈波动

### 修正后的统一图景

1. 暗物质steering效果 = 转导贡献(~60-70%) + 范数贡献(~30-40%)
   - 中浅层: 转导为主
   - 深层: 范数和转导各半
   
2. 暗物质维度不是7, 而是随概念数增长(eff_rank ≈ 0.8 × n_concepts)
   - "7维概念核"是样本量限制的伪结论
   - 真实暗物质维度可能>20, 需要更多概念验证
   
3. 同类概念(concrete/abstract/relational)共享约5-6维暗物质子空间
   - 跨组暗物质方向几乎正交(PC1 cos=0.04-0.21)
   → 暗物质空间有语义分组结构!

4. 线性传播方程在alpha≤0.5下基本成立:
   delta_wu_(l+1) = λ_wu × delta_wu_l + τ_dark × delta_dark_l
   λ_wu ≈ 0.76, τ_dark ≈ 0.025 (Qwen3)

5. alpha>0.7进入非线性区, 可能存在"激活范数稳态调节"

[CCXIX 范数混淆部分成立(中浅层转导为主/深层范数贡献30-40%)/暗物质维度7被推翻/20概念时eff_rank=15-17随概念数线性增长/同类概念暗物质共享5-6维子空间/跨组PC1对齐极低0.04-0.21/线性传播方程在alpha<=0.5下基本成立/lambda_wu~0.76/tau_dark~0.025/"7维概念核"是样本量限制伪结论 时间标记: {now}]
"""

with open(memo_path, 'a', encoding='utf-8') as f:
    f.write(content)

print(f"MEMO updated at {now}")
