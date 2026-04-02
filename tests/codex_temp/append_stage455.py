# -*- coding: utf-8 -*-
"""追加Stage455研究记录到AGI_GLM5_MEMO.md"""

import datetime
from pathlib import Path

MEMO_PATH = Path(r"D:\develop\TransformerLens-main\research\glm5\docs\AGI_GLM5_MEMO.md")

entry = f"""
---

## Stage455: SVD/ICA语义因子分解（2026-04-01 01:35）

### 核心方法
- 构建偏置矩阵：132个概念 × 5层 × 隐状态维度 = 132×48640矩阵
- SVD分解发现语义因子，ANOVA关联属性

### 突破性发现：语义因子可解释！

**因子-属性关联（ANOVA eta²）**:
| 属性 | 最佳因子 | eta² | 解释 |
|------|---------|------|------|
| **profession.field** | factor_3 | **0.956** | 职业领域几乎完全由一个因子决定！ |
| **fruit.color** | factor_5 | **0.771** | 颜色因子独立存在 |
| **fruit.tropical** | factor_11 | **0.776** | 热带属性因子 |
| **animal.domestic** | factor_18 | **0.767** | 驯化属性因子 |
| **vehicle.speed** | factor_14 | **0.766** | 速度因子 |
| **fruit.shape** | factor_15 | **0.718** | 形状因子 |
| **fruit.texture** | factor_4 | **0.716** | 质地因子 |
| **animal.habitat** | factor_11 | **0.716** | 栖息地因子 |
| **vehicle.size** | factor_4 | **0.711** | 大小因子 |
| **animal.size** | factor_7 | **0.662** | 动物大小因子 |
| **material.hardness** | factor_7 | **0.671** | 硬度因子 |
| **profession.creative** | factor_3 | **0.733** | 创造性因子 |

### 因子空间中的概念算术
| 源→目标 | 原始空间余弦 | 因子空间余弦 | 最近邻 |
|--------|-----------|-----------|-------|
| dog→cat | 0.28 | **0.80** | cat(0.80) |
| lemon→banana | 0.04 | 0.28 | lime(0.92) |
| ship→boat | 0.34 | **0.63** | boat(0.63) |
| car→bus | 0.14 | **0.47** | truck(0.60) |

**因子空间算术比原始空间有效2-3倍！**

### 理论方程（最终版）
```
概念编码 = B_category + Σ_i (α_i × F_i)
  B_category: 类别基底（跨层加权）
  F_i: 第i个独立语义因子（颜色、大小、功能...）
  α_i: 概念在因子i上的投影系数
  
  概念算术: apple - banana ≈ α_color_diff × F_color + α_size_diff × F_size + ...
```

### 硬伤与瓶颈
1. SVD前20因子仅解释21.5%方差，低维近似不够精确
2. 因子空间算术apple→banana反而更差（-0.16），部分概念行为异常
3. 需要1000+词来发现更完整的因子体系

### 生成文件
- `tests/codex/stage455_svd_semantic_factors.py`
- `tests/codex_temp/stage455_svd_semantic_factors_20260401/REPORT.md`
"""

now = datetime.datetime.now()

if MEMO_PATH.exists():
    content = MEMO_PATH.read_text(encoding="utf-8")
    content += f"\n{entry}"
    MEMO_PATH.write_text(content, encoding="utf-8")
    print(f"[{now.strftime('%Y-%m-%d %H:%M')}] MEMO updated")
else:
    MEMO_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEMO_PATH.write_text(f"# AGI GLM5 Research Memo\n\n{entry}", encoding="utf-8")
    print(f"[{now.strftime('%Y-%m-%d %H:%M')}] MEMO created")
