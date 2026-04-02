# -*- coding: utf-8 -*-
"""追加Stage459研究记录到AGI_GLM5_MEMO.md"""

import datetime
from pathlib import Path

MEMO_PATH = Path(r"D:\develop\TransformerLens-main\research\glm5\docs\AGI_GLM5_MEMO.md")

now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d %H:%M")

entry = f"""

---

## Stage459: 超大规模概念验证 + 本征维度估算
**时间**: {timestamp}
**模型**: DeepSeek-7B (28层), 447概念, 15类别
**脚本**: tests/codex/stage459_large_scale_dim.py

### 核心结果

#### 1. ★★★ Scaling Law：SVD方差解释 vs 概念数 ★★★
| 概念数 | Top-5 | Top-10 | Top-20 | Top-50 |
|--------|-------|--------|--------|--------|
| 50 | 20.9% | 33.4% | 53.8% | - |
| 132 (Stage455) | - | - | 21.5% | - |
| 220 (Stage456) | 13.1% | 18.8% | 27.4% | - |
| **447 (Stage459)** | **11.6%** | **16.0%** | **22.4%** | **35.2%** |

**关键发现**: 增加概念数后SVD方差解释比**下降**而非上升！
- 50词→447词，Top-5从20.9%降到11.6%（-44%）
- 这说明SVD低维因子在概念子集上过拟合，大规模上真实结构更复杂

#### 2. ★★★ 本征维度估算 ★★★
| 方法 | 本征维度 | 含义 |
|------|---------|------|
| **Participation Ratio** | **80.9** | 有效维度数≈81 |
| **MLE (k=5)** | **37.0** | 局部本征维度≈37 |
| **MLE (k=10)** | **35.3** | 局部本征维度≈35 |
| **PCA 50%方差** | **100维** | 需要100个PCA分量 |
| **PCA 70%方差** | **200维** | 需要200个PCA分量 |
| TwoNN | -0.8 | 失败（cosine距离不适用） |

**核心结论**: 偏置空间的本征维度约为**35-81维**，远高于之前假设的20维。
这意味着：前20个SVD因子只能捕捉一小部分语义结构，需要至少35-80个因子才能近似。

#### 3. 因子语义（15类别）
前几个因子仍然可解释：
- Factor 0: 自然/人造物分离（olive, desert + vs red, tree）
- Factor 8: 脚部穿着（boots, shoes, jeans）
- Factor 14: 球类运动（soccer, football, rugby）
- Factor 10: 浅色/紫色水果（strawberry, lavender, dragonfruit）

但大部分因子语义变得模糊，因为15个类别混合后因子更分散。

### 理论修正

1. **SVD方差不是概念数的函数**：更多概念→更多方差需要解释→每个因子的解释比例下降
2. **本征维度≈35-81**：远高于任何实验中使用的因子数，说明我们只看到了冰山一角
3. **概念子集过拟合**：小规模（50词）的高方差解释比是假象，大规模才反映真实结构
4. **线性低维假设需要修正**：偏置空间确实有低维结构（PR=81 vs 94720维），但维度是35-81，不是10-20

### 下一步突破方向

既然本征维度≈35-81，那么：
1. 用50-80个SVD因子重新分析属性关联（之前只用了前20个）
2. 验证高阶因子（30-80）是否有语义含义
3. 测试偏置空间是否可用80维有效近似（压缩比≈1200:1）
"""

if MEMO_PATH.exists():
    with open(MEMO_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    content += entry
    with open(MEMO_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Appended to {MEMO_PATH}")
else:
    print(f"WARNING: {MEMO_PATH} not found")
