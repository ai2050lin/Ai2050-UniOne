# -*- coding: utf-8 -*-
"""追加Stage460记录到AGI_GLM5_MEMO"""
from pathlib import Path

MEMO_PATH = Path(r"D:\develop\TransformerLens-main\research\glm5\docs\AGI_GLM5_MEMO.md")

entry = """

---

## Stage460: 80维高阶SVD因子分析 — 双模型交叉验证
**时间**: 2026-04-01 08:50-09:40
**模型**: Qwen3-4B (36层) + DeepSeek-7B (28层)
**概念数**: 405 (15类别)
**脚本**: tests/codex/stage460_high_dim_factors.py

### 核心结果

#### 方差解释里程碑
| 维度 | Qwen3-4B | DeepSeek-7B |
|------|---------|------------|
| Top-10 | 10.4% | 9.0% |
| Top-20 | 17.3% | 14.8% |
| Top-50 | 32.7% | 28.7% |
| Top-80 | **44.1%** | **39.5%** |
| 30%方差 | 44维 | 54维 |
| 40%方差 | 69维 | N/A(80维才39.5%) |

#### 关键发现
1. **80维SVD解释44.1%(Qwen3)/39.5%(DeepSeek)** — 远超之前的22.4%
2. **压缩比惊人**: Qwen3 730:1, DeepSeek 1421:1 (高维→80维保留44%信息)
3. **50%方差未达到** — 即使80维也不够，需要100+维
4. **高阶因子(21-80)全部有跨类别语义** — 但单个因子解释力很弱(0.44-0.64%)
5. **因子-类别eta²全部<0.3** — 偏置矩阵的列数太多(6层×hidden_dim)，稀释了类别信号
6. **重建余弦**: 80维时Qwen3=0.576, DeepSeek=0.557 — 接近中等精度

#### 双模型对比
- Qwen3在所有维度上方差解释比DeepSeek高约4-5%
- DeepSeek-7B的隐藏维度更大(4096 vs 3584),概念表示更分散
- 两模型高阶因子语义不同(个体差异),但总体趋势一致

#### 理论意义
- **偏置空间确实是线性的**(确认Stage456): 80维线性SVD就能解释44%方差
- **语义信息高度分散**: 没有少数"超级因子",而是大量弱因子共同编码
- **高维→低维压缩有效**: 1421:1压缩比下仍保留39.5%方差
- **概念编码的本征维度确实在50-100范围**(与Stage459的PR=80.9一致)

#### 硬伤与瓶颈
1. 80维SVD仍未达50%方差,剩余56%未被解释
2. 高阶因子可解释性差: 单个因子<0.65%,难以赋予明确语义
3. 偏置矩阵维度过高(6层×4096=24576维),导致SVD效率降低
4. 类内分散度DeepSeek普遍高于Qwen3,说明7B模型概念表示更复杂

#### 下一步
- Stage461: 使用单层偏置(而非6层拼接)重做SVD,降低矩阵维度
- Stage462: 100+维SVD测试能否突破50%
- 重新审视偏置矩阵构建方式: 也许应该用PCA先降维到1000维再做SVD
"""

if MEMO_PATH.exists():
    content = MEMO_PATH.read_text(encoding="utf-8")
    content += entry
    MEMO_PATH.write_text(content, encoding="utf-8")
    print("OK: appended Stage460 to MEMO")
else:
    print("ERROR: MEMO not found")
