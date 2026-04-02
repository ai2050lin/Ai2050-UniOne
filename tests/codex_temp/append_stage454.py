# -*- coding: utf-8 -*-
"""追加Stage452-454研究记录到AGI_GLM5_MEMO.md"""

import datetime
from pathlib import Path

MEMO_PATH = Path(r"D:\develop\TransformerLens-main\research\glm5\docs\AGI_GLM5_MEMO.md")

entry = f"""
---

## Stage452-454: 概念基底-偏置编码机制验证（2026-04-01 01:20）

### Stage452: 基底-偏置验证（Qwen3-4B + DeepSeek-7B）
- **时间**: 2026-03-31 22:50 - 2026-04-01 00:30
- **核心假设**: 概念编码 = 基底(类别) + 偏置(个体)
- **结果**:
  - 基底解释方差比(R²): Qwen3=0.38, DeepSeek-7B=0.39
  - 偏置稀疏度: ~66%
  - 偏置正交性: ~-0.03（近似正交）
  - 概念算术命中率(top-5): 2/8
- **结论**: 基底-偏置机制存在，但简单平均基底精度不够

### Stage453: 深度概念编码分析
- **时间**: 2026-04-01 00:45 - 01:15
- **改进**: 分层基底、层权重优化、同类别算术、属性编码分析
- **核心结果**:

| 指标 | Qwen3-4B | DeepSeek-7B |
|------|----------|-------------|
| 同类概念算术(加权) | 0.48 | **0.77** |
| 跨类概念算术(加权) | 0.29 | **0.49** |
| 最高层R² | 0.81 (L35) | 0.66 (L27) |

- **关键发现**:
  1. 分层编码：中间层编码概念类别，最后层整合
  2. DeepSeek-7B同类算术达到0.77，支持基底-偏置假设
  3. 属性编码集中在最后3-5层
  4. 概念树聚类纯度(purity_10)≈0.50
  5. 跨类相似对：cat~car(0.18), bird~boat(0.15)

### Stage454: 理论框架
- **编码方程**: C(word) = Σ_layer [ w_layer × (B_category + a_individual) ]
- **三层模型**: 语法层 → 语义概念层 → 整合输出层
- **下一步**: SVD语义因子分解、大规模验证、跨语言一致性

### 生成文件
- `tests/codex/stage452_concept_basis_bias.py` - 基底-偏置验证
- `tests/codex/stage453_deep_concept_encoding.py` - 深度分析
- `tests/codex_temp/stage452_concept_basis_bias_20260331/` - Stage452结果
- `tests/codex_temp/stage453_deep_concept_encoding_20260401/` - Stage453结果
- `research/glm5/docs/AGI_GLM5_MEMO_STAGE454.md` - 理论报告
"""

now = datetime.datetime.now()
time_str = now.strftime("%Y-%m-%d %H:%M")

if MEMO_PATH.exists():
    content = MEMO_PATH.read_text(encoding="utf-8")
    content += f"\n{entry}"
    MEMO_PATH.write_text(content, encoding="utf-8")
    print(f"[{time_str}] AGI_GLM5_MEMO.md updated (appended Stage452-454)")
else:
    MEMO_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEMO_PATH.write_text(f"# AGI GLM5 Research Memo\n\n{entry}", encoding="utf-8")
    print(f"[{time_str}] AGI_GLM5_MEMO.md created")
