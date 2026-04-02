# -*- coding: utf-8 -*-
"""追加Stage462记录到AGI_GLM5_MEMO"""
from pathlib import Path

MEMO_PATH = Path(r"D:\develop\TransformerLens-main\research\glm5\docs\AGI_GLM5_MEMO.md")

entry = """

---

## Stage462: 概念算术测试 — Qwen3-4B + DeepSeek-7B 单模型独立验证
**时间**: 2026-04-01 09:45-10:30
**模型**: Qwen3-4B (36层) + DeepSeek-7B (28层)
**概念数**: 74 (4类别: 水果/动物/交通工具/职业)
**脚本**: tests/codex/stage462_concept_arithmetic.py

### 核心结果

#### 1. 逐层SVD方差解释（复现Stage461）
| 模型 | 黄金层 | Top-10 | Top-20 | Top-100 |
|------|--------|--------|--------|---------|
| Qwen3-4B | L2 | 69.7% | 79.6% | 100.0% |
| DeepSeek-7B | L1 | 58.7% | 73.8% | 100.0% |

#### 2. Embedding vs MLP激活 对比
| 空间 | Qwen3 Top-10 | DeepSeek Top-10 |
|------|-------------|----------------|
| Embedding | 28.1% | 27.0% |
| MLP激活(黄金层) | 69.7% | 58.7% |

→ MLP激活比Embedding的SVD方差高2.5倍！说明MLP层确实加工了embedding信息

#### 3. 概念算术精度（关键结果）

| 方法 | Qwen3-4B | DeepSeek-7B | 说明 |
|------|---------|------------|------|
| 原始偏置余弦(概念间相似度) | 0.035 | 0.019 | apple-banana本身不相似 |
| 最近邻算术(偏置空间) | -0.002 | -0.019 | 无效 |
| 因子空间最近邻 | -0.018 | -0.030 | 无效 |
| 因子空间Top-3近邻 | -0.054 | -0.060 | 无效 |
| **3Cos类比(属性参考)** | **0.291** | **0.269** | 最佳方法 |

#### 4. apple→banana 详细结果
| 模型 | 最佳层 | 最佳余弦 | 方法 |
|------|--------|---------|------|
| Qwen3 | L2 | 0.410(近邻)/-0.531(3Cos) | 3Cos对水果无效 |
| DeepSeek | L1 | 0.389(原始)/0.410(近邻) | 近邻算术有效 |

### 关键发现

1. **概念间原始相似度极低**（apple-banana余弦=0.04~0.39），偏置空间中概念相互几乎正交
2. **3Cos类比是唯一有效的方法**（平均0.27-0.29），但仅对部分对有效
3. **最近邻/因子NN方法全面失败**（负值），说明不能用"找最近邻居"来预测
4. **L1/L2层的概念区分度最好**，但算术精度不一定最高
5. **偏置空间本质上不适合做概念算术**：偏置=激活-类别均值，去除了类别信息

### 核心问题
- 偏置空间的概念算术精度很低（<0.3），远不够实用
- 原因：偏置空间去除类别信息后，概念间差异被均匀化，无法通过简单向量操作预测
- 可能需要在**原始激活空间**而非偏置空间做算术
"""

if MEMO_PATH.exists():
    content = MEMO_PATH.read_text(encoding="utf-8")
    content += entry
    MEMO_PATH.write_text(content, encoding="utf-8")
    print("OK: appended Stage462 to MEMO")
else:
    print("ERROR: MEMO not found")
