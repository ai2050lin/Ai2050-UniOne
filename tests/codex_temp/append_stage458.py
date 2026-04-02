# -*- coding: utf-8 -*-
"""追加Stage458研究记录到AGI_GLM5_MEMO.md"""

import datetime
from pathlib import Path

MEMO_PATH = Path(r"D:\develop\TransformerLens-main\research\glm5\docs\AGI_GLM5_MEMO.md")

now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d %H:%M")

entry = f"""

---

## Stage458: 中英文双语概念编码一致性验证
**时间**: {timestamp}
**模型**: DeepSeek-7B (Qwen tokenizer, 中英双语)
**脚本**: tests/codex/stage458_bilingual_concepts.py
**概念对**: 50个（5类别×10个）

### 核心结果

#### 1. ★★★ 中英文偏置向量部分一致 ★★★
- 平均余弦相似度: **0.325**（远高于随机预期~0.05）
- 最相似: motorcycle/摩托车(0.53), cat/猫(0.51), bicycle/自行车(0.50)
- 最不相似: pillow/枕头(0.02), desk/书桌(0.03), plane/飞机(0.08)
- **解读**: 中英文概念编码部分共享同一语义空间，但不是完全对齐

#### 2. ★★★ 联合SVD因子高度相关（核心发现）★★★
- 15个因子的中英文平均相关: **0.746**
- Factor 9: EN-ZH相关=0.876
- Factor 12: EN-ZH相关=0.861
- Factor 2: EN-ZH相关=0.854
- Factor 4: EN-ZH相关=0.843
- **解读**: 中英文在因子层面高度对齐！虽然原始偏置相似度只有0.33，
  但提取因子后相关性达0.75，说明**语义因子结构是语言无关的**

#### 3. 跨语言概念算术
- EN→ZH平均: 0.237, ZH→EN平均: 0.246
- 最佳: ZH dog→cat预测EN = 0.51
- **解读**: 跨语言算术部分可行，但不如同语言算术有效

### 理论意义

**偏置相似度(0.33) << 因子相关(0.75)** → 这意味着：
1. 原始偏置空间中英文有差异（语言表层特征）
2. 但底层的语义因子结构高度一致（概念深层结构）
3. **SVD分解起到了"语言去噪"的作用**——提取出语言无关的语义因子

这支持了核心假设：深度神经网络提取了**超越语言的数学结构**

### 局限
1. 偏置相似度仅0.33，原始空间中英不对齐
2. 偏移前2个因子（f0, f1）相关低（0.23, 0.60），可能编码语言特异信息
3. 跨语言算术精度不高（0.24），需要更大概念集验证
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
