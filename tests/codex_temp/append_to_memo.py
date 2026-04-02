"""追加测试记录到AGI_GPT5_MEMO.md"""

memo_content = """
---

时间戳: 2026-03-31 09:10
状态: DeepSeek-7B测试完成，跨模型对比分析完成
文件:
- tests/codex/single_model_neuron_extraction_stage431.py
- tests/codex_temp/neuron_extraction_deepseek_7b_stage431.json
- tests/codex_temp/cross_model_comparison_stage431.py
- CROSS_MODEL_NEURON_COMPARISON_STAGE431.md

### Stage431: 跨模型神经元提取对比分析（完成）

#### 测试模型
1. Qwen3-4B: 36层，2560隐藏层，92,160个神经元 - 完成
2. DeepSeek-7B: 28层，3584隐藏层，100,352个神经元 - 完成

#### 核心发现

**词性质心层相关性**:
- Pearson相关系数: 0.8908
- 结论: 不同架构的模型采用了相似的词性编码策略

**层级功能分化**:
- 早期层（0-12层）: 基础语法处理（代词、介词）
- 中期层（13-24层）: 语义整合（名词、动词、形容词）
- 晚期层（25-36层）: 上下文整合（副词）

**内容词 vs 功能词**:
- 内容词（名词、动词、形容词）: 质心层0.38-0.48，分布均匀
- 功能词（代词、介词）: 质心层0.16-0.24，强烈早期层优势

#### 技术突破
1. 单模型加载框架: 避免GPU溢出
2. 具体神经元ID提取: 层索引+神经元索引
3. 跨模型对比方法: 归一化层索引+质心层相关性分析

#### 下一步工作
- 短期: 扩展测试词库、神经元干预实验
- 中期: 建立精确映射、分析上下文依赖
- 长期: 构建AGI语言能力的数学模型
"""

# 追加到文件
with open("D:/develop/TransformerLens-main/research/gpt5/docs/AGI_GPT5_MEMO.md", "a", encoding="utf-8") as f:
    f.write(memo_content)

print("已追加测试记录到 AGI_GPT5_MEMO.md")
