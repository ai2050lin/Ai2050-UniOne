"""追加Stage432-433记录到AGI_GPT5_MEMO.md"""

memo_content = """
---

时间戳: 2026-03-31 13:25
状态: Stage432扩展词库测试完成，Stage433消融实验框架建立（技术瓶颈待解决）
文件:
- tests/codex/neuron_extraction_extended_stage432.py
- tests/codex_temp/neuron_extraction_extended_qwen3_4b_stage432.json
- tests/codex_temp/neuron_extraction_extended_deepseek_7b_stage432.json
- tests/codex/neuron_ablation_experiment_stage433.py
- tests/codex_temp/neuron_ablation_experiment_qwen3_4b_stage433.json
- STAGE432_433_SUMMARY.md

### Stage432: 扩展词库测试（完成）

#### 测试配置
- 每个词性测试30个单词（从原来的5个扩展）
- 总测试单词：180个（6个词性×30个单词）

#### 核心发现

**质心层稳定性**:
- 内容词（名词、形容词）: 质心层稳定，差异<0.01
- 功能词（代词、介词）: 质心层显著变化，从早期层（0.16-0.17）变成中期层（0.45-0.48）
- 结论: 小样本量（5个单词）对功能词的估计不稳定

**跨模型相关性**:
- 5个单词时: 相关系数0.8908
- 30个单词时: 相关系数0.6900
- 结论: 大样本量揭示了真实的模型差异

**神经元特异性**:
- 所有词性的top-10神经元都有极高的特异性（接近1.0）
- 说明确实存在专门处理特定词性的神经元

### Stage433: 神经元消融实验（框架建立，技术瓶颈待解决）

#### 实验目的
验证神经元的因果作用

#### 技术瓶颈
- 问题: 无法精确定位和修改神经元权重
- 原因: Transformer模型的神经元嵌入在复杂的架构中
- 结果: 消融前后处理能力变化为0.00%，说明消融未生效

#### 下一步需要解决
1. 精确定位神经元对应的参数
2. 实现有效的消融方法
3. 设计更robust的评估指标

### 理论洞察

**样本量效应**: 证明了小样本量可能产生误导性结论
**编码多样性**: 不同模型的词性编码策略存在真实差异
**神经元特异性**: 确认了高特异性神经元的存在
"""

# 追加到文件
with open("D:/develop/TransformerLens-main/research/gpt5/docs/AGI_GPT5_MEMO.md", "a", encoding="utf-8") as f:
    f.write(memo_content)

print("已追加Stage432-433记录到 AGI_GPT5_MEMO.md")
