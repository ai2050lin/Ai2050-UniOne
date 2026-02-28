# Gemini 路线: DNN结构分析

## 研究目标

从训练好的深度神经网络(DNN)中提取特征编码结构，理解特征如何涌现和编码。

## 核心问题

1. DNN内部形成了什么样的特征编码？
2. 这些特征是如何在训练中涌现的？
3. 大脑可能用什么机制实现类似编码？

## 当前进度: 50%

### 已完成
- [x] SAE特征提取框架
- [x] 四特性评估系统
- [x] 机制分析框架（动态追踪、因果干预、对比分析）
- [x] GPT-2初步分析（250样本）

### 核心发现

| 发现 | 数据 |
|-----|------|
| 稀疏编码存在 | 78% L0稀疏度 |
| 高正交性 | 97% 特征正交性 |
| 抽象层递增 | 分散度 12.6 vs 11.9 |
| 关键层识别 | Layer 2, 10, 11 |
| 复杂度驱动 | 范数 996→1990 |

## 目录结构

```
gemini/
├── code/                    # 分析代码
│   ├── mechanism_analysis/  # 机制分析框架
│   ├── feature_extractor.py
│   ├── four_properties_evaluator.py
│   └── sparse_coding_analyzer.py
├── data/                    # 数据文件
├── results/                 # 分析结果
│   └── feature_analysis/
└── docs/                    # 文档
```

## 快速开始

```bash
# 特征涌现追踪
python research/gemini/code/run_quick_emergence.py

# 完整机制分析
python research/gemini/code/run_full_mechanism_analysis.py

# 特征演化分析
python research/gemini/code/run_feature_evolution.py
```

## 下一步

- [ ] 运行大规模涌现追踪
- [ ] 验证因果干预效果
- [ ] 扩展到GPT-2 Medium/Large
- [ ] 增加样本到1000+
