# 项目结构说明

**最后更新**: 2026-02-28

---

## 核心研究框架

### 研究目标

实现人类水平的智能系统

### 核心假设

1. 大脑各脑区运行同一数学机制（参数不同）
2. DNN已部分还原这个结构
3. 可通过分析DNN逆向还原数学本质

### 核心问题

**神经网络如何从信号流中提取特征、形成编码？**

这是一切能力的基石。不要预设任何理论正确，先观察，后假说。

---

## 目录结构总览

```
TransformerLens-Project/
│
├── research/                      # 【核心研究】三条独立路线
│   ├── gemini/                    # Gemini路线: DNN结构分析 (50%)
│   ├── gpt5/                      # GPT5路线: 大脑机制还原 (10%)
│   ├── glm5/                      # GLM5路线: 特征涌现与编码 (5%)
│   └── PROGRESS.md                # 总进展追踪
│
├── shared/                        # 【共享资源】只读
│   ├── code/transformer_lens/     # Transformer分析核心库
│   ├── data/                      # 公共数据集
│   └── docs/                      # 公共文档
│
├── archive/                       # 【归档区】
│   ├── deprecated/                # 已弃用
│   ├── backup/                    # 备份
│   └── tempdata/                  # 临时数据
│
├── scripts/                       # 脚本工具
├── server/                        # 后端服务
├── frontend/                      # 前端界面
├── tests/                         # 测试代码
├── data/                          # 原始数据
├── tempdata/                      # 临时数据
│
├── INDEX.md                       # 项目索引
├── README.md                      # 项目说明
└── PROJECT_STRUCTURE.md           # 本文件
```

---

## 三条研究路线

### Gemini 路线: DNN结构分析

**进度**: 50%
**目标**: 理解DNN特征编码机制

```
research/gemini/
├── code/                    # 分析代码
│   ├── mechanism_analysis/  # 机制分析框架
│   ├── feature_extractor.py
│   ├── four_properties_evaluator.py
│   └── sparse_coding_analyzer.py
├── data/                    # 数据文件
├── results/                 # 分析结果
└── docs/                    # 文档
```

**核心发现**:
- 稀疏编码: 78% L0稀疏度
- 正交性: 97%
- 抽象概念占据更大空间

**关键问题**: 知道"是什么"，不知道"为什么"

---

### GPT5 路线: 大脑机制还原

**进度**: 10%
**目标**: 验证DNN机制是否存在于大脑

```
research/gpt5/
├── code/                    # 分析代码
│   ├── brain_dnn_comparison/
│   └── brain_mechanism_inference.py
├── data/                    # 数据文件
├── results/                 # 分析结果
└── docs/                    # 文档
```

**关键对比**:
| 大脑 | DNN | 差异 |
|-----|-----|------|
| 20W | GPU 300W+ | 能效 |
| ~2% 稀疏度 | ~78% | 40倍 |

**阻塞问题**: 需要神经科学数据

---

### GLM5 路线: 特征涌现与编码机制

**进度**: 5%
**目标**: 回答核心问题 - 特征如何提取、编码如何形成

```
research/glm5/
├── code/                    # 核心代码
│   ├── emergence_tracker/   # 特征涌现追踪
│   ├── coding_analyzer/     # 编码分析
│   └── experiment_runner/   # 实验框架
├── experiments/             # 实验代码
├── data/                    # 数据文件
├── results/                 # 实验结果
└── docs/                    # 文档
```

**研究原则**:
1. 不预设理论正确
2. 聚焦核心问题
3. 先观察，后假说
4. 让结构自然浮现

**研究计划**:
- Phase 1: 特征涌现追踪 (1-2月)
- Phase 2: 编码基本单位分析 (2-3月)
- Phase 3: 稀疏性与正交性机制 (2-3月)

---

## 共享资源

```
shared/
├── code/
│   └── transformer_lens/    # Transformer分析核心库
├── data/
│   ├── MNIST/              # MNIST数据集
│   └── iso_corpus.jsonl    # 语料库
└── docs/
    ├── theory/              # 理论文档
    │   ├── AGI_RESEARCH_MEMO.md
    │   ├── AGI_THEORY_PAPER.md
    │   └── ...
    └── PROJECT_ROADMAP.md
```

---

## 研究方法论

### 问题链

```
一切都是基石: 神经网络如何从信号流中提取特征、形成编码？
    ↓
在此之上形成: 特征如何形成层级结构？
    ↓
在此之上形成: 层级结构如何支持抽象与精确并存？
    ↓
在此之上形成: 不同模态如何统一编码？
    ↓
最终涌现: AGI能力
```

### 方法转变

```
旧方法: 理论假设 → 实验验证 → 修改理论
新方法: 观察 → 记录 → 模式识别 → 假说 → 再观察

核心: 不预设答案，让结构自然浮现
```

### 已获得的拼图块

| 拼图块 | 来源 | 置信度 |
|-------|------|-------|
| 稀疏编码 (78%) | Gemini | 高 |
| 高正交性 (97%) | Gemini | 高 |
| 抽象概念占更大空间 | Gemini | 高 |
| Layer 2,10,11关键 | Gemini | 中 |
| 复杂输入激活更强 | Gemini | 高 |

### 缺失的拼图块

```
关键缺失:
├── 特征如何在训练中涌现？
├── 为什么稀疏度是78%？
├── 编码的"基本单位"是什么？
├── 大脑的编码与DNN有何本质不同？
├── 局部可塑性如何产生全局稳态？
├── 特异性是如何实现的？
└── 系统性是如何实现的？
```

---

## 多客户端协作

| 终端 | 工作目录 | 路线 | 核心任务 |
|-----|---------|------|---------|
| 终端A | `research/gemini/` | DNN结构分析 | 机制理解 |
| 终端B | `research/gpt5/` | 大脑机制还原 | 数据验证 |
| 终端C | `research/glm5/` | 特征涌现与编码 | 核心问题 |

**协作规则**:
1. 各终端在各自目录内独立工作
2. 共享资源只读，修改需协调
3. 临时数据放入 `archive/tempdata/`
4. 定期同步 `PROGRESS.md`

---

## 快速命令

```bash
# Gemini: DNN分析
python research/gemini/code/run_quick_emergence.py
python research/gemini/code/run_full_mechanism_analysis.py

# GPT5: 大脑对比
python research/gpt5/code/brain_dnn_comparison/brain_dnn_validator.py

# GLM5: 特征涌现
python research/glm5/experiments/train_from_scratch.py
```

---

## 进度追踪

```
总体进度: 25%

Gemini: ██████████░░░░░░░░░░ 50%  框架完成，待深入
GPT5:   ██░░░░░░░░░░░░░░░░░░ 10%  数据缺失
GLM5:   █░░░░░░░░░░░░░░░░░░░  5%   方向调整
```

详见: `research/PROGRESS.md`

---

## 下一步行动

### 第一优先级

```
GLM5路线 - 特征涌现追踪实验:
├── 从随机初始化训练模型
├── 每100步记录激活状态
├── 追踪特征如何从无到有形成
└── 这是回答一切问题的基础
```

### 时间预估

| 阶段 | 时间 | 目标 |
|-----|------|------|
| Phase 1 | 1-2月 | 特征涌现时间线 |
| Phase 2 | 2-3月 | 编码基本单位 |
| Phase 3 | 2-3月 | 稀疏度机制 |
| Phase 4 | 3-4月 | 层级形成机制 |
| Phase 5 | 持续 | 大脑对比验证 |

---

## 注意事项

1. **不预设理论**: 目前思路可能都是错的
2. **聚焦核心**: 一切围绕"特征如何提取、编码如何形成"
3. **完成拼图**: 先收集足够多的观察，再尝试还原
4. **路径引用**: 代码中的import路径可能需要更新
5. **版本控制**: 重要变更需提交到git
