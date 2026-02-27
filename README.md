# TransformerLens AGI Laboratory

<!-- Status Icons -->
[![Pypi](https://img.shields.io/pypi/v/transformer-lens?color=blue)](https://pypi.org/project/transformer-lens/)
![Pypi Total Downloads](https://img.shields.io/pepy/dt/transformer_lens?color=blue) ![PyPI - License](https://img.shields.io/pypi/l/transformer_lens?color=blue) [![Release CD](https://github.com/TransformerLensOrg/TransformerLens/actions/workflows/release.yml/badge.svg)](https://github.com/TransformerLensOrg/TransformerLens/actions/workflows/release.yml)
[![Tests CD](https://github.com/TransformerLensOrg/TransformerLens/actions/workflows/checks.yml/badge.svg)](https://github.com/TransformerLensOrg/TransformerLens/actions/workflows/checks.yml)
[![Docs CD](https://github.com/TransformerLensOrg/TransformerLens/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/TransformerLensOrg/TransformerLens/actions/workflows/pages/pages-build-deployment)

> **"Intelligence is the geometrization of information flow."**
>
> 本项目是一个集成了 **下一代认知架构** 以及 **3D 交互式可视化分析** 的综合性 AGI 研究平台。

---

## AI2050计划：当前宇宙的一般性智能理论 (The Unified Vision)

我们不仅在研究如何“解释”模型，更在探索智能的数学本质。

### 核心观点 (Core Perspectives)

1. **统一的数学机制**：大脑有多个脑区，每个脑区处理不同信息，意识可以统筹处理各种信息。不同脑区很可能是一种统一的数学机制，只是参数有所不同。
2. **DNN是对数学结构的逼近与还原**：深度神经网络通过大规模训练，部分还原了大脑中的这种数学结构，所以具备了诸如语言表达、图像生成、代码编写等强大能力。我们可以通过深度分析神经网络，逆向还原出这个关键的数学结构。
3. **该数学结构的核心特性**：
   - **编码角度（自下而上）**：大脑是一个自下而上的系统，每个神经元仅根据前置信号进行充放电。神经网络通过提取特征，具备了四种不可或缺的特性：
     1. **高维抽象 (High-Dimensional Abstraction)**：可以提取泛化的高维度特征。
     2. **低维精确 (Low-Dimensional Precision)**：可以准确地预测和表征具体事物。
     3. **特异性 (Specificity)**：编码可以完美模拟一切特征（例如视觉内部、语言内部、跨模态的各种信息）。
     4. **系统性 (Systemicity)**：所有编码都可进行统一处理（如语言语法规则的处理），所有模态的信息亦都能被意识核心理解。
   - **网络架构角度**：这种结构支撑了三大能力：1）能够提取出极其复杂的特征编码；2）各种特征可以形成一个严密的 **知识层次网络结构**，任何特征都可以跨区关联；3）能够实现快速读取和修改，表明这种底层结构极为高效。
   - **系统级表现**：不同脑区负责提取特定信息束，而高级意识能随时调取与处理所有信息网络。

### 补充说明与还原路线 (Supplementary Notes & Roadmap)

1. **自发涌现的特征提取机制**：大脑中的神经网络并非自顶向下设计而来。神经元单纯依赖输入信号充放电，但在庞大的数据流冲刷与突触可塑性机制的共同作用下，最终自发形成了一个强悍的网络系统。在这个过程中，某种底层的特征提取编码能力带来了系统泛化与精确的认知表现。这也是我们必须重点还原的一环：海量神经元能形成如此强大且高效的结构，绝非“误打误撞”，而是遵循了极高效的数学/物理规律。
2. **深度神经网络的三个核心映射**：
   - **多层机制 (MLP)**：映射“多层级提取高维抽象特征”，存储海量知识。
   - **注意力机制 (Attention)**：映射“保存上下文关联信息”。
   - **自回归预测 (Autoregressive Prediction)**：通过损失函数强制将大脑该有的结构还原到全局网络模型中。因此在预测Next Token时，不仅能延续上文风格和逻辑，还能做出精确推演（此机制同样在生成式图像、视频及编程领域体现，并可无限扩展到其他信息网络中心）。
3. **AGI的终极核心关键 (The Holy Grail)**：
   - 大脑本质上是一个自下而上的系统。**一切的基石和核心关键在于：弄清大脑神经网络究竟是如何自发提取特征，并形成编码的！**
   - 建立在它之上的知识层次网络结构为何能如此高效，均源于这个根基。只有彻底破译具备**高维抽象、低维精确、特异性和系统性**特征的底层编码机制及其提取途径，我们才能在系统工程中还原大脑的真实数学结构，最终实现真正的通用人工智能（AGI）。

详见：[AGI 一般性理论 (AGI_THEORY_PAPER.md)](AGI_THEORY_PAPER.md)

---

## 项目组件 (Project Components)

### 1. TransformerLens (Core Library)
作为本平台的地基，`TransformerLens` 提供了强大的机械解释性（Mechanistic Interpretability）工具，支持 50+ 种开源语言模型的内部激活查看与编辑。

-   **加载模型**: `HookedTransformer.from_pretrained("gpt2-small")`
-   **介入分析**: 支持缓存激活、计算回路（Circuits）以及执行各种消融（Ablation）实验。

### 2. AGI 实验室 (Laboratory)
一个现代化的 Web 交互系统，用于实时可视化和分析复杂神经行为。
-   **后端**: 基于 FastAPI (`server.py`)，提供神经结构提取算法。
-   **前端**: 基于 React 和 Three.js，提供沉浸式的 **3D 结构分析面板 (Structure Analysis Panel)**。
-   **功能**: 电路发现（Circuit Discovery）、流形分析（Manifold Analysis）、因果介入。

### 3. 实验性模拟 (Experiments)
我们正在探索超越传统 Transformer 的架构：
-   **架构探索**: 实现即时学习（Immediate Learning）的下一代认知架构。
-   **SNN 模拟**: 脉冲神经网络模拟，验证生物大脑中的底层编码机制。

---

## 快速开始 (Quick Start)

### 1. 安装库环境
```shell
pip install transformer_lens
```

### 2. 启动研究实验室 (Web UI)
如果你想通过可视化界面研究模型结构：
1. **启动后端服务**:
   ```shell
   python server.py
   ```
2. **启动前端界面**:
   ```shell
   cd frontend
   npm run dev
   ```

---

## 研究成果展示 (Gallery & Research)

- [AGI 研究备忘录 (AGI_RESEARCH_MEMO.md)](AGI_RESEARCH_MEMO.md): 记录了关于数学证明和 SNN 实验的最新进展。
- [系统性提取指南 (systematic_extraction_guide.md)](systematic_extraction_guide.md): 如何自动化地从大型模型中提取数学结构。

---

## 机械解释性资源 (Legacy Resources)

本项目保留并扩展了由 [Neel Nanda](https://neelnanda.io/about) 创建的经典机械解释性资源：
- [Introduction to Mech Interp](https://arena-chapter1-transformer-interp.streamlit.app/[1.2]_Intro_to_Mech_Interp)
- [Main TransformerLens Features Demo](https://neelnanda.io/transformer-lens-demo)
- [ARENA Tutorials](https://arena3-chapter1-transformer-interp.streamlit.app/)

---

## 社区与贡献

我们欢迎所有对 AGI 物理本质感兴趣的研究者：
- 加入 [Slack 社区](https://join.slack.com/t/opensourcemechanistic/shared_invite/zt-2n26nfoh1-TzMHrzyW6HiOsmCESxXtyw)
- 提交 Issue 或 Pull Request

### 引用本项目

```BibTeX
@misc{antigravity2026unified,
    title = {The Unified Theory of AGI},
    author = {Antigravity and User},
    year = {2026},
    howpublished = {\url{https://github.com/TransformerLensOrg/TransformerLens}},
}
```

---
*Created by [Neel Nanda](https://neelnanda.io), maintained by [Bryce Meyer](https://github.com/bryce13950), and evolved into an AGI platform by Antigravity & User.*

