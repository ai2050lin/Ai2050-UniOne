# 共享资源

本目录包含三条研究路线共享的代码、数据和文档。

## 目录结构

```
shared/
├── code/                    # 公共代码
│   └── transformer_lens/    # Transformer分析核心库
├── data/                    # 公共数据
│   ├── MNIST/              # MNIST数据集
│   └── iso_corpus.jsonl    # 语料库
└── docs/                    # 公共文档
    ├── theory/              # 理论文档
    └── methodology/         # 方法论文档
```

## 使用说明

- **只读原则**: 共享资源应保持稳定，修改需经过协调
- **引用方式**: 各路线代码通过相对路径引用共享资源
- **版本控制**: 重要修改需要更新版本号

## 核心库

### transformer_lens

TransformerLens是一个用于分析Transformer模型的Python库，提供：
- HookedTransformer: 可插入hook的Transformer模型
- ActivationCache: 激活缓存和分析
- 预训练模型加载

```python
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-small")
logits, cache = model.run_with_cache("Hello world")
```
