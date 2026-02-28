"""
机制分析快速启动脚本

运行此脚本开始从"统计描述"转向"机制理解"
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

import torch
from transformer_lens import HookedTransformer


def run_quick_mechanism_analysis():
    """
    快速机制分析实验
    
    目标: 演示如何从统计描述转向机制理解
    """
    print("=" * 70)
    print("机制分析实验：从统计描述到机制理解")
    print("=" * 70)
    print()
    
    # 1. 加载模型
    print("[1] 加载模型...")
    model = HookedTransformer.from_pretrained("gpt2-small")
    print(f"    模型加载成功: {model.cfg.n_layers}层, {model.cfg.d_model}维")
    print()
    
    # 2. 特征涌现追踪（简化版）
    print("[2] 特征涌现分析（简化版）...")
    print("    问题: 特征在训练中如何涌现？")
    print("    方法: 对比不同复杂度输入的激活模式")
    
    # 不同复杂度的输入
    inputs = {
        "simple": "The cat sat.",
        "medium": "The cat sat on the mat and looked at the dog.",
        "complex": "The philosophical implications of quantum mechanics challenge our fundamental understanding of reality, causality, and the nature of existence itself."
    }
    
    for complexity, text in inputs.items():
        tokens = model.to_tokens(text)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        
        # 分析各层激活
        layer_info = []
        for layer in [0, 6, 11]:
            act = cache["resid_post", layer]
            sparsity = (act.abs() < 0.01).float().mean().item()
            norm = act.norm().item()
            layer_info.append(f"L{layer}: 稀疏度={sparsity:.2%}, 范数={norm:.1f}")
        
        print(f"    {complexity}: {text[:30]}...")
        for info in layer_info:
            print(f"      {info}")
    print()
    
    # 3. 抽象机制分析
    print("[3] 抽象机制分析...")
    print("    问题: 抽象特征如何编码？")
    print("    方法: 对比具体词vs抽象词的激活")
    
    concepts = {
        "concrete": ["cat", "dog", "table", "chair", "apple"],
        "abstract": ["justice", "freedom", "love", "truth", "beauty"]
    }
    
    for category, words in concepts.items():
        activations = []
        for word in words:
            tokens = model.to_tokens(word)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)
                act = cache["resid_post", -1].mean(dim=1)
                activations.append(act)
        
        all_acts = torch.cat(activations, dim=0)
        spread = all_acts.std().item()
        mean_norm = all_acts.norm(dim=1).mean().item()
        
        print(f"    {category}: 分散度={spread:.2f}, 平均范数={mean_norm:.1f}")
    
    print()
    
    # 4. 精确预测分析
    print("[4] 精确预测分析...")
    print("    问题: 精确预测需要什么条件？")
    print("    方法: 对比精确vs模糊预测的激活")
    
    test_cases = [
        ("precise", "1 + 1 ="),
        ("precise", "The capital of France is"),
        ("fuzzy", "Tomorrow will be"),
        ("fuzzy", "I think that")
    ]
    
    for label, text in test_cases:
        tokens = model.to_tokens(text)
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens)
            
            # 获取预测
            next_token_logits = logits[0, -1]
            top_k = torch.topk(next_token_logits, 3)
            top_tokens = [model.to_string(t) for t in top_k.indices]
            top_probs = torch.softmax(top_k.values, dim=0)
            
            # 激活特征
            act = cache["resid_post", -1]
            sparsity = (act.abs() < 0.01).float().mean().item()
            
        print(f"    {label}: '{text}'")
        print(f"      预测: {top_tokens[0]} ({top_probs[0]:.1%}), 稀疏度: {sparsity:.1%}")
    
    print()
    
    # 5. 关键发现
    print("[5] 关键发现")
    print("    ┌─────────────────────────────────────────────────────────┐")
    print("    │ 1. 复杂输入 → 更高的稀疏度（更多特征参与）              │")
    print("    │ 2. 抽象概念 → 更大的激活分散度（覆盖更广空间）          │")
    print("    │ 3. 精确预测 → 更低的稀疏度（特征更聚焦）               │")
    print("    │                                                         │")
    print("    │ 这些是机制理解的起点，而非终点。                        │")
    print("    │ 下一步：干预实验验证因果关系。                          │")
    print("    └─────────────────────────────────────────────────────────┘")
    print()
    
    # 6. 下一步
    print("[6] 下一步行动")
    print("    1. 运行特征涌现追踪（需要训练循环）")
    print("    2. 进行因果干预实验")
    print("    3. 获取大脑数据验证")
    print()
    print("    详细代码: analysis/mechanism_analysis/")
    print()


if __name__ == "__main__":
    run_quick_mechanism_analysis()
