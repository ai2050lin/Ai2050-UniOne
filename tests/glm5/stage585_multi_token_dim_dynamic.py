"""
Stage 585: 多token序列维度动态——重新定义"维度坍缩"

之前的"维度坍缩"基于单token有效秩，但单向量秩永远=1。
本脚本用完整句式所有token的hidden state矩阵计算有效秩，
真正观察维度在不同层的动态变化。

核心方法：
- 对每个句子，收集所有token在每层的hidden states [seq_len, hidden_dim]
- 用SVD计算有效秩 = (sum(S))^2 / sum(S^2)
- 逐层追踪有效秩的变化曲线
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# ========== 配置 ==========
MODEL_PATH = r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"
TEST_SENTENCES = [
    # 短输入（之前发现坍缩的）
    ("bank", "bank"),
    ("apple", "apple"),
    ("red apple", "red apple"),
    ("The bank", "The bank"),
    ("The river bank", "The river bank"),
    ("The financial bank", "The financial bank"),
    
    # 3词（阈值附近）
    ("I went to the bank", "I went to the bank"),
    ("She ate the red apple", "She ate the red apple"),
    
    # 长输入（12词+）
    ("The cat sat on the river bank and watched the boats", "The cat sat on the river bank and watched the boats"),
    ("The financial bank approved the loan for the business", "The financial bank approved the loan for the business"),
    ("She picked up the red apple from the tree branch", "She picked up the red apple from the tree branch"),
    ("The green apple tasted sour but the red apple was sweet", "The green apple tasted sour but the red apple was sweet"),
    
    # 歧义对比
    ("bank-river", "The children played by the river bank"),
    ("bank-financial", "The bank approved the loan application"),
    ("apple-fruit", "She ate a delicious red apple for lunch"),
    ("apple-company", "Apple released a new product today"),
    
    # 消歧信号注入
    ("bank-weak", "I went to the bank"),
    ("bank-strong", "The financial bank approved the loan"),
]

LAYER_INDICES = list(range(0, 36, 3)) + [35]  # 0,3,6,...,33,35 (Qwen3-4B=36层)


def effective_rank(matrix):
    """计算有效秩：E = (sum S)^2 / sum(S^2)"""
    if matrix.shape[0] <= 1:
        return 1.0
    # 去均值
    mean = matrix.mean(dim=0, keepdim=True)
    centered = matrix - mean
    # SVD
    try:
        _, S, _ = torch.linalg.svd(centered, full_matrices=False)
    except RuntimeError:
        # 回退到numpy
        C = centered.numpy()
        _, S_np, _ = np.linalg.svd(C, full_matrices=False)
        S = torch.from_numpy(S_np)
    # 有效秩
    S = S[S > 1e-8]  # 过滤零奇异值
    if len(S) == 0:
        return 1.0
    rank_eff = (S.sum() ** 2) / (S ** 2).sum()
    return min(rank_eff.item(), matrix.shape[1])


def main():
    print("=" * 60)
    print("Stage 585: 多token序列维度动态")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    
    print("加载模型...")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'codex'))
    from qwen3_language_shared import load_qwen3_model
    model, tokenizer = load_qwen3_model(prefer_cuda=True)
    layers = model.model.layers
    n_layers = len(layers)
    print(f"层数: {n_layers}")
    
    results = {}
    
    for name, sentence in TEST_SENTENCES:
        print(f"\n--- {name}: \"{sentence}\" ---")
        enc = tokenizer(sentence, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        
        layer_ranks = {}
        for li in LAYER_INDICES:
            hs = out.hidden_states[li][0].float().cpu()  # [seq_len, hidden_dim]
            eff_rank = effective_rank(hs)
            layer_ranks[li] = round(eff_rank, 2)
        
        results[name] = {
            "sentence": sentence,
            "seq_len": enc["input_ids"].shape[1],
            "layer_ranks": layer_ranks,
        }
        
        print(f"  序列长度: {enc['input_ids'].shape[1]}")
        for li, rank in layer_ranks.items():
            bar = "#" * int(rank / 5)
            print(f"  L{li:2d}: 有效秩={rank:6.2f} {bar}")
    
    # ========== 分析 ==========
    print("\n" + "=" * 60)
    print("分析")
    print("=" * 60)
    
    # 1. 序列长度 vs 有效秩的关系
    print("\n实验1: 序列长度 vs 有效秩(L35)")
    len_rank = []
    for name, data in results.items():
        rank_l35 = data["layer_ranks"][35]
        len_rank.append((data["seq_len"], name, rank_l35))
    len_rank.sort()
    for sl, name, rank in len_rank:
        print(f"  len={sl:2d}, {name:20s}: L35有效秩={rank:.2f}")
    
    # 2. 歧义消歧 vs 维度变化
    print("\n实验2: 歧义消歧——维度动态对比")
    for pair in [("bank-river", "bank-financial"), ("apple-fruit", "apple-company")]:
        r1, r2 = results[pair[0]], results[pair[1]]
        print(f"\n  {pair[0]} vs {pair[1]}:")
        for li in LAYER_INDICES:
            rank_diff = r1["layer_ranks"][li] - r2["layer_ranks"][li]
            print(f"    L{li:2d}: {r1['layer_ranks'][li]:.1f} vs {r2['layer_ranks'][li]:.1f}, diff={rank_diff:+.1f}")
    
    # 3. 短输入 vs 长输入的维度差异
    print("\n实验3: 短输入 vs 长输入（维度动态曲线）")
    short = results["I went to the bank"]
    long_ = results["The cat sat on the river bank and watched the boats"]
    print(f"  短(9词) vs 长(12词):")
    for li in LAYER_INDICES:
        print(f"    L{li:2d}: {short['layer_ranks'][li]:.1f} vs {long_['layer_ranks'][li]:.1f}")
    
    # 4. 消歧强度 vs 维度变化
    print("\n实验4: 消歧强度(弱vs强) vs 维度动态")
    weak = results["bank-weak"]
    strong = results["bank-strong"]
    print(f"  弱消歧(\"I went to the bank\") vs 强消歧(\"The financial bank approved the loan\"):")
    for li in LAYER_INDICES:
        print(f"    L{li:2d}: {weak['layer_ranks'][li]:.1f} vs {strong['layer_ranks'][li]:.1f}")
    
    # 保存
    out_path = "tests/glm5_temp/stage585_multi_token_dim_dynamic.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存到: {out_path}")
    
    # 关键结论
    print("\n" + "=" * 60)
    print("关键结论")
    print("=" * 60)
    
    # 找维度变化最大的层
    dim_changes = {}
    for li in LAYER_INDICES:
        if li == 0:
            continue
        changes = []
        for name, data in results.items():
            if 0 in data["layer_ranks"] and li in data["layer_ranks"]:
                change = data["layer_ranks"][li] - data["layer_ranks"][0]
                changes.append(change)
        if changes:
            dim_changes[li] = np.mean(changes)
    
    if dim_changes:
        max_layer = max(dim_changes, key=dim_changes.get)
        min_layer = min(dim_changes, key=dim_changes.get)
        print(f"  维度增长最大层: L{max_layer} (avg change = {dim_changes[max_layer]:+.1f})")
        print(f"  维度增长最小层: L{min_layer} (avg change = {dim_changes[min_layer]:+.1f})")
    
    # 长短输入差异
    long_ranks = [data["layer_ranks"][35] for data in results.values() if data["seq_len"] >= 10]
    short_ranks = [data["layer_ranks"][35] for data in results.values() if data["seq_len"] <= 5]
    if long_ranks and short_ranks:
        print(f"  长输入(>=10词) L35有效秩: mean={np.mean(long_ranks):.1f}")
        print(f"  短输入(<=5词) L35有效秩: mean={np.mean(short_ranks):.1f}")


if __name__ == "__main__":
    main()
