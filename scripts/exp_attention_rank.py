# -*- coding: utf-8 -*-
"""
Phase XXXVI 拼图 #3: GPT-2 Attention 矩阵有效秩测量
====================================================
核心问题: Attention 到底在做什么？
- 有效秩是否极低？(说明只用了极少几种关联模式)
- 不同 Head 是否有不同的"模式角色"？
- 浅层 vs 深层的 Attention 模式是否有质变？

如果有效秩低，大脑的"注意力"就不需要 O(N^2) 全连接，
而可能仅靠几种固定的同步振荡模式切换实现。
"""

import torch
import numpy as np
import json
import os
import sys
from tqdm import tqdm

def effective_rank(matrix):
    """
    计算矩阵的有效秩 (Effective Rank)
    r_eff = exp(H(p)), 其中 p_i = sigma_i / sum(sigma_i), H = Shannon 熵
    """
    # SVD 分解
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    
    # 归一化奇异值为概率分布
    S_positive = S[S > 1e-10]
    if len(S_positive) == 0:
        return 1.0
    
    p = S_positive / S_positive.sum()
    
    # Shannon 熵
    entropy = -np.sum(p * np.log(p + 1e-10))
    
    # 有效秩
    return float(np.exp(entropy))

def attention_pattern_type(attn_matrix):
    """
    尝试分类 Attention 的模式类型
    返回: 模式名称和强度
    """
    n = attn_matrix.shape[0]
    if n < 3:
        return "too_short", 0.0
    
    # 1. 对角线模式 (关注自己)
    diag_strength = np.mean(np.diag(attn_matrix))
    
    # 2. 前一个 Token 模式
    prev_strength = np.mean(np.diag(attn_matrix, -1)) if n > 1 else 0.0
    
    # 3. 第一个 Token 模式 (BOS attention)
    first_col_strength = np.mean(attn_matrix[:, 0])
    
    # 4. 均匀模式
    uniform_value = 1.0 / n
    uniform_distance = np.mean((attn_matrix - uniform_value) ** 2)
    uniform_strength = 1.0 / (1.0 + uniform_distance * n * n)
    
    # 选择最强的模式
    patterns = {
        "自注意(diagonal)": diag_strength,
        "前词注意(prev)": prev_strength,
        "首词注意(BOS)": first_col_strength,
        "均匀注意(uniform)": uniform_strength,
    }
    
    best_pattern = max(patterns, key=patterns.get)
    return best_pattern, patterns[best_pattern]

def main():
    print("=" * 70)
    print("Phase XXXVI 拼图 #3: GPT-2 Attention 矩阵有效秩测量")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[设备] {device}")
    
    print("\n[1/3] 加载 GPT-2 Small...")
    try:
        from transformer_lens import HookedTransformer
    except ImportError:
        print("错误: 请安装 transformer_lens: pip install transformer_lens")
        sys.exit(1)
    
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    n_layers = model.cfg.n_layers   # 12
    n_heads = model.cfg.n_heads     # 12
    d_head = model.cfg.d_head       # 64
    
    print(f"  层数: {n_layers}, 注意力头数: {n_heads}, 头维度: {d_head}")
    
    # 测试语料
    test_corpus = [
        "The cat that the dog chased ran up the tree quickly.",
        "Although the experiment failed, the scientists discovered something unexpected.",
        "The professor who taught mathematics retired after thirty years of service.",
        "When it rains heavily, the river floods the surrounding farmland and villages.",
        "The book that I borrowed from the library was written by a famous physicist.",
        "If you multiply a matrix by its inverse, you get the identity matrix.",
    ]
    
    # 收集 Attention 模式
    hook_names = [f"blocks.{l}.attn.hook_pattern" for l in range(n_layers)]
    
    print(f"\n[2/3] 提取 {n_layers * n_heads} 个 Attention Head 的模式...")
    
    # 结果存储: [layer][head] -> list of effective ranks
    head_ranks = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
    head_patterns = [[{} for _ in range(n_heads)] for _ in range(n_layers)]
    
    for text in tqdm(test_corpus, desc="  处理语料"):
        _, cache = model.run_with_cache(text, names_filter=lambda name: name in hook_names)
        
        for l in range(n_layers):
            # attn_pattern: [batch, n_heads, seq, seq]
            pattern = cache[hook_names[l]].squeeze(0)  # [n_heads, seq, seq]
            
            for h in range(n_heads):
                attn_mat = pattern[h].cpu().numpy()  # [seq, seq]
                
                # 计算有效秩
                eff_rank = effective_rank(attn_mat)
                head_ranks[l][h].append(eff_rank)
                
                # 分类模式
                pat_name, pat_strength = attention_pattern_type(attn_mat)
                if pat_name not in head_patterns[l][h]:
                    head_patterns[l][h][pat_name] = []
                head_patterns[l][h][pat_name].append(pat_strength)
    
    # ==========================================
    # 汇总报告
    # ==========================================
    print("\n[3/3] 生成分析报告...")
    
    # 计算每个 Head 的平均有效秩
    avg_ranks = [[float(np.mean(head_ranks[l][h])) for h in range(n_heads)] for l in range(n_layers)]
    
    print("\n  === Attention 有效秩矩阵 (行=层, 列=头) ===")
    print(f"  {'':>4}", end="")
    for h in range(n_heads):
        print(f" | H{h:>2}", end="")
    print(f" | {'平均':>4}")
    print(f"  {'-'*4}", end="")
    for h in range(n_heads):
        print(f" | {'-'*4}", end="")
    print(f" | {'-'*4}")
    
    layer_avg_ranks = []
    for l in range(n_layers):
        print(f"  L{l:>2} ", end="")
        for h in range(n_heads):
            rank = avg_ranks[l][h]
            print(f" | {rank:>4.1f}", end="")
        layer_mean = float(np.mean(avg_ranks[l]))
        layer_avg_ranks.append(layer_mean)
        print(f" | {layer_mean:>4.1f}")
    
    # 统计分析
    all_ranks = [avg_ranks[l][h] for l in range(n_layers) for h in range(n_heads)]
    
    print(f"\n  === 有效秩统计 ===")
    print(f"  全局平均有效秩:  {np.mean(all_ranks):.2f} / {max(len(test_corpus[0].split()), 10)} (理论最大=序列长度)")
    print(f"  全局中位数:      {np.median(all_ranks):.2f}")
    print(f"  最小有效秩:      {np.min(all_ranks):.2f}")
    print(f"  最大有效秩:      {np.max(all_ranks):.2f}")
    print(f"  有效秩 < 3 的 Head 数: {sum(1 for r in all_ranks if r < 3)} / {n_layers * n_heads}")
    print(f"  有效秩 < 5 的 Head 数: {sum(1 for r in all_ranks if r < 5)} / {n_layers * n_heads}")
    
    # 浅层 vs 深层
    shallow_avg = np.mean(layer_avg_ranks[:4])
    deep_avg = np.mean(layer_avg_ranks[8:])
    print(f"\n  浅层 (L0-3) 平均有效秩: {shallow_avg:.2f}")
    print(f"  深层 (L8-11) 平均有效秩: {deep_avg:.2f}")
    
    # 主要模式分类
    print(f"\n  === 各 Head 主导模式分类 ===")
    pattern_counts = {}
    for l in range(n_layers):
        for h in range(n_heads):
            # 找该 Head 最常出现的模式
            best_pat = None
            best_score = -1
            for pat_name, scores in head_patterns[l][h].items():
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_pat = pat_name
            if best_pat:
                pattern_counts[best_pat] = pattern_counts.get(best_pat, 0) + 1
    
    for pat_name, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f"  {pat_name}: {count} 个 Head ({count/(n_layers*n_heads)*100:.0f}%)")
    
    # 保存报告
    os.makedirs("tempdata", exist_ok=True)
    report = {
        "experiment": "Phase XXXVI 拼图 #3: Attention 有效秩测量",
        "model": "gpt2-small",
        "effective_rank_matrix": avg_ranks,
        "layer_average_ranks": layer_avg_ranks,
        "global_stats": {
            "mean_rank": float(np.mean(all_ranks)),
            "median_rank": float(np.median(all_ranks)),
            "min_rank": float(np.min(all_ranks)),
            "max_rank": float(np.max(all_ranks)),
            "heads_with_rank_lt_3": int(sum(1 for r in all_ranks if r < 3)),
            "heads_with_rank_lt_5": int(sum(1 for r in all_ranks if r < 5)),
            "total_heads": n_layers * n_heads,
        },
        "pattern_distribution": pattern_counts,
    }
    
    report_path = "tempdata/exp_attention_rank_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n[完成] 报告保存至 {report_path}")
    
    print("\n" + "=" * 70)
    print("结论摘要")
    print("=" * 70)
    low_rank_pct = sum(1 for r in all_ranks if r < 5) / len(all_ranks) * 100
    print(f"1. {low_rank_pct:.0f}% 的 Head 有效秩 < 5")
    if low_rank_pct > 50:
        print('   [OK] Attention 确实只用了极少几种关联模式,而非复杂全连接')
        print('   -> 大脑的"注意力"可能仅靠少数同步振荡模式切换实现')
    else:
        print('   [!!] Attention 秩较高,可能确实需要复杂的全连接关联')
    
    print(f"2. 浅层 vs 深层有效秩: {shallow_avg:.1f} vs {deep_avg:.1f}")
    if deep_avg < shallow_avg:
        print('   -> 深层 Attention 更加"聚焦",只关注极少数关键位置')
    else:
        print('   -> 深层 Attention 更加"弥散",整合更多位置的信息')

if __name__ == "__main__":
    main()
