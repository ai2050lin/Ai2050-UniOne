# -*- coding: utf-8 -*-
"""
Phase XXXVI 拼图 #4: GPT-2 MLP 稀疏激活解剖
=============================================
核心问题: DNN 的 MLP 层到底在做什么？
- 激活是否稀疏？(稀疏度量化)
- 不同 Token 是否激活不同的 MLP 神经元子集？(专家化度量)
- MLP 的"知识"是密集矩阵乘法还是稀疏字典查找？

所有测量均为纯观测，不训练任何新模型。
"""

import torch
import numpy as np
import json
import os
import sys
from tqdm import tqdm

def main():
    print("=" * 70)
    print("Phase XXXVI 拼图 #4: GPT-2 MLP 稀疏激活解剖")
    print("=" * 70)
    
    # 检查 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[设备] {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 加载模型
    print("\n[1/4] 加载 GPT-2 Small (透明解剖模式)...")
    try:
        from transformer_lens import HookedTransformer
    except ImportError:
        print("错误: 请安装 transformer_lens: pip install transformer_lens")
        sys.exit(1)
    
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    n_layers = model.cfg.n_layers  # 12
    d_mlp = model.cfg.d_mlp       # 3072
    d_model = model.cfg.d_model    # 768
    print(f"  层数: {n_layers}, MLP 中间维度: {d_mlp}, 模型维度: {d_model}")
    print(f"  MLP 参数比例: {2 * d_mlp * d_model * n_layers / sum(p.numel() for p in model.parameters()) * 100:.1f}%")
    
    # 准备测试语料 - 覆盖不同领域
    test_corpus = [
        # 科学
        "The theory of general relativity describes gravity as a geometric property of spacetime.",
        "Quantum entanglement allows particles to be correlated regardless of distance.",
        # 日常
        "She walked to the kitchen and made herself a cup of hot coffee with milk.",
        "The children played in the park until the sun went down behind the hills.",
        # 编程
        "The function returns a list of integers sorted in ascending order by default.",
        "We need to refactor the database layer to support horizontal sharding.",
        # 法律
        "The defendant was found guilty of fraud under Section 18 of the Criminal Code.",
        "The court ruled that the contract was void due to lack of consideration.",
        # 数学
        "The eigenvalues of a symmetric matrix are always real and the eigenvectors are orthogonal.",
        "A topological space is compact if every open cover has a finite subcover.",
    ]
    
    # ==========================================
    # 测量 1: 各层 MLP 的稀疏度
    # ==========================================
    print("\n[2/4] 测量各层 MLP 激活的稀疏度...")
    
    # 收集所有层的 MLP 中间激活
    hook_names = [f"blocks.{l}.mlp.hook_post" for l in range(n_layers)]
    
    layer_sparsity = {l: [] for l in range(n_layers)}
    layer_activations = {l: [] for l in range(n_layers)}
    
    for text in tqdm(test_corpus, desc="  处理语料"):
        _, cache = model.run_with_cache(
            text,
            names_filter=lambda name: name in hook_names
        )
        
        for l in range(n_layers):
            acts = cache[hook_names[l]].squeeze(0)  # [seq_len, d_mlp]
            # GPT-2 使用 GELU, 不是 ReLU, 所以"稀疏"的定义是 |act| < threshold
            # 用绝对值的中位数作为阈值
            threshold = acts.abs().median().item() * 0.1  # 非常小的阈值
            sparsity = (acts.abs() < threshold).float().mean().item()
            layer_sparsity[l].append(sparsity)
            layer_activations[l].append(acts.detach().cpu())
    
    print("\n  === MLP 稀疏度报告 ===")
    print(f"  {'层':>4} | {'平均稀疏度':>10} | {'解释':>30}")
    print(f"  {'-'*4} | {'-'*10} | {'-'*30}")
    
    sparsity_report = {}
    for l in range(n_layers):
        avg_sparsity = np.mean(layer_sparsity[l])
        sparsity_report[f"layer_{l}"] = float(avg_sparsity)
        label = "极稀疏" if avg_sparsity > 0.5 else ("中等稀疏" if avg_sparsity > 0.3 else "密集")
        print(f"  L{l:>2}  | {avg_sparsity:>9.4f} | {label}")
    
    # ==========================================
    # 测量 2: 语义专家化程度
    # ==========================================
    print("\n[3/4] 测量 MLP 神经元的语义专家化程度...")
    
    # 对每个领域，统计哪些 MLP 神经元被强烈激活
    # 语料分组: 科学(0-1), 日常(2-3), 编程(4-5), 法律(6-7), 数学(8-9)
    domain_names = ["科学", "日常", "编程", "法律", "数学"]
    domain_indices = [(0,1), (2,3), (4,5), (6,7), (8,9)]
    
    # 选取中间层 (Layer 6) 分析专家化
    target_layer = 6
    print(f"  分析 Layer {target_layer} 的专家化模式...")
    
    domain_top_neurons = {}
    domain_activation_profiles = {}
    
    for domain_name, (idx_start, idx_end) in zip(domain_names, domain_indices):
        # 收集该领域所有 Token 的 MLP 激活
        domain_acts = torch.cat([
            layer_activations[target_layer][i] 
            for i in range(idx_start, idx_end + 1)
        ], dim=0)  # [total_tokens, d_mlp]
        
        # 计算每个神经元在该领域的平均激活强度
        mean_act = domain_acts.mean(dim=0)  # [d_mlp]
        
        # 找出 Top-50 最活跃的神经元
        top_vals, top_indices = mean_act.abs().topk(50)
        domain_top_neurons[domain_name] = set(top_indices.tolist())
        domain_activation_profiles[domain_name] = mean_act.numpy()
    
    # 计算领域间的神经元重叠度
    print("\n  === 领域间 Top-50 神经元重叠矩阵 ===")
    print(f"  {'':>6}", end="")
    for name in domain_names:
        print(f" | {name:>4}", end="")
    print()
    
    overlap_matrix = {}
    for i, name_i in enumerate(domain_names):
        print(f"  {name_i:>6}", end="")
        overlap_matrix[name_i] = {}
        for j, name_j in enumerate(domain_names):
            overlap = len(domain_top_neurons[name_i] & domain_top_neurons[name_j])
            overlap_matrix[name_i][name_j] = overlap
            print(f" | {overlap:>4}", end="")
        print()
    
    # 计算领域间的余弦相似度
    print("\n  === 领域激活 Profile 余弦相似度 ===")
    print(f"  {'':>6}", end="")
    for name in domain_names:
        print(f" | {name:>6}", end="")
    print()
    
    cosine_matrix = {}
    for i, name_i in enumerate(domain_names):
        print(f"  {name_i:>6}", end="")
        cosine_matrix[name_i] = {}
        for j, name_j in enumerate(domain_names):
            profile_i = domain_activation_profiles[name_i]
            profile_j = domain_activation_profiles[name_j]
            cos_sim = np.dot(profile_i, profile_j) / (np.linalg.norm(profile_i) * np.linalg.norm(profile_j) + 1e-8)
            cosine_matrix[name_i][name_j] = float(cos_sim)
            print(f" | {cos_sim:>6.3f}", end="")
        print()
    
    # ==========================================
    # 测量 3: 激活值分布（是否符合稀疏编码假说）
    # ==========================================
    print("\n[4/4] 分析激活值分布形态...")
    
    # 取 Layer 6 的所有激活拼接
    all_acts_l6 = torch.cat(layer_activations[target_layer], dim=0).numpy()  # [total_tokens, d_mlp]
    
    # 统计基本分布特征
    flat_acts = all_acts_l6.flatten()
    stats = {
        "mean": float(np.mean(flat_acts)),
        "std": float(np.std(flat_acts)),
        "median": float(np.median(flat_acts)),
        "kurtosis": float(np.mean(((flat_acts - np.mean(flat_acts)) / np.std(flat_acts))**4)) - 3,  # 超额峰度
        "percent_near_zero": float((np.abs(flat_acts) < 0.1).mean()),
        "percent_near_zero_strict": float((np.abs(flat_acts) < 0.01).mean()),
        "max_abs": float(np.max(np.abs(flat_acts))),
        "l1_l2_ratio": float(np.mean(np.abs(flat_acts))) / (float(np.sqrt(np.mean(flat_acts**2))) + 1e-8),
    }
    
    print(f"\n  === Layer {target_layer} 激活值分布统计 ===")
    print(f"  均值:          {stats['mean']:.6f}")
    print(f"  标准差:        {stats['std']:.6f}")
    print(f"  中位数:        {stats['median']:.6f}")
    print(f"  超额峰度:      {stats['kurtosis']:.2f} (>0 = 尖峰重尾 = 稀疏证据)")
    print(f"  |act| < 0.1:   {stats['percent_near_zero']*100:.1f}%")
    print(f"  |act| < 0.01:  {stats['percent_near_zero_strict']*100:.1f}%")
    print(f"  最大|act|:     {stats['max_abs']:.4f}")
    print(f"  L1/L2 比率:    {stats['l1_l2_ratio']:.4f} (越低越稀疏, 均匀分布≈0.8)")
    
    # ==========================================
    # 生成报告
    # ==========================================
    os.makedirs("tempdata", exist_ok=True)
    
    report = {
        "experiment": "Phase XXXVI 拼图 #4: GPT-2 MLP 稀疏激活解剖",
        "model": "gpt2-small",
        "device": device,
        "n_layers": n_layers,
        "d_mlp": d_mlp,
        "corpus_size": len(test_corpus),
        "sparsity_by_layer": sparsity_report,
        "expert_overlap_matrix": overlap_matrix,
        "activation_distribution": stats,
        "cosine_similarity_matrix": cosine_matrix,
    }
    
    report_path = "tempdata/exp_mlp_sparsity_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n[完成] 报告保存至 {report_path}")
    
    # 总结
    print("\n" + "=" * 70)
    print("结论摘要")
    print("=" * 70)
    print(f"1. MLP 稀疏度: 浅层平均 {np.mean([np.mean(layer_sparsity[l]) for l in range(4)]):.1%}, "
          f"深层平均 {np.mean([np.mean(layer_sparsity[l]) for l in range(8,12)]):.1%}")
    print(f"2. 超额峰度 = {stats['kurtosis']:.2f} ({'强烈稀疏' if stats['kurtosis'] > 3 else '中等稀疏' if stats['kurtosis'] > 0 else '不稀疏'})")
    
    # 计算平均对角线 vs 非对角线重叠
    diag_overlaps = [overlap_matrix[n][n] for n in domain_names]
    off_diag = []
    for i, ni in enumerate(domain_names):
        for j, nj in enumerate(domain_names):
            if i != j:
                off_diag.append(overlap_matrix[ni][nj])
    avg_diag = np.mean(diag_overlaps)
    avg_off = np.mean(off_diag)
    
    print(f"3. 专家化: 同领域重叠 {avg_diag:.0f}/50, 跨领域重叠 {avg_off:.1f}/50 "
          f"({'强专家化' if avg_diag / (avg_off + 1) > 2 else '弱专家化'})")
    print(f"4. 对大脑的启示: MLP {'确实是' if stats['kurtosis'] > 0 else '不是'}稀疏字典查找")

if __name__ == "__main__":
    main()
