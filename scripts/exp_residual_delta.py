# -*- coding: utf-8 -*-
"""
Phase XXXVI 拼图 #2: GPT-2 逐层残差增量 SVD 分析
=================================================
核心问题: 深层网络的每一层到底贡献了什么？
- 是"大幅度重构信号"还是"精细微调"？
- 各层的信息增量在哪些方向上？
- 浅层 vs 深层的贡献模式是否有质变？

这个实验将揭示大脑多层皮层的真实功能分工。
"""

import torch
import numpy as np
import json
import os
import sys
from tqdm import tqdm

def main():
    print("=" * 70)
    print("Phase XXXVI 拼图 #2: GPT-2 逐层残差增量 SVD")
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
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    
    # 测试语料
    test_corpus = [
        "The cat sat on the mat and looked at the window with curiosity.",
        "Newton discovered that gravity is a universal force acting between all masses.",
        "The government announced a new policy to reduce carbon emissions by 2030.",
        "In quantum mechanics, particles can exist in superposition of multiple states.",
        "She wrote a recursive function to traverse the binary tree in Python.",
        "The Supreme Court ruled that the law was unconstitutional and must be repealed.",
        "The artist painted a beautiful landscape of mountains reflected in the lake.",
        "Machine learning models require large amounts of labeled training data.",
    ]
    
    # 收集所有层的残差流：hook_resid_pre (进入该层之前) 和 hook_resid_post (离开该层之后)
    # 以及 Attention 和 MLP 各自的输出
    hook_names = []
    for l in range(n_layers):
        hook_names.extend([
            f"blocks.{l}.hook_resid_pre",
            f"blocks.{l}.hook_resid_post",
            f"blocks.{l}.hook_attn_out",
            f"blocks.{l}.hook_mlp_out",
        ])
    
    print(f"\n[2/3] 拦截 {len(hook_names)} 个检查点的残差流...")
    
    # 存储结果
    all_results = {
        "relative_delta_norm": [],         # ||Δx|| / ||x||
        "attn_contribution_norm": [],     # ||attn_out|| / ||x||
        "mlp_contribution_norm": [],      # ||mlp_out|| / ||x||
        "delta_svd_top5_variance": [],    # 增量的 Top-5 SVD 方差占比
        "delta_intrinsic_dim_95": [],     # 增量的 95% 内禀维度
        "cosine_with_previous": [],       # 相邻层增量方向的余弦相似度
    }
    
    for text in tqdm(test_corpus, desc="  处理语料"):
        _, cache = model.run_with_cache(text, names_filter=lambda name: name in hook_names)
        
        per_text_delta_norms = []
        per_text_attn_norms = []
        per_text_mlp_norms = []
        per_text_svd_top5 = []
        per_text_intdim = []
        per_text_cosine = []
        
        prev_delta_mean = None
        
        for l in range(n_layers):
            resid_pre = cache[f"blocks.{l}.hook_resid_pre"].squeeze(0).cpu()   # [seq, d_model]
            resid_post = cache[f"blocks.{l}.hook_resid_post"].squeeze(0).cpu() # [seq, d_model]
            attn_out = cache[f"blocks.{l}.hook_attn_out"].squeeze(0).cpu()     # [seq, d_model]
            mlp_out = cache[f"blocks.{l}.hook_mlp_out"].squeeze(0).cpu()       # [seq, d_model]
            
            # 1. 残差增量
            delta = resid_post - resid_pre  # [seq, d_model]
            
            # 相对增量范数 (对每个 Token 计算，取平均)
            pre_norm = resid_pre.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            rel_delta = (delta.norm(dim=-1) / pre_norm.squeeze(-1)).mean().item()
            per_text_delta_norms.append(rel_delta)
            
            # Attention 和 MLP 各自的贡献范数
            attn_rel = (attn_out.norm(dim=-1) / pre_norm.squeeze(-1)).mean().item()
            mlp_rel = (mlp_out.norm(dim=-1) / pre_norm.squeeze(-1)).mean().item()
            per_text_attn_norms.append(attn_rel)
            per_text_mlp_norms.append(mlp_rel)
            
            # 2. 增量的 SVD 分析
            delta_np = delta.numpy()
            delta_centered = delta_np - delta_np.mean(axis=0, keepdims=True)
            
            if delta_centered.shape[0] > 1:
                U, S, Vt = np.linalg.svd(delta_centered, full_matrices=False)
                var_explained = (S ** 2) / (S ** 2).sum()
                top5_var = var_explained[:5].sum()
                cumvar = np.cumsum(var_explained)
                int_dim_95 = int(np.argmax(cumvar >= 0.95)) + 1
            else:
                top5_var = 1.0
                int_dim_95 = 1
            
            per_text_svd_top5.append(float(top5_var))
            per_text_intdim.append(int_dim_95)
            
            # 3. 与前一层增量方向的余弦相似度
            curr_delta_mean = delta.mean(dim=0).numpy()  # [d_model]
            if prev_delta_mean is not None:
                cos = np.dot(curr_delta_mean, prev_delta_mean) / (
                    np.linalg.norm(curr_delta_mean) * np.linalg.norm(prev_delta_mean) + 1e-8
                )
                per_text_cosine.append(float(cos))
            else:
                per_text_cosine.append(0.0)
            prev_delta_mean = curr_delta_mean
        
        all_results["relative_delta_norm"].append(per_text_delta_norms)
        all_results["attn_contribution_norm"].append(per_text_attn_norms)
        all_results["mlp_contribution_norm"].append(per_text_mlp_norms)
        all_results["delta_svd_top5_variance"].append(per_text_svd_top5)
        all_results["delta_intrinsic_dim_95"].append(per_text_intdim)
        all_results["cosine_with_previous"].append(per_text_cosine)
    
    # ==========================================
    # 汇总报告
    # ==========================================
    print("\n[3/3] 生成分析报告...")
    
    # 对所有语料取平均
    avg = lambda key: [float(np.mean([all_results[key][t][l] for t in range(len(test_corpus))])) for l in range(n_layers)]
    
    avg_delta = avg("relative_delta_norm")
    avg_attn = avg("attn_contribution_norm")
    avg_mlp = avg("mlp_contribution_norm")
    avg_svd5 = avg("delta_svd_top5_variance")
    avg_intdim = avg("delta_intrinsic_dim_95")
    avg_cos = avg("cosine_with_previous")
    
    print("\n  === 逐层残差增量分析 ===")
    print(f"  {'层':>4} | {'|Δx|/|x|':>8} | {'Attn贡献':>8} | {'MLP贡献':>8} | {'Top5方差%':>9} | {'内禀维度':>8} | {'与前层cos':>9}")
    print(f"  {'-'*4} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*9} | {'-'*8} | {'-'*9}")
    
    for l in range(n_layers):
        print(f"  L{l:>2}  | {avg_delta[l]:>8.4f} | {avg_attn[l]:>8.4f} | {avg_mlp[l]:>8.4f} | "
              f"{avg_svd5[l]*100:>8.1f}% | {avg_intdim[l]:>8.1f} | {avg_cos[l]:>9.4f}")
    
    # 分析趋势
    shallow_delta = np.mean(avg_delta[:4])
    deep_delta = np.mean(avg_delta[8:])
    shallow_intdim = np.mean(avg_intdim[:4])
    deep_intdim = np.mean(avg_intdim[8:])
    
    print(f"\n  === 趋势分析 ===")
    print(f"  浅层 (L0-3) 平均增量幅度: {shallow_delta:.4f}")
    print(f"  深层 (L8-11) 平均增量幅度: {deep_delta:.4f}")
    print(f"  深/浅比: {deep_delta/shallow_delta:.2f}x {'(深层微调)' if deep_delta < shallow_delta else '(深层放大)'}")
    print(f"  浅层平均内禀维度: {shallow_intdim:.1f}")
    print(f"  深层平均内禀维度: {deep_intdim:.1f}")
    print(f"  维度{'收缩' if deep_intdim < shallow_intdim else '扩张'}倍数: {shallow_intdim/deep_intdim:.2f}x")
    
    # MLP vs Attention 对比
    total_attn = sum(avg_attn)
    total_mlp = sum(avg_mlp)
    print(f"\n  Attention 总贡献: {total_attn:.4f}")
    print(f"  MLP 总贡献:       {total_mlp:.4f}")
    print(f"  MLP/Attn 比:      {total_mlp/total_attn:.2f}x")
    
    # 保存报告
    os.makedirs("tempdata", exist_ok=True)
    report = {
        "experiment": "Phase XXXVI 拼图 #2: GPT-2 逐层残差增量 SVD",
        "model": "gpt2-small",
        "relative_delta_norm_by_layer": avg_delta,
        "attn_contribution_by_layer": avg_attn,
        "mlp_contribution_by_layer": avg_mlp,
        "top5_svd_variance_by_layer": avg_svd5,
        "intrinsic_dim_95_by_layer": avg_intdim,
        "cosine_with_previous_by_layer": avg_cos,
        "analysis": {
            "shallow_delta_mean": float(shallow_delta),
            "deep_delta_mean": float(deep_delta),
            "deep_shallow_ratio": float(deep_delta / shallow_delta),
            "mlp_attn_ratio": float(total_mlp / total_attn),
        }
    }
    
    report_path = "tempdata/exp_residual_delta_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n[完成] 报告保存至 {report_path}")
    
    print("\n" + "=" * 70)
    print("结论摘要")
    print("=" * 70)
    if deep_delta < shallow_delta:
        print("✅ 深层确实在做\"微调修正\"而非\"全面重构\"")
        print("   → 大脑的深层皮层可能是渐进精炼，而非传统认为的阶梯式处理")
    else:
        print("⚠️ 深层增量并未减小，可能是\"持续重构\"模式")
        print("   → 需要重新审视\"残差叠加\"假设")

if __name__ == "__main__":
    main()
