"""
CCIII(353) 跨模型综合分析
==========================================
"""

import json, os, sys
import numpy as np
from pathlib import Path
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TEMP = Path("tests/glm5_temp")

def load_results(model_name):
    path = TEMP / f"cciii_{model_name}_results.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    models = ["qwen3", "glm4", "deepseek7b"]
    domains = ["animal10", "emotion10", "profession10", "color10", "vehicle10"]
    
    # 合并所有数据
    all_data = []
    for model_name in models:
        data = load_results(model_name)
        for d in data["summary_data"]:
            all_data.append(d)
    
    print("=" * 70)
    print("CCIII 跨模型综合分析")
    print("=" * 70)
    
    n_total = len(all_data)
    print(f"\n总数据点: {n_total} (3模型 × 5领域 × 10层)")
    
    # === 1. 全局统计 ===
    print("\n--- 1. 全局统计 ---")
    
    avg_R2_both = np.mean([d["R2_both"] for d in all_data])
    avg_beta_emb = np.mean([d["beta_emb"] for d in all_data])
    avg_beta_sem = np.mean([d["beta_sem"] for d in all_data])
    avg_beta_inter = np.mean([d["beta_inter"] for d in all_data])
    avg_delta_emb = np.mean([d["delta_R2_emb"] for d in all_data])
    avg_delta_sem = np.mean([d["delta_R2_sem"] for d in all_data])
    avg_delta_inter = np.mean([d["delta_R2_inter"] for d in all_data])
    avg_contrast = np.mean([d["contrast_index"] for d in all_data])
    
    beta_emb_pos = sum(1 for d in all_data if d["beta_emb"] > 0)
    beta_sem_pos = sum(1 for d in all_data if d["beta_sem"] > 0)
    
    print(f"  R²_both均值: {avg_R2_both:.3f}")
    print(f"  β_emb均值: {avg_beta_emb:+.3f} (正: {beta_emb_pos}/{n_total}={100*beta_emb_pos/n_total:.0f}%)")
    print(f"  β_sem均值: {avg_beta_sem:+.3f} (正: {beta_sem_pos}/{n_total}={100*beta_sem_pos/n_total:.0f}%)")
    print(f"  β_inter均值: {avg_beta_inter:+.3f}")
    print(f"  ΔR²_emb均值: {avg_delta_emb:.3f}")
    print(f"  ΔR²_sem均值: {avg_delta_sem:.3f}")
    print(f"  ΔR²_inter均值: {avg_delta_inter:.3f}")
    print(f"  对比指数均值: {avg_contrast:+.3f}")
    
    # === 2. Embedding vs Semantic 胜出 ===
    print("\n--- 2. Embedding vs Semantic 胜出 ---")
    emb_wins = sum(1 for d in all_data if d["delta_R2_emb"] > d["delta_R2_sem"])
    sem_wins = n_total - emb_wins
    print(f"  ΔR²_emb > ΔR²_sem: {emb_wins}/{n_total} ({100*emb_wins/n_total:.0f}%)")
    print(f"  ΔR²_sem ≥ ΔR²_emb: {sem_wins}/{n_total} ({100*sem_wins/n_total:.0f}%)")
    
    # === 3. 按领域分组的跨模型平均 ===
    print("\n--- 3. 按领域分组(跨模型平均) ---")
    for domain in domains:
        dd = [d for d in all_data if d["domain"] == domain]
        n_d = len(dd)
        
        avg_re2 = np.mean([d["R2_both"] for d in dd])
        avg_be = np.mean([d["beta_emb"] for d in dd])
        avg_bs = np.mean([d["beta_sem"] for d in dd])
        avg_bi = np.mean([d["beta_inter"] for d in dd])
        avg_de = np.mean([d["delta_R2_emb"] for d in dd])
        avg_ds = np.mean([d["delta_R2_sem"] for d in dd])
        avg_ci = np.mean([d["contrast_index"] for d in dd])
        
        be_pos = sum(1 for d in dd if d["beta_emb"] > 0)
        bs_pos = sum(1 for d in dd if d["beta_sem"] > 0)
        emb_w = sum(1 for d in dd if d["delta_R2_emb"] > d["delta_R2_sem"])
        ci_pos = sum(1 for d in dd if d["contrast_index"] > 0)
        
        print(f"  {domain:14s}: R²={avg_re2:.3f}, β_emb={avg_be:+.3f}({be_pos}/{n_d}), "
              f"β_sem={avg_bs:+.3f}({bs_pos}/{n_d}), "
              f"winner:emb={emb_w}/{n_d}, contrast={avg_ci:+.3f}({ci_pos}/{n_d}+)")
    
    # === 4. 按模型分组 ===
    print("\n--- 4. 按模型分组 ---")
    for model_name in models:
        md = [d for d in all_data if d["model"] == model_name]
        
        avg_re2 = np.mean([d["R2_both"] for d in md])
        avg_be = np.mean([d["beta_emb"] for d in md])
        avg_bs = np.mean([d["beta_sem"] for d in md])
        avg_ci = np.mean([d["contrast_index"] for d in md])
        
        be_pos = sum(1 for d in md if d["beta_emb"] > 0)
        ci_pos = sum(1 for d in md if d["contrast_index"] > 0)
        
        print(f"  {model_name:12s}: R²={avg_re2:.3f}, β_emb={avg_be:+.3f}({be_pos}/50), "
              f"β_sem={avg_bs:+.3f}, contrast={avg_ci:+.3f}({ci_pos}/50+)")
    
    # === 5. β_emb的符号分析: 何时为负? ===
    print("\n--- 5. β_emb为负的情况 ---")
    neg_emb = [d for d in all_data if d["beta_emb"] < 0]
    print(f"  总数: {len(neg_emb)}/{n_total} ({100*len(neg_emb)/n_total:.0f}%)")
    
    for d in neg_emb:
        print(f"    {d['model']:10s} {d['domain']:14s} L{d['layer']:2d}: "
              f"β_emb={d['beta_emb']:+.3f}, β_sem={d['beta_sem']:+.3f}, "
              f"R²={d['R2_both']:.3f}")
    
    # === 6. β_sem的符号分析: 何时为负? ===
    print("\n--- 6. β_sem为负的情况 ---")
    neg_sem = [d for d in all_data if d["beta_sem"] < 0]
    print(f"  总数: {len(neg_sem)}/{n_total} ({100*len(neg_sem)/n_total:.0f}%)")
    
    # 按领域分组统计
    for domain in domains:
        ns = [d for d in neg_sem if d["domain"] == domain]
        print(f"    {domain:14s}: {len(ns)}/{sum(1 for d in all_data if d['domain']==domain)}")
    
    # === 7. 对比指数分析 ===
    print("\n--- 7. 对比指数(增强对比)分析 ---")
    ci_pos = sum(1 for d in all_data if d["contrast_index"] > 0)
    ci_neg = sum(1 for d in all_data if d["contrast_index"] < 0)
    print(f"  增强(>0): {ci_pos}/{n_total} ({100*ci_pos/n_total:.0f}%)")
    print(f"  压缩(<0): {ci_neg}/{n_total} ({100*ci_neg/n_total:.0f}%)")
    
    # 按领域×模型
    print("\n  按领域×模型的对比指数:")
    for domain in domains:
        vals = []
        for model_name in models:
            md = [d for d in all_data if d["domain"] == domain and d["model"] == model_name]
            avg_ci = np.mean([d["contrast_index"] for d in md])
            pos = sum(1 for d in md if d["contrast_index"] > 0)
            vals.append(f"{model_name}={avg_ci:+.3f}({pos}/10)")
        print(f"    {domain:14s}: {', '.join(vals)}")
    
    # === 8. 交互效应分析 ===
    print("\n--- 8. 交互效应(β_inter)分析 ---")
    avg_inter = np.mean([d["beta_inter"] for d in all_data])
    inter_pos = sum(1 for d in all_data if d["beta_inter"] > 0)
    inter_neg = sum(1 for d in all_data if d["beta_inter"] < 0)
    print(f"  β_inter均值: {avg_inter:+.3f}")
    print(f"  β_inter>0: {inter_pos}/{n_total}")
    print(f"  β_inter<0: {inter_neg}/{n_total}")
    
    # 按领域
    for domain in domains:
        dd = [d for d in all_data if d["domain"] == domain]
        avg_i = np.mean([d["beta_inter"] for d in dd])
        i_pos = sum(1 for d in dd if d["beta_inter"] > 0)
        print(f"    {domain:14s}: β_inter={avg_i:+.3f} ({i_pos}/{len(dd)}+)")
    
    # === 9. 核心发现总结 ===
    print("\n" + "=" * 70)
    print("CCIII 核心发现总结")
    print("=" * 70)
    
    print(f"""
1. ★★★★★ Embedding距离是比语义距离更强的预测因子
   ΔR²_emb均值={avg_delta_emb:.3f} vs ΔR²_sem均值={avg_delta_sem:.3f}
   Embedding胜出: {emb_wins}/{n_total} ({100*emb_wins/n_total:.0f}%)
   
   → 几何距离主要由分布相似性(共现模式)决定
   → 语义距离的独立贡献较小但稳定

2. ★★★★★ β_emb符号的领域依赖性
   β_emb为负: {len(neg_emb)}/{n_total} ({100*len(neg_emb)/n_total:.0f}%)
   集中在: vehicle领域(GLM4: -0.24, DS7B: -0.38)
   
   → 某些领域中, embedding相似的类别几何更远!
   → 这不是"判别性分离", 而是"语义重排序":
      embedding空间和几何空间的组织轴不同!
   
3. ★★★★ β_sem符号的领域依赖性
   β_sem为负: {len(neg_sem)}/{n_total} ({100*len(neg_sem)/n_total:.0f}%)
   集中在: profession领域
   
   → profession中, 语义相似的类别几何更远?
   → 可能: prestige维度是核心组织轴(β_emb强),
      但手定义的prestige/income/education/caring维度
      没有完美捕捉真正的组织轴

4. ★★★★ 对比指数的模型差异
   Qwen3: 42%增强 vs 58%压缩
   GLM4:  22%增强 vs 78%压缩
   DS7B:  66%增强 vs 34%压缩
   
   → Qwen3和GLM4倾向于"压缩对比"(近对偏正残差)
   → DS7B倾向于"增强对比"(近对偏负残差)
   → CCI的"增强对比"不是所有模型的普遍模式!

5. ★★★ R²_both仍然很低(0.12-0.18)
   → emb+sem只能解释12-18%的几何距离方差
   → 82-88%的方差无法解释
   → 需要更多因素或更大N
""")


if __name__ == "__main__":
    main()
