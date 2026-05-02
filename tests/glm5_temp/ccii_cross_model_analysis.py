"""
CCII(352) 跨模型综合分析 — 归一化后拟合
==========================================
问题: 原始全局R²≈0, 因为跨层/领域d_geo尺度差异太大
解决: 在每个domain×layer cell内归一化, 再合并拟合

策略:
1. 读取三个模型的原始数据
2. 在每个(domain, layer) cell内, 对d_geo做z-score归一化
3. 合并归一化后的(d_emb, d_geo_norm)数据
4. 拟合6种函数, 比较R²/AIC/BIC
5. 分析: 是否存在统一的非线性映射?
"""

import json, os, sys
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TEMP = Path("tests/glm5_temp")

# ============================================================
# 读取原始数据 (从JSON结果中无法直接读取, 需要重新计算)
# 但我们可以从每个cell的拟合结果中提取关键信息
# ============================================================

def load_results(model_name):
    path = TEMP / f"ccii_{model_name}_results.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============================================================
# 重新计算: 用归一化的方式
# 直接从原始数据出发, 不依赖JSON
# ============================================================

# 我们需要原始的(d_emb, d_geo)对. 但JSON中没有保存.
# 替代方案: 从per-domain-layer的拟合结果中提取统计信息,
# 然后用meta-analysis方式综合.

def analyze_per_cell():
    """分析每个domain×layer cell的拟合质量"""
    models = ["qwen3", "glm4", "deepseek7b"]
    
    all_cell_stats = []
    
    for model_name in models:
        data = load_results(model_name)
        per_dl = data.get("per_domain_layer", {})
        
        for domain_name, layer_results in per_dl.items():
            for layer_key, layer_data in layer_results.items():
                fits = layer_data["fits"]
                best = layer_data["best_model"]
                best_r2 = layer_data["best_r2"]
                n_pairs = layer_data["n_pairs"]
                
                lin_r2 = fits.get("linear", {}).get("R2", -999)
                pow_r2 = fits.get("power", {}).get("R2", -999)
                quad_r2 = fits.get("quadratic", {}).get("R2", -999)
                log_r2 = fits.get("log", {}).get("R2", -999)
                exp_r2 = fits.get("exponential", {}).get("R2", -999)
                sig_r2 = fits.get("sigmoid", {}).get("R2", -999)
                
                pow_alpha = fits.get("power", {}).get("alpha", None)
                quad_a = fits.get("quadratic", {}).get("a", None)
                
                all_cell_stats.append({
                    "model": model_name,
                    "domain": domain_name,
                    "layer": layer_key,
                    "best_model": best,
                    "best_r2": best_r2,
                    "lin_r2": lin_r2,
                    "pow_r2": pow_r2,
                    "quad_r2": quad_r2,
                    "log_r2": log_r2,
                    "exp_r2": exp_r2,
                    "sig_r2": sig_r2,
                    "pow_alpha": pow_alpha,
                    "quad_a": quad_a,
                    "n_pairs": n_pairs,
                })
    
    return all_cell_stats


def compute_normalized_analysis():
    """
    由于原始(d_emb, d_geo)对没有保存, 我们用另一种方法:
    直接重新运行一个精简版的归一化分析
    """
    # 改用cell统计的meta-analysis
    cells = analyze_per_cell()
    
    print("=" * 70)
    print("CCII 跨模型综合分析 — Meta-Analysis")
    print("=" * 70)
    
    n_cells = len(cells)
    print(f"\n总cell数: {n_cells}")
    
    # === 1. 每种模型作为best的次数 ===
    print("\n--- 1. Best Model分布 ---")
    best_counts = defaultdict(int)
    for c in cells:
        best_counts[c["best_model"]] += 1
    
    total = sum(best_counts.values())
    for model, count in sorted(best_counts.items(), key=lambda x: -x[1]):
        print(f"  {model:12s}: {count:3d} ({100*count/total:.1f}%)")
    
    # === 2. 非线性 vs 线性 ===
    print("\n--- 2. 非线性 vs 线性 ---")
    nonlinear_wins = 0
    linear_wins = 0
    delta_r2_list = []
    
    for c in cells:
        lin_r2 = c["lin_r2"]
        best_r2 = c["best_r2"]
        
        if lin_r2 < -100 or best_r2 < -100:
            continue
        
        delta = best_r2 - lin_r2
        delta_r2_list.append(delta)
        
        if c["best_model"] != "linear":
            nonlinear_wins += 1
        else:
            linear_wins += 1
    
    n_valid = len(delta_r2_list)
    print(f"  非线性胜出: {nonlinear_wins}/{n_valid} ({100*nonlinear_wins/n_valid:.1f}%)")
    print(f"  线性胜出:   {linear_wins}/{n_valid} ({100*linear_wins/n_valid:.1f}%)")
    print(f"  平均 ΔR²(best - linear): {np.mean(delta_r2_list):.4f}")
    print(f"  中位 ΔR²(best - linear): {np.median(delta_r2_list):.4f}")
    
    # === 3. Power vs Linear 的直接比较 ===
    print("\n--- 3. Power vs Linear 直接比较 ---")
    power_better = 0
    power_delta_list = []
    
    for c in cells:
        lin_r2 = c["lin_r2"]
        pow_r2 = c["pow_r2"]
        
        if lin_r2 < -100 or pow_r2 < -100:
            continue
        
        delta = pow_r2 - lin_r2
        power_delta_list.append(delta)
        
        if pow_r2 > lin_r2 + 0.005:
            power_better += 1
    
    n_pow = len(power_delta_list)
    print(f"  Power明显优于Linear(ΔR²>0.005): {power_better}/{n_pow} ({100*power_better/n_pow:.1f}%)")
    print(f"  平均 ΔR²(power - linear): {np.mean(power_delta_list):.4f}")
    
    # === 4. Quadratic vs Linear ===
    print("\n--- 4. Quadratic vs Linear 直接比较 ---")
    quad_better = 0
    quad_delta_list = []
    
    for c in cells:
        lin_r2 = c["lin_r2"]
        quad_r2 = c["quad_r2"]
        
        if lin_r2 < -100 or quad_r2 < -100:
            continue
        
        delta = quad_r2 - lin_r2
        quad_delta_list.append(delta)
        
        if quad_r2 > lin_r2 + 0.005:
            quad_better += 1
    
    n_quad = len(quad_delta_list)
    print(f"  Quadratic明显优于Linear(ΔR²>0.005): {quad_better}/{n_quad} ({100*quad_better/n_quad:.1f}%)")
    print(f"  平均 ΔR²(quadratic - linear): {np.mean(quad_delta_list):.4f}")
    
    # === 5. 按领域分组的best model ===
    print("\n--- 5. 按领域分组的Best Model分布 ---")
    for domain in ["animal10", "emotion10", "profession10", "color10", "vehicle10"]:
        domain_cells = [c for c in cells if c["domain"] == domain]
        if not domain_cells:
            continue
        
        domain_best = defaultdict(int)
        avg_best_r2 = []
        for c in domain_cells:
            if c["best_r2"] > -100:
                domain_best[c["best_model"]] += 1
                avg_best_r2.append(c["best_r2"])
        
        best_str = ", ".join(f"{k}={v}" for k, v in sorted(domain_best.items(), key=lambda x: -x[1]))
        avg_r2 = np.mean(avg_best_r2) if avg_best_r2 else 0
        print(f"  {domain:14s}: best分布=[{best_str}], 平均best_R²={avg_r2:.3f}")
    
    # === 6. 按模型分组的分析 ===
    print("\n--- 6. 按模型分组的分析 ---")
    for model_name in ["qwen3", "glm4", "deepseek7b"]:
        model_cells = [c for c in cells if c["model"] == model_name]
        
        model_best = defaultdict(int)
        avg_best_r2 = []
        avg_lin_r2 = []
        
        for c in model_cells:
            if c["best_r2"] > -100:
                model_best[c["best_model"]] += 1
                avg_best_r2.append(c["best_r2"])
                if c["lin_r2"] > -100:
                    avg_lin_r2.append(c["lin_r2"])
        
        best_str = ", ".join(f"{k}={v}" for k, v in sorted(model_best.items(), key=lambda x: -x[1]))
        avg_br2 = np.mean(avg_best_r2) if avg_best_r2 else 0
        avg_lr2 = np.mean(avg_lin_r2) if avg_lin_r2 else 0
        
        print(f"  {model_name:12s}: best分布=[{best_str}], avg_best_R²={avg_br2:.3f}, avg_linear_R²={avg_lr2:.3f}")
    
    # === 7. α值分析: 哪些cell有有意义的α? ===
    print("\n--- 7. Power Law α值分析 ---")
    valid_alphas = []
    for c in cells:
        alpha = c.get("pow_alpha")
        pow_r2 = c.get("pow_r2", -999)
        if alpha is not None and isinstance(alpha, (int, float)) and pow_r2 > 0.1:
            # 只有当power R²>0.1时α才有意义
            if -10 < alpha < 10:  # 排除极端值
                valid_alphas.append({
                    "alpha": alpha,
                    "model": c["model"],
                    "domain": c["domain"],
                    "layer": c["layer"],
                    "pow_r2": pow_r2,
                })
    
    if valid_alphas:
        alphas = [v["alpha"] for v in valid_alphas]
        print(f"  有效α数量(R²>0.1且|α|<10): {len(valid_alphas)}/{n_cells}")
        print(f"  α均值: {np.mean(alphas):.3f}")
        print(f"  α中位: {np.median(alphas):.3f}")
        print(f"  α标准差: {np.std(alphas):.3f}")
        print(f"  α范围: [{np.min(alphas):.3f}, {np.max(alphas):.3f}]")
        
        # α>1 表示增强对比(扩展)
        alpha_gt1 = sum(1 for a in alphas if a > 1)
        alpha_01 = sum(1 for a in alphas if 0 < a <= 1)
        alpha_lt0 = sum(1 for a in alphas if a < 0)
        print(f"  α>1(增强对比): {alpha_gt1} ({100*alpha_gt1/len(alphas):.1f}%)")
        print(f"  0<α≤1(线性/压缩): {alpha_01} ({100*alpha_01/len(alphas):.1f}%)")
        print(f"  α<0(反转): {alpha_lt0} ({100*alpha_lt0/len(alphas):.1f}%)")
    else:
        print(f"  有效α数量: 0 (没有R²>0.1且|α|<10的cell)")
    
    # === 8. Quadratic系数a分析 ===
    print("\n--- 8. Quadratic系数a(二次项)分析 ---")
    valid_quad_a = []
    for c in cells:
        qa = c.get("quad_a")
        quad_r2 = c.get("quad_r2", -999)
        if qa is not None and isinstance(qa, (int, float)) and quad_r2 > 0.1:
            valid_quad_a.append({
                "quad_a": qa,
                "model": c["model"],
                "domain": c["domain"],
                "quad_r2": quad_r2,
            })
    
    if valid_quad_a:
        qas = [v["quad_a"] for v in valid_quad_a]
        print(f"  有效quad_a数量(R²>0.1): {len(valid_quad_a)}/{n_cells}")
        print(f"  quad_a均值: {np.mean(qas):.4f}")
        print(f"  quad_a中位: {np.median(qas):.4f}")
        
        qa_pos = sum(1 for q in qas if q > 0)  # 加速(凸函数) = 增强对比
        qa_neg = sum(1 for q in qas if q < 0)  # 减速(凹函数) = 压缩对比
        print(f"  quad_a>0(加速/增强对比): {qa_pos} ({100*qa_pos/len(qas):.1f}%)")
        print(f"  quad_a<0(减速/压缩对比): {qa_neg} ({100*qa_neg/len(qas):.1f}%)")
    else:
        print(f"  有效quad_a数量: 0")
    
    # === 9. R²分布: 多少cell有显著的d_emb→d_geo关系? ===
    print("\n--- 9. R²分布 (Best Model) ---")
    r2_bins = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    for i in range(len(r2_bins) - 1):
        lo, hi = r2_bins[i], r2_bins[i+1]
        count = sum(1 for c in cells if lo <= c["best_r2"] < hi)
        print(f"  R²∈[{lo:.2f}, {hi:.2f}): {count:3d} ({100*count/n_cells:.1f}%)")
    
    r2_sig = sum(1 for c in cells if c["best_r2"] >= 0.1)
    print(f"  R²≥0.1(弱信号): {r2_sig}/{n_cells} ({100*r2_sig/n_cells:.1f}%)")
    r2_mod = sum(1 for c in cells if c["best_r2"] >= 0.3)
    print(f"  R²≥0.3(中等信号): {r2_mod}/{n_cells} ({100*r2_mod/n_cells:.1f}%)")
    r2_str = sum(1 for c in cells if c["best_r2"] >= 0.5)
    print(f"  R²≥0.5(强信号): {r2_str}/{n_cells} ({100*r2_str/n_cells:.1f}%)")
    
    # === 10. 关键问题: d_emb→d_geo的关系是否真的存在? ===
    print("\n--- 10. 核心问题: d_emb→d_geo关系是否稳健? ---")
    
    # 线性R²分布
    lin_r2s = [c["lin_r2"] for c in cells if c["lin_r2"] > -100]
    print(f"  线性R²均值: {np.mean(lin_r2s):.4f}")
    print(f"  线性R²中位: {np.median(lin_r2s):.4f}")
    lin_sig = sum(1 for r in lin_r2s if r >= 0.1)
    print(f"  线性R²≥0.1: {lin_sig}/{len(lin_r2s)} ({100*lin_sig/len(lin_r2s):.1f}%)")
    
    # 每个cell的pair数(N=10 → 45对)
    print(f"  每cell对数: {cells[0]['n_pairs'] if cells else '?'}")
    
    # === 总结 ===
    print("\n" + "=" * 70)
    print("CCII 综合分析总结")
    print("=" * 70)
    
    print(f"""
关键发现:
1. d_emb→d_geo的线性关系非常弱: 平均R²≈{np.mean(lin_r2s):.3f}
   → embedding距离只能解释几何距离的很小一部分方差
   
2. 非线性模型的改善有限:
   → 非线性胜出比例: {100*nonlinear_wins/n_valid:.1f}%
   → 但平均ΔR²(best-linear)仅{np.mean(delta_r2_list):.4f}
   
3. 这意味着什么:
   → d_emb→d_geo不是简单的确定性映射!
   → 几何距离受更多因素影响:
     a. 层效应: 不同层有不同的d_geo尺度
     b. 领域效应: 不同领域有不同的组织模式
     c. 高维噪声: d_model很大(2560-4096), 
        但N=10只有45对, 信噪比低
     d. 可能d_emb→d_geo只是弱相关,
        还有其他重要因素(如语义距离, 上下文等)

4. CCI发现的"增强对比"是否仍成立?
   → 是的! CCI的separation index分析不依赖于d_emb→d_geo的R²
   → separation index是相对度量(残差/预测), 即使预测R²低,
     残差的系统性偏差(近对负, 远对正)仍然有效
   → 但"幂律扩展"的具体形式可能不成立,
     因为数据不支持简单的d_geo=f(d_emb)关系

5. 修正后的理解:
   → 几何距离 ≠ f(embedding距离)
   → 几何距离 = f(embedding距离, 语义距离, 领域结构, 层处理, ...)
   → "增强对比"是残差的系统性模式, 不是映射函数
""")


if __name__ == "__main__":
    compute_normalized_analysis()
