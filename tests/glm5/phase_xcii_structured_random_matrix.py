"""
Phase XCII-P447/448/449/450/451: 结构化随机矩阵理论
======================================================================

核心目标: 解释"2×PR/d"因子，建立DNN信号传播的解析理论

P447: W_U完整奇异值谱分析 + Marchenko-Pastur检验 (★★★★★最高优先级★★★★★)
  - 对W_U做完整SVD(或尽可能多的分量)，获取奇异值谱
  - 与Marchenko-Pastur分布对比，检验W_U是否符合随机矩阵
  - 关键: 如果W_U谱偏离MP → 说明有训练结构，2×因子可能来自结构
  - 关键: 如果W_U谱符合MP → 说明2×因子来自截断偏差或其他数学效应

P448: 本征值聚焦效应 — 投影能量的加权分析
  - 计算W_U行空间中每个本征方向的"投影浓度"
  - 如果投影能量集中在少数大奇异值方向 → 聚焦效应解释2×因子
  - 如果投影能量均匀分布 → 聚焦效应不成立

P449: gain/||Δ||比值的层间分析 — 伪相关检验
  - P445发现delta_norm与gain r=0.996，但这可能是伪相关
  - 检验: gain/||Δ||是否为常数?
  - 如果是常数 → 相关是平凡的(gain = ||proj|| = ||U^T·Δ||, 由||Δ||主导)
  - 如果不是常数 → 有额外结构需要解释

P450: 逃逸能量全谱SVD结构分析
  - 99.6%的信号能量逃逸了W_U行空间，这些能量有结构吗?
  - 对逃逸分量做SVD分析，看是否是纯噪声
  - 如果有结构 → 逃逸能量携带信息，需要"非线性纠错码"理论
  - 如果纯噪声 → 信号传播确实是"各向同性放大×随机旋转"

P451: 多尺度PR分析 (★★★★★关键判别★★★★★)
  - 用不同的k_svd(10→2000)截断SVD，看PR如何变化
  - 如果PR随k快速增长 → k=300的截断严重低估了PR → 2×因子是数学伪影
  - 如果PR在k≈300已收敛 → 2×因子是真实的 → 需要新理论

实验模型: qwen3 → deepseek7b → glm4 (串行，避免GPU溢出)
"""

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.linalg import svds

# ===== 导入标准模型工具 =====
sys.path.insert(0, str(Path(__file__).parent))
from model_utils import (
    MODEL_CONFIGS,
    load_model,
    get_layers,
    get_model_info,
    get_layer_weights,
    get_W_U,
    release_model,
    get_sample_layers,
    get_attr_direction,
    inject_at_embed,
    collect_layer_outputs,
    compute_cos,
    compute_recoding_ratio,
    compute_recoding_ratio_cached,
    LayerWeights,
    ModelInfo,
)

# ===== 实验配置 =====
PHASE_NAME = "Phase XCII"
EXPERIMENT_DESC = "结构化随机矩阵理论—破解2×因子"

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 测试数据
TEST_ATTRS = ["red", "blue", "big", "small", "hot", "cold", "sweet", "sour"]
PROMPT = "The apple is"
BETA = 8.0


# ================================================================
# P447: W_U完整奇异值谱分析 + Marchenko-Pastur检验
# ================================================================

def run_p447(model, tokenizer, device, model_info):
    """
    P447: W_U完整奇异值谱分析 + Marchenko-Pastur检验
    
    原理:
      Marchenko-Pastur定律: 对一个m×n的随机矩阵X(元素i.i.d. N(0,σ²/m)),
      其奇异值谱(即X^T X的本征值)满足:
        λ ∈ [(1-√(n/m))², (1+√(n/m))²] × σ²
      
      对于W_U (vocab × d_model), m=vocab, n=d_model
      理论MP下界 = (1 - √(d_model/vocab))²
      理论MP上界 = (1 + √(d_model/vocab))²
      
      如果W_U的训练结构使某些奇异值偏离MP分布:
      → 偏离的部分代表"语言结构"，可能解释2×因子
    
    方法:
      1. 对W_U^T做尽可能多的SVD分量
      2. 获取完整奇异值谱
      3. 计算MP理论分布参数
      4. 对比: 实际谱 vs MP理论
      5. 分析偏离: 哪些奇异值超出MP边界?
    """
    print(f"\n{'='*60}")
    print(f"P447: W_U完整奇异值谱 + MP检验 - {model_info.name}")
    print(f"{'='*60}")
    
    results = {}
    d_model = model_info.d_model
    
    # Step 1: 获取W_U并做SVD
    W_U = get_W_U(model)  # [vocab, d_model]
    vocab_size = W_U.shape[0]
    wu_shape = list(W_U.shape)
    
    print(f"  W_U shape: {wu_shape}")
    print(f"  d_model={d_model}, vocab={vocab_size}")
    
    W_U_T = W_U.T.astype(np.float32)  # [d_model, vocab]
    del W_U
    import gc; gc.collect()
    
    # 尽可能多的SVD分量
    max_k = min(W_U_T.shape[0], W_U_T.shape[1]) - 2  # d_model - 2
    # 对于d_model=2560, max_k=2558; 但scipy svds可能对大k不稳定
    # 使用多个批次: 小k(稳定) + 中k + 大k
    k_values = [min(300, max_k), min(800, max_k), min(1500, max_k), max_k]
    k_values = sorted(set(k_values))
    
    print(f"  SVD k值: {k_values}")
    print(f"  max_k={max_k}")
    
    all_svd_results = {}
    
    for k in k_values:
        print(f"\n  对W_U^T做SVD, k={k}...")
        try:
            U_raw, s_raw, Vt_raw = svds(W_U_T, k=k)
            s = np.abs(np.asarray(s_raw, dtype=np.float64).ravel())
            sort_idx = np.argsort(-s)
            s = s[sort_idx]
            U_sorted = np.asarray(U_raw, dtype=np.float64)[:, sort_idx]
            
            # 基本统计
            total_energy = np.sum(s ** 2)
            PR = (np.sum(s) ** 2) / max(total_energy, 1e-20)
            
            # 累积能量
            cum_energy = np.cumsum(s ** 2) / total_energy
            k_50 = int(np.searchsorted(cum_energy, 0.50)) + 1
            k_90 = int(np.searchsorted(cum_energy, 0.90)) + 1
            k_95 = int(np.searchsorted(cum_energy, 0.95)) + 1
            k_99 = int(np.searchsorted(cum_energy, 0.99)) + 1
            
            svd_info = {
                "k": k,
                "PR": float(PR),
                "PR_over_d": float(PR / d_model),
                "total_energy": float(total_energy),
                "singular_values_top20": s[:20].tolist(),
                "singular_values_tail20": s[-20:].tolist() if len(s) >= 20 else s.tolist(),
                "singular_values_sample": s[::max(1, len(s)//50)].tolist(),  # 50个采样点
                "cum_energy_at_50pct": k_50,
                "cum_energy_at_90pct": k_90,
                "cum_energy_at_95pct": k_95,
                "cum_energy_at_99pct": k_99,
                "s_mean": float(np.mean(s)),
                "s_std": float(np.std(s)),
                "s_max": float(np.max(s)),
                "s_min": float(np.min(s)),
            }
            
            all_svd_results[str(k)] = svd_info
            
            print(f"    PR={PR:.1f}, PR/d={PR/d_model:.4f}")
            print(f"    累积能量: 50%@k={k_50}, 90%@k={k_90}, 95%@k={k_95}, 99%@k={k_99}")
            print(f"    奇异值范围: [{np.min(s):.2f}, {np.max(s):.2f}], mean={np.mean(s):.2f}")
            
            # 保存最大k的U矩阵用于后续实验
            if k == k_values[-1]:
                U_wut_maxk = U_sorted
                s_maxk = s
            
            del U_raw, Vt_raw, U_sorted
            gc.collect()
            
        except Exception as e:
            print(f"    SVD k={k} failed: {e}")
            all_svd_results[str(k)] = {"error": str(e)}
    
    del W_U_T; gc.collect()
    
    # Step 2: Marchenko-Pastur理论计算
    print(f"\n  Step 2: Marchenko-Pastur理论...")
    
    # MP分布参数: X是m×n, 元素i.i.d. N(0, σ²/m)
    # W_U是 vocab × d_model, 所以 m=vocab, n=d_model
    # 但W_U不是i.i.d.的，我们用等价参数
    gamma_mp = d_model / vocab_size  # n/m ratio
    
    # 对于等价的随机矩阵，σ²需要从数据估计
    # 用最大k的SVD结果估计σ²
    if str(k_values[-1]) in all_svd_results and "error" not in all_svd_results[str(k_values[-1])]:
        # MP的σ²估计: 均值的平方(因为MP分布的均值≈σ²)
        # 但更准确: 用Frobenius范数 / (m*n)
        # ||W_U||_F² / (vocab * d_model) ≈ σ²
        # ||W_U||_F² = Σ σ_i² = total_energy
        # 但我们只有k个奇异值，total_energy是下界
        
        # 用奇异值的中位数估计bulk的σ²
        s_bulk = s_maxk[len(s_maxk)//4: 3*len(s_maxk)//4]  # 中间50%
        sigma2_est = float(np.median(s_bulk ** 2))
        
        # MP边界 (用奇异值的平方=本征值)
        lambda_min_mp = sigma2_est * (1 - np.sqrt(gamma_mp)) ** 2
        lambda_max_mp = sigma2_est * (1 + np.sqrt(gamma_mp)) ** 2
        
        mp_info = {
            "gamma": float(gamma_mp),
            "sigma2_estimate": float(sigma2_est),
            "lambda_min_mp": float(lambda_min_mp),
            "lambda_max_mp": float(lambda_max_mp),
            "sv_min_mp": float(np.sqrt(max(lambda_min_mp, 0))),
            "sv_max_mp": float(np.sqrt(max(lambda_max_mp, 0))),
        }
        
        print(f"    γ=n/m={gamma_mp:.6f}")
        print(f"    sigma2_est={sigma2_est:.2f}")
        print(f"    MP奇异值范围: [{np.sqrt(max(lambda_min_mp,0)):.2f}, {np.sqrt(max(lambda_max_mp,0)):.2f}]")
        
        # Step 3: 谱偏离分析
        print(f"\n  Step 3: 谱偏离分析...")
        
        # 统计超出MP上界的奇异值
        sv_min_mp = np.sqrt(max(lambda_min_mp, 0))
        sv_max_mp = np.sqrt(max(lambda_max_mp, 0))
        
        n_above = int(np.sum(s_maxk > sv_max_mp))
        n_below = int(np.sum(s_maxk < sv_min_mp))
        n_in_bulk = len(s_maxk) - n_above - n_below
        
        # 超出MP上界的奇异值能量
        outlier_svs = s_maxk[s_maxk > sv_max_mp]
        outlier_energy = float(np.sum(outlier_svs ** 2))
        bulk_energy = float(np.sum(s_maxk[s_maxk <= sv_max_mp] ** 2))
        total_energy_est = outlier_energy + bulk_energy
        
        outlier_fraction = outlier_energy / max(total_energy_est, 1e-20)
        
        print(f"    MP范围内: {n_in_bulk}个, 超出上界: {n_above}个, 低于下界: {n_below}个")
        print(f"    Outlier能量占比: {outlier_fraction:.4f} ({outlier_fraction*100:.1f}%)")
        
        # 前N个outlier的奇异值
        n_top = min(30, len(outlier_svs))
        mp_deviation = {
            "n_above_mp": n_above,
            "n_below_mp": n_below,
            "n_in_bulk": n_in_bulk,
            "outlier_energy_fraction": float(outlier_fraction),
            "outlier_top_svs": outlier_svs[:n_top].tolist(),
            "mp_boundary_sv": float(sv_max_mp),
        }
        
        # Step 4: 投影能量在不同奇异值区间的分布
        print(f"\n  Step 4: 投影能量分布分析...")
        
        # 用最大k的U_wut测量真实信号的投影能量分布
        # 取一个属性方向做测试
        W_U = get_W_U(model)
        direction, _ = get_attr_direction(model, tokenizer, "red", W_U=W_U)
        del W_U; gc.collect()
        
        if direction is not None:
            inputs_base, inputs_interv, _, pos_ids = inject_at_embed(
                model, tokenizer, device, PROMPT, direction, BETA
            )
            n_layers = model_info.n_layers
            base_out = collect_layer_outputs(model, inputs_base, pos_ids, n_layers)
            interv_out = collect_layer_outputs(model, inputs_interv, pos_ids, n_layers)
            
            mid_layer = n_layers // 2
            key = f"L{mid_layer}"
            if key in base_out and key in interv_out:
                h_base = base_out[key][0, -1, :].numpy()
                h_interv = interv_out[key][0, -1, :].numpy()
                delta_h = h_interv - h_base
                
                # 投影到U_wut的各列
                proj_coeffs = U_wut_maxk.T @ delta_h  # [k]
                proj_energy_per_sv = proj_coeffs ** 2  # 每个奇异值方向的能量
                
                # 分区统计: outlier区间 vs bulk区间
                is_outlier = s_maxk > sv_max_mp
                
                energy_in_outlier = float(np.sum(proj_energy_per_sv[is_outlier]))
                energy_in_bulk = float(np.sum(proj_energy_per_sv[~is_outlier]))
                total_proj_energy = energy_in_outlier + energy_in_bulk
                
                delta_norm_sq = float(np.sum(delta_h ** 2))
                
                proj_dist = {
                    "layer": mid_layer,
                    "delta_norm": float(np.linalg.norm(delta_h)),
                    "total_proj_energy": float(total_proj_energy),
                    "total_ratio": float(total_proj_energy / max(delta_norm_sq, 1e-20)),
                    "energy_in_outlier_sv": float(energy_in_outlier),
                    "energy_in_bulk_sv": float(energy_in_bulk),
                    "outlier_sv_energy_fraction": float(energy_in_outlier / max(total_proj_energy, 1e-20)),
                    "n_outlier_svs": int(np.sum(is_outlier)),
                    "n_bulk_svs": int(np.sum(~is_outlier)),
                }
                
                print(f"    Δh在L{mid_layer}: ||Δ||={np.linalg.norm(delta_h):.4f}")
                print(f"    投影比: {total_proj_energy/max(delta_norm_sq,1e-20):.4f}")
                print(f"    Outlier方向能量占比: {energy_in_outlier/max(total_proj_energy,1e-20):.4f}")
                print(f"    Bulk方向能量占比: {energy_in_bulk/max(total_proj_energy,1e-20):.4f}")
                
                # 进一步: 按奇异值大小排序的投影能量分布
                # 把奇异值分成10个区间，每个区间的平均投影能量
                n_bins = 10
                bin_size = len(s_maxk) // n_bins
                bin_stats = []
                for bi in range(n_bins):
                    start = bi * bin_size
                    end = min((bi + 1) * bin_size, len(s_maxk))
                    bin_energy = float(np.sum(proj_energy_per_sv[start:end]))
                    bin_sv_mean = float(np.mean(s_maxk[start:end]))
                    bin_stats.append({
                        "bin": bi,
                        "sv_range": [start, end],
                        "sv_mean": bin_sv_mean,
                        "proj_energy": bin_energy,
                        "energy_per_sv": bin_energy / max(end - start, 1),
                    })
                
                proj_dist["bin_stats"] = bin_stats
                mp_deviation["proj_distribution"] = proj_dist
    else:
        mp_info = {"error": "SVD failed for max k"}
        mp_deviation = {"error": "No SVD results"}
    
    results["svd_multiscale"] = all_svd_results
    results["marchenko_pastur"] = mp_info
    results["mp_deviation"] = mp_deviation
    
    # Step 5: 2×因子判别 — 不同k下的PR/d vs 实测ratio
    print(f"\n  Step 5: 2×因子判别...")
    
    pr_d_values = {}
    for k_str, svd_info in all_svd_results.items():
        if "error" not in svd_info:
            pr_d_values[k_str] = svd_info["PR_over_d"]
    
    # 如果有实测ratio (从P443/P444的结果), 对比
    # 这里我们重新测量一次(简单版)
    print(f"  PR/d随k变化:")
    for k_str, pr_d in sorted(pr_d_values.items(), key=lambda x: int(x[0])):
        print(f"    k={k_str}: PR/d={pr_d:.4f}")
    
    results["pr_d_vs_k"] = pr_d_values
    
    return results


# ================================================================
# P448: 本征值聚焦效应
# ================================================================

def run_p448(model, tokenizer, device, model_info):
    """
    P448: 本征值聚焦效应 — 投影能量的加权分析
    
    原理:
      如果W_U行空间中，投影能量集中在少数大奇异值方向，
      则"有效覆盖维度" < PR，实测ratio > PR/d。
      
      定义"聚焦因子":
        F = (Σ w_i · e_i) / (Σ e_i)
      其中 w_i = σ_i² / (Σ σ_j²) 是第i个奇异值的权重,
            e_i 是信号在第i个方向的投影能量占比
      
      如果F > 1 → 能量集中在大奇异值方向 → 聚焦效应 → 解释2×因子
      如果F ≈ 1 → 能量均匀分布 → 无聚焦效应 → 需要其他解释
    
    方法:
      1. 对W_U做SVD获取奇异值和基向量
      2. 对多个属性的delta_h，计算每个奇异值方向的投影能量
      3. 分析投影能量是否与奇异值大小相关
      4. 计算聚焦因子
    """
    print(f"\n{'='*60}")
    print(f"P448: 本征值聚焦效应 - {model_info.name}")
    print(f"{'='*60}")
    
    results = {}
    d_model = model_info.d_model
    n_layers = model_info.n_layers
    
    # W_U SVD
    W_U = get_W_U(model)
    vocab_size = W_U.shape[0]
    
    # 预计算属性方向
    attr_directions = {}
    for attr in TEST_ATTRS:
        direction, _ = get_attr_direction(model, tokenizer, attr, W_U=W_U)
        if direction is not None:
            attr_directions[attr] = direction
    
    # W_U^T SVD
    W_U_T = W_U.T.astype(np.float32)
    del W_U; import gc; gc.collect()
    
    max_k = min(W_U_T.shape[0], W_U_T.shape[1]) - 2
    k_svd = min(500, max_k)  # 用较大k获得更好的分辨率
    print(f"  对W_U^T做SVD, k={k_svd}...")
    
    U_raw, s_raw, _ = svds(W_U_T, k=k_svd)
    del W_U_T; gc.collect()
    
    s = np.abs(np.asarray(s_raw, dtype=np.float64).ravel())
    sort_idx = np.argsort(-s)
    s = s[sort_idx]
    U_wut = np.asarray(U_raw, dtype=np.float64)[:, sort_idx]
    del U_raw, s_raw; gc.collect()
    
    total_energy = np.sum(s ** 2)
    sv_weights = s ** 2 / total_energy  # 每个奇异值的能量权重
    
    print(f"  PR={float((np.sum(s)**2)/total_energy):.1f}, top5 SVs={s[:5].tolist()}")
    
    # 对每个属性×层，计算投影能量分布
    print(f"  计算投影能量分布...")
    
    sample_layers = get_sample_layers(n_layers, n_samples=8)
    
    all_focus_data = []
    
    for attr in list(attr_directions.keys())[:4]:  # 取4个属性减少计算量
        direction = attr_directions[attr]
        
        inputs_base, inputs_interv, _, pos_ids = inject_at_embed(
            model, tokenizer, device, PROMPT, direction, BETA
        )
        
        base_out = collect_layer_outputs(model, inputs_base, pos_ids, n_layers)
        interv_out = collect_layer_outputs(model, inputs_interv, pos_ids, n_layers)
        
        for li in sample_layers:
            key = f"L{li}"
            if key not in base_out or key not in interv_out:
                continue
            
            h_base = base_out[key][0, -1, :].numpy()
            h_interv = interv_out[key][0, -1, :].numpy()
            delta_h = h_interv - h_base
            
            # 投影系数
            proj_coeffs = U_wut.T @ delta_h  # [k]
            proj_energy_per_sv = proj_coeffs ** 2  # 每个方向的能量
            total_proj = np.sum(proj_energy_per_sv)
            
            if total_proj < 1e-10:
                continue
            
            proj_energy_frac = proj_energy_per_sv / total_proj  # 归一化
            
            # 聚焦因子: 投影能量与奇异值权重的相关性
            # 如果投影能量集中在大奇异值方向 → corr(sv_weights, proj_energy_frac) > 0
            corr_sv_proj = float(np.corrcoef(sv_weights, proj_energy_frac)[0, 1])
            
            # 加权平均: 信号"看到"的有效维度
            # 如果能量均匀分布: effective_k = total_proj² / Σ(proj_i²) × Σ(proj_i²)/total_proj² = ???
            # 更好的指标: 投影能量的Participation Ratio
            proj_PR = float(np.sum(proj_energy_per_sv) ** 2 / max(np.sum(proj_energy_per_sv ** 2), 1e-20))
            
            # 对比: 奇异值权重的PR
            sv_PR = float(np.sum(sv_weights) ** 2 / max(np.sum(sv_weights ** 2), 1e-20))
            
            # 聚焦比: 如果信号集中在少数大奇异值方向，proj_PR < sv_PR
            focus_ratio = proj_PR / max(sv_PR, 1e-20)
            
            # Top-k投影能量占比
            sorted_proj = np.sort(proj_energy_frac)[::-1]
            top10_frac = float(np.sum(sorted_proj[:10]))
            top50_frac = float(np.sum(sorted_proj[:50]))
            top100_frac = float(np.sum(sorted_proj[:100]))
            
            # recoding_ratio
            delta_norm_sq = float(np.sum(delta_h ** 2))
            ratio = total_proj / max(delta_norm_sq, 1e-20)
            
            focus_entry = {
                "attr": attr,
                "layer": li,
                "ratio": float(ratio),
                "corr_sv_proj": corr_sv_proj,
                "proj_PR": proj_PR,
                "sv_PR": sv_PR,
                "focus_ratio": focus_ratio,
                "top10_frac": top10_frac,
                "top50_frac": top50_frac,
                "top100_frac": top100_frac,
            }
            
            all_focus_data.append(focus_entry)
    
    # 汇总
    if all_focus_data:
        mean_corr = np.mean([d["corr_sv_proj"] for d in all_focus_data])
        mean_focus = np.mean([d["focus_ratio"] for d in all_focus_data])
        mean_top10 = np.mean([d["top10_frac"] for d in all_focus_data])
        mean_top50 = np.mean([d["top50_frac"] for d in all_focus_data])
        
        print(f"\n  聚焦效应汇总 (n={len(all_focus_data)}):")
        print(f"    SV权重 vs 投影能量 相关: r={mean_corr:.4f}")
        print(f"    聚焦比(proj_PR/sv_PR): {mean_focus:.4f}")
        print(f"    Top10方向投影能量: {mean_top10:.4f}")
        print(f"    Top50方向投影能量: {mean_top50:.4f}")
        
        results["summary"] = {
            "mean_corr_sv_proj": float(mean_corr),
            "mean_focus_ratio": float(mean_focus),
            "mean_top10_frac": float(mean_top10),
            "mean_top50_frac": float(mean_top50),
        }
    
    results["focus_data"] = all_focus_data
    
    return results


# ================================================================
# P449: gain/||Δ||比值的层间分析 — 伪相关检验
# ================================================================

def run_p449(model, tokenizer, device, model_info):
    """
    P449: gain/||Δ||比值的层间分析
    
    原理:
      P445发现 delta_norm vs gain 的 r=0.996，但这可能是伪相关:
      - gain = ||U^T · Δ|| = sqrt(Σ (u_i^T · Δ)²)
      - 如果Δ的各分量在U^T各列上的投影是随机的，
        则 E[gain²] = Σ E[(u_i^T · Δ)²] = Σ ||u_i||² · ||Δ||²/d ≈ k · ||Δ||²/d
        即 gain ≈ sqrt(k/d) · ||Δ||
      - 所以 gain ∝ ||Δ|| 是平凡的!
      
      验证方法: 计算gain/||Δ||, 看是否为常数
      - 如果是常数 → 相关是平凡的(几何约束)
      - 如果不是常数 → 有额外结构
    
    方法:
      1. 对多个属性×层，计算 gain, ||Δ||, gain/||Δ||
      2. 分析gain/||Δ||的层间变化
      3. 与理论预测 sqrt(PR/d_model) 对比
    """
    print(f"\n{'='*60}")
    print(f"P449: gain/||Δ||伪相关检验 - {model_info.name}")
    print(f"{'='*60}")
    
    results = {}
    d_model = model_info.d_model
    n_layers = model_info.n_layers
    
    # W_U SVD
    W_U = get_W_U(model)
    
    attr_directions = {}
    for attr in TEST_ATTRS:
        direction, _ = get_attr_direction(model, tokenizer, attr, W_U=W_U)
        if direction is not None:
            attr_directions[attr] = direction
    
    W_U_T = W_U.T.astype(np.float32)
    del W_U; import gc; gc.collect()
    
    max_k = min(W_U_T.shape[0], W_U_T.shape[1]) - 2
    k_svd = min(300, max_k)
    U_raw, s_raw, _ = svds(W_U_T, k=k_svd)
    del W_U_T; gc.collect()
    
    s = np.abs(np.asarray(s_raw, dtype=np.float64).ravel())
    sort_idx = np.argsort(-s)
    s = s[sort_idx]
    U_wut = np.asarray(U_raw, dtype=np.float64)[:, sort_idx]
    del U_raw, s_raw; gc.collect()
    
    total_energy = np.sum(s ** 2)
    PR = float((np.sum(s) ** 2) / total_energy)
    theoretical_gain_per_norm = np.sqrt(k_svd / d_model)  # 理论: sqrt(k/d)
    theoretical_gain_per_norm_pr = np.sqrt(PR / d_model)  # 理论: sqrt(PR/d)
    
    print(f"  PR={PR:.1f}, k_svd={k_svd}")
    print(f"  理论gain/||Δ||: sqrt(k/d)={theoretical_gain_per_norm:.4f}, sqrt(PR/d)={theoretical_gain_per_norm_pr:.4f}")
    
    # 全层测量
    sample_layers = get_sample_layers(n_layers, n_samples=12)
    
    all_data = []
    
    for attr in list(attr_directions.keys())[:4]:
        direction = attr_directions[attr]
        
        inputs_base, inputs_interv, _, pos_ids = inject_at_embed(
            model, tokenizer, device, PROMPT, direction, BETA
        )
        
        base_out = collect_layer_outputs(model, inputs_base, pos_ids, n_layers)
        interv_out = collect_layer_outputs(model, inputs_interv, pos_ids, n_layers)
        
        for li in sample_layers:
            key = f"L{li}"
            if key not in base_out or key not in interv_out:
                continue
            
            h_base = base_out[key][0, -1, :].numpy()
            h_interv = interv_out[key][0, -1, :].numpy()
            delta_h = h_interv - h_base
            
            rc = compute_recoding_ratio_cached(delta_h, U_wut)
            gain = rc["gain"]
            delta_norm = np.linalg.norm(delta_h)
            ratio = rc["ratio"]
            
            gain_per_norm = gain / max(delta_norm, 1e-10)
            sqrt_ratio = np.sqrt(max(ratio, 0))  # sqrt(ratio) 理论上 ≈ gain/||Δ||
            
            all_data.append({
                "attr": attr,
                "layer": li,
                "gain": float(gain),
                "delta_norm": float(delta_norm),
                "ratio": float(ratio),
                "gain_per_norm": float(gain_per_norm),
                "sqrt_ratio": float(sqrt_ratio),
            })
    
    # 分析
    if all_data:
        gains = [d["gain"] for d in all_data]
        delta_norms = [d["delta_norm"] for d in all_data]
        gain_per_norms = [d["gain_per_norm"] for d in all_data]
        sqrt_ratios = [d["sqrt_ratio"] for d in all_data]
        
        # gain vs ||Δ|| 相关
        corr_gain_delta = np.corrcoef(gains, delta_norms)[0, 1]
        
        # gain/||Δ|| 的变异系数
        cv_gain_per_norm = np.std(gain_per_norms) / max(np.mean(gain_per_norms), 1e-10)
        
        # gain/||Δ|| 与 sqrt(ratio) 的相关 (应该接近1)
        corr_gpn_sqr = np.corrcoef(gain_per_norms, sqrt_ratios)[0, 1]
        
        # gain/||Δ|| 的均值 vs 理论值
        mean_gpn = np.mean(gain_per_norms)
        
        print(f"\n  伪相关检验结果 (n={len(all_data)}):")
        print(f"    gain vs ||Δ|| 相关: r={corr_gain_delta:.4f}")
        print(f"    gain/||Δ|| 变异系数: {cv_gain_per_norm:.4f} ({cv_gain_per_norm*100:.1f}%)")
        print(f"    gain/||Δ|| 均值: {mean_gpn:.4f}")
        print(f"    理论 sqrt(k/d): {theoretical_gain_per_norm:.4f}")
        print(f"    理论 sqrt(PR/d): {theoretical_gain_per_norm_pr:.4f}")
        print(f"    实测/理论(PR/d): {mean_gpn/theoretical_gain_per_norm_pr:.2f}x")
        print(f"    gain/||Δ|| vs sqrt(ratio): r={corr_gpn_sqr:.4f}")
        
        # 判断
        if cv_gain_per_norm < 0.1:
            verdict = "伪相关: gain/||Δ||近似常数, gain∝||Δ||是平凡关系"
        elif cv_gain_per_norm < 0.3:
            verdict = "弱结构: gain/||Δ||有中等变异, 存在额外结构"
        else:
            verdict = "强结构: gain/||Δ||变化显著, 有深层机制"
        
        print(f"    结论: {verdict}")
        
        # 层间趋势
        layer_gpn = {}
        for d in all_data:
            li = d["layer"]
            if li not in layer_gpn:
                layer_gpn[li] = []
            layer_gpn[li].append(d["gain_per_norm"])
        
        layer_trend = []
        for li in sorted(layer_gpn.keys()):
            mean_val = np.mean(layer_gpn[li])
            layer_trend.append({"layer": li, "mean_gain_per_norm": float(mean_val)})
        
        results["summary"] = {
            "corr_gain_delta": float(corr_gain_delta),
            "cv_gain_per_norm": float(cv_gain_per_norm),
            "mean_gain_per_norm": float(mean_gpn),
            "theoretical_sqrt_k_d": float(theoretical_gain_per_norm),
            "theoretical_sqrt_PR_d": float(theoretical_gain_per_norm_pr),
            "measured_over_theory": float(mean_gpn / theoretical_gain_per_norm_pr),
            "verdict": verdict,
        }
        results["layer_trend"] = layer_trend
    
    results["detail_data"] = all_data
    
    return results


# ================================================================
# P450: 逃逸能量全谱SVD结构分析
# ================================================================

def run_p450(model, tokenizer, device, model_info):
    """
    P450: 逃逸能量全谱SVD结构分析
    
    原理:
      recoding_ratio ≈ 0.22 意味着 78% 的信号能量逃逸了W_U行空间。
      这些逃逸能量是纯噪声还是有结构?
      
      如果有结构 → 逃逸能量携带信息，需要"非线性纠错码"理论
      如果纯噪声 → 信号传播确实是"各向同性放大×随机旋转"
    
    方法:
      1. 计算delta在W_U行空间的投影 proj = U·(U^T·Δ)
      2. 计算逃逸分量 escape = Δ - proj
      3. 对多个Δ的逃逸分量收集
      4. 做SVD分析逃逸分量的结构
    """
    print(f"\n{'='*60}")
    print(f"P450: 逃逸能量结构分析 - {model_info.name}")
    print(f"{'='*60}")
    
    results = {}
    d_model = model_info.d_model
    n_layers = model_info.n_layers
    
    # W_U SVD
    W_U = get_W_U(model)
    
    attr_directions = {}
    for attr in TEST_ATTRS:
        direction, _ = get_attr_direction(model, tokenizer, attr, W_U=W_U)
        if direction is not None:
            attr_directions[attr] = direction
    
    W_U_T = W_U.T.astype(np.float32)
    del W_U; import gc; gc.collect()
    
    max_k = min(W_U_T.shape[0], W_U_T.shape[1]) - 2
    k_svd = min(300, max_k)
    U_raw, s_raw, _ = svds(W_U_T, k=k_svd)
    del W_U_T; gc.collect()
    
    s = np.abs(np.asarray(s_raw, dtype=np.float64).ravel())
    sort_idx = np.argsort(-s)
    U_wut = np.asarray(U_raw, dtype=np.float64)[:, sort_idx]
    del U_raw, s_raw; gc.collect()
    
    # 收集多个逃逸向量
    escape_vectors = []
    
    for attr in list(attr_directions.keys())[:4]:
        direction = attr_directions[attr]
        
        inputs_base, inputs_interv, _, pos_ids = inject_at_embed(
            model, tokenizer, device, PROMPT, direction, BETA
        )
        
        base_out = collect_layer_outputs(model, inputs_base, pos_ids, n_layers)
        interv_out = collect_layer_outputs(model, inputs_interv, pos_ids, n_layers)
        
        for li in get_sample_layers(n_layers, n_samples=6):
            key = f"L{li}"
            if key not in base_out or key not in interv_out:
                continue
            
            h_base = base_out[key][0, -1, :].numpy()
            h_interv = interv_out[key][0, -1, :].numpy()
            delta_h = h_interv - h_base
            
            # 投影到W_U行空间
            proj_coeffs = U_wut.T @ delta_h  # [k]
            proj = U_wut @ proj_coeffs  # [d_model] 投影
            
            # 逃逸分量
            escape = delta_h - proj
            
            delta_norm = np.linalg.norm(delta_h)
            proj_norm = np.linalg.norm(proj)
            escape_norm = np.linalg.norm(escape)
            
            escape_vectors.append({
                "attr": attr,
                "layer": li,
                "escape": escape,
                "delta_norm": float(delta_norm),
                "proj_norm": float(proj_norm),
                "escape_norm": float(escape_norm),
                "escape_ratio": float(escape_norm ** 2 / max(delta_norm ** 2, 1e-20)),
            })
    
    print(f"  收集了 {len(escape_vectors)} 个逃逸向量")
    
    if len(escape_vectors) < 3:
        results["error"] = "Too few escape vectors"
        return results
    
    # 逃逸向量的SVD分析
    # 构造矩阵: 每列是一个逃逸向量
    escape_matrix = np.column_stack([e["escape"] for e in escape_vectors])  # [d_model, n_vectors]
    
    print(f"  逃逸矩阵 shape: {escape_matrix.shape}")
    
    # SVD
    n_vecs = escape_matrix.shape[1]
    k_escape = min(n_vecs - 1, 50)  # 最多50个分量
    if k_escape < 1:
        k_escape = 1
    
    try:
        U_esc, s_esc, Vt_esc = np.linalg.svd(escape_matrix, full_matrices=False)
        # s_esc 已经降序排列
        
        total_esc_energy = np.sum(s_esc ** 2)
        esc_PR = float(np.sum(s_esc) ** 2 / max(total_esc_energy, 1e-20))
        
        # 累积能量
        cum_esc = np.cumsum(s_esc ** 2) / total_esc_energy
        
        print(f"  逃逸SVD: PR={esc_PR:.1f}")
        print(f"  前5个奇异值: {s_esc[:5].tolist()}")
        print(f"  累积能量: 50%@k={int(np.searchsorted(cum_esc,0.50))+1}, 90%@k={int(np.searchsorted(cum_esc,0.90))+1}")
        
        # 与随机基线对比: 随机向量的SVD PR应该≈n_vectors
        # 如果逃逸向量的PR << n_vectors → 有强结构(少数方向主导)
        # 如果PR ≈ n_vectors → 无结构(各方向均匀)
        
        n_random = 100
        random_prs = []
        np.random.seed(42)
        for _ in range(n_random):
            # 生成与逃逸向量同分布的随机向量
            rand_vecs = np.random.randn(d_model, n_vecs).astype(np.float32)
            # 归一化到与逃逸向量相同的平均范数
            mean_esc_norm = np.mean([e["escape_norm"] for e in escape_vectors])
            rand_norms = np.linalg.norm(rand_vecs, axis=0, keepdims=True)
            rand_vecs = rand_vecs * (mean_esc_norm / rand_norms)
            
            _, s_rand, _ = np.linalg.svd(rand_vecs, full_matrices=False)
            total_rand = np.sum(s_rand ** 2)
            rand_PR = float(np.sum(s_rand) ** 2 / max(total_rand, 1e-20))
            random_prs.append(rand_PR)
        
        mean_random_pr = np.mean(random_prs)
        std_random_pr = np.std(random_prs)
        
        # 逃逸向量之间的余弦相似度
        cos_matrix = np.abs(escape_matrix.T @ escape_matrix)
        # 归一化
        norms = np.linalg.norm(escape_matrix, axis=0, keepdims=True)
        cos_matrix_norm = cos_matrix / (norms.T @ norms + 1e-10)
        # 非对角线均值
        n_off = 0
        off_diag_sum = 0
        for i in range(n_vecs):
            for j in range(i+1, n_vecs):
                off_diag_sum += cos_matrix_norm[i, j]
                n_off += 1
        mean_cos_escape = off_diag_sum / max(n_off, 1)
        
        # 逃逸分量与W_U行空间的对齐度
        # 如果逃逸分量与W_U正交(理论上应该如此)，cos应该≈0
        escape_wu_cos = []
        for e in escape_vectors:
            cos_val = float(np.abs(U_wut.T @ e["escape"]).sum() / 
                          max(np.linalg.norm(e["escape"]) * np.sqrt(k_svd), 1e-10))
            escape_wu_cos.append(cos_val)
        mean_esc_wu_cos = np.mean(escape_wu_cos)
        
        esc_pr_ratio = esc_PR / max(mean_random_pr, 1e-10)
        
        print(f"\n  逃逸结构分析:")
        print(f"    逃逸PR={esc_PR:.1f} vs 随机PR={mean_random_pr:.1f}±{std_random_pr:.1f}")
        print(f"    PR比(逃逸/随机): {esc_pr_ratio:.2f}")
        print(f"    逃逸向量间平均|cos|: {mean_cos_escape:.4f}")
        print(f"    逃逸与W_U对齐度: {mean_esc_wu_cos:.4f}")
        
        if esc_pr_ratio < 0.5:
            structure_verdict = "强结构: 逃逸能量集中在少数方向，携带信息"
        elif esc_pr_ratio < 0.8:
            structure_verdict = "中等结构: 逃逸能量有部分集中"
        else:
            structure_verdict = "弱结构/无结构: 逃逸能量接近随机分布"
        
        print(f"    结论: {structure_verdict}")
        
        results["escape_svd"] = {
            "PR": esc_PR,
            "singular_values": s_esc.tolist(),
            "cum_energy_50pct_k": int(np.searchsorted(cum_esc, 0.50)) + 1,
            "cum_energy_90pct_k": int(np.searchsorted(cum_esc, 0.90)) + 1,
        }
        results["random_baseline"] = {
            "mean_PR": float(mean_random_pr),
            "std_PR": float(std_random_pr),
        }
        results["escape_cos_analysis"] = {
            "mean_cos_between_escapes": float(mean_cos_escape),
            "mean_cos_escape_wu": float(mean_esc_wu_cos),
        }
        results["structure_verdict"] = structure_verdict
        results["escape_ratio_mean"] = float(np.mean([e["escape_ratio"] for e in escape_vectors]))
        
    except Exception as e:
        print(f"  逃逸SVD失败: {e}")
        results["escape_svd_error"] = str(e)
    
    # 清理
    for e in escape_vectors:
        del e["escape"]
    
    return results


# ================================================================
# P451: 多尺度PR分析 (关键判别实验)
# ================================================================

def run_p451(model, tokenizer, device, model_info):
    """
    P451: 多尺度PR分析 — 2×因子是截断伪影还是真实效应?
    
    原理:
      当前所有SVD分析都用k=300截断，可能严重低估PR。
      如果PR随k快速增长(k>300后仍在增长):
      → k=300截断严重低估PR → 2×因子是数学伪影
      → 真实PR/d_model ≈ 实测ratio
      
      如果PR在k≈300已收敛:
      → 截断不是问题 → 2×因子是真实的 → 需要新理论
    
    方法:
      1. 对W_U^T做不同k值的SVD: k = 10, 20, 50, 100, 200, 300, 500, 800, 1200, max
      2. 每个k下计算PR
      3. 画PR(k)曲线，看是否收敛
      4. 同时计算每个k下的recoding_ratio(理论预测=PR(k)/d)
      5. 与实测ratio对比
    """
    print(f"\n{'='*60}")
    print(f"P451: 多尺度PR分析 - {model_info.name}")
    print(f"{'='*60}")
    
    results = {}
    d_model = model_info.d_model
    
    W_U = get_W_U(model)
    vocab_size = W_U.shape[0]
    wu_shape = list(W_U.shape)
    
    print(f"  W_U shape: {wu_shape}")
    
    W_U_T = W_U.T.astype(np.float32)
    del W_U; import gc; gc.collect()
    
    max_k = min(W_U_T.shape[0], W_U_T.shape[1]) - 2
    
    # 多个k值
    k_values = [10, 20, 50, 100, 200, 300, 500, 800, 1200]
    if max_k > 1200:
        k_values.append(max_k)
    k_values = [k for k in k_values if k <= max_k]
    k_values = sorted(set(k_values))
    
    print(f"  k值: {k_values}")
    
    pr_vs_k = {}
    # 为了效率，我们做一次大k的SVD，然后逐步截断
    # 但scipy的svds不能保证返回所有分量
    # 替代方案: 多次svds调用，但大k时可能不稳定
    
    # 策略: 用numpy的svd对W_U^T做完整SVD(如果内存允许)
    # W_U^T shape=[d_model, vocab], d_model=2560/3584/4096
    # 完整SVD需要存储 U[d,vocab] — 太大!
    # 所以还是用多次svds
    
    for k in k_values:
        print(f"  SVD k={k}...", end=" ")
        try:
            U_raw, s_raw, _ = svds(W_U_T, k=k)
            s = np.abs(np.asarray(s_raw, dtype=np.float64).ravel())
            sort_idx = np.argsort(-s)
            s = s[sort_idx]
            
            total_energy = np.sum(s ** 2)
            PR = float((np.sum(s) ** 2) / total_energy)
            
            # 累积能量
            cum_energy = np.cumsum(s ** 2) / total_energy
            k_90 = int(np.searchsorted(cum_energy, 0.90)) + 1
            k_95 = int(np.searchsorted(cum_energy, 0.95)) + 1
            k_99 = int(np.searchsorted(cum_energy, 0.99)) + 1
            
            pr_vs_k[str(k)] = {
                "k": k,
                "PR": PR,
                "PR_over_d": PR / d_model,
                "total_energy": float(total_energy),
                "k_90": k_90,
                "k_95": k_95,
                "k_99": k_99,
                "s_top5": s[:5].tolist(),
                "s_tail5": s[-5:].tolist() if len(s) >= 5 else s.tolist(),
            }
            
            print(f"PR={PR:.1f}, PR/d={PR/d_model:.4f}, k_90={k_90}, k_95={k_95}")
            
            del U_raw, s_raw
            
        except Exception as e:
            print(f"failed: {e}")
            pr_vs_k[str(k)] = {"k": k, "error": str(e)}
        
        import gc; gc.collect()
    
    del W_U_T; gc.collect()
    
    # PR收敛分析
    valid_ks = {int(k): v for k, v in pr_vs_k.items() if "error" not in v}
    
    if len(valid_ks) >= 3:
        # PR增长率: PR(k2) - PR(k1) / (k2 - k1)
        ks_sorted = sorted(valid_ks.keys())
        pr_growth = {}
        for i in range(1, len(ks_sorted)):
            k1, k2 = ks_sorted[i-1], ks_sorted[i]
            pr1 = valid_ks[k1]["PR"]
            pr2 = valid_ks[k2]["PR"]
            growth_rate = (pr2 - pr1) / (k2 - k1)
            pr_growth[f"{k1}->{k2}"] = float(growth_rate)
        
        # PR是否收敛: 看最后两个k值的增长率
        last_two = ks_sorted[-2:]
        pr_last = valid_ks[last_two[-1]]["PR"]
        pr_second_last = valid_ks[last_two[-2]]["PR"]
        pr_diff = pr_last - pr_second_last
        
        max_pr = max(v["PR"] for v in valid_ks.values())
        
        # 判断: 最后的增长率 < 1% → 收敛
        if pr_diff / max(max_pr, 1) < 0.01:
            convergence_verdict = "PR已收敛: 2×因子是真实的，需要新理论"
        elif pr_diff / max(max_pr, 1) < 0.05:
            convergence_verdict = "PR接近收敛: 2×因子基本真实，可能有5%截断偏差"
        elif pr_diff / max(max_pr, 1) < 0.15:
            convergence_verdict = "PR仍在增长: 2×因子可能部分是截断伪影"
        else:
            convergence_verdict = "PR显著增长: 2×因子很可能是截断伪影"
        
        print(f"\n  PR收敛分析:")
        print(f"    最大k={ks_sorted[-1]}: PR={valid_ks[ks_sorted[-1]]['PR']:.1f}")
        print(f"    最后增长: PR差={pr_diff:.1f}, 相对增长={pr_diff/max_pr*100:.1f}%")
        print(f"    结论: {convergence_verdict}")
        
        # 与实测ratio对比
        # 实测ratio ≈ 0.22 (Qwen3), 需要PR/d ≈ 0.22 → PR ≈ 563
        # 如果max k的PR << 563 → 截断严重
        # 如果max k的PR ≈ 563 → PR/d = ratio, 无2×因子
        
        max_k_pr = valid_ks[ks_sorted[-1]]["PR"]
        target_pr_for_ratio = {
            "qwen3": 0.22 * 2560,   # ≈ 563
            "glm4": 0.089 * 4096,   # ≈ 365
            "deepseek7b": 0.101 * 3584,  # ≈ 362
        }
        
        model_name = model_info.name
        if model_name in target_pr_for_ratio:
            target_pr = target_pr_for_ratio[model_name]
            print(f"\n  如果ratio=PR/d, 需要PR约={target_pr:.0f}")
            print(f"  当前最大PR={max_k_pr:.0f}, 差距={target_pr-max_k_pr:.0f}")
        
        results["convergence"] = {
            "pr_growth_rates": pr_growth,
            "pr_diff_last": float(pr_diff),
            "pr_diff_relative": float(pr_diff / max(max_pr, 1)),
            "max_pr": float(max_pr),
            "verdict": convergence_verdict,
        }
    
    results["pr_vs_k"] = pr_vs_k
    
    return results


# ================================================================
# 主流程
# ================================================================

def run_single_model(model_name, experiments=None):
    """运行单个模型的所有实验"""
    print(f"\n{'#'*60}")
    print(f"# 模型: {model_name}")
    print(f"{'#'*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"  类: {model_info.model_class}")
    print(f"  层数: {model_info.n_layers}, 维度: {model_info.d_model}")
    print(f"  MLP类型: {model_info.mlp_type}")
    
    all_results = {
        "model": model_name,
        "model_class": model_info.model_class,
        "n_layers": model_info.n_layers,
        "d_model": model_info.d_model,
        "mlp_type": model_info.mlp_type,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M")
    
    exp_map = {
        "p447": run_p447,
        "p448": run_p448,
        "p449": run_p449,
        "p450": run_p450,
        "p451": run_p451,
    }
    
    try:
        if experiments is None or "all" in experiments:
            exps_to_run = list(exp_map.items())
        else:
            exps_to_run = [(e, exp_map[e]) for e in experiments if e in exp_map]
        
        for exp_name, exp_fn in exps_to_run:
            print(f"\n--- Running {exp_name} ---")
            result = exp_fn(model, tokenizer, device, model_info)
            all_results[exp_name] = result
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    
    finally:
        out_file = OUT_DIR / f"phase_xcii_p447_451_{model_name}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResults saved to {out_file}")
        
        release_model(model)


def main():
    parser = argparse.ArgumentParser(description=f"{PHASE_NAME}: {EXPERIMENT_DESC}")
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, default="all",
                       choices=["all", "p447", "p448", "p449", "p450", "p451"])
    args = parser.parse_args()
    
    experiments = [args.experiment] if args.experiment != "all" else None
    run_single_model(args.model, experiments)


if __name__ == "__main__":
    main()
