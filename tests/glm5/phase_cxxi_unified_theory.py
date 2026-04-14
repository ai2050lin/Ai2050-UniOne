"""
Phase CXXI-CXXII: 流形动力学的统一理论
=========================================

Phase CXIX-CXX核心发现:
- P542流形假设: Qwen3 dim_90=140维; GLM4=87.5维; DS7B=37.8维(极低!)
- P543动力系统: DS7B中层旋转0.1-0.8deg(几乎不旋转!)
- P544吸引子: DS7B跨输入频谱相关0.999
- P546跨语言: 变异系数0.1-0.5%, 频谱相关0.94-1.00

Phase CXXI-CXXII核心思路 - 统一理论:
1. P547: 从dim_90推导ratio(k)的解析公式
   - 如果h的90%能量在dim_90维子空间中, 那么ratio(k)的理论值是什么?
   - 假设1: 均匀分布 → ratio(k) = k/dim_90
   - 假设2: 幂律分布 → ratio(k) = 1 - (1 - k/dim_90)^(1+alpha)
   - 假设3: 指数衰减 → ratio(k) = 1 - exp(-k/dim_90 * c)
   - 验证: 哪个假设最接近实测ratio(k)?

2. P548: 频谱的普适形状函数
   - P544发现末层频谱幂律指数: Qwen3=-0.65, GLM4=-1.52, DS7B=-1.31
   - 如果频谱密度S(i) = C * i^(-beta), 则ratio(k) = sum_{i=1}^k S(i) / sum S(i)
   - 推导ratio(k) = (zeta(beta, 1) - zeta(beta, k+1)) / zeta(beta, 1)
     其中zeta是Hurwitz zeta函数
   - 简化: ratio(k) ≈ 1 - (k/(k+1))^(1-beta) 当beta<1时
   - 验证: 幂律预测vs实测

3. P549: 吸引子的精确模型
   - LayerNorm: h_norm = (h - mean) / std * gamma
   - 残差: h_out = h_in + f_mlp(LN(h_in))
   - 不动点条件: 频谱形状不变
   - 假设: LN的归一化+gamma缩放使频谱向W_U top方向集中
   - 验证: gamma(即LN权重)的频谱与h频谱的关系

4. P550: 为什么DS7B比Qwen3更"稳定"?
   - DS7B dim_90=38 vs Qwen3=140, 旋转0.1deg vs 5-15deg
   - 可能原因: (1)R1蒸馏 (2)架构差异 (3)训练数据差异
   - 验证: 分析W_down的奇异值谱, LN gamma的分布, 中间层激活的稀疏性

使用方法:
    python phase_cxxi_unified_theory.py --model qwen3 --experiment p547
    python phase_cxxi_unified_theory.py --model glm4 --experiment p548
    python phase_cxxi_unified_theory.py --model deepseek7b --experiment p550
"""

import sys
import os
import argparse
import numpy as np
import torch
import json
import time
from scipy.stats import spearmanr, pearsonr
from scipy.sparse.linalg import svds

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))
from model_utils import (
    load_model, get_model_info, get_layers, get_layer_weights,
    get_W_U, release_model, get_sample_layers
)


def to_native(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.item() if v.numel() == 1 else v.cpu().tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


# ===== P547: 从dim_90推导ratio(k)的解析公式 =====
def run_p547(model, tokenizer, device, model_name):
    """
    从dim_90推导ratio(k)的解析公式
    
    三种假设:
    1. 均匀分布: ratio(k) = min(k/dim_90, 1.0)
    2. 幂律分布: ratio(k) = 1 - (1 - k/dim_90)^(1+alpha)
    3. 指数衰减: ratio(k) = 1 - exp(-k/dim_90 * c)
    """
    print("\n" + "="*70)
    print("P547: 从dim_90推导ratio(k)的解析公式")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    # W_U SVD
    W_U = get_W_U(model)
    k_wu = min(200, min(W_U.shape) - 2)
    print(f"计算W_U SVD (k={k_wu})...")
    U_wu, S_wu, _ = svds(W_U.T.astype(np.float32), k=k_wu)
    U_wu = U_wu[:, ::-1]
    S_wu = S_wu[::-1]
    
    test_text = "The fundamental nature of reality can be understood through mathematical structures that underlie physical phenomena."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    sample_layers = get_sample_layers(info.n_layers, min(8, info.n_layers))
    print(f"采样层: {sample_layers}")
    
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_hidden_states=True)
        h_states = outputs.hidden_states
    
    # 收集各层的ratio(k)和dim_90
    target_k_values = [10, 20, 30, 50, 80, 100, 150, 200]
    
    layer_data = []
    for l_idx in sample_layers:
        h = h_states[l_idx][0, -1].cpu().float().numpy()
        h_wu = U_wu.T @ h
        h_wu_sq = h_wu**2
        h_wu_total = np.sum(h_wu_sq) + 1e-10
        
        # ratio(k)实测
        ratio_k_measured = {}
        for k_target in target_k_values:
            if k_target > k_wu:
                continue
            top_idx = np.argsort(h_wu_sq)[-k_target:]
            ratio_k_measured[k_target] = float(np.sum(h_wu_sq[top_idx]) / h_wu_total)
        
        # dim_90
        sorted_sq = np.sort(h_wu_sq)[::-1]
        cumsum = np.cumsum(sorted_sq)
        dim_90 = int(np.searchsorted(cumsum / h_wu_total, 0.9) + 1)
        dim_95 = int(np.searchsorted(cumsum / h_wu_total, 0.95) + 1)
        dim_99 = int(np.searchsorted(cumsum / h_wu_total, 0.99) + 1)
        
        # 频谱幂律拟合
        nonzero = sorted_sq > 1e-15
        if np.sum(nonzero) > 20:
            ranks = np.arange(1, len(sorted_sq)+1)[nonzero]
            log_ranks = np.log(ranks)
            log_spectrum = np.log(sorted_sq[nonzero] + 1e-30)
            try:
                coeffs = np.polyfit(log_ranks[:100], log_spectrum[:100], 1)
                beta_power = float(-coeffs[0])
            except:
                beta_power = 1.0
        else:
            beta_power = 1.0
        
        layer_data.append({
            'layer': l_idx,
            'ratio_k': ratio_k_measured,
            'dim_90': dim_90,
            'dim_95': dim_95,
            'dim_99': dim_99,
            'beta_power': beta_power,
            'sorted_sq': sorted_sq,
        })
        
        print(f"  L{l_idx}: dim_90={dim_90}, dim_95={dim_95}, beta={beta_power:.3f}, "
              f"ratio(50)={ratio_k_measured.get(50, 0):.4f}")
    
    # ===== 三种假设的拟合 =====
    print("\n--- 三种假设的拟合 ---")
    
    # 收集所有数据点
    all_k = []
    all_ratio = []
    all_dim90 = []
    
    for ld in layer_data:
        for k, r in ld['ratio_k'].items():
            all_k.append(k)
            all_ratio.append(r)
            all_dim90.append(ld['dim_90'])
    
    all_k = np.array(all_k, dtype=float)
    all_ratio = np.array(all_ratio)
    all_dim90 = np.array(all_dim90, dtype=float)
    
    # 假设1: 均匀分布 ratio(k) = min(k/dim_90, 1.0)
    pred_uniform = np.minimum(all_k / all_dim90, 1.0)
    err_uniform = float(np.mean((pred_uniform - all_ratio)**2))
    r2_uniform = float(1 - np.sum((pred_uniform - all_ratio)**2) / 
                       (np.sum((all_ratio - np.mean(all_ratio))**2) + 1e-10))
    
    # 假设2: 幂律分布 ratio(k) = (k/dim_90)^alpha
    # 用最小二乘拟合alpha
    valid = (all_k > 0) & (all_ratio > 0) & (all_ratio < 1)
    if np.sum(valid) > 5:
        log_k_norm = np.log(all_k[valid] / all_dim90[valid])
        log_ratio = np.log(all_ratio[valid])
        # 避免log(0)
        good = np.isfinite(log_k_norm) & np.isfinite(log_ratio)
        if np.sum(good) > 3:
            alpha_fit = float(np.polyfit(log_k_norm[good], log_ratio[good], 1)[0])
        else:
            alpha_fit = 1.0
    else:
        alpha_fit = 1.0
    
    pred_powerlaw = (all_k / all_dim90)**alpha_fit
    pred_powerlaw = np.clip(pred_powerlaw, 0, 1)
    err_powerlaw = float(np.mean((pred_powerlaw - all_ratio)**2))
    r2_powerlaw = float(1 - np.sum((pred_powerlaw - all_ratio)**2) / 
                        (np.sum((all_ratio - np.mean(all_ratio))**2) + 1e-10))
    
    # 假设3: 指数衰减 ratio(k) = 1 - exp(-k/dim_90 * c)
    # 拟合c
    if np.sum(valid) > 5:
        k_norm = all_k[valid] / all_dim90[valid]
        r_vals = all_ratio[valid]
        # 1 - ratio = exp(-k/dim_90 * c) -> -log(1-ratio) = k/dim_90 * c
        log_complement = -np.log(np.maximum(1 - r_vals, 1e-10))
        good2 = np.isfinite(log_complement) & (r_vals < 0.99)
        if np.sum(good2) > 3:
            c_fit = float(np.polyfit(k_norm[good2], log_complement[good2], 1)[0])
        else:
            c_fit = 1.0
    else:
        c_fit = 1.0
    
    pred_exp = 1 - np.exp(-all_k / all_dim90 * c_fit)
    pred_exp = np.clip(pred_exp, 0, 1)
    err_exp = float(np.mean((pred_exp - all_ratio)**2))
    r2_exp = float(1 - np.sum((pred_exp - all_ratio)**2) / 
                   (np.sum((all_ratio - np.mean(all_ratio))**2) + 1e-10))
    
    print(f"  均匀分布: ratio(k) = min(k/dim_90, 1) -> MSE={err_uniform:.6f}, R2={r2_uniform:.4f}")
    print(f"  幂律分布: ratio(k) = (k/dim_90)^{alpha_fit:.3f} -> MSE={err_powerlaw:.6f}, R2={r2_powerlaw:.4f}")
    print(f"  指数衰减: ratio(k) = 1-exp(-k/dim_90*{c_fit:.3f}) -> MSE={err_exp:.6f}, R2={r2_exp:.4f}")
    
    # ===== 假设4: 用频谱幂律直接推导 =====
    print("\n--- 假设4: 频谱幂律直接推导 ---")
    # S(i) = C * i^(-beta), ratio(k) = sum_{i=1}^k i^(-beta) / sum_{i=1}^N i^(-beta)
    # 简化: ratio(k) ≈ H(k, beta) / H(N, beta), H是广义调和数
    
    avg_beta = float(np.mean([ld['beta_power'] for ld in layer_data]))
    
    def powerlaw_ratio(k, beta, N=200):
        """用幂律频谱计算ratio(k)"""
        ranks = np.arange(1, N+1, dtype=float)
        spectrum = ranks**(-beta)
        total = np.sum(spectrum)
        if k >= N:
            return 1.0
        return float(np.sum(spectrum[:k]) / total)
    
    pred_spectrum = np.array([powerlaw_ratio(int(k), avg_beta) for k in all_k])
    err_spectrum = float(np.mean((pred_spectrum - all_ratio)**2))
    r2_spectrum = float(1 - np.sum((pred_spectrum - all_ratio)**2) / 
                        (np.sum((all_ratio - np.mean(all_ratio))**2) + 1e-10))
    
    print(f"  频谱幂律: S(i)=i^(-{avg_beta:.3f}), ratio(k)=cumsum/total -> MSE={err_spectrum:.6f}, R2={r2_spectrum:.4f}")
    
    # ===== 最优假设 =====
    print("\n--- 最优假设 ---")
    models = {
        '均匀分布': (r2_uniform, err_uniform, f'ratio(k) = min(k/dim_90, 1)'),
        '幂律分布': (r2_powerlaw, err_powerlaw, f'ratio(k) = (k/dim_90)^{alpha_fit:.3f}'),
        '指数衰减': (r2_exp, err_exp, f'ratio(k) = 1-exp(-k/dim_90*{c_fit:.3f})'),
        '频谱幂律': (r2_spectrum, err_spectrum, f'S(i)=i^(-{avg_beta:.3f})'),
    }
    
    best_name = max(models, key=lambda x: models[x][0])
    for name, (r2, mse, formula) in sorted(models.items(), key=lambda x: -x[1][0]):
        print(f"  {name}: R2={r2:.4f}, MSE={mse:.6f}")
    
    print(f"\n  最优模型: {best_name}")
    print(f"  公式: {models[best_name][2]}")
    
    results = {
        'experiment': 'P547',
        'model': model_name,
        'alpha_fit': alpha_fit,
        'c_fit': c_fit,
        'avg_beta_power': avg_beta,
        'r2_uniform': r2_uniform,
        'r2_powerlaw': r2_powerlaw,
        'r2_exp': r2_exp,
        'r2_spectrum': r2_spectrum,
        'best_model': best_name,
    }
    
    out_path = f"results/phase_cxxi/P547_{model_name}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(to_native(results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")
    
    torch.cuda.empty_cache()
    return results


# ===== P548: 频谱的普适形状函数 =====
def run_p548(model, tokenizer, device, model_name):
    """
    频谱的普适形状函数
    
    核心思路: 用Zeta函数从幂律指数推导ratio(k)
    S(i) = C * i^(-beta)
    ratio(k) = sum_{i=1}^k i^(-beta) / zeta(beta)
    
    但实测频谱可能不是纯幂律, 需要修正:
    1. 截断幂律: S(i) = C * i^(-beta) * exp(-i/L) (L是截断尺度)
    2. 双幂律: S(i) = C1 * i^(-beta1) (i<L) + C2 * i^(-beta2) (i>=L)
    3. 对数正态: S(i) = C * exp(-(log i - mu)^2 / (2*sigma^2))
    """
    print("\n" + "="*70)
    print("P548: 频谱的普适形状函数")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    # W_U SVD
    W_U = get_W_U(model)
    k_wu = min(200, min(W_U.shape) - 2)
    print(f"计算W_U SVD (k={k_wu})...")
    U_wu, S_wu, _ = svds(W_U.T.astype(np.float32), k=k_wu)
    U_wu = U_wu[:, ::-1]
    
    test_text = "The fundamental nature of reality can be understood through mathematical structures that underlie physical phenomena."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    sample_layers = get_sample_layers(info.n_layers, min(8, info.n_layers))
    
    # 收集多个输入的频谱
    test_texts = [
        "The fundamental nature of reality can be understood through mathematical structures.",
        "In the beginning was the Word, and the Word was with God.",
        "The quantum mechanical description requires probabilistic interpretation.",
        "Artificial intelligence systems process information through layered transformations.",
        "The beauty of mathematics lies in its ability to reveal hidden patterns.",
    ]
    
    all_spectra = []
    
    for text_idx, text in enumerate(test_texts):
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        
        for l_idx in sample_layers:
            h = outputs.hidden_states[l_idx][0, -1].cpu().float().numpy()
            h_wu = U_wu.T @ h
            h_wu_sq = h_wu**2
            h_wu_total = np.sum(h_wu_sq) + 1e-10
            spectrum = h_wu_sq / h_wu_total  # 归一化频谱
            all_spectra.append(spectrum)
        
        torch.cuda.empty_cache()
    
    all_spectra = np.array(all_spectra)  # [n_samples, k_wu]
    avg_spectrum = np.mean(all_spectra, axis=0)
    
    # ===== 分析1: 频谱形状拟合 =====
    print("\n--- 频谱形状拟合 ---")
    
    ranks = np.arange(1, k_wu + 1, dtype=float)
    log_ranks = np.log(ranks)
    log_spectrum = np.log(avg_spectrum + 1e-30)
    
    # 纯幂律拟合
    valid = avg_spectrum > 1e-15
    if np.sum(valid) > 10:
        coeffs_pl = np.polyfit(log_ranks[valid], log_spectrum[valid], 1)
        beta_pl = float(-coeffs_pl[0])
        intercept_pl = float(coeffs_pl[1])
        pred_pl = intercept_pl - beta_pl * log_ranks
        r2_pl = float(1 - np.sum((log_spectrum[valid] - pred_pl[valid])**2) / 
                      np.sum((log_spectrum[valid] - np.mean(log_spectrum[valid]))**2))
    else:
        beta_pl = 1.0
        r2_pl = 0
    
    # 分段幂律拟合(前20和后180)
    n_split = min(20, k_wu // 5)
    valid_early = valid.copy()
    valid_early[n_split:] = False
    valid_late = valid.copy()
    valid_late[:n_split] = False
    
    if np.sum(valid_early) > 3 and np.sum(valid_late) > 3:
        coeffs_early = np.polyfit(log_ranks[valid_early], log_spectrum[valid_early], 1)
        beta_early = float(-coeffs_early[0])
        coeffs_late = np.polyfit(log_ranks[valid_late], log_spectrum[valid_late], 1)
        beta_late = float(-coeffs_late[0])
    else:
        beta_early = beta_pl
        beta_late = beta_pl
    
    # 截断幂律拟合
    # S(i) = C * i^(-beta) * exp(-i/L)
    # log S = log C - beta * log i - i/L
    def truncated_powerlaw_mse(params, ranks, spectrum):
        beta, L, log_C = params
        pred = log_C - beta * np.log(ranks) - ranks / max(L, 1.0)
        return np.mean((pred - np.log(spectrum + 1e-30))**2)
    
    from scipy.optimize import minimize
    try:
        res = minimize(truncated_powerlaw_mse, [beta_pl, 100.0, np.log(avg_spectrum[0]+1e-10)],
                      args=(ranks[valid], avg_spectrum[valid]), method='Nelder-Mead')
        beta_trunc, L_trunc, log_C_trunc = res.x
        pred_trunc = log_C_trunc - beta_trunc * log_ranks - ranks / max(L_trunc, 1.0)
        r2_trunc = float(1 - np.sum((log_spectrum[valid] - pred_trunc[valid])**2) / 
                         np.sum((log_spectrum[valid] - np.mean(log_spectrum[valid]))**2))
    except:
        beta_trunc = beta_pl
        L_trunc = 1e6
        r2_trunc = r2_pl
    
    print(f"  纯幂律: beta={beta_pl:.3f}, R2={r2_pl:.4f}")
    print(f"  分段幂律: beta_early={beta_early:.3f}, beta_late={beta_late:.3f}")
    print(f"  截断幂律: beta={beta_trunc:.3f}, L={L_trunc:.1f}, R2={r2_trunc:.4f}")
    
    # ===== 分析2: 从频谱形状预测ratio(k) =====
    print("\n--- 从频谱形状预测ratio(k) ---")
    
    target_k_values = [10, 20, 30, 50, 80, 100, 150, 200]
    
    # 实测ratio(k)
    measured_ratios = {}
    for k_target in target_k_values:
        if k_target <= k_wu:
            measured_ratios[k_target] = float(np.sum(avg_spectrum[:k_target]))
    
    # 幂律预测
    def powerlaw_ratio(k, beta):
        ranks_k = np.arange(1, k + 1, dtype=float)
        ranks_N = np.arange(1, k_wu + 1, dtype=float)
        return float(np.sum(ranks_k**(-beta)) / np.sum(ranks_N**(-beta)))
    
    # 截断幂律预测
    def truncated_ratio(k, beta, L):
        ranks_k = np.arange(1, k + 1, dtype=float)
        ranks_N = np.arange(1, k_wu + 1, dtype=float)
        spec_k = ranks_k**(-beta) * np.exp(-ranks_k / max(L, 1.0))
        spec_N = ranks_N**(-beta) * np.exp(-ranks_N / max(L, 1.0))
        return float(np.sum(spec_k) / np.sum(spec_N))
    
    pred_errors = {}
    for k_target in target_k_values:
        if k_target > k_wu or k_target not in measured_ratios:
            continue
        r_meas = measured_ratios[k_target]
        r_pl = powerlaw_ratio(k_target, beta_pl)
        r_trunc = truncated_ratio(k_target, beta_trunc, L_trunc)
        
        pred_errors[k_target] = {
            'measured': r_meas,
            'powerlaw': r_pl,
            'truncated': r_trunc,
            'err_pl': abs(r_pl - r_meas),
            'err_trunc': abs(r_trunc - r_meas),
        }
        print(f"  k={k_target}: measured={r_meas:.4f}, powerlaw={r_pl:.4f}(err={abs(r_pl-r_meas):.4f}), "
              f"truncated={r_trunc:.4f}(err={abs(r_trunc-r_meas):.4f})")
    
    # ===== 分析3: 频谱的"普适"性 =====
    print("\n--- 频谱的普适性 ---")
    
    # 计算各频谱间的相关
    spectra_corrs = []
    for i in range(len(all_spectra)):
        for j in range(i+1, len(all_spectra)):
            c = float(np.corrcoef(all_spectra[i], all_spectra[j])[0, 1])
            spectra_corrs.append(c)
    
    avg_corr = float(np.mean(spectra_corrs)) if spectra_corrs else 0
    print(f"  频谱间平均相关: {avg_corr:.4f} ({len(spectra_corrs)}对)")
    
    if avg_corr > 0.95:
        print("  >> 频谱形状高度普适: 存在统一的频谱函数")
    elif avg_corr > 0.8:
        print("  >> 频谱形状中度普适: 有统一趋势但细节不同")
    else:
        print("  >> 频谱形状不普适: 随输入/层变化大")
    
    results = {
        'experiment': 'P548',
        'model': model_name,
        'beta_powerlaw': beta_pl,
        'r2_powerlaw': r2_pl,
        'beta_early': beta_early,
        'beta_late': beta_late,
        'beta_truncated': beta_trunc,
        'L_truncated': L_trunc,
        'r2_truncated': r2_trunc,
        'avg_spectra_corr': avg_corr,
    }
    
    out_path = f"results/phase_cxxi/P548_{model_name}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(to_native(results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")
    
    torch.cuda.empty_cache()
    return results


# ===== P549: 吸引子的精确模型 =====
def run_p549(model, tokenizer, device, model_name):
    """
    吸引子的精确模型
    
    核心思路: LayerNorm + 残差连接产生了不动点
    
    不动点分析:
    h_out = h_in + MLP(LN(h_in))
    
    频谱层面的不动点条件:
    如果h_in和h_out有相同的频谱形状, 则:
    spectrum(h_out) ≈ spectrum(h_in) * (1 + small_correction)
    
    LayerNorm的作用:
    LN(h) = (h - mean(h)) / std(h) * gamma
    gamma(LN权重)的频谱决定了频谱的"偏向"
    
    验证: gamma的频谱分布是否与h的频谱分布相关?
    """
    print("\n" + "="*70)
    print("P549: 吸引子的精确模型")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    # W_U SVD
    W_U = get_W_U(model)
    k_wu = min(200, min(W_U.shape) - 2)
    print(f"计算W_U SVD (k={k_wu})...")
    U_wu, S_wu, _ = svds(W_U.T.astype(np.float32), k=k_wu)
    U_wu = U_wu[:, ::-1]
    
    sample_layers = get_sample_layers(info.n_layers, min(8, info.n_layers))
    print(f"采样层: {sample_layers}")
    
    # ===== 分析1: LN gamma权重的频谱 =====
    print("\n--- LN gamma权重的频谱 ---")
    
    gamma_spectra = {}
    for l_idx in sample_layers:
        lw = get_layer_weights(layers[l_idx], info.d_model, info.mlp_type)
        
        # 输入LN和post-attn LN的gamma
        if lw.input_layernorm_weight is not None:
            gamma_in = lw.input_layernorm_weight
            gamma_in_wu = U_wu.T @ gamma_in
            gamma_in_wu_sq = gamma_in_wu**2
            gamma_in_total = np.sum(gamma_in_wu_sq) + 1e-10
            
            # gamma在W_U top-50上的能量比
            top50_idx = np.argsort(gamma_in_wu_sq)[-50:]
            ratio_50_gamma_in = float(np.sum(gamma_in_wu_sq[top50_idx]) / gamma_in_total)
            
            # gamma频谱的PR
            gamma_in_norm = np.sqrt(gamma_in_wu_sq)
            gamma_in_norm = gamma_in_norm / (gamma_in_norm[0] + 1e-10)
            PR_gamma_in = float(gamma_in_norm.sum()**2 / (k_wu * (gamma_in_norm**2).sum()))
        else:
            ratio_50_gamma_in = 0
            PR_gamma_in = 0
        
        if lw.post_attn_layernorm_weight is not None:
            gamma_post = lw.post_attn_layernorm_weight
            gamma_post_wu = U_wu.T @ gamma_post
            gamma_post_wu_sq = gamma_post_wu**2
            gamma_post_total = np.sum(gamma_post_wu_sq) + 1e-10
            
            top50_idx = np.argsort(gamma_post_wu_sq)[-50:]
            ratio_50_gamma_post = float(np.sum(gamma_post_wu_sq[top50_idx]) / gamma_post_total)
            
            gamma_post_norm = np.sqrt(gamma_post_wu_sq)
            gamma_post_norm = gamma_post_norm / (gamma_post_norm[0] + 1e-10)
            PR_gamma_post = float(gamma_post_norm.sum()**2 / (k_wu * (gamma_post_norm**2).sum()))
        else:
            ratio_50_gamma_post = 0
            PR_gamma_post = 0
        
        gamma_spectra[l_idx] = {
            'ratio_50_input': ratio_50_gamma_in,
            'PR_input': PR_gamma_in,
            'ratio_50_post': ratio_50_gamma_post,
            'PR_post': PR_gamma_post,
        }
        
        print(f"  L{l_idx}: gamma_input ratio(50)={ratio_50_gamma_in:.4f} PR={PR_gamma_in:.4f}, "
              f"gamma_post ratio(50)={ratio_50_gamma_post:.4f} PR={PR_gamma_post:.4f}")
    
    # ===== 分析2: gamma频谱 vs h频谱的相关 =====
    print("\n--- gamma频谱 vs h频谱 ---")
    
    test_text = "The fundamental nature of reality can be understood through mathematical structures that underlie physical phenomena."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_hidden_states=True)
    
    gamma_h_corrs = []
    for l_idx in sample_layers:
        h = outputs.hidden_states[l_idx][0, -1].cpu().float().numpy()
        h_wu = U_wu.T @ h
        h_wu_sq = h_wu**2
        h_wu_total = np.sum(h_wu_sq) + 1e-10
        h_spectrum = h_wu_sq / h_wu_total
        
        lw = get_layer_weights(layers[l_idx], info.d_model, info.mlp_type)
        
        # gamma_input的频谱
        if lw.input_layernorm_weight is not None:
            gamma_in = lw.input_layernorm_weight
            gamma_in_wu = U_wu.T @ gamma_in
            gamma_in_wu_sq = gamma_in_wu**2
            gamma_in_total = np.sum(gamma_in_wu_sq) + 1e-10
            gamma_spectrum = gamma_in_wu_sq / gamma_in_total
            
            corr = float(np.corrcoef(h_spectrum, gamma_spectrum)[0, 1])
            gamma_h_corrs.append(corr)
            print(f"  L{l_idx}: gamma_input vs h频谱相关 = {corr:.4f}")
    
    avg_gamma_h_corr = float(np.mean(gamma_h_corrs)) if gamma_h_corrs else 0
    print(f"  平均相关: {avg_gamma_h_corr:.4f}")
    
    # ===== 分析3: 不动点的理论推导 =====
    print("\n--- 不动点理论推导 ---")
    
    # 如果h的频谱是不动点, 则:
    # spectrum(h_out) = spectrum(h_in) + small_correction
    # 
    # LayerNorm效果: h_norm = (h - mean)/std * gamma
    # 在W_U空间中: h_norm_wu ≈ gamma_wu .* h_wu / std (忽略mean, 因为LN后均值为0)
    # 
    # 如果gamma的频谱与h的频谱正相关, 则LN会"放大"h已有的频谱结构
    # 这就是吸引子的机制!
    
    if avg_gamma_h_corr > 0.5:
        print("  >> gamma与h频谱强正相关: LN放大已有频谱结构, 形成正反馈吸引子")
    elif avg_gamma_h_corr > 0:
        print("  >> gamma与h频谱弱正相关: LN轻微放大频谱结构")
    else:
        print("  >> gamma与h频谱无关/负相关: LN不形成频谱吸引子")
    
    # ===== 分析4: 吸引子的"自洽"验证 =====
    print("\n--- 吸引子自洽验证 ---")
    
    # 如果gamma确实形成吸引子, 则:
    # ratio_50(gamma) 应该与 ratio_50(h) 正相关
    ratio50_gamma = [gamma_spectra[l]['ratio_50_input'] for l in sample_layers]
    ratio50_h = []
    for l_idx in sample_layers:
        h = outputs.hidden_states[l_idx][0, -1].cpu().float().numpy()
        h_wu = U_wu.T @ h
        h_wu_sq = h_wu**2
        h_wu_total = np.sum(h_wu_sq) + 1e-10
        top50 = np.argsort(h_wu_sq)[-50:]
        ratio50_h.append(float(np.sum(h_wu_sq[top50]) / h_wu_total))
    
    if len(ratio50_gamma) > 2 and len(ratio50_h) > 2:
        corr_ratio = float(np.corrcoef(ratio50_gamma, ratio50_h)[0, 1])
    else:
        corr_ratio = 0
    
    print(f"  ratio(50)_gamma vs ratio(50)_h 相关: {corr_ratio:.4f}")
    
    # 平均gamma ratio(50) vs 平均h ratio(50)
    avg_ratio50_gamma = float(np.mean(ratio50_gamma))
    avg_ratio50_h = float(np.mean(ratio50_h))
    print(f"  平均 ratio(50)_gamma: {avg_ratio50_gamma:.4f}")
    print(f"  平均 ratio(50)_h: {avg_ratio50_h:.4f}")
    
    if avg_ratio50_gamma > avg_ratio50_h:
        print("  >> gamma比h更集中: LN的gamma权重偏向前方, 会增强频谱集中")
    else:
        print("  >> gamma比h更分散: LN的gamma权重分布较均匀")
    
    results = {
        'experiment': 'P549',
        'model': model_name,
        'avg_gamma_h_corr': avg_gamma_h_corr,
        'corr_ratio50': corr_ratio,
        'avg_ratio50_gamma': avg_ratio50_gamma,
        'avg_ratio50_h': avg_ratio50_h,
        'gamma_spectra': {str(k): v for k, v in gamma_spectra.items()},
    }
    
    out_path = f"results/phase_cxxi/P549_{model_name}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(to_native(results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")
    
    torch.cuda.empty_cache()
    return results


# ===== P550: 为什么DS7B比Qwen3更"稳定"? =====
def run_p550(model, tokenizer, device, model_name):
    """
    DS7B超稳定性的来源分析
    
    P542-P546发现DS7B比Qwen3/GLM4更"稳定":
    - dim_90: DS7B=38 vs Qwen3=140
    - 旋转角度: DS7B=0.1deg vs Qwen3=5-15deg
    - 跨输入频谱相关: DS7B=0.999 vs Qwen3=0.905
    
    可能原因:
    1. W_down奇异值谱更集中(低秩更强)
    2. LN gamma更集中(频谱吸引子更强)
    3. 激活更稀疏(中间层活跃神经元更少)
    4. 残差连接的保持更完美(preservation更高)
    """
    print("\n" + "="*70)
    print("P550: DS7B超稳定性的来源分析")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    # W_U SVD
    W_U = get_W_U(model)
    k_wu = min(200, min(W_U.shape) - 2)
    print(f"计算W_U SVD (k={k_wu})...")
    U_wu, S_wu, _ = svds(W_U.T.astype(np.float32), k=k_wu)
    U_wu = U_wu[:, ::-1]
    S_wu = S_wu[::-1]
    
    sample_layers = get_sample_layers(info.n_layers, min(8, info.n_layers))
    print(f"采样层: {sample_layers}")
    
    # ===== 原因1: W_down奇异值谱 =====
    print("\n--- 原因1: W_down奇异值谱集中度 ---")
    
    wdown_prs = []
    for l_idx in sample_layers:
        lw = get_layer_weights(layers[l_idx], info.d_model, info.mlp_type)
        W_down = lw.W_down  # [d_model, intermediate]
        
        # W_down的top-50奇异值的PR
        k_svd = min(50, min(W_down.shape) - 1)
        U_wd, S_wd, _ = svds(W_down.astype(np.float32), k=k_svd)
        S_wd = S_wd[::-1]
        
        PR_wd = float(S_wd.sum()**2 / (k_svd * (S_wd**2).sum() + 1e-10))
        # top-1占比
        top1_ratio = float(S_wd[0]**2 / (S_wd**2).sum())
        
        wdown_prs.append(PR_wd)
        print(f"  L{l_idx}: W_down PR(50)={PR_wd:.4f}, top1_ratio={top1_ratio:.4f}")
    
    avg_wd_pr = float(np.mean(wdown_prs))
    print(f"  平均W_down PR(50): {avg_wd_pr:.4f}")
    
    # ===== 原因2: LN gamma集中度 =====
    print("\n--- 原因2: LN gamma集中度 ---")
    
    gamma_concentrations = []
    for l_idx in sample_layers:
        lw = get_layer_weights(layers[l_idx], info.d_model, info.mlp_type)
        
        if lw.input_layernorm_weight is not None:
            gamma = lw.input_layernorm_weight
            gamma_wu = U_wu.T @ gamma
            gamma_wu_sq = gamma_wu**2
            gamma_total = np.sum(gamma_wu_sq) + 1e-10
            
            # gamma在W_U top-50的集中度
            top50 = np.argsort(gamma_wu_sq)[-50:]
            ratio_50 = float(np.sum(gamma_wu_sq[top50]) / gamma_total)
            gamma_concentrations.append(ratio_50)
            print(f"  L{l_idx}: gamma ratio(50)={ratio_50:.4f}")
    
    avg_gamma_conc = float(np.mean(gamma_concentrations)) if gamma_concentrations else 0
    print(f"  平均gamma ratio(50): {avg_gamma_conc:.4f}")
    
    # ===== 原因3: 激活稀疏性 =====
    print("\n--- 原因3: 激活稀疏性 ---")
    
    test_text = "The fundamental nature of reality can be understood through mathematical structures that underlie physical phenomena."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    sparsity_data = []
    for l_idx in sample_layers:
        lw = get_layer_weights(layers[l_idx], info.d_model, info.mlp_type)
        W_up = lw.W_up
        W_gate = lw.W_gate
        post_ln_w = lw.post_attn_layernorm_weight
        
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
            h = outputs.hidden_states[l_idx][0, -1].cpu().float().numpy()
        
        # 计算f_mlp
        h_centered = h - h.mean()
        sigma = np.std(h_centered) + 1e-10
        ln_h = h_centered / sigma
        if post_ln_w is not None:
            ln_h = ln_h * post_ln_w
        
        if W_gate is not None:
            gate_pre = W_gate @ ln_h
            gate_act = 1.0 / (1.0 + np.exp(-gate_pre))
        else:
            gate_act = np.ones(W_up.shape[0])
        up_out = W_up @ ln_h
        f_mlp = gate_act * up_out
        
        # 稀疏性: 活跃神经元比例(>1% max)
        max_val = np.max(np.abs(f_mlp))
        active_ratio = float(np.sum(np.abs(f_mlp) > 0.01 * max_val) / len(f_mlp))
        
        # top-10占比
        top10_vals = np.sort(np.abs(f_mlp))[-10:]
        top10_ratio = float(np.sum(top10_vals**2) / (np.sum(f_mlp**2) + 1e-10))
        
        sparsity_data.append({
            'active_ratio': active_ratio,
            'top10_ratio': top10_ratio,
        })
        print(f"  L{l_idx}: 活跃比={active_ratio:.4f}, top10占比={top10_ratio:.4f}")
        
        torch.cuda.empty_cache()
    
    avg_active = float(np.mean([d['active_ratio'] for d in sparsity_data]))
    avg_top10 = float(np.mean([d['top10_ratio'] for d in sparsity_data]))
    print(f"  平均活跃比: {avg_active:.4f}")
    print(f"  平均top10占比: {avg_top10:.4f}")
    
    # ===== 原因4: 残差保持 =====
    print("\n--- 原因4: 残差保持 ---")
    
    preservation_data = []
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_hidden_states=True)
    
    for l_idx in sample_layers:
        h_in = outputs.hidden_states[l_idx][0, -1].cpu().float().numpy()
        h_out = outputs.hidden_states[l_idx + 1][0, -1].cpu().float().numpy()
        delta_h = h_out - h_in
        
        h_in_wu = U_wu.T @ h_in
        h_out_wu = U_wu.T @ h_out
        
        h_in_wu_sq = h_in_wu**2
        h_out_wu_sq = h_out_wu**2
        
        h_in_total = np.sum(h_in_wu_sq) + 1e-10
        h_out_total = np.sum(h_out_wu_sq) + 1e-10
        
        # preservation: h_in在h_out top-50上的能量比
        top50_hout = np.argsort(h_out_wu_sq)[-50:]
        pres = float(np.sum(h_in_wu_sq[top50_hout]) / h_in_total)
        
        # ||delta|| / ||h_in|| 比
        delta_ratio = float(np.linalg.norm(delta_h) / (np.linalg.norm(h_in) + 1e-10))
        
        preservation_data.append({
            'preservation': pres,
            'delta_ratio': delta_ratio,
        })
        print(f"  L{l_idx}: preservation(50)={pres:.4f}, delta_ratio={delta_ratio:.4f}")
    
    avg_pres = float(np.mean([d['preservation'] for d in preservation_data]))
    avg_delta = float(np.mean([d['delta_ratio'] for d in preservation_data]))
    print(f"  平均preservation(50): {avg_pres:.4f}")
    print(f"  平均delta_ratio: {avg_delta:.4f}")
    
    # ===== 综合分析 =====
    print("\n--- 综合分析 ---")
    print(f"  W_down PR(50): {avg_wd_pr:.4f} (越低=越集中)")
    print(f"  Gamma ratio(50): {avg_gamma_conc:.4f} (越高=越集中)")
    print(f"  活跃比: {avg_active:.4f} (越低=越稀疏)")
    print(f"  Top10占比: {avg_top10:.4f} (越高=越稀疏)")
    print(f"  Preservation(50): {avg_pres:.4f} (越高=越稳定)")
    print(f"  Delta ratio: {avg_delta:.4f} (越低=越稳定)")
    
    # 稳定性分数(综合)
    # 更集中(WD PR低), 更集中(gamma ratio高), 更稀疏(活跃比低/top10高), 更稳定(pres高/delta低)
    stability_score = (
        (1 - avg_wd_pr) * 0.2 +  # W_down集中度
        avg_gamma_conc * 0.2 +     # Gamma集中度
        avg_top10 * 0.2 +          # 稀疏性
        avg_pres * 0.2 +           # 保持性
        (1 - min(avg_delta, 1.0)) * 0.2  # 微小变化
    )
    print(f"\n  稳定性综合分数: {stability_score:.4f}")
    
    results = {
        'experiment': 'P550',
        'model': model_name,
        'avg_wd_PR50': avg_wd_pr,
        'avg_gamma_ratio50': avg_gamma_conc,
        'avg_active_ratio': avg_active,
        'avg_top10_ratio': avg_top10,
        'avg_preservation50': avg_pres,
        'avg_delta_ratio': avg_delta,
        'stability_score': stability_score,
    }
    
    out_path = f"results/phase_cxxi/P550_{model_name}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(to_native(results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")
    
    torch.cuda.empty_cache()
    return results


# ===== 主函数 =====
EXPERIMENTS = {
    'p547': run_p547,
    'p548': run_p548,
    'p549': run_p549,
    'p550': run_p550,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase CXXI-CXXII: 流形动力学的统一理论")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True,
                       choices=list(EXPERIMENTS.keys()))
    args = parser.parse_args()
    
    print(f"模型: {args.model}, 实验: {args.experiment}")
    
    model, tokenizer, device = load_model(args.model)
    try:
        result = EXPERIMENTS[args.experiment](model, tokenizer, device, args.model)
        print(f"\n实验 {args.experiment} 完成!")
    finally:
        release_model(model)
