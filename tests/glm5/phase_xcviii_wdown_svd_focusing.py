"""
Phase XCVIII-P472/473/474/475: 权重矩阵SVD谱与信号聚焦
======================================================================

核心目标: 理解为什么c/delta与d/n_L相关, 推导c/delta = f(d/n_L)的解析公式

Phase XCVII核心成果:
1. 频谱演化确认非马尔可夫(xi=3-6层)
2. AR2是最佳演化模型(R2=0.66-0.70 vs Fokker-Planck 0.19-0.33)
3. c/delta与d/n_L强相关(Corr~0.994):
   - Qwen3: d/n_L=71.1, c/delta=1.09
   - GLM4: d/n_L=102.4, c/delta=1.76
   - DS7B: d/n_L=128.0, c/delta=2.56
4. 跨模型统一gamma公式R2=0.991

关键问题:
1. 为什么d/n_L越大, 信号聚焦越强?
2. W_down的SVD谱与d/n_L有什么关系?
3. c/delta = f(d/n_L)的解析公式是什么?

Phase XCVIII目标:
1. 分析W_down的SVD谱与d/n_L的关系
2. 推导c/delta = f(d/n_L)的解析公式
3. 建立信号聚焦相图: alpha vs d/n_L
4. 验证: 权重矩阵的"频谱丰满度"是否决定信号聚焦

P472: W_down SVD谱分析
  - 目标: 测量每层W_down的SVD谱, 分析与d/n_L的关系
  - 方法:
    a) 对每层的W_down做完整SVD
    b) 测量SVD谱的特征: 衰减率delta_W, 频谱丰满度F, 条件数kappa
    c) 分析这些特征与层位置, d/n_L, alpha的关系
  - 关键指标:
    - delta_W: W_down奇异值谱衰减率 (s_i ~ i^(-delta_W))
    - F = sum(s_i^2) / (k * s_max^2): 频谱丰满度 (F=1: 完美均匀, F<<1: 集中)
    - kappa = s_max / s_min: 条件数
  - 预期: d/n_L越大, F越大(频谱越丰满), delta_W越小

P473: c/delta = f(d/n_L)解析公式
  - 目标: 从W_down的SVD统计推导c/delta的解析公式
  - 方法:
    a) 收集三模型的(delta_W, F, kappa, d/n_L, c/delta)数据
    b) 拟合c/delta = a * (d/n_L)^b + c
    c) 检验c/delta是否与F或delta_W相关
    d) 推导c/delta的理论上限和下限
  - 候选公式:
    - c/delta = 1 + a * (d/n_L - d0/n_L0)
    - c/delta = a * ln(d/n_L) + b
    - c/delta = a * sqrt(d/n_L / delta_W) + b

P474: 信号聚焦相图
  - 目标: 建立alpha vs d/n_L的相图, 找出聚焦/分散的边界
  - 方法:
    a) 对三模型, 测量每层的alpha和d/n_L
    b) 在alpha-d/n_L平面上画散点图
    c) 找出alpha=0的分界线(聚焦/分散边界)
    d) 分析alpha的层间演化在相图上的轨迹
  - 预期: alpha>0(聚焦)和alpha<0(分散)有明显的区域划分

P475: 权重矩阵频谱丰满度与信号聚焦的因果分析
  - 目标: 检验W_down的频谱丰满度F是否因果决定alpha
  - 方法:
    a) 测量每层的F和alpha
    b) 计算F与alpha的层间相关性
    c) Granger因果检验: F(L)是否预测alpha(L+1)?
    d) 反向: alpha(L)是否预测F(L+1)? (不应该, 因为F由权重决定)
  - 预期: F(L) -> alpha(L+1) (因果), alpha(L) -/-> F(L+1) (非因果)
"""

import sys
import os
import argparse
import numpy as np
from scipy.sparse.linalg import svds
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_utils import (
    load_model, get_layers, get_layer_weights, get_model_info,
    get_W_U, release_model, MODEL_CONFIGS
)


def safe_svd(matrix, k, random_state=42):
    """Memory-safe SVD"""
    k = min(k, min(matrix.shape) - 1)
    if max(matrix.shape) > 50000:
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=k, random_state=random_state)
        if matrix.shape[0] < matrix.shape[1]:
            svd.fit(matrix.T.astype(np.float32))
            U = svd.components_.T.astype(np.float64)
            s = svd.singular_values_.astype(np.float64)
        else:
            svd.fit(matrix.astype(np.float32))
            U = svd.components_.T.astype(np.float64)
            s = svd.singular_values_.astype(np.float64)
        sort_idx = np.argsort(s)[::-1]
        U = U[:, sort_idx]
        s = s[sort_idx]
    else:
        U, s, _ = svds(matrix.astype(np.float64), k=k)
        s = np.sort(s)[::-1]
        sort_idx = np.argsort(s)[::-1]
        U = U[:, sort_idx]
    return U, s


def compute_spectral_features(s_values, k_max=None):
    """
    计算SVD谱的特征指标
    
    Returns:
        delta_W: 衰减率 s_i ~ i^(-delta_W)
        F: 频谱丰满度 = sum(s_i^2) / (k * s_max^2)
        kappa: 条件数 = s_max / s_min
        participation_ratio: PR = (sum s_i^2)^2 / sum s_i^4
    """
    if k_max is not None:
        s = s_values[:k_max]
    else:
        s = s_values
    
    # delta_W: 幂律衰减率
    i_arr = np.arange(1, len(s) + 1, dtype=np.float64)
    valid = s > 0
    if np.sum(valid) > 10:
        log_i = np.log(i_arr[valid])
        log_s = np.log(s[valid])
        n_fit = int(0.8 * len(log_i))
        coeffs = np.polyfit(log_i[:n_fit], log_s[:n_fit], 1)
        delta_W = -coeffs[0]
    else:
        delta_W = 0
    
    # F: 频谱丰满度
    s2 = s ** 2
    s_max = s[0] if len(s) > 0 else 1
    F = np.sum(s2) / (len(s) * s_max ** 2) if s_max > 0 else 0
    
    # kappa: 条件数
    s_min_pos = np.min(s[s > 0]) if np.any(s > 0) else 1
    kappa = s_max / s_min_pos if s_min_pos > 0 else float('inf')
    
    # PR: 参与率
    PR = (np.sum(s2)) ** 2 / np.sum(s2 ** 2) if np.sum(s2 ** 2) > 0 else 0
    
    return delta_W, F, kappa, PR


def compute_alpha(delta_h, U_wut, s_wut, k_max):
    """计算信号聚焦参数alpha"""
    projections = U_wut[:, :k_max].T @ delta_h
    e_i = projections ** 2
    
    s_i = s_wut[:k_max]
    valid = (e_i > 0) & (s_i > 0)
    if np.sum(valid) > 10:
        log_s = np.log(s_i[valid])
        log_e = np.log(e_i[valid])
        n_fit = int(0.8 * len(log_s))
        coeffs = np.polyfit(log_s[:n_fit], log_e[:n_fit], 1)
        return coeffs[0]
    return 0.0


# ============================================================
# P472: W_down SVD谱分析
# ============================================================
def run_p472(model_name, model, tokenizer, device):
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    intermediate_size = info.intermediate_size
    layers = get_layers(model)
    
    print(f"\n  P472: W_down SVD spectral analysis - {model_name}")
    print(f"  d_model={d_model}, n_layers={n_layers}, intermediate={intermediate_size}")
    print(f"  d/n_L = {d_model/n_layers:.1f}")
    
    # 1. W_U^T SVD for alpha computation
    W_U = get_W_U(model)
    W_Ut = W_U.T
    k_max_svd = min(400, min(W_Ut.shape) - 1)
    if max(W_Ut.shape) > 50000:
        k_max_svd = min(k_max_svd, 400)
    U_wut, s_wut = safe_svd(W_Ut, k=k_max_svd)
    
    # delta of W_U^T
    i_arr = np.arange(1, k_max_svd + 1, dtype=np.float64)
    valid_sv = s_wut[:k_max_svd] > 0
    log_i = np.log(i_arr[valid_sv])
    log_s = np.log(s_wut[:k_max_svd][valid_sv])
    n_fit = int(0.8 * len(log_i))
    coeffs = np.polyfit(log_i[:n_fit], log_s[:n_fit], 1)
    delta_WU = -coeffs[0]
    print(f"  delta(W_U^T) = {delta_WU:.4f}")
    
    # 2. Attribute words for alpha
    attribute_words = ["red", "blue", "green", "yellow", "big", "small", "hot", "cold"]
    base_text = "The apple is"
    base_ids = tokenizer.encode(base_text, return_tensors="pt").to(device)
    
    # 3. Sample layers
    sampled_layers = list(range(0, n_layers, max(1, n_layers // 10)))
    
    # 4. For each layer, compute W_down SVD features and alpha
    results = []
    
    for layer_idx in sampled_layers:
        print(f"\n  Layer {layer_idx}/{n_layers-1}...")
        
        # W_down SVD
        lw = get_layer_weights(layers[layer_idx], d_model, MODEL_CONFIGS[model_name]['mlp_type'])
        W_down = lw.W_down  # [d_model, intermediate_size]
        
        # Full SVD of W_down (it's typically [d_model, intermediate_size] ~ [2560, 7168])
        # Use truncated SVD with k=min(d_model, intermediate_size)-1 for speed
        k_wd = min(min(W_down.shape) - 1, 500)
        U_wd, s_wd = safe_svd(W_down, k=k_wd)
        
        delta_W, F_W, kappa_W, PR_W = compute_spectral_features(s_wd)
        
        # Also compute W_o SVD features
        W_o = lw.W_o  # [d_model, d_model] or similar
        k_wo = min(min(W_o.shape) - 1, 200)
        if k_wo > 1:
            U_wo, s_wo = safe_svd(W_o, k=k_wo)
            delta_o, F_o, kappa_o, PR_o = compute_spectral_features(s_wo)
        else:
            delta_o, F_o, kappa_o, PR_o = 0, 0, 0, 0
        
        # Alpha computation
        with torch.no_grad():
            base_out = model(base_ids, output_hidden_states=True)
            base_h = base_out.hidden_states[layer_idx + 1][0, -1].cpu().float().numpy()
        
        alpha_avg = 0
        for attr_word in attribute_words:
            intervened_text = f"The {attr_word} apple is"
            interv_ids = tokenizer.encode(intervened_text, return_tensors="pt").to(device)
            with torch.no_grad():
                interv_out = model(interv_ids, output_hidden_states=True)
                interv_h = interv_out.hidden_states[layer_idx + 1][0, -1].cpu().float().numpy()
            delta_h = interv_h - base_h
            alpha_avg += compute_alpha(delta_h, U_wut, s_wut, k_max_svd)
        alpha_avg /= len(attribute_words)
        
        result = {
            "layer": layer_idx,
            "alpha": alpha_avg,
            "delta_WU": delta_WU,
            "delta_W": delta_W,
            "F_W": F_W,
            "kappa_W": kappa_W,
            "PR_W": PR_W,
            "delta_o": delta_o,
            "F_o": F_o,
            "kappa_o": kappa_o,
            "PR_o": PR_o,
            "d_model": d_model,
            "n_layers": n_layers,
            "intermediate_size": intermediate_size,
            "d_over_nL": d_model / n_layers,
        }
        results.append(result)
        
        if layer_idx % 5 == 0 or layer_idx == sampled_layers[-1]:
            print(f"    alpha={alpha_avg:.3f}, delta_W={delta_W:.4f}, F_W={F_W:.4f}, "
                  f"PR_W={PR_W:.1f}, kappa_W={kappa_W:.1f}")
    
    # 5. Summary
    alphas = [r["alpha"] for r in results]
    delta_Ws = [r["delta_W"] for r in results]
    F_Ws = [r["F_W"] for r in results]
    PR_Ws = [r["PR_W"] for r in results]
    
    print(f"\n  P472 Summary for {model_name}:")
    print(f"  d/n_L = {d_model/n_layers:.1f}")
    print(f"  alpha range: {min(alphas):.3f} ~ {max(alphas):.3f}")
    print(f"  delta_W range: {min(delta_Ws):.4f} ~ {max(delta_Ws):.4f}")
    print(f"  F_W range: {min(F_Ws):.4f} ~ {max(F_Ws):.4f}")
    print(f"  PR_W range: {min(PR_Ws):.1f} ~ {max(PR_Ws):.1f}")
    
    # Correlation analysis
    if len(alphas) > 3:
        corr_alpha_deltaW = np.corrcoef(alphas, delta_Ws)[0, 1]
        corr_alpha_FW = np.corrcoef(alphas, F_Ws)[0, 1]
        corr_alpha_PRW = np.corrcoef(alphas, PR_Ws)[0, 1]
        print(f"\n  Correlations with alpha:")
        print(f"    alpha vs delta_W: {corr_alpha_deltaW:.3f}")
        print(f"    alpha vs F_W:     {corr_alpha_FW:.3f}")
        print(f"    alpha vs PR_W:    {corr_alpha_PRW:.3f}")
    
    return results


# ============================================================
# P473: c/delta = f(d/n_L) 解析公式
# ============================================================
def run_p473(model_name, model, tokenizer, device):
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    layers = get_layers(model)
    
    print(f"\n  P473: c/delta = f(d/n_L) formula - {model_name}")
    print(f"  d_model={d_model}, n_layers={n_layers}, d/n_L={d_model/n_layers:.1f}")
    
    # 1. W_U^T SVD
    W_U = get_W_U(model)
    W_Ut = W_U.T
    k_max = min(400, min(W_Ut.shape) - 1)
    if max(W_Ut.shape) > 50000:
        k_max = min(k_max, 400)
    U_wut, s_wut = safe_svd(W_Ut, k=k_max)
    
    # delta of W_U^T
    i_arr = np.arange(1, k_max + 1, dtype=np.float64)
    valid_sv = s_wut[:k_max] > 0
    log_i = np.log(i_arr[valid_sv])
    log_s = np.log(s_wut[:k_max][valid_sv])
    n_fit = int(0.8 * len(log_i))
    coeffs = np.polyfit(log_i[:n_fit], log_s[:n_fit], 1)
    delta_WU = -coeffs[0]
    
    # 2. Compute alpha and gamma for all layers
    attribute_words = ["red", "blue", "green", "yellow", "big", "small", "hot", "cold"]
    base_text = "The apple is"
    base_ids = tokenizer.encode(base_text, return_tensors="pt").to(device)
    
    sampled_layers = list(range(2, n_layers - 1, max(1, n_layers // 12)))
    
    alpha_list = []
    gamma_list = []
    
    for layer_idx in sampled_layers:
        with torch.no_grad():
            base_out = model(base_ids, output_hidden_states=True)
            base_h = base_out.hidden_states[layer_idx + 1][0, -1].cpu().float().numpy()
        
        e_i_total = np.zeros(k_max)
        for attr_word in attribute_words:
            intervened_text = f"The {attr_word} apple is"
            interv_ids = tokenizer.encode(intervened_text, return_tensors="pt").to(device)
            with torch.no_grad():
                interv_out = model(interv_ids, output_hidden_states=True)
                interv_h = interv_out.hidden_states[layer_idx + 1][0, -1].cpu().float().numpy()
            delta_h = interv_h - base_h
            e_i, _, _, _ = compute_alpha_with_energy(delta_h, U_wut, s_wut, k_max)
            e_i_total += e_i
        
        # alpha
        s_i = s_wut[:k_max]
        valid = (e_i_total > 0) & (s_i > 0)
        if np.sum(valid) > 10:
            log_s = np.log(s_i[valid])
            log_e = np.log(e_i_total[valid])
            n_fit_a = int(0.8 * len(log_s))
            coeffs = np.polyfit(log_s[:n_fit_a], log_e[:n_fit_a], 1)
            alpha = coeffs[0]
        else:
            alpha = 0
        
        # gamma: ratio(k) power law
        total_energy = np.sum(e_i_total)
        cumsum = np.cumsum(e_i_total)
        ratio_k = cumsum / total_energy if total_energy > 0 else np.zeros(k_max)
        k_vals = np.arange(1, k_max + 1, dtype=float)
        valid_r = (ratio_k > 0) & (ratio_k < 1)
        if np.sum(valid_r) > 10:
            log_k = np.log(k_vals[valid_r])
            log_r = np.log(ratio_k[valid_r])
            n_fit_g = int(0.8 * len(log_k))
            coeffs = np.polyfit(log_k[:n_fit_g], log_r[:n_fit_g], 1)
            gamma = coeffs[0]
        else:
            gamma = 1.0
        
        alpha_list.append(alpha)
        gamma_list.append(gamma)
    
    alpha_arr = np.array(alpha_list)
    gamma_arr = np.array(gamma_list)
    
    # 3. Compute c2: gamma = c3 - c2*alpha
    if np.std(alpha_arr) > 1e-10:
        X = np.column_stack([-alpha_arr, np.ones(len(alpha_arr))])
        coeffs, _, _, _ = np.linalg.lstsq(X, gamma_arr, rcond=None)
        c2 = -coeffs[0]
        c3 = coeffs[1]
    else:
        c2 = delta_WU
        c3 = 1.0
    
    c_over_delta = c2 / delta_WU
    
    print(f"\n  delta = {delta_WU:.4f}")
    print(f"  c2 = {c2:.4f}")
    print(f"  c3 = {c3:.4f}")
    print(f"  c/delta = {c_over_delta:.4f}")
    print(f"  d/n_L = {d_model/n_layers:.1f}")
    
    # 4. W_down spectral features (all layers averaged)
    W_down_features = []
    for layer_idx in range(0, n_layers, max(1, n_layers // 5)):
        lw = get_layer_weights(layers[layer_idx], d_model, MODEL_CONFIGS[model_name]['mlp_type'])
        W_down = lw.W_down
        k_wd = min(min(W_down.shape) - 1, 500)
        U_wd, s_wd = safe_svd(W_down, k=k_wd)
        delta_W, F_W, kappa_W, PR_W = compute_spectral_features(s_wd)
        W_down_features.append({
            "delta_W": delta_W, "F_W": F_W, "kappa_W": kappa_W, "PR_W": PR_W
        })
    
    avg_delta_W = np.mean([f["delta_W"] for f in W_down_features])
    avg_F_W = np.mean([f["F_W"] for f in W_down_features])
    avg_PR_W = np.mean([f["PR_W"] for f in W_down_features])
    
    print(f"\n  W_down spectral features (averaged):")
    print(f"  delta_W = {avg_delta_W:.4f}")
    print(f"  F_W (fullness) = {avg_F_W:.4f}")
    print(f"  PR_W (participation) = {avg_PR_W:.1f}")
    
    # 5. Candidate formulas for c/delta
    d_nL = d_model / n_layers
    
    # Using known data from all models (hardcoded for now)
    all_models = {
        "qwen3": {"d_nL": 71.1, "c_delta": 1.09, "delta_WU": 0.175},
        "glm4": {"d_nL": 102.4, "c_delta": 1.76, "delta_WU": 0.181},
        "deepseek7b": {"d_nL": 128.0, "c_delta": 2.56, "delta_WU": 0.194},
    }
    all_models[model_name] = {"d_nL": d_nL, "c_delta": c_over_delta, "delta_WU": delta_WU}
    
    print(f"\n  Cross-model c/delta vs d/n_L:")
    d_nL_list = [all_models[m]["d_nL"] for m in all_models]
    c_delta_list = [all_models[m]["c_delta"] for m in all_models]
    
    # Fit: c/delta = a * (d/n_L)^b
    if len(d_nL_list) >= 3:
        log_d = np.log(d_nL_list)
        log_c = np.log(c_delta_list)
        coeffs_pow = np.polyfit(log_d, log_c, 1)
        b_pow = coeffs_pow[0]
        a_pow = np.exp(coeffs_pow[1])
        c_pred_pow = [a_pow * d ** b_pow for d in d_nL_list]
        ss_res = np.sum((np.array(c_delta_list) - np.array(c_pred_pow)) ** 2)
        ss_tot = np.sum((np.array(c_delta_list) - np.mean(c_delta_list)) ** 2)
        R2_pow = 1 - ss_res / ss_tot if ss_tot > 0 else -1
        print(f"  Power law: c/delta = {a_pow:.4f} * (d/n_L)^{b_pow:.3f}, R2={R2_pow:.3f}")
    
    # Fit: c/delta = a * ln(d/n_L) + b
    if len(d_nL_list) >= 2:
        X_ln = np.column_stack([np.log(d_nL_list), np.ones(len(d_nL_list))])
        coeffs_ln, _, _, _ = np.linalg.lstsq(X_ln, c_delta_list, rcond=None)
        a_ln = coeffs_ln[0]
        b_ln = coeffs_ln[1]
        c_pred_ln = [a_ln * np.log(d) + b_ln for d in d_nL_list]
        ss_res = np.sum((np.array(c_delta_list) - np.array(c_pred_ln)) ** 2)
        R2_ln = 1 - ss_res / ss_tot if ss_tot > 0 else -1
        print(f"  Log: c/delta = {a_ln:.4f} * ln(d/n_L) + {b_ln:.4f}, R2={R2_ln:.3f}")
    
    # Fit: c/delta = a * (d/n_L) + b
    X_lin = np.column_stack([np.array(d_nL_list), np.ones(len(d_nL_list))])
    coeffs_lin, _, _, _ = np.linalg.lstsq(X_lin, c_delta_list, rcond=None)
    a_lin = coeffs_lin[0]
    b_lin = coeffs_lin[1]
    c_pred_lin = [a_lin * d + b_lin for d in d_nL_list]
    ss_res = np.sum((np.array(c_delta_list) - np.array(c_pred_lin)) ** 2)
    R2_lin = 1 - ss_res / ss_tot if ss_tot > 0 else -1
    print(f"  Linear: c/delta = {a_lin:.6f} * (d/n_L) + {b_lin:.4f}, R2={R2_lin:.3f}")
    
    # Theoretical formula: c/delta = 1 + sigma_W^2 * (d/n_L) / (some constant)
    # Check W_down spectral fullness
    print(f"\n  Theoretical analysis:")
    print(f"  c/delta = 1 + amplification_factor")
    print(f"  amplification_factor = (c/delta - 1) = {c_over_delta - 1:.3f}")
    print(f"  F_W = {avg_F_W:.4f}")
    print(f"  PR_W / d_model = {avg_PR_W / d_model:.4f}")
    print(f"  sigma_W^2 * (d/n_L) = {avg_F_W * d_nL:.2f}")
    
    return {
        "model": model_name,
        "d_nL": d_nL,
        "c2": c2,
        "c3": c3,
        "c_over_delta": c_over_delta,
        "delta_WU": delta_WU,
        "delta_W": avg_delta_W,
        "F_W": avg_F_W,
        "PR_W": avg_PR_W,
    }


def compute_alpha_with_energy(delta_h, U_wut, s_wut, k_max):
    """Compute alpha and return energy distribution"""
    projections = U_wut[:, :k_max].T @ delta_h
    e_i = projections ** 2
    
    s_i = s_wut[:k_max]
    valid = (e_i > 0) & (s_i > 0)
    if np.sum(valid) > 10:
        log_s = np.log(s_i[valid])
        log_e = np.log(e_i[valid])
        n_fit = int(0.8 * len(log_s))
        coeffs = np.polyfit(log_s[:n_fit], log_e[:n_fit], 1)
        alpha = coeffs[0]
    else:
        alpha = 0
    
    return e_i, alpha, 0, 0


# ============================================================
# P474: 信号聚焦相图
# ============================================================
def run_p474(model_name, model, tokenizer, device):
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    layers = get_layers(model)
    
    print(f"\n  P474: Signal focusing phase diagram - {model_name}")
    print(f"  d_model={d_model}, n_layers={n_layers}")
    
    # 1. W_U^T SVD
    W_U = get_W_U(model)
    W_Ut = W_U.T
    k_max = min(400, min(W_Ut.shape) - 1)
    if max(W_Ut.shape) > 50000:
        k_max = min(k_max, 400)
    U_wut, s_wut = safe_svd(W_Ut, k=k_max)
    
    # delta of W_U^T
    i_arr = np.arange(1, k_max + 1, dtype=np.float64)
    valid_sv = s_wut[:k_max] > 0
    log_i = np.log(i_arr[valid_sv])
    log_s = np.log(s_wut[:k_max][valid_sv])
    n_fit = int(0.8 * len(log_i))
    coeffs = np.polyfit(log_i[:n_fit], log_s[:n_fit], 1)
    delta_WU = -coeffs[0]
    
    # 2. Compute alpha and W_down features for ALL layers
    attribute_words = ["red", "blue", "green", "yellow", "big", "small", "hot", "cold"]
    base_text = "The apple is"
    base_ids = tokenizer.encode(base_text, return_tensors="pt").to(device)
    
    all_layers_data = []
    
    for layer_idx in range(1, n_layers):
        # Alpha
        with torch.no_grad():
            base_out = model(base_ids, output_hidden_states=True)
            base_h = base_out.hidden_states[layer_idx + 1][0, -1].cpu().float().numpy()
        
        alpha_avg = 0
        for attr_word in attribute_words:
            intervened_text = f"The {attr_word} apple is"
            interv_ids = tokenizer.encode(intervened_text, return_tensors="pt").to(device)
            with torch.no_grad():
                interv_out = model(interv_ids, output_hidden_states=True)
                interv_h = interv_out.hidden_states[layer_idx + 1][0, -1].cpu().float().numpy()
            delta_h = interv_h - base_h
            alpha_avg += compute_alpha(delta_h, U_wut, s_wut, k_max)
        alpha_avg /= len(attribute_words)
        
        # W_down features (only for sampled layers to save time)
        if layer_idx % max(1, n_layers // 5) == 0:
            lw = get_layer_weights(layers[layer_idx], d_model, MODEL_CONFIGS[model_name]['mlp_type'])
            W_down = lw.W_down
            k_wd = min(min(W_down.shape) - 1, 200)
            U_wd, s_wd = safe_svd(W_down, k=k_wd)
            delta_W, F_W, kappa_W, PR_W = compute_spectral_features(s_wd)
        else:
            delta_W, F_W, kappa_W, PR_W = 0, 0, 0, 0
        
        all_layers_data.append({
            "layer": layer_idx,
            "layer_frac": layer_idx / n_layers,
            "alpha": alpha_avg,
            "delta_WU": delta_WU,
            "delta_W": delta_W,
            "F_W": F_W,
            "d_nL": d_model / n_layers,
        })
        
        if layer_idx % 5 == 0:
            print(f"    L{layer_idx}: alpha={alpha_avg:.3f}")
    
    # 3. Phase diagram analysis
    alphas = np.array([d["alpha"] for d in all_layers_data])
    layer_fracs = np.array([d["layer_frac"] for d in all_layers_data])
    
    # Find alpha=0 boundary
    alpha_sign_changes = []
    for i in range(len(alphas) - 1):
        if alphas[i] * alphas[i + 1] < 0:
            # Linear interpolation
            x0, x1 = layer_fracs[i], layer_fracs[i + 1]
            a0, a1 = alphas[i], alphas[i + 1]
            x_cross = x0 - a0 * (x1 - x0) / (a1 - a0)
            alpha_sign_changes.append(x_cross)
    
    print(f"\n  Phase diagram analysis:")
    print(f"  alpha range: {alphas.min():.3f} ~ {alphas.max():.3f}")
    print(f"  Layers with alpha>0 (focusing): {np.sum(alphas > 0)}/{len(alphas)}")
    print(f"  Layers with alpha<0 (dispersing): {np.sum(alphas < 0)}/{len(alphas)}")
    
    if alpha_sign_changes:
        print(f"  alpha=0 boundary at layer_frac: {alpha_sign_changes}")
    
    # 4. Alpha trajectory in phase space
    print(f"\n  Alpha trajectory (layer -> alpha):")
    for i in range(0, len(all_layers_data), max(1, len(all_layers_data) // 8)):
        d = all_layers_data[i]
        marker = "FOCUS" if d["alpha"] > 0 else "DISP"
        print(f"    L{d['layer']:2d} ({d['layer_frac']:.2f}): alpha={d['alpha']:+.3f} [{marker}]")
    
    # 5. Layer-wise alpha statistics
    n_focusing = np.sum(alphas > 0)
    n_dispersing = np.sum(alphas < 0)
    alpha_mean = np.mean(alphas)
    alpha_std = np.std(alphas)
    
    print(f"\n  Alpha statistics:")
    print(f"  Mean alpha = {alpha_mean:.3f}")
    print(f"  Std alpha = {alpha_std:.3f}")
    print(f"  Focusing ratio = {n_focusing/len(alphas):.2f}")
    
    # 6. Cross-model comparison
    print(f"\n  Cross-model comparison:")
    print(f"  {model_name}: d/n_L={d_model/n_layers:.1f}, mean_alpha={alpha_mean:.3f}, "
          f"focusing_ratio={n_focusing/len(alphas):.2f}")
    
    return all_layers_data


# ============================================================
# P475: 频谱丰满度与信号聚焦的因果分析
# ============================================================
def run_p475(model_name, model, tokenizer, device):
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    layers = get_layers(model)
    
    print(f"\n  P475: Causality: F_W -> alpha? - {model_name}")
    print(f"  d_model={d_model}, n_layers={n_layers}")
    
    # 1. W_U^T SVD
    W_U = get_W_U(model)
    W_Ut = W_U.T
    k_max = min(400, min(W_Ut.shape) - 1)
    if max(W_Ut.shape) > 50000:
        k_max = min(k_max, 400)
    U_wut, s_wut = safe_svd(W_Ut, k=k_max)
    
    # 2. Compute F_W and alpha for sampled layers
    attribute_words = ["red", "blue", "green", "yellow", "big", "small", "hot", "cold"]
    base_text = "The apple is"
    base_ids = tokenizer.encode(base_text, return_tensors="pt").to(device)
    
    # Sample every other layer
    sampled_layers = list(range(1, n_layers, 2))
    
    F_W_list = []
    alpha_list = []
    PR_W_list = []
    
    for layer_idx in sampled_layers:
        # F_W from W_down
        lw = get_layer_weights(layers[layer_idx], d_model, MODEL_CONFIGS[model_name]['mlp_type'])
        W_down = lw.W_down
        k_wd = min(min(W_down.shape) - 1, 300)
        U_wd, s_wd = safe_svd(W_down, k=k_wd)
        delta_W, F_W, kappa_W, PR_W = compute_spectral_features(s_wd)
        
        # Alpha from signal
        with torch.no_grad():
            base_out = model(base_ids, output_hidden_states=True)
            base_h = base_out.hidden_states[layer_idx + 1][0, -1].cpu().float().numpy()
        
        alpha_avg = 0
        for attr_word in attribute_words:
            intervened_text = f"The {attr_word} apple is"
            interv_ids = tokenizer.encode(intervened_text, return_tensors="pt").to(device)
            with torch.no_grad():
                interv_out = model(interv_ids, output_hidden_states=True)
                interv_h = interv_out.hidden_states[layer_idx + 1][0, -1].cpu().float().numpy()
            delta_h = interv_h - base_h
            alpha_avg += compute_alpha(delta_h, U_wut, s_wut, k_max)
        alpha_avg /= len(attribute_words)
        
        F_W_list.append(F_W)
        alpha_list.append(alpha_avg)
        PR_W_list.append(PR_W)
    
    F_W_arr = np.array(F_W_list)
    alpha_arr = np.array(alpha_list)
    PR_W_arr = np.array(PR_W_list)
    
    # 3. Synchronous correlation
    corr_F_alpha = np.corrcoef(F_W_arr, alpha_arr)[0, 1] if len(F_W_arr) > 2 else 0
    corr_PR_alpha = np.corrcoef(PR_W_arr, alpha_arr)[0, 1] if len(PR_W_arr) > 2 else 0
    
    print(f"\n  Synchronous correlations:")
    print(f"  Corr(F_W, alpha) = {corr_F_alpha:.3f}")
    print(f"  Corr(PR_W, alpha) = {corr_PR_alpha:.3f}")
    
    # 4. Granger causality (simplified: lagged correlation)
    # F_W(L) -> alpha(L+1)?
    if len(F_W_arr) > 3:
        corr_F_lag1 = np.corrcoef(F_W_arr[:-1], alpha_arr[1:])[0, 1]
        corr_alpha_lag1 = np.corrcoef(alpha_arr[:-1], F_W_arr[1:])[0, 1]
        
        print(f"\n  Lagged correlations (Granger-like):")
        print(f"  Corr(F_W(L), alpha(L+1)) = {corr_F_lag1:.3f}  [F -> alpha causality]")
        print(f"  Corr(alpha(L), F_W(L+1)) = {corr_alpha_lag1:.3f}  [alpha -> F causality]")
        
        if abs(corr_F_lag1) > abs(corr_alpha_lag1):
            print(f"  -> F_W(L) -> alpha(L+1) is stronger: F_W causally influences alpha")
        else:
            print(f"  -> alpha(L) -> F_W(L+1) is stronger: unlikely (weights are fixed)")
    
    # 5. Partial correlation (controlling for layer position)
    layer_pos = np.array(sampled_layers, dtype=float) / n_layers
    
    if len(F_W_arr) > 4:
        # Regress out layer position
        from numpy.linalg import lstsq
        
        X_control = np.column_stack([layer_pos, np.ones(len(layer_pos))])
        
        # Residuals of F_W after removing layer effect
        c_FW, _, _, _ = lstsq(X_control, F_W_arr, rcond=None)
        FW_resid = F_W_arr - X_control @ c_FW
        
        # Residuals of alpha after removing layer effect
        c_alpha, _, _, _ = lstsq(X_control, alpha_arr, rcond=None)
        alpha_resid = alpha_arr - X_control @ c_alpha
        
        # Partial correlation
        if np.std(FW_resid) > 1e-10 and np.std(alpha_resid) > 1e-10:
            partial_corr = np.corrcoef(FW_resid, alpha_resid)[0, 1]
        else:
            partial_corr = 0
        
        print(f"\n  Partial correlation (controlling for layer position):")
        print(f"  Corr(F_W, alpha | layer) = {partial_corr:.3f}")
    
    # 6. Summary
    print(f"\n  P475 Summary:")
    print(f"  F_W range: {F_W_arr.min():.4f} ~ {F_W_arr.max():.4f}")
    print(f"  alpha range: {alpha_arr.min():.3f} ~ {alpha_arr.max():.3f}")
    print(f"  PR_W range: {PR_W_arr.min():.1f} ~ {PR_W_arr.max():.1f}")
    print(f"  Sync Corr(F_W, alpha) = {corr_F_alpha:.3f}")
    
    if abs(corr_F_alpha) > 0.3:
        print(f"  -> F_W and alpha are correlated")
        print(f"  -> Spectral fullness of W_down is associated with signal focusing")
    else:
        print(f"  -> F_W and alpha are weakly correlated")
        print(f"  -> Other factors (beyond W_down spectral shape) determine alpha")
    
    return {
        "model": model_name,
        "corr_F_alpha": corr_F_alpha,
        "corr_PR_alpha": corr_PR_alpha,
        "F_W_range": (F_W_arr.min(), F_W_arr.max()),
        "alpha_range": (alpha_arr.min(), alpha_arr.max()),
    }


# ============================================================
# Main
# ============================================================
import torch

EXPERIMENTS = {
    "p472": run_p472,
    "p473": run_p473,
    "p474": run_p474,
    "p475": run_p475,
}


def main():
    parser = argparse.ArgumentParser(description="Phase XCVIII: W_down SVD Spectrum and Signal Focusing")
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["p472", "p473", "p474", "p475"])
    args = parser.parse_args()
    
    model_name = args.model
    experiment = args.experiment
    
    print(f"\n{'='*70}")
    print(f"  Phase XCVIII: W_down SVD Spectrum and Signal Focusing")
    print(f"  Model: {model_name}, Experiment: {experiment}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    
    try:
        result = EXPERIMENTS[experiment](model_name, model, tokenizer, device)
    finally:
        release_model(model)
    
    print(f"\n  Experiment {experiment} on {model_name} complete!")


if __name__ == "__main__":
    main()
