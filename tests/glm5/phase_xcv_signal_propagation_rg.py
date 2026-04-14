"""
Phase XCV-P460/461/462/463: 信号传播重正化群(Signal Propagation Renormalization Group)
======================================================================

核心目标: 建立信号传播方程,从重正化群(RG)角度推导alpha(L)的解析公式

关键假说: ratio(k) = int_0^k S_Delta(omega)domega
-> 核心问题: S_Delta(omega) 是什么？

P460: 信号频谱密度 S_Delta(omega) 的测量
  - Phase XCIV发现alpha是信号传播动力学效应,不是权重静态属性
  - 本实验: 直接测量每层Δ(L)的频谱密度函数
  - 方法:
    a) 对8个属性词,获取每层隐藏状态差Δ(L) = h_intervened(L) - h_base(L)
    b) 对Δ(L)在W_U^T的SVD基上分解: e_i = (u_i^T · Δ(L))²
    c) 计算频谱密度: S_Delta(omega_i) = e_i / (s_i^2) 对所有属性平均
    d) 验证 ratio(k) = sum_{i=1}^k S_Delta(omega_i) * Δomega 是否等于测量值
  - 关键: S_Delta(omega) 是否是幂律分布? S_Delta(omega) ~ omega^(-alpha)?

P461: 频谱分布假说的精确验证
  - 假说: ratio(k) = int_0^k S_Delta(omega)domega
  - 如果 S_Delta(omega) = C * omega^(alpha-1) (幂律), 则 ratio(k) = C/alpha * k^alpha
  - 这与 Phase XC 的 ratio(k) ~ k^(0.903) 一致吗?
  - 方法:
    a) 测量S_Delta(omega)的实际形状
    b) 用幂律拟合: S_Delta(omega) = C * omega^(beta-1)
    c) 计算ratio(k)的理论值: ratio_theory(k) = int_0^k C * omega^(beta-1) domega
    d) 与实测ratio(k)对比,计算R2
  - 预期: 如果S_Delta(omega)是幂律,则beta应该接近0.903

P462: 信号传播方程 Δ(L+1) = F(Δ(L), W_L, LN_L)
  - 目标: 建立层间信号传播的动力学方程
  - 方法:
    a) 测量相邻层的频谱密度: S_Delta(omega, L) 和 S_Delta(omega, L+1)
    b) 建立传播核: K(omega, omega', L) = <Δ(L+1)_omega · Δ(L)_omega'> / <Δ(L)_omega'²>
    c) 简化: 假设对角传播(频率不混合), K(omega, L) = S_Delta(omega, L+1) / S_Delta(omega, L)
    d) 检验: K(omega, L) 是否与权重矩阵的SVD有关?
  - 预期: K(omega, L) 可能是 omega 的单调函数, 类似RG的beta函数

P463: alpha(L)的RG流方程推导
  - 如果S_Delta(omega, L) = C(L) * omega^(alpha(L)-1), 则alpha的流方程:
    dalpha/dL = beta(alpha, W_L, LN_L)
  - 方法:
    a) 从P460/P462的数据提取alpha(L)
    b) 计算dalpha/dL (数值微分)
    c) 寻找beta(alpha)的函数形式: 线性? 指数? 对数?
    d) 类比统计物理: alpha是"序参量", beta是RG的beta函数
  - 预期: alpha可能有一个不动点(fixed point) alpha*,
    如果dalpha/dL > 0, alpha递增(聚焦); dalpha/dL < 0, alpha递减(分散)
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.sparse.linalg import svds
from scipy.optimize import curve_fit

# 添加项目路径
_project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, _project_root)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_utils
from model_utils import (
    load_model, get_layers, get_layer_weights, get_model_info,
    release_model, get_W_U, MODEL_CONFIGS, LayerWeights
)
import torch


def compute_spectral_density(delta_h, U_wut, s_wut, k=800):
    """
    计算信号差Δh在W_U^T SVD基上的频谱密度
    
    参数:
        delta_h: [d_model] 信号差向量
        U_wut: [d_model, k_max] W_U^T的左奇异向量
        s_wut: [k_max] W_U^T的奇异值
        k: 使用的SVD分量数
    
    返回:
        e_i: [k] 各方向投影能量
        S_omega: [k] 频谱密度 S_Delta(omega_i) = e_i / (s_i²)
    """
    k_use = min(k, U_wut.shape[1], len(s_wut))
    U_k = U_wut[:, :k_use]
    s_k = s_wut[:k_use]
    
    # 投影能量: e_i = (u_i^T · Δh)²
    projections = U_k.T @ delta_h  # [k]
    e_i = projections ** 2  # [k]
    
    # 频谱密度: S(omega_i) = e_i / s_i²
    # 归一化: s_i越大,该方向"越容易"有能量,所以要除以s_i²得到"单位频段的能量密度"
    s_sq = s_k ** 2
    s_sq[s_sq < 1e-30] = 1e-30  # 避免除零
    S_omega = e_i / s_sq
    
    return e_i, S_omega


def compute_ratio_from_spectrum(e_i, total_energy):
    """
    从频谱能量计算ratio(k)
    ratio(k) = sum_{i=1}^k e_i / total_energy
    """
    cumsum = np.cumsum(e_i)
    ratio_k = cumsum / total_energy
    return ratio_k


def fit_power_law(x, y, x_min=10):
    """
    幂律拟合: y = C * x^(beta-1)
    在log-log空间: log(y) = log(C) + (beta-1) * log(x)
    
    返回: C, beta, R2
    """
    mask = (x >= x_min) & (y > 0)
    if np.sum(mask) < 10:
        return 0, 0, 0
    
    log_x = np.log10(x[mask]).reshape(-1, 1)
    log_y = np.log10(y[mask])
    
    # 线性拟合
    A = np.hstack([log_x, np.ones_like(log_x)])
    result = np.linalg.lstsq(A, log_y, rcond=None)
    slope, intercept = result[0]
    
    beta = slope + 1  # 因为 y = C * x^(beta-1) -> slope = beta-1
    C = 10 ** intercept
    
    # R2
    y_pred = intercept + slope * log_x.flatten()
    ss_res = np.sum((log_y - y_pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    return C, beta, R2


def safe_svd(matrix, k, random_state=42):
    """Memory-safe SVD: uses randomized SVD for large matrices"""
    k = min(k, min(matrix.shape) - 1)
    if max(matrix.shape) > 50000:
        # For very large matrices, use truncated SVD with smaller k
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=k, random_state=random_state)
        # Fit on transposed if needed to reduce memory
        if matrix.shape[0] < matrix.shape[1]:
            # matrix: [d_model, vocab], fit on transpose: [vocab, d_model]
            svd.fit(matrix.T.astype(np.float32))  # Use float32 to save memory
            # svd.components_ is [k, d_model], we need U: [d_model, k]
            U = svd.components_.T.astype(np.float64)
            s = svd.singular_values_.astype(np.float64)
        else:
            svd.fit(matrix.astype(np.float32))
            U = svd.components_.T.astype(np.float64)
            s = svd.singular_values_.astype(np.float64)
        # Sort by descending singular values
        sort_idx = np.argsort(s)[::-1]
        U = U[:, sort_idx]
        s = s[sort_idx]
    else:
        U, s, _ = svds(matrix.astype(np.float64), k=k)
        s = np.sort(s)[::-1]
        sort_idx = np.argsort(s)[::-1]
        U = U[:, sort_idx]
    return U, s


def run_p460(model_name, model, tokenizer, device):
    """
    P460: 信号频谱密度 S_Delta(omega) 的测量
    核心目标: 直接测量每层Δ(L)的频谱密度函数,验证S_Delta(omega)的形状
    """
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    mlp_type = MODEL_CONFIGS[model_name]['mlp_type']
    
    print(f"\n  P460: 信号频谱密度测量 - {model_name}")
    print(f"  d_model={info.d_model}, n_layers={info.n_layers}")
    
    # 1. 获取W_U^T的SVD基
    W_U = get_W_U(model)
    W_Ut = W_U.T  # [d_model, vocab_size]
    
    k_max = min(500, min(W_Ut.shape) - 1)
    U_wut, s_wut = safe_svd(W_Ut, k=k_max)
    
    print(f"  W_U^T SVD: k_max={k_max}, s_max={s_wut[0]:.2f}, s_min={s_wut[-1]:.4f}")
    
    # 2. 属性词列表
    attribute_words = [
        "red", "blue", "green", "yellow",  # 颜色
        "big", "small", "hot", "cold",      # 大小/温度
    ]
    
    # 3. 选择层位置 (浅/中/深)
    layer_positions = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    
    # 4. 对每层测量频谱密度
    results = {
        "model": model_name,
        "d_model": d_model,
        "n_layers": n_layers,
        "k_max": k_max,
        "layers": {},
        "spectral_fits": {},
    }
    
    base_text = "The apple is"
    
    for layer_idx in layer_positions:
        print(f"\n  Layer {layer_idx}/{n_layers-1}...")
        
        base_ids = tokenizer.encode(base_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            base_out = model(base_ids, output_hidden_states=True)
            base_h = base_out.hidden_states[layer_idx + 1][0, -1].cpu().float().numpy()
        
        # 收集所有属性的频谱
        all_e_i = []
        all_S_omega = []
        
        for attr_word in attribute_words:
            intervened_text = f"The {attr_word} apple is"
            interv_ids = tokenizer.encode(intervened_text, return_tensors="pt").to(device)
            
            with torch.no_grad():
                interv_out = model(interv_ids, output_hidden_states=True)
                interv_h = interv_out.hidden_states[layer_idx + 1][0, -1].cpu().float().numpy()
            
            delta_h = interv_h - base_h
            
            # 计算频谱密度
            e_i, S_omega = compute_spectral_density(delta_h, U_wut, s_wut, k=k_max)
            all_e_i.append(e_i)
            all_S_omega.append(S_omega)
        
        # 平均频谱密度
        avg_e_i = np.mean(all_e_i, axis=0)
        avg_S_omega = np.mean(all_S_omega, axis=0)
        total_energy = np.sum(avg_e_i)
        
        # 计算ratio(k)从频谱
        ratio_from_spectrum = compute_ratio_from_spectrum(avg_e_i, total_energy)
        
        # 幂律拟合 S_Delta(omega) vs omega (用s_wut作为omega的代理)
        # S_Delta(omega_i) 应该是 omega_i = s_wut[i] 的函数
        # 但更自然的是: S_Delta(omega_i) 是频率指标 i 的函数
        # 频率指标 i 代表第i大的奇异值方向
        
        # 拟合1: S_Delta(i) = C * i^(beta-1), i是频率指标
        x_idx = np.arange(1, k_max + 1, dtype=float)
        C_fit, beta_fit, R2_fit = fit_power_law(x_idx, avg_S_omega, x_min=10)
        
        # 拟合2: e_i = C * s_i^alpha (Phase XCIII的alpha定义)
        # 在log-log: log(e_i) = alpha * log(s_i) + beta
        mask_s = (s_wut[:k_max] > 1e-6) & (avg_e_i > 0)
        if np.sum(mask_s) > 10:
            log_s = np.log10(s_wut[:k_max][mask_s]).reshape(-1, 1)
            log_e = np.log10(avg_e_i[mask_s])
            A = np.hstack([log_s, np.ones_like(log_s)])
            res = np.linalg.lstsq(A, log_e, rcond=None)
            alpha_layer = res[0][0]
            R2_alpha = 1 - np.sum((log_e - (res[0][0] * log_s.flatten() + res[0][1])) ** 2) / np.sum((log_e - np.mean(log_e)) ** 2)
        else:
            alpha_layer = 0
            R2_alpha = 0
        
        print(f"    alpha={alpha_layer:.3f}(R2={R2_alpha:.3f}), S_Delta beta={beta_fit:.3f}(R2={R2_fit:.3f})")
        print(f"    ratio(k=100)={ratio_from_spectrum[99]:.4f}, ratio(k=400)={ratio_from_spectrum[399]:.4f}, ratio(k=800)={ratio_from_spectrum[799] if k_max >= 800 else ratio_from_spectrum[-1]:.4f}")
        
        results["layers"][str(layer_idx)] = {
            "alpha": float(alpha_layer),
            "R2_alpha": float(R2_alpha),
            "S_beta": float(beta_fit),
            "S_R2": float(R2_fit),
            "S_C": float(C_fit),
            "ratio_k100": float(ratio_from_spectrum[99]),
            "ratio_k400": float(ratio_from_spectrum[399]),
            "ratio_k800": float(ratio_from_spectrum[min(799, k_max - 1)]),
            "total_energy": float(total_energy),
            "avg_e_top5": [float(x) for x in avg_e_i[:5]],
            "avg_S_top5": [float(x) for x in avg_S_omega[:5]],
        }
    
    # 5. 全局频谱拟合 (所有层平均)
    all_alpha = [results["layers"][l]["alpha"] for l in results["layers"]]
    all_beta = [results["layers"][l]["S_beta"] for l in results["layers"]]
    all_ratio800 = [results["layers"][l]["ratio_k800"] for l in results["layers"]]
    
    print(f"\n  === P460 {model_name} 全局结果 ===")
    print(f"  alpha范围: {min(all_alpha):.3f} ~ {max(all_alpha):.3f}")
    print(f"  S_Delta beta range: {min(all_beta):.3f} ~ {max(all_beta):.3f}")
    print(f"  ratio(k=800) range: {min(all_ratio800):.4f} ~ {max(all_ratio800):.4f}")
    print(f"  note: beta=0.903 means S_Delta(i)~i^(-0.097), nearly uniform")
    
    return results


def run_p461(model_name, model, tokenizer, device):
    """
    P461: 频谱分布假说的精确验证
    假说: ratio(k) = int_0^k S_Delta(omega)domega
    如果 S_Delta(omega) = C * omega^(beta-1), 则 ratio(k) = (C/beta) * k^beta
    Phase XC发现 ratio(k) ~ k^(0.903), 所以 beta应该≈0.903
    """
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    mlp_type = MODEL_CONFIGS[model_name]['mlp_type']
    
    print(f"\n  P461: 频谱分布假说验证 - {model_name}")
    print(f"  d_model={info.d_model}, n_layers={info.n_layers}")
    
    # 1. 获取W_U^T的SVD基
    W_U = get_W_U(model)
    W_Ut = W_U.T
    
    k_max = min(500, min(W_Ut.shape) - 1)
    U_wut, s_wut = safe_svd(W_Ut, k=k_max)
    
    # 2. 属性词
    attribute_words = ["red", "blue", "green", "yellow", "big", "small", "hot", "cold"]
    
    # 3. 选择中间层进行详细分析
    target_layer = n_layers // 2
    print(f"  分析层: {target_layer}")
    
    base_text = "The apple is"
    base_ids = tokenizer.encode(base_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        base_out = model(base_ids, output_hidden_states=True)
        base_h = base_out.hidden_states[target_layer + 1][0, -1].cpu().float().numpy()
    
    # 收集所有属性的频谱
    all_e_i = []
    all_S_omega = []
    
    for attr_word in attribute_words:
        intervened_text = f"The {attr_word} apple is"
        interv_ids = tokenizer.encode(intervened_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            interv_out = model(interv_ids, output_hidden_states=True)
            interv_h = interv_out.hidden_states[target_layer + 1][0, -1].cpu().float().numpy()
        
        delta_h = interv_h - base_h
        e_i, S_omega = compute_spectral_density(delta_h, U_wut, s_wut, k=k_max)
        all_e_i.append(e_i)
        all_S_omega.append(S_omega)
    
    avg_e_i = np.mean(all_e_i, axis=0)
    avg_S_omega = np.mean(all_S_omega, axis=0)
    total_energy = np.sum(avg_e_i)
    
    # 4. 实测ratio(k)
    ratio_measured = compute_ratio_from_spectrum(avg_e_i, total_energy)
    
    # 5. 从S_Delta(omega)积分推导ratio(k)
    # ratio(k) = sum_{i=1}^k e_i / sum_i₌₁^k_max e_i
    # 注意: 这不是积分,而是求和! 因为频率是离散的
    # 离散频谱: ratio(k) = sum_{i=1}^k e_i / total_energy
    # 这与实测的ratio(k)应该完全一致(因为它们是同一个东西)
    
    # 6. 但关键问题是: ratio(k)能否用幂律近似?
    # 如果 S_Delta(i) = C * i^(beta-1), 则 e_i = S_Delta(i) * s_i² 
    # ratio(k) = sum_{i=1}^k e_i / total_energy
    # ≈ (C * k^beta / beta) / (C * k_max^beta / beta) * (k/k_max)^0 ... 这不对
    
    # 正确的推导:
    # 如果 e_i = A * s_i^alpha (Phase XCIII定义)
    # 而 s_i ≈ s_1 * (i/i_max)^(1/PR_exponent)...
    # 这太复杂了,直接用数值方法
    
    # 7. 幂律拟合 ratio(k) vs k
    k_values = np.arange(1, k_max + 1, dtype=float)
    
    # 拟合 ratio(k) = C_ratio * k^gamma
    mask_k = (k_values >= 5) & (ratio_measured > 0) & (ratio_measured < 1)
    if np.sum(mask_k) > 10:
        log_k = np.log10(k_values[mask_k]).reshape(-1, 1)
        log_r = np.log10(ratio_measured[mask_k])
        A = np.hstack([log_k, np.ones_like(log_k)])
        res = np.linalg.lstsq(A, log_r, rcond=None)
        gamma_ratio = res[0][0]
        C_ratio = 10 ** res[0][1]
        R2_ratio = 1 - np.sum((log_r - (gamma_ratio * log_k.flatten() + np.log10(C_ratio))) ** 2) / np.sum((log_r - np.mean(log_r)) ** 2)
    else:
        gamma_ratio = 0
        C_ratio = 0
        R2_ratio = 0
    
    print(f"  ratio(k) power law: ratio(k) = {C_ratio:.6f} * k^{gamma_ratio:.3f}, R2={R2_ratio:.4f}")
    
    # 8. 如果 ratio(k) = C_ratio * k^gamma, 则 gamma 应该接近 0.903 (Phase XC结果)
    # 验证: ratio(k=100), ratio(k=400), ratio(k=800)
    for k_test in [50, 100, 200, 400, 800]:
        if k_test <= k_max:
            r_meas = ratio_measured[k_test - 1]
            r_theory = C_ratio * k_test ** gamma_ratio
            print(f"  ratio(k={k_test}): measured={r_meas:.4f}, theory={r_theory:.4f}, error={abs(r_meas - r_theory)/r_meas*100:.1f}%")
    
    # 9. 验证 S_Delta(i) 的幂律 vs e_i 的幂律
    # e_i = S_Delta(i) * s_i²
    # 如果 e_i ~ s_i^alpha, 则 S_Delta(i) = e_i/s_i² ~ s_i^(alpha-2)
    # 如果 s_i 近似幂律: s_i ~ i^(-delta), 则 S_Delta(i) ~ i^(-delta(alpha-2))
    
    # 拟合 e_i vs s_i 得到 alpha
    mask_s = (s_wut[:k_max] > 1e-6) & (avg_e_i > 0)
    if np.sum(mask_s) > 10:
        log_s = np.log10(s_wut[:k_max][mask_s]).reshape(-1, 1)
        log_e = np.log10(avg_e_i[mask_s])
        A = np.hstack([log_s, np.ones_like(log_s)])
        res = np.linalg.lstsq(A, log_e, rcond=None)
        alpha_layer = res[0][0]
    
    # 拟合 s_i vs i 得到 delta (奇异值谱指数)
    mask_si = s_wut[:k_max] > 1e-6
    if np.sum(mask_si) > 10:
        log_i = np.log10(k_values[mask_si]).reshape(-1, 1)
        log_sv = np.log10(s_wut[:k_max][mask_si])
        A = np.hstack([log_i, np.ones_like(log_i)])
        res = np.linalg.lstsq(A, log_sv, rcond=None)
        delta_sv = -res[0][0]  # s_i ~ i^(-delta)
    
    # 拟合 S_Delta(i) vs i 得到 beta
    mask_S = (avg_S_omega > 0) & (k_values >= 10)
    if np.sum(mask_S) > 10:
        log_i2 = np.log10(k_values[mask_S]).reshape(-1, 1)
        log_S = np.log10(avg_S_omega[mask_S])
        A = np.hstack([log_i2, np.ones_like(log_i2)])
        res = np.linalg.lstsq(A, log_S, rcond=None)
        beta_S = res[0][0] + 1  # S_Delta(i) ~ i^(beta-1)
    
    print(f"\n  === 频谱分布假说验证 ===")
    print(f"  alpha (e_i ~ s_i^alpha): {alpha_layer:.3f}")
    print(f"  delta (s_i ~ i^(-delta)): {delta_sv:.3f}")
    print(f"  beta (S_Delta(i) ~ i^(beta-1)): {beta_S:.3f}")
    print(f"  gamma (ratio(k) ~ k^gamma): {gamma_ratio:.3f}")
    
    # theory:
    # S_Delta(i) = e_i / s_i^2 ~ s_i^alpha / s_i^2 = s_i^(alpha-2)
    # if s_i ~ i^(-delta), then S_Delta(i) ~ i^(-delta*(alpha-2))
    # so beta - 1 = -delta * (alpha - 2)
    # i.e. beta = 1 - delta * (alpha - 2)
    beta_theory = 1 - delta_sv * (alpha_layer - 2)
    print(f"  beta_theory: 1 - delta*(alpha-2) = 1 - {delta_sv:.3f}*({alpha_layer:.3f}-2) = {beta_theory:.3f}")
    print(f"  beta_measured: {beta_S:.3f}, error: {abs(beta_S - beta_theory):.3f}")
    
    # gamma_theory: 
    # ratio(k) = sum_{i=1}^k e_i / sum_i e_i
    # if e_i ~ s_i^alpha ~ i^(-delta*alpha)
    # then sum_{i=1}^k e_i ~ k^(1-delta*alpha) (when 1-delta*alpha>0)
    # total energy sum_i e_i ~ k_max^(1-delta*alpha)
    # so ratio(k) ~ (k/k_max)^(1-delta*alpha)
    # i.e. gamma = 1 - delta*alpha
    gamma_theory = 1 - delta_sv * alpha_layer
    print(f"  gamma_theory: 1 - delta*alpha = 1 - {delta_sv:.3f}*{alpha_layer:.3f} = {gamma_theory:.3f}")
    print(f"  gamma_measured: {gamma_ratio:.3f}, error: {abs(gamma_ratio - gamma_theory):.3f}")
    
    # compare with Phase XC gamma=0.903
    print(f"\n  Phase XC result: gamma=0.903")
    print(f"  Current result: gamma={gamma_ratio:.3f}")
    print(f"  Difference: {abs(gamma_ratio - 0.903):.3f}")
    
    results = {
        "model": model_name,
        "layer": target_layer,
        "alpha": float(alpha_layer),
        "delta_sv": float(delta_sv),
        "beta_S": float(beta_S),
        "beta_theory": float(beta_theory),
        "gamma_ratio": float(gamma_ratio),
        "gamma_theory": float(gamma_theory),
        "C_ratio": float(C_ratio),
        "R2_ratio": float(R2_ratio),
        "ratio_k50": float(ratio_measured[49]),
        "ratio_k100": float(ratio_measured[99]),
        "ratio_k400": float(ratio_measured[399]),
        "ratio_k800": float(ratio_measured[min(799, k_max - 1)]),
    }
    
    return results


def run_p462(model_name, model, tokenizer, device):
    """
    P462: 信号传播方程 Δ(L+1) = F(Δ(L), W_L, LN_L)
    目标: 建立层间信号传播的动力学方程
    核心方法: 测量传播核 K(omega, L) = S_Delta(omega, L+1) / S_Delta(omega, L)
    """
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    mlp_type = MODEL_CONFIGS[model_name]['mlp_type']
    
    print(f"\n  P462: 信号传播方程 - {model_name}")
    print(f"  d_model={info.d_model}, n_layers={info.n_layers}")
    
    # 1. 获取W_U^T的SVD基
    W_U = get_W_U(model)
    W_Ut = W_U.T
    
    k_max = min(400, min(W_Ut.shape) - 1)  # 降低k以节省内存
    U_wut, s_wut = safe_svd(W_Ut, k=k_max)
    
    # 2. 属性词
    attribute_words = ["red", "blue", "green", "yellow", "big", "small", "hot", "cold"]
    
    # 3. 选择层对: (L, L+1)
    # 每隔4层取样,加上末层
    layer_pairs = []
    for l in range(0, n_layers - 1, max(1, n_layers // 10)):
        layer_pairs.append((l, l + 1))
    # 确保包含最后几层
    if layer_pairs[-1][1] < n_layers - 1:
        layer_pairs.append((n_layers - 2, n_layers - 1))
    
    print(f"  分析{len(layer_pairs)}个层对")
    
    base_text = "The apple is"
    
    # 4. 预计算所有层的隐藏状态
    print(f"  预计算各层隐藏状态...")
    base_ids = tokenizer.encode(base_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        base_out = model(base_ids, output_hidden_states=True)
    
    # 存储每层每个属性的频谱
    layer_spectra = {}  # layer_idx -> avg_e_i, avg_S_omega
    
    for attr_word in attribute_words:
        intervened_text = f"The {attr_word} apple is"
        interv_ids = tokenizer.encode(intervened_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            interv_out = model(interv_ids, output_hidden_states=True)
        
        for layer_idx in range(n_layers):
            base_h = base_out.hidden_states[layer_idx + 1][0, -1].cpu().float().numpy()
            interv_h = interv_out.hidden_states[layer_idx + 1][0, -1].cpu().float().numpy()
            delta_h = interv_h - base_h
            
            e_i, S_omega = compute_spectral_density(delta_h, U_wut, s_wut, k=k_max)
            
            if layer_idx not in layer_spectra:
                layer_spectra[layer_idx] = {"e_i": [], "S_omega": []}
            layer_spectra[layer_idx]["e_i"].append(e_i)
            layer_spectra[layer_idx]["S_omega"].append(S_omega)
    
    # 平均
    for l in layer_spectra:
        layer_spectra[l]["avg_e_i"] = np.mean(layer_spectra[l]["e_i"], axis=0)
        layer_spectra[l]["avg_S_omega"] = np.mean(layer_spectra[l]["S_omega"], axis=0)
    
    # 5. 计算传播核 K(i, L) = S_Delta(i, L+1) / S_Delta(i, L)
    print(f"\n  计算传播核 K(i, L)...")
    
    results = {
        "model": model_name,
        "k_max": k_max,
        "layer_pairs": {},
        "propagation_kernels": {},
    }
    
    for l, l_next in layer_pairs:
        if l not in layer_spectra or l_next not in layer_spectra:
            continue
        
        S_L = layer_spectra[l]["avg_S_omega"]
        S_L1 = layer_spectra[l_next]["avg_S_omega"]
        
        # 传播核: K(i) = S_Delta(i, L+1) / S_Delta(i, L)
        # 小值保护
        mask = (S_L > 1e-30) & (S_L1 > 1e-30)
        K = np.zeros_like(S_L)
        K[mask] = S_L1[mask] / S_L[mask]
        
        # K的统计
        K_valid = K[mask]
        if len(K_valid) > 0:
            K_mean = np.mean(K_valid)
            K_median = np.median(K_valid)
            K_std = np.std(K_valid)
            
            # K在不同频段的平均值
            n_bands = 5
            band_size = k_max // n_bands
            K_bands = []
            for b in range(n_bands):
                start = b * band_size
                end = (b + 1) * band_size
                band_mask = mask[start:end]
                if np.sum(band_mask) > 0:
                    K_bands.append(float(np.mean(K[start:end][band_mask])))
                else:
                    K_bands.append(0.0)
            
            # 拟合 K(i) = A * i^p
            i_vals = np.arange(1, k_max + 1, dtype=float)
            K_pos = K[mask]
            i_pos = i_vals[mask]
            if np.sum(K_pos > 0) > 10:
                C_K, p_K, R2_K = fit_power_law(i_pos, K_pos, x_min=10)
            else:
                C_K, p_K, R2_K = 0, 0, 0
        else:
            K_mean = K_median = K_std = 0
            K_bands = [0.0] * n_bands
            C_K, p_K, R2_K = 0, 0, 0
        
        # alpha变化
        alpha_L = compute_alpha(layer_spectra[l]["avg_e_i"], s_wut, k_max)
        alpha_L1 = compute_alpha(layer_spectra[l_next]["avg_e_i"], s_wut, k_max)
        d_alpha = alpha_L1 - alpha_L
        
        print(f"  L{l}->L{l_next}: K_mean={K_mean:.4f}, K_bands={[f'{x:.3f}' for x in K_bands]}, alpha: {alpha_L:.3f}->{alpha_L1:.3f} (dalpha={d_alpha:+.3f})")
        
        results["layer_pairs"][f"{l}_{l_next}"] = {
            "K_mean": float(K_mean),
            "K_median": float(K_median),
            "K_std": float(K_std),
            "K_bands": K_bands,
            "K_power": float(p_K),
            "K_power_R2": float(R2_K),
            "alpha_L": float(alpha_L),
            "alpha_L1": float(alpha_L1),
            "d_alpha": float(d_alpha),
        }
    
    # 6. 全局传播核分析
    all_K_means = [results["layer_pairs"][k]["K_mean"] for k in results["layer_pairs"]]
    all_d_alphas = [results["layer_pairs"][k]["d_alpha"] for k in results["layer_pairs"]]
    
    print(f"\n  === P462 {model_name} 全局结果 ===")
    print(f"  K_mean范围: {min(all_K_means):.4f} ~ {max(all_K_means):.4f}")
    print(f"  d_alpha范围: {min(all_d_alphas):.4f} ~ {max(all_d_alphas):.4f}")
    
    # K_mean > 1 意味着信号能量在传播中增强
    # K_mean < 1 意味着信号能量在传播中衰减
    # K_mean ≈ 1 意味着信号能量守恒
    
    avg_K = np.mean(all_K_means)
    print(f"  平均传播核: {avg_K:.4f}")
    if avg_K > 1:
        print(f"  -> 信号能量在传播中整体增强")
    elif avg_K < 1:
        print(f"  -> 信号能量在传播中整体衰减")
    else:
        print(f"  -> 信号能量在传播中近似守恒")
    
    # K_bands的趋势: 高频vs低频
    all_K_bands = [results["layer_pairs"][k]["K_bands"] for k in results["layer_pairs"]]
    avg_K_bands = np.mean(all_K_bands, axis=0)
    print(f"  各频段传播核: {[f'{x:.4f}' for x in avg_K_bands]}")
    print(f"  -> {'低频增强' if avg_K_bands[0] > avg_K_bands[-1] else '高频增强' if avg_K_bands[-1] > avg_K_bands[0] else '均匀传播'}")
    
    return results


def compute_alpha(e_i, s_wut, k_max):
    """计算alpha: log(e_i) = alpha * log(s_i) + beta"""
    mask = (s_wut[:k_max] > 1e-6) & (e_i[:k_max] > 0)
    if np.sum(mask) < 10:
        return 0
    log_s = np.log10(s_wut[:k_max][mask]).reshape(-1, 1)
    log_e = np.log10(e_i[:k_max][mask])
    A = np.hstack([log_s, np.ones_like(log_s)])
    res = np.linalg.lstsq(A, log_e, rcond=None)
    return float(res[0][0])


def run_p463(model_name, model, tokenizer, device):
    """
    P463: alpha(L)的RG流方程推导
    如果S_Delta(omega, L) = C(L) * omega^(alpha(L)-1), 则alpha的流方程:
    dalpha/dL = beta(alpha, W_L, LN_L)
    核心方法: 从P462的传播核数据提取alpha(L),推导RG流方程
    """
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    mlp_type = MODEL_CONFIGS[model_name]['mlp_type']
    
    print(f"\n  P463: alpha(L)的RG流方程 - {model_name}")
    print(f"  d_model={info.d_model}, n_layers={info.n_layers}")
    
    # 1. 获取W_U^T的SVD基
    W_U = get_W_U(model)
    W_Ut = W_U.T
    
    k_max = min(400, min(W_Ut.shape) - 1)
    U_wut, s_wut = safe_svd(W_Ut, k=k_max)
    
    # 2. 属性词
    attribute_words = ["red", "blue", "green", "yellow", "big", "small", "hot", "cold"]
    
    # 3. 计算所有层的alpha
    print(f"  计算所有层的alpha...")
    
    base_text = "The apple is"
    
    # 预计算base隐藏状态
    base_ids = tokenizer.encode(base_text, return_tensors="pt").to(device)
    with torch.no_grad():
        base_out = model(base_ids, output_hidden_states=True)
    
    alphas = []
    energies = []
    
    # 每3层采样
    sample_layers = list(range(0, n_layers, max(1, n_layers // 15)))
    if sample_layers[-1] != n_layers - 1:
        sample_layers.append(n_layers - 1)
    
    for layer_idx in sample_layers:
        all_e_i = []
        
        for attr_word in attribute_words:
            intervened_text = f"The {attr_word} apple is"
            interv_ids = tokenizer.encode(intervened_text, return_tensors="pt").to(device)
            
            with torch.no_grad():
                interv_out = model(interv_ids, output_hidden_states=True)
            
            base_h = base_out.hidden_states[layer_idx + 1][0, -1].cpu().float().numpy()
            interv_h = interv_out.hidden_states[layer_idx + 1][0, -1].cpu().float().numpy()
            delta_h = interv_h - base_h
            
            e_i, _ = compute_spectral_density(delta_h, U_wut, s_wut, k=k_max)
            all_e_i.append(e_i)
        
        avg_e_i = np.mean(all_e_i, axis=0)
        alpha_L = compute_alpha(avg_e_i, s_wut, k_max)
        total_e = np.sum(avg_e_i)
        
        alphas.append(alpha_L)
        energies.append(total_e)
        
        print(f"  L{layer_idx}: alpha={alpha_L:.3f}, energy={total_e:.2f}")
    
    # 4. 计算dalpha/dL (数值微分)
    alphas = np.array(alphas)
    energies = np.array(energies)
    layers_arr = np.array(sample_layers, dtype=float)
    
    # 中心差分
    d_alpha = np.zeros_like(alphas)
    for i in range(1, len(alphas) - 1):
        dL = layers_arr[i + 1] - layers_arr[i - 1]
        if dL > 0:
            d_alpha[i] = (alphas[i + 1] - alphas[i - 1]) / dL
    # 边界: 前向/后向差分
    if len(alphas) > 1:
        d_alpha[0] = (alphas[1] - alphas[0]) / (layers_arr[1] - layers_arr[0]) if (layers_arr[1] - layers_arr[0]) > 0 else 0
        d_alpha[-1] = (alphas[-1] - alphas[-2]) / (layers_arr[-1] - layers_arr[-2]) if (layers_arr[-1] - layers_arr[-2]) > 0 else 0
    
    # 5. 拟合 dalpha/dL = f(alpha)
    # 模型1: 线性 dalpha/dL = a + b*alpha
    # 模型2: 指数趋近 dalpha/dL = -lambda*(alpha - alpha*)  (不动点)
    # 模型3: 对数 dalpha/dL = a*log(alpha) + b
    
    valid = (np.abs(d_alpha) < 10) & (alphas > 0)  # 去掉异常值
    
    if np.sum(valid) > 5:
        # 模型1: 线性
        A1 = np.vstack([alphas[valid], np.ones(np.sum(valid))]).T
        res1 = np.linalg.lstsq(A1, d_alpha[valid], rcond=None)
        a1, b1 = res1[0]
        
        # 模型2: 指数趋近不动点
        # dalpha/dL = -lambda*(alpha - alpha*) -> 如果a1 < 0, 则不动点 alpha* = -b1/a1
        if a1 < 0:
            alpha_star = -b1 / a1
            lam = -a1
        else:
            alpha_star = None
            lam = 0
        
        # R2 for linear model
        pred1 = a1 * alphas[valid] + b1
        ss_res = np.sum((d_alpha[valid] - pred1) ** 2)
        ss_tot = np.sum((d_alpha[valid] - np.mean(d_alpha[valid])) ** 2)
        R2_linear = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # 模型3: 二次 dalpha/dL = a + b*alpha + c*alpha²
        A3 = np.vstack([alphas[valid] ** 2, alphas[valid], np.ones(np.sum(valid))]).T
        res3 = np.linalg.lstsq(A3, d_alpha[valid], rcond=None)
        c3, b3, a3 = res3[0]
        
        pred3 = c3 * alphas[valid] ** 2 + b3 * alphas[valid] + a3
        ss_res3 = np.sum((d_alpha[valid] - pred3) ** 2)
        R2_quadratic = 1 - ss_res3 / ss_tot if ss_tot > 0 else 0
        
        # 二次模型的不动点: a + b*alpha + c*alpha² = 0
        # alpha* = (-b ± sqrt(b²-4ac)) / (2c)
        discriminant = b3 ** 2 - 4 * c3 * a3
        if discriminant > 0 and abs(c3) > 1e-10:
            fp1 = (-b3 + np.sqrt(discriminant)) / (2 * c3)
            fp2 = (-b3 - np.sqrt(discriminant)) / (2 * c3)
        else:
            fp1 = fp2 = None
    else:
        a1 = b1 = c3 = b3 = a3 = 0
        R2_linear = R2_quadratic = 0
        alpha_star = None
        fp1 = fp2 = None
    
    print(f"\n  === P463 {model_name} RG流方程结果 ===")
    print(f"  alpha范围: {alphas.min():.3f} ~ {alphas.max():.3f}")
    print(f"  dalpha/dL范围: {d_alpha[valid].min():.4f} ~ {d_alpha[valid].max():.4f}" if np.sum(valid) > 0 else "  无有效数据")
    print(f"\n  模型1(线性): dalpha/dL = {a1:.4f} + {b1:.4f}*alpha, R2={R2_linear:.4f}")
    if alpha_star is not None:
        print(f"  -> 不动点: alpha* = {alpha_star:.3f}, 趋近速率: lambda = {lam:.4f}")
        print(f"  -> 如果alpha < alpha*, dalpha/dL > 0 (alpha递增趋向alpha*)")
        print(f"  -> 如果alpha > alpha*, dalpha/dL < 0 (alpha递减趋向alpha*)")
    print(f"\n  model3(quadratic): dalpha/dL = {a3:.4f} + {b3:.4f}*alpha + {c3:.4f}*alpha^2, R2={R2_quadratic:.4f}")
    if fp1 is not None:
        print(f"  -> 不动点: alpha1* = {fp1:.3f}, alpha2* = {fp2:.3f}")
        # 稳定性分析: 不动点处d2alpha/dalpha2 = b3 + 2*c3*alpha*
        for fp in [fp1, fp2]:
            stability = b3 + 2 * c3 * fp
            print(f"  -> alpha*={fp:.3f}: 稳定性 = {stability:.4f} ({'稳定' if stability < 0 else '不稳定'})")
    
    # 6. alpha与能量关系
    if np.sum(valid) > 3:
        # log(energy) vs alpha
        log_e = np.log10(energies[valid])
        A_e = np.vstack([alphas[valid], np.ones(np.sum(valid))]).T
        res_e = np.linalg.lstsq(A_e, log_e, rcond=None)
        slope_e, intercept_e = res_e[0]
        print(f"\n  alpha与能量关系: log10(energy) = {slope_e:.3f}*alpha + {intercept_e:.3f}")
    
    results = {
        "model": model_name,
        "n_layers": n_layers,
        "sample_layers": sample_layers,
        "alphas": [float(x) for x in alphas],
        "d_alpha": [float(x) for x in d_alpha],
        "energies": [float(x) for x in energies],
        "linear_model": {"a": float(a1), "b": float(b1), "R2": float(R2_linear)},
        "quadratic_model": {"a": float(a3), "b": float(b3), "c": float(c3), "R2": float(R2_quadratic)},
        "alpha_star_linear": float(alpha_star) if alpha_star is not None else None,
        "fixed_points_quadratic": [float(fp1) if fp1 is not None else None, float(fp2) if fp2 is not None else None],
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase XCV: 信号传播重正化群")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["qwen3", "glm4", "deepseek7b"],
                       help="模型名称")
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["p460", "p461", "p462", "p463"],
                       help="实验编号")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Phase XCV: 信号传播重正化群")
    print(f"模型: {args.model}, 实验: {args.experiment}")
    print(f"{'='*60}")
    
    # 加载模型
    print(f"\n加载模型 {args.model}...")
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    print(f"  d_model={info.d_model}, n_layers={info.n_layers}")
    
    # 运行实验
    if args.experiment == "p460":
        results = run_p460(args.model, model, tokenizer, device)
    elif args.experiment == "p461":
        results = run_p461(args.model, model, tokenizer, device)
    elif args.experiment == "p462":
        results = run_p462(args.model, model, tokenizer, device)
    elif args.experiment == "p463":
        results = run_p463(args.model, model, tokenizer, device)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(__file__).resolve().parent.parent / "glm5_temp"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"phase_xcv_{args.experiment}_{args.model}_{timestamp}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存: {output_file}")
    
    # 释放模型
    release_model(model)
    print("模型已释放")


if __name__ == "__main__":
    main()
