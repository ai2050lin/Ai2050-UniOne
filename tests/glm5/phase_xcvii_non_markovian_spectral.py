"""
Phase XCVII-P468/469/470/471: 非马尔可夫频谱演化
======================================================================

核心目标: 解释为什么简单Fokker-Planck失败,建立正确的频谱演化方程

Phase XCVI核心成果:
1. gamma=1-c*alpha, c是模型相关有效耦合常数(0.15-0.55)
2. gamma=1-delta*alpha只是零阶近似, 在GLM4/DS7B上R2<0
3. Fokker-Planck形式失败(R2为负), D2为负(物理不合理)
4. alpha*候选公式: 1/(1-delta), 对GLM4/DS7B误差0.15-0.21

关键问题:
1. 为什么c >> delta? 信号在传播中被重复放大(5-18倍)
2. Fokker-Planck为什么失败? 频谱演化不是马尔可夫过程
3. c的解析表达式是什么?

Phase XCVII目标:
1. 检验记忆核: G(L, L') = dS(L)/dS(L') (层间频谱关联)
2. 建立积分微分方程: dS/dL = int G(L,L')*S(L')dL'
3. 推导c = delta * f(n_layers, d_model) 的解析形式
4. 跨模型统一gamma公式

P468: 记忆核测量 - 层间频谱关联
  - 核心假设: 频谱演化有记忆效应, S(L)依赖于S(L'), L'<L
  - 方法:
    a) 测量传播核K(i,L) = S(i,L+1)/S(i,L) (已做)
    b) 测量记忆核G(L1,L2) = Cov(dS/dL|L1, dS/dL|L2) / Var(dS/dL|L2)
    c) 如果G(L1,L2)在|L1-L2|>1时不为零, 则有记忆效应
    d) 测量记忆长度 xi (关联长度)
  - 预期: 记忆长度 xi 可能与层数/模型规模有关

P469: 积分微分演化方程
  - 目标: 建立包含记忆效应的演化方程
  - 方法:
    a) dS(i,L)/dL = sum_{L'<L} K(L,L') * F(S(i,L'), W_L')
    b) 简化: 假设核K(L,L') = exp(-|L-L'|/xi)
    c) 拟合记忆长度xi
    d) 验证积分方程的预测精度
  - 预期: 含记忆的方程R2 >> Fokker-Planck的R2

P470: 有效耦合常数c的解析推导
  - 目标: 从模型参数推导c = delta * f(n_layers, d_model)
  - 方法:
    a) 测量三模型的c: Qwen3=0.15, GLM4=0.245, DS7B=0.553
    b) 测量三模型的delta: ~0.17
    c) c/delta比值: Qwen3=0.87, GLM4=1.46, DS7B=3.12
    d) 检验c/delta是否与n_layers, d_model, 或其他参数有关
  - 候选公式:
    - c = delta * n_layers / L_eff (L_eff是有效传播长度)
    - c = delta * (1 + sigma_W^2 * d_model / n_layers) (权重方差修正)
    - c = delta * (1 + alpha_cum) (累积聚焦修正)

P471: 跨模型统一gamma公式
  - 目标: gamma = f(alpha, delta, d_model, n_layers) 的完整形式
  - 方法:
    a) 收集三模型所有层的(alpha, delta, gamma)数据
    b) 拟合gamma = 1 - c(delta, d_model, n_layers) * alpha
    c) 验证c = delta * f(d_model, n_layers)
    d) 检验gamma的更高阶修正
  - 预期: 得到跨模型统一的gamma公式
"""

import sys
import os
import argparse
import numpy as np
from scipy.sparse.linalg import svds
from scipy.optimize import curve_fit

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_utils import (
    load_model, get_layers, get_layer_weights, get_model_info,
    get_W_U, release_model, MODEL_CONFIGS
)


def safe_svd(matrix, k, random_state=42):
    """Memory-safe SVD: uses TruncatedSVD for large matrices"""
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


def compute_spectral_density(delta_h, U_wut, s_wut, k_max):
    """
    计算信号在W_U^T SVD基上的频谱密度
    
    Returns:
        e_i: 投影能量 [k_max]
        S_delta: 频谱密度 S(i) = e_i / s_i^2 [k_max]
        alpha: e_i ~ s_i^alpha 的幂律指数
        delta_sv: s_i ~ i^(-delta) 的衰减率
    """
    # 投影
    projections = U_wut[:, :k_max].T @ delta_h  # [k_max]
    e_i = projections ** 2  # 投影能量
    
    # 频谱密度
    s_i = s_wut[:k_max]
    s_i_safe = np.maximum(s_i, 1e-10)
    S_delta = e_i / s_i_safe ** 2
    
    # alpha: e_i ~ s_i^alpha
    valid = (e_i > 0) & (s_i > 0)
    if np.sum(valid) > 10:
        log_s = np.log(s_i[valid])
        log_e = np.log(e_i[valid])
        # 使用前80%的数据拟合
        n_fit = int(0.8 * len(log_s))
        coeffs = np.polyfit(log_s[:n_fit], log_e[:n_fit], 1)
        alpha = coeffs[0]
    else:
        alpha = 0.0
    
    # delta: s_i ~ i^(-delta)
    i_arr = np.arange(1, k_max + 1, dtype=np.float64)
    valid_sv = s_i > 0
    if np.sum(valid_sv) > 10:
        log_i = np.log(i_arr[valid_sv])
        log_s_v = np.log(s_i[valid_sv])
        n_fit = int(0.8 * len(log_i))
        coeffs = np.polyfit(log_i[:n_fit], log_s_v[:n_fit], 1)
        delta_sv = -coeffs[0]
    else:
        delta_sv = 0.15
    
    return e_i, S_delta, alpha, delta_sv


def compute_ratio_k(e_i, k_max):
    """计算ratio(k) = sum_{i=1}^k e_i / total_energy"""
    total = np.sum(e_i)
    if total == 0:
        return np.zeros(k_max)
    cumsum = np.cumsum(e_i)
    return cumsum / total


def power_law_fit(x, y, n_fit=None):
    """幂律拟合 y = C * x^beta"""
    valid = (x > 0) & (y > 0)
    if np.sum(valid) < 5:
        return 0, 0, -1
    log_x = np.log(x[valid])
    log_y = np.log(y[valid])
    if n_fit is not None:
        log_x = log_x[:n_fit]
        log_y = log_y[:n_fit]
    coeffs = np.polyfit(log_x, log_y, 1)
    beta = coeffs[0]
    C = np.exp(coeffs[1])
    # R2
    y_pred = C * x[valid][:len(log_x)] ** beta
    ss_res = np.sum((y[valid][:len(log_x)] - y_pred) ** 2)
    ss_tot = np.sum((y[valid][:len(log_x)] - np.mean(y[valid][:len(log_x)])) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else -1
    return C, beta, R2


# ============================================================
# P468: 记忆核测量 - 层间频谱关联
# ============================================================
def run_p468(model_name, model, tokenizer, device):
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    layers = get_layers(model)
    
    print(f"\n  P468: Memory kernel measurement - {model_name}")
    print(f"  d_model={d_model}, n_layers={n_layers}")
    
    # 1. W_U^T SVD
    W_U = get_W_U(model)
    W_Ut = W_U.T
    k_max = min(400, min(W_Ut.shape) - 1)
    if max(W_Ut.shape) > 50000:
        k_max = min(k_max, 400)
    
    U_wut, s_wut = safe_svd(W_Ut, k=k_max)
    print(f"  W_U^T SVD: k_max={k_max}")
    
    # 2. 属性词
    attribute_words = ["red", "blue", "green", "yellow", "big", "small", "hot", "cold"]
    base_text = "The apple is"
    base_ids = tokenizer.encode(base_text, return_tensors="pt").to(device)
    
    # 3. 计算所有层的频谱密度和dS/dL
    # 采样层: 每2层采一个
    sampled_layers = list(range(1, n_layers, 2))
    n_sampled = len(sampled_layers)
    
    S_all = []  # [n_sampled, k_max]
    alpha_all = []
    delta_all = []
    
    for idx, layer_idx in enumerate(sampled_layers):
        with torch.no_grad():
            base_out = model(base_ids, output_hidden_states=True)
            base_h = base_out.hidden_states[layer_idx + 1][0, -1].cpu().float().numpy()
        
        e_i_avg = np.zeros(k_max)
        for attr_word in attribute_words:
            intervened_text = f"The {attr_word} apple is"
            interv_ids = tokenizer.encode(intervened_text, return_tensors="pt").to(device)
            
            with torch.no_grad():
                interv_out = model(interv_ids, output_hidden_states=True)
                interv_h = interv_out.hidden_states[layer_idx + 1][0, -1].cpu().float().numpy()
            
            delta_h = interv_h - base_h
            e_i, S_delta, alpha, delta_sv = compute_spectral_density(
                delta_h, U_wut, s_wut, k_max)
            e_i_avg += e_i
        
        e_i_avg /= len(attribute_words)
        S_avg = e_i_avg / np.maximum(s_wut[:k_max], 1e-10) ** 2
        S_all.append(S_avg)
        
        # Compute alpha for this layer
        _, _, alpha_l, delta_l = compute_spectral_density(delta_h, U_wut, s_wut, k_max)
        alpha_all.append(alpha_l)
        delta_all.append(delta_l)
        
        if idx % 5 == 0:
            print(f"    Layer {layer_idx}: alpha={alpha_l:.3f}, delta={delta_l:.3f}")
    
    S_all = np.array(S_all)  # [n_sampled, k_max]
    alpha_all = np.array(alpha_all)
    delta_all = np.array(delta_all)
    
    # 4. 计算dS/dL (中心差分)
    dS_dL = np.zeros_like(S_all)
    for i in range(n_sampled):
        if i == 0:
            dS_dL[i] = S_all[1] - S_all[0]
        elif i == n_sampled - 1:
            dS_dL[i] = S_all[-1] - S_all[-2]
        else:
            dS_dL[i] = (S_all[i + 1] - S_all[i - 1]) / 2
    
    # 5. 计算记忆核 G(L1, L2)
    # G(L1, L2) = <dS/dL(L1) * dS/dL(L2)> / <dS/dL(L2)^2>
    # 在频谱维度上平均
    print(f"\n  Computing memory kernel G(L1, L2)...")
    
    # 使用log频谱密度(更稳定)
    dS_log = np.log10(np.maximum(S_all, 1e-20))
    dS_log_dL = np.zeros_like(dS_log)
    for i in range(n_sampled):
        if i == 0:
            dS_log_dL[i] = dS_log[1] - dS_log[0]
        elif i == n_sampled - 1:
            dS_log_dL[i] = dS_log[-1] - dS_log[-2]
        else:
            dS_log_dL[i] = (dS_log[i + 1] - dS_log[i - 1]) / 2
    
    # 对前50个频段计算关联
    n_freq = min(50, k_max)
    G_matrix = np.zeros((n_sampled, n_sampled))
    
    for i in range(n_sampled):
        for j in range(n_sampled):
            # 在频段维度上的相关系数
            x = dS_log_dL[i, :n_freq]
            y = dS_log_dL[j, :n_freq]
            if np.std(y) > 1e-10:
                G_matrix[i, j] = np.corrcoef(x, y)[0, 1]
            else:
                G_matrix[i, j] = 0
    
    # 6. 分析记忆长度
    # 对角线附近的关联衰减
    print(f"\n  Memory kernel analysis:")
    
    # 计算对角线平均 G(d) = mean(G(i, i+d))
    max_lag = min(10, n_sampled // 2)
    G_diag = np.zeros(max_lag)
    G_diag_count = np.zeros(max_lag)
    
    for d in range(max_lag):
        for i in range(n_sampled - d):
            G_diag[d] += abs(G_matrix[i, i + d])
            G_diag_count[d] += 1
        if G_diag_count[d] > 0:
            G_diag[d] /= G_diag_count[d]
    
    # 拟合指数衰减 G(d) ~ exp(-d/xi)
    if G_diag[0] > 0 and len(G_diag) > 2:
        # Normalize
        G_diag_norm = G_diag / G_diag[0]
        d_arr = np.arange(max_lag, dtype=float)
        valid = G_diag_norm > 0
        if np.sum(valid) > 2:
            log_G = np.log(np.maximum(G_diag_norm[valid], 1e-10))
            coeffs = np.polyfit(d_arr[valid], log_G, 1)
            xi_memory = -1.0 / coeffs[0] if coeffs[0] < 0 else float('inf')
        else:
            xi_memory = float('inf')
    else:
        xi_memory = 0
    
    print(f"  G(d=0)={G_diag[0]:.3f}")
    print(f"  G(d=1)={G_diag[1]:.3f}" if max_lag > 1 else "")
    print(f"  G(d=2)={G_diag[2]:.3f}" if max_lag > 2 else "")
    print(f"  Memory length xi={xi_memory:.2f} layers")
    
    # 7. alpha的层间关联
    print(f"\n  Alpha layer correlations:")
    if len(alpha_all) > 3:
        # alpha的自相关
        alpha_centered = alpha_all - np.mean(alpha_all)
        if np.std(alpha_all) > 1e-10:
            for lag in range(min(4, len(alpha_all) // 2)):
                if lag == 0:
                    ac = 1.0
                else:
                    ac = np.corrcoef(alpha_centered[:-lag], alpha_centered[lag:])[0, 1]
                print(f"    Autocorr(lag={lag}) = {ac:.3f}")
    
    # 8. 频段间的记忆效应
    print(f"\n  Cross-frequency memory:")
    # 对L和L+1, 检验S(i,L+1)是否与S(j,L)关联(i!=j)
    if n_sampled > 2:
        # 使用中间层
        mid = n_sampled // 2
        S_L = dS_log[mid]
        S_L1 = dS_log[mid + 1] if mid + 1 < n_sampled else dS_log[mid]
        
        # 对不同频段间隔的关联
        for freq_gap in [0, 1, 5, 10, 20]:
            n_pairs = n_freq - freq_gap
            if n_pairs > 0 and freq_gap < n_freq:
                x = S_L1[:n_pairs]
                y = S_L[freq_gap:freq_gap + n_pairs]
                corr = np.corrcoef(x, y)[0, 1] if np.std(y) > 1e-10 else 0
                print(f"    Corr(S(i,L+1), S(i+{freq_gap},L)) = {corr:.3f}")
    
    # 9. 保存结果
    results = {
        "model": model_name,
        "n_sampled": n_sampled,
        "xi_memory": xi_memory,
        "G_diag": G_diag.tolist(),
        "alpha_all": alpha_all.tolist(),
        "delta_all": delta_all.tolist(),
        "G_matrix_diag5": G_matrix[:5, :5].tolist() if n_sampled >= 5 else [],
    }
    
    print(f"\n  P468 Summary:")
    print(f"  Memory length xi = {xi_memory:.2f} layers")
    print(f"  Alpha range: {min(alpha_all):.3f} ~ {max(alpha_all):.3f}")
    print(f"  If xi > 1: non-Markovian process (memory effects present)")
    print(f"  If xi <= 1: approximately Markovian")
    
    return results


# ============================================================
# P469: 积分微分演化方程
# ============================================================
def run_p469(model_name, model, tokenizer, device):
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    
    print(f"\n  P469: Integral-differential evolution equation - {model_name}")
    print(f"  d_model={d_model}, n_layers={n_layers}")
    
    # 1. W_U^T SVD
    W_U = get_W_U(model)
    W_Ut = W_U.T
    k_max = min(400, min(W_Ut.shape) - 1)
    if max(W_Ut.shape) > 50000:
        k_max = min(k_max, 400)
    
    U_wut, s_wut = safe_svd(W_Ut, k=k_max)
    
    # 2. 属性词
    attribute_words = ["red", "blue", "green", "yellow", "big", "small", "hot", "cold"]
    base_text = "The apple is"
    base_ids = tokenizer.encode(base_text, return_tensors="pt").to(device)
    
    # 3. 计算所有层的频谱密度
    sampled_layers = list(range(1, n_layers, 2))
    n_sampled = len(sampled_layers)
    
    S_all = []  # log spectral density
    
    for layer_idx in sampled_layers:
        with torch.no_grad():
            base_out = model(base_ids, output_hidden_states=True)
            base_h = base_out.hidden_states[layer_idx + 1][0, -1].cpu().float().numpy()
        
        e_i_avg = np.zeros(k_max)
        for attr_word in attribute_words:
            intervened_text = f"The {attr_word} apple is"
            interv_ids = tokenizer.encode(intervened_text, return_tensors="pt").to(device)
            
            with torch.no_grad():
                interv_out = model(interv_ids, output_hidden_states=True)
                interv_h = interv_out.hidden_states[layer_idx + 1][0, -1].cpu().float().numpy()
            
            delta_h = interv_h - base_h
            e_i, _, _, _ = compute_spectral_density(delta_h, U_wut, s_wut, k_max)
            e_i_avg += e_i
        
        e_i_avg /= len(attribute_words)
        S_avg = e_i_avg / np.maximum(s_wut[:k_max], 1e-10) ** 2
        S_all.append(np.log10(np.maximum(S_avg, 1e-20)))
    
    S_all = np.array(S_all)  # [n_sampled, k_max]
    
    # 4. dS/dL
    dS_dL = np.zeros_like(S_all)
    for i in range(n_sampled):
        if i == 0:
            dS_dL[i] = S_all[1] - S_all[0]
        elif i == n_sampled - 1:
            dS_dL[i] = S_all[-1] - S_all[-2]
        else:
            dS_dL[i] = (S_all[i + 1] - S_all[i - 1]) / 2
    
    # 5. 模型1: 马尔可夫模型 (Fokker-Planck)
    # dS/dL(L) = a * S(L) + b (only depends on current state)
    print(f"\n  Model 1: Markov (dS/dL = a*S + b)")
    
    # 使用前30个频段
    n_freq = min(30, k_max)
    
    R2_markov_all = []
    for freq in range(n_freq):
        y = dS_dL[:, freq]
        X = S_all[:, freq]
        # Linear fit
        if np.std(X) > 1e-10 and np.std(y) > 1e-10:
            coeffs = np.polyfit(X, y, 1)
            y_pred = np.polyval(coeffs, X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            R2 = 1 - ss_res / ss_tot if ss_tot > 0 else -1
            R2_markov_all.append(R2)
    
    R2_markov = np.mean(R2_markov_all) if R2_markov_all else -1
    print(f"    Average R2 = {R2_markov:.3f}")
    
    # 6. 模型2: 单步记忆 (AR2)
    # dS/dL(L) = a * S(L) + b * S(L-1) + c
    print(f"\n  Model 2: Single-step memory (AR2: dS/dL = a*S(L) + b*S(L-1) + c)")
    
    R2_ar2_all = []
    for freq in range(n_freq):
        y = dS_dL[1:, freq]
        X = np.column_stack([S_all[1:, freq], S_all[:-1, freq]])
        # Add constant
        X_aug = np.column_stack([X, np.ones(len(y))])
        # Least squares
        coeffs, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        y_pred = X_aug @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else -1
        R2_ar2_all.append(R2)
    
    R2_ar2 = np.mean(R2_ar2_all) if R2_ar2_all else -1
    print(f"    Average R2 = {R2_ar2:.3f}")
    
    # 7. 模型3: 指数记忆核
    # dS/dL(L) = int_0^L exp(-(L-L')/xi) * [a*S(L') + b] dL'
    print(f"\n  Model 3: Exponential memory kernel")
    print(f"    dS/dL(L) = sum_{{L'<L}} exp(-(L-L')/xi) * [a*S(L') + b]")
    
    best_xi = 1.0
    best_R2 = -1
    
    for xi in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0]:
        R2_xi_all = []
        for freq in range(n_freq):
            y = dS_dL[:, freq]
            # Build feature matrix with exponential kernel
            n = len(y)
            X_mem = np.zeros(n)
            for L in range(n):
                for Lp in range(L):
                    weight = np.exp(-(L - Lp) / xi)
                    X_mem[L] += weight * S_all[Lp, freq]
            
            if np.std(X_mem) > 1e-10 and np.std(y) > 1e-10:
                coeffs = np.polyfit(X_mem, y, 1)
                y_pred = np.polyval(coeffs, X_mem)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                R2 = 1 - ss_res / ss_tot if ss_tot > 0 else -1
                R2_xi_all.append(R2)
        
        R2_avg = np.mean(R2_xi_all) if R2_xi_all else -1
        if R2_avg > best_R2:
            best_R2 = R2_avg
            best_xi = xi
        print(f"    xi={xi:.1f}: R2={R2_avg:.3f}")
    
    print(f"  Best xi = {best_xi:.1f}, R2 = {best_R2:.3f}")
    
    # 8. 模型4: 完全积分方程
    # dS/dL(L) = sum_{L'<L} K(L-L') * a * S(L')
    # K(d) = exp(-d/xi) * d^(-theta) (stretched exponential or power-law kernel)
    print(f"\n  Model 4: Power-law memory kernel")
    print(f"    K(d) = exp(-d/xi) * d^(-theta)")
    
    best_theta = 0
    best_R2_4 = best_R2
    best_xi_4 = best_xi
    
    for xi in [best_xi, 5.0, 10.0]:
        for theta in [0.0, 0.3, 0.5, 0.8, 1.0]:
            R2_all = []
            for freq in range(n_freq):
                y = dS_dL[:, freq]
                n = len(y)
                X_mem = np.zeros(n)
                for L in range(n):
                    for Lp in range(L):
                        d = L - Lp
                        if d > 0:
                            weight = np.exp(-d / xi) * (d ** (-theta)) if theta > 0 else np.exp(-d / xi)
                            X_mem[L] += weight * S_all[Lp, freq]
                
                if np.std(X_mem) > 1e-10 and np.std(y) > 1e-10:
                    coeffs = np.polyfit(X_mem, y, 1)
                    y_pred = np.polyval(coeffs, X_mem)
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else -1
                    R2_all.append(R2)
            
            R2_avg = np.mean(R2_all) if R2_all else -1
            if R2_avg > best_R2_4:
                best_R2_4 = R2_avg
                best_xi_4 = xi
                best_theta = theta
    
    print(f"  Best: xi={best_xi_4:.1f}, theta={best_theta:.1f}, R2={best_R2_4:.3f}")
    
    # 9. 模型5: alpha演化方程(非马尔可夫)
    # dalpha/dL = sum w(L') * alpha(L') + b
    print(f"\n  Model 5: Alpha evolution with memory")
    
    # 计算所有层的alpha
    alpha_all = []
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
            e_i, _, _, _ = compute_spectral_density(delta_h, U_wut, s_wut, k_max)
            e_i_total += e_i
            e_i_total += e_i
        
        # Average alpha
        s_i = s_wut[:k_max]
        valid = (e_i_total > 0) & (s_i > 0)
        if np.sum(valid) > 10:
            log_s = np.log(s_i[valid])
            log_e = np.log(e_i_total[valid])
            n_fit = int(0.8 * len(log_s))
            coeffs = np.polyfit(log_s[:n_fit], log_e[:n_fit], 1)
            alpha_all.append(coeffs[0])
        else:
            alpha_all.append(0)
    
    alpha_all = np.array(alpha_all)
    dalpha_dL = np.gradient(alpha_all)
    
    # AR(1): dalpha/dL = a*alpha + b
    if len(alpha_all) > 3 and np.std(alpha_all) > 1e-10:
        X = alpha_all
        y = dalpha_dL
        coeffs_ar1 = np.polyfit(X, y, 1)
        y_pred = np.polyval(coeffs_ar1, X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        R2_ar1 = 1 - ss_res / ss_tot if ss_tot > 0 else -1
    else:
        R2_ar1 = -1
        coeffs_ar1 = [0, 0]
    
    # AR(2): dalpha/dL = a1*alpha(L) + a2*alpha(L-1) + b
    if len(alpha_all) > 4:
        X = np.column_stack([alpha_all[1:], alpha_all[:-1]])
        y = dalpha_dL[1:]
        X_aug = np.column_stack([X, np.ones(len(y))])
        coeffs_ar2, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        y_pred = X_aug @ coeffs_ar2
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        R2_ar2_alpha = 1 - ss_res / ss_tot if ss_tot > 0 else -1
    else:
        R2_ar2_alpha = -1
    
    print(f"    AR(1): dalpha/dL = {coeffs_ar1[0]:.3f}*alpha + {coeffs_ar1[1]:.3f}, R2={R2_ar1:.3f}")
    print(f"    AR(2): R2={R2_ar2_alpha:.3f}")
    
    # 10. 总结
    print(f"\n  P469 Summary:")
    print(f"  Markov (Fokker-Planck): R2={R2_markov:.3f}")
    print(f"  Single-step memory (AR2): R2={R2_ar2:.3f}")
    print(f"  Exponential kernel (xi={best_xi:.1f}): R2={best_R2:.3f}")
    print(f"  Power-law kernel (xi={best_xi_4:.1f}, theta={best_theta:.1f}): R2={best_R2_4:.3f}")
    print(f"  Alpha AR(1): R2={R2_ar1:.3f}")
    print(f"  Alpha AR(2): R2={R2_ar2_alpha:.3f}")
    
    best_overall = max(R2_markov, R2_ar2, best_R2, best_R2_4)
    print(f"  Best model R2 = {best_overall:.3f}")
    
    if best_overall > 0.5:
        print(f"  -> Non-Markovian evolution confirmed!")
        print(f"  -> Memory kernel xi = {best_xi_4:.1f} layers")
    else:
        print(f"  -> No simple evolution equation found")
        print(f"  -> Spectral evolution may be chaotic or require more complex models")
    
    return {
        "model": model_name,
        "R2_markov": R2_markov,
        "R2_ar2": R2_ar2,
        "R2_exp_kernel": best_R2,
        "best_xi": best_xi_4,
        "best_theta": best_theta,
        "R2_pl_kernel": best_R2_4,
    }


# ============================================================
# P470: 有效耦合常数c的解析推导
# ============================================================
def run_p470(model_name, model, tokenizer, device):
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    layers = get_layers(model)
    
    print(f"\n  P470: Effective coupling constant c derivation - {model_name}")
    print(f"  d_model={d_model}, n_layers={n_layers}")
    
    # 1. W_U^T SVD
    W_U = get_W_U(model)
    W_Ut = W_U.T
    k_max = min(400, min(W_Ut.shape) - 1)
    if max(W_Ut.shape) > 50000:
        k_max = min(k_max, 400)
    
    U_wut, s_wut = safe_svd(W_Ut, k=k_max)
    
    # 2. delta of W_U^T
    i_arr = np.arange(1, k_max + 1, dtype=np.float64)
    valid_sv = s_wut[:k_max] > 0
    log_i = np.log(i_arr[valid_sv])
    log_s = np.log(s_wut[:k_max][valid_sv])
    n_fit = int(0.8 * len(log_i))
    coeffs = np.polyfit(log_i[:n_fit], log_s[:n_fit], 1)
    delta_WU = -coeffs[0]
    print(f"  delta(W_U^T) = {delta_WU:.4f}")
    
    # 3. 计算所有层的alpha和gamma
    attribute_words = ["red", "blue", "green", "yellow", "big", "small", "hot", "cold"]
    base_text = "The apple is"
    base_ids = tokenizer.encode(base_text, return_tensors="pt").to(device)
    
    sampled_layers = list(range(2, n_layers - 1, max(1, n_layers // 15)))
    
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
            e_i, _, _, _ = compute_spectral_density(delta_h, U_wut, s_wut, k_max)
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
        ratio_k = compute_ratio_k(e_i_total, k_max)
        # Fit ratio(k) = C * k^gamma
        k_vals = np.arange(1, k_max + 1, dtype=float)
        valid_r = (ratio_k > 0) & (ratio_k < 1)
        if np.sum(valid_r) > 10:
            C_r, gamma_r, R2_r = power_law_fit(k_vals[valid_r], ratio_k[valid_r])
        else:
            gamma_r = 1.0
        
        alpha_list.append(alpha)
        gamma_list.append(gamma_r)
    
    alpha_arr = np.array(alpha_list)
    gamma_arr = np.array(gamma_list)
    
    # 4. 拟合gamma = c3 - c2 * alpha
    # Model 1: gamma = 1 - delta * alpha (theory)
    gamma_theory = 1 - delta_WU * alpha_arr
    ss_res = np.sum((gamma_arr - gamma_theory) ** 2)
    ss_tot = np.sum((gamma_arr - np.mean(gamma_arr)) ** 2)
    R2_theory = 1 - ss_res / ss_tot if ss_tot > 0 else -1
    
    # Model 2: gamma = c3 - c2 * alpha (free fit)
    if np.std(alpha_arr) > 1e-10:
        X = np.column_stack([-alpha_arr, np.ones(len(alpha_arr))])
        coeffs_c, _, _, _ = np.linalg.lstsq(X, gamma_arr, rcond=None)
        c2 = -coeffs_c[0]  # Note: we fit gamma = -c2*(-alpha) + c3
        c3 = coeffs_c[1]
        gamma_pred = c3 - c2 * alpha_arr
        ss_res2 = np.sum((gamma_arr - gamma_pred) ** 2)
        R2_free = 1 - ss_res2 / ss_tot if ss_tot > 0 else -1
    else:
        c2 = delta_WU
        c3 = 1.0
        R2_free = -1
    
    print(f"\n  gamma = 1 - delta*alpha: c2={delta_WU:.4f}, R2={R2_theory:.3f}")
    print(f"  gamma = c3 - c2*alpha:   c2={c2:.4f}, c3={c3:.4f}, R2={R2_free:.3f}")
    print(f"  c2/delta = {c2/delta_WU:.2f}")
    
    # 5. 测量权重矩阵的统计性质
    print(f"\n  Weight matrix statistics:")
    
    # 采样5个层
    layer_sample = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    
    W_down_norms = []
    W_attn_norms = []
    sigma_W_list = []
    
    for l_idx in layer_sample:
        lw = get_layer_weights(layers[l_idx], d_model, MODEL_CONFIGS[model_name]['mlp_type'])
        
        # W_down的Frobenius norm
        W_down_norm = np.linalg.norm(lw.W_down)
        W_down_norms.append(W_down_norm)
        
        # W_o的Frobenius norm
        W_attn_norm = np.linalg.norm(lw.W_o)
        W_attn_norms.append(W_attn_norm)
        
        # W_down的奇异值
        s_down = np.linalg.svd(lw.W_down.astype(np.float32), compute_uv=False)
        sigma_W_list.append(np.std(s_down))
    
    print(f"  W_down norm range: {min(W_down_norms):.2f} ~ {max(W_down_norms):.2f}")
    print(f"  W_attn norm range: {min(W_attn_norms):.2f} ~ {max(W_attn_norms):.2f}")
    print(f"  sigma(W_down) range: {min(sigma_W_list):.4f} ~ {max(sigma_W_list):.4f}")
    
    avg_sigma_W = np.mean(sigma_W_list)
    
    # 6. 候选公式
    print(f"\n  Candidate formulas for c:")
    
    # c = delta * (1 + sigma_W^2 * d_model / n_layers)
    c_cand1 = delta_WU * (1 + avg_sigma_W ** 2 * d_model / n_layers)
    print(f"  1) delta*(1+sigma_W^2*d/n_L) = {c_cand1:.4f} (c_measured={c2:.4f}, error={abs(c_cand1-c2):.4f})")
    
    # c = delta * n_layers / (n_layers - xi) where xi is memory length
    # Estimate xi from alpha autocorrelation
    if len(alpha_arr) > 3:
        alpha_ac1 = np.corrcoef(alpha_arr[:-1], alpha_arr[1:])[0, 1]
        if alpha_ac1 < 1:
            xi_est = -1.0 / np.log(max(alpha_ac1, 0.01))
        else:
            xi_est = 100
    else:
        xi_est = 5
    c_cand2 = delta_WU * n_layers / max(n_layers - xi_est, 1)
    print(f"  2) delta*n_L/(n_L-xi) = {c_cand2:.4f} (xi={xi_est:.1f}, error={abs(c_cand2-c2):.4f})")
    
    # c = delta * (1 + alpha_avg)
    alpha_avg = np.mean(np.abs(alpha_arr))
    c_cand3 = delta_WU * (1 + alpha_avg)
    print(f"  3) delta*(1+|alpha|_avg) = {c_cand3:.4f} (|alpha|_avg={alpha_avg:.3f}, error={abs(c_cand3-c2):.4f})")
    
    # c = delta * (1 + 0.5 * alpha_max)
    alpha_max = np.max(np.abs(alpha_arr))
    c_cand4 = delta_WU * (1 + 0.5 * alpha_max)
    print(f"  4) delta*(1+0.5*|alpha|_max) = {c_cand4:.4f} (|alpha|_max={alpha_max:.3f}, error={abs(c_cand4-c2):.4f})")
    
    # c = delta + (1-delta) * alpha_avg / d_model * n_layers
    c_cand5 = delta_WU + (1 - delta_WU) * alpha_avg / d_model * n_layers
    print(f"  5) delta + (1-delta)*alpha*d*n_L = {c_cand5:.4f} (error={abs(c_cand5-c2):.4f})")
    
    # c proportional to n_layers
    c_cand6 = delta_WU * np.sqrt(n_layers)
    print(f"  6) delta*sqrt(n_L) = {c_cand6:.4f} (error={abs(c_cand6-c2):.4f})")
    
    c_cand7 = delta_WU * np.log(n_layers)
    print(f"  7) delta*ln(n_L) = {c_cand7:.4f} (error={abs(c_cand7-c2):.4f})")
    
    # 7. 总结
    print(f"\n  P470 Summary:")
    print(f"  delta = {delta_WU:.4f}")
    print(f"  c2 (measured) = {c2:.4f}")
    print(f"  c2/delta = {c2/delta_WU:.2f} (amplification factor)")
    print(f"  sigma_W = {avg_sigma_W:.4f}")
    print(f"  alpha range: {min(alpha_arr):.3f} ~ {max(alpha_arr):.3f}")
    print(f"  alpha_avg = {alpha_avg:.3f}")
    print(f"  n_layers = {n_layers}")
    print(f"  d_model = {d_model}")
    
    return {
        "model": model_name,
        "delta": delta_WU,
        "c2": c2,
        "c3": c3,
        "c2_over_delta": c2 / delta_WU,
        "sigma_W": avg_sigma_W,
        "alpha_avg": alpha_avg,
        "alpha_max": alpha_max,
        "n_layers": n_layers,
        "d_model": d_model,
    }


# ============================================================
# P471: 跨模型统一gamma公式
# ============================================================

# 全局数据收集
all_model_data = []

def run_p471(model_name, model, tokenizer, device):
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    layers = get_layers(model)
    
    print(f"\n  P471: Cross-model unified gamma formula - {model_name}")
    print(f"  d_model={d_model}, n_layers={n_layers}")
    
    # 1. W_U^T SVD
    W_U = get_W_U(model)
    W_Ut = W_U.T
    k_max = min(400, min(W_Ut.shape) - 1)
    if max(W_Ut.shape) > 50000:
        k_max = min(k_max, 400)
    
    U_wut, s_wut = safe_svd(W_Ut, k=k_max)
    
    # delta
    i_arr = np.arange(1, k_max + 1, dtype=np.float64)
    valid_sv = s_wut[:k_max] > 0
    log_i = np.log(i_arr[valid_sv])
    log_s = np.log(s_wut[:k_max][valid_sv])
    n_fit = int(0.8 * len(log_i))
    coeffs = np.polyfit(log_i[:n_fit], log_s[:n_fit], 1)
    delta_WU = -coeffs[0]
    
    # 2. 计算所有层的alpha, gamma, 和ratio特征
    attribute_words = ["red", "blue", "green", "yellow", "big", "small", "hot", "cold"]
    base_text = "The apple is"
    base_ids = tokenizer.encode(base_text, return_tensors="pt").to(device)
    
    sampled_layers = list(range(2, n_layers - 1, max(1, n_layers // 12)))
    
    model_data = []
    
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
            e_i, _, _, _ = compute_spectral_density(delta_h, U_wut, s_wut, k_max)
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
        
        # gamma
        ratio_k = compute_ratio_k(e_i_total, k_max)
        k_vals = np.arange(1, k_max + 1, dtype=float)
        valid_r = (ratio_k > 0) & (ratio_k < 1)
        if np.sum(valid_r) > 10:
            C_r, gamma_r, R2_r = power_law_fit(k_vals[valid_r], ratio_k[valid_r])
        else:
            gamma_r = 1.0
            R2_r = -1
        
        # ratio特征
        total_energy = np.sum(e_i_total)
        k100 = min(100, k_max)
        ratio_k100 = np.sum(e_i_total[:k100]) / total_energy if total_energy > 0 else 0
        k200 = min(200, k_max)
        ratio_k200 = np.sum(e_i_total[:k200]) / total_energy if total_energy > 0 else 0
        
        model_data.append({
            "alpha": alpha,
            "gamma": gamma_r,
            "gamma_R2": R2_r,
            "ratio_k100": ratio_k100,
            "ratio_k200": ratio_k200,
            "delta": delta_WU,
            "d_model": d_model,
            "n_layers": n_layers,
            "layer_frac": layer_idx / n_layers,
            "model": model_name,
        })
    
    # 3. 本模型的gamma公式分析
    alphas = np.array([d["alpha"] for d in model_data])
    gammas = np.array([d["gamma"] for d in model_data])
    
    # gamma = 1 - c*alpha
    if np.std(alphas) > 1e-10:
        X = np.column_stack([-alphas, np.ones(len(alphas))])
        coeffs, _, _, _ = np.linalg.lstsq(X, gammas, rcond=None)
        c2_local = -coeffs[0]
        c3_local = coeffs[1]
        gammas_pred = c3_local - c2_local * alphas
        ss_res = np.sum((gammas - gammas_pred) ** 2)
        ss_tot = np.sum((gammas - np.mean(gammas)) ** 2)
        R2_local = 1 - ss_res / ss_tot if ss_tot > 0 else -1
    else:
        c2_local = delta_WU
        c3_local = 1.0
        R2_local = -1
    
    print(f"\n  Local: gamma = {c3_local:.3f} - {c2_local:.3f}*alpha, R2={R2_local:.3f}")
    print(f"  c/delta = {c2_local/delta_WU:.2f}")
    
    # 4. 多变量拟合
    # gamma = a1*delta + a2*alpha + a3*d_model/n_layers + a4*layer_frac + a5
    print(f"\n  Multi-variable fit:")
    
    # Build feature matrix
    X_multi = np.column_stack([
        np.array([d["delta"] for d in model_data]),
        np.array([d["alpha"] for d in model_data]),
        np.array([d["d_model"] / d["n_layers"] for d in model_data]),
        np.array([d["layer_frac"] for d in model_data]),
        np.ones(len(model_data)),
    ])
    y_multi = gammas
    
    # Regularized least squares
    from numpy.linalg import lstsq
    coeffs_multi, _, _, _ = lstsq(X_multi, y_multi, rcond=None)
    y_pred_multi = X_multi @ coeffs_multi
    ss_res = np.sum((y_multi - y_pred_multi) ** 2)
    ss_tot = np.sum((y_multi - np.mean(y_multi)) ** 2)
    R2_multi = 1 - ss_res / ss_tot if ss_tot > 0 else -1
    
    print(f"  gamma = {coeffs_multi[0]:.3f}*delta + {coeffs_multi[1]:.3f}*alpha + "
          f"{coeffs_multi[2]:.4f}*(d/n_L) + {coeffs_multi[3]:.3f}*layer_frac + {coeffs_multi[4]:.3f}")
    print(f"  R2 = {R2_multi:.3f}")
    
    # 5. 收集到全局数据
    all_model_data.extend(model_data)
    
    print(f"\n  P471 partial results for {model_name}:")
    print(f"  {len(model_data)} layer data points collected")
    print(f"  alpha range: {min(alphas):.3f} ~ {max(alphas):.3f}")
    print(f"  gamma range: {min(gammas):.3f} ~ {max(gammas):.3f}")
    
    return model_data


def run_p471_cross_model_analysis():
    """在所有模型测试完成后, 进行跨模型分析"""
    if len(all_model_data) < 5:
        print("  Not enough data for cross-model analysis")
        return
    
    print(f"\n{'='*70}")
    print(f"  P471 Cross-model unified gamma formula analysis")
    print(f"  Total data points: {len(all_model_data)}")
    print(f"{'='*70}")
    
    alphas = np.array([d["alpha"] for d in all_model_data])
    gammas = np.array([d["gamma"] for d in all_model_data])
    deltas = np.array([d["delta"] for d in all_model_data])
    d_models = np.array([d["d_model"] for d in all_model_data])
    n_layers_arr = np.array([d["n_layers"] for d in all_model_data])
    layer_fracs = np.array([d["layer_frac"] for d in all_model_data])
    
    # Model A: gamma = 1 - delta*alpha (Phase XCV theory)
    gamma_A = 1 - deltas * alphas
    ss_res_A = np.sum((gammas - gamma_A) ** 2)
    ss_tot = np.sum((gammas - np.mean(gammas)) ** 2)
    R2_A = 1 - ss_res_A / ss_tot if ss_tot > 0 else -1
    
    # Model B: gamma = c3 - c2*alpha (free c2, c3)
    X_B = np.column_stack([-alphas, np.ones(len(alphas))])
    coeffs_B, _, _, _ = np.linalg.lstsq(X_B, gammas, rcond=None)
    c2_B = -coeffs_B[0]
    c3_B = coeffs_B[1]
    gamma_B = c3_B - c2_B * alphas
    ss_res_B = np.sum((gammas - gamma_B) ** 2)
    R2_B = 1 - ss_res_B / ss_tot if ss_tot > 0 else -1
    
    # Model C: gamma = 1 - c(delta, d, n_L)*alpha
    # c = delta * (1 + a*d/n_L + b)
    # Test: c = delta * k, where k is to be determined
    X_C = np.column_stack([-deltas * alphas, -alphas, np.ones(len(alphas))])
    coeffs_C, _, _, _ = np.linalg.lstsq(X_C, gammas, rcond=None)
    # gamma = coeffs_C[2] + coeffs_C[0]*(-delta*alpha) + coeffs_C[1]*(-alpha)
    # = coeffs_C[2] - coeffs_C[0]*delta*alpha - coeffs_C[1]*alpha
    # = coeffs_C[2] - (coeffs_C[0]*delta + coeffs_C[1])*alpha
    gamma_C = X_C @ coeffs_C
    ss_res_C = np.sum((gammas - gamma_C) ** 2)
    R2_C = 1 - ss_res_C / ss_tot if ss_tot > 0 else -1
    
    effective_c = coeffs_C[0] * np.mean(deltas) + coeffs_C[1]
    
    # Model D: Full multi-variable
    X_D = np.column_stack([
        deltas, alphas, deltas * alphas,
        d_models / n_layers_arr, layer_fracs,
        np.ones(len(alphas)),
    ])
    coeffs_D, _, _, _ = np.linalg.lstsq(X_D, gammas, rcond=None)
    gamma_D = X_D @ coeffs_D
    ss_res_D = np.sum((gammas - gamma_D) ** 2)
    R2_D = 1 - ss_res_D / ss_tot if ss_tot > 0 else -1
    
    print(f"\n  Model comparison:")
    print(f"  A) gamma = 1 - delta*alpha:             R2 = {R2_A:.3f}")
    print(f"  B) gamma = {c3_B:.3f} - {c2_B:.3f}*alpha:        R2 = {R2_B:.3f}")
    print(f"  C) gamma = c0 - (c1*delta + c2)*alpha:  R2 = {R2_C:.3f}")
    print(f"     effective c = {effective_c:.4f}")
    print(f"  D) Full multi-variable:                  R2 = {R2_D:.3f}")
    
    print(f"\n  Coefficients for Model D:")
    labels = ["delta", "alpha", "delta*alpha", "d/n_L", "layer_frac", "const"]
    for i, label in enumerate(labels):
        print(f"    {label}: {coeffs_D[i]:.4f}")
    
    # Per-model c values
    print(f"\n  Per-model effective coupling constants:")
    for mn in set(d["model"] for d in all_model_data):
        mask = np.array([d["model"] == mn for d in all_model_data])
        a_sub = alphas[mask]
        g_sub = gammas[mask]
        if np.std(a_sub) > 1e-10:
            X_sub = np.column_stack([-a_sub, np.ones(len(a_sub))])
            c_sub, _, _, _ = np.linalg.lstsq(X_sub, g_sub, rcond=None)
            c2_sub = -c_sub[0]
            c3_sub = c_sub[1]
            d_sub = deltas[mask][0]
            n_l = n_layers_arr[mask][0]
            d_m = d_models[mask][0]
            print(f"    {mn}: c2={c2_sub:.4f}, c3={c3_sub:.4f}, c/delta={c2_sub/d_sub:.2f}, "
                  f"n_L={n_l}, d={d_m}, d/n_L={d_m/n_l:.1f}")
    
    # Best unified formula
    best_R2 = max(R2_A, R2_B, R2_C, R2_D)
    best_model = ["A", "B", "C", "D"][[R2_A, R2_B, R2_C, R2_D].index(best_R2)]
    
    print(f"\n  Best model: {best_model} (R2={best_R2:.3f})")
    
    if R2_D > 0.7:
        print(f"  -> Unified gamma formula found!")
        print(f"  gamma = {coeffs_D[0]:.3f}*delta + {coeffs_D[1]:.3f}*alpha + "
              f"{coeffs_D[2]:.3f}*delta*alpha + {coeffs_D[3]:.4f}*(d/n_L) + "
              f"{coeffs_D[4]:.3f}*layer_frac + {coeffs_D[5]:.3f}")
    elif R2_B > 0.5:
        print(f"  -> Simple formula gamma = {c3_B:.3f} - {c2_B:.3f}*alpha works best")
        print(f"  -> But c2 is model-dependent, not universal")
    else:
        print(f"  -> No simple universal gamma formula found")
        print(f"  -> gamma may depend on additional factors not captured here")


# ============================================================
# Main
# ============================================================
import torch

EXPERIMENTS = {
    "p468": run_p468,
    "p469": run_p469,
    "p470": run_p470,
    "p471": run_p471,
}


def main():
    parser = argparse.ArgumentParser(description="Phase XCVII: Non-Markovian Spectral Evolution")
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["p468", "p469", "p470", "p471"])
    args = parser.parse_args()
    
    model_name = args.model
    experiment = args.experiment
    
    print(f"\n{'='*70}")
    print(f"  Phase XCVII: Non-Markovian Spectral Evolution")
    print(f"  Model: {model_name}, Experiment: {experiment}")
    print(f"{'='*70}")
    
    # Load model
    model, tokenizer, device = load_model(model_name)
    
    try:
        result = EXPERIMENTS[experiment](model_name, model, tokenizer, device)
    finally:
        release_model(model)
    
    print(f"\n  Experiment {experiment} on {model_name} complete!")
    
    # P471 cross-model analysis after all models
    if experiment == "p471":
        run_p471_cross_model_analysis()


if __name__ == "__main__":
    main()
