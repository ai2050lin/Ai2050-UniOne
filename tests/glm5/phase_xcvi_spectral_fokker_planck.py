"""
Phase XCVI-P464/465/466/467: 频谱Fokker-Planck方程
======================================================================

核心目标: 从第一性原理推导alpha的解析解,建立频谱Fokker-Planck方程

Phase XCV核心成果:
1. gamma = 1 - delta*alpha (ratio(k)第一性原理公式)
2. alpha* ~ 1.3-1.4 (RG流稳定不动点)
3. 信号传播3阶段: 浅层激增, 中层稳定, 末层衰减

Phase XCVI目标:
1. 用更大k验证gamma公式 (修复k_max太小的问题)
2. 建立频谱密度演化方程: dS_Delta/dL = F(S_Delta, W_L)
3. 从权重矩阵统计性质解析推导alpha不动点
4. 寻找gamma公式的高阶修正项

P464: 大k验证gamma公式
  - Phase XCV的k_max=500-800, 导致ratio(k=800)=1.0
  - 本实验: 用k=min(d_model-1, 2000)重新验证
  - 方法:
    a) 对W_U^T做完整SVD(或k=2000截断)
    b) 在3个层位置计算alpha, delta, gamma
    c) 验证gamma_theory = 1 - delta*alpha
  - 预期: gamma_measured更接近gamma_theory

P465: 频谱密度演化方程 dS_Delta/dL
  - 目标: 建立频谱密度的连续演化方程
  - 方法:
    a) 测量每层的S_Delta(i, L)
    b) 计算dS/Delta/dL = (S_Delta(i, L+1) - S_Delta(i, L-1)) / 2
    c) 检验Fokker-Planck形式: dS/Delta/dL = -D1(i)*dS/di + D2(i)*d2S/di2
    d) 提取漂移项D1(i)和扩散项D2(i)
  - 预期: D1(i)可能与delta(奇异值衰减率)有关

P466: alpha不动点的理论推导
  - 目标: 从权重矩阵的统计性质推导alpha* ~ 1.3-1.4
  - 方法:
    a) 假设e_i = A * s_i^alpha, 计算alpha的self-consistency方程
    b) alpha由传播方程决定: alpha(L+1) = f(alpha(L), W_L)
    c) 不动点条件: alpha* = f(alpha*, W)
    d) 检验alpha*是否等于1/(delta+1)或2/(delta+1)等解析公式
  - 预期: alpha*可能与delta有关, 如alpha* = 1/delta ~ 6-7?

P467: gamma公式的高阶修正
  - 目标: gamma = 1 - delta*alpha + c2*alpha^2 + c3*delta^2 + ...
  - 方法:
    a) 用多模型多层数据拟合: gamma = 1 + c1*delta + c2*alpha + c3*delta*alpha + ...
    b) 与理论公式gamma = 1 - delta*alpha对比
    c) 计算各修正项的贡献
  - 预期: 高阶修正可能解释GLM4的0.194误差
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.sparse.linalg import svds
from scipy.optimize import curve_fit
from sklearn.decomposition import TruncatedSVD

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


def safe_svd(matrix, k, random_state=42):
    """Memory-safe SVD: uses TruncatedSVD for large matrices"""
    k = min(k, min(matrix.shape) - 1)
    if max(matrix.shape) > 50000:
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


def compute_alpha(e_i, s_wut, k_max):
    """Compute alpha: log(e_i) = alpha * log(s_i) + beta"""
    mask = (s_wut[:k_max] > 1e-6) & (e_i[:k_max] > 0)
    if np.sum(mask) < 10:
        return 0, 0
    log_s = np.log10(s_wut[:k_max][mask]).reshape(-1, 1)
    log_e = np.log10(e_i[:k_max][mask])
    A = np.hstack([log_s, np.ones_like(log_s)])
    res = np.linalg.lstsq(A, log_e, rcond=None)
    alpha = float(res[0][0])
    # R2
    pred = res[0][0] * log_s.flatten() + res[0][1]
    ss_res = np.sum((log_e - pred) ** 2)
    ss_tot = np.sum((log_e - np.mean(log_e)) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return alpha, R2


def compute_spectral_density(delta_h, U_wut, s_wut, k):
    """Compute spectral density S_Delta(i) = e_i / s_i^2"""
    k_use = min(k, U_wut.shape[1], len(s_wut))
    U_k = U_wut[:, :k_use]
    s_k = s_wut[:k_use]
    
    projections = U_k.T @ delta_h
    e_i = projections ** 2
    
    s_sq = s_k ** 2
    s_sq[s_sq < 1e-30] = 1e-30
    S_omega = e_i / s_sq
    
    return e_i, S_omega


def run_p464(model_name, model, tokenizer, device):
    """
    P464: 大k验证gamma公式
    - 用k=min(d_model-1, 2000)重新验证gamma = 1 - delta*alpha
    """
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    
    print(f"\n  P464: 大k验证gamma公式 - {model_name}")
    print(f"  d_model={info.d_model}, n_layers={info.n_layers}")
    
    # 1. 获取W_U^T的SVD基, 用更大的k
    W_U = get_W_U(model)
    W_Ut = W_U.T
    
    # k = min(d_model-2, 2000), 确保不超过矩阵维度
    # 对于大vocab模型(如GLM4), 限制k以避免内存溢出
    k_max = min(d_model - 2, 2000, min(W_Ut.shape) - 1)
    if max(W_Ut.shape) > 50000:
        k_max = min(k_max, 500)  # 大vocab模型限制k<=500
    print(f"  k_max={k_max} (d_model={d_model})")
    
    U_wut, s_wut = safe_svd(W_Ut, k=k_max)
    
    print(f"  W_U^T SVD: k_max={k_max}, s_max={s_wut[0]:.2f}, s_min={s_wut[-1]:.4f}")
    
    # 2. 属性词
    attribute_words = ["red", "blue", "green", "yellow", "big", "small", "hot", "cold"]
    
    # 3. 选择层位置: 浅/中/深
    layer_positions = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]
    
    base_text = "The apple is"
    base_ids = tokenizer.encode(base_text, return_tensors="pt").to(device)
    
    results = {"model": model_name, "k_max": k_max, "d_model": d_model, "layers": {}}
    
    for layer_idx in layer_positions:
        print(f"\n  Layer {layer_idx}/{n_layers-1}...")
        
        with torch.no_grad():
            base_out = model(base_ids, output_hidden_states=True)
            base_h = base_out.hidden_states[layer_idx + 1][0, -1].cpu().float().numpy()
        
        all_e_i = []
        all_S_omega = []
        
        for attr_word in attribute_words:
            intervened_text = f"The {attr_word} apple is"
            interv_ids = tokenizer.encode(intervened_text, return_tensors="pt").to(device)
            
            with torch.no_grad():
                interv_out = model(interv_ids, output_hidden_states=True)
                interv_h = interv_out.hidden_states[layer_idx + 1][0, -1].cpu().float().numpy()
            
            delta_h = interv_h - base_h
            e_i, S_omega = compute_spectral_density(delta_h, U_wut, s_wut, k=k_max)
            all_e_i.append(e_i)
            all_S_omega.append(S_omega)
        
        avg_e_i = np.mean(all_e_i, axis=0)
        total_energy = np.sum(avg_e_i)
        
        # 计算alpha (e_i vs s_i)
        alpha_L, R2_alpha = compute_alpha(avg_e_i, s_wut, k_max)
        
        # 计算delta (s_i vs i)
        i_vals = np.arange(1, k_max + 1, dtype=float)
        mask_s = s_wut[:k_max] > 1e-6
        if np.sum(mask_s) > 10:
            log_i = np.log10(i_vals[mask_s]).reshape(-1, 1)
            log_sv = np.log10(s_wut[:k_max][mask_s])
            A = np.hstack([log_i, np.ones_like(log_i)])
            res = np.linalg.lstsq(A, log_sv, rcond=None)
            delta_sv = -res[0][0]
        else:
            delta_sv = 0
        
        # 计算ratio(k)和gamma
        ratio_k = np.cumsum(avg_e_i) / total_energy
        
        # gamma: 拟合ratio(k) vs k
        mask_r = (i_vals >= 5) & (ratio_k > 0) & (ratio_k < 1)
        if np.sum(mask_r) > 10:
            log_k = np.log10(i_vals[mask_r]).reshape(-1, 1)
            log_r = np.log10(ratio_k[mask_r])
            A = np.hstack([log_k, np.ones_like(log_k)])
            res = np.linalg.lstsq(A, log_r, rcond=None)
            gamma_meas = res[0][0]
            C_ratio = 10 ** res[0][1]
            R2_gamma = 1 - np.sum((log_r - (gamma_meas * log_k.flatten() + np.log10(C_ratio))) ** 2) / np.sum((log_r - np.mean(log_r)) ** 2)
        else:
            gamma_meas = 0
            C_ratio = 0
            R2_gamma = 0
        
        # gamma_theory = 1 - delta*alpha
        gamma_theory = 1 - delta_sv * alpha_L
        
        # ratio at key k values
        for k_test in [100, 200, 500, 1000, 1500, 2000]:
            if k_test <= k_max:
                r_k = ratio_k[k_test - 1]
                r_theory = C_ratio * k_test ** gamma_meas
                print(f"  ratio(k={k_test}): measured={r_k:.4f}, theory={r_theory:.4f}")
        
        print(f"  alpha={alpha_L:.3f}(R2={R2_alpha:.3f}), delta={delta_sv:.3f}")
        print(f"  gamma_measured={gamma_meas:.3f}(R2={R2_gamma:.4f}), gamma_theory={gamma_theory:.3f}")
        print(f"  gamma_error={abs(gamma_meas - gamma_theory):.3f}")
        
        results["layers"][str(layer_idx)] = {
            "alpha": float(alpha_L),
            "delta": float(delta_sv),
            "gamma_measured": float(gamma_meas),
            "gamma_theory": float(gamma_theory),
            "gamma_error": float(abs(gamma_meas - gamma_theory)),
            "R2_alpha": float(R2_alpha),
            "R2_gamma": float(R2_gamma),
            "C_ratio": float(C_ratio),
        }
    
    # 全局总结
    all_errors = [results["layers"][l]["gamma_error"] for l in results["layers"]]
    print(f"\n  === P464 {model_name} Summary ===")
    print(f"  k_max={k_max}, d_model={d_model}, k/d={k_max/d_model:.3f}")
    print(f"  gamma_error range: {min(all_errors):.3f} ~ {max(all_errors):.3f}")
    print(f"  gamma_error mean: {np.mean(all_errors):.3f}")
    
    return results


def run_p465(model_name, model, tokenizer, device):
    """
    P465: 频谱密度演化方程 dS_Delta/dL
    - 检验Fokker-Planck形式: dS/dL = -D1(i)*dS/di + D2(i)*d2S/di2
    """
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    
    print(f"\n  P465: 频谱密度演化方程 - {model_name}")
    print(f"  d_model={info.d_model}, n_layers={info.n_layers}")
    
    # 1. W_U^T SVD
    W_U = get_W_U(model)
    W_Ut = W_U.T
    k_max = min(400, min(W_Ut.shape) - 1)
    U_wut, s_wut = safe_svd(W_Ut, k=k_max)
    
    # 2. 属性词
    attribute_words = ["red", "blue", "green", "yellow", "big", "small", "hot", "cold"]
    
    # 3. 预计算所有层的频谱密度
    print(f"  Computing spectral density for all layers...")
    base_text = "The apple is"
    base_ids = tokenizer.encode(base_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        base_out = model(base_ids, output_hidden_states=True)
    
    # 对每个属性预计算, 然后平均
    layer_S_omega = {}  # layer -> avg_S_omega
    layer_e_i = {}      # layer -> avg_e_i
    
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
            
            if layer_idx not in layer_S_omega:
                layer_S_omega[layer_idx] = []
                layer_e_i[layer_idx] = []
            layer_S_omega[layer_idx].append(S_omega)
            layer_e_i[layer_idx].append(e_i)
    
    # 平均
    for l in layer_S_omega:
        layer_S_omega[l] = np.mean(layer_S_omega[l], axis=0)
        layer_e_i[l] = np.mean(layer_e_i[l], axis=0)
    
    print(f"  Computed spectral density for {len(layer_S_omega)} layers")
    
    # 4. 计算dS/dL (中心差分)
    # 只对中间层(跳过首尾2层)
    sample_layers = list(range(2, n_layers - 2))
    
    dS_dL = {}  # layer -> dS_Delta/dL
    for l in sample_layers:
        if l-1 in layer_S_omega and l+1 in layer_S_omega:
            dS_dL[l] = (layer_S_omega[l+1] - layer_S_omega[l-1]) / 2.0
    
    # 5. 提取Fokker-Planck系数
    # dS/dL = -D1(i) * dS/di + D2(i) * d2S/di2
    # 用最小二乘法估计D1(i)和D2(i)
    
    # 对每个层位置, 计算dS/di和d2S/di2
    results = {"model": model_name, "k_max": k_max, "fp_coefficients": {}}
    
    fp_layers = list(range(4, n_layers - 4, max(1, n_layers // 5)))
    
    for l in fp_layers:
        if l not in dS_dL:
            continue
        
        S = layer_S_omega[l]
        dSdt = dS_dL[l]
        
        # dS/di (中心差分)
        dSdi = np.zeros_like(S)
        dSdi[1:-1] = (S[2:] - S[:-2]) / 2.0
        dSdi[0] = S[1] - S[0]
        dSdi[-1] = S[-1] - S[-2]
        
        # d2S/di2 (中心差分)
        d2Sdi2 = np.zeros_like(S)
        d2Sdi2[1:-1] = (S[2:] - 2*S[1:-1] + S[:-2])
        d2Sdi2[0] = d2Sdi2[1]
        d2Sdi2[-1] = d2Sdi2[-2]
        
        # 最小二乘: dS/dt = -D1 * dS/di + D2 * d2S/di2
        # 即 dS/dt = [dS/di, d2S/di2] @ [-D1, D2]
        valid = (np.abs(dSdt) > 0) & (np.abs(dSdi) > 0)
        if np.sum(valid) > 10:
            A = np.vstack([dSdi[valid], d2Sdi2[valid]]).T
            b = dSdt[valid]
            res = np.linalg.lstsq(A, b, rcond=None)
            D1 = -res[0][0]  # 漂移系数
            D2 = res[0][1]   # 扩散系数
            
            # R2
            pred = -D1 * dSdi[valid] + D2 * d2Sdi2[valid]
            ss_res = np.sum((dSdt[valid] - pred) ** 2)
            ss_tot = np.sum((dSdt[valid] - np.mean(dSdt[valid])) ** 2)
            R2_fp = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        else:
            D1 = D2 = 0
            R2_fp = 0
        
        # 分频段分析D1和D2
        n_bands = 5
        band_size = k_max // n_bands
        D1_bands = []
        D2_bands = []
        for b in range(n_bands):
            start = b * band_size
            end = (b + 1) * band_size
            v = valid[start:end]
            if np.sum(v) > 3:
                A_b = np.vstack([dSdi[start:end][v], d2Sdi2[start:end][v]]).T
                b_b = dSdt[start:end][v]
                res_b = np.linalg.lstsq(A_b, b_b, rcond=None)
                D1_bands.append(float(-res_b[0][0]))
                D2_bands.append(float(res_b[0][1]))
            else:
                D1_bands.append(0.0)
                D2_bands.append(0.0)
        
        # alpha for this layer
        alpha_L, _ = compute_alpha(layer_e_i[l], s_wut, k_max)
        
        print(f"  L{l}: D1={D1:.6f}, D2={D2:.6f}, R2={R2_fp:.4f}, alpha={alpha_L:.3f}")
        print(f"    D1_bands={[f'{x:.4f}' for x in D1_bands]}")
        print(f"    D2_bands={[f'{x:.6f}' for x in D2_bands]}")
        
        results["fp_coefficients"][str(l)] = {
            "D1": float(D1),
            "D2": float(D2),
            "R2_fp": float(R2_fp),
            "D1_bands": D1_bands,
            "D2_bands": D2_bands,
            "alpha": float(alpha_L),
        }
    
    # 6. 全局分析: D1, D2与alpha的关系
    all_D1 = [results["fp_coefficients"][l]["D1"] for l in results["fp_coefficients"]]
    all_D2 = [results["fp_coefficients"][l]["D2"] for l in results["fp_coefficients"]]
    all_alpha = [results["fp_coefficients"][l]["alpha"] for l in results["fp_coefficients"]]
    
    print(f"\n  === P465 {model_name} Summary ===")
    print(f"  D1 range: {min(all_D1):.6f} ~ {max(all_D1):.6f}")
    print(f"  D2 range: {min(all_D2):.6f} ~ {max(all_D2):.6f}")
    
    # D1 vs alpha相关性
    if len(all_alpha) > 3:
        A_corr = np.vstack([all_alpha, np.ones(len(all_alpha))]).T
        res_corr = np.linalg.lstsq(A_corr, all_D1, rcond=None)
        print(f"  D1 vs alpha: D1 = {res_corr[0][0]:.6f}*alpha + {res_corr[0][1]:.6f}")
    
    return results


def run_p466(model_name, model, tokenizer, device):
    """
    P466: alpha不动点的理论推导
    - 检验alpha*是否等于某种解析公式
    - 假设: alpha由self-consistency方程决定
    """
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    
    print(f"\n  P466: alpha不动点理论推导 - {model_name}")
    print(f"  d_model={info.d_model}, n_layers={info.n_layers}")
    
    # 1. W_U^T SVD
    W_U = get_W_U(model)
    W_Ut = W_U.T
    k_max = min(400, min(W_Ut.shape) - 1)
    U_wut, s_wut = safe_svd(W_Ut, k=k_max)
    
    # 2. 计算delta (W_U^T奇异值谱衰减率)
    i_vals = np.arange(1, k_max + 1, dtype=float)
    mask_s = s_wut[:k_max] > 1e-6
    log_i = np.log10(i_vals[mask_s]).reshape(-1, 1)
    log_sv = np.log10(s_wut[:k_max][mask_s])
    A = np.hstack([log_i, np.ones_like(log_i)])
    res = np.linalg.lstsq(A, log_sv, rcond=None)
    delta_global = -res[0][0]
    print(f"  delta(W_U^T SV spectrum) = {delta_global:.4f}")
    
    # 3. 计算各层权重矩阵的delta
    layers = get_layers(model)
    weight_deltas = {}
    
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    if sample_layers[-1] != n_layers - 1:
        sample_layers.append(n_layers - 1)
    
    for l in sample_layers:
        lw = get_layer_weights(layers[l], d_model, MODEL_CONFIGS[model_name]['mlp_type'])
        weight_deltas[l] = {}
        
        for w_name in ['W_down', 'W_up', 'W_o', 'W_q']:
            W = getattr(lw, w_name, None)
            if W is None:
                continue
            k_w = min(200, min(W.shape) - 1)
            try:
                _, s_w, _ = svds(W.astype(np.float64), k=k_w)
                s_w = np.sort(s_w)[::-1]
                i_w = np.arange(1, k_w + 1, dtype=float)
                mask_w = s_w > 1e-6
                if np.sum(mask_w) > 10:
                    log_iw = np.log10(i_w[mask_w]).reshape(-1, 1)
                    log_sw = np.log10(s_w[mask_w])
                    A_w = np.hstack([log_iw, np.ones_like(log_iw)])
                    res_w = np.linalg.lstsq(A_w, log_sw, rcond=None)
                    weight_deltas[l][w_name] = float(-res_w[0][0])
            except:
                weight_deltas[l][w_name] = 0.0
    
    # 4. 计算每层的alpha (与P463相同方法)
    attribute_words = ["red", "blue", "green", "yellow", "big", "small", "hot", "cold"]
    base_text = "The apple is"
    base_ids = tokenizer.encode(base_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        base_out = model(base_ids, output_hidden_states=True)
    
    layer_alphas = {}
    for l in sample_layers:
        all_e_i = []
        for attr_word in attribute_words:
            intervened_text = f"The {attr_word} apple is"
            interv_ids = tokenizer.encode(intervened_text, return_tensors="pt").to(device)
            with torch.no_grad():
                interv_out = model(interv_ids, output_hidden_states=True)
            base_h = base_out.hidden_states[l + 1][0, -1].cpu().float().numpy()
            interv_h = interv_out.hidden_states[l + 1][0, -1].cpu().float().numpy()
            delta_h = interv_h - base_h
            e_i, _ = compute_spectral_density(delta_h, U_wut, s_wut, k=k_max)
            all_e_i.append(e_i)
        avg_e_i = np.mean(all_e_i, axis=0)
        alpha_L, _ = compute_alpha(avg_e_i, s_wut, k_max)
        layer_alphas[l] = alpha_L
    
    # 5. alpha*的候选解析公式
    # 公式1: alpha* = 1/delta
    alpha_star_1 = 1.0 / delta_global if delta_global > 0 else float('inf')
    # 公式2: alpha* = 2/(1+delta)
    alpha_star_2 = 2.0 / (1 + delta_global)
    # 公式3: alpha* = 1/(1-delta) (当delta<1时)
    alpha_star_3 = 1.0 / (1 - delta_global) if delta_global < 1 else float('inf')
    # 公式4: alpha* = 2 - delta
    alpha_star_4 = 2.0 - delta_global
    # 公式5: alpha* = 1 + delta
    alpha_star_5 = 1.0 + delta_global
    # 公式6: alpha* = sqrt(2/delta)
    alpha_star_6 = np.sqrt(2.0 / delta_global) if delta_global > 0 else float('inf')
    
    # 实测alpha*(P463的结果)
    alpha_measured_star = {
        "qwen3": 0.0,  # Qwen3没有不动点
        "glm4": 1.41,
        "deepseek7b": 1.36,
    }
    alpha_star_real = alpha_measured_star.get(model_name, 1.4)
    
    print(f"\n  === P466 {model_name} Alpha* Theory ===")
    print(f"  delta(W_U^T) = {delta_global:.4f}")
    print(f"  alpha*_measured = {alpha_star_real:.3f}")
    print(f"\n  Candidate formulas:")
    print(f"  1. alpha* = 1/delta = {alpha_star_1:.3f}")
    print(f"  2. alpha* = 2/(1+delta) = {alpha_star_2:.3f}")
    print(f"  3. alpha* = 1/(1-delta) = {alpha_star_3:.3f}")
    print(f"  4. alpha* = 2-delta = {alpha_star_4:.3f}")
    print(f"  5. alpha* = 1+delta = {alpha_star_5:.3f}")
    print(f"  6. alpha* = sqrt(2/delta) = {alpha_star_6:.3f}")
    
    # 各公式的误差
    formulas = {
        "1/delta": alpha_star_1,
        "2/(1+delta)": alpha_star_2,
        "1/(1-delta)": alpha_star_3,
        "2-delta": alpha_star_4,
        "1+delta": alpha_star_5,
        "sqrt(2/delta)": alpha_star_6,
    }
    
    if alpha_star_real > 0:
        print(f"\n  Errors vs measured alpha*={alpha_star_real:.3f}:")
        for name, val in formulas.items():
            if val != float('inf'):
                err = abs(val - alpha_star_real)
                print(f"  {name}: {val:.3f}, error={err:.3f}")
    
    # 6. alpha与权重delta的关系
    print(f"\n  === Alpha vs Weight Delta ===")
    for l in sample_layers:
        alpha_l = layer_alphas.get(l, 0)
        w_deltas = weight_deltas.get(l, {})
        avg_w_delta = np.mean(list(w_deltas.values())) if w_deltas else 0
        print(f"  L{l}: alpha={alpha_l:.3f}, avg_weight_delta={avg_w_delta:.3f}, W_U_delta={delta_global:.3f}")
    
    # 7. Self-consistency方程
    # 如果e_i = A * s_i^alpha, 且信号经过权重W传播,
    # 那么alpha(L+1) = g(alpha(L), delta_W, delta_U)
    # 不动点: alpha* = g(alpha*, delta_W, delta_U)
    
    # 简化假设: g是线性函数 alpha(L+1) = a + b*alpha(L)
    # 不动点: alpha* = a/(1-b)
    
    alphas_arr = np.array([layer_alphas[l] for l in sample_layers if l in layer_alphas])
    layers_arr = np.array([l for l in sample_layers if l in layer_alphas], dtype=float)
    
    if len(alphas_arr) > 3:
        # alpha(L+1) vs alpha(L) 拟合
        alpha_curr = alphas_arr[:-1]
        alpha_next = alphas_arr[1:]
        
        if len(alpha_curr) > 2:
            A_sc = np.vstack([alpha_curr, np.ones(len(alpha_curr))]).T
            res_sc = np.linalg.lstsq(A_sc, alpha_next, rcond=None)
            b_sc, a_sc = res_sc[0]
            
            # R2
            pred_sc = b_sc * alpha_curr + a_sc
            ss_res = np.sum((alpha_next - pred_sc) ** 2)
            ss_tot = np.sum((alpha_next - np.mean(alpha_next)) ** 2)
            R2_sc = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            alpha_star_sc = a_sc / (1 - b_sc) if abs(1 - b_sc) > 1e-10 else float('inf')
            
            print(f"\n  Self-consistency: alpha(L+1) = {b_sc:.4f}*alpha(L) + {a_sc:.4f}, R2={R2_sc:.4f}")
            print(f"  alpha* = a/(1-b) = {alpha_star_sc:.3f}")
            print(f"  Stability: b={b_sc:.4f} ({'stable' if abs(b_sc) < 1 else 'unstable'})")
    
    results = {
        "model": model_name,
        "delta_WUt": float(delta_global),
        "alpha_star_measured": float(alpha_star_real),
        "formulas": {k: float(v) if v != float('inf') else None for k, v in formulas.items()},
        "layer_alphas": {str(k): float(v) for k, v in layer_alphas.items()},
        "weight_deltas": {str(k): v for k, v in weight_deltas.items()},
    }
    
    return results


def run_p467(model_name, model, tokenizer, device):
    """
    P467: gamma公式的高阶修正
    - gamma = 1 + c1*delta + c2*alpha + c3*delta*alpha + c4*delta^2 + c5*alpha^2 + ...
    - 用多模型多层数据拟合
    """
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    
    print(f"\n  P467: gamma公式高阶修正 - {model_name}")
    print(f"  d_model={info.d_model}, n_layers={info.n_layers}")
    
    # 1. W_U^T SVD
    W_U = get_W_U(model)
    W_Ut = W_U.T
    k_max = min(400, min(W_Ut.shape) - 1)
    U_wut, s_wut = safe_svd(W_Ut, k=k_max)
    
    # 2. 计算delta
    i_vals = np.arange(1, k_max + 1, dtype=float)
    mask_s = s_wut[:k_max] > 1e-6
    log_i = np.log10(i_vals[mask_s]).reshape(-1, 1)
    log_sv = np.log10(s_wut[:k_max][mask_s])
    A = np.hstack([log_i, np.ones_like(log_i)])
    res = np.linalg.lstsq(A, log_sv, rcond=None)
    delta_global = -res[0][0]
    print(f"  delta(W_U^T) = {delta_global:.4f}")
    
    # 3. 对多个层位置计算alpha, delta(局部), gamma
    attribute_words = ["red", "blue", "green", "yellow", "big", "small", "hot", "cold"]
    base_text = "The apple is"
    base_ids = tokenizer.encode(base_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        base_out = model(base_ids, output_hidden_states=True)
    
    sample_layers = list(range(2, n_layers - 2, max(1, n_layers // 10)))
    
    data_points = []  # (alpha, delta, gamma_measured, gamma_theory)
    
    for l in sample_layers:
        all_e_i = []
        for attr_word in attribute_words:
            intervened_text = f"The {attr_word} apple is"
            interv_ids = tokenizer.encode(intervened_text, return_tensors="pt").to(device)
            with torch.no_grad():
                interv_out = model(interv_ids, output_hidden_states=True)
            base_h = base_out.hidden_states[l + 1][0, -1].cpu().float().numpy()
            interv_h = interv_out.hidden_states[l + 1][0, -1].cpu().float().numpy()
            delta_h = interv_h - base_h
            e_i, _ = compute_spectral_density(delta_h, U_wut, s_wut, k=k_max)
            all_e_i.append(e_i)
        
        avg_e_i = np.mean(all_e_i, axis=0)
        total_energy = np.sum(avg_e_i)
        
        # alpha
        alpha_L, _ = compute_alpha(avg_e_i, s_wut, k_max)
        
        # gamma (ratio(k) power law)
        ratio_k = np.cumsum(avg_e_i) / total_energy
        mask_r = (i_vals >= 5) & (ratio_k > 0) & (ratio_k < 1)
        if np.sum(mask_r) > 10:
            log_k = np.log10(i_vals[mask_r]).reshape(-1, 1)
            log_r = np.log10(ratio_k[mask_r])
            A_r = np.hstack([log_k, np.ones_like(log_k)])
            res_r = np.linalg.lstsq(A_r, log_r, rcond=None)
            gamma_meas = res_r[0][0]
        else:
            gamma_meas = 0
        
        gamma_theory = 1 - delta_global * alpha_L
        
        data_points.append({
            "layer": l,
            "alpha": float(alpha_L),
            "delta": float(delta_global),
            "gamma_measured": float(gamma_meas),
            "gamma_theory": float(gamma_theory),
            "residual": float(gamma_meas - gamma_theory),
        })
    
    # 4. 拟合高阶修正
    # 模型0: gamma = 1 - delta*alpha (Phase XCV结果)
    # 模型1: gamma = 1 + c1*delta + c2*alpha
    # 模型2: gamma = 1 + c1*delta + c2*alpha + c3*delta*alpha
    # 模型3: gamma = 1 + c1*delta + c2*alpha + c3*delta*alpha + c4*delta^2 + c5*alpha^2
    
    alphas = np.array([d["alpha"] for d in data_points])
    deltas = np.array([d["delta"] for d in data_points])
    gammas = np.array([d["gamma_measured"] for d in data_points])
    
    # 模型0: gamma = 1 - delta*alpha
    gamma0 = 1 - deltas * alphas
    R2_0 = 1 - np.sum((gammas - gamma0) ** 2) / np.sum((gammas - np.mean(gammas)) ** 2) if np.sum((gammas - np.mean(gammas)) ** 2) > 0 else 0
    
    # 模型1: gamma = 1 + c1*delta + c2*alpha
    A1 = np.vstack([deltas, alphas, np.ones(len(alphas))]).T
    res1 = np.linalg.lstsq(A1, gammas, rcond=None)
    gamma1 = A1 @ res1[0]
    R2_1 = 1 - np.sum((gammas - gamma1) ** 2) / np.sum((gammas - np.mean(gammas)) ** 2) if np.sum((gammas - np.mean(gammas)) ** 2) > 0 else 0
    
    # 模型2: gamma = 1 + c1*delta + c2*alpha + c3*delta*alpha
    A2 = np.vstack([deltas, alphas, deltas * alphas, np.ones(len(alphas))]).T
    res2 = np.linalg.lstsq(A2, gammas, rcond=None)
    gamma2 = A2 @ res2[0]
    R2_2 = 1 - np.sum((gammas - gamma2) ** 2) / np.sum((gammas - np.mean(gammas)) ** 2) if np.sum((gammas - np.mean(gammas)) ** 2) > 0 else 0
    
    # 模型3: full quadratic
    A3 = np.vstack([deltas, alphas, deltas * alphas, deltas**2, alphas**2, np.ones(len(alphas))]).T
    res3 = np.linalg.lstsq(A3, gammas, rcond=None)
    gamma3 = A3 @ res3[0]
    R2_3 = 1 - np.sum((gammas - gamma3) ** 2) / np.sum((gammas - np.mean(gammas)) ** 2) if np.sum((gammas - np.mean(gammas)) ** 2) > 0 else 0
    
    print(f"\n  === P467 {model_name} Higher-Order Corrections ===")
    print(f"  N data points: {len(data_points)}")
    print(f"  alpha range: {alphas.min():.3f} ~ {alphas.max():.3f}")
    print(f"  delta: {delta_global:.4f}")
    print(f"  gamma range: {gammas.min():.3f} ~ {gammas.max():.3f}")
    print(f"\n  Model 0: gamma = 1 - delta*alpha")
    print(f"    R2 = {R2_0:.4f}")
    print(f"\n  Model 1: gamma = {res1[0][0]:.4f}*delta + {res1[0][1]:.4f}*alpha + {res1[0][2]:.4f}")
    print(f"    R2 = {R2_1:.4f}")
    print(f"\n  Model 2: gamma = {res2[0][0]:.4f}*delta + {res2[0][1]:.4f}*alpha + {res2[0][2]:.4f}*delta*alpha + {res2[0][3]:.4f}")
    print(f"    R2 = {R2_2:.4f}")
    print(f"\n  Model 3: gamma = {res3[0][0]:.4f}*delta + {res3[0][1]:.4f}*alpha + {res3[0][2]:.4f}*delta*alpha + {res3[0][3]:.4f}*delta^2 + {res3[0][4]:.4f}*alpha^2 + {res3[0][5]:.4f}")
    print(f"    R2 = {R2_3:.4f}")
    
    # 5. 残差分析
    residuals_0 = gammas - gamma0
    residuals_2 = gammas - gamma2
    
    print(f"\n  Residual analysis:")
    print(f"  Model 0: mean={np.mean(residuals_0):.4f}, std={np.std(residuals_0):.4f}")
    print(f"  Model 2: mean={np.mean(residuals_2):.4f}, std={np.std(residuals_2):.4f}")
    
    # 残差与alpha的关系
    if len(alphas) > 3:
        A_res = np.vstack([alphas, np.ones(len(alphas))]).T
        res_res = np.linalg.lstsq(A_res, residuals_0, rcond=None)
        print(f"  Residual_0 vs alpha: slope={res_res[0][0]:.4f}")
        print(f"    -> alpha contributes {res_res[0][0]:.4f} to residual per unit alpha")
    
    results = {
        "model": model_name,
        "delta": float(delta_global),
        "n_points": len(data_points),
        "model0_R2": float(R2_0),
        "model1_R2": float(R2_1),
        "model2_R2": float(R2_2),
        "model3_R2": float(R2_3),
        "model2_coeffs": [float(x) for x in res2[0]],
        "data_points": data_points,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase XCVI: Spectral Fokker-Planck Equation")
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"],
                       help="Model name")
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["p464", "p465", "p466", "p467"],
                       help="Experiment ID")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Phase XCVI: Spectral Fokker-Planck Equation")
    print(f"Model: {args.model}, Experiment: {args.experiment}")
    print(f"{'='*60}")
    
    print(f"\nLoading model {args.model}...")
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    print(f"  d_model={info.d_model}, n_layers={info.n_layers}")
    
    if args.experiment == "p464":
        results = run_p464(args.model, model, tokenizer, device)
    elif args.experiment == "p465":
        results = run_p465(args.model, model, tokenizer, device)
    elif args.experiment == "p466":
        results = run_p466(args.model, model, tokenizer, device)
    elif args.experiment == "p467":
        results = run_p467(args.model, model, tokenizer, device)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(__file__).resolve().parent.parent / "glm5_temp"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"phase_xcvi_{args.experiment}_{args.model}_{timestamp}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved: {output_file}")
    
    release_model(model)
    print("Model released")


if __name__ == "__main__":
    main()
