"""
Phase CXVII-CXVIII-P538/P539/P540/P541: 统一频谱力学的深化
==========================================================

Phase CXV-CXVI核心发现:
- J_LN复合降秩: Qwen3 compound PR=0.82 vs W_active PR=0.97; DS7B L16无额外降秩
- 修正传播方程: ratio(k)=alpha*pres+beta*contrib+gamma; Qwen3 R2=0.88, DS7B R2=0.39
- 激活模式预测>>v_max预测: Qwen3误差0.022 vs 0.046; DS7B 0.035 vs 0.124
- DS7B因果链三重遮蔽: MLP贡献0.007-0.08, LN缩放0.005-0.01, lambda_max与非线性比正相关r=0.75

Phase CXVII-CXVIII核心思路:
1. P538: 频谱力学的因果修正 — 用preservation修正频谱力学方程
   - 原频谱力学: spectral_force(l,k) = lambda_max(l) * |v_max(l)^T * u_k|^2
   - 修正: spectral_force(l,k) = alpha*preservation(l,k) + beta*contribution(l,k)
   - 对DS7B是否有效?

2. P539: DS7B中层1-秩的真正机制 — W_down*W_up投影结构分析
   - 如果不是J_LN, 那是什么使Jacobian仍1-秩?
   - 假设: W_down[:, active] * W_up[active, :] 的乘积矩阵本身极度低秩
   - 即使W_down[:, active]不是1-秩, 乘以W_up后可能降为1-秩

3. P540: 从激活模式推导统一频谱力学方程
   - 用f_mlp的W_U对齐度(而非v_max)作为频谱力学的核心变量
   - unified_force(l,k) = |f_mlp(l)^T * W_down[:, active]^T * u_k|^2

4. P541: 残差保持的精确模型 — 为什么DS7B preservation(k)接近常数0.99?
   - 假设: h_in的范数远大于delta_h, 所以h_out约等于h_in
   - 验证: ||h_in|| / ||h_out|| 在不同层的分布
   - 精确模型: preservation(k) = h_in_W_U_topk / h_out_W_U_topk

使用方法:
    python phase_cxvii_spectral_deepening.py --model qwen3 --experiment p538
    python phase_cxvii_spectral_deepening.py --model glm4 --experiment p539
    python phase_cxvii_spectral_deepening.py --model deepseek7b --experiment p540
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


def compute_f_mlp(h_in, W_gate, W_up, post_ln_w):
    h_in_centered = h_in - h_in.mean()
    sigma = np.std(h_in_centered) + 1e-10
    ln_h_in = h_in_centered / sigma
    if post_ln_w is not None:
        ln_h_in = ln_h_in * post_ln_w
    if W_gate is not None:
        gate_pre = W_gate @ ln_h_in
        gate_act = 1.0 / (1.0 + np.exp(-gate_pre))
    else:
        gate_act = np.ones(W_up.shape[0])
    up_out = W_up @ ln_h_in
    f_mlp = gate_act * up_out
    return f_mlp, ln_h_in


def compute_jacobian_probes(model, layers, l_idx, inputs, info, n_probes=30, alpha=0.01):
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_hidden_states=True)
        h_out_baseline = outputs.hidden_states[l_idx + 1][0, -1].detach().clone()
    J_probes = []
    for i in range(n_probes):
        orig_w = layers[l_idx].mlp.down_proj.weight.data.clone()
        layers[l_idx].mlp.down_proj.weight.data = orig_w * (1 - alpha)
        with torch.no_grad():
            out = model(inputs["input_ids"], output_hidden_states=True)
            h_perturbed = out.hidden_states[l_idx + 1][0, -1].detach().clone()
        layers[l_idx].mlp.down_proj.weight.data = orig_w
        delta_h = (h_perturbed - h_out_baseline) / alpha
        J_probes.append(delta_h.cpu().float())
        if i % 20 == 0:
            torch.cuda.empty_cache()
    J_proj = torch.stack(J_probes)
    return J_proj, h_out_baseline


# ===== P538: 频谱力学的因果修正 =====
def run_p538(model, tokenizer, device, model_name):
    """
    频谱力学的因果修正
    
    原频谱力学: spectral_force(l,k) = lambda_max(l) * |v_max(l)^T * u_k|^2
    预测 ratio(k) = sum_l spectral_force(l,k) / sum_l sum_k spectral_force(l,k)
    
    修正: 用preservation和contribution替代lambda_max*v_max
    ratio(k) = alpha * preservation(k) + beta * contribution(k) + gamma
    
    但这是全局方程。层级别修正:
    layer_contribution(l,k) = preservation(l,k) + contribution(l,k)
    
    其中:
    preservation(l,k) = h_in在W_U top-k上的能量 / h_in总能量 (残差保持)
    contribution(l,k) = delta_h在W_U top-k上的能量 / delta_h总能量 (MLP贡献)
    
    验证: 逐层累积的修正预测是否收敛到实测ratio(k)
    """
    print("\n" + "="*70)
    print("P538: 频谱力学的因果修正")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    test_text = "The fundamental nature of reality can be understood through mathematical structures that underlie physical phenomena."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    sample_layers = get_sample_layers(info.n_layers, min(8, info.n_layers))
    print(f"采样层: {sample_layers}")
    
    # W_U SVD
    W_U = get_W_U(model)
    k_wu = min(200, min(W_U.shape) - 2)
    print(f"计算W_U SVD (k={k_wu})...")
    U_wu, S_wu, _ = svds(W_U.T.astype(np.float32), k=k_wu)
    U_wu = U_wu[:, ::-1]
    
    target_k_values = [10, 20, 50, 100]
    
    # 逐层收集preservation和contribution
    layer_spectra = []
    
    for l_idx in sample_layers:
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
            h_states = outputs.hidden_states
            h_in = h_states[l_idx][0, -1].cpu().float().numpy()
            h_out = h_states[l_idx + 1][0, -1].cpu().float().numpy()
        
        delta_h = h_out - h_in
        
        # h_in和delta_h在W_U空间中的频谱
        h_in_wu = U_wu.T @ h_in
        h_in_wu_e = h_in_wu**2
        h_in_total = np.sum(h_in_wu_e) + 1e-10
        
        delta_wu = U_wu.T @ delta_h
        delta_wu_e = delta_wu**2
        delta_total = np.sum(delta_wu_e) + 1e-10
        
        h_out_wu = U_wu.T @ h_out
        h_out_wu_e = h_out_wu**2
        h_out_total = np.sum(h_out_wu_e) + 1e-10
        
        # 逐k的preservation和contribution
        pres_k = {}
        contrib_k = {}
        for k in target_k_values:
            top_k_idx = np.argsort(h_out_wu_e)[-k:]
            pres_k[k] = float(np.sum(h_in_wu_e[top_k_idx]) / h_in_total)
            contrib_k[k] = float(np.sum(delta_wu_e[top_k_idx]) / delta_total)
        
        layer_spectra.append({
            'layer': l_idx,
            'h_in_total': h_in_total,
            'delta_total': delta_total,
            'h_out_total': h_out_total,
            'pres_k': pres_k,
            'contrib_k': contrib_k,
            'h_out_wu_e': h_out_wu_e,
        })
        
        print(f"  L{l_idx}: h_in={h_in_total:.2f}, delta={delta_total:.2f}, "
              f"pres(50)={pres_k[50]:.4f}, contrib(50)={contrib_k[50]:.4f}")
        torch.cuda.empty_cache()
    
    # ===== 逐层累积的修正预测 =====
    print("\n--- 逐层累积预测 ---")
    
    # 最终实测ratio(k)
    final_h_out_wu_e = layer_spectra[-1]['h_out_wu_e']
    final_total = layer_spectra[-1]['h_out_total']
    actual_ratios = {}
    for k in target_k_values:
        top_k_idx = np.argsort(final_h_out_wu_e)[-k:]
        actual_ratios[k] = float(np.sum(final_h_out_wu_e[top_k_idx]) / final_total)
    
    # 修正频谱力学: 逐层累积
    # 核心思想: 每层对W_U top-k方向的贡献 = preservation权重的信号 + contribution权重的MLP
    # ratio(k) = (sum_l h_in_l * pres_k(l) + delta_l * contrib_k(l)) / (sum_l (h_in_l + delta_l))
    # 但这不对, 因为h_out = h_in + delta, 信号是串联的
    
    # 正确的累积: 信号从L0到L_last逐层通过
    # 经过层l后: h_out(l) = h_in(l) + delta(l)
    # h_in(l+1) = h_out(l)
    # 所以最终h_out的W_U频谱 = 逐层累积
    
    # 简化模型: 直接用最终层的ratio与中间层的preservation/contribution的关系
    # 线性回归: ratio(k) = a * mean_preservation(k) + b * mean_contribution(k) + c
    
    mean_pres = {}
    mean_contrib = {}
    for k in target_k_values:
        mean_pres[k] = np.mean([ls['pres_k'][k] for ls in layer_spectra])
        mean_contrib[k] = np.mean([ls['contrib_k'][k] for ls in layer_spectra])
    
    # 拟合修正频谱力学方程
    all_actual = []
    all_pres = []
    all_contrib = []
    
    for ls in layer_spectra:
        for k in target_k_values:
            # 用该层的preservation和contribution预测该层的ratio
            top_k_idx = np.argsort(ls['h_out_wu_e'])[-k:]
            layer_ratio = float(np.sum(ls['h_out_wu_e'][top_k_idx]) / (ls['h_out_total'] + 1e-10))
            all_actual.append(layer_ratio)
            all_pres.append(ls['pres_k'][k])
            all_contrib.append(ls['contrib_k'][k])
    
    all_actual = np.array(all_actual)
    all_pres = np.array(all_pres)
    all_contrib = np.array(all_contrib)
    
    # 修正频谱力学: ratio(k) = a * pres(k) + b * contrib(k) + c
    A = np.column_stack([all_pres, all_contrib, np.ones_like(all_actual)])
    params, _, _, _ = np.linalg.lstsq(A, all_actual, rcond=None)
    a_fit, b_fit, c_fit = params
    
    predicted = a_fit * all_pres + b_fit * all_contrib + c_fit
    error = np.abs(predicted - all_actual)
    mean_error = float(np.mean(error))
    r2 = 1 - np.sum(error**2) / (np.sum((all_actual - np.mean(all_actual))**2) + 1e-10)
    
    # 对比原频谱力学(v_max)
    print("\n--- 对比: 修正频谱力学 vs 原频谱力学 ---")
    
    # 原频谱力学: 用v_max方向预测(简化: 用Jacobian probe)
    # 由于需要Jacobian, 只在采样层计算
    vmax_errors = []
    corrected_errors = []
    
    for ls in layer_spectra:
        l_idx = ls['layer']
        # 获取Jacobian
        J_proj, _ = compute_jacobian_probes(model, layers, l_idx, inputs, info, n_probes=60)
        J_proj_T = J_proj.T
        U_jt, S_j, _ = torch.linalg.svd(J_proj_T, full_matrices=False)
        v_max = U_jt[:, 0].cpu().float().numpy()
        v_max = v_max / (np.linalg.norm(v_max) + 1e-10)
        lambda_max = float(S_j[0])
        
        # v_max预测的ratio(k)
        vmax_wu = U_wu.T @ v_max
        vmax_wu_e = vmax_wu**2
        vmax_total = np.sum(vmax_wu_e) + 1e-10
        
        for k in target_k_values:
            top_k_idx = np.argsort(ls['h_out_wu_e'])[-k:]
            actual_ratio = float(np.sum(ls['h_out_wu_e'][top_k_idx]) / (ls['h_out_total'] + 1e-10))
            
            vmax_topk = np.sum(np.sort(vmax_wu_e)[-k:]) / vmax_total
            vmax_err = abs(vmax_topk - actual_ratio)
            
            corrected_pred = a_fit * ls['pres_k'][k] + b_fit * ls['contrib_k'][k] + c_fit
            corrected_err = abs(corrected_pred - actual_ratio)
            
            vmax_errors.append(vmax_err)
            corrected_errors.append(corrected_err)
        
        torch.cuda.empty_cache()
    
    print(f"  原频谱力学(v_max)平均误差: {np.mean(vmax_errors):.4f}")
    print(f"  修正频谱力学平均误差: {np.mean(corrected_errors):.4f}")
    improvement = (np.mean(vmax_errors) - np.mean(corrected_errors)) / np.mean(vmax_errors) * 100
    print(f"  改善: {improvement:.1f}%")
    
    # 汇总
    print("\n" + "="*70)
    print("P538 汇总: 频谱力学因果修正")
    print("="*70)
    
    print(f"  修正方程: ratio(k) = {a_fit:.4f} * preservation(k) + {b_fit:.4f} * contribution(k) + {c_fit:.4f}")
    print(f"  R2 = {r2:.4f}")
    print(f"  原频谱力学误差: {np.mean(vmax_errors):.4f}")
    print(f"  修正频谱力学误差: {np.mean(corrected_errors):.4f}")
    print(f"  改善: {improvement:.1f}%")
    
    if improvement > 30:
        print("  >> 修正频谱力学显著优于原方程!")
    elif improvement > 10:
        print("  >> 修正频谱力学有改善, 但不够显著")
    else:
        print("  >> 修正未带来显著改善")
    
    results = [{
        'a': float(a_fit),
        'b': float(b_fit),
        'c': float(c_fit),
        'r2': float(r2),
        'mean_vmax_error': float(np.mean(vmax_errors)),
        'mean_corrected_error': float(np.mean(corrected_errors)),
        'improvement_pct': float(improvement),
        'mean_pres_50': float(mean_pres[50]),
        'mean_contrib_50': float(mean_contrib[50]),
    }]
    
    result_path = f"tests/glm5/results/phase_cxvii/p538_{model_name}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({'results': [to_native(r) for r in results], 'model': model_name}, f, indent=2)
    print(f"\n结果已保存: {result_path}")
    
    return results


# ===== P539: DS7B中层1-秩的真正机制 =====
def run_p539(model, tokenizer, device, model_name):
    """
    DS7B中层1-秩的真正机制: W_down*W_up投影结构分析
    
    Phase CXIV发现: DS7B中层PR(W_active)=0.49但Jacobian仍1-秩
    Phase CXV发现: J_LN没有额外降秩
    
    新假设: W_down[:, active] * W_up[active, :] 的乘积矩阵本身极度低秩
    即使W_down[:, active]不是1-秩, 但W_down[:, active] * diag(f_active) * W_up[active, :]
    的乘积可能因为"有效秩坍缩"而成为1-秩
    
    原理: 
    M = W_down[:, k] * diag(f_k) * W_up[k, :]  -- [d_model, d_model]
    如果f_k使某一行特别大, 则M近似为外积
    或者W_up[k, :]的各行高度相关, 则乘积低秩
    
    验证:
    1. W_down[:, top_k] * diag(f_topk) * W_up[top_k, :] 的参与率
    2. 乘积矩阵M的秩 vs W_down[:, top_k]的秩
    3. W_up[top_k, :]的参与率
    """
    print("\n" + "="*70)
    print("P539: DS7B中层1-秩的真正机制")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    test_text = "The fundamental nature of reality can be understood through mathematical structures that underlie physical phenomena."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    sample_layers = get_sample_layers(info.n_layers, min(6, info.n_layers))
    print(f"采样层: {sample_layers}")
    
    results = []
    
    for l_idx in sample_layers:
        print(f"\n--- 层 {l_idx} ---")
        
        weights = get_layer_weights(layers[l_idx], info.d_model, info.mlp_type)
        W_down = weights.W_down       # [d_model, intermediate]
        W_gate = weights.W_gate
        W_up = weights.W_up           # [intermediate, d_model]
        post_ln_w = weights.post_attn_layernorm_weight
        
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
            h_in = outputs.hidden_states[l_idx][0, -1].cpu().float().numpy()
        
        f_mlp, ln_h_in = compute_f_mlp(h_in, W_gate, W_up, post_ln_w)
        f_mlp_abs = np.abs(f_mlp)
        
        # σ'(z)
        if W_gate is not None:
            gate_pre = W_gate @ ln_h_in
            gate_act = 1.0 / (1.0 + np.exp(-gate_pre))
            gate_deriv = gate_act * (1 - gate_act)
        else:
            gate_deriv = np.ones(W_up.shape[0])
        
        f_deriv = gate_deriv * (W_up @ ln_h_in)
        f_deriv_abs = np.abs(f_deriv)
        
        # ===== 分析1: W_down[:, top_k] 的PR =====
        for k_active in [5, 10, 20]:
            top_k_idx = np.argsort(f_deriv_abs)[-k_active:]
            
            W_down_k = W_down[:, top_k_idx]  # [d_model, k_active]
            f_deriv_k = f_deriv[top_k_idx]
            W_up_k = W_up[top_k_idx, :]      # [k_active, d_model]
            
            # W_down_k * diag(f_deriv_k) 的PR
            W_active = W_down_k * f_deriv_k[np.newaxis, :]
            if min(W_active.shape) > 1:
                _, S_wa, _ = svds(W_active.astype(np.float32), k=min(5, min(W_active.shape)-1))
                S_wa = S_wa[::-1]
                PR_W_active = float(S_wa.sum()**2 / (len(S_wa) * (S_wa**2).sum() + 1e-10))
            else:
                PR_W_active = 1.0
            
            # W_up_k 的PR
            if min(W_up_k.shape) > 1:
                _, S_wu_k, _ = svds(W_up_k.astype(np.float32), k=min(5, min(W_up_k.shape)-1))
                S_wu_k = S_wu_k[::-1]
                PR_W_up_k = float(S_wu_k.sum()**2 / (len(S_wu_k) * (S_wu_k**2).sum() + 1e-10))
            else:
                PR_W_up_k = 1.0
            
            # ===== 分析2: 乘积矩阵 M = W_down_k * diag(f_k) * W_up_k =====
            # 这是[d_model, d_model]的矩阵
            # 但d_model可能很大, 用近似方法
            # M = (W_down_k * f_k) @ W_up_k = W_active @ W_up_k
            # M的秩 <= min(rank(W_active), rank(W_up_k))
            # 如果W_up_k的PR低, 则M的PR更低
            
            # 不直接计算M(太大), 而是看M的"有效秩"
            # rank(M) <= min(k_active, rank(W_up_k))
            # 有效秩 = (sum(S))^2 / (k * sum(S^2))
            
            # 更精确: M = W_active @ W_up_k
            # SVD of M ≈ 用随机投影估计
            # 简化: 用PR(W_active) * PR(W_up_k)估计
            PR_product_est = PR_W_active * PR_W_up_k
            
            # 实际计算小矩阵的PR
            # M_samples = W_active @ W_up_k[:, :50] (采样50维)
            n_sample_dims = min(50, W_up_k.shape[1])
            M_sample = W_active @ W_up_k[:, :n_sample_dims]  # [d_model, 50]
            
            if min(M_sample.shape) > 1:
                _, S_M, _ = svds(M_sample.astype(np.float32), k=min(5, min(M_sample.shape)-1))
                S_M = S_M[::-1]
                PR_M_sample = float(S_M.sum()**2 / (len(S_M) * (S_M**2).sum() + 1e-10))
            else:
                PR_M_sample = 1.0
            
            if k_active == 10:
                result_k10 = {
                    'PR_W_active_k10': PR_W_active,
                    'PR_W_up_k10': PR_W_up_k,
                    'PR_product_est_k10': PR_product_est,
                    'PR_M_sample_k10': PR_M_sample,
                }
        
        # ===== 分析3: f_deriv的分布 — 是否极端集中 =====
        f_deriv_sorted = np.sort(f_deriv_abs)[::-1]
        f_top1_ratio = float(f_deriv_sorted[0]**2 / (np.sum(f_deriv_sorted**2) + 1e-10))
        f_top5_ratio = float(np.sum(f_deriv_sorted[:5]**2) / (np.sum(f_deriv_sorted**2) + 1e-10))
        f_top10_ratio = float(np.sum(f_deriv_sorted[:10]**2) / (np.sum(f_deriv_sorted**2) + 1e-10))
        
        # ===== 分析4: Jacobian验证 =====
        J_proj, _ = compute_jacobian_probes(model, layers, l_idx, inputs, info, n_probes=60)
        J_proj_T = J_proj.T
        U_jt, S_j, _ = torch.linalg.svd(J_proj_T, full_matrices=False)
        S_j_np = S_j.cpu().numpy()
        total_energy = np.sum(S_j_np**2)
        n_90 = int(np.searchsorted(np.cumsum(S_j_np**2) / total_energy, 0.9)) + 1
        top1_J = float(S_j_np[0]**2 / total_energy)
        
        result = {
            'layer': l_idx,
            'n_90_J': n_90,
            'top1_J': top1_J,
            **result_k10,
            'f_top1_ratio': f_top1_ratio,
            'f_top5_ratio': f_top5_ratio,
            'f_top10_ratio': f_top10_ratio,
        }
        results.append(result)
        
        print(f"  n_90_J={n_90}, top1_J={top1_J:.6f}")
        print(f"  PR(W_active_k10)={result_k10['PR_W_active_k10']:.4f}")
        print(f"  PR(W_up_k10)={result_k10['PR_W_up_k10']:.4f}")
        print(f"  PR(M_sample_k10)={result_k10['PR_M_sample_k10']:.4f}")
        print(f"  f_deriv集中度: top1={f_top1_ratio:.4f}, top5={f_top5_ratio:.4f}, top10={f_top10_ratio:.4f}")
        
        # 判断
        if result_k10['PR_M_sample_k10'] < 0.5 and n_90 == 1:
            print(f"  >> 乘积矩阵极度低秩(PR={result_k10['PR_M_sample_k10']:.4f}) -> 1-秩来自W_down*W_up投影!")
        
        torch.cuda.empty_cache()
    
    # 汇总
    print("\n" + "="*70)
    print("P539 汇总: 1-秩真正机制")
    print("="*70)
    
    mean_PR_active = np.mean([r['PR_W_active_k10'] for r in results])
    mean_PR_Wup = np.mean([r['PR_W_up_k10'] for r in results])
    mean_PR_M = np.mean([r['PR_M_sample_k10'] for r in results])
    mean_f_top1 = np.mean([r['f_top1_ratio'] for r in results])
    mean_f_top5 = np.mean([r['f_top5_ratio'] for r in results])
    
    print(f"  mean PR(W_active_k10): {mean_PR_active:.4f}")
    print(f"  mean PR(W_up_k10): {mean_PR_Wup:.4f}")
    print(f"  mean PR(M_sample_k10): {mean_PR_M:.4f}")
    print(f"  mean f_deriv top1: {mean_f_top1:.4f}")
    print(f"  mean f_deriv top5: {mean_f_top5:.4f}")
    
    print("\n  机制分析:")
    if mean_PR_M < 0.5:
        print(f"  >> W_down*W_up乘积矩阵极度低秩(PR={mean_PR_M:.4f}) -> 1-秩来自投影结构!")
    elif mean_f_top1 > 0.5:
        print(f"  >> f_deriv极度集中(top1={mean_f_top1:.4f}) -> 1-秩来自激活集中!")
    else:
        print("  >> 需要进一步分析1-秩的来源")
    
    result_path = f"tests/glm5/results/phase_cxvii/p539_{model_name}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({'results': [to_native(r) for r in results], 'model': model_name}, f, indent=2)
    print(f"\n结果已保存: {result_path}")
    
    return results


# ===== P540: 从激活模式推导统一频谱力学方程 =====
def run_p540(model, tokenizer, device, model_name):
    """
    从激活模式推导统一频谱力学方程
    
    P536发现: 激活模式预测远优于v_max预测
    核心洞察: f_mlp的激活模式决定了W_down哪些列被"选择"
    这些被选择的列的W_U对齐度决定了MLP输出在W_U空间中的分布
    
    统一频谱力学:
    spectral_force(l,k) = |sum_i f_mlp_i * (W_down[:,i]^T * u_k)|^2
                        = |f_mlp^T * W_down^T * u_k|^2
                        = |delta_predicted^T * u_k|^2
    
    其中 delta_predicted = W_down @ f_mlp
    
    验证:
    1. 从f_mlp预测的delta_h频谱 vs 实测delta_h频谱
    2. 逐层累积的预测ratio(k) vs 实测ratio(k)
    """
    print("\n" + "="*70)
    print("P540: 从激活模式推导统一频谱力学方程")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    test_text = "The fundamental nature of reality can be understood through mathematical structures that underlie physical phenomena."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    sample_layers = get_sample_layers(info.n_layers, min(8, info.n_layers))
    print(f"采样层: {sample_layers}")
    
    # W_U SVD
    W_U = get_W_U(model)
    k_wu = min(200, min(W_U.shape) - 2)
    print(f"计算W_U SVD (k={k_wu})...")
    U_wu, S_wu, _ = svds(W_U.T.astype(np.float32), k=k_wu)
    U_wu = U_wu[:, ::-1]
    
    target_k_values = [10, 20, 50, 100]
    results = []
    
    for l_idx in sample_layers:
        print(f"\n--- 层 {l_idx} ---")
        
        weights = get_layer_weights(layers[l_idx], info.d_model, info.mlp_type)
        W_down = weights.W_down
        W_gate = weights.W_gate
        W_up = weights.W_up
        post_ln_w = weights.post_attn_layernorm_weight
        
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
            h_states = outputs.hidden_states
            h_in = h_states[l_idx][0, -1].cpu().float().numpy()
            h_out = h_states[l_idx + 1][0, -1].cpu().float().numpy()
        
        delta_h = h_out - h_in
        
        # f_mlp
        f_mlp, ln_h_in = compute_f_mlp(h_in, W_gate, W_up, post_ln_w)
        
        # ===== 方法1: 激活模式预测delta_h =====
        # delta_predicted = W_down @ f_mlp
        delta_pred = W_down @ f_mlp
        
        # 频谱比较
        delta_wu = U_wu.T @ delta_h
        delta_wu_e = delta_wu**2
        delta_total = np.sum(delta_wu_e) + 1e-10
        
        pred_wu = U_wu.T @ delta_pred
        pred_wu_e = pred_wu**2
        pred_total = np.sum(pred_wu_e) + 1e-10
        
        # ===== 方法2: 统一频谱力学方程 =====
        # spectral_force(l,k) = |f_mlp^T * W_down^T * u_k|^2
        # = |delta_pred^T * u_k|^2 = pred_wu_e[k]
        
        # ===== 方法3: 加入residual的完整方程 =====
        # h_out = h_in + delta_h
        # ratio(k) = h_out在W_U top-k的能量 / h_out总能量
        # 用预测的delta: h_out_pred = h_in + delta_pred
        
        h_out_pred = h_in + delta_pred
        h_out_pred_wu = U_wu.T @ h_out_pred
        h_out_pred_wu_e = h_out_pred_wu**2
        h_out_pred_total = np.sum(h_out_pred_wu_e) + 1e-10
        
        h_out_wu = U_wu.T @ h_out
        h_out_wu_e = h_out_wu**2
        h_out_total = np.sum(h_out_wu_e) + 1e-10
        
        # 逐k比较
        layer_result = {'layer': l_idx}
        for k in target_k_values:
            # 实测ratio
            top_k_idx = np.argsort(h_out_wu_e)[-k:]
            actual_ratio = float(np.sum(h_out_wu_e[top_k_idx]) / h_out_total)
            
            # 激活模式预测(仅MLP部分)
            pred_topk = np.sum(np.sort(pred_wu_e)[-k:]) / pred_total
            
            # 完整预测(h_in + delta_pred)
            top_k_pred_idx = np.argsort(h_out_pred_wu_e)[-k:]
            full_pred_ratio = float(np.sum(h_out_pred_wu_e[top_k_pred_idx]) / h_out_pred_total)
            
            full_pred_err = abs(full_pred_ratio - actual_ratio)
            mlp_only_err = abs(pred_topk - np.sum(np.sort(delta_wu_e)[-k:]) / delta_total)
            
            layer_result[f'actual_ratio_{k}'] = actual_ratio
            layer_result[f'full_pred_ratio_{k}'] = full_pred_ratio
            layer_result[f'full_pred_error_{k}'] = full_pred_err
            layer_result[f'mlp_pred_error_{k}'] = mlp_only_err
        
        # cos(delta_h, delta_pred)
        cos_delta_pred = abs(np.dot(delta_h, delta_pred) / (np.linalg.norm(delta_h) * np.linalg.norm(delta_pred) + 1e-10))
        layer_result['cos_delta_pred'] = cos_delta_pred
        
        results.append(layer_result)
        
        print(f"  cos(delta_h, delta_pred)={cos_delta_pred:.4f}")
        for k in target_k_values:
            print(f"  k={k}: actual={layer_result[f'actual_ratio_{k}']:.4f}, "
                  f"full_pred={layer_result[f'full_pred_ratio_{k}']:.4f}, "
                  f"error={layer_result[f'full_pred_error_{k}']:.4f}")
        
        torch.cuda.empty_cache()
    
    # 汇总
    print("\n" + "="*70)
    print("P540 汇总: 统一频谱力学方程")
    print("="*70)
    
    mean_cos = np.mean([r['cos_delta_pred'] for r in results])
    print(f"  mean cos(delta_h, delta_pred): {mean_cos:.4f}")
    
    for k in target_k_values:
        full_errors = [r[f'full_pred_error_{k}'] for r in results]
        mlp_errors = [r[f'mlp_pred_error_{k}'] for r in results]
        print(f"  k={k}: 完整预测误差={np.mean(full_errors):.4f}, MLP预测误差={np.mean(mlp_errors):.4f}")
    
    if mean_cos > 0.9:
        print("  >> W_down@f_mlp极好地近似了实际delta_h -> 统一方程可行!")
    elif mean_cos > 0.7:
        print("  >> W_down@f_mlp较好地近似了delta_h -> 统一方程基本可行")
    else:
        print("  >> W_down@f_mlp与delta_h差异较大 -> 需要更精确的模型")
    
    result_path = f"tests/glm5/results/phase_cxvii/p540_{model_name}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({'results': [to_native(r) for r in results], 'model': model_name}, f, indent=2)
    print(f"\n结果已保存: {result_path}")
    
    return results


# ===== P541: 残差保持的精确模型 =====
def run_p541(model, tokenizer, device, model_name):
    """
    残差保持的精确模型
    
    P535发现: DS7B的preservation(k)接近常数0.99
    这意味着: h_in的W_U频谱几乎完全保留到h_out
    
    假设: ||h_in|| >> ||delta_h||, 所以h_out ≈ h_in
    验证: ||h_in|| / ||h_out|| 和 ||delta_h|| / ||h_in|| 的分布
    
    精确模型:
    preservation(k) = h_in在W_U top-k方向上的能量 / h_in总能量
    (如果||delta_h|| << ||h_in||, 则preservation(k) ≈ h_in频谱的集中度)
    
    进一步: 为什么h_in的频谱集中度本身就接近0.99?
    因为: h_in = embedding + sum_l delta(l), 残差连接累积使h_in巨大
    而delta(l)的W_U频谱集中度也很高(DS7B top10=0.78)
    所以累积后h_in的频谱更集中
    
    验证:
    1. 逐层的||h_in||和||delta_h||的比值
    2. preservation(k) vs h_in的W_U频谱集中度
    3. 累积频谱模型: h_in(l)的频谱 = alpha*h_in(l-1)的频谱 + beta*delta(l)的频谱
    """
    print("\n" + "="*70)
    print("P541: 残差保持的精确模型")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    test_text = "The fundamental nature of reality can be understood through mathematical structures that underlie physical phenomena."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    sample_layers = get_sample_layers(info.n_layers, min(10, info.n_layers))
    print(f"采样层: {sample_layers}")
    
    # W_U SVD
    W_U = get_W_U(model)
    k_wu = min(200, min(W_U.shape) - 2)
    print(f"计算W_U SVD (k={k_wu})...")
    U_wu, S_wu, _ = svds(W_U.T.astype(np.float32), k=k_wu)
    U_wu = U_wu[:, ::-1]
    
    target_k_values = [10, 50, 100]
    results = []
    
    for l_idx in sample_layers:
        print(f"\n--- 层 {l_idx} ---")
        
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
            h_states = outputs.hidden_states
            h_in = h_states[l_idx][0, -1].cpu().float().numpy()
            h_out = h_states[l_idx + 1][0, -1].cpu().float().numpy()
        
        delta_h = h_out - h_in
        
        h_in_norm = np.linalg.norm(h_in)
        h_out_norm = np.linalg.norm(h_out)
        delta_norm = np.linalg.norm(delta_h)
        
        # h_in和h_out的W_U频谱
        h_in_wu = U_wu.T @ h_in
        h_in_wu_e = h_in_wu**2
        h_in_total = np.sum(h_in_wu_e) + 1e-10
        
        h_out_wu = U_wu.T @ h_out
        h_out_wu_e = h_out_wu**2
        h_out_total = np.sum(h_out_wu_e) + 1e-10
        
        delta_wu = U_wu.T @ delta_h
        delta_wu_e = delta_wu**2
        delta_total = np.sum(delta_wu_e) + 1e-10
        
        # 比值
        delta_to_hin = delta_norm / (h_in_norm + 1e-10)
        hin_to_hout = h_in_norm / (h_out_norm + 1e-10)
        
        # 逐k分析
        layer_result = {
            'layer': l_idx,
            'h_in_norm': h_in_norm,
            'delta_norm': delta_norm,
            'h_out_norm': h_out_norm,
            'delta_to_hin': delta_to_hin,
            'hin_to_hout': hin_to_hout,
        }
        
        for k in target_k_values:
            # h_in的W_U top-k集中度
            h_in_topk = float(np.sum(np.sort(h_in_wu_e)[-k:]) / h_in_total)
            
            # h_out的W_U top-k集中度
            h_out_topk = float(np.sum(np.sort(h_out_wu_e)[-k:]) / h_out_total)
            
            # delta的W_U top-k集中度
            delta_topk = float(np.sum(np.sort(delta_wu_e)[-k:]) / delta_total)
            
            # preservation(k): h_out在W_U top-k方向的能量中, h_in贡献的比例
            top_k_idx = np.argsort(h_out_wu_e)[-k:]
            h_in_in_topk = float(np.sum(h_in_wu_e[top_k_idx]) / h_in_total)
            
            layer_result[f'h_in_concentration_{k}'] = h_in_topk
            layer_result[f'h_out_concentration_{k}'] = h_out_topk
            layer_result[f'delta_concentration_{k}'] = delta_topk
            layer_result[f'preservation_{k}'] = h_in_in_topk
        
        # 频谱形状相似度
        # h_in频谱 vs h_out频谱的Pearson相关
        h_in_spectrum = h_in_wu_e / h_in_total
        h_out_spectrum = h_out_wu_e / h_out_total
        spectrum_corr, _ = pearsonr(h_in_spectrum, h_out_spectrum)
        layer_result['spectrum_corr'] = spectrum_corr
        
        # 累积模型: h_out频谱 = alpha * h_in频谱 + beta * delta频谱
        h_in_spec = h_in_wu_e / h_in_total
        delta_spec = delta_wu_e / delta_total
        h_out_spec = h_out_wu_e / h_out_total
        
        # 线性拟合: h_out_spec = alpha * h_in_spec + beta * delta_spec
        A = np.column_stack([h_in_spec, delta_spec, np.ones_like(h_out_spec)])
        params, _, _, _ = np.linalg.lstsq(A, h_out_spec, rcond=None)
        alpha_spec, beta_spec, gamma_spec = params
        
        layer_result['alpha_spectrum'] = float(alpha_spec)
        layer_result['beta_spectrum'] = float(beta_spec)
        
        results.append(layer_result)
        
        print(f"  ||delta||/||h_in||={delta_to_hin:.4f}, ||h_in||/||h_out||={hin_to_hout:.4f}")
        print(f"  h_in集中度(k50)={layer_result['h_in_concentration_50']:.4f}")
        print(f"  h_out集中度(k50)={layer_result['h_out_concentration_50']:.4f}")
        print(f"  preservation(k50)={layer_result['preservation_50']:.4f}")
        print(f"  频谱相关={spectrum_corr:.4f}")
        print(f"  频谱分解: alpha={alpha_spec:.4f}, beta={beta_spec:.4f}")
        
        torch.cuda.empty_cache()
    
    # 汇总
    print("\n" + "="*70)
    print("P541 汇总: 残差保持精确模型")
    print("="*70)
    
    mean_delta_hin = np.mean([r['delta_to_hin'] for r in results])
    mean_hin_hout = np.mean([r['hin_to_hout'] for r in results])
    mean_pres_50 = np.mean([r['preservation_50'] for r in results])
    mean_hin_conc = np.mean([r['h_in_concentration_50'] for r in results])
    mean_spec_corr = np.mean([r['spectrum_corr'] for r in results])
    mean_alpha = np.mean([r['alpha_spectrum'] for r in results])
    mean_beta = np.mean([r['beta_spectrum'] for r in results])
    
    print(f"  mean ||delta||/||h_in||: {mean_delta_hin:.4f}")
    print(f"  mean ||h_in||/||h_out||: {mean_hin_hout:.4f}")
    print(f"  mean preservation(k50): {mean_pres_50:.4f}")
    print(f"  mean h_in集中度(k50): {mean_hin_conc:.4f}")
    print(f"  mean 频谱相关: {mean_spec_corr:.4f}")
    print(f"  mean 频谱分解: alpha={mean_alpha:.4f}, beta={mean_beta:.4f}")
    
    print("\n  精确模型:")
    if mean_delta_hin < 0.1:
        print(f"  >> delta远小于h_in({mean_delta_hin:.4f}) -> 残差保持近似完美")
        print(f"     preservation(k)约等于h_in的W_U集中度({mean_hin_conc:.4f})")
    if mean_alpha > 0.9:
        print(f"  >> h_out频谱主要由h_in频谱决定(alpha={mean_alpha:.4f})")
    if mean_spec_corr > 0.99:
        print(f"  >> h_in和h_out频谱极度相似(r={mean_spec_corr:.4f})")
        print("     -> 信号通过残差连接几乎无损传播")
    
    result_path = f"tests/glm5/results/phase_cxvii/p541_{model_name}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({'results': [to_native(r) for r in results], 'model': model_name}, f, indent=2)
    print(f"\n结果已保存: {result_path}")
    
    return results


# ===== 主函数 =====
EXPERIMENTS = {
    'p538': run_p538,
    'p539': run_p539,
    'p540': run_p540,
    'p541': run_p541,
}


def main():
    parser = argparse.ArgumentParser(description="Phase CXVII-CXVIII: 统一频谱力学的深化")
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"],
                        help="模型名称")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=list(EXPERIMENTS.keys()),
                        help="实验编号 (p538/p539/p540/p541)")
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Phase CXVII-CXVIII: {args.experiment.upper()}")
    print(f"模型: {args.model}")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    print(f"模型信息: {info.model_class}, {info.n_layers}层, d_model={info.d_model}")
    
    try:
        result = EXPERIMENTS[args.experiment](model, tokenizer, device, args.model)
    finally:
        release_model(model)
    
    print(f"\n实验 {args.experiment.upper()} 完成!")


if __name__ == "__main__":
    main()
