"""
Phase CXXV-CXXVI: 频谱力学的深度理论
P555: 频谱不动点的精确求解 — 从W_down和W_o的频谱推导不动点
P556: 为什么W_down频谱决定h频谱 — MLP激活的稀疏结构传递
P557: 频谱幂律指数的训练动力学 — 频谱与模型规模/深度的关系
P558: 统一理论的闭合验证 — 用5个参数预测完整频谱
"""

import argparse
import json
import os
import sys
import numpy as np
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(__file__))
from model_utils import load_model, get_model_info, get_W_U, get_sample_layers


def compute_wu_svd(model, k=200):
    """计算W_U的SVD"""
    W_U = get_W_U(model)
    d_model, n_vocab = W_U.shape
    k = min(k, min(d_model, n_vocab) - 1)
    U_wu, S_wu, Vt_wu = svds(W_U.T, k=k)
    sort_idx = np.argsort(S_wu)[::-1]
    U_wu = U_wu[:, sort_idx]
    S_wu = S_wu[sort_idx]
    return U_wu, S_wu, W_U


def project_and_spectrum(h, U_wu, k):
    """投影h到W_U空间并计算频谱"""
    coeffs = h @ U_wu[:, :k]
    spectrum = np.mean(coeffs**2, axis=0) if coeffs.ndim > 1 else coeffs**2
    total = np.sum(spectrum)
    if total > 0:
        spectrum = spectrum / total
    return spectrum


def weight_spectrum(W, U_wu, k=200):
    """计算权重矩阵在W_U空间中的频谱"""
    W_proj = U_wu[:, :k].T @ W  # [k, d_intermediate]
    spec = np.mean(W_proj**2, axis=1)
    spec = spec / (np.sum(spec) + 1e-10)
    return spec


def experiment_p555(model, tokenizer, device, model_name):
    """P555: 频谱不动点的精确求解"""
    print("\n" + "="*70)
    print("P555: 频谱不动点的精确求解")
    print("="*70)
    
    U_wu, S_wu, W_U = compute_wu_svd(model, k=200)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    k_wu = 200
    
    text = "The development of artificial intelligence has transformed many aspects of modern life."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model(inputs["input_ids"], output_hidden_states=True)
    h_states = outputs.hidden_states
    
    sample_layers = get_sample_layers(n_layers, 10)
    print(f"采样层: {sample_layers}")
    
    # === 步骤1: 收集W_down和W_o的频谱 ===
    print("\n--- 步骤1: W_down和W_o的频谱 ---")
    
    wdown_specs = []
    wo_specs = []
    gamma_specs = []
    
    for layer in sample_layers:
        try:
            W_down = model.model.layers[layer].mlp.down_proj.weight.detach().cpu().float().numpy()
            spec_wd = weight_spectrum(W_down, U_wu, k_wu)
            wdown_specs.append(spec_wd)
            
            W_o = model.model.layers[layer].self_attn.o_proj.weight.detach().cpu().float().numpy()
            spec_wo = weight_spectrum(W_o, U_wu, k_wu)
            wo_specs.append(spec_wo)
            
            gamma = model.model.layers[layer].input_layernorm.weight.detach().cpu().float().numpy()
            gamma_proj = gamma @ U_wu[:, :k_wu]
            gamma_spec = gamma_proj**2
            gamma_spec = gamma_spec / (np.sum(gamma_spec) + 1e-10)
            gamma_specs.append(gamma_spec)
        except:
            wdown_specs.append(np.zeros(k_wu))
            wo_specs.append(np.zeros(k_wu))
            gamma_specs.append(np.zeros(k_wu))
    
    # 平均频谱
    avg_wd_spec = np.mean(wdown_specs, axis=0)
    avg_wo_spec = np.mean(wo_specs, axis=0)
    avg_gamma_spec = np.mean(gamma_specs, axis=0)
    
    # === 步骤2: 实测h频谱 ===
    print("\n--- 步骤2: 实测h频谱 ---")
    
    h_specs = []
    for layer in sample_layers:
        h_l = h_states[layer][0, -1].detach().cpu().float().numpy()
        spec_h = project_and_spectrum(h_l, U_wu, k_wu)
        h_specs.append(spec_h)
    
    # 末层频谱作为"不动点"
    h_last = h_states[n_layers - 1][0, -1].detach().cpu().float().numpy()
    spec_last = project_and_spectrum(h_last, U_wu, k_wu)
    
    # === 步骤3: 从W_down频谱预测不动点 ===
    print("\n--- 步骤3: 从W_down频谱预测不动点 ---")
    
    # 假设1: 不动点频谱 ≈ W_down频谱 (因为W_down主导MLP输出)
    r_wd, _ = pearsonr(avg_wd_spec[:50], spec_last[:50])
    cos_wd = np.dot(avg_wd_spec[:50], spec_last[:50]) / (
        np.linalg.norm(avg_wd_spec[:50]) * np.linalg.norm(spec_last[:50]) + 1e-10)
    print(f"  W_down平均频谱 vs 末层: 相关={r_wd:.4f}, 余弦={cos_wd:.4f}")
    
    # 假设2: 不动点频谱 ≈ W_o频谱 (attention输出)
    r_wo, _ = pearsonr(avg_wo_spec[:50], spec_last[:50])
    cos_wo = np.dot(avg_wo_spec[:50], spec_last[:50]) / (
        np.linalg.norm(avg_wo_spec[:50]) * np.linalg.norm(spec_last[:50]) + 1e-10)
    print(f"  W_o平均频谱 vs 末层: 相关={r_wo:.4f}, 余弦={cos_wo:.4f}")
    
    # 假设3: 加权组合 alpha*W_down + beta*W_o
    # 用最小二乘求解
    A = np.vstack([avg_wd_spec[:50], avg_wo_spec[:50]]).T  # [50, 2]
    b = spec_last[:50]
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    spec_combined = coeffs[0] * avg_wd_spec + coeffs[1] * avg_wo_spec
    spec_combined = spec_combined / (np.sum(spec_combined) + 1e-10)
    r_comb, _ = pearsonr(spec_combined[:50], spec_last[:50])
    print(f"  W_down*{coeffs[0]:.3f} + W_o*{coeffs[1]:.3f} vs 末层: 相关={r_comb:.4f}")
    
    # === 步骤4: 逐层预测h频谱 ===
    print("\n--- 步骤4: 逐层用W_down频谱预测h频谱 ---")
    
    layer_preds = []
    for idx, layer in enumerate(sample_layers):
        if idx >= len(wdown_specs):
            break
        r, _ = pearsonr(wdown_specs[idx][:50], h_specs[idx][:50])
        layer_preds.append((layer, r))
        if idx < 8:
            print(f"  L{layer}: W_down频谱 vs h频谱 = {r:.4f}")
    
    avg_layer_pred = np.mean([r for _, r in layer_preds])
    print(f"  平均: {avg_layer_pred:.4f}")
    
    # === 步骤5: 不动点的自洽验证 ===
    print("\n--- 步骤5: 不动点自洽验证 ---")
    # 如果S*是不动点, 则 S* = alpha*S* + c*W_down_spec
    # => S* = c*W_down_spec / (1-alpha)
    
    # 计算alpha
    alphas = []
    for l_idx in range(1, len(sample_layers)):
        spec_prev = h_specs[l_idx - 1]
        spec_curr = h_specs[l_idx]
        alpha = np.dot(spec_curr[:50], spec_prev[:50]) / (np.dot(spec_prev[:50], spec_prev[:50]) + 1e-10)
        alphas.append(alpha)
    alpha = np.mean(alphas)
    
    # 最优c
    # S* ≈ c * W_down_spec / (1-alpha)
    # => c = S* * (1-alpha) / W_down_spec
    c_opt = spec_last * (1 - alpha) / (avg_wd_spec + 1e-10)
    c_mean = np.mean(c_opt[:50])
    
    spec_fixed = c_mean * avg_wd_spec / (1 - alpha + 1e-10)
    spec_fixed = spec_fixed / (np.sum(spec_fixed) + 1e-10)
    r_fixed, _ = pearsonr(spec_fixed[:50], spec_last[:50])
    cos_fixed = np.dot(spec_fixed[:50], spec_last[:50]) / (
        np.linalg.norm(spec_fixed[:50]) * np.linalg.norm(spec_last[:50]) + 1e-10)
    
    print(f"  alpha={alpha:.4f}, c={c_mean:.4f}")
    print(f"  不动点 S*=c*W_down/(1-alpha) vs 末层: 相关={r_fixed:.4f}, 余弦={cos_fixed:.4f}")
    
    if r_fixed > 0.9:
        print("  >> 不动点方程闭合: S*=c*W_down/(1-alpha) 有效!")
    else:
        print("  >> 不动点方程不闭合: 需要更多项")
    
    results = {
        'model': model_name, 'experiment': 'p555',
        'wd_vs_last_corr': float(r_wd),
        'wd_vs_last_cos': float(cos_wd),
        'wo_vs_last_corr': float(r_wo),
        'combined_vs_last_corr': float(r_comb),
        'combined_coeffs': [float(coeffs[0]), float(coeffs[1])],
        'avg_layer_pred_corr': float(avg_layer_pred),
        'alpha': float(alpha),
        'c_mean': float(c_mean),
        'fixed_point_corr': float(r_fixed),
        'fixed_point_cos': float(cos_fixed),
    }
    
    out_dir = 'results/phase_cxxv'
    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/P555_{model_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {out_dir}/P555_{model_name}.json")
    return results


def experiment_p556(model, tokenizer, device, model_name):
    """P556: 为什么W_down频谱决定h频谱"""
    print("\n" + "="*70)
    print("P556: 为什么W_down频谱决定h频谱")
    print("="*70)
    
    U_wu, S_wu, W_U = compute_wu_svd(model, k=200)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    k_wu = 200
    
    text = "The development of artificial intelligence has transformed many aspects of modern life."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model(inputs["input_ids"], output_hidden_states=True)
    h_states = outputs.hidden_states
    
    sample_layers = get_sample_layers(n_layers, 8)
    
    # === 分析1: MLP激活的稀疏性 ===
    print("\n--- 分析1: MLP激活稀疏性 vs 频谱传递 ---")
    
    for layer in sample_layers[:5]:
        try:
            W_gate = model.model.layers[layer].mlp.gate_proj.weight.detach().cpu().float().numpy()
            W_up = model.model.layers[layer].mlp.up_proj.weight.detach().cpu().float().numpy()
            W_down = model.model.layers[layer].mlp.down_proj.weight.detach().cpu().float().numpy()
            
            # 计算W_down在W_U空间中的频谱
            spec_wd = weight_spectrum(W_down, U_wu, k_wu)
            
            # 计算h频谱
            h_l = h_states[layer][0, -1].detach().cpu().float().numpy()
            spec_h = project_and_spectrum(h_l, U_wu, k_wu)
            
            r, _ = pearsonr(spec_wd[:50], spec_h[:50])
            
            # W_gate的稀疏性(Gini系数)
            gate_norms = np.linalg.norm(W_gate, axis=1)
            gate_sorted = np.sort(gate_norms)
            n = len(gate_sorted)
            gini = 2 * np.sum((np.arange(1, n+1)) * gate_sorted) / (n * np.sum(gate_sorted) + 1e-10) - (n+1)/n
            
            # W_down的列范数分布(哪一列/神经元贡献最大)
            wd_col_norms = np.linalg.norm(W_down, axis=0)
            top1_ratio = np.max(wd_col_norms) / (np.sum(wd_col_norms) + 1e-10)
            top10_ratio = np.sum(np.sort(wd_col_norms)[-10:]) / (np.sum(wd_col_norms) + 1e-10)
            
            print(f"  L{layer}: W_down-h相关={r:.3f}, Gini={gini:.3f}, top1={top1_ratio:.3f}, top10={top10_ratio:.3f}")
        except Exception as e:
            print(f"  L{layer}: 跳过 ({e})")
    
    # === 分析2: W_down列的空间结构 ===
    print("\n--- 分析2: W_down列在W_U空间中的方向 ---")
    
    for layer in sample_layers[:3]:
        try:
            W_down = model.model.layers[layer].mlp.down_proj.weight.detach().cpu().float().numpy()
            # W_down: [d_model, d_intermediate]
            # 计算W_down的每一列在W_U空间中的投影
            W_down_in_wu = U_wu[:, :k_wu].T @ W_down  # [k_wu, d_intermediate]
            
            # 每一列的范数
            col_norms = np.linalg.norm(W_down_in_wu, axis=0)
            top_cols = np.argsort(col_norms)[-5:][::-1]
            
            # 这些top列在W_U空间中的方向
            print(f"  L{layer}: Top-5 MLP神经元的频谱分布")
            for rank, col_idx in enumerate(top_cols):
                col_spec = W_down_in_wu[:, col_idx]**2
                col_spec = col_spec / (np.sum(col_spec) + 1e-10)
                # 这个神经元集中在哪些W_U方向?
                top3_dirs = np.argsort(col_spec)[-3:][::-1]
                print(f"    神经元{col_idx}: 范数={col_norms[col_idx]:.2f}, top-3 W_U方向={top3_dirs}, 占比={np.sum(col_spec[top3_dirs]):.3f}")
        except Exception as e:
            print(f"  L{layer}: 跳过 ({e})")
    
    # === 分析3: 激活稀疏性如何传递频谱 ===
    print("\n--- 分析3: 激活稀疏性与频谱传递的关系 ---")
    
    # 理论: 如果只有少数MLP神经元被激活, h的频谱 = sum(激活_i * W_down[:,i]的频谱)
    # 如果top-10神经元贡献了大部分能量, 且它们的W_down列有相似的频谱分布,
    # 那么h的频谱就继承了W_down列的平均频谱
    
    sparsity_spectrum_corr = []
    for layer in sample_layers[:5]:
        try:
            W_down = model.model.layers[layer].mlp.down_proj.weight.detach().cpu().float().numpy()
            spec_wd = weight_spectrum(W_down, U_wu, k_wu)
            
            h_l = h_states[layer][0, -1].detach().cpu().float().numpy()
            spec_h = project_and_spectrum(h_l, U_wu, k_wu)
            
            r, _ = pearsonr(spec_wd[:50], spec_h[:50])
            
            # W_down列的频谱相似度
            W_down_in_wu = U_wu[:, :k_wu].T @ W_down
            col_norms = np.linalg.norm(W_down_in_wu, axis=0)
            top_cols = np.argsort(col_norms)[-20:]
            
            # top-20列的频谱间的平均相关
            col_corrs = []
            for i in range(len(top_cols)):
                for j in range(i+1, len(top_cols)):
                    ci = W_down_in_wu[:, top_cols[i]]
                    cj = W_down_in_wu[:, top_cols[j]]
                    ci_spec = ci**2 / (np.sum(ci**2) + 1e-10)
                    cj_spec = cj**2 / (np.sum(cj**2) + 1e-10)
                    rc, _ = pearsonr(ci_spec, cj_spec)
                    col_corrs.append(rc)
            
            avg_col_corr = np.mean(col_corrs) if col_corrs else 0
            sparsity_spectrum_corr.append((layer, r, avg_col_corr))
            print(f"  L{layer}: W_down-h相关={r:.3f}, top-20列频谱间相关={avg_col_corr:.3f}")
        except:
            pass
    
    avg_wd_h = np.mean([r for _, r, _ in sparsity_spectrum_corr]) if sparsity_spectrum_corr else 0
    avg_col_corr = np.mean([c for _, _, c in sparsity_spectrum_corr]) if sparsity_spectrum_corr else 0
    print(f"  平均W_down-h相关: {avg_wd_h:.4f}")
    print(f"  平均top列频谱间相关: {avg_col_corr:.4f}")
    
    if avg_col_corr > 0.8:
        print("  >> Top MLP神经元的频谱高度一致: 激活稀疏性有效传递频谱!")
    else:
        print("  >> Top MLP神经元的频谱不一致: 稀疏性传递机制需要修正")
    
    # === 分析4: W_up和W_gate的角色 ===
    print("\n--- 分析4: W_up和W_gate的频谱 ---")
    
    for layer in sample_layers[:3]:
        try:
            W_down = model.model.layers[layer].mlp.down_proj.weight.detach().cpu().float().numpy()
            W_up = model.model.layers[layer].mlp.up_proj.weight.detach().cpu().float().numpy()
            W_gate = model.model.layers[layer].mlp.gate_proj.weight.detach().cpu().float().numpy()
            
            spec_wd = weight_spectrum(W_down, U_wu, k_wu)
            spec_wu_mlp = weight_spectrum(W_up.T, U_wu, k_wu)  # W_up.T: [d_intermediate, d_model]
            spec_wg = weight_spectrum(W_gate.T, U_wu, k_wu)
            
            h_l = h_states[layer][0, -1].detach().cpu().float().numpy()
            spec_h = project_and_spectrum(h_l, U_wu, k_wu)
            
            r_wd, _ = pearsonr(spec_wd[:50], spec_h[:50])
            r_wu, _ = pearsonr(spec_wu_mlp[:50], spec_h[:50])
            r_wg, _ = pearsonr(spec_wg[:50], spec_h[:50])
            
            print(f"  L{layer}: W_down-h={r_wd:.3f}, W_up-h={r_wu:.3f}, W_gate-h={r_wg:.3f}")
        except:
            pass
    
    results = {
        'model': model_name, 'experiment': 'p556',
        'avg_wd_h_corr': float(avg_wd_h),
        'avg_col_spectrum_corr': float(avg_col_corr),
    }
    
    out_dir = 'results/phase_cxxv'
    with open(f'{out_dir}/P556_{model_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {out_dir}/P556_{model_name}.json")
    return results


def experiment_p557(model, tokenizer, device, model_name):
    """P557: 频谱幂律指数与模型深度的关系"""
    print("\n" + "="*70)
    print("P557: 频谱幂律指数与模型深度的关系")
    print("="*70)
    
    U_wu, S_wu, W_U = compute_wu_svd(model, k=200)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    k_wu = 200
    
    text = "The development of artificial intelligence has transformed many aspects of modern life."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model(inputs["input_ids"], output_hidden_states=True)
    h_states = outputs.hidden_states
    
    # === 分析1: 频谱幂律指数随层的变化 ===
    print("\n--- 分析1: 频谱幂律指数vs层深度 ---")
    
    layer_betas = []
    layer_ratios = []
    
    for layer in range(n_layers):
        h_l = h_states[layer][0, -1].detach().cpu().float().numpy()
        spec_h = project_and_spectrum(h_l, U_wu, k_wu)
        
        # 拟合幂律
        valid = spec_h[:100] > 1e-10
        if np.sum(valid) > 5:
            log_i = np.log(np.arange(1, 101)[valid])
            log_S = np.log(spec_h[:100][valid])
            try:
                coeffs = np.polyfit(log_i, log_S, 1)
                beta = -coeffs[0]
                SS_res = np.sum((log_S - np.polyval(coeffs, log_i))**2)
                SS_tot = np.sum((log_S - np.mean(log_S))**2)
                R2 = 1 - SS_res / (SS_tot + 1e-10)
            except:
                beta = 0
                R2 = 0
        else:
            beta = 0
            R2 = 0
        
        # ratio(50)
        ratio50 = np.sum(spec_h[:50])
        
        layer_betas.append((layer, beta, R2))
        layer_ratios.append((layer, ratio50))
        
        if layer % 5 == 0 or layer == n_layers - 1:
            print(f"  L{layer}: beta={beta:.3f}, R2={R2:.3f}, ratio(50)={ratio50:.4f}")
    
    # 趋势分析
    betas = [b for _, b, _ in layer_betas]
    ratios = [r for _, r in layer_ratios]
    layers = list(range(n_layers))
    
    # 线性趋势
    if len(betas) > 2:
        beta_trend = np.polyfit(layers, betas, 1)
        ratio_trend = np.polyfit(layers, ratios, 1)
        print(f"\n  beta趋势: {beta_trend[0]:.6f}*L + {beta_trend[1]:.3f}")
        print(f"  ratio(50)趋势: {ratio_trend[0]:.6f}*L + {ratio_trend[1]:.3f}")
        
        if beta_trend[0] > 0.001:
            print("  >> beta随层增加: 频谱变得更陡峭(更集中)")
        elif beta_trend[0] < -0.001:
            print("  >> beta随层减少: 频谱变得更平坦(更分散)")
        else:
            print("  >> beta几乎不随层变化")
    
    # === 分析2: W_down频谱的层间演化 ===
    print("\n--- 分析2: W_down频谱的层间演化 ---")
    
    sample_layers = get_sample_layers(n_layers, 8)
    wdown_betas = []
    
    for layer in sample_layers:
        try:
            W_down = model.model.layers[layer].mlp.down_proj.weight.detach().cpu().float().numpy()
            spec_wd = weight_spectrum(W_down, U_wu, k_wu)
            
            valid = spec_wd[:100] > 1e-10
            if np.sum(valid) > 5:
                log_i = np.log(np.arange(1, 101)[valid])
                log_S = np.log(spec_wd[:100][valid])
                try:
                    coeffs = np.polyfit(log_i, log_S, 1)
                    beta_wd = -coeffs[0]
                except:
                    beta_wd = 0
            else:
                beta_wd = 0
            
            wdown_betas.append((layer, beta_wd))
            print(f"  L{layer}: W_down beta={beta_wd:.3f}")
        except:
            pass
    
    # === 分析3: 频谱形状与ratio(k)的解析关系 ===
    print("\n--- 分析3: 从beta预测ratio(50) ---")
    
    # 如果频谱是S(i)=i^(-beta), 则ratio(k) = sum(i^(-beta), i=1..k) / sum(i^(-beta), i=1..200)
    predicted_ratios = []
    actual_ratios = []
    
    for layer, beta, R2 in layer_betas:
        if beta > 0 and R2 > 0.3:
            # 用幂律预测ratio(50)
            i_vals = np.arange(1, 201, dtype=float)
            S_pred = i_vals ** (-beta)
            ratio_pred = np.sum(S_pred[:50]) / np.sum(S_pred)
            predicted_ratios.append(ratio_pred)
            actual_ratios.append(ratios[layer])
    
    if len(predicted_ratios) > 2:
        r_pred, _ = pearsonr(predicted_ratios, actual_ratios)
        mse_pred = np.mean((np.array(predicted_ratios) - np.array(actual_ratios))**2)
        print(f"  幂律预测ratio(50): 相关={r_pred:.4f}, MSE={mse_pred:.6f}")
    
    # === 分析4: 频谱与深度的标度律 ===
    print("\n--- 分析4: 频谱指标与深度/宽度的标度 ---")
    
    # 中层和末层的频谱差异
    mid_layer = n_layers // 2
    spec_mid = project_and_spectrum(h_states[mid_layer][0, -1].detach().cpu().float().numpy(), U_wu, k_wu)
    spec_last = project_and_spectrum(h_states[n_layers - 1][0, -1].detach().cpu().float().numpy(), U_wu, k_wu)
    
    r_mid_last, _ = pearsonr(spec_mid[:50], spec_last[:50])
    print(f"  中层 vs 末层频谱相关: {r_mid_last:.4f}")
    print(f"  中层 ratio(50): {np.sum(spec_mid[:50]):.4f}")
    print(f"  末层 ratio(50): {np.sum(spec_last[:50]):.4f}")
    
    results = {
        'model': model_name, 'experiment': 'p557',
        'n_layers': n_layers, 'd_model': d_model,
        'beta_trend_slope': float(beta_trend[0]) if len(betas) > 2 else 0,
        'ratio_trend_slope': float(ratio_trend[0]) if len(ratios) > 2 else 0,
        'mid_last_corr': float(r_mid_last),
        'mid_ratio50': float(np.sum(spec_mid[:50])),
        'last_ratio50': float(np.sum(spec_last[:50])),
    }
    
    out_dir = 'results/phase_cxxv'
    with open(f'{out_dir}/P557_{model_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {out_dir}/P557_{model_name}.json")
    return results


def experiment_p558(model, tokenizer, device, model_name):
    """P558: 统一理论的闭合验证"""
    print("\n" + "="*70)
    print("P558: 统一理论的闭合验证")
    print("用5个参数预测完整频谱: alpha, W_down_spec, W_o_spec, gamma_spec, beta")
    print("="*70)
    
    U_wu, S_wu, W_U = compute_wu_svd(model, k=200)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    k_wu = 200
    
    text = "The development of artificial intelligence has transformed many aspects of modern life."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model(inputs["input_ids"], output_hidden_states=True)
    h_states = outputs.hidden_states
    
    # === 步骤1: 提取5个参数 ===
    print("\n--- 步骤1: 提取5个参数 ---")
    
    # alpha: 频谱保持系数
    sample_layers = get_sample_layers(n_layers, 10)
    h_specs = {}
    for layer in sample_layers:
        h_l = h_states[layer][0, -1].detach().cpu().float().numpy()
        h_specs[layer] = project_and_spectrum(h_l, U_wu, k_wu)
    
    alphas = []
    for l_idx in range(1, len(sample_layers)):
        spec_prev = h_specs[sample_layers[l_idx - 1]]
        spec_curr = h_specs[sample_layers[l_idx]]
        alpha = np.dot(spec_curr[:50], spec_prev[:50]) / (np.dot(spec_prev[:50], spec_prev[:50]) + 1e-10)
        alphas.append(alpha)
    alpha = np.mean(alphas)
    print(f"  alpha = {alpha:.4f}")
    
    # W_down频谱(平均)
    wdown_specs = []
    for layer in sample_layers:
        try:
            W_down = model.model.layers[layer].mlp.down_proj.weight.detach().cpu().float().numpy()
            wdown_specs.append(weight_spectrum(W_down, U_wu, k_wu))
        except:
            pass
    avg_wd = np.mean(wdown_specs, axis=0) if wdown_specs else np.zeros(k_wu)
    
    # W_o频谱(平均)
    wo_specs = []
    for layer in sample_layers:
        try:
            W_o = model.model.layers[layer].self_attn.o_proj.weight.detach().cpu().float().numpy()
            wo_specs.append(weight_spectrum(W_o, U_wu, k_wu))
        except:
            pass
    avg_wo = np.mean(wo_specs, axis=0) if wo_specs else np.zeros(k_wu)
    
    # gamma频谱(平均)
    gamma_specs = []
    for layer in sample_layers:
        try:
            gamma = model.model.layers[layer].input_layernorm.weight.detach().cpu().float().numpy()
            gamma_proj = gamma @ U_wu[:, :k_wu]
            gs = gamma_proj**2 / (np.sum(gamma_proj**2) + 1e-10)
            gamma_specs.append(gs)
        except:
            pass
    avg_gamma = np.mean(gamma_specs, axis=0) if gamma_specs else np.zeros(k_wu)
    
    # beta: MLP微调幅度
    betas = []
    for l in range(1, min(n_layers, 20)):
        h_prev = h_states[l - 1][0, -1].detach().cpu().float().numpy()
        h_curr = h_states[l][0, -1].detach().cpu().float().numpy()
        betas.append(np.linalg.norm(h_curr - h_prev) / (np.linalg.norm(h_prev) + 1e-10))
    beta = np.mean(betas)
    print(f"  beta = {beta:.4f}")
    
    # === 步骤2: 闭合预测模型 ===
    print("\n--- 步骤2: 闭合预测模型 ---")
    
    # 模型A: 仅alpha保持 (S(l)=alpha*S(l-1))
    # 模型B: alpha + W_down不动点 (S*=c*W_down/(1-alpha))
    # 模型C: alpha + W_down + W_o组合
    # 模型D: 最优线性组合
    
    h_0 = h_states[0][0, -1].detach().cpu().float().numpy()
    spec_0 = project_and_spectrum(h_0, U_wu, k_wu)
    
    # 末层目标
    spec_last = h_specs[n_layers - 1] if n_layers - 1 in h_specs else project_and_spectrum(
        h_states[n_layers - 1][0, -1].detach().cpu().float().numpy(), U_wu, k_wu)
    
    # 模型A: alpha^L * S(0)
    spec_A = (alpha ** n_layers) * spec_0
    spec_A = spec_A / (np.sum(spec_A) + 1e-10)
    r_A, _ = pearsonr(spec_A[:50], spec_last[:50])
    
    # 模型B: 不动点 c*W_down/(1-alpha)
    c_B = np.dot(spec_last[:50], avg_wd[:50]) / (np.dot(avg_wd[:50], avg_wd[:50]) + 1e-10)
    spec_B = c_B * avg_wd
    spec_B = spec_B / (np.sum(spec_B) + 1e-10)
    r_B, _ = pearsonr(spec_B[:50], spec_last[:50])
    
    # 模型C: W_down + W_o组合
    A_mat = np.vstack([avg_wd[:50], avg_wo[:50]]).T
    coeffs_C, _, _, _ = np.linalg.lstsq(A_mat, spec_last[:50], rcond=None)
    spec_C = coeffs_C[0] * avg_wd + coeffs_C[1] * avg_wo
    spec_C = spec_C / (np.sum(spec_C) + 1e-10)
    r_C, _ = pearsonr(spec_C[:50], spec_last[:50])
    
    # 模型D: W_down + W_o + gamma组合
    A_mat2 = np.vstack([avg_wd[:50], avg_wo[:50], avg_gamma[:50]]).T
    coeffs_D, _, _, _ = np.linalg.lstsq(A_mat2, spec_last[:50], rcond=None)
    spec_D = coeffs_D[0] * avg_wd + coeffs_D[1] * avg_wo + coeffs_D[2] * avg_gamma
    spec_D = spec_D / (np.sum(spec_D) + 1e-10)
    r_D, _ = pearsonr(spec_D[:50], spec_last[:50])
    
    print(f"  模型A(alpha^L*S0): {r_A:.4f}")
    print(f"  模型B(c*W_down): {r_B:.4f}")
    print(f"  模型C(W_down+W_o): {r_C:.4f}, coeffs=[{coeffs_C[0]:.3f}, {coeffs_C[1]:.3f}]")
    print(f"  模型D(W_down+W_o+gamma): {r_D:.4f}, coeffs=[{coeffs_D[0]:.3f}, {coeffs_D[1]:.3f}, {coeffs_D[2]:.3f}]")
    
    best_model = max([('A', r_A), ('B', r_B), ('C', r_C), ('D', r_D)], key=lambda x: x[1])
    print(f"  最优模型: {best_model[0]}, 相关={best_model[1]:.4f}")
    
    # === 步骤3: 逐层预测精度 ===
    print("\n--- 步骤3: 逐层预测精度(模型C) ---")
    
    layer_corrs_C = []
    layer_corrs_alpha = []
    
    for layer in sample_layers:
        if layer not in h_specs:
            continue
        spec_actual = h_specs[layer]
        
        # 用该层的W_down和W_o预测
        try:
            W_down_l = model.model.layers[layer].mlp.down_proj.weight.detach().cpu().float().numpy()
            W_o_l = model.model.layers[layer].self_attn.o_proj.weight.detach().cpu().float().numpy()
            spec_wd_l = weight_spectrum(W_down_l, U_wu, k_wu)
            spec_wo_l = weight_spectrum(W_o_l, U_wu, k_wu)
            
            A_l = np.vstack([spec_wd_l[:50], spec_wo_l[:50]]).T
            coeffs_l, _, _, _ = np.linalg.lstsq(A_l, spec_actual[:50], rcond=None)
            spec_pred_l = coeffs_l[0] * spec_wd_l + coeffs_l[1] * spec_wo_l
            spec_pred_l = spec_pred_l / (np.sum(spec_pred_l) + 1e-10)
            r_l, _ = pearsonr(spec_pred_l[:50], spec_actual[:50])
            layer_corrs_C.append(r_l)
        except:
            r_l = 0
        layer_corrs_C.append(r_l)
        
        # alpha保持
        if layer > 0 and (layer - 1) in h_specs:
            spec_prev = h_specs[layer - 1] if layer - 1 in h_specs else spec_0
            spec_pred_a = alpha * spec_prev
            spec_pred_a = spec_pred_a / (np.sum(spec_pred_a) + 1e-10)
            r_a, _ = pearsonr(spec_pred_a[:50], spec_actual[:50])
            layer_corrs_alpha.append(r_a)
    
    avg_corr_C = np.mean(layer_corrs_C) if layer_corrs_C else 0
    avg_corr_alpha = np.mean(layer_corrs_alpha) if layer_corrs_alpha else 0
    print(f"  逐层W_down+W_o预测平均: {avg_corr_C:.4f}")
    print(f"  逐层alpha保持平均: {avg_corr_alpha:.4f}")
    
    # === 步骤4: 跨输入验证 ===
    print("\n--- 步骤4: 跨输入验证(模型C) ---")
    
    texts = [
        "Machine learning algorithms process vast amounts of data efficiently.",
        "The history of civilization spans thousands of years of development.",
    ]
    
    cross_corrs = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = model(inputs["input_ids"], output_hidden_states=True)
        h_test = outputs.hidden_states
        
        test_corrs = []
        for layer in sample_layers[3:6]:  # 中层
            try:
                h_l = h_test[layer][0, -1].detach().cpu().float().numpy()
                spec_actual = project_and_spectrum(h_l, U_wu, k_wu)
                
                W_down_l = model.model.layers[layer].mlp.down_proj.weight.detach().cpu().float().numpy()
                W_o_l = model.model.layers[layer].self_attn.o_proj.weight.detach().cpu().float().numpy()
                spec_wd_l = weight_spectrum(W_down_l, U_wu, k_wu)
                spec_wo_l = weight_spectrum(W_o_l, U_wu, k_wu)
                
                r_wd, _ = pearsonr(spec_wd_l[:50], spec_actual[:50])
                test_corrs.append(r_wd)
            except:
                pass
        
        avg_test = np.mean(test_corrs) if test_corrs else 0
        cross_corrs.append(avg_test)
        print(f"  验证文本: W_down预测相关={avg_test:.4f}")
    
    avg_cross = np.mean(cross_corrs) if cross_corrs else 0
    
    results = {
        'model': model_name, 'experiment': 'p558',
        'alpha': float(alpha), 'beta': float(beta),
        'model_A_corr': float(r_A),
        'model_B_corr': float(r_B),
        'model_C_corr': float(r_C),
        'model_D_corr': float(r_D),
        'model_C_coeffs': [float(coeffs_C[0]), float(coeffs_C[1])],
        'model_D_coeffs': [float(coeffs_D[0]), float(coeffs_D[1]), float(coeffs_D[2])],
        'per_layer_wd_wo_corr': float(avg_corr_C),
        'per_layer_alpha_corr': float(avg_corr_alpha),
        'cross_input_corr': float(avg_cross),
    }
    
    out_dir = 'results/phase_cxxv'
    with open(f'{out_dir}/P558_{model_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {out_dir}/P558_{model_name}.json")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                       choices=['qwen3', 'glm4', 'deepseek7b'])
    parser.add_argument('--experiment', type=str, required=True,
                       choices=['p555', 'p556', 'p557', 'p558'])
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model)
    
    if args.experiment == 'p555':
        experiment_p555(model, tokenizer, device, args.model)
    elif args.experiment == 'p556':
        experiment_p556(model, tokenizer, device, args.model)
    elif args.experiment == 'p557':
        experiment_p557(model, tokenizer, device, args.model)
    elif args.experiment == 'p558':
        experiment_p558(model, tokenizer, device, args.model)
    
    print(f"\n实验 {args.experiment} 完成!")
    
    del model
    import torch
    torch.cuda.empty_cache()
