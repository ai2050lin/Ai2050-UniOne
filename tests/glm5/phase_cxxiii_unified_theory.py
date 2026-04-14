"""
Phase CXXIII-CXXIV: 频谱力学的闭合理论
P551: 残差保持的精确数学模型 — alpha≈1时h(n+1)≈h(n)+epsilon, 推导不动点频谱
P552: 为什么频谱是幂律? — 随机矩阵乘积的奇异值谱理论
P553: MLP微调的频谱修正 — beta项的来源和结构
P554: 统一方程验证: h(L)频谱 = alpha^L * h(0)频谱 + f(MLP, beta, L)
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


def get_hidden_states(model, inputs, n_layers):
    """获取所有层的隐藏状态"""
    outputs = model(inputs["input_ids"], output_hidden_states=True)
    return outputs.hidden_states


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


def experiment_p551(model, tokenizer, device, model_name):
    """P551: 残差保持的精确数学模型"""
    print("\n" + "="*70)
    print("P551: 残差保持的精确数学模型")
    print("="*70)
    
    U_wu, S_wu, W_U = compute_wu_svd(model, k=200)
    k_wu = 200
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    text = "The development of artificial intelligence has transformed many aspects of modern life."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    h_states = get_hidden_states(model, inputs, n_layers)
    
    sample_layers = get_sample_layers(n_layers, 10)
    print(f"采样层: {sample_layers}")
    
    # === 理论1: alpha^L衰减模型 ===
    print("\n--- 理论1: alpha^L衰减模型 ---")
    h_0 = h_states[0][0, -1].detach().cpu().float().numpy()
    spec_0 = project_and_spectrum(h_0, U_wu, k_wu)
    
    alphas = []
    pred_errors_power = []
    
    for l_idx in range(1, len(sample_layers)):
        layer_prev = sample_layers[l_idx - 1]
        layer_curr = sample_layers[l_idx]
        
        h_prev = h_states[layer_prev][0, -1].detach().cpu().float().numpy()
        h_curr = h_states[layer_curr][0, -1].detach().cpu().float().numpy()
        spec_prev = project_and_spectrum(h_prev, U_wu, k_wu)
        spec_curr = project_and_spectrum(h_curr, U_wu, k_wu)
        
        alpha = np.dot(spec_curr[:50], spec_prev[:50]) / (np.dot(spec_prev[:50], spec_prev[:50]) + 1e-10)
        alphas.append(alpha)
        
        n_steps = layer_curr - layer_prev
        spec_pred = (alpha ** n_steps) * spec_prev
        err = np.mean((spec_curr[:50] - spec_pred[:50])**2)
        pred_errors_power.append(err)
    
    avg_alpha = np.mean(alphas) if alphas else 0
    avg_err_power = np.mean(pred_errors_power) if pred_errors_power else 0
    print(f"  平均alpha(频谱保持系数): {avg_alpha:.4f}")
    print(f"  alpha^L衰减模型误差: {avg_err_power:.6f}")
    
    # === 理论2: 不动点频谱 ===
    print("\n--- 理论2: 不动点频谱推导 ---")
    epsilon_specs = []
    for l_idx in range(1, len(sample_layers)):
        layer_prev = sample_layers[l_idx - 1]
        layer_curr = sample_layers[l_idx]
        h_prev = h_states[layer_prev][0, -1].detach().cpu().float().numpy()
        h_curr = h_states[layer_curr][0, -1].detach().cpu().float().numpy()
        eps = h_curr - h_prev
        spec_eps = project_and_spectrum(eps, U_wu, k_wu)
        epsilon_specs.append(spec_eps)
    
    avg_epsilon_spec = np.mean(epsilon_specs, axis=0)
    
    if avg_alpha < 0.999:
        fixed_point_spec = avg_epsilon_spec / (1 - avg_alpha + 1e-10)
    else:
        fixed_point_spec = avg_epsilon_spec * 100
    
    fixed_point_spec = fixed_point_spec / (np.sum(fixed_point_spec) + 1e-10)
    
    h_last = h_states[n_layers - 1][0, -1].detach().cpu().float().numpy()
    spec_last = project_and_spectrum(h_last, U_wu, k_wu)
    
    corr_fixed, _ = pearsonr(fixed_point_spec[:50], spec_last[:50])
    cos_fixed = np.dot(fixed_point_spec[:50], spec_last[:50]) / (
        np.linalg.norm(fixed_point_spec[:50]) * np.linalg.norm(spec_last[:50]) + 1e-10)
    
    print(f"  alpha: {avg_alpha:.4f}")
    print(f"  不动点频谱 vs 末层频谱: 相关={corr_fixed:.4f}, 余弦={cos_fixed:.4f}")
    
    if corr_fixed > 0.5:
        print("  >> 不动点频谱与末层频谱正相关: 支持不动点理论")
    else:
        print("  >> 不动点频谱与末层频谱相关弱: 不动点模型不充分")
    
    # === 理论3: epsilon频谱的层间稳定性 ===
    print("\n--- 理论3: epsilon频谱的层间稳定性 ---")
    eps_corrs = []
    for i in range(len(epsilon_specs)):
        for j in range(i + 1, len(epsilon_specs)):
            r, _ = pearsonr(epsilon_specs[i][:50], epsilon_specs[j][:50])
            eps_corrs.append(r)
    avg_eps_corr = np.mean(eps_corrs) if eps_corrs else 0
    print(f"  epsilon频谱层间平均相关: {avg_eps_corr:.4f}")
    
    if avg_eps_corr > 0.5:
        print("  >> epsilon频谱高度稳定: 不同层的微调方向一致")
    else:
        print("  >> epsilon频谱不稳定: 不同层的微调方向不同")
    
    results = {
        'model': model_name, 'experiment': 'p551',
        'avg_alpha': float(avg_alpha),
        'err_power_model': float(avg_err_power),
        'fixed_point_corr': float(corr_fixed),
        'fixed_point_cos': float(cos_fixed),
        'epsilon_spec_corr': float(avg_eps_corr),
    }
    
    out_dir = 'results/phase_cxxiii'
    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/P551_{model_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {out_dir}/P551_{model_name}.json")
    return results


def experiment_p552(model, tokenizer, device, model_name):
    """P552: 为什么频谱是幂律?"""
    print("\n" + "="*70)
    print("P552: 为什么频谱是幂律? — 随机矩阵乘积理论")
    print("="*70)
    
    U_wu, S_wu, W_U = compute_wu_svd(model, k=200)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    # === 实验1: W_down矩阵乘积的频谱 ===
    print("\n--- 实验1: W_down矩阵乘积的频谱 ---")
    
    sample_layers = get_sample_layers(n_layers, 5)
    
    W_downs = []
    for layer in sample_layers:
        try:
            W_down = model.model.layers[layer].mlp.down_proj.weight.detach().cpu().float().numpy()
            W_downs.append(W_down)
        except:
            pass
    
    beta_prod = 0
    R2_prod = 0
    
    if len(W_downs) >= 2:
        product = W_downs[0]
        for W in W_downs[1:]:
            try:
                product = product @ W[:product.shape[1], :]
            except:
                break
        
        if min(product.shape) <= 200:
            _, S_prod, _ = svd(product, full_matrices=False)
        else:
            _, S_prod, _ = svds(product, k=min(200, min(product.shape) - 1))
            S_prod = np.sort(S_prod)[::-1]
        
        valid = S_prod > 0
        if np.sum(valid) > 5:
            log_i = np.log(np.arange(1, np.sum(valid) + 1))
            log_S = np.log(S_prod[valid])
            coeffs = np.polyfit(log_i, log_S, 1)
            beta_prod = -coeffs[0]
            SS_res = np.sum((log_S - np.polyval(coeffs, log_i))**2)
            SS_tot = np.sum((log_S - np.mean(log_S))**2)
            R2_prod = 1 - SS_res / (SS_tot + 1e-10)
            print(f"  W_down乘积的频谱幂律: beta={beta_prod:.3f}, R2={R2_prod:.4f}")
    
    # === 实验2: W_down频谱与h频谱的关系 ===
    print("\n--- 实验2: W_down频谱与h频谱的关系 ---")
    
    text = "The development of artificial intelligence has transformed many aspects of modern life."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    h_states = get_hidden_states(model, inputs, n_layers)
    
    wdown_h_corrs = []
    for layer in sample_layers[:5]:
        h_l = h_states[layer][0, -1].detach().cpu().float().numpy()
        spec_h = project_and_spectrum(h_l, U_wu, 200)
        
        try:
            W_down = model.model.layers[layer].mlp.down_proj.weight.detach().cpu().float().numpy()
            W_down_proj = U_wu[:, :200].T @ W_down
            spec_wd = np.mean(W_down_proj**2, axis=1)
            spec_wd = spec_wd / (np.sum(spec_wd) + 1e-10)
            
            r, _ = pearsonr(spec_wd[:50], spec_h[:50])
            wdown_h_corrs.append(r)
            print(f"  L{layer}: W_down频谱 vs h频谱相关 = {r:.4f}")
        except:
            pass
    
    avg_wd_corr = np.mean(wdown_h_corrs) if wdown_h_corrs else 0
    print(f"  平均相关: {avg_wd_corr:.4f}")
    
    # === 实验3: 随机矩阵乘积频谱(对照) ===
    print("\n--- 实验3: 随机矩阵乘积频谱(对照) ---")
    
    d_int = 4096
    n_random = 5
    random_product = np.random.randn(d_model, d_int) / np.sqrt(d_model)
    for _ in range(n_random - 1):
        random_product = random_product @ (np.random.randn(d_int, d_int) / np.sqrt(d_int))
    
    beta_random = 0
    R2_random = 0
    
    if min(random_product.shape) <= 200:
        _, S_random, _ = svd(random_product, full_matrices=False)
    else:
        _, S_random, _ = svds(random_product, k=min(200, min(random_product.shape) - 1))
        S_random = np.sort(S_random)[::-1]
    
    valid = S_random > 0
    if np.sum(valid) > 5:
        log_i = np.log(np.arange(1, np.sum(valid) + 1))
        log_S = np.log(S_random[valid])
        coeffs = np.polyfit(log_i, log_S, 1)
        beta_random = -coeffs[0]
        SS_res = np.sum((log_S - np.polyval(coeffs, log_i))**2)
        SS_tot = np.sum((log_S - np.mean(log_S))**2)
        R2_random = 1 - SS_res / (SS_tot + 1e-10)
        print(f"  随机矩阵乘积的频谱幂律: beta={beta_random:.3f}, R2={R2_random:.4f}")
    
    # === 实验4: 自由概率论预测 ===
    print("\n--- 实验4: 自由概率论(Free Probability)预测 ---")
    
    if len(W_downs) > 0:
        d_out = W_downs[0].shape[0]
        d_in = W_downs[0].shape[1]
        d_ratio = d_out / d_in
        n_mat = len(W_downs)
        beta_predicted = np.sqrt(2 * n_mat * abs(np.log(d_ratio + 1e-10)))
        print(f"  d_out={d_out}, d_in={d_in}, d_ratio={d_ratio:.3f}")
        print(f"  n_matrices={n_mat}")
        print(f"  理论预测beta: {beta_predicted:.3f}")
        print(f"  实测beta: {beta_prod:.3f}")
    
    results = {
        'model': model_name, 'experiment': 'p552',
        'product_spectrum_beta': float(beta_prod),
        'product_spectrum_R2': float(R2_prod),
        'wdown_h_corr': float(avg_wd_corr),
        'random_product_beta': float(beta_random),
        'random_product_R2': float(R2_random),
    }
    
    out_dir = 'results/phase_cxxiii'
    with open(f'{out_dir}/P552_{model_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {out_dir}/P552_{model_name}.json")
    return results


def experiment_p553(model, tokenizer, device, model_name):
    """P553: MLP微调的频谱修正"""
    print("\n" + "="*70)
    print("P553: MLP微调的频谱修正")
    print("="*70)
    
    U_wu, S_wu, W_U = compute_wu_svd(model, k=200)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    text = "The development of artificial intelligence has transformed many aspects of modern life."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    h_states = get_hidden_states(model, inputs, n_layers)
    
    sample_layers = get_sample_layers(n_layers, 8)
    
    # === 分析1: delta频谱的结构 ===
    print("\n--- 分析1: delta频谱的结构 ---")
    
    delta_specs = []
    for l_idx in range(1, len(sample_layers)):
        layer_prev = sample_layers[l_idx - 1]
        layer_curr = sample_layers[l_idx]
        h_prev = h_states[layer_prev][0, -1].detach().cpu().float().numpy()
        h_curr = h_states[layer_curr][0, -1].detach().cpu().float().numpy()
        delta = h_curr - h_prev
        spec_delta = project_and_spectrum(delta, U_wu, 200)
        delta_specs.append(spec_delta)
    
    delta_corrs = []
    for i in range(len(delta_specs)):
        for j in range(i + 1, len(delta_specs)):
            r, _ = pearsonr(delta_specs[i][:50], delta_specs[j][:50])
            delta_corrs.append(r)
    avg_delta_corr = np.mean(delta_corrs) if delta_corrs else 0
    print(f"  delta频谱层间平均相关: {avg_delta_corr:.4f}")
    
    # delta频谱与h频谱的对齐
    h_delta_corrs = []
    for idx in range(len(delta_specs)):
        layer = sample_layers[idx + 1] if idx + 1 < len(sample_layers) else sample_layers[idx]
        h_l = h_states[layer][0, -1].detach().cpu().float().numpy()
        spec_h = project_and_spectrum(h_l, U_wu, 200)
        r, _ = pearsonr(delta_specs[idx][:50], spec_h[:50])
        h_delta_corrs.append(r)
    avg_h_delta = np.mean(h_delta_corrs) if h_delta_corrs else 0
    print(f"  delta频谱 vs h频谱平均相关: {avg_h_delta:.4f}")
    
    # === 分析2: MLP输出的频谱贡献 ===
    print("\n--- 分析2: MLP/Attn频谱贡献分解 ---")
    
    mlp_contributions = []
    attn_contributions = []
    
    for layer in sample_layers[:5]:
        try:
            W_down = model.model.layers[layer].mlp.down_proj.weight.detach().cpu().float().numpy()
            W_down_proj = U_wu[:, :200].T @ W_down
            mlp_spec = np.mean(W_down_proj**2, axis=1)
            mlp_spec = mlp_spec / (np.sum(mlp_spec) + 1e-10)
            
            W_o = model.model.layers[layer].self_attn.o_proj.weight.detach().cpu().float().numpy()
            W_o_proj = U_wu[:, :200].T @ W_o
            attn_spec = np.mean(W_o_proj**2, axis=1)
            attn_spec = attn_spec / (np.sum(attn_spec) + 1e-10)
            
            # 与delta频谱的相关
            delta_idx = None
            for di, sl in enumerate(sample_layers):
                if sl == layer and di > 0:
                    delta_idx = di - 1
                    break
            
            if delta_idx is not None and delta_idx < len(delta_specs):
                r_mlp, _ = pearsonr(mlp_spec[:50], delta_specs[delta_idx][:50])
                r_attn, _ = pearsonr(attn_spec[:50], delta_specs[delta_idx][:50])
                mlp_contributions.append(r_mlp)
                attn_contributions.append(r_attn)
                print(f"  L{layer}: MLP频谱 vs delta={r_mlp:.3f}, Attn频谱 vs delta={r_attn:.3f}")
        except Exception as e:
            pass
    
    avg_mlp = np.mean(mlp_contributions) if mlp_contributions else 0
    avg_attn = np.mean(attn_contributions) if attn_contributions else 0
    print(f"  平均MLP-delta相关: {avg_mlp:.4f}")
    print(f"  平均Attn-delta相关: {avg_attn:.4f}")
    
    # === 分析3: beta项的量化 ===
    print("\n--- 分析3: beta项(MLP微调幅度)量化 ---")
    
    betas = []
    for l in range(1, min(n_layers, 20)):
        h_prev = h_states[l - 1][0, -1].detach().cpu().float().numpy()
        h_curr = h_states[l][0, -1].detach().cpu().float().numpy()
        delta = h_curr - h_prev
        beta_val = np.linalg.norm(delta) / (np.linalg.norm(h_prev) + 1e-10)
        betas.append(beta_val)
    
    for l_idx, beta_val in enumerate(betas[:8]):
        print(f"  L{l_idx+1}: beta = {beta_val:.4f}")
    
    avg_beta = np.mean(betas) if betas else 0
    print(f"  平均beta: {avg_beta:.4f}")
    
    if avg_beta < 0.1:
        print("  >> beta极小: MLP微调是残差上的小扰动, 支持alpha≈1假设")
    else:
        print("  >> beta不可忽略: MLP微调幅度大")
    
    results = {
        'model': model_name, 'experiment': 'p553',
        'delta_spec_corr': float(avg_delta_corr),
        'h_delta_corr': float(avg_h_delta),
        'mlp_delta_corr': float(avg_mlp),
        'attn_delta_corr': float(avg_attn),
        'avg_beta': float(avg_beta),
    }
    
    out_dir = 'results/phase_cxxiii'
    with open(f'{out_dir}/P553_{model_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {out_dir}/P553_{model_name}.json")
    return results


def experiment_p554(model, tokenizer, device, model_name):
    """P554: 统一方程验证"""
    print("\n" + "="*70)
    print("P554: 统一方程验证")
    print("h(L)频谱 ≈ alpha^L * h(0)频谱 + f(MLP, beta, L)")
    print("="*70)
    
    U_wu, S_wu, W_U = compute_wu_svd(model, k=200)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    text = "The development of artificial intelligence has transformed many aspects of modern life."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    h_states = get_hidden_states(model, inputs, n_layers)
    
    # === 步骤1: 计算alpha和beta ===
    print("\n--- 步骤1: 估计alpha和beta ---")
    
    h_0 = h_states[0][0, -1].detach().cpu().float().numpy()
    spec_0 = project_and_spectrum(h_0, U_wu, 200)
    
    alphas = []
    betas = []
    epsilon_specs = []
    
    for l in range(1, min(n_layers, 20)):
        h_prev = h_states[l - 1][0, -1].detach().cpu().float().numpy()
        h_curr = h_states[l][0, -1].detach().cpu().float().numpy()
        spec_prev = project_and_spectrum(h_prev, U_wu, 200)
        spec_curr = project_and_spectrum(h_curr, U_wu, 200)
        
        alpha = np.dot(spec_curr[:50], spec_prev[:50]) / (
            np.linalg.norm(spec_curr[:50]) * np.linalg.norm(spec_prev[:50]) + 1e-10)
        alphas.append(alpha)
        
        delta = h_curr - h_prev
        beta = np.linalg.norm(delta) / (np.linalg.norm(h_prev) + 1e-10)
        betas.append(beta)
        
        spec_delta = project_and_spectrum(delta, U_wu, 200)
        epsilon_specs.append(spec_delta)
    
    alpha = np.mean(alphas)
    beta = np.mean(betas)
    print(f"  alpha(频谱保持): {alpha:.4f}")
    print(f"  beta(微调幅度): {beta:.4f}")
    
    # === 步骤2: 逐步预测精度(方程3: S(l)=alpha*S(l-1)+epsilon) ===
    print("\n--- 步骤2: 逐步预测精度 ---")
    
    step_corrs = []
    alpha_only_corrs = []
    
    for l in range(1, min(n_layers, 30)):
        h_prev = h_states[l - 1][0, -1].detach().cpu().float().numpy()
        h_curr = h_states[l][0, -1].detach().cpu().float().numpy()
        spec_prev = project_and_spectrum(h_prev, U_wu, 200)
        spec_curr = project_and_spectrum(h_curr, U_wu, 200)
        
        # 方程3: alpha + epsilon
        eps = epsilon_specs[l - 1] if l - 1 < len(epsilon_specs) else np.zeros(200)
        spec_pred_full = alpha * spec_prev + eps
        spec_pred_full = spec_pred_full / (np.sum(spec_pred_full) + 1e-10)
        r_full, _ = pearsonr(spec_pred_full[:50], spec_curr[:50])
        step_corrs.append(r_full)
        
        # 仅alpha
        spec_pred_alpha = alpha * spec_prev
        spec_pred_alpha = spec_pred_alpha / (np.sum(spec_pred_alpha) + 1e-10)
        r_alpha, _ = pearsonr(spec_pred_alpha[:50], spec_curr[:50])
        alpha_only_corrs.append(r_alpha)
    
    avg_step_corr = np.mean(step_corrs) if step_corrs else 0
    avg_alpha_corr = np.mean(alpha_only_corrs) if alpha_only_corrs else 0
    epsilon_contrib = avg_step_corr - avg_alpha_corr
    
    print(f"  alpha+epsilon逐步预测相关: {avg_step_corr:.4f}")
    print(f"  仅alpha保持预测相关: {avg_alpha_corr:.4f}")
    print(f"  epsilon额外贡献: {epsilon_contrib:.4f}")
    
    # === 步骤3: 长程预测(方程1: alpha^L * S(0)) ===
    print("\n--- 步骤3: 长程预测(alpha^L衰减) ---")
    
    sample_layers = get_sample_layers(n_layers, 8)
    long_range_errors = []
    
    for layer in sample_layers:
        if layer == 0:
            continue
        h_l = h_states[layer][0, -1].detach().cpu().float().numpy()
        spec_actual = project_and_spectrum(h_l, U_wu, 200)
        
        spec_pred_long = (alpha ** layer) * spec_0
        spec_pred_long = spec_pred_long / (np.sum(spec_pred_long) + 1e-10)
        err = np.mean((spec_actual[:50] - spec_pred_long[:50])**2)
        long_range_errors.append(err)
    
    avg_long_err = np.mean(long_range_errors) if long_range_errors else 0
    print(f"  alpha^L长程预测MSE: {avg_long_err:.6f}")
    
    # === 步骤4: 跨输入验证 ===
    print("\n--- 步骤4: 跨输入验证统一方程 ---")
    
    texts = [
        "Machine learning algorithms process vast amounts of data efficiently.",
        "The history of civilization spans thousands of years of development.",
        "Quantum physics describes the behavior of subatomic particles.",
    ]
    
    cross_input_corrs = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        h_test = get_hidden_states(model, inputs, n_layers)
        
        test_corrs = []
        for l in range(1, min(n_layers, 15)):
            h_prev = h_test[l - 1][0, -1].detach().cpu().float().numpy()
            h_curr = h_test[l][0, -1].detach().cpu().float().numpy()
            spec_prev = project_and_spectrum(h_prev, U_wu, 200)
            spec_curr = project_and_spectrum(h_curr, U_wu, 200)
            
            spec_pred = alpha * spec_prev
            spec_pred = spec_pred / (np.sum(spec_pred) + 1e-10)
            
            r, _ = pearsonr(spec_pred[:50], spec_curr[:50])
            test_corrs.append(r)
        
        avg_test = np.mean(test_corrs) if test_corrs else 0
        cross_input_corrs.append(avg_test)
        print(f"  文本验证平均相关: {avg_test:.4f}")
    
    avg_cross = np.mean(cross_input_corrs) if cross_input_corrs else 0
    print(f"  跨输入平均验证相关: {avg_cross:.4f}")
    
    if avg_cross > 0.9:
        print("  >> 统一方程跨输入有效: alpha是模型内在属性")
    elif avg_cross > 0.5:
        print("  >> 统一方程跨输入部分有效")
    else:
        print("  >> 统一方程跨输入无效")
    
    results = {
        'model': model_name, 'experiment': 'p554',
        'alpha': float(alpha), 'beta': float(beta),
        'step_pred_corr': float(avg_step_corr),
        'alpha_only_corr': float(avg_alpha_corr),
        'epsilon_contribution': float(epsilon_contrib),
        'long_range_mse': float(avg_long_err),
        'cross_input_corr': float(avg_cross),
    }
    
    out_dir = 'results/phase_cxxiii'
    with open(f'{out_dir}/P554_{model_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {out_dir}/P554_{model_name}.json")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                       choices=['qwen3', 'glm4', 'deepseek7b'])
    parser.add_argument('--experiment', type=str, required=True,
                       choices=['p551', 'p552', 'p553', 'p554'])
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model)
    
    if args.experiment == 'p551':
        experiment_p551(model, tokenizer, device, args.model)
    elif args.experiment == 'p552':
        experiment_p552(model, tokenizer, device, args.model)
    elif args.experiment == 'p553':
        experiment_p553(model, tokenizer, device, args.model)
    elif args.experiment == 'p554':
        experiment_p554(model, tokenizer, device, args.model)
    
    print(f"\n实验 {args.experiment} 完成!")
    
    del model
    import torch
    torch.cuda.empty_cache()
