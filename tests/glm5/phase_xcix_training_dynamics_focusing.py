"""
Phase XCIX-P476/477/478/479: 训练动态与信号聚焦
======================================================================

核心目标: 理解什么决定了c/delta的跨模型差异

Phase XCVIII核心成果:
1. F_W跨模型几乎相同(0.175~0.216), 不是c/delta变化的原因
2. d/n_L>100为"全聚焦相", d/n_L<80为"混合相"
3. GLM4支持F_W->alpha因果, 但Qwen3/DS7B不支持
4. 信号聚焦的真正原因可能与训练动态/初始化有关

关键问题:
1. intermediate_size/d_model与c/delta有什么关系?
2. 权重初始化残留(init residual)是否影响alpha?
3. 残差流信号强度(信号衰减/增长模式)的层间演化规律?
4. 训练动态的"痕迹"(如权重范数变化)能否预测alpha?

Phase XCIX目标:
1. 分析intermediate_size/d_model与c/delta的关系
2. 测量权重初始化残留: sigma_init = ||W - W_init|| / ||W||
3. 分析残差流信号强度演化: ||h_L|| / ||h_0||
4. 建立alpha的多变量预测模型

P476: 架构参数与信号聚焦
  - 目标: 分析intermediate_size/d_model, n_heads, head_dim等架构参数与alpha的关系
  - 方法:
    a) 收集三模型的架构参数: d_model, n_layers, intermediate_size, n_heads, head_dim
    b) 分析这些参数与(alpha, gamma, c/delta)的关联
    c) 特别关注: intermediate_size/d_model (FFN扩展比)
    d) 检验FFN扩展比是否与聚焦比相关
  - 关键指标:
    - FFN_ratio = intermediate_size / d_model
    - param_per_layer = d_model^2 * (4 + 2*FFN_ratio) / n_layers
    - alpha_focus = mean(|alpha|) over layers

P477: 权重初始化残留分析
  - 目标: 测量训练后权重偏离初始化的程度
  - 方法:
    a) 用相同架构的初始化方案生成"随机权重"
    b) 比较训练后权重与随机权重的SVD谱差异
    c) 测量"训练残留": ||W_trained - W_init_predicted|| / ||W_trained||
    d) 分析训练残留与alpha的关系
  - 关键思想:
    - 初始化权重服从Marchenko-Pastur分布(随机矩阵)
    - 训练后权重偏离MP分布的程度 = 训练动态的"痕迹"
    - 训练残留越多 -> 结构越多 -> 可能影响alpha

P478: 残差流信号强度演化
  - 目标: 分析残差流中信号的实际衰减/增长模式
  - 方法:
    a) 前向传播输入文本, 提取每层残差流
    b) 测量||h_L|| / ||h_0||随L的变化
    c) 分析信号增长与alpha的关系
    d) 检验: 信号增长快的层是否有更大的alpha?
  - 关键指标:
    - signal_ratio(L) = ||h_L|| / ||h_0||
    - signal_growth(L) = ||h_{L+1} - h_L|| / ||h_L||
    - alpha vs signal_growth 的相关性

P479: 多变量alpha预测模型
  - 目标: 建立alpha的多变量预测模型
  - 方法:
    a) 收集所有特征: delta_W, F_W, PR_W, FFN_ratio, layer_frac, sigma_W, kappa_W
    b) 逐步回归: 从最重要的特征开始
    c) 交叉验证: 避免过拟合
    d) 建立最终预测公式
  - 候选模型:
    - alpha = a * FFN_ratio + b * delta_W + c * F_W + d
    - alpha = a * layer_frac^2 + b * layer_frac + c * FFN_ratio + d
    - alpha = a * signal_growth + b * sigma_W + c
"""

import sys
import os
import argparse
import numpy as np
import torch
from scipy import stats
from scipy.linalg import svd

# 添加项目根目录到sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_utils import (
    load_model, get_layers, get_layer_weights, get_model_info,
    get_W_U, release_model, MODEL_CONFIGS
)


# ===== 通用函数 =====

def compute_spectral_density(delta_h, U_wut, s_wut, k_max):
    """
    计算delta_h在W_U^T SVD基上的频谱密度
    
    Args:
        delta_h: 信号向量 [d_model]
        U_wut: W_U^T的左奇异向量 [k_max, d_model]
        s_wut: W_U^T的奇异值 [k_max]
        k_max: 截断维度
    
    Returns:
        (e_i, alpha, alpha_fit, r_squared)
    """
    # 投影
    proj = U_wut[:k_max] @ delta_h  # [k_max]
    # 能量密度
    e_i = proj**2  # [k_max]
    
    # 拟合alpha: e_i ~ s_i^(2*alpha)
    log_s = np.log(s_wut[:k_max] + 1e-30)
    log_e = np.log(e_i + 1e-30)
    
    # 只用有效点(e_i > 0)
    valid = e_i > 1e-20
    if valid.sum() < 5:
        return e_i, 0.0, None, 0.0
    
    log_s_v = log_s[valid]
    log_e_v = log_e[valid]
    
    # 线性回归
    slope, intercept, r_val, p_val, std_err = stats.linregress(log_s_v, log_e_v)
    alpha = slope / 2.0  # e_i ~ s^(2*alpha) => slope = 2*alpha
    
    # R^2
    ss_res = np.sum((log_e_v - (slope * log_s_v + intercept))**2)
    ss_tot = np.sum((log_e_v - np.mean(log_e_v))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    return e_i, alpha, (slope, intercept), r_squared


def compute_alpha_for_layer(layer_idx, model, tokenizer, device, U_wut, s_wut, k_max):
    """计算某层的alpha值(基于属性干预)"""
    # 基准文本
    base_text = "The apple is"
    base_ids = tokenizer.encode(base_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        base_out = model(base_ids, output_hidden_states=True)
        base_h = base_out.hidden_states[layer_idx + 1][0, -1].cpu().float().numpy()
    
    # 属性词干预
    attribute_words = ["red", "green", "big", "small", "sweet", "sour"]
    alpha_list = []
    for attr_word in attribute_words:
        intervened_text = f"The {attr_word} apple is"
        interv_ids = tokenizer.encode(intervened_text, return_tensors="pt").to(device)
        with torch.no_grad():
            interv_out = model(interv_ids, output_hidden_states=True)
            interv_h = interv_out.hidden_states[layer_idx + 1][0, -1].cpu().float().numpy()
        delta_h = interv_h - base_h
        _, alpha, _, r2 = compute_spectral_density(delta_h, U_wut, s_wut, k_max)
        alpha_list.append(alpha)  # 收集所有alpha, 不设R2阈值
    
    if len(alpha_list) == 0:
        return 0.0, 0.0
    return np.mean(alpha_list), np.std(alpha_list)


def compute_wdown_svd_features(W_down, k_max=300):
    """计算W_down的SVD频谱特征"""
    U, s, Vt = svd(W_down, full_matrices=False)
    s = s[:k_max]
    
    # 衰减率delta_W
    log_s = np.log(s + 1e-30)
    log_i = np.log(np.arange(1, len(s) + 1, dtype=float))
    valid = s > 1e-10
    if valid.sum() < 10:
        delta_W = 1.0
    else:
        slope, _, _, _, _ = stats.linregress(log_i[valid], log_s[valid])
        delta_W = -slope  # s ~ i^(-delta_W)
    
    # 频谱丰满度F_W
    F_W = np.sum(s**2) / (len(s) * s[0]**2)
    
    # 参与率PR
    pr = (np.sum(s**2))**2 / np.sum(s**4)
    
    # 条件数
    kappa = s[0] / (s[-1] + 1e-30)
    
    # sigma_W = ||W||_F / sqrt(m*n)
    sigma_W = np.linalg.norm(W_down, 'fro') / np.sqrt(W_down.shape[0] * W_down.shape[1])
    
    return {
        'delta_W': delta_W,
        'F_W': F_W,
        'PR': pr,
        'kappa': kappa,
        'sigma_W': sigma_W,
        's_max': s[0],
        's_min': s[-1],
    }


# ===== P476: 架构参数与信号聚焦 =====

def run_p476(model_name):
    """架构参数与信号聚焦"""
    print(f"\n{'='*70}")
    print(f"P476: Architecture Parameters & Signal Focusing - {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    
    # 架构参数
    d_model = info.d_model
    n_layers = info.n_layers
    intermediate_size = info.intermediate_size
    FFN_ratio = intermediate_size / d_model if d_model > 0 else 0
    d_over_n = d_model * 1.0 / n_layers  # d/n_L (参数密度)
    
    # 从模型配置获取n_heads
    config = model.config
    n_heads = getattr(config, 'num_attention_heads', getattr(config, 'n_heads', 0))
    head_dim = d_model // n_heads if n_heads > 0 else 0
    
    # 每层参数量估计
    # Attn: 4 * d_model^2 (QKV+O)
    # FFN: d_model * intermediate * 2 (up+down) + d_model * intermediate (gate)
    attn_params = 4 * d_model**2
    ffn_params = d_model * intermediate_size * (3 if info.mlp_type == 'split_gate_up' else 2)
    params_per_layer = attn_params + ffn_params
    
    print(f"\n  Architecture Parameters:")
    print(f"    d_model = {d_model}")
    print(f"    n_layers = {n_layers}")
    print(f"    intermediate_size = {intermediate_size}")
    print(f"    FFN_ratio = {FFN_ratio:.3f}")
    print(f"    d/n_L = {d_over_n:.1f}")
    print(f"    n_heads = {n_heads}")
    print(f"    head_dim = {head_dim}")
    print(f"    params_per_layer ~ {params_per_layer/1e6:.1f}M")
    print(f"    MLP_type = {info.mlp_type}")
    
    # W_U^T SVD
    W_U = get_W_U(model)
    W_UT = W_U.T  # [d_model, vocab_size]
    k_wut = min(400, min(W_UT.shape) - 1)
    U_wut, s_wut, _ = svd(W_UT, full_matrices=False)
    U_wut = U_wut[:, :k_wut].T  # [k_wut, d_model]
    s_wut = s_wut[:k_wut]
    
    # delta (W_U^T的衰减率)
    log_s_wut = np.log(s_wut[:200] + 1e-30)
    log_i = np.log(np.arange(1, 201, dtype=float))
    slope_wut, _, _, _, _ = stats.linregress(log_i, log_s_wut)
    delta = -slope_wut
    
    # 对所有层计算alpha和W_down特征
    sample_layers = list(range(0, n_layers, max(1, n_layers // 15))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    results = []
    for layer_idx in sample_layers:
        layers = get_layers(model)
        lw = get_layer_weights(layers[layer_idx], d_model, info.mlp_type)
        W_down = lw.W_down
        
        if W_down is None:
            continue
        
        # Alpha
        alpha_mean, alpha_std = compute_alpha_for_layer(
            layer_idx, model, tokenizer, device, U_wut, s_wut, k_wut
        )
        
        # W_down 特征
        wdown_feat = compute_wdown_svd_features(W_down)
        
        results.append({
            'layer': layer_idx,
            'layer_frac': layer_idx / max(1, n_layers - 1),
            'alpha': alpha_mean,
            'alpha_std': alpha_std,
            **wdown_feat,
        })
    
    release_model(model)
    
    # 分析架构参数与alpha的关系
    alphas = [r['alpha'] for r in results]
    layer_fracs = [r['layer_frac'] for r in results]
    delta_Ws = [r['delta_W'] for r in results]
    F_Ws = [r['F_W'] for r in results]
    sigma_Ws = [r['sigma_W'] for r in results]
    
    mean_alpha = np.mean(alphas)
    focus_ratio = sum(1 for a in alphas if a > 0) / len(alphas)
    
    # Alpha vs layer_frac 的二次拟合
    lf = np.array(layer_fracs)
    al = np.array(alphas)
    if len(lf) > 3:
        coeffs = np.polyfit(lf, al, 2)
        alpha_pred = np.polyval(coeffs, lf)
        ss_res = np.sum((al - alpha_pred)**2)
        ss_tot = np.sum((al - np.mean(al))**2)
        r2_poly = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    else:
        r2_poly = 0
        coeffs = [0, 0, np.mean(alphas)]
    
    print(f"\n  Alpha Statistics:")
    print(f"    mean_alpha = {mean_alpha:.3f}")
    print(f"    alpha_range = [{min(alphas):.3f}, {max(alphas):.3f}]")
    print(f"    focusing_ratio = {focus_ratio:.2f}")
    print(f"    alpha vs layer_frac R2 (quadratic) = {r2_poly:.3f}")
    
    print(f"\n  Architecture-Alpha Relationship:")
    print(f"    FFN_ratio = {FFN_ratio:.3f}")
    print(f"    d/n_L = {d_over_n:.1f}")
    print(f"    delta (W_U) = {delta:.3f}")
    
    # 架构参数总结
    print(f"\n  Key Architecture Metrics:")
    print(f"    FFN_ratio (intermediate/d_model) = {FFN_ratio:.3f}")
    print(f"    param_density (d/n_L) = {d_over_n:.1f}")
    print(f"    heads/layer = {n_heads}")
    print(f"    head_dim = {head_dim}")
    
    # 各层的delta_W, F_W, sigma_W统计
    print(f"\n  W_down Spectral Features (layer averages):")
    print(f"    mean_delta_W = {np.mean(delta_Ws):.3f}")
    print(f"    mean_F_W = {np.mean(F_Ws):.3f}")
    print(f"    mean_sigma_W = {np.mean(sigma_Ws):.3f}")
    
    # 每层详细数据
    print(f"\n  Per-layer Data:")
    print(f"    {'Layer':>5} {'Lfrac':>6} {'alpha':>7} {'dW':>6} {'FW':>6} {'sW':>6}")
    for r in results:
        print(f"    {r['layer']:5d} {r['layer_frac']:6.2f} {r['alpha']:7.3f} "
              f"{r['delta_W']:6.3f} {r['F_W']:6.3f} {r['sigma_W']:6.3f}")
    
    return {
        'model': model_name,
        'd_model': d_model,
        'n_layers': n_layers,
        'intermediate_size': intermediate_size,
        'FFN_ratio': FFN_ratio,
        'd_over_n': d_over_n,
        'n_heads': n_heads,
        'head_dim': head_dim,
        'delta': delta,
        'mean_alpha': mean_alpha,
        'focus_ratio': focus_ratio,
        'r2_poly': r2_poly,
        'poly_coeffs': coeffs.tolist() if len(coeffs) > 0 else [0, 0, 0],
        'layer_data': results,
    }


# ===== P477: 权重初始化残留分析 =====

def run_p477(model_name):
    """权重初始化残留分析"""
    print(f"\n{'='*70}")
    print(f"P477: Weight Initialization Residual - {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    
    print(f"\n  核心思想: 训练后的权重W = W_init + Delta_W")
    print(f"  如果Delta_W很小 -> 权重接近随机 -> 频谱接近Marchenko-Pastur")
    print(f"  如果Delta_W很大 -> 权重高度结构化 -> 频谱偏离MP分布")
    
    # W_U^T SVD
    W_U = get_W_U(model)
    W_UT = W_U.T
    k_wut = min(400, min(W_UT.shape) - 1)
    U_wut, s_wut, _ = svd(W_UT, full_matrices=False)
    U_wut = U_wut[:, :k_wut].T
    s_wut = s_wut[:k_wut]
    
    # 对所有层分析
    sample_layers = list(range(0, n_layers, max(1, n_layers // 15))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    results = []
    for layer_idx in sample_layers:
        layers = get_layers(model)
        lw = get_layer_weights(layers[layer_idx], d_model, info.mlp_type)
        W_down = lw.W_down
        
        if W_down is None:
            continue
        
        m, n = W_down.shape  # [d_model, intermediate_size]
        
        # 1. 训练权重的截断SVD(避免大矩阵内存问题)
        k_svd = min(300, min(W_down.shape) - 1)
        try:
            from sklearn.utils.extmath import randomized_svd
            U_tr, s_tr, Vt_tr = randomized_svd(W_down, n_components=k_svd, random_state=42)
        except ImportError:
            U_tr, s_tr, Vt_tr = svd(W_down, full_matrices=False)
            s_tr = s_tr[:k_svd]
        
        # 2. 随机矩阵理论值(避免大矩阵SVD的内存问题)
        # 标准初始化: sigma ~ 1/sqrt(fan_in)
        sigma_init = 1.0 / np.sqrt(n)  # Xavier初始化
        
        # MP分布理论值
        mp_radius = sigma_init**2 * (1 + np.sqrt(m / n))**2
        # 随机矩阵的期望范数: ||W||_F ~ sqrt(m*n) * sigma
        norm_random = np.sqrt(m * n) * sigma_init
        # 随机矩阵的最大奇异值期望: sigma * (sqrt(m) + sqrt(n))
        s_max_random = sigma_init * (np.sqrt(m) + np.sqrt(n))
        
        # 3. 计算训练残留
        # 方法1: 范数比 ||W_trained|| / ||W_random||
        norm_trained = np.linalg.norm(W_down, 'fro')
        norm_ratio = norm_trained / norm_random
        
        # 方法2: SVD谱偏离MP分布
        actual_radius = s_tr[0]**2
        mp_deviation = actual_radius / mp_radius
        
        # 方法3: 最大奇异值比 s_max_trained / s_max_random(理论值)
        s_max_ratio = s_tr[0] / s_max_random
        
        # 方法4: 信息量 = 谱熵(只对训练权重, 与MP理论熵比较)
        p_tr = s_tr**2 / np.sum(s_tr**2)
        entropy_tr = -np.sum(p_tr * np.log(p_tr + 1e-30))
        # MP分布的理论熵: log(min(m,n)) (均匀分布)
        entropy_mp = np.log(min(m, n))
        info_ratio = entropy_tr / entropy_mp  # <1 表示更集中(更多结构)
        
        # 方法5: 前10个奇异值的能量占比 vs MP理论
        top10_tr = np.sum(s_tr[:10]**2) / np.sum(s_tr**2)
        # MP理论: 前10个的占比 ~ 10/min(m,n)
        top10_mp = 10.0 / min(m, n)
        top10_ratio = top10_tr / top10_mp  # >1 表示更集中
        
        # Alpha
        alpha_mean, alpha_std = compute_alpha_for_layer(
            layer_idx, model, tokenizer, device, U_wut, s_wut, k_wut
        )
        
        results.append({
            'layer': layer_idx,
            'layer_frac': layer_idx / max(1, n_layers - 1),
            'alpha': alpha_mean,
            'norm_ratio': norm_ratio,
            'mp_deviation': mp_deviation,
            's_max_ratio': s_max_ratio,
            'info_ratio': info_ratio,
            'top10_ratio': top10_ratio,
            'top10_trained': top10_tr,
            'entropy_trained': entropy_tr,
            'entropy_random': entropy_mp,
        })
    
    release_model(model)
    
    # 分析训练残留与alpha的关系
    alphas = np.array([r['alpha'] for r in results])
    norm_ratios = np.array([r['norm_ratio'] for r in results])
    mp_devs = np.array([r['mp_deviation'] for r in results])
    s_max_ratios = np.array([r['s_max_ratio'] for r in results])
    info_ratios = np.array([r['info_ratio'] for r in results])
    top10_ratios = np.array([r['top10_ratio'] for r in results])
    
    # 相关性
    corr_norm = np.corrcoef(norm_ratios, alphas)[0, 1] if len(alphas) > 2 else 0
    corr_mp = np.corrcoef(mp_devs, alphas)[0, 1] if len(alphas) > 2 else 0
    corr_smax = np.corrcoef(s_max_ratios, alphas)[0, 1] if len(alphas) > 2 else 0
    corr_info = np.corrcoef(info_ratios, alphas)[0, 1] if len(alphas) > 2 else 0
    corr_top10 = np.corrcoef(top10_ratios, alphas)[0, 1] if len(alphas) > 2 else 0
    
    print(f"\n  Training Residual Metrics:")
    print(f"    mean_norm_ratio = {np.mean(norm_ratios):.3f}")
    print(f"    mean_mp_deviation = {np.mean(mp_devs):.3f}")
    print(f"    mean_s_max_ratio = {np.mean(s_max_ratios):.3f}")
    print(f"    mean_info_ratio = {np.mean(info_ratios):.3f}")
    print(f"    mean_top10_ratio = {np.mean(top10_ratios):.3f}")
    
    print(f"\n  Correlation with alpha:")
    print(f"    norm_ratio vs alpha: {corr_norm:.3f}")
    print(f"    mp_deviation vs alpha: {corr_mp:.3f}")
    print(f"    s_max_ratio vs alpha: {corr_smax:.3f}")
    print(f"    info_ratio vs alpha: {corr_info:.3f}")
    print(f"    top10_ratio vs alpha: {corr_top10:.3f}")
    
    # 每层详细
    print(f"\n  Per-layer Data:")
    print(f"    {'Layer':>5} {'alpha':>7} {'NR':>6} {'MP':>6} {'SMR':>6} {'IR':>6} {'T10':>6}")
    for r in results:
        print(f"    {r['layer']:5d} {r['alpha']:7.3f} {r['norm_ratio']:6.3f} "
              f"{r['mp_deviation']:6.2f} {r['s_max_ratio']:6.3f} "
              f"{r['info_ratio']:6.3f} {r['top10_ratio']:6.3f}")
    
    # 最佳预测因子
    correlations = {
        'norm_ratio': corr_norm,
        'mp_deviation': corr_mp,
        's_max_ratio': corr_smax,
        'info_ratio': corr_info,
        'top10_ratio': corr_top10,
    }
    best_feat = max(correlations, key=lambda k: abs(correlations[k]))
    print(f"\n  Best predictor of alpha: {best_feat} (corr={correlations[best_feat]:.3f})")
    
    return {
        'model': model_name,
        'correlations': correlations,
        'best_predictor': best_feat,
        'mean_norm_ratio': float(np.mean(norm_ratios)),
        'mean_mp_deviation': float(np.mean(mp_devs)),
        'mean_s_max_ratio': float(np.mean(s_max_ratios)),
        'mean_info_ratio': float(np.mean(info_ratios)),
        'mean_top10_ratio': float(np.mean(top10_ratios)),
        'layer_data': results,
    }


# ===== P478: 残差流信号强度演化 =====

def run_p478(model_name):
    """残差流信号强度演化"""
    print(f"\n{'='*70}")
    print(f"P478: Residual Stream Signal Evolution - {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    print(f"\n  核心思想: 信号通过残差连接传播 ||h_L|| = ||h_0 + sum_A(h_l)||")
    print(f"  信号增长 = alpha的宏观表现: alpha>0 => 信号增长, alpha<0 => 信号衰减")
    
    # W_U^T SVD
    W_U = get_W_U(model)
    W_UT = W_U.T
    k_wut = min(400, min(W_UT.shape) - 1)
    U_wut, s_wut, _ = svd(W_UT, full_matrices=False)
    U_wut = U_wut[:, :k_wut].T
    s_wut = s_wut[:k_wut]
    
    # 前向传播获取残差流
    test_texts = [
        "The apple is",
        "A cat sat on the",
        "In the world of",
        "The most important",
        "Scientists discovered that",
    ]
    
    # 收集所有文本的hidden states
    all_h_norms = []  # [n_texts, n_layers+1]
    all_alphas = []   # [n_texts, n_layers]
    
    for text in test_texts:
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_states = outputs.hidden_states  # tuple of [1, seq_len, d_model]
        
        # 取最后一个token的hidden state
        h_norms = []
        for L in range(len(h_states)):
            h_L = h_states[L][0, -1].cpu().float().numpy()
            h_norms.append(np.linalg.norm(h_L))
        all_h_norms.append(h_norms)
    
    # 平均
    mean_h_norms = np.mean(all_h_norms, axis=0)  # [n_layers+1]
    h0_norm = mean_h_norms[0]
    
    # 信号比
    signal_ratios = mean_h_norms / h0_norm  # [n_layers+1]
    
    # 信号增长率
    signal_growths = []
    for L in range(1, len(signal_ratios)):
        growth = (signal_ratios[L] - signal_ratios[L-1]) / signal_ratios[L-1]
        signal_growths.append(growth)
    
    # Alpha (对所有层)
    sample_layers = list(range(0, n_layers, max(1, n_layers // 15))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    results = []
    for layer_idx in sample_layers:
        alpha_mean, alpha_std = compute_alpha_for_layer(
            layer_idx, model, tokenizer, device, U_wut, s_wut, k_wut
        )
        
        # 信号特征
        sig_ratio = signal_ratios[layer_idx + 1] if layer_idx + 1 < len(signal_ratios) else signal_ratios[-1]
        sig_growth = signal_growths[layer_idx] if layer_idx < len(signal_growths) else 0
        
        results.append({
            'layer': layer_idx,
            'layer_frac': layer_idx / max(1, n_layers - 1),
            'alpha': alpha_mean,
            'signal_ratio': sig_ratio,
            'signal_growth': sig_growth,
            'h_norm': mean_h_norms[layer_idx + 1] if layer_idx + 1 < len(mean_h_norms) else mean_h_norms[-1],
        })
    
    release_model(model)
    
    # Alpha vs signal features
    alphas = np.array([r['alpha'] for r in results])
    sig_ratios = np.array([r['signal_ratio'] for r in results])
    sig_growths = np.array([r['signal_growth'] for r in results])
    
    corr_ratio = np.corrcoef(sig_ratios, alphas)[0, 1] if len(alphas) > 2 else 0
    corr_growth = np.corrcoef(sig_growths, alphas)[0, 1] if len(alphas) > 2 else 0
    
    print(f"\n  Signal Evolution Summary:")
    print(f"    h_0 norm = {h0_norm:.2f}")
    print(f"    h_final norm = {mean_h_norms[-1]:.2f}")
    print(f"    final signal_ratio = {signal_ratios[-1]:.3f}")
    print(f"    max signal_ratio = {max(signal_ratios):.3f} at layer {np.argmax(signal_ratios)}")
    
    print(f"\n  Alpha vs Signal Correlation:")
    print(f"    alpha vs signal_ratio: {corr_ratio:.3f}")
    print(f"    alpha vs signal_growth: {corr_growth:.3f}")
    
    # 每层详细
    print(f"\n  Per-layer Data:")
    print(f"    {'Layer':>5} {'alpha':>7} {'SigR':>7} {'SigG':>7} {'h_norm':>8}")
    for r in results:
        print(f"    {r['layer']:5d} {r['alpha']:7.3f} {r['signal_ratio']:7.3f} "
              f"{r['signal_growth']:7.4f} {r['h_norm']:8.1f}")
    
    # 信号增长的总体趋势
    log_signal = np.log(signal_ratios[1:] + 1e-10)
    log_L = np.log(np.arange(1, len(log_signal) + 1, dtype=float))
    
    if len(log_L) > 3:
        slope, intercept, r_val, p_val, std_err = stats.linregress(log_L, log_signal)
        print(f"\n  Signal Growth Law:")
        print(f"    log(||h_L||/||h_0||) ~ {slope:.3f} * log(L) + {intercept:.3f}")
        print(f"    => ||h_L|| / ||h_0|| ~ L^{slope:.3f}")
        print(f"    R = {r_val:.3f}")
    
    return {
        'model': model_name,
        'signal_ratios': signal_ratios.tolist(),
        'mean_h_norms': mean_h_norms.tolist(),
        'corr_ratio': float(corr_ratio),
        'corr_growth': float(corr_growth),
        'layer_data': results,
    }


# ===== P479: 多变量alpha预测模型 =====

def run_p479(model_name):
    """多变量alpha预测模型"""
    print(f"\n{'='*70}")
    print(f"P479: Multivariate Alpha Prediction - {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    intermediate_size = info.intermediate_size
    FFN_ratio = intermediate_size / d_model
    d_over_n = d_model * 1.0 / n_layers
    
    # W_U^T SVD
    W_U = get_W_U(model)
    W_UT = W_U.T
    k_wut = min(400, min(W_UT.shape) - 1)
    U_wut, s_wut, _ = svd(W_UT, full_matrices=False)
    U_wut = U_wut[:, :k_wut].T
    s_wut = s_wut[:k_wut]
    
    # delta
    log_s_wut = np.log(s_wut[:200] + 1e-30)
    log_i = np.log(np.arange(1, 201, dtype=float))
    slope_wut, _, _, _, _ = stats.linregress(log_i, log_s_wut)
    delta = -slope_wut
    
    # 所有层的完整数据
    sample_layers = list(range(0, n_layers, max(1, n_layers // 15))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    results = []
    for layer_idx in sample_layers:
        layers = get_layers(model)
        lw = get_layer_weights(layers[layer_idx], d_model, info.mlp_type)
        W_down = lw.W_down
        
        if W_down is None:
            continue
        
        # Alpha
        alpha_mean, alpha_std = compute_alpha_for_layer(
            layer_idx, model, tokenizer, device, U_wut, s_wut, k_wut
        )
        
        # W_down 特征
        wdown_feat = compute_wdown_svd_features(W_down)
        
        # 训练残留
        m, n = W_down.shape
        sigma_init = 1.0 / np.sqrt(n)
        norm_trained = np.linalg.norm(W_down, 'fro')
        norm_random = np.sqrt(m * n) * sigma_init
        norm_ratio = norm_trained / norm_random
        
        # 前向传播的信号强度(近似: 使用LayerNorm的权重范数)
        ln_weight = lw.post_attn_layernorm_weight
        ln_norm = np.linalg.norm(ln_weight) if ln_weight is not None else 1.0
        
        layer_frac = layer_idx / max(1, n_layers - 1)
        
        results.append({
            'layer': layer_idx,
            'layer_frac': layer_frac,
            'alpha': alpha_mean,
            'delta_W': wdown_feat['delta_W'],
            'F_W': wdown_feat['F_W'],
            'PR': wdown_feat['PR'],
            'kappa': wdown_feat['kappa'],
            'sigma_W': wdown_feat['sigma_W'],
            'norm_ratio': norm_ratio,
            'ln_norm': ln_norm,
        })
    
    release_model(model)
    
    if len(results) < 5:
        print(f"  Not enough data points ({len(results)})")
        return {'model': model_name, 'n_data': len(results)}
    
    # 构建特征矩阵
    feature_names = ['layer_frac', 'delta_W', 'F_W', 'PR', 'kappa', 
                     'sigma_W', 'norm_ratio', 'ln_norm']
    
    X = np.array([[r[f] for f in feature_names] for r in results])
    y = np.array([r['alpha'] for r in results])
    
    # 1. 单变量相关性
    print(f"\n  Single Feature Correlations with alpha:")
    single_corrs = {}
    for i, fname in enumerate(feature_names):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        single_corrs[fname] = corr
        print(f"    {fname:>12}: corr = {corr:.3f}")
    
    # 2. 逐步回归
    print(f"\n  Stepwise Regression:")
    remaining = list(range(len(feature_names)))
    selected = []
    best_r2 = -1
    
    while remaining:
        best_new_r2 = -1
        best_feat_idx = -1
        
        for idx in remaining:
            trial = selected + [idx]
            X_trial = X[:, trial]
            
            # OLS
            X_aug = np.column_stack([X_trial, np.ones(len(y))])
            try:
                coeffs, residuals, rank, sv = np.linalg.lstsq(X_aug, y, rcond=None)
                y_pred = X_aug @ coeffs
                ss_res = np.sum((y - y_pred)**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                
                # 调整R2 (惩罚参数数量)
                n = len(y)
                k = len(trial)
                r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else r2
                
                if r2_adj > best_new_r2:
                    best_new_r2 = r2_adj
                    best_feat_idx = idx
                    best_coeffs = coeffs
            except:
                continue
        
        if best_feat_idx < 0:
            break
        
        selected.append(best_feat_idx)
        remaining.remove(best_feat_idx)
        
        # 计算当前模型的R2
        X_sel = np.column_stack([X[:, selected], np.ones(len(y))])
        coeffs, _, _, _ = np.linalg.lstsq(X_sel, y, rcond=None)
        y_pred = X_sel @ coeffs
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot
        n = len(y)
        k = len(selected)
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        
        sel_names = [feature_names[i] for i in selected]
        print(f"    Step {len(selected)}: add {feature_names[best_feat_idx]}")
        print(f"      features = {sel_names}")
        print(f"      R2 = {r2:.3f}, R2_adj = {r2_adj:.3f}")
        
        # 如果R2不再显著改善, 停止
        if r2_adj < best_r2 + 0.01:
            break
        best_r2 = r2_adj
    
    # 3. 最终模型
    print(f"\n  Final Model:")
    sel_names = [feature_names[i] for i in selected]
    X_final = np.column_stack([X[:, selected], np.ones(len(y))])
    coeffs_final, _, _, _ = np.linalg.lstsq(X_final, y, rcond=None)
    
    y_pred_final = X_final @ coeffs_final
    ss_res = np.sum((y - y_pred_final)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2_final = 1 - ss_res / ss_tot
    
    print(f"    alpha = ", end="")
    for i, name in enumerate(sel_names):
        print(f"{coeffs_final[i]:+.3f}*{name} ", end="")
    print(f"{coeffs_final[-1]:+.3f}")
    print(f"    R2 = {r2_final:.3f}")
    
    # 4. 带架构参数的全局模型
    print(f"\n  Architecture-augmented Model:")
    print(f"    (delta={delta:.3f}, FFN_ratio={FFN_ratio:.3f}, d/n_L={d_over_n:.1f})")
    
    # 添加全局特征
    X_global = np.column_stack([X, 
                                 np.full(len(y), delta),
                                 np.full(len(y), FFN_ratio),
                                 np.full(len(y), d_over_n)])
    global_feature_names = feature_names + ['delta_global', 'FFN_ratio', 'd_over_n']
    
    # 逐步回归(全局版)
    remaining_g = list(range(len(global_feature_names)))
    selected_g = []
    best_r2_g = -1
    
    while remaining_g:
        best_new_r2 = -1
        best_feat_idx = -1
        
        for idx in remaining_g:
            trial = selected_g + [idx]
            X_trial = X_global[:, trial]
            X_aug = np.column_stack([X_trial, np.ones(len(y))])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
                y_pred = X_aug @ coeffs
                ss_res = np.sum((y - y_pred)**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                n = len(y)
                k = len(trial)
                r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else r2
                
                if r2_adj > best_new_r2:
                    best_new_r2 = r2_adj
                    best_feat_idx = idx
            except:
                continue
        
        if best_feat_idx < 0:
            break
        
        selected_g.append(best_feat_idx)
        remaining_g.remove(best_feat_idx)
        
        X_sel = np.column_stack([X_global[:, selected_g], np.ones(len(y))])
        coeffs, _, _, _ = np.linalg.lstsq(X_sel, y, rcond=None)
        y_pred = X_sel @ coeffs
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot
        n = len(y)
        k = len(selected_g)
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        
        sel_names_g = [global_feature_names[i] for i in selected_g]
        print(f"    Step {len(selected_g)}: add {global_feature_names[best_feat_idx]}")
        print(f"      R2 = {r2:.3f}, R2_adj = {r2_adj:.3f}")
        
        if r2_adj < best_r2_g + 0.01:
            break
        best_r2_g = r2_adj
    
    # 最终全局模型
    sel_names_g = [global_feature_names[i] for i in selected_g]
    X_final_g = np.column_stack([X_global[:, selected_g], np.ones(len(y))])
    coeffs_final_g, _, _, _ = np.linalg.lstsq(X_final_g, y, rcond=None)
    
    y_pred_fg = X_final_g @ coeffs_final_g
    ss_res = np.sum((y - y_pred_fg)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2_fg = 1 - ss_res / ss_tot
    
    print(f"\n  Final Global Model:")
    print(f"    alpha = ", end="")
    for i, name in enumerate(sel_names_g):
        print(f"{coeffs_final_g[i]:+.3f}*{name} ", end="")
    print(f"{coeffs_final_g[-1]:+.3f}")
    print(f"    R2 = {r2_fg:.3f}")
    
    return {
        'model': model_name,
        'n_data': len(results),
        'single_corrs': single_corrs,
        'final_features': sel_names,
        'final_coeffs': coeffs_final.tolist(),
        'r2_final': float(r2_final),
        'global_features': sel_names_g,
        'global_coeffs': coeffs_final_g.tolist(),
        'r2_global': float(r2_fg),
        'delta': delta,
        'FFN_ratio': FFN_ratio,
        'd_over_n': d_over_n,
        'layer_data': results,
    }


# ===== 主函数 =====

EXPERIMENTS = {
    'p476': run_p476,
    'p477': run_p477,
    'p478': run_p478,
    'p479': run_p479,
}

def main():
    parser = argparse.ArgumentParser(description="Phase XCIX: Training Dynamics & Signal Focusing")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["qwen3", "glm4", "deepseek7b"],
                       help="Model to test")
    parser.add_argument("--experiment", type=str, required=True,
                       choices=list(EXPERIMENTS.keys()),
                       help="Experiment to run")
    args = parser.parse_args()
    
    print(f"\nPhase XCIX: Training Dynamics & Signal Focusing")
    print(f"Model: {args.model}, Experiment: {args.experiment}")
    
    result = EXPERIMENTS[args.experiment](args.model)
    
    # 保存结果
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5_temp')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.experiment}_{args.model}.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Phase XCIX - {args.experiment} - {args.model}\n")
        f.write(f"="*50 + "\n")
        for key, val in result.items():
            if key == 'layer_data':
                f.write(f"\nLayer Data ({len(val)} layers):\n")
                for r in val:
                    f.write(f"  {r}\n")
            else:
                f.write(f"{key}: {val}\n")
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
