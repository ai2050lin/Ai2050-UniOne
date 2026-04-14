"""
Phase CI-P484/485/486/487: 因果链统一框架
======================================================================

核心目标: 建立统一的信号聚焦理论, 解决三模型因果链不同的困境

Phase C核心成果:
1. 三模型架构相同(RMSNorm+pre-norm), 但因果链完全不同
2. LN因果效应模型相关: Qwen3正向, DS7B负向, GLM4层依赖
3. alpha→gamma因果链只在Qwen3/GLM4中成立, DS7B中R2=0.01
4. GLM4二次增长(beta≈1.89)来自急剧增长的post_ln权重

关键问题:
1. gamma的定义是否有问题? gamma=1-c*alpha是否只在特定模型成立?
2. alpha的属性干预方法是否在所有模型中都有效?
3. 是否存在更深层的统一量, 可以替代gamma和alpha?
4. "权重驱动聚焦"和"动态驱动聚焦"能否统一?

Phase CI目标:
1. 重新审视gamma的计算方法, 检查之前的gamma-alpha关系
2. 用多种方式定义alpha和gamma, 比较其模型相关性
3. 寻找更深层的统一量(如信息几何量、有效维度等)
4. 建立分类理论: 聚焦模式分类

P484: Gamma定义的重新审视
  - 目标: 检查gamma的定义和计算, 验证gamma=1-c*alpha关系
  - 方法:
    a) 重新计算gamma: 用W_U^T的完整谱, 不只是前k个
    b) 用不同的k值计算gamma, 检查稳健性
    c) 检查gamma与alpha的完整关系(不只是线性)
    d) 分析gamma的定义是否对特定模型有偏

P485: Alpha的多种定义与比较
  - 目标: 比较不同alpha定义的模型相关性
  - 方法:
    a) alpha_attribution: 属性干预(当前方法)
    b) alpha_spectral: W_down SVD谱的幂律指数
    c) alpha_intrinsic: 子空间内在维度
    d) alpha_entropy: delta_h的谱熵比
  - 比较: 哪种定义在不同模型中最一致?

P486: 寻找统一量 - 信息几何与有效维度
  - 目标: 寻找跨模型一致的信号聚焦指标
  - 方法:
    a) 有效维度: d_eff = (sum sigma_i)^2 / sum sigma_i^2
    b) 参与比: PR = (sum sigma_i)^2 / (n * sum sigma_i^2)
    c) Fisher信息量: I = sum (sigma_i - sigma_i_random)^2 / sigma_i_random^2
    d) 信息密度: rho = I / d_eff
  - 关键: 这些量是否在跨模型中更一致?

P487: 聚焦模式分类理论
  - 目标: 建立信号聚焦的分类理论
  - 方法:
    a) 用P484-P486的所有特征做聚类
    b) 识别聚焦模式: "权重驱动型" vs "动态驱动型" vs "混合型"
    c) 建立分类判据: 用哪些特征可以区分模式?
    d) 验证: 同类模式的因果链是否更相似?
"""

import sys
import os
import argparse
import numpy as np
from scipy.linalg import svd
from scipy.stats import pearsonr, spearmanr
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))
from model_utils import (load_model, get_layers, get_layer_weights,
                         get_model_info, release_model, get_sample_layers,
                         get_W_U, get_attr_direction, inject_at_embed)

# 结果输出目录
RESULT_DIR = Path(__file__).parent.parent / "glm5_temp"
RESULT_DIR.mkdir(exist_ok=True)


def truncated_svd(matrix, k_max=400):
    """截断SVD, 避免大矩阵内存问题"""
    k = min(k_max, min(matrix.shape) - 1)
    # 对大矩阵限制k以避免内存问题
    if matrix.shape[0] > 50000 or matrix.shape[1] > 50000:
        k = min(k, 200)
    if matrix.shape[0] > 100000 or matrix.shape[1] > 100000:
        k = min(k, 100)
    try:
        from sklearn.utils.extmath import randomized_svd
        U, s, Vt = randomized_svd(matrix.astype(np.float32), n_components=k, random_state=42)
        return U, s, Vt
    except (ImportError, MemoryError):
        # 降级到更小的k或使用np.linalg.svd
        k = min(k, 100)
        try:
            from sklearn.utils.extmath import randomized_svd
            U, s, Vt = randomized_svd(matrix.astype(np.float32), n_components=k, random_state=42)
            return U, s, Vt
        except:
            U, s, Vt = svd(matrix[:min(5000, matrix.shape[0])], full_matrices=False)
            return U[:, :k], s[:k], Vt[:k, :]


def compute_spectral_density(delta_h, U_wut, s_wut, k_max):
    """
    计算delta_h在W_U^T谱上的密度分布
    返回: (lambdas, alpha, fit_R2, r2)
    """
    # 投影到W_U^T的奇异空间
    proj = U_wut @ delta_h  # [k]
    
    # 加权: 按奇异值缩放
    weights = s_wut[:len(proj)]
    
    # 谱密度: 投影的绝对值 vs 奇异值
    lambdas = weights
    densities = np.abs(proj)
    
    # 幂律拟合: density ~ lambda^alpha
    valid = (lambdas > 0) & (densities > 0)
    if np.sum(valid) < 5:
        return lambdas, 0.0, 0.0, 0.0
    
    log_lambda = np.log(lambdas[valid])
    log_density = np.log(densities[valid])
    
    # 线性回归
    try:
        from numpy.polynomial import polynomial as P
        coeffs = P.polyfit(log_lambda, log_density, 1)
        alpha = coeffs[1]
        pred = P.polyval(log_lambda, coeffs)
        ss_res = np.sum((log_density - pred)**2)
        ss_tot = np.sum((log_density - np.mean(log_density))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    except:
        alpha = 0.0
        r2 = 0.0
    
    return lambdas, alpha, 0.0, r2


def compute_alpha_attribution(layer_idx, model, tokenizer, device, U_wut, s_wut, k_max):
    """P485a: 属性干预alpha (当前方法)"""
    base_text = "The apple is"
    base_ids = tokenizer.encode(base_text, return_tensors="pt").to(device)
    with torch.no_grad():
        base_out = model(base_ids, output_hidden_states=True)
        base_h = base_out.hidden_states[layer_idx + 1][0, -1].cpu().float().numpy()
    
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
        alpha_list.append(alpha)
    
    return np.mean(alpha_list) if alpha_list else 0.0


def compute_alpha_spectral(W_down, k_svd=300):
    """P485b: W_down SVD谱的幂律指数"""
    U, s, Vt = truncated_svd(W_down, k_svd)
    
    # 拟合 s_i ~ i^(-alpha)
    valid = s > 0
    if np.sum(valid) < 10:
        return 0.0, 0.0
    
    indices = np.arange(1, len(s) + 1)[valid]
    log_idx = np.log(indices)
    log_s = np.log(s[valid])
    
    try:
        from numpy.polynomial import polynomial as P
        coeffs = P.polyfit(log_idx, log_s, 1)
        alpha = -coeffs[1]  # 负号因为s_i ~ i^(-alpha)
        pred = P.polyval(log_idx, coeffs)
        ss_res = np.sum((log_s - pred)**2)
        ss_tot = np.sum((log_s - np.mean(log_s))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    except:
        alpha = 0.0
        r2 = 0.0
    
    return alpha, r2


def compute_alpha_intrinsic(W_down, threshold=0.99):
    """P485c: 子空间内在维度"""
    U, s, Vt = truncated_svd(W_down, 300)
    
    # 累计能量占比
    energy = np.cumsum(s**2) / np.sum(s**2)
    d_intrinsic = np.searchsorted(energy, threshold) + 1
    
    # 归一化
    d_intrinsic_norm = d_intrinsic / len(s)
    
    return d_intrinsic, d_intrinsic_norm


def compute_alpha_entropy(W_down, k_svd=300):
    """P485d: W_down谱的归一化熵"""
    U, s, Vt = truncated_svd(W_down, k_svd)
    
    # 归一化谱
    p = s**2 / np.sum(s**2)
    
    # 熵
    entropy = -np.sum(p * np.log(p + 1e-30))
    
    # 最大熵(均匀分布)
    max_entropy = np.log(len(s))
    
    # 归一化熵 [0, 1], 越小越集中
    entropy_norm = entropy / max_entropy if max_entropy > 0 else 1.0
    
    return entropy, entropy_norm


def safe_WUT_svd(W_U, k_max=400):
    """
    安全的W_U^T SVD, 对大词汇表使用替代方法
    
    Returns:
        U_wut: [k, d_model] 投影矩阵
        s_wut: [k] 奇异值
    """
    d_model = W_U.shape[1]
    vocab_size = W_U.shape[0]
    
    if vocab_size > 50000:
        # 大词汇表: 用W_U子集做QR分解
        k = min(k_max, vocab_size, d_model)
        # 采样一些token方向做QR
        np.random.seed(42)
        indices = np.random.choice(vocab_size, min(k * 3, vocab_size), replace=False)
        W_sub = W_U[indices]  # [k*3, d_model]
        Q, R = np.linalg.qr(W_sub.T)  # Q: [d_model, k*3]
        U_wut = Q[:, :k].T  # [k, d_model]
        # 用R的对角线近似奇异值
        s_wut = np.abs(np.diag(R[:k, :k]))
        if len(s_wut) < k:
            s_wut = np.pad(s_wut, (0, k - len(s_wut)))
        return U_wut, s_wut
    else:
        # 小词汇表: 用截断SVD
        W_UT = W_U.T
        k = min(k_max, min(W_UT.shape) - 1)
        U, s, Vt = truncated_svd(W_UT.astype(np.float32), k)
        return U.T, s  # U.T: [k, d_model], s: [k]


def compute_effective_dimension(s):
    """P486a: 有效维度 d_eff = (sum sigma_i)^2 / sum sigma_i^2"""
    s2 = s**2
    d_eff = np.sum(s)**2 / np.sum(s2) if np.sum(s2) > 0 else 0
    return d_eff


def compute_participation_ratio(s):
    """P486b: 参与比 PR = (sum sigma_i^2)^2 / (n * sum sigma_i^4)"""
    s2 = s**2
    s4 = s**4
    n = len(s)
    PR = np.sum(s2)**2 / (n * np.sum(s4)) if np.sum(s4) > 0 else 0
    return PR


def compute_fisher_information(s_trained, s_random_theory):
    """P486c: Fisher信息量 (训练谱vs随机谱)"""
    # s_random_theory: 随机矩阵的理论奇异值(MP分布)
    valid = s_random_theory > 0
    if np.sum(valid) < 5:
        return 0.0
    
    # Fisher信息: sum (sigma_trained - sigma_random)^2 / sigma_random^2
    I = np.sum((s_trained[valid] - s_random_theory[valid])**2 / s_random_theory[valid]**2)
    return I


def compute_gamma_comprehensive(delta_h_list, W_U, k_values=[50, 100, 200, 400]):
    """
    P484: 用不同k值计算gamma, 检查稳健性
    
    gamma定义: delta_h在W_U^T子空间中的能量集中度
    gamma = ||P_k delta_h||^2 / ||delta_h||^2
    其中P_k是W_U^T前k个奇异向量的投影
    
    对大词汇表模型(如GLM4), 使用增量投影避免大矩阵SVD
    """
    d_model = W_U.shape[1]
    vocab_size = W_U.shape[0]
    
    # 对大词汇表使用增量方法
    if vocab_size > 50000:
        # 不做完整SVD, 而是用W_U本身作为投影基
        # gamma_k = ||W_U[:k] @ delta_h||^2 / ||delta_h||^2
        # 这里W_U[:k]是词汇表前k个token的方向
        gamma_dict = {}
        for k in k_values:
            k_actual = min(k, vocab_size)
            # 用W_U的前k行作为投影基(不完美但避免内存问题)
            W_sub = W_U[:k_actual]  # [k, d_model]
            # 正交化
            Q, R = np.linalg.qr(W_sub.T)  # Q: [d_model, k]
            
            gamma_list = []
            for delta_h in delta_h_list:
                proj = Q.T @ delta_h  # [k]
                energy_proj = np.sum(proj**2)
                energy_total = np.sum(delta_h**2)
                
                if energy_total > 0:
                    gamma = energy_proj / energy_total
                else:
                    gamma = 0.0
                gamma_list.append(gamma)
            
            gamma_dict[k] = {
                'mean': np.mean(gamma_list),
                'std': np.std(gamma_list),
                'min': np.min(gamma_list),
                'max': np.max(gamma_list),
            }
        return gamma_dict
    
    # 小词汇表: 用截断SVD
    W_UT = W_U.T  # [d_model, vocab_size]
    max_k = min(W_UT.shape) - 1
    
    gamma_dict = {}
    for k in k_values:
        k_actual = min(k, max_k)
        if k_actual < 10:
            continue
        U, s, Vt = truncated_svd(W_UT.astype(np.float32), k_actual)
        U_proj = U.T  # [k, d_model]
        
        gamma_list = []
        for delta_h in delta_h_list:
            proj = U_proj @ delta_h  # [k]
            energy_proj = np.sum(proj**2)
            energy_total = np.sum(delta_h**2)
            
            if energy_total > 0:
                gamma = energy_proj / energy_total
            else:
                gamma = 0.0
            gamma_list.append(gamma)
        
        gamma_dict[k] = {
            'mean': np.mean(gamma_list),
            'std': np.std(gamma_list),
            'min': np.min(gamma_list),
            'max': np.max(gamma_list),
        }
    
    return gamma_dict


# ============================================================
# P484: Gamma定义的重新审视
# ============================================================
def run_p484(model_name):
    import torch
    
    print(f"\n{'='*70}")
    print(f"P484: Gamma定义的重新审视 - {model_name}")
    print(f"{'='*70}")
    print(f"  目标: 检查gamma的计算稳健性和模型相关性")
    print(f"  方法: 用不同k值计算gamma, 检查gamma-alpha关系")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers = info.n_layers
    d_model = info.d_model
    
    print(f"\n  模型: {model_name}, n_L={n_layers}, d={d_model}")
    
    # W_U
    W_U = get_W_U(model)
    
    # 采样层
    sample_layers = get_sample_layers(n_layers, 8)
    print(f"  采样层: {sample_layers}")
    
    # 对每层计算alpha和gamma
    results = []
    for l_idx in sample_layers:
        print(f"\n  --- Layer {l_idx}/{n_layers-1} ---")
        
        # 计算alpha_attribution (安全的W_U^T SVD)
        U_wut, s_wut = safe_WUT_svd(W_U, 400)
        
        alpha_attr = compute_alpha_attribution(l_idx, model, tokenizer, device, U_wut, s_wut, len(s_wut))
        base_text = "The apple is"
        base_ids = tokenizer.encode(base_text, return_tensors="pt").to(device)
        with torch.no_grad():
            base_out = model(base_ids, output_hidden_states=True)
            base_h = base_out.hidden_states[l_idx + 1][0, -1].cpu().float().numpy()
        
        delta_h_list = []
        for attr_word in ["red", "green", "big", "small", "sweet", "sour"]:
            intervened_text = f"The {attr_word} apple is"
            interv_ids = tokenizer.encode(intervened_text, return_tensors="pt").to(device)
            with torch.no_grad():
                interv_out = model(interv_ids, output_hidden_states=True)
                interv_h = interv_out.hidden_states[l_idx + 1][0, -1].cpu().float().numpy()
            delta_h_list.append(interv_h - base_h)
        
        # 用不同k值计算gamma
        gamma_dict = compute_gamma_comprehensive(delta_h_list, W_U, k_values=[50, 100, 200, 400])
        
        # delta_h的范数统计
        delta_norms = [np.linalg.norm(dh) for dh in delta_h_list]
        
        layer_frac = l_idx / max(n_layers - 1, 1)
        
        result = {
            'layer': l_idx,
            'layer_frac': layer_frac,
            'alpha_attribution': alpha_attr,
            'mean_delta_norm': np.mean(delta_norms),
            'std_delta_norm': np.std(delta_norms),
        }
        for k, v in gamma_dict.items():
            result[f'gamma_k{k}_mean'] = v['mean']
            result[f'gamma_k{k}_std'] = v['std']
        
        results.append(result)
        print(f"    alpha_attr={alpha_attr:.4f}, gamma_k50={gamma_dict[50]['mean']:.4f}, "
              f"gamma_k100={gamma_dict[100]['mean']:.4f}, gamma_k200={gamma_dict[200]['mean']:.4f}")
    
    # 分析gamma的k稳健性
    print(f"\n  === Gamma的k稳健性 ===")
    k_values = [50, 100, 200, 400]
    gamma_means = {k: [r[f'gamma_k{k}_mean'] for r in results] for k in k_values}
    alpha_list = [r['alpha_attribution'] for r in results]
    
    for k in k_values:
        corr, p = pearsonr(gamma_means[k], alpha_list) if len(gamma_means[k]) > 2 else (0, 1)
        print(f"  gamma_k{k} vs alpha: corr={corr:.3f}, p={p:.4f}, mean={np.mean(gamma_means[k]):.4f}")
    
    # gamma之间的相关性
    print(f"\n  === Gamma之间的跨k相关性 ===")
    for i, k1 in enumerate(k_values):
        for j, k2 in enumerate(k_values):
            if j > i:
                corr, _ = pearsonr(gamma_means[k1], gamma_means[k2]) if len(gamma_means[k1]) > 2 else (0, 1)
                print(f"  gamma_k{k1} vs gamma_k{k2}: corr={corr:.3f}")
    
    # 检查gamma与layer_frac的关系
    layer_fracs = [r['layer_frac'] for r in results]
    print(f"\n  === Gamma与层位置的关系 ===")
    for k in k_values:
        corr, _ = pearsonr(gamma_means[k], layer_fracs) if len(gamma_means[k]) > 2 else (0, 1)
        print(f"  gamma_k{k} vs layer_frac: corr={corr:.3f}")
    
    # 保存结果
    out_path = RESULT_DIR / f"p484_{model_name}.txt"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"P484: Gamma定义的重新审视 - {model_name}\n")
        f.write(f"{'='*60}\n\n")
        for r in results:
            f.write(f"Layer {r['layer']:3d} (frac={r['layer_frac']:.3f}): "
                    f"alpha={r['alpha_attribution']:.4f}, "
                    f"delta_norm={r['mean_delta_norm']:.2f}±{r['std_delta_norm']:.2f}\n")
            for k in k_values:
                f.write(f"  gamma_k{k}={r[f'gamma_k{k}_mean']:.4f}±{r[f'gamma_k{k}_std']:.4f}\n")
        
        f.write(f"\n=== 跨k相关性 ===\n")
        for i, k1 in enumerate(k_values):
            for j, k2 in enumerate(k_values):
                if j > i:
                    corr, _ = pearsonr(gamma_means[k1], gamma_means[k2]) if len(gamma_means[k1]) > 2 else (0, 1)
                    f.write(f"gamma_k{k1} vs gamma_k{k2}: corr={corr:.3f}\n")
    
    print(f"\n  结果已保存到 {out_path}")
    
    release_model(model)
    return results


# ============================================================
# P485: Alpha的多种定义与比较
# ============================================================
def run_p485(model_name):
    import torch
    
    print(f"\n{'='*70}")
    print(f"P485: Alpha的多种定义与比较 - {model_name}")
    print(f"{'='*70}")
    print(f"  目标: 比较不同alpha定义的模型一致性")
    print(f"  方法: 计算4种alpha, 分析它们之间的相关性和与gamma的关系")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers = info.n_layers
    d_model = info.d_model
    
    print(f"\n  模型: {model_name}, n_L={n_layers}, d={d_model}")
    
    # W_U (安全的SVD)
    W_U = get_W_U(model)
    U_wut, s_wut = safe_WUT_svd(W_U, 400)
    
    # 采样层
    sample_layers = get_sample_layers(n_layers, 10)
    print(f"  采样层: {sample_layers}")
    
    results = []
    for l_idx in sample_layers:
        print(f"\n  --- Layer {l_idx}/{n_layers-1} ---")
        layer_frac = l_idx / max(n_layers - 1, 1)
        
        # 获取权重
        lw = get_layer_weights(layers[l_idx], d_model, info.mlp_type)
        W_down = lw.W_down
        
        # Alpha a: 属性干预
        alpha_attr = compute_alpha_attribution(l_idx, model, tokenizer, device, U_wut, s_wut, len(s_wut))
        
        # Alpha b: W_down谱幂律指数
        alpha_spectral, alpha_spec_r2 = compute_alpha_spectral(W_down)
        
        # Alpha c: 内在维度
        d_intrinsic, d_intrinsic_norm = compute_alpha_intrinsic(W_down)
        
        # Alpha d: 归一化熵
        entropy, entropy_norm = compute_alpha_entropy(W_down)
        
        # W_down SVD (复用)
        U_wd, s_wd, Vt_wd = truncated_svd(W_down, 300)
        
        # 有效维度和参与比
        d_eff = compute_effective_dimension(s_wd)
        PR = compute_participation_ratio(s_wd)
        
        # 条件数
        kappa = s_wd[0] / s_wd[-1] if s_wd[-1] > 0 else float('inf')
        
        # 随机矩阵理论值
        m, n = W_down.shape
        sigma_init = 1.0 / np.sqrt(n)
        s_max_random = sigma_init * (np.sqrt(m) + np.sqrt(n))
        
        # 训练残留
        norm_ratio = np.linalg.norm(W_down, 'fro') / (np.sqrt(m * n) * sigma_init)
        
        result = {
            'layer': l_idx,
            'layer_frac': layer_frac,
            'alpha_attribution': alpha_attr,
            'alpha_spectral': alpha_spectral,
            'alpha_spec_r2': alpha_spec_r2,
            'd_intrinsic': d_intrinsic,
            'd_intrinsic_norm': d_intrinsic_norm,
            'entropy': entropy,
            'entropy_norm': entropy_norm,
            'd_eff': d_eff,
            'PR': PR,
            'kappa': kappa,
            'norm_ratio': norm_ratio,
            's_max_ratio': s_wd[0] / s_max_random,
        }
        results.append(result)
        print(f"    alpha_attr={alpha_attr:.4f}, alpha_spec={alpha_spectral:.4f}(R2={alpha_spec_r2:.3f}), "
              f"d_intr={d_intrinsic}, entropy_norm={entropy_norm:.4f}, PR={PR:.4f}")
    
    # 分析4种alpha之间的相关性
    print(f"\n  === 4种Alpha之间的相关性 ===")
    alpha_names = ['alpha_attribution', 'alpha_spectral', 'd_intrinsic_norm', 'entropy_norm']
    alpha_data = {name: [r[name] for r in results] for name in alpha_names}
    
    for i, n1 in enumerate(alpha_names):
        for j, n2 in enumerate(alpha_names):
            if j > i:
                corr, p = pearsonr(alpha_data[n1], alpha_data[n2]) if len(alpha_data[n1]) > 2 else (0, 1)
                print(f"  {n1} vs {n2}: corr={corr:.3f}, p={p:.4f}")
    
    # 每种alpha与layer_frac的相关性
    layer_fracs = [r['layer_frac'] for r in results]
    print(f"\n  === Alpha与层位置的关系 ===")
    for name in alpha_names:
        corr, _ = pearsonr(alpha_data[name], layer_fracs) if len(alpha_data[name]) > 2 else (0, 1)
        print(f"  {name} vs layer_frac: corr={corr:.3f}")
    
    # 每种alpha与其他特征的关系
    print(f"\n  === Alpha与结构特征的关系 ===")
    features = ['d_eff', 'PR', 'kappa', 'norm_ratio', 's_max_ratio']
    for alpha_name in alpha_names:
        best_corr = 0
        best_feat = ''
        for feat in features:
            feat_data = [r[feat] for r in results]
            if len(feat_data) > 2 and np.std(feat_data) > 0 and np.std(alpha_data[alpha_name]) > 0:
                corr, _ = pearsonr(alpha_data[alpha_name], feat_data)
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_feat = feat
        print(f"  {alpha_name}: best_predictor={best_feat}(corr={best_corr:.3f})")
    
    # 保存结果
    out_path = RESULT_DIR / f"p485_{model_name}.txt"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"P485: Alpha的多种定义与比较 - {model_name}\n")
        f.write(f"{'='*60}\n\n")
        for r in results:
            f.write(f"Layer {r['layer']:3d} (frac={r['layer_frac']:.3f}):\n")
            f.write(f"  alpha_attribution={r['alpha_attribution']:.4f}\n")
            f.write(f"  alpha_spectral={r['alpha_spectral']:.4f}(R2={r['alpha_spec_r2']:.3f})\n")
            f.write(f"  d_intrinsic={r['d_intrinsic']}, d_intrinsic_norm={r['d_intrinsic_norm']:.4f}\n")
            f.write(f"  entropy_norm={r['entropy_norm']:.4f}\n")
            f.write(f"  d_eff={r['d_eff']:.2f}, PR={r['PR']:.4f}, kappa={r['kappa']:.2f}\n")
            f.write(f"  norm_ratio={r['norm_ratio']:.4f}, s_max_ratio={r['s_max_ratio']:.4f}\n\n")
    
    print(f"\n  结果已保存到 {out_path}")
    
    release_model(model)
    return results


# ============================================================
# P486: 寻找统一量 - 信息几何与有效维度
# ============================================================
def run_p486(model_name):
    import torch
    
    print(f"\n{'='*70}")
    print(f"P486: 寻找统一量 - 信息几何与有效维度 - {model_name}")
    print(f"{'='*70}")
    print(f"  目标: 寻找跨模型一致的信号聚焦指标")
    print(f"  方法: 计算多种信息几何量, 检查跨模型一致性")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers = info.n_layers
    d_model = info.d_model
    
    print(f"\n  模型: {model_name}, n_L={n_layers}, d={d_model}")
    
    # W_U (安全的SVD)
    W_U = get_W_U(model)
    U_wut, s_wut = safe_WUT_svd(W_U, 400)
    
    # 采样层
    sample_layers = get_sample_layers(n_layers, 10)
    print(f"  采样层: {sample_layers}")
    
    results = []
    for l_idx in sample_layers:
        print(f"\n  --- Layer {l_idx}/{n_layers-1} ---")
        layer_frac = l_idx / max(n_layers - 1, 1)
        
        # 获取权重
        lw = get_layer_weights(layers[l_idx], d_model, info.mlp_type)
        W_down = lw.W_down
        
        # W_down SVD
        U_wd, s_wd, Vt_wd = truncated_svd(W_down, 300)
        
        # 1. 有效维度
        d_eff = compute_effective_dimension(s_wd)
        
        # 2. 参与比
        PR = compute_participation_ratio(s_wd)
        
        # 3. Fisher信息量
        m, n = W_down.shape
        sigma_init = 1.0 / np.sqrt(n)
        # MP分布的理论谱 (近似: 均匀分布在[lambda_min, lambda_max])
        gamma_ratio = m / n  # aspect ratio
        lambda_max = sigma_init**2 * (1 + np.sqrt(gamma_ratio))**2
        lambda_min = sigma_init**2 * (1 - np.sqrt(gamma_ratio))**2 if gamma_ratio < 1 else 0
        # 理论奇异值: 均匀分布在sqrt范围
        mp_s = np.linspace(np.sqrt(lambda_max), np.sqrt(max(lambda_min, 0)), len(s_wd))
        
        fisher_I = compute_fisher_information(s_wd, mp_s)
        
        # 4. 信息密度
        info_density = fisher_I / d_eff if d_eff > 0 else 0
        
        # 5. delta_h分析
        alpha_attr = compute_alpha_attribution(l_idx, model, tokenizer, device, U_wut, s_wut, len(s_wut))
        
        # delta_h的谱结构
        base_text = "The apple is"
        base_ids = tokenizer.encode(base_text, return_tensors="pt").to(device)
        with torch.no_grad():
            base_out = model(base_ids, output_hidden_states=True)
            base_h = base_out.hidden_states[l_idx + 1][0, -1].cpu().float().numpy()
        
        delta_h_list = []
        for attr_word in ["red", "green", "big", "small", "sweet", "sour"]:
            intervened_text = f"The {attr_word} apple is"
            interv_ids = tokenizer.encode(intervened_text, return_tensors="pt").to(device)
            with torch.no_grad():
                interv_out = model(interv_ids, output_hidden_states=True)
                interv_h = interv_out.hidden_states[l_idx + 1][0, -1].cpu().float().numpy()
            delta_h_list.append(interv_h - base_h)
        
        # delta_h在W_U^T子空间中的投影
        proj_list = [U_wut @ dh for dh in delta_h_list]
        
        # delta_h的有效维度
        mean_proj = np.mean(proj_list, axis=0)
        proj_energies = [np.sum(p**2) for p in proj_list]
        delta_energies = [np.sum(dh**2) for dh in delta_h_list]
        
        # gamma = 投影能量 / 总能量
        gamma_list = [pe / de if de > 0 else 0 for pe, de in zip(proj_energies, delta_energies)]
        mean_gamma = np.mean(gamma_list)
        
        # delta_h的内在维度
        cov = np.cov(np.array(delta_h_list).T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = eigvals[::-1]  # 降序
        eigvals = eigvals[eigvals > 0]
        d_eff_delta = compute_effective_dimension(np.sqrt(eigvals))
        
        # LN权重
        input_ln_w = lw.input_layernorm_weight
        post_ln_w = lw.post_attn_layernorm_weight
        ln1_mean = np.mean(input_ln_w) if input_ln_w is not None else 0
        ln2_mean = np.mean(post_ln_w) if post_ln_w is not None else 0
        
        result = {
            'layer': l_idx,
            'layer_frac': layer_frac,
            'alpha_attribution': alpha_attr,
            'mean_gamma': mean_gamma,
            'd_eff': d_eff,
            'PR': PR,
            'fisher_I': fisher_I,
            'info_density': info_density,
            'd_eff_delta': d_eff_delta,
            'kappa': s_wd[0] / s_wd[-1] if s_wd[-1] > 0 else float('inf'),
            'ln1_mean': ln1_mean,
            'ln2_mean': ln2_mean,
            'norm_ratio': np.linalg.norm(W_down, 'fro') / (np.sqrt(m * n) * sigma_init),
            'mean_delta_norm': np.mean(delta_energies),
        }
        results.append(result)
        print(f"    alpha={alpha_attr:.4f}, gamma={mean_gamma:.4f}, "
              f"d_eff={d_eff:.1f}, PR={PR:.4f}, Fisher={fisher_I:.1f}, "
              f"info_dens={info_density:.4f}, d_eff_delta={d_eff_delta:.1f}")
    
    # 分析统一量与alpha/gamma的关系
    print(f"\n  === 统一量与alpha/gamma的关系 ===")
    alpha_list = [r['alpha_attribution'] for r in results]
    gamma_list = [r['mean_gamma'] for r in results]
    
    unified_features = ['d_eff', 'PR', 'fisher_I', 'info_density', 'd_eff_delta', 'kappa', 'norm_ratio']
    for feat in unified_features:
        feat_data = [r[feat] for r in results]
        if np.std(feat_data) > 0 and np.std(alpha_list) > 0:
            corr_alpha, _ = pearsonr(feat_data, alpha_list) if len(feat_data) > 2 else (0, 1)
            corr_gamma, _ = pearsonr(feat_data, gamma_list) if len(feat_data) > 2 else (0, 1)
            print(f"  {feat}: corr_alpha={corr_alpha:.3f}, corr_gamma={corr_gamma:.3f}")
    
    # alpha vs gamma
    corr_ag, _ = pearsonr(alpha_list, gamma_list) if len(alpha_list) > 2 else (0, 1)
    print(f"\n  alpha vs gamma: corr={corr_ag:.3f}")
    
    # 保存结果
    out_path = RESULT_DIR / f"p486_{model_name}.txt"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"P486: 寻找统一量 - {model_name}\n")
        f.write(f"{'='*60}\n\n")
        for r in results:
            f.write(f"Layer {r['layer']:3d} (frac={r['layer_frac']:.3f}): "
                    f"alpha={r['alpha_attribution']:.4f}, gamma={r['mean_gamma']:.4f}, "
                    f"d_eff={r['d_eff']:.1f}, PR={r['PR']:.4f}, "
                    f"Fisher={r['fisher_I']:.1f}, info_dens={r['info_density']:.4f}, "
                    f"d_eff_delta={r['d_eff_delta']:.1f}\n")
    
    print(f"\n  结果已保存到 {out_path}")
    
    release_model(model)
    return results


# ============================================================
# P487: 聚焦模式分类理论
# ============================================================
def run_p487(model_name):
    import torch
    
    print(f"\n{'='*70}")
    print(f"P487: 聚焦模式分类理论 - {model_name}")
    print(f"{'='*70}")
    print(f"  目标: 对每层进行聚焦模式分类")
    print(f"  方法: 基于P484-P486的特征, 识别聚焦模式")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers = info.n_layers
    d_model = info.d_model
    
    print(f"\n  模型: {model_name}, n_L={n_layers}, d={d_model}")
    
    # W_U (安全的SVD)
    W_U = get_W_U(model)
    U_wut, s_wut = safe_WUT_svd(W_U, 400)
    
    # 所有层
    all_layers = list(range(n_layers))
    print(f"  分析所有 {n_layers} 层")
    
    results = []
    for l_idx in all_layers:
        layer_frac = l_idx / max(n_layers - 1, 1)
        
        # 获取权重
        lw = get_layer_weights(layers[l_idx], d_model, info.mlp_type)
        W_down = lw.W_down
        
        # W_down SVD
        U_wd, s_wd, Vt_wd = truncated_svd(W_down, 300)
        
        # 计算关键特征
        d_eff = compute_effective_dimension(s_wd)
        PR = compute_participation_ratio(s_wd)
        entropy, entropy_norm = compute_alpha_entropy(W_down)
        d_intrinsic, d_intrinsic_norm = compute_alpha_intrinsic(W_down)
        alpha_spectral, _ = compute_alpha_spectral(W_down)
        
        # LN权重
        input_ln_w = lw.input_layernorm_weight
        post_ln_w = lw.post_attn_layernorm_weight
        ln1_mean = np.mean(input_ln_w) if input_ln_w is not None else 0
        ln2_mean = np.mean(post_ln_w) if post_ln_w is not None else 0
        
        # 训练残留
        m, n = W_down.shape
        sigma_init = 1.0 / np.sqrt(n)
        norm_ratio = np.linalg.norm(W_down, 'fro') / (np.sqrt(m * n) * sigma_init)
        
        # 条件数
        kappa = s_wd[0] / s_wd[-1] if s_wd[-1] > 0 else float('inf')
        
        # top10能量占比
        top10_energy = np.sum(s_wd[:10]**2) / np.sum(s_wd**2)
        
        # 采样层计算alpha
        if l_idx % max(1, n_layers // 8) == 0 or l_idx == n_layers - 1:
            alpha_attr = compute_alpha_attribution(l_idx, model, tokenizer, device, U_wut, s_wut, len(s_wut))
            
            # delta_h gamma
            base_text = "The apple is"
            base_ids = tokenizer.encode(base_text, return_tensors="pt").to(device)
            with torch.no_grad():
                base_out = model(base_ids, output_hidden_states=True)
                base_h = base_out.hidden_states[l_idx + 1][0, -1].cpu().float().numpy()
            
            delta_h_list = []
            for attr_word in ["red", "green", "big"]:
                intervened_text = f"The {attr_word} apple is"
                interv_ids = tokenizer.encode(intervened_text, return_tensors="pt").to(device)
                with torch.no_grad():
                    interv_out = model(interv_ids, output_hidden_states=True)
                    interv_h = interv_out.hidden_states[l_idx + 1][0, -1].cpu().float().numpy()
                delta_h_list.append(interv_h - base_h)
            
            proj_list = [U_wut @ dh for dh in delta_h_list]
            proj_energies = [np.sum(p**2) for p in proj_list]
            delta_energies = [np.sum(dh**2) for dh in delta_h_list]
            gamma_list = [pe / de if de > 0 else 0 for pe, de in zip(proj_energies, delta_energies)]
            mean_gamma = np.mean(gamma_list)
        else:
            alpha_attr = np.nan
            mean_gamma = np.nan
        
        # 聚焦模式分类规则
        # 模式1: 权重驱动聚焦 (高PR, 高top10, 高ln2_mean)
        # 模式2: 动态驱动聚焦 (低PR, 低top10, 高norm_ratio)
        # 模式3: 混合型
        # 模式4: 非聚焦 (低PR, 低top10, 低ln2_mean)
        
        # 简化分类: 基于PR和top10_energy
        if PR > 0.3 and top10_energy > 0.3:
            mode = "weight_driven"  # 权重驱动
        elif PR < 0.15 and top10_energy < 0.15:
            mode = "dynamic_driven"  # 动态驱动
        elif PR > 0.2 and top10_energy > 0.2:
            mode = "mixed"  # 混合型
        else:
            mode = "low_focus"  # 低聚焦
        
        result = {
            'layer': l_idx,
            'layer_frac': layer_frac,
            'alpha_attribution': alpha_attr,
            'mean_gamma': mean_gamma,
            'PR': PR,
            'd_eff': d_eff,
            'entropy_norm': entropy_norm,
            'd_intrinsic_norm': d_intrinsic_norm,
            'alpha_spectral': alpha_spectral,
            'kappa': kappa,
            'top10_energy': top10_energy,
            'norm_ratio': norm_ratio,
            'ln1_mean': ln1_mean,
            'ln2_mean': ln2_mean,
            'mode': mode,
        }
        results.append(result)
    
    # 统计各模式的比例
    mode_counts = {}
    for r in results:
        mode = r['mode']
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    
    print(f"\n  === 聚焦模式分布 ===")
    for mode, count in sorted(mode_counts.items()):
        frac = count / len(results)
        # 各模式的平均特征
        mode_results = [r for r in results if r['mode'] == mode]
        avg_PR = np.mean([r['PR'] for r in mode_results])
        avg_top10 = np.mean([r['top10_energy'] for r in mode_results])
        avg_ln2 = np.mean([r['ln2_mean'] for r in mode_results])
        print(f"  {mode}: {count}层({frac:.1%}), avg_PR={avg_PR:.4f}, "
              f"avg_top10={avg_top10:.4f}, avg_ln2={avg_ln2:.4f}")
    
    # 各模式的层分布
    print(f"\n  === 各模式的层分布 ===")
    for mode in ['weight_driven', 'mixed', 'dynamic_driven', 'low_focus']:
        mode_layers = [r['layer_frac'] for r in results if r['mode'] == mode]
        if mode_layers:
            print(f"  {mode}: layer_frac范围=[{min(mode_layers):.3f}, {max(mode_layers):.3f}], "
                  f"mean={np.mean(mode_layers):.3f}")
    
    # alpha在不同模式中的分布
    valid_results = [r for r in results if not np.isnan(r['alpha_attribution'])]
    if valid_results:
        print(f"\n  === Alpha在不同模式中的分布 ===")
        for mode in ['weight_driven', 'mixed', 'dynamic_driven', 'low_focus']:
            mode_alphas = [r['alpha_attribution'] for r in valid_results if r['mode'] == mode]
            if mode_alphas:
                print(f"  {mode}: mean_alpha={np.mean(mode_alphas):.4f}, "
                      f"std_alpha={np.std(mode_alphas):.4f}")
    
    # 保存结果
    out_path = RESULT_DIR / f"p487_{model_name}.txt"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"P487: 聚焦模式分类理论 - {model_name}\n")
        f.write(f"{'='*60}\n\n")
        
        f.write(f"=== 模式分布 ===\n")
        for mode, count in sorted(mode_counts.items()):
            frac = count / len(results)
            f.write(f"{mode}: {count}层({frac:.1%})\n")
        
        f.write(f"\n=== 逐层详情 ===\n")
        for r in results:
            f.write(f"Layer {r['layer']:3d} (frac={r['layer_frac']:.3f}): "
                    f"mode={r['mode']}, PR={r['PR']:.4f}, top10={r['top10_energy']:.4f}, "
                    f"alpha={r['alpha_attribution'] if not np.isnan(r['alpha_attribution']) else 'N/A'}, "
                    f"ln2={r['ln2_mean']:.4f}\n")
    
    print(f"\n  结果已保存到 {out_path}")
    
    release_model(model)
    return results


# ============================================================
# 主函数
# ============================================================
EXPERIMENTS = {
    'p484': run_p484,
    'p485': run_p485,
    'p486': run_p486,
    'p487': run_p487,
}

MODELS = ['qwen3', 'glm4', 'deepseek7b']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase CI: 因果链统一框架')
    parser.add_argument('--model', type=str, required=True, choices=MODELS,
                       help='模型名称')
    parser.add_argument('--experiment', type=str, required=True, choices=EXPERIMENTS.keys(),
                       help='实验编号 (p484/p485/p486/p487)')
    
    args = parser.parse_args()
    
    import torch  # 延迟import, 避免argparse前加载
    
    EXPERIMENTS[args.experiment](args.model)
