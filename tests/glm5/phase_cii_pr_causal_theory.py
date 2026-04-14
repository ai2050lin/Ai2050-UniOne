"""
Phase CII-P488/489/490/491: 信号聚焦的统一理论与因果机制
======================================================================

核心目标: 基于PR建立统一信号聚焦理论, 验证PR的因果性

Phase CI核心成果:
1. Alpha_attribution不是通用指标(DS7B中几乎为0)
2. Gamma层位置趋势模型相关(Qwen3正/GLM4-DS7B负)
3. PR(参与比)是跨模型最一致的gamma预测因子(mean_corr=0.64)
4. Fisher信息量vs alpha跨模型方向一致

关键问题:
1. PR是否因果影响gamma? (修改PR→gamma变化?)
2. PR由什么决定? (架构参数? 训练动态?)
3. 基于信息几何的alpha_new能否替代alpha_attribution?
4. PR→gamma→语言能力的完整因果链是否存在?

Phase CII目标:
1. PR的因果干预实验: 修改W_down的谱结构使PR变化
2. PR的架构决定因素: d/n_L, FFN_ratio, 训练残留
3. 基于PR的新alpha定义: alpha_PR
4. 完整因果链验证: PR→gamma→语言能力

P488: PR的因果干预实验
  - 目标: 验证PR是否因果影响gamma
  - 方法:
    a) 对W_down做SVD: W = U diag(s) V^T
    b) 修改谱: s' = s^p (p>1使谱更尖锐→PR减小, p<1使谱更平坦→PR增大)
    c) 重建W': W' = U diag(s') V^T
    d) 用W'替代W_down, 前向传播, 计算gamma
    e) 验证: PR增大→gamma是否增大?
  - 关键: 这是因果干预, 不是相关性分析!

P489: PR的架构决定因素
  - 目标: PR由什么决定?
  - 方法:
    a) 收集所有层的PR值
    b) 分析PR与架构参数(d/n_L, FFN_ratio)的关系
    c) 分析PR与训练残留(norm_ratio, s_max_ratio)的关系
    d) 分析PR与层位置(layer_frac)的关系
    e) 多变量回归: PR = f(架构, 训练, 层位置)

P490: 基于PR的新Alpha定义
  - 目标: 建立基于PR的alpha_new, 替代alpha_attribution
  - 方法:
    a) alpha_PR = PR(W_down) 的归一化版本
    b) alpha_info = Fisher_I / d_eff (信息密度)
    c) alpha_combined = w1*PR + w2*alpha_info (组合)
    d) 比较: alpha_new vs gamma的相关性是否跨模型一致?
    e) 验证: alpha_new是否比alpha_attribution更通用?

P491: 完整因果链验证
  - 目标: 验证 PR → gamma → 语言能力 的因果链
  - 方法:
    a) PR → gamma: P488的因果干预结果
    b) gamma → 语言能力: 用PPL(perplexity)衡量
    c) 中介分析: PR → gamma → PPL, gamma是否中介PR对PPL的效应?
    d) 比较三模型: 因果链的强度是否与模型能力相关?
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
                         get_W_U)

# 结果输出目录
RESULT_DIR = Path(__file__).parent.parent / "glm5_temp"
RESULT_DIR.mkdir(exist_ok=True)


def truncated_svd(matrix, k_max=400):
    """截断SVD, 避免大矩阵内存问题"""
    k = min(k_max, min(matrix.shape) - 1)
    if matrix.shape[0] > 50000 or matrix.shape[1] > 50000:
        k = min(k, 200)
    if matrix.shape[0] > 100000 or matrix.shape[1] > 100000:
        k = min(k, 100)
    try:
        from sklearn.utils.extmath import randomized_svd
        U, s, Vt = randomized_svd(matrix.astype(np.float32), n_components=k, random_state=42)
        return U, s, Vt
    except (ImportError, MemoryError):
        k = min(k, 100)
        try:
            from sklearn.utils.extmath import randomized_svd
            U, s, Vt = randomized_svd(matrix.astype(np.float32), n_components=k, random_state=42)
            return U, s, Vt
        except:
            U, s, Vt = svd(matrix[:min(5000, matrix.shape[0])], full_matrices=False)
            return U[:, :k], s[:k], Vt[:k, :]


def safe_WUT_svd(W_U, k_max=400):
    """安全的W_U^T SVD"""
    d_model = W_U.shape[1]
    vocab_size = W_U.shape[0]
    
    if vocab_size > 50000:
        k = min(k_max, vocab_size, d_model)
        np.random.seed(42)
        indices = np.random.choice(vocab_size, min(k * 3, vocab_size), replace=False)
        W_sub = W_U[indices]
        Q, R = np.linalg.qr(W_sub.T)
        U_wut = Q[:, :k].T
        s_wut = np.abs(np.diag(R[:k, :k]))
        if len(s_wut) < k:
            s_wut = np.pad(s_wut, (0, k - len(s_wut)))
        return U_wut, s_wut
    else:
        W_UT = W_U.T
        k = min(k_max, min(W_UT.shape) - 1)
        U, s, Vt = truncated_svd(W_UT.astype(np.float32), k)
        return U.T, s


def compute_participation_ratio(s):
    """参与比 PR = (sum sigma_i^2)^2 / (n * sum sigma_i^4)"""
    s2 = s**2
    s4 = s**4
    n = len(s)
    PR = np.sum(s2)**2 / (n * np.sum(s4)) if np.sum(s4) > 0 else 0
    return PR


def compute_effective_dimension(s):
    """有效维度 d_eff = (sum sigma_i)^2 / sum sigma_i^2"""
    s2 = s**2
    d_eff = np.sum(s)**2 / np.sum(s2) if np.sum(s2) > 0 else 0
    return d_eff


def compute_gamma_for_layer(model, tokenizer, device, layer_idx, W_U):
    """计算某层的gamma (delta_h在W_U空间的能量占比)"""
    base_text = "The apple is"
    attr_words = ["red", "green", "big", "small", "sweet", "sour"]
    
    import torch
    
    # W_U的投影矩阵
    U_wut, s_wut = safe_WUT_svd(W_U, 200)
    
    base_ids = tokenizer.encode(base_text, return_tensors="pt").to(device)
    with torch.no_grad():
        base_out = model(base_ids, output_hidden_states=True)
        base_h = base_out.hidden_states[layer_idx + 1][0, -1].cpu().float().numpy()
    
    gamma_list = []
    for attr_word in attr_words:
        intervened_text = f"The {attr_word} apple is"
        interv_ids = tokenizer.encode(intervened_text, return_tensors="pt").to(device)
        with torch.no_grad():
            interv_out = model(interv_ids, output_hidden_states=True)
            interv_h = interv_out.hidden_states[layer_idx + 1][0, -1].cpu().float().numpy()
        
        delta_h = interv_h - base_h
        proj = U_wut @ delta_h
        energy_proj = np.sum(proj**2)
        energy_total = np.sum(delta_h**2)
        gamma = energy_proj / energy_total if energy_total > 0 else 0
        gamma_list.append(gamma)
    
    return np.mean(gamma_list), np.std(gamma_list)


# ============================================================
# P488: PR的因果干预实验
# ============================================================
def run_p488(model_name):
    import torch
    
    print(f"\n{'='*70}")
    print(f"P488: PR的因果干预实验 - {model_name}")
    print(f"{'='*70}")
    print(f"  目标: 验证PR是否因果影响gamma")
    print(f"  方法: 修改W_down的谱结构使PR变化, 测量gamma变化")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers = info.n_layers
    d_model = info.d_model
    
    print(f"\n  模型: {model_name}, n_L={n_layers}, d={d_model}")
    
    W_U = get_W_U(model)
    
    # 选3个目标层
    target_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]
    print(f"  目标层: {target_layers}")
    
    # 谱修改指数: p<1使谱平坦(PR增大), p>1使谱尖锐(PR减小)
    p_values = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]
    
    results = []
    for l_idx in target_layers:
        print(f"\n  --- Layer {l_idx}/{n_layers-1} ---")
        
        # 获取W_down
        lw = get_layer_weights(layers[l_idx], d_model, info.mlp_type)
        W_down_orig = lw.W_down.copy()
        
        # W_down SVD
        U_wd, s_wd, Vt_wd = truncated_svd(W_down_orig, 300)
        
        # 原始PR
        PR_orig = compute_participation_ratio(s_wd)
        print(f"    原始PR={PR_orig:.4f}")
        
        # 计算原始gamma
        gamma_orig, gamma_std = compute_gamma_for_layer(model, tokenizer, device, l_idx, W_U)
        print(f"    原始gamma={gamma_orig:.4f}")
        
        # 对每个p值修改谱
        layer_results = []
        for p in p_values:
            # 修改谱: s' = s^p (保持符号为正, 因为s是奇异值)
            s_modified = s_wd ** p
            # 归一化使Frobenius范数不变
            norm_orig = np.linalg.norm(W_down_orig, 'fro')
            W_modified = U_wd @ np.diag(s_modified) @ Vt_wd
            norm_modified = np.linalg.norm(W_modified, 'fro')
            if norm_modified > 0:
                W_modified = W_modified * (norm_orig / norm_modified)
            
            # 计算修改后的PR
            PR_modified = compute_participation_ratio(s_modified)
            
            # 修改W_down权重
            # 匹配原始权重的dtype
            orig_dtype = layers[l_idx].mlp.down_proj.weight.dtype
            W_down_tensor = torch.tensor(W_modified, dtype=orig_dtype, device=device)
            
            # 临时替换W_down
            if info.mlp_type == "split_gate_up":
                orig_weight = layers[l_idx].mlp.down_proj.weight.data.clone()
                layers[l_idx].mlp.down_proj.weight.data = W_down_tensor
            elif info.mlp_type == "merged_gate_up":
                orig_weight = layers[l_idx].mlp.down_proj.weight.data.clone()
                layers[l_idx].mlp.down_proj.weight.data = W_down_tensor
            
            # 计算修改后的gamma
            gamma_modified, _ = compute_gamma_for_layer(model, tokenizer, device, l_idx, W_U)
            
            # 恢复原始权重
            if info.mlp_type == "split_gate_up":
                layers[l_idx].mlp.down_proj.weight.data = orig_weight
            elif info.mlp_type == "merged_gate_up":
                layers[l_idx].mlp.down_proj.weight.data = orig_weight
            
            delta_PR = PR_modified - PR_orig
            delta_gamma = gamma_modified - gamma_orig
            sensitivity = delta_gamma / delta_PR if abs(delta_PR) > 1e-8 else 0
            
            result = {
                'layer': l_idx,
                'p': p,
                'PR_orig': PR_orig,
                'PR_modified': PR_modified,
                'delta_PR': delta_PR,
                'gamma_orig': gamma_orig,
                'gamma_modified': gamma_modified,
                'delta_gamma': delta_gamma,
                'sensitivity': sensitivity,
            }
            layer_results.append(result)
            results.append(result)
            
            print(f"    p={p:.1f}: PR={PR_modified:.4f}(delta={delta_PR:+.4f}), "
                  f"gamma={gamma_modified:.4f}(delta={delta_gamma:+.4f}), "
                  f"sensitivity={sensitivity:.4f}")
        
        # 计算PR-gamma因果敏感性
        PRs = [r['PR_modified'] for r in layer_results]
        gammas = [r['gamma_modified'] for r in layer_results]
        if len(PRs) > 2 and np.std(PRs) > 0 and np.std(gammas) > 0:
            corr, _ = pearsonr(PRs, gammas)
            print(f"    ** PR-gamma因果corr = {corr:.3f} **")
    
    # 汇总分析
    print(f"\n  === P488 汇总 ===")
    all_PRs = [r['PR_modified'] for r in results]
    all_gammas = [r['gamma_modified'] for r in results]
    all_sensitivities = [r['sensitivity'] for r in results if abs(r['delta_PR']) > 1e-6]
    
    if len(all_PRs) > 2:
        corr, _ = pearsonr(all_PRs, all_gammas)
        print(f"  总体PR-gamma因果corr = {corr:.3f}")
    
    if all_sensitivities:
        print(f"  平均sensitivity = {np.mean(all_sensitivities):.4f}")
        positive_sens = [s for s in all_sensitivities if s > 0]
        negative_sens = [s for s in all_sensitivities if s < 0]
        print(f"  正sensitivity: {len(positive_sens)}/{len(all_sensitivities)} "
              f"({len(positive_sens)/len(all_sensitivities)*100:.0f}%)")
        print(f"  负sensitivity: {len(negative_sens)}/{len(all_sensitivities)} "
              f"({len(negative_sens)/len(all_sensitivities)*100:.0f}%)")
    
    # 保存
    out_path = RESULT_DIR / f"p488_{model_name}.txt"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"P488: PR因果干预 - {model_name}\n")
        f.write(f"{'='*60}\n\n")
        for r in results:
            f.write(f"Layer {r['layer']:3d}, p={r['p']:.1f}: "
                    f"PR={r['PR_modified']:.4f}(delta={r['delta_PR']:+.4f}), "
                    f"gamma={r['gamma_modified']:.4f}(delta={r['delta_gamma']:+.4f}), "
                    f"sens={r['sensitivity']:.4f}\n")
    
    print(f"\n  结果已保存到 {out_path}")
    
    release_model(model)
    return results


# ============================================================
# P489: PR的架构决定因素
# ============================================================
def run_p489(model_name):
    import torch
    
    print(f"\n{'='*70}")
    print(f"P489: PR的架构决定因素 - {model_name}")
    print(f"{'='*70}")
    print(f"  目标: PR由什么决定? 架构参数? 训练残留? 层位置?")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers = info.n_layers
    d_model = info.d_model
    intermediate_size = info.intermediate_size
    
    print(f"\n  模型: {model_name}, n_L={n_layers}, d={d_model}, intermediate={intermediate_size}")
    
    FFN_ratio = intermediate_size / d_model
    d_over_nL = d_model / n_layers
    
    # 所有层
    results = []
    for l_idx in range(n_layers):
        layer_frac = l_idx / max(n_layers - 1, 1)
        
        lw = get_layer_weights(layers[l_idx], d_model, info.mlp_type)
        W_down = lw.W_down
        
        # SVD
        U_wd, s_wd, Vt_wd = truncated_svd(W_down, 300)
        
        # PR
        PR = compute_participation_ratio(s_wd)
        
        # 有效维度
        d_eff = compute_effective_dimension(s_wd)
        
        # 条件数
        kappa = s_wd[0] / s_wd[-1] if s_wd[-1] > 0 else float('inf')
        
        # 训练残留
        m, n = W_down.shape
        sigma_init = 1.0 / np.sqrt(n)
        norm_ratio = np.linalg.norm(W_down, 'fro') / (np.sqrt(m * n) * sigma_init)
        s_max_ratio = s_wd[0] / (sigma_init * (np.sqrt(m) + np.sqrt(n)))
        
        # top10能量占比
        top10_energy = np.sum(s_wd[:10]**2) / np.sum(s_wd**2)
        
        # 谱熵
        p = s_wd**2 / np.sum(s_wd**2)
        entropy = -np.sum(p * np.log(p + 1e-30))
        entropy_norm = entropy / np.log(len(s_wd))
        
        # LN权重
        input_ln_w = lw.input_layernorm_weight
        post_ln_w = lw.post_attn_layernorm_weight
        ln1_mean = np.mean(input_ln_w) if input_ln_w is not None else 0
        ln2_mean = np.mean(post_ln_w) if post_ln_w is not None else 0
        
        # 注意力权重范数
        W_q_norm = np.linalg.norm(lw.W_q, 'fro')
        W_k_norm = np.linalg.norm(lw.W_k, 'fro')
        W_v_norm = np.linalg.norm(lw.W_v, 'fro')
        W_o_norm = np.linalg.norm(lw.W_o, 'fro')
        attn_norm = W_q_norm + W_k_norm + W_v_norm + W_o_norm
        
        result = {
            'layer': l_idx,
            'layer_frac': layer_frac,
            'PR': PR,
            'd_eff': d_eff,
            'kappa': kappa,
            'norm_ratio': norm_ratio,
            's_max_ratio': s_max_ratio,
            'top10_energy': top10_energy,
            'entropy_norm': entropy_norm,
            'ln1_mean': ln1_mean,
            'ln2_mean': ln2_mean,
            'attn_norm': attn_norm,
        }
        results.append(result)
    
    # 分析PR的决定因素
    print(f"\n  === PR与各特征的相关性 ===")
    PR_list = [r['PR'] for r in results]
    
    features = ['layer_frac', 'd_eff', 'kappa', 'norm_ratio', 's_max_ratio',
                'top10_energy', 'entropy_norm', 'ln1_mean', 'ln2_mean', 'attn_norm']
    
    for feat in features:
        feat_data = [r[feat] for r in results]
        if np.std(feat_data) > 0 and np.std(PR_list) > 0:
            try:
                corr, p = pearsonr(feat_data, PR_list)
                print(f"  PR vs {feat}: corr={corr:.3f}, p={p:.4f}")
            except:
                print(f"  PR vs {feat}: 计算失败")
    
    # 多变量回归
    print(f"\n  === 多变量回归: PR = f(features) ===")
    from numpy.polynomial import polynomial as P
    
    # 选最佳特征组合
    feature_data = {}
    for feat in features:
        vals = [r[feat] for r in results]
        if np.std(vals) > 0:
            feature_data[feat] = np.array(vals)
    
    PR_arr = np.array(PR_list)
    
    # 逐步回归
    selected = []
    remaining = list(feature_data.keys())
    best_R2 = 0
    
    for step in range(min(5, len(remaining))):
        best_feat = None
        best_step_R2 = best_R2
        
        for feat in remaining:
            X_cols = [feature_data[f] for f in selected] + [feature_data[feat]]
            X = np.column_stack(X_cols)
            try:
                coeffs = np.linalg.lstsq(X, PR_arr, rcond=None)[0]
                pred = X @ coeffs
                ss_res = np.sum((PR_arr - pred)**2)
                ss_tot = np.sum((PR_arr - np.mean(PR_arr))**2)
                R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                n = len(PR_arr)
                k = X.shape[1]
                R2_adj = 1 - (1 - R2) * (n - 1) / (n - k - 1) if n - k - 1 > 0 else 0
                
                if R2_adj > best_step_R2:
                    best_step_R2 = R2_adj
                    best_feat = feat
                    best_step_coeffs = coeffs
                    best_step_R2_raw = R2
            except:
                pass
        
        if best_feat is not None:
            selected.append(best_feat)
            remaining.remove(best_feat)
            best_R2 = best_step_R2
            print(f"  Step {step+1}: add {best_feat}, R2={best_step_R2_raw:.4f}, "
                  f"R2_adj={best_step_R2:.4f}, features={selected}")
        else:
            break
    
    # 保存
    out_path = RESULT_DIR / f"p489_{model_name}.txt"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"P489: PR架构决定因素 - {model_name}\n")
        f.write(f"{'='*60}\n")
        f.write(f"d={d_model}, n_L={n_layers}, FFN_ratio={FFN_ratio:.3f}, d/n_L={d_over_nL:.1f}\n\n")
        
        f.write("=== 逐步回归结果 ===\n")
        f.write(f"最佳特征: {selected}\n")
        f.write(f"R2_adj = {best_R2:.4f}\n\n")
        
        f.write("=== 逐层PR ===\n")
        for r in results:
            f.write(f"Layer {r['layer']:3d}: PR={r['PR']:.4f}, d_eff={r['d_eff']:.1f}, "
                    f"kappa={r['kappa']:.1f}, norm_ratio={r['norm_ratio']:.4f}, "
                    f"ln2={r['ln2_mean']:.4f}\n")
    
    print(f"\n  结果已保存到 {out_path}")
    
    release_model(model)
    return results


# ============================================================
# P490: 基于PR的新Alpha定义
# ============================================================
def run_p490(model_name):
    import torch
    
    print(f"\n{'='*70}")
    print(f"P490: 基于PR的新Alpha定义 - {model_name}")
    print(f"{'='*70}")
    print(f"  目标: 建立基于PR和信息几何的新alpha, 替代alpha_attribution")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers = info.n_layers
    d_model = info.d_model
    
    print(f"\n  模型: {model_name}, n_L={n_layers}, d={d_model}")
    
    W_U = get_W_U(model)
    U_wut, s_wut = safe_WUT_svd(W_U, 400)
    
    # 采样层
    sample_layers = get_sample_layers(n_layers, 10)
    print(f"  采样层: {sample_layers}")
    
    results = []
    for l_idx in sample_layers:
        print(f"\n  --- Layer {l_idx}/{n_layers-1} ---")
        layer_frac = l_idx / max(n_layers - 1, 1)
        
        lw = get_layer_weights(layers[l_idx], d_model, info.mlp_type)
        W_down = lw.W_down
        
        # W_down SVD
        U_wd, s_wd, Vt_wd = truncated_svd(W_down, 300)
        
        # === 旧alpha: alpha_attribution ===
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
        
        # alpha_attribution: 投影密度的幂律指数
        alpha_attr_list = []
        for delta_h in delta_h_list:
            proj = U_wut @ delta_h
            weights = s_wut[:len(proj)]
            valid = (weights > 0) & (np.abs(proj) > 0)
            if np.sum(valid) > 5:
                log_w = np.log(weights[valid])
                log_p = np.log(np.abs(proj[valid]))
                try:
                    from numpy.polynomial import polynomial as P
                    coeffs = P.polyfit(log_w, log_p, 1)
                    alpha_attr_list.append(coeffs[1])
                except:
                    alpha_attr_list.append(0)
        alpha_attr = np.mean(alpha_attr_list) if alpha_attr_list else 0
        
        # === 新alpha定义 ===
        
        # 1. alpha_PR = 归一化PR
        PR = compute_participation_ratio(s_wd)
        alpha_PR = PR  # PR本身在[0,1]范围
        
        # 2. alpha_info = 信息密度 = Fisher / d_eff
        m, n = W_down.shape
        sigma_init = 1.0 / np.sqrt(n)
        gamma_ratio = m / n
        lambda_max = sigma_init**2 * (1 + np.sqrt(gamma_ratio))**2
        lambda_min = sigma_init**2 * (1 - np.sqrt(gamma_ratio))**2 if gamma_ratio < 1 else 0
        mp_s = np.linspace(np.sqrt(lambda_max), np.sqrt(max(lambda_min, 0)), len(s_wd))
        fisher_I = np.sum((s_wd - mp_s)**2 / (mp_s**2 + 1e-30))
        d_eff = compute_effective_dimension(s_wd)
        alpha_info = fisher_I / d_eff if d_eff > 0 else 0
        
        # 3. alpha_entropy = 1 - entropy_norm (越集中越大)
        p = s_wd**2 / np.sum(s_wd**2)
        entropy = -np.sum(p * np.log(p + 1e-30))
        max_entropy = np.log(len(s_wd))
        entropy_norm = entropy / max_entropy if max_entropy > 0 else 1
        alpha_entropy = 1 - entropy_norm  # 反转: 越集中越大
        
        # 4. alpha_spectral = W_down谱的幂律指数
        valid = s_wd > 0
        indices = np.arange(1, len(s_wd) + 1)[valid]
        log_idx = np.log(indices)
        log_s = np.log(s_wd[valid])
        try:
            from numpy.polynomial import polynomial as P
            coeffs = P.polyfit(log_idx, log_s, 1)
            alpha_spectral = -coeffs[1]
        except:
            alpha_spectral = 0
        
        # === Gamma ===
        proj_list = [U_wut @ dh for dh in delta_h_list]
        proj_energies = [np.sum(p**2) for p in proj_list]
        delta_energies = [np.sum(dh**2) for dh in delta_h_list]
        gamma_list = [pe / de if de > 0 else 0 for pe, de in zip(proj_energies, delta_energies)]
        mean_gamma = np.mean(gamma_list)
        
        # LN权重
        input_ln_w = lw.input_layernorm_weight
        post_ln_w = lw.post_attn_layernorm_weight
        ln2_mean = np.mean(post_ln_w) if post_ln_w is not None else 0
        
        result = {
            'layer': l_idx,
            'layer_frac': layer_frac,
            'alpha_attribution': alpha_attr,
            'alpha_PR': alpha_PR,
            'alpha_info': alpha_info,
            'alpha_entropy': alpha_entropy,
            'alpha_spectral': alpha_spectral,
            'mean_gamma': mean_gamma,
            'PR': PR,
            'd_eff': d_eff,
            'ln2_mean': ln2_mean,
        }
        results.append(result)
        print(f"    alpha_attr={alpha_attr:.4f}, alpha_PR={alpha_PR:.4f}, "
              f"alpha_info={alpha_info:.2f}, alpha_entropy={alpha_entropy:.4f}, "
              f"gamma={mean_gamma:.4f}")
    
    # 分析各alpha与gamma的相关性
    print(f"\n  === 各Alpha与Gamma的相关性 ===")
    gamma_list = [r['mean_gamma'] for r in results]
    alpha_names = ['alpha_attribution', 'alpha_PR', 'alpha_info', 'alpha_entropy', 'alpha_spectral']
    
    for alpha_name in alpha_names:
        alpha_data = [r[alpha_name] for r in results]
        if np.std(alpha_data) > 0 and np.std(gamma_list) > 0:
            corr, p = pearsonr(alpha_data, gamma_list)
            print(f"  {alpha_name} vs gamma: corr={corr:.3f}, p={p:.4f}")
    
    # 各alpha与layer_frac的关系
    print(f"\n  === 各Alpha与层位置的关系 ===")
    layer_fracs = [r['layer_frac'] for r in results]
    for alpha_name in alpha_names:
        alpha_data = [r[alpha_name] for r in results]
        if np.std(alpha_data) > 0:
            corr, _ = pearsonr(alpha_data, layer_fracs)
            print(f"  {alpha_name} vs layer_frac: corr={corr:.3f}")
    
    # 保存
    out_path = RESULT_DIR / f"p490_{model_name}.txt"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"P490: 基于PR的新Alpha定义 - {model_name}\n")
        f.write(f"{'='*60}\n\n")
        for r in results:
            f.write(f"Layer {r['layer']:3d} (frac={r['layer_frac']:.3f}): "
                    f"alpha_attr={r['alpha_attribution']:.4f}, "
                    f"alpha_PR={r['alpha_PR']:.4f}, "
                    f"alpha_info={r['alpha_info']:.2f}, "
                    f"alpha_entropy={r['alpha_entropy']:.4f}, "
                    f"gamma={r['mean_gamma']:.4f}\n")
    
    print(f"\n  结果已保存到 {out_path}")
    
    release_model(model)
    return results


# ============================================================
# P491: 完整因果链验证
# ============================================================
def run_p491(model_name):
    import torch
    
    print(f"\n{'='*70}")
    print(f"P491: 完整因果链验证 - {model_name}")
    print(f"{'='*70}")
    print(f"  目标: 验证 PR -> gamma -> PPL 的因果链")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers = info.n_layers
    d_model = info.d_model
    
    print(f"\n  模型: {model_name}, n_L={n_layers}, d={d_model}")
    
    W_U = get_W_U(model)
    
    # 采样层
    sample_layers = get_sample_layers(n_layers, 8)
    print(f"  采样层: {sample_layers}")
    
    # 计算PPL的测试文本
    test_texts = [
        "The apple is red and sweet.",
        "The sky is blue and clear today.",
        "The cat sat on the warm mat.",
        "Science advances through careful observation.",
        "Language models process text using transformers.",
    ]
    
    results = []
    for l_idx in sample_layers:
        print(f"\n  --- Layer {l_idx}/{n_layers-1} ---")
        layer_frac = l_idx / max(n_layers - 1, 1)
        
        lw = get_layer_weights(layers[l_idx], d_model, info.mlp_type)
        W_down = lw.W_down
        
        # W_down SVD
        U_wd, s_wd, Vt_wd = truncated_svd(W_down, 300)
        
        # PR
        PR = compute_participation_ratio(s_wd)
        d_eff = compute_effective_dimension(s_wd)
        
        # gamma
        gamma_mean, _ = compute_gamma_for_layer(model, tokenizer, device, l_idx, W_U)
        
        # 层对PPL的影响: 移除该层后的PPL变化
        # 方法: 将该层W_down置零, 测量PPL
        ppl_original = 0
        ppl_ablated = 0
        
        with torch.no_grad():
            for text in test_texts:
                ids = tokenizer.encode(text, return_tensors="pt").to(device)
                
                # 原始PPL
                out = model(ids, labels=ids)
                ppl_original += out.loss.item()
                
                # 置零该层W_down
                orig_weight = layers[l_idx].mlp.down_proj.weight.data.clone()
                layers[l_idx].mlp.down_proj.weight.data.zero_()
                
                out_abl = model(ids, labels=ids)
                ppl_ablated += out_abl.loss.item()
                
                # 恢复
                layers[l_idx].mlp.down_proj.weight.data = orig_weight
        
        ppl_original /= len(test_texts)
        ppl_ablated /= len(test_texts)
        delta_ppl = ppl_ablated - ppl_original  # 正值=该层重要
        
        # 层重要性 = delta_ppl / ppl_original
        layer_importance = delta_ppl / ppl_original if ppl_original > 0 else 0
        
        result = {
            'layer': l_idx,
            'layer_frac': layer_frac,
            'PR': PR,
            'd_eff': d_eff,
            'gamma': gamma_mean,
            'ppl_original': ppl_original,
            'ppl_ablated': ppl_ablated,
            'delta_ppl': delta_ppl,
            'layer_importance': layer_importance,
        }
        results.append(result)
        print(f"    PR={PR:.4f}, gamma={gamma_mean:.4f}, "
              f"PPL_orig={ppl_original:.2f}, PPL_abl={ppl_ablated:.2f}, "
              f"delta_PPL={delta_ppl:+.2f}, importance={layer_importance:.4f}")
    
    # 分析因果链
    print(f"\n  === 因果链分析 ===")
    PR_list = [r['PR'] for r in results]
    gamma_list = [r['gamma'] for r in results]
    importance_list = [r['layer_importance'] for r in results]
    
    # PR → gamma
    if np.std(PR_list) > 0 and np.std(gamma_list) > 0:
        corr, p = pearsonr(PR_list, gamma_list)
        print(f"  PR -> gamma: corr={corr:.3f}, p={p:.4f}")
    
    # gamma → importance
    if np.std(gamma_list) > 0 and np.std(importance_list) > 0:
        corr, p = pearsonr(gamma_list, importance_list)
        print(f"  gamma -> importance: corr={corr:.3f}, p={p:.4f}")
    
    # PR → importance (直接效应)
    if np.std(PR_list) > 0 and np.std(importance_list) > 0:
        corr, p = pearsonr(PR_list, importance_list)
        print(f"  PR -> importance: corr={corr:.3f}, p={p:.4f}")
    
    # 中介分析: PR -> gamma -> importance
    # 总效应 = PR -> importance
    # 直接效应 = PR -> importance (控制gamma后)
    # 间接效应 = PR -> gamma -> importance
    from numpy.polynomial import polynomial as P
    
    # Step 1: PR -> importance (总效应)
    X_pr = np.array(PR_list)
    Y_imp = np.array(importance_list)
    coeffs_total = np.linalg.lstsq(np.column_stack([X_pr, np.ones(len(X_pr))]), Y_imp, rcond=None)[0]
    total_effect = coeffs_total[0]
    
    # Step 2: PR -> gamma
    M_gamma = np.array(gamma_list)
    coeffs_pr_gamma = np.linalg.lstsq(np.column_stack([X_pr, np.ones(len(X_pr))]), M_gamma, rcond=None)[0]
    a_path = coeffs_pr_gamma[0]
    
    # Step 3: PR + gamma -> importance
    X_both = np.column_stack([X_pr, M_gamma, np.ones(len(X_pr))])
    coeffs_both = np.linalg.lstsq(X_both, Y_imp, rcond=None)[0]
    direct_effect = coeffs_both[0]  # PR的直接效应
    b_path = coeffs_both[1]  # gamma的效应
    
    indirect_effect = a_path * b_path  # 间接效应 = a * b
    
    print(f"\n  === 中介分析: PR -> gamma -> importance ===")
    print(f"  总效应(PR->importance): {total_effect:.4f}")
    print(f"  a路径(PR->gamma): {a_path:.4f}")
    print(f"  b路径(gamma->importance): {b_path:.4f}")
    print(f"  直接效应(PR->importance, 控制gamma): {direct_effect:.4f}")
    print(f"  间接效应(PR->gamma->importance): {indirect_effect:.4f}")
    print(f"  中介比例: {abs(indirect_effect)/(abs(total_effect)+1e-10)*100:.1f}%")
    
    # 保存
    out_path = RESULT_DIR / f"p491_{model_name}.txt"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"P491: 完整因果链验证 - {model_name}\n")
        f.write(f"{'='*60}\n\n")
        
        f.write("=== 中介分析 ===\n")
        f.write(f"总效应: {total_effect:.4f}\n")
        f.write(f"a路径(PR->gamma): {a_path:.4f}\n")
        f.write(f"b路径(gamma->importance): {b_path:.4f}\n")
        f.write(f"直接效应: {direct_effect:.4f}\n")
        f.write(f"间接效应: {indirect_effect:.4f}\n")
        f.write(f"中介比例: {abs(indirect_effect)/(abs(total_effect)+1e-10)*100:.1f}%\n\n")
        
        f.write("=== 逐层详情 ===\n")
        for r in results:
            f.write(f"Layer {r['layer']:3d}: PR={r['PR']:.4f}, gamma={r['gamma']:.4f}, "
                    f"importance={r['layer_importance']:.4f}, "
                    f"PPL_orig={r['ppl_original']:.2f}, PPL_abl={r['ppl_ablated']:.2f}\n")
    
    print(f"\n  结果已保存到 {out_path}")
    
    release_model(model)
    return results


# ============================================================
# 主函数
# ============================================================
EXPERIMENTS = {
    'p488': run_p488,
    'p489': run_p489,
    'p490': run_p490,
    'p491': run_p491,
}

MODELS = ['qwen3', 'glm4', 'deepseek7b']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase CII: 信号聚焦的统一理论与因果机制')
    parser.add_argument('--model', type=str, required=True, choices=MODELS,
                       help='模型名称')
    parser.add_argument('--experiment', type=str, required=True, choices=EXPERIMENTS.keys(),
                       help='实验编号 (p488/p489/p490/p491)')
    
    args = parser.parse_args()
    
    import torch
    
    EXPERIMENTS[args.experiment](args.model)
