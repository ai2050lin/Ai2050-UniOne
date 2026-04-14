"""
Phase CIII-P492/493/494/495: 信号聚焦的重新定义与语言能力机制
======================================================================

核心目标: 解决gamma→语言能力链断裂问题, 重新定义信号聚焦

Phase CII核心成果:
1. PR因果影响gamma但效应弱(delta<0.02)且方向层依赖
2. alpha_PR是跨模型最一致的gamma预测因子(三模型全正)
3. PR由训练后权重谱特征决定(R2>0.97)
4. gamma→语言能力(importance)链断裂(corr<0.05)
5. 第一层有极端重要性(importance>0.38)

关键问题:
1. gamma的定义是否有问题? (能量占比vs信息量)
2. "语言能力"指标是否合适? (PPL太粗糙)
3. 信号聚焦的本质是什么? (能量集中 vs 信息密度)
4. 为什么第一层如此重要?

Phase CIII目标:
1. 用信息论重新定义gamma: MI(h, W_U)替代能量占比
2. 分析gamma→PPL链断裂的原因: 是否是PPL的问题?
3. 研究层0的特殊功能: 第一层的信息编码机制
4. 寻找新的因果链: 从权重结构直接预测语言能力

P492: 信息论gamma - 用互信息替代能量占比
  - 目标: gamma_info = I(h_l; W_U) = H(h_l) - H(h_l|W_U)
  - 方法:
    a) 对每层的hidden state计算与W_U的互信息
    b) 用kNN估计法估计MI
    c) 比较gamma_info与gamma_energy的相关性
    d) 检验gamma_info是否与层重要性(importance)相关

P493: 语言能力的精细指标 - 不只是PPL
  - 目标: 寻找比PPL更敏感的语言能力指标
  - 方法:
    a) 层消融后计算多种指标: PPL, next-token准确率, top-5命中率
    b) 语义一致性: 消融后输出是否保持语义
    c) 语法正确性: 消融后输出是否保持语法
    d) 分析各指标与gamma/PR的关系

P494: 层0的特殊功能分析
  - 目标: 为什么第一层如此重要?
  - 方法:
    a) 分析L0→L1的非线性转换结构
    b) 比较L0与其他层的W_down谱特征
    c) L0的信号注入模式: 哪些方向被注入?
    d) L0的"初始化"角色: 是否为后续层建立了"工作空间"?

P495: 从权重结构直接预测语言能力
  - 目标: 跳过gamma, 直接从W_down谱特征预测层重要性
  - 方法:
    a) 收集所有层的谱特征(PR, top10, kappa, entropy_norm, etc.)
    b) 回归分析: importance ~ f(谱特征)
    c) 因果验证: 修改谱特征→importance变化?
    d) 跨模型验证: 同一公式是否跨模型有效?

使用方法:
    python phase_ciii_signal_refocus.py --model qwen3 --experiment p492
    python phase_ciii_signal_refocus.py --model glm4 --experiment p493
    python phase_ciii_signal_refocus.py --model deepseek7b --experiment p494
"""

import sys
import os
import argparse
import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from collections import namedtuple

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 直接添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, 'tests', 'glm5'))

from model_utils import (
    load_model, get_model_info, get_W_U, get_layer_weights,
    get_sample_layers, compute_recoding_ratio,
)

# 兼容性别名
def load_model_safely(model_name, device):
    model, tokenizer, model_device = load_model(model_name)
    if model_device != device:
        model = model.to(device)
    return model, tokenizer

def compute_residual_stream(model, input_ids, device):
    """简化版残差流计算 - 用hook收集各层hidden state"""
    hidden_states = []
    
    def hook_fn(module, input, output):
        # 捕获残差流(层归一化后的输出)
        if isinstance(output, tuple):
            hidden_states.append(output[0].detach())
        else:
            hidden_states.append(output.detach())
    
    # 注册hook到每层
    hooks = []
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for layer in model.model.layers:
            h = layer.register_forward_hook(hook_fn)
            hooks.append(h)
    
    try:
        with torch.no_grad():
            model(input_ids)
    except Exception as e:
        print(f"  [compute_residual_stream] Forward failed: {e}")
    
    for h in hooks:
        h.remove()
    
    if hidden_states:
        # 返回每层最后一个token的hidden state
        result = []
        for hs in hidden_states:
            result.append(hs[0, -1].cpu().float().numpy())
        return result
    return None

def compute_effective_dimension(s):
    """有效维度 = (sum s)^2 / (n * sum s^2)"""
    s_sq = s**2
    return (np.sum(s)**2) / (len(s) * np.sum(s_sq) + 1e-30)

def compute_participation_ratio(s):
    """参与比 = (sum s^2)^2 / (n * sum s^4)"""
    s_sq = s**2
    s_sq_norm = s_sq / (np.sum(s_sq) + 1e-30)
    return 1.0 / (len(s) * np.sum(s_sq_norm**2) + 1e-30)

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
        U, s, Vt = np.linalg.svd(W_UT.astype(np.float32), full_matrices=False)
        return U[:, :k].T, s[:k]

LayerWeights = namedtuple('LayerWeights', ['W_Q', 'W_K', 'W_V', 'W_O', 'W_up', 'W_gate', 'W_down'])


def compute_information_gamma(h_list, W_U, n_neighbors=5):
    """
    P492: 用互信息替代能量占比定义gamma
    
    gamma_info = I(h_l; W_U) 的kNN估计
    
    方法: 用W_U的行向量定义"语言子空间", 
    计算h_l与该子空间的互信息
    
    简化版本: 用h_l在W_U子空间的投影方差与总方差的比
    作为MI的近似(线性MI)
    """
    d_model = W_U.shape[1]
    
    # W_U子空间投影矩阵 (用QR避免大矩阵SVD)
    vocab_size = W_U.shape[0]
    k_proj = min(400, d_model, vocab_size)
    
    if vocab_size > 50000:
        # 大词汇表: 采样
        np.random.seed(42)
        indices = np.random.choice(vocab_size, min(k_proj * 3, vocab_size), replace=False)
        W_sub = W_U[indices]
    else:
        W_sub = W_U[:min(k_proj * 3, vocab_size)]
    
    Q, R = np.linalg.qr(W_sub.T)  # Q: [d_model, k_proj*3]
    P_lang = Q[:, :k_proj]  # 语言子空间投影基
    
    results = []
    for h in h_list:
        h_np = h.cpu().numpy() if torch.is_tensor(h) else h
        
        # 投影到语言子空间
        h_proj = P_lang.T @ h_np  # [k_proj]
        
        # 能量gamma (原始定义)
        energy_proj = np.sum(h_proj**2)
        energy_total = np.sum(h_np**2)
        gamma_energy = energy_proj / max(energy_total, 1e-10)
        
        # 信息gamma (基于投影系数的熵)
        # 高信息 = 投影系数分布均匀(使用所有维度)
        # 低信息 = 投影系数集中在少数维度
        h_proj_norm = np.abs(h_proj) / max(np.sum(np.abs(h_proj)), 1e-10)
        h_proj_norm = h_proj_norm[h_proj_norm > 1e-10]
        if len(h_proj_norm) > 0:
            entropy_proj = -np.sum(h_proj_norm * np.log(h_proj_norm))
            max_entropy = np.log(len(h_proj_norm))
            gamma_info = entropy_proj / max(max_entropy, 1e-10)
        else:
            gamma_info = 0.0
        
        # cos(h, W_U空间) - 方向对齐
        if energy_total > 1e-10:
            cos_lang = np.sqrt(energy_proj / energy_total)
        else:
            cos_lang = 0.0
        
        results.append({
            'gamma_energy': gamma_energy,
            'gamma_info': gamma_info,
            'cos_lang': cos_lang,
            'entropy_proj': entropy_proj if len(h_proj_norm) > 0 else 0,
        })
    
    return results


def compute_layer_importance_detailed(model, tokenizer, device, text, layers, l_idx, model_name):
    """
    P493: 计算层重要性的精细指标
    不只是PPL, 还包括next-token准确率、top-5命中率等
    """
    info = get_model_info(model, model_name)
    
    # 基线前向传播
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        baseline_outputs = model(input_ids)
        # 提取logits
        if hasattr(baseline_outputs, 'logits'):
            baseline_logits = baseline_outputs.logits
        elif isinstance(baseline_outputs, torch.Tensor):
            baseline_logits = baseline_outputs
        else:
            print("  无法提取logits")
            return None
    
    if baseline_logits.dim() == 3:
        baseline_logits = baseline_logits[0]
    
    # 基线指标
    baseline_next_tokens = input_ids[0, 1:]  # 正确的下一个token
    baseline_pred_logits = baseline_logits[:-1]  # 预测logits (shift)
    
    # 基线next-token准确率
    baseline_preds = torch.argmax(baseline_pred_logits, dim=-1)
    baseline_accuracy = (baseline_preds == baseline_next_tokens).float().mean().item()
    
    # 基线top-5命中率
    _, baseline_top5 = torch.topk(baseline_pred_logits, 5, dim=-1)
    baseline_top5_hit = (baseline_top5 == baseline_next_tokens.unsqueeze(-1)).any(dim=-1).float().mean().item()
    
    # 基线PPL
    shift_logits = baseline_pred_logits
    shift_labels = baseline_next_tokens
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    baseline_loss = loss_fn(shift_logits, shift_labels).item()
    baseline_ppl = np.exp(min(baseline_loss, 20))
    
    # 消融W_down
    if info.mlp_type == "split_gate_up":
        orig_weight = layers[l_idx].mlp.down_proj.weight.data.clone()
        layers[l_idx].mlp.down_proj.weight.data.zero_()
    elif info.mlp_type == "merged_gate_up":
        orig_weight = layers[l_idx].mlp.down_proj.weight.data.clone()
        layers[l_idx].mlp.down_proj.weight.data.zero_()
    else:
        return None
    
    # 消融后前向传播
    with torch.no_grad():
        ablated_outputs = model(input_ids)
        # 提取logits
        if hasattr(ablated_outputs, 'logits'):
            ablated_logits = ablated_outputs.logits
        elif isinstance(ablated_outputs, torch.Tensor):
            ablated_logits = ablated_outputs
        else:
            # 恢复权重并返回
            if info.mlp_type == "split_gate_up":
                layers[l_idx].mlp.down_proj.weight.data = orig_weight
            elif info.mlp_type == "merged_gate_up":
                layers[l_idx].mlp.down_proj.weight.data = orig_weight
            return None
    
    if ablated_logits.dim() == 3:
        ablated_logits = ablated_logits[0]
    
    ablated_pred_logits = ablated_logits[:-1]
    
    # 消融后指标
    ablated_preds = torch.argmax(ablated_pred_logits, dim=-1)
    ablated_accuracy = (ablated_preds == baseline_next_tokens).float().mean().item()
    
    _, ablated_top5 = torch.topk(ablated_pred_logits, 5, dim=-1)
    ablated_top5_hit = (ablated_top5 == baseline_next_tokens.unsqueeze(-1)).any(dim=-1).float().mean().item()
    
    ablated_loss = loss_fn(ablated_pred_logits, shift_labels).item()
    ablated_ppl = np.exp(min(ablated_loss, 20))
    
    # 恢复权重
    if info.mlp_type == "split_gate_up":
        layers[l_idx].mlp.down_proj.weight.data = orig_weight
    elif info.mlp_type == "merged_gate_up":
        layers[l_idx].mlp.down_proj.weight.data = orig_weight
    
    # 计算变化
    delta_ppl = ablated_ppl - baseline_ppl
    delta_accuracy = baseline_accuracy - ablated_accuracy
    delta_top5 = baseline_top5_hit - ablated_top5_hit
    
    importance_ppl = delta_ppl / max(baseline_ppl, 1e-10)
    importance_acc = delta_accuracy / max(baseline_accuracy, 1e-10)
    importance_top5 = delta_top5 / max(baseline_top5_hit, 1e-10)
    
    # KL散度
    kl_div = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(ablated_pred_logits, dim=-1),
        torch.nn.functional.softmax(baseline_pred_logits, dim=-1),
        reduction='batchmean'
    ).item()
    
    return {
        'baseline_ppl': baseline_ppl,
        'ablated_ppl': ablated_ppl,
        'delta_ppl': delta_ppl,
        'importance_ppl': importance_ppl,
        'baseline_accuracy': baseline_accuracy,
        'ablated_accuracy': ablated_accuracy,
        'delta_accuracy': delta_accuracy,
        'importance_acc': importance_acc,
        'baseline_top5': baseline_top5_hit,
        'ablated_top5': ablated_top5_hit,
        'delta_top5': delta_top5,
        'importance_top5': importance_top5,
        'kl_div': kl_div,
    }


def analyze_layer0_special(model, tokenizer, device, layers, info, d_model, model_name):
    """
    P494: 分析层0的特殊功能
    为什么第一层如此重要?
    """
    n_layers = len(layers)
    
    results = {}
    
    # 1. L0的W_down谱特征 vs 其他层
    l0_weights = get_layer_weights(layers[0], d_model, info.mlp_type)
    W_down_l0 = l0_weights.W_down
    
    # SVD
    U, s, Vt = np.linalg.svd(W_down_l0.astype(np.float32), full_matrices=False)
    PR_l0 = compute_participation_ratio(s)
    d_eff_l0 = compute_effective_dimension(s)
    top10_l0 = np.sum(s[:min(10, len(s))]**2) / np.sum(s**2)
    kappa_l0 = s[0] / max(s[-1], 1e-10)
    
    results['l0_spectral'] = {
        'PR': PR_l0,
        'd_eff': d_eff_l0,
        'top10_energy': top10_l0,
        'kappa': kappa_l0,
        's_max': s[0],
        's_min': s[-1],
        'norm_frobenius': np.sqrt(np.sum(s**2)),
    }
    
    # 2. 比较L0与其他层的谱特征
    other_PRs = []
    other_top10s = []
    other_kappas = []
    for l_idx in range(1, n_layers, max(1, n_layers // 10)):
        lw = get_layer_weights(layers[l_idx], d_model, info.mlp_type)
        W_d = lw.W_down
        _, s_l, _ = np.linalg.svd(W_d.astype(np.float32), full_matrices=False)
        other_PRs.append(compute_participation_ratio(s_l))
        other_top10s.append(np.sum(s_l[:min(10, len(s_l))]**2) / np.sum(s_l**2))
        other_kappas.append(s_l[0] / max(s_l[-1], 1e-10))
    
    results['spectral_comparison'] = {
        'l0_PR': PR_l0,
        'other_PR_mean': np.mean(other_PRs),
        'l0_top10': top10_l0,
        'other_top10_mean': np.mean(other_top10s),
        'l0_kappa': kappa_l0,
        'other_kappa_mean': np.mean(other_kappas),
        'l0_PR_zscore': (PR_l0 - np.mean(other_PRs)) / max(np.std(other_PRs), 1e-10),
    }
    
    # 3. L0的信号注入模式
    test_texts = [
        "The apple is red and sweet",
        "The sky is blue and vast",
        "The cat sat on the mat",
        "In mathematics, a group is a set",
        "The weather today is very cold",
    ]
    
    injection_patterns = []
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            h_all = compute_residual_stream(model, inputs["input_ids"], device)
        
        if h_all is not None and len(h_all) > 1:
            h0 = h_all[0]
            h1 = h_all[1]
            delta_h = h1 - h0  # L0→L1的信号注入
            
            if torch.is_tensor(delta_h):
                delta_h = delta_h.cpu().numpy()
            
            # delta_h的统计特性
            injection_patterns.append({
                'delta_norm': np.linalg.norm(delta_h),
                'delta_max': np.max(np.abs(delta_h)),
                'delta_std': np.std(delta_h),
                'delta_sparsity': np.sum(np.abs(delta_h) > 0.01 * np.max(np.abs(delta_h))) / len(delta_h),
            })
    
    if injection_patterns:
        results['injection_patterns'] = {
            'delta_norm_mean': np.mean([p['delta_norm'] for p in injection_patterns]),
            'delta_max_mean': np.mean([p['delta_max'] for p in injection_patterns]),
            'delta_std_mean': np.mean([p['delta_std'] for p in injection_patterns]),
            'delta_sparsity_mean': np.mean([p['delta_sparsity'] for p in injection_patterns]),
        }
    
    # 4. L0 vs 其他层的权重范数
    l0_norm = np.linalg.norm(W_down_l0)
    other_norms = []
    for l_idx in range(1, n_layers, max(1, n_layers // 10)):
        lw = get_layer_weights(layers[l_idx], d_model, info.mlp_type)
        other_norms.append(np.linalg.norm(lw.W_down))
    
    results['norm_comparison'] = {
        'l0_norm': l0_norm,
        'other_norm_mean': np.mean(other_norms),
        'l0_norm_ratio': l0_norm / max(np.mean(other_norms), 1e-10),
    }
    
    # 5. L0的"工作空间"分析: W_down是否映射到更均匀的空间?
    # 如果L0的W_down输出分布更均匀→L0为后续层建立了"工作空间"
    W_down_l0_output = W_down_l0  # [d_mlp, d_model]
    if W_down_l0_output.shape[0] > W_down_l0_output.shape[1]:
        # 行方向分析
        row_norms = np.linalg.norm(W_down_l0_output, axis=1)
    else:
        row_norms = np.linalg.norm(W_down_l0_output, axis=0)
    
    results['workspace'] = {
        'row_norms_cv': np.std(row_norms) / max(np.mean(row_norms), 1e-10),
        'row_norms_max_ratio': np.max(row_norms) / max(np.min(row_norms), 1e-10),
    }
    
    return results


def predict_importance_from_spectrum(spectral_features, importance_values):
    """
    P495: 从谱特征直接预测层重要性
    """
    from sklearn.linear_model import LinearRegression, Lasso
    from sklearn.preprocessing import StandardScaler
    
    # 准备数据
    X = []
    feature_names = []
    for feat_name in spectral_features[0].keys():
        if isinstance(spectral_features[0][feat_name], (int, float, np.floating)):
            feature_names.append(feat_name)
    
    for sf in spectral_features:
        row = [sf[fn] for fn in feature_names]
        X.append(row)
    
    X = np.array(X)
    y = np.array(importance_values)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # OLS回归
    ols = LinearRegression()
    ols.fit(X_scaled, y)
    y_pred_ols = ols.predict(X_scaled)
    r2_ols = 1 - np.sum((y - y_pred_ols)**2) / max(np.sum((y - np.mean(y))**2), 1e-10)
    
    # Lasso回归
    lasso = Lasso(alpha=0.01)
    lasso.fit(X_scaled, y)
    y_pred_lasso = lasso.predict(X_scaled)
    r2_lasso = 1 - np.sum((y - y_pred_lasso)**2) / max(np.sum((y - np.mean(y))**2), 1e-10)
    
    # 逐特征相关性
    feature_corrs = {}
    for i, fn in enumerate(feature_names):
        r, p = spearmanr(X[:, i], y)
        feature_corrs[fn] = {'corr': r, 'p_value': p}
    
    # 重要特征排序
    sorted_features = sorted(feature_corrs.items(), key=lambda x: abs(x[1]['corr']), reverse=True)
    
    return {
        'r2_ols': r2_ols,
        'r2_lasso': r2_lasso,
        'ols_coefs': dict(zip(feature_names, ols.coef_)),
        'lasso_coefs': dict(zip(feature_names, lasso.coef_)),
        'feature_corrs': feature_corrs,
        'top5_features': [(fn, fc['corr'], fc['p_value']) for fn, fc in sorted_features[:5]],
        'n_features': len(feature_names),
        'n_samples': len(y),
    }


# ============================================================
# 主实验函数
# ============================================================

def run_p492(model_name, device):
    """P492: 信息论gamma"""
    print(f"\n{'='*60}")
    print(f"P492: 信息论gamma - 用互信息替代能量占比 [{model_name}]")
    print(f"{'='*60}")
    
    model, tokenizer = load_model_safely(model_name, device)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        print("  无法获取层列表")
        return
    
    W_U = get_W_U(model)
    
    test_texts = [
        "The apple is red and sweet",
        "The sky is blue and vast", 
        "The cat sat on the mat",
        "In mathematics, a group is a set",
        "The weather today is very cold",
    ]
    
    sample_layers = get_sample_layers(n_layers, 12)
    print(f"  采样层: {sample_layers}")
    
    results = []
    for l_idx in sample_layers:
        layer_frac = l_idx / max(n_layers - 1, 1)
        
        gamma_info_list = []
        gamma_energy_list = []
        
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                h_all = compute_residual_stream(model, inputs["input_ids"], device)
            
            if h_all is not None and l_idx < len(h_all):
                h_l = h_all[l_idx]
                if torch.is_tensor(h_l):
                    h_l = h_l.cpu().numpy()
                
                # 取最后一个token的隐状态
                if h_l.ndim > 1:
                    h_last = h_l[-1]
                else:
                    h_last = h_l
                
                info_result = compute_information_gamma([h_last], W_U)
                gamma_info_list.append(info_result[0]['gamma_info'])
                gamma_energy_list.append(info_result[0]['gamma_energy'])
        
        if gamma_info_list:
            results.append({
                'layer': l_idx,
                'layer_frac': layer_frac,
                'gamma_info_mean': np.mean(gamma_info_list),
                'gamma_info_std': np.std(gamma_info_list),
                'gamma_energy_mean': np.mean(gamma_energy_list),
                'gamma_energy_std': np.std(gamma_energy_list),
            })
    
    # 分析
    if len(results) > 2:
        gf = [r['layer_frac'] for r in results]
        gi = [r['gamma_info_mean'] for r in results]
        ge = [r['gamma_energy_mean'] for r in results]
        
        r_info_vs_frac, p_info = spearmanr(gf, gi)
        r_energy_vs_frac, p_energy = spearmanr(gf, ge)
        r_info_vs_energy, p_ie = spearmanr(gi, ge)
        
        print(f"\n  gamma_info vs layer_frac: r={r_info_vs_frac:.3f}, p={p_info:.4f}")
        print(f"  gamma_energy vs layer_frac: r={r_energy_vs_frac:.3f}, p={p_energy:.4f}")
        print(f"  gamma_info vs gamma_energy: r={r_info_vs_energy:.3f}, p={p_ie:.4f}")
        
        print(f"\n  逐层详情:")
        for r in results:
            print(f"    L{r['layer']:2d} (f={r['layer_frac']:.2f}): "
                  f"gamma_info={r['gamma_info_mean']:.4f}, "
                  f"gamma_energy={r['gamma_energy_mean']:.4f}")
        
        # 结论
        print(f"\n  === P492 结论 ===")
        if abs(r_info_vs_frac) > abs(r_energy_vs_frac):
            print(f"  gamma_info与层位置的相关性({r_info_vs_frac:.3f}) > gamma_energy({r_energy_vs_frac:.3f})")
            print(f"  -> 信息论gamma可能是更好的信号聚焦指标")
        else:
            print(f"  gamma_energy与层位置的相关性({r_energy_vs_frac:.3f}) > gamma_info({r_info_vs_frac:.3f})")
            print(f"  -> 能量gamma可能仍然是更好的指标")
        
        print(f"  gamma_info vs gamma_energy: r={r_info_vs_energy:.3f}")
        if abs(r_info_vs_energy) > 0.8:
            print(f"  -> 两种gamma高度相关, 可能捕捉相同信息")
        elif abs(r_info_vs_energy) < 0.3:
            print(f"  -> 两种gamma几乎无关, 捕捉不同信息!")
    
    # 保存结果
    output_path = os.path.join(project_root, f"tests/glm5_temp/p492_{model_name}.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"P492: 信息论gamma [{model_name}]\n")
        for r in results:
            f.write(f"L{r['layer']}: gamma_info={r['gamma_info_mean']:.4f}, gamma_energy={r['gamma_energy_mean']:.4f}\n")
        if len(results) > 2:
            f.write(f"gamma_info vs layer_frac: r={r_info_vs_frac:.3f}\n")
            f.write(f"gamma_energy vs layer_frac: r={r_energy_vs_frac:.3f}\n")
            f.write(f"gamma_info vs gamma_energy: r={r_info_vs_energy:.3f}\n")
    print(f"  结果已保存到 {output_path}")
    
    return results


def run_p493(model_name, device):
    """P493: 语言能力的精细指标"""
    print(f"\n{'='*60}")
    print(f"P493: 语言能力的精细指标 [{model_name}]")
    print(f"{'='*60}")
    
    model, tokenizer = load_model_safely(model_name, device)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        print("  无法获取层列表")
        return
    
    W_U = get_W_U(model)
    
    test_text = "The apple is red and sweet. The sky is blue and vast. In mathematics, a group is a set equipped with an operation."
    
    sample_layers = get_sample_layers(n_layers, 12)
    print(f"  采样层: {sample_layers}")
    
    # 先计算每层的谱特征
    spectral_results = []
    for l_idx in sample_layers:
        lw = get_layer_weights(layers[l_idx], d_model, info.mlp_type)
        W_down = lw.W_down
        _, s, _ = np.linalg.svd(W_down.astype(np.float32), full_matrices=False)
        PR = compute_participation_ratio(s)
        d_eff = compute_effective_dimension(s)
        top10 = np.sum(s[:min(10, len(s))]**2) / np.sum(s**2)
        entropy_norm = -np.sum((s**2 / np.sum(s**2)) * np.log(s**2 / np.sum(s**2) + 1e-30)) / np.log(len(s))
        
        spectral_results.append({
            'layer': l_idx,
            'PR': PR,
            'd_eff': d_eff,
            'top10': top10,
            'entropy_norm': entropy_norm,
            'kappa': s[0] / max(s[-1], 1e-10),
        })
    
    # 计算层重要性(精细指标)
    importance_results = []
    for l_idx in sample_layers:
        print(f"  消融层 {l_idx}/{n_layers-1}...")
        imp = compute_layer_importance_detailed(model, tokenizer, device, test_text, layers, l_idx, model_name)
        if imp is not None:
            imp['layer'] = l_idx
            imp['layer_frac'] = l_idx / max(n_layers - 1, 1)
            importance_results.append(imp)
    
    # 分析各种重要性指标与谱特征的关系
    if len(importance_results) > 2 and len(spectral_results) > 2:
        # 合并数据
        combined = []
        for ir in importance_results:
            for sr in spectral_results:
                if ir['layer'] == sr['layer']:
                    combined.append({**ir, **sr})
                    break
        
        if len(combined) > 2:
            importance_metrics = ['importance_ppl', 'importance_acc', 'importance_top5', 'kl_div']
            spectral_feats = ['PR', 'd_eff', 'top10', 'entropy_norm', 'kappa']
            
            print(f"\n  === 各重要性指标 vs 谱特征 ===")
            for im in importance_metrics:
                print(f"\n  {im}:")
                im_vals = [c[im] for c in combined if im in c and np.isfinite(c[im])]
                for sf in spectral_feats:
                    sf_vals = [c[sf] for c in combined if im in c and np.isfinite(c[im])]
                    if len(im_vals) > 2:
                        r, p = spearmanr(im_vals, sf_vals)
                        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                        print(f"    vs {sf}: r={r:.3f}, p={p:.4f} {sig}")
            
            # 找最佳指标组合
            print(f"\n  === 最佳重要性指标 (与谱特征最相关) ===")
            best_metric = None
            best_mean_corr = 0
            for im in importance_metrics:
                im_vals = [c[im] for c in combined if im in c and np.isfinite(c[im])]
                corrs = []
                for sf in spectral_feats:
                    sf_vals = [c[sf] for c in combined if im in c and np.isfinite(c[im])]
                    if len(im_vals) > 2:
                        r, _ = spearmanr(im_vals, sf_vals)
                        corrs.append(abs(r))
                if corrs:
                    mean_corr = np.mean(corrs)
                    if mean_corr > best_mean_corr:
                        best_mean_corr = mean_corr
                        best_metric = im
            
            if best_metric:
                print(f"  最佳指标: {best_metric} (mean |r| = {best_mean_corr:.3f})")
    
    # 保存结果
    output_path = os.path.join(project_root, f"tests/glm5_temp/p493_{model_name}.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"P493: 语言能力精细指标 [{model_name}]\n")
        for ir in importance_results:
            f.write(f"L{ir['layer']}: importance_ppl={ir.get('importance_ppl',0):.4f}, "
                    f"importance_acc={ir.get('importance_acc',0):.4f}, "
                    f"importance_top5={ir.get('importance_top5',0):.4f}, "
                    f"kl_div={ir.get('kl_div',0):.4f}\n")
    print(f"  结果已保存到 {output_path}")
    
    return importance_results


def run_p494(model_name, device):
    """P494: 层0的特殊功能分析"""
    print(f"\n{'='*60}")
    print(f"P494: 层0的特殊功能分析 [{model_name}]")
    print(f"{'='*60}")
    
    model, tokenizer = load_model_safely(model_name, device)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        print("  无法获取层列表")
        return
    
    results = analyze_layer0_special(model, tokenizer, device, layers, info, d_model, model_name)
    
    # 打印结果
    print(f"\n  === L0 谱特征 ===")
    for k, v in results['l0_spectral'].items():
        print(f"    {k}: {v:.4f}")
    
    print(f"\n  === L0 vs 其他层谱比较 ===")
    for k, v in results['spectral_comparison'].items():
        print(f"    {k}: {v:.4f}")
    
    if 'injection_patterns' in results:
        print(f"\n  === L0->L1 信号注入模式 ===")
        for k, v in results['injection_patterns'].items():
            print(f"    {k}: {v:.4f}")
    
    print(f"\n  === L0 vs 其他层范数比较 ===")
    for k, v in results['norm_comparison'].items():
        print(f"    {k}: {v:.4f}")
    
    print(f"\n  === L0 工作空间分析 ===")
    for k, v in results['workspace'].items():
        print(f"    {k}: {v:.4f}")
    
    # 结论
    print(f"\n  === P494 结论 ===")
    sc = results['spectral_comparison']
    if sc['l0_PR_zscore'] > 1.96:
        print(f"  L0的PR显著高于其他层(z={sc['l0_PR_zscore']:.2f})")
        print(f"  -> L0的谱更平坦, 更多奇异值参与")
    elif sc['l0_PR_zscore'] < -1.96:
        print(f"  L0的PR显著低于其他层(z={sc['l0_PR_zscore']:.2f})")
        print(f"  -> L0的谱更尖锐, 信号更集中")
    else:
        print(f"  L0的PR与其他层无显著差异(z={sc['l0_PR_zscore']:.2f})")
        print(f"  -> L0的特殊性不在谱结构, 而在功能角色")
    
    nc = results['norm_comparison']
    if nc['l0_norm_ratio'] > 1.5:
        print(f"  L0权重范数是其他层的{nc['l0_norm_ratio']:.2f}倍")
        print(f"  -> L0有更强的信号处理能力")
    elif nc['l0_norm_ratio'] < 0.67:
        print(f"  L0权重范数是其他层的{nc['l0_norm_ratio']:.2f}倍(更弱)")
        print(f"  -> L0不是通过权重强度, 而是通过位置效应起作用")
    
    # 保存结果
    output_path = os.path.join(project_root, f"tests/glm5_temp/p494_{model_name}.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"P494: 层0特殊功能 [{model_name}]\n")
        for section_name, section_data in results.items():
            f.write(f"\n[{section_name}]\n")
            for k, v in section_data.items():
                f.write(f"  {k}: {v:.4f}\n")
    print(f"  结果已保存到 {output_path}")
    
    return results


def run_p495(model_name, device):
    """P495: 从权重结构直接预测语言能力"""
    print(f"\n{'='*60}")
    print(f"P495: 从权重结构直接预测语言能力 [{model_name}]")
    print(f"{'='*60}")
    
    model, tokenizer = load_model_safely(model_name, device)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        print("  无法获取层列表")
        return
    
    W_U = get_W_U(model)
    
    test_text = "The apple is red and sweet. The sky is blue and vast. In mathematics, a group is a set equipped with an operation."
    
    # 收集所有层的谱特征
    all_spectral = []
    all_importance = []
    
    for l_idx in range(n_layers):
        print(f"  处理层 {l_idx}/{n_layers-1}...")
        layer_frac = l_idx / max(n_layers - 1, 1)
        
        # 谱特征
        lw = get_layer_weights(layers[l_idx], d_model, info.mlp_type)
        W_down = lw.W_down
        # 用截断SVD避免大矩阵内存问题
        try:
            from sklearn.utils.extmath import randomized_svd
            k_svd = min(400, min(W_down.shape) - 1)
            _, s, _ = randomized_svd(W_down.astype(np.float32), n_components=k_svd, random_state=42)
        except (ImportError, MemoryError):
            k_svd = min(200, min(W_down.shape) - 1)
            _, s, _ = np.linalg.svd(W_down[:min(2000, W_down.shape[0])].astype(np.float32), full_matrices=False)
            s = s[:k_svd]
        
        s_sq = s**2
        s_sq_norm = s_sq / np.sum(s_sq)
        
        PR = compute_participation_ratio(s)
        d_eff = compute_effective_dimension(s)
        top10 = np.sum(s[:min(10, len(s))]**2) / np.sum(s**2)
        top50 = np.sum(s[:min(50, len(s))]**2) / np.sum(s**2)
        top100 = np.sum(s[:min(100, len(s))]**2) / np.sum(s**2)
        entropy_norm = -np.sum(s_sq_norm * np.log(s_sq_norm + 1e-30)) / np.log(len(s))
        kappa = s[0] / max(s[-1], 1e-10)
        norm_fro = np.sqrt(np.sum(s_sq))
        s_max = s[0]
        s_mean = np.mean(s)
        s_std = np.std(s)
        s_cv = s_std / max(s_mean, 1e-10)
        
        # 层位置
        layer_position = layer_frac
        layer_position_sq = layer_frac**2
        
        spectral_features = {
            'PR': PR,
            'd_eff': d_eff,
            'top10': top10,
            'top50': top50,
            'top100': top100,
            'entropy_norm': entropy_norm,
            'kappa': kappa,
            'norm_fro': norm_fro,
            's_max': s_max,
            's_mean': s_mean,
            's_std': s_std,
            's_cv': s_cv,
            'layer_frac': layer_position,
            'layer_frac_sq': layer_position_sq,
        }
        all_spectral.append(spectral_features)
        
        # 重要性(用PPL消融)
        imp = compute_layer_importance_detailed(model, tokenizer, device, test_text, layers, l_idx, model_name)
        if imp is not None:
            importance_val = imp['importance_ppl']
        else:
            importance_val = 0
        all_importance.append(importance_val)
    
    # 回归分析
    if len(all_spectral) > 3:
        result = predict_importance_from_spectrum(all_spectral, all_importance)
        
        print(f"\n  === 回归结果 ===")
        print(f"  OLS R2: {result['r2_ols']:.3f}")
        print(f"  Lasso R2: {result['r2_lasso']:.3f}")
        
        print(f"\n  Top-5 预测特征:")
        for fn, corr, p in result['top5_features']:
            print(f"    {fn}: r={corr:.3f}, p={p:.4f}")
        
        print(f"\n  Lasso 非零系数:")
        for fn, coef in result['lasso_coefs'].items():
            if abs(coef) > 0.001:
                print(f"    {fn}: {coef:.4f}")
        
        # 去掉L0后重新分析
        if n_layers > 3:
            spectral_no_l0 = all_spectral[1:]
            importance_no_l0 = all_importance[1:]
            result_no_l0 = predict_importance_from_spectrum(spectral_no_l0, importance_no_l0)
            
            print(f"\n  === 去掉L0后的回归结果 ===")
            print(f"  OLS R2: {result_no_l0['r2_ols']:.3f} (vs 全部: {result['r2_ols']:.3f})")
            print(f"  Lasso R2: {result_no_l0['r2_lasso']:.3f} (vs 全部: {result['r2_lasso']:.3f})")
            
            print(f"\n  去掉L0后 Top-5 预测特征:")
            for fn, corr, p in result_no_l0['top5_features']:
                print(f"    {fn}: r={corr:.3f}, p={p:.4f}")
    
    # 保存结果
    output_path = os.path.join(project_root, f"tests/glm5_temp/p495_{model_name}.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"P495: 权重结构预测语言能力 [{model_name}]\n")
        for i, (sf, imp) in enumerate(zip(all_spectral, all_importance)):
            f.write(f"L{i}: importance={imp:.4f}, PR={sf['PR']:.4f}, top10={sf['top10']:.4f}\n")
        if 'result' in dir():
            f.write(f"\nOLS R2: {result['r2_ols']:.3f}\n")
            f.write(f"Lasso R2: {result['r2_lasso']:.3f}\n")
    print(f"  结果已保存到 {output_path}")
    
    return {'spectral': all_spectral, 'importance': all_importance}


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase CIII: 信号聚焦重新定义")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["p492", "p493", "p494", "p495"])
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if args.experiment == "p492":
        run_p492(args.model, device)
    elif args.experiment == "p493":
        run_p493(args.model, device)
    elif args.experiment == "p494":
        run_p494(args.model, device)
    elif args.experiment == "p495":
        run_p495(args.model, device)


if __name__ == "__main__":
    main()
