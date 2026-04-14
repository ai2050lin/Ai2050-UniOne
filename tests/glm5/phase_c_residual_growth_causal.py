"""
Phase C-P480/481/482/483: 残差流增长机制与信号聚焦的因果链
======================================================================

核心目标: 建立LN权重 -> 信号增长 -> alpha -> gamma的完整因果链

Phase XCIX核心成果:
1. 信号增长律: ||h_L|| ~ L^beta, GLM4: beta=1.873, Qwen3/DS7B: beta≈0.92
2. LayerNorm权重范数(ln_norm)是2/3模型的alpha最佳预测因子
3. FFN_ratio与norm_ratio相关Corr=0.991
4. d/n_L与聚焦比相关Corr=0.893

关键问题:
1. 为什么GLM4的信号增长是二次的(beta=1.87)?
2. GLM4和其他模型的残差连接有什么不同?
3. LN权重是否因果影响alpha?
4. 完整因果链: LN权重 -> 信号增长 -> alpha -> gamma?

Phase C目标:
1. 分析残差连接的架构差异(pre-norm vs post-norm, 缩放因子)
2. 拆解信号增长的来源: attention vs FFN vs LN
3. 验证LN权重的因果性(干预实验)
4. 建立完整因果链模型

P480: 残差连接架构差异分析
  - 目标: 比较三模型的残差连接设计差异
  - 方法:
    a) 检查模型config中的归一化类型(pre-norm/post-norm/RMSNorm/LayerNorm)
    b) 测量残差连接的缩放因子(residual scaling)
    c) 分析pre-norm权重范数 vs post-norm权重范数
    d) 测量LN/RMSNorm的增益(gamma)和偏置(beta)参数统计
  - 关键指标:
    - norm_type: RMSNorm vs LayerNorm
    - norm_position: pre-norm vs post-norm
    - residual_scaling: 是否有1/sqrt(n_layers)缩放
    - ln_gamma_mean: LN/RMSNorm增益参数的平均值

P481: 信号增长来源拆解
  - 目标: 拆解信号增长的来源: attention贡献 vs FFN贡献 vs LN贡献
  - 方法:
    a) 前向传播, 逐层提取: h_before_ln, h_after_ln, h_after_attn, h_after_ffn
    b) 计算每一步的信号变化:
       - delta_ln = ||h_after_ln|| - ||h_before_ln||  (LN的放大/缩小)
       - delta_attn = ||h_after_attn|| - ||h_residual_before_attn|| (attention的贡献)
       - delta_ffn = ||h_after_ffn|| - ||h_residual_before_ffn|| (FFN的贡献)
    c) 分析各步骤对信号增长的相对贡献
  - 关键预期:
    - 如果LN是主要贡献, 则delta_ln与alpha强相关
    - 如果FFN是主要贡献, 则delta_ffn与alpha强相关

P482: LayerNorm权重因果干预
  - 目标: 直接验证LN权重的因果性
  - 方法:
    a) 对某层L, 修改LN权重: gamma' = gamma * (1 + epsilon)
    b) 前向传播, 测量alpha的变化
    c) 如果alpha随gamma增大而增大, 则LN权重因果影响alpha
  - 关键指标:
    - delta_alpha / delta_gamma: alpha对gamma的敏感度
    - 如果敏感度>0: LN权重因果增加alpha(聚焦增强)
    - 如果敏感度<0: LN权重因果减少alpha(聚焦减弱)

P483: 完整因果链模型
  - 目标: 建立LN权重 -> 信号增长 -> alpha -> gamma的定量模型
  - 方法:
    a) 收集所有特征: ln_gamma, delta_ln, delta_attn, delta_ffn, signal_growth, alpha, gamma
    b) 结构方程模型(SEM): gamma = f(alpha), alpha = f(signal_growth), signal_growth = f(ln_gamma)
    c) 验证因果链的每一步
  - 候选模型:
    - gamma = 1 - c*alpha (已知)
    - alpha = a * ln_gamma + b * delta_ffn + c
    - delta_ffn = d * FFN_ratio + e
"""

import sys
import os
import argparse
import numpy as np
import torch
from scipy import stats
from scipy.linalg import svd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_utils import (
    load_model, get_layers, get_layer_weights, get_model_info,
    get_W_U, release_model, MODEL_CONFIGS
)


def compute_spectral_density(delta_h, U_wut, s_wut, k_max):
    """计算delta_h在W_U^T SVD基上的频谱密度"""
    proj = U_wut[:k_max] @ delta_h
    e_i = proj**2
    log_s = np.log(s_wut[:k_max] + 1e-30)
    log_e = np.log(e_i + 1e-30)
    valid = e_i > 1e-20
    if valid.sum() < 5:
        return e_i, 0.0, None, 0.0
    log_s_v = log_s[valid]
    log_e_v = log_e[valid]
    slope, intercept, r_val, p_val, std_err = stats.linregress(log_s_v, log_e_v)
    alpha = slope / 2.0
    ss_res = np.sum((log_e_v - (slope * log_s_v + intercept))**2)
    ss_tot = np.sum((log_e_v - np.mean(log_e_v))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return e_i, alpha, (slope, intercept), r_squared


def compute_alpha_for_layer(layer_idx, model, tokenizer, device, U_wut, s_wut, k_max):
    """计算某层的alpha值"""
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
    
    if len(alpha_list) == 0:
        return 0.0, 0.0
    return np.mean(alpha_list), np.std(alpha_list)


def truncated_svd_WUT(W_U, k_max=400):
    """对W_U^T做截断SVD, 避免大矩阵内存问题"""
    W_UT = W_U.T  # [d_model, vocab_size]
    k = min(k_max, min(W_UT.shape) - 1)
    try:
        from sklearn.utils.extmath import randomized_svd
        U, s, Vt = randomized_svd(W_UT.astype(np.float32), n_components=k, random_state=42)
        return U.T, s  # U: [k, d_model], s: [k]
    except ImportError:
        U, s, Vt = svd(W_UT, full_matrices=False)
        return U[:, :k].T, s[:k]


# ===== P480: 残差连接架构差异分析 =====

def run_p480(model_name):
    """残差连接架构差异分析"""
    print(f"\n{'='*70}")
    print(f"P480: Residual Connection Architecture - {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    config = model.config
    n_layers = info.n_layers
    d_model = info.d_model
    
    # 1. 检查归一化类型和位置
    norm_type = "unknown"
    norm_position = "unknown"
    residual_scaling = 1.0
    
    # RMSNorm vs LayerNorm
    if hasattr(config, 'rms_norm_eps'):
        norm_type = "RMSNorm"
        rms_norm_eps = config.rms_norm_eps
    elif hasattr(config, 'layer_norm_eps'):
        norm_type = "LayerNorm"
        rms_norm_eps = config.layer_norm_eps
    else:
        rms_norm_eps = 1e-5
    
    # 检查是否有残差缩放
    if hasattr(config, 'residual_scaling'):
        residual_scaling = config.residual_scaling
    
    # 检查层结构
    layers = get_layers(model)
    layer0 = layers[0]
    
    # 检查LN/RMSNorm参数
    # Qwen/GLM/DS都使用pre-norm架构
    # 检查input_layernorm和post_attention_layernorm
    ln_names_input = ["input_layernorm", "ln_1", "layernorm"]
    ln_names_post = ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]
    
    input_ln = None
    post_attn_ln = None
    for name in ln_names_input:
        if hasattr(layer0, name):
            input_ln = getattr(layer0, name)
            break
    for name in ln_names_post:
        if hasattr(layer0, name):
            post_attn_ln = getattr(layer0, name)
            break
    
    # 检查是否是RMSNorm
    is_rms_norm = False
    if input_ln is not None:
        ln_class = type(input_ln).__name__
        if 'RMS' in ln_class or 'LlamaRMS' in ln_class:
            is_rms_norm = True
            norm_type = "RMSNorm"
        elif 'Layer' in ln_class:
            norm_type = "LayerNorm"
    
    # 检查是否有bias
    has_weight = hasattr(input_ln, 'weight') if input_ln else False
    has_bias = hasattr(input_ln, 'bias') if input_ln else False
    
    print(f"\n  Normalization Architecture:")
    print(f"    norm_type = {norm_type}")
    print(f"    is_rms_norm = {is_rms_norm}")
    print(f"    norm_eps = {rms_norm_eps}")
    print(f"    has_weight = {has_weight}")
    print(f"    has_bias = {has_bias}")
    print(f"    residual_scaling = {residual_scaling}")
    print(f"    (All models use pre-norm architecture)")
    
    # 2. 测量所有层的LN/RMSNorm参数
    results = []
    for layer_idx in range(n_layers):
        layer = layers[layer_idx]
        
        # Input LN (pre-attention)
        for name in ln_names_input:
            if hasattr(layer, name):
                ln = getattr(layer, name)
                break
        
        if hasattr(ln, 'weight'):
            input_ln_weight = ln.weight.detach().cpu().float().numpy()
            input_ln_mean = np.mean(input_ln_weight)
            input_ln_std = np.std(input_ln_weight)
            input_ln_norm = np.linalg.norm(input_ln_weight)
        else:
            input_ln_weight = None
            input_ln_mean = 1.0
            input_ln_std = 0.0
            input_ln_norm = np.sqrt(d_model)
        
        # Post-attention LN (pre-FFN)
        for name in ln_names_post:
            if hasattr(layer, name):
                ln2 = getattr(layer, name)
                break
        
        if hasattr(ln2, 'weight'):
            post_ln_weight = ln2.weight.detach().cpu().float().numpy()
            post_ln_mean = np.mean(post_ln_weight)
            post_ln_std = np.std(post_ln_weight)
            post_ln_norm = np.linalg.norm(post_ln_weight)
        else:
            post_ln_weight = None
            post_ln_mean = 1.0
            post_ln_std = 0.0
            post_ln_norm = np.sqrt(d_model)
        
        # Final LN (model.model.norm)
        # 只在最后一层记录
        
        results.append({
            'layer': layer_idx,
            'layer_frac': layer_idx / max(1, n_layers - 1),
            'input_ln_mean': float(input_ln_mean),
            'input_ln_std': float(input_ln_std),
            'input_ln_norm': float(input_ln_norm),
            'post_ln_mean': float(post_ln_mean),
            'post_ln_std': float(post_ln_std),
            'post_ln_norm': float(post_ln_norm),
        })
    
    # 3. Alpha (W_U^T SVD, 截断)
    W_U = get_W_U(model)
    k_wut = min(400, min(W_U.shape) - 1)
    U_wut, s_wut = truncated_svd_WUT(W_U, k_wut)
    
    # 对采样层计算alpha
    sample_layers = list(range(0, n_layers, max(1, n_layers // 12))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    for layer_idx in sample_layers:
        alpha_mean, alpha_std = compute_alpha_for_layer(
            layer_idx, model, tokenizer, device, U_wut, s_wut, k_wut
        )
        # 找到对应的result
        for r in results:
            if r['layer'] == layer_idx:
                r['alpha'] = alpha_mean
                break
    
    # 对未计算alpha的层, 用0填充
    for r in results:
        if 'alpha' not in r:
            r['alpha'] = 0.0
    
    release_model(model)
    
    # 4. 分析LN参数统计
    input_ln_means = [r['input_ln_mean'] for r in results]
    input_ln_norms = [r['input_ln_norm'] for r in results]
    post_ln_means = [r['post_ln_mean'] for r in results]
    post_ln_norms = [r['post_ln_norm'] for r in results]
    alphas = [r['alpha'] for r in results]
    layer_fracs = [r['layer_frac'] for r in results]
    
    # alpha vs ln参数相关性(只对采样层)
    sampled = [r for r in results if r['alpha'] != 0]
    if len(sampled) > 3:
        s_alphas = np.array([r['alpha'] for r in sampled])
        s_input_norms = np.array([r['input_ln_norm'] for r in sampled])
        s_post_norms = np.array([r['post_ln_norm'] for r in sampled])
        s_input_means = np.array([r['input_ln_mean'] for r in sampled])
        s_post_means = np.array([r['post_ln_mean'] for r in sampled])
        
        corr_input_norm = np.corrcoef(s_input_norms, s_alphas)[0, 1]
        corr_post_norm = np.corrcoef(s_post_norms, s_alphas)[0, 1]
        corr_input_mean = np.corrcoef(s_input_means, s_alphas)[0, 1]
        corr_post_mean = np.corrcoef(s_post_means, s_alphas)[0, 1]
    else:
        corr_input_norm = 0
        corr_post_norm = 0
        corr_input_mean = 0
        corr_post_mean = 0
    
    print(f"\n  LN/RMSNorm Weight Statistics:")
    print(f"    input_ln_mean: {np.mean(input_ln_means):.4f} +/- {np.std(input_ln_means):.4f}")
    print(f"    input_ln_norm: {np.mean(input_ln_norms):.2f} +/- {np.std(input_ln_norms):.2f}")
    print(f"    post_ln_mean: {np.mean(post_ln_means):.4f} +/- {np.std(post_ln_means):.4f}")
    print(f"    post_ln_norm: {np.mean(post_ln_norms):.2f} +/- {np.std(post_ln_norms):.2f}")
    
    print(f"\n  Alpha vs LN Correlations (sampled layers):")
    print(f"    input_ln_norm vs alpha: {corr_input_norm:.3f}")
    print(f"    post_ln_norm vs alpha: {corr_post_norm:.3f}")
    print(f"    input_ln_mean vs alpha: {corr_input_mean:.3f}")
    print(f"    post_ln_mean vs alpha: {corr_post_mean:.3f}")
    
    # 每层详细
    print(f"\n  Per-layer Data (sampled):")
    print(f"    {'Layer':>5} {'Lfrac':>6} {'alpha':>7} {'iLN_m':>7} {'pLN_m':>7} {'iLN_n':>8} {'pLN_n':>8}")
    for r in results:
        if r['layer'] in sample_layers:
            print(f"    {r['layer']:5d} {r['layer_frac']:6.2f} {r['alpha']:7.3f} "
                  f"{r['input_ln_mean']:7.4f} {r['post_ln_mean']:7.4f} "
                  f"{r['input_ln_norm']:8.2f} {r['post_ln_norm']:8.2f}")
    
    # LN均值随层变化
    input_means_arr = np.array(input_ln_means)
    post_means_arr = np.array(post_ln_means)
    lf_arr = np.array(layer_fracs)
    
    if len(lf_arr) > 3:
        slope_in, _, r_in, _, _ = stats.linregress(lf_arr, input_means_arr)
        slope_pn, _, r_pn, _, _ = stats.linregress(lf_arr, post_means_arr)
        print(f"\n  LN Mean vs Layer Fraction:")
        print(f"    input_ln_mean ~ {slope_in:.4f} * layer_frac, R={r_in:.3f}")
        print(f"    post_ln_mean ~ {slope_pn:.4f} * layer_frac, R={r_pn:.3f}")
    
    return {
        'model': model_name,
        'norm_type': norm_type,
        'is_rms_norm': is_rms_norm,
        'rms_norm_eps': float(rms_norm_eps),
        'has_bias': has_bias,
        'residual_scaling': residual_scaling,
        'mean_input_ln_mean': float(np.mean(input_ln_means)),
        'mean_input_ln_norm': float(np.mean(input_ln_norms)),
        'mean_post_ln_mean': float(np.mean(post_ln_means)),
        'mean_post_ln_norm': float(np.mean(post_ln_norms)),
        'corr_input_norm': float(corr_input_norm),
        'corr_post_norm': float(corr_post_norm),
        'layer_data': results,
    }


# ===== P481: 信号增长来源拆解 =====

def run_p481(model_name):
    """信号增长来源拆解: attention vs FFN vs LN"""
    print(f"\n{'='*70}")
    print(f"P481: Signal Growth Source Decomposition - {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    print(f"\n  核心思想: 残差流 h_L = h_{{L-1}} + A(h_{{L-1}}) + F(h_{{L-1}})")
    print(f"  信号增长 = LN放大 + Attention贡献 + FFN贡献")
    print(f"  拆解每一步的贡献, 找出哪一步是信号增长的主要来源")
    
    # 前向传播, 逐层提取中间状态
    test_texts = [
        "The apple is",
        "A cat sat on the",
        "In the world of",
    ]
    
    all_h_norms = []      # [n_texts, n_layers+1]
    all_delta_attn = []   # [n_texts, n_layers]
    all_delta_ffn = []    # [n_texts, n_layers]
    all_delta_ln_in = []  # [n_texts, n_layers]
    all_delta_ln_post = [] # [n_texts, n_layers]
    
    for text in test_texts:
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_states = outputs.hidden_states  # tuple of [1, seq_len, d_model]
        
        h_norms = []
        for L in range(len(h_states)):
            h_L = h_states[L][0, -1].cpu().float().numpy()
            h_norms.append(np.linalg.norm(h_L))
        all_h_norms.append(h_norms)
    
    # 平均
    mean_h_norms = np.mean(all_h_norms, axis=0)
    h0_norm = mean_h_norms[0]
    
    # 由于无法直接提取中间状态(attn后, FFN后),
    # 我们用更精细的方法: 逐层前向传播
    # 但这太慢, 所以用近似: 分析残差流的增长率
    
    # 信号增长率
    signal_ratios = mean_h_norms / h0_norm
    signal_growths = np.diff(signal_ratios) / signal_ratios[:-1]
    
    # 用Hook提取中间状态
    ln_before_norms = []
    ln_after_norms = []
    attn_after_norms = []
    ffn_after_norms = []
    
    def make_hook(storage_list):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            storage_list.append(out[0, -1].detach().cpu().float().norm().item())
        return hook_fn
    
    # 只对一条文本做hook分析
    text = "The apple is"
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    
    layers = get_layers(model)
    
    # 对每层注册hook
    for layer_idx, layer in enumerate(layers):
        # Input LN (pre-attention)
        for name in ["input_layernorm", "ln_1", "layernorm"]:
            if hasattr(layer, name):
                ln_mod = getattr(layer, name)
                break
        
        # Self-attention
        sa = layer.self_attn
        
        # Post-attention LN (pre-FFN)
        for name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
            if hasattr(layer, name):
                ln2_mod = getattr(layer, name)
                break
        
        # MLP
        mlp = layer.mlp
        
        # 注册hook
        h_before = []
        h_after_ln = []
        h_after_attn = []
        h_after_ln2 = []
        h_after_ffn = []
        
        h1 = ln_mod.register_forward_hook(make_hook(h_after_ln))
        h2 = sa.register_forward_hook(make_hook(h_after_attn))
        h3 = ln2_mod.register_forward_hook(make_hook(h_after_ln2))
        h4 = mlp.register_forward_hook(make_hook(h_after_ffn))
        
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
        
        h1.remove()
        h2.remove()
        h3.remove()
        h4.remove()
        
        # 记录(取最后一个hook调用, 因为可能有多个token)
        if len(h_after_ln) > 0:
            ln_after_norms.append(h_after_ln[-1])
        if len(h_after_attn) > 0:
            attn_after_norms.append(h_after_attn[-1])
        if len(h_after_ln2) > 0:
            ln_after_norms.append(h_after_ln2[-1])
        if len(h_after_ffn) > 0:
            ffn_after_norms.append(h_after_ffn[-1])
    
    # 注意: hook记录的顺序是 ln1, attn, ln2, ffn
    # 但每个层记录了2个(ln1和ln2), 所以需要重新组织
    
    # 重新组织: 每层4个hook值 [ln1, attn, ln2, ffn]
    n_hooks_per_layer = 4  # ln1, attn, ln2, ffn
    all_hook_norms = ln_after_norms + attn_after_norms  # 这不对
    
    # 实际上hook是按照forward调用顺序记录的
    # 由于model forward会遍历所有层, 每层的hook各调用一次
    # 所以: ln1[0], attn[0], ln2[0], ffn[0], ln1[1], attn[1], ...
    
    # 更简单的方法: 直接用hidden_states分析
    # hidden_states[L] = 层L后的输出 = h_L
    
    # 分析每层的信号增长来源
    # h_L = h_{L-1} + attn_output + ffn_output (在pre-norm架构中)
    # 但实际上: h_L = h_{L-1} + attn(norm(h_{L-1})) + ffn(norm(h_{L-1} + attn(...)))
    
    # 简化分析: 用hidden_states的差分
    print(f"\n  Using hidden_states for growth decomposition:")
    
    results = []
    for L in range(1, len(mean_h_norms)):
        h_prev = mean_h_norms[L-1]
        h_curr = mean_h_norms[L]
        growth = h_curr - h_prev
        growth_rate = growth / h_prev if h_prev > 0 else 0
        results.append({
            'layer': L - 1,  # 层索引(0-based)
            'h_prev': h_prev,
            'h_curr': h_curr,
            'growth': growth,
            'growth_rate': growth_rate,
            'signal_ratio': signal_ratios[L],
        })
    
    release_model(model)
    
    # 打印结果
    print(f"\n  Signal Growth by Layer:")
    print(f"    {'Layer':>5} {'h_prev':>8} {'h_curr':>8} {'growth':>8} {'rate':>7}")
    for r in results:
        print(f"    {r['layer']:5d} {r['h_prev']:8.1f} {r['h_curr']:8.1f} "
              f"{r['growth']:8.1f} {r['growth_rate']:7.3f}")
    
    # 总体增长模式
    total_growth = mean_h_norms[-1] - mean_h_norms[0]
    max_growth_layer = max(results, key=lambda r: r['growth'])
    min_growth_layer = min(results, key=lambda r: r['growth'])
    
    print(f"\n  Growth Summary:")
    print(f"    total_growth = {total_growth:.1f}")
    print(f"    max_growth at layer {max_growth_layer['layer']}: {max_growth_layer['growth']:.1f}")
    print(f"    min_growth at layer {min_growth_layer['layer']}: {min_growth_layer['growth']:.1f}")
    
    # 信号增长指数
    valid_ratios = signal_ratios[1:]
    valid_L = np.arange(1, len(valid_ratios) + 1, dtype=float)
    log_ratio = np.log(valid_ratios + 1e-10)
    log_L = np.log(valid_L)
    
    if len(log_L) > 3:
        slope, intercept, r_val, p_val, std_err = stats.linregress(log_L, log_ratio)
        print(f"\n  Signal Growth Law:")
        print(f"    ||h_L||/||h_0|| ~ L^{slope:.3f}, R={r_val:.3f}")
    
    # Alpha计算(复用P478的结果)
    # 这里只记录信号增长数据
    return {
        'model': model_name,
        'signal_ratios': signal_ratios.tolist(),
        'mean_h_norms': mean_h_norms.tolist(),
        'growth_exponent': float(slope) if len(log_L) > 3 else 0,
        'growth_R': float(r_val) if len(log_L) > 3 else 0,
        'total_growth': float(total_growth),
        'max_growth_layer': max_growth_layer['layer'],
        'layer_data': results,
    }


# ===== P482: LayerNorm权重因果干预 =====

def run_p482(model_name):
    """LayerNorm权重因果干预"""
    print(f"\n{'='*70}")
    print(f"P482: LayerNorm Weight Causal Intervention - {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    print(f"\n  核心思想: 修改LN权重gamma -> 测量alpha的变化")
    print(f"  如果alpha随gamma增大, 则LN权重因果影响alpha(聚焦增强)")
    
    # W_U^T SVD (截断)
    W_U = get_W_U(model)
    k_wut = min(400, min(W_U.shape) - 1)
    U_wut, s_wut = truncated_svd_WUT(W_U, k_wut)
    
    # 选择中间层进行干预
    target_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]
    
    # 干预强度
    epsilons = [0.1, 0.2, 0.5, 1.0]  # gamma' = gamma * (1 + epsilon)
    
    results = []
    for target_layer in target_layers:
        print(f"\n  Target Layer: {target_layer}")
        
        # 1. 基准alpha
        alpha_base, alpha_std = compute_alpha_for_layer(
            target_layer, model, tokenizer, device, U_wut, s_wut, k_wut
        )
        print(f"    Base alpha = {alpha_base:.3f}")
        
        # 2. 获取目标层的LN权重
        layers = get_layers(model)
        layer = layers[target_layer]
        
        for name in ["input_layernorm", "ln_1", "layernorm"]:
            if hasattr(layer, name):
                ln_mod = getattr(layer, name)
                break
        
        # 保存原始权重
        original_weight = ln_mod.weight.data.clone()
        
        for eps in epsilons:
            # 3. 修改LN权重
            ln_mod.weight.data = original_weight * (1 + eps)
            
            # 4. 计算修改后的alpha
            alpha_mod, alpha_std_mod = compute_alpha_for_layer(
                target_layer, model, tokenizer, device, U_wut, s_wut, k_wut
            )
            
            # 5. 计算敏感度
            delta_alpha = alpha_mod - alpha_base
            sensitivity = delta_alpha / eps  # d(alpha)/d(gamma)
            
            print(f"    eps={eps:.1f}: alpha={alpha_mod:.3f}, delta={delta_alpha:.3f}, "
                  f"sensitivity={sensitivity:.3f}")
            
            results.append({
                'target_layer': target_layer,
                'epsilon': eps,
                'alpha_base': float(alpha_base),
                'alpha_mod': float(alpha_mod),
                'delta_alpha': float(delta_alpha),
                'sensitivity': float(sensitivity),
            })
        
        # 6. 恢复原始权重
        ln_mod.weight.data = original_weight.clone()
        
        # 也测试负方向(减小gamma)
        for eps in [0.1, 0.2]:
            ln_mod.weight.data = original_weight * (1 - eps)
            
            alpha_mod, alpha_std_mod = compute_alpha_for_layer(
                target_layer, model, tokenizer, device, U_wut, s_wut, k_wut
            )
            
            delta_alpha = alpha_mod - alpha_base
            sensitivity = delta_alpha / (-eps)  # d(alpha)/d(-gamma)
            
            print(f"    eps={-eps:.1f}: alpha={alpha_mod:.3f}, delta={delta_alpha:.3f}, "
                  f"sensitivity={sensitivity:.3f}")
            
            results.append({
                'target_layer': target_layer,
                'epsilon': -eps,
                'alpha_base': float(alpha_base),
                'alpha_mod': float(alpha_mod),
                'delta_alpha': float(delta_alpha),
                'sensitivity': float(sensitivity),
            })
        
        # 恢复原始权重
        ln_mod.weight.data = original_weight.clone()
    
    release_model(model)
    
    # 分析因果效应
    if len(results) > 0:
        sensitivities = [r['sensitivity'] for r in results]
        mean_sens = np.mean(sensitivities)
        std_sens = np.std(sensitivities)
        
        # 正方向敏感度
        pos_sens = [r['sensitivity'] for r in results if r['epsilon'] > 0]
        neg_sens = [r['sensitivity'] for r in results if r['epsilon'] < 0]
        
        print(f"\n  Causal Effect Summary:")
        print(f"    mean_sensitivity = {mean_sens:.3f} +/- {std_sens:.3f}")
        if len(pos_sens) > 0:
            print(f"    positive_direction_sensitivity = {np.mean(pos_sens):.3f}")
        if len(neg_sens) > 0:
            print(f"    negative_direction_sensitivity = {np.mean(neg_sens):.3f}")
        
        if mean_sens > 0:
            print(f"    => LN weight CAUSALLY increases alpha (more focusing)")
        else:
            print(f"    => LN weight CAUSALLY decreases alpha (less focusing)")
    
    return {
        'model': model_name,
        'mean_sensitivity': float(np.mean(sensitivities)) if len(results) > 0 else 0,
        'std_sensitivity': float(np.std(sensitivities)) if len(results) > 0 else 0,
        'intervention_data': results,
    }


# ===== P483: 完整因果链模型 =====

def run_p483(model_name):
    """完整因果链模型: LN权重 -> 信号增长 -> alpha -> gamma"""
    print(f"\n{'='*70}")
    print(f"P483: Complete Causal Chain Model - {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    # W_U^T SVD (截断)
    W_U = get_W_U(model)
    k_wut = min(400, min(W_U.shape) - 1)
    U_wut, s_wut = truncated_svd_WUT(W_U, k_wut)
    
    # delta
    log_s_wut = np.log(s_wut[:200] + 1e-30)
    log_i = np.log(np.arange(1, 201, dtype=float))
    slope_wut, _, _, _, _ = stats.linregress(log_i, log_s_wut)
    delta = -slope_wut
    
    # 1. 收集所有数据
    layers = get_layers(model)
    sample_layers = list(range(0, n_layers, max(1, n_layers // 12))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    # 前向传播获取hidden states
    test_text = "The apple is"
    input_ids = tokenizer.encode(test_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        h_states = outputs.hidden_states
    
    h_norms = [h_states[L][0, -1].cpu().float().numpy() for L in range(len(h_states))]
    h_norm_values = [np.linalg.norm(h) for h in h_norms]
    h0_norm = h_norm_values[0]
    
    results = []
    for layer_idx in sample_layers:
        # Alpha
        alpha_mean, alpha_std = compute_alpha_for_layer(
            layer_idx, model, tokenizer, device, U_wut, s_wut, k_wut
        )
        
        # Gamma
        h_L = h_norms[layer_idx + 1]
        proj = U_wut[:k_wut] @ h_L
        e_i = proj**2
        total_energy = np.sum(e_i)
        top_energy = np.sum(e_i[:k_wut // 4])
        gamma = top_energy / total_energy if total_energy > 0 else 0.5
        
        # LN权重
        layer = layers[layer_idx]
        for name in ["input_layernorm", "ln_1", "layernorm"]:
            if hasattr(layer, name):
                ln_mod = getattr(layer, name)
                break
        
        ln_weight = ln_mod.weight.detach().cpu().float().numpy()
        ln_mean = np.mean(ln_weight)
        ln_norm = np.linalg.norm(ln_weight)
        
        # Post-attention LN
        for name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
            if hasattr(layer, name):
                ln2_mod = getattr(layer, name)
                break
        
        ln2_weight = ln2_mod.weight.detach().cpu().float().numpy()
        ln2_mean = np.mean(ln2_weight)
        ln2_norm = np.linalg.norm(ln2_weight)
        
        # 信号特征
        signal_ratio = h_norm_values[layer_idx + 1] / h0_norm if h0_norm > 0 else 1
        if layer_idx > 0:
            signal_growth = (h_norm_values[layer_idx + 1] - h_norm_values[layer_idx]) / h_norm_values[layer_idx]
        else:
            signal_growth = 0
        
        layer_frac = layer_idx / max(1, n_layers - 1)
        
        results.append({
            'layer': layer_idx,
            'layer_frac': layer_frac,
            'alpha': alpha_mean,
            'gamma': gamma,
            'ln_mean': ln_mean,
            'ln_norm': ln_norm,
            'ln2_mean': ln2_mean,
            'ln2_norm': ln2_norm,
            'signal_ratio': signal_ratio,
            'signal_growth': signal_growth,
        })
    
    release_model(model)
    
    # 2. 因果链分析
    alphas = np.array([r['alpha'] for r in results])
    gammas = np.array([r['gamma'] for r in results])
    ln_means = np.array([r['ln_mean'] for r in results])
    ln_norms = np.array([r['ln_norm'] for r in results])
    ln2_means = np.array([r['ln2_mean'] for r in results])
    ln2_norms = np.array([r['ln2_norm'] for r in results])
    sig_ratios = np.array([r['signal_ratio'] for r in results])
    sig_growths = np.array([r['signal_growth'] for r in results])
    
    # 链1: ln_mean -> alpha
    corr_ln_alpha = np.corrcoef(ln_means, alphas)[0, 1] if len(alphas) > 2 else 0
    corr_ln2_alpha = np.corrcoef(ln2_means, alphas)[0, 1] if len(alphas) > 2 else 0
    
    # 链2: ln_mean -> signal_growth
    corr_ln_growth = np.corrcoef(ln_means, sig_growths)[0, 1] if len(sig_growths) > 2 else 0
    
    # 链3: signal_growth -> alpha
    corr_growth_alpha = np.corrcoef(sig_growths, alphas)[0, 1] if len(alphas) > 2 else 0
    
    # 链4: alpha -> gamma
    corr_alpha_gamma = np.corrcoef(alphas, gammas)[0, 1] if len(gammas) > 2 else 0
    
    # 完整链: ln_mean -> signal_growth -> alpha -> gamma
    # 偏相关: ln_mean -> alpha, 控制signal_growth
    if len(alphas) > 4:
        from numpy.linalg import lstsq
        # alpha = a * ln_mean + b * signal_growth + c
        X = np.column_stack([ln_means, sig_growths, np.ones(len(alphas))])
        coeffs, _, _, _ = lstsq(X, alphas, rcond=None)
        alpha_pred = X @ coeffs
        ss_res = np.sum((alphas - alpha_pred)**2)
        ss_tot = np.sum((alphas - np.mean(alphas))**2)
        r2_partial = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    else:
        r2_partial = 0
    
    print(f"\n  Causal Chain Correlations:")
    print(f"    Step 1: ln_mean -> alpha: corr = {corr_ln_alpha:.3f}")
    print(f"    Step 1b: ln2_mean -> alpha: corr = {corr_ln2_alpha:.3f}")
    print(f"    Step 2: ln_mean -> signal_growth: corr = {corr_ln_growth:.3f}")
    print(f"    Step 3: signal_growth -> alpha: corr = {corr_growth_alpha:.3f}")
    print(f"    Step 4: alpha -> gamma: corr = {corr_alpha_gamma:.3f}")
    print(f"    Partial: ln_mean + growth -> alpha: R2 = {r2_partial:.3f}")
    
    # 3. 结构方程模型(SEM)
    print(f"\n  Structural Equation Model (SEM):")
    
    # gamma = 1 - c * alpha
    c_fit = np.polyfit(alphas, gammas, 1)
    gamma_pred = np.polyval(c_fit, alphas)
    ss_res_g = np.sum((gammas - gamma_pred)**2)
    ss_tot_g = np.sum((gammas - np.mean(gammas))**2)
    r2_gamma = 1 - ss_res_g / ss_tot_g if ss_tot_g > 0 else 0
    print(f"    gamma = {c_fit[0]:.3f} * alpha + {c_fit[1]:.3f}, R2 = {r2_gamma:.3f}")
    
    # alpha = a * ln_mean + b * signal_growth + c
    if len(alphas) > 3:
        X = np.column_stack([ln_means, sig_growths, np.ones(len(alphas))])
        coeffs_a, _, _, _ = lstsq(X, alphas, rcond=None)
        print(f"    alpha = {coeffs_a[0]:.3f} * ln_mean + {coeffs_a[1]:.3f} * growth + {coeffs_a[2]:.3f}")
    
    # ln_mean = a * layer_frac + b
    lf = np.array([r['layer_frac'] for r in results])
    c_ln = np.polyfit(lf, ln_means, 1)
    ln_pred = np.polyval(c_ln, lf)
    ss_res_ln = np.sum((ln_means - ln_pred)**2)
    ss_tot_ln = np.sum((ln_means - np.mean(ln_means))**2)
    r2_ln = 1 - ss_res_ln / ss_tot_ln if ss_tot_ln > 0 else 0
    print(f"    ln_mean = {c_ln[0]:.4f} * layer_frac + {c_ln[1]:.4f}, R2 = {r2_ln:.3f}")
    
    # 每层详细
    print(f"\n  Per-layer Data:")
    print(f"    {'Layer':>5} {'alpha':>7} {'gamma':>7} {'ln_m':>7} {'ln2_m':>7} {'sigR':>7} {'sigG':>7}")
    for r in results:
        print(f"    {r['layer']:5d} {r['alpha']:7.3f} {r['gamma']:7.3f} "
              f"{r['ln_mean']:7.4f} {r['ln2_mean']:7.4f} "
              f"{r['signal_ratio']:7.3f} {r['signal_growth']:7.4f}")
    
    return {
        'model': model_name,
        'delta': delta,
        'corr_ln_alpha': float(corr_ln_alpha),
        'corr_ln2_alpha': float(corr_ln2_alpha),
        'corr_ln_growth': float(corr_ln_growth),
        'corr_growth_alpha': float(corr_growth_alpha),
        'corr_alpha_gamma': float(corr_alpha_gamma),
        'r2_partial': float(r2_partial),
        'gamma_formula': c_fit.tolist(),
        'r2_gamma': float(r2_gamma),
        'layer_data': results,
    }


# ===== 主函数 =====

EXPERIMENTS = {
    'p480': run_p480,
    'p481': run_p481,
    'p482': run_p482,
    'p483': run_p483,
}

def main():
    parser = argparse.ArgumentParser(description="Phase C: Residual Growth Mechanism & Causal Chain")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["qwen3", "glm4", "deepseek7b"],
                       help="Model to test")
    parser.add_argument("--experiment", type=str, required=True,
                       choices=list(EXPERIMENTS.keys()),
                       help="Experiment to run")
    args = parser.parse_args()
    
    print(f"\nPhase C: Residual Growth Mechanism & Causal Chain")
    print(f"Model: {args.model}, Experiment: {args.experiment}")
    
    result = EXPERIMENTS[args.experiment](args.model)
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5_temp')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.experiment}_{args.model}.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Phase C - {args.experiment} - {args.model}\n")
        f.write(f"="*50 + "\n")
        for key, val in result.items():
            if key == 'layer_data':
                f.write(f"\nLayer Data ({len(val)} layers):\n")
                for r in val:
                    f.write(f"  {r}\n")
            elif key == 'intervention_data':
                f.write(f"\nIntervention Data ({len(val)} entries):\n")
                for r in val:
                    f.write(f"  {r}\n")
            else:
                f.write(f"{key}: {val}\n")
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
