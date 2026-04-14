"""
Phase CV-P499/P500/P501: Jacobian链与信号传播增益
==================================================

Phase CIV核心瓶颈:
- 输出端因果链后半段已验证: Δh_final→Δlogit(r=0.96-1.00), Δh_final→Δkl_div(r=0.70-0.90)
- 但前半段断裂: gain_l = ||Δh_final|| / (alpha × ||W_down||) 不可预测
- gain与层位置/谱特征(PR, kappa, top10)均无关(r<0.4)

Phase CV核心思路:
gain_l = ||J_L × J_{L-1} × ... × J_{l+1}||, Jacobian链决定信号传播增益

P499: 逐层Jacobian测量
  - 计算每层的Jacobian J_l = ∂h_l/∂h_{l-1} (有限差分法)
  - 计算Jacobian链范数: ||J_{L:l}|| = ||J_L × J_{L-1} × ... × J_{l+1}||
  - 验证: gain_l 是否与 ||J_{L:l}|| 相关?
  - 分析: Jacobian的条件数、奇异值谱随层如何变化?

P500: Jacobian谱结构与层位置
  - 分析每层Jacobian的奇异值谱
  - Jacobian条件数cond(J_l)是否与层位置相关?
  - 深层的Jacobian是否更"各向异性"?
  - 残差连接的贡献: J_l ≈ I + ∂(attn+ffn)/∂h_{l-1}

P501: 从Jacobian链推导importance公式
  - 如果gain_l = f(||J_{L:l}||), 则importance_l ≈ ||J_{L:l}|| × ||W_down_l|| × ||h_l||
  - 验证这个公式是否与实测importance(importance_ppl)相关
  - 目标: R2>0.5

使用方法:
    python phase_cv_jacobian_chain.py --model qwen3 --experiment p499
    python phase_cv_jacobian_chain.py --model glm4 --experiment p500
    python phase_cv_jacobian_chain.py --model deepseek7b --experiment p501
    python phase_cv_jacobian_chain.py --model qwen3 --experiment all
"""

import sys
import os
import argparse
import numpy as np
import torch
import json
import time
from scipy.stats import spearmanr, pearsonr
from collections import namedtuple

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, 'tests', 'glm5'))

from model_utils import (
    load_model, get_model_info, get_W_U, get_layer_weights,
    get_sample_layers, get_layers,
)


# ============================================================
# 工具函数
# ============================================================

def compute_participation_ratio(s):
    """参与比 = (sum s^2)^2 / (n * sum s^4)"""
    s_sq = s**2
    s_sq_norm = s_sq / (np.sum(s_sq) + 1e-30)
    return 1.0 / (len(s) * np.sum(s_sq_norm**2) + 1e-30)


def compute_effective_dimension(s):
    """有效维度 = (sum s)^2 / (n * sum s^2)"""
    return (np.sum(s)**2) / (len(s) * np.sum(s**2) + 1e-30)


def compute_kl_divergence(logits_baseline, logits_ablated):
    """计算KL散度"""
    p = torch.nn.functional.softmax(logits_baseline, dim=-1)
    q = torch.nn.functional.log_softmax(logits_ablated, dim=-1)
    kl = torch.nn.functional.kl_div(q, p, reduction='batchmean')
    return kl.item()


def compute_residual_stream(model, input_ids, device):
    """用hook收集各层hidden state"""
    hidden_states = []
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_states.append(output[0].detach())
        else:
            hidden_states.append(output.detach())
    
    hooks = []
    layers_list = get_layers(model)
    for layer in layers_list:
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
        result = []
        for hs in hidden_states:
            result.append(hs[0, -1].cpu().float().numpy())
        return result
    return None


def perturb_w_down(layers, l_idx, alpha, mlp_type):
    """对W_down施加扰动: W_down → W_down * (1 - alpha), 返回原始权重"""
    orig_weight = layers[l_idx].mlp.down_proj.weight.data.clone()
    layers[l_idx].mlp.down_proj.weight.data = orig_weight * (1 - alpha)
    return orig_weight


def restore_w_down(layers, l_idx, orig_weight, mlp_type):
    """恢复W_down权重"""
    layers[l_idx].mlp.down_proj.weight.data = orig_weight


def compute_importance_measures(model, tokenizer, device, text, layers, l_idx, alpha, mlp_type):
    """计算消融W_down后的所有重要性指标"""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        baseline_outputs = model(input_ids)
        if hasattr(baseline_outputs, 'logits'):
            baseline_logits = baseline_outputs.logits
        elif isinstance(baseline_outputs, torch.Tensor):
            baseline_logits = baseline_outputs
        else:
            return None
    
    if baseline_logits.dim() == 3:
        baseline_logits = baseline_logits[0]
    
    h_all_baseline = compute_residual_stream(model, input_ids, device)
    if h_all_baseline is None:
        return None
    h_final_baseline = h_all_baseline[-1]
    
    baseline_next_tokens = input_ids[0, 1:]
    baseline_pred_logits = baseline_logits[:-1]
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    baseline_loss = loss_fn(baseline_pred_logits, baseline_next_tokens).item()
    baseline_ppl = np.exp(min(baseline_loss, 20))
    
    orig_weight = perturb_w_down(layers, l_idx, alpha, mlp_type)
    
    with torch.no_grad():
        ablated_outputs = model(input_ids)
        if hasattr(ablated_outputs, 'logits'):
            ablated_logits = ablated_outputs.logits
        elif isinstance(ablated_outputs, torch.Tensor):
            ablated_logits = ablated_outputs
        else:
            restore_w_down(layers, l_idx, orig_weight, mlp_type)
            return None
    
    if ablated_logits.dim() == 3:
        ablated_logits = ablated_logits[0]
    
    h_all_ablated = compute_residual_stream(model, input_ids, device)
    h_final_ablated = h_all_ablated[-1] if h_all_ablated is not None else None
    
    restore_w_down(layers, l_idx, orig_weight, mlp_type)
    
    ablated_pred_logits = ablated_logits[:-1]
    ablated_loss = loss_fn(ablated_pred_logits, baseline_next_tokens).item()
    ablated_ppl = np.exp(min(ablated_loss, 20))
    
    delta_ppl = ablated_ppl - baseline_ppl
    importance_ppl = delta_ppl / max(baseline_ppl, 1e-10)
    kl_div = compute_kl_divergence(baseline_pred_logits, ablated_pred_logits)
    
    delta_h_final = h_final_ablated - h_final_baseline if h_final_ablated is not None else None
    delta_h_norm = np.linalg.norm(delta_h_final) if delta_h_final is not None else 0
    h_norm = np.linalg.norm(h_final_baseline)
    
    delta_logits = ablated_logits - baseline_logits
    delta_logits_norm = torch.norm(delta_logits).item()
    
    return {
        'delta_h_final': delta_h_final,
        'delta_h_norm': delta_h_norm,
        'h_norm': h_norm,
        'delta_h_relative': delta_h_norm / max(h_norm, 1e-10),
        'delta_logits_norm': delta_logits_norm,
        'baseline_ppl': baseline_ppl,
        'ablated_ppl': ablated_ppl,
        'delta_ppl': delta_ppl,
        'importance_ppl': importance_ppl,
        'kl_div': kl_div,
        'alpha': alpha,
        'l_idx': l_idx,
    }


# ============================================================
# Jacobian计算核心函数
# ============================================================

def compute_layer_jacobian(model, tokenizer, device, text, l_idx, eps=1e-3, n_directions=50):
    """
    计算单层Jacobian J_l = ∂h_l/∂h_{l-1} 的近似
    
    方法: 有限差分法
    - 对h_{l-1}施加小扰动δ, 测量h_l的变化
    - J_l ≈ (h_l(h_{l-1}+εv) - h_l(h_{l-1}-εv)) / (2ε)
    
    但直接计算完整Jacobian矩阵[d_model, d_model]太大
    → 用随机方向近似: J_l v ≈ (Δh_l) / ε
    
    这里测量的是Jacobian在随机方向上的投影,用来估计Jacobian的范数
    
    更高效的方法:
    直接测量"Jacobian的谱范数" = max singular value of J_l
    ≈ max ||J_l v|| / ||v|| over random v
    
    返回:
    - jacobian_norm: Jacobian的近似谱范数
    - jacobian_trace_approx: Jacobian的近似迹(用随机方向估计)
    - residual_norm_ratio: ||J_l v|| / ||v|| 的统计量
    """
    # 使用hook来截获和修改中间隐状态
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # 先获取基线隐状态
    h_all_baseline = compute_residual_stream(model, input_ids, device)
    if h_all_baseline is None or l_idx >= len(h_all_baseline):
        return None
    
    h_prev_baseline = h_all_baseline[l_idx]  # 层l的输入=层l-1的输出
    h_curr_baseline = h_all_baseline[l_idx]   # 层l的输出
    
    d_model = len(h_prev_baseline)
    
    # 生成随机方向
    np.random.seed(42 + l_idx)
    directions = np.random.randn(n_directions, d_model)
    for i in range(n_directions):
        directions[i] /= np.linalg.norm(directions[i])
    
    # 对每个方向, 扰动h_{l-1}并测量h_l的变化
    # 这需要用hook来修改中间隐状态
    # 但由于hook机制的限制, 我们用另一种方法:
    # 直接用W_down扰动+已知h_l来反推Jacobian的影响
    
    # 更实际的方法: 测量"层间Jacobian"的谱范数
    # 通过随机扰动输入, 测量相邻层输出的变化比
    
    jacobian_norms = []
    
    for d_idx in range(n_directions):
        # 这个方法需要修改中间隐状态, 在transformer中不太方便
        # 改用替代方案: 通过两步前向传播来估计
        pass
    
    # === 替代方案: 用残差结构直接计算Jacobian的近似 ===
    # 残差连接: h_l = h_{l-1} + attn(h_{l-1}) + ffn(h_{l-1})
    # Jacobian: J_l = I + ∂attn/∂h_{l-1} + ∂ffn/∂h_{l-1}
    # 对于W_down相关的Jacobian, 关键是FFN部分的Jacobian
    
    # 最实用的方法: 直接测量从层l到最后一层的信号传播
    # 这就是P496中的gain! 但我们需要更精细的分析
    
    return None  # 这个方法不实际, 改用下面的run_p499


def compute_jacobian_chain_norm_numerically(
    model, tokenizer, device, text, layers, l_idx, 
    mlp_type, n_layers, d_model, alpha=0.05, n_perturb=20
):
    """
    数值计算Jacobian链范数: ||J_{L:l}|| = ||∂h_final/∂h_l||
    
    方法:
    1. 基线前向传播, 获取h_l(层l的输出)和h_final
    2. 对h_l施加随机小扰动v (||v|| = eps)
    3. 从层l+1开始继续前向传播, 获取h_final'
    4. ||J_{L:l}|| ≈ ||h_final' - h_final|| / eps
    
    问题: 如何从中间层开始前向传播?
    在标准transformer中不容易做到。
    
    替代方案(更简单):
    1. 对W_down_l施加小扰动alpha
    2. 测量各层h的变化: Δh_l, Δh_{l+1}, ..., Δh_L
    3. 逐层传播比: ratio_{k} = ||Δh_{k+1}|| / ||Δh_k||
    4. 总Jacobian链范数: ||J_{L:l}|| ≈ ∏_{k=l}^{L-1} ratio_k
    """
    # 基线
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    h_all_baseline = compute_residual_stream(model, input_ids, device)
    if h_all_baseline is None:
        return None
    
    # 扰动W_down
    orig_weight = perturb_w_down(layers, l_idx, alpha, mlp_type)
    
    h_all_perturbed = compute_residual_stream(model, input_ids, device)
    
    restore_w_down(layers, l_idx, orig_weight, mlp_type)
    
    if h_all_perturbed is None:
        return None
    
    # 计算每层的变化量
    delta_h_per_layer = []
    for k in range(len(h_all_baseline)):
        dh = np.linalg.norm(h_all_perturbed[k] - h_all_baseline[k])
        h_norm = max(np.linalg.norm(h_all_baseline[k]), 1e-10)
        delta_h_per_layer.append({
            'layer': k,
            'delta_h_norm': dh,
            'delta_h_relative': dh / h_norm,
            'h_norm': h_norm,
        })
    
    # 逐层传播比
    propagation_ratios = []
    for k in range(l_idx, len(delta_h_per_layer) - 1):
        dh_curr = max(delta_h_per_layer[k]['delta_h_norm'], 1e-30)
        dh_next = delta_h_per_layer[k + 1]['delta_h_norm']
        ratio = dh_next / dh_curr
        propagation_ratios.append({
            'from_layer': k,
            'to_layer': k + 1,
            'ratio': ratio,
            'dh_curr': delta_h_per_layer[k]['delta_h_norm'],
            'dh_next': delta_h_per_layer[k + 1]['delta_h_norm'],
        })
    
    # Jacobian链范数(用累积乘积)
    # ||J_{L:l}|| ≈ ||Δh_L|| / ||Δh_l||
    dh_l = max(delta_h_per_layer[l_idx]['delta_h_norm'], 1e-30)
    dh_L = delta_h_per_layer[-1]['delta_h_norm']
    jacobian_chain_norm = dh_L / dh_l
    
    # 也计算几何平均传播比
    if propagation_ratios:
        log_ratios = [np.log(max(r['ratio'], 1e-30)) for r in propagation_ratios]
        geo_mean_ratio = np.exp(np.mean(log_ratios))
    else:
        geo_mean_ratio = 1.0
    
    return {
        'l_idx': l_idx,
        'alpha': alpha,
        'delta_h_per_layer': delta_h_per_layer,
        'propagation_ratios': propagation_ratios,
        'jacobian_chain_norm': jacobian_chain_norm,
        'geo_mean_ratio': geo_mean_ratio,
        'dh_l': dh_l,
        'dh_L': dh_L,
    }


def compute_inter_layer_jacobian_norm(
    model, tokenizer, device, text, layers, l_idx,
    mlp_type, n_layers, d_model, alpha=0.01
):
    """
    计算相邻层间的Jacobian范数: ||J_l|| = ||∂h_l/∂h_{l-1}||
    
    方法: 对h_{l-1}(即层l-1的输出)施加小扰动, 测量h_l的变化
    
    由于直接修改中间隐状态困难, 使用替代方法:
    对层l-1的W_down施加扰动, 测量层l和层l-1的变化比
    
    更精确的方法: 对输入embedding施加不同方向的扰动
    但这只能测量从输入到每层的Jacobian, 不能测量层间Jacobian
    
    最实用的方法:
    使用两层差分法:
    - 扰动层l的输入(W_down_{l-1}扰动), 得到Δh_{l-1}和Δh_l
    - J_l ≈ Δh_l / Δh_{l-1} (范数比)
    
    但这假设扰动方向与Jacobian的主奇异向量对齐, 不太精确
    
    更好的方法: 
    直接用理论公式计算Jacobian的近似
    残差结构: h_l = LN(h_{l-1} + attn(h_{l-1}) + ffn(h_{l-1}))
    
    在LN(层归一化)下, Jacobian有一个简化形式:
    J_l ≈ LN' × (I + W_attn + W_ffn)
    
    其中LN'是LayerNorm的Jacobian(已知解析形式)
    """
    # 实用方法: 对层l-1扰动, 测量层l的变化
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # 基线
    h_all_baseline = compute_residual_stream(model, input_ids, device)
    if h_all_baseline is None:
        return None
    
    # 扰动层l-1的W_down
    if l_idx < 1:
        return None
    
    orig_weight = perturb_w_down(layers, l_idx - 1, alpha, mlp_type)
    h_all_perturbed = compute_residual_stream(model, input_ids, device)
    restore_w_down(layers, l_idx - 1, orig_weight, mlp_type)
    
    if h_all_perturbed is None:
        return None
    
    # Jacobian范数估计
    dh_prev = np.linalg.norm(h_all_perturbed[l_idx - 1] - h_all_baseline[l_idx - 1])
    dh_curr = np.linalg.norm(h_all_perturbed[l_idx] - h_all_baseline[l_idx])
    
    if dh_prev < 1e-30:
        return None
    
    jacobian_norm = dh_curr / dh_prev
    
    return {
        'l_idx': l_idx,
        'alpha': alpha,
        'dh_prev': dh_prev,
        'dh_curr': dh_curr,
        'jacobian_norm': jacobian_norm,
    }


# ============================================================
# P499: 逐层Jacobian测量 - Jacobian链与gain的关系
# ============================================================

def run_p499(model_name, device):
    """
    P499: 逐层Jacobian测量
    
    核心问题: gain_l = ||Δh_final|| / (alpha × ||W_down||) 为什么不可预测?
    假设: gain_l = ||J_{L:l}|| × f(W_down, h_l), Jacobian链决定传播增益
    
    方法:
    1. 对采样层的W_down施加小扰动alpha
    2. 收集所有层的隐状态变化Δh_k (k=l,l+1,...,L)
    3. 计算逐层传播比: ratio_k = ||Δh_{k+1}|| / ||Δh_k||
    4. 计算Jacobian链范数: ||J_{L:l}|| = ||Δh_L|| / ||Δh_l||
    5. 验证: gain_l vs ||J_{L:l}|| 是否相关?
    6. 分析: 传播比ratio_k是否与层位置相关?
    """
    print(f"\n{'='*60}")
    print(f"P499: 逐层Jacobian测量 [{model_name}]")
    print(f"{'='*60}")
    
    model, tokenizer, model_device = load_model(model_name)
    if model_device != device:
        model = model.to(device)
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    mlp_type = info.mlp_type
    
    layers_list = get_layers(model)
    
    test_text = "The apple is red and sweet, and it grows on trees in the garden."
    alpha = 0.1
    
    # 采样层(只采样前面和中间的层, 因为最后几层没有后续层传播)
    sample_layers = get_sample_layers(n_layers, 10)
    # 排除最后2层(没有足够的后续层来计算Jacobian链)
    sample_layers = [l for l in sample_layers if l < n_layers - 2]
    print(f"  采样层: {sample_layers}")
    
    results = []
    
    for l_idx in sample_layers:
        layer_frac = l_idx / max(n_layers - 1, 1)
        print(f"  层L{l_idx}...")
        
        # 获取W_down谱特征
        lw = get_layer_weights(layers_list[l_idx], d_model, mlp_type)
        W_down = lw.W_down
        W_down_norm = np.linalg.norm(W_down)
        
        # 1. 计算Jacobian链(从层l到最后一层)
        jacobian_result = compute_jacobian_chain_norm_numerically(
            model, tokenizer, device, test_text,
            layers_list, l_idx, mlp_type, n_layers, d_model,
            alpha=alpha
        )
        
        if jacobian_result is None:
            print(f"    L{l_idx} Jacobian链计算失败")
            continue
        
        # 2. 计算importance指标(与P496相同的扰动)
        meas = compute_importance_measures(
            model, tokenizer, device, test_text,
            layers_list, l_idx, alpha, mlp_type
        )
        
        if meas is None:
            print(f"    L{l_idx} importance计算失败")
            continue
        
        # 3. 计算gain
        gain = meas['delta_h_norm'] / max(alpha * W_down_norm, 1e-10)
        
        # 4. 提取逐层传播比的统计量
        prop_ratios = jacobian_result['propagation_ratios']
        ratio_values = [r['ratio'] for r in prop_ratios if l_idx <= r['from_layer']]
        mean_ratio = np.mean(ratio_values) if ratio_values else 1.0
        std_ratio = np.std(ratio_values) if len(ratio_values) > 1 else 0.0
        max_ratio = np.max(ratio_values) if ratio_values else 1.0
        min_ratio = np.min(ratio_values) if ratio_values else 1.0
        
        # 5. 找到"放大层"和"衰减层"
        n_amplifying = sum(1 for r in ratio_values if r > 1.0)
        n_damping = sum(1 for r in ratio_values if r < 1.0)
        n_prop_layers = len(ratio_values)
        
        result = {
            'layer': l_idx,
            'layer_frac': layer_frac,
            'alpha': alpha,
            'gain': gain,
            'jacobian_chain_norm': jacobian_result['jacobian_chain_norm'],
            'geo_mean_ratio': jacobian_result['geo_mean_ratio'],
            'dh_l': jacobian_result['dh_l'],
            'dh_L': jacobian_result['dh_L'],
            'mean_ratio': mean_ratio,
            'std_ratio': std_ratio,
            'max_ratio': max_ratio,
            'min_ratio': min_ratio,
            'n_amplifying': n_amplifying,
            'n_damping': n_damping,
            'n_prop_layers': n_prop_layers,
            'importance_ppl': meas['importance_ppl'],
            'kl_div': meas['kl_div'],
            'delta_h_relative': meas['delta_h_relative'],
            'W_down_norm': W_down_norm,
        }
        results.append(result)
        
        print(f"    gain={gain:.3f}, J_chain_norm={jacobian_result['jacobian_chain_norm']:.3f}, "
              f"geo_mean={jacobian_result['geo_mean_ratio']:.3f}, "
              f"mean_ratio={mean_ratio:.3f}, n_amp={n_amplifying}/{n_prop_layers}")
    
    # 释放模型
    del model
    torch.cuda.empty_cache()
    
    # 统计分析
    if len(results) < 5:
        print("  数据不足, 无法统计分析")
        return results
    
    print(f"\n--- P499 统计分析 [{model_name}] ---")
    
    gains = [r['gain'] for r in results]
    j_chain_norms = [r['jacobian_chain_norm'] for r in results]
    geo_means = [r['geo_mean_ratio'] for r in results]
    lfracs = [r['layer_frac'] for r in results]
    importances = [r['importance_ppl'] for r in results]
    
    # 1. gain vs jacobian_chain_norm (核心!)
    r1, p1 = spearmanr(gains, j_chain_norms)
    print(f"  gain vs ||J_{{L:l}}||: r={r1:.3f}, p={p1:.4f}")
    
    # 2. gain vs geo_mean_ratio
    r2, p2 = spearmanr(gains, geo_means)
    print(f"  gain vs geo_mean_ratio: r={r2:.3f}, p={p2:.4f}")
    
    # 3. gain vs layer_frac
    r3, p3 = spearmanr(gains, lfracs)
    print(f"  gain vs layer_frac: r={r3:.3f}, p={p3:.4f}")
    
    # 4. jacobian_chain_norm vs layer_frac
    r4, p4 = spearmanr(j_chain_norms, lfracs)
    print(f"  ||J_{{L:l}}|| vs layer_frac: r={r4:.3f}, p={p4:.4f}")
    
    # 5. importance_ppl vs jacobian_chain_norm (直接预测!)
    r5, p5 = spearmanr(importances, j_chain_norms)
    print(f"  importance_ppl vs ||J_{{L:l}}||: r={r5:.3f}, p={p5:.4f}")
    
    # 6. importance_ppl vs gain
    r6, p6 = spearmanr(importances, gains)
    print(f"  importance_ppl vs gain: r={r6:.3f}, p={p6:.4f}")
    
    # 7. 多元回归: importance ~ f(gain, ||J_{L:l}||, layer_frac, W_down_norm)
    from sklearn.linear_model import LinearRegression, Lasso
    from sklearn.preprocessing import StandardScaler
    
    X_features = []
    y_target = []
    for r in results:
        X_features.append([
            r['jacobian_chain_norm'],
            r['geo_mean_ratio'],
            r['layer_frac'],
            r['W_down_norm'],
            r['gain'],
        ])
        y_target.append(r['importance_ppl'])
    
    if len(X_features) >= 5:
        X = np.array(X_features)
        y = np.array(y_target)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        
        # OLS
        ols = LinearRegression()
        ols.fit(X_s, y)
        y_pred = ols.predict(X_s)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        
        print(f"\n  importance ~ f(J_chain, geo_mean, lfrac, W_down_norm, gain): R2={r2:.3f}")
        feat_names = ['J_chain_norm', 'geo_mean_ratio', 'layer_frac', 'W_down_norm', 'gain']
        for fn, c in zip(feat_names, ols.coef_):
            print(f"    {fn}: {c:.4f}")
        
        # Lasso
        lasso = Lasso(alpha=0.01)
        lasso.fit(X_s, y)
        y_pred_l = lasso.predict(X_s)
        ss_res_l = np.sum((y - y_pred_l)**2)
        r2_l = 1 - ss_res_l / max(ss_tot, 1e-10)
        print(f"  Lasso: R2={r2_l:.3f}")
        for fn, c in zip(feat_names, lasso.coef_):
            print(f"    {fn}: {c:.4f}")
    
    # 8. 传播比的统计分析
    all_ratios = []
    for r in results:
        # 需要重新获取详细的传播比数据
        pass
    
    print(f"\n  逐层传播比统计:")
    for r in results:
        print(f"    L{r['layer']}: mean_ratio={r['mean_ratio']:.3f}, "
              f"n_amp={r['n_amplifying']}/{r['n_prop_layers']}, "
              f"max_ratio={r['max_ratio']:.3f}")
    
    # 保存结果
    out_dir = os.path.join(project_root, 'tests', 'glm5_temp')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"p499_{model_name}.json")
    
    # 清理不可序列化的数据
    results_clean = []
    for r in results:
        rc = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
        results_clean.append(rc)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results_clean, f, indent=2, ensure_ascii=False, default=str)
    print(f"  结果已保存: {out_path}")
    
    return results


# ============================================================
# P500: Jacobian谱结构与层位置
# ============================================================

def run_p500(model_name, device):
    """
    P500: Jacobian谱结构与层位置
    
    核心问题: 逐层传播比ratio_k = ||Δh_{k+1}|| / ||Δh_k|| 由什么决定?
    假设: 比率由层k的Jacobian的谱特征决定
    
    方法:
    1. 对每层k, 用有限差分法估计J_k的谱范数
    2. 计算J_k的条件数(最大/最小奇异值比)
    3. 分析这些量是否与层位置相关
    4. 特别关注: 深层的Jacobian是否更"各向异性"?
    5. 残差连接的贡献: J_k = I + ΔJ_k, ΔJ_k的范数如何?
    """
    print(f"\n{'='*60}")
    print(f"P500: Jacobian谱结构与层位置 [{model_name}]")
    print(f"{'='*60}")
    
    model, tokenizer, model_device = load_model(model_name)
    if model_device != device:
        model = model.to(device)
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    mlp_type = info.mlp_type
    
    layers_list = get_layers(model)
    
    test_text = "The apple is red and sweet, and it grows on trees in the garden."
    alpha = 0.05  # 更小的扰动, 更精确的Jacobian估计
    
    # 对所有层计算层间Jacobian
    sample_layers = get_sample_layers(n_layers, 15)
    # 排除第0层(没有前驱层)
    sample_layers = [l for l in sample_layers if l >= 1]
    print(f"  采样层: {sample_layers}")
    
    results = []
    
    for l_idx in sample_layers:
        layer_frac = l_idx / max(n_layers - 1, 1)
        print(f"  层L{l_idx}...", end="", flush=True)
        
        # 1. 计算层间Jacobian范数(层l的Jacobian: ∂h_l/∂h_{l-1})
        jacobian_result = compute_inter_layer_jacobian_norm(
            model, tokenizer, device, test_text,
            layers_list, l_idx, mlp_type, n_layers, d_model,
            alpha=alpha
        )
        
        if jacobian_result is None:
            print(" 失败")
            continue
        
        # 2. 获取层l的权重和谱特征
        lw = get_layer_weights(layers_list[l_idx], d_model, mlp_type)
        W_down = lw.W_down
        W_down_norm = np.linalg.norm(W_down)
        W_up = lw.W_up
        W_up_norm = np.linalg.norm(W_up) if W_up is not None else 0
        W_gate = lw.W_gate
        W_gate_norm = np.linalg.norm(W_gate) if W_gate is not None else 0
        
        # 3. 获取LayerNorm权重
        ln_weight = lw.input_layernorm_weight
        ln_norm = np.linalg.norm(ln_weight) if ln_weight is not None else 0
        
        # 4. 理论分析: 残差连接的Jacobian
        # J_l = I + ∂attn/∂h + ∂ffn/∂h
        # 近似: ||J_l|| ≈ ||I|| + ||∂attn/∂h|| + ||∂ffn/∂h||
        # 但这不是精确的(矩阵范数不满足三角不等式的等号)
        
        # 5. 用W_down和W_up估计FFN的Jacobian贡献
        # ffn(h) = W_down @ act(W_gate @ h) * (W_up @ h)
        # ∂ffn/∂h ≈ W_down @ diag(act'(·)) @ W_gate + W_down @ act(·) @ W_up
        # 简化估计: ||∂ffn/∂h|| ≤ ||W_down|| × (||W_gate|| + ||W_up||)
        
        # 6. 获取基线隐状态的范数
        inputs = tokenizer(test_text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        h_all = compute_residual_stream(model, input_ids, device)
        
        if h_all is None:
            print(" h_all=None")
            continue
        
        h_prev_norm = np.linalg.norm(h_all[l_idx - 1]) if l_idx > 0 else 0
        h_curr_norm = np.linalg.norm(h_all[l_idx])
        h_ratio = h_curr_norm / max(h_prev_norm, 1e-10)
        
        # 7. 计算残差流的变化
        # h_l - h_{l-1} = attn(h_{l-1}) + ffn(h_{l-1})
        delta_h_residual = np.linalg.norm(h_all[l_idx] - h_all[l_idx - 1]) if l_idx > 0 else 0
        delta_h_relative = delta_h_residual / max(h_prev_norm, 1e-10)
        
        # 8. 计算隐状态cos相似度
        if l_idx > 0 and h_prev_norm > 1e-10 and h_curr_norm > 1e-10:
            cos_sim = np.dot(h_all[l_idx], h_all[l_idx - 1]) / (h_prev_norm * h_curr_norm)
        else:
            cos_sim = 0
        
        result = {
            'layer': l_idx,
            'layer_frac': layer_frac,
            'jacobian_norm': jacobian_result['jacobian_norm'],
            'W_down_norm': W_down_norm,
            'W_up_norm': W_up_norm,
            'W_gate_norm': W_gate_norm,
            'ln_norm': ln_norm,
            'h_prev_norm': h_prev_norm,
            'h_curr_norm': h_curr_norm,
            'h_ratio': h_ratio,
            'delta_h_residual': delta_h_residual,
            'delta_h_relative': delta_h_relative,
            'cos_sim_h': cos_sim,
            'alpha': alpha,
        }
        results.append(result)
        
        print(f" J_norm={jacobian_result['jacobian_norm']:.3f}, "
              f"h_ratio={h_ratio:.3f}, cos_sim={cos_sim:.3f}, "
              f"Δh_rel={delta_h_relative:.3f}")
    
    # 释放模型
    del model
    torch.cuda.empty_cache()
    
    # 统计分析
    if len(results) < 5:
        print("  数据不足, 无法统计分析")
        return results
    
    print(f"\n--- P500 统计分析 [{model_name}] ---")
    
    j_norms = [r['jacobian_norm'] for r in results]
    lfracs = [r['layer_frac'] for r in results]
    h_ratios = [r['h_ratio'] for r in results]
    cos_sims = [r['cos_sim_h'] for r in results]
    delta_h_rels = [r['delta_h_relative'] for r in results]
    W_down_norms = [r['W_down_norm'] for r in results]
    W_up_norms = [r['W_up_norm'] for r in results]
    h_prev_norms = [r['h_prev_norm'] for r in results]
    
    # 1. Jacobian范数 vs 层位置
    r1, p1 = spearmanr(j_norms, lfracs)
    print(f"  ||J_l|| vs layer_frac: r={r1:.3f}, p={p1:.4f}")
    
    # 2. Jacobian范数 vs h_ratio
    r2, p2 = spearmanr(j_norms, h_ratios)
    print(f"  ||J_l|| vs h_ratio: r={r2:.3f}, p={p2:.4f}")
    
    # 3. Jacobian范数 vs cos_sim
    r3, p3 = spearmanr(j_norms, cos_sims)
    print(f"  ||J_l|| vs cos_sim: r={r3:.3f}, p={p3:.4f}")
    
    # 4. Jacobian范数 vs delta_h_relative
    r4, p4 = spearmanr(j_norms, delta_h_rels)
    print(f"  ||J_l|| vs delta_h_relative: r={r4:.3f}, p={p4:.4f}")
    
    # 5. Jacobian范数 vs W_down_norm
    r5, p5 = spearmanr(j_norms, W_down_norms)
    print(f"  ||J_l|| vs W_down_norm: r={r5:.3f}, p={p5:.4f}")
    
    # 6. Jacobian范数 vs W_up_norm
    r6, p6 = spearmanr(j_norms, W_up_norms)
    print(f"  ||J_l|| vs W_up_norm: r={r6:.3f}, p={p6:.4f}")
    
    # 7. Jacobian范数 vs h_prev_norm (隐状态范数越大,Jacobian越小?)
    r7, p7 = spearmanr(j_norms, h_prev_norms)
    print(f"  ||J_l|| vs h_prev_norm: r={r7:.3f}, p={p7:.4f}")
    
    # 8. 多元回归: ||J_l|| ~ f(W_down_norm, W_up_norm, h_prev_norm, layer_frac)
    from sklearn.linear_model import LinearRegression, Lasso
    from sklearn.preprocessing import StandardScaler
    
    X_features = []
    y_target = []
    for r in results:
        X_features.append([
            r['W_down_norm'],
            r['W_up_norm'],
            r['h_prev_norm'],
            r['layer_frac'],
            r['delta_h_relative'],
        ])
        y_target.append(r['jacobian_norm'])
    
    if len(X_features) >= 5:
        X = np.array(X_features)
        y = np.array(y_target)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        
        ols = LinearRegression()
        ols.fit(X_s, y)
        y_pred = ols.predict(X_s)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        
        print(f"\n  ||J_l|| ~ f(W_down, W_up, h_prev, lfrac, Δh_rel): R2={r2:.3f}")
        feat_names = ['W_down_norm', 'W_up_norm', 'h_prev_norm', 'layer_frac', 'delta_h_relative']
        for fn, c in zip(feat_names, ols.coef_):
            print(f"    {fn}: {c:.4f}")
        
        lasso = Lasso(alpha=0.01)
        lasso.fit(X_s, y)
        y_pred_l = lasso.predict(X_s)
        ss_res_l = np.sum((y - y_pred_l)**2)
        r2_l = 1 - ss_res_l / max(ss_tot, 1e-10)
        print(f"  Lasso: R2={r2_l:.3f}")
        for fn, c in zip(feat_names, lasso.coef_):
            print(f"    {fn}: {c:.4f}")
    
    # 9. 分析残差结构: Jacobian范数偏离1的程度
    j_deviation = [abs(j - 1.0) for j in j_norms]
    r_dev, p_dev = spearmanr(j_deviation, lfracs)
    print(f"\n  |J_l - 1| vs layer_frac: r={r_dev:.3f}, p={p_dev:.4f}")
    print(f"  Jacobian范数统计: mean={np.mean(j_norms):.3f}, "
          f"std={np.std(j_norms):.3f}, "
          f"min={np.min(j_norms):.3f}, max={np.max(j_norms):.3f}")
    
    # 保存结果
    out_dir = os.path.join(project_root, 'tests', 'glm5_temp')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"p500_{model_name}.json")
    
    results_clean = []
    for r in results:
        rc = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
        results_clean.append(rc)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results_clean, f, indent=2, ensure_ascii=False, default=str)
    print(f"  结果已保存: {out_path}")
    
    return results


# ============================================================
# P501: 从Jacobian链推导importance公式
# ============================================================

def run_p501(model_name, device):
    """
    P501: 从Jacobian链推导importance公式
    
    核心目标: 如果gain_l = f(||J_{L:l}||), 则
    importance_l ≈ ||J_{L:l}|| × ||W_down_l|| × ||h_l||
    
    方法:
    1. 对采样层计算: gain_l, ||J_{L:l}||, ||W_down_l||, ||h_l||
    2. 验证公式: importance ≈ ||J_{L:l}|| × ||W_down_l|| × ||h_l||
    3. 如果不行, 尝试其他形式:
       - importance ≈ gain_l × ||W_down_l|| × ||h_l||
       - importance ≈ ||J_{L:l}|| × delta_h_relative
    4. 目标: R2>0.5
    """
    print(f"\n{'='*60}")
    print(f"P501: 从Jacobian链推导importance公式 [{model_name}]")
    print(f"{'='*60}")
    
    model, tokenizer, model_device = load_model(model_name)
    if model_device != device:
        model = model.to(device)
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    mlp_type = info.mlp_type
    
    layers_list = get_layers(model)
    W_U = get_W_U(model)
    
    test_text = "The apple is red and sweet, and it grows on trees in the garden."
    alpha = 0.1
    
    # 多个alpha值(用于验证线性性)
    alphas = [0.05, 0.1, 0.2]
    
    sample_layers = get_sample_layers(n_layers, 10)
    sample_layers = [l for l in sample_layers if l < n_layers - 2]
    print(f"  采样层: {sample_layers}")
    
    results = []
    
    # 先获取基线隐状态(所有层)
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    h_all_baseline = compute_residual_stream(model, input_ids, device)
    
    if h_all_baseline is None:
        print("  基线隐状态获取失败")
        del model
        torch.cuda.empty_cache()
        return results
    
    for l_idx in sample_layers:
        layer_frac = l_idx / max(n_layers - 1, 1)
        print(f"  层L{l_idx}...", end="", flush=True)
        
        # 获取W_down
        lw = get_layer_weights(layers_list[l_idx], d_model, mlp_type)
        W_down = lw.W_down
        W_down_norm = np.linalg.norm(W_down)
        
        # h_l范数
        h_l = h_all_baseline[l_idx]
        h_l_norm = np.linalg.norm(h_l)
        
        for alpha_val in alphas:
            # 1. 计算Jacobian链范数
            jacobian_result = compute_jacobian_chain_norm_numerically(
                model, tokenizer, device, test_text,
                layers_list, l_idx, mlp_type, n_layers, d_model,
                alpha=alpha_val
            )
            
            if jacobian_result is None:
                continue
            
            # 2. 计算importance
            meas = compute_importance_measures(
                model, tokenizer, device, test_text,
                layers_list, l_idx, alpha_val, mlp_type
            )
            
            if meas is None:
                continue
            
            # 3. 计算gain
            gain = meas['delta_h_norm'] / max(alpha_val * W_down_norm, 1e-10)
            
            # 4. 公式预测
            j_chain_norm = jacobian_result['jacobian_chain_norm']
            geo_mean = jacobian_result['geo_mean_ratio']
            
            # 公式1: importance ≈ ||J_{L:l}|| × ||W_down|| × ||h_l||
            pred_formula1 = j_chain_norm * W_down_norm * h_l_norm
            
            # 公式2: importance ≈ gain × ||W_down|| × ||h_l||
            pred_formula2 = gain * W_down_norm * h_l_norm
            
            # 公式3: importance ≈ ||J_{L:l}|| × ||Δh_l|| × alpha
            pred_formula3 = j_chain_norm * jacobian_result['dh_l'] * alpha_val
            
            # 公式4: importance ≈ delta_h_relative × ||W_down|| × ||h_l||
            pred_formula4 = meas['delta_h_relative'] * W_down_norm * h_l_norm
            
            # 公式5: importance ≈ ||J_{L:l}|| × ||W_down|| × ||h_l|| × alpha
            pred_formula5 = j_chain_norm * W_down_norm * h_l_norm * alpha_val
            
            result = {
                'layer': l_idx,
                'layer_frac': layer_frac,
                'alpha': alpha_val,
                'gain': gain,
                'jacobian_chain_norm': j_chain_norm,
                'geo_mean_ratio': geo_mean,
                'W_down_norm': W_down_norm,
                'h_l_norm': h_l_norm,
                'importance_ppl': meas['importance_ppl'],
                'kl_div': meas['kl_div'],
                'delta_h_relative': meas['delta_h_relative'],
                'delta_h_norm': meas['delta_h_norm'],
                'pred_formula1': pred_formula1,
                'pred_formula2': pred_formula2,
                'pred_formula3': pred_formula3,
                'pred_formula4': pred_formula4,
                'pred_formula5': pred_formula5,
            }
            results.append(result)
        
        print(" done")
    
    # 释放模型
    del model
    torch.cuda.empty_cache()
    
    # 统计分析
    if len(results) < 5:
        print("  数据不足, 无法统计分析")
        return results
    
    print(f"\n--- P501 统计分析 [{model_name}] ---")
    
    # 对alpha=0.1的数据进行重点分析
    results_01 = [r for r in results if r['alpha'] == 0.1]
    
    if len(results_01) < 5:
        results_01 = results  # fallback
    
    importances = [r['importance_ppl'] for r in results_01]
    kl_divs = [r['kl_div'] for r in results_01]
    
    # 1. 各公式与importance的相关性
    for formula_name in ['pred_formula1', 'pred_formula2', 'pred_formula3', 'pred_formula4', 'pred_formula5']:
        preds = [r[formula_name] for r in results_01]
        if len(preds) >= 3 and np.std(preds) > 1e-30:
            r_val, p_val = spearmanr(preds, importances)
            print(f"  {formula_name} vs importance_ppl: r={r_val:.3f}, p={p_val:.4f}")
        else:
            print(f"  {formula_name}: 预测值方差过小, 无法计算相关")
    
    # 2. 各公式与kl_div的相关性
    for formula_name in ['pred_formula1', 'pred_formula2', 'pred_formula3', 'pred_formula4', 'pred_formula5']:
        preds = [r[formula_name] for r in results_01]
        if len(preds) >= 3 and np.std(preds) > 1e-30 and np.std(kl_divs) > 1e-30:
            r_val, p_val = spearmanr(preds, kl_divs)
            print(f"  {formula_name} vs kl_div: r={r_val:.3f}, p={p_val:.4f}")
    
    # 3. gain vs jacobian_chain_norm (验证P499的发现)
    gains = [r['gain'] for r in results_01]
    j_chain_norms = [r['jacobian_chain_norm'] for r in results_01]
    if len(gains) >= 3 and np.std(j_chain_norms) > 1e-30:
        r_gj, p_gj = spearmanr(gains, j_chain_norms)
        print(f"\n  gain vs ||J_{{L:l}}||: r={r_gj:.3f}, p={p_gj:.4f}")
    
    # 4. 多元回归: importance ~ f(J_chain, W_down_norm, h_l_norm, layer_frac)
    from sklearn.linear_model import LinearRegression, Lasso
    from sklearn.preprocessing import StandardScaler
    
    X_features = []
    y_target = []
    for r in results_01:
        X_features.append([
            r['jacobian_chain_norm'],
            r['W_down_norm'],
            r['h_l_norm'],
            r['layer_frac'],
            r['gain'],
        ])
        y_target.append(r['importance_ppl'])
    
    if len(X_features) >= 5:
        X = np.array(X_features)
        y = np.array(y_target)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        
        ols = LinearRegression()
        ols.fit(X_s, y)
        y_pred = ols.predict(X_s)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        
        print(f"\n  importance ~ f(J_chain, W_down, h_l, lfrac, gain): R2={r2:.3f}")
        feat_names = ['J_chain_norm', 'W_down_norm', 'h_l_norm', 'layer_frac', 'gain']
        for fn, c in zip(feat_names, ols.coef_):
            print(f"    {fn}: {c:.4f}")
        
        lasso = Lasso(alpha=0.01)
        lasso.fit(X_s, y)
        y_pred_l = lasso.predict(X_s)
        ss_res_l = np.sum((y - y_pred_l)**2)
        r2_l = 1 - ss_res_l / max(ss_tot, 1e-10)
        print(f"  Lasso: R2={r2_l:.3f}")
        for fn, c in zip(feat_names, lasso.coef_):
            print(f"    {fn}: {c:.4f}")
    
    # 5. alpha线性性检验(不同alpha的gain是否一致)
    if len(alphas) > 1:
        print(f"\n  alpha线性性检验:")
        for l_idx in sample_layers[:3]:  # 只检验前3层
            gains_by_alpha = []
            for r in results:
                if r['layer'] == l_idx:
                    gains_by_alpha.append((r['alpha'], r['gain']))
            if len(gains_by_alpha) >= 2:
                gains_by_alpha.sort()
                print(f"    L{l_idx}: " + ", ".join(
                    f"α={a:.2f}→gain={g:.3f}" for a, g in gains_by_alpha
                ))
    
    # 保存结果
    out_dir = os.path.join(project_root, 'tests', 'glm5_temp')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"p501_{model_name}.json")
    
    results_clean = []
    for r in results:
        rc = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
        results_clean.append(rc)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results_clean, f, indent=2, ensure_ascii=False, default=str)
    print(f"  结果已保存: {out_path}")
    
    return results


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase CV: Jacobian链与信号传播增益")
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"],
                       help="模型名称")
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["p499", "p500", "p501", "all"],
                       help="实验编号")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    if args.experiment == "p499":
        run_p499(args.model, device)
    elif args.experiment == "p500":
        run_p500(args.model, device)
    elif args.experiment == "p501":
        run_p501(args.model, device)
    elif args.experiment == "all":
        print("依次运行P499, P500, P501...")
        run_p499(args.model, device)
        run_p500(args.model, device)
        run_p501(args.model, device)
    
    print("\n完成!")


if __name__ == "__main__":
    main()
