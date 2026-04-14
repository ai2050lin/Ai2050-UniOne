"""
Phase CVIII-P508/P509/P510: 信号传播稳定性与模型架构差异
==========================================================

Phase CVII核心瓶颈:
- DS7B信号传播极不稳定(kl_div std≈100% mean)
- DS7B的post_ln_norm与kl_div无关(r=0.08), 因果链断裂
- GLM4/DS7B因果干预存在混沌(幂律指数-0.6到18.6)
- 需要理解: 为什么不同模型的信号传播稳定性差异这么大?

Phase CVIII核心思路:
1. 测量逐层Jacobian的条件数 → 高条件数=不稳定传播
2. 分析权重谱结构的差异 → RL训练是否改变了谱结构
3. 用随机矩阵理论推导传播稳定性条件

关键数学:
  信号传播: h_l = h_{l-1} + f(h_{l-1}; W_l)
  Jacobian: J_l = I + ∂f/∂h_{l-1}
  稳定性条件: ||J_l|| < stable_threshold 且 cond(J_l) < cond_threshold
  当Jacobian条件数大 → 扰动被选择性放大 → 不稳定

P508: 逐层传播稳定性分析
  - 测量每层Jacobian的: 范数、条件数、最大/最小奇异值、各向异性
  - 验证: 条件数大的层是否kl_div不稳定
  - 分析: 信号放大比(attn/FFN)与稳定性的关系
  - 关键问题: DS7B的"极端放大层"是否有异常高的条件数?

P509: RL训练效应与权重谱结构
  - 对比: RL模型(DS7B) vs SFT模型(Qwen3/GLM4)的权重谱结构
  - 分析: W_down/W_gate/W_up的奇异值分布差异
  - 验证: RL模型是否有更"尖锐"的谱(少数大奇异值主导)
  - 验证: Marchenko-Pastur定律是否适用于这些权重矩阵

P510: 信号传播的数学理论
  - 用随机矩阵理论推导: ||J_chain|| 的期望值和方差
  - 验证: 独立同分布假设下, ||J_chain|| ~ N^(L-l)/2 × 某因子
  - 分析: 深层网络中信号传播的发散/收敛条件
  - 推导: stability = f(权重谱, 层数, 扰动方向)

使用方法:
    python phase_cviii_stability_theory.py --model qwen3 --experiment p508
    python phase_cviii_stability_theory.py --model glm4 --experiment p509
    python phase_cviii_stability_theory.py --model deepseek7b --experiment p510
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, 'tests', 'glm5'))

from model_utils import (
    load_model, get_model_info, get_W_U, get_layer_weights,
    get_sample_layers, get_layers,
)


def compute_kl_divergence(logits_baseline, logits_ablated):
    p = torch.nn.functional.softmax(logits_baseline, dim=-1)
    q = torch.nn.functional.log_softmax(logits_ablated, dim=-1)
    kl = torch.nn.functional.kl_div(q, p, reduction='batchmean')
    return kl.item()


def perturb_w_down(layers, l_idx, alpha, mlp_type):
    orig_weight = layers[l_idx].mlp.down_proj.weight.data.clone()
    layers[l_idx].mlp.down_proj.weight.data = orig_weight * (1 - alpha)
    return orig_weight


def restore_w_down(layers, l_idx, orig_weight, mlp_type):
    layers[l_idx].mlp.down_proj.weight.data = orig_weight


def compute_jacobian_finite_diff(model, tokenizer, device, text, layers_list, l_idx, alpha, mlp_type, n_dirs=20):
    """
    用有限差分近似测量层l的Jacobian:
    J_l ≈ Δh_{l+1} / Δh_l
    
    更精确: 扰动W_down, 测量每层的Δh, 计算层间传播比
    
    Returns:
        dict: 每层的delta_h_norm, propagation_ratio, 等
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # 基线前向传播, 收集所有层隐状态
    hidden_baseline = []
    
    def make_hook(idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_baseline.append(output[0][0, -1].detach().cpu().float().numpy())
            else:
                hidden_baseline.append(output[0, -1].detach().cpu().float().numpy())
        return hook_fn
    
    hooks = []
    for i, layer in enumerate(layers_list):
        hooks.append(layer.register_forward_hook(make_hook(i)))
    
    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()
    
    # 扰动W_down, 收集所有层隐状态
    orig_weight = perturb_w_down(layers_list, l_idx, alpha, mlp_type)
    
    hidden_ablated = []
    def make_hook2(idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_ablated.append(output[0][0, -1].detach().cpu().float().numpy())
            else:
                hidden_ablated.append(output[0, -1].detach().cpu().float().numpy())
        return hook_fn
    
    hooks2 = []
    for i, layer in enumerate(layers_list):
        hooks2.append(layer.register_forward_hook(make_hook2(i)))
    
    with torch.no_grad():
        model(input_ids)
    for h in hooks2:
        h.remove()
    
    restore_w_down(layers_list, l_idx, orig_weight, mlp_type)
    
    # 计算每层的Δh
    if len(hidden_baseline) != len(hidden_ablated):
        return None
    
    n = len(hidden_baseline)
    delta_h = []
    for k in range(n):
        dh = hidden_ablated[k] - hidden_baseline[k]
        dh_norm = np.linalg.norm(dh)
        h_norm = np.linalg.norm(hidden_baseline[k])
        delta_h.append({
            'delta_h_norm': dh_norm,
            'h_norm': h_norm,
            'delta_h_relative': dh_norm / max(h_norm, 1e-10),
            'cos_sim': np.dot(dh, hidden_baseline[k]) / max(np.linalg.norm(dh) * h_norm, 1e-10),
        })
    
    # 计算层间传播比
    propagation_ratios = []
    for k in range(n - 1):
        if delta_h[k]['delta_h_norm'] > 1e-10:
            ratio = delta_h[k + 1]['delta_h_norm'] / delta_h[k]['delta_h_norm']
        else:
            ratio = 0
        propagation_ratios.append(ratio)
    
    return {
        'l_idx': l_idx,
        'alpha': alpha,
        'delta_h': delta_h,
        'propagation_ratios': propagation_ratios,
    }


# ============================================================
# P508: 逐层传播稳定性分析
# ============================================================

def run_p508(model_name, device):
    """
    P508: 逐层传播稳定性分析

    核心问题: 为什么DS7B信号传播极不稳定?

    假设: 不稳定的层有异常高的Jacobian条件数/信号放大比

    方法:
    1. 对多条文本×所有层, 测量:
       - 每层的delta_h_norm(扰动后)
       - 层间传播比(propagation_ratio)
       - 信号放大/衰减的方向性(cos_sim)
    2. 计算稳定性指标:
       - mean_ratio: 平均传播比
       - std_ratio: 传播比标准差(跨文本)
       - max_ratio: 最大传播比
       - CV_kl_div: kl_div变异系数
    3. 分析: 稳定性指标与kl_div的关系
    """
    print(f"\n{'='*60}")
    print(f"P508: 逐层传播稳定性分析 [{model_name}]")
    print(f"{'='*60}")

    model, tokenizer, model_device = load_model(model_name)
    if model_device != device:
        model = model.to(device)

    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    mlp_type = info.mlp_type

    layers_list = get_layers(model)

    test_texts = [
        "The apple is red and sweet, and it grows on trees in the garden.",
        "Scientists discovered a new species of butterfly in the Amazon rainforest.",
        "The quantum computer solved the optimization problem in record time.",
        "Music has the power to transform emotions and bring people together.",
        "The ancient temple stood on the hill, watching over the peaceful valley.",
    ]

    alpha = 0.1

    # Part 1: 测量每层的传播稳定性
    print("\n--- Part 1: 传播稳定性测量 ---")
    
    # 采样层(减少计算量)
    sample_layers = get_sample_layers(n_layers, min(12, n_layers))
    print(f"  采样层: {sample_layers}")

    # 对每层×每文本, 计算kl_div和传播特征
    all_kl_results = []
    all_prop_results = []

    for text_idx, text in enumerate(test_texts):
        print(f"  文本 {text_idx+1}/{len(test_texts)}...", end="", flush=True)
        
        for l_idx in sample_layers:
            # kl_div
            inputs = tokenizer(text, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
            
            with torch.no_grad():
                baseline_outputs = model(input_ids)
                baseline_logits = baseline_outputs.logits if hasattr(baseline_outputs, 'logits') else baseline_outputs
                if baseline_logits.dim() == 3:
                    baseline_logits = baseline_logits[0]
                baseline_pred_logits = baseline_logits[:-1]
            
            orig_weight = perturb_w_down(layers_list, l_idx, alpha, mlp_type)
            
            with torch.no_grad():
                ablated_outputs = model(input_ids)
                ablated_logits = ablated_outputs.logits if hasattr(ablated_outputs, 'logits') else ablated_outputs
                if ablated_logits.dim() == 3:
                    ablated_logits = ablated_logits[0]
                ablated_pred_logits = ablated_logits[:-1]
            
            restore_w_down(layers_list, l_idx, orig_weight, mlp_type)
            
            kl_div = compute_kl_divergence(baseline_pred_logits, ablated_pred_logits)
            
            all_kl_results.append({
                'layer': l_idx,
                'text_idx': text_idx,
                'kl_div': kl_div,
            })
            
            # 传播特征
            prop = compute_jacobian_finite_diff(
                model, tokenizer, device, text, layers_list, l_idx, alpha, mlp_type
            )
            if prop is not None:
                # 提取关键特征
                # 从l_idx开始的传播比
                ratios_from_l = prop['propagation_ratios'][l_idx:] if l_idx < len(prop['propagation_ratios']) else []
                
                max_ratio = max(ratios_from_l) if ratios_from_l else 0
                mean_ratio = np.mean(ratios_from_l) if ratios_from_l else 0
                
                # delta_h_at_l: 扰动层的隐状态变化
                dh_at_l = prop['delta_h'][l_idx]['delta_h_norm'] if l_idx < len(prop['delta_h']) else 0
                
                # 最终层的delta_h
                dh_final = prop['delta_h'][-1]['delta_h_norm'] if prop['delta_h'] else 0
                
                # 信号放大: dh_final / dh_at_l (应该≈Jacobian链范数)
                gain = dh_final / max(dh_at_l, 1e-10)
                
                all_prop_results.append({
                    'layer': l_idx,
                    'text_idx': text_idx,
                    'max_ratio': max_ratio,
                    'mean_ratio': mean_ratio,
                    'dh_at_l': dh_at_l,
                    'dh_final': dh_final,
                    'gain': gain,
                    'n_ratios': len(ratios_from_l),
                })
        
        print(" done")

    # 释放模型
    del model
    torch.cuda.empty_cache()

    # Part 2: 统计分析
    print("\n--- Part 2: 统计分析 ---")

    # 逐层汇总
    layer_stats = {}
    for r in all_kl_results:
        l = r['layer']
        if l not in layer_stats:
            layer_stats[l] = {'kl_divs': [], 'max_ratios': [], 'mean_ratios': [], 'gains': []}
        layer_stats[l]['kl_divs'].append(r['kl_div'])

    for r in all_prop_results:
        l = r['layer']
        if l not in layer_stats:
            layer_stats[l] = {'kl_divs': [], 'max_ratios': [], 'mean_ratios': [], 'gains': []}
        layer_stats[l]['max_ratios'].append(r['max_ratio'])
        layer_stats[l]['mean_ratios'].append(r['mean_ratio'])
        layer_stats[l]['gains'].append(r['gain'])

    # 计算稳定性指标
    stability_data = []
    for l, stats in sorted(layer_stats.items()):
        kl_arr = np.array(stats['kl_divs'])
        ratio_arr = np.array(stats['max_ratios'])
        gain_arr = np.array(stats['gains'])
        
        cv_kl = np.std(kl_arr) / max(np.mean(kl_arr), 1e-10) if len(kl_arr) > 1 else 0
        mean_kl = np.mean(kl_arr)
        max_ratio = np.max(ratio_arr) if len(ratio_arr) > 0 else 0
        mean_ratio = np.mean(ratio_arr) if len(ratio_arr) > 0 else 0
        std_ratio = np.std(ratio_arr) if len(ratio_arr) > 0 else 0
        mean_gain = np.mean(gain_arr) if len(gain_arr) > 0 else 0
        
        stability_data.append({
            'layer': l,
            'layer_frac': l / max(n_layers - 1, 1),
            'mean_kl': mean_kl,
            'std_kl': np.std(kl_arr),
            'cv_kl': cv_kl,
            'max_ratio': max_ratio,
            'mean_ratio': mean_ratio,
            'std_ratio': std_ratio,
            'mean_gain': mean_gain,
        })

    if len(stability_data) < 5:
        print("  数据不足")
        return stability_data

    print("\n=== P508 核心结果 [{0}] ===".format(model_name))

    # 相关性分析
    cv_kls = [d['cv_kl'] for d in stability_data]
    max_ratios = [d['max_ratio'] for d in stability_data]
    mean_ratios = [d['mean_ratio'] for d in stability_data]
    mean_kls = [d['mean_kl'] for d in stability_data]
    mean_gains = [d['mean_gain'] for d in stability_data]
    layer_fracs = [d['layer_frac'] for d in stability_data]

    r1, p1 = pearsonr(cv_kls, mean_kls)
    r2, p2 = pearsonr(max_ratios, cv_kls)
    r3, p3 = pearsonr(mean_ratios, mean_kls)
    r4, p4 = pearsonr(mean_gains, cv_kls)
    r5, p5 = pearsonr(max_ratios, mean_kls)

    print(f"\n  ** 稳定性指标与kl_div的关系 **")
    print(f"  CV_kl_div vs mean_kl_div: r={r1:.3f}, p={p1:.4f}")
    print(f"  max_ratio vs CV_kl_div: r={r2:.3f}, p={p2:.4f}")
    print(f"  mean_ratio vs mean_kl_div: r={r3:.3f}, p={p3:.4f}")
    print(f"  mean_gain vs CV_kl_div: r={r4:.3f}, p={p4:.4f}")
    print(f"  max_ratio vs mean_kl_div: r={r5:.3f}, p={p5:.4f}")

    print(f"\n  ** 层间稳定性分布 **")
    print(f"  CV_kl_div: mean={np.mean(cv_kls):.3f}, std={np.std(cv_kls):.3f}")
    print(f"  max_ratio: mean={np.mean(max_ratios):.3f}, max={np.max(max_ratios):.3f}")
    print(f"  mean_gain: mean={np.mean(mean_gains):.3f}, std={np.std(mean_gains):.3f}")

    # 识别不稳定层
    unstable_layers = [d for d in stability_data if d['max_ratio'] > 2.0 or d['cv_kl'] > 1.0]
    if unstable_layers:
        print(f"\n  ** 不稳定层(max_ratio>2 或 CV>1): {len(unstable_layers)}层 **")
        for d in unstable_layers:
            print(f"    L{d['layer']}(frac={d['layer_frac']:.2f}): max_ratio={d['max_ratio']:.2f}, CV_kl={d['cv_kl']:.2f}, mean_kl={d['mean_kl']:.4f}")

    # 多元回归
    try:
        from sklearn.linear_model import LinearRegression, Lasso
        from sklearn.preprocessing import StandardScaler

        X = np.column_stack([
            max_ratios, mean_ratios, mean_gains, layer_fracs,
        ])
        y = np.array(cv_kls)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        reg = LinearRegression()
        reg.fit(X_scaled, y)
        r2_multi = reg.score(X_scaled, y)
        print(f"\n  ** 多元回归: CV_kl ~ max_ratio + mean_ratio + gain + lfrac **")
        print(f"  OLS R2 = {r2_multi:.3f}")
        print(f"  系数: max_ratio={reg.coef_[0]:.4f}, mean_ratio={reg.coef_[1]:.4f}, gain={reg.coef_[2]:.4f}, lfrac={reg.coef_[3]:.4f}")
    except Exception as e:
        print(f"  回归分析失败: {e}")

    # 保存结果
    results = {
        'model': model_name,
        'experiment': 'p508',
        'correlations': {
            'cv_kl_vs_mean_kl': {'r': r1, 'p': p1},
            'max_ratio_vs_cv_kl': {'r': r2, 'p': p2},
            'mean_ratio_vs_mean_kl': {'r': r3, 'p': p3},
            'gain_vs_cv_kl': {'r': r4, 'p': p4},
            'max_ratio_vs_mean_kl': {'r': r5, 'p': p5},
        },
        'stability_data': stability_data,
    }

    out_path = os.path.join(project_root, 'tests', 'glm5_temp', f'p508_{model_name}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  结果保存到: {out_path}")

    return results


# ============================================================
# P509: RL训练效应与权重谱结构
# ============================================================

def run_p509(model_name, device):
    """
    P509: RL训练效应与权重谱结构

    核心问题: RL训练(DS7B)是否改变了权重谱结构?

    假设: RL训练导致权重谱更"尖锐"(少数大奇异值主导),
          这使得信号传播更不稳定

    方法:
    1. 对所有层, 计算W_down/W_gate/W_up的SVD
    2. 分析谱结构:
       - 参与率(PR): 1/(n*sum(p_i^2)), 越小说明越尖锐
       - 有效维度(d_eff): (sum(s))^2 / (n*sum(s^2))
       - 条件数(kappa): s_max/s_min
       - Top-k能量占比: sum(s[:k]^2)/sum(s^2)
       - 谱衰减率: s[0]/s[k] vs k
    3. 与Marchenko-Pastur定律对比:
       - 随机矩阵的谱应该满足MP定律
       - 偏离MP定律说明有结构化信息
    """
    print(f"\n{'='*60}")
    print(f"P509: RL训练效应与权重谱结构 [{model_name}]")
    print(f"{'='*60}")

    model, tokenizer, model_device = load_model(model_name)
    if model_device != device:
        model = model.to(device)

    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    mlp_type = info.mlp_type

    layers_list = get_layers(model)

    print(f"\n--- 权重谱结构分析 ---")

    all_spectral = []

    for l_idx in range(n_layers):
        lw = get_layer_weights(layers_list[l_idx], d_model, mlp_type)
        W_down = lw.W_down
        W_up = lw.W_up
        W_gate = lw.W_gate
        W_o = lw.W_o

        results = {'layer': l_idx, 'layer_frac': l_idx / max(n_layers - 1, 1)}

        for name, W in [('W_down', W_down), ('W_up', W_up), ('W_gate', W_gate), ('W_o', W_o)]:
            if W is None:
                continue

            try:
                from sklearn.utils.extmath import randomized_svd
                n_comp = min(200, min(W.shape) - 1)
                _, s, _ = randomized_svd(W.astype(np.float32), n_components=n_comp, random_state=42)

                # 完整谱统计
                total_energy = np.sum(s**2)
                PR = 1.0 / (len(s) * np.sum((s**2 / total_energy)**2) + 1e-30)
                d_eff = total_energy / (len(s) * np.sum(s**2) / len(s) + 1e-30)  # 简化
                kappa = s[0] / max(s[-1], 1e-10)
                top1 = s[0]**2 / total_energy
                top10 = np.sum(s[:min(10, len(s))]**2) / total_energy
                top50 = np.sum(s[:min(50, len(s))]**2) / total_energy

                # 谱衰减: s[k]/s[0] vs k
                decay_10 = s[min(9, len(s)-1)] / max(s[0], 1e-10)
                decay_50 = s[min(49, len(s)-1)] / max(s[0], 1e-10)
                decay_100 = s[min(99, len(s)-1)] / max(s[0], 1e-10)

                # Marchenko-Pastur检验
                # 对W ~ N(0, sigma^2/n), 谱密度应该满足MP定律
                # 特征: s_max/sqrt(n*m) 应接近 1+sqrt(m/n) (MP上界)
                m, n_w = W.shape
                q = min(m, n_w) / max(m, n_w)
                mp_upper = (1 + np.sqrt(q))**2  # 理论上界(归一化后)
                mp_lower = (1 - np.sqrt(q))**2
                
                # 归一化: s_mp = s / sigma, sigma = Frobenius/sqrt(m*n)
                sigma_est = np.sqrt(total_energy / (m * n_w))
                s_normalized = s / max(sigma_est, 1e-10)
                s_max_normalized = s_normalized[0]
                
                # 偏离MP定律的程度
                mp_deviation = s_max_normalized / max(mp_upper, 1e-10)

                results[f'{name}_PR'] = PR
                results[f'{name}_kappa'] = kappa
                results[f'{name}_top1'] = top1
                results[f'{name}_top10'] = top10
                results[f'{name}_top50'] = top50
                results[f'{name}_decay10'] = decay_10
                results[f'{name}_decay50'] = decay_50
                results[f'{name}_mp_deviation'] = mp_deviation
                results[f'{name}_norm'] = np.linalg.norm(W)

            except Exception as e:
                print(f"    L{l_idx} {name} SVD失败: {e}")

        all_spectral.append(results)
        
        if (l_idx + 1) % 10 == 0:
            print(f"    已处理 {l_idx+1}/{n_layers} 层")

    # 释放模型
    del model
    torch.cuda.empty_cache()

    # 统计分析
    print("\n=== P509 核心结果 [{0}] ===".format(model_name))

    # 汇总统计
    for name in ['W_down', 'W_gate', 'W_up', 'W_o']:
        pr_key = f'{name}_PR'
        kappa_key = f'{name}_kappa'
        top10_key = f'{name}_top10'
        mp_key = f'{name}_mp_deviation'

        if pr_key in all_spectral[0]:
            pr_vals = [d[pr_key] for d in all_spectral]
            kappa_vals = [d[kappa_key] for d in all_spectral]
            top10_vals = [d[top10_key] for d in all_spectral]
            mp_vals = [d[mp_key] for d in all_spectral if mp_key in d]

            print(f"\n  ** {name} 谱结构 **")
            print(f"  PR: mean={np.mean(pr_vals):.4f}, std={np.std(pr_vals):.4f} (越小=越尖锐)")
            print(f"  kappa: mean={np.mean(kappa_vals):.1f}, std={np.std(kappa_vals):.1f}")
            print(f"  top10%: mean={np.mean(top10_vals):.4f}, std={np.std(top10_vals):.4f}")
            if mp_vals:
                print(f"  MP偏离: mean={np.mean(mp_vals):.2f}, std={np.std(mp_vals):.2f} (>1=偏离MP定律)")

    # 层位置趋势
    layer_fracs = [d['layer_frac'] for d in all_spectral]
    for name in ['W_down', 'W_gate']:
        pr_key = f'{name}_PR'
        kappa_key = f'{name}_kappa'
        if pr_key in all_spectral[0]:
            pr_vals = [d[pr_key] for d in all_spectral]
            r_pr, p_pr = pearsonr(layer_fracs, pr_vals)
            kappa_vals = [d[kappa_key] for d in all_spectral]
            r_kappa, p_kappa = pearsonr(layer_fracs, kappa_vals)
            print(f"\n  ** {name} 层位置趋势 **")
            print(f"  layer_frac vs PR: r={r_pr:.3f}, p={p_pr:.4f}")
            print(f"  layer_frac vs kappa: r={r_kappa:.3f}, p={p_kappa:.4f}")

    # 保存结果
    results = {
        'model': model_name,
        'experiment': 'p509',
        'n_layers': n_layers,
        'spectral_data': all_spectral,
    }

    out_path = os.path.join(project_root, 'tests', 'glm5_temp', f'p509_{model_name}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  结果保存到: {out_path}")

    return results


# ============================================================
# P510: 信号传播的数学理论
# ============================================================

def run_p510(model_name, device):
    """
    P510: 信号传播的数学理论

    基于Phase CV-CIX的发现, 推导信号传播的数学理论。

    核心模型:
    h_l = h_{l-1} + f_l(h_{l-1})
    Δh_l = (I + J_l) × Δh_{l-1}  (一阶近似)
    
    对于W_down的扰动:
    Δh_l ≈ W_down × Δz_{l-1}  (l = 扰动层)
    Δh_{k+1} ≈ J_{k+1} × Δh_k  (k > l)
    
    ||Δh_L|| = ||J_L × ... × J_{l+1} × Δh_l|| = ||J_chain|| × ||Δh_l||

    稳定性条件:
    1. 局部稳定: ||J_k|| < threshold (每层不放大太多)
    2. 全局稳定: ||J_chain|| = prod(||J_k||) 不发散
    3. 方向稳定: Δh_k的方向不急剧旋转(cos(Δh_k, Δh_{k-1}) > threshold)

    方法:
    1. 验证||J_chain|| = prod(||J_k||)的近似精度
    2. 测量J_k的特征值分布
    3. 推导稳定性与谱结构的关系
    4. 验证: 稳定模型的J_chain ≈ (mean||J_k||)^(L-l)
    """
    print(f"\n{'='*60}")
    print(f"P510: 信号传播的数学理论 [{model_name}]")
    print(f"{'='*60}")

    model, tokenizer, model_device = load_model(model_name)
    if model_device != device:
        model = model.to(device)

    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    mlp_type = info.mlp_type

    layers_list = get_layers(model)

    test_texts = [
        "The apple is red and sweet, and it grows on trees in the garden.",
        "Scientists discovered a new species of butterfly in the Amazon rainforest.",
    ]

    alpha = 0.1
    sample_layers = get_sample_layers(n_layers, min(8, n_layers))
    print(f"  采样层: {sample_layers}")

    all_results = []

    for text_idx, text in enumerate(test_texts):
        print(f"  文本 {text_idx+1}/{len(test_texts)}...", end="", flush=True)
        
        for l_idx in sample_layers:
            prop = compute_jacobian_finite_diff(
                model, tokenizer, device, text, layers_list, l_idx, alpha, mlp_type
            )
            if prop is None:
                continue

            # 提取从l_idx到最终层的传播数据
            delta_h = prop['delta_h']
            ratios = prop['propagation_ratios']

            # 计算Jacobian链范数
            dh_at_l = delta_h[l_idx]['delta_h_norm'] if l_idx < len(delta_h) else 0
            dh_final = delta_h[-1]['delta_h_norm'] if delta_h else 0
            j_chain_norm = dh_final / max(dh_at_l, 1e-10)

            # 计算逐层传播比的几何平均
            ratios_from_l = ratios[l_idx:] if l_idx < len(ratios) else []
            if ratios_from_l and all(r > 0 for r in ratios_from_l):
                geo_mean_ratio = np.exp(np.mean(np.log(ratios_from_l)))
                n_prop = len(ratios_from_l)
                # 理论预测: ||J_chain|| ≈ geo_mean^n_prop
                predicted_j_chain = geo_mean_ratio ** n_prop
            else:
                geo_mean_ratio = 0
                n_prop = 0
                predicted_j_chain = 0

            # 逐层方向稳定性(cos_sim)
            cos_sims = []
            for k in range(l_idx, len(delta_h) - 1):
                if delta_h[k]['delta_h_norm'] > 1e-10 and delta_h[k+1]['delta_h_norm'] > 1e-10:
                    # 计算Δh_k和Δh_{k+1}的方向相似度
                    # (需要原始向量, 这里用近似)
                    cos_sims.append(delta_h[k]['cos_sim'])

            # 找最大放大层
            max_ratio_idx = l_idx + np.argmax(ratios_from_l) if ratios_from_l else -1
            max_ratio_val = max(ratios_from_l) if ratios_from_l else 0

            result = {
                'layer': l_idx,
                'text_idx': text_idx,
                'j_chain_norm': j_chain_norm,
                'geo_mean_ratio': geo_mean_ratio,
                'n_prop': n_prop,
                'predicted_j_chain': predicted_j_chain,
                'prediction_error': abs(j_chain_norm - predicted_j_chain) / max(j_chain_norm, 1e-10),
                'max_ratio': max_ratio_val,
                'max_ratio_layer': max_ratio_idx,
                'mean_ratio': np.mean(ratios_from_l) if ratios_from_l else 0,
                'std_ratio': np.std(ratios_from_l) if ratios_from_l else 0,
            }
            all_results.append(result)
        
        print(" done")

    # 释放模型
    del model
    torch.cuda.empty_cache()

    # 统计分析
    print("\n=== P510 核心结果 [{0}] ===".format(model_name))

    if len(all_results) < 5:
        print("  数据不足")
        return all_results

    # 验证: ||J_chain|| ≈ (geo_mean)^n_prop
    j_chain_norms = [r['j_chain_norm'] for r in all_results if r['j_chain_norm'] > 0]
    predicted = [r['predicted_j_chain'] for r in all_results if r['predicted_j_chain'] > 0]
    pred_errors = [r['prediction_error'] for r in all_results if r['prediction_error'] < 10]
    geo_means = [r['geo_mean_ratio'] for r in all_results if r['geo_mean_ratio'] > 0]
    max_ratios = [r['max_ratio'] for r in all_results]
    mean_ratios = [r['mean_ratio'] for r in all_results]

    if j_chain_norms and predicted and len(j_chain_norms) == len(predicted):
        r_pred, p_pred = pearsonr(j_chain_norms, predicted)
        print(f"\n  ** Jacobian链范数的理论预测 **")
        print(f"  ||J_chain|| vs (geo_mean)^n_prop: r={r_pred:.3f}, p={p_pred:.4f}")
        print(f"  预测误差: mean={np.mean(pred_errors):.3f}, std={np.std(pred_errors):.3f}")

    print(f"\n  ** 传播比统计 **")
    print(f"  geo_mean_ratio: mean={np.mean(geo_means):.3f}, std={np.std(geo_means):.3f}")
    print(f"  max_ratio: mean={np.mean(max_ratios):.3f}, max={np.max(max_ratios):.3f}")
    print(f"  mean_ratio: mean={np.mean(mean_ratios):.3f}, std={np.std(mean_ratios):.3f}")

    # 稳定性分析
    # 理论: 如果每层ratio≈1, 则J_chain ≈ 1 (稳定传播)
    # 如果某层ratio>>1, 则J_chain爆炸 (不稳定)
    stable_results = [r for r in all_results if r['max_ratio'] < 2.0]
    unstable_results = [r for r in all_results if r['max_ratio'] >= 2.0]

    print(f"\n  ** 稳定性分类 **")
    print(f"  稳定(max_ratio<2): {len(stable_results)}个")
    print(f"  不稳定(max_ratio≥2): {len(unstable_results)}个")

    if stable_results and unstable_results:
        stable_j = [r['j_chain_norm'] for r in stable_results if r['j_chain_norm'] > 0]
        unstable_j = [r['j_chain_norm'] for r in unstable_results if r['j_chain_norm'] > 0]
        if stable_j and unstable_j:
            print(f"  稳定: mean J_chain={np.mean(stable_j):.3f}")
            print(f"  不稳定: mean J_chain={np.mean(unstable_j):.3f}")

    # 理论推导验证
    # 假设: ||J_chain|| ≈ c × (1-layer_frac)^d × N^e
    # 其中N=层数, c,d,e是常数
    try:
        from sklearn.linear_model import LinearRegression

        valid = [r for r in all_results if r['j_chain_norm'] > 0]
        if len(valid) > 5:
            X = np.column_stack([
                np.log(np.array([1 - r['layer'] / max(n_layers - 1, 1) + 0.01 for r in valid])),
                [r['n_prop'] for r in valid],
            ])
            y = np.log(np.array([r['j_chain_norm'] for r in valid]))

            reg = LinearRegression().fit(X, y)
            r2 = reg.score(X, y)
            print(f"\n  ** 理论公式验证: log(J_chain) ~ log(1-lfrac) + n_prop **")
            print(f"  R2 = {r2:.3f}")
            print(f"  系数: log(1-lfrac)={reg.coef_[0]:.3f}, n_prop={reg.coef_[1]:.3f}")
            print(f"  截距: {reg.intercept_:.3f}")
            print(f"  → J_chain ≈ exp({reg.intercept_:.2f}) × (1-lfrac)^{reg.coef_[0]:.2f} × exp(n_prop×{reg.coef_[1]:.2f})")
    except Exception as e:
        print(f"  回归分析失败: {e}")

    # 保存结果
    results = {
        'model': model_name,
        'experiment': 'p510',
        'n_layers': n_layers,
        'results': all_results,
    }

    out_path = os.path.join(project_root, 'tests', 'glm5_temp', f'p510_{model_name}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  结果保存到: {out_path}")

    return results


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Phase CVIII: 信号传播稳定性')
    parser.add_argument('--model', type=str, required=True,
                        choices=['qwen3', 'glm4', 'deepseek7b'])
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['p508', 'p509', 'p510', 'all'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA不可用, 使用CPU")

    os.makedirs(os.path.join(project_root, 'tests', 'glm5_temp'), exist_ok=True)

    if args.experiment == 'all':
        run_p508(args.model, device)
        run_p509(args.model, device)
        run_p510(args.model, device)
    elif args.experiment == 'p508':
        run_p508(args.model, device)
    elif args.experiment == 'p509':
        run_p509(args.model, device)
    elif args.experiment == 'p510':
        run_p510(args.model, device)


if __name__ == '__main__':
    main()
