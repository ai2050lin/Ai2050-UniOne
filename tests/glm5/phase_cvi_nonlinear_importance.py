"""
Phase CVI-P502/P503/P504: importance的非线性解析
==================================================

Phase CV核心瓶颈:
- gain = f(||J_{L:l}||)已验证(r=0.72-0.86)
- 但简单乘积公式(importance = J × W × h)失败(r<0.2)
- 原因: (1)importance_ppl太小且噪声大 (2)h_l_norm跨层变化64倍(6.8→438)
  (3)需要正确的归一化/缩放/非线性形式

Phase CVI核心思路:
1. 用kl_div替代importance_ppl(更稳定, 动态范围更大)
2. 用幂律拟合: log(kl_div) ≈ a*log(J_chain) + b*log(W_down) + c*log(h_l) + ...
3. 用多条文本增加统计量
4. 分析为什么简单乘积公式失败——是否是LayerNorm的归一化效应?

P502: 幂律拟合与非线性分析
  - 用log-log回归拟合幂律: kl_div ≈ J_chain^a × W_down^b × h_l^c × ...
  - 分析各幂律指数的物理含义
  - 对比线性vs幂律vs对数的拟合效果
  - 关键问题: h_l_norm跨层64倍增长是否导致importance的层间差异?

P503: LayerNorm归一化效应与importance的真正决定因素
  - 假设: LayerNorm消除了h_l_norm的增长效应, 使得||Δh_final||不与||h_l||成正比
  - 验证: delta_h_final vs delta_h_before_LN vs h_l_norm的关系
  - 如果LN效应主导, 则importance ≈ f(J_chain, W_down_norm, delta_h_relative)
  - 不再需要h_l_norm!

P504: 从纯权重特征预测importance (代理模型)
  - 训练小MLP/决策树从纯权重特征预测kl_div
  - 权重特征: W_down_norm, W_up_norm, W_gate_norm, PR, kappa, top10, layer_frac
  - 目标: 交叉验证R2>0.5

使用方法:
    python phase_cvi_nonlinear_importance.py --model qwen3 --experiment p502
    python phase_cvi_nonlinear_importance.py --model glm4 --experiment p503
    python phase_cvi_nonlinear_importance.py --model deepseek7b --experiment p504
    python phase_cvi_nonlinear_importance.py --model qwen3 --experiment all
"""

import sys
import os
import argparse
import numpy as np
import torch
import json
import time
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import curve_fit
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
    s_sq = s**2
    s_sq_norm = s_sq / (np.sum(s_sq) + 1e-30)
    return 1.0 / (len(s) * np.sum(s_sq_norm**2) + 1e-30)


def compute_effective_dimension(s):
    return (np.sum(s)**2) / (len(s) * np.sum(s**2) + 1e-30)


def compute_kl_divergence(logits_baseline, logits_ablated):
    p = torch.nn.functional.softmax(logits_baseline, dim=-1)
    q = torch.nn.functional.log_softmax(logits_ablated, dim=-1)
    kl = torch.nn.functional.kl_div(q, p, reduction='batchmean')
    return kl.item()


def compute_residual_stream(model, input_ids, device):
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
    orig_weight = layers[l_idx].mlp.down_proj.weight.data.clone()
    layers[l_idx].mlp.down_proj.weight.data = orig_weight * (1 - alpha)
    return orig_weight


def restore_w_down(layers, l_idx, orig_weight, mlp_type):
    layers[l_idx].mlp.down_proj.weight.data = orig_weight


def compute_importance_measures(model, tokenizer, device, text, layers, l_idx, alpha, mlp_type):
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
    
    # 收集每层的变化(用于Jacobian链分析)
    delta_h_per_layer = None
    if h_all_ablated is not None:
        delta_h_per_layer = []
        for k in range(len(h_all_baseline)):
            dh = np.linalg.norm(h_all_ablated[k] - h_all_baseline[k])
            h_norm = max(np.linalg.norm(h_all_baseline[k]), 1e-10)
            delta_h_per_layer.append({
                'delta_h_norm': dh,
                'delta_h_relative': dh / h_norm,
                'h_norm': h_norm,
            })
    
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
    
    return {
        'delta_h_final': delta_h_final,
        'delta_h_norm': delta_h_norm,
        'h_norm': h_norm,
        'delta_h_relative': delta_h_norm / max(h_norm, 1e-10),
        'baseline_ppl': baseline_ppl,
        'ablated_ppl': ablated_ppl,
        'delta_ppl': delta_ppl,
        'importance_ppl': importance_ppl,
        'kl_div': kl_div,
        'delta_h_per_layer': delta_h_per_layer,
        'alpha': alpha,
        'l_idx': l_idx,
    }


# ============================================================
# P502: 幂律拟合与非线性分析
# ============================================================

def run_p502(model_name, device):
    """
    P502: 幂律拟合与非线性分析
    
    核心问题: 简单乘积公式失败, importance不是线性组合
    假设: importance是非线性函数, 可能是幂律形式
    
    方法:
    1. 用kl_div作为目标变量(更稳定, 动态范围更大)
    2. 幂律拟合: log(kl_div) = a*log(J_chain) + b*log(W_down) + c*log(h_l) + ...
    3. 对比不同函数形式的拟合效果
    4. 多条文本平均, 减少噪声
    """
    print(f"\n{'='*60}")
    print(f"P502: 幂律拟合与非线性分析 [{model_name}]")
    print(f"{'='*60}")
    
    model, tokenizer, model_device = load_model(model_name)
    if model_device != device:
        model = model.to(device)
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    mlp_type = info.mlp_type
    
    layers_list = get_layers(model)
    
    # 多条测试文本(减少噪声)
    test_texts = [
        "The apple is red and sweet, and it grows on trees in the garden.",
        "Scientists discovered a new species of butterfly in the Amazon rainforest.",
        "The quantum computer solved the optimization problem in record time.",
        "Music has the power to transform emotions and bring people together.",
        "The ancient temple stood on the hill, watching over the peaceful valley.",
    ]
    
    alpha = 0.1
    
    # 密集采样
    sample_layers = get_sample_layers(n_layers, 15)
    sample_layers = [l for l in sample_layers if l < n_layers - 2]
    print(f"  采样层: {sample_layers}")
    
    all_results = []
    
    for text_idx, text in enumerate(test_texts):
        print(f"\n  文本 {text_idx+1}/{len(test_texts)}: {text[:50]}...")
        
        # 获取基线隐状态
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        h_all_baseline = compute_residual_stream(model, input_ids, device)
        
        if h_all_baseline is None:
            continue
        
        for l_idx in sample_layers:
            layer_frac = l_idx / max(n_layers - 1, 1)
            
            # 获取W_down和谱特征
            lw = get_layer_weights(layers_list[l_idx], d_model, mlp_type)
            W_down = lw.W_down
            W_down_norm = np.linalg.norm(W_down)
            W_up = lw.W_up
            W_up_norm = np.linalg.norm(W_up) if W_up is not None else 0
            W_gate = lw.W_gate
            W_gate_norm = np.linalg.norm(W_gate) if W_gate is not None else 0
            
            # SVD谱特征
            try:
                from sklearn.utils.extmath import randomized_svd
                _, s_wd, _ = randomized_svd(W_down.astype(np.float32), n_components=min(100, min(W_down.shape)-1), random_state=42)
                PR_wd = compute_participation_ratio(s_wd)
                kappa_wd = s_wd[0] / max(s_wd[-1], 1e-10)
                top10_wd = np.sum(s_wd[:min(10, len(s_wd))]**2) / max(np.sum(s_wd**2), 1e-10)
            except:
                PR_wd = kappa_wd = top10_wd = 0
            
            # h_l范数
            h_l_norm = np.linalg.norm(h_all_baseline[l_idx])
            
            # 计算importance
            meas = compute_importance_measures(
                model, tokenizer, device, text,
                layers_list, l_idx, alpha, mlp_type
            )
            
            if meas is None:
                continue
            
            # 计算Jacobian链范数
            dh_l = meas['delta_h_per_layer'][l_idx]['delta_h_norm'] if meas['delta_h_per_layer'] else 0
            dh_L = meas['delta_h_norm']
            jacobian_chain_norm = dh_L / max(dh_l, 1e-30)
            
            gain = meas['delta_h_norm'] / max(alpha * W_down_norm, 1e-10)
            
            # delta_h_relative at layer l
            delta_h_rel_l = meas['delta_h_per_layer'][l_idx]['delta_h_relative'] if meas['delta_h_per_layer'] else 0
            
            result = {
                'layer': l_idx,
                'layer_frac': layer_frac,
                'text_idx': text_idx,
                'alpha': alpha,
                'kl_div': meas['kl_div'],
                'importance_ppl': meas['importance_ppl'],
                'delta_ppl': meas['delta_ppl'],
                'delta_h_relative': meas['delta_h_relative'],
                'delta_h_norm': meas['delta_h_norm'],
                'jacobian_chain_norm': jacobian_chain_norm,
                'gain': gain,
                'W_down_norm': W_down_norm,
                'W_up_norm': W_up_norm,
                'W_gate_norm': W_gate_norm,
                'h_l_norm': h_l_norm,
                'PR_wd': PR_wd,
                'kappa_wd': kappa_wd,
                'top10_wd': top10_wd,
                'delta_h_rel_l': delta_h_rel_l,
            }
            all_results.append(result)
    
    # 释放模型
    del model
    torch.cuda.empty_cache()
    
    # 统计分析
    if len(all_results) < 10:
        print("  数据不足, 无法统计分析")
        return all_results
    
    print(f"\n--- P502 统计分析 [{model_name}] ---")
    print(f"  总数据点: {len(all_results)}")
    
    # 对每条文本的平均
    kl_divs = [r['kl_div'] for r in all_results]
    j_chains = [r['jacobian_chain_norm'] for r in all_results]
    W_down_norms = [r['W_down_norm'] for r in all_results]
    h_l_norms = [r['h_l_norm'] for r in all_results]
    lfracs = [r['layer_frac'] for r in all_results]
    gains = [r['gain'] for r in all_results]
    dh_rels = [r['delta_h_relative'] for r in all_results]
    PR_wds = [r['PR_wd'] for r in all_results]
    dh_rel_ls = [r['delta_h_rel_l'] for r in all_results]
    
    # 1. 基础相关性(用kl_div作为目标)
    print(f"\n  === 基础相关性 (vs kl_div) ===")
    for name, vals in [
        ('J_chain', j_chains),
        ('W_down_norm', W_down_norms),
        ('h_l_norm', h_l_norms),
        ('layer_frac', lfracs),
        ('gain', gains),
        ('delta_h_relative', dh_rels),
        ('PR_wd', PR_wds),
        ('delta_h_rel_l', dh_rel_ls),
    ]:
        if np.std(vals) > 1e-30 and np.std(kl_divs) > 1e-30:
            r_val, p_val = spearmanr(vals, kl_divs)
            print(f"  {name} vs kl_div: r={r_val:.3f}, p={p_val:.4f}")
    
    # 2. 对数空间相关性
    print(f"\n  === 对数空间相关性 (log-log) ===")
    log_kl = np.log(np.array(kl_divs) + 1e-30)
    for name, vals in [
        ('log(J_chain)', np.log(np.array(j_chains) + 1e-30)),
        ('log(W_down)', np.log(np.array(W_down_norms) + 1e-30)),
        ('log(h_l)', np.log(np.array(h_l_norms) + 1e-30)),
        ('log(gain)', np.log(np.array(gains) + 1e-30)),
        ('log(delta_h_rel)', np.log(np.array(dh_rels) + 1e-30)),
    ]:
        if np.std(vals) > 1e-30:
            r_val, p_val = spearmanr(vals, log_kl)
            print(f"  {name} vs log(kl_div): r={r_val:.3f}, p={p_val:.4f}")
    
    # 3. 幂律拟合: log(kl_div) = a*log(J) + b*log(W) + c*log(h) + d
    from sklearn.linear_model import LinearRegression, Lasso, Ridge
    from sklearn.preprocessing import StandardScaler
    
    # 准备特征矩阵(对数空间)
    log_features = []
    y_log_kl = []
    for r in all_results:
        if r['kl_div'] > 1e-30 and r['jacobian_chain_norm'] > 1e-30:
            log_features.append([
                np.log(r['jacobian_chain_norm']),
                np.log(r['W_down_norm']),
                np.log(r['h_l_norm']),
                r['layer_frac'],  # layer_frac在[0,1], 不取log
                np.log(r['gain'] + 1e-30),
                np.log(r['delta_h_relative'] + 1e-30),
            ])
            y_log_kl.append(np.log(r['kl_div']))
    
    if len(log_features) >= 10:
        X_log = np.array(log_features)
        y_log = np.array(y_log_kl)
        scaler = StandardScaler()
        X_log_s = scaler.fit_transform(X_log)
        
        # OLS in log space
        ols = LinearRegression()
        ols.fit(X_log_s, y_log)
        y_pred = ols.predict(X_log_s)
        ss_res = np.sum((y_log - y_pred)**2)
        ss_tot = np.sum((y_log - np.mean(y_log))**2)
        r2_log = 1 - ss_res / max(ss_tot, 1e-10)
        
        print(f"\n  === 幂律拟合: log(kl_div) = f(log(J), log(W), log(h), lfrac, log(gain), log(dh_rel)) ===")
        print(f"  OLS R2={r2_log:.3f}")
        feat_names = ['log(J_chain)', 'log(W_down)', 'log(h_l)', 'layer_frac', 'log(gain)', 'log(dh_rel)']
        for fn, c in zip(feat_names, ols.coef_):
            print(f"    {fn}: {c:.4f}")
        print(f"    intercept: {ols.intercept_:.4f}")
        
        # Lasso in log space
        lasso = Lasso(alpha=0.01)
        lasso.fit(X_log_s, y_log)
        y_pred_l = lasso.predict(X_log_s)
        ss_res_l = np.sum((y_log - y_pred_l)**2)
        r2_l = 1 - ss_res_l / max(ss_tot, 1e-10)
        print(f"  Lasso R2={r2_l:.3f}")
        for fn, c in zip(feat_names, lasso.coef_):
            print(f"    {fn}: {c:.4f}")
        
        # Ridge in log space
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_log_s, y_log)
        y_pred_r = ridge.predict(X_log_s)
        ss_res_r = np.sum((y_log - y_pred_r)**2)
        r2_r = 1 - ss_res_r / max(ss_tot, 1e-10)
        print(f"  Ridge R2={r2_r:.3f}")
        for fn, c in zip(feat_names, ridge.coef_):
            print(f"    {fn}: {c:.4f}")
    
    # 4. 对比线性空间vs对数空间
    # 线性空间回归
    lin_features = []
    y_lin = []
    for r in all_results:
        lin_features.append([
            r['jacobian_chain_norm'],
            r['W_down_norm'],
            r['h_l_norm'],
            r['layer_frac'],
            r['gain'],
            r['delta_h_relative'],
        ])
        y_lin.append(r['kl_div'])
    
    if len(lin_features) >= 10:
        X_lin = np.array(lin_features)
        y_lin = np.array(y_lin)
        scaler_lin = StandardScaler()
        X_lin_s = scaler_lin.fit_transform(X_lin)
        
        ols_lin = LinearRegression()
        ols_lin.fit(X_lin_s, y_lin)
        y_pred_lin = ols_lin.predict(X_lin_s)
        ss_res_lin = np.sum((y_lin - y_pred_lin)**2)
        ss_tot_lin = np.sum((y_lin - np.mean(y_lin))**2)
        r2_lin = 1 - ss_res_lin / max(ss_tot_lin, 1e-10)
        
        print(f"\n  === 对比: 线性空间 vs 对数空间 ===")
        print(f"  线性空间 R2: {r2_lin:.3f}")
        print(f"  对数空间 R2: {r2_log:.3f}")
        print(f"  提升: {(r2_log - r2_lin):.3f}")
    
    # 5. 简化幂律公式(只用最重要的2-3个特征)
    print(f"\n  === 简化幂律公式 ===")
    for subset_name, subset_idx in [
        ('log(J) + log(dh_rel)', [0, 5]),
        ('log(J) + lfrac', [0, 3]),
        ('log(dh_rel) + lfrac', [5, 3]),
        ('log(J) + log(dh_rel) + lfrac', [0, 5, 3]),
        ('log(gain) + log(dh_rel)', [4, 5]),
        ('log(gain) + lfrac', [4, 3]),
    ]:
        X_sub = X_log_s[:, subset_idx]
        ols_sub = LinearRegression()
        ols_sub.fit(X_sub, y_log)
        y_pred_sub = ols_sub.predict(X_sub)
        ss_res_sub = np.sum((y_log - y_pred_sub)**2)
        r2_sub = 1 - ss_res_sub / max(ss_tot, 1e-10)
        print(f"  {subset_name}: R2={r2_sub:.3f}")
    
    # 6. 每文本的层平均分析
    print(f"\n  === 每层平均kl_div (跨文本) ===")
    from collections import defaultdict
    layer_kl = defaultdict(list)
    for r in all_results:
        layer_kl[r['layer']].append(r['kl_div'])
    
    layer_avg = {}
    for l in sorted(layer_kl.keys()):
        avg_kl = np.mean(layer_kl[l])
        std_kl = np.std(layer_kl[l])
        layer_avg[l] = avg_kl
        print(f"    L{l}: avg_kl={avg_kl:.6f} ± {std_kl:.6f} (n={len(layer_kl[l])})")
    
    # 7. 逐层平均后的相关性
    if len(layer_avg) >= 5:
        layers_sorted = sorted(layer_avg.keys())
        avg_kls = [layer_avg[l] for l in layers_sorted]
        avg_lfracs = [l / max(n_layers - 1, 1) for l in layers_sorted]
        
        # 逐层平均的J_chain
        layer_jchain = defaultdict(list)
        for r in all_results:
            layer_jchain[r['layer']].append(r['jacobian_chain_norm'])
        avg_jchains = [np.mean(layer_jchain[l]) for l in layers_sorted]
        
        r_kl_lfrac, _ = spearmanr(avg_kls, avg_lfracs)
        r_kl_jchain, _ = spearmanr(avg_kls, avg_jchains)
        print(f"\n  逐层平均: kl_div vs layer_frac: r={r_kl_lfrac:.3f}")
        print(f"  逐层平均: kl_div vs J_chain: r={r_kl_jchain:.3f}")
    
    # 保存结果
    out_dir = os.path.join(project_root, 'tests', 'glm5_temp')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"p502_{model_name}.json")
    
    results_clean = []
    for r in all_results:
        rc = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
        results_clean.append(rc)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results_clean, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果已保存: {out_path}")
    
    return all_results


# ============================================================
# P503: LayerNorm归一化效应
# ============================================================

def run_p503(model_name, device):
    """
    P503: LayerNorm归一化效应与importance的真正决定因素
    
    核心假设: LayerNorm消除了h_l_norm的增长效应
    如果h_l经过LayerNorm后范数恒定, 则||Δh_final||不与||h_l||成正比
    
    关键问题:
    1. delta_h_relative vs h_l_norm: 如果LN有效, 两者应负相关(大h→小相对变化)
    2. delta_h_norm vs h_l_norm: 如果LN无效, 两者应正相关(大h→大绝对变化)
    3. 实际的delta_h_norm跨层变化如何?
    
    方法:
    1. 对所有层计算: h_l_norm, delta_h_norm(扰动后), delta_h_relative
    2. 分析delta_h_norm和delta_h_relative与h_l_norm的关系
    3. 如果delta_h_norm ≈ const(跨层), 说明LN完全归一化
    4. 如果delta_h_relative ∝ 1/h_l_norm, 说明LN部分归一化
    """
    print(f"\n{'='*60}")
    print(f"P503: LayerNorm归一化效应 [{model_name}]")
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
    
    # 所有层
    sample_layers = list(range(n_layers))
    print(f"  所有层: {n_layers}层")
    
    results = []
    
    # 基线
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    h_all_baseline = compute_residual_stream(model, input_ids, device)
    
    if h_all_baseline is None:
        print("  基线获取失败")
        del model
        torch.cuda.empty_cache()
        return results
    
    # 收集每层的h_l_norm
    h_norms = [np.linalg.norm(h_all_baseline[l]) for l in range(n_layers)]
    print(f"  h_l_norm范围: {min(h_norms):.1f} ~ {max(h_norms):.1f} (倍数={max(h_norms)/max(min(h_norms),1e-10):.1f})")
    
    for l_idx in sample_layers:
        layer_frac = l_idx / max(n_layers - 1, 1)
        
        # 获取W_down
        lw = get_layer_weights(layers_list[l_idx], d_model, mlp_type)
        W_down = lw.W_down
        W_down_norm = np.linalg.norm(W_down)
        
        # 计算importance
        meas = compute_importance_measures(
            model, tokenizer, device, test_text,
            layers_list, l_idx, alpha, mlp_type
        )
        
        if meas is None:
            continue
        
        # 每层的delta_h
        if meas['delta_h_per_layer'] is not None:
            dh_l = meas['delta_h_per_layer'][l_idx]
            dh_final = meas['delta_h_per_layer'][-1]
        else:
            continue
        
        result = {
            'layer': l_idx,
            'layer_frac': layer_frac,
            'h_l_norm': h_norms[l_idx],
            'delta_h_norm_at_l': dh_l['delta_h_norm'],
            'delta_h_rel_at_l': dh_l['delta_h_relative'],
            'delta_h_norm_final': dh_final['delta_h_norm'],
            'delta_h_rel_final': dh_final['delta_h_relative'],
            'delta_h_norm': meas['delta_h_norm'],
            'delta_h_relative': meas['delta_h_relative'],
            'W_down_norm': W_down_norm,
            'kl_div': meas['kl_div'],
            'importance_ppl': meas['importance_ppl'],
            'gain': meas['delta_h_norm'] / max(alpha * W_down_norm, 1e-10),
        }
        results.append(result)
    
    # 释放模型
    del model
    torch.cuda.empty_cache()
    
    # 统计分析
    if len(results) < 10:
        print("  数据不足, 无法统计分析")
        return results
    
    print(f"\n--- P503 统计分析 [{model_name}] ---")
    
    h_norms_arr = np.array([r['h_l_norm'] for r in results])
    dh_norms_at_l = np.array([r['delta_h_norm_at_l'] for r in results])
    dh_rels_at_l = np.array([r['delta_h_rel_at_l'] for r in results])
    dh_norms_final = np.array([r['delta_h_norm_final'] for r in results])
    dh_rels_final = np.array([r['delta_h_rel_final'] for r in results])
    kl_divs = np.array([r['kl_div'] for r in results])
    lfracs = np.array([r['layer_frac'] for r in results])
    W_down_norms = np.array([r['W_down_norm'] for r in results])
    gains = np.array([r['gain'] for r in results])
    
    # 1. h_l_norm vs delta_h_norm_at_l (LN归一化效应)
    r1, p1 = spearmanr(h_norms_arr, dh_norms_at_l)
    print(f"  h_l_norm vs delta_h_norm_at_l: r={r1:.3f}, p={p1:.4f}")
    
    # 2. h_l_norm vs delta_h_rel_at_l
    r2, p2 = spearmanr(h_norms_arr, dh_rels_at_l)
    print(f"  h_l_norm vs delta_h_rel_at_l: r={r2:.3f}, p={p2:.4f}")
    
    # 3. h_l_norm vs delta_h_norm_final
    r3, p3 = spearmanr(h_norms_arr, dh_norms_final)
    print(f"  h_l_norm vs delta_h_norm_final: r={r3:.3f}, p={p3:.4f}")
    
    # 4. delta_h_norm_at_l vs kl_div
    r4, p4 = spearmanr(dh_norms_at_l, kl_divs)
    print(f"  delta_h_norm_at_l vs kl_div: r={r4:.3f}, p={p4:.4f}")
    
    # 5. delta_h_rel_at_l vs kl_div
    r5, p5 = spearmanr(dh_rels_at_l, kl_divs)
    print(f"  delta_h_rel_at_l vs kl_div: r={r5:.3f}, p={p5:.4f}")
    
    # 6. h_l_norm vs kl_div
    r6, p6 = spearmanr(h_norms_arr, kl_divs)
    print(f"  h_l_norm vs kl_div: r={r6:.3f}, p={p6:.4f}")
    
    # 7. delta_h_norm_at_l统计
    print(f"\n  delta_h_norm_at_l: mean={np.mean(dh_norms_at_l):.3f}, std={np.std(dh_norms_at_l):.3f}")
    print(f"  delta_h_rel_at_l: mean={np.mean(dh_rels_at_l):.4f}, std={np.std(dh_rels_at_l):.4f}")
    print(f"  delta_h_norm_final: mean={np.mean(dh_norms_final):.3f}, std={np.std(dh_norms_final):.3f}")
    
    # 8. 核心判断: delta_h_norm_at_l是否跨层恒定?
    cv_at_l = np.std(dh_norms_at_l) / max(np.mean(dh_norms_at_l), 1e-10)
    cv_final = np.std(dh_norms_final) / max(np.mean(dh_norms_final), 1e-10)
    print(f"\n  变异系数(CV, Coefficient of Variation):")
    print(f"    delta_h_norm_at_l: CV={cv_at_l:.3f}")
    print(f"    delta_h_norm_final: CV={cv_final:.3f}")
    if cv_at_l < 0.3:
        print(f"    → delta_h_norm_at_l跨层基本恒定! LayerNorm归一化效应强")
    else:
        print(f"    → delta_h_norm_at_l跨层变化大, LayerNorm归一化效应弱")
    
    # 9. 关键回归: kl_div ~ f(delta_h_norm, delta_h_rel, h_l_norm, gain)
    from sklearn.linear_model import LinearRegression, Lasso
    from sklearn.preprocessing import StandardScaler
    
    X_features = []
    y_target = []
    for r in results:
        X_features.append([
            r['delta_h_norm_at_l'],
            r['delta_h_rel_at_l'],
            r['h_l_norm'],
            r['layer_frac'],
            r['gain'],
            r['W_down_norm'],
        ])
        y_target.append(r['kl_div'])
    
    if len(X_features) >= 10:
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
        
        print(f"\n  kl_div ~ f(dh_norm_l, dh_rel_l, h_l, lfrac, gain, W_down): R2={r2:.3f}")
        feat_names = ['dh_norm_l', 'dh_rel_l', 'h_l_norm', 'layer_frac', 'gain', 'W_down_norm']
        for fn, c in zip(feat_names, ols.coef_):
            print(f"    {fn}: {c:.4f}")
        
        lasso = Lasso(alpha=0.001)
        lasso.fit(X_s, y)
        y_pred_l = lasso.predict(X_s)
        ss_res_l = np.sum((y - y_pred_l)**2)
        r2_l = 1 - ss_res_l / max(ss_tot, 1e-10)
        print(f"  Lasso: R2={r2_l:.3f}")
        for fn, c in zip(feat_names, lasso.coef_):
            print(f"    {fn}: {c:.4f}")
    
    # 保存
    out_dir = os.path.join(project_root, 'tests', 'glm5_temp')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"p503_{model_name}.json")
    
    results_clean = []
    for r in results:
        rc = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
        results_clean.append(rc)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results_clean, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果已保存: {out_path}")
    
    return results


# ============================================================
# P504: 从纯权重特征预测importance (代理模型)
# ============================================================

def run_p504(model_name, device):
    """
    P504: 从纯权重特征预测importance
    
    目标: 用纯权重特征(不需要前向传播)预测kl_div
    权重特征: W_down_norm, W_up_norm, W_gate_norm, PR, kappa, top10, layer_frac
    
    方法:
    1. 收集所有层的权重特征
    2. 用kl_div作为目标
    3. 多条文本平均
    4. 留一交叉验证
    """
    print(f"\n{'='*60}")
    print(f"P504: 纯权重特征预测importance [{model_name}]")
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
    sample_layers = list(range(n_layers))
    
    # 先收集所有层的权重特征(只需要做一次)
    weight_features = {}
    for l_idx in sample_layers:
        lw = get_layer_weights(layers_list[l_idx], d_model, mlp_type)
        W_down = lw.W_down
        W_up = lw.W_up
        W_gate = lw.W_gate
        
        W_down_norm = np.linalg.norm(W_down)
        W_up_norm = np.linalg.norm(W_up) if W_up is not None else 0
        W_gate_norm = np.linalg.norm(W_gate) if W_gate is not None else 0
        
        try:
            from sklearn.utils.extmath import randomized_svd
            _, s_wd, _ = randomized_svd(W_down.astype(np.float32), n_components=min(100, min(W_down.shape)-1), random_state=42)
            PR_wd = compute_participation_ratio(s_wd)
            d_eff_wd = compute_effective_dimension(s_wd)
            kappa_wd = s_wd[0] / max(s_wd[-1], 1e-10)
            top10_wd = np.sum(s_wd[:min(10, len(s_wd))]**2) / max(np.sum(s_wd**2), 1e-10)
            # W_down Frobenius norm / 核范数比
            frobenius = np.sqrt(np.sum(s_wd**2))
            nuclear = np.sum(s_wd)
            fn_ratio = frobenius / max(nuclear, 1e-10)
        except:
            PR_wd = d_eff_wd = kappa_wd = top10_wd = fn_ratio = 0
        
        # LN权重
        ln_weight = lw.input_layernorm_weight
        ln_norm = np.linalg.norm(ln_weight) if ln_weight is not None else 0
        post_ln_weight = lw.post_attn_layernorm_weight
        post_ln_norm = np.linalg.norm(post_ln_weight) if post_ln_weight is not None else 0
        
        weight_features[l_idx] = {
            'W_down_norm': W_down_norm,
            'W_up_norm': W_up_norm,
            'W_gate_norm': W_gate_norm,
            'PR_wd': PR_wd,
            'd_eff_wd': d_eff_wd,
            'kappa_wd': kappa_wd,
            'top10_wd': top10_wd,
            'fn_ratio': fn_ratio,
            'ln_norm': ln_norm,
            'post_ln_norm': post_ln_norm,
            'layer_frac': l_idx / max(n_layers - 1, 1),
        }
    
    print(f"  权重特征收集完成: {n_layers}层")
    
    # 收集kl_div数据(多条文本)
    all_results = []
    
    for text_idx, text in enumerate(test_texts):
        print(f"  文本 {text_idx+1}/{len(test_texts)}...", end="", flush=True)
        
        for l_idx in sample_layers:
            meas = compute_importance_measures(
                model, tokenizer, device, text,
                layers_list, l_idx, alpha, mlp_type
            )
            
            if meas is None:
                continue
            
            result = {
                'layer': l_idx,
                'text_idx': text_idx,
                'kl_div': meas['kl_div'],
                'importance_ppl': meas['importance_ppl'],
                'delta_h_relative': meas['delta_h_relative'],
                **weight_features[l_idx],
            }
            all_results.append(result)
        
        print(" done")
    
    # 释放模型
    del model
    torch.cuda.empty_cache()
    
    # 统计分析
    if len(all_results) < 20:
        print("  数据不足, 无法统计分析")
        return all_results
    
    print(f"\n--- P504 统计分析 [{model_name}] ---")
    print(f"  总数据点: {len(all_results)}")
    
    from sklearn.linear_model import LinearRegression, Lasso, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    
    # 权重特征列表
    weight_feat_names = [
        'W_down_norm', 'W_up_norm', 'W_gate_norm',
        'PR_wd', 'd_eff_wd', 'kappa_wd', 'top10_wd', 'fn_ratio',
        'ln_norm', 'post_ln_norm', 'layer_frac',
    ]
    
    # 准备数据
    X_all = np.array([[r[fn] for fn in weight_feat_names] for r in all_results])
    y_all = np.array([r['kl_div'] for r in all_results])
    groups = np.array([r['layer'] for r in all_results])
    
    # 1. 线性回归(全数据)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_all)
    
    ols = LinearRegression()
    ols.fit(X_s, y_all)
    y_pred = ols.predict(X_s)
    ss_res = np.sum((y_all - y_pred)**2)
    ss_tot = np.sum((y_all - np.mean(y_all))**2)
    r2_ols = 1 - ss_res / max(ss_tot, 1e-10)
    
    print(f"\n  === OLS (全数据) ===")
    print(f"  R2={r2_ols:.3f}")
    for fn, c in zip(weight_feat_names, ols.coef_):
        print(f"    {fn}: {c:.6f}")
    
    # 2. Lasso (全数据)
    lasso = Lasso(alpha=0.0001)
    lasso.fit(X_s, y_all)
    y_pred_l = lasso.predict(X_s)
    ss_res_l = np.sum((y_all - y_pred_l)**2)
    r2_lasso = 1 - ss_res_l / max(ss_tot, 1e-10)
    
    print(f"\n  === Lasso (全数据) ===")
    print(f"  R2={r2_lasso:.3f}")
    for fn, c in zip(weight_feat_names, lasso.coef_):
        print(f"    {fn}: {c:.6f}")
    
    # 3. 对数空间回归
    log_mask = (y_all > 1e-30)
    if np.sum(log_mask) >= 20:
        X_log = X_s[log_mask]
        y_log = np.log(y_all[log_mask])
        
        ols_log = LinearRegression()
        ols_log.fit(X_log, y_log)
        y_pred_log = ols_log.predict(X_log)
        ss_res_log = np.sum((y_log - y_pred_log)**2)
        ss_tot_log = np.sum((y_log - np.mean(y_log))**2)
        r2_log = 1 - ss_res_log / max(ss_tot_log, 1e-10)
        
        print(f"\n  === 对数空间OLS ===")
        print(f"  R2={r2_log:.3f}")
        for fn, c in zip(weight_feat_names, ols_log.coef_):
            print(f"    {fn}: {c:.4f}")
    
    # 4. 决策树(非线性的)
    dt = DecisionTreeRegressor(max_depth=5, min_samples_leaf=3)
    dt.fit(X_s, y_all)
    y_pred_dt = dt.predict(X_s)
    ss_res_dt = np.sum((y_all - y_pred_dt)**2)
    r2_dt = 1 - ss_res_dt / max(ss_tot, 1e-10)
    
    print(f"\n  === 决策树 (max_depth=5) ===")
    print(f"  R2={r2_dt:.3f}")
    
    # 特征重要性
    imp = dt.feature_importances_
    for fn, i in sorted(zip(weight_feat_names, imp), key=lambda x: -x[1]):
        print(f"    {fn}: {i:.4f}")
    
    # 5. Gradient Boosting(更强非线性)
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    gb.fit(X_s, y_all)
    y_pred_gb = gb.predict(X_s)
    ss_res_gb = np.sum((y_all - y_pred_gb)**2)
    r2_gb = 1 - ss_res_gb / max(ss_tot, 1e-10)
    
    print(f"\n  === Gradient Boosting ===")
    print(f"  R2={r2_gb:.3f}")
    imp_gb = gb.feature_importances_
    for fn, i in sorted(zip(weight_feat_names, imp_gb), key=lambda x: -x[1]):
        print(f"    {fn}: {i:.4f}")
    
    # 6. 留一层交叉验证(LOLO - Leave One Layer Out)
    print(f"\n  === 留一层交叉验证 (LOLO) ===")
    logo = LeaveOneGroupOut()
    
    # OLS LOLO
    try:
        scores_ols = cross_val_score(LinearRegression(), X_s, y_all, cv=logo.split(X_s, y_all, groups), scoring='r2')
        print(f"  OLS LOLO R2: mean={np.mean(scores_ols):.3f}, std={np.std(scores_ols):.3f}")
    except:
        print(f"  OLS LOLO: 失败(可能某些层样本太少)")
    
    # GB LOLO
    try:
        scores_gb = cross_val_score(
            GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42),
            X_s, y_all, cv=logo.split(X_s, y_all, groups), scoring='r2'
        )
        print(f"  GB LOLO R2: mean={np.mean(scores_gb):.3f}, std={np.std(scores_gb):.3f}")
    except:
        print(f"  GB LOLO: 失败")
    
    # 7. 逐层平均后的预测
    from collections import defaultdict
    layer_data = defaultdict(lambda: {'kl_divs': [], 'feats': []})
    for r in all_results:
        layer_data[r['layer']]['kl_divs'].append(r['kl_div'])
        layer_data[r['layer']]['feats'].append([r[fn] for fn in weight_feat_names])
    
    X_layer_avg = []
    y_layer_avg = []
    for l in sorted(layer_data.keys()):
        X_layer_avg.append(np.mean(layer_data[l]['feats'], axis=0))
        y_layer_avg.append(np.mean(layer_data[l]['kl_divs']))
    
    X_layer_avg = np.array(X_layer_avg)
    y_layer_avg = np.array(y_layer_avg)
    
    scaler_layer = StandardScaler()
    X_layer_s = scaler_layer.fit_transform(X_layer_avg)
    
    ols_layer = LinearRegression()
    ols_layer.fit(X_layer_s, y_layer_avg)
    y_pred_layer = ols_layer.predict(X_layer_s)
    ss_res_layer = np.sum((y_layer_avg - y_pred_layer)**2)
    ss_tot_layer = np.sum((y_layer_avg - np.mean(y_layer_avg))**2)
    r2_layer = 1 - ss_res_layer / max(ss_tot_layer, 1e-10)
    
    print(f"\n  === 逐层平均后OLS ===")
    print(f"  R2={r2_layer:.3f}")
    for fn, c in zip(weight_feat_names, ols_layer.coef_):
        print(f"    {fn}: {c:.6f}")
    
    # 保存
    out_dir = os.path.join(project_root, 'tests', 'glm5_temp')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"p504_{model_name}.json")
    
    results_clean = []
    for r in all_results:
        rc = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
        results_clean.append(rc)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results_clean, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果已保存: {out_path}")
    
    return all_results


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase CVI: importance的非线性解析")
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"],
                       help="模型名称")
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["p502", "p503", "p504", "all"],
                       help="实验编号")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    if args.experiment == "p502":
        run_p502(args.model, device)
    elif args.experiment == "p503":
        run_p503(args.model, device)
    elif args.experiment == "p504":
        run_p504(args.model, device)
    elif args.experiment == "all":
        print("依次运行P502, P503, P504...")
        run_p502(args.model, device)
        run_p503(args.model, device)
        run_p504(args.model, device)
    
    print("\n完成!")


if __name__ == "__main__":
    main()
