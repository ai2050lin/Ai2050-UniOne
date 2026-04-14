"""
Phase CVII-P505/P506/P507: Post-LN权重的物理角色与层重要性理论
============================================================

Phase CVI核心瓶颈:
- 纯权重预测kl_div逐层平均R2=0.54-0.76
- post_ln_norm是最重要权重特征(DT重要性>0.5)
- 但LOLO交叉验证泛化失败(R2为负)
- 不理解post_ln_norm的物理角色

Phase CVII核心思路:
1. 理解Post-LN权重在transformer残差连接中的物理角色
2. Post-LN控制了attn/FFN信号→残差流的缩放比例
3. 如果post_ln_norm大→attn/FFN信号缩放大→层扰动影响大→importance高

关键数学:
  h_l = h_{l-1} + post_ln_l(attn_l(h_{l-1})) + ffn_part
  post_ln(x) = gamma * (x - mu) / sigma + beta
  当gamma(post_ln_weight)范数大→信号缩放大

P505: Post-LN权重与残差信号缩放
  - 测量每层attn输出范数、post_ln后范数、缩放比
  - 验证: post_ln_norm与attn信号缩放比的相关性
  - 验证: 缩放比与kl_div的相关性
  - 分析: post_ln_norm为什么能预测层重要性

P506: 层重要性的统一理论
  - 整合: importance ≈ f(post_ln_norm, J_chain, W_down_norm, layer_frac)
  - 推导解析公式: kl_div ∝ post_ln_norm^a × J_chain^b × W_down^c
  - 验证跨层平均后的预测能力
  - 对比不同公式的预测效果

P507: 因果干预验证
  - 直接干预post_ln_norm: 缩放post_ln_weight
  - 验证: 缩放post_ln_norm后kl_div是否按预期变化
  - 因果链: post_ln_norm → 信号缩放 → delta_h → kl_div

使用方法:
    python phase_cvii_ln_causal_theory.py --model qwen3 --experiment p505
    python phase_cvii_ln_causal_theory.py --model glm4 --experiment p506
    python phase_cvii_ln_causal_theory.py --model deepseek7b --experiment p507
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


def compute_participation_ratio(s):
    s_sq = s**2
    s_sq_norm = s_sq / (np.sum(s_sq) + 1e-30)
    return 1.0 / (len(s) * np.sum(s_sq_norm**2) + 1e-30)


def compute_kl_divergence(logits_baseline, logits_ablated):
    p = torch.nn.functional.softmax(logits_baseline, dim=-1)
    q = torch.nn.functional.log_softmax(logits_ablated, dim=-1)
    kl = torch.nn.functional.kl_div(q, p, reduction='batchmean')
    return kl.item()


def compute_residual_stream_with_attn(model, input_ids, device):
    """收集每层的: 隐状态、attn输出(post-ln前)、attn输出(post-ln后)、ffn输出(post-ln后)"""
    layer_data = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0].detach()
            else:
                hidden = output.detach()
            layer_data[layer_idx]['hidden'] = hidden[0, -1].cpu().float().numpy()
        return hook_fn

    layers_list = get_layers(model)
    hooks = []

    for i, layer in enumerate(layers_list):
        layer_data.append({
            'hidden': None,
            'attn_output_pre_ln': None,
            'attn_output_post_ln': None,
        })

        # Hook: 整层输出
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    try:
        with torch.no_grad():
            model(input_ids)
    except Exception as e:
        print(f"  [hook] Forward failed: {e}")

    for h in hooks:
        h.remove()

    return layer_data


def perturb_w_down(layers, l_idx, alpha, mlp_type):
    orig_weight = layers[l_idx].mlp.down_proj.weight.data.clone()
    layers[l_idx].mlp.down_proj.weight.data = orig_weight * (1 - alpha)
    return orig_weight


def restore_w_down(layers, l_idx, orig_weight, mlp_type):
    layers[l_idx].mlp.down_proj.weight.data = orig_weight


def perturb_post_ln(layers, l_idx, scale_factor):
    """缩放post_ln权重: gamma *= scale_factor"""
    layer = layers[l_idx]
    post_ln = None
    for ln_name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
        if hasattr(layer, ln_name):
            post_ln = getattr(layer, ln_name)
            break

    if post_ln is None or not hasattr(post_ln, 'weight'):
        return None

    orig_weight = post_ln.weight.data.clone()
    post_ln.weight.data = orig_weight * scale_factor
    return orig_weight


def restore_post_ln(layers, l_idx, orig_weight):
    """恢复post_ln权重"""
    layer = layers[l_idx]
    for ln_name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
        if hasattr(layer, ln_name):
            post_ln = getattr(layer, ln_name)
            post_ln.weight.data = orig_weight
            return
    raise ValueError(f"Cannot find post_ln in layer {l_idx}")


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

    restore_w_down(layers, l_idx, orig_weight, mlp_type)

    ablated_pred_logits = ablated_logits[:-1]
    ablated_loss = loss_fn(ablated_pred_logits, baseline_next_tokens).item()
    ablated_ppl = np.exp(min(ablated_loss, 20))

    delta_ppl = ablated_ppl - baseline_ppl
    importance_ppl = delta_ppl / max(baseline_ppl, 1e-10)
    kl_div = compute_kl_divergence(baseline_pred_logits, ablated_pred_logits)

    return {
        'importance_ppl': importance_ppl,
        'kl_div': kl_div,
        'baseline_ppl': baseline_ppl,
        'alpha': alpha,
        'l_idx': l_idx,
    }


def get_attn_and_ffn_norms(model, tokenizer, device, text, layers_list, n_layers, mlp_type):
    """
    测量每层的attn输出范数和FFN输出范数(在post-ln之后)

    在transformer中:
    h_l = h_{l-1} + post_ln(attn_output) + ffn_output (RMSNorm/LN后)

    我们需要:
    1. h_{l-1} (输入)
    2. attn_output (attention输出, pre-LN后)
    3. h_l (输出)

    ffn_output = h_l - h_{l-1} - post_ln(attn_output)
    """

    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # 收集所有层的隐状态
    hidden_states = []
    attn_outputs = []

    def make_hidden_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states.append(output[0][0, -1].detach().cpu().float().numpy())
            else:
                hidden_states.append(output[0, -1].detach().cpu().float().numpy())
        return hook_fn

    def make_attn_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                attn_outputs.append(output[0][0, -1].detach().cpu().float().numpy())
            else:
                attn_outputs.append(output[0, -1].detach().cpu().float().numpy())
        return hook_fn

    hooks = []
    for i, layer in enumerate(layers_list):
        # Hook on entire layer for hidden states
        h1 = layer.register_forward_hook(make_hidden_hook(i))
        # Hook on self_attn for attention output
        h2 = layer.self_attn.register_forward_hook(make_attn_hook(i))
        hooks.extend([h1, h2])

    try:
        with torch.no_grad():
            model(input_ids)
    except Exception as e:
        print(f"  [hook] Forward failed: {e}")

    for h in hooks:
        h.remove()

    if len(hidden_states) < n_layers:
        print(f"  Warning: only got {len(hidden_states)} hidden states, expected {n_layers}")
        return None

    results = []
    for l in range(min(len(hidden_states), n_layers)):
        h_in = hidden_states[l] if l == 0 else hidden_states[l-1]  # 输入隐状态
        h_out = hidden_states[l]  # 输出隐状态

        # 残差增量
        delta = h_out - h_in
        delta_norm = np.linalg.norm(delta)

        # attention输出范数
        attn_out = attn_outputs[l] if l < len(attn_outputs) else None
        attn_norm = np.linalg.norm(attn_out) if attn_out is not None else 0

        # h_in norm
        h_in_norm = np.linalg.norm(h_in)

        # 增量相对大小
        delta_relative = delta_norm / max(h_in_norm, 1e-10)

        # 获取post_ln权重范数
        lw = get_layer_weights(layers_list[l], h_in.shape[0], mlp_type)
        post_ln_norm = np.linalg.norm(lw.post_attn_layernorm_weight) if lw.post_attn_layernorm_weight is not None else 0
        ln_norm = np.linalg.norm(lw.input_layernorm_weight) if lw.input_layernorm_weight is not None else 0

        results.append({
            'layer': l,
            'h_in_norm': h_in_norm,
            'h_out_norm': np.linalg.norm(h_out),
            'delta_norm': delta_norm,
            'delta_relative': delta_relative,
            'attn_norm': attn_norm,
            'post_ln_norm': post_ln_norm,
            'ln_norm': ln_norm,
            'layer_frac': l / max(n_layers - 1, 1),
        })

    return results


# ============================================================
# P505: Post-LN权重与残差信号缩放
# ============================================================

def run_p505(model_name, device):
    """
    P505: Post-LN权重与残差信号缩放

    核心问题: post_ln_norm为什么是最重要的权重特征?

    假设: Post-LN控制了attn/FFN信号在残差连接中的缩放比例
      h_l = h_{l-1} + gamma_post * LN(attn_out) + gamma_post_ffn * LN(ffn_out)
      当gamma_post范数大 → 信号注入残差流的缩放大 → 扰动该层影响大

    方法:
    1. 测量每层的: h_in_norm, delta_norm, attn_norm, post_ln_norm
    2. 验证: post_ln_norm vs delta_norm的相关性
    3. 验证: post_ln_norm vs attn_norm的相关性
    4. 分析: post_ln_norm → 信号缩放 → importance的因果链
    """
    print(f"\n{'='*60}")
    print(f"P505: Post-LN权重与残差信号缩放 [{model_name}]")
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

    # Part 1: 测量每层的信号特征
    print("\n--- Part 1: 层间信号特征测量 ---")
    all_layer_features = {}

    for text_idx, text in enumerate(test_texts):
        print(f"  文本 {text_idx+1}/{len(test_texts)}...", end="", flush=True)
        layer_data = get_attn_and_ffn_norms(model, tokenizer, device, text, layers_list, n_layers, mlp_type)
        if layer_data is None:
            print(" FAILED")
            continue

        for ld in layer_data:
            l = ld['layer']
            if l not in all_layer_features:
                all_layer_features[l] = {
                    'h_in_norm': [], 'h_out_norm': [], 'delta_norm': [],
                    'delta_relative': [], 'attn_norm': [],
                    'post_ln_norm': ld['post_ln_norm'],
                    'ln_norm': ld['ln_norm'],
                    'layer_frac': ld['layer_frac'],
                }
            all_layer_features[l]['h_in_norm'].append(ld['h_in_norm'])
            all_layer_features[l]['h_out_norm'].append(ld['h_out_norm'])
            all_layer_features[l]['delta_norm'].append(ld['delta_norm'])
            all_layer_features[l]['delta_relative'].append(ld['delta_relative'])
            all_layer_features[l]['attn_norm'].append(ld['attn_norm'])
        print(" done")

    # Part 2: 测量kl_div
    print("\n--- Part 2: kl_div测量 ---")
    alpha = 0.1
    all_kl_results = []

    for text_idx, text in enumerate(test_texts):
        print(f"  文本 {text_idx+1}/{len(test_texts)}...", end="", flush=True)

        for l_idx in range(n_layers):
            meas = compute_importance_measures(
                model, tokenizer, device, text,
                layers_list, l_idx, alpha, mlp_type
            )
            if meas is None:
                continue
            all_kl_results.append({
                'layer': l_idx,
                'text_idx': text_idx,
                'kl_div': meas['kl_div'],
                'importance_ppl': meas['importance_ppl'],
            })
        print(" done")

    # 释放模型
    del model
    torch.cuda.empty_cache()

    # Part 3: 统计分析
    print("\n--- Part 3: 统计分析 ---")

    # 逐层平均
    layer_avg_kl = {}
    layer_avg_features = {}

    for r in all_kl_results:
        l = r['layer']
        if l not in layer_avg_kl:
            layer_avg_kl[l] = []
        layer_avg_kl[l].append(r['kl_div'])

    for l, feats in all_layer_features.items():
        layer_avg_features[l] = {
            'h_in_norm': np.mean(feats['h_in_norm']),
            'h_out_norm': np.mean(feats['h_out_norm']),
            'delta_norm': np.mean(feats['delta_norm']),
            'delta_relative': np.mean(feats['delta_relative']),
            'attn_norm': np.mean(feats['attn_norm']),
            'post_ln_norm': feats['post_ln_norm'],
            'ln_norm': feats['ln_norm'],
            'layer_frac': feats['layer_frac'],
        }

    # 合并数据
    combined = []
    for l in layer_avg_kl:
        if l in layer_avg_features:
            row = {'layer': l, 'kl_div_avg': np.mean(layer_avg_kl[l])}
            row.update(layer_avg_features[l])
            combined.append(row)

    if len(combined) < 5:
        print("  数据不足")
        return combined

    # 相关性分析
    print("\n=== P505 核心结果 [{0}] ===".format(model_name))

    # 1. post_ln_norm vs delta_norm
    post_ln_norms = [c['post_ln_norm'] for c in combined]
    delta_norms = [c['delta_norm'] for c in combined]
    delta_relatives = [c['delta_relative'] for c in combined]
    attn_norms = [c['attn_norm'] for c in combined]
    kl_divs = [c['kl_div_avg'] for c in combined]
    layer_fracs = [c['layer_frac'] for c in combined]
    h_in_norms = [c['h_in_norm'] for c in combined]

    r1, p1 = pearsonr(post_ln_norms, delta_norms)
    r2, p2 = pearsonr(post_ln_norms, delta_relatives)
    r3, p3 = pearsonr(post_ln_norms, attn_norms)
    r4, p4 = pearsonr(post_ln_norms, kl_divs)
    r5, p5 = pearsonr(delta_norms, kl_divs)
    r6, p6 = pearsonr(attn_norms, kl_divs)
    r7, p7 = pearsonr(layer_fracs, post_ln_norms)
    r8, p8 = pearsonr(h_in_norms, post_ln_norms)

    print(f"\n  ** Post-LN权重的物理角色 **")
    print(f"  post_ln_norm vs delta_norm(残差增量绝对值): r={r1:.3f}, p={p1:.4f}")
    print(f"  post_ln_norm vs delta_relative(残差增量相对值): r={r2:.3f}, p={p2:.4f}")
    print(f"  post_ln_norm vs attn_norm(attn输出范数): r={r3:.3f}, p={p3:.4f}")
    print(f"  post_ln_norm vs kl_div(语言能力影响): r={r4:.3f}, p={p4:.4f}")
    print(f"\n  ** 因果链验证 **")
    print(f"  delta_norm vs kl_div: r={r5:.3f}, p={p5:.4f}")
    print(f"  attn_norm vs kl_div: r={r6:.3f}, p={p6:.4f}")
    print(f"\n  ** post_ln_norm的层位置依赖 **")
    print(f"  layer_frac vs post_ln_norm: r={r7:.3f}, p={p7:.4f}")
    print(f"  h_in_norm vs post_ln_norm: r={r8:.3f}, p={p8:.4f}")

    # 2. 多元回归: kl_div ~ post_ln_norm + delta_norm + attn_norm + layer_frac
    try:
        from sklearn.linear_model import LinearRegression, Lasso
        from sklearn.preprocessing import StandardScaler

        X = np.column_stack([
            post_ln_norms, delta_norms, attn_norms,
            layer_fracs, h_in_norms,
        ])
        y = np.array(kl_divs)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        reg = LinearRegression()
        reg.fit(X_scaled, y)
        r2_multi = reg.score(X_scaled, y)
        print(f"\n  ** 多元回归: kl_div ~ post_ln + delta + attn + lfrac + h_in **")
        print(f"  OLS R2 = {r2_multi:.3f}")
        print(f"  系数: post_ln={reg.coef_[0]:.4f}, delta={reg.coef_[1]:.4f}, attn={reg.coef_[2]:.4f}")
        print(f"         lfrac={reg.coef_[3]:.4f}, h_in={reg.coef_[4]:.4f}")

        lasso = Lasso(alpha=0.01)
        lasso.fit(X_scaled, y)
        r2_lasso = lasso.score(X_scaled, y)
        print(f"  Lasso R2 = {r2_lasso:.3f}")
        print(f"  Lasso系数: post_ln={lasso.coef_[0]:.4f}, delta={lasso.coef_[1]:.4f}, attn={lasso.coef_[2]:.4f}")
        print(f"              lfrac={lasso.coef_[3]:.4f}, h_in={lasso.coef_[4]:.4f}")
    except Exception as e:
        print(f"  回归分析失败: {e}")

    # 3. 关键比率: delta_norm / post_ln_norm
    ratios = [d / max(p, 1e-10) for d, p in zip(delta_norms, post_ln_norms)]
    print(f"\n  ** 信号缩放比 delta_norm / post_ln_norm **")
    print(f"  mean={np.mean(ratios):.4f}, std={np.std(ratios):.4f}, CV={np.std(ratios)/max(np.mean(ratios),1e-10):.3f}")
    r_ratio, p_ratio = pearsonr(ratios, kl_divs)
    print(f"  ratio vs kl_div: r={r_ratio:.3f}, p={p_ratio:.4f}")

    # 保存结果
    results = {
        'model': model_name,
        'experiment': 'p505',
        'correlations': {
            'post_ln_vs_delta_norm': {'r': r1, 'p': p1},
            'post_ln_vs_delta_relative': {'r': r2, 'p': p2},
            'post_ln_vs_attn_norm': {'r': r3, 'p': p3},
            'post_ln_vs_kl_div': {'r': r4, 'p': p4},
            'delta_norm_vs_kl_div': {'r': r5, 'p': p5},
            'attn_norm_vs_kl_div': {'r': r6, 'p': p6},
            'layer_frac_vs_post_ln': {'r': r7, 'p': p7},
            'h_in_vs_post_ln': {'r': r8, 'p': p8},
            'ratio_vs_kl_div': {'r': r_ratio, 'p': p_ratio},
        },
        'layer_data': combined,
    }

    out_path = os.path.join(project_root, 'tests', 'glm5_temp', f'p505_{model_name}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  结果保存到: {out_path}")

    return results


# ============================================================
# P506: 层重要性的统一理论
# ============================================================

def run_p506(model_name, device):
    """
    P506: 层重要性的统一理论

    整合所有发现，推导importance的解析公式。

    已知关系:
    1. gain = f(||J_{L:l}||), r>0.72 (Phase CV)
    2. ||J_{L:l}|| vs layer_frac r<-0.97 (Phase CV)
    3. h_l_norm vs delta_h_norm_at_l r>0.97 (Phase CVI)
    4. post_ln_norm是最重要权重特征 (Phase CVI)
    5. GB R2=0.44-0.84 (Phase CVI)

    公式候选:
    A) kl_div ~ post_ln_norm^a × layer_frac^b × W_down^c  (纯权重)
    B) kl_div ~ post_ln_norm × delta_norm × J_chain       (信号传播)
    C) kl_div ~ post_ln_norm × (1-layer_frac)^d × h_in_norm^e

    方法:
    1. 收集所有层的权重特征+kl_div
    2. 对比多种公式的预测效果
    3. 找到最佳解析公式
    """
    print(f"\n{'='*60}")
    print(f"P506: 层重要性的统一理论 [{model_name}]")
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

    # 收集权重特征
    print("\n--- 权重特征收集 ---")
    weight_features = {}
    for l_idx in range(n_layers):
        lw = get_layer_weights(layers_list[l_idx], d_model, mlp_type)
        W_down = lw.W_down
        W_up = lw.W_up
        W_gate = lw.W_gate

        W_down_norm = np.linalg.norm(W_down)
        W_up_norm = np.linalg.norm(W_up) if W_up is not None else 0
        W_gate_norm = np.linalg.norm(W_gate) if W_gate is not None else 0
        W_o_norm = np.linalg.norm(lw.W_o)

        ln_norm = np.linalg.norm(lw.input_layernorm_weight) if lw.input_layernorm_weight is not None else 0
        post_ln_norm = np.linalg.norm(lw.post_attn_layernorm_weight) if lw.post_attn_layernorm_weight is not None else 0

        try:
            from sklearn.utils.extmath import randomized_svd
            _, s_wd, _ = randomized_svd(W_down.astype(np.float32), n_components=min(100, min(W_down.shape)-1), random_state=42)
            kappa_wd = s_wd[0] / max(s_wd[-1], 1e-10)
            fn_ratio = np.sqrt(np.sum(s_wd**2)) / max(np.sum(s_wd), 1e-10)
        except:
            kappa_wd = fn_ratio = 0

        weight_features[l_idx] = {
            'W_down_norm': W_down_norm,
            'W_up_norm': W_up_norm,
            'W_gate_norm': W_gate_norm,
            'W_o_norm': W_o_norm,
            'ln_norm': ln_norm,
            'post_ln_norm': post_ln_norm,
            'kappa_wd': kappa_wd,
            'fn_ratio': fn_ratio,
            'layer_frac': l_idx / max(n_layers - 1, 1),
        }

    print(f"  权重特征收集完成: {n_layers}层")

    # 收集kl_div和信号特征
    print("\n--- kl_div和信号特征收集 ---")
    all_results = []
    all_layer_signals = {}

    for text_idx, text in enumerate(test_texts):
        print(f"  文本 {text_idx+1}/{len(test_texts)}...", end="", flush=True)

        # 先收集信号特征
        layer_data = get_attn_and_ffn_norms(model, tokenizer, device, text, layers_list, n_layers, mlp_type)
        if layer_data is not None:
            for ld in layer_data:
                l = ld['layer']
                if l not in all_layer_signals:
                    all_layer_signals[l] = {'delta_norm': [], 'attn_norm': [], 'h_in_norm': []}
                all_layer_signals[l]['delta_norm'].append(ld['delta_norm'])
                all_layer_signals[l]['attn_norm'].append(ld['attn_norm'])
                all_layer_signals[l]['h_in_norm'].append(ld['h_in_norm'])

        # 收集kl_div
        for l_idx in range(n_layers):
            meas = compute_importance_measures(
                model, tokenizer, device, text,
                layers_list, l_idx, alpha, mlp_type
            )
            if meas is None:
                continue
            all_results.append({
                'layer': l_idx,
                'text_idx': text_idx,
                'kl_div': meas['kl_div'],
            })
        print(" done")

    # 释放模型
    del model
    torch.cuda.empty_cache()

    # 逐层平均
    layer_avg_kl = {}
    for r in all_results:
        l = r['layer']
        if l not in layer_avg_kl:
            layer_avg_kl[l] = []
        layer_avg_kl[l].append(r['kl_div'])

    layer_avg_signals = {}
    for l, sigs in all_layer_signals.items():
        layer_avg_signals[l] = {
            'delta_norm_avg': np.mean(sigs['delta_norm']),
            'attn_norm_avg': np.mean(sigs['attn_norm']),
            'h_in_norm_avg': np.mean(sigs['h_in_norm']),
        }

    # 合并
    combined = []
    for l in layer_avg_kl:
        if l in weight_features and l in layer_avg_signals:
            row = {'layer': l, 'kl_div_avg': np.mean(layer_avg_kl[l])}
            row.update(weight_features[l])
            row.update(layer_avg_signals[l])
            combined.append(row)

    if len(combined) < 5:
        print("  数据不足")
        return combined

    print(f"\n  合并数据: {len(combined)}层")

    # 公式对比
    print("\n=== P506 核心结果 [{0}] ===".format(model_name))

    kl_divs = [c['kl_div_avg'] for c in combined]
    log_kl = np.log(np.array(kl_divs) + 1e-20)

    # 公式A: 纯权重
    # A1: post_ln_norm^a × (1-layer_frac)^b
    # A2: post_ln_norm × W_down_norm × (1-layer_frac)^b
    # A3: post_ln_norm^a × kappa_wd^b

    from sklearn.linear_model import LinearRegression

    formulas = {}

    # A1: log(kl_div) ~ log(post_ln) + log(1-lfrac)
    try:
        X_a1 = np.column_stack([
            np.log(np.array([c['post_ln_norm'] for c in combined]) + 1e-10),
            np.log(np.array([1 - c['layer_frac'] + 0.01 for c in combined])),
        ])
        reg_a1 = LinearRegression().fit(X_a1, log_kl)
        r2_a1 = reg_a1.score(X_a1, log_kl)
        formulas['A1: log(kl)~log(post_ln)+log(1-lfrac)'] = {
            'r2': r2_a1,
            'coefs': {'log_post_ln': reg_a1.coef_[0], 'log_1_lfrac': reg_a1.coef_[1]},
        }
    except:
        pass

    # A2: log(kl) ~ log(post_ln) + log(W_down) + log(1-lfrac)
    try:
        X_a2 = np.column_stack([
            np.log(np.array([c['post_ln_norm'] for c in combined]) + 1e-10),
            np.log(np.array([c['W_down_norm'] for c in combined]) + 1e-10),
            np.log(np.array([1 - c['layer_frac'] + 0.01 for c in combined])),
        ])
        reg_a2 = LinearRegression().fit(X_a2, log_kl)
        r2_a2 = reg_a2.score(X_a2, log_kl)
        formulas['A2: log(kl)~log(post_ln)+log(W_down)+log(1-lfrac)'] = {
            'r2': r2_a2,
            'coefs': {
                'log_post_ln': reg_a2.coef_[0],
                'log_W_down': reg_a2.coef_[1],
                'log_1_lfrac': reg_a2.coef_[2],
            },
        }
    except:
        pass

    # A3: log(kl) ~ log(post_ln) + log(W_o) + log(1-lfrac)
    try:
        X_a3 = np.column_stack([
            np.log(np.array([c['post_ln_norm'] for c in combined]) + 1e-10),
            np.log(np.array([c['W_o_norm'] for c in combined]) + 1e-10),
            np.log(np.array([1 - c['layer_frac'] + 0.01 for c in combined])),
        ])
        reg_a3 = LinearRegression().fit(X_a3, log_kl)
        r2_a3 = reg_a3.score(X_a3, log_kl)
        formulas['A3: log(kl)~log(post_ln)+log(W_o)+log(1-lfrac)'] = {
            'r2': r2_a3,
            'coefs': {
                'log_post_ln': reg_a3.coef_[0],
                'log_W_o': reg_a3.coef_[1],
                'log_1_lfrac': reg_a3.coef_[2],
            },
        }
    except:
        pass

    # B1: kl ~ post_ln × delta_norm × (1-lfrac)^d
    # 线性空间: kl ~ post_ln + delta + lfrac (交互)
    try:
        X_b1 = np.column_stack([
            np.array([c['post_ln_norm'] for c in combined]),
            np.array([c['delta_norm_avg'] for c in combined]),
            np.array([1 - c['layer_frac'] for c in combined]),
            np.array([c['post_ln_norm'] for c in combined]) * np.array([c['delta_norm_avg'] for c in combined]),
        ])
        reg_b1 = LinearRegression().fit(X_b1, kl_divs)
        r2_b1 = reg_b1.score(X_b1, kl_divs)
        formulas['B1: kl~post_ln+delta+1-lfrac+interaction'] = {
            'r2': r2_b1,
            'coefs': {
                'post_ln': reg_b1.coef_[0],
                'delta': reg_b1.coef_[1],
                '1_lfrac': reg_b1.coef_[2],
                'post_ln_x_delta': reg_b1.coef_[3],
            },
        }
    except:
        pass

    # B2: kl ~ post_ln × attn_norm
    try:
        X_b2 = np.column_stack([
            np.array([c['post_ln_norm'] for c in combined]),
            np.array([c['attn_norm_avg'] for c in combined]),
            np.array([1 - c['layer_frac'] for c in combined]),
        ])
        reg_b2 = LinearRegression().fit(X_b2, kl_divs)
        r2_b2 = reg_b2.score(X_b2, kl_divs)
        formulas['B2: kl~post_ln+attn+1-lfrac'] = {'r2': r2_b2}
    except:
        pass

    # C1: 全特征OLS
    try:
        feat_names = ['post_ln_norm', 'ln_norm', 'W_down_norm', 'W_up_norm',
                      'W_gate_norm', 'W_o_norm', 'kappa_wd', 'fn_ratio', 'layer_frac']
        X_c1 = np.column_stack([np.array([c[f] for c in combined]) for f in feat_names])
        reg_c1 = LinearRegression().fit(X_c1, kl_divs)
        r2_c1 = reg_c1.score(X_c1, kl_divs)
        formulas['C1: kl~all_weight_features'] = {'r2': r2_c1}
    except:
        pass

    # C2: 全特征+信号
    try:
        feat_names2 = ['post_ln_norm', 'W_down_norm', 'delta_norm_avg', 'attn_norm_avg',
                       'h_in_norm_avg', 'layer_frac']
        X_c2 = np.column_stack([np.array([c[f] for c in combined]) for f in feat_names2])
        reg_c2 = LinearRegression().fit(X_c2, kl_divs)
        r2_c2 = reg_c2.score(X_c2, kl_divs)
        formulas['C2: kl~post_ln+W_down+delta+attn+h_in+lfrac'] = {'r2': r2_c2}
    except:
        pass

    # 打印结果
    print(f"\n  ** 公式对比 (逐层平均, n={len(combined)}) **")
    sorted_formulas = sorted(formulas.items(), key=lambda x: -x[1]['r2'])
    for name, data in sorted_formulas:
        print(f"  {name}: R2={data['r2']:.3f}")
        if 'coefs' in data:
            for k, v in data['coefs'].items():
                print(f"    {k} = {v:.4f}")

    # 最佳公式的详细分析
    if sorted_formulas:
        best_name, best_data = sorted_formulas[0]
        print(f"\n  ** 最佳公式: {best_name} **")
        print(f"  R2 = {best_data['r2']:.3f}")

        # 残差分析
        if 'coefs' in best_data:
            print(f"  幂律指数:")
            for k, v in best_data['coefs'].items():
                if k.startswith('log_'):
                    print(f"    {k}: 指数={v:.3f} (1.0=线性, >1超线性, <1亚线性)")

    # 保存结果
    results = {
        'model': model_name,
        'experiment': 'p506',
        'formulas': {k: {kk: vv for kk, vv in v.items()} for k, v in formulas.items()},
        'layer_data': combined,
    }

    out_path = os.path.join(project_root, 'tests', 'glm5_temp', f'p506_{model_name}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  结果保存到: {out_path}")

    return results


# ============================================================
# P507: 因果干预验证
# ============================================================

def run_p507(model_name, device):
    """
    P507: 因果干预验证

    如果post_ln_norm → 信号缩放 → importance的因果链成立，
    那么直接缩放post_ln_norm应该按预期改变kl_div。

    假设: 缩放post_ln_norm为原来的s倍 → kl_div变为原来的s^a倍
    (a待估计，如果post_ln是线性缩放则a≈1)

    方法:
    1. 对选定层(5-8层)，缩放post_ln_norm: s = 0.5, 0.8, 1.2, 1.5, 2.0
    2. 测量kl_div的变化
    3. 验证: kl_div(s) / kl_div(1.0) ≈ s^a
    4. 对比: W_down扰动的kl_div vs post_ln缩放的kl_div
    """
    print(f"\n{'='*60}")
    print(f"P507: 因果干预验证 [{model_name}]")
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

    alpha = 0.1  # W_down扰动幅度

    # 选择有代表性的层(浅、中浅、中、中深、深)
    sample_layers = [
        max(0, n_layers // 8),
        n_layers // 4,
        n_layers // 2,
        3 * n_layers // 4,
        min(n_layers - 1, 7 * n_layers // 8),
    ]
    # 去重
    sample_layers = sorted(list(set(sample_layers)))
    print(f"  采样层: {sample_layers}")

    scale_factors = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]

    all_results = []

    for text_idx, text in enumerate(test_texts):
        print(f"\n  文本 {text_idx+1}/{len(test_texts)}: '{text[:50]}...'")
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        for l_idx in sample_layers:
            print(f"    层 {l_idx}...", end="", flush=True)

            # Step 1: 基线kl_div (W_down扰动, alpha=0.1)
            meas_baseline = compute_importance_measures(
                model, tokenizer, device, text,
                layers_list, l_idx, alpha, mlp_type
            )
            kl_baseline = meas_baseline['kl_div'] if meas_baseline else 0

            # Step 2: 对post_ln施加不同缩放
            for s in scale_factors:
                if s == 1.0:
                    kl_with_scale = kl_baseline
                else:
                    # 缩放post_ln
                    orig_ln = perturb_post_ln(layers_list, l_idx, s)
                    if orig_ln is None:
                        print(f" (no post_ln at L{l_idx})", end="", flush=True)
                        continue

                    # 在缩放后的post_ln下，测量W_down扰动的kl_div
                    meas_scaled = compute_importance_measures(
                        model, tokenizer, device, text,
                        layers_list, l_idx, alpha, mlp_type
                    )
                    kl_with_scale = meas_scaled['kl_div'] if meas_scaled else 0

                    # 恢复post_ln
                    restore_post_ln(layers_list, l_idx, orig_ln)

                result = {
                    'layer': l_idx,
                    'text_idx': text_idx,
                    'scale_factor': s,
                    'kl_div': kl_with_scale,
                    'kl_div_baseline': kl_baseline,
                    'kl_ratio': kl_with_scale / max(kl_baseline, 1e-20),
                }
                all_results.append(result)

            print(" done")

    # 同时收集: 纯post_ln缩放(不扰动W_down)的kl_div
    print("\n--- 纯Post-LN缩放效应(不扰动W_down) ---")
    for text_idx, text in enumerate(test_texts):
        print(f"  文本 {text_idx+1}...", end="", flush=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        # 基线logits
        with torch.no_grad():
            baseline_outputs = model(input_ids)
            baseline_logits = baseline_outputs.logits if hasattr(baseline_outputs, 'logits') else baseline_outputs
            if baseline_logits.dim() == 3:
                baseline_logits = baseline_logits[0]
            baseline_pred_logits = baseline_logits[:-1]

        for l_idx in sample_layers:
            for s in [0.5, 1.5, 2.0]:
                orig_ln = perturb_post_ln(layers_list, l_idx, s)
                if orig_ln is None:
                    continue

                with torch.no_grad():
                    scaled_outputs = model(input_ids)
                    scaled_logits = scaled_outputs.logits if hasattr(scaled_outputs, 'logits') else scaled_outputs
                    if scaled_logits.dim() == 3:
                        scaled_logits = scaled_logits[0]
                    scaled_pred_logits = scaled_logits[:-1]

                kl_pure_ln = compute_kl_divergence(baseline_pred_logits, scaled_pred_logits)

                all_results.append({
                    'layer': l_idx,
                    'text_idx': text_idx,
                    'scale_factor': s,
                    'kl_div': kl_pure_ln,
                    'type': 'pure_ln_scale',
                    'kl_div_baseline': 0,
                    'kl_ratio': 0,
                })

                restore_post_ln(layers_list, l_idx, orig_ln)
        print(" done")

    # 释放模型
    del model
    torch.cuda.empty_cache()

    # 统计分析
    print("\n=== P507 核心结果 [{0}] ===".format(model_name))

    # 分析W_down扰动下的kl_div vs scale_factor
    wd_results = [r for r in all_results if 'type' not in r]

    if len(wd_results) > 10:
        print("\n  ** Post-LN缩放对W_down扰动kl_div的影响 **")

        for l_idx in sample_layers:
            layer_data = [r for r in wd_results if r['layer'] == l_idx]
            if not layer_data:
                continue

            # 逐scale_factor平均
            scale_avg = {}
            for r in layer_data:
                s = r['scale_factor']
                if s not in scale_avg:
                    scale_avg[s] = []
                scale_avg[s].append(r['kl_ratio'])

            print(f"\n    层 {l_idx}:")
            for s in sorted(scale_avg.keys()):
                avg_ratio = np.mean(scale_avg[s])
                print(f"      scale={s:.1f}: kl_ratio={avg_ratio:.3f}")

            # 拟合kl_ratio = s^a
            try:
                scales = sorted(scale_avg.keys())
                ratios = [np.mean(scale_avg[s]) for s in scales]
                # 取log: log(ratio) = a * log(s)
                log_s = [np.log(max(s, 0.01)) for s in scales]
                log_r = [np.log(max(r, 1e-10)) for r in ratios]

                # 只用s!=1.0的数据拟合
                fit_idx = [i for i, s in enumerate(scales) if s != 1.0]
                if len(fit_idx) >= 2:
                    a_fit = np.polyfit(
                        [log_s[i] for i in fit_idx],
                        [log_r[i] for i in fit_idx],
                        1
                    )[0]
                    print(f"      幂律指数 a = {a_fit:.3f} (a=1: 线性缩放, a≠1: 非线性)")
            except:
                pass

    # 分析纯post_ln缩放
    pure_ln_results = [r for r in all_results if r.get('type') == 'pure_ln_scale']
    if pure_ln_results:
        print("\n  ** 纯Post-LN缩放的kl_div(不扰动W_down) **")
        for l_idx in sample_layers:
            layer_data = [r for r in pure_ln_results if r['layer'] == l_idx]
            if not layer_data:
                continue

            scale_avg = {}
            for r in layer_data:
                s = r['scale_factor']
                if s not in scale_avg:
                    scale_avg[s] = []
                scale_avg[s].append(r['kl_div'])

            print(f"    层 {l_idx}:")
            for s in sorted(scale_avg.keys()):
                avg_kl = np.mean(scale_avg[s])
                print(f"      scale={s:.1f}: kl_div={avg_kl:.6f}")

    # 保存结果
    results = {
        'model': model_name,
        'experiment': 'p507',
        'sample_layers': sample_layers,
        'scale_factors': scale_factors,
        'results': all_results,
    }

    out_path = os.path.join(project_root, 'tests', 'glm5_temp', f'p507_{model_name}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  结果保存到: {out_path}")

    return results


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Phase CVII: Post-LN权重的物理角色')
    parser.add_argument('--model', type=str, required=True,
                        choices=['qwen3', 'glm4', 'deepseek7b'])
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['p505', 'p506', 'p507', 'all'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA不可用, 使用CPU")

    os.makedirs(os.path.join(project_root, 'tests', 'glm5_temp'), exist_ok=True)

    if args.experiment == 'all':
        run_p505(args.model, device)
        run_p506(args.model, device)
        run_p507(args.model, device)
    elif args.experiment == 'p505':
        run_p505(args.model, device)
    elif args.experiment == 'p506':
        run_p506(args.model, device)
    elif args.experiment == 'p507':
        run_p507(args.model, device)


if __name__ == '__main__':
    main()
