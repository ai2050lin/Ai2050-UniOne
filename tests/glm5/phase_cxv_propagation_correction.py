"""
Phase CXV-CXVI-P534/P535/P536/P537: 统一传播方程的修正
=======================================================

Phase CXIV核心发现:
- 1-秩双重机制: Qwen3 W_down激活子矩阵低秩(PR=0.97); DS7B中层PR仅0.49但仍1-秩→额外机制
- v_max不稳定根源: f_mlp激活模式层间完全不同(top10_overlap≈0)
- DS7B频谱集中机制: RL训练功能性选择激活列(对齐W_U top 0.63 vs 随机0.29)
- 频谱有序≠因果链有效: DS7B W_down频谱层间相关0.97但因果链断裂
- 统一传播方程DS7B失败: 预测误差k50=0.36

Phase CXV-CXVI核心思路:
1. P534: DS7B 1-秩的额外机制 — W_down*J_LN复合效应分析
   - 即使W_down激活子矩阵PR=0.49, 乘以J_LN后可能进一步降秩
   - 验证: W_down_active * J_LN的PR vs W_down_active的PR

2. P535: 修正传播方程 — 加入残差连接"信号保持"项
   - 当前方程假设信号沿v_max方向传播, 但实际信号走残差"旁路"
   - 修正: ratio(k) = α*residual_preservation(k) + β*MLP_contribution(k)
   - 其中residual_preservation来自层归一化后的信号保持

3. P536: 从激活模式预测ratio(k) — 用f_mlp的W_U对齐度预测
   - 核心洞察: 不是v_max方向, 而是f_mlp的激活模式决定了信号走向
   - 用激活列的W_U对齐度(而非v_max方向)预测每层ratio(k)

4. P537: DS7B因果链断裂的深层原因 — λ_max遮蔽分析
   - 为什么频谱有序但因果链无效?
   - 假设: λ_max被其他因素"遮蔽", 如LayerNorm归一化、残差稀释
   - 验证: 去除残差连接后的因果链; LayerNorm对λ_max的衰减因子

使用方法:
    python phase_cxv_propagation_correction.py --model qwen3 --experiment p534
    python phase_cxv_propagation_correction.py --model glm4 --experiment p535
    python phase_cxv_propagation_correction.py --model deepseek7b --experiment p536
"""

import sys
import os
import argparse
import numpy as np
import torch
import json
import time
from scipy.stats import spearmanr, pearsonr
from scipy.sparse.linalg import svds

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))
from model_utils import (
    load_model, get_model_info, get_layers, get_layer_weights,
    get_W_U, release_model, get_sample_layers
)


def to_native(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.item() if v.numel() == 1 else v.cpu().tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def compute_f_mlp(h_in, W_gate, W_up, post_ln_w):
    """计算MLP中间激活 f(LN(h_in)) = σ(W_gate * LN(h_in)) ⊙ (W_up * LN(h_in))"""
    h_in_centered = h_in - h_in.mean()
    sigma = np.std(h_in_centered) + 1e-10
    ln_h_in = h_in_centered / sigma
    if post_ln_w is not None:
        ln_h_in = ln_h_in * post_ln_w
    
    if W_gate is not None:
        gate_pre = W_gate @ ln_h_in
        gate_act = 1.0 / (1.0 + np.exp(-gate_pre))
    else:
        gate_act = np.ones(W_up.shape[0])
    
    up_out = W_up @ ln_h_in
    f_mlp = gate_act * up_out
    return f_mlp, ln_h_in


def compute_jacobian_probes(model, layers, l_idx, inputs, info, n_probes=30, alpha=0.01):
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_hidden_states=True)
        h_out_baseline = outputs.hidden_states[l_idx + 1][0, -1].detach().clone()
    
    J_probes = []
    for i in range(n_probes):
        orig_w = layers[l_idx].mlp.down_proj.weight.data.clone()
        layers[l_idx].mlp.down_proj.weight.data = orig_w * (1 - alpha)
        
        with torch.no_grad():
            out = model(inputs["input_ids"], output_hidden_states=True)
            h_perturbed = out.hidden_states[l_idx + 1][0, -1].detach().clone()
        
        layers[l_idx].mlp.down_proj.weight.data = orig_w
        delta_h = (h_perturbed - h_out_baseline) / alpha
        J_probes.append(delta_h.cpu().float())
        
        if i % 20 == 0:
            torch.cuda.empty_cache()
    
    J_proj = torch.stack(J_probes)
    return J_proj, h_out_baseline


# ===== P534: DS7B 1-秩的额外机制 =====
def run_p534(model, tokenizer, device, model_name):
    """
    DS7B 1-秩的额外机制: W_down*J_LN复合效应
    
    Phase CXIV发现: DS7B中层PR(W_active)=0.49但Jacobian仍1-秩
    说明1-秩不仅来自W_down激活子矩阵低秩, 还有额外机制
    
    假设: LayerNorm的Jacobian J_LN将信号压缩到低维空间,
    使得 W_down_active * J_LN 的秩更低
    
    J_LN = (1/σ) * (I - μ·1^T - h_norm·σ^T/σ)
    其中μ和σ是输入的均值和方差
    
    验证:
    1. W_down_active的PR vs W_down_active*J_LN的PR
    2. J_LN本身的秩/参与率
    3. W_down_active*(diag(f')*W_combined)*J_LN的PR — 完整链路
    """
    print("\n" + "="*70)
    print("P534: DS7B 1-秩的额外机制 — W_down*J_LN复合效应")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    test_text = "The fundamental nature of reality can be understood through mathematical structures that underlie physical phenomena."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    sample_layers = get_sample_layers(info.n_layers, min(6, info.n_layers))
    print(f"采样层: {sample_layers}")
    
    results = []
    
    for l_idx in sample_layers:
        print(f"\n--- 层 {l_idx} ---")
        
        weights = get_layer_weights(layers[l_idx], info.d_model, info.mlp_type)
        W_down = weights.W_down       # [d_model, intermediate]
        W_gate = weights.W_gate       # [intermediate, d_model]
        W_up = weights.W_up           # [intermediate, d_model]
        post_ln_w = weights.post_attn_layernorm_weight  # [d_model]
        
        # 获取隐藏状态
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
            h_states = outputs.hidden_states
            h_in = h_states[l_idx][0, -1].cpu().float().numpy()
        
        # 计算f_mlp
        f_mlp, ln_h_in = compute_f_mlp(h_in, W_gate, W_up, post_ln_w)
        f_mlp_abs = np.abs(f_mlp)
        
        # σ'(z) for sigmoid
        if W_gate is not None:
            gate_pre = W_gate @ ln_h_in
            gate_act = 1.0 / (1.0 + np.exp(-gate_pre))
            gate_deriv = gate_act * (1 - gate_act)
        else:
            gate_deriv = np.ones(W_up.shape[0])
        
        # f' ≈ σ'(z) * (W_up·x)
        f_deriv = gate_deriv * (W_up @ ln_h_in)
        f_deriv_abs = np.abs(f_deriv)
        
        # ===== Step 1: W_down_active的PR =====
        for k_active in [5, 10, 20]:
            top_k_idx = np.argsort(f_deriv_abs)[-k_active:]
            W_down_k = W_down[:, top_k_idx]  # [d_model, k_active]
            f_deriv_k = f_deriv[top_k_idx]
            
            # W_down_active = W_down_k * diag(f_deriv_k)
            W_active = W_down_k * f_deriv_k[np.newaxis, :]
            
            if min(W_active.shape) > 1:
                U_wa, S_wa, _ = svds(W_active.astype(np.float32), k=min(5, min(W_active.shape)-1))
                S_wa = S_wa[::-1]
                PR_W_active = float(S_wa.sum()**2 / (len(S_wa) * (S_wa**2).sum() + 1e-10))
            else:
                PR_W_active = 1.0
            
            # ===== Step 2: 计算J_LN =====
            # LayerNorm: y = (x - μ) / σ * γ
            # J_LN = (1/σ) * (I - (1/n)*1*1^T - y_norm*(1/n)*1^T) * diag(γ)
            # 简化: 用数值方法计算J_LN
            h_in_centered = h_in - h_in.mean()
            sigma_h = np.std(h_in_centered) + 1e-10
            ln_h = h_in_centered / sigma_h
            if post_ln_w is not None:
                ln_h_scaled = ln_h * post_ln_w
            else:
                ln_h_scaled = ln_h
            
            # 数值J_LN: 用扰动法
            eps = 1e-4
            J_LN_cols = []
            for dim in range(min(info.d_model, 50)):  # 采样50个维度
                h_pert = h_in.copy()
                h_pert[dim] += eps
                # 重新计算LN
                h_pert_centered = h_pert - h_pert.mean()
                sigma_pert = np.std(h_pert_centered) + 1e-10
                ln_pert = h_pert_centered / sigma_pert
                if post_ln_w is not None:
                    ln_pert = ln_pert * post_ln_w
                J_LN_col = (ln_pert - ln_h_scaled) / eps
                J_LN_cols.append(J_LN_col)
            
            J_LN_sampled = np.column_stack(J_LN_cols)  # [d_model, 50]
            
            # J_LN的秩 (用参与率)
            if min(J_LN_sampled.shape) > 1:
                U_jln, S_jln, _ = svds(J_LN_sampled.astype(np.float32), k=min(10, min(J_LN_sampled.shape)-1))
                S_jln = S_jln[::-1]
                PR_J_LN = float(S_jln.sum()**2 / (len(S_jln) * (S_jln**2).sum() + 1e-10))
                top1_J_LN = float(S_jln[0]**2 / (np.sum(S_jln**2) + 1e-10))
            else:
                PR_J_LN = 1.0
                top1_J_LN = 1.0
            
            # ===== Step 3: W_down_active * W_up_active * J_LN 的复合PR =====
            # 完整Jacobian路径: W_down * diag(f') * W_up * J_LN
            # W_active = W_down[:, top_k] * diag(f_deriv_k): [d_model, k_active]
            # W_up[top_k, :]: [k_active, d_model]
            # J_LN_sampled: [d_model, 50]
            # 复合: W_active @ W_up[top_k, :] @ J_LN_sampled → [d_model, 50]
            
            W_up_topk = W_up[top_k_idx, :]  # [k_active, d_model]
            W_up_JLN = W_up_topk @ J_LN_sampled  # [k_active, 50]
            W_active_JLN = W_active @ W_up_JLN   # [d_model, 50]... 不对
            # W_active是[d_model, k_active], W_up_JLN是[k_active, 50] → [d_model, 50] ✓
            # 但这实际上是: W_down*diag(f')*W_up*J_LN 的采样版本
            # 注意这里少了一项: diag(f')应该在W_up之后乘(因为gate和up是并行的)
            # 完整: J_MLP = W_down * diag(gate_act * (W_up @ x)) * (gate_deriv * (W_up @ x) + gate_act * W_up) * J_LN
            # 简化为: W_down * diag(f_deriv) * W_up * J_LN (主项)
            
            # 直接用数值方法: full_chain = W_active @ W_up_JLN
            compound_matrix = W_active @ W_up_JLN  # [d_model, 50]
            
            if min(compound_matrix.shape) > 1:
                U_cj, S_cj, _ = svds(compound_matrix.astype(np.float32), k=min(5, min(compound_matrix.shape)-1))
                S_cj = S_cj[::-1]
                PR_compound = float(S_cj.sum()**2 / (len(S_cj) * (S_cj**2).sum() + 1e-10))
                top1_compound = float(S_cj[0]**2 / (np.sum(S_cj**2) + 1e-10))
            else:
                PR_compound = 1.0
                top1_compound = 1.0
            
            # 保存k_active=10的结果 (主要关注)
            if k_active == 10:
                result_k10 = {
                    'PR_W_active_k10': PR_W_active,
                    'PR_J_LN': PR_J_LN,
                    'top1_J_LN': top1_J_LN,
                    'PR_compound_k10': PR_compound,
                    'top1_compound_k10': top1_compound,
                }
        
        # ===== Step 4: 完整链路验证 =====
        # W_down * diag(f_deriv) * W_up * J_LN (简化: 省略gate)
        top10_idx = np.argsort(f_deriv_abs)[-10:]
        W_down_10 = W_down[:, top10_idx]  # [d_model, 10]
        W_up_10 = W_up[top10_idx, :]      # [10, d_model]
        f_deriv_10 = f_deriv[top10_idx]    # [10]
        
        # 完整: W_down_10 * diag(f_deriv_10) * W_up_10 * J_LN
        full_chain = W_down_10 * f_deriv_10[np.newaxis, :]  # [d_model, 10]
        full_chain = full_chain @ W_up_10  # [d_model, d_model]
        full_chain = full_chain @ J_LN_sampled  # [d_model, 50]
        
        if min(full_chain.shape) > 1:
            U_fc, S_fc, _ = svds(full_chain.astype(np.float32), k=min(5, min(full_chain.shape)-1))
            S_fc = S_fc[::-1]
            PR_full_chain = float(S_fc.sum()**2 / (len(S_fc) * (S_fc**2).sum() + 1e-10))
            top1_full_chain = float(S_fc[0]**2 / (np.sum(S_fc**2) + 1e-10))
        else:
            PR_full_chain = 1.0
            top1_full_chain = 1.0
        
        # Jacobian probe验证
        J_proj, _ = compute_jacobian_probes(model, layers, l_idx, inputs, info, n_probes=60)
        J_proj_T = J_proj.T
        U_jt, S_j, _ = torch.linalg.svd(J_proj_T, full_matrices=False)
        S_j_np = S_j.cpu().numpy()
        total_energy = np.sum(S_j_np**2)
        n_90 = int(np.searchsorted(np.cumsum(S_j_np**2) / total_energy, 0.9)) + 1
        top1_J_measured = float(S_j_np[0]**2 / total_energy)
        
        result = {
            'layer': l_idx,
            'n_90_measured': n_90,
            'top1_J_measured': top1_J_measured,
            'lambda_max': float(S_j[0]),
            **result_k10,
            'PR_full_chain_k10': PR_full_chain,
            'top1_full_chain_k10': top1_full_chain,
        }
        results.append(result)
        
        print(f"  n_90={n_90}, top1_J={top1_J_measured:.6f}")
        print(f"  PR(W_active_k10)={result_k10['PR_W_active_k10']:.4f}")
        print(f"  PR(J_LN)={PR_J_LN:.4f}, top1_J_LN={top1_J_LN:.4f}")
        print(f"  PR(compound_k10)={result_k10['PR_compound_k10']:.4f}")
        print(f"  PR(full_chain_k10)={PR_full_chain:.4f}")
        
        # 关键判断: compound PR是否比W_active PR更低
        if PR_compound < result_k10['PR_W_active_k10']:
            print(f"  >> J_LN进一步降秩! compound({PR_compound:.4f}) < W_active({result_k10['PR_W_active_k10']:.4f})")
        
        torch.cuda.empty_cache()
    
    # 汇总
    print("\n" + "="*70)
    print("P534 汇总: 1-秩额外机制")
    print("="*70)
    
    mean_PR_active = np.mean([r['PR_W_active_k10'] for r in results])
    mean_PR_JLN = np.mean([r['PR_J_LN'] for r in results])
    mean_PR_compound = np.mean([r['PR_compound_k10'] for r in results])
    mean_PR_full = np.mean([r['PR_full_chain_k10'] for r in results])
    mean_top1_JLN = np.mean([r['top1_J_LN'] for r in results])
    
    print(f"  mean PR(W_active_k10): {mean_PR_active:.4f}")
    print(f"  mean PR(J_LN): {mean_PR_JLN:.4f}")
    print(f"  mean top1(J_LN): {mean_top1_JLN:.4f}")
    print(f"  mean PR(compound): {mean_PR_compound:.4f}")
    print(f"  mean PR(full_chain): {mean_PR_full:.4f}")
    
    print("\n  额外机制分析:")
    if mean_PR_compound < mean_PR_active - 0.05:
        print("  >> J_LN确实进一步降秩 -> 1-秩的双重来源")
        print(f"     W_active贡献: PR={mean_PR_active:.4f}")
        print(f"     J_LN贡献: PR={mean_PR_JLN:.4f}, top1={mean_top1_JLN:.4f}")
        print(f"     复合效应: PR={mean_PR_compound:.4f} (降低{mean_PR_active-mean_PR_compound:.4f})")
    else:
        print("  >> J_LN未显著降秩 -> 额外机制不在J_LN")
        if mean_top1_JLN > 0.5:
            print("  >> J_LN本身也极度1-秩 -> 可能J_LN和W_active共同作用")
    
    result_path = f"tests/glm5/results/phase_cxv/p534_{model_name}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({'results': [to_native(r) for r in results], 'model': model_name}, f, indent=2)
    print(f"\n结果已保存: {result_path}")
    
    return results


# ===== P535: 修正传播方程 =====
def run_p535(model, tokenizer, device, model_name):
    """
    修正传播方程: 加入残差连接"信号保持"项
    
    Phase CXIV发现: 统一传播方程在DS7B失败(误差k50=0.36)
    原因: v_max不反映实际信号传播方向, 信号走残差"旁路"
    
    修正模型:
    h_out = LN(h_in + MLP(h_in)) = LN(h_in) * γ_res + J_MLP * h_in
    
    在W_U奇异空间中:
    ratio(k) = α * preservation(k) + β * contribution(k)
    
    preservation(k): h_in在W_U top-k方向上的能量, 经LN后保留的比例
    contribution(k): MLP贡献在W_U top-k方向上的能量
    
    验证:
    1. 实测preservation(k)和contribution(k)
    2. 拟合α,β参数
    3. 对比修正方程与原方程的预测精度
    """
    print("\n" + "="*70)
    print("P535: 修正传播方程 — 残差保持+MLP贡献")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    test_text = "The fundamental nature of reality can be understood through mathematical structures that underlie physical phenomena."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    sample_layers = get_sample_layers(info.n_layers, min(8, info.n_layers))
    print(f"采样层: {sample_layers}")
    
    # W_U SVD
    W_U = get_W_U(model)
    k_wu = min(200, min(W_U.shape) - 2)
    print(f"计算W_U SVD (k={k_wu})...")
    U_wu, S_wu, _ = svds(W_U.T.astype(np.float32), k=k_wu)
    U_wu = U_wu[:, ::-1]
    
    # 目标ratio(k): 从实测delta_h计算
    target_k_values = [10, 20, 50, 100, 200]
    
    # 收集所有层的h_in, h_out, delta_h
    layer_data = []
    for l_idx in sample_layers:
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
            h_states = outputs.hidden_states
            h_in = h_states[l_idx][0, -1].cpu().float().numpy()
            h_out = h_states[l_idx + 1][0, -1].cpu().float().numpy()
        
        delta_h = h_out - h_in
        mlp_out = delta_h  # 残差连接: h_out = h_in + MLP(LN(h_in))
        
        # h_in在W_U空间中的频谱
        h_in_wu = U_wu.T @ h_in  # [k_wu]
        h_in_wu_energy = h_in_wu**2
        h_in_total = np.sum(h_in_wu_energy)
        
        # h_out在W_U空间中的频谱
        h_out_wu = U_wu.T @ h_out
        h_out_wu_energy = h_out_wu**2
        h_out_total = np.sum(h_out_wu_energy)
        
        # delta_h (MLP贡献)在W_U空间中的频谱
        delta_wu = U_wu.T @ delta_h
        delta_wu_energy = delta_wu**2
        delta_total = np.sum(delta_wu_energy)
        
        # 实测ratio(k)
        measured_ratios = {}
        for k in target_k_values:
            top_k_idx = np.argsort(h_out_wu_energy)[-k:]
            measured_ratios[k] = float(np.sum(h_out_wu_energy[top_k_idx]) / (h_out_total + 1e-10))
        
        # preservation(k): h_in的top-k能量经残差连接后的保留比例
        # 经过层后: h_out = h_in + delta_h
        # h_out在W_U top-k方向的能量 = h_in在top-k方向的能量 + delta_h在top-k方向的能量
        preservation_ratios = {}
        contribution_ratios = {}
        for k in target_k_values:
            top_k_idx = np.argsort(h_out_wu_energy)[-k:]
            h_in_in_topk = float(np.sum(h_in_wu_energy[top_k_idx]) / (h_in_total + 1e-10))
            delta_in_topk = float(np.sum(delta_wu_energy[top_k_idx]) / (delta_total + 1e-10))
            preservation_ratios[k] = h_in_in_topk
            contribution_ratios[k] = delta_in_topk
        
        layer_data.append({
            'layer': l_idx,
            'measured_ratios': measured_ratios,
            'preservation_ratios': preservation_ratios,
            'contribution_ratios': contribution_ratios,
            'h_in_total': h_in_total,
            'h_out_total': h_out_total,
            'delta_total': delta_total,
            'h_in': h_in,
            'h_out': h_out,
            'delta_h': delta_h,
        })
        
        print(f"  L{l_idx}: h_in_total={h_in_total:.2f}, delta_total={delta_total:.2f}, "
              f"ratio(50)={measured_ratios[50]:.4f}")
        torch.cuda.empty_cache()
    
    # ===== 修正方程拟合 =====
    # ratio(k) = α * preservation(k) + β * contribution(k) + γ
    # 用最小二乘法拟合α, β
    
    print("\n--- 修正方程拟合 ---")
    
    # 收集所有层所有k的数据点
    all_measured = []
    all_preservation = []
    all_contribution = []
    
    for ld in layer_data:
        for k in target_k_values:
            all_measured.append(ld['measured_ratios'][k])
            all_preservation.append(ld['preservation_ratios'][k])
            all_contribution.append(ld['contribution_ratios'][k])
    
    all_measured = np.array(all_measured)
    all_preservation = np.array(all_preservation)
    all_contribution = np.array(all_contribution)
    
    # 线性回归: measured = α * preservation + β * contribution + γ
    A = np.column_stack([all_preservation, all_contribution, np.ones_like(all_measured)])
    params, residuals, rank, sv = np.linalg.lstsq(A, all_measured, rcond=None)
    alpha_fit, beta_fit, gamma_fit = params
    
    predicted = alpha_fit * all_preservation + beta_fit * all_contribution + gamma_fit
    error = np.abs(predicted - all_measured)
    mean_error = float(np.mean(error))
    max_error = float(np.max(error))
    r2 = 1 - np.sum(error**2) / (np.sum((all_measured - np.mean(all_measured))**2) + 1e-10)
    
    print(f"  拟合参数: α={alpha_fit:.4f}, β={beta_fit:.4f}, γ={gamma_fit:.4f}")
    print(f"  预测误差: mean={mean_error:.4f}, max={max_error:.4f}")
    print(f"  R2={r2:.4f}")
    
    # 对比原方程(仅用v_max方向预测)
    print("\n--- 对比: 原方程(v_max方向) vs 修正方程 ---")
    
    # 原方程误差 (从CXIV结果看, 这里简化计算)
    # 用contribution_ratios作为原方程的近似(因为v_max预测的就是MLP贡献)
    original_predicted = all_contribution  # 简化: 原方程只考虑MLP
    original_error = np.abs(original_predicted - all_measured)
    original_mean_error = float(np.mean(original_error))
    
    print(f"  原方程误差: mean={original_mean_error:.4f}")
    print(f"  修正方程误差: mean={mean_error:.4f}")
    print(f"  改善: {(original_mean_error - mean_error) / original_mean_error * 100:.1f}%")
    
    # 逐k分析
    print("\n--- 逐k预测精度 ---")
    results = []
    for k in target_k_values:
        k_measured = [ld['measured_ratios'][k] for ld in layer_data]
        k_preservation = [ld['preservation_ratios'][k] for ld in layer_data]
        k_contribution = [ld['contribution_ratios'][k] for ld in layer_data]
        
        k_predicted = [alpha_fit * p + beta_fit * c + gamma_fit 
                       for p, c in zip(k_preservation, k_contribution)]
        k_error = np.abs(np.array(k_predicted) - np.array(k_measured))
        
        print(f"  k={k}: measured={np.mean(k_measured):.4f}, "
              f"predicted={np.mean(k_predicted):.4f}, "
              f"error={np.mean(k_error):.4f}")
        
        results.append({
            'k': k,
            'mean_measured': float(np.mean(k_measured)),
            'mean_predicted': float(np.mean(k_predicted)),
            'mean_error': float(np.mean(k_error)),
            'mean_preservation': float(np.mean(k_preservation)),
            'mean_contribution': float(np.mean(k_contribution)),
        })
    
    # 汇总
    print("\n" + "="*70)
    print("P535 汇总: 修正传播方程")
    print("="*70)
    
    print(f"  修正方程: ratio(k) = {alpha_fit:.4f} * preservation(k) + {beta_fit:.4f} * contribution(k) + {gamma_fit:.4f}")
    print(f"  R2 = {r2:.4f}")
    
    if alpha_fit > 0.5:
        print("  >> 残差保持是主导项 -> 信号主要通过残差连接传播")
    if beta_fit > 0.3:
        print("  >> MLP贡献显著 -> MLP确实在塑造频谱")
    if r2 > 0.9:
        print("  >> 修正方程高度精确!")
    elif r2 > 0.7:
        print("  >> 修正方程较好, 但仍有改善空间")
    else:
        print("  >> 修正方程不够精确, 需要更复杂模型")
    
    result_path = f"tests/glm5/results/phase_cxv/p535_{model_name}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({
            'results': [to_native(r) for r in results],
            'model': model_name,
            'alpha': float(alpha_fit),
            'beta': float(beta_fit),
            'gamma': float(gamma_fit),
            'r2': float(r2),
            'mean_error': float(mean_error),
        }, f, indent=2)
    print(f"\n结果已保存: {result_path}")
    
    return results


# ===== P536: 从激活模式预测ratio(k) =====
def run_p536(model, tokenizer, device, model_name):
    """
    用f_mlp的W_U对齐度预测每层ratio(k)
    
    Phase CXIV发现: 激活列的W_U对齐度(DS7B 0.63 vs 随机0.29)比v_max方向
    更能反映信号走向
    
    核心洞察: 
    - v_max方向每层变化, 不能直接用于预测ratio
    - 但f_mlp的激活模式决定了W_down哪些列被"选择"
    - 这些被选择的列的W_U对齐度决定了MLP输出在W_U空间中的分布
    
    预测模型:
    contribution_W_U_topk(l) = Σ_{i ∈ top_active} |f_mlp_i| * (W_down[:,i]在W_U top-k上的投影)
    
    验证:
    1. 激活模式预测的contribution vs 实测delta_h的W_U分布
    2. 对比v_max预测 vs 激活模式预测
    """
    print("\n" + "="*70)
    print("P536: 从激活模式预测ratio(k)")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    test_text = "The fundamental nature of reality can be understood through mathematical structures that underlie physical phenomena."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    sample_layers = get_sample_layers(info.n_layers, min(8, info.n_layers))
    print(f"采样层: {sample_layers}")
    
    # W_U SVD
    W_U = get_W_U(model)
    k_wu = min(200, min(W_U.shape) - 2)
    print(f"计算W_U SVD (k={k_wu})...")
    U_wu, S_wu, _ = svds(W_U.T.astype(np.float32), k=k_wu)
    U_wu = U_wu[:, ::-1]
    
    target_k_values = [10, 20, 50, 100]
    results = []
    
    for l_idx in sample_layers:
        print(f"\n--- 层 {l_idx} ---")
        
        weights = get_layer_weights(layers[l_idx], info.d_model, info.mlp_type)
        W_down = weights.W_down
        W_gate = weights.W_gate
        W_up = weights.W_up
        post_ln_w = weights.post_attn_layernorm_weight
        
        # 获取隐藏状态
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
            h_states = outputs.hidden_states
            h_in = h_states[l_idx][0, -1].cpu().float().numpy()
            h_out = h_states[l_idx + 1][0, -1].cpu().float().numpy()
        
        delta_h = h_out - h_in
        
        # f_mlp计算
        f_mlp, ln_h_in = compute_f_mlp(h_in, W_gate, W_up, post_ln_w)
        f_mlp_abs = np.abs(f_mlp)
        
        # ===== 方法1: 激活模式预测 =====
        # W_down[:, i] * f_mlp[i] 在W_U top-k上的投影
        # 预测delta_h ≈ W_down @ f_mlp (简化, 忽略LN)
        predicted_delta = W_down @ f_mlp  # [d_model]
        
        # predicted_delta在W_U空间中的频谱
        pred_wu = U_wu.T @ predicted_delta
        pred_wu_energy = pred_wu**2
        pred_total = np.sum(pred_wu_energy)
        
        # 实测delta_h在W_U空间中的频谱
        actual_wu = U_wu.T @ delta_h
        actual_wu_energy = actual_wu**2
        actual_total = np.sum(actual_wu_energy)
        
        # 预测ratio(k) vs 实测ratio(k)
        pred_ratios = {}
        actual_ratios = {}
        for k in target_k_values:
            pred_topk = np.sum(np.sort(pred_wu_energy)[-k:]) / (pred_total + 1e-10)
            actual_topk = np.sum(np.sort(actual_wu_energy)[-k:]) / (actual_total + 1e-10)
            pred_ratios[k] = pred_topk
            actual_ratios[k] = actual_topk
        
        # ===== 方法2: v_max预测 (对比) =====
        J_proj, _ = compute_jacobian_probes(model, layers, l_idx, inputs, info, n_probes=60)
        J_proj_T = J_proj.T
        U_jt, S_j, _ = torch.linalg.svd(J_proj_T, full_matrices=False)
        v_max = U_jt[:, 0].cpu().float().numpy()
        v_max = v_max / (np.linalg.norm(v_max) + 1e-10)
        lambda_max = float(S_j[0])
        
        # v_max预测: delta_h方向 ≈ v_max, 大小 ≈ lambda_max * h_in_norm
        vmax_predicted_delta = v_max * lambda_max * np.linalg.norm(h_in)
        vmax_wu = U_wu.T @ vmax_predicted_delta
        vmax_wu_energy = vmax_wu**2
        vmax_total = np.sum(vmax_wu_energy)
        
        vmax_ratios = {}
        for k in target_k_values:
            vmax_topk = np.sum(np.sort(vmax_wu_energy)[-k:]) / (vmax_total + 1e-10)
            vmax_ratios[k] = vmax_topk
        
        # ===== 方法3: top激活列的W_U对齐度预测 =====
        # 只用top10激活列的方向预测
        top10_active_idx = np.argsort(f_mlp_abs)[-10:]
        top10_alignment = []
        for ai in top10_active_idx:
            col = W_down[:, ai]
            col_norm = np.linalg.norm(col)
            if col_norm < 1e-10:
                continue
            proj = U_wu.T @ (col / col_norm)
            proj_e = proj**2
            top10_e = np.sum(np.sort(proj_e)[-10:]) / (np.sum(proj_e) + 1e-10)
            top10_alignment.append(top10_e)
        mean_active_alignment = float(np.mean(top10_alignment)) if top10_alignment else 0
        
        # ===== 逐k对比 =====
        layer_result = {
            'layer': l_idx,
            'lambda_max': lambda_max,
            'mean_active_alignment': mean_active_alignment,
        }
        
        for k in target_k_values:
            pred_err = abs(pred_ratios[k] - actual_ratios[k])
            vmax_err = abs(vmax_ratios[k] - actual_ratios[k])
            
            layer_result[f'actual_ratio_{k}'] = actual_ratios[k]
            layer_result[f'pred_ratio_{k}'] = pred_ratios[k]
            layer_result[f'vmax_ratio_{k}'] = vmax_ratios[k]
            layer_result[f'pred_error_{k}'] = pred_err
            layer_result[f'vmax_error_{k}'] = vmax_err
        
        results.append(layer_result)
        
        print(f"  实测 ratio(k50)={actual_ratios[50]:.4f}")
        print(f"  激活模式预测 ratio(k50)={pred_ratios[50]:.4f}, error={abs(pred_ratios[50]-actual_ratios[50]):.4f}")
        print(f"  v_max预测 ratio(k50)={vmax_ratios[50]:.4f}, error={abs(vmax_ratios[50]-actual_ratios[50]):.4f}")
        print(f"  mean_active_alignment={mean_active_alignment:.4f}")
        
        torch.cuda.empty_cache()
    
    # 汇总
    print("\n" + "="*70)
    print("P536 汇总: 激活模式预测vs v_max预测")
    print("="*70)
    
    for k in target_k_values:
        pred_errors = [r[f'pred_error_{k}'] for r in results]
        vmax_errors = [r[f'vmax_error_{k}'] for r in results]
        print(f"  k={k}: 激活模式误差={np.mean(pred_errors):.4f}, v_max误差={np.mean(vmax_errors):.4f}")
    
    mean_active_align = np.mean([r['mean_active_alignment'] for r in results])
    print(f"\n  mean激活列W_U top10对齐度: {mean_active_align:.4f}")
    
    # 判断哪种方法更好
    mean_pred_err = np.mean([r[f'pred_error_50'] for r in results])
    mean_vmax_err = np.mean([r[f'vmax_error_50'] for r in results])
    
    print("\n  预测方法对比(k=50):")
    if mean_pred_err < mean_vmax_err:
        print(f"  >> 激活模式预测更优! error={mean_pred_err:.4f} vs {mean_vmax_err:.4f}")
    else:
        print(f"  >> v_max预测更优(意外): error={mean_vmax_err:.4f} vs {mean_pred_err:.4f}")
    
    result_path = f"tests/glm5/results/phase_cxv/p536_{model_name}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({'results': [to_native(r) for r in results], 'model': model_name}, f, indent=2)
    print(f"\n结果已保存: {result_path}")
    
    return results


# ===== P537: DS7B因果链断裂的深层原因 =====
def run_p537(model, tokenizer, device, model_name):
    """
    DS7B因果链断裂的深层原因: λ_max遮蔽分析
    
    Phase CXIV发现: DS7B W_down频谱层间相关0.97但因果链断裂
    λ_max与权重特征无关(ρ<0.15)
    
    核心假设:
    1. λ_max被LayerNorm归一化"遮蔽" — LN将信号重归一化, 使λ_max的效果被抵消
    2. 残差稀释 — λ_max虽然大, 但相对于残差连接的信号只是小修正
    3. 非线性截断 — SiLU/Sigmoid的非线性使大信号被截断
    
    验证:
    1. 计算每层的"有效放大率" = ||delta_h|| / ||h_in|| (包含LN和残差)
    2. 有效放大率 vs λ_max的相关性
    3. LayerNorm的"衰减因子": LN如何缩放MLP的输出
    4. 残差稀释因子: MLP贡献 / 总信号
    """
    print("\n" + "="*70)
    print("P537: 因果链断裂的深层原因 — λ_max遮蔽分析")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    test_text = "The fundamental nature of reality can be understood through mathematical structures that underlie physical phenomena."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    sample_layers = get_sample_layers(info.n_layers, min(8, info.n_layers))
    print(f"采样层: {sample_layers}")
    
    results = []
    
    for l_idx in sample_layers:
        print(f"\n--- 层 {l_idx} ---")
        
        weights = get_layer_weights(layers[l_idx], info.d_model, info.mlp_type)
        W_down = weights.W_down
        W_gate = weights.W_gate
        W_up = weights.W_up
        post_ln_w = weights.post_attn_layernorm_weight
        
        # 获取隐藏状态
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
            h_states = outputs.hidden_states
            h_in = h_states[l_idx][0, -1].cpu().float().numpy()
            h_out = h_states[l_idx + 1][0, -1].cpu().float().numpy()
        
        delta_h = h_out - h_in
        mlp_out = delta_h  # h_out = h_in + MLP(LN(h_in))
        
        # ===== 1. 有效放大率 =====
        h_in_norm = np.linalg.norm(h_in)
        delta_h_norm = np.linalg.norm(delta_h)
        effective_gain = delta_h_norm / (h_in_norm + 1e-10)
        
        # ===== 2. MLP贡献比例 (残差稀释因子) =====
        h_out_norm = np.linalg.norm(h_out)
        mlp_contribution_ratio = delta_h_norm / (h_out_norm + 1e-10)
        residual_contribution_ratio = h_in_norm / (h_out_norm + 1e-10)
        
        # ===== 3. LayerNorm衰减因子 =====
        # LN(h_in)的范数 vs h_in的范数
        h_in_centered = h_in - h_in.mean()
        sigma_h = np.std(h_in_centered) + 1e-10
        ln_h = h_in_centered / sigma_h
        if post_ln_w is not None:
            ln_h_scaled = ln_h * post_ln_w
        else:
            ln_h_scaled = ln_h
        
        ln_norm = np.linalg.norm(ln_h_scaled)
        LN_scale = ln_norm / (h_in_norm + 1e-10)  # LN的缩放因子
        
        # ===== 4. λ_max vs 有效放大率 =====
        J_proj, _ = compute_jacobian_probes(model, layers, l_idx, inputs, info, n_probes=60)
        J_proj_T = J_proj.T
        U_jt, S_j, _ = torch.linalg.svd(J_proj_T, full_matrices=False)
        lambda_max = float(S_j[0])
        
        # ===== 5. W_down范数 vs λ_max =====
        W_down_frobenius = np.linalg.norm(W_down)
        W_down_spectral = float(svds(W_down.astype(np.float32), k=1, return_singular_vectors=False)[0])
        
        # ===== 6. 非线性截断 =====
        f_mlp, ln_h_in = compute_f_mlp(h_in, W_gate, W_up, post_ln_w)
        # MLP的实际输出 vs 线性预测
        linear_mlp_out = W_down @ (W_up @ ln_h_in)  # 线性部分
        actual_mlp_out = W_down @ f_mlp              # 实际(含非线性)
        
        linear_norm = np.linalg.norm(linear_mlp_out)
        actual_norm = np.linalg.norm(actual_mlp_out)
        nonlinearity_ratio = actual_norm / (linear_norm + 1e-10)  # <1表示非线性截断
        
        # ===== 7. cos(delta_h, v_max) =====
        v_max = U_jt[:, 0].cpu().float().numpy()
        v_max = v_max / (np.linalg.norm(v_max) + 1e-10)
        cos_delta_vmax = abs(np.dot(delta_h / (delta_h_norm + 1e-10), v_max))
        
        # ===== 8. post-LN norm (如果存在) =====
        post_ln_norm = float(np.linalg.norm(post_ln_w)) if post_ln_w is not None else 1.0
        
        result = {
            'layer': l_idx,
            'lambda_max': lambda_max,
            'effective_gain': effective_gain,
            'mlp_contribution_ratio': mlp_contribution_ratio,
            'residual_contribution_ratio': residual_contribution_ratio,
            'LN_scale': LN_scale,
            'W_down_frobenius': W_down_frobenius,
            'W_down_spectral': W_down_spectral,
            'nonlinearity_ratio': nonlinearity_ratio,
            'cos_delta_vmax': cos_delta_vmax,
            'post_ln_norm': post_ln_norm,
            'h_in_norm': h_in_norm,
            'delta_h_norm': delta_h_norm,
            'h_out_norm': h_out_norm,
        }
        results.append(result)
        
        print(f"  λ_max={lambda_max:.2f}, effective_gain={effective_gain:.4f}")
        print(f"  MLP贡献比={mlp_contribution_ratio:.4f}, 残差贡献比={residual_contribution_ratio:.4f}")
        print(f"  LN缩放={LN_scale:.4f}, 非线性比={nonlinearity_ratio:.4f}")
        print(f"  cos(delta_h, v_max)={cos_delta_vmax:.4f}")
        
        torch.cuda.empty_cache()
    
    # ===== 跨层相关性分析 =====
    print("\n" + "="*70)
    print("P537 汇总: λ_max遮蔽分析")
    print("="*70)
    
    lambda_maxs = [r['lambda_max'] for r in results]
    effective_gains = [r['effective_gain'] for r in results]
    mlp_contributions = [r['mlp_contribution_ratio'] for r in results]
    LN_scales = [r['LN_scale'] for r in results]
    nonlinearity_ratios = [r['nonlinearity_ratio'] for r in results]
    cos_delta_vmaxs = [r['cos_delta_vmax'] for r in results]
    post_ln_norms = [r['post_ln_norm'] for r in results]
    W_down_spectrals = [r['W_down_spectral'] for r in results]
    
    # λ_max与各种因素的相关性
    if len(lambda_maxs) > 2:
        corr_gain, _ = pearsonr(lambda_maxs, effective_gains)
        corr_mlp, _ = pearsonr(lambda_maxs, mlp_contributions)
        corr_ln, _ = pearsonr(lambda_maxs, LN_scales)
        corr_nonlin, _ = pearsonr(lambda_maxs, nonlinearity_ratios)
        corr_Wdown, _ = pearsonr(lambda_maxs, W_down_spectrals)
        
        print(f"  λ_max vs effective_gain: r={corr_gain:.4f}")
        print(f"  λ_max vs MLP贡献比: r={corr_mlp:.4f}")
        print(f"  λ_max vs LN缩放: r={corr_ln:.4f}")
        print(f"  λ_max vs 非线性比: r={corr_nonlin:.4f}")
        print(f"  λ_max vs W_down谱范数: r={corr_Wdown:.4f}")
    
    mean_mlp_contrib = np.mean(mlp_contributions)
    mean_cos_delta = np.mean(cos_delta_vmaxs)
    mean_nonlin = np.mean(nonlinearity_ratios)
    
    print(f"\n  mean MLP贡献比: {mean_mlp_contrib:.4f}")
    print(f"  mean cos(delta_h, v_max): {mean_cos_delta:.4f}")
    print(f"  mean 非线性比: {mean_nonlin:.4f}")
    
    # 关键判断
    print("\n  遮蔽机制分析:")
    if mean_mlp_contrib < 0.3:
        print(f"  >> 残差稀释是主要遮蔽! MLP贡献仅{mean_mlp_contrib:.1%}")
        print("     λ_max虽大, 但相对于残差连接只是小修正")
    if mean_cos_delta < 0.3:
        print(f"  >> v_max方向与实际delta_h不对齐! cos={mean_cos_delta:.4f}")
        print("     λ_max方向不代表信号传播方向")
    if mean_nonlin < 0.5:
        print(f"  >> 非线性截断显著! 非线性比={mean_nonlin:.4f}")
        print("     SiLU/Sigmoid使大信号被截断")
    
    # 有效因果链: λ_max * cos(delta,v_max) * mlp_contribution
    effective_causal_chain = [l * c * m for l, c, m in zip(lambda_maxs, cos_delta_vmaxs, mlp_contributions)]
    print(f"\n  有效因果链强度: mean={np.mean(effective_causal_chain):.4f}")
    print(f"  λ_max范围: [{min(lambda_maxs):.2f}, {max(lambda_maxs):.2f}]")
    
    result_path = f"tests/glm5/results/phase_cxv/p537_{model_name}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({
            'results': [to_native(r) for r in results],
            'model': model_name,
            'mean_mlp_contribution': float(mean_mlp_contrib),
            'mean_cos_delta_vmax': float(mean_cos_delta),
            'mean_nonlinearity_ratio': float(mean_nonlin),
        }, f, indent=2)
    print(f"\n结果已保存: {result_path}")
    
    return results


# ===== 主函数 =====
EXPERIMENTS = {
    'p534': run_p534,
    'p535': run_p535,
    'p536': run_p536,
    'p537': run_p537,
}


def main():
    parser = argparse.ArgumentParser(description="Phase CXV-CXVI: 统一传播方程的修正")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["qwen3", "glm4", "deepseek7b"],
                        help="模型名称")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=list(EXPERIMENTS.keys()),
                        help="实验编号 (p534/p535/p536/p537)")
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Phase CXV-CXVI: {args.experiment.upper()}")
    print(f"模型: {args.model}")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    print(f"模型信息: {info.model_class}, {info.n_layers}层, d_model={info.d_model}")
    
    try:
        result = EXPERIMENTS[args.experiment](model, tokenizer, device, args.model)
    finally:
        release_model(model)
    
    print(f"\n实验 {args.experiment.upper()} 完成!")


if __name__ == "__main__":
    main()
