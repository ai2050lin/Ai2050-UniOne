"""
Phase CXIV-P528/P529/P530/P531/P532/P533: 从低秩结构到统一传播方程
==================================================================

Phase CXIII核心发现:
- 1-秩Jacobian来自W_down激活列低秩(PR_active≈0.99)
- v_max层间极不稳定(cos(prev)<0.15)
- 频谱力学Qwen3/GLM4验证成功(r>0.92), DS7B失败(r=0.61)
- DS7B因果链断裂(λ_max与权重无关, ρ<0.15)
- DS7B频谱极度集中(top10_energy=0.78)

Phase CXIV核心思路:
1. P528: v_max层间不稳定的根源 — 分析v_max方向随层变化的动力学
   - 假设: v_max方向由f_mlp的top激活神经元决定, 而这些神经元每层不同
   - 验证: v_max是否与W_down的top激活列方向对齐?

2. P529: Jacobian 1-秩的数学证明 — 从W_down子矩阵低秩推导1-秩Jacobian
   - 核心推导: J ≈ W_down * diag(f') * W_gate * J_LN
   - 如果f(LN(h_in))极度稀疏, 则J的低秩由f的稀疏度决定
   - 验证: rank(J) ≈ n_active(f_mlp)?

3. P530: 跨层传播模型 — 即使v_max每层变化, 信号如何有效传播?
   - 关键问题: 不存在固定传播方向时, 信号如何跨层传递?
   - 假设: v_max(l)在W_U奇异空间中有"重叠", 信号通过重叠区域传播
   - 测量: v_max(l)与v_max(l+1)在W_U奇异空间中的投影重叠度

4. P531: DS7B频谱集中的机制 — RL训练如何改变频谱结构
   - DS7B top10_energy=0.78, 远高于Qwen3的0.17
   - 假设: RL训练使W_down列空间更对齐W_U top奇异方向
   - 测量: W_down列方向在W_U奇异空间中的投影分布

5. P532: 统一传播方程 — λ_max(l) * S_Δ(l,k) → ratio(k) 的完整推导
   - 理论: ratio(k) = Σ_l λ_max(l) * |v_max(l)^T * u_k|² / Σ_l λ_max(l)
   - 验证: 逐层累积的预测ratio是否收敛到实测ratio?

6. P533: 训练动力学差异 — GLM4完美因果链 vs DS7B断裂
   - 对比GLM4/DS7B的W_down频谱、激活模式、LN增益的层间相关性
   - 假设: GLM4的W_down频谱结构更"有序"(层间相关高)

使用方法:
    python phase_cxiv_propagation_unification.py --model qwen3 --experiment p528
    python phase_cxiv_propagation_unification.py --model glm4 --experiment p529
    python phase_cxiv_propagation_unification.py --model deepseek7b --experiment p530
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


# ===== P528: v_max层间不稳定的根源 =====
def run_p528(model, tokenizer, device, model_name):
    """
    v_max层间不稳定的根源分析
    
    核心假设: v_max方向由f_mlp的top激活神经元决定
    如果这些神经元每层不同, v_max自然每层不同
    
    验证:
    1. v_max是否与W_down[:, top_k_active]的top奇异向量对齐?
    2. top激活神经元的层间重叠度
    3. v_max方向的层间变化是否能被f_mlp的层间变化解释?
    """
    print("\n" + "="*70)
    print("P528: v_max层间不稳定的根源")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    test_text = "The fundamental nature of reality can be understood through mathematical structures that underlie physical phenomena."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    sample_layers = get_sample_layers(info.n_layers, min(8, info.n_layers))
    print(f"采样层: {sample_layers}")
    
    # W_U SVD (用于后续分析)
    W_U = get_W_U(model)
    k_wu = min(100, min(W_U.shape) - 2)
    U_wu, S_wu, _ = svds(W_U.T.astype(np.float32), k=k_wu)
    U_wu = U_wu[:, ::-1]
    
    results = []
    prev_v_max = None
    prev_top_active = None
    
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
        
        # 计算f_mlp
        f_mlp, ln_h_in = compute_f_mlp(h_in, W_gate, W_up, post_ln_w)
        
        # f_mlp的top激活神经元
        f_mlp_abs = np.abs(f_mlp)
        top10_active = set(np.argsort(f_mlp_abs)[-10:])
        top20_active = set(np.argsort(f_mlp_abs)[-20:])
        
        # W_down在top10激活列上的子矩阵
        top10_indices = sorted(top10_active)
        W_down_active = W_down[:, top10_indices]
        
        # 子矩阵的top奇异向量
        if W_down_active.shape[1] > 1:
            U_act, S_act, _ = svds(W_down_active.astype(np.float32), k=min(3, W_down_active.shape[1]-1))
            S_act = S_act[::-1]
            U_act = U_act[:, ::-1]
            active_top1 = U_act[:, 0] / (np.linalg.norm(U_act[:, 0]) + 1e-10)
        else:
            active_top1 = W_down_active[:, 0] / (np.linalg.norm(W_down_active[:, 0]) + 1e-10)
            S_act = np.array([np.linalg.norm(W_down_active[:, 0])])
        
        # Jacobian probe获取v_max
        J_proj, _ = compute_jacobian_probes(model, layers, l_idx, inputs, info, n_probes=80)
        J_proj_T = J_proj.T
        U_jt, S_j, Vt_jt = torch.linalg.svd(J_proj_T, full_matrices=False)
        v_max = U_jt[:, 0].cpu().float().numpy()
        v_max = v_max / (np.linalg.norm(v_max) + 1e-10)
        
        # v_max与W_down激活子矩阵top1的对齐度
        cos_vmax_active_top1 = abs(np.dot(v_max, active_top1))
        
        # 层间v_max稳定性
        cos_prev_vmax = 0.0
        if prev_v_max is not None:
            cos_prev_vmax = abs(np.dot(v_max, prev_v_max))
        
        # 层间top激活神经元重叠度
        top10_overlap = 0.0
        top20_overlap = 0.0
        if prev_top_active is not None:
            top10_overlap = len(top10_active & prev_top_active[0]) / 10.0
            top20_overlap = len(top20_active & prev_top_active[1]) / 20.0
        
        # v_max在W_U奇异空间中的分布
        v_max_wu_proj = U_wu.T @ v_max  # [k_wu]
        v_max_wu_energy = v_max_wu_proj**2
        v_max_wu_top10 = float(np.sum(np.sort(v_max_wu_energy)[-10:]))
        v_max_wu_top50 = float(np.sum(np.sort(v_max_wu_energy)[-50:]))
        v_max_wu_total = float(np.sum(v_max_wu_energy))
        
        # W_down列方向在W_U奇异空间中的投影
        # 采样50列计算
        n_sample_cols = min(50, W_down.shape[1])
        col_indices = np.random.choice(W_down.shape[1], n_sample_cols, replace=False)
        col_wu_energy = []
        for ci in col_indices:
            col = W_down[:, ci]
            col_norm = np.linalg.norm(col)
            if col_norm > 1e-10:
                col_proj = U_wu.T @ (col / col_norm)
                col_energy = col_proj**2
                col_top10 = float(np.sum(np.sort(col_energy)[-10:]))
                col_wu_energy.append(col_top10)
        mean_col_wu_top10 = float(np.mean(col_wu_energy)) if col_wu_energy else 0
        
        result = {
            'layer': l_idx,
            'cos_vmax_active_top1': float(cos_vmax_active_top1),
            'cos_prev_vmax': float(cos_prev_vmax),
            'top10_overlap': float(top10_overlap),
            'top20_overlap': float(top20_overlap),
            'v_max_wu_top10_ratio': float(v_max_wu_top10 / (v_max_wu_total + 1e-10)),
            'v_max_wu_top50_ratio': float(v_max_wu_top50 / (v_max_wu_total + 1e-10)),
            'mean_col_wu_top10': mean_col_wu_top10,
            'lambda_max': float(S_j[0]),
            'f_mlp_sparsity': float(1.0 - np.sum(f_mlp_abs > 0.01 * np.max(f_mlp_abs)) / len(f_mlp)),
        }
        results.append(result)
        
        print(f"  cos(v_max, active_top1)={cos_vmax_active_top1:.4f}")
        print(f"  cos(prev_v_max)={cos_prev_vmax:.4f}")
        print(f"  top10_overlap={top10_overlap:.4f}, top20_overlap={top20_overlap:.4f}")
        print(f"  v_max W_U top10_ratio={result['v_max_wu_top10_ratio']:.4f}")
        print(f"  mean_col W_U top10={mean_col_wu_top10:.4f}")
        
        prev_v_max = v_max.copy()
        prev_top_active = (top10_active.copy(), top20_active.copy())
        torch.cuda.empty_cache()
    
    # 汇总
    print("\n" + "="*70)
    print("P528 汇总: v_max不稳定根源")
    print("="*70)
    
    mean_cos_active = np.mean([r['cos_vmax_active_top1'] for r in results])
    mean_cos_prev = np.mean([r['cos_prev_vmax'] for r in results if r['cos_prev_vmax'] > 0])
    mean_top10_overlap = np.mean([r['top10_overlap'] for r in results if r['top10_overlap'] > 0])
    mean_vmax_wu = np.mean([r['v_max_wu_top10_ratio'] for r in results])
    mean_col_wu = np.mean([r['mean_col_wu_top10'] for r in results])
    
    print(f"  mean cos(v_max, active_top1): {mean_cos_active:.4f}")
    print(f"  mean cos(prev_v_max): {mean_cos_prev:.4f}")
    print(f"  mean top10_overlap: {mean_top10_overlap:.4f}")
    print(f"  mean v_max W_U top10_ratio: {mean_vmax_wu:.4f}")
    print(f"  mean col W_U top10: {mean_col_wu:.4f}")
    
    # 关键判断
    print("\n  根源分析:")
    if mean_cos_active > 0.5:
        print("  >> v_max与W_down激活子矩阵top1高度对齐 → 不稳定来自f_mlp激活模式变化")
    elif mean_top10_overlap < 0.2:
        print("  >> top激活神经元层间几乎无重叠 → 不稳定来自激活模式完全改变")
    elif mean_vmax_wu > 0.3:
        print("  >> v_max在W_U空间中集中 → 不稳定可能来自W_U top方向的结构")
    
    result_path = f"tests/glm5/results/phase_cxiv/p528_{model_name}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({'results': [to_native(r) for r in results], 'model': model_name}, f, indent=2)
    print(f"\n结果已保存: {result_path}")
    
    return results


# ===== P529: Jacobian 1-秩的数学证明 =====
def run_p529(model, tokenizer, device, model_name):
    """
    从W_down子矩阵低秩推导1-秩Jacobian的数学证明
    
    核心推导:
    J_l = dh_out/dh_in = I + W_down * diag(f') * [W_up + σ'·W_gate] * J_LN
    
    其中 f' = d(σ(z)⊙(W_up·x))/dx
    
    由于σ(z)的稀疏性, f'≈0对大部分中间神经元
    只有top_k个激活神经元有非零贡献
    
    因此: rank(J_l - I) ≈ rank(W_down[:, active] * diag(f'_active) * W_combined[active, :])
    
    如果W_down[:, active]的秩=1, 则rank(J_l-I)=1
    
    验证:
    1. 实测Jacobian的秩 vs 预测(从f_mlp稀疏度+PR_active)
    2. 不同稀疏度阈值下的预测精度
    """
    print("\n" + "="*70)
    print("P529: Jacobian 1-秩的数学证明")
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
        W_down = weights.W_down
        W_gate = weights.W_gate
        W_up = weights.W_up
        post_ln_w = weights.post_attn_layernorm_weight
        
        # 获取隐藏状态
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
            h_states = outputs.hidden_states
            h_in = h_states[l_idx][0, -1].cpu().float().numpy()
        
        # 计算f_mlp和σ'(z)
        f_mlp, ln_h_in = compute_f_mlp(h_in, W_gate, W_up, post_ln_w)
        f_mlp_abs = np.abs(f_mlp)
        
        # σ'(z) for sigmoid: σ'(z) = σ(z)(1-σ(z))
        if W_gate is not None:
            gate_pre = W_gate @ ln_h_in
            gate_act = 1.0 / (1.0 + np.exp(-gate_pre))
            gate_deriv = gate_act * (1 - gate_act)  # σ'(z)
        else:
            gate_deriv = np.ones(W_up.shape[0])
        
        # f' = d(σ(z)⊙(W_up·x))/dx
        # f' = σ'(z) * (W_up·x) + σ(z) * W_up  (链式法则)
        # 简化: 只考虑对z的导数 (MLP中间的导数)
        # f'_i = σ'(z_i) * (W_up·x)_i + σ(z_i) * (W_up)_i,j
        # 对每个激活神经元i: f'_i ≈ σ'(z_i) * (W_up·x)_i (主项)
        f_deriv = gate_deriv * (W_up @ ln_h_in)  # 近似导数
        
        # 有效激活神经元 (|f_deriv|大的)
        f_deriv_abs = np.abs(f_deriv)
        
        # 不同阈值下的有效激活数
        thresholds = [0.01, 0.05, 0.1, 0.2]
        n_active_at_thresh = {}
        for thresh in thresholds:
            n_active = int(np.sum(f_deriv_abs > thresh * np.max(f_deriv_abs)))
            n_active_at_thresh[f'n_active_{thresh}'] = n_active
        
        # 核心验证: W_down * diag(f'_active) 的秩
        # 选取top-k激活 (k=5,10,20)
        for k_active in [5, 10, 20]:
            top_k_idx = np.argsort(f_deriv_abs)[-k_active:]
            W_down_k = W_down[:, top_k_idx]  # [d_model, k_active]
            f_deriv_k = f_deriv[top_k_idx]    # [k_active]
            
            # W_down_k * diag(f_deriv_k) = [d_model, k_active]
            W_active = W_down_k * f_deriv_k[np.newaxis, :]
            
            # 这个矩阵的秩
            if min(W_active.shape) > 1:
                U_wa, S_wa, _ = svds(W_active.astype(np.float32), k=min(5, min(W_active.shape)-1))
                S_wa = S_wa[::-1]
                PR_W_active = float(S_wa.sum()**2 / (len(S_wa) * (S_wa**2).sum() + 1e-10))
                top1_energy_W_active = float(S_wa[0]**2 / (np.sum(S_wa**2) + 1e-10))
            else:
                PR_W_active = 1.0
                top1_energy_W_active = 1.0
            
            n_active_at_thresh[f'PR_W_active_k{k_active}'] = PR_W_active
            n_active_at_thresh[f'top1_energy_k{k_active}'] = top1_energy_W_active
        
        # Jacobian probe验证
        J_proj, _ = compute_jacobian_probes(model, layers, l_idx, inputs, info, n_probes=100)
        J_proj_T = J_proj.T
        U_jt, S_j, Vt_jt = torch.linalg.svd(J_proj_T, full_matrices=False)
        
        # 实测Jacobian的有效秩
        S_j_np = S_j.cpu().numpy()
        total_energy = np.sum(S_j_np**2)
        cum_energy = np.cumsum(S_j_np**2)
        n_90 = int(np.searchsorted(cum_energy / total_energy, 0.9)) + 1
        n_99 = int(np.searchsorted(cum_energy / total_energy, 0.99)) + 1
        top1_energy_J = float(S_j_np[0]**2 / total_energy)
        
        # 预测: 如果W_down_k*diag(f'_k)是1-秩, 则Jacobian也应该是1-秩
        # 但还要乘以W_combined * J_LN, 这会增加秩
        # 预测n_90: 由W_down_k的秩决定
        PR_pred_k10 = n_active_at_thresh.get('PR_W_active_k10', 1.0)
        
        result = {
            'layer': l_idx,
            'lambda_max': float(S_j[0]),
            'lambda_2': float(S_j[1]) if len(S_j) > 1 else 0,
            'n_90_measured': n_90,
            'n_99_measured': n_99,
            'top1_energy_J': top1_energy_J,
            'PR_W_active_k5': n_active_at_thresh.get('PR_W_active_k5', 0),
            'PR_W_active_k10': n_active_at_thresh.get('PR_W_active_k10', 0),
            'PR_W_active_k20': n_active_at_thresh.get('PR_W_active_k20', 0),
            'top1_energy_k5': n_active_at_thresh.get('top1_energy_k5', 0),
            'top1_energy_k10': n_active_at_thresh.get('top1_energy_k10', 0),
            'top1_energy_k20': n_active_at_thresh.get('top1_energy_k20', 0),
            'n_active_0.01': n_active_at_thresh.get('n_active_0.01', 0),
            'n_active_0.05': n_active_at_thresh.get('n_active_0.05', 0),
            'n_active_0.1': n_active_at_thresh.get('n_active_0.1', 0),
        }
        results.append(result)
        
        print(f"  n_90={n_90}, n_99={n_99}, top1_energy_J={top1_energy_J:.6f}")
        print(f"  PR(W_active_k5)={n_active_at_thresh.get('PR_W_active_k5', 0):.4f}")
        print(f"  PR(W_active_k10)={n_active_at_thresh.get('PR_W_active_k10', 0):.4f}")
        print(f"  PR(W_active_k20)={n_active_at_thresh.get('PR_W_active_k20', 0):.4f}")
        print(f"  n_active(0.01)={n_active_at_thresh.get('n_active_0.01', 0)}")
        print(f"  n_active(0.05)={n_active_at_thresh.get('n_active_0.05', 0)}")
        
        torch.cuda.empty_cache()
    
    # 汇总
    print("\n" + "="*70)
    print("P529 汇总: Jacobian 1-秩数学证明")
    print("="*70)
    
    mean_n90 = np.mean([r['n_90_measured'] for r in results])
    mean_n99 = np.mean([r['n_99_measured'] for r in results])
    mean_top1 = np.mean([r['top1_energy_J'] for r in results])
    mean_PR_k10 = np.mean([r['PR_W_active_k10'] for r in results])
    mean_PR_k5 = np.mean([r['PR_W_active_k5'] for r in results])
    
    print(f"  mean n_90_measured: {mean_n90:.2f}")
    print(f"  mean n_99_measured: {mean_n99:.2f}")
    print(f"  mean top1_energy_J: {mean_top1:.6f}")
    print(f"  mean PR(W_active_k5): {mean_PR_k5:.4f}")
    print(f"  mean PR(W_active_k10): {mean_PR_k10:.4f}")
    
    # 证明判定
    print("\n  数学证明验证:")
    if mean_n90 == 1 and mean_top1 > 0.999:
        print("  >> Jacobian确实是1-秩!")
        if mean_PR_k5 > 0.9:
            print("  >> W_down激活子矩阵极度低秩 → 证明了1-秩来自W_down结构")
        else:
            print("  >> W_down激活子矩阵不太低秩 → 1-秩来源需要进一步分析")
    elif mean_n90 <= 2:
        print("  >> Jacobian近似1-秩(n_90≤2)")
    
    result_path = f"tests/glm5/results/phase_cxiv/p529_{model_name}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({'results': [to_native(r) for r in results], 'model': model_name}, f, indent=2)
    print(f"\n结果已保存: {result_path}")
    
    return results


# ===== P530: 跨层传播模型 =====
def run_p530(model, tokenizer, device, model_name):
    """
    即使v_max每层变化, 信号如何有效传播?
    
    核心思想: "频谱接力"模型
    v_max(l)在W_U奇异空间中的投影与v_max(l+1)的投影有重叠
    信号通过重叠区域跨层传递
    
    测量:
    1. v_max(l)和v_max(l+1)在W_U奇异空间中的投影重叠度
    2. 逐层累积的"有效传播能量"
    3. 构建传播矩阵T(l,l') = |v_max(l)^T * W_U^T * W_U * v_max(l')| = |(U_wu^T*v_max(l))^T*(U_wu^T*v_max(l'))|
    """
    print("\n" + "="*70)
    print("P530: 跨层传播模型")
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
    
    # 先收集所有层的v_max和lambda_max
    layer_data = []
    for l_idx in sample_layers:
        J_proj, _ = compute_jacobian_probes(model, layers, l_idx, inputs, info, n_probes=80)
        J_proj_T = J_proj.T
        U_jt, S_j, _ = torch.linalg.svd(J_proj_T, full_matrices=False)
        v_max = U_jt[:, 0].cpu().float().numpy()
        v_max = v_max / (np.linalg.norm(v_max) + 1e-10)
        lambda_max = float(S_j[0])
        
        # v_max在W_U奇异空间中的投影
        v_max_wu = U_wu.T @ v_max  # [k_wu]
        
        layer_data.append({
            'layer': l_idx,
            'v_max': v_max,
            'v_max_wu': v_max_wu,
            'lambda_max': lambda_max,
        })
        
        print(f"  L{l_idx}: λ_max={lambda_max:.2f}, ||v_max_wu||={np.linalg.norm(v_max_wu):.4f}")
        torch.cuda.empty_cache()
    
    # 计算层间传播矩阵
    n_layers_sampled = len(layer_data)
    T_matrix = np.zeros((n_layers_sampled, n_layers_sampled))
    for i in range(n_layers_sampled):
        for j in range(n_layers_sampled):
            # T(i,j) = |v_max(i)^T * v_max(j)| 在W_U空间中的重叠
            # = |v_max_wu(i)^T * v_max_wu(j)|
            T_matrix[i, j] = abs(np.dot(layer_data[i]['v_max_wu'], layer_data[j]['v_max_wu']))
    
    # 相邻层重叠度
    adjacent_overlap = []
    for i in range(n_layers_sampled - 1):
        overlap = T_matrix[i, i+1]
        adjacent_overlap.append(overlap)
    
    # v_max本身的层间cos (直接空间)
    direct_cos = []
    for i in range(n_layers_sampled - 1):
        cos = abs(np.dot(layer_data[i]['v_max'], layer_data[i+1]['v_max']))
        direct_cos.append(cos)
    
    # 传播模型: 从L0到L_last的累积传播
    # 信号通过每层时, 在W_U奇异空间中保留的比例
    # 传播效率 = ∏_l (1 - loss_l), 其中loss_l = 1 - |cos(v_max_wu(l), v_max_wu(l+1))|
    
    # 更精细的模型: 逐层追踪信号在W_U top-k方向上的能量
    k_track = 10
    cumulative_signal = np.zeros(k_wu)  # 追踪信号在W_U奇异向量上的能量
    cumulative_signal[:k_track] = 1.0  # 初始: 假设信号在top10方向
    
    propagation_results = []
    for i in range(n_layers_sampled):
        # 当前层的放大和旋转
        lambda_l = layer_data[i]['lambda_max']
        v_wu = layer_data[i]['v_max_wu']
        
        # v_wu定义了信号如何被旋转到W_U奇异空间中的不同方向
        # 信号经过这层后, 方向对齐v_max, 能量乘以lambda_max
        
        # 简化模型: 经过每层后, 信号的W_U频谱变为v_wu的频谱 (因为1-秩)
        if i == 0:
            current_spectrum = v_wu**2 * lambda_l
        else:
            # 混合: 保留前一层的部分信号 + 新层的贡献
            prev_spectrum = propagation_results[-1]['spectrum']
            mix_ratio = 0.5  # 残差连接的混合比
            current_spectrum = mix_ratio * prev_spectrum + (1 - mix_ratio) * v_wu**2 * lambda_l
        
        total_energy = np.sum(current_spectrum)
        top10_ratio = float(np.sum(np.sort(current_spectrum)[-10:]) / (total_energy + 1e-10))
        top50_ratio = float(np.sum(np.sort(current_spectrum)[-50:]) / (total_energy + 1e-10))
        
        propagation_results.append({
            'layer': layer_data[i]['layer'],
            'spectrum': current_spectrum,
            'total_energy': float(total_energy),
            'top10_ratio': top10_ratio,
            'top50_ratio': top50_ratio,
        })
    
    # 汇总结果
    results = []
    for i, ld in enumerate(layer_data):
        # 层间重叠 (W_U空间)
        wu_overlap_next = adjacent_overlap[i] if i < len(adjacent_overlap) else 0
        wu_overlap_prev = adjacent_overlap[i-1] if i > 0 else 0
        
        # 直接cos
        direct_cos_next = direct_cos[i] if i < len(direct_cos) else 0
        direct_cos_prev = direct_cos[i-1] if i > 0 else 0
        
        # 传播频谱
        pr = propagation_results[i]
        
        result = {
            'layer': ld['layer'],
            'lambda_max': ld['lambda_max'],
            'wu_overlap_next': float(wu_overlap_next),
            'wu_overlap_prev': float(wu_overlap_prev),
            'direct_cos_next': float(direct_cos_next),
            'direct_cos_prev': float(direct_cos_prev),
            'propagation_top10_ratio': pr['top10_ratio'],
            'propagation_top50_ratio': pr['top50_ratio'],
            'propagation_total_energy': pr['total_energy'],
        }
        results.append(result)
    
    print("\n" + "="*70)
    print("P530 汇总: 跨层传播模型")
    print("="*70)
    
    mean_wu_overlap = np.mean(adjacent_overlap) if adjacent_overlap else 0
    mean_direct_cos = np.mean(direct_cos) if direct_cos else 0
    
    print(f"  mean W_U空间层间重叠: {mean_wu_overlap:.4f}")
    print(f"  mean 直接cos(v_max, next): {mean_direct_cos:.4f}")
    
    # 最终传播频谱
    final_pr = propagation_results[-1]
    print(f"  最终传播 top10_ratio: {final_pr['top10_ratio']:.4f}")
    print(f"  最终传播 top50_ratio: {final_pr['top50_ratio']:.4f}")
    
    # 关键判断
    print("\n  传播机制:")
    if mean_wu_overlap > 0.5:
        print("  >> W_U空间中层间重叠高 → 信号通过'频谱接力'有效传播")
    elif mean_direct_cos < 0.2 and mean_wu_overlap > mean_direct_cos:
        print("  >> 直接空间v_max不稳定, 但W_U空间有重叠 → 频谱接力是关键机制")
    else:
        print("  >> 传播机制复杂, 需要更精细的模型")
    
    result_path = f"tests/glm5/results/phase_cxiv/p530_{model_name}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        # 不保存spectrum (太大)
        save_results = [{k: v for k, v in r.items()} for r in results]
        json.dump({'results': save_results, 'model': model_name,
                   'mean_wu_overlap': float(mean_wu_overlap),
                   'mean_direct_cos': float(mean_direct_cos)}, f, indent=2)
    print(f"\n结果已保存: {result_path}")
    
    return results


# ===== P531: DS7B频谱集中的机制 =====
def run_p531(model, tokenizer, device, model_name):
    """
    DS7B top10_energy=0.78, 远高于Qwen3的0.17
    为什么RL训练使信号集中在W_U top方向?
    
    假设: RL训练使W_down列方向更对齐W_U top奇异方向
    
    测量:
    1. W_down每列在W_U奇异空间中的投影分布
    2. W_down列方向与W_U top-k的集中度
    3. 对比Qwen3/GLM4/DS7B的W_down→W_U对齐度
    """
    print("\n" + "="*70)
    print("P531: DS7B频谱集中的机制")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    sample_layers = get_sample_layers(info.n_layers, min(6, info.n_layers))
    print(f"采样层: {sample_layers}")
    
    # W_U SVD
    W_U = get_W_U(model)
    k_wu = min(200, min(W_U.shape) - 2)
    print(f"计算W_U SVD (k={k_wu})...")
    U_wu, S_wu, _ = svds(W_U.T.astype(np.float32), k=k_wu)
    U_wu = U_wu[:, ::-1]
    S_wu = S_wu[::-1]
    
    results = []
    
    for l_idx in sample_layers:
        print(f"\n--- 层 {l_idx} ---")
        
        weights = get_layer_weights(layers[l_idx], info.d_model, info.mlp_type)
        W_down = weights.W_down  # [d_model, intermediate]
        W_gate = weights.W_gate
        W_up = weights.W_up
        post_ln_w = weights.post_attn_layernorm_weight
        
        # W_down每列在W_U奇异空间中的投影
        n_cols = W_down.shape[1]
        n_sample = min(200, n_cols)
        col_indices = np.random.choice(n_cols, n_sample, replace=False)
        
        col_top10_energies = []
        col_top50_energies = []
        col_wu_spectra = []
        
        for ci in col_indices:
            col = W_down[:, ci]
            col_norm = np.linalg.norm(col)
            if col_norm < 1e-10:
                continue
            col_normed = col / col_norm
            
            # 在W_U奇异空间中的投影
            proj = U_wu.T @ col_normed  # [k_wu]
            proj_energy = proj**2
            total = np.sum(proj_energy)
            
            top10_e = np.sum(np.sort(proj_energy)[-10:]) / (total + 1e-10)
            top50_e = np.sum(np.sort(proj_energy)[-50:]) / (total + 1e-10)
            
            col_top10_energies.append(top10_e)
            col_top50_energies.append(top50_e)
            col_wu_spectra.append(proj_energy)
        
        mean_col_top10 = float(np.mean(col_top10_energies))
        mean_col_top50 = float(np.mean(col_top50_energies))
        std_col_top10 = float(np.std(col_top10_energies))
        
        # W_down整体在W_U奇异空间中的投影 (用SVD)
        # W_down^T * U_wu → [intermediate, k_wu]
        W_down_T_Uwu = (W_down.T @ U_wu)  # [intermediate, k_wu]
        # 每个W_U奇异方向上的W_down总能量
        wu_energy_per_dir = np.sum(W_down_T_Uwu**2, axis=0)  # [k_wu]
        total_wu_energy = np.sum(wu_energy_per_dir)
        wu_top10_energy = float(np.sum(np.sort(wu_energy_per_dir)[-10:]) / (total_wu_energy + 1e-10))
        wu_top50_energy = float(np.sum(np.sort(wu_energy_per_dir)[-50:]) / (total_wu_energy + 1e-10))
        
        # W_down整体频谱的集中度
        k_wd = min(50, min(W_down.shape) - 2)
        U_wd, S_wd, _ = svds(W_down.astype(np.float32), k=k_wd)
        S_wd = S_wd[::-1]
        PR_Wdown = float(S_wd.sum()**2 / (len(S_wd) * (S_wd**2).sum() + 1e-10))
        
        # W_gate激活模式 vs W_U对齐
        # 激活的列方向是否更对齐W_U top?
        test_text = "The fundamental nature of reality can be understood through mathematical structures that underlie physical phenomena."
        inputs_tok = tokenizer(test_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(inputs_tok["input_ids"], output_hidden_states=True)
            h_in = outputs.hidden_states[l_idx][0, -1].cpu().float().numpy()
        
        f_mlp, ln_h_in = compute_f_mlp(h_in, W_gate, W_up, post_ln_w)
        f_mlp_abs = np.abs(f_mlp)
        top10_active_idx = np.argsort(f_mlp_abs)[-10:]
        
        # top10激活列的W_U对齐度
        active_top10_energies = []
        for ai in top10_active_idx:
            col = W_down[:, ai]
            col_norm = np.linalg.norm(col)
            if col_norm < 1e-10:
                continue
            proj = U_wu.T @ (col / col_norm)
            proj_e = proj**2
            top10_e = np.sum(np.sort(proj_e)[-10:]) / (np.sum(proj_e) + 1e-10)
            active_top10_energies.append(top10_e)
        
        mean_active_top10 = float(np.mean(active_top10_energies)) if active_top10_energies else 0
        
        result = {
            'layer': l_idx,
            'mean_col_top10_wu': mean_col_top10,
            'mean_col_top50_wu': mean_col_top50,
            'std_col_top10_wu': std_col_top10,
            'wu_top10_energy_Wdown': wu_top10_energy,
            'wu_top50_energy_Wdown': wu_top50_energy,
            'mean_active_col_top10_wu': mean_active_top10,
            'PR_Wdown': PR_Wdown,
        }
        results.append(result)
        
        print(f"  mean col W_U top10: {mean_col_top10:.4f} ± {std_col_top10:.4f}")
        print(f"  W_down整体 W_U top10: {wu_top10_energy:.4f}")
        print(f"  激活列 W_U top10: {mean_active_top10:.4f}")
        print(f"  PR_Wdown: {PR_Wdown:.4f}")
        
        torch.cuda.empty_cache()
    
    # 汇总
    print("\n" + "="*70)
    print("P531 汇总: 频谱集中机制")
    print("="*70)
    
    mean_col_top10 = np.mean([r['mean_col_top10_wu'] for r in results])
    mean_wu_top10 = np.mean([r['wu_top10_energy_Wdown'] for r in results])
    mean_active_top10 = np.mean([r['mean_active_col_top10_wu'] for r in results])
    
    print(f"  mean col W_U top10: {mean_col_top10:.4f}")
    print(f"  mean W_down整体 W_U top10: {mean_wu_top10:.4f}")
    print(f"  mean 激活列 W_U top10: {mean_active_top10:.4f}")
    
    # 判断
    print("\n  频谱集中机制:")
    if mean_col_top10 > 0.3:
        print("  >> W_down列方向本身集中对齐W_U top → 结构性对齐")
    if mean_active_top10 > mean_col_top10 + 0.1:
        print("  >> 激活列比随机列更对齐W_U top → 功能性选择")
    
    result_path = f"tests/glm5/results/phase_cxiv/p531_{model_name}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({'results': [to_native(r) for r in results], 'model': model_name}, f, indent=2)
    print(f"\n结果已保存: {result_path}")
    
    return results


# ===== P532: 统一传播方程 =====
def run_p532(model, tokenizer, device, model_name):
    """
    统一传播方程: λ_max(l) * S_Δ(l,k) → ratio(k)
    
    理论:
    ratio(k) = Σ_l w_l * |v_max(l)^T * u_k|² / Σ_l w_l
    其中 w_l = λ_max(l) (权重由放大倍数决定)
    
    验证:
    1. 逐层累积的预测ratio是否收敛到实测ratio?
    2. 不同k值下的预测精度
    """
    print("\n" + "="*70)
    print("P532: 统一传播方程")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    test_text = "The fundamental nature of reality can be understood through mathematical structures that underleneath physical phenomena."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    sample_layers = get_sample_layers(info.n_layers, min(8, info.n_layers))
    print(f"采样层: {sample_layers}")
    
    # W_U SVD
    W_U = get_W_U(model)
    k_wu = min(200, min(W_U.shape) - 2)
    print(f"计算W_U SVD (k={k_wu})...")
    U_wu, S_wu, _ = svds(W_U.T.astype(np.float32), k=k_wu)
    U_wu = U_wu[:, ::-1]
    
    # 收集所有层的v_max, lambda_max和delta_h
    layer_data = []
    for l_idx in sample_layers:
        # 获取隐藏状态
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
            h_states = outputs.hidden_states
            h_in = h_states[l_idx][0, -1].cpu().float().numpy()
            h_out = h_states[l_idx + 1][0, -1].cpu().float().numpy()
        
        delta_h = h_out - h_in
        delta_h_norm = np.linalg.norm(delta_h)
        
        # Jacobian probe
        J_proj, _ = compute_jacobian_probes(model, layers, l_idx, inputs, info, n_probes=80)
        J_proj_T = J_proj.T
        U_jt, S_j, _ = torch.linalg.svd(J_proj_T, full_matrices=False)
        v_max = U_jt[:, 0].cpu().float().numpy()
        v_max = v_max / (np.linalg.norm(v_max) + 1e-10)
        lambda_max = float(S_j[0])
        
        # v_max在W_U奇异空间中的投影
        v_max_wu = U_wu.T @ v_max  # [k_wu]
        v_max_wu_energy = v_max_wu**2
        
        # delta_h在W_U奇异空间中的投影
        if delta_h_norm > 1e-10:
            delta_wu = U_wu.T @ delta_h
            delta_wu_energy = delta_wu**2
            delta_total = np.sum(delta_wu_energy)
            
            # 实测ratio(k)
            measured_ratio_k10 = float(np.sum(delta_wu_energy[:10]) / (delta_total + 1e-10))
            measured_ratio_k50 = float(np.sum(delta_wu_energy[:50]) / (delta_total + 1e-10))
        else:
            measured_ratio_k10 = 0
            measured_ratio_k50 = 0
            delta_total = 0
        
        layer_data.append({
            'layer': l_idx,
            'v_max_wu_energy': v_max_wu_energy,
            'lambda_max': lambda_max,
            'delta_h_norm': delta_h_norm,
            'measured_ratio_k10': measured_ratio_k10,
            'measured_ratio_k50': measured_ratio_k50,
        })
        
        print(f"  L{l_idx}: λ_max={lambda_max:.2f}, ratio_k10={measured_ratio_k10:.4f}, ratio_k50={measured_ratio_k50:.4f}")
        torch.cuda.empty_cache()
    
    # 统一传播方程验证
    # ratio(k) = Σ_l λ_max(l) * v_max_wu_energy(l, k) / Σ_l λ_max(l) * Σ_k v_max_wu_energy(l, k)
    
    # 方法1: 加权平均
    lambda_maxs = np.array([ld['lambda_max'] for ld in layer_data])
    weights = lambda_maxs / (np.sum(lambda_maxs) + 1e-10)
    
    # 预测的ratio(k): 加权平均v_max_wu_energy
    k_wu_actual = len(layer_data[0]['v_max_wu_energy'])
    predicted_spectrum = np.zeros(k_wu_actual)
    for i, ld in enumerate(layer_data):
        predicted_spectrum += weights[i] * ld['v_max_wu_energy']
    
    # 归一化
    predicted_spectrum = predicted_spectrum / (np.sum(predicted_spectrum) + 1e-10)
    
    # 预测的累积ratio
    cum_predicted = np.cumsum(predicted_spectrum)
    
    # 实测的ratio: 取所有层的平均
    mean_measured_k10 = np.mean([ld['measured_ratio_k10'] for ld in layer_data])
    mean_measured_k50 = np.mean([ld['measured_ratio_k50'] for ld in layer_data])
    
    # 预测的ratio at k=10, 50
    predicted_k10 = float(cum_predicted[9]) if len(cum_predicted) >= 10 else 0
    predicted_k50 = float(cum_predicted[49]) if len(cum_predicted) >= 50 else 0
    
    # 逐层累积验证
    cumulative_results = []
    cum_spectrum = np.zeros(k_wu_actual)
    cum_weight = 0
    
    for i, ld in enumerate(layer_data):
        cum_spectrum += ld['lambda_max'] * ld['v_max_wu_energy']
        cum_weight += ld['lambda_max']
        
        cum_pred = cum_spectrum / (np.sum(cum_spectrum) + 1e-10)
        cum_ratio_k10 = float(np.sum(cum_pred[:10]))
        cum_ratio_k50 = float(np.sum(cum_pred[:50]))
        
        cumulative_results.append({
            'n_layers': i + 1,
            'predicted_k10': cum_ratio_k10,
            'predicted_k50': cum_ratio_k50,
        })
    
    results = []
    for i, ld in enumerate(layer_data):
        result = {
            'layer': ld['layer'],
            'lambda_max': ld['lambda_max'],
            'measured_ratio_k10': ld['measured_ratio_k10'],
            'measured_ratio_k50': ld['measured_ratio_k50'],
        }
        if i < len(cumulative_results):
            result['cum_predicted_k10'] = cumulative_results[i]['predicted_k10']
            result['cum_predicted_k50'] = cumulative_results[i]['predicted_k50']
        results.append(result)
    
    print("\n" + "="*70)
    print("P532 汇总: 统一传播方程")
    print("="*70)
    
    print(f"  mean measured ratio(k=10): {mean_measured_k10:.4f}")
    print(f"  mean measured ratio(k=50): {mean_measured_k50:.4f}")
    print(f"  predicted ratio(k=10): {predicted_k10:.4f}")
    print(f"  predicted ratio(k=50): {predicted_k50:.4f}")
    
    # 误差
    err_k10 = abs(predicted_k10 - mean_measured_k10)
    err_k50 = abs(predicted_k50 - mean_measured_k50)
    print(f"  |error| k=10: {err_k10:.4f}")
    print(f"  |error| k=50: {err_k50:.4f}")
    
    # 逐层累积收敛
    if len(cumulative_results) > 2:
        print(f"\n  逐层累积预测:")
        for cr in cumulative_results:
            print(f"    n={cr['n_layers']}: pred_k10={cr['predicted_k10']:.4f}, pred_k50={cr['predicted_k50']:.4f}")
    
    result_path = f"tests/glm5/results/phase_cxiv/p532_{model_name}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({
            'results': [to_native(r) for r in results],
            'model': model_name,
            'mean_measured_k10': float(mean_measured_k10),
            'mean_measured_k50': float(mean_measured_k50),
            'predicted_k10': float(predicted_k10),
            'predicted_k50': float(predicted_k50),
        }, f, indent=2)
    print(f"\n结果已保存: {result_path}")
    
    return results


# ===== P533: 训练动力学差异 =====
def run_p533(model, tokenizer, device, model_name):
    """
    对比GLM4/DS7B的W_down频谱、激活模式、LN增益的层间相关性
    为什么GLM4有完美因果链而DS7B没有?
    
    测量:
    1. W_down频谱的层间相似度 (SVD谱形状的层间相关)
    2. LN增益的层间变化模式
    3. λ_max的层间相关结构
    4. post_ln_norm与W_down_norm的层间共变性
    """
    print("\n" + "="*70)
    print("P533: 训练动力学差异")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    sample_layers = get_sample_layers(info.n_layers, min(10, info.n_layers))
    print(f"采样层: {sample_layers}")
    
    results = []
    prev_W_down_spectrum = None
    
    for l_idx in sample_layers:
        print(f"\n--- 层 {l_idx} ---")
        
        weights = get_layer_weights(layers[l_idx], info.d_model, info.mlp_type)
        W_down = weights.W_down
        W_gate = weights.W_gate
        W_up = weights.W_up
        post_ln_w = weights.post_attn_layernorm_weight
        input_ln_w = weights.input_layernorm_weight
        
        # W_down频谱
        k_wd = min(50, min(W_down.shape) - 2)
        U_wd, S_wd, _ = svds(W_down.astype(np.float32), k=k_wd)
        S_wd = S_wd[::-1]
        S_wd_normed = S_wd / (S_wd[0] + 1e-10)  # 归一化到[0,1]
        
        # W_down频谱的层间相似度
        spectrum_corr = 0.0
        if prev_W_down_spectrum is not None:
            min_len = min(len(S_wd_normed), len(prev_W_down_spectrum))
            try:
                spectrum_corr, _ = pearsonr(S_wd_normed[:min_len], prev_W_down_spectrum[:min_len])
            except:
                spectrum_corr = 0
        
        # W_down norm
        W_down_norm = float(np.linalg.norm(W_down))
        W_down_frobenius = float(np.linalg.norm(W_down, 'fro'))
        
        # LN增益
        post_ln_norm = float(np.linalg.norm(post_ln_w)) if post_ln_w is not None else 0
        input_ln_norm = float(np.linalg.norm(input_ln_w)) if input_ln_w is not None else 0
        
        # W_gate和W_up的范数
        W_gate_norm = float(np.linalg.norm(W_gate)) if W_gate is not None else 0
        W_up_norm = float(np.linalg.norm(W_up))
        
        # PR
        PR_Wdown = float(S_wd.sum()**2 / (len(S_wd) * (S_wd**2).sum() + 1e-10))
        
        # 条件数
        cond_Wdown = float(S_wd[0] / (S_wd[-1] + 1e-10))
        
        # 谱衰减率
        spectral_decay_5 = float(S_wd[min(4, len(S_wd)-1)] / (S_wd[0] + 1e-10))
        spectral_decay_10 = float(S_wd[min(9, len(S_wd)-1)] / (S_wd[0] + 1e-10))
        
        result = {
            'layer': l_idx,
            'layer_frac': l_idx / info.n_layers,
            'W_down_norm': W_down_norm,
            'W_down_frobenius': W_down_frobenius,
            'post_ln_norm': post_ln_norm,
            'input_ln_norm': input_ln_norm,
            'W_gate_norm': W_gate_norm,
            'W_up_norm': W_up_norm,
            'PR_Wdown': PR_Wdown,
            'cond_Wdown': cond_Wdown,
            'spectral_decay_5': spectral_decay_5,
            'spectral_decay_10': spectral_decay_10,
            'spectrum_corr_prev': float(spectrum_corr),
        }
        results.append(result)
        
        print(f"  W_down_norm={W_down_norm:.2f}, post_ln_norm={post_ln_norm:.4f}")
        print(f"  PR={PR_Wdown:.4f}, cond={cond_Wdown:.1f}")
        print(f"  spectrum_corr_prev={spectrum_corr:.4f}")
        
        prev_W_down_spectrum = S_wd_normed.copy()
        torch.cuda.empty_cache()
    
    # 汇总
    print("\n" + "="*70)
    print("P533 汇总: 训练动力学差异")
    print("="*70)
    
    # 层间相关性分析
    W_down_norms = [r['W_down_norm'] for r in results]
    post_ln_norms = [r['post_ln_norm'] for r in results]
    W_gate_norms = [r['W_gate_norm'] for r in results]
    W_up_norms = [r['W_up_norm'] for r in results]
    PRs = [r['PR_Wdown'] for r in results]
    spectrum_corrs = [r['spectrum_corr_prev'] for r in results if r['spectrum_corr_prev'] != 0]
    
    # 各权重范数的层间共变性
    if len(results) > 3:
        try:
            rho_down_ln, _ = spearmanr(W_down_norms, post_ln_norms)
            rho_down_gate, _ = spearmanr(W_down_norms, W_gate_norms)
            rho_ln_gate, _ = spearmanr(post_ln_norms, W_gate_norms)
            rho_pr_down, _ = spearmanr(PRs, W_down_norms)
            
            print(f"  Spearman(W_down_norm, post_ln_norm): {rho_down_ln:.4f}")
            print(f"  Spearman(W_down_norm, W_gate_norm): {rho_down_gate:.4f}")
            print(f"  Spearman(post_ln_norm, W_gate_norm): {rho_ln_gate:.4f}")
            print(f"  Spearman(PR, W_down_norm): {rho_pr_down:.4f}")
        except:
            print("  Spearman计算失败")
    
    mean_spectrum_corr = np.mean(spectrum_corrs) if spectrum_corrs else 0
    mean_PR = np.mean(PRs)
    
    print(f"\n  mean W_down频谱层间相关: {mean_spectrum_corr:.4f}")
    print(f"  mean PR_Wdown: {mean_PR:.4f}")
    
    # 判断因果链质量
    print("\n  因果链质量评估:")
    if mean_spectrum_corr > 0.95:
        print("  >> W_down频谱高度层间相关 → 训练产生有序结构")
    elif mean_spectrum_corr > 0.8:
        print("  >> W_down频谱中度层间相关 → 部分有序")
    else:
        print("  >> W_down频谱层间低相关 → 无序/RL训练破坏了结构")
    
    result_path = f"tests/glm5/results/phase_cxiv/p533_{model_name}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({'results': [to_native(r) for r in results], 'model': model_name,
                   'mean_spectrum_corr': float(mean_spectrum_corr)}, f, indent=2)
    print(f"\n结果已保存: {result_path}")
    
    return results


# ===== 主函数 =====
def main():
    parser = argparse.ArgumentParser(description="Phase CXIV: 从低秩结构到统一传播方程")
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["p528", "p529", "p530", "p531", "p532", "p533"])
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Phase CXIV: {args.experiment.upper()} | 模型: {args.model}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, model_name=args.model)
    print(f"模型: {info.model_class}, 层数: {info.n_layers}, d_model: {info.d_model}, "
          f"mlp_type: {info.mlp_type}")
    
    start_time = time.time()
    
    if args.experiment == "p528":
        results = run_p528(model, tokenizer, device, args.model)
    elif args.experiment == "p529":
        results = run_p529(model, tokenizer, device, args.model)
    elif args.experiment == "p530":
        results = run_p530(model, tokenizer, device, args.model)
    elif args.experiment == "p531":
        results = run_p531(model, tokenizer, device, args.model)
    elif args.experiment == "p532":
        results = run_p532(model, tokenizer, device, args.model)
    elif args.experiment == "p533":
        results = run_p533(model, tokenizer, device, args.model)
    
    elapsed = time.time() - start_time
    print(f"\n实验耗时: {elapsed:.1f}秒")
    
    release_model(model)


if __name__ == "__main__":
    main()
