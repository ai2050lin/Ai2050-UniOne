"""
Phase CXII-P520/P521/P522: Jacobian低秩结构解析
==================================================

Phase CXI核心发现:
- Jacobian极度低秩: PR_J≈0.02-0.03, 仅1-2个方向被放大
- λ_max指数增长: Qwen3 37→933, GLM4 1.8→1050
- λ_max∝post_ln_norm(GLM4 ρ=1.000)∝h_in_norm(Qwen3 ρ=1.00)
- DS7B传播机制根本不同: λ_max非单调, 无稳定预测因子
- 方向无关传播: global_ratio与扰动方向无关

Phase CXII核心思路:
1. P520: 验证PR_J≈0.02是真实低秩还是投影伪影(增加探测方向数)
2. P521: 识别被极大放大的1-2个方向(与W_U/h_in的关系)
3. P522: λ_max因果干预(缩放post_ln_norm, 验证因果链)

P520: Jacobian真实秩验证
  - 用100/200/500个随机探测方向估计Jacobian
  - 如果PR_J随n_probes增加而增大 → 投影伪影
  - 如果PR_J保持≈0.02 → 真实低秩
  - 直接测量: 对每个探测方向, 计算Jacobian-vector乘积Jv
  - 构建J_proj = [Jv_1, ..., Jv_n], SVD获取真实秩

P521: 被放大方向的识别
  - 找到J_proj的top-1奇异向量方向 v_max
  - 测量v_max与以下方向的对齐度:
    a) h_in方向 (当前隐藏状态方向)
    b) W_U top奇异向量 (解码空间主成分)
    c) W_down top奇异向量 (MLP输出主成分)
    d) 残差流方向 (h_out - h_in)
    e) LayerNorm权重方向 (post_ln_weight)
  - 如果v_max与h_in对齐 → 信号在自身方向被放大
  - 如果v_max与W_U对齐 → 训练选择了解码友好方向

P522: λ_max因果干预
  - 直接缩放post_ln_norm: weight *= scale_factor
  - 测量缩放后λ_max的变化
  - 如果λ_max∝scale → 因果关系(不是相关)
  - 如果λ_max不变 → 相关关系(post_ln只是指示器)
  - 对比: 缩放input_ln_norm, 缩放W_down, 看哪个改变λ_max

使用方法:
    python phase_cxii_jacobian_structure.py --model qwen3 --experiment p520
    python phase_cxii_jacobian_structure.py --model glm4 --experiment p521
    python phase_cxii_jacobian_structure.py --model deepseek7b --experiment p522
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


# 工具函数
def to_native(d):
    """确保所有值是Python原生类型"""
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
    """
    通过W_down扰动估计Jacobian的投影
    
    返回: J_proj [n_probes, d_model], h_out_baseline
    """
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_hidden_states=True)
        h_out_baseline = outputs.hidden_states[l_idx + 1][0, -1].detach().clone()
    
    J_probes = []
    for i in range(n_probes):
        # 扰动W_down
        orig_w = layers[l_idx].mlp.down_proj.weight.data.clone()
        layers[l_idx].mlp.down_proj.weight.data = orig_w * (1 - alpha)
        
        with torch.no_grad():
            out = model(inputs["input_ids"], output_hidden_states=True)
            h_perturbed = out.hidden_states[l_idx + 1][0, -1].detach().clone()
        
        # 恢复
        layers[l_idx].mlp.down_proj.weight.data = orig_w
        
        delta_h = (h_perturbed - h_out_baseline) / alpha
        J_probes.append(delta_h.cpu().float())
        
        if i % 20 == 0:
            torch.cuda.empty_cache()
    
    J_proj = torch.stack(J_probes)  # [n_probes, d_model]
    return J_proj, h_out_baseline


# ===== P520: Jacobian真实秩验证 =====
def run_p520(model, tokenizer, device, model_name):
    """
    用不同数量的探测方向验证Jacobian的PR_J是否是投影伪影
    
    如果PR_J≈0.02是投影伪影 → 随n_probes增加, PR_J应该增大
    如果PR_J≈0.02是真实低秩 → 随n_probes增加, PR_J保持不变
    """
    print("\n" + "="*70)
    print("P520: Jacobian真实秩验证")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    test_text = "The fundamental nature of reality can be understood through mathematical structures that underlie physical phenomena."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    # 采样3个代表性层: 早期/中期/晚期
    sample_layers = [info.n_layers // 4, info.n_layers // 2, 3 * info.n_layers // 4]
    print(f"采样层: {sample_layers}")
    
    # 探测方向数
    n_probes_list = [30, 60, 100, 150, 200]
    
    results = []
    
    for l_idx in sample_layers:
        print(f"\n--- 层 {l_idx} ---")
        
        for n_probes in n_probes_list:
            print(f"  n_probes={n_probes}...", end=" ", flush=True)
            
            J_proj, h_out = compute_jacobian_probes(model, layers, l_idx, inputs, info, n_probes=n_probes)
            
            # SVD — 对J_proj^T做SVD获取输出空间的奇异向量
            J_proj_T = J_proj.T  # [d_model, n_probes]
            U_jt, S_j, Vt_jt = torch.linalg.svd(J_proj_T, full_matrices=False)
            
            # 参与率
            PR_J = (S_j.sum())**2 / (len(S_j) * (S_j**2).sum() + 1e-10)
            
            # 有效秩: PR达到90%所需的奇异值数
            S_j_np = S_j.cpu().numpy()
            cum_energy = np.cumsum(S_j_np**2) / (np.sum(S_j_np**2) + 1e-10)
            n_90 = np.searchsorted(cum_energy, 0.9) + 1
            n_99 = np.searchsorted(cum_energy, 0.99) + 1
            
            # 前5个奇异值的能量占比
            top1_energy = S_j_np[0]**2 / (np.sum(S_j_np**2) + 1e-10)
            top5_energy = np.sum(S_j_np[:5]**2) / (np.sum(S_j_np**2) + 1e-10)
            
            lambda_max = S_j[0].item()
            lambda_2 = S_j[1].item() if len(S_j) > 1 else 0
            ratio_1_2 = lambda_max / (lambda_2 + 1e-10)
            
            result = {
                'layer': l_idx,
                'n_probes': n_probes,
                'PR_J': float(PR_J),
                'lambda_max': float(lambda_max),
                'lambda_2': float(lambda_2),
                'ratio_1_2': float(ratio_1_2),
                'n_90': int(n_90),
                'n_99': int(n_99),
                'top1_energy': float(top1_energy),
                'top5_energy': float(top5_energy),
            }
            results.append(result)
            
            print(f"PR_J={PR_J:.4f}, n_90={n_90}, n_99={n_99}, top1={top1_energy:.3f}, λ_max/λ_2={ratio_1_2:.1f}")
            
            torch.cuda.empty_cache()
    
    # 汇总分析
    print("\n" + "="*70)
    print("P520 汇总分析: PR_J vs n_probes")
    print("="*70)
    
    for l_idx in sample_layers:
        layer_results = [r for r in results if r['layer'] == l_idx]
        print(f"\n  层 {l_idx}:")
        for r in layer_results:
            print(f"    n_probes={r['n_probes']:3d}: PR_J={r['PR_J']:.4f}, "
                  f"n_90={r['n_90']:2d}, n_99={r['n_99']:2d}, "
                  f"top1={r['top1_energy']:.3f}, λ_max/λ_2={r['ratio_1_2']:.1f}")
    
    # 判断: PR_J是否随n_probes增长
    print("\n  判断PR_J是否随n_probes增长:")
    for l_idx in sample_layers:
        layer_results = [r for r in results if r['layer'] == l_idx]
        pr_values = [r['PR_J'] for r in layer_results]
        n_values = [r['n_probes'] for r in layer_results]
        if len(pr_values) >= 2:
            # 简单线性趋势
            pr_min, pr_max = min(pr_values), max(pr_values)
            if pr_max / (pr_min + 1e-10) > 2:
                print(f"    L{l_idx}: PR_J显著增长({pr_min:.4f}→{pr_max:.4f}), 可能是投影伪影")
            else:
                print(f"    L{l_idx}: PR_J稳定({pr_min:.4f}→{pr_max:.4f}), 真实低秩")
    
    # 保存结果
    result_path = f"tests/glm5/results/phase_cxii/p520_{model_name}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({'results': [to_native(r) for r in results], 'model': model_name}, f, indent=2)
    print(f"\n结果已保存: {result_path}")
    
    return results


# ===== P521: 被放大方向的识别 =====
def run_p521(model, tokenizer, device, model_name):
    """
    识别Jacobian被极大放大的1-2个方向, 与已知方向对齐
    
    方法: 用足够多的探测方向估计J_proj, 找到top奇异向量v_max,
    测量v_max与h_in/W_U/W_down/residual/LN的对齐度
    """
    print("\n" + "="*70)
    print("P521: 被放大方向的识别")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    test_text = "The fundamental nature of reality can be understood through mathematical structures that underlie physical phenomena."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    sample_layers = get_sample_layers(info.n_layers, min(6, info.n_layers))
    print(f"采样层: {sample_layers}")
    
    # 获取W_U的SVD(用于对齐度计算)
    W_U = get_W_U(model)  # [vocab, d_model]
    k_wu = min(20, min(W_U.shape) - 2)
    U_wu, S_wu, Vt_wu = svds(W_U.T.astype(np.float32), k=k_wu)
    # U_wu: [d_model, k], W_U行空间的top奇异向量
    U_wu = U_wu[:, ::-1]  # 降序
    
    results = []
    
    for l_idx in sample_layers:
        print(f"\n--- 层 {l_idx} ---")
        
        # 获取基线隐藏状态
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
            h_states = outputs.hidden_states
        
        h_in = h_states[l_idx][0, -1].detach().clone()
        h_out = h_states[l_idx + 1][0, -1].detach().clone()
        
        # 获取权重
        weights = get_layer_weights(layers[l_idx], info.d_model, info.mlp_type)
        W_down = weights.W_down
        W_gate = weights.W_gate
        post_ln_w = weights.post_attn_layernorm_weight
        input_ln_w = weights.input_layernorm_weight
        
        # 用100个探测方向估计Jacobian
        n_probes = 100
        J_proj, _ = compute_jacobian_probes(model, layers, l_idx, inputs, info, n_probes=n_probes)
        
        # SVD获取Jacobian的top奇异向量
        # J_proj shape: [n_probes, d_model]
        # SVD: U[n_probes, n_probes] S[min] Vt[d_model, d_model]
        # Vt的行是输入空间(d_model)的奇异向量
        # 但我们需要的是输出空间(d_model)的奇异向量
        # J_proj = U S Vt, J_proj^T = V S U^T
        # 对J_proj^T做SVD: V S U^T → 输出方向在V的列中
        J_proj_T = J_proj.T  # [d_model, n_probes]
        U_jt, S_j, Vt_jt = torch.linalg.svd(J_proj_T, full_matrices=False)
        # U_jt: [d_model, min(d_model, n_probes)] — 输出空间的奇异向量
        # Vt_jt: [min(d_model, n_probes), n_probes] — 输入空间
        
        # v_max: Jacobian最大奇异值对应的输出方向
        v_max = U_jt[:, 0].cpu().float().numpy()  # [d_model]
        v_2 = U_jt[:, 1].cpu().float().numpy() if U_jt.shape[1] > 1 else np.zeros_like(v_max)
        
        # h_in方向
        h_in_np = h_in.cpu().float().numpy()
        h_in_dir = h_in_np / (np.linalg.norm(h_in_np) + 1e-10)
        
        # h_out方向
        h_out_np = h_out.cpu().float().numpy()
        h_out_dir = h_out_np / (np.linalg.norm(h_out_np) + 1e-10)
        
        # 残差方向
        residual = h_out_np - h_in_np
        residual_dir = residual / (np.linalg.norm(residual) + 1e-10)
        
        # W_down top奇异向量(输出空间d_model)
        # W_down shape: [d_model, intermediate_size]
        # svds: U[d_model, k] S[k] Vt[k, intermediate_size]
        # W_down的输出空间就是d_model, U的列是d_model维的奇异向量
        k_wd = min(5, min(W_down.shape) - 1)
        U_wd, S_wd, _ = svds(W_down.astype(np.float32), k=k_wd)
        U_wd = U_wd[:, ::-1]
        w_down_top1 = U_wd[:, 0]  # [d_model] — 输出空间方向
        w_down_top1 = w_down_top1 / (np.linalg.norm(w_down_top1) + 1e-10)
        
        # W_gate top奇异向量(如果存在) — 使用Vt的行(输入空间d_model)
        w_gate_top1 = None
        if W_gate is not None:
            k_wg = min(5, min(W_gate.shape) - 1)
            U_wg, S_wg, Vt_wg = svds(W_gate.astype(np.float32), k=k_wg)
            Vt_wg = Vt_wg[::-1]  # 降序
            # Vt_wg的行是输入空间(d_model)的奇异向量
            w_gate_top1 = Vt_wg[0]  # [d_model]
            w_gate_top1 = w_gate_top1 / (np.linalg.norm(w_gate_top1) + 1e-10)
        
        # post_ln方向
        post_ln_dir = post_ln_w / (np.linalg.norm(post_ln_w) + 1e-10) if post_ln_w is not None else np.zeros(info.d_model)
        input_ln_dir = input_ln_w / (np.linalg.norm(input_ln_w) + 1e-10) if input_ln_w is not None else np.zeros(info.d_model)
        
        # W_U top奇异向量方向
        wu_top1 = U_wu[:, 0]
        wu_top1 = wu_top1 / (np.linalg.norm(wu_top1) + 1e-10)
        
        # 计算对齐度(cosine similarity)
        def cos_sim(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
        
        alignments = {
            'cos_vmax_h_in': cos_sim(v_max, h_in_dir),
            'cos_vmax_h_out': cos_sim(v_max, h_out_dir),
            'cos_vmax_residual': cos_sim(v_max, residual_dir),
            'cos_vmax_W_U_top1': cos_sim(v_max, wu_top1),
            'cos_vmax_W_down_top1': cos_sim(v_max, w_down_top1),
            'cos_vmax_post_ln': cos_sim(v_max, post_ln_dir),
            'cos_vmax_input_ln': cos_sim(v_max, input_ln_dir),
            'cos_v2_h_in': cos_sim(v_2, h_in_dir),
            'cos_v2_h_out': cos_sim(v_2, h_out_dir),
            'cos_v2_residual': cos_sim(v_2, residual_dir),
            'cos_v2_W_U_top1': cos_sim(v_2, wu_top1),
        }
        
        if w_gate_top1 is not None:
            alignments['cos_vmax_W_gate_top1'] = cos_sim(v_max, w_gate_top1)
            alignments['cos_v2_W_gate_top1'] = cos_sim(v_2, w_gate_top1)
        
        # Jacobian谱特征
        S_j_np = S_j.cpu().numpy()
        PR_J = float((S_j.sum())**2 / (len(S_j) * (S_j**2).sum() + 1e-10))
        top1_energy = float(S_j_np[0]**2 / (np.sum(S_j_np**2) + 1e-10))
        
        result = {
            'layer': l_idx,
            'layer_frac': l_idx / info.n_layers,
            'lambda_max': float(S_j[0]),
            'lambda_2': float(S_j[1]) if len(S_j) > 1 else 0,
            'PR_J': PR_J,
            'top1_energy': top1_energy,
            'h_in_norm': float(h_in.norm()),
            'h_out_norm': float(h_out.norm()),
            **alignments,
        }
        results.append(result)
        
        # 打印对齐度
        print(f"  λ_max={S_j[0]:.2f}, top1_energy={top1_energy:.3f}, PR_J={PR_J:.4f}")
        print(f"  v_max对齐度:")
        for k, v in sorted(alignments.items()):
            if k.startswith('cos_vmax'):
                name = k.replace('cos_vmax_', '')
                print(f"    {name}: {v:.4f}")
        print(f"  v_2对齐度:")
        for k, v in sorted(alignments.items()):
            if k.startswith('cos_v2'):
                name = k.replace('cos_v2_', '')
                print(f"    {name}: {v:.4f}")
        
        torch.cuda.empty_cache()
    
    # 汇总分析
    print("\n" + "="*70)
    print("P521 汇总分析: v_max最对齐的方向")
    print("="*70)
    
    # 对每个层找最大对齐度
    alignment_keys = [k for k in results[0].keys() if k.startswith('cos_vmax_')]
    
    for r in results:
        max_align = 0
        max_dir = ""
        for k in alignment_keys:
            if abs(r[k]) > abs(max_align):
                max_align = r[k]
                max_dir = k.replace('cos_vmax_', '')
        print(f"  L{r['layer']}: v_max最对齐={max_dir}(cos={max_align:.4f}), "
              f"λ_max={r['lambda_max']:.2f}, top1_energy={r['top1_energy']:.3f}")
    
    # v_max与各方向的平均对齐度
    print(f"\n  各方向平均|cos_sim|:")
    for k in alignment_keys:
        name = k.replace('cos_vmax_', '')
        mean_cos = np.mean([abs(r[k]) for r in results])
        print(f"    {name}: {mean_cos:.4f}")
    
    # 保存结果
    result_path = f"tests/glm5/results/phase_cxii/p521_{model_name}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({'results': [to_native(r) for r in results], 'model': model_name}, f, indent=2)
    print(f"\n结果已保存: {result_path}")
    
    return results


# ===== P522: λ_max因果干预 =====
def run_p522(model, tokenizer, device, model_name):
    """
    通过因果干预验证λ_max∝post_ln_norm是否是因果关系
    
    方法: 直接缩放post_ln_norm, 测量λ_max的变化
    - 如果λ_max∝scale → 因果关系
    - 如果λ_max不变 → 相关关系
    对比: 缩放input_ln_norm, 缩放W_down
    """
    print("\n" + "="*70)
    print("P522: λ_max因果干预")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    test_text = "The fundamental nature of reality can be understood through mathematical structures that underlie physical phenomena."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    # 只测3个层(早期/中期/晚期)
    sample_layers = [info.n_layers // 4, info.n_layers // 2, 3 * info.n_layers // 4]
    print(f"采样层: {sample_layers}")
    
    # 缩放因子
    scale_factors = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    
    # 干预类型
    interventions = ['post_ln_norm', 'input_ln_norm', 'W_down']
    
    results = []
    
    for l_idx in sample_layers:
        print(f"\n--- 层 {l_idx} ---")
        
        # 基线λ_max
        J_proj_baseline, h_out_baseline = compute_jacobian_probes(
            model, layers, l_idx, inputs, info, n_probes=30
        )
        J_proj_T = J_proj_baseline.T
        U_jt, S_j, _ = torch.linalg.svd(J_proj_T, full_matrices=False)
        lambda_max_baseline = S_j[0].item()
        
        # 基线权重
        weights = get_layer_weights(layers[l_idx], info.d_model, info.mlp_type)
        post_ln_norm_baseline = np.linalg.norm(weights.post_attn_layernorm_weight) if weights.post_attn_layernorm_weight is not None else 0
        input_ln_norm_baseline = np.linalg.norm(weights.input_layernorm_weight) if weights.input_layernorm_weight is not None else 0
        w_down_norm_baseline = np.linalg.norm(weights.W_down)
        
        print(f"  基线: λ_max={lambda_max_baseline:.4f}, post_ln_norm={post_ln_norm_baseline:.2f}, "
              f"input_ln_norm={input_ln_norm_baseline:.2f}, W_down_norm={w_down_norm_baseline:.2f}")
        
        for intervention in interventions:
            print(f"\n  干预: {intervention}")
            
            for scale in scale_factors:
                # 保存原始权重
                orig_post_ln = None
                orig_input_ln = None
                orig_w_down = None
                
                # 应用干预
                if intervention == 'post_ln_norm':
                    for ln_name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
                        if hasattr(layers[l_idx], ln_name):
                            ln = getattr(layers[l_idx], ln_name)
                            if hasattr(ln, "weight"):
                                orig_post_ln = ln.weight.data.clone()
                                ln.weight.data = orig_post_ln * scale
                                break
                elif intervention == 'input_ln_norm':
                    for ln_name in ["input_layernorm", "ln_1", "layernorm"]:
                        if hasattr(layers[l_idx], ln_name):
                            ln = getattr(layers[l_idx], ln_name)
                            if hasattr(ln, "weight"):
                                orig_input_ln = ln.weight.data.clone()
                                ln.weight.data = orig_input_ln * scale
                                break
                elif intervention == 'W_down':
                    orig_w_down = layers[l_idx].mlp.down_proj.weight.data.clone()
                    layers[l_idx].mlp.down_proj.weight.data = orig_w_down * scale
                
                # 测量干预后λ_max
                J_proj_int, _ = compute_jacobian_probes(
                    model, layers, l_idx, inputs, info, n_probes=30
                )
                J_proj_T_int = J_proj_int.T
                U_jt_int, S_j_int, _ = torch.linalg.svd(J_proj_T_int, full_matrices=False)
                lambda_max_int = S_j_int[0].item()
                
                # 恢复权重
                if orig_post_ln is not None:
                    for ln_name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
                        if hasattr(layers[l_idx], ln_name):
                            ln = getattr(layers[l_idx], ln_name)
                            if hasattr(ln, "weight"):
                                ln.weight.data = orig_post_ln
                                break
                if orig_input_ln is not None:
                    for ln_name in ["input_layernorm", "ln_1", "layernorm"]:
                        if hasattr(layers[l_idx], ln_name):
                            ln = getattr(layers[l_idx], ln_name)
                            if hasattr(ln, "weight"):
                                ln.weight.data = orig_input_ln
                                break
                if orig_w_down is not None:
                    layers[l_idx].mlp.down_proj.weight.data = orig_w_down
                
                # 计算变化
                ratio = lambda_max_int / (lambda_max_baseline + 1e-10)
                
                result = {
                    'layer': l_idx,
                    'intervention': intervention,
                    'scale': scale,
                    'lambda_max_int': float(lambda_max_int),
                    'lambda_max_baseline': float(lambda_max_baseline),
                    'ratio': float(ratio),
                    'log_ratio': float(np.log(ratio + 1e-10)),
                }
                results.append(result)
                
                print(f"    scale={scale:.2f}: λ_max={lambda_max_int:.4f}, ratio={ratio:.4f}")
                
                torch.cuda.empty_cache()
    
    # 汇总分析
    print("\n" + "="*70)
    print("P522 汇总分析: λ_max因果干预")
    print("="*70)
    
    for intervention in interventions:
        print(f"\n  {intervention}:")
        int_results = [r for r in results if r['intervention'] == intervention]
        
        # 对每个scale计算平均ratio
        for scale in scale_factors:
            scale_results = [r for r in int_results if r['scale'] == scale]
            if scale_results:
                mean_ratio = np.mean([r['ratio'] for r in scale_results])
                print(f"    scale={scale:.2f}: mean_ratio={mean_ratio:.4f}")
        
        # 回归: ratio vs scale
        scales = np.array([r['scale'] for r in int_results])
        ratios = np.array([r['ratio'] for r in int_results])
        if len(scales) > 2:
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression().fit(scales.reshape(-1, 1), ratios)
            r2 = reg.score(scales.reshape(-1, 1), ratios)
            print(f"    线性回归 ratio~scale: R2={r2:.4f}, slope={reg.coef_[0]:.4f}")
            
            # 二次回归
            X = np.column_stack([scales, scales**2])
            reg2 = LinearRegression().fit(X, ratios)
            r2_quad = reg2.score(X, ratios)
            print(f"    二次回归: R2={r2_quad:.4f}")
    
    # 判断因果关系
    print("\n  因果判断:")
    for intervention in interventions:
        int_results = [r for r in results if r['intervention'] == intervention]
        scales = np.array([r['scale'] for r in int_results])
        ratios = np.array([r['ratio'] for r in int_results])
        if len(scales) > 2:
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression().fit(scales.reshape(-1, 1), ratios)
            r2 = reg.score(scales.reshape(-1, 1), ratios)
            slope = reg.coef_[0]
            
            if r2 > 0.9 and slope > 0.3:
                print(f"    {intervention}: 强因果(R2={r2:.3f}, slope={slope:.3f})")
            elif r2 > 0.7:
                print(f"    {intervention}: 中等因果(R2={r2:.3f}, slope={slope:.3f})")
            else:
                print(f"    {intervention}: 弱因果/非因果(R2={r2:.3f}, slope={slope:.3f})")
    
    # 保存结果
    result_path = f"tests/glm5/results/phase_cxii/p522_{model_name}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({'results': [to_native(r) for r in results], 'model': model_name}, f, indent=2)
    print(f"\n结果已保存: {result_path}")
    
    return results


# ===== 主函数 =====
def main():
    parser = argparse.ArgumentParser(description="Phase CXII: Jacobian低秩结构解析")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["p520", "p521", "p522"])
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Phase CXII: {args.experiment.upper()} | 模型: {args.model}")
    print(f"{'='*70}")
    
    # 加载模型
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, model_name=args.model)
    print(f"模型: {info.model_class}, 层数: {info.n_layers}, d_model: {info.d_model}, "
          f"mlp_type: {info.mlp_type}")
    
    start_time = time.time()
    
    # 运行实验
    if args.experiment == "p520":
        results = run_p520(model, tokenizer, device, args.model)
    elif args.experiment == "p521":
        results = run_p521(model, tokenizer, device, args.model)
    elif args.experiment == "p522":
        results = run_p522(model, tokenizer, device, args.model)
    
    elapsed = time.time() - start_time
    print(f"\n实验耗时: {elapsed:.1f}秒")
    
    # 释放模型
    release_model(model)


if __name__ == "__main__":
    main()
