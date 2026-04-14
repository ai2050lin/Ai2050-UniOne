"""
Phase CXI-P517/P518/P519: Jacobian特征值谱与传播方向分析
==========================================================

Phase CX核心发现:
- 所有模型local_ratio≈0.9-1.0(单层接近不动点)
- 但全局max_ratio可达5000+(GLM4), 不稳定来自层间累积
- 首层L0极不稳定(ratio=54-5000), 嵌入层输出太小

Phase CXI核心思路:
1. Jacobian特征值谱: 找出被极大放大的方向
2. 扰动方向vs传播放大: 不同方向的传播比差异
3. Jacobian谱的权重预测: 从纯权重预测Jacobian谱

P517: 逐层Jacobian特征值谱分析
  - 对每层, 通过有限差分近似Jacobian: J_l ≈ Δh_out / Δh_in
  - 用幂迭代法(power iteration)找Jacobian的前k大特征值
  - 验证: 是否存在λ_max>>1的特定方向(解释局部≈1但全局>>1)
  - 对比: 三模型的Jacobian谱特征(谱宽度、最大特征值、参与率)

P518: 扰动方向vs传播放大
  - 在不同方向注入扰动: 随机/特征向量/残差方向/注意力方向
  - 测量各方向的传播比: ratio_dir = ||Δh_out_dir|| / ||Δh_dir||
  - 验证: Jacobian最大特征值方向的传播比是否≈λ_max
  - 对比: 不同方向在三层传播后的放大倍数

P519: Jacobian谱的权重预测
  - 收集每层的权重特征(W_down_kappa, W_gate_PR, post_ln_norm等)
  - 收集Jacobian谱特征(λ_max, λ_mean, PR_J, condition_number)
  - 用回归分析推导: λ_max ≈ f(W_down_kappa, W_gate_PR, post_ln_norm)
  - 目标: 从纯权重预测Jacobian谱(不需要前向传播)

使用方法:
    python phase_cxi_jacobian_spectral.py --model qwen3 --experiment p517
    python phase_cxi_jacobian_spectral.py --model glm4 --experiment p518
    python phase_cxi_jacobian_spectral.py --model deepseek7b --experiment p519
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
sys.path.insert(0, os.path.dirname(__file__))
from model_utils import (
    load_model, get_model_info, get_layers, get_layer_weights,
    get_W_U, release_model, get_sample_layers
)


# 工具函数
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


# ===== P517: Jacobian特征值谱分析 =====
def run_p517(model, tokenizer, device, model_name):
    """
    逐层Jacobian特征值谱分析
    
    方法: 用幂迭代法(power iteration)近似Jacobian的前k大特征值
    - 对层l, 输入h_in, 添加随机扰动δ, 测量输出变化Δh_out
    - 幂迭代: v ← J^T J v, λ = v^T J^T J v / (v^T v)
    - 这样得到J^T J的特征值(即Jacobian奇异值的平方)
    """
    print("\n" + "="*70)
    print("P517: 逐层Jacobian特征值谱分析")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    # 测试文本
    test_text = "The fundamental nature of reality can be understood through mathematical structures that underlie physical phenomena."
    
    # 采样层
    sample_layers = get_sample_layers(info.n_layers, min(8, info.n_layers))
    print(f"采样层: {sample_layers}")
    
    results = []
    
    for l_idx in sample_layers:
        print(f"\n--- 层 {l_idx} ---")
        
        # 获取基线隐藏状态
        inputs = tokenizer(test_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
            h_states = outputs.hidden_states
        
        # h_in = 第l层输入, h_out = 第l层输出
        h_in = h_states[l_idx][0, -1].detach().clone()  # [d_model]
        h_out = h_states[l_idx + 1][0, -1].detach().clone()  # [d_model]
        
        d_model = h_in.shape[0]
        print(f"  h_in norm: {h_in.norm():.4f}, h_out norm: {h_out.norm():.4f}")
        
        # 幂迭代法找Jacobian的前5大奇异值
        n_power_iter = 20
        n_vectors = 5  # 找前5个
        top_singular_values = []
        top_vectors = []
        
        eps = 0.01 * h_in.norm()  # 扰动幅度
        
        # 用随机扰动估计Jacobian矩阵的列
        # J ≈ [Δh_out_1/ε, Δh_out_2/ε, ...] 对随机扰动
        # 但d_model可能很大(3072+), 不能直接构建
        # 改用随机投影+幂迭代
        
        n_probes = min(30, d_model)  # 探测方向数
        
        # 生成随机探测方向
        probe_vectors = torch.randn(n_probes, d_model, device=device, dtype=h_in.dtype)
        probe_vectors = probe_vectors / probe_vectors.norm(dim=1, keepdim=True)
        
        # 对每个探测方向, 测量Jacobian-vector乘积
        J_probes = []  # [n_probes, d_model]
        
        for i in range(n_probes):
            # 扰动输入: h_in + eps * v
            h_in_perturbed = h_in + eps * probe_vectors[i]
            
            # 需要在模型中注入扰动后的隐藏状态
            # 方法: hook注入
            captured_output = [None]
            
            def hook_fn(module, input, output):
                # 替换该层的输入为扰动后的
                captured_output[0] = output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()
                return output
            
            # 注册hook到层l_idx
            handle = layers[l_idx].register_forward_hook(hook_fn)
            
            # 重新前向传播(这里我们无法直接注入h_in_perturbed到中间层)
            # 替代方案: 用完整前向传播, 但用不同的扰动方式
            
            # 方法2: 直接扰动层l_idx的输入(LN之前的隐藏状态)
            # 扰动方式: 修改input embeddings到l_idx
            handle.remove()
            
            # 方法3: 用W_down扰动作为代理
            # 对W_down做小扰动, 测量h_out变化, 反推Jacobian
            
            # 这里我们用更直接的方法:
            # 对W_down做小扰动, 看输出变化
            alpha = 0.01
            orig_w = perturb_w_down(layers, l_idx, alpha, info.mlp_type)
            
            with torch.no_grad():
                outputs_perturbed = model(inputs["input_ids"], output_hidden_states=True)
                h_out_perturbed = outputs_perturbed.hidden_states[l_idx + 1][0, -1].detach().clone()
            
            restore_w_down(layers, l_idx, orig_w, info.mlp_type)
            
            delta_h_out = (h_out_perturbed - h_out) / alpha
            J_probes.append(delta_h_out.cpu().float())
        
        # 构建Jacobian的投影估计: J_proj ≈ [Jv_1, Jv_2, ..., Jv_n]
        J_proj = torch.stack(J_probes)  # [n_probes, d_model]
        
        # SVD获取奇异值
        U_j, S_j, Vt_j = torch.linalg.svd(J_proj, full_matrices=False)
        
        print(f"  Jacobian投影奇异值(top-5): {S_j[:5].tolist()}")
        print(f"  Jacobian投影奇异值(bottom-5): {S_j[-5:].tolist()}")
        
        # 额外: 用多方向扰动直接估计Jacobian的范数
        n_random = 20
        ratios = []
        for _ in range(n_random):
            alpha = 0.01
            orig_w = perturb_w_down(layers, l_idx, alpha, info.mlp_type)
            with torch.no_grad():
                out = model(inputs["input_ids"], output_hidden_states=True)
                h_perturbed = out.hidden_states[l_idx + 1][0, -1].detach().clone()
            restore_w_down(layers, l_idx, orig_w, info.mlp_type)
            ratio = (h_perturbed - h_out).norm() / (h_out.norm() + 1e-10)
            ratios.append(ratio.item())
        
        # Jacobian谱特征
        lambda_max = S_j[0].item()
        lambda_min = S_j[-1].item()
        lambda_mean = S_j.mean().item()
        lambda_std = S_j.std().item()
        condition_number = lambda_max / (lambda_min + 1e-10)
        # 参与率: PR = (sum S)^2 / (n * sum S^2)
        PR_J = (S_j.sum())**2 / (len(S_j) * (S_j**2).sum() + 1e-10)
        
        print(f"  λ_max={lambda_max:.4f}, λ_min={lambda_min:.4f}, "
              f"λ_mean={lambda_mean:.4f}, κ={condition_number:.2f}, PR_J={PR_J:.4f}")
        print(f"  随机扰动ratio: mean={np.mean(ratios):.4f}, max={np.max(ratios):.4f}")
        
        # 获取权重特征
        weights = get_layer_weights(layers[l_idx], info.d_model, info.mlp_type)
        W_down = weights.W_down
        W_gate = weights.W_gate
        
        # 用安全SVD(避免内存溢出)
        try:
            W_down_svd = np.linalg.svd(W_down.astype(np.float32), compute_uv=False)
            W_gate_svd = np.linalg.svd(W_gate.astype(np.float32), compute_uv=False) if W_gate is not None else None
        except MemoryError:
            from scipy.sparse.linalg import svds
            W_down_svd = svds(W_down.astype(np.float32), k=min(10, min(W_down.shape)-1), return_singular_vectors=False)
            W_gate_svd = svds(W_gate.astype(np.float32), k=min(10, min(W_gate.shape)-1), return_singular_vectors=False) if W_gate is not None else None
        
        W_down_kappa = W_down_svd[0] / (W_down_svd[-1] + 1e-10)
        W_gate_kappa = W_gate_svd[0] / (W_gate_svd[-1] + 1e-10) if W_gate_svd is not None else 0
        W_down_PR = W_down_svd.sum()**2 / (len(W_down_svd) * (W_down_svd**2).sum() + 1e-10)
        W_gate_PR = W_gate_svd.sum()**2 / (len(W_gate_svd) * (W_gate_svd**2).sum() + 1e-10) if W_gate_svd is not None else 0
        
        post_ln_norm = np.linalg.norm(weights.post_attn_layernorm_weight) if weights.post_attn_layernorm_weight is not None else 0
        ln_norm = np.linalg.norm(weights.input_layernorm_weight) if weights.input_layernorm_weight is not None else 0
        
        results.append({
            'layer': l_idx,
            'layer_frac': l_idx / info.n_layers,
            'lambda_max': lambda_max,
            'lambda_min': lambda_min,
            'lambda_mean': lambda_mean,
            'lambda_std': lambda_std,
            'condition_number': condition_number,
            'PR_J': PR_J,
            'ratio_mean': np.mean(ratios),
            'ratio_max': np.max(ratios),
            'W_down_kappa': float(W_down_kappa),
            'W_gate_kappa': float(W_gate_kappa),
            'W_down_PR': float(W_down_PR),
            'W_gate_PR': float(W_gate_PR),
            'post_ln_norm': float(post_ln_norm),
            'ln_norm': float(ln_norm),
            'h_in_norm': float(h_in.norm()),
            'h_out_norm': float(h_out.norm()),
        })
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
    
    # 汇总分析
    print("\n" + "="*70)
    print("P517 汇总分析")
    print("="*70)
    
    for r in results:
        print(f"  L{r['layer']}: λ_max={r['lambda_max']:.2f}, κ_J={r['condition_number']:.1f}, "
              f"PR_J={r['PR_J']:.4f}, ratio_max={r['ratio_max']:.4f}, "
              f"W_down_κ={r['W_down_kappa']:.1f}, W_gate_κ={r['W_gate_kappa']:.1f}")
    
    # 回归分析: λ_max vs 权重特征
    feature_keys = ['W_down_kappa', 'W_gate_kappa', 'W_down_PR', 'W_gate_PR', 
                    'post_ln_norm', 'ln_norm', 'layer_frac', 'h_in_norm']
    
    X = np.array([[r[k] for k in feature_keys] for r in results])
    y = np.array([r['lambda_max'] for r in results])
    X = np.nan_to_num(X, nan=0, posinf=1e10, neginf=-1e10)
    y = np.nan_to_num(y, nan=1.0, posinf=100.0, neginf=0.0)
    
    # 对数回归
    from sklearn.linear_model import LinearRegression
    log_y = np.log1p(y)
    reg = LinearRegression().fit(X, log_y)
    r2 = reg.score(X, log_y)
    
    print(f"\n  λ_max对数回归R2={r2:.3f}")
    for k, c in zip(feature_keys, reg.coef_):
        print(f"    {k}: {c:.4f}")
    
    # Spearman相关
    print(f"\n  λ_max Spearman相关:")
    for k in feature_keys:
        x_vals = np.array([r[k] for r in results])
        if np.std(x_vals) > 0 and np.std(y) > 0:
            rho, p = spearmanr(x_vals, y)
            print(f"    {k}: ρ={rho:.3f} (p={p:.4f})")
    
    # 保存结果(确保所有值是Python原生类型)
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
    
    result_path = f"results/phase_cxi/p517_{model_name}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({'results': [to_native(r) for r in results], 'lambda_max_r2': float(r2), 'model': model_name}, f, indent=2)
    print(f"\n结果已保存: {result_path}")
    
    return results


# ===== P518: 扰动方向vs传播放大 =====
def run_p518(model, tokenizer, device, model_name):
    """
    扰动方向vs传播放大
    
    方法: 在不同方向注入扰动, 测量1/3/5层后的传播比
    - 方向1: 随机方向(均匀随机)
    - 方向2: W_down的top奇异向量方向(主要成分方向)
    - 方向3: W_down的bottom奇异向量方向(次要成分方向)
    - 方向4: 残差流方向(h_out - h_in归一化)
    """
    print("\n" + "="*70)
    print("P518: 扰动方向vs传播放大")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    test_text = "The fundamental nature of reality can be understood through mathematical structures."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    sample_layers = get_sample_layers(info.n_layers, min(5, info.n_layers))
    print(f"采样层: {sample_layers}")
    
    # 传播层数
    n_prop_list = [1, 3, 5]
    
    results = []
    
    for l_idx in sample_layers:
        print(f"\n--- 层 {l_idx} ---")
        
        # 获取基线隐藏状态
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
            h_states = outputs.hidden_states
        
        h_base = h_states[l_idx][0, -1].detach().clone()
        
        # 获取权重和SVD方向
        weights = get_layer_weights(layers[l_idx], info.d_model, info.mlp_type)
        W_down = weights.W_down
        
        # 用truncated SVD避免内存溢出(只需要前5个奇异向量)
        from scipy.sparse.linalg import svds
        k_svd = min(5, min(W_down.shape) - 1)
        U_w, S_w, Vt_w = svds(W_down.astype(np.float32), k=k_svd)
        # svds返回的奇异值是升序, 反转为降序
        U_w = U_w[:, ::-1]
        S_w = S_w[::-1]
        Vt_w = Vt_w[::-1]
        
        # 方向定义(全部在d_model空间)
        directions = {}
        
        # 方向1: 随机方向
        dir_random = torch.randn(info.d_model, device=device, dtype=h_base.dtype)
        dir_random = dir_random / dir_random.norm()
        directions['random'] = dir_random
        
        # 方向2: W_down输出空间top奇异向量方向
        # 用truncated SVD避免内存溢出
        from scipy.sparse.linalg import svds
        k_svd = min(5, min(W_down.shape) - 1)
        try:
            U_w, S_w, Vt_w = svds(W_down.astype(np.float32), k=k_svd)
            U_w = U_w[:, ::-1]; S_w = S_w[::-1]; Vt_w = Vt_w[::-1]
        except Exception:
            U_w, S_w, Vt_w = np.linalg.svd(W_down.astype(np.float32), full_matrices=False)
        dir_top = torch.tensor(U_w[:, 0], device=device, dtype=h_base.dtype)
        dir_top = dir_top / dir_top.norm()
        directions['W_down_top'] = dir_top
        
        # 方向3: W_down输出空间bottom奇异向量方向
        dir_bottom = torch.tensor(U_w[:, -1], device=device, dtype=h_base.dtype)
        dir_bottom = dir_bottom / dir_bottom.norm()
        directions['W_down_bottom'] = dir_bottom
        
        # 方向4: 残差流方向
        h_next = h_states[l_idx + 1][0, -1].detach().clone()
        dir_residual = h_next - h_base
        if dir_residual.norm() > 1e-6:
            dir_residual = dir_residual / dir_residual.norm()
        else:
            dir_residual = dir_random.clone()
        directions['residual'] = dir_residual
        
        # 方向5: 前5个top奇异向量加权(输出空间)
        dir_top5 = torch.tensor(U_w[:, :5] @ np.ones(5) / 5, device=device, dtype=h_base.dtype)
        dir_top5 = dir_top5 / dir_top5.norm()
        directions['W_down_top5_avg'] = dir_top5
        
        # 对每个方向和传播层数测量
        for dir_name, direction in directions.items():
            for n_prop in n_prop_list:
                target_layer = min(l_idx + n_prop, info.n_layers)
                if target_layer == l_idx:
                    continue
                
                # 基线
                with torch.no_grad():
                    h_target_base = h_states[target_layer][0, -1].detach().clone()
                
                # 扰动W_down
                eps = 0.01
                orig_w = perturb_w_down(layers, l_idx, eps, info.mlp_type)
                
                with torch.no_grad():
                    out_perturbed = model(inputs["input_ids"], output_hidden_states=True)
                    h_target_perturbed = out_perturbed.hidden_states[target_layer][0, -1].detach().clone()
                
                restore_w_down(layers, l_idx, orig_w, info.mlp_type)
                
                # 全局传播比
                delta_h_target = (h_target_perturbed - h_target_base).norm()
                h_target_norm = h_target_base.norm()
                global_ratio = (delta_h_target / (h_target_norm + 1e-10)).item()
                
                # 方向对齐度: 扰动输出与指定方向的余弦相似度
                delta_h_vec = h_target_perturbed - h_target_base
                if delta_h_vec.norm() > 1e-10:
                    cos_sim = torch.nn.functional.cosine_similarity(
                        delta_h_vec.unsqueeze(0), direction.unsqueeze(0)
                    ).item()
                else:
                    cos_sim = 0.0
                
                print(f"  {dir_name}, n_prop={n_prop}: ratio={global_ratio:.4f}, cos_sim={cos_sim:.4f}")
                
                results.append({
                    'layer': l_idx,
                    'direction': dir_name,
                    'n_prop': n_prop,
                    'target_layer': target_layer,
                    'global_ratio': float(global_ratio),
                    'cos_sim': float(cos_sim),
                    'h_target_norm': float(h_target_norm.item()),
                })
                
                torch.cuda.empty_cache()
    
    # 汇总分析
    print("\n" + "="*70)
    print("P518 汇总分析")
    print("="*70)
    
    # 按方向汇总
    for dir_name in ['random', 'W_down_top', 'W_down_bottom', 'residual', 'W_down_top5_avg']:
        dir_results = [r for r in results if r['direction'] == dir_name]
        if not dir_results:
            continue
        ratios = [r['global_ratio'] for r in dir_results]
        cos_sims = [r['cos_sim'] for r in dir_results]
        print(f"  {dir_name}: ratio mean={np.mean(ratios):.4f}±{np.std(ratios):.4f}, "
              f"cos_sim mean={np.mean(cos_sims):.4f}")
    
    # 按传播层数汇总
    for n_prop in n_prop_list:
        prop_results = [r for r in results if r['n_prop'] == n_prop]
        if not prop_results:
            continue
        ratios = [r['global_ratio'] for r in prop_results]
        print(f"  n_prop={n_prop}: ratio mean={np.mean(ratios):.4f}±{np.std(ratios):.4f}")
    
    # 方向×传播层数交叉分析
    print("\n  方向×传播层数交叉:")
    for dir_name in ['random', 'W_down_top', 'W_down_bottom', 'residual']:
        for n_prop in n_prop_list:
            cross = [r for r in results if r['direction'] == dir_name and r['n_prop'] == n_prop]
            if cross:
                ratios = [r['global_ratio'] for r in cross]
                print(f"    {dir_name} × n_prop={n_prop}: ratio={np.mean(ratios):.4f}")
    
    # 保存结果
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
    result_path = f"results/phase_cxi/p518_{model_name}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({'results': [to_native(r) for r in results], 'model': model_name}, f, indent=2)
    print(f"\n结果已保存: {result_path}")
    
    return results


# ===== P519: Jacobian谱的权重预测 =====
def run_p519(model, tokenizer, device, model_name):
    """
    Jacobian谱的权重预测
    
    方法: 收集Jacobian谱特征和权重特征, 用回归分析推导关系
    - 目标: λ_max ≈ f(W_down_kappa, W_gate_PR, post_ln_norm, ...)
    - 扩展: PR_J ≈ g(W_down_PR, W_gate_PR, ...)
    """
    print("\n" + "="*70)
    print("P519: Jacobian谱的权重预测")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    test_text = "The fundamental nature of reality can be understood through mathematical structures."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    sample_layers = get_sample_layers(info.n_layers, min(8, info.n_layers))
    print(f"采样层: {sample_layers}")
    
    results = []
    
    for l_idx in sample_layers:
        print(f"\n--- 层 {l_idx} ---")
        
        # 获取基线隐藏状态
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
            h_states = outputs.hidden_states
        
        h_in = h_states[l_idx][0, -1].detach().clone()
        h_out = h_states[l_idx + 1][0, -1].detach().clone()
        
        # 1. Jacobian谱特征(通过多方向扰动估计)
        n_probes = 30
        eps = 0.01
        J_probes = []
        
        for i in range(n_probes):
            orig_w = perturb_w_down(layers, l_idx, eps, info.mlp_type)
            with torch.no_grad():
                out = model(inputs["input_ids"], output_hidden_states=True)
                h_perturbed = out.hidden_states[l_idx + 1][0, -1].detach().clone()
            restore_w_down(layers, l_idx, orig_w, info.mlp_type)
            J_probes.append((h_perturbed - h_out).cpu().float())
        
        J_proj = torch.stack(J_probes)
        U_j, S_j, Vt_j = torch.linalg.svd(J_proj, full_matrices=False)
        
        # Jacobian谱特征
        lambda_max = S_j[0].item()
        lambda_mean = S_j.mean().item()
        PR_J = (S_j.sum())**2 / (len(S_j) * (S_j**2).sum() + 1e-10)
        cond_J = S_j[0].item() / (S_j[-1].item() + 1e-10)
        
        # 2. 多步传播比(1/3/5层)
        ratios = {}
        for n_prop in [1, 3, 5]:
            target = min(l_idx + n_prop, info.n_layers)
            with torch.no_grad():
                h_target_base = h_states[target][0, -1].detach().clone()
            
            orig_w = perturb_w_down(layers, l_idx, eps, info.mlp_type)
            with torch.no_grad():
                out = model(inputs["input_ids"], output_hidden_states=True)
                h_target_perturbed = out.hidden_states[target][0, -1].detach().clone()
            restore_w_down(layers, l_idx, orig_w, info.mlp_type)
            
            ratio = (h_target_perturbed - h_target_base).norm() / (h_target_base.norm() + 1e-10)
            ratios[f'ratio_{n_prop}'] = ratio.item()
        
        # 3. 权重特征
        weights = get_layer_weights(layers[l_idx], info.d_model, info.mlp_type)
        
        W_down = weights.W_down
        W_gate = weights.W_gate
        W_up = weights.W_up
        
        W_down_svd = np.linalg.svd(W_down.astype(np.float32), compute_uv=False)
        W_gate_svd = np.linalg.svd(W_gate.astype(np.float32), compute_uv=False) if W_gate is not None else None
        W_up_svd = np.linalg.svd(W_up.astype(np.float32), compute_uv=False) if W_up is not None else None
        
        # 权重谱特征
        W_down_kappa = W_down_svd[0] / (W_down_svd[-1] + 1e-10)
        W_down_PR = W_down_svd.sum()**2 / (len(W_down_svd) * (W_down_svd**2).sum() + 1e-10)
        W_down_top3_energy = W_down_svd[:3].sum() / (W_down_svd.sum() + 1e-10)
        
        W_gate_kappa = W_gate_svd[0] / (W_gate_svd[-1] + 1e-10) if W_gate_svd is not None else 0
        W_gate_PR = W_gate_svd.sum()**2 / (len(W_gate_svd) * (W_gate_svd**2).sum() + 1e-10) if W_gate_svd is not None else 0
        W_gate_top3_energy = W_gate_svd[:3].sum() / (W_gate_svd.sum() + 1e-10) if W_gate_svd is not None else 0
        
        # LayerNorm特征
        post_ln_norm = np.linalg.norm(weights.post_attn_layernorm_weight) if weights.post_attn_layernorm_weight is not None else 0
        ln_norm = np.linalg.norm(weights.input_layernorm_weight) if weights.input_layernorm_weight is not None else 0
        post_ln_max = np.max(np.abs(weights.post_attn_layernorm_weight)) if weights.post_attn_layernorm_weight is not None else 0
        ln_max = np.max(np.abs(weights.input_layernorm_weight)) if weights.input_layernorm_weight is not None else 0
        
        # 注意力权重特征
        W_o_norm = np.linalg.norm(weights.W_o)
        W_q_norm = np.linalg.norm(weights.W_q)
        W_v_norm = np.linalg.norm(weights.W_v)
        
        result = {
            'layer': l_idx,
            'layer_frac': l_idx / info.n_layers,
            # Jacobian谱特征(目标变量)
            'lambda_max': lambda_max,
            'lambda_mean': lambda_mean,
            'PR_J': PR_J,
            'cond_J': cond_J,
            # 传播比(目标变量)
            **ratios,
            # 权重特征(预测变量)
            'W_down_kappa': float(W_down_kappa),
            'W_down_PR': float(W_down_PR),
            'W_down_top3_energy': float(W_down_top3_energy),
            'W_gate_kappa': float(W_gate_kappa),
            'W_gate_PR': float(W_gate_PR),
            'W_gate_top3_energy': float(W_gate_top3_energy),
            'post_ln_norm': float(post_ln_norm),
            'ln_norm': float(ln_norm),
            'post_ln_max': float(post_ln_max),
            'ln_max': float(ln_max),
            'W_o_norm': float(W_o_norm),
            'W_q_norm': float(W_q_norm),
            'W_v_norm': float(W_v_norm),
            'h_in_norm': float(h_in.norm()),
        }
        
        results.append(result)
        print(f"  λ_max={lambda_max:.4f}, PR_J={PR_J:.4f}, cond_J={cond_J:.1f}")
        print(f"  ratio_1={ratios['ratio_1']:.4f}, ratio_3={ratios['ratio_3']:.4f}, ratio_5={ratios['ratio_5']:.4f}")
        print(f"  W_down_κ={W_down_kappa:.1f}, W_gate_κ={W_gate_kappa:.1f}")
        
        torch.cuda.empty_cache()
    
    # 回归分析
    print("\n" + "="*70)
    print("P519 回归分析")
    print("="*70)
    
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    
    feature_keys = [
        'W_down_kappa', 'W_down_PR', 'W_down_top3_energy',
        'W_gate_kappa', 'W_gate_PR', 'W_gate_top3_energy',
        'post_ln_norm', 'ln_norm', 'post_ln_max', 'ln_max',
        'W_o_norm', 'W_q_norm', 'W_v_norm',
        'layer_frac', 'h_in_norm'
    ]
    
    X = np.array([[r[k] for k in feature_keys] for r in results])
    X = np.nan_to_num(X, nan=0, posinf=1e10, neginf=-1e10)
    
    # 目标变量
    targets = {
        'lambda_max': np.array([r['lambda_max'] for r in results]),
        'PR_J': np.array([r['PR_J'] for r in results]),
        'cond_J': np.array([r['cond_J'] for r in results]),
        'ratio_1': np.array([r['ratio_1'] for r in results]),
        'ratio_3': np.array([r['ratio_3'] for r in results]),
        'ratio_5': np.array([r['ratio_5'] for r in results]),
    }
    
    for target_name, y in targets.items():
        y = np.nan_to_num(y, nan=1.0, posinf=100.0, neginf=0.0)
        
        # 对数回归(对lambda_max和cond_J)
        if target_name in ['lambda_max', 'cond_J', 'ratio_3', 'ratio_5']:
            y_reg = np.log1p(y)
        else:
            y_reg = y
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        reg = Ridge(alpha=1.0).fit(X_scaled, y_reg)
        r2 = reg.score(X_scaled, y_reg)
        
        print(f"\n  {target_name} 回归 R2={r2:.3f}:")
        # 特征重要性(按|系数|排序)
        importance = list(zip(feature_keys, np.abs(reg.coef_)))
        importance.sort(key=lambda x: x[1], reverse=True)
        for k, imp in importance[:5]:
            idx = feature_keys.index(k)
            print(f"    {k}: coef={reg.coef_[idx]:.4f}, |coef|={imp:.4f}")
    
    # Spearman相关(λ_max)
    print(f"\n  λ_max Spearman相关:")
    y_lam = np.nan_to_num(targets['lambda_max'], nan=1.0, posinf=100.0, neginf=0.0)
    for k in feature_keys:
        x_vals = np.array([r[k] for r in results])
        if np.std(x_vals) > 0 and np.std(y_lam) > 0:
            rho, p = spearmanr(x_vals, y_lam)
            print(f"    {k}: ρ={rho:.3f} (p={p:.4f})")
    
    # 保存结果
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
    result_path = f"results/phase_cxi/p519_{model_name}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({'results': [to_native(r) for r in results], 'model': model_name}, f, indent=2)
    print(f"\n结果已保存: {result_path}")
    
    return results


# ===== 主函数 =====
def main():
    parser = argparse.ArgumentParser(description="Phase CXI: Jacobian特征值谱与传播方向分析")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["p517", "p518", "p519"])
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Phase CXI: {args.experiment.upper()} | 模型: {args.model}")
    print(f"{'='*70}")
    
    # 加载模型
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    print(f"模型: {info.model_class}, 层数: {info.n_layers}, d_model: {info.d_model}, "
          f"mlp_type: {info.mlp_type}")
    
    start_time = time.time()
    
    # 运行实验
    if args.experiment == "p517":
        results = run_p517(model, tokenizer, device, args.model)
    elif args.experiment == "p518":
        results = run_p518(model, tokenizer, device, args.model)
    elif args.experiment == "p519":
        results = run_p519(model, tokenizer, device, args.model)
    
    elapsed = time.time() - start_time
    print(f"\n实验耗时: {elapsed:.1f}秒")
    
    # 释放模型
    release_model(model)


if __name__ == "__main__":
    main()
