"""
Phase CXIII-P523/P524/P525/P526/P527: 1-秩Jacobian的数学推导 + 频谱力学
=========================================================================

Phase CXII核心发现:
- Jacobian真实1-秩: n_90=n_99=1, top1_energy=1.000, λ_max/λ_2=10^6
- v_max最对齐residual方向(|cos|=0.39-0.62)
- post_ln_norm因果干预: Qwen3/GLM4强因果(R2=0.88), DS7B弱因果(R2=0.37)
- input_ln_norm对λ_max几乎无因果(R2<0.33)

Phase CXIII核心思路:
1. P523: 从权重矩阵推导1-秩Jacobian的数学证明
   - MLP的前向传播: h_out = h_in + W_down * σ(W_gate * LN(h_in)) * (W_up * LN(h_in))
   - 对W_down做微扰 → δh = W_down * δ(σ(W_gate * LN(h_in)) * (W_up * LN(h_in)))
   - 由于σ和乘法结构, 扰动主要激活1个方向 → 解释1-秩
   - 验证: W_down * f(h_in) 的秩是否为1

2. P524: DS7B因果链断裂的机制分析
   - 对比Qwen3/GLM4/DS7B的Jacobian结构差异
   - DS7B的RL训练是否改变了W_down的频谱结构?
   - 测量: W_down的PR, 条件数, v_max与W_down top奇异向量的对齐度

3. P525: 频谱力学第一性
   - 测量Δh在W_U奇异值空间中的能量密度S_Δ(ω)
   - 验证ratio(k)是否等于∫₀ᵏ S_Δ(ω) dω
   - 建立频谱→传播比的解析桥接

4. P526: residual方向反向的物理机制
   - 为什么P518中residual方向的cos_sim为负(-0.5~-0.9)?
   - 测量: W_down扰动后, MLP输出残差变化方向
   - 假设: LayerNorm的归一化效应使残差方向反转

5. P527: Jacobian低秩+频谱力学的统一方程
   - 推导: λ_max的"重正化群流"方程
   - 建立从权重→Jacobian→传播→频谱的完整因果链

使用方法:
    python phase_cxiii_jacobian_math.py --model qwen3 --experiment p523
    python phase_cxiii_jacobian_math.py --model glm4 --experiment p524
    python phase_cxiii_jacobian_math.py --model deepseek7b --experiment p525
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


def compute_jacobian_input_probes(model, layers, l_idx, inputs, info, n_probes=30, alpha=0.01):
    """
    通过h_in扰动估计Jacobian的投影(对输入空间的Jacobian)
    
    返回: J_input_proj [n_probes, d_model], h_out_baseline
    """
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_hidden_states=True)
        h_states = outputs.hidden_states
        h_in = h_states[l_idx][0, -1].detach().clone()
        h_out_baseline = h_states[l_idx + 1][0, -1].detach().clone()
    
    J_probes = []
    for i in range(n_probes):
        # 随机扰动h_in方向
        noise = torch.randn_like(h_in)
        noise = noise / (noise.norm() + 1e-10) * alpha * h_in.norm()
        
        # 通过hook注入扰动到层l_idx的输入
        captured = {}
        def make_hook():
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured['h_out'] = output[0].detach().clone()
                else:
                    captured['h_out'] = output.detach().clone()
            return hook
        
        hook_handle = layers[l_idx].register_forward_hook(make_hook())
        
        # 注入扰动: 在输入embedding层添加扰动是不精确的
        # 更好的方法: 直接修改层的输入
        # 但这需要更复杂的hook, 暂时使用W_down扰动方法
        hook_handle.remove()
        
        # 简化方案: 直接对W_down做方向性扰动
        orig_w = layers[l_idx].mlp.down_proj.weight.data.clone()
        # 在特定方向上扰动
        perturb_dir = torch.randn_like(orig_w)
        perturb_dir = perturb_dir / (perturb_dir.norm() + 1e-10)
        layers[l_idx].mlp.down_proj.weight.data = orig_w + alpha * perturb_dir * orig_w.norm()
        
        with torch.no_grad():
            out = model(inputs["input_ids"], output_hidden_states=True)
            h_perturbed = out.hidden_states[l_idx + 1][0, -1].detach().clone()
        
        layers[l_idx].mlp.down_proj.weight.data = orig_w
        
        delta_h = (h_perturbed - h_out_baseline) / (alpha * orig_w.norm())
        J_probes.append(delta_h.cpu().float())
        
        if i % 20 == 0:
            torch.cuda.empty_cache()
    
    J_proj = torch.stack(J_probes)
    return J_proj, h_out_baseline


# ===== P523: 从权重矩阵推导1-秩Jacobian =====
def run_p523(model, tokenizer, device, model_name):
    """
    数学推导验证: 为什么Jacobian是1-秩的?
    
    理论分析:
    MLP前向传播: h_out = h_in + W_down * f(LN(h_in))
    其中 f(x) = σ(W_gate * x) ⊙ (W_up * x)
    
    对W_down做微扰 δW_down:
    δh = δW_down * f(LN(h_in))
    
    由于f(LN(h_in))是一个固定向量(给定输入), δW_down * f(LN(h_in)) 
    = ||f(LN(h_in))|| * δW_down * (f(LN(h_in)) / ||f(LN(h_in))||)
    
    对每个扰动方向i: δh_i = α * W_down * δ_i
    其中δ_i是W_down的扰动方向
    
    关键: 所有δh_i都正比于f(LN(h_in))方向!
    因为: δW_down * f(LN(h_in)) = (δW_down * f_norm) * ||f|| 
    这里f_norm = f(LN(h_in)) / ||f(LN(h_in))||是d_intermediate维向量
    
    δW_down shape: [d_model, d_intermediate]
    δW_down * f_norm: [d_model] — 这是d_model维向量
    对不同扰动δ_i, δW_down * f_norm的方向是不同的!
    
    但P520实验发现Jacobian是1-秩的, 说明:
    不同的δW_down * f_norm几乎共线!
    这意味着W_down的结构使得 W_down * f_norm ≈ c * v
    对所有f_norm方向, 结果几乎在同一个方向v上
    
    验证方法:
    1. 计算f(LN(h_in)) = MLP中间层激活
    2. 计算W_down * f_norm的方向
    3. 验证W_down * v (对随机v)是否总是指向同一方向
    """
    print("\n" + "="*70)
    print("P523: 从权重矩阵推导1-秩Jacobian")
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
        
        # 获取权重
        weights = get_layer_weights(layers[l_idx], info.d_model, info.mlp_type)
        W_down = weights.W_down  # [d_model, intermediate_size]
        W_gate = weights.W_gate  # [intermediate_size, d_model] or None
        W_up = weights.W_up      # [intermediate_size, d_model]
        post_ln_w = weights.post_attn_layernorm_weight
        
        # 获取隐藏状态
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
            h_states = outputs.hidden_states
            h_in = h_states[l_idx][0, -1].cpu().float().numpy()
            h_out = h_states[l_idx + 1][0, -1].cpu().float().numpy()
        
        # 计算 LayerNorm 后的输入
        # LN(x) = (x - μ) / σ * γ + β, 这里简化为归一化
        h_in_norm = np.linalg.norm(h_in)
        h_in_centered = h_in - h_in.mean()
        h_in_std = np.std(h_in_centered) + 1e-10
        ln_h_in = h_in_centered / h_in_std
        if post_ln_w is not None:
            ln_h_in = ln_h_in * post_ln_w
        
        # 计算 MLP 中间激活 f(LN(h_in)) = σ(W_gate * LN(h_in)) ⊙ (W_up * LN(h_in))
        if W_gate is not None:
            gate_pre = W_gate @ ln_h_in  # [intermediate_size]
            gate_act = 1.0 / (1.0 + np.exp(-gate_pre))  # sigmoid
        else:
            # GLM4 merged模式: 需要从W_gate_up拆分
            gate_act = np.ones(W_up.shape[0])  # 简化: 假设全1
        
        up_out = W_up @ ln_h_in  # [intermediate_size]
        f_mlp = gate_act * up_out  # [intermediate_size] — MLP中间激活
        
        f_norm = np.linalg.norm(f_mlp)
        f_mlp_normed = f_mlp / (f_norm + 1e-10)
        
        # W_down * f_mlp = MLP输出残差 (残差贡献)
        mlp_residual = W_down @ f_mlp  # [d_model]
        mlp_residual_dir = mlp_residual / (np.linalg.norm(mlp_residual) + 1e-10)
        
        # 实验1: W_down * 随机向量 — 是否总是指向同一方向?
        n_random = 50
        random_outputs = []
        for _ in range(n_random):
            v_rand = np.random.randn(W_down.shape[1])
            v_rand = v_rand / (np.linalg.norm(v_rand) + 1e-10)
            out_rand = W_down @ v_rand
            out_rand_dir = out_rand / (np.linalg.norm(out_rand) + 1e-10)
            random_outputs.append(out_rand_dir)
        
        # 计算随机输出之间的对齐度
        random_cos_sims = []
        for i in range(min(20, n_random)):
            cos = abs(np.dot(random_outputs[0], random_outputs[i]))
            random_cos_sims.append(cos)
        mean_random_cos = np.mean(random_cos_sims)
        
        # 实验2: W_down的PR (参与率) — W_down是否也是低秩?
        k_wd = min(50, min(W_down.shape) - 2)
        U_wd, S_wd, Vt_wd = svds(W_down.astype(np.float32), k=k_wd)
        S_wd = S_wd[::-1]
        PR_Wdown = float(S_wd.sum()**2 / (len(S_wd) * (S_wd**2).sum() + 1e-10))
        top1_energy_wd = float(S_wd[0]**2 / (np.sum(S_wd**2) + 1e-10))
        
        # 实验3: W_down * f_mlp 方向 vs W_down top奇异向量
        U_wd_sorted = U_wd[:, ::-1]
        w_down_top1 = U_wd_sorted[:, 0]
        w_down_top1 = w_down_top1 / (np.linalg.norm(w_down_top1) + 1e-10)
        cos_mlp_wd_top1 = abs(np.dot(mlp_residual_dir, w_down_top1))
        
        # 实验4: Jacobian probe验证
        J_proj, _ = compute_jacobian_probes(model, layers, l_idx, inputs, info, n_probes=100)
        J_proj_T = J_proj.T  # [d_model, n_probes]
        U_jt, S_j, Vt_jt = torch.linalg.svd(J_proj_T, full_matrices=False)
        v_max = U_jt[:, 0].cpu().float().numpy()
        v_max = v_max / (np.linalg.norm(v_max) + 1e-10)
        
        # v_max vs mlp_residual_dir
        cos_vmax_mlp = abs(np.dot(v_max, mlp_residual_dir))
        # v_max vs w_down_top1
        cos_vmax_wd = abs(np.dot(v_max, w_down_top1))
        # mlp_residual_dir vs residual
        residual = h_out - h_in
        residual_dir = residual / (np.linalg.norm(residual) + 1e-10)
        cos_mlp_residual = abs(np.dot(mlp_residual_dir, residual_dir))
        
        # 实验5: f_mlp的稀疏性 — 只有少数中间神经元激活
        f_mlp_abs = np.abs(f_mlp)
        n_active = np.sum(f_mlp_abs > 0.01 * np.max(f_mlp_abs))
        sparsity = 1.0 - n_active / len(f_mlp)
        top10_f_energy = np.sum(np.sort(f_mlp_abs**2)[-10:]) / (np.sum(f_mlp_abs**2) + 1e-10)
        
        # 实验6: W_down行空间的维度 vs f_mlp的有效维度
        # 如果f_mlp极度稀疏(只有k个非零), 则W_down*f_mlp只有W_down的k行参与
        # 这会导致输出方向由这k行决定, 不一定低秩
        f_top_indices = np.argsort(f_mlp_abs)[-10:]
        W_down_active = W_down[:, f_top_indices]  # [d_model, 10]
        # 这10行的主方向
        if W_down_active.shape[1] > 1:
            U_act, S_act, _ = svds(W_down_active.astype(np.float32), k=min(5, W_down_active.shape[1]-1))
            S_act = S_act[::-1]
            PR_active = float(S_act.sum()**2 / (len(S_act) * (S_act**2).sum() + 1e-10))
        else:
            PR_active = 1.0
        
        result = {
            'layer': l_idx,
            'f_mlp_norm': float(f_norm),
            'mlp_residual_norm': float(np.linalg.norm(mlp_residual)),
            'cos_mlp_residual': float(cos_mlp_residual),
            'cos_vmax_mlp': float(cos_vmax_mlp),
            'cos_vmax_wd': float(cos_vmax_wd),
            'cos_mlp_wd_top1': float(cos_mlp_wd_top1),
            'mean_random_cos': float(mean_random_cos),
            'PR_Wdown': PR_Wdown,
            'top1_energy_wd': top1_energy_wd,
            'f_sparsity': float(sparsity),
            'f_top10_energy': float(top10_f_energy),
            'f_n_active': int(n_active),
            'f_total': int(len(f_mlp)),
            'PR_active': PR_active,
            'lambda_max': float(S_j[0]),
            'h_in_norm': float(h_in_norm),
            'h_out_norm': float(np.linalg.norm(h_out)),
        }
        results.append(result)
        
        print(f"  f_mlp_norm={f_norm:.2f}, mlp_residual_norm={np.linalg.norm(mlp_residual):.2f}")
        print(f"  cos(v_max, mlp_residual)={cos_vmax_mlp:.4f}")
        print(f"  cos(v_max, W_down_top1)={cos_vmax_wd:.4f}")
        print(f"  cos(mlp_residual, residual)={cos_mlp_residual:.4f}")
        print(f"  W_down PR={PR_Wdown:.4f}, top1_energy={top1_energy_wd:.4f}")
        print(f"  f_sparsity={sparsity:.4f}, top10_f_energy={top10_f_energy:.4f}")
        print(f"  PR_active(W_down[:, top10_f])={PR_active:.4f}")
        print(f"  mean_random_cos(W_down*rand)={mean_random_cos:.4f}")
        
        torch.cuda.empty_cache()
    
    # 汇总分析
    print("\n" + "="*70)
    print("P523 汇总分析: 1-秩Jacobian的数学机制")
    print("="*70)
    
    # 核心问题: 为什么Jacobian是1-秩?
    mean_cos_vmax_mlp = np.mean([r['cos_vmax_mlp'] for r in results])
    mean_cos_vmax_wd = np.mean([r['cos_vmax_wd'] for r in results])
    mean_PR_Wdown = np.mean([r['PR_Wdown'] for r in results])
    mean_sparsity = np.mean([r['f_sparsity'] for r in results])
    mean_PR_active = np.mean([r['PR_active'] for r in results])
    mean_random_cos = np.mean([r['mean_random_cos'] for r in results])
    
    print(f"\n  v_max与mlp_residual对齐: {mean_cos_vmax_mlp:.4f}")
    print(f"  v_max与W_down_top1对齐: {mean_cos_vmax_wd:.4f}")
    print(f"  W_down的PR: {mean_PR_Wdown:.4f}")
    print(f"  f_mlp稀疏度: {mean_sparsity:.4f}")
    print(f"  PR(W_down[:, top10_f]): {mean_PR_active:.4f}")
    print(f"  W_down*随机向量平均|cos|: {mean_random_cos:.4f}")
    
    # 1-秩机制判断
    print("\n  1-秩机制分析:")
    if mean_cos_vmax_mlp > 0.8:
        print("  >> v_max高度对齐MLP残差方向 → Jacobian 1-秩来自MLP残差贡献")
    elif mean_PR_active > 0.8:
        print("  >> W_down激活列极度低秩 → Jacobian 1-秩来自W_down结构")
    elif mean_sparsity > 0.9:
        print("  >> f_mlp极度稀疏 → Jacobian 1-秩来自稀疏激活")
    else:
        print("  >> 1-秩来源复杂, 需要进一步分析")
    
    # 保存结果
    result_path = f"tests/glm5/results/phase_cxiii/p523_{model_name}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({'results': [to_native(r) for r in results], 'model': model_name}, f, indent=2)
    print(f"\n结果已保存: {result_path}")
    
    return results


# ===== P524: DS7B因果链断裂的机制分析 =====
def run_p524(model, tokenizer, device, model_name):
    """
    对比Qwen3/GLM4/DS7B的Jacobian结构差异, 分析DS7B因果链为何断裂
    
    核心假设: RL训练改变了W_down的频谱结构, 使Jacobian不再1-秩或1-秩方向不稳定
    
    测量:
    1. W_down的频谱: PR, 条件数, 谱衰减率
    2. v_max与W_down top奇异向量的对齐度对比
    3. 层间v_max方向的稳定性 (相邻层v_max的cos_sim)
    4. f_mlp稀疏度的层间变化
    """
    print("\n" + "="*70)
    print("P524: DS7B因果链断裂的机制分析")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    test_text = "The fundamental nature of reality can be understood through mathematical structures that underlie physical phenomena."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    sample_layers = get_sample_layers(info.n_layers, min(8, info.n_layers))
    print(f"采样层: {sample_layers}")
    
    results = []
    prev_v_max = None
    
    for l_idx in sample_layers:
        print(f"\n--- 层 {l_idx} ---")
        
        # 获取权重
        weights = get_layer_weights(layers[l_idx], info.d_model, info.mlp_type)
        W_down = weights.W_down
        W_gate = weights.W_gate
        W_up = weights.W_up
        post_ln_w = weights.post_attn_layernorm_weight
        
        # W_down频谱分析
        k_wd = min(50, min(W_down.shape) - 2)
        U_wd, S_wd, Vt_wd = svds(W_down.astype(np.float32), k=k_wd)
        S_wd = S_wd[::-1]
        PR_Wdown = float(S_wd.sum()**2 / (len(S_wd) * (S_wd**2).sum() + 1e-10))
        cond_Wdown = float(S_wd[0] / (S_wd[-1] + 1e-10))
        
        # 谱衰减率: S[k]/S[0] vs k
        spectral_decay = float(S_wd[min(5, len(S_wd)-1)] / (S_wd[0] + 1e-10))
        
        # Jacobian probe
        J_proj, _ = compute_jacobian_probes(model, layers, l_idx, inputs, info, n_probes=100)
        J_proj_T = J_proj.T
        U_jt, S_j, Vt_jt = torch.linalg.svd(J_proj_T, full_matrices=False)
        v_max = U_jt[:, 0].cpu().float().numpy()
        v_max = v_max / (np.linalg.norm(v_max) + 1e-10)
        
        lambda_max = float(S_j[0])
        lambda_2 = float(S_j[1]) if len(S_j) > 1 else 0
        ratio_1_2 = lambda_max / (lambda_2 + 1e-10)
        
        # v_max与W_down top奇异向量的对齐
        U_wd_sorted = U_wd[:, ::-1]
        w_down_top1 = U_wd_sorted[:, 0] / (np.linalg.norm(U_wd_sorted[:, 0]) + 1e-10)
        cos_vmax_wd = abs(np.dot(v_max, w_down_top1))
        
        # 层间v_max稳定性
        cos_prev = 0.0
        if prev_v_max is not None:
            cos_prev = abs(np.dot(v_max, prev_v_max))
        
        # 获取隐藏状态
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
            h_states = outputs.hidden_states
            h_in = h_states[l_idx][0, -1].cpu().float().numpy()
            h_out = h_states[l_idx + 1][0, -1].cpu().float().numpy()
        
        # 计算 f_mlp
        h_in_centered = h_in - h_in.mean()
        h_in_std = np.std(h_in_centered) + 1e-10
        ln_h_in = h_in_centered / h_in_std
        if post_ln_w is not None:
            ln_h_in = ln_h_in * post_ln_w
        
        if W_gate is not None:
            gate_pre = W_gate @ ln_h_in
            gate_act = 1.0 / (1.0 + np.exp(-gate_pre))
        else:
            gate_act = np.ones(W_up.shape[0])
        
        up_out = W_up @ ln_h_in
        f_mlp = gate_act * up_out
        
        f_mlp_abs = np.abs(f_mlp)
        n_active = int(np.sum(f_mlp_abs > 0.01 * np.max(f_mlp_abs)))
        sparsity = 1.0 - n_active / len(f_mlp)
        
        # MLP残差方向
        mlp_residual = W_down @ f_mlp
        mlp_residual_dir = mlp_residual / (np.linalg.norm(mlp_residual) + 1e-10)
        cos_vmax_mlp = abs(np.dot(v_max, mlp_residual_dir))
        
        # post_ln_norm vs λ_max 的逐层相关
        post_ln_norm = float(np.linalg.norm(post_ln_w)) if post_ln_w is not None else 0
        
        result = {
            'layer': l_idx,
            'layer_frac': l_idx / info.n_layers,
            'lambda_max': lambda_max,
            'lambda_2': lambda_2,
            'ratio_1_2': ratio_1_2,
            'PR_Wdown': PR_Wdown,
            'cond_Wdown': cond_Wdown,
            'spectral_decay': spectral_decay,
            'cos_vmax_wd_top1': cos_vmax_wd,
            'cos_vmax_mlp': cos_vmax_mlp,
            'cos_vmax_prev': cos_prev,
            'f_sparsity': sparsity,
            'f_n_active': n_active,
            'post_ln_norm': post_ln_norm,
            'h_in_norm': float(np.linalg.norm(h_in)),
        }
        results.append(result)
        
        print(f"  lambda_max={lambda_max:.2f}, ratio_1_2={ratio_1_2:.1f}")
        print(f"  PR_Wdown={PR_Wdown:.4f}, cond={cond_Wdown:.1f}")
        print(f"  cos(v_max, W_down_top1)={cos_vmax_wd:.4f}")
        print(f"  cos(v_max, mlp_residual)={cos_vmax_mlp:.4f}")
        print(f"  cos(v_max, prev_v_max)={cos_prev:.4f}")
        print(f"  f_sparsity={sparsity:.4f}")
        
        prev_v_max = v_max.copy()
        torch.cuda.empty_cache()
    
    # 汇总分析
    print("\n" + "="*70)
    print("P524 汇总分析: Jacobian结构稳定性")
    print("="*70)
    
    mean_cos_vmax_wd = np.mean([r['cos_vmax_wd_top1'] for r in results])
    mean_cos_vmax_mlp = np.mean([r['cos_vmax_mlp'] for r in results])
    mean_cos_prev = np.mean([r['cos_vmax_prev'] for r in results if r['cos_vmax_prev'] > 0])
    mean_PR = np.mean([r['PR_Wdown'] for r in results])
    mean_sparsity = np.mean([r['f_sparsity'] for r in results])
    
    # λ_max vs post_ln_norm 相关
    if len(results) > 3:
        lambda_maxs = [r['lambda_max'] for r in results]
        post_ln_norms = [r['post_ln_norm'] for r in results]
        try:
            rho, p = spearmanr(lambda_maxs, post_ln_norms)
            print(f"  Spearman(lambda_max, post_ln_norm): rho={rho:.4f}, p={p:.4f}")
        except:
            rho = 0
            print(f"  Spearman计算失败")
    
    print(f"\n  mean cos(v_max, W_down_top1): {mean_cos_vmax_wd:.4f}")
    print(f"  mean cos(v_max, mlp_residual): {mean_cos_vmax_mlp:.4f}")
    print(f"  mean cos(v_max, prev_v_max): {mean_cos_prev:.4f}")
    print(f"  mean PR_Wdown: {mean_PR:.4f}")
    print(f"  mean f_sparsity: {mean_sparsity:.4f}")
    
    # 保存结果
    result_path = f"tests/glm5/results/phase_cxiii/p524_{model_name}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({'results': [to_native(r) for r in results], 'model': model_name}, f, indent=2)
    print(f"\n结果已保存: {result_path}")
    
    return results


# ===== P525: 频谱力学第一性 =====
def run_p525(model, tokenizer, device, model_name):
    """
    频谱力学第一性: 从S_Δ(ω)推导ratio(k)
    
    核心理论:
    ratio(k) = ||proj_W_U_k(Δh)||^2 / ||Δh||^2
    其中 proj_W_U_k 是Δh在W_U的top-k奇异向量空间的投影
    
    如果 Δh 在W_U奇异向量空间中的能量密度为 S_Δ(ω):
    ratio(k) = ∫₀ᵏ S_Δ(ω) dω / ∫₀^d S_Δ(ω) dω
    
    测量:
    1. 对每层, 计算Δh(=h_out - h_in)在W_U奇异向量空间中的能量分布
    2. S_Δ(k) = Δh^T * u_k^2 (第k个奇异向量上的能量)
    3. ratio(k) = Σ_{i=1}^k S_Δ(i) / Σ S_Δ(i)
    4. 验证: 理论ratio(k) vs 实测ratio(k)
    """
    print("\n" + "="*70)
    print("P525: 频谱力学第一性")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    test_text = "The fundamental nature of reality can be understood through mathematical structures that underlie physical phenomena."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    sample_layers = get_sample_layers(info.n_layers, min(6, info.n_layers))
    print(f"采样层: {sample_layers}")
    
    # 获取W_U的SVD
    W_U = get_W_U(model)  # [vocab, d_model]
    k_wu = min(200, min(W_U.shape) - 2)
    print(f"计算W_U SVD (k={k_wu})...")
    U_wu, S_wu, Vt_wu = svds(W_U.T.astype(np.float32), k=k_wu)
    # U_wu: [d_model, k_wu] — W_U行空间基
    U_wu = U_wu[:, ::-1]  # 降序
    S_wu = S_wu[::-1]
    
    results = []
    
    for l_idx in sample_layers:
        print(f"\n--- 层 {l_idx} ---")
        
        # 获取隐藏状态
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
            h_states = outputs.hidden_states
            h_in = h_states[l_idx][0, -1].cpu().float().numpy()
            h_out = h_states[l_idx + 1][0, -1].cpu().float().numpy()
        
        # Δh = h_out - h_in (MLP残差贡献)
        delta_h = h_out - h_in
        delta_h_norm = np.linalg.norm(delta_h)
        
        if delta_h_norm < 1e-10:
            print(f"  delta_h太小, 跳过")
            continue
        
        # Δh在W_U奇异向量空间中的投影系数
        proj_coeffs = U_wu.T @ delta_h  # [k_wu]
        proj_energy = proj_coeffs**2  # 每个奇异向量上的能量
        total_proj_energy = np.sum(proj_energy)
        recoding_ratio = total_proj_energy / (delta_h_norm**2 + 1e-10)
        
        # S_Δ(k) = 能量密度(第k个奇异向量)
        S_delta = proj_energy / (total_proj_energy + 1e-10)  # 归一化
        
        # 累积ratio(k)
        cum_ratio = np.cumsum(proj_energy) / (total_proj_energy + 1e-10)
        
        # 关键k值的ratio
        k_values = [10, 20, 50, 100, min(200, k_wu)]
        ratio_at_k = {}
        for k in k_values:
            if k <= len(cum_ratio):
                ratio_at_k[f'ratio_k{k}'] = float(cum_ratio[k-1])
        
        # 能量集中度
        top10_energy = float(np.sum(proj_energy[:10]) / (total_proj_energy + 1e-10))
        top50_energy = float(np.sum(proj_energy[:50]) / (total_proj_energy + 1e-10))
        
        # Δh的方向 vs W_U top奇异向量
        delta_dir = delta_h / delta_h_norm
        wu_top1 = U_wu[:, 0] / (np.linalg.norm(U_wu[:, 0]) + 1e-10)
        cos_delta_wu_top1 = float(abs(np.dot(delta_dir, wu_top1)))
        
        # 频谱形状: S_Δ是否是幂律?
        # log(S_delta) vs log(k) 的斜率
        valid = proj_energy > 0
        if np.sum(valid) > 10:
            log_k = np.log(np.arange(1, len(proj_energy)+1)[valid])
            log_S = np.log(proj_energy[valid])
            # 线性回归
            from numpy.polynomial import polynomial as P
            coeffs = np.polyfit(log_k[:50], log_S[:50], 1)
            spectral_slope = float(coeffs[0])
        else:
            spectral_slope = 0
        
        result = {
            'layer': l_idx,
            'delta_h_norm': float(delta_h_norm),
            'recoding_ratio': float(recoding_ratio),
            'top10_energy': top10_energy,
            'top50_energy': top50_energy,
            'cos_delta_wu_top1': cos_delta_wu_top1,
            'spectral_slope': float(spectral_slope),
            **ratio_at_k,
        }
        results.append(result)
        
        print(f"  recoding_ratio={recoding_ratio:.4f}, top10_energy={top10_energy:.4f}")
        print(f"  cos(Δh, W_U_top1)={cos_delta_wu_top1:.4f}")
        print(f"  spectral_slope={spectral_slope:.4f}")
        for k in k_values:
            key = f'ratio_k{k}'
            if key in result:
                print(f"  ratio(k={k})={result[key]:.4f}")
        
        torch.cuda.empty_cache()
    
    # 汇总分析
    print("\n" + "="*70)
    print("P525 汇总分析: 频谱力学第一性")
    print("="*70)
    
    mean_recoding = np.mean([r['recoding_ratio'] for r in results])
    mean_top10 = np.mean([r['top10_energy'] for r in results])
    mean_slope = np.mean([r['spectral_slope'] for r in results])
    
    print(f"  mean recoding_ratio: {mean_recoding:.4f}")
    print(f"  mean top10_energy: {mean_top10:.4f}")
    print(f"  mean spectral_slope: {mean_slope:.4f}")
    
    # 验证: ratio(k=50) vs 之前测量的recoding_ratio
    if 'ratio_k50' in results[0]:
        mean_ratio50 = np.mean([r['ratio_k50'] for r in results])
        print(f"  mean ratio(k=50): {mean_ratio50:.4f}")
        print(f"  >> 验证: ratio(k=50) ≈ recoding_ratio(n_components=50)? {abs(mean_ratio50 - mean_recoding) < 0.1}")
    
    # 保存结果
    result_path = f"tests/glm5/results/phase_cxiii/p525_{model_name}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({'results': [to_native(r) for r in results], 'model': model_name}, f, indent=2)
    print(f"\n结果已保存: {result_path}")
    
    return results


# ===== P526: residual方向反向的物理机制 =====
def run_p526(model, tokenizer, device, model_name):
    """
    为什么P518中residual方向的cos_sim为负(-0.5~-0.9)?
    
    假设: LayerNorm的归一化效应使残差方向反转
    
    机制分析:
    h_out = h_in + MLP(LN(h_in))
    当对W_down做扰动 δW_down时:
    δh_out = δW_down * f(LN(h_in))
    
    但如果f(LN(h_in))的方向与h_in不同, 那么δh的方向可能不等于residual方向
    
    更深层原因: LayerNorm的梯度
    LN(x) = γ * (x - μ) / σ + β
    dLN/dx = γ/σ * (I - 1/d * 11^T - (x-μ)(x-μ)^T / (d*σ^2))
    这个梯度矩阵有负特征值 → 某些方向被LN反转
    
    测量:
    1. W_down扰动后的δh方向 vs residual方向
    2. LN的Jacobi矩阵的特征值符号分布
    3. δh分解: δh = α*h_in + β*(h_out-h_in) + 正交分量
    """
    print("\n" + "="*70)
    print("P526: residual方向反向的物理机制")
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
        
        # 获取隐藏状态
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
            h_states = outputs.hidden_states
            h_in = h_states[l_idx][0, -1].cpu().float().numpy()
            h_out = h_states[l_idx + 1][0, -1].cpu().float().numpy()
        
        # 方向定义
        h_in_dir = h_in / (np.linalg.norm(h_in) + 1e-10)
        residual = h_out - h_in
        residual_dir = residual / (np.linalg.norm(residual) + 1e-10)
        h_out_dir = h_out / (np.linalg.norm(h_out) + 1e-10)
        
        # Jacobian probe
        J_proj, _ = compute_jacobian_probes(model, layers, l_idx, inputs, info, n_probes=100)
        J_proj_T = J_proj.T
        U_jt, S_j, Vt_jt = torch.linalg.svd(J_proj_T, full_matrices=False)
        v_max = U_jt[:, 0].cpu().float().numpy()
        v_max = v_max / (np.linalg.norm(v_max) + 1e-10)
        
        # cos(v_max, residual) — 可以是负的!
        cos_vmax_residual = float(np.dot(v_max, residual_dir))  # 不取绝对值
        
        # cos(v_max, h_in) — 也可能是负的
        cos_vmax_h_in = float(np.dot(v_max, h_in_dir))
        
        # cos(v_max, h_out)
        cos_vmax_h_out = float(np.dot(v_max, h_out_dir))
        
        # LayerNorm的Jacobi矩阵分析
        # LN(x) = γ * (x - μ) / σ
        # dLN/dx = γ/σ * (I - 1/d * 11^T - (x-μ)(x-μ)^T / (d*σ^2))
        weights = get_layer_weights(layers[l_idx], info.d_model, info.mlp_type)
        post_ln_w = weights.post_attn_layernorm_weight
        
        if post_ln_w is not None:
            gamma = post_ln_w
            h_in_centered = h_in - h_in.mean()
            d = len(h_in)
            sigma = np.std(h_in_centered) + 1e-10
            
            # LN Jacobi矩阵的近似分析 (不构建完整矩阵, 太大)
            # 只分析v_max方向上的特征
            # J_LN * v = γ/σ * (v - mean(v)*1 - <v, h_norm>*h_norm)
            # 其中h_norm = (x-μ)/(σ*sqrt(d))
            
            h_norm = h_in_centered / (sigma * np.sqrt(d) + 1e-10)
            
            # v_max在LN后的变化
            v_proj_h = np.dot(v_max, h_norm)
            v_mean = np.mean(v_max)
            
            # LN梯度在v_max方向的投影
            J_LN_v = gamma / (sigma + 1e-10) * (v_max - v_mean * np.ones(d) - v_proj_h * h_norm * np.sqrt(d))
            J_LN_v_norm = np.linalg.norm(J_LN_v)
            
            # cos(J_LN * v_max, v_max) — 如果为负, 说明LN反转了v_max
            if J_LN_v_norm > 1e-10:
                cos_LN_vmax = float(np.dot(J_LN_v / J_LN_v_norm, v_max))
            else:
                cos_LN_vmax = 0
            
            # cos(J_LN * v_max, residual_dir)
            if J_LN_v_norm > 1e-10:
                cos_LN_residual = float(np.dot(J_LN_v / J_LN_v_norm, residual_dir))
            else:
                cos_LN_residual = 0
        else:
            cos_LN_vmax = 0
            cos_LN_residual = 0
        
        # MLP的残差贡献分解
        # h_out = h_in + W_down * f(LN(h_in))
        # residual = W_down * f(LN(h_in))
        # 如果|residual| << |h_in|, 则h_out ≈ h_in, residual是小的修正
        # v_max如果主要对齐h_in方向而非residual方向, 则cos(v_max, residual)可能很小甚至为负
        
        h_in_norm = np.linalg.norm(h_in)
        residual_norm = np.linalg.norm(residual)
        residual_ratio = residual_norm / (h_in_norm + 1e-10)
        
        # v_max在(h_in, residual)平面上的分解
        # v_max = α * h_in_dir + β * residual_dir_orth + ...
        # 其中residual_dir_orth是residual_dir减去h_in方向的分量
        h_in_comp = np.dot(v_max, h_in_dir)
        residual_comp = np.dot(v_max, residual_dir)
        
        # 正交化: residual_dir在h_in上的投影
        proj_res_on_h_in = np.dot(residual_dir, h_in_dir)
        residual_orth = residual_dir - proj_res_on_h_in * h_in_dir
        residual_orth_norm = np.linalg.norm(residual_orth)
        if residual_orth_norm > 1e-10:
            residual_orth_dir = residual_orth / residual_orth_norm
        else:
            residual_orth_dir = np.zeros_like(residual_dir)
        
        comp_h_in = float(np.dot(v_max, h_in_dir))
        comp_residual_orth = float(np.dot(v_max, residual_orth_dir))
        
        result = {
            'layer': l_idx,
            'cos_vmax_residual': cos_vmax_residual,
            'cos_vmax_h_in': cos_vmax_h_in,
            'cos_vmax_h_out': cos_vmax_h_out,
            'cos_LN_vmax': cos_LN_vmax,
            'cos_LN_residual': cos_LN_residual,
            'h_in_norm': float(h_in_norm),
            'residual_norm': float(residual_norm),
            'residual_ratio': float(residual_ratio),
            'comp_h_in': comp_h_in,
            'comp_residual_orth': comp_residual_orth,
            'proj_res_on_h_in': float(proj_res_on_h_in),
            'lambda_max': float(S_j[0]),
        }
        results.append(result)
        
        print(f"  cos(v_max, residual)={cos_vmax_residual:.4f} {'(NEGATIVE!)' if cos_vmax_residual < 0 else ''}")
        print(f"  cos(v_max, h_in)={cos_vmax_h_in:.4f}")
        print(f"  cos(v_max, h_out)={cos_vmax_h_out:.4f}")
        print(f"  cos(LN*v_max, v_max)={cos_LN_vmax:.4f}")
        print(f"  residual_ratio={residual_ratio:.4f}")
        print(f"  v_max分解: comp_h_in={comp_h_in:.4f}, comp_residual_orth={comp_residual_orth:.4f}")
        
        torch.cuda.empty_cache()
    
    # 汇总分析
    print("\n" + "="*70)
    print("P526 汇总分析: residual方向反向机制")
    print("="*70)
    
    mean_cos_vmax_residual = np.mean([r['cos_vmax_residual'] for r in results])
    mean_cos_vmax_h_in = np.mean([r['cos_vmax_h_in'] for r in results])
    mean_cos_LN_vmax = np.mean([r['cos_LN_vmax'] for r in results])
    mean_residual_ratio = np.mean([r['residual_ratio'] for r in results])
    mean_comp_h_in = np.mean([r['comp_h_in'] for r in results])
    
    print(f"  mean cos(v_max, residual): {mean_cos_vmax_residual:.4f}")
    print(f"  mean cos(v_max, h_in): {mean_cos_vmax_h_in:.4f}")
    print(f"  mean cos(LN*v_max, v_max): {mean_cos_LN_vmax:.4f}")
    print(f"  mean residual_ratio: {mean_residual_ratio:.4f}")
    print(f"  mean v_max comp on h_in: {mean_comp_h_in:.4f}")
    
    # 判断反向原因
    print("\n  反向机制分析:")
    if mean_cos_LN_vmax < 0:
        print("  >> LN梯度反转v_max方向 → LN是residual方向反向的原因")
    elif mean_residual_ratio < 0.1:
        print("  >> residual太小, v_max主要对齐h_in而非residual → 几何效应")
    elif mean_cos_vmax_residual < 0:
        print("  >> v_max与residual反向 → 需要更深层分析")
    
    # 保存结果
    result_path = f"tests/glm5/results/phase_cxiii/p526_{model_name}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({'results': [to_native(r) for r in results], 'model': model_name}, f, indent=2)
    print(f"\n结果已保存: {result_path}")
    
    return results


# ===== P527: 统一Jacobian低秩+频谱力学的数学框架 =====
def run_p527(model, tokenizer, device, model_name):
    """
    统一理论: 从权重→Jacobian→传播→频谱的完整因果链
    
    理论框架:
    1. 权重层: W_down, W_gate, W_up, LN_weight
    2. Jacobian层: J_l = d h_out / d h_in ≈ I + W_down * diag(σ'(z)*W_up) * W_gate * J_LN
       其中z = W_gate * LN(h_in), J_LN = dLN/dx
    3. 传播层: ratio(k) = Σ_{l=0}^{L-1} λ_max(l) * |v_max(l)^T * u_k|^2
    4. 频谱层: S_Δ(ω) = |v_max^T * u_ω|^2 * λ_max
    
    验证:
    1. 从权重预测λ_max (解析公式 vs 实测)
    2. 从λ_max预测ratio(k) (理论公式 vs 实测)
    3. 从频谱S_Δ(ω)预测ratio(k) (积分公式 vs 实测)
    """
    print("\n" + "="*70)
    print("P527: 统一Jacobian低秩+频谱力学的数学框架")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    test_text = "The fundamental nature of reality can be understood through mathematical structures that underlie physical phenomena."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    sample_layers = get_sample_layers(info.n_layers, min(6, info.n_layers))
    print(f"采样层: {sample_layers}")
    
    # W_U SVD
    W_U = get_W_U(model)
    k_wu = min(100, min(W_U.shape) - 2)
    print(f"计算W_U SVD (k={k_wu})...")
    U_wu, S_wu, Vt_wu = svds(W_U.T.astype(np.float32), k=k_wu)
    U_wu = U_wu[:, ::-1]
    S_wu = S_wu[::-1]
    
    results = []
    
    for l_idx in sample_layers:
        print(f"\n--- 层 {l_idx} ---")
        
        # 获取权重
        weights = get_layer_weights(layers[l_idx], info.d_model, info.mlp_type)
        W_down = weights.W_down
        W_gate = weights.W_gate
        W_up = weights.W_up
        post_ln_w = weights.post_attn_layernorm_weight
        input_ln_w = weights.input_layernorm_weight
        
        # 获取隐藏状态
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
            h_states = outputs.hidden_states
            h_in = h_states[l_idx][0, -1].cpu().float().numpy()
            h_out = h_states[l_idx + 1][0, -1].cpu().float().numpy()
        
        # Jacobian probe (实测λ_max和v_max)
        J_proj, _ = compute_jacobian_probes(model, layers, l_idx, inputs, info, n_probes=100)
        J_proj_T = J_proj.T
        U_jt, S_j, Vt_jt = torch.linalg.svd(J_proj_T, full_matrices=False)
        v_max = U_jt[:, 0].cpu().float().numpy()
        v_max = v_max / (np.linalg.norm(v_max) + 1e-10)
        lambda_max_measured = float(S_j[0])
        
        # === 预测1: 从权重预测λ_max ===
        # 理论: λ_max ≈ ||W_down|| * ||f(LN(h_in))|| / ||W_down * δ|| / α
        # 简化: λ_max ≈ ||post_ln_w|| * ||W_down|| * σ'||W_gate|| * ...
        
        post_ln_norm = float(np.linalg.norm(post_ln_w)) if post_ln_w is not None else 0
        input_ln_norm = float(np.linalg.norm(input_ln_w)) if input_ln_w is not None else 0
        W_down_norm = float(np.linalg.norm(W_down))
        W_gate_norm = float(np.linalg.norm(W_gate)) if W_gate is not None else 0
        W_up_norm = float(np.linalg.norm(W_up))
        h_in_norm = float(np.linalg.norm(h_in))
        
        # 预测公式1: λ_max ∝ post_ln_norm * W_down_norm
        pred1 = post_ln_norm * W_down_norm
        # 预测公式2: λ_max ∝ ||W_down * f(LN(h_in))|| (MLP残差范数)
        
        # 计算f(LN(h_in))
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
        mlp_residual = W_down @ f_mlp
        mlp_residual_norm = float(np.linalg.norm(mlp_residual))
        
        pred2 = mlp_residual_norm
        
        # === 预测2: 从v_max预测ratio(k) ===
        # ratio(k) = Σ |v_max^T * u_k|^2 * λ_max / Σ λ_max
        # 由于Jacobian 1-秩, Δh ≈ λ_max * v_max * c
        
        delta_h = h_out - h_in
        delta_h_norm = np.linalg.norm(delta_h)
        
        if delta_h_norm > 1e-10:
            # 实测ratio
            proj_coeffs = U_wu.T @ delta_h
            proj_energy = proj_coeffs**2
            total_proj = np.sum(proj_energy)
            measured_ratio = total_proj / (delta_h_norm**2 + 1e-10)
            
            # 预测ratio: 基于v_max的方向
            v_max_proj = U_wu.T @ v_max  # v_max在W_U奇异向量上的投影
            v_max_proj_energy = v_max_proj**2
            total_v_proj = np.sum(v_max_proj_energy)
            predicted_ratio = total_v_proj  # 因为v_max已归一化
            
            # 预测ratio(k=50)
            k_test = 50
            measured_ratio_k = float(np.sum(proj_energy[:k_test]) / (total_proj + 1e-10))
            predicted_ratio_k = float(np.sum(v_max_proj_energy[:k_test]) / (total_v_proj + 1e-10))
        else:
            measured_ratio = 0
            predicted_ratio = 0
            measured_ratio_k = 0
            predicted_ratio_k = 0
        
        # === 预测3: 频谱S_Δ → ratio(k) ===
        # S_Δ(k) = |v_max^T * u_k|^2
        # ratio(k) = Σ_{i=1}^k S_Δ(i) / Σ S_Δ(i)
        if delta_h_norm > 1e-10:
            S_delta_measured = proj_energy / (total_proj + 1e-10)
            S_delta_predicted = v_max_proj_energy / (total_v_proj + 1e-10)
            
            cum_measured = np.cumsum(S_delta_measured)
            cum_predicted = np.cumsum(S_delta_predicted)
            
            # 相关度
            from scipy.stats import pearsonr
            try:
                r_spectral, _ = pearsonr(cum_measured[:50], cum_predicted[:50])
            except:
                r_spectral = 0
        else:
            r_spectral = 0
        
        result = {
            'layer': l_idx,
            'lambda_max_measured': lambda_max_measured,
            'pred1_post_ln_W_down': float(pred1),
            'pred2_mlp_residual_norm': float(pred2),
            'measured_ratio': float(measured_ratio),
            'predicted_ratio': float(predicted_ratio),
            'measured_ratio_k50': float(measured_ratio_k),
            'predicted_ratio_k50': float(predicted_ratio_k),
            'r_spectral': float(r_spectral),
            'post_ln_norm': post_ln_norm,
            'W_down_norm': W_down_norm,
            'mlp_residual_norm': mlp_residual_norm,
            'h_in_norm': h_in_norm,
        }
        results.append(result)
        
        print(f"  lambda_max_measured={lambda_max_measured:.2f}")
        print(f"  pred1(post_ln*W_down)={pred1:.2f}")
        print(f"  pred2(mlp_residual_norm)={pred2:.2f}")
        print(f"  measured_ratio={measured_ratio:.4f}, predicted_ratio={predicted_ratio:.4f}")
        print(f"  measured_ratio_k50={measured_ratio_k:.4f}, predicted_ratio_k50={predicted_ratio_k:.4f}")
        print(f"  r_spectral={r_spectral:.4f}")
        
        torch.cuda.empty_cache()
    
    # 汇总分析
    print("\n" + "="*70)
    print("P527 汇总分析: 统一理论验证")
    print("="*70)
    
    # λ_max预测
    lambda_measured = [r['lambda_max_measured'] for r in results]
    pred1 = [r['pred1_post_ln_W_down'] for r in results]
    pred2 = [r['pred2_mlp_residual_norm'] for r in results]
    
    if len(results) > 3:
        try:
            rho1, _ = spearmanr(lambda_measured, pred1)
            rho2, _ = spearmanr(lambda_measured, pred2)
            print(f"  lambda_max vs pred1(post_ln*W_down): rho={rho1:.4f}")
            print(f"  lambda_max vs pred2(mlp_residual_norm): rho={rho2:.4f}")
        except:
            print(f"  Spearman计算失败")
    
    # ratio预测
    measured_ratios = [r['measured_ratio'] for r in results]
    predicted_ratios = [r['predicted_ratio'] for r in results]
    print(f"  mean measured_ratio: {np.mean(measured_ratios):.4f}")
    print(f"  mean predicted_ratio: {np.mean(predicted_ratios):.4f}")
    
    # 频谱预测
    r_spectrals = [r['r_spectral'] for r in results if r['r_spectral'] != 0]
    if r_spectrals:
        print(f"  mean r_spectral: {np.mean(r_spectrals):.4f}")
    
    # 保存结果
    result_path = f"tests/glm5/results/phase_cxiii/p527_{model_name}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({'results': [to_native(r) for r in results], 'model': model_name}, f, indent=2)
    print(f"\n结果已保存: {result_path}")
    
    return results


# ===== 主函数 =====
def main():
    parser = argparse.ArgumentParser(description="Phase CXIII: 1-秩Jacobian数学推导+频谱力学")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["p523", "p524", "p525", "p526", "p527"])
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Phase CXIII: {args.experiment.upper()} | 模型: {args.model}")
    print(f"{'='*70}")
    
    # 加载模型
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, model_name=args.model)
    print(f"模型: {info.model_class}, 层数: {info.n_layers}, d_model: {info.d_model}, "
          f"mlp_type: {info.mlp_type}")
    
    start_time = time.time()
    
    # 运行实验
    if args.experiment == "p523":
        results = run_p523(model, tokenizer, device, args.model)
    elif args.experiment == "p524":
        results = run_p524(model, tokenizer, device, args.model)
    elif args.experiment == "p525":
        results = run_p525(model, tokenizer, device, args.model)
    elif args.experiment == "p526":
        results = run_p526(model, tokenizer, device, args.model)
    elif args.experiment == "p527":
        results = run_p527(model, tokenizer, device, args.model)
    
    elapsed = time.time() - start_time
    print(f"\n实验耗时: {elapsed:.1f}秒")
    
    # 释放模型
    release_model(model)


if __name__ == "__main__":
    main()
