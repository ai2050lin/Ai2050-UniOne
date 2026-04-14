"""
Phase XCIV-P456/457/458/459: 从频谱力学到训练动力学
======================================================================

核心目标: 理解alpha(频谱耦合参数)的物理意义,建立频谱策略的第一性原理

P456: alpha与架构参数的关联 (★★★★★)
  - Phase XCIII发现三模型alpha差异巨大: Qwen3=0.46, GLM4=3.09, DS7B=-2.25
  - 本实验: alpha与d_model, n_layers, n_heads, d_ff的关系
  - 方法: 
    a) 逐层alpha分析: alpha在每层是否相同? 
    b) alpha与权重矩阵频谱的关系: W_q/W_k/W_v/W_o/W_up/W_down的奇异值谱
    c) alpha与层深度的关系: 浅层vs深层alpha是否不同?
  - 预期: alpha可能由权重矩阵的奇异值谱决定

P457: W_down vs W_U双空间结构化分析
  - P453发现逃逸方向与W_down对齐(7-11x基线)
  - 本实验: 
    a) W_down的完整奇异值谱 vs W_U的奇异值谱
    b) W_down行空间 vs W_U行空间的重叠度(主角度分析)
    c) 双空间的正交分解: V_model = V_WU + V_Wdown + V_residual
    d) 信号在三个子空间中的能量分布
  - 核心问题: W_down和W_U是正交的还是重叠的?

P458: 频谱策略与模型能力关系
  - alpha是"语言编码策略参数", 但它和模型能力有什么关系?
  - 本实验:
    a) 不同层alpha的变化: 浅层(特征提取)vs深层(语义整合)
    b) alpha与层数的关系: 是否有统一的深度-频谱定律?
    c) 幂律vs指数饱和的物理机制差异
  - 方法: 对每层单独计算alpha, 观察层间变化

P459: 从权重统计推导alpha的第一性原理尝试
  - 如果alpha可以从权重矩阵的统计性质推导, 就完成了"语言编码的第一性原理"
  - 本实验:
    a) 各层权重矩阵的奇异值分布(完整谱)
    b) W_U行空间的基在各层权重矩阵中的"表示度"
    c) 从权重频谱推导alpha的理论公式
    d) 与实测alpha对比, 验证理论预测

实验模型: qwen3 -> glm4 -> deepseek7b (串行, 避免GPU溢出)
"""

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.linalg import svds
from scipy.linalg import subspace_angles

# 添加项目路径
_project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, _project_root)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_utils
from model_utils import (
    load_model, get_layers, get_layer_weights, get_model_info,
    release_model, get_W_U, MODEL_CONFIGS, LayerWeights
)

# 输出目录
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_result(model_name, experiment, data):
    """保存实验结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    fname = f"phase_xciv_{experiment}_{model_name}_{timestamp}.json"
    fpath = OUTPUT_DIR / fname
    
    # 转换numpy类型
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(convert(data), f, ensure_ascii=False, indent=2)
    print(f"  结果已保存: {fpath}")
    return str(fpath)


# ============================================================
# P456: alpha与架构参数的关联 - 逐层alpha分析
# ============================================================
def run_p456(model_name, model, tokenizer, device):
    """
    P456: 逐层alpha分析
    - 对每层单独计算alpha(信号频谱与W_U频谱的耦合参数)
    - 分析alpha与层深度的关系
    - 分析alpha与权重矩阵频谱的关系
    """
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    mlp_type = MODEL_CONFIGS[model_name]['mlp_type']
    
    print(f"\n  P456: alpha与架构参数关联 - {model_name}")
    print(f"  d_model={d_model}, n_layers={n_layers}")
    
    # Step 1: 预计算W_U的SVD (k=800, 平衡精度和内存)
    k_svd = min(800, d_model - 1)
    W_U = get_W_U(model)
    W_U_T = W_U.T.astype(np.float32)
    
    print(f"  计算W_U SVD (k={k_svd})...")
    U_wut, s_wut, Vt_wut = svds(W_U_T, k=k_svd)
    sort_idx = np.argsort(s_wut)[::-1]
    s_wut = s_wut[sort_idx]
    U_wut = U_wut[:, sort_idx]
    
    # Step 2: 生成属性干预信号
    attr_pairs = [
        ("red", "the"), ("blue", "the"), ("big", "the"), ("small", "the"),
        ("hot", "the"), ("cold", "the"), ("good", "the"), ("bad", "the")
    ]
    
    # 获取属性向量
    attr_vectors = {}
    for attr, baseline in attr_pairs:
        attr_id = tokenizer.encode(attr, add_special_tokens=False)
        base_id = tokenizer.encode(baseline, add_special_tokens=False)
        if attr_id and base_id:
            v_attr = W_U[attr_id[0]] - W_U[base_id[0]]
            attr_vectors[attr] = v_attr / (np.linalg.norm(v_attr) + 1e-10)
    
    print(f"  有效属性: {list(attr_vectors.keys())}")
    
    # Step 3: 前向传播获取各层信号
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    
    all_alpha_per_layer = {}
    all_sv_correlation = {}
    
    # 对每个属性, 逐层计算alpha
    for attr_name, v_attr in attr_vectors.items():
        print(f"\n  属性: {attr_name}")
        
        # 构造干预输入
        base_text = "The thing is"
        base_ids = tokenizer.encode(base_text, return_tensors="pt").to(device)
        
        # 获取base和intervened的隐藏状态
        with torch.no_grad():
            base_out = model(base_ids, output_hidden_states=True)
            base_hs = [hs[0, -1].cpu().float().numpy() for hs in base_out.hidden_states]
        
        # 在embedding层注入属性
        with torch.no_grad():
            embed_layer = model.get_input_embeddings()
            embed_base = embed_layer(base_ids).detach()
            
            # 注入到最后一个token的embedding
            delta_v_p456 = torch.tensor(v_attr, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            embed_interv = embed_base.clone()
            embed_interv[:, -1, :] += delta_v_p456[0, 0, :] * 0.5
            
            interv_out = model(inputs_embeds=embed_interv, output_hidden_states=True)
            interv_hs = [hs[0, -1].cpu().float().numpy() for hs in interv_out.hidden_states]
        
        # Step 4: 逐层计算alpha
        alpha_values = []
        sv_corr_values = []
        
        for li, layer_idx in enumerate(sample_layers):
            if layer_idx >= len(interv_hs):
                continue
            
            delta_h = interv_hs[layer_idx] - base_hs[layer_idx]
            delta_norm = np.linalg.norm(delta_h)
            if delta_norm < 1e-8:
                alpha_values.append(None)
                sv_corr_values.append(None)
                continue
            
            # 计算每个SV方向上的投影能量
            proj_coeffs = U_wut.T @ delta_h  # [k_svd]
            e_i = proj_coeffs ** 2  # 投影能量
            e_i = e_i / np.sum(e_i)  # 归一化
            
            # s_i也归一化
            s_i = s_wut / np.sum(s_wut)
            
            # 在对数空间拟合: log(e_i) = alpha * log(s_i) + beta
            valid = (e_i > 1e-15) & (s_i > 1e-15)
            if valid.sum() < 10:
                alpha_values.append(None)
                sv_corr_values.append(None)
                continue
            
            log_e = np.log10(e_i[valid])
            log_s = np.log10(s_i[valid])
            
            # 线性拟合
            if len(log_s) > 1:
                coeffs = np.polyfit(log_s, log_e, 1)
                alpha = coeffs[0]
                corr = np.corrcoef(log_s, log_e)[0, 1]
            else:
                alpha = 0.0
                corr = 0.0
            
            alpha_values.append(alpha)
            sv_corr_values.append(corr)
        
        all_alpha_per_layer[attr_name] = alpha_values
        all_sv_correlation[attr_name] = sv_corr_values
    
    # Step 5: 分析alpha与层深度的关系
    mean_alpha = []
    for i in range(len(sample_layers)):
        vals = [all_alpha_per_layer[a][i] for a in attr_vectors.keys() 
                if all_alpha_per_layer[a][i] is not None]
        mean_alpha.append(np.mean(vals) if vals else None)
    
    mean_corr = []
    for i in range(len(sample_layers)):
        vals = [all_sv_correlation[a][i] for a in attr_vectors.keys() 
                if all_sv_correlation[a][i] is not None]
        mean_corr.append(np.mean(vals) if vals else None)
    
    print(f"\n  逐层alpha分析:")
    print(f"  {'层':>6} {'alpha':>8} {'corr_r':>8}")
    for i, li in enumerate(sample_layers):
        if mean_alpha[i] is not None:
            print(f"  {li:>6} {mean_alpha[i]:>8.3f} {mean_corr[i]:>8.3f}")
    
    # Step 6: alpha与权重矩阵频谱的关系
    print(f"\n  alpha与权重矩阵频谱的关系:")
    
    # 取中间层的权重矩阵
    mid_layer_idx = n_layers // 2
    mid_layer = get_layers(model)[mid_layer_idx]
    lw = get_layer_weights(mid_layer, d_model, mlp_type)
    
    weight_matrices = {
        'W_q': np.asarray(lw.W_q).astype(np.float32),
        'W_k': np.asarray(lw.W_k).astype(np.float32),
        'W_v': np.asarray(lw.W_v).astype(np.float32),
        'W_o': np.asarray(lw.W_o).astype(np.float32),
        'W_up': np.asarray(lw.W_up).astype(np.float32),
        'W_down': np.asarray(lw.W_down).astype(np.float32),
    }
    
    weight_alpha = {}
    for name, W in weight_matrices.items():
        # 计算W的奇异值
        if min(W.shape) > 800:
            _, s_W, _ = svds(W.astype(np.float64), k=800)
        else:
            s_W = np.linalg.svd(W.astype(np.float64), compute_uv=False)
        s_W = np.sort(s_W)[::-1]
        
        # alpha_W = (奇异值谱的斜率) - 用对数空间的斜率表示
        valid = s_W > 1e-10
        if valid.sum() > 10:
            log_s = np.log10(s_W[valid])
            # 用前半和后半的比值估计斜率
            n_half = len(log_s) // 2
            front_mean = np.mean(log_s[:n_half])
            back_mean = np.mean(log_s[n_half:])
            slope = front_mean - back_mean  # 正值=快速衰减, 负值=慢速衰减
            weight_alpha[name] = slope
            print(f"    {name}: spectral_slope={slope:.3f}, SV范围=[{s_W[0]:.2f}, {s_W[-1]:.4f}]")
    
    # Step 7: 分析alpha与权重spectral_slope的关系
    if mean_alpha and any(v is not None for v in mean_alpha):
        mid_alpha = [v for v in mean_alpha if v is not None]
        if mid_alpha:
            mean_alpha_mid = np.mean(mid_alpha)
            print(f"\n  中间层alpha={mean_alpha_mid:.3f}")
            print(f"  权重spectral_slope:")
            for name, slope in weight_alpha.items():
                print(f"    {name}: {slope:.3f}, ratio(alpha/slope)={mean_alpha_mid/max(abs(slope), 0.001):.2f}")
    
    result = {
        'experiment': 'P456',
        'model': model_name,
        'd_model': d_model,
        'n_layers': n_layers,
        'sample_layers': sample_layers,
        'mean_alpha_per_layer': mean_alpha,
        'mean_corr_per_layer': mean_corr,
        'weight_spectral_slopes': weight_alpha,
        'alpha_per_attr': {k: [v if v is not None else 'None' for v in vals] 
                          for k, vals in all_alpha_per_layer.items()},
    }
    
    save_result(model_name, 'p456', result)
    return result


# ============================================================
# P457: W_down vs W_U双空间结构化分析
# ============================================================
def run_p457(model_name, model, tokenizer, device):
    """
    P457: W_down vs W_U双空间结构化分析
    - W_down的奇异值谱 vs W_U的奇异值谱
    - 主角度分析(principal angles)
    - V_model的三空间分解
    - 信号在三空间中的能量分布
    """
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    mlp_type = MODEL_CONFIGS[model_name]['mlp_type']
    
    print(f"\n  P457: W_down vs W_U双空间分析 - {model_name}")
    
    # Step 1: W_U的SVD
    k_svd = min(800, d_model - 1)
    W_U = get_W_U(model)
    W_U_T = W_U.T.astype(np.float32)
    
    print(f"  计算W_U SVD (k={k_svd})...")
    U_wut, s_wut, _ = svds(W_U_T, k=k_svd)
    sort_idx = np.argsort(s_wut)[::-1]
    s_wut = s_wut[sort_idx]
    U_wut = U_wut[:, sort_idx]
    
    # Step 2: 各层W_down的SVD和主角度分析
    sample_layers = list(range(0, n_layers, max(1, n_layers // 6)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    
    results_per_layer = []
    
    for layer_idx in sample_layers:
        layer = get_layers(model)[layer_idx]
        lw = get_layer_weights(layer, d_model, mlp_type)
        W_down = np.asarray(lw.W_down).astype(np.float32)
        
        # W_down的SVD
        k_down = min(k_svd, min(W_down.shape) - 1)
        U_down, s_down, _ = svds(W_down.astype(np.float64), k=k_down)
        sort_idx_d = np.argsort(s_down)[::-1]
        s_down = s_down[sort_idx_d]
        U_down = U_down[:, sort_idx_d]
        
        # 主角度分析: W_U行空间 vs W_down行空间
        # 取前k个基向量
        k_compare = min(k_svd, k_down, 300)
        U1 = U_wut[:, :k_compare]
        U2 = U_down[:, :k_compare]
        
        # 正交化(确保是正交基)
        U1, _ = np.linalg.qr(U1)
        U2, _ = np.linalg.qr(U2)
        
        # 计算主角度
        # principal angles = arccos(svd(U1^T @ U2))
        M = U1.T @ U2
        cos_angles = np.linalg.svd(M, compute_uv=False)
        cos_angles = np.clip(cos_angles, 0, 1)
        principal_angles = np.arccos(cos_angles) * 180 / np.pi  # 度数
        
        # 平均主角度
        mean_angle = np.mean(principal_angles[:50])  # 前50个基的角度
        min_angle = np.min(principal_angles[:50])
        max_cos = np.max(cos_angles)
        
        # 子空间重叠度 = ||U1^T @ U2||_F^2 / k
        overlap = np.sum(M ** 2) / k_compare
        
        # Step 3: 三空间分解
        # V_WU = span(U_wut的前k_compare个)
        # V_Wdown = span(U_down的前k_compare个)
        # V_residual = 正交补
        
        # 计算W_U和W_down的联合基
        joint_basis = np.hstack([U_wut[:, :k_compare], U_down[:, :k_compare]])
        Q_joint, R_joint = np.linalg.qr(joint_basis)
        joint_rank = np.sum(np.abs(np.diag(R_joint)) > 1e-8)
        
        # V_WU_only: 在U_wut中但不在U_down中的成分
        proj_to_down = U_down[:, :k_compare] @ (U_down[:, :k_compare].T @ U_wut[:, :k_compare])
        WU_only = U_wut[:, :k_compare] - proj_to_down
        # 正交化
        WU_only_norm = np.sum(WU_only ** 2, axis=0)
        n_WU_only = np.sum(WU_only_norm > 1e-8)
        
        # V_Wdown_only: 在U_down中但不在U_wut中的成分
        proj_to_wu = U_wut[:, :k_compare] @ (U_wut[:, :k_compare].T @ U_down[:, :k_compare])
        Wdown_only = U_down[:, :k_compare] - proj_to_wu
        Wdown_only_norm = np.sum(Wdown_only ** 2, axis=0)
        n_Wdown_only = np.sum(Wdown_only_norm > 1e-8)
        
        layer_result = {
            'layer': layer_idx,
            's_down_top10': s_down[:10].tolist(),
            's_down_range': [float(s_down[0]), float(s_down[-1])],
            'mean_principal_angle_deg': float(mean_angle),
            'min_principal_angle_deg': float(min_angle),
            'max_cos_similarity': float(max_cos),
            'subspace_overlap': float(overlap),
            'joint_rank': int(joint_rank),
            'n_WU_only_directions': int(n_WU_only),
            'n_Wdown_only_directions': int(n_Wdown_only),
        }
        results_per_layer.append(layer_result)
        
        print(f"  Layer {layer_idx}:")
        print(f"    W_down SV范围: [{s_down[0]:.2f}, {s_down[-1]:.4f}]")
        print(f"    主角度(平均): {mean_angle:.1f}deg, 最小: {min_angle:.1f}deg")
        print(f"    最大余弦相似度: {max_cos:.4f}")
        print(f"    子空间重叠度: {overlap:.4f}")
        print(f"    联合秩: {joint_rank}, WU独有: {n_WU_only}, Wdown独有: {n_Wdown_only}")
    
    # Step 4: 信号在三空间中的能量分布
    print(f"\n  信号在三空间中的能量分布:")
    
    attr_pairs = [("red", "the"), ("blue", "the"), ("big", "the"), ("small", "the")]
    attr_vectors = {}
    for attr, baseline in attr_pairs:
        attr_id = tokenizer.encode(attr, add_special_tokens=False)
        base_id = tokenizer.encode(baseline, add_special_tokens=False)
        if attr_id and base_id:
            v_attr = W_U[attr_id[0]] - W_U[base_id[0]]
            attr_vectors[attr] = v_attr / (np.linalg.norm(v_attr) + 1e-10)
    
    for layer_idx in [n_layers // 4, n_layers // 2, 3 * n_layers // 4]:
        if layer_idx >= n_layers:
            continue
        
        layer = get_layers(model)[layer_idx]
        lw = get_layer_weights(layer, d_model, mlp_type)
        W_down = np.asarray(lw.W_down).astype(np.float32)
        
        # W_down基
        k_d = min(200, min(W_down.shape) - 1)
        U_d, s_d, _ = svds(W_down.astype(np.float64), k=k_d)
        sort_d = np.argsort(s_d)[::-1]
        U_d = U_d[:, sort_d]
        U_d, _ = np.linalg.qr(U_d)
        
        # 取W_U基
        k_wu = min(300, d_model - 1)
        U_wu, s_wu, _ = svds(W_U_T, k=k_wu)
        sort_wu = np.argsort(s_wu)[::-1]
        U_wu = U_wu[:, sort_wu]
        U_wu, _ = np.linalg.qr(U_wu)
        
        # 正交化W_down相对于W_U
        proj = U_wu @ (U_wu.T @ U_d)
        U_d_orth = U_d - proj
        U_d_orth, R = np.linalg.qr(U_d_orth)
        # 去掉零向量
        norms = np.linalg.norm(U_d_orth, axis=0)
        valid = norms > 1e-8
        if valid.sum() > 0:
            U_d_orth = U_d_orth[:, valid]
        
        for attr_name, v_attr in list(attr_vectors.items())[:2]:  # 只取2个属性
            # 前向传播
            base_text = "The thing is"
            base_ids = tokenizer.encode(base_text, return_tensors="pt").to(device)
            
            with torch.no_grad():
                base_out = model(base_ids, output_hidden_states=True)
                base_h = base_out.hidden_states[layer_idx + 1][0, -1].cpu().float().numpy()
                
                embed_layer = model.get_input_embeddings()
                embed_base = embed_layer(base_ids).detach()
                delta_v = torch.tensor(v_attr, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                embed_interv = embed_base.clone()
                embed_interv[:, -1, :] += delta_v[0, 0, :] * 0.5
                
                interv_out = model(inputs_embeds=embed_interv, output_hidden_states=True)
                interv_h = interv_out.hidden_states[layer_idx + 1][0, -1].cpu().float().numpy()
            
            delta_h = interv_h - base_h
            delta_norm_sq = np.sum(delta_h ** 2)
            
            # 在W_U空间中的能量
            proj_wu = U_wu @ (U_wu.T @ delta_h)
            e_wu = np.sum(proj_wu ** 2) / delta_norm_sq
            
            # 在W_down正交空间中的能量
            if U_d_orth.shape[1] > 0:
                proj_d_orth = U_d_orth @ (U_d_orth.T @ delta_h)
                e_d_orth = np.sum(proj_d_orth ** 2) / delta_norm_sq
            else:
                e_d_orth = 0.0
            
            # 残差能量
            e_residual = 1.0 - e_wu - e_d_orth
            
            print(f"    Layer {layer_idx}, attr={attr_name}: W_U={e_wu:.3f}, W_down_orth={e_d_orth:.3f}, residual={e_residual:.3f}")
    
    result = {
        'experiment': 'P457',
        'model': model_name,
        'd_model': d_model,
        'n_layers': n_layers,
        'per_layer': results_per_layer,
    }
    
    save_result(model_name, 'p457', result)
    return result


# ============================================================
# P458: 频谱策略与模型能力关系
# ============================================================
def run_p458(model_name, model, tokenizer, device):
    """
    P458: 频谱策略与模型能力关系
    - 逐层alpha的变化趋势
    - alpha与层深度的拟合
    - 浅层vs深层频谱策略的差异
    """
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    mlp_type = MODEL_CONFIGS[model_name]['mlp_type']
    
    print(f"\n  P458: 频谱策略与模型能力 - {model_name}")
    
    # 复用P456的数据, 但更详细分析
    k_svd = min(800, d_model - 1)
    W_U = get_W_U(model)
    W_U_T = W_U.T.astype(np.float32)
    
    print(f"  计算W_U SVD (k={k_svd})...")
    U_wut, s_wut, _ = svds(W_U_T, k=k_svd)
    sort_idx = np.argsort(s_wut)[::-1]
    s_wut = s_wut[sort_idx]
    U_wut = U_wut[:, sort_idx]
    
    # 更密的层采样
    sample_layers = list(range(0, n_layers, max(1, n_layers // 12)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    
    attr_pairs = [("red", "the"), ("blue", "the"), ("big", "the"), ("small", "the"),
                  ("hot", "the"), ("cold", "the")]
    attr_vectors = {}
    for attr, baseline in attr_pairs:
        attr_id = tokenizer.encode(attr, add_special_tokens=False)
        base_id = tokenizer.encode(baseline, add_special_tokens=False)
        if attr_id and base_id:
            v_attr = W_U[attr_id[0]] - W_U[base_id[0]]
            attr_vectors[attr] = v_attr / (np.linalg.norm(v_attr) + 1e-10)
    
    # 前向传播(一次性获取所有层)
    base_text = "The thing is"
    base_ids = tokenizer.encode(base_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        base_out = model(base_ids, output_hidden_states=True)
        base_hs = [hs[0, -1].cpu().float().numpy() for hs in base_out.hidden_states]
    
    # 对每个属性收集各层alpha
    all_layer_alphas = {li: [] for li in sample_layers}
    all_layer_ratios = {li: [] for li in sample_layers}
    all_layer_deltas = {li: [] for li in sample_layers}
    
    for attr_name, v_attr in attr_vectors.items():
        print(f"  处理属性: {attr_name}")
        
        with torch.no_grad():
            embed_layer = model.get_input_embeddings()
            embed_base = embed_layer(base_ids).detach()
            delta_v_p458 = torch.tensor(v_attr, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            embed_interv = embed_base.clone()
            embed_interv[:, -1, :] += delta_v_p458[0, 0, :] * 0.5
            
            interv_out = model(inputs_embeds=embed_interv, output_hidden_states=True)
            interv_hs = [hs[0, -1].cpu().float().numpy() for hs in interv_out.hidden_states]
        
        for li in sample_layers:
            if li >= len(interv_hs):
                continue
            
            delta_h = interv_hs[li] - base_hs[li]
            delta_norm = np.linalg.norm(delta_h)
            
            if delta_norm < 1e-8:
                continue
            
            # ratio (k=800)
            proj_coeffs = U_wut.T @ delta_h
            ratio = np.sum(proj_coeffs ** 2) / (delta_norm ** 2)
            
            # alpha
            e_i = proj_coeffs ** 2
            e_i_norm = e_i / (np.sum(e_i) + 1e-15)
            s_i_norm = s_wut / (np.sum(s_wut) + 1e-15)
            
            valid = (e_i_norm > 1e-15) & (s_i_norm > 1e-15)
            if valid.sum() > 10:
                log_e = np.log10(e_i_norm[valid])
                log_s = np.log10(s_i_norm[valid])
                alpha = np.polyfit(log_s, log_e, 1)[0]
            else:
                alpha = 0.0
            
            all_layer_alphas[li].append(alpha)
            all_layer_ratios[li].append(ratio)
            all_layer_deltas[li].append(delta_norm)
    
    # 汇总
    mean_alphas = []
    mean_ratios = []
    mean_deltas = []
    
    print(f"\n  逐层频谱参数:")
    print(f"  {'层':>6} {'alpha':>8} {'ratio':>8} {'||delta||':>10}")
    for li in sample_layers:
        alphas = all_layer_alphas[li]
        ratios = all_layer_ratios[li]
        deltas = all_layer_deltas[li]
        
        if alphas:
            ma = np.mean(alphas)
            mr = np.mean(ratios)
            md = np.mean(deltas)
            mean_alphas.append(ma)
            mean_ratios.append(mr)
            mean_deltas.append(md)
            print(f"  {li:>6} {ma:>8.3f} {mr:>8.3f} {md:>10.4f}")
        else:
            mean_alphas.append(None)
            mean_ratios.append(None)
            mean_deltas.append(None)
    
    # Step: alpha与层深度的关系拟合
    valid_layers = [(li, a, r, d) for li, a, r, d in 
                    zip(sample_layers, mean_alphas, mean_ratios, mean_deltas)
                    if a is not None]
    
    if len(valid_layers) > 3:
        layers_v = np.array([v[0] for v in valid_layers], dtype=float)
        alphas_v = np.array([v[1] for v in valid_layers])
        ratios_v = np.array([v[2] for v in valid_layers])
        deltas_v = np.array([v[3] for v in valid_layers])
        
        # alpha vs 层深度
        # 尝试线性拟合
        alpha_slope = np.polyfit(layers_v, alphas_v, 1)
        alpha_r2 = 1 - np.sum((alphas_v - np.polyval(alpha_slope, layers_v))**2) / np.sum((alphas_v - np.mean(alphas_v))**2)
        
        # ratio vs 层深度
        ratio_slope = np.polyfit(layers_v, ratios_v, 1)
        ratio_r2 = 1 - np.sum((ratios_v - np.polyval(ratio_slope, layers_v))**2) / np.sum((ratios_v - np.mean(ratios_v))**2)
        
        print(f"\n  alpha与层深度关系: alpha = {alpha_slope[0]:.4f}*L + {alpha_slope[1]:.3f}, R2={alpha_r2:.3f}")
        print(f"  ratio与层深度关系: ratio = {ratio_slope[0]:.6f}*L + {ratio_slope[1]:.3f}, R2={ratio_r2:.3f}")
        
        # 浅层vs深层
        n_half = len(layers_v) // 2
        shallow_alpha = np.mean(alphas_v[:n_half])
        deep_alpha = np.mean(alphas_v[n_half:])
        shallow_ratio = np.mean(ratios_v[:n_half])
        deep_ratio = np.mean(ratios_v[n_half:])
        
        print(f"\n  浅层(alpha={shallow_alpha:.3f}, ratio={shallow_ratio:.3f})")
        print(f"  深层(alpha={deep_alpha:.3f}, ratio={deep_ratio:.3f})")
        print(f"  深层/浅层 alpha比: {deep_alpha/max(abs(shallow_alpha), 0.001):.2f}")
        print(f"  深层/浅层 ratio比: {deep_ratio/max(abs(shallow_ratio), 0.001):.2f}")
        
        depth_analysis = {
            'alpha_slope': float(alpha_slope[0]),
            'alpha_intercept': float(alpha_slope[1]),
            'alpha_R2': float(alpha_r2),
            'ratio_slope': float(ratio_slope[0]),
            'ratio_intercept': float(ratio_slope[1]),
            'ratio_R2': float(ratio_r2),
            'shallow_alpha': float(shallow_alpha),
            'deep_alpha': float(deep_alpha),
            'shallow_ratio': float(shallow_ratio),
            'deep_ratio': float(deep_ratio),
        }
    else:
        depth_analysis = {}
    
    result = {
        'experiment': 'P458',
        'model': model_name,
        'd_model': d_model,
        'n_layers': n_layers,
        'sample_layers': sample_layers,
        'mean_alpha_per_layer': mean_alphas,
        'mean_ratio_per_layer': mean_ratios,
        'mean_delta_per_layer': mean_deltas,
        'depth_analysis': depth_analysis,
    }
    
    save_result(model_name, 'p458', result)
    return result


# ============================================================
# P459: 从权重统计推导alpha的第一性原理尝试
# ============================================================
def run_p459(model_name, model, tokenizer, device):
    """
    P459: 从权重统计推导alpha
    - 各层权重矩阵的奇异值分布(完整谱)
    - W_U行空间的基在各层权重矩阵中的"表示度"
    - 理论: alpha = f(权重频谱, W_U频谱)
    """
    info = get_model_info(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    mlp_type = MODEL_CONFIGS[model_name]['mlp_type']
    
    print(f"\n  P459: 从权重统计推导alpha - {model_name}")
    
    # Step 1: W_U的SVD
    k_svd = min(800, d_model - 1)
    W_U = get_W_U(model)
    W_U_T = W_U.T.astype(np.float32)
    
    print(f"  计算W_U SVD (k={k_svd})...")
    U_wut, s_wut, _ = svds(W_U_T, k=k_svd)
    sort_idx = np.argsort(s_wut)[::-1]
    s_wut = s_wut[sort_idx]
    U_wut = U_wut[:, sort_idx]
    
    # Step 2: 各层权重矩阵的频谱分析
    sample_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    sample_layers = [li for li in sample_layers if li < n_layers]
    
    layer_spectra = {}
    
    for layer_idx in sample_layers:
        layer = get_layers(model)[layer_idx]
        lw = get_layer_weights(layer, d_model, mlp_type)
        
        print(f"\n  Layer {layer_idx}:")
        
        weight_matrices = {
            'W_q': np.asarray(lw.W_q).astype(np.float32),
            'W_k': np.asarray(lw.W_k).astype(np.float32),
            'W_v': np.asarray(lw.W_v).astype(np.float32),
            'W_o': np.asarray(lw.W_o).astype(np.float32),
            'W_up': np.asarray(lw.W_up).astype(np.float32),
            'W_down': np.asarray(lw.W_down).astype(np.float32),
        }
        
        layer_result = {}
        
        for name, W in weight_matrices.items():
            # W的奇异值
            k_w = min(300, min(W.shape) - 1)
            _, s_W, _ = svds(W.astype(np.float64), k=k_w)
            s_W = np.sort(s_W)[::-1]
            
            # W_U基在W中的表示度
            # 如果W的输出维度=d_model(W_down, W_o, W_q等): 
            #   WU_repr = ||U_wut^T @ W||_F^2 / ||W||_F^2
            # 如果W的输入维度=d_model(W_up, W_gate):
            #   WU_repr = ||W @ U_wut||_F^2 / ||W||_F^2
            energy_total = np.sum(W ** 2)
            if W.shape[0] == d_model:
                # 输出维度=d_model: W的输出行在d_model空间中
                WU_in_W = U_wut[:, :k_w].T @ W  # [k_w, W_cols]
                energy_in_WU = np.sum(WU_in_W ** 2)
            elif W.shape[1] == d_model:
                # 输入维度=d_model: W的输入来自d_model空间
                WU_in_W = W @ U_wut[:, :k_w]  # [W_rows, k_w]
                energy_in_WU = np.sum(WU_in_W ** 2)
            else:
                energy_in_WU = 0.0
            representation = energy_in_WU / energy_total
            
            # 频谱衰减率
            if len(s_W) > 10:
                # 幂律衰减: s_i ~ i^(-gamma)
                valid = s_W > 1e-10
                if valid.sum() > 10:
                    log_s = np.log10(s_W[valid])
                    log_i = np.log10(np.arange(1, len(log_s) + 1, dtype=float))
                    gamma = -np.polyfit(log_i, log_s, 1)[0]  # 衰减率
                else:
                    gamma = 0.0
            else:
                gamma = 0.0
            
            layer_result[name] = {
                'sv_top5': s_W[:5].tolist(),
                'sv_range': [float(s_W[0]), float(s_W[-1])],
                'WU_representation': float(representation),
                'spectral_decay_rate': float(gamma),
            }
            
            print(f"    {name}: WU_repr={representation:.3f}, decay_rate={gamma:.3f}, SV=[{s_W[0]:.2f}..{s_W[-1]:.4f}]")
        
        layer_spectra[layer_idx] = layer_result
    
    # Step 3: 理论推导尝试
    # 假设: alpha = f(各层W的WU_representation的加权平均)
    # 理论: 如果W的输出大部分在W_U行空间中, 信号经过W后也被拉向W_U行空间
    
    print(f"\n  理论推导: alpha与WU_representation的关系:")
    
    # 收集所有层的W_down的WU_representation
    wdown_repr = [layer_spectra[li]['W_down']['WU_representation'] for li in sample_layers]
    wup_repr = [layer_spectra[li]['W_up']['WU_representation'] for li in sample_layers]
    attn_repr = [np.mean([layer_spectra[li][n]['WU_representation'] 
                          for n in ['W_q', 'W_k', 'W_v', 'W_o']]) 
                 for li in sample_layers]
    
    print(f"  W_down WU_representation: {wdown_repr}")
    print(f"  W_up WU_representation: {wup_repr}")
    print(f"  Attention WU_representation: {attn_repr}")
    
    # 加权平均(假设每层贡献等权重)
    avg_wdown_repr = np.mean(wdown_repr)
    avg_wup_repr = np.mean(wup_repr)
    avg_attn_repr = np.mean(attn_repr)
    avg_all_repr = np.mean(wdown_repr + wup_repr + attn_repr)
    
    print(f"\n  平均WU_representation:")
    print(f"    W_down: {avg_wdown_repr:.3f}")
    print(f"    W_up: {avg_wup_repr:.3f}")
    print(f"    Attention: {avg_attn_repr:.3f}")
    print(f"    总平均: {avg_all_repr:.3f}")
    
    # 对比各层的spectral_decay_rate
    wdown_decay = [layer_spectra[li]['W_down']['spectral_decay_rate'] for li in sample_layers]
    wup_decay = [layer_spectra[li]['W_up']['spectral_decay_rate'] for li in sample_layers]
    
    print(f"\n  W_down spectral_decay: {wdown_decay}")
    print(f"  W_up spectral_decay: {wup_decay}")
    
    result = {
        'experiment': 'P459',
        'model': model_name,
        'd_model': d_model,
        'n_layers': n_layers,
        'layer_spectra': layer_spectra,
        'avg_WU_representation': {
            'W_down': float(avg_wdown_repr),
            'W_up': float(avg_wup_repr),
            'Attention': float(avg_attn_repr),
            'Overall': float(avg_all_repr),
        },
    }
    
    save_result(model_name, 'p459', result)
    return result


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Phase XCIV: 训练动力学与频谱策略")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["qwen3", "glm4", "deepseek7b"],
                       help="模型名称")
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["p456", "p457", "p458", "p459", "all"],
                       help="实验编号")
    args = parser.parse_args()
    
    print(f"=" * 70)
    print(f"Phase XCIV: 训练动力学与频谱策略")
    print(f"模型: {args.model}, 实验: {args.experiment}")
    print(f"=" * 70)
    
    # 加载模型
    print(f"\n加载模型: {args.model}...")
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    print(f"  d_model={info.d_model}, n_layers={info.n_layers}")
    
    try:
        experiments = ["p456", "p457", "p458", "p459"] if args.experiment == "all" else [args.experiment]
        
        for exp in experiments:
            print(f"\n{'='*50}")
            print(f"运行: {exp}")
            print(f"{'='*50}")
            
            start_time = time.time()
            
            if exp == "p456":
                run_p456(args.model, model, tokenizer, device)
            elif exp == "p457":
                run_p457(args.model, model, tokenizer, device)
            elif exp == "p458":
                run_p458(args.model, model, tokenizer, device)
            elif exp == "p459":
                run_p459(args.model, model, tokenizer, device)
            
            elapsed = time.time() - start_time
            print(f"\n  {exp} 完成, 耗时: {elapsed:.1f}s")
    
    except Exception as e:
        print(f"\n  实验出错: {e}")
        traceback.print_exc()
    
    finally:
        # 释放模型
        print(f"\n释放模型: {args.model}...")
        release_model(model)
        print(f"  GPU内存已释放")


if __name__ == "__main__":
    main()
