"""
Phase CXIX-CXX: 流形动力系统验证
==================================

Phase CXVII-CXVIII核心发现:
- P538修正频谱力学层级别无效(反而差38%)
- P539: 乘积矩阵M=W_active@W_up进一步降秩(Qwen3 PR=0.86 vs 0.97)
- P541: DS7B中层h_out频谱≈h_in频谱: alpha≈1, beta≈0
- DS7B preservation(k50)=0.987, 频谱相关=1.0000

Phase CXIX-CXX核心思路 - 第一性原理验证:
1. P542: 流形假设验证 — h的内在维度(intrinsic dimensionality)是否远小于d_model?
   - 用W_U奇异值的参与率(PR)估计h的内在维度
   - 用最近邻距离比估计局部内在维度
   - 用PCA解释方差比估计全局内在维度

2. P543: 动力系统假设验证 — h在W_U奇异空间中的轨迹是否沿固定方向演化?
   - 追踪h在W_U top-10奇异方向上的坐标随层的变化
   - 计算h_in->h_out的旋转角度分布
   - 检查轨迹是否沿W_U top方向"滑行"

3. P544: 吸引子假设验证 — 是否存在"吸引子"方向使h收敛?
   - 分析h的范数随层的变化(归一化效应)
   - 计算h_in和h_out在W_U空间中的频谱距离
   - 验证是否有"不动点"方向(频谱形状不变的方向)

4. P545: 训练隐式正则化验证 — 低秩结构是否来自梯度下降?
   - 分析W_down*W_up的秩 vs 随机初始化矩阵的秩
   - 对比不同层深度的低秩程度(早期层vs后期层)
   - 验证低秩是否集中在语义相关的方向

5. P546: 跨语言频谱验证 — 不同语言的ratio(k)是否相同?
   - 用中/英/日/法/德5种语言输入测试
   - 对比ratio(k50)和频谱形状
   - 验证频谱结构是否语言无关

使用方法:
    python phase_cxix_manifold_dynamics.py --model qwen3 --experiment p542
    python phase_cxix_manifold_dynamics.py --model glm4 --experiment p543
    python phase_cxix_manifold_dynamics.py --model deepseek7b --experiment p546
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


# ===== P542: 流形假设验证 =====
def run_p542(model, tokenizer, device, model_name):
    """
    流形假设: h存在于低维流形上, 内在维度远小于d_model
    
    验证方法:
    1. W_U奇异值PR: 如果h主要在W_U top-k空间中, 则内在维度≈k
    2. h在W_U空间的频谱集中度: PR(h_wu) = (sum s_i)^2 / (k * sum s_i^2)
    3. 多token平均: 用多个不同token的h计算平均内在维度
    4. 逐层追踪: 内在维度是否随层变化
    """
    print("\n" + "="*70)
    print("P542: 流形假设验证 — h的内在维度")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    # W_U SVD
    W_U = get_W_U(model)
    k_wu = min(200, min(W_U.shape) - 2)
    print(f"计算W_U SVD (k={k_wu})...")
    U_wu, S_wu, _ = svds(W_U.T.astype(np.float32), k=k_wu)
    U_wu = U_wu[:, ::-1]
    S_wu = S_wu[::-1]
    
    # W_U频谱的PR
    S_wu_norm = S_wu / (S_wu[0] + 1e-10)
    PR_WU = float(S_wu_norm.sum()**2 / (k_wu * (S_wu_norm**2).sum()))
    print(f"W_U频谱 PR(200) = {PR_WU:.4f}")
    
    # 累积解释方差比
    S_wu_sq = S_wu**2
    cumvar = np.cumsum(S_wu_sq) / np.sum(S_wu_sq)
    for k_target in [10, 20, 50, 100, 200]:
        print(f"  W_U top-{k_target} 解释方差: {cumvar[k_target-1]*100:.1f}%")
    
    # 90%方差对应的维度
    dim_90 = int(np.searchsorted(cumvar, 0.9) + 1)
    dim_95 = int(np.searchsorted(cumvar, 0.95) + 1)
    dim_99 = int(np.searchsorted(cumvar, 0.99) + 1)
    print(f"  W_U 90%方差维度: {dim_90}, 95%: {dim_95}, 99%: {dim_99}")
    
    # 多个不同输入测试
    test_texts = [
        "The fundamental nature of reality can be understood through mathematical structures.",
        "In the beginning was the Word, and the Word was with God, and the Word was God.",
        "The quantum mechanical description of nature requires probabilistic interpretation.",
        "Artificial intelligence systems process information through layered transformations.",
        "The beauty of mathematics lies in its ability to reveal hidden patterns in nature.",
    ]
    
    sample_layers = get_sample_layers(info.n_layers, min(8, info.n_layers))
    print(f"\n采样层: {sample_layers}")
    
    all_results = []
    
    for text_idx, test_text in enumerate(test_texts):
        inputs = tokenizer(test_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
            h_states = outputs.hidden_states
        
        text_results = []
        for l_idx in sample_layers:
            h = h_states[l_idx][0, -1].cpu().float().numpy()
            
            # h在W_U空间中的投影
            h_wu = U_wu.T @ h  # [k_wu]
            h_wu_sq = h_wu**2
            h_wu_total = np.sum(h_wu_sq) + 1e-10
            
            # h在W_U top-k上的能量比
            ratio_k = {}
            for k_target in [10, 20, 50, 100, 200]:
                top_idx = np.argsort(h_wu_sq)[-k_target:]
                ratio_k[k_target] = float(np.sum(h_wu_sq[top_idx]) / h_wu_total)
            
            # h_wu频谱的PR(内在维度估计)
            h_wu_norm = np.sqrt(h_wu_sq)
            h_wu_norm = h_wu_norm / (h_wu_norm[0] + 1e-10)
            PR_h = float(h_wu_norm.sum()**2 / (k_wu * (h_wu_norm**2).sum()))
            
            # h的总范数和W_U投影范数
            h_norm = float(np.linalg.norm(h))
            h_wu_proj_norm = float(np.sqrt(np.sum(h_wu_sq)))
            wu_coverage = float(h_wu_proj_norm**2 / (h_norm**2 + 1e-10))
            
            text_results.append({
                'layer': l_idx,
                'PR_h_wu': PR_h,
                'wu_coverage': wu_coverage,
                'ratio_k': ratio_k,
                'h_norm': h_norm,
                'dim_90': int(np.searchsorted(np.cumsum(h_wu_sq) / h_wu_total, 0.9) + 1),
                'dim_95': int(np.searchsorted(np.cumsum(h_wu_sq) / h_wu_total, 0.95) + 1),
            })
        
        all_results.append(text_results)
        torch.cuda.empty_cache()
        print(f"  Text {text_idx+1}: ratio(50)={text_results[len(text_results)//2]['ratio_k'][50]:.4f}, "
              f"PR_h={text_results[len(text_results)//2]['PR_h_wu']:.4f}")
    
    # ===== 汇总分析 =====
    print("\n--- 流形假设汇总 ---")
    
    # 各层平均内在维度
    for li, l_idx in enumerate(sample_layers):
        ratios = [r[li]['ratio_k'][50] for r in all_results]
        prs = [r[li]['PR_h_wu'] for r in all_results]
        dims_90 = [r[li]['dim_90'] for r in all_results]
        coverages = [r[li]['wu_coverage'] for r in all_results]
        print(f"  L{l_idx}: ratio(50)={np.mean(ratios):.4f}+/-{np.std(ratios):.4f}, "
              f"PR_h={np.mean(prs):.4f}, dim_90={np.mean(dims_90):.1f}, "
              f"WU覆盖={np.mean(coverages)*100:.1f}%")
    
    # 关键判断
    avg_ratio_50 = np.mean([r[li]['ratio_k'][50] for r in all_results for li in range(len(sample_layers))])
    avg_dim_90 = np.mean([r[li]['dim_90'] for r in all_results for li in range(len(sample_layers))])
    avg_wu_cov = np.mean([r[li]['wu_coverage'] for r in all_results for li in range(len(sample_layers))])
    
    print(f"\n--- 关键指标 ---")
    print(f"  平均ratio(50): {avg_ratio_50:.4f} (如果>0.5则流形维度<50)")
    print(f"  平均dim_90: {avg_dim_90:.1f} (h的90%能量所在维度)")
    print(f"  平均WU覆盖: {avg_wu_cov*100:.1f}% (h在W_U空间中的能量占比)")
    
    if avg_ratio_50 > 0.5:
        print("  >> 流形假设支持: h的50%以上能量集中在W_U top-50空间")
    if avg_wu_cov > 0.8:
        print("  >> h几乎完全在W_U行空间中, 流形维度受W_U秩限制")
    if avg_dim_90 < 100:
        print(f"  >> 内在维度极低! h的90%能量仅需{avg_dim_90:.0f}维")
    else:
        print(f"  >> 内在维度较高: {avg_dim_90:.0f}维, 流形假设需修正")
    
    results = {
        'experiment': 'P542',
        'model': model_name,
        'PR_WU': PR_WU,
        'WU_dim_90': dim_90, 'WU_dim_95': dim_95, 'WU_dim_99': dim_99,
        'avg_ratio_50': avg_ratio_50,
        'avg_dim_90': avg_dim_90,
        'avg_wu_coverage': avg_wu_cov,
        'layer_details': all_results,
    }
    
    out_path = f"results/phase_cxix/P542_{model_name}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(to_native(results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")
    return results


# ===== P543: 动力系统假设验证 =====
def run_p543(model, tokenizer, device, model_name):
    """
    动力系统假设: h在W_U空间中的轨迹沿固定方向演化
    
    验证方法:
    1. 追踪h在W_U top-10方向上的坐标随层的变化
    2. 计算h_in->h_out的"旋转角度"(频谱空间的向量夹角)
    3. 检查轨迹是否沿W_U top方向"滑行"(而不是随机游走)
    """
    print("\n" + "="*70)
    print("P543: 动力系统假设验证 — h在W_U空间中的轨迹")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    # W_U SVD
    W_U = get_W_U(model)
    k_wu = min(100, min(W_U.shape) - 2)
    U_wu, S_wu, _ = svds(W_U.T.astype(np.float32), k=k_wu)
    U_wu = U_wu[:, ::-1]
    
    test_text = "The fundamental nature of reality can be understood through mathematical structures that underlie physical phenomena."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_hidden_states=True)
        h_states = outputs.hidden_states
    
    # 收集所有层的h
    all_h = []
    for l_idx in range(info.n_layers + 1):
        h = h_states[l_idx][0, -1].cpu().float().numpy()
        all_h.append(h)
    
    # h在W_U空间中的坐标
    all_h_wu = np.array([U_wu.T @ h for h in all_h])  # [n_layers+1, k_wu]
    
    # ===== 分析1: top-10方向坐标的层间变化 =====
    print("\n--- Top-10方向坐标的层间变化 ---")
    top10_coords = all_h_wu[:, :10]  # [n_layers+1, 10]
    
    # 每个方向的层间差分
    diffs = np.diff(top10_coords, axis=0)  # [n_layers, 10]
    
    # 每个方向的层间变化方向的一致性
    for d in range(5):
        coord = top10_coords[:, d]
        diff = diffs[:, d]
        # 变化方向一致性: 正变化/总变化
        pos_ratio = float(np.sum(diff > 0) / len(diff))
        # 自相关
        if len(diff) > 2:
            autocorr = float(np.corrcoef(diff[:-1], diff[1:])[0, 1])
        else:
            autocorr = 0
        # 坐标范围
        coord_range = float(np.max(coord) - np.min(coord))
        print(f"  W_U dir {d}: pos_ratio={pos_ratio:.3f}, "
              f"autocorr={autocorr:.3f}, range={coord_range:.2f}")
    
    # ===== 分析2: 层间旋转角度 =====
    print("\n--- 层间频谱旋转角度 ---")
    sample_layers = get_sample_layers(info.n_layers, min(8, info.n_layers))
    
    rotation_angles = []
    spectral_shifts = []
    
    for l_idx in sample_layers:
        h_in = all_h[l_idx]
        h_out = all_h[l_idx + 1]
        delta_h = h_out - h_in
        
        h_in_wu = U_wu.T @ h_in
        h_out_wu = U_wu.T @ h_out
        delta_wu = U_wu.T @ delta_h
        
        # 频谱空间中的"旋转角度"
        cos_angle = float(np.dot(h_in_wu, h_out_wu) / 
                         (np.linalg.norm(h_in_wu) * np.linalg.norm(h_out_wu) + 1e-10))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle_deg = float(np.degrees(np.arccos(cos_angle)))
        
        # delta在h_in方向上的分量(切向) vs 垂直分量(旋转)
        h_in_norm = h_in_wu / (np.linalg.norm(h_in_wu) + 1e-10)
        tangential = float(np.dot(delta_wu, h_in_norm))
        perpendicular = float(np.linalg.norm(delta_wu - tangential * h_in_norm))
        rotation_ratio = perpendicular / (perpendicular + abs(tangential) + 1e-10)
        
        rotation_angles.append(angle_deg)
        spectral_shifts.append(rotation_ratio)
        
        print(f"  L{l_idx}: angle={angle_deg:.2f}deg, "
              f"rotation_ratio={rotation_ratio:.4f}")
    
    # ===== 分析3: 轨迹的主方向 =====
    print("\n--- 轨迹主方向分析 ---")
    
    # 所有层间delta的PCA
    all_deltas_wu = np.diff(all_h_wu, axis=0)  # [n_layers, k_wu]
    all_deltas_centered = all_deltas_wu - all_deltas_wu.mean(axis=0)
    
    if min(all_deltas_centered.shape) > 1:
        U_delta, S_delta, _ = svds(all_deltas_centered.astype(np.float32), 
                                    k=min(10, min(all_deltas_centered.shape) - 1))
        S_delta = S_delta[::-1]
        
        PR_delta = float(S_delta.sum()**2 / (len(S_delta) * (S_delta**2).sum() + 1e-10))
        print(f"  delta轨迹PR(10) = {PR_delta:.4f}")
        
        # 轨迹主方向与W_U top方向的对齐度
        # U_delta: [n_layers-1, k], 每行是一层的delta频谱
        # 对齐度: delta频谱主模式与W_U奇异方向的对齐
        n_layers_sampled = U_delta.shape[0]
        for d in range(min(5, U_delta.shape[1])):
            delta_dir = U_delta[:, d]  # [n_layers-1]
            # 检查delta主模式在不同层的权重分布
            top_layer = int(np.argmax(np.abs(delta_dir)))
            print(f"  delta主方向{d}: 主要来自层{sample_layers[top_layer] if top_layer < len(sample_layers) else top_layer}, "
                  f"奇异值={S_delta[d]:.2f}")
    else:
        PR_delta = 1.0
        print("  层数不足, 无法计算delta轨迹PR")
    
    # ===== 分析4: 轨迹是否沿W_U top方向"滑行" =====
    print("\n--- 轨迹滑行分析 ---")
    
    # h的范数变化
    h_norms = [float(np.linalg.norm(h)) for h in all_h]
    print(f"  h范数: 首层={h_norms[0]:.2f}, 末层={h_norms[-1]:.2f}, "
          f"变化={abs(h_norms[-1]-h_norms[0])/h_norms[0]*100:.1f}%")
    
    # 频谱分布的层间相关
    h_wu_sq = all_h_wu**2
    h_wu_norm = h_wu_sq / (h_wu_sq.sum(axis=1, keepdims=True) + 1e-10)
    
    # 相邻层频谱相关
    inter_layer_corr = []
    for l in range(info.n_layers):
        corr = float(np.corrcoef(h_wu_norm[l], h_wu_norm[l+1])[0, 1])
        inter_layer_corr.append(corr)
    
    avg_corr = float(np.mean(inter_layer_corr))
    min_corr = float(np.min(inter_layer_corr))
    print(f"  相邻层频谱相关: avg={avg_corr:.4f}, min={min_corr:.4f}")
    
    # 关键判断
    print(f"\n--- 关键指标 ---")
    print(f"  平均旋转角度: {np.mean(rotation_angles):.2f}deg")
    print(f"  旋转比率(垂直/总): {np.mean(spectral_shifts):.4f}")
    print(f"  delta轨迹PR: {PR_delta:.4f}")
    print(f"  频谱层间相关: {avg_corr:.4f}")
    
    if np.mean(rotation_angles) < 5:
        print("  >> 动力系统假设强支持: h几乎不旋转, 沿固定方向演化")
    elif np.mean(rotation_angles) < 15:
        print("  >> 动力系统假设部分支持: h有小幅旋转, 但方向大致稳定")
    else:
        print("  >> 动力系统假设不支持: h有显著旋转")
    
    if avg_corr > 0.99:
        print("  >> 频谱形状极端稳定: 动力系统沿吸引子滑行")
    
    results = {
        'experiment': 'P543',
        'model': model_name,
        'avg_rotation_angle': float(np.mean(rotation_angles)),
        'avg_rotation_ratio': float(np.mean(spectral_shifts)),
        'PR_delta': PR_delta,
        'avg_inter_layer_corr': avg_corr,
        'min_inter_layer_corr': min_corr,
        'h_norm_change_pct': float(abs(h_norms[-1]-h_norms[0])/h_norms[0]*100),
    }
    
    out_path = f"results/phase_cxix/P543_{model_name}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(to_native(results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")
    
    torch.cuda.empty_cache()
    return results


# ===== P544: 吸引子假设验证 =====
def run_p544(model, tokenizer, device, model_name):
    """
    吸引子假设: 存在"吸引子"方向使h收敛
    
    验证方法:
    1. 分析h范数随层的变化(LN归一化效应)
    2. 多输入的h是否收敛到相同的频谱分布?
    3. "不动点"方向: 频谱形状不变的方向
    4. 吸引子的吸引域大小
    """
    print("\n" + "="*70)
    print("P544: 吸引子假设验证 — h的收敛行为")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    # W_U SVD
    W_U = get_W_U(model)
    k_wu = min(100, min(W_U.shape) - 2)
    U_wu, S_wu, _ = svds(W_U.T.astype(np.float32), k=k_wu)
    U_wu = U_wu[:, ::-1]
    
    # 多种不同输入
    test_texts = [
        "The fundamental nature of reality can be understood through mathematical structures.",
        "In the beginning was the Word, and the Word was with God.",
        "The quantum mechanical description requires probabilistic interpretation.",
        "Colorless green ideas sleep furiously in the deep dark forest.",
        "She sells seashells by the seashore every morning at dawn.",
        "12345 plus 67890 equals 80235 in standard decimal arithmetic.",
        "Paris is the capital of France and Berlin is the capital of Germany.",
        "The cat sat on the mat while the dog played in the yard.",
    ]
    
    sample_layers = get_sample_layers(info.n_layers, min(10, info.n_layers))
    print(f"采样层: {sample_layers}")
    
    # 收集所有输入在所有层的频谱
    all_spectra = []  # [n_texts, n_sample_layers, k_wu]
    
    for text_idx, test_text in enumerate(test_texts):
        inputs = tokenizer(test_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
            h_states = outputs.hidden_states
        
        spectra = []
        for l_idx in sample_layers:
            h = h_states[l_idx][0, -1].cpu().float().numpy()
            h_wu = U_wu.T @ h
            h_wu_sq = h_wu**2
            # 归一化为频谱分布
            h_wu_dist = h_wu_sq / (np.sum(h_wu_sq) + 1e-10)
            spectra.append(h_wu_dist)
        
        all_spectra.append(spectra)
        torch.cuda.empty_cache()
        print(f"  Text {text_idx+1}/{len(test_texts)} 完成")
    
    all_spectra = np.array(all_spectra)  # [n_texts, n_sample_layers, k_wu]
    
    # ===== 分析1: 同层不同输入的频谱相似度 =====
    print("\n--- 同层不同输入的频谱相似度 ---")
    
    inter_text_corrs = []
    for li, l_idx in enumerate(sample_layers):
        spectra_at_layer = all_spectra[:, li, :]  # [n_texts, k_wu]
        # 两两相关
        corrs = []
        for i in range(len(test_texts)):
            for j in range(i+1, len(test_texts)):
                c = float(np.corrcoef(spectra_at_layer[i], spectra_at_layer[j])[0, 1])
                corrs.append(c)
        avg_corr = float(np.mean(corrs))
        inter_text_corrs.append(avg_corr)
        print(f"  L{l_idx}: 跨输入频谱相关 avg={avg_corr:.4f}")
    
    # ===== 分析2: 频谱分布是否随层收敛 =====
    print("\n--- 频谱分布的层间收敛 ---")
    
    # 每个输入: 首层与末层的频谱相关 vs 相邻层频谱相关
    convergence_data = []
    for text_idx in range(len(test_texts)):
        first_last_corr = float(np.corrcoef(all_spectra[text_idx, 0], 
                                             all_spectra[text_idx, -1])[0, 1])
        # 相邻层平均相关
        adj_corrs = []
        for li in range(len(sample_layers) - 1):
            c = float(np.corrcoef(all_spectra[text_idx, li], 
                                   all_spectra[text_idx, li+1])[0, 1])
            adj_corrs.append(c)
        convergence_data.append({
            'first_last': first_last_corr,
            'avg_adjacent': float(np.mean(adj_corrs)),
        })
        print(f"  Text {text_idx+1}: 首末层相关={first_last_corr:.4f}, "
              f"相邻层平均={np.mean(adj_corrs):.4f}")
    
    # ===== 分析3: 吸引子频谱形状 =====
    print("\n--- 吸引子频谱形状 ---")
    
    # 末层所有输入的平均频谱
    final_avg_spectrum = np.mean(all_spectra[:, -1, :], axis=0)
    
    # 用幂律拟合
    nonzero = final_avg_spectrum > 1e-10
    if np.sum(nonzero) > 10:
        ranks = np.arange(1, k_wu + 1)[nonzero]
        log_ranks = np.log(ranks)
        log_spectrum = np.log(final_avg_spectrum[nonzero] + 1e-20)
        
        # 线性拟合
        try:
            coeffs = np.polyfit(log_ranks, log_spectrum, 1)
            exponent = float(coeffs[0])
            print(f"  末层平均频谱幂律指数: {exponent:.3f}")
        except:
            exponent = 0
            print("  幂律拟合失败")
    
    # ===== 分析4: 吸引域大小 =====
    print("\n--- 吸引域分析 ---")
    
    # 首层频谱的散度 vs 末层频谱的散度
    first_layer_var = float(np.mean(np.var(all_spectra[:, 0, :], axis=0)))
    final_layer_var = float(np.mean(np.var(all_spectra[:, -1, :], axis=0)))
    
    var_reduction = float(1 - final_layer_var / (first_layer_var + 1e-10))
    print(f"  首层频谱方差: {first_layer_var:.6f}")
    print(f"  末层频谱方差: {final_layer_var:.6f}")
    print(f"  方差缩减: {var_reduction*100:.1f}%")
    
    # 关键判断
    avg_inter_text = float(np.mean(inter_text_corrs))
    avg_first_last = float(np.mean([d['first_last'] for d in convergence_data]))
    
    print(f"\n--- 关键指标 ---")
    print(f"  跨输入频谱相关: {avg_inter_text:.4f}")
    print(f"  首末层频谱相关: {avg_first_last:.4f}")
    print(f"  频谱方差缩减: {var_reduction*100:.1f}%")
    
    if avg_inter_text > 0.9:
        print("  >> 吸引子假设强支持: 不同输入收敛到相似频谱")
    elif avg_inter_text > 0.7:
        print("  >> 吸引子假设部分支持: 不同输入频谱有中等相似度")
    else:
        print("  >> 吸引子假设不支持: 不同输入频谱差异大")
    
    if var_reduction > 0.5:
        print("  >> 强收敛: 频谱方差减少>50%, 存在强吸引子")
    elif var_reduction > 0:
        print("  >> 弱收敛: 频谱方差减少, 但吸引子较弱")
    else:
        print("  >> 无收敛: 频谱方差增加, 无吸引子效应")
    
    results = {
        'experiment': 'P544',
        'model': model_name,
        'avg_inter_text_corr': avg_inter_text,
        'avg_first_last_corr': avg_first_last,
        'variance_reduction': var_reduction,
        'first_layer_var': first_layer_var,
        'final_layer_var': final_layer_var,
        'spectral_exponent': exponent,
    }
    
    out_path = f"results/phase_cxix/P544_{model_name}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(to_native(results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")
    
    torch.cuda.empty_cache()
    return results


# ===== P545: 训练隐式正则化验证 =====
def run_p545(model, tokenizer, device, model_name):
    """
    训练隐式正则化: 低秩结构是否来自梯度下降的隐式正则化?
    
    验证方法:
    1. 对比W_down*W_up的PR vs 随机矩阵的PR
    2. 不同层深度的低秩程度(早期vs后期)
    3. 低秩方向与W_U的对齐度(语义相关性)
    """
    print("\n" + "="*70)
    print("P545: 训练隐式正则化验证 — 低秩来源")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    # W_U SVD
    W_U = get_W_U(model)
    k_wu = min(100, min(W_U.shape) - 2)
    U_wu, S_wu, _ = svds(W_U.T.astype(np.float32), k=k_wu)
    U_wu = U_wu[:, ::-1]
    
    sample_layers = get_sample_layers(info.n_layers, min(10, info.n_layers))
    print(f"采样层: {sample_layers}")
    
    # 对比: 训练后的W_down*W_up vs 随机初始化
    print("\n--- 训练后 vs 随机矩阵的低秩对比 ---")
    
    np.random.seed(42)
    
    trained_prs = []
    random_prs = []
    wu_alignments = []
    
    for l_idx in sample_layers:
        lw = get_layer_weights(layers[l_idx], info.d_model, info.mlp_type)
        W_down = lw.W_down
        W_up = lw.W_up
        
        intermediate = W_down.shape[1]
        
        # 训练后的W_down*W_up的PR
        # 采样top-k active neurons
        k_active = min(10, intermediate)
        
        # 用W_down的列范数确定active neurons
        col_norms = np.linalg.norm(W_down, axis=0)
        top_k_idx = np.argsort(col_norms)[-k_active:]
        
        W_down_k = W_down[:, top_k_idx]  # [d_model, k]
        W_up_k = W_up[top_k_idx, :]      # [k, d_model]
        
        M_trained = W_down_k @ W_up_k  # [d_model, d_model]
        
        # PR of M_trained
        k_svd = min(10, min(M_trained.shape) - 1)
        U_m, S_m, _ = svds(M_trained.astype(np.float32), k=k_svd)
        S_m = S_m[::-1]
        PR_trained = float(S_m.sum()**2 / (k_svd * (S_m**2).sum() + 1e-10))
        
        # 随机矩阵: 同形状的高斯随机矩阵
        M_random = np.random.randn(info.d_model, intermediate) @ np.random.randn(intermediate, info.d_model)
        M_random_k = M_random  # [d_model, d_model]
        U_r, S_r, _ = svds(M_random_k.astype(np.float32), k=k_svd)
        S_r = S_r[::-1]
        PR_random = float(S_r.sum()**2 / (k_svd * (S_r**2).sum() + 1e-10))
        
        # M_trained的top方向与W_U的对齐度
        top_dir = U_m[:, 0]  # M_trained的top奇异方向
        wu_alignment = float(abs(np.dot(top_dir, U_wu[:, 0])))
        
        trained_prs.append(PR_trained)
        random_prs.append(PR_random)
        wu_alignments.append(wu_alignment)
        
        print(f"  L{l_idx}: PR(trained)={PR_trained:.4f}, PR(random)={PR_random:.4f}, "
              f"WU对齐={wu_alignment:.4f}")
    
    # ===== 分析2: 层深度与低秩的关系 =====
    print("\n--- 层深度与低秩的关系 ---")
    
    # 线性拟合: PR vs 层深度
    layer_indices = np.array(sample_layers, dtype=float)
    pr_array = np.array(trained_prs)
    
    if len(layer_indices) > 2:
        corr_depth_pr = float(np.corrcoef(layer_indices, pr_array)[0, 1])
        slope = float(np.polyfit(layer_indices, pr_array, 1)[0])
        print(f"  PR与层深度相关: r={corr_depth_pr:.3f}, slope={slope:.6f}")
        
        # 早期层vs后期层
        mid = len(sample_layers) // 2
        early_pr = float(np.mean(trained_prs[:mid]))
        late_pr = float(np.mean(trained_prs[mid:]))
        print(f"  早期层PR: {early_pr:.4f}, 后期层PR: {late_pr:.4f}")
    else:
        corr_depth_pr = 0
        early_pr = late_pr = float(np.mean(trained_prs))
    
    # ===== 分析3: 低秩方向与语义方向的对齐 =====
    print("\n--- 低秩方向与语义方向的对齐 ---")
    
    avg_wu_align = float(np.mean(wu_alignments))
    
    # 对比: 随机方向的期望对齐度
    n_random_trials = 100
    random_aligns = []
    for _ in range(n_random_trials):
        rand_dir = np.random.randn(info.d_model)
        rand_dir = rand_dir / np.linalg.norm(rand_dir)
        random_aligns.append(float(abs(np.dot(rand_dir, U_wu[:, 0]))))
    expected_random_align = float(np.mean(random_aligns))
    
    print(f"  M_trained top方向与W_U top方向平均对齐: {avg_wu_align:.4f}")
    print(f"  随机方向的期望对齐: {expected_random_align:.4f}")
    print(f"  对齐度提升: {avg_wu_align/expected_random_align:.1f}x")
    
    # 关键判断
    avg_trained_pr = float(np.mean(trained_prs))
    avg_random_pr = float(np.mean(random_prs))
    
    print(f"\n--- 关键指标 ---")
    print(f"  平均PR(trained): {avg_trained_pr:.4f}")
    print(f"  平均PR(random): {avg_random_pr:.4f}")
    print(f"  低秩比: {avg_trained_pr/avg_random_pr:.2f}")
    
    if avg_trained_pr < avg_random_pr * 0.5:
        print("  >> 隐式正则化强支持: 训练后矩阵远比随机矩阵低秩")
    elif avg_trained_pr < avg_random_pr * 0.8:
        print("  >> 隐式正则化部分支持: 训练后矩阵比随机矩阵低秩")
    else:
        print("  >> 隐式正则化不支持: 训练后矩阵不比随机矩阵更低秩")
    
    if avg_wu_align > 3 * expected_random_align:
        print("  >> 低秩方向与语义方向高度对齐: 正则化是有方向的")
    else:
        print("  >> 低秩方向与语义方向对齐度低: 正则化可能是无方向的")
    
    results = {
        'experiment': 'P545',
        'model': model_name,
        'avg_trained_PR': avg_trained_pr,
        'avg_random_PR': avg_random_pr,
        'low_rank_ratio': avg_trained_pr / avg_random_pr,
        'avg_wu_alignment': avg_wu_align,
        'expected_random_alignment': expected_random_align,
        'alignment_enhancement': avg_wu_align / expected_random_align,
        'corr_depth_pr': corr_depth_pr,
    }
    
    out_path = f"results/phase_cxix/P545_{model_name}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(to_native(results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")
    
    return results


# ===== P546: 跨语言频谱验证 =====
def run_p546(model, tokenizer, device, model_name):
    """
    跨语言验证: 不同语言的ratio(k)和频谱是否相同?
    
    验证方法:
    1. 用5种语言输入测试
    2. 对比ratio(k50)和频谱形状
    3. 验证频谱结构是否语言无关
    """
    print("\n" + "="*70)
    print("P546: 跨语言频谱验证")
    print("="*70)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    # W_U SVD
    W_U = get_W_U(model)
    k_wu = min(100, min(W_U.shape) - 2)
    U_wu, S_wu, _ = svds(W_U.T.astype(np.float32), k=k_wu)
    U_wu = U_wu[:, ::-1]
    
    # 5种语言的等价文本
    # 注意: Qwen3/DS7B支持中文, GLM4支持中文
    test_texts = {
        "en": "The fundamental nature of reality can be understood through mathematical structures.",
        "zh": "现实的基本本质可以通过数学结构来理解。",
        "ja": "現実の根本的な性質は数学的構造を通じて理解できる。",
        "fr": "La nature fondamentale de la realite peut etre comprise a travers des structures mathematiques.",
        "de": "Die grundlegende Natur der Realitaet kann durch mathematische Strukturen verstanden werden.",
    }
    
    sample_layers = get_sample_layers(info.n_layers, min(8, info.n_layers))
    print(f"采样层: {sample_layers}")
    
    all_lang_results = {}
    
    for lang, text in test_texts.items():
        print(f"\n--- 语言: {lang} ---")
        
        try:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(inputs["input_ids"], output_hidden_states=True)
                h_states = outputs.hidden_states
            
            lang_ratios = {}
            lang_spectra = {}
            
            for l_idx in sample_layers:
                h = h_states[l_idx][0, -1].cpu().float().numpy()
                h_wu = U_wu.T @ h
                h_wu_sq = h_wu**2
                h_wu_total = np.sum(h_wu_sq) + 1e-10
                
                # ratio(k)
                for k_target in [10, 20, 50, 100]:
                    top_idx = np.argsort(h_wu_sq)[-k_target:]
                    if k_target not in lang_ratios:
                        lang_ratios[k_target] = []
                    lang_ratios[k_target].append(float(np.sum(h_wu_sq[top_idx]) / h_wu_total))
                
                # 频谱分布
                h_wu_dist = h_wu_sq / h_wu_total
                lang_spectra[l_idx] = h_wu_dist
            
            all_lang_results[lang] = {
                'ratios': lang_ratios,
                'spectra': lang_spectra,
            }
            
            print(f"  ratio(50) avg: {np.mean(lang_ratios[50]):.4f}")
            
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  语言 {lang} 处理失败: {e}")
            all_lang_results[lang] = None
    
    # ===== 跨语言对比 =====
    print("\n--- 跨语言频谱对比 ---")
    
    valid_langs = [lang for lang, res in all_lang_results.items() if res is not None]
    
    # ratio(k50)对比
    print("ratio(50) 各语言:")
    ratio_50_values = []
    for lang in valid_langs:
        avg_ratio = float(np.mean(all_lang_results[lang]['ratios'][50]))
        ratio_50_values.append(avg_ratio)
        print(f"  {lang}: {avg_ratio:.4f}")
    
    # 跨语言频谱相关(末层)
    print("\n跨语言频谱相关(末层):")
    final_langs = []
    for lang in valid_langs:
        if sample_layers[-1] in all_lang_results[lang]['spectra']:
            final_langs.append(lang)
    
    if len(final_langs) >= 2:
        cross_lang_corrs = []
        for i, lang1 in enumerate(final_langs):
            for j, lang2 in enumerate(final_langs):
                if j > i:
                    spec1 = all_lang_results[lang1]['spectra'][sample_layers[-1]]
                    spec2 = all_lang_results[lang2]['spectra'][sample_layers[-1]]
                    corr = float(np.corrcoef(spec1, spec2)[0, 1])
                    cross_lang_corrs.append(corr)
                    print(f"  {lang1}-{lang2}: {corr:.4f}")
        
        avg_cross_corr = float(np.mean(cross_lang_corrs)) if cross_lang_corrs else 0
    else:
        avg_cross_corr = 0
        print("  有效语言数不足, 无法计算跨语言相关")
    
    # 关键判断
    ratio_50_std = float(np.std(ratio_50_values)) if len(ratio_50_values) > 1 else 0
    ratio_50_mean = float(np.mean(ratio_50_values)) if ratio_50_values else 0
    
    print(f"\n--- 关键指标 ---")
    print(f"  ratio(50) 跨语言标准差: {ratio_50_std:.4f}")
    print(f"  ratio(50) 跨语言变异系数: {ratio_50_std/ratio_50_mean*100:.1f}%")
    print(f"  跨语言频谱平均相关: {avg_cross_corr:.4f}")
    
    if ratio_50_std / (ratio_50_mean + 1e-10) < 0.05:
        print("  >> 频谱结构语言无关: ratio(k)跨语言稳定")
    elif ratio_50_std / (ratio_50_mean + 1e-10) < 0.15:
        print("  >> 频谱结构弱语言相关: ratio(k)跨语言有中等变化")
    else:
        print("  >> 频谱结构强语言相关: ratio(k)跨语言差异大")
    
    if avg_cross_corr > 0.95:
        print("  >> 频谱形状跨语言高度一致: 支持语言无关的频谱结构")
    
    results = {
        'experiment': 'P546',
        'model': model_name,
        'ratio_50_mean': ratio_50_mean,
        'ratio_50_std': ratio_50_std,
        'ratio_50_cv': ratio_50_std / (ratio_50_mean + 1e-10),
        'avg_cross_lang_corr': avg_cross_corr,
        'lang_details': {lang: {'ratio_50': float(np.mean(res['ratios'][50]))} 
                        for lang, res in all_lang_results.items() if res is not None},
    }
    
    out_path = f"results/phase_cxix/P546_{model_name}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(to_native(results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")
    
    return results


# ===== 主函数 =====
EXPERIMENTS = {
    'p542': run_p542,
    'p543': run_p543,
    'p544': run_p544,
    'p545': run_p545,
    'p546': run_p546,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase CXIX-CXX: 流形动力系统验证")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True,
                       choices=list(EXPERIMENTS.keys()))
    args = parser.parse_args()
    
    print(f"模型: {args.model}, 实验: {args.experiment}")
    
    model, tokenizer, device = load_model(args.model)
    try:
        result = EXPERIMENTS[args.experiment](model, tokenizer, device, args.model)
        print(f"\n实验 {args.experiment} 完成!")
    finally:
        release_model(model)
