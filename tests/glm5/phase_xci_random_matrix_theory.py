"""
Phase XCI-P443/444/445/446: 随机矩阵理论验证 + 跨模型同构 + LCS预测力
======================================================================

核心目标: 验证"信号传播 = 各向同性放大 × 随机旋转 × 固定比例能量分配"的数学模型

P443: 随机矩阵理论验证 (★★★★★最核心★★★★★)
  - 假设: recoding_ratio≈0.22是随机旋转在固定维空间中的统计不变量
  - 预测: 对d_model维空间做随机旋转, 投影到W_U的k维行空间中, ratio = k/d_model
  - 验证: 如果Qwen3 d=2560, W_U rank≈250, 则ratio≈250/2560≈0.098?
         但实测ratio≈0.22! → 需要修正: ratio取决于W_U的有效覆盖维度
  - 关键实验: 用随机正交矩阵Q旋转信号, 测量Q·delta在W_U空间中的投影比

P444: 跨模型recoding_ratio同构验证
  - 在三模型(Qwen3/GLM4/DS7B)上测量recoding_ratio
  - 如果ratio在三个模型中都≈0.22 → 支持统计不变量假说
  - 如果ratio随d_model或W_U rank变化 → 需要修正公式
  - 同时测量: recoding_gain曲线, W_U PR, 方向保持度曲线

P445: LCS预测力验证
  - LCS(Layer Communication Specification) = W_up_norm × gain_factor × rank_factor
  - 预测: LCS(layer) ∝ recoding_gain(layer)
  - 验证: 计算每层的LCS, 与recoding_gain做相关分析
  - 如果R²>0.8 → LCS是有效的层间通信规范

P446: 随机权重 vs 真实权重
  - 生成随机正交权重矩阵(与真实权重同shape)
  - 用随机权重替换真实权重, 测量recoding_ratio
  - 如果随机权重也产生ratio≈0.22 → ratio是几何约束, 不是训练结果
  - 如果随机权重ratio不同 → ratio是训练产生的结构

实验模型: qwen3 → glm4 → deepseek7b (串行)
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
from scipy.stats import ortho_group

# ===== 导入标准模型工具 =====
sys.path.insert(0, str(Path(__file__).parent))
from model_utils import (
    MODEL_CONFIGS,
    load_model,
    get_layers,
    get_model_info,
    get_layer_weights,
    get_W_U,
    release_model,
    get_sample_layers,
    get_attr_direction,
    inject_at_embed,
    collect_layer_outputs,
    compute_cos,
    compute_recoding_ratio,
    compute_recoding_ratio_cached,
    LayerWeights,
    ModelInfo,
)

# ===== 实验配置 =====
PHASE_NAME = "Phase XCI"
EXPERIMENT_DESC = "随机矩阵理论验证+跨模型同构+LCS预测力"

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 测试数据
TEST_ATTRS = ["red", "blue", "big", "small", "hot", "cold", "sweet", "sour"]
PROMPT = "The apple is"
BETA = 8.0


# ================================================================
# P443: 随机矩阵理论验证
# ================================================================

def run_p443(model, tokenizer, device, model_info):
    """
    P443: 验证recoding_ratio是否是随机旋转的统计不变量
    
    原理:
      如果信号传播 = 各向同性放大 × 随机旋转 × 固定比例能量分配
      那么对任意delta, 经随机旋转Q后, 在W_U行空间中的投影比应该是常数
      
      数学推导:
      设W_U行空间的基为V (k个正交基), delta为d_model维向量
      ratio = ||V^T · delta||² / ||delta||²
      如果delta经随机旋转Q: delta' = Q · delta
      ratio' = ||V^T · Q · delta||² / ||Q·delta||²
             = ||V^T · Q · delta||² / ||delta||²  (Q保范数)
      
      关键: ratio' 是否 ≈ ratio?
      如果Q是Haar测度的随机正交矩阵, 则:
      E[ratio'] = k/d_model (其中k是W_U行空间的维度)
      
      但我们测到的ratio≈0.22, k≈250, d=2560 → k/d≈0.098 ≠ 0.22
      所以要么k≠250, 要么旋转不是纯随机的
    
    方法:
      1. 获取真实信号的recoding_ratio
      2. 对同一信号做随机旋转, 测量旋转后的ratio
      3. 用W_U的SVD计算有效覆盖维度k_eff
      4. 验证ratio ≈ k_eff/d_model
    """
    print(f"\n{'='*60}")
    print(f"P443: 随机矩阵理论验证 - {model_info.name}")
    print(f"{'='*60}")
    
    results = {}
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model)  # [vocab_size, d_model]
    wu_shape = list(W_U.shape)
    wu_max_rank = min(W_U.shape)
    
    # 先获取所有属性的direction (节省后续内存)
    attr_directions = {}
    for attr in TEST_ATTRS:
        direction, attr_tok_id = get_attr_direction(model, tokenizer, attr, W_U=W_U)
        if direction is not None:
            attr_directions[attr] = direction
    
    # Step 1: 计算W_U的有效秩(覆盖维度)
    print("  Step 1: 计算W_U有效秩...")
    # W_U shape=[vocab, d_model], 对W_U^T做SVD更高效
    # W_U^T shape=[d_model, vocab]=[2560, 151936], 远小于W_U
    # svds对W_U^T做SVD: W_U^T = U S Vt → 奇异值S相同
    W_U_T = W_U.T.astype(np.float32)  # [d_model, vocab]
    del W_U; import gc; gc.collect()  # 释放W_U
    max_k = min(W_U_T.shape[0], W_U_T.shape[1]) - 2  # min(2560, 151936)-2=2558
    k_svd = min(300, max_k)
    print(f"  对W_U^T做SVD, shape={W_U_T.shape}, k={k_svd}...")
    U_wut_raw, s_wu_raw, Vt_wut_raw = svds(W_U_T, k=k_svd)
    del W_U_T, Vt_wut_raw; gc.collect()
    # 按奇异值降序排列
    s_wu = np.abs(np.asarray(s_wu_raw, dtype=np.float64).ravel())
    sort_idx = np.argsort(-s_wu)
    s_wu = s_wu[sort_idx]
    U_wut = np.asarray(U_wut_raw, dtype=np.float64)[:, sort_idx]  # [d_model, k] — W_U行空间基(按奇异值排序)
    del U_wut_raw, s_wu_raw; gc.collect()
    
    # Participation Ratio (有效秩的连续版本)
    total_energy = np.sum(s_wu ** 2)
    PR = (np.sum(s_wu) ** 2) / max(total_energy, 1e-20)
    
    # 累积能量比
    cum_energy = np.cumsum(s_wu ** 2) / total_energy
    k_90_in_sample = int(np.searchsorted(cum_energy, 0.90)) + 1
    
    print(f"  W_U: shape={wu_shape}, PR={PR:.1f}, max_rank={wu_max_rank}")
    print(f"  SVD sample: k={k_svd}, top5 SVs={s_wu[:5].tolist()}")
    print(f"  k_90_in_sample={k_90_in_sample}, cum_energy[-1]={cum_energy[-1]:.4f}")
    print(f"  预测ratio(PR/d)={PR/d_model:.4f}")
    
    results["wu_analysis"] = {
        "shape": wu_shape,
        "PR": float(PR),
        "max_rank": wu_max_rank,
        "k_svd_used": k_svd,
        "k_90_in_sample": k_90_in_sample,
        "cum_energy_last": float(cum_energy[-1]),
        "predicted_ratio_PR_over_d": PR / d_model,
        "singular_values_top20": s_wu[:20].tolist(),
    }
    
    # Step 2: 获取真实信号的recoding_ratio (多个属性+多层)
    print("  Step 2: 测量真实信号的recoding_ratio...")
    sample_layers = get_sample_layers(n_layers, n_samples=8)
    
    real_ratios = {}  # {attr: {layer: ratio}}
    
    for attr in TEST_ATTRS:
        if attr not in attr_directions:
            continue
        direction = attr_directions[attr]
        
        inputs_base, inputs_interv, input_ids, pos_ids = inject_at_embed(
            model, tokenizer, device, PROMPT, direction, BETA
        )
        
        base_out = collect_layer_outputs(model, inputs_base, pos_ids, n_layers)
        interv_out = collect_layer_outputs(model, inputs_interv, pos_ids, n_layers)
        
        attr_ratios = {}
        for li in sample_layers:
            key = f"L{li}"
            if key not in base_out or key not in interv_out:
                continue
            
            h_base = base_out[key][0, -1, :].numpy()
            h_interv = interv_out[key][0, -1, :].numpy()
            delta_h = h_interv - h_base
            
            rc = compute_recoding_ratio_cached(delta_h, U_wut)
            attr_ratios[li] = rc
    
        real_ratios[attr] = attr_ratios
        
        if attr_ratios:
            mean_ratio = np.mean([r["ratio"] for r in attr_ratios.values()])
            print(f"  {attr}: mean_ratio={mean_ratio:.4f}")
    
    results["real_ratios"] = real_ratios
    
    # Step 3: 随机旋转验证
    print("  Step 3: 随机旋转验证...")
    
    # 用第一个属性的真实delta作为基准
    first_attr = list(real_ratios.keys())[0] if real_ratios else None
    if first_attr is None:
        print("  ERROR: No real ratios available")
        return results
    
    direction = attr_directions[first_attr]
    inputs_base, inputs_interv, input_ids, pos_ids = inject_at_embed(
        model, tokenizer, device, PROMPT, direction, BETA
    )
    base_out = collect_layer_outputs(model, inputs_base, pos_ids, n_layers)
    interv_out = collect_layer_outputs(model, inputs_interv, pos_ids, n_layers)
    
    # 取中间层的delta
    mid_layer = sample_layers[len(sample_layers)//2]
    key = f"L{mid_layer}"
    if key in base_out and key in interv_out:
        h_base = base_out[key][0, -1, :].numpy()
        h_interv = interv_out[key][0, -1, :].numpy()
        delta_real = h_interv - h_base
    else:
        # 用L0的delta
        delta_real = BETA * direction
    
    delta_norm = np.linalg.norm(delta_real)
    rc_real = compute_recoding_ratio_cached(delta_real, U_wut)
    
    print(f"  真实delta: ||Δ||={delta_norm:.4f}, ratio={rc_real['ratio']:.4f}")
    
    # 生成N个随机旋转, 测量ratio分布
    N_RANDOM = 20
    random_ratios = []
    np.random.seed(42)
    
    # 用小维度的正交群(如果d_model太大, 用随机高斯+QR分解)
    for i in range(N_RANDOM):
        # 生成随机正交矩阵Q: QR分解
        G = np.random.randn(d_model, d_model).astype(np.float32)
        Q, _ = np.linalg.qr(G)
        
        # 旋转delta
        delta_rotated = Q @ delta_real
        
        # 测量recoding_ratio
        rc = compute_recoding_ratio_cached(delta_rotated, U_wut)
        random_ratios.append(rc["ratio"])
    
    mean_random_ratio = np.mean(random_ratios)
    std_random_ratio = np.std(random_ratios)
    
    print(f"  随机旋转: mean_ratio={mean_random_ratio:.4f} ± {std_random_ratio:.4f}")
    print(f"  真实vs随机: real={rc_real['ratio']:.4f}, random={mean_random_ratio:.4f}")
    print(f"  差异: {abs(rc_real['ratio'] - mean_random_ratio):.4f}")
    
    # Step 4: 验证理论预测
    # 理论: E[ratio] = k_eff / d_model, 其中k_eff是W_U行空间的有效维度
    # W_U的秩最多=min(vocab, d_model)=d_model, 用PR作为有效维度
    predicted_ratios = {
        "PR/d": PR / d_model,
        "k_90_in_sample/d": k_90_in_sample / d_model,
        "min_shape/d": min(wu_shape) / d_model,  # =1.0 当vocab>d_model
        "d_model/d_model": 1.0,  # 上界
        "1/sqrt(d)": 1.0 / np.sqrt(d_model),
        "0.22_target": 0.22,
    }
    
    print(f"\n  理论预测对比:")
    print(f"  实测ratio={rc_real['ratio']:.4f}, 随机ratio={mean_random_ratio:.4f}")
    for name, pred in predicted_ratios.items():
        match = "OK" if abs(pred - mean_random_ratio) < 0.05 else "NO"
        print(f"  {name}={pred:.4f} {match}")
    
    results["random_rotation_test"] = {
        "real_delta_ratio": rc_real["ratio"],
        "random_ratios_mean": mean_random_ratio,
        "random_ratios_std": std_random_ratio,
        "random_ratios_all": [float(x) for x in random_ratios],
        "predicted_ratios": predicted_ratios,
        "layer_used": mid_layer,
    }
    
    # Step 5: 不同范数delta的ratio (测试各向同性)
    print("  Step 5: 不同范数delta的ratio (各向同性测试)...")
    norm_ratios = {}
    for scale in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        delta_scaled = scale * direction
        rc = compute_recoding_ratio_cached(delta_scaled, U_wut)
        norm_ratios[scale] = rc["ratio"]
        print(f"    scale={scale}: ratio={rc['ratio']:.4f}")
    
    results["isotropy_test"] = norm_ratios
    
    return results


# ================================================================
# P444: 跨模型recoding_ratio同构验证
# ================================================================

def run_p444(model, tokenizer, device, model_info):
    """
    P444: 跨模型recoding_ratio同构验证
    
    原理:
      如果recoding_ratio是统计不变量, 那么在不同模型中:
      - ratio ≈ k_eff / d_model (k_eff取决于W_U的PR)
      - 不同模型的ratio可能不同(因为d_model和PR不同)
      - 但ratio应该与PR/d_model成正比
    
    方法:
      1. 在当前模型上测量多个属性、多层的recoding_ratio
      2. 同时记录recoding_gain曲线
      3. 记录W_U的PR, k_90等指标
      4. 三模型结果汇总后对比
    """
    print(f"\n{'='*60}")
    print(f"P444: 跨模型recoding_ratio同构 - {model_info.name}")
    print(f"{'='*60}")
    
    results = {}
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # 先获取所有属性的direction (只需要W_U的几行, 节省内存)
    W_U = get_W_U(model)
    attr_directions = {}
    for attr in TEST_ATTRS:
        direction, attr_tok_id = get_attr_direction(model, tokenizer, attr, W_U=W_U)
        if direction is not None:
            attr_directions[attr] = direction
    
    # W_U分析 — 对W_U^T做SVD更稳定
    W_U_T = W_U.T.astype(np.float32)  # [d_model, vocab]
    vocab_size = W_U.shape[0]
    del W_U  # 释放W_U, 只保留W_U_T用于SVD
    import gc; gc.collect()
    
    max_k = min(W_U_T.shape[0], W_U_T.shape[1]) - 2
    k_svd = min(300, max_k)
    U_wut_raw, s_wu_raw, _ = svds(W_U_T, k=k_svd)
    s_wu = np.abs(np.asarray(s_wu_raw, dtype=np.float64).ravel())
    sort_idx = np.argsort(-s_wu)
    s_wu = s_wu[sort_idx]
    U_wut_444 = np.asarray(U_wut_raw, dtype=np.float64)[:, sort_idx]  # [d_model, k] — W_U行空间基
    del W_U_T, U_wut_raw, s_wu_raw; gc.collect()
    
    total_energy = np.sum(s_wu ** 2)
    PR_wu = (np.sum(s_wu) ** 2) / max(total_energy, 1e-20)
    
    results["wu_summary"] = {
        "d_model": d_model,
        "vocab_size": vocab_size,
        "PR": float(PR_wu),
        "predicted_ratio_PR_over_d": PR_wu / d_model,
    }
    
    # 全层recoding_ratio和gain测量
    print("  测量全层recoding_ratio和gain...")
    sample_layers = get_sample_layers(n_layers, n_samples=10)
    
    # 多属性平均
    all_layer_ratios = {li: [] for li in sample_layers}
    all_layer_gains = {li: [] for li in sample_layers}
    all_layer_cos = {li: [] for li in sample_layers}
    
    for attr in TEST_ATTRS:
        if attr not in attr_directions:
            continue
        direction = attr_directions[attr]
        
        inputs_base, inputs_interv, _, pos_ids = inject_at_embed(
            model, tokenizer, device, PROMPT, direction, BETA
        )
        
        base_out = collect_layer_outputs(model, inputs_base, pos_ids, n_layers)
        interv_out = collect_layer_outputs(model, inputs_interv, pos_ids, n_layers)
        
        for li in sample_layers:
            key = f"L{li}"
            if key not in base_out or key not in interv_out:
                continue
            
            h_base = base_out[key][0, -1, :].numpy()
            h_interv = interv_out[key][0, -1, :].numpy()
            delta_h = h_interv - h_base
            
            # recoding_ratio (使用预计算的U_wut)
            rc = compute_recoding_ratio_cached(delta_h, U_wut_444)
            all_layer_ratios[li].append(rc["ratio"])
            all_layer_gains[li].append(rc["gain"])
            
            # cos with inject direction
            cos_val = compute_cos(delta_h, direction)
            all_layer_cos[li].append(cos_val)
    
    # 汇总
    layer_summary = []
    for li in sample_layers:
        if all_layer_ratios[li]:
            layer_summary.append({
                "layer": li,
                "ratio_mean": round(float(np.mean(all_layer_ratios[li])), 4),
                "ratio_std": round(float(np.std(all_layer_ratios[li])), 4),
                "gain_mean": round(float(np.mean(all_layer_gains[li])), 4),
                "cos_mean": round(float(np.mean(all_layer_cos[li])), 4),
            })
    
    results["layer_summary"] = layer_summary
    
    # 打印关键数据
    print(f"\n  {model_info.name} (d={d_model}, PR={PR_wu:.0f}):")
    print(f"  预测ratio = PR/d = {PR_wu/d_model:.4f}")
    for ls in layer_summary:
        print(f"    L{ls['layer']}: ratio={ls['ratio_mean']:.4f}±{ls['ratio_std']:.4f}, "
              f"gain={ls['gain_mean']:.4f}, cos={ls['cos_mean']:.4f}")
    
    # 权重矩阵的谱范数和PR (用于LCS)
    print("  计算权重矩阵谱分析...")
    weight_spectra = {}
    check_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    layers = get_layers(model)
    
    for li in check_layers:
        if li >= n_layers:
            continue
        lw = get_layer_weights(layers[li], d_model, model_info.mlp_type)
        
        # W_up谱范数
        if lw.W_up is not None:
            # 截断SVD求最大奇异值
            k_w = min(10, lw.W_up.shape[0] - 1, lw.W_up.shape[1] - 1)
            try:
                s_up = svds(lw.W_up.astype(np.float32), k=k_w, return_singular_vectors=False)
                w_up_spectral = float(np.max(np.abs(s_up)))
            except:
                w_up_spectral = float(np.linalg.norm(lw.W_up[:100, :], ord=2))
        else:
            w_up_spectral = 0
        
        # W_down谱范数
        if lw.W_down is not None:
            try:
                k_w = min(10, lw.W_down.shape[0] - 1, lw.W_down.shape[1] - 1)
                s_down = svds(lw.W_down.astype(np.float32), k=k_w, return_singular_vectors=False)
                w_down_spectral = float(np.max(np.abs(s_down)))
            except:
                w_down_spectral = float(np.linalg.norm(lw.W_down[:100, :], ord=2))
        else:
            w_down_spectral = 0
        
        # W_o谱范数
        W_o = lw.W_o
        if W_o.shape[1] != d_model:
            W_o = W_o.T
        try:
            k_w = min(10, W_o.shape[0] - 1, W_o.shape[1] - 1)
            s_o = svds(W_o.astype(np.float32), k=k_w, return_singular_vectors=False)
            w_o_spectral = float(np.max(np.abs(s_o)))
        except:
            w_o_spectral = 0
        
        weight_spectra[li] = {
            "W_up_spectral": round(w_up_spectral, 4),
            "W_down_spectral": round(w_down_spectral, 4),
            "W_o_spectral": round(w_o_spectral, 4),
        }
    
    results["weight_spectra"] = weight_spectra
    
    return results


# ================================================================
# P445: LCS预测力验证
# ================================================================

def run_p445(model, tokenizer, device, model_info):
    """
    P445: LCS(Layer Communication Specification)预测力验证
    
    原理:
      LCS定义: 每层的"通信能力"取决于权重矩阵的谱性质
      假设: LCS(layer) = ||W_up||_spectral × f(PR_W_U)
      
      如果LCS有效, 则:
      - LCS应该能预测recoding_gain的变化趋势
      - 不同层的LCS差异应该解释不同层的信号放大差异
    
    方法:
      1. 计算每层的LCS指标 (W_up谱范数, W_down谱范数, PR等)
      2. 计算每层的recoding_gain
      3. 分析LCS与recoding_gain的相关性
    """
    print(f"\n{'='*60}")
    print(f"P445: LCS预测力验证 - {model_info.name}")
    print(f"{'='*60}")
    
    results = {}
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    
    # 先获取direction (节省内存)
    W_U = get_W_U(model)
    direction, _ = get_attr_direction(model, tokenizer, "red", W_U=W_U)
    
    # W_U的PR — 对W_U^T做SVD (预计算用于cached ratio)
    W_U_T = W_U.T.astype(np.float32)
    del W_U; import gc; gc.collect()
    max_k = min(W_U_T.shape[0], W_U_T.shape[1]) - 2
    k_svd = min(300, max_k)
    U_wut_raw_445, s_wu_raw_445, _ = svds(W_U_T, k=k_svd)
    del W_U_T; gc.collect()
    s_wu = np.abs(np.asarray(s_wu_raw_445, dtype=np.float64).ravel())
    sort_idx_445 = np.argsort(-s_wu)
    s_wu = s_wu[sort_idx_445]
    U_wut_445 = np.asarray(U_wut_raw_445, dtype=np.float64)[:, sort_idx_445]  # [d_model, k]
    del U_wut_raw_445, s_wu_raw_445; gc.collect()
    PR_wu = float((np.sum(s_wu) ** 2) / max(np.sum(s_wu ** 2), 1e-20))
    
    # 1. 计算每层权重谱指标
    print("  计算每层权重谱指标...")
    layer_metrics = []
    
    sample_layers = get_sample_layers(n_layers, n_samples=10)
    
    for li in sample_layers:
        lw = get_layer_weights(layers[li], d_model, model_info.mlp_type)
        
        # 谱范数 (用Frobenius范数近似, 更快)
        metrics = {"layer": li}
        
        if lw.W_up is not None:
            metrics["W_up_frobenius"] = float(np.linalg.norm(lw.W_up, 'fro'))
            metrics["W_up_spectral_approx"] = float(np.linalg.norm(lw.W_up[:200, :], 2)) if lw.W_up.shape[0] > 200 else float(np.linalg.norm(lw.W_up, 2))
        if lw.W_down is not None:
            metrics["W_down_frobenius"] = float(np.linalg.norm(lw.W_down, 'fro'))
        if lw.W_gate is not None:
            metrics["W_gate_frobenius"] = float(np.linalg.norm(lw.W_gate, 'fro'))
        
        W_o = lw.W_o
        if W_o.shape[1] != d_model:
            W_o = W_o.T
        metrics["W_o_frobenius"] = float(np.linalg.norm(W_o, 'fro'))
        
        # LayerNorm γ的统计
        if lw.input_layernorm_weight is not None:
            metrics["LN_gamma_mean"] = float(np.mean(lw.input_layernorm_weight))
            metrics["LN_gamma_std"] = float(np.std(lw.input_layernorm_weight))
            metrics["LN_gamma_norm"] = float(np.linalg.norm(lw.input_layernorm_weight))
        
        layer_metrics.append(metrics)
    
    # 2. 计算每层recoding_gain
    print("  计算每层recoding_gain...")
    
    # direction已在前面获取
    if direction is None:
        return results
    
    inputs_base, inputs_interv, _, pos_ids = inject_at_embed(
        model, tokenizer, device, PROMPT, direction, BETA
    )
    
    base_out = collect_layer_outputs(model, inputs_base, pos_ids, n_layers)
    interv_out = collect_layer_outputs(model, inputs_interv, pos_ids, n_layers)
    
    gain_data = []
    for li in sample_layers:
        key = f"L{li}"
        if key not in base_out or key not in interv_out:
            continue
        
        h_base = base_out[key][0, -1, :].numpy()
        h_interv = interv_out[key][0, -1, :].numpy()
        delta_h = h_interv - h_base
        
        # recoding指标
        rc = compute_recoding_ratio_cached(delta_h, U_wut_445)
        
        gain_data.append({
            "layer": li,
            "recoding_gain": rc["gain"],
            "recoding_ratio": rc["ratio"],
            "delta_norm": float(np.linalg.norm(delta_h)),
        })
    
    # 3. LCS与recoding_gain的相关分析
    print("  分析LCS与recoding_gain的相关性...")
    
    # 合并数据
    combined = []
    for lm in layer_metrics:
        li = lm["layer"]
        gd = [g for g in gain_data if g["layer"] == li]
        if gd:
            entry = {**lm, **gd[0]}
            combined.append(entry)
    
    if len(combined) >= 3:
        # 简单线性相关
        gains = [c["recoding_gain"] for c in combined]
        
        # W_up与gain的相关
        if "W_up_frobenius" in combined[0]:
            wup_vals = [c["W_up_frobenius"] for c in combined]
            corr_wup = np.corrcoef(wup_vals, gains)[0, 1]
        else:
            corr_wup = 0
        
        # W_o与gain的相关
        wo_vals = [c["W_o_frobenius"] for c in combined]
        corr_wo = np.corrcoef(wo_vals, gains)[0, 1]
        
        # LN与gain的相关
        if "LN_gamma_norm" in combined[0]:
            ln_vals = [c["LN_gamma_norm"] for c in combined]
            corr_ln = np.corrcoef(ln_vals, gains)[0, 1]
        else:
            corr_ln = 0
        
        # delta_norm与gain的相关
        delta_vals = [c["delta_norm"] for c in combined]
        corr_delta = np.corrcoef(delta_vals, gains)[0, 1]
        
        print(f"\n  相关性分析 (n={len(combined)}层):")
        print(f"    W_up_frobenius vs gain: r={corr_wup:.4f}")
        print(f"    W_o_frobenius vs gain:  r={corr_wo:.4f}")
        print(f"    LN_gamma_norm vs gain:  r={corr_ln:.4f}")
        print(f"    delta_norm vs gain:     r={corr_delta:.4f}")
        
        results["correlations"] = {
            "W_up_vs_gain": float(corr_wup),
            "W_o_vs_gain": float(corr_wo),
            "LN_vs_gain": float(corr_ln),
            "delta_vs_gain": float(corr_delta),
        }
    
    results["layer_metrics"] = layer_metrics
    results["gain_data"] = gain_data
    results["combined"] = combined
    results["PR_wu"] = PR_wu
    
    return results


# ================================================================
# P446: 随机权重 vs 真实权重
# ================================================================

def run_p446(model, tokenizer, device, model_info):
    """
    P446: 随机权重 vs 真实权重 — recoding_ratio是训练结果还是几何约束?
    
    原理:
      如果recoding_ratio≈0.22是纯几何效应(由d_model和W_U维度决定):
      → 用随机正交权重替换MLP权重后, ratio应该不变
      
      如果ratio是训练产生的结构:
      → 随机权重替换后ratio会改变
      
    方法:
      1. 用真实权重做forward, 测量recoding_ratio
      2. 将L0的MLP权重替换为随机正交矩阵
      3. 用替换后的模型做forward, 测量recoding_ratio
      4. 对比: ratio是否改变
    
    注意: 替换权重后模型会生成乱码, 但我们只关心信号传播的统计性质
    """
    print(f"\n{'='*60}")
    print(f"P446: 随机权重vs真实权重 - {model_info.name}")
    print(f"{'='*60}")
    
    results = {}
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    
    # 先获取direction和W_U SVD
    W_U = get_W_U(model)
    direction, _ = get_attr_direction(model, tokenizer, "red", W_U=W_U)
    if direction is None:
        return results
    
    # 1. 真实权重的recoding_ratio
    print("  Step 1: 真实权重baseline...")
    inputs_base, inputs_interv, _, pos_ids = inject_at_embed(
        model, tokenizer, device, PROMPT, direction, BETA
    )
    
    base_out = collect_layer_outputs(model, inputs_base, pos_ids, n_layers)
    interv_out = collect_layer_outputs(model, inputs_interv, pos_ids, n_layers)

    # 预计算W_U^T的SVD (用于cached ratio)
    W_U_T_446 = W_U.T.astype(np.float32)
    del W_U; import gc; gc.collect()
    max_k_446 = min(W_U_T_446.shape[0], W_U_T_446.shape[1]) - 2
    k_svd_446 = min(200, max_k_446)
    U_wut_raw_446, s_wut_raw_446, _ = svds(W_U_T_446, k=k_svd_446)
    del W_U_T_446; gc.collect()
    s_wut_446 = np.abs(np.asarray(s_wut_raw_446, dtype=np.float64).ravel())
    sort_idx_446 = np.argsort(-s_wut_446)
    U_wut_446 = np.asarray(U_wut_raw_446, dtype=np.float64)[:, sort_idx_446]  # [d_model, k]
    del U_wut_raw_446, s_wut_raw_446; gc.collect()
    
    real_ratios = {}
    for li in [0, 1, 2, n_layers // 2, n_layers - 1]:
        if li >= n_layers:
            continue
        key = f"L{li}"
        if key not in base_out or key not in interv_out:
            continue
        h_base = base_out[key][0, -1, :].numpy()
        h_interv = interv_out[key][0, -1, :].numpy()
        delta_h = h_interv - h_base
        rc = compute_recoding_ratio_cached(delta_h, U_wut_446)
        real_ratios[li] = rc["ratio"]
        print(f"    L{li}: ratio={rc['ratio']:.4f}")
    
    results["real_weight_ratios"] = real_ratios
    
    # 2. 替换L0的MLP权重为随机正交矩阵
    print("  Step 2: 替换L0 MLP权重为随机矩阵...")
    
    layer0 = layers[0]
    mlp = layer0.mlp
    
    # 保存原始权重
    orig_weights = {}
    for name in ["up_proj", "down_proj", "gate_proj", "gate_up_proj"]:
        if hasattr(mlp, name):
            orig_weights[name] = getattr(mlp, name).weight.data.clone()
    
    # 生成随机正交权重并替换
    np.random.seed(42)
    random_weight_shapes = {}
    
    for name, orig_w in orig_weights.items():
        shape = orig_w.shape
        random_weight_shapes[name] = list(shape)
        
        # 生成随机正交矩阵 (保持shape)
        if shape[0] <= shape[1]:
            Q = np.random.randn(shape[0], shape[1]).astype(np.float32)
            Q, _ = np.linalg.qr(Q.T)
            Q = Q.T[:shape[0], :]
        else:
            Q = np.random.randn(shape[0], shape[1]).astype(np.float32)
            Q, _ = np.linalg.qr(Q)
            Q = Q[:, :shape[1]]
        
        # 缩放到与原始权重相同的范数
        orig_norm = orig_w.norm().item()
        Q = Q * (orig_norm / np.linalg.norm(Q))
        
        # 替换
        getattr(mlp, name).weight.data = torch.tensor(Q, dtype=orig_w.dtype, device=orig_w.device)
        print(f"    {name}: shape={shape}, orig_norm={orig_norm:.2f}")
    
    # 3. 随机权重forward
    print("  Step 3: 随机权重forward...")
    
    try:
        base_out_rand = collect_layer_outputs(model, inputs_base, pos_ids, n_layers)
        interv_out_rand = collect_layer_outputs(model, inputs_interv, pos_ids, n_layers)
        
        random_ratios = {}
        for li in [0, 1, 2, n_layers // 2, n_layers - 1]:
            if li >= n_layers:
                continue
            key = f"L{li}"
            if key not in base_out_rand or key not in interv_out_rand:
                continue
            h_base = base_out_rand[key][0, -1, :].numpy()
            h_interv = interv_out_rand[key][0, -1, :].numpy()
            delta_h = h_interv - h_base
            
            # 检查NaN
            if np.any(np.isnan(delta_h)) or np.any(np.isinf(delta_h)):
                random_ratios[li] = "NaN"
                print(f"    L{li}: NaN detected!")
                continue
            
            rc = compute_recoding_ratio_cached(delta_h, U_wut_446)
            random_ratios[li] = rc["ratio"]
            print(f"    L{li}: ratio={rc['ratio']:.4f}")
        
        results["random_weight_ratios"] = random_ratios
        
        # 对比
        print(f"\n  *** 真实 vs 随机权重 recoding_ratio ***")
        for li in real_ratios:
            real_r = real_ratios[li]
            rand_r = random_ratios.get(li, "N/A")
            if isinstance(rand_r, float):
                diff = abs(real_r - rand_r)
                print(f"    L{li}: real={real_r:.4f}, random={rand_r:.4f}, diff={diff:.4f}")
            else:
                print(f"    L{li}: real={real_r:.4f}, random={rand_r}")
    
    except Exception as e:
        print(f"  Random weight forward failed: {e}")
        results["random_weight_error"] = str(e)
    
    finally:
        # 恢复原始权重 (非常重要!)
        print("  恢复原始权重...")
        for name, orig_w in orig_weights.items():
            getattr(mlp, name).weight.data = orig_w
    
    results["random_weight_shapes"] = random_weight_shapes
    
    # 4. 只替换W_up的测试 (W_up是关键因子)
    print("  Step 4: 仅替换W_up...")
    
    if "up_proj" in orig_weights or "gate_up_proj" in orig_weights:
        # 保存
        if "up_proj" in orig_weights:
            orig_up = mlp.up_proj.weight.data.clone()
            shape = orig_up.shape
        else:
            # GLM4: 不能只替换up, gate_up_proj是合并的
            print("    GLM4 merged gate_up_proj, skipping W_up-only test")
            results["wup_only_ratios"] = "skipped_GLM4"
            return results
        
        # 随机W_up
        np.random.seed(123)
        Q_up = np.random.randn(shape[0], shape[1]).astype(np.float32)
        Q_up, _ = np.linalg.qr(Q_up.T)
        Q_up = Q_up.T[:shape[0], :]
        orig_norm = orig_up.norm().item()
        Q_up = Q_up * (orig_norm / np.linalg.norm(Q_up))
        mlp.up_proj.weight.data = torch.tensor(Q_up, dtype=orig_up.dtype, device=orig_up.device)
        
        try:
            base_out_wup = collect_layer_outputs(model, inputs_base, pos_ids, n_layers)
            interv_out_wup = collect_layer_outputs(model, inputs_interv, pos_ids, n_layers)
            
            wup_ratios = {}
            for li in [0, 1, 2]:
                key = f"L{li}"
                if key in base_out_wup and key in interv_out_wup:
                    h_base = base_out_wup[key][0, -1, :].numpy()
                    h_interv = interv_out_wup[key][0, -1, :].numpy()
                    delta_h = h_interv - h_base
                    if not np.any(np.isnan(delta_h)):
                        rc = compute_recoding_ratio_cached(delta_h, U_wut_446)
                        wup_ratios[li] = rc["ratio"]
                        print(f"    L{li}: ratio={rc['ratio']:.4f}")
            
            results["wup_only_ratios"] = wup_ratios
        
        except Exception as e:
            print(f"  W_up-only forward failed: {e}")
            results["wup_only_error"] = str(e)
        
        finally:
            # 恢复
            mlp.up_proj.weight.data = orig_up
    
    return results


# ================================================================
# 主函数
# ================================================================

def run_single_model(model_name, experiments=None):
    """运行单个模型的所有实验"""
    print(f"\n{'#'*60}")
    print(f"# 模型: {model_name}")
    print(f"{'#'*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"  类: {model_info.model_class}")
    print(f"  层数: {model_info.n_layers}, 维度: {model_info.d_model}")
    print(f"  MLP类型: {model_info.mlp_type}")
    
    all_results = {
        "model": model_name,
        "model_class": model_info.model_class,
        "n_layers": model_info.n_layers,
        "d_model": model_info.d_model,
        "mlp_type": model_info.mlp_type,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M")
    
    exp_map = {
        "p443": run_p443,
        "p444": run_p444,
        "p445": run_p445,
        "p446": run_p446,
    }
    
    try:
        if experiments is None or "all" in experiments:
            exps_to_run = list(exp_map.items())
        else:
            exps_to_run = [(e, exp_map[e]) for e in experiments if e in exp_map]
        
        for exp_name, exp_fn in exps_to_run:
            print(f"\n--- Running {exp_name} ---")
            result = exp_fn(model, tokenizer, device, model_info)
            all_results[exp_name] = result
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    
    finally:
        out_file = OUT_DIR / f"phase_xci_p443_446_{model_name}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResults saved to {out_file}")
        
        release_model(model)


def main():
    parser = argparse.ArgumentParser(description=f"{PHASE_NAME}: {EXPERIMENT_DESC}")
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, default="all",
                       choices=["all", "p443", "p444", "p445", "p446"])
    args = parser.parse_args()
    
    experiments = [args.experiment] if args.experiment != "all" else None
    run_single_model(args.model, experiments)


if __name__ == "__main__":
    main()
